# Qwen3.5 Linear Decoder Layer CUDA Graph Plan

## 当前结论

目标不是让整层从“图外”变成“图内”。从当前 prefill 日志看，外层 piecewise CUDA graph 已经在工作：

- 模型：`Qwen3_5MoeForConditionalGeneration`
- 设备：CUDA
- full attention backend：`fa3`
- linear attention backend：`triton`
- `enable_two_batch_overlap=False`
- `enable_breakable_cuda_graph=False`
- `disable_piecewise_cuda_graph=False`
- `enforce_piecewise_cuda_graph=True`
- `piecewise_cuda_graph_compiler='eager'`
- 日志中多数 prefill 为 `cuda graph: True`

所以当前缺口不是 `Qwen3_5LinearDecoderLayer` 完全没有被 PCG 捕获，而是 `RadixLinearAttention.forward` 中的 `unified_linear_attention_with_output` 是一个 split op。PCG 会捕获它前后的子图，但这个 op 自身仍按 eager 方式执行。

当前只考虑这条 CUDA/Qwen3.5/GDN 路径，不考虑 Ascend、LightningAttentionBackend 和 TBO 包装场景。

## 当前调用链

```text
Qwen3_5LinearDecoderLayer.forward
  -> Qwen3_5GatedDeltaNet.forward
    -> RadixLinearAttention.forward
      -> unified_linear_attention_with_output
        -> forward_batch.attn_backend.forward
          -> HybridLinearAttnBackend.forward_extend
            -> GDNAttnBackend.forward_extend
```

当前 backend 关系：

```text
ModelRunner.attn_backend
  = HybridLinearAttnBackend(
      full_attn_backend = FA3 backend,
      linear_attn_backend = GDNAttnBackend,
      full_attn_layers = qwen3_5 full attention layer ids,
    )

GDNAttnBackend -> MambaAttnBackendBase
Qwen3_5GatedDeltaNet -> RadixLinearAttention
```

## 为什么现在 `unified_linear_attention_with_output` 不在 PCG 子图里

`unified_linear_attention_with_output` 同时带有两个装饰器：

```python
@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_linear_attention_with_output(...):
    ...
```

`register_custom_op` 让它成为 `torch.ops.sglang.unified_linear_attention_with_output` 这种 opaque op。`register_split_op` 又把这个 opaque op 加入 `CompilationConfig.split_ops`。PCG backend 在 FX graph 中遇到 split op 时，会把它单独切成 splitting graph，只编译和捕获非 splitting graph。

因此：

- `unified_linear_attention_with_output` 前后的 projection/norm/MLP 等计算可以进入 PCG。
- `unified_linear_attention_with_output` 自身不会进入这些 PCG 子图。
- 它内部的 `forward_batch.attn_backend.forward(...)`、GDN Triton kernels、mamba cache 更新仍在 eager 调用路径上执行。

之前加的 `eager_on_graph(True)` 包装只对 Breakable CUDA Graph 有效。当前日志里 `enable_breakable_cuda_graph=False`，所以这个包装不会让当前 piecewise CUDA graph 路径多捕获任何东西。

## 推荐路线

当前场景下不建议直接删除 `@register_split_op()`。

直接删除 split op 的问题是：这个 op 内部依赖 `ForwardBatch`、全局 forward context、attention backend dispatch、mamba cache state 和动态 metadata。如果让它直接落入外层 PCG 子图，CUDA graph 会捕获 capture 时的 metadata tensor 地址；replay 时如果这些 metadata 每次重新分配，graph 看到的仍是旧地址，结果会错。

推荐第一版保留 split op，让外层 PCG 继续稳定捕获 layer 前后计算；然后在 `unified_linear_attention_with_output` 内部为当前 Qwen3.5 GDN normal prefill 路径增加一个专用 CUDA graph runner。也就是：

```text
outer PCG graph piece
  -> split op: unified_linear_attention_with_output
       -> if eligible: replay/capture GDN prefill inner CUDA graph
       -> else: current eager path
  -> outer PCG graph piece
```

这样改动面最小，并且不会牵连 full attention、Lightning、TBO 或 Ascend。

## 需要的适配

### 1. 增加当前场景的 eligibility guard

只对当前日志对应路径启用 inner graph：

- 只在外层 piecewise CUDA graph 运行态尝试；普通 eager 路径和 PCG torch.compile 阶段都直接走现有 eager split op。
- device 是 CUDA。
- `forward_batch.forward_mode` 是 normal extend；如果上层把 `MIXED` 规范化成 `EXTEND`，也只处理其中的 prefill/extend 部分。
- `spec_info is None`，不处理 target verify / speculative path。
- `forward_batch.attn_backend` 是 `HybridLinearAttnBackend`。
- `linear_attn_backend` 是 `GDNAttnBackend`。
- GDN kernel dispatcher 的 decode/extend backend 是当前已验证的 Triton 路径。
- 当前不经过 `TboAttnBackend`。
- token 数在可捕获范围内；日志里 `#new-token: 5888` 的 prefill 已经是 `cuda graph: False`，第一版不需要覆盖。

不满足条件时继续走当前 eager split op。

### 2. 引入 GDN prefill inner CUDA graph runner

建议新增一个只服务当前场景的 runner，例如放在 `gdn_backend.py` 或一个相邻模块中：

```text
Qwen35GDNPrefillCudaGraphRunner
  key: (layer_id, static_num_tokens, static_bs, dtype, device)
  capture: 执行一次 GDNAttnBackend.forward_extend 并捕获 kernels
  replay: 更新稳定输入/metadata buffer，然后 replay graph
```

这个 inner runner 是按 bucket 懒捕获，但只允许在外层 PCG capture 过程里创建或重捕获 graph；普通 replay 阶段不临时建图。`torch.cuda.graph(...)` 的职责是记录当前 bucket 的 kernel 提交流程；为了让首次 capture 调用也得到实际输出和 mamba state 更新，capture 成功后必须立即 replay 刚捕获的 graph 一次。这个 replay 是当前调用的实际执行，不是额外重复执行；后续同 bucket 命中时才复用同一个 graph 做普通 replay。

capture 时优先复用外层 PCG 已初始化的 global graph pool；如果当前进程还没有 global graph pool，再退回 PyTorch 默认 graph pool。这样可以避免每个 layer/bucket 的 inner graph 都创建独立私有 pool，和现有 CUDA graph runner 的内存管理方式保持一致。

inner runner 必须区分外层 PCG 的不同阶段。`unified_linear_attention_with_output` 是 `register_split_op`，会被拆成 splitting graph；非 splitting 子图由 `CUDAPiecewiseBackend` 捕获，split op 本身仍在 PCG capture/replay 过程中执行 Python。因此 inner runner 可以在 split op 内更新 graph-owned metadata，并 replay 自己的 inner graph。当前实现用 `get_pcg_capture_stream() is not None` 作为 `allow_capture` 条件，并额外用 `torch.cuda.is_current_stream_capturing()` 防止未来意外嵌套 stream capture：

- dummy warmup / torch.compile 阶段：不创建 inner graph，不 recapture，直接 fallback 到 eager split op。
- 外层 PCG capture 过程中的 split op 调用：capture stream 已设置，允许首次 capture inner graph。如果同一 bucket 输入地址从启动临时地址切到稳定地址，也只允许一次 recapture。
- 如果未来 split op 意外在已经处于 `torch.cuda.graph(...)` 的 stream capture 中执行：不允许新建或重捕获 inner graph；只允许 replay 已经存在且输入签名一致的 inner graph。
- 外层实际 replay 阶段：只允许复用已存在且输入签名一致的 inner graph；如果 entry 不存在、graph 不存在或地址签名不一致，直接 fallback，不在 replay 阶段临时 capture/recapture。

这样既保留 split op replay 时更新真实 metadata 的能力，也避免在已经处于 stream capture 时嵌套 `torch.cuda.graph(...)`，或在真实 replay 中因为地址变化临时重捕获，破坏 graph 复用语义。

这里必须使用 `static_num_tokens` 做图 key，不能使用 `real_num_tokens`。原因是 graph 模式的价值就在于复用同一个 bucket graph；如果按 `real_num_tokens` 建图，就退化成 exact-shape graph，和外层 PCG 的 bucket 化模式不一致，也无法解决 padded replay 的核心问题。

`real_num_tokens` 只能作为 replay 时写入固定 metadata/buffer 的运行时内容，不能作为 graph key。inner graph 的 tensor shape 必须按 `static_num_tokens` 固定；真实 token 数小于 bucket 时，padding 区域要在静态图里被安全建模，而不是通过 `mixed_qkv[:real_num_tokens]` 改变 kernel 输入 shape。

当前第一版只 capture `static_bs=1`，即一个真实 normal extend request。`static_num_tokens` 不再使用任意大小，而是限制在 1 到 2k 之间、64 粒度的 bucket：

```text
static_num_tokens in {64, 128, 192, ..., 2048}
static_num_tokens = ceil_align(real_num_tokens, 64)
```

因此当前方案要求外层 PCG 的 capture token 列表也按同一组 64 粒度 bucket 配置，例如显式设置：

```bash
--piecewise-cuda-graph-tokens '[64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024,1088,1152,1216,1280,1344,1408,1472,1536,1600,1664,1728,1792,1856,1920,1984,2048]'
```

如果继续使用默认 PCG token schedule，64 到 512 区间会包含 `80/96/112/...` 这类 16 粒度 bucket。遇到这种 static shape 时，inner graph 会按 `static_num_tokens_not_bucket` fallback 到 eager，保证正确性，但不会命中当前 Qwen3.5 GDN inner graph。

例如 `real_num_tokens=1984` 时命中 `static_num_tokens=1984`，`real_num_tokens=1985` 时命中 `static_num_tokens=2048`。这里仍然不是按任意 `real_num_tokens` 建图，而是把图数量收敛到最多 32 个 64-token bucket。

这也带来一个必须解决的问题：现有 eager 路径会在 `unified_linear_attention_with_output` 内部对 `mixed_qkv/a/b` 做 `[:real_num_tokens]`，从而让 GDN backend 完全看不到 padded token。inner CUDA graph 不能这么做，因为 shape 必须保持 `static_num_tokens`。

当前场景收敛为 normal extend、`static_bs=1`。这里不再引入 padding request；`[real_num_tokens, static_num_tokens)` 只是一段 token padding tail，不能被建模成第二个真实 sequence：

- `mixed_qkv/a/b/output` 的 tensor shape 固定为 `static_num_tokens`。
- `query_start_loc` replay 为 `[0, real_num_tokens]`。
- `mamba_cache_indices` 只包含真实 request 的 slot。
- padding tail 只存在于静态 tensor shape 中，不属于任何有效 sequence。
- `output` 只把 `[:real_num_tokens]` 返回给后续逻辑，padding 区域输出被忽略。

如果现有 GDN/conv/FLA kernels 不能在 `static_num_tokens` launch 下用 `query_start_loc` mask 掉 padding tail，就需要先改 kernel wrapper 或新增 graph-friendly path；不能改回 `real_num_tokens` key，也不能把 tail 建成长度为 `static_num_tokens - real_num_tokens` 的真实 request。

### 3. 稳定 graph 输入地址

inner CUDA graph replay 要求输入 tensor 地址稳定。这里有两种实现方式：

1. 复用外层 PCG 产生的 `mixed_qkv/a/b/output` 地址，并在 debug 模式检查地址是否和 capture 时一致。
2. 给 inner runner 预分配自己的 `mixed_qkv/a/b/output` buffer，replay 前 copy 真实输入进去，replay 后把结果 copy 回 `unified_linear_attention_with_output` 的 `output`。

第一版更推荐方案 1，因为当前外层 PCG 的相邻子图输出地址理论上对同一个 layer/token bucket 是稳定的，额外 copy 少。必须加地址检查；如果首次懒捕获使用了启动阶段的临时地址，可以允许一次 recapture；如果同一 bucket 之后仍然地址不稳定，应 fallback 到 eager 或切到方案 2。

需要检查的地址至少包括：

- `mixed_qkv[:static_num_tokens]`
- `a[:static_num_tokens]`
- `b[:static_num_tokens]`
- `output[:, :static_num_tokens]`
- GDN `conv_weights` / `bias` / `A_log` / `dt_bias`
- mamba `conv_states` / `ssm_states` pool tensor
- graph-owned `query_start_loc_buf`
- graph-owned `mamba_cache_indices_buf`
- fixed-length `seq_lens_cpu` host metadata；保持 `[real_num_tokens]` 语义

`real_num_tokens` 只用于 replay 前填充 metadata 和 replay 后裁剪有效输出；不能用于改变 graph 输入 tensor shape。

### 4. 稳定 GDN metadata 地址

这是最关键的适配。

当前 `MambaAttnBackendBase._forward_metadata(forward_batch)` 在 normal extend 中会根据真实 batch 新建：

- `query_start_loc`
- `mamba_cache_indices`
- prefix/mamba tracking 相关索引

这在 eager 下没问题，但 CUDA graph replay 不能依赖每次新建的 tensor。inner graph 需要 graph-owned metadata buffer：

- `query_start_loc_buf`: shape `[static_bs + 1]`, `int32`, CUDA
- `mamba_cache_indices_buf`: shape `[static_bs]`, CUDA
- `seq_lens_cpu`: fixed-length host metadata，值为 `[real_num_tokens]`
- 可选 `mamba_track_mask_buf`: shape `[static_bs]`
- 可选 `mamba_track_indices_buf`: shape `[static_bs]`
- 可选 `mamba_track_seqlens_buf`: shape `[static_bs]`
- 可选 tracking 派生索引 buffer，见下一节

replay 前把真实 batch metadata copy 到这些固定 buffer。当前场景不再做泛化的 `raw_bs + 1` 适配，直接固定为 `static_bs=1`：

```text
query_start_loc_buf = [0, real_num_tokens]

mamba_cache_indices_buf = [real_slot]

extend_prefix_lens_buf = [real_prefix_len]
```

`seq_lens_cpu` 需要单独看待。它的原语义是每个 request 的真实 extend 长度，不能改成 bucket launch 描述。当前场景下应保持：

```text
seq_lens_cpu = [real_num_tokens]
```

当前 causal conv Triton wrapper 只在 Python launch grid 里使用 `len(seq_lens_cpu)` 和 `max(seq_lens_cpu)`，kernel 内的真实 sequence 边界来自 `query_start_loc`。因此如果直接保留原实现，`max(seq_lens_cpu)` 仍会把真实长度带进 launch grid。正确做法是新增一个 graph-only 的静态 launch 上限，例如 `grid_max_seq_len=static_num_tokens`，让 grid 固定由这个参数决定，而不是改变 `seq_lens_cpu` 的含义。

capture 和 replay 都要让 `GDNAttnBackend.forward_extend` 看到同一组 buffer 地址。实现方式可以是：

- 给 inner runner 构造一个 `ForwardMetadata`，里面引用这些稳定 buffer。
- capture/replay 前临时把 `gdn_backend.forward_metadata` 指向这个 graph metadata。
- eager fallback 时仍使用原来的 `init_forward_metadata(forward_batch)` 路径。

### 5. 处理 `seq_lens_cpu` 这类 CPU launch 参数

`GDNAttnBackend.forward_extend` 的 causal conv 路径会调用：

```python
causal_conv1d_fn(
    ...,
    cache_indices=cache_indices,
    query_start_loc=query_start_loc,
    seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
)
```

当前 CUDA/GDN 路径实际会走 `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn`。由于 `GDNAttnBackend.forward_extend` 里先做了 `mixed_qkv.transpose(0, 1)`，传入 causal conv 的 `x.stride(-1) != 1`，并且调用参数里带有 `seq_lens_cpu`，所以 wrapper 会进入 Triton fallback：

```python
use_triton = not _HAS_SGL_KERNEL or (
    x.stride(-1) != 1 and "seq_lens_cpu" in kwargs
)
```

在 `causal_conv1d_triton.py` 中，`seq_lens_cpu` 只用于 Python 侧 launch grid：

```python
def grid(META):
    max_seq_len = max(seq_lens_cpu)
    return (
        len(seq_lens_cpu),
        (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
        triton.cdiv(dim, META["BLOCK_N"]),
    )
```

Triton kernel 内部不再读取 `seq_lens_cpu`，真实 sequence 边界来自 `query_start_loc`：

```python
sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
seqlen = sequence_end_index - sequence_start_index
segment_len = min(BLOCK_M, seqlen - token_offset)
if segment_len <= 0:
    return
```

因此 causal conv 这一步可以规避 `seq_lens_cpu` 变化带来的 grid 变化，但前提是不要再让 `max(seq_lens_cpu)` 决定 graph launch grid。当前场景必须保留 `seq_lens_cpu` 原语义，并额外传入 graph-only 的静态 grid 上限：

```text
static_num_tokens = ceil_align(real_num_tokens, 64)  # 64..2048
static_bs = 1

capture/replay:
seq_lens_cpu = [real_num_tokens]     # 真实 request 长度语义
grid_max_seq_len = static_num_tokens # 只用于固定 launch grid
query_start_loc replay = [0, real_num_tokens]
mamba_cache_indices replay = [real_slot]
```

这里 `seq_lens_cpu` 仍表示真实 request 长度：

- 唯一 request 长度是 `real_num_tokens`。
- `len(seq_lens_cpu)=1` 固定 batch 维。
- `grid_max_seq_len=static_num_tokens` 固定 token grid 覆盖范围。
- 对超过 `real_num_tokens` 的 program，kernel 通过 `query_start_loc` 算出 `segment_len <= 0` 后 no-op。

不要把 `seq_lens_cpu` replay 成 `[real_num_tokens, static_num_tokens - real_num_tokens]`，因为那会把 padding tail 表示成真实 request。

`causal_conv1d_fn` / `causal_conv1d_triton.py` 需要增加这个 graph-only 参数，例如 `grid_max_seq_len` 或直接传 `static_num_tokens`：

```python
grid_max_seq_len = kwargs.pop("grid_max_seq_len", None)

def grid(META):
    max_seq_len = grid_max_seq_len if grid_max_seq_len is not None else max(seq_lens_cpu)
    return (
        len(seq_lens_cpu),
        (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
        triton.cdiv(dim, META["BLOCK_N"]),
    )
```

另一个可选方案是给 inner graph 预分配 contiguous 的 `[dim, static_num_tokens]` causal-conv 输入 buffer，使 `x.stride(-1) == 1`，从而走 `sgl_kernel.causal_conv1d_fwd` 路径；这个路径不需要 `seq_lens_cpu`。代价是 replay 前多一次 layout/copy。第一版更建议先保留当前 Triton fallback，并增加 `grid_max_seq_len`，因为改动面更小且不改变 `seq_lens_cpu` 语义。

`_causal_conv1d_fwd_kernel` 里的 stride 参数都是内存布局描述，基本都是 `tl.constexpr`，会参与 Triton specialization，也会被 CUDA graph capture 固化。它们不会随 `seq_lens_cpu` 变化，但要求 capture/replay 使用同样的 tensor layout：

- `stride_x_dim` / `stride_x_token`: `x` 中 feature 维和 token 维的步长。当前 GDN 输入来自 `mixed_qkv.transpose(0, 1)`，通常是 `[dim, static_num_tokens]` 视图，`stride_x_dim=1`、`stride_x_token=qkv_dim`。只要外层 PCG bucket 产生的 `mixed_qkv` layout 稳定，这两个 stride 就稳定。
- `stride_w_dim` / `stride_w_width`: conv weight 的布局，来自模型权重，固定。
- `stride_istate_seq` / `stride_istate_dim` / `stride_istate_token`: `conv_states` mamba cache 的布局，来自 memory pool，固定。
- `stride_o_dim` / `stride_o_token`: `out=torch.empty_like(x)` 的输出布局，通常跟 `x` 一致；如果改成 graph-owned output buffer，也必须保持同样 layout。
- `stride_x_seq` / `stride_o_seq`: 3D layout 下的 batch/sequence 维步长；当前 2D varlen path 中设置为 0，实际索引主要靠 `query_start_loc + stride_x_token`。

所以 stride 对 capture 的影响是：它们本身不是动态障碍，但不能让 replay 时的 `x/out/conv_states/weight` layout 跟 capture 不同。第一版 inner graph 应加 debug assert，检查这些 stride 和 capture 时一致。

需要注意：这只解决 causal conv 的 `seq_lens_cpu` grid 问题。后面的 `TritonGDNKernel.extend -> chunk_gated_delta_rule` 会基于 `cu_seqlens/query_start_loc` 在 Python 侧生成 `chunk_indices` 和 `chunk_offsets`，并用 `len(chunk_indices)` 决定部分 kernel grid。这个路径通过 64 粒度 bucket 稳定化：

- capture 时先把 graph-owned `query_start_loc_buf` 填成 `[0, static_num_tokens]`。
- 使用这个 static metadata 生成固定的 `chunk_indices/chunk_offsets`。
- `static_num_tokens` 必须是 64 的倍数，因此 `len(chunk_indices) = static_num_tokens / 64`。
- replay 时只更新同一个 `query_start_loc_buf` 的内容为 `[0, real_num_tokens]`。
- 因为 `static_num_tokens = ceil_align(real_num_tokens, 64)`，所以 static FLA chunk 数等于 `ceil(real_num_tokens / 64)`，不会额外多出完整 FLA chunk；最后一个 chunk 只可能是部分有效 token。

FLA 的 `prepare_chunk_indices/prepare_chunk_offsets` 带有 `@tensor_cache`，cache key 是 tensor 对象 identity，不是 tensor 内容。因此 graph-owned `query_start_loc_buf` 第一次进入 FLA 时必须已经是当前 bucket 的 static 内容；如果第一次按 `[0, real_num_tokens]` 生成 exact chunk metadata，capture 得到的 graph 就会固化 exact chunk metadata，破坏 bucket graph 复用。

inner runner 必须在 capture/recapture 前用 `[0, static_num_tokens]` prime 当前 entry 的 `chunk_indices/chunk_offsets`，再把同一 `query_start_loc_buf` 写回 `[0, real_num_tokens]` 进入 warmup/capture。capture 成功后，CUDA graph 的 kernel 参数已经保存了这些 static metadata tensor 的地址，所以 runner 必须用 `entry.static_chunk_refs` 持有它们，直到 graph 被 clear 或 recapture。replay 阶段不能重新生成并覆盖 `static_chunk_refs`，否则旧 metadata tensor 可能被释放，而已捕获 graph 仍然引用旧地址。

所以第一版 eligibility 可以收敛为：

- 只支持当前 normal extend。
- 固定 `static_bs=1`：只包含一个真实 request，不引入 padding request。
- `static_num_tokens` 只取 64 粒度 bucket，范围是 `[64, 2048]`。
- `seq_lens_cpu=[real_num_tokens]` 保持真实长度语义。
- `grid_max_seq_len=static_num_tokens` 固定 causal conv launch grid。
- 对 `real_num_tokens=1984` 这类已经 64 对齐的长度，映射到 `static_num_tokens=1984`；对 `real_num_tokens=1985` 这类非对齐长度，映射到 `static_num_tokens=2048`。padding tail 不属于任何有效 sequence，只作为静态输入 tail 被后续裁掉/忽略。

如果 causal conv 无法在 `static_num_tokens` launch 下用 metadata mask 掉 padding，那么当前 kernel 路径暂时不能入 inner graph，只能 fallback；不能退回 exact `real_num_tokens` graph。

### 5.1 其他 CPU/Python 侧 capture 风险

除 `seq_lens_cpu` 外，`unified_linear_attention_with_output -> GDNAttnBackend.forward_extend` 这条路径里还有几类会影响 capture/replay 的 CPU/Python 侧变量。

#### `real_num_tokens`

来源：

```python
real_num_tokens = forward_batch.num_token_non_padded_cpu
```

当前 eager 路径用它做 slicing：

```python
mixed_qkv[:real_num_tokens]
a[:real_num_tokens]
b[:real_num_tokens]
output[:, :real_num_tokens]
```

inner graph 不能让这些 slicing 改变 kernel 输入 shape。实现时必须把 GDN core 的输入 shape 固定为 `static_num_tokens`，`real_num_tokens` 只用于：

- replay 前写 `query_start_loc`。
- replay 后决定有效输出长度。
- 校验 padding tail 是否没有污染 mamba cache。

#### `out_cache_loc` / `out_cache_loc_swa`

`unified_linear_attention_with_output` 还会临时改写：

```python
forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]
forward_batch.out_cache_loc_swa = original_out_cache_loc_swa[:real_num_tokens]
token_to_kv_pool.set_swa_loc(...)
```

这同样是 Python 侧 mutation。当前 GDN CUDA normal extend 主路径主要使用 `query_start_loc`、`mamba_cache_indices` 和 mamba state pool，不依赖 `out_cache_loc` 写 mamba state；但 inner graph 不能捕获一个依赖临时 view 的隐式状态。第一版 graph path 应跳过这组 narrow mutation，或把它保持在图外且只作用于 eager fallback；如果后续某个 linear backend 需要这些 loc，则必须改成 graph-owned 固定地址 buffer，而不是 `[:real_num_tokens]` view。

#### `has_mamba_track_mask`

`MambaAttnBackendBase._forward_metadata` 会把：

```python
forward_batch.mamba_track_mask is not None
and forward_batch.mamba_track_mask.any()
```

转成 Python bool `forward_metadata.has_mamba_track_mask`。`GDNAttnBackend.forward_extend` 随后用它决定是否执行：

```python
mixed_qkv_to_track = mixed_qkv[:, forward_metadata.track_conv_indices]
conv_states[forward_metadata.conv_states_mask_indices] = mixed_qkv_to_track
```

这是一个真实的 Python 分支，而且 `track_conv_indices / conv_states_mask_indices` 是动态索引。第一版应当：

- `has_mamba_track_mask=False` 才允许 inner graph。
- 如果要覆盖 cached-token tracking，则必须把 tracking indices/counts 做成 graph-owned 固定 buffer，并让分支形态固定。

#### `extend_prefix_lens`

normal extend 分支里：

```python
has_initial_states = forward_batch.extend_prefix_lens > 0
```

这个值作为 device tensor 传给 causal conv。它本身不应该改变 launch grid，但 tensor 地址和 shape 必须稳定。当前 `static_bs=1` 下应使用 graph-owned `[static_bs]` tensor，replay 前写成 `[real_prefix_len]`，而不是每次让 `ForwardBatch` 挂一块新 tensor。

它的 CPU 侧风险主要来自 mamba tracking：`_init_track_ssm_indices` 会把 `extend_prefix_lens`、`extend_seq_lens`、`mamba_track_*` 做 `.cpu()` 并生成变长索引。第一版如果 `has_mamba_track_mask=False`，这部分不会进入 inner graph；如果要支持 tracking，则这些派生索引也必须固定 buffer 化。

#### FLA `chunk_indices` / `chunk_offsets`

这是除 causal conv 外最重要的 CPU/Python 动态点。`TritonGDNKernel.extend` 会调用：

```python
chunk_gated_delta_rule(..., cu_seqlens=query_start_loc)
```

内部会根据 `cu_seqlens` 生成 CPU/Python 决定的辅助 tensor：

```python
prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
```

其中 `prepare_chunk_indices` 使用：

```python
triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
```

后续多个 kernel 又用：

```python
len(chunk_indices)
len(cu_seqlens) - 1
```

来决定 grid，并分配中间 tensor：

```python
A = torch.zeros(B, T, H, BT, ...)
h = k.new_empty(B, NT, H, V, K)
o = torch.zeros_like(v)
```

所以不能只稳定 `seq_lens_cpu`。inner graph 还需要一种 graph-friendly 的 GDN extend path。当前第一版不采用统一 2k 上限，而是用 64 粒度 bucket 固定 FLA chunk metadata：

```python
static_bs = 1
static_num_tokens = ceil_align(real_num_tokens, 64)  # 64..2048
num_chunks = static_num_tokens // FLA_CHUNK_SIZE

chunk_indices_static = [[0, i] for i in range(num_chunks)]
chunk_offsets_static = [0, num_chunks]
```

- `chunk_indices_static` 和 `chunk_offsets_static` 按当前 `static_num_tokens` bucket 预生成，地址和长度固定。
- capture 生成 metadata 时使用 static `query_start_loc=[0, static_num_tokens]`，避免 `@tensor_cache` 记住 exact real length。
- replay 只更新 graph-owned `query_start_loc` 内容为 `[0, real_num_tokens]`。
- 因为 `static_num_tokens=ceil_align(real_num_tokens, 64)`，`num_chunks` 等于 `ceil(real_num_tokens / FLA_CHUNK_SIZE)`；不会额外多出完整 FLA chunk，只有最后一个 chunk 可能包含 padding tail。
- 中间 tensor shape 按 static bucket 固定，例如 `h` 的 chunk 维固定为 `num_chunks`。

如果继续让 `prepare_chunk_indices` 在 replay 里根据真实 `query_start_loc` 重新生成 tensor，那么 graph 仍然会被真实长度和 Python 分配绑住。

#### `fused_gdn_gating`

`fused_gdn_gating` 的 Python grid 是：

```python
grid = (batch, seq_len, triton.cdiv(num_heads, 8))
```

在当前 GDN extend 里，`a/b` 应当按 `static_num_tokens` 固定 shape，因此它不引入额外 CPU 动态性。需要保证不要再按 `real_num_tokens` 裁剪 `a/b` 后再调用它。

#### FLA `input_guard` / contiguous copy

`ChunkGatedDeltaRuleFunction.forward` 带有 `@input_guard`，会对 tensor 参数调用 `.contiguous()`。如果 capture 和 replay 的输入 layout 不一致，可能在 graph 内引入额外 copy/分配，或者让后续 kernel 看到不同地址。

第一版应要求进入 GDN chunk path 的 `q/k/v/g/beta` layout 固定；更稳妥的做法是 inner runner 使用 graph-owned contiguous buffers，或在 capture 前确认这些 tensor 已经是预期 contiguous layout。

### 6. prefix/mamba state tracking 的处理

当前服务参数是 `mamba_scheduler_strategy='extra_buffer'`，也就是 `enable_mamba_extra_buffer()` 为 true。不能长期假设 mamba tracking 不存在。

但是第一版可以分两档：

#### 6.1 最小原型

如果：

```text
forward_metadata.has_mamba_track_mask == False
```

则允许 inner graph；否则 fallback 到 eager。

这能先验证 GDN core 主路径：

```text
causal_conv1d_fn
fused_gdn_gating
TritonGDNKernel.extend
output copy
```

#### 6.2 覆盖当前 cached-token prefill

如果当前 workload 的 cached-token prefill 会触发 mamba tracking，就必须把 tracking metadata 也稳定化。

需要预分配并在 replay 前填充：

- `track_conv_indices_buf`: shape `[static_bs, conv_state_len]`
- `conv_states_mask_indices_buf`: shape `[static_bs]`
- `track_ssm_h_src_buf`
- `track_ssm_h_dst_buf`
- `track_ssm_final_src_buf`
- `track_ssm_final_dst_buf`
- 对应的 count/mask buffer，用于标记有效元素

当前 `_init_track_ssm_indices` 会在 Python 中做 `.cpu()` 往返并生成变长 tensor。第一版可以接受这些 CPU 计算发生在 graph replay 前，但不能让 graph 捕获新分配出来的变长 tensor 地址。计算结果必须 copy 到 graph-owned 固定 buffer。

如果 `TritonGDNKernel.extend` 返回的 `h` shape 取决于 chunk 数，则 `h` 的 shape 也要按 64 粒度 bucket 稳定。不能把真实 `total_h_chunks = sum(ceil(seq_len_i / FLA_CHUNK_SIZE))` 纳入 key；当前场景只支持 `static_bs=1` 且 `static_num_tokens=ceil_align(real_num_tokens, 64)`，因此 `h` 的 chunk 维应固定为 `static_num_tokens / FLA_CHUNK_SIZE`。真实长度仍只通过 `query_start_loc=[0, real_num_tokens]` 控制有效 token，padding tail 不能被建成第二个真实 sequence。

### 7. 保持 `ForwardBatch` 的临时 mutation 可重放

`unified_linear_attention_with_output` 当前会临时修改：

- `forward_batch.out_cache_loc`
- `forward_batch.out_cache_loc_swa`
- `token_to_kv_pool.swa_loc`

这些 Python mutation 在 CUDA graph replay 时不会重新执行。当前 GDN CUDA normal extend 主路径不依赖 `out_cache_loc/out_cache_loc_swa` 写 mamba state，因此 graph path 不应该捕获一个依赖临时 view 的隐式状态。

因此：

- 当前 GDN CUDA normal extend graph path 应直接跳过这组 `out_cache_loc/out_cache_loc_swa` narrow mutation。
- eager fallback 继续保留现有 mutation/restore 逻辑，保证行为不变。
- 如果后续某个 linear backend 确实需要这些 loc，必须改成 graph-owned 固定地址 buffer；不能在 graph path 里捕获 `[:real_num_tokens]` 这种临时 view。

### 8. 输出写回策略

当前函数签名是 mutating op：

```python
unified_linear_attention_with_output(..., output, layer_id) -> None
```

第一版建议让 inner graph 写入静态 output buffer，然后只把有效前缀暴露给外层逻辑。如果输出地址检查失败，再退到专用 output buffer：

```text
inner_graph_output[:, :static_num_tokens] -> output[:, :static_num_tokens]
外层只消费 output[:, :real_num_tokens]
```

后者会多一个图外 copy kernel，但仍能先验证 GDN core 入图。

## 不推荐的第一版做法

### 直接删除 `@register_split_op()`

这会把 opaque custom op 放回外层 PCG 子图里。理论上 CUDA graph capture 可以记录 custom op body 里 launch 的 kernels，但仍会遇到同样的 metadata 地址问题：

- `query_start_loc` 每次新分配。
- `mamba_cache_indices` 每次从 pool 查询得到新 tensor。
- tracking 索引可能是新分配的变长 tensor。
- `seq_lens_cpu` 可能影响 launch 参数。

所以即使删除 split op，也必须先完成 metadata 稳定化。否则 replay 会使用 capture 时的旧 metadata 地址。

### 暂不适配 TBO / Lightning / Ascend

当前日志没有走这些路径。第一版不处理：

- `TboAttnBackend`
- `LightningAttentionBackend`
- Ascend/NPU backend
- target verify / speculative decode

## 推荐实施顺序

1. 在文档和代码注释中明确：当前问题是 split op 内部 GDN core 图外，不是外层 PCG 没命中。
2. 在 `unified_linear_attention_with_output` 前后增加 inner graph eligibility 判断；只有外层 piecewise CUDA graph 运行态才尝试，PCG torch.compile 阶段和普通 eager 路径完全保持现有 eager split op。同时把 `get_pcg_capture_stream() is not None` 作为 `allow_capture` 传给 inner runner，确保只有外层 PCG capture 阶段能创建或重捕获 inner graph。
3. 为 `GDNAttnBackend.forward_extend` 增加当前场景专用 inner CUDA graph runner，key 使用 `(layer_id, static_num_tokens, static_bs)`；其中 `static_bs=1`，`static_num_tokens` 只允许 64 粒度 bucket：`64, 128, ..., 2048`。某个 bucket 首次 capture 成功后，runner 需要立即 replay 一次刚捕获的 graph 来产出当前请求结果和 state 更新。
4. 给 inner runner 增加输入/输出地址检查；同一 bucket 只在 `allow_capture=True` 时允许一次 recapture，之后地址仍不稳定则 fallback。`allow_capture=False` 时如果 entry 不存在、graph 不存在或地址签名不一致，都只 fallback，不临时 capture/recapture。
5. 增加 graph-owned `ForwardMetadata` buffer，capture/replay 都用同一组 `query_start_loc` 和 `mamba_cache_indices` 地址。
6. 处理 `extend_seq_lens_cpu`：当前只支持 `static_bs=1`、`seq_lens_cpu=[real_num_tokens]`、`query_start_loc=[0, real_num_tokens]`，并通过 `grid_max_seq_len=static_num_tokens` 固定 causal conv grid；`static_num_tokens` 必须是覆盖真实长度的 64 粒度 bucket，其他 batch 形态 fallback。
7. 先在 `has_mamba_track_mask=False` 时验证 GDN core 入图。
8. 如果当前 cached-token prefill 会触发 tracking，再把 tracking 派生索引改成固定 buffer replay。
9. 验证 logits、mamba cache 状态和 graph hit rate；第一版不扩大到其他 `static_bs` 或非 64 粒度 token bucket。

## 验证点

必须验证：

- 关闭 inner graph 和开启 inner graph 的 logits 一致。
- `mixed_qkv/a/b/output/query_start_loc/mamba_cache_indices` replay 地址稳定。
- 某个 bucket 的首次 capture path 会在 capture 后立即 replay 一次，并且首请求的 logits 和 mamba cache state 与 eager 一致。
- dummy warmup 阶段不会创建 inner graph；外层实际 replay 阶段只复用已有 inner graph，不发生临时 capture/recapture。
- `static_num_tokens=64`、`128`、`1984`、`2048` 等 64 粒度 bucket 能命中 inner graph。
- `real_num_tokens=1984` 这类已 64 对齐的长度命中 `static_num_tokens=1984`；`real_num_tokens=1985` 这类非对齐长度命中 `static_num_tokens=2048`，且 padding tail 不污染 mamba cache。
- `#new-seq > 1` 或非 `static_bs=1` 的 batch 明确 fallback。
- `#new-token=5888` 这类外层 PCG 已经 fallback 的 batch 不触发 inner graph。
- mamba cache 没有被 padding tail 或 dummy token 污染。
- `has_mamba_track_mask=True` 时明确 fallback eager；第一版不验证 tracking 入图。

当前 runner 已有用于验证的计数字段：

```text
capture_total
recapture_total
replay_total
fallback_total
last_fallback_reason
```

访问路径是：

```python
runner = model_runner.attn_backend.linear_attn_backend.qwen35_prefill_cuda_graph_runner
```

这些计数器只用于验证当前改造，不需要在热路径长期打日志。

注意：已捕获 graph 的 `replay()` 运行期异常不应被静默吞掉再改走 eager。CUDA graph replay 失败可能意味着 CUDA 上下文已有异步错误或部分 kernel 状态不可判定，继续执行 eager 容易掩盖真实问题。当前 runner 只对 capture/recapture 阶段的失败做状态恢复和 eager fallback；稳定 graph 的 replay 阶段应暴露异常，用于定位 capture/replay 不一致问题。
