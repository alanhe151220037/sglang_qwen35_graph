from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

_TensorCudaGraphSignature = Tuple[
    int, Tuple[int, ...], Tuple[int, ...], torch.dtype, torch.device
]

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.fla.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


@dataclass
class _Qwen35GDNPrefillCudaGraphEntry:
    query_start_loc: torch.Tensor
    mamba_cache_indices: torch.Tensor
    extend_prefix_lens: torch.Tensor
    forward_metadata: ForwardMetadata
    graph: Optional[object] = None
    input_signature: Optional[Tuple[_TensorCudaGraphSignature, ...]] = None
    output_ref: Optional[torch.Tensor] = None
    static_chunk_refs: Tuple[torch.Tensor, ...] = ()
    recapture_count: int = 0
    failed: bool = False
    failure_reason: Optional[str] = None


class Qwen35GDNPrefillCudaGraphRunner:
    """Inner CUDA graph runner for the current Qwen3.5 GDN prefill path."""

    MIN_NUM_TOKENS = 64
    MAX_NUM_TOKENS = 2048
    TOKEN_BUCKET_SIZE = 64
    STATIC_BS = 1
    MAX_RECAPTURES_PER_ENTRY = 1

    def __init__(self, backend: "GDNAttnBackend"):
        self.backend = backend
        self.entries: Dict[
            Tuple[int, int, int, torch.dtype, torch.device],
            _Qwen35GDNPrefillCudaGraphEntry,
        ] = {}
        self.capture_total = 0
        self.recapture_total = 0
        self.replay_total = 0
        self.fallback_total = 0
        self.last_fallback_reason = ""

    def _fallback(self, reason: str) -> bool:
        self.fallback_total += 1
        self.last_fallback_reason = reason
        return False

    @staticmethod
    def _entry_failure_reason(
        entry: _Qwen35GDNPrefillCudaGraphEntry, fallback_reason: str
    ) -> str:
        if entry.failure_reason is None:
            return fallback_reason
        return f"{fallback_reason}: {entry.failure_reason}"

    @classmethod
    def _ceil_token_bucket(cls, num_tokens: int) -> int:
        return (
            (num_tokens + cls.TOKEN_BUCKET_SIZE - 1)
            // cls.TOKEN_BUCKET_SIZE
            * cls.TOKEN_BUCKET_SIZE
        )

    def try_run(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        real_num_tokens: int,
        allow_capture: bool,
    ) -> bool:
        ineligible_reason = self._eligibility_failure_reason(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            output=output,
            real_num_tokens=real_num_tokens,
        )
        if ineligible_reason is not None:
            return self._fallback(ineligible_reason)

        is_stream_capturing = torch.cuda.is_current_stream_capturing()
        static_num_tokens = mixed_qkv.shape[0]
        key = (
            layer.layer_id,
            static_num_tokens,
            self.STATIC_BS,
            mixed_qkv.dtype,
            mixed_qkv.device,
        )
        entry = self.entries.get(key)
        if entry is None:
            if not allow_capture or is_stream_capturing:
                return self._fallback("capture_not_allowed")
            try:
                entry = self._new_entry(static_num_tokens, mixed_qkv.device)
            except Exception as exc:
                return self._fallback(f"entry_init_failed: {repr(exc)}")
            self.entries[key] = entry

        if entry.failed:
            return self._fallback(self._entry_failure_reason(entry, "entry_failed"))

        try:
            self._update_entry(entry, forward_batch, real_num_tokens)
        except Exception as exc:
            entry.failure_reason = repr(exc)
            return self._fallback(
                self._entry_failure_reason(entry, "metadata_update_failed")
            )
        entry.failure_reason = None
        input_signature = self._input_signature(
            entry, layer, mixed_qkv, a, b, output
        )

        if entry.graph is None:
            if not allow_capture or is_stream_capturing:
                return self._fallback("capture_not_allowed")
            captured = self._capture(
                entry=entry,
                layer=layer,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                output=output,
                static_num_tokens=static_num_tokens,
                real_num_tokens=real_num_tokens,
                input_signature=input_signature,
            )
            if not captured:
                return self._fallback(
                    self._entry_failure_reason(entry, "capture_failed")
                )
            self.capture_total += 1
            return True

        if entry.input_signature != input_signature:
            if not allow_capture or is_stream_capturing:
                return self._fallback("input_signature_mismatch_replay")
            # Allow one recapture to get past startup/warmup addresses. If the
            # signature keeps moving, graph reuse is not meaningful for this bucket.
            if entry.recapture_count >= self.MAX_RECAPTURES_PER_ENTRY:
                entry.failed = True
                entry.failure_reason = "input_signature_unstable"
                self._clear_graph(entry)
                return self._fallback(
                    self._entry_failure_reason(entry, "entry_failed")
                )
            self._clear_graph(entry)
            captured = self._capture(
                entry=entry,
                layer=layer,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                output=output,
                static_num_tokens=static_num_tokens,
                real_num_tokens=real_num_tokens,
                input_signature=input_signature,
            )
            if not captured:
                return self._fallback(
                    self._entry_failure_reason(entry, "recapture_failed")
                )
            self.recapture_total += 1
            entry.recapture_count += 1
            return True

        self._replay(entry, mixed_qkv.device)
        return True

    def _eligibility_failure_reason(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        real_num_tokens: int,
    ) -> Optional[str]:
        if not is_cuda() or not mixed_qkv.is_cuda:
            return "not_cuda"
        if not a.is_cuda or not b.is_cuda or not output.is_cuda:
            return "non_cuda_input"
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return f"forward_mode_{forward_batch.forward_mode}"
        if forward_batch.spec_info is not None:
            return "spec_info"
        if forward_batch.batch_size != self.STATIC_BS:
            return f"batch_size_{forward_batch.batch_size}"
        if forward_batch.extend_num_tokens != real_num_tokens:
            return "extend_num_tokens_mismatch"
        if (
            forward_batch.extend_seq_lens_cpu is None
            or len(forward_batch.extend_seq_lens_cpu) != self.STATIC_BS
        ):
            return "extend_seq_lens_cpu_shape"
        if self._first_seq_len_cpu(forward_batch.extend_seq_lens_cpu) != real_num_tokens:
            return "extend_seq_lens_cpu_mismatch"
        if real_num_tokens <= 0:
            return f"real_num_tokens_{real_num_tokens}"
        if mixed_qkv.ndim != 2 or a.ndim != 2 or b.ndim != 2 or output.ndim != 4:
            return "input_rank_mismatch"

        static_num_tokens = mixed_qkv.shape[0]
        if static_num_tokens != self._ceil_token_bucket(real_num_tokens):
            return "static_num_tokens_not_bucket"
        if (
            static_num_tokens < self.MIN_NUM_TOKENS
            or static_num_tokens > self.MAX_NUM_TOKENS
            or static_num_tokens % self.TOKEN_BUCKET_SIZE != 0
        ):
            return f"static_num_tokens_{static_num_tokens}"
        if a.shape[0] != static_num_tokens or b.shape[0] != static_num_tokens:
            return "gating_shape_mismatch"
        if output.shape[0] != self.STATIC_BS or output.shape[1] != static_num_tokens:
            return "output_shape_mismatch"
        if not isinstance(
            self.backend.kernel_dispatcher.extend_kernel, TritonGDNKernel
        ):
            return "non_triton_extend_kernel"

        forward_metadata = self.backend.forward_metadata
        if forward_metadata is None:
            return "missing_forward_metadata"
        if forward_metadata.has_mamba_track_mask:
            return "mamba_state_tracking"
        if (
            forward_metadata.query_start_loc is None
            or forward_metadata.mamba_cache_indices is None
        ):
            return "missing_mamba_metadata"
        if not forward_metadata.query_start_loc.is_cuda:
            return "query_start_loc_not_cuda"
        if forward_metadata.query_start_loc.dtype != torch.int32:
            return "query_start_loc_dtype"
        if forward_metadata.query_start_loc.ndim != 1:
            return "query_start_loc_rank"
        if forward_metadata.query_start_loc.numel() != self.STATIC_BS + 1:
            return "query_start_loc_shape"
        if not forward_metadata.mamba_cache_indices.is_cuda:
            return "mamba_cache_indices_not_cuda"
        if forward_metadata.mamba_cache_indices.dtype != torch.int32:
            return "mamba_cache_indices_dtype"
        if forward_metadata.mamba_cache_indices.numel() != self.STATIC_BS:
            return "mamba_cache_indices_shape"
        if forward_metadata.mamba_cache_indices.ndim != 1:
            return "mamba_cache_indices_rank"
        if (
            forward_batch.extend_prefix_lens is None
            or forward_batch.extend_prefix_lens.numel() != self.STATIC_BS
        ):
            return "extend_prefix_lens_shape"
        if not forward_batch.extend_prefix_lens.is_cuda:
            return "extend_prefix_lens_not_cuda"
        if forward_batch.extend_prefix_lens.ndim != 1:
            return "extend_prefix_lens_rank"
        return None

    @staticmethod
    def _first_seq_len_cpu(seq_lens_cpu) -> int:
        value = seq_lens_cpu[0]
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    def _input_signature(
        self,
        entry: _Qwen35GDNPrefillCudaGraphEntry,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
    ) -> Tuple[_TensorCudaGraphSignature, ...]:
        layer_cache = self.backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        tensors = (
            mixed_qkv,
            a,
            b,
            output,
            layer.conv_weights,
            layer.bias,
            layer.A_log,
            layer.dt_bias,
            layer_cache.conv[0],
            layer_cache.temporal,
            entry.query_start_loc,
            entry.mamba_cache_indices,
            entry.extend_prefix_lens,
        )
        return tuple(
            (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
            )
            for tensor in self._iter_signature_tensors(tensors)
        )

    @staticmethod
    def _iter_signature_tensors(values):
        for value in values:
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        yield item

    @staticmethod
    def _clear_graph(entry: _Qwen35GDNPrefillCudaGraphEntry) -> None:
        entry.graph = None
        entry.input_signature = None
        entry.output_ref = None
        entry.static_chunk_refs = ()

    @staticmethod
    def _torch_index(index: torch.Tensor) -> torch.Tensor:
        return index if index.dtype == torch.long else index.to(dtype=torch.long)

    def _replay(
        self, entry: _Qwen35GDNPrefillCudaGraphEntry, device: torch.device
    ) -> None:
        self._replay_graph(entry, device)
        self.replay_total += 1

    @staticmethod
    def _replay_graph(
        entry: _Qwen35GDNPrefillCudaGraphEntry, device: torch.device
    ) -> None:
        assert entry.graph is not None
        assert entry.static_chunk_refs
        with torch.cuda.device(device):
            entry.graph.replay()

    @staticmethod
    def _graph_capture_kwargs() -> Dict[str, object]:
        from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
            get_global_graph_memory_pool,
        )

        graph_pool = get_global_graph_memory_pool()
        if graph_pool is None:
            return {}
        return {"pool": graph_pool}

    def _new_entry(
        self, static_num_tokens: int, device: torch.device
    ) -> _Qwen35GDNPrefillCudaGraphEntry:
        query_start_loc = torch.empty(
            (self.STATIC_BS + 1,), dtype=torch.int32, device=device
        )
        mamba_cache_indices = torch.empty(
            (self.STATIC_BS,), dtype=torch.int32, device=device
        )
        extend_prefix_lens = torch.empty(
            (self.STATIC_BS,), dtype=torch.int32, device=device
        )
        forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            has_mamba_track_mask=False,
        )
        entry = _Qwen35GDNPrefillCudaGraphEntry(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            extend_prefix_lens=extend_prefix_lens,
            forward_metadata=forward_metadata,
        )
        self._prime_fla_chunk_metadata(entry, static_num_tokens)
        return entry

    def _update_entry(
        self,
        entry: _Qwen35GDNPrefillCudaGraphEntry,
        forward_batch: ForwardBatch,
        real_num_tokens: int,
    ) -> None:
        entry.query_start_loc[0].fill_(0)
        entry.query_start_loc[1].fill_(real_num_tokens)
        entry.mamba_cache_indices.copy_(
            self.backend.forward_metadata.mamba_cache_indices[: self.STATIC_BS]
        )
        entry.extend_prefix_lens.copy_(
            forward_batch.extend_prefix_lens[: self.STATIC_BS]
        )

    def _prime_fla_chunk_metadata(
        self, entry: _Qwen35GDNPrefillCudaGraphEntry, static_num_tokens: int
    ) -> None:
        assert entry.graph is None
        # FLA's tensor_cache keys on tensor identity. Seed the cache with the
        # static bucket shape before capture. The captured graph stores pointers
        # to these tensors, so keep entry.static_chunk_refs alive for replay.
        entry.query_start_loc[0].fill_(0)
        entry.query_start_loc[1].fill_(static_num_tokens)
        chunk_indices = prepare_chunk_indices(entry.query_start_loc, FLA_CHUNK_SIZE)
        chunk_offsets = prepare_chunk_offsets(entry.query_start_loc, FLA_CHUNK_SIZE)
        entry.static_chunk_refs = (chunk_indices, chunk_offsets)

    def _run_forward(
        self,
        *,
        entry: _Qwen35GDNPrefillCudaGraphEntry,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        static_num_tokens: int,
    ) -> torch.Tensor:
        original_metadata = self.backend.forward_metadata
        original_extend_prefix_lens = forward_batch.extend_prefix_lens
        self.backend.forward_metadata = entry.forward_metadata
        forward_batch.extend_prefix_lens = entry.extend_prefix_lens
        try:
            return self.backend.forward_extend(
                layer=layer,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                grid_max_seq_len=static_num_tokens,
            )
        finally:
            self.backend.forward_metadata = original_metadata
            forward_batch.extend_prefix_lens = original_extend_prefix_lens

    def _backup_states(
        self,
        *,
        layer: RadixLinearAttention,
        entry: _Qwen35GDNPrefillCudaGraphEntry,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_cache = self.backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        cache_indices = self._torch_index(entry.mamba_cache_indices)
        return (
            conv_states,
            ssm_states,
            conv_states[cache_indices].clone(),
            ssm_states[cache_indices].clone(),
        )

    def _restore_states(
        self,
        *,
        entry: _Qwen35GDNPrefillCudaGraphEntry,
        states_backup: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        conv_states, ssm_states, conv_backup, ssm_backup = states_backup
        cache_indices = self._torch_index(entry.mamba_cache_indices)
        conv_states[cache_indices] = conv_backup
        ssm_states[cache_indices] = ssm_backup

    def _capture(
        self,
        *,
        entry: _Qwen35GDNPrefillCudaGraphEntry,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        static_num_tokens: int,
        real_num_tokens: int,
        input_signature: Tuple[_TensorCudaGraphSignature, ...],
    ) -> bool:
        states_backup = None
        try:
            entry.failure_reason = None
            self._prime_fla_chunk_metadata(entry, static_num_tokens)
            self._update_entry(entry, forward_batch, real_num_tokens)
            states_backup = self._backup_states(layer=layer, entry=entry)
            with torch.no_grad():
                self._run_forward(
                    entry=entry,
                    layer=layer,
                    forward_batch=forward_batch,
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    static_num_tokens=static_num_tokens,
                )
            self._restore_states(entry=entry, states_backup=states_backup)
            torch.cuda.synchronize(mixed_qkv.device)

            with torch.no_grad(), torch.cuda.device(mixed_qkv.device):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, **self._graph_capture_kwargs()):
                    ret = self._run_forward(
                        entry=entry,
                        layer=layer,
                        forward_batch=forward_batch,
                        mixed_qkv=mixed_qkv,
                        a=a,
                        b=b,
                        static_num_tokens=static_num_tokens,
                    )
                    output.copy_(ret)
        except Exception as exc:
            if states_backup is not None:
                self._restore_states(entry=entry, states_backup=states_backup)
            self._clear_graph(entry)
            entry.failed = True
            entry.failure_reason = repr(exc)
            return False

        entry.graph = graph
        entry.input_signature = input_signature
        entry.output_ref = ret
        assert entry.static_chunk_refs
        try:
            self._replay_graph(entry, mixed_qkv.device)
        except Exception as exc:
            if states_backup is not None:
                self._restore_states(entry=entry, states_backup=states_backup)
            self._clear_graph(entry)
            entry.failed = True
            entry.failure_reason = repr(exc)
            return False
        return True


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer only when the selected FlashInfer kernel
        # supports MTP verify. On SM100+ FlashInfer GDN decode is supported, but
        # its MTP verify path is not, so keep Triton as the verify fallback.
        if (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )
        self.qwen35_prefill_cuda_graph_runner = (
            Qwen35GDNPrefillCudaGraphRunner(self)
            if is_cuda() and self._is_qwen35_model(model_runner)
            else None
        )

    @staticmethod
    def _is_qwen35_model(model_runner: ModelRunner) -> bool:
        model_config = getattr(model_runner, "model_config", None)
        configs = (
            getattr(model_config, "hf_config", None),
            getattr(model_config, "hf_text_config", None),
        )
        return any(
            str(getattr(config, "model_type", "")).startswith("qwen3_5")
            for config in configs
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

    def try_run_qwen35_prefill_cuda_graph(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        real_num_tokens: int,
        allow_capture: bool,
    ) -> bool:
        if self.qwen35_prefill_cuda_graph_runner is None:
            return False
        return self.qwen35_prefill_cuda_graph_runner.try_run(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            output=output,
            real_num_tokens=real_num_tokens,
            allow_capture=allow_capture,
        )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            conv_kwargs = {}
            grid_max_seq_len = kwargs.get("grid_max_seq_len")
            if grid_max_seq_len is not None:
                conv_kwargs["grid_max_seq_len"] = grid_max_seq_len

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                **conv_kwargs,
            ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out
