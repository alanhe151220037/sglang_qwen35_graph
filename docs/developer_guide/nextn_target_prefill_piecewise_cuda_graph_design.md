# NEXTN Target Prefill Piecewise CUDA Graph Design

## Status

Implemented on the current branch. CPU unit coverage and the NEXTN GPU
integration assertion are included; full GPU validation is still required in
the deployment environment.

## Summary

The target model should keep using Piecewise CUDA Graph (PCG) for normal
prefill/extend when speculative decoding is enabled with `NEXTN`. The target
prefill must return full hidden states for the NEXTN draft prefill, while the
draft model itself remains outside PCG.

The target PCG is captured with `CaptureHiddenMode.FULL`, uses that same mode
during warmup and graph capture, and retains the existing runtime mode-equality
guard. After replay, the real-token view of the graph output is handed directly
to the NEXTN draft prefill. Mamba tracking inputs must also remain available to
the target graph when `extra_buffer` radix caching is enabled.

## Goals

1. Let NEXTN target-model prefill use the existing
   `PiecewiseCudaGraphRunner`.
2. Return the full target hidden-state tensor required by NEXTN draft prefill.
3. Keep the NEXTN draft model outside PCG.
4. Keep target verification outside PCG.
5. Preserve Mamba state tracking during target prefill when
   `extra_buffer` and radix caching are enabled.
6. Preserve current behavior for non-speculative prefill and configurations
   that explicitly request hidden states.

## Non-Goals

- Capturing the NEXTN draft-model forward in PCG.
- Capturing target verification in PCG.
- Combining target and draft execution into one CUDA graph.
- Changing the NEXTN draft CUDA graph, decode CUDA graph, or speculative
  acceptance logic.
- Passing `EagleVerifyInput` or other speculative `spec_info` into target
  prefill capture.
- Changing Mamba radix-cache milestone, insertion, matching, or eviction
  policy.
- Changing PD cache transfer contents or pointer layout.
- Enabling PCG automatically for PD disaggregation without the existing
  deployment override.

## Current Execution Flow

NEXTN is normalized to the built-in `EAGLE` implementation for the relevant
Qwen3.5 deployment. Target and draft prefill are still separate forwards:

```text
EAGLE/NEXTN worker
  -> set target batch capture_hidden_mode = FULL
  -> target_worker.forward_batch_generation(...)
       -> ModelRunner.forward_extend(...)
            -> PiecewiseCudaGraphRunner.can_run(...)
            -> PCG replay or eager target forward
  -> read target LogitsProcessorOutput.hidden_states
  -> draft_worker._draft_extend_for_prefill(..., target_hidden_states, ...)
```

The draft `ModelRunner` does not create a `PiecewiseCudaGraphRunner` because
`ModelRunner.init_piecewise_cuda_graphs()` returns immediately for
`is_draft_worker`. This separation is already the desired architecture and
must remain unchanged.

`PiecewiseCudaGraphRunner.replay()` slices `output.hidden_states` back to the
unpadded token count and returns that view to the draft prefill.

## Current Failure Mode

`PiecewiseCudaGraphRunner` currently has three conflicting hidden-mode values:

1. Runner initialization normally sets `self.capture_hidden_mode` to `NULL`.
2. Both warmup and graph capture construct their `ForwardBatch` with a
   hard-coded `CaptureHiddenMode.NULL`.
3. NEXTN target prefill changes the runtime batch to
   `CaptureHiddenMode.FULL` because draft prefill needs every target hidden
   state.

`can_run()` correctly requires the runtime mode to equal the runner capture
mode:

```text
runtime FULL != runner NULL -> PCG rejected -> target prefill runs eager
```

Removing this equality check would be incorrect. A graph captured with `NULL`
does not satisfy the output contract of a `FULL` request and can return missing
hidden states to the draft worker.

## Proposed Design

### 1. Select One Runner-Level Hidden Mode

Derive the PCG capture mode once during target runner initialization:

```python
def select_pcg_capture_hidden_mode(model_runner):
    if model_runner.server_args.enable_return_hidden_states:
        return CaptureHiddenMode.FULL
    if (
        not model_runner.is_draft_worker
        and not model_runner.server_args.enable_multi_layer_eagle
        and model_runner.spec_algorithm == SpeculativeAlgorithm.EAGLE
    ):
        return CaptureHiddenMode.FULL
    return CaptureHiddenMode.NULL
```

The `is_draft_worker` check documents the target-only contract even though
draft workers already skip PCG initialization.

`NEXTN` and the ordinary single-layer `EAGLE` spelling are indistinguishable
after server-argument normalization. The predicate should intentionally use
the exact normalized `SpeculativeAlgorithm.EAGLE` value instead of the broader
`is_eagle()` helper, which also includes EAGLE3 and `FROZEN_KV_MTP`. Supporting
those variants requires separate validation and is outside this change. The
exact predicate still does not enable PCG for the single-layer EAGLE/NEXTN
draft worker. It also explicitly excludes `--enable-multi-layer-eagle`, whose
worker and hidden-state contract require separate validation.

### 2. Use the Selected Mode Everywhere

Use `self.capture_hidden_mode` instead of hard-coded `NULL` in both synthetic
`ForwardBatch` construction sites:

- PCG warmup/compile forward;
- per-token-bucket CUDA graph capture.

The same value then governs all three phases:

```text
warmup mode == capture mode == accepted replay mode
```

This also fixes the existing inconsistency for
`--enable-return-hidden-states`, where runner initialization can select `FULL`
while the synthetic capture batches still use `NULL`.

### 3. Preserve Runtime Admission Guards

Keep the following `can_run()` behavior:

- reject `TARGET_VERIFY`;
- reject runtime `capture_hidden_mode` values that differ from
  `self.capture_hidden_mode`;
- retain all existing input-embedding, replacement-embedding, logprob, and
  maximum-token guards.

Target verification has a speculative attention layout and non-null
`spec_info`; it must continue through its existing non-PCG path. Normal target
prefill remains `EXTEND` with `spec_info=None` and is the only newly admitted
NEXTN operation.

### 4. Keep Capture Free of Draft Semantics

Do not construct draft inputs or `EagleVerifyInput` for PCG target-prefill
capture. The synthetic capture batch remains:

```text
forward mode: EXTEND
spec_info: None
capture hidden mode: FULL for an EAGLE/NEXTN target runner
```

The runtime static `ForwardBatch` may retain the configured speculative
algorithm as ordinary metadata, but the captured target-prefill path must not
branch into draft or target-verification behavior.

### 5. Preserve the Full Hidden-State Output

PCG replay post-processing removes padding before returning the graph output:

```python
hidden_states = output.hidden_states[:raw_num_tokens]
```

The implementation must retain this slicing because a PCG token bucket may be
larger than the real prefill token count. Target replay and draft consumption
remain sequential on the same CUDA stream.

The resulting execution remains sequential:

```text
target PCG replay returns graph-owned FULL hidden states
  -> slice to the real token count
  -> NEXTN draft prefill consumes the returned view
```

### 6. Preserve Mamba Tracking Inputs

When `extra_buffer` is enabled, the scheduler computes per-request target
prefill tracking data:

- `mamba_track_mask`: whether this request tracks a checkpoint;
- `mamba_track_indices`: the persistent Mamba-pool destination slot;
- `mamba_track_seqlens`: the sequence position whose state is tracked.

PCG cannot directly capture request-specific tensor addresses. It allocates
fixed-address buffers and copies each runtime batch's values into them before
replay.

The current PCG allocation gate requires `spec_algorithm.is_none()`, so an
EAGLE/NEXTN target runner allocates none of these buffers. Once target prefill
is admitted to PCG, leaving this gate unchanged would drop the scheduler's
tracking inputs and could leave the radix cache referring to an unrefreshed
tracking slot.

Allow fixed Mamba tracking buffers for either:

```text
non-speculative target prefill
or
EAGLE/NEXTN target prefill
```

The remaining conditions stay unchanged:

```text
extra_buffer enabled
and radix cache enabled
and target prefill is eligible
```

This change does not put Mamba state handling into the NEXTN draft graph. The
buffers belong to the target `PiecewiseCudaGraphRunner`, and target
verification is still rejected by `can_run()`.

## Expected Behavior Matrix

| Operation | Hidden mode | PCG result |
|---|---:|---|
| Normal non-speculative target prefill | `NULL` | Existing PCG behavior |
| `--enable-return-hidden-states` prefill | `FULL` | PCG captured/replayed as `FULL` |
| NEXTN/single-layer EAGLE target prefill | `FULL` | Newly allowed in PCG |
| NEXTN/single-layer EAGLE draft prefill | `LAST` or draft-specific mode | No PCG runner |
| NEXTN/single-layer EAGLE target verify | `FULL`, with speculative metadata | Rejected by PCG |
| EAGLE3 or `FROZEN_KV_MTP` target prefill | Variant-specific | Unchanged by this design |
| Standalone speculative target prefill | `NULL` | Existing behavior |
| Runtime mode different from captured mode | Any mismatch | Rejected by PCG |

## Source Changes

The implementation should remain concentrated in
`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`:

1. Select `self.capture_hidden_mode` from the target runner contract.
2. Replace both synthetic-batch `CaptureHiddenMode.NULL` values with
   `self.capture_hidden_mode`.
3. Extend the Mamba tracking-buffer eligibility condition to normal or
   normalized single-layer EAGLE/NEXTN target prefill.
4. Keep `can_run()`, replay hidden-state slicing, draft-runner initialization,
   and target-verify handling
   otherwise unchanged.

No changes should be required in:

- `eagle_worker.py` or `eagle_worker_v2.py`;
- NEXTN draft model implementations;
- target/draft transfer code;
- Mamba radix-cache ownership or eviction code.

## Configuration and PD Deployment

No new server argument is required. The relevant speculative option remains:

```text
--speculative-algorithm NEXTN
```

PCG is normally enabled for supported configurations, but the current server
argument handling auto-disables it for PD disaggregation. A PD prefill server
that intentionally uses this path must continue to pass:

```text
--enforce-piecewise-cuda-graph
```

The decode server does not need PCG for this design. Its speculative decode
graphs and draft execution are unaffected.

## Validation Plan

### Functional Validation

1. Start Qwen3.5 with `NEXTN`, `extra_buffer`, radix cache, PCG, and metrics
   enabled.
2. Send a prefill request whose token count falls within a captured PCG bucket.
3. Verify the target prefill reports `cuda graph: True`.
4. Verify NEXTN draft prefill receives non-null hidden states with exactly the
   real target token count.
5. Verify generation completes and speculative acceptance remains above the
   existing test threshold.

### PCG Hit Validation

Strengthen the existing NEXTN PCG integration test. The current GSM8K score and
average speculative acceptance checks prove output quality and NEXTN activity,
but they do not prove target prefill actually used PCG.

The test must additionally assert that the prefill graph metric increases:

```text
sglang:cuda_graph_passes_total{mode="prefill_cuda_graph"}
```

An equivalent direct assertion on the returned `can_run_cuda_graph` value is
acceptable in a lower-level GPU test.

### Mamba Tracking Validation

With `extra_buffer` enabled:

1. Issue related prompts that produce a tracked Mamba checkpoint.
2. Verify the fixed PCG tracking buffers receive the runtime mask, destination
   index, and sequence length.
3. Verify a subsequent prefix match restores a valid state from that checkpoint.
4. Run radix-tree and pool sanity checks to detect stale slots, duplicate
   ownership, or leaks.

### Regression Validation

- Non-speculative PCG continues to capture with `NULL` by default.
- `--enable-return-hidden-states` captures and replays with `FULL`.
- Target verification still returns `can_run_graph=False` for PCG.
- Draft `ModelRunner.piecewise_cuda_graph_runner` remains `None`.
- Requests larger than the maximum PCG token bucket still fall back to eager.
- Hidden states returned by PCG and eager target prefill agree numerically for
  the unpadded token range within the model's normal tolerance.

## Risks and Mitigations

### Increased Graph Memory

Capturing full hidden-state output can retain more graph memory than `NULL`.
Measure graph memory at startup before broad deployment.

### Hidden-Mode Drift

A future speculative worker could request a different hidden mode. Retaining
the strict runtime-versus-capture equality guard makes such a change fall back
to eager instead of replaying an incompatible graph.

### Accidental Speculative-Verify Capture

Selecting `FULL` based on the runner's speculative algorithm is not sufficient
to make target verification graph-compatible. The explicit
`TARGET_VERIFY` rejection remains mandatory.

### Stale Mamba Tracking State

Admitting target prefill without fixed tracking buffers can cause scheduler
metadata and actual Mamba-pool contents to diverge. The Mamba eligibility
change is therefore a correctness requirement whenever `extra_buffer` is in
use, not an optional performance extension.

## Acceptance Criteria

The implementation is complete when all of the following hold:

1. NEXTN target prefill within a captured token bucket reports a PCG hit.
2. The returned target hidden states are non-null and exclude PCG padding.
3. NEXTN draft prefill remains outside `PiecewiseCudaGraphRunner` and consumes
   the target output through the existing interface.
4. Target verification remains outside PCG.
5. Mamba tracking and subsequent prefix reuse remain correct with
   `extra_buffer` enabled.
6. Non-speculative PCG behavior is unchanged.
7. The integration test asserts an actual target-prefill PCG hit rather than
   only checking generation accuracy and speculative acceptance.

## Runtime Investigation: Inner GDN Graph with NEXTN Prefill

### Status

The target-prefill PCG implementation satisfies the static hidden-state and
Mamba-tracking contracts described above. Runtime testing on Qwen3.5 found a
separate asynchronous failure when the target model's inner GDN CUDA graph is
followed by NEXTN draft prefill. This section records the completed experiments
and the current deployment decision. It does not claim that the underlying
failure has been fixed.

### Controlled Experiments

| Configuration | Result | Interpretation |
|---|---|---|
| Target PCG + inner GDN graph, NEXTN disabled | No failure | Target prefill and the inner graph can run alone. |
| Target PCG + inner GDN graph + NEXTN | Stable failure | The NEXTN path is a required condition. |
| Same configuration with `CUDA_LAUNCH_BLOCKING=1` | No failure | Execution ordering, buffer lifetime, or delayed CUDA error reporting is involved. |
| Target PCG + NEXTN, inner GDN graph disabled | No failure | The inner GDN graph is also a required condition. |
| Target `CaptureHiddenMode.FULL`, normal draft prefill | Failure | This is the original failing configuration. |
| Target `CaptureHiddenMode.NULL` plus an independent zero hidden-state input for draft prefill | Failure | FULL hidden-state capture and direct aliasing of the target output are not necessary for the failure. |
| Target `CaptureHiddenMode.FULL`, draft prefill bypassed | No failure | Executing draft prefill after the target inner graph is another required condition. |

An external persistent buffer for the final target hidden state and temporary
CUDA graph memory-pool changes were reverted and did not produce a conclusive
fix. They are not evidence for or against the remaining hypotheses.

### Current Conclusion

The observed failure requires the following combination:

```text
target prefill inner GDN CUDA graph
  + NEXTN enabled
  + subsequent draft prefill execution
```

The experiments rule out `CaptureHiddenMode.FULL` itself as the sole cause.
They also rule out a simple alias between the target graph's final hidden-state
view and the draft input as a necessary cause. The remaining scope is the
boundary between target inner-graph execution and draft prefill, including:

- temporary-buffer or allocator lifetime across graph replay;
- shared target/draft logical KV indices and physical pools;
- metadata or cache addresses retained by graph capture;
- an earlier asynchronous out-of-bounds write that is first observed by draft
  prefill.

Using one CUDA stream does not eliminate these possibilities. Stream ordering
orders submitted CUDA work, but it does not by itself validate host-side object
lifetime, allocator reuse, graph-retained addresses, or kernel bounds.

### Diagnostic Bypass

A temporary `eagle_worker_v2.py` branch kept target
`CaptureHiddenMode.FULL`, skipped `_draft_extend_for_prefill()`, and returned a
shape-compatible zero-valued `EagleDraftInput`. The absence of the failure in
that mode established the draft-prefill condition above.

The bypass has been removed. It was never a valid production mode because it
left draft KV unwritten while PD transfer could still register draft buffers,
and it replaced speculative bootstrap probabilities, token indices, and hidden
states with zeros.

### Temporary Deployment Decision

Until the target-inner-graph/draft-prefill boundary is diagnosed, the Prefill
server will run without NEXTN and the Decode server may run with NEXTN. This
asymmetric setup requires an independent PD compatibility review. In
particular, it must not be treated as correct merely because target KV transfer
succeeds: draft KV population and speculative bootstrap metadata are separate
requirements.

### PD Compatibility Review for Decode-Only NEXTN

The selected temporary deployment intentionally uses best-effort Decode-only
NEXTN initialization:

- PD registers and transfers target KV only. Draft KV pointers and draft state
  pointers are not added to either side's transfer descriptor.
- Speculative `output_topk_p`, `output_topk_index`, and
  `output_hidden_states` buffers are omitted from the PD auxiliary descriptor;
  only ordinary request metadata and `bootstrap_room` are transferred.
- Prefill registration no longer publishes speculative configuration, and
  Decode radix cache no longer requires Prefill and Decode speculative options
  to match.
- The Decode-local target/draft allocator and page-geometry validation has been
  removed with the draft-transfer contract.
- Decode does not initialize newly allocated draft prompt pages before target
  KV transfer. This is a temporary TTFT diagnostic choice: reused pages may
  contain stale local draft KV, but target verification still controls which
  tokens are committed.
- Every reused Decode metadata slot has `output_topk_p`, `output_topk_index`,
  and `output_hidden_states` reset to zero before the request enters transfer.

The first Decode NEXTN step therefore starts from untouched local draft KV and
zero speculative bootstrap metadata when Prefill does not run NEXTN. This is
intentionally approximate and nondeterministic after page reuse: early
speculative acceptance can be low until Decode-side generation establishes
useful local draft state. Target-model verification still determines committed
output tokens, so speculative misses affect throughput rather than target-token
correctness.
