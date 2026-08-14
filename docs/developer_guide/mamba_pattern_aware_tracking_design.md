# Pattern-Aware Mamba State Tracking Design

## Status

The initial non-session implementation is complete. Native SGLang
`session_params` requests intentionally retain the existing tracking behavior.

The feature detects the last complete configured opening/closing token-pattern
pair in a tokenized generation request and uses the page-aligned position
immediately before its opening pattern as the preferred Mamba tracking point. A
radix-tree branch point remains higher priority than the pattern-derived point.

The initial target deployment uses the token patterns for this Qwen3.5 block:

```text
<reminder> ... </reminder>
```

The opening pattern is currently `[27, 75240, 29]`. The deployment must obtain
and configure the closing pattern from the exact tokenizer revision in use.
These are ordinary tokenizer tokens, not tokenizer special tokens. Therefore,
the feature must be opt-in instead of silently assigning global semantics to
these sequences for every model and request.

## Motivation

The current `extra_buffer` scheduler strategy normally tracks the last Mamba
state that can be aligned within the current extend forward. For a prompt whose
last section is a temporary `<reminder>...</reminder>` block, this places the
cached state after some or all of the temporary block.

In the next turn, the old reminder block is removed from conversation history.
The stable prefix and the new prompt then diverge near the start of the previous
reminder. With page-size alignment, the old request commonly leaves a short
leaf, for example:

```text
stable prefix -- split at 36096 -- old reminder tail [36096, 36160)
                              `-- next turn continues from 36096
```

Tracking immediately before the last reminder makes the cached terminal prefix
end on stable content. A later request can grow that path instead of repeatedly
creating a short reminder-tail branch.

## Goals

1. Find the last complete configured opening/closing token-pattern pair after
   all tokenizer-side transformations are complete.
2. Propagate its position to the scheduler without rescanning ordinary requests
   in the scheduler hot path.
3. Track the Mamba state at the last valid cache alignment before the opening
   pattern.
4. Preserve radix split-point tracking as the highest-priority requirement.
5. Preserve chunked-prefill progress before the opening pattern is reached.
6. Support both legacy `MambaRadixCache` and the unified radix tree through the
   shared scheduler tracking fields.
7. Leave behavior unchanged when the feature is disabled or a complete pair is
   not present.

## Non-Goals

- Parsing XML or validating the semantic contents of `<reminder>` blocks.
- Treating `<reminder>` as a tokenizer special token.
- Changing Mamba milestone retention or Mamba/full-KV eviction policy.
- Retroactively deleting deeper states that were inserted before this feature
  was enabled.
- Changing decode-side PD cache policy or transmitting additional cache tensors.
- Caching full KV past the selected Mamba state in `MambaRadixCache`.

## Configuration

Add an optional server argument:

```text
--mamba-track-anchor-token-pattern 27 75240 29
--mamba-track-anchor-end-token-pattern <closing-token-ids>
```

The corresponding `ServerArgs` fields are optional lists of non-negative token
ids. Both unset disables the feature. Startup validation rejects:

- configuring only one of the two patterns;
- an empty explicitly supplied pattern;
- negative token ids;
- token ids outside the model vocabulary, when the vocabulary size is known;
- use with a Mamba scheduler strategy other than `extra_buffer`, or alternatively
  log once that the option has no effect. Rejecting it is preferred because it
  makes deployment mistakes visible.

Using token ids instead of a configured string also supports
`--skip-tokenizer-init` and avoids depending on tokenizer encode options. The
deployment must verify the ids again when switching tokenizer revisions.

## Position Semantics

Let `p` be the zero-based index of the opening pattern in the last complete
matched pair inside the final scheduler-visible `input_ids`:

```text
input_ids = tokens_before + opening + reminder_body + closing + tokens_after
                            ^ p
```

The Mamba state immediately before the opening pattern has sequence length `p`,
because it represents tokens in `[0, p)`.

The actual cache target is:

```python
alignment = server_args.mamba_cache_chunk_size
anchor_seqlen = (p // alignment) * alignment
```

`mamba_cache_chunk_size` is reused instead of aligning only to `page_size`. It is
currently `max(FLA_CHUNK_SIZE, page_size)` and is the existing contract that
makes a tracked state valid for both full-KV pages and the FLA state extraction
path. Supported page sizes must continue to satisfy the existing divisibility
assumption.

Consequences:

- If `p` is aligned, `anchor_seqlen == p`.
- Otherwise, up to `alignment - 1` stable tokens before the pattern are not
  inserted into the radix tree.
- If `anchor_seqlen == 0`, there is no non-empty aligned prefix to track.

The request field should be named `mamba_track_anchor_pos` and store `p`, not the
already aligned value. Keeping the raw position makes its meaning independent of
the scheduler's page and FLA configuration.

## Tokenizer-Side Detection

Detection belongs in `TokenizerManager._create_tokenized_object()`, after:

1. text tokenization or acceptance of caller-provided `input_ids`;
2. multimodal placeholder expansion;
3. request validation and optional automatic truncation.

This is the first common point where the ids have exactly the offsets seen by
the scheduler. Scanning the original string would be incorrect for callers that
provide token ids directly and for multimodal requests that expand placeholders.

Use reverse subsequence searches to find the final closing pattern and then the
final opening pattern that ends before it. This selects the last complete block
and ignores an unmatched opening pattern at the end of the request:

```python
def find_last_enclosed_token_pattern(input_ids, opening, closing):
    closing_start = find_last_token_pattern(input_ids, closing)
    if closing_start is None:
        return None
    return find_last_token_pattern(
        input_ids,
        opening,
        search_end=closing_start,
    )
```

The configured patterns are short, so this adds two `O(n)` reverse scans with a
small constant after an already `O(n)` tokenization operation. Only generation
requests are scanned. Requests using `input_embeds` must receive `None`, because
the scheduler replaces their ids with synthetic ids and the original offset is
not meaningful.

The search is intentionally exact and syntactic:

- no complete opening/closing pair produces `None`;
- multiple complete blocks select the last block's opening pattern;
- a trailing unmatched opening pattern is ignored;
- occurrences written by the user inside ordinary content also match;
- matching remains syntactic and does not parse XML nesting or contents.

## Request Propagation

Add the optional field in the following path:

```text
TokenizerManager
  -> TokenizedGenerateReqInput.mamba_track_anchor_pos
  -> Scheduler.handle_generate_request()
  -> Req.mamba_track_anchor_pos
  -> ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend()
```

`TokenizedGenerateReqInput` is already transported as a serialized dataclass, so
the tokenizer-to-scheduler channel needs no new side message. Normal batching
also needs no separate field because `BatchTokenizedGenerateReqInput` contains
the individual tokenized objects.

Any alternate production constructor that converts a
`TokenizedGenerateReqInput` to `Req`, including the encoder-disaggregation
receiver, must copy the field. Synthetic profiling requests can leave it as
`None`.

## Scheduler Selection Policy

Extract target selection into a small pure helper so the policy can be tested
without running a model. Define:

```text
prefix_len = len(req.prefix_indices)
forward_end = prefix_len + req.extend_input_len
alignment = server_args.mamba_cache_chunk_size
```

The priority is:

```text
valid radix branch point > reminder anchor > normal aligned extend endpoint
```

The selection rules are:

1. **Branch point in this forward**
   - Accept an aligned branch point in `(prefix_len, forward_end]`, including a
     branch exactly at the forward endpoint.
   - Select `req.mamba_branching_seqlen` even if it is after the reminder anchor.
   - This exception is required to keep a reusable state at a real tree split.

2. **No pattern in the request**
   - Preserve the current normal endpoint behavior exactly.

3. **The anchor is ahead of this chunk**
   - If `forward_end < anchor_seqlen`, preserve normal endpoint tracking.
   - These intermediate states are required for chunked-prefill progress and
     request-local continuation before the reminder is reached.

4. **This forward reaches the anchor**
   - If `prefix_len < anchor_seqlen <= forward_end`, select
     `anchor_seqlen`.
   - If the anchor is internal to the forward, pass
     `_force_track_h(anchor_seqlen)` to the backend while recording the actual
     cached length as `anchor_seqlen`, matching current branch-point handling.

5. **The anchor is already at or behind the current prefix**
   - If `anchor_seqlen <= prefix_len`, do not select the normal endpoint.
   - Set the track mask to false and leave `mamba_last_track_seqlen` as `None`.
   - A later valid branch point still overrides this rule.

In pseudocode:

```python
if valid_branch_in_forward(req, prefix_len, forward_end, alignment):
    target = req.mamba_branching_seqlen
    reason = "branch"
elif req.mamba_track_anchor_pos is None:
    target = normal_target(prefix_len, req.extend_input_len, alignment)
    reason = "normal"
else:
    anchor = align_down(req.mamba_track_anchor_pos, alignment)
    if anchor <= prefix_len:
        target = None
        reason = "anchor_already_reached"
    elif anchor <= forward_end:
        target = anchor
        reason = "anchor"
    else:
        target = normal_target(prefix_len, req.extend_input_len, alignment)
        reason = "normal_before_anchor"
```

The existing code initially derives `mamba_track_mask` only from
`extend_input_len >= alignment`. With this policy, the mask must instead reflect
whether a valid target was selected. The ping-pong track index must advance only
when the mask is true.

## Chunked Prefill Example

Assume an 8K prefill chunk, a 64-token Mamba/full-KV alignment, and a last
pattern beginning at token `34177`:

```text
anchor_seqlen = floor(34177 / 64) * 64 = 34112
```

Expected behavior:

| Forward range | Selected state |
| --- | --- |
| `[0, 8192)` | normal endpoint `8192` |
| `[8192, 16384)` | normal endpoint `16384` |
| `[16384, 24576)` | normal endpoint `24576` |
| `[24576, 32768)` | normal endpoint `32768` |
| forward containing `34112` | reminder anchor `34112` |
| later forwards | none, unless a branch point must be tracked |

The full KV after `34112` remains available in the request's own allocation, so
the current prefill can continue normally. It is not inserted into the shared
radix tree and is freed when the request finishes. This distinction is essential:
the feature changes shared-cache retention, not model execution for the current
request.

## Full-KV and Mamba Coupling

Under the current Mamba radix cache contract, `req.mamba_last_track_seqlen` is
also the insertion length used for full KV. Therefore, selecting a reminder
anchor at `34112` has both effects:

- cache the Mamba state representing `[0, 34112)`;
- cache full KV only through `[0, 34112)`.

It does not leave a full-KV-only terminal suffix after the anchor. This is
consistent with the current leaf invariant and with prefix matching, which can
only reuse full KV through a retained Mamba state.

The existing milestone cleanup remains independent. The anchor node is retained
while it is a leaf. If it later becomes a linear internal node, the current
milestone/parent-cleanup policy may remove its state; if it becomes a branch,
branch-state retention preserves it.

## Existing Cache and Rollout Behavior

The new policy affects states tracked after it is enabled. It does not prune an
old tree that already contains a Mamba state deeper than the reminder anchor.
If prefix matching starts beyond `anchor_seqlen`, the scheduler cannot recreate
an earlier state without recomputing the prefix, so it suppresses new normal
tracking but leaves the old cached node intact.

For deterministic validation, start with an empty radix cache or restart the
server after enabling the option.

## Session Requests

Native SGLang `session_params` requests are intentionally excluded from the
initial implementation. Their tokenizer-derived position is relative to only
the newly supplied token ids, while `Session.create_req()` can prepend, truncate,
or replace previous session tokens. The initial implementation therefore leaves
`Req.mamba_track_anchor_pos` as `None` on this path and preserves the existing
tracking behavior.

Standard OpenAI multi-turn requests that resend the whole rendered prompt are
supported because the tokenizer sees the complete scheduler-visible sequence.

## PD Disaggregation and Unified Tree

The metadata is scheduler control data, not cache tensor data. No additional
Mamba or full-KV payload is sent between prefill and decode servers.

On a prefill server using `extra_buffer`, the shared scheduler code selects the
new tracking point before the state is inserted and transferred through the
existing path. A decode server configured not to cache Mamba states ignores the
field.

Both legacy and unified radix trees consume `req.mamba_last_track_seqlen` during
cache insertion. Keeping selection in `ScheduleBatch` means the two tree
implementations receive the same target without duplicating pattern logic.

## Observability

Add a debug-only log at selection time, guarded by the existing Mamba radix-tree
logging switch or a similarly narrow switch:

```text
rid=<rid> pattern_pos=34177 anchor=34112 prefix=32768 forward_end=40960 selected=34112 reason=anchor
```

The `reason` values should be `branch`, `anchor`, `normal`,
`normal_before_anchor`, or `anchor_already_reached`. Do not log token contents.

The existing tree dump is sufficient to verify the result: the terminal node
should end at the selected aligned anchor and report `mamba_state=yes`.

For targeted request inspection, setting
`SGLANG_MAMBA_CACHE_FILTERED_REQUEST_DUMP_PATH` to a JSONL file makes the TP0
scheduler append complete rendered text and scheduler-visible token ids for
new requests whose cache hit is exactly 34112 tokens and whose input is longer
than `8192 * 3` tokens. The filter runs after request admission so its
`cached_tokens` value matches the request-level contribution to the prefill
batch statistic. This dump is disabled when the environment variable is unset.

## Validation Plan

### Unit Tests

1. Pattern search:
   - no occurrence;
   - one occurrence;
   - multiple occurrences select the last;
   - pattern at index zero;
   - caller-provided token ids;
   - occurrence removed by automatic truncation;
   - `input_embeds` produces no anchor.

2. Scheduler target selection:
   - no configured anchor preserves the current endpoint;
   - aligned and unaligned pattern positions;
   - zero-length aligned prefix;
   - anchor ahead of the current chunk;
   - chunk containing the anchor;
   - chunk after the anchor suppresses normal tracking;
   - branch before and after the anchor both take priority;
   - track mask and ping-pong index change only for a selected target.

3. Propagation:
   - single and batch tokenized requests;
   - normal `Req` construction;
   - encoder-disaggregation request construction.

### Integration Tests

1. Send two prompts where the first contains a reminder suffix and the second
   removes that suffix while extending the preceding stable content.
2. Verify that the first tree terminal ends at the aligned position before the
   pattern rather than at the final aligned page.
3. Verify that the second request grows the stable path without creating the old
   one-page reminder-tail branch.
4. Repeat with a prompt spanning multiple chunked-prefill batches and verify
   normal progress before the anchor and no new normal state after it.
5. Repeat with a real split and verify the split point overrides the reminder
   anchor.
6. Run the same cases with unified radix tree disabled and enabled.
7. Run a PD prefill/decode smoke test and verify that transfer sizes and decode
   behavior remain valid.

## Expected Code Changes

The core implementation is localized and requires no CUDA or model-layer change:

- `server_args.py`: optional pattern configuration and validation;
- `io_struct.py`: tokenized-request metadata field;
- `tokenizer_manager.py`: reverse pattern search after final token processing;
- `schedule_batch.py`: `Req` field and target-selection policy;
- `scheduler.py`: normal request propagation;
- `encode_receiver.py`: alternate request-construction propagation;
- focused tokenizer, scheduler-policy, and Mamba radix tests.

The non-session path is a small change. Correct chunked-prefill behavior and
tests make the overall implementation medium-sized, but the change does not
require backend kernel work or a cache-tree format change.

## Risks

1. **False semantic match**: the three ordinary tokens can occur in user content.
   Opt-in configuration limits the blast radius, but the search remains
   syntactic.
2. **Reduced reusable prefix**: alignment can discard up to one alignment unit
   of stable full KV before the marker.
3. **Incorrect chunk handling**: suppressing normal tracking before the anchor
   would break useful chunked-prefill progress; the phased policy above is
   required.
4. **Stale cache during rollout**: existing deeper states are not automatically
   removed, so clean-cache validation is necessary.
5. **Native session scope**: native `session_params` requests intentionally use
   the existing tracking behavior until offset translation is implemented.
6. **Configuration drift**: tokenizer revisions may assign different ids to the
   same text, so deployment should verify the configured pattern.
