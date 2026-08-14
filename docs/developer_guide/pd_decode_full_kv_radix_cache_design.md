# PD Decode Full-KV-Only Radix Cache Design

## Status

Proposed and implemented for decode-side PD disaggregation when
`--disaggregation-decode-enable-radix-cache` is enabled.

## Goal

Allow a decode server for a hybrid Mamba/SSM model to reuse locally cached full
KV prefixes only. The reuse reduces prefill-to-decode KV transfer to the delta
suffix, while Mamba state remains request-local data transferred from prefill.

This is deliberately not a Mamba prefix cache. Decode does not COW a Mamba
state from the radix tree, does not retain Mamba states in the tree, and does
not use the Mamba extra-buffer tracking path.

## Existing Limitation

The scheduler currently rejects decode-side radix cache for hybrid SSM models.
The normal hybrid cache is `MambaRadixCache`, whose prefix validity requires a
Mamba state at the cached endpoint. Its no-buffer path consequently requires
`page_size == 1`.

For PD decode, that requirement is unnecessary: the decode cache is used only
to identify full-KV pages that do not need transfer. Prefill still sends the
final Mamba state for every admitted request, so decode never resumes a Mamba
recurrence from a locally cached prefix.

## Cache Type and State Ownership

Introduce a decode-only full-KV radix cache derived from `RadixCache`.

- It stores only full-KV indices and uses normal page-aligned matching.
- `supports_mamba()` remains `False`.
- Decode prefix matching therefore calls `cow_mamba=False`.
- `HybridReqToTokenPool.alloc()` still allocates one request-local Mamba slot
  for a hybrid model. This slot is the destination in PD transfer metadata.
- Prefill transfers its final Mamba state into that slot on the final transfer
  chunk. No tree node owns a Mamba pool slot.

The cache may use `page_size > 1`; its endpoints need only be page-aligned
because no Mamba state is associated with an endpoint.

## NEXTN Draft/Target Shared Page Indices

When decode enables `NEXTN`, the target model and draft model own distinct
physical KV buffers but use the same logical page-index space. They must share
the target model's token-to-KV allocator and must have identical `page_size`
and token capacity.

The decode radix tree stores one logical page index per cached position. That
single index owns both physical resources at the same position:

```text
logical page index i
  -> target full-attention KV page i
  -> NEXTN draft KV page i
```

Consequently:

- Allocating an index reserves both the target and draft locations.
- A radix-node lock prevents the shared index from being reused, preserving
  both target and draft KV contents.
- Eviction or request-tail release returns the shared index to the allocator
  exactly once. It must not separately free a draft index.
- Splitting or replacing a decode radix node applies the same ownership rule:
  freeing the removed logical suffix makes both physical suffixes reusable.

Decode sends one delta page-index list to prefill. The transfer manager applies
that same list to the registered target and draft KV buffer pointers, so a
decode radix hit skips the same prefix range for both models. The prefill and
decode servers must enable the same NEXTN configuration; prefill also supplies
the EAGLE top-k probabilities, token indices, and hidden state used to seed
decode-side draft execution. Prefill publishes its speculative algorithm and
shape-affecting NEXTN settings through the existing bootstrap handshake; decode
rejects the connection before accepting requests when they do not match.

This does not change Mamba ownership. The decode tree still never stores a
Mamba state. Prefill transfers the final target-model Mamba state into the
request-local decode Mamba slot independently of the shared target/draft KV
page indices.

### EAGLE Bigram Alignment

`NEXTN` resolves to EAGLE and therefore uses radix bigram keys. A raw sequence
of `N` tokens contains at most `N - 1` cacheable bigrams. The decode cache then
rounds that logical length down to `page_size`, so the reusable prefix is:

```text
floor((N - 1) / page_size) * page_size
```

This is intentional and preserves the boundary token required by EAGLE. It
also means that an 8192-token prompt with `page_size=64` has a maximum cached
prefix of 8128 logical KV positions.

## Bounded Tree Shape

The decode cache is a small transfer-deduplication index, not a general prefix
cache. Its tree shape is bounded as follows:

- Maximum depth is one: root plus at most one cached child edge.
- Maximum width is four: root has at most four children.

### Insert Rules

1. **Exact prefix hit**: reuse the existing root child. Do not append a deeper
   suffix node; KV beyond that child remains request-local and is transferred
   as delta on future requests.
2. **Partial match of a root child**: split the existing child at the matched,
   page-aligned prefix. Keep only the new split parent and its full KV prefix.
   Free the old child suffix KV and do not insert either the old or incoming
   suffix below the parent. The resulting split parent remains a depth-one
   root child.
3. **New root child with width below four**: insert it normally as one root
   child.
4. **New root child at width four**: replace only an unlocked child that has
   never been split. Choose the normal full-KV eviction candidate among those
   eligible children. If no such child exists, skip insertion and release the
   request's non-protected KV as in a cache miss.

A split parent is marked non-replaceable even though it is a leaf after its
suffix children are removed. This preserves shared-prefix anchors instead of
replacing them as ordinary one-request entries.

### Locked Nodes

The cache must not free or rewrite a locked root child. If a partial match would
split a locked child, it is treated as a cache miss for admission/transfer and
the cache is not structurally modified. If width replacement finds only locked
or split children, insertion is skipped.

## PD Transfer Flow

```text
decode radix match (full KV only)
  -> page-aligned prefix_len
  -> decode sends destination KV indices only for [prefix_len, prompt_len)
  -> prefill computes prompt and sends delta KV pages
  -> prefill sends final request Mamba state
  -> decode receives final Mamba state into req.mamba_pool_idx
```

The cache changes only the transferred full-KV range. It does not change Mamba
state transfer, ownership, or decode execution semantics.

## Eviction

This cache uses `RadixCache` full-KV eviction only. There is no Mamba LRU,
Mamba tombstone, Mamba state eviction, or Mamba backtracking path.

Because all cached children are depth-one leaves, normal leaf eviction removes
one complete cache entry. Split parents are still evictable under memory
pressure; the non-replaceable rule applies only to width admission, not memory
eviction.

## Compatibility

- Applies only to `disaggregation_mode=decode` with
  `disaggregation_decode_enable_radix_cache=True`.
- Replaces the hybrid-SWA/SSM incompatibility check for hybrid SSM only.
- Hybrid SWA remains unsupported because its specialized full/SWA allocator and
  transfer semantics are not covered by this design.
- Ordinary single-layer `NEXTN`/EAGLE is supported when target and draft share
  one allocator and have identical page geometry. EAGLE3, multi-layer EAGLE,
  and other speculative algorithms remain unsupported until their ownership
  and transfer layouts are validated independently.
- Prefill and decode must use matching `page_size`, speculative algorithm,
  draft-token count, EAGLE top-k, speculative-step settings, and attention
  TP/CP/PP topology. Heterogeneous P/D topology remains available without
  speculative decoding but is outside the initial NEXTN support scope.
- The bounded decode cache bypasses `MambaRadixCache`, `HiMambaRadixCache`, and
  unified Mamba components, so the Mamba cache policy is unchanged for normal
  aggregated serving and prefill servers.

### Deployment Requirements

Both sides enable the same NEXTN settings and page size. For a hybrid
Mamba/full-attention model, prefill retains `extra_buffer`, while decode is
automatically changed to `no_buffer` because it caches no Mamba state:

```text
prefill:
  --disaggregation-mode prefill
  --speculative-algorithm NEXTN
  --mamba-scheduler-strategy extra_buffer
  --page-size 64

decode:
  --disaggregation-mode decode
  --disaggregation-decode-enable-radix-cache
  --speculative-algorithm NEXTN
  --page-size 64
```

The values of `--speculative-num-steps`, `--speculative-eagle-topk`, and
`--speculative-num-draft-tokens` must also match. Spec-v2 is enabled by default;
deployments that override `SGLANG_ENABLE_SPEC_V2` must keep it consistent on
both sides.

## Tests

1. Hybrid SSM decode construction selects the full-KV-only cache and permits
   `page_size > 1`.
2. Decode match never allocates or copies a Mamba cache state from the tree.
3. A partial root-child match leaves only the split prefix at depth one.
4. A fifth distinct root child replaces an unlocked unsplit child.
5. A fifth distinct root child is not inserted when all four children are split
   anchors or locked.
6. Cached prefix pages are omitted from decode-to-prefill KV transfer while the
   Mamba transfer metadata remains present.
7. NEXTN target and draft pools are rejected at startup unless they share the
   same allocator, page size, and token capacity.
8. An EAGLE bigram prefix hit omits the same page range from target and draft
   transfer, while the unaligned boundary and speculative tail remain
   request-owned.
9. Speculative accept, reject, finish, and retract paths return every shared
   page index exactly once and do not leak allocator capacity.
