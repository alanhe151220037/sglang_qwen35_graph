# MambaRadixCache Milestone Eviction Design

## Current Rollout Status

Intermediate milestone retention is temporarily disabled. The active policy
retains Mamba states at leaves, topology branch nodes, and aligned Mamba branch
checkpoints, while an ordinary linear internal node is not milestone-retainable.
Parent cleanup, branch/leaf LRU priority, and Mamba
eviction backtracking remain enabled. The milestone predicate and segment
bookkeeping stay in the implementation so the target `8K, 24K, 56K, ...`
policy can be restored by setting `MAMBA_MILESTONE_BASE_TOKENS` to `8 * 1024`.

The previously documented split-child cleanup gap remains unchanged during this
temporary rollout.

## Background

`MambaRadixCache` currently stores full KV indices and Mamba state indices on the
same radix-tree node:

- `TreeNode.value`: full KV cache indices for the compressed key segment.
- `TreeNode.mamba_value`: one Mamba state slot for the sequence prefix ending at
  this node.
- `full_lru_list`: eviction order for full KV. Full KV eviction only selects
  unlocked leaves.
- `mamba_lru_list`: eviction order for Mamba states. Mamba eviction can either
  tombstone an internal node or delete a leaf together with its full KV.

The current behavior refreshes every non-tombstone Mamba node on the matched
path. This makes all path states look recently used, including intermediate
states that are often redundant when a path has no branch.

## Goals

1. Avoid retaining extra Mamba states on non-branching paths.
2. Retain Mamba states only at structural branch boundaries and at configured
   distance milestones on long linear paths.
3. Reverse Mamba eviction priority so states closer to leaves are evicted before
   states closer to the root.
4. Keep full KV semantics unchanged unless a Mamba leaf must be deleted as part
   of existing leaf eviction.

## Terminology

- **Topology branch node**: a non-root node with `len(node.children) > 1`.
- **Aligned Mamba branch checkpoint**: the aligned prefix selected from
  `mamba_branching_seqlen`. Its divergent suffix is not necessarily inserted, so
  this node may have only one child even though it represents a reusable branch
  boundary.
- **Path anchor**: root, a topology branch node, or an aligned Mamba branch
  checkpoint.
- **Linear segment**: a path from one path anchor to the next branch node, or
  from one path anchor to a leaf, excluding the anchor itself and including the
  downstream endpoint.
- **Segment distance of a node**: the sum of `len(n.value)` over nodes from the
  path anchor's child through this node.
- **Milestones**: cumulative lengths `8K, 8K + 16K, 8K + 16K + 32K, ...`, i.e.
`8K * (2^k - 1)` for `k >= 1`.
- **Retainable Mamba state**: a Mamba state that is allowed to remain after
  structural cleanup.

## Proposed Invariant

For every linear segment, intermediate Mamba states may exist only at the first
available node after each milestone threshold, plus topology branch nodes and
aligned Mamba branch checkpoints that serve as future anchors. Leaf nodes keep a
Mamba state while they remain leaves.

Equivalently:

- On a path from a branch/root anchor to a leaf, an intermediate non-branch node
  may retain a Mamba state only after crossing the next threshold in
      `8K, 24K, 56K, 120K, ...`. The retained node need not land on the exact threshold.
- On a path between two branch nodes, the downstream branch node may keep its
  Mamba state because it is a structural anchor for its children. Non-branch
  nodes between the two branch nodes may keep Mamba states only at milestones.
- When a leaf grows into a non-branch parent, its previous Mamba state is removed
  unless that node is milestone-retainable or has become a branch node.
- Leaf nodes must continue to have a Mamba state. A leaf must not become a Mamba
  tombstone.

This keeps the existing leaf invariant: if a leaf's Mamba state is selected for
removal, the cache should delete that leaf's full KV as well, and then keep
walking upward through Mamba-tombstone parents until it reaches a node that still
has a Mamba state or reaches a structural stop condition. In other words, Mamba
cleanup is allowed to shrink the tree, not create tombstone leaves.

## Milestone Predicate

Use a small helper:

```python
def _is_mamba_milestone(distance: int, previous_checkpoint: int) -> bool:
    next_milestone = 8 * 1024
    while next_milestone <= previous_checkpoint:
        next_milestone = next_milestone * 2 + 8 * 1024
    return distance >= next_milestone
```

The thresholds are `8K, 24K, 56K, 120K, ...` because:

```text
distance = 8 * (2^k - 1)
distance / 8 + 1 = 2^k
```

## Structural Cleanup

Add a post-insert cleanup pass scoped to the modified branch instead of scanning
the whole tree.

### Helper Functions

1. `_is_branch_node(node)`
   - Returns `len(node.children) > 1`.
   - Root is treated as an anchor even though it is not a branch by children
     count.

2. `_find_segment_anchor(node)`
   - Walks upward until root or the nearest branch node.
   - Returns the anchor.

3. `_segment_distance(anchor, node)`
   - Walks from `node` up to `anchor`, summing `len(cur.value)`.
   - Requires `anchor` to be an ancestor.

4. `_can_retain_mamba(node)`
   - Returns `False` for root.
   - Returns `True` if `node` is a topology branch node or an aligned Mamba
     branch checkpoint.
   - Otherwise computes distance to the nearest anchor and checks whether it
     has crossed the next milestone after the closest retained checkpoint.

5. `_drop_internal_mamba_state(node)`
   - Requires `node.mamba_value is not None` and `node.mamba_lock_ref == 0`.
   - Requires `len(node.children) > 0`.
   - Frees `node.mamba_value` from `req_to_token_pool.mamba_pool`.
   - Removes the node from `mamba_lru_list`.
   - Decrements `mamba_evictable_size_`.
   - Sets `node.mamba_value = None`.
   - Does not touch `node.value` or full LRU.

6. Leaf deletion on Mamba eviction
   - Reuse the existing `_evict_leaf_node()` path.
   - It deletes the leaf and then iteratively deletes tombstone parents, freeing
     their full KV and removing them from full LRU, until the parent has a Mamba
     state, is root, has children, or is locked.

### Insert Flow Changes

After `_insert_helper` finishes, run cleanup on the direct parent of the terminal
node:

1. Identify the final inserted or matched node.
2. Inspect only `terminal_node.parent`.
3. If that parent is an internal node with `mamba_value is not None`:
   - Keep it if `_can_retain_mamba(node)` is true.
   - Otherwise call `_drop_internal_mamba_state(node)`.
4. If an existing node gained a second child due to insertion, it becomes a
   branch node and its Mamba state is retainable.

The current `_insert_helper` returns only `(prefix_len, mamba_exist)`. The
implementation should return the terminal node as well, for example:

```python
return total_prefix_length, mamba_value_exist, node
```

The public `insert()` can then run cleanup and fold the cleanup result into
`InsertResult.mamba_exist`.

## Split Node Handling

`_split_node()` currently creates `new_node.mamba_value = None` because Mamba
state cannot be split. This remains correct.

After insertion, cleanup intentionally inspects only the direct parent of the
terminal node. A split parent starts without a Mamba state, while an old leaf
that becomes that direct parent is checked for milestone retention.

An aligned repair insertion can split an existing node exactly at
`mamba_branching_seqlen`. Since only KV up to that checkpoint is inserted, the
new split parent may still have one child. The request cache path marks the
terminal node with `is_mamba_branch_checkpoint`; this marker is independent of
`len(node.children)` and prevents deferred unlock cleanup from discarding the
repaired state. The marker remains after Mamba-pressure eviction so a later
recreated state at the same logical boundary is still retainable.

## Match Behavior

The matching rule should stay conservative:

- `_match_prefix_helper()` continues to choose the deepest node on the path that
  has `mamba_value`.
- If full KV extends deeper than the deepest retained Mamba state, the match is
  truncated to the Mamba-supported prefix.
- `mamba_branching_seqlen` remains the signal for recomputing Mamba state past a
  tombstone gap.

This is the intended tradeoff: full KV may still be physically present deeper in
the tree, but a prefix is only reusable up to the deepest retained Mamba state.

## Extra-Buffer Checkpoint Limitation

With `mamba_scheduler_strategy=extra_buffer`, one request exports at most one
Mamba track checkpoint in a forward pass. When a match reports
`mamba_branching_seqlen`, the scheduler gives that checkpoint priority over the
normal final chunk-aligned checkpoint so that the Mamba gap at the split point
can be repaired.

For example, if a request resumes from `state@64`, must repair a split point at
`192`, and this forward computes through `320`, the tracked state is `state@192`.
The cache insert is consequently limited to the prefix through `192`; the
already-computed `193..320` suffix remains request-local and is not inserted as
a cache leaf in that pass. This preserves the invariant that every cached leaf
has a state at its own endpoint, but may leave a long uncached tail.

This is an accepted current tradeoff. The two ping-pong slots only provide
overlap safety and do not support two checkpoints per request in one forward.
Future options are:

- split the extend at the repair checkpoint and cache the trailing leaf in a
  second forward; or
- extend the forward metadata and Mamba state scatter path to export multiple
  checkpoints per request.

Neither option is included in this change.

## Mamba LRU Direction

Current Mamba LRU refresh inserts the leaf first and then parents, making leaves
more recently used than ancestors. The required priority is structural:
root-side branch states are most recently used, followed by leaf states, then
intermediate milestone states.

The simplest implementation is to keep the list API unchanged but change Mamba
path refresh order:

- Full LRU keeps the existing leaf-to-root refresh behavior.
- Mamba LRU refresh groups matched states as: topology branch nodes and aligned
  branch checkpoints from root to leaf, then leaf nodes, then intermediate
  milestone nodes.
- With `head` as MRU and `tail.prev` as LRU, shared root-side branch states are
  newer than request-specific leaves, and milestone checkpoints are oldest.
- `evict_mamba()` can continue using `get_lru_no_lock()`, which will tend to
  choose milestone states first, then leaves, then branch states.

Concretely, add a Mamba-only helper such as:

```python
def reset_node_and_parents_mru_by_mamba_priority(self, node, root_node):
    nodes = []
    while node != root_node:
        if node.mamba_value is not None:
            nodes.append(node)
        node = node.parent
    branches = [node for node in reversed(nodes) if node.is_mamba_branch]
    leaves = [node for node in nodes if not node.is_mamba_branch and len(node.children) == 0]
    milestones = [node for node in nodes if not node.is_mamba_branch and len(node.children) == 1]
    for node in reversed(branches + leaves + milestones):
        self._remove_node(node)
        self._add_node(node)
```

This preserves the linked-list invariant while reversing relative recency along
one matched path.

## Eviction Semantics After the Change

### Mamba Pressure

`evict_mamba(mamba_num)` still evicts from `mamba_lru_list.get_lru_no_lock()`.
Because refresh order changes, this now tends to select leaf-side retained
states before branch/root-side retained states.

Internal Mamba eviction remains tombstone-only:

- free Mamba state
- remove from Mamba LRU
- keep full KV

Leaf Mamba eviction remains full-node deletion:

- free full KV
- free Mamba state
- remove from both LRUs
- delete the leaf
- walk upward and delete any newly exposed tombstone leaf parents until reaching
  a node that still has Mamba state

### Full KV Pressure

`evict_full(num_tokens)` remains leaf-only and can keep the existing assumption
that a selectable leaf has Mamba state. The cleanup path must preserve this by
deleting a leaf entirely when its Mamba state should not be retained.

## Accounting Changes

Any new Mamba drop path must update:

- `mamba_evictable_size_`
- `mamba_protected_size_` only if a protected state is ever considered for drop
  (the design should avoid dropping protected states)
- `mamba_lru_list`
- `node.mamba_value`

The cleanup pass skips a node while `mamba_lock_ref > 0`. When `dec_lock_ref()`
unlocks that node, it retries the same non-retained-state cleanup immediately.
This avoids leaving a leaf-turned-parent state indefinitely after the owning
request finishes. Full-KV deletion remains subject to `full_lock_ref`.

## Tests

Add focused unit tests under `test/registered/unit/mem_cache/test_mamba_unittest.py`
or a new Mamba radix cache test file:

1. Linear growth without milestone:
   - Insert prefixes that create a chain below one anchor.
   - Verify previous non-branch parent Mamba states are removed.

2. Milestone retention:
    - Build linear paths that first cross `8K`, `24K`, and `56K`.
   - Verify the first tracked node after each threshold retains its Mamba state
     after further growth.

3. Non-milestone leaf cleanup:
   - Insert a leaf that has not crossed the next milestone, then grow past it.
   - Verify the old leaf is either deleted as part of upward cleanup or kept only
     if it becomes branch/milestone-retainable.
   - Verify no leaf remains with `mamba_value is None`.

4. Branch retention:
   - Grow one path, then insert a second child to make an ancestor branch.
   - Verify the branch node's Mamba state is retained if present or can be
     restored on a later insert.

5. Aligned branch-checkpoint retention:
   - Split an existing path at `mamba_branching_seqlen` while caching only up to
     the aligned checkpoint, leaving the checkpoint with one child.
   - Verify final request unlock does not remove its Mamba state.

6. Mamba LRU direction:
   - Match a path with multiple retained Mamba states.
   - Evict one Mamba state.
   - Verify the leaf-side retained state is selected before the root-side state.

7. Leaf invariant:
   - After inserts, Mamba cleanup, and Mamba eviction, traverse the tree.
   - Verify every leaf has `mamba_value is not None`.

## Risks

- The existing code assumes leaves are never Mamba tombstones. The new cleanup
  must preserve this invariant by deleting leaves rather than tombstoning them.
- Fewer retained Mamba states reduce memory pressure but may reduce prefix hit
  depth. This is expected, but benchmark coverage should include long shared
  prefixes.
- HiCache Mamba support has a subclass (`HiMambaRadixCache`) with parallel logic.
  The initial implementation should either gate this behavior to
  `MambaRadixCache` only or mirror the cleanup and leaf-tombstone behavior in
   `HiMambaRadixCache`.
- **Known implementation gap -- split child cleanup:** splitting a retained
  linear internal node can leave the old child with a Mamba state even when its
  new distance from the newly-created branch node is below the next 8K
  milestone. The post-insert cleanup currently checks only the terminal
  node's direct parent, so it does not inspect that moved child. This retains
  an extra Mamba state and can stop later full-KV tombstone reclamation at the
  child. The eventual fix should check that one moved child after `_split_node`;
  it must not turn into recursive ancestor cleanup.

## Implementation Order

1. Add helper predicates and a scoped cleanup pass in `MambaRadixCache`.
2. Change `_insert_helper` to return the terminal node.
3. Run cleanup after insert and adjust `InsertResult.mamba_exist` behavior.
4. Add leaf cleanup that deletes full KV until reaching a Mamba-state boundary.
5. Add priority-ordered Mamba LRU refresh and use it only for `mamba_lru_list`.
6. Add unit tests for milestone cleanup, deferred cleanup after unlock, leaf
   invariant, and LRU direction.
