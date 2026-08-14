"""Small full-KV-only radix cache for PD decode transfer deduplication."""

from __future__ import annotations

import time

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


class PDDecodeRadixCache(RadixCache):
    """Bounded cache owning shared target/draft KV indices, never Mamba states."""

    MAX_WIDTH = 4

    @staticmethod
    def _is_split_anchor(node: TreeNode) -> bool:
        return bool(getattr(node, "_pd_decode_split_anchor", False))

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key, _ = params.key.maybe_to_bigram_view(self.is_eagle)
        key = key.page_aligned(self.page_size)
        if self.disable or len(key) == 0:
            return self._empty_match_result

        child = self.root_node.children.get(key.child_key(self.page_size))
        if child is None:
            return self._empty_match_result

        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            if child.lock_ref > 0 or prefix_len == 0:
                return self._empty_match_result
            child = self._split_root_child(child, prefix_len)

        child.last_access_time = time.monotonic()
        return MatchResult(child.value, child, child, child)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key, value = params.key.maybe_to_bigram_view(self.is_eagle, params.value)
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return InsertResult(prefix_len=0)
        value = value[: len(key)].to(dtype=torch.int64, copy=True)

        child_key = key.child_key(self.page_size)
        child = self.root_node.children.get(child_key)
        if child is not None:
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                if child.lock_ref > 0 or prefix_len == 0:
                    return InsertResult(prefix_len=0)
                self._split_root_child(child, prefix_len)
                return InsertResult(prefix_len=prefix_len)
            # Depth is bounded to one: keep the existing prefix, never append.
            return InsertResult(prefix_len=len(child.key))

        if len(self.root_node.children) >= self.MAX_WIDTH:
            replacement = self._replacement_candidate()
            if replacement is None:
                return InsertResult(prefix_len=0)
            self.token_to_kv_pool_allocator.free(replacement.value)
            self._delete_leaf(replacement)

        node = TreeNode(priority=params.priority or 0)
        node.parent = self.root_node
        node.key = key
        node.value = value
        node.last_access_time = time.monotonic()
        self.root_node.children[child_key] = node
        self.evictable_size_ += len(value)
        self._update_leaf_status(node)
        self._record_store_event(node)
        return InsertResult(prefix_len=0)

    def cache_finished_req(self, req, is_insert: bool = True):
        """Release an uncached tail when the depth/width limit rejects it."""
        kv_committed_len = req.pop_committed_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        if self.disable:
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle).page_aligned(
            self.page_size
        )
        key_len = len(key)
        result = InsertResult(prefix_len=0)
        cached_full = False
        if is_insert:
            result = self.insert(
                InsertParams(
                    key=key,
                    value=kv_indices[:key_len].to(dtype=torch.int64, copy=True),
                    priority=getattr(req, "priority", 0) or 0,
                )
            )
            cached_full = (
                len(self.match_prefix(MatchPrefixParams(key=key)).device_indices)
                == key_len
            )

        if is_insert and cached_full:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req, chunked=False):
        """Cache only when the whole page-aligned request fits in the shallow tree."""
        if self.disable:
            return
        key = RadixKey(req.fill_ids, req.extra_key, is_bigram=self.is_eagle).page_aligned(
            self.page_size
        )
        if len(key) == 0:
            return
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(key)
        ]
        result = self.insert(
            InsertParams(
                key=key,
                value=kv_indices.to(dtype=torch.int64, copy=True),
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        match_result = self.match_prefix(MatchPrefixParams(key=key))
        new_indices = match_result.device_indices
        if len(new_indices) != len(key):
            return

        # ``insert`` above already transferred ownership of newly admitted KV
        # pages to this tree. Calling RadixCache.cache_unfinished_req() would
        # insert a second time and free those same pages as duplicate prefix.
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : result.prefix_len]
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(match_result.last_device_node)
        req.cache_protected_len = len(new_indices)
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices[len(new_indices) :]]
        )
        req.last_node = match_result.last_device_node

    def _split_root_child(self, child: TreeNode, split_len: int) -> TreeNode:
        assert child.parent is self.root_node and child.lock_ref == 0
        parent = TreeNode(priority=child.priority)
        parent.parent = self.root_node
        parent.key = child.key[:split_len]
        parent.value = child.value[:split_len].clone()
        parent.hit_count = child.hit_count
        parent._pd_decode_split_anchor = True

        self.token_to_kv_pool_allocator.free(child.value[split_len:])
        self.evictable_size_ -= len(child.value) - split_len
        self.root_node.children.pop(child.key.child_key(self.page_size))
        if child in self.evictable_leaves:
            self.evictable_leaves.remove(child)
        self.root_node.children[parent.key.child_key(self.page_size)] = parent
        self._update_leaf_status(parent)
        return parent

    def _replacement_candidate(self) -> TreeNode | None:
        candidates = [
            node
            for node in self.root_node.children.values()
            if node.lock_ref == 0 and not self._is_split_anchor(node)
        ]
        if not candidates:
            return None
        return min(candidates, key=self.eviction_strategy.get_priority)
