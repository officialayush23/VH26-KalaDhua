//! A slab-backed intrusive doubly linked list.
//!
//! Every recency-ordered policy in this crate needs three operations in constant time:
//! move a node to the front, unlink an arbitrary node, and pop the back. `std`'s
//! `LinkedList` gives none of those by handle, and a `BTreeMap` keyed on an access counter
//! turns each of them into a logarithmic operation — which shows up as a real slowdown
//! once a benchmark replays ten million requests.
//!
//! Nodes live in a `Vec` and are addressed by index, so there are no pointers and no
//! `unsafe`. A policy stores the slot index next to its own entry and hands it back here.

use crate::types::KeyId;

const NIL: usize = usize::MAX;

#[derive(Debug, Clone, Copy)]
struct Node {
    prev: usize,
    next: usize,
    key: KeyId,
    live: bool,
}

/// Doubly linked list of keys with O(1) unlink by slot.
///
/// Front is the most recently used end by convention; policies that want FIFO order simply
/// never call [`IntrusiveList::move_to_front`].
#[derive(Debug, Clone, Default)]
pub struct IntrusiveList {
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    len: usize,
}

impl IntrusiveList {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), free: Vec::new(), head: NIL, tail: NIL, len: 0 }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slots currently allocated, live or free. Reported as part of the policy's memory
    /// overhead so the benchmark can charge every policy for what it actually costs.
    pub fn slots(&self) -> usize {
        self.nodes.len()
    }

    pub fn key_at(&self, slot: usize) -> Option<KeyId> {
        self.nodes.get(slot).filter(|n| n.live).map(|n| n.key)
    }

    pub fn front(&self) -> Option<KeyId> {
        self.key_at(self.head)
    }

    pub fn back(&self) -> Option<KeyId> {
        self.key_at(self.tail)
    }

    pub fn back_slot(&self) -> Option<usize> {
        if self.tail == NIL {
            None
        } else {
            Some(self.tail)
        }
    }

    fn alloc(&mut self, key: KeyId) -> usize {
        match self.free.pop() {
            Some(slot) => {
                self.nodes[slot] = Node { prev: NIL, next: NIL, key, live: true };
                slot
            }
            None => {
                self.nodes.push(Node { prev: NIL, next: NIL, key, live: true });
                self.nodes.len() - 1
            }
        }
    }

    /// Insert `key` at the front and return its slot.
    pub fn push_front(&mut self, key: KeyId) -> usize {
        let slot = self.alloc(key);
        self.link_front(slot);
        self.len += 1;
        slot
    }

    /// Insert `key` at the back and return its slot.
    pub fn push_back(&mut self, key: KeyId) -> usize {
        let slot = self.alloc(key);
        self.link_back(slot);
        self.len += 1;
        slot
    }

    fn link_front(&mut self, slot: usize) {
        let old_head = self.head;
        self.nodes[slot].prev = NIL;
        self.nodes[slot].next = old_head;
        if old_head != NIL {
            self.nodes[old_head].prev = slot;
        } else {
            self.tail = slot;
        }
        self.head = slot;
    }

    fn link_back(&mut self, slot: usize) {
        let old_tail = self.tail;
        self.nodes[slot].next = NIL;
        self.nodes[slot].prev = old_tail;
        if old_tail != NIL {
            self.nodes[old_tail].next = slot;
        } else {
            self.head = slot;
        }
        self.tail = slot;
    }

    fn unlink(&mut self, slot: usize) {
        let (prev, next) = (self.nodes[slot].prev, self.nodes[slot].next);
        if prev != NIL {
            self.nodes[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.nodes[next].prev = prev;
        } else {
            self.tail = prev;
        }
        self.nodes[slot].prev = NIL;
        self.nodes[slot].next = NIL;
    }

    /// Move an existing slot to the front. This is the LRU hit path.
    pub fn move_to_front(&mut self, slot: usize) {
        debug_assert!(self.nodes[slot].live, "moving a freed slot");
        if self.head == slot {
            return;
        }
        self.unlink(slot);
        self.link_front(slot);
    }

    pub fn move_to_back(&mut self, slot: usize) {
        debug_assert!(self.nodes[slot].live, "moving a freed slot");
        if self.tail == slot {
            return;
        }
        self.unlink(slot);
        self.link_back(slot);
    }

    /// Remove a slot and return the key it held.
    pub fn remove(&mut self, slot: usize) -> Option<KeyId> {
        if slot >= self.nodes.len() || !self.nodes[slot].live {
            return None;
        }
        self.unlink(slot);
        self.nodes[slot].live = false;
        self.free.push(slot);
        self.len -= 1;
        Some(self.nodes[slot].key)
    }

    /// Remove and return the least recently used key.
    pub fn pop_back(&mut self) -> Option<KeyId> {
        let slot = self.back_slot()?;
        self.remove(slot)
    }

    pub fn pop_front(&mut self) -> Option<KeyId> {
        if self.head == NIL {
            return None;
        }
        self.remove(self.head)
    }

    /// Walk from the least recently used end. Used by policies that scan for a victim
    /// rather than taking the tail unconditionally (SIEVE, and the sampled candidate
    /// generator).
    pub fn iter_from_back(&self) -> BackIter<'_> {
        BackIter { list: self, cursor: self.tail }
    }

    /// Slot immediately before `slot` walking towards the front, i.e. the next candidate
    /// when scanning from the back.
    pub fn prev_of(&self, slot: usize) -> Option<usize> {
        let p = self.nodes.get(slot)?.prev;
        if p == NIL {
            None
        } else {
            Some(p)
        }
    }

    pub fn next_of(&self, slot: usize) -> Option<usize> {
        let n = self.nodes.get(slot)?.next;
        if n == NIL {
            None
        } else {
            Some(n)
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.head = NIL;
        self.tail = NIL;
        self.len = 0;
    }

    pub fn memory_bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node>()
            + self.free.capacity() * std::mem::size_of::<usize>()
    }
}

#[derive(Debug)]
pub struct BackIter<'a> {
    list: &'a IntrusiveList,
    cursor: usize,
}

impl Iterator for BackIter<'_> {
    type Item = (usize, KeyId);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor == NIL {
            return None;
        }
        let slot = self.cursor;
        let node = self.list.nodes[slot];
        self.cursor = node.prev;
        Some((slot, node.key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys_front_to_back(list: &IntrusiveList) -> Vec<KeyId> {
        let mut out: Vec<KeyId> = list.iter_from_back().map(|(_, k)| k).collect();
        out.reverse();
        out
    }

    #[test]
    fn push_and_pop_preserve_order() {
        let mut l = IntrusiveList::new();
        let slots: Vec<usize> = (0..5).map(|k| l.push_front(k)).collect();
        assert_eq!(l.len(), 5);
        assert_eq!(keys_front_to_back(&l), vec![4, 3, 2, 1, 0]);
        assert_eq!(l.pop_back(), Some(0));
        assert_eq!(l.pop_front(), Some(4));
        assert_eq!(keys_front_to_back(&l), vec![3, 2, 1]);
        assert_eq!(slots.len(), 5);
    }

    #[test]
    fn move_to_front_reorders_without_reallocating() {
        let mut l = IntrusiveList::new();
        let a = l.push_front(1);
        let _b = l.push_front(2);
        let _c = l.push_front(3);
        let slots_before = l.slots();
        l.move_to_front(a);
        assert_eq!(keys_front_to_back(&l), vec![1, 3, 2]);
        assert_eq!(l.slots(), slots_before, "reordering must not allocate a new slot");
        assert_eq!(l.front(), Some(1));
        assert_eq!(l.back(), Some(2));
    }

    #[test]
    fn removal_frees_slots_for_reuse() {
        let mut l = IntrusiveList::new();
        let a = l.push_front(1);
        l.push_front(2);
        assert_eq!(l.remove(a), Some(1));
        assert_eq!(l.remove(a), None, "removing twice must be a no-op");
        let slots_before = l.slots();
        let c = l.push_front(3);
        assert_eq!(c, a, "the freed slot should be reused");
        assert_eq!(l.slots(), slots_before);
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn single_element_edges() {
        let mut l = IntrusiveList::new();
        let a = l.push_back(9);
        assert_eq!(l.front(), Some(9));
        assert_eq!(l.back(), Some(9));
        l.move_to_front(a);
        assert_eq!(l.len(), 1);
        assert_eq!(l.pop_back(), Some(9));
        assert!(l.is_empty());
        assert_eq!(l.pop_back(), None);
        assert_eq!(l.front(), None);
    }

    #[test]
    fn survives_a_long_random_workload() {
        // The invariant that matters: after any sequence of operations the list is still a
        // consistent chain of exactly `len` live nodes.
        let mut rng = crate::rng::Rng::seed_from_u64(1);
        let mut l = IntrusiveList::new();
        let mut slots: Vec<usize> = Vec::new();
        for i in 0..20_000u64 {
            match rng.below(3) {
                0 => slots.push(l.push_front(i)),
                1 => {
                    if !slots.is_empty() {
                        let idx = rng.below(slots.len() as u64) as usize;
                        l.move_to_front(slots[idx]);
                    }
                }
                _ => {
                    if !slots.is_empty() {
                        let idx = rng.below(slots.len() as u64) as usize;
                        let slot = slots.swap_remove(idx);
                        l.remove(slot);
                    }
                }
            }
        }
        assert_eq!(l.iter_from_back().count(), l.len());
        assert_eq!(l.len(), slots.len());
    }
}
