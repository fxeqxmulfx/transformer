//! One vertex represents both anchor orders. The top bits of each entry
//! store its parent, bit one stores the path parity and bit zero records
//! whether reversing the pair changes its component after flattening.
//! See OrientedAnchors.at_root.
pub(super) struct OrientedComponents {
    pub classes: Vec<u32>,
    rank: Vec<u8>,
}

impl OrientedComponents {
    const TWIST: u8 = 128;

    pub fn new(size: usize) -> Self {
        Self {
            classes: (0..size as u32).map(|u| (u << 2) | 1).collect(),
            rank: vec![0; size],
        }
    }

    /// Return twice the root index plus the path orientation.
    fn root(&mut self, mut i: usize) -> u32 {
        let mut parity = 0;
        loop {
            let first = self.classes[i];
            let next = (first >> 2) as usize;
            if next == i {
                return ((i as u32) << 1) | parity;
            }
            let second = self.classes[next];
            let flip = (first ^ second) & 2;
            self.classes[i] = (second & !3) | flip;
            parity ^= flip >> 1;
            i = (second >> 2) as usize;
        }
    }

    pub fn join(&mut self, a: usize, b: usize, reverse: bool) {
        let x = self.root(a);
        let y = self.root(b);
        let flip = (x ^ y ^ u32::from(reverse)) & 1;
        let (mut u, mut v) = ((x >> 1) as usize, (y >> 1) as usize);
        if u == v {
            if flip == 1 {
                self.rank[u] |= Self::TWIST;
            }
            return;
        }
        let ru = self.rank[u] & !Self::TWIST;
        let rv = self.rank[v] & !Self::TWIST;
        if ru < rv {
            std::mem::swap(&mut u, &mut v);
        }
        self.classes[v] = ((u as u32) << 2) | (flip << 1);
        self.rank[u] |= self.rank[v] & Self::TWIST;
        self.rank[u] += u8::from(ru == rv);
    }

    pub fn flatten(&mut self) {
        // The root search may shorten paths through previously visited nodes.
        // Mark the final orientation keys after all parent pointers are direct.
        for i in 0..self.classes.len() {
            let key = self.root(i);
            self.classes[i] = key << 1;
        }
        for entry in &mut self.classes {
            let parent = *entry >> 2;
            *entry = if self.rank[parent as usize] & Self::TWIST != 0 {
                parent << 2
            } else {
                *entry | 1
            };
        }
    }
}

pub(super) fn role_size(slots: usize) -> usize {
    slots * (slots - 1) / 2 * slots
}

/// Row for the unordered anchors. The orientation is relative to slot order.
pub(super) fn pair(a: usize, b: usize, n: usize) -> usize {
    debug_assert_ne!(a, b);
    let hi = a.max(b);
    (hi * (hi - 1) / 2 + a.min(b)) * n
}

/// Changing orientation exchanges the two keys only in an untwisted component.
pub(super) fn reorient(key: u32, swap: bool) -> u32 {
    key ^ ((u32::from(swap) & key) << 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_anchor_orders_address_each_dense_row_once() {
        for slots in [1, 2, 7, 97, 193, 256] {
            let mut start = 0;
            for high in 0..slots {
                for low in 0..high {
                    assert_eq!(pair(low, high, slots), start);
                    assert_eq!(pair(high, low, slots), start);
                    start += slots;
                }
            }
            assert_eq!(start, role_size(slots));
        }
    }

    #[test]
    fn branches_compressed_during_flatten_keep_their_orientation() {
        let n = 64;
        let order = |i: usize| (13 * i + 9) % n;
        let mut forest = OrientedComponents::new(n);
        for stride in [1, 2, 4, 8, 16, 32] {
            for start in (0..n).step_by(2 * stride) {
                let left = order(start);
                let right = order(start + stride);
                forest.join(left, right, (left ^ right) & 1 != 0);
            }
        }
        forest.flatten();
        for (i, &key) in forest.classes.iter().enumerate() {
            assert_eq!(reorient(key, i & 1 != 0), forest.classes[0]);
            assert_ne!(reorient(key, i & 1 == 0), forest.classes[0]);
        }
        forest.join(0, 1, false);
        forest.flatten();
        for key in forest.classes.iter().copied() {
            assert_eq!(reorient(key, true), forest.classes[0]);
        }
    }

    #[test]
    fn compressed_components_match_both_sheets_after_new_and_conflicting_edges() {
        for n in [2, 13, 65] {
            let mut compact = OrientedComponents::new(n);
            let mut explicit = super::super::Components::new(2 * n);
            let mut seed = 17_u32;
            let mut next = || {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                seed as usize
            };
            for k in 0..3 * n {
                let (a, b, flip) = (next() % n, next() % n, (next() >> 7) & 1);
                compact.join(a, b, flip == 1);
                for s in 0..2 {
                    explicit.join(2 * a + s, 2 * b + (s ^ flip));
                }
                if k % 4 == 0 || k + 1 == 3 * n {
                    compact.flatten();
                    explicit.flatten();
                    for i in 0..2 * n {
                        for j in 0..2 * n {
                            assert_eq!(
                                reorient(compact.classes[i / 2], (i ^ j) & 1 == 1)
                                    == compact.classes[j / 2],
                                explicit.parent[i] == explicit.parent[j]
                            );
                        }
                    }
                }
            }
        }
    }
}
