//! The aggregate a hull node carries, and how a tie is resolved.
//!
//! Ported from `HullMeta` in `attention/hull2d_cht.h`.  Two things about it are
//! worth stating, because both are load-bearing in the audit (`todo3.md`):
//!
//!   * `Latest` is resolved by `last_seq`, an integer sequence number held
//!     beside the payload.  That is outside the model, and it is the right
//!     design: it is exactly `Meta.resolveLatest`, which `ALM.SoftmaxLatestMass`
//!     proves the attention head against.
//!   * `Average` divides by the count, and `vsum * (1.0 / count)` is not exact.
//!     The cumulative sums built on it (instruction pointer, stack depth, call
//!     depth) come back off the integer grid for about a quarter of the inputs,
//!     and the lookups downstream survive on a margin of one ulp (`todo3.md` §8).

/// How a head resolves several keys that score the same.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TieBreak {
    /// Return the mean of the tied payloads.  This is also how prefix sums are
    /// built: every position writes the same key, so every position ties.
    Average,
    /// Return the payload written latest, by sequence number.
    Latest,
}

/// Sum, count and latest payload for the entries under one hull node.
#[derive(Clone, Copy, Debug)]
pub struct HullMeta {
    pub vsum: [f64; 2],
    pub vlast: [f64; 2],
    pub count: i32,
    pub last_seq: i32,
}

impl Default for HullMeta {
    fn default() -> Self {
        HullMeta { vsum: [0.0; 2], vlast: [0.0; 2], count: 0, last_seq: -1 }
    }
}

impl HullMeta {
    /// The aggregate of a single entry.
    pub fn of(val: [f64; 2], seq: i32) -> Self {
        let mut m = HullMeta::default();
        m.add(val, seq);
        m
    }

    /// Fold one entry in.
    pub fn add(&mut self, val: [f64; 2], seq: i32) {
        self.vsum[0] += val[0];
        self.vsum[1] += val[1];
        self.count += 1;
        if seq > self.last_seq {
            self.last_seq = seq;
            self.vlast = val;
        }
    }

    /// Fold another aggregate in.
    pub fn merge(&mut self, other: &HullMeta) {
        self.vsum[0] += other.vsum[0];
        self.vsum[1] += other.vsum[1];
        self.count += other.count;
        if other.last_seq > self.last_seq {
            self.last_seq = other.last_seq;
            self.vlast = other.vlast;
        }
    }

    /// The head's answer for this aggregate.
    ///
    /// An empty aggregate answers zero, as the C++ does; the caller is expected
    /// to have checked emptiness itself.
    pub fn resolve(&self, tb: TieBreak) -> [f64; 2] {
        if self.count == 0 {
            return [0.0, 0.0];
        }
        match tb {
            TieBreak::Latest => self.vlast,
            TieBreak::Average => {
                let inv = 1.0 / f64::from(self.count);
                [self.vsum[0] * inv, self.vsum[1] * inv]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_is_by_sequence_number_not_by_insertion_order() {
        let mut m = HullMeta::default();
        m.add([7.0, 0.0], 5);
        m.add([3.0, 0.0], 2);
        assert_eq!(m.resolve(TieBreak::Latest), [7.0, 0.0]);
        assert_eq!(m.resolve(TieBreak::Average), [5.0, 0.0]);
    }

    #[test]
    fn merge_agrees_with_adding_one_at_a_time() {
        let mut a = HullMeta::of([1.0, 2.0], 0);
        a.add([3.0, 4.0], 1);
        let mut b = HullMeta::of([1.0, 2.0], 0);
        b.merge(&HullMeta::of([3.0, 4.0], 1));
        assert_eq!(a.vsum, b.vsum);
        assert_eq!(a.count, b.count);
        assert_eq!(a.vlast, b.vlast);
    }

    #[test]
    fn the_average_leaves_the_integer_grid() {
        // todo3.md section 8, measured: `fl(fl(s * fl(1/p)) * p) == s` fails for
        // about a quarter of the payloads.  The smallest witness is s = 3 over
        // p = 5, and it is reached through this very function.
        let mut m = HullMeta::default();
        for _ in 0..2 {
            m.add([0.0, 0.0], 0);
        }
        for _ in 0..3 {
            m.add([1.0, 0.0], 0);
        }
        let avg = m.resolve(TieBreak::Average)[0];
        assert_ne!(avg * 5.0, 3.0);
        assert_eq!(avg * 5.0, 3.0000000000000004);
    }
}
