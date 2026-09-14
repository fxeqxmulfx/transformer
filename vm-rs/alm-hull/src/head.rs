//! The two hard-attention heads: the hull one, and the brute-force reference.
//!
//! Ported from `HullHalf`, `HardAttentionHead` and `BruteAttentionHead` in
//! `attention/hull2d_cht.h`.  The pair exists to be differentially tested
//! against each other: the C++ release disagrees with its own reference on ties
//! (`todo3.md` section 3), and reproducing both is how that is measured here.

use core::cell::Cell;

use crate::breakpoint::Break;
use crate::gap::ScoreGaps;
use crate::cht::{Cht, Slope};
use crate::grid::GridWitness;
use crate::meta::{HullMeta, TieBreak};

/// One envelope: the upper hull maximises `kx * m + ky`, the lower minimises it
/// by maximising the negated line.
pub struct HullHalf {
    cht: Cht,
    is_upper: bool,
}

/// What a query found: the resolved payload, the winning score and key.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub out: [f64; 2],
    pub score: f64,
    pub best_kx: f64,
    /// The winning key, as the head stored it.
    pub best_key: [f64; 2],
}

impl HullHalf {
    pub fn new(is_upper: bool) -> HullHalf {
        HullHalf { cht: Cht::new(), is_upper }
    }

    pub fn len(&self) -> usize {
        self.cht.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cht.is_empty()
    }

    pub fn clear(&mut self) {
        self.cht.clear();
    }

    pub fn insert(&mut self, kx: f64, ky: f64, val: [f64; 2], seq: i32) {
        let meta = HullMeta::of(val, seq);
        if self.is_upper {
            self.cht.add_line(kx, ky, meta);
        } else {
            self.cht.add_line(-kx, -ky, meta);
        }
    }

    /// The stored key of the line at `s`, undoing the negation of a lower hull.
    fn key_at(&self, s: Slope) -> (f64, f64) {
        let b = self.cht.get(s).unwrap().b;
        if self.is_upper {
            (s.get(), b)
        } else {
            (-s.get(), -b)
        }
    }

    pub fn query(&self, qx: f64, qy: f64, tb: TieBreak) -> Option<Hit> {
        if self.cht.is_empty() {
            return None;
        }

        let at = if qy == 0.0 {
            if qx >= 0.0 {
                Break::PosInf
            } else {
                Break::NegInf
            }
        } else {
            Break::ratio(qx, qy)
        };
        let best = self.cht.argmax(at)?;

        let (kx_best, ky_best) = self.key_at(best);
        let best_score = qx * kx_best + qy * ky_best;

        // A query with `qy == 0` reads one extreme of the envelope and stops:
        // the ties there are already collapsed into that node's aggregate.
        if qy == 0.0 {
            let out = self.cht.get(best).unwrap().meta.resolve(tb);
            return Some(Hit { out, score: best_score, best_kx: kx_best, best_key: [kx_best, ky_best] });
        }

        let mut combined = HullMeta::default();
        combined.merge(&self.cht.get(best).unwrap().meta);

        let mut left = best;
        while let Some(prev) = self.cht.pred(left) {
            let (kx, ky) = self.key_at(prev);
            if qx * kx + qy * ky == best_score {
                combined.merge(&self.cht.get(prev).unwrap().meta);
                left = prev;
            } else {
                break;
            }
        }

        let mut right = self.cht.succ(best);
        while let Some(next) = right {
            let (kx, ky) = self.key_at(next);
            if qx * kx + qy * ky == best_score {
                combined.merge(&self.cht.get(next).unwrap().meta);
                right = self.cht.succ(next);
            } else {
                break;
            }
        }

        Some(Hit { out: combined.resolve(tb), score: best_score, best_kx: kx_best, best_key: [kx_best, ky_best] })
    }
}

/// The `O(log n)` hard-attention head: two envelopes plus the degenerate cases.
pub struct HardAttentionHead {
    upper: HullHalf,
    lower: HullHalf,
    global: HullMeta,
    left_meta: HullMeta,
    right_meta: HullMeta,
    min_kx: f64,
    max_kx: f64,
    n: usize,
    /// Written from `query`, which takes `&self`: the witness is an
    /// observation of the head, not part of its answer.
    grid: Cell<GridWitness>,
}

impl Default for HardAttentionHead {
    fn default() -> Self {
        HardAttentionHead {
            upper: HullHalf::new(true),
            lower: HullHalf::new(false),
            global: HullMeta::default(),
            left_meta: HullMeta::default(),
            right_meta: HullMeta::default(),
            min_kx: f64::INFINITY,
            max_kx: f64::NEG_INFINITY,
            n: 0,
            grid: Cell::new(GridWitness::default()),
        }
    }
}

impl HardAttentionHead {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn insert(&mut self, key: [f64; 2], val: [f64; 2], seq: i32) {
        self.global.add(val, seq);

        if key[0] < self.min_kx {
            self.min_kx = key[0];
            self.left_meta = HullMeta::default();
        }
        if key[0] == self.min_kx {
            self.left_meta.add(val, seq);
        }

        if key[0] > self.max_kx {
            self.max_kx = key[0];
            self.right_meta = HullMeta::default();
        }
        if key[0] == self.max_kx {
            self.right_meta.add(val, seq);
        }

        self.upper.insert(key[0], key[1], val, seq);
        self.lower.insert(key[0], key[1], val, seq);
        self.n += 1;
    }

    /// What this head has answered that float64 could not separate — empty
    /// unless a *winning score* has crossed `2^53`.  See `grid`.
    pub fn grid_witness(&self) -> GridWitness {
        self.grid.get()
    }

    /// The margin a winner must beat: for the parabolic keys of this machine
    /// the runner-up's true score is at least one integer step away, and one
    /// step in the score is `|qy|`.  The `qy == 0` shortcut compares along the
    /// other axis, where a step is `|qx|`.
    fn note(&self, score: f64, margin: f64, query: [f64; 2], key: Option<[f64; 2]>) {
        let mut w = self.grid.get();
        w.observe(score, margin, query, key);
        self.grid.set(w);
    }

    pub fn query(&self, q: [f64; 2], tb: TieBreak) -> Option<[f64; 2]> {
        let (qx, qy) = (q[0], q[1]);
        if qy == 0.0 {
            if self.n == 0 {
                return None;
            }
            return Some(if qx > 0.0 {
                self.note(qx * self.max_kx, qx, q, None);
                self.right_meta.resolve(tb)
            } else if qx < 0.0 {
                self.note(qx * self.min_kx, qx, q, None);
                self.left_meta.resolve(tb)
            } else {
                self.global.resolve(tb)
            });
        }
        let half = if qy > 0.0 { &self.upper } else { &self.lower };
        half.query(qx, qy, tb).map(|h| {
            self.note(h.score, qy, q, Some(h.best_key));
            h.out
        })
    }
}

/// The `O(n)` reference head: score everything, keep the maxima.
#[derive(Default)]
pub struct BruteAttentionHead {
    entries: Vec<([f64; 2], [f64; 2], i32)>,
    grid: Cell<GridWitness>,
    gaps: Cell<ScoreGaps>,
}

impl BruteAttentionHead {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.grid.set(GridWitness::default());
        self.gaps.set(ScoreGaps::default());
    }

    /// How close the runner-up came, in key steps — the diagnostic of
    /// `todo3.md` section 2a.  Only this head can report it: it scores every
    /// entry anyway, while the hull visits the winner and its ties and stops.
    pub fn gap_witness(&self) -> ScoreGaps {
        self.gaps.get()
    }

    pub fn insert(&mut self, key: [f64; 2], val: [f64; 2], seq: i32) {
        self.entries.push((key, val, seq));
    }

    /// As `HardAttentionHead::grid_witness`: the wall is a property of the
    /// arithmetic, so the brute head crosses it on exactly the same query.
    pub fn grid_witness(&self) -> GridWitness {
        self.grid.get()
    }

    pub fn query(&self, q: [f64; 2], tb: TieBreak) -> Option<[f64; 2]> {
        if self.entries.is_empty() {
            return None;
        }
        let score = |k: &[f64; 2]| q[0] * k[0] + q[1] * k[1];
        let max = self.entries.iter().map(|(k, _, _)| score(k)).fold(f64::NEG_INFINITY, f64::max);
        let winner = self.entries.iter().find(|(k, _, _)| score(k) == max).map(|(k, _, _)| *k);
        let mut w = self.grid.get();
        w.observe(max, if q[1] != 0.0 { q[1] } else { q[0] }, q, winner);
        self.grid.set(w);
        let second = self
            .entries
            .iter()
            .map(|(k, _, _)| score(k))
            .filter(|&s| s < max)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut g = self.gaps.get();
        g.observe(max, second, q);
        self.gaps.set(g);

        let mut meta = HullMeta::default();
        for (k, v, seq) in &self.entries {
            if score(k) == max {
                meta.add(*v, *seq);
            }
        }
        Some(meta.resolve(tb))
    }
}
