//! The key-value cache: one hard-attention head per (layer, head) pair.
//!
//! Ported from `attention/hull_cache.py` and the head loop of
//! `model/transformer.cpp`.  Two details of the original are easy to miss and
//! are reproduced exactly, because the latest-write tie-break depends on both:
//! the sequence counter advances once per *layer step*, not once per token, and
//! every head in a layer therefore shares one sequence number.

use alm_hull::{BruteAttentionHead, GridWitness, HardAttentionHead, IntegerQueries, ScoreGaps, TieBreak};

/// Which head implementation answers the queries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CacheKind {
    /// The convex-hull head: `O(log n)` per query.
    Hull,
    /// The linear scan: `O(n)` per query, and the reference the hull is
    /// checked against.
    Brute,
}

enum Heads {
    Hull(Vec<HardAttentionHead>),
    Brute(Vec<BruteAttentionHead>),
}

pub struct KvCache {
    heads: Heads,
    tie: Vec<TieBreak>,
    n_heads: usize,
    seq: i32,
    grid: bool,
    queries: IntegerQueries,
}

/// Rescale a query to unit scale without changing what it selects.
///
/// `todo3.md` section 0: the compiler multiplies every hard-attention query by
/// `HARD_K * sqrt(2) = 1.41e10`, a softmax temperature applied on a path that
/// takes an argmax.  `argmax_k <q, k>` is invariant under `q -> q / s` for any
/// `s > 0`, so the scale decides nothing — it only lifts `2qk - k^2` off the
/// integer grid, which is the one hypothesis
/// `ALM.FloatGrid.fp_eval_exact_of_grid` needs to call the score exact.
///
/// Dividing by `|q_y|` puts it back.  The division is the only rounding
/// introduced, and it is harmless: `q_x` is the correctly rounded `s * q` for
/// an integer `q`, so `q_x / |q_y|` is within `q * 2^-52` of `q`, and for every
/// `q` below the wall that rounds back to `q` exactly.
fn on_the_grid(q: [f64; 2]) -> [f64; 2] {
    let s = if q[1] != 0.0 { q[1].abs() } else { q[0].abs() };
    if s == 0.0 || !s.is_finite() {
        return q;
    }
    [q[0] / s, q[1] / s]
}

impl KvCache {
    pub fn new(n_layers: usize, n_heads: usize, kind: CacheKind) -> KvCache {
        let n = n_layers * n_heads;
        let heads = match kind {
            CacheKind::Hull => Heads::Hull((0..n).map(|_| HardAttentionHead::new()).collect()),
            CacheKind::Brute => Heads::Brute((0..n).map(|_| BruteAttentionHead::new()).collect()),
        };
        KvCache {
            heads,
            tie: vec![TieBreak::Average; n],
            n_heads,
            seq: -1,
            grid: false,
            queries: IntegerQueries::default(),
        }
    }

    /// Answer queries at unit scale rather than at the compiler's — the fix of
    /// `todo3.md` section 0.  See `on_the_grid`.
    pub fn set_grid(&mut self, on: bool) {
        self.grid = on;
    }

    /// Mark a head as resolving ties to the latest write.
    pub fn set_tie_break(&mut self, layer: usize, head: usize, tb: TieBreak) {
        self.tie[layer * self.n_heads + head] = tb;
    }

    /// Insert this position's key-value pair into every head of one layer and
    /// answer this position's queries.
    ///
    /// `keys`, `queries` and `values` are the layer's projections, `d_model`
    /// wide, read two coordinates at a time — one 2D head per pair.
    pub fn layer_step(&mut self, layer: usize, keys: &[f64], queries: &[f64], values: &[f64]) -> Vec<f64> {
        self.seq += 1;
        let seq = self.seq;
        let mut out = vec![0.0; 2 * self.n_heads];

        for h in 0..self.n_heads {
            let i = layer * self.n_heads + h;
            let (k, q, v) = (
                [keys[2 * h], keys[2 * h + 1]],
                [queries[2 * h], queries[2 * h + 1]],
                [values[2 * h], values[2 * h + 1]],
            );
            self.queries.observe(q);
            let q = if self.grid { on_the_grid(q) } else { q };
            let answer = match &mut self.heads {
                Heads::Hull(hs) => {
                    hs[i].insert(k, v, seq);
                    hs[i].query(q, self.tie[i])
                }
                Heads::Brute(bs) => {
                    bs[i].insert(k, v, seq);
                    bs[i].query(q, self.tie[i])
                }
            };
            // A head always holds this position's own entry, so it is never
            // empty here; the zero is what the C++ leaves behind regardless.
            if let Some(a) = answer {
                out[2 * h] = a[0];
                out[2 * h + 1] = a[1];
            }
        }
        out
    }

    /// Everything the heads have seen that float64 cannot separate, summed
    /// across the stack.  `todo3.md` section 4: above `2^53` the construction
    /// is not slow or approximate, it is wrong, and the original says nothing.
    /// Here it is reported rather than asserted — an assertion would kill a
    /// three-minute run to tell it something the last line could have said.
    /// What the run's queries were, as values rather than as directions:
    /// `todo3.md` section 8 asks whether they are the integers the exactness
    /// argument assumes, and this is the count.
    pub fn query_witness(&self) -> IntegerQueries {
        self.queries
    }

    /// How close the runner-up came, in key steps, over the whole run —
    /// `todo3.md` section 2a.  Empty under the hull cache, which never looks
    /// past the winner's ties.
    pub fn gap_witness(&self) -> ScoreGaps {
        let mut all = ScoreGaps::default();
        if let Heads::Brute(bs) = &self.heads {
            bs.iter().for_each(|h| all.merge(&h.gap_witness()));
        }
        all
    }

    pub fn grid_witness(&self) -> GridWitness {
        let mut all = GridWitness::default();
        match &self.heads {
            Heads::Hull(hs) => hs.iter().for_each(|h| all.merge(&h.grid_witness())),
            Heads::Brute(bs) => bs.iter().for_each(|h| all.merge(&h.grid_witness())),
        }
        all
    }
}
