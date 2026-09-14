//! The key-value cache: one hard-attention head per (layer, head) pair.
//!
//! Ported from `attention/hull_cache.py` and the head loop of
//! `model/transformer.cpp`.  Two details of the original are easy to miss and
//! are reproduced exactly, because the latest-write tie-break depends on both:
//! the sequence counter advances once per *layer step*, not once per token, and
//! every head in a layer therefore shares one sequence number.

use alm_hull::{BruteAttentionHead, HardAttentionHead, TieBreak};

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
}

impl KvCache {
    pub fn new(n_layers: usize, n_heads: usize, kind: CacheKind) -> KvCache {
        let n = n_layers * n_heads;
        let heads = match kind {
            CacheKind::Hull => Heads::Hull((0..n).map(|_| HardAttentionHead::new()).collect()),
            CacheKind::Brute => Heads::Brute((0..n).map(|_| BruteAttentionHead::new()).collect()),
        };
        KvCache { heads, tie: vec![TieBreak::Average; n], n_heads, seq: -1 }
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
}
