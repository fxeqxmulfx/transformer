//! The transformer itself.
//!
//! Ported from `VanillaTransformer` in `transformer_vm/model/transformer.py`
//! and its C++ twin.  The shape of the computation is unusual for a tensor
//! framework and worth stating: generation is one token at a time with no
//! batch, so every operation here is a matrix-vector product against a residual
//! stream 38 wide, and the attention is not a tensor operation at all but a
//! query into a convex hull, which is why the forward pass hands the
//! projections out to `alm_hull` and takes a vector back.
//!
//! Those two facts are why there is no tensor framework under this.  A
//! `[1, 38] x [38, 114]` product is a hundred nanoseconds of arithmetic and
//! several times that in dispatch, allocation and the round trip through a
//! tensor type; measured against the same loop written out, a tensor
//! framework's `Linear` cost 3.5 times the arithmetic it performed, and the
//! projections are 60 % of a run.  What is here instead is the loop
//! `transformer.cpp` writes.
//!
//! The element type is `f64` throughout, as it is in the original
//! (`torch.set_default_dtype(torch.float64)`).  It is not an implementation
//! detail: the exactness the construction claims is exactness of integers below
//! `2^53`, and nothing about it survives a narrower float.

use std::time::Instant;

use crate::cache::KvCache;
use crate::linear::{Dense, SparseHead};
use crate::weights::{RawModel, Shapes};

/// Where the time goes, split the way `transformer.cpp` splits it, so the two
/// runtimes can be compared line for line.
#[derive(Clone, Copy, Default, Debug)]
pub struct Timings {
    /// The four projections of every layer.
    pub proj: f64,
    /// Insert and query, across every head.
    pub hull: f64,
    /// The output head: one 915-wide argmax per generated token.
    pub head: f64,
}

/// The four projections of one layer.
struct LayerWeights {
    qkv: Dense,
    out: Dense,
    ff_in: Dense,
    ff_out: Dense,
}

/// The buffers one position's forward pass writes into.
///
/// Held by the caller rather than allocated per token: the whole of a step is
/// a few microseconds, and five allocations inside it are not free at that
/// scale.  `Alm::forward` keeps one for callers that do not want to.
pub struct Scratch {
    x: Vec<f64>,
    qkv: Vec<f64>,
    attn_out: Vec<f64>,
    ff: Vec<f64>,
    gated: Vec<f64>,
    back: Vec<f64>,
}

impl Scratch {
    pub fn new(s: Shapes) -> Scratch {
        let (d, f) = (s.d_model, s.d_ffn);
        Scratch {
            x: vec![0.0; d],
            qkv: vec![0.0; 3 * d],
            attn_out: vec![0.0; d],
            ff: vec![0.0; 2 * f],
            gated: vec![0.0; f],
            back: vec![0.0; d],
        }
    }
}

/// **Which recency feature position `1` of the residual stream carries.**
///
/// The compiler does not compute this feature, it reads the dimension and
/// scales it by `LATEST_ALPHA = 0.3` (`graph/core.py:316`), so any monotone
/// function of the position into `[0, 1 / ln 2)` is a legal recency term and
/// none of the weights change with the choice.  What the choice decides is how
/// long rewriting an address keeps working.
///
/// The term is added to `-k^2`, so it is visible only while it exceeds
/// `ulp(a^2)` at the address `a`.  That gives one budget for the address space
/// and the rewrite depth together, `addresses^2 * rewrites <= 0.4328 * 2^52`,
/// and `InvLogPos` reaches nowhere near it: its increments decay like
/// `1 / ((p + 2) ln^2(p + 2))` while `ulp(a^2)` does not, so at address `10^5`
/// consecutive rewrites stop being ordered after position 2552, against the
/// 226 916 the budget allows.  `Linear` spends the same span in equal steps and
/// attains it.  Measured in `alm-margin`, `rewrite::Recency`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Recency {
    /// What the shipped model was compiled against: `1/log 2 - 1/log(p + 2)`,
    /// `transformer_vm/evaluator.py:230`.  Needs no horizon, and saturates.
    #[default]
    InvLogPos,
    /// `(p / horizon) / log 2`, held at `1 / log 2` from `horizon` on.  Orders
    /// every rewrite up to the horizon and none after it; the price of not
    /// saturating is having to name the horizon in advance.
    Linear { horizon: u64 },
}

impl Recency {
    /// The value the feature takes at this position.  Both variants stay
    /// within `[0, 1 / log 2)`, which is what keeps the scaled term below the
    /// unit gap between distinct integer keys.
    #[inline]
    pub fn feature(self, p: f64) -> f64 {
        match self {
            Recency::InvLogPos => 1.0 / std::f64::consts::LN_2 - 1.0 / ((p + 2.0).ln()),
            Recency::Linear { horizon } => {
                (p / horizon.max(1) as f64).min(1.0) / std::f64::consts::LN_2
            }
        }
    }
}

/// The compiled transformer.
pub struct Alm {
    pub shapes: Shapes,
    pub tokens: Vec<String>,
    embedding: Vec<f64>,
    layers: Vec<LayerWeights>,
    head: SparseHead,
    recency: Recency,
}

impl Alm {
    pub fn from_raw(raw: &RawModel) -> Alm {
        let s = raw.shapes;
        let (d, f, v) = (s.d_model, s.d_ffn, s.vocab);
        let layers = raw
            .layers
            .iter()
            .map(|l| LayerWeights {
                qkv: Dense::of(&l.qkv, 3 * d, d),
                out: Dense::of(&l.out, d, d),
                ff_in: Dense::of(&l.ff_in, 2 * f, d),
                ff_out: Dense::of(&l.ff_out, d, f),
            })
            .collect();
        Alm {
            shapes: s,
            tokens: raw.tokens.clone(),
            embedding: raw.embedding.clone(),
            layers,
            head: SparseHead::of(&raw.head, v, d),
            recency: Recency::default(),
        }
    }

    /// Feed a different recency feature.  The weights are untouched -- the
    /// compiler only ever scales this dimension -- so this is the one knob
    /// that moves the rewrite horizon without recompiling anything.  The
    /// default is what the shipped model was compiled against.
    pub fn with_recency(mut self, r: Recency) -> Alm {
        self.recency = r;
        self
    }

    /// Which one it is carrying.
    pub fn recency(&self) -> Recency {
        self.recency
    }

    /// The residual stream at the start of a position: the token embedding plus
    /// the three position features the compiler assumes are there.
    ///
    /// [`Recency`] is the recency feature the latest-write perturbation is
    /// built from, `1/log 2 - 1/log(pos + 2)` unless it has been replaced, and
    /// `pos * pos` is what makes a parabolic key of the position.
    ///
    /// The feature is `Transformer.ALM.LatestWindow.invLogPos`, the same
    /// expression the compiler computes (`transformer_vm/evaluator.py:230`),
    /// and it is the one coordinate of the residual stream that is not an
    /// integer.  What the head needs of it is proved there and in
    /// `ALM.LatestClose`: it rises with the position, so the latest write
    /// wins a tie, and consecutive positions close up like `1/log(pos + 2)`,
    /// which is the window `ALM.HullNear` and `alm_hull`'s `SepWitness`
    /// measure a run against.
    pub fn embed(&self, token: usize, pos: usize) -> Vec<f64> {
        let d = self.shapes.d_model;
        let mut x = vec![0.0; d];
        self.embed_into(token, pos, &mut x);
        x
    }

    fn embed_into(&self, token: usize, pos: usize, x: &mut [f64]) {
        let d = self.shapes.d_model;
        x.copy_from_slice(&self.embedding[token * d..(token + 1) * d]);
        let p = pos as f64;
        x[0] += p;
        x[1] += self.recency.feature(p);
        x[2] += p * p;
    }

    /// One position through the whole stack, updating the cache as it goes.
    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache) -> Vec<f64> {
        let mut s = Scratch::new(self.shapes);
        self.forward_timed(token, pos, cache, &mut s, &mut Timings::default());
        s.x
    }

    /// The same, charging each part of the step to `t` and leaving the residual
    /// stream in `s.x`.
    ///
    /// Three parts of this loop carry a theorem and the rest does not.  The
    /// projections are `Transformer.ALM.apply_eq_rowMajor`, the attention is
    /// `KvCache::layer_step` and everything `alm_hull` is checked by, and the
    /// decode is `ALM.firstMax_sparse_eq`.  The two residual additions and the
    /// gated feed-forward are a straight port of `transformer.cpp`: that the
    /// port is faithful rests on `alm-vm/tests/reference.rs`, which reproduces
    /// the released traces token for token, and on nothing else — the traces
    /// are evidence only as far as they reach.
    ///
    /// What `ALM.GateGrid` adds is where they stop.  `max(ff, 0) * ff'`
    /// multiplies two coordinates of the residual stream together, so it
    /// leaves the integer grid once the stream passes 94 906 265
    /// (`the_ffn_wall`, `isBinary_gate_of_wall`) — the same wall
    /// `ALM.ScoreWall` finds for the score `2qk - k^2`, reached by squaring
    /// the stream's range instead of the score's.  Under it the gate and both
    /// `+=` are exact (`isBinary_gate`, `isBinary_add`), and the rectifier is
    /// free at any magnitude (`relu_isBinary`): `max a 0` is `a` or `0`, and
    /// both are representable as soon as `a` is.  All of that is conditional
    /// on integer operands, which one lane of the stream is not — `x[1]`
    /// carries the recency feature `embed_into` writes there — and for that lane
    /// nothing here replaces the rounding it costs.
    ///
    /// The gate is spelled differently from its original and answers the same
    /// (`ALM.GateGrid.gate_eq_ite`, over the reals, where the two cases below
    /// do not exist): `transformer.cpp:315` writes `(ff[i] > 0 ? ff[i] : 0.0)`
    /// and this writes `.max(0.0)`, which agree on `NaN` and, as compiled here, on a
    /// negative zero — the one case the standard leaves to the
    /// implementation, and one no comparison downstream can see anyway.
    pub fn forward_timed<'s>(
        &self,
        token: usize,
        pos: usize,
        cache: &mut KvCache,
        s: &'s mut Scratch,
        t: &mut Timings,
    ) -> &'s [f64] {
        let (d, f) = (self.shapes.d_model, self.shapes.d_ffn);
        self.embed_into(token, pos, &mut s.x);

        for (li, layer) in self.layers.iter().enumerate() {
            let mark = Instant::now();
            layer.qkv.apply(&s.x, &mut s.qkv);
            t.proj += mark.elapsed().as_secs_f64();

            let mark = Instant::now();
            let attn = cache.layer_step(li, &s.qkv[d..2 * d], &s.qkv[..d], &s.qkv[2 * d..]);
            t.hull += mark.elapsed().as_secs_f64();

            let mark = Instant::now();
            layer.out.apply(&attn, &mut s.attn_out);
            for i in 0..d {
                s.x[i] += s.attn_out[i];
            }

            layer.ff_in.apply(&s.x, &mut s.ff);
            for i in 0..f {
                s.gated[i] = s.ff[i].max(0.0) * s.ff[f + i];
            }
            layer.ff_out.apply(&s.gated, &mut s.back);
            for i in 0..d {
                s.x[i] += s.back[i];
            }
            t.proj += mark.elapsed().as_secs_f64();
        }
        &s.x
    }

    /// The token the residual stream decodes to.
    pub fn decode(&self, x: &[f64]) -> usize {
        self.head.argmax(x)
    }

    /// How many of the head's entries are actually nonzero — the C++ prints
    /// this at load time, and it is the justification for the sparse form.
    pub fn head_density(&self) -> (usize, usize) {
        (self.head.nnz(), self.head.rows() * self.shapes.d_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The default is the shipped feature, to the bit.**  Anything else and
    /// the released weights stop reproducing, which `alm-vm`'s reference test
    /// would catch a whole run later.
    #[test]
    fn the_default_is_the_shipped_feature() {
        for pos in [0u32, 1, 2, 17, 1000, 100_000, 3_000_000] {
            let p = pos as f64;
            let want = 1.0 / std::f64::consts::LN_2 - 1.0 / ((p + 2.0).ln());
            assert_eq!(Recency::default().feature(p).to_bits(), want.to_bits());
        }
        assert_eq!(Recency::default(), Recency::InvLogPos);
    }

    /// Both features stay inside `[0, 1 / log 2)`, so the scaled term stays
    /// below the unit gap between distinct integer keys either way.
    #[test]
    fn both_features_stay_under_the_unit_gap() {
        const ALPHA: f64 = 0.3;
        let lin = Recency::Linear { horizon: 1024 };
        for p in [0.0f64, 1.0, 512.0, 1023.0, 1e6, 1e12] {
            for r in [Recency::InvLogPos, lin] {
                let f = r.feature(p);
                assert!((0.0..=1.0 / std::f64::consts::LN_2).contains(&f), "p = {p}");
                assert!(ALPHA * f < 1.0);
            }
        }
        assert_eq!(lin.feature(1024.0), 1.0 / std::f64::consts::LN_2);
        assert_eq!(lin.feature(4096.0), lin.feature(1024.0));
    }

    /// **And the linear one steps evenly where the logarithm collapses.**  The
    /// span is the same; how it is spent is the whole difference, and it is
    /// what decides how late an address can still be rewritten.
    #[test]
    fn the_linear_feature_steps_evenly() {
        let lin = Recency::Linear { horizon: 100_000 };
        let step = lin.feature(1.0) - lin.feature(0.0);
        for p in [0.0f64, 1.0, 500.0, 50_000.0, 99_998.0] {
            let got = lin.feature(p + 1.0) - lin.feature(p);
            assert!((got - step).abs() < 1e-15, "p = {p}: {got} vs {step}");
        }
        let log = Recency::InvLogPos;
        let early = log.feature(1.0) - log.feature(0.0);
        let late = log.feature(50_001.0) - log.feature(50_000.0);
        assert!(late < early / 1e4, "inv_log_pos should have collapsed by 5e4");
        // 1.71e-7 against the even 1.44e-5: eighty-five times under it, and
        // the gap widens with the position while the even step does not.
        assert!(late < step / 50.0, "and fallen far under the even step");
    }
}
