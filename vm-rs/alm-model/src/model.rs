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

/// A dense matrix, row-major `[rows, cols]`, exactly as `model.bin` stores it.
struct Dense {
    cols: usize,
    w: Vec<f64>,
}

impl Dense {
    fn of(w: &[f64], rows: usize, cols: usize) -> Dense {
        assert_eq!(w.len(), rows * cols, "the weight file declares its own shapes");
        Dense { cols, w: w.to_vec() }
    }

    /// `y = W x`, each row summed left to right.
    ///
    /// The order is the one `transformer.cpp` uses and the one the reference
    /// traces were generated under; float addition is not associative, so it
    /// is part of the answer rather than of the schedule.
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.cols);
        for (row, out) in self.w.chunks_exact(self.cols).zip(y.iter_mut()) {
            let mut s = 0.0;
            for (a, b) in row.iter().zip(x) {
                s += a * b;
            }
            *out = s;
        }
    }
}

/// The four projections of one layer.
struct LayerWeights {
    qkv: Dense,
    out: Dense,
    ff_in: Dense,
    ff_out: Dense,
}

/// The output head, in compressed sparse rows.
///
/// It is the one projection the C++ runtime does not do densely, and the
/// reason is in the numbers: the head is `vocab x d_model`, 915 x 38 here and
/// 85 % zero, and it runs once per generated token.  Skipping the zeros is
/// exact — adding `0.0 * x` to a finite partial sum never changes it — so
/// this is the same argmax, not an approximation of it.
struct SparseHead {
    rows: usize,
    /// `row i` occupies `col[ptr[i]..ptr[i+1]]`.
    ptr: Vec<usize>,
    col: Vec<usize>,
    val: Vec<f64>,
}

impl SparseHead {
    fn of(w: &[f64], rows: usize, cols: usize) -> SparseHead {
        let mut head = SparseHead { rows, ptr: vec![0], col: Vec::new(), val: Vec::new() };
        for i in 0..rows {
            for j in 0..cols {
                let v = w[i * cols + j];
                if v != 0.0 {
                    head.col.push(j);
                    head.val.push(v);
                }
            }
            head.ptr.push(head.col.len());
        }
        head
    }

    /// The first index attaining the maximum, as `Tensor::argmax` and the C++
    /// loop both resolve it.
    fn argmax(&self, x: &[f64]) -> usize {
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..self.rows {
            let mut s = 0.0;
            for k in self.ptr[i]..self.ptr[i + 1] {
                s += self.val[k] * x[self.col[k]];
            }
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        best
    }
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

/// The compiled transformer.
pub struct Alm {
    pub shapes: Shapes,
    pub tokens: Vec<String>,
    embedding: Vec<f64>,
    layers: Vec<LayerWeights>,
    head: SparseHead,
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
        }
    }

    /// The residual stream at the start of a position: the token embedding plus
    /// the three position features the compiler assumes are there.
    ///
    /// `1/log 2 - 1/log(pos + 2)` is the recency feature the latest-write
    /// perturbation is built from, and `pos * pos` is what makes a parabolic
    /// key of the position.
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
        x[1] += 1.0 / std::f64::consts::LN_2 - 1.0 / ((p + 2.0).ln());
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
        (self.head.val.len(), self.head.rows * self.shapes.d_model)
    }
}
