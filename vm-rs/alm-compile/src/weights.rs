//! The analytical weights: the port of the second half of
//! `model/weights.py::build_model`, and of `save_weights`.
//!
//! Nothing here is learned.  Every row is an expression of the computation
//! graph read through the slot colouring, so the whole model is a
//! transcription of `wasm/interpreter.py` into float64 matrices.

use std::io::Write as _;

use crate::expr::{DimId, Expr};
use crate::graph::{DimData, Graph, TieBreak};
use crate::interpreter::MachineGraph;
use crate::plan::Plan;
use crate::slots;

/// A dense row-major matrix of float64, which is what the file format is.
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    fn set_row(&mut self, r: usize, v: &[f64]) {
        self.data[r * self.cols..(r + 1) * self.cols].copy_from_slice(v);
    }
    fn at(&mut self, r: usize, c: usize) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }
}

pub struct Layer {
    /// `(3 * d_model, d_model)`: query rows, then key rows, then value rows.
    pub in_proj: Mat,
    /// `(d_model, d_model)`.
    pub out_proj: Mat,
    /// `(2 * d_ffn, d_model)`: gate rows, then value rows.
    pub ff_in: Mat,
    /// `(d_model, d_ffn)`.
    pub ff_out: Mat,
}

pub struct Model {
    pub tokens: Vec<String>,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ffn: usize,
    pub stop_token_id: i32,
    pub tok: Mat,
    pub head: Mat,
    pub layers: Vec<Layer>,
    pub attn_erase: Vec<Vec<usize>>,
    pub ffn_erase: Vec<Vec<usize>>,
    pub head_tiebreak: Vec<Vec<i32>>,
}

/// `src_slot -> dst_slot -> coefficient`, both levels in insertion order,
/// because the packing below turns the outer order into head indices and the
/// inner order into the order the coefficients are summed.
#[derive(Default)]
struct Passthroughs {
    entries: Vec<(usize, Vec<(usize, f64)>)>,
}

impl Passthroughs {
    fn add(&mut self, src: usize, dst: usize, c: f64) {
        let row = match self.entries.iter().position(|e| e.0 == src) {
            Some(i) => &mut self.entries[i].1,
            None => {
                self.entries.push((src, Vec::new()));
                &mut self.entries.last_mut().unwrap().1
            }
        };
        match row.iter().position(|e| e.0 == dst) {
            Some(i) => row[i].1 += c,
            // The original starts from `defaultdict(float)`, so the first
            // contribution is `0.0 + c` and not `c`; they differ at `-0.0`.
            None => row.push((dst, 0.0 + c)),
        }
    }
}

fn persist_expr(g: &Graph, d: DimId) -> &Expr {
    match &g.dim(d).data {
        DimData::Persist { expr } => expr,
        _ => panic!("{} is not a persist dimension", g.name_of(d)),
    }
}

fn reglu_exprs(g: &Graph, d: DimId) -> (&Expr, &Expr) {
    match &g.dim(d).data {
        DimData::ReGlu { a, b } => (a, b),
        _ => panic!("{} is not a ReGLU dimension", g.name_of(d)),
    }
}

pub fn build(mg: &MachineGraph, plan: &Plan, use_erase: bool) -> Model {
    let g = &mg.graph;
    let lay = slots::compute(g, plan, &mg.output_tokens);
    let (d, dffn, nh) = (lay.d_model, lay.d_ffn, lay.n_heads);

    let mut tokens: Vec<String> = mg
        .input_tokens
        .iter()
        .chain(mg.output_tokens.iter())
        .map(|(t, _)| t.clone())
        .collect();
    tokens.sort();
    tokens.dedup();
    let idx = |t: &str| {
        tokens
            .binary_search_by(|p| p.as_str().cmp(t))
            .expect("token is in vocab")
    };
    let stop_token_id = tokens
        .binary_search_by(|p| p.as_str().cmp("halt"))
        .unwrap_or(0) as i32;

    // ── Embedding and output head ─────────────────────────────────────
    let mut tok = Mat::zeros(tokens.len(), d);
    for (name, e) in &mg.input_tokens {
        let mut row = lay.expr_to_row(e);
        // The positional slots are written by the encoding, not the token.
        row[..slots::PROTECTED].fill(0.0);
        tok.set_row(idx(name), &row);
    }
    let mut head = Mat::zeros(tokens.len(), d);
    for (name, e) in &mg.output_tokens {
        head.set_row(idx(name), &lay.expr_to_row(e));
    }

    let one_row = lay.expr_to_row(&g.one_expr());
    let pos_row = lay.expr_to_row(&Expr::dim(g.position));
    let pos2_row = lay.expr_to_row(&Expr::scaled(g.position, 2.0));

    let mut layers = Vec::new();
    let mut head_tiebreak = Vec::new();

    for (li, lp) in plan.layers.iter().enumerate() {
        let mut ip = Mat::zeros(3 * d, d);
        let mut op = Mat::zeros(d, d);
        let mut fi = Mat::zeros(2 * dffn, d);
        let mut fo = Mat::zeros(d, dffn);

        // ── Attention half ────────────────────────────────────────────
        let mut h = 0usize;
        let mut tiebreak: Vec<i32> = Vec::new();
        // dim -> (head, which of the head's two value channels)
        let mut from_head: Vec<Option<(usize, usize)>> = vec![None; g.dims.len()];

        for (lu_id, p) in slots::heads_of(g, &lp.attention) {
            let lu = g.lookup(lu_id);
            let nv = lu.value_exprs.len();
            tiebreak.push(i32::from(lu.tie_break == TieBreak::Latest));
            // todo3.md section 0: no `HARD_K * sqrt(d_head)` here — the hull
            // takes an argmax, which a positive scale cannot change, and the
            // scale is what lifts the score off the integer grid.
            ip.set_row(h * 2, &lay.expr_to_row(&lu.query_2d[0]));
            ip.set_row(h * 2 + 1, &lay.expr_to_row(&lu.query_2d[1]));
            ip.set_row(d + h * 2, &lay.expr_to_row(&lu.key_2d[0]));
            ip.set_row(d + h * 2 + 1, &lay.expr_to_row(&lu.key_2d[1]));
            for c in 0..2 {
                if p * 2 + c >= nv {
                    break;
                }
                ip.set_row(
                    2 * d + h * 2 + c,
                    &lay.expr_to_row(&lu.value_exprs[p * 2 + c]),
                );
                let dim = lu.dims[p * 2 + c];
                from_head[dim as usize] = Some((h, c));
                if !lay.internal_lookups.contains(&dim) {
                    *op.at(lay.slot(dim).expect("a read value has a slot"), h * 2 + c) = 1.0;
                }
            }
            h += 1;
        }

        let mut pt = Passthroughs::default();
        for &pd in &lp.persist1 {
            let Some(dst) = lay.slot(pd) else { continue };
            for &(dim, c) in persist_expr(g, pd).terms() {
                if let Some((hh, comp)) = from_head[dim as usize] {
                    *op.at(dst, hh * 2 + comp) += c;
                } else if let Some(src) = lay.slot(dim) {
                    pt.add(src, dst, c);
                }
            }
        }
        if use_erase {
            for &s in &lay.erased_at[li][0] {
                pt.add(s, s, -1.0);
            }
        }

        // Two source slots per head: the head reads a value and the residual
        // stream adds it back wherever the persist expressions asked for it.
        for pair in pt.entries.chunks(2) {
            ip.set_row(h * 2, &pos_row);
            ip.set_row(h * 2 + 1, &one_row);
            ip.set_row(d + h * 2, &pos2_row);
            ip.set_row(d + h * 2 + 1, &one_row);
            for (c, (src, dsts)) in pair.iter().enumerate() {
                *ip.at(2 * d + h * 2 + c, *src) = 1.0;
                for &(dst, coeff) in dsts {
                    *op.at(dst, h * 2 + c) += coeff;
                }
            }
            h += 1;
        }
        assert!(h <= nh, "L{li} attention: {h} heads > {nh}");
        tiebreak.resize(nh, 0);

        // ── FFN half ──────────────────────────────────────────────────
        let mut j = 0usize;
        let mut gate_of: Vec<Option<usize>> = vec![None; g.dims.len()];
        for &rg in &lp.ffn {
            let (a, b) = reglu_exprs(g, rg);
            fi.set_row(j, &lay.expr_to_row(b));
            fi.set_row(dffn + j, &lay.expr_to_row(a));
            gate_of[rg as usize] = Some(j);
            if !lay.internal_reglus.contains(&rg) {
                *fo.at(lay.slot(rg).expect("a kept ReGLU has a slot"), j) = 1.0;
            }
            j += 1;
        }

        let mut pt_ffn = Passthroughs::default();
        for &pd in &lp.persist2 {
            let Some(dst) = lay.slot(pd) else { continue };
            for &(dim, c) in persist_expr(g, pd).terms() {
                if let Some(gate) = gate_of[dim as usize] {
                    *fo.at(dst, gate) += c;
                } else if let Some(src) = lay.slot(dim) {
                    pt_ffn.add(src, dst, c);
                }
            }
        }
        if use_erase {
            for &s in &lay.erased_at[li][1] {
                pt_ffn.add(s, s, -1.0);
            }
        }
        // `relu(1) * x` is `x`, so a neuron whose gate is the constant one and
        // whose value row is a single slot is a copy of that slot.
        for (src, dsts) in &pt_ffn.entries {
            fi.set_row(j, &one_row);
            *fi.at(dffn + j, *src) = 1.0;
            for &(dst, coeff) in dsts {
                *fo.at(dst, j) += coeff;
            }
            j += 1;
        }
        assert!(j <= dffn, "L{li} FFN: {j} neurons > {dffn}");

        layers.push(Layer {
            in_proj: ip,
            out_proj: op,
            ff_in: fi,
            ff_out: fo,
        });
        head_tiebreak.push(tiebreak);
    }

    Model {
        tokens,
        d_model: d,
        n_heads: nh,
        d_ffn: dffn,
        stop_token_id,
        tok,
        head,
        layers,
        attn_erase: lay.erased_at.iter().map(|h| h[0].clone()).collect(),
        ffn_erase: lay.erased_at.iter().map(|h| h[1].clone()).collect(),
        head_tiebreak,
    }
}

impl Model {
    /// The flat format the C++ engine and `alm-vm` read: a `<6i` header, the
    /// vocabulary as length-prefixed bytes, then every matrix row-major in
    /// little-endian float64, then the erase and tie-break tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let i32s = |o: &mut Vec<u8>, vs: &[i32]| {
            for v in vs {
                o.extend_from_slice(&v.to_le_bytes());
            }
        };
        i32s(
            &mut out,
            &[
                self.tokens.len() as i32,
                self.d_model as i32,
                self.layers.len() as i32,
                self.n_heads as i32,
                self.d_ffn as i32,
                self.stop_token_id,
            ],
        );
        for t in &self.tokens {
            out.extend_from_slice(&(t.len() as u32).to_le_bytes());
            out.extend_from_slice(t.as_bytes());
        }
        let mat = |o: &mut Vec<u8>, m: &Mat| {
            for x in &m.data {
                o.extend_from_slice(&x.to_le_bytes());
            }
        };
        mat(&mut out, &self.tok);
        for l in &self.layers {
            mat(&mut out, &l.in_proj);
            mat(&mut out, &l.out_proj);
            mat(&mut out, &l.ff_in);
            mat(&mut out, &l.ff_out);
        }
        mat(&mut out, &self.head);

        i32s(&mut out, &[1]);
        for li in 0..self.layers.len() {
            for table in [&self.attn_erase, &self.ffn_erase] {
                i32s(&mut out, &[table[li].len() as i32]);
                for &s in &table[li] {
                    i32s(&mut out, &[s as i32]);
                }
            }
        }
        i32s(&mut out, &[1]);
        for row in &self.head_tiebreak {
            i32s(&mut out, row);
        }
        out
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::File::create(path)?.write_all(&self.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Plan;

    /// The whole point of the port: the file this builds is the file the
    /// Python builder writes, to the byte.  Both are skipped when the vendored
    /// checkout is absent — `plan.yaml` and `model.bin` are build products of
    /// the Python and are not in this repository.
    ///
    /// `model.bin` has to have been built with `patches/reproducible-build.patch`
    /// applied; without it the Python picks a different permutation of layer
    /// 5's FFN passthrough neurons on most runs (todo3.md section 9).
    #[test]
    fn the_model_comes_out_byte_for_byte_the_python_one() {
        let (Ok(plan_text), Ok(want)) = (
            std::fs::read_to_string("../transformer-vm/plan.yaml"),
            std::fs::read("../transformer-vm/model.bin"),
        ) else {
            return;
        };
        let mg = crate::interpreter::build();
        let plan = Plan::load(&plan_text, &mg.graph).expect("plan.yaml resolves");
        let model = build(&mg, &plan, true);

        assert_eq!(model.tokens.len(), 915);
        assert_eq!((model.d_model, model.n_heads, model.d_ffn), (38, 19, 47));

        let got = model.to_bytes();
        assert_eq!(got.len(), want.len(), "same size as the Python model.bin");
        let first = got.iter().zip(&want).position(|(a, b)| a != b);
        assert_eq!(first, None, "first differing byte");
    }
}
