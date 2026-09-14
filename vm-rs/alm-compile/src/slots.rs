//! Where each dimension lives in the residual stream, and how wide the model
//! has to be to hold them.
//!
//! The port of the first half of `model/weights.py::build_model`: interval
//! colouring over the schedule, the internal dimensions that never get a slot
//! at all, and the head and FFN counts that fall out of the colouring.

use std::collections::HashSet;

use crate::expr::{DimId, Expr};
use crate::graph::{DimData, Graph, LookUpId};
use crate::plan::Plan;

/// The three positional inputs are pinned to slots 0, 1, 2 and never reused:
/// the embedding writes zero there and the positional encoding owns them.
pub const PROTECTED: usize = 3;

/// The colouring, and everything downstream of it that the weight builder needs.
pub struct Layout {
    /// Slot of each dimension by id, `None` for the ones that never get one.
    pub slot_of: Vec<Option<usize>>,
    pub input_dims: Vec<DimId>,
    /// Slots zeroed at the end of each half-layer, sorted.  Indexed as
    /// `[layer][0]` for the attention half and `[layer][1]` for the FFN half.
    pub erased_at: Vec<[Vec<usize>; 2]>,
    /// Dimensions that are computed and consumed inside one half-layer and are
    /// never alive across a boundary, so they are never written to a slot.
    pub internal: HashSet<DimId>,
    pub internal_reglus: HashSet<DimId>,
    pub internal_lookups: HashSet<DimId>,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ffn: usize,
}

impl Layout {
    pub fn slot(&self, d: DimId) -> Option<usize> {
        self.slot_of[d as usize]
    }

    /// The row of the residual stream an expression contributes, with internal
    /// dimensions dropped and reused slots accumulating in term order.
    pub fn expr_to_row(&self, e: &Expr) -> Vec<f64> {
        let mut w = vec![0.0f64; self.d_model];
        for &(d, c) in e.terms() {
            if self.internal.contains(&d) {
                continue;
            }
            if let Some(s) = self.slot(d) {
                w[s] += c;
            }
        }
        w
    }
}

fn persist_expr(g: &Graph, d: DimId) -> &Expr {
    match &g.dim(d).data {
        DimData::Persist { expr } => expr,
        other => panic!(
            "plan lists {} as a persist, but it is {:?}",
            g.name_of(d),
            other
        ),
    }
}

/// Interval colouring, exactly as `build_model` does it.
///
/// A slot freed at the end of a half-layer becomes available only at the
/// *next* boundary (`pending_free`), because the erase that frees it is itself
/// one of the writes of the half-layer that frees it.
pub fn compute(g: &Graph, plan: &Plan, out_tokens: &[(String, Expr)]) -> Layout {
    let n = g.dims.len();
    let input_dims: Vec<DimId> = g
        .dims
        .iter()
        .filter(|d| d.is_input())
        .map(|d| d.id)
        .collect();

    let mut slot_of: Vec<Option<usize>> = vec![None; n];
    slot_of[g.position as usize] = Some(0);
    slot_of[g.inv_log_pos as usize] = Some(1);
    slot_of[g.position_sq as usize] = Some(2);
    let mut next_slot = PROTECTED;
    for &d in &input_dims {
        if slot_of[d as usize].is_none() {
            slot_of[d as usize] = Some(next_slot);
            next_slot += 1;
        }
    }

    let mut free_slots: Vec<usize> = Vec::new();
    let mut pending_free: Vec<usize> = Vec::new();
    let mut erased_at: Vec<[Vec<usize>; 2]> = Vec::new();

    let mut cur: HashSet<DimId> = input_dims.iter().copied().collect();
    for lp in &plan.layers {
        let mut halves: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for (half, alive) in [(0usize, &lp.after_persist1), (1usize, &lp.after_persist2)] {
            let nxt: HashSet<DimId> = match alive {
                Some(v) => v.iter().copied().collect(),
                None => cur.clone(),
            };

            free_slots.append(&mut pending_free);
            free_slots.sort_unstable();

            // `erased` is a set in the original and its iteration order is an
            // address accident; see todo3.md section 9.  Sorted here, which is
            // what `patches/reproducible-build.patch` makes the Python do.
            let mut erased: Vec<usize> = Vec::new();
            for &d in cur.difference(&nxt) {
                let s = slot_of[d as usize].expect("a live dimension has a slot");
                if s < PROTECTED {
                    continue;
                }
                pending_free.push(s);
                erased.push(s);
            }
            erased.sort_unstable();

            let mut born: Vec<DimId> = nxt.difference(&cur).copied().collect();
            born.sort_unstable();
            for d in born {
                let s = if free_slots.is_empty() {
                    next_slot += 1;
                    next_slot - 1
                } else {
                    free_slots.remove(0)
                };
                slot_of[d as usize] = Some(s);
            }

            halves[half] = erased;
            cur = nxt;
        }
        erased_at.push(halves);
    }

    // Dimensions the output head reads but no half-layer keeps alive.
    let mut need: Vec<DimId> = Vec::new();
    for (_, e) in out_tokens {
        for &(d, _) in e.terms() {
            if slot_of[d as usize].is_none() && !need.contains(&d) {
                need.push(d);
            }
        }
    }
    need.sort_unstable();
    for d in need {
        let s = if free_slots.is_empty() {
            next_slot += 1;
            next_slot - 1
        } else {
            free_slots.remove(0)
        };
        slot_of[d as usize] = Some(s);
    }

    let mut d_model = next_slot;
    d_model += d_model % 2;

    // ── Internal dimensions ───────────────────────────────────────────
    let mut non_internal: HashSet<DimId> = input_dims.iter().copied().collect();
    for lp in &plan.layers {
        for alive in [&lp.after_persist1, &lp.after_persist2]
            .into_iter()
            .flatten()
        {
            non_internal.extend(alive.iter().copied());
        }
    }
    for (_, e) in out_tokens {
        non_internal.extend(e.terms().iter().map(|&(d, _)| d));
    }
    let pick = |f: &dyn Fn(&crate::graph::Dim) -> bool| -> HashSet<DimId> {
        g.dims
            .iter()
            .filter(|d| f(d) && !non_internal.contains(&d.id))
            .map(|d| d.id)
            .collect()
    };
    let internal_reglus = pick(&|d| d.is_reglu());
    let internal_lookups = pick(&|d| d.is_lookup());
    let internal_persists = pick(&|d| d.is_persist());
    let mut internal = internal_reglus.clone();
    internal.extend(internal_lookups.iter().copied());
    internal.extend(internal_persists.iter().copied());

    // ── How many heads and FFN neurons the schedule needs ─────────────
    let mut max_heads = 0usize;
    let mut max_ffn = 1usize;
    let mut cur: HashSet<DimId> = input_dims.iter().copied().collect();
    for (li, lp) in plan.layers.iter().enumerate() {
        let nxt1: HashSet<DimId> = match &lp.after_persist1 {
            Some(v) => v.iter().copied().collect(),
            None => cur.clone(),
        };

        let mut lookup_dims: HashSet<DimId> = HashSet::new();
        let mut n_lu_heads = 0usize;
        for &lu in &lp.attention {
            let lu = g.lookup(lu);
            lookup_dims.extend(lu.dims.iter().copied());
            n_lu_heads += lu.value_exprs.len().div_ceil(2);
        }

        let mut pt: HashSet<usize> = erased_at[li][0].iter().copied().collect();
        for &pd in &lp.persist1 {
            for &(d, _) in persist_expr(g, pd).terms() {
                if internal.contains(&d) {
                    continue;
                }
                let Some(s) = slot_of[d as usize] else {
                    continue;
                };
                if !(g.dim(d).is_lookup() && lookup_dims.contains(&d)) {
                    pt.insert(s);
                }
            }
        }
        max_heads = max_heads.max(n_lu_heads + pt.len().div_ceil(2));
        cur = nxt1;

        let nxt3: HashSet<DimId> = match &lp.after_persist2 {
            Some(v) => v.iter().copied().collect(),
            None => cur.clone(),
        };
        let same_rg: HashSet<DimId> = lp.ffn.iter().copied().collect();
        let mut pt_ffn: HashSet<usize> = erased_at[li][1].iter().copied().collect();
        for &pd in &lp.persist2 {
            for &(d, _) in persist_expr(g, pd).terms() {
                if internal.contains(&d) {
                    continue;
                }
                let Some(s) = slot_of[d as usize] else {
                    continue;
                };
                if !same_rg.contains(&d) {
                    pt_ffn.insert(s);
                }
            }
        }
        max_ffn = max_ffn.max(lp.ffn.len() + pt_ffn.len());
        cur = nxt3;
    }

    if 2 * max_heads > d_model {
        d_model = 2 * max_heads;
        d_model += d_model % 2;
    }

    Layout {
        slot_of,
        input_dims,
        erased_at,
        internal,
        internal_reglus,
        internal_lookups,
        d_model,
        n_heads: d_model / 2,
        d_ffn: max_ffn,
    }
}

/// The lookups a layer attends with, as the weight builder wants them: one
/// entry per head, carrying which value channels that head serves.
pub fn heads_of(g: &Graph, attention: &[LookUpId]) -> Vec<(LookUpId, usize)> {
    let mut out = Vec::new();
    for &lu in attention {
        for p in 0..g.lookup(lu).value_exprs.len().div_ceil(2) {
            out.push((lu, p));
        }
    }
    out
}
