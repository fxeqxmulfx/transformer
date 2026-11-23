//! A single learned signed sum of repeated causal associative reads.
//! See TypedLookupPaths for shared-read saturation and tied-margin gain.
use crate::{head::PathRead, slot_memory::dot};
use std::borrow::Cow;
use std::num::NonZeroUsize;
#[path = "typed_lookup/agreement.rs"]
mod agreement;
use agreement::Agreement;
#[path = "typed_lookup/coefficients.rs"]
mod coefficients;
use coefficients::Coefficients;
#[path = "typed_lookup/forward.rs"]
mod forward;
#[path = "typed_lookup/gain.rs"]
mod gain;
#[path = "typed_lookup/paths.rs"]
mod paths;
use gain::NormalizedGain;
pub(crate) use gain::scaled_weight;
use paths::{BINS, Store, Witness};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Walk {
    Sequential,
    JumpTables,
}

pub fn parameters(d: usize) -> usize {
    BINS + d + 1
}
fn row(v: &[f32], i: usize, d: usize) -> &[f32] {
    &v[i * d..][..d]
}
fn row_mut(v: &mut [f32], i: usize, d: usize) -> &mut [f32] {
    &mut v[i * d..][..d]
}
fn add(dst: &mut [f32], x: &[f32], s: f32) {
    for (v, &b) in dst.iter_mut().zip(x) {
        *v += s * b;
    }
}

fn vote(witness: Witness, votes: &mut [f32], table: &[f32], agreement: Option<&Agreement>) {
    if let Some(rule) = agreement {
        votes[witness.position] += rule.weight(table, witness);
    } else {
        witness.vote(votes, table);
    }
}

/// Final-position reads on one prefix. Each requested budget must be positive.
pub fn final_vote_budgets(
    addresses: &[Option<usize>],
    budgets: &[usize],
    table: &[f32],
    ordered: bool,
    agreement_power: Option<u8>,
    walk: Walk,
) -> Vec<Vec<f32>> {
    assert!(table.len() == BINS && budgets.iter().all(|&n| n > 0) && budgets.is_sorted());
    let mut store = Store::new(ordered);
    for (t, &key) in addresses.iter().enumerate() {
        store.observe(key, t);
    }
    let mut votes = vec![vec![0.; addresses.len()]; budgets.len()];
    let rule = agreement_power.map(|power| Agreement::new(table, power));
    if let Some(rule) = &rule
        && ordered
        && walk == Walk::JumpTables
        && budgets.len() > 1
    {
        let mut coefficients = Coefficients::new(table, rule);
        store.visit_budgets(budgets, walk, |i, w| coefficients.vote(w, &mut votes[i]));
    } else {
        store.visit_budgets(budgets, walk, |i, w| vote(w, &mut votes[i], table, rule.as_ref()));
    }
    votes
}

pub fn inference_table(raw: &[f32], read: PathRead) -> Cow<'_, [f32]> {
    if read.agreement || read.gain_power == 0 {
        Cow::Borrowed(raw)
    } else {
        Cow::Owned(NormalizedGain::new(raw, read.gain_power).effective)
    }
}

/// Enumerate all exact role tuples sharing one signed coefficient bucket.
/// The ordinary forward keeps just the packed hash in its gradient tape.
pub(crate) fn visit_final_bucket(
    addresses: &[Option<usize>],
    budgets: &[usize],
    ordered: bool,
    bucket: usize,
    walk: Walk,
    mut found: impl FnMut(usize, [usize; 3], usize, f32),
) {
    let mut store = Store::new(ordered);
    for (t, &key) in addresses.iter().enumerate() {
        store.observe(key, t);
    }
    store.visit_roles(budgets, walk, |i, w, roles| {
        let (bin, sign) = w.weight();
        if bin == bucket {
            found(i, roles, w.position, sign);
        }
    });
}

pub(crate) fn worker_limit() -> usize {
    let limit = std::env::var("PHASOR_LOOKUP_THREADS")
        .ok()
        .and_then(|s| s.parse::<NonZeroUsize>().ok())
        .map_or(16, NonZeroUsize::get);
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(16)
        .min(limit)
}

#[derive(Default)]
struct Frame {
    witnesses: Vec<Witness>,
    terminals: Vec<(usize, f32)>,
    raw: Vec<f32>,
    norm: Option<VoteNorm>,
    gate: f32,
    derivative: f32,
}

/// EndpointScale gives the bounded energy and the derivative at zero.
#[derive(Clone, Copy)]
struct VoteNorm {
    inverse_energy: f64,
    scale: f32,
}

impl VoteNorm {
    fn new(raw: &[f32]) -> Self {
        let energy = raw.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
        let inverse_energy = (1. + energy).recip();
        Self {
            inverse_energy,
            scale: (raw.len() as f64 * inverse_energy).sqrt() as f32,
        }
    }
}

impl Frame {
    // The last multiplier stays outside the dot product in the linear
    // read, preserving its original FP32 accumulation order.
    fn gradients<'a>(&self, slope: &'a [f32]) -> (f32, Cow<'a, [f32]>, f32) {
        let Some(norm) = &self.norm else {
            return (
                dot(slope, &self.raw) * self.derivative,
                Cow::Borrowed(slope),
                self.gate,
            );
        };
        let radial: f64 = slope
            .iter()
            .zip(&self.raw)
            .map(|(&g, &v)| f64::from(g) * f64::from(v))
            .sum();
        let gate_gradient = radial as f32 * norm.scale * self.derivative;
        let value_gradient = slope
            .iter()
            .zip(&self.raw)
            .map(|(&g, &v)| {
                (f64::from(norm.scale * self.gate)
                    * (f64::from(g) - f64::from(v) * radial * norm.inverse_energy))
                    as f32
            })
            .collect();
        (gate_gradient, Cow::Owned(value_gradient), 1.)
    }
}

pub struct Tape {
    gain: Option<NormalizedGain>,
    agreement: Option<Agreement>,
    d: usize,
    length: usize,
    input: Vec<f32>,
    frames: Vec<Frame>,
}

pub(crate) enum Outputs<'a> {
    All { retain_adjoint: bool },
    Positions(&'a [usize]),
}
impl Tape {
    pub fn add_output(&self, out: &mut [f32]) {
        for (frame, dest) in self.frames.iter().zip(out.chunks_exact_mut(self.d)) {
            let scale = frame.norm.as_ref().map_or(1., |n| n.scale);
            add(dest, &frame.raw, scale * frame.gate);
        }
    }
    pub fn backward(
        &self,
        h: &[f32],
        w: &[f32],
        dy: &[f32],
        dx: &mut [f32],
        dh: &mut [f32],
        dw: &mut [f32],
    ) {
        let d = self.d;
        let scale = (d as f32).sqrt().recip();
        let (table_gradient, gates) = dw.split_at_mut(BINS);
        let mut cubic_gradient = self.gain.as_ref().map(|_| vec![0.; BINS]);
        let coordinate_gradient = cubic_gradient.as_deref_mut().unwrap_or(table_gradient);
        let mut radial = 0.;
        for (r, frame) in self.frames.iter().enumerate() {
            if frame.witnesses.is_empty() {
                continue;
            }
            let t = r % self.length;
            let base = r - t;
            let slope = row(dy, r, d);
            let (dg, value_gradient, gain) = frame.gradients(slope);
            gates[d] += dg;
            add(&mut gates[..d], row(h, r, d), scale * dg);
            add(row_mut(dh, r, d), &w[BINS..][..d], scale * dg);
            let mut effect = vec![None; t + 1];
            for &path in &frame.witnesses {
                let g = effect[path.position].get_or_insert_with(|| {
                    dot(&value_gradient, row(&self.input, base + path.position, d)) * gain
                });
                if let Some(rule) = &self.agreement {
                    rule.add_gradient(&w[..BINS], path, *g, coordinate_gradient, &mut radial);
                } else {
                    let (idx, sign) = path.weight();
                    coordinate_gradient[idx] += sign * *g;
                }
            }
            for &(pos, weight) in &frame.terminals {
                add(row_mut(dx, base + pos, d), &value_gradient, weight * gain);
            }
        }
        if let (Some(gain), Some(gradient)) = (&self.gain, cubic_gradient) {
            gain.backward(&w[..BINS], &gradient, table_gradient);
        }
        if let Some(rule) = &self.agreement {
            rule.maximum_gradient(&w[..BINS], radial, table_gradient);
        }
    }
}
#[cfg(test)]
#[path = "typed_lookup/tests.rs"]
mod tests;
