//! A single learned signed sum of repeated causal associative reads.
//! See TypedLookupPaths for shared-read saturation and tied-margin gain.
use crate::slot_memory::dot;
use std::collections::HashMap;
#[path = "typed_lookup/paths.rs"]
mod paths;
use paths::{BINS, Store, Witness};

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

fn votes(rows: usize, witnesses: &[Witness], table: &[f32]) -> Vec<f32> {
    let mut votes = vec![0.; rows];
    for &witness in witnesses {
        let (bin, sign) = witness.weight();
        votes[witness.position] += sign * table[bin];
    }
    votes
}

/// The same read at the final position of an independently supplied prefix.
pub fn final_votes(addresses: &[Option<usize>], passes: usize, table: &[f32]) -> Vec<f32> {
    assert!(passes > 0 && table.len() == BINS);
    let mut store = Store::default();
    for (t, &key) in addresses.iter().enumerate() {
        store.observe(key, t);
    }
    votes(addresses.len(), &store.witnesses(passes), table)
}

struct Frame {
    witnesses: Vec<Witness>,
    terminals: Vec<(usize, f32)>,
    raw: Vec<f32>,
    gate: f32,
    derivative: f32,
}

pub struct Tape {
    d: usize,
    length: usize,
    input: Vec<f32>,
    frames: Vec<Frame>,
}
impl Tape {
    pub fn new(
        input: &[f32],
        hidden: &[f32],
        w: &[f32],
        d: usize,
        length: usize,
        passes: usize,
        address: impl Fn(&[f32]) -> Option<usize>,
    ) -> Self {
        assert!(d > 0 && length > 0 && input.len().is_multiple_of(length * d));
        assert_eq!(input.len(), hidden.len());
        assert_eq!(w.len(), parameters(d));
        assert!(passes > 0);
        let mut tape = Self {
            d,
            length,
            input: input.to_vec(),
            frames: vec![],
        };
        if length < 3 {
            return tape;
        }
        let mut known = HashMap::<Vec<u32>, Option<usize>>::new();
        let scale = (d as f32).sqrt().recip();
        for (batch, sequence) in input.chunks_exact(d * length).enumerate() {
            let mut store = Store::default();
            for (t, vector) in sequence.chunks_exact(d).enumerate() {
                let bits: Vec<_> = vector.iter().map(|x| x.to_bits()).collect();
                let key = *known.entry(bits).or_insert_with(|| address(vector));
                store.observe(key, t);
                let witnesses = store.witnesses(passes);
                let terminals: Vec<_> = votes(t + 1, &witnesses, w)
                    .into_iter()
                    .enumerate()
                    .filter(|&(_, v)| v != 0.)
                    .collect();
                let mut raw = vec![0.; d];
                for &(pos, weight) in &terminals {
                    add(&mut raw, row(sequence, pos, d), weight);
                }
                let h = row(hidden, batch * length + t, d);
                let g = (w[BINS + d] + scale * dot(h, &w[BINS..][..d])).tanh();
                tape.frames.push(Frame {
                    witnesses,
                    terminals,
                    raw,
                    gate: 1. + g,
                    derivative: 1. - g * g,
                });
            }
        }
        tape
    }
    pub fn add_output(&self, out: &mut [f32]) {
        for (frame, dest) in self.frames.iter().zip(out.chunks_exact_mut(self.d)) {
            add(dest, &frame.raw, frame.gate);
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
        for (r, frame) in self.frames.iter().enumerate() {
            if frame.witnesses.is_empty() {
                continue;
            }
            let t = r % self.length;
            let base = r - t;
            let slope = row(dy, r, d);
            let dg = dot(slope, &frame.raw) * frame.derivative;
            dw[BINS + d] += dg;
            add(&mut dw[BINS..][..d], row(h, r, d), scale * dg);
            add(row_mut(dh, r, d), &w[BINS..][..d], scale * dg);
            let mut effect = vec![None; t + 1];
            for &path in &frame.witnesses {
                let g = effect[path.position].get_or_insert_with(|| {
                    dot(slope, row(&self.input, base + path.position, d)) * frame.gate
                });
                let (idx, sign) = path.weight();
                dw[idx] += sign * *g;
            }
            for &(pos, weight) in &frame.terminals {
                add(row_mut(dx, base + pos, d), slope, weight * frame.gate);
            }
        }
    }
}
#[cfg(test)]
#[path = "typed_lookup/tests.rs"]
mod tests;
