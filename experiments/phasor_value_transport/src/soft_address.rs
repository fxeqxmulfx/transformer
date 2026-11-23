//! Shared causal memory loop with independent continuous read and write addresses.
//! Each pass uses its original query, a gated prior message, and the strength
//! of its present match. See TripleTransport.MessageLoop and FractionalWriteGate.
use crate::slot_memory::dot;
pub mod audit;
mod cells;
pub mod index_audit;
mod planar;
use cells::{Cells, Read, Write};
fn add(out: &mut [f32], x: &[f32], s: f32) {
    for (a, b) in out.iter_mut().zip(x) {
        *a += s * b;
    }
}
fn row(x: &[f32], r: usize, d: usize) -> &[f32] {
    &x[r * d..][..d]
}
fn row_mut(x: &mut [f32], r: usize, d: usize) -> &mut [f32] {
    &mut x[r * d..][..d]
}
fn sigmoid(x: f32) -> f32 {
    0.5 + 0.5 * (0.5 * x).tanh()
}
#[derive(Clone, Copy)]
#[repr(usize)]
enum Map {
    WriteKey,
    WriteRelation,
    ReadKey,
    RelationQuery,
    RelationMessage,
    MessageQuery,
    MessageState,
    AcceptQuery,
    AcceptState,
    WriteGate,
    Value,
    Out,
}
use Map::*;
const MAPS: usize = 12;
const CONTEXT: usize = 0;
const WRITE_BIAS: usize = 1;
const MESSAGE_BIAS: usize = 2;
const ACCEPT_BIAS: usize = 3;
const STRENGTH: usize = 4;
const OUT_BIAS: usize = 5;
pub fn parameters(d: usize) -> usize {
    MAPS * d + 6
}
pub fn initialize(w: &mut [f32], d: usize, seed: u32) {
    let mut rng = crate::random::Random::new(seed.wrapping_add(81_337));
    for v in &mut w[..5 * d] {
        *v = rng.gauss(0., 1.) as f32;
    }
    w[MAPS * d + CONTEXT] = 1.;
    w[MAPS * d + STRENGTH] = 1.;
}
fn projection(w: &[f32], map: Map, x: &[f32]) -> f32 {
    dot(row(w, map as usize, x.len()), x)
}
fn affine_grad(w: &[f32], dw: &mut [f32], map: Map, x: &[f32], dx: &mut [f32], s: f32) {
    add(row_mut(dw, map as usize, x.len()), x, s);
    add(dx, row(w, map as usize, x.len()), s);
}
struct Pass {
    state: Vec<f32>,
    read: Read,
    message: f32,
    relation: f32,
    accept: f32,
}
struct Frame {
    write: Option<Write>,
    passes: Vec<Pass>,
    delta: Vec<f32>,
    output_gate: f32,
    write_relation: f32,
}
pub struct Tape {
    d: usize,
    length: usize,
    query: Vec<f32>,
    frames: Vec<Frame>,
}
impl Tape {
    pub fn new(
        input: &[f32],
        hidden: &[f32],
        w: &[f32],
        d: usize,
        length: usize,
        budget: usize,
        retain: bool,
    ) -> Self {
        let hscale = (d as f32).sqrt();
        let bias = &w[MAPS * d..];
        let query: Vec<_> = input
            .iter()
            .zip(hidden)
            .map(|(x, h)| x + bias[CONTEXT] * h)
            .collect();
        let mut frames = Vec::new();
        let mut memory = Cells::new(d + 1);
        for (r, q) in query.chunks_exact(d).enumerate() {
            let t = r % length;
            if t == 0 {
                memory.reset();
            }
            let h = row(hidden, r, d);
            let mut frame = Frame {
                write: None,
                passes: Vec::new(),
                delta: vec![0.; d],
                output_gate: 0.,
                write_relation: 0.,
            };
            if t >= 2 {
                let rel = (projection(w, WriteRelation, row(&query, r - 1, d)) / hscale).tanh();
                let addr = projection(w, WriteKey, row(&query, r - 2, d)) + hscale * rel;
                let value_gate = sigmoid(bias[WRITE_BIAS] + projection(w, WriteGate, h) / hscale);
                let mut target = row(input, r, d).to_vec();
                for ((x, g), v) in target.iter_mut().zip(h).zip(row(w, Value as usize, d)) {
                    *x += g * v;
                }
                target.push(1.); // The interpolated write strength is another memory channel.
                frame.write = Write::apply(&mut memory, addr, &target, value_gate);
                frame.write_relation = rel;
                let mut state = q.to_vec();
                let key = projection(w, ReadKey, q);
                let rel0 = projection(w, RelationQuery, q);
                let msg0 = bias[MESSAGE_BIAS] + projection(w, MessageQuery, q) / hscale;
                let accept0 = bias[ACCEPT_BIAS] + projection(w, AcceptQuery, q) / hscale;
                for _ in 0..budget {
                    let message = sigmoid(msg0 + projection(w, MessageState, &state) / hscale);
                    let relation =
                        ((rel0 + projection(w, RelationMessage, &state)) / hscale).tanh();
                    let coord = key + message * projection(w, ReadKey, &state) + hscale * relation;
                    let timing = (!retain
                        && t + 1 == length
                        && index_audit::enabled()
                        && !index_audit::hard())
                    .then(std::time::Instant::now);
                    let read = Read::at(&memory, coord);
                    if let Some(start) = timing {
                        memory.time_last(coord, read.quality, start.elapsed().as_nanos());
                    }
                    if !retain && t + 1 == length {
                        audit::record(read.quality);
                    }
                    let accept = sigmoid(
                        accept0
                            + projection(w, AcceptState, &state) / hscale
                            + bias[STRENGTH] * read.value[d],
                    );
                    let next: Vec<_> = state
                        .iter()
                        .zip(&read.value)
                        .map(|(a, b)| a + accept * (b - a))
                        .collect();
                    if retain {
                        frame.passes.push(Pass {
                            state,
                            read,
                            message,
                            relation,
                            accept,
                        });
                    }
                    state = next;
                }
                frame.delta = state.iter().zip(q).map(|(a, b)| a - b).collect();
                frame.output_gate = (bias[OUT_BIAS] + projection(w, Out, h) / hscale).tanh();
                if !retain {
                    frame.write = None;
                }
            }
            frames.push(frame);
        }
        Self {
            d,
            length,
            query: if retain { query } else { Vec::new() },
            frames,
        }
    }
    pub fn add_output(&self, out: &mut [f32]) {
        for (r, dest) in self.frames.iter().zip(out.chunks_exact_mut(self.d)) {
            add(dest, &r.delta, r.output_gate);
        }
    }
    pub fn backward(
        &self,
        hidden: &[f32],
        w: &[f32],
        dy: &[f32],
        dx: &mut [f32],
        dh: &mut [f32],
        dw: &mut [f32],
    ) {
        let d = self.d;
        let bias = MAPS * d;
        let scale = (d as f32).sqrt().recip();
        let mut dquery = vec![0.; self.query.len()];
        let mut memory = Cells::new(d + 1);
        for (r, frame) in self.frames.iter().enumerate().rev() {
            let t = r % self.length;
            if t + 1 == self.length {
                memory.reset();
            }
            if t < 2 {
                continue;
            }
            let h = row(hidden, r, d);
            let q = row(&self.query, r, d);
            let slope = row(dy, r, d);
            let dout = dot(slope, &frame.delta) * (1. - frame.output_gate * frame.output_gate);
            dw[bias + OUT_BIAS] += dout;
            affine_grad(w, dw, Out, h, row_mut(dh, r, d), scale * dout);
            let mut dstate: Vec<_> = slope.iter().map(|y| y * frame.output_gate).collect();
            let dq = row_mut(&mut dquery, r, d);
            add(dq, &dstate, -1.);
            for pass in frame.passes.iter().rev() {
                let s = &pass.state;
                let a = pass.accept;
                let da = dstate
                    .iter()
                    .zip(&pass.read.value)
                    .zip(s)
                    .map(|((g, v), prev)| g * (v - prev))
                    .sum::<f32>()
                    * a
                    * (1. - a);
                let mut dval: Vec<_> = dstate.iter().map(|g| g * a).collect();
                dval.push(da * w[bias + STRENGTH]);
                let address = pass.read.backward(&mut memory, &dval);
                for g in &mut dstate {
                    *g *= 1. - a;
                }
                dw[bias + ACCEPT_BIAS] += da;
                dw[bias + STRENGTH] += da * pass.read.value[d];
                affine_grad(w, dw, AcceptQuery, q, dq, scale * da);
                affine_grad(w, dw, AcceptState, s, &mut dstate, scale * da);
                let dm = address * projection(w, ReadKey, s) * pass.message * (1. - pass.message);
                let drel = address * (1. - pass.relation * pass.relation);
                dw[bias + MESSAGE_BIAS] += dm;
                affine_grad(w, dw, MessageQuery, q, dq, scale * dm);
                affine_grad(w, dw, MessageState, s, &mut dstate, scale * dm);
                affine_grad(w, dw, RelationQuery, q, dq, drel);
                affine_grad(w, dw, RelationMessage, s, &mut dstate, drel);
                affine_grad(w, dw, ReadKey, q, dq, address);
                affine_grad(w, dw, ReadKey, s, &mut dstate, address * pass.message);
            }
            add(dq, &dstate, 1.);
            if let Some(write) = &frame.write {
                let (ds, dg, dvalue) = write.backward(&mut memory);
                dw[bias + WRITE_BIAS] += dg;
                affine_grad(w, dw, WriteGate, h, row_mut(dh, r, d), scale * dg);
                for j in 0..d {
                    dx[r * d + j] += dvalue[j];
                    dw[Value as usize * d + j] += h[j] * dvalue[j];
                    dh[r * d + j] += w[Value as usize * d + j] * dvalue[j];
                }
                affine_grad(
                    w,
                    dw,
                    WriteKey,
                    row(&self.query, r - 2, d),
                    row_mut(&mut dquery, r - 2, d),
                    ds,
                );
                affine_grad(
                    w,
                    dw,
                    WriteRelation,
                    row(&self.query, r - 1, d),
                    row_mut(&mut dquery, r - 1, d),
                    ds * (1. - frame.write_relation * frame.write_relation),
                );
            }
        }
        for (j, &g) in dquery.iter().enumerate() {
            dx[j] += g;
            dh[j] += g * w[bias + CONTEXT];
            dw[bias + CONTEXT] += g * hidden[j];
        }
    }
}
#[cfg(test)]
mod tests;
