//! Consecutive rows can share raw votes on their identical dense input prefixes.
//! A worker may begin inside a sequence after replaying its earlier writes.
use super::*;
use std::collections::HashMap;
use std::ops::Range;

struct Batch<'a> {
    input: &'a [f32],
    hidden: &'a [f32],
    w: &'a [f32],
    table: &'a [f32],
    agreement: Option<&'a Agreement>,
    keys: Vec<Option<usize>>,
    shared: Vec<usize>,
    d: usize,
    length: usize,
    read: PathRead,
    retain: bool,
}

impl Batch<'_> {
    /// Only the emission is lazy. Every earlier write reaches the store at
    /// the row's requested position, independently of the other sequences.
    fn frames_at(&self, positions: &[usize]) -> Vec<Frame> {
        let mut frames: Vec<_> = std::iter::repeat_with(Frame::default)
            .take(self.keys.len())
            .collect();
        for (b, &t) in positions.iter().enumerate() {
            let base = b * self.length;
            let mut store = Store::new(self.read.ordered);
            for (i, &key) in self.keys[base..=base + t].iter().enumerate() {
                store.observe(key, i);
            }
            let walk = if cfg!(feature = "indexed-final-reads") {
                Walk::JumpTables
            } else {
                Walk::Sequential
            };
            frames[base + t] = self.frame(base, t, &store, false, walk);
        }
        frames
    }

    fn frame(&self, base: usize, t: usize, store: &Store, reuse: bool, walk: Walk) -> Frame {
        let d = self.d;
        let mut witnesses = Vec::new();
        let mut terminals = Vec::new();
        let (raw, norm) = if reuse {
            (Vec::new(), None)
        } else {
            let mut votes = vec![0.; t + 1];
            store.visit_budgets(&[self.read.passes], walk, |_, path| {
                vote(path, &mut votes, self.table, self.agreement);
                if self.retain {
                    witnesses.push(path);
                }
            });
            let mut raw = vec![0.; d];
            for (pos, weight) in votes.into_iter().enumerate().filter(|&(_, v)| v != 0.) {
                add(&mut raw, row(self.input, base + pos, d), weight);
                if self.retain {
                    terminals.push((pos, weight));
                }
            }
            let norm = self.read.normalize_vote.then(|| VoteNorm::new(&raw));
            (raw, norm)
        };
        let scale = (d as f32).sqrt().recip();
        let h = row(self.hidden, base + t, d);
        let g = (self.w[BINS + d] + scale * dot(h, &self.w[BINS..][..d])).tanh();
        Frame {
            witnesses,
            terminals,
            raw,
            norm,
            gate: 1. + g,
            derivative: 1. - g * g,
        }
    }

    fn block(&self, rows: Range<usize>) -> Vec<Frame> {
        let mut frames = Vec::with_capacity(rows.len());
        let first = rows.start / self.length;
        let last = (rows.end - 1) / self.length;
        for (i, &common) in self.shared[first..=last].iter().enumerate() {
            let base = (first + i) * self.length;
            let start = rows.start.saturating_sub(base);
            let stop = (rows.end - base).min(self.length);
            let mut store = Store::new(self.read.ordered);
            for (t, &key) in self.keys[base..base + stop].iter().enumerate() {
                store.observe(key, t);
                if t >= start {
                    frames.push(self.frame(base, t, &store, t < common, Walk::Sequential));
                }
            }
        }
        frames
    }

    fn frames(&self, limit: usize) -> Vec<Frame> {
        let rows = self.keys.len();
        if rows == 0 {
            return Vec::new();
        }
        let span = if self.retain && self.length >= 32 {
            // Each sequence starts with an empty association store. Keep its
            // entire adjoint tape on one worker and join in minibatch order.
            let sequences = rows / self.length;
            sequences.div_ceil(limit) * self.length
        } else if !self.retain && self.length >= 96 {
            rows.div_ceil(limit.min(rows.div_ceil(64)))
        } else {
            rows
        };
        let mut frames = if span >= rows {
            self.block(0..rows)
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..rows)
                    .step_by(span)
                    .map(|start| scope.spawn(move || self.block(start..(start + span).min(rows))))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("lookup worker panicked"))
                    .collect()
            })
        };
        // Each reused frame already has its own row's contextual gate.
        for (b, &count) in self
            .shared
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, n)| **n > 0)
        {
            let (past, following) = frames.split_at_mut(b * self.length);
            for (dest, src) in following[..count]
                .iter_mut()
                .zip(&past[(b - 1) * self.length..])
            {
                dest.raw.clone_from(&src.raw);
                dest.norm = src.norm;
            }
        }
        frames
    }
}

impl Tape {
    #[cfg(test)]
    pub fn new(
        input: &[f32],
        hidden: &[f32],
        w: &[f32],
        shape: (usize, usize),
        read: PathRead,
        retain_adjoint: bool,
        address: impl Fn(&[f32]) -> Option<usize>,
    ) -> Self {
        Self::with_outputs(
            input,
            hidden,
            w,
            shape,
            read,
            Outputs::All { retain_adjoint },
            address,
        )
    }

    pub(crate) fn with_outputs(
        input: &[f32],
        hidden: &[f32],
        w: &[f32],
        shape: (usize, usize),
        read: PathRead,
        outputs: Outputs<'_>,
        address: impl Fn(&[f32]) -> Option<usize>,
    ) -> Self {
        let (d, length) = shape;
        let (retain_adjoint, positions) = match outputs {
            Outputs::All { retain_adjoint } => (retain_adjoint, None),
            Outputs::Positions(positions) => (false, Some(positions)),
        };
        assert!(d > 0 && length > 0 && input.len().is_multiple_of(length * d));
        if let Some(positions) = positions {
            assert!(
                positions.len() == input.len() / (length * d)
                    && positions.iter().all(|&t| t < length)
            );
        }
        assert_eq!(input.len(), hidden.len());
        assert_eq!(w.len(), parameters(d));
        assert!(read.passes > 0);
        let mut tape = Self {
            gain: (!read.agreement && read.gain_power > 0 && length >= 3)
                .then(|| NormalizedGain::new(&w[..BINS], read.gain_power)),
            agreement: (read.agreement && length >= 3)
                .then(|| Agreement::new(&w[..BINS], read.gain_power)),
            d,
            length,
            input: if retain_adjoint && length >= 3 {
                input.to_vec()
            } else {
                Vec::new()
            },
            frames: Vec::new(),
        };
        if length < 3 {
            return tape;
        }
        let table = tape
            .gain
            .as_ref()
            .map_or(&w[..BINS], |g| g.effective.as_slice());
        let mut known = HashMap::new();
        let keys = input
            .chunks_exact(d)
            .map(|vector| {
                let bits: Vec<_> = vector.iter().map(|x| x.to_bits()).collect();
                *known.entry(bits).or_insert_with(|| address(vector))
            })
            .collect();
        let stride = length * d;
        let mut shared = vec![0; input.len() / stride];
        if !retain_adjoint && positions.is_none() {
            for (i, (prev, next)) in input
                .chunks_exact(stride)
                .zip(input.chunks_exact(stride).skip(1))
                .enumerate()
            {
                shared[i + 1] = prev
                    .chunks_exact(d)
                    .zip(next.chunks_exact(d))
                    .take_while(|(a, b)| a.iter().zip(*b).all(|(x, y)| x.to_bits() == y.to_bits()))
                    .count();
            }
        }
        let batch = Batch {
            input,
            hidden,
            w,
            table,
            agreement: tape.agreement.as_ref(),
            keys,
            shared,
            d,
            length,
            read,
            retain: retain_adjoint,
        };
        tape.frames = match positions {
            Some(positions) => batch.frames_at(positions),
            None => batch.frames(worker_limit()),
        };
        tape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joining_retained_sequences_preserves_witnesses_and_all_three_pullbacks() {
        let (d, length, sequences) = (9, 43, 7);
        let rows = sequences * length;
        let keys: Vec<_> = (0..rows)
            .map(|i| (i % 17 != 0).then_some((i * 7 + i / 13 + i / length) % d))
            .collect();
        let x: Vec<_> = keys
            .iter()
            .flat_map(|&key| (0..d).map(move |j| f32::from(key == Some(j))))
            .collect();
        let hidden: Vec<_> = (0..x.len())
            .map(|i| ((i * 17 % 83) as f32 - 40.) / 61.)
            .collect();
        let slope: Vec<_> = (0..x.len())
            .map(|i| ((i * 53 % 97) as f32 - 48.) / 73.)
            .collect();
        let w: Vec<_> = (0..parameters(d))
            .map(|i| ((i * 17 % 113) as f32 - 50.) / 83.)
            .collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (ordered, paired) in [(false, false), (true, false), (false, true), (true, true)] {
            let read = PathRead {
                passes: 10,
                gain_power: 4,
                agreement: paired,
                ordered,
                ..PathRead::NORMALIZED
            };
            let gain = NormalizedGain::new(&w[..BINS], read.gain_power);
            let agreement = paired.then(|| Agreement::new(&w[..BINS], read.gain_power));
            let batch = Batch {
                input: &x,
                hidden: &hidden,
                w: &w,
                table: if paired { &w[..BINS] } else { &gain.effective },
                agreement: agreement.as_ref(),
                keys: keys.clone(),
                shared: vec![0; sequences],
                d,
                length,
                read,
                retain: true,
            };
            let tape = |limit| Tape {
                gain: (!paired).then(|| NormalizedGain::new(&w[..BINS], read.gain_power)),
                agreement: paired.then(|| Agreement::new(&w[..BINS], read.gain_power)),
                d,
                length,
                input: x.clone(),
                frames: batch.frames(limit),
            };
            let run = |t: &Tape| {
                let mut out = hidden.clone();
                let mut dx = vec![0.; x.len()];
                let mut dh = dx.clone();
                let mut dw = vec![0.; w.len()];
                t.add_output(&mut out);
                t.backward(&hidden, &w, &slope, &mut dx, &mut dh, &mut dw);
                assert!([&dx, &dh, &dw].iter().all(|v| v.iter().any(|&g| g != 0.)));
                [bits(&out), bits(&dx), bits(&dh), bits(&dw)]
            };
            let serial = tape(1);
            let expected = run(&serial);
            for limit in [2, 5] {
                let joined = tape(limit);
                assert_eq!(joined.frames.len(), serial.frames.len());
                for (a, b) in joined.frames.iter().zip(&serial.frames) {
                    assert_eq!(a.witnesses, b.witnesses);
                    let terminals = |f: &Frame| {
                        f.terminals
                            .iter()
                            .map(|(j, w)| (*j, w.to_bits()))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(terminals(a), terminals(b));
                    assert_eq!(bits(&a.raw), bits(&b.raw));
                    let norm = |f: &Frame| {
                        f.norm
                            .map(|n| (n.inverse_energy.to_bits(), n.scale.to_bits()))
                    };
                    assert_eq!(norm(a), norm(b));
                    assert_eq!(a.gate.to_bits(), b.gate.to_bits());
                    assert_eq!(a.derivative.to_bits(), b.derivative.to_bits());
                }
                assert_eq!(run(&joined), expected);
            }
        }
    }
}
