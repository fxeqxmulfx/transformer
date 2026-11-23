//! Consecutive rows can share raw votes on their identical dense input prefixes.
//! A worker may begin inside a sequence after replaying its earlier writes.
use super::*;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::ops::Range;

struct Batch<'a> {
    input: &'a [f32],
    hidden: &'a [f32],
    w: &'a [f32],
    table: &'a [f32],
    keys: Vec<Option<usize>>,
    shared: Vec<usize>,
    d: usize,
    length: usize,
    read: PathRead,
    retain: bool,
}

impl Batch<'_> {
    fn frame(&self, base: usize, t: usize, store: &Store, reuse: bool) -> Frame {
        let d = self.d;
        let mut witnesses = Vec::new();
        let mut terminals = Vec::new();
        let (raw, norm) = if reuse {
            (Vec::new(), None)
        } else {
            let mut votes = vec![0.; t + 1];
            store.visit(self.read.passes, |path| {
                path.vote(&mut votes, self.table);
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
                    frames.push(self.frame(base, t, &store, t < common));
                }
            }
        }
        frames
    }

    fn frames(&self) -> Vec<Frame> {
        let rows = self.keys.len();
        if rows == 0 {
            return Vec::new();
        }
        let workers = if !self.retain && self.length >= 96 {
            let limit = std::env::var("PHASOR_LOOKUP_THREADS")
                .ok()
                .and_then(|s| s.parse::<NonZeroUsize>().ok())
                .map_or(16, NonZeroUsize::get);
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .min(16)
                .min(limit)
                .min(rows.div_ceil(64))
        } else {
            1
        };
        let mut frames = if workers == 1 {
            self.block(0..rows)
        } else {
            let span = rows.div_ceil(workers);
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
    pub fn new(
        input: &[f32],
        hidden: &[f32],
        w: &[f32],
        shape: (usize, usize),
        read: PathRead,
        retain_adjoint: bool,
        address: impl Fn(&[f32]) -> Option<usize>,
    ) -> Self {
        let (d, length) = shape;
        assert!(d > 0 && length > 0 && input.len().is_multiple_of(length * d));
        assert_eq!(input.len(), hidden.len());
        assert_eq!(w.len(), parameters(d));
        assert!(read.passes > 0);
        let mut tape = Self {
            gain: (read.gain_power > 0 && length >= 3)
                .then(|| NormalizedGain::new(&w[..BINS], read.gain_power)),
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
        if !retain_adjoint {
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
        tape.frames = Batch {
            input,
            hidden,
            w,
            table,
            keys,
            shared,
            d,
            length,
            read,
            retain: retain_adjoint,
        }
        .frames();
        tape
    }
}
