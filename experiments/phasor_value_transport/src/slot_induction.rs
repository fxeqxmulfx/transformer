//! A causal induction read over the actual normalized dense input sequence.
//! DenseInduction.latest and prefix_congr cover the backward search.
//! The original shared gate selects this value when a past address matches.
use crate::slot_memory::dot;

mod index;

fn accepts(score: f32, left_energy: f32, right_energy: f32) -> bool {
    score > 0. && 2. * score * score > left_energy * right_energy
}

fn latest(input: &[f32], energy: &[f32], width: usize, start: usize, row: usize) -> Option<usize> {
    let query = &input[row * width..][..width];
    (start..row.saturating_sub(1)).rev().find_map(|past| {
        let score = dot(query, &input[past * width..][..width]);
        accepts(score, energy[row], energy[past]).then_some(past + 1)
    })
}

pub struct Prefix {
    input: Vec<f32>,
    donors: Vec<Option<usize>>,
    width: usize,
}

impl Prefix {
    pub fn new(input: Vec<f32>, width: usize, length: usize) -> Self {
        assert!(width > 0 && length > 0 && input.len().is_multiple_of(width * length));
        let energy: Vec<_> = input.chunks_exact(width).map(|v| dot(v, v)).collect();
        let donors = if length < 96 {
            (0..energy.len())
                .map(|r| latest(&input, &energy, width, r - r % length, r))
                .collect()
        } else {
            index::donors(&input, &energy, width, length)
        };
        Self {
            input,
            donors,
            width,
        }
    }

    pub fn source(&self, row: usize) -> Option<usize> {
        self.donors[row]
    }

    pub fn read(&self, row: usize) -> Option<(usize, &[f32])> {
        let source = self.source(row)?;
        let start = source * self.width;
        Some((source, &self.input[start..][..self.width]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_mixture_differentiates_the_earlier_dense_value_and_shared_gate() {
        use crate::{
            head::{Head, PeerRead},
            slot_memory::{self, Layout, Routing},
            slot_orbits::ABSENT,
        };
        let mut cfg = crate::config(1).unwrap();
        cfg.head = Head::PeerTriplePhase(PeerRead::DENSE_INDUCTION);
        cfg.width = 4;
        cfg.vocab = 4;
        let l = Layout::new(&cfg);
        let route = Routing::new(cfg.head);
        // All permanent slots abstain. The input values are off-dictionary.
        let cells = [ABSENT; 4];
        let x = vec![
            2., 0.1, 0., 0., 0., 1.4, 0.15, 0., 0., 0., -3., 0., 1.1, 0.05, 0., 0.,
        ];
        let mut h = vec![0.; 16];
        h[12] = 0.3;
        let mut w = vec![0.; l.end()];
        w[l.bias + 2] = 0.2;
        w[l.gates + 2 * l.d] = 0.1;
        let prefix = Prefix::new(x.clone(), 4, 4);
        let (rows, out) = slot_memory::forward(&l, &w, &route, &cells, &h, 4, Some(&prefix));
        assert_eq!(prefix.source(3), Some(1));
        assert!(out[13] > 0.5);
        assert_eq!(
            slot_memory::forward(&l, &w, &route, &cells, &h, 4, None).1[13],
            0.
        );
        let mut dy = vec![0.; 16];
        dy[13] = 1.;
        let gradient = slot_memory::backward(&l, &w, &h, &rows, &dy, 4, Some(&prefix));
        let dw = gradient.parameters.dense(&l);
        let loss = |input: Vec<f32>, state: &[f32], weights: &[f32]| {
            let prefix = Prefix::new(input, 4, 4);
            slot_memory::forward(&l, weights, &route, &cells, state, 4, Some(&prefix)).1[13]
        };
        for (which, index, analytic) in [
            (0, 5, gradient.input[5]),
            (0, 1, gradient.input[1]),
            (1, 12, gradient.hidden[12]),
            (2, l.bias + 2, dw[l.bias + 2]),
            (2, l.gates + 2 * l.d, dw[l.gates + 2 * l.d]),
        ] {
            let value = |delta: f32| {
                let (mut input, mut state, mut weights) = (x.clone(), h.clone(), w.clone());
                match which {
                    0 => input[index] += delta,
                    1 => state[index] += delta,
                    _ => weights[index] += delta,
                }
                loss(input, &state, &weights)
            };
            let numerical = (value(0.001) - value(-0.001)) / 0.002;
            assert!(
                (analytic - numerical).abs() < 0.0001,
                "{which} {index} {analytic} {numerical}"
            );
        }
    }

    #[test]
    fn most_recent_angular_match_uses_a_completed_value_from_the_same_sequence() {
        let data = [
            [2., 0.],
            [0., 3.],
            [-1., 0.],
            [1., 0.2],
            [0., -4.],
            [-1., 0.],
            [1., 0.1],
            [1., 0.1],
            [0., 0.],
            [0., -1.],
            [-1., 0.],
            [0., 2.],
            [0., 0.],
            [0., 3.],
        ];
        let prefix = Prefix::new(data.concat(), 2, 7);
        for (row, donor) in [
            (0, None),
            (1, None),
            (2, None),
            (3, Some(1)),
            (6, Some(4)),
            (7, None),
            (8, None),
            (9, None),
            (12, None),
            (13, Some(12)),
        ] {
            assert_eq!(prefix.source(row), donor);
            assert_eq!(prefix.read(row), donor.map(|i| (i, data[i].as_slice())));
        }
    }
}
