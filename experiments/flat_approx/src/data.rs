//! SplitMix64 and split convention from rotate90_complex b70811e.
use crate::{algebra::singular_values, model::P};
use num_complex::Complex64;
use serde_json::{Value, json};

pub struct Random(pub u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, x: &mut [T]) {
        for end in (1..x.len()).rev() {
            let bound = (end + 1) as u64;
            let threshold = bound.wrapping_neg() % bound;
            let j = loop {
                let value = self.next();
                if value >= threshold {
                    break (value % bound) as usize;
                }
            };
            x.swap(end, j);
        }
    }
}

pub fn rows() -> Vec<u32> {
    (0..P as u32)
        .flat_map(|a| (0..P as u32).flat_map(move |b| [a, b, (a + b).pow(3) % P as u32]))
        .collect()
}

pub struct Split {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub mask: Vec<bool>,
}
impl Split {
    pub fn new(seed: u64) -> Self {
        let mut order: Vec<_> = (0..P * P).collect();
        Random(seed ^ 0x73706c6974730002).shuffle(&mut order);
        let mut mask = vec![false; P * P];
        for &i in &order[..4704] {
            mask[i] = true;
        }
        Self {
            train: order[..4704].to_vec(),
            validation: order[4704..].to_vec(),
            mask,
        }
    }
    pub fn bytes(&self, train: bool) -> Vec<u8> {
        let rows = rows();
        let indices = if train { &self.train } else { &self.validation };
        indices
            .iter()
            .flat_map(|&i| rows[3 * i..3 * i + 3].iter().flat_map(|x| x.to_le_bytes()))
            .collect()
    }
}

fn augment(i: usize, adj: &[Vec<usize>], seen: &mut [bool], owner: &mut [Option<usize>]) -> bool {
    for &j in &adj[i] {
        if seen[j] {
            continue;
        }
        seen[j] = true;
        if owner[j].is_none() || augment(owner[j].unwrap(), adj, seen, owner) {
            owner[j] = Some(i);
            return true;
        }
    }
    false
}

pub fn certificate(split: &Split, seed: u64) -> (f64, Value) {
    let mut b = vec![0.; P * P];
    let mut rng = Random(seed ^ 0x6365727469666963);
    for _ in 0..8 {
        let mut adj: Vec<Vec<_>> = (0..P)
            .map(|i| (0..P).filter(|&j| split.mask[i * P + j]).collect())
            .collect();
        for row in &mut adj {
            rng.shuffle(row);
        }
        let mut order: Vec<_> = (0..P).collect();
        rng.shuffle(&mut order);
        let mut owner = vec![None; P];
        for i in order {
            assert!(
                augment(i, &adj, &mut [false; P], &mut owner),
                "mask lacks a perfect matching"
            );
        }
        for (j, i) in owner.into_iter().enumerate() {
            b[i.unwrap() * P + j] += 1. / 8.;
        }
    }
    for i in 0..P {
        assert_eq!(b[i * P..(i + 1) * P].iter().sum::<f64>(), 1.);
        assert_eq!((0..P).map(|j| b[j * P + i]).sum::<f64>(), 1.);
    }
    assert!(b.iter().zip(&split.mask).all(|(b, m)| *m || *b == 0.));
    let three_edge = (0..P * P).all(|q| {
        if split.mask[q] {
            return true;
        }
        let (a, b) = (q / P, q % P);
        (0..P).any(|i| {
            split.mask[i * P + b] && (0..P).any(|j| split.mask[a * P + j] && split.mask[i * P + j])
        })
    });
    assert!(three_edge, "conditioning bound 4 unavailable");
    let centered: Vec<_> = b
        .iter()
        .map(|x| Complex64::new(x - 1. / P as f64, 0.))
        .collect();
    let sigma = singular_values(&centered, P)[0];
    assert!(sigma < 1.);
    let slack = (1. - sigma) / 2.;
    (
        slack,
        json!({"seed":seed,"matchings":8,"supported":true,"doubly_stochastic":true,
        "three_edge_paths":three_edge,"hidden_entry_conditioning":4,
        "estimated_normal_operator_norm":sigma,"chosen_slack":slack,
        "arithmetic_status":"FP64 SVD estimate; not a Lean execution certificate"}),
    )
}
