//! Training-only bases and stable projected feature energies; RD2 and RD4.
use crate::{algebra::linear, gradient::energy, svd};
use std::collections::BTreeMap;

pub const RANKS: [usize; 3] = [16, 32, 64];

pub struct Basis {
    pub requested: usize,
    pub achieved: usize,
    pub q: Vec<f64>,
}
pub struct Group {
    pub key: usize,
    pub label: usize,
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub center: Vec<f64>,
    pub bases: Vec<Basis>,
}

pub fn selected(x: &[f64], indices: &[usize], width: usize) -> Vec<f64> {
    indices
        .iter()
        .flat_map(|&i| x[i * width..(i + 1) * width].iter().copied())
        .collect()
}
pub fn displacement(x: &[f64], indices: &[usize], center: &[f64]) -> Vec<f64> {
    let width = center.len();
    indices
        .iter()
        .flat_map(|&i| (0..width).map(move |k| x[i * width + k] - center[k]))
        .collect()
}
pub fn contrasts(e: &[f64], label: usize, width: usize) -> Vec<f64> {
    (0..e.len() / width)
        .filter(|&c| c != label)
        .flat_map(|c| (0..width).map(move |k| e[c * width + k] - e[label * width + k]))
        .collect()
}
pub fn project(x: &[f64], q: &[f64], width: usize) -> Vec<f64> {
    assert_eq!(x.len() % width, 0);
    linear(x, x.len() / width, width, q, width)
}
pub fn projector(right: &[f64], rank: usize, width: usize) -> Vec<f64> {
    let transposed: Vec<_> = (0..width)
        .flat_map(|k| (0..rank).map(move |j| right[j * width + k]))
        .collect();
    let gram = linear(&transposed, width, rank, &transposed, width);
    gram.iter()
        .enumerate()
        .map(|(i, v)| f64::from(i / width == i % width) - v)
        .collect()
}

pub fn groups(
    states: &[f64],
    training: &[(usize, usize)],
    width: usize,
    orbit: bool,
) -> Vec<Group> {
    const P: usize = 97;
    assert_eq!(states.len(), P * P * width);
    let mut observed = vec![None; P * P];
    for &(i, label) in training {
        assert!(observed[i].replace(label).is_none());
    }
    let mut sets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..P * P {
        let s = (i / P + i % P) % P;
        let key = if orbit {
            s.min(35 * s % P).min(61 * s % P)
        } else {
            s
        };
        sets.entry(key).or_default().push(i);
    }
    sets.into_iter()
        .map(|(key, indices)| {
            let train: Vec<_> = indices
                .iter()
                .copied()
                .filter(|&i| observed[i].is_some())
                .collect();
            assert!(!train.is_empty(), "missing observed group");
            let anchor = train[0];
            let label = observed[anchor].unwrap();
            assert!(
                train.iter().all(|&i| observed[i] == Some(label)),
                "inconsistent training labels"
            );
            let anchor_sum = (anchor / P + anchor % P) % P;
            for &i in &indices {
                let sum = (i / P + i % P) % P;
                let allowed: &[usize] = if orbit { &[1, 35, 61] } else { &[1] };
                let u = *allowed
                    .iter()
                    .find(|&&u| u * anchor_sum % P == sum)
                    .expect("missing route");
                assert_eq!(u.pow(3) % P, 1);
                let shift = (i / P + P - u * (anchor / P) % P) % P;
                assert_eq!((u * (anchor / P) + shift) % P, i / P);
                assert_eq!((u * (anchor % P) + P - shift) % P, i % P);
            }
            let mut center = vec![0.; width];
            for &i in &train {
                for k in 0..width {
                    center[k] += states[i * width + k] / train.len() as f64;
                }
            }
            let x = displacement(states, &train, &center);
            let s = svd::thin(&x, train.len(), width);
            assert_eq!(s.left.len(), train.len() * s.values.len());
            let available = s
                .values
                .iter()
                .filter(|&&v| v > s.values[0] * 1e-10)
                .count();
            let bases = RANKS
                .iter()
                .map(|&requested| {
                    let achieved = requested.min(available);
                    Basis {
                        requested,
                        achieved,
                        q: projector(&s.right, achieved, width),
                    }
                })
                .collect();
            Group {
                key,
                label,
                train,
                validation: indices
                    .into_iter()
                    .filter(|&i| observed[i].is_none())
                    .collect(),
                center,
                bases,
            }
        })
        .collect()
}

pub struct FeatureMoment {
    weighted_right: Vec<f64>,
    width: usize,
}
impl FeatureMoment {
    pub fn new(x: &[f64], width: usize) -> Self {
        let n = x.len() / width;
        let s = svd::thin(x, n, width);
        // All singular directions are retained, with no truncation or subtraction of traces.
        let weighted_right = s
            .right
            .iter()
            .enumerate()
            .map(|(i, v)| v * s.values[i / width] / (n as f64).sqrt())
            .collect();
        Self {
            weighted_right,
            width,
        }
    }
    pub fn residual_energy(&self, q: &[f64]) -> f64 {
        energy(&project(&self.weighted_right, q, self.width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_svd_moment_matches_direct_features_for_nonorthogonal_maps() {
        let x = [1., 2., -0.5, 3., 2., -1.];
        let q = [0.2, -0.7, 0.3, 1.1];
        let actual = FeatureMoment::new(&x, 2).residual_energy(&q);
        assert!((actual - energy(&project(&x, &q, 2)) / 3.).abs() < 1e-12);
    }
}
