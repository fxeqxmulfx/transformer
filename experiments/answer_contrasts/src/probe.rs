//! Fixed-design indicator probes; pooling and affine mass correction from AC1/AC3.
use crate::{
    algebra::{linear, ridge},
    data::Split,
    model::{Forward, P, W},
};

pub const D: usize = W + 1;

pub struct Probe {
    pub dim: usize,
    pub weights: Vec<f64>,
    pub values: Vec<f64>,
    pub penalty: f64,
}

pub fn design(out: &Forward) -> Vec<f64> {
    out.states
        .as_chunks::<W>()
        .0
        .iter()
        .flat_map(|s| s.iter().copied().chain([1.]))
        .collect()
}

pub fn fit(design: &[f64], split: &Split, target: &[usize], reference: &[f64]) -> Probe {
    let dim = reference.len();
    assert!((reference.iter().sum::<f64>() - 1.).abs() < 1e-12);
    let train_x: Vec<_> = split
        .train
        .iter()
        .flat_map(|&q| design[q * D..(q + 1) * D].iter().copied())
        .collect();
    let train_y: Vec<_> = split
        .train
        .iter()
        .flat_map(|&q| (0..dim).map(move |j| f64::from(j == target[q])))
        .collect();
    let (mut weights, penalty) = ridge(&train_x, &train_y, split.train.len(), D, dim);
    // Affine correction is applied to the coefficients, so all predictions
    // and the source-alignment expansion use exactly the same affine map.
    for k in 0..D {
        let correction = f64::from(k == W) - (0..dim).map(|j| weights[j * D + k]).sum::<f64>();
        for j in 0..dim {
            weights[j * D + k] += correction * reference[j];
        }
    }
    let values = linear(design, P * P, D, &weights, dim);
    Probe {
        dim,
        weights,
        values,
        penalty,
    }
}

pub fn pool(rows: &[f64], old_dim: usize, mapping: &[usize], new_dim: usize) -> Vec<f64> {
    assert_eq!(mapping.len(), old_dim);
    assert_eq!(rows.len() % old_dim, 0);
    let mut pooled = vec![0.; rows.len() / old_dim * new_dim];
    for (q, row) in rows.chunks_exact(old_dim).enumerate() {
        for (j, &value) in row.iter().enumerate() {
            pooled[q * new_dim + mapping[j]] += value;
        }
    }
    pooled
}

pub fn decoder(
    out: &Forward,
    split: &Split,
    target: &[usize],
    dim: usize,
) -> (Vec<f64>, Vec<usize>) {
    let mut coefficients = vec![0.; P * dim];
    let mut counts = vec![0; dim];
    for &q in &split.train {
        counts[target[q]] += 1;
        let center = out.raw[q * P..(q + 1) * P].iter().sum::<f64>() / P as f64;
        for v in 0..P {
            coefficients[v * dim + target[q]] += out.raw[q * P + v] - center;
        }
    }
    assert!(counts.iter().all(|&n| n > 0));
    for v in 0..P {
        for j in 0..dim {
            coefficients[v * dim + j] /= counts[j] as f64;
        }
    }
    (coefficients, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ridge_commutes_with_pooling_targets() {
        let x = [1., 0., 1., 0., 1., 1., 1., 1., 1., 2., -1., 1.];
        let y = [1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.];
        let mapping = [0, 1, 0];
        let pooled = pool(&y, 3, &mapping, 2);
        let (full, _) = ridge(&x, &y, 4, 3, 3);
        let (quotient, _) = ridge(&x, &pooled, 4, 3, 2);
        let predicted = pool(&linear(&x, 4, 3, &full, 3), 3, &mapping, 2);
        let direct = linear(&x, 4, 3, &quotient, 2);
        assert!(
            predicted
                .iter()
                .zip(direct)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn fiber_cancellation_and_sharp_range_bound() {
        let mass_error = [100., -100., 0.25, -0.25];
        assert_eq!(pool(&mass_error, 4, &[0, 0, 1, 1], 2), [0., 0.]);
        let coefficients = [10001., 9999.];
        let error = [0.25, -0.25];
        let combined: f64 = coefficients.iter().zip(error).map(|(d, e)| d * e).sum();
        let centered_bound =
            (coefficients[0] - coefficients[1]) / 2. * error.iter().map(|e| e.abs()).sum::<f64>();
        assert_eq!(combined.abs(), centered_bound);
        assert_eq!(combined, 0.5);
    }
}
