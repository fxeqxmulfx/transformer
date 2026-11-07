//! ST1-ST3: signed training readouts and two residual norms with an explicit defect.
use crate::{
    algebra::linear,
    svd,
    transport::{Bound, RANKS, summary},
};
use serde_json::{Value, json};

fn transpose(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..cols)
        .flat_map(|j| (0..rows).map(move |i| x[i * cols + j]))
        .collect()
}
fn norms(x: &[f64], width: usize) -> Vec<f64> {
    x.chunks_exact(width)
        .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect()
}

pub fn evaluate(
    states: &[f64],
    embedding: &[f64],
    references: &[usize],
    queries: &[usize],
    width: usize,
    classes: usize,
    label: usize,
) -> (Vec<Vec<Bound>>, Value) {
    let nt = references.len();
    let nq = queries.len();
    let training: Vec<_> = references
        .iter()
        .flat_map(|&q| states[q * width..(q + 1) * width].iter().copied())
        .collect();
    let mut center = vec![0.; width];
    for row in training.chunks_exact(width) {
        for (k, v) in row.iter().enumerate() {
            center[k] += v / nt as f64;
        }
    }
    let centered: Vec<_> = training
        .chunks_exact(width)
        .flat_map(|row| row.iter().zip(&center).map(|(v, c)| v - c))
        .collect();
    let displacement: Vec<_> = queries
        .iter()
        .flat_map(|&q| {
            states[q * width..(q + 1) * width]
                .iter()
                .zip(&center)
                .map(|(v, c)| v - c)
        })
        .collect();
    let displacement_norms = norms(&displacement, width);
    // Only observed training rows enter this readout.
    let logits = linear(&training, nt, width, embedding, classes);
    let odds: Vec<_> = logits
        .chunks_exact(classes)
        .flat_map(|row| row.iter().map(move |v| v - row[label]))
        .collect();
    let mut mean_odds = vec![0.; classes];
    for row in odds.chunks_exact(classes) {
        for (c, v) in row.iter().enumerate() {
            mean_odds[c] += v / nt as f64;
        }
    }
    let centered_odds: Vec<_> = odds
        .chunks_exact(classes)
        .flat_map(|row| row.iter().zip(&mean_odds).map(|(v, c)| v - c))
        .collect();
    let contrast: Vec<_> = (0..classes)
        .flat_map(|c| {
            (0..width).map(move |k| embedding[c * width + k] - embedding[label * width + k])
        })
        .collect();
    let contrast_norms = norms(&contrast, width);
    let spectrum = svd::thin(&centered, nt, width);
    let thin = nt.min(width);
    let available = spectrum
        .values
        .iter()
        .filter(|&&s| s > 1e-10 * spectrum.values[0])
        .count();
    let total_energy: f64 = spectrum.values.iter().map(|s| s * s).sum();
    let centered_t = transpose(&centered, nt, width);
    let odds_t = transpose(&centered_odds, nt, classes);
    let mut results = Vec::new();
    let mut rank_stats = Vec::new();
    for requested in RANKS {
        let rank = requested.min(available);
        let basis = &spectrum.right[..rank * width];
        let basis_t = transpose(basis, rank, width);
        let loadings: Vec<_> = (0..nt)
            .flat_map(|i| {
                (0..rank).map({
                    let spectrum = &spectrum;
                    move |j| spectrum.left[i * thin + j] / spectrum.values[j]
                })
            })
            .collect();
        let coordinates = linear(&displacement, nq, width, basis, rank);
        let coefficients = linear(&coordinates, nq, rank, &loadings, nt);
        let reconstructed = linear(&coefficients, nq, nt, &centered_t, width);
        let residual: Vec<_> = displacement
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| a - b)
            .collect();
        let residual_norms = norms(&residual, width);
        let signed = linear(&coefficients, nq, nt, &odds_t, classes);
        let readout_coefficients = linear(&contrast, classes, width, basis, rank);
        let readout_projection = linear(&readout_coefficients, classes, rank, &basis_t, width);
        let readout_residual: Vec<_> = contrast
            .iter()
            .zip(&readout_projection)
            .map(|(a, b)| a - b)
            .collect();
        let readout_norms = norms(&readout_residual, width);
        let defects = linear(&residual, nq, width, basis, rank);
        let defect_penalties = linear(
            &defects.iter().map(|v| v.abs()).collect::<Vec<_>>(),
            nq,
            rank,
            &readout_coefficients
                .iter()
                .map(|v| v.abs())
                .collect::<Vec<_>>(),
            classes,
        );
        let mut plain = Vec::new();
        let mut dual = Vec::new();
        for q in 0..nq {
            let prediction: Vec<_> = (0..classes)
                .map(|c| mean_odds[c] + signed[q * classes + c])
                .collect();
            let plain_upper: Vec<_> = (0..classes)
                .map(|c| prediction[c] + residual_norms[q] * contrast_norms[c])
                .collect();
            let dual_upper: Vec<_> = (0..classes)
                .map(|c| {
                    prediction[c]
                        + residual_norms[q] * readout_norms[c]
                        + defect_penalties[q * classes + c]
                })
                .collect();
            plain.push(Bound::from_upper(&plain_upper, label));
            dual.push(Bound::from_upper(&dual_upper, label));
        }
        results.push(plain);
        results.push(dual);
        let basis_from_training = linear(
            &transpose(&loadings, nt, rank),
            rank,
            nt,
            &centered_t,
            width,
        );
        let basis_reconstruction_error = basis
            .iter()
            .zip(&basis_from_training)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let basis_gram = linear(basis, rank, width, basis, rank);
        let orthogonality_error = (0..rank)
            .flat_map(|i| {
                (0..rank).map({
                    let basis_gram = &basis_gram;
                    move |j| (basis_gram[i * rank + j] - f64::from(i == j)).abs()
                })
            })
            .fold(0f64, f64::max);
        let relative_residuals: Vec<_> = residual_norms
            .iter()
            .zip(&displacement_norms)
            .map(|(r, d)| if *d > 0. { r / d } else { 0. })
            .collect();
        let relative_readout: Vec<_> = (0..classes)
            .filter(|&c| c != label)
            .map(|c| {
                if contrast_norms[c] > 0. {
                    readout_norms[c] / contrast_norms[c]
                } else {
                    0.
                }
            })
            .collect();
        rank_stats.push(json!({"requested_rank":requested,"achieved_rank":rank,
            "retained_training_energy":if total_energy>0. {spectrum.values[..rank].iter().map(|s|s*s).sum::<f64>()/total_energy} else {0.},
            "residual_norm":summary(&residual_norms),"relative_residual_norm":summary(&relative_residuals),
            "coefficient_norm":summary(&norms(&coefficients,nt)),"readout_residual_norm":summary(&readout_norms),
            "relative_readout_residual_norm":summary(&relative_readout),"orthogonality_defect_penalty":summary(&defect_penalties),
            "basis_reconstruction_max_error":basis_reconstruction_error,"basis_orthogonality_max_error":orthogonality_error}));
    }
    (
        results,
        json!({"training_count":nt,"query_count":nq,"ambient_dimension":width,"available_rank":available,
        "singular_values":spectrum.values,"displacement_norm":summary(&displacement_norms),"ranks":rank_stats}),
    )
}
