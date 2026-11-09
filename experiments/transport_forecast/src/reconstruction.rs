//! ST1–ST3 from signed_transport/reconstruction.rs at 07c328f1; retain classwise
//! endpoints for TF1–TF2 instead of only probability/margin summaries.
use crate::{algebra::linear, basis_svd};
use serde_json::{Value, json};
pub const RANKS: [usize; 3] = [0, 32, 128];
fn transpose(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..cols)
        .flat_map(|j| (0..rows).map(move |i| x[i * cols + j]))
        .collect()
}
fn norms(x: &[f64], width: usize) -> Vec<f64> {
    x.chunks_exact(width)
        .map(|r| r.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect()
}
pub struct Problem<'a> {
    pub states: &'a [f64],
    pub embedding: &'a [f64],
    pub references: &'a [usize],
    pub queries: &'a [usize],
    pub width: usize,
    pub classes: usize,
    pub label: usize,
}
pub fn evaluate(p: Problem<'_>) -> (Vec<Vec<f64>>, Value) {
    let Problem {
        states,
        embedding,
        references,
        queries,
        width,
        classes,
        label,
    } = p;
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
        .flat_map(|r| r.iter().zip(&center).map(|(v, c)| v - c))
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
    // Readout evaluation here sees only the observed training rows.
    let logits = linear(&training, nt, width, embedding, classes);
    let odds: Vec<_> = logits
        .chunks_exact(classes)
        .flat_map(|r| r.iter().map(move |v| v - r[label]))
        .collect();
    let mut mean = vec![0.; classes];
    for row in odds.chunks_exact(classes) {
        for (c, v) in row.iter().enumerate() {
            mean[c] += v / nt as f64;
        }
    }
    let centered_odds: Vec<_> = odds
        .chunks_exact(classes)
        .flat_map(|r| r.iter().zip(&mean).map(|(v, c)| v - c))
        .collect();
    let contrasts: Vec<_> = (0..classes)
        .flat_map(|c| {
            (0..width).map(move |k| embedding[c * width + k] - embedding[label * width + k])
        })
        .collect();
    let spectrum = basis_svd::thin(&centered, nt, width);
    let available = spectrum
        .values
        .iter()
        .filter(|&&s| s > 1e-10 * spectrum.values[0])
        .count();
    let total: f64 = spectrum.values.iter().map(|s| s * s).sum();
    let mut endpoints = Vec::new();
    let mut stats = Vec::new();
    for requested in RANKS {
        let rank = requested.min(available);
        let basis = &spectrum.right[..rank * width];
        let loadings: Vec<_> = (0..nt)
            .flat_map(|i| {
                (0..rank).map({
                    let spectrum = &spectrum;
                    move |j| spectrum.left[i * nt.min(width) + j] / spectrum.values[j]
                })
            })
            .collect();
        let coordinates = linear(&displacement, nq, width, basis, rank);
        let coefficients = linear(&coordinates, nq, rank, &loadings, nt);
        let rebuilt = linear(
            &coefficients,
            nq,
            nt,
            &transpose(&centered, nt, width),
            width,
        );
        let residual: Vec<_> = displacement
            .iter()
            .zip(&rebuilt)
            .map(|(a, b)| a - b)
            .collect();
        let residual_norm = norms(&residual, width);
        let signed = linear(
            &coefficients,
            nq,
            nt,
            &transpose(&centered_odds, nt, classes),
            classes,
        );
        let b = linear(&contrasts, classes, width, basis, rank);
        let projected = linear(&b, classes, rank, &transpose(basis, rank, width), width);
        let remaining: Vec<_> = contrasts
            .iter()
            .zip(&projected)
            .map(|(a, b)| a - b)
            .collect();
        let remaining_norm = norms(&remaining, width);
        let defects = linear(&residual, nq, width, basis, rank);
        let penalties = linear(
            &defects.iter().map(|x| x.abs()).collect::<Vec<_>>(),
            nq,
            rank,
            &b.iter().map(|x| x.abs()).collect::<Vec<_>>(),
            classes,
        );
        let mut upper = Vec::with_capacity(nq * classes);
        for q in 0..nq {
            for c in 0..classes {
                upper.push(if c == label {
                    0.
                } else {
                    mean[c]
                        + signed[q * classes + c]
                        + residual_norm[q] * remaining_norm[c]
                        + penalties[q * classes + c]
                });
            }
        }
        stats.push(json!({"requested_rank":requested,"achieved_rank":rank,
            "retained_energy":if total>0. {spectrum.values[..rank].iter().map(|s|s*s).sum::<f64>()/total} else {0.},
            "mean_residual_norm":residual_norm.iter().sum::<f64>()/nq as f64,
            "max_orthogonality_penalty":penalties.iter().copied().fold(0.,f64::max)}));
        endpoints.push(upper);
    }
    (
        endpoints,
        json!({"reference_count":nt,"query_count":nq,"available_rank":available,"singular_values":spectrum.values,"ranks":stats}),
    )
}
