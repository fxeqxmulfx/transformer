//! Observable readout geometry: extend transport_forecast at b52f0d84 with
//! class-centered readout directions reconstructed from the old training span.
use crate::{algebra::linear, basis_svd};
use serde_json::{Value, json};
pub const RANKS: [usize; 3] = [0, 32, 128];
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Hidden,
    Readout,
}
impl Family {
    pub const ALL: [Self; 2] = [Self::Hidden, Self::Readout];
    pub fn name(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::Readout => "readout",
        }
    }
}
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
    pub family: Family,
    pub states: &'a [f64],
    pub embedding: &'a [f64],
    pub references: &'a [usize],
    pub queries: &'a [usize],
    pub width: usize,
    pub classes: usize,
    pub label: usize,
}
pub fn evaluate(p: Problem<'_>) -> (Vec<Vec<f64>>, Value) {
    evaluate_at(p, &RANKS)
}
fn evaluate_at(p: Problem<'_>, ranks: &[usize]) -> (Vec<Vec<f64>>, Value) {
    let Problem {
        family,
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
    let training_basis = &spectrum.right[..available * width];
    let centered_embedding: Vec<_> = (0..classes)
        .flat_map(|c| {
            (0..width).map(move |k| {
                embedding[c * width + k]
                    - (0..classes).map(|v| embedding[v * width + k]).sum::<f64>() / classes as f64
            })
        })
        .collect();
    let observed = linear(
        &centered_embedding,
        classes,
        width,
        training_basis,
        available,
    );
    let projected = linear(
        &observed,
        classes,
        available,
        &transpose(training_basis, available, width),
        width,
    );
    let observable = basis_svd::thin(&projected, classes, width);
    let observable_rank = observable
        .values
        .iter()
        .filter(|&&s| s > 1e-10 * observable.values[0])
        .count()
        .min(available);
    let mut endpoints = Vec::new();
    let mut stats = Vec::new();
    for &requested in ranks {
        let rank = requested.min(if family == Family::Hidden {
            available
        } else {
            observable_rank
        });
        let basis = if family == Family::Hidden {
            &spectrum.right[..rank * width]
        } else {
            &observable.right[..rank * width]
        };
        let coordinates_in_training = linear(basis, rank, width, training_basis, available);
        let loadings: Vec<_> = (0..nt)
            .flat_map(|i| {
                (0..rank).map({
                    let spectrum = &spectrum;
                    let coordinates_in_training = &coordinates_in_training;
                    move |j| {
                        if family == Family::Hidden {
                            spectrum.left[i * nt.min(width) + j] / spectrum.values[j]
                        } else {
                            (0..available)
                                .map(|l| {
                                    spectrum.left[i * nt.min(width) + l] / spectrum.values[l]
                                        * coordinates_in_training[j * available + l]
                                })
                                .sum()
                        }
                    }
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
        let training_energy: f64 = linear(&centered, nt, width, basis, rank)
            .iter()
            .map(|v| v * v)
            .sum();
        let readout_total: f64 = centered_embedding.iter().map(|v| v * v).sum();
        let readout_energy: f64 = linear(&centered_embedding, classes, width, basis, rank)
            .iter()
            .map(|v| v * v)
            .sum();
        stats.push(json!({"requested_rank":requested,"achieved_rank":rank,
            "retained_training_energy":if total>0. {training_energy/total} else {0.},
            "retained_readout_energy":if readout_total>0. {readout_energy/readout_total} else {0.},
            "max_coefficient_norm":norms(&coefficients,nt).into_iter().fold(0.,f64::max),
            "mean_residual_norm":residual_norm.iter().sum::<f64>()/nq as f64,
            "max_orthogonality_penalty":penalties.iter().copied().fold(0.,f64::max)}));
        endpoints.push(upper);
    }
    (
        endpoints,
        json!({"family":family.name(),"reference_count":nt,"query_count":nq,"available_rank":available,
            "smallest_retained_training_singular_value":if available>0 {spectrum.values[available-1]} else {0.},
            "observable_rank":observable_rank,"observable_singular_values":observable.values,
            "singular_values":spectrum.values,"ranks":stats}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readout_direction_can_be_low_variance_with_a_nonzero_residual() {
        let states = [
            10., 0., 0., -10., 0., 0., 0., 1., 0., 0., -1., 0., 0., 3., 0.25,
        ];
        let e = [0., 2., 0., 0., 0., 1.];
        let run = |family| {
            evaluate_at(
                Problem {
                    family,
                    states: &states,
                    embedding: &e,
                    references: &[0, 1, 2, 3],
                    queries: &[4],
                    width: 3,
                    classes: 2,
                    label: 0,
                },
                &[1],
            )
            .0[0][1]
        };
        assert!(run(Family::Hidden) > 0.);
        assert!((run(Family::Readout) + 5.75).abs() < 1e-11);
    }
}
