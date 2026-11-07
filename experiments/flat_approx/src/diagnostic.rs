//! Fixed training-only probes, fixed Fourier decoder and separate held-out accounting.
use crate::{
    algebra::{linear, ridge, singular_values},
    data::Split,
    model::{Forward, P, W},
};
use num_complex::Complex64;
use serde_json::{Value, json};
const F: usize = 48;

fn basis(sum: usize) -> Vec<f64> {
    let mut v = vec![1.];
    for k in 1..=F {
        let a = std::f64::consts::TAU * (k * sum) as f64 / P as f64;
        v.extend([a.cos(), a.sin()]);
    }
    v
}

pub fn correct(scores: &[f64], indices: &[usize]) -> usize {
    indices
        .iter()
        .filter(|&&q| {
            let row = &scores[q * P..(q + 1) * P];
            let mut winner = 0;
            for c in 1..P {
                if row[c] > row[winner] {
                    winner = c;
                }
            }
            winner == ((q / P + q % P).pow(3) % P)
        })
        .count()
}

fn summarize(values: &[f64]) -> Value {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    json!({"min":v[0],"median":v[v.len()/2],"max":v[v.len()-1],"mean":v.iter().sum::<f64>()/v.len() as f64})
}

pub fn analyze(out: &Forward, split: &Split, slack: f64) -> Value {
    let n = P * P;
    let d = W + 1;
    let design: Vec<_> = out
        .states
        .as_chunks::<W>()
        .0
        .iter()
        .flat_map(|s| s.iter().copied().chain([1.]))
        .collect();
    let ideals: Vec<_> = (0..n).flat_map(|q| basis((q / P + q % P) % P)).collect();
    let train_x: Vec<_> = split
        .train
        .iter()
        .flat_map(|&q| design[q * d..(q + 1) * d].iter().copied())
        .collect();
    let train_y: Vec<_> = split
        .train
        .iter()
        .flat_map(|&q| ideals[q * P + 1..(q + 1) * P].iter().copied())
        .collect();
    let (probe, penalty) = ridge(&train_x, &train_y, split.train.len(), d, 2 * F);
    let features = linear(&design, n, d, &probe, 2 * F);
    let z: Vec<_> = features
        .as_chunks::<{ 2 * F }>()
        .0
        .iter()
        .flat_map(|s| [1.].into_iter().chain(s.iter().copied()))
        .collect();
    let mut means = vec![0.; P * P];
    let mut counts = vec![0; P];
    for &q in &split.train {
        let s = (q / P + q % P) % P;
        counts[s] += 1;
        for c in 0..P {
            means[s * P + c] += out.raw[q * P + c];
        }
    }
    assert!(counts.iter().all(|&n| n > 0));
    for s in 0..P {
        for c in 0..P {
            means[s * P + c] /= counts[s] as f64;
        }
    }
    let mut decoder = vec![0.; P * P];
    for s in 0..P {
        let v = basis(s);
        for c in 0..P {
            for j in 0..P {
                decoder[c * P + j] +=
                    means[s * P + c] * v[j] * if j == 0 { 1. / P as f64 } else { 2. / P as f64 };
            }
        }
    }
    let ideal_scores = linear(&ideals, n, P, &decoder, P);
    let recovered_scores = linear(&z, n, P, &decoder, P);
    let mut fourier_error = 0f64;
    for q in 0..n {
        let s = (q / P + q % P) % P;
        for c in 0..P {
            fourier_error = fourier_error.max((ideal_scores[q * P + c] - means[s * P + c]).abs());
        }
    }
    assert!(fourier_error < 1e-8, "DFT decoder inconsistent");
    let mut feature_reports = Vec::new();
    let mut error_bounds = Vec::new();
    for k in 0..F {
        let mut x = Vec::with_capacity(n);
        let mut e = Vec::with_capacity(n);
        let mut train_error = 0.;
        let mut valid_error = 0.;
        let mut max_hidden = 0f64;
        for q in 0..n {
            let value = Complex64::new(z[q * P + 2 * k + 1], z[q * P + 2 * k + 2]);
            let target = Complex64::new(ideals[q * P + 2 * k + 1], ideals[q * P + 2 * k + 2]);
            let error = value - target;
            x.push(value);
            e.push(if split.mask[q] {
                error
            } else {
                Complex64::default()
            });
            if split.mask[q] {
                train_error += error.norm_sqr();
            } else {
                valid_error += error.norm_sqr();
                max_hidden = max_hidden.max(error.norm());
            }
        }
        let norm = singular_values(&x, P).iter().sum::<f64>();
        let rho = singular_values(&e, P).iter().sum::<f64>();
        let epsilon = norm - P as f64;
        assert!(
            epsilon + rho >= -1e-8,
            "certificate triangle consistency failed"
        );
        let bound = 4. * (epsilon + rho).max(0.) / slack + rho;
        error_bounds.push(bound);
        feature_reports.push(
            json!({"frequency":k+1,"nuclear_norm":norm,"objective_excess":epsilon,
            "noise_nuclear_norm":rho,"entry_error_bound":bound,"max_heldout_entry_error":max_hidden,
            "train_mean_squared_error":train_error/split.train.len() as f64,
            "validation_mean_squared_error":valid_error/split.validation.len() as f64}),
        );
    }
    let mut spectral = vec![0.; P * P];
    for t in 0..P {
        for c in 0..P {
            for k in 0..F {
                let dr = decoder[t * P + 2 * k + 1] - decoder[c * P + 2 * k + 1];
                let di = decoder[t * P + 2 * k + 2] - decoder[c * P + 2 * k + 2];
                spectral[t * P + c] += dr.hypot(di) * error_bounds[k];
            }
        }
    }
    let mut gaps = vec![0.; n];
    let mut direct = vec![0.; n];
    let mut budget = vec![0.; n];
    let mut align = vec![0.; n];
    for q in 0..n {
        let target = (q / P + q % P).pow(3) % P;
        let mut gap = f64::INFINITY;
        let mut bspec = 0f64;
        let mut bdirect = 0f64;
        let mut ba = 0f64;
        for c in 0..P {
            if c == target {
                continue;
            }
            let raw = out.raw[q * P + target] - out.raw[q * P + c];
            let ideal = ideal_scores[q * P + target] - ideal_scores[q * P + c];
            let recovered = recovered_scores[q * P + target] - recovered_scores[q * P + c];
            let alignment = (raw - recovered).abs();
            gap = gap.min(ideal);
            bdirect = bdirect.max((raw - ideal).abs());
            ba = ba.max(alignment);
            bspec = bspec.max(alignment + spectral[target * P + c]);
        }
        gaps[q] = gap;
        direct[q] = gap - bdirect;
        budget[q] = gap - bspec;
        align[q] = ba;
    }
    let part = |indices: &[usize]| {
        let select = |v: &[f64]| indices.iter().map(|&q| v[q]).collect::<Vec<_>>();
        json!({"count":indices.len(),"source_correct":correct(&out.logits,indices),
            "residual_only_correct":correct(&out.residual,indices),"ideal_decoder_correct":correct(&ideal_scores,indices),
            "direct_budget_positive_at_eta0":indices.iter().filter(|&&q|direct[q]>0.).count(),
            "spectral_budget_positive_at_eta0":indices.iter().filter(|&&q|budget[q]>0.).count(),
            "ideal_raw_gap":summarize(&select(&gaps)),"direct_budget":summarize(&select(&direct)),
            "spectral_budget":summarize(&select(&budget)),"alignment_contrast_error":summarize(&select(&align)),
            "final_rms_denominator":summarize(&select(&out.denominators))})
    };
    json!({"probe_ridge":penalty,"training_sum_counts":counts,"decoder_dft_max_error":fourier_error,
        "train":part(&split.train),"validation":part(&split.validation),"features":feature_reports,
        "status":"Numerical sufficient-budget diagnostic only; execution error optimistically zero"})
}
