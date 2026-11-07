//! AC1-AC4 bounds on actual combined contrasts; all fitting uses training rows.
use crate::{
    algebra::linear,
    data::Split,
    model::{Forward, P, W},
    probe::{self, D, Probe},
};
use serde_json::{Value, json};

fn summary(values: &[f64]) -> Value {
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    json!({"min":v[0],"median":v[v.len()/2],"max":v[v.len()-1],"mean":v.iter().sum::<f64>()/v.len() as f64})
}

fn winner(row: &[f64]) -> usize {
    let mut best = 0;
    for j in 1..row.len() {
        if row[j] > row[best] {
            best = j;
        }
    }
    best
}

fn correct(scores: &[f64], indices: &[usize]) -> usize {
    indices
        .iter()
        .filter(|&&q| winner(&scores[q * P..(q + 1) * P]) == (q / P + q % P).pow(3) % P)
        .count()
}

struct ErrorMass {
    l1: Vec<f64>,
    squared_l2: Vec<f64>,
    total: Vec<f64>,
}

fn errors(probe: &Probe, target: &[usize]) -> ErrorMass {
    let mut l1 = Vec::new();
    let mut squared_l2 = Vec::new();
    let mut total = Vec::new();
    for (q, row) in probe.values.chunks_exact(probe.dim).enumerate() {
        let e: Vec<_> = row
            .iter()
            .enumerate()
            .map(|(j, x)| x - f64::from(j == target[q]))
            .collect();
        l1.push(e.iter().map(|x| x.abs()).sum());
        squared_l2.push(e.iter().map(|x| x * x).sum());
        total.push(e.iter().sum::<f64>().abs());
    }
    ErrorMass {
        l1,
        squared_l2,
        total,
    }
}

// Return the affine coefficients of actual raw scores minus reconstructed scores.
fn alignment_weights(weights: &[f64], probe: &Probe, decoder: &[f64]) -> Vec<f64> {
    let mut residual = vec![0.; P * D];
    for v in 0..P {
        for k in 0..D {
            residual[v * D + k] = if k < W { weights[v * W + k] } else { 0. };
            for j in 0..probe.dim {
                residual[v * D + k] -= decoder[v * probe.dim + j] * probe.weights[j * D + k];
            }
        }
    }
    residual
}

struct Analysis<'a> {
    out: &'a Forward,
    split: &'a Split,
    design: &'a [f64],
    weights: &'a [f64],
    probe: &'a Probe,
    target: &'a [usize],
    unpooled: &'a ErrorMass,
}

fn analyze_decoder(a: Analysis<'_>) -> Value {
    let Analysis {
        out,
        split,
        design,
        weights,
        probe,
        target,
        unpooled,
    } = a;
    let n = P * P;
    let k = probe.dim;
    let (decoder, counts) = probe::decoder(out, split, target, k);
    let recovered = linear(&probe.values, n, k, &decoder, P);
    let ideal: Vec<_> = target
        .iter()
        .flat_map(|&j| {
            (0..P).map({
                let decoder = &decoder;
                move |v| decoder[v * k + j]
            })
        })
        .collect();
    let residual = alignment_weights(weights, probe, &decoder);
    let expanded_alignment = linear(design, n, D, &residual, P);
    let error = errors(probe, target);
    let mut radius = vec![0.; P * P];
    let mut midpoint = vec![0.; P * P];
    let mut residual_norm = vec![0.; P * P];
    let mut residual_bias = vec![0.; P * P];
    for t in 0..P {
        for v in 0..P {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for j in 0..k {
                let d = decoder[t * k + j] - decoder[v * k + j];
                lo = lo.min(d);
                hi = hi.max(d);
            }
            radius[t * P + v] = (hi - lo) / 2.;
            midpoint[t * P + v] = (hi + lo) / 2.;
            residual_norm[t * P + v] = (0..W)
                .map(|j| (residual[t * D + j] - residual[v * D + j]).powi(2))
                .sum::<f64>()
                .sqrt();
            residual_bias[t * P + v] = (residual[t * D + W] - residual[v * D + W]).abs();
        }
    }
    let names = [
        "direct",
        "combined",
        "coordinate",
        "range",
        "unpooled_range",
        "weight_alignment_combined",
        "weight_alignment_range",
    ];
    let mut budgets = vec![vec![f64::INFINITY; n]; names.len()];
    let mut alignment = vec![0f64; n];
    let mut weight_alignment = vec![0f64; n];
    let mut gaps = vec![f64::INFINITY; n];
    let mut combined_error = vec![0f64; n];
    let mut coordinate_error = vec![0f64; n];
    let mut range_error = vec![0f64; n];
    let mut expansion_error = 0f64;
    let mut reconstruction_error = 0f64;
    for q in 0..n {
        let t = (q / P + q % P).pow(3) % P;
        let state_norm = out.states[q * W..(q + 1) * W]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        // Both input residual and output branch are independently normalized.
        assert!(state_norm <= 2. * (W as f64).sqrt() + 1e-10);
        for v in 0..P {
            if v == t {
                continue;
            }
            let raw = out.raw[q * P + t] - out.raw[q * P + v];
            let gap = ideal[q * P + t] - ideal[q * P + v];
            let decoded = recovered[q * P + t] - recovered[q * P + v];
            let align = (raw - decoded).abs();
            let norm_align = residual_norm[t * P + v] * state_norm + residual_bias[t * P + v];
            let expanded = expanded_alignment[q * P + t] - expanded_alignment[q * P + v];
            expansion_error = expansion_error.max((expanded - (raw - decoded)).abs());
            assert!(
                align <= norm_align + 1e-8,
                "Cauchy alignment bound violated"
            );
            let mut combined = 0.;
            let mut coordinate = 0.;
            for j in 0..k {
                let e = probe.values[q * k + j] - f64::from(j == target[q]);
                let term = (decoder[t * k + j] - decoder[v * k + j]) * e;
                combined += term;
                coordinate += term.abs();
            }
            reconstruction_error = reconstruction_error.max((combined - (decoded - gap)).abs());
            let combined = combined.abs();
            let rounding = midpoint[t * P + v].abs() * error.total[q];
            let range = radius[t * P + v] * error.l1[q] + rounding;
            let raw_range =
                radius[t * P + v] * unpooled.l1[q] + midpoint[t * P + v].abs() * unpooled.total[q];
            assert!(combined <= coordinate + 1e-8 && combined <= range + 1e-8);
            let costs = [
                (raw - gap).abs(),
                align + combined,
                align + coordinate,
                align + range,
                align + raw_range,
                norm_align + combined,
                norm_align + range,
            ];
            for (i, cost) in costs.into_iter().enumerate() {
                budgets[i][q] = budgets[i][q].min(gap - cost);
            }
            alignment[q] = alignment[q].max(align);
            weight_alignment[q] = weight_alignment[q].max(norm_align);
            gaps[q] = gaps[q].min(gap);
            combined_error[q] = combined_error[q].max(combined);
            coordinate_error[q] = coordinate_error[q].max(coordinate);
            range_error[q] = range_error[q].max(range);
        }
    }
    assert!(expansion_error < 1e-8 && reconstruction_error < 1e-8);
    let part = |indices: &[usize]| {
        let select = |x: &[f64]| indices.iter().map(|&q| x[q]).collect::<Vec<_>>();
        let mut positive = serde_json::Map::new();
        let mut summaries = serde_json::Map::new();
        for (i, name) in names.iter().enumerate() {
            let count = indices.iter().filter(|&&q| budgets[i][q] > 0.).count();
            assert!(count <= correct(&out.logits, indices));
            positive.insert((*name).to_owned(), json!(count));
            summaries.insert((*name).to_owned(), summary(&select(&budgets[i])));
        }
        json!({"count":indices.len(),"source_correct":correct(&out.logits,indices),
            "reconstructed_correct":correct(&recovered,indices),"ideal_decoder_correct":correct(&ideal,indices),
            "representation_correct":indices.iter().filter(|&&q|winner(&probe.values[q*k..(q+1)*k])==target[q]).count(),
            "positive":positive,"budgets":summaries,"l1_error":summary(&select(&error.l1)),
            "squared_l2_error":summary(&select(&error.squared_l2)),"mass_residual":summary(&select(&error.total)),
            "alignment":summary(&select(&alignment)),"weight_alignment_bound":summary(&select(&weight_alignment)),
            "ideal_gap":summary(&select(&gaps)),"combined_error":summary(&select(&combined_error)),
            "coordinate_error":summary(&select(&coordinate_error)),"range_error":summary(&select(&range_error))})
    };
    json!({"dimension":k,"ridge_penalty":probe.penalty,"training_coordinate_counts":counts,
        "affine_expansion_max_error":expansion_error,"combined_expansion_max_error":reconstruction_error,
        "coefficient_half_range":summary(&radius),"train":part(&split.train),"validation":part(&split.validation)})
}

pub fn analyze(out: &Forward, split: &Split, weights: &[f64]) -> Value {
    let design = probe::design(out);
    let sums: Vec<_> = (0..P * P).map(|q| (q / P + q % P) % P).collect();
    let mut labels: Vec<_> = (0..P).map(|s| s.pow(3) % P).collect();
    labels.sort();
    labels.dedup();
    assert_eq!(labels.len(), 33);
    let mapping: Vec<_> = (0..P)
        .map(|s| labels.binary_search(&(s.pow(3) % P)).unwrap())
        .collect();
    let answers: Vec<_> = sums.iter().map(|&s| mapping[s]).collect();
    let reference = vec![1. / P as f64; P];
    let quotient_reference = probe::pool(&reference, P, &mapping, labels.len());
    let full = probe::fit(&design, split, &sums, &reference);
    let quotient = probe::fit(&design, split, &answers, &quotient_reference);
    let pooled = probe::pool(&full.values, P, &mapping, labels.len());
    let pooling_error = pooled
        .iter()
        .zip(&quotient.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(
        pooling_error < 1e-8,
        "independent quotient probe disagrees with pooling"
    );
    let unpooled = errors(&full, &sums);
    let pooled_error = errors(&quotient, &answers);
    assert!(
        pooled_error
            .l1
            .iter()
            .zip(&unpooled.l1)
            .all(|(p, u)| p <= &(u + 1e-8))
    );
    let full_report = analyze_decoder(Analysis {
        out,
        split,
        design: &design,
        weights,
        probe: &full,
        target: &sums,
        unpooled: &unpooled,
    });
    let quotient_report = analyze_decoder(Analysis {
        out,
        split,
        design: &design,
        weights,
        probe: &quotient,
        target: &answers,
        unpooled: &unpooled,
    });
    let part = |indices: &[usize]| {
        json!({"count":indices.len(),"source_correct":correct(&out.logits,indices),
        "residual_only_correct":correct(&out.residual,indices),
        "final_rms_denominator":summary(&indices.iter().map(|&q|out.denominators[q]).collect::<Vec<_>>()),
        "pooled_to_unpooled_l1_ratio":summary(&indices.iter().map(|&q|pooled_error.l1[q]/unpooled.l1[q]).collect::<Vec<_>>())})
    };
    json!({"sum":full_report,"quotient":quotient_report,"answer_labels":labels,
        "answer_reference":quotient_reference,"pooling_max_error":pooling_error,
        "train":part(&split.train),"validation":part(&split.validation),
        "status":"Retrospective numerical contrast budgets; feature error measured on each query; execution error optimistically zero"})
}
