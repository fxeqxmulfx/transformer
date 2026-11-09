//! Independent post-freeze audits; labels appear only in this reporting module.
use crate::{
    algebra::linear,
    domain::{
        Result,
        data::Data,
        model::{E, W},
    },
    forecast_jet::norm,
    infrastructure::storage,
    inputs::require,
    measure::Snapshot,
    serving::{Service, predictions, winner},
};
use serde_json::{Value, json};
use std::path::Path;
pub struct Reference<'a> {
    pub data: &'a Data,
    pub old_real: &'a [f64],
    pub new_real: &'a [f64],
    pub old_executed: &'a [f32],
    pub new: &'a Snapshot,
    pub next: &'a [f64],
    pub pairs: &'a [[u32; 2]],
    pub directory: &'a Path,
}
pub fn run(s: &mut Service<'_, '_>, r: Reference<'_>) -> Result<Value> {
    require(
        r.pairs.len() == 9409,
        "audit requires the full operand domain",
    )?;
    let mut labels = vec![0usize; 9409];
    let mut training = vec![false; 9409];
    for (train, rows) in [(true, &r.data.train), (false, &r.data.validation)] {
        for row in rows.chunks_exact(3) {
            let q = (97 * row[0] + row[1]) as usize;
            labels[q] = row[2] as usize;
            training[q] = train;
        }
    }
    let expected: Vec<_> = r
        .new
        .logits
        .chunks_exact(97)
        .map(|row| (1..97).fold(0, |w, c| if row[c] > row[w] { c } else { w }))
        .collect();
    let mut approx = vec![];
    let mut accepted = [vec![], vec![]];
    let mut final_predictions = [vec![], vec![]];
    let (mut source_differences, mut cache_differences, mut reuse_differences) =
        (0usize, 0usize, 0usize);
    for (block, pairs) in r.pairs.chunks(512).enumerate() {
        let offset = block * 512;
        let source = s.engine.inference(pairs, None, s.version, s.io)?;
        let cached = s
            .engine
            .inference(pairs, Some(s.weights), s.version, s.io)?;
        source_differences += source
            .iter()
            .enumerate()
            .filter(|(i, v)| v.to_bits() != (r.new.logits[offset * 97 + i] as f32).to_bits())
            .count();
        cache_differences += source
            .iter()
            .zip(&cached)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        for m in 0..2 {
            let (logits, d) = s.approximate(pairs, m)?;
            if m == 0 {
                approx.extend_from_slice(&logits);
            } else {
                reuse_differences += logits
                    .iter()
                    .zip(&approx[offset * 97..])
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            }
            accepted[m].extend_from_slice(&d.accepted);
            final_predictions[m].extend(s.finish(pairs, d)?);
        }
    }
    let old: Vec<_> = r.old_executed.iter().map(|&x| x as f64).collect();
    let approximate_real = linear(&old, 9409, W, &r.next[..E], 97);
    let executed_real = linear(&r.new.features, 9409, W, &r.next[..E], 97);
    let mut movements = vec![];
    let (mut old_allowance, mut new_allowance, mut approx_allowance, mut logit_allowance) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut old_max, mut new_max, mut approx_max, mut logit_max) = (0f64, 0f64, 0f64, 0f64);
    let mut ball_violations = [0usize; 2];
    let mut contrast_violations = [0usize; 2];
    for q in 0..9409 {
        let distance = |a: &[f64], b: &[f64]| {
            norm(
                &(0..W)
                    .map(|k| a[q * W + k] - b[q * W + k])
                    .collect::<Vec<_>>(),
            )
        };
        let oe = distance(&old, r.old_real);
        let ne = distance(&r.new.features, r.new_real);
        let movement = distance(r.new_real, r.old_real);
        movements.push(movement);
        old_max = old_max.max(oe);
        new_max = new_max.max(ne);
        old_allowance += usize::from(oe > 1e-3);
        new_allowance += usize::from(ne > 1e-3);
        for (m, v) in ball_violations.iter_mut().enumerate() {
            let b = s.bounds[m].radii[q];
            *v += usize::from(movement > b + 1e-8 * (1. + movement.max(b)));
        }
        for c in 0..97 {
            let ae = (approximate_real[q * 97 + c] - approx[q * 97 + c] as f64).abs();
            let le = (executed_real[q * 97 + c] - r.new.logits[q * 97 + c]).abs();
            approx_max = approx_max.max(ae);
            logit_max = logit_max.max(le);
            approx_allowance += usize::from(ae > 1e-3);
            logit_allowance += usize::from(le > 1e-3);
            for d in 0..c {
                let gap = approx[q * 97 + c] as f64 - approx[q * 97 + d] as f64;
                let actual = r.new.logits[q * 97 + c] - r.new.logits[q * 97 + d];
                for (m, v) in contrast_violations.iter_mut().enumerate() {
                    let limit =
                        s.bounds[m].contrasts[c * 97 + d] * (s.bounds[m].radii[q] + 2e-3) + 4e-3;
                    *v += usize::from(
                        (actual - gap).abs()
                            > limit + 1e-8 * (1. + actual.abs().max(gap.abs()).max(limit)),
                    );
                }
            }
        }
    }
    let approximate = predictions(&approx);
    let subset = |indices: Vec<usize>| {
        json!({"count":indices.len(),
        "actual_correct":indices.iter().filter(|&&q|expected[q]==labels[q]).count(),
        "wrong_approximate":indices.iter().filter(|&&q|approximate[q]!=expected[q]).count(),
        "methods":(0..2).map(|m|json!({"method":if m==0{"reuse"}else{"tight-reuse"},
            "accepted":indices.iter().filter(|&&q|accepted[m][q]).count(),
            "coverage":indices.iter().filter(|&&q|accepted[m][q]).count()as f64/indices.len()as f64,
            "false_certificates":indices.iter().filter(|&&q|accepted[m][q]&&approximate[q]!=expected[q]).count(),
            "final_disagreements":indices.iter().filter(|&&q|final_predictions[m][q]!=expected[q]).count(),
            "final_correct":indices.iter().filter(|&&q|final_predictions[m][q]==labels[q]).count()})).collect::<Vec<_>>()})
    };
    let subsets = json!({"all":subset((0..9409).collect()),"train":subset((0..9409).filter(|&q|training[q]).collect()),
        "validation":subset((0..9409).filter(|&q|!training[q]).collect()),
        "timing":subset(r.data.validation.chunks_exact(3).take(512).map(|r|(97*r[0]+r[1])as usize).collect())});
    let passed = source_differences == 0
        && cache_differences == 0
        && reuse_differences == 0
        && old_allowance == 0
        && new_allowance == 0
        && approx_allowance == 0
        && logit_allowance == 0
        && ball_violations == [0, 0]
        && contrast_violations == [0, 0]
        && subsets["all"]["methods"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["false_certificates"] == 0 && m["final_disagreements"] == 0);
    storage::json_write(
        &r.directory.join("queries.json"),
        &json!({"fields":["query","target","reference","approximate","accepted","final","real_movement","radii"],
        "rows":(0..9409).map(|q|json!([q,labels[q],expected[q],approximate[q],[accepted[0][q],accepted[1][q]],
            [final_predictions[0][q],final_predictions[1][q]],movements[q],[s.bounds[0].radii[q],s.bounds[1].radii[q]]])).collect::<Vec<_>>()}),
    )?;
    Ok(
        json!({"passed":passed,"queries":9409,"unordered_class_pairs_per_track":9409usize*97*96/2,
        "source_logit_differences":source_differences,"cached_logit_differences":cache_differences,"reuse_logit_differences":reuse_differences,
        "ball_violations":ball_violations,"contrast_violations":contrast_violations,"allowance_violations":[old_allowance,new_allowance,approx_allowance,logit_allowance],
        "max_errors":{"old_feature":old_max,"new_feature":new_max,"approximate_logit":approx_max,"exact_logit":logit_max},"subsets":subsets,
        "mean_radii":s.bounds.iter().map(|b|b.radii.iter().sum::<f64>()/9409.).collect::<Vec<_>>(),
        "mean_movement":movements.iter().sum::<f64>()/9409.}),
    )
}
