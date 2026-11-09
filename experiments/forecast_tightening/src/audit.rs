//! Post-freeze diagnostics only: actual future values never enter preparation.
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
    intervals::{Frozen, NAMES, TRACKS},
    measure::Snapshot,
    network::Network,
    query_step::{self, tolerance},
    statistics::{self, Actual},
};
use serde_json::{Value, json};
use std::path::Path;
pub fn split_indices(data: &Data) -> [Vec<usize>; 2] {
    [&data.train, &data.validation].map(|rows| {
        rows.as_chunks::<3>()
            .0
            .iter()
            .map(|r| (97 * r[0] + r[1]) as usize)
            .collect()
    })
}
pub fn difficult(data: &Data, old: &Snapshot) -> [Vec<usize>; 2] {
    // Same old-margin ordering and ceil-decile rule as task #188 query_summary.
    [&data.train, &data.validation].map(|rows| {
        let mut scored: Vec<_> = rows
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| {
                let q = (97 * r[0] + r[1]) as usize;
                let t = r[2] as usize;
                let wrong = (0..97)
                    .filter(|&c| c != t)
                    .map(|c| old.logits[q * 97 + c])
                    .fold(f64::NEG_INFINITY, f64::max);
                (q, old.logits[q * 97 + t] - wrong)
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(scored.len().div_ceil(10));
        scored.into_iter().map(|(q, _)| q).collect()
    })
}
pub struct Problem<'a> {
    pub frozen: &'a Frozen,
    pub actual: &'a Network,
    pub old: &'a Snapshot,
    pub new: &'a Snapshot,
    pub next: &'a [f64],
    pub rows: &'a [u32],
    pub data: &'a Data,
    pub difficult: &'a [Vec<usize>; 2],
    pub directory: &'a Path,
}
pub fn run(p: Problem<'_>) -> Result<Value> {
    let f = p.frozen;
    let n = p.rows.len() / 3;
    let executed_math = linear(&p.new.features, n, W, &p.next[..E], 97);
    let mut stages = vec![];
    let mut layer_violations = 0usize;
    require(
        f.network.stages.len() == p.actual.stages.len(),
        "stage count mismatch",
    )?;
    for (s, a) in f.network.stages.iter().zip(&p.actual.stages) {
        require(
            s.name == a.name && s.width == a.width,
            "stage identity mismatch",
        )?;
        let errors: Vec<_> = (0..n)
            .map(|q| {
                norm(
                    &(0..s.width)
                        .map(|k| a.center[q * s.width + k] - s.center[q * s.width + k])
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let methods:Vec<_>=(0..4).map(|m|{
            let violations=(0..n).filter(|&q|errors[q]>s.radii[m][q]+tolerance(&[errors[q],s.radii[m][q]])).count();
            layer_violations+=violations;
            json!({"method":NAMES[m],"violations":violations,"radius":statistics::distribution(s.radii[m].iter().copied()),
                "effective_amplification":statistics::distribution(s.gains[m].iter().copied()),
                "max_error_over_radius":(0..n).map(|q|errors[q]/(s.radii[m][q]+1e-30)).fold(0.,f64::max)})
        }).collect();
        stages.push(json!({"name":s.name,"width":s.width,"actual_error":statistics::distribution(errors),
            "injected_defect":statistics::distribution(s.defects.iter().copied()),"methods":methods}));
    }
    let mut actual = Actual {
        old: vec![],
        new: vec![],
        correct: vec![],
    };
    let (mut old_fp, mut new_fp, mut logit_error) = (0f64, 0f64, 0f64);
    let (mut feature_allowance_violations, mut logit_allowance_violations) = (0usize, 0usize);
    let mut gap_violations = [0usize; TRACKS];
    let mut outcome_violations = [0usize; TRACKS];
    let mut records = vec![];
    for (q, row) in p.rows.as_chunks::<3>().0.iter().enumerate() {
        let t = row[2] as usize;
        let distance = |a: &[f64], b: &[f64]| {
            norm(
                &(0..W)
                    .map(|k| a[q * W + k] - b[q * W + k])
                    .collect::<Vec<_>>(),
            )
        };
        let a = distance(&f.network.output.base.a, &p.old.features);
        let b = distance(&p.actual.output.base.a, &p.new.features);
        old_fp = old_fp.max(a);
        new_fp = new_fp.max(b);
        feature_allowance_violations += usize::from(a > 1e-3) + usize::from(b > 1e-3);
        let old_gap: Vec<_> = (0..97)
            .map(|c| p.old.logits[q * 97 + c] - p.old.logits[q * 97 + t])
            .collect();
        let new_gap: Vec<_> = (0..97)
            .map(|c| p.new.logits[q * 97 + c] - p.new.logits[q * 97 + t])
            .collect();
        let old = query_step::interval(&old_gap, &old_gap, t);
        let new = query_step::interval(&new_gap, &new_gap, t);
        let winner = (1..97).fold(0, |best, c| {
            if p.new.logits[q * 97 + c] > p.new.logits[q * 97 + best] {
                c
            } else {
                best
            }
        });
        actual.old.push(old);
        actual.new.push(new);
        actual.correct.push(winner == t);
        for c in 0..97 {
            let error = (executed_math[q * 97 + c] - p.new.logits[q * 97 + c]).abs();
            logit_error = logit_error.max(error);
            logit_allowance_violations += usize::from(error > 1e-3);
            if c == t {
                continue;
            }
            for (m, violations) in gap_violations.iter_mut().enumerate() {
                let [lo, hi] = f.gaps[m][q * 97 + c];
                let tol = tolerance(&[lo, hi, new_gap[c]]);
                *violations += usize::from(new_gap[c] < lo - tol || new_gap[c] > hi + tol);
            }
        }
        let bounds: Vec<_> = (0..TRACKS)
            .map(|m| {
                let b = f.outcomes[m][q];
                let mt = tolerance(&[b.margin[0], b.margin[1], new.margin[0]]);
                let lt = tolerance(&[b.loss[0], b.loss[1], new.loss[0]]);
                outcome_violations[m] += usize::from(
                    new.margin[0] < b.margin[0] - mt
                        || new.margin[0] > b.margin[1] + mt
                        || new.loss[0] < b.loss[0] - lt
                        || new.loss[0] > b.loss[1] + lt,
                );
                [b.margin[0], b.margin[1], b.loss[0], b.loss[1]]
            })
            .collect();
        records.push(json!([
            q,
            t,
            winner,
            old.margin[0],
            new.margin[0],
            old.loss[0],
            new.loss[0],
            bounds
        ]));
    }
    let split = split_indices(p.data);
    let subsets = json!({"train":statistics::subset(&split[0],&actual,f),"validation":statistics::subset(&split[1],&actual,f),
        "difficult_train":statistics::subset(&p.difficult[0],&actual,f),"difficult_validation":statistics::subset(&p.difficult[1],&actual,f)});
    let false_certificates = subsets
        .as_object()
        .unwrap()
        .values()
        .flat_map(|s| s["methods"].as_array().unwrap())
        .map(|m| m["false_certificates"].as_u64().unwrap())
        .sum::<u64>();
    storage::json_write(
        &p.directory.join("queries.json"),
        &json!({"fields":["query","target","winner","old_margin","new_margin","old_ce","new_ce","method_margin_lower_upper_loss_lower_upper"],"methods":NAMES,"rows":records}),
    )?;
    Ok(
        json!({"passed":layer_violations==0&&feature_allowance_violations==0&&logit_allowance_violations==0&&gap_violations.iter().all(|&v|v==0)&&outcome_violations.iter().all(|&v|v==0)&&false_certificates==0,
        "queries":n,"class_pairs_per_track":n*96,"layer_violations":layer_violations,"feature_allowance_violations":feature_allowance_violations,
        "logit_allowance_violations":logit_allowance_violations,"gap_violations":gap_violations,"outcome_violations":outcome_violations,
        "false_certificates_in_reported_subsets":false_certificates,"old_fp_max":old_fp,"new_fp_max":new_fp,"logit_error_max":logit_error,
        "stages":stages,"subsets":subsets}),
    )
}
