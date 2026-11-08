//! Frozen feature balls, then independent future-state checks and outcome coverage.
use crate::{
    algebra::linear,
    domain::{
        Result,
        model::{E, W},
    },
    forecast_jet::{Jet, norm},
    forecast_network,
    inputs::{Case, hash, require},
    measure::Snapshot,
    query_step::{self, Interval, Row, tolerance},
    query_summary,
};
use serde_json::{Value, json};
use std::{fs::OpenOptions, io::Write};

const FEATURE_ALLOWANCE: f64 = 1e-3;
const LOGIT_ALLOWANCE: f64 = 1e-3;
pub struct Frozen {
    jet: Jet,
    centers: Vec<f64>,
    gaps: Vec<f64>,
    radii: Vec<f64>,
    intervals: Vec<Interval>,
    tangent_error: f64,
    binary: Value,
}
pub fn prepare(case: &Case<'_>, old: &[f32], next: &[f32]) -> Result<Frozen> {
    let old: Vec<_> = old.iter().map(|&x| x as f64).collect();
    let next: Vec<_> = next.iter().map(|&x| x as f64).collect();
    let step: Vec<_> = next.iter().zip(&old).map(|(n, o)| n - o).collect();
    let tangent_error =
        forecast_network::tangent_check(case.arm, &old, &step, &case.data.train[..24])?;
    let rows = forecast_network::canonical_rows(case.data);
    let jet = forecast_network::features(case.arm, &old, &step, &rows);
    let centers: Vec<_> = jet.a.iter().zip(&jet.v).map(|(a, v)| a + v).collect();
    require(
        centers.iter().chain(&jet.r).all(|x| x.is_finite()) && jet.r.iter().all(|&r| r >= 0.),
        "nonfinite forecast",
    )?;
    let predicted = linear(&centers, 9409, W, &next[..E], 97);
    let contrast: Vec<_> = (0..97)
        .flat_map(|t| {
            let next = &next;
            (0..97).map(move |c| {
                (0..W)
                    .map(|k| (next[c * W + k] - next[t * W + k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
        })
        .collect();
    let (mut gaps, mut radii, mut intervals) = (vec![], vec![], vec![]);
    for (q, row) in rows.as_chunks::<3>().0.iter().enumerate() {
        let target = row[2] as usize;
        let mut lower = vec![0.; 97];
        let mut upper = vec![0.; 97];
        for c in 0..97 {
            let gap = predicted[q * 97 + c] - predicted[q * 97 + target];
            let radius = if c == target {
                0.
            } else {
                contrast[target * 97 + c] * (jet.r[q] + FEATURE_ALLOWANCE) + 2. * LOGIT_ALLOWANCE
            };
            gaps.push(gap);
            radii.push(radius);
            lower[c] = gap - radius;
            upper[c] = gap + radius;
        }
        intervals.push(query_step::interval(&lower, &upper, target));
    }
    // Persist the frozen centers/radii before the candidate network is evaluated.
    // Record: little-endian u64 rows,width; then each row's W f64 centers and radius.
    let mut bytes = Vec::with_capacity(16 + 9409 * (W + 1) * 8);
    bytes.extend(9409u64.to_le_bytes());
    bytes.extend((W as u64).to_le_bytes());
    for q in 0..9409 {
        for x in &centers[q * W..(q + 1) * W] {
            bytes.extend(x.to_le_bytes());
        }
        bytes.extend(jet.r[q].to_le_bytes());
    }
    let mut file = OpenOptions::new().append(true).open(case.forecast_output)?;
    let offset = file.metadata()?.len();
    file.write_all(&bytes)?;
    file.flush()?;
    let binary = json!({"offset":offset,"bytes":bytes.len(),"sha256":hash(&bytes),
        "format":"u64 rows,width; row-major f64[width] centers and f64 real-arithmetic radius; little endian"});
    Ok(Frozen {
        jet,
        centers,
        gaps,
        radii,
        intervals,
        tangent_error,
        binary,
    })
}
fn named_subset(indices: &[usize], rows: &[Row]) -> Result<Value> {
    let mut result = query_summary::subset(indices, rows)?;
    let object = result.as_object_mut().unwrap();
    let forecast = object.remove("joint").unwrap();
    let retrospective = object.remove("separate").unwrap();
    object.insert("forecast".into(), forecast);
    object.insert("retrospective_joint".into(), retrospective);
    Ok(result)
}
impl Frozen {
    pub fn audit(
        self,
        case: &Case<'_>,
        old: &Snapshot,
        new: &Snapshot,
        next: &[f32],
        rows: &[Row],
        difficult: &[Vec<usize>; 2],
    ) -> Result<Value> {
        let next: Vec<_> = next.iter().map(|&x| x as f64).collect();
        let canonical = forecast_network::canonical_rows(case.data);
        // This evaluation occurs only after prepare has frozen and persisted its output.
        let actual = forecast_network::features(case.arm, &next, &vec![0.; next.len()], &canonical);
        let executed_math = linear(&new.features, 9409, W, &next[..E], 97);
        let (mut records, mut feature_rows) = (Vec::new(), Vec::new());
        let (mut max_old_fp, mut max_new_fp, mut max_logit, mut max_gap) = (0f64, 0f64, 0f64, 0f64);
        let (mut max_error, mut max_radius, mut max_ratio) = (0f64, 0f64, 0f64);
        for q in 0..9409 {
            let distance = |a: &[f64], b: &[f64]| {
                norm(
                    &(0..W)
                        .map(|k| a[q * W + k] - b[q * W + k])
                        .collect::<Vec<_>>(),
                )
            };
            let old_fp = distance(&self.jet.a, &old.features);
            let new_fp = distance(&actual.a, &new.features);
            let error = distance(&actual.a, &self.centers);
            let gpu_error = distance(&new.features, &self.centers);
            let radius = self.jet.r[q];
            max_old_fp = max_old_fp.max(old_fp);
            max_new_fp = max_new_fp.max(new_fp);
            max_error = max_error.max(error);
            max_radius = max_radius.max(radius);
            max_ratio = max_ratio.max(error / (radius + 1e-30));
            require(
                old_fp <= FEATURE_ALLOWANCE && new_fp <= FEATURE_ALLOWANCE,
                "fixed FP32 feature allowance violated",
            )?;
            require(
                error <= radius + tolerance(&[error, radius]),
                "real-arithmetic feature forecast enclosure violated",
            )?;
            require(
                gpu_error <= radius + FEATURE_ALLOWANCE + tolerance(&[gpu_error, radius]),
                "executed feature forecast enclosure violated",
            )?;
            let mut record = rows[q].clone();
            record.intervals = [self.intervals[q], rows[q].intervals[0]];
            for c in 0..97 {
                let at = q * 97 + c;
                let t = q * 97 + rows[q].target;
                let fp = (new.logits[at] - executed_math[at]).abs();
                max_logit = max_logit.max(fp);
                require(fp <= LOGIT_ALLOWANCE, "fixed FP32 logit allowance violated")?;
                let gap = new.logits[at] - new.logits[t];
                let excess = ((gap - self.gaps[at]).abs() - self.radii[at])
                    / (1. + gap.abs().max(self.gaps[at].abs()).max(self.radii[at]));
                max_gap = max_gap.max(excess);
                require(excess <= 1e-8, "forecast competitor gap enclosure violated")?;
            }
            for (a, b) in [
                (record.margin[1], self.intervals[q].margin),
                (record.loss[1], self.intervals[q].loss),
            ] {
                require(
                    b[0] <= a + tolerance(&b) && a <= b[1] + tolerance(&b),
                    "forecast margin/loss interval violated",
                )?;
            }
            feature_rows.push(json!([error, radius, gpu_error, old_fp, new_fp]));
            records.push(record);
        }
        let mut result = json!({});
        for (split, (name, input)) in [
            ("train", &case.data.train),
            ("validation", &case.data.validation),
        ]
        .into_iter()
        .enumerate()
        {
            let indices: Vec<_> = input
                .as_chunks::<3>()
                .0
                .iter()
                .map(|r| (r[0] * 97 + r[1]) as usize)
                .collect();
            let errors: Vec<_> = indices
                .iter()
                .copied()
                .filter(|&q| !rows[q].correct[0])
                .collect();
            result[name] = json!({"all":named_subset(&indices,&records)?,"old_bottom_decile":named_subset(&difficult[split],&records)?,
                "old_error_rows":named_subset(&errors,&records)?});
        }
        result["checks"] = json!({"passed":true,"queries":9409,"competitor_pairs":9409*96,
            "tangent_scaled_error":self.tangent_error,"max_real_feature_error":max_error,"max_real_feature_radius":max_radius,
            "max_error_radius_ratio":max_ratio,"max_old_fp32_feature_error":max_old_fp,"max_new_fp32_feature_error":max_new_fp,
            "max_fp32_logit_error":max_logit,"max_scaled_gap_excess":max_gap,"violations":0});
        result["frozen_binary"] = self.binary;
        result["feature_row_fields"] = json!([
            "real_error",
            "real_radius",
            "gpu_error",
            "old_fp32_feature_error",
            "new_fp32_feature_error"
        ]);
        result["feature_rows"] = json!(feature_rows);
        result["query_row_fields"] = json!(query_step::ROW_FIELDS.map(|s| {
            s.replace("joint_", "forecast_")
                .replace("separate_", "retrospective_")
        }));
        result["query_rows"] = json!(records.iter().map(Row::json).collect::<Vec<_>>());
        Ok(result)
    }
}
