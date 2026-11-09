//! DP3: finite-displacement forecasts, frozen before any endpoint query evaluation.
use crate::{
    Result, audit,
    evidence::{Ledger, hash, require, scalar, word, write},
    intervals::Frozen,
    model,
    network::{self, Factors},
};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub const N: usize = 9409;
pub const P: usize = 97;
pub const TRACKS: [&str; 4] = [
    "direct_directional",
    "transport32",
    "transport33",
    "transport128",
];
pub fn parameters(bytes: &[u8], step: u64, digest: &Value) -> Result<Vec<f64>> {
    let count = model::Arm::Plain.count();
    require(
        bytes.len() == 8 + 12 * count && word(bytes, 0) == step && hash(bytes) == *digest,
        "state source mismatch",
    )?;
    let p: Vec<_> = bytes[8..8 + 4 * count]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x) as f64)
        .collect();
    require(p.iter().all(|x| x.is_finite()), "nonfinite state")?;
    Ok(p)
}
pub fn probe(reference: &Value, step: u64) -> Result<&Value> {
    reference["geometry"]
        .as_array()
        .ok_or("missing probes")?
        .iter()
        .find(|v| v["step"] == step)
        .ok_or("missing declared state".into())
}
pub struct Case<'a> {
    pub name: &'a str,
    pub seed: u64,
    pub anchor: u64,
    pub end: u64,
    pub root: &'a Path,
    pub reference: &'a Value,
    pub support: &'a Path,
    pub support_report: &'a Value,
    pub training: &'a [bool],
    pub out: &'a Path,
}
pub fn run(c: Case<'_>, ledger: &mut Ledger) -> Result<Value> {
    let start = Instant::now();
    let old_root = c.root.join(format!("step-{:06}", c.anchor));
    let next_root = c.root.join(format!("step-{:06}", c.end));
    let old_probe = probe(c.reference, c.anchor)?;
    let new_probe = probe(c.reference, c.end)?;
    let old = parameters(
        &ledger.read(&old_root.join("state.bin"))?,
        c.anchor,
        &old_probe["state_sha256"],
    )?;
    let next = parameters(
        &ledger.read(&next_root.join("state.bin"))?,
        c.end,
        &new_probe["state_sha256"],
    )?;
    let old_support = c
        .support
        .join(format!("{}-seed{}-step{:06}", c.name, c.seed, c.anchor));
    let prior = ledger.read(&old_support.join("margins.bin"))?;
    let support_case = c.support_report["cases"]
        .as_array()
        .ok_or("missing support cases")?
        .iter()
        .find(|r| r["cohort"] == c.name && r["seed"] == c.seed && r["step"] == c.anchor)
        .ok_or("missing prior support")?;
    require(
        prior.len() == 8 * (4 + N + 8 * N)
            && (0..4).map(|i| word(&prior, i)).collect::<Vec<_>>() == [1, N as u64, 2, 4]
            && hash(&prior) == support_case["margins_sha256"]
            && old_probe["state_sha256"] == support_case["state_sha256"],
        "prior support changed",
    )?;
    let targets: Vec<_> = (0..N).map(|q| word(&prior, 4 + q) as usize).collect();
    require(
        targets.iter().all(|&t| t < P),
        "missing training-derived target",
    )?;
    let rows: Vec<_> = (0..N)
        .flat_map(|q| [q as u32 / 97, q as u32 % 97, targets[q] as u32])
        .collect();
    let old_forward = model::forward(model::Arm::Plain, &old, &rows);
    let delta: Vec<_> = next.iter().zip(&old).map(|(n, o)| n - o).collect();
    let factors = Factors::new(crate::domain::model::Arm::Plain, &next);
    let network = network::features(
        crate::domain::model::Arm::Plain,
        &old,
        &delta,
        &rows,
        &factors,
    );
    let frozen = Frozen::new(network, &next, &rows)?;
    let upper: Vec<_> = frozen.gaps[4].iter().map(|v| v[1]).collect();
    let mut lowers = vec![vec![0.; N]; 4];
    let old_bounds: Vec<Vec<_>> = [5, 6, 7]
        .into_iter()
        .map(|m| (0..N).map(|q| scalar(&prior, 4 + N + m * N + q)).collect())
        .collect();
    let mut change = vec![0.; N];
    for q in 0..N {
        let t = targets[q];
        let mut maximum = f64::NEG_INFINITY;
        let mut shift = f64::NEG_INFINITY;
        for r in (0..P).filter(|&r| r != t) {
            maximum = maximum.max(upper[q * P + r]);
            let gap = old_forward.logits[q * P + r] - old_forward.logits[q * P + t];
            shift = shift.max(upper[q * P + r] - gap);
        }
        lowers[0][q] = -maximum;
        change[q] = shift;
        for m in 0..3 {
            lowers[m + 1][q] = old_bounds[m][q] - shift;
        }
    }
    require(
        upper
            .iter()
            .chain(lowers.iter().flatten())
            .chain(&change)
            .all(|x| x.is_finite()),
        "nonfinite forecast",
    )?;
    let mut bytes = Vec::with_capacity(8 * (4 + N * (P + 9)));
    for v in [1, N as u64, P as u64, 4] {
        bytes.extend(v.to_le_bytes());
    }
    for &t in &targets {
        bytes.extend((t as u64).to_le_bytes());
    }
    for v in upper
        .iter()
        .chain(old_bounds.iter().flatten())
        .chain(lowers.iter().flatten())
        .chain(&change)
    {
        bytes.extend(v.to_le_bytes());
    }
    fs::write(c.out.join("bounds.bin"), &bytes)?;
    let metadata = json!({"old_state_sha256":old_probe["state_sha256"],"next_state_sha256":new_probe["state_sha256"],
        "prior_sha256":hash(&prior),"bounds_sha256":hash(&bytes),"tracks":TRACKS,"bytes":bytes.len(),
        "frozen_before_endpoint_evaluation":true,"old_root":old_root,"next_root":next_root,"old_support":old_support,
        "format":"u64 LE version,queries,classes,tracks; labels[N]; f64 upper[N*97], old bounds[3*N], lower[4*N], max gap increase[N]",
        "matrix_bounds":[factors.first.json(),factors.second.json()],"parameter_displacement_norm":crate::forecast_jet::norm(&delta),
        "mean_combined_feature_radius":frozen.network.output.radii[3].iter().sum::<f64>()/N as f64});
    write(&c.out.join("frozen.json"), &metadata)?;
    let prepare_seconds = start.elapsed().as_secs_f64();
    // Only now read endpoint outcomes and independently evaluate its network.
    ledger.read(&next_root.join("queries.json"))?;
    let mut result = audit::case(c.out, c.anchor, c.end, c.training)?;
    result["prepare_seconds"] = json!(prepare_seconds);
    result["seconds"] = json!(start.elapsed().as_secs_f64());
    result["bounds_sha256"] = json!(hash(&bytes));
    result["parameter_displacement_norm"] = metadata["parameter_displacement_norm"].clone();
    result["mean_combined_feature_radius"] = metadata["mean_combined_feature_radius"].clone();
    write(&c.out.join("result.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_displacement_forecasts_enclose_an_independent_scalar_network() {
        use crate::domain::model::{Arm, initialize};
        let arm = Arm::Plain;
        let old: Vec<_> = initialize(arm, 7).into_iter().map(|v| v as f64).collect();
        let delta: Vec<_> = (0..old.len())
            .map(|k| 0.02 * ((k % 5) as f64 - 2.))
            .collect();
        let next: Vec<_> = old.iter().zip(&delta).map(|(a, d)| a + d).collect();
        let rows = [2, 7, 3, 17, 43, 5];
        let factors = Factors::new(arm, &next);
        let frozen = Frozen::new(
            network::features(arm, &old, &delta, &rows, &factors),
            &next,
            &rows,
        )
        .unwrap();
        let actual = crate::oracle::logits(arm, &next, &rows);
        for (q, t) in [3, 5].into_iter().enumerate() {
            for c in 0..97 {
                let gap = actual[q * 97 + c] - actual[q * 97 + t];
                for track in &frozen.gaps {
                    assert!(
                        track[q * 97 + c][0] <= gap + 1e-8 && gap <= track[q * 97 + c][1] + 1e-8
                    );
                }
            }
        }
    }
}
