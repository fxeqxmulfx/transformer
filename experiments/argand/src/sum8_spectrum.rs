//! Score the learned phase table from weights alone. The binary phase code
//! sends token 0 to +1 and token 1 to -1, so E[code] is tanh((z0-z1)/2).
use crate::{
    Result,
    engine::Engine,
    suite::{load_weights, publish},
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn table(engine: &mut Engine) -> Result<Vec<f64>> {
    let mut table = Vec::with_capacity(256);
    for start in (0..256u32).step_by(engine.config.max_batch) {
        let stop = (start + engine.config.max_batch as u32).min(256);
        let tokens: Vec<_> = (start..stop)
            .flat_map(|w| (0..8).rev().map(move |b| (w >> b) & 1))
            .collect();
        engine.forward(
            &tokens,
            &vec![u32::MAX; tokens.len()],
            (stop - start) as usize,
            8,
        )?;
        let logits = engine
            .logits
            .part(..2 * tokens.len())
            .ok_or("logit capacity")?
            .read()?;
        table.extend(
            logits
                .as_chunks::<16>()
                .0
                .iter()
                .map(|z| ((f64::from(z[14]) - f64::from(z[15])) * 0.5).tanh()),
        );
    }
    if table.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite phase table".into());
    }
    Ok(table)
}

fn analyze(x: &[f64], diagonal: &[f64; 16], edge: &[f64; 15]) -> Value {
    let s = crate::svd::thin(x, 16, 16);
    let (mut residual, mut min_factor) = (0.0_f64, f64::INFINITY);
    let mut min_slack = f64::INFINITY;
    let mut worst_decoded = 0.0_f64;
    let mut certified_cells = 0;
    // Fix the first column sign and propagate the 31 supplied signs along the path.
    let mut cols = [1.0; 16];
    for i in 0..15 {
        cols[i + 1] = cols[i] * diagonal[i] * edge[i];
    }
    let mut observed_signs = true;
    let mut worst_seen = 0.0_f64;
    for (i, &value) in x.iter().enumerate() {
        let (row, col) = (i / 16, i % 16);
        let rank_one = s.values[0] * s.left[16 * row] * s.right[col];
        residual = residual.max((value - rank_one).abs());
        min_factor = min_factor.min(rank_one.abs());
        min_slack = min_slack.min(rank_one.abs() - (value - rank_one).abs());
        let decoded = diagonal[row] * cols[row] * cols[col];
        worst_decoded = worst_decoded.max((value - decoded).abs());
        certified_cells += usize::from(value * decoded > 0.0);
        if col == row || col == row + 1 {
            let target = if col == row { diagonal[row] } else { edge[row] };
            observed_signs &= target * rank_one > 0.0;
            worst_seen = worst_seen.max((value - target).abs());
        }
    }
    json!({"singular_values":s.values,"nuclear_norm":s.values.iter().sum::<f64>(),
        "target_nuclear_norm":16,"max_observed_phase_error":worst_seen,"rank_one_fits_staircase":observed_signs,
        "min_rank_one_magnitude":min_factor,"max_entrywise_residual":residual,
        "min_entrywise_svd_slack":min_slack,"svd_sign_certificate":observed_signs && min_slack>0.0,
        "rank_one_sign_certificate":observed_signs && residual<min_factor,
        "unit_factor_max_residual":worst_decoded,"unit_factor_strict_cells":certified_cells,
        "unit_factor_certificate":certified_cells==256})
}

pub fn run(dir: &Path) -> Result<()> {
    let (diagonal, edge) = crate::sum8::observed_phase_labels();
    let cohort: Value = serde_json::from_slice(&fs::read(dir.join("summary.json"))?)?;
    if cohort["replayed"] != true || cohort["protocol"]["open_words"] != 31 {
        return Err("expected a replayed staircase cohort".into());
    }
    let mut output = Vec::new();
    for (i, r) in cohort["records"]
        .as_array()
        .ok_or("missing draws")?
        .iter()
        .enumerate()
    {
        let mut engine = crate::sum8::new_engine(0)?;
        load_weights(
            &dir.join(format!("draw-{i}")),
            "weights.f32",
            &r["weights_sha256"],
            &mut engine,
        )?;
        let spectrum = analyze(&table(&mut engine)?, &diagonal, &edge);
        if spectrum["unit_factor_strict_cells"].as_u64()
            != Some(
                r["observed"]["correct"]
                    .as_u64()
                    .ok_or("missing observed score")?
                    + r["hidden"]["correct"]
                        .as_u64()
                        .ok_or("missing hidden score")?,
            )
        {
            return Err(
                "staircase-only phase certificate disagrees with the strict-label replay".into(),
            );
        }
        output.push(json!({"draw":i,"step":r["optimizer"]["step"],"weights_sha256":r["weights_sha256"],"spectrum":spectrum}));
    }
    publish(dir, "spectra.json", &serde_json::to_vec_pretty(&output)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parity_has_one_singular_value_of_sixteen_and_a_strict_certificate() {
        let signs: Vec<_> = (0..256_u32)
            .map(|v| {
                if v.count_ones().is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let (diagonal, edge) = crate::sum8::observed_phase_labels();
        let result = analyze(&signs, &diagonal, &edge);
        assert!(result["rank_one_sign_certificate"] == true);
        assert!((result["nuclear_norm"].as_f64().unwrap() - 16.0).abs() < 1e-12);
    }
}
