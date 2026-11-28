//! A minibatch of unlabeled concatenations supplies a loss on the shape
//! of the model's logit table. Only the 31 observed words supply CE labels.
use super::{LENGTH, word_tokens};
use crate::{Result, engine::Engine, pcg::Pcg, rank_one_loss};
use serde_json::{Value, json};
use std::{fs, path::Path};

pub(super) const SIDE: usize = 5;

pub(super) fn accumulate(model: &mut Engine, rng: &mut Pcg) -> Result<f64> {
    let mut rows: [u32; 16] = std::array::from_fn(|i| i as u32);
    let mut cols = rows;
    rng.shuffle(&mut rows);
    rng.shuffle(&mut cols);
    let mut tokens = Vec::with_capacity(SIDE * SIDE * LENGTH);
    for &row in &rows[..SIDE] {
        for &col in &cols[..SIDE] {
            tokens.extend(word_tokens(16 * row + col));
        }
    }
    let labels = [u32::MAX; SIDE * SIDE * LENGTH];
    model.forward(&tokens, &labels, SIDE * SIDE, LENGTH)?;
    let logits = model
        .logits
        .part(..2 * tokens.len())
        .ok_or("rectangle logit capacity")?
        .read()?;
    let gaps: Vec<_> = logits
        .chunks_exact(2 * LENGTH)
        .map(|word| f64::from(word[2 * LENGTH - 2]) - f64::from(word[2 * LENGTH - 1]))
        .collect();
    let (loss, dgap) = rank_one_loss::value_gradient(&gaps, SIDE, SIDE)?;
    let mut dlogits = vec![0.0; logits.len()];
    for (word, grad) in dlogits.chunks_exact_mut(2 * LENGTH).zip(dgap) {
        word[2 * LENGTH - 2] = grad as f32;
        word[2 * LENGTH - 1] = -grad as f32;
    }
    model.accumulate_logit_gradient(&dlogits)?;
    Ok(loss)
}

/// Check the additive sign identity on the complete logit table. A tie
/// precludes strict recovery. Only the supplied staircase signs are read.
fn audit_gaps(gaps: &[f64], diagonal: &[f64; 16], edge: &[f64; 15]) -> Result<Value> {
    let (global_loss, _) = rank_one_loss::value_gradient(gaps, 16, 16)?;
    let absolute_margin = gaps.iter().map(|z| z.abs()).fold(f64::INFINITY, f64::min);
    let correct_observed = (0..16)
        .filter(|&i| gaps[17 * i] * diagonal[i] > 0.0)
        .count()
        + (0..15)
            .filter(|&i| gaps[17 * i + 1] * edge[i] > 0.0)
            .count();
    let mut violated = 0;
    let mut tied = 0;
    for i in 0..16 {
        for k in i + 1..16 {
            for j in 0..16 {
                for l in j + 1..16 {
                    let corners = [
                        gaps[16 * i + j],
                        gaps[16 * k + l],
                        gaps[16 * i + l],
                        gaps[16 * k + j],
                    ];
                    if corners.contains(&0.0) {
                        tied += 1;
                    } else if corners.iter().filter(|&&x| x < 0.0).count() % 2 == 1 {
                        violated += 1;
                    }
                }
            }
        }
    }
    Ok(
        json!({"rectangles":14400,"violated_strict_rectangles":violated,"rectangles_with_ties":tied,
        "correct_observed":correct_observed,"minimum_absolute_gap":absolute_margin,
        "full_table_half_mean_squared_rank_one_residual":global_loss,
        "sufficient_mean_patch_loss_below":absolute_margin.powi(2)/7200.0,
        "all_words_certified":correct_observed==31 && tied==0 && violated==0}),
    )
}

pub(crate) fn audit(cell: &Path) -> Result<Value> {
    let record: Value = serde_json::from_slice(&fs::read(cell.join("result.json"))?)?;
    if record["hidden"]["total"] != 225 || record["observed"]["total"] != 31 {
        return Err("expected an eight-bit sum checkpoint".into());
    }
    let mut model = super::new_engine(super::read_seed(&record["seeds"], 0)?)?;
    crate::suite::load_weights(cell, "weights.f32", &record["weights_sha256"], &mut model)?;
    let (diagonal, edge) = super::observed_phase_labels();
    let report = audit_gaps(
        &crate::sum8_spectrum::logit_gaps(&mut model)?,
        &diagonal,
        &edge,
    )?;
    let all_correct =
        record["observed"]["all_strict"] == true && record["hidden"]["all_strict"] == true;
    if report["all_words_certified"] != all_correct
        || report["correct_observed"] != record["observed"]["correct"]
    {
        return Err("rectangle certificate disagrees with the independent label replay".into());
    }
    Ok(
        json!({"weights_sha256":record["weights_sha256"],"step":record["optimizer"]["step"],"certificate":report}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_flipped_hidden_sign_violates_exactly_225_rectangles() -> Result<()> {
        let (d, e) = super::super::observed_phase_labels();
        let mut x = crate::sum8_spectrum::staircase_signs(&d, &e);
        assert_eq!(audit_gaps(&x, &d, &e)?["all_words_certified"], true);
        x[8] *= -1.0;
        let bad = audit_gaps(&x, &d, &e)?;
        assert_eq!(bad["correct_observed"], 31);
        assert_eq!(bad["violated_strict_rectangles"], 225);
        x[8] = 0.0;
        let tied = audit_gaps(&x, &d, &e)?;
        assert_eq!(tied["all_words_certified"], false);
        assert_eq!(tied["rectangles_with_ties"], 225);
        Ok(())
    }
}
