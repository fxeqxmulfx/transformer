//! PM1-PM4 in review/flat_spectrum.md and the corresponding Lean declarations.
//! These FP64 calculations have numerical tests, not a floating-point theorem.
//! A caller must supply valid bounds over every possible future training input.

/// Sharp uniform-contrast ceiling; the log-odds implementation avoids overflow.
/// Source: Geometry.probability_le_checkpoint_ceiling. Negative budgets are valid.
pub fn ceiling(q: f64, contrast_budget: f64) -> f64 {
    assert!(q.is_finite() && (0.0..=1.0).contains(&q));
    assert!(contrast_budget.is_finite());
    if q == 0.0 || q == 1.0 {
        return q;
    }
    let log_odds = q.ln() - (-q).ln_1p() + contrast_budget;
    if log_odds >= 0.0 {
        1.0 / (1.0 + (-log_odds).exp())
    } else {
        let ratio = log_odds.exp();
        ratio / (1.0 + ratio)
    }
}

/// Retain every competitor and the class's own denominator term.
/// Source: Geometry.probability_le_motion_envelope. `budgets[u]` is unused
/// because a class's contrast with itself is identically zero.
pub fn rival_ceiling(old_logits: &[f64], u: usize, budgets: &[f64]) -> f64 {
    assert!(u < old_logits.len() && old_logits.len() == budgets.len());
    assert!(old_logits.iter().chain(budgets).all(|x| x.is_finite()));
    let relative = |c| {
        if c == u {
            0.0
        } else {
            old_logits[c] - old_logits[u] - budgets[c]
        }
    };
    let maximum = (0..old_logits.len()).map(relative).fold(0.0, f64::max);
    assert!(maximum.is_finite(), "relative-logit arithmetic overflow");
    (-maximum).exp()
        / (0..old_logits.len())
            .map(|c| (relative(c) - maximum).exp())
            .sum::<f64>()
}

/// Full finite bilinear contrast motion, with execution change retained.
/// Source: GradientEvidence.executed_contrast_motion_le. Inputs are norm caps.
pub fn contrast_radius(
    old_readout: f64,
    old_feature: f64,
    readout_motion: f64,
    feature_motion: f64,
    execution_motion: f64,
) -> f64 {
    assert!(
        [
            old_readout,
            old_feature,
            readout_motion,
            feature_motion,
            execution_motion
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0)
    );
    readout_motion * old_feature
        + old_readout * feature_motion
        + readout_motion * feature_motion
        + execution_motion
}

/// PM4's coefficient with common clipping bounded by one.
/// A value at most two controls the partial-CE reference, not the full update.
pub fn stability_number(step: f64, q: f64, largest_weight: f64, feature_energy: f64) -> f64 {
    assert!(
        [step, q, largest_weight, feature_energy]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
    );
    assert!(q <= 1.0);
    19.0 * step * q * largest_weight * feature_energy
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent direct softmax, used only at moderate logits in these tests.
    fn probability(logits: &[f64], u: usize) -> f64 {
        logits[u].exp() / logits.iter().map(|x| x.exp()).sum::<f64>()
    }
    fn close(a: f64, b: f64) {
        assert!(
            (a - b).abs() <= 2e-14 * (1.0 + a.abs().max(b.abs())),
            "{a} != {b}"
        );
    }

    #[test]
    fn nonzero_motion_attains_the_lean_witness() {
        let old = [4.0f64.ln(), 0.0];
        let new = [4.0f64.ln(), 2.0f64.ln()];
        close(probability(&old, 1), 0.2);
        close(probability(&new, 1), 1.0 / 3.0);
        close(ceiling(0.2, 2.0f64.ln()), probability(&new, 1));
        close(
            rival_ceiling(&old, 1, &[2.0f64.ln(); 2]),
            probability(&new, 1),
        );
    }

    #[test]
    fn all_competitors_and_common_shifts_are_accounted_for() {
        let old = [-2.0, 1.0, 0.3, -0.5];
        let change = [0.4, -0.1, 0.2, 0.7];
        let next: Vec<_> = old.iter().zip(change).map(|(x, d)| x + d).collect();
        for u in 0..4 {
            let exact: Vec<_> = change.iter().map(|d| change[u] - d).collect();
            close(rival_ceiling(&old, u, &exact), probability(&next, u));
            let loose: Vec<_> = exact.iter().map(|d| d + 0.1).collect();
            assert!(rival_ceiling(&old, u, &loose) >= probability(&next, u));
            let shifted: Vec<_> = old.iter().map(|x| x + 10.0).collect();
            close(
                rival_ceiling(&old, u, &loose),
                rival_ceiling(&shifted, u, &loose),
            );
            let budget = exact.iter().copied().fold(0.0, f64::max);
            let q = probability(&old, u);
            assert!(probability(&next, u) <= ceiling(q, budget) + 1e-14);
        }
        // A singleton has probability one regardless of the supplied self budget.
        assert_eq!(rival_ceiling(&[2.0], 0, &[100.0]), 1.0);
    }

    #[test]
    fn cumulative_motion_matches_composed_ceilings_with_signed_budgets() {
        for q in [1e-8, 0.01, 0.2, 0.8, 1.0 - 1e-8] {
            let budgets = [0.4, -0.3, 1.0, -0.2];
            let mut bound = q;
            let mut total = 0.0;
            for budget in budgets {
                bound = ceiling(bound, budget);
                total += budget;
                close(bound, ceiling(q, total));
            }
            for budget in [-1000.0, -2.0, 0.0, 2.0, 1000.0] {
                let result = ceiling(q, budget);
                assert!(result.is_finite() && (0.0..=1.0).contains(&result));
            }
        }
        assert_eq!(ceiling(0.0, 1000.0), 0.0);
        assert_eq!(ceiling(1.0, -1000.0), 1.0);
    }

    #[test]
    fn bilinear_bound_keeps_the_mixed_term_and_execution_error() {
        let old = 2.0 * 1.0;
        let new = 3.0 * 2.0;
        assert_eq!(new - old, contrast_radius(2.0, 1.0, 1.0, 1.0, 0.0));
        // Zero old vectors have no first-order term; their joint motion is nonzero.
        assert_eq!(contrast_radius(0.0, 0.0, 2.0, 3.0, 0.1), 6.1);
        assert!(stability_number(0.01, ceiling(0.2, 2.0f64.ln()), 2.0, 5.0) <= 2.0);
        assert!(stability_number(0.01, 1.0, 2.0, 128.0) > 2.0);
    }
}
