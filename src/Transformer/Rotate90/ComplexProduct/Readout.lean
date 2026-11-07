module

public import Transformer.Rotate90.ComplexProduct.Basic

/-!
# Source margins and numerical decision tolerance

Source: `review/flat_spectrum.md`, *Complex-product Rotate90 bridge*, CP2–CP3.
A margin is measured before the final RMSNorm, on the actual residual state.
The same positive denominator divides every class logit. Thus a raw margin d
tolerates per-class execution error e whenever 2 e times that denominator is
less than d. Execution accuracy remains an explicit premise.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open scoped BigOperators
open Grokking.FlatSpectrum

variable {V K H : Type*} [Fintype K] [Fintype H]

/-- Exact pre-final-normalization expression for the implemented logits (CP2). -/
theorem realLogit_eq_readout_div (eps : ℝ) (w : Weights V K H) (a b q : V) :
    realLogit eps w a b q =
      readout w.embedding (state eps w a b) q / denominator eps (state eps w a b) := by
  rw [realLogit_eq_logit, logit, readout_norm]

/-- Positive final RMSNorm preserves every strict raw-score comparison (CP2). -/
theorem realLogit_lt_iff {eps : ℝ} (heps : 0 < eps) (w : Weights V K H)
    (a b c t : V) :
    realLogit eps w a b c < realLogit eps w a b t ↔
      readout w.embedding (state eps w a b) c < readout w.embedding (state eps w a b) t := by
  rw [realLogit_eq_readout_div, realLogit_eq_readout_div]
  exact div_lt_div_iff_of_pos_right (denominator_pos heps _)

/-- **A source margin certifies an accurate execution**, source CP3.
The premise compares actual executed scores against the independently defined
expanded source model. Both competitors' errors and the true denominator count. -/
theorem executed_correct_of_raw_margin {eps : ℝ} (heps : 0 < eps)
    (w : Weights V K H) (a b target : V) (executed : V → ℝ) {δ η : ℝ}
    (hgap : ∀ c, c ≠ target → δ ≤
      readout w.embedding (state eps w a b) target - readout w.embedding (state eps w a b) c)
    (herr : ∀ c, |executed c - realLogit eps w a b c| ≤ η)
    (hbudget : 2 * η * denominator eps (state eps w a b) < δ) :
    ∀ c, c ≠ target → executed c < executed target := by
  have hd := denominator_pos heps (state eps w a b)
  apply FeatureCodes.strict_winner_of_score_error
    (s := realLogit eps w a b) (δ := δ / denominator eps (state eps w a b)) ?_ herr
    ((lt_div_iff₀ hd).mpr hbudget)
  intro c hc
  rw [realLogit_eq_readout_div, realLogit_eq_readout_div, ← sub_div]
  exact (div_le_div_iff_of_pos_right hd).mpr (hgap c hc)

/-- **Approximate separating features classify in the real source model**, source CP2.
The error is on its computed residual state, not on an abstract vector substituted
for the model. Unit codes and the weighted-margin bound are explicit conditions. -/
theorem realLogit_correct_of_feature_error {eps : ℝ} (heps : 0 < eps)
    (w : Weights V K H) (a b target : V)
    (hunit : ∀ q k, Complex.normSq (w.embedding q k) = 1) {δ η : ℝ}
    (hgap : ∀ c, c ≠ target → δ ≤
      FeatureCodes.codeScore (fun _ => 1) w.embedding target target -
        FeatureCodes.codeScore (fun _ => 1) w.embedding target c)
    (herr : ∑ k, ‖state eps w a b k - w.embedding target k‖ ≤ η)
    (hbudget : 2 * η < δ) :
    ∀ c, c ≠ target → realLogit eps w a b c < realLogit eps w a b target := by
  have h := FeatureCodes.strict_winner_of_feature_error
    (w := fun _ => 1) (fun _ => zero_le_one) w.embedding hunit (state eps w a b) target
    hgap (by simpa only [one_mul] using herr) hbudget
  intro c hc
  apply (realLogit_lt_iff heps w a b c target).mpr
  simpa only [FeatureCodes.score, readout, one_mul] using h c hc

end Transformer.Rotate90.ComplexProduct
