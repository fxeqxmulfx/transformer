module

public import Transformer.Rotate90.ComplexProduct.Readout

/-!
# A concrete normalized source classifier

Source: `review/flat_spectrum.md`, *Complex-product Rotate90 bridge*, CP5.
Two opposite unit embeddings classify the first operand when the second is zero.
The zero output matrix makes this a residual-path witness, not a claim that the
product branch discovers modular features. `activeWeights` separately exercises
the real/complex correspondence with nonzero complex matrices.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open scoped BigOperators
open Grokking.FlatSpectrum

/-- A nonconstant two-label source instance for CP5. -/
def copyWeights : Weights (Fin 2) (Fin 1) (Fin 1) :=
  { embedding := fun q _ => if q = 0 then 1 else -1
    gate := fun _ _ => 1
    value := fun _ _ => 1
    output := fun _ _ => 0 }

/-- Nonzero complex parameters used to instantiate all algebraic identities (CP1). -/
def activeWeights : Weights (Fin 2) (Fin 1) (Fin 1) where
  embedding q _ := if q = 0 then 1 + Complex.I else 2 - Complex.I
  gate _ _ := 2 + Complex.I
  value _ _ := 1 - Complex.I
  output _ _ := 1 + 2 * Complex.I

/-- The computed residual state of CP5, with positive epsilon retained. -/
theorem copy_state (q : Fin 2) :
    state 1 copyWeights q 0 =
      fun _ => ((if q = 0 then 1 else -1) + Complex.I) / (Real.sqrt 2 : ℂ) := by
  funext k
  fin_cases q <;>
    norm_num [state, copyWeights, branch, denseC, product, norm, denominator,
      energy, Complex.normSq_apply]

/-- Actual raw tied scores for the two different inputs in CP5. -/
theorem copy_readout (q v : Fin 2) :
    readout copyWeights.embedding (state 1 copyWeights q 0) v =
      (if v = 0 then 1 else -1) * (if q = 0 then 1 else -1) / Real.sqrt 2 := by
  rw [copy_state]
  fin_cases q <;> fin_cases v <;> simp [readout, copyWeights] <;> ring

/-- Both different labels have a strictly positive raw margin (CP5). -/
theorem copy_gap (q v : Fin 2) (hv : v ≠ q) :
    readout copyWeights.embedding (state 1 copyWeights q 0) q -
      readout copyWeights.embedding (state 1 copyWeights q 0) v = 2 / Real.sqrt 2 := by
  rw [copy_readout, copy_readout]
  fin_cases q <;> fin_cases v <;> norm_num at * <;> field_simp <;>
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]

/-- Unit-modulus codes in the concrete source witness (CP5). -/
theorem copy_unit (q : Fin 2) (k : Fin 1) :
    Complex.normSq (copyWeights.embedding q k) = 1 := by
  fin_cases q <;> norm_num [copyWeights]

/-- The ideal code margin of CP5 is two, with two genuinely different labels. -/
theorem copy_code_gap (q v : Fin 2) (hv : v ≠ q) :
    FeatureCodes.codeScore (fun _ => 1) copyWeights.embedding q q -
      FeatureCodes.codeScore (fun _ => 1) copyWeights.embedding q v = 2 := by
  fin_cases q <;> fin_cases v <;>
    norm_num [FeatureCodes.codeScore, FeatureCodes.score, copyWeights] at *

/-- The computed state is within 4/5 of the correct unit code (CP5). -/
theorem copy_feature_error (q : Fin 2) :
    ∑ k, ‖state 1 copyWeights q 0 k - copyWeights.embedding q k‖ ≤ (4 / 5 : ℝ) := by
  rw [copy_state]
  simp only [Fin.sum_univ_one]
  have hs : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hb : (7 / 5 : ℝ) < Real.sqrt 2 :=
    (Real.lt_sqrt (by norm_num)).mpr (by norm_num)
  have hsq : ∀ s : ℂ, s = 1 ∨ s = -1 →
      Complex.normSq ((s + Complex.I) / (Real.sqrt 2 : ℂ) - s) = 2 - Real.sqrt 2 := by
    intro s h
    rcases h with rfl | rfl <;>
      simp [Complex.normSq_apply] <;> field_simp <;> nlinarith
  have he : ∀ s : ℂ, s = 1 ∨ s = -1 →
      ‖(s + Complex.I) / (Real.sqrt 2 : ℂ) - s‖ ≤ (4 / 5 : ℝ) := by
    intro s h
    have hnorm := hsq s h
    rw [Complex.normSq_eq_norm_sq] at hnorm
    nlinarith [norm_nonneg ((s + Complex.I) / (Real.sqrt 2 : ℂ) - s)]
  fin_cases q
  · simpa [copyWeights] using he 1 (Or.inl rfl)
  · simpa [copyWeights] using he (-1) (Or.inr rfl)

/-- The feature-margin bridge classifies both inputs in the expanded real model (CP5). -/
theorem copy_classifies : ∀ q v : Fin 2, v ≠ q →
    realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  intro q
  apply realLogit_correct_of_feature_error (by norm_num) copyWeights q 0 q copy_unit
    (δ := 2) (η := 4 / 5) ?_ (copy_feature_error q) (by norm_num)
  intro v hv
  rw [copy_code_gap q v hv]

example := realify_branch activeWeights (fun _ => 1 + Complex.I)
example := realify_state 1 activeWeights 0 1
example := realLogit_eq_logit 1 activeWeights 0 1 0
example := realLogit_eq_readout_div 1 activeWeights 0 1 0
example := realLogit_lt_iff (by norm_num : (0 : ℝ) < 1) activeWeights 0 1 0 1
example := copy_state 0
example := copy_readout 1 0
example := copy_gap 0 1 (by decide)
example := copy_unit 0 0
example := copy_code_gap 0 1 (by decide)
example := copy_feature_error 1
example := copy_classifies 0 1 (by decide)

end Transformer.Rotate90.ComplexProduct
