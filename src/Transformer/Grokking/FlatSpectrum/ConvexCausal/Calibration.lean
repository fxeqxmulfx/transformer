module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Optimality

/-!
# Stochastic outputs, calibration and the confidence limit

Source: `review/convex_causal.md`, CCa6. Deterministic controls and noisy
forecasting have different optimal conditional distributions. The repaired
readout represents every distribution above its declared floor. Expected
ordinary CE is minimized at the true conditional distribution when it is
representable. This is a population statement, not a claim that finite data
identify an unknown distribution. The deterministic loss floor tends to zero.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators

variable {X C : Type*} [Fintype C] [Nonempty C]

/-- Inverse coordinates for a requested distribution above the floor (CCa6). -/
noncomputable def unsmooth (ε : ℝ) (P : X → C → ℝ) (x : X) (c : C) : ℝ :=
  (P x c - ε / Fintype.card C) / (1 - ε)

/-- The repaired output is fully calibrated above its fixed floor (CCa6). -/
theorem unsmooth_rows {P : X → C → ℝ} (hP : Rows P) {ε : ℝ} (hε : ε < 1)
    (hfloor : ∀ x c, ε / Fintype.card C ≤ P x c) : Rows (unsmooth ε P) := by
  have hn : (Fintype.card C : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have he : 1 - ε ≠ 0 := ne_of_gt (sub_pos.mpr hε)
  constructor
  · intro x c
    exact div_nonneg (sub_nonneg.mpr (hfloor x c)) (sub_pos.mpr hε).le
  · intro x
    simp only [unsmooth, ← Finset.sum_div, Finset.sum_sub_distrib, hP.2,
      Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    field_simp

/-- Executing the readout returns the requested conditional probabilities exactly (CCa6). -/
theorem smooth_unsmooth (P : X → C → ℝ) {ε : ℝ} (hε : ε < 1) :
    smooth ε (unsmooth ε P) = P := by
  have he : 1 - ε ≠ 0 := ne_of_gt (sub_pos.mpr hε)
  funext x c
  unfold smooth unsmooth
  field_simp
  ring

/-- Conditional expected negative log likelihood over all output classes (CCa6). -/
noncomputable def expected (p q : C → ℝ) : ℝ := ∑ c, p c * (-Real.log (q c))

omit [Nonempty C] in
/-- Normalized expected CE is minimized by the conditional distribution itself (CCa6).
Both distributions are positive so all displayed finite logits are defined. -/
theorem expected_minimum {p q : C → ℝ} (hp : ∀ c, 0 < p c) (hq : ∀ c, 0 < q c)
    (hpsum : ∑ c, p c = 1) (hqsum : ∑ c, q c = 1) : expected p p ≤ expected p q := by
  have hc (c : C) : p c * Real.log (q c) - p c * Real.log (p c) ≤ q c - p c := by
    have hl := Real.log_le_sub_one_of_pos (div_pos (hq c) (hp c))
    rw [Real.log_div (hq c).ne' (hp c).ne'] at hl
    have h := mul_le_mul_of_nonneg_left hl (hp c).le
    have he : p c * (q c / p c - 1) = q c - p c := by
      field_simp [ne_of_gt (hp c)]
    rw [he] at h
    nlinarith only [h]
  have hs := Finset.sum_le_sum (fun c (_ : c ∈ (Finset.univ : Finset C)) => hc c)
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, hqsum, hpsum] at hs
  simp only [expected, mul_neg, Finset.sum_neg_distrib]
  linarith

/-- The stochastic optimum is for the existing ordinary softmax CE (CCa6). -/
theorem expected_crossEntropy {P Q : X → C → ℝ} (hP : Rows P) (hQ : Rows Q)
    {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1) (x : X) :
    (∑ c, smooth ε P x c * Landscape.Neural.crossEntropy
      (fun d => Real.log (smooth ε P x d)) c) ≤
    ∑ c, smooth ε P x c * Landscape.Neural.crossEntropy
      (fun d => Real.log (smooth ε Q x d)) c := by
  simp_rw [crossEntropy_eq hP hε hε1, crossEntropy_eq hQ hε hε1]
  exact expected_minimum (smooth_pos hP hε hε1 x) (smooth_pos hQ hε hε1 x)
    (smooth_sum hP ε x) (smooth_sum hQ ε x)

/-- Shrinking the fixed probability floor removes the unit-logit CE ceiling (CCa6).
The theorem concerns a family of convex problems, not joint optimization of the floor. -/
theorem ideal_tendsto_zero : Filter.Tendsto (ideal C) (nhds (0 : ℝ)) (nhds 0) := by
  have hc : ContinuousAt (ideal C) 0 := by
    unfold ideal
    fun_prop (disch := norm_num)
  simpa [ideal] using hc.tendsto

omit [Nonempty C] in
/-- The Bayes accuracy on an independent uniform next token is exactly `1 / |C|` (CCa6).
The prediction can itself be randomized; no deterministic learning algorithm escapes this. -/
theorem uniform_accuracy {q : C → ℝ} (hq : ∑ c, q c = 1) :
    (∑ c, q c * (Fintype.card C : ℝ)⁻¹) = (Fintype.card C : ℝ)⁻¹ := by
  rw [← Finset.sum_mul, hq, one_mul]

example := unsmooth_rows (P := fun (_ : Unit) (c : Bool) => if c then 3 / 4 else 1 / 4)
  (by
    constructor
    · intro x c; cases c <;> norm_num
    · intro x; norm_num [Fintype.sum_bool])
  (ε := 1 / 4) (by norm_num) (by intro x c; cases c <;> norm_num)

example := smooth_unsmooth (fun (_ : Unit) (c : Bool) => if c then 3 / 4 else 1 / 4)
  (ε := 1 / 4) (by norm_num)

example := expected_minimum
  (p := fun c : Bool => if c then 3 / 4 else 1 / 4) (q := fun _ => 1 / 2)
  (by intro c; cases c <;> norm_num) (by intro c; norm_num)
  (by norm_num [Fintype.sum_bool]) (by norm_num [Fintype.sum_bool])

example := expected_crossEntropy (rows_point (id : Bool → Bool)) (rows_point Bool.not)
  (ε := 1 / 4) (by norm_num) (by norm_num) true

example : (∑ c : Bool, (if c then (3 / 4 : ℝ) else 1 / 4) * (2 : ℝ)⁻¹) = 1 / 2 := by
  simpa using uniform_accuracy (q := fun c : Bool => if c then 3 / 4 else 1 / 4)
    (by norm_num [Fintype.sum_bool])

end Transformer.Grokking.FlatSpectrum.ConvexCausal
