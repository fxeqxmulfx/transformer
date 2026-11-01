module

public import Transformer.SequenceModel.Attention.Execution.Rounding

/-!
# Check normalizer outputs by algebraic residuals

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K1--K2;
diff_gpt b98c665b, `gpt.py`, `block.py`, `mha.py`. A recorded output is
compared with a chosen nonnegative scale times its own input. A squared
residual certifies that scale without assuming a hardware sqrt/rsqrt error.
The scale is a certificate witness, not an assertion about a fused kernel's
internal instruction sequence. No new normalizer-error hypothesis is used.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open GPTMini Execution

/-- K1: a nonnegative approximate unit has error bounded by its square residual. -/
theorem unit_error_of_square (a : ℝ) (ha : 0 ≤ a) : |a - 1| ≤ |a ^ 2 - 1| := by
  by_cases h : 1 ≤ a
  · rw [abs_of_nonneg (by linarith), abs_of_nonneg (by nlinarith)]
    nlinarith
  · rw [abs_of_nonpos (by linarith), abs_of_nonpos (by nlinarith)]
    nlinarith

/-- K1: product error and an algebraic scale residual certify a normalized
vector. The bound uses the normalized norm, not inverse epsilon amplification. -/
theorem normalization_residual {d : ℕ} (x z : EucSpace d) (a b β ρ B : ℝ)
    (ha : 0 ≤ a) (hb : 0 < b) (hr : |a ^ 2 * b ^ 2 - 1| ≤ ρ)
    (hz : ‖z - a • x‖ ≤ β) (hn : ‖(1 / b) • x‖ ≤ B) :
    ‖z - (1 / b) • x‖ ≤ β + ρ * B := by
  have hρ : 0 ≤ ρ := (abs_nonneg _).trans hr
  have hs : |a * b - 1| ≤ ρ := by
    have h := unit_error_of_square (a * b) (mul_nonneg ha hb.le)
    rw [mul_pow] at h
    exact h.trans hr
  have he : a • x - (1 / b) • x = (a * b - 1) • ((1 / b) • x) := by
    rw [smul_smul, ← sub_smul]
    congr 1
    field_simp
  have hn' : ‖a • x - (1 / b) • x‖ ≤ ρ * B := by
    rw [he, norm_smul, Real.norm_eq_abs]
    exact mul_le_mul hs hn (norm_nonneg _) hρ
  exact (norm_sub_le_norm_sub_add_norm_sub z (a • x) ((1 / b) • x)).trans (add_le_add hz hn')

/-- K2: the RMS scale check is polynomial in the recorded input, its scale
and epsilon. Together with the rounded multiplication residual it discharges
the actual RMSNorm operation contract, including its source epsilon. -/
theorem rms_residual_error {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (x z : EucSpace d) (a β ρ : ℝ) (ha : 0 ≤ a)
    (hr : |a ^ 2 * (‖x‖ ^ 2 + (d : ℝ) * eps) / d - 1| ≤ ρ)
    (hz : ‖z - a • x‖ ≤ β) :
    ‖z - rmsNormEps eps x‖ ≤ β + ρ * Real.sqrt (d : ℝ) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hq : 0 < ‖x‖ ^ 2 + (d : ℝ) * eps := by positivity
  let b := Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) / Real.sqrt (d : ℝ)
  have hb : 0 < b := by dsimp [b]; positivity
  have he : (1 / b) • x = rmsNormEps eps x := by
    dsimp [b, rmsNormEps]
    congr 1
    field_simp
  rw [← he]
  apply normalization_residual x z a b β ρ (Real.sqrt (d : ℝ)) ha hb
  · simpa only [b, div_pow, Real.sq_sqrt hq.le, Real.sq_sqrt hd'.le, mul_div_assoc] using hr
  · exact hz
  · rw [he]
    exact rmsNormEps_norm_le eps heps hd x

/-- K2: the same check handles clipped L2 normalization, retaining the max
with epsilon. On either side of the clipping boundary its squared denominator
can be checked without trusting the implementation's square root. -/
theorem l2_residual_error {d : ℕ} {eps : ℝ} (heps : 0 < eps)
    (x z : EucSpace d) (a β ρ : ℝ) (ha : 0 ≤ a)
    (hr : |a ^ 2 * (max ‖x‖ eps) ^ 2 - 1| ≤ ρ) (hz : ‖z - a • x‖ ≤ β) :
    ‖z - normL2 eps x‖ ≤ β + ρ := by
  have h := normalization_residual x z a (max ‖x‖ eps) β ρ 1 ha
    (lt_max_of_lt_right heps) hr hz (normL2_norm_le eps heps.le x)
  simpa only [normL2, mul_one] using h

example := unit_error_of_square (3 / 4) (by norm_num)
example := normalization_residual (EuclideanSpace.single 0 1 : EucSpace 4)
  ((3 / 4 : ℝ) • EuclideanSpace.single 0 1) (3 / 4) 1 0 (7 / 16) 1
  (by norm_num) (by norm_num) (by norm_num) (by simp) (by simp)
example := rms_residual_error (d := 4) (by decide) (eps := 3 / 4) (by norm_num)
  (EuclideanSpace.single 0 1) (EuclideanSpace.single 0 1) 1 0 0 (by norm_num)
  (by norm_num) (by simp)
example := l2_residual_error (eps := 2) (by norm_num)
  (EuclideanSpace.single 0 1 : EucSpace 4) ((1 / 2 : ℝ) • EuclideanSpace.single 0 1)
  (1 / 2) 0 0 (by norm_num) (by norm_num) (by simp)

end Transformer.SequenceModel.Attention.Backend
