module

public import Transformer.SequenceModel.Attention.Execution.Readout

/-!
# Normalization sensitivity at a recorded input

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M1.
The RMS proof refines `GPTMini.rmsNormEps_lipschitz` by retaining its
first denominator. The clipped L2 proof likewise retains max(norm(x), eps).
Both estimates hold for every other input y, including zero and clip crossings.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Adaptive
open GPTMini
variable {d : ℕ}

/-- M1: an asymmetric RMS secant bound using the recorded input denominator. -/
theorem rms_secant (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (x y : EucSpace d) :
    ‖rmsNormEps eps x - rmsNormEps eps y‖ ≤ 2 * Real.sqrt (d : ℝ) / Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) * ‖x - y‖ := by
  have hd' : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have hdeps : (0 : ℝ) < (d : ℝ) * eps := mul_pos hd' heps
  set a := ‖x‖ ^ 2 + (d : ℝ) * eps with ha
  set b := ‖y‖ ^ 2 + (d : ℝ) * eps with hb
  have hapos : 0 < a := add_pos_of_nonneg_of_pos (sq_nonneg _) hdeps
  have hbpos : 0 < b := add_pos_of_nonneg_of_pos (sq_nonneg _) hdeps
  have hsa : 0 < Real.sqrt a := Real.sqrt_pos.mpr hapos
  have hsb : 0 < Real.sqrt b := Real.sqrt_pos.mpr hbpos
  have hsd : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr hd'
  -- The two vectors sit inside their own denominators.
  have hnx : ‖x‖ ≤ Real.sqrt a :=
    (Real.le_sqrt (norm_nonneg _) hapos.le).mpr (by rw [ha]; linarith)
  have hny : ‖y‖ ≤ Real.sqrt b :=
    (Real.le_sqrt (norm_nonneg _) hbpos.le).mpr (by rw [hb]; linarith)
  -- `|√b - √a| (√a + √b) = |b - a| ≤ ‖x - y‖ (‖x‖ + ‖y‖)`.
  have hsq : (Real.sqrt b - Real.sqrt a) * (Real.sqrt a + Real.sqrt b) = b - a := by
    have h1 : Real.sqrt a ^ 2 = a := Real.sq_sqrt hapos.le
    have h2 : Real.sqrt b ^ 2 = b := Real.sq_sqrt hbpos.le
    linear_combination h2 - h1
  have hba : |b - a| ≤ ‖x - y‖ * (‖x‖ + ‖y‖) := by
    have hdiff : b - a = (‖y‖ - ‖x‖) * (‖x‖ + ‖y‖) := by rw [ha, hb]; ring
    have hnorm : |‖y‖ - ‖x‖| ≤ ‖x - y‖ := by
      rw [← norm_neg (x - y), neg_sub]
      exact abs_norm_sub_norm_le y x
    rw [hdiff, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖x‖ + ‖y‖)]
    exact mul_le_mul_of_nonneg_right hnorm (by positivity)
  have h1 : |Real.sqrt b - Real.sqrt a| * (Real.sqrt a + Real.sqrt b)
      ≤ ‖x - y‖ * (‖x‖ + ‖y‖) := by
    rw [← abs_of_pos (show (0 : ℝ) < Real.sqrt a + Real.sqrt b by linarith), ← abs_mul, hsq]
    exact hba
  have h3 : |Real.sqrt b - Real.sqrt a| * ‖y‖ ≤ ‖x - y‖ * Real.sqrt b := by
    refine le_of_mul_le_mul_right ?_
      (show (0 : ℝ) < Real.sqrt a + Real.sqrt b by linarith)
    have hy := mul_le_mul_of_nonneg_right h1 (norm_nonneg y)
    have hz : ‖x - y‖ * ((‖x‖ + ‖y‖) * ‖y‖)
        ≤ ‖x - y‖ * ((Real.sqrt a + Real.sqrt b) * Real.sqrt b) :=
      mul_le_mul_of_nonneg_left
        (mul_le_mul (by linarith) hny (norm_nonneg _) (by linarith)) (norm_nonneg _)
    nlinarith [hy, hz]
  -- The coefficient term.
  have hrw : |Real.sqrt (d : ℝ) / Real.sqrt a - Real.sqrt (d : ℝ) / Real.sqrt b|
      = Real.sqrt (d : ℝ) * |Real.sqrt b - Real.sqrt a| / (Real.sqrt a * Real.sqrt b) := by
    rw [div_sub_div _ _ hsa.ne' hsb.ne', abs_div, abs_of_pos (mul_pos hsa hsb),
      show Real.sqrt (d : ℝ) * Real.sqrt b - Real.sqrt a * Real.sqrt (d : ℝ)
        = Real.sqrt (d : ℝ) * (Real.sqrt b - Real.sqrt a) from by ring,
      abs_mul, abs_of_pos hsd]
  have hcoef : |Real.sqrt (d : ℝ) / Real.sqrt a - Real.sqrt (d : ℝ) / Real.sqrt b| * ‖y‖
      ≤ Real.sqrt (d : ℝ) / Real.sqrt a * ‖x - y‖ := by
    rw [hrw, div_mul_eq_mul_div, div_mul_eq_mul_div,
      div_le_div_iff₀ (mul_pos hsa hsb) hsa]
    have hh := mul_le_mul_of_nonneg_left h3 (mul_nonneg hsd.le hsa.le)
    nlinarith
  -- Assemble.
  have hdecomp : rmsNormEps eps x - rmsNormEps eps y
      = (Real.sqrt (d : ℝ) / Real.sqrt a) • (x - y)
        + (Real.sqrt (d : ℝ) / Real.sqrt a - Real.sqrt (d : ℝ) / Real.sqrt b) • y := by
    rw [rmsNormEps, rmsNormEps, ← ha, ← hb]
    module
  calc ‖rmsNormEps eps x - rmsNormEps eps y‖
      ≤ ‖(Real.sqrt (d : ℝ) / Real.sqrt a) • (x - y)‖
        + ‖(Real.sqrt (d : ℝ) / Real.sqrt a - Real.sqrt (d : ℝ) / Real.sqrt b) • y‖ := by
        rw [hdecomp]; exact norm_add_le _ _
    _ = Real.sqrt (d : ℝ) / Real.sqrt a * ‖x - y‖
        + |Real.sqrt (d : ℝ) / Real.sqrt a - Real.sqrt (d : ℝ) / Real.sqrt b| * ‖y‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
          abs_of_nonneg (by positivity : (0 : ℝ) ≤ Real.sqrt (d : ℝ) / Real.sqrt a)]
    _ ≤ Real.sqrt (d : ℝ) / Real.sqrt a * ‖x - y‖ +
        Real.sqrt (d : ℝ) / Real.sqrt a * ‖x - y‖ := add_le_add le_rfl hcoef
    _ = 2 * Real.sqrt (d : ℝ) / Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) * ‖x - y‖ := by
        rw [ha]; ring


/-- M1: clipped L2 retains the recorded denominator across clipping boundaries. -/
theorem l2_secant {d : ℕ} (eps : ℝ) (heps : 0 < eps) (x y : EucSpace d) :
    ‖normL2 eps x - normL2 eps y‖ ≤ 2 / max ‖x‖ eps * ‖x - y‖ := by
  have hx : (0 : ℝ) < max ‖x‖ eps := lt_max_of_lt_right heps
  have hy : (0 : ℝ) < max ‖y‖ eps := lt_max_of_lt_right heps
  have hnorm : |max ‖y‖ eps - max ‖x‖ eps| ≤ ‖x - y‖ := by
    refine (abs_max_sub_max_le_abs _ _ _).trans ?_
    rw [← norm_neg (x - y), neg_sub]
    exact abs_norm_sub_norm_le y x
  have hcoef : |1 / max ‖x‖ eps - 1 / max ‖y‖ eps| * ‖y‖ ≤ 1 / max ‖x‖ eps * ‖x - y‖ := by
    have hdiff : 1 / max ‖x‖ eps - 1 / max ‖y‖ eps
        = (max ‖y‖ eps - max ‖x‖ eps) / (max ‖x‖ eps * max ‖y‖ eps) := by
      field_simp
    have hsplit : |max ‖y‖ eps - max ‖x‖ eps| / (max ‖x‖ eps * max ‖y‖ eps) * ‖y‖
        = |max ‖y‖ eps - max ‖x‖ eps| * (1 / max ‖x‖ eps) * (‖y‖ / max ‖y‖ eps) := by
      field_simp
    have hlast : ‖y‖ / max ‖y‖ eps ≤ 1 := by
      rw [div_le_one hy]; exact le_max_left _ _
    rw [hdiff, abs_div, abs_of_pos (mul_pos hx hy), hsplit]
    calc |max ‖y‖ eps - max ‖x‖ eps| * (1 / max ‖x‖ eps) * (‖y‖ / max ‖y‖ eps)
        ≤ ‖x - y‖ * (1 / max ‖x‖ eps) * 1 :=
          mul_le_mul
            (mul_le_mul_of_nonneg_right hnorm (one_div_nonneg.mpr hx.le))
            hlast (div_nonneg (norm_nonneg _) hy.le)
            (mul_nonneg (norm_nonneg _) (one_div_nonneg.mpr hx.le))
      _ = 1 / max ‖x‖ eps * ‖x - y‖ := by ring
  have hdecomp : normL2 eps x - normL2 eps y
      = (1 / max ‖x‖ eps) • (x - y)
        + (1 / max ‖x‖ eps - 1 / max ‖y‖ eps) • y := by
    rw [normL2, normL2]; module
  calc ‖normL2 eps x - normL2 eps y‖
      ≤ ‖(1 / max ‖x‖ eps) • (x - y)‖
        + ‖(1 / max ‖x‖ eps - 1 / max ‖y‖ eps) • y‖ := by
        rw [hdecomp]; exact norm_add_le _ _
    _ = 1 / max ‖x‖ eps * ‖x - y‖
        + |1 / max ‖x‖ eps - 1 / max ‖y‖ eps| * ‖y‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
          abs_of_nonneg (one_div_nonneg.mpr hx.le)]
    _ ≤ 1 / max ‖x‖ eps * ‖x - y‖ + 1 / max ‖x‖ eps * ‖x - y‖ := by
        gcongr
    _ = 2 / max ‖x‖ eps * ‖x - y‖ := by ring


example := rms_secant (d := 4) (1 / 8388608) (by norm_num) (by decide)
  (EuclideanSpace.single 0 1) 0
example := l2_secant (d := 4) (1 / 1000000000000) (by norm_num)
  0 (EuclideanSpace.single 0 1)

end Transformer.SequenceModel.Attention.Adaptive
