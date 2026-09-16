/-
# RMSNorm (without trainable scale)

Formalization of the `RMSNorm` class from `reference/model.py`:

```python
class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        ...
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
```

Mathematically:

  `RMSNorm(x) = x / sqrt(mean(x²) + eps) = x · sqrt(d) / sqrt(||x||² + d·eps)`,

so without `eps`, `||RMSNorm(x)||₂ = sqrt(d)` exactly.

We define both the `eps`-perturbed and the ideal version, and prove the
post-norm L2 norm.
-/

import Transformer.Basic
import Transformer.GPTMini.Config
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable {d : ℕ}

/-- Ideal RMS normalization on `EucSpace d`, no `eps`:

  `RMSNorm(x) = x · sqrt(d) / ||x||₂`,

defined as `0` when `x = 0` (numerical convention; in practice `eps > 0`
avoids the `x = 0` case). -/
noncomputable def rmsNorm (x : EucSpace d) : EucSpace d :=
  if ‖x‖ = 0 then 0
  else (Real.sqrt (d : ℝ) / ‖x‖) • x

/-- `eps`-perturbed RMS normalization, as in `F.rms_norm(x, (d,), eps)`:

  `RMSNorm_eps(x) = x · sqrt(d) / sqrt(||x||² + d · eps)`. -/
noncomputable def rmsNormEps (eps : ℝ) (x : EucSpace d) : EucSpace d :=
  (Real.sqrt (d : ℝ) /
    Real.sqrt (‖x‖^2 + (d : ℝ) * eps)) • x

/-- The ideal `rmsNorm` puts non-zero vectors on the sphere of radius `√d`. -/
theorem rmsNorm_norm_eq_sqrt_d (x : EucSpace d) (hx : x ≠ 0) :
    ‖rmsNorm x‖ = Real.sqrt (d : ℝ) := by
  unfold rmsNorm
  have h_norm_ne : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  rw [ite_eq_right h_norm_ne]
  rw [norm_smul, Real.norm_eq_abs]
  rw [abs_of_nonneg (div_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))]
  exact div_mul_cancel₀ _ h_norm_ne

/-- The `eps`-perturbed version yields norm at most `√d`. -/
theorem rmsNormEps_norm_le (eps : ℝ) (heps : 0 < eps) (hd : 0 < d) (x : EucSpace d) :
    ‖rmsNormEps eps x‖ ≤ Real.sqrt (d : ℝ) := by
  have hd' : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have hpos : 0 < ‖x‖ ^ 2 + (d : ℝ) * eps := by positivity
  have hs : 0 < Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) := Real.sqrt_pos.mpr hpos
  have hle : ‖x‖ ≤ Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) :=
    (Real.le_sqrt (norm_nonneg x) hpos.le).mpr (by nlinarith)
  rw [rmsNormEps, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (by positivity : (0 : ℝ) ≤ Real.sqrt (d : ℝ) /
      Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps)),
    div_mul_eq_mul_div, div_le_iff₀ hs]
  exact mul_le_mul_of_nonneg_left hle (Real.sqrt_nonneg _)

/-- **`rmsNormEps` is globally Lipschitz, with constant `2 / √eps`.**

Writing `c(x) = √d / √(‖x‖² + d·eps)`, the difference splits as
`c(x)·(x - y) + (c(x) - c(y))·y`.  The first term is controlled by
`c(x) ≤ √d / √(d·eps) = 1/√eps`; for the second,
`|√b - √a| = |b - a| / (√a + √b)` with `|b - a| ≤ ‖x - y‖(‖x‖ + ‖y‖)`, and
`(‖x‖ + ‖y‖)‖y‖ ≤ (√a + √b)√b`, leaving `√d/√a ≤ 1/√eps` again.

This is the factor that every Lipschitz estimate for a Pre-LN sub-layer must
carry: `eps` small makes the normalization steep.  Source:
`reference/model.py` (`RMSNorm.forward`, `F.rms_norm` with `eps`). -/
theorem rmsNormEps_lipschitz (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (x y : EucSpace d) :
    ‖rmsNormEps eps x - rmsNormEps eps y‖ ≤ 2 / Real.sqrt eps * ‖x - y‖ := by
  have hd' : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have hdeps : (0 : ℝ) < (d : ℝ) * eps := mul_pos hd' heps
  set a := ‖x‖ ^ 2 + (d : ℝ) * eps with ha
  set b := ‖y‖ ^ 2 + (d : ℝ) * eps with hb
  have hapos : 0 < a := add_pos_of_nonneg_of_pos (sq_nonneg _) hdeps
  have hbpos : 0 < b := add_pos_of_nonneg_of_pos (sq_nonneg _) hdeps
  have hsa : 0 < Real.sqrt a := Real.sqrt_pos.mpr hapos
  have hsb : 0 < Real.sqrt b := Real.sqrt_pos.mpr hbpos
  have hse : 0 < Real.sqrt eps := Real.sqrt_pos.mpr heps
  have hsd : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr hd'
  -- `√(d·eps) ≤ √a`, the source of the `1/√eps` in both terms.
  have hlow : Real.sqrt (d : ℝ) * Real.sqrt eps ≤ Real.sqrt a := by
    rw [← Real.sqrt_mul hd'.le]
    exact Real.sqrt_le_sqrt (by rw [ha]; nlinarith [sq_nonneg ‖x‖])
  have hca : Real.sqrt (d : ℝ) / Real.sqrt a ≤ 1 / Real.sqrt eps := by
    rw [div_le_div_iff₀ hsa hse]
    linarith [hlow]
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
      ≤ 1 / Real.sqrt eps * ‖x - y‖ := by
    rw [one_div, inv_mul_eq_div, hrw, div_mul_eq_mul_div,
      div_le_div_iff₀ (mul_pos hsa hsb) hse]
    calc Real.sqrt (d : ℝ) * |Real.sqrt b - Real.sqrt a| * ‖y‖ * Real.sqrt eps
        = (Real.sqrt (d : ℝ) * Real.sqrt eps)
            * (|Real.sqrt b - Real.sqrt a| * ‖y‖) := by ring
      _ ≤ Real.sqrt a * (‖x - y‖ * Real.sqrt b) :=
          mul_le_mul hlow h3 (by positivity) hsa.le
      _ = ‖x - y‖ * (Real.sqrt a * Real.sqrt b) := by ring
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
    _ ≤ 1 / Real.sqrt eps * ‖x - y‖ + 1 / Real.sqrt eps * ‖x - y‖ := by
        gcongr
    _ = 2 / Real.sqrt eps * ‖x - y‖ := by ring

/-- `rmsNorm` is positively 0-homogeneous: `RMSNorm(c·x) = RMSNorm(x)` for `c > 0`. -/
theorem rmsNorm_pos_homog (x : EucSpace d) (c : ℝ) (hc : 0 < c) :
    rmsNorm (c • x) = rmsNorm x := by
  unfold rmsNorm
  have h_smul_norm : ‖c • x‖ = c * ‖x‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos hc]
  rw [h_smul_norm]
  by_cases hx : ‖x‖ = 0
  · simp [hx]
  · have hcx : c * ‖x‖ ≠ 0 := mul_ne_zero (ne_of_gt hc) hx
    rw [ite_eq_right hcx, ite_eq_right hx, smul_smul]
    congr 1
    field_simp

/-- The map `rmsNorm` is continuous on `{x : ‖x‖ ≠ 0}`. -/
theorem continuous_rmsNorm :
    ContinuousOn (rmsNorm : EucSpace d → EucSpace d) {x | x ≠ 0} := by
  refine ContinuousOn.congr (f := fun x : EucSpace d => (Real.sqrt (d : ℝ) / ‖x‖) • x) ?_ ?_
  · apply ContinuousOn.fun_smul
    · exact continuousOn_const.div continuous_norm.continuousOn
        fun x hx => norm_ne_zero_iff.mpr hx
    · exact continuousOn_id
  · intro x hx
    simp [rmsNorm, norm_ne_zero_iff.mpr hx]

end GPTMini
end Transformer
