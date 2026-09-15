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
  -- Calculation: ‖rmsNormEps eps x‖ = (√d / √(‖x‖² + d·eps)) · ‖x‖.
  -- We bound: ‖x‖ ≤ √(‖x‖² + d·eps), so the ratio is ≤ 1, so result ≤ √d.
  -- Detailed Mathlib-API manipulation deferred.
  sorry

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
  sorry

end GPTMini
end Transformer
