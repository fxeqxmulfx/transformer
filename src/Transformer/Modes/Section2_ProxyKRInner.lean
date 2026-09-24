/-
# The number of modes of a Gaussian KDE — the inner proxy Kac–Rice integral

For `eq:int-phi-final`, the real Gaussian moment of `lem:phi-t` must be
identified with the nonnegative Lebesgue integral used by the Kac–Rice
formula.  Positivity of `α_t` suffices for integrability and this conversion.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:int-phi-final`.
-/

import Transformer.Modes.Section2_PhiTDelta
import Transformer.Modes.Section2_ProxyKRScale

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-- The first moment of the Gaussian proxy is integrable over `y > 0` when
its completed-square curvature is positive.

Source: arXiv:2412.09080v3, `lem:phi-t`. -/
theorem integrableOn_mul_krPhi {n : ℕ} {β t : ℝ} (hα : 0 < phiAlpha β t) :
    IntegrableOn (fun y : ℝ => y * krPhi n β t 0 y) (Set.Ioi 0) := by
  have hF : sigmaFst β t ≠ 0 := by
    intro h
    simp [phiAlpha, h] at hα
  have hD : sigmaDet β t ≠ 0 := by
    intro h
    simp [phiAlpha, h] at hα
  have heq : (fun y : ℝ => y * krPhi n β t 0 y) =
      (fun y => ((2 * Real.pi)⁻¹ * Real.exp (-phiA n β t / 2)) *
        (y * Real.exp (-(phiAlpha β t / 2) * (y - phiDelta n β t) ^ 2))) := by
    funext y
    change y * ((2 * Real.pi)⁻¹ * Real.exp (-krQuad n β t 0 y / 2)) = _
    rw [krQuad_zero_eq hF hD y,
      show -(phiA n β t + phiAlpha β t * (y - phiDelta n β t) ^ 2) / 2 =
        -phiA n β t / 2 + -(phiAlpha β t / 2) * (y - phiDelta n β t) ^ 2 by ring,
      Real.exp_add]
    ring
  rw [heq]
  exact (integrableOn_mul_exp_neg_shift_sq hα).const_mul _

/-- The `lintegral` in the proxy Kac–Rice expression is the `ofReal` of the
integral evaluated by `lem:phi-t`.

Source: arXiv:2412.09080v3, `eq:int-phi-final`. -/
theorem lintegral_krPhi_eq_ofReal_integral {n : ℕ} {β t : ℝ}
    (hα : 0 < phiAlpha β t) :
    (∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal (y * krPhi n β t 0 y)) =
      ENNReal.ofReal (∫ y in Set.Ioi (0 : ℝ), y * krPhi n β t 0 y) := by
  have hnonneg : 0 ≤ᵐ[volume.restrict (Set.Ioi (0 : ℝ))]
      (fun y : ℝ => y * krPhi n β t 0 y) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
    exact mul_nonneg hy.le (by unfold krPhi; positivity)
  exact (ofReal_integral_eq_lintegral_ofReal (integrableOn_mul_krPhi hα) hnonneg).symm

/-- Pull the positive covariance determinant out of the inner nonnegative
Kac–Rice integral.  The remaining integral is exactly the real moment of
`lem:phi-t`.

Source: arXiv:2412.09080v3, `eq:int-phi-final`. -/
theorem lintegral_det_krPhi_eq {n : ℕ} {β t : ℝ}
    (hα : 0 < phiAlpha β t) (hD : 0 < sigmaDet β t) :
    (∫⁻ y in Set.Ioi (0 : ℝ),
      ENNReal.ofReal (y * sigmaDet β t ^ (-(1 : ℝ) / 2) * krPhi n β t 0 y)) =
      ENNReal.ofReal (sigmaDet β t ^ (-(1 : ℝ) / 2)) *
        ENNReal.ofReal (∫ y in Set.Ioi (0 : ℝ), y * krPhi n β t 0 y) := by
  have hpow : 0 ≤ sigmaDet β t ^ (-(1 : ℝ) / 2) := Real.rpow_nonneg hD.le _
  have heq (y : ℝ) :
      ENNReal.ofReal (y * sigmaDet β t ^ (-(1 : ℝ) / 2) * krPhi n β t 0 y) =
        ENNReal.ofReal (sigmaDet β t ^ (-(1 : ℝ) / 2)) *
          ENNReal.ofReal (y * krPhi n β t 0 y) := by
    rw [show y * sigmaDet β t ^ (-(1 : ℝ) / 2) * krPhi n β t 0 y =
      sigmaDet β t ^ (-(1 : ℝ) / 2) * (y * krPhi n β t 0 y) by ring,
      ENNReal.ofReal_mul hpow]
  simp_rw [heq]
  rw [lintegral_const_mul' _ _ (ENNReal.ofReal_ne_top)]
  rw [lintegral_krPhi_eq_ofReal_integral hα]

/-- The curvature and determinant hypotheses are simultaneously satisfiable
along the `β = n` regime. -/
example : ∃ β t : ℝ, 0 < phiAlpha β t ∧ 0 < sigmaDet β t := by
  obtain ⟨C₁, C₂, hC₁, hC₂, hα⟩ :=
    phiAlpha_isTheta isRegime_succ isSlowGrowth_sqrt_log_log
  obtain ⟨k, hk⟩ :=
    (hα.and (eventually_sigmaDet_pos isRegime_succ isSlowGrowth_sqrt_log_log)).exists
  refine ⟨((k + 1 : ℕ) : ℝ), 0, ?_, hk.2 0 (zero_mem_intervalT _ _ _)⟩
  have h := (hk.1 0 (zero_mem_intervalT _ _ _)).1
  have hp : 0 < (((k + 1 : ℕ) : ℝ) ^ ((1 : ℝ) / 2) * Real.exp ((0 : ℝ) ^ 2 / 2)) := by
    positivity
  exact lt_of_lt_of_le (mul_pos hC₁ hp) h

end Modes
end Transformer
