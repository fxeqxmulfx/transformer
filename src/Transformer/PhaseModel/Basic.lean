/-
# Phase model — the energy and its derivatives

The phase energy is nonnegative and vanishes exactly where every relation
holds (`phaseEnergy_eq_zero_iff`); a translation `v` with `⟪m_r, v⟫ ∈ 2πℤ`
for every row leaves it invariant (`phaseEnergy_add`).  It is smooth
(`contDiff_phaseEnergy`), with gradient `Σ_r sin ⟪m_r, ψ⟫ m_r`
(`gradient_phaseEnergy`) and Hessian `h ↦ Σ_r cos ⟪m_r, ψ⟫ ⟪m_r, h⟫ m_r`
(`hasFDerivAt_gradient_phaseEnergy`); the gradient is Lipschitz with
constant `Σ_r ‖m_r‖²` (`lipschitzWith_gradient_phaseEnergy`).  At a zero of
the energy the Hessian is `h ↦ Σ_r ⟪m_r, h⟫ m_r`, injective when no nonzero
vector is orthogonal to every row
(`injective_fderiv_gradient_of_phaseEnergy_eq_zero`).
-/

import Transformer.PhaseModel.Defs
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

open Real

namespace Transformer.PhaseModel

variable {E R : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [Fintype R] (m : R → E)

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- The phase energy is nonnegative. -/
theorem phaseEnergy_nonneg (ψ : E) : 0 ≤ phaseEnergy m ψ :=
  Finset.sum_nonneg fun _ _ => sub_nonneg.2 (cos_le_one _)

/-- **The phase energy vanishes exactly where every relation holds**:
`⟪m_r, ψ⟫ ∈ 2πℤ` for every row. -/
theorem phaseEnergy_eq_zero_iff (ψ : E) :
    phaseEnergy m ψ = 0 ↔ ∀ r, ∃ k : ℤ, ⟪m r, ψ⟫ = k * (2 * π) := by
  rw [phaseEnergy, Finset.sum_eq_zero_iff_of_nonneg fun _ _ => sub_nonneg.2 (cos_le_one _)]
  simp only [Finset.mem_univ, true_implies, sub_eq_zero, eq_comm (a := (1 : ℝ)), cos_eq_one_iff]
  exact forall_congr' fun r => exists_congr fun k => eq_comm

/-- **Translations keeping every relation leave the phase energy
invariant**: if `⟪m_r, v⟫ ∈ 2πℤ` for every row, then `V(ψ + v) = V(ψ)`. -/
theorem phaseEnergy_add {v : E} (hv : ∀ r, ∃ k : ℤ, ⟪m r, v⟫ = k * (2 * π)) (ψ : E) :
    phaseEnergy m (ψ + v) = phaseEnergy m ψ := by
  refine Finset.sum_congr rfl fun r _ => ?_
  obtain ⟨k, hk⟩ := hv r
  rw [inner_add_right, hk, cos_add_int_mul_two_pi]

/-- The phase energy is smooth. -/
theorem contDiff_phaseEnergy {k : WithTop ℕ∞} : ContDiff ℝ k (phaseEnergy m) :=
  ContDiff.sum fun _ _ => contDiff_const.sub (ContDiff.inner ℝ contDiff_const contDiff_id).cos

variable [CompleteSpace E]

/-- The gradient of the phase energy at `ψ` is `Σ_r sin ⟪m_r, ψ⟫ m_r`. -/
theorem hasGradientAt_phaseEnergy (ψ : E) :
    HasGradientAt (phaseEnergy m) (∑ r, sin ⟪m r, ψ⟫ • m r) ψ := by
  have h := HasFDerivAt.fun_sum (u := Finset.univ) fun r _ =>
    (((innerSL ℝ (m r)).hasFDerivAt (x := ψ)).cos).const_sub 1
  simp only [innerSL_apply_apply] at h
  rw [hasGradientAt_iff_hasFDerivAt]
  refine h.congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  simp only [FunLike.coe_sum, Finset.sum_apply, neg_apply, smul_apply, innerSL_apply_apply,
    smul_eq_mul, neg_mul, neg_neg, InnerProductSpace.toDual_apply_apply, sum_inner,
    real_inner_smul_left]

/-- **The gradient of the phase energy**: `∇V(ψ) = Σ_r sin ⟪m_r, ψ⟫ m_r`. -/
theorem gradient_phaseEnergy :
    gradient (phaseEnergy m) = fun ψ => ∑ r, sin ⟪m r, ψ⟫ • m r :=
  funext fun ψ => (hasGradientAt_phaseEnergy m ψ).gradient

/-- **The Hessian of the phase energy**: the derivative of the gradient at
`ψ` is `h ↦ Σ_r cos ⟪m_r, ψ⟫ ⟪m_r, h⟫ m_r`. -/
theorem hasFDerivAt_gradient_phaseEnergy (ψ : E) :
    HasFDerivAt (gradient (phaseEnergy m))
      (∑ r, (cos ⟪m r, ψ⟫ • innerSL ℝ (m r)).smulRight (m r)) ψ := by
  rw [gradient_phaseEnergy]
  exact HasFDerivAt.fun_sum fun r _ => ((innerSL ℝ (m r)).hasFDerivAt.sin).smul_const (m r)

/-- **The gradient of the phase energy is Lipschitz** with constant
`Σ_r ‖m_r‖²`, as `sin` is `1`-Lipschitz. -/
theorem lipschitzWith_gradient_phaseEnergy :
    LipschitzWith (∑ r, ‖m r‖₊ ^ 2) (gradient (phaseEnergy m)) := by
  refine LipschitzWith.of_dist_le_mul fun ψ φ => ?_
  rw [gradient_phaseEnergy, dist_eq_norm, dist_eq_norm, ← Finset.sum_sub_distrib,
    NNReal.coe_sum, Finset.sum_mul]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun r _ => ?_)
  rw [← sub_smul, norm_smul, Real.norm_eq_abs, NNReal.coe_pow, coe_nnnorm]
  calc |sin ⟪m r, ψ⟫ - sin ⟪m r, φ⟫| * ‖m r‖ ≤ |⟪m r, ψ⟫ - ⟪m r, φ⟫| * ‖m r‖ :=
        mul_le_mul_of_nonneg_right (abs_sin_sub_sin_le _ _) (norm_nonneg _)
    _ = |⟪m r, ψ - φ⟫| * ‖m r‖ := by rw [inner_sub_right]
    _ ≤ ‖m r‖ * ‖ψ - φ‖ * ‖m r‖ :=
        mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
    _ = ‖m r‖ ^ 2 * ‖ψ - φ‖ := by ring

/-- **The Hessian of the phase energy is injective where the cosines of all
relations share a strict sign** `s` and no nonzero vector is orthogonal to
every row: `s ⟪H h, h⟫ = Σ_r s cos ⟪m_r, z⟫ ⟪m_r, h⟫²` vanishes only when `h`
is orthogonal to every row. -/
theorem injective_fderiv_gradient_phaseEnergy {z : E} {s : ℝ}
    (hs : ∀ r, 0 < s * cos ⟪m r, z⟫) (hspan : ∀ h, (∀ r, ⟪m r, h⟫ = 0) → h = 0) :
    Function.Injective (fderiv ℝ (gradient (phaseEnergy m)) z) := by
  rw [(hasFDerivAt_gradient_phaseEnergy m z).fderiv]
  refine (injective_iff_map_eq_zero _).2 fun h hh => hspan h fun r => ?_
  have h1 : ∑ r, s * cos ⟪m r, z⟫ * ⟪m r, h⟫ ^ 2 = 0 := by
    have h2 := congrArg (fun y => s * ⟪y, h⟫) hh
    simp only [FunLike.coe_sum, Finset.sum_apply, ContinuousLinearMap.smulRight_apply,
      smul_apply, innerSL_apply_apply, smul_eq_mul, sum_inner, real_inner_smul_left,
      inner_zero_left, mul_zero, Finset.mul_sum] at h2
    rw [← h2]
    exact Finset.sum_congr rfl fun r _ => by ring
  have h3 := (Finset.sum_eq_zero_iff_of_nonneg fun r _ =>
    mul_nonneg (hs r).le (sq_nonneg _)).1 h1 r (Finset.mem_univ r)
  exact pow_eq_zero_iff two_ne_zero |>.1 ((mul_eq_zero.1 h3).resolve_left (hs r).ne')

/-- The gradient of the phase energy vanishes at its zeros. -/
theorem gradient_eq_zero_of_phaseEnergy_eq_zero {z : E} (hz : phaseEnergy m z = 0) :
    gradient (phaseEnergy m) z = 0 := by
  rw [gradient_phaseEnergy]
  refine Finset.sum_eq_zero fun r _ => ?_
  obtain ⟨k, hk⟩ := (phaseEnergy_eq_zero_iff m z).1 hz r
  rw [sin_eq_zero_iff_cos_eq.2 (Or.inl (by rw [hk, cos_int_mul_two_pi])), zero_smul]

/-- **The zeros of the phase energy are nondegenerate** when no nonzero
vector is orthogonal to every row: there every cosine is `1`. -/
theorem injective_fderiv_gradient_of_phaseEnergy_eq_zero {z : E} (hz : phaseEnergy m z = 0)
    (hspan : ∀ h, (∀ r, ⟪m r, h⟫ = 0) → h = 0) :
    Function.Injective (fderiv ℝ (gradient (phaseEnergy m)) z) := by
  refine injective_fderiv_gradient_phaseEnergy m (s := 1) (fun r => ?_) hspan
  obtain ⟨k, hk⟩ := (phaseEnergy_eq_zero_iff m z).1 hz r
  rw [hk, cos_int_mul_two_pi, one_mul]
  exact one_pos

/-- The hypotheses of the theorems above hold for the single row `1` on the
line, where `V(ψ) = 1 - cos ψ`: the translation `2π` keeps the relation, `0`
is a zero, the cosine at `π` has the strict sign `-1`, and only `0` is
orthogonal to `1`. -/
example : let m := fun _ : Unit => (1 : ℝ)
    (∀ r, ∃ k : ℤ, ⟪m r, 2 * π⟫ = k * (2 * π)) ∧ phaseEnergy m 0 = 0 ∧
      (∀ r, 0 < -1 * cos ⟪m r, π⟫) ∧ ∀ h, (∀ r, ⟪m r, h⟫ = 0) → h = 0 := by
  intro m
  refine ⟨fun _ => ⟨1, by simp [m]⟩, by simp [m, phaseEnergy], fun _ => by simp [m],
    fun h hh => ?_⟩
  simpa [m] using hh ()

end Transformer.PhaseModel
