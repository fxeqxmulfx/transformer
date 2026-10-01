/-
# The actual deterministic Euler comparison map

arXiv:2506.12543v1, Section 4.3, Theorem 1's finite-horizon weak comparison.
The map uses precisely the optimizer's mean drift. Its derivative constants
approach one and zero respectively as the step size tends to zero.
-/

import Transformer.BatchSize.Section4_ComposedDerivativeLipschitz

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A genuine deterministic Euler step for the mean drift in
Section 4.3 (2)--(3), used as a weak comparison process. -/
def deterministicEulerMap {d : ℕ} (b : EucSpace d → EucSpace d) (η : NNReal)
    (x : EucSpace d) : EucSpace d := x + (η : ℝ) • b x

/-- The true Euler comparison step preserves C2 regularity,
Section 4.3, Theorem 1. -/
theorem deterministicEulerMap_contDiff {d : ℕ} (b : EucSpace d → EucSpace d) (η : NNReal)
    (hb : ContDiff ℝ 2 b) : ContDiff ℝ 2 (deterministicEulerMap b η) :=
  contDiff_id.add (hb.const_smul (η : ℝ))

/-- The actual deterministic Euler step has Lipschitz constant
1 + eta*K, Section 4.3, Theorem 1. -/
theorem deterministicEulerMap_lipschitz {d : ℕ} (b : EucSpace d → EucSpace d) (η K : NNReal)
    (hK : LipschitzWith K b) : LipschitzWith (1 + η * K) (deterministicEulerMap b η) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  rw [dist_eq_norm, dist_eq_norm]
  change ‖(x + (η : ℝ) • b x) - (y + (η : ℝ) • b y)‖ ≤ _
  rw [add_sub_add_comm, ← smul_sub]
  calc
    _ ≤ ‖x - y‖ + ‖(η : ℝ) • (b x - b y)‖ := norm_add_le _ _
    _ = ‖x - y‖ + (η : ℝ) * ‖b x - b y‖ := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg η.coe_nonneg]
    _ ≤ ‖x - y‖ + (η : ℝ) * (K * ‖x - y‖) :=
      add_le_add le_rfl (mul_le_mul_of_nonneg_left (hK.norm_sub_le x y) η.coe_nonneg)
    _ = _ := by simp only [NNReal.coe_add, NNReal.coe_one, NNReal.coe_mul]; ring

/-- The derivative of the actual Euler step is Id + eta*Db,
Section 4.3, Theorem 1's comparison argument. -/
theorem deterministicEulerMap_fderiv {d : ℕ} (b : EucSpace d → EucSpace d) (η : NNReal)
    (hb : ContDiff ℝ 2 b) (x : EucSpace d) :
    fderiv ℝ (deterministicEulerMap b η) x = ContinuousLinearMap.id ℝ (EucSpace d) +
      (η : ℝ) • fderiv ℝ b x := by
  have h : HasFDerivAt (deterministicEulerMap b η)
      (ContinuousLinearMap.id ℝ (EucSpace d) + (η : ℝ) • fderiv ℝ b x) x :=
    (hasFDerivAt_id x).add ((hb.differentiable (by norm_num) x).hasFDerivAt.const_smul (η : ℝ))
  exact h.fderiv

/-- The true derivative Lipschitz constant of the deterministic
Euler step is eta*H, Section 4.3, Theorem 1. Thus repeated composition
does not insert a fixed error independent of the step size. -/
theorem deterministicEulerMap_fderiv_lipschitz {d : ℕ}
    (b : EucSpace d → EucSpace d) (η H : NNReal) (hb : ContDiff ℝ 2 b)
    (hH : LipschitzWith H (fderiv ℝ b)) :
    LipschitzWith (η * H) (fderiv ℝ (deterministicEulerMap b η)) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  rw [dist_eq_norm, dist_eq_norm, deterministicEulerMap_fderiv b η hb x,
    deterministicEulerMap_fderiv b η hb y, add_sub_add_left_eq_sub]
  have heq : (η : ℝ) • fderiv ℝ b x - (η : ℝ) • fderiv ℝ b y =
      (η : ℝ) • (fderiv ℝ b x - fderiv ℝ b y) := by
    apply ContinuousLinearMap.ext
    intro v
    simp only [sub_apply, smul_apply]
    exact (smul_sub (η : ℝ) (fderiv ℝ b x v) (fderiv ℝ b y v)).symm
  rw [heq, norm_smul, Real.norm_eq_abs, abs_of_nonneg η.coe_nonneg]
  simpa only [NNReal.coe_mul, mul_assoc] using
    mul_le_mul_of_nonneg_left (hH.norm_sub_le x y) η.coe_nonneg

/-- Joint nonvacuity of the smooth Euler-map hypotheses,
Section 4.3: the actual identity drift and its constant derivative. -/
example : ContDiff ℝ 2 (id : EucSpace 1 → EucSpace 1) ∧
    LipschitzWith 1 (id : EucSpace 1 → EucSpace 1) ∧
    LipschitzWith 0 (fderiv ℝ (id : EucSpace 1 → EucSpace 1)) := by
  refine ⟨contDiff_id, LipschitzWith.id, ?_⟩
  have heq : fderiv ℝ (id : EucSpace 1 → EucSpace 1) =
      fun _ => ContinuousLinearMap.id ℝ (EucSpace 1) := by funext x; exact fderiv_id
  rw [heq]
  exact LipschitzWith.const _

end Transformer.BatchSize
