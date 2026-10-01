/-
# Uniform derivative control under actual smooth composition

arXiv:2506.12543v1, Section 4.3, Theorem 1's finite-horizon weak comparison.
The chain rule gives a sharp derivative Lipschitz recurrence, without a
factor independent of the time step that would explode along an Euler grid.
-/

import Transformer.BatchSize.Section4_DriftDerivativeBounds

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Bounded first derivatives propagate under actual smooth
composition, Section 4.3, Theorem 1. -/
theorem composed_fderiv_norm_le {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    (f : E → F) (g : F → G) (hf : ContDiff ℝ 2 f) (hg : ContDiff ℝ 2 g)
    (K L : NNReal) (hK : LipschitzWith K f) (hL : ∀ y, ‖fderiv ℝ g y‖ ≤ L) (x : E) :
    ‖fderiv ℝ (g ∘ f) x‖ ≤ L * K := by
  rw [fderiv_comp x (hg.differentiable (by norm_num) (f x)) (hf.differentiable (by norm_num) x)]
  exact (ContinuousLinearMap.opNorm_comp_le _ _).trans
    (mul_le_mul (hL _) (norm_fderiv_le_of_lipschitz ℝ hK) (norm_nonneg _) L.coe_nonneg)

/-- The actual derivative Lipschitz constant under composition
satisfies the sharp chain-rule recurrence L*H + Q*K^2,
Section 4.3, Theorem 1. This controls second derivatives without
introducing a spurious fixed factor at every Euler step. -/
theorem composed_fderiv_lipschitz {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    (f : E → F) (g : F → G) (hf : ContDiff ℝ 2 f) (hg : ContDiff ℝ 2 g)
    (K H L Q : NNReal) (hK : LipschitzWith K f) (hH : LipschitzWith H (fderiv ℝ f))
    (hL : ∀ y, ‖fderiv ℝ g y‖ ≤ L) (hQ : LipschitzWith Q (fderiv ℝ g)) :
    LipschitzWith (L * H + Q * K ^ 2) (fderiv ℝ (g ∘ f)) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  rw [dist_eq_norm, dist_eq_norm,
    fderiv_comp x (hg.differentiable (by norm_num) (f x)) (hf.differentiable (by norm_num) x),
    fderiv_comp y (hg.differentiable (by norm_num) (f y)) (hf.differentiable (by norm_num) y)]
  have heq : (fderiv ℝ g (f x)).comp (fderiv ℝ f x) -
      (fderiv ℝ g (f y)).comp (fderiv ℝ f y) =
      (fderiv ℝ g (f x)).comp (fderiv ℝ f x - fderiv ℝ f y) +
        (fderiv ℝ g (f x) - fderiv ℝ g (f y)).comp (fderiv ℝ f y) := by
    ext v
    simp only [ContinuousLinearMap.comp_apply, sub_apply, add_apply, map_sub]
    abel
  rw [heq]
  have hfg : ‖fderiv ℝ g (f x) - fderiv ℝ g (f y)‖ ≤ (Q : ℝ) * (K * ‖x - y‖) :=
    (hQ.norm_sub_le (f x) (f y)).trans (mul_le_mul_of_nonneg_left (hK.norm_sub_le x y) Q.coe_nonneg)
  calc
    _ ≤ ‖(fderiv ℝ g (f x)).comp (fderiv ℝ f x - fderiv ℝ f y)‖ +
        ‖(fderiv ℝ g (f x) - fderiv ℝ g (f y)).comp (fderiv ℝ f y)‖ := norm_add_le _ _
    _ ≤ ‖fderiv ℝ g (f x)‖ * ‖fderiv ℝ f x - fderiv ℝ f y‖ +
        ‖fderiv ℝ g (f x) - fderiv ℝ g (f y)‖ * ‖fderiv ℝ f y‖ :=
      add_le_add (ContinuousLinearMap.opNorm_comp_le _ _) (ContinuousLinearMap.opNorm_comp_le _ _)
    _ ≤ (L : ℝ) * (H * ‖x - y‖) + (Q * (K * ‖x - y‖)) * K :=
      add_le_add (mul_le_mul (hL _) (hH.norm_sub_le x y) (norm_nonneg _) L.coe_nonneg)
        (mul_le_mul hfg (norm_fderiv_le_of_lipschitz ℝ hK) (norm_nonneg _) (by positivity))
    _ = _ := by simp only [NNReal.coe_add, NNReal.coe_mul, NNReal.coe_pow]; ring

/-- A genuine Lipschitz bound on the derivative is a global Hessian
bound in the actual iterated Frechet norm, Section 4.3, Theorem 1. -/
theorem iteratedFDeriv_two_le_of_fderiv_lipschitz {E F : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : E → F) (H : NNReal) (hH : LipschitzWith H (fderiv ℝ f)) (x : E) :
    ‖iteratedFDeriv ℝ 2 f x‖ ≤ H := by
  rw [← norm_iteratedFDeriv_fderiv, norm_iteratedFDeriv_one]
  exact norm_fderiv_le_of_lipschitz ℝ hH

/-- Joint nonvacuity of all composition derivative hypotheses,
Section 4.3: actual identity maps with unit first derivative and
constant, hence Lipschitz, derivative maps. -/
example : ContDiff ℝ 2 (id : ℝ → ℝ) ∧ LipschitzWith 1 (id : ℝ → ℝ) ∧
    (∀ x : ℝ, ‖fderiv ℝ (id : ℝ → ℝ) x‖ ≤ (1 : NNReal)) ∧
    LipschitzWith 0 (fderiv ℝ (id : ℝ → ℝ)) := by
  refine ⟨contDiff_id, LipschitzWith.id, (fun x => norm_fderiv_le_of_lipschitz ℝ LipschitzWith.id), ?_⟩
  have heq : fderiv ℝ (id : ℝ → ℝ) = fun _ => ContinuousLinearMap.id ℝ ℝ := by
    funext x
    exact fderiv_id
  rw [heq]
  exact LipschitzWith.const _

end Transformer.BatchSize
