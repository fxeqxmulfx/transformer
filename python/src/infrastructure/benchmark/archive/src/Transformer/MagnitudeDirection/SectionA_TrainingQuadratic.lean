/-
# Nonconstant quadratic witnesses for MD training

User-requested convergence extension of arXiv:2606.25971v2, §3.1 and
Appendix A, Algorithm 2. Translated quadratics have genuine derivatives,
smoothness and strong convexity. A minimum opposite the initial scalar
direction exposes a limitation of normalized positive-gain MD steps.
-/

import Transformer.MagnitudeDirection.SectionA_TrainingMinimum

open scoped InnerProductSpace

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- A genuine quadratic loss with a prescribed minimizer, used to check
training convergence. Source: extension of arXiv:2606.25971v2, §3.1 and
Appendix A, Algorithm 2. -/
def trainingQuadratic (star x : E) : ℝ := energy (x - star)

/-- The derivative is computed from the loss, not supplied as an
independent vector field. Source: arXiv:2606.25971v2, Appendix A,
training counterexample and convergence witness. -/
theorem trainingQuadratic_hasGradientAt (star x : E) :
    HasGradientAt (trainingQuadratic star) (x - star) x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  have h := (energy_hasGradientAt (x - star)).hasFDerivAt.comp x
    ((hasFDerivAt_id x).sub_const star)
  convert h using 1 <;> ext y <;> rfl

/-- Actual objective gradient at every current weight,
arXiv:2606.25971v2, Appendix A, training extension. -/
theorem trainingQuadratic_gradient (star : E) :
    gradient (trainingQuadratic star) = fun x => x - star :=
  gradient_eq (trainingQuadratic_hasGradientAt star)

/-- Translated quadratics have smoothness and strong-convexity constant
one, and are lower bounded by zero. Source: arXiv:2606.25971v2, Appendix A,
training counterexample and convergence witness. -/
theorem trainingQuadratic_models (star : E) :
    SmoothObjective (trainingQuadratic star) 1 ∧
      StrongLowerModel (trainingQuadratic star) 1 ∧
      (∀ x, 0 ≤ trainingQuadratic star x) := by
  have hdiff (x y : E) : (y - star) - (x - star) = y - x := by abel
  refine ⟨⟨fun x => (trainingQuadratic_hasGradientAt star x).differentiableAt, ?_⟩, ?_, ?_⟩
  · intro x y
    have h := (energy_models (E := E)).1.2 (x - star) (y - star)
    rw [energy_gradient, id_eq, hdiff] at h
    simpa only [trainingQuadratic, trainingQuadratic_gradient] using h
  · intro x y
    have h := (energy_models (E := E)).2 (x - star) (y - star)
    rw [energy_gradient, id_eq, hdiff] at h
    simpa only [trainingQuadratic, trainingQuadratic_gradient] using h
  · intro x
    exact div_nonneg (sq_nonneg _) (by norm_num)

/-- In a one-entry matrix the genuine Frobenius norm is the absolute
entry value. Source: arXiv:2606.25971v2, §2, Frobenius convention;
Appendix A, training counterexample. -/
theorem single_frobeniusNorm (W : Matrix (Fin 1) (Fin 1) ℝ) :
    frobeniusNorm W = |W 0 0| := by
  have h := frobeniusNorm_sq W
  simp only [Fin.sum_univ_one] at h
  nlinarith [abs_nonneg (W 0 0), sq_abs (W 0 0),
    show 0 ≤ frobeniusNorm W from norm_nonneg _]

/-- A smooth strongly convex matrix loss minimized at the negative unit
weight. Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
def oppositeQuadratic : MatrixSpace 1 1 → ℝ :=
  trainingQuadratic (-(fromMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ)))

/-- The counterexample loss is actually smooth, strongly convex and
lower bounded, rather than carrying those facts as hypotheses.
Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
theorem oppositeQuadratic_models : SmoothObjective oppositeQuadratic 1 ∧
    StrongLowerModel oppositeQuadratic 1 ∧ (∀ x, 0 ≤ oppositeQuadratic x) :=
  trainingQuadratic_models _

/-- Every current gradient entry is `weight + 1` for the counterexample.
Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
theorem oppositeQuadratic_gradient (x : MatrixSpace 1 1) :
    toMatrix (gradient oppositeQuadratic x) 0 0 = toMatrix x 0 0 + 1 := by
  rw [show oppositeQuadratic = trainingQuadratic
    (-(fromMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ))) from rfl, trainingQuadratic_gradient]
  simp [toMatrix, fromMatrix]

/-- Any positive scalar fused weight has a genuine opposite-quadratic
gradient norm greater than or equal to one, regardless of its gains.
Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
theorem oppositeQuadratic_gradient_lower_bound (W : Matrix (Fin 1) (Fin 1) ℝ)
    (hW : 0 < W 0 0) : 1 ≤ ‖gradient oppositeQuadratic (fromMatrix W)‖ := by
  have hn (x : MatrixSpace 1 1) : ‖x‖ = frobeniusNorm (toMatrix x) := by
    change ‖x‖ = ‖fromMatrix (toMatrix x)‖
    rw [fromMatrix_toMatrix]
  rw [hn, single_frobeniusNorm, oppositeQuadratic_gradient, toMatrix_fromMatrix,
    abs_of_pos (by linarith : 0 < W 0 0 + 1)]
  linarith

/-- Positive scalar weights satisfy the gradient-bound premise,
arXiv:2606.25971v2, Appendix A, training counterexample. -/
example : (0 : ℝ) < (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 := by norm_num

end Transformer.MagnitudeDirection
