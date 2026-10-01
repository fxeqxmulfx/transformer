/-
# Coordinate smoothness does not imply the paper's joint descent bound

Formalization and correction of arXiv:2602.15322v1, Section 5,
assumption:layerwise_smooth, prop:magma_descent, and Appendix A.2,
sequential_descent. A two-block convex polynomial suffices to refute
the unsupported passage from one-block updates to simultaneous updates.
The partial gradients below are actual derivatives of the objective.
-/

import Transformer.Magma.Section2_MaskLaw
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul

noncomputable section

namespace Transformer.Magma

/-- The source's smoothness assumption specialized to two scalar blocks,
with actual partial derivatives. Source: arXiv:2602.15322v1, Section 5,
assumption:layerwise_smooth. -/
def CoordinateSmooth (loss : ℝ → ℝ → ℝ) (Lx Ly : ℝ) : Prop :=
  (∀ x y u, loss (x + u) y ≤ loss x y +
    u * deriv (fun z => loss z y) x + Lx / 2 * u ^ 2) ∧
  (∀ x y u, loss x (y + u) ≤ loss x y +
    u * deriv (fun z => loss x z) y + Ly / 2 * u ^ 2)

/-- Joint diagonal smoothness for two blocks. This is a strictly stronger
condition than the one printed in the source. Source: correction to
arXiv:2602.15322v1, Appendix A.2, sequential_descent. -/
def JointSmooth (loss : ℝ → ℝ → ℝ) (Lx Ly : ℝ) : Prop :=
  ∀ x y u v, loss (x + u) (y + v) ≤ loss x y +
    u * deriv (fun z => loss z y) x + v * deriv (fun z => loss x z) y +
      (Lx * u ^ 2 + Ly * v ^ 2) / 2

/-- A smooth convex nonconstant objective with positive mixed curvature.
Source: counterexample to arXiv:2602.15322v1, Appendix A.2,
sequential_descent. -/
def coupledQuadratic (x y : ℝ) : ℝ := (x + y) ^ 2 / 2

/-- Actual first-block gradient of the witness. Source:
arXiv:2602.15322v1, Section 5, assumption:layerwise_smooth, counterexample. -/
theorem coupledQuadratic_deriv_x (x y : ℝ) :
    deriv (fun z => coupledQuadratic z y) x = x + y := by
  have h : HasDerivAt (fun z => coupledQuadratic z y) (x + y) x := by
    convert (((hasDerivAt_id x).add_const y).pow 2).div_const 2 using 1 <;>
      simp [coupledQuadratic]
  exact h.deriv

/-- Actual second-block gradient of the witness. Source:
arXiv:2602.15322v1, Section 5, assumption:layerwise_smooth, counterexample. -/
theorem coupledQuadratic_deriv_y (x y : ℝ) :
    deriv (fun z => coupledQuadratic x z) y = x + y := by
  have h : HasDerivAt (fun z => coupledQuadratic x z) (x + y) y := by
    convert (((hasDerivAt_id y).const_add x).pow 2).div_const 2 using 1 <;>
      simp [coupledQuadratic]
  exact h.deriv

/-- Every coordinate obeys the source's condition with L=1, and the
objective is lower bounded by zero. Source: arXiv:2602.15322v1,
Section 5, assumptions; Appendix A.2 counterexample. -/
theorem coupledQuadratic_coordinateSmooth :
    CoordinateSmooth coupledQuadratic 1 1 ∧ ∀ x y, 0 ≤ coupledQuadratic x y := by
  constructor
  · constructor
    · intro x y u
      rw [coupledQuadratic_deriv_x]
      unfold coupledQuadratic
      nlinarith
    · intro x y u
      rw [coupledQuadratic_deriv_y]
      unfold coupledQuadratic
      nlinarith
  · intro x y
    exact div_nonneg (sq_nonneg _) (by norm_num)

/-- The simultaneous bound with the same constants is false, even for
this convex quadratic. Refuted claim: Appendix A.2, sequential_descent,
follows by applying assumption:layerwise_smooth sequentially.
Source: arXiv:2602.15322v1, Appendix A.2. -/
theorem coordinateSmooth_not_jointSmooth :
    ¬ JointSmooth coupledQuadratic 1 1 := by
  intro h
  have hbad := h 0 0 1 1
  rw [coupledQuadratic_deriv_x, coupledQuadratic_deriv_y] at hbad
  norm_num [coupledQuadratic] at hbad

/-- A fixed-size correction works: the global curvature of this witness
is bounded by diagonal constants 2. It is not valid to reuse 1.
Source: correction to arXiv:2602.15322v1, Appendix A.2. -/
theorem coupledQuadratic_jointSmooth_two : JointSmooth coupledQuadratic 2 2 := by
  intro x y u v
  rw [coupledQuadratic_deriv_x, coupledQuadratic_deriv_y]
  unfold coupledQuadratic
  nlinarith [sq_nonneg (u - v)]

/-- The manuscript's one-step descent estimate also fails on its stated
coordinate-smooth domain. This deterministic case has unbiased exact
gradients, p=1, and s=1/2 in both blocks. At (1,1), eta=1 gives (0,0),
while eq:one_step_descent_magma would upper-bound the loss by -1.
Source: arXiv:2602.15322v1, Section 5, prop:magma_descent, Appendix A.2. -/
theorem paper_descent_counterexample :
    let gx := deriv (fun z => coupledQuadratic z 1) 1
    let gy := deriv (fun z => coupledQuadratic 1 z) 1
    coupledQuadratic (1 - (1 / 2) * gx) (1 - (1 / 2) * gy) >
      coupledQuadratic 1 1 - ((1 / 2) * gx ^ 2 + (1 / 2) * gy ^ 2) +
        ((1 / 2 * gx) ^ 2 + (1 / 2 * gy) ^ 2) / 2 := by
  dsimp only
  rw [coupledQuadratic_deriv_x, coupledQuadratic_deriv_y]
  norm_num [coupledQuadratic]

end Transformer.Magma
