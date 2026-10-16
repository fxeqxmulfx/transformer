/-
# Two actual ReLU neurons with a prescribed spherical equilibrium

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: neural.ode.sphere`,
and Appendix, the construction toward `lem: two.balls`.

Two distinct hidden coordinates implement the sum of two actual ridge
fields with attention switched off. The matrices remain constant and
use the original width `d`; the coordinate distinction is explicit.
The field is continuous, tangent on the sphere, and zero where both
activations are inactive.

For two inward cap activations, the second output can be chosen so that
the combined output at the prescribed target is radial. The target is
therefore an actual equilibrium even when the first output points toward
a different cap center. Global attraction is a subsequent dynamical claim.
-/

import Transformer.Interpolation.Section3_RidgeWellPosed

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The sum of the two original projected ridge activations.
Source: arXiv:2411.04551v3, §1, the perceptron in `eq: vf`. -/
noncomputable def twoRidgeField (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x : EucSpace d) : EucSpace d :=
  max (b - inner ℝ x u) 0 • proj d x p + max (c - inner ℝ x v) 0 • proj d x q

/-- Two explicit hidden coordinates of the original constant perceptron.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
noncomputable def twoRidgeParams (i j : Fin d) (p u : EucSpace d) (b : ℝ)
    (q v : EucSpace d) (c : ℝ) : Params d :=
  { V := 0
    B := 0
    W := (reluRidgeParams i p u b).W + (reluRidgeParams j q v c).W
    U := (reluRidgeParams i p u b).U + (reluRidgeParams j q v c).U
    b := (reluRidgeParams i p u b).b + (reluRidgeParams j q v c).b }

/-- The constructed original controls switch off both attention matrices.
Source: arXiv:2411.04551v3, §1, the perceptron specialization of `eq: vf`. -/
theorem twoRidgeParams_attention_zero (i j : Fin d) (p u : EucSpace d) (b : ℝ)
    (q v : EucSpace d) (c : ℝ) :
    (twoRidgeParams i j p u b q v c).V = 0 ∧ (twoRidgeParams i j p u b q v c).B = 0 :=
  ⟨rfl, rfl⟩

/-- Distinct hidden coordinates realize exactly the actual two-ridge
field, for every original probability law. Source: arXiv:2411.04551v3,
§1, `eq: vf`, with attention switched off. -/
theorem fullVF_twoRidgeParams (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ρ : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (fun _ => twoRidgeParams i j p u b q v c) ρ t x = twoRidgeField p u b q v c x := by
  have hrelu : (EuclideanSpace.equiv (Fin d) ℝ).symm (fun k => max
      ((EuclideanSpace.equiv (Fin d) ℝ) ((twoRidgeParams i j p u b q v c).U x +
        (twoRidgeParams i j p u b q v c).b) k) 0) =
      max (b - inner ℝ x u) 0 • EuclideanSpace.single i 1 +
        max (c - inner ℝ x v) 0 • EuclideanSpace.single j 1 := by
    ext k
    by_cases hki : k = i
    · subst k
      simp [twoRidgeParams, reluRidgeParams, hij, real_inner_comm x u]
      congr 1
      ring
    by_cases hkj : k = j
    · subst k
      simp [twoRidgeParams, reluRidgeParams, Ne.symm hij, real_inner_comm x v]
      congr 1
      ring
    simp [twoRidgeParams, reluRidgeParams, hki, hkj]
  have hV : (twoRidgeParams i j p u b q v c).V = 0 := rfl
  simp only [fullVF, hV, zero_apply, zero_add]
  change proj d x ((twoRidgeParams i j p u b q v c).W _) = _
  rw [hrelu]
  simp only [twoRidgeParams, add_apply, map_add, map_smul]
  simp only [reluRidgeParams, ContinuousLinearMap.smulRight_apply]
  simp [hij, Ne.symm hij, twoRidgeField, proj, inner_add_right, inner_smul_right]
  module

/-- The original two-neuron field is continuous across both ReLU kinks.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_twoRidgeField (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) :
    Continuous (twoRidgeField p u b q v c) := by
  unfold twoRidgeField proj
  fun_prop

/-- The actual two-neuron field is tangent to the original sphere.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem inner_twoRidgeField_self (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x : SSphere d) :
    inner ℝ (x : EucSpace d) (twoRidgeField p u b q v c x) = 0 := by
  simp only [twoRidgeField, inner_add_right, real_inner_smul_right,
    inner_proj_eq_zero (mem_sphere_zero_iff_norm.mp x.2), mul_zero, add_zero]

/-- Both inactive activations make the actual original field vanish.
Source: arXiv:2411.04551v3, Appendix, the fixed region in `lem: two.balls`. -/
theorem twoRidgeField_inactive (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x : EucSpace d) (hu : b - inner ℝ x u ≤ 0) (hv : c - inner ℝ x v ≤ 0) :
    twoRidgeField p u b q v c x = 0 := by
  rw [twoRidgeField, max_eq_right hu, max_eq_right hv, zero_smul, zero_smul, add_zero]


/-- An explicit second output makes the desired target an equilibrium
of the actual constant two-neuron field. This auxiliary construction toward
arXiv:2411.04551v3, Appendix, `lem: two.balls`, retains both original activations;
it does not assume convergence to that equilibrium. -/
theorem twoRidgeField_target (z w ω : SSphere d) (c h ℓ : ℝ)
    (hα : 0 ≤ inner ℝ (ω : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) :
    twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c)
      ((inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)⁻¹ •
        (ℓ • (ω : EucSpace d) - (inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) • (z : EucSpace d)))
      (-(w : EucSpace d)) (-h) ω = 0 := by
  let α := inner ℝ (ω : EucSpace d) (z : EucSpace d) - c
  let β := inner ℝ (ω : EucSpace d) (w : EucSpace d) - h
  have hα' : -c - inner ℝ (ω : EucSpace d) (-(z : EucSpace d)) = α := by
    dsimp only [α]; rw [inner_neg_right]; ring
  have hβ' : -h - inner ℝ (ω : EucSpace d) (-(w : EucSpace d)) = β := by
    dsimp only [β]; rw [inner_neg_right]; ring
  have he : α • (z : EucSpace d) + β • (β⁻¹ • (ℓ • (ω : EucSpace d) - α • (z : EucSpace d))) =
      ℓ • (ω : EucSpace d) := by
    rw [smul_smul, mul_inv_cancel₀ hβ.ne', one_smul]
    module
  have hp := proj_smul_self (mem_sphere_zero_iff_norm.mp ω.2) ℓ
  change twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c)
    (β⁻¹ • (ℓ • (ω : EucSpace d) - α • (z : EucSpace d))) (-(w : EucSpace d)) (-h) ω = 0
  unfold twoRidgeField
  rw [hα', hβ', max_eq_left hα, max_eq_left hβ.le]
  calc
    _ = proj d ω (α • (z : EucSpace d) + β • (β⁻¹ • (ℓ • (ω : EucSpace d) - α • (z : EucSpace d)))) := by
      simp only [proj, inner_add_right, real_inner_smul_right]
      module
    _ = 0 := by rw [he]; exact hp

/-- Two distinct actual coordinates, different unit cap and target axes,
and a common inactive point jointly realize the matrix, inactive-region
and prescribed-equilibrium hypotheses. -/
example : ∃ z w ω y : SSphere 2,
    (0 : Fin 2) ≠ 1 ∧
    0 ≤ inner ℝ (ω : EucSpace 2) (z : EucSpace 2) - (-1 / 2) ∧
    0 < inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - 0 ∧
    -(-1 / 2) - inner ℝ (y : EucSpace 2) (-(z : EucSpace 2)) ≤ 0 ∧
    -0 - inner ℝ (y : EucSpace 2) (-(w : EucSpace 2)) ≤ 0 := by
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let w : SSphere 2 := basePoint 1
  refine ⟨z, w, w, antipode 2 z, ?_⟩
  norm_num [z, w, antipode, basePoint, EuclideanSpace.inner_single_left, EuclideanSpace.inner_single_right]

end Transformer.Interpolation
