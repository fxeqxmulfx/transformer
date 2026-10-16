/-
# Parameter-dependent bounds for the complete original field

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`,
and the parameter estimate in `thm: targets.atoms`.
The original attention average has norm at most one. Coordinatewise
ReLU is a contraction in the original Euclidean norm. These facts give
speed and spatial estimates retaining all five original parameter blocks.

In particular the attention contribution is linear in the value norm,
whereas the perceptron contribution contains the product of its output
norm and the input and bias norms. The resulting constants are uniform
over every original spherical probability law and every unit query.
The exact Dirac identity retains the complete field and score matrix.
-/

import Transformer.Interpolation.Section2_AttentionLipschitz
import Transformer.Interpolation.Section4_NeuralSpeed

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Coordinatewise original ReLU does not increase Euclidean distances.
Source auxiliary: arXiv:2411.04551v3, §1, the perceptron in `eq: vf`. -/
theorem norm_coordinate_relu_sub_le (u v : EucSpace d) :
    ‖(EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (u i) 0) -
      (EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (v i) 0)‖ ≤ ‖u - v‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  apply Finset.sum_le_sum
  intro i _
  change (max (u i) 0 - max (v i) 0) ^ 2 ≤ (u i - v i) ^ 2
  simpa only [sq_abs] using
    (sq_le_sq₀ (abs_nonneg _) (abs_nonneg _)).mpr (abs_max_sub_max_le_abs (u i) (v i) 0)

/-- The complete original field splits into its actual attention and
neural contributions. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem fullVF_eq_proj_attention_add_neural (θ : TimeParams d)
    (ν : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d θ ν t x = proj d x ((θ t).V (attentionAverage (θ t).B ν x)) +
      neuralVF d (fun s => (θ s).W) (fun s => (θ s).U) (fun s => (θ s).b) t x := by
  simp only [fullVF, attentionAverage, Perspective.partitionMu, one_mul, neuralVF,
    proj, inner_add_right, add_smul]
  module

/-- The exact original speed bound retains the value norm and the
quadratic perceptron dependence. Source: arXiv:2411.04551v3, §1,
`eq: vf`, and §1.1, `thm: targets.atoms`. -/
theorem fullVF_norm_le (θ : TimeParams d) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) (hx : ‖x‖ = 1) :
    ‖fullVF d θ ν t x‖ ≤ ‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖) := by
  have hav : ‖((θ t).V) (attentionAverage (θ t).B ν x)‖ ≤ ‖(θ t).V‖ := by
    apply (((θ t).V).le_opNorm _).trans
    simpa only [mul_one] using mul_le_mul_of_nonneg_left
      (norm_attentionAverage_le_one (θ t).B ν x) (norm_nonneg ((θ t).V))
  have ha : ‖proj d x ((θ t).V (attentionAverage (θ t).B ν x))‖ ≤ ‖(θ t).V‖ :=
    (norm_proj_le hx _).trans hav
  have hu : ‖(θ t).U x + (θ t).b‖ ≤ ‖(θ t).U‖ + ‖(θ t).b‖ :=
    (norm_add_le _ _).trans (add_le_add (by simpa only [hx, mul_one] using (θ t).U.le_opNorm x) le_rfl)
  have hn : ‖neuralVF d (fun s => (θ s).W) (fun s => (θ s).U) (fun s => (θ s).b) t x‖ ≤
      ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖) := by
    have hr : ‖(EuclideanSpace.equiv (Fin d) ℝ).symm
        (fun i => max (((θ t).U x + (θ t).b) i) 0)‖ ≤ ‖(θ t).U‖ + ‖(θ t).b‖ :=
      (norm_coordinate_relu_le ((θ t).U x + (θ t).b)).trans hu
    change ‖proj d x ((θ t).W ((EuclideanSpace.equiv (Fin d) ℝ).symm
      (fun i => max (((θ t).U x + (θ t).b) i) 0)))‖ ≤ _
    exact (norm_proj_le hx _).trans ((((θ t).W).le_opNorm _).trans
      (mul_le_mul_of_nonneg_left hr (norm_nonneg ((θ t).W))))
  rw [fullVF_eq_proj_attention_add_neural]
  exact (norm_add_le _ _).trans (add_le_add ha hn)

/-- A common bound K on the original parameter blocks gives speed at
most K+2K², uniformly over laws. Source auxiliary: arXiv:2411.04551v3,
§1.1, the norm estimate of `thm: targets.atoms`. -/
theorem fullVF_norm_le_of_component_bounds (θ : TimeParams d) (ν : Perspective.ProbSphere d)
    (t K : ℝ) (x : EucSpace d) (hx : ‖x‖ = 1)
    (hV : ‖(θ t).V‖ ≤ K) (hW : ‖(θ t).W‖ ≤ K)
    (hU : ‖(θ t).U‖ ≤ K) (hb : ‖(θ t).b‖ ≤ K) :
    ‖fullVF d θ ν t x‖ ≤ K + 2 * K ^ 2 := by
  have hK : 0 ≤ K := (norm_nonneg _).trans hV
  apply (fullVF_norm_le θ ν t x hx).trans
  have hp := mul_le_mul hW (add_le_add hU hb) (by positivity) hK
  nlinarith

/-- The original attention at a Dirac law is its atom, even at a
different ambient query. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem attentionAverage_dirac (B : ParamMatrix d) (p : SSphere d) (x : EucSpace d) :
    attentionAverage B (Perspective.diracProb d p) x = (p : EucSpace d) := by
  change (∫ y : SSphere d, exp (1 * inner ℝ (B x) (y : EucSpace d)) ∂Measure.dirac p)⁻¹ •
    (∫ y : SSphere d, exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d) ∂Measure.dirac p) = _
  rw [integral_dirac, integral_dirac, one_mul, smul_smul,
    inv_mul_cancel₀ (exp_pos _).ne', one_smul]

/-- The complete original Dirac field retains the value and perceptron
blocks and loses its score dependence by exact normalization.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: vf`. -/
theorem fullVF_dirac_eq (θ : TimeParams d) (t : ℝ) (p : SSphere d) (x : EucSpace d) :
    fullVF d θ (Perspective.diracProb d p) t x = proj d x ((θ t).V (p : EucSpace d)) +
      neuralVF d (fun s => (θ s).W) (fun s => (θ s).U) (fun s => (θ s).b) t x := by
  rw [fullVF_eq_proj_attention_add_neural, attentionAverage_dirac]

/-- The unprojected perceptron is Lipschitz with the product of the
original input and output operator norms. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: vf`, and §4, the neural field. -/
theorem norm_perceptron_argument_sub_le (W U : ParamMatrix d) (b x y : EucSpace d) :
    ‖W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max ((U x + b) i) 0)) -
      W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max ((U y + b) i) 0))‖ ≤
      (‖W‖ * ‖U‖) * ‖x - y‖ := by
  have he : (U x + b) - (U y + b) = U (x - y) := by
    rw [map_sub]
    abel
  have hrelu := norm_coordinate_relu_sub_le (U x + b) (U y + b)
  rw [he] at hrelu
  rw [← map_sub]
  calc
    _ ≤ ‖W‖ * ‖(EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max ((U x + b) i) 0) -
        (EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max ((U y + b) i) 0)‖ := W.le_opNorm _
    _ ≤ ‖W‖ * ‖U (x - y)‖ := mul_le_mul_of_nonneg_left hrelu (norm_nonneg _)
    _ ≤ ‖W‖ * (‖U‖ * ‖x - y‖) := mul_le_mul_of_nonneg_left (U.le_opNorm _) (norm_nonneg _)
    _ = _ := by ring

/-- The complete original field is tangent at every unit query,
independently of its law and all five parameter blocks.
Source: arXiv:2411.04551v3, §1, layer normalization in `eq: vf`. -/
theorem inner_fullVF_eq_zero (θ : TimeParams d) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) (hx : ‖x‖ = 1) : inner ℝ x (fullVF d θ ν t x) = 0 := by
  apply inner_proj_eq_zero hx

/-- Nonzero values, scores, perceptron input and output matrices, and
bias satisfy all unit-query and component-bound hypotheses jointly. -/
example : ∃ (θ : TimeParams 2) (ν : Perspective.ProbSphere 2) (p : SSphere 2)
    (x y : EucSpace 2) (K : ℝ),
    ‖x‖ = 1 ∧ ‖y‖ = 1 ∧ x ≠ y ∧
    ‖(θ 0).V‖ ≤ K ∧ ‖(θ 0).W‖ ≤ K ∧ ‖(θ 0).U‖ ≤ K ∧ ‖(θ 0).b‖ ≤ K ∧
    (θ 0).B (p : EucSpace 2) = p ∧ ν = Perspective.diracProb 2 p := by
  let p : SSphere 2 := basePoint 1
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let θ : TimeParams 2 := fun _ => ⟨I, I, I, I, (p : EucSpace 2)⟩
  have hp : ‖(p : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hne : (p : EucSpace 2) ≠ -(p : EucSpace 2) := by
    intro he
    have hz : (2 : ℝ) • (p : EucSpace 2) = 0 := by
      rw [two_smul]
      nth_rewrite 2 [he]
      exact add_neg_cancel _
    have hp0 := (smul_eq_zero.mp hz).resolve_left (by norm_num : (2 : ℝ) ≠ 0)
    rw [hp0, norm_zero] at hp
    exact zero_ne_one hp
  refine ⟨θ, Perspective.diracProb 2 p, p, p, -(p : EucSpace 2), 1,
    hp, by simpa only [norm_neg] using hp, hne, ?_, ?_, ?_, hp.le, rfl, rfl⟩ <;>
    simp [θ, I]

end Transformer.Interpolation
