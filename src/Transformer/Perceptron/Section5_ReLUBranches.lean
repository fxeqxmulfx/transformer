/-
# The original unnormalized ReLU branch fields

ArXiv:2601.21366v2, §5.2 and §5.3, `eq:quad.pol.I` and
`eq:global-Id`. On an actual activation pattern the original ReLU
field equals the original linear field with inactive weights set to zero.
The actual masked weights below retain every active original coefficient.

The proved derivative of the original linear first variation therefore
applies to each actual quadratic branch. If its original projected field
vanishes on the whole sphere, the actual potential is constant there.
Its original exponential transform is an actual finite quadratic sum.
The proved original inverse and nonnegative polynomial-support theorem
then give full support of that same original probability measure.

The finitely many weak activation patterns cover every actual sphere
point, including all boundaries and zero neurons. A positive original
uniform measure of support must therefore give a positive measure to
one actual pattern intersection. No genericity or nonzero-neuron
assumption is needed in this original finite-cover step.
-/

import Transformer.Perceptron.Section5_LinearStationarity
import Transformer.Perceptron.Section5_PolynomialSupport

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- Original active weights and actual zero inactive weights on the
supplied original branch. Source: arXiv:2601.21366v2, §5.3,
`eq:quad.pol.I`. The actual field restriction is proved below. -/
def branchMaskedWeights {d : ℕ} (ω : Idx d → ℝ) (J : Finset (Idx d)) : Idx d → ℝ :=
  fun j => if j ∈ J then ω j else 0

/-- The original linear drift at the actual masked weights is the actual
projected original quadratic branch vector. Source: arXiv:2601.21366v2,
§5.3, `eq:quad.pol.I`. Every inactive original neuron has actual weight zero. -/
theorem drift_linear_masked_eq_projected_branch {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    drift (fun s => s) (branchMaskedWeights ω J) a x = proj d x (branchVector ω a J x) := by
  unfold drift branchVector branchMaskedWeights
  congr 1
  simp only [ite_mul, zero_mul, ite_smul, zero_smul]
  exact Fintype.sum_ite_mem J _

/-- At an actual original pattern, the projected original branch
derivative is the original temperature times the original ReLU field.
Source: arXiv:2601.21366v2, §5.3, `eq:suppmu.zero.g`.
The actual masked linear drift is identified with the original ReLU drift. -/
theorem projected_linearAttentionVector_eq_relu_at_pattern {d : ℕ} (β : ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (x : EucSpace d) (hx : IsActivationPattern a J x) :
    proj d x (linearAttentionVector β (branchMaskedWeights ω J) a μ x) =
      β • energyGrad β (fun s => max s 0) ω a μ x := by
  rw [projected_linearAttentionVector_eq_energyGrad]
  congr 1
  unfold energyGrad
  rw [drift_linear_masked_eq_projected_branch, drift_relu_eq_projected_branchVector ω a J x hx]

/-- A globally zero actual quadratic branch gives the actual finite
quadratic original transform. Source: arXiv:2601.21366v2, §5.3,
`eq:global-Id`. The constancy follows from the proved actual derivative;
no stationary or support-interior conclusion is an input. -/
theorem exists_const_add_squares_transform_of_zero_projected_branch {d : ℕ}
    (hd : 2 ≤ d) (β : ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d,
      proj d (x : EucSpace d) (linearAttentionVector β (branchMaskedWeights ω J) a μ (x : EucSpace d)) = 0) :
    ∃ C : ℝ, ∀ x : SSphere d, attentionTransform β μ x =
      C + ∑ j, (-β / 2 * branchMaskedWeights ω J j) * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2 := by
  let w := branchMaskedWeights ω J
  have hconst := eq_on_sphere_of_projected_derivative_zero hd (linearAttentionPotential β w a μ)
    (linearAttentionVector β w a μ) (hasFDerivAt_linearAttentionPotential β w a μ) fun z hz => by
      exact hzero ⟨z, mem_sphere_zero_iff_norm.mpr hz⟩
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  refine ⟨linearAttentionPotential β w a μ (x₀ : EucSpace d), fun x => ?_⟩
  have h := hconst x x₀
  change attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) +
    β * branchHalfPotential w a Finset.univ (x : EucSpace d) =
      linearAttentionPotential β w a μ (x₀ : EucSpace d) at h
  have hW : attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) = attentionTransform β μ x := by
    simp only [attentionWeightMap, attentionTransform, LinearMap.smul_apply,
      LinearMap.id_apply, real_inner_smul_right]
  rw [hW] at h
  have heq : (∑ j, (-β / 2 * w j) * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2) =
      -(β * ((1 / 2 : ℝ) * ∑ j, w j * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2)) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring
  change attentionTransform β μ x = linearAttentionPotential β w a μ (x₀ : EucSpace d) + _
  rw [heq]
  change attentionTransform β μ x + β * ((1 / 2 : ℝ) *
    ∑ j, w j * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2) =
      linearAttentionPotential β w a μ (x₀ : EucSpace d) at h
  linarith

/-- A globally zero original quadratic branch gives full original
support at positive temperature. Source: arXiv:2601.21366v2, §5.3,
the full-support inference following `eq:global-Id`.
The same original probability is identified by the proved kernel inverse. -/
theorem support_eq_univ_of_zero_projected_relu_branch {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d,
      proj d (x : EucSpace d) (linearAttentionVector β (branchMaskedWeights ω J) a μ (x : EucSpace d)) = 0) :
    (μ : Measure (SSphere d)).support = univ := by
  obtain ⟨C, hC⟩ := exists_const_add_squares_transform_of_zero_projected_branch hd β ω a μ J hzero
  exact support_eq_univ_of_const_add_squares_transform hd Finset.univ β hβ μ C
    (fun j => -β / 2 * branchMaskedWeights ω J j) a hC

/-- One actual activation-pattern intersection has positive original
measure whenever the original set does. Source: arXiv:2601.21366v2,
§5.3, the finite-cell selection before `eq:global-Id`.
Weak patterns cover all original boundary points as well as the open cells. -/
theorem exists_positive_measure_activationPattern {d : ℕ} (ν : Measure (SSphere d))
    (S : Set (SSphere d)) (hS : 0 < ν S) (a : Idx d → EucSpace d) :
    ∃ J : Finset (Idx d), 0 < ν (S ∩ {x : SSphere d | IsActivationPattern a J (x : EucSpace d)}) := by
  have hcover : (⋃ J : Finset (Idx d), S ∩ {x : SSphere d | IsActivationPattern a J (x : EucSpace d)}) = S := by
    ext x
    constructor
    · intro hx
      obtain ⟨J, hx⟩ := mem_iUnion.mp hx
      exact hx.1
    · intro hx
      obtain ⟨J, hJ⟩ := exists_activationPattern a (x : EucSpace d)
      exact mem_iUnion.mpr ⟨J, hx, hJ⟩
  apply exists_measure_pos_of_not_measure_iUnion_null
  rw [hcover]
  exact hS.ne'

/-- All branch, vanishing-field, positive-measure, temperature and
dimensional inputs hold jointly for the actual uniform circle law and
zero original neuron weights. Source: arXiv:2601.21366v2, §5.3,
`eq:quad.pol.I` and `eq:global-Id`. The zero-neuron boundary is included. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (basePoint 1 : EucSpace 2) ∧
    (∀ x : SSphere 2, proj 2 (x : EucSpace 2)
      (linearAttentionVector 1 (branchMaskedWeights (0 : Idx 2 → ℝ) ∅) 0 μ (x : EucSpace 2)) = 0) ∧
    0 < (μ : Measure (SSphere 2)) univ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  refine ⟨μ, le_rfl, one_pos, ?_, ?_, by simp⟩
  · exact ⟨fun j hj => by simp at hj, fun _ _ => by simp⟩
  · intro x
    have hx : x ∈ (μ : Measure (SSphere 2)).support := by rw [support_eq_univ_of_uniform μ hν]; exact mem_univ _
    rw [projected_linearAttentionVector_eq_energyGrad]
    have hw : branchMaskedWeights (0 : Idx 2 → ℝ) ∅ = 0 := by ext j; simp [branchMaskedWeights]
    rw [hw, isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) 0 μ hν x hx, smul_zero]

end Transformer.Perceptron
