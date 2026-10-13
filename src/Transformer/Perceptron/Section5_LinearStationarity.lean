/-
# The original quadratic transform forced by linear stationarity

ArXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
For the original linear activation, multiplying the original first
variation by the temperature gives the actual exponential weight plus
the temperature times the original half quadratic potential.

The actual derivative and projected field of this original function
are proved below. The latter is the temperature times the original
stationary field. It is analytic in the actual ambient space. The
nonempty interior of the original support has positive actual uniform
measure, so the proved spherical vector uniqueness makes it zero on
the entire original sphere. The original function is therefore constant
there, forcing its original exponential weight to be quadratic.

The source fixes positive temperature. This identity also holds at
every real temperature: the proof multiplies the original field and
never divides by the temperature. Density inversion still requires
the source's positive-temperature hypothesis in the following result.
-/

import Transformer.Perceptron.Section6_AmbientBranches
import Transformer.Perceptron.Section6_SphericalFields
import Transformer.Perceptron.Section5_SignedTransform
import Transformer.Perceptron.Section5_QuadraticPolynomials
import Transformer.Perceptron.Section5_QuadraticDensity

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The original first variation for the linear activation, multiplied
by the original temperature. Source: arXiv:2601.21366v2, §5.3,
`rem:genericity-vacuity`, with the original half-potential normalization. -/
noncomputable def linearAttentionPotential {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) : ℝ :=
  attentionWeightMap (β • LinearMap.id) μ x + β * branchHalfPotential ω a Finset.univ x

/-- The actual ambient vector in that original derivative.
Source: arXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
Its derivative and projected-field identities are proved below. -/
noncomputable def linearAttentionVector {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) : EucSpace d :=
  ambientAttentionGradMap (β • LinearMap.id) μ x + β • branchVector ω a Finset.univ x

/-- The displayed original vector represents the actual derivative of
the original scaled first variation. Source: arXiv:2601.21366v2,
§5.3, `rem:genericity-vacuity`. Both original derivatives and every
original temperature factor are retained in the covector equality. -/
theorem hasFDerivAt_linearAttentionPotential {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    HasFDerivAt (linearAttentionPotential β ω a μ) (innerSL ℝ (linearAttentionVector β ω a μ x)) x := by
  have hW := hasFDerivAt_attentionWeightMap (β • LinearMap.id) μ x
  rw [attentionCovectorMap_eq_innerSL_ambientGradMap] at hW
  have h := hW.add ((hasFDerivAt_branchHalfPotential ω a Finset.univ x).const_mul β)
  apply h.congr_fderiv
  ext v
  change inner (𝕜 := ℝ) (ambientAttentionGradMap (β • LinearMap.id) μ x) v +
    β * inner (𝕜 := ℝ) (branchVector ω a Finset.univ x) v =
      inner (𝕜 := ℝ) (ambientAttentionGradMap (β • LinearMap.id) μ x +
        β • branchVector ω a Finset.univ x) v
  rw [inner_add_left, real_inner_smul_left]

/-- The actual projected linear-activation field is analytic in the
original ambient space. Source: arXiv:2601.21366v2, §5.3,
the local-to-global quadratic argument in `rem:genericity-vacuity`.
The actual projection is treated as its original bilinear formula. -/
theorem analyticOnNhd_projected_linearAttentionVector {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    AnalyticOnNhd ℝ (fun x => proj d x (linearAttentionVector β ω a μ x)) univ := by
  intro x hx
  have hD : AnalyticAt ℝ (linearAttentionVector β ω a μ) x :=
    (analyticOnNhd_ambientAttentionGradMap (β • LinearMap.id) μ x hx).fun_add
      (analyticAt_const.fun_smul (analyticOnNhd_branchVector ω a Finset.univ x hx))
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  have hi : AnalyticAt ℝ (fun z => inner (𝕜 := ℝ) z (linearAttentionVector β ω a μ z)) x :=
    (L.analyticAt_bilinear (x, linearAttentionVector β ω a μ x)).comp
      (f := fun z => (z, linearAttentionVector β ω a μ z)) (analyticAt_id.prod hD)
  exact hD.fun_sub (hi.fun_smul analyticAt_id)

/-- The actual projected derivative is the original temperature times
the original linear stationary field. Source: arXiv:2601.21366v2,
§5.3, `rem:genericity-vacuity`, using `eq: steady.state`.
This identity is proved for the original integrals and actual neuron drift. -/
theorem projected_linearAttentionVector_eq_energyGrad {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    proj d x (linearAttentionVector β ω a μ x) = β • energyGrad β (fun s => s) ω a μ x := by
  have hp (y v : EucSpace d) : proj d x (y + β • v) = proj d x y + β • proj d x v := by
    simp only [proj, inner_add_right, real_inner_smul_right, add_smul, mul_smul, smul_sub]
    abel
  rw [linearAttentionVector, hp, ← attentionGradMap_eq_proj_ambientGradMap,
    attentionGradMap_smul_id]
  change β • attentionGrad β μ x + β • drift (fun s => s) ω a x =
    β • (attentionGrad β μ x + drift (fun s => s) ω a x)
  rw [smul_add]

/-- Linear stationarity with actual support of nonempty interior forces
the original transform to be the original constant plus a finite sum of
actual squared neuron coordinates. Source: arXiv:2601.21366v2, §5.3,
`rem:genericity-vacuity`. The local field is made globally zero by the
proved analytic uniqueness; the original potential is then made constant
by its proved actual derivative, without assuming either conclusion. -/
theorem exists_const_add_squares_transform_of_linear_stationary {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationary β (fun s => s) ω a μ)
    (hint : (interior (μ : Measure (SSphere d)).support).Nonempty) :
    ∃ C : ℝ, ∀ x : SSphere d, attentionTransform β μ x =
      C + ∑ j, (-β / 2 * ω j) * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2 := by
  let F := fun x => proj d x (linearAttentionVector β ω a μ x)
  have hν := Metastability.isUniformOn_uniformLaw (d := d) (by omega)
  let ν := MeanField.uniformLaw d
  let _ : ν.IsOpenPosMeasure := isOpenPosMeasure_of_uniform ν hν
  have hpos : 0 < ν {x : SSphere d | F (x : EucSpace d) = 0} :=
    (isOpen_interior.measure_pos ν hint).trans_le (measure_mono (fun x hx => by
      change proj d (x : EucSpace d) (linearAttentionVector β ω a μ (x : EucSpace d)) = 0
      rw [projected_linearAttentionVector_eq_energyGrad, hμ x (interior_subset hx), smul_zero]))
  have hglobal := vector_eq_zero_on_sphere_of_positive_zero_measure hd F
    (analyticOnNhd_projected_linearAttentionVector β ω a μ) ν hν hpos
  have hconst := eq_on_sphere_of_projected_derivative_zero hd (linearAttentionPotential β ω a μ)
    (linearAttentionVector β ω a μ) (hasFDerivAt_linearAttentionPotential β ω a μ) fun z hz => by
      let x : SSphere d := ⟨z, mem_sphere_zero_iff_norm.mpr hz⟩
      exact hglobal x
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  refine ⟨linearAttentionPotential β ω a μ (x₀ : EucSpace d), fun x => ?_⟩
  have h := hconst x x₀
  have hW : attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) = attentionTransform β μ x := by
    simp only [attentionWeightMap, attentionTransform, LinearMap.smul_apply,
      LinearMap.id_apply, real_inner_smul_right]
  change attentionWeightMap (β • LinearMap.id) μ (x : EucSpace d) +
    β * branchHalfPotential ω a Finset.univ (x : EucSpace d) =
      linearAttentionPotential β ω a μ (x₀ : EucSpace d) at h
  rw [hW] at h
  have heq : (∑ j, (-β / 2 * ω j) * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2) =
      -(β * ((1 / 2 : ℝ) * ∑ j, ω j * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2)) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [heq]
  change attentionTransform β μ x + β * ((1 / 2 : ℝ) *
    ∑ j, ω j * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2) =
      linearAttentionPotential β ω a μ (x₀ : EucSpace d) at h
  linarith

/-- **Remark `rem:genericity-vacuity`:** the original linear stationary
probability with support of nonempty interior has an actual quadratic
density against every original uniform law, and is not finitely atomic.
Source: arXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
The source specifies `σ(s)=s`; its canonical primitive is `s²`.
The earlier independent primitive and derivative binders were unnecessary
for that claim and have been deleted. All stationary, dimensional,
temperature, probability and interior hypotheses and conclusions remain. -/
theorem linear_stationary_polyDensity (d : ℕ) (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationary β (fun s => s) ω a μ)
    (hint : (interior (μ : Measure (SSphere d)).support).Nonempty) :
    (∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
        ∃ g : SSphere d → ℝ, IsPolyOfDegreeLE 2 g ∧
          (μ : Measure (SSphere d)) = ν.withDensity fun y => ENNReal.ofReal (g y)) ∧
      ¬ IsFinitelyAtomic μ := by
  obtain ⟨C, hC⟩ := exists_const_add_squares_transform_of_linear_stationary hd β ω a μ hμ hint
  refine ⟨fun ν hν => ?_, ?_⟩
  · obtain ⟨g, hg, -, heq⟩ := exists_quadratic_density_of_const_add_squares_transform
      hd Finset.univ β hβ μ C (fun j => -β / 2 * ω j) a hC ν hν
    exact ⟨g, hg, heq⟩
  · exact not_isFinitelyAtomic_of_const_add_squares_transform
      hd Finset.univ β hβ μ C (fun j => -β / 2 * ω j) a hC

/-- All original positive-temperature, stationary and support-interior
hypotheses hold jointly for the genuine uniform circle probability.
Source: arXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
The original activation is linear and its actual support is the full sphere. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    IsStationary 1 (fun s => s) (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2) μ ∧
    (interior (μ : Measure (SSphere 2)).support).Nonempty := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, le_rfl, one_pos, isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) 0 μ hν,
    interior_support_nonempty_of_uniform μ hν⟩

end Transformer.Perceptron
