/-
# Non-analytic original potentials obstruct a globally zero ReLU branch

ArXiv:2601.21366v2, §5.3, proof of `thm: any.d` (i),
following `eq:global-Id`. A globally zero original quadratic branch
forces full support of the original probability by the proved kernel
inverse. Original stationarity then holds at every actual sphere point.

The actual derivative of the original first variation consequently has
zero spherical projection everywhere. Its original scalar value is
constant on the actual connected sphere. Solving its original expression
for the original perceptron potential gives an ambient-analytic extension,
because the original exponential interaction weight is analytic.

Thus the source's actual non-analytic potential rules out an identically
zero branch. The primitive of twice the original activation is arbitrary,
with the actual derivative hypothesis retained. No primitive is replaced,
and no potential analyticity or full-support conclusion is an input in
the ReLU branch obstruction. Both full-support and obstruction hypotheses
have genuine joint examples from the original uniform and pinned laws.
The scalar full-support implication also applies at negative nonzero
temperature; the branch inverse retains the source's positive temperature.
-/

import Transformer.Perceptron.Section5_ReLUBranches
import Transformer.Perceptron.FirstVariation

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The original scalar partition integral is the same actual analytic
exponential weight. Source: arXiv:2601.21366v2, §5.3,
the analyticity step preceding `eq:global-Id`.
Its actual temperature, original measure and ambient point are retained. -/
theorem analyticOnNhd_partitionMu_original {d : ℕ} (β : ℝ) (μ : Perspective.ProbSphere d) :
    AnalyticOnNhd ℝ (Perspective.partitionMu d β μ) univ := by
  have heq : Perspective.partitionMu d β μ = attentionWeightMap (β • LinearMap.id) μ := by
    funext x
    simp only [Perspective.partitionMu, attentionWeightMap, LinearMap.smul_apply,
      LinearMap.id_apply, real_inner_smul_right]
  rw [heq]
  exact analyticOnNhd_attentionWeightMap (β • LinearMap.id) μ

/-- The original first variation is constant on the actual sphere when
the actual original stationary probability has full support.
Source: arXiv:2601.21366v2, §5.3, following `eq:global-Id`.
Its actual derivative is supplied by the proved original gradient formula. -/
theorem firstVariation_eq_of_stationary_full_support {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationary β σ ω a μ) (hfull : (μ : Measure (SSphere d)).support = univ) :
    ∀ x y : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) =
      firstVariation β φ ω a μ (y : EucSpace d) := by
  let D := fun x : EucSpace d =>
    (∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d) ∂(μ : Measure (SSphere d))) +
      ∑ j, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j
  have hD (x : EucSpace d) : HasFDerivAt (firstVariation β φ ω a μ) (innerSL ℝ (D x)) x := by
    have h := hasGradientAt_iff_hasFDerivAt.mp (hasGradientAt_firstVariation hβ φ σ hφ ω a μ x)
    apply h.congr_fderiv
    ext v
    rfl
  have hp (z : EucSpace d) (hz : ‖z‖ = 1) : proj d z (D z) = 0 := by
    let x : SSphere d := ⟨z, mem_sphere_zero_iff_norm.mpr hz⟩
    have hx : x ∈ (μ : Measure (SSphere d)).support := by rw [hfull]; exact mem_univ _
    rw [← energyGrad_eq_proj]
    exact hμ x hx
  exact eq_on_sphere_of_projected_derivative_zero hd (firstVariation β φ ω a μ) D hD hp

/-- An original full-support stationary probability forces the original
perceptron potential to be analytic on the actual sphere.
Source: arXiv:2601.21366v2, §5.3, the final contradiction after `eq:global-Id`.
The original primitive derivative hypothesis is retained, and the original
first variation is made constant using its proved actual derivative. -/
theorem isAnalyticOnSphere_of_stationary_full_support {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationary β σ ω a μ) (hfull : (μ : Measure (SSphere d)).support = univ) :
    IsAnalyticOnSphere φ ω a := by
  have hconst := firstVariation_eq_of_stationary_full_support hd β hβ φ σ hφ ω a μ hμ hfull
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  let C := firstVariation β φ ω a μ (x₀ : EucSpace d)
  refine ⟨fun z => 2 * (C - β⁻¹ * Perspective.partitionMu d β μ z), ?_, fun z hz => ?_⟩
  · intro z _
    exact analyticAt_const.fun_mul (analyticAt_const.fun_sub
      (analyticAt_const.fun_mul (analyticOnNhd_partitionMu_original β μ z (mem_univ z))))
  · have h := hconst (⟨z, mem_sphere_zero_iff_norm.mpr hz⟩ : SSphere d) x₀
    change β⁻¹ * Perspective.partitionMu d β μ z + (2 : ℝ)⁻¹ * potential φ ω a z = C at h
    linarith

/-- The original non-analytic potential excludes full original support
for every original stationary probability in dimension at least two.
Source: arXiv:2601.21366v2, §5.3, the final contradiction after `eq:global-Id`.
The activation and its actual primitive need only the original derivative
relation here; no analyticity of either is assumed. -/
theorem not_full_support_of_nonanalytic_stationary {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β σ ω a μ) :
    (μ : Measure (SSphere d)).support ≠ univ := by
  intro hfull
  exact hana (isAnalyticOnSphere_of_stationary_full_support hd β hβ φ σ hφ ω a μ hμ hfull)

/-- A stationary original ReLU measure with the source's actual
non-analytic potential has no globally zero original quadratic branch.
Source: arXiv:2601.21366v2, §5.3, contradiction after `eq:global-Id`.
The branch first gives full support by the proved original inverse,
then original stationarity would give the forbidden potential analyticity. -/
theorem not_zero_projected_relu_branch_of_nonanalytic {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β (fun s => max s 0) ω a μ)
    (J : Finset (Idx d)) :
    ¬ ∀ x : SSphere d, proj d (x : EucSpace d)
      (linearAttentionVector β (branchMaskedWeights ω J) a μ (x : EucSpace d)) = 0 := by
  intro hzero
  have hfull := support_eq_univ_of_zero_projected_relu_branch hd β hβ ω a μ J hzero
  exact hana (isAnalyticOnSphere_of_stationary_full_support hd β hβ.ne' φ
    (fun s => max s 0) hφ ω a μ hμ hfull)

/-- Every full-support stationary and primitive input holds jointly for
the actual original uniform circle law, the original ReLU primitive and
zero original weights. Source: arXiv:2601.21366v2, §5.3,
the full-support first-variation step after `eq:global-Id`.
The individual primitive is retained even though the weights are zero. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (1 : ℝ) ≠ 0 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    IsStationary 1 (fun s => max s 0) (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2) μ ∧
    (μ : Measure (SSphere 2)).support = univ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, le_rfl, one_ne_zero, hasDerivAt_reluSq,
    isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => max s 0) 0 μ hν,
    support_eq_univ_of_uniform μ hν⟩

/-- Every non-analytic branch-obstruction input holds jointly for the
source's actual pinned original Dirac probability and original one-neuron
ReLU potential. Source: arXiv:2601.21366v2, §3.1, `thm: circle`,
and §5.3, the contradiction after `eq:global-Id`. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨le_rfl, one_pos, one_ne_zero, hasDerivAt_reluSq,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1)⟩

end Transformer.Perceptron
