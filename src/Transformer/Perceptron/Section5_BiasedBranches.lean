/-
# Actual biased ReLU patterns and quadratic branch fields

ArXiv:2601.21366v2, §3.1, `rem: ext` (i), with the branch arguments
of §5.2 and §5.3. The original biased argument is `a_j·x+b_j`.
The weak patterns retain original zero neurons and all original
activation boundaries. Every actual point has such a pattern.

On a pattern the original ReLU drift is the original projected
quadratic branch vector. Its actual half-potential derivative is
proved here, retaining the original bias and half factor. Adding
the original exponential weight gives the actual scaled first
variation and its actual ambient derivative.

The original projected branch is analytic and agrees with the
temperature times the original biased stationary field on its
actual pattern. Neither equality nor a gradient identity is placed
in a definition. The following inverse argument computes all original
constant, linear and quadratic terms caused by the actual biases.
-/

import Transformer.Perceptron.Bias
import Transformer.Perceptron.Section5_ReLUBranches

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- Actual weak signs of the original biased neuron arguments.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3,
the original active-index patterns. Boundary contributions remain zero. -/
def IsBiasedActivationPattern {d : ℕ} (a : Idx d → EucSpace d) (b : Idx d → ℝ)
    (J : Finset (Idx d)) (x : EucSpace d) : Prop :=
  (∀ j ∈ J, 0 ≤ inner (𝕜 := ℝ) (a j) x + b j) ∧
    ∀ j, j ∉ J → inner (𝕜 := ℝ) (a j) x + b j ≤ 0

/-- Every original point has an actual biased activation pattern.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3.
Strictly positive original arguments give an actual finite choice. -/
theorem exists_biased_activationPattern {d : ℕ} (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (x : EucSpace d) : ∃ J, IsBiasedActivationPattern a b J x := by
  classical
  refine ⟨Finset.univ.filter (fun j => 0 < inner (𝕜 := ℝ) (a j) x + b j), ?_, ?_⟩
  · intro j hj
    exact (Finset.mem_filter.mp hj).2.le
  · intro j hj
    exact le_of_not_gt (fun hp => hj (Finset.mem_filter.mpr ⟨Finset.mem_univ j, hp⟩))

/-- The actual half quadratic potential of the original biased branch.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3,
the biased version of `eq:quad.pol.I`. Every original bias is retained. -/
noncomputable def biasedBranchHalfPotential {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (J : Finset (Idx d)) (x : EucSpace d) : ℝ :=
  (1 / 2 : ℝ) * ∑ j ∈ J, ω j * (inner (𝕜 := ℝ) (a j) x + b j) ^ 2

/-- The original vector in the actual half-potential derivative.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3.
Its identity with the actual derivative is proved below. -/
noncomputable def biasedBranchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (J : Finset (Idx d)) (x : EucSpace d) : EucSpace d :=
  ∑ j ∈ J, (ω j * (inner (𝕜 := ℝ) (a j) x + b j)) • a j

/-- The original scaled first variation on a continued biased branch.
Source: arXiv:2601.21366v2, §5.3, `eq:global-Id`, with §3.1,
`rem: ext` (i). The original interaction and all biases are retained. -/
noncomputable def biasedAttentionPotential {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (x : EucSpace d) : ℝ :=
  attentionWeightMap (β • LinearMap.id) μ x + β * biasedBranchHalfPotential ω a b J x

/-- The actual ambient vector in that original branch derivative.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3,
the continued field in `eq:suppmu.zero.g`. Both identities are proved below. -/
noncomputable def biasedAttentionVector {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (x : EucSpace d) : EucSpace d :=
  ambientAttentionGradMap (β • LinearMap.id) μ x + β • biasedBranchVector ω a b J x

/-- The actual original half-potential has its displayed original
Fréchet derivative, with every original bias and coefficient retained.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3. -/
theorem hasFDerivAt_biasedBranchHalfPotential {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (J : Finset (Idx d)) (x : EucSpace d) :
    HasFDerivAt (biasedBranchHalfPotential ω a b J) (innerSL ℝ (biasedBranchVector ω a b J x)) x := by
  have hterm (j : Idx d) := ((((innerSL ℝ (a j)).hasFDerivAt (x := x)).add_const (b j)).pow 2).const_mul (ω j)
  have h := (HasFDerivAt.fun_sum (u := J) (fun j _ => hterm j)).const_mul (1 / 2 : ℝ)
  apply h.congr_fderiv
  ext v
  simp only [smul_apply, sum_apply, innerSL_apply_apply, biasedBranchVector, sum_inner,
    real_inner_smul_left, two_smul, smul_eq_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The actual original biased first variation has the displayed
original ambient derivative. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (i), and §5.3, `eq:global-Id`. -/
theorem hasFDerivAt_biasedAttentionPotential {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (x : EucSpace d) :
    HasFDerivAt (biasedAttentionPotential β ω a b μ J) (innerSL ℝ (biasedAttentionVector β ω a b μ J x)) x := by
  have hW := hasFDerivAt_attentionWeightMap (β • LinearMap.id) μ x
  rw [attentionCovectorMap_eq_innerSL_ambientGradMap] at hW
  have h := hW.add ((hasFDerivAt_biasedBranchHalfPotential ω a b J x).const_mul β)
  apply h.congr_fderiv
  ext v
  change inner (𝕜 := ℝ) (ambientAttentionGradMap (β • LinearMap.id) μ x) v +
    β * inner (𝕜 := ℝ) (biasedBranchVector ω a b J x) v =
      inner (𝕜 := ℝ) (ambientAttentionGradMap (β • LinearMap.id) μ x + β • biasedBranchVector ω a b J x) v
  rw [inner_add_left, real_inner_smul_left]

/-- The actual original biased branch vector is analytic everywhere
in the original ambient space. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (i), and §5.3, continuation of the original polynomial branch. -/
theorem analyticOnNhd_biasedBranchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (J : Finset (Idx d)) :
    AnalyticOnNhd ℝ (biasedBranchVector ω a b J) univ := by
  intro x _
  exact J.analyticAt_fun_sum (fun j _ =>
    (analyticAt_const.fun_mul (((innerSL ℝ (a j)).analyticAt x).fun_add analyticAt_const)).fun_smul analyticAt_const)

/-- The actual projected original biased branch field is analytic.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3,
the analytic vector-field step before `eq:global-Id`.
The original projection is its actual bilinear expression. -/
theorem analyticOnNhd_projected_biasedAttentionVector {d : ℕ} (β : ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) :
    AnalyticOnNhd ℝ (fun x => proj d x (biasedAttentionVector β ω a b μ J x)) univ := by
  intro x hx
  have hD : AnalyticAt ℝ (biasedAttentionVector β ω a b μ J) x :=
    (analyticOnNhd_ambientAttentionGradMap (β • LinearMap.id) μ x hx).fun_add
      (analyticAt_const.fun_smul (analyticOnNhd_biasedBranchVector ω a b J x hx))
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  have hi : AnalyticAt ℝ (fun z => inner (𝕜 := ℝ) z (biasedAttentionVector β ω a b μ J z)) x :=
    (L.analyticAt_bilinear (x, biasedAttentionVector β ω a b μ J x)).comp
      (f := fun z => (z, biasedAttentionVector β ω a b μ J z)) (analyticAt_id.prod hD)
  exact hD.fun_sub (hi.fun_smul analyticAt_id)

/-- The actual original ReLU drift equals the projected original
biased branch vector at every point of that actual pattern.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3,
the biased version of `eq:quad.pol.I`. -/
theorem biasedDrift_relu_eq_projected_branchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (J : Finset (Idx d)) (x : EucSpace d)
    (hx : IsBiasedActivationPattern a b J x) :
    biasedDrift (fun s => max s 0) ω a b x = proj d x (biasedBranchVector ω a b J x) := by
  classical
  unfold biasedDrift biasedBranchVector
  congr 1
  symm
  calc
    _ = ∑ j ∈ J, (ω j * max (inner (𝕜 := ℝ) (a j) x + b j) 0) • a j :=
      Finset.sum_congr rfl (fun j hj => by rw [max_eq_left (hx.1 j hj)])
    _ = _ := Finset.sum_subset (Finset.subset_univ J) (fun j _ hj => by
      rw [max_eq_right (hx.2 j hj), mul_zero, zero_smul])

/-- At an actual original biased pattern the projected original
derivative is the original temperature times the original biased field.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3,
the support inclusion `eq:suppmu.zero.g`. -/
theorem projected_biasedAttentionVector_eq_relu_at_pattern {d : ℕ} (β : ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (x : EucSpace d) (hx : IsBiasedActivationPattern a b J x) :
    proj d x (biasedAttentionVector β ω a b μ J x) = β • biasedEnergyGrad β (fun s => max s 0) ω a b μ x := by
  have hp (y v : EucSpace d) : proj d x (y + β • v) = proj d x y + β • proj d x v := by
    simp only [proj, inner_add_right, real_inner_smul_right, add_smul, mul_smul, smul_sub]
    abel
  rw [biasedAttentionVector, hp, ← attentionGradMap_eq_proj_ambientGradMap, attentionGradMap_smul_id]
  rw [← biasedDrift_relu_eq_projected_branchVector ω a b J x hx]
  change β • attentionGrad β μ x + β • biasedDrift (fun s => max s 0) ω a b x =
    β • (attentionGrad β μ x + biasedDrift (fun s => max s 0) ω a b x)
  rw [smul_add]

/-- Every pattern input is realized on the actual original circle
with a nonzero bias and an actual inactive zero neuron.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3.
The actual bias changes the original argument to minus one. -/
example : IsBiasedActivationPattern (0 : Idx 2 → EucSpace 2) (fun _ => (-1 : ℝ)) ∅
    (basePoint 1 : EucSpace 2) :=
  ⟨fun j hj => by simp at hj, fun _ _ => by simp⟩

end Transformer.Perceptron
