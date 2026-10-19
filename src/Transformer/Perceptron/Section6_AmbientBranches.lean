module

public import Transformer.Perceptron.Section6_AmbientGradient
public import Transformer.Perceptron.Section6_NormalizedForce
public import Mathlib.Analysis.Calculus.FDeriv.Pow

/-!
# Actual normalized branch fields in every ambient dimension

ArXiv:2601.21366v2, §6, proof of `prop: unified.log` and
`rem:unifiedlog-B`. Each active pattern replaces the original ReLU
force by its original quadratic branch. The actual branch vector
is the derivative of half that quadratic potential, including the
source's factor one half.

The projected branch and the original exponential interaction field
are analytic in the ambient space. Their weighted sum is the actual
normalized stationary field whenever the original point has the
given activation pattern. The original probability measure, neuron
weights and every activation boundary are retained. No identity
between the stationary field and a gradient is assumed.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The actual half quadratic branch at the original ambient point.
Source: arXiv:2601.21366v2, §6, the branch term in `H_log`. -/
noncomputable def branchHalfPotential {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) : ℝ :=
  (1 / 2 : ℝ) * ∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) x ^ 2

/-- The actual vector derivative of the half quadratic branch.
Source: arXiv:2601.21366v2, §6, the globally continued branch gradient. -/
noncomputable def branchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) : EucSpace d :=
  ∑ j ∈ J, (ω j * inner (𝕜 := ℝ) (a j) x) • a j

/-- The original normalized field on a globally continued actual branch.
Source: arXiv:2601.21366v2, §6, multiplying `g_log` by its positive
interaction weight as in `eq: fulltrans.stat`. -/
noncomputable def normalizedBranchField {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (x : EucSpace d) : EucSpace d :=
  attentionGradMap B μ x + attentionWeightMap B μ x • proj d x (branchVector ω a J x)

/-- The actual quadratic branch is even at the antipodal point.
Source: arXiv:2601.21366v2, §6, evenness in `eq:log-identity-global`. -/
theorem branchHalfPotential_neg {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    branchHalfPotential ω a J (-x) = branchHalfPotential ω a J x := by
  simp only [branchHalfPotential, inner_neg_right, neg_sq]

/-- The actual branch vector is odd at the original antipodal point.
Source: arXiv:2601.21366v2, §6, comparing the two branch gradients. -/
theorem branchVector_neg {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    branchVector ω a J (-x) = -branchVector ω a J x := by
  simp only [branchVector, inner_neg_right, mul_neg, neg_smul, Finset.sum_neg_distrib]

/-- The displayed branch vector is proved to represent the actual
Fréchet derivative of the original half potential. Source:
arXiv:2601.21366v2, §6, differentiating the branch term in `H_log`.
The coefficients retain the original weights and the factor one half. -/
theorem hasFDerivAt_branchHalfPotential {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    HasFDerivAt (branchHalfPotential ω a J) (innerSL ℝ (branchVector ω a J x)) x := by
  have hterm (j : Idx d) := (((innerSL ℝ (a j)).hasFDerivAt (x := x)).pow 2).const_mul (ω j)
  have h := (HasFDerivAt.fun_sum (u := J) (fun j _ => hterm j)).const_mul (1 / 2 : ℝ)
  apply h.congr_fderiv
  ext v
  simp only [smul_apply, sum_apply, innerSL_apply_apply, branchVector, sum_inner,
    real_inner_smul_left, two_smul, smul_eq_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The actual vector branch is analytic at every original ambient point.
Source: arXiv:2601.21366v2, §6, the polynomial branch continuation. -/
theorem analyticOnNhd_branchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) :
    AnalyticOnNhd ℝ (branchVector ω a J) univ := by
  intro x _
  exact J.analyticAt_fun_sum (fun j _ =>
    (analyticAt_const.fun_mul ((innerSL ℝ (a j)).analyticAt x)).fun_smul analyticAt_const)

/-- The globally continued original normalized branch field is analytic.
Source: arXiv:2601.21366v2, §6, the actual vector field used
for positive-measure uniqueness in `prop: unified.log`, part (a). -/
theorem analyticOnNhd_normalizedBranchField {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) : AnalyticOnNhd ℝ (normalizedBranchField B ω a μ J) univ := by
  intro x hx
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  have hb := analyticOnNhd_branchVector ω a J x hx
  have hi : AnalyticAt ℝ (fun z => inner (𝕜 := ℝ) z (branchVector ω a J z)) x :=
    (L.analyticAt_bilinear (x, branchVector ω a J x)).comp
      (f := fun z => (z, branchVector ω a J z)) (analyticAt_id.prod hb)
  have hp : AnalyticAt ℝ (fun z => proj d z (branchVector ω a J z)) x :=
    hb.fun_sub (hi.fun_smul analyticAt_id)
  exact (analyticOnNhd_attentionGradMap B μ x hx).fun_add
    ((analyticOnNhd_attentionWeightMap B μ x hx).fun_smul hp)

/-- At an actual point of the supplied pattern, the original ReLU drift
is the projected original branch vector. Source: arXiv:2601.21366v2,
§6, the restriction to `J_I`. Zero neurons and boundaries retain their
actual zero contribution, without an additional regularity hypothesis. -/
theorem drift_relu_eq_projected_branchVector {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d)
    (hx : IsActivationPattern a J x) :
    drift (fun s => max s 0) ω a x = proj d x (branchVector ω a J x) := by
  classical
  unfold drift branchVector
  congr 1
  symm
  calc
    _ = ∑ j ∈ J, (ω j * max (inner (𝕜 := ℝ) (a j) x) 0) • a j :=
      Finset.sum_congr rfl (fun j hj => by rw [max_eq_left (hx.1 j hj)])
    _ = _ := Finset.sum_subset (Finset.subset_univ J) (fun j _ hj => by
      rw [max_eq_right (hx.2 j hj), mul_zero, zero_smul])

/-- The continued branch equals the actual original stationary field
at each point with that pattern. Source: arXiv:2601.21366v2,
§6, the support inclusion used in `prop: unified.log`, part (a). -/
theorem normalizedBranchField_eq_field_at_pattern {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) (x : EucSpace d)
    (hx : IsActivationPattern a J x) :
    normalizedBranchField B ω a μ J x = attentionGradMap B μ x +
      attentionWeightMap B μ x • drift (fun s => max s 0) ω a x := by
  rw [normalizedBranchField, drift_relu_eq_projected_branchVector ω a J x hx]

/-- Normalized stationarity makes the actual branch field zero at every
actual support point with that pattern. Source: arXiv:2601.21366v2,
§6, the actual support inclusion in `prop: unified.log`, part (a). -/
theorem normalizedBranchField_zero_on_support {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ)
    (J : Finset (Idx d)) (x : SSphere d) (hx : x ∈ (μ : Measure (SSphere d)).support)
    (hpat : IsActivationPattern a J (x : EucSpace d)) :
    normalizedBranchField B ω a μ J (x : EucSpace d) = 0 := by
  rw [normalizedBranchField_eq_field_at_pattern B ω a μ J x hpat]
  exact hμ x hx

/-- Pattern, original normalized stationarity and actual support inputs
hold together for the source's pinned Dirac, identity matrix, one negative
neuron and empty active set. Source: arXiv:2601.21366v2,
§6, `eq: fulltrans.stat` and the branch in `H_log`. -/
example : IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => max s 0) (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧
    basePoint 1 ∈ (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support ∧
    IsActivationPattern (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∅ (basePoint 1 : EucSpace 2) := by
  refine ⟨isNormalizedStationaryMap_relu_pin 0 _, ?_, isActivationPattern_single_neg_empty 0 _⟩
  apply (Measure.mem_support_iff_forall _).mpr
  intro U hU
  have hx : basePoint 1 ∈ U := mem_of_mem_nhds hU
  change 0 < Measure.dirac (basePoint 1) U
  simp [hx]

end Transformer.Perceptron
