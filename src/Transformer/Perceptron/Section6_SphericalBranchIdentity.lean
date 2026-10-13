/-
# The actual global normalized branch identity on the sphere

ArXiv:2601.21366v2, §6, `eq:log-identity-global` in the proof of
`prop: unified.log`, part (a), and `rem:unifiedlog-B`.
The actual integrating factor is the original weight multiplied by
the exponential of the original half quadratic branch.

Its actual derivative is represented by
`exp(P) (G + F T)`, with the original ambient interaction integral `G`,
weight `F`, and original branch vector `T`. The tangent projection is
`exp(P)` times the original normalized branch field. Thus a globally
zero branch field makes the original integrating factor constant on
the actual sphere, by the proved derivative constancy theorem.

Cancelling the positive exponential gives the source's full global
exponential-quadratic identity and its evenness. The derivative and
constancy calculations need neither invertibility nor symmetry of `B`.
Invertibility is used later to recover antipodal measure invariance.
All global-zero hypotheses here have actual nonempty measure witnesses.
-/

import Transformer.Perceptron.Section6_AmbientBranches
import Transformer.Perceptron.Section6_SphericalFields
import Transformer.Perceptron.Section3_StationaryInterior
import Transformer.Perceptron.Transform

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The actual integrating factor before any global branch conclusion.
Source: arXiv:2601.21366v2, §6, `eq:log-identity-global`.
It is the original probability integral multiplied by `exp(P)`. -/
noncomputable def branchIntegratingFactor {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (x : EucSpace d) : ℝ :=
  attentionWeightMap B μ x * Real.exp (branchHalfPotential ω a J x)

/-- The original branch field is the actual projection of the ambient
vector derivative before the integrating factor. Source:
arXiv:2601.21366v2, §6, the original `g_log` multiplied by its weight. -/
theorem normalizedBranchField_eq_projected {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) (x : EucSpace d) :
    normalizedBranchField B ω a μ J x = proj d x
      (ambientAttentionGradMap B μ x + attentionWeightMap B μ x • branchVector ω a J x) := by
  rw [normalizedBranchField, attentionGradMap_eq_proj_ambientGradMap, proj_add_smul]

/-- The derivative of the original integrating factor is proved from
the actual weight and branch derivatives. Source: arXiv:2601.21366v2,
§6, integrating the continued branch to obtain `eq:log-identity-global`.
No force or gradient identity is introduced as an input. -/
theorem hasFDerivAt_branchIntegratingFactor {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) (x : EucSpace d) :
    HasFDerivAt (branchIntegratingFactor B ω a μ J)
      (innerSL ℝ (Real.exp (branchHalfPotential ω a J x) •
        (ambientAttentionGradMap B μ x + attentionWeightMap B μ x • branchVector ω a J x))) x := by
  apply ((hasFDerivAt_attentionWeightMap B μ x).fun_mul
    (hasFDerivAt_branchHalfPotential ω a J x).exp).congr_fderiv
  ext v
  simp only [smul_apply, add_apply, attentionCovectorMap_eq_innerSL_ambientGradMap,
    innerSL_apply_apply, inner_add_left, real_inner_smul_left, smul_eq_mul]
  ring

/-- A globally zero actual branch field makes the original integrating
factor constant on the actual sphere. Source: arXiv:2601.21366v2,
§6, `eq:log-identity-global`. Only the original tangent projection
vanishes; no constancy in the ambient space is asserted. -/
theorem branchIntegratingFactor_eq_on_sphere {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, normalizedBranchField B ω a μ J (x : EucSpace d) = 0) :
    ∀ x y : SSphere d, branchIntegratingFactor B ω a μ J (x : EucSpace d) =
      branchIntegratingFactor B ω a μ J (y : EucSpace d) := by
  apply eq_on_sphere_of_projected_derivative_zero hd _
    (fun x => Real.exp (branchHalfPotential ω a J x) •
      (ambientAttentionGradMap B μ x + attentionWeightMap B μ x • branchVector ω a J x))
    (hasFDerivAt_branchIntegratingFactor B ω a μ J)
  intro x hx
  rw [proj_smul_right, ← normalizedBranchField_eq_projected]
  have hz := hzero ⟨x, mem_sphere_zero_iff_norm.mpr hx⟩
  rw [hz, smul_zero]

/-- The original probability weight has the source's actual global
exponential-quadratic form. Source: arXiv:2601.21366v2, §6,
`eq:log-identity-global`. The constant is expressed at another actual
sphere point, with the original half quadratic potential retained. -/
theorem attentionWeightMap_eq_exp_branch_of_spherical_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, normalizedBranchField B ω a μ J (x : EucSpace d) = 0)
    (x y : SSphere d) : attentionWeightMap B μ (x : EucSpace d) =
      attentionWeightMap B μ (y : EucSpace d) * Real.exp
        (branchHalfPotential ω a J (y : EucSpace d) - branchHalfPotential ω a J (x : EucSpace d)) := by
  apply mul_right_cancel₀ (Real.exp_pos (branchHalfPotential ω a J (x : EucSpace d))).ne'
  rw [mul_assoc, ← Real.exp_add, sub_add_cancel]
  exact branchIntegratingFactor_eq_on_sphere hd B ω a μ J hzero x y

/-- The source's literal global exponential-quadratic identity, with
its real constant produced from an actual point and the proved positive
original weight. Source: arXiv:2601.21366v2, §6,
`eq:log-identity-global`. No constant or logarithmic identity is assumed. -/
theorem exists_exp_branch_identity_of_spherical_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, normalizedBranchField B ω a μ J (x : EucSpace d) = 0) :
    ∃ C : ℝ, ∀ x : SSphere d, attentionWeightMap B μ (x : EucSpace d) =
      Real.exp (C - branchHalfPotential ω a J (x : EucSpace d)) := by
  obtain ⟨y⟩ := nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d))
  refine ⟨Real.log (attentionWeightMap B μ (y : EucSpace d)) +
    branchHalfPotential ω a J (y : EucSpace d), fun x => ?_⟩
  rw [attentionWeightMap_eq_exp_branch_of_spherical_zero hd B ω a μ J hzero x y]
  rw [show Real.log (attentionWeightMap B μ (y : EucSpace d)) +
      branchHalfPotential ω a J (y : EucSpace d) - branchHalfPotential ω a J (x : EucSpace d) =
    Real.log (attentionWeightMap B μ (y : EucSpace d)) +
      (branchHalfPotential ω a J (y : EucSpace d) - branchHalfPotential ω a J (x : EucSpace d)) by ring,
    Real.exp_add, Real.exp_log (attentionWeightMap_pos B μ (y : EucSpace d))]

/-- The actual original weight is even when one actual branch field
vanishes globally on the sphere. Source: arXiv:2601.21366v2,
§6, evenness of `eq:log-identity-global`. Matrix invertibility
is not used until the subsequent measure parity argument. -/
theorem even_attentionTransformMap_of_spherical_branch_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, normalizedBranchField B ω a μ J (x : EucSpace d) = 0) :
    ∀ x : SSphere d, attentionTransformMap B μ (antipodeMap d x) = attentionTransformMap B μ x := by
  intro x
  have h := branchIntegratingFactor_eq_on_sphere hd B ω a μ J hzero (antipodeMap d x) x
  change attentionWeightMap B μ (-(x : EucSpace d)) *
      Real.exp (branchHalfPotential ω a J (-(x : EucSpace d))) =
    attentionWeightMap B μ (x : EucSpace d) * Real.exp (branchHalfPotential ω a J (x : EucSpace d)) at h
  rw [branchHalfPotential_neg] at h
  exact mul_right_cancel₀ (Real.exp_pos _).ne' h

/-- The original zero-matrix, empty-branch field actually vanishes.
Source: arXiv:2601.21366v2, §6, the branch identity before
invertibility enters. This supplies the actual global-zero witness. -/
theorem normalizedBranchField_zero_matrix_empty {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    normalizedBranchField 0 ω a μ ∅ x = 0 := by
  simp [normalizedBranchField, attentionGradMap, branchVector, proj]

/-- All dimension and global-zero hypotheses of the spherical identity,
exponential formula and evenness statements hold jointly for the actual
pinned Dirac and empty branch, with zero attention matrix. Invertibility
is not an input to those statements. Source: arXiv:2601.21366v2,
§6, `eq:log-identity-global` before its measure parity step. -/
example : (2 : ℕ) ≤ 2 ∧ ∀ x : SSphere 2, normalizedBranchField 0
    (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
    (Perspective.diracProb 2 (basePoint 1)) ∅ (x : EucSpace 2) = 0 :=
  ⟨le_rfl, fun x => normalizedBranchField_zero_matrix_empty _ _ _ (x : EucSpace 2)⟩

end Transformer.Perceptron
