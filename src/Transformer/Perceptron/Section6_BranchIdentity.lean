/-
# The global identity of an accumulating normalized circle branch

ArXiv:2601.21366v2, §6, proof of `prop: unified.log`, part (b),
and `eq:log-identity-global`. Infinitely many branch zeros in a
compact set force the actual analytic branch force to vanish globally.

The global first-order equation is integrated using the actual
factor `exp(P)`: the derivative of `F exp(P)` is `exp(P)(F'+F P')`.
Its constancy and the branch's period `π` make the original weight
even on the sphere. This is the same global identity as the source,
with an integrating factor in place of differentiating `log(F)`.
The resulting weight formula retains both the original probability
measure and the source's factor one half in the perceptron potential.
The global-zero hypotheses below precede the non-degeneracy contradiction.
The integrating-factor step requires neither matrix symmetry nor
invertibility. Invertibility is used later to recover the measure's
antipodal invariance from evenness of its actual interaction transform.
-/

import Transformer.Perceptron.Section6_BranchCalculus
import Mathlib.Analysis.Analytic.IsolatedZeros
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Order.Interval.Set.Infinite

open Real Set Filter
open scoped Topology

namespace Transformer.Perceptron

/-- An analytic function on the real line with infinitely many zeros
in a compact set vanishes globally. Source: arXiv:2601.21366v2, §6,
the identity theorem used in part (b) of `prop: unified.log`.
The accumulation point need not belong to the selected branch. -/
theorem eq_zero_of_analytic_infinite_zeros (f : ℝ → ℝ)
    (hf : AnalyticOnNhd ℝ f univ) (K : Set ℝ) (hK : IsCompact K)
    (S : Set ℝ) (hS : S.Infinite) (hSK : S ⊆ K) (hzero : ∀ θ ∈ S, f θ = 0) :
    ∀ θ, f θ = 0 := by
  obtain ⟨p, -, hacc⟩ := hS.exists_accPt_of_subset_isCompact hK hSK
  have hcl : p ∈ closure (S \ {p}) :=
    mem_closure_iff_clusterPt.mpr (accPt_principal_iff_clusterPt.mp hacc)
  have hsub : S \ {p} ⊆ {θ | f θ = 0} \ {p} :=
    fun θ hθ => ⟨hzero θ hθ.1, hθ.2⟩
  have hzeros : p ∈ closure ({θ | f θ = 0} \ {p}) := closure_mono hsub hcl
  exact fun θ => hf.eqOn_zero_of_preconnected_of_mem_closure
    isPreconnected_univ (mem_univ p) hzeros (mem_univ θ)

/-- All inputs to the identity theorem hold jointly for the zero
function on the genuine infinite compact interval `[0,1]`.
Source: arXiv:2601.21366v2, §6, the accumulation-point argument. -/
example : AnalyticOnNhd ℝ (fun _ : ℝ => (0 : ℝ)) univ ∧
    IsCompact (Icc (0 : ℝ) 1) ∧ (Icc (0 : ℝ) 1).Infinite ∧
    (Icc (0 : ℝ) 1 ⊆ Icc 0 1) ∧
    (∀ θ ∈ Icc (0 : ℝ) 1, (fun _ : ℝ => (0 : ℝ)) θ = 0) :=
  ⟨fun _ _ => analyticAt_const, isCompact_Icc, Icc_infinite zero_lt_one,
    subset_refl _, fun _ _ => rfl⟩

/-- The actual integrating factor differentiates to the multiplied
branch force. Source: arXiv:2601.21366v2, §6, integrating the
global continuation of `g_log` to obtain `eq:log-identity-global`. -/
theorem hasDerivAt_circleBranchIntegratingFactor (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) (θ : ℝ) :
    HasDerivAt (fun r => circleAngularWeightMap B μ r *
        Real.exp (circleBranchHalfPotential ω a J r))
      (Real.exp (circleBranchHalfPotential ω a J θ) *
        circleNormalizedBranchForce B ω a μ J θ) θ := by
  have hF := (analyticOnNhd_circleAngularWeightMap B μ θ (mem_univ θ)).differentiableAt.hasDerivAt
  have hP := (hasDerivAt_circleBranchHalfPotential ω a J θ).exp
  exact (hF.mul hP).congr_deriv (by unfold circleNormalizedBranchForce; ring)

/-- A globally zero actual branch force makes the actual integrating
factor constant. Source: arXiv:2601.21366v2, §6,
`eq:log-identity-global`. No parity or finite support is assumed. -/
theorem circleBranchIntegratingFactor_constant (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) (hzero : ∀ θ, circleNormalizedBranchForce B ω a μ J θ = 0)
    (θ ψ : ℝ) :
    circleAngularWeightMap B μ θ * Real.exp (circleBranchHalfPotential ω a J θ) =
      circleAngularWeightMap B μ ψ * Real.exp (circleBranchHalfPotential ω a J ψ) := by
  refine is_const_of_deriv_eq_zero
    (f := fun r => circleAngularWeightMap B μ r * Real.exp (circleBranchHalfPotential ω a J r))
    (fun r => (hasDerivAt_circleBranchIntegratingFactor B ω a μ J r).differentiableAt) ?_ θ ψ
  intro r
  rw [(hasDerivAt_circleBranchIntegratingFactor B ω a μ J r).deriv, hzero r, mul_zero]

/-- The actual original weight has the global exponential-quadratic
form. Source: arXiv:2601.21366v2, §6, `eq:log-identity-global`.
Writing the constant using another actual angle avoids introducing a
logarithm; the ratio is the same as the source's exponential identity. -/
theorem circleAngularWeightMap_eq_exp_branch (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) (hzero : ∀ θ, circleNormalizedBranchForce B ω a μ J θ = 0)
    (θ ψ : ℝ) : circleAngularWeightMap B μ θ = circleAngularWeightMap B μ ψ *
      Real.exp (circleBranchHalfPotential ω a J ψ - circleBranchHalfPotential ω a J θ) := by
  apply mul_right_cancel₀ (Real.exp_pos (circleBranchHalfPotential ω a J θ)).ne'
  rw [circleBranchIntegratingFactor_constant B ω a μ J hzero θ ψ,
    mul_assoc, ← Real.exp_add, sub_add_cancel]

/-- The original interaction weight is even when one actual branch
vanishes globally. Source: arXiv:2601.21366v2, §6, the evenness
of `eq:log-identity-global` before the transform's parity theorem. -/
theorem periodic_circleAngularWeightMap_of_branch_zero (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) (hzero : ∀ θ, circleNormalizedBranchForce B ω a μ J θ = 0) :
    Function.Periodic (circleAngularWeightMap B μ) π := by
  intro θ
  have heq := circleBranchIntegratingFactor_constant B ω a μ J hzero (θ + π) θ
  rw [periodic_circleBranchHalfPotential ω a J θ] at heq
  exact mul_right_cancel₀ (Real.exp_pos _).ne' heq

/-- The period in the actual circle chart gives evenness on every
actual sphere point. Source: arXiv:2601.21366v2, §6,
`eq:log-identity-global`. The entire sphere is covered by the chart. -/
theorem even_attentionTransformMap_of_branch_zero (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) (hzero : ∀ θ, circleNormalizedBranchForce B ω a μ J θ = 0) :
    ∀ x, attentionTransformMap B μ (antipodeMap 2 x) = attentionTransformMap B μ x := by
  intro x
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  rw [← circlePoint_add_pi]
  exact periodic_circleAngularWeightMap_of_branch_zero B ω a μ J hzero θ

/-- With zero attention matrix the actual probability weight is one.
Source: arXiv:2601.21366v2, §6, strict positivity of the interaction.
This supplies a genuine measure witness for global branch identities. -/
theorem circleAngularWeightMap_zero (μ : Perspective.ProbSphere 2) :
    circleAngularWeightMap 0 μ = fun _ => (1 : ℝ) := by
  funext θ
  simp [circleAngularWeightMap, attentionWeightMap]

/-- For the empty active set and zero matrix, the actual global branch
force vanishes. Source: arXiv:2601.21366v2, §6, the branch field
before invertibility and non-degeneracy enter the parity contradiction. -/
theorem circleNormalizedBranchForce_zero_empty (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    circleNormalizedBranchForce 0 ω a μ ∅ θ = 0 := by
  simp [circleNormalizedBranchForce, circleAngularWeightMap_zero, circleBranchForce]

/-- The global zero-force hypotheses of the integrating-factor,
periodicity and evenness statements hold together for an actual pinned
Dirac measure and empty branch. The two neurons and one weight are real
source data; matrix invertibility is not an input to these statements.
Source: arXiv:2601.21366v2, §6, `eq:log-identity-global`. -/
example : ∀ θ, circleNormalizedBranchForce 0 (Pi.single 0 1)
    (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
    (Perspective.diracProb 2 (basePoint 1)) ∅ θ = 0 :=
  circleNormalizedBranchForce_zero_empty _ _ _

end Transformer.Perceptron
