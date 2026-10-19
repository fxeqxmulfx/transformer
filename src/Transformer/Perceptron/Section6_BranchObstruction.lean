module

public import Transformer.Perceptron.Section6_SphericalBranchIdentity
public import Transformer.Perceptron.Section6_SignedQuadratic
public import Transformer.Perceptron.Section6_SphericalFields

/-!
# The original complementary-branch obstruction and pattern cover

ArXiv:2601.21366v2, §6, `eq:ratio-const-correct` in the proof of
`prop: unified.log`, part (a), and `rem:unifiedlog-B`.
If both actual complementary normalized branches vanished on the
whole sphere, their proved exponential-quadratic identities would
make the original signed Gram quadratic form constant. The proved
polarization argument contradicts the source's non-degeneracy.

Consequently, when one actual branch vanishes globally, the actual
complementary field is nonzero somewhere. Its proved analyticity and
the proved uniform-measure uniqueness theorem make its zero set null.
Invertibility is not used in this algebraic step; it enters later in
recovering the original measure's antipodal invariance.

The finite weak-pattern cover also retains zero neurons and every
activation boundary. Positive measure of the original support thus
supplies positive measure in one actual pattern without assuming an
open component or discarding a boundary set.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The original signed Gram action is the difference of the actual
two complementary branch vectors. Source: arXiv:2601.21366v2,
§6, `eq: nondeg.normalized`. Both sides use the same original
weighted rank-one summands, without a normalization or missing index. -/
theorem branchVector_sub_compl_eq_signedGram {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    branchVector ω a J x - branchVector ω a Jᶜ x = signedGram ω a J x := by
  simp only [branchVector, signedGram]

/-- The original complementary half-potential difference is half the
original signed Gram quadratic form in every dimension. Source:
arXiv:2601.21366v2, §6, `eq:ratio-const-correct`. -/
theorem branchHalfPotential_sub_compl {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    branchHalfPotential ω a J x - branchHalfPotential ω a Jᶜ x =
      (1 / 2 : ℝ) * inner (𝕜 := ℝ) (signedGram ω a J x) x := by
  rw [inner_signedGram_self]
  unfold branchHalfPotential
  ring

/-- Non-degeneracy excludes simultaneous global zeros of the actual
two complementary normalized branches. Source: arXiv:2601.21366v2,
§6, the contradiction from `eq:ratio-const-correct`.
The original probability measure and arbitrary original matrix are
retained; positivity and the global identities were proved for them. -/
theorem not_both_normalizedBranches_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (hnd : IsNonDegenerate ω a) (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) :
    ¬ ((∀ x : SSphere d, normalizedBranchField B ω a μ J (x : EucSpace d) = 0) ∧
      ∀ x : SSphere d, normalizedBranchField B ω a μ Jᶜ (x : EucSpace d) = 0) := by
  rintro ⟨hJ, hC⟩
  obtain ⟨CJ, hCJ⟩ := exists_exp_branch_identity_of_spherical_zero hd B ω a μ J hJ
  obtain ⟨CC, hCC⟩ := exists_exp_branch_identity_of_spherical_zero hd B ω a μ Jᶜ hC
  apply not_constant_signedGram_of_nonDegenerate ω a hnd J
  refine ⟨2 * (CJ - CC), fun x hx => ?_⟩
  let xp : SSphere d := ⟨x, mem_sphere_zero_iff_norm.mpr hx⟩
  have hlog : CJ - branchHalfPotential ω a J x = CC - branchHalfPotential ω a Jᶜ x :=
    Real.exp_injective ((hCJ xp).symm.trans (hCC xp))
  have hquad := branchHalfPotential_sub_compl ω a J x
  linarith

/-- If one actual branch is globally zero, its actual complement is
nonzero at an original sphere point. Source: arXiv:2601.21366v2,
§6, the complementary branch contradiction in `prop: unified.log`. -/
theorem exists_complement_branch_nonzero_of_spherical_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (hnd : IsNonDegenerate ω a) (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, normalizedBranchField B ω a μ J (x : EucSpace d) = 0) :
    ∃ x : SSphere d, normalizedBranchField B ω a μ Jᶜ (x : EucSpace d) ≠ 0 := by
  by_contra! hC
  exact not_both_normalizedBranches_zero hd B ω a hnd μ J ⟨hzero, hC⟩

/-- The actual complementary branch's zeros are uniform-null whenever
one original branch vanishes globally under non-degeneracy.
Source: arXiv:2601.21366v2, §6, the second uniqueness step and
`eq:ratio-const-correct` in `prop: unified.log`, part (a). -/
theorem measure_complement_branch_zeros_eq_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (hnd : IsNonDegenerate ω a) (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, normalizedBranchField B ω a μ J (x : EucSpace d) = 0)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν {x : SSphere d | normalizedBranchField B ω a μ Jᶜ (x : EucSpace d) = 0} = 0 :=
  measure_sphere_vector_zeros_eq_zero_of_analytic_nonzero hd _
    (analyticOnNhd_normalizedBranchField B ω a μ Jᶜ) ν hν
    (exists_complement_branch_nonzero_of_spherical_zero hd B ω a hnd μ J hzero)

/-- Positive measure of the original support gives positive measure
in one of its finitely many actual weak activation patterns.
Source: arXiv:2601.21366v2, §6, selecting `I` and `J_I` in
`prop: unified.log`, part (a). The finite cover includes all boundaries
and zero neurons; no geometric component hypothesis is needed. -/
theorem exists_positive_support_pattern {d : ℕ} (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d))
    (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∃ J : Finset (Idx d), 0 < ν ((μ : Measure (SSphere d)).support ∩
      {x : SSphere d | IsActivationPattern a J (x : EucSpace d)}) := by
  classical
  have hcover : (μ : Measure (SSphere d)).support ⊆
      ⋃ J : Finset (Idx d), (μ : Measure (SSphere d)).support ∩
        {x : SSphere d | IsActivationPattern a J (x : EucSpace d)} := by
    intro x hx
    obtain ⟨J, hJ⟩ := exists_activationPattern a (x : EucSpace d)
    exact mem_iUnion.mpr ⟨J, hx, hJ⟩
  exact exists_measure_pos_of_not_measure_iUnion_null
    (hpos.trans_le (measure_mono hcover)).ne'

/-- The non-degeneracy, dimension, globally zero branch and original
uniform-law inputs hold jointly for the actual pinned Dirac with zero
matrix and empty active set. Invertibility is not required above.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized` and
`eq:ratio-const-correct` before the matrix parity step. -/
example : (2 : ℕ) ≤ 2 ∧ IsNonDegenerate (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    (∀ x : SSphere 2, normalizedBranchField 0 (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∅ (x : EucSpace 2) = 0) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  by
    refine ⟨le_rfl, isNonDegenerate_pin, ?_,
      Metastability.isUniformOn_uniformLaw (by norm_num)⟩
    intro x
    exact normalizedBranchField_zero_matrix_empty (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) (x : EucSpace 2)

/-- The support-pattern positivity input is realized by the actual
pinned Dirac measured against itself. The inactive pattern contains
its genuine support point. Source: arXiv:2601.21366v2,
§6, the finite-pattern selection in `prop: unified.log`. -/
example : 0 < (Measure.dirac (basePoint 1 : SSphere 2))
    (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support := by
  have hx : basePoint 1 ∈ (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support := by
    apply (Measure.mem_support_iff_forall _).mpr
    intro U hU
    have hmem : basePoint 1 ∈ U := mem_of_mem_nhds hU
    change 0 < Measure.dirac (basePoint 1) U
    simp [hmem]
  simp [hx]

end Transformer.Perceptron
