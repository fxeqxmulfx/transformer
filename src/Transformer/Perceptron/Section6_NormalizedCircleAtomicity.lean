module

public import Transformer.Perceptron.Section6_BranchIdentity
public import Transformer.Perceptron.Section6_SignedQuadratic
public import Transformer.Perceptron.Section6_NormalizedForce
public import Transformer.Perceptron.TransformMap
public import Transformer.Perceptron.Atomicity

/-!
# Finite normalized stationary support on the circle

ArXiv:2601.21366v2, §6, `prop: unified.log`, part (b), and
`rem:unifiedlog-B`. Infinite actual support supplies infinitely many
angles of one activation pattern in a compact interval. The actual
normalized branch force vanishes there, so analyticity makes it zero
globally and gives the source's exponential-quadratic weight identity.

Evenness of that weight and injectivity of the actual invertible-matrix
transform give antipodal invariance of the original measure. The actual
stationary force then vanishes in the complementary pattern at the
antipodal angles. Analytic continuation makes both branch forces zero
globally. Positivity of the actual weight makes the complementary
quadratic difference constant, contradicting the signed Gram condition.

The source requires a symmetric nonsingular matrix. The proved field
argument needs invertibility alone, so the matrix result is strengthened
by dropping symmetry. For nonsymmetric matrices its hypothesis remains
the actual weighted field of `IsNormalizedStationaryMap`, not an energy
interpretation of that field. Weak activation patterns include zero
neurons and all hyperplane boundaries without additional assumptions.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Normalized ReLU stationarity and the signed Gram non-degeneracy
condition force finite actual topological support on the circle.
Source: arXiv:2601.21366v2, §6, `prop: unified.log`, part (b),
and `rem:unifiedlog-B`. Invertibility of `B` suffices: the source's
symmetry assumption is removed for the actual weighted-field statement.
Every support point and every neuron is retained throughout the proof. -/
theorem finite_support_of_normalizedStationaryMap_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite := by
  by_contra hInf
  let S : Set ℝ := Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support
  have hS : S.Infinite := by
    intro hfinite
    apply hInf
    apply (hfinite.image circlePoint).subset
    intro x hx
    obtain ⟨θ, hθ, heq⟩ := exists_circlePoint_eq x
    refine ⟨θ, ⟨hθ, ?_⟩, heq⟩
    change circlePoint θ ∈ (μ : Measure (SSphere 2)).support
    rwa [heq]
  obtain ⟨J, hT⟩ := exists_infinite_activation_angles a S hS
  let T : Set ℝ := S ∩ {θ | IsActivationPattern a J (circlePoint θ : EucSpace 2)}
  have hTsub : T ⊆ Icc (-π) π := fun _ hθ => hθ.1.1
  have hzeroJ : ∀ θ, circleNormalizedBranchForce B ω a μ J θ = 0 :=
    eq_zero_of_analytic_infinite_zeros _ (analyticOnNhd_circleNormalizedBranchForce B ω a μ J)
      (Icc (-π) π) isCompact_Icc T hT hTsub (fun θ hθ =>
        circleNormalizedBranchForce_zero_on_support B ω a μ hμ J θ hθ.1.2 hθ.2)
  have hperiod := periodic_circleAngularWeightMap_of_branch_zero B ω a μ J hzeroJ
  have hanti : (μ : Measure (SSphere 2)).map (antipodeMap 2) = (μ : Measure (SSphere 2)) :=
    (even_attentionTransformMap_iff B hB μ).mp
      (even_attentionTransformMap_of_branch_zero B ω a μ J hzeroJ)
  have hzeroComplement : ∀ θ ∈ T, circleNormalizedBranchForce B ω a μ Jᶜ θ = 0 := by
    intro θ hθ
    have hxanti : circlePoint (θ + π) ∈ (μ : Measure (SSphere 2)).support := by
      rw [circlePoint_add_pi, ← hanti]
      exact mem_support_map_of_continuous_circle _ (antipodeMap 2)
        (Perspective.continuous_sphereMap 2 (LinearIsometryEquiv.neg ℝ))
        (measurable_antipodeMap 2) (circlePoint θ) hθ.1.2
    have hp : IsActivationPattern a Jᶜ (circlePoint (θ + π) : EucSpace 2) := by
      rw [coe_circlePoint_add_pi]
      exact isActivationPattern_compl_neg a J (circlePoint θ : EucSpace 2) hθ.2
    have hz := circleNormalizedBranchForce_zero_on_support B ω a μ hμ Jᶜ (θ + π) hxanti hp
    simpa only [circleNormalizedBranchForce, (periodic_deriv_real hperiod) θ,
      hperiod θ, periodic_circleBranchForce ω a Jᶜ θ] using hz
  have hzeroC : ∀ θ, circleNormalizedBranchForce B ω a μ Jᶜ θ = 0 :=
    eq_zero_of_analytic_infinite_zeros _ (analyticOnNhd_circleNormalizedBranchForce B ω a μ Jᶜ)
      (Icc (-π) π) isCompact_Icc T hT hTsub hzeroComplement
  have hq (θ : ℝ) : circleBranchForce ω a J θ = circleBranchForce ω a Jᶜ θ := by
    apply mul_left_cancel₀ (circleAngularWeightMap_pos B μ θ).ne'
    have hj := hzeroJ θ
    have hc := hzeroC θ
    unfold circleNormalizedBranchForce at hj hc
    linarith
  have hd (θ : ℝ) : HasDerivAt (fun r => circleBranchHalfPotential ω a J r -
      circleBranchHalfPotential ω a Jᶜ r) 0 θ := by
    exact ((hasDerivAt_circleBranchHalfPotential ω a J θ).sub
      (hasDerivAt_circleBranchHalfPotential ω a Jᶜ θ)).congr_deriv (by rw [hq θ, sub_self])
  have hconstant := is_const_of_deriv_eq_zero
    (f := fun r => circleBranchHalfPotential ω a J r - circleBranchHalfPotential ω a Jᶜ r)
    (fun θ => (hd θ).differentiableAt) (fun θ => (hd θ).deriv)
  exact not_constant_circleBranchDifference ω a hnd J
    ⟨circleBranchHalfPotential ω a J 0 - circleBranchHalfPotential ω a Jᶜ 0,
      fun θ => hconstant θ 0⟩

/-- The original probability measure has an actual finite Dirac
presentation. Source: arXiv:2601.21366v2, §6, `prop: unified.log`,
part (b), and `rem:unifiedlog-B`. This retains the strengthened
invertible-matrix hypothesis of the support theorem. -/
theorem isFinitelyAtomic_of_normalizedStationaryMap_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_normalizedStationaryMap_circle B hB ω a hnd μ hμ)

/-- The original scalar normalized dynamics have finite actual circle
support. Source: arXiv:2601.21366v2, §6, `prop: unified.log`, part (b).
The source assumes `β>0`; invertibility of `β I` only needs `β≠0`,
so this intermediate support statement strengthens that assumption. -/
theorem finite_support_of_normalizedStationary_circle (β : ℝ) (hβ : β ≠ 0)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationary β (fun s => max s 0) ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite := by
  have hB : Function.Bijective (β • LinearMap.id (R := ℝ) (M := EucSpace 2)) := by
    constructor
    · intro x y hxy
      have h := congrArg (fun z : EucSpace 2 => β⁻¹ • z) hxy
      simpa only [LinearMap.smul_apply, LinearMap.id_apply, smul_smul,
        inv_mul_cancel₀ hβ, one_smul] using h
    · intro y
      refine ⟨β⁻¹ • y, ?_⟩
      simp only [LinearMap.smul_apply, LinearMap.id_apply, smul_smul,
        mul_inv_cancel₀ hβ, one_smul]
  exact finite_support_of_normalizedStationaryMap_circle _ hB ω a hnd μ
    ((isNormalizedStationaryMap_smul_id_iff β hβ _ ω a μ).mpr hμ)

/-- Invertibility, non-degeneracy and normalized stationarity all hold
jointly for the actual pinned Dirac and identity matrix. The scalar
nonzero parameter and stationarity inputs hold there at `β=1` too.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`,
`prop: unified.log` and `rem:unifiedlog-B`. The zero second neuron
is retained, so the example also checks the pattern cover at its boundary. -/
example : Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsNonDegenerate (Pi.single 0 1)
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
    IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => max s 0) (Pi.single 0 1)
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧ (1 : ℝ) ≠ 0 ∧
    IsNormalizedStationary 1 (fun s => max s 0) (Pi.single 0 1)
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨Function.bijective_id, isNonDegenerate_pin, isNormalizedStationaryMap_relu_pin 0 (basePoint 1),
    one_ne_zero, isNormalizedStationary_relu_pin 1 0 (basePoint 1)⟩

end Transformer.Perceptron
