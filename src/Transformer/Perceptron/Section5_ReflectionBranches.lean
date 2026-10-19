module

public import Transformer.Perceptron.Section5_NeuronReflection

/-!
# Comparing the original ReLU field across one neuron reflection

ArXiv:2601.21366v2, §5.2, `eq:toggle-diff-general`, and §6,
the symmetry step of `rem:unifiedlog-B`. An actual isometry negating
the first neuron and fixing the second reverses just the first input.
The original weak pattern inequalities determine both reflected ReLU
inputs, including all activation boundaries.

Subtracting the original ReLU field from the continued masked linear
field cancels the actual attention term and the second neuron. What
remains is the signed first-neuron coefficient times its actual
spherical projection. This identity does not assume stationarity,
nonzero inputs, an invertible matrix, or invariance of the law.

The separate law-invariance result uses invertibility and explicit
commutation of the actual matrix with the reflection. These are extra
premises of this intermediate case; the source's full general-matrix
claim remains separately stated in `GeneralAtomicity`.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The original projector is additive in the projected vector.
Source: arXiv:2601.21366v2, §2, `eq:WGF-main`. -/
theorem proj_sub_right {d : ℕ} (x u v : EucSpace d) :
    proj d x (u - v) = proj d x u - proj d x v := by
  simp only [proj, inner_sub_right, sub_smul]
  abel

/-- The actual reflected ReLU inputs switch precisely the first
original activation. Source: arXiv:2601.21366v2, §5.2,
the branch comparison in `eq:toggle-diff-general`. -/
theorem relu_inputs_at_neuron_reflection (a : Idx 2 → EucSpace 2)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (h₀ : U (a 0) = -a 0) (h₁ : U (a 1) = a 1)
    (J : Finset (Idx 2)) (x : EucSpace 2) (hx : IsActivationPattern a J x) :
    max (inner (𝕜 := ℝ) (a 0) (U x)) 0 =
        (if 0 ∈ J then 0 else inner (𝕜 := ℝ) (a 0) (U x)) ∧
      max (inner (𝕜 := ℝ) (a 1) (U x)) 0 =
        (if 1 ∈ J then inner (𝕜 := ℝ) (a 1) (U x) else 0) := by
  obtain ⟨hi₀, hi₁⟩ := inner_neuron_reflection_inputs U (a 0) (a 1) h₀ h₁ x
  constructor
  · by_cases hj : 0 ∈ J
    · rw [ite_eq_left hj]
      apply max_eq_right
      linarith [hx.1 0 hj]
    · rw [ite_eq_right hj]
      apply max_eq_left
      linarith [hx.2 0 hj]
  · by_cases hj : 1 ∈ J
    · rw [ite_eq_left hj, hi₁]
      exact max_eq_left (hx.1 1 hj)
    · rw [ite_eq_right hj, hi₁]
      exact max_eq_right (hx.2 1 hj)

/-- Comparing the continued original branch with the original ReLU
field after reflection leaves exactly the first projected neuron.
Source: arXiv:2601.21366v2, §5.2, `eq:toggle-diff-general`,
and §6, the original matrix field in `rem:unifiedlog-B`. -/
theorem energyGradMap_sub_relu_at_neuron_reflection
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2)
    (h₀ : U (a 0) = -a 0) (h₁ : U (a 1) = a 1)
    (J : Finset (Idx 2)) (x : EucSpace 2) (hx : IsActivationPattern a J x) :
    energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (U x) -
        energyGradMap B (fun s => max s 0) ω a μ (U x) =
      (signedBranchWeight ω J 0 * inner (𝕜 := ℝ) (a 0) (U x)) • proj 2 (U x) (a 0) := by
  obtain ⟨hm₀, hm₁⟩ := relu_inputs_at_neuron_reflection a U h₀ h₁ J x hx
  have hsum : (∑ j : Idx 2,
        (branchMaskedWeights ω J j * inner (𝕜 := ℝ) (a j) (U x)) • a j) -
      (∑ j : Idx 2, (ω j * max (inner (𝕜 := ℝ) (a j) (U x)) 0) • a j) =
        (signedBranchWeight ω J 0 * inner (𝕜 := ℝ) (a 0) (U x)) • a 0 := by
    simp only [Fin.sum_univ_two, hm₀, hm₁, branchMaskedWeights, signedBranchWeight]
    by_cases hj₀ : 0 ∈ J <;> by_cases hj₁ : 1 ∈ J <;>
      simp [hj₀, hj₁, neg_mul]
  rw [energyGradMap_eq_attentionGradMap_add_drift,
    energyGradMap_eq_attentionGradMap_add_drift]
  calc
    _ = drift (fun s => s) (branchMaskedWeights ω J) a (U x) -
        drift (fun s => max s 0) ω a (U x) := by abel
    _ = proj 2 (U x) ((∑ j : Idx 2,
        (branchMaskedWeights ω J j * inner (𝕜 := ℝ) (a j) (U x)) • a j) -
      (∑ j : Idx 2, (ω j * max (inner (𝕜 := ℝ) (a j) (U x)) 0) • a j)) :=
        (proj_sub_right (U x) _ _).symm
    _ = _ := by rw [hsum, proj_smul_right]

/-- A globally zero original branch makes the original probability
invariant under a commuting neuron reflection.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`, with the
explicit additional commutation premise for this individual reflection. -/
theorem map_neuron_reflection_eq_of_zero_branch
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2))
    (hzero : ∀ x : SSphere 2, energyGradMap B (fun s => s)
      (branchMaskedWeights ω J) a μ (x : EucSpace 2) = 0)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (hcomm : ∀ y, B (U y) = U (B y))
    (h₀ : U (a 0) = -a 0) (h₁ : U (a 1) = a 1) :
    (μ : Measure (SSphere 2)).map (Perspective.sphereMap 2 U) = (μ : Measure (SSphere 2)) := by
  apply map_isometry_eq_of_transform_invariant B hB U hcomm μ
  apply transformMap_invariant_of_zero_branch_isometric_signs le_rfl B ω a μ J hzero U ![-1, 1]
  · intro j
    fin_cases j <;> norm_num
  · intro j
    fin_cases j
    · simpa using h₀
    · simpa using h₁

/-- Actual nonzero coordinate neurons, identity attention and its
uniform stationary law jointly satisfy the reflection, commutation,
invertibility, actual weak-pattern and globally zero branch premises.
Zero original weights allow the genuine zero-branch hypothesis without
assuming away either original neuron or either activation boundary. -/
example : ∃ μ : Perspective.ProbSphere 2,
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    (∀ y : EucSpace 2, LinearMap.id (R := ℝ) (circleCoordinateReflection y) =
      circleCoordinateReflection (LinearMap.id (R := ℝ) y)) ∧
    circleCoordinateReflection (basePoint 1 : EucSpace 2) = -(basePoint 1 : EucSpace 2) ∧
    circleCoordinateReflection secondAxis = secondAxis ∧
    IsActivationPattern ![(basePoint 1 : EucSpace 2), secondAxis] Finset.univ
      (basePoint 1 : EucSpace 2) ∧
    (∀ x : SSphere 2, energyGradMap LinearMap.id (fun s => s)
      (branchMaskedWeights (0 : Idx 2 → ℝ) Finset.univ)
      ![(basePoint 1 : EucSpace 2), secondAxis] μ (x : EucSpace 2) = 0) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hfull := support_eq_univ_of_uniform μ hν
  refine ⟨μ, Function.bijective_id, fun _ => rfl,
    circleCoordinateReflection_first, circleCoordinateReflection_second, ?_, ?_⟩
  · constructor
    · intro j hj
      fin_cases j <;>
        simp [basePoint, secondAxis, EuclideanSpace.inner_single_left]
    · intro j hj
      exact False.elim (hj (Finset.mem_univ j))
  · intro x
    have hw : branchMaskedWeights (0 : Idx 2 → ℝ) Finset.univ = 0 := by
      ext j
      simp [branchMaskedWeights]
    rw [hw]
    exact (isStationaryMap_id_iff (fun s => s) 0 _ μ).mpr
      (isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) _ μ hν) x
      (hfull.symm ▸ mem_univ x)

end Transformer.Perceptron
