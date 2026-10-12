/-
# The signed quadratic obstruction in normalized stationarity

ArXiv:2601.21366v2, §6, `eq: nondeg.normalized` and
`eq:ratio-const-correct`. The actual signed Gram operator is linear
and symmetric. If its quadratic form is constant on the unit sphere,
homogeneity and polarization make the operator a scalar identity.

The quadratic form below is computed from the original neurons and
weights. Along the actual circle it is exactly twice the difference
of the two complementary half potentials, retaining the source's
normalization. Thus non-degeneracy rules out a constant difference.
-/

import Transformer.Perceptron.Section6_BranchCalculus
import Mathlib.Analysis.InnerProductSpace.Symmetric

open Real
open scoped BigOperators

namespace Transformer.Perceptron

/-- The source's signed Gram matrix as an actual linear operator.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`.
Each rank-one summand sends `x` to `ω⟪a,x⟫a`. -/
noncomputable def signedGramLinearMap {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) : EucSpace d →ₗ[ℝ] EucSpace d :=
  (∑ j ∈ J, ω j • ((innerSL ℝ (a j)).toLinearMap.smulRight (a j))) -
    ∑ j ∈ Jᶜ, ω j • ((innerSL ℝ (a j)).toLinearMap.smulRight (a j))

/-- The bundled operator is the original signed Gram action.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`.
No scalar-identity or quadratic-form property is placed in its definition. -/
theorem signedGramLinearMap_apply {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    signedGramLinearMap ω a J x = signedGram ω a J x := by
  simp [signedGramLinearMap, signedGram, LinearMap.sum_apply,
    LinearMap.smulRight_apply, smul_smul]

/-- The original signed Gram action is symmetric for arbitrary real
weights. Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`. -/
theorem isSymmetric_signedGramLinearMap {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) :
    (signedGramLinearMap ω a J).IsSymmetric := by
  intro x y
  have hsum (K : Finset (Idx d)) :
      inner (𝕜 := ℝ) (∑ j ∈ K, (ω j * inner (𝕜 := ℝ) (a j) x) • a j) y =
        inner (𝕜 := ℝ) x (∑ j ∈ K, (ω j * inner (𝕜 := ℝ) (a j) y) • a j) := by
    rw [sum_inner, inner_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [real_inner_smul_left, real_inner_smul_right, real_inner_comm (a j) x]
    ring
  simp only [signedGramLinearMap_apply, signedGram, inner_sub_left, inner_sub_right]
  rw [hsum J, hsum Jᶜ]

/-- The actual quadratic form is the signed sum of squared neuron
inner products. Source: arXiv:2601.21366v2, §6,
the left-hand side of `eq:ratio-const-correct`. -/
theorem inner_signedGram_self {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (J : Finset (Idx d)) (x : EucSpace d) :
    inner (𝕜 := ℝ) (signedGram ω a J x) x =
      (∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) x ^ 2) -
        ∑ j ∈ Jᶜ, ω j * inner (𝕜 := ℝ) (a j) x ^ 2 := by
  simp only [signedGram, inner_sub_left, sum_inner, real_inner_smul_left, pow_two, mul_assoc]

/-- A symmetric operator with constant unit-sphere quadratic form is a
scalar identity. Source: arXiv:2601.21366v2, §6, the final inference
from `eq:ratio-const-correct`. The proof extends from unit vectors by
homogeneity and uses polarization of the actual symmetric operator. -/
theorem eq_smul_id_of_constant_unit_quadratic {d : ℕ}
    (T : EucSpace d →ₗ[ℝ] EucSpace d) (hT : T.IsSymmetric) (c : ℝ)
    (hc : ∀ x : EucSpace d, ‖x‖ = 1 → inner (𝕜 := ℝ) (T x) x = c) :
    T = c • LinearMap.id := by
  have hR : (T - c • LinearMap.id).IsSymmetric :=
    hT.sub (LinearMap.IsSymmetric.id.smul (by simp))
  apply sub_eq_zero.mp
  apply hR.inner_map_self_eq_zero.mp
  intro x
  by_cases hx : x = 0
  · simp [hx]
  have hn : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hu : ‖‖x‖⁻¹ • x‖ = 1 := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (inv_nonneg.mpr (norm_nonneg x)),
      inv_mul_cancel₀ hn]
  have h := hc (‖x‖⁻¹ • x) hu
  simp only [map_smul, real_inner_smul_left, real_inner_smul_right] at h
  have hq : inner (𝕜 := ℝ) (T x) x = c * ‖x‖ ^ 2 := by
    field_simp at h
    nlinarith
  rw [LinearMap.sub_apply, LinearMap.smul_apply, LinearMap.id_apply,
    inner_sub_left, real_inner_smul_left, real_inner_self_eq_norm_sq, hq, sub_self]

/-- The symmetry and constant-quadratic-form inputs hold jointly for
the actual nonzero identity on the two-dimensional space.
Source: arXiv:2601.21366v2, §6, the scalar-identity alternative. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    (∀ x : EucSpace 2, ‖x‖ = 1 →
      inner (𝕜 := ℝ) (LinearMap.id (R := ℝ) (M := EucSpace 2) x) x = 1) := by
  refine ⟨LinearMap.IsSymmetric.id, ?_⟩
  intro x hx
  rw [LinearMap.id_apply, real_inner_self_eq_norm_sq, hx, one_pow]

/-- Non-degeneracy excludes a constant actual signed quadratic form
on the unit sphere. Source: arXiv:2601.21366v2, §6,
`eq: nondeg.normalized` and `eq:ratio-const-correct`. -/
theorem not_constant_signedGram_of_nonDegenerate {d : ℕ} (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (hnd : IsNonDegenerate ω a) (J : Finset (Idx d)) :
    ¬ ∃ c : ℝ, ∀ x : EucSpace d, ‖x‖ = 1 →
      inner (𝕜 := ℝ) (signedGram ω a J x) x = c := by
  rintro ⟨c, hc⟩
  have hT := eq_smul_id_of_constant_unit_quadratic (signedGramLinearMap ω a J)
    (isSymmetric_signedGramLinearMap ω a J) c (by simpa only [signedGramLinearMap_apply] using hc)
  apply hnd J
  refine ⟨c, fun x => ?_⟩
  simpa only [signedGramLinearMap_apply, LinearMap.smul_apply, LinearMap.id_apply] using
    congrArg (fun L : EucSpace d →ₗ[ℝ] EucSpace d => L x) hT

/-- The complementary branch difference is exactly half the actual
signed Gram quadratic form. Source: arXiv:2601.21366v2, §6,
`eq:ratio-const-correct`, with the half potentials used in `H_log`. -/
theorem circleBranchHalfPotential_sub_compl (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) (θ : ℝ) :
    circleBranchHalfPotential ω a J θ - circleBranchHalfPotential ω a Jᶜ θ =
      (1 / 2 : ℝ) * inner (𝕜 := ℝ)
        (signedGram ω a J (circlePoint θ : EucSpace 2)) (circlePoint θ : EucSpace 2) := by
  rw [inner_signedGram_self]
  unfold circleBranchHalfPotential
  ring

/-- Non-degeneracy excludes a constant complementary branch difference
on the actual circle. Source: arXiv:2601.21366v2, §6,
the contradiction from `eq:ratio-const-correct` in part (b).
Every unit vector is an actual point of the circle chart. -/
theorem not_constant_circleBranchDifference (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a) (J : Finset (Idx 2)) :
    ¬ ∃ c : ℝ, ∀ θ, circleBranchHalfPotential ω a J θ -
      circleBranchHalfPotential ω a Jᶜ θ = c := by
  rintro ⟨c, hc⟩
  apply not_constant_signedGram_of_nonDegenerate ω a hnd J
  refine ⟨2 * c, ?_⟩
  intro x hx
  let xp : SSphere 2 := ⟨x, by simpa only [Metric.mem_sphere, dist_zero_right] using hx⟩
  obtain ⟨θ, -, heq⟩ := exists_circlePoint_eq xp
  have he : (circlePoint θ : EucSpace 2) = x := congrArg Subtype.val heq
  have h := hc θ
  rw [circleBranchHalfPotential_sub_compl, he] at h
  linarith

/-- The non-degeneracy input is realized by the source's actual pinned
one-neuron perceptron, including its second zero neuron.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`. -/
example : IsNonDegenerate (Pi.single 0 1)
    (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) := isNonDegenerate_pin

end Transformer.Perceptron
