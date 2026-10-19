module

public import Transformer.Perceptron.Section5_ConformalAtomicity

/-!
# Geometry of a nonzero scalar signed Gram map on the circle

ArXiv:2601.21366v2, §5.2, `thm: circle`, and §6,
`eq:ratio-const-correct`. The remaining scalar alternative involves
the sum of exactly two original signed rank-one neuron actions.
If this sum is a nonzero scalar identity, both neurons are nonzero,
their directions are orthogonal, and each signed weighted squared
norm equals the same original scalar.

The argument uses actual unit perpendicular vectors in the original
plane. A rank-one action cannot equal a nonzero scalar identity.
Applying the two-term identity at a perpendicular vector gives a
nonzero coefficient; taking its original inner product then forces
orthogonality. Applying the identity at each neuron gives its weight.
No condition is inserted into the original neuron definitions.
-/

@[expose] public section

open Real

namespace Transformer.Perceptron

/-- Every original plane vector has an actual perpendicular unit
vector, including the zero input. Source: arXiv:2601.21366v2, §5.2,
the circle tangent geometry used in `thm: circle`. -/
theorem exists_unit_perp_circle_vector (u : EucSpace 2) :
    ∃ w : EucSpace 2, ‖w‖ = 1 ∧ inner (𝕜 := ℝ) u w = 0 := by
  by_cases hu : u = 0
  · exact ⟨(basePoint 1 : EucSpace 2), norm_basePoint_one, by rw [hu, inner_zero_left]⟩
  have ht : circleTangent u ≠ 0 := by
    intro h
    have hn := norm_circleTangent u
    rw [h, norm_zero] at hn
    exact hu (norm_eq_zero.mp hn.symm)
  refine ⟨‖circleTangent u‖⁻¹ • circleTangent u, norm_smul_inv_norm ht, ?_⟩
  rw [real_inner_smul_right, inner_circleTangent, mul_zero]

/-- A single original rank-one action cannot be a nonzero scalar
identity on the plane. Source: arXiv:2601.21366v2, §6,
the rank-one terms of `eq: nondeg.normalized`. -/
theorem rankone_ne_nonzero_scalar_circle (v : EucSpace 2) (B c : ℝ) (hc : c ≠ 0) :
    ¬ ∀ x : EucSpace 2, (B * inner (𝕜 := ℝ) v x) • v = c • x := by
  intro h
  obtain ⟨w, hwn, hvw⟩ := exists_unit_perp_circle_vector v
  have hw : w ≠ 0 := fun h => zero_ne_one (by rw [h, norm_zero] at hwn; exact hwn)
  have hz := h w
  rw [hvw, mul_zero, zero_smul] at hz
  exact hc ((smul_eq_zero.mp hz.symm).resolve_right hw)

/-- The original directions are orthogonal when their two signed
rank-one actions sum to a nonzero scalar identity.
Source: arXiv:2601.21366v2, §6, `eq:ratio-const-correct`, nonzero scalar. -/
theorem inner_eq_zero_of_two_rankone_scalar (u v : EucSpace 2) (A B c : ℝ)
    (hc : c ≠ 0) (h : ∀ x, (A * inner (𝕜 := ℝ) u x) • u +
      (B * inner (𝕜 := ℝ) v x) • v = c • x) : inner (𝕜 := ℝ) u v = 0 := by
  obtain ⟨w, hwn, huw⟩ := exists_unit_perp_circle_vector u
  have hw : w ≠ 0 := fun h => zero_ne_one (by rw [h, norm_zero] at hwn; exact hwn)
  have hz := h w
  rw [huw, mul_zero, zero_smul, zero_add] at hz
  have hB : B * inner (𝕜 := ℝ) v w ≠ 0 := by
    intro hzero
    rw [hzero, zero_smul] at hz
    exact hc ((smul_eq_zero.mp hz.symm).resolve_right hw)
  have hi := congrArg (fun z => inner (𝕜 := ℝ) u z) hz
  rw [real_inner_smul_right, real_inner_smul_right, huw, mul_zero] at hi
  exact (mul_eq_zero.mp hi).resolve_left hB

/-- Both original neuron directions are nonzero in the two-term
nonzero scalar case. Source: arXiv:2601.21366v2, §6,
`eq:ratio-const-correct`, nonzero scalar. -/
theorem directions_ne_zero_of_two_rankone_scalar (u v : EucSpace 2) (A B c : ℝ)
    (hc : c ≠ 0) (h : ∀ x, (A * inner (𝕜 := ℝ) u x) • u +
      (B * inner (𝕜 := ℝ) v x) • v = c • x) : u ≠ 0 ∧ v ≠ 0 := by
  constructor
  · intro hu
    apply rankone_ne_nonzero_scalar_circle v B c hc
    exact fun x => by simpa only [hu, inner_zero_left, mul_zero, smul_zero, zero_add] using h x
  · intro hv
    apply rankone_ne_nonzero_scalar_circle u A c hc
    exact fun x => by simpa only [hv, inner_zero_left, mul_zero, smul_zero, add_zero] using h x

/-- Each original signed coefficient times its actual squared neuron
norm equals the same scalar. Source: arXiv:2601.21366v2, §6,
`eq:ratio-const-correct`, nonzero scalar. -/
theorem weighted_norms_eq_of_two_rankone_scalar (u v : EucSpace 2) (A B c : ℝ)
    (hc : c ≠ 0) (h : ∀ x, (A * inner (𝕜 := ℝ) u x) • u +
      (B * inner (𝕜 := ℝ) v x) • v = c • x) :
    A * ‖u‖ ^ 2 = c ∧ B * ‖v‖ ^ 2 = c := by
  have huv := inner_eq_zero_of_two_rankone_scalar u v A B c hc h
  have hvu : inner (𝕜 := ℝ) v u = 0 := (real_inner_comm u v).trans huv
  obtain ⟨hu, hv⟩ := directions_ne_zero_of_two_rankone_scalar u v A B c hc h
  have h₀ := h u
  rw [hvu, mul_zero, zero_smul, add_zero, real_inner_self_eq_norm_sq] at h₀
  have h₁ := h v
  rw [huv, mul_zero, zero_smul, zero_add, real_inner_self_eq_norm_sq] at h₁
  have hz₀ : (A * ‖u‖ ^ 2 - c) • u = 0 := by rw [sub_smul, h₀, sub_self]
  have hz₁ : (B * ‖v‖ ^ 2 - c) • v = 0 := by rw [sub_smul, h₁, sub_self]
  exact ⟨sub_eq_zero.mp ((smul_eq_zero.mp hz₀).resolve_right hu),
    sub_eq_zero.mp ((smul_eq_zero.mp hz₁).resolve_right hv)⟩

/-- The remaining nonzero scalar signed Gram alternative consists of
two nonzero orthogonal original neurons, with equal signed weighted
squared norms. Source: arXiv:2601.21366v2, §5.2 and §6,
`eq:ratio-const-correct`. The active-set signs remain the original ones. -/
theorem two_neuron_geometry_of_nonzero_scalar_signedGram (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2)) (c : ℝ) (hc : c ≠ 0)
    (hscalar : ∀ x, signedGram ω a J x = c • x) :
    inner (𝕜 := ℝ) (a 0) (a 1) = 0 ∧ a 0 ≠ 0 ∧ a 1 ≠ 0 ∧
      signedBranchWeight ω J 0 * ‖a 0‖ ^ 2 = c ∧
      signedBranchWeight ω J 1 * ‖a 1‖ ^ 2 = c ∧ (∀ j, ω j ≠ 0) := by
  have h (x : EucSpace 2) : (signedBranchWeight ω J 0 * inner (𝕜 := ℝ) (a 0) x) • a 0 +
      (signedBranchWeight ω J 1 * inner (𝕜 := ℝ) (a 1) x) • a 1 = c • x := by
    rw [← signedGram_two_eq]
    exact hscalar x
  have hi := inner_eq_zero_of_two_rankone_scalar _ _ _ _ c hc h
  obtain ⟨ha₀, ha₁⟩ := directions_ne_zero_of_two_rankone_scalar _ _ _ _ c hc h
  obtain ⟨hw₀, hw₁⟩ := weighted_norms_eq_of_two_rankone_scalar _ _ _ _ c hc h
  refine ⟨hi, ha₀, ha₁, hw₀, hw₁, ?_⟩
  intro j hj
  have hs : signedBranchWeight ω J j = 0 := (signedBranchWeight_eq_zero_iff ω J j).mpr hj
  fin_cases j
  · change signedBranchWeight ω J 0 = 0 at hs
    rw [hs, zero_mul] at hw₀
    exact hc hw₀.symm
  · change signedBranchWeight ω J 1 = 0 at hs
    rw [hs, zero_mul] at hw₁
    exact hc hw₁.symm

/-- The two-term scalar hypotheses are jointly realized by the actual
orthonormal coordinate axes and the original nonzero scalar one. -/
example : (1 : ℝ) ≠ 0 ∧ ∀ x : EucSpace 2,
    ((1 : ℝ) * inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) x) • (basePoint 1 : EucSpace 2) +
      ((1 : ℝ) * inner (𝕜 := ℝ) secondAxis x) • secondAxis = (1 : ℝ) • x := by
  refine ⟨one_ne_zero, ?_⟩
  intro x
  ext j
  fin_cases j <;> simp [basePoint, secondAxis, EuclideanSpace.inner_single_left]

/-- The actual signed Gram hypotheses hold for two original coordinate
neurons, unit weights and the original full activation index set. -/
example : (1 : ℝ) ≠ 0 ∧ ∀ x : EucSpace 2,
    signedGram (fun _ : Idx 2 => (1 : ℝ)) ![(basePoint 1 : EucSpace 2), secondAxis]
      Finset.univ x = (1 : ℝ) • x := by
  refine ⟨one_ne_zero, ?_⟩
  intro x
  rw [signedGram_two_eq]
  simp only [signedBranchWeight, Finset.mem_univ, ite_true, Matrix.cons_val_zero, Matrix.cons_val_one,
    one_mul, one_smul]
  ext j
  fin_cases j <;> simp [basePoint, secondAxis, EuclideanSpace.inner_single_left]

end Transformer.Perceptron
