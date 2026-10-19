module

public import Transformer.Normalization.Section2_RadialInvariant

/-!
# Normalization — Pre-LN and Peri-LN preserve the sphere

The direction variables in arXiv:2510.22026v2, §2, equation (NA), lie on
the unit sphere. This constraint follows from the ambient equations in
`SchemeDynamics` and unit initial data; it is not added as a path hypothesis.

The radial invariant is zero. At a point with squared norm different from
one it forces the radius to vanish, hence the angular velocity vanishes.
At a point with squared norm one the projection is tangent. Thus the squared
norm has zero derivative everywhere on the forward half-line and is constant.
This argument also covers the zero-radius branch of the formal equations.

The singleton identity model supplies actual solutions of both schemes:
the direction is fixed and the radius grows at rate one.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Normalization

/-- With one token, the positive softmax weight cancels its denominator.
Source: arXiv:2510.22026v2, §2, the definition of `A_j`. -/
theorem attentionVec_singleton {d : ℕ} (β : ℝ) (Q K V : ParamMatrix d)
    (Θ : Idx 1 → EucSpace d) (j : Idx 1) :
    attentionVec d 1 β Q K V Θ j = V (Θ j) := by
  rw [Fin.fin_one_eq_zero j]
  simp only [attentionVec, Fin.sum_univ_one, smul_smul]
  rw [inv_mul_cancel₀ (Real.exp_ne_zero _), one_smul]

/-- For the identity parameters, one unit token has fixed direction and
radius `r₀ + t` in both Pre-LN and Peri-LN. The radial derivative is one,
and the projected attention is zero.
Source: arXiv:2510.22026v2, §2, equation (NA), Table 2. -/
theorem schemeDynamics_singleton {d : ℕ} (β r₀ τ : ℝ) (α : ℝ → ℝ)
    (x : EucSpace d) (hx : ‖x‖ = 1) (scheme : Scheme)
    (hscheme : scheme = .pre ∨ scheme = .peri) :
    SchemeDynamics d 1 β (idParams d) (idParams d) (idParams d) α τ scheme
      (fun _ _ => x) (fun t _ => r₀ + t) := by
  intro t _ j
  have hA : attentionVec d 1 β (idParams d t) (idParams d t) (idParams d t)
      (fun _ => x) j = x := by
    simpa only [idParams, ContinuousLinearMap.id_apply] using
      attentionVec_singleton β (idParams d t) (idParams d t) (idParams d t) (fun _ => x) j
  have hp : proj d x x = 0 := by
    simpa only [one_smul] using Transformer.proj_smul_self hx (1 : ℝ)
  constructor
  · rw [hA, hp, smul_zero]
    exact hasDerivWithinAt_const _ _ _
  · rcases hscheme with rfl | rfl <;>
      simpa [radialDerivative, hA, real_inner_self_eq_norm_sq, hx] using
        ((hasDerivAt_id t).const_add r₀).hasDerivWithinAt

/-- A unit base point and the Pre-LN scheme meet all singleton hypotheses. -/
example (d : ℕ) : ‖(basePoint d : EucSpace (d + 1))‖ = 1 ∧
    (Scheme.pre = .pre ∨ Scheme.pre = .peri) :=
  ⟨by simp [basePoint, PiLp.norm_single], Or.inl rfl⟩

/-- The squared norm has derivative zero along either actual scheme.
The radial invariant handles non-unit points, including zero radii.
Source: arXiv:2510.22026v2, §2, equation (NA), Table 2. -/
theorem SchemeDynamics.hasDerivWithinAt_norm_sq_zero {d n : ℕ}
    {β τ : ℝ} {Q K V : ℝ → ParamMatrix d} {α : ℝ → ℝ} {scheme : Scheme}
    {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}
    (hD : SchemeDynamics d n β Q K V α τ scheme θ r)
    (hscheme : scheme = .pre ∨ scheme = .peri) (h0 : ∀ j, ‖θ 0 j‖ = 1)
    (t : ℝ) (ht : 0 ≤ t) (j : Idx n) :
    HasDerivWithinAt (fun u => ‖θ u j‖ ^ 2) 0 (Ici 0) t := by
  have hdefect := hD.radial_norm_defect_zero hscheme h0 t ht j
  by_cases hnorm : ‖θ t j‖ ^ 2 = 1
  · convert (hD t ht j).1.norm_sq using 1
    simp only [real_inner_smul_right, proj, inner_sub_right, real_inner_self_eq_norm_sq,
      hnorm, mul_one, sub_self, mul_zero]
  · have hrzero : r t j = 0 := sq_eq_zero_iff.mp
      ((mul_eq_zero.mp hdefect).resolve_right (sub_ne_zero.mpr hnorm))
    rcases hscheme with rfl | rfl <;>
      simpa [speedFactor, hrzero] using (hD t ht j).1.norm_sq

/-- The singleton solution meets every derivative-zero hypothesis. -/
example (d : ℕ) :
    SchemeDynamics (d + 1) 1 1 (idParams (d + 1)) (idParams (d + 1)) (idParams (d + 1))
      0 0 .pre (fun _ _ => (basePoint d : EucSpace (d + 1))) (fun t _ => 1 + t) ∧
    (Scheme.pre = .pre ∨ Scheme.pre = .peri) ∧
    (∀ j : Idx 1, ‖(fun _ _ => (basePoint d : EucSpace (d + 1))) (0 : ℝ) j‖ = 1) ∧
    (0 : ℝ) ≤ 1 := by
  have hx : ‖(basePoint d : EucSpace (d + 1))‖ = 1 := by
    simp [basePoint, PiLp.norm_single]
  exact ⟨schemeDynamics_singleton 1 1 0 0 _ hx .pre (Or.inl rfl),
    Or.inl rfl, fun _ => hx, zero_le_one⟩

/-- Unit initial directions remain unit for all forward times, without any
assumption that the radii or the speed factors stay away from zero.
Source: arXiv:2510.22026v2, §2, equation (NA) and Table 2. -/
theorem SchemeDynamics.norm_eq_one {d n : ℕ}
    {β τ : ℝ} {Q K V : ℝ → ParamMatrix d} {α : ℝ → ℝ} {scheme : Scheme}
    {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}
    (hD : SchemeDynamics d n β Q K V α τ scheme θ r)
    (hscheme : scheme = .pre ∨ scheme = .peri) (h0 : ∀ j, ‖θ 0 j‖ = 1)
    (t : ℝ) (ht : 0 ≤ t) (j : Idx n) : ‖θ t j‖ = 1 := by
  have hsq := eq_initial_of_hasDerivWithinAt_zero (fun u => ‖θ u j‖ ^ 2)
    (fun u hu => hD.hasDerivWithinAt_norm_sq_zero hscheme h0 u hu j) t ht
  rw [h0 j] at hsq
  nlinarith [norm_nonneg (θ t j)]

/-- Peri-LN has the same realizable initial and time conditions. -/
example (d : ℕ) :
    SchemeDynamics (d + 1) 1 1 (idParams (d + 1)) (idParams (d + 1)) (idParams (d + 1))
      0 0 .peri (fun _ _ => (basePoint d : EucSpace (d + 1))) (fun t _ => 1 + t) ∧
    (Scheme.peri = .pre ∨ Scheme.peri = .peri) ∧
    (∀ j : Idx 1, ‖(fun _ _ => (basePoint d : EucSpace (d + 1))) (0 : ℝ) j‖ = 1) ∧
    (0 : ℝ) ≤ 1 := by
  have hx : ‖(basePoint d : EucSpace (d + 1))‖ = 1 := by
    simp [basePoint, PiLp.norm_single]
  exact ⟨schemeDynamics_singleton 1 1 0 0 _ hx .peri (Or.inr rfl),
    Or.inr rfl, fun _ => hx, zero_le_one⟩

/-- Normalizing nonzero initial tokens supplies the unit initial condition
of the sphere-invariance theorem. The sampled law is irrelevant for this
deterministic implication.
Source: arXiv:2510.22026v2, §3, the initialization of `thm:convergence`. -/
theorem SchemeDynamics.norm_eq_one_of_normalized_init {d n : ℕ}
    {β τ : ℝ} {Q K V : ℝ → ParamMatrix d} {α : ℝ → ℝ} {scheme : Scheme}
    {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}
    (hD : SchemeDynamics d n β Q K V α τ scheme θ r)
    (hscheme : scheme = .pre ∨ scheme = .peri) (X₀ : Idx n → EucSpace d)
    (hX₀ : ∀ j, X₀ j ≠ 0) (hθ₀ : ∀ j, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j)
    (t : ℝ) (ht : 0 ≤ t) (j : Idx n) : ‖θ t j‖ = 1 := by
  apply hD.norm_eq_one hscheme ?_ t ht j
  intro i
  rw [hθ₀ i]
  exact norm_smul_inv_norm (hX₀ i)

/-- A base-point initial token witnesses the complete normalization input. -/
example (d : ℕ) :
    SchemeDynamics (d + 1) 1 1 (idParams (d + 1)) (idParams (d + 1)) (idParams (d + 1))
      0 0 .pre (fun _ _ => (basePoint d : EucSpace (d + 1))) (fun t _ => 1 + t) ∧
    (Scheme.pre = .pre ∨ Scheme.pre = .peri) ∧
    (∀ j : Idx 1, (fun _ => (basePoint d : EucSpace (d + 1))) j ≠ 0) ∧
    (∀ j : Idx 1, (fun _ => (basePoint d : EucSpace (d + 1))) j =
      ‖(fun _ => (basePoint d : EucSpace (d + 1))) j‖⁻¹ •
        (fun _ => (basePoint d : EucSpace (d + 1))) j) ∧
    (0 : ℝ) ≤ 1 := by
  have hx : ‖(basePoint d : EucSpace (d + 1))‖ = 1 := by
    simp [basePoint, PiLp.norm_single]
  refine ⟨schemeDynamics_singleton 1 1 0 0 _ hx .pre (Or.inl rfl), Or.inl rfl,
    fun _ => norm_pos_iff.mp (by rw [hx]; exact one_pos), fun _ => ?_, zero_le_one⟩
  simp [hx]

end Transformer.Normalization
