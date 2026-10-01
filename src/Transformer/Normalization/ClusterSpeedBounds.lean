/-
# Comparing normalization speed factors to their asymptotic scales

Table 2 and Appendix C, proof of Theorem 4.3 of arXiv:2510.22026v2.
The starting time absorbs the initial radii and the Mix-LN switching time.
-/

import Transformer.Normalization.ClusterRadii

open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- In a cluster of height at least `7/8`, the inverse speed factors are
eventually between `2/(3 scale)` and `4/(3 scale)` for every scheme.
Source: arXiv:2510.22026v2, Table 2 and Appendix C, the final speed estimates
in the proof of Theorem 4.3, with the corrected nGPT scale `α⁻¹`. -/
theorem eventual_inverse_speed_bounds (β τ : ℝ)
    (Q K : ℝ → ParamMatrix d) (α : ℝ → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ) (scheme : Scheme)
    (hα : ∀ t, 0 < α t) (hr : ∀ j, 0 < r 0 j)
    (hunit : ∀ t, 0 ≤ t → ∀ j, ‖θ t j‖ = 1)
    (hpair : ∀ t, 0 ≤ t → ∀ j k, (7 / 8 : ℝ) ≤ inner (𝕜 := ℝ) (θ t j) (θ t k))
    (hrmono : ∀ t, 0 ≤ t → ∀ j, r 0 j ≤ r t j)
    (hdyn : SchemeDynamics d n β Q K (idParams d) α τ scheme θ r) :
    ∃ T : ℝ, 1 ≤ T ∧ ∀ t, T ≤ t →
      0 < varScale α scheme t ∧ ∀ j,
        (2 / 3 : ℝ) * (varScale α scheme t)⁻¹ ≤
          (speedFactor d n β Q K (idParams d) α θ r τ scheme t j)⁻¹ ∧
        (speedFactor d n β Q K (idParams d) α θ r τ scheme t j)⁻¹ ≤
          (4 / 3 : ℝ) * (varScale α scheme t)⁻¹ := by
  let T : ℝ := max 1 (max (2 * ∑ j, r 0 j) (8 * (max τ 0 + 1)))
  have hT : 1 ≤ T := le_max_left _ _
  refine ⟨T, hT, ?_⟩
  intro t ht
  have ht1 : 1 ≤ t := hT.trans ht
  have ht0 : 0 ≤ t := by linarith
  have htpos : 0 < t := by linarith
  have htime : 8 * (max τ 0 + 1) ≤ t :=
    (le_max_right _ _).trans ((le_max_right _ _).trans ht)
  have hstart : max τ 0 + 1 ≤ t := by linarith [le_max_right τ 0]
  have hswitch : τ < t := by linarith [le_max_left τ 0]
  have hscale : 0 < varScale α scheme t := by
    cases scheme <;> simp only [varScale]
    · exact one_pos
    · exact htpos
    · exact htpos
    · exact htpos
    · exact inv_pos.mpr (hα t)
    · exact Real.sqrt_pos.mpr htpos
  refine ⟨hscale, ?_⟩
  intro j
  have hr0 : r 0 j ≤ t / 2 := by
    have hs : r 0 j ≤ ∑ k, r 0 k :=
      Finset.single_le_sum (fun k _ => (hr k).le) (Finset.mem_univ j)
    have htR : 2 * (∑ k, r 0 k) ≤ t :=
      (le_max_left _ _).trans ((le_max_right _ _).trans ht)
    linarith
  have hg := scheme_radius_growth β (7 / 8) τ (by norm_num) Q K α θ r scheme
    hunit hpair hrmono hdyn t ht0 j
  have hrupper : r t j ≤ (3 / 2 : ℝ) * t := by linarith [hg.1]
  let A := attentionVec d n β (Q t) (K t) (idParams d t) (θ t) j
  have hnorm1 : ‖A‖ ≤ 1 := norm_attentionVec_le_one d n β (Q t) (K t)
    (ContinuousLinearMap.id ℝ _) ContinuousLinearMap.norm_id_le (θ t) (hunit t ht0) j
  have hnorm : (7 / 8 : ℝ) ≤ ‖A‖ := by
    have hi := inner_attentionVec_id_ge β (7 / 8) (Q t) (K t) (θ t) (hpair t ht0) j
    have hcs := real_inner_le_norm (θ t j) A
    rw [hunit t ht0 j, one_mul] at hcs
    exact hi.trans hcs
  have hbounds : (3 / 4 : ℝ) * varScale α scheme t ≤
      speedFactor d n β Q K (idParams d) α θ r τ scheme t j ∧
      speedFactor d n β Q K (idParams d) α θ r τ scheme t j ≤
        (3 / 2 : ℝ) * varScale α scheme t := by
    cases scheme with
    | post => norm_num [speedFactor, varScale]
    | pre =>
      have hlo := hg.2.1 (Or.inl rfl)
      simp only [speedFactor, varScale]
      constructor <;> linarith [hr j]
    | mix =>
      have hlo := hg.2.2 ⟨rfl, hstart⟩
      simp only [speedFactor, varScale, ite_eq_right (not_le.mpr hswitch)]
      constructor <;> linarith [hr j]
    | peri =>
      have hlo := hg.2.1 (Or.inr rfl)
      have hrlo : (7 / 8 : ℝ) * t ≤ r t j := by linarith [hr j]
      change (3 / 4 : ℝ) * t ≤ r t j * ‖A‖ ∧ r t j * ‖A‖ ≤ (3 / 2 : ℝ) * t
      constructor
      · calc (3 / 4 : ℝ) * t ≤ ((7 / 8 : ℝ) * t) * (7 / 8) := by nlinarith
          _ ≤ r t j * ‖A‖ := mul_le_mul hrlo hnorm (by norm_num) (by linarith)
      · exact (mul_le_mul_of_nonneg_left hnorm1 (by linarith)).trans (by simpa using hrupper)
    | nGPT =>
      change (3 / 4 : ℝ) * (α t)⁻¹ ≤ (α t)⁻¹ * ‖A‖ ∧
        (α t)⁻¹ * ‖A‖ ≤ (3 / 2 : ℝ) * (α t)⁻¹
      have hinv := (inv_pos.mpr (hα t)).le
      constructor
      · have h := mul_le_mul_of_nonneg_left hnorm hinv
        nlinarith
      · have h := mul_le_mul_of_nonneg_left hnorm1 hinv
        nlinarith
    | CoD =>
      simp only [speedFactor, varScale]
      have hs := Real.sq_sqrt ht0
      have hs1 := Real.sq_sqrt (show 0 ≤ t + 1 by linarith)
      constructor <;> nlinarith [Real.sqrt_nonneg t, Real.sqrt_nonneg (t + 1)]
  have hspeed : 0 < speedFactor d n β Q K (idParams d) α θ r τ scheme t j :=
    lt_of_lt_of_le (mul_pos (by norm_num) hscale) hbounds.1
  constructor
  · rw [show (2 / 3 : ℝ) * (varScale α scheme t)⁻¹ = (2 / 3) / varScale α scheme t by ring,
      ← one_div, div_le_div_iff₀ hscale hspeed]
    nlinarith [hbounds.2]
  · rw [show (4 / 3 : ℝ) * (varScale α scheme t)⁻¹ = (4 / 3) / varScale α scheme t by ring,
      ← one_div, div_le_div_iff₀ hspeed hscale]
    nlinarith [hbounds.1]

/-- Constant unit directions and radii meet the global cluster assumptions
for a stationary Post-LN trajectory. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    let θ : ℝ → Idx 1 → EucSpace 1 := fun _ _ => u
    (∀ t : ℝ, 0 ≤ t → ∀ j, ‖θ t j‖ = 1) ∧
      (∀ t : ℝ, 0 ≤ t → ∀ j k, (7 / 8 : ℝ) ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) ∧
      SchemeDynamics 1 1 1 (idParams 1) (idParams 1) (idParams 1) (fun _ => 1) 0
        .post θ (fun _ _ => 1) := by
  dsimp
  refine ⟨fun _ _ _ => by simp, fun _ _ _ _ => by norm_num,
    fun t _ j => ⟨?_, ?_⟩⟩
  · simpa [speedFactor, attentionVec, idParams, proj] using
      (hasDerivAt_const t (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))).hasDerivWithinAt
  · exact (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt

end Transformer.Normalization
