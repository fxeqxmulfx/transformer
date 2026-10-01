/-
# Counterexample to the source clustering hypotheses

An antipodal stationary solution refutes the uncorrected form of
arXiv:2510.22026v2, §4.3, Theorem 4.3(ii). Smallness independent of the
inverse temperature is necessary; see `clustering_rate` for the correction.
-/

import Transformer.Normalization.Velocities

open scoped BigOperators
open Real

namespace Transformer.Normalization

variable (d n : ℕ)

/-- With `Q = K = V = I_d`, tokens on one line `ℝ u` have attention vectors on
that line. Source: arXiv:2510.22026v2, equation (NA). -/
theorem attentionVec_id_smul (β : ℝ) (u : EucSpace d) (s : Idx n → ℝ) (j : Idx n) :
    ∃ c : ℝ, attentionVec d n β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun k => s k • u) j = c • u := by
  simp only [attentionVec, ContinuousLinearMap.id_apply]
  simp_rw [smul_smul]
  rw [← Finset.sum_smul, smul_smul]
  exact ⟨_, rfl⟩

/-- The tangent projection at `±u` kills the line `ℝ u`.
Source: arXiv:2510.22026v2, equation (NA). -/
theorem proj_smul_self (u : EucSpace d) (hu : ‖u‖ = 1) (s c : ℝ) (hs : s ^ 2 = 1) :
    proj d (s • u) (c • u) = 0 := by
  unfold proj
  rw [inner_smul_left, inner_smul_right, real_inner_self_eq_norm_sq, hu, smul_smul]
  simp only [conj_trivial, one_pow, mul_one]
  rw [show s * c * s = c * s ^ 2 by ring, hs, mul_one, sub_self]

/-- A unit vector and `s=1` satisfy the projection lemma's hypotheses. -/
example : ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 ∧ (1 : ℝ) ^ 2 = 1 := by simp

/-- The signs `±1` of the antipodal counterexample to arXiv:2510.22026v2,
§4.3, `thm: preln-slow (ii)`. -/
def pairSign : Idx 2 → ℝ := ![1, -1]

/-- Both signs have unit square. Source: arXiv:2510.22026v2, §4.3,
the antipodal counterexample to `thm: preln-slow (ii)`. -/
theorem pairSign_sq (j : Idx 2) : pairSign j ^ 2 = 1 := by
  fin_cases j <;> simp [pairSign]

/-- **`thm: preln-slow (ii)` is false under the source's hypotheses.**

At `β = 1/100`, `n = 2` the condition `δ < 1/(100 n² β²)` admits `δ = 5/2`,
which every pair of unit vectors meets.  The antipodal pair `u, -u` with unit
radii is then a stationary Post-LN solution, `Var ≡ 1`, and no `c > 0` gives
`Var' ≤ -c Var` at any time.  In every dimension `d ≥ 1`, for every `α, τ`.

Source: arXiv:2510.22026v2, §4.3, `thm: preln-slow` (ii). -/
theorem not_clustering_rate (hd : 0 < d) (α : ℝ → ℝ) (τ : ℝ) :
    ¬ ∀ β δ : ℝ, δ < 1 / (100 * ((2 : ℕ) : ℝ) ^ 2 * β ^ 2) →
      ∀ Q K : ℝ → ParamMatrix d,
        (∀ t : ℝ, ∀ x y : EucSpace d,
          |inner (𝕜 := ℝ) (Q t x) (K t y)| ≤ ‖x‖ * ‖y‖) →
        ∃ c C : ℝ, 0 < c ∧ c ≤ C ∧
          ∀ (θ : ℝ → Idx 2 → EucSpace d) (r : ℝ → Idx 2 → ℝ),
            (∀ j : Idx 2, ‖θ 0 j‖ = 1) → (∀ j : Idx 2, 0 < r 0 j) →
            (∀ j k : Idx 2, 1 - δ ≤ inner (𝕜 := ℝ) (θ 0 j) (θ 0 k)) →
            SchemeDynamics d 2 β Q K (idParams d) α τ .post θ r →
              ∃ T₀ : ℝ, ∀ t : ℝ, T₀ ≤ t → ∃ v : ℝ,
                HasDerivAt (intraClusterVar d 2 θ) v t ∧
                -(C * intraClusterVar d 2 θ t / varScale α .post t) ≤ v ∧
                v ≤ -(c * intraClusterVar d 2 θ t / varScale α .post t) := by
  intro h
  set u : EucSpace d := EuclideanSpace.single (⟨0, hd⟩ : Fin d) (1 : ℝ) with hu_def
  have hu : ‖u‖ = 1 := by simp [hu_def]
  obtain ⟨c, C, hc, -, hall⟩ := h (1 / 100) (5 / 2) (by norm_num) (idParams d) (idParams d)
    (fun _ x y => abs_real_inner_le_norm x y)
  have hvar : intraClusterVar d 2 (fun _ k => pairSign k • u) = fun _ => 1 := by
    funext t
    simp [intraClusterVar, Fin.sum_univ_two, pairSign, hu]
    norm_num
  have hdyn : SchemeDynamics d 2 (1 / 100) (idParams d) (idParams d) (idParams d) α τ .post
      (fun _ k => pairSign k • u) (fun _ _ => 1) := by
    refine fun t _ j => ⟨?_, (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt⟩
    obtain ⟨a, ha⟩ := attentionVec_id_smul d 2 (1 / 100) u pairSign j
    have hzero : proj d (pairSign j • u) (attentionVec d 2 (1 / 100) (idParams d t)
        (idParams d t) (idParams d t) (fun k => pairSign k • u) j) = 0 := by
      rw [show idParams d t = ContinuousLinearMap.id ℝ (EucSpace d) from rfl, ha]
      exact proj_smul_self d u hu _ a (pairSign_sq j)
    rw [hzero, smul_zero]
    exact (hasDerivAt_const t _).hasDerivWithinAt
  obtain ⟨T₀, hT⟩ := hall (fun _ k => pairSign k • u) (fun _ _ => 1)
    (fun j => by fin_cases j <;> simp [pairSign, hu])
    (fun _ => one_pos)
    (fun j k => by
      fin_cases j <;> fin_cases k <;>
        simp [pairSign, hu] <;>
        norm_num)
    hdyn
  obtain ⟨v, hv, -, hle⟩ := hT T₀ le_rfl
  rw [hvar] at hv hle
  have hv0 : v = 0 := hv.unique (hasDerivAt_const T₀ (1 : ℝ))
  simp only [varScale, hv0] at hle
  linarith

/-- Dimension one satisfies the counterexample's dimension hypothesis. -/
example : 0 < 1 := Nat.one_pos

end Transformer.Normalization
