/-
# Escape classification for actual original discrete multi-head trajectories

ArXiv:2305.05465v6, §12, `sec:conclusion`, and §7,
`l:auxiliary`, `l:unboundedparticles`, and the proof of `l:onlyone`.
The value sum and pull coefficient below are the explicit sums of the
original scalar head coefficients. Positivity follows from the source's
original matrix hypotheses and an actual head index.

The exact vector update commutes with sign reversal. Its proved scalar
coordinate equation gives the self-pull recurrence for each token and
its negative. The discrete sequence estimates therefore give one common
threshold: every original token either stays within it at every layer
or tends to positive or negative infinity. If the original data contain
two distinct tokens, the spreading estimate forces at least one of these
actual original coordinates to tend to infinity in one direction.
-/

import Transformer.Clusters.Section12_DiscreteEscape
import Transformer.Clusters.Section12_MultiHeadSpread

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {n m H : ℕ}

/-- The actual total scalar value coefficient of the original heads.
Source: arXiv:2305.05465v6, §12, the sum of the original value-weighted heads. -/
noncomputable def multiHeadValueCoefficient (V : Idx H → ParamMatrix 1) : ℝ :=
  ∑ h, (V h unit1) 0

/-- The explicit pull error coefficient of the original heads for n original tokens.
Source: arXiv:2305.05465v6, §7, `e:lowerboundxi`, summed over §12's separate heads. -/
noncomputable def multiHeadPullCoefficient (n : ℕ) (Q K V : Idx H → ParamMatrix 1) : ℝ :=
  n * ∑ h, (V h unit1) 0 / inner (𝕜 := ℝ) (Q h unit1) (K h unit1)

/-- The actual original coefficients have the signs needed for the discrete escape estimate.
Source: arXiv:2305.05465v6, §2, `t:boolean`, applied separately to the heads of §12. -/
theorem multiHead_coefficients_pos (n : ℕ) (Q K V : Idx H → ParamMatrix 1)
    (hV : ∀ h, IsPosDefOp (V h)) (hQK : ∀ h, IsPosDefQK (Q h) (K h)) (h : Idx H) :
    0 < multiHeadValueCoefficient V ∧ 0 ≤ multiHeadPullCoefficient n Q K V := by
  have hc := fun a => (scalar_head_parameters_pos _ _ _ (hV a) (hQK a)).1
  have hv := fun a => (scalar_head_parameters_pos _ _ _ (hV a) (hQK a)).2
  constructor
  · exact (hv h).trans_le (Finset.single_le_sum (fun a _ => (hv a).le) (Finset.mem_univ h))
  · exact mul_nonneg (Nat.cast_nonneg n) (Finset.sum_nonneg fun a _ => div_nonneg (hv a).le (hc a).le)

/-- Simultaneous token sign reversal preserves the actual original vector iteration.
Source: arXiv:2305.05465v6, §7, proof of `l:auxiliary`, the involution, for §12's heads. -/
theorem multiHeadTransformer_neg {Δt : ℝ} {Q K V : Idx H → ParamMatrix 1}
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X) :
    MultiHeadTransformer Δt Q K V (fun k i => -X k i) := by
  apply (multiHeadTransformer_iff_coord Δt Q K V _).mpr
  intro k i
  change -(X (k + 1) i 0) = -(X k i 0) + Δt *
    multiHeadScalarDrift (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1))
      (fun h => (V h unit1) 0) (fun j => -(X k j 0)) (-(X k i 0))
  rw [multiHeadScalarDrift_neg, multiHeadTransformer_coord hX k i]
  ring

/-- Every positive original query has the proved pull toward any chosen original reference token.
Source: arXiv:2305.05465v6, §7, `e:lowerboundxi`, for the exact discrete head sum of §12. -/
theorem multiHeadTransformer_reference_pull {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (k : ℕ) (i M : Idx (m + 1)) (hi : 0 < X k i 0) :
    X k i 0 + Δt * (multiHeadValueCoefficient V * X k M 0 -
      multiHeadPullCoefficient (m + 1) Q K V / X k i 0) ≤ X (k + 1) i 0 := by
  rw [multiHeadTransformer_coord hX k i]
  apply add_le_add_right
  apply mul_le_mul_of_nonneg_left _ hΔt
  simp only [multiHeadValueCoefficient, multiHeadPullCoefficient, Nat.cast_add, Nat.cast_one]
  exact multiHeadScalarDrift_ge_reference _ _
    (fun h => (scalar_head_parameters_pos _ _ _ (hV h) (hQK h)).1)
    (fun h => (scalar_head_parameters_pos _ _ _ (hV h) (hQK h)).2.le)
    (fun j => X k j 0) (X k i 0) hi M

/-- Every positive original coordinate satisfies the proved self-pull recurrence.
Source: arXiv:2305.05465v6, §7, `e:lowerboundxi`, for the exact discrete head sum of §12. -/
theorem multiHeadTransformer_self_pull {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (k : ℕ) (i : Idx (m + 1)) (hi : 0 < X k i 0) :
    X k i 0 + Δt * (multiHeadValueCoefficient V * X k i 0 -
      multiHeadPullCoefficient (m + 1) Q K V / X k i 0) ≤ X (k + 1) i 0 := by
  exact multiHeadTransformer_reference_pull hΔt hV hQK hX k i i hi

/-- A common fixed threshold expands every original coordinate and its negative.
Source: arXiv:2305.05465v6, §7, `l:auxiliary`, for §12's original positive heads. -/
theorem multiHeadTransformer_growth_threshold {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) :
    ∃ R : ℝ, 0 < R ∧ ∀ i,
      (∀ k, R ≤ X k i 0 → (1 + Δt * multiHeadValueCoefficient V / 2) * X k i 0 ≤ X (k + 1) i 0) ∧
      (∀ k, R ≤ -X k i 0 → (1 + Δt * multiHeadValueCoefficient V / 2) * (-X k i 0) ≤ -X (k + 1) i 0) := by
  obtain ⟨hΛ, hB⟩ := multiHead_coefficients_pos (m + 1) Q K V hV hQK h
  obtain ⟨R, hR, hg⟩ := exists_discrete_self_growth_threshold Δt
    (multiHeadValueCoefficient V) (multiHeadPullCoefficient (m + 1) Q K V) hΔt hΛ hB
  refine ⟨R, hR, ?_⟩
  intro i
  constructor
  · intro k hk
    exact (hg _ hk).trans (multiHeadTransformer_self_pull hΔt.le hV hQK hX k i (hR.trans_le hk))
  · intro k hk
    exact (hg _ hk).trans (multiHeadTransformer_self_pull hΔt.le hV hQK
      (multiHeadTransformer_neg hX) k i (hR.trans_le hk))

/-- Every original token is uniformly bounded or tends to one of the two infinities.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, for §12's original discrete head sum. -/
theorem multiHeadTransformer_escape_alternatives {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) :
    ∃ R : ℝ, 0 < R ∧ ∀ i, (∀ k, |X k i 0| ≤ R) ∨
      Tendsto (fun k => X k i 0) atTop atTop ∨ Tendsto (fun k => X k i 0) atTop atBot := by
  obtain ⟨R, hR, hg⟩ := multiHeadTransformer_growth_threshold hΔt hV hQK hX h
  have hΛ := (multiHead_coefficients_pos (m + 1) Q K V hV hQK h).1
  have ha : 1 < 1 + Δt * multiHeadValueCoefficient V / 2 := by nlinarith [mul_pos hΔt hΛ]
  exact ⟨R, hR, fun i => discrete_bounded_or_tendsto_infinity (fun k => X k i 0) ha hR (hg i).1 (hg i).2⟩

/-- Some actual original coordinate escapes when the original data contain two distinct tokens.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, first paragraph, for §12's original equation. -/
theorem multiHeadTransformer_exists_escaping {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (L M : Idx (m + 1))
    (hLM : X 0 L 0 < X 0 M 0) (h : Idx H) :
    ∃ i, Tendsto (fun k => X k i 0) atTop atTop ∨ Tendsto (fun k => X k i 0) atTop atBot := by
  obtain ⟨R, hR, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
  by_contra hnone
  apply multiHeadTransformer_not_all_bounded hΔt hV hQK hX L M hLM h
  refine ⟨R, ?_⟩
  intro k i
  rcases halt i with hb | hp | hn
  · exact hb k
  · exact (hnone ⟨i, Or.inl hp⟩).elim
  · exact (hnone ⟨i, Or.inr hn⟩).elim

/-- The original iteration and every strict positivity premise, with an actual head,
a positive original coordinate, and a distinct initial pair, hold jointly for two heads and two tokens. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 2 → EucSpace 1) (h : Idx 2),
    (0 : ℝ) < 1 ∧ (∀ a, IsPosDefOp (V a)) ∧ (∀ a, IsPosDefQK (Q a) (K a)) ∧
    MultiHeadTransformer 1 Q K V X ∧ X 0 0 0 < X 0 1 0 ∧ 0 < X 0 1 0 ∧ h = 0 := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  let Y : Idx 2 → EucSpace 1 := ![-unit1, unit1]
  refine ⟨Q, Q, Q, multiHeadTrajectory 1 Q Q Q Y, 0, one_pos,
    fun _ => isPosDefOp_id 1, fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1),
    multiHeadTrajectory_solution _ _ _ _ _, ?_, ?_, rfl⟩
  · change Y 0 0 < Y 1 0; norm_num [Y, unit1]
  · change (0 : ℝ) < Y 1 0; norm_num [Y, unit1]

end Transformer.Clusters
