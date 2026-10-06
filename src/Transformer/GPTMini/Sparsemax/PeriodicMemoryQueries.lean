import Transformer.GPTMini.Sparsemax.PeriodicMemoryKeys

/-!
# Learned bounded queries realize every local score at width three

New construction before arXiv:1602.02068v2, Eq. (1). Each possible
destination contributes its desired score divided by the corresponding
learned key scale to that destination's periodic query coordinate. Distinct
local classes prevent interference. The genuine three-coordinate QK dot
product recovers every assigned score on the local mask.

Both families change with learned edge weights. Query coordinates stay in
`[0,1]`, so their squared norm is at most three, independently of prototype
count. Key squared norms are at most four by the preceding module. A later
module proves the actual variational sparsemax after masking, rather than
assuming the target weights or an embedding factorization.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Width-three learned queries assembled from local scores and learned key scales.
Source: the new masked constant-width chart before arXiv:1602.02068v2, Eq. (1). -/
def periodicMemoryQuery {N : ℕ} (t : Fin N → ℝ) (i : Fin (N + 1)) : Fin 3 → ℝ :=
  fun d => ∑ j ∈ localMemoryNeighbours i,
    if d = periodicMemoryClass j then memoryGramScores (localMemoryCore t) i j /
      periodicMemoryScale t j else 0

/-- The genuine QK dot product of the learned three-coordinate families.
Source: the new constant-width input to arXiv:1602.02068v2, Eq. (1). -/
def periodicMemoryScore {N : ℕ} (t : Fin N → ℝ) (i j : Fin (N + 1)) : ℝ :=
  ∑ d, periodicMemoryQuery t i d * periodicMemoryKey t j d

/-- Evaluating actual QK products recovers every possible local score exactly.
Source: the new width-three realization before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryScore_local {N : ℕ} (t : Fin N → ℝ)
    (ht : ∀ e, 0 ≤ t e) (i j : Fin (N + 1)) (hj : j ∈ localMemoryNeighbours i) :
    periodicMemoryScore t i j = memoryGramScores (localMemoryCore t) i j := by
  have hscale := periodicMemoryScale_one_le t ht j
  unfold periodicMemoryScore periodicMemoryKey
  simp only [mul_ite, mul_zero]
  rw [Fintype.sum_ite_eq']
  unfold periodicMemoryQuery
  have he (k : Fin (N + 1)) (hk : k ∈ localMemoryNeighbours i) :
      (if periodicMemoryClass j = periodicMemoryClass k then
        memoryGramScores (localMemoryCore t) i k / periodicMemoryScale t k else 0) =
      if k = j then memoryGramScores (localMemoryCore t) i j / periodicMemoryScale t j else 0 := by
    by_cases h : k = j
    · subst k
      simp
    · have hc : periodicMemoryClass j ≠ periodicMemoryClass k := by
        intro hc
        exact h (periodicMemoryClass_injective_local i j k hj hk hc).symm
      simp only [hc, h, ite_false]
  rw [Finset.sum_congr rfl he, Finset.sum_ite_eq', ite_eq_left hj]
  exact div_mul_cancel₀ _ (by linarith)

/-- A changed genuine off-diagonal score inhabits every width-three realization premise. -/
example : periodicMemoryScore (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 2 =
    memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) 1 2 :=
  periodicMemoryScore_local _ (fun _ => by norm_num) _ _
    (localMemoryNeighbours_of_adjacent _ _ (Or.inr (Or.inl (by norm_num))))

/-- Every genuine query coordinate lies in `[0,1]` under the local probability-score bounds.
Source: the new bounded constant-width chart before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryQuery_bounds {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor)
    (i : Fin (N + 1)) (d : Fin 3) : 0 ≤ periodicMemoryQuery t i d ∧ periodicMemoryQuery t i d ≤ 1 := by
  have hn (j : Fin (N + 1)) := incidentMemory_scores_nonneg floor t hf ht i j
  have hs (j : Fin (N + 1)) := periodicMemoryScale_one_le t ht.1 j
  constructor
  · apply Finset.sum_nonneg
    intro j hj
    split_ifs
    · exact div_nonneg (hn j) (by linarith [hs j])
    · exact le_rfl
  · calc
      _ ≤ ∑ j ∈ localMemoryNeighbours i, memoryGramScores (localMemoryCore t) i j := by
        apply Finset.sum_le_sum
        intro j hj
        split_ifs
        · exact div_le_self (hn j) (hs j)
        · exact hn j
      _ ≤ ∑ j, memoryGramScores (localMemoryCore t) i j :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) (fun j _ _ => hn j)
      _ = 1 := localMemoryCore_scores_rowSum t i

/-- A changed interior query satisfies the coordinate-bound hypotheses. -/
example : 0 ≤ periodicMemoryQuery (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 2 ∧
    periodicMemoryQuery (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 2 ≤ 1 :=
  periodicMemoryQuery_bounds _ _ (by norm_num) incidentMemoryExampleWeights_mem _ _

/-- Every actual query has squared norm at most three for arbitrarily many prototypes.
Source: the new fixed-width score chart before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryQuery_sq_bound {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) (i : Fin (N + 1)) :
    (∑ d, (periodicMemoryQuery t i d) ^ 2) ≤ 3 := by
  calc
    _ ≤ ∑ _ : Fin 3, (1 : ℝ) := by
      apply Finset.sum_le_sum
      intro d hd
      have hb := periodicMemoryQuery_bounds floor t hf ht i d
      have hs := mul_self_le_mul_self hb.1 hb.2
      nlinarith
    _ = 3 := by norm_num

/-- Nonzero learned queries satisfy every norm-bound premise. -/
example : (∑ d, (periodicMemoryQuery (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 d) ^ 2) ≤ 3 :=
  periodicMemoryQuery_sq_bound _ _ (by norm_num) incidentMemoryExampleWeights_mem _

/-- All row scores can be recovered with exactly three genuine query and key coordinates.
Source: the new bounded masked realization before arXiv:1602.02068v2, Eq. (1).
The existence conclusion is proved by explicit Q/K families, not assumed as an oracle. -/
theorem periodicMemory_width_three {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    ∃ Q K : Fin (N + 1) → Fin 3 → ℝ,
      (∀ i, (∑ d, (Q i d) ^ 2) ≤ 3) ∧ (∀ j, (∑ d, (K j d) ^ 2) ≤ 4) ∧
      ∀ i j, j ∈ localMemoryNeighbours i →
        (∑ d, Q i d * K j d) = memoryGramScores (localMemoryCore t) i j := by
  exact ⟨periodicMemoryQuery t, periodicMemoryKey t,
    periodicMemoryQuery_sq_bound floor t hf ht, periodicMemoryKey_sq_bound floor t hf ht,
      periodicMemoryScore_local t ht.1⟩

/-- Four prototypes and three simultaneous learned edges inhabit the constant-width premises. -/
example : ∃ Q K : Fin 4 → Fin 3 → ℝ,
    (∀ i, (∑ d, (Q i d) ^ 2) ≤ 3) ∧ (∀ j, (∑ d, (K j d) ^ 2) ≤ 4) ∧
    ∀ i j, j ∈ localMemoryNeighbours i →
      (∑ d, Q i d * K j d) = memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) i j :=
  periodicMemory_width_three _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- A learned edge change really changes the query family; queries are not frozen interpolation anchors.
Source: a genuine fixed-width witness before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryQuery_changes :
    periodicMemoryQuery (fun _ : Fin 3 => (1 / 8 : ℝ)) (1 : Fin 4) ≠
      periodicMemoryQuery (0 : Fin 3 → ℝ) (1 : Fin 4) := by
  intro h
  have he := congrFun h 1
  norm_num [periodicMemoryQuery, localMemoryNeighbours, localSlotClamp, periodicMemoryClass,
    periodicMemoryScale, localIncidentWeight, Fin.sum_univ_succ, localMemoryCore_scores_apply,
    Fintype.sum_option, localMemoryWeights, localMemoryPermutation, Equiv.swap_apply_def] at he

/-- The same learned edge change changes the key family as well.
Source: a genuine fixed-width witness before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryKey_changes :
    periodicMemoryKey (fun _ : Fin 3 => (1 / 8 : ℝ)) (1 : Fin 4) ≠
      periodicMemoryKey (0 : Fin 3 → ℝ) (1 : Fin 4) := by
  intro h
  have he := congrFun h 1
  norm_num [periodicMemoryKey, periodicMemoryScale, periodicMemoryClass,
    localIncidentWeight, Fin.sum_univ_succ] at he

end Transformer.GPTMini.Sparsemax
