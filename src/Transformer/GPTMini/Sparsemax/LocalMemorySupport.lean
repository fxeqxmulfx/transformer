import Transformer.GPTMini.Sparsemax.LocalMemoryFeasibility
import Transformer.GPTMini.Sparsemax.NearestPrototypeCodes

/-!
# At most three actual query routes in compact learned memory

Derived path restriction for arXiv:1602.02068v2, Eq. (1). Each genuine
Gram atom sends a dictionary query to itself or a neighboring slot. Their
learned affine mixture is exactly zero outside those three possible slots,
and independent squared-norm additions do not affect cross scores. On the
proved structural domain, actual variational sparsemax preserves these zeros.

A nearest-observation data code selects one learned dictionary query.
Therefore every actual query, including an unseen one, has at most three
active attention entries for every feasible learned parameter point. This
is a bound on actual attention, not merely its input code. The possible
path is fixed; learned supports can still change within it. No assertion
about learned semantic prototype order or faster nearest search is made.
Nearest search may still scan the complete prototype table.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Clamp an integer dictionary index to its last slot.
Source: boundary handling for the derived arXiv:1602.02068v2, Eq. (1) path memory. -/
def localSlotClamp (N n : ℕ) : Fin (N + 1) := ⟨min n N, by
  have h := Nat.min_le_right n N
  omega⟩

/-- The only possible actual memory destinations of a query in the path architecture.
Source: the adjacent-swap atoms preceding arXiv:1602.02068v2, Eq. (1). -/
def localMemoryNeighbours {N : ℕ} (i : Fin (N + 1)) : Finset (Fin (N + 1)) :=
  {localSlotClamp N (i.val - 1), i, localSlotClamp N (i.val + 1)}

/-- The destination set has at most three slots, also at dictionary boundaries.
Source: the derived compact path before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryNeighbours_card {N : ℕ} (i : Fin (N + 1)) :
    (localMemoryNeighbours i).card ≤ 3 := by
  calc
    _ ≤ (insert i {localSlotClamp N (i.val + 1)} : Finset (Fin (N + 1))).card + 1 :=
      Finset.card_insert_le _ _
    _ ≤ ({localSlotClamp N (i.val + 1)} : Finset (Fin (N + 1))).card + 1 + 1 :=
      Nat.add_le_add_right (Finset.card_insert_le _ _) 1
    _ = 3 := by rw [Finset.card_singleton]

/-- A fixed slot or an immediate predecessor/successor belongs to the allowed destinations.
Source: the local permutation geometry before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryNeighbours_of_adjacent {N : ℕ} (i j : Fin (N + 1))
    (h : j = i ∨ i.val + 1 = j.val ∨ j.val + 1 = i.val) :
    j ∈ localMemoryNeighbours i := by
  rcases h with h | h | h
  · subst j
    simp only [localMemoryNeighbours, Finset.mem_insert, Finset.mem_singleton, true_or, or_true]
  · have he : localSlotClamp N (i.val + 1) = j := by
      apply Fin.ext
      change min (i.val + 1) N = j.val
      rw [h, Nat.min_eq_left (by omega)]
    rw [← he]
    simp only [localMemoryNeighbours, Finset.mem_insert, Finset.mem_singleton, or_true]
  · have he : localSlotClamp N (i.val - 1) = j := by
      apply Fin.ext
      change min (i.val - 1) N = j.val
      have hv : i.val - 1 = j.val := by omega
      rw [hv, Nat.min_eq_left (by omega)]
    rw [← he]
    simp only [localMemoryNeighbours, Finset.mem_insert, true_or]

/-- A genuine interior neighbor inhabits the geometric premise. -/
example : (2 : Fin 4) ∈ localMemoryNeighbours (1 : Fin 4) :=
  localMemoryNeighbours_of_adjacent _ _ (Or.inr (Or.inl (by norm_num)))

/-- Every genuine path atom routes inside the three-slot destination set.
Source: the derived atom feature products before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryPermutation_mem_neighbours {N : ℕ} (e : Option (Fin N)) (i : Fin (N + 1)) :
    localMemoryPermutation e i ∈ localMemoryNeighbours i :=
  localMemoryNeighbours_of_adjacent i _ (localMemoryPermutation_moves e i)

/-- Core scores outside the neighboring slots are exactly zero for all learned edge weights.
Source: the genuine affine atom mixture preceding arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_zero_of_not_mem {N : ℕ} (t : Fin N → ℝ) (i j : Fin (N + 1))
    (hj : j ∉ localMemoryNeighbours i) : memoryGramScores (localMemoryCore t) i j = 0 := by
  rw [localMemoryCore_scores_apply]
  apply Finset.sum_eq_zero
  intro e he
  have hn : localMemoryPermutation e i ≠ j := by
    intro h
    exact hj (h ▸ localMemoryPermutation_mem_neighbours e i)
  simp only [ite_eq_right hn, mul_zero]

/-- Positive edge coordinates and a genuinely distant slot inhabit the exact-zero premise. -/
example : memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 20 : ℝ))) 0 3 = 0 := by
  apply localMemoryCore_zero_of_not_mem
  norm_num [localMemoryNeighbours, localSlotClamp]

/-- Actual variational sparsemax has exact zeros outside the neighboring dictionary slots.
Source: arXiv:1602.02068v2, Eq. (1), applied to the proved compact probability-score domain. -/
theorem localMemoryAttention_zero_of_not_mem {N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (hf : 0 ≤ floor)
    (hp : p ∈ localMemoryParameterDomain N cap floor) (i j : Fin (N + 1))
    (hj : j ∉ localMemoryNeighbours i) : memoryGramAttention (localMemoryGram p) i j = 0 := by
  rw [localMemoryAttention_normalized cap floor p hf hp]
  exact localMemoryCore_zero_of_not_mem p.1 i j hj

/-- Actual nonzero four-slot embeddings inhabit every structural zero premise. -/
example : memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters 3)) 0 3 = 0 :=
  localMemoryAttention_zero_of_not_mem 4 (3 / 4) _ (by norm_num)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)) _ _
    (by norm_num [localMemoryNeighbours, localSlotClamp])

/-- Every actual compact memory query has at most three nonzero attention entries.
Source: the structural support bound for variational arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryAttention_support_card {N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (hf : 0 ≤ floor)
    (hp : p ∈ localMemoryParameterDomain N cap floor) (i : Fin (N + 1)) :
    (Finset.univ.filter (fun j => memoryGramAttention (localMemoryGram p) i j ≠ 0)).card ≤ 3 := by
  classical
  apply (Finset.card_le_card ?_).trans (localMemoryNeighbours_card i)
  intro j hj
  by_contra hn
  exact (Finset.mem_filter.1 hj).2 (localMemoryAttention_zero_of_not_mem cap floor p hf hp i j hn)

/-- A dictionary larger than the bound inhabits every actual support-cardinality premise. -/
example : (Finset.univ.filter (fun j =>
    memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters 3)) 1 j ≠ 0)).card ≤ 3 :=
  localMemoryAttention_support_card 4 (3 / 4) _ (by norm_num)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)) _

/-- Every nearest-data query has at most three active actual sparsemax routes, including unseen data.
Source: the nearest mixture and compact memory domain for arXiv:1602.02068v2, Eq. (1). -/
theorem contextNearestLocalAttention_support_card {Key : Type*} {R N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (hf : 0 ≤ floor) (hp : p ∈ localMemoryParameterDomain N cap floor) (r : Fin R) :
    (Finset.univ.filter (fun j => contextMemoryAttention (localMemoryGram p)
      (nearestPrototypeCodes distance prototypes queries) r j ≠ 0)).card ≤ 3 := by
  classical
  rw [contextNearestAttention_row cap floor _ _ _ _ (localMemoryGram_mem cap floor p hf hp)]
  exact localMemoryAttention_support_card cap floor p hf hp _

/-- An actual unseen real query and four memory slots inhabit the full three-route bound. -/
example : (Finset.univ.filter (fun j => contextMemoryAttention
    (localMemoryGram (0 : LocalMemoryParameters 3))
    (nearestPrototypeCodes (fun x y : ℝ => |x - y|)
      (fun j : Fin 4 => (j.val : ℝ)) (fun _ : Fin 1 => (1 / 2 : ℝ))) 0 j ≠ 0)).card ≤ 3 :=
  contextNearestLocalAttention_support_card 4 (3 / 4) _ _ _ _ (by norm_num)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)) _

end Transformer.GPTMini.Sparsemax
