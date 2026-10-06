import Transformer.GPTMini.Sparsemax.IncidentMemoryFeasibility

/-!
# Three periodic key coordinates with size-independent norm bounds

New fixed-width architecture before arXiv:1602.02068v2, Eq. (1).
Prototype indices have three periodic classes. In each existing local
destination triple the classes are distinct, including dictionary boundaries.
Keys use their class coordinate with a learned local scale in `[1,2-floor]`.

The structural mask is an explicit architectural change. It fixes possible
neighbours, not learned edge weights or actual sparse supports. Subsequent
queries reproduce every feasible path attention with width three. Both Q
and K depend on learned geometry, without independent norm gauge variables.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The clamped destination set is exactly self, predecessor and successor.
Source: the local path restriction before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryNeighbours_iff {N : ℕ} (i j : Fin (N + 1)) :
    j ∈ localMemoryNeighbours i ↔ j = i ∨ i.val + 1 = j.val ∨ j.val + 1 = i.val := by
  constructor
  · intro hj
    simp only [localMemoryNeighbours, Finset.mem_insert, Finset.mem_singleton] at hj
    rcases hj with hj | hj | hj
    · have hv := congrArg Fin.val hj
      change j.val = min (i.val - 1) N at hv
      rw [Nat.min_eq_left (by omega)] at hv
      by_cases hi : i.val = 0
      · left
        apply Fin.ext
        omega
      · exact Or.inr (Or.inr (by omega))
    · exact Or.inl hj
    · have hv := congrArg Fin.val hj
      change j.val = min (i.val + 1) N at hv
      by_cases hi : i.val < N
      · rw [Nat.min_eq_left (by omega)] at hv
        exact Or.inr (Or.inl hv.symm)
      · rw [Nat.min_eq_right (by omega)] at hv
        left
        apply Fin.ext
        omega
  · exact localMemoryNeighbours_of_adjacent i j

/-- Three recurring positional classes; the class count is independent of memory size.
Source: the new locally masked chart before arXiv:1602.02068v2, Eq. (1). -/
def periodicMemoryClass {N : ℕ} (j : Fin (N + 1)) : Fin 3 :=
  ⟨j.val % 3, Nat.mod_lt _ (by decide)⟩

/-- Local destinations have distinct periodic classes even though distant slots reuse them.
Source: the new width-three construction preceding arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryClass_injective_local {N : ℕ} (i j k : Fin (N + 1))
    (hj : j ∈ localMemoryNeighbours i) (hk : k ∈ localMemoryNeighbours i)
    (he : periodicMemoryClass j = periodicMemoryClass k) : j = k := by
  have hv := congrArg Fin.val he
  change j.val % 3 = k.val % 3 at hv
  rcases (localMemoryNeighbours_iff i j).mp hj with hj | hj | hj <;>
    rcases (localMemoryNeighbours_iff i k).mp hk with hk | hk | hk
  all_goals
    apply Fin.ext
    try subst j
    try subst k
    omega

/-- A genuine boundary pair inhabits the local-class injectivity premises. -/
example : (0 : Fin 5) = 0 :=
  periodicMemoryClass_injective_local 0 0 0
    (localMemoryNeighbours_of_adjacent _ _ (Or.inl rfl))
    (localMemoryNeighbours_of_adjacent _ _ (Or.inl rfl)) rfl

/-- A learned key scale determined by actual incident edge mass.
Source: the new fixed-width Q/K chart preceding arXiv:1602.02068v2, Eq. (1). -/
def periodicMemoryScale {N : ℕ} (t : Fin N → ℝ) (j : Fin (N + 1)) : ℝ :=
  1 + localIncidentWeight t j

/-- Nonnegative learned edges keep every key scale away from zero.
Source: the new shared learned scale before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryScale_one_le {N : ℕ} (t : Fin N → ℝ)
    (ht : ∀ e, 0 ≤ t e) (j : Fin (N + 1)) : 1 ≤ periodicMemoryScale t j := by
  have h := localIncidentWeight_nonneg t ht j
  unfold periodicMemoryScale
  linarith

/-- Positive simultaneous learned edges satisfy the denominator bound. -/
example : 1 ≤ periodicMemoryScale (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 :=
  periodicMemoryScale_one_le _ (fun _ => by norm_num) _

/-- Local budgets give a prototype-count independent key-scale upper bound.
Source: the new bounded-width chart before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryScale_le {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (ht : t ∈ incidentMemoryWeightDomain N floor) (j : Fin (N + 1)) :
    periodicMemoryScale t j ≤ 2 - floor := by
  have h := ht.2 j
  unfold periodicMemoryScale
  linarith

/-- A changed interior key satisfies the size-independent scale premises. -/
example : periodicMemoryScale (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 ≤ 2 - (3 / 4 : ℝ) :=
  periodicMemoryScale_le _ _ incidentMemoryExampleWeights_mem _

/-- A genuine width-three key; its nonzero coordinate is learned from local edges.
Source: the new masked QK architecture before arXiv:1602.02068v2, Eq. (1). -/
def periodicMemoryKey {N : ℕ} (t : Fin N → ℝ) (j : Fin (N + 1)) : Fin 3 → ℝ :=
  fun d => if d = periodicMemoryClass j then periodicMemoryScale t j else 0

/-- The actual squared key norm is the square of its learned scale.
Source: the explicit three-coordinate key before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryKey_sq_sum {N : ℕ} (t : Fin N → ℝ) (j : Fin (N + 1)) :
    (∑ d, (periodicMemoryKey t j d) ^ 2) = (periodicMemoryScale t j) ^ 2 := by
  unfold periodicMemoryKey
  simp only [ite_pow, zero_pow (by decide : (2 : ℕ) ≠ 0)]
  exact Fintype.sum_ite_eq' (periodicMemoryClass j) _

/-- Every genuine key has squared norm at most four, for arbitrarily many prototypes.
Source: the new bounded fixed-width construction before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryKey_sq_bound {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) (j : Fin (N + 1)) :
    (∑ d, (periodicMemoryKey t j d) ^ 2) ≤ 4 := by
  rw [periodicMemoryKey_sq_sum]
  have hl := periodicMemoryScale_one_le t ht.1 j
  have hu := periodicMemoryScale_le floor t ht j
  have hb : periodicMemoryScale t j ≤ 2 := by linarith
  nlinarith [mul_self_le_mul_self (by linarith : 0 ≤ periodicMemoryScale t j) hb]

/-- Changed nonzero keys inhabit the squared-norm premises. -/
example : (∑ d, (periodicMemoryKey (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 d) ^ 2) ≤ 4 :=
  periodicMemoryKey_sq_bound _ _ (by norm_num) incidentMemoryExampleWeights_mem _

/-- Nonnegative edges give a genuinely nonzero three-coordinate key at every prototype.
Source: the explicit fixed-width keys before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryKey_ne_zero {N : ℕ} (t : Fin N → ℝ)
    (ht : ∀ e, 0 ≤ t e) (j : Fin (N + 1)) : periodicMemoryKey t j ≠ 0 := by
  intro h
  have he := congrFun h (periodicMemoryClass j)
  change (if periodicMemoryClass j = periodicMemoryClass j then periodicMemoryScale t j else 0) = 0 at he
  simp only [ite_true] at he
  have hs := periodicMemoryScale_one_le t ht j
  linarith

/-- Multiple changed edges satisfy the nonzero-key premises. -/
example : periodicMemoryKey (fun _ : Fin 3 => (1 / 8 : ℝ)) 2 ≠ 0 :=
  periodicMemoryKey_ne_zero _ (fun _ => by norm_num) _

/-- Distant prototypes share a coordinate; their local masks disambiguate them. -/
example : periodicMemoryClass (0 : Fin 5) = periodicMemoryClass (3 : Fin 5) := by
  apply Fin.ext
  norm_num [periodicMemoryClass]

end Transformer.GPTMini.Sparsemax
