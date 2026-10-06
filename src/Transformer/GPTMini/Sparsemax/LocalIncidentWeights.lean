import Transformer.GPTMini.Sparsemax.LocalMemoryWeights

/-!
# Separate incident-edge budgets for compact sparse attention

Derived extension of the path memory before arXiv:1602.02068v2, Eq. (1).
Each slot bounds only the learned edges touching it. Distant edges no
longer share one global budget. The new domain is convex, has the same N
stored edge coordinates, and strictly contains the former global domain.

The residual global identity coefficient may now be negative. Therefore
the former convex-mixture PSD proof does not apply; the stochastic
outer-product construction supplies the new proof in later modules.
The path and same-family orthogonality remain architectural restrictions.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Total learned mass of the path edges touching a slot.
Source: the local self-weight budget before arXiv:1602.02068v2, Eq. (1). -/
def localIncidentWeight {N : ℕ} (t : Fin N → ℝ) (i : Fin (N + 1)) : ℝ :=
  ∑ e, if i = e.castSucc ∨ i = e.succ then t e else 0

/-- A self-weight budget separately for each memory slot.
Source: the derived compact local normalization for arXiv:1602.02068v2, Eq. (1). -/
def incidentMemoryWeightDomain (N : ℕ) (floor : ℝ) : Set (Fin N → ℝ) :=
  {t | (∀ e, 0 ≤ t e) ∧ ∀ i, localIncidentWeight t i ≤ 1 - floor}

/-- Incident mass is linear in all learned edge coordinates.
Source: the local linear constraints preceding arXiv:1602.02068v2, Eq. (1). -/
theorem localIncidentWeight_linear {N : ℕ} (t s : Fin N → ℝ) (a b : ℝ)
    (i : Fin (N + 1)) :
    localIncidentWeight (a • t + b • s) i =
      a * localIncidentWeight t i + b * localIncidentWeight s i := by
  unfold localIncidentWeight
  change (∑ e, if i = e.castSucc ∨ i = e.succ then a * t e + b * s e else 0) = _
  have he (e : Fin N) :
      (if i = e.castSucc ∨ i = e.succ then a * t e + b * s e else 0) =
      a * (if i = e.castSucc ∨ i = e.succ then t e else 0) +
        b * (if i = e.castSucc ∨ i = e.succ then s e else 0) := by
    split_ifs <;> ring
  simp_rw [he]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- The relaxed edge domain remains convex without fixing actual support.
Source: separate row budgets before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryWeightDomain_convex (N : ℕ) (floor : ℝ) :
    Convex ℝ (incidentMemoryWeightDomain N floor) := by
  intro t ht s hs a b ha hb hab
  refine ⟨?_, ?_⟩
  · intro e
    change 0 ≤ a * t e + b * s e
    exact add_nonneg (mul_nonneg ha (ht.1 e)) (mul_nonneg hb (hs.1 e))
  · intro i
    rw [localIncidentWeight_linear]
    calc
      _ ≤ a * (1 - floor) + b * (1 - floor) :=
        add_le_add (mul_le_mul_of_nonneg_left (ht.2 i) ha)
          (mul_le_mul_of_nonneg_left (hs.2 i) hb)
      _ = 1 - floor := by rw [← add_mul, hab, one_mul]

/-- Nonnegative edge coordinates make each incident mass nonnegative.
Source: the local probability-score restriction before arXiv:1602.02068v2, Eq. (1). -/
theorem localIncidentWeight_nonneg {N : ℕ} (t : Fin N → ℝ)
    (ht : ∀ e, 0 ≤ t e) (i : Fin (N + 1)) : 0 ≤ localIncidentWeight t i := by
  apply Finset.sum_nonneg
  intro e he
  split_ifs
  · exact ht e
  · exact le_refl 0

/-- Positive edges inhabit the incident nonnegativity premises. -/
example : 0 ≤ localIncidentWeight (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 :=
  localIncidentWeight_nonneg _ (fun e => by norm_num) _

/-- Every edge is bounded by the budget of its left endpoint.
Source: a genuine coordinate consequence of the local arXiv:1602.02068v2, Eq. (1) domain. -/
theorem incidentMemoryWeightDomain_coordinateBound {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (ht : t ∈ incidentMemoryWeightDomain N floor) (e : Fin N) :
    0 ≤ t e ∧ t e ≤ 1 - floor := by
  refine ⟨ht.1 e, ?_⟩
  have hs := Finset.single_le_sum
    (s := Finset.univ) (f := fun j : Fin N =>
      if e.castSucc = j.castSucc ∨ e.castSucc = j.succ then t j else 0)
    (fun j hj => by split_ifs; exact ht.1 j; exact le_refl 0) (Finset.mem_univ e)
  simp only [eq_self, true_or, ite_true] at hs
  exact hs.trans (ht.2 e.castSucc)

/-- A global-budget point satisfies every separate local budget.
Source: comparison of the two derived domains before arXiv:1602.02068v2, Eq. (1). -/
theorem localWeightDomain_subset_incident (N : ℕ) (floor : ℝ) :
    localWeightDomain N floor ⊆ incidentMemoryWeightDomain N floor := by
  intro t ht
  refine ⟨ht.1, ?_⟩
  intro i
  apply le_trans ?_ ht.2
  apply Finset.sum_le_sum
  intro e he
  split_ifs
  · exact le_refl _
  · exact ht.1 e

/-- Zero edges inhabit all floors at most one in the relaxed domain.
Source: the identity endpoint of the local arXiv:1602.02068v2, Eq. (1) restriction. -/
theorem zero_mem_incidentMemoryWeightDomain (N : ℕ) (floor : ℝ) (hf : floor ≤ 1) :
    (0 : Fin N → ℝ) ∈ incidentMemoryWeightDomain N floor :=
  localWeightDomain_subset_incident N floor (zero_mem_localWeightDomain N floor hf)

/-- The relaxed zero-edge domain has an actual admissible strict floor. -/
example : (0 : Fin 3 → ℝ) ∈ incidentMemoryWeightDomain 3 (3 / 4) :=
  zero_mem_incidentMemoryWeightDomain _ _ (by norm_num)

/-- Separate budgets are feasible exactly for floors at most one.
Source: nonnegative incident mass in the derived arXiv:1602.02068v2, Eq. (1) domain. -/
theorem incidentMemoryWeightDomain_nonempty_iff (N : ℕ) (floor : ℝ) :
    (incidentMemoryWeightDomain N floor).Nonempty ↔ floor ≤ 1 := by
  constructor
  · rintro ⟨t, ht⟩
    have hn := localIncidentWeight_nonneg t ht.1 0
    have hb := ht.2 0
    linarith
  · intro hf
    exact ⟨0, zero_mem_incidentMemoryWeightDomain N floor hf⟩

/-- Three nonzero edges share two local budgets rather than one global budget.
Source: a strict enlargement witness for arXiv:1602.02068v2, Eq. (1) memory. -/
theorem incidentMemoryExampleWeights_mem :
    (fun _ : Fin 3 => (1 / 8 : ℝ)) ∈ incidentMemoryWeightDomain 3 (3 / 4) := by
  refine ⟨fun e => by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [localIncidentWeight, Fin.sum_univ_three]

/-- The new feasible point fails the former global budget.
Source: counterexample to necessity of the global budget for the derived Eq. (1) memory. -/
theorem incidentMemoryExampleWeights_not_global :
    (fun _ : Fin 3 => (1 / 8 : ℝ)) ∉ localWeightDomain 3 (3 / 4) := by
  intro ht
  have hb := ht.2
  norm_num [Fin.sum_univ_three] at hb

/-- Positive relaxed edges inhabit the coordinate-bound assumptions. -/
example : 0 ≤ (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 ∧
    (fun _ : Fin 3 => (1 / 8 : ℝ)) 1 ≤ 1 - (3 / 4 : ℝ) :=
  incidentMemoryWeightDomain_coordinateBound _ _ incidentMemoryExampleWeights_mem 1

end Transformer.GPTMini.Sparsemax
