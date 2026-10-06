import Transformer.GPTMini.Sparsemax.AffineValueDecoder

/-!
# Size-independent scalar constraints for generated sparse attention

New compact geometry following arXiv:1602.02068v2, Eq. (1).
Normalize the learned scalar by `(N+2)^2`, generating all quadratic path
edges from one number. A scalar budget `0 <= rho <= 1-floor` implies
every original nonnegative-edge and local self-weight constraint, without
enumerating or storing a vector of learned weights in the model state.

The actual earlier width-three queries, keys, structural mask and
variational sparsemax therefore apply for arbitrarily many slots.
These are sufficient scalar constraints, not the entire old edge domain.
The conservative normalization keeps local weights bounded as the number
of prototypes grows; supports still change between zero and positive rho.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The incident mass of generated edges has a closed formula at every slot and boundary.
Source: the invariant-feature profile before sparsemax Eq. (1). -/
theorem affineValueEdges_incident (N : ℕ) (c : ℝ) (i : Fin (N + 2)) :
    localIncidentWeight (affineValueEdges N c) i =
      c * ((i.val + 1) * (N + 1 - i.val) + i.val * (N + 2 - i.val)) := by
  unfold localIncidentWeight
  have he (e : Fin (N + 1)) :
      (if i = e.castSucc ∨ i = e.succ then affineValueEdges N c e else 0) =
      c * (if i = e.castSucc then ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) +
      c * (if i = e.succ then ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) := by
    have hn : e.castSucc ≠ e.succ := by
      intro h
      have hv := congrArg Fin.val h
      simp only [Fin.val_castSucc, Fin.val_succ] at hv
      omega
    by_cases hl : i = e.castSucc
    · subst i
      simp only [eq_self, hn, true_or, ite_true, ite_false, mul_zero, add_zero, affineValueEdges]
      ring
    · by_cases hr : i = e.succ
      · subst i
        simp only [eq_self, Ne.symm hn, or_true, ite_true, ite_false, mul_zero, zero_add, affineValueEdges]
        ring
      · simp only [hl, hr, or_self, ite_false, mul_zero, add_zero]
  simp_rw [he]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    affineValueProfile_right, affineValueProfile_left]
  ring

/-- A normalized scalar can bound all local budgets using only the dictionary size.
Source: the quadratic generated profile before arXiv:1602.02068v2, Eq. (1). -/
theorem affineValueEdges_incident_bound (N : ℕ) (c : ℝ) (hc : 0 ≤ c) (i : Fin (N + 2)) :
    localIncidentWeight (affineValueEdges N c) i ≤ c * (N + 2) ^ 2 := by
  rw [affineValueEdges_incident]
  apply mul_le_mul_of_nonneg_left _ hc
  have hn : 0 ≤ (N : ℝ) := Nat.cast_nonneg N
  nlinarith [sq_nonneg (2 * (i.val : ℝ) - (N + 1))]

/-- Four nonidentity slots inhabit every conservative local-budget premise. -/
example : localIncidentWeight (affineValueEdges 2 (1 / 48)) (1 : Fin 4) ≤
    (1 / 48 : ℝ) * (2 + 2) ^ 2 :=
  affineValueEdges_incident_bound 2 (1 / 48) (by norm_num) 1

/-- The learned scalar is rescaled analytically, without a stored per-edge normalization table.
Source: the new compact scalar geometry before sparsemax Eq. (1). -/
def affineValueNormalizedMix (N : ℕ) (rho : ℝ) : ℝ := rho / (N + 2) ^ 2

/-- Normalized geometry depends linearly on its single learned scalar.
Source: the generated compact geometry before arXiv:1602.02068v2, Eq. (1). -/
theorem affineValueNormalizedMix_linear (N : ℕ) (rho sigma a b : ℝ) :
    affineValueNormalizedMix N (a * rho + b * sigma) =
      a * affineValueNormalizedMix N rho + b * affineValueNormalizedMix N sigma := by
  unfold affineValueNormalizedMix
  ring

/-- A scalar interval implies all genuine local feasibility constraints at every dictionary size.
Source: sufficient scalar normalization for the actual sparsemax Eq. (1) path domain. -/
theorem affineValueNormalizedEdges_mem (N : ℕ) (floor rho : ℝ)
    (h0 : 0 ≤ rho) (h1 : rho ≤ 1 - floor) :
    affineValueEdges N (affineValueNormalizedMix N rho) ∈
      incidentMemoryWeightDomain (N + 1) floor := by
  have hd : 0 < ((N : ℝ) + 2) ^ 2 := by positivity
  have hc : 0 ≤ affineValueNormalizedMix N rho := div_nonneg h0 hd.le
  constructor
  · intro e
    have hv : (e.val : ℝ) ≤ N := by exact_mod_cast (by omega : e.val ≤ N)
    unfold affineValueEdges
    exact mul_nonneg (mul_nonneg hc (by positivity)) (by linarith)
  · intro i
    have he : affineValueNormalizedMix N rho * (N + 2) ^ 2 = rho :=
      div_mul_cancel₀ rho (ne_of_gt hd)
    have hb := affineValueEdges_incident_bound N _ hc i
    rw [he] at hb
    exact hb.trans h1

/-- A thousand slots retain genuine feasibility from the same single scalar budget. -/
example : affineValueEdges 998 (affineValueNormalizedMix 998 (1 / 8)) ∈
    incidentMemoryWeightDomain 999 (3 / 4) :=
  affineValueNormalizedEdges_mem 998 (3 / 4) (1 / 8) (by norm_num) (by norm_num)

/-- Scalar feasibility retains the uniform nonconstant inverse-feature bound.
Source: the original strict-floor guarantee after arXiv:1602.02068v2, Eq. (1). -/
theorem affineValueNormalizedMix_eigenvalue (N : ℕ) (floor rho : ℝ)
    (h0 : 0 ≤ rho) (h1 : rho ≤ 1 - floor) :
    2 * floor - 1 ≤ 1 - 2 * affineValueNormalizedMix N rho :=
  affineValueMix_eigenvalue N floor _ (affineValueNormalizedEdges_mem N floor rho h0 h1)

/-- A positive learned scalar inhabits the normalized inverse-bound premises. -/
example : 2 * (3 / 4 : ℝ) - 1 ≤ 1 - 2 * affineValueNormalizedMix 998 (1 / 8) :=
  affineValueNormalizedMix_eigenvalue 998 (3 / 4) (1 / 8) (by norm_num) (by norm_num)

/-- Arbitrarily many slots retain genuine width-three Q/K under the one-scalar geometry.
Source: the proved physical sparsemax Eq. (1) realization applied to generated feasible edges. -/
theorem affineValueNormalized_width_three (N : ℕ) (floor rho : ℝ)
    (hf : 0 ≤ floor) (h0 : 0 ≤ rho) (h1 : rho ≤ 1 - floor) :
    ∃ Q K : Fin (N + 2) → Fin 3 → ℝ,
      (∀ i, (∑ d, (Q i d) ^ 2) ≤ 3) ∧ (∀ j, (∑ d, (K j d) ^ 2) ≤ 4) ∧
      ∀ i j, j ∈ localMemoryNeighbours i →
        (∑ d, Q i d * K j d) =
          memoryGramScores (localMemoryCore (affineValueEdges N (affineValueNormalizedMix N rho))) i j :=
  periodicMemory_width_three floor _ hf (affineValueNormalizedEdges_mem N floor rho h0 h1)

/-- Thousand-slot genuine queries and keys still use only three physical coordinates. -/
example : ∃ Q K : Fin 1000 → Fin 3 → ℝ,
    (∀ i, (∑ d, (Q i d) ^ 2) ≤ 3) ∧ (∀ j, (∑ d, (K j d) ^ 2) ≤ 4) ∧
    ∀ i j, j ∈ localMemoryNeighbours i → (∑ d, Q i d * K j d) =
      memoryGramScores (localMemoryCore (affineValueEdges 998 (affineValueNormalizedMix 998 (1 / 8)))) i j :=
  affineValueNormalized_width_three 998 (3 / 4) (1 / 8) (by norm_num) (by norm_num) (by norm_num)

/-- A genuine query needs values only at its at most three possible local destinations.
Source: actual sparsemax Eq. (1) support on the structural mask and generated common values. -/
theorem affineValueForward_local {R N D : ℕ} (floor : ℝ) (code : Fin R → Fin (N + 2))
    (c : ℝ) (W : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (ht : affineValueEdges N c ∈ incidentMemoryWeightDomain (N + 1) floor) (r : Fin R) (d : Fin D) :
    affineValueForward code c W r d =
      ∑ j ∈ localMemoryNeighbours (code r), periodicMemoryAttention (affineValueEdges N c) (code r) j *
        affineValueOutput N (affineValueCoefficients c W) j d := by
  unfold affineValueForward
  rw [Matrix.mul_apply]
  apply Eq.symm
  apply Finset.sum_subset (Finset.subset_univ _)
  intro j hj hnot
  rw [periodicMemoryAttention_normalized floor _ (by linarith) ht,
    localMemoryCore_zero_of_not_mem _ _ _ hnot, zero_mul]

/-- Four nonconstant value rows are evaluated only on the three possible routes of one query. -/
example : affineValueForward (fun j : Fin 4 => j) (1 / 48)
    (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ))) 1 0 =
    ∑ j ∈ localMemoryNeighbours (1 : Fin 4), periodicMemoryAttention (affineValueEdges 2 (1 / 48)) 1 j *
      affineValueOutput 2 (affineValueCoefficients (1 / 48)
        (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ)))) j 0 :=
  affineValueForward_local (3 / 4) _ _ _ (by norm_num) affineValueFourEdges_mem 1 0

end Transformer.GPTMini.Sparsemax
