import Transformer.GPTMini.Sparsemax.Basic
import Mathlib.Topology.Order.IntermediateValue

/-!
# Closed form of the actual causal simplex projection

arXiv:1602.02068v2, §2.2, Proposition 1, equations `sparsemax_closedform`
and `threshold_closedform`. The causal mask restricts the paper's simplex
to positions at most the query. We prove the clipped-threshold formula
for the existing variational `sparseWeights`, rather than defining its
optimality or its derivative into a new weight function.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The paper's clipped candidate, with forbidden causal positions zero.
Source: arXiv:1602.02068v2, §2.2, equation `sparsemax_closedform`. -/
def thresholdWeights {T : ℕ} (scores : Fin T → ℝ) (i : Fin T)
    (τ : ℝ) : Fin T → ℝ :=
  fun j => if j ≤ i then max (scores j - τ) 0 else 0

/-- A normalized clipped candidate is the actual variational projection.
Source: arXiv:1602.02068v2, §2.2, Proposition 1. The proof certifies its
global objective against every feasible row, including inactive entries. -/
theorem thresholdWeights_eq_sparseWeights {T : ℕ} (scores : Fin T → ℝ)
    (i : Fin T) (τ : ℝ) (hsum : ∑ j, thresholdWeights scores i τ j = 1) :
    thresholdWeights scores i τ = sparseWeights scores i := by
  classical
  let p := thresholdWeights scores i τ
  have hp : p ∈ simplexOn {j : Fin T | j ≤ i} := by
    refine ⟨?_, hsum, ?_⟩
    · intro j
      simp only [p, thresholdWeights]
      split_ifs
      · exact le_max_right _ _
      · exact le_rfl
    · intro j hj
      change ¬ j ≤ i at hj
      simp [p, thresholdWeights, hj]
  have hmin : ∀ b ∈ simplexOn {j : Fin T | j ≤ i},
      routingObjective (fun j => -scores j / 2) p ≤
        routingObjective (fun j => -scores j / 2) b := by
    intro b hb
    have hres : 0 ≤ ∑ j, (b j - p j) * (p j - scores j + τ) := by
      apply Finset.sum_nonneg
      intro j _
      by_cases hj : j ≤ i
      · by_cases hs : 0 ≤ scores j - τ
        · have he : p j = scores j - τ := by
            simp [p, thresholdWeights, hj, max_eq_left hs]
          rw [he]
          ring_nf
          exact le_rfl
        · have he : p j = 0 := by
            simp [p, thresholdWeights, hj, max_eq_right (le_of_not_ge hs)]
          rw [he]
          exact mul_nonneg (by simpa using hb.1 j) (by linarith)
      · have hpj : p j = 0 := hp.2.2 j hj
        have hbj : b j = 0 := hb.2.2 j hj
        simp [hpj, hbj]
    have hzero : (∑ j, (b j - p j)) = 0 := by
      rw [Finset.sum_sub_distrib, hb.2.1, hp.2.1, sub_self]
    have hid : (∑ j, (b j - p j) ^ 2) =
        (∑ j, (b j) ^ 2) - (∑ j, (p j) ^ 2) -
          2 * (∑ j, (b j - p j) * scores j) +
          2 * τ * (∑ j, (b j - p j)) -
          2 * (∑ j, (b j - p j) * (p j - scores j + τ)) := by
      calc
        _ = ∑ j, ((b j) ^ 2 - (p j) ^ 2 -
            2 * ((b j - p j) * scores j) +
            2 * τ * (b j - p j) -
            2 * ((b j - p j) * (p j - scores j + τ))) := by
          apply Finset.sum_congr rfl
          intro j _
          ring
        _ = _ := by
          rw [Finset.sum_sub_distrib, Finset.sum_add_distrib,
            Finset.sum_sub_distrib, Finset.sum_sub_distrib,
            ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
    have hlin : (∑ j, (b j - p j) * scores j) =
        (∑ j, b j * scores j) - ∑ j, p j * scores j := by
      simp_rw [sub_mul]
      rw [Finset.sum_sub_distrib]
    have hcost (a : Fin T → ℝ) : (∑ j, a j * (-scores j / 2)) =
        -(∑ j, a j * scores j) / 2 := by
      calc
        _ = ∑ j, -(a j * scores j) / 2 := by
          apply Finset.sum_congr rfl
          intro j _
          ring
        _ = _ := by rw [← Finset.sum_div, Finset.sum_neg_distrib]
    have hsq : 0 ≤ ∑ j, (b j - p j) ^ 2 :=
      Finset.sum_nonneg fun j _ => sq_nonneg _
    rw [hzero, hlin] at hid
    unfold routingObjective
    rw [hcost, hcost]
    linarith
  exact routing_minimizers_eq {j : Fin T | j ≤ i}
    (fun j => -scores j / 2) p (sparseWeights scores i) hp
    (sparseWeights_spec scores i).1 hmin (sparseWeights_spec scores i).2

/-- A two-position full-support row satisfies the normalization premise.
Source context: arXiv:1602.02068v2, §2.2, Proposition 1. -/
example : ∑ j : Fin 2, thresholdWeights (fun _ => 0) 1 (-(1 / 2)) j = 1 := by
  norm_num [thresholdWeights, Fin.sum_univ_two]

/-- A normalized threshold exists for every causal row. Source:
arXiv:1602.02068v2, §2.2, Proposition 1. This proves existence from
continuity and the allowed self position; no threshold oracle is assumed. -/
theorem thresholdWeights_exists {T : ℕ} (scores : Fin T → ℝ) (i : Fin T) :
    ∃ τ : ℝ, ∑ j, thresholdWeights scores i τ j = 1 := by
  classical
  let upper := ∑ j, |scores j|
  have hupper (j : Fin T) : scores j ≤ upper := by
    exact le_trans (le_abs_self _) (Finset.single_le_sum
      (fun k _ => abs_nonneg (scores k)) (Finset.mem_univ j))
  let mass := fun τ : ℝ => ∑ j, thresholdWeights scores i τ j
  have hc : Continuous mass := by
    apply continuous_finsetSum
    intro j _
    by_cases hj : j ≤ i
    · simp only [thresholdWeights, hj, ite_true]
      exact (continuous_const.sub continuous_id).max continuous_const
    · simp only [thresholdWeights, hj, ite_false]
      exact continuous_const
  have hlower : 1 ≤ mass (scores i - 1) := by
    have hterm : thresholdWeights scores i (scores i - 1) i = 1 := by
      simp [thresholdWeights]
    have hle : thresholdWeights scores i (scores i - 1) i ≤ mass (scores i - 1) := by
      apply Finset.single_le_sum
      · intro j _
        unfold thresholdWeights
        split_ifs
        · exact le_max_right _ _
        · exact le_rfl
      · exact Finset.mem_univ i
    simpa only [hterm] using hle
  have hzero : mass upper = 0 := by
    apply Finset.sum_eq_zero
    intro j _
    by_cases hj : j ≤ i
    · simp [thresholdWeights, hj, max_eq_right (sub_nonpos.mpr (hupper j))]
    · simp [thresholdWeights, hj]
  have hab : scores i - 1 ≤ upper := by linarith [hupper i]
  obtain ⟨τ, _, hτ⟩ := intermediate_value_Icc' hab hc.continuousOn
    (show (1 : ℝ) ∈ Set.Icc (mass upper) (mass (scores i - 1)) from
      ⟨by rw [hzero]; norm_num, hlower⟩)
  exact ⟨τ, hτ⟩

/-- The existing causal projection has the paper's clipped-threshold
form. Source: arXiv:1602.02068v2, §2.2, Proposition 1. Masking is the
only deviation from the paper's unmasked row. -/
theorem sparseWeights_exists_threshold {T : ℕ} (scores : Fin T → ℝ)
    (i : Fin T) : ∃ τ : ℝ, sparseWeights scores i = thresholdWeights scores i τ := by
  obtain ⟨τ, hτ⟩ := thresholdWeights_exists scores i
  exact ⟨τ, (thresholdWeights_eq_sparseWeights scores i τ hτ).symm⟩

end Transformer.GPTMini.Sparsemax
