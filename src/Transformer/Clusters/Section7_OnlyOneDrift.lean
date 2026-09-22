/-
# The emergence of clusters in self-attention dynamics — the drift toward the largest token

The estimate `e:lowerboundxi` of §7 of arXiv:2305.05465v6, for a token that
need not be the largest: if `y_k > 0` and `y_M` is the largest coordinate,
the weight `P_kM` is at least `1/n`, and every other term `P_kj y_j` is at
least `min(0, e^{y_k y_j} y_j)` (`add_sum_min_le_drift`).  The two
specializations used by `l:onlyone` follow: `e^{az} z ≥ -1/a` gives
`Σ_j P_kj y_j ≥ y_M/n - n/y_k` (`div_sub_le_drift_of_max`), the form of
`e:lowerboundxi` itself.

Source: arXiv:2305.05465v6, `e:lowerboundxi`, `e:minorationpourxn`.
-/

import Transformer.Clusters.Section7_DriftAverage

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- Under nonnegative scores `a y_l` the largest coordinate `y_M` carries
weight at least `1/n`. -/
theorem inv_le_softmaxWeight_of_max (y : Idx (m + 1) → ℝ) {a : ℝ} (ha : 0 ≤ a)
    (M : Idx (m + 1)) (hM : ∀ j, y j ≤ y M) :
    1 / ((m : ℝ) + 1) ≤ Perspective.softmaxWeight (fun l => a * y l) M := by
  have hZ : ∑ k : Idx (m + 1), Real.exp (a * y k) ≤ ((m : ℝ) + 1) * Real.exp (a * y M) := by
    have : ∑ k : Idx (m + 1), Real.exp (a * y k) ≤ ∑ _k : Idx (m + 1), Real.exp (a * y M) :=
      Finset.sum_le_sum fun k _ => Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left (hM k) ha)
    simpa using this
  have hZ0 := Perspective.softmaxPartition_pos (Nat.succ_pos m) (fun l => a * y l)
  show 1 / ((m : ℝ) + 1) ≤ Real.exp (a * y M) / ∑ k : Idx (m + 1), Real.exp (a * y k)
  rw [div_le_div_iff₀ (by positivity) hZ0]
  linarith

/-- If some score `a y_M` is nonnegative, every weight is at most `e^{a y_j}`:
the partition function is at least `e^{a y_M} ≥ 1`. -/
theorem softmaxWeight_le_exp_of_nonneg (y : Idx (m + 1) → ℝ) (a : ℝ) (M : Idx (m + 1))
    (hM : 0 ≤ a * y M) (j : Idx (m + 1)) :
    Perspective.softmaxWeight (fun l => a * y l) j ≤ Real.exp (a * y j) := by
  have hZ : 1 ≤ ∑ k : Idx (m + 1), Real.exp (a * y k) :=
    (Real.one_le_exp hM).trans (Finset.single_le_sum (f := fun k => Real.exp (a * y k))
      (fun k _ => (Real.exp_pos _).le) (Finset.mem_univ M))
  show Real.exp (a * y j) / ∑ k : Idx (m + 1), Real.exp (a * y k) ≤ Real.exp (a * y j)
  exact div_le_self (Real.exp_pos _).le hZ

/-- **The drift of a positive token, toward the largest one.**  With `y_k > 0`
and `y_M` the largest coordinate,
`Σ_j P_kj y_j ≥ y_M/n + Σ_j min(0, e^{y_k y_j} y_j)`.

Source: arXiv:2305.05465v6, `e:lowerboundxi`, `e:minorationpourxn`. -/
theorem add_sum_min_le_drift (y : Idx (m + 1) → ℝ) (k M : Idx (m + 1)) (hk : 0 < y k)
    (hM : ∀ j, y j ≤ y M) :
    y M / ((m : ℝ) + 1) + ∑ j, min 0 (Real.exp (y k * y j) * y j) ≤
      ∑ j, Perspective.softmaxWeight (fun l => y k * y l) j * y j := by
  set p := Perspective.softmaxWeight (fun l => y k * y l)
  have hyM : 0 < y M := hk.trans_le (hM k)
  have hle : ∀ j, p j ≤ Real.exp (y k * y j) :=
    softmaxWeight_le_exp_of_nonneg y (y k) M (mul_pos hk hyM).le
  have hterm : ∀ j, (if j = M then y M / ((m : ℝ) + 1) else 0) +
      min 0 (Real.exp (y k * y j) * y j) ≤ p j * y j := by
    intro j
    by_cases hj : j = M
    · subst hj
      have h1 := inv_le_softmaxWeight_of_max y hk.le j hM
      have h2 : min 0 (Real.exp (y k * y j) * y j) = 0 :=
        min_eq_left (mul_pos (Real.exp_pos _) hyM).le
      rw [ite_eq_left rfl, h2, add_zero, div_eq_mul_one_div, mul_comm (p j)]
      exact mul_le_mul_of_nonneg_left h1 hyM.le
    · rw [ite_eq_right hj, zero_add]
      rcases le_or_gt 0 (y j) with h | h
      · exact (min_le_left _ _).trans (mul_nonneg (Perspective.softmaxWeight_nonneg _ _) h)
      · exact (min_le_right _ _).trans (mul_le_mul_of_nonpos_right (hle j) h.le)
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hterm j
  rwa [Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ M, ite_eq_left (Finset.mem_univ M)]
    at hsum

/-- **`e:lowerboundxi`.**  With `y_k > 0` and `y_M` the largest coordinate,
`Σ_j P_kj y_j ≥ y_M/n - n/y_k`.

Source: arXiv:2305.05465v6, `e:lowerboundxi`. -/
theorem div_sub_le_drift_of_max (y : Idx (m + 1) → ℝ) (k M : Idx (m + 1)) (hk : 0 < y k)
    (hM : ∀ j, y j ≤ y M) :
    y M / ((m : ℝ) + 1) - ((m : ℝ) + 1) / y k ≤
      ∑ j, Perspective.softmaxWeight (fun l => y k * y l) j * y j := by
  have h := add_sum_min_le_drift y k M hk hM
  have hj : ∀ j, -(1 / y k) ≤ min 0 (Real.exp (y k * y j) * y j) := fun j =>
    le_min (neg_nonpos.2 (by positivity)) (neg_inv_le_exp_mul_mul hk _)
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hj j
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
  push_cast at hsum
  have : ((m : ℝ) + 1) * -(1 / y k) = -(((m : ℝ) + 1) / y k) := by ring
  linarith

/-- The hypotheses of `add_sum_min_le_drift` and `div_sub_le_drift_of_max`
are satisfiable: one positive coordinate. -/
example : ∃ y : Idx 1 → ℝ, 0 < y 0 ∧ ∀ j, y j ≤ y 0 := ⟨fun _ => 1, one_pos, fun _ => le_rfl⟩

end Clusters
end Transformer
