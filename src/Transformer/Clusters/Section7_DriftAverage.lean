/-
# The emergence of clusters in self-attention dynamics — drift averages

Two estimates of §7 of arXiv:2305.05465v6 on the drift
`Σ_j P_ij x_j` of `e:Idnonresca` in `d = 1`, used by `l:exactasymptotic`:
it never exceeds the largest coordinate (`drift_le`), and a positive token
moves almost at the speed of the largest one (`sub_drift_le`).  With them, a
monotonicity criterion on `[T, ∞)` (`antitoneOn_Ici_of_hasDerivAt`).

Source: arXiv:2305.05465v6, proof of `l:exactasymptotic`.
-/

import Transformer.Clusters.Section7_UnboundedParticles

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- The drift is an average: `Σ_j P_ij y_j ≤ y_N` when `y_N` is the largest
coordinate. -/
theorem drift_le (y : Idx (m + 1) → ℝ) (i N : Idx (m + 1)) (hmax : ∀ j, y j ≤ y N) :
    ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * y j ≤ y N := by
  calc ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * y j
      ≤ ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * y N :=
        Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_left (hmax j) (Perspective.softmaxWeight_nonneg _ _)
    _ = y N := by
      rw [← Finset.sum_mul, Perspective.sum_softmaxWeight (Nat.succ_pos m), one_mul]

/-- A positive token moves almost at the speed of the largest one:
`y_N - Σ_j P_ij y_j ≤ n / y_i` for `y_i > 0`, since
`P_ij (y_N - y_j) ≤ e^{-y_i w} w ≤ 1/y_i` with `w = y_N - y_j`.

Source: arXiv:2305.05465v6, proof of `l:exactasymptotic`. -/
theorem sub_drift_le (y : Idx (m + 1) → ℝ) (i N : Idx (m + 1)) (hy : 0 < y i)
    (hmax : ∀ j, y j ≤ y N) :
    y N - ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * y j ≤ (m + 1) / y i := by
  have hs := Perspective.sum_softmaxWeight (Nat.succ_pos m) (fun l => y i * y l)
  have hD : y N - ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * y j =
      ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * (y N - y j) := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hs, one_mul]
  have hj : ∀ j, Perspective.softmaxWeight (fun l => y i * y l) j * (y N - y j) ≤ 1 / y i := by
    intro j
    have h1 := neg_inv_le_exp_mul_mul hy (y j - y N)
    have h2 := mul_le_mul_of_nonneg_right (softmaxWeight_le_exp_sub (fun l => y i * y l) N j)
      (sub_nonneg.2 (hmax j))
    have h3 : Real.exp (y i * y j - y i * y N) = Real.exp (y i * (y j - y N)) := by ring_nf
    rw [h3] at h2
    nlinarith
  rw [hD]
  calc _ ≤ ∑ _j : Idx (m + 1), 1 / y i := Finset.sum_le_sum fun j _ => hj j
    _ = (m + 1) / y i := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      push_cast; ring

/-- A function with a nonpositive derivative on `[T, ∞)` does not increase
there. -/
theorem antitoneOn_Ici_of_hasDerivAt {f f' : ℝ → ℝ} {T : ℝ}
    (hf : ∀ t, T ≤ t → HasDerivAt f (f' t) t) (h0 : ∀ t, T ≤ t → f' t ≤ 0) :
    AntitoneOn f (Set.Ici T) := by
  refine antitoneOn_of_hasDerivWithinAt_nonpos (f' := f') (convex_Ici T)
    (fun t ht => (hf t ht).continuousAt.continuousWithinAt) ?_ ?_
  · intro t ht
    rw [interior_Ici] at ht
    exact (hf t (le_of_lt ht)).hasDerivWithinAt
  · intro t ht
    rw [interior_Ici] at ht
    exact h0 t (le_of_lt ht)

/-- The hypotheses of `antitoneOn_Ici_of_hasDerivAt` are satisfiable: a
constant. -/
example : ∀ t : ℝ, (0 : ℝ) ≤ t → HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t :=
  fun t _ => hasDerivAt_const t 1

/-- The hypotheses of `drift_le` and `sub_drift_le` are satisfiable: one
positive coordinate. -/
example : ∃ y : Idx 1 → ℝ, 0 < y 0 ∧ ∀ j, y j ≤ y 0 := ⟨fun _ => 1, one_pos, fun _ => le_rfl⟩

end Clusters
end Transformer
