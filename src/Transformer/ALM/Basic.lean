/-
# Exact keyed lookup, in arbitrary key dimension

Basic API for `Transformer.ALM.score`.  Two facts carry the construction:

* `score_gap` / `score_lt_of_ne` — the score is *uniquely* maximized at
  `k = q`, with an exact deficiency `‖k - q‖²`.  This is the hard-max
  statement, and it holds in every key dimension; the blog states the
  generalization to dimension `3` only, as a remark about convex-hull
  efficiency.
* `one_le_dist_sq_of_int` — on an integer lattice distinct keys are separated
  by squared distance at least `1`, so the score gap is uniformly `≥ 1`.
  This is what makes the construction *exact* rather than approximate.

The softmax counterpart is `Transformer.ALM.Softmax`.
-/

import Transformer.ALM.Defs

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m : ℕ}

/-! ### Uniqueness of the argmax -/

/-- The score at an exact key match is `‖q‖²`. -/
lemma score_self (q : EucSpace m) : score q q = ‖q‖ ^ 2 := by
  unfold score
  rw [real_inner_self_eq_norm_sq]
  ring

/-- **Exact deficiency.**  The amount by which any key `k` falls short of the
matching key `q` is exactly the squared distance:

  `score q q - score q k = ‖k - q‖²`. -/
theorem score_gap (q k : EucSpace m) :
    score q q - score q k = ‖k - q‖ ^ 2 := by
  rw [score_self]
  unfold score
  rw [norm_sub_sq_real]
  ring

/-- **Uniqueness of the argmax.**  For `k ≠ q` the score is strictly smaller.
This is the exact-lookup property of the head. -/
theorem score_lt_of_ne {q k : EucSpace m} (hne : k ≠ q) :
    score q k < score q q := by
  have hpos : 0 < ‖k - q‖ ^ 2 := by
    have : k - q ≠ 0 := sub_ne_zero_of_ne hne
    positivity
  have := score_gap q k
  linarith

/-- The matching key is the greatest element for `score q`. -/
theorem score_isGreatest (q : EucSpace m) :
    ∀ k : EucSpace m, score q k ≤ score q q := by
  intro k
  rcases eq_or_ne k q with rfl | hne
  · exact le_refl _
  · exact le_of_lt (score_lt_of_ne hne)

/-! ### Integer keys give a gap of at least one -/

/-- The squared norm of a Euclidean vector is the sum of squared coordinates. -/
lemma norm_sq_eq_sum (x : EucSpace m) : ‖x‖ ^ 2 = ∑ i, (x i) ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  simp [Real.norm_eq_abs, sq_abs]

/-- **Unit gap on the integer lattice.**  If two keys have integer
coordinates and differ, their squared distance is at least `1`. -/
theorem one_le_dist_sq_of_int (k q : EucSpace m)
    (hk : ∀ i, ∃ z : ℤ, k i = (z : ℝ))
    (hq : ∀ i, ∃ z : ℤ, q i = (z : ℝ))
    (hne : k ≠ q) :
    1 ≤ ‖k - q‖ ^ 2 := by
  obtain ⟨i, hi⟩ : ∃ i, k i ≠ q i := by
    by_contra h
    push Not at h
    exact hne (by ext i; exact h i)
  obtain ⟨a, ha⟩ := hk i
  obtain ⟨b, hb⟩ := hq i
  have hab : a ≠ b := by
    intro h; apply hi; rw [ha, hb, h]
  -- the `i`-th coordinate of `k - q` is the nonzero integer `a - b`
  have hcoord : (k - q) i = (a : ℝ) - (b : ℝ) := by
    simp [ha, hb]
  have hsq : (1 : ℝ) ≤ ((a : ℝ) - (b : ℝ)) ^ 2 := by
    have hz : (1 : ℤ) ≤ (a - b) ^ 2 := by
      rcases lt_trichotomy (a - b) 0 with h | h | h
      · nlinarith
      · exact absurd h (sub_ne_zero_of_ne hab)
      · nlinarith
    have hr := (Int.cast_le (R := ℝ)).mpr hz
    push_cast at hr
    linarith
  have hone : (1 : ℝ) ≤ ((k - q) i) ^ 2 := by
    rw [hcoord]; exact hsq
  calc (1 : ℝ) ≤ ((k - q) i) ^ 2 := hone
    _ ≤ ∑ j, ((k - q) j) ^ 2 := by
        exact Finset.single_le_sum (f := fun j => ((k - q) j) ^ 2)
          (fun j _ => sq_nonneg _) (Finset.mem_univ i)
    _ = ‖k - q‖ ^ 2 := (norm_sq_eq_sum _).symm

end ALM
end Transformer
