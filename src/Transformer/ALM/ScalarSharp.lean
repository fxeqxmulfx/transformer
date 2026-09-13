/-
# The bound the quadratic gap actually gives

`softmax_winner_scalar_int` of `Transformer.ALM.ScalarInt` feeds the counting
lemma the *linear* margin `|k_j - q|`, while the exact margin available from
`score_gap` is the *square* `(k_j - q)²`.  That discards a square, and the
discarded amount is not asymptotically negligible at the temperatures the
machine actually runs at.

This file redoes the estimate with the quadratic margin.  The competitor sum
becomes a theta-function tail `2 ∑_{t ≥ 1} r^{t²}` instead of a geometric
series `2 ∑_{t ≥ 1} r^t`, and `t² ≥ 3t - 2` bounds it by `2r/(1 - r³)`:

  `w i₀ ≥ 1 / (1 + 2 e^{-β}/(1 - e^{-3β}))`.

`scalar_int_sharp_lt` records that this is strictly better than the bound of
`ScalarInt` at every temperature, and the final example makes the size of the
gain concrete at `β = log 3`, the point where the old bound first reaches
`1/2`: there the new bound already gives `13/22`.

Only two inequalities remain in the chain, and both are tight: at most two
integer keys sit at any distance from the query, and `t² ≥ 3t - 2` is an
equality at `t = 1` and `t = 2`.
-/

import Transformer.ALM.ScalarInt

open scoped BigOperators
open Real

namespace Transformer
namespace ALM

/-! ### The competitor sum under the quadratic margin -/

/-- **The exact competitor sum, bounded.**  For distinct integer scalar keys
the term of `j` is *exactly* `r^{(k_j - q)²}`, not merely at most `r^{|k_j-q|}`.
Grouping by distance — at most two keys per distance — and applying `geom_sq_le`
gives a bound with `1 - r³` in the denominator. -/
theorem sum_exp_sq_gap_le {n : ℕ} (β : ℝ) (hβ : 0 < β)
    (K : Fin n → ℤ) (hinj : Function.Injective K) (i₀ : Fin n) :
    ∑ j ∈ Finset.univ.erase i₀,
        Real.exp (β * (sScore (K i₀) (K j) - sScore (K i₀) (K i₀)))
      ≤ 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3)) := by
  set E := Finset.univ.erase i₀ with hE
  set r := Real.exp (-β) with hr
  have hr0 : 0 < r := Real.exp_pos _
  have hr1 : r < 1 := by rw [hr]; exact Real.exp_lt_one_iff.mpr (by linarith)
  set d : Fin n → ℕ := fun j => (K j - K i₀).natAbs with hd
  have hdcast : ∀ j, ((d j : ℝ)) ^ 2 = ((K j : ℝ) - (K i₀ : ℝ)) ^ 2 := by
    intro j
    rw [hd]
    push_cast [Nat.cast_natAbs]
    rw [sq_abs]
  -- every term is exactly a power of `r`
  have hterm : ∀ j ∈ E, Real.exp (β * (sScore (K i₀) (K j) - sScore (K i₀) (K i₀)))
      = r ^ ((d j) ^ 2) := by
    intro j _
    have hgap : sScore (K i₀) (K j) - sScore (K i₀) (K i₀)
        = -(((K j : ℝ) - (K i₀ : ℝ)) ^ 2) := by simp only [sScore]; ring
    rw [hgap, ← hdcast j, hr, ← Real.exp_nat_mul]
    congr 1
    push_cast
    ring
  rw [Finset.sum_congr rfl hterm]
  -- group by distance
  set N := E.sup d with hN
  have hmaps : ∀ j ∈ E, d j ∈ Finset.range (N + 1) := fun j hj =>
    Finset.mem_range.mpr (Nat.lt_succ_of_le (Finset.le_sup hj))
  have hfib : ∑ j ∈ E, r ^ ((d j) ^ 2)
      = ∑ t ∈ Finset.range (N + 1),
          ((E.filter (fun j => d j = t)).card : ℝ) * r ^ (t ^ 2) := by
    rw [← Finset.sum_fiberwise_of_maps_to' hmaps (fun t => r ^ (t ^ 2))]
    refine Finset.sum_congr rfl fun t _ => ?_
    rw [Finset.sum_const, nsmul_eq_mul]
  have hzero : (E.filter (fun j => d j = 0)).card = 0 := by
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro j hj
    have hjne : j ≠ i₀ := Finset.ne_of_mem_erase hj
    have : d j ≠ 0 := by
      rw [hd]
      simpa [Int.natAbs_eq_zero, sub_eq_zero] using fun h => hjne (hinj h)
    exact this
  have hmem0 : (0 : ℕ) ∈ Finset.range (N + 1) := Finset.mem_range.mpr (Nat.succ_pos N)
  have hdrop : ∑ t ∈ Finset.range (N + 1),
        ((E.filter (fun j => d j = t)).card : ℝ) * r ^ (t ^ 2)
      = ∑ t ∈ (Finset.range (N + 1)).erase 0,
          ((E.filter (fun j => d j = t)).card : ℝ) * r ^ (t ^ 2) := by
    rw [← Finset.add_sum_erase _ _ hmem0, hzero]
    simp
  -- at most two integer keys at any given distance
  have hM : ∀ t : ℕ, ((E.filter (fun j => d j = t)).card : ℝ) ≤ 2 := by
    intro t
    have hcard : (E.filter (fun j => d j = t)).card
        ≤ ({K i₀ - (t : ℤ), K i₀ + (t : ℤ)} : Finset ℤ).card := by
      refine Finset.card_le_card_of_injOn K ?_ ?_
      · intro j hj
        have ht : (K j - K i₀).natAbs = t := (Finset.mem_filter.mp hj).2
        rcases Int.natAbs_eq (K j - K i₀) with h | h
        · have heq : K j = K i₀ + (t : ℤ) := by omega
          rw [heq]; exact Finset.mem_insert_of_mem (Finset.mem_singleton.mpr rfl)
        · have heq : K j = K i₀ - (t : ℤ) := by omega
          rw [heq]; exact Finset.mem_insert_self _ _
      · intro a _ b _ h; exact hinj h
    have : (E.filter (fun j => d j = t)).card ≤ 2 :=
      le_trans hcard (le_trans (Finset.card_insert_le _ _) (by simp))
    exact_mod_cast this
  have hbound : ∑ t ∈ (Finset.range (N + 1)).erase 0,
        ((E.filter (fun j => d j = t)).card : ℝ) * r ^ (t ^ 2)
      ≤ ∑ t ∈ (Finset.range (N + 1)).erase 0, 2 * r ^ (t ^ 2) :=
    Finset.sum_le_sum fun t _ =>
      mul_le_mul_of_nonneg_right (hM t) (pow_nonneg hr0.le _)
  calc ∑ j ∈ E, r ^ ((d j) ^ 2) = _ := hfib
    _ = _ := hdrop
    _ ≤ _ := hbound
    _ = 2 * ∑ t ∈ (Finset.range (N + 1)).erase 0, r ^ (t ^ 2) := by rw [Finset.mul_sum]
    _ ≤ 2 * (r / (1 - r ^ 3)) := by
        exact mul_le_mul_of_nonneg_left (geom_sq_le hr0.le hr1 N) (by norm_num)

/-! ### The sharp bound -/

/-- **The sharp length-free bound for the VM.**  For distinct integer scalar
keys,

  `w i₀ ≥ 1 / (1 + 2 e^{-β}/(1 - e^{-3β}))`,

with no dependence on the token count `n`.  This is `softmax_winner_scalar_int`
with the quadratic margin restored. -/
theorem softmax_winner_scalar_int_sharp {n : ℕ} (β : ℝ) (hβ : 0 < β)
    (K : Fin n → ℤ) (hinj : Function.Injective K) (i₀ : Fin n) :
    1 / (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3)))
      ≤ Real.exp (β * sScore (K i₀) (K i₀)) / ∑ j, Real.exp (β * sScore (K i₀) (K j)) :=
  softmax_winner_sharp β (fun j => sScore (K i₀) (K j)) i₀ _
    (sum_exp_sq_gap_le β hβ K hinj i₀)

/-- **Strictly better at every temperature.**  `r³ < r`, so the denominator
`1 - r³` is larger and the competitor budget smaller. -/
theorem scalar_int_sharp_lt {β : ℝ} (hβ : 0 < β) :
    1 / (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β))))
      < 1 / (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3))) := by
  set r := Real.exp (-β) with hr
  have hr0 : 0 < r := Real.exp_pos _
  have hr1 : r < 1 := by rw [hr]; exact Real.exp_lt_one_iff.mpr (by linarith)
  have h1 : (0 : ℝ) < 1 - r := by linarith
  have hcube : r ^ 3 < r := pow_lt_self_of_lt_one₀ hr0 hr1 (by norm_num)
  have h3 : (0 : ℝ) < 1 - r ^ 3 := by nlinarith
  have hlt : r / (1 - r ^ 3) < r / (1 - r) := by
    apply div_lt_div_of_pos_left hr0 h1
    linarith
  apply one_div_lt_one_div_of_lt
  · have : 0 < r / (1 - r ^ 3) := div_pos hr0 h3
    linarith
  · linarith

/-- The hypotheses are satisfiable: three distinct integer keys. -/
example : Function.Injective (fun j : Fin 3 => (j.val : ℤ)) := by
  intro a b h
  have hab : (a.val : ℤ) = (b.val : ℤ) := h
  exact Fin.ext (by exact_mod_cast hab)

/-- **How much is gained.**  At `β = log 3` the bound of `ScalarInt` reaches
exactly `1/2`; the bound proved here is already `13/22`. -/
example :
    1 / (1 + 2 * (Real.exp (-Real.log 3) / (1 - Real.exp (-Real.log 3)))) = 1 / 2 ∧
      1 / (1 + 2 * (Real.exp (-Real.log 3) / (1 - Real.exp (-Real.log 3) ^ 3)))
        = 13 / 22 := by
  have h3 : Real.exp (-Real.log 3) = 1 / 3 := by
    rw [Real.exp_neg, Real.exp_log (by norm_num)]
    norm_num
  rw [h3]
  norm_num

end ALM
end Transformer
