/-
# Removing the factor `n`: a lattice counting bound

`softmax_winner_ge` bounds every one of the `n-1` competitors by the single
worst gap `δ`, which throws away the lattice structure.  For integer keys the
true competitor sum is

  `∑_{j ≠ i₀} exp(-β ‖k_j - q‖²)`,

and distinct integer keys put a bounded number of keys at each distance `t`
from `q`.  Summing the resulting geometric series gives a bound that does not
mention `n` at all.

The consequence is architectural: the inverse temperature `β` needed for an
exact simulation does **not** have to grow with the length of the trace.
-/

import Transformer.ALM.Softmax
import Mathlib.Algebra.Field.GeomSum

open scoped BigOperators
open Real

namespace Transformer
namespace ALM

/-! ### A geometric tail -/

/-- `∑_{t=1}^{N} r^t ≤ r/(1-r)` for `0 ≤ r < 1`. -/
lemma geom_pos_le {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r < 1) (N : ℕ) :
    ∑ t ∈ (Finset.range (N + 1)).erase 0, r ^ t ≤ r / (1 - r) := by
  have h1r : (0 : ℝ) < 1 - r := by linarith
  have hfull : ∑ t ∈ Finset.range (N + 1), r ^ t ≤ 1 / (1 - r) := by
    rw [geom_sum_eq (by linarith : r ≠ 1)]
    have hrw : (r ^ (N + 1) - 1) / (r - 1) = (1 - r ^ (N + 1)) / (1 - r) := by
      rw [← neg_div_neg_eq]; ring_nf
    rw [hrw]
    have hkey : 1 / (1 - r) - (1 - r ^ (N + 1)) / (1 - r) = r ^ (N + 1) / (1 - r) := by
      field_simp
      ring
    have hpos : 0 ≤ r ^ (N + 1) / (1 - r) :=
      div_nonneg (pow_nonneg hr0 _) (le_of_lt h1r)
    linarith
  have hmem : (0 : ℕ) ∈ Finset.range (N + 1) := Finset.mem_range.mpr (Nat.succ_pos N)
  have hsplit : ∑ t ∈ Finset.range (N + 1), r ^ t
      = r ^ 0 + ∑ t ∈ (Finset.range (N + 1)).erase 0, r ^ t :=
    (Finset.add_sum_erase (Finset.range (N + 1)) (fun t => r ^ t) hmem).symm
  simp only [pow_zero] at hsplit
  rw [hsplit] at hfull
  have hid : (1 : ℝ) / (1 - r) - 1 = r / (1 - r) := by
    field_simp
    ring
  linarith

/-! ### The counting bound -/

/-- **Lattice counting bound.**  Suppose every competitor `j ≠ i₀` loses by at
least an integer amount `g j ≥ 1`, and at most `M` competitors share any given
value of `g`.  Then the competitor sum is at most `M · e^{-β}/(1-e^{-β})`,
*independently of `n`*. -/
lemma sum_exp_gap_le {n : ℕ} (β : ℝ) (hβ : 0 < β)
    (s : Fin n → ℝ) (i₀ : Fin n) (g : Fin n → ℕ) (M : ℕ)
    (hgap : ∀ j, j ≠ i₀ → s j + (g j : ℝ) ≤ s i₀)
    (hg1 : ∀ j, j ≠ i₀ → 1 ≤ g j)
    (hM : ∀ t : ℕ, ((Finset.univ.erase i₀).filter (fun j => g j = t)).card ≤ M) :
    ∑ j ∈ Finset.univ.erase i₀, Real.exp (β * (s j - s i₀))
      ≤ (M : ℝ) * (Real.exp (-β) / (1 - Real.exp (-β))) := by
  set E := Finset.univ.erase i₀ with hE
  set r := Real.exp (-β) with hr
  have hr0 : 0 < r := Real.exp_pos _
  have hr1 : r < 1 := by
    rw [hr]; exact Real.exp_lt_one_iff.mpr (by linarith)
  -- Step 1: each term is at most `r ^ (g j)`.
  have hterm : ∀ j ∈ E, Real.exp (β * (s j - s i₀)) ≤ r ^ (g j) := by
    intro j hj
    have hjne : j ≠ i₀ := Finset.ne_of_mem_erase hj
    have h1 : β * (s j - s i₀) ≤ (g j : ℝ) * (-β) := by
      have := hgap j hjne
      nlinarith [le_of_lt hβ]
    calc Real.exp (β * (s j - s i₀)) ≤ Real.exp ((g j : ℝ) * (-β)) := Real.exp_le_exp.mpr h1
      _ = r ^ (g j) := by rw [hr, ← Real.exp_nat_mul]
  have hstep1 : ∑ j ∈ E, Real.exp (β * (s j - s i₀)) ≤ ∑ j ∈ E, r ^ (g j) :=
    Finset.sum_le_sum hterm
  -- Step 2: group the right-hand sum by the value of `g`.
  set N := E.sup g with hN
  have hmaps : ∀ j ∈ E, g j ∈ Finset.range (N + 1) := by
    intro j hj
    exact Finset.mem_range.mpr (Nat.lt_succ_of_le (Finset.le_sup hj))
  have hfib : ∑ j ∈ E, r ^ (g j)
      = ∑ t ∈ Finset.range (N + 1), ((E.filter (fun j => g j = t)).card : ℝ) * r ^ t := by
    rw [← Finset.sum_fiberwise_of_maps_to' hmaps (fun t => r ^ t)]
    refine Finset.sum_congr rfl fun t _ => ?_
    rw [Finset.sum_const, nsmul_eq_mul]
  -- Step 3: the fiber over `0` is empty, so the `t = 0` term vanishes.
  have hzero : (E.filter (fun j => g j = 0)).card = 0 := by
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro j hj
    have := hg1 j (Finset.ne_of_mem_erase hj)
    omega
  have hmem0 : (0 : ℕ) ∈ Finset.range (N + 1) := Finset.mem_range.mpr (Nat.succ_pos N)
  have hdrop : ∑ t ∈ Finset.range (N + 1), ((E.filter (fun j => g j = t)).card : ℝ) * r ^ t
      = ∑ t ∈ (Finset.range (N + 1)).erase 0,
          ((E.filter (fun j => g j = t)).card : ℝ) * r ^ t := by
    rw [← Finset.add_sum_erase _ _ hmem0, hzero]
    simp
  -- Step 4: each fiber has at most `M` elements; sum the geometric series.
  have hbound : ∑ t ∈ (Finset.range (N + 1)).erase 0,
        ((E.filter (fun j => g j = t)).card : ℝ) * r ^ t
      ≤ ∑ t ∈ (Finset.range (N + 1)).erase 0, (M : ℝ) * r ^ t := by
    refine Finset.sum_le_sum fun t _ => ?_
    have hcard : ((E.filter (fun j => g j = t)).card : ℝ) ≤ (M : ℝ) := by
      exact_mod_cast hM t
    exact mul_le_mul_of_nonneg_right hcard (pow_nonneg (le_of_lt hr0) t)
  have hgeo : ∑ t ∈ (Finset.range (N + 1)).erase 0, (M : ℝ) * r ^ t
      ≤ (M : ℝ) * (r / (1 - r)) := by
    rw [← Finset.mul_sum]
    exact mul_le_mul_of_nonneg_left (geom_pos_le (le_of_lt hr0) hr1 N) (Nat.cast_nonneg M)
  calc ∑ j ∈ E, Real.exp (β * (s j - s i₀)) ≤ ∑ j ∈ E, r ^ (g j) := hstep1
    _ = _ := hfib
    _ = _ := hdrop
    _ ≤ _ := hbound
    _ ≤ (M : ℝ) * (r / (1 - r)) := hgeo

/-! ### The length-free bound -/

/-- **Length-free softmax bound.**  Combining `softmax_winner_sharp` with the
counting bound: if competitors lose by integer margins with multiplicity at
most `M`, the winner's softmax weight is bounded below by a quantity that does
not involve `n`. -/
theorem softmax_winner_lengthfree {n : ℕ} (β : ℝ) (hβ : 0 < β)
    (s : Fin n → ℝ) (i₀ : Fin n) (g : Fin n → ℕ) (M : ℕ)
    (hgap : ∀ j, j ≠ i₀ → s j + (g j : ℝ) ≤ s i₀)
    (hg1 : ∀ j, j ≠ i₀ → 1 ≤ g j)
    (hM : ∀ t : ℕ, ((Finset.univ.erase i₀).filter (fun j => g j = t)).card ≤ M) :
    1 / (1 + (M : ℝ) * (Real.exp (-β) / (1 - Real.exp (-β))))
      ≤ Real.exp (β * s i₀) / ∑ j, Real.exp (β * s j) :=
  softmax_winner_sharp β s i₀ _ (sum_exp_gap_le β hβ s i₀ g M hgap hg1 hM)

end ALM
end Transformer
