/-
# Attention's forward pass and Frank-Wolfe — the score of another group

`lem: metastab.1`, `cl: group.cluster.bound`: when every particle sits near its own vertex, the
attention scores a particle gives to particles of another group add up to at most
`n e^{-β c₀ / 2}`.  The proof compares each such score with the score of the particle for itself.
-/

import Transformer.FrankWolfe.Section5_Process

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- A particle near its vertex `v_a` scores another particle near a different vertex `v_b` below
itself by at least `c₀ - 4Rr - r²`:

  `⟨x, x'⟩ - ⟨x, x⟩ ≤ -c₀ + 4Rr + r²`

when `⟨v_a, v_b⟩ + c₀ ≤ ⟨v_a, v_a⟩`, `‖v_a‖, ‖v_b‖ ≤ R` and `‖x - v_a‖, ‖x' - v_b‖ ≤ r`. -/
theorem inner_sub_self_le {va vb x x' : EucSpace d} {c₀ R r : ℝ}
    (hsep : inner (𝕜 := ℝ) va vb + c₀ ≤ inner (𝕜 := ℝ) va va) (hva : ‖va‖ ≤ R) (hvb : ‖vb‖ ≤ R)
    (hx : ‖x - va‖ ≤ r) (hx' : ‖x' - vb‖ ≤ r) :
    inner (𝕜 := ℝ) x x' - inner (𝕜 := ℝ) x x ≤ -c₀ + 4 * R * r + r ^ 2 := by
  obtain ⟨y, rfl⟩ : ∃ y, x = va + y := ⟨x - va, by abel⟩
  obtain ⟨y', rfl⟩ : ∃ y', x' = vb + y' := ⟨x' - vb, by abel⟩
  have hy : ‖y‖ ≤ r := by simpa using hx
  have hy' : ‖y'‖ ≤ r := by simpa using hx'
  have hR : 0 ≤ R := (norm_nonneg _).trans hva
  have hr : 0 ≤ r := (norm_nonneg _).trans hy
  have e1 : inner (𝕜 := ℝ) va y' ≤ R * r :=
    (real_inner_le_norm _ _).trans (mul_le_mul hva hy' (norm_nonneg _) hR)
  have e2 : inner (𝕜 := ℝ) y vb ≤ r * R :=
    (real_inner_le_norm _ _).trans (mul_le_mul hy hvb (norm_nonneg _) hr)
  have e3 : inner (𝕜 := ℝ) y y' ≤ r * r :=
    (real_inner_le_norm _ _).trans (mul_le_mul hy hy' (norm_nonneg _) hr)
  have e4 : -inner (𝕜 := ℝ) y va ≤ r * R :=
    (neg_le_abs _).trans ((abs_real_inner_le_norm _ _).trans
      (mul_le_mul hy hva (norm_nonneg _) hr))
  have e5 : 0 ≤ inner (𝕜 := ℝ) y y := real_inner_self_nonneg
  simp only [inner_add_left, inner_add_right, real_inner_comm y va]
  nlinarith

/-- The score of particle `i` for particle `j` is at most `e^{β(⟨x_i, x_j⟩ - ⟨x_i, x_i⟩)}`: the
partition function contains the term of `j = i`. -/
theorem attWeight_le_exp_sub (β : ℝ) (X : Idx n → EucSpace d) (i j : Idx n) :
    attWeight β X i j ≤
      Real.exp (β * (inner (𝕜 := ℝ) (X i) (X j) - inner (𝕜 := ℝ) (X i) (X i))) := by
  unfold attWeight
  have hpos : 0 < ∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (X i) (X k)) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩
  rw [div_le_iff₀ hpos]
  have hi : Real.exp (β * inner (𝕜 := ℝ) (X i) (X i)) ≤
      ∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (X i) (X k)) :=
    Finset.single_le_sum (f := fun k => Real.exp (β * inner (𝕜 := ℝ) (X i) (X k)))
      (fun _ _ => (Real.exp_pos _).le) (Finset.mem_univ i)
  calc Real.exp (β * inner (𝕜 := ℝ) (X i) (X j))
      = Real.exp (β * (inner (𝕜 := ℝ) (X i) (X j) - inner (𝕜 := ℝ) (X i) (X i))) *
          Real.exp (β * inner (𝕜 := ℝ) (X i) (X i)) := by
        rw [← Real.exp_add]
        congr 1
        ring
    _ ≤ _ := mul_le_mul_of_nonneg_left hi (Real.exp_pos _).le

/-- **`cl: group.cluster.bound`.**  If every particle is within `r` of its own vertex, where
`4Rr + 2r² ≤ c₀/2`, the scores that particle `i` gives to particles attached to another vertex
add up to at most `n e^{-β c₀ / 2}`. -/
theorem sum_attWeight_other_le {v : Idx κ → EucSpace d} {σ : Idx n → Idx κ} {c₀ R r β : ℝ}
    (hsep : ∀ a b : Idx κ, a ≠ b →
      inner (𝕜 := ℝ) (v a) (v b) + c₀ ≤ inner (𝕜 := ℝ) (v a) (v a))
    (hR : ∀ a, ‖v a‖ ≤ R) (hsmall : 4 * R * r + 2 * r ^ 2 ≤ c₀ / 2) (hβ : 0 ≤ β)
    (z : Idx n → EucSpace d) (hz : ∀ j, ‖z j - v (σ j)‖ ≤ r) (i : Idx n) :
    ∑ j ∈ Finset.univ.filter (fun j => σ j ≠ σ i), attWeight β z i j ≤
      n * Real.exp (-(β * (c₀ / 2))) := by
  have hterm : ∀ j ∈ Finset.univ.filter (fun j => σ j ≠ σ i),
      attWeight β z i j ≤ Real.exp (-(β * (c₀ / 2))) := by
    intro j hj
    have hne : σ j ≠ σ i := (Finset.mem_filter.mp hj).2
    refine (attWeight_le_exp_sub β z i j).trans (Real.exp_le_exp.mpr ?_)
    have h1 := inner_sub_self_le (hsep (σ i) (σ j) hne.symm) (hR _) (hR _) (hz i) (hz j)
    nlinarith [sq_nonneg r, mul_le_mul_of_nonneg_left
      (show -c₀ + 4 * R * r + r ^ 2 ≤ -(c₀ / 2) by nlinarith [sq_nonneg r]) hβ]
  calc ∑ j ∈ Finset.univ.filter (fun j => σ j ≠ σ i), attWeight β z i j
      ≤ (Finset.univ.filter (fun j => σ j ≠ σ i)).card • Real.exp (-(β * (c₀ / 2))) :=
        Finset.sum_le_card_nsmul _ _ _ hterm
    _ ≤ n * Real.exp (-(β * (c₀ / 2))) := by
        rw [nsmul_eq_mul]
        refine mul_le_mul_of_nonneg_right ?_ (Real.exp_pos _).le
        exact_mod_cast (Finset.card_filter_le _ _).trans (by simp)

end FrankWolfe
end Transformer
