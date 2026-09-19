/-
# The equiangular configuration after one attention layer: closed forms

The three limits of `thm: long-context-phase-transition` are limits of
`equiOutCos`, which is defined by a double sum over the token indices.  This
module evaluates that double sum.

Everything rests on two features of the equiangular Gram matrix
`equiGram n ρ i j = if i = j then 1 else ρ`: the softmax normalizer
`Z = e^β + (n-1) e^{βρ}` does not depend on the row, so each row of the
attention matrix is the *same* two-valued vector up to a permutation --
`a = e^β / Z` on the diagonal, `b = e^{βρ} / Z` off it, with
`a + (n-1) b = 1` -- and `G = (1-ρ) I + ρ J`, so contracting a row of the
attention matrix against `G` returns `ρ + (1-ρ)` times that row.

The outcome is `equiOutCos_zero_one`: the cosine the theorem takes to the
limit is a rational function of `a` and `b` alone.

Source: arXiv:2512.01868v4, §6.
-/

import Transformer.MeanField.Equiangular

open scoped BigOperators

namespace Transformer
namespace MeanField

variable {n : ℕ} (β ρ : ℝ)

/-- The softmax normalizer `Z = e^β + (n-1) e^{βρ}` is positive as soon as
there is at least one token. -/
theorem equiNorm_pos (hn : 1 ≤ n) :
    0 < Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ) := by
  have h1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have : (0 : ℝ) ≤ ((n : ℝ) - 1) * Real.exp (β * ρ) :=
    mul_nonneg (by linarith) (Real.exp_pos _).le
  linarith [Real.exp_pos β]

/-- Every row of the unnormalized attention matrix sums to `Z`. -/
theorem sum_exp_equiGram (i : Idx n) :
    ∑ j : Idx n, Real.exp (β * equiGram n ρ i j)
      = Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ) := by
  have h : ∀ j : Idx n, Real.exp (β * equiGram n ρ i j)
      = Real.exp (β * ρ) + (if j = i then Real.exp β - Real.exp (β * ρ) else 0) := by
    intro j
    by_cases hj : j = i
    · subst hj; simp [equiGram]
    · simp [equiGram, hj, Ne.symm hj]
  rw [Finset.sum_congr rfl fun j _ => h j, Finset.sum_add_distrib, Finset.sum_const,
    Finset.sum_ite_eq' Finset.univ i (fun _ => Real.exp β - Real.exp (β * ρ))]
  simp [Finset.card_univ]
  ring

/-- Each row of the attention matrix sums to `1`. -/
theorem sum_equiWeight (i : Idx n) : ∑ j : Idx n, equiWeight n β ρ i j = 1 := by
  have hn : 1 ≤ n := i.pos
  have hZ := (equiNorm_pos β ρ hn).ne'
  simp only [equiWeight, ← Finset.sum_div, sum_exp_equiGram]
  exact div_self hZ

/-- The diagonal entry `a = e^β / Z` of the equiangular attention matrix. -/
noncomputable def equiDiag (n : ℕ) (β ρ : ℝ) : ℝ :=
  Real.exp β / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ))

/-- An off-diagonal entry `b = e^{βρ} / Z` of the equiangular attention
matrix. -/
noncomputable def equiOff (n : ℕ) (β ρ : ℝ) : ℝ :=
  Real.exp (β * ρ) / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ))

/-- The diagonal entry of the attention matrix. -/
theorem equiWeight_self (i : Idx n) : equiWeight n β ρ i i = equiDiag n β ρ := by
  simp [equiWeight, equiGram, equiDiag]

/-- An off-diagonal entry of the attention matrix. -/
theorem equiWeight_of_ne {i j : Idx n} (h : i ≠ j) :
    equiWeight n β ρ i j = equiOff n β ρ := by
  simp [equiWeight, equiGram, h, equiOff]

/-- **Contracting a row against the Gram matrix.**  Because
`G = (1-ρ) I + ρ J` and every row of the attention matrix sums to `1`,
`(w G)_k = ρ + (1-ρ) w_k`. -/
theorem sum_equiWeight_mul_equiGram (j k : Idx n) :
    ∑ l : Idx n, equiWeight n β ρ j l * equiGram n ρ k l
      = ρ + (1 - ρ) * equiWeight n β ρ j k := by
  have h : ∀ l : Idx n, equiWeight n β ρ j l * equiGram n ρ k l
      = ρ * equiWeight n β ρ j l
        + (if l = k then (1 - ρ) * equiWeight n β ρ j l else 0) := by
    intro l
    by_cases hl : l = k
    · subst hl; simp [equiGram]; ring
    · simp [equiGram, hl, Ne.symm hl]; ring
  rw [Finset.sum_congr rfl fun l _ => h l, Finset.sum_add_distrib, ← Finset.mul_sum,
    sum_equiWeight β ρ j,
    Finset.sum_ite_eq' Finset.univ k (fun l => (1 - ρ) * equiWeight n β ρ j l)]
  simp

/-- **The Gram matrix after one layer.**  `⟨y_i, y_j⟩ = ρ + (1-ρ) ⟨w_i, w_j⟩`:
the double sum collapses to the inner product of the two attention rows. -/
theorem equiOutInner_eq (i j : Idx n) :
    equiOutInner n β ρ i j
      = ρ + (1 - ρ) * ∑ k : Idx n, equiWeight n β ρ i k * equiWeight n β ρ j k := by
  have hrow : ∀ k : Idx n,
      ∑ l : Idx n, equiWeight n β ρ i k * equiWeight n β ρ j l * equiGram n ρ k l
        = ρ * equiWeight n β ρ i k
          + (1 - ρ) * (equiWeight n β ρ i k * equiWeight n β ρ j k) := by
    intro k
    rw [Finset.sum_congr rfl fun l _ => mul_assoc _ _ _, ← Finset.mul_sum,
      sum_equiWeight_mul_equiGram β ρ j k]
    ring
  rw [equiOutInner, Finset.sum_congr rfl fun k _ => hrow k, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, sum_equiWeight β ρ i]
  ring

/-- `a` and `b` are positive, and `a + (n-1) b = 1`. -/
theorem equiDiag_pos (hn : 1 ≤ n) : 0 < equiDiag n β ρ :=
  div_pos (Real.exp_pos _) (equiNorm_pos β ρ hn)

theorem equiOff_pos (hn : 1 ≤ n) : 0 < equiOff n β ρ :=
  div_pos (Real.exp_pos _) (equiNorm_pos β ρ hn)

theorem equiDiag_add_equiOff (hn : 1 ≤ n) :
    equiDiag n β ρ + ((n : ℝ) - 1) * equiOff n β ρ = 1 := by
  have hZ := (equiNorm_pos β ρ hn).ne'
  rw [equiDiag, equiOff]
  field_simp

/-- The squared norm of a row: `a² + (n-1) b²`. -/
theorem sum_equiWeight_sq (i : Idx n) :
    ∑ k : Idx n, equiWeight n β ρ i k * equiWeight n β ρ i k
      = equiDiag n β ρ ^ 2 + ((n : ℝ) - 1) * equiOff n β ρ ^ 2 := by
  have h : ∀ k : Idx n, equiWeight n β ρ i k * equiWeight n β ρ i k
      = equiOff n β ρ ^ 2
        + (if k = i then equiDiag n β ρ ^ 2 - equiOff n β ρ ^ 2 else 0) := by
    intro k
    by_cases hk : k = i
    · subst hk; rw [equiWeight_self]; simp; ring
    · rw [equiWeight_of_ne β ρ (Ne.symm hk)]; simp [hk]; ring
  rw [Finset.sum_congr rfl fun k _ => h k, Finset.sum_add_distrib, Finset.sum_const,
    Finset.sum_ite_eq' Finset.univ i
      (fun _ => equiDiag n β ρ ^ 2 - equiOff n β ρ ^ 2)]
  simp [Finset.card_univ]
  ring

/-- The inner product of two distinct rows: `2ab + (n-2) b²`. -/
theorem sum_equiWeight_mul {i j : Idx n} (hij : i ≠ j) :
    ∑ k : Idx n, equiWeight n β ρ i k * equiWeight n β ρ j k
      = 2 * equiDiag n β ρ * equiOff n β ρ + ((n : ℝ) - 2) * equiOff n β ρ ^ 2 := by
  have h : ∀ k : Idx n, equiWeight n β ρ i k * equiWeight n β ρ j k
      = equiOff n β ρ ^ 2
        + (if k = i then equiDiag n β ρ * equiOff n β ρ - equiOff n β ρ ^ 2 else 0)
        + (if k = j then equiDiag n β ρ * equiOff n β ρ - equiOff n β ρ ^ 2 else 0) := by
    intro k
    by_cases hk : k = i
    · subst hk
      rw [equiWeight_self, equiWeight_of_ne β ρ (Ne.symm hij)]
      simp [hij]
    · by_cases hk' : k = j
      · subst hk'
        rw [equiWeight_of_ne β ρ (Ne.symm hk), equiWeight_self]
        simp [hk]
        ring
      · rw [equiWeight_of_ne β ρ (Ne.symm hk), equiWeight_of_ne β ρ (Ne.symm hk')]
        simp [hk, hk']
        ring
  rw [Finset.sum_congr rfl fun k _ => h k, Finset.sum_add_distrib, Finset.sum_add_distrib,
    Finset.sum_const,
    Finset.sum_ite_eq' Finset.univ i
      (fun _ => equiDiag n β ρ * equiOff n β ρ - equiOff n β ρ ^ 2),
    Finset.sum_ite_eq' Finset.univ j
      (fun _ => equiDiag n β ρ * equiOff n β ρ - equiOff n β ρ ^ 2)]
  simp [Finset.card_univ]
  ring

/-- **The cosine after one layer, in closed form.**  Both output vectors have
the same norm -- the configuration is invariant under the transposition of the
two tokens -- so the square root in `equiOutCos` disappears and what is left is
a rational function of `a` and `b`. -/
theorem equiOutCos_of_ne (hρ₀ : 0 < ρ) (hρ₁ : ρ < 1) {i j : Idx n} (hij : i ≠ j) :
    equiOutCos n β ρ i j
      = (ρ + (1 - ρ) * (2 * equiDiag n β ρ * equiOff n β ρ
            + ((n : ℝ) - 2) * equiOff n β ρ ^ 2))
        / (ρ + (1 - ρ) * (equiDiag n β ρ ^ 2 + ((n : ℝ) - 1) * equiOff n β ρ ^ 2)) := by
  have hn : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast i.pos
  have hS : 0 ≤ equiDiag n β ρ ^ 2 + ((n : ℝ) - 1) * equiOff n β ρ ^ 2 := by
    have : 0 ≤ ((n : ℝ) - 1) * equiOff n β ρ ^ 2 :=
      mul_nonneg (by linarith) (sq_nonneg _)
    nlinarith [sq_nonneg (equiDiag n β ρ)]
  have hD : 0 < ρ + (1 - ρ) * (equiDiag n β ρ ^ 2 + ((n : ℝ) - 1) * equiOff n β ρ ^ 2) := by
    nlinarith
  rw [equiOutCos, equiOutInner_eq β ρ i j, equiOutInner_eq β ρ i i, equiOutInner_eq β ρ j j,
    sum_equiWeight_mul β ρ hij, sum_equiWeight_sq β ρ i, sum_equiWeight_sq β ρ j,
    Real.sqrt_mul_self hD.le]

/-- The hypotheses above are satisfiable: two tokens, `ρ = 1/2`. -/
example : (1 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 ∧ (0 : Idx 2) ≠ 1 :=
  ⟨by norm_num, by norm_num, by norm_num, by decide⟩

end MeanField
end Transformer
