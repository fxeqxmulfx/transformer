module

public import Transformer.LeRoPEBlock.PiHead

/-!
# The π head sees the order of the first two keys

Source: the blocks of `Transformer.LeRoPEBlock.Defs` (DiffGPT `5a37c51`); the bound and its proof
follow `Transformer.ReRoPEBlock.norm_piHead_sub_of_cos` (`4647ed60`), with the parity of the
natural distance, `(−1)^δ`, in place of its `cos(π ψ)`. The block `piHead` returns the output of
the π head of `Transformer.LeRoPEBlock.PiHead` at the query `i`: the values `u` of the tokens,
weighted by the softmax of the scores `ln(i + 1) κ (−1)^{i − j}`.

* `piHead_sub_permute_swap`: swapping the first two tokens before the query `n + 2` changes it by
  `(e^{s_0} − e^{s_1}) / ∑_j e^{s_j}` times `u_a − u_b`, with `s_j = g (−1)^{n + 2 − j}`,
  `g = ln(n + 3) κ`;
* `norm_piHead_sub_swap`: the first two keys lie at distances `n + 2` and `n + 1`, of opposite
  parity, so they score `± g`: the change is at least `(1 − N^{−2κ}) / N` times `‖u_a − u_b‖`,
  `N = n + 3`, `κ = qᵀq/d`.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {X E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {p : ℕ}

/-- **The π head as a block**: the head of `leropeLogit_pi` with the frequencies `θ`, reading the
values `u` of the tokens; the block returns its output. -/
noncomputable def piHead (θ : Fin p → ℝ) (d : ℝ) (q : Fin 2 × Fin p → ℝ) (u : X → E)
    (x : ℕ → X) (i : ℕ) : E :=
  block (fun _ a => a 0) (fun (_ : Fin 1) => leropeLogit θ 0 d (fun _ => q) (fun _ => q))
    (fun _ => u) x i

/-- The π head is the attention with the scores `ln(i + 1) κ (−1)^{i − j}`. -/
theorem piHead_eq (θ : Fin p → ℝ) {q : Fin 2 × Fin p → ℝ}
    (hq : ∀ l, θ l ≠ Real.pi → q (0, l) = 0 ∧ q (1, l) = 0) (d : ℝ) (u : X → E) (x : ℕ → X)
    (i : ℕ) :
    piHead θ d q u x i =
      attend (fun j : Fin (i + 1) =>
        Real.log ((i : ℝ) + 1) * (q ⬝ᵥ q / d) * (-1) ^ (i - (j : ℕ))) fun j => u (x j) := by
  simp only [piHead, block, leropeLogit_pi θ hq]

/-- The π head of width `2` with `q = 1`, `κ = 1`, at the query `3`, on a sequence of reals read
as values. -/
example (x : ℕ → ℝ) :=
  piHead_eq (fun _ : Fin 1 => Real.pi) (q := fun _ : Fin 2 × Fin 1 => (1 : ℝ))
    (fun _ h => absurd rfl h) 2 (fun t : ℝ => t) x 3

/-- **Swapping the first two tokens** before the query `n + 2` changes the π head by
`(e^{s_0} − e^{s_1}) / ∑_j e^{s_j}` times `u_a − u_b`, where `s_j = g (−1)^{n + 2 − j}` and
`g = ln(n + 3) κ`. -/
theorem piHead_sub_permute_swap (θ : Fin p → ℝ) {q : Fin 2 × Fin p → ℝ}
    (hq : ∀ l, θ l ≠ Real.pi → q (0, l) = 0 ∧ q (1, l) = 0) (d : ℝ) (u : X → E) (x : ℕ → X)
    (n : ℕ) :
    piHead θ d q u x (n + 2) -
        piHead θ d q u (permute (Equiv.swap (0 : Fin (n + 2 + 1)) 1) x) (n + 2) =
      ((Real.exp (Real.log ((n : ℝ) + 3) * (q ⬝ᵥ q / d) * (-1) ^ (n + 2)) -
          Real.exp (Real.log ((n : ℝ) + 3) * (q ⬝ᵥ q / d) * (-1) ^ (n + 1))) /
        ∑ j : Fin (n + 2 + 1),
          Real.exp (Real.log ((n : ℝ) + 3) * (q ⬝ᵥ q / d) * (-1) ^ (n + 2 - (j : ℕ)))) •
        (u (x 0) - u (x 1)) := by
  have h3 : ((n + 2 : ℕ) : ℝ) + 1 = (n : ℝ) + 3 := by push_cast; ring
  simp only [piHead_eq θ hq, h3, permute_apply]
  rw [attend_sub_attend_swap _ (fun j : Fin (n + 2 + 1) => u (x j)) (a := 0) (b := 1)
    (by simp)]
  simp

/-- The same π head, at the query `3` (`n = 1`). -/
example (x : ℕ → ℝ) :=
  piHead_sub_permute_swap (fun _ : Fin 1 => Real.pi) (q := fun _ : Fin 2 × Fin 1 => (1 : ℝ))
    (fun _ h => absurd rfl h) 2 (fun t : ℝ => t) x 1

/-- **The π head sees the order of the first two keys**: at the query `n + 2`, `0 < d`, swapping
the first two tokens changes it by at least `(1 − N^{−2κ}) / N` times `‖u_a − u_b‖`, `N = n + 3`,
`κ = qᵀq/d`: the two keys score `± κ ln N`, and each of the `N` weights is at most `N^κ`. -/
theorem norm_piHead_sub_swap (θ : Fin p → ℝ) {q : Fin 2 × Fin p → ℝ}
    (hq : ∀ l, θ l ≠ Real.pi → q (0, l) = 0 ∧ q (1, l) = 0) {d : ℝ} (hd : 0 < d) (u : X → E)
    (x : ℕ → X) (n : ℕ) :
    (1 - ((n : ℝ) + 3) ^ (-2 * (q ⬝ᵥ q / d))) / ((n : ℝ) + 3) * ‖u (x 0) - u (x 1)‖ ≤
      ‖piHead θ d q u x (n + 2) -
        piHead θ d q u (permute (Equiv.swap (0 : Fin (n + 2 + 1)) 1) x) (n + 2)‖ := by
  rw [piHead_sub_permute_swap θ hq, norm_smul, Real.norm_eq_abs]
  refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
  set N : ℝ := (n : ℝ) + 3 with hN
  set g : ℝ := Real.log N * (q ⬝ᵥ q / d) with hg
  have hN0 : 0 < N := by positivity
  have hg0 : 0 ≤ g := mul_nonneg (Real.log_nonneg (by linarith [(n.cast_nonneg : (0 : ℝ) ≤ n)]))
    (div_nonneg (Finset.sum_nonneg fun l _ => mul_self_nonneg (q l)) hd.le)
  have hpow : N ^ (-2 * (q ⬝ᵥ q / d)) = Real.exp (-2 * g) := by
    rw [Real.rpow_def_of_pos hN0, hg]
    ring_nf
  have hs : ∀ k : ℕ, Real.exp (g * (-1) ^ k) ≤ Real.exp g := fun k =>
    Real.exp_le_exp.2 (by
      rcases neg_one_pow_eq_or ℝ k with h | h <;> rw [h] <;> linarith)
  set D := ∑ j : Fin (n + 2 + 1), Real.exp (g * (-1) ^ (n + 2 - (j : ℕ)))
  have hD0 : 0 < D := Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  have hD : D ≤ N * Real.exp g := by
    calc D ≤ ∑ _j : Fin (n + 2 + 1), Real.exp g := Finset.sum_le_sum fun j _ => hs _
      _ = N * Real.exp g := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hN]
          push_cast
          ring
  have hnum : Real.exp g - Real.exp (-g) ≤
      |Real.exp (g * (-1) ^ (n + 2)) - Real.exp (g * (-1) ^ (n + 1))| := by
    have ha : (-1 : ℝ) ^ (n + 2) = -(-1) ^ (n + 1) := by rw [pow_succ]; ring
    rw [ha]
    rcases neg_one_pow_eq_or ℝ (n + 1) with h | h <;> rw [h]
    · rw [mul_neg_one, mul_one, abs_sub_comm]
      exact le_abs_self _
    · rw [neg_neg, mul_one, mul_neg_one]
      exact le_abs_self _
  have h2 : (1 - Real.exp (-2 * g)) * Real.exp g = Real.exp g - Real.exp (-g) := by
    rw [sub_mul, one_mul, ← Real.exp_add, show -2 * g + g = -g by ring]
  have h1 : 0 ≤ 1 - Real.exp (-2 * g) := sub_nonneg.2 (Real.exp_le_one_iff.2 (by linarith))
  rw [abs_div, abs_of_pos hD0, hpow]
  calc (1 - Real.exp (-2 * g)) / N
      = (1 - Real.exp (-2 * g)) * Real.exp g / (N * Real.exp g) :=
        (mul_div_mul_right _ _ (Real.exp_ne_zero g)).symm
    _ ≤ (Real.exp g - Real.exp (-g)) / D := by
        rw [h2]
        exact div_le_div₀ (h2 ▸ mul_nonneg h1 (Real.exp_pos g).le) le_rfl hD0 hD
    _ ≤ _ := div_le_div_of_nonneg_right hnum hD0.le

/-- The π head of width `2` with `q = 1`, `κ = 1`, on a sequence of reals read as values, at the
query `3`. -/
example (x : ℕ → ℝ) :=
  norm_piHead_sub_swap (fun _ : Fin 1 => Real.pi) (q := fun _ : Fin 2 × Fin 1 => (1 : ℝ))
    (fun _ h => absurd rfl h) (d := 2) two_pos (fun t : ℝ => t) x 1

end Transformer.LeRoPEBlock
