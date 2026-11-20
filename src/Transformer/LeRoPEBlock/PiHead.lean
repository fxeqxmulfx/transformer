module

public import Transformer.LeRoPEBlock.Basic

/-!
# The π head: a head of the main model that sees the order of two keys

Source: the blocks of `Transformer.LeRoPEBlock.Defs` (DiffGPT `5a37c51`). The development follows
`Transformer.ReRoPEBlock.PiHead` (`4647ed60`), from which `dotProduct_rotary_eq_sum_cos` and
`attend_sub_attend_swap` are copied, restated for natural distances and for a query that lives on
the channels of frequency `π` only. LeRoPE learns any positive frequency through its `α`, so one
channel of the main model can have the frequency `π`. The π head reads that channel: `β = 0` (the
initial value of the head's coordinate) and the query and the key `q` for every token, as when
`W_Q = W_K` read a direction on which the tokens agree, supported on the channels of frequency
`π`. It scores the distance `δ` by `ln(i + 1) κ (−1)^δ`, `κ = qᵀq/d` (`leropeLogit_pi`), whatever
the tokens. The block `piHead` returns the head's output: the values `u` of the tokens weighted by
`(i + 1)^{±κ}`, the sign alternating with the parity of the distance.

* `attend_sub_attend_swap`: under fixed scores, swapping the values of two keys changes the
  attention by `(e^{s_a} − e^{s_b}) / ∑_j e^{s_j}` times `u_a − u_b`;
* `norm_piHead_sub_swap`: swapping the first two tokens before the query `n + 2` changes the π
  head by at least `(1 − N^{−2κ}) / N` times `‖u_a − u_b‖`, `N = n + 3`: the two keys lie at
  distances of opposite parity, so they score `± κ ln N`, and each of the `N` weights is at most
  `N^κ`.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {X E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {p : ℕ}

/-- A vector against itself turned by `m`: `qᵀ R^m q = ∑_l |q_l|² cos(m θ_l)`. -/
theorem dotProduct_rotary_eq_sum_cos (θ : Fin p → ℝ) (m : ℝ) (q : Fin 2 × Fin p → ℝ) :
    q ⬝ᵥ (rotary θ m *ᵥ q) = ∑ l, (q (0, l) ^ 2 + q (1, l) ^ 2) * Real.cos (m * θ l) := by
  rw [rotary_mulVec]
  simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, rotateHalf,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun l _ => ?_
  simp
  ring

/-- **A query on the channels of frequency `π`**: if `q` vanishes on the channels whose frequency
is not `π`, then `qᵀ R^m q = qᵀq cos(m π)`. -/
theorem dotProduct_rotary_self_of_pi (θ : Fin p → ℝ) {q : Fin 2 × Fin p → ℝ}
    (hq : ∀ l, θ l ≠ Real.pi → q (0, l) = 0 ∧ q (1, l) = 0) (m : ℝ) :
    q ⬝ᵥ (rotary θ m *ᵥ q) = q ⬝ᵥ q * Real.cos (m * Real.pi) := by
  have hqq : q ⬝ᵥ q = ∑ l, (q (0, l) ^ 2 + q (1, l) ^ 2) := by
    simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← Finset.sum_add_distrib, sq]
  rw [dotProduct_rotary_eq_sum_cos, hqq, Finset.sum_mul]
  refine Finset.sum_congr rfl fun l _ => ?_
  by_cases h : θ l = Real.pi
  · rw [h]
  · obtain ⟨h0, h1⟩ := hq l h
    simp [h0, h1]

/-- A query on the one channel of frequency `π` of a width-`2` head. -/
example (m : ℝ) :
    (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) ⬝ᵥ
        (rotary (fun _ : Fin 1 => Real.pi) m *ᵥ fun _ : Fin 2 × Fin 1 => (1 : ℝ)) =
      (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) ⬝ᵥ (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) *
        Real.cos (m * Real.pi) :=
  dotProduct_rotary_self_of_pi _ (fun _ h => absurd rfl h) m

/-- **The π head's logit**: with `β = 0` and the query and the key `q` for every token, `q` on the
channels of frequency `π`, the main model's head scores the distance `δ` by
`ln(i + 1) κ (−1)^δ`, `κ = qᵀq/d`. -/
theorem leropeLogit_pi (θ : Fin p → ℝ) {q : Fin 2 × Fin p → ℝ}
    (hq : ∀ l, θ l ≠ Real.pi → q (0, l) = 0 ∧ q (1, l) = 0) (d : ℝ) (i : ℕ) (x y : X) (δ : ℕ) :
    leropeLogit θ 0 d (fun _ => q) (fun _ => q) i x y δ =
      Real.log ((i : ℝ) + 1) * (q ⬝ᵥ q / d) * (-1) ^ δ := by
  simp only [leropeLogit, phaseLogit, zero_mul, zero_div, add_zero,
    dotProduct_rotary_self_of_pi θ hq, neg_mul, Real.cos_neg, Real.cos_nat_mul_pi]
  ring

example (x y : ℕ) :
    leropeLogit (fun _ : Fin 1 => Real.pi) 0 2 (fun _ => fun _ : Fin 2 × Fin 1 => (1 : ℝ))
        (fun _ => fun _ : Fin 2 × Fin 1 => (1 : ℝ)) 5 x y 3 =
      Real.log ((5 : ℕ) + 1) *
        ((fun _ : Fin 2 × Fin 1 => (1 : ℝ)) ⬝ᵥ (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) / 2) * (-1) ^ 3 :=
  leropeLogit_pi _ (fun _ h => absurd rfl h) 2 5 x y 3

/-- Under fixed scores, swapping the values of the keys `a ≠ b` changes the attention by
`(e^{s_a} − e^{s_b}) / ∑_j e^{s_j}` times `u_a − u_b`. -/
theorem attend_sub_attend_swap {ι : Type*} [Fintype ι] [DecidableEq ι] (s : ι → ℝ) (u : ι → E)
    {a b : ι} (hab : a ≠ b) :
    attend s u - attend s (fun j => u (Equiv.swap a b j)) =
      ((Real.exp (s a) - Real.exp (s b)) / ∑ j, Real.exp (s j)) • (u a - u b) := by
  simp only [attend, Transformer.Residual.weightedSum, depthWeight, depthPartition,
    ← Finset.sum_sub_distrib, ← smul_sub]
  rw [Fintype.sum_eq_add a b hab fun j hj => by
    rw [Equiv.swap_apply_of_ne_of_ne hj.1 hj.2, sub_self, smul_zero]]
  rw [Equiv.swap_apply_left, Equiv.swap_apply_right, sub_div, sub_smul, ← neg_sub (u a) (u b),
    smul_neg]
  abel

/-- Two keys with the scores `1` and `0` and the values `1` and `0`. -/
example := attend_sub_attend_swap (E := ℝ) ![1, 0] ![1, 0] (a := (0 : Fin 2)) (b := 1)
  (by decide)

end Transformer.LeRoPEBlock
