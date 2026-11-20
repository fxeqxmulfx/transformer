module

public import Transformer.ReRoPEBlock.Basic

/-!
# The π head: a head of the main model that sees the order of two keys

Source: the blocks of `Transformer.ReRoPEBlock.Defs` (DiffGPT `5a37c51`). The π head is one head
of the main model with every frequency `π` (LeRoPE learns any positive frequency through its
`α`), `β = 0` (the initial value of the head's coordinate) and the query and the key `q` for
every token, as when `W_Q = W_K` read a direction on which the tokens agree. It scores the
position `t` by `ln(i + 1) κ cos(π t)`, `κ = qᵀq/d` (`headScore_pi`), whatever the tokens. The
block `piHead ψ` returns the head's output at the positions `ψ`: the values `u` of the tokens
weighted by `(i + 1)^{κ cos(π ψ(i − j))}`.

* `attend_sub_attend_swap`: under fixed scores, swapping the values of two keys changes the
  attention by `(e^{s_a} − e^{s_b}) / ∑_j e^{s_j}` times `u_a − u_b`;
* `piHead_sub_permute_swap`: so swapping the first two tokens before the query `i + 1` changes
  `piHead ψ` by that factor at the positions `ψ (i + 1)` and `ψ i`;
* `norm_piHead_sub_of_cos`: if `cos(π ψ(i + 1)) = −cos(π ψ i) = ±1`, the change is at least
  `(1 − N^{−2κ}) / N` times `‖u_a − u_b‖`, `N = i + 2`; `norm_piHead_sub_lePosition`: LeRoPE's
  positions are such.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.RoPE.ReRoPE Transformer.RoPE.RoFormer
open Transformer.Residual

namespace Transformer.ReRoPEBlock

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

/-- **The π head's score**: with every frequency `π`, `β = 0` and the query and the key `q` for
every token, the main model's head scores the position `t` by `ln(i + 1) κ cos(π t)`,
`κ = qᵀq/d`. -/
theorem headScore_pi (d : ℝ) (q : Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) (t : ℝ) :
    headScore (fun _ => Real.pi) 0 d (fun _ => q) (fun _ => q) i x y t =
      Real.log ((i : ℝ) + 1) * (q ⬝ᵥ q / d) * Real.cos (t * Real.pi) := by
  simp only [headScore, phaseScore, zero_mul, zero_div, add_zero, dotProduct_rotary_eq_sum_cos,
    neg_mul, Real.cos_neg, ← Finset.sum_mul]
  have hq : ∑ l, (q (0, l) ^ 2 + q (1, l) ^ 2) = q ⬝ᵥ q := by
    simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← Finset.sum_add_distrib, sq]
  rw [hq]
  ring

/-- Under fixed scores, swapping the values of the keys `a ≠ b` changes the attention by
`(e^{s_a} − e^{s_b}) / ∑_j e^{s_j}` times `u_a − u_b`. -/
theorem attend_sub_attend_swap {ι : Type*} [Fintype ι] [DecidableEq ι] (s : ι → ℝ) (u : ι → E)
    {a b : ι} (hab : a ≠ b) :
    attend s u - attend s (fun j => u (Equiv.swap a b j)) =
      ((Real.exp (s a) - Real.exp (s b)) / ∑ j, Real.exp (s j)) • (u a - u b) := by
  simp only [attend, weightedSum, depthWeight, depthPartition, ← Finset.sum_sub_distrib,
    ← smul_sub]
  rw [Fintype.sum_eq_add a b hab fun j hj => by
    rw [Equiv.swap_apply_of_ne_of_ne hj.1 hj.2, sub_self, smul_zero]]
  rw [Equiv.swap_apply_left, Equiv.swap_apply_right, sub_div, sub_smul, ← neg_sub (u a) (u b),
    smul_neg]
  abel

/-- Two keys with the scores `1` and `0` and the values `1` and `0`. -/
example := attend_sub_attend_swap (E := ℝ) ![1, 0] ![1, 0] (a := (0 : Fin 2)) (b := 1)
  (by decide)

/-- **The π head as a block**: the head of `headScore_pi` at the positions `ψ`, reading the
values `u` of the tokens; the block returns its output. -/
noncomputable def piHead (ψ : ℕ → ℝ) (d : ℝ) (q : Fin 2 × Fin p → ℝ) (u : X → E) (x : ℕ → X)
    (i : ℕ) : E :=
  block ψ (fun _ a => a 0)
    (fun (_ : Fin 1) => headScore (fun _ => Real.pi) 0 d (fun _ => q) (fun _ => q))
    (fun _ => u) x i

/-- **Swapping the first two tokens** before the query `i + 1` changes the π head by
`(e^{s(ψ(i+1))} − e^{s(ψ i)}) / ∑_j e^{s(ψ(i+1−j))}` times `u_a − u_b`, where
`s t = ln(i + 2) κ cos(π t)`. -/
theorem piHead_sub_permute_swap (ψ : ℕ → ℝ) (d : ℝ) (q : Fin 2 × Fin p → ℝ) (u : X → E)
    (x : ℕ → X) (i : ℕ) :
    piHead ψ d q u x (i + 1) -
        piHead ψ d q u (permute (Equiv.swap (0 : Fin (i + 1 + 1)) 1) x) (i + 1) =
      ((Real.exp (Real.log ((i : ℝ) + 2) * (q ⬝ᵥ q / d) * Real.cos (ψ (i + 1) * Real.pi)) -
          Real.exp (Real.log ((i : ℝ) + 2) * (q ⬝ᵥ q / d) * Real.cos (ψ i * Real.pi))) /
        ∑ j : Fin (i + 1 + 1), Real.exp (Real.log ((i : ℝ) + 2) * (q ⬝ᵥ q / d) *
          Real.cos (ψ (i + 1 - j) * Real.pi))) • (u (x 0) - u (x 1)) := by
  have h2 : ((i + 1 : ℕ) : ℝ) + 1 = (i : ℝ) + 2 := by push_cast; ring
  simp only [piHead, block, headScore_pi, h2, permute_apply]
  rw [attend_sub_attend_swap _ (fun j : Fin (i + 1 + 1) => u (x j)) (by simp)]
  simp

/-- **A π head that sees the order of the first two keys**: if the positions of the distances
`i + 1` and `i` score `cos(π ψ(i + 1)) = −cos(π ψ i) = ±1`, then at the query `i + 1`, `0 < d`,
swapping the first two tokens changes the π head by at least `(1 − N^{−2κ}) / N` times
`‖u_a − u_b‖`, `N = i + 2`, `κ = qᵀq/d`: the two keys score `± κ ln N`, and each of the `N`
weights is at most `N^κ`. -/
theorem norm_piHead_sub_of_cos {ψ : ℕ → ℝ} {i : ℕ}
    (hψ : Real.cos (ψ (i + 1) * Real.pi) = -Real.cos (ψ i * Real.pi))
    (hψ1 : |Real.cos (ψ i * Real.pi)| = 1) {d : ℝ} (hd : 0 < d) (q : Fin 2 × Fin p → ℝ)
    (u : X → E) (x : ℕ → X) :
    (1 - ((i : ℝ) + 2) ^ (-2 * (q ⬝ᵥ q / d))) / ((i : ℝ) + 2) * ‖u (x 0) - u (x 1)‖ ≤
      ‖piHead ψ d q u x (i + 1) -
        piHead ψ d q u (permute (Equiv.swap (0 : Fin (i + 1 + 1)) 1) x) (i + 1)‖ := by
  rw [piHead_sub_permute_swap, norm_smul, Real.norm_eq_abs, hψ]
  refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
  set N : ℝ := (i : ℝ) + 2 with hN
  set g : ℝ := Real.log N * (q ⬝ᵥ q / d) with hg
  have hN0 : 0 < N := by positivity
  have hg0 : 0 ≤ g := mul_nonneg (Real.log_nonneg (by linarith [(i.cast_nonneg : (0 : ℝ) ≤ i)]))
    (div_nonneg (Finset.sum_nonneg fun l _ => mul_self_nonneg (q l)) hd.le)
  have hpow : N ^ (-2 * (q ⬝ᵥ q / d)) = Real.exp (-2 * g) := by
    rw [Real.rpow_def_of_pos hN0, hg]
    ring_nf
  set D := ∑ j : Fin (i + 1 + 1), Real.exp (g * Real.cos (ψ (i + 1 - (j : ℕ)) * Real.pi))
  have hD0 : 0 < D := Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  have hD : D ≤ N * Real.exp g := by
    calc D ≤ ∑ _j : Fin (i + 1 + 1), Real.exp g := Finset.sum_le_sum fun j _ =>
          Real.exp_le_exp.2 (mul_le_of_le_one_right hg0 (Real.cos_le_one _))
      _ = N * Real.exp g := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hN]
          push_cast
          ring
  have hnum : Real.exp g - Real.exp (-g) ≤ |Real.exp (g * -Real.cos (ψ i * Real.pi)) -
      Real.exp (g * Real.cos (ψ i * Real.pi))| := by
    rcases (abs_eq zero_le_one).1 hψ1 with h | h <;> rw [h]
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

/-- LeRoPE's positions alternate under the frequency `π`. -/
theorem cos_lePosition_succ (i : ℕ) :
    Real.cos (lePosition (i + 1) * Real.pi) = -Real.cos (lePosition i * Real.pi) ∧
      |Real.cos (lePosition i * Real.pi)| = 1 := by
  refine ⟨?_, ?_⟩ <;> simp only [lePosition, Real.cos_nat_mul_pi]
  · ring
  · rw [abs_pow, abs_neg, abs_one, one_pow]

/-- The π head of width `2` with `q = 1`, `κ = 1`, at LeRoPE's positions `4` and `3`. -/
example (x : ℕ → ℝ) :=
  norm_piHead_sub_of_cos (cos_lePosition_succ 3).1 (cos_lePosition_succ 3).2 (d := 2) two_pos
    (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) x

/-- **LeRoPE's π head sees the order of the first two keys**: at the query `i + 1`, `0 < d`,
swapping them changes it by at least `(1 − N^{−2κ}) / N` times `‖u_a − u_b‖`, `N = i + 2`. -/
theorem norm_piHead_sub_lePosition {d : ℝ} (hd : 0 < d) (q : Fin 2 × Fin p → ℝ) (u : X → E)
    (x : ℕ → X) (i : ℕ) :
    (1 - ((i : ℝ) + 2) ^ (-2 * (q ⬝ᵥ q / d))) / ((i : ℝ) + 2) * ‖u (x 0) - u (x 1)‖ ≤
      ‖piHead lePosition d q u x (i + 1) -
        piHead lePosition d q u (permute (Equiv.swap (0 : Fin (i + 1 + 1)) 1) x) (i + 1)‖ :=
  norm_piHead_sub_of_cos (cos_lePosition_succ i).1 (cos_lePosition_succ i).2 hd q u x

/-- The π head of width `2` with `q = 1`, `κ = 1`, on a sequence of reals read as values. -/
example (x : ℕ → ℝ) :=
  norm_piHead_sub_lePosition (d := 2) two_pos (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t)
    x 3

end Transformer.ReRoPEBlock
