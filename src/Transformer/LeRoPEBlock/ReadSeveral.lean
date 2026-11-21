module

public import Transformer.LeRoPEBlock.Defs
public import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# Several heads read a key exactly

Source: `review/lerope_block.md`, *Reading a slot* (added), RS4.

One positional head keeps at most a share below one on a key (RS1, RS3), so its average mixes the
key's value with the others. Several heads can cancel the mixture. Let the head `h` put the weight
`A_hj` on the key `j`. If coefficients `b` combine the weights into the indicator of the key `t`,
`∑_h b_h A_hj = [j = t]`, the same combination of the heads' averages is the value at `t`, whatever
the values (`sum_smul_attend_eq`). An error `ε_h` in the averages reaches the read value as
`∑_h b_h ε_h`, and `b` cannot be short when the heads put little weight on `t`:
`‖b‖ ‖A e_t‖ ≥ 1` (`one_le_sum_sq_mul_sum_sq`).

Heads that each put `p` on a key of their own and `q` on every other key read every key. The head
of `s` averages to `o_s = (p − q) v_s + q ∑_j v_j` (`weightedSum_ite_eq`), so
`v_s = (o_s − q ∑_j v_j) / (p − q)` (`decode_weightedSum_ite`), where `∑_j v_j` is the sum of all
the averages when each head's weights sum to one (`sum_weightedSum_ite`), or `N` times the average
of a uniform head. The reading is stable: `∑_s ‖o_s‖² ≥ (p − q)² ∑_j ‖v_j‖²` for `0 ≤ q ≤ p`
(`sum_norm_sq_weightedSum_ite_ge`), applied to the difference of two sets of values. So `p − q` is
the smallest singular value of the weight matrix `(p − q) I + q J`.
-/

@[expose] public section

open Finset
open scoped RealInnerProductSpace
open Transformer.AttentionResiduals Transformer.Residual

namespace Transformer.LeRoPEBlock

variable {κ ι E : Type*} [Fintype κ] [Fintype ι] [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **A combination of heads reads the key `t`** when it combines their weights into the indicator
of `t`: then `∑_h b_h o_h = v_t` for the heads' averages `o_h`, whatever the values `v`. -/
theorem sum_smul_attend_eq [DecidableEq ι] (s : κ → ι → ℝ) (b : κ → ℝ) (t : ι)
    (hb : ∀ j, ∑ h, b h * depthWeight (s h) j = if j = t then 1 else 0) (v : ι → E) :
    ∑ h, b h • attend (s h) v = v t := by
  simp only [attend, weightedSum, smul_sum, smul_smul]
  rw [sum_comm]
  simp only [← sum_smul, hb, ite_smul, one_smul, zero_smul, sum_ite_eq', mem_univ, ite_true]

/-- One key: the one head keeps all of it, and its average is the value. -/
example (v : Fin 1 → ℝ) : ∑ _h : Fin 1, (1 : ℝ) • attend (fun _ : Fin 1 => (0 : ℝ)) v = v 0 :=
  sum_smul_attend_eq (fun _ _ => 0) (fun _ => 1) 0
    (fun j => by simp [depthWeight_zero, Fin.fin_one_eq_zero j]) v

/-- **`‖b‖ ‖A e_t‖ ≥ 1`**: the coefficients `b` that read a key cannot be short if the heads put
the weights `a_h` on it, by Cauchy–Schwarz. -/
theorem one_le_sum_sq_mul_sum_sq (a b : κ → ℝ) (hb : ∑ h, b h * a h = 1) :
    1 ≤ (∑ h, b h ^ 2) * ∑ h, a h ^ 2 := by
  have h := sum_mul_sq_le_sq_mul_sq univ b a
  rwa [hb, one_pow] at h

/-- A head with the weight `1/2` on the key needs the coefficient `2`. -/
example : 1 ≤ (∑ _h : Fin 1, (2 : ℝ) ^ 2) * ∑ _h : Fin 1, (1 / 2 : ℝ) ^ 2 :=
  one_le_sum_sq_mul_sum_sq (fun _ => 1 / 2) (fun _ => 2) (by norm_num)

/-- **A head that puts `p` on the key `s` and `q` on every other key** averages to
`(p − q) v_s + q ∑_j v_j`. -/
theorem weightedSum_ite_eq [DecidableEq ι] (s : ι) (p q : ℝ) (v : ι → E) :
    weightedSum (fun j => if j = s then p else q) v = (p - q) • v s + q • ∑ j, v j := by
  have h (j : ι) : (if j = s then p else q) • v j =
      (if j = s then (p - q) • v j else 0) + q • v j := by
    split_ifs <;> simp [sub_smul]
  simp only [weightedSum, h, sum_add_distrib, sum_ite_eq', mem_univ, ite_true, smul_sum]

/-- **The value back from the average**: for `p ≠ q` the value at `s` is
`(o_s − q ∑_j v_j) / (p − q)`. -/
theorem decode_weightedSum_ite [DecidableEq ι] {p q : ℝ} (hpq : p ≠ q) (v : ι → E) (s : ι) :
    (p - q)⁻¹ • (weightedSum (fun j => if j = s then p else q) v - q • ∑ j, v j) = v s := by
  rw [weightedSum_ite_eq, add_sub_cancel_right, smul_smul, inv_mul_cancel₀ (sub_ne_zero.2 hpq),
    one_smul]

/-- Three keys, `p = 1/2` and `q = 1/4`. -/
example (v : Fin 3 → ℝ) :
    ((1 / 2 : ℝ) - 1 / 4)⁻¹ • (weightedSum (fun j => if j = 0 then (1 / 2 : ℝ) else 1 / 4) v -
      (1 / 4 : ℝ) • ∑ j, v j) = v 0 :=
  decode_weightedSum_ite (by norm_num) v 0

/-- **The averages of all the heads sum to the values' sum** when each head's weights sum to one,
`p + (N − 1) q = 1`. -/
theorem sum_weightedSum_ite [DecidableEq ι] {p q : ℝ}
    (h1 : p + ((Fintype.card ι : ℝ) - 1) * q = 1) (v : ι → E) :
    ∑ s, weightedSum (fun j => if j = s then p else q) v = ∑ j, v j := by
  simp only [weightedSum_ite_eq, sum_add_distrib, ← smul_sum, sum_const, card_univ,
    ← Nat.cast_smul_eq_nsmul ℝ, smul_smul, ← add_smul]
  rw [show p - q + q * (Fintype.card ι : ℝ) = 1 by linarith, one_smul]

/-- Three keys, `p = 1/2`, `q = 1/4`. -/
example (v : Fin 3 → ℝ) :
    ∑ s, weightedSum (fun j => if j = s then (1 / 2 : ℝ) else 1 / 4) v = ∑ j, v j :=
  sum_weightedSum_ite (by norm_num) v

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- **The reading is stable**: for `0 ≤ q ≤ p` the averages of the heads of all keys, each `p` on
its key and `q` elsewhere, have `∑_s ‖o_s‖² ≥ (p − q)² ∑_j ‖v_j‖²`. On the difference of two sets
of values: the averages move by at least `p − q` times as much as the values. -/
theorem sum_norm_sq_weightedSum_ite_ge [DecidableEq ι] {p q : ℝ} (hq : 0 ≤ q) (hqp : q ≤ p)
    (v : ι → F) :
    (p - q) ^ 2 * ∑ j, ‖v j‖ ^ 2 ≤ ∑ s, ‖weightedSum (fun j => if j = s then p else q) v‖ ^ 2 := by
  set S := ∑ j, v j
  have h (s : ι) : ‖weightedSum (fun j => if j = s then p else q) v‖ ^ 2 =
      (p - q) ^ 2 * ‖v s‖ ^ 2 + 2 * ((p - q) * q) * ⟪v s, S⟫ + q ^ 2 * ‖S‖ ^ 2 := by
    rw [weightedSum_ite_eq, norm_add_sq_real, norm_smul, norm_smul, real_inner_smul_left,
      real_inner_smul_right, mul_pow, mul_pow, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, sq_abs]
    ring
  have hS : ∑ s, ⟪v s, S⟫ = ‖S‖ ^ 2 := by rw [← sum_inner, real_inner_self_eq_norm_sq]
  simp only [h, sum_add_distrib, ← mul_sum, hS, sum_const, card_univ, nsmul_eq_mul]
  have hpq : 0 ≤ (p - q) * q := mul_nonneg (sub_nonneg.2 hqp) hq
  have h0 : 0 ≤ 2 * ((p - q) * q) * ‖S‖ ^ 2 + (Fintype.card ι : ℝ) * (q ^ 2 * ‖S‖ ^ 2) := by
    positivity
  linarith

/-- Two keys, `p = 3/4`, `q = 1/4`: the averages move by at least half as much as the values. -/
example (v : Fin 2 → ℝ) :
    ((3 / 4 : ℝ) - 1 / 4) ^ 2 * ∑ j, ‖v j‖ ^ 2 ≤
      ∑ s, ‖weightedSum (fun j => if j = s then (3 / 4 : ℝ) else 1 / 4) v‖ ^ 2 :=
  sum_norm_sq_weightedSum_ite_ge (by norm_num) (by norm_num) v

end Transformer.LeRoPEBlock
