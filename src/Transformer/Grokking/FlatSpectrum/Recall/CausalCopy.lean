module

public import Transformer.Grokking.FlatSpectrum.Recall.InductionPhases

/-!
# The causal previous-token head: a value token copies as if its prefix were the context

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

`Recall.Induction` reads the context cyclically. Here it is read causally: `b` tokens without a
key (a beginning-of-sequence token when `b = 1`), then `k₀ v₀ k₁ v₁ …`, so the key token of `c`
is at `b + 2c` and its value token at `b + 2c + 1`. The token at `p` reads the `p + 1` positions
up to it at the scale `s log (p + 1)`, with the raw score `1` at `p − 1` and `0` elsewhere, and
copies the key channel (`causalCopy`). In closed form (`causalCopy_eq`) its copy of the key `c` is

* `0` before the key token of `c`, which it does not see (`causalCopy_of_lt`);
* `1 / (K^s + K − 1)` at the key token itself, over its prefix of `K = b + 2c + 1` positions
  (`causalCopy_key`), and `1 / ((p + 1)^s + p)` after the value token (`causalCopy_of_gt`);
* `L^s / (L^s + L − 1)` at the value token, over its prefix of `L = b + 2c + 2` positions
  (`causalCopy_value`).

The last is the copy `copyWeight s L` of the cyclic head over `L` tokens (`prevTokenHead_value`,
`causalCopy_value_eq_prevTokenHead`): the copy of a key is set by its position, not by the length
of the context. Over the keys it tends to `0`, `1/2` or `1` as `s` is below, at or above `1`
(`tendsto_causalCopy_value_of_lt`, `tendsto_causalCopy_value_one`,
`tendsto_causalCopy_value_of_gt`). Above `s = 1` the late keys copy almost perfectly, the first
only at `(b + 2)^s / ((b + 2)^s + b + 1)`; at `s = 1` the late keys keep half, below it nothing.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **The causal previous-token head**, layer 1 (`review/flat_spectrum.md`, SA5): over the
context of `b` tokens without a key and then `k₀ v₀ k₁ v₁ …`, the token at the position `p`
reads the `p + 1` positions up to it at the scale `s log (p + 1)`, with the positional raw score
`1` at the position `p − 1` and `0` elsewhere, and copies the coordinate `c` of the key channel:
`1` at the key token of `c`, the position `b + 2c`, and `0` elsewhere. -/
noncomputable def causalCopy (s : ℝ) (b p c : ℕ) : ℝ :=
  logScaleAttention s (fun j : Fin (p + 1) => if (j : ℕ) + 1 = p then 1 else 0)
    fun j => if (j : ℕ) = b + 2 * c then 1 else 0

/-- **The causal copy in closed form** (`review/flat_spectrum.md`, SA5): the token at the
position `p` copies the key `c` with the weight
`(((p + 1)^s − 1) [p = b + 2c + 1] + [b + 2c ≤ p]) / ((p + 1)^s + p)`. -/
theorem causalCopy_eq (s : ℝ) (b p c : ℕ) :
    causalCopy s b p c = ((((p : ℝ) + 1) ^ s - 1) * (if p = b + 2 * c + 1 then 1 else 0) +
      if b + 2 * c ≤ p then 1 else 0) / (((p : ℝ) + 1) ^ s + p) := by
  have hS : ∑ j : Fin (p + 1), (if (j : ℕ) = b + 2 * c then (1 : ℝ) else 0) =
      if b + 2 * c ≤ p then 1 else 0 := by
    rw [Fin.sum_univ_eq_sum_range (fun j => if j = b + 2 * c then (1 : ℝ) else 0),
      sum_ite_eq']
    simp only [mem_range, Nat.lt_succ_iff]
  rcases Nat.eq_zero_or_pos p with rfl | hp
  · rw [causalCopy, logScaleAttention_eq_rpow, Fintype.card_fin]
    simp only [Nat.cast_zero, zero_add, Nat.cast_one, Real.one_rpow, one_mul, hS, sub_self,
      zero_mul, add_zero, sum_const, card_univ, Fintype.card_fin, one_smul, div_one]
  · have hraw : (fun j : Fin (p + 1) => if (j : ℕ) + 1 = p then (1 : ℝ) else 0) =
        fun j => if j = ⟨p - 1, by omega⟩ then 1 else 0 := by
      funext j
      have h : (j : ℕ) + 1 = p ↔ j = ⟨p - 1, by omega⟩ := by simp only [Fin.ext_iff]; omega
      simp only [h]
    have hv : (p - 1 = b + 2 * c) ↔ p = b + 2 * c + 1 := by omega
    rw [causalCopy, hraw, logScaleAttention_ite, hS, Fintype.card_fin]
    simp only [hv, sub_zero, mul_one]
    push_cast
    ring_nf

/-- **The copy weight** of a previous-token head over `x` tokens: `x^s / (x^s + x − 1)`, the
weight at which it copies the key before a token to that token (`prevTokenHead_eq`). -/
noncomputable def copyWeight (s x : ℝ) : ℝ := x ^ s / (x ^ s + x - 1)

/-- The cyclic head over `n = 2m + 2` tokens copies the key `c` to its value token with the
weight `copyWeight s n`. -/
lemma prevTokenHead_value (s : ℝ) {m : ℕ} (c : Fin (m + 1)) :
    prevTokenHead s (c, true) c = copyWeight s (2 * m + 2) := by
  rw [prevTokenHead_eq, ite_eq_left rfl, copyWeight]
  ring_nf

/-- **The value token copies its key as over its prefix** (`review/flat_spectrum.md`, SA5): with
the weight `copyWeight s L` for the `L = b + 2c + 2` positions up to it. -/
theorem causalCopy_value (s : ℝ) (b c : ℕ) :
    causalCopy s b (b + 2 * c + 1) c = copyWeight s (b + 2 * c + 2) := by
  rw [causalCopy_eq, ite_eq_left rfl, ite_eq_left (by omega), copyWeight]
  push_cast
  ring_nf

/-- **The causal copy is the cyclic copy of the prefix** (`review/flat_spectrum.md`, SA5):
without tokens before the first key, the value token of `c` copies its key as the cyclic head
over the `c + 1` keys up to it. -/
theorem causalCopy_value_eq_prevTokenHead (s : ℝ) (c : ℕ) :
    causalCopy s 0 (2 * c + 1) c = prevTokenHead s (Fin.last c, true) (Fin.last c) := by
  rw [show 2 * c + 1 = 0 + 2 * c + 1 by ring, causalCopy_value, prevTokenHead_value]
  push_cast
  ring_nf

/-- **The key token copies itself** (`review/flat_spectrum.md`, SA5) with the weight
`1 / (K^s + K − 1)` for the `K = b + 2c + 1` positions up to it. -/
theorem causalCopy_key (s : ℝ) (b c : ℕ) :
    causalCopy s b (b + 2 * c) c = 1 / (((b : ℝ) + 2 * c + 1) ^ s + b + 2 * c) := by
  rw [causalCopy_eq, ite_eq_right (by omega), ite_eq_left le_rfl]
  push_cast
  ring_nf

/-- A token before the key token of `c` does not see it. -/
theorem causalCopy_of_lt (s : ℝ) {b p c : ℕ} (h : p < b + 2 * c) : causalCopy s b p c = 0 := by
  rw [causalCopy_eq, ite_eq_right (by omega), ite_eq_right (by omega)]
  simp

/-- The token at the position `0` precedes the key token of `1`. -/
example := causalCopy_of_lt 2 (b := 0) (p := 0) (c := 1) (by norm_num)

/-- A token after the value token of `c` copies it with the weight `1 / ((p + 1)^s + p)`. -/
theorem causalCopy_of_gt (s : ℝ) {b p c : ℕ} (h : b + 2 * c + 1 < p) :
    causalCopy s b p c = 1 / (((p : ℝ) + 1) ^ s + p) := by
  rw [causalCopy_eq, ite_eq_right (by omega), ite_eq_left (by omega)]
  ring

/-- The token at the position `2` follows the value token of `0`. -/
example := causalCopy_of_gt 2 (b := 0) (p := 2) (c := 0) (by norm_num)

/-- The causal copy is a weight. -/
lemma causalCopy_nonneg (s : ℝ) (b p c : ℕ) : 0 ≤ causalCopy s b p c := by
  have hx : (0 : ℝ) < ((p : ℝ) + 1) ^ s := Real.rpow_pos_of_pos (by positivity) _
  rw [causalCopy_eq]
  refine div_nonneg ?_ (by positivity)
  split_ifs <;> first | omega | linarith

/-- The copy weight is `1 − (1 − 1/x) / (x^{s − 1} + 1 − 1/x)` above `x = 1`. -/
lemma copyWeight_eq {s x : ℝ} (hx : 1 < x) :
    copyWeight s x = 1 - (1 - x⁻¹) / (x ^ (s - 1) + 1 - x⁻¹) := by
  have hx0 : 0 < x := by linarith
  have hZ : 0 < x ^ s + x - 1 := by linarith [Real.rpow_pos_of_pos hx0 s]
  rw [copyWeight, Real.rpow_sub_one hx0.ne', show x ^ s / x + 1 - x⁻¹ = (x ^ s + x - 1) / x by
      field_simp, show 1 - x⁻¹ = (x - 1) / x by field_simp, div_div_div_cancel_right₀
      hx0.ne', one_sub_div hZ.ne']
  ring_nf

/-- If `x^{s − 1}` tends to `p`, the copy weight tends to `1 − 1 / (p + 1)`. -/
lemma tendsto_copyWeight_of_tendsto {s p : ℝ}
    (hp : Tendsto (fun x : ℝ => x ^ (s - 1)) atTop (𝓝 p)) (hp1 : p + 1 ≠ 0) :
    Tendsto (copyWeight s) atTop (𝓝 (1 - 1 / (p + 1))) := by
  have hi : Tendsto (fun x : ℝ => x⁻¹) atTop (𝓝 0) := tendsto_inv_atTop_zero
  have h : Tendsto (fun x : ℝ => 1 - (1 - x⁻¹) / (x ^ (s - 1) + 1 - x⁻¹)) atTop
      (𝓝 (1 - (1 - 0) / (p + 1 - 0))) :=
    tendsto_const_nhds.sub ((tendsto_const_nhds.sub hi).div ((hp.add_const 1).sub hi)
      (by rwa [sub_zero]))
  simp only [sub_zero] at h
  exact h.congr' ((eventually_gt_atTop 1).mono fun x hx => (copyWeight_eq hx).symm)

/-- At `s = 1`, `x^{s − 1} = 1`. -/
example := tendsto_copyWeight_of_tendsto (s := 1) (p := 1)
  (by simp only [sub_self, Real.rpow_zero]; exact tendsto_const_nhds) (by norm_num)

/-- The value token of the key `c` has `b + 2c + 2 → ∞` positions up to it. -/
lemma tendsto_valuePrefix (b : ℕ) : Tendsto (fun c : ℕ => (b + 2 * c + 2 : ℝ)) atTop atTop :=
  tendsto_atTop_add_const_right _ 2 (tendsto_atTop_add_const_left _ _
    (tendsto_natCast_atTop_atTop.const_mul_atTop two_pos))

/-- **Below `s = 1` the late keys lose their copy** (`review/flat_spectrum.md`, SA5). -/
theorem tendsto_causalCopy_value_of_lt {s : ℝ} (hs : s < 1) (b : ℕ) :
    Tendsto (fun c : ℕ => causalCopy s b (b + 2 * c + 1) c) atTop (𝓝 0) := by
  have hp := tendsto_rpow_neg_atTop (by linarith : 0 < 1 - s)
  rw [neg_sub] at hp
  have h := tendsto_copyWeight_of_tendsto hp (by norm_num)
  rw [zero_add, div_one, sub_self] at h
  exact (h.comp (tendsto_valuePrefix b)).congr fun c => (causalCopy_value s b c).symm

/-- At `s = 1/2` after a beginning-of-sequence token. -/
example := tendsto_causalCopy_value_of_lt (s := 1 / 2) (by norm_num) 1

/-- **At `s = 1` the late keys keep half their copy** (`review/flat_spectrum.md`, SA5). -/
theorem tendsto_causalCopy_value_one (b : ℕ) :
    Tendsto (fun c : ℕ => causalCopy 1 b (b + 2 * c + 1) c) atTop (𝓝 (1 / 2)) := by
  have h := tendsto_copyWeight_of_tendsto (s := 1) (p := 1)
    (by simp only [sub_self, Real.rpow_zero]; exact tendsto_const_nhds) (by norm_num)
  rw [show (1 : ℝ) - 1 / (1 + 1) = 1 / 2 by norm_num] at h
  exact (h.comp (tendsto_valuePrefix b)).congr fun c => (causalCopy_value 1 b c).symm

/-- **Above `s = 1` the late keys keep their whole copy** (`review/flat_spectrum.md`, SA5). -/
theorem tendsto_causalCopy_value_of_gt {s : ℝ} (hs : 1 < s) (b : ℕ) :
    Tendsto (fun c : ℕ => causalCopy s b (b + 2 * c + 1) c) atTop (𝓝 1) := by
  have hp := tendsto_rpow_atTop (by linarith : 0 < s - 1)
  have hi := (tendsto_const_nhds (x := (1 : ℝ))).sub (tendsto_inv_atTop_zero (𝕜 := ℝ))
  have h := (tendsto_const_nhds (x := (1 : ℝ))).sub (hi.mul (hp.atTop_add hi).inv_tendsto_atTop)
  simp only [sub_zero, mul_zero] at h
  have hw : Tendsto (copyWeight s) atTop (𝓝 1) := by
    refine h.congr' ((eventually_gt_atTop 1).mono fun x hx => ?_)
    simp only [Pi.inv_apply]
    rw [copyWeight_eq hx, div_eq_mul_inv, add_sub_assoc]
  exact (hw.comp (tendsto_valuePrefix b)).congr fun c => (causalCopy_value s b c).symm

/-- At `s = 2` after a beginning-of-sequence token. -/
example := tendsto_causalCopy_value_of_gt (s := 2) (by norm_num) 1

end Transformer.Grokking.FlatSpectrum.Recall
