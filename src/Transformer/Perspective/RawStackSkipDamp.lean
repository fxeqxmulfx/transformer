/-
# What the gauge does to a skip, and what it does not

**Not a statement of any paper.**  `Perspective.RawStackSkip` writes the skip
of modded-nanogpt (`x = x + skip_gate_out * cache[3]`, `train_gpt.py`,
`GPT.forward`) in the gauge: what reaches the directions is not the gate
`s_{k,i}` but `s_{k,i} Λ_{m,i} / Λ_{k+1,i}`, the gate divided by the gain the
stream accumulates between the snapshot at depth `m` and the skip at depth `k`.
Here that ratio is read.

Above, at gains of at least `1 + c`, the gain accumulated over `q - p`
sublayers is at least `(1+c)^{q-p}` (`pow_mul_gainProd_le`), so the gate is
divided by that much: `|s| (1+c)^{-(k+1-m)}` (`gainProd_div_le`,
`abs_skipWeight_le`).  A skip over a deep enough gap is damped away.

Below, at gains of at most `1 + C`, the division is by no more than
`(1+C)^{q-p}` (`gainProd_le_pow_mul`, `le_gainProd_div`,
`le_abs_skipWeight`).  At a fixed gap `D` that is `|s| (1+C)^{-D}` at every
depth, the same number at every layer: the weights of skips at a fixed distance
do not shrink with depth, and `rawStack_frozen` — which sums the weights the
directions see — has no route to them.  The block outputs `g_{k,i}` are damped
by `Λ_{k+1,i}⁻¹` and a skip is not, which is the sense in which a skip is not a
bounded drive.

And at the record's own gains neither bound bites.  The skip reads `cache[3]`,
the state at depth `8`, and takes the step at depth `12` at gain `1`, so four
sublayers at `1.1^{1/2}` lie between them and `100/121` of its gate survives
(`gainProd_div_record`).
-/

import Transformer.Perspective.RawStackSkip

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {n : ℕ}

/-- **Gains above `1 + c` accumulate at least geometrically between two
depths:** `(1+c)^{q-p} Λ_{p,i} ≤ Λ_{q,i}` for `p ≤ q`.

Source: none — posed here; `pow_le_gainProd` of `Perspective.RawStackFrozen`
starts from `p = 0`, and a skip starts wherever the snapshot was taken. -/
theorem pow_mul_gainProd_le {lam : ℕ → Idx n → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hlam : ∀ j i, 1 + c ≤ lam j i) {p q : ℕ} (hpq : p ≤ q) (i : Idx n) :
    (1 + c) ^ (q - p) * gainProd lam p i ≤ gainProd lam q i := by
  have hpos : ∀ (j : ℕ) (i : Idx n), 0 < lam j i := fun j i =>
    lt_of_lt_of_le (by linarith) (hlam j i)
  induction q, hpq using Nat.le_induction with
  | base => simp
  | succ q hq ih =>
      have hstep : gainProd lam q i * lam q i = gainProd lam (q + 1) i :=
        (Finset.prod_range_succ (fun j => lam j i) q).symm
      have hsub : q + 1 - p = (q - p) + 1 := Nat.succ_sub hq
      rw [hsub, pow_succ, ← hstep]
      calc (1 + c) ^ (q - p) * (1 + c) * gainProd lam p i
          = ((1 + c) ^ (q - p) * gainProd lam p i) * (1 + c) := by ring
        _ ≤ gainProd lam q i * lam q i :=
            mul_le_mul ih (hlam q i) (by linarith) (gainProd_pos hpos q i).le

/-- **The gauge damps a skip exponentially in the depth it jumps:**
`Λ_{p,i} / Λ_{q,i} ≤ (1+c)^{-(q-p)}`.

Source: none — posed here; the weight `s_{k,i} Λ_{m,i} / Λ_{k+1,i}` of
`ungauged_rec_skip`, at gains above `1 + c`. -/
theorem gainProd_div_le {lam : ℕ → Idx n → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hlam : ∀ j i, 1 + c ≤ lam j i) {p q : ℕ} (hpq : p ≤ q) (i : Idx n) :
    gainProd lam p i / gainProd lam q i ≤ ((1 + c)⁻¹) ^ (q - p) := by
  have hpos : ∀ (j : ℕ) (i : Idx n), 0 < lam j i := fun j i =>
    lt_of_lt_of_le (by linarith) (hlam j i)
  have hq : 0 < gainProd lam q i := gainProd_pos hpos q i
  have hpow : (0 : ℝ) < (1 + c) ^ (q - p) := by positivity
  rw [div_le_iff₀ hq, inv_pow, inv_mul_eq_div, le_div_iff₀ hpow]
  calc gainProd lam p i * (1 + c) ^ (q - p)
      = (1 + c) ^ (q - p) * gainProd lam p i := by ring
    _ ≤ gainProd lam q i := pow_mul_gainProd_le hc hlam hpq i

/-- **The rescaled gate of a skip:** `|s| (1+c)^{-(k+1-m)}`, the gate divided by
the gain accumulated over the sublayers it jumps.

Source: none — posed here; `ungauged_rec_skip` and `gainProd_div_le`. -/
theorem abs_skipWeight_le {lam s : ℕ → Idx n → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hlam : ∀ j i, 1 + c ≤ lam j i) {m k : ℕ} (hm : m ≤ k + 1) (i : Idx n) :
    |s k i * gainProd lam m i / gainProd lam (k + 1) i|
      ≤ |s k i| * ((1 + c)⁻¹) ^ (k + 1 - m) := by
  have hpos : ∀ (j : ℕ) (i : Idx n), 0 < lam j i := fun j i =>
    lt_of_lt_of_le (by linarith) (hlam j i)
  have hratio : 0 < gainProd lam m i / gainProd lam (k + 1) i :=
    div_pos (gainProd_pos hpos m i) (gainProd_pos hpos (k + 1) i)
  rw [mul_div_assoc, abs_mul, abs_of_pos hratio]
  exact mul_le_mul_of_nonneg_left (gainProd_div_le hc hlam hm i) (abs_nonneg _)

/-- The hypotheses of `pow_mul_gainProd_le`, `gainProd_div_le` and
`abs_skipWeight_le` are satisfiable: `c = 0`, gains `1`, and the depths `8` and
`12` of the record's own skip, `cache[3]` and the step that adds it. -/
example : (0 : ℝ) ≤ 0 ∧ (∀ (_ : ℕ) (_ : Idx 1), (1 : ℝ) + 0 ≤ 1) ∧ 8 ≤ 12 + 1 :=
  ⟨le_rfl, fun _ _ => by norm_num, by norm_num⟩

/-- **Gains below `1 + C` accumulate at most geometrically:**
`Λ_{q,i} ≤ (1+C)^{q-p} Λ_{p,i}` for `p ≤ q`.

Source: none — posed here; the other side of `pow_mul_gainProd_le`. -/
theorem gainProd_le_pow_mul {lam : ℕ → Idx n → ℝ} {C : ℝ} (hC : 0 ≤ C)
    (hpos : ∀ j i, 0 < lam j i) (hlam : ∀ j i, lam j i ≤ 1 + C) {p q : ℕ} (hpq : p ≤ q)
    (i : Idx n) : gainProd lam q i ≤ (1 + C) ^ (q - p) * gainProd lam p i := by
  induction q, hpq using Nat.le_induction with
  | base => simp
  | succ q hq ih =>
      have hstep : gainProd lam q i * lam q i = gainProd lam (q + 1) i :=
        (Finset.prod_range_succ (fun j => lam j i) q).symm
      have hsub : q + 1 - p = (q - p) + 1 := Nat.succ_sub hq
      rw [hsub, pow_succ, ← hstep]
      calc gainProd lam q i * lam q i
          ≤ ((1 + C) ^ (q - p) * gainProd lam p i) * (1 + C) :=
            mul_le_mul ih (hlam q i) (hpos q i).le
              (mul_nonneg (by positivity) (gainProd_pos hpos p i).le)
        _ = (1 + C) ^ (q - p) * (1 + C) * gainProd lam p i := by ring

/-- **And no more than geometrically:** `(1+C)^{-(q-p)} ≤ Λ_{p,i} / Λ_{q,i}`.

Source: none — posed here; the other side of `gainProd_div_le`. -/
theorem le_gainProd_div {lam : ℕ → Idx n → ℝ} {C : ℝ} (hC : 0 ≤ C)
    (hpos : ∀ j i, 0 < lam j i) (hlam : ∀ j i, lam j i ≤ 1 + C) {p q : ℕ} (hpq : p ≤ q)
    (i : Idx n) : ((1 + C)⁻¹) ^ (q - p) ≤ gainProd lam p i / gainProd lam q i := by
  have hq : 0 < gainProd lam q i := gainProd_pos hpos q i
  have hpow : (0 : ℝ) < (1 + C) ^ (q - p) := by positivity
  rw [le_div_iff₀ hq, inv_pow, inv_mul_eq_div, div_le_iff₀ hpow]
  calc gainProd lam q i ≤ (1 + C) ^ (q - p) * gainProd lam p i :=
        gainProd_le_pow_mul hC hpos hlam hpq i
    _ = gainProd lam p i * (1 + C) ^ (q - p) := by ring

/-- **A skip is not damped below `|s| (1+C)^{-(k+1-m)}`.**  At a fixed gap
`k + 1 - m = D` this is the same number at every depth, so the weights of skips
at a fixed distance do not shrink as the stack deepens — unlike the block
outputs, which the gauge divides by `Λ_{k+1,i}`.  Summing them, the route
`rawStack_frozen` takes, is therefore not available for such skips.

Source: none — posed here; `ungauged_rec_skip` and `le_gainProd_div`. -/
theorem le_abs_skipWeight {lam s : ℕ → Idx n → ℝ} {C : ℝ} (hC : 0 ≤ C)
    (hpos : ∀ j i, 0 < lam j i) (hlam : ∀ j i, lam j i ≤ 1 + C) {m k : ℕ} (hm : m ≤ k + 1)
    (i : Idx n) : |s k i| * ((1 + C)⁻¹) ^ (k + 1 - m)
      ≤ |s k i * gainProd lam m i / gainProd lam (k + 1) i| := by
  have hratio : 0 < gainProd lam m i / gainProd lam (k + 1) i :=
    div_pos (gainProd_pos hpos m i) (gainProd_pos hpos (k + 1) i)
  rw [mul_div_assoc, abs_mul, abs_of_pos hratio]
  exact mul_le_mul_of_nonneg_left (le_gainProd_div hC hpos hlam hm i) (abs_nonneg _)

/-- The hypotheses of `gainProd_le_pow_mul`, `le_gainProd_div` and
`le_abs_skipWeight` are satisfiable: `C = 0`, gains `1`, and the depths `8` and
`12` of the record's skip. -/
example : (0 : ℝ) ≤ 0 ∧ (∀ (_ : ℕ) (_ : Idx 1), (0 : ℝ) < 1) ∧
    (∀ (_ : ℕ) (_ : Idx 1), (1 : ℝ) ≤ 1 + 0) ∧ 8 ≤ 12 + 1 :=
  ⟨le_rfl, fun _ _ => one_pos, fun _ _ => by norm_num, by norm_num⟩

/-- **At the record's own gains the damping is weak.**  The skip of
modded-nanogpt reads `cache[3]`, written at the end of loop iteration `3`, after
its MLP: the state at depth `8`.  It takes the place of the attention sublayer
of layer `6`, the step at depth `12`, and carries no `resid_lambdas`: that step
has gain `1`.  So its gate is multiplied by
`Λ_{8,i} / Λ_{13,i} = (λ_{8,i} λ_{9,i} λ_{10,i} λ_{11,i})⁻¹`, at the initial
gains `1.1^{1/2}` that is `(11/10)^{-2} = 100/121`, and more than four fifths of
the gate survives: the exponent of `gainProd_div_le` is not yet doing anything.

This corrects `lt_inv_pow_record`, which took the snapshot for the state at
depth `6` and all six steps from there for gains `1.1^{1/2}`, and concluded
`3/4 < (11/10)^{-3}`.

Source: `x = x + skip_gate_out * cache[3]` at `i == 6`, `cache_layers = [3, 7]`
and `torch.full((num_layers, 2), 1.1**0.5)` (modded-nanogpt, `train_gpt.py`,
`GPT.forward`, `GPT.init_attn`, `GPT.init_misc`), read against `gainProd_div_le`. -/
theorem gainProd_div_record {lam : ℕ → Idx n → ℝ} {c : ℝ} (hsq : (1 + c) ^ 2 = 11 / 10)
    (hpos : ∀ j i, 0 < lam j i) (hlam : ∀ j i, 8 ≤ j → j < 12 → lam j i = 1 + c)
    (h12 : ∀ i, lam 12 i = 1) (i : Idx n) :
    gainProd lam 8 i / gainProd lam 13 i = 100 / 121 := by
  have h4 : (1 + c) ^ 4 = 121 / 100 := by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, hsq]
    norm_num
  have hsplit : gainProd lam 13 i = gainProd lam 8 i * (1 + c) ^ 4 := by
    rw [show (13 : ℕ) = 8 + 5 from rfl, gainProd, gainProd, Finset.prod_range_add]
    congr 1
    simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul, Nat.reduceAdd]
    rw [hlam 8 i (by norm_num) (by norm_num), hlam 9 i (by norm_num) (by norm_num),
      hlam 10 i (by norm_num) (by norm_num), hlam 11 i (by norm_num) (by norm_num), h12 i]
    ring
  rw [hsplit, div_mul_eq_div_div, div_self (gainProd_pos hpos 8 i).ne', h4]
  norm_num

/-- The hypotheses of `gainProd_div_record` are satisfiable, at the record's own
gains: `c = (11/10)^{1/2} - 1`, gain `1` at depth `12` and `1.1^{1/2}` elsewhere. -/
example : ∃ (c : ℝ) (lam : ℕ → Idx 1 → ℝ), (1 + c) ^ 2 = 11 / 10 ∧ (∀ j i, 0 < lam j i) ∧
    (∀ j i, 8 ≤ j → j < 12 → lam j i = 1 + c) ∧ ∀ i, lam 12 i = 1 := by
  refine ⟨Real.sqrt (11 / 10) - 1, fun j _ => if j = 12 then 1 else Real.sqrt (11 / 10),
    ?_, fun j _ => ?_, fun j _ _ hj => by simp [Nat.ne_of_lt hj], fun _ => by simp⟩
  · rw [add_sub_cancel, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 11 / 10)]
  · dsimp only
    split_ifs <;> positivity

end Perspective
end Transformer
