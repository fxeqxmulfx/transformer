/-
# Growth freezes the directions of a stack

**Not a statement of any paper.**  `Perspective.RawStack` writes a stack of
different blocks as its gauge `Λ_{k,i} = ∏_{j<k} λ_{j,i}` times the same stack
with the gains divided out, and the directions are those of the second
(`normalize_rawStack`).  Here the gains are read: at `λ ≥ 1 + c`, `c > 0`, the
gauge grows at least as `(1 + c)^k` (`pow_le_gainProd`), so the steps
`g_{k,i} / Λ_{k+1,i}` the directions do see are summable whatever the blocks
compute, and the whole stack moves each direction by at most
`2 M / (c ‖x_{0,i}‖)` (`rawStack_frozen`).

That is `rawStream_frozen` of `Perspective.RawGrowth` without its continuous
time, and without what continuous time forced: the gain is now per block and
per token instead of one constant `c`, the blocks are all different — eleven of
them, as in the model — and `Σ_{j≥1} (1+c)^{-j} = 1/c` replaces Gronwall.  With
`resid_lambdas` of modded-nanogpt at `1.1^{1/2}` this is `c = 1.1^{1/2} - 1`,
and a stack of any depth moves a direction by at most `2 M / (c ‖x_{0,i}‖)`.

Then the constants are read back against the record, and they are not in this
regime: the bound exceeds `40` once the blocks may output as much as the stream
they start from (`lt_rawStack_frozen_bound`), while the directions live on a
sphere of diameter `2`, and the gauge the whole stack accumulates at
initialisation is a factor under `3` (`gainProd_lt_three`).  What the theorem
rules out is a stack that starts long and is asked to turn; what the record has
is the exact factorisation of `Perspective.RawStack`, which needs no constants
at all.
-/

import Transformer.Perspective.RawStack
import Mathlib.Algebra.Order.Field.GeomSum

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Gains above `1 + c` accumulate at least geometrically:**
`(1 + c)^k ≤ Λ_{k,i}`.

Source: none — posed here; the hypothesis of `rawStack_frozen`. -/
theorem pow_le_gainProd {lam : ℕ → Idx n → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hlam : ∀ j i, 1 + c ≤ lam j i) (k : ℕ) (i : Idx n) : (1 + c) ^ k ≤ gainProd lam k i := by
  induction k with
  | zero => simp [gainProd]
  | succ k ih =>
      have h0 : (0 : ℝ) ≤ (1 + c) ^ k := by positivity
      rw [gainProd, Finset.prod_range_succ, ← gainProd, pow_succ]
      exact mul_le_mul ih (hlam k i) (by linarith) (h0.trans ih)

/-- The hypotheses of `pow_le_gainProd` are satisfiable: `c = 0` and gains `1`. -/
example : (0 : ℝ) ≤ 0 ∧ ∀ (_ : ℕ) (_ : Idx 1), (1 : ℝ) + 0 ≤ 1 :=
  ⟨le_rfl, fun _ _ => by norm_num⟩

/-- **A stack of growing blocks freezes its own directions.**  Let every gain
be at least `1 + c` with `c > 0` and every block output be at most `M`:
`x_{k+1,i} = λ_{k,i} x_{k,i} + g_{k,i}`, `λ_{k,i} ≥ 1 + c`, `‖g_{k,i}‖ ≤ M`.
Then at every depth `k` the direction of token `i` is within
`2 M / (c ‖x_{0,i}‖)` of the direction it started with — a bound on the whole
stack, uniform in its depth and in what its blocks compute.

This is `rawStream_frozen` in discrete time, and it drops what that one had to
assume: the gain is now per block and per token instead of one constant `c`,
the blocks are all different, and `Σ_{j≥1} (1+c)^{-j} = 1/c` replaces Gronwall.

Source: none — posed here; a stack of pre-norm blocks with `resid_lambdas > 1`
(modded-nanogpt, `train_gpt.py`, `GPT.forward`). -/
theorem rawStack_frozen {lam : ℕ → Idx n → ℝ} {x g : ℕ → Idx n → EucSpace d} {c M : ℝ}
    (hc : 0 < c) (hM : 0 ≤ M) (hlam : ∀ j i, 1 + c ≤ lam j i)
    (hx : ∀ k i, x (k + 1) i = lam k i • x k i + g k i) (hg : ∀ j i, ‖g j i‖ ≤ M)
    (h0 : ∀ i, x 0 i ≠ 0) (k : ℕ) (i : Idx n) :
    ‖‖x k i‖⁻¹ • x k i - ‖x 0 i‖⁻¹ • x 0 i‖ ≤ 2 * M / (c * ‖x 0 i‖) := by
  have hpos : ∀ (j : ℕ) (i : Idx n), 0 < lam j i := fun j i =>
    lt_of_lt_of_le (by linarith) (hlam j i)
  have hx0 : 0 < ‖x 0 i‖ := norm_pos_iff.2 (h0 i)
  have hc1 : (0 : ℝ) < 1 + c := by linarith
  have hr0 : (0 : ℝ) ≤ (1 + c)⁻¹ := by positivity
  have hr1 : (1 + c)⁻¹ < 1 := inv_lt_one_of_one_lt₀ (by linarith)
  have hgeom : ∑ j ∈ Finset.range k, ((1 + c)⁻¹) ^ (j + 1) ≤ 1 / c := by
    have hshift : ∑ j ∈ Finset.range k, ((1 + c)⁻¹) ^ (j + 1)
        = (1 + c)⁻¹ * ∑ j ∈ Finset.range k, ((1 + c)⁻¹) ^ j := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    have hbase : ∑ j ∈ Finset.range k, ((1 + c)⁻¹) ^ j ≤ 1 / (1 - (1 + c)⁻¹) := by
      rw [Finset.range_eq_Ico]
      simpa using geom_sum_Ico_le_of_lt_one hr0 hr1 (m := 0) (n := k)
    have hmono : (1 + c)⁻¹ * ∑ j ∈ Finset.range k, ((1 + c)⁻¹) ^ j
        ≤ (1 + c)⁻¹ * (1 / (1 - (1 + c)⁻¹)) := by gcongr
    rw [hshift]
    refine hmono.trans_eq ?_
    field_simp
    rw [show (1 : ℝ) + c - 1 = c by ring, div_self hc.ne']
  have hsum : ‖gaugeStack lam (x 0) g k i - x 0 i‖ ≤ M / c := by
    have hgauge : gaugeStack lam (x 0) g k i - x 0 i
        = ∑ j ∈ Finset.range k, (gainProd lam (j + 1) i)⁻¹ • g j i := by
      rw [gaugeStack, add_sub_cancel_left]
    have hterm : ∀ j ∈ Finset.range k, ‖(gainProd lam (j + 1) i)⁻¹ • g j i‖
        ≤ M * ((1 + c)⁻¹) ^ (j + 1) := fun j _ => by
      have hpow : (0 : ℝ) < (1 + c) ^ (j + 1) := by positivity
      have hle : (1 + c) ^ (j + 1) ≤ gainProd lam (j + 1) i :=
        pow_le_gainProd hc.le hlam (j + 1) i
      have hinv : (gainProd lam (j + 1) i)⁻¹ ≤ ((1 + c)⁻¹) ^ (j + 1) := by
        rw [inv_pow]
        exact inv_anti₀ hpow hle
      rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.2 (gainProd_pos hpos (j + 1) i).le),
        mul_comm]
      exact mul_le_mul (hg j i) hinv (inv_nonneg.2 (gainProd_pos hpos (j + 1) i).le) hM
    rw [hgauge]
    calc ‖∑ j ∈ Finset.range k, (gainProd lam (j + 1) i)⁻¹ • g j i‖
        ≤ ∑ j ∈ Finset.range k, ‖(gainProd lam (j + 1) i)⁻¹ • g j i‖ := norm_sum_le _ _
      _ ≤ ∑ j ∈ Finset.range k, M * ((1 + c)⁻¹) ^ (j + 1) := Finset.sum_le_sum hterm
      _ = M * ∑ j ∈ Finset.range k, ((1 + c)⁻¹) ^ (j + 1) := by rw [Finset.mul_sum]
      _ ≤ M * (1 / c) := mul_le_mul_of_nonneg_left hgeom hM
      _ = M / c := by ring
  rw [normalize_rawStack hpos hx k i, norm_sub_rev]
  calc ‖‖x 0 i‖⁻¹ • x 0 i - ‖gaugeStack lam (x 0) g k i‖⁻¹ • gaugeStack lam (x 0) g k i‖
      ≤ 2 * ‖x 0 i - gaugeStack lam (x 0) g k i‖ / ‖x 0 i‖ := norm_normalize_sub_le_div (h0 i)
    _ ≤ 2 * (M / c) / ‖x 0 i‖ := by
        gcongr
        rw [norm_sub_rev]
        exact hsum
    _ = 2 * M / (c * ‖x 0 i‖) := by field_simp

/-- The hypotheses of `rawStack_frozen` are satisfiable: gains `2` (so `c = 1`),
no block output (`M = 0`) and the stack `x_k = 2^k e₀`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (∀ (_ : ℕ) (_ : Idx 1), (1 : ℝ) + 1 ≤ 2) ∧
    (∀ (k : ℕ) (i : Idx 1),
      (fun k (_ : Idx 1) => (2 : ℝ) ^ k • (basePoint 0 : EucSpace 1)) (k + 1) i
        = (2 : ℝ) • (fun k (_ : Idx 1) => (2 : ℝ) ^ k • (basePoint 0 : EucSpace 1)) k i + 0) ∧
    (∀ (_ : ℕ) (_ : Idx 1), ‖(0 : EucSpace 1)‖ ≤ 0) ∧
    ∀ _ : Idx 1, (2 : ℝ) ^ 0 • (basePoint 0 : EucSpace 1) ≠ 0 :=
  ⟨one_pos, le_rfl, fun _ _ => by norm_num,
    fun k _ => by simp [smul_smul, pow_succ, mul_comm],
    fun _ _ => by simp, fun _ => by simp [basePoint]⟩

/-- **At the record's own gains the bound of `rawStack_frozen` is empty.**  A
gain of `1.1^{1/2}` per sublayer is `1 + c` with `(1 + c)^2 = 11/10`, so
`c < 1/20`, and the bound `2 M / (c ‖x_{0,i}‖)` then exceeds `40` as soon as the
blocks may output as much as the stream they start from (`‖x_{0,i}‖ ≤ M`, which
is the model's own scale: a pre-norm block reads `norm(x)` and adds something of
that size).  The directions live on a sphere of diameter `2`, so a bound above
`40` rules nothing out.

Source: `nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))` (modded-nanogpt,
`train_gpt.py`, `GPT.__init__`), read against `rawStack_frozen`. -/
theorem lt_rawStack_frozen_bound {c M r : ℝ} (hc : 0 < c) (hsq : (1 + c) ^ 2 = 11 / 10)
    (hr : 0 < r) (hM : r ≤ M) : 40 < 2 * M / (c * r) := by
  have hclt : c < 1 / 20 := by nlinarith [sq_nonneg c]
  have hcr : 0 < c * r := mul_pos hc hr
  have h1 : (40 : ℝ) < 2 * r / (c * r) := by
    rw [lt_div_iff₀ hcr]
    nlinarith [mul_pos hr (sub_pos.mpr hclt)]
  exact h1.trans_le (by gcongr)

/-- **And the gauge the record accumulates is a factor under `3`.**  Eleven
layers of two sublayers each, at `1.1^{1/2}` per sublayer, is
`Λ_{22} = (11/10)^{11} < 3`.  So the regime of `rawStack_frozen` — a gauge large
enough that the steps `g_{k,i} / Λ_{k+1,i}` the directions see are spent — is
not the one the record is in at initialisation.  `resid_lambdas` is trained, and
where training takes it this says nothing.

Source: `num_layers=11` and
`nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))` (modded-nanogpt,
`train_gpt.py`), read against `pow_le_gainProd`. -/
theorem gainProd_lt_three {c : ℝ} (hsq : (1 + c) ^ 2 = 11 / 10) (i : Idx n) :
    gainProd (fun _ _ => 1 + c) 22 i < 3 := by
  have h22 : (1 + c) ^ 22 = (11 / 10 : ℝ) ^ 11 := by
    rw [← hsq, ← pow_mul]
  rw [gainProd, Finset.prod_const, Finset.card_range, h22]
  norm_num

/-- The hypotheses of `lt_rawStack_frozen_bound` and `gainProd_lt_three` are
satisfiable, at the record's own gain `c = (11/10)^{1/2} - 1` and a stream and
blocks of the same size. -/
example : ∃ c : ℝ, 0 < c ∧ (1 + c) ^ 2 = 11 / 10 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 := by
  refine ⟨Real.sqrt (11 / 10) - 1, ?_, ?_, one_pos, le_rfl⟩
  · have h : (1 : ℝ) < Real.sqrt (11 / 10) := (Real.lt_sqrt zero_le_one).mpr (by norm_num)
    linarith
  · have h : (1 : ℝ) + (Real.sqrt (11 / 10) - 1) = Real.sqrt (11 / 10) := by ring
    rw [h, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 11 / 10)]


end Perspective
end Transformer
