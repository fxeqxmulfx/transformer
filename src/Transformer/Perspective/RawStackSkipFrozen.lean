/-
# Freezing survives one skip

**Not a statement of any paper.**  `rawStack_frozen` of
`Perspective.RawStackFrozen` freezes the directions of a stack whose gains
exceed `1 + c` and whose blocks output at most `M`, and it needs every update to
be `x_{k+1,i} = λ_{k,i} x_{k,i} + g_{k,i}`.  Modded-nanogpt has one update that
is not of that form: at layer `6` it adds the snapshot of layer `3` instead of a
block output (`x = x + skip_gate_out * cache[3]`, `train_gpt.py`,
`GPT.forward`).  That is one skip in twenty-two sublayers, and the question this
module answers is what it costs.

It costs one term.  A stack with a single skip — `hx` at every depth but `K`,
`hxK` at `K`, reading depth `m ≤ K` with a gate `s_i` per token — is, in the
gauge, the skip-free gauge stack plus one fixed vector
`(s_i Λ_{m,i} / Λ_{K+1,i}) G_{m,i}` from depth `K + 1` onwards
(`ungauged_eq_gaugeStack_of_le`, `ungauged_eq_gaugeStack_add_skip`).  So the
displacement `M / c` of `norm_gaugeStack_sub_le` gains one summand, damped by
the depth the skip jumps, and every direction of the stack stays within

  `2 (M/c + |s_i| (1+c)^{-(K+1-m)} (‖x_{0,i}‖ + M/c)) / ‖x_{0,i}‖`

of the direction it started with (`rawStack_frozen_skip`), at every depth and
whatever the blocks compute.

That is the whole architecture of the record, its one skip included, and not a
model of it with the skip dropped.  What it does not cover is a skip at every
layer: `le_abs_skipWeight` of `Perspective.RawStackSkipDamp` shows their
rescaled gates do not shrink with depth, so the sum this proof takes over one
skip does not converge over infinitely many.
-/

import Transformer.Perspective.RawStackFrozen
import Transformer.Perspective.RawStackSkipDamp

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Before the skip, the stack is its gauge stack.**  Up to and including the
depth `K` at which the one skip happens, every update is
`x_{k+1,i} = λ_{k,i} x_{k,i} + g_{k,i}`, so the ungauged stream is exactly the
gauge stack of `Perspective.RawStack`.

Source: none — posed here; `rawStack_eq_gauge` restricted to the depths below a
skip. -/
theorem ungauged_eq_gaugeStack_of_le {lam : ℕ → Idx n → ℝ} {x g : ℕ → Idx n → EucSpace d}
    {K : ℕ} (hlam : ∀ j i, 0 < lam j i)
    (hx : ∀ k i, k ≠ K → x (k + 1) i = lam k i • x k i + g k i)
    {k : ℕ} (hk : k ≤ K) (i : Idx n) :
    ungauged lam x k i = gaugeStack lam (x 0) g k i := by
  induction k with
  | zero => simp [ungauged, gaugeStack, gainProd]
  | succ k ih =>
      have hkK : k < K := lt_of_lt_of_le (Nat.lt_succ_self k) hk
      rw [ungauged_step hlam (hx k i hkK.ne), ih hkK.le, gaugeStack_succ]

/-- **After the skip, the stack is its gauge stack plus one fixed vector.**  A
single skip at depth `K`, reading the stream at depth `m ≤ K` with a gate `s_i`
per token, adds to every later depth the same vector
`(s_i Λ_{m,i} / Λ_{K+1,i}) G_{m,i}`, where `G` is the gauge stack: the skip is
not a new dynamics, it is one displacement of the ungauged stream.

Source: none — posed here; `x = x + skip_gate_out * cache[3]` at layer `6` of
modded-nanogpt (`train_gpt.py`, `GPT.forward`), with `ungauged_rec_skip` of
`Perspective.RawStackSkip`. -/
theorem ungauged_eq_gaugeStack_add_skip {lam : ℕ → Idx n → ℝ} {s : Idx n → ℝ}
    {x g : ℕ → Idx n → EucSpace d} {K m : ℕ} (hlam : ∀ j i, 0 < lam j i) (hmK : m ≤ K)
    (hx : ∀ k i, k ≠ K → x (k + 1) i = lam k i • x k i + g k i)
    (hxK : ∀ i, x (K + 1) i = lam K i • x K i + s i • x m i + g K i)
    {k : ℕ} (hk : K + 1 ≤ k) (i : Idx n) :
    ungauged lam x k i = gaugeStack lam (x 0) g k i
      + (s i * gainProd lam m i / gainProd lam (K + 1) i) • gaugeStack lam (x 0) g m i := by
  induction k, hk using Nat.le_induction with
  | base =>
      have hxm : x m i = gainProd lam m i • gaugeStack lam (x 0) g m i := by
        rw [← ungauged_eq_gaugeStack_of_le hlam hx hmK i, ungauged,
          smul_inv_smul₀ (gainProd_pos hlam m i).ne']
      have hcoef : (gainProd lam (K + 1) i)⁻¹ * s i * gainProd lam m i
          = s i * gainProd lam m i / gainProd lam (K + 1) i := by ring
      rw [ungauged_step hlam (by rw [hxK i, add_assoc]),
        ungauged_eq_gaugeStack_of_le hlam hx le_rfl i, smul_add, ← add_assoc, gaugeStack_succ,
        hxm, smul_smul, smul_smul, hcoef]
      abel
  | succ k hk ih =>
      have hkK : k ≠ K := by omega
      rw [ungauged_step hlam (hx k i hkK), ih, gaugeStack_succ]
      abel

/-- **A stack of growing blocks with one skip still freezes its own
directions.**  Every gain at least `1 + c`, every block output at most `M`, and
one skip at depth `K` reading depth `m ≤ K` with a gate `s_i` per token: then at
every depth the direction of token `i` is within

  `2 (M/c + |s_i| (1+c)^{-(K+1-m)} (‖x_{0,i}‖ + M/c)) / ‖x_{0,i}‖`

of the direction it started with.  The skip costs exactly one term, damped by
the gain accumulated over the depth it jumps (`abs_skipWeight_le`), and at
`s = 0` the bound is `rawStack_frozen`'s.

Source: none — posed here; the residual stack of modded-nanogpt with its layer-6
skip (`train_gpt.py`, `GPT.forward`). -/
theorem rawStack_frozen_skip {lam : ℕ → Idx n → ℝ} {s : Idx n → ℝ}
    {x g : ℕ → Idx n → EucSpace d} {c M : ℝ} {K m : ℕ} (hc : 0 < c) (hM : 0 ≤ M)
    (hlam : ∀ j i, 1 + c ≤ lam j i) (hmK : m ≤ K)
    (hx : ∀ k i, k ≠ K → x (k + 1) i = lam k i • x k i + g k i)
    (hxK : ∀ i, x (K + 1) i = lam K i • x K i + s i • x m i + g K i)
    (hg : ∀ j i, ‖g j i‖ ≤ M) (h0 : ∀ i, x 0 i ≠ 0) (k : ℕ) (i : Idx n) :
    ‖‖x k i‖⁻¹ • x k i - ‖x 0 i‖⁻¹ • x 0 i‖
      ≤ 2 * (M / c + |s i| * ((1 + c)⁻¹) ^ (K + 1 - m) * (‖x 0 i‖ + M / c)) / ‖x 0 i‖ := by
  have hpos : ∀ (j : ℕ) (i : Idx n), 0 < lam j i := fun j i =>
    lt_of_lt_of_le (by linarith) (hlam j i)
  have hx0 : 0 < ‖x 0 i‖ := norm_pos_iff.2 (h0 i)
  have hgm : ‖gaugeStack lam (x 0) g m i‖ ≤ ‖x 0 i‖ + M / c := by
    have h := norm_gaugeStack_sub_le (x₀ := x 0) hc hM hlam hg m i
    have h' := norm_sub_norm_le (gaugeStack lam (x 0) g m i) (x 0 i)
    linarith
  have hw : |s i * gainProd lam m i / gainProd lam (K + 1) i|
      ≤ |s i| * ((1 + c)⁻¹) ^ (K + 1 - m) := by
    have hmK1 : m ≤ K + 1 := le_trans hmK (Nat.le_succ K)
    exact abs_skipWeight_le (s := fun _ j => s j) hc.le hlam hmK1 i
  have hkey : ‖ungauged lam x k i - x 0 i‖
      ≤ M / c + |s i| * ((1 + c)⁻¹) ^ (K + 1 - m) * (‖x 0 i‖ + M / c) := by
    rcases le_or_gt k K with hk | hk
    · rw [ungauged_eq_gaugeStack_of_le hpos hx hk i]
      have hextra : 0 ≤ |s i| * ((1 + c)⁻¹) ^ (K + 1 - m) * (‖x 0 i‖ + M / c) := by
        have : 0 ≤ ‖x 0 i‖ + M / c := by positivity
        positivity
      linarith [norm_gaugeStack_sub_le (x₀ := x 0) hc hM hlam hg k i]
    · rw [ungauged_eq_gaugeStack_add_skip hpos hmK hx hxK hk i]
      calc ‖gaugeStack lam (x 0) g k i
              + (s i * gainProd lam m i / gainProd lam (K + 1) i)
                • gaugeStack lam (x 0) g m i - x 0 i‖
          = ‖(gaugeStack lam (x 0) g k i - x 0 i)
              + (s i * gainProd lam m i / gainProd lam (K + 1) i)
                • gaugeStack lam (x 0) g m i‖ := by rw [show ∀ a b e : EucSpace d,
                  a + b - e = a - e + b from fun a b e => by abel]
        _ ≤ ‖gaugeStack lam (x 0) g k i - x 0 i‖
              + ‖(s i * gainProd lam m i / gainProd lam (K + 1) i)
                • gaugeStack lam (x 0) g m i‖ := norm_add_le _ _
        _ ≤ M / c + |s i| * ((1 + c)⁻¹) ^ (K + 1 - m) * (‖x 0 i‖ + M / c) := by
            refine add_le_add (norm_gaugeStack_sub_le (x₀ := x 0) hc hM hlam hg k i) ?_
            rw [norm_smul, Real.norm_eq_abs]
            exact mul_le_mul hw hgm (norm_nonneg _) (by positivity)
  rw [← normalize_ungauged hpos x k i, norm_sub_rev]
  calc ‖‖x 0 i‖⁻¹ • x 0 i - ‖ungauged lam x k i‖⁻¹ • ungauged lam x k i‖
      ≤ 2 * ‖x 0 i - ungauged lam x k i‖ / ‖x 0 i‖ := norm_normalize_sub_le_div (h0 i)
    _ ≤ 2 * (M / c + |s i| * ((1 + c)⁻¹) ^ (K + 1 - m) * (‖x 0 i‖ + M / c)) / ‖x 0 i‖ := by
        gcongr
        rw [norm_sub_rev]
        exact hkey

/-- The hypotheses of `ungauged_eq_gaugeStack_of_le`,
`ungauged_eq_gaugeStack_add_skip` and `rawStack_frozen_skip` are satisfiable,
and with a skip that does something: gains `2` (so `c = 1`), no block output
(`M = 0`), a skip of gate `1` at depth `K = 0` reading the start `m = 0`, and
the stack it forces — `x_0 = e₀`, `x_k = 3 · 2^{k-1} e₀` afterwards. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (∀ (_ : ℕ) (_ : Idx 1), (1 : ℝ) + 1 ≤ 2) ∧
    (0 : ℕ) ≤ 0 ∧
    (∀ (k : ℕ) (i : Idx 1), k ≠ 0 →
      (fun k (_ : Idx 1) =>
          (if k = 0 then (1 : ℝ) else 3 / 2 * 2 ^ k) • (basePoint 0 : EucSpace 1)) (k + 1) i
        = (2 : ℝ) • (fun k (_ : Idx 1) =>
            (if k = 0 then (1 : ℝ) else 3 / 2 * 2 ^ k) • (basePoint 0 : EucSpace 1)) k i + 0) ∧
    (∀ i : Idx 1,
      (fun k (_ : Idx 1) =>
          (if k = 0 then (1 : ℝ) else 3 / 2 * 2 ^ k) • (basePoint 0 : EucSpace 1)) (0 + 1) i
        = (2 : ℝ) • (fun k (_ : Idx 1) =>
            (if k = 0 then (1 : ℝ) else 3 / 2 * 2 ^ k) • (basePoint 0 : EucSpace 1)) 0 i
          + (1 : ℝ) • (fun k (_ : Idx 1) =>
            (if k = 0 then (1 : ℝ) else 3 / 2 * 2 ^ k) • (basePoint 0 : EucSpace 1)) 0 i + 0) ∧
    (∀ (_ : ℕ) (_ : Idx 1), ‖(0 : EucSpace 1)‖ ≤ 0) ∧
    ∀ _ : Idx 1, (1 : ℝ) • (basePoint 0 : EucSpace 1) ≠ 0 := by
  refine ⟨one_pos, le_rfl, fun _ _ => by norm_num, le_rfl, fun k _ hk => ?_, fun _ => ?_,
    fun _ _ => by simp, fun _ => by simp [basePoint]⟩
  · simp only [ite_eq_right hk, ite_eq_right (Nat.succ_ne_zero k), add_zero, smul_smul, pow_succ]
    congr 1
    ring
  · norm_num
    module

end Perspective
end Transformer
