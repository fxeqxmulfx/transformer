/-
# Freezing survives any number of skips

**Not a statement of any paper.**  `rawStack_frozen` of
`Perspective.RawStackFrozen` freezes the directions of a stack whose gains
exceed `1 + c` and whose blocks output at most `M`, and `rawStack_frozen_skip`
of `Perspective.RawStackSkipFrozen` lets one update add an earlier state.  Here
every update may add any number of them,

  `x_{k+1,i} = λ_{k,i} x_{k,i} + Σ_{j ≤ k} s_{k,j,i} x_{j,i} + g_{k,i}`,

and the damped gauge meets the Gronwall bound of
`Perspective.RawStackDenseGronwall`: the skip-free gauge stack stays within
`M / c` of the start (`norm_gaugeStack_sub_le`), and each rescaled skip weight
is at most its gate damped by the depth it jumps (`abs_skipWeight_le`), so
`W_{j,i} ≤ V_{j,i} = Σ_{l ≤ j} |s_{j,l,i}| (1+c)^{-(j+1-l)}` (`skipMass_le`).
At every depth, whatever the blocks compute, the direction of token `i` is
within

  `2 (M/c + (‖x_{0,i}‖ + M/c) (Π_{j<k} (1 + V_{j,i}) - 1)) / ‖x_{0,i}‖`

of the one it started with (`rawStack_frozen_dense`).  Without skips the product
is `1` and this is the bound of `rawStack_frozen`; with one skip it is at most
that of `rawStack_frozen_skip`, and equal to it past the skip.

Of the record this covers every update, all twenty-three.  A step of gain below
`1 + c` — the skip of layer `6` and the step after the loop have gain `1`, and
any gain, of any sign, will do — is taken at gain `1 + c`, the difference
carried by a skip onto the state it updates (`Perspective.RawStackDense`), which
adds `|gain - 1 - c| / (1 + c)` to its `V`; the skips proper, of layers `6` and
`10` and after the loop, are the other summands.  The price is the product:
every skipping step multiplies the growth by one plus what its skips weigh, and
skips at a fixed distance keep their weight at every depth
(`le_abs_skipWeight`), so over infinitely many the product need not converge.
It is a bound for a stack of given depth, which a record is.

What `M` is for the record changes at layer `4`.  The attentions and MLPs read
normalized states, and up to layer `4` the gates of the `x0` and bigram
injections are computed from the normalized embedding, so a constant of the
weights bounds the outputs.  From layer `4` on, the gates of those injections
and of the layer-`6` skip, with coefficients of three attentions, come from
`forward_mudd_gate` applied to the raw state at depth `8`; the gains and gates
of layer `10` and of the step after the loop come from `forward_mudd` applied
to the raw stream; and the values of the last attention also carry
`v_mudd = mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x`, raw states that the
attention mixes across tokens (`train_gpt.py`, `GPT.forward`).  There `M`, `λ`
and `s` are properties of the trajectory, and `rawStack_frozen_dense` is a
bound in terms of them.  At initialisation `mudd_w2` and `mudd_gate_w2` are
zero, and every coefficient is a constant.
-/

import Transformer.Perspective.RawStackDenseGronwall
import Transformer.Perspective.RawStackFrozen
import Transformer.Perspective.RawStackSkipDamp

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **What the skips of one step weigh, damped:** with every gain at least
`1 + c`, `W_{k,i} ≤ Σ_{j ≤ k} |s_{k,j,i}| (1+c)^{-(k+1-j)}`, each gate damped by
the depth it jumps.

Source: none — posed here; `abs_skipWeight_le`, summed over the depths read. -/
theorem skipMass_le {lam : ℕ → Idx n → ℝ} {s : ℕ → ℕ → Idx n → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hlam : ∀ j i, 1 + c ≤ lam j i) (k : ℕ) (i : Idx n) :
    skipMass lam s k i ≤ ∑ j ∈ Finset.range (k + 1), |s k j i| * ((1 + c)⁻¹) ^ (k + 1 - j) :=
  Finset.sum_le_sum fun j hj =>
    abs_skipWeight_le (s := fun k i => s k j i) hc hlam (Finset.mem_range.1 hj).le i

/-- **The growth, damped:** with every gain at least `1 + c`, `P_{k,i}` is at
most `Π_{j<k} (1 + Σ_{l ≤ j} |s_{j,l,i}| (1+c)^{-(j+1-l)})`, a product of the
gates alone.

Source: none — posed here; `skipMass_le`, factor by factor. -/
theorem skipGrowth_le {lam : ℕ → Idx n → ℝ} {s : ℕ → ℕ → Idx n → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hlam : ∀ j i, 1 + c ≤ lam j i) (k : ℕ) (i : Idx n) :
    skipGrowth lam s k i ≤ ∏ j ∈ Finset.range k,
      (1 + ∑ l ∈ Finset.range (j + 1), |s j l i| * ((1 + c)⁻¹) ^ (j + 1 - l)) :=
  Finset.prod_le_prod₀ (fun j _ => by linarith [skipMass_nonneg lam s j i])
    fun j _ => by linarith [skipMass_le (s := s) hc hlam j i]

/-- **A stack of growing blocks with any number of skips still freezes its own
directions.**  Every gain at least `1 + c`, every block output at most `M`, and
at every step any number of skips onto any depth so far, with a gate per step,
depth and token: then at every depth the direction of token `i` is within

  `2 (M/c + (‖x_{0,i}‖ + M/c) (Π_{j<k} (1 + V_{j,i}) - 1)) / ‖x_{0,i}‖`,
  `V_{j,i} = Σ_{l ≤ j} |s_{j,l,i}| (1+c)^{-(j+1-l)}`,

of the direction it started with.  Without skips this is `rawStack_frozen`'s
bound, and with one skip at most `rawStack_frozen_skip`'s.

Source: none — posed here; every update of modded-nanogpt (`train_gpt.py`,
`GPT.forward`), the skips of layer `6`, of layer `10` and after the loop
included, a gain below `1 + c` carried by a skip onto the state it updates. -/
theorem rawStack_frozen_dense {lam : ℕ → Idx n → ℝ} {s : ℕ → ℕ → Idx n → ℝ}
    {x g : ℕ → Idx n → EucSpace d} {c M : ℝ} (hc : 0 < c) (hM : 0 ≤ M)
    (hlam : ∀ j i, 1 + c ≤ lam j i)
    (hx : ∀ k i, x (k + 1) i
      = lam k i • x k i + ∑ j ∈ Finset.range (k + 1), s k j i • x j i + g k i)
    (hg : ∀ j i, ‖g j i‖ ≤ M) (h0 : ∀ i, x 0 i ≠ 0) (k : ℕ) (i : Idx n) :
    ‖‖x k i‖⁻¹ • x k i - ‖x 0 i‖⁻¹ • x 0 i‖
      ≤ 2 * (M / c + (‖x 0 i‖ + M / c) * (∏ j ∈ Finset.range k,
          (1 + ∑ l ∈ Finset.range (j + 1), |s j l i| * ((1 + c)⁻¹) ^ (j + 1 - l)) - 1))
        / ‖x 0 i‖ := by
  have hpos : ∀ (j : ℕ) (i : Idx n), 0 < lam j i := fun j i =>
    lt_of_lt_of_le (by linarith) (hlam j i)
  refine (norm_normalize_sub_le_dense (s := s) hpos hx i
    (fun j _ => norm_gaugeStack_sub_le (x₀ := x 0) hc hM hlam hg j i) (h0 i)).trans ?_
  have hP := skipGrowth_le (s := s) hc.le hlam k i
  have hMc : 0 ≤ ‖x 0 i‖ + M / c := by positivity
  gcongr

/-- The hypotheses of `skipMass_le`, `skipGrowth_le` and `rawStack_frozen_dense`
are satisfiable, with skips that do something: gains `2` (so `c = 1`), no block
output (`M = 0`), a skip of gate `1` onto the start at every step, and the stack
it forces, `x_k = (2^{k+1} - 1) e₀`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (∀ (_ : ℕ) (_ : Idx 1), (1 : ℝ) + 1 ≤ 2) ∧
    (∀ (k : ℕ) (i : Idx 1),
      (fun k (_ : Idx 1) => ((2 : ℝ) ^ (k + 1) - 1) • (basePoint 0 : EucSpace 1)) (k + 1) i
        = (2 : ℝ)
            • (fun k (_ : Idx 1) => ((2 : ℝ) ^ (k + 1) - 1) • (basePoint 0 : EucSpace 1)) k i
          + ∑ j ∈ Finset.range (k + 1), (if j = 0 then (1 : ℝ) else 0)
            • (fun k (_ : Idx 1) => ((2 : ℝ) ^ (k + 1) - 1) • (basePoint 0 : EucSpace 1)) j i
          + 0) ∧
    (∀ (_ : ℕ) (_ : Idx 1), ‖(0 : EucSpace 1)‖ ≤ 0) ∧
    ∀ _ : Idx 1, ((2 : ℝ) ^ (0 + 1) - 1) • (basePoint 0 : EucSpace 1) ≠ 0 := by
  refine ⟨one_pos, le_rfl, fun _ _ => by norm_num, fun k _ => ?_, fun _ _ => by simp,
    fun _ => by norm_num [basePoint]⟩
  simp only [ite_smul, one_smul, zero_smul, Finset.sum_ite_eq', Finset.mem_range,
    Nat.zero_lt_succ, ite_true, add_zero, pow_succ]
  module

end Perspective
end Transformer
