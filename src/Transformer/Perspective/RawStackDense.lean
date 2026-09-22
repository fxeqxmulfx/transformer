/-
# Any number of skips, onto any depth

**Not a statement of any paper.**  `Perspective.RawStackSkip` takes one skip per
step, and `Perspective.RawStackSkipFrozen` one skip in the whole stack.
Modded-nanogpt (github.com/KellerJordan/modded-nanogpt, `train_gpt.py`,
`GPT.forward`) skips in three places, and two of them add several earlier
states in one step:

- layer `6` skips attention and adds `skip_gate_out * cache[3]`, the state at
  depth `8`, at gain `1`;
- layer `10` recombines its input, `x = (1 + mu[5]) * x + mu[3] * cache[0] +
  mu[4] * cache[7]`, and nothing but its attention step reads the result,
  `x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0]`: one step of gain
  `mu[8] (1 + mu[5])` that adds the states at depths `0` and `16`;
- the step after the loop adds `mu[0] * cache[0] + mu[1] * cache[7] +
  mu[2] * cache[9] + mu[4] * cache[3]`, the states at depths `0`, `16`, `20`
  and `8`, at gain `1`,

where every `mu[·]` is one scalar per token (`GPT.forward_mudd`).  So every
update of the record — all twenty-three, counting sublayers — is of the form

  `x_{k+1,i} = λ_{k,i} x_{k,i} + Σ_{j ≤ k} s_{k,j,i} x_{j,i} + g_{k,i}`,

with the attention and MLP outputs, the `x0` and bigram injections and
`mu[3] * ve_bank0` in `g`.  The sum runs up to `j = k`, and that is not a
detail: a skip onto the current state is a change of gain, so `λ` may be chosen
freely — `1 + c`, say — and the step's own gain, of any sign, carried by the
self-skip `s_{k,k,i} = (its gain) - λ_{k,i}`.

The gauge is still exact.  Dividing out `Λ_{k,i} = Π_{j<k} λ_{j,i}` sets every
gain to `1` and every skip weight to `s_{k,j,i} Λ_{j,i} / Λ_{k+1,i}`
(`ungauged_rec_dense`), and the ungauged stream is the skip-free gauge stack
plus all the rescaled skips taken so far (`ungauged_sub_gaugeStack_dense`).
`Perspective.RawStackDenseGronwall` bounds that sum, and the directions with it,
and `Perspective.RawStackDenseFrozen` reads the bound off gains of at least
`1 + c` and outputs of at most `M`.
-/

import Transformer.Perspective.RawStackSkip

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Any number of skips is gauge-covariant.**  If

  `x_{k+1,i} = λ_{k,i} x_{k,i} + Σ_{j ≤ k} s_{k,j,i} x_{j,i} + g_{k,i}`

with every gain positive, then the ungauged stream obeys the same recursion
with every gain `1`, the step `g_{k,i}/Λ_{k+1,i}` and the skip weights
`s_{k,j,i} Λ_{j,i} / Λ_{k+1,i}`.  The gates are arbitrary, one per step, depth
read and token, and so are the outputs.

Source: none — posed here; the skips of modded-nanogpt at layer `6`, at layer
`10` and after the loop (`train_gpt.py`, `GPT.forward`, with the per-token
coefficients of `GPT.forward_mudd`), several in one step, against the residual
update `ungauged_step` is written for. -/
theorem ungauged_rec_dense {lam : ℕ → Idx n → ℝ} {s : ℕ → ℕ → Idx n → ℝ}
    {x g : ℕ → Idx n → EucSpace d} (hlam : ∀ j i, 0 < lam j i)
    (hx : ∀ k i, x (k + 1) i
      = lam k i • x k i + ∑ j ∈ Finset.range (k + 1), s k j i • x j i + g k i)
    (k : ℕ) (i : Idx n) :
    ungauged lam x (k + 1) i
      = ungauged lam x k i
        + ∑ j ∈ Finset.range (k + 1),
            (s k j i * gainProd lam j i / gainProd lam (k + 1) i) • ungauged lam x j i
        + (gainProd lam (k + 1) i)⁻¹ • g k i := by
  have hkne : gainProd lam (k + 1) i ≠ 0 := (gainProd_pos hlam (k + 1) i).ne'
  have hterm : ∀ j ∈ Finset.range (k + 1),
      (gainProd lam (k + 1) i)⁻¹ • s k j i • x j i
        = (s k j i * gainProd lam j i / gainProd lam (k + 1) i) • ungauged lam x j i := by
    intro j _
    have hjne : gainProd lam j i ≠ 0 := (gainProd_pos hlam j i).ne'
    rw [ungauged, smul_smul, smul_smul]
    congr 1
    field_simp
  rw [ungauged_step hlam (by rw [hx k i, add_assoc]), smul_add, Finset.smul_sum,
    Finset.sum_congr rfl hterm, add_assoc]

/-- **The ungauged stream is its gauge stack plus the skips it has taken.**  The
difference between the ungauged stream and the skip-free gauge stack of
`Perspective.RawStack` is the sum, over every step so far and every depth that
step reads, of the rescaled skip weight times the ungauged state it reads.
Unlike a single skip (`ungauged_eq_gaugeStack_add_skip`), this is not a closed
form: the states on the right are the stream's own.

Source: none — posed here; `ungauged_rec_dense`, summed over the depths. -/
theorem ungauged_sub_gaugeStack_dense {lam : ℕ → Idx n → ℝ} {s : ℕ → ℕ → Idx n → ℝ}
    {x g : ℕ → Idx n → EucSpace d} (hlam : ∀ j i, 0 < lam j i)
    (hx : ∀ k i, x (k + 1) i
      = lam k i • x k i + ∑ j ∈ Finset.range (k + 1), s k j i • x j i + g k i)
    (k : ℕ) (i : Idx n) :
    ungauged lam x k i - gaugeStack lam (x 0) g k i
      = ∑ j ∈ Finset.range k, ∑ l ∈ Finset.range (j + 1),
          (s j l i * gainProd lam l i / gainProd lam (j + 1) i) • ungauged lam x l i := by
  induction k with
  | zero => simp [ungauged, gaugeStack, gainProd]
  | succ k ih =>
      rw [ungauged_rec_dense hlam hx, gaugeStack_succ]
      conv_rhs => rw [Finset.sum_range_succ, ← ih]
      abel

/-- The hypotheses of `ungauged_rec_dense` and `ungauged_sub_gaugeStack_dense`
are satisfiable, with skips that do something: gains `1`, no block output, a
skip of gate `1` onto the start at every step, and the stack it forces,
`x_k = (k + 1) e₀`. -/
example : (∀ (_ : ℕ) (_ : Idx 1), (0 : ℝ) < 1) ∧
    ∀ (k : ℕ) (i : Idx 1),
      (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) (k + 1) i
        = (1 : ℝ) • (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) k i
          + ∑ j ∈ Finset.range (k + 1), (if j = 0 then (1 : ℝ) else 0)
            • (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) j i
          + 0 := by
  refine ⟨fun _ _ => one_pos, fun k _ => ?_⟩
  simp only [ite_smul, one_smul, zero_smul, Finset.sum_ite_eq', Finset.mem_range,
    Nat.zero_lt_succ, ite_true, add_zero]
  push_cast
  module

end Perspective
end Transformer
