/-
# A stack of different blocks, and the gauge that its residual gains are

**Not a statement of any paper.**  Each block of modded-nanogpt
(github.com/KellerJordan/modded-nanogpt, `train_gpt.py`, `GPT.forward`)
rescales the residual stream before adding to it,

  `x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0 * x0_gates[i]`,
  `x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * mlp_fn(normed, *mlp_args)`

(`normed = norm(x)`), and `resid_lambdas` is one *scalar* per layer and
sublayer, `nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))`, above `1` at
the start.  A step of the stack below is one of those updates, so a model of
eleven layers is a stack of twenty-two.

`Perspective.RawGrowth` had that gain in continuous time, as one constant `c`
in `ẋ = c x + g`: one block, applied for a while.  A stack is not that — eleven
blocks, each with its own gain and its own output — and in discrete time
nothing has to be assumed away to say so: `x_{k+1,i} = λ_{k,i} x_{k,i} + g_{k,i}`
with the gains positive and the outputs `g_{k,i}` arbitrary, attention, MLP and
injection of the `k`-th block together.

The gains then factor out exactly (`rawStack_eq_gauge`): the stack is its gauge
`Λ_{k,i} = ∏_{j<k} λ_{j,i}` times the stack of the same blocks with the gains
removed, whose `k`-th step is `g_{k,i} / Λ_{k+1,i}`.  A positive scalar does not
turn a vector, so the directions — all a pre-norm block ever reads — are those
of the gauge stack whatever the gains are (`normalize_rawStack`), and only the
ratios `‖g_{k,i}‖ / Λ_{k+1,i}` reach them.

The gain may differ from token to token here, which is more than
modded-nanogpt asks: a gate on the residual branch is a gauge too.  A gain that
differs from channel to channel — `mix[0] ⊙ x` of parameter-golf — is not.

`Perspective.RawStackFrozen` reads the ratios off a stack whose gains exceed
`1`, and finds `rawStream_frozen` there without its continuous time.
-/

import Transformer.Perspective.RawStream

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The gauge of a stack:** the gain token `i` has accumulated over the first
`k` blocks, `Λ_{k,i} = ∏_{j<k} λ_{j,i}`. -/
noncomputable def gainProd (lam : ℕ → Idx n → ℝ) (k : ℕ) (i : Idx n) : ℝ :=
  ∏ j ∈ Finset.range k, lam j i

/-- **The stack with its gains divided out:**
`x_{0,i} + Σ_{j<k} g_{j,i} / Λ_{j+1,i}`, where the same blocks are added to a
stream whose every gain is `1`. -/
noncomputable def gaugeStack (lam : ℕ → Idx n → ℝ) (x₀ : Idx n → EucSpace d)
    (g : ℕ → Idx n → EucSpace d) (k : ℕ) (i : Idx n) : EucSpace d :=
  x₀ i + ∑ j ∈ Finset.range k, (gainProd lam (j + 1) i)⁻¹ • g j i

/-- **Positive gains, positive gauge.**

Source: none — posed here; `resid_lambdas` of modded-nanogpt starts at
`1.1^{1/2}` and the statements below need only its sign. -/
theorem gainProd_pos {lam : ℕ → Idx n → ℝ} (hlam : ∀ j i, 0 < lam j i) (k : ℕ) (i : Idx n) :
    0 < gainProd lam k i :=
  Finset.prod_pos fun j _ => hlam j i

/-- **A stack of positive gains is its gauge times the stack without them.**
If `x_{k+1,i} = λ_{k,i} x_{k,i} + g_{k,i}` with every `λ_{k,i} > 0`, then

  `x_{k,i} = Λ_{k,i} (x_{0,i} + Σ_{j<k} g_{j,i} / Λ_{j+1,i})`.

The blocks are arbitrary and all different: `g_{k,i}` is whatever the `k`-th
block adds to token `i`.

Source: none — posed here; the residual update of modded-nanogpt
(`train_gpt.py`, `GPT.forward`), `x ← resid_lambdas[k] * x + (block output)`,
solved for the stream. -/
theorem rawStack_eq_gauge {lam : ℕ → Idx n → ℝ} {x g : ℕ → Idx n → EucSpace d}
    (hlam : ∀ j i, 0 < lam j i) (hx : ∀ k i, x (k + 1) i = lam k i • x k i + g k i)
    (k : ℕ) (i : Idx n) : x k i = gainProd lam k i • gaugeStack lam (x 0) g k i := by
  induction k with
  | zero => simp [gainProd, gaugeStack]
  | succ k ih =>
      have hne : gainProd lam (k + 1) i ≠ 0 := (gainProd_pos hlam (k + 1) i).ne'
      have hstep : gainProd lam k i * lam k i = gainProd lam (k + 1) i :=
        (Finset.prod_range_succ (fun j => lam j i) k).symm
      calc x (k + 1) i = lam k i • (gainProd lam k i • gaugeStack lam (x 0) g k i) + g k i := by
            rw [hx k i, ih]
        _ = gainProd lam (k + 1) i • gaugeStack lam (x 0) g k i + g k i := by
            rw [smul_smul, mul_comm, hstep]
        _ = gainProd lam (k + 1) i •
              (gaugeStack lam (x 0) g k i + (gainProd lam (k + 1) i)⁻¹ • g k i) := by
            rw [smul_add, smul_smul, mul_inv_cancel₀ hne, one_smul]
        _ = gainProd lam (k + 1) i • gaugeStack lam (x 0) g (k + 1) i := by
            rw [gaugeStack, gaugeStack, Finset.sum_range_succ, add_assoc]

/-- **The gains are invisible behind the norm.**  The direction of the stack at
depth `k` is the direction of the same stack with its gains divided out — for
any positive gains, per token as well as per block.

Source: none — posed here; a pre-norm block reads `norm(x)` only
(modded-nanogpt, `train_gpt.py`, `GPT.forward`), and `norm` is homogeneous. -/
theorem normalize_rawStack {lam : ℕ → Idx n → ℝ} {x g : ℕ → Idx n → EucSpace d}
    (hlam : ∀ j i, 0 < lam j i) (hx : ∀ k i, x (k + 1) i = lam k i • x k i + g k i)
    (k : ℕ) (i : Idx n) : ‖x k i‖⁻¹ • x k i
      = ‖gaugeStack lam (x 0) g k i‖⁻¹ • gaugeStack lam (x 0) g k i := by
  rw [rawStack_eq_gauge hlam hx k i, normalize_smul_of_pos (gainProd_pos hlam k i)]

/-- The hypotheses of `gainProd_pos`, `rawStack_eq_gauge` and
`normalize_rawStack` are satisfiable: gains `1`, no block output, a stream at
rest. -/
example : (∀ (_ : ℕ) (_ : Idx 1), (0 : ℝ) < 1) ∧
    ∀ (k : ℕ) (i : Idx 1), (fun _ (_ : Idx 1) => (basePoint 0 : EucSpace 1)) (k + 1) i
      = (1 : ℝ) • (fun _ (_ : Idx 1) => (basePoint 0 : EucSpace 1)) k i + 0 :=
  ⟨fun _ _ => one_pos, fun _ _ => by simp⟩

end Perspective
end Transformer
