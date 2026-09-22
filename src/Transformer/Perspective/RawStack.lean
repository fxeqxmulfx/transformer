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

/-- **One step of the gauge stack:** the `k`-th block enters it divided by the
gauge it is added behind. -/
theorem gaugeStack_succ (lam : ℕ → Idx n → ℝ) (x₀ : Idx n → EucSpace d)
    (g : ℕ → Idx n → EucSpace d) (k : ℕ) (i : Idx n) :
    gaugeStack lam x₀ g (k + 1) i
      = gaugeStack lam x₀ g k i + (gainProd lam (k + 1) i)⁻¹ • g k i := by
  rw [gaugeStack, gaugeStack, Finset.sum_range_succ, add_assoc]

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
            rw [gaugeStack_succ]

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

/-- **A gain that differs from channel to channel is not a gauge.**  The claim
refuted: `normalize_rawStack` for a per-channel gain — that a stack whose
blocks do nothing but rescale each channel by a positive factor,
`x_{k+1} = A x_k` with `A e_p = a_p e_p`, `a_p > 0`, and no output at all, keeps
the direction it started with.  It fails in `ℝ²` already, at `a = (2, 1)` and
`x_k = 2^k e₀ + e₁`: the direction at depth `1` is not the direction at depth
`0`, and deeper in the stack it turns all the way onto `e₀`.

So `mix[0] ⊙ x` of parameter-golf
(github.com/openai/parameter-golf, `records/track_10min_16mb/`
`2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`, `Block.forward`) is
not a change of variable, while the scalar `resid_lambdas` of modded-nanogpt is
one whatever it does (`normalize_rawStack`) — and a gain per token is one too.
A gain that is not a multiple of the identity has an effect on the directions
that no amount of scalar gain can imitate.

Source: none — posed here; it refutes the extension of `normalize_rawStack` to
a gain acting channel by channel. -/
theorem not_channelGain_gauge :
    ¬ ∀ (e : ℕ) (A : EucSpace e →L[ℝ] EucSpace e) (a : Idx e → ℝ) (x : ℕ → EucSpace e),
      (∀ p, 0 < a p) →
      (∀ p, A (EuclideanSpace.single p 1) = a p • EuclideanSpace.single p 1) →
      (∀ k, x (k + 1) = A (x k)) → (∀ k, x k ≠ 0) →
      ∀ k, ‖x k‖⁻¹ • x k = ‖x 0‖⁻¹ • x 0 := by
  intro h
  have huu : inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) = 1 := by simp
  have hww : inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ))
      (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 1 := by simp
  have huw : inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 0 := by
    simp [EuclideanSpace.inner_single_left]
  have hwu : inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ))
      (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) = 0 := by
    simp [EuclideanSpace.inner_single_left]
  have hiu : ∀ c : ℝ, inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (c • EuclideanSpace.single (0 : Fin 2) (1 : ℝ)
        + EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = c := fun c => by
    rw [inner_add_right, real_inner_smul_right, huu, huw, mul_one, add_zero]
  have hiw : ∀ c : ℝ, inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ))
      (c • EuclideanSpace.single (0 : Fin 2) (1 : ℝ)
        + EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 1 := fun c => by
    rw [inner_add_right, real_inner_smul_right, hwu, hww, mul_zero, zero_add]
  have hne : ∀ k : ℕ, (2 : ℝ) ^ k • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      + EuclideanSpace.single (1 : Fin 2) (1 : ℝ) ≠ 0 := fun k hk => by
    have h1 := congrArg (inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ))) hk
    rw [hiw, inner_zero_right] at h1
    exact one_ne_zero h1
  have key := h 2
    (ContinuousLinearMap.id ℝ (EucSpace 2)
      + (innerSL ℝ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))).smulRight
        (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)))
    ![2, 1]
    (fun k => (2 : ℝ) ^ k • EuclideanSpace.single (0 : Fin 2) (1 : ℝ)
      + EuclideanSpace.single (1 : Fin 2) (1 : ℝ))
    (fun p => by fin_cases p <;> norm_num)
    (fun p => by fin_cases p <;> simp [two_smul, EuclideanSpace.inner_single_left])
    (fun k => by
      simp only [add_apply, ContinuousLinearMap.id_apply,
        ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, hiu, pow_succ]
      module)
    hne 1
  have hpar : ∀ c₁ c₀ : ℝ, c₁ • ((2 : ℝ) • EuclideanSpace.single (0 : Fin 2) (1 : ℝ)
      + EuclideanSpace.single (1 : Fin 2) (1 : ℝ))
      = c₀ • ((1 : ℝ) • EuclideanSpace.single (0 : Fin 2) (1 : ℝ)
        + EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) → c₁ = 0 := fun c₁ c₀ hc => by
    have h0 := congrArg (inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))) hc
    have h1 := congrArg (inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ))) hc
    rw [real_inner_smul_right, real_inner_smul_right, hiu, hiu] at h0
    rw [real_inner_smul_right, real_inner_smul_right, hiw, hiw] at h1
    linarith
  rw [pow_one, pow_zero] at key
  exact hne 1 (by
    rw [pow_one]
    exact norm_eq_zero.mp (inv_eq_zero.mp (hpar _ _ key)))

end Perspective
end Transformer
