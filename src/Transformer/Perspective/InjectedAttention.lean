/-
# Softmax attention with the input mixed back in

**Not a statement of any paper.**  The dynamics `eq: transformerSd.QKV` of
arXiv:2312.10794v5 (§2.2) with `V = I_d` and a constant vector `z_i` added to
the drive of token `i`,

  `ẋ_i = Proj_{x_i}( Z_{β,i}⁻¹ Σ_j e^{β⟨Q(t) x_i, K(t) x_j⟩} x_j + z_i )`
  (`injectedODE`).

At `z = 0` it is `eq: transformerSd.QKV` with `V = I_d`
(`injectedODE_zero_iff`), which for constant `Q`, `K` collapses from any open
hemisphere (`expConvergent_injectedODE_zero`, from `cone_collapse`,
`lem: hemisphere.clustering`, §6.1).  For every `z` it is an injected flow
whose weights, the attention matrix `eq:P`, have row sums `1`
(`injectedODE.isInjectedFlow`), so `IsInjectedFlow.spread` applies to it;
`Perspective.InjectedConsensus` draws the consequences.

The model is the residual mix of a block in parameter-golf
(github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`,
`Block.forward`): `x_in = mix[0] * x + mix[1] * x0` adds the embedding `x0` of
every token back in before every block, the injection `z_i = mix[1] ⊙ x0_i`.
What this model leaves out: the MLP, the RMS norm (the sphere stands in for
it), `mix[0] ≠ 1`, `attn_scale`, `V ≠ I_d`, and a `mix[1]` that changes from
block to block — `z` is constant in time, as for one block applied again and
again.  `Perspective.BlockDrive` and `Perspective.BlockSpread` take in all of
these but the norm, for a stack of different blocks, at the price of two
independent differences `z_k - z_l` in place of two independent injections,
a price `Perspective.BlockConsensus` shows cannot be avoided.
`Perspective.RawStream` and `Perspective.RawGrowth` take in the norm, and for
`mix[0] > 1` in every channel the conclusion does not survive it.
-/

import Transformer.Perspective.InjectedSpread
import Transformer.Perspective.Section5_ExpRate

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

variable (d n) in
/-- **Softmax attention with injection:** `eq: transformerSd.QKV` with
`V = I_d` and the constant vector `z_i` added to the drive of token `i`,

  `ẋ_i = Proj_{x_i}( Z_{β,i}⁻¹ Σ_j e^{β⟨Q(t) x_i, K(t) x_j⟩} x_j + z_i )`.

Source: none — posed here; `transformerODE` (arXiv:2312.10794v5, §2.2,
`eq: transformerSd.QKV`) with `V = I_d` and `z_i` added inside the projection,
at the place where `eq: albert` (§2.3) adds the feed-forward term. -/
def injectedODE (β : ℝ) (Q K : TimeParam d) (z : Idx n → EucSpace d)
    (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        (((partitionQKV d n β Q K X t i)⁻¹ •
          ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ)
                        ((Q t) ((X t i : EucSpace d)))
                        ((K t) ((X t j : EucSpace d))))
            • (X t j : EucSpace d)) + z i)) t

/-- Without injection, `injectedODE` is `eq: transformerSd.QKV` with `V = I_d`.

Source: arXiv:2312.10794v5, §2.2, `eq: transformerSd.QKV` at `V = I_d`. -/
theorem injectedODE_zero_iff (β : ℝ) (Q K : TimeParam d) (X : ℝ → SphereTuple d n) :
    injectedODE d n β Q K 0 X ↔
      transformerODE d n β Q K (fun _ => ContinuousLinearMap.id ℝ _) X := by
  simp only [injectedODE, transformerODE, Pi.zero_apply, add_zero, ContinuousLinearMap.id_apply]

/-- **Without injection, collapse** — `lem: hemisphere.clustering` for
`eq: transformerSd.QKV` with `V = I_d`: for constant `Q`, `K` and initial
tokens in an open hemisphere, every solution of `injectedODE` at `z = 0`
converges to one point at an exponential rate.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, through
`cone_collapse`, whose departures from the survey it inherits: `β > 0` and
`d ≥ 2` are dropped, and a solution is one on all of `ℝ`. -/
theorem expConvergent_injectedODE_zero (β : ℝ) (Q K : ParamMatrix d) (X₀ : SphereTuple d n)
    (hX₀ : ∃ w : SSphere d, ∀ i : Idx n,
      0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d))) :
    ExpConvergent d n (injectedODE d n β (fun _ => Q) (fun _ => K) 0) X₀ := by
  have h : injectedODE d n β (fun _ => Q) (fun _ => K) 0 =
      transformerODE d n β (fun _ => Q) (fun _ => K) (fun _ => ContinuousLinearMap.id ℝ _) :=
    funext fun X => propext (injectedODE_zero_iff β _ _ X)
  rw [h]
  exact ((cone_collapse d n β X₀ hX₀).2.2 Q K).1

/-- The hypothesis of `expConvergent_injectedODE_zero` is satisfiable: one
token, at the pole the hemisphere is centred on. -/
example : ∃ w : SSphere 1, ∀ i : Idx 1,
    0 < inner (𝕜 := ℝ) (((fun _ => basePoint 0 : SphereTuple 1 1) i : EucSpace 1))
      ((w : EucSpace 1)) :=
  ⟨basePoint 0, fun _ => by simp [basePoint]⟩

/-- **The attention weights of a token sum to one:** `Σ_j |A_ij(t)| = 1` for
the attention matrix `eq:P`.

Source: arXiv:2312.10794v5, §2.2, `eq:P` and `eq: SA.QKV`. -/
theorem sum_abs_attention (β : ℝ) (Q K : TimeParam d) (X : ℝ → SphereTuple d n) (t : ℝ)
    (i : Idx n) : ∑ j, |attention d n β Q K X t i j| = 1 := by
  have hZ : 0 < partitionQKV d n β Q K X t i :=
    Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩
  simp only [attention, abs_div, Real.abs_exp, abs_of_pos hZ, ← Finset.sum_div]
  exact div_self hZ.ne'

/-- **The rows of the attention matrix sum to one:** `Σ_j A_ij(t) = 1` for the
attention matrix `eq:P`, for every `β`, `Q`, `K` and every configuration.

Source: arXiv:2312.10794v5, §2.2, after `eq:P`: "The `n × n` stochastic matrix
`A(t)` (rows are probability vectors)". -/
theorem sum_attention (β : ℝ) (Q K : TimeParam d) (X : ℝ → SphereTuple d n) (t : ℝ)
    (i : Idx n) : ∑ j, attention d n β Q K X t i j = 1 := by
  have hZ : 0 < partitionQKV d n β Q K X t i :=
    Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩
  simp only [attention, ← Finset.sum_div]
  exact div_self hZ.ne'

/-- **Softmax attention with injection is an injected flow**, with the
attention matrix `eq:P` as its weights.

Source: none — posed here; `eq:P` of arXiv:2312.10794v5, §2.2. -/
theorem injectedODE.isInjectedFlow {β : ℝ} {Q K : TimeParam d} {z : Idx n → EucSpace d}
    {X : ℝ → SphereTuple d n} (hX : injectedODE d n β Q K z X) :
    IsInjectedFlow X (attention d n β Q K X) z := fun t i =>
  (hX t i).congr_deriv (by simp only [attention, div_eq_inv_mul, mul_smul, Finset.smul_sum])

/-- **Rest points.**  A configuration `Y` at which every drive is normal to the
sphere, `Z_{β,i}⁻¹ Σ_j e^{β⟨Q(t) y_i, K(t) y_j⟩} y_j + z_i ∈ ℝ y_i` at all
times, is a solution of `injectedODE` that stays at `Y`.

Source: none — posed here, for `injectedODE`. -/
theorem injectedODE_const {β : ℝ} {Q K : TimeParam d} {z : Idx n → EucSpace d}
    {Y : SphereTuple d n}
    (h : ∀ t i, ∃ c : ℝ, ((partitionQKV d n β Q K (fun _ => Y) t i)⁻¹ •
      ∑ j, Real.exp (β * inner (𝕜 := ℝ) ((Q t) (Y i : EucSpace d)) ((K t) (Y j : EucSpace d)))
        • (Y j : EucSpace d)) + z i = c • (Y i : EucSpace d)) :
    injectedODE d n β Q K z (fun _ => Y) := fun t i => by
  obtain ⟨c, hc⟩ := h t i
  refine (hasDerivAt_const t (Y i : EucSpace d)).congr_deriv ?_
  beta_reduce
  rw [hc, proj_smul_self (norm_coe_tuple Y i)]

/-- **Parallel injections allow consensus.**  If every `z_i = c_i x` is a
multiple of one unit vector `x`, all tokens resting at `x` is a solution of
`injectedODE`: the linear independence of two injections that rules collapse
out cannot be dropped.

Source: none — posed here, for `injectedODE`. -/
theorem injectedODE_consensus (β : ℝ) (Q K : TimeParam d) (x : SSphere d) (c : Idx n → ℝ) :
    injectedODE d n β Q K (fun i => c i • (x : EucSpace d)) (fun _ _ => x) :=
  injectedODE_const fun t i => ⟨_, by rw [← Finset.sum_smul, smul_smul, ← add_smul]⟩

/-- The hypotheses of `injectedODE.isInjectedFlow` and of `injectedODE_const`
are satisfiable: all tokens at one point, as in `injectedODE_consensus`. -/
example : injectedODE 1 1 0 (fun _ => 0) (fun _ => 0)
    (fun i => (fun _ => (1 : ℝ)) i • (basePoint 0 : EucSpace 1)) (fun _ _ => basePoint 0) :=
  injectedODE_consensus 0 _ _ (basePoint 0) fun _ => 1

end Perspective
end Transformer
