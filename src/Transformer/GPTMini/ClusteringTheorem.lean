/-
# Clustering of `gpt-mini` representations

The capstone statement of the development:

> with the value projection equal to the identity, the token directions of
> `gpt-mini` cluster to one point of the unit sphere as the depth grows, for
> almost every initial configuration.

The chain of bridges that is supposed to prove it:

  1. `Bridge.SphereResidence`     — the RMSNorm direction lies on the sphere,
  2. `Bridge.RoPEAsTimeVarying`   — RoPE is a time-varying `Q, K`,
  3. `Bridge.CausalConnection`    — the causal mask matches `eq: csa`,
  4. `Bridge.XSAEquivalence`      — XSA at `V = I` is the spherical projection,
  5. `Causal.MainTheorem`         — clustering for causal SA with `V = I`,
  6. `Section5_HighD`             — the exponential rate inside a cap,
  7. `Normalization.Convergence`  — Pre-LN clustering.

None of the three statements below is proved here, and each rests on
`sorry`-leaves of the papers it composes, so each is a `Prop`-valued
definition.  Three things are worth reading off them before they are used.

*The layer dynamics are the Pre-LN residual recursion of one head*, written
out in `PreLNHead`: the sub-layer reads `rmsNormEps` of the stream and its
output is added back.  `V = I_d` is imposed structurally — the value argument
of `attentionHead` is the normalized stream itself, not a projection of it —
rather than as a hypothesis on `AttnParams`, whose single `W_qkv` has no
`V`-block accessor.  The one head and the absent FFN are a restriction: what
is stated is the depth behaviour of the attention recursion, not of
`GPTMini.forward`, whose `Block.attnSubLayer` is still a placeholder.

*The rate is polynomial in the depth, not exponential.*  The exponential rate
of `Section5_HighD` is exponential in the time of the continuous dynamics, and
under Pre-LN the layer index is not that time: the residual stream grows while
the sub-layer output stays bounded, so the direction moves less and less.
`LayerClustering` therefore states convergence only, and the quantitative
`PolynomialRate` states the `1/L³` of `thm: preln-slow`.

*`W₂` is a parameter* — Mathlib has no Wasserstein distance — and the
mean-field limit `n → ∞` is not taken: `MeanFieldClustering` is the finite-`T`
statement that the empirical distribution of the directions converges to a
Dirac, the limit in `T` not being expressible at a fixed `Fin T`.
-/

import Transformer.GPTMini.Bridge
import Transformer.GPTMini.Model
import Transformer.Perspective.Section5_HighD
import Transformer.Causal.MainTheorem
import Transformer.Normalization.Convergence
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.MeasureTheory.Constructions.Pi

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-- **The Pre-LN layer recursion of a single head with `V = I_d`.**

  `x_{L+1}(i) = x_L(i) + head(RMSNorm_eps(x_L))(i)`,

where `head` is `CausalMHA.attentionHead`: QK-norm, RoPE, causal softmax and
XSA, with the value stream equal to its own input — which is what `V = I_d`
means.  Source: `reference/model.py` (`Block.forward`), and
arXiv:2411.04990v2, `eq: csa`, for the sub-layer. -/
def PreLNHead (alpha eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (x : ℕ → Fin T → EucSpace cfg.head_dim) : Prop :=
  ∀ (L : ℕ) (i : Fin T),
    x (L + 1) i
      = x L i
        + attentionHead cfg alpha eps
            (fun j => rmsNormEps eps (x L j)) (fun j => rmsNormEps eps (x L j))
            (fun j => rmsNormEps eps (x L j)) positions i

/-- **Clustering of the representations.**

For `d_head ≥ 3`, for almost every initial stream — "almost every" for the
Lebesgue measure of `(ℝ^{d_head})^T`, the product of the volumes of
`EuclideanSpace ℝ (Fin d_head)` — every token direction `Φ(x_L(i))` converges,
as the depth `L` grows, to one common point `x_∞` of the unit sphere.

The stream is assumed to stay away from the origin, where the direction map
`Bridge.toSphere` is not defined.

Source: arXiv:2411.04990v2, §4 (`thm1`), through the bridges listed in the
module docstring. -/
def LayerClustering (T : ℕ) : Prop :=
  3 ≤ cfg.head_dim →
  ∀ (alpha eps : ℝ) (positions : Fin T → ℝ), 0 < eps →
    ∀ᵐ x₀ : Fin T → EucSpace cfg.head_dim, ∀ x : ℕ → Fin T → EucSpace cfg.head_dim,
      x 0 = x₀ → PreLNHead cfg alpha eps positions x → (∀ (L : ℕ) (i : Fin T), x L i ≠ 0) →
        ∃ xinf : EucSpace cfg.head_dim, ‖xinf‖ = 1 ∧
          ∀ i : Fin T,
            Filter.Tendsto (fun L : ℕ => Bridge.toSphere cfg.head_dim (x L i))
              Filter.atTop (nhds xinf)

/-- **Mean-field form of the same conclusion.**

The empirical distribution of the `T` token directions at depth `L`,

  `(1/T) Σ_i δ_{Φ(x_L(i))}`,

converges to `δ_{x_∞}` in the Wasserstein distance `W₂`, which is a parameter:
Mathlib has no Wasserstein distance, and nothing below constrains `W₂` to be
one, so the statement is only as strong as the `W₂` it is applied to.

Source: arXiv:2512.01868v4, §2 (mean-field clustering). -/
def MeanFieldClustering
    (W₂ : Measure (EucSpace cfg.head_dim) → Measure (EucSpace cfg.head_dim) → ℝ)
    (T : ℕ) : Prop :=
  3 ≤ cfg.head_dim → 0 < T →
  ∀ (alpha eps : ℝ) (positions : Fin T → ℝ), 0 < eps →
    ∀ᵐ x₀ : Fin T → EucSpace cfg.head_dim, ∀ x : ℕ → Fin T → EucSpace cfg.head_dim,
      x 0 = x₀ → PreLNHead cfg alpha eps positions x → (∀ (L : ℕ) (i : Fin T), x L i ≠ 0) →
        ∃ xinf : EucSpace cfg.head_dim, ‖xinf‖ = 1 ∧
          Filter.Tendsto
            (fun L : ℕ =>
              W₂ (((T : ℝ)⁻¹).toNNReal •
                  ∑ i : Fin T, Measure.dirac (Bridge.toSphere cfg.head_dim (x L i)))
                (Measure.dirac xinf))
            Filter.atTop (nhds (0 : ℝ))

/-- **The depth rate under Pre-LN is polynomial.**

With the temperatures bounded, `|α| ≤ A`, the directions approach their common
limit at the rate

  `‖Φ(x_L(i)) - x_∞‖ ≤ C / L³`,

with `C` depending on `A` alone — the quantifier order is what says so, since
`C` is chosen before `α`, `eps` and the initial stream.  This is slower than
the exponential rate of the Post-LN scheme of arXiv:2312.10794v5, and it is
the Pre-LN scheme that `gpt-mini` uses.

Source: arXiv:2510.22026v2, `thm: preln-slow`. -/
def PolynomialRate (A : ℝ) (T : ℕ) : Prop :=
  3 ≤ cfg.head_dim → 0 ≤ A →
  ∃ C : ℝ, 0 < C ∧
    ∀ (alpha eps : ℝ) (positions : Fin T → ℝ), |alpha| ≤ A → 0 < eps →
      ∀ᵐ x₀ : Fin T → EucSpace cfg.head_dim, ∀ x : ℕ → Fin T → EucSpace cfg.head_dim,
        x 0 = x₀ → PreLNHead cfg alpha eps positions x →
        (∀ (L : ℕ) (i : Fin T), x L i ≠ 0) →
          ∃ xinf : EucSpace cfg.head_dim, ‖xinf‖ = 1 ∧
            ∀ (L : ℕ), 1 ≤ L → ∀ i : Fin T,
              ‖Bridge.toSphere cfg.head_dim (x L i) - xinf‖ ≤ C / (L : ℝ) ^ 3

end GPTMini
end Transformer
