/-
# Metastability — the predicate "`thm: metastability` applies to `X₀`"

`IsMetastable d n β X₀` is the vocabulary in which §4 of arXiv:2410.06833v1 asks its question
about energy levels: *"metastability, as stated in `thm: metastability`, holds"*.  It carries the
hypotheses of the theorem on the cap data — the cover of `Definition hyp: init`, `γ(β) > 0`, the two
bounds of `eq: lambda.3` — together with its conclusion, the times `T₁ < T₂` and the two items.

The conclusion follows from the hypotheses (`Metastability.metastability`), which is
`isMetastable_of_cover`; it is kept in the predicate so that it says what the source says.
-/

import Transformer.Metastability.DirectProof

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **Metastability "as stated in `thm: metastability`" holds for `X₀`.**

`X₀` is *metastable at inverse temperature `β`* if `β > 1` and there are `ε ∈ (0, 1/16)`, centres
`w_1,…,w_k` with `k ≤ n`, and a rate `λ` such that

* every `x_i(0)` lies in some cap `𝒮_q(ε)` (`Definition hyp: init`, item 1);
* `γ(β) > 0` (`eq: gamma`, with `α = α(ε)` the maximal inner product of `αDist`);
* `0 < λ` and `λ` is below the two bounds of `eq: lambda.3` (the sign of the logarithm in the
  first one is corrected, see `metastability`);

and there are times `T₂ > T₁ > 0` with `T₁ ≤ 2 n e^{8εβ} + e n λ β²/(β-1)` and
`T₂ ≥ (ε/n) e^{(1-α)β}` (`MetastabilityTimes`) such that every solution of `(SA)` or of `(USA)`
started at `X₀`

1. keeps `x_i(t) ∈ 𝒮_q(2ε)` for `t ∈ [0, T₂]` whenever `x_i(0) ∈ 𝒮_q(ε)`, and
2. has `‖x_i(t) - x_j(t)‖² ≤ 2 e^{-λβ}` for `t ∈ [T₁, T₂]` whenever `x_i(t), x_j(t) ∈ 𝒮_q(2ε)`.

By `metastability` the last two items are consequences of the hypotheses: the predicate holds
exactly for the `(β, ε)`-separated configurations that admit a rate `λ` (`isMetastable_of_cover`).
A cover is nonempty as soon as there is a token (`n ≥ 1`); for `n = 0` there are no times, since
`T₁ > 0` and `T₁ ≤ 0`.

**What the source says and what is changed here.**  The source asks (§4) whether the energy
window forces "metastability, as stated in `thm: metastability`", and does not define the
phrase further.  The earlier version of this predicate kept only the conclusion, with the cap
data `k, w, ε`, the rate `λ` and the times `T₁ < T₂` existentially quantified and no other
condition on them.  That admitted `k = 0`: the caps `𝒮_q(·)`, `q ∈ Idx 0`, do not exist, both
items are about an empty family, and the predicate held for *every* `d, n, β, X₀` (verified in
Lean, without `sorry`).  The cap family is now nonempty and covers the initial configuration, and
the times and the rate obey the bounds under which the source states the theorem: without
them the theorem has no content, since for `λ ≤ λ_* = β⁻¹ log(1/(8ε))` item 2 is a fact about the
geometry of a cap (`inner_ge_of_mem_cap`).  As in `metastability`, `Ω(1)` for `γ` and `λ` is
not encoded: it is a statement about a family indexed by `β`.

Source: arXiv:2410.06833v1, §2, `thm: metastability`, `Definition hyp: init`, `eq: lambda.3`;
§4, `sec: energy.levels`. -/
def IsMetastable (β : ℝ) (X₀ : SphereTuple d n) : Prop :=
  1 < β ∧ ∃ (ε : ℝ) (k : ℕ) (w : Idx k → SSphere d) (lam : ℝ),
    0 < ε ∧ ε < 1 / 16 ∧ k ≤ n ∧
    (∀ i : Idx n, ∃ q : Idx k, X₀ i ∈ sphericalCap d (w q) ε) ∧
    0 < γβ n β (αDist d k w ε) ε ∧ 0 < lam ∧
    lam < Real.exp ((1 - αDist d k w ε + β⁻¹ * Real.log ((β - 1) * ε /
        (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))) * β) *
      (1 - Real.exp (-(γβ n β (αDist d k w ε) ε * β))) ∧
    lam < 1 - αDist d k w ε - Real.log (2 * (n : ℝ) ^ 2 /
        (1 - Real.exp (-(lamStar β ε * β)))) / β - Real.exp (-(lamStar β ε * β)) ∧
    ∃ T : MetastabilityTimes β ε (αDist d k w ε) n lam,
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ →
        (Perspective.SA d n β X ∨ unnormalizedSA d n β X) →
        (∀ (i : Idx n) (q : Idx k), X₀ i ∈ sphericalCap d (w q) ε →
          ∀ t : ℝ, 0 ≤ t → t ≤ T.T2 → X t i ∈ sphericalCap d (w q) (2 * ε)) ∧
        ∀ (q : Idx k) (t : ℝ), T.T1 ≤ t → t ≤ T.T2 → ∀ i j : Idx n,
          X t i ∈ sphericalCap d (w q) (2 * ε) → X t j ∈ sphericalCap d (w q) (2 * ε) →
            ‖(X t i : EucSpace d) - (X t j : EucSpace d)‖ ^ 2 ≤ 2 * Real.exp (-(lam * β))

variable {d n}

/-- **A `(β, ε)`-separated configuration with an admissible rate is metastable.**  The
hypotheses of `metastability`, together with `k ≤ n`, give `IsMetastable`: the conclusion of the
predicate is that of the theorem.

Source: arXiv:2410.06833v1, §2, `thm: metastability`. -/
theorem isMetastable_of_cover {β ε : ℝ} (hβ : 1 < β) (hε : 0 < ε) (hε' : ε < 1 / 16)
    (hn : 1 ≤ n) (X₀ : SphereTuple d n) {k : ℕ} (hk : k ≤ n) (w : Idx k → SSphere d)
    (hcover : ∀ i : Idx n, ∃ q : Idx k, X₀ i ∈ sphericalCap d (w q) ε)
    (hγ : 0 < γβ n β (αDist d k w ε) ε) {lam : ℝ} (hlam : 0 < lam)
    (hlam₁ : lam < Real.exp ((1 - αDist d k w ε + β⁻¹ * Real.log ((β - 1) * ε /
        (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))) * β) *
      (1 - Real.exp (-(γβ n β (αDist d k w ε) ε * β))))
    (hlam₂ : lam < 1 - αDist d k w ε - Real.log (2 * (n : ℝ) ^ 2 /
        (1 - Real.exp (-(lamStar β ε * β)))) / β - Real.exp (-(lamStar β ε * β))) :
    IsMetastable d n β X₀ :=
  ⟨hβ, ε, k, w, lam, hε, hε', hk, hcover, hγ, hlam, hlam₁, hlam₂,
    metastability hβ hε hε' hn X₀ w hcover hγ hlam hlam₁ hlam₂⟩

end Metastability
end Transformer
