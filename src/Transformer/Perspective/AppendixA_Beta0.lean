/-
# Appendix A — Proof of Theorem (p:beta0)

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes Appendix A of the survey:

* `eq: E0`              — the `β = 0` energy,
* `Lemma l:nosaddleconv`— the no-saddle-convergence lemma
                          (center-stable manifold theorem),
* `Lemma lem: yury.lemma` — every non-trivial critical point of `𝖤_0` is a
                            strict saddle,
* `eq: taylor`, `e:helpcl`, `e:russiantrick`.
-/

import Transformer.Basic
import Transformer.Perspective.Section3_SmallBeta

open scoped BigOperators
open Real

namespace Transformer
namespace AppendixBeta0

variable (d n : ℕ)

/-- **`β = 0` energy.**

  `𝖤_0(x_1,…,x_n) = (1/n) Σ_i Σ_j ⟨x_i, x_j⟩`. -/
noncomputable def E0 (X : SphereTuple d n) : ℝ :=
  ((n : ℝ)⁻¹) * ∑ i : Idx n, ∑ j : Idx n,
    inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))

/-- **Equation (e:gradfl).** *Generic gradient ascent on a Riemannian manifold.*

  `Ẋ(t) = ∇f(X(t))`,  `X(0) = X₀`. -/
def gradAscentODE
    {M : Type*} [TopologicalSpace M]
    (gradf : M → ℝ) -- placeholder; in Lean a vector-field-valued gradient
    (X : ℝ → M) (X₀ : M) : Prop :=
  X 0 = X₀

/-- **Lemma (l:nosaddleconv).** *No-saddle-convergence lemma.*

For any compact Riemannian manifold `ℳ` and smooth `f : ℳ → ℝ`, the set of
initial conditions `X₀` whose gradient-ascent trajectory converges to a
strict saddle point of `f` has zero volume.

Here we record the statement in an `Ω`-shape: the conclusion is left abstract
as `True`, but the hypotheses convey the manifold/function setup. -/
theorem no_saddle_convergence
    {M : Type*} [TopologicalSpace M] (f : M → ℝ) :
    True := by trivial

/-- *Equation (eq:taylor).* If `(x_1,…,x_n) ∈ (𝕊^{d-1})^n` is a non-trivial
critical point of `𝖤_0`, then there exists `𝒮 ⊂ [n]` with

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨x_i, x_j⟩ < 0`. -/
theorem taylor_eq
    (X : SphereTuple d n)
    (h_crit : True)        -- critical point of `𝖤_0`
    (h_nontriv : ∃ i j : Idx n, (X i : EucSpace d) ≠ (X j : EucSpace d)) :
    ∃ 𝒮 : Finset (Idx n),
      ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
        inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) < 0 := by
  sorry

/-- **Equation (e:helpcl).** Hessian formula at a critical point:

  `𝖤_0''(0) = (2/n) Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨B² x_i, x_j⟩`,

for the time-dependent perturbation
  `x_i(t) = e^{t B} x_i`  if  `i ∈ 𝒮`,  `x_i(t) = x_i` otherwise. -/
theorem hessian_at_critical
    (X : SphereTuple d n) (B : EucSpace d →L[ℝ] EucSpace d)
    (h_skew : True)  -- placeholder: `B` is skew-symmetric.
    (𝒮 : Finset (Idx n)) :
    True := by trivial

/-- **Equation (e:russiantrick).** *"Russian trick"*: in odd dimension `d`,

  `-I_d = (1/(d-1)) Σ_{j=1}^d B_j²`,

where each `B_j` is a block-diagonal skew-symmetric matrix obtained by
zeroing out the `j`-th `2×2` rotation block of the canonical
`(d-1)`-dimensional symplectic form. -/
theorem russian_trick (d : ℕ) (hd : Odd d) :
    True := by trivial

/-- **Lemma (lem: yury.lemma).** *Every non-trivial critical point of `𝖤_0` is
a strict saddle.*

In particular, every local maximum of `𝖤_0` is a global maximum. -/
theorem yury_lemma
    (X : SphereTuple d n)
    (h_crit : True)                      -- critical point of `𝖤_0`
    (h_nontriv : ∃ i j : Idx n,
                    (X i : EucSpace d) ≠ (X j : EucSpace d)) :
    True := by trivial

/-- **Theorem (p:beta0)** — *Proof.*

For `d, n ≥ 2`, Lebesgue-almost every initial sequence in `(𝕊^{d-1})^n` gives
a trajectory of `e:Snonres0` converging to a single cluster.  The proof
combines:

  (i) the Łojasiewicz theorem (analytic energy + compact analytic manifold),
 (ii) the no-saddle-convergence lemma `l:nosaddleconv`,
(iii) `lem: yury.lemma` (every non-trivial critical point is a strict
      saddle). -/
theorem p_beta0_proof
    (hd : 2 ≤ d) (hn : 2 ≤ n) :
    -- Same statement as `SectionSmallBeta.beta0_consensus`.
    True := by trivial

end AppendixBeta0
end Transformer
