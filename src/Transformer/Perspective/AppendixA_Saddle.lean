/-
# Appendix A — Proof of Theorem (p:beta0), part 2

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes the second half of Appendix A of the survey:

* `e:helpcl`            — the Hessian of `𝖤_0` at a critical point,
* `e:russiantrick`      — the "Russian trick",
* `Lemma lem: yury.lemma` — every non-trivial critical point of `𝖤_0` is a
                            strict saddle,
* `Lemma l:nosaddleconv`  — gradient ascent avoids strict saddles,
* the assembly of `Theorem p:beta0`.

The energy itself, its gradient flow and `eq: taylor` are in
`Perspective.AppendixA_Beta0`.  Everything below is a `Prop`-valued
definition: the statements are written out in full, none of them is proved
here.
-/

import Transformer.Perspective.AppendixA_Beta0

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- A skew-symmetric endomorphism of `ℝ^d`: `⟨Bx, y⟩ = -⟨x, By⟩`.

These are exactly the generators of the rotations, `e^{tB} ∈ O(d)`, which is
why the perturbation below stays on the sphere. -/
def IsSkew (B : ParamMatrix d) : Prop :=
  ∀ x y : EucSpace d, inner (𝕜 := ℝ) (B x) y = -inner (𝕜 := ℝ) x (B y)

/-- The perturbation of `X` used in `e:helpcl`:

  `x_i(t) = e^{tB} x_i` for `i ∈ 𝒮`,   `x_i(t) = x_i` otherwise.

The rotated particles are described by the differential equation
`ẋ_i(t) = B x_i(t)` they solve rather than by the matrix exponential, which is
the same thing for the initial condition `Y 0 = X` and keeps the statement
inside the `HasDerivAt` API used everywhere else here. -/
def PerturbationBy
    (B : ParamMatrix d) (𝒮 : Finset (Idx n))
    (X : SphereTuple d n) (Y : ℝ → SphereTuple d n) : Prop :=
  Y 0 = X ∧
  (∀ i ∈ 𝒮, ∀ t : ℝ,
    HasDerivAt (fun s => (Y s i : EucSpace d)) (B ((Y t i : EucSpace d))) t) ∧
  (∀ i ∉ 𝒮, ∀ t : ℝ, (Y t i : EucSpace d) = (X i : EucSpace d))

/-- `c` is the second derivative of `t ↦ 𝖤_0(Y(t))` at `t = 0`: the energy is
differentiable along the whole curve, and its derivative is again
differentiable at `0`, with derivative `c`. -/
def SecondDerivE0At (Y : ℝ → SphereTuple d n) (c : ℝ) : Prop :=
  ∃ f' : ℝ → ℝ,
    (∀ t : ℝ, HasDerivAt (fun s => E0 d n (Y s)) (f' t) t) ∧ HasDerivAt f' c 0

/-- **Equation (e:helpcl).** *Hessian of `𝖤_0` at a critical point.*

For a skew-symmetric `B`, a subset `𝒮 ⊂ [n]`, and the perturbation
`x_i(t) = e^{tB} x_i` (`i ∈ 𝒮`), `x_i(t) = x_i` (`i ∉ 𝒮`) of a critical point,

  `𝖤_0''(0) = (2/n) Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨B² x_i, x_j⟩`.

The first-order term is absent precisely because `X` is critical, which is why
criticality is a hypothesis rather than decoration.

A `Prop`-valued definition and not a theorem: the second-order expansion is
not carried out here.

Source: arXiv:2312.10794v5, Appendix A, `e:helpcl`. -/
def HessianAtCritical : Prop :=
  ∀ (X : SphereTuple d n) (B : ParamMatrix d) (𝒮 : Finset (Idx n)),
    IsSkew d B → IsCriticalE0 d n X →
    ∀ Y : ℝ → SphereTuple d n, PerturbationBy d n B 𝒮 X Y →
      SecondDerivE0At d n Y
        ((2 * (n : ℝ)⁻¹) * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
          inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d)))

/-- **Equation (e:russiantrick).** *"Russian trick".*  In odd dimension `d`
there are `d` skew-symmetric matrices `B_1, …, B_d` with

  `-I_d = (1/(d-1)) Σ_k B_k²`.

Each `B_k` is obtained from the canonical `(d-1)`-dimensional symplectic form
by zeroing out its `k`-th `2×2` rotation block; the identity is what turns the
Hessian formula `e:helpcl` into a *sum* of directions along which `𝖤_0` can be
increased.

A `Prop`-valued definition and not a theorem: the construction of the blocks
is not formalized here.

Source: arXiv:2312.10794v5, Appendix A, `e:russiantrick`. -/
def RussianTrick : Prop :=
  Odd d →
    ∃ B : Idx d → ParamMatrix d,
      (∀ k : Idx d, IsSkew d (B k)) ∧
      ∀ x : EucSpace d, (((d : ℝ) - 1)⁻¹) • ∑ k : Idx d, B k (B k x) = -x

/-- **Lemma (lem: yury.lemma).** *Every non-trivial critical point of `𝖤_0` is
a strict saddle.*

Strictness is spelled out as: there is a rotation direction `B` and a subset
`𝒮` whose perturbation has *positive* second derivative of the energy at
`t = 0`.  Since `𝖤_0` is being maximized along `e:gradfl`, such a direction
means the critical point is not a local maximum — and hence, by
`no_saddle_convergence`, is reached from a null set of initial data.  In
particular every local maximum of `𝖤_0` is a consensus configuration, hence a
global maximum.

A `Prop`-valued definition and not a theorem: the proof combines `eq: taylor`,
`e:helpcl` and `e:russiantrick`, none of which is proved here.

Source: arXiv:2312.10794v5, Appendix A, `lem: yury.lemma`. -/
def YuryLemma : Prop :=
  ∀ X : SphereTuple d n, IsCriticalE0 d n X → NonTrivialTuple d n X →
    ∃ (B : ParamMatrix d) (𝒮 : Finset (Idx n)) (Y : ℝ → SphereTuple d n) (c : ℝ),
      IsSkew d B ∧ PerturbationBy d n B 𝒮 X Y ∧ SecondDerivE0At d n Y c ∧ 0 < c

/-- **Lemma (l:nosaddleconv).** *No-saddle-convergence lemma.*

On a compact Riemannian manifold the set of initial conditions whose
gradient-ascent trajectory converges to a strict saddle of a smooth `f` has
zero volume (center-stable manifold theorem).  It is stated here in the only
instance the survey uses it: `ℳ = (𝕊^{d-1})^n`, `f = 𝖤_0`, and — by
`YuryLemma` — the strict saddles are the non-trivial critical points.  Volume
is the uniform measure `UniformTuple`.

A `Prop`-valued definition and not a theorem: the center-stable manifold
theorem is not available here.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`. -/
def NoSaddleConvergence : Prop :=
  ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
    P { X₀ : SphereTuple d n |
        ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ E0GradientAscent d n X ∧
          ∃ Z : SphereTuple d n, IsCriticalE0 d n Z ∧ NonTrivialTuple d n Z ∧
            ∀ i : Idx n,
              Filter.Tendsto
                (fun t : ℝ => ((X t i : EucSpace d) - (Z i : EucSpace d)))
                Filter.atTop (nhds 0) } = 0

/-- **Theorem (p:beta0)**, in the almost-sure form Appendix A proves: for
`d, n ≥ 2` the set of initial sequences whose `β = 0` trajectory does *not*
converge to a single point is null for the uniform law on `(𝕊^{d-1})^n`.

This is the statement `Perspective.beta0_consensus` should have; the latter
quantifies over every initial sequence, which is false at the exceptional
null set.

Source: arXiv:2312.10794v5, §4, `p:beta0`. -/
def AlmostSureConsensusBeta0 : Prop :=
  2 ≤ d → 2 ≤ n →
  ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
    P { X₀ : SphereTuple d n | ¬ ∃ x_star : SSphere d,
          ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → beta0Dynamics d n X →
            ∀ i : Idx n,
              Filter.Tendsto
                (fun t : ℝ => ((X t i : EucSpace d) - (x_star : EucSpace d)))
                Filter.atTop (nhds 0) } = 0

/-- *Assembly of the proof of `p:beta0`.*  The three ingredients are:

  (i) Łojasiewicz — `𝖤_0` is analytic on a compact analytic manifold, so every
      trajectory of `e:gradfl` converges to a critical point (the hypothesis
      spelled out first below);
 (ii) `l:nosaddleconv` — the non-trivial critical points are reached from a
      null set;
(iii) `lem: yury.lemma` — those are exactly the strict saddles.

Together they give `AlmostSureConsensusBeta0`.

A `Prop`-valued definition and not a theorem: the implication itself, like its
three hypotheses, is not proved here.

Source: arXiv:2312.10794v5, Appendix A. -/
def PBeta0Proof : Prop :=
  (∀ X : ℝ → SphereTuple d n, E0GradientAscent d n X →
      ∃ Z : SphereTuple d n, IsCriticalE0 d n Z ∧
        ∀ i : Idx n,
          Filter.Tendsto
            (fun t : ℝ => ((X t i : EucSpace d) - (Z i : EucSpace d)))
            Filter.atTop (nhds 0)) →
    NoSaddleConvergence d n → YuryLemma d n → AlmostSureConsensusBeta0 d n

end Perspective
end Transformer
