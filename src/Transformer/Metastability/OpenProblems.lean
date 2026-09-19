/-
# Metastability — the open problems of 2410.06833v1

Three questions the survey poses and leaves open:

* the `problem` of `sec: energy.levels` — is an assumption on the *energy
  level* of the initial configuration, relative to the energy of a uniform
  sample, enough for metastability?
* `Problem conj: saddle-to-saddle` — does the energy of `SA` follow a
  staircase profile along some time reparametrization?
* the `problem` of the reparametrization candidate — does it hold for the
  specific reparametrization `τ̇_β = log β / ‖∇𝖤_β‖`?

Each is a `theorem … := by sorry`: open is a kind of unproved, and the sorry
count is where unproved is recorded.  Nothing in this development may be built
on them, which is what the `sorry` says.

What stays a `Prop`-valued definition is what is a genuine predicate of its
arguments — `IsMetastable`, `HasStaircaseProfile`, `IsGradientReparam`,
`IsEnergyGradNorm` — and those are the vocabulary the three statements are
written in, not statements themselves.
-/

import Transformer.Basic
import Transformer.Metastability.Basic
import Transformer.Metastability.MainTheorem
import Transformer.Metastability.InitialUniform
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- The conclusion of `thm: metastability`, detached from its hypotheses.

`X₀` is *metastable at inverse temperature `β`* if there are caps
`𝒮_1(ε),…,𝒮_k(ε)` and times `0 < T₁ < T₂` such that every solution of `SA`
started at `X₀`

1. stays in the doubled caps up to `T₂`, and
2. has exponentially small within-cap spread on `[T₁, T₂]`.

This is exactly the conclusion of `Metastability.metastability`, with the
cap data `k, w, ε` and the rate `λ` existentially quantified: under an
assumption phrased on the energy alone no cap structure is handed to us,
so the caps have to be part of what is asserted.

Source: arXiv:2410.06833v1, §2, `thm: metastability`. -/
def IsMetastable (β : ℝ) (X₀ : SphereTuple d n) : Prop :=
  ∃ (ε : ℝ) (k : ℕ) (w : Idx k → SSphere d) (lam T₁ T₂ : ℝ),
    0 < ε ∧ ε < 1 / 16 ∧ k ≤ n ∧ 0 < lam ∧ 0 < T₁ ∧ T₁ < T₂ ∧
    ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
      (∀ i : Idx n, ∀ q : Idx k,
        X₀ i ∈ sphericalCap d (w q) ε →
        ∀ t : ℝ, 0 ≤ t → t ≤ T₂ → X t i ∈ sphericalCap d (w q) (2 * ε)) ∧
      ∀ q : Idx k, ∀ t : ℝ, T₁ ≤ t → t ≤ T₂ → ∀ i j : Idx n,
        X t i ∈ sphericalCap d (w q) (2 * ε) →
        X t j ∈ sphericalCap d (w q) (2 * ε) →
          ‖((X t i : EucSpace d)) - ((X t j : EucSpace d))‖ ^ 2
            ≤ 2 * Real.exp (-(lam * β))

/-- **Problem (sec: energy.levels).** *Metastability from an energy level.*

Fix `d, n ≥ 2` and `β > 0`, and let `U_1,…,U_n` be i.i.d. uniform on
`𝕊^{d-1}`.  Can one find `1 > c₂ > c₁ > 0`, depending on `β`, such that
every `(x_1,…,x_n) ∈ (𝕊^{d-1})^n` with

  `c₂ ≥ 𝖤_β(x_1,…,x_n) - 𝔼[𝖤_β(U_1,…,U_n)] ≥ c₁`

is metastable in the sense of `thm: metastability`?

The uniform measure is not constructed in this development: it is pinned down
by `Metastability.IsUniformOn` — a rotation-invariant probability measure, of
which there is exactly one — and `𝔼[𝖤_β(U)]` is the integral of `𝖤_β` against
the `n`-fold product `iidSphere d n ν`, exactly as in `InitialUniform`.

Not proved here; the survey leaves it open.

Source: arXiv:2410.06833v1, §4, `sec: energy.levels`. -/
theorem energy_level_metastability (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ ν : Measure (SSphere d), IsUniformOn d ν → ∀ β : ℝ, 0 < β →
      ∃ c₁ c₂ : ℝ, 0 < c₁ ∧ c₁ < c₂ ∧ c₂ < 1 ∧
        ∀ X₀ : SphereTuple d n,
          c₁ ≤ Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) →
          Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) ≤ c₂ →
            IsMetastable d n β X₀ := by
  sorry

/-- The hypotheses of `energy_level_metastability` are satisfiable: `d = n = 2`.
The uniformity of `ν` stays inside the statement — no rotation-invariant
measure on `𝕊^{d-1}` is constructed here, so there is none to exhibit. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

/-- The **staircase profile** of `conj: saddle-to-saddle`.

`X β` is the solution of `SA` at inverse temperature `β`, and `τ β` the
reparametrization at that temperature.  The profile asks for jump times
`0 = T_0 < T_1 < ⋯ < T_k` (with `T_{k+1} = +∞`, which is why only the
indices `i < k` carry a strict inequality) and a piecewise-constant
`φ_∞ ∈ L^∞(ℝ_{≥0}; [0,1])` such that

  `φ_β(t) := 𝖤_β(x_1(τ_β(t)),…,x_n(τ_β(t)))`

converges to `φ_∞` uniformly on each `(T_i, T_{i+1})`, `i < k`, as
`β → ∞`.  The uniform convergence is written out with an `ε`/`B` pair so
that the threshold `B` is common to all plateaux.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`. -/
def HasStaircaseProfile
    (X : ℝ → ℝ → SphereTuple d n) (τ : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
    1 ≤ k ∧ k ≤ n ∧ T 0 = 0 ∧
    (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
    (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
    (∀ i : ℕ, i < k → ∀ s t : ℝ, s ∈ Set.Ico (T i) (T (i + 1)) →
      t ∈ Set.Ico (T i) (T (i + 1)) → φ s = φ t) ∧
    ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      ∀ i : ℕ, i < k → ∀ t : ℝ, t ∈ Set.Ioo (T i) (T (i + 1)) →
        |Eβ d n β (X β (τ β t)) - φ t| < ε

/-- **Problem (conj: saddle-to-saddle).** *Staircase profile of the energy.*

Fix `d, n ≥ 2` and an initial configuration, and let `X β` be the solution of
`SA` at inverse temperature `β`.  Does there exist a family of continuous
reparametrizations `(τ_β)_{β}` of `ℝ_{≥0}` along which the energy has the
staircase profile above?

The survey answers this affirmatively only for the modified `USA` dynamics on
the circle (`Metastability.staircase_profile`); in the
generality below it is open.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`. -/
theorem saddle_to_saddle (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ (X₀ : SphereTuple d n) (X : ℝ → ℝ → SphereTuple d n),
      (∀ β : ℝ, 1 < β → X β 0 = X₀ ∧ Perspective.SA d n β (X β)) →
      ∃ τ : ℝ → ℝ → ℝ,
        (∀ β : ℝ, 1 < β → Continuous (τ β)) ∧
        (∀ β t : ℝ, 0 ≤ t → 0 ≤ τ β t) ∧
        HasStaircaseProfile d n X τ := by
  sorry

/-- The hypotheses of `saddle_to_saddle` are satisfiable: `d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

/-- **The reparametrization candidate.**

  `τ̇_β(t) = log β / ‖∇𝖤_β(τ_β(t))‖`,   `τ_β(0) = 0`,

so that the dynamics is accelerated wherever the gradient is small.  The
Riemannian gradient of `𝖤_β` on `(𝕊^{d-1})^n` is not constructed here: its
norm is carried as a parameter `gradNorm β`, as in `OttoReznikoff`.

Source: arXiv:2410.06833v1, §6, "A reparametrization candidate". -/
def IsGradientReparam
    (gradNorm : ℝ → SphereTuple d n → ℝ) (X : ℝ → ℝ → SphereTuple d n)
    (τ : ℝ → ℝ → ℝ) : Prop :=
  ∀ β : ℝ, 1 < β →
    τ β 0 = 0 ∧
    ∀ t : ℝ, HasDerivAt (τ β) (Real.log β / gradNorm β (X β (τ β t))) t

/-- `gradNorm` is the Riemannian gradient norm of `𝖤_β` along `X β`.

The gradient itself is not constructed here, so it is pinned down by the one
identity that characterizes its norm along a gradient *ascent* — which `SA`
is, `𝖤_β` increasing along it:

  `d/dt 𝖤_β(X_β(t)) = ‖∇𝖤_β(X_β(t))‖²`,   `‖∇𝖤_β‖ ≥ 0`.

Without this, `saddle_to_saddle_gradient_reparam` read over an arbitrary
`gradNorm` would be a claim about an arbitrary time change, and false: the
reparametrization candidate is the one built from the *actual* gradient. -/
def IsEnergyGradNorm (gradNorm : ℝ → SphereTuple d n → ℝ)
    (X : ℝ → ℝ → SphereTuple d n) : Prop :=
  ∀ β : ℝ, 1 < β → ∀ t : ℝ,
    0 ≤ gradNorm β (X β t) ∧
    HasDerivAt (fun s => Eβ d n β (X β s)) ((gradNorm β (X β t)) ^ 2) t

/-- **Problem (the reparametrization candidate).**

Does `conj: saddle-to-saddle` hold for the explicit reparametrization
`IsGradientReparam`, rather than for some reparametrization produced by the
proof?  Writing `φ_β(t) := 𝖤_β(u(τ_β(t)))`, the candidate is the one for
which `φ̇_β(t) = log β · ‖∇𝖤_β(u(τ_β(t)))‖`, the hope being that a jump in
the energy then takes a time independent of `β`.

Not proved here; the survey leaves it open.

Source: arXiv:2410.06833v1, §6, "A reparametrization candidate". -/
theorem saddle_to_saddle_gradient_reparam (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ (gradNorm : ℝ → SphereTuple d n → ℝ)
      (X₀ : SphereTuple d n) (X : ℝ → ℝ → SphereTuple d n),
      (∀ β : ℝ, 1 < β → X β 0 = X₀ ∧ Perspective.SA d n β (X β)) →
      IsEnergyGradNorm d n gradNorm X →
      ∀ τ : ℝ → ℝ → ℝ, IsGradientReparam d n gradNorm X τ →
        HasStaircaseProfile d n X τ := by
  sorry

/-- The hypotheses of `saddle_to_saddle_gradient_reparam` are satisfiable:
`d = n = 2`.  That `gradNorm` is the gradient norm of `𝖤_β` stays inside the
statement — the gradient is not constructed here, so there is none to
exhibit. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

end Metastability
end Transformer
