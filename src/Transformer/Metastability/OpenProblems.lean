/-
# Metastability — the open problems of 2410.06833v1

Two remaining questions the survey poses and leaves open. The energy-level
question of §4 is proved for a nonempty window near consensus at sufficiently
large `β` in `Section4_EnergyWindow`, imported here.

* `Problem conj: saddle-to-saddle` — does the energy of `SA` follow a
  staircase profile along some time reparametrization?
* the `problem` of the reparametrization candidate — does it hold for the
  specific reparametrization `τ̇_β = log β / ‖∇𝖤_β‖`?

Each is a `theorem … := by sorry`: open is a kind of unproved, and the sorry
count is where unproved is recorded.  Nothing in this development may be built
on them, which is what the `sorry` says.

What stays a `Prop`-valued definition is what is a genuine predicate of its
arguments — `IsMetastable` (`Metastability.IsMetastable`), `HasStaircaseProfile`,
`IsGradientReparam`, `IsEnergyGradNorm` — and those are the vocabulary the
statements are written in, not statements themselves.
-/

import Transformer.Metastability.Section4_EnergyWindow
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- The **staircase profile** of `conj: saddle-to-saddle`.

`X β` is the solution of `SA` at inverse temperature `β`, and `τ β` the
reparametrization at that temperature.  The profile asks for jump times
`0 = T_0 < T_1 < ⋯ < T_k < T_{k+1} = +∞` and a piecewise-constant
`φ_∞ ∈ L^∞(ℝ_{≥0}; [0,1])`, constant on each `[T_i, T_{i+1})`, such that

  `φ_β(t) := 2β 𝖤_β(x_1(τ_β(t)),…,x_n(τ_β(t)))`

converges to `φ_∞` uniformly on each `(T_i, T_{i+1})`, `i ∈ {0,…,k}`, as
`β → ∞`.  The uniform convergence is written out with an `ε`/`B` pair so
that the threshold `B` is common to all plateaux; `T_{k+1} = +∞` is encoded
by dropping the upper bound when `i = k`.

**What the source says and what is changed here.**

* The source defines `φ_β` with `𝖤_β` itself, whose printed normalization
  `1/(2β e^β n²)` makes it at most `1/(2β)`: then every family converges to
  `φ_∞ ≡ 0` and the problem is empty (`printed_staircase_trivial`).  The
  factor `2β` gives the energy the maximum `1` that the section's figure
  assigns to the last step.
* The source asks for uniform convergence on `(T_i, T_{i+1})` for
  `i ∈ {0,…,k-1}` only, leaving out the last plateau `(T_k, +∞)`.  Then a
  time change slow enough — `τ_β(t) = t e^{-β²}` — keeps every trajectory at
  its initial energy on the bounded window `[0, T_k]`, and the question is
  again empty.  `thm: staircase`, the one case the source proves, quantifies
  over the last plateau (`i ∈ {1,…,k}` there), and so does this profile.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`. -/
def HasStaircaseProfile
    (X : ℝ → ℝ → SphereTuple d n) (τ : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
    1 ≤ k ∧ k ≤ n ∧ T 0 = 0 ∧
    (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
    (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
    (∀ i : ℕ, i ≤ k → ∀ s t : ℝ, T i ≤ s → T i ≤ t →
      (i < k → s < T (i + 1)) → (i < k → t < T (i + 1)) → φ s = φ t) ∧
    ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      ∀ i : ℕ, i ≤ k → ∀ t : ℝ, T i < t → (i < k → t < T (i + 1)) →
        |2 * β * Eβ d n β (X β (τ β t)) - φ t| < ε

/-- **Problem (conj: saddle-to-saddle).** *Staircase profile of the energy.*

Fix `d, n ≥ 2` and an initial configuration, and let `X β` be the solution of
`SA` at inverse temperature `β`.  Does there exist a family of continuous
reparametrizations `(τ_β)_{β}` of `ℝ_{≥0}` along which the energy has the
staircase profile above?

The survey answers this affirmatively only for the modified `USA` dynamics on
the circle (`Metastability.staircase_profile`); in the
generality below it is open.

**What the source says and what is changed here.**  The source asks only
for `τ_β ∈ 𝒞⁰(ℝ_{≥0}; ℝ_{≥0})`.  Read literally, `τ_β ≡ 0` is admitted, the
energy along it is the constant `2β 𝖤_β(X₀)`, which converges as `β → ∞`,
and the question is empty.  A reparametrization of time is meant: here
`τ_β(0) = 0`, `τ_β` is strictly increasing on `ℝ_{≥0}` and unbounded, so
that the whole trajectory is traversed.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`. -/
theorem saddle_to_saddle (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ (X₀ : SphereTuple d n) (X : ℝ → ℝ → SphereTuple d n),
      (∀ β : ℝ, 1 < β → X β 0 = X₀ ∧ Perspective.SA d n β (X β)) →
      ∃ τ : ℝ → ℝ → ℝ,
        (∀ β : ℝ, 1 < β → Continuous (τ β)) ∧
        (∀ β : ℝ, 1 < β → τ β 0 = 0 ∧ StrictMonoOn (τ β) (Set.Ici 0) ∧
          Filter.Tendsto (τ β) Filter.atTop Filter.atTop) ∧
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
