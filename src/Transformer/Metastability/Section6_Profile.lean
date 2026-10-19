module

public import Transformer.Metastability.Basic
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

/-!
# The staircase profile and its proposed gradient clock

ArXiv:2410.06833v1, §6, `conj: saddle-to-saddle` and
"A reparametrization candidate". These are predicates of actual trajectories,
time changes, and a supplied gradient norm. No convergence or existence
claim is placed in their definitions.

The full-trajectory profile retains the two previously documented changes:
normalization by `2β`, and inclusion of the final unbounded plateau.
The continuity obstruction and its actual attention counterexample are
proved in the following modules. The printed problem's different
normalization and omission of its last plateau are kept explicit.

The gradient norm is supplied as a function of temperature and configuration.
Its nonnegativity and the energy-derivative identity constrain that function
along every trajectory at every time, for each temperature greater than one.
The proposed clock has its initial value and its differential equation
recorded separately from the profile it is supposed to produce.

The examples below realize these inputs simultaneously at equilibrium.
They include two nonzero sphere tokens and their actual attention equation,
as well as the energy identity and the clock equation. This also records
the behavior of the totalized clock formula when its denominator vanishes.
-/

@[expose] public section

open Real

namespace Transformer.Metastability

variable (d n : ℕ)

/-- The **staircase profile** of `conj: saddle-to-saddle`.

`X β` is the solution of `SA` at inverse temperature `β`, and `τ β` the
reparametrization at that temperature. The profile asks for jump times
`0 = T_0 < T_1 < ⋯ < T_k < T_{k+1} = +∞` and a piecewise-constant
`φ_∞ ∈ L^∞(ℝ_{≥0}; [0,1])`, constant on each `[T_i, T_{i+1})`, such that

  `φ_β(t) := 2β 𝖤_β(x_1(τ_β(t)),…,x_n(τ_β(t)))`

converges to `φ_∞` uniformly on each `(T_i, T_{i+1})`, `i ∈ {0,…,k}`, as
`β → ∞`. The uniform convergence is written out with an `ε`/`B` pair so
that the threshold `B` is common to all plateaux; `T_{k+1} = +∞` is encoded
by dropping the upper bound when `i = k`.

**What the source says and what is changed here.**

* The source defines `φ_β` with `𝖤_β` itself, whose printed normalization
  `1/(2β e^β n²)` makes it at most `1/(2β)`: then every family converges to
  `φ_∞ ≡ 0` and the problem is empty (`printed_staircase_trivial`). The
  factor `2β` gives the energy the maximum `1` that the section's figure
  assigns to the last step.
* The source asks for uniform convergence on `(T_i, T_{i+1})` for
  `i ∈ {0,…,k-1}` only, leaving out the last plateau `(T_k, +∞)`. Then a
  time change slow enough — `τ_β(t) = t e^{-β²}` — keeps every trajectory at
  its initial energy on the bounded window `[0, T_k]`, and the question is
  again empty. `thm: staircase`, the one case the source proves, quantifies
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

/-- **The reparametrization candidate.**

  `τ̇_β(t) = log β / ‖∇𝖤_β(τ_β(t))‖`,   `τ_β(0) = 0`,

so that the dynamics is accelerated wherever the gradient is small. The
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

/-- Both clock predicates are satisfiable on a stationary consensus
curve: the actual energy is constant, its gradient norm is zero, and the
clock is constant. Source: arXiv:2410.06833v1, §6, the proposed gradient
clock; division by zero is Lean's totalized convention at equilibrium. -/
example (x : SSphere d) :
    IsEnergyGradNorm d n (fun _ _ => 0) (fun _ _ _ => x) ∧
    IsGradientReparam d n (fun _ _ => 0) (fun _ _ _ => x) (fun _ _ => 0) := by
  constructor
  · intro β _ t
    refine ⟨le_rfl, ?_⟩
    simpa using hasDerivAt_const t (Eβ d n β (fun _ => x))
  · intro β _
    refine ⟨rfl, fun t => ?_⟩
    change HasDerivAt (fun _ : ℝ => (0 : ℝ)) (Real.log β / 0) t
    rw [div_zero]
    exact hasDerivAt_const t 0

/-- The complete dynamical inputs of the remaining gradient-clock question
are simultaneously realized on the circle with two consensus tokens.
The curve solves the original `SA`, starts from the same tuple at every
temperature, and supplies both its zero energy-gradient norm and its clock.
Source: arXiv:2410.06833v1, §6, "A reparametrization candidate".
The clock at equilibrium uses the totalized division convention above. -/
example : ∃ (gradNorm : ℝ → SphereTuple 2 2 → ℝ)
    (X₀ : SphereTuple 2 2) (X : ℝ → ℝ → SphereTuple 2 2) (τ : ℝ → ℝ → ℝ),
    (2 : ℕ) ≤ 2 ∧ (2 : ℕ) ≤ 2 ∧
    (∀ β : ℝ, 1 < β → X β 0 = X₀ ∧ Perspective.SA 2 2 β (X β)) ∧
    IsEnergyGradNorm 2 2 gradNorm X ∧ IsGradientReparam 2 2 gradNorm X τ := by
  refine ⟨fun _ _ => 0, fun _ => basePoint 1, fun _ _ _ => basePoint 1,
    fun _ _ => 0, le_rfl, le_rfl, ?_, ?_, ?_⟩
  · intro β _
    exact ⟨rfl, Perspective.SA_const_consensus 2 2 (by norm_num) β (basePoint 1)⟩
  · intro β _ t
    refine ⟨le_rfl, ?_⟩
    simpa using hasDerivAt_const t (Eβ 2 2 β (fun _ => basePoint 1))
  · intro β _
    refine ⟨rfl, fun t => ?_⟩
    change HasDerivAt (fun _ : ℝ => (0 : ℝ)) (Real.log β / 0) t
    rw [div_zero]
    exact hasDerivAt_const t 0

/-- **Problem (the reparametrization candidate).**

Does the staircase conclusion hold for the explicit reparametrization
`IsGradientReparam`, rather than for some reparametrization produced by the
proof? Writing `φ_β(t) := 𝖤_β(u(τ_β(t)))`, the candidate is the one for
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
`d = n = 2`. The complete stationary consensus witness, including
its actual gradient norm and its clock, is in `Section6_Profile`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩


end Transformer.Metastability
