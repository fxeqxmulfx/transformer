/-
# Homogenized Transformers — the limiting logistic SDE

Formalization of `eq: sde.logistic` of arXiv:2604.01978v1, *Homogenized
Transformers*, and of the long-time behavior worked out for it in
`sec: sde.logistic`:

  `du(t) = -u(t)(1 - u(t)²) dt + √2 (1 - u(t)²) dB(t)`.

This is the scalar limit of the overlap `⟨x⁽¹⁾(td), x⁽²⁾(td)⟩` in
`thm:large_beta_meta`.  Mathlib has no stochastic integral, so the equation is
written as its martingale problem, with the generator

  `𝖫φ(u) = -u(1-u²) φ'(u) + (1-u²)² φ''(u)`,

the `(1-u²)²` being `½ (√2(1-u²))²`.  The solution concept carries no initial
condition: `eq: sde.logistic`'s `u(0) = 0` is a hypothesis of the statements
that use it, and leaving it out makes the absorbing states `u ≡ ±1` available
as the witness that the concept is satisfiable.

The source's `sec: sde.logistic` sets `θ = arcsin u`; it writes
"θ(t) = sin u(t)", which cannot be meant — the computation that follows,
`dθ = √2 cos θ dB`, is the Itô formula for `arcsin`, and it is `arcsin` that
makes `θ(t) ∈ [-π/2, π/2]` automatic.  `arcsin` is what is written here.
-/

import Transformer.Homogenized.MvGenerator
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The equation -/

/-- The generator of `eq: sde.logistic`,

  `𝖫φ(u) = -u(1-u²) φ'(u) + (1-u²)² φ''(u)`.

Source: arXiv:2604.01978v1, `eq: sde.logistic`. -/
noncomputable def logisticGenerator (φ : ℝ → ℝ) (u : ℝ) : ℝ :=
  -(u * (1 - u ^ 2)) * deriv φ u + (1 - u ^ 2) ^ 2 * deriv (deriv φ) u

/-- Both coefficients of `eq: sde.logistic` vanish at `u = 1`: the endpoint is
absorbing. -/
@[simp] theorem logisticGenerator_one (φ : ℝ → ℝ) : logisticGenerator φ 1 = 0 := by
  simp [logisticGenerator]

/-- And at `u = -1`. -/
@[simp] theorem logisticGenerator_neg_one (φ : ℝ → ℝ) : logisticGenerator φ (-1) = 0 := by
  simp [logisticGenerator]

/-- The compensated process `φ(u(t)) - φ(u(0)) - ∫₀^t 𝖫φ(u(s)) ds`. -/
noncomputable def logisticMart {Ω : Type*} (u : ℝ → Ω → ℝ) (φ : ℝ → ℝ) (t : ℝ) (ω : Ω) : ℝ :=
  φ (u t ω) - φ (u 0 ω) - ∫ s in (0 : ℝ)..t, logisticGenerator φ (u s ω)

/-- **`u` solves `eq: sde.logistic`**, as a martingale problem: a continuous
adapted process for which `φ(u(t)) - φ(u(0)) - ∫₀^t 𝖫φ(u(s)) ds` is a
martingale on every finite horizon, for every smooth `φ`.

The initial condition `u(0) = 0` of `eq: sde.logistic` is *not* part of this;
it is carried separately by the statements that need it, so that this concept
is about the equation alone.

Source: arXiv:2604.01978v1, `eq: sde.logistic`. -/
structure IsLogisticSolution {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) : Prop where
  /-- The process is adapted. -/
  adapted : ∀ t, 0 ≤ t → StronglyMeasurable[ℱ t] (u t)
  /-- The paths are continuous. -/
  cont : ∀ ω, ContinuousOn (fun t => u t ω) (Set.Ici 0)
  /-- The compensated process is a martingale on every finite horizon. -/
  mart : ∀ φ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ → ∀ T : ℝ, 0 ≤ T →
    IsMartingaleOn T ℱ P (logisticMart u φ)

/-- **`IsLogisticSolution` is satisfiable.**  `u ≡ 1` is a solution: both
coefficients of `eq: sde.logistic` vanish there, so the compensated process is
identically `0`.  This is the absorbing state the long-time analysis of
`sec: sde.logistic` converges to. -/
theorem isLogisticSolution_one {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    [IsProbabilityMeasure P] (ℱ : Filtration ℝ m) :
    IsLogisticSolution P ℱ (fun _ _ => 1) := by
  refine ⟨fun _ _ => stronglyMeasurable_const, fun _ => continuousOn_const, ?_⟩
  intro φ _ T _
  have hzero : logisticMart (fun (_ : ℝ) (_ : Ω) => (1 : ℝ)) φ = fun _ _ => 0 := by
    funext t ω
    simp [logisticMart]
  rw [hzero]
  exact isMartingaleOn_const T ℱ P 0

/-! ### Long-time behavior -/

/-- **`[-1,1]` is invariant.**  A solution of `eq: sde.logistic` started in
`[-1,1]` stays there, which is what the source's `θ(t) ∈ [-π/2, π/2]` records.

Not proved here.

Source: arXiv:2604.01978v1, `sec: sde.logistic`. -/
theorem logistic_range {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Set.Icc (-1 : ℝ) 1) :
    ∀ t, 0 ≤ t → ∀ᵐ ω ∂P, u t ω ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

/-- **`θ = arcsin u` is a bounded martingale** with quadratic variation
`⟨θ⟩_t = ∫₀^t 2 cos²θ(s) ds`:

  `θ(t) = θ(0) + ∫₀^t √2 cos θ(s) dB(s)`.

Written in `u`, using `cos(arcsin u) = √(1-u²)`, the bracket is
`∫₀^t 2(1 - u(s)²) ds`; the martingale property of the bracket-compensated
square is how `⟨θ⟩` is said without a stochastic integral, exactly as in
`McKeanVlasov.lean`.

Not proved here.

Source: arXiv:2604.01978v1, `sec: sde.logistic`. -/
theorem logistic_arcsin_martingale {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hrange : ∀ t, 0 ≤ t → ∀ ω, u t ω ∈ Set.Icc (-1 : ℝ) 1) (T : ℝ) (hT : 0 ≤ T) :
    IsMartingaleOn T ℱ P (fun t ω => Real.arcsin (u t ω)) ∧
      IsMartingaleOn T ℱ P (fun t ω =>
        (Real.arcsin (u t ω) - Real.arcsin (u 0 ω)) ^ 2
          - ∫ s in (0 : ℝ)..t, 2 * (1 - u s ω ^ 2)) := by
  sorry

/-- **The limit of `eq: sde.logistic`.**  Started at a deterministic
`u₀ ∈ [-1,1]`, the solution converges almost surely to one of the two
absorbing states, and

  `P(u_∞ = 1) = arcsin(u₀)/π + 1/2`,

which is the source's `P_{θ₀}(θ_∞ = π/2) = θ(0)/π + 1/2` with `θ(0) = arcsin u₀`.
At `u₀ = 0` — the initial condition of `eq: sde.logistic` — `arcsin 0 = 0` and
this is the source's `P(u_∞ = -1) = P(u_∞ = 1) = 1/2`.

Not proved here.

Source: arXiv:2604.01978v1, `sec: sde.logistic`. -/
theorem logistic_limit {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    (hP : IsProbabilityMeasure P) (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ)
    (hu : IsLogisticSolution P ℱ u) (u₀ : ℝ) (hu₀ : u₀ ∈ Set.Icc (-1 : ℝ) 1)
    (hinit : ∀ ω, u 0 ω = u₀) :
    ∃ uinf : Ω → ℝ,
      (∀ᵐ ω ∂P, Filter.Tendsto (fun t => u t ω) Filter.atTop (nhds (uinf ω))) ∧
        (∀ᵐ ω ∂P, uinf ω = 1 ∨ uinf ω = -1) ∧
          (P {ω | uinf ω = 1}).toReal = Real.arcsin u₀ / π + 1 / 2 ∧
            (P {ω | uinf ω = -1}).toReal = 1 / 2 - Real.arcsin u₀ / π := by
  sorry

/-- The hypotheses of `logistic_range`, `logistic_arcsin_martingale` and
`logistic_limit` are satisfiable at once, on the absorbing solution `u ≡ 1`:
it solves the equation, it starts and stays in `[-1,1]`, and its initial value
is the deterministic `u₀ = 1`.  The conclusion of `logistic_limit` is then the
true statement `P(u_∞ = 1) = arcsin(1)/π + 1/2 = 1`. -/
example {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) :
    IsLogisticSolution P ℱ (fun _ _ => 1) ∧
      (∀ _ω : Ω, (1 : ℝ) ∈ Set.Icc (-1 : ℝ) 1) ∧
        (1 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧
          Real.arcsin 1 / π + 1 / 2 = 1 := by
  refine ⟨isLogisticSolution_one P ℱ, fun _ => ⟨by norm_num, le_refl _⟩,
    ⟨by norm_num, le_refl _⟩, ?_⟩
  rw [Real.arcsin_one]
  field_simp
  norm_num

end Homogenized
end Transformer
