module

public import Transformer.Homogenized.MvGenerator
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

/-!
# Homogenized Transformers — the limiting logistic SDE

Formalization of `eq: sde.logistic` of arXiv:2604.01978v1, *Homogenized
Transformers*, and of the long-time behavior worked out for it in
`sec: sde.logistic`:

  `du(t) = -u(t)(1 - u(t)²) dt + √2 (1 - u(t)²) dB(t)`.

This is the scalar limit of the overlap `⟨x⁽¹⁾(td), x⁽²⁾(td)⟩` in
`thm:large_beta_meta`. Mathlib has no stochastic integral, so the equation is
written as its martingale problem, with the generator

  `𝖫φ(u) = -u(1-u²) φ'(u) + (1-u²)² φ''(u)`,

the `(1-u²)²` being `½ (√2(1-u²))²`. The solution concept carries no initial
condition: `eq: sde.logistic`'s `u(0) = 0` is a hypothesis of the statements
that use it, and leaving it out makes the absorbing states `u ≡ ±1` available
as the witness that the concept is satisfiable.

The source's `sec: sde.logistic` sets `θ = arcsin u`; it writes
"θ(t) = sin u(t)", which cannot be meant — the computation that follows,
`dθ = √2 cos θ dB`, is the Itô formula for `arcsin`, and it is `arcsin` that
makes `θ(t) ∈ [-π/2, π/2]` automatic. `arcsin` is what is written here.
-/

@[expose] public section

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
identically `0`. This is the absorbing state the long-time analysis of
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

/-- The other endpoint is a genuine absorbing solution as well: both original
coefficients vanish at -1. Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`,
the two absorbing states in the long-time limit. -/
theorem isLogisticSolution_neg_one {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    [IsProbabilityMeasure P] (ℱ : Filtration ℝ m) :
    IsLogisticSolution P ℱ (fun _ _ => -1) := by
  refine ⟨fun _ _ => stronglyMeasurable_const, fun _ => continuousOn_const, ?_⟩
  intro φ _ T _
  have hzero : logisticMart (fun (_ : ℝ) (_ : Ω) => (-1 : ℝ)) φ = fun _ _ => 0 := by
    funext t ω
    simp [logisticMart]
  rw [hzero]
  exact isMartingaleOn_const T ℱ P 0

/-- Both original absorbing endpoints are genuine full smooth-test solutions.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the two limiting absorbing states. -/
theorem isLogisticSolution_endpoint {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    [IsProbabilityMeasure P] (ℱ : Filtration ℝ m) (a : ℝ) (ha : a = 1 ∨ a = -1) :
    IsLogisticSolution P ℱ (fun _ _ => a) := by
  rcases ha with h | h
  · rw [h]
    exact isLogisticSolution_one P ℱ
  · rw [h]
    exact isLogisticSolution_neg_one P ℱ

/-- Both signs occur as initial values of actual solutions under a probability law. -/
example : IsLogisticSolution (Measure.dirac ())
    (Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le)
    (fun _ _ => -1) :=
  isLogisticSolution_endpoint _ _ (-1) (Or.inr rfl)

/-! ### Long-time behavior -/

/- Interval invariance is proved in Section7_LogisticRange. -/

/- The angle and bracket theorem is proved in Section7_LogisticArcsin. -/

/- The original long-time limit is proved in Section7_LogisticLimit. -/

/-- The hypotheses of `logistic_range`, `logistic_arcsin_martingale` and
`logistic_limit` are satisfiable at once, on the absorbing solution `u ≡ 1`:
it solves the equation, it starts and stays in `[-1,1]`, and its initial value
is the deterministic `u₀ = 1`. The conclusion of `logistic_limit` is then the
true statement `P(u_∞ = 1) = arcsin(1)/π + 1/2 = 1`. -/
example {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P]
    (ℱ : Filtration ℝ m) :
    IsLogisticSolution P ℱ (fun _ _ => 1) ∧
      (∀ _ : Ω, (1 : ℝ) ∈ Set.Icc (-1 : ℝ) 1) ∧
        (1 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧
          Real.arcsin 1 / π + 1 / 2 = 1 := by
  refine ⟨isLogisticSolution_one P ℱ, fun _ => ⟨by norm_num, le_refl _⟩,
    ⟨by norm_num, le_refl _⟩, ?_⟩
  rw [Real.arcsin_one]
  field_simp
  norm_num

end Homogenized
end Transformer
