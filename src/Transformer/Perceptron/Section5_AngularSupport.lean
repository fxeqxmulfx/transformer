module

public import Transformer.Perceptron.Section5_CircleGeometry
public import Transformer.Perceptron.Section5_CriticalSupport
public import Mathlib.MeasureTheory.Measure.Support
public import Mathlib.MeasureTheory.Measure.Dirac.Basic

/-!
# Finite stationary support from an actual angular curvature condition

ArXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron` and
`eq:strictSOPD-sufficient`. A compact angular representative interval
turns isolated critical support points into a finite set. The image
of that finite set covers the entire topological support of the
actual measure on `𝕊¹`, because the angular chart is surjective.

A measure with finite topological support is then represented by
its actual singleton masses and Dirac measures, using the general
measure-theory lemmas in `Atomicity.lean`. No atomicity conclusion
of the paper is assumed by the support reduction.

The angular zero and curvature conditions are explicit inputs.
The later first-variation calculation must establish them for the
actual interaction and perceptron energy before applying this result.
-/

@[expose] public section

open Set Real MeasureTheory
open scoped Topology

namespace Transformer.Perceptron

/-- An actual angular force that vanishes at support angles and has
a nonzero derivative there isolates those angles, making support finite.
Source: arXiv:2601.21366v2, §5.5, the zero-isolation step behind
`eq:strictSOPD-sufficient`. This calculus step needs neither positivity
nor continuity of the supplied derivative witnesses. The angular chart
covers every sphere point on a compact interval of representatives. -/
theorem finite_support_of_angular_force_nonzero_derivative (μ : Perspective.ProbSphere 2)
    (g : ℝ → ℝ)
    (hzero : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support → g θ = 0)
    (hderiv : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      ∃ d : ℝ, HasDerivAt g d θ ∧ d ≠ 0) :
    (μ : Measure (SSphere 2)).support.Finite := by
  let S : Set ℝ := Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support
  have hcompact : IsCompact S := isCompact_Icc.inter_right
    (Measure.isClosed_support.preimage continuous_circlePoint)
  have hfinite : S.Finite := finite_zero_set_of_nonzero_derivative g S hcompact
    (fun θ hθ => hzero θ hθ.2) (fun θ hθ => hderiv θ hθ.2)
  apply (hfinite.image circlePoint).subset
  intro x hx
  obtain ⟨θ, hθ, heq⟩ := exists_circlePoint_eq x
  refine ⟨θ, ⟨hθ, ?_⟩, heq⟩
  change circlePoint θ ∈ (μ : Measure (SSphere 2)).support
  rwa [heq]

/-- Nonzero actual second derivatives suffice for the topological
support reduction. Source: arXiv:2601.21366v2, §5.5, the calculus step
behind `eq:strictSOPD-sufficient`. This is stronger than its positive
curvature input, without changing any later strict SOPD criterion. -/
theorem finite_support_of_nonzero_angular_curvature (μ : Perspective.ProbSphere 2)
    (f : ℝ → ℝ)
    (hcrit : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support → deriv f θ = 0)
    (hcurv : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      deriv^[2] f θ ≠ 0) : (μ : Measure (SSphere 2)).support.Finite := by
  apply finite_support_of_angular_force_nonzero_derivative μ (deriv f) hcrit
  intro θ hθ
  have hn := hcurv θ hθ
  change deriv (deriv f) θ ≠ 0 at hn
  exact ⟨deriv (deriv f) θ, (differentiableAt_of_deriv_ne_zero hn).hasDerivAt, hn⟩

/-- **Positive angular curvature at critical support points forces
the topological support to be finite.** Source: arXiv:2601.21366v2,
§5.5, the angular stationarity equation and `eq:strictSOPD-sufficient`.
The supplied `f` will be the actual first variation. The theorem
does not assume any finite presentation of the measure. -/
theorem finite_support_of_positive_angular_curvature (μ : Perspective.ProbSphere 2)
    (f : ℝ → ℝ)
    (hcrit : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support → deriv f θ = 0)
    (hcurv : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      0 < deriv^[2] f θ) :
    (μ : Measure (SSphere 2)).support.Finite := by
  exact finite_support_of_nonzero_angular_curvature μ f hcrit (fun θ hθ => (hcurv θ hθ).ne')

/-- The isolated-force condition gives the source's finite Dirac
presentation of the actual measure. Source: arXiv:2601.21366v2,
§5.5, the support reduction in `rem:strictSOPD-perceptron`.
Only the actual force zero and derivative hypotheses are supplied. -/
theorem isFinitelyAtomic_of_angular_force_nonzero_derivative (μ : Perspective.ProbSphere 2)
    (g : ℝ → ℝ)
    (hzero : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support → g θ = 0)
    (hderiv : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      ∃ d : ℝ, HasDerivAt g d θ ∧ d ≠ 0) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_angular_force_nonzero_derivative μ g hzero hderiv)

/-- Nonzero actual angular curvature at critical support angles gives
the same finite Dirac conclusion. Source: arXiv:2601.21366v2, §5.5,
the zero-isolation step behind `eq:strictSOPD-sufficient`.
No finite-support or global smoothness input is assumed. -/
theorem isFinitelyAtomic_of_nonzero_angular_curvature (μ : Perspective.ProbSphere 2)
    (f : ℝ → ℝ)
    (hcrit : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support → deriv f θ = 0)
    (hcurv : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      deriv^[2] f θ ≠ 0) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_nonzero_angular_curvature μ f hcrit hcurv)

/-- **The angular curvature condition gives actual finite atomicity.**
Source: arXiv:2601.21366v2, §5.5, the support reduction behind
`rem:strictSOPD-perceptron`. This is the calculus and measure-theory
part of that reduction; the zero and curvature conditions on the
actual angular first variation remain explicit. -/
theorem isFinitelyAtomic_of_positive_angular_curvature (μ : Perspective.ProbSphere 2)
    (f : ℝ → ℝ)
    (hcrit : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support → deriv f θ = 0)
    (hcurv : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      0 < deriv^[2] f θ) : IsFinitelyAtomic μ := by
  exact isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_positive_angular_curvature μ f hcrit hcurv)

/-- The two angular conditions are jointly realized by the actual
support of the Dirac mass at the first basis vector and `f(θ)=-cos θ`.
Every representative of that point has cosine one and sine zero.
This includes all periodic representatives, not only angle zero. -/
example : let μ := Perspective.diracProb 2 (basePoint 1)
    (∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      deriv (fun s : ℝ => -Real.cos s) θ = 0) ∧
    (∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      0 < deriv^[2] (fun s : ℝ => -Real.cos s) θ) ∧
    (∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      deriv^[2] (fun s : ℝ => -Real.cos s) θ ≠ 0) ∧
    (∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support → Real.sin θ = 0) ∧
    (∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      ∃ d : ℝ, HasDerivAt Real.sin d θ ∧ d ≠ 0) := by
  intro μ
  have hfirst : deriv (fun s : ℝ => -Real.cos s) = Real.sin := by
    funext θ
    simpa only [Pi.neg_def, neg_neg] using (Real.hasDerivAt_cos θ).neg.deriv
  have hsecond (θ : ℝ) : deriv^[2] (fun s : ℝ => -Real.cos s) θ = Real.cos θ := by
    change deriv (deriv (fun s : ℝ => -Real.cos s)) θ = Real.cos θ
    rw [hfirst, Real.deriv_sin]
  have hcoords (θ : ℝ) (hθ : circlePoint θ ∈ (μ : Measure (SSphere 2)).support) :
      Real.cos θ = 1 ∧ Real.sin θ = 0 := by
    have heq : circlePoint θ = (basePoint 1 : SSphere 2) :=
      Interpolation.eq_of_mem_support_dirac hθ
    have hc := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 0) heq
    have hs := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 1) heq
    constructor
    · rw [circlePoint_coord_zero] at hc
      simpa [basePoint] using hc
    · rw [circlePoint_coord_one] at hs
      simpa [basePoint] using hs
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro θ hθ
    rw [hfirst, (hcoords θ hθ).2]
  · intro θ hθ
    rw [hsecond, (hcoords θ hθ).1]
    exact zero_lt_one
  · intro θ hθ
    rw [hsecond, (hcoords θ hθ).1]
    exact one_ne_zero
  · intro θ hθ
    exact (hcoords θ hθ).2
  · intro θ hθ
    exact ⟨Real.cos θ, Real.hasDerivAt_sin θ, by rw [(hcoords θ hθ).1]; exact one_ne_zero⟩

end Transformer.Perceptron
