/-
# Finite stationary support from an actual angular curvature condition

ArXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron` and
`eq:strictSOPD-sufficient`. A compact angular representative interval
turns isolated critical support points into a finite set. The image
of that finite set covers the entire topological support of the
actual measure on `𝕊¹`, because the angular chart is surjective.

A measure with finite topological support is then represented by
its actual singleton masses and Dirac measures. The step uses the
fact that a Borel measure on this second-countable space carries
its topological support; no atomicity result of the paper is used.

The angular zero and curvature conditions are explicit inputs.
The later first-variation calculation must establish them for the
actual interaction and perceptron energy before applying this result.
-/

import Transformer.Perceptron.Section5_CircleGeometry
import Transformer.Perceptron.Section5_CriticalSupport
import Mathlib.MeasureTheory.Measure.Support
import Mathlib.MeasureTheory.Measure.Dirac.Basic

open Set Real MeasureTheory
open scoped Topology

namespace Transformer.Perceptron

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
  let S : Set ℝ := Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support
  have hcompact : IsCompact S := isCompact_Icc.inter_right
    (Measure.isClosed_support.preimage continuous_circlePoint)
  have hfinite : S.Finite := finite_critical_set_of_positive_second_derivative f S hcompact
    (fun θ hθ => hcrit θ hθ.2) (fun θ hθ => hcurv θ hθ.2)
  apply (hfinite.image circlePoint).subset
  intro x hx
  obtain ⟨θ, hθ, heq⟩ := exists_circlePoint_eq x
  refine ⟨θ, ⟨hθ, ?_⟩, heq⟩
  change circlePoint θ ∈ (μ : Measure (SSphere 2)).support
  rwa [heq]

/-- **A probability measure with finite topological support is a finite
combination of Dirac masses, with its actual singleton masses.**
Source: arXiv:2601.21366v2, §3.1, the meaning of finite atomicity
in `thm: circle` and `thm: circle.gelu`.
The ambient dimension is arbitrary; the sphere has a Borel measure
space with measurable singletons and second-countable topology. -/
theorem isFinitelyAtomic_of_finite_topological_support {d : ℕ}
    (μ : Perspective.ProbSphere d) (hfinite : (μ : Measure (SSphere d)).support.Finite) :
    IsFinitelyAtomic μ := by
  classical
  have hcarry : ∀ᵐ x ∂(μ : Measure (SSphere d)), x ∈ hfinite.toFinset := by
    filter_upwards [Measure.support_mem_ae (μ := (μ : Measure (SSphere d)))] with x hx
    exact hfinite.mem_toFinset.mpr hx
  exact ⟨hfinite.toFinset, fun x => (μ : Measure (SSphere d)) {x},
    Measure.ae_mem_finset_iff.mp hcarry⟩

/-- The finite-support input is realized on the actual nonempty
one-dimensional sphere by its Dirac probability measure. -/
example : (Perspective.diracProb 1 (basePoint 0) : Measure (SSphere 1)).support.Finite := by
  apply (finite_singleton (basePoint 0)).subset
  intro x hx
  exact mem_singleton_iff.mpr (Interpolation.eq_of_mem_support_dirac hx)

/-- **A finite Dirac presentation has finite topological support.**
Source: arXiv:2601.21366v2, §3.1, the meaning of finite atomicity
in `thm: circle` and `thm: circle.gelu`. Together with the converse
above this verifies that the calculus step yields the source's
actual measure-theoretic conclusion. Zero-mass atoms are permitted. -/
theorem finite_support_of_finitely_atomic {d : ℕ} (μ : Perspective.ProbSphere d)
    (hμ : IsFinitelyAtomic μ) : (μ : Measure (SSphere d)).support.Finite := by
  classical
  obtain ⟨F, m, heq⟩ := hμ
  have hsub (G : Finset (SSphere d)) :
      (∑ z ∈ G, m z • Measure.dirac z).support ⊆ (G : Set (SSphere d)) := by
    induction G using Finset.induction_on with
    | empty => simp
    | @insert z G hz ih =>
      rw [Finset.sum_insert hz, Measure.support_add]
      intro x hx
      rcases hx with hx | hx
      · have hxz : x = z := Interpolation.eq_of_mem_support_dirac
          (Measure.smul_absolutelyContinuous.support_mono hx)
        subst x
        exact Finset.mem_insert_self z G
      · exact Finset.mem_insert_of_mem (ih hx)
  rw [heq]
  exact F.finite_toSet.subset (hsub F)

/-- The finite-presentation input is satisfied by an actual Dirac
probability measure on a nonempty sphere. -/
example : IsFinitelyAtomic (Perspective.diracProb 1 (basePoint 0)) :=
  isFinitelyAtomic_diracProb (basePoint 0)

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
      0 < deriv^[2] (fun s : ℝ => -Real.cos s) θ) := by
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
  constructor
  · intro θ hθ
    rw [hfirst, (hcoords θ hθ).2]
  · intro θ hθ
    rw [hsecond, (hcoords θ hθ).1]
    exact zero_lt_one

end Transformer.Perceptron
