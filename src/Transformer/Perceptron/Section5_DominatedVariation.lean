module

public import Mathlib.Analysis.Calculus.ParametricIntegral
public import Mathlib.MeasureTheory.Function.LocallyIntegrable
public import Mathlib.Analysis.Normed.Group.Bounded
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Dominated second variations on compact probability spaces

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. Differentiating
the transported energy for a measure with arbitrary support needs
two applications of differentiation under the integral sign.
Continuity on the compact integration space gives measurability
and integrability. Uniform bounds for the two actual derivative
integrands provide domination at every transport time.

The hypotheses below are calculus inputs, not assumptions of the
paper's atomicity theorem. The rotated kernel and perceptron must
supply them before that theorem can use the second variation.
-/

@[expose] public section

open Real MeasureTheory Filter Set Topology

namespace Transformer.Perceptron

variable {α : Type*} [TopologicalSpace α] [CompactSpace α]

/-- A continuous real function on the compact integration space has
a positive uniform bound. Source: arXiv:2601.21366v2, §5.2, the
bounded angular speeds used in `eq:Hess-bilinear.std`. -/
theorem exists_pos_bound_continuous_compact (f : α → ℝ) (hf : Continuous f) :
    ∃ B : ℝ, 0 < B ∧ ∀ x : α, |f x| ≤ B := by
  obtain ⟨C, hC⟩ := isCompact_univ.exists_bound_of_continuousOn hf.continuousOn
  refine ⟨max C 0 + 1, by positivity, fun x => ?_⟩
  have hx : |f x| ≤ C := by simpa only [Real.norm_eq_abs] using hC x (mem_univ x)
  exact hx.trans ((le_max_left C 0).trans (by linarith))

/-- The continuity input is realized by a nonzero constant function
on the nonempty compact space `Unit`. -/
example : Continuous (fun _ : Unit => (1 : ℝ)) := continuous_const

variable [MeasurableSpace α] [OpensMeasurableSpace α]

/-- Differentiation under a finite measure on a compact space.
Source: arXiv:2601.21366v2, §5.2, the first differentiation
preceding `eq:Hess-bilinear.std`. The supplied derivative is an
actual pointwise derivative and has a uniform integrable bound. -/
theorem hasDerivAt_integral_on_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℝ → α → ℝ) (hF : ∀ t, Continuous (F t))
    (hF' : ∀ t, Continuous (F' t)) (B : ℝ)
    (hB : ∀ t x, |F' t x| ≤ B)
    (hD : ∀ t x, HasDerivAt (fun r => F r x) (F' t x) t) (t : ℝ) :
    HasDerivAt (fun r => ∫ x, F r x ∂μ) (∫ x, F' t x ∂μ) t := by
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := μ) (F := F) (F' := F') (bound := fun _ => B)
    (s := univ) (x₀ := t) (by simp)
    (Eventually.of_forall fun r => (hF r).aestronglyMeasurable)
    ((hF t).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _))
    (hF' t).aestronglyMeasurable
    (ae_of_all _ fun x r _ => by simpa only [Real.norm_eq_abs] using hB r x)
    (integrable_const B) (ae_of_all _ fun x r _ => hD r x)
  exact h.2

/-- The justified first derivative as an equality of actual functions.
Source: arXiv:2601.21366v2, §5.2, the first differentiation of
the transported energy before `eq:Hess-bilinear.std`. -/
theorem deriv_integral_on_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℝ → α → ℝ) (hF : ∀ t, Continuous (F t))
    (hF' : ∀ t, Continuous (F' t)) (B : ℝ)
    (hB : ∀ t x, |F' t x| ≤ B)
    (hD : ∀ t x, HasDerivAt (fun r => F r x) (F' t x) t) :
    deriv (fun r => ∫ x, F r x ∂μ) = fun r => ∫ x, F' r x ∂μ := by
  funext r
  exact (hasDerivAt_integral_on_compact μ F F' hF hF' B hB hD r).deriv

/-- Two dominated differentiations give the actual second derivative
of the integral. Source: arXiv:2601.21366v2, §5.2,
`eq:Hess-bilinear.std`. The first derivative is established as a
function identity before the second differentiation is taken. -/
theorem hasDerivAt_deriv_integral_on_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' F'' : ℝ → α → ℝ) (hF : ∀ t, Continuous (F t))
    (hF' : ∀ t, Continuous (F' t)) (hF'' : ∀ t, Continuous (F'' t))
    (B₁ B₂ : ℝ) (hB₁ : ∀ t x, |F' t x| ≤ B₁)
    (hB₂ : ∀ t x, |F'' t x| ≤ B₂)
    (hD : ∀ t x, HasDerivAt (fun r => F r x) (F' t x) t)
    (hD' : ∀ t x, HasDerivAt (fun r => F' r x) (F'' t x) t) (t : ℝ) :
    HasDerivAt (deriv (fun r => ∫ x, F r x ∂μ)) (∫ x, F'' t x ∂μ) t := by
  rw [deriv_integral_on_compact μ F F' hF hF' B₁ hB₁ hD]
  exact hasDerivAt_integral_on_compact μ F' F'' hF' hF'' B₂ hB₂ hD' t

/-- The proved second derivative as an actual iterated-derivative value.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`.
The same domination inputs are used, including existence of both
pointwise derivatives; the total derivative is not assigned a value. -/
theorem secondDeriv_integral_on_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' F'' : ℝ → α → ℝ) (hF : ∀ t, Continuous (F t))
    (hF' : ∀ t, Continuous (F' t)) (hF'' : ∀ t, Continuous (F'' t))
    (B₁ B₂ : ℝ) (hB₁ : ∀ t x, |F' t x| ≤ B₁)
    (hB₂ : ∀ t x, |F'' t x| ≤ B₂)
    (hD : ∀ t x, HasDerivAt (fun r => F r x) (F' t x) t)
    (hD' : ∀ t x, HasDerivAt (fun r => F' r x) (F'' t x) t) (t : ℝ) :
    deriv^[2] (fun r => ∫ x, F r x ∂μ) t = ∫ x, F'' t x ∂μ := by
  exact (hasDerivAt_deriv_integral_on_compact μ F F' F'' hF hF' hF''
    B₁ B₂ hB₁ hB₂ hD hD' t).deriv

/-- All continuity, domination, first- and second-derivative inputs
are jointly realized by the actual cosine function and its two
derivatives, integrated against the probability measure `δ_()`. -/
example :
    (∀ t : ℝ, Continuous (fun _ : Unit => Real.cos t)) ∧
    (∀ t : ℝ, Continuous (fun _ : Unit => -Real.sin t)) ∧
    (∀ t : ℝ, Continuous (fun _ : Unit => -Real.cos t)) ∧
    (∀ t : ℝ, |-(Real.sin t)| ≤ 1) ∧
    (∀ t : ℝ, |-(Real.cos t)| ≤ 1) ∧
    (∀ t : ℝ, HasDerivAt (fun r => Real.cos r) (-Real.sin t) t) ∧
    (∀ t : ℝ, HasDerivAt (fun r => -Real.sin r) (-Real.cos t) t) ∧
    HasDerivAt (deriv (fun r => ∫ _ : Unit, Real.cos r ∂Measure.dirac ())) (-1) 0 := by
  refine ⟨fun _ => continuous_const, fun _ => continuous_const, fun _ => continuous_const,
    fun t => ?_, fun t => ?_, fun t => Real.hasDerivAt_cos t,
    fun t => (Real.hasDerivAt_sin t).neg, ?_⟩
  · simpa only [abs_neg] using Real.abs_sin_le_one t
  · simpa only [abs_neg] using Real.abs_cos_le_one t
  · have h := hasDerivAt_deriv_integral_on_compact (Measure.dirac ())
      (fun (t : ℝ) (_ : Unit) => Real.cos t) (fun (t : ℝ) (_ : Unit) => -Real.sin t)
      (fun (t : ℝ) (_ : Unit) => -Real.cos t)
      (fun _ => continuous_const) (fun _ => continuous_const) (fun _ => continuous_const)
      1 1 (fun t _ => by simpa only [abs_neg] using Real.abs_sin_le_one t)
      (fun t _ => by simpa only [abs_neg] using Real.abs_cos_le_one t)
      (fun t _ => Real.hasDerivAt_cos t) (fun t _ => (Real.hasDerivAt_sin t).neg) 0
    simpa using h

/-- Bounded angular speeds bound their difference by twice the bound.
Source: arXiv:2601.21366v2, §5.2, the pair-speed difference in
`eq:Hess-bilinear.std`. No regularity of the speed is needed here. -/
theorem abs_speed_sub_le {ι : Type*} (s : ι → ℝ) (B : ℝ)
    (hB : ∀ x, |s x| ≤ B) (x y : ι) : |s x - s y| ≤ 2 * B := by
  have h : |s x - s y| ≤ |s x| + |s y| := by
    simpa only [sub_zero, zero_sub, abs_neg] using abs_sub_le (s x) 0 (s y)
  linarith [hB x, hB y]

/-- The same domination for the squared speed difference in the
second variation. Source: arXiv:2601.21366v2, §5.2,
`eq:Hess-bilinear.std`. The nonnegative bound is kept explicit. -/
theorem speed_sub_sq_le {ι : Type*} (s : ι → ℝ) (B : ℝ) (hB₀ : 0 ≤ B)
    (hB : ∀ x, |s x| ≤ B) (x y : ι) : (s x - s y) ^ 2 ≤ (2 * B) ^ 2 := by
  have h := abs_speed_sub_le s B hB x y
  have hsq : |s x - s y| ^ 2 ≤ (2 * B) ^ 2 := by
    nlinarith [abs_nonneg (s x - s y)]
  simpa only [sq_abs] using hsq

/-- All speed-domination inputs are realized by constant unit speed
and bound one, on the same nonempty compact integration space. -/
example : (0 : ℝ) ≤ 1 ∧ ∀ x : Unit, |(fun _ : Unit => (1 : ℝ)) x| ≤ 1 := by
  norm_num

end Transformer.Perceptron
