module

public import Mathlib.Analysis.Calculus.ParametricIntegral
public import Mathlib.Analysis.Complex.CauchyIntegral
public import Mathlib.Analysis.Complex.RealDeriv
public import Mathlib.Analysis.Analytic.Constructions
public import Mathlib.Analysis.Analytic.Linear
public import Mathlib.MeasureTheory.Function.LocallyIntegrable
public import Mathlib.Analysis.Normed.Group.Bounded

/-!
# Analytic interaction integrals on compact spaces

ArXiv:2601.21366v2, §6, proof of `prop: unified.log` and
`rem:unifiedlog-B`: the actual exponential interaction is analytic.
This module proves the integral step through a complex extension.
The extension and its actual pointwise derivative are explicit inputs;
the circle kernel will supply them in the next module.

Joint continuity bounds that derivative on a closed complex disk
times the compact integration space. A finite measure then gives
local domination and an actual derivative of the integral.
The same argument applies at every complex parameter.
Complex differentiability implies analyticity. Restricting to real
arguments and taking the real part gives the needed real analyticity.
The trace lemma identifies the integral with its original real kernel.
-/

@[expose] public section

open Complex MeasureTheory Filter Set Metric
open scoped Topology

namespace Transformer.Perceptron

variable {α : Type*} [TopologicalSpace α] [CompactSpace α]

/-- A continuous complex integrand is bounded on a unit disk times the
actual compact integration space. Source: arXiv:2601.21366v2, §6,
the local domination behind the analyticity used in `prop: unified.log`. -/
theorem exists_local_bound_complex_compact (F : ℂ × α → ℂ) (hF : Continuous F) (z : ℂ) :
    ∃ C : ℝ, ∀ w ∈ ball z 1, ∀ x : α, ‖F (w, x)‖ ≤ C := by
  have hcompact : IsCompact (closedBall z 1 ×ˢ (univ : Set α)) :=
    (ProperSpace.isCompact_closedBall z 1).prod isCompact_univ
  obtain ⟨C, hC⟩ := hcompact.exists_bound_of_continuousOn hF.continuousOn
  refine ⟨C, fun w hw x => hC (w, x) ⟨?_, mem_univ x⟩⟩
  exact mem_closedBall.mpr (mem_ball.mp hw).le

variable [MeasurableSpace α] [OpensMeasurableSpace α]

/-- The complex integral has the integral of the actual pointwise
derivative as its actual derivative. Source: arXiv:2601.21366v2, §6,
the analytic exponential interaction in `prop: unified.log`.
Joint continuity supplies local domination, without a global bound. -/
theorem hasDerivAt_complex_integral_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℂ → α → ℂ)
    (hF : Continuous (fun q : ℂ × α => F q.1 q.2))
    (hF' : Continuous (fun q : ℂ × α => F' q.1 q.2))
    (hD : ∀ z x, HasDerivAt (fun w => F w x) (F' z x) z) (z : ℂ) :
    HasDerivAt (fun w => ∫ x, F w x ∂μ) (∫ x, F' z x ∂μ) z := by
  obtain ⟨C, hC⟩ := exists_local_bound_complex_compact _ hF' z
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := μ)
    (F := F) (F' := F') (bound := fun _ => C) (s := ball z 1) (x₀ := z)
    (ball_mem_nhds z one_pos)
    (Eventually.of_forall fun w =>
      (hF.comp (continuous_const.prodMk continuous_id)).aestronglyMeasurable)
    ((hF.comp (continuous_const.prodMk continuous_id)).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _))
    (hF'.comp (continuous_const.prodMk continuous_id)).aestronglyMeasurable
    (ae_of_all _ fun x w hw => hC w hw x) (integrable_const C)
    (ae_of_all _ fun x w _ => hD w x)
  exact h.2

/-- The proved derivatives make the actual integral complex differentiable
at every point. Source: arXiv:2601.21366v2, §6, the analyticity of the
interaction used by `prop: unified.log` and `rem:unifiedlog-B`. -/
theorem differentiable_complex_integral_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℂ → α → ℂ)
    (hF : Continuous (fun q : ℂ × α => F q.1 q.2))
    (hF' : Continuous (fun q : ℂ × α => F' q.1 q.2))
    (hD : ∀ z x, HasDerivAt (fun w => F w x) (F' z x) z) :
    Differentiable ℂ (fun w => ∫ x, F w x ∂μ) :=
  fun z => (hasDerivAt_complex_integral_compact μ F F' hF hF' hD z).differentiableAt

/-- The actual complex integral is analytic on the whole complex plane.
Source: arXiv:2601.21366v2, §6, the interaction analyticity used in
`prop: unified.log`. The existence of its derivatives was proved above. -/
theorem analyticOnNhd_complex_integral_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℂ → α → ℂ)
    (hF : Continuous (fun q : ℂ × α => F q.1 q.2))
    (hF' : Continuous (fun q : ℂ × α => F' q.1 q.2))
    (hD : ∀ z x, HasDerivAt (fun w => F w x) (F' z x) z) :
    AnalyticOnNhd ℂ (fun w => ∫ x, F w x ∂μ) univ :=
  fun z _ => (differentiable_complex_integral_compact μ F F' hF hF' hD).analyticAt z

/-- Restricting the actual complex integral to real parameters and taking
its real part preserves analyticity. Source: arXiv:2601.21366v2, §6,
the real-analytic interaction in the proof of `prop: unified.log`. -/
theorem analyticOnNhd_real_part_integral_compact (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℂ → α → ℂ)
    (hF : Continuous (fun q : ℂ × α => F q.1 q.2))
    (hF' : Continuous (fun q : ℂ × α => F' q.1 q.2))
    (hD : ∀ z x, HasDerivAt (fun w => F w x) (F' z x) z) :
    AnalyticOnNhd ℝ (fun r : ℝ => (∫ x, F (r : ℂ) x ∂μ).re) univ := by
  intro r _
  have ha : AnalyticAt ℝ (fun w : ℂ => ∫ x, F w x ∂μ) (r : ℂ) :=
    ((analyticOnNhd_complex_integral_compact μ F F' hF hF' hD) _ (mem_univ _)).restrictScalars
  exact (Complex.reCLM.analyticAt _).comp (ha.comp (Complex.ofRealCLM.analyticAt r))

/-- The real part of the actual integral also has its actual derivative
given by the real part of the proved complex derivative. Source:
arXiv:2601.21366v2, §6, the first-variation derivative in
`prop: unified.log`. Restriction uses the original real parameter. -/
theorem hasDerivAt_real_part_complex_integral (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℂ → α → ℂ)
    (hF : Continuous (fun q : ℂ × α => F q.1 q.2))
    (hF' : Continuous (fun q : ℂ × α => F' q.1 q.2))
    (hD : ∀ z x, HasDerivAt (fun w => F w x) (F' z x) z) (r : ℝ) :
    HasDerivAt (fun t : ℝ => (∫ x, F (t : ℂ) x ∂μ).re)
      ((∫ x, F' (r : ℂ) x ∂μ).re) r :=
  (hasDerivAt_complex_integral_compact μ F F' hF hF' hD (r : ℂ)).real_of_complex

/-- An actual complex extension gives analyticity of the integral of
its original real trace. Source: arXiv:2601.21366v2, §6,
the exponential interaction in `prop: unified.log`.
The pointwise trace equality keeps the original real kernel explicit. -/
theorem analyticOnNhd_integral_real_trace (μ : Measure α) [IsFiniteMeasure μ]
    (F F' : ℂ → α → ℂ)
    (hF : Continuous (fun q : ℂ × α => F q.1 q.2))
    (hF' : Continuous (fun q : ℂ × α => F' q.1 q.2))
    (hD : ∀ z x, HasDerivAt (fun w => F w x) (F' z x) z)
    (f : ℝ → α → ℝ) (htrace : ∀ (r : ℝ) x, F (r : ℂ) x = (f r x : ℂ)) :
    AnalyticOnNhd ℝ (fun r => ∫ x, f r x ∂μ) univ := by
  have heq : (fun r : ℝ => (∫ x, F (r : ℂ) x ∂μ).re) = fun r => ∫ x, f r x ∂μ := by
    funext r
    simp only [htrace, integral_complex_ofReal, Complex.ofReal_re]
  rw [← heq]
  exact analyticOnNhd_real_part_integral_compact μ F F' hF hF' hD

/-- All joint continuity, actual derivative and trace inputs hold
simultaneously for the exponential and its actual derivative, integrated
against the actual Dirac measure on the nonempty compact space `Unit`. -/
example : IsFiniteMeasure (Measure.dirac ()) ∧
    Continuous (fun q : ℂ × Unit => Complex.exp q.1) ∧
    Continuous (fun q : ℂ × Unit => Complex.exp q.1) ∧
    (∀ z : ℂ, ∀ x : Unit, HasDerivAt (fun w => (fun r (_ : Unit) => Complex.exp r) w x)
      (Complex.exp z) z) ∧
    (∀ r : ℝ, ∀ x : Unit, (fun z (_ : Unit) => Complex.exp z) (r : ℂ) x =
      ((fun t (_ : Unit) => Real.exp t) r x : ℂ)) :=
  ⟨inferInstance, Complex.continuous_exp.comp continuous_fst,
    Complex.continuous_exp.comp continuous_fst, fun z _ => Complex.hasDerivAt_exp z,
    fun r _ => (Complex.ofReal_exp r).symm⟩

end Transformer.Perceptron
