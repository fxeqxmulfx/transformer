/-
# A concrete Gaussian comparison operator

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The scalar constant-coefficient Gaussian transition acts on the same
bounded-observable type as the weak-error telescope. Its contraction,
iteration law and propagated smoothness bounds are proved, rather than
left as premises about an unnamed continuous comparison operator.
-/

import Transformer.BatchSize.Section4_GaussianHeatEquation
import Transformer.BatchSize.Section4_ExpectationOperator

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- Actual Gaussian transition expectations on bounded measurable
observables, Section 4.3's continuous comparison for constant scalar noise. -/
def gaussianExpectationOperator (t : ℝ) : BoundedObservable ℝ → BoundedObservable ℝ := fun φ =>
  ⟨gaussianTransition φ.val t,
    ((φ.property.1.comp (measurable_fst.add
      (measurable_const.mul measurable_snd))).stronglyMeasurable.integral_prod_right').measurable,
    by obtain ⟨C, hC⟩ := φ.property.2; exact ⟨C, gaussianTransition_abs_le hC t⟩⟩

/-- Gaussian comparison contracts uniform error, Section 4.3's weak-error
telescope. This holds also for nonsmooth bounded measurable observables. -/
theorem gaussianExpectationOperator_contraction (t : ℝ) (φ ψ : BoundedObservable ℝ)
    (r : ℝ) (hclose : ∀ x, |φ.val x - ψ.val x| ≤ r) (x : ℝ) :
    |(gaussianExpectationOperator t φ).val x -
      (gaussianExpectationOperator t ψ).val x| ≤ r := by
  have hg : Measurable (fun z : ℝ => x + Real.sqrt t * z) := by fun_prop
  exact probability_expectation_contraction (gaussianReal 0 1)
    (φ.comp (fun z => x + Real.sqrt t * z) hg)
    (ψ.comp (fun z => x + Real.sqrt t * z) hg) r
    (fun z => hclose (x + Real.sqrt t * z))

/-- Joint nonvacuity of contraction's closeness hypothesis, Section 4.3. -/
example : ∃ φ ψ : BoundedObservable ℝ, ∀ x, |φ.val x - ψ.val x| ≤ (0 : ℝ) := by
  let φ : BoundedObservable ℝ := ⟨fun _ => 1, measurable_const, 1, by simp⟩
  exact ⟨φ, φ, by simp⟩

/-- The iterated comparison equals a single Gaussian transition at elapsed
time n*t, Section 4.3. Independent variances add under the proved semigroup law. -/
theorem gaussianExpectationOperator_iterate (φ : BoundedObservable ℝ)
    (hf : Continuous φ.val) (t : ℝ) (ht : 0 ≤ t) (n : ℕ) :
    ((gaussianExpectationOperator t)^[n] φ).val = gaussianTransition φ.val (n * t) := by
  obtain ⟨C, hC⟩ := φ.property.2
  induction n with
  | zero => funext x; simp [gaussianTransition_zero]
  | succ n ih =>
    rw [Function.iterate_succ_apply']
    change gaussianTransition (((gaussianExpectationOperator t)^[n] φ).val) t = _
    rw [ih]
    funext x
    rw [← gaussianTransition_add hf hC ht (mul_nonneg (Nat.cast_nonneg n) ht) x]
    congr 1
    push_cast
    ring

/-- Joint nonvacuity of all iteration hypotheses, Section 4.3. -/
example : ∃ φ : BoundedObservable ℝ, Continuous φ.val ∧ (0 : ℝ) ≤ 1 / 1000 :=
  ⟨⟨fun _ => 1, measurable_const, 1, by simp⟩, continuous_const, by norm_num⟩

/-- Propagated tests obey the original derivative bound at every step
of this actual comparison operator, Section 4.3. The result uses n finite
derivative orders, without an artificial normalization at later times. -/
theorem gaussianExpectationOperator_propagated_bounds {m : ℕ} (φ : BoundedObservable ℝ)
    (hf : ContDiff ℝ m φ.val) {R : ℝ}
    (hR : ∀ j ≤ m, ∀ x, |iteratedDeriv j φ.val x| ≤ R)
    (t : ℝ) (ht : 0 ≤ t) (n : ℕ) :
    ContDiff ℝ m (((gaussianExpectationOperator t)^[n] φ).val) ∧
      ∀ j ≤ m, ∀ x, |iteratedDeriv j (((gaussianExpectationOperator t)^[n] φ).val) x| ≤ R := by
  rw [gaussianExpectationOperator_iterate φ hf.continuous t ht n]
  exact gaussianTransition_preserves_smooth_bound hf hR (n * t)

/-- Joint nonvacuity of the propagated smooth bounds, Section 4.3. -/
example : ∃ φ : BoundedObservable ℝ, ContDiff ℝ 6 φ.val ∧
    (∀ j ≤ 6, ∀ x, |iteratedDeriv j φ.val x| ≤ 1) ∧ (0 : ℝ) ≤ 1 / 1000 := by
  let φ : BoundedObservable ℝ := ⟨fun _ => 1, measurable_const, 1, by simp⟩
  refine ⟨φ, contDiff_const, ?_, by norm_num⟩
  intro j hj x
  rw [show φ.val = (fun _ : ℝ => (1 : ℝ)) from rfl, iteratedDeriv_const]
  split_ifs <;> norm_num

end Transformer.BatchSize
