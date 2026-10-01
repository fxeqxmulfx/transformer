/-
# Spatial regularity of Gaussian transitions

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
For constant coefficients, the continuous comparison operator is the actual
Gaussian expectation. Its spatial derivative commutes with expectation,
with all integrability and differentiation hypotheses discharged.
-/

import Transformer.BatchSize.Section4_GaussianExpansion

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

noncomputable section

namespace Transformer.BatchSize

/-- Gaussian transition from x with variance max(t,0), Section 4.3's
constant-coefficient Brownian driving noise. -/
def gaussianTransition (f : ℝ → ℝ) (t x : ℝ) : ℝ :=
  gaussianFlow (fun y => f (x + y)) t

/-- The comparison operator preserves a uniform observable bound,
Section 4.3's weak approximation. -/
theorem gaussianTransition_abs_le {f : ℝ → ℝ} {C : ℝ}
    (hC : ∀ y, |f y| ≤ C) (t x : ℝ) : |gaussianTransition f t x| ≤ C :=
  gaussianFlow_abs_le (fun y => hC (x + y)) t

/-- Nonvacuity of the bounded-transition hypothesis, Section 4.3. -/
example : ∀ y : ℝ, |Real.sin y| ≤ 1 := Real.abs_sin_le_one

/-- Gaussian transitions preserve continuity in the initial state,
Section 4.3's weak approximation. -/
theorem continuous_gaussianTransition {f : ℝ → ℝ} (hf : Continuous f)
    {C : ℝ} (hC : ∀ y, |f y| ≤ C) (t : ℝ) : Continuous (gaussianTransition f t) := by
  refine continuous_iff_continuousAt.mpr fun x => ?_
  exact tendsto_integral_filter_of_dominated_convergence (fun _ => C)
    (Eventually.of_forall fun x' =>
      (hf.comp (continuous_const.add (continuous_const.mul continuous_id))).aestronglyMeasurable)
    (Eventually.of_forall fun x' => ae_of_all _ fun z => by
      simpa only [Real.norm_eq_abs] using hC (x' + Real.sqrt t * z))
    (integrable_const C) (ae_of_all _ fun z =>
      (hf.comp (continuous_id.add continuous_const)).tendsto x)

/-- Joint nonvacuity of continuity and boundedness, Section 4.3. -/
example : Continuous Real.sin ∧ ∀ y : ℝ, |Real.sin y| ≤ 1 :=
  ⟨Real.continuous_sin, Real.abs_sin_le_one⟩

/-- Spatial differentiation commutes with the actual Gaussian expectation,
Section 4.3's constant-coefficient comparison operator. The derivative
bound dominates the parameter derivative on the entire real line. -/
theorem gaussianTransition_hasDerivAt {f f' : ℝ → ℝ}
    (hf : ∀ y, HasDerivAt f (f' y) y) (hf' : Continuous f')
    {C D : ℝ} (hC : ∀ y, |f y| ≤ C) (hD : ∀ y, |f' y| ≤ D) (t x : ℝ) :
    HasDerivAt (gaussianTransition f t) (gaussianTransition f' t x) x := by
  have hfc : Continuous f := continuous_iff_continuousAt.mpr fun y => (hf y).continuousAt
  refine (hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := gaussianReal 0 1)
    (F := fun x' z => f (x' + Real.sqrt t * z))
    (F' := fun x' z => f' (x' + Real.sqrt t * z)) (bound := fun _ => D)
    (s := Set.univ) (by simp)
    (Eventually.of_forall fun x' =>
      (hfc.comp (continuous_const.add (continuous_const.mul continuous_id))).aestronglyMeasurable)
    ((integrable_const C).mono'
      (hfc.comp (continuous_const.add (continuous_const.mul continuous_id))).aestronglyMeasurable
      (ae_of_all _ fun z => by simpa only [Real.norm_eq_abs] using hC _))
    (by fun_prop) (ae_of_all _ fun z x' _ => by
      simpa only [Real.norm_eq_abs] using hD (x' + Real.sqrt t * z))
    (integrable_const D) ?_).2
  refine ae_of_all _ fun z x' _ => ?_
  convert (hf (x' + Real.sqrt t * z)).comp x'
    ((hasDerivAt_id x').add_const (Real.sqrt t * z)) using 1 <;>
    simp only [Function.comp_def, id_eq, mul_one]

/-- Joint nonvacuity of the spatial derivative hypotheses, Section 4.3. -/
example : (∀ y : ℝ, HasDerivAt Real.sin (Real.cos y) y) ∧ Continuous Real.cos ∧
    (∀ y : ℝ, |Real.sin y| ≤ 1) ∧ (∀ y : ℝ, |Real.cos y| ≤ 1) :=
  ⟨Real.hasDerivAt_sin, Real.continuous_cos, Real.abs_sin_le_one, Real.abs_cos_le_one⟩

end Transformer.BatchSize
