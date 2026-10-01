/-
# Actual probability laws of continuous Euler approximations

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The approximations are measures on the very continuous-path space used by
the martingale problem. Their endpoint marginals are the computed chains.
-/

import Transformer.BatchSize.Section4_EulerPaths

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The law of an actual finite Euler chain's endpoint,
Section 4.3 (2)--(3). -/
def eulerEndpointLaw {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ) : Measure (EucSpace d) :=
  (brownianNoiseLaw d).map (eulerChain b a t x₀ n)

/-- The law of the actual continuous Euler interpolation,
Section 4.3 (2)--(3). It is a finite approximation, not a postulated SDE law. -/
def eulerPathLaw {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ) : Measure (DiffusionPath d) :=
  (brownianNoiseLaw d).map (eulerPath b a t x₀ n)

/-- Normalization of the continuous Euler law, Section 4.3 (2)--(3),
verified from the constructed sampler's measurability. -/
theorem eulerPathLaw_probability {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) : IsProbabilityMeasure (eulerPathLaw b a t x₀ n) :=
  (Measure.isProbabilityMeasure_map_iff (eulerPath_measurable b a hb ha t hmono x₀ n).aemeasurable).mpr
    (by infer_instance)

/-- The actual path measure starts at x0, Section 4.3 (2)--(3). -/
theorem eulerPathLaw_initial {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) : ∀ᵐ ω ∂eulerPathLaw b a t x₀ n, ω 0 = x₀ := by
  have hevent : MeasurableSet {ω : DiffusionPath d | ω 0 = x₀} :=
    measurableSet_eq_fun (ContinuousMap.measurable_eval (0 : ℝ)) measurable_const
  rw [eulerPathLaw, ae_map_iff (eulerPath_measurable b a hb ha t hmono x₀ n).aemeasurable hevent]
  exact Filter.Eventually.of_forall (eulerPath_zero b a t x₀ n)

/-- The continuous Euler law's final marginal agrees with its actual
finite chain's endpoint distribution, Section 4.3 (2)--(3). -/
theorem eulerPathLaw_endpoint {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) :
    HasLaw (fun ω : DiffusionPath d => ω (t n)) (eulerEndpointLaw b a t x₀ n) (eulerPathLaw b a t x₀ n) := by
  refine ⟨(ContinuousMap.measurable_eval (t n : ℝ)).aemeasurable, ?_⟩
  rw [eulerPathLaw, Measure.map_map (ContinuousMap.measurable_eval (t n : ℝ))
    (eulerPath_measurable b a hb ha t hmono x₀ n)]
  unfold eulerEndpointLaw
  congr 1
  funext ω
  exact eulerPath_endpoint b a t hmono x₀ n ω

/-- Joint nonvacuity of all Euler-law hypotheses, Section 4.3:
linear drift, positive constant diagonal amplitudes, and unit grid times. -/
example : Continuous (id : EucSpace 2 → EucSpace 2) ∧
    (∀ k : Fin 2, Continuous (fun _ : EucSpace 2 => (k : ℝ) + 1)) ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) :=
  ⟨continuous_id, fun _ => continuous_const,
    fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h⟩

end Transformer.BatchSize
