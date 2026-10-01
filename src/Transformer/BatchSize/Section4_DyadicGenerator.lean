/-
# The measurable frozen generator of an actual Brownian Euler path

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The coefficients use the genuine preceding dyadic state, while the test
is evaluated on the complete continuous Brownian interpolation.
-/

import Transformer.BatchSize.Section4_DyadicIntervals
import Transformer.BatchSize.Section4_EulerIntervals

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual preceding state on the stopped dyadic Brownian grid,
Section 4.3 (2)--(3). -/
def dyadicLeftState {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (x₀ : EucSpace d) (T : NNReal) (m : ℕ)
    (u : ℝ) (ω : BrownianSample d) : EucSpace d :=
  eulerChain b a (dyadicGrid T m) x₀ (dyadicLeftIndex u.toNNReal m) ω

/-- The genuine frozen generator along the full dyadic Euler path,
Section 4.3 (2)--(3). Both arguments are actual random states. -/
def dyadicGeneratorValue {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (x₀ : EucSpace d) (T : NNReal) (m : ℕ)
    (φ : EucSpace d → ℝ) (u : ℝ) (ω : BrownianSample d) : ℝ :=
  frozenGaussianGenerator (b (dyadicLeftState b a x₀ T m u ω))
    (a (dyadicLeftState b a x₀ T m u ω)) φ
    (eulerPathValue b a (dyadicGrid T m) x₀ (dyadicEndpointIndex T m) ω u)

/-- The preceding actual random state is jointly measurable in
time and Brownian sample, Section 4.3 (2)--(3). -/
theorem dyadicLeftState_measurable {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (x₀ : EucSpace d) (T : NNReal) (m : ℕ) :
    Measurable (fun p : ℝ × BrownianSample d => dyadicLeftState b a x₀ T m p.1 p.2) := by
  have hchain : Measurable (fun p : ℕ × BrownianSample d =>
      eulerChain b a (dyadicGrid T m) x₀ p.1 p.2) :=
    measurable_from_prod_countable_right fun j =>
      ((eulerChain_adapted b a hb ha _ (dyadicGrid_monotone T m) x₀ j).mono
        ((brownianFiltration d).le _)).measurable
  exact hchain.comp (((dyadicLeftIndex_measurable m).comp measurable_fst).prodMk measurable_snd)

/-- Joint measurability of the actual interpolated Euler state,
Section 4.3 (2)--(3), follows from its continuous actual samples. -/
theorem eulerPathValue_measurable_time_sample {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (t : ℕ → NNReal) (hmono : Monotone t) (x₀ : EucSpace d) (N : ℕ) :
    Measurable (fun p : ℝ × BrownianSample d => eulerPathValue b a t x₀ N p.2 p.1) :=
  measurable_uncurry_of_continuous_of_measurable
    (eulerPathValue_continuous b a t x₀ N)
    (fun u => (ContinuousMap.measurable_eval u).comp
      (eulerPath_measurable b a hb ha t hmono x₀ N))

/-- The actual frozen generator is jointly measurable, for every
genuine bounded C2 test, Section 4.3 (2)--(3). -/
theorem dyadicGeneratorValue_measurable {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (x₀ : EucSpace d) (T : NNReal) (m : ℕ) (φ : EucSpace d → ℝ)
    (hφ : BoundedSmoothTest 2 φ) :
    Measurable (fun p : ℝ × BrownianSample d => dyadicGeneratorValue b a x₀ T m φ p.1 p.2) :=
  frozenGaussianGenerator_measurable_parameters _ _ _
    (hb.measurable.comp (dyadicLeftState_measurable b a hb ha x₀ T m))
    (fun k => (ha k).measurable.comp (dyadicLeftState_measurable b a hb ha x₀ T m))
    (eulerPathValue_measurable_time_sample b a hb ha _ (dyadicGrid_monotone T m) x₀ _) φ hφ

/-- The weighted actual generator expectation is measurable in
time and uniformly bounded, independently of mesh size,
Section 4.3 (2)--(3). These are actual Bochner integrals. -/
theorem dyadicGenerator_expectation_bounded {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (M A : NNReal) (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (T : NNReal) (m : ℕ) (φ : EucSpace d → ℝ)
    (hφ : BoundedSmoothTest 2 φ) (F : BrownianSample d → ℝ)
    (hFm : Measurable F) (hF1 : ∀ ω, |F ω| ≤ 1) :
    Measurable (fun u => ∫ ω, F ω * dyadicGeneratorValue b a x₀ T m φ u ω ∂brownianNoiseLaw d) ∧
      ∀ u, |∫ ω, F ω * dyadicGeneratorValue b a x₀ T m φ u ω ∂brownianNoiseLaw d| ≤
        (M : ℝ) + (d : ℝ) * (A : ℝ) ^ 2 / 2 := by
  have hm := (hFm.comp measurable_snd).mul (dyadicGeneratorValue_measurable b a hb ha x₀ T m φ hφ)
  refine ⟨hm.stronglyMeasurable.integral_prod_right'.measurable, fun u => ?_⟩
  have h := norm_integral_le_of_norm_le_const (μ := brownianNoiseLaw d)
    (f := fun ω => F ω * dyadicGeneratorValue b a x₀ T m φ u ω)
    (C := (M : ℝ) + (d : ℝ) * (A : ℝ) ^ 2 / 2) (Eventually.of_forall fun ω => ?_)
  · simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using h
  · rw [Real.norm_eq_abs, abs_mul]
    exact (mul_le_mul (hF1 ω) (frozenGaussianGenerator_uniform_bound _ _ M A
      (hbM _) (haA _) φ hφ _) (abs_nonneg _) (by norm_num)).trans_eq (one_mul _)

/-- Joint nonvacuity of the measurable bounded generator hypotheses,
Section 4.3: constant bounded drift and positive diagonal noise,
with a nonzero bounded past weight and normalized C2 test. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ k : Fin 1, Continuous (fun _ : EucSpace 1 => (k : ℝ) + 1)) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : NNReal)) ∧
    (∀ k : Fin 1, |(k : ℝ) + 1| ≤ (1 : NNReal)) ∧
    Measurable (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨continuous_const, fun _ => continuous_const, fun _ => by simp,
    (fun k => by fin_cases k; norm_num), measurable_const, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
