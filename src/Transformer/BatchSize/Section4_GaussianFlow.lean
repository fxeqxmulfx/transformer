/-
# Gaussian flow and its time derivative

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The scalar Brownian expectation is differentiated against a fixed standard
Gaussian measure, with an integrable domination bound. Gaussian integration
by parts then identifies its generator as one half of the second derivative.
-/

import Transformer.BatchSize.Section4_GaussianIntegration

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

noncomputable section

namespace Transformer.BatchSize

/-- Expectation along a centered Gaussian with variance max(t,0),
Section 4.3's Brownian driving noise. At t = 0 it evaluates the observable
at zero; the Brownian-time results use nonnegative t. -/
def gaussianFlow (f : ℝ → ℝ) (t : ℝ) : ℝ :=
  ∫ z, f (Real.sqrt t * z) ∂gaussianReal 0 1

/-- Initial value of the Gaussian flow, Section 4.3, equations (2)--(3). -/
theorem gaussianFlow_zero (f : ℝ → ℝ) : gaussianFlow f 0 = f 0 := by
  simp [gaussianFlow]

/-- Bounded observables remain bounded under Gaussian flow, Section 4.3,
equations (2)--(3). -/
theorem gaussianFlow_abs_le {f : ℝ → ℝ} {C : ℝ} (hC : ∀ x, |f x| ≤ C) (t : ℝ) :
    |gaussianFlow f t| ≤ C := by
  simpa only [gaussianFlow, Real.norm_eq_abs, probReal_univ, mul_one] using
    norm_integral_le_of_norm_le_const (μ := gaussianReal 0 1)
      (f := fun z => f (Real.sqrt t * z)) (ae_of_all _ fun z => by
        simpa only [Real.norm_eq_abs] using hC (Real.sqrt t * z))

/-- Nonvacuity of the bounded-flow hypothesis, Section 4.3. -/
example : ∀ x : ℝ, |Real.sin x| ≤ 1 := Real.abs_sin_le_one

/-- Continuity includes the zero-time boundary, Section 4.3's Brownian
flow. The fixed Gaussian measure permits dominated convergence at zero. -/
theorem continuous_gaussianFlow {f : ℝ → ℝ} (hf : Continuous f)
    {C : ℝ} (hC : ∀ x, |f x| ≤ C) : Continuous (gaussianFlow f) := by
  refine continuous_iff_continuousAt.mpr fun t => ?_
  exact tendsto_integral_filter_of_dominated_convergence (fun _ => C)
    (Eventually.of_forall fun s =>
      (hf.comp (continuous_const.mul continuous_id)).aestronglyMeasurable)
    (Eventually.of_forall fun s => ae_of_all _ fun z => by
      simpa only [Real.norm_eq_abs] using hC (Real.sqrt s * z))
    (integrable_const C) (ae_of_all _ fun z =>
      (hf.comp (Real.continuous_sqrt.mul continuous_const)).tendsto t)

/-- Joint nonvacuity of continuity and the uniform bound, Section 4.3. -/
example : Continuous Real.sin ∧ ∀ x : ℝ, |Real.sin x| ≤ 1 :=
  ⟨Real.continuous_sin, Real.abs_sin_le_one⟩

/-- Differentiation under the fixed Gaussian integral at positive time,
Section 4.3, equations (2)--(3). The derivative is dominated on s > t/2
by a constant times |Z|, which has finite Gaussian expectation. -/
theorem gaussianFlow_hasDerivAt_rescaled {f f' : ℝ → ℝ}
    (hf : ∀ x, HasDerivAt f (f' x) x) (hf' : Continuous f')
    {C D : ℝ} (hC : ∀ x, |f x| ≤ C) (hD : ∀ x, |f' x| ≤ D)
    {t : ℝ} (ht : 0 < t) :
    HasDerivAt (gaussianFlow f)
      (∫ z, f' (Real.sqrt t * z) * (z / (2 * Real.sqrt t)) ∂gaussianReal 0 1) t := by
  have hfc : Continuous f := continuous_iff_continuousAt.mpr fun x => (hf x).continuousAt
  have hD0 : 0 ≤ D := (abs_nonneg (f' 0)).trans (hD 0)
  let F' : ℝ → ℝ → ℝ := fun s z => f' (Real.sqrt s * z) * (z / (2 * Real.sqrt s))
  have hint : Integrable (fun z => f (Real.sqrt t * z)) (gaussianReal 0 1) :=
    (integrable_const C).mono'
      (hfc.comp (continuous_const.mul continuous_id)).aestronglyMeasurable
      (ae_of_all _ fun z => by simpa only [Real.norm_eq_abs] using hC _)
  have hz := ((memLp_id_gaussianReal (μ := 0) (v := 1) 1).integrable (by norm_num)).norm
  refine (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := gaussianReal 0 1)
    (F := fun s z => f (Real.sqrt s * z)) (F' := F')
    (bound := fun z => (D / (2 * Real.sqrt (t / 2))) * |z|)
    (Ioi_mem_nhds (by linarith : t / 2 < t))
    (Eventually.of_forall fun s =>
      (hfc.comp (continuous_const.mul continuous_id)).aestronglyMeasurable)
    hint (by dsimp [F']; fun_prop) ?_ ?_ ?_).2
  · refine ae_of_all _ fun z s hs => ?_
    have hspos : 0 < s := by have hs' : t / 2 < s := hs; linarith
    have hsq : Real.sqrt (t / 2) ≤ Real.sqrt s := Real.sqrt_le_sqrt hs.le
    dsimp [F']
    rw [abs_mul, abs_div, abs_of_pos (by positivity : 0 < 2 * Real.sqrt s)]
    calc |f' (Real.sqrt s * z)| * (|z| / (2 * Real.sqrt s))
        ≤ D * (|z| / (2 * Real.sqrt s)) :=
          mul_le_mul_of_nonneg_right (hD _) (by positivity)
      _ ≤ D * (|z| / (2 * Real.sqrt (t / 2))) := by
          gcongr
      _ = D / (2 * Real.sqrt (t / 2)) * |z| := by ring
  · simpa only [Real.norm_eq_abs, id_eq] using hz.const_mul (D / (2 * Real.sqrt (t / 2)))
  · refine ae_of_all _ fun z s hs => ?_
    have hspos : 0 < s := by have hs' : t / 2 < s := hs; linarith
    convert (hf (Real.sqrt s * z)).comp s
      ((Real.hasDerivAt_sqrt hspos.ne').mul_const z) using 1 <;>
      simp only [F', Function.comp_def]
    ring

/-- Joint nonvacuity of the rescaled derivative hypotheses, Section 4.3. -/
example : (∀ x : ℝ, HasDerivAt Real.sin (Real.cos x) x) ∧ Continuous Real.cos ∧
    (∀ x : ℝ, |Real.sin x| ≤ 1) ∧ (∀ x : ℝ, |Real.cos x| ≤ 1) ∧ (0 : ℝ) < 1 :=
  ⟨Real.hasDerivAt_sin, Real.continuous_cos, Real.abs_sin_le_one,
    Real.abs_cos_le_one, by norm_num⟩

/-- The actual scalar Gaussian flow has generator one half of the second
derivative, Section 4.3's Brownian term in equations (2)--(3). -/
theorem gaussianFlow_hasDerivAt {f f' f'' : ℝ → ℝ}
    (hf : ∀ x, HasDerivAt f (f' x) x) (hf' : ∀ x, HasDerivAt f' (f'' x) x)
    (hf'' : Continuous f'') {C D E : ℝ}
    (hC : ∀ x, |f x| ≤ C) (hD : ∀ x, |f' x| ≤ D) (hE : ∀ x, |f'' x| ≤ E)
    {t : ℝ} (ht : 0 < t) :
    HasDerivAt (gaussianFlow f) ((1 / 2) * gaussianFlow f'' t) t := by
  have hf'c : Continuous f' := continuous_iff_continuousAt.mpr fun x => (hf' x).continuousAt
  have hstein := standardGaussian_integration_by_parts
    (g := fun z => f' (Real.sqrt t * z))
    (g' := fun z => f'' (Real.sqrt t * z) * Real.sqrt t)
    (fun z => by
      convert (hf' _).comp z ((hasDerivAt_id z).const_mul (Real.sqrt t)) using 1 <;>
        simp only [Function.comp_def, id_eq, mul_one])
    (by fun_prop) (fun z => hD (Real.sqrt t * z))
    (D := E * Real.sqrt t) (fun z => by
      rw [abs_mul, abs_of_nonneg (Real.sqrt_nonneg t)]
      exact mul_le_mul_of_nonneg_right (hE _) (Real.sqrt_nonneg t))
  rw [integral_mul_const] at hstein
  convert gaussianFlow_hasDerivAt_rescaled hf hf'c hC hD ht using 1
  have heq : (fun z => f' (Real.sqrt t * z) * (z / (2 * Real.sqrt t))) =
      (fun z => (z * f' (Real.sqrt t * z)) / (2 * Real.sqrt t)) := by funext z; ring
  rw [heq]
  change (1 / 2) * gaussianFlow f'' t =
    ∫ z, (z * f' (Real.sqrt t * z)) / (2 * Real.sqrt t) ∂gaussianReal 0 1
  rw [integral_div, hstein]
  dsimp [gaussianFlow]
  field_simp [(Real.sqrt_pos.mpr ht).ne']

/-- Joint nonvacuity of the heat generator hypotheses, Section 4.3. -/
example : (∀ x : ℝ, HasDerivAt Real.sin (Real.cos x) x) ∧
    (∀ x : ℝ, HasDerivAt Real.cos (-Real.sin x) x) ∧ Continuous (fun x => -Real.sin x) ∧
    (∀ x : ℝ, |Real.sin x| ≤ 1) ∧ (∀ x : ℝ, |Real.cos x| ≤ 1) ∧
    (∀ x : ℝ, |-Real.sin x| ≤ 1) ∧ (0 : ℝ) < 1 := by
  exact ⟨Real.hasDerivAt_sin, Real.hasDerivAt_cos, Real.continuous_sin.neg,
    Real.abs_sin_le_one, Real.abs_cos_le_one,
    fun x => by simpa only [abs_neg] using Real.abs_sin_le_one x, by norm_num⟩

end Transformer.BatchSize
