/-
# Integrability for fourth-moment expansions with adapted coefficients

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
All terms in the polynomial expansion are genuinely integrable.
The bounded amplitude may depend on the whole vector Brownian past.
-/

import Transformer.BatchSize.Section4_BrownianHigherMoments
import Transformer.BatchSize.Section4_ItoSums

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal ENNReal

noncomputable section

namespace Transformer.BatchSize

/-- Finite fourth moment gives integrability of every lower natural
power on a finite measure, Section 4.3, Theorem 1. -/
theorem memLp_four_integrable_pow {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsFiniteMeasure P] (Z : Ω → ℝ) (hZ : MemLp Z 4 P)
    (j : ℕ) (hj : j ≤ 4) : Integrable (fun ω => Z ω ^ j) P := by
  have hZj : MemLp Z (j : ℝ≥0∞) P := hZ.mono_exponent (by exact_mod_cast hj)
  have hm := (continuous_pow j).comp_aestronglyMeasurable hZ.aestronglyMeasurable
  apply (integrable_norm_iff hm).mp
  simpa only [Real.norm_eq_abs, abs_pow] using hZj.integrable_norm_pow'

/-- A bounded adapted coefficient times a fresh Brownian increment
has finite fourth moment, Section 4.3 (2)--(3). -/
theorem brownianIncrement_bounded_memLp_four {d : ℕ} (k : Fin d) (s t : ℝ≥0)
    (H : BrownianSample d → ℝ) (hH : StronglyMeasurable[brownianFiltration d s] H)
    (A : ℝ≥0) (hbound : ∀ ω, |H ω| ≤ A) :
    MemLp (fun ω => H ω * brownianIncrement k s t ω) 4 (brownianNoiseLaw d) := by
  have hm : AEStronglyMeasurable H (brownianNoiseLaw d) :=
    (hH.mono ((brownianFiltration d).le s)).aestronglyMeasurable
  have hHtop : MemLp H ∞ (brownianNoiseLaw d) := MemLp.of_bound hm A
    (Eventually.of_forall fun ω => by simpa only [Real.norm_eq_abs] using hbound ω)
  exact hHtop.mul (brownianIncrement_memLp_four k s t)

/-- The adapted power kernel through fourth order includes actual
integrability of its product, Section 4.3 (2)--(3). -/
theorem brownianIncrement_adapted_power {d : ℕ} (k : Fin d) (s t : ℝ≥0) (hst : s ≤ t)
    (H : BrownianSample d → ℝ) (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHint : Integrable H (brownianNoiseLaw d)) (j : ℕ) (hj : j ≤ 4) :
    Integrable (fun ω => H ω * brownianIncrement k s t ω ^ j) (brownianNoiseLaw d) ∧
      (∫ ω, H ω * brownianIncrement k s t ω ^ j ∂brownianNoiseLaw d) =
        (∫ ω, H ω ∂brownianNoiseLaw d) *
          (∫ ω, brownianIncrement k s t ω ^ j ∂brownianNoiseLaw d) := by
  have hi := (brownianIncrement_independent_adapted k s t hst hH).comp
    (φ := id) (ψ := fun z : ℝ => z ^ j) measurable_id (continuous_pow j).measurable
  simp only [Function.comp_def, id_eq] at hi
  have hΔ := memLp_four_integrable_pow (brownianNoiseLaw d) _
    (brownianIncrement_memLp_four k s t) j hj
  exact ⟨hi.integrable_mul hHint hΔ,
    hi.integral_fun_mul_eq_mul_integral hHint.aestronglyMeasurable hΔ.aestronglyMeasurable⟩

/-- A bounded adapted stochastic sum has finite fourth moment,
Section 4.3 (2)--(3). This uses actual finite sums of Brownian increments. -/
theorem brownianItoSum_memLp_four {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (H : ℕ → BrownianSample d → ℝ)
    (hH : ∀ j, StronglyMeasurable[brownianFiltration d (t j)] (H j))
    (A : ℝ≥0) (hbound : ∀ j ω, |H j ω| ≤ A) (n : ℕ) :
    MemLp (brownianItoSum k t H n) 4 (brownianNoiseLaw d) :=
  memLp_finsetSum _ fun j _ => brownianIncrement_bounded_memLp_four k (t j) (t (j + 1))
    (H j) (hH j) A (hbound j)

/-- Joint nonvacuity of lower-power integrability, Section 4.3:
an actual nonconstant Gaussian coordinate and power three. -/
example : MemLp (coordinateBrownian (0 : Fin 1) 1) 4 (brownianNoiseLaw 1) ∧ (3 : ℕ) ≤ 4 :=
  ⟨(coordinateBrownian_isBrownian (0 : Fin 1)).isGaussianProcess.hasGaussianLaw_eval 1 |>.memLp
    (by norm_num), by norm_num⟩

/-- Joint nonvacuity of bounded-product and adapted power kernels,
Section 4.3: the constant unit coefficient on a genuine future increment. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    Integrable (fun _ : BrownianSample 1 => (1 : ℝ)) (brownianNoiseLaw 1) ∧
    (∀ ω : BrownianSample 1, |(fun _ : BrownianSample 1 => (1 : ℝ)) ω| ≤ (1 : ℝ≥0)) ∧
    (4 : ℕ) ≤ 4 :=
  ⟨by norm_num, stronglyMeasurable_const, integrable_const _, by simp, le_rfl⟩

/-- Joint nonvacuity of bounded fourth-moment sums, Section 4.3:
unit coefficients on an increasing half-unit grid. -/
example : (∀ j : ℕ, StronglyMeasurable[brownianFiltration 1 ((j : ℝ≥0) / 2)]
    ((fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) j)) ∧
    (∀ (j : ℕ) (ω : BrownianSample 1),
      |(fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) j ω| ≤ (1 : ℝ≥0)) :=
  ⟨fun _ => stronglyMeasurable_const, by simp⟩

end Transformer.BatchSize
