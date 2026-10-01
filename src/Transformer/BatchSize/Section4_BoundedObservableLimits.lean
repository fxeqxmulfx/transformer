/-
# Passing bounded continuous observables to actual stochastic limits

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Subsequence extraction and dominated convergence suffice for the C2
generator tests, whose Hessians need not be globally Lipschitz.
-/

import Transformer.BatchSize.Section4_LimitMoments

open MeasureTheory Filter
open scoped Topology

noncomputable section

namespace Transformer.BatchSize

/-- Bounded observables continuous in the convergent state pass to
expectations under genuine convergence in measure, Section 4.3,
Theorem 1. The observable may also depend on a fixed random past
parameter; no continuity of that parameter is imposed. -/
theorem integral_bounded_observable_tendsto {Ω E : Type*} [MeasurableSpace Ω]
    [PseudoEMetricSpace E] (P : Measure Ω) [IsFiniteMeasure P]
    (X : ℕ → Ω → E) (Y : Ω → E) (hlim : TendstoInMeasure P X atTop Y)
    (g : Ω → E → ℝ) (hgc : ∀ ω, Continuous (g ω))
    (hgm : ∀ n, AEStronglyMeasurable (fun ω => g ω (X n ω)) P)
    (C : ℝ) (hC : ∀ ω y, |g ω y| ≤ C) :
    Tendsto (fun n => ∫ ω, g ω (X n ω) ∂P) atTop (𝓝 (∫ ω, g ω (Y ω) ∂P)) := by
  apply tendsto_of_subseq_tendsto
  intro ns hns
  obtain ⟨ms, _, hae⟩ := (hlim.comp hns).exists_seq_tendsto_ae
  refine ⟨ms, tendsto_integral_of_dominated_convergence (fun _ => C)
    (fun n => hgm (ns (ms n)))
    (integrable_const C) (fun n => Eventually.of_forall fun ω => ?_) ?_⟩
  · simpa only [Real.norm_eq_abs] using hC ω (X (ns (ms n)) ω)
  · filter_upwards [hae] with ω hω
    exact (hgc ω).continuousAt.tendsto.comp hω

/-- Two actual state sequences converging in measure can be passed
together to any bounded continuous observable, Section 4.3, Theorem 1.
This applies to the left frozen state and the interpolated state in
the actual generator; the random past weight remains fixed. -/
theorem integral_bounded_pair_observable_tendsto {Ω E D : Type*} [MeasurableSpace Ω]
    [PseudoEMetricSpace E] [PseudoEMetricSpace D] (P : Measure Ω) [IsFiniteMeasure P]
    (X : ℕ → Ω → E) (Y : Ω → E) (hX : TendstoInMeasure P X atTop Y)
    (Z : ℕ → Ω → D) (W : Ω → D) (hZ : TendstoInMeasure P Z atTop W)
    (g : Ω → E × D → ℝ) (hgc : ∀ ω, Continuous (g ω))
    (hgm : ∀ n, AEStronglyMeasurable (fun ω => g ω (X n ω, Z n ω)) P)
    (C : ℝ) (hC : ∀ ω y, |g ω y| ≤ C) :
    Tendsto (fun n => ∫ ω, g ω (X n ω, Z n ω) ∂P) atTop
      (𝓝 (∫ ω, g ω (Y ω, W ω) ∂P)) := by
  apply tendsto_of_subseq_tendsto
  intro ns hns
  obtain ⟨ms, hms, haeX⟩ := (hX.comp hns).exists_seq_tendsto_ae
  obtain ⟨ks, hks, haeZ⟩ := (hZ.comp (hns.comp hms.tendsto_atTop)).exists_seq_tendsto_ae
  refine ⟨ms ∘ ks, tendsto_integral_of_dominated_convergence (fun _ => C)
    (fun n => hgm (ns (ms (ks n)))) (integrable_const C)
    (fun n => Eventually.of_forall fun ω => ?_) ?_⟩
  · simpa only [Real.norm_eq_abs, Function.comp_def] using
      hC ω (X (ns (ms (ks n))) ω, Z (ns (ms (ks n))) ω)
  · filter_upwards [haeX, haeZ] with ω hωX hωZ
    exact (hgc ω).continuousAt.tendsto.comp
      ((hωX.comp hks.tendsto_atTop).prodMk_nhds hωZ)

/-- Joint nonvacuity of the two-state convergence hypotheses,
Section 4.3: two identical nonzero constant state sequences. -/
example : TendstoInMeasure (brownianNoiseLaw 1)
    (fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) atTop (fun _ => (1 : ℝ)) ∧
    Continuous (fun p : ℝ × ℝ => Real.sin (p.1 + p.2)) ∧
    AEStronglyMeasurable (fun _ : BrownianSample 1 => Real.sin (1 + 1)) (brownianNoiseLaw 1) ∧
    (∀ p : ℝ × ℝ, |Real.sin (p.1 + p.2)| ≤ 1) :=
  ⟨meanSquare_tendstoInMeasure (brownianNoiseLaw 1) _ (fun _ => memLp_const _) _
      (memLp_const _) (by simp),
    Real.continuous_sin.comp (continuous_fst.add continuous_snd), aestronglyMeasurable_const,
    fun p => Real.abs_sin_le_one _⟩

/-- Joint nonvacuity of bounded observable limit hypotheses,
Section 4.3: the nonconstant sine observable with a fixed random
parameter, applied to nonzero deterministic approximations. -/
example : TendstoInMeasure (brownianNoiseLaw 1)
    (fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) atTop (fun _ => (1 : ℝ)) ∧
    (∀ ω : BrownianSample 1, Continuous (fun y : ℝ => Real.sin (y + coordinateBrownian 0 1 ω))) ∧
    (AEStronglyMeasurable
      (fun ω : BrownianSample 1 => Real.sin (1 + coordinateBrownian 0 1 ω)) (brownianNoiseLaw 1)) ∧
    (∀ (ω : BrownianSample 1) (y : ℝ), |Real.sin (y + coordinateBrownian 0 1 ω)| ≤ 1) := by
  refine ⟨?_, fun _ => Real.continuous_sin.comp (continuous_id.add continuous_const),
    (Real.continuous_sin.measurable.comp
      (measurable_const.add (coordinateBrownian_measurable 0 1))).aestronglyMeasurable,
    fun _ _ => Real.abs_sin_le_one _⟩
  exact meanSquare_tendstoInMeasure (brownianNoiseLaw 1) _ (fun _ => memLp_const _) _
    (memLp_const _) (by simp)

end Transformer.BatchSize
