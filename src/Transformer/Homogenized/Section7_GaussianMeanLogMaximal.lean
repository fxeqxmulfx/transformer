module

public import Transformer.Homogenized.Section7_GaussianMeanLogChernoff

/-!
# Two-sided original logarithmic mean noise and its measurable paths

ArXiv:2604.01978v1, §7, the maximal argument in the proof of
`thm:clustering_small_beta`. Both signs of the actual compensated
logarithmic observable give its two-sided uniform tail. Its paths
are continuous and its uniform event is measurable, with regularity
derived from the original Gaussian generator and sphere solution.

The logarithm uses a fixed positive regularization. The full original
martingale problem supplies every selected exponential test; no
additional tail estimate, regularity of the drift, or measurable
uniform event is assumed.
-/

@[expose] public section

open Set MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- Both signs give the actual continuous-time two-sided logarithmic mean-noise bound.
Source: arXiv:2604.01978v1, §7, the maximal argument after the mean bracket estimate,
with coefficient `4/d` derived for the genuine regularized logarithmic test. -/
theorem gaussian_logMean_noise_maximal {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (γ : ℝ) (hinit : ∀ ω, logMeanOverlap (meanLogRegularization d β σA) (X 0 ω) = γ)
    (r : ℝ) (hr : 0 < r) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T,
      r < |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω|} ≤
      2 * ENNReal.ofReal (Real.exp (-(r ^ 2 * (d : ℝ) / (8 * T)))) := by
  have hp := gaussian_logMean_signed_maximal hd hn β hβ σV σA ρ hρ hσ hsmall P ℱ T hT X hMP
    γ hinit r hr 1 (by norm_num)
  have hm := gaussian_logMean_signed_maximal hd hn β hβ σV σA ρ hρ hσ hsmall P ℱ T hT X hMP
    γ hinit r hr (-1) (by norm_num)
  have hsub : {ω | ∃ t ∈ Icc (0 : ℝ) T,
      r < |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω|} ⊆
      {ω | ∃ t ∈ Icc (0 : ℝ) T,
        r < 1 * logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω} ∪
      {ω | ∃ t ∈ Icc (0 : ℝ) T,
        r < (-1) * logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω} := by
    rintro ω ⟨t, ht, hc⟩
    rcases lt_abs.mp hc with hpos | hneg
    · exact Or.inl ⟨t, ht, by simpa only [one_mul] using hpos⟩
    · exact Or.inr ⟨t, ht, by simpa only [neg_one_mul] using hneg⟩
  exact (measure_mono hsub).trans ((measure_union_le _ _).trans
    ((add_le_add hp hm).trans_eq (by rw [two_mul])))

/-- The actual logarithmic mean residual has continuous paths on the source horizon.
Source: arXiv:2604.01978v1, §7, the mean decomposition after Appendix `def:ito.formula`.
Continuity of the genuine Gaussian logarithmic generator supplies the primitive. -/
theorem continuousOn_logMeanNoiseResidual {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    {a : ℝ} (ha : 0 < a) (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (γ : ℝ) (ω : Ω) :
    ContinuousOn (fun t => logMeanNoiseResidual a β ρ X γ t ω) (Icc (0 : ℝ) T) := by
  let D := fun t : ℝ => sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) (X t ω)
  have hDc : ContinuousOn D (Icc (0 : ℝ) T) :=
    (continuousOn_sphGenerator_logMeanOverlap_gaussian hd ha hn β σV σA ρ hρ hσ).comp'
      (hMP.2.2.1 ω) (fun t ht => hMP.1 t ht ω)
  have hi : IntegrableOn D (uIcc (0 : ℝ) T) volume := by
    rw [uIcc_of_le hT.le]
    exact hDc.integrableOn_Icc
  have hprim : ContinuousOn (fun t => ∫ s in (0 : ℝ)..t, D s) (Icc (0 : ℝ) T) := by
    simpa only [uIcc_of_le hT.le] using intervalIntegral.continuousOn_primitive_interval hi
  exact ((((contDiff_logMeanOverlap ha).continuous.comp_continuousOn (hMP.2.2.1 ω)).sub
    continuousOn_const).sub hprim)

/-- The actual uniform-in-time logarithmic mean-noise event is measurable.
Source: arXiv:2604.01978v1, §7, the continuous maximal argument for
`thm:clustering_small_beta`; adaptation and path regularity come from the original solution. -/
theorem measurableSet_logMeanNoiseResidual_le {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    {a : ℝ} (ha : 0 < a) (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (γ r : ℝ) :
    MeasurableSet {ω | ∀ t ∈ Icc (0 : ℝ) T, |logMeanNoiseResidual a β ρ X γ t ω| ≤ r} := by
  let D := fun t : ℝ => fun ω : Ω => sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
    (logMeanOverlap a) (X t ω)
  have hfield := continuousOn_sphGenerator_logMeanOverlap_gaussian hd ha hn β σV σA ρ hρ hσ
  have hDc (ω : Ω) : ContinuousOn (fun t => D t ω) (Icc (0 : ℝ) T) :=
    hfield.comp' (hMP.2.2.1 ω) (fun t ht => hMP.1 t ht ω)
  have hDm (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (D t) :=
    stronglyMeasurable_comp_of_continuousOn_range _ _ hfield (X t) (hMP.2.1 t ht).measurable (hMP.1 t ht)
  apply measurableSet_continuous_time_le (fun t ω => |logMeanNoiseResidual a β ρ X γ t ω|) T hT
  · intro t ht
    have hprim := stronglyMeasurable_adapted_timeIntegral ℱ T D hDm hDc t ht
    have hres : StronglyMeasurable[ℱ t] (fun ω => logMeanNoiseResidual a β ρ X γ t ω) :=
      ((((contDiff_logMeanOverlap ha).continuous.comp_stronglyMeasurable (hMP.2.1 t ht)).sub
        stronglyMeasurable_const).sub hprim)
    have habs : Continuous (fun z : ℝ => |z|) := continuous_id.abs
    exact ((habs.comp_stronglyMeasurable hres).mono (ℱ.le t)).measurable
  · intro ω
    exact (continuousOn_logMeanNoiseResidual hd hn ha β σV σA ρ hρ hσ P ℱ T hT X hMP γ ω).abs

/-- The measurable event of uniform original logarithmic noise control
has the complement probability furnished by its actual two-sided maximal bound.
Source: arXiv:2604.01978v1, §7, the maximal estimate in
`thm:clustering_small_beta`, after the genuine logarithmic test. -/
theorem gaussian_logMean_noise_good_measure {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (γ : ℝ) (hinit : ∀ ω, logMeanOverlap (meanLogRegularization d β σA) (X 0 ω) = γ)
    (r : ℝ) (hr : 0 < r) :
    1 - 2 * ENNReal.ofReal (Real.exp (-(r ^ 2 * (d : ℝ) / (8 * T)))) ≤
      P {ω | ∀ t ∈ Icc (0 : ℝ) T,
        |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω| ≤ r} := by
  let Bad : Set Ω := {ω | ∃ t ∈ Icc (0 : ℝ) T,
    r < |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω|}
  have hGood : {ω | ∀ t ∈ Icc (0 : ℝ) T,
      |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω| ≤ r} = Badᶜ := by
    ext ω
    simp only [Bad, mem_ofPred_eq, mem_compl_iff, not_exists, not_and, not_lt]
  have hGoodMeas := measurableSet_logMeanNoiseResidual_le hd hn
    (meanLogRegularization_pos (d := d) (by omega) β σA) β σV σA ρ hρ hσ P ℱ T hT X hMP γ r
  rw [hGood] at hGoodMeas
  have hBadMeas : MeasurableSet Bad := by simpa only [compl_compl] using hGoodMeas.compl
  rw [hGood, measure_compl hBadMeas (measure_ne_top P Bad), measure_univ]
  exact tsub_le_tsub_left
    (gaussian_logMean_noise_maximal hd hn β hβ σV σA ρ hρ hσ hsmall P ℱ T hT X hMP γ hinit r hr) 1

/-- The positive-variance original Gaussian head and frozen antipodal sphere
solution jointly realize the maximal and path-measurability premises. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)) (γ : ℝ),
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ meanLogRegularization 2 0 1 ≤ 1 ∧
    0 < meanLogRegularization 2 0 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧
    IsSphereMartingaleSolution (gaussHeadLaw 2 (stdSigmaV 2) 1)
      (bField 0 (gaussHeadLaw 2 (stdSigmaV 2) 1)) (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1))
      (Measure.dirac ()) ℱ 1 X ∧
    (∀ ω, logMeanOverlap (meanLogRegularization 2 0 1) (X 0 ω) = γ) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let K : Idx 2 → EucSpace 2 := ![v, -v]
  refine ⟨ℱ, fun _ _ => K, logMeanOverlap (meanLogRegularization 2 0 1) K,
    le_rfl, by decide, by norm_num, isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2,
    ?_, meanLogRegularization_pos (by decide) 0 1, one_pos, by norm_num, ?_, fun ω => rfl⟩
  · norm_num [meanLogRegularization]
  · exact isSphereMartingaleSolution_antipodal_zero_beta _ (Measure.dirac ()) ℱ 1 v (by simp [v])

end Transformer.Homogenized
