/-
# The original Gaussian mean path obeys the source error on its proved noise event

ArXiv:2604.01978v1, §7, `thm:clustering_small_beta` and
`eq:smallbeta_main_bound`, at the adopted value variance `1/d`.
The actual mean stays in `[0,1]` on the original sphere solution.
Its actual logarithmic Gaussian generator is continuous and has
the proved relative drift bounds. Its initial mean is positive
under the original pairwise lower bound.

The full original compensated logarithmic noise event therefore
implies the source error with universal constant 4096. No scalar
drift estimate, reference range, or selected exponential martingale
is an additional stochastic hypothesis. This pathwise application
allows horizon zero as well as positive horizons; the probability
application uses the original strictly positive horizon.
-/

import Transformer.Homogenized.Section7_MeanLogPathError

open Set MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The genuine sphere martingale problem at its initial time is satisfied
by every fixed unit configuration, with its actual original fields.
Source: arXiv:2604.01978v1, Appendix `def:ito.formula`, at horizon zero. -/
theorem isSphereMartingaleSolution_zero_horizon {d n : ℕ} (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m)
    (K : Idx n → EucSpace d) (hK : ∀ i, ‖K i‖ = 1) :
    IsSphereMartingaleSolution ρ B G P ℱ 0 (fun _ _ => K) := by
  refine ⟨fun t ht ω i => hK i, fun t ht => stronglyMeasurable_const,
    fun ω => continuousOn_const, ?_⟩
  intro φ _
  let M := fun t : ℝ => fun _ : Ω => φ K - φ K - ∫ s in (0 : ℝ)..t, sphGenerator ρ B G φ K
  have hz (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 0) : M t = fun _ => (0 : ℝ) := by
    have ht0 : t = 0 := le_antisymm ht.2 ht.1
    funext ω
    simp only [M, ht0, sub_self, intervalIntegral.integral_same]
  change IsMartingaleOn 0 ℱ P M
  refine ⟨fun t ht => ?_, fun t ht => ?_, fun s hs t ht hst => ?_⟩
  · rw [hz t ht]
    exact stronglyMeasurable_const
  · rw [hz t ht]
    exact integrable_const 0
  · rw [hz t ht, hz s hs]
    exact (isMartingaleOn_const 0 ℱ P 0).2.2 s hs t ht hst

/-- The original positive pairwise lower bound makes the deterministic initial
mean positive on the genuine sphere solution.
Source: arXiv:2604.01978v1, §7, the initial hemisphere condition in `thm:clustering_small_beta`. -/
theorem meanOverlap_initial_pos_of_sphere_solution {d n : ℕ} (hn : 0 < n)
    (ρ : Measure (HeadParam d)) (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 ≤ T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ B G P ℱ T X)
    (x₀ : Idx n → EucSpace d) (hinit : ∀ ω, X 0 ω = x₀) (γ : ℝ) (hγ : γ ∈ Ioo (0 : ℝ) 1)
    (hpair : ∀ i j : Idx n, i ≠ j → γ ≤ inner (𝕜 := ℝ) (x₀ i) (x₀ j)) (ω : Ω) :
    0 < meanOverlap x₀ := by
  have hunit : ∀ i, ‖x₀ i‖ = 1 := by
    rw [← hinit ω]
    exact hMP.1 0 ⟨le_rfl, hT⟩ ω
  exact hγ.1.trans_le (le_meanOverlap_of_pairwise_lower hn x₀ hunit γ hγ.2.le hpair)

/-- The original normalized Gaussian solution satisfies the source mean error
on its actual proved logarithmic-noise event.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound` and the initial
hemisphere assumption of `thm:clustering_small_beta`. -/
theorem gaussian_mean_clustering_on_log_noise_event {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ : Filtration ℝ m)
    (T δ : ℝ) (hT : 0 ≤ T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (x₀ : Idx n → EucSpace d) (hinit : ∀ ω, X 0 ω = x₀)
    (γ : ℝ) (hγ : γ ∈ Ioo (0 : ℝ) 1)
    (hpair : ∀ i j : Idx n, i ≠ j → γ ≤ inner (𝕜 := ℝ) (x₀ i) (x₀ j))
    (u : ℝ → ℝ) (huinit : u 0 = meanOverlap x₀)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (ω : Ω) (hnoise : ∀ t ∈ Icc (0 : ℝ) T,
      |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X
        (logMeanOverlap (meanLogRegularization d β σA) x₀) t ω| ≤ meanLogNoiseRadius d T δ) :
    ∀ t ∈ Icc (0 : ℝ) T, |meanOverlap (X t ω) - u t| ≤ 4096 * Real.exp (4096 * T) *
      max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2) +
      4096 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ)) := by
  have ha := meanLogRegularization_pos (d := d) (by omega) β σA
  have hpos : 0 < u 0 := by
    rw [huinit]
    exact meanOverlap_initial_pos_of_sphere_solution hn ρ (bField β ρ) (noiseField β 1 1 ρ)
      P ℱ T hT X hMP x₀ hinit γ hγ hpair ω
  let D := fun t : ℝ => sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
    (logMeanOverlap (meanLogRegularization d β σA)) (X t ω)
  apply mean_log_path_error_le_source_scale d β σA T δ (by omega) hT hsmall
    (fun t => meanOverlap (X t ω)) D u
  · exact fun t ht => meanOverlap_mem_Icc (X t ω) (hMP.1 t ht ω)
  · exact (continuousOn_sphGenerator_logMeanOverlap_gaussian hd ha hn β σV σA ρ hρ hσ).comp'
      (hMP.2.2.1 ω) (fun t ht => hMP.1 t ht ω)
  · intro t ht
    exact sphGenerator_logMeanOverlap_mem_Icc hd ha hn β hβ hρ hσ
      (temperature_error_le_meanLogRegularization d β σA) (X t ω) (hMP.1 t ⟨ht.1, ht.2.le⟩ ω)
  · rw [hinit ω, huinit]
  · exact hpos
  · exact hu
  · intro t ht
    rw [hinit ω]
    exact hnoise t ht

/-- Positive normalized original Gaussian variances, a positive initial mean,
and the actual full sphere martingale problem are compatible at initial time.
The witness also realizes the genuine logarithmic noise event and logistic reference. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)) (x₀ : Idx 2 → EucSpace 2) (u : ℝ → ℝ),
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ meanLogRegularization 2 0 1 ≤ 1 ∧
    (0 : ℝ) ≤ 0 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧ (∀ i, ‖x₀ i‖ = 1) ∧
    (∀ i j : Idx 2, i ≠ j → (1 / 2 : ℝ) ≤ inner (𝕜 := ℝ) (x₀ i) (x₀ j)) ∧
    IsSphereMartingaleSolution (gaussHeadLaw 2 (stdSigmaV 2) 1)
      (bField 0 (gaussHeadLaw 2 (stdSigmaV 2) 1)) (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1))
      (Measure.dirac ()) ℱ 0 X ∧ (∀ ω, X 0 ω = x₀) ∧ u 0 = meanOverlap x₀ ∧
    (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t) ∧
    (∀ t ∈ Icc (0 : ℝ) 0,
      |logMeanNoiseResidual (meanLogRegularization 2 0 1) 0 (gaussHeadLaw 2 (stdSigmaV 2) 1) X
        (logMeanOverlap (meanLogRegularization 2 0 1) x₀) t ()| ≤ meanLogNoiseRadius 2 0 (1 / 2)) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let K : Idx 2 → EucSpace 2 := fun _ => v
  have hv : ‖v‖ = 1 := by simp [v]
  have hK : ∀ i, ‖K i‖ = 1 := fun i => hv
  refine ⟨ℱ, fun _ _ => K, K, fun _ => 1, le_rfl, by decide, by norm_num,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, ?_, le_rfl, by norm_num, hK, ?_,
    isSphereMartingaleSolution_zero_horizon _ _ _ (Measure.dirac ()) ℱ K hK,
    fun ω => rfl, (meanOverlap_const (n := 1) v hv).symm, ?_, ?_⟩
  · norm_num [meanLogRegularization]
  · intro i j hij
    change (1 / 2 : ℝ) ≤ inner (𝕜 := ℝ) v v
    rw [real_inner_self_eq_norm_sq, hv]
    norm_num
  · intro t ht
    simpa only [sub_self, mul_zero] using (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt (s := Ici 0)
  · intro t ht
    have ht0 : t = 0 := le_antisymm ht.2 ht.1
    rw [ht0, logMeanNoiseResidual_zero _ _ _ _ _ (fun ω => rfl)]
    simpa only [abs_zero] using meanLogNoiseRadius_nonneg 2 0 (1 / 2)

end Transformer.Homogenized
