/-
# Original logarithmic mean noise and its signed maximal tail

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the exponential maximal argument following the mean bracket estimate.
The original noise is tested after taking its fixed regularized logarithm.
The actual Gaussian generator and the full sphere martingale problem
give its signed maximal tail, with variance coefficient `4/d`.

The initial logarithm is deterministic. Its actual exponential
expectation cancels in the maximal quotient. No tail bound or
selected exponential martingale is an additional hypothesis.
-/

import Transformer.Homogenized.Section7_GaussianMeanLogVille

open Set MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The genuine compensated original logarithmic mean observable.
Source: arXiv:2604.01978v1, §7, the mean decomposition `eq:m_decomp_clean`,
tested through the fixed regularized logarithm by Appendix `def:ito.formula`. -/
noncomputable def logMeanNoiseResidual {d n : ℕ} (a β : ℝ) (ρ : Measure (HeadParam d)) {Ω : Type*}
    (X : ℝ → Ω → (Idx n → EucSpace d)) (γ t : ℝ) (ω : Ω) : ℝ :=
  logMeanOverlap a (X t ω) - γ - ∫ s in (0 : ℝ)..t,
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) (X s ω)

/-- The actual compensated logarithmic noise starts at zero.
Source: arXiv:2604.01978v1, §7, the original mean decomposition. -/
theorem logMeanNoiseResidual_zero {d n : ℕ} (a β : ℝ) (ρ : Measure (HeadParam d)) {Ω : Type*}
    (X : ℝ → Ω → (Idx n → EucSpace d)) (γ : ℝ)
    (hinit : ∀ ω, logMeanOverlap a (X 0 ω) = γ) (ω : Ω) :
    logMeanNoiseResidual a β ρ X γ 0 ω = 0 := by
  simp only [logMeanNoiseResidual, hinit ω, sub_self, intervalIntegral.integral_same]

/-- The original logarithm is its initial value plus the actual drift
integral and compensated noise. Source: arXiv:2604.01978v1, §7,
`eq:m_decomp_clean`, after the regularized logarithmic test. -/
theorem logMeanNoiseResidual_reconstruct {d n : ℕ} (a β : ℝ) (ρ : Measure (HeadParam d)) {Ω : Type*}
    (X : ℝ → Ω → (Idx n → EucSpace d)) (γ t : ℝ) (ω : Ω) :
    logMeanOverlap a (X t ω) = γ +
      (∫ s in (0 : ℝ)..t, sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) (X s ω)) +
      logMeanNoiseResidual a β ρ X γ t ω := by
  dsimp only [logMeanNoiseResidual]
  ring

/-- The original logarithmic mean noise obeys its signed Chernoff maximal bound.
Source: arXiv:2604.01978v1, §7, the maximal argument after the mean bracket,
with coefficient `4/d` proved for the actual regularized logarithmic test. -/
theorem gaussian_logMean_signed_chernoff {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (c β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (γ : ℝ) (hinit : ∀ ω, logMeanOverlap (meanLogRegularization d β σA) (X 0 ω) = γ) (r : ℝ) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T,
      c * r < c * logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω} ≤
      ENNReal.ofReal (Real.exp (-c * r + c ^ 2 / 2 * (4 / (d : ℝ)) * T)) := by
  let a := meanLogRegularization d β σA
  let D := fun t : ℝ => fun ω : Ω => sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
    (logMeanOverlap a) (X t ω)
  let b := Real.exp (c * γ + c * r - c ^ 2 / 2 * (4 / (d : ℝ)) * T)
  have ha : 0 < a := meanLogRegularization_pos (by omega) β σA
  have hmax := gaussian_logMean_exponential_ville hd hn c β hβ σV σA ρ hρ hσ hsmall
    P ℱ T hT X hMP b (Real.exp_pos _)
  have hD (ω : Ω) : ContinuousOn (fun t => D t ω) (Icc (0 : ℝ) T) :=
    (continuousOn_sphGenerator_logMeanOverlap_gaussian hd ha hn β σV σA ρ hρ hσ).comp'
      (hMP.2.2.1 ω) (fun t ht => hMP.1 t ht ω)
  have hsub : {ω | ∃ t ∈ Icc (0 : ℝ) T, c * r < c * logMeanNoiseResidual a β ρ X γ t ω} ⊆
      {ω | ∃ t ∈ Icc (0 : ℝ) T,
        b < Real.exp (-(∫ s in (0 : ℝ)..t, c * D s ω + c ^ 2 / 2 * (4 / (d : ℝ)))) *
          exponentialLogMeanOverlap c a (X t ω)} := by
    rintro ω ⟨t, ht, hc⟩
    refine ⟨t, ht, ?_⟩
    have hint : IntervalIntegrable (fun s => D s ω) volume 0 t := by
      apply ContinuousOn.intervalIntegrable
      rw [uIcc_of_le ht.1]
      exact (hD ω).mono (fun s hs => ⟨hs.1, hs.2.trans ht.2⟩)
    change Real.exp (c * γ + c * r - c ^ 2 / 2 * (4 / (d : ℝ)) * T) < _
    rw [exp_logMean_compensation_eq c a (4 / (d : ℝ)) t (X t ω) (fun s => D s ω) hint]
    apply Real.exp_lt_exp.mpr
    change c * r < c * (logMeanOverlap a (X t ω) - γ - ∫ s in (0 : ℝ)..t, D s ω) at hc
    have hq : 0 ≤ c ^ 2 / 2 * (4 / (d : ℝ)) := by positivity
    have htime := mul_le_mul_of_nonneg_left ht.2 hq
    nlinarith
  have h := (measure_mono hsub).trans hmax
  have hi : (∫ ω, exponentialLogMeanOverlap c a (X 0 ω) ∂P) = Real.exp (c * γ) := by
    have heq : (fun ω => exponentialLogMeanOverlap c a (X 0 ω)) = (fun _ : Ω => Real.exp (c * γ)) := by
      funext ω
      dsimp only [a]
      simp only [exponentialLogMeanOverlap, hinit ω]
    rw [heq, integral_const, probReal_univ, one_smul]
  rw [hi] at h
  exact h.trans_eq (ofReal_exp_initial_div_exp_compensation c γ r (4 / (d : ℝ)) T)

/-- Either sign of the original logarithmic noise has maximal exponent `-r²d/(8T)`.
Source: arXiv:2604.01978v1, §7, optimization of the exponential maximal bound
with the actual regularized logarithmic variance coefficient `4/d`. -/
theorem gaussian_logMean_signed_maximal {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (γ : ℝ) (hinit : ∀ ω, logMeanOverlap (meanLogRegularization d β σA) (X 0 ω) = γ)
    (r : ℝ) (hr : 0 < r) (ε : ℝ) (hε : ε ^ 2 = 1) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T,
      r < ε * logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω} ≤
      ENNReal.ofReal (Real.exp (-(r ^ 2 * (d : ℝ) / (8 * T)))) := by
  let b := r * (d : ℝ) / (4 * T)
  have hd0 : (0 : ℝ) < d := by exact_mod_cast (show 0 < d by omega)
  have hb : 0 < b := by dsimp [b]; positivity
  have h := gaussian_logMean_signed_chernoff hd hn (ε * b) β hβ σV σA ρ hρ hσ hsmall
    P ℱ T hT X hMP γ hinit (ε * r)
  have hcomp (z : ℝ) : (ε * b) * (ε * r) < (ε * b) * z ↔ r < ε * z := by
    have hlin : (ε * b) * (ε * r) = b * r := by
      calc
        _ = ε ^ 2 * (b * r) := by ring
        _ = _ := by rw [hε, one_mul]
    rw [hlin, show (ε * b) * z = b * (ε * z) by ring, mul_lt_mul_iff_right₀ hb]
  simp_rw [hcomp] at h
  have hlin : -(ε * b) * (ε * r) = -b * r := by
    calc
      _ = -(ε ^ 2) * (b * r) := by ring
      _ = _ := by rw [hε]; ring
  rw [hlin, mul_pow, hε, one_mul] at h
  exact h.trans_eq (congrArg (fun z => ENNReal.ofReal (Real.exp z))
    (logMean_chernoff_optimal_exponent (d : ℝ) T r hd0.ne' hT.ne'))

/-- A positive-variance Gaussian head, genuine frozen antipodal solution,
deterministic initial logarithm, positive horizon and radius, and either
sign jointly satisfy the original signed maximal hypotheses. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)) (γ : ℝ),
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ meanLogRegularization 2 0 1 ≤ 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (-1 : ℝ) ^ 2 = 1 ∧
    IsSphereMartingaleSolution (gaussHeadLaw 2 (stdSigmaV 2) 1)
      (bField 0 (gaussHeadLaw 2 (stdSigmaV 2) 1)) (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1))
      (Measure.dirac ()) ℱ 1 X ∧
    (∀ ω, logMeanOverlap (meanLogRegularization 2 0 1) (X 0 ω) = γ) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let K : Idx 2 → EucSpace 2 := ![v, -v]
  refine ⟨ℱ, fun _ _ => K, logMeanOverlap (meanLogRegularization 2 0 1) K,
    le_rfl, by decide, by norm_num, isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2,
    ?_, one_pos, by norm_num, by norm_num, ?_, fun ω => rfl⟩
  · norm_num [meanLogRegularization]
  · exact isSphereMartingaleSolution_antipodal_zero_beta _ (Measure.dirac ()) ℱ 1 v (by simp [v])

end Transformer.Homogenized
