module

public import Transformer.Homogenized.Section7_MeanNoiseContinuity
public import Transformer.Homogenized.Section7_GaussianOverlapVille

/-!
# Actual logarithmic mean exponential maximal inequality

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the maximal estimate after the original mean bracket calculation,
and Appendix, `def:ito.formula`. The original Gaussian mean is
tested through its genuine fixed regularized logarithm.

Its actual exponential generator is bounded by the actual exponential
test times `c Lψ + c²(4/d)/2`. The full adapted continuous sphere
martingale problem supplies this test's compensated martingale identity.
The proved continuity and uniform bounds supply all hypotheses of
the integrating-factor maximal theorem.

Consequently the actual process `exp(-∫(c Lψ+c²(4/d)/2)) exp(cψ)`
has its continuous maximal measure bound. No weighted martingale,
quadratic-variation process, or maximal probability is assumed.
Both signs of `c` are retained, and its deterministic initial
exponential is kept for the subsequent original-noise cancellation.
-/

@[expose] public section

open Real Set MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The genuine logarithmic compensation coefficient has a uniform
original-unit-configuration bound. Source: arXiv:2604.01978v1,
§7, the bounded coefficient in the exponential maximal argument. -/
theorem abs_logMean_compensation_coefficient_le {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (c β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |c * sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap (meanLogRegularization d β σA)) X +
        c ^ 2 / 2 * (4 / (d : ℝ))| ≤ 9 * |c| + c ^ 2 / 2 * (4 / (d : ℝ)) := by
  have h := abs_sphGenerator_logMeanOverlap_regularization_le_nine hd hn β hβ hρ hσ hsmall X hX
  apply (abs_add_le _ _).trans
  rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ c ^ 2 / 2 * (4 / (d : ℝ)))]
  exact (add_le_add (mul_le_mul_of_nonneg_left h (abs_nonneg c)) le_rfl).trans_eq (by ring)

/-- The actual exponential integral is exactly the compensated
original logarithmic observable's exponential.
Source: arXiv:2604.01978v1, §7, the original exponential maximal argument. -/
theorem exp_logMean_compensation_eq {d n : ℕ} (c a q t : ℝ) (X : Idx n → EucSpace d)
    (D : ℝ → ℝ) (hD : IntervalIntegrable D volume 0 t) :
    Real.exp (-(∫ s in (0 : ℝ)..t, c * D s + c ^ 2 / 2 * q)) * exponentialLogMeanOverlap c a X =
      Real.exp (c * (logMeanOverlap a X - ∫ s in (0 : ℝ)..t, D s) - c ^ 2 / 2 * q * t) :=
  exp_overlap_compensation_eq c q (logMeanOverlap a X) t D hD

/-- Keeping the original initial logarithm explicit separates the
genuine centered mean noise inside the actual exponential.
Source: arXiv:2604.01978v1, §7, the exponential maximal argument
applied to the compensated original mean observable. -/
theorem exp_logMean_centered_compensation_eq {d n : ℕ} (c a q t : ℝ)
    (X X₀ : Idx n → EucSpace d) (D : ℝ → ℝ) (hD : IntervalIntegrable D volume 0 t) :
    Real.exp (-(∫ s in (0 : ℝ)..t, c * D s + c ^ 2 / 2 * q)) * exponentialLogMeanOverlap c a X =
      Real.exp (c * logMeanOverlap a X₀ +
        c * (logMeanOverlap a X - logMeanOverlap a X₀ - ∫ s in (0 : ℝ)..t, D s) - c ^ 2 / 2 * q * t) := by
  rw [exp_logMean_compensation_eq c a q t X D hD]
  congr 1
  ring

/-- Optimizing the genuine Gaussian logarithmic variance coefficient
`4/d` gives the original mean-noise Chernoff exponent.
Source: arXiv:2604.01978v1, §7, the exponent `-r²d/(8T)` after
the original mean bracket estimate. -/
theorem logMean_chernoff_optimal_exponent (d T r : ℝ) (hd : d ≠ 0) (hT : T ≠ 0) :
    -(r * d / (4 * T)) * r + (r * d / (4 * T)) ^ 2 / 2 * (4 / d) * T =
      -(r ^ 2 * d / (8 * T)) := by
  field_simp
  ring

/-- **The actual logarithmic mean exponential maximal bound follows
from the full original adapted continuous sphere martingale problem.**
Source: arXiv:2604.01978v1, §7, the original mean-noise maximal argument
and Appendix, `def:ito.formula`, at the adopted value variance `1/d`. -/
theorem gaussian_logMean_exponential_ville {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (c β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (r : ℝ) (hr : 0 < r) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T,
      r < Real.exp (-(∫ s in (0 : ℝ)..t,
        c * sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
          (logMeanOverlap (meanLogRegularization d β σA)) (X s ω) + c ^ 2 / 2 * (4 / (d : ℝ)))) *
        exponentialLogMeanOverlap c (meanLogRegularization d β σA) (X t ω)} ≤
      ENNReal.ofReal (∫ ω, exponentialLogMeanOverlap c (meanLogRegularization d β σA) (X 0 ω) ∂P) /
        ENNReal.ofReal r := by
  let a := meanLogRegularization d β σA
  have ha : 0 < a := meanLogRegularization_pos (by omega) β σA
  let k := fun t : ℝ => fun ω : Ω => c * sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
    (logMeanOverlap a) (X t ω) + c ^ 2 / 2 * (4 / (d : ℝ))
  let Y := fun t : ℝ => fun ω : Ω => exponentialLogMeanOverlap c a (X t ω)
  let H := fun t : ℝ => fun ω : Ω => sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
    (exponentialLogMeanOverlap c a) (X t ω)
  let A := 9 * |c| + c ^ 2 / 2 * (4 / (d : ℝ))
  let B := Real.exp (|c| * (|Real.log a| + |Real.log (1 + a)|))
  have hD := continuousOn_sphGenerator_logMeanOverlap_gaussian hd ha hn β σV σA ρ hρ hσ
  have hH := continuousOn_sphGenerator_exponentialLogMeanOverlap_gaussian hd c ha hn β σV σA ρ hρ hσ
  have hφ : Continuous (exponentialLogMeanOverlap (d := d) (n := n) c a) :=
    (contDiff_exponentialLogMeanOverlap c ha).continuous
  have hk (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (k t) := by
    have hDm : StronglyMeasurable[ℱ t] (fun ω => sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
        (logMeanOverlap a) (X t ω)) :=
      stronglyMeasurable_comp_of_continuousOn_range _ _ hD (X t) (hMP.2.1 t ht).measurable (hMP.1 t ht)
    exact (hDm.const_mul c).add_const _
  have hY (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (Y t) :=
    hφ.comp_stronglyMeasurable (hMP.2.1 t ht)
  have hc (ω : Ω) : ContinuousOn (fun t => k t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) T) := by
    have hDt := hD.comp' (hMP.2.2.1 ω) (fun t ht => hMP.1 t ht ω)
    exact ⟨(continuousOn_const.mul hDt).add_const _, hφ.comp_continuousOn (hMP.2.2.1 ω),
      hH.comp' (hMP.2.2.1 ω) (fun t ht => hMP.1 t ht ω)⟩
  apply exp_neg_timeIntegral_mul_ville P ℱ T hT k Y H hk hY hc
    (IsSphereMartingaleSolution.exponentialLogMeanOverlap c ha hMP) A B (B * A)
    (by dsimp [A]; positivity) (by dsimp [A, B]; positivity)
    (fun t ht ω => abs_logMean_compensation_coefficient_le hd hn c β hβ hρ hσ hsmall (X t ω) (hMP.1 t ht ω))
    (fun t ht ω => abs_exponentialLogMeanOverlap_le c ha (X t ω) (hMP.1 t ht ω))
    (fun t ht ω => abs_sphGenerator_exponentialLogMeanOverlap_regularization_le hd c hn β hβ hρ hσ hsmall
      (X t ω) (hMP.1 t ht ω))
    (fun t ht ω => (exponentialLogMeanOverlap_pos c a (X t ω)).le) ?_ r hr
  intro t ht ω
  exact (sphGenerator_exponentialLogMeanOverlap_le c ha hn β hβ hρ hσ
    (temperature_error_le_meanLogRegularization d β σA) (X t ω) (hMP.1 t ht ω)).trans_eq (mul_comm _ _)

/-- Positive original Gaussian variances and genuine antipodal unit
tokens realize the full sphere martingale solution and every maximal
bound hypothesis at positive horizon and threshold. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)),
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ meanLogRegularization 2 0 1 ≤ 1 ∧
    IsSphereMartingaleSolution (gaussHeadLaw 2 (stdSigmaV 2) 1)
      (bField 0 (gaussHeadLaw 2 (stdSigmaV 2) 1)) (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1))
      (Measure.dirac ()) ℱ 1 X := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  refine ⟨ℱ, fun _ _ => ![v, -v], le_rfl, by decide, by norm_num, one_pos, by norm_num,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, ?_, ?_⟩
  · norm_num [meanLogRegularization]
  · exact isSphereMartingaleSolution_antipodal_zero_beta _ (Measure.dirac ()) ℱ 1 v (by simp [v])

end Transformer.Homogenized
