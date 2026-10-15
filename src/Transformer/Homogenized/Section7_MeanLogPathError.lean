/-
# Pathwise error of the original mean from its actual logarithmic decomposition

ArXiv:2604.01978v1, §7, `eq:m_decomp_clean`,
`eq:logistic_u_smallbeta`, and `eq:smallbeta_main_bound`.
The original mean path remains in `[0,1]`. Its actual logarithmic
compensator has derivative bounded between `1-Z-c-K/Z` and
`1-Z+K/Z`, for `Z=m+a`. Its centered actual logarithmic noise
is bounded by `r` on the original finite horizon.

The proved upper and lower comparisons and every complementary
parameter regime then give `|m-u|≤128 a exp(2T)+16r`.
The reference is the original logistic Cauchy problem with the actual
initial mean. All scalar drift and noise inputs are explicit here;
the Gaussian application derives them from the genuine original fields.
No derivative is assumed for the original mean or its noise path.
-/

import Transformer.Homogenized.Section7_MeanLogErrorEnvelope

open Set MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The actual continuous logarithmic decomposition controls the original mean
with deterministic exponential bias and an unamplified bounded noise term.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`.
This deterministic lemma exposes exactly the original drift and noise bounds
which the subsequent Gaussian application proves from the source solution. -/
theorem mean_log_path_error_bound (T r a c K : ℝ) (hT : 0 ≤ T) (hr : 0 ≤ r) (ha : 0 < a)
    (hc : 0 ≤ c) (hK : 0 ≤ K) (hca : c ≤ 4 * a) (hKa : K ≤ 6 * a)
    (m D u : ℝ → ℝ) (hm : ∀ t ∈ Icc (0 : ℝ) T, m t ∈ Icc (0 : ℝ) 1)
    (hD : ContinuousOn D (Icc (0 : ℝ) T))
    (hDrift : ∀ t ∈ Ico (0 : ℝ) T, D t ∈ Icc
      (1 - (m t + a) - c - K / (m t + a)) (1 - (m t + a) + K / (m t + a)))
    (hinit : u 0 = m 0) (hpos : 0 < u 0)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (hnoise : ∀ t ∈ Icc (0 : ℝ) T,
      |Real.log (m t + a) - Real.log (m 0 + a) - ∫ s in (0 : ℝ)..t, D s| ≤ r) :
    ∀ t ∈ Icc (0 : ℝ) T, |m t - u t| ≤ 128 * a * Real.exp (2 * T) + 16 * r := by
  let Z := fun t => m t + a
  let F := fun t => Real.log (m 0 + a) + ∫ s in (0 : ℝ)..t, D s
  let A := meanLogUpperShift a K
  let v := fun t => u t + A t
  let v' := fun t => u t * (1 - u t) + A t + K
  have hz (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : 0 < Z t := by dsimp [Z]; linarith [(hm t ht).1]
  have hu0 : u 0 ≤ 1 := by rw [hinit]; exact (hm 0 ⟨le_rfl, hT⟩).2
  have hur (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : u t ∈ Ioc (0 : ℝ) 1 :=
    logistic_ode_mem_Ioc u hpos hu0 hu t ht.1
  have hF0 : F 0 = Real.log (m 0 + a) := by simp only [F, intervalIntegral.integral_same, add_zero]
  have hF : ContinuousOn F (Icc (0 : ℝ) T) := by
    have hi : IntegrableOn D (uIcc (0 : ℝ) T) volume := by
      rw [uIcc_of_le hT]
      exact hD.integrableOn_Icc
    have hp : ContinuousOn (fun t => ∫ s in (0 : ℝ)..t, D s) (Icc (0 : ℝ) T) := by
      simpa only [uIcc_of_le hT] using intervalIntegral.continuousOn_primitive_interval hi
    exact hp.const_add _
  have hF' (t : ℝ) (ht : t ∈ Ico (0 : ℝ) T) : HasDerivWithinAt F (D t) (Ici t) t :=
    (hasDerivWithinAt_continuous_timeIntegral D T hD t ht).const_add _
  have hn (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : |Real.log (Z t) - F t| ≤ r := by
    simpa only [Z, F, sub_add_eq_sub_sub] using hnoise t ht
  have hA (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : 0 ≤ A t := meanLogUpperShift_nonneg ha.le hK ht.1
  have hvpos (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : 0 < v t := by
    dsimp only [v]
    linarith [(hur t ht).1, hA t ht]
  have hv : ContinuousOn v (Icc (0 : ℝ) T) :=
    (logistic_ode_continuousOn u hu T).add (continuous_meanLogUpperShift a K).continuousOn
  have hv' (t : ℝ) (ht : t ∈ Ico (0 : ℝ) T) : HasDerivWithinAt v (v' t) (Ici t) t := by
    have hu' : HasDerivWithinAt u (u t * (1 - u t)) (Ici t) t :=
      (hu t ht.1).mono (fun s hs => ht.1.trans hs)
    have hA' : HasDerivWithinAt A (A t + K) (Ici t) t :=
      (hasDerivAt_meanLogUpperShift a K t).hasDerivWithinAt
    have heq : v' t = u t * (1 - u t) + (A t + K) := by dsimp only [v']; ring
    rw [heq]
    exact hu'.add hA'
  have hupper : ∀ t ∈ Icc (0 : ℝ) T, Real.log (Z t) ≤ Real.log (v t) + 2 * r := by
    apply log_mean_upper_barrier T r K hr hK Z F D v v' hz hF hF' hv hvpos hv'
    · rw [hF0]
      dsimp only [v, A]
      rw [meanLogUpperShift_zero, hinit]
    · exact hn
    · exact fun t ht => (hDrift t ht).2
    · intro t ht
      exact logistic_meanLog_shift_supersolution (hur t ⟨ht.1, ht.2.le⟩).1.le
        (hA t ⟨ht.1, ht.2.le⟩) (hvpos t ⟨ht.1, ht.2.le⟩)
  have hlower : 2 * r + (c + 2 * K / u 0) * T < Real.log 2 →
      ∀ t ∈ Icc (0 : ℝ) T, Real.log (u t) - Real.log (Z t) ≤ 2 * r + (c + 2 * K / u 0) * t := by
    intro hs
    apply log_mean_lower_barrier T r c K (u 0) hr hc hK hpos Z F D u hz hF hF'
      (continuousOn_log_logistic_ode u hpos hu0 hu T)
    · intro t ht
      exact (hasDerivWithinAt_log_logistic_ode u hpos hu0 hu t ht.1).mono (fun s hs => ht.1.trans hs)
    · exact fun t ht => logistic_ode_initial_le u hpos hu0 hu t ht.1
    · rw [hF0, ← hinit]
      exact (Real.log_le_log_iff hpos (by linarith : 0 < u 0 + a)).mpr (by linarith)
    · exact hn
    · exact fun t ht => (hDrift t ht).1
    · exact hs
  apply mean_log_original_error_uniform T r a c K (u 0) hT hr ha hc hK hca hKa hpos m u A hur hm
  · intro t ht
    exact (logistic_ode_exp_upper u hpos hu t ht.1).trans
      (mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ht.2) hpos.le)
  · intro t ht
    refine ⟨hA t ht, (meanLogUpperShift_le_seven hK hKa t).trans ?_⟩
    exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ht.2) (by positivity : 0 ≤ 7 * a)
  · exact hupper
  · exact hlower

/-- The actual canonical logarithmic drift perturbation from the proved Gaussian bounds.
Source: arXiv:2604.01978v1, §7, the drift comparison for `eq:smallbeta_main_bound`. -/
noncomputable def meanLogPerturbation (d : ℕ) (β : ℝ) (σA : ℝ≥0) : ℝ :=
  meanLogRegularization d β σA + meanLogRegularization d β σA ^ 2 +
    80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4

/-- The actual canonical logarithmic drift and confidence scales give
the source's original mean error with universal constant 4096.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`.
All scalar path inputs remain explicit for the original Gaussian application. -/
theorem mean_log_path_error_le_source_scale (d : ℕ) (β : ℝ) (σA : ℝ≥0) (T δ : ℝ)
    (hd : 0 < d) (hT : 0 ≤ T) (hsmall : meanLogRegularization d β σA ≤ 1)
    (m D u : ℝ → ℝ) (hm : ∀ t ∈ Icc (0 : ℝ) T, m t ∈ Icc (0 : ℝ) 1)
    (hD : ContinuousOn D (Icc (0 : ℝ) T))
    (hDrift : ∀ t ∈ Ico (0 : ℝ) T, D t ∈ Icc
      (1 - (m t + meanLogRegularization d β σA) - 4 / (d : ℝ) -
        meanLogPerturbation d β σA / (m t + meanLogRegularization d β σA))
      (1 - (m t + meanLogRegularization d β σA) +
        meanLogPerturbation d β σA / (m t + meanLogRegularization d β σA)))
    (hinit : u 0 = m 0) (hpos : 0 < u 0)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (hnoise : ∀ t ∈ Icc (0 : ℝ) T,
      |Real.log (m t + meanLogRegularization d β σA) - Real.log (m 0 + meanLogRegularization d β σA) -
        ∫ s in (0 : ℝ)..t, D s| ≤ meanLogNoiseRadius d T δ) :
    ∀ t ∈ Icc (0 : ℝ) T, |m t - u t| ≤ 4096 * Real.exp (4096 * T) *
      max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2) +
      4096 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ)) := by
  have ha := meanLogRegularization_pos hd β σA
  have hK : 0 ≤ meanLogPerturbation d β σA := by dsimp [meanLogPerturbation]; positivity
  have hca : 4 / (d : ℝ) ≤ 4 * meanLogRegularization d β σA := by
    have h := mul_le_mul_of_nonneg_left (one_div_dim_le_meanLogRegularization d β σA) (by norm_num : (0 : ℝ) ≤ 4)
    simpa only [div_eq_mul_inv, one_mul] using h
  have h := mean_log_path_error_bound T (meanLogNoiseRadius d T δ) (meanLogRegularization d β σA)
    (4 / (d : ℝ)) (meanLogPerturbation d β σA) hT (meanLogNoiseRadius_nonneg d T δ) ha
    (by positivity) hK hca (meanLogRegularization_perturbation_le_six d β σA hsmall)
    m D u hm hD hDrift hinit hpos hu hnoise
  exact fun t ht => (h t ht).trans (mean_log_error_le_source_scale d β σA T δ hT)

/-- A genuine positive original equilibrium with nonzero regularization and
an actual zero drift primitive realizes every deterministic pathwise input. -/
example : ∃ m D u : ℝ → ℝ,
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) ≤ 0 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 4 * (1 / 4) ∧ (1 / 2 : ℝ) ≤ 6 * (1 / 4) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, m t ∈ Icc (0 : ℝ) 1) ∧ ContinuousOn D (Icc (0 : ℝ) 1) ∧
    (∀ t ∈ Ico (0 : ℝ) 1, D t ∈ Icc
      (1 - (m t + 1 / 4) - 0 - (1 / 2) / (m t + 1 / 4))
      (1 - (m t + 1 / 4) + (1 / 2) / (m t + 1 / 4))) ∧
    u 0 = m 0 ∧ 0 < u 0 ∧
    (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t) ∧
    (∀ t ∈ Icc (0 : ℝ) 1,
      |Real.log (m t + 1 / 4) - Real.log (m 0 + 1 / 4) - ∫ s in (0 : ℝ)..t, D s| ≤ 0) := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => 1,
    zero_le_one, le_rfl, by norm_num, le_rfl, by norm_num, by norm_num, by norm_num,
    fun t ht => by norm_num, continuousOn_const, fun t ht => by norm_num, rfl, one_pos, ?_, ?_⟩
  · intro t ht
    simpa only [sub_self, mul_zero] using (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt (s := Ici 0)
  · intro t ht
    simp only [intervalIntegral.integral_zero, sub_self, abs_zero, le_refl]

/-- The canonical original dimension, regularization, perturbation, and positive
logistic equilibrium jointly realize every source-scale pathwise hypothesis. -/
example : ∀ t ∈ Icc (0 : ℝ) 1, |(1 : ℝ) - 1| ≤
    4096 * Real.exp (4096 * 1) * max (1 / (2 : ℝ)) ((2 : ℝ) ^ 2 * (1 : ℝ) ^ 4 * 0 ^ 2) +
      4096 * Real.sqrt (1 / (2 : ℝ) * Real.log (4 / (1 / 2 : ℝ))) := by
  apply mean_log_path_error_le_source_scale 2 0 1 1 (1 / 2) (by decide) zero_le_one
    (by norm_num [meanLogRegularization]) (fun _ => 1) (fun _ => 0) (fun _ => 1)
  · exact fun t ht => by norm_num
  · exact continuousOn_const
  · exact fun t ht => by norm_num [meanLogPerturbation, meanLogRegularization]
  · rfl
  · exact one_pos
  · intro t ht
    simpa only [sub_self, mul_zero] using (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt (s := Ici 0)
  · intro t ht
    simpa only [intervalIntegral.integral_zero, sub_self, sub_zero, abs_zero] using
      meanLogNoiseRadius_nonneg 2 1 (1 / 2)

end Transformer.Homogenized
