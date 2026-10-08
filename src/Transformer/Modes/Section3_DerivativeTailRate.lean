import Transformer.Modes.Section3_DerivativeTail
import Transformer.Modes.Section3_FrequencyMoments
import Transformer.Modes.Section3_FirstDerivativeFrequency
/-!
# Quantitative exterior rates for characteristic derivatives

The geometric damping in arXiv:2412.09080v3, §5.4 `eq:big-z-exp`,
dominates every fixed natural polynomial in the sample count. Combining
this fact with the proved derivative tail bounds gives the rate
`1 / sqrt n` for the first two normalized-sum Fourier derivatives.

The integrable characteristic power and finite second moment are
retained explicitly. The constants depend on the fixed law and exterior
radius, as in §3 `thm:br`; no uniformity in a varying law is claimed.
Gaussian scaling then transfers the same exterior bounds to the actual
standard Gaussian derivatives, without an additional derivative model.
Their integrability follows from the genuine normalized-sum identity.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Geometric damping dominates any fixed natural polynomial at rate `1 / sqrt n`.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, its fixed-law
polynomial prefactor. The retained integrable-power shift is explicit. -/
theorem eventually_nat_pow_mul_geometric_le_inv_sqrt {ε : ℝ} (hε0 : 0 < ε)
    (hε1 : ε < 1) (ν k : ℕ) :
    ∀ᶠ n : ℕ in atTop, (n : ℝ) ^ k * ε ^ (n - ν) ≤ (ε ^ ν)⁻¹ * (Real.sqrt n)⁻¹ := by
  have hlim := (summable_pow_mul_geometric_of_norm_lt_one (k + 1)
    (show ‖ε‖ < 1 by simpa [Real.norm_eq_abs, abs_of_pos hε0] using hε1)).tendsto_atTop_zero
  have hevent : ∀ᶠ n : ℕ in atTop, (n : ℝ) ^ (k + 1) * ε ^ n < 1 :=
    hlim.eventually (gt_mem_nhds (show (0 : ℝ) < 1 by norm_num))
  filter_upwards [hevent, eventually_ge_atTop (max ν 1)] with n hn hnν
  have hn1 : 1 ≤ n := (le_max_right ν 1).trans hnν
  have hνn : ν ≤ n := (le_max_left ν 1).trans hnν
  have hnR : (1 : ℝ) ≤ n := by exact_mod_cast hn1
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr (by linarith)
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt (by positivity)
  have hsle : Real.sqrt (n : ℝ) ≤ n := by nlinarith [Real.sqrt_nonneg (n : ℝ)]
  have hcoef : (n : ℝ) ^ k * Real.sqrt n * ε ^ n ≤ 1 := by
    calc
      _ ≤ (n : ℝ) ^ (k + 1) * ε ^ n := by
        rw [pow_succ]
        gcongr
      _ ≤ _ := hn.le
  have hp : 0 < ε ^ ν * Real.sqrt (n : ℝ) := by positivity
  have hdiv : (n : ℝ) ^ k * ε ^ (n - ν) ≤ 1 / (ε ^ ν * Real.sqrt (n : ℝ)) := by
    apply (le_div_iff₀ hp).mpr
    have he : (n : ℝ) ^ k * ε ^ (n - ν) * (ε ^ ν * Real.sqrt (n : ℝ)) =
        (n : ℝ) ^ k * Real.sqrt n * ε ^ n := by
      calc
        _ = (n : ℝ) ^ k * Real.sqrt n * (ε ^ (n - ν) * ε ^ ν) := by ring
        _ = _ := by rw [← pow_add, Nat.sub_add_cancel hνn]
    rw [he]
    exact hcoef
  simpa [one_div, mul_inv_rev, mul_comm] using hdiv


example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

/-- The first actual characteristic derivative has an exterior `1 / sqrt n` rate.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, derivative order one.
The same eventual threshold gives whole-space derivative integrability. -/
theorem eventually_integral_fderiv_characteristic_scaledSum_tail_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ)
    (hcf : HasIntegrableCharFun μ) {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      Integrable (fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) ∧
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖) ≤
          C * (Real.sqrt n)⁻¹ := by
  obtain ⟨ε, hε0, hε1, hgap⟩ := characteristic2_gap_away_zero μ hcf ha
  obtain ⟨ν, hν, hI⟩ := hcf
  let A : ℝ := 2 * ∫ z, ‖z‖ ∂μ
  let J : ℝ := ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν
  have hM : 0 ≤ ∫ z, ‖z‖ ∂μ := integral_nonneg fun z => norm_nonneg z
  have hA : 0 ≤ A := by dsimp [A]; positivity
  have hJ : 0 ≤ J := integral_nonneg fun ξ => by positivity
  refine ⟨(ε ^ (ν + 1))⁻¹ * A * J, by positivity, ?_⟩
  filter_upwards [eventually_nat_pow_mul_geometric_le_inv_sqrt hε0 hε1 (ν + 1) 2,
    eventually_ge_atTop (ν + 1)] with n hrate hn
  have hT := integrable_fderiv_characteristic_scaledSum_and_tail μ hmom hν hn hI hε0.le hgap
  refine ⟨hT.1, ?_⟩
  have hnR : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr (by linarith)
  have hc : (Real.sqrt (n : ℝ))⁻¹ ≤ 1 := (inv_le_one₀ hs).mpr (Real.one_le_sqrt.mpr hnR)
  have hsub : n - 1 - ν = n - (ν + 1) := by omega
  calc
    _ ≤ (n : ℝ) ^ 2 * (Real.sqrt n)⁻¹ * A * ε ^ (n - 1 - ν) * J := hT.2
    _ = ((n : ℝ) ^ 2 * ε ^ (n - (ν + 1))) * (Real.sqrt n)⁻¹ * A * J := by rw [hsub]; ring
    _ ≤ ((ε ^ (ν + 1))⁻¹ * (Real.sqrt n)⁻¹) * (Real.sqrt n)⁻¹ * A * J := by gcongr
    _ ≤ ((ε ^ (ν + 1))⁻¹ * (Real.sqrt n)⁻¹) * 1 * A * J := by gcongr
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 ∧
    HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2, one_pos⟩

/-- The second actual characteristic derivative has an exterior `1 / sqrt n` rate.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, derivative order two.
The same eventual threshold gives whole-space derivative integrability. -/
theorem eventually_integral_iteratedFDeriv_two_characteristic_scaledSum_tail_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ)
    (hcf : HasIntegrableCharFun μ) {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      Integrable (iteratedFDeriv ℝ 2
        (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) ∧
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ 2
          (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖) ≤
          C * (Real.sqrt n)⁻¹ := by
  obtain ⟨ε, hε0, hε1, hgap⟩ := characteristic2_gap_away_zero μ hcf ha
  obtain ⟨ν, hν, hI⟩ := hcf
  let B : ℝ := (2 * ∫ z, ‖z‖ ∂μ) ^ 2 + 4 * ∫ z, ‖z‖ ^ 2 ∂μ
  let J : ℝ := ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 2 ∂μ := integral_nonneg fun z => sq_nonneg _
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hJ : 0 ≤ J := integral_nonneg fun ξ => by positivity
  refine ⟨(ε ^ (ν + 2))⁻¹ * B * J, by positivity, ?_⟩
  filter_upwards [eventually_nat_pow_mul_geometric_le_inv_sqrt hε0 hε1 (ν + 2) 2,
    eventually_ge_atTop (ν + 2)] with n hrate hn
  have hT := integrable_iteratedFDeriv_two_characteristic_scaledSum_and_tail μ hmom hν hn hI hε0.le hgap
  refine ⟨hT.1, ?_⟩
  have hsub : n - 2 - ν = n - (ν + 2) := by omega
  calc
    _ ≤ (n : ℝ) ^ 2 * B * ε ^ (n - 2 - ν) * J := hT.2
    _ = ((n : ℝ) ^ 2 * ε ^ (n - (ν + 2))) * B * J := by rw [hsub]; ring
    _ ≤ ((ε ^ (ν + 2))⁻¹ * (Real.sqrt n)⁻¹) * B * J := by gcongr
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 ∧
    HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2, one_pos⟩

/-- The first two actual Gaussian characteristic derivatives are integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal` and `eq:big-phi-poly`.
The normalized-sum Gaussian identity reduces this to the proved power threshold. -/
theorem integrable_characteristic2_stdGauss2_derivatives :
    Integrable (fderiv ℝ (characteristic2 stdGauss2)) ∧
      Integrable (iteratedFDeriv ℝ 2 (characteristic2 stdGauss2)) := by
  have hI := eventually_integrable_characteristic_scaledSum_derivatives stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) hasIntegrableCharFun_stdGauss2
  obtain ⟨n, hDn, hn⟩ := (hI.and (eventually_ge_atTop 1)).exists
  have he : characteristic2 ((Measure.pi fun _ : Fin n => stdGauss2).map (scaledSum n)) =
      characteristic2 stdGauss2 := by
    funext ξ
    rw [characteristic_scaledSum, characteristic2_stdGauss2_scaled_power n hn]
  rw [he] at hDn
  exact ⟨hDn.2.1, hDn.2.2⟩

/-- Both Gaussian derivative tails have the required rate on the growing exterior.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`, derivative orders
one and two. The bound follows from the literal Gaussian scaling identity. -/
theorem eventually_integral_characteristic2_stdGauss2_derivative_tail_rate
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, ‖fderiv ℝ (characteristic2 stdGauss2) ξ‖) +
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖) ≤
        C * (Real.sqrt n)⁻¹ := by
  obtain ⟨C1, hC1, hT1⟩ := eventually_integral_fderiv_characteristic_scaledSum_tail_rate stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) hasIntegrableCharFun_stdGauss2 ha
  obtain ⟨C2, hC2, hT2⟩ := eventually_integral_iteratedFDeriv_two_characteristic_scaledSum_tail_rate stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) hasIntegrableCharFun_stdGauss2 ha
  refine ⟨C1 + C2, by positivity, ?_⟩
  filter_upwards [hT1, hT2, eventually_ge_atTop 1] with n hn1 hn2 hn
  have he : characteristic2 ((Measure.pi fun _ : Fin n => stdGauss2).map (scaledSum n)) =
      characteristic2 stdGauss2 := by
    funext ξ
    rw [characteristic_scaledSum, characteristic2_stdGauss2_scaled_power n hn]
  rw [he] at hn1 hn2
  exact (add_le_add hn1.2 hn2.2).trans_eq (by ring)

example : (0 : ℝ) < 1 := one_pos

end Transformer.Modes
