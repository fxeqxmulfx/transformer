module

public import Transformer.Modes.Section3_ThirdPowerBounds
public import Transformer.Modes.Section3_DerivativeTail
public import Transformer.Modes.Section3_CubicTailRates

/-!
# Integrable third derivative of the actual normalized-sum characteristic function

The large-frequency product rule in arXiv:2412.09080v3, §5.4
`eq:big-z-exp`, retains `n-3` undifferentiated characteristic factors
after three derivatives. The coefficient below contains the actual
first, second, and third norm moments. A nonsharp polynomial in `n`
is enough, since the strict exterior gap later damps every polynomial.

Keeping an integrable characteristic power gives both whole-space
integrability of the third derivative and its quantitative exterior
norm integral. The literal frequency substitution `1 / sqrt n` has
two-dimensional Jacobian `n`, yielding the explicit fourth-power
prefactor in the tail estimate. The normalization is that of the
actual `scaledSum`, independently of the source's block decomposition.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual sum's third derivative retains the characteristic power `n-3` with finite-moment coefficients.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, derivative order
three. The coefficient is deliberately bounded by a cubic polynomial
in `n`, preserving the damping needed for the exterior integral. -/
theorem norm_iteratedFDeriv_three_characteristic_scaledSum_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    {n : ℕ} (hn : 3 ≤ n) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 3
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖ ≤
      (n : ℝ) ^ 3 * ((2 * ∫ z, ‖z‖ ∂μ) ^ 3 +
        3 * (2 * ∫ z, ‖z‖ ∂μ) * (4 * ∫ z, ‖z‖ ^ 2 ∂μ) +
        8 * ∫ z, ‖z‖ ^ 3 ∂μ) *
          ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ (n - 3) := by
  let c : ℝ := (Real.sqrt n)⁻¹
  let η : ℝ × ℝ := c • ξ
  let A : ℝ := 2 * ∫ z, ‖z‖ ∂μ
  let B : ℝ := 4 * ∫ z, ‖z‖ ^ 2 ∂μ
  let C : ℝ := 8 * ∫ z, ‖z‖ ^ 3 ∂μ
  have hM1 : 0 ≤ ∫ z, ‖z‖ ∂μ := integral_nonneg fun z => norm_nonneg z
  have hM2 : 0 ≤ ∫ z, ‖z‖ ^ 2 ∂μ := integral_nonneg fun z => sq_nonneg _
  have hM3 : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hA : 0 ≤ A := by dsimp [A]; positivity
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hnR : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr (by linarith)
  have hc : 0 ≤ c := inv_nonneg.mpr (Real.sqrt_nonneg _)
  have hc1 : c ≤ 1 := (inv_le_one₀ hs).mpr (Real.one_le_sqrt.mpr hnR)
  have hc3 : c ^ 3 ≤ 1 := by simpa using pow_le_pow_left₀ hc hc1 3
  have hN2 : (n : ℝ) ^ 2 ≤ (n : ℝ) ^ 3 := by
    nlinarith [mul_nonneg (sq_nonneg (n : ℝ)) (sub_nonneg.mpr hnR)]
  have hN1 : (n : ℝ) ≤ (n : ℝ) ^ 3 := by nlinarith [hN2]
  have hD1 : ‖fderiv ℝ (characteristic2 μ) η‖ ≤ A :=
    norm_fderiv_characteristic2_le μ (hmom.mono_exponent (by norm_num)) η
  have hD2 : ‖iteratedFDeriv ℝ 2 (characteristic2 μ) η‖ ≤ B := by
    convert norm_iteratedFDeriv_characteristic2_le μ (k := 2)
      (hmom.mono_exponent (by norm_num)) η using 1
    norm_num [B]
  have hD3 : ‖iteratedFDeriv ℝ 3 (characteristic2 μ) η‖ ≤ C := by
    convert norm_iteratedFDeriv_characteristic2_le μ hmom η using 1
    norm_num [C]
  have hf := contDiff_characteristic2 μ (k := 3) hmom
  have hP := norm_iteratedFDeriv_three_complex_power_le_of_norm_le_one
    (characteristic2 μ) n hf η (norm_characteristic2_le_one μ η)
  have he : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun ζ => characteristic2 μ (c • ζ) ^ n := by
    funext ζ
    exact characteristic_scaledSum μ n ζ
  rw [he, norm_iteratedFDeriv_complex_comp_smul _ c 3 (hf.pow n) ξ,
    abs_of_nonneg hc]
  calc
    _ ≤ c ^ 3 * (((n : ℝ) ^ 3 * ‖fderiv ℝ (characteristic2 μ) η‖ ^ 3 +
        3 * (n : ℝ) ^ 2 * ‖fderiv ℝ (characteristic2 μ) η‖ *
          ‖iteratedFDeriv ℝ 2 (characteristic2 μ) η‖ +
        (n : ℝ) * ‖iteratedFDeriv ℝ 3 (characteristic2 μ) η‖) *
          ‖characteristic2 μ η‖ ^ (n - 3)) :=
      mul_le_mul_of_nonneg_left hP (by positivity)
    _ ≤ 1 * (((n : ℝ) ^ 3 * A ^ 3 + 3 * (n : ℝ) ^ 3 * A * B +
        (n : ℝ) ^ 3 * C) * ‖characteristic2 μ η‖ ^ (n - 3)) := by gcongr
    _ = _ := by dsimp [A, B, C, η, c]; ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ 3 ≤ (4 : ℕ) :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), by omega⟩

/-- A retained integrable characteristic power controls the actual third derivative over the whole space and its exterior.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, with the
literal two-dimensional frequency Jacobian and finite third moment. -/
theorem integrable_iteratedFDeriv_three_characteristic_scaledSum_and_tail
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    {ν n : ℕ} (hν : 1 ≤ ν) (hn : ν + 3 ≤ n)
    (hI : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν))
    {a ε : ℝ} (hε : 0 ≤ ε)
    (hgap : ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ ε) :
    Integrable (iteratedFDeriv ℝ 3
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) ∧
    (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
      ‖iteratedFDeriv ℝ 3
        (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖) ≤
      (n : ℝ) ^ 4 * ((2 * ∫ z, ‖z‖ ∂μ) ^ 3 +
        3 * (2 * ∫ z, ‖z‖ ∂μ) * (4 * ∫ z, ‖z‖ ^ 2 ∂μ) +
        8 * ∫ z, ‖z‖ ^ 3 ∂μ) * ε ^ (n - 3 - ν) *
          ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν := by
  have hM1 : 0 ≤ ∫ z, ‖z‖ ∂μ := integral_nonneg fun z => norm_nonneg z
  have hM2 : 0 ≤ ∫ z, ‖z‖ ^ 2 ∂μ := integral_nonneg fun z => sq_nonneg _
  have hM3 : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hT := integrable_power_majorant_and_tail μ (j := 3) hν hn hI
    (iteratedFDeriv ℝ 3
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))))
    (ContDiff.continuous_iteratedFDeriv (by norm_num)
      (contDiff_characteristic_scaledSum_of_memLp μ 3 (by simpa using hmom) n))
    (by positivity) (norm_iteratedFDeriv_three_characteristic_scaledSum_le μ hmom (by omega))
    hε hgap
  exact ⟨hT.1, hT.2.trans_eq (by ring)⟩

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ 1 ≤ (1 : ℕ) ∧
    1 + 3 ≤ (4 : ℕ) ∧ Integrable (fun ξ => ‖characteristic2 stdGauss2 ξ‖ ^ 1) ∧
    (0 : ℝ) ≤ 1 ∧ (∀ ξ : ℝ × ℝ, 0 ≤ ‖ξ‖ → ‖characteristic2 stdGauss2 ξ‖ ≤ 1) :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), by omega, by omega,
    by simpa using integrable_characteristic2_stdGauss2.norm,
    zero_le_one, fun ξ _ => norm_characteristic2_le_one stdGauss2 ξ⟩

/-- The source's integrable-power condition eventually makes the actual third sum derivative integrable.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:higher-error-goal`. Integrability is derived rather than assumed. -/
theorem eventually_integrable_iteratedFDeriv_three_characteristic_scaledSum
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop, Integrable (iteratedFDeriv ℝ 3
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) := by
  obtain ⟨ν, hν, hI⟩ := hcf
  filter_upwards [eventually_ge_atTop (ν + 3)] with n hn
  exact (integrable_iteratedFDeriv_three_characteristic_scaledSum_and_tail μ hmom hν hn hI
    zero_le_one (a := 0) (fun ξ _ => norm_characteristic2_le_one μ ξ)).1

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

/-- The actual third characteristic derivative has the exterior inverse-n rate needed for the cubic weight.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:big-z-exp`. Its polynomial coefficient is dominated by the
strict modulus gap, while the integrable power remains in the bound. -/
theorem eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_tail_rate_inv_nat
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hcf : HasIntegrableCharFun μ) {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      Integrable (iteratedFDeriv ℝ 3
        (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) ∧
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ 3
          (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖) ≤
            C * (n : ℝ)⁻¹ := by
  obtain ⟨ε, hε0, hε1, hgap⟩ := characteristic2_gap_away_zero μ hcf ha
  obtain ⟨ν, hν, hI⟩ := hcf
  let K : ℝ := (2 * ∫ z, ‖z‖ ∂μ) ^ 3 +
    3 * (2 * ∫ z, ‖z‖ ∂μ) * (4 * ∫ z, ‖z‖ ^ 2 ∂μ) + 8 * ∫ z, ‖z‖ ^ 3 ∂μ
  let J : ℝ := ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν
  have hM1 : 0 ≤ ∫ z, ‖z‖ ∂μ := integral_nonneg fun z => norm_nonneg z
  have hM2 : 0 ≤ ∫ z, ‖z‖ ^ 2 ∂μ := integral_nonneg fun z => sq_nonneg _
  have hM3 : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hK : 0 ≤ K := by dsimp [K]; positivity
  have hJ : 0 ≤ J := integral_nonneg fun ξ => by positivity
  refine ⟨(ε ^ (ν + 3))⁻¹ * K * J, by positivity, ?_⟩
  filter_upwards [eventually_nat_pow_mul_geometric_le_inv_pow hε0 hε1 (ν + 3) 4 1,
    eventually_ge_atTop (ν + 3)] with n hrate hn
  simp only [pow_one] at hrate
  have hT := integrable_iteratedFDeriv_three_characteristic_scaledSum_and_tail
    μ hmom hν hn hI hε0.le hgap
  refine ⟨hT.1, ?_⟩
  have hsub : n - 3 - ν = n - (ν + 3) := by omega
  calc
    _ ≤ (n : ℝ) ^ 4 * K * ε ^ (n - 3 - ν) * J := hT.2
    _ = ((n : ℝ) ^ 4 * ε ^ (n - (ν + 3))) * K * J := by rw [hsub]; ring
    _ ≤ ((ε ^ (ν + 3))⁻¹ * (n : ℝ)⁻¹) * K * J := by gcongr
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2, one_pos⟩

end Transformer.Modes
