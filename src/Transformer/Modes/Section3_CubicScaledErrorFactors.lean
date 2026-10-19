module

public import Transformer.Modes.Section3_CubicScaledDerivativeBounds

/-!
# Small derivative profiles for the exact cubic-error factors

The factorization used in arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
has three small factors: the Gaussian comparison, the Gaussian-damped
cubic coefficient, and their corrected difference. We supply common
bounds for every actual derivative through order three, uniformly in
the literal frequency scale `0 ≤ c ≤ 1` on the scaled unit ball.

The Gaussian comparison and the damped cubic coefficient both retain
`c³`. The corrected difference retains `c⁴`. To obtain the latter,
the bare Taylor remainder is combined with `B (1-G)`; the extra term
has `c⁵`, which is at most `c⁴`. Gaussian differentiation remains in
each product bound. The polynomial degrees are deliberately enlarged
to obtain one envelope valid at all four derivative orders.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Every actual small factor in the cubic power error has its correct scale through derivative order three.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
the literal scaled third- and fourth-order one-summand errors. -/
theorem exists_norm_iteratedFDeriv_scaled_cubic_error_factors_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) :
    ∃ D B A : ℝ, 0 ≤ D ∧ 0 ≤ B ∧ 0 ≤ A ∧
      ∀ c : ℝ, 0 ≤ c → c ≤ 1 → ∀ ξ : ℝ × ℝ, ‖c • ξ‖ ≤ 1 → ∀ k : ℕ, k ≤ 3 →
        ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ (c • η) -
          characteristic2 stdGauss2 (c • η)) ξ‖ ≤ D * c ^ 3 * (1 + ‖ξ‖) ^ 3 ∧
        ‖iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 (c • η) *
          cubicCharacteristicCorrection μ (c • η)) ξ‖ ≤ B * c ^ 3 * (1 + ‖ξ‖) ^ 4 ∧
        ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ (c • η) -
          characteristic2 stdGauss2 (c • η) - characteristic2 stdGauss2 (c • η) *
            cubicCharacteristicCorrection μ (c • η)) ξ‖ ≤ A * c ^ 4 * (1 + ‖ξ‖) ^ 5 := by
  let D : ℝ := 1 + 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2)
  let B0 : ℝ := 8 * ∫ z, ‖z‖ ^ 3 ∂μ
  let A0 : ℝ := 5 + 16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2)
  let G : ℝ := 11 + 8 * ∫ z, ‖z‖ ^ 3 ∂stdGauss2
  let H : ℝ := 10 + 8 * ∫ z, ‖z‖ ^ 3 ∂stdGauss2
  have hM3 : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hM4 : 0 ≤ ∫ z, ‖z‖ ^ 4 ∂μ := integral_nonneg fun z => by positivity
  have hG3 : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hG4 : 0 ≤ ∫ z, ‖z‖ ^ 4 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hD : 0 ≤ D := by dsimp [D]; positivity
  have hB0 : 0 ≤ B0 := by dsimp [B0]; positivity
  have hA0 : 0 ≤ A0 := by dsimp [A0]; positivity
  have hG : 0 ≤ G := by dsimp [G]; positivity
  have hH : 0 ≤ H := by dsimp [H]; positivity
  have hmom3 : MemLp id 3 μ := hmom.mono_exponent (by norm_num)
  have hf := contDiff_characteristic2 μ (k := 3) hmom3
  have hg := contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))
  have hb : ContDiff ℝ 3 (cubicCharacteristicCorrection μ) :=
    (contDiff_cubicCharacteristicCorrection_of_memLp μ hmom3).of_le (by simp)
  refine ⟨D, 8 * G * B0, A0 + 8 * B0 * H, hD, by positivity, by positivity, ?_⟩
  intro c hc hc1 ξ hξ k hk
  let t : ℝ := 1 + ‖ξ‖
  have ht : 1 ≤ t := by dsimp [t]; linarith [norm_nonneg ξ]
  have hscale : ContDiff ℝ 3 (fun η : ℝ × ℝ => c • η) := by fun_prop
  have hFC : ContDiff ℝ 3 (fun η => characteristic2 μ (c • η)) := hf.comp hscale
  have hGC : ContDiff ℝ 3 (fun η => characteristic2 stdGauss2 (c • η)) := hg.comp hscale
  have hBC : ContDiff ℝ 3 (fun η => cubicCharacteristicCorrection μ (c • η)) := hb.comp hscale
  have hDiff (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => characteristic2 μ (c • η) -
        characteristic2 stdGauss2 (c • η)) ξ‖ ≤ D * c ^ 3 * t ^ 3 :=
    norm_iteratedFDeriv_comp_smul_le_of_homogeneous_bound _ j 3
      ((hf.sub hg).of_le (by exact_mod_cast hj)) hj c hc ξ hD
      (norm_iteratedFDeriv_characteristic2_sub_gaussian_local_le μ hμ hmom3 (c • ξ) hξ j hj)
  have hCube (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => cubicCharacteristicCorrection μ (c • η)) ξ‖ ≤ B0 * c ^ 3 * t ^ 3 :=
    norm_iteratedFDeriv_comp_smul_le_of_homogeneous_bound _ j 3
      (hb.of_le (by exact_mod_cast hj)) hj c hc ξ hB0
      (norm_iteratedFDeriv_cubicCharacteristicCorrection_homogeneous_le μ hmom3 (c • ξ) j hj)
  have hGauss (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => characteristic2 stdGauss2 (c • η)) ξ‖ ≤ G * t :=
    norm_iteratedFDeriv_scaled_characteristic2_le stdGauss2 isStandardized_stdGauss2
      (IsGaussian.memLp_id _ _ (by simp)) c hc hc1 ξ hξ j hj
  have hOne (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => 1 - characteristic2 stdGauss2 (c • η)) ξ‖ ≤ H * c ^ 2 * t ^ 2 :=
    norm_iteratedFDeriv_one_sub_scaled_gaussian_le c hc hc1 ξ hξ j hj
  have hBare (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => characteristic2 μ (c • η) -
        characteristic2 stdGauss2 (c • η) - cubicCharacteristicCorrection μ (c • η)) ξ‖ ≤
          A0 * c ^ 4 * t ^ 4 :=
    norm_iteratedFDeriv_comp_smul_le_of_homogeneous_bound _ j 4
      (((hf.sub hg).sub hb).of_le (by exact_mod_cast hj)) (by omega) c hc ξ hA0
      (norm_iteratedFDeriv_characteristic2_sub_gaussian_sub_cubic_local_le μ hμ hmom (c • ξ) hξ j hj)
  refine ⟨hDiff k hk, ?_, ?_⟩
  · calc
      _ ≤ 8 * (G * t) * (B0 * c ^ 3 * t ^ 3) :=
        norm_iteratedFDeriv_mul_le_uniform_three _ _ hGC hBC ξ
          (by positivity) (by positivity) hGauss hCube k hk
      _ = _ := by dsimp [t]; ring
  · let e := fun η => characteristic2 μ (c • η) - characteristic2 stdGauss2 (c • η) -
        cubicCharacteristicCorrection μ (c • η)
    let w := fun η => cubicCharacteristicCorrection μ (c • η) *
        (1 - characteristic2 stdGauss2 (c • η))
    have heC : ContDiff ℝ 3 e := (hFC.sub hGC).sub hBC
    have hwC : ContDiff ℝ 3 w := hBC.mul (contDiff_const.sub hGC)
    have he : (fun η => characteristic2 μ (c • η) - characteristic2 stdGauss2 (c • η) -
        characteristic2 stdGauss2 (c • η) * cubicCharacteristicCorrection μ (c • η)) =
        fun η => e η + w η := by
      funext η
      dsimp [e, w]
      ring
    have hW : ‖iteratedFDeriv ℝ k w ξ‖ ≤ 8 * B0 * H * c ^ 5 * t ^ 5 := by
      calc
        _ ≤ 8 * (B0 * c ^ 3 * t ^ 3) * (H * c ^ 2 * t ^ 2) :=
          norm_iteratedFDeriv_mul_le_uniform_three _ _ hBC (contDiff_const.sub hGC) ξ
            (by positivity) (by positivity) hCube hOne k hk
        _ = _ := by ring
    have hc54 : c ^ 5 ≤ c ^ 4 := pow_le_pow_of_le_one hc hc1 (by omega)
    have ht45 : t ^ 4 ≤ t ^ 5 := pow_le_pow_right₀ ht (by omega)
    rw [he, fun_iteratedFDeriv_add (heC.of_le (by exact_mod_cast hk)) (hwC.of_le (by exact_mod_cast hk))]
    calc
      _ ≤ ‖iteratedFDeriv ℝ k e ξ‖ + ‖iteratedFDeriv ℝ k w ξ‖ := norm_add_le _ _
      _ ≤ A0 * c ^ 4 * t ^ 4 + 8 * B0 * H * c ^ 5 * t ^ 5 := add_le_add (hBare k hk) hW
      _ ≤ A0 * c ^ 4 * t ^ 5 + 8 * B0 * H * c ^ 4 * t ^ 5 := by gcongr
      _ = _ := by dsimp [t]; ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual normalized frequency supplies all three small-factor profiles with fixed law-dependent constants.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
literal `1 / sqrt n` scale of the normalized characteristic power. -/
theorem exists_norm_iteratedFDeriv_normalized_cubic_error_factors_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) :
    ∃ D B A : ℝ, 0 ≤ D ∧ 0 ≤ B ∧ 0 ≤ A ∧
      ∀ n : ℕ, 1 ≤ n → ∀ ξ : ℝ × ℝ, ‖ξ‖ ≤ Real.sqrt n → ∀ k : ℕ, k ≤ 3 →
        let c : ℝ := (Real.sqrt n)⁻¹
        ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ (c • η) -
          characteristic2 stdGauss2 (c • η)) ξ‖ ≤ D * c ^ 3 * (1 + ‖ξ‖) ^ 3 ∧
        ‖iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 (c • η) *
          cubicCharacteristicCorrection μ (c • η)) ξ‖ ≤ B * c ^ 3 * (1 + ‖ξ‖) ^ 4 ∧
        ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ (c • η) -
          characteristic2 stdGauss2 (c • η) - characteristic2 stdGauss2 (c • η) *
            cubicCharacteristicCorrection μ (c • η)) ξ‖ ≤ A * c ^ 4 * (1 + ‖ξ‖) ^ 5 := by
  obtain ⟨D, B, A, hD, hB, hA, hfactors⟩ :=
    exists_norm_iteratedFDeriv_scaled_cubic_error_factors_le μ hμ hmom
  refine ⟨D, B, A, hD, hB, hA, ?_⟩
  intro n hn ξ hξ k hk
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hs1 : 1 ≤ Real.sqrt (n : ℝ) := Real.one_le_sqrt.mpr (by exact_mod_cast hn)
  have hc1 : (Real.sqrt (n : ℝ))⁻¹ ≤ 1 := (inv_le_one₀ hs).mpr hs1
  have hsmall : ‖(Real.sqrt n)⁻¹ • ξ‖ ≤ 1 := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    exact (inv_mul_le_iff₀ hs).mpr (by simpa only [mul_one] using hξ)
  exact hfactors (Real.sqrt n)⁻¹ (by positivity) hc1 ξ hsmall k hk

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
