import Transformer.Modes.Section3_NormalizedGeometricDerivatives
import Transformer.Modes.Section3_CubicScaledErrorFactors

/-!
# Local Gaussian bounds for the actual cubic-corrected derivatives

For arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the exact corrected
power factorization now has all its actual inputs bounded. Its
fourth-order remainder is multiplied by a geometric factor of size
`n`, while the product of two third-order factors is multiplied by
a second geometric factor of size `n²`.

At the literal frequency scale `c = 1 / sqrt n`, both products
retain `1/n`: `n c⁴ = n² c⁶ = 1/n`. Every derivative through
order three is therefore bounded by a fixed polynomial of degree
thirteen times Gaussian decay. The constants depend on the fixed
standardized law and its fourth moment. Exponential moments are
not needed for this characteristic-function estimate.

The damping exponent is `1/8`, rather than the source's `1/4`,
as in the preceding mixed-power bounds. This still gives an
integrable envelope for the Fourier derivative estimates.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual cubic-corrected sum error is the corrected power error at the literal normalized scale.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and
§5.4 `eq:br-9.10`, before differentiation. -/
theorem characteristic_scaledSum_cubic_error_eq_corrected_power
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {n : ℕ} (hn : 1 ≤ n)
    (ξ : ℝ × ℝ) :
    let c : ℝ := (Real.sqrt n)⁻¹
    let f := characteristic2 μ (c • ξ)
    let g := characteristic2 stdGauss2 (c • ξ)
    let b := g * cubicCharacteristicCorrection μ (c • ξ)
    characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
      characteristic2 stdGauss2 ξ - (c : ℂ) * characteristic2 stdGauss2 ξ *
        cubicCharacteristicCorrection μ ξ =
      f ^ n - g ^ n - (n : ℂ) * g ^ (n - 1) * b := by
  exact (characteristic_scaledSum_cubic_error_factorization μ hn ξ).trans
    (complex_power_sub_cubic_factorization _ _ _ n).symm

example : IsProbabilityMeasure stdGauss2 ∧ (1 : ℕ) ≤ 1 := ⟨inferInstance, le_rfl⟩

/-- All actual cubic-corrected derivatives through order three have a local inverse-n Gaussian envelope.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, `s = 3`.
The explicit fixed-law envelope uses degree thirteen and damping exponent `1/8`. -/
theorem exists_iteratedFDeriv_characteristic_scaledSum_cubic_error_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) :
    ∃ a C : ℝ, 0 < a ∧ a ≤ 1 ∧ 0 ≤ C ∧ ∀ n : ℕ, 16 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n → ∀ k : ℕ, k ≤ 3 →
      ‖iteratedFDeriv ℝ k (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖ ≤
        C * (n : ℝ)⁻¹ * (1 + ‖ξ‖) ^ 13 * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  have hmom3 : MemLp id 3 μ := hmom.mono_exponent (by norm_num)
  obtain ⟨a, C, ha, ha1, hC, hgeom⟩ :=
    exists_norm_iteratedFDeriv_normalized_geometric_sums_gaussian_bound μ hμ hmom3
  obtain ⟨D, B, A, hD, hB, hA, hfactors⟩ :=
    exists_norm_iteratedFDeriv_normalized_cubic_error_factors_le μ hμ hmom
  refine ⟨a, (8 * A + 64 * B * D) * C, ha, ha1, by positivity, ?_⟩
  intro n hn ξ hξ k hk
  have hn1 : 1 ≤ n := by omega
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hξ1 : ‖ξ‖ ≤ Real.sqrt n := hξ.trans (by
    simpa only [one_mul] using mul_le_mul_of_nonneg_right ha1 (Real.sqrt_nonneg (n : ℝ)))
  let c : ℝ := (Real.sqrt n)⁻¹
  let t : ℝ := 1 + ‖ξ‖
  let E : ℝ := Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)
  let f := fun η => characteristic2 μ (c • η)
  let g := fun η => characteristic2 stdGauss2 (c • η)
  let b := fun η => g η * cubicCharacteristicCorrection μ (c • η)
  have hc : 0 ≤ c := by dsimp [c]; positivity
  have ht : 1 ≤ t := by dsimp [t]; linarith [norm_nonneg ξ]
  have hscale : ContDiff ℝ 3 (fun η : ℝ × ℝ => c • η) := by fun_prop
  have hf : ContDiff ℝ 3 f := (contDiff_characteristic2 μ (k := 3) hmom3).comp hscale
  have hg : ContDiff ℝ 3 g :=
    (contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))).comp hscale
  have hb : ContDiff ℝ 3 b := hg.mul
    (((contDiff_cubicCharacteristicCorrection_of_memLp μ hmom3).of_le (by simp)).comp hscale)
  have hDd (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => f η - g η) ξ‖ ≤ D * c ^ 3 * t ^ 3 :=
    (hfactors n hn1 ξ hξ1 j hj).1
  have hDb (j : ℕ) (hj : j ≤ 3) : ‖iteratedFDeriv ℝ j b ξ‖ ≤ B * c ^ 3 * t ^ 4 :=
    (hfactors n hn1 ξ hξ1 j hj).2.1
  have hR (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => f η - g η - b η) ξ‖ ≤ A * c ^ 4 * t ^ 5 :=
    (hfactors n hn1 ξ hξ1 j hj).2.2
  have hS (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => ∑ i ∈ Finset.range n,
        f η ^ i * g η ^ (n - 1 - i)) ξ‖ ≤ C * (n : ℝ) * t ^ 6 * E :=
    (hgeom n hn ξ hξ j hj).1
  have hT (j : ℕ) (hj : j ≤ 3) :
      ‖iteratedFDeriv ℝ j (fun η => ∑ i ∈ Finset.range n, ∑ l ∈ Finset.range i,
        f η ^ l * g η ^ (n - 2 - l)) ξ‖ ≤ C * (n : ℝ) ^ 2 * t ^ 6 * E :=
    (hgeom n hn ξ hξ j hj).2
  have he : (fun η =>
      characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
        characteristic2 stdGauss2 η - (c : ℂ) * characteristic2 stdGauss2 η *
          cubicCharacteristicCorrection μ η) =
      fun η => f η ^ n - g η ^ n - (n : ℂ) * g η ^ (n - 1) * b η := by
    funext η
    exact characteristic_scaledSum_cubic_error_eq_corrected_power μ hn1 η
  have hc2 : c ^ 2 = (n : ℝ)⁻¹ := by
    dsimp [c]
    rw [inv_pow, Real.sq_sqrt hn0.le]
  have hnscale : (n : ℝ) * c ^ 2 = 1 := by rw [hc2, mul_inv_cancel₀ hn0.ne']
  have hc4 : (n : ℝ) * c ^ 4 = (n : ℝ)⁻¹ := by
    calc
      _ = ((n : ℝ) * c ^ 2) * c ^ 2 := by ring
      _ = _ := by rw [hnscale, hc2, one_mul]
  have hc6 : (n : ℝ) ^ 2 * c ^ 6 = (n : ℝ)⁻¹ := by
    calc
      _ = ((n : ℝ) * c ^ 2) ^ 2 * c ^ 2 := by ring
      _ = _ := by rw [hnscale, hc2, one_pow, one_mul]
  have ht13 : t ^ 11 ≤ t ^ 13 := pow_le_pow_right₀ ht (by omega)
  rw [he]
  calc
    _ ≤ 8 * (A * c ^ 4 * t ^ 5) * (C * (n : ℝ) * t ^ 6 * E) +
        64 * (B * c ^ 3 * t ^ 4) * (D * c ^ 3 * t ^ 3) * (C * (n : ℝ) ^ 2 * t ^ 6 * E) :=
      norm_iteratedFDeriv_complex_cubic_error_le_of_geometric_bounds f g b hf hg hb ξ n
        (by positivity) (by positivity) (by positivity) (by dsimp [E]; positivity)
        (by dsimp [E]; positivity) hR hDb hDd hS hT k hk
    _ = 8 * A * C * ((n : ℝ) * c ^ 4) * t ^ 11 * E +
        64 * B * D * C * ((n : ℝ) ^ 2 * c ^ 6) * t ^ 13 * E := by ring
    _ ≤ 8 * A * C * (n : ℝ)⁻¹ * t ^ 13 * E +
        64 * B * D * C * (n : ℝ)⁻¹ * t ^ 13 * E := by
      rw [hc4, hc6]
      gcongr
    _ = _ := by dsimp [t, E]; ring

/-- The Gaussian law satisfies the moment and standardization hypotheses
of the full derivative estimate. -/
example :
    ∃ a C : ℝ, 0 < a ∧ a ≤ 1 ∧ 0 ≤ C ∧ ∀ n : ℕ, 16 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n → ∀ k : ℕ, k ≤ 3 →
      ‖iteratedFDeriv ℝ k (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => stdGauss2).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection stdGauss2 η) ξ‖ ≤
        C * (n : ℝ)⁻¹ * (1 + ‖ξ‖) ^ 13 * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  exact exists_iteratedFDeriv_characteristic_scaledSum_cubic_error_gaussian_bound stdGauss2
    isStandardized_stdGauss2 (IsGaussian.memLp_id _ _ (by simp))

end Transformer.Modes
