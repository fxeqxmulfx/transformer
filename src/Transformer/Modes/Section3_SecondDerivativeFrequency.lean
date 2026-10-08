import Transformer.Modes.Section3_FirstDerivativeFrequency
import Transformer.Modes.Section3_GaussianDerivativeComparison
import Transformer.Modes.Section3_GaussianDensity
/-!
# Second-derivative Gaussian error for normalized sums

The second-derivative part of arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
is proved on a frequency ball of radius proportional to `sqrt n`.
The two product-rule terms yield linear, cubic, and quintic frequency
factors, each multiplied by Gaussian damping and `1 / sqrt n`.

Only standardization and finite third moment are needed. The first
characteristic derivatives are small near zero, their comparison has
quadratic error, and the second-derivative comparison has linear error.
The elementary power estimates give the nonsharp exponent `1/8` instead
of the source's `1/4`. The full multilinear operator norm is controlled.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

set_option maxHeartbeats 1000000 in
/-- Second-derivative error has an integrable Gaussian majorant at rate `1 / sqrt n`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4
`eq:br-9.10`, derivative order two. Both normalized-sum product-rule
terms are retained; the proof's nonsharp damping constant is `1/8`. -/
theorem exists_iteratedFDeriv_two_characteristic_scaledSum_sub_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) :
    ∃ a C : ℝ, 0 < a ∧ 0 ≤ C ∧ ∀ n : ℕ, 6 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n →
      ‖iteratedFDeriv ℝ 2
          (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ -
          iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖ ≤
        C * (Real.sqrt n)⁻¹ * (‖ξ‖ + ‖ξ‖ ^ 3 + ‖ξ‖ ^ 5) *
          Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  obtain ⟨a, ha, hb⟩ := exists_characteristic2_gaussian_bound μ hμ hmom
  let K0 : ℝ := 4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1
  let K1 : ℝ := 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2)
  let KG : ℝ := 2 + 8 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2)
  let KM : ℝ := 2 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hMG : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hK0 : 0 ≤ K0 := by dsimp [K0]; positivity
  have hK1 : 0 ≤ K1 := by dsimp [K1]; positivity
  have hKG : 0 ≤ KG := by dsimp [KG]; positivity
  have hKM : 0 ≤ KM := by dsimp [KM]; positivity
  refine ⟨min a 1, K1 * (KM + KG) + K0 * KG ^ 2 + K1 + 8 * K0, by positivity, by positivity, ?_⟩
  intro n hn ξ hξ
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt hn0.le
  let c : ℝ := (Real.sqrt (n : ℝ))⁻¹
  let η : ℝ × ℝ := c • ξ
  let Q : ℝ := ξ.1 ^ 2 + ξ.2 ^ 2
  let q : ℝ := η.1 ^ 2 + η.2 ^ 2
  let r : ℝ := Real.exp (-q / 4)
  let E : ℝ := Real.exp (-Q / 8)
  have hc : 0 < c := by dsimp [c]; positivity
  have hη : ‖η‖ ≤ min a 1 := by
    dsimp [η, c]
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    exact (inv_mul_le_iff₀ hs).mpr (by simpa [mul_comm] using hξ)
  have hη1 : ‖η‖ ≤ 1 := (le_min_iff.mp hη).2
  have hχ : ‖characteristic2 μ η‖ ≤ r := hb η ((le_min_iff.mp hη).1)
  have hG : ‖characteristic2 stdGauss2 η‖ ≤ r := by
    rw [characteristic2_stdGauss2, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _)]
    apply Real.exp_le_exp.mpr
    dsimp [r, q]; nlinarith [sq_nonneg η.1, sq_nonneg η.2]
  have hcoef : (n : ℝ) * c ^ 2 = 1 := by
    dsimp [c]
    calc
      _ = (Real.sqrt n) ^ 2 * (Real.sqrt n)⁻¹ ^ 2 := by rw [hsq]
      _ = _ := by field_simp
  have hnq : (n : ℝ) * q = Q := by
    dsimp [q, η, Q]
    calc
      _ = ((n : ℝ) * c ^ 2) * (ξ.1 ^ 2 + ξ.2 ^ 2) := by ring
      _ = _ := by rw [hcoef, one_mul]
  have hdecay (j : ℕ) (hj : j ≤ 3) : r ^ (n - j) ≤ E := by
    dsimp [r, E]
    rw [← Real.exp_nat_mul]
    apply Real.exp_le_exp.mpr
    have hn6 : (6 : ℝ) ≤ n := by exact_mod_cast hn
    have hj3 : (j : ℝ) ≤ 3 := by exact_mod_cast hj
    have hcast : ((n - j : ℕ) : ℝ) = (n : ℝ) - j := by
      rw [Nat.cast_sub (by omega)]
    rw [hcast]
    have hq0 : 0 ≤ q := by dsimp [q]; positivity
    nlinarith [mul_nonneg (show 0 ≤ (n : ℝ) - 2 * j by linarith) hq0]
  have hR0 : ‖characteristic2 μ η - characteristic2 stdGauss2 η‖ ≤ K0 * ‖η‖ ^ 3 := by
    rw [characteristic2_stdGauss2]
    exact norm_characteristic2_sub_gaussian μ hμ hmom η hη1
  have hnorm : ‖η‖ = c * ‖ξ‖ := by rw [show η = c • ξ from rfl, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
  have hscale1 : c ^ 2 * (n : ℝ) * ‖η‖ = c * ‖ξ‖ := by
    rw [hnorm]
    calc
      _ = ((n : ℝ) * c ^ 2) * c * ‖ξ‖ := by ring
      _ = _ := by rw [hcoef, one_mul]
  have hscale3 : c ^ 2 * (n : ℝ) ^ 2 * ‖η‖ ^ 3 = c * ‖ξ‖ ^ 3 := by
    rw [hnorm]
    calc
      _ = c * ((n : ℝ) * c ^ 2) ^ 2 * ‖ξ‖ ^ 3 := by ring
      _ = _ := by rw [hcoef]; ring
  have hscale5 : c ^ 2 * (n : ℝ) ^ 3 * ‖η‖ ^ 5 = c * ‖ξ‖ ^ 5 := by
    rw [hnorm]
    calc
      _ = c * ((n : ℝ) * c ^ 2) ^ 3 * ‖ξ‖ ^ 5 := by ring
      _ = _ := by rw [hcoef]; ring
  have hf := contDiff_characteristic2 μ (k := 2) hμ.memLp
  have hg := contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp))
  have hSum : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun ζ => (fun η => characteristic2 μ η ^ n) (c • ζ) := by
    funext ζ
    exact characteristic_scaledSum μ n ζ
  have hGP : characteristic2 stdGauss2 =
      fun ζ => (fun η => characteristic2 stdGauss2 η ^ n) (c • ζ) := by
    funext ζ
    exact (characteristic2_stdGauss2_scaled_power n (by omega) ζ).symm
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  rw [Fin.prod_univ_two]
  change ‖iteratedFDeriv ℝ 2
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ v -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ v‖ ≤ _
  have hR1 (w : ℝ × ℝ) := norm_fderiv_characteristic2_sub_gaussian_apply_le μ hμ hmom η w
  have hR2 := norm_iteratedFDeriv_two_characteristic2_sub_gaussian_apply_le μ hμ hmom η v
  have hDM (w : ℝ × ℝ) := norm_fderiv_characteristic2_apply_small μ hμ hmom η w hη1
  have hDG (w : ℝ × ℝ) := norm_fderiv_characteristic2_apply_small stdGauss2 isStandardized_stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) η w hη1
  have hD2G : ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) η v‖ ≤ 8 * ‖v 0‖ * ‖v 1‖ := by
    have h := (iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) η).le_opNorm v
    rw [Fin.prod_univ_two] at h
    calc
      _ ≤ ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) η‖ * (‖v 0‖ * ‖v 1‖) := h
      _ ≤ 8 * (‖v 0‖ * ‖v 1‖) := by
        gcongr
        exact norm_iteratedFDeriv_two_characteristic2_of_standardized stdGauss2 isStandardized_stdGauss2 η
      _ = _ := by ring
  have hP := norm_iteratedFDeriv_two_complex_power_sub_apply_le
    (characteristic2 μ) (characteristic2 stdGauss2) n (by omega) hf hg η v hχ hG
  have hnPred1 : ((n - 1 : ℕ) : ℝ) ≤ (n : ℝ) := by exact_mod_cast (show n - 1 ≤ n by omega)
  have hnPred2 : ((n - 2 : ℕ) : ℝ) ≤ (n : ℝ) := by exact_mod_cast (show n - 2 ≤ n by omega)
  rw [hSum, hGP,
    iteratedFDeriv_two_complex_comp_smul (fun η => characteristic2 μ η ^ n) c (hf.pow n),
    iteratedFDeriv_two_complex_comp_smul (fun η => characteristic2 stdGauss2 η ^ n) c (hg.pow n)]
  change ‖(c : ℂ) ^ 2 * iteratedFDeriv ℝ 2 (fun η => characteristic2 μ η ^ n) η v -
    (c : ℂ) ^ 2 * iteratedFDeriv ℝ 2 (fun η => characteristic2 stdGauss2 η ^ n) η v‖ ≤ _
  rw [← mul_sub, norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hc]
  calc
    _ ≤ c ^ 2 * ((n : ℝ) * (n - 1 : ℕ) *
        (r ^ (n - 2) *
          (‖fderiv ℝ (characteristic2 μ) η (v 0)‖ *
            ‖fderiv ℝ (characteristic2 μ) η (v 1) - fderiv ℝ (characteristic2 stdGauss2) η (v 1)‖ +
            ‖fderiv ℝ (characteristic2 μ) η (v 0) - fderiv ℝ (characteristic2 stdGauss2) η (v 0)‖ *
            ‖fderiv ℝ (characteristic2 stdGauss2) η (v 1)‖) +
          (n - 2 : ℕ) * ‖characteristic2 μ η - characteristic2 stdGauss2 η‖ * r ^ (n - 3) *
            ‖fderiv ℝ (characteristic2 stdGauss2) η (v 0)‖ *
            ‖fderiv ℝ (characteristic2 stdGauss2) η (v 1)‖) +
        (n : ℝ) * (r ^ (n - 1) * ‖iteratedFDeriv ℝ 2 (characteristic2 μ) η v -
            iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) η v‖ +
          (n - 1 : ℕ) * ‖characteristic2 μ η - characteristic2 stdGauss2 η‖ *
            r ^ (n - 2) * ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) η v‖)) :=
      mul_le_mul_of_nonneg_left hP (sq_nonneg c)
    _ ≤ c ^ 2 * ((n : ℝ) * n *
        (E * ((KM * ‖η‖ * ‖v 0‖) * (K1 * ‖η‖ ^ 2 * ‖v 1‖) +
          (K1 * ‖η‖ ^ 2 * ‖v 0‖) * (KG * ‖η‖ * ‖v 1‖)) +
          (n : ℝ) * (K0 * ‖η‖ ^ 3) * E * (KG * ‖η‖ * ‖v 0‖) * (KG * ‖η‖ * ‖v 1‖)) +
        (n : ℝ) * (E * (K1 * ‖η‖ * ‖v 0‖ * ‖v 1‖) +
          (n : ℝ) * (K0 * ‖η‖ ^ 3) * E * (8 * ‖v 0‖ * ‖v 1‖))) := by
      gcongr
      all_goals first | exact hdecay _ (by omega) | exact hR1 _ | exact hDM _ | exact hDG _
    _ = (K1 * (KM + KG) * (c ^ 2 * (n : ℝ) ^ 2 * ‖η‖ ^ 3) +
        K0 * KG ^ 2 * (c ^ 2 * (n : ℝ) ^ 3 * ‖η‖ ^ 5) +
        K1 * (c ^ 2 * (n : ℝ) * ‖η‖) + 8 * K0 * (c ^ 2 * (n : ℝ) ^ 2 * ‖η‖ ^ 3)) *
          E * (‖v 0‖ * ‖v 1‖) := by ring
    _ = (K1 * (KM + KG) * (c * ‖ξ‖ ^ 3) + K0 * KG ^ 2 * (c * ‖ξ‖ ^ 5) +
        K1 * (c * ‖ξ‖) + 8 * K0 * (c * ‖ξ‖ ^ 3)) * E * (‖v 0‖ * ‖v 1‖) := by
      rw [hscale1, hscale3, hscale5]
    _ ≤ (K1 * (KM + KG) + K0 * KG ^ 2 + K1 + 8 * K0) * c *
        (‖ξ‖ + ‖ξ‖ ^ 3 + ‖ξ‖ ^ 5) * E * (‖v 0‖ * ‖v 1‖) := by
      gcongr
      have h0 : 0 ≤ K1 * (KM + KG) := by positivity
      have h1 : 0 ≤ K0 * KG ^ 2 := by positivity
      have h2 : 0 ≤ 8 * K0 := by positivity
      nlinarith [mul_nonneg h0 (norm_nonneg ξ), mul_nonneg h0 (pow_nonneg (norm_nonneg ξ) 5),
        mul_nonneg h1 (norm_nonneg ξ), mul_nonneg h1 (pow_nonneg (norm_nonneg ξ) 3),
        mul_nonneg hK1 (pow_nonneg (norm_nonneg ξ) 3), mul_nonneg hK1 (pow_nonneg (norm_nonneg ξ) 5),
        mul_nonneg h2 (norm_nonneg ξ), mul_nonneg h2 (pow_nonneg (norm_nonneg ξ) 5)]

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
