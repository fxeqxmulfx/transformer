import Transformer.Modes.Section3_PowerComparison
import Transformer.Modes.Section3_GaussianDerivativeComparison
import Transformer.Modes.Section3_GaussianDensity
/-!
# First-derivative Gaussian error for normalized sums

The first-derivative part of arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
is proved on a frequency ball of radius proportional to `sqrt n`.
The one-summand value and derivative errors, together with centering,
give a polynomial times a Gaussian majorant at rate `1 / sqrt n`.

The product sup norm determines the displayed polynomial. The elementary
power comparison gives the nonsharp damping exponent `1/8` instead of
the source's `1/4`. Finite third moment suffices; no exponential moment
or integrable-characteristic-power hypothesis is used in this local bound.
The exact Gaussian scaling identity permits comparison with its actual
characteristic derivative without introducing a model derivative.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The literal normalized-sum characteristic power fixes the standard Gaussian.
Source: arXiv:2412.09080v3, §3 `thm:br`, `Q₀ = φ`, and §5.4
`eq:br-9.10`. The frequency factor is the exact `1 / sqrt n` scale. -/
theorem characteristic2_stdGauss2_scaled_power (n : ℕ) (hn : 1 ≤ n) (ξ : ℝ × ℝ) :
    characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • ξ) ^ n = characteristic2 stdGauss2 ξ := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt hn0.le
  have hcoef : (n : ℝ) * (Real.sqrt n)⁻¹ ^ 2 = 1 := by
    calc
      _ = (Real.sqrt n) ^ 2 * (Real.sqrt n)⁻¹ ^ 2 := by rw [hsq]
      _ = _ := by field_simp
  have hq : (n : ℝ) * ((((Real.sqrt n)⁻¹ • ξ).1) ^ 2 +
      (((Real.sqrt n)⁻¹ • ξ).2) ^ 2) = ξ.1 ^ 2 + ξ.2 ^ 2 := by
    simp only [Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
    calc
      _ = ((n : ℝ) * (Real.sqrt n)⁻¹ ^ 2) * (ξ.1 ^ 2 + ξ.2 ^ 2) := by ring
      _ = _ := by rw [hcoef, one_mul]
  rw [characteristic2_stdGauss2, characteristic2_stdGauss2,
    ← Complex.ofReal_pow, ← Real.exp_nat_mul]
  congr 1
  apply congrArg Real.exp
  linarith [hq]

example : 1 ≤ (2 : ℕ) := by omega

/-- First-derivative error has an integrable Gaussian majorant at rate `1 / sqrt n`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4
`eq:br-9.10`, derivative order one. The proof's nonsharp damping
constant is `1/8`; the normalized-sum and derivative are literal. -/
theorem exists_fderiv_characteristic_scaledSum_sub_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) :
    ∃ a C : ℝ, 0 < a ∧ 0 ≤ C ∧ ∀ n : ℕ, 4 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n →
      ‖fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ -
          fderiv ℝ (characteristic2 stdGauss2) ξ‖ ≤
        C * (Real.sqrt n)⁻¹ * (‖ξ‖ ^ 2 + ‖ξ‖ ^ 4) *
          Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  obtain ⟨a, ha, hb⟩ := exists_characteristic2_gaussian_bound μ hμ hmom
  let K0 : ℝ := 4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1
  let K1 : ℝ := 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2)
  let KG : ℝ := 2 + 8 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2)
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hMG : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hK0 : 0 ≤ K0 := by dsimp [K0]; positivity
  have hK1 : 0 ≤ K1 := by dsimp [K1]; positivity
  have hKG : 0 ≤ KG := by dsimp [KG]; positivity
  refine ⟨min a 1, K1 + K0 * KG, by positivity, by positivity, ?_⟩
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
  have hdecay (j : ℕ) (hj : j ≤ 2) : r ^ (n - j) ≤ E := by
    dsimp [r, E]
    rw [← Real.exp_nat_mul]
    apply Real.exp_le_exp.mpr
    have hn4 : (4 : ℝ) ≤ n := by exact_mod_cast hn
    have hj2 : (j : ℝ) ≤ 2 := by exact_mod_cast hj
    have hcast : ((n - j : ℕ) : ℝ) = (n : ℝ) - j := by
      rw [Nat.cast_sub (by omega)]
    rw [hcast]
    have hq0 : 0 ≤ q := by dsimp [q]; positivity
    nlinarith [mul_nonneg (show 0 ≤ (n : ℝ) - 2 * j by linarith) hq0]
  have hR0 : ‖characteristic2 μ η - characteristic2 stdGauss2 η‖ ≤ K0 * ‖η‖ ^ 3 := by
    rw [characteristic2_stdGauss2]
    exact norm_characteristic2_sub_gaussian μ hμ hmom η hη1
  have hnorm : ‖η‖ = c * ‖ξ‖ := by rw [show η = c • ξ from rfl, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
  have hscale1 : c * (n : ℝ) * ‖η‖ ^ 2 = c * ‖ξ‖ ^ 2 := by
    rw [hnorm]
    calc
      _ = c * ((n : ℝ) * c ^ 2) * ‖ξ‖ ^ 2 := by ring
      _ = _ := by rw [hcoef, mul_one]
  have hscale2 : c * (n : ℝ) ^ 2 * ‖η‖ ^ 4 = c * ‖ξ‖ ^ 4 := by
    rw [hnorm]
    calc
      _ = c * ((n : ℝ) * c ^ 2) ^ 2 * ‖ξ‖ ^ 4 := by ring
      _ = _ := by rw [hcoef]; ring
  have hf := (contDiff_characteristic2 μ (k := 2) hμ.memLp).differentiable (by norm_num)
  have hg := (contDiff_characteristic2 stdGauss2 (k := 2)
    (IsGaussian.memLp_id _ _ (by simp))).differentiable (by norm_num)
  have hSum : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun ζ => (fun η => characteristic2 μ η ^ n) (c • ζ) := by
    funext ζ
    exact characteristic_scaledSum μ n ζ
  have hGP : characteristic2 stdGauss2 =
      fun ζ => (fun η => characteristic2 stdGauss2 η ^ n) (c • ζ) := by
    funext ζ
    exact (characteristic2_stdGauss2_scaled_power n (by omega) ζ).symm
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  intro v
  change ‖fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ v -
    fderiv ℝ (characteristic2 stdGauss2) ξ v‖ ≤ _
  have hR1 := norm_fderiv_characteristic2_sub_gaussian_apply_le μ hμ hmom η v
  have hDG := norm_fderiv_characteristic2_apply_small stdGauss2 isStandardized_stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) η v hη1
  have hP := norm_fderiv_complex_power_sub_apply_le (characteristic2 μ) (characteristic2 stdGauss2)
    n (by omega) η v (hf η) (hg η) hχ hG
  have hnPred : ((n - 1 : ℕ) : ℝ) ≤ (n : ℝ) := by exact_mod_cast (show n - 1 ≤ n by omega)
  rw [hSum, hGP,
    fderiv_comp_smul (𝕜 := ℝ) (f := fun η => characteristic2 μ η ^ n) c,
    fderiv_comp_smul (𝕜 := ℝ) (f := fun η => characteristic2 stdGauss2 η ^ n) c]
  change ‖c • (fderiv ℝ (fun η => characteristic2 μ η ^ n) η v) -
    c • (fderiv ℝ (fun η => characteristic2 stdGauss2 η ^ n) η v)‖ ≤ _
  rw [← smul_sub, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
  calc
    _ ≤ c * ((n : ℝ) * (r ^ (n - 1) * ‖fderiv ℝ (characteristic2 μ) η v -
        fderiv ℝ (characteristic2 stdGauss2) η v‖ + (n - 1 : ℕ) *
        ‖characteristic2 μ η - characteristic2 stdGauss2 η‖ * r ^ (n - 2) *
        ‖fderiv ℝ (characteristic2 stdGauss2) η v‖)) := mul_le_mul_of_nonneg_left hP hc.le
    _ ≤ c * ((n : ℝ) * (E * (K1 * ‖η‖ ^ 2 * ‖v‖) + (n : ℝ) *
        (K0 * ‖η‖ ^ 3) * E * (KG * ‖η‖ * ‖v‖))) := by
      gcongr
      · exact hdecay 1 (by omega)
      · exact hdecay 2 (by omega)
    _ = (K1 * (c * (n : ℝ) * ‖η‖ ^ 2) + K0 * KG * (c * (n : ℝ) ^ 2 * ‖η‖ ^ 4)) * E * ‖v‖ := by ring
    _ = (K1 * (c * ‖ξ‖ ^ 2) + K0 * KG * (c * ‖ξ‖ ^ 4)) * E * ‖v‖ := by rw [hscale1, hscale2]
    _ ≤ (K1 + K0 * KG) * c * (‖ξ‖ ^ 2 + ‖ξ‖ ^ 4) * E * ‖v‖ := by
      have hExtra : 0 ≤ K1 * ‖ξ‖ ^ 4 + K0 * KG * ‖ξ‖ ^ 2 := by positivity
      gcongr
      nlinarith

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
