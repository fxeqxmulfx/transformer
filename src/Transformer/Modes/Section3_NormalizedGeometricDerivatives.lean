module

public import Transformer.Modes.Section3_GeometricDerivativeBounds
public import Transformer.Modes.Section3_CubicScaledDerivativeBounds

/-!
# Gaussian decay of the actual normalized geometric factors

The two geometric factors in arXiv:2412.09080v3, §5.4
`eq:br-9.10`, are now evaluated on the actual characteristic
functions at frequency `ξ / sqrt n`. Their positive derivatives
have the common bound `L (1+‖ξ‖) / n`. Taking the mixed powers
therefore leaves only a polynomial of degree six in frequency.

The retained powers of the common Gaussian modulus have exponents
`n-7` and `n-8`. For `n ≥ 16` they are bounded by
`exp(-|ξ|²/8)` on a ball of radius `a sqrt n`. This elementary
bound uses `1/8` where the source's display uses `1/4`; the weaker
decay still has every required polynomial moment integrable.
Only a finite third moment and standardization are used here.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Losing at most half the normalized Gaussian powers retains fixed Gaussian decay.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
modulus powers left after differentiating the actual normalized sum. -/
theorem normalized_gaussian_frequency_power_le (n j : ℕ)
    (hn : 1 ≤ n) (hj : 2 * j ≤ n) (ξ : ℝ × ℝ) :
    (Real.exp (-(((Real.sqrt n)⁻¹ • ξ).1 ^ 2 +
      ((Real.sqrt n)⁻¹ • ξ).2 ^ 2) / 4)) ^ (n - j) ≤
        Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  let c : ℝ := (Real.sqrt n)⁻¹
  let q : ℝ := (c • ξ).1 ^ 2 + (c • ξ).2 ^ 2
  have hc2 : c ^ 2 = (n : ℝ)⁻¹ := by
    dsimp [c]
    rw [inv_pow, Real.sq_sqrt hn0.le]
  have hcoef : (n : ℝ) * c ^ 2 = 1 := by
    rw [hc2, mul_inv_cancel₀ hn0.ne']
  have hnq : (n : ℝ) * q = ξ.1 ^ 2 + ξ.2 ^ 2 := by
    dsimp [q]
    calc
      _ = ((n : ℝ) * c ^ 2) * (ξ.1 ^ 2 + ξ.2 ^ 2) := by ring
      _ = _ := by rw [hcoef, one_mul]
  change (Real.exp (-q / 4)) ^ (n - j) ≤ _
  rw [← Real.exp_nat_mul]
  apply Real.exp_le_exp.mpr
  rw [Nat.cast_sub (by omega)]
  have hjn : (2 : ℝ) * j ≤ n := by exact_mod_cast hj
  have hq0 : 0 ≤ q := by dsimp [q]; positivity
  nlinarith [mul_nonneg (show 0 ≤ (n : ℝ) - 2 * j by linarith) hq0]

example : (1 : ℕ) ≤ 2 ∧ 2 * (1 : ℕ) ≤ 2 := by norm_num

/-- Both actual normalized geometric-factor derivatives have polynomial Gaussian bounds through order three.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
the mixed powers in the exact cubic-error factorization.
The fixed-law envelope uses damping exponent `1/8`. -/
theorem exists_norm_iteratedFDeriv_normalized_geometric_sums_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) :
    ∃ a C : ℝ, 0 < a ∧ a ≤ 1 ∧ 0 ≤ C ∧ ∀ n : ℕ, 16 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n → ∀ k : ℕ, k ≤ 3 →
        let f := fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η)
        let g := fun η => characteristic2 stdGauss2 ((Real.sqrt n)⁻¹ • η)
        ‖iteratedFDeriv ℝ k (fun η => ∑ i ∈ Finset.range n,
          f η ^ i * g η ^ (n - 1 - i)) ξ‖ ≤
            C * (n : ℝ) * (1 + ‖ξ‖) ^ 6 * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) ∧
        ‖iteratedFDeriv ℝ k (fun η => ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range i,
          f η ^ j * g η ^ (n - 2 - j)) ξ‖ ≤
            C * (n : ℝ) ^ 2 * (1 + ‖ξ‖) ^ 6 * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  obtain ⟨a, ha, hb⟩ := exists_characteristic2_gaussian_bound μ hμ hmom
  let L : ℝ := 10 + 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2)
  let K : ℝ := 7 * (1 + L) ^ 3
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hMG : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hL : 0 ≤ L := by dsimp [L]; positivity
  have hK : 0 ≤ K := by dsimp [K]; positivity
  refine ⟨min a 1, 8 * K ^ 2, by positivity, min_le_right _ _, by positivity, ?_⟩
  intro n hn ξ hξ k hk
  have hn1 : 1 ≤ n := by omega
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  let c : ℝ := (Real.sqrt n)⁻¹
  let η : ℝ × ℝ := c • ξ
  let f := fun ζ => characteristic2 μ (c • ζ)
  let g := fun ζ => characteristic2 stdGauss2 (c • ζ)
  let q : ℝ := η.1 ^ 2 + η.2 ^ 2
  let Q : ℝ := ξ.1 ^ 2 + ξ.2 ^ 2
  let r : ℝ := Real.exp (-q / 4)
  let E : ℝ := Real.exp (-Q / 8)
  let t : ℝ := 1 + ‖ξ‖
  let A : ℝ := L * (n : ℝ)⁻¹ * t
  let P : ℝ := L * t
  have ht : 1 ≤ t := by dsimp [t]; linarith [norm_nonneg ξ]
  have hA : 0 ≤ A := by dsimp [A, t]; positivity
  have hP : 0 ≤ P := by dsimp [P, t]; positivity
  have hξ1 : ‖ξ‖ ≤ Real.sqrt n := hξ.trans (by
    simpa only [one_mul] using mul_le_mul_of_nonneg_right (min_le_right a 1) (Real.sqrt_nonneg (n : ℝ)))
  have hη : ‖η‖ ≤ min a 1 := by
    dsimp [η, c]
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    exact (inv_mul_le_iff₀ hs).mpr (by simpa only [mul_comm] using hξ)
  have hmodf : ‖f ξ‖ ≤ r := hb η ((le_min_iff.mp hη).1)
  have hmodg : ‖g ξ‖ ≤ r := by
    change ‖characteristic2 stdGauss2 η‖ ≤ r
    rw [characteristic2_stdGauss2, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    apply Real.exp_le_exp.mpr
    dsimp [r, q]
    nlinarith [sq_nonneg η.1, sq_nonneg η.2]
  have hr : r ≤ 1 := Real.exp_le_one_iff.mpr (by dsimp [q]; nlinarith [sq_nonneg η.1, sq_nonneg η.2])
  have hscale : ContDiff ℝ 3 (fun ζ : ℝ × ℝ => c • ζ) := by fun_prop
  have hf : ContDiff ℝ 3 f := (contDiff_characteristic2 μ (k := 3) hmom).comp hscale
  have hg : ContDiff ℝ 3 g :=
    (contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))).comp hscale
  have hDf (j : ℕ) (hj1 : 1 ≤ j) (hj3 : j ≤ 3) : ‖iteratedFDeriv ℝ j f ξ‖ ≤ A := by
    calc
      _ ≤ (10 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * (n : ℝ)⁻¹ * t :=
        norm_iteratedFDeriv_normalized_characteristic2_positive_le μ hμ hmom hn1 ξ hξ1 j hj1 hj3
      _ ≤ _ := by dsimp [A, L]; gcongr; linarith
  have hDg (j : ℕ) (hj1 : 1 ≤ j) (hj3 : j ≤ 3) : ‖iteratedFDeriv ℝ j g ξ‖ ≤ A := by
    calc
      _ ≤ (10 + 8 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2)) * (n : ℝ)⁻¹ * t :=
        norm_iteratedFDeriv_normalized_characteristic2_positive_le stdGauss2 isStandardized_stdGauss2
          (IsGaussian.memLp_id _ _ (by simp)) hn1 ξ hξ1 j hj1 hj3
      _ ≤ _ := by dsimp [A, L]; gcongr; linarith
  have hnA : (n : ℝ) * A ≤ P := by
    dsimp [A, P]
    have he : (n : ℝ) * (L * (n : ℝ)⁻¹ * t) = L * ((n : ℝ) * (n : ℝ)⁻¹) * t := by ring
    rw [he, mul_inv_cancel₀ hn0.ne', mul_one]
  have hpoly : 1 + P + 4 * P ^ 2 + P ^ 3 ≤ K * t ^ 3 := by
    have hP1 : 1 + P ≤ (1 + L) * t := by dsimp [P, t]; nlinarith [norm_nonneg ξ]
    calc
      _ ≤ 7 * (1 + P) ^ 3 := by nlinarith [sq_nonneg P, pow_nonneg hP 3]
      _ ≤ 7 * ((1 + L) * t) ^ 3 := by gcongr
      _ = _ := by dsimp [K]; ring
  have hdecay (j : ℕ) (hj : j ≤ 8) : r ^ (n - j) ≤ E := by
    exact normalized_gaussian_frequency_power_le n j hn1 (by omega) ξ
  obtain ⟨hS, hT⟩ := norm_iteratedFDeriv_geometric_sums_le_of_base_derivative_bounds
    f g hf hg ξ n hr hmodf hmodg hA hP hnA hDf hDg k hk
  refine ⟨hS.trans ?_, hT.trans ?_⟩
  · calc
      _ ≤ (n : ℝ) * 8 * (K * t ^ 3) ^ 2 * E := by gcongr; exact hdecay 7 (by omega)
      _ = _ := by dsimp [t, E, Q]; ring
  · calc
      _ ≤ (n : ℝ) ^ 2 * 8 * (K * t ^ 3) ^ 2 * E := by gcongr; exact hdecay 8 le_rfl
      _ = _ := by dsimp [t, E, Q]; ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
