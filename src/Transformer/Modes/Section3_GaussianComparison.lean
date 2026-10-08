import Transformer.Modes.Section3_SmallFrequency
/-!
# A quantitative Gaussian comparison for the actual normalized sum

The zero-derivative small-frequency estimate in §5.4 of
arXiv:2412.09080v3 follows from finite third moments, standardization,
and the exact normalized-sum characteristic-function formula. A power
estimate retains Gaussian damping while summing the single-summand
errors. Its normalization gives the explicit `1 / sqrt n` factor.
-/

open Real MeasureTheory Filter
open scoped ENNReal
namespace Transformer.Modes

/-- A telescoping power estimate retaining the common modulus bound.
Source: arXiv:2412.09080v3, §5.4, the characteristic-function power step
in `eq:br-9.10`. -/
theorem norm_complex_pow_succ_sub_le (z w : ℂ) {r : ℝ} (hz : ‖z‖ ≤ r) (hw : ‖w‖ ≤ r)
    (n : ℕ) : ‖z ^ (n + 1) - w ^ (n + 1)‖ ≤ ((n : ℝ) + 1) * ‖z - w‖ * r ^ n := by
  have hr : 0 ≤ r := (norm_nonneg z).trans hz
  induction n with
  | zero => simp
  | succ n ih =>
    have he : z ^ (n + 1 + 1) - w ^ (n + 1 + 1) =
        (z ^ (n + 1) - w ^ (n + 1)) * z + w ^ (n + 1) * (z - w) := by
      rw [pow_succ z (n + 1), pow_succ w (n + 1)]
      ring
    rw [he]
    calc
      _ ≤ ‖(z ^ (n + 1) - w ^ (n + 1)) * z‖ + ‖w ^ (n + 1) * (z - w)‖ := norm_add_le _ _
      _ = ‖z ^ (n + 1) - w ^ (n + 1)‖ * ‖z‖ + ‖w‖ ^ (n + 1) * ‖z - w‖ := by
        rw [norm_mul, norm_mul, norm_pow]
      _ ≤ (((n : ℝ) + 1) * ‖z - w‖ * r ^ n) * r + r ^ (n + 1) * ‖z - w‖ :=
        add_le_add (mul_le_mul ih hz (norm_nonneg _) (by positivity))
          (mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (norm_nonneg _) hw _) (norm_nonneg _))
      _ = _ := by rw [pow_succ]; push_cast; ring

example : ‖(1 : ℂ)‖ ≤ (1 : ℝ) ∧ ‖(0 : ℂ)‖ ≤ (1 : ℝ) := by norm_num

/-- A finite third moment controls the difference from the Gaussian
characteristic function inside the unit ball. Source: arXiv:2412.09080v3,
§3 `thm:br`, `s = 2`, and §5.4 `eq:br-9.10`. -/
theorem norm_characteristic2_sub_gaussian
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖characteristic2 μ ξ - ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ)‖ ≤
      (4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1) * ‖ξ‖ ^ 3 := by
  let Q : ℝ := ξ.1 ^ 2 + ξ.2 ^ 2
  have hQ0 : 0 ≤ Q := by dsimp [Q]; positivity
  have hQ2 : Q ≤ 2 * ‖ξ‖ ^ 2 := dot_self_le_two_norm_sq ξ
  have hr2 : ‖ξ‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg ξ]
  have hn : ‖((-(Q / 2) : ℝ) : ℂ)‖ = Q / 2 := by
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_nonpos (by linarith)]
    ring
  have hb := Complex.norm_exp_sub_one_sub_id_le (x := ((-(Q / 2) : ℝ) : ℂ)) (by rw [hn]; linarith)
  have hG : ‖((Real.exp (-Q / 2) : ℝ) : ℂ) - (1 - (Q : ℂ) / 2)‖ ≤ Q ^ 2 / 4 := by
    have he : ((Real.exp (-Q / 2) : ℝ) : ℂ) - (1 - (Q : ℂ) / 2) =
        Complex.exp ((-(Q / 2) : ℝ) : ℂ) - 1 - ((-(Q / 2) : ℝ) : ℂ) := by
      rw [Complex.ofReal_exp]
      have hq : -Q / 2 = -(Q / 2) := by ring
      rw [hq]
      push_cast
      ring
    rw [he]
    rw [hn] at hb
    nlinarith [hb]
  have he : characteristic2 μ ξ - ((Real.exp (-Q / 2) : ℝ) : ℂ) =
      (characteristic2 μ ξ - (1 - (Q : ℂ) / 2)) +
        ((1 - (Q : ℂ) / 2) - ((Real.exp (-Q / 2) : ℝ) : ℂ)) := by ring
  rw [he]
  have hR := norm_characteristic2_sub_quadratic μ hμ hmom ξ
  have hfour : Q ^ 2 / 4 ≤ ‖ξ‖ ^ 3 := by
    have hq := pow_le_pow_left₀ hQ0 hQ2 2
    have hp : ‖ξ‖ ^ 4 ≤ ‖ξ‖ ^ 3 := by nlinarith [pow_nonneg (norm_nonneg ξ) 3]
    nlinarith [hq]
  calc
    _ ≤ ‖characteristic2 μ ξ - (1 - (Q : ℂ) / 2)‖ +
        ‖(1 - (Q : ℂ) / 2) - ((Real.exp (-Q / 2) : ℝ) : ℂ)‖ := norm_add_le _ _
    _ ≤ _ := by rw [norm_sub_rev (1 - (Q : ℂ) / 2)]; linarith

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 := by
  exact ⟨inferInstance, isStandardized_stdGauss2,
    ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp), by norm_num⟩

/-- The actual normalized-sum characteristic function has Gaussian error
`O(n^(-1/2) |ξ|³ exp(-|ξ|²/8))` on `|ξ| ≤ a sqrt n`, under precisely the
third-moment and standardization hypotheses. Source: arXiv:2412.09080v3,
§5.4 `eq:br-9.10`, the zero-derivative `s = 2` step. The elementary power
argument here gives a nonsharp `1/8` exponent, where the source uses `1/4`;
no derivative estimate or large-frequency error is asserted. -/
theorem exists_characteristic_scaledSum_sub_gaussian_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) :
    ∃ a : ℝ, 0 < a ∧ ∀ n : ℕ, 2 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n →
      ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
          ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ)‖ ≤
        (4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1) * (Real.sqrt n)⁻¹ * ‖ξ‖ ^ 3 *
          Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  obtain ⟨a, ha, hb⟩ := exists_characteristic2_gaussian_bound μ hμ hmom
  refine ⟨min a 1, by positivity, ?_⟩
  intro n hn ξ hξ
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt hn0.le
  let η : ℝ × ℝ := (Real.sqrt (n : ℝ))⁻¹ • ξ
  let Q : ℝ := ξ.1 ^ 2 + ξ.2 ^ 2
  let q : ℝ := η.1 ^ 2 + η.2 ^ 2
  let C : ℝ := 4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1
  have hC : 0 ≤ C := by
    have hM : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
    dsimp [C]; linarith
  have hη : ‖η‖ ≤ min a 1 := by
    dsimp [η]
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    exact (inv_mul_le_iff₀ hs).mpr (by simpa [mul_comm] using hξ)
  have hqa : ‖η‖ ≤ a := (le_min_iff.mp hη).1
  have hq1 : ‖η‖ ≤ 1 := (le_min_iff.mp hη).2
  have hχ : ‖characteristic2 μ η‖ ≤ Real.exp (-q / 4) := hb η hqa
  have hG : ‖((Real.exp (-q / 2) : ℝ) : ℂ)‖ ≤ Real.exp (-q / 4) := by
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    apply Real.exp_le_exp.mpr
    dsimp [q]; nlinarith [sq_nonneg η.1, sq_nonneg η.2]
  have hnq : (n : ℝ) * q = Q := by
    dsimp [q, η, Q]
    have hcoef : (n : ℝ) * (Real.sqrt n)⁻¹ ^ 2 = 1 := by
      calc
        _ = (Real.sqrt n) ^ 2 * (Real.sqrt n)⁻¹ ^ 2 := by rw [hsq]
        _ = _ := by field_simp
    calc
      _ = ((n : ℝ) * (Real.sqrt n)⁻¹ ^ 2) * (ξ.1 ^ 2 + ξ.2 ^ 2) := by ring
      _ = _ := by rw [hcoef, one_mul]
  have hpow : (((Real.exp (-q / 2) : ℝ) : ℂ)) ^ n =
      ((Real.exp (-Q / 2) : ℝ) : ℂ) := by
    rw [← Complex.ofReal_pow, ← Real.exp_nat_mul]
    congr 1
    apply congrArg Real.exp
    linarith [hnq]
  have hnPred : n - 1 + 1 = n := by omega
  have hpowBound := norm_complex_pow_succ_sub_le (characteristic2 μ η)
    ((Real.exp (-q / 2) : ℝ) : ℂ) hχ hG (n - 1)
  rw [hnPred] at hpowBound
  have hcast : ((n - 1 : ℕ) : ℝ) + 1 = (n : ℝ) := by exact_mod_cast hnPred
  rw [hcast, hpow] at hpowBound
  have hR : ‖characteristic2 μ η - ((Real.exp (-q / 2) : ℝ) : ℂ)‖ ≤ C * ‖η‖ ^ 3 :=
    norm_characteristic2_sub_gaussian μ hμ hmom η hq1
  have hdecay : Real.exp (-q / 4) ^ (n - 1) ≤ Real.exp (-Q / 8) := by
    rw [← Real.exp_nat_mul]
    apply Real.exp_le_exp.mpr
    have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast hn
    have hq0 : 0 ≤ q := by dsimp [q]; positivity
    have hrel : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by linarith [hcast]
    rw [hrel]
    nlinarith [mul_nonneg (show 0 ≤ (n : ℝ) - 2 by linarith) hq0]
  have hscale : (n : ℝ) * ‖η‖ ^ 3 = (Real.sqrt n)⁻¹ * ‖ξ‖ ^ 3 := by
    dsimp [η]
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    calc
      _ = (Real.sqrt n) ^ 2 * ((Real.sqrt n)⁻¹ * ‖ξ‖) ^ 3 := by rw [hsq]
      _ = _ := by field_simp
  rw [characteristic_scaledSum]
  change ‖characteristic2 μ η ^ n - ((Real.exp (-Q / 2) : ℝ) : ℂ)‖ ≤ _
  calc
    _ ≤ (n : ℝ) * (C * ‖η‖ ^ 3) * (Real.exp (-q / 4) ^ (n - 1)) := by
      exact hpowBound.trans (mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left hR hn0.le) (by positivity))
    _ ≤ (n : ℝ) * (C * ‖η‖ ^ 3) * Real.exp (-Q / 8) :=
      mul_le_mul_of_nonneg_left hdecay (by positivity)
    _ = _ := by
      calc
        _ = C * ((n : ℝ) * ‖η‖ ^ 3) * Real.exp (-Q / 8) := by ring
        _ = _ := by rw [hscale]; dsimp [C, Q]; ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
