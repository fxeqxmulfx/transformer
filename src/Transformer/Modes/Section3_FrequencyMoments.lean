import Transformer.Modes.Section3_FourierTail
import Mathlib.Analysis.SpecificLimits.Normed

/-!
# Integrating polynomial Gaussian bounds and characteristic tails

Section 5.4 of arXiv:2412.09080v3 bounds the Fourier error separately
on small and large frequency regions. The polynomial Gaussian majorant
in `eq:small-z` is integrable in dimension two. A first Gaussian moment
also controls the exterior integral at radius `a sqrt n`.

For a fixed law, the proved integrable-power characteristic tail
`n ε^(n-ν)` is eventually `O(1 / sqrt n)`, as required by the
zero-derivative `s = 2` case of `eq:big-z-exp`. All constants here may
depend on the law and the fixed radius coefficient `a`; this is the
fixed-law step in §3 `thm:br`.
-/

open Real MeasureTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Every polynomial norm weight times a two-dimensional Gaussian is
integrable, with the project's product norm. Source: arXiv:2412.09080v3,
§5.4 `eq:small-z`; comparison with sums of coordinate powers proves
integrability without imposing an inner product on the product norm. -/
theorem integrable_norm_pow_gaussian_frequency (k : ℕ) {b : ℝ} (hb : 0 < b) :
    Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ ^ k * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
  have hpow : Integrable (fun x : ℝ => |x| ^ k * Real.exp (-b * x ^ 2)) := by
    have h := (integrable_rpow_mul_exp_neg_mul_sq hb
      (show (-1 : ℝ) < (k : ℝ) by
        have hk : (0 : ℝ) ≤ k := by positivity
        linarith)).norm
    simpa [Real.rpow_natCast, Real.norm_eq_abs, abs_mul, abs_pow,
      abs_of_pos (Real.exp_pos _)] using h
  have hzero := integrable_exp_neg_mul_sq hb
  have hI := (hpow.mul_prod hzero).add (hzero.mul_prod hpow)
  rw [← Measure.volume_eq_prod] at hI
  apply hI.mono' (by fun_prop)
  refine ae_of_all _ fun ξ => ?_
  have hmax : ‖ξ‖ ^ k ≤ |ξ.1| ^ k + |ξ.2| ^ k := by
    rw [Prod.norm_def, Real.norm_eq_abs, Real.norm_eq_abs]
    rcases le_total |ξ.1| |ξ.2| with h | h
    · rw [max_eq_right h]
      exact le_add_of_nonneg_left (pow_nonneg (abs_nonneg _) _)
    · rw [max_eq_left h]
      exact le_add_of_nonneg_right (pow_nonneg (abs_nonneg _) _)
  rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  calc
    _ ≤ (|ξ.1| ^ k + |ξ.2| ^ k) * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) :=
      mul_le_mul_of_nonneg_right hmax (Real.exp_pos _).le
    _ = _ := by
      have he : -b * (ξ.1 ^ 2 + ξ.2 ^ 2) = -b * ξ.1 ^ 2 + -b * ξ.2 ^ 2 := by ring
      rw [he, Real.exp_add]
      dsimp only [Pi.add_apply]
      ring

example : (0 : ℝ) < 1 / 8 := by norm_num

/-- A first Gaussian moment bounds its exterior mass by `1 / r`.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`, the
zero-derivative `s = 2` tail step. This polynomial bound suffices for
that step; the source gives the stronger exponential tail. -/
theorem integral_gaussian_frequency_tail_le {b r : ℝ} (hb : 0 < b) (hr : 0 < r) :
    (∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖}, Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) ≤
      r⁻¹ * ∫ ξ : ℝ × ℝ, ‖ξ‖ * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) := by
  have hI0 : Integrable (fun ξ : ℝ × ℝ => Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
    simpa using integrable_norm_pow_gaussian_frequency 0 hb
  have hI1 : Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
    simpa using integrable_norm_pow_gaussian_frequency 1 hb
  calc
    _ ≤ ∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖},
        r⁻¹ * (‖ξ‖ * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
      apply setIntegral_mono_on hI0.integrableOn (hI1.const_mul _).integrableOn
        (isClosed_le continuous_const continuous_norm).measurableSet
      intro ξ hξ
      change r ≤ ‖ξ‖ at hξ
      have h : 1 ≤ r⁻¹ * ‖ξ‖ := (le_inv_mul_iff₀ hr).mpr (by simpa using hξ)
      nlinarith [Real.exp_pos (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))]
    _ ≤ ∫ ξ : ℝ × ℝ, r⁻¹ * (‖ξ‖ * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) :=
      setIntegral_le_integral (hI1.const_mul _) (ae_of_all _ fun ξ => by positivity)
    _ = _ := integral_const_mul _ _

example : (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 := by norm_num

/-- A geometric tail with the two-dimensional scaling factor `n`
is eventually bounded at the required `1 / sqrt n` rate. Source:
arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, for a fixed law's `ε < 1`.
The integrable power `ν` is retained explicitly. -/
theorem eventually_nat_mul_geometric_le_inv_sqrt {ε : ℝ} (hε0 : 0 < ε)
    (hε1 : ε < 1) (ν : ℕ) :
    ∀ᶠ n : ℕ in atTop, (n : ℝ) * ε ^ (n - ν) ≤ (ε ^ ν)⁻¹ * (Real.sqrt n)⁻¹ := by
  have hlim := (summable_pow_mul_geometric_of_norm_lt_one 2
    (show ‖ε‖ < 1 by simpa [Real.norm_eq_abs, abs_of_pos hε0] using hε1)).tendsto_atTop_zero
  have hevent : ∀ᶠ n : ℕ in atTop, (n : ℝ) ^ 2 * ε ^ n < 1 :=
    hlim.eventually (gt_mem_nhds (show (0 : ℝ) < 1 by norm_num))
  filter_upwards [hevent, eventually_ge_atTop (max ν 1)] with n hn hnν
  have hn1 : 1 ≤ n := (le_max_right ν 1).trans hnν
  have hνn : ν ≤ n := (le_max_left ν 1).trans hnν
  have hnR : (1 : ℝ) ≤ n := by exact_mod_cast hn1
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr (by linarith)
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt (by positivity)
  have hsle : Real.sqrt (n : ℝ) ≤ n := by nlinarith [Real.sqrt_nonneg (n : ℝ)]
  have hcoef : (n : ℝ) * Real.sqrt n * ε ^ n ≤ 1 := by
    calc
      _ ≤ (n : ℝ) ^ 2 * ε ^ n := by
        exact mul_le_mul_of_nonneg_right (by nlinarith) (by positivity)
      _ ≤ _ := hn.le
  have hp : 0 < ε ^ ν * Real.sqrt (n : ℝ) := by positivity
  have hdiv : (n : ℝ) * ε ^ (n - ν) ≤ 1 / (ε ^ ν * Real.sqrt (n : ℝ)) := by
    apply (le_div_iff₀ hp).mpr
    have he : (n : ℝ) * ε ^ (n - ν) * (ε ^ ν * Real.sqrt (n : ℝ)) =
        (n : ℝ) * Real.sqrt n * ε ^ n := by
      calc
        _ = (n : ℝ) * Real.sqrt n * (ε ^ (n - ν) * ε ^ ν) := by ring
        _ = _ := by rw [← pow_add, Nat.sub_add_cancel hνn]
    rw [he]
    exact hcoef
  simpa [one_div, mul_inv_rev, mul_comm] using hdiv

example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

/-- The actual normalized-sum characteristic tail is eventually
`O(1 / sqrt n)` for every fixed law satisfying the integrable-power
condition. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4
`eq:big-z-exp`, the zero-derivative `s = 2` case. -/
theorem eventually_integral_characteristic_scaledSum_tail_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ)
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, ∀ᶠ n : ℕ in atTop,
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖) ≤
          C * (Real.sqrt n)⁻¹ := by
  obtain ⟨ν, _, _, ε, hε0, hε1, htail⟩ :=
    exists_integral_characteristic_scaledSum_large_frequency_bound μ hcf ha
  let A : ℝ := ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν
  have hA : 0 ≤ A := integral_nonneg fun ξ => by positivity
  refine ⟨(ε ^ ν)⁻¹ * A, ?_⟩
  filter_upwards [eventually_nat_mul_geometric_le_inv_sqrt hε0 hε1 ν,
    eventually_ge_atTop ν] with n hrate hn
  calc
    _ ≤ (n : ℝ) * ε ^ (n - ν) * A := htail n hn
    _ ≤ ((ε ^ ν)⁻¹ * (Real.sqrt n)⁻¹) * A := mul_le_mul_of_nonneg_right hrate hA
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧
    (0 : ℝ) < 1 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2, one_pos⟩

end Transformer.Modes
