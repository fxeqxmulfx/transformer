import Transformer.Modes.Section3_CubicGaussianComparison
/-!
# Exterior Fourier rates for the cubic Edgeworth comparison

The `s = 3` case of arXiv:2412.09080v3, §5.4, requires the exterior
integrals in `eq:big-z-exp` and `eq:big-phi-poly` to have rate `1 / n`.
An integrable characteristic norm power and a strict modulus gap retain
integrability while geometric damping dominates every fixed polynomial
prefactor. The following result allows any natural inverse-power rate,
including `1 / n`, and keeps the integrable-power shift explicit.

For a Gaussian polynomial moment, multiplying by additional powers of
the frequency norm gives arbitrary inverse-radius tail bounds. This
uses the finite whole-space Gaussian moments, rather than a pointwise
constant bound on a region with infinite volume. At radius `a sqrt n`,
a second extra moment gives the required `1 / n` Gaussian rate.

The actual normalized-sum characteristic tail uses its exact frequency
Jacobian `n`. The constants may depend on the fixed law and radius
coefficient, as in §3 `thm:br`. The source's stronger exponential
Gaussian tail is replaced here by the inverse-radius bound needed for
the stated fixed-law rate. Polynomial weights in the Gaussian lemma
also permit estimating the cubic correction and its derivatives later.
The actual Gaussian function has finite moment integrals at every degree,
so all constants in these comparisons are finite real quantities.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Every polynomial norm weight of the actual Gaussian characteristic function is integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:small-z` and `eq:big-phi-poly`,
with the Gaussian characteristic function's literal normalization. -/
theorem integrable_norm_pow_characteristic2_stdGauss2 (k : ℕ) :
    Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ ^ k * ‖characteristic2 stdGauss2 ξ‖) := by
  convert integrable_norm_pow_gaussian_frequency k (b := 1 / 2) (by norm_num) using 1
  funext ξ
  rw [characteristic2_stdGauss2, Complex.norm_real, Real.norm_eq_abs,
    abs_of_pos (Real.exp_pos _)]
  congr 2
  ring

/-- Any additional Gaussian moment gives a quantitative inverse-radius tail.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly` and `eq:big-z-poly`.
The source bounds these tails exponentially; this elementary moment
comparison gives every natural inverse-radius power needed here. -/
theorem integral_norm_pow_gaussian_frequency_tail_le (k l : ℕ) {b r : ℝ}
    (hb : 0 < b) (hr : 0 < r) :
    (∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖}, ‖ξ‖ ^ k * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) ≤
      (r ^ l)⁻¹ * ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ (k + l) * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) := by
  have hIk := integrable_norm_pow_gaussian_frequency k hb
  have hIkl := integrable_norm_pow_gaussian_frequency (k + l) hb
  calc
    _ ≤ ∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖},
        (r ^ l)⁻¹ * (‖ξ‖ ^ (k + l) * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
      apply setIntegral_mono_on hIk.integrableOn (hIkl.const_mul _).integrableOn
        (isClosed_le continuous_const continuous_norm).measurableSet
      intro ξ hξ
      change r ≤ ‖ξ‖ at hξ
      have hp : r ^ l ≤ ‖ξ‖ ^ l := pow_le_pow_left₀ hr.le hξ l
      have hcoef : 1 ≤ (r ^ l)⁻¹ * ‖ξ‖ ^ l := (le_inv_mul_iff₀ (pow_pos hr l)).mpr (by simpa using hp)
      calc
        _ = 1 * (‖ξ‖ ^ k * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by ring
        _ ≤ ((r ^ l)⁻¹ * ‖ξ‖ ^ l) * (‖ξ‖ ^ k * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) :=
          mul_le_mul_of_nonneg_right hcoef (by positivity)
        _ = _ := by rw [pow_add]; ring
    _ ≤ ∫ ξ : ℝ × ℝ, (r ^ l)⁻¹ * (‖ξ‖ ^ (k + l) * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) :=
      setIntegral_le_integral (hIkl.const_mul _) (ae_of_all _ fun ξ => by positivity)
    _ = _ := integral_const_mul _ _

example : (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 := by norm_num

/-- Geometric damping dominates fixed polynomial factors at any inverse-power rate.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, including the explicit
integrable-power shift. The inverse power may be zero. -/
theorem eventually_nat_pow_mul_geometric_le_inv_pow {ε : ℝ} (hε0 : 0 < ε)
    (hε1 : ε < 1) (ν k l : ℕ) :
    ∀ᶠ n : ℕ in atTop, (n : ℝ) ^ k * ε ^ (n - ν) ≤ (ε ^ ν)⁻¹ * ((n : ℝ) ^ l)⁻¹ := by
  have hlim := (summable_pow_mul_geometric_of_norm_lt_one (k + l)
    (show ‖ε‖ < 1 by simpa [Real.norm_eq_abs, abs_of_pos hε0] using hε1)).tendsto_atTop_zero
  have hevent : ∀ᶠ n : ℕ in atTop, (n : ℝ) ^ (k + l) * ε ^ n < 1 :=
    hlim.eventually (gt_mem_nhds (show (0 : ℝ) < 1 by norm_num))
  filter_upwards [hevent, eventually_ge_atTop (max ν 1)] with n hn hnν
  have hn1 : 1 ≤ n := (le_max_right ν 1).trans hnν
  have hνn : ν ≤ n := (le_max_left ν 1).trans hnν
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hp : 0 < ε ^ ν * (n : ℝ) ^ l := by positivity
  have hdiv : (n : ℝ) ^ k * ε ^ (n - ν) ≤ 1 / (ε ^ ν * (n : ℝ) ^ l) := by
    apply (le_div_iff₀ hp).mpr
    have he : (n : ℝ) ^ k * ε ^ (n - ν) * (ε ^ ν * (n : ℝ) ^ l) =
        (n : ℝ) ^ (k + l) * ε ^ n := by
      calc
        _ = ((n : ℝ) ^ k * (n : ℝ) ^ l) * (ε ^ (n - ν) * ε ^ ν) := by ring
        _ = _ := by rw [← pow_add, ← pow_add, Nat.sub_add_cancel hνn]
    rw [he]
    exact hn.le
  simpa [one_div, mul_inv_rev, mul_comm] using hdiv

example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

/-- The actual normalized-sum characteristic tail has the `s = 3` inverse-n rate.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4 `eq:big-z-exp`,
its zero-derivative fixed-law step. The constant is nonnegative. -/
theorem eventually_integral_characteristic_scaledSum_tail_rate_inv_nat
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ)
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨ν, _, _, ε, hε0, hε1, htail⟩ :=
    exists_integral_characteristic_scaledSum_large_frequency_bound μ hcf ha
  let A : ℝ := ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν
  have hA : 0 ≤ A := integral_nonneg fun ξ => by positivity
  refine ⟨(ε ^ ν)⁻¹ * A, by positivity, ?_⟩
  filter_upwards [eventually_nat_pow_mul_geometric_le_inv_pow hε0 hε1 ν 1 1,
    eventually_ge_atTop ν] with n hrate hn
  simp only [pow_one] at hrate
  calc
    _ ≤ (n : ℝ) * ε ^ (n - ν) * A := htail n hn
    _ ≤ ((ε ^ ν)⁻¹ * (n : ℝ)⁻¹) * A := mul_le_mul_of_nonneg_right hrate hA
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧
    (0 : ℝ) < 1 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2, one_pos⟩

/-- The actual Gaussian characteristic tail is bounded by its second moment over n.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`, the `s = 3`
zero-derivative step with the literal radius `a sqrt n`. -/
theorem integral_characteristic2_stdGauss2_tail_le_inv_nat {a : ℝ} (ha : 0 < a)
    {n : ℕ} (hn : 1 ≤ n) :
    (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, ‖characteristic2 stdGauss2 ξ‖) ≤
      (a ^ 2)⁻¹ * (∫ ξ : ℝ × ℝ, ‖ξ‖ ^ 2 * ‖characteristic2 stdGauss2 ξ‖) * (n : ℝ)⁻¹ := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have he (ξ : ℝ × ℝ) : Real.exp (-(1 / 2 : ℝ) * (ξ.1 ^ 2 + ξ.2 ^ 2)) =
      ‖characteristic2 stdGauss2 ξ‖ := by
    rw [characteristic2_stdGauss2, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    congr 1
    ring
  have hR : (a * Real.sqrt (n : ℝ)) ^ 2 = a ^ 2 * n := by rw [mul_pow, Real.sq_sqrt hn0.le]
  have h := integral_norm_pow_gaussian_frequency_tail_le 0 2 (b := 1 / 2) (by norm_num) (mul_pos ha hs)
  simp only [pow_zero, zero_add, one_mul, he] at h
  rw [hR, mul_inv_rev] at h
  exact h.trans_eq (by ring)

example : (0 : ℝ) < 1 ∧ 1 ≤ (1 : ℕ) := ⟨one_pos, le_rfl⟩

end Transformer.Modes
