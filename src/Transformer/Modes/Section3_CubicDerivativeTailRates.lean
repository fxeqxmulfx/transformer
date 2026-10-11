import Transformer.Modes.Section3_CubicCorrectionDerivativeBounds

/-!
# Inverse-n exterior rates for all Gaussian and cubic correction derivatives

The two Gaussian terms in arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`
and `eq:big-z-poly`, have all polynomially weighted derivative norm
integrals finite. On the exterior of radius `r`, one additional norm
moment of order `l` bounds the unweighted integral by `r^(-l)` times
the whole-space moment. This direct integral argument preserves
integrability on the exterior, whose volume is infinite.

Taking `l = 2` and the literal radius `a sqrt n` gives `1 / n`.
Summing the four moment integrals supplies one nonnegative constant
for every derivative order from zero through three. We apply the
general argument to the actual Gaussian characteristic function and
its product with the literal cubic coefficient. The source's stronger
exponential tail is replaced by the inverse-n bound needed at fixed law.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- A finite norm moment gives a quantitative exterior bound for the actual norm integral.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly` and `eq:big-z-poly`,
the inverse-radius consequence of the finite Gaussian polynomial moments. -/
theorem integral_norm_tail_le_norm_moment {F : Type*} [NormedAddCommGroup F]
    (f : ℝ × ℝ → F) (hI : Integrable f) (l : ℕ)
    (hM : Integrable (fun ξ => ‖ξ‖ ^ l * ‖f ξ‖)) {r : ℝ} (hr : 0 < r) :
    (∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖}, ‖f ξ‖) ≤
      (r ^ l)⁻¹ * ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ l * ‖f ξ‖ := by
  have hMJ := hM.const_mul ((r ^ l)⁻¹)
  calc
    _ ≤ ∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖}, (r ^ l)⁻¹ * (‖ξ‖ ^ l * ‖f ξ‖) := by
      apply setIntegral_mono_on hI.norm.integrableOn hMJ.integrableOn
        (isClosed_le continuous_const continuous_norm).measurableSet
      intro ξ hξ
      change r ≤ ‖ξ‖ at hξ
      have hp : r ^ l ≤ ‖ξ‖ ^ l := pow_le_pow_left₀ hr.le hξ l
      have hcoef : 1 ≤ (r ^ l)⁻¹ * ‖ξ‖ ^ l :=
        (le_inv_mul_iff₀ (pow_pos hr l)).mpr (by simpa using hp)
      calc
        _ = 1 * ‖f ξ‖ := (one_mul _).symm
        _ ≤ ((r ^ l)⁻¹ * ‖ξ‖ ^ l) * ‖f ξ‖ :=
          mul_le_mul_of_nonneg_right hcoef (norm_nonneg _)
        _ = _ := by ring
    _ ≤ ∫ ξ : ℝ × ℝ, (r ^ l)⁻¹ * (‖ξ‖ ^ l * ‖f ξ‖) :=
      setIntegral_le_integral hMJ (ae_of_all _ fun ξ => by positivity)
    _ = _ := integral_const_mul _ _

example : Integrable (characteristic2 stdGauss2) ∧
    Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ ^ 2 * ‖characteristic2 stdGauss2 ξ‖) ∧
    (0 : ℝ) < 1 :=
  ⟨integrable_characteristic2_stdGauss2, integrable_norm_pow_characteristic2_stdGauss2 2, one_pos⟩

/-- Finite second norm moments of the derivative norms give one inverse-n exterior constant through order three.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly` and
`eq:big-z-poly`, at the literal radius `a sqrt n`. Every moment
input is retained as a hypothesis and supplied separately below. -/
theorem exists_integral_iteratedFDeriv_tail_rate_inv_nat_of_norm_moments
    (f : ℝ × ℝ → ℂ)
    (hI : ∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖))
    (hM : ∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖ξ‖ ^ 2 * ‖iteratedFDeriv ℝ k f ξ‖))
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ n : ℕ, 1 ≤ n → ∀ k : ℕ, k ≤ 3 →
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, ‖iteratedFDeriv ℝ k f ξ‖) ≤
        C * (n : ℝ)⁻¹ := by
  let J : ℕ → ℝ := fun k => ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ 2 * ‖iteratedFDeriv ℝ k f ξ‖
  let K : ℝ := ∑ k ∈ Finset.range 4, J k
  have hJ (k : ℕ) : 0 ≤ J k := integral_nonneg fun ξ => by positivity
  have hK : 0 ≤ K := Finset.sum_nonneg fun k _ => hJ k
  refine ⟨(a ^ 2)⁻¹ * K, by positivity, ?_⟩
  intro n hn k hk
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hR : (a * Real.sqrt (n : ℝ)) ^ 2 = a ^ 2 * n := by
    rw [mul_pow, Real.sq_sqrt hn0.le]
  have hJK : J k ≤ K := Finset.single_le_sum (fun j _ => hJ j)
    (Finset.mem_range.mpr (by omega))
  have hT := integral_norm_tail_le_norm_moment
    (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖) (hI k hk) 2
    (by simpa only [norm_norm] using hM k hk) (mul_pos ha hs)
  simp only [norm_norm] at hT
  rw [hR, mul_inv_rev] at hT
  calc
    _ ≤ (n : ℝ)⁻¹ * (a ^ 2)⁻¹ * J k := hT
    _ ≤ (n : ℝ)⁻¹ * (a ^ 2)⁻¹ * K := mul_le_mul_of_nonneg_left hJK (by positivity)
    _ = _ := by ring

example : (∀ k : ℕ, k ≤ 3 → Integrable (fun ξ =>
    ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖)) ∧
    (∀ k : ℕ, k ≤ 3 → Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ ^ 2 *
      ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖)) ∧ (0 : ℝ) < 1 :=
  ⟨fun k hk => by simpa using integrable_norm_pow_iteratedFDeriv_characteristic2_stdGauss2 0 k hk,
    fun k hk => integrable_norm_pow_iteratedFDeriv_characteristic2_stdGauss2 2 k hk, one_pos⟩

/-- All actual Gaussian characteristic derivative tails through order three have one inverse-n bound.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`,
the finite-moment exterior estimate at radius `a sqrt n`. -/
theorem exists_integral_characteristic2_stdGauss2_derivative_tail_rate_inv_nat
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ n : ℕ, 1 ≤ n → ∀ k : ℕ, k ≤ 3 →
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  apply exists_integral_iteratedFDeriv_tail_rate_inv_nat_of_norm_moments
    (characteristic2 stdGauss2) _ _ ha
  · intro k hk
    simpa only [pow_zero, one_mul] using
      integrable_norm_pow_iteratedFDeriv_characteristic2_stdGauss2 0 k hk
  · exact fun k hk => integrable_norm_pow_iteratedFDeriv_characteristic2_stdGauss2 2 k hk

example : (0 : ℝ) < 1 := one_pos

/-- Every actual Gaussian-damped cubic coefficient derivative through order three has an inverse-n exterior rate.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`, including
all Gaussian damping derivatives. No exponential moment is needed
to estimate the literal cubic moment coefficient. -/
theorem exists_integral_gaussian_cubicCorrection_derivative_tail_rate_inv_nat
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ n : ℕ, 1 ≤ n → ∀ k : ℕ, k ≤ 3 →
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 η *
          cubicCharacteristicCorrection μ η) ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  apply exists_integral_iteratedFDeriv_tail_rate_inv_nat_of_norm_moments
    (fun η => characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) _ _ ha
  · intro k hk
    simpa only [pow_zero, one_mul] using
      integrable_norm_pow_iteratedFDeriv_gaussian_cubicCorrection μ hmom 0 k hk
  · exact fun k hk => integrable_norm_pow_iteratedFDeriv_gaussian_cubicCorrection μ hmom 2 k hk

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), one_pos⟩

/-- The actual third derivative of the Gaussian cubic term is integrable with an inverse-n exterior rate.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`, the derivative
order needed for the cubic spatial weight in `eq:higher-error-goal`. -/
theorem exists_integrable_third_gaussian_cubicCorrection_tail_rate_inv_nat
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ n : ℕ, 1 ≤ n →
      Integrable (iteratedFDeriv ℝ 3 (fun η => characteristic2 stdGauss2 η *
        cubicCharacteristicCorrection μ η)) ∧
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ 3 (fun η => characteristic2 stdGauss2 η *
          cubicCharacteristicCorrection μ η) ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨C, hC, htail⟩ :=
    exists_integral_gaussian_cubicCorrection_derivative_tail_rate_inv_nat μ hmom ha
  exact ⟨C, hC, fun n hn =>
    ⟨integrable_iteratedFDeriv_gaussian_cubicCorrection μ hmom 3 le_rfl,
      htail n hn 3 le_rfl⟩⟩

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), one_pos⟩

end Transformer.Modes
