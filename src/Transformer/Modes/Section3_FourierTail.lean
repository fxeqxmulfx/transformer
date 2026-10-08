import Transformer.Modes.Section3_LargeFrequency

/-!
# The integrable large-frequency tail of the actual normalized sum

The damping bound `ε^n` alone cannot be integrated over an exterior
region of infinite volume. Section 3 `thm:br` of arXiv:2412.09080v3
provides an integrable characteristic norm power `ν`. Splitting the
actual `n`th power into `n-ν` damped factors and `ν` integrable factors
gives an exterior integral bounded by
`n ε^(n-ν) ∫ |χ|^ν`. The factor `n` is the exact two-dimensional
Jacobian for frequency scaling by `1 / sqrt n`.

This proves the zero-derivative large-frequency integral step of §5.4
`eq:big-z-exp` for any fixed probability law with the source condition.
Dependence on the law is retained; tracking this bound uniformly in a
varying bandwidth or translation requires further estimates.
-/

open Real MeasureTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Larger norm powers remain integrable because a probability law's
characteristic function has modulus at most one. Source:
arXiv:2412.09080v3, §3 `thm:br` and §5.4 `eq:big-z-exp`. -/
theorem integrable_characteristic2_pow_of_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {ν n : ℕ} (hn : ν ≤ n)
    (hI : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν)) :
    Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ n) := by
  apply hI.mono' ((continuous_characteristic2 μ).norm.pow n).aestronglyMeasurable
  exact ae_of_all _ fun ξ => by
    change ‖‖characteristic2 μ ξ‖ ^ n‖ ≤ _
    rw [Real.norm_eq_abs, abs_of_nonneg (pow_nonneg (norm_nonneg _) n)]
    exact pow_le_pow_of_le_one (norm_nonneg _) (norm_characteristic2_le_one μ ξ) hn

example : ∃ ν : ℕ, ν ≤ ν ∧ Integrable (fun ξ => ‖characteristic2 stdGauss2 ξ‖ ^ ν) := by
  obtain ⟨ν, _, hI⟩ := hasIntegrableCharFun_stdGauss2
  exact ⟨ν, le_rfl, hI⟩

/-- Scaling frequency by `1 / sqrt n` contributes exactly the factor
`n` to the two-dimensional integral. Source: arXiv:2412.09080v3, §5.4
`eq:big-z-exp`; this uses the actual normalized-sum frequency scaling. -/
theorem integral_characteristic2_pow_scaledFrequency (μ : Measure (ℝ × ℝ))
    {n : ℕ} (hn : 1 ≤ n) (ν : ℕ) :
    (∫ ξ, ‖characteristic2 μ ((Real.sqrt (n : ℝ))⁻¹ • ξ)‖ ^ ν) =
      (n : ℝ) * ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have h := Measure.integral_comp_smul_of_nonneg volume
    (fun ξ : ℝ × ℝ => ‖characteristic2 μ ξ‖ ^ ν) (Real.sqrt (n : ℝ))⁻¹
    (hR := inv_nonneg.mpr hs.le)
  simpa [Module.finrank_prod, Real.sq_sqrt hn0.le, smul_eq_mul] using h

example : 1 ≤ (1 : ℕ) := le_rfl

/-- Retaining an integrable norm power makes exponential damping
integrable over the whole exterior region. Its explicit factor `n`
comes from the two-dimensional frequency scaling. Source:
arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, the zero-derivative case. -/
theorem integral_characteristic_scaledSum_large_frequency_of_gap
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {ν n : ℕ} (hν : 1 ≤ ν) (hn : ν ≤ n)
    (hI : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν)) {a ε : ℝ} (hε : 0 ≤ ε)
    (hgap : ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ ε) :
    (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
      ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖) ≤
        (n : ℝ) * ε ^ (n - ν) * ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν := by
  have hn1 : 1 ≤ n := hν.trans hn
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn1)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  let f : ℝ × ℝ → ℝ := fun ξ =>
    ε ^ (n - ν) * ‖characteristic2 μ ((Real.sqrt (n : ℝ))⁻¹ • ξ)‖ ^ ν
  have hIf : Integrable f := (hI.comp_smul (inv_ne_zero hs.ne')).const_mul _
  have hISum := integrable_characteristic_scaledSum_of_power μ hn1
    (integrable_characteristic2_pow_of_le μ hn hI)
  have hSf : ∀ ξ, 0 ≤ f ξ := by intro ξ; dsimp [f]; positivity
  have hS : MeasurableSet {ξ : ℝ × ℝ | a * Real.sqrt n ≤ ‖ξ‖} :=
    (isClosed_le continuous_const continuous_norm).measurableSet
  calc
    _ ≤ ∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, f ξ := by
      apply setIntegral_mono_on hISum.norm.integrableOn hIf.integrableOn hS
      intro ξ hξ
      change a * Real.sqrt n ≤ ‖ξ‖ at hξ
      have hη : a ≤ ‖(Real.sqrt (n : ℝ))⁻¹ • ξ‖ := by
        rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hs.le)]
        exact (le_inv_mul_iff₀ hs).mpr (by simpa only [mul_comm] using hξ)
      rw [characteristic_scaledSum, norm_pow]
      have he : ‖characteristic2 μ ((Real.sqrt (n : ℝ))⁻¹ • ξ)‖ ^ n =
          ‖characteristic2 μ ((Real.sqrt (n : ℝ))⁻¹ • ξ)‖ ^ (n - ν) *
            ‖characteristic2 μ ((Real.sqrt (n : ℝ))⁻¹ • ξ)‖ ^ ν := by
        rw [← pow_add, Nat.sub_add_cancel hn]
      rw [he]
      dsimp [f]
      exact mul_le_mul_of_nonneg_right
        (pow_le_pow_left₀ (norm_nonneg _) (hgap _ hη) _) (by positivity)
    _ ≤ ∫ ξ : ℝ × ℝ, f ξ := setIntegral_le_integral hIf (ae_of_all _ hSf)
    _ = _ := by
      dsimp [f]
      rw [integral_const_mul, integral_characteristic2_pow_scaledFrequency μ hn1 ν]
      ring

example : ∃ (ν : ℕ) (ε : ℝ), 1 ≤ ν ∧ ν ≤ ν ∧
    Integrable (fun ξ => ‖characteristic2 stdGauss2 ξ‖ ^ ν) ∧ 0 ≤ ε ∧
    ∀ ξ : ℝ × ℝ, 1 ≤ ‖ξ‖ → ‖characteristic2 stdGauss2 ξ‖ ≤ ε := by
  obtain ⟨ν, hν, hI⟩ := hasIntegrableCharFun_stdGauss2
  obtain ⟨ε, hε, _, hgap⟩ := characteristic2_gap_away_zero stdGauss2
    hasIntegrableCharFun_stdGauss2 (a := 1) one_pos
  exact ⟨ν, ε, hν, le_rfl, hI, hε.le, hgap⟩

/-- Every fixed law satisfying the source's integrable-characteristic-power
condition has an exponentially damped large-frequency integral for its
actual normalized sums. The exponent retains the power `ν` needed for
integrability. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4
`eq:big-z-exp`; no bounded-density-to-Fourier-integrability inference is used. -/
theorem exists_integral_characteristic_scaledSum_large_frequency_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ)
    {a : ℝ} (ha : 0 < a) :
    ∃ ν : ℕ, 1 ≤ ν ∧ Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν) ∧
      ∃ ε : ℝ, 0 < ε ∧ ε < 1 ∧ ∀ n : ℕ, ν ≤ n →
        (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
          ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖) ≤
            (n : ℝ) * ε ^ (n - ν) * ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν := by
  obtain ⟨ν, hν, hI⟩ := hcf
  change Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν) at hI
  obtain ⟨ε, hε0, hε1, hgap⟩ := characteristic2_gap_away_zero μ ⟨ν, hν, hI⟩ ha
  exact ⟨ν, hν, hI, ε, hε0, hε1, fun n hn =>
    integral_characteristic_scaledSum_large_frequency_of_gap μ hν hn hI hε0.le hgap⟩

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2, one_pos⟩

/-- An explicit integrable characteristic power `ν` gives bounded
continuous densities for every actual normalized sum with `n ≥ ν`.
This retains the concrete threshold implicit in the suitable condition
of arXiv:2412.09080v3, §3 `thm:br`, using the proved inversion result. -/
theorem exists_continuous_density_scaledSum_of_power_threshold
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {ν n : ℕ}
    (hν : 1 ≤ ν) (hn : ν ≤ n) (hI : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν)) :
    ∃ q, Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q ∧
      ∃ C : ℝ, ∀ z, |q z| ≤ C := by
  obtain ⟨q, hq, hnonneg, heq, hb⟩ := exists_bounded_continuous_density_of_integrable_characteristic2
    ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))
    (integrable_characteristic_scaledSum_of_power μ (hν.trans hn)
      (integrable_characteristic2_pow_of_le μ hn hI))
  exact ⟨q, hq, ⟨hq.measurable, hnonneg, heq⟩, hb⟩

example : ∃ ν : ℕ, 1 ≤ ν ∧ ν ≤ ν ∧ Integrable (fun ξ => ‖characteristic2 stdGauss2 ξ‖ ^ ν) := by
  obtain ⟨ν, hν, hI⟩ := hasIntegrableCharFun_stdGauss2
  exact ⟨ν, hν, le_rfl, hI⟩

end Transformer.Modes
