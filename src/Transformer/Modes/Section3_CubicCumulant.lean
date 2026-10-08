import Transformer.Modes.Section3_CharacteristicCubic
import Transformer.Modes.Section3_CumulantBounds
/-!
# The actual cubic phase coefficient and the third cumulants

The correction `ψ` of arXiv:2412.09080v3, §3.1 `eq:psi`, uses the
third mixed derivatives of the actual logarithmic moment generating
function. Under the source's standardization and exponential moments,
these derivatives equal the centered third mixed moments, as proved
in `Section3_CumulantMoment`.

Expanding the literal third moment of a linear form then gives exactly
`E⟨ξ,X⟩³ / 6 = Σ κ^(k,3-k) ξ₁^k ξ₂^(3-k) / (k!(3-k)!)`.
This identifies the characteristic cubic coefficient with the same
cumulants appearing in the source's density correction. It does not
replace those cumulants by a new definition or assume their equality.

The actual frequency-dependent moment is a smooth cubic polynomial.
Its homogeneity retains the source's normalization: `n (1/√n)³ = 1/√n`
for positive sample counts. These identities supply the cubic input
before Fourier inversion and the spatially weighted third-order error
estimate of §5.4 `eq:higher-error-goal`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual third logarithmic cumulant equals its centered mixed moment.
Source: arXiv:2412.09080v3, §3.1, the moment identity after `eq:psi`.
Exponential moments justify the literal logarithmic derivatives. -/
theorem cumulant_three_eq_integral_monomial
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (k : ℕ) (hk : k ≤ 3) :
    cumulantOf μ k (3 - k) = ∫ z, z.1 ^ k * z.2 ^ (3 - k) ∂μ := by
  have h := cumulant_three_eq_integral_hermite μ hμ hexp k hk
  obtain ⟨ε, hε, hE⟩ := hexp
  exact h.trans (integral_hermite3_eq hE hε hμ k hk)

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ 1 ≤ (3 : ℕ) :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2, by norm_num⟩

/-- The actual third linear-form moment is its explicit mixed-moment polynomial.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the third-order characteristic coefficient. All four mixed moments are integrable. -/
theorem integral_dot_cube_eq_mixed_moments (μ : Measure (ℝ × ℝ))
    (hexp : HasExpMoments μ) (ξ : ℝ × ℝ) :
    (∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ) =
      ξ.1 ^ 3 * (∫ z, z.1 ^ 3 ∂μ) + 3 * ξ.1 ^ 2 * ξ.2 * (∫ z, z.1 ^ 2 * z.2 ∂μ) +
        3 * ξ.1 * ξ.2 ^ 2 * (∫ z, z.1 * z.2 ^ 2 ∂μ) + ξ.2 ^ 3 * (∫ z, z.2 ^ 3 ∂μ) := by
  obtain ⟨ε, hε, hE⟩ := hexp
  have h30 : Integrable (fun z : ℝ × ℝ => z.1 ^ 3) μ := by
    simpa only [pow_zero, mul_one] using integrable_pow_mul_pow hE hε 3 0
  have h21 : Integrable (fun z : ℝ × ℝ => z.1 ^ 2 * z.2) μ := by
    simpa only [pow_one] using integrable_pow_mul_pow hE hε 2 1
  have h12 : Integrable (fun z : ℝ × ℝ => z.1 * z.2 ^ 2) μ := by
    simpa only [pow_one] using integrable_pow_mul_pow hE hε 1 2
  have h03 : Integrable (fun z : ℝ × ℝ => z.2 ^ 3) μ := by
    simpa only [pow_zero, one_mul] using integrable_pow_mul_pow hE hε 0 3
  have he : (fun z : ℝ × ℝ => (ξ.1 * z.1 + ξ.2 * z.2) ^ 3) = fun z =>
      ξ.1 ^ 3 * z.1 ^ 3 + 3 * ξ.1 ^ 2 * ξ.2 * (z.1 ^ 2 * z.2) +
        3 * ξ.1 * ξ.2 ^ 2 * (z.1 * z.2 ^ 2) + ξ.2 ^ 3 * z.2 ^ 3 := by
    funext z
    ring
  have hA := h30.const_mul (ξ.1 ^ 3)
  have hB := h21.const_mul (3 * ξ.1 ^ 2 * ξ.2)
  have hC := h12.const_mul (3 * ξ.1 * ξ.2 ^ 2)
  have hD := h03.const_mul (ξ.2 ^ 3)
  rw [he, integral_add
    (f := fun z : ℝ × ℝ => ξ.1 ^ 3 * z.1 ^ 3 + 3 * ξ.1 ^ 2 * ξ.2 * (z.1 ^ 2 * z.2) +
      3 * ξ.1 * ξ.2 ^ 2 * (z.1 * z.2 ^ 2)) ((hA.add hB).add hC) hD,
    integral_add
      (f := fun z : ℝ × ℝ => ξ.1 ^ 3 * z.1 ^ 3 + 3 * ξ.1 ^ 2 * ξ.2 * (z.1 ^ 2 * z.2))
      (hA.add hB) hC,
    integral_add hA hB, integral_const_mul, integral_const_mul, integral_const_mul, integral_const_mul]

example : HasExpMoments stdGauss2 := hasExpMoments_stdGauss2

/-- The cubic phase coefficient uses exactly the cumulants and factorials of `ψ`.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`.
The equality is between the actual moment and actual logarithmic cumulants. -/
theorem integral_dot_cube_div_six_eq_cumulant_polynomial
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (ξ : ℝ × ℝ) :
    (∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ) / 6 =
      ∑ k ∈ Finset.range 4, cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * ξ.1 ^ k * ξ.2 ^ (3 - k) := by
  have h0 := cumulant_three_eq_integral_monomial μ hμ hexp 0 (by omega)
  have h1 := cumulant_three_eq_integral_monomial μ hμ hexp 1 (by omega)
  have h2 := cumulant_three_eq_integral_monomial μ hμ hexp 2 (by omega)
  have h3 := cumulant_three_eq_integral_monomial μ hμ hexp 3 (by omega)
  norm_num at h0 h1 h2 h3
  rw [integral_dot_cube_eq_mixed_moments μ hexp ξ, ← h0, ← h1, ← h2, ← h3]
  norm_num [Finset.sum_range_succ]
  ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2⟩

example : IsProbabilityMeasure (lawY 3 0) ∧ IsStandardized (lawY 3 0) ∧
    HasExpMoments (lawY 3 0) :=
  ⟨inferInstance, isStandardized_lawY (by norm_num : (0 : ℝ) < 3) 0,
    hasExpMoments_lawY (by norm_num : (0 : ℝ) < 3) 0⟩

/-- The actual frequency-dependent third moment is a smooth cubic polynomial.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the differentiated
cubic correction; the proved mixed-moment identity justifies every derivative. -/
theorem contDiff_integral_dot_cube (μ : Measure (ℝ × ℝ)) (hexp : HasExpMoments μ) :
    ContDiff ℝ ⊤ (fun ξ : ℝ × ℝ => ∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ) := by
  have he : (fun ξ : ℝ × ℝ => ∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ) = fun ξ =>
      ξ.1 ^ 3 * (∫ z, z.1 ^ 3 ∂μ) + 3 * ξ.1 ^ 2 * ξ.2 * (∫ z, z.1 ^ 2 * z.2 ∂μ) +
        3 * ξ.1 * ξ.2 ^ 2 * (∫ z, z.1 * z.2 ^ 2 ∂μ) + ξ.2 ^ 3 * (∫ z, z.2 ^ 3 ∂μ) :=
    funext (integral_dot_cube_eq_mixed_moments μ hexp)
  rw [he]
  fun_prop

example : HasExpMoments stdGauss2 := hasExpMoments_stdGauss2

/-- Scaling the actual frequency scales its third moment by `c³`.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4, the normalized
characteristic argument. The identity retains the literal frequency scale. -/
theorem integral_dot_cube_smul_frequency (μ : Measure (ℝ × ℝ)) (c : ℝ) (ξ : ℝ × ℝ) :
    (∫ z, ((c • ξ).1 * z.1 + (c • ξ).2 * z.2) ^ 3 ∂μ) =
      c ^ 3 * ∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ := by
  have he : (fun z : ℝ × ℝ => ((c • ξ).1 * z.1 + (c • ξ).2 * z.2) ^ 3) =
      fun z => c ^ 3 * (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 := by
    funext z
    change (c * ξ.1 * z.1 + c * ξ.2 * z.2) ^ 3 = _
    ring
  rw [he, integral_const_mul]

/-- The normalized-sum cubic coefficient has the exact rate `1 / sqrt n`.
Source: arXiv:2412.09080v3, §3 `thm:br` and §3.1 `eq:psi`,
the third-order phase normalization, with positive sample counts explicit. -/
theorem normalized_cubic_moment (μ : Measure (ℝ × ℝ)) {n : ℕ} (hn : 1 ≤ n) (ξ : ℝ × ℝ) :
    (n : ℝ) * (∫ z, (((Real.sqrt n)⁻¹ • ξ).1 * z.1 + ((Real.sqrt n)⁻¹ • ξ).2 * z.2) ^ 3 ∂μ) =
      (Real.sqrt n)⁻¹ * ∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : Real.sqrt (n : ℝ) ≠ 0 := (Real.sqrt_pos.mpr hn0).ne'
  rw [integral_dot_cube_smul_frequency]
  have hscale : (n : ℝ) * ((Real.sqrt n)⁻¹) ^ 3 = (Real.sqrt n)⁻¹ := by
    have hsq := Real.sq_sqrt hn0.le
    field_simp
    nlinarith [hsq]
  rw [← mul_assoc, hscale]

example : 1 ≤ (1 : ℕ) := le_rfl

end Transformer.Modes
