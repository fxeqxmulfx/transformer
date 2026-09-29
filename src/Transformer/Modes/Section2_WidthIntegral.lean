/-
# The number of modes of a Gaussian KDE — the width of the integral over `T'`

The last step of `lem:main-int-phi` on `T'`.  On `T'`, write
`R(t) = β^{-3/2} n e^{-t²/2} ≥ 1` (`one_le_rateScale`).  The integrand of the
proxy Kac–Rice integral is bounded by `√β` times
`e^{-κ R t²}(1 + K √R (4 + 6t²))` (`Section2_ProxyKRTPrime.lean`): the factor
`√R` is the size of the shift `√(α_t δ_t²)` of the Gaussian in `y`, which is not
bounded on `T'` for `n ≫ β^{5/2}`.  It is paid for by the width of `e^{-κ R t²}`
in `t`, which is `R^{-1/2}`: near `t = 0`, where `R ≈ n β^{-3/2}`, the integral
of `√R e^{-κ R t²}` is bounded by a constant, and for `|t| ≥ 1` the factor
`√R` is absorbed by `e^{-κ R t²}` since `R t² ≥ R`
(`width_integrand_le`, `Section2_WidthTerms.lean`).

The bound is a uniform constant, for all `n ≥ 1` and `β ≥ 1`: no regime is
involved.

Source: arXiv:2412.09080v3, `eq:int-phi-b` and the proof of `lem:main-int-phi`.
-/

import Transformer.Modes.Section2_IntervalTPrime
import Transformer.Modes.Section2_WidthTerms

open Real MeasureTheory
open scoped ENNReal

namespace Transformer
namespace Modes

/-- **The integral over `T'` of the width majorant is bounded by a constant**, uniformly
in `n ≥ 1`, `β ≥ 1`: the `√ρ` of the near-zero part is cancelled by the width
`√(π/(κρ/2))` of its Gaussian.

Source: arXiv:2412.09080v3, `eq:int-phi-b`, proof of `lem:main-int-phi`. -/
theorem lintegral_width_le {κ K₁ : ℝ} (hκ : 0 < κ) (hK₁ : 0 ≤ K₁) :
    ∃ K : ℝ, 0 ≤ K ∧ ∀ {n : ℕ} {β : ℝ}, 1 ≤ n → 1 ≤ β →
      ∫⁻ t in intervalT' n β, ENNReal.ofReal
        (Real.exp (-κ * (rateScale n β t * t ^ 2)) *
          (1 + K₁ * Real.sqrt (rateScale n β t) * (4 + 6 * t ^ 2))) ≤ ENNReal.ofReal K := by
  refine ⟨Real.sqrt (π / κ) + 10 * K₁ * Real.sqrt (π / (κ / 2)) +
    K₁ * (2 / κ) * (4 + 24 / κ) * Real.sqrt (π / (κ / 4)), by positivity, ?_⟩
  intro n β hn hβ
  have hβ0 : 0 < β := by linarith
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  obtain ⟨ρ, hρ_def⟩ : ∃ ρ : ℝ, ρ = β ^ (-(3 : ℝ) / 2) * n := ⟨_, rfl⟩
  have hρ : 0 < ρ := by rw [hρ_def]; positivity
  have hrs (t : ℝ) : rateScale n β t = ρ * Real.exp (-(t ^ 2) / 2) := by
    rw [hρ_def]; unfold rateScale; ring
  let M : ℝ → ℝ := fun t =>
    Real.exp (-κ * t ^ 2) + (10 * K₁ * Real.sqrt ρ) * Real.exp (-(κ / 2 * ρ) * t ^ 2) +
      (K₁ * (2 / κ) * (4 + 24 / κ)) * Real.exp (-(κ / 4) * t ^ 2)
  have h1 : Integrable fun t : ℝ => Real.exp (-κ * t ^ 2) := integrable_exp_neg_mul_sq hκ
  have h2 : Integrable fun t : ℝ => (10 * K₁ * Real.sqrt ρ) * Real.exp (-(κ / 2 * ρ) * t ^ 2) :=
    (integrable_exp_neg_mul_sq (by positivity)).const_mul _
  have h3 : Integrable fun t : ℝ => (K₁ * (2 / κ) * (4 + 24 / κ)) * Real.exp (-(κ / 4) * t ^ 2) :=
    (integrable_exp_neg_mul_sq (by positivity)).const_mul _
  have hMi : Integrable M := (h1.add h2).add h3
  have hM0 : ∀ t, 0 ≤ M t := fun t => by positivity
  have hint : ∫ t, M t = Real.sqrt (π / κ) + 10 * K₁ * Real.sqrt (π / (κ / 2)) +
      K₁ * (2 / κ) * (4 + 24 / κ) * Real.sqrt (π / (κ / 4)) := by
    change ∫ t, (Real.exp (-κ * t ^ 2) + (10 * K₁ * Real.sqrt ρ) * Real.exp (-(κ / 2 * ρ) * t ^ 2) +
      (K₁ * (2 / κ) * (4 + 24 / κ)) * Real.exp (-(κ / 4) * t ^ 2)) = _
    rw [integral_add (f := fun t : ℝ => Real.exp (-κ * t ^ 2) +
        (10 * K₁ * Real.sqrt ρ) * Real.exp (-(κ / 2 * ρ) * t ^ 2))
      (g := fun t : ℝ => (K₁ * (2 / κ) * (4 + 24 / κ)) * Real.exp (-(κ / 4) * t ^ 2))
      (h1.add h2) h3, integral_add h1 h2, integral_const_mul, integral_const_mul,
      integral_gaussian, integral_gaussian, integral_gaussian]
    have hsq : Real.sqrt ρ * Real.sqrt (π / (κ / 2 * ρ)) = Real.sqrt (π / (κ / 2)) := by
      rw [← Real.sqrt_mul hρ.le]
      congr 1
      field_simp
    rw [mul_assoc (10 * K₁), hsq]
  calc ∫⁻ t in intervalT' n β, ENNReal.ofReal
        (Real.exp (-κ * (rateScale n β t * t ^ 2)) *
          (1 + K₁ * Real.sqrt (rateScale n β t) * (4 + 6 * t ^ 2)))
      ≤ ∫⁻ t in intervalT' n β, ENNReal.ofReal (M t) := by
        refine setLIntegral_mono' (measurableSet_intervalT' n β) fun t ht => ?_
        refine ENNReal.ofReal_le_ofReal ?_
        have hR : 1 ≤ ρ * Real.exp (-(t ^ 2) / 2) := hrs t ▸ one_le_rateScale hn hβ ht
        have := width_integrand_le hκ hK₁ (sq_nonneg t) hR
        rw [hrs t]
        exact this
    _ ≤ ∫⁻ t, ENNReal.ofReal (M t) := setLIntegral_le_lintegral _ _
    _ = ENNReal.ofReal (∫ t, M t) :=
        (ofReal_integral_eq_lintegral_ofReal hMi (Filter.Eventually.of_forall hM0)).symm
    _ = _ := by rw [hint]

/-- The hypotheses of `lintegral_width_le` are satisfiable, with `T'` nonempty:
`n = 8`, `β = 3`. -/
example : ∃ K : ℝ, ∫⁻ t in intervalT' 8 (3 : ℝ), ENNReal.ofReal
    (Real.exp (-(1 : ℝ) * (rateScale 8 3 t * t ^ 2)) *
      (1 + 0 * Real.sqrt (rateScale 8 3 t) * (4 + 6 * t ^ 2))) ≤ ENNReal.ofReal K ∧
    (0 : ℝ) ∈ intervalT' 8 (3 : ℝ) := by
  obtain ⟨K, -, hK⟩ := lintegral_width_le (κ := 1) (K₁ := 0) one_pos le_rfl
  refine ⟨K, hK (by norm_num) (by norm_num), ?_⟩
  unfold intervalT'
  have h : (3 : ℝ) ≤ ((8 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) := by
    have h8 : ((8 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) = 4 := by
      rw [Nat.cast_ofNat, show (8 : ℝ) = 2 ^ (3 : ℝ) by norm_num,
        ← Real.rpow_mul (by norm_num)]
      norm_num
    linarith
  simp only [h, ite_true]
  exact ⟨by simp, Real.sqrt_nonneg _⟩

end Modes
end Transformer
