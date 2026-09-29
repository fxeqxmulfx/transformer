import Transformer.Modes.Section5_PtBddFourier
import Transformer.Modes.Section5_PtBddSmallBall

/-
# The number of modes of a Gaussian KDE — `eq:uniform-decay` fails for `β > 2`

`eq:uniform-decay` of arXiv:2412.09080v3, `|𝓕ν_t(ξ)| ≲ (1 + ‖ξ‖)^{-1/2}`, is proved in
the source by cutting `ℝ` at a *fixed* radius `R`: on `[-2R, 2R]` stationary and
non-stationary phase give `‖ξ‖^{-1/2}` uniformly in the direction, and the rest is
bounded by `∫_{|x| ≥ R} e^{-x²/2} dx`, which "as `R` is fixed, is smaller than
`(1 + ρ)^{-1/2}` whenever `ρ` is large enough".  That last step is false: a fixed
positive number is not `o(ρ^{-1/2})`.  And the tail is not negligible: both entries of
`(G(t), G'(t)) = (g, g')(t - X)` tend to `0` as `|X| → ∞`, so the far tail of `X`
does not oscillate.  For `β > 2` this breaks the estimate itself.

**What is proved.**  `G'(t) = e^{-βu²/2}(1 - βu²)`, `u = t - X`, is at most
`(2β/(β-2)) e^{-((β+2)/4) z²}` once `|u| ≥ z ≥ 1` (`abs_bigG'_le_far`), while the Gaussian
gives that tail mass `≳ e^{-z²/2 - O(z)}`: so `ℙ(|G'(t)| ≤ r) ≳ r^{2/(β+2) - o(1)}`,
which beats `O(√r)` exactly when `β > 2`.  A law whose Fourier transform decays like
`(1 + |s|)^{-1/2}` has `ℙ(|W| ≤ r) = O(√r)` (`measureReal_abs_le_of_decay`, a Fejér
kernel argument, `Section5_PtBddSmallBall.lean`), so the decay fails along `ξ = (0, s)`:
`not_uniform_decay_of_two_lt`, for every `t`.

Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay` and the sentence "as `R` is fixed,
is smaller than `(1 + ρ)^{-1/2}` whenever `ρ` is large enough".
-/

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

/-- `2 v ≤ e^v` for `v ≥ 0`. -/
theorem two_mul_le_exp {v : ℝ} (hv : 0 ≤ v) : 2 * v ≤ Real.exp v := by
  have h := Real.add_one_le_exp (v / 2)
  have h2 : Real.exp v = Real.exp (v / 2) * Real.exp (v / 2) := by rw [← Real.exp_add]; ring_nf
  nlinarith [sq_nonneg (v / 2 - 1), Real.exp_pos (v / 2)]

/-- The hypothesis of `two_mul_le_exp` is satisfiable: `v = 0`. -/
example : 2 * (0 : ℝ) ≤ Real.exp 0 := two_mul_le_exp le_rfl

/-- **`G'(t)` is small far out.**  For `β > 2`, `|G'(t)| ≤ (2β/(β-2)) e^{-((β+2)/4) z²}` as
soon as `|t - x| ≥ z ≥ 1`. -/
theorem abs_bigG'_le_far {β : ℝ} (hβ : 2 < β) (t : ℝ) {x z : ℝ} (hz : 1 ≤ z)
    (hx : z ≤ |t - x|) :
    |bigG' β t x| ≤ (2 * β / (β - 2)) * Real.exp (-((β + 2) / 4) * z ^ 2) := by
  obtain ⟨u, hu⟩ : ∃ u, u = t - x := ⟨_, rfl⟩
  rw [← hu] at hx
  have hu2 : z ^ 2 ≤ u ^ 2 := by
    have := pow_le_pow_left₀ (by linarith) hx 2
    rwa [sq_abs] at this
  have hz2 : 1 ≤ z ^ 2 := by nlinarith
  have hβ0 : 0 < β := by linarith
  have hbu : 1 ≤ β * u ^ 2 := by nlinarith
  have habs : |bigG' β t x| ≤ Real.exp (-(β / 2) * u ^ 2) * (β * u ^ 2) := by
    unfold bigG'
    rw [← hu, abs_mul, abs_of_pos (Real.exp_pos _), abs_of_nonpos (by linarith)]
    gcongr
    linarith
  have hexp : Real.exp (-(β / 2) * u ^ 2) =
      Real.exp (-((β - 2) / 4) * u ^ 2) * Real.exp (-((β + 2) / 4) * u ^ 2) := by
    rw [← Real.exp_add]; congr 1; ring
  have hkey : u ^ 2 * Real.exp (-((β - 2) / 4) * u ^ 2) ≤ 2 / (β - 2) := by
    have h1 := two_mul_le_exp (v := (β - 2) / 4 * u ^ 2) (by positivity)
    have h2 : 0 < Real.exp (-((β - 2) / 4) * u ^ 2) := Real.exp_pos _
    have h3 : Real.exp ((β - 2) / 4 * u ^ 2) * Real.exp (-((β - 2) / 4) * u ^ 2) = 1 := by
      rw [← Real.exp_add]; simp
    rw [le_div_iff₀ (by linarith)]
    nlinarith
  calc |bigG' β t x| ≤ Real.exp (-(β / 2) * u ^ 2) * (β * u ^ 2) := habs
    _ = β * (u ^ 2 * Real.exp (-((β - 2) / 4) * u ^ 2)) *
          Real.exp (-((β + 2) / 4) * u ^ 2) := by rw [hexp]; ring
    _ ≤ β * (2 / (β - 2)) * Real.exp (-((β + 2) / 4) * u ^ 2) := by gcongr
    _ ≤ β * (2 / (β - 2)) * Real.exp (-((β + 2) / 4) * z ^ 2) := by
          have h : -((β + 2) / 4) * u ^ 2 ≤ -((β + 2) / 4) * z ^ 2 := by nlinarith
          exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 h)
            (mul_nonneg hβ0.le (div_nonneg (by norm_num) (by linarith)))
    _ = _ := by ring

/-- The hypotheses of `abs_bigG'_le_far` are satisfiable: `β = 3`, `t = 0`, `x = -2`, `z = 1`. -/
example : |bigG' 3 0 (-2)| ≤ (2 * 3 / (3 - 2)) * Real.exp (-((3 + 2) / 4) * 1 ^ 2) :=
  abs_bigG'_le_far (by norm_num) 0 le_rfl (by norm_num)

/-- **The Gaussian mass of a unit interval far out on the left.**  For `t ≤ z`,
`γ([t - z - 1, t - z]) ≥ (2π)^{-1/2} e^{-(z+1-t)²/2}`. -/
theorem gaussian_Icc_lower {t z : ℝ} (hz : t ≤ z) :
    (Real.sqrt (2 * π))⁻¹ * Real.exp (-(z + 1 - t) ^ 2 / 2) ≤
      (gaussianReal 0 1).real (Set.Icc (t - z - 1) (t - z)) := by
  rw [measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero,
    ENNReal.toReal_ofReal (setIntegral_nonneg measurableSet_Icc fun x _ =>
      gaussianPDFReal_nonneg 0 1 x)]
  have hlen : (volume.real (Set.Icc (t - z - 1) (t - z))) = 1 := by
    simp [measureReal_def, Real.volume_Icc]
  calc (Real.sqrt (2 * π))⁻¹ * Real.exp (-(z + 1 - t) ^ 2 / 2)
      = ∫ _x in Set.Icc (t - z - 1) (t - z),
          (Real.sqrt (2 * π))⁻¹ * Real.exp (-(z + 1 - t) ^ 2 / 2) := by
        rw [setIntegral_const, hlen, smul_eq_mul, one_mul]
    _ ≤ ∫ x in Set.Icc (t - z - 1) (t - z), gaussianPDFReal 0 1 x := by
        refine setIntegral_mono_on (integrableOn_const (by simp [Real.volume_Icc]))
          (integrable_gaussianPDFReal 0 1).integrableOn measurableSet_Icc fun x hx => ?_
        rw [gaussianPDFReal_def]
        simp only [NNReal.coe_one, mul_one, sub_zero]
        have h1 : x ≤ 0 := by linarith [hx.2]
        have h2 : -(z + 1 - t) ≤ x := by linarith [hx.1]
        have h3 : x ^ 2 ≤ (z + 1 - t) ^ 2 := by nlinarith
        refine mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 ?_) (by positivity)
        linarith

/-- **`eq:uniform-decay` is false for `β > 2`, whatever `t`.**  No `C` bounds
`|𝓕ν_t(ξ)| ≤ C (1 + ‖ξ‖)^{-1/2}` for all `ξ ∈ ℝ²`: along `ξ = (0, s)` that would give
`ℙ(|G'(t)| ≤ r) ≤ 8C √r` (`measureReal_abs_le_of_decay`), while
`ℙ(|G'(t)| ≤ r) ≥ (2π)^{-1/2} e^{-(z+1-t)²/2}` for `r = (2β/(β-2)) e^{-((β+2)/4) z²}`
(`abs_bigG'_le_far`, `gaussian_Icc_lower`), and `((β+2)/8) z² - (z+1-t)²/2 → ∞`.

Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`; refuted for `β > 2`. -/
theorem not_uniform_decay_of_two_lt {β : ℝ} (hβ : 2 < β) (t : ℝ) :
    ¬ ∃ C : ℝ, ∀ ξ : ℝ × ℝ, ‖fourierNu β t ξ‖ ≤ C / Real.sqrt (1 + ‖ξ‖) := by
  rintro ⟨C, hC⟩
  have hβ0 : 0 < β := by linarith
  have hb2 : 0 < β - 2 := by linarith
  have hW : Measurable (bigG' β t) := by
    have hc : Continuous (bigG' β t) := by unfold bigG'; fun_prop
    exact hc.measurable
  have hC0 : 0 ≤ C := by
    have h := (norm_nonneg _).trans (hC 0)
    simpa using h
  have hdec : ∀ s : ℝ, 0 ≤ s →
      ‖∫ x, Complex.exp (-(Complex.I * ((s * bigG' β t x : ℝ) : ℂ))) ∂gaussianReal 0 1‖ ≤
        C / Real.sqrt (1 + s) := by
    intro s hs
    have h := hC (0, s)
    simpa [fourierNu, Prod.norm_def, abs_of_nonneg hs, max_eq_right hs] using h
  set K : ℝ := 2 * β / (β - 2) with hK
  have hK0 : 0 < K := div_pos (by linarith) hb2
  set q : ℝ := Real.sqrt (2 * π) with hq
  have hq0 : 0 < q := Real.sqrt_pos.2 (by positivity)
  set M : ℝ := 8 * C * Real.sqrt K * q with hM
  have hM0 : 0 ≤ M := by positivity
  set κ : ℝ := (β - 2) / 8 with hκ
  have hκ0 : 0 < κ := div_pos hb2 (by norm_num)
  set b : ℝ := |1 - t| with hb
  have hb0 : 0 ≤ b := abs_nonneg _
  obtain ⟨z, hz1, hzt, hzE⟩ : ∃ z : ℝ, 1 ≤ z ∧ t ≤ z ∧ M + b ^ 2 / 2 + b * z ≤ κ * z ^ 2 := by
    obtain ⟨z, hz⟩ : ∃ z, z = max (max 1 |t|) ((b + (M + b ^ 2 / 2)) / κ) := ⟨_, rfl⟩
    have h1 : 1 ≤ z := hz ▸ le_trans (le_max_left _ _) (le_max_left _ _)
    have h2 : t ≤ z := hz ▸ (le_abs_self t).trans ((le_max_right _ _).trans (le_max_left _ _))
    have h3 : (b + (M + b ^ 2 / 2)) / κ ≤ z := hz ▸ le_max_right _ _
    have h4 : b + (M + b ^ 2 / 2) ≤ κ * z := by rwa [div_le_iff₀ hκ0, mul_comm] at h3
    have hA : 0 ≤ M + b ^ 2 / 2 := by positivity
    refine ⟨z, h1, h2, ?_⟩
    nlinarith [mul_le_mul_of_nonneg_left h4 (le_trans zero_le_one h1),
      mul_nonneg (sub_nonneg.2 h1) hA]
  have hr0 : 0 < K * Real.exp (-((β + 2) / 4) * z ^ 2) := mul_pos hK0 (Real.exp_pos _)
  have hfro := measureReal_abs_le_of_decay (gaussianReal 0 1) hW hdec
    (a := 1 / (K * Real.exp (-((β + 2) / 4) * z ^ 2))) (one_div_pos.2 hr0)
  rw [one_div_one_div] at hfro
  have hsub : Set.Icc (t - z - 1) (t - z) ⊆
      {x | |bigG' β t x| ≤ K * Real.exp (-((β + 2) / 4) * z ^ 2)} := by
    intro x hx
    have hx' : z ≤ |t - x| := by
      rw [abs_of_nonneg (by linarith [hx.2])]
      linarith [hx.2]
    exact abs_bigG'_le_far hβ t hz1 hx'
  have h2 := (gaussian_Icc_lower hzt).trans
    ((measureReal_mono hsub (measure_ne_top _ _)).trans hfro)
  have hsqrt : 8 * C / Real.sqrt (1 / (K * Real.exp (-((β + 2) / 4) * z ^ 2))) =
      8 * C * (Real.sqrt K * Real.exp (-((β + 2) / 4) * z ^ 2 / 2)) := by
    rw [one_div, Real.sqrt_inv, div_inv_eq_mul, Real.sqrt_mul hK0.le, Real.exp_half]
  rw [hsqrt, inv_mul_le_iff₀ hq0] at h2
  have hexp : Real.exp (((β + 2) / 4 * z ^ 2) / 2) *
      Real.exp (-((β + 2) / 4) * z ^ 2 / 2) = 1 := by
    rw [← Real.exp_add]; ring_nf; simp
  have hE : Real.exp (((β + 2) / 4 * z ^ 2) / 2 - (z + 1 - t) ^ 2 / 2) ≤ M := by
    have hsplit : Real.exp (((β + 2) / 4 * z ^ 2) / 2 - (z + 1 - t) ^ 2 / 2) =
        Real.exp (-(z + 1 - t) ^ 2 / 2) * Real.exp (((β + 2) / 4 * z ^ 2) / 2) := by
      rw [← Real.exp_add]; congr 1; ring
    rw [hsplit]
    calc Real.exp (-(z + 1 - t) ^ 2 / 2) * Real.exp (((β + 2) / 4 * z ^ 2) / 2)
        ≤ (q * (8 * C * (Real.sqrt K * Real.exp (-((β + 2) / 4) * z ^ 2 / 2)))) *
            Real.exp (((β + 2) / 4 * z ^ 2) / 2) :=
          mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
      _ = M := by rw [hM]; linear_combination (8 * C * Real.sqrt K * q) * hexp
  have hEge : M ≤ ((β + 2) / 4 * z ^ 2) / 2 - (z + 1 - t) ^ 2 / 2 := by
    have hb2' : (1 - t) ^ 2 = b ^ 2 := (sq_abs _).symm
    have hbz : (1 - t) * z ≤ b * z := mul_le_mul_of_nonneg_right (le_abs_self _) (by linarith)
    nlinarith [hzE, hbz, hb2']
  linarith [Real.add_one_le_exp (((β + 2) / 4 * z ^ 2) / 2 - (z + 1 - t) ^ 2 / 2)]

/-- The hypothesis of `gaussian_Icc_lower` is satisfiable: `t = z = 0`. -/
example : (Real.sqrt (2 * π))⁻¹ * Real.exp (-(0 + 1 - 0) ^ 2 / 2) ≤
    (gaussianReal 0 1).real (Set.Icc (0 - 0 - 1) (0 - 0)) := gaussian_Icc_lower le_rfl

/-- The hypothesis of `not_uniform_decay_of_two_lt` is satisfiable: `β = 3`, `t = 0`. -/
example : ¬ ∃ C : ℝ, ∀ ξ : ℝ × ℝ, ‖fourierNu 3 0 ξ‖ ≤ C / Real.sqrt (1 + ‖ξ‖) :=
  not_uniform_decay_of_two_lt (by norm_num) 0

end Modes
end Transformer
