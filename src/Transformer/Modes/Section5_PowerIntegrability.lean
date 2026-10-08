import Transformer.Modes.Section5_FractionalFourier
import Transformer.Modes.Section3_BR
import Mathlib.Analysis.SpecialFunctions.JapaneseBracket

/-!
# Integrable Fourier powers for every positive bandwidth

In §5.5 of arXiv:2412.09080v3, the source deduces integrability of the
`n`th power from its claimed inverse square root decay when `n > 4`.
That decay fails for `β > 2`. `Section5_FractionalFourier` instead proves
exponent `1 / (2(β + 1))` for every `β > 0`, with the shifted Gaussian
weight and dependence on fixed `t` retained.

This module transfers that genuine estimate to integrability whenever
`n > 4(β + 1)`. It first proves continuity of the original expectation
by dominated convergence, using the unit modulus of its integrand.
The generic power estimate then dominates the continuous norm power
by an integrable radial function on the actual two-dimensional space.
The result is also stated as finiteness of the nonnegative integral.

The characteristic function convention in §3 uses a positive sign,
whereas `fourierNu` uses a negative sign. The exact pushforward integral
identity relates them at opposite frequencies. Reflection preserves
integrability, so some positive integer power of the characteristic
function of the actual Gaussian curve law is integrable for every
positive bandwidth. This supplies the `HasIntegrableCharFun` condition
of the Bhattacharya–Rao statements for that law.

The number of powers is allowed to depend on bandwidth. This repairs
the source's use of five summands in §5.4: the integrability input is
proved for a sufficiently large power, rather than deduced merely from
boundedness of a density. No density existence is assumed in this proof.
Source: arXiv:2412.09080v3, §5.4 and §5.5, after `eq:uniform-decay`.
-/

open Real MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Transformer.Modes

/-- Continuity of the original Gaussian Fourier expectation follows from
its unit-modulus integrand. Source: arXiv:2412.09080v3, §5.5, Fourier display. -/
theorem continuous_fourierNu (β t : ℝ) : Continuous (fourierNu β t) := by
  unfold fourierNu
  apply continuous_of_dominated (bound := fun _ => (1 : ℝ))
  · intro ξ
    have hc : Continuous fun x : ℝ =>
        Complex.exp (-(Complex.I * ((ξ.1 * bigG β t x + ξ.2 * bigG' β t x : ℝ) : ℂ))) := by
      unfold bigG bigG'
      fun_prop
    exact hc.aestronglyMeasurable
  · exact fun ξ => ae_of_all _ fun x => by simp [Complex.norm_exp]
  · exact integrable_const _
  · exact ae_of_all _ fun x => by fun_prop

/-- On `ℝ²`, a continuous Fourier transform with fractional decay has
integrable `n`th norm power once `αn > 2`.
Source: arXiv:2412.09080v3, §5.5, the display after `eq:uniform-decay`. -/
theorem integrable_pow_of_fractional_decay {F : ℝ × ℝ → ℂ} (hcont : Continuous F)
    {C α : ℝ} (hF : ∀ ξ, ‖F ξ‖ ≤ C / (1 + ‖ξ‖) ^ α)
    {n : ℕ} (hn : 2 < α * (n : ℝ)) : Integrable (fun ξ => ‖F ξ‖ ^ n) := by
  have hint : Integrable (fun ξ : ℝ × ℝ => C ^ n * (1 + ‖ξ‖) ^ (-(α * (n : ℝ)))) := by
    refine (integrable_one_add_norm ?_).const_mul _
    simpa [Module.finrank_prod, Module.finrank_self] using hn
  apply hint.mono' (hcont.norm.pow n).aestronglyMeasurable
  exact ae_of_all _ fun ξ => by
    change ‖‖F ξ‖ ^ n‖ ≤ _
    rw [Real.norm_eq_abs, abs_of_nonneg (pow_nonneg (norm_nonneg _) n)]
    have ha : 0 < 1 + ‖ξ‖ := by positivity
    refine (pow_le_pow_left₀ (norm_nonneg _) (hF ξ) n).trans_eq ?_
    rw [div_pow, ← Real.rpow_mul_natCast ha.le, Real.rpow_neg ha.le, div_eq_mul_inv]

example : Continuous (fun _ : ℝ × ℝ => (0 : ℂ)) ∧
    (∀ ξ : ℝ × ℝ, ‖(0 : ℂ)‖ ≤ 0 / (1 + ‖ξ‖) ^ (1 : ℝ)) ∧
    (2 : ℝ) < 1 * (3 : ℕ) := by
  exact ⟨continuous_const, fun _ => by simp, by norm_num⟩

/-- The actual Fourier transform has an integrable norm power for every
`β > 0` when `n > 4(β + 1)`. The source uses `n > 4` from the false
global square root rate; this explicit threshold repairs that argument.
Source: arXiv:2412.09080v3, §5.5, after `eq:uniform-decay`. -/
theorem integrable_fourierNu_pow_all_bandwidths {β : ℝ} (hβ : 0 < β) (t : ℝ)
    {n : ℕ} (hn : 4 * (β + 1) < (n : ℝ)) : Integrable (fun ξ => ‖fourierNu β t ξ‖ ^ n) := by
  obtain ⟨C, _, hC⟩ := fourierNu_decay_all_bandwidths hβ t
  apply integrable_pow_of_fractional_decay (continuous_fourierNu β t) hC
  have hden : 0 < 2 * (β + 1) := by positivity
  rw [one_div_mul_eq_div, lt_div_iff₀ hden]
  nlinarith

example : (0 : ℝ) < 3 ∧ 4 * (3 + 1) < ((17 : ℕ) : ℝ) := by norm_num

/-- The positive-sign characteristic function of the actual curve law
is `fourierNu` at the opposite frequency. No density assumption is needed.
Source: arXiv:2412.09080v3, §3, `thm:br`, and §5.5, definition of `ν_t`. -/
theorem characteristic_gaussian_curve (β t : ℝ) (ξ : ℝ × ℝ) :
    (∫ z : ℝ × ℝ, Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I)
      ∂(gaussianReal 0 1).map (fun x => (bigG β t x, bigG' β t x))) =
      fourierNu β t (-ξ) := by
  have hm : Measurable fun x => (bigG β t x, bigG' β t x) := by
    unfold bigG bigG'
    fun_prop
  have hc : Continuous fun z : ℝ × ℝ =>
      Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) := by fun_prop
  rw [integral_map hm.aemeasurable hc.aestronglyMeasurable]
  unfold fourierNu
  apply integral_congr_ae
  exact ae_of_all _ fun x => by
    apply congrArg Complex.exp
    simp only [Prod.fst_neg, Prod.snd_neg, Complex.ofReal_add, Complex.ofReal_mul,
      Complex.ofReal_neg]
    ring

/-- Every positive bandwidth meets the integrable characteristic power
condition for the actual Gaussian curve law, with a bandwidth-dependent
number of powers. Source: arXiv:2412.09080v3, §3, `thm:br`, and §5.4–§5.5. -/
theorem hasIntegrableCharFun_gaussian_curve {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    HasIntegrableCharFun ((gaussianReal 0 1).map (fun x => (bigG β t x, bigG' β t x))) := by
  obtain ⟨n, hn⟩ := exists_nat_gt (4 * (β + 1))
  have hnpos : 0 < n := by
    have : (0 : ℝ) < (n : ℝ) := by linarith
    exact_mod_cast this
  refine ⟨n, hnpos, ?_⟩
  have hint := (integrable_fourierNu_pow_all_bandwidths hβ t hn).comp_neg
  simpa only [characteristic_gaussian_curve] using hint

example : (0 : ℝ) < 3 := by norm_num

/-- Finiteness of the actual Fourier norm power as a nonnegative integral,
with the corrected bandwidth-dependent threshold.
Source: arXiv:2412.09080v3, §5.5, after `eq:uniform-decay`. -/
theorem lintegral_fourierNu_pow_all_bandwidths {β : ℝ} (hβ : 0 < β) (t : ℝ)
    {n : ℕ} (hn : 4 * (β + 1) < (n : ℝ)) :
    ∫⁻ ξ : ℝ × ℝ, ‖fourierNu β t ξ‖ₑ ^ n < ∞ := by
  have h := (integrable_fourierNu_pow_all_bandwidths hβ t hn).lintegral_lt_top
  simpa only [ENNReal.ofReal_pow (norm_nonneg _), ofReal_norm] using h

example : (0 : ℝ) < 3 ∧ 4 * (3 + 1) < ((17 : ℕ) : ℝ) := by norm_num

/-- A finite positive power is always available, without restricting
positive bandwidths to `β < 2`. Source: arXiv:2412.09080v3, §5.4–§5.5,
the integrability condition for the density expansion. -/
theorem exists_integrable_fourierNu_pow {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    ∃ n : ℕ, 1 ≤ n ∧ Integrable (fun ξ => ‖fourierNu β t ξ‖ ^ n) := by
  obtain ⟨n, hn⟩ := exists_nat_gt (4 * (β + 1))
  have hnpos : 0 < n := by
    have : (0 : ℝ) < (n : ℝ) := by linarith
    exact_mod_cast this
  exact ⟨n, hnpos, integrable_fourierNu_pow_all_bandwidths hβ t hn⟩

example : (0 : ℝ) < 3 := by norm_num

end Transformer.Modes
