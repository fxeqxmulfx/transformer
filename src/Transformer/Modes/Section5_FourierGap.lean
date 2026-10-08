import Transformer.Modes.Section5_PowerIntegrability
import Transformer.Modes.Section3_YCharFun
import Mathlib.Analysis.Convex.Integral
import Mathlib.Analysis.InnerProductSpace.Convex

/-!
# A strict characteristic-function bound away from zero

The large-frequency argument in §5.4 of arXiv:2412.09080v3 uses a
characteristic-function bound strictly below one outside a positive
frequency radius. This module proves that property for the actual
Gaussian curve law at every positive bandwidth and fixed translation.

Strict convexity of the complex norm says that a unit-bounded integrand
whose expectation has norm one is constant almost everywhere. For an
oscillatory exponential this forces the Fourier transform at every
natural multiple of the frequency to equal the corresponding power of
its value. Those powers keep norm one. At a nonzero frequency, this
contradicts the all-bandwidth fractional decay already proved in
`Section5_FractionalFourier`. Thus every nonzero frequency has norm
strictly less than one.

For a fixed positive radius, the decay bounds sufficiently large
frequencies by one half. Continuity and compactness supply an attained
maximum below one on the remaining annulus. Their maximum gives a
single positive bound below one on all frequencies outside the radius.
The bound depends on `β`, fixed `t`, and the radius; uniformity in a
varying bandwidth or translation is not asserted.

The explicit whitening frequency map sends nonzero frequencies to
nonzero frequencies for a positive definite covariance. The final
norm identity therefore gives the same pointwise strict inequality
for the actual standardized law `lawY β t` used in §3.
Source: arXiv:2412.09080v3, §5.4, definition of `ε`, and §3 `thm:br`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped Topology

namespace Transformer.Modes

/-- Norm one forces the oscillatory exponential to be almost surely
constant, so every natural frequency multiple gives the same power.
Source: arXiv:2412.09080v3, §5.4, justifying the strict bound defining `ε`. -/
theorem fourierNu_nat_mul_eq_pow_of_norm_eq_one (β t : ℝ) (ξ : ℝ × ℝ)
    (hξ : ‖fourierNu β t ξ‖ = 1) (n : ℕ) :
    fourierNu β t ((n : ℝ) • ξ) = fourierNu β t ξ ^ n := by
  let f : ℝ → ℂ := fun x =>
    Complex.exp (-(Complex.I * ((ξ.1 * bigG β t x + ξ.2 * bigG' β t x : ℝ) : ℂ)))
  have hbound : ∀ᵐ x ∂gaussianReal 0 1, ‖f x‖ ≤ 1 :=
    ae_of_all _ fun x => by dsimp [f]; simp [Complex.norm_exp]
  rcases ae_eq_const_or_norm_integral_lt_of_norm_le_const hbound with hc | hs
  · have he : f =ᵐ[gaussianReal 0 1] fun _ => fourierNu β t ξ := by
      filter_upwards [hc] with x hx
      change f x = (⨍ y, f y ∂gaussianReal 0 1) at hx
      rw [average_eq_integral] at hx
      exact hx
    have hphase (x : ℝ) :
        Complex.exp (-(Complex.I * ((((n : ℝ) • ξ).1 * bigG β t x +
          ((n : ℝ) • ξ).2 * bigG' β t x : ℝ) : ℂ))) = f x ^ n := by
      rw [← Complex.exp_nat_mul]
      apply congrArg Complex.exp
      simp only [Prod.smul_fst, Prod.smul_snd, smul_eq_mul, Complex.ofReal_add,
        Complex.ofReal_mul, Complex.ofReal_natCast]
      ring
    unfold fourierNu
    calc _ = ∫ x, f x ^ n ∂gaussianReal 0 1 := integral_congr_ae (ae_of_all _ hphase)
      _ = ∫ _ : ℝ, fourierNu β t ξ ^ n ∂gaussianReal 0 1 :=
        integral_congr_ae (he.fun_comp (fun z => z ^ n))
      _ = _ := by simp [fourierNu]
  · have hlt : ‖fourierNu β t ξ‖ < 1 := by simpa only [probReal_univ, one_mul, f, fourierNu] using hs
    exact False.elim (by linarith)

example : ‖fourierNu 1 0 ((0, 0) : ℝ × ℝ)‖ = 1 := by simp [fourierNu]

/-- A positive fractional exponent makes the whole frequency envelope
converge to zero. Source: arXiv:2412.09080v3, §5.5, the corrected decay
used in the large-frequency argument of §5.4. -/
theorem tendsto_fractional_frequency_envelope_zero {α : ℝ} (hα : 0 < α) (C : ℝ) :
    Tendsto (fun r : ℝ => C / (1 + r) ^ α) atTop (𝓝 0) := by
  have hbase : Tendsto (fun r : ℝ => 1 + r) atTop atTop :=
    tendsto_atTop_add_const_left _ _ tendsto_id
  have h := (tendsto_inv_atTop_zero.comp ((tendsto_rpow_atTop hα).comp hbase)).const_mul C
  simpa only [Function.comp_def, div_eq_mul_inv, mul_zero] using h

example : (0 : ℝ) < 1 := one_pos

/-- The actual Fourier expectation has norm strictly below one at each
nonzero frequency when `β > 0`; norm one would contradict decay along
natural frequency multiples. Source: arXiv:2412.09080v3, §5.4, definition of `ε`. -/
theorem norm_fourierNu_lt_one {β : ℝ} (hβ : 0 < β) (t : ℝ) {ξ : ℝ × ℝ}
    (hξ : 0 < ‖ξ‖) : ‖fourierNu β t ξ‖ < 1 := by
  by_contra hlt
  have heq : ‖fourierNu β t ξ‖ = 1 := le_antisymm (norm_fourierNu_le_one β t ξ) (not_lt.mp hlt)
  obtain ⟨C, _, hC⟩ := fourierNu_decay_all_bandwidths hβ t
  have hN : Tendsto (fun n : ℕ => ‖(n : ℝ) • ξ‖) atTop atTop := by
    have h := Tendsto.atTop_mul_const hξ
      (tendsto_natCast_atTop_atTop : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop)
    convert h using 1
    funext n
    rw [norm_smul, Real.norm_of_nonneg (Nat.cast_nonneg _)]
  have hα : 0 < 1 / (2 * (β + 1)) := by positivity
  have hd := (tendsto_fractional_frequency_envelope_zero hα C).comp hN
  have hle : 1 ≤ (0 : ℝ) := ge_of_tendsto' hd fun n => by
    have hp := fourierNu_nat_mul_eq_pow_of_norm_eq_one β t ξ heq n
    have hh := hC ((n : ℝ) • ξ)
    rw [hp, norm_pow, heq, one_pow] at hh
    exact hh
  linarith

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < ‖((1, 0) : ℝ × ℝ)‖ := by
  norm_num [Prod.norm_def, Real.norm_eq_abs]

/-- Outside any fixed positive radius, the actual curve law has a single
positive Fourier norm bound below one. The bound retains dependence on
bandwidth, translation, and radius.
Source: arXiv:2412.09080v3, §5.4, the strict bound defining `ε`. -/
theorem fourierNu_gap_away_zero {β : ℝ} (hβ : 0 < β) (t : ℝ) {a : ℝ} (ha : 0 < a) :
    ∃ ε : ℝ, 0 < ε ∧ ε < 1 ∧ ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖fourierNu β t ξ‖ ≤ ε := by
  obtain ⟨C, _, hC⟩ := fourierNu_decay_all_bandwidths hβ t
  have hα : 0 < 1 / (2 * (β + 1)) := by positivity
  have ht := (tendsto_fractional_frequency_envelope_zero hα C).eventually_lt_const
    (by norm_num : (0 : ℝ) < 1 / 2)
  obtain ⟨R, hR⟩ := eventually_atTop.mp ht
  let S : Set (ℝ × ℝ) := {ξ | a ≤ ‖ξ‖} ∩ Metric.closedBall 0 (max R a)
  have hS : IsCompact S :=
    (isCompact_closedBall (0 : ℝ × ℝ) (max R a)).inter_left (isClosed_le continuous_const continuous_norm)
  have hab : ‖((a, 0) : ℝ × ℝ)‖ = a := by
    simp [Prod.norm_def, Real.norm_eq_abs, abs_of_pos ha, ha.le]
  have hne : S.Nonempty := ⟨(a, 0), by
    simp only [S, Set.mem_inter_iff, Set.mem_ofPred_eq, Metric.mem_closedBall, dist_zero_right, hab]
    exact ⟨le_rfl, le_max_right _ _⟩⟩
  obtain ⟨η, hη, hmax⟩ := hS.exists_isMaxOn hne (continuous_fourierNu β t).norm.continuousOn
  have hηpos : 0 < ‖η‖ := ha.trans_le hη.1
  have hηlt : ‖fourierNu β t η‖ < 1 := norm_fourierNu_lt_one hβ t hηpos
  refine ⟨max ‖fourierNu β t η‖ (1 / 2), lt_max_of_lt_right (by norm_num),
    max_lt hηlt (by norm_num), ?_⟩
  intro ξ hξ
  by_cases hsmall : ‖ξ‖ ≤ max R a
  · have hmem : ξ ∈ S := ⟨hξ, by simpa only [Metric.mem_closedBall, dist_zero_right] using hsmall⟩
    exact (hmax hmem).trans (le_max_left _ _)
  · have hlarge : R ≤ ‖ξ‖ := (le_max_left R a).trans (not_le.mp hsmall).le
    exact (hC ξ).trans ((hR ‖ξ‖ hlarge).le.trans (le_max_right _ _))

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 := by norm_num

/-- Positive covariance makes the actual frequency whitening map send
nonzero vectors to nonzero vectors.
Source: arXiv:2412.09080v3, §2.3 `eq:Yi` and §5.4. -/
theorem whitenFrequency_ne_zero {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2)
    {ξ : ℝ × ℝ} (hξ : ξ ≠ 0) : whitenFrequency a b d ξ ≠ 0 := by
  intro h
  have hs : Real.sqrt (a * (a * d - b ^ 2)) ≠ 0 := (Real.sqrt_pos.mpr (mul_pos ha hD)).ne'
  have h2 := congrArg Prod.snd h
  rw [whitenFrequency_apply] at h2
  change a * ξ.2 / Real.sqrt (a * (a * d - b ^ 2)) = 0 at h2
  have hx2 : ξ.2 = 0 := (mul_eq_zero.mp ((div_eq_zero_iff.mp h2).resolve_right hs)).resolve_left ha.ne'
  have h1 := congrArg Prod.fst h
  rw [whitenFrequency_apply, hx2] at h1
  have hx1 : ξ.1 = 0 := by
    have hd : ξ.1 / Real.sqrt a = 0 := by simpa using h1
    exact (div_eq_zero_iff.mp hd).resolve_right (Real.sqrt_pos.mpr ha).ne'
  exact hξ (Prod.ext hx1 hx2)

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 * 1 - 0 ^ 2 ∧ ((1, 0) : ℝ × ℝ) ≠ 0 := by norm_num

/-- The characteristic function of the actual standardized summand is
strictly below one at every nonzero frequency.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4, definition of `ε`. -/
theorem norm_characteristic_lawY_lt_one {β : ℝ} (hβ : 0 < β) (t : ℝ) {ξ : ℝ × ℝ}
    (hξ : ξ ≠ 0) :
    ‖∫ z : ℝ × ℝ, Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) ∂lawY β t‖ < 1 := by
  rw [norm_characteristic_lawY]
  apply norm_fourierNu_lt_one hβ t
  rw [norm_neg, norm_pos_iff]
  exact whitenFrequency_ne_zero (sigmaFst_pos hβ t) (sigmaDet_pos hβ t) hξ

example : (0 : ℝ) < 3 ∧ ((1, 0) : ℝ × ℝ) ≠ 0 := by norm_num

end Transformer.Modes
