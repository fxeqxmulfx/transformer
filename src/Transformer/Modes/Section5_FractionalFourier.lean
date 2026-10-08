import Transformer.Modes.Section5_FractionalEnvelope
import Transformer.Modes.Section5_FourierIntegral

/-!
# Fourier decay with a bandwidth-dependent exponent

The source's `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, prints
exponent `1/2` for every positive bandwidth and a constant independent
of the translation. Both claims have proved counterexamples in the
Fourier counterexample modules. The corrected `uniform_decay` for
`0 < β < 2` is proved separately and retains that exponent.

Here the actual Fourier transform has a positive decay exponent for
any `β > 0`. The local bounds from `Section5_FractionalEnvelope` are
summed over all integer unit intervals. No spatial tail is discarded,
and the Gaussian weight retains its shift by the fixed translation.
For `0 < p ≤ 1` and `βp < 2`, the whole integral is bounded by
`C / |ρ|^(p/2)`, uniformly over every normalized direction.

The exact expectation-to-integral identity then transfers this bound
to `fourierNu`. Combining it with the unit bound at small frequencies
gives the denominator `(1 + ‖ξ‖)^(p/2)` for every frequency, including
zero. Choosing `p = 1 / (β + 1)` proves the all-bandwidth estimate with
exponent `1 / (2(β + 1))`. The constant depends on both `β` and fixed `t`.

In two dimensions a positive power of this estimate is integrable once
its exponent exceeds two. Thus the subsequent integrability result can
choose a finite number of summands for every bandwidth, instead of the
source's fixed five. This module establishes the Fourier estimate itself;
the power integrability statement is in `Section5_PowerIntegrability`.
Source: arXiv:2412.09080v3, §5.5, repairing `eq:uniform-decay`.
-/

open Real MeasureTheory

namespace Transformer.Modes

/-- Sum the actual local integrals with their positive-rate Gaussian
envelope. Source: arXiv:2412.09080v3, §5.5, the corrected tail estimate. -/
theorem gaussian_phase_fractional_integral_bound {β p : ℝ}
    (hβ : 0 < β) (hp : 0 < p) (hp1 : p ≤ 1) (hβp : β * p < 2) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ρ : ℝ, ρ ≠ 0 → ∀ θ : ℝ × ℝ, max |θ.1| |θ.2| = 1 →
      ‖∫ u : ℝ, gaussianPhaseIntegrand β ρ θ t u‖ ≤ C / |ρ| ^ (p / 2) := by
  obtain ⟨C, hC, hlocal⟩ := gaussian_phase_fractional_local_envelope hβ hp hp1 hβp t
  let F := fun m : ℤ => (1 + |(m : ℝ)|) ^ 3 *
    Real.exp (-((2 - β * p) / 8) * (m : ℝ) ^ 2)
  have hF : Summable F := summable_cubic_gaussian (by linarith)
  have hS : 0 ≤ ∑' m, F m := tsum_nonneg (fun _ => by dsimp [F]; positivity)
  refine ⟨C * (1 + ∑' m, F m), by positivity, ?_⟩
  intro ρ hρ θ hθ
  let J := fun m : ℤ => ∫ u in (m : ℝ)..(m : ℝ) + 1, gaussianPhaseIntegrand β ρ θ t u
  have hb (m : ℤ) : ‖J m‖ ≤ (C / |ρ| ^ (p / 2)) * F m := hlocal ρ hρ θ hθ m
  have hmajor : Summable fun m : ℤ => (C / |ρ| ^ (p / 2)) * F m := hF.mul_left _
  have hJ : Summable fun m : ℤ => ‖J m‖ :=
    Summable.of_nonneg_of_le (fun m => norm_nonneg _) hb hmajor
  rw [integral_eq_tsum_unit_intervals (integrable_gaussianPhaseIntegrand β ρ θ t)]
  change ‖∑' m, J m‖ ≤ _
  calc _ ≤ ∑' m, ‖J m‖ := norm_tsum_le_tsum_norm hJ
    _ ≤ ∑' m, (C / |ρ| ^ (p / 2)) * F m := hJ.tsum_le_tsum hb hmajor
    _ = (C / |ρ| ^ (p / 2)) * ∑' m, F m := tsum_mul_left
    _ ≤ C * (1 + ∑' m, F m) / |ρ| ^ (p / 2) := by
      have hnon : 0 ≤ C / |ρ| ^ (p / 2) := by positivity
      have hh := mul_le_mul_of_nonneg_left (show (∑' m, F m) ≤ 1 + ∑' m, F m by linarith) hnon
      convert hh using 1
      ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 ∧
    (3 : ℝ) * (1 / 4) < 2 ∧ (1 : ℝ) ≠ 0 ∧ max |(1 : ℝ)| |0| = 1 := by norm_num

/-- The normalized direction and exact shifted integral identity give
fractional decay at every nonzero frequency.
Source: arXiv:2412.09080v3, §5.5, the Fourier integral display. -/
theorem fourierNu_fractional_decay_nonzero {β p : ℝ}
    (hβ : 0 < β) (hp : 0 < p) (hp1 : p ≤ 1) (hβp : β * p < 2) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ξ : ℝ × ℝ, 0 < ‖ξ‖ →
      ‖fourierNu β t ξ‖ ≤ C / ‖ξ‖ ^ (p / 2) := by
  obtain ⟨C, hC, hbound⟩ := gaussian_phase_fractional_integral_bound hβ hp hp1 hβp t
  let A := (Real.sqrt (2 * Real.pi))⁻¹
  have hA : 0 < A := by dsimp [A]; positivity
  refine ⟨A * C, mul_pos hA hC, ?_⟩
  intro ξ hξ
  let θ : ℝ × ℝ := (ξ.1 / ‖ξ‖, ξ.2 / ‖ξ‖)
  have hθ : max |θ.1| |θ.2| = 1 := normalized_fourier_direction hξ
  have hrepr : (‖ξ‖ * θ.1, ‖ξ‖ * θ.2) = ξ := by
    apply Prod.ext
    · exact mul_div_cancel₀ _ hξ.ne'
    · exact mul_div_cancel₀ _ hξ.ne'
  have heq := fourierNu_eq_gaussian_phase_integral β ‖ξ‖ t θ
  rw [hrepr] at heq
  rw [heq, norm_smul, Real.norm_eq_abs, abs_of_pos hA]
  calc _ ≤ A * (C / ‖ξ‖ ^ (p / 2)) :=
        mul_le_mul_of_nonneg_left (by simpa only [abs_of_pos hξ] using hbound ‖ξ‖ hξ.ne' θ hθ) hA.le
    _ = _ := by ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 ∧
    (3 : ℝ) * (1 / 4) < 2 ∧ (0 : ℝ) < ‖((1, 0) : ℝ × ℝ)‖ := by
  norm_num [Prod.norm_def, Real.norm_eq_abs]

/-- Include zero and small frequencies using the unit bound on the
Gaussian expectation. Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`,
with fixed `t` and the explicitly corrected exponent `p/2`. -/
theorem uniform_fractional_decay {β p : ℝ}
    (hβ : 0 < β) (hp : 0 < p) (hp1 : p ≤ 1) (hβp : β * p < 2) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ξ : ℝ × ℝ,
      ‖fourierNu β t ξ‖ ≤ C / (1 + ‖ξ‖) ^ (p / 2) := by
  obtain ⟨C, hC, hdecay⟩ := fourierNu_fractional_decay_nonzero hβ hp hp1 hβp t
  have h2 : 0 < (2 : ℝ) ^ (p / 2) := Real.rpow_pos_of_pos (by norm_num) _
  refine ⟨(2 : ℝ) ^ (p / 2) * (C + 1), by positivity, ?_⟩
  intro ξ
  have hpow : 0 < (1 + ‖ξ‖) ^ (p / 2) := Real.rpow_pos_of_pos (by positivity) _
  by_cases hsmall : ‖ξ‖ ≤ 1
  · apply (norm_fourierNu_le_one β t ξ).trans
    apply (le_div_iff₀ hpow).mpr
    calc 1 * (1 + ‖ξ‖) ^ (p / 2) ≤ (2 : ℝ) ^ (p / 2) := by
          simpa only [one_mul] using Real.rpow_le_rpow (by positivity)
            (by linarith : 1 + ‖ξ‖ ≤ 2) (by positivity : 0 ≤ p / 2)
      _ ≤ (2 : ℝ) ^ (p / 2) * (C + 1) := by nlinarith
  · have hlarge : 1 ≤ ‖ξ‖ := (not_le.mp hsmall).le
    have hξ : 0 < ‖ξ‖ := zero_lt_one.trans_le hlarge
    have hs : (1 + ‖ξ‖) ^ (p / 2) ≤ (2 : ℝ) ^ (p / 2) * ‖ξ‖ ^ (p / 2) := by
      rw [← Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) (norm_nonneg ξ)]
      exact Real.rpow_le_rpow (by positivity) (by linarith) (by positivity)
    apply (hdecay ξ hξ).trans
    apply (div_le_div_iff₀ (Real.rpow_pos_of_pos hξ _) hpow).mpr
    calc _ ≤ C * ((2 : ℝ) ^ (p / 2) * ‖ξ‖ ^ (p / 2)) := mul_le_mul_of_nonneg_left hs hC.le
      _ ≤ (C + 1) * ((2 : ℝ) ^ (p / 2) * ‖ξ‖ ^ (p / 2)) :=
        mul_le_mul_of_nonneg_right (by linarith) (by positivity)
      _ = _ := by ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 ∧
    (3 : ℝ) * (1 / 4) < 2 := by norm_num

/-- Every positive bandwidth admits Fourier decay with exponent
`1 / (2(β + 1))`. The source prints the false exponent `1/2` uniformly
in `β`; here the exponent is explicitly changed and the constant also
depends on fixed `t`. Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`. -/
theorem fourierNu_decay_all_bandwidths {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ξ : ℝ × ℝ,
      ‖fourierNu β t ξ‖ ≤ C / (1 + ‖ξ‖) ^ (1 / (2 * (β + 1))) := by
  have hden : 0 < β + 1 := by linarith
  have hp : 0 < 1 / (β + 1) := by positivity
  have hp1 : 1 / (β + 1) ≤ 1 := (div_le_iff₀ hden).mpr (by linarith)
  have hβp : β * (1 / (β + 1)) < 2 := by
    rw [mul_one_div, div_lt_iff₀ hden]
    linarith
  have heq : (1 / (β + 1)) / 2 = 1 / (2 * (β + 1)) := by field_simp
  simpa only [heq] using uniform_fractional_decay hβ hp hp1 hβp t

example : (0 : ℝ) < 3 := by norm_num

end Transformer.Modes
