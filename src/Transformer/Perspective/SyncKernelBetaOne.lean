/-
# `τ_β ∫ |f_β'''|₊ < 4` for `β ≥ 1`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, Appendix A: `corollary:tau2`
and `lemma:beta1`; `landscape_plan.md`, M3.1e.

* `τ_β_star_two_lt_inv_sqrt` is `corollary:tau2`: `τ_β < 1/√β` for every
  `β > 0`, where `τ_β = τ_β_star 2 β = arccos((√(1 + 4β²) - 1)/(2β))`.  It is
  the peak `Causal.interactionPeak β` of `e^{β(cos x - 1)} sin x`
  (`τ_β_star_two_eq_interactionPeak`).
* `f_β''_le_of_one_le`, `neg_le_f_β''_of_one_le`: `-(8/5)√β ≤ f_β'' ≤ (2/5)√β`
  on `[0, π]` for `β ≥ 1`, from the two cases of `Perspective.SyncKernelSecond`.
* `τ_mul_f_β''_sub_lt_two` is `lemma:beta1`: for `β ≥ 1`,
  `τ_β (f_β''(a) - f_β''(b)) < 2`.
* `τ_mul_integral_posPart_f_β'''_lt_four` gives `τ_β ∫_{-π}^{π} |f_β'''|₊ < 4`
  for `β ≥ 1`.  That is `τ⟨1, |f'''|₊⟩ < 4f'(0)`, which §5 needs for
  `cor:conjecture`, since `f_β'(0) = 1`.

The source derives `corollary:tau2` from `lemma:tau1` (`√x τ_x` is increasing)
and the limit `√x τ_x → 1`.  It notes that the corollary also appears as
Lemma 4 of arXiv:2411.04990v2 (`lemma:interaction` (3), for `β ≥ 1`), which
`Causal.interactionPeak_lt_scale` proves for every `β > 0` from
`cos u ≤ 1 - u²/2 + u⁴/24` at `u = 1/√β`.  Here it is that theorem.
`landscape_plan.md` (M3.1e) allows this replacement, so `lemma:tau1` is not
formalized.

For `β ∈ [1, 2)` the source also needs `√β τ_β < √2 τ_2 < 0.96`, from
`lemma:tau1`.  The bounds on `f_β''` here differ by `2√β`, so `√β τ_β < 1`
suffices for every `β ≥ 1`.
-/

import Transformer.Perspective.SyncKernelSecond
import Transformer.Perspective.SyncKernelIntegral
import Transformer.Causal.SectionB_PeakTemperature

open Real

namespace Transformer
namespace Perspective

/-- **`f_β'' ≤ (2/5)√β` on `[0, π]` for `β ≥ 1`.**  Where `B ≤ 0`, `f_β'' ≤ 0`;
where `B ≥ 0`, `f_β''² ≤ (4/25)β` by `beta1_upper_core`.

Source: arXiv:2507.22857v1, Appendix A, proof of `lemma:beta1`, which bounds
`g_β = f_β''/√β` at `a` by `0.512 √(2 - 1.4/β)/(√2 β) + 0.375 √(2 - 1.5/β)/√2`. -/
theorem f_β''_le_of_one_le {β x : ℝ} (hβ : 1 ≤ β) (hx : x ∈ Set.Icc 0 π) :
    f_β'' β x ≤ 2 / 5 * Real.sqrt β := by
  have hβ0 : 0 < β := by linarith
  obtain ⟨t, z, B, ht0, ht1, hz0, htz, hB, hf, hsq⟩ := f_β''_substitution x hβ
  have hs : 0 ≤ Real.sin x := Real.sin_nonneg_of_nonneg_of_le_pi hx.1 hx.2
  rcases le_or_gt B 0 with hB0 | hB0
  · have : f_β'' β x ≤ 0 := by
      rw [hf]
      exact mul_nonpos_of_nonpos_of_nonneg
        (mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hβ0.le hs) hB0) (Real.exp_pos _).le
    linarith [Real.sqrt_nonneg β]
  · have hcore := beta1_upper_core ht0 ht1 hz0 htz (hB ▸ hB0.le)
    rw [← hB] at hcore
    have h1 : f_β'' β x ^ 2 * Real.exp (2 * z) ≤ 4 / 25 * β * Real.exp (2 * z) := by
      rw [hsq]; nlinarith
    have hbound := le_of_mul_le_mul_right h1 (Real.exp_pos _)
    nlinarith [Real.sq_sqrt hβ0.le, Real.sqrt_nonneg β]

/-- **`f_β'' ≥ -(8/5)√β` on `[0, π]` for `β ≥ 1`.**  Where `B ≥ 0`, `f_β'' ≥ 0`;
where `B ≤ 0`, `f_β''² ≤ (64/25)β` by `beta1_lower_core`.

Source: arXiv:2507.22857v1, Appendix A, proof of `lemma:beta1`, which bounds
`g_β` at `b` by `-0.247 √(2 - 0.18/β)/(√2 β) - 1.381 √(2 - 0.18/β)/√2`
(`eq:minval`). -/
theorem neg_le_f_β''_of_one_le {β x : ℝ} (hβ : 1 ≤ β) (hx : x ∈ Set.Icc 0 π) :
    -(8 / 5 * Real.sqrt β) ≤ f_β'' β x := by
  have hβ0 : 0 < β := by linarith
  obtain ⟨t, z, B, ht0, ht1, hz0, htz, hB, hf, hsq⟩ := f_β''_substitution x hβ
  have hs : 0 ≤ Real.sin x := Real.sin_nonneg_of_nonneg_of_le_pi hx.1 hx.2
  rcases le_or_gt 0 B with hB0 | hB0
  · have : 0 ≤ f_β'' β x := by
      rw [hf]
      exact mul_nonneg (mul_nonneg (mul_nonneg hβ0.le hs) hB0) (Real.exp_pos _).le
    linarith [Real.sqrt_nonneg β]
  · have hcore := beta1_lower_core ht0 ht1 hz0 htz (hB ▸ hB0.le)
    rw [← hB] at hcore
    have h1 : f_β'' β x ^ 2 * Real.exp (2 * z) ≤ 64 / 25 * β * Real.exp (2 * z) := by
      rw [hsq]; nlinarith
    have hbound := le_of_mul_le_mul_right h1 (Real.exp_pos _)
    nlinarith [Real.sq_sqrt hβ0.le, Real.sqrt_nonneg β]

/-- The hypotheses of both bounds hold at `β = 1`, `x = π/2`. -/
example : -(8 / 5 * Real.sqrt 1) ≤ f_β'' 1 (π / 2) ∧ f_β'' 1 (π / 2) ≤ 2 / 5 * Real.sqrt 1 :=
  have hx : π / 2 ∈ Set.Icc 0 π := ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩
  ⟨neg_le_f_β''_of_one_le le_rfl hx, f_β''_le_of_one_le le_rfl hx⟩

/-- `τ_β = τ_β_star 2 β` is the peak `Causal.interactionPeak β` of the
interaction `e^{β(cos x - 1)} sin x` of arXiv:2411.04990v2: both are
`arccos((√(1 + 4β²) - 1)/(2β))`. -/
theorem τ_β_star_two_eq_interactionPeak (β : ℝ) :
    τ_β_star 2 β = Transformer.Causal.interactionPeak β := by
  rw [τ_β_star, cosTauStar, Transformer.Causal.interactionPeak,
    Transformer.Causal.interactionPeakCos, add_comm (4 * β ^ 2)]
  congr 1
  norm_num
  ring

/-- **`corollary:tau2`: `τ_β < 1/√β` for every `β > 0`.**  `τ_β` is the peak
`Causal.interactionPeak β` (`τ_β_star_two_eq_interactionPeak`), which
`Causal.interactionPeak_lt_scale` bounds by `β^{-1/2}`, from
`cos u ≤ 1 - u²/2 + u⁴/24` at `u = 1/√β`.

Source: arXiv:2507.22857v1, Appendix A, `corollary:tau2`,
"`arccos((√(1+4x²) - 1)/(2x)) < 1/√x` over `(0, ∞)`", proved there from
`lemma:tau1`.  The source notes that it also appears in `[Lemma 4]` of
Karagodin, Polyanskiy and Rigollet, "Clustering in causal attention masking":
arXiv:2411.04990v2, Appendix A.2, `lemma:interaction` (3), stated there for
`β ≥ 1`. -/
theorem τ_β_star_two_lt_inv_sqrt {β : ℝ} (hβ : 0 < β) : τ_β_star 2 β < 1 / Real.sqrt β := by
  rw [τ_β_star_two_eq_interactionPeak, Real.sqrt_eq_rpow, one_div, ← Real.rpow_neg hβ.le]
  exact Transformer.Causal.interactionPeak_lt_scale β hβ

/-- `corollary:tau2` multiplied out: `√β τ_β < 1` for every `β > 0`. -/
theorem sqrt_mul_τ_β_star_two_lt_one {β : ℝ} (hβ : 0 < β) :
    Real.sqrt β * τ_β_star 2 β < 1 := by
  have h := τ_β_star_two_lt_inv_sqrt hβ
  rw [lt_div_iff₀ (Real.sqrt_pos.2 hβ)] at h
  linarith

/-- **`lemma:beta1`: `τ_β (f_β''(a) - f_β''(b)) < 2` for `β ≥ 1`**, for all
`a, b ∈ [0, π]`.  `f_β''(a) - f_β''(b) ≤ (2/5 + 8/5)√β = 2√β` and
`√β τ_β < 1`.

Source: arXiv:2507.22857v1, Appendix A, `lemma:beta1`, "If `β ≥ 1` then
`τ(f''(a) - f''(b)) < 2`", where `0 < b < a < π` are the ends of the positive
region of `f'''` (`lemma:positiveregion`, here
`integral_posPart_f_β'''_of_one_third_lt`).  Here `a` and `b` are any points
of `[0, π]`, which is stronger.  The source bounds the two values separately,
with its own constants and a case split at `β = 2`; the constants here are
`2/5` and `-8/5` (see the module docstring). -/
theorem τ_mul_f_β''_sub_lt_two {β a b : ℝ} (hβ : 1 ≤ β) (ha : a ∈ Set.Icc 0 π)
    (hb : b ∈ Set.Icc 0 π) : τ_β_star 2 β * (f_β'' β a - f_β'' β b) < 2 := by
  have hβ0 : 0 < β := by linarith
  have hτ := sqrt_mul_τ_β_star_two_lt_one hβ0
  have hτ0 := (τ_β_star_two_mem hβ0).1
  have hdiff : f_β'' β a - f_β'' β b ≤ 2 * Real.sqrt β := by
    linarith [f_β''_le_of_one_le hβ ha, neg_le_f_β''_of_one_le hβ hb]
  nlinarith [mul_le_mul_of_nonneg_left hdiff hτ0.le]

/-- **`τ_β ∫_{-π}^{π} |f_β'''|₊ < 4` for `β ≥ 1`.**  By
`integral_posPart_f_β'''_le` with `L = -(8/5)√β` and `U = (2/5)√β`, the
integral is at most `4√β`, and `√β τ_β < 1`.

Source: arXiv:2507.22857v1, §5, proof of `cor:conjecture`: "By
`thm:ascentnormal`, it suffices to show that `τ⟨1, |f'''|₊⟩ < 4f'(0)`".  For
`β > 1/3` it reduces this to `τ(f''(a) - f''(b)) < 2` through
`∫ |f'''|₊ = 2(f''(a) - f''(b))`, and for `β ≥ 1` to `lemma:beta1`.  Here
`f_β'(0) = 1`.  The bound also gives the hypothesis
`τ ∫ |f'''|₊ ≤ 4(1 + τ/2π) f'(0)` of `thm:main`. -/
theorem τ_mul_integral_posPart_f_β'''_lt_four {β : ℝ} (hβ : 1 ≤ β) :
    τ_β_star 2 β * ∫ x in (-π)..π, max (f_β''' β x) 0 < 4 := by
  have hβ0 : 0 < β := by linarith
  have hτ := sqrt_mul_τ_β_star_two_lt_one hβ0
  have hτ0 := (τ_β_star_two_mem hβ0).1
  have hint := integral_posPart_f_β'''_le hβ0
    (fun x hx => neg_le_f_β''_of_one_le hβ hx) (fun x hx => f_β''_le_of_one_le hβ hx)
  have h4 : 2 * (2 / 5 * Real.sqrt β - -(8 / 5 * Real.sqrt β)) = 4 * Real.sqrt β := by ring
  rw [h4] at hint
  nlinarith [mul_le_mul_of_nonneg_left hint hτ0.le]

/-- The hypotheses are satisfiable: `β = 1`, and `a = π`, `b = 0`. -/
example : τ_β_star 2 1 < 1 / Real.sqrt 1 ∧ Real.sqrt 1 * τ_β_star 2 1 < 1 ∧
    τ_β_star 2 1 * (f_β'' 1 π - f_β'' 1 0) < 2 ∧
    τ_β_star 2 1 * ∫ x in (-π)..π, max (f_β''' 1 x) 0 < 4 :=
  ⟨τ_β_star_two_lt_inv_sqrt one_pos, sqrt_mul_τ_β_star_two_lt_one one_pos,
    τ_mul_f_β''_sub_lt_two le_rfl ⟨Real.pi_pos.le, le_rfl⟩ ⟨le_rfl, Real.pi_pos.le⟩,
    τ_mul_integral_posPart_f_β'''_lt_four le_rfl⟩

end Perspective
end Transformer
