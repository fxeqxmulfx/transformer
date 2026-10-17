/-
# `lemma:taumax`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §3, `lemma:taumax`.

A configuration `x ∈ 𝕋^n` that is stationary (C1), cut-stable (C2) and not
synchronized satisfies

  `∫_{-π}^{π} |f'''|₊ ≥ 4 (1/τ + 1/τ_max) f'(0)`

(`taumax_sharp`), and so the source's `min(8/τ, 4/τ + 4/τ_max) f'(0)` bound
(`taumax`).  Each gap gives `4 (N_k + N_{k+1}) f'(0)/Δ_k ≤ ∫_{θ_k}^{θ_{k+1}} |φ''|₊`
(`circleMult_add_mul_le_integral`); the gaps add up to at most
`n ∫_{-π}^{π} |f'''|₊` (`sum_integral_circleSite_le`), and the counting step
`le_sum_div_of_le` bounds `Σ_k (N_k + N_{k+1})/Δ_k` below by `n/τ + n/τ_max`,
since every gap but the largest is at most `τ` (`circleGap_le_of_lt_circleGap`)
and at most `τ_max`.
-/

import Transformer.Perspective.CircleGapSum

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {n : ℕ}

/-- **Lemma (lemma:taumax), sharp form.**  Let `f` be `2π`-periodic with
`f' = f₁`, `f'' = f₂`, `f''' = f₃` continuous, and `f' < 0` off `[-τ, τ]` on
`𝕋`, that is on `(τ, 2π - τ)`.  If `x` is stationary (C1), cut-stable (C2) for
every `S`, and not synchronized, then

  `4 (1/τ + 1/τ_max(x)) f'(0) ≤ ∫_{-π}^{π} |f'''|₊`.

Source: arXiv:2507.22857v1, §3, proof of `lemma:taumax`.  For `f'(0) ≥ 0`
the bound is at least the source's `min(8/τ, 4/τ + 4/τ_max) f'(0)`; the
source's proof gives it with the counting step of `le_sum_div_of_le`, and
`cor:main` needs it when `M ≤ τ`.  Changed as in
`lemma:twoconsecutive_general`: (C2) for every `S`, and `τ ≤ π` is not
needed.  No sign of `f'(0)` is needed. -/
theorem taumax_sharp {f f₁ f₂ f₃ : ℝ → ℝ} (hf : Function.Periodic f (2 * π))
    (h₁ : ∀ y, HasDerivAt f (f₁ y) y) (h₂ : ∀ y, HasDerivAt f₁ (f₂ y) y)
    (h₃ : ∀ y, HasDerivAt f₂ (f₃ y) y) (hc₃ : Continuous f₃) {τ : ℝ} (hτ : 0 < τ)
    (hneg : ∀ u ∈ Set.Ioo τ (2 * π - τ), f₁ u < 0) {x : Angles n}
    (hC1 : ∀ i, ∑ j, f (x i - x j) = 0)
    (hC2 : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, f₁ (x i - x j))
    (hx : ¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π)) :
    4 * (1 / τ + 1 / circleGapMax x) * f₁ 0 ≤ ∫ u in -π..π, max (f₃ u) 0 := by
  have hK := two_le_card_circleSites x hx
  have h0 : 0 < (circleSites x).card := by omega
  have hp₁ := periodic_of_hasDerivAt hf h₁
  have hp₃ := periodic_of_hasDerivAt (periodic_of_hasDerivAt hp₁ h₂) h₃
  have hΔ := circleGap_pos x h0
  have hI : 0 ≤ ∫ u in -π..π, max (f₃ u) 0 :=
    intervalIntegral.integral_nonneg (by linarith [pi_pos]) fun u _ => le_max_right _ _
  obtain ⟨ℓ, hℓ, hmax⟩ := exists_circleGap_eq_circleGapMax x h0
  rw [← hmax]
  rcases lt_or_ge (f₁ 0) 0 with hf₀ | hf₀
  · have : 0 < 4 * (1 / τ + 1 / circleGap x ℓ) := by have := hΔ ℓ; positivity
    exact (mul_neg_of_pos_of_neg this hf₀).le.trans hI
  have hn : (0 : ℝ) < n := by
    have hle : (circleSites x).card ≤ n :=
      Finset.card_image_le.trans_eq (by rw [Finset.card_univ, Fintype.card_fin])
    exact_mod_cast (show 0 < n by omega)
  have hcount := le_sum_div_of_le (s := Finset.range (circleSites x).card)
    (w := fun k => (circleMult x k : ℝ) + circleMult x (k + 1)) (m := n)
    (Finset.mem_range.2 hℓ) (fun k _ => hΔ k) (fun k _ => by positivity)
    (fun k hk hkℓ => by
      have hle := (circleGap_le_circleGapMax x (Finset.mem_range.1 hk)).trans_eq hmax.symm
      refine ⟨?_, hle⟩
      by_cases hτℓ : τ < circleGap x ℓ
      · exact circleGap_le_of_lt_circleGap f₁ hp₁ τ hneg x hC2 (Finset.mem_range.1 hk) hℓ
          hkℓ hτℓ
      · linarith)
    (by rw [Finset.sum_add_distrib, sum_circleMult_succ, sum_circleMult]; ring)
    (circleMult_add_circleMult_succ_le x hK hℓ)
  have hgaps : 4 * f₁ 0 * ∑ k ∈ Finset.range (circleSites x).card,
      ((circleMult x k : ℝ) + circleMult x (k + 1)) / circleGap x k ≤
      ∑ k ∈ Finset.range (circleSites x).card,
        ∫ y in circleSite x k..circleSite x (k + 1), max (∑ j, f₃ (y - x j)) 0 := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun k _ => ?_
    have := circleMult_add_mul_le_integral hf h₁ h₂ h₃ hc₃ hC1 hC2 h0 k
    rw [mul_div_assoc', div_le_iff₀ (hΔ k)]
    linarith
  refine le_of_mul_le_mul_left ?_ hn
  rw [show (n : ℝ) * (4 * (1 / τ + 1 / circleGap x ℓ) * f₁ 0) =
    4 * f₁ 0 * (n / τ + n / circleGap x ℓ) by ring]
  exact (mul_le_mul_of_nonneg_left hcount (by positivity)).trans
    (hgaps.trans (sum_integral_circleSite_le hp₃ hc₃ x h0))

/-- **Lemma (lemma:taumax).**  Under the hypotheses of `taumax_sharp`,

  `min(8/τ, 4/τ + 4/τ_max(x)) f'(0) ≤ ∫_{-π}^{π} |f'''|₊`.

Source: arXiv:2507.22857v1, §3, `lemma:taumax`, where `⟨1, |f'''|₊⟩_{L_2}` is
`∫_{-π}^{π} |f'''|₊`.  Changed as in `lemma:twoconsecutive_general`: (C2) for
every `S`, and `τ ≤ π` is not needed. -/
theorem taumax {f f₁ f₂ f₃ : ℝ → ℝ} (hf : Function.Periodic f (2 * π))
    (h₁ : ∀ y, HasDerivAt f (f₁ y) y) (h₂ : ∀ y, HasDerivAt f₁ (f₂ y) y)
    (h₃ : ∀ y, HasDerivAt f₂ (f₃ y) y) (hc₃ : Continuous f₃) {τ : ℝ} (hτ : 0 < τ)
    (hneg : ∀ u ∈ Set.Ioo τ (2 * π - τ), f₁ u < 0) {x : Angles n}
    (hC1 : ∀ i, ∑ j, f (x i - x j) = 0)
    (hC2 : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, f₁ (x i - x j))
    (hx : ¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π)) :
    min (8 / τ) (4 / τ + 4 / circleGapMax x) * f₁ 0 ≤ ∫ u in -π..π, max (f₃ u) 0 := by
  have hs := taumax_sharp hf h₁ h₂ h₃ hc₃ hτ hneg hC1 hC2 hx
  rcases lt_or_ge (f₁ 0) 0 with hf₀ | hf₀
  · have hg := circleGapMax_pos x (by have := two_le_card_circleSites x hx; omega)
    have : 0 < min (8 / τ) (4 / τ + 4 / circleGapMax x) :=
      lt_min (by positivity) (by positivity)
    exact (mul_neg_of_pos_of_neg this hf₀).le.trans
      (intervalIntegral.integral_nonneg (by linarith [pi_pos]) fun u _ => le_max_right _ _)
  calc min (8 / τ) (4 / τ + 4 / circleGapMax x) * f₁ 0
      ≤ (4 / τ + 4 / circleGapMax x) * f₁ 0 := mul_le_mul_of_nonneg_right (min_le_right _ _) hf₀
    _ = 4 * (1 / τ + 1 / circleGapMax x) * f₁ 0 := by ring
    _ ≤ _ := hs

/-- The hypotheses of `taumax_sharp` and `taumax` hold together for two
antipodal particles, `x = (0, π)`, the kernel `f = sin 2·` and `τ = π`, so that
`(τ, 2π - τ)` is empty: every difference is a multiple of `π`, where `sin 2·`
vanishes and `2 cos 2· = 2`, and `π` is not a multiple of `2π`. -/
example :
    let x : Angles 2 := fun i => (i : ℕ) * π
    Function.Periodic (fun y => sin (2 * y)) (2 * π) ∧
      (∀ y, HasDerivAt (fun y => sin (2 * y)) (2 * cos (2 * y)) y) ∧
      (∀ y, HasDerivAt (fun y => 2 * cos (2 * y)) (-(4 * sin (2 * y))) y) ∧
      (∀ y, HasDerivAt (fun y => -(4 * sin (2 * y))) (-(8 * cos (2 * y))) y) ∧
      Continuous (fun y => -(8 * cos (2 * y))) ∧ (0 : ℝ) < π ∧
      (∀ u ∈ Set.Ioo π (2 * π - π), 2 * cos (2 * u) < 0) ∧
      (∀ i, ∑ j, sin (2 * (x i - x j)) = 0) ∧
      (∀ S : Finset (Idx 2), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, 2 * cos (2 * (x i - x j))) ∧
      ¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π) := by
  intro x
  have hx : ∀ i j, ∃ k : ℤ, x i - x j = k * π := fun i j =>
    ⟨(i : ℕ) - (j : ℕ), by simp only [x]; push_cast; ring⟩
  have h2 : ∀ y : ℝ, HasDerivAt (fun y => 2 * y) 2 y := fun y =>
    ((hasDerivAt_id' y).const_mul 2).congr_deriv (mul_one 2)
  refine ⟨fun y => ?_, fun y => (h2 y).sin.congr_deriv (by ring),
    fun y => ((h2 y).cos.const_mul 2).congr_deriv (by ring),
    fun y => ((h2 y).sin.const_mul 4).fun_neg.congr_deriv (by ring), by fun_prop, pi_pos,
    fun u hu => by linarith [hu.1, hu.2], fun i => Finset.sum_eq_zero fun j _ => ?_,
    fun S => Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => ?_, fun h => ?_⟩
  · show sin (2 * (y + 2 * π)) = sin (2 * y)
    rw [show 2 * (y + 2 * π) = 2 * y + 2 * π + 2 * π by ring, sin_add_two_pi, sin_add_two_pi]
  · obtain ⟨k, hk⟩ := hx i j
    rw [hk, show 2 * (k * π) = ((2 * k : ℤ) : ℝ) * π by push_cast; ring]
    exact sin_int_mul_pi _
  · obtain ⟨k, hk⟩ := hx i j
    rw [hk, show 2 * (k * π) = (k : ℝ) * (2 * π) by ring, cos_int_mul_two_pi]
    norm_num
  · obtain ⟨k, hk⟩ := h 1 0
    have h1 : π * 1 = π * (2 * k) := by simp [x] at hk; linarith
    have h2 : (1 : ℤ) = 2 * k := by exact_mod_cast mul_left_cancel₀ pi_pos.ne' h1
    omega

end Perspective
end Transformer
