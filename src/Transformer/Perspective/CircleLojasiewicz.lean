/-
# The synchronized states, and Łojasiewicz for `-𝖤_β` on angles

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5, §7 and
Appendix A.

For `β > 0` the synchronized states are exactly the global maxima of `𝖤_β`
(`synchronized_of_isGlobalMaxTorusEnergy`, the converse of
`isGlobalMaxTorusEnergy_of_synchronized`), and they are critical
(`isCriticalTorusEnergy_of_synchronized`).  There every `g_β(θ_i - θ_j)` is
`e^β`, so the second derivative vanishes only along the line `ℝ𝟙` of common
rotations (`eq_of_secondDerivTorusEnergy_synchronized`).

The gradient ascent `eq:onangles` of `𝖤_β` is the gradient descent of
`-𝖤_β`, the form `SaddleAvoidance` treats.  `-𝖤_β` satisfies the Łojasiewicz
inequality with exponent `1/2` at a critical point where only the common
rotations have zero second derivative (`isLojasiewiczAt_neg_eucTorusEnergy`):
this is `Landscape.SaddleAvoidance.isLojasiewiczAt_of_invariant` with `W = ℝ𝟙`.  That
covers the synchronized states (`isLojasiewiczAt_of_synchronized`) and, for
two tokens, every critical point (`isLojasiewiczAt_of_isCritical_two`).
-/

import Transformer.Perspective.CircleEnergyHessian
import Transformer.Landscape.SaddleAvoidance.Invariant

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- `sin (2πk) = 0` and `cos (2πk) = 1`, so `g_β(2πk) = e^β`. -/
theorem g_β_2d_int_mul_two_pi (β : ℝ) (k : ℤ) : g_β_2d β (k * (2 * π)) = exp β := by
  have hs : sin (k * (2 * π)) = 0 := by simpa using sin_add_int_mul_two_pi 0 k
  simp [g_β_2d, hs]

/-- **Synchronized states are critical**: every `sin(θ_j - θ_i)` vanishes. -/
theorem isCriticalTorusEnergy_of_synchronized {β : ℝ} (hβ : β ≠ 0) {θ : Angles n}
    (hθ : ∀ i j, ∃ k : ℤ, θ i - θ j = k * (2 * π)) : IsCriticalTorusEnergy n β θ := by
  intro i
  convert hasDerivAt_torusEnergy n β hβ θ i using 1
  refine (mul_eq_zero_of_right _ (Finset.sum_eq_zero fun j _ => ?_)).symm
  obtain ⟨k, hk⟩ := hθ j i
  rw [hk, show sin (k * (2 * π)) = 0 by simpa using sin_add_int_mul_two_pi 0 k, mul_zero]

/-- **For `β > 0` the global maxima of `𝖤_β` are synchronized**, the converse of
`isGlobalMaxTorusEnergy_of_synchronized`: if `cos(θ_i - θ_j) < 1` for one pair,
that term of `𝖤_β(θ)` is below `e^β` and `𝖤_β(θ) < 𝖤_β(0)`.

Source: arXiv:2312.10794v5, §7, after `eq:energyF`: `𝖤_β` "is maximized when
`θ_i = θ^*` for some `θ^* ∈ 𝕋` and for all `i ∈ [n]`". -/
theorem synchronized_of_isGlobalMaxTorusEnergy {β : ℝ} (hβ : 0 < β) {θ : Angles n}
    (h : IsGlobalMaxTorusEnergy n β θ) : ∀ i j, ∃ k : ℤ, θ i - θ j = k * (2 * π) := by
  intro i j
  by_contra hne
  have hcos : cos (θ i - θ j) < 1 := lt_of_le_of_ne (cos_le_one _) fun h1 =>
    hne (((cos_eq_one_iff _).1 h1).imp fun _ hk => hk.symm)
  have hle : ∀ a b : Idx n, exp (β * cos (θ a - θ b)) ≤ exp β :=
    fun a b => exp_le_exp.2 (mul_le_of_le_one_right hβ.le (cos_le_one _))
  have hlt : ∑ a, ∑ b, exp (β * cos (θ a - θ b)) < ∑ _a : Idx n, ∑ _b : Idx n, exp β :=
    Finset.sum_lt_sum (fun a _ => Finset.sum_le_sum fun b _ => hle a b) ⟨i, Finset.mem_univ _,
      Finset.sum_lt_sum (fun b _ => hle i b) ⟨j, Finset.mem_univ _,
        exp_lt_exp.2 (mul_lt_of_lt_one_right hβ hcos)⟩⟩
  have hn : (0 : ℝ) < n := Nat.cast_pos.2 (Fin.pos i)
  have hpos : 0 < 1 / (2 * β * (n : ℝ) ^ 2) := by positivity
  have h0 := h 0
  simp only [torusEnergy, Pi.zero_apply, sub_self, cos_zero, mul_one] at h0
  linarith [mul_lt_mul_of_pos_left hlt hpos]

/-- **At a synchronized state only the common rotations have zero second
derivative**: there every `g_β(θ_i - θ_j)` is `e^β`, so the second derivative
along `v` is `-(e^β/(2n²)) Σ_i Σ_j (v_i - v_j)²`. -/
theorem eq_of_secondDerivTorusEnergy_synchronized {β : ℝ} (hβ : β ≠ 0) {θ : Angles n}
    (hθ : ∀ i j, ∃ k : ℤ, θ i - θ j = k * (2 * π)) {v : Angles n}
    (hv : SecondDerivTorusEnergy n β θ v 0) (i j : Idx n) : v i = v j := by
  have h := secondDerivTorusEnergy_unique n hv (secondDerivTorusEnergy_line n hβ θ v)
  simp only [fun a b => (hθ a b).elim fun k hk => hk ▸ g_β_2d_int_mul_two_pi β k] at h
  have hn : (0 : ℝ) < n := Nat.cast_pos.2 (Fin.pos i)
  have hsum : ∑ a, ∑ b, (v a - v b) ^ 2 * exp β = 0 := by
    have hK : -(1 / (2 * (n : ℝ) ^ 2)) ≠ 0 := by
      have : 0 < 1 / (2 * (n : ℝ) ^ 2) := by positivity
      linarith
    exact (mul_eq_zero.1 h.symm).resolve_left hK
  have hi := (Finset.sum_eq_zero_iff_of_nonneg fun a _ => Finset.sum_nonneg fun b _ =>
    by positivity).1 hsum i (Finset.mem_univ _)
  have hij := (Finset.sum_eq_zero_iff_of_nonneg fun b _ => by positivity).1 hi j
    (Finset.mem_univ _)
  have hsq := (mul_eq_zero.1 hij).resolve_right (exp_pos β).ne'
  exact sub_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 hsq)

/-- **Łojasiewicz across the common rotations.**  At a critical point of
`𝖤_β` where only the common rotations `v ∈ ℝ𝟙` have zero second derivative,
`-𝖤_β` satisfies the Łojasiewicz inequality with exponent `1/2`.  If the
derivative `A` of `∇(-𝖤_β)` kills `v`, then `⟪A v, v⟫ = 0`, so the second
derivative along `v` is `0` (`inner_fderiv_gradient_eucTorusEnergy`) and `v`
is a common rotation; on `(ℝ𝟙)ᗮ` that forces `v = 0`, which is what
`Landscape.SaddleAvoidance.isLojasiewiczAt_of_invariant` asks. -/
theorem isLojasiewiczAt_neg_eucTorusEnergy {β : ℝ} (hβ : β ≠ 0) {θ : Angles n}
    (hθ : IsCriticalTorusEnergy n β θ)
    (hv : ∀ v : Angles n, SecondDerivTorusEnergy n β θ v 0 → ∀ i j, v i = v j) :
    Landscape.SaddleAvoidance.IsLojasiewiczAt (fun ψ => -eucTorusEnergy n β ψ) (WithLp.toLp 2 θ) (1 / 2) := by
  set W := Submodule.span ℝ {WithLp.toLp 2 fun _ : Idx n => (1 : ℝ)}
  have hf : ContDiff ℝ 2 fun ψ => -eucTorusEnergy n β ψ := (contDiff_eucTorusEnergy n β).neg
  refine Landscape.SaddleAvoidance.isLojasiewiczAt_of_invariant (W := W) (hf.differentiable (by norm_num))
    (fun w hw y => by simp only [eucTorusEnergy_add_mem_span_one n β hw])
    ((Landscape.SaddleAvoidance.contDiff_gradient hf).differentiable one_ne_zero _).hasFDerivAt
    ((gradient_neg_eucTorusEnergy_eq_zero_iff n hβ _).2 hθ) fun v hvW hAv => ?_
  rw [show gradient (fun ψ => -eucTorusEnergy n β ψ) = -gradient (eucTorusEnergy n β) from
    funext (gradient_neg_eucTorusEnergy n β), fderiv_neg, neg_apply, neg_eq_zero] at hAv
  have hc := inner_fderiv_gradient_eucTorusEnergy n (secondDerivTorusEnergy_line n hβ θ v.ofLp)
  rw [WithLp.toLp_ofLp, hAv, inner_zero_left] at hc
  have h0 := secondDerivTorusEnergy_line n hβ θ v.ofLp
  rw [← hc] at h0
  have hvW' : v ∈ W := by
    rcases isEmpty_or_nonempty (Idx n) with hn | ⟨⟨i₀⟩⟩
    · rw [Subsingleton.elim v 0]
      exact W.zero_mem
    · refine Submodule.mem_span_singleton.2 ⟨v.ofLp i₀, ?_⟩
      ext i
      simp [hv _ h0 i i₀]
  have hbot : v ∈ W ⊓ Wᗮ := ⟨hvW', hvW⟩
  rwa [Submodule.inf_orthogonal_eq_bot, Submodule.mem_bot] at hbot

/-- **Łojasiewicz at the synchronized states**: there only the common
rotations have zero second derivative
(`eq_of_secondDerivTorusEnergy_synchronized`), so `-𝖤_β` satisfies the
Łojasiewicz inequality with exponent `1/2`. -/
theorem isLojasiewiczAt_of_synchronized {β : ℝ} (hβ : β ≠ 0) {θ : Angles n}
    (hθ : ∀ i j, ∃ k : ℤ, θ i - θ j = k * (2 * π)) :
    Landscape.SaddleAvoidance.IsLojasiewiczAt (fun ψ => -eucTorusEnergy n β ψ) (WithLp.toLp 2 θ) (1 / 2) :=
  isLojasiewiczAt_neg_eucTorusEnergy n hβ (isCriticalTorusEnergy_of_synchronized n hβ hθ)
    fun _ hv => eq_of_secondDerivTorusEnergy_synchronized n hβ hθ hv

/-- **For two tokens, `-𝖤_β` satisfies the Łojasiewicz inequality at every
critical point.**  A critical `θ` has `sin(θ_0 - θ_1) = 0`, so
`g_β(θ_0 - θ_1) = cos(θ_0 - θ_1) e^{β cos(θ_0 - θ_1)} ≠ 0`, and the second
derivative `-(1/4)(v_0 - v_1)² g_β(θ_0 - θ_1)` vanishes only if `v_0 = v_1`. -/
theorem isLojasiewiczAt_of_isCritical_two {β : ℝ} (hβ : β ≠ 0) {θ : Angles 2}
    (hθ : IsCriticalTorusEnergy 2 β θ) :
    Landscape.SaddleAvoidance.IsLojasiewiczAt (fun ψ => -eucTorusEnergy 2 β ψ) (WithLp.toLp 2 θ) (1 / 2) := by
  have hs : sin (θ 0 - θ 1) = 0 := by
    have h := (hasDerivAt_torusEnergy 2 β hβ θ 0).unique (hθ 0)
    simp [Fin.sum_univ_two, exp_ne_zero] at h
    rw [← neg_sub, sin_neg, h, neg_zero]
  have hg : g_β_2d β (θ 0 - θ 1) ≠ 0 := by
    have hc : cos (θ 0 - θ 1) ≠ 0 := fun hc => by
      simpa [hs, hc] using sin_sq_add_cos_sq (θ 0 - θ 1)
    simp [g_β_2d, hs, hc, exp_ne_zero]
  refine isLojasiewiczAt_neg_eucTorusEnergy 2 hβ hθ fun v hv => ?_
  have h := secondDerivTorusEnergy_unique 2 hv (secondDerivTorusEnergy_line 2 hβ θ v)
  simp only [Fin.sum_univ_two, sub_self, g_β_2d_symm β (θ 1) (θ 0)] at h
  push_cast at h
  have h01 : (v 0 - v 1) ^ 2 * g_β_2d β (θ 0 - θ 1) = 0 := by linear_combination 4 * h
  have hv01 : v 0 = v 1 :=
    sub_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 ((mul_eq_zero.1 h01).resolve_right hg))
  intro i j
  fin_cases i <;> fin_cases j <;> simp [hv01]

/-- The hypotheses of the theorems above at a synchronized state hold at
`β = 1` and `θ = 0`: every difference is `0 · 2π`, so `0` is critical and a
global maximum (`isGlobalMaxTorusEnergy_of_synchronized`), the second
derivative along `𝟙` is `0`, and only the common rotations have zero second
derivative. -/
example (n : ℕ) : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ i j : Idx n, ∃ k : ℤ, (0 : Angles n) i - (0 : Angles n) j = k * (2 * π)) ∧
    IsCriticalTorusEnergy n 1 0 ∧ IsGlobalMaxTorusEnergy n 1 0 ∧
    SecondDerivTorusEnergy n 1 0 (fun _ => 1) 0 ∧
    ∀ v : Angles n, SecondDerivTorusEnergy n 1 0 v 0 → ∀ i j, v i = v j := by
  have h0 : ∀ i j : Idx n, ∃ k : ℤ, (0 : Angles n) i - (0 : Angles n) j = k * (2 * π) :=
    fun _ _ => ⟨0, by simp⟩
  refine ⟨one_ne_zero, one_pos, h0, isCriticalTorusEnergy_of_synchronized n one_ne_zero h0,
    isGlobalMaxTorusEnergy_of_synchronized n 1 one_pos 0 h0, ?_,
    fun _ hv => eq_of_secondDerivTorusEnergy_synchronized n one_ne_zero h0 hv⟩
  simpa using secondDerivTorusEnergy_line n one_ne_zero 0 fun _ => 1

/-- The hypotheses of `isLojasiewiczAt_of_isCritical_two` hold at `β = 1` for
the antipodal pair `θ = (0, π)`: every `sin(θ_j - θ_i)` vanishes. -/
example : (1 : ℝ) ≠ 0 ∧ IsCriticalTorusEnergy 2 1 ![0, π] :=
  ⟨one_ne_zero, fun i => (hasDerivAt_torusEnergy 2 1 one_ne_zero _ i).congr_deriv
    (by fin_cases i <;> simp [Fin.sum_univ_two])⟩

end Perspective
end Transformer
