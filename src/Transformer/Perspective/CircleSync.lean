/-
# The synchronization criterion on the circle

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §2, `thm:main`, and §3, `cor:main`.

For the mean-field model `ẋ_i = -Σ_j f(x_i - x_j)` on `𝕋` (S1), with `f`
`2π`-periodic, `C³`, `f'(0) > 0` and `f' < 0` off `[-τ, τ]`: a configuration
that is stationary (C1) and cut-stable (C2) for every `S` is synchronized,
`x_i - x_j ∈ 2πℤ`, as soon as

  `τ ∫_{-π}^{π} |f'''|₊ ≤ 4 (1 + τ/M) f'(0)`

for an `M` above its largest gap unless it is synchronized
(`synchronized_of_circleGapMax_lt`, `cor:main`), in particular for `M = 2π`
(`synchronized_of_integral_le`, `thm:main`).  Both follow from the sharp form
of `lemma:taumax` (`taumax_sharp`).

The source states both without `f'(0) > 0`, which they need: `f = 0` and
`τ = π` satisfy their other hypotheses with `M = 2π`, every configuration is
then stationary and cut-stable, and `(0, π)` is not synchronized
(`zero_kernel_counterexample`).
-/

import Transformer.Perspective.CircleTaumax
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Analysis.Real.Pi.Bounds

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {n : ℕ}

/-- **Corollary (cor:main).**  Let `f` be `2π`-periodic with `f' = f₁`,
`f'' = f₂`, `f''' = f₃` continuous, `f'(0) > 0`, and `f' < 0` off `[-τ, τ]` on
`𝕋`, that is on `(τ, 2π - τ)`.  If `x` is stationary (C1), cut-stable (C2) for
every `S`, its largest gap is below `M` unless it is synchronized, and

  `τ ∫_{-π}^{π} |f'''|₊ ≤ 4 (1 + τ/M) f'(0)`,

then `x` is synchronized: `x_i - x_j ∈ 2πℤ` for all `i, j`.

Source: arXiv:2507.22857v1, §3, `cor:main`.  Changed:
* `f'(0) > 0` is added; without it the claim fails
  (`zero_kernel_counterexample`).
* "Stable" is replaced by (C2) for every `S`, the condition the proof uses,
  as in `lemma:twoconsecutive_general`; the source derives (C2) from
  stability, and for `𝖤_β` it is `cutStable_of_torusHessianNonPos`.
* The source assumes `τ_max < M` at every stable, stationary, non-synchronized
  point; here only at `x`.  `M > 0` and `τ ≤ π` are not needed.
* The source's proof bounds `∫ |f'''|₊` below by `min(8/τ, 4/τ + 4/τ_max) f'(0)`,
  which exceeds `4 (1/τ + 1/M) f'(0)` only when `M > τ`; the sharp bound
  `taumax_sharp` covers every `M`. -/
theorem synchronized_of_circleGapMax_lt {f f₁ f₂ f₃ : ℝ → ℝ}
    (hf : Function.Periodic f (2 * π)) (h₁ : ∀ y, HasDerivAt f (f₁ y) y)
    (h₂ : ∀ y, HasDerivAt f₁ (f₂ y) y) (h₃ : ∀ y, HasDerivAt f₂ (f₃ y) y)
    (hc₃ : Continuous f₃) {τ : ℝ} (hτ : 0 < τ) (hneg : ∀ u ∈ Set.Ioo τ (2 * π - τ), f₁ u < 0)
    (hf₀ : 0 < f₁ 0) {M : ℝ}
    (hint : τ * ∫ u in -π..π, max (f₃ u) 0 ≤ 4 * (1 + τ / M) * f₁ 0) {x : Angles n}
    (hC1 : ∀ i, ∑ j, f (x i - x j) = 0)
    (hC2 : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, f₁ (x i - x j))
    (hM : (¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π)) → circleGapMax x < M) :
    ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π) := by
  by_contra hx
  have hg := circleGapMax_pos x (by have := two_le_card_circleSites x hx; omega)
  have hs := mul_le_mul_of_nonneg_left (taumax_sharp hf h₁ h₂ h₃ hc₃ hτ hneg hC1 hC2 hx) hτ.le
  have h1 : 1 / M < 1 / circleGapMax x := one_div_lt_one_div_of_lt hg (hM hx)
  have key : τ * (4 * (1 / τ + 1 / circleGapMax x) * f₁ 0) =
      4 * (τ * (1 / τ)) * f₁ 0 + 4 * (τ * (1 / circleGapMax x)) * f₁ 0 := by ring
  rw [mul_one_div_cancel hτ.ne'] at key
  rw [show τ / M = τ * (1 / M) by ring] at hint
  nlinarith [mul_pos (mul_pos hτ (sub_pos.2 h1)) hf₀]

/-- **Theorem (thm:main).**  Let `f` be `2π`-periodic with `f' = f₁`,
`f'' = f₂`, `f''' = f₃` continuous, `f'(0) > 0`, and `f' < 0` off `[-τ, τ]` on
`𝕋`.  If

  `τ ∫_{-π}^{π} |f'''|₊ ≤ 4 (1 + τ/2π) f'(0)`,

then every configuration that is stationary (C1) and cut-stable (C2) for
every `S` is synchronized.

Source: arXiv:2507.22857v1, §2, `thm:main`, which concludes that every
stationary point is locally unstable or synchronized.  Changed as in
`cor:main`: `f'(0) > 0` is added, "not locally unstable" is replaced by (C2)
for every `S`, and `τ ≤ π` is not needed.  It is `cor:main` with `M = 2π`:
with two positions, every gap is below `2π` (`circleGapMax_lt_two_pi`). -/
theorem synchronized_of_integral_le {f f₁ f₂ f₃ : ℝ → ℝ}
    (hf : Function.Periodic f (2 * π)) (h₁ : ∀ y, HasDerivAt f (f₁ y) y)
    (h₂ : ∀ y, HasDerivAt f₁ (f₂ y) y) (h₃ : ∀ y, HasDerivAt f₂ (f₃ y) y)
    (hc₃ : Continuous f₃) {τ : ℝ} (hτ : 0 < τ) (hneg : ∀ u ∈ Set.Ioo τ (2 * π - τ), f₁ u < 0)
    (hf₀ : 0 < f₁ 0)
    (hint : τ * ∫ u in -π..π, max (f₃ u) 0 ≤ 4 * (1 + τ / (2 * π)) * f₁ 0) {x : Angles n}
    (hC1 : ∀ i, ∑ j, f (x i - x j) = 0)
    (hC2 : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, f₁ (x i - x j)) :
    ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π) :=
  synchronized_of_circleGapMax_lt hf h₁ h₂ h₃ hc₃ hτ hneg hf₀ hint hC1 hC2 fun hx =>
    circleGapMax_lt_two_pi x (two_le_card_circleSites x hx)

/-- **`f'(0) > 0` is needed in `thm:main` and `cor:main`.**  For `f = 0` and
`τ = π`, every hypothesis of `synchronized_of_integral_le` but `f'(0) > 0`
holds (so does every hypothesis of `synchronized_of_circleGapMax_lt` with
`M = 2π`, by `circleGapMax_lt_two_pi`): every configuration is stationary (C1)
and cut-stable (C2), and `(0, π)` is not synchronized.  The Jacobian of (S1)
vanishes, so no stationary point is locally unstable either.

Source: arXiv:2507.22857v1, §2, `thm:main`, and §3, `cor:main`, which state
no sign of `f'(0)`. -/
theorem zero_kernel_counterexample :
    let f : ℝ → ℝ := fun _ => 0
    let x : Angles 2 := fun i => (i : ℕ) * π
    Function.Periodic f (2 * π) ∧ (∀ y, HasDerivAt f (f y) y) ∧ Continuous f ∧
      (0 : ℝ) < π ∧ (∀ u ∈ Set.Ioo π (2 * π - π), f u < 0) ∧
      π * ∫ u in -π..π, max (f u) 0 ≤ 4 * (1 + π / (2 * π)) * f 0 ∧
      (∀ i, ∑ j, f (x i - x j) = 0) ∧
      (∀ S : Finset (Idx 2), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, f (x i - x j)) ∧
      ¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π) := by
  intro f x
  refine ⟨fun _ => rfl, fun y => hasDerivAt_const y 0, continuous_const, pi_pos,
    fun u hu => by linarith [hu.1, hu.2], by simp [f], fun i => by simp [f],
    fun S => by simp [f], fun h => ?_⟩
  obtain ⟨k, hk⟩ := h 1 0
  have h1 : π * 1 = π * (2 * k) := by simp [x] at hk; linarith
  have h2 : (1 : ℤ) = 2 * k := by exact_mod_cast mul_left_cancel₀ pi_pos.ne' h1
  omega

/-- The hypotheses of `synchronized_of_circleGapMax_lt` and
`synchronized_of_integral_le` hold together for the Kuramoto kernel `f = sin`,
`τ = π/2`, `M = 2π` and two particles at `0`: `cos < 0` on `(π/2, 3π/2)`;
`|f'''|₊ = |-cos|₊ ≤ (1 - cos)/2`, whose integral is `π`; and
`π/2 · π ≤ 5 = 4 (1 + 1/4) cos 0`. -/
example :
    let x : Angles 2 := fun _ => 0
    Function.Periodic sin (2 * π) ∧ (∀ y, HasDerivAt sin (cos y) y) ∧
      (∀ y, HasDerivAt cos (-sin y) y) ∧ (∀ y, HasDerivAt (fun y => -sin y) (-cos y) y) ∧
      Continuous (fun y => -cos y) ∧ (0 : ℝ) < π / 2 ∧
      (∀ u ∈ Set.Ioo (π / 2) (2 * π - π / 2), cos u < 0) ∧ (0 : ℝ) < cos 0 ∧
      π / 2 * ∫ u in -π..π, max (-cos u) 0 ≤ 4 * (1 + π / 2 / (2 * π)) * cos 0 ∧
      (∀ i, ∑ j, sin (x i - x j) = 0) ∧
      (∀ S : Finset (Idx 2), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, cos (x i - x j)) ∧
      ((¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π)) → circleGapMax x < 2 * π) := by
  intro x
  have hI : ∫ u in -π..π, max (-cos u) 0 ≤ π := by
    calc ∫ u in -π..π, max (-cos u) 0 ≤ ∫ u in -π..π, (1 - cos u) / 2 :=
          intervalIntegral.integral_mono_on (by linarith [pi_pos])
            ((continuous_cos.neg.max continuous_const).intervalIntegrable _ _)
            (((continuous_const.sub continuous_cos).div_const 2).intervalIntegrable _ _)
            fun u _ => max_le (by linarith [neg_one_le_cos u]) (by linarith [cos_le_one u])
      _ = π := by
          rw [intervalIntegral.integral_div, intervalIntegral.integral_sub
            intervalIntegrable_const intervalIntegral.intervalIntegrable_cos, integral_one,
            integral_cos, sin_neg, sin_pi]
          ring
  have hπ : π / 2 / (2 * π) = 1 / 4 := by field_simp; ring
  refine ⟨sin_periodic, hasDerivAt_sin, hasDerivAt_cos, fun y => (hasDerivAt_sin y).fun_neg,
    continuous_cos.neg, by positivity, fun u hu => cos_neg_of_pi_div_two_lt_of_lt hu.1
      (by linarith [hu.2]), by rw [cos_zero]; norm_num, ?_, fun i => by simp [x],
    fun S => Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => by simp [x],
    fun hx => circleGapMax_lt_two_pi x (two_le_card_circleSites x hx)⟩
  rw [hπ, cos_zero]
  nlinarith [mul_le_mul_of_nonneg_left hI (by positivity : (0 : ℝ) ≤ π / 2),
    mul_lt_mul'' pi_lt_d2 pi_lt_d2 pi_pos.le pi_pos.le]

end Perspective
end Transformer
