/-
# The two estimates of `lemma:taumax`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §3, proof of `lemma:taumax`.

The two estimates of the proof that do not see the configuration: the
integral inequality on one gap, and the counting step over all gaps.

On a gap `[a, b]` with `χ(a) = χ(b)`, the weight `Ψ(y) = (y - a)(b - y)/2`,
which vanishes at both ends and has `Ψ'' = -1`, turns two integrations by
parts into

  `∫_a^b Ψ φ'' = (b - a)(φ(a) + φ(b))/2`,   `φ = χ'`

(`integral_bump_mul_eq`), and `0 ≤ Ψ ≤ (b - a)²/8` gives
`φ(a) + φ(b) ≤ (b - a)/4 ∫_a^b |φ''|₊` (`add_le_mul_integral_max_zero`).  The
source takes `Ψ_i = a_i (x - θ_i)(θ_{i+1} - x)/2` and later `a_i = 1/Δ_i`;
here `a_i = 1`, and `1/Δ_i` is divided out when the gaps are summed.

For a configuration, `χ = Σ_j f(· - x_j)` and its derivatives are sums of
translates (`hasDerivAt_sum_comp_sub`), periodic when `f` is
(`periodic_of_hasDerivAt`); the positive part of a sum is at most the sum of
the positive parts (`max_sum_le_sum_max`), and a translate of a periodic
function has the same integral over every period
(`integral_comp_sub_eq_of_periodic`).

Over the gaps, `Σ_k w_k/Δ_k ≥ m/τ + m/Δ_ℓ` once `Σ w = 2m`, `w_ℓ ≤ m`, and
every other gap is at most `τ` and at most `Δ_ℓ` (`le_sum_div_of_le`).
-/

import Transformer.Perspective.CircleSites
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Periodic
import Mathlib.Analysis.Calculus.Deriv.Shift

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

/-- **The derivative of a periodic function is periodic.** -/
theorem periodic_of_hasDerivAt {f f' : ℝ → ℝ} {T : ℝ} (hf : Function.Periodic f T)
    (h : ∀ y, HasDerivAt f (f' y) y) : Function.Periodic f' T := fun y => by
  have hT := (h (y + T)).comp_add_const y T
  rw [show (fun z => f (z + T)) = f from funext hf] at hT
  exact hT.unique (h y)

/-- The sum of the translates `f(· - c_j)` has the sum of the translates of
`f'` as its derivative. -/
theorem hasDerivAt_sum_comp_sub {ι : Type*} (s : Finset ι) (c : ι → ℝ) {f f' : ℝ → ℝ}
    (h : ∀ y, HasDerivAt f (f' y) y) (y : ℝ) :
    HasDerivAt (fun z => ∑ j ∈ s, f (z - c j)) (∑ j ∈ s, f' (y - c j)) y :=
  HasDerivAt.fun_sum fun j _ => (h (y - c j)).comp_sub_const y (c j)

/-- The positive part of a sum is at most the sum of the positive parts. -/
theorem max_sum_le_sum_max {ι : Type*} (s : Finset ι) (a : ι → ℝ) :
    max (∑ j ∈ s, a j) 0 ≤ ∑ j ∈ s, max (a j) 0 :=
  max_le (Finset.sum_le_sum fun _ _ => le_max_left _ _)
    (Finset.sum_nonneg fun _ _ => le_max_right _ _)

/-- **A translate of a `2π`-periodic function has the same integral over
every period**: `∫_t^{t+2π} g(y - c) dy = ∫_{-π}^{π} g`. -/
theorem integral_comp_sub_eq_of_periodic {g : ℝ → ℝ} (hg : Function.Periodic g (2 * π))
    (t c : ℝ) : ∫ y in t..t + 2 * π, g (y - c) = ∫ u in -π..π, g u := by
  rw [intervalIntegral.integral_comp_sub_right, show t + 2 * π - c = t - c + 2 * π by ring,
    hg.intervalIntegral_add_eq (t - c) (-π), show -π + 2 * π = π by ring]

/-- **Two integrations by parts on a gap.**  If `χ' = φ`, `φ' = φ₁`, `φ₁' = φ₂`
with `φ₂` continuous, and `χ(a) = χ(b)`, then with `Ψ(y) = (y - a)(b - y)/2`

  `∫_a^b Ψ φ₂ = (b - a)(φ(a) + φ(b))/2`.

Source: arXiv:2507.22857v1, §3, proof of `lemma:taumax`, the identity for
`⟨Ψ_i, φ''⟩` with `a_i = 1`, where `χ(θ_i) = χ(θ_{i+1})` comes from (C1). -/
theorem integral_bump_mul_eq {χ φ φ₁ φ₂ : ℝ → ℝ} {a b : ℝ}
    (hχ : ∀ y, HasDerivAt χ (φ y) y) (hφ : ∀ y, HasDerivAt φ (φ₁ y) y)
    (hφ₁ : ∀ y, HasDerivAt φ₁ (φ₂ y) y) (hφ₂ : Continuous φ₂) (hab : χ a = χ b) :
    ∫ y in a..b, (y - a) * (b - y) / 2 * φ₂ y = (b - a) * (φ a + φ b) / 2 := by
  have hG : ∀ y, HasDerivAt
      (fun y => (y - a) * (b - y) / 2 * φ₁ y - (a + b - 2 * y) / 2 * φ y - χ y)
      ((y - a) * (b - y) / 2 * φ₂ y) y := fun y => by
    have hΨ := (((hasDerivAt_id' y).sub_const a).fun_mul
      ((hasDerivAt_const y b).fun_sub (hasDerivAt_id' y))).div_const 2
    have hL := ((hasDerivAt_const y (a + b)).fun_sub ((hasDerivAt_id' y).const_mul 2)).div_const 2
    exact (((hΨ.fun_mul (hφ₁ y)).fun_sub (hL.fun_mul (hφ y))).fun_sub (hχ y)).congr_deriv
      (by ring)
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun y _ => hG y)
    ((by fun_prop : Continuous fun y => (y - a) * (b - y) / 2 * φ₂ y).intervalIntegrable a b)]
  rw [hab]
  ring

/-- **The gap inequality of `lemma:taumax`.**  If `χ' = φ`, `φ' = φ₁`,
`φ₁' = φ₂` with `φ₂` continuous, `a < b` and `χ(a) = χ(b)`, then

  `φ(a) + φ(b) ≤ (b - a)/4 ∫_a^b |φ₂|₊`.

Source: arXiv:2507.22857v1, §3, proof of `lemma:taumax`: from `Ψ_i ≤ a_i Δ_i²/8`,
`⟨a_i Δ_i² 1_{(θ_i, θ_{i+1})}, |φ''|₊⟩ ≥ 4 a_i Δ_i (φ(θ_i) + φ(θ_{i+1}))`, here with
`a_i = 1` and before `φ(θ_i) ≥ N_i f'(0)` is used. -/
theorem add_le_mul_integral_max_zero {χ φ φ₁ φ₂ : ℝ → ℝ} {a b : ℝ} (hab : a < b)
    (hχ : ∀ y, HasDerivAt χ (φ y) y) (hφ : ∀ y, HasDerivAt φ (φ₁ y) y)
    (hφ₁ : ∀ y, HasDerivAt φ₁ (φ₂ y) y) (hφ₂ : Continuous φ₂) (hχab : χ a = χ b) :
    φ a + φ b ≤ (b - a) / 4 * ∫ y in a..b, max (φ₂ y) 0 := by
  have hΨ : Continuous fun y : ℝ => (y - a) * (b - y) / 2 := by fun_prop
  have hmax : Continuous fun y => max (φ₂ y) 0 := hφ₂.max continuous_const
  have h1 : ∫ y in a..b, (y - a) * (b - y) / 2 * φ₂ y ≤
      ∫ y in a..b, (b - a) ^ 2 / 8 * max (φ₂ y) 0 := by
    refine intervalIntegral.integral_mono_on hab.le ((hΨ.mul hφ₂).intervalIntegrable a b)
      ((continuous_const.mul hmax).intervalIntegrable a b) fun y hy => ?_
    have h0 : 0 ≤ (y - a) * (b - y) / 2 :=
      div_nonneg (mul_nonneg (sub_nonneg.2 hy.1) (sub_nonneg.2 hy.2)) zero_le_two
    have hle : (y - a) * (b - y) / 2 ≤ (b - a) ^ 2 / 8 := by
      nlinarith [sq_nonneg (2 * y - a - b)]
    calc (y - a) * (b - y) / 2 * φ₂ y ≤ (y - a) * (b - y) / 2 * max (φ₂ y) 0 :=
          mul_le_mul_of_nonneg_left (le_max_left _ _) h0
      _ ≤ (b - a) ^ 2 / 8 * max (φ₂ y) 0 := mul_le_mul_of_nonneg_right hle (le_max_right _ _)
  rw [integral_bump_mul_eq hχ hφ hφ₁ hφ₂ hχab, intervalIntegral.integral_const_mul,
    show (b - a) * (φ a + φ b) / 2 = (b - a) / 2 * (φ a + φ b) by ring,
    show (b - a) ^ 2 / 8 * ∫ y in a..b, max (φ₂ y) 0 =
      (b - a) / 2 * ((b - a) / 4 * ∫ y in a..b, max (φ₂ y) 0) by ring] at h1
  exact le_of_mul_le_mul_left h1 (by linarith)

/-- The hypotheses of `periodic_of_hasDerivAt`, `hasDerivAt_sum_comp_sub`,
`integral_comp_sub_eq_of_periodic`, `integral_bump_mul_eq` and
`add_le_mul_integral_max_zero` hold for `χ = sin` on `[0, π]`, with `φ = cos`,
`φ' = -sin`, `φ'' = -cos`, and the period `2π` of `sin`. -/
example : Function.Periodic sin (2 * π) ∧ (0 : ℝ) < π ∧ (∀ y, HasDerivAt sin (cos y) y) ∧
    (∀ y, HasDerivAt cos (-sin y) y) ∧ (∀ y, HasDerivAt (fun y => -sin y) (-cos y) y) ∧
    Continuous (fun y => -cos y) ∧ sin 0 = sin π :=
  ⟨sin_periodic, pi_pos, hasDerivAt_sin, hasDerivAt_cos, fun y => (hasDerivAt_sin y).neg,
    continuous_cos.neg, by simp⟩

/-- **The counting step of `lemma:taumax`.**  Let `Δ_k > 0`, `w_k ≥ 0` on `s`,
with `Σ_s w = 2m` and `w_ℓ ≤ m`.  If every `Δ_k`, `k ≠ ℓ`, is at most `τ` and
at most `Δ_ℓ`, then `m/τ + m/Δ_ℓ ≤ Σ_s w_k/Δ_k`.

Source: arXiv:2507.22857v1, §3, the last display of the proof of
`lemma:taumax`, with `w_k = N_k + N_{k+1}`, `m = n` and `Δ_ℓ = τ_max`.  The
source bounds `1/Δ_k` below by `1/τ` only, which gives `8/τ` when
`τ_max ≤ τ`; with `1/Δ_k ≥ max(1/τ, 1/τ_max)` it gives `1/τ + 1/τ_max`. -/
theorem le_sum_div_of_le {s : Finset ℕ} {w Δ : ℕ → ℝ} {ℓ : ℕ} {m τ : ℝ} (hℓ : ℓ ∈ s)
    (hΔ : ∀ k ∈ s, 0 < Δ k) (hw : ∀ k ∈ s, 0 ≤ w k)
    (hle : ∀ k ∈ s, k ≠ ℓ → Δ k ≤ τ ∧ Δ k ≤ Δ ℓ) (hsum : ∑ k ∈ s, w k = 2 * m)
    (hwℓ : w ℓ ≤ m) : m / τ + m / Δ ℓ ≤ ∑ k ∈ s, w k / Δ k := by
  set c := max (1 / τ) (1 / Δ ℓ)
  have hrest : c * (2 * m - w ℓ) ≤ ∑ k ∈ s.erase ℓ, w k / Δ k := by
    rw [← Finset.sum_erase_add _ _ hℓ] at hsum
    rw [show 2 * m - w ℓ = ∑ k ∈ s.erase ℓ, w k by linarith, Finset.mul_sum]
    refine Finset.sum_le_sum fun k hk => ?_
    obtain ⟨hkℓ, hks⟩ := Finset.mem_erase.1 hk
    obtain ⟨h1, h2⟩ := hle k hks hkℓ
    have hc : c ≤ 1 / Δ k := max_le (one_div_le_one_div_of_le (hΔ k hks) h1)
      (one_div_le_one_div_of_le (hΔ k hks) h2)
    rw [div_eq_mul_one_div, mul_comm]
    exact mul_le_mul_of_nonneg_left hc (hw k hks)
  rw [← Finset.add_sum_erase _ _ hℓ, div_eq_mul_one_div (w ℓ), div_eq_mul_one_div m τ,
    div_eq_mul_one_div m]
  have hm : 0 ≤ m := (hw ℓ hℓ).trans hwℓ
  nlinarith [mul_nonneg (sub_nonneg.2 hwℓ) (sub_nonneg.2 (le_max_right (1 / τ) (1 / Δ ℓ))),
    mul_nonneg hm (sub_nonneg.2 (le_max_left (1 / τ) (1 / Δ ℓ)))]

/-- The hypotheses of `le_sum_div_of_le` hold for two unit gaps, `τ = 1`,
`w = 1`, `m = 1` and `ℓ = 0`; then both sides are `2`. -/
example :
    let w : ℕ → ℝ := fun _ => 1
    0 ∈ Finset.range 2 ∧ (∀ k ∈ Finset.range 2, 0 < w k) ∧
      (∀ k ∈ Finset.range 2, 0 ≤ w k) ∧
      (∀ k ∈ Finset.range 2, k ≠ 0 → w k ≤ 1 ∧ w k ≤ w 0) ∧
      ∑ k ∈ Finset.range 2, w k = 2 * 1 ∧ w 0 ≤ 1 := by
  intro w
  norm_num [w]

end Perspective
end Transformer
