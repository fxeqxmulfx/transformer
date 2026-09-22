/-
# The number of modes of a Gaussian KDE — the moments against their scale

The asymptotics of `lem:moments-p` are read off the closed forms of §5.2 of
arXiv:2412.09080v3 by dividing each by its claimed leading term.  This file
writes each quotient out, as an explicit function of

* `ε = 1/β`, `x = t²/β` — both `→ 0` in the regime of §5.2 — and
* a bounded function of `t` alone, `t²e^{-t²/2}` and its kin,

so that the leading term is recovered exactly: `u = F · v`, with `F → 1`
visibly.  The exponentials `e^{-βt²/(2(β+1))}` and `e^{-βt²/(2β+1)}` are
split as `e^{-t²/2}·e^{xr/2}` and `e^{-t²/2}·e^{xs/2}`, where `r = 1/(1+ε)`
and `s = 1/(2+ε)`: the source's remark that `exp Θ(t²/β) → 1`.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`).
-/

import Transformer.Modes.Section5_Moments

open Real Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-- `u ~ v` as soon as `u = F · v` with `F → 1`. -/
theorem isEquivalent_of_eq_mul {u v F : ℕ → ℝ} (hF : Tendsto F atTop (𝓝 1))
    (h : ∀ᶠ k in atTop, u k = F k * v k) : u ~[atTop] v := by
  have h1 : (fun k => F k - 1) =o[atTop] (fun _ => (1 : ℝ)) :=
    (isLittleO_one_iff ℝ).mpr (by simpa using hF.sub_const 1)
  refine ((h1.mul_isBigO (isBigO_refl v atTop)).congr' ?_ ?_)
  · filter_upwards [h] with k hk
    simp only [Pi.sub_apply, hk]
    ring
  · exact Eventually.of_forall fun _ => one_mul _

/-- The hypotheses of `isEquivalent_of_eq_mul` are satisfiable: `F = 1`. -/
example : (fun k : ℕ => (k : ℝ)) ~[atTop] fun k => (k : ℝ) :=
  isEquivalent_of_eq_mul (F := fun _ => 1) tendsto_const_nhds
    (Eventually.of_forall fun _ => (one_mul _).symm)

/-- `F → 1` survives adding a bounded sequence times one that tends to `0`:
the shape of every quotient of `Σ_t`, whose product-of-means term is a
bounded function of `t` times a power of `ε`. -/
theorem tendsto_add_mul_of_bdd {A b Ψ : ℕ → ℝ} {C : ℝ} (hA : Tendsto A atTop (𝓝 1))
    (hb : ∀ᶠ k in atTop, |b k| ≤ C) (hΨ : Tendsto Ψ atTop (𝓝 0)) :
    Tendsto (fun k => A k + b k * Ψ k) atTop (𝓝 1) := by
  have h0 : Tendsto (fun k => b k * Ψ k) atTop (𝓝 0) := by
    refine squeeze_zero_norm' ?_ (by simpa using hΨ.norm.const_mul C)
    filter_upwards [hb] with k hk
    rw [norm_mul]
    exact mul_le_mul_of_nonneg_right (by simpa using hk) (norm_nonneg _)
  simpa using hA.add h0

/-- The hypotheses of `tendsto_add_mul_of_bdd` are satisfiable. -/
example : Tendsto (fun _ : ℕ => (1 : ℝ) + 0 * 0) atTop (𝓝 1) :=
  tendsto_add_mul_of_bdd (C := 0) tendsto_const_nhds (Eventually.of_forall fun _ => by simp)
    tendsto_const_nhds

/-- `t² e^{-t²/2} ≤ 2`, from `e^{t²/2} ≥ 1 + t²/2`. -/
theorem sq_mul_exp_neg_half_sq_le (t : ℝ) : t ^ 2 * Real.exp (-(t ^ 2) / 2) ≤ 2 := by
  have h := Real.add_one_le_exp (t ^ 2 / 2)
  have h1 : Real.exp (-(t ^ 2) / 2) * Real.exp (t ^ 2 / 2) = 1 := by
    rw [← Real.exp_add]; ring_nf; exact Real.exp_zero
  nlinarith [Real.exp_pos (-(t ^ 2) / 2), sq_nonneg t]

/-- `t⁴ e^{-t²/2} ≤ 8`, from `e^{t²/2} ≥ (t²/2)²/2`. -/
theorem pow_four_mul_exp_neg_half_sq_le (t : ℝ) : t ^ 4 * Real.exp (-(t ^ 2) / 2) ≤ 8 := by
  have h := Real.quadratic_le_exp_of_nonneg (by positivity : 0 ≤ t ^ 2 / 2)
  have h1 : Real.exp (-(t ^ 2) / 2) * Real.exp (t ^ 2 / 2) = 1 := by
    rw [← Real.exp_add]; ring_nf; exact Real.exp_zero
  nlinarith [Real.exp_pos (-(t ^ 2) / 2), sq_nonneg t]

/-- **`E G(t)` against `β^{-3/2}e^{-t²/2}t`.**  The quotient is
`e^{xr/2} r √r`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem meanG_eq_mul {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG β t = Real.exp (t ^ 2 / β * (1 / (1 + β⁻¹)) / 2) * (1 / (1 + β⁻¹))
      * √(1 / (1 + β⁻¹)) * (β ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2) * t) := by
  rw [meanG_eq hβ]
  have hr : 1 / (1 + β⁻¹) = β / (β + 1) := by field_simp
  have hE : Real.exp (-(β * t ^ 2) / (2 * (β + 1)))
      = Real.exp (-(t ^ 2) / 2) * Real.exp (t ^ 2 / β * (β / (β + 1)) / 2) := by
    rw [← Real.exp_add]; congr 1; field_simp; ring
  rw [hr, hE, Real.sqrt_div' _ (by linarith : (0 : ℝ) ≤ β + 1),
    show (3 : ℝ) / 2 = (2 * ((1 : ℕ) : ℝ) + 1) / 2 by norm_num, rpow_odd_half (by linarith),
    show -(3 : ℝ) / 2 = -((2 * ((1 : ℕ) : ℝ) + 1) / 2) by norm_num, Real.rpow_neg hβ.le,
    rpow_odd_half hβ]
  have hb := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ β + 1)
  have hb0 : 0 < √(β + 1) := Real.sqrt_pos.mpr (by linarith)
  generalize √(β + 1) = b at hb hb0 ⊢
  rw [← hb]
  have ha := Real.sq_sqrt hβ.le
  have ha0 : 0 < √β := Real.sqrt_pos.mpr hβ
  generalize √β = a at ha ha0 ⊢
  subst ha
  field_simp

/-- The hypothesis of `meanG_eq_mul` is satisfiable. -/
example : meanG 1 0 = Real.exp (0 ^ 2 / 1 * (1 / (1 + 1⁻¹)) / 2) * (1 / (1 + 1⁻¹))
      * √(1 / (1 + 1⁻¹)) * (1 ^ (-(3 : ℝ) / 2) * Real.exp (-(0 ^ 2) / 2) * 0) :=
  meanG_eq_mul one_pos 0

/-- **`E G'(t)` against `β^{-3/2}e^{-t²/2}(1 - t² + 1/β)`.**  The quotient is
`e^{xr/2} r² √r`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem meanG'_eq_mul {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG' β t = Real.exp (t ^ 2 / β * (1 / (1 + β⁻¹)) / 2) * (1 / (1 + β⁻¹)) ^ 2
      * √(1 / (1 + β⁻¹))
      * (β ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2) * (1 - t ^ 2 + β⁻¹)) := by
  rw [meanG'_eq hβ]
  have hr : 1 / (1 + β⁻¹) = β / (β + 1) := by field_simp
  have hE : Real.exp (-(β * t ^ 2) / (2 * (β + 1)))
      = Real.exp (-(t ^ 2) / 2) * Real.exp (t ^ 2 / β * (β / (β + 1)) / 2) := by
    rw [← Real.exp_add]; congr 1; field_simp; ring
  rw [hr, hE, Real.sqrt_div' _ (by linarith : (0 : ℝ) ≤ β + 1),
    show (5 : ℝ) / 2 = (2 * ((2 : ℕ) : ℝ) + 1) / 2 by norm_num, rpow_odd_half (by linarith),
    show -(3 : ℝ) / 2 = -((2 * ((1 : ℕ) : ℝ) + 1) / 2) by norm_num, Real.rpow_neg hβ.le,
    rpow_odd_half hβ]
  have hb := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ β + 1)
  have hb0 : 0 < √(β + 1) := Real.sqrt_pos.mpr (by linarith)
  generalize √(β + 1) = b at hb hb0 ⊢
  rw [← hb]
  have ha := Real.sq_sqrt hβ.le
  have ha0 : 0 < √β := Real.sqrt_pos.mpr hβ
  generalize √β = a at ha ha0 ⊢
  subst ha
  field_simp
  ring

/-- The hypothesis of `meanG'_eq_mul` is satisfiable. -/
example : meanG' 1 0 = Real.exp (0 ^ 2 / 1 * (1 / (1 + 1⁻¹)) / 2) * (1 / (1 + 1⁻¹)) ^ 2
      * √(1 / (1 + 1⁻¹)) * (1 ^ (-(3 : ℝ) / 2) * Real.exp (-(0 ^ 2) / 2) * (1 - 0 ^ 2 + 1⁻¹)) :=
  meanG'_eq_mul one_pos 0

end Modes
end Transformer
