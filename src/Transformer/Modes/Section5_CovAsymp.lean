/-
# The number of modes of a Gaussian KDE — the covariance against its scale

The three entries of `Σ_t` in `lem:moments-p`, each divided by its claimed
leading term, as in `Transformer.Modes.Section5_MomentsAsymp`: with
`ε = 1/β`, `x = t²/β`, `r = 1/(1+ε)`, `s = 1/(2+ε)`, the quotient is a
function of these that tends to `1`, plus a bounded function of `t` alone
(`t²e^{-t²/2}`, `e^{-t²/2}(1 - t² + ε)`, `e^{-t²/2}(1 + ε - t²)²`) times a
power of `ε` — the product of first moments, which is of lower order.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`).
-/

import Transformer.Modes.Section5_MomentsAsymp

open Real

namespace Transformer
namespace Modes

/-- Splitting `e^{-βt²/(2β+1)}` as `e^{-t²/2} e^{xs/2}`. -/
theorem exp_split_two {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    Real.exp (-(β * t ^ 2) / (2 * β + 1))
      = Real.exp (-(t ^ 2) / 2) * Real.exp (t ^ 2 / β * (β / (2 * β + 1)) / 2) := by
  have : 0 < 2 * β + 1 := by linarith
  rw [← Real.exp_add]; congr 1; field_simp; ring

/-- The hypothesis of `exp_split_two` is satisfiable. -/
example : Real.exp (-(1 * 0 ^ 2) / (2 * 1 + 1))
    = Real.exp (-(0 ^ 2) / 2) * Real.exp (0 ^ 2 / 1 * (1 / (2 * 1 + 1)) / 2) :=
  exp_split_two one_pos 0

/-- Splitting `e^{-βt²/(2(β+1))}` as `e^{-t²/2} e^{xr/2}`. -/
theorem exp_split_one {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    Real.exp (-(β * t ^ 2) / (2 * (β + 1)))
      = Real.exp (-(t ^ 2) / 2) * Real.exp (t ^ 2 / β * (β / (β + 1)) / 2) := by
  have : 0 < β + 1 := by linarith
  rw [← Real.exp_add]; congr 1; field_simp; ring

/-- The hypothesis of `exp_split_one` is satisfiable. -/
example : Real.exp (-(1 * 0 ^ 2) / (2 * (1 + 1)))
    = Real.exp (-(0 ^ 2) / 2) * Real.exp (0 ^ 2 / 1 * (1 / (1 + 1)) / 2) :=
  exp_split_one one_pos 0

/-- **`Var G(t)` against `2^{-5/2}β^{-3/2}e^{-t²/2}·2`.**

Source: arXiv:2412.09080v3, §5.2. -/
theorem varG_eq_mul {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sqMeanG β t - meanG β t ^ 2
      = (Real.exp (t ^ 2 / β * (1 / (2 + β⁻¹)) / 2) * (1 + t ^ 2 / β * (1 / (2 + β⁻¹)))
          * (2 * √2) * (1 / (2 + β⁻¹)) * √(1 / (2 + β⁻¹))
        - 2 * √2 * (t ^ 2 * Real.exp (-(t ^ 2) / 2))
          * Real.exp (t ^ 2 / β * (1 / (1 + β⁻¹)) / 2) ^ 2 * (1 / (1 + β⁻¹)) ^ 3
          * β⁻¹ * √β⁻¹)
        * (2 ^ (-(5 : ℝ) / 2) * (β ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)) * 2) := by
  rw [sqMeanG_eq hβ, meanG_eq hβ]
  have hr : 1 / (1 + β⁻¹) = β / (β + 1) := by field_simp
  have hs : 1 / (2 + β⁻¹) = β / (2 * β + 1) := by field_simp
  rw [hr, hs, exp_split_one hβ, exp_split_two hβ,
    Real.sqrt_div' _ (by linarith : (0 : ℝ) ≤ 2 * β + 1), Real.sqrt_inv,
    show (5 : ℝ) / 2 = (2 * ((2 : ℕ) : ℝ) + 1) / 2 by norm_num,
    rpow_odd_half (x := 2 * β + 1) (by linarith) 2,
    show (3 : ℝ) / 2 = (2 * ((1 : ℕ) : ℝ) + 1) / 2 by norm_num,
    rpow_odd_half (x := β + 1) (by linarith) 1,
    show -(5 : ℝ) / 2 = -((2 * ((2 : ℕ) : ℝ) + 1) / 2) by norm_num,
    Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2), rpow_odd_half (x := 2) two_pos 2,
    show -(3 : ℝ) / 2 = -((2 * ((1 : ℕ) : ℝ) + 1) / 2) by norm_num, Real.rpow_neg hβ.le,
    rpow_odd_half hβ, show t ^ 2 + 2 * β + 1 = t ^ 2 + (2 * β + 1) by ring]
  have h2 : 0 < √2 := by positivity
  have hb := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ β + 1)
  have hb0 : 0 < √(β + 1) := Real.sqrt_pos.mpr (by linarith)
  have hc := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ 2 * β + 1)
  have hc0 : 0 < √(2 * β + 1) := Real.sqrt_pos.mpr (by linarith)
  generalize √(β + 1) = b at hb hb0 ⊢
  generalize √(2 * β + 1) = c at hc hc0 ⊢
  rw [← hb, ← hc]
  have ha := Real.sq_sqrt hβ.le
  have ha0 : 0 < √β := Real.sqrt_pos.mpr hβ
  generalize √β = a at ha ha0 ⊢
  subst ha
  field_simp
  ring

/-- The hypothesis of `varG_eq_mul` is satisfiable: `β = 1`. -/
example (t : ℝ) := varG_eq_mul (β := 1) one_pos t

/-- **`Cov(G(t), G'(t))` against `2^{-5/2}β^{-3/2}e^{-t²/2}·(-t)`.**

Source: arXiv:2412.09080v3, §5.2. -/
theorem covG_eq_mul {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    mulMeanGG' β t - meanG β t * meanG' β t
      = (Real.exp (t ^ 2 / β * (1 / (2 + β⁻¹)) / 2)
          * (1 - β⁻¹ / 2 + t ^ 2 / β / 2 - (β⁻¹) ^ 2 / 2)
          * (8 * √2) * (1 / (2 + β⁻¹)) ^ 3 * √(1 / (2 + β⁻¹))
        + 4 * √2 * (Real.exp (-(t ^ 2) / 2) * (1 - t ^ 2 + β⁻¹))
          * Real.exp (t ^ 2 / β * (1 / (1 + β⁻¹)) / 2) ^ 2 * (1 / (1 + β⁻¹)) ^ 4
          * β⁻¹ * √β⁻¹)
        * (2 ^ (-(5 : ℝ) / 2) * (β ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)) * (-t)) := by
  rw [mulMeanGG'_eq hβ, meanG_eq hβ, meanG'_eq hβ]
  have hr : 1 / (1 + β⁻¹) = β / (β + 1) := by field_simp
  have hs : 1 / (2 + β⁻¹) = β / (2 * β + 1) := by field_simp
  rw [hr, hs, exp_split_one hβ, exp_split_two hβ,
    Real.sqrt_div' _ (by linarith : (0 : ℝ) ≤ 2 * β + 1), Real.sqrt_inv,
    show (7 : ℝ) / 2 = (2 * ((3 : ℕ) : ℝ) + 1) / 2 by norm_num,
    rpow_odd_half (x := 2 * β + 1) (by linarith) 3,
    show (5 : ℝ) / 2 = (2 * ((2 : ℕ) : ℝ) + 1) / 2 by norm_num,
    rpow_odd_half (x := β + 1) (by linarith) 2,
    show (3 : ℝ) / 2 = (2 * ((1 : ℕ) : ℝ) + 1) / 2 by norm_num,
    rpow_odd_half (x := β + 1) (by linarith) 1,
    show -(5 : ℝ) / 2 = -((2 * ((2 : ℕ) : ℝ) + 1) / 2) by norm_num,
    Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2), rpow_odd_half (x := 2) two_pos 2,
    show -(3 : ℝ) / 2 = -((2 * ((1 : ℕ) : ℝ) + 1) / 2) by norm_num, Real.rpow_neg hβ.le,
    rpow_odd_half hβ]
  have h2 : 0 < √2 := by positivity
  have hb := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ β + 1)
  have hb0 : 0 < √(β + 1) := Real.sqrt_pos.mpr (by linarith)
  have hc := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ 2 * β + 1)
  have hc0 : 0 < √(2 * β + 1) := Real.sqrt_pos.mpr (by linarith)
  generalize √(β + 1) = b at hb hb0 ⊢
  generalize √(2 * β + 1) = c at hc hc0 ⊢
  rw [← hb, ← hc]
  have ha := Real.sq_sqrt hβ.le
  have ha0 : 0 < √β := Real.sqrt_pos.mpr hβ
  generalize √β = a at ha ha0 ⊢
  subst ha
  field_simp
  ring

/-- The hypothesis of `covG_eq_mul` is satisfiable: `β = 1`. -/
example (t : ℝ) := covG_eq_mul (β := 1) one_pos t

/-- **`Var G'(t)` against `2^{-5/2}β^{-3/2}e^{-t²/2}·3β`.**

Source: arXiv:2412.09080v3, §5.2. -/
theorem varG'_eq_mul {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sqMeanG' β t - meanG' β t ^ 2
      = (Real.exp (t ^ 2 / β * (1 / (2 + β⁻¹)) / 2)
          * (1 + (t ^ 2 / β + 5 * β⁻¹) / 3
            + ((t ^ 2 / β) ^ 2 - 2 * (t ^ 2 / β) * β⁻¹ + 15 * (β⁻¹) ^ 2) / 12
            - ((t ^ 2 / β) * (β⁻¹) ^ 2 - 3 * (β⁻¹) ^ 3) / 6 + (β⁻¹) ^ 4 / 12)
          * (16 * √2) * (1 / (2 + β⁻¹)) ^ 4 * √(1 / (2 + β⁻¹))
        - 4 * √2 / 3 * (Real.exp (-(t ^ 2) / 2) * (1 + β⁻¹ - t ^ 2) ^ 2)
          * Real.exp (t ^ 2 / β * (1 / (1 + β⁻¹)) / 2) ^ 2 * (1 / (1 + β⁻¹)) ^ 5
          * (β⁻¹) ^ 2 * √β⁻¹)
        * (2 ^ (-(5 : ℝ) / 2) * (β ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)) * (3 * β)) := by
  rw [sqMeanG'_eq hβ, meanG'_eq hβ]
  have hr : 1 / (1 + β⁻¹) = β / (β + 1) := by field_simp
  have hs : 1 / (2 + β⁻¹) = β / (2 * β + 1) := by field_simp
  rw [hr, hs, exp_split_one hβ, exp_split_two hβ,
    Real.sqrt_div' _ (by linarith : (0 : ℝ) ≤ 2 * β + 1), Real.sqrt_inv,
    show (9 : ℝ) / 2 = (2 * ((4 : ℕ) : ℝ) + 1) / 2 by norm_num,
    rpow_odd_half (x := 2 * β + 1) (by linarith) 4,
    show (5 : ℝ) / 2 = (2 * ((2 : ℕ) : ℝ) + 1) / 2 by norm_num,
    rpow_odd_half (x := β + 1) (by linarith) 2,
    show -(5 : ℝ) / 2 = -((2 * ((2 : ℕ) : ℝ) + 1) / 2) by norm_num,
    Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2), rpow_odd_half (x := 2) two_pos 2,
    show -(3 : ℝ) / 2 = -((2 * ((1 : ℕ) : ℝ) + 1) / 2) by norm_num, Real.rpow_neg hβ.le,
    rpow_odd_half hβ]
  have h2 : 0 < √2 := by positivity
  have hb := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ β + 1)
  have hb0 : 0 < √(β + 1) := Real.sqrt_pos.mpr (by linarith)
  have hc := Real.sq_sqrt (by linarith : (0 : ℝ) ≤ 2 * β + 1)
  have hc0 : 0 < √(2 * β + 1) := Real.sqrt_pos.mpr (by linarith)
  generalize √(β + 1) = b at hb hb0 ⊢
  generalize √(2 * β + 1) = c at hc hc0 ⊢
  rw [← hb, ← hc]
  have ha := Real.sq_sqrt hβ.le
  have ha0 : 0 < √β := Real.sqrt_pos.mpr hβ
  generalize √β = a at ha ha0 ⊢
  subst ha
  field_simp
  ring

/-- The hypothesis of `varG'_eq_mul` is satisfiable: `β = 1`. -/
example (t : ℝ) := varG'_eq_mul (β := 1) one_pos t

end Modes
end Transformer
