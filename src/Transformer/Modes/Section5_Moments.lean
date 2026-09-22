/-
# The number of modes of a Gaussian KDE — the first two moments of `(G, G')`

§5.2 of arXiv:2412.09080v3, the proof of `lem:moments-p`: the five Gaussian
expectations `E G`, `E G'`, `E G²`, `E GG'`, `E G'²` in closed form, for
`X ~ N(0,1)` and the pair `(G(t), G'(t))` of `eq: Gt`.

**What the source says and what is carried here.**

* The source completes the square in the exponent,
  `β z²/2 + (z-t)²/2 = (β+1)u²/2 + βt²/(2(β+1))` for the first moments and
  `β z² + (z-t)²/2 = (2β+1)u²/2 + βt²/(2β+1)` for the second, and reads the
  moments of `u` off `lem:gaussian-int`.  All five are proved that way: the
  square is completed in the variable `x` of `N(0,1)`, at the centre
  `βt/(β+1)` (resp. `2βt/(2β+1)`), which is the source's `u` after `z = t - x`,
  and `integral_gaussianReal_of_square` does the rest.

* The source works with `β > 0`; that is the hypothesis carried.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`).
-/

import Transformer.Modes.Section5_GaussMoments

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

/-! ### The five moments -/

/-- `E G(t)`, for `X ~ N(0,1)`.

Source: arXiv:2412.09080v3, `lem:moments-p`. -/
noncomputable def meanG (β t : ℝ) : ℝ := ∫ x, bigG β t x ∂gaussianReal 0 1

/-- `E G'(t)`, for `X ~ N(0,1)`.

Source: arXiv:2412.09080v3, `lem:moments-p`. -/
noncomputable def meanG' (β t : ℝ) : ℝ := ∫ x, bigG' β t x ∂gaussianReal 0 1

/-- `E G(t)²`, for `X ~ N(0,1)`.

Source: arXiv:2412.09080v3, §5.2. -/
noncomputable def sqMeanG (β t : ℝ) : ℝ := ∫ x, bigG β t x ^ 2 ∂gaussianReal 0 1

/-- `E [G(t) G'(t)]`, for `X ~ N(0,1)`.

Source: arXiv:2412.09080v3, §5.2. -/
noncomputable def mulMeanGG' (β t : ℝ) : ℝ := ∫ x, bigG β t x * bigG' β t x ∂gaussianReal 0 1

/-- `E G'(t)²`, for `X ~ N(0,1)`.

Source: arXiv:2412.09080v3, §5.2. -/
noncomputable def sqMeanG' (β t : ℝ) : ℝ := ∫ x, bigG' β t x ^ 2 ∂gaussianReal 0 1

/-! ### Their closed forms -/

/-- **§5.2, `E G(t)`.**  `E G(t) = t e^{-βt²/(2(β+1))} / (β+1)^{3/2}`.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem meanG_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG β t = t * Real.exp (-(β * t ^ 2) / (2 * (β + 1))) / (β + 1) ^ ((3 : ℝ) / 2) := by
  have hb : 0 < β + 1 := by linarith
  have key := integral_gaussianReal_of_square (f := bigG β t)
    (a := (β + 1) / 2) (c := β * t / (β + 1))
    (C := Real.exp (-(β * t ^ 2) / (2 * (β + 1))))
    (p₀ := (t - β * t / (β + 1))) (p₁ := -1)
    (p₂ := 0) (p₃ := 0) (p₄ := 0)
    (by positivity) fun x => by
      have he : Real.exp (-x ^ 2 / 2) * Real.exp (-(β / 2) * (t - x) ^ 2)
          = Real.exp (-(β * t ^ 2) / (2 * (β + 1)))
            * Real.exp (-((β + 1) / 2) * (x - β * t / (β + 1)) ^ 2) := by
        simp only [← Real.exp_add]
        congr 1
        field_simp
        ring
      simp only [bigG]
      linear_combination (t - x) * he
  rw [meanG, key, show (3 : ℝ) / 2 = (2 * (((1 : ℕ)) : ℝ) + 1) / 2 by norm_num, rpow_odd_half hb,
    Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_div hb.le]
  have h2 : 0 < √2 := by positivity
  have hπ : 0 < √π := by positivity
  have hs : 0 < √(β + 1) := Real.sqrt_pos.mpr hb
  field_simp
  ring

/-- **§5.2, `E G'(t)`.**
`E G'(t) = e^{-βt²/(2(β+1))} (1 + β - βt²) / (β+1)^{5/2}`.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem meanG'_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG' β t = Real.exp (-(β * t ^ 2) / (2 * (β + 1))) * (1 + β - β * t ^ 2)
      / (β + 1) ^ ((5 : ℝ) / 2) := by
  have hb : 0 < β + 1 := by linarith
  have key := integral_gaussianReal_of_square (f := bigG' β t)
    (a := (β + 1) / 2) (c := β * t / (β + 1))
    (C := Real.exp (-(β * t ^ 2) / (2 * (β + 1))))
    (p₀ := 1 - β * (t - β * t / (β + 1)) ^ 2) (p₁ := 2 * β * (t - β * t / (β + 1)))
    (p₂ := -β) (p₃ := 0) (p₄ := 0)
    (by positivity) fun x => by
      have he : Real.exp (-x ^ 2 / 2) * Real.exp (-(β / 2) * (t - x) ^ 2)
          = Real.exp (-(β * t ^ 2) / (2 * (β + 1)))
            * Real.exp (-((β + 1) / 2) * (x - β * t / (β + 1)) ^ 2) := by
        simp only [← Real.exp_add]
        congr 1
        field_simp
        ring
      simp only [bigG']
      linear_combination (1 - β * (t - x) ^ 2) * he
  rw [meanG', key, show (5 : ℝ) / 2 = (2 * (((2 : ℕ)) : ℝ) + 1) / 2 by norm_num, rpow_odd_half hb,
    Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_div hb.le]
  have h2 : 0 < √2 := by positivity
  have hπ : 0 < √π := by positivity
  have hs : 0 < √(β + 1) := Real.sqrt_pos.mpr hb
  field_simp
  ring

/-- **§5.2, `E G(t)²`.**
`E G(t)² = e^{-βt²/(2β+1)} (t² + 2β + 1) / (2β+1)^{5/2}`.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem sqMeanG_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sqMeanG β t = Real.exp (-(β * t ^ 2) / (2 * β + 1)) * (t ^ 2 + 2 * β + 1)
      / (2 * β + 1) ^ ((5 : ℝ) / 2) := by
  have hb : 0 < 2 * β + 1 := by linarith
  have key := integral_gaussianReal_of_square (f := fun x => bigG β t x ^ 2)
    (a := (2 * β + 1) / 2) (c := 2 * β * t / (2 * β + 1))
    (C := Real.exp (-(β * t ^ 2) / (2 * β + 1)))
    (p₀ := (t - 2 * β * t / (2 * β + 1)) ^ 2) (p₁ := -2 * (t - 2 * β * t / (2 * β + 1)))
    (p₂ := 1) (p₃ := 0) (p₄ := 0)
    (by positivity) fun x => by
      have he : Real.exp (-x ^ 2 / 2) * Real.exp (-(β / 2) * (t - x) ^ 2) * Real.exp (-(β / 2) * (t - x) ^ 2)
          = Real.exp (-(β * t ^ 2) / (2 * β + 1))
            * Real.exp (-((2 * β + 1) / 2) * (x - 2 * β * t / (2 * β + 1)) ^ 2) := by
        simp only [← Real.exp_add]
        congr 1
        field_simp
        ring
      simp only [bigG]
      linear_combination (t - x) ^ 2 * he
  rw [sqMeanG, key, show (5 : ℝ) / 2 = (2 * (((2 : ℕ)) : ℝ) + 1) / 2 by norm_num, rpow_odd_half hb,
    Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_div hb.le]
  have h2 : 0 < √2 := by positivity
  have hπ : 0 < √π := by positivity
  have hs : 0 < √(2 * β + 1) := Real.sqrt_pos.mpr hb
  field_simp
  ring

/-- **§5.2, `E [G(t)G'(t)]`.**
`E [G(t)G'(t)] = e^{-βt²/(2β+1)} (-2β²t + βt - βt³ + t) / (2β+1)^{7/2}`.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem mulMeanGG'_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    mulMeanGG' β t = Real.exp (-(β * t ^ 2) / (2 * β + 1))
      * (-2 * β ^ 2 * t + β * t - β * t ^ 3 + t) / (2 * β + 1) ^ ((7 : ℝ) / 2) := by
  have hb : 0 < 2 * β + 1 := by linarith
  have key := integral_gaussianReal_of_square (f := fun x => bigG β t x * bigG' β t x)
    (a := (2 * β + 1) / 2) (c := 2 * β * t / (2 * β + 1))
    (C := Real.exp (-(β * t ^ 2) / (2 * β + 1)))
    (p₀ := (t - 2 * β * t / (2 * β + 1)) - β * (t - 2 * β * t / (2 * β + 1)) ^ 3) (p₁ := -1 + 3 * β * (t - 2 * β * t / (2 * β + 1)) ^ 2)
    (p₂ := -3 * β * (t - 2 * β * t / (2 * β + 1))) (p₃ := β) (p₄ := 0)
    (by positivity) fun x => by
      have he : Real.exp (-x ^ 2 / 2) * Real.exp (-(β / 2) * (t - x) ^ 2) * Real.exp (-(β / 2) * (t - x) ^ 2)
          = Real.exp (-(β * t ^ 2) / (2 * β + 1))
            * Real.exp (-((2 * β + 1) / 2) * (x - 2 * β * t / (2 * β + 1)) ^ 2) := by
        simp only [← Real.exp_add]
        congr 1
        field_simp
        ring
      simp only [bigG, bigG']
      linear_combination (t - x) * (1 - β * (t - x) ^ 2) * he
  rw [mulMeanGG', key, show (7 : ℝ) / 2 = (2 * (((3 : ℕ)) : ℝ) + 1) / 2 by norm_num, rpow_odd_half hb,
    Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_div hb.le]
  have h2 : 0 < √2 := by positivity
  have hπ : 0 < √π := by positivity
  have hs : 0 < √(2 * β + 1) := Real.sqrt_pos.mpr hb
  field_simp
  ring

/-- **§5.2, `E G'(t)²`.**
`E G'(t)² = e^{-βt²/(2β+1)} (12β⁴ + 4β³(t²+5) + β²(t⁴-2t²+15) - 2β(t²-3) + 1)
/ (2β+1)^{9/2}`.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem sqMeanG'_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sqMeanG' β t = Real.exp (-(β * t ^ 2) / (2 * β + 1))
      * (12 * β ^ 4 + 4 * β ^ 3 * (t ^ 2 + 5) + β ^ 2 * (t ^ 4 - 2 * t ^ 2 + 15)
        - 2 * β * (t ^ 2 - 3) + 1) / (2 * β + 1) ^ ((9 : ℝ) / 2) := by
  have hb : 0 < 2 * β + 1 := by linarith
  have key := integral_gaussianReal_of_square (f := fun x => bigG' β t x ^ 2)
    (a := (2 * β + 1) / 2) (c := 2 * β * t / (2 * β + 1))
    (C := Real.exp (-(β * t ^ 2) / (2 * β + 1)))
    (p₀ := 1 - 2 * β * (t - 2 * β * t / (2 * β + 1)) ^ 2 + β ^ 2 * (t - 2 * β * t / (2 * β + 1)) ^ 4) (p₁ := 4 * β * (t - 2 * β * t / (2 * β + 1)) - 4 * β ^ 2 * (t - 2 * β * t / (2 * β + 1)) ^ 3)
    (p₂ := -2 * β + 6 * β ^ 2 * (t - 2 * β * t / (2 * β + 1)) ^ 2) (p₃ := -4 * β ^ 2 * (t - 2 * β * t / (2 * β + 1))) (p₄ := β ^ 2)
    (by positivity) fun x => by
      have he : Real.exp (-x ^ 2 / 2) * Real.exp (-(β / 2) * (t - x) ^ 2) * Real.exp (-(β / 2) * (t - x) ^ 2)
          = Real.exp (-(β * t ^ 2) / (2 * β + 1))
            * Real.exp (-((2 * β + 1) / 2) * (x - 2 * β * t / (2 * β + 1)) ^ 2) := by
        simp only [← Real.exp_add]
        congr 1
        field_simp
        ring
      simp only [bigG']
      linear_combination (1 - β * (t - x) ^ 2) ^ 2 * he
  rw [sqMeanG', key, show (9 : ℝ) / 2 = (2 * (((4 : ℕ)) : ℝ) + 1) / 2 by norm_num, rpow_odd_half hb,
    Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_div hb.le]
  have h2 : 0 < √2 := by positivity
  have hπ : 0 < √π := by positivity
  have hs : 0 < √(2 * β + 1) := Real.sqrt_pos.mpr hb
  field_simp
  ring

/-- The hypothesis of the five closed forms is satisfiable: `β = 1`. -/
example : meanG 1 0 = 0 * Real.exp (-(1 * 0 ^ 2) / (2 * (1 + 1))) / (1 + 1) ^ ((3 : ℝ) / 2) :=
  meanG_eq one_pos 0

end Modes
end Transformer
