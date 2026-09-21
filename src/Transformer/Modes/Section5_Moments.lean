/-
# The number of modes of a Gaussian KDE — the first two moments of `(G, G')`

§5.2 of arXiv:2412.09080v3, the proof of `lem:moments-p`: the five Gaussian
expectations `E G`, `E G'`, `E G²`, `E GG'`, `E G'²` in closed form, for
`X ~ N(0,1)` and the pair `(G(t), G'(t))` of `eq: Gt`.

**What the source says and what is carried here.**

* The source completes the square in the exponent,
  `β z²/2 + (z-t)²/2 = (β+1)u²/2 + βt²/(2(β+1))` for the first moments and
  `β z² + (z-t)²/2 = (2β+1)u²/2 + βt²/(2β+1)` for the second, and reads the
  moments of `u` off `lem:gaussian-int`.  The five closed forms are stated
  here as theorems and are unproved.

* All five were checked numerically against quadrature (three values of
  `(β, t)`, agreement to `10⁻¹⁶`), so none of them is suspected of an error.

* The source works with `β > 0`; that is the hypothesis carried.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`).
-/

import Transformer.Modes.Section2_Gt

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

Not proved here.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem meanG_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG β t = t * Real.exp (-(β * t ^ 2) / (2 * (β + 1))) / (β + 1) ^ ((3 : ℝ) / 2) := by
  sorry

/-- **§5.2, `E G'(t)`.**
`E G'(t) = e^{-βt²/(2(β+1))} (1 + β - βt²) / (β+1)^{5/2}`.

Not proved here.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem meanG'_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG' β t = Real.exp (-(β * t ^ 2) / (2 * (β + 1))) * (1 + β - β * t ^ 2)
      / (β + 1) ^ ((5 : ℝ) / 2) := by
  sorry

/-- **§5.2, `E G(t)²`.**
`E G(t)² = e^{-βt²/(2β+1)} (t² + 2β + 1) / (2β+1)^{5/2}`.

Not proved here.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem sqMeanG_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sqMeanG β t = Real.exp (-(β * t ^ 2) / (2 * β + 1)) * (t ^ 2 + 2 * β + 1)
      / (2 * β + 1) ^ ((5 : ℝ) / 2) := by
  sorry

/-- **§5.2, `E [G(t)G'(t)]`.**
`E [G(t)G'(t)] = e^{-βt²/(2β+1)} (-2β²t + βt - βt³ + t) / (2β+1)^{7/2}`.

Not proved here.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem mulMeanGG'_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    mulMeanGG' β t = Real.exp (-(β * t ^ 2) / (2 * β + 1))
      * (-2 * β ^ 2 * t + β * t - β * t ^ 3 + t) / (2 * β + 1) ^ ((7 : ℝ) / 2) := by
  sorry

/-- **§5.2, `E G'(t)²`.**
`E G'(t)² = e^{-βt²/(2β+1)} (12β⁴ + 4β³(t²+5) + β²(t⁴-2t²+15) - 2β(t²-3) + 1)
/ (2β+1)^{9/2}`.

Not proved here.

Source: arXiv:2412.09080v3, §5.2 (`sec:moments`). -/
theorem sqMeanG'_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sqMeanG' β t = Real.exp (-(β * t ^ 2) / (2 * β + 1))
      * (12 * β ^ 4 + 4 * β ^ 3 * (t ^ 2 + 5) + β ^ 2 * (t ^ 4 - 2 * t ^ 2 + 15)
        - 2 * β * (t ^ 2 - 3) + 1) / (2 * β + 1) ^ ((9 : ℝ) / 2) := by
  sorry

/-- The hypothesis of the five closed forms is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

end Modes
end Transformer
