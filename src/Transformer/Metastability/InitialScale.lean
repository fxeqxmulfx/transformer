/-
# Numerical bounds in the projected Gaussian-mixture proposition

The centred-cap condition bounds the cap height in the nontrivial range
`ε < 2`. The mixture scale condition then forces the quadratic Gaussian
radius threshold above `2d + log n`, which is enough for the elementary
exponential-moment estimate.

Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`.
-/

import Transformer.Metastability.InitialGeometry
import Transformer.Metastability.NotSeparated

open Real

namespace Transformer
namespace Metastability

/-- In the nontrivial cap range, the centred condition forces `ε < 1/4`.
The lower bound `α(ε) ≥ -1` and the positivity of the logarithm are enough.
Source: arXiv:2410.06833v1, §4, `d: separated_mixtures`. -/
theorem centered_epsilon_lt_quarter_of_lt_two (d n r : ℕ) (hn : 2 ≤ n)
    (β ε : ℝ) (hβ : 0 < β) (hε : 0 < ε) (hε2 : ε < 2)
    (w : Idx r → SSphere d) (hcent : isCentered d n β ε r w) :
    ε < 1 / 4 := by
  have hα : -1 ≤ αDist d r w ε :=
    neg_one_le_sSup fun c hc => mem_Icc_of_mem_αDist hc
  have hnℝ : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have harg : 1 < 2 * (n : ℝ) ^ 2 / ε := by
    rw [one_lt_div hε]
    nlinarith
  have hlog : 0 ≤ Real.log (2 * (n : ℝ) ^ 2 / ε) :=
    Real.log_nonneg harg.le
  have hterm : 0 ≤ β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / ε) :=
    mul_nonneg (inv_nonneg.mpr hβ.le) hlog
  have hγ := hcent.2
  unfold γβ at hγ
  linarith

/-- The displayed scale hypothesis gives enough Gaussian-radius slack for
an exponential Markov bound: with `δ = σ/√r`, its threshold
`ε/(8δ²)` exceeds `2d + log n` when `ε < 1/4`.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem mixture_scale_radius_lower (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (δ ε : ℝ) (hδpos : 0 < δ) (hε : 0 < ε) (hε4 : ε < 1 / 4)
    (hscale : 6 * δ * Real.sqrt d / (1 + δ * Real.sqrt d) +
      δ * Real.sqrt (2 * d * Real.log n) ≤ ε) :
    2 * (d : ℝ) + Real.log n ≤ ε / (8 * δ ^ 2) := by
  have hdℝ : (2 : ℝ) ≤ d := by exact_mod_cast hd
  have hnℝ : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hlog : 0 ≤ Real.log n := Real.log_nonneg (by linarith)
  have hrootd : 0 ≤ Real.sqrt (d : ℝ) := Real.sqrt_nonneg _
  have hrootlog : 0 ≤ Real.sqrt (2 * (d : ℝ) * Real.log n) := Real.sqrt_nonneg _
  let u : ℝ := δ * Real.sqrt d
  have hu0 : 0 ≤ u := mul_nonneg hδpos.le hrootd
  have hden : 0 < 1 + u := by linarith
  have hA : 6 * u / (1 + u) ≤ ε := by
    dsimp [u]
    have hAnonneg : 0 ≤ δ * Real.sqrt (2 * (d : ℝ) * Real.log n) := by
      positivity
    have heq : 6 * (δ * Real.sqrt (d : ℝ)) / (1 + δ * Real.sqrt d) =
        6 * δ * Real.sqrt d / (1 + δ * Real.sqrt d) := by ring
    rw [heq]
    linarith
  have hA' : 6 * u ≤ ε * (1 + u) := (div_le_iff₀ hden).mp hA
  have hcross : ε * u ≤ u / 4 := by
    nlinarith [mul_nonneg (le_of_lt (sub_pos.mpr hε4)) hu0]
  have hu : u ≤ ε / 5 := by linarith
  have hu_sq : δ ^ 2 * (d : ℝ) ≤ (ε / 5) ^ 2 := by
    have hroot : (Real.sqrt (d : ℝ)) ^ 2 = (d : ℝ) :=
      Real.sq_sqrt (by linarith)
    have hsquare := pow_le_pow_left₀ hu0 hu 2
    dsimp [u] at hsquare
    rw [mul_pow, hroot] at hsquare
    exact hsquare
  have hB : δ * Real.sqrt (2 * (d : ℝ) * Real.log n) ≤ ε := by
    have hAnonneg : 0 ≤ 6 * δ * Real.sqrt d / (1 + δ * Real.sqrt d) := by
      positivity
    linarith
  have hB_sq : δ ^ 2 * (2 * (d : ℝ) * Real.log n) ≤ ε ^ 2 := by
    have hroot : (Real.sqrt (2 * (d : ℝ) * Real.log n)) ^ 2 =
        2 * (d : ℝ) * Real.log n := Real.sq_sqrt (by positivity)
    have hsquare := pow_le_pow_left₀ (mul_nonneg hδpos.le hrootlog) hB 2
    rw [mul_pow, hroot] at hsquare
    exact hsquare
  have hεsq : ε ^ 2 ≤ ε / 4 := by
    nlinarith [mul_nonneg hε.le (le_of_lt (sub_pos.mpr hε4))]
  have hδ2 : 0 < 8 * δ ^ 2 := by positivity
  have hR12 : 12 * (d : ℝ) ≤ ε / (8 * δ ^ 2) := by
    apply (le_div_iff₀ hδ2).mpr
    nlinarith [hu_sq, hεsq]
  have hprod : 0 ≤ δ ^ 2 * Real.log n := mul_nonneg (sq_nonneg _) hlog
  have hRlog : 2 * Real.log n ≤ ε / (8 * δ ^ 2) := by
    apply (le_div_iff₀ hδ2).mpr
    nlinarith [hB_sq, hεsq, mul_nonneg (by linarith : 0 ≤ (d : ℝ) - 2) hprod]
  linarith

/-- The numerical hypotheses are satisfiable at `d = n = 2`, sufficiently
small noise and cap height `ε = 1/8`. -/
example : (2 : ℕ) ≤ 2 ∧ (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 / 100 ∧
    (0 : ℝ) < 1 / 8 ∧ (1 / 8 : ℝ) < 1 / 4 := by norm_num

end Metastability
end Transformer
