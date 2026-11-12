module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction.Extraction

/-!
# Amplitude error bounds the remaining coefficient products

Source: `review/flat_spectrum.md`, CM20, quantitative bound. This concerns an
exact separated Fourier subspace. Error in estimating that subspace and the
convergence of a numerical extraction algorithm are separate questions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction

open Finset

variable {q : ℕ} [NeZero q] {ι : Type*} [Fintype ι]

/-- Subtracting any real constant leaves the nonconstant amplitude Fourier
coefficient unchanged, CM20. -/
theorem powerCoefficient_sub (k : ι → ZMod q) (a : ι → ℂ) (ρ : ℝ)
    {t : ZMod q} (ht : t ≠ 0) : powerCoefficient k a t =
      ∑ x, star (ZMod.stdAddChar (x * t)) *
        ((Complex.normSq (wave k a x) : ℂ) - (ρ : ℂ)) := by
  simp_rw [mul_sub, sum_sub_distrib]
  rw [← sum_mul, ← star_sum, sum_stdAddChar_mul]
  simp [ht, powerCoefficient]

/-- Total squared-amplitude deviation controls every cross-mode coefficient
product when differences are separated, CM20. -/
theorem pair_le_deviation {k : ι → ZMod q} (h : Separated k) (a : ι → ℂ) (ρ : ℝ)
    {i j : ι} (hij : i ≠ j) :
    (q : ℝ) * (‖a i‖ * ‖a j‖) ≤ ∑ x, |Complex.normSq (wave k a x) - ρ| := by
  calc
    _ = ‖powerCoefficient k a (k j - k i)‖ := by
      rw [h.coefficient a hij]
      simp
    _ ≤ ∑ x, ‖star (ZMod.stdAddChar (x * (k j - k i))) *
        ((Complex.normSq (wave k a x) : ℂ) - (ρ : ℂ))‖ := by
      rw [powerCoefficient_sub k a ρ (h.difference_ne_zero hij)]
      exact norm_sum_le _ _
    _ = _ := by
      apply sum_congr rfl
      intro x hx
      rw [norm_mul, norm_star, ZMod.stdAddChar_apply, Circle.norm_coe, one_mul,
        ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs]

/-- A uniform squared-amplitude error of at most `ε` bounds every product of
two different coefficient magnitudes by `ε`, CM20. -/
theorem approximate_pair {k : ι → ZMod q} (h : Separated k) (a : ι → ℂ)
    (ρ ε : ℝ) (herror : ∀ x, |Complex.normSq (wave k a x) - ρ| ≤ ε)
    {i j : ι} (hij : i ≠ j) : ‖a i‖ * ‖a j‖ ≤ ε := by
  have hs : (∑ x, |Complex.normSq (wave k a x) - ρ|) ≤ (q : ℝ) * ε := by
    calc
      _ ≤ ∑ x : ZMod q, ε := sum_le_sum fun x hx => herror x
      _ = _ := by simp
  exact le_of_mul_le_mul_left ((pair_le_deviation h a ρ hij).trans hs)
    (Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne q)))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction
