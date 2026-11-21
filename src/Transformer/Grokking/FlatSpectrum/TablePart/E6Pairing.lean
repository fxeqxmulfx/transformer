module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6ParityCubic

/-!
# PT6: pairing a table with the flat kernel of `a³ + a b`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Since `C / √97` is a
contraction for the flat kernel `C` of `a³ + a b`, a table whose kernel correlates with `C` has a
large nuclear norm (`nuclearNorm_ge_pairing`). The distance from a phase to half of another phase
is one line of algebra (`norm_sq_sub_half`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix
open scoped ComplexOrder

attribute [local instance] fact_prime_97

/-- A double sum over the lines is a double sum over `ZMod 97`. -/
lemma sum_lines {M : Type*} [AddCommMonoid M] (G : ZMod 97 → ZMod 97 → M) :
    ∑ u : Fin 1 → ZMod 97, ∑ v : Fin 1 → ZMod 97, G (u 0) (v 0) = ∑ a, ∑ b, G a b := by
  rw [← (Equiv.funUnique (Fin 1) (ZMod 97)).sum_comp (fun a => ∑ b, G a b)]
  refine Finset.sum_congr rfl fun u _ => ?_
  exact (Equiv.funUnique (Fin 1) (ZMod 97)).sum_comp (fun b => G (u 0) b)

/-- **A table that correlates with the flat kernel has a large norm.** -/
lemma nuclearNorm_ge_pairing {f : ZMod 97 → ZMod 97 → ZMod 97} {s : ℝ}
    (hs : s ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
      (ZMod.stdAddChar (f a b - (a ^ 3 + a * b) + 24) : ℂ).re) :
    (√97)⁻¹ * s ≤ nuclearNorm (tableKernel f) := by
  have h := norm_trace_le_nuclearNorm isContraction_cubicLine (tableKernel f)
  rw [conjTranspose_smul, Matrix.smul_mul, trace_smul, norm_smul, norm_star, norm_inv,
    Complex.norm_real, Real.norm_of_nonneg (Real.sqrt_nonneg _)] at h
  have ht : (ZMod.stdAddChar (24 : ZMod 97) : ℂ) * (cubicLineᴴ * tableKernel f).trace =
      ∑ u : Fin 1 → ZMod 97, ∑ v : Fin 1 → ZMod 97,
        (ZMod.stdAddChar (f (u 0) (v 0) - ((u 0) ^ 3 + u 0 * v 0) + 24) : ℂ) := by
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, conjTranspose_apply, cubicLine,
      tableKernel, of_apply, star_stdAddChar_mul]
    conv_lhs => rw [Finset.sum_comm, Finset.mul_sum]
    refine Finset.sum_congr rfl fun u _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun v _ => ?_
    rw [← AddChar.map_add_eq_mul, add_comm (24 : ZMod 97)]
  have hre : s ≤ ‖(cubicLineᴴ * tableKernel f).trace‖ := by
    have hn : ‖(ZMod.stdAddChar (24 : ZMod 97) : ℂ) * (cubicLineᴴ * tableKernel f).trace‖ =
        ‖(cubicLineᴴ * tableKernel f).trace‖ := by
      rw [norm_mul]
      simp
    rw [← hn]
    refine le_trans hs (le_trans (le_of_eq ?_) (Complex.re_le_norm _))
    rw [ht, Complex.re_sum]
    simp only [Complex.re_sum]
    exact (sum_lines fun a b => (ZMod.stdAddChar (f a b - (a ^ 3 + a * b) + 24) : ℂ).re).symm
  exact le_trans (mul_le_mul_of_nonneg_left hre (inv_nonneg.2 (Real.sqrt_nonneg _))) h

example (f : ZMod 97 → ZMod 97 → ZMod 97) :
    (√97)⁻¹ * (∑ a : ZMod 97, ∑ b : ZMod 97,
      (ZMod.stdAddChar (f a b - (a ^ 3 + a * b) + 24) : ℂ).re) ≤ nuclearNorm (tableKernel f) :=
  nuclearNorm_ge_pairing le_rfl

/-- The distance from a phase to half of another phase. -/
lemma norm_sq_sub_half (x y : ZMod 97) :
    ‖(ZMod.stdAddChar x : ℂ) - ((1 / 2 : ℝ) : ℂ) * ZMod.stdAddChar y‖ ^ 2 =
      5 / 4 - (ZMod.stdAddChar (x - y) : ℂ).re := by
  have hz (w : ZMod 97) : (ZMod.stdAddChar w : ℂ).re * (ZMod.stdAddChar w : ℂ).re +
      (ZMod.stdAddChar w : ℂ).im * (ZMod.stdAddChar w : ℂ).im = 1 := by
    rw [← Complex.normSq_apply, ← Complex.sq_norm]
    simp
  have h1 : (ZMod.stdAddChar (x - y) : ℂ) = star (ZMod.stdAddChar y : ℂ) * ZMod.stdAddChar x :=
    (star_stdAddChar_mul y x).symm
  rw [Complex.sq_norm, Complex.normSq_apply, h1]
  simp only [Complex.sub_re, Complex.sub_im, Complex.mul_re, Complex.mul_im, Complex.ofReal_re,
    Complex.ofReal_im, Complex.star_def, Complex.conj_re, Complex.conj_im]
  linear_combination hz x + (1 / 4) * hz y

end Transformer.Grokking.FlatSpectrum.TablePart.E6
