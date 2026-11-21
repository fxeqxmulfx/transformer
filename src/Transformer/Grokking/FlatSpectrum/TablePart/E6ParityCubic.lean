module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import Transformer.Grokking.FlatSpectrum.Kernels
public import Transformer.Grokking.FlatSpectrum.SeenShare
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Lines

/-!
# PT6: M misses a table one step away from `a³ + a b`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The kernel `C` of
`a³ + a b` has `Cᴴ C = 97`, so `C / √97` is a contraction. A table within one step of `a³ + a b` has
phases within `2π / 97` of those of `C`, so `tr(Cᴴ K)` has real part at least `0.9 · 97²`, and
`‖K‖_S1 ≥ 0.9 · 97^{3/2}`. A minimum over the training half has `‖K‖²_S1 ≤ 97 · 4704`, which is
less. The table of `parity-cubic` is such a table.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix
open scoped ComplexOrder

attribute [local instance] fact_prime_97

/-- The kernel of `a³ + a b` on the lines. -/
noncomputable def cubicLine : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  tableKernel fun a b => a ^ 3 + a * b

lemma conjTranspose_mul_cubicLine : cubicLineᴴ * cubicLine = ((97 : ℕ) : ℂ) • 1 := by
  have h : cubicLine = (cubicKernel 97).submatrix (Equiv.funUnique (Fin 1) (ZMod 97))
      (Equiv.funUnique (Fin 1) (ZMod 97)) := by
    ext u v
    simp [cubicLine, tableKernel, cubicKernel, taskKernel]
  rw [h, conjTranspose_submatrix, submatrix_mul_equiv, conjTranspose_mul_cubicKernel]
  ext u v
  have huv : u 0 = v 0 ↔ u = v :=
    ⟨fun h => funext fun t => by rw [Subsingleton.elim t 0]; exact h, fun h => by rw [h]⟩
  simp [Matrix.one_apply, huv]

lemma isContraction_cubicLine : IsContraction (((√97 : ℝ) : ℂ)⁻¹ • cubicLine) := by
  have hs : ((√97 : ℝ) : ℂ) * ((√97 : ℝ) : ℂ) = 97 := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by norm_num)]
    norm_num
  have hne : ((√97 : ℝ) : ℂ) ≠ 0 := by
    intro h
    rw [h] at hs
    norm_num at hs
  have h1 : star ((√97 : ℝ) : ℂ)⁻¹ * ((√97 : ℝ) : ℂ)⁻¹ * ((97 : ℕ) : ℂ) = 1 := by
    simp only [star_inv₀, Complex.star_def, Complex.conj_ofReal]
    push_cast
    rw [← hs]
    field_simp
  rw [IsContraction, conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
    conjTranspose_mul_cubicLine, smul_smul, smul_smul, h1, one_smul, sub_self]
  exact PosSemidef.zero

lemma re_stdAddChar_ge {e : ZMod 97} (he : e = 0 ∨ e = 1 ∨ e = -1) :
    (9 / 10 : ℝ) ≤ (ZMod.stdAddChar e : ℂ).re := by
  have key (j : ℤ) (hj : j = 1 ∨ j = -1) :
      (9 / 10 : ℝ) ≤ (ZMod.stdAddChar ((j : ℤ) : ZMod 97) : ℂ).re := by
    rw [ZMod.stdAddChar_coe]
    have harg : (2 * Real.pi * Complex.I * j / (97 : ℕ) : ℂ) =
        ((2 * Real.pi * j / 97 : ℝ) : ℂ) * Complex.I := by
      push_cast
      ring
    rw [harg, Complex.exp_ofReal_mul_I_re]
    have hcos := Real.one_sub_sq_div_two_le_cos (x := 2 * Real.pi * j / 97)
    have hj2 : (j : ℝ) ^ 2 = 1 := by
      rcases hj with rfl | rfl <;> norm_num
    have hpi : Real.pi ^ 2 ≤ 16 := by nlinarith [Real.pi_pos, Real.pi_le_four]
    have hx : (2 * Real.pi * j / 97) ^ 2 = 4 * Real.pi ^ 2 * (j : ℝ) ^ 2 / 9409 := by ring
    rw [hx, hj2] at hcos
    linarith
  rcases he with rfl | rfl | rfl
  · simp
    norm_num
  · simpa using key 1 (Or.inl rfl)
  · simpa using key (-1) (Or.inr rfl)

example : (9 / 10 : ℝ) ≤ (ZMod.stdAddChar (1 : ZMod 97) : ℂ).re :=
  re_stdAddChar_ge (Or.inr (Or.inl rfl))

/-- **A table within one step of `a³ + a b` has a large nuclear norm.** -/
theorem nuclearNorm_ge_of_near {f : ZMod 97 → ZMod 97 → ZMod 97}
    (hf : ∀ a b, f a b - (a ^ 3 + a * b) = 0 ∨ f a b - (a ^ 3 + a * b) = 1 ∨
      f a b - (a ^ 3 + a * b) = -1) :
    (√97)⁻¹ * (9409 * (9 / 10)) ≤ nuclearNorm (tableKernel f) := by
  have h := norm_trace_le_nuclearNorm isContraction_cubicLine (tableKernel f)
  rw [conjTranspose_smul, Matrix.smul_mul, trace_smul, norm_smul, norm_star, norm_inv,
    Complex.norm_real, Real.norm_of_nonneg (Real.sqrt_nonneg _)] at h
  have ht : (cubicLineᴴ * tableKernel f).trace = ∑ v : Fin 1 → ZMod 97, ∑ u : Fin 1 → ZMod 97,
      (ZMod.stdAddChar (f (u 0) (v 0) - ((u 0) ^ 3 + u 0 * v 0)) : ℂ) := by
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, conjTranspose_apply, cubicLine,
      tableKernel, of_apply, star_stdAddChar_mul]
  have hre : (9409 * (9 / 10) : ℝ) ≤ ‖(cubicLineᴴ * tableKernel f).trace‖ := by
    refine le_trans ?_ (Complex.re_le_norm _)
    rw [ht, Complex.re_sum]
    simp only [Complex.re_sum]
    calc (9409 * (9 / 10) : ℝ)
        = ∑ _v : Fin 1 → ZMod 97, ∑ _u : Fin 1 → ZMod 97, (9 / 10 : ℝ) := by
          simp [ZMod.card]
          norm_num
      _ ≤ _ := Finset.sum_le_sum fun v _ => Finset.sum_le_sum fun u _ =>
          re_stdAddChar_ge (hf _ _)
  exact le_trans (mul_le_mul_of_nonneg_left hre (inv_nonneg.2 (Real.sqrt_nonneg _))) h

/-- **M does not return a table within one step of `a³ + a b` from the training half.** -/
theorem not_returned_of_near {f : ZMod 97 → ZMod 97 → ZMod 97}
    (hf : ∀ a b, f a b - (a ^ 3 + a * b) = 0 ∨ f a b - (a ^ 3 + a * b) = 1 ∨
      f a b - (a ^ 3 + a * b) = -1) : NotReturned f := by
  intro hmin
  have h1 := nuclearNorm_ge_of_near hf
  have h2 := nuclearNorm_sq_le_card_of_isMinOn (K := tableKernel f) (Ω := half)
    (fun i j => by simp [tableKernel]) hmin
  rw [card_half] at h2
  simp only [Fintype.card_fun, Fintype.card_fin, ZMod.card, pow_one, min_self] at h2
  have hs : (√97 : ℝ) ^ 2 = 97 := Real.sq_sqrt (by norm_num)
  have hspos : (0 : ℝ) < √97 := Real.sqrt_pos.2 (by norm_num)
  have hsq : ((√97)⁻¹ * (9409 * (9 / 10)) : ℝ) ^ 2 = 9409 ^ 2 * (9 / 10) ^ 2 / 97 := by
    rw [mul_pow, inv_pow, hs]
    ring
  have hle := pow_le_pow_left₀ (by positivity) h1 2
  rw [hsq] at hle
  norm_num at h2 hle
  linarith

/-- The table of `parity-cubic`, `2 ⌊(a³ + a b) / 2⌋ + (a + b) mod 2`. -/
def parityTable (a b : ZMod 97) : ZMod 97 :=
  ((2 * ((a ^ 3 + a * b).val / 2) + (a.val + b.val) % 2 : ℕ) : ZMod 97)

lemma parityTable_near : ∀ a b : ZMod 97, parityTable a b - (a ^ 3 + a * b) = 0 ∨
    parityTable a b - (a ^ 3 + a * b) = 1 ∨ parityTable a b - (a ^ 3 + a * b) = -1 := by
  decide +kernel

example := nuclearNorm_ge_of_near parityTable_near

example : NotReturned parityTable := not_returned_of_near parityTable_near

/-- **M does not return the table of `parity-cubic` from the training half.** -/
theorem not_returned_parityCubic : NotReturned parityTable :=
  not_returned_of_near parityTable_near

end Transformer.Grokking.FlatSpectrum.TablePart.E6
