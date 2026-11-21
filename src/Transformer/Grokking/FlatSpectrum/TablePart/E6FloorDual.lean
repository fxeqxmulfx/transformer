module

public import Transformer.Grokking.FlatSpectrum.TablePart.DivClasses
public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorGram
public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorPhase
public import Transformer.Grokking.FlatSpectrum.TablePart.RowSum

/-!
# PT6: a lower bound on the nuclear norm of `⌊a b / 97⌋`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The rounded polar
factor `Ẑ` has row sums of `Ẑᴴ Ẑ` at most `1.0044²` (`TablePart.E6FloorGram`), so `Ẑ / 1.0044` is a
contraction, and `‖K‖_S1 ≥ Re tr(Ẑᴴ K) / 1.0044`. Each `K` entry is a power of `ζ`, replaced by its
table entry at a certified cost, and the kernel adds up the 9409 cells (`dualSum_ge`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- The table of `floor-product`, `⌊a b / 97⌋`. -/
def floorTable (a b : ZMod 97) : ZMod 97 := ((a.val * b.val / 97 : ℕ) : ZMod 97)

/-- The rounded polar factor of the kernel. -/
noncomputable def zMat : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun u v =>
    ⟨(zre (u 0).val (v 0).val : ℝ) / 2 ^ 13, (zim (u 0).val (v 0).val : ℝ) / 2 ^ 13⟩

/-- A sum over the lines is a sum over the residues. -/
lemma sum_line_val {M : Type*} [AddCommMonoid M] (G : ℕ → M) :
    ∑ u : Fin 1 → ZMod 97, G (u 0).val = ∑ n ∈ range 97, G n := by
  rw [← sum_zmod_eq_sum_range 97 G]
  exact (Equiv.funUnique (Fin 1) (ZMod 97)).sum_comp fun a => G a.val

/-- A double sum over the lines is a double sum over the residues. -/
lemma sum_lines_val {M : Type*} [AddCommMonoid M] (F : ℕ → ℕ → M) :
    ∑ u : Fin 1 → ZMod 97, ∑ v : Fin 1 → ZMod 97, F (u 0).val (v 0).val =
      ∑ i ∈ range 97, ∑ j ∈ range 97, F i j := by
  rw [← sum_line_val fun i => ∑ j ∈ range 97, F i j]
  exact Finset.sum_congr rfl fun u _ => sum_line_val fun j => F (u 0).val j

lemma star_mul_pair (a b c d : ℝ) :
    star (⟨a / 2 ^ 13, b / 2 ^ 13⟩ : ℂ) * ⟨c / 2 ^ 13, d / 2 ^ 13⟩ =
      ⟨(a * c + b * d) / 2 ^ 26, (a * d - b * c) / 2 ^ 26⟩ := by
  apply Complex.ext <;> simp only [Complex.mul_re, Complex.mul_im, Complex.star_def,
    Complex.conj_re, Complex.conj_im] <;> ring

/-- The entries of `Ẑᴴ Ẑ`. -/
lemma gram_apply (p q : Fin 1 → ZMod 97) :
    (zMatᴴ * zMat) p q = ⟨(gramRe (p 0).val (q 0).val : ℝ) / 2 ^ 26,
      (gramIm (p 0).val (q 0).val : ℝ) / 2 ^ 26⟩ := by
  rw [Matrix.mul_apply]
  simp only [conjTranspose_apply, zMat, of_apply, star_mul_pair]
  rw [sum_line_val fun l => (⟨((zre l (p 0).val : ℝ) * zre l (q 0).val +
      zim l (p 0).val * zim l (q 0).val) / 2 ^ 26, ((zre l (p 0).val : ℝ) * zim l (q 0).val -
      zim l (p 0).val * zre l (q 0).val) / 2 ^ 26⟩ : ℂ)]
  apply Complex.ext
  · rw [Complex.re_sum, gramRe, ← Finset.sum_div]
    push_cast
    rfl
  · rw [Complex.im_sum, gramIm, ← Finset.sum_div]
    push_cast
    rfl

/-- **The rounded polar factor, over `1.0044`, is a contraction.** -/
theorem isContraction_zMat : IsContraction (((1.0044 : ℝ) : ℂ)⁻¹ • zMat) := by
  refine isContraction_of_row_sum (by norm_num) fun p => ?_
  have hcell (q : Fin 1 → ZMod 97) : ‖(zMatᴴ * zMat) p q‖ ≤
      (|(gramRe (p 0).val (q 0).val : ℝ)| + |(gramIm (p 0).val (q 0).val : ℝ)|) / 2 ^ 26 := by
    refine (Complex.norm_le_abs_re_add_abs_im _).trans (le_of_eq ?_)
    rw [gram_apply, add_div, abs_div, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ 26)]
  refine (Finset.sum_le_sum fun q _ => hcell q).trans ?_
  rw [sum_line_val fun n => (|(gramRe (p 0).val n : ℝ)| + |(gramIm (p 0).val n : ℝ)|) / 2 ^ 26,
    ← Finset.sum_div]
  have h := sum_abs_gram_le (ZMod.val_lt (p 0))
  have h1 : ((∑ q ∈ range 97, (|gramRe (p 0).val q| + |gramIm (p 0).val q|) : ℤ) : ℝ) ≤
      ((67700000 : ℤ) : ℝ) := Int.cast_le.2 h
  push_cast at h1
  rw [div_le_iff₀ (by positivity)]
  refine h1.trans ?_
  norm_num

/-- The value of the cell `(i, j)` in the dual sum, in units of `2⁻³³`. -/
def dualTerm (i j : ℕ) : ℤ :=
  zre i j * phC (i * j / 97) + zim i j * phS (i * j / 97) -
    (Int.natAbs (zre i j) + Int.natAbs (zim i j)) * (4 * ((i * j / 97 : ℕ) + 1))

/-- **The kernel adds up the dual sum** over all 9409 cells. -/
theorem dualSum_ge : (1326297271500 : ℤ) ≤
    sumFrom (fun i => sumFrom (fun j => dualTerm i j) 0 97) 0 97 := by
  decide +kernel

/-- One cell of the pairing, against a phase next to the table entry. -/
lemma cell_ge (x y C S : ℝ) {w : ℂ} {e : ℝ} (hw : ‖w - ⟨C / 2 ^ 20, S / 2 ^ 20⟩‖ ≤ e / 2 ^ 20) :
    (x * C + y * S - (|x| + |y|) * e) / 2 ^ 33 ≤
      (star (⟨x / 2 ^ 13, y / 2 ^ 13⟩ : ℂ) * w).re := by
  set z : ℂ := ⟨x / 2 ^ 13, y / 2 ^ 13⟩ with hz
  set φ : ℂ := ⟨C / 2 ^ 20, S / 2 ^ 20⟩ with hφ
  have hn : ‖z‖ ≤ (|x| + |y|) / 2 ^ 13 := by
    refine (Complex.norm_le_abs_re_add_abs_im z).trans (le_of_eq ?_)
    rw [hz, abs_div, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ 13), add_div]
  have h1 : (star z * φ).re = (x * C + y * S) / 2 ^ 33 := by
    simp only [hz, hφ, Complex.mul_re, Complex.star_def, Complex.conj_re, Complex.conj_im]
    ring
  have h2 : -(‖z‖ * ‖w - φ‖) ≤ (star z * (w - φ)).re := by
    have h3 := Complex.abs_re_le_norm (star z * (w - φ))
    rw [norm_mul, norm_star] at h3
    exact (abs_le.1 h3).1
  have h4 : ‖z‖ * ‖w - φ‖ ≤ (|x| + |y|) / 2 ^ 13 * (e / 2 ^ 20) :=
    mul_le_mul hn hw (norm_nonneg _) (by positivity)
  have h5 : star z * w = star z * φ + star z * (w - φ) := by ring
  have h6 : (|x| + |y|) / 2 ^ 13 * (e / 2 ^ 20) = (|x| + |y|) * e / 2 ^ 33 := by ring
  have h7 : (x * C + y * S - (|x| + |y|) * e) / 2 ^ 33 =
      (x * C + y * S) / 2 ^ 33 - (|x| + |y|) * e / 2 ^ 33 := by ring
  rw [h5, Complex.add_re, h1, h7]
  linarith

example : ((0 : ℝ) * 0 + 0 * 0 - (|(0 : ℝ)| + |(0 : ℝ)|) * 0) / 2 ^ 33 ≤
    (star (⟨0 / 2 ^ 13, 0 / 2 ^ 13⟩ : ℂ) * ⟨0 / 2 ^ 20, 0 / 2 ^ 20⟩).re :=
  cell_ge 0 0 0 0 (by simp)

/-- A cell of the kernel is a power of `ζ`, with an exponent below 97. -/
lemma floor_exponent_lt (u v : Fin 1 → ZMod 97) : (u 0).val * (v 0).val / 97 < 97 :=
  Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' (ZMod.val_lt _) (ZMod.val_lt _))

lemma kernel_floor (u v : Fin 1 → ZMod 97) : tableKernel floorTable u v =
    (ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ ((u 0).val * (v 0).val / 97) := by
  simp only [tableKernel, of_apply, floorTable]
  exact stdAddChar_natCast _

/-- One cell of the trace, bounded below by the integer the kernel adds up. -/
lemma dualTerm_le (u v : Fin 1 → ZMod 97) :
    (dualTerm (u 0).val (v 0).val : ℝ) / 2 ^ 33 ≤
      (star (zMat u v) * tableKernel floorTable u v).re := by
  have hphase := norm_pow_sub_phaseHat_le (floor_exponent_lt u v)
  have h := cell_ge (zre (u 0).val (v 0).val : ℝ) (zim (u 0).val (v 0).val : ℝ)
    (phC ((u 0).val * (v 0).val / 97) : ℝ) (phS ((u 0).val * (v 0).val / 97) : ℝ)
    (w := tableKernel floorTable u v) (e := 4 * (((u 0).val * (v 0).val / 97 : ℕ) + 1)) (by
      rw [kernel_floor]
      exact hphase)
  refine le_trans (le_of_eq ?_) h
  simp only [dualTerm]
  push_cast [Int.natCast_natAbs]
  rfl

/-- **The nuclear norm of the kernel of `⌊a b / 97⌋` is at least 153.7.** -/
theorem nuclearNorm_floor_ge : (153.7 : ℝ) ≤ nuclearNorm (tableKernel floorTable) := by
  have h := norm_trace_le_nuclearNorm isContraction_zMat (tableKernel floorTable)
  rw [conjTranspose_smul, Matrix.smul_mul, trace_smul, norm_smul, norm_star, norm_inv,
    Complex.norm_real, Real.norm_of_nonneg (by norm_num : (0 : ℝ) ≤ 1.0044)] at h
  have hsum : (1326297271500 : ℝ) ≤ ∑ i ∈ range 97, ∑ j ∈ range 97, (dualTerm i j : ℝ) := by
    have h0 := dualSum_ge
    simp only [sumFrom_eq, zero_add] at h0
    exact_mod_cast h0
  have htr : (zMatᴴ * tableKernel floorTable).trace.re =
      ∑ u, ∑ v, (star (zMat u v) * tableKernel floorTable u v).re := by
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, conjTranspose_apply, Complex.re_sum]
    exact Finset.sum_comm
  have hre : (154.4 : ℝ) ≤ ‖(zMatᴴ * tableKernel floorTable).trace‖ := by
    refine le_trans ?_ (Complex.re_le_norm _)
    rw [htr]
    have hcells : ∑ u : Fin 1 → ZMod 97, ∑ v : Fin 1 → ZMod 97,
        (dualTerm (u 0).val (v 0).val : ℝ) / 2 ^ 33 ≤
          ∑ u, ∑ v, (star (zMat u v) * tableKernel floorTable u v).re :=
      Finset.sum_le_sum fun u _ => Finset.sum_le_sum fun v _ => dualTerm_le u v
    refine le_trans ?_ hcells
    simp only [← Finset.sum_div]
    rw [sum_lines_val fun i j => (dualTerm i j : ℝ), le_div_iff₀ (by positivity)]
    refine le_trans ?_ hsum
    norm_num
  have h2 : (1.0044 : ℝ)⁻¹ * 154.4 ≤ (1.0044 : ℝ)⁻¹ *
      ‖(zMatᴴ * tableKernel floorTable).trace‖ :=
    mul_le_mul_of_nonneg_left hre (by norm_num)
  have h3 : (153.7 : ℝ) ≤ (1.0044 : ℝ)⁻¹ * 154.4 := by norm_num
  linarith

end Transformer.Grokking.FlatSpectrum.TablePart.E6
