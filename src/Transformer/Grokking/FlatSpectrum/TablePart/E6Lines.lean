module

public import Transformer.Grokking.FlatSpectrum.CertificateLines
public import Transformer.Grokking.FlatSpectrum.TablePart.DivFlat
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Half

/-!
# PT6: three tables with a line orthogonal to all the others

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. These kernels are not
flat: `a² b` has equal rows `a` and `−a`, `a³ + a b² + b` has columns `b` and `−b` equal up to a
phase, and in the mixed task the columns of even `b` are parallel. Each of them still has one line
orthogonal to all the others: the row `a = 0`, the column `b = 0` and the column `b = 3`. So that
line has to be seen in full (`not_isMinOn_of_orthogonal_row`, `not_isMinOn_of_orthogonal_column`),
and the training half misses a cell of it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- The kernel `[ψ(f(a, b))]` of a table on `ℤ/97`. -/
noncomputable def tableKernel (f : ZMod 97 → ZMod 97 → ZMod 97) :
    Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun u v => (ZMod.stdAddChar (f (u 0) (v 0)) : ℂ)

/-- M does not return the kernel of `f` from the training half of E6. -/
abbrev NotReturned (f : ZMod 97 → ZMod 97 → ZMod 97) : Prop :=
  ¬ IsMinOn nuclearNorm
    {X | ∀ ij ∈ (half : Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))),
      X ij.1 ij.2 = tableKernel f ij.1 ij.2} (tableKernel f)

lemma tableKernel_ne_zero (f : ZMod 97 → ZMod 97 → ZMod 97) (u v : Fin 1 → ZMod 97) :
    tableKernel f u v ≠ 0 :=
  norm_ne_zero_iff.mp (by simp [tableKernel])

/-- **A cell lies in the half exactly when its bit in `halfMask` is set.** -/
theorem mem_half_iff {a b : ℕ} (ha : a < 97) (hb : b < 97) :
    ((fun _ => (a : ZMod 97), fun _ => (b : ZMod 97)) :
      (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) ∈ half ↔ halfMask.testBit (97 * a + b) = true := by
  have hT : marksFast.2.1 = halfMask := certificate.2.2.2.2.2.2.2.2.2.2
  have spec := mark_spec shuffleFast.2.2 4704 count_marks
  constructor
  · intro h
    obtain ⟨i, hi, hc⟩ := mem_image.1 h
    have hlt := spec.2.1 i (mem_range.1 hi)
    have hcell : cell (get shuffleFast.2.2 i) = cell (97 * a + b) := by rw [hc, cell_add hb]
    have hget := cell_inj hlt (by omega) hcell
    rw [← hT, marksFast_eq]
    exact (spec.2.2 _).2 ⟨i, mem_range.1 hi, hget⟩
  · intro h
    exact mem_half hb (by rw [hT]; exact h)

example : ((fun _ => ((0 : ℕ) : ZMod 97), fun _ => ((0 : ℕ) : ZMod 97)) :
    (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) ∉ half := fun h => by
  have := (mem_half_iff (by norm_num) (by norm_num)).1 h
  revert this
  decide +kernel

/-- The sum of `ψ(y x + c)` over a line vanishes for `x ≠ 0`. -/
lemma sum_line_add {x : ZMod 97} (c : ZMod 97) (hx : x ≠ 0) :
    ∑ u : Fin 1 → ZMod 97, (ZMod.stdAddChar (u 0 * x + c) : ℂ) = 0 := by
  simp only [AddChar.map_add_eq_mul]
  rw [← Finset.sum_mul, sum_line_apply (fun y => (ZMod.stdAddChar (y * x) : ℂ)),
    sum_stdAddChar_mul, ite_eq_right hx, zero_mul]

example : ∑ u : Fin 1 → ZMod 97, (ZMod.stdAddChar (u 0 * 1 + 0) : ℂ) = 0 :=
  sum_line_add 0 one_ne_zero

/-- A line different from `fun _ => c` differs from it at `0`. -/
lemma apply_zero_ne {u : Fin 1 → ZMod 97} {c : ZMod 97} (h : u ≠ fun _ => c) : u 0 ≠ c :=
  fun h0 => h (funext fun t => by rw [Subsingleton.elim t 0, h0])

example : (fun _ => 1 : Fin 1 → ZMod 97) 0 ≠ 0 :=
  apply_zero_ne (u := fun _ => 1) (c := 0) fun h => one_ne_zero (congrFun h 0)

/-- **M does not return `a² b` from the training half** (`review/flat_spectrum.md`, PT6): the row
`a = 0` is orthogonal to the others, and the half misses its cell `b = 0`. -/
theorem not_returned_squareTimes : NotReturned fun a b => a ^ 2 * b := by
  have hΩ : ((fun _ => ((0 : ℕ) : ZMod 97), fun _ => ((0 : ℕ) : ZMod 97)) :
      (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) ∉ half := fun h => by
    have := (mem_half_iff (by norm_num) (by norm_num)).1 h
    revert this
    decide +kernel
  refine not_isMinOn_of_orthogonal_row (i₀ := fun _ => ((0 : ℕ) : ZMod 97))
    (j₀ := fun _ => ((0 : ℕ) : ZMod 97)) (fun i hi => ?_) (tableKernel_ne_zero _ _ _)
    (fun h => hΩ (mem_coe.1 h))
  have hi0 : i 0 ≠ 0 := by simpa using apply_zero_ne hi
  have hterm (v : Fin 1 → ZMod 97) : tableKernel (fun a b => a ^ 2 * b) i v *
      star (tableKernel (fun a b => a ^ 2 * b) (fun _ => ((0 : ℕ) : ZMod 97)) v) =
        (ZMod.stdAddChar (v 0 * i 0 ^ 2 + 0) : ℂ) := by
    simp [tableKernel, mul_comm]
  simp only [hterm]
  exact sum_line_add 0 (pow_ne_zero 2 hi0)

/-- **M does not return `a³ + a b² + b` from the training half** (`review/flat_spectrum.md`, PT6):
the column `b = 0` is orthogonal to the others, and the half misses its cell `a = 0`. -/
theorem not_returned_cubicMixed : NotReturned fun a b => a ^ 3 + a * b ^ 2 + b := by
  have hΩ : ((fun _ => ((0 : ℕ) : ZMod 97), fun _ => ((0 : ℕ) : ZMod 97)) :
      (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) ∉ half := fun h => by
    have := (mem_half_iff (by norm_num) (by norm_num)).1 h
    revert this
    decide +kernel
  refine not_isMinOn_of_orthogonal_column (i₀ := fun _ => ((0 : ℕ) : ZMod 97))
    (j₀ := fun _ => ((0 : ℕ) : ZMod 97)) (fun j hj => ?_) (tableKernel_ne_zero _ _ _)
    (fun h => hΩ (mem_coe.1 h))
  have hj0 : j 0 ≠ 0 := by simpa using apply_zero_ne hj
  have hterm (u : Fin 1 → ZMod 97) : star (tableKernel (fun a b => a ^ 3 + a * b ^ 2 + b) u j) *
      tableKernel (fun a b => a ^ 3 + a * b ^ 2 + b) u (fun _ => ((0 : ℕ) : ZMod 97)) =
        (ZMod.stdAddChar (u 0 * (-j 0 ^ 2) + -j 0) : ℂ) := by
    simp only [tableKernel, of_apply, star_stdAddChar_mul]
    congr 1
    push_cast
    ring
  simp only [hterm]
  exact sum_line_add _ (neg_ne_zero.2 (pow_ne_zero 2 hj0))

/-- The table of the mixed task: `a / b` for odd `b` and `a − b` for even `b`. -/
def mixedTable (a b : ZMod 97) : ZMod 97 := if b.val % 2 = 1 then a / b else a - b

/-- **M does not return the mixed task from the training half** (`review/flat_spectrum.md`, PT6):
the column `b = 3` is orthogonal to the others, and the half misses its cell `a = 4`. -/
theorem not_returned_mixed : NotReturned mixedTable := by
  have hΩ : ((fun _ => ((4 : ℕ) : ZMod 97), fun _ => ((3 : ℕ) : ZMod 97)) :
      (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) ∉ half := fun h => by
    have := (mem_half_iff (by norm_num) (by norm_num)).1 h
    revert this
    decide +kernel
  refine not_isMinOn_of_orthogonal_column (i₀ := fun _ => ((4 : ℕ) : ZMod 97))
    (j₀ := fun _ => ((3 : ℕ) : ZMod 97)) (fun j hj => ?_) (tableKernel_ne_zero _ _ _)
    (fun h => hΩ (mem_coe.1 h))
  have hj3 : j 0 ≠ 3 := by simpa using apply_zero_ne hj
  have h3 : ((3 : ℕ) : ZMod 97).val % 2 = 1 := by decide +kernel
  have h31 : (3 : ZMod 97)⁻¹ - 1 ≠ 0 := by decide +kernel
  by_cases hodd : (j 0).val % 2 = 1
  · have hx : (3 : ZMod 97)⁻¹ - (j 0)⁻¹ ≠ 0 := fun h => hj3 (inv_injective (sub_eq_zero.1 h)).symm
    have hterm (u : Fin 1 → ZMod 97) : star (tableKernel mixedTable u j) *
        tableKernel mixedTable u (fun _ => ((3 : ℕ) : ZMod 97)) =
          (ZMod.stdAddChar (u 0 * ((3 : ZMod 97)⁻¹ - (j 0)⁻¹) + 0) : ℂ) := by
      simp only [tableKernel, of_apply, star_stdAddChar_mul, mixedTable, ite_eq_left hodd, ite_eq_left h3]
      congr 1
      push_cast
      ring
    simp only [hterm]
    exact sum_line_add 0 hx
  · have hterm (u : Fin 1 → ZMod 97) : star (tableKernel mixedTable u j) *
        tableKernel mixedTable u (fun _ => ((3 : ℕ) : ZMod 97)) =
          (ZMod.stdAddChar (u 0 * ((3 : ZMod 97)⁻¹ - 1) + j 0) : ℂ) := by
      simp only [tableKernel, of_apply, star_stdAddChar_mul, mixedTable, ite_eq_right hodd, ite_eq_left h3]
      congr 1
      push_cast
      ring
    simp only [hterm]
    exact sum_line_add _ h31

end Transformer.Grokking.FlatSpectrum.TablePart.E6
