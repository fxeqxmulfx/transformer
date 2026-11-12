module

public import Transformer.Grokking.FlatSpectrum.Training.SignPath
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSystem

/-!
# Samples of the code `±1` in distinct rows and columns: the gradient

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE20; the items
"Not covered" of XE17 and XE19.

Samples `s` of two classes at the cells `cell s = (i, j)` of an `m × n` product, read with the
code `±1` (`codeProbe signCode cell`) and labelled `y s`, whose sign is `classSign (y s)`.

* The term of the sample `s` in the gradient is `−y w(y Re X_ij) E_ij`
  (`ceGrad_term_signCode`), with the weight `signWeight` of XE19; the gradient is the sum of
  these terms (`ceGrad_signCode`).
* With no two samples in one row, the row `i` of `G V` is `−y w` times the row `j` of `V`
  (`ceGrad_signCode_mul_apply`); with no two samples in one column, the row `j` of `Gᴴ U` is
  `−y w` times the row `i` of `U` (`conjTranspose_ceGrad_signCode_mul_apply`).

`SeparateCellsRows` follows the rows of a sample along the run without weight decay, and
`SeparateCellsRun` has the norm, the loss and the margin. The witness is `diagCell`, the two
diagonal cells of a `2 × 2` product.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype k]
  [Fintype S] {cell : S → m × n} {y : S → Bool} {η : ℝ}

/-- The sign of a class of the code `±1`. -/
def classSign (b : Bool) : ℝ := if b then 1 else -1

lemma classSign_mul_self (b : Bool) : classSign b * classSign b = 1 := by
  cases b <;> simp [classSign]

omit [Fintype S] in
/-- **The term of one sample in the gradient of the code `±1`** (`review/flat_spectrum.md`,
item XE20): `−y w(y Re X_ij) E_ij` at the cell `(i, j)` of the sample, with its label `y = ±1`. -/
lemma ceGrad_term_signCode (s : S) (X : Matrix m n ℂ) :
    (∑ c, Geometry.probability (fun c : Bool => pairing (codeProbe signCode cell s c) X) c •
        codeProbe signCode cell s c) - codeProbe signCode cell s (y s) =
      (-(classSign (y s) * signWeight (classSign (y s) * (X (cell s).1 (cell s).2).re))) •
        Matrix.single (cell s).1 (cell s).2 (1 : ℂ) := by
  have hz : (fun c : Bool => pairing (codeProbe signCode cell s c) X) =
      fun c => (signCode c).re * (X (cell s).1 (cell s).2).re :=
    funext fun c => pairing_codeProbe_signCode s c X
  rw [hz]
  generalize (X (cell s).1 (cell s).2).re = r
  have hp := Geometry.sum_probability (fun c : Bool => (signCode c).re * r) true
  have hf : Geometry.probability (fun c : Bool => (signCode c).re * r) false =
      signWeight r / 2 := by
    simp only [Geometry.probability, Fintype.sum_bool, signCode, ↓reduceIte,
      Bool.false_eq_true, signWeight, Complex.one_re, Complex.neg_re, one_mul, neg_mul]
    rw [Real.exp_neg, two_mul, Real.exp_add]
    have he := Real.exp_pos r
    field_simp
    ring
  have ht : Geometry.probability (fun c : Bool => (signCode c).re * r) true =
      signWeight (-r) / 2 := by
    simp only [Geometry.probability, Fintype.sum_bool, signCode, ↓reduceIte,
      Bool.false_eq_true, signWeight, Complex.one_re, Complex.neg_re, one_mul, neg_mul]
    rw [mul_neg, Real.exp_neg, Real.exp_neg, two_mul, Real.exp_add]
    have he := Real.exp_pos r
    field_simp
  rw [Fintype.sum_bool] at hp
  have hE : codeProbe signCode cell s false = -Matrix.single (cell s).1 (cell s).2 (1 : ℂ) := by
    ext a b
    by_cases h : (cell s).1 = a ∧ (cell s).2 = b <;>
      simp [codeProbe, signCode, h]
  have hE1 : codeProbe signCode cell s true = Matrix.single (cell s).1 (cell s).2 (1 : ℂ) := by
    simp [codeProbe, signCode]
  rw [Fintype.sum_bool, hE, hE1]
  cases y s
  · rw [hE, eq_sub_of_add_eq' hp, ht]
    simp only [classSign, Bool.false_eq_true, ↓reduceIte, neg_mul, one_mul]
    module
  · rw [hE1, eq_sub_of_add_eq hp, hf]
    simp only [classSign, ↓reduceIte, one_mul]
    module

/-- **The gradient of samples of the code `±1`** (`review/flat_spectrum.md`, item XE20). -/
lemma ceGrad_signCode (X : Matrix m n ℂ) :
    ceGrad (codeProbe signCode cell) y X =
      ∑ s, (-(classSign (y s) * signWeight (classSign (y s) * (X (cell s).1 (cell s).2).re))) •
        Matrix.single (cell s).1 (cell s).2 (1 : ℂ) :=
  Finset.sum_congr rfl fun s _ => ceGrad_term_signCode s X

omit [Fintype k] in
/-- With no two samples in one row, the gradient times `V` has, in the row of the sample `s`,
the row of `V` at the column of `s`, times `−y w`. -/
lemma ceGrad_signCode_mul_apply (hrow : Function.Injective fun s => (cell s).1)
    (X : Matrix m n ℂ) (V : Matrix n k ℂ) (s : S) (b : k) :
    (ceGrad (codeProbe signCode cell) y X * V) (cell s).1 b =
      (-(classSign (y s) * signWeight (classSign (y s) * (X (cell s).1 (cell s).2).re))) •
        V (cell s).2 b := by
  rw [ceGrad_signCode, Matrix.sum_mul, Matrix.sum_apply, Finset.sum_eq_single s]
  · rw [Matrix.smul_mul, Matrix.smul_apply, Matrix.single_mul_apply_same, one_mul]
  · intro s' _ hs'
    rw [Matrix.smul_mul, Matrix.smul_apply,
      Matrix.single_mul_apply_of_ne _ _ _ _ _ (fun h => hs' (hrow h.symm)), smul_zero]
  · exact fun h => absurd (Finset.mem_univ s) h

omit [Fintype k] in
/-- With no two samples in one column, the adjoint of the gradient times `U` has, in the row of
the column of the sample `s`, the row of `U` at the row of `s`, times `−y w`. -/
lemma conjTranspose_ceGrad_signCode_mul_apply (hcol : Function.Injective fun s => (cell s).2)
    (X : Matrix m n ℂ) (U : Matrix m k ℂ) (s : S) (b : k) :
    ((ceGrad (codeProbe signCode cell) y X)ᴴ * U) (cell s).2 b =
      (-(classSign (y s) * signWeight (classSign (y s) * (X (cell s).1 (cell s).2).re))) •
        U (cell s).1 b := by
  rw [ceGrad_signCode, conjTranspose_sum, Matrix.sum_mul, Matrix.sum_apply,
    Finset.sum_eq_single s]
  · rw [conjTranspose_smul, conjTranspose_single, star_one, star_trivial, Matrix.smul_mul,
      Matrix.smul_apply, Matrix.single_mul_apply_same, one_mul]
  · intro s' _ hs'
    rw [conjTranspose_smul, conjTranspose_single, Matrix.smul_mul, Matrix.smul_apply,
      Matrix.single_mul_apply_of_ne _ _ _ _ _ (fun h => hs' (hcol h.symm)), smul_zero]
  · exact fun h => absurd (Finset.mem_univ s) h

/-! The witness: the two samples at the diagonal cells of a `2 × 2` product, with the labels
`(+, −)`. -/

/-- The diagonal cells of a `2 × 2` product. -/
def diagCell (s : Fin 2) : Fin 2 × Fin 2 := (s, s)

lemma injective_diagCell_fst : Function.Injective fun s => (diagCell s).1 := fun _ _ h => h

lemma injective_diagCell_snd : Function.Injective fun s => (diagCell s).2 := fun _ _ h => h

example := ceGrad_signCode_mul_apply (y := ![true, false]) injective_diagCell_fst
  (1 : Matrix (Fin 2) (Fin 2) ℂ) (1 : Matrix (Fin 2) (Fin 2) ℂ) 0 0

example := conjTranspose_ceGrad_signCode_mul_apply (y := ![true, false]) injective_diagCell_snd
  (1 : Matrix (Fin 2) (Fin 2) ℂ) (1 : Matrix (Fin 2) (Fin 2) ℂ) 0 0

end Transformer.Grokking.FlatSpectrum.Training.Scope
