module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SeparateCells
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRunLimit

/-!
# Samples of the code `±1` in distinct rows and columns: the rows of a sample

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE20,
statement 1; the items "Not covered" of XE17 and XE19.

Let no two samples lie in one row or in one column. For the sample `s` at the cell `(i, j)`
with the label `y = ±1`, write `u = U_i` and `v = y V_j`, as matrices with one row (`rowOf`).

* `Re (u vᴴ) = y Re X_ij` (`pairing_one_rowOf`), so a step without weight decay is
  `u' = u + η w v`, `v' = v + η w u` with the weight `w = w(Re (u vᴴ))`: a step on the sign
  sample of XE19 (`rowOf_decayStep_signCode`).
* So along the run the pair `(u, v)` is the run on the sign sample from `(u₀, v₀)`
  (`rowOf_decayRun_signCode`).
* The loss of the sample is the loss of the sign sample at `u vᴴ`
  (`crossEntropy_signCode_mul`), so the loss along the run is the sum of the losses of these
  runs (`ceLoss_decayRun_signCode`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype k]
  [Fintype S] {cell : S → m × n} {y : S → Bool} {η : ℝ}

/-- The row `i` of a factor, as a matrix with one row. -/
def rowOf {p : Type*} (i : p) (U : Matrix p k ℂ) : Matrix (Fin 1) k ℂ :=
  U.submatrix (fun _ => i) id

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- `Re (u (σ v)ᴴ) = σ Re (U Vᴴ)_ij` for the rows `u`, `v` of `U`, `V`. -/
lemma pairing_one_rowOf (i : m) (j : n) (σ : ℝ) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    pairing (1 : Matrix (Fin 1) (Fin 1) ℂ) (rowOf i U * (σ • rowOf j V)ᴴ) =
      σ * ((U * Vᴴ) i j).re := by
  rw [conjTranspose_smul, star_trivial, Matrix.mul_smul, pairing_smul_right]
  congr 1
  simp [pairing, Matrix.trace_fin_one, Matrix.mul_apply, rowOf]

local notation "R₁" => decayRun (ceGrad signProbe fun _ => true)

/-- **A step acts on the rows of a sample as a step on the sign sample**
(`review/flat_spectrum.md`, item XE20, statement 1): with no two samples in one row or in one
column, the rows `u = U_i` and `v = y V_j` of the sample at the cell `(i, j)` with the label
`y = ±1` follow `u' = u + η w v`, `v' = v + η w u`. -/
lemma rowOf_decayStep_signCode (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (η : ℝ)
    (W : Matrix m k ℂ × Matrix n k ℂ) (s : S) :
    decayStep (ceGrad signProbe fun _ => true) η 0
        (rowOf (cell s).1 W.1, classSign (y s) • rowOf (cell s).2 W.2) =
      (rowOf (cell s).1 (decayStep (ceGrad (codeProbe signCode cell) y) η 0 W).1,
        classSign (y s) •
          rowOf (cell s).2 (decayStep (ceGrad (codeProbe signCode cell) y) η 0 W).2) := by
  have hσ := classSign_mul_self (y s)
  rw [decayStep_signProbe]
  simp only [pairing_one_rowOf]
  refine Prod.ext ?_ ?_
  · ext a b
    simp only [decayStep, rowOf, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      Matrix.submatrix_apply, id, ceGrad_signCode_mul_apply hrow, zero_smul, add_zero]
    module
  · ext a b
    simp only [decayStep, rowOf, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      Matrix.submatrix_apply, id, conjTranspose_ceGrad_signCode_mul_apply hcol, zero_smul,
      add_zero]
    generalize classSign (y s) = σ at hσ ⊢
    generalize signWeight (σ * ((W.1 * W.2ᴴ) (cell s).1 (cell s).2).re) = w
    have key : σ • (W.2 (cell s).2 b - η • -(σ * w) • W.1 (cell s).1 b) =
        σ • W.2 (cell s).2 b + (η * w * (σ * σ)) • W.1 (cell s).1 b := by module
    rw [key, hσ, mul_one]

/-- **The rows of a sample along the run are a run on the sign sample**
(`review/flat_spectrum.md`, item XE20, statement 1). -/
lemma rowOf_decayRun_signCode (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (η : ℝ) (U₀ : Matrix m k ℂ)
    (V₀ : Matrix n k ℂ) (s : S) (t : ℕ) :
    R₁ η 0 (rowOf (cell s).1 U₀) (classSign (y s) • rowOf (cell s).2 V₀) t =
      (rowOf (cell s).1 (decayRun (ceGrad (codeProbe signCode cell) y) η 0 U₀ V₀ t).1,
        classSign (y s) •
          rowOf (cell s).2 (decayRun (ceGrad (codeProbe signCode cell) y) η 0 U₀ V₀ t).2) := by
  induction t with
  | zero => rfl
  | succ t ih => rw [decayRun_succ, ih, rowOf_decayStep_signCode hrow hcol, ← decayRun_succ]

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- A row has at most the norm of the matrix. -/
lemma frobeniusSq_rowOf_le {p : Type*} [Fintype p] (i : p) (U : Matrix p k ℂ) :
    frobeniusSq (rowOf i U) ≤ frobeniusSq U := by
  simp only [frobeniusSq, rowOf, Matrix.submatrix_apply, id, Fin.sum_univ_one]
  exact Finset.single_le_sum (f := fun i => ∑ j, ‖U i j‖ ^ 2)
    (fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg _) (Finset.mem_univ i)

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- The rows `u = U_i` and `v = σ V_j` sum to zero exactly when `U_i + σ V_j` vanishes. -/
lemma rowOf_add_ne_zero_iff (i : m) (j : n) (σ : ℝ) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    rowOf i U + σ • rowOf j V ≠ 0 ↔ ∃ b, U i b + σ • V j b ≠ 0 := by
  constructor
  · intro h
    by_contra hcon
    have hall : ∀ b, U i b + σ • V j b = 0 := fun b => by_contra fun hb => hcon ⟨b, hb⟩
    exact h (by ext a b; simpa [rowOf] using hall b)
  · rintro ⟨b, hb⟩ h
    exact hb (by simpa [rowOf] using congrFun (congrFun h 0) b)

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- `⟨−A, X⟩ = −⟨A, X⟩`. -/
lemma pairing_neg_left {p q : Type*} [Fintype p] [Fintype q] (A X : Matrix p q ℂ) :
    pairing (-A) X = -pairing A X := by
  rw [show -A = (-1 : ℝ) • A by simp, pairing_smul_left]
  ring

omit [Fintype S] in
/-- **The loss of a sample is the loss of the sign sample at its rows**
(`review/flat_spectrum.md`, item XE20, statement 1). -/
lemma crossEntropy_signCode_mul (s : S) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    Landscape.Neural.crossEntropy
        (fun c => pairing (codeProbe signCode cell s c) (U * Vᴴ)) (y s) =
      ceLoss signProbe (fun _ => true)
        (rowOf (cell s).1 U * (classSign (y s) • rowOf (cell s).2 V)ᴴ) := by
  have hL : (fun c : Bool => pairing (codeProbe signCode cell s c) (U * Vᴴ)) =
      fun c => (signCode c).re * ((U * Vᴴ) (cell s).1 (cell s).2).re :=
    funext fun c => pairing_codeProbe_signCode s c _
  have hR : (fun c : Bool => pairing (signProbe 0 c)
      (rowOf (cell s).1 U * (classSign (y s) • rowOf (cell s).2 V)ᴴ)) =
      fun c => (signCode c).re * (classSign (y s) * ((U * Vᴴ) (cell s).1 (cell s).2).re) :=
    funext fun c => by
      cases c
      · simp only [signProbe, signCode, Bool.false_eq_true, ↓reduceIte, Complex.neg_re,
          Complex.one_re]
        rw [pairing_neg_left, pairing_one_rowOf]
        ring
      · simp only [signProbe, signCode, ↓reduceIte, Complex.one_re]
        rw [pairing_one_rowOf]
        ring
  rw [ceLoss, Fin.sum_univ_one, hL, hR]
  generalize ((U * Vᴴ) (cell s).1 (cell s).2).re = r
  cases y s
  · simp only [Landscape.Neural.crossEntropy, Fintype.sum_bool, signCode, classSign,
      Bool.false_eq_true, ↓reduceIte, Complex.neg_re, Complex.one_re]
    ring_nf
  · simp only [Landscape.Neural.crossEntropy, Fintype.sum_bool, signCode, classSign,
      ↓reduceIte, Complex.one_re]
    ring_nf

local notation "RC" => decayRun (ceGrad (codeProbe signCode cell) y)

/-- **The loss along the run is the sum of the losses of the sign runs of the samples**
(`review/flat_spectrum.md`, item XE20, statement 1). -/
lemma ceLoss_decayRun_signCode (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (η : ℝ) (U₀ : Matrix m k ℂ)
    (V₀ : Matrix n k ℂ) (t : ℕ) :
    ceLoss (codeProbe signCode cell) y ((RC η 0 U₀ V₀ t).1 * (RC η 0 U₀ V₀ t).2ᴴ) =
      ∑ s, ceLoss signProbe (fun _ => true)
        (signProduct η (rowOf (cell s).1 U₀) (classSign (y s) • rowOf (cell s).2 V₀) t) := by
  rw [ceLoss]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [crossEntropy_signCode_mul, signProduct, rowOf_decayRun_signCode hrow hcol]

/-! The witness: the two samples at the diagonal cells of a `2 × 2` product, with the labels
`(+, −)`, the start `U₀ = 1`, `V₀ = 0` and the step `1/10`. -/

example := rowOf_decayStep_signCode (y := ![true, false]) injective_diagCell_fst
  injective_diagCell_snd (1 / 10) ((1 : Matrix (Fin 2) (Fin 2) ℂ), (0 : Matrix (Fin 2) (Fin 2) ℂ)) 1

example := rowOf_decayRun_signCode (y := ![true, false]) injective_diagCell_fst
  injective_diagCell_snd (1 / 10) (1 : Matrix (Fin 2) (Fin 2) ℂ) (0 : Matrix (Fin 2) (Fin 2) ℂ) 1 3

example := ceLoss_decayRun_signCode (y := ![true, false]) injective_diagCell_fst
  injective_diagCell_snd (1 / 10) (1 : Matrix (Fin 2) (Fin 2) ℂ) (0 : Matrix (Fin 2) (Fin 2) ℂ) 3

end Transformer.Grokking.FlatSpectrum.Training.Scope
