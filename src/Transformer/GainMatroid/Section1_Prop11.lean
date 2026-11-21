module

public import Transformer.GainMatroid.Section1_Elementary

/-!
# Proposition 1.1

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.1. `elemOps_iff_projEquiv` proves that sequences
of the source's elementary operations are exactly `ProjEquiv`, also between matrices with different
numbers of rows. `elemOps_iff_exists_mul_mul` is Proposition 1.1 (Brylawski and Lucas; Oxley,
Proposition 6.3.12). The source assumes at least one row, which the proof does not use.
-/

@[expose] public section

open Matrix Relation

namespace Transformer.GainMatroid

variable {E F : Type*} [Field F]

/-- Multiplying by an invertible diagonal matrix is a sequence of row scalings. -/
theorem reach_diagonal {k : ℕ} (D : Fin k → F) (hD : ∀ i, D i ≠ 0) (M : Matrix (Fin k) E F) :
    ReflTransGen ElemOp (⟨k, M⟩ : RowMatrix E F) ⟨k, diagonal D * M⟩ := by
  suffices h : ∀ (t : Finset (Fin k)) (D : Fin k → F), (∀ i, D i ≠ 0) → (∀ i, i ∉ t → D i = 1) →
      ∀ M : Matrix (Fin k) E F,
        ReflTransGen ElemOp (⟨k, M⟩ : RowMatrix E F) ⟨k, diagonal D * M⟩ from
    h Finset.univ D hD (fun i hi => (hi (Finset.mem_univ i)).elim) M
  intro t
  induction t using Finset.induction_on with
  | empty =>
    intro D _ hD M
    have : D = fun _ => 1 := funext fun i => hD i (by simp)
    rw [this, diagonal_one, Matrix.one_mul]
  | insert a t ha ih =>
    intro D hne hD M
    have hstep : ElemOp (⟨k, M⟩ : RowMatrix E F)
        ⟨k, M.updateRow a (((Units.mk0 (D a) (hne a) : Fˣ) : F) • M a)⟩ :=
      ElemOp.scaleRow M a _
    have hrest := ih (Function.update D a 1)
      (fun i => by
        by_cases hi : i = a
        · subst hi; simp
        · simpa [Function.update_of_ne hi] using hne i)
      (fun i hi => by
        by_cases hia : i = a
        · subst hia; simp
        · rw [Function.update_of_ne hia]
          exact hD i (by simp [hia, hi]))
      (M.updateRow a (((Units.mk0 (D a) (hne a) : Fˣ) : F) • M a))
    have heq : diagonal (Function.update D a 1) *
        M.updateRow a (((Units.mk0 (D a) (hne a) : Fˣ) : F) • M a) = diagonal D * M := by
      ext i j
      rw [diagonal_mul, diagonal_mul, updateRow_apply]
      by_cases hi : i = a
      · subst hi; simp
      · simp [hi]
    rw [heq] at hrest
    exact ReflTransGen.head hstep hrest

/-- Multiplying by a transvection is a sequence of two row scalings and one row addition. -/
theorem reach_transvection {k : ℕ} (t : TransvectionStruct (Fin k) F) (M : Matrix (Fin k) E F) :
    ReflTransGen ElemOp (⟨k, M⟩ : RowMatrix E F) ⟨k, t.toMatrix * M⟩ := by
  obtain ⟨i, j, hij, c⟩ := t
  by_cases hc : c = 0
  · subst hc
    simp only [TransvectionStruct.toMatrix, transvection_zero, Matrix.one_mul]
    exact ReflTransGen.refl
  · let u : Fˣ := Units.mk0 c hc
    let M₁ : Matrix (Fin k) E F := M.updateRow j ((u : F) • M j)
    let M₂ : Matrix (Fin k) E F := M₁.updateRow i (M₁ i + M₁ j)
    let M₃ : Matrix (Fin k) E F := M₂.updateRow j (((u⁻¹ : Fˣ) : F) • M₂ j)
    have h₁ : ElemOp (⟨k, M⟩ : RowMatrix E F) ⟨k, M₁⟩ := ElemOp.scaleRow M j u
    have h₂ : ElemOp (⟨k, M₁⟩ : RowMatrix E F) ⟨k, M₂⟩ := ElemOp.addRow M₁ hij
    have h₃ : ElemOp (⟨k, M₂⟩ : RowMatrix E F) ⟨k, M₃⟩ := ElemOp.scaleRow M₂ j u⁻¹
    have hu : ((u⁻¹ : Fˣ) : F) * (u : F) = 1 := Units.inv_mul u
    have hM₁i : M₁ i = M i := updateRow_ne hij
    have hM₁j : M₁ j = (u : F) • M j := updateRow_self
    have hM₂i : M₂ i = M i + (u : F) • M j := by
      change M₁.updateRow i (M₁ i + M₁ j) i = _
      rw [updateRow_self, hM₁i, hM₁j]
    have hM₂j : M₂ j = (u : F) • M j := by
      change M₁.updateRow i (M₁ i + M₁ j) j = _
      rw [updateRow_ne hij.symm, hM₁j]
    have hM : M₃ = transvection i j c * M := by
      ext a b
      by_cases haj : a = j
      · subst haj
        have : M₃ a = ((u⁻¹ : Fˣ) : F) • M₂ a := updateRow_self
        rw [this, hM₂j, smul_smul, hu, one_smul,
          transvection_mul_apply_of_ne i a a b hij.symm]
      · have h₃a : M₃ a = M₂ a := updateRow_ne haj
        by_cases hai : a = i
        · subst hai
          rw [h₃a, hM₂i, transvection_mul_apply_same]
          simp [u]
        · have h₂a : M₂ a = M₁ a := updateRow_ne hai
          have h₁a : M₁ a = M a := updateRow_ne haj
          rw [h₃a, h₂a, h₁a, transvection_mul_apply_of_ne i j a b hai]
    change ReflTransGen ElemOp (⟨k, M⟩ : RowMatrix E F) ⟨k, transvection i j c * M⟩
    rw [← hM]
    exact (ReflTransGen.single h₁).tail h₂ |>.tail h₃

/-- Multiplying on the left by a nonsingular matrix is a sequence of row operations. -/
theorem reach_mul {k : ℕ} (T : Matrix (Fin k) (Fin k) F) (hT : T.det ≠ 0)
    (M : Matrix (Fin k) E F) :
    ReflTransGen ElemOp (⟨k, M⟩ : RowMatrix E F) ⟨k, T * M⟩ := by
  revert M
  refine diagonal_transvection_induction_of_det_ne_zero
    (fun T => ∀ M : Matrix (Fin k) E F,
      ReflTransGen ElemOp (⟨k, M⟩ : RowMatrix E F) ⟨k, T * M⟩) T hT ?_ ?_ ?_
  · intro D hD M
    refine reach_diagonal D (fun i hi => hD ?_) M
    rw [det_diagonal]
    exact Finset.prod_eq_zero (Finset.mem_univ i) hi
  · exact fun t M => reach_transvection t M
  · intro A B _ _ hA hB M
    rw [Matrix.mul_assoc]
    exact (hB M).trans (hA (B * M))

/-- A nonsingular diagonal matrix is the diagonal matrix of its unit entries. -/
theorem exists_units_of_isDiag_of_isUnit {n : Type*} [Fintype n] [DecidableEq n]
    {S : Matrix n n F} (hD : S.IsDiag) (hU : IsUnit S) :
    ∃ s : n → Fˣ, diagonal (fun j => (s j : F)) = S := by
  have hSeq : diagonal (diag S) = S := hD.diagonal_diag
  have hdet : S.det ≠ 0 := ((Matrix.isUnit_iff_isUnit_det S).1 hU).ne_zero
  rw [← hSeq, det_diagonal] at hdet
  have hne : ∀ j, S j j ≠ 0 := fun j => Finset.prod_ne_zero_iff.1 hdet j (Finset.mem_univ j)
  exact ⟨fun j => Units.mk0 (S j j) (hne j), hSeq⟩

/-- Zero rows can be appended up to any larger number of rows, reversibly. -/
theorem exists_pad {a : ℕ} (X : Matrix (Fin a) E F) (k : ℕ) (hk : a ≤ k) :
    ∃ X' : Matrix (Fin k) E F, rowSpace X' = rowSpace X ∧
      ReflTransGen ElemOp (⟨a, X⟩ : RowMatrix E F) ⟨k, X'⟩ ∧
      ReflTransGen ElemOp (⟨k, X'⟩ : RowMatrix E F) ⟨a, X⟩ := by
  induction k, hk using Nat.le_induction with
  | base => exact ⟨X, rfl, .refl, .refl⟩
  | succ k _ ih =>
    obtain ⟨X', h₁, h₂, h₃⟩ := ih
    exact ⟨snocZero X', by rw [rowSpace_snocZero, h₁], h₂.tail (ElemOp.addZeroRow X'),
      h₃.head (ElemOp.removeZeroRow X')⟩

/-- Sequences of the elementary operations of §1.1 are exactly `ProjEquiv`, also between matrices
with different numbers of rows. -/
theorem elemOps_iff_projEquiv {m n : ℕ} [Finite E] (A : Matrix (Fin m) E F)
    (B : Matrix (Fin n) E F) :
    ReflTransGen ElemOp (⟨m, A⟩ : RowMatrix E F) ⟨n, B⟩ ↔ ProjEquiv A B := by
  constructor
  · intro h
    have key : ∀ Y : RowMatrix E F, ReflTransGen ElemOp (⟨m, A⟩ : RowMatrix E F) Y →
        ProjEquiv A Y.2 := by
      intro Y hY
      induction hY with
      | refl => exact ProjEquiv.refl A
      | tail _ hstep ih => exact ih.trans hstep.projEquiv
    exact key _ h
  · rintro ⟨s, hs⟩
    obtain ⟨A', hA₁, hA₂, -⟩ := exists_pad (scaleCols s A) (max m n) (le_max_left m n)
    obtain ⟨B', hB₁, -, hB₃⟩ := exists_pad B (max m n) (le_max_right m n)
    obtain ⟨T, hT, hTA⟩ := exists_isUnit_mul_eq_of_rowSpace_eq A' B' (by rw [hA₁, hB₁, hs])
    have hdet : T.det ≠ 0 := ((Matrix.isUnit_iff_isUnit_det T).1 hT).ne_zero
    have hmid := reach_mul T hdet A'
    rw [hTA] at hmid
    exact (reach_scaleCols A s).trans (hA₂.trans (hmid.trans hB₃))

/-- **Proposition 1.1** (Brylawski and Lucas; Oxley, Proposition 6.3.12). `r × n` matrices with the
same labelled columns are projectively equivalent, that is, joined by a sequence of the elementary
operations of §1.1, exactly when `T A S = B` for a nonsingular `T` and a nonsingular diagonal `S`.
The source assumes `r ≥ 1`, which is not needed. -/
theorem elemOps_iff_exists_mul_mul {r : ℕ} [Fintype E] [DecidableEq E]
    (A B : Matrix (Fin r) E F) :
    ReflTransGen ElemOp (⟨r, A⟩ : RowMatrix E F) ⟨r, B⟩ ↔
      ∃ (T : Matrix (Fin r) (Fin r) F) (S : Matrix E E F),
        IsUnit T ∧ IsUnit S ∧ S.IsDiag ∧ T * A * S = B := by
  rw [elemOps_iff_projEquiv, projEquiv_iff_exists_mul_mul_diagonal]
  constructor
  · rintro ⟨T, s, hT, rfl⟩
    refine ⟨T, diagonal fun j => (s j : F), hT, ?_, isDiag_diagonal _, rfl⟩
    rw [Matrix.isUnit_iff_isUnit_det, det_diagonal, isUnit_iff_ne_zero]
    exact Finset.prod_ne_zero_iff.2 fun j _ => (s j).ne_zero
  · rintro ⟨T, S, hT, hS, hD, rfl⟩
    obtain ⟨s, rfl⟩ := exists_units_of_isDiag_of_isUnit hD hS
    exact ⟨T, s, hT, rfl⟩

example : ReflTransGen ElemOp (⟨2, 1⟩ : RowMatrix (Fin 2) ℚ)
    ⟨2, diagonal (fun _ => 1) * (1 : Matrix (Fin 2) (Fin 2) ℚ)⟩ :=
  reach_diagonal _ (fun _ => one_ne_zero) _

example : ReflTransGen ElemOp (⟨2, 1⟩ : RowMatrix (Fin 2) ℚ)
    ⟨2, (1 : Matrix (Fin 2) (Fin 2) ℚ) * (1 : Matrix (Fin 2) (Fin 2) ℚ)⟩ :=
  reach_mul _ (by simp) _

example : ∃ X' : Matrix (Fin 3) (Fin 2) ℚ, rowSpace X' = rowSpace (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧
    ReflTransGen ElemOp (⟨2, 1⟩ : RowMatrix (Fin 2) ℚ) ⟨3, X'⟩ ∧
    ReflTransGen ElemOp (⟨3, X'⟩ : RowMatrix (Fin 2) ℚ) ⟨2, 1⟩ :=
  exists_pad _ 3 (by decide)

end Transformer.GainMatroid
