module

public import Mathlib.LinearAlgebra.Matrix.IsDiag
public import Mathlib.LinearAlgebra.Matrix.Transvection
public import Mathlib.Logic.Relation
public import Transformer.GainMatroid.Section1_Projective

/-!
# The elementary operations of projective equivalence

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.1. The source defines projective equivalence by
sequences of elementary operations: interchange two rows, multiply a row by a nonzero scalar,
replace a row by its sum with another row, add or remove a zero row, interchange two columns along
with their labels, and multiply a column by a nonzero scalar. A column here is a function of its
label, so interchanging two columns along with their labels leaves the matrix as it is. A zero row
is added or removed at the end; an interchange moves it to any other place.

This module defines the operations and proves that each one preserves `ProjEquiv`, and that
scaling all columns and multiplying on the left by a nonsingular matrix are sequences of them.
-/

@[expose] public section

open Matrix Relation

namespace Transformer.GainMatroid

variable {E F : Type*} [Field F]

/-- Matrices with columns labelled by `E` and any finite number of rows. -/
abbrev RowMatrix (E F : Type*) := Σ m : ℕ, Matrix (Fin m) E F

/-- `A` with a zero row appended. -/
def snocZero {m : ℕ} (A : Matrix (Fin m) E F) : Matrix (Fin (m + 1)) E F :=
  Matrix.of (Fin.snoc (α := fun _ => E → F) A.row 0)

/-- One elementary operation of §1.1 (see the module docstring for the column interchange). -/
inductive ElemOp : RowMatrix E F → RowMatrix E F → Prop
  | swapRows {m : ℕ} (A : Matrix (Fin m) E F) (i j : Fin m) :
      ElemOp ⟨m, A⟩ ⟨m, A.submatrix (Equiv.swap i j) id⟩
  | scaleRow {m : ℕ} (A : Matrix (Fin m) E F) (i : Fin m) (c : Fˣ) :
      ElemOp ⟨m, A⟩ ⟨m, A.updateRow i ((c : F) • A i)⟩
  | addRow {m : ℕ} (A : Matrix (Fin m) E F) {i j : Fin m} (hij : i ≠ j) :
      ElemOp ⟨m, A⟩ ⟨m, A.updateRow i (A i + A j)⟩
  | addZeroRow {m : ℕ} (A : Matrix (Fin m) E F) : ElemOp ⟨m, A⟩ ⟨m + 1, snocZero A⟩
  | removeZeroRow {m : ℕ} (A : Matrix (Fin m) E F) : ElemOp ⟨m + 1, snocZero A⟩ ⟨m, A⟩
  | scaleCol {m : ℕ} (A : Matrix (Fin m) E F) (j : E) (s : E → Fˣ) (hs : ∀ k, k ≠ j → s k = 1) :
      ElemOp ⟨m, A⟩ ⟨m, scaleCols s A⟩

theorem row_mem_rowSpace {R : Type*} (M : Matrix R E F) (i : R) : M i ∈ rowSpace M :=
  Submodule.subset_span ⟨i, rfl⟩

theorem rowSpace_le_of_forall_mem {R R' : Type*} {M : Matrix R E F} {N : Matrix R' E F}
    (h : ∀ i, M i ∈ rowSpace N) : rowSpace M ≤ rowSpace N :=
  Submodule.span_le.2 (by rintro _ ⟨i, rfl⟩; exact h i)

theorem projEquiv_of_rowSpace_eq {R R' : Type*} {A : Matrix R E F} {B : Matrix R' E F}
    (h : rowSpace A = rowSpace B) : ProjEquiv A B :=
  ⟨1, by rw [scaleCols_one, h]⟩

theorem rowSpace_snocZero {m : ℕ} (A : Matrix (Fin m) E F) :
    rowSpace (snocZero A) = rowSpace A := by
  have h : Set.range (snocZero A).row = insert 0 (Set.range A.row) :=
    Fin.range_snoc A.row (0 : E → F)
  rw [rowSpace, h, Submodule.span_insert_zero, rowSpace]

theorem rowSpace_swapRows {m : ℕ} (A : Matrix (Fin m) E F) (i j : Fin m) :
    rowSpace (A.submatrix (Equiv.swap i j) id) = rowSpace A := by
  refine le_antisymm (rowSpace_le_of_forall_mem fun k => row_mem_rowSpace A (Equiv.swap i j k))
    (rowSpace_le_of_forall_mem fun k => ?_)
  have h : A k = (A.submatrix (Equiv.swap i j) id) (Equiv.swap i j k) := by
    ext b
    simp
  rw [h]
  exact row_mem_rowSpace _ _

theorem rowSpace_scaleRow {m : ℕ} (A : Matrix (Fin m) E F) (i : Fin m) (c : Fˣ) :
    rowSpace (A.updateRow i ((c : F) • A i)) = rowSpace A := by
  refine le_antisymm (rowSpace_le_of_forall_mem fun k => ?_)
    (rowSpace_le_of_forall_mem fun k => ?_)
  · by_cases hk : k = i
    · subst hk
      rw [updateRow_self]
      exact Submodule.smul_mem _ _ (row_mem_rowSpace A k)
    · rw [updateRow_ne hk]
      exact row_mem_rowSpace A k
  · by_cases hk : k = i
    · subst hk
      have h := Submodule.smul_mem _ ((c⁻¹ : Fˣ) : F)
        (row_mem_rowSpace (A.updateRow k ((c : F) • A k)) k)
      rw [updateRow_self, smul_smul, Units.inv_mul, one_smul] at h
      exact h
    · have h := row_mem_rowSpace (A.updateRow i ((c : F) • A i)) k
      rwa [updateRow_ne hk] at h

theorem rowSpace_addRow {m : ℕ} (A : Matrix (Fin m) E F) {i j : Fin m} (hij : i ≠ j) :
    rowSpace (A.updateRow i (A i + A j)) = rowSpace A := by
  refine le_antisymm (rowSpace_le_of_forall_mem fun k => ?_)
    (rowSpace_le_of_forall_mem fun k => ?_)
  · by_cases hk : k = i
    · subst hk
      rw [updateRow_self]
      exact Submodule.add_mem _ (row_mem_rowSpace A k) (row_mem_rowSpace A j)
    · rw [updateRow_ne hk]
      exact row_mem_rowSpace A k
  · by_cases hk : k = i
    · subst hk
      have h₁ := row_mem_rowSpace (A.updateRow k (A k + A j)) k
      have h₂ := row_mem_rowSpace (A.updateRow k (A k + A j)) j
      rw [updateRow_self] at h₁
      rw [updateRow_ne hij.symm] at h₂
      simpa using Submodule.sub_mem _ h₁ h₂
    · have h := row_mem_rowSpace (A.updateRow i (A i + A j)) k
      rwa [updateRow_ne hk] at h

/-- Every elementary operation preserves `ProjEquiv`. -/
theorem ElemOp.projEquiv {X Y : RowMatrix E F} (h : ElemOp X Y) : ProjEquiv X.2 Y.2 := by
  cases h with
  | swapRows A i j => exact projEquiv_of_rowSpace_eq (rowSpace_swapRows A i j).symm
  | scaleRow A i c => exact projEquiv_of_rowSpace_eq (rowSpace_scaleRow A i c).symm
  | addRow A hij => exact projEquiv_of_rowSpace_eq (rowSpace_addRow A hij).symm
  | addZeroRow A => exact projEquiv_of_rowSpace_eq (rowSpace_snocZero A).symm
  | removeZeroRow A => exact projEquiv_of_rowSpace_eq (rowSpace_snocZero A)
  | scaleCol A j s hs => exact ⟨s, rfl⟩

/-- Scaling all columns is a sequence of single column scalings. -/
theorem reach_scaleCols {m : ℕ} [Finite E] (A : Matrix (Fin m) E F) (s : E → Fˣ) :
    ReflTransGen ElemOp (⟨m, A⟩ : RowMatrix E F) ⟨m, scaleCols s A⟩ := by
  classical
  cases nonempty_fintype E
  suffices h : ∀ (t : Finset E) (u : E → Fˣ), (∀ k, k ∉ t → u k = 1) →
      ∀ A : Matrix (Fin m) E F,
        ReflTransGen ElemOp (⟨m, A⟩ : RowMatrix E F) ⟨m, scaleCols u A⟩ from
    h Finset.univ s (fun k hk => (hk (Finset.mem_univ k)).elim) A
  intro t
  induction t using Finset.induction_on with
  | empty =>
    intro u hu A
    have : u = 1 := funext fun k => hu k (by simp)
    rw [this, scaleCols_one]
  | insert a t ha ih =>
    intro u hu A
    have hvw : Function.update (1 : E → Fˣ) a (u a) * Function.update u a 1 = u := by
      funext k
      by_cases hk : k = a
      · subst hk; simp
      · simp [Function.update_of_ne hk]
    have hw : ∀ k, k ∉ t → Function.update u a 1 k = 1 := by
      intro k hk
      by_cases hka : k = a
      · subst hka; simp
      · rw [Function.update_of_ne hka]
        exact hu k (by simp [hka, hk])
    have hstep : ElemOp (⟨m, A⟩ : RowMatrix E F)
        ⟨m, scaleCols (Function.update (1 : E → Fˣ) a (u a)) A⟩ :=
      ElemOp.scaleCol A a _ fun k hk => by simp [Function.update_of_ne hk]
    have htail := ih _ hw (scaleCols (Function.update (1 : E → Fˣ) a (u a)) A)
    rw [scaleCols_scaleCols, hvw] at htail
    exact ReflTransGen.head hstep htail

example : rowSpace (1 : Matrix (Fin 2) (Fin 2) ℚ) ≤ rowSpace (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
  rowSpace_le_of_forall_mem fun i => row_mem_rowSpace _ i

example : ProjEquiv (1 : Matrix (Fin 2) (Fin 2) ℚ) (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
  projEquiv_of_rowSpace_eq rfl

example : rowSpace ((1 : Matrix (Fin 2) (Fin 2) ℚ).updateRow 0
    ((1 : Matrix (Fin 2) (Fin 2) ℚ) 0 + (1 : Matrix (Fin 2) (Fin 2) ℚ) 1)) =
    rowSpace (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
  rowSpace_addRow _ (by decide)

example : ProjEquiv (1 : Matrix (Fin 2) (Fin 2) ℚ) (snocZero (1 : Matrix (Fin 2) (Fin 2) ℚ)) :=
  (ElemOp.addZeroRow (1 : Matrix (Fin 2) (Fin 2) ℚ)).projEquiv

end Transformer.GainMatroid
