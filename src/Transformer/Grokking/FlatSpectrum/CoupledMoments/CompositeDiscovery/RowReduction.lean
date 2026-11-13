module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Basic

/-!
# Exact row-program certificates over composite rings

Source: `review/flat_spectrum.md`, CM28, the elimination certificate. A finite
record of actual swaps, scalings and row additions is replayed on the original
matrix. If the final matrix contains every coordinate row, its original
kernel is zero. This only uses preservation of homogeneous equations and does
not pretend that every nonzero residue is invertible. The native constructor
must separately check any unit inverse it uses to produce this certificate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

variable {R I V : Type*} [CommRing R] [Fintype V]

/-- The actual finite row action, CM28. -/
def rowAction (A : I → V → R) (x : V → R) (i : I) : R := ∑ j, A i j * x j

/-- An explicit elementary row instruction, CM28. No correctness claim or
unproved input is stored in an instruction. -/
inductive RowOp (I R : Type*)
  | swap (i j : I)
  | scale (i : I) (a : R)
  | add (i j : I) (a : R)

variable [DecidableEq I]

/-- Execute an instruction on every coefficient of the actual matrix, CM28. -/
def applyRow (op : RowOp I R) (A : I → V → R) : I → V → R :=
  match op with
  | .swap i j => fun k => if k = i then A j else if k = j then A i else A k
  | .scale i a => fun k v => if k = i then a * A k v else A k v
  | .add i j a => fun k v => if k = i then A k v + a * A j v else A k v

/-- Replay the complete finite instruction sequence in its stated order. -/
def reduceRows : List (RowOp I R) → (I → V → R) → I → V → R
  | [], A => A
  | op :: rest, A => reduceRows rest (applyRow op A)

/-- Every allowed instruction preserves solutions of the original
homogeneous system, CM28, including in rings with zero divisors. -/
theorem applyRow_zero (op : RowOp I R) {A : I → V → R} {x : V → R}
    (h : ∀ i, rowAction A x i = 0) : ∀ i, rowAction (applyRow op A) x i = 0 := by
  intro k
  cases op with
  | swap i j =>
    by_cases hi : k = i
    · subst k
      simpa [rowAction, applyRow] using h j
    · by_cases hj : k = j
      · subst k
        simpa [rowAction, applyRow, hi] using h i
      · simpa [rowAction, applyRow, hi, hj] using h k
  | scale i a =>
    by_cases hi : k = i
    · simpa [rowAction, applyRow, hi, mul_assoc, ← Finset.mul_sum] using
        congrArg (a * ·) (h k)
    · simpa [rowAction, applyRow, hi] using h k
  | add i j a =>
    by_cases hi : k = i
    · simp only [rowAction, applyRow, hi, ite_eq_left, add_mul, mul_assoc,
        Finset.sum_add_distrib, ← Finset.mul_sum]
      change rowAction A x i + a * rowAction A x j = 0
      rw [h i, h j, mul_zero, add_zero]
    · simpa [rowAction, applyRow, hi] using h k

/-- A replayed row program preserves every original homogeneous solution,
CM28. No property of a solver's reported dimension is assumed. -/
theorem reduceRows_zero (ops : List (RowOp I R)) {A : I → V → R} {x : V → R}
    (h : ∀ i, rowAction A x i = 0) : ∀ i, rowAction (reduceRows ops A) x i = 0 := by
  induction ops generalizing A with
  | nil => exact h
  | cons op rest ih => exact ih (applyRow_zero op h)

variable [DecidableEq V]

/-- If the replayed matrix contains each coordinate equation, the actual
original kernel is zero, CM28. Rows need not form a square matrix. -/
theorem kernel_zero_of_rows (ops : List (RowOp I R)) (A : I → V → R)
    (pivot : V → I)
    (hrows : ∀ j k, reduceRows ops A (pivot j) k = if j = k then 1 else 0)
    {x : V → R} (hx : ∀ i, rowAction A x i = 0) : x = 0 := by
  have hz := reduceRows_zero ops hx
  funext j
  simpa [rowAction, hrows, ite_mul] using hz (pivot j)

/-- Both leading entries are nonunits modulo six, but row addition exposes
a unit. The full program below reduces this nontrivial matrix to identity. -/
def sixMatrix : Fin 2 → Fin 2 → ZMod 6 := ![![2, 1], ![3, 1]]

/-- Actual instructions; multiplying the new leading entry five by five
normalizes it modulo six. No inverse of two or three is used. -/
def sixProgram : List (RowOp (Fin 2) (ZMod 6)) :=
  [.add 0 1 1, .scale 0 5, .add 1 0 3, .add 0 1 2]

/-- Kernel-checked execution of the composite witness program, CM28. -/
theorem sixProgram_identity : ∀ j k,
    reduceRows sixProgram sixMatrix j k = if j = k then 1 else 0 := by
  decide

example : ∀ i, rowAction
    (applyRow (.add 0 1 2) (![![1, 1], ![2, 2]] : Fin 2 → Fin 2 → ZMod 6))
    ![1, 5] i = 0 :=
  applyRow_zero _ (by decide)

example : ∀ i, rowAction
    (reduceRows sixProgram (![![1, 1], ![2, 2]] : Fin 2 → Fin 2 → ZMod 6))
    ![1, 5] i = 0 :=
  reduceRows_zero _ (by decide)

example : (fun _ : Fin 2 => (0 : ZMod 6)) = 0 :=
  kernel_zero_of_rows sixProgram sixMatrix id sixProgram_identity (by decide)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery
