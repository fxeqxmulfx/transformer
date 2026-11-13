module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientHead

/-!
# Bounded state-logit tables in actual complex coordinates

Source: `review/flat_spectrum.md`, CM32. Extend CFt5's one-hot initializer
bound to every real table in a coordinate box. Unitary inversion realizes
the table exactly and bounds both learned complex coordinates by `2 A N`.
This permits explicit feasible updates without a dense global projection.
The code is fixed; neither its discovery nor optimizer convergence is claimed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix

variable {I C : Type*} [Fintype I] [DecidableEq I]

/-- Embed a real state-logit table into complex matrices without changing
its entries, CM32. -/
def complexTable (T : Matrix I C ℝ) : Matrix I C ℂ := fun s c => (T s c : ℂ)

/-- The conjugate transform of an arbitrary real state-logit table, CM32.
Both real and imaginary coordinates are retained in the actual head. -/
noncomputable def tableHead (U : Matrix I I ℂ) (T : Matrix I C ℝ) : Matrix I C ℂ :=
  U.conjTranspose * complexTable T

/-- Unitarity realizes every bounded or unbounded real table through the
actual complex forward multiplication, CM32. -/
theorem tableHead_logits (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (T : Matrix I C ℝ) (s : I) :
    headLogits (complexFeatures U) (complexCoefficients (tableHead U T)) s = T s := by
  have hu : U * U.conjTranspose = 1 := Unitary.mul_star_self_of_mem hU
  funext c
  rw [complex_head_logits, tableHead, ← Matrix.mul_assoc, hu,
    Matrix.one_mul]
  rfl

omit [DecidableEq I] in
/-- Conjugate transport is linear in the complete real table, so an update
does not introduce independently learned multiplicative factors, CM32. -/
theorem tableHead_linear (U : Matrix I I ℂ) (T V : Matrix I C ℝ) (a b : ℝ) :
    tableHead U (a • T + b • V) = a • tableHead U T + b • tableHead U V := by
  have h : complexTable (a • T + b • V) = a • complexTable T + b • complexTable V := by
    ext s c
    simp [complexTable, Complex.ofReal_mul]
  simp only [tableHead, h, Matrix.mul_add, Matrix.mul_smul]

omit [DecidableEq I] in
/-- A proposed table update transports to the actual complex head update,
CM32. The direction is arbitrary here; native CE derivatives need a separate
numerical check, and feasibility is supplied by the later box clipping. -/
theorem tableHead_proposal (U : Matrix I I ℂ) (T G : Matrix I C ℝ) (η : ℝ) :
    tableHead U (T - η • G) = tableHead U T - η • tableHead U G := by
  have h := tableHead_linear U T G 1 (-η)
  simpa only [one_smul, neg_smul, sub_eq_add_neg] using h

/-- A unitary conjugate transform preserves each column's squared energy,
not just the sum over all classes. This removes a spurious vocabulary factor. -/
theorem unitary_star_mul_column_sq (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (A : Matrix I C ℂ) (c : C) :
    (∑ k, ‖(U.conjTranspose * A) k c‖^2) = ∑ k, ‖A k c‖^2 := by
  have hu : U * U.conjTranspose = 1 := Unitary.mul_star_self_of_mem hU
  have hg : (U.conjTranspose * A).conjTranspose * (U.conjTranspose * A) =
      A.conjTranspose * A := by
    rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose,
      Matrix.mul_assoc, ← Matrix.mul_assoc U U.conjTranspose A,
      hu, Matrix.one_mul]
  have h := congrArg (fun B : Matrix C C ℂ => (B c c).re) hg
  simpa only [conjTranspose_mul_self_apply_self, Complex.ofReal_re] using h

/-- Both real coefficient coordinates have exactly the table's per-class
energy, CM32. This identity applies at every feasible iterate. -/
theorem tableHead_column_sq (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (T : Matrix I C ℝ) (c : C) :
    (∑ k : I × Bool, (complexCoefficients (tableHead U T) k c)^2) =
      ∑ s, (T s c)^2 := by
  rw [complexCoefficients_sq, tableHead, unitary_star_mul_column_sq U hU]
  simp only [complexTable, Complex.norm_real, Real.norm_eq_abs, sq_abs]

/-- An entrywise bound controls every output column independently of the
number of classes, CM32. -/
theorem tableHead_l1_sq (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (T : Matrix I C ℝ) {A : ℝ} (hA : 0 ≤ A) (hT : ∀ s c, |T s c| ≤ A) (c : C) :
    (∑ k : I × Bool, |complexCoefficients (tableHead U T) k c|)^2 ≤
      2 * (Fintype.card I : ℝ)^2 * A^2 := by
  have hcs := sq_sum_le_card_mul_sum_sq (s := Finset.univ)
    (f := fun k : I × Bool => |complexCoefficients (tableHead U T) k c|)
  simp only [Finset.card_univ, Fintype.card_prod, Fintype.card_bool, Nat.cast_mul,
    Nat.cast_ofNat, sq_abs] at hcs
  rw [tableHead_column_sq U hU] at hcs
  have he : (∑ s, (T s c)^2) ≤ (Fintype.card I : ℝ) * A^2 := by
    calc
      _ ≤ ∑ _ : I, A^2 := by
        apply Finset.sum_le_sum
        intro s hs
        have ht := hT s c
        nlinarith [sq_abs (T s c), abs_nonneg (T s c)]
      _ = _ := by simp
  calc
    _ ≤ ((Fintype.card I : ℝ) * 2) * ∑ s, (T s c)^2 := hcs
    _ ≤ ((Fintype.card I : ℝ) * 2) * ((Fintype.card I : ℝ) * A^2) :=
      mul_le_mul_of_nonneg_left he (by positivity)
    _ = _ := by ring

/-- The actual complex head of every table in `[-A,A]` satisfies the same
per-class budget `2 A N`, CM32. This is not restricted to initial margins. -/
theorem tableHead_budget (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (T : Matrix I C ℝ) {A : ℝ} (hA : 0 ≤ A) (hT : ∀ s c, |T s c| ≤ A) :
    complexCoefficients (tableHead U T) ∈ headBudget (2 * A * (Fintype.card I : ℝ)) := by
  intro c
  have hs := tableHead_l1_sq U hU T hA hT c
  have hn : (0 : ℝ) ≤ Fintype.card I := Nat.cast_nonneg _
  have hp : 0 ≤ (Fintype.card I : ℝ)^2 * A^2 := mul_nonneg (sq_nonneg _) (sq_nonneg _)
  have hnon : 0 ≤ 2 * A * (Fintype.card I : ℝ) := by positivity
  nlinarith

/-- Native box radius32 therefore fits the already declared coefficient
radius64 times the actual number of states, CM32. -/
theorem tableHead_native_budget (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (T : Matrix I C ℝ) (hT : ∀ s c, |T s c| ≤ 32) :
    complexCoefficients (tableHead U T) ∈ headBudget (64 * (Fintype.card I : ℝ)) := by
  have h := tableHead_budget U hU T (A := 32) (by norm_num) hT
  norm_num at h
  exact h

private def exampleTable (s : Fin 4) (c : Bool) : ℝ :=
  if c then (s.val : ℝ) - 1 else 2 - (s.val : ℝ)

example := tableHead_logits fourier4 fourier4_unitary exampleTable 2

example := tableHead_proposal fourier4 exampleTable (fun _ _ => 1) (1 / 2)

example := unitary_star_mul_column_sq fourier4 fourier4_unitary
  (fun s c => (exampleTable s c : ℂ)) true

example := tableHead_column_sq fourier4 fourier4_unitary exampleTable false

example := tableHead_l1_sq fourier4 fourier4_unitary exampleTable
  (A := 2) (by norm_num) (by intro s c; fin_cases s <;> cases c <;> norm_num [exampleTable]) true

example := tableHead_budget fourier4 fourier4_unitary exampleTable
  (A := 2) (by norm_num) (by intro s c; fin_cases s <;> cases c <;> norm_num [exampleTable])

example := tableHead_native_budget fourier4 fourier4_unitary exampleTable
  (by intro s c; fin_cases s <;> cases c <;> norm_num [exampleTable])

end Transformer.Grokking.FlatSpectrum.ConvexDepth
