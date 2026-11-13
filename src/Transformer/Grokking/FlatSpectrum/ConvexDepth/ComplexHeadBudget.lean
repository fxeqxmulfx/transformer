module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHead
public import Transformer.Grokking.FlatSpectrum.Factorization
public import Mathlib.Algebra.Order.Chebyshev

/-!
# A sufficient convex budget for the actual complex initializer

Source: `review/flat_spectrum.md`, CFt5. Unitary inversion preserves the
one-hot margin table's squared energy. Cauchy--Schwarz controls every class's
real/imaginary L1 column without asserting flatness of the learned head.
The native margin24 initializer fits radius64 times the total state count.
The column budget counts both real and imaginary coefficients together.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Matrix Finset

variable {I C : Type*} [Fintype I]

/-- Realification preserves squared column energy, CFt5. -/
theorem complexCoefficients_sq (H : Matrix I C ℂ) (c : C) :
    (∑ k : I × Bool, (complexCoefficients H k c)^2) = ∑ k, ‖H k c‖^2 := by
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro k hk
  simp only [Fintype.sum_bool, complexCoefficients, Bool.false_eq_true, ite_false, ite_true,
    Complex.sq_norm, Complex.normSq_apply]
  ring

variable [DecidableEq I] [Fintype C]

/-- Left multiplication by the conjugate of a unitary matrix preserves
the full squared Frobenius norm, CFt5. -/
theorem unitary_star_mul_frobenius (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (A : Matrix I C ℂ) :
    frobeniusSq (U.conjTranspose * A) = frobeniusSq A := by
  have hu : U * U.conjTranspose = 1 := Unitary.mul_star_self_of_mem hU
  rw [frobeniusSq_eq_re_trace, frobeniusSq_eq_re_trace,
    Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose,
    Matrix.mul_assoc, ← Matrix.mul_assoc U U.conjTranspose A, hu, Matrix.one_mul]

variable [DecidableEq C]

omit [DecidableEq I] in
/-- Exactly one assigned class per state gives squared energy `|I| a²`,
CFt5, regardless of repeated class assignments. -/
theorem stateMargins_frobenius (labels : I → C) (a : ℝ) :
    frobeniusSq (stateMargins labels a) = (Fintype.card I : ℝ) * a^2 := by
  simp [frobeniusSq, stateMargins, apply_ite norm, apply_ite (fun x : ℝ => x^2)]

/-- The initialized complex head has the same exact energy as its declared
margin table, CFt5. No target labels beyond that table are inferred. -/
theorem complexInitializer_frobenius (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) (a : ℝ) :
    frobeniusSq (complexInitializer U labels a) = (Fintype.card I : ℝ) * a^2 := by
  rw [complexInitializer, unitary_star_mul_frobenius U hU, stateMargins_frobenius]

/-- Any single class column has at most the full initialized energy, CFt5. -/
theorem complexInitializer_column_sq (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) (a : ℝ) (c : C) :
    (∑ k, ‖complexInitializer U labels a k c‖^2) ≤ (Fintype.card I : ℝ) * a^2 := by
  rw [← complexInitializer_frobenius U hU labels a]
  dsimp only [frobeniusSq]
  apply Finset.sum_le_sum
  intro k hk
  exact Finset.single_le_sum
    (f := fun d : C => ‖complexInitializer U labels a k d‖^2)
    (fun d _ => sq_nonneg _) (Finset.mem_univ c)

/-- Cauchy--Schwarz on the actual `2|I|` real coordinates gives the
dimension-explicit squared L1 bound, CFt5. -/
theorem complexInitializer_l1_sq (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) (a : ℝ) (c : C) :
    (∑ k : I × Bool, |complexCoefficients (complexInitializer U labels a) k c|)^2 ≤
      2 * (Fintype.card I : ℝ)^2 * a^2 := by
  have hcs := sq_sum_le_card_mul_sum_sq (s := Finset.univ)
    (f := fun k : I × Bool => |complexCoefficients (complexInitializer U labels a) k c|)
  simp only [Finset.card_univ, Fintype.card_prod, Fintype.card_bool, Nat.cast_mul,
    Nat.cast_ofNat, sq_abs] at hcs
  rw [complexCoefficients_sq] at hcs
  calc
    _ ≤ ((Fintype.card I : ℝ) * 2) * ∑ k, ‖complexInitializer U labels a k c‖^2 := hcs
    _ ≤ ((Fintype.card I : ℝ) * 2) * ((Fintype.card I : ℝ) * a^2) :=
      mul_le_mul_of_nonneg_left (complexInitializer_column_sq U hU labels a c) (by positivity)
    _ = _ := by ring

/-- The explicit initializer is feasible in CD8's jointly convex head
domain at radius `2|a||I|`, CFt5. -/
theorem complexInitializer_budget (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) (a : ℝ) :
    complexCoefficients (complexInitializer U labels a) ∈
      headBudget (2 * |a| * (Fintype.card I : ℝ)) := by
  intro c
  have hs := complexInitializer_l1_sq U hU labels a c
  have hn : (0 : ℝ) ≤ Fintype.card I := Nat.cast_nonneg _
  have ha := abs_nonneg a
  have hsq : |a|^2 = a^2 := sq_abs a
  have hnon : 0 ≤ 2 * |a| * (Fintype.card I : ℝ) := by positivity
  have hp : 0 ≤ (Fintype.card I : ℝ)^2 * a^2 := mul_nonneg (sq_nonneg _) (sq_nonneg _)
  nlinarith

/-- The native margin24 and radius64 times total states are a concrete
feasible instance, CFt5. The slack avoids a floating-point equality gate. -/
theorem complexInitializer_native_budget (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) :
    complexCoefficients (complexInitializer U labels 24) ∈
      headBudget (64 * (Fintype.card I : ℝ)) := by
  intro c
  have h := complexInitializer_budget U hU labels 24 c
  norm_num at h
  have hn : (0 : ℝ) ≤ Fintype.card I := Nat.cast_nonneg _
  linarith

example := unitary_star_mul_frobenius fourier4 fourier4_unitary
  (stateMargins (fun i : Fin 4 => decide (i.val < 2)) 24)

example := complexInitializer_frobenius fourier4 fourier4_unitary
  (fun i : Fin 4 => decide (i.val < 2)) 24

example := complexInitializer_column_sq fourier4 fourier4_unitary
  (fun i : Fin 4 => decide (i.val < 2)) 24 true

example := complexInitializer_l1_sq fourier4 fourier4_unitary
  (fun i : Fin 4 => decide (i.val < 2)) 24 true

example := complexInitializer_budget fourier4 fourier4_unitary
  (fun i : Fin 4 => decide (i.val < 2)) 24

example := complexInitializer_native_budget fourier4 fourier4_unitary
  (fun i : Fin 4 => decide (i.val < 2))

example : frobeniusSq (complexInitializer fourier4
    (fun i : Fin 4 => decide (i.val < 2)) 24) = 2304 := by
  rw [complexInitializer_frobenius _ fourier4_unitary]
  norm_num

example : complexCoefficients (complexInitializer fourier4
    (fun i : Fin 4 => decide (i.val < 2)) 24) ∈ headBudget 256 := by
  have h := complexInitializer_native_budget fourier4 fourier4_unitary
    (fun i : Fin 4 => decide (i.val < 2))
  norm_num at h
  exact h

end Transformer.Grokking.FlatSpectrum.ConvexDepth
