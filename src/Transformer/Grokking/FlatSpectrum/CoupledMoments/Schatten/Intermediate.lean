module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Hull
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Vertices
public import Transformer.Grokking.FlatSpectrum.PartialIsometry

/-!
# The two endpoint budgets miss an intermediate spectral shape

Source: `review/flat_spectrum.md`, CM9. A duplicated 2-by-2 sign matrix has size four,
rank two, operator norm `sqrt 8`, and nuclear norm `2 sqrt 8`. It is an actual moment
of a one-hot probability table. Neither old endpoint budget admits it, while a fixed
intermediate budget does. This is a capacity witness, not a missing-label theorem or
a convergence claim about any native optimizer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Matrix Finset
open scoped Matrix.Norms.L2Operator

/-- A nonzero matrix whose Gram square is `c` times its Gram has operator norm
`sqrt c` (CM9). No trained factors or flatness equality enter the model domain. -/
theorem operator_eq_sqrt_of_gram_square {m n : Type*} [Fintype m] [Fintype n]
    [DecidableEq n] {X : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (hX : X ≠ 0)
    (h : Xᴴ * X * (Xᴴ * X) = (c : ℂ) • (Xᴴ * X)) : ‖X‖ = √c := by
  have hn := l2_opNorm_conjTranspose_mul_self X
  have hg := l2_opNorm_conjTranspose_mul_self (Xᴴ * X)
  rw [conjTranspose_mul, conjTranspose_conjTranspose, h, norm_smul,
    Complex.norm_real, Real.norm_of_nonneg hc.le] at hg
  have hx : 0 < ‖X‖ := norm_pos_iff.mpr hX
  have hp : 0 < ‖Xᴴ * X‖ := by rw [hn]; positivity
  have he : ‖Xᴴ * X‖ = c := by nlinarith
  rw [he] at hn
  nlinarith [Real.sqrt_nonneg c, Real.sq_sqrt hc.le]

example := operator_eq_sqrt_of_gram_square (X := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (c := 1) (by norm_num) one_ne_zero (by simp)

/-- A duplicated Hadamard sign table, with no arithmetic operation supplied (CM9). -/
def intermediateMatrix : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 1, 1, 1; 1, 1, 1, 1; 1, 1, -1, -1; 1, 1, -1, -1]

/-- Binary labels represented by the sign table (CM9). -/
def intermediateLabel (i j : Fin 4) : Bool := !(2 ≤ i.val && 2 ≤ j.val)

/-- One-hot class coordinates for the intermediate witness (CM9). -/
def intermediateTable : Table (Fin 4) (Fin 4) Bool := pointMass intermediateLabel

/-- The fixed centered binary code used for this capacity witness (CM9). -/
def intermediateCode : Fin 1 → Bool → ℂ := ![fun c => if c then 1 else -1]

/-- The probability table produces the actual displayed matrix, not independent moments
(CM9). -/
theorem intermediate_moment : moment (intermediateCode 0) intermediateTable =
    intermediateMatrix := by
  rw [intermediateTable, moment_pointMass]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [intermediateCode, intermediateLabel, intermediateMatrix]

/-- The Gram matrix has nonzero eigenvalue eight (CM9). -/
theorem intermediate_gram_square :
    intermediateMatrixᴴ * intermediateMatrix * (intermediateMatrixᴴ * intermediateMatrix) =
      (8 : ℂ) • (intermediateMatrixᴴ * intermediateMatrix) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [intermediateMatrix, Matrix.mul_apply, Fin.sum_univ_succ]

/-- The four-by-four sign table has Frobenius energy sixteen (CM9). -/
theorem intermediate_energy : frobeniusSq intermediateMatrix = 16 := by
  norm_num [frobeniusSq, intermediateMatrix, Fin.sum_univ_succ]

/-- Its actual rank is two, strictly between the two endpoint shapes (CM9). -/
theorem intermediate_rank : intermediateMatrix.rank = 2 := by
  have h := frobeniusSq_eq_rank_mul intermediate_gram_square
  rw [intermediate_energy] at h
  norm_num at h
  have hr : (intermediateMatrix.rank : ℝ) = 2 := by nlinarith
  exact_mod_cast hr

/-- The largest singular value is `sqrt 8`, above the full-rank budget two (CM9). -/
theorem intermediate_operator : ‖intermediateMatrix‖ = √8 := by
  apply operator_eq_sqrt_of_gram_square (by norm_num) ?_ intermediate_gram_square
  intro h
  have he := congrFun (congrFun h 0) 0
  norm_num [intermediateMatrix] at he

/-- The singular-value sum is `2 sqrt 8`, above the rank-one budget four (CM9). -/
theorem intermediate_nuclear : nuclearNorm intermediateMatrix = 2 * √8 := by
  rw [nuclearNorm_eq_rank_mul_sqrt intermediate_gram_square, intermediate_rank]
  norm_num

/-- Neither old endpoint budget contains this probability table (CM9). This is a
capacity obstruction even if an optimizer solves each old problem exactly. -/
theorem intermediate_excluded :
    ¬ BudgetFeasible intermediateCode (fun _ => 2) (fun _ => 8) intermediateTable ∧
    ¬ BudgetFeasible intermediateCode (fun _ => 4) (fun _ => 4) intermediateTable := by
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ 8 by norm_num)
  have hp := Real.sqrt_nonneg (8 : ℝ)
  constructor
  · intro h
    have ho := h.1.2 0
    rw [intermediate_moment, intermediate_operator] at ho
    nlinarith
  · intro h
    have hn := h.2 0
    rw [intermediate_moment, intermediate_nuclear] at hn
    nlinarith

/-- Fixed intermediate upper bounds admit the witness and remain the convex CM5 domain.
Admissibility alone does not identify unobserved labels (CM9). -/
theorem intermediate_admitted :
    BudgetFeasible intermediateCode (fun _ => √8) (fun _ => 2 * √8) intermediateTable := by
  refine ⟨⟨rows_pointMass _, ?_⟩, ?_⟩
  · intro f
    fin_cases f
    change ‖moment (intermediateCode 0) intermediateTable‖ ≤ √8
    rw [intermediate_moment, intermediate_operator]
  · intro f
    fin_cases f
    change nuclearNorm (moment (intermediateCode 0) intermediateTable) ≤ 2 * √8
    rw [intermediate_moment, intermediate_nuclear]

/-- Convexifying the two old spectral families still excludes the deterministic witness
(CM9). This concerns the spectral hull; an unrestricted lookup expert can represent it. -/
theorem intermediate_excluded_hull :
    intermediateTable ∉ Hull intermediateCode
      (fun k : Bool => fun _ => if k then 4 else 2)
      (fun k : Bool => fun _ => if k then 4 else 8) := by
  intro h
  have hm := (pointMass_mem_convexHull_iff intermediateLabel _
    (fun P hP => by
      obtain ⟨k, hk⟩ := Set.mem_iUnion.mp hP
      exact hk.1.1)).mp h
  obtain ⟨k, hk⟩ := Set.mem_iUnion.mp hm
  cases k
  · exact intermediate_excluded.1 hk
  · exact intermediate_excluded.2 hk

example := budget_fit_minimum _ _ _
  (fun p : Fin 4 × Fin 4 => p) (fun p => intermediateLabel p.1 p.2)
  intermediate_admitted (by
    intro p
    funext c
    simp [intermediateTable, pointMass, ConvexRecall.targetLogits])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
