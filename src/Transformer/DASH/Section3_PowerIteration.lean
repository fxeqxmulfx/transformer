/-
# DASH — Power Iteration, pooling, and the unsafe factor-two claim

arXiv:2602.02016v2, §3.4–3.5. A pool improves the best observed
Rayleigh quotient but does not turn it into a certified upper bound.
-/

import Transformer.DASH.Section3_Rayleigh
import Mathlib.Data.Finset.Lattice.Fold

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Normalized matrix-vector Power Iteration, arXiv:2602.02016v2, §3.4–3.5.
The zero result uses Lean's total inverse and stays zero. -/
def powerStep (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  let y := A *ᵥ x
  (Real.sqrt (dotProduct y y))⁻¹ • y

/-- The iterates from a specified starting vector,
arXiv:2602.02016v2, §3.5, the pool of starting vectors. -/
def powerIterate (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : ℕ → Fin n → ℝ
  | 0 => x
  | k + 1 => powerStep A (powerIterate A x k)

/-- The estimate selected from a nonempty pool, arXiv:2602.02016v2, §3.5. -/
def pooledRayleigh {ι : Type} [Fintype ι] [Nonempty ι]
    (A : Matrix (Fin n) (Fin n) ℝ) (starts : ι → Fin n → ℝ) (k : ℕ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun j => rayleigh A (powerIterate A (starts j) k))

/-- Pooling never reduces the estimate relative to a member of the pool.
Source: arXiv:2602.02016v2, §3.5, “choose ... the largest Rayleigh quotient”. -/
theorem pooledRayleigh_ge_member {ι : Type} [Fintype ι] [Nonempty ι]
    (A : Matrix (Fin n) (Fin n) ℝ) (starts : ι → Fin n → ℝ) (k : ℕ) (j : ι) :
    rayleigh A (powerIterate A (starts j) k) ≤ pooledRayleigh A starts k :=
  Finset.le_sup' (fun t => rayleigh A (powerIterate A (starts t) k)) (Finset.mem_univ j)

/-- Any upper bound valid for all candidates remains valid for the pool.
Source: arXiv:2602.02016v2, §3.5. -/
theorem pooledRayleigh_le {ι : Type} [Fintype ι] [Nonempty ι]
    (A : Matrix (Fin n) (Fin n) ℝ) (starts : ι → Fin n → ℝ) (k : ℕ) (μ : ℝ)
    (hbound : ∀ j, rayleigh A (powerIterate A (starts j) k) ≤ μ) :
    pooledRayleigh A starts k ≤ μ :=
  Finset.sup'_le Finset.univ_nonempty _ (fun j _ => hbound j)

/-- Pool-bound hypotheses are satisfiable, arXiv:2602.02016v2, §3.5. -/
example : ∀ j : Fin 1, rayleigh (1 : Matrix (Fin 1) (Fin 1) ℝ)
    (powerIterate 1 ((fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) j) 0) ≤ 1 := by
  norm_num [rayleigh, powerIterate, dotProduct]

/-- The missing certificate for safe factor-two scaling is an actual
upper bound `μ < 2r`, not merely `r ≤ μ` from the Rayleigh quotient.
Source: arXiv:2602.02016v2, §3.4, final paragraph. -/
theorem factor_two_scaling_safe (μ r : ℝ) (hr : 0 < r) (hbound : μ < 2 * r) :
    μ / (2 * r) < 1 := (div_lt_one (by positivity)).mpr hbound

/-- Safe-scaling hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 * 1 := by norm_num

/-- A positive-definite matrix whose largest eigenvalue is 100,
arXiv:2602.02016v2, §3.4–3.5, the counterexample to guaranteed PI scaling. -/
def unsafePowerMatrix : Matrix (Fin 2) (Fin 2) ℝ := !![100, 0; 0, 1]

/-- A unit starting vector orthogonal to the maximal eigenspace,
arXiv:2602.02016v2, §3.4–3.5. -/
def unsafePowerStart : Fin 2 → ℝ := ![0, 1]

/-- The initial vector remains at the smaller eigenvector for every number
of iterations, not just for a prematurely terminated run.
Source: arXiv:2602.02016v2, §3.5, convergence to a smaller eigenvector. -/
theorem powerIteration_stuck (k : ℕ) :
    powerIterate unsafePowerMatrix unsafePowerStart k = unsafePowerStart := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [powerIterate, ih]
    norm_num [powerStep, unsafePowerMatrix, unsafePowerStart, Matrix.mulVec,
      dotProduct, Fin.sum_univ_two]
    ext i
    fin_cases i <;> rfl

/-- The unsafe-scaling counterexample is positive definite, so the failure
does not come from a singular or indefinite input.
Source: arXiv:2602.02016v2, §3.4, the claimed PI scaling guarantee. -/
theorem unsafePowerMatrix_posDef : unsafePowerMatrix.PosDef := by
  have heq : unsafePowerMatrix = Matrix.diagonal (![100, 1] : Fin 2 → ℝ) := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [unsafePowerMatrix, Matrix.diagonal]
  rw [heq]
  apply Matrix.PosDef.diagonal
  intro i
  fin_cases i <;> norm_num

/-- Dividing by twice the PI estimate leaves an eigenvalue 50, outside both
the discussed interval `(0,1)` and the NDB interval `(0,2)`. This refutes
the source's assertion that the factor two “makes sure” convergence holds.
Source: arXiv:2602.02016v2, §3.4, final paragraph. -/
theorem factor_two_scaling_counterexample (k : ℕ) :
    rayleigh unsafePowerMatrix (powerIterate unsafePowerMatrix unsafePowerStart k) = 1 ∧
    ((2 * rayleigh unsafePowerMatrix
      (powerIterate unsafePowerMatrix unsafePowerStart k))⁻¹ • unsafePowerMatrix) *ᵥ ![1, 0] =
      (50 : ℝ) • (![1, 0] : Fin 2 → ℝ) ∧ (2 : ℝ) < 50 := by
  rw [powerIteration_stuck]
  norm_num [rayleigh, unsafePowerMatrix, unsafePowerStart, Matrix.mulVec,
    dotProduct, Fin.sum_univ_two]
  ext i
  fin_cases i <;> rfl

/-- Even pools of 16 or 32 starts can all miss the maximal eigenspace.
This does not refute the paper's probabilistic motivation; it refutes a
deterministic guarantee without assumptions on the starts.
Source: arXiv:2602.02016v2, §3.5. -/
theorem pooled_power_counterexample {ι : Type} [Fintype ι] [Nonempty ι] (k : ℕ) :
    pooledRayleigh unsafePowerMatrix (fun _ : ι => unsafePowerStart) k = 1 := by
  simp only [pooledRayleigh, powerIteration_stuck]
  have h : rayleigh unsafePowerMatrix unsafePowerStart = 1 := by
    norm_num [rayleigh, unsafePowerMatrix, unsafePowerStart, Matrix.mulVec,
      dotProduct, Fin.sum_univ_two]
  simp only [h, Finset.sup'_const]

end Transformer.DASH
