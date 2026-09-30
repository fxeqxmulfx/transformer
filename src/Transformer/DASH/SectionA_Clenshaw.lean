/-
# DASH — Clenshaw evaluation over rings

arXiv:2602.02016v2, Appendix A, `algorithm:cbshv-clenshaw-scalar`
and `algorithm:cbshv-clenshaw-matrix`. The generic ring includes
both real numbers and noncommutative square matrices.
-/

import Mathlib.Tactic

namespace Transformer.DASH

variable {R : Type*} [Ring R]

/-- First-kind Chebyshev recurrence, arXiv:2602.02016v2, Appendix A. -/
def chebT (x : R) : ℕ → R
  | 0 => 1
  | 1 => x
  | k + 2 => 2 * x * chebT x (k + 1) - chebT x k

/-- Shifted second-kind recurrence used to verify Clenshaw's backward loop.
Here `chebAux x (k+1)` is the second-kind polynomial of degree `k`.
Source: arXiv:2602.02016v2, Appendix A, the shared Chebyshev recurrence. -/
def chebAux (x : R) : ℕ → R
  | 0 => 0
  | 1 => 1
  | k + 2 => 2 * x * chebAux x (k + 1) - chebAux x k

/-- The source's finite Chebyshev expansion, beginning at a specified index.
Source: arXiv:2602.02016v2, Appendix A, `Σ c_k T_k(x)`.
Coefficients multiply on the right; scalar matrix coefficients commute. -/
def chebSeriesFrom (x : R) (k : ℕ) : List R → R
  | [] => 0
  | c :: cs => chebT x k * c + chebSeriesFrom x (k + 1) cs

/-- The auxiliary finite expansion for the backward-loop invariant,
arXiv:2602.02016v2, Appendix A, Clenshaw recurrence. -/
def chebAuxSeriesFrom (x : R) (k : ℕ) : List R → R
  | [] => 0
  | c :: cs => chebAux x k * c + chebAuxSeriesFrom x (k + 1) cs

/-- Exact backward recurrence, with coefficients ordered `c_0,...,c_d`.
Source: arXiv:2602.02016v2, Appendix A, both Clenshaw algorithms. -/
def clenshawState (x : R) : List R → R × R
  | [] => (0, 0)
  | c :: cs =>
      let b := clenshawState x cs
      (2 * x * b.1 - b.2 + c, b.1)

/-- The returned polynomial `b_0-x b_1`,
arXiv:2602.02016v2, Appendix A, `algorithm:cbshv-clenshaw-scalar`. -/
def clenshaw (x : R) (cs : List R) : R :=
  (clenshawState x cs).1 - x * (clenshawState x cs).2

/-- First-kind polynomials are the difference of adjacent auxiliary terms.
Source: arXiv:2602.02016v2, Appendix A, justification of Clenshaw evaluation. -/
theorem chebT_eq_aux (x : R) (k : ℕ) :
    chebT x k = chebAux x (k + 1) - x * chebAux x k := by
  induction k using Nat.twoStepInduction with
  | zero => simp [chebT, chebAux]
  | one => simp [chebT, chebAux]; noncomm_ring
  | more k ih ih' =>
    simp only [chebT, chebAux, ih, ih']
    noncomm_ring

/-- Finite auxiliary expansions satisfy the same second-order recurrence.
Source: arXiv:2602.02016v2, Appendix A, Clenshaw's backward recurrence. -/
theorem chebAuxSeries_recurrence (x : R) (cs : List R) (k : ℕ) :
    chebAuxSeriesFrom x (k + 2) cs =
      2 * x * chebAuxSeriesFrom x (k + 1) cs - chebAuxSeriesFrom x k cs := by
  induction cs generalizing k with
  | nil => simp [chebAuxSeriesFrom]
  | cons c cs ih =>
    simp only [chebAuxSeriesFrom, chebAux, ih]
    noncomm_ring

/-- Exact closed form of the backward-loop state.
Source: arXiv:2602.02016v2, Appendix A, `b_k=2x b_(k+1)-b_(k+2)+c_k`. -/
theorem clenshawState_eq_aux (x : R) (cs : List R) :
    clenshawState x cs = (chebAuxSeriesFrom x 1 cs, chebAuxSeriesFrom x 0 cs) := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [clenshawState, ih, chebAuxSeriesFrom, chebAux, one_mul, zero_mul, zero_add]
    rw [chebAuxSeries_recurrence x cs 0]
    congr 1
    noncomm_ring

/-- The finite first-kind series has the corresponding auxiliary expression.
Source: arXiv:2602.02016v2, Appendix A, the finite Chebyshev expansion. -/
theorem chebSeries_eq_aux (x : R) (cs : List R) (k : ℕ) :
    chebSeriesFrom x k cs =
      chebAuxSeriesFrom x (k + 1) cs - x * chebAuxSeriesFrom x k cs := by
  induction cs generalizing k with
  | nil => simp [chebSeriesFrom, chebAuxSeriesFrom]
  | cons c cs ih =>
    simp only [chebSeriesFrom, chebAuxSeriesFrom, chebT_eq_aux, ih]
    noncomm_ring

/-- Clenshaw evaluates exactly the stated polynomial, for scalars and
matrices and for every finite coefficient list. Approximation of an inverse
root is a separate property of the chosen coefficients.
Source: arXiv:2602.02016v2, Appendix A, the scalar and matrix algorithms. -/
theorem clenshaw_eq_series (x : R) (cs : List R) :
    clenshaw x cs = chebSeriesFrom x 0 cs := by
  simp only [clenshaw, clenshawState_eq_aux, chebSeries_eq_aux]

/-- The final optimized formula avoids constructing `b_0`. The prose
immediately before the source's displayed calculation has `-B_1` where
the recurrence and calculation correctly use `-B_2`.
Source: arXiv:2602.02016v2, Appendix A, optimized Clenshaw algorithm. -/
theorem clenshaw_optimized_final (x c : R) (cs : List R) :
    clenshaw x (c :: cs) = x * (clenshawState x cs).1 - (clenshawState x cs).2 + c := by
  simp only [clenshaw, clenshawState]
  noncomm_ring

/-- The two leading backward steps have the stated closed forms, avoiding
zero and identity matrix multiplications.
Source: arXiv:2602.02016v2, Appendix A, initialization of optimized Clenshaw. -/
theorem clenshaw_two_coefficients (x c₁ c₂ : R) :
    clenshawState x [c₁, c₂] = (2 * x * c₂ + c₁, c₂) := by
  simp [clenshawState]

end Transformer.DASH
