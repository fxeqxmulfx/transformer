/-
# Convexifying Transformers: finite-dimensional models

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.1–§3.2.
The index types `Fin N`, `Fin n`, `Fin d`, and `Fin h` represent samples,
tokens, embedding coordinates, and attention heads.  A scalar attention
weight is a vector in the unit simplex, as required by the scalar model in
equation (8); the matrix simplex introduced in §3.1 is used for sequence
outputs instead.
-/

import Mathlib

open scoped BigOperators

namespace Transformer.Convexifying

/-- A vector of `d` real coordinates. -/
abbrev Vec (d : ℕ) := Fin d → ℝ

/-- A real matrix represented by its row and column indices. -/
abbrev Mat (m n : ℕ) := Fin m → Fin n → ℝ

/-- A training set of `N` token matrices with `n` tokens and dimension `d`. -/
abbrev Data (N n d : ℕ) := Fin N → Fin n → Vec d

/-- The squared Euclidean norm used in §3.2. -/
def normSq {d : ℕ} (v : Vec d) : ℝ := ∑ q, (v q) ^ 2

/-- The Euclidean norm used in the group regularizer of equation (9). -/
noncomputable def norm₂ {d : ℕ} (v : Vec d) : ℝ := Real.sqrt (normSq v)

/-- Scalar attention weights lie in the unit simplex of §3.1. -/
def IsSimplex {n : ℕ} (a : Vec n) : Prop :=
  (∀ k, 0 ≤ a k) ∧ ∑ k, a k = 1

/-- The row-stochastic matrix simplex corresponding to rowwise softmax.
The notation in §3.1 writes `1ᵀ W_i = 1` for each `i`, which would make
columns stochastic if `W_i` denotes a column.  The row interpretation is
forced by the stated softmax identity. -/
def IsRowStochastic {n : ℕ} (A : Mat n n) : Prop :=
  ∀ r, IsSimplex (A r)

/-- The scalar-output parameters of equation (8). -/
structure ScalarParameters (h n d : ℕ) where
  attention : Fin h → Vec n
  value : Fin h → Vec d
  output : Fin h → ℝ

/-- Feasibility of the attention weights in equation (8). -/
def ScalarParameters.Feasible {h n d : ℕ} (p : ScalarParameters h n d) : Prop :=
  ∀ j, IsSimplex (p.attention j)

/-- Prediction of one scalar attention head in equation (8). -/
def scalarHead {n d : ℕ} (X : Fin n → Vec d)
    (a : Vec n) (v : Vec d) (b : ℝ) : ℝ :=
  (∑ k, a k * (∑ q, X k q * v q)) * b

/-- Multihead scalar prediction in equation (8). -/
def scalarPrediction {h n d : ℕ} (X : Fin n → Vec d)
    (p : ScalarParameters h n d) : ℝ :=
  ∑ j, scalarHead X (p.attention j) (p.value j) (p.output j)

/-- The original scalar objective, equation (8), with arbitrary loss `L`. -/
noncomputable def scalarObjective {N h n d : ℕ} (X : Data N n d) (y : Vec N)
    (L : ℝ → ℝ → ℝ) (β : ℝ) (p : ScalarParameters h n d) : ℝ :=
  (∑ i, L (scalarPrediction (X i) p) (y i)) +
    β / 2 * ∑ j, (normSq (p.value j) + (p.output j) ^ 2)

/-- Prediction of the unrestricted matrix `Z` in equation (9). -/
def convexPrediction {n d : ℕ} (X : Fin n → Vec d)
    (Z : Fin n → Vec d) : ℝ :=
  ∑ k, ∑ q, Z k q * X k q

/-- The convex objective *as printed* in equation (9).  Its loss has an
additional factor `1/2` absent from equation (8). -/
noncomputable def printedConvexObjective {N n d : ℕ} (X : Data N n d) (y : Vec N)
    (L : ℝ → ℝ → ℝ) (β : ℝ) (Z : Fin n → Vec d) : ℝ :=
  (1 / 2 : ℝ) * (∑ i, L (convexPrediction (X i) Z) (y i)) +
    β * ∑ k, norm₂ (Z k)

/-- The convex objective after removing the loss-factor discrepancy in
equation (9), matching the last display in its appendix proof. -/
noncomputable def correctedConvexObjective {N n d : ℕ} (X : Data N n d) (y : Vec N)
    (L : ℝ → ℝ → ℝ) (β : ℝ) (Z : Fin n → Vec d) : ℝ :=
  (∑ i, L (convexPrediction (X i) Z) (y i)) +
    β * ∑ k, norm₂ (Z k)

/-- Nonnegative squared Euclidean norm, as used in all regularizers of §3. -/
theorem normSq_nonneg {d : ℕ} (v : Vec d) : 0 ≤ normSq v := by
  unfold normSq
  exact Finset.sum_nonneg fun q _ => sq_nonneg (v q)

/-- The Euclidean norm is nonnegative. -/
theorem norm₂_nonneg {d : ℕ} (v : Vec d) : 0 ≤ norm₂ v :=
  Real.sqrt_nonneg _

/-- Squaring the Euclidean norm gives the sum of coordinate squares. -/
theorem norm₂_sq {d : ℕ} (v : Vec d) : (norm₂ v) ^ 2 = normSq v := by
  exact Real.sq_sqrt (normSq_nonneg v)

/-- The unit simplex is nonempty whenever the token set is nonempty.
Source: arXiv:2211.11052v1, §3.1. -/
theorem simplex_basis {n : ℕ} (k : Fin n) :
    IsSimplex (fun j => if j = k then 1 else 0) := by
  constructor
  · intro j
    by_cases h : j = k <;> simp [h]
  · simp

end Transformer.Convexifying
