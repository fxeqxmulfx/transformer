/-
# Sequence-valued targets

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, Appendix A.7,
`theo:attn_matrix_convex`, equations (30)–(33).

The paper does not define its mixed matrix norm `‖·‖_{1,∞}`.  The convex
objective therefore takes that norm as an explicit argument.  The concrete
counterexample below works for every such norm giving value at most one to
each displayed coordinate matrix, including the usual mixed norms.
-/

import Transformer.Convexifying.Section3_FCN

open scoped BigOperators

namespace Transformer.Convexifying

/-- Parameters of the sequence-output model in Appendix A.7. -/
structure MatrixParameters (h n d c : ℕ) where
  attention : Fin h → Mat n n
  value : Fin h → Mat d c
  output : Fin h → ℝ

/-- Feasibility of the matrix attention weights. -/
def MatrixParameters.Feasible {h n d c : ℕ} (p : MatrixParameters h n d c) : Prop :=
  ∀ j, IsRowStochastic (p.attention j)

/-- Entrywise matrix ℓ₁ norm of the value weight.  This choice has the
entrywise ℓ∞ dual used in the appendix's displayed dual constraint. -/
def matrixNorm₁ {d c : ℕ} (B : Mat d c) : ℝ :=
  ∑ q, ∑ l, |B q l|

/-- Sequence-output prediction from equation (30). -/
def matrixPrediction {h n d c : ℕ} (X : Fin n → Vec d)
    (p : MatrixParameters h n d c) : Mat n c :=
  fun r l => ∑ j,
    ((∑ k, p.attention j r k * (∑ q, X k q * p.value j q l)) *
      p.output j)

/-- The original matrix-target objective in equation (30). -/
noncomputable def matrixObjective {N h n d c : ℕ} (X : Data N n d)
    (y : Fin N → Mat n c) (L : Mat n c → Mat n c → ℝ) (β : ℝ)
    (p : MatrixParameters h n d c) : ℝ :=
  (∑ i, L (matrixPrediction (X i) p) (y i)) +
    β / 2 * ∑ j, ((matrixNorm₁ (p.value j)) ^ 2 + (p.output j) ^ 2)

/-- The convex sequence-output prediction in equation (33). -/
def matrixConvexPrediction {n d c : ℕ} (X : Fin n → Vec d)
    (Cpos Cneg : Fin d → Fin c → Mat n n) : Mat n c :=
  fun r l => ∑ q, ∑ k,
    (Cpos q l r k - Cneg q l r k) * X k q

/-- The convex objective in equation (33), with its unspecified mixed
matrix norm supplied explicitly as `mixedNorm`. -/
noncomputable def matrixConvexObjective {N n d c : ℕ} (X : Data N n d)
    (y : Fin N → Mat n c) (L : Mat n c → Mat n c → ℝ) (β : ℝ)
    (mixedNorm : Mat n n → ℝ) (Cpos Cneg : Fin d → Fin c → Mat n n) : ℝ :=
  (∑ i, L (matrixConvexPrediction (X i) Cpos Cneg) (y i)) +
    β * ∑ l, ∑ q, (mixedNorm (Cpos q l) + mixedNorm (Cneg q l))

/-- Nonnegative matrix constraint of equation (33). -/
def NonnegativeMatrices {n d c : ℕ} (C : Fin d → Fin c → Mat n n) : Prop :=
  ∀ q l r k, 0 ≤ C q l r k

/-- Squared Frobenius loss for sequence targets. -/
def matrixSquareLoss {n c : ℕ} (prediction target : Mat n c) : ℝ :=
  ∑ r, ∑ l, (prediction r l - target r l) ^ 2

/-- Two sequence targets with opposite signs in the first output row. -/
def separatingMatrixTargets : Fin 2 → Mat 2 1 :=
  fun i r _ => if r = 0 then separatingTargets i else 0

/-- Positive and negative convex matrices fitting the two targets. -/
def matrixPositiveWitness : Fin 1 → Fin 1 → Mat 2 2 :=
  fun _ _ r k => if r = 0 ∧ k = 0 then 1 else 0

def matrixNegativeWitness : Fin 1 → Fin 1 → Mat 2 2 :=
  fun _ _ r k => if r = 0 ∧ k = 1 then 1 else 0

/-- The witness matrices satisfy the nonnegativity constraint. -/
theorem matrixWitness_nonnegative :
    NonnegativeMatrices matrixPositiveWitness ∧
      NonnegativeMatrices matrixNegativeWitness := by
  constructor
  · intro q l r k
    by_cases h : r = 0 ∧ k = 0 <;> simp [matrixPositiveWitness, h]
  · intro q l r k
    by_cases h : r = 0 ∧ k = 1 <;> simp [matrixNegativeWitness, h]

/-- Both positive and negative witnesses fit the sequence targets. -/
theorem matrixWitness_fit (i : Fin 2) :
    matrixConvexPrediction (separatingData i)
      matrixPositiveWitness matrixNegativeWitness = separatingMatrixTargets i := by
  funext r l
  fin_cases i <;> fin_cases r <;> fin_cases l <;>
    norm_num [matrixConvexPrediction, separatingData, separatingMatrixTargets,
      separatingTargets, matrixPositiveWitness, matrixNegativeWitness,
      Fin.sum_univ_two, Fin.sum_univ_one]

/-- A one-head sequence model cannot give opposite signs to the active
output row on the two nonnegative input matrices. -/
theorem oneHead_matrix_lower_bound (p : MatrixParameters 1 2 1 1)
    (hp : p.Feasible) :
    1 ≤ matrixObjective separatingData separatingMatrixTargets
      matrixSquareLoss (1 / 8) p := by
  have ha0 : 0 ≤ p.attention 0 0 0 := ((hp 0) 0).1 0
  have ha1 : 0 ≤ p.attention 0 0 1 := ((hp 0) 0).1 1
  let t : ℝ := p.value 0 0 0 * p.output 0
  have hsign : 1 ≤ (p.attention 0 0 0 * t - 1) ^ 2 +
      (p.attention 0 0 1 * t + 1) ^ 2 := by
    rcases le_total 0 t with ht | ht
    · have hn : 0 ≤ p.attention 0 0 1 * t := mul_nonneg ha1 ht
      nlinarith [sq_nonneg (p.attention 0 0 0 * t - 1)]
    · have hn : p.attention 0 0 0 * t ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos ha0 ht
      nlinarith [sq_nonneg (p.attention 0 0 1 * t + 1)]
  simp [matrixObjective, matrixSquareLoss, matrixPrediction, matrixNorm₁,
    separatingData, separatingMatrixTargets, separatingTargets,
    Fin.sum_univ_two] at *
  nlinarith [sq_nonneg (p.attention 0 1 0 * p.value 0 0 0 * p.output 0),
    sq_nonneg (p.attention 0 1 1 * p.value 0 0 0 * p.output 0)]

/-- The one-head matrix-model feasibility hypothesis is satisfiable. -/
example :
    (⟨fun _ _ => fun k : Fin 2 => if k = 0 then 1 else 0,
      fun _ _ _ => 0, fun _ => 0⟩ : MatrixParameters 1 2 1 1).Feasible := by
  intro _ _
  exact simplex_basis 0

/-- Under the displayed bound on the norm of the two coordinate matrices,
the convex witness has objective at most `1/4`. -/
theorem matrixWitness_convex_upper_bound (mixedNorm : Mat 2 2 → ℝ)
    (hpos : mixedNorm (matrixPositiveWitness 0 0) ≤ 1)
    (hneg : mixedNorm (matrixNegativeWitness 0 0) ≤ 1) :
    matrixConvexObjective separatingData separatingMatrixTargets
      matrixSquareLoss (1 / 8) mixedNorm
      matrixPositiveWitness matrixNegativeWitness ≤ 1 / 4 := by
  simp [matrixConvexObjective, matrixSquareLoss, matrixWitness_fit]
  linarith

/-- **Counterexample to the appendix matrix-target theorem at one head.**
Its claimed unrestricted convex formulation has a strictly smaller sublevel
set threshold than the one-head sequence model.  The only property needed of
the paper's undefined mixed norm is that each coordinate matrix has norm at
most one.
Source: arXiv:2211.11052v1, Appendix A.7, `theo:attn_matrix_convex`. -/
theorem matrix_equivalence_false_for_one_head (mixedNorm : Mat 2 2 → ℝ)
    (hpos : mixedNorm (matrixPositiveWitness 0 0) ≤ 1)
    (hneg : mixedNorm (matrixNegativeWitness 0 0) ≤ 1) :
    ¬ ∀ r : ℝ,
      (∃ p : MatrixParameters 1 2 1 1,
        p.Feasible ∧
          matrixObjective separatingData separatingMatrixTargets
            matrixSquareLoss (1 / 8) p ≤ r) ↔
      (∃ Cpos Cneg : Fin 1 → Fin 1 → Mat 2 2,
        NonnegativeMatrices Cpos ∧ NonnegativeMatrices Cneg ∧
          matrixConvexObjective separatingData separatingMatrixTargets
            matrixSquareLoss (1 / 8) mixedNorm Cpos Cneg ≤ r) := by
  intro h
  obtain ⟨p, hp, hbound⟩ := (h (1 / 2)).2
    ⟨matrixPositiveWitness, matrixNegativeWitness,
      matrixWitness_nonnegative.1, matrixWitness_nonnegative.2,
      (matrixWitness_convex_upper_bound mixedNorm hpos hneg).trans (by norm_num)⟩
  have hlower := oneHead_matrix_lower_bound p hp
  linarith

/-- A concrete matrix norm witnessing that the hypotheses of the
counterexample are satisfiable. -/
def entrywiseMatrixNorm₁ (C : Mat 2 2) : ℝ :=
  ∑ r, ∑ k, |C r k|

/-- Each coordinate matrix has unit entrywise ℓ₁ norm. -/
theorem matrixWitness_entrywise_norms :
    entrywiseMatrixNorm₁ (matrixPositiveWitness 0 0) = 1 ∧
      entrywiseMatrixNorm₁ (matrixNegativeWitness 0 0) = 1 := by
  constructor <;>
    norm_num [entrywiseMatrixNorm₁, matrixPositiveWitness,
      matrixNegativeWitness, Fin.sum_univ_two]

/-- The matrix theorem's norm hypotheses are jointly satisfiable. -/
example : ∃ mixedNorm : Mat 2 2 → ℝ,
    mixedNorm (matrixPositiveWitness 0 0) ≤ 1 ∧
      mixedNorm (matrixNegativeWitness 0 0) ≤ 1 := by
  exact ⟨entrywiseMatrixNorm₁,
    matrixWitness_entrywise_norms.1.le, matrixWitness_entrywise_norms.2.le⟩

end Transformer.Convexifying
