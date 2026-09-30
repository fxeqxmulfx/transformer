/-
# AdaFisher: a nondegenerate block-Kronecker nonconvex counterexample

arXiv:2405.16397v3, Proposition 3.4 and Appendix A.2.
The four-dimensional witness realizes a nonconstant normalized pair of KFs,
avoiding the constant-factor domain issue. The objective is 0.001 sin(x₀).
-/

import Transformer.AdaFisher.Section3_NonconvexFalse
import Transformer.AdaFisher.SectionA_StochasticTerms
import Transformer.AdaFisher.Section3_InverseError
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.Calculus.Deriv.Comp

open scoped BigOperators
open Filter Topology MeasureTheory

noncomputable section

namespace Transformer.AdaFisher

/-- Column-first parameter indexing of a 2×2 layer block, §2. -/
def witnessCoordinate (i : Fin 4) : Fin 2 × Fin 2 :=
  (⟨i.val / 2, by omega⟩, ⟨i.val % 2, by omega⟩)

/-- A genuine nonconstant-factor efficient Fisher block, Proposition 3.2,
with normalized factors `(0,1)` and `(0,1)` and damping 0.001. -/
def witnessFisher (i : Fin 4) : ℝ :=
  fisherDiagonal (1 / 1000) (fun j : Fin 2 => (j : ℝ))
    (fun j : Fin 2 => (j : ℝ)) (witnessCoordinate i)

/-- The witness is exactly the min-max-normalized block of positive
nonconstant factors `(1,2)`, Proposition 3.2. Thus the printed normalization
formula is defined without the constant-factor extension. -/
theorem witnessFisher_normalized (i : Fin 4) :
    witnessFisher i = fisherDiagonal (1 / 1000)
      (minMaxDiagonal (fun j : Fin 2 => 1 + (j : ℝ)))
      (minMaxDiagonal (fun j : Fin 2 => 1 + (j : ℝ))) (witnessCoordinate i) := by
  rw [inverseError_normalized_factor]
  rfl

/-- Smooth bounded objective on all four layer parameters, Proposition 3.4. -/
def witnessLoss (x : Fin 4 → ℝ) : ℝ := sineLoss (x 0)

/-- Euclidean gradient of the witness objective, Proposition 3.4. -/
def witnessGradient (x : Fin 4 → ℝ) (i : Fin 4) : ℝ :=
  if i = 0 then (1 / 1000) * Real.cos (x 0) else 0

/-- The derivative is the claimed gradient functional, Proposition 3.4.
This verifies differentiability of the actual objective, not a supplied
gradient stream unrelated to the loss. -/
theorem witnessLoss_hasFDerivAt (x : Fin 4 → ℝ) :
    HasFDerivAt witnessLoss (((1 / 1000 : ℝ) * Real.cos (x 0)) •
      ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 4 => ℝ) 0) x := by
  exact (sineLoss_hasDerivAt (x 0)).comp_hasFDerivAt x (hasFDerivAt_apply 0 x)

/-- The derivative functional pairs with the Euclidean gradient,
Proposition 3.4, assumption (i). -/
theorem witnessGradient_pairing (x u : Fin 4 → ℝ) :
    pairing (witnessGradient x) u = (1 / 1000 : ℝ) * Real.cos (x 0) * u 0 := by
  simp [pairing, witnessGradient]

/-- The gradient bound of Proposition 3.4 holds everywhere, with λ=0.001. -/
theorem witnessGradient_bound (x : Fin 4 → ℝ) :
    squaredNorm (witnessGradient x) ≤ (1 / 1000 : ℝ) ^ 2 := by
  have hs : ((1 / 1000 : ℝ) * Real.cos (x 0)) ^ 2 ≤ (1 / 1000 : ℝ) ^ 2 := by
    have hx := sineLoss_assumptions.2.1 (x 0)
    nlinarith [sq_abs ((1 / 1000 : ℝ) * Real.cos (x 0)),
      abs_nonneg ((1 / 1000 : ℝ) * Real.cos (x 0))]
  simpa [squaredNorm, witnessGradient] using hs

/-- Lower boundedness and the actual diagonal Fisher bound in
Proposition 3.4, assumption (i), with L=2 and damping λ=0.001. -/
theorem witness_bounds :
    (∀ x : Fin 4 → ℝ, -(1 / 1000 : ℝ) ≤ witnessLoss x) ∧
    (∀ i, (1 / 1000 : ℝ) ≤ witnessFisher i ∧ witnessFisher i < 2) := by
  constructor
  · intro x
    exact sineLoss_assumptions.1 (x 0)
  · intro i
    fin_cases i <;> norm_num [witnessFisher, witnessCoordinate, fisherDiagonal]

/-- The true gradient is L=2 Lipschitz in the paper's Euclidean norm,
Proposition 3.4. Written equivalently as the inequality between squares. -/
theorem witnessGradient_lipschitz_squared (x y : Fin 4 → ℝ) :
    squaredNorm (witnessGradient x - witnessGradient y) ≤ 2 ^ 2 * squaredNorm (x - y) := by
  have h := sineLoss_assumptions.2.2 (x 0) (y 0)
  have hterm : (x 0 - y 0) ^ 2 ≤ squaredNorm (x - y) := by
    exact Finset.single_le_sum (fun i _ => sq_nonneg ((x - y) i)) (Finset.mem_univ 0)
  have heq : squaredNorm (witnessGradient x - witnessGradient y) =
      ((1 / 1000 : ℝ) * Real.cos (x 0) - (1 / 1000) * Real.cos (y 0)) ^ 2 := by
    norm_num [squaredNorm, witnessGradient, Pi.sub_apply, Fin.sum_univ_succ]
  rw [heq]
  have h0 := abs_nonneg ((1 / 1000 : ℝ) * Real.cos (x 0) - (1 / 1000) * Real.cos (y 0))
  nlinarith [sq_abs ((1 / 1000 : ℝ) * Real.cos (x 0) - (1 / 1000) * Real.cos (y 0)),
    sq_abs (x 0 - y 0)]

/-- The β=1 run on this valid Fisher block, Proposition 3.4.
The gradients supplied to the run equal the actual stationary gradient. -/
def witnessRun : ℕ → Fin 4 → ℝ :=
  adaFisherRun (stepSize 1) 0 1 (fun _ => witnessFisher)
    (fun _ => witnessGradient 0) 0

/-- Every iterate stays at zero and has squared gradient 10⁻⁶,
Proposition 3.4's admitted endpoint. -/
theorem witnessRun_stationary (t : ℕ) :
    witnessRun t = 0 ∧ squaredNorm (witnessGradient (witnessRun t)) = (1 / 1000 : ℝ) ^ 2 := by
  have ht : witnessRun t = 0 := adaFisherRun_one _ _ _ _ t
  refine ⟨ht, ?_⟩
  rw [ht]
  norm_num [squaredNorm, witnessGradient]

/-- The monotonicity assumption on Fᵗ/ηᵗ holds at every coordinate of
the nondegenerate block witness, Proposition 3.4. -/
theorem witnessFisher_effective_monotone (i : Fin 4) :
    Monotone (fun t => witnessFisher i / stepSize 1 t) := by
  intro a b hab
  simp only [stepSize, one_div, div_inv_eq_mul]
  apply mul_le_mul_of_nonneg_left _ (le_trans (by norm_num) (witness_bounds.2 i).1)
  apply Real.sqrt_le_sqrt
  exact_mod_cast Nat.add_le_add_right hab 1

/-- Full horizon refutation of Proposition 3.4 with an actual 2×2
block-Kronecker FIM. For any four finite constants, at some positive horizon
every expected squared gradient exceeds the claimed RHS. Deterministic
Dirac randomness realizes unbiased independent zero noise. -/
theorem nonconvex_block_counterexample (C₁ C₂ C₃ C₄ : ℝ) :
    ∃ T : ℕ, 0 < T ∧ ∀ t ∈ Finset.range T,
      nonconvexRate 2 1 (1 / 1000) 4 C₁ C₂ C₃ C₄ T <
        ∫ _ : ℝ, squaredNorm (witnessGradient (witnessRun t)) ∂Measure.dirac (0 : ℝ) := by
  have hlt := (nonconvexRate_tendsto 2 1 (1 / 1000) 4 C₁ C₂ C₃ C₄).eventually_lt
    tendsto_const_nhds (by norm_num : (0 : ℝ) < (1 / 1000 : ℝ) ^ 2)
  obtain ⟨T, hT, hrate⟩ := (hlt.and (eventually_gt_atTop 0)).exists
  refine ⟨T, hrate, ?_⟩
  intro t ht
  simpa [(witnessRun_stationary t).2] using hT

end Transformer.AdaFisher
