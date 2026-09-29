/-
# Aggregating vector attention heads

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.3 and Appendix A.2/A.5.
-/

import Transformer.Convexifying.Section3_ScalarBridge
import Transformer.Convexifying.Section3_Vector

open scoped BigOperators

namespace Transformer.Convexifying

/-- Project one output coordinate of equation (11) to the scalar model.
Source: arXiv:2211.11052v1, §3.3, equations (11)–(12). -/
def vectorScalarParameters {h n d c : ℕ} (p : VectorParameters h n d c)
    (l : Fin c) : ScalarParameters h n d :=
  ⟨p.attention, p.value, fun j => p.output j l⟩
/-- Aggregate each output coordinate into a convex token-row matrix.
Source: arXiv:2211.11052v1, §3.3, equation (12). -/
def vectorConvexOf {h n d c : ℕ} (p : VectorParameters h n d c) :
    Fin c → Fin n → Vec d := fun l => scalarConvexOf (vectorScalarParameters p l)
/-- Aggregation preserves each output coordinate's prediction.
Source: arXiv:2211.11052v1, §3.3, equations (11)–(12). -/
theorem vectorConvexOf_prediction {h n d c : ℕ} (X : Fin n → Vec d)
    (p : VectorParameters h n d c) (l : Fin c) :
    convexPrediction X (vectorConvexOf p l) = vectorPrediction X p l := by
  exact scalarConvexOf_prediction X (vectorScalarParameters p l)
/-- The output ℓ₁ norm is nonnegative.
Source: arXiv:2211.11052v1, §3.3, equation (11). -/
theorem norm₁_nonneg {c : ℕ} (v : Vec c) : 0 ≤ norm₁ v := by
  exact Finset.sum_nonneg (fun l _ => abs_nonneg _)
/-- The multi-output group penalty is bounded by the original quadratic
head penalty.
Source: arXiv:2211.11052v1, §3.3 and Appendix A.5. -/
theorem vectorConvexOf_reg_le {h n d c : ℕ} (p : VectorParameters h n d c)
    (hp : p.Feasible) :
    (∑ l, ∑ k, norm₂ (vectorConvexOf p l k)) ≤
      ∑ j, (normSq (p.value j) + (norm₁ (p.output j)) ^ 2) / 2 := by
  have h₁ : (∑ l, ∑ k, norm₂ (vectorConvexOf p l k)) ≤
      ∑ l, ∑ j, |p.output j l| * norm₂ (p.value j) := by
    apply Finset.sum_le_sum
    intro l _
    exact scalarConvexOf_reg_le (vectorScalarParameters p l) hp
  have h₂ : (∑ l, ∑ j, |p.output j l| * norm₂ (p.value j)) =
      ∑ j, norm₂ (p.value j) * norm₁ (p.output j) := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j _
    rw [← Finset.sum_mul]
    simp [norm₁, mul_comm]
  have h₃ : (∑ j, norm₂ (p.value j) * norm₁ (p.output j)) ≤
      ∑ j, (normSq (p.value j) + (norm₁ (p.output j)) ^ 2) / 2 := by
    apply Finset.sum_le_sum
    intro j _
    have hh := head_product_bound (p.value j) (norm₁ (p.output j))
    rwa [abs_of_nonneg (norm₁_nonneg _)] at hh
  exact h₁.trans (h₂.le.trans h₃)
/-- With separable loss, convex aggregation does not increase the objective.
Source: arXiv:2211.11052v1, §3.3, Theorem 2, corrected hypotheses. -/
theorem vectorConvexOf_objective_le {N h n d c : ℕ}
    (X : Data N n d) (y : Fin N → Vec c)
    (L : Vec c → Vec c → ℝ) (scalarLoss : ℝ → ℝ → ℝ) (β : ℝ)
    (p : VectorParameters h n d c) (hL : IsSeparableLoss L scalarLoss)
    (hp : p.Feasible) (hβ : 0 ≤ β) :
    vectorConvexObjective X y scalarLoss β (vectorConvexOf p) ≤
      vectorObjective X y L β p := by
  have hloss : (∑ i, ∑ l,
      scalarLoss (convexPrediction (X i) (vectorConvexOf p l)) (y i l)) =
      ∑ i, L (vectorPrediction (X i) p) (y i) := by
    apply Finset.sum_congr rfl
    intro i _
    rw [hL]
    apply Finset.sum_congr rfl
    intro l _
    rw [vectorConvexOf_prediction]
  unfold vectorConvexObjective vectorObjective
  rw [hloss]
  have hmul := mul_le_mul_of_nonneg_left (vectorConvexOf_reg_le p hp) hβ
  rw [← Finset.sum_div] at hmul
  nlinarith

end Transformer.Convexifying
