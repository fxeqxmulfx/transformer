/-
# Scalar heads and convex token rows

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.2, Theorem 1 and
Appendix A.2/A.4.  This file constructs the two directions of the
finite-width equivalence after the paper's loss-factor error is corrected.
-/

import Transformer.Convexifying.Section3_Scaling

open scoped BigOperators

namespace Transformer.Convexifying

/-- Aggregate the heads into the unrestricted token-row matrix.
Source: arXiv:2211.11052v1, §3.2, equation (9). -/
def scalarConvexOf {h n d : ℕ} (p : ScalarParameters h n d) : Fin n → Vec d :=
  fun k q => ∑ j, p.attention j k * (p.value j q * p.output j)

/-- Aggregation preserves predictions.
Source: arXiv:2211.11052v1, §3.2, equations (8)–(9). -/
theorem scalarConvexOf_prediction {h n d : ℕ} (X : Fin n → Vec d)
    (p : ScalarParameters h n d) :
    convexPrediction X (scalarConvexOf p) = scalarPrediction X p := by
  unfold convexPrediction scalarConvexOf scalarPrediction
  simp_rw [scalarHead_eq_product_sum]
  calc
    (∑ k, ∑ q, (∑ j, p.attention j k * (p.value j q * p.output j)) * X k q)
        = ∑ k, ∑ q, ∑ j,
            p.attention j k * (X k q * (p.value j q * p.output j)) := by
          apply Finset.sum_congr rfl
          intro k _
          apply Finset.sum_congr rfl
          intro q _
          rw [Finset.sum_mul]
          apply Finset.sum_congr rfl
          intro j _
          ring
    _ = ∑ k, ∑ j, ∑ q,
          p.attention j k * (X k q * (p.value j q * p.output j)) := by
          apply Finset.sum_congr rfl
          intro k _
          exact Finset.sum_comm
    _ = ∑ j, ∑ k, ∑ q,
          p.attention j k * (X k q * (p.value j q * p.output j)) :=
          Finset.sum_comm
    _ = ∑ j, ∑ k, p.attention j k *
          (∑ q, X k q * (p.value j q * p.output j)) := by
          apply Finset.sum_congr rfl
          intro j _
          apply Finset.sum_congr rfl
          intro k _
          rw [Finset.mul_sum]
    _ = _ := rfl

/-- The group norm of the aggregated matrix is bounded by the sum of
head product norms.
Source: arXiv:2211.11052v1, Appendix A.3. -/
theorem scalarConvexOf_reg_le {h n d : ℕ} (p : ScalarParameters h n d)
    (hp : p.Feasible) :
    (∑ k, norm₂ (scalarConvexOf p k)) ≤
      ∑ j, |p.output j| * norm₂ (p.value j) := by
  have hk (k : Fin n) : norm₂ (scalarConvexOf p k) ≤
      ∑ j, p.attention j k * (|p.output j| * norm₂ (p.value j)) := by
    have heq : scalarConvexOf p k =
        (fun q => ∑ j, (p.attention j k * p.output j) * p.value j q) := by
      funext q
      simp [scalarConvexOf]
      apply Finset.sum_congr rfl
      intro j _
      ring
    rw [heq]
    calc
      norm₂ (fun q => ∑ j, (p.attention j k * p.output j) * p.value j q)
          ≤ ∑ j, |p.attention j k * p.output j| * norm₂ (p.value j) :=
          norm₂_linearCombination_le _ _
      _ = ∑ j, p.attention j k * (|p.output j| * norm₂ (p.value j)) := by
          apply Finset.sum_congr rfl
          intro j _
          rw [abs_mul, abs_of_nonneg ((hp j).1 k)]
          ring
  calc
    (∑ k, norm₂ (scalarConvexOf p k)) ≤
        ∑ k, ∑ j, p.attention j k * (|p.output j| * norm₂ (p.value j)) :=
      Finset.sum_le_sum (fun k _ => hk k)
    _ = ∑ j, ∑ k, p.attention j k * (|p.output j| * norm₂ (p.value j)) :=
      Finset.sum_comm
    _ = ∑ j, |p.output j| * norm₂ (p.value j) := by
      apply Finset.sum_congr rfl
      intro j _
      rw [← Finset.sum_mul, (hp j).2]
      ring

/-- A basis-attention head reproduces one convex token row.
Source: arXiv:2211.11052v1, §3.2, `prop:mapping`. -/
theorem scalarHead_basis_unit {n d : ℕ} (X : Fin n → Vec d)
    (z : Vec d) (k : Fin n) :
    scalarHead X (fun t => if t = k then 1 else 0)
      (unitPair z).1 (unitPair z).2 = ∑ q, z q * X k q := by
  rw [scalarHead_eq_product_sum]
  have hprod := (unitPair_spec z).1
  simp [hprod, mul_comm]

/-- Recover `n` token heads and pad with `m` zero heads.  A token index
`k₀` supplies valid simplex weights for the padding.
Source: arXiv:2211.11052v1, Appendix A.2/A.4. -/
noncomputable def scalarHeadsOf {n d : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin n → Vec d) : ScalarParameters (n + m) n d where
  attention := Fin.addCases (fun k => fun t => if t = k then 1 else 0)
    (fun _ => fun t => if t = k₀ then 1 else 0)
  value := Fin.addCases (fun k => (unitPair (Z k)).1) (fun _ => fun _ => 0)
  output := Fin.addCases (fun k => (unitPair (Z k)).2) (fun _ => 0)

/-- The recovered heads satisfy the scaled constraints.
Source: arXiv:2211.11052v1, §3.2, Lemma 1 and Proposition 1. -/
theorem scalarHeadsOf_feasible {n d : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin n → Vec d) : (scalarHeadsOf m k₀ Z).ScaledFeasible := by
  constructor
  · intro j
    refine Fin.addCases (fun k => ?_) (fun j => ?_) j
    · simpa [scalarHeadsOf] using (simplex_basis k)
    · simpa [scalarHeadsOf] using (simplex_basis k₀)
  · intro j
    refine Fin.addCases (fun k => ?_) (fun j => ?_) j
    · simpa [scalarHeadsOf] using (unitPair_spec (Z k)).2.1
    · simp [scalarHeadsOf, norm₂, normSq]

/-- Recovered heads preserve every convex prediction.
Source: arXiv:2211.11052v1, §3.2, `prop:mapping`. -/
theorem scalarHeadsOf_prediction {n d : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin n → Vec d) (X : Fin n → Vec d) :
    scalarPrediction X (scalarHeadsOf m k₀ Z) = convexPrediction X Z := by
  unfold scalarPrediction
  rw [Fin.sum_univ_add]
  have hpad : (∑ j : Fin m,
      scalarHead X ((scalarHeadsOf m k₀ Z).attention (Fin.natAdd n j))
        ((scalarHeadsOf m k₀ Z).value (Fin.natAdd n j))
        ((scalarHeadsOf m k₀ Z).output (Fin.natAdd n j))) = 0 := by
    simp [scalarHeadsOf, scalarHead]
  rw [hpad, add_zero]
  calc
    (∑ k : Fin n,
      scalarHead X ((scalarHeadsOf m k₀ Z).attention (Fin.castAdd m k))
        ((scalarHeadsOf m k₀ Z).value (Fin.castAdd m k))
        ((scalarHeadsOf m k₀ Z).output (Fin.castAdd m k)))
        = ∑ k, ∑ q, Z k q * X k q := by
          apply Finset.sum_congr rfl
          intro k _
          simpa [scalarHeadsOf] using scalarHead_basis_unit X (Z k) k
    _ = convexPrediction X Z := by simp [convexPrediction]

/-- Recovered heads reproduce the group regularizer.
Source: arXiv:2211.11052v1, Appendix A.2. -/
theorem scalarHeadsOf_reg {n d : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin n → Vec d) :
    (∑ j, |(scalarHeadsOf m k₀ Z).output j|) = ∑ k, norm₂ (Z k) := by
  rw [Fin.sum_univ_add]
  simp [scalarHeadsOf, (unitPair_spec _).2.2]

/-- The convex objective of aggregated scaled heads is no larger than their
scaled nonconvex objective.
Source: arXiv:2211.11052v1, §3.2, Theorem 1, corrected loss factor. -/
theorem scalarConvexOf_objective_le {N h n d : ℕ}
    (X : Data N n d) (y : Vec N) (L : ℝ → ℝ → ℝ) (β : ℝ)
    (p : ScalarParameters h n d) (hp : p.ScaledFeasible) (hβ : 0 ≤ β) :
    correctedConvexObjective X y L β (scalarConvexOf p) ≤
      scaledScalarObjective X y L β p := by
  have hreg₁ := scalarConvexOf_reg_le p hp.1
  have hreg₂ : (∑ j, |p.output j| * norm₂ (p.value j)) ≤
      ∑ j, |p.output j| := by
    apply Finset.sum_le_sum
    intro j _
    have habs := abs_nonneg (p.output j)
    have hunit := hp.2 j
    nlinarith [mul_nonneg (sub_nonneg.mpr hunit) habs]
  have hreg := hreg₁.trans hreg₂
  unfold correctedConvexObjective scaledScalarObjective
  simp only [scalarConvexOf_prediction]
  exact add_le_add_right (mul_le_mul_of_nonneg_left hreg hβ) _

/-- Token-row recovery gives exactly the corrected convex objective.
Source: arXiv:2211.11052v1, Appendix A.2, `eq:scaling_proof_final`. -/
theorem scalarHeadsOf_objective {N n d : ℕ} (m : ℕ) (k₀ : Fin n)
    (X : Data N n d) (y : Vec N) (L : ℝ → ℝ → ℝ) (β : ℝ)
    (Z : Fin n → Vec d) :
    scaledScalarObjective X y L β (scalarHeadsOf m k₀ Z) =
      correctedConvexObjective X y L β Z := by
  unfold scaledScalarObjective correctedConvexObjective
  simp only [scalarHeadsOf_prediction, scalarHeadsOf_reg]

end Transformer.Convexifying
