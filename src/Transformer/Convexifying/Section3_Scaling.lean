/-
# Rescaling scalar attention heads

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.2, Lemma 1
(`lemma:scaling`) and Appendix A.1, equations (18)–(22).
-/

import Transformer.Convexifying.Section3_Factorization

open scoped BigOperators

namespace Transformer.Convexifying

/-- The scaled problem in equation (8)'s rescaling lemma.  Feasibility also
requires the value vector of each head to have norm at most `1`. -/
noncomputable def scaledScalarObjective {N h n d : ℕ} (X : Data N n d) (y : Vec N)
    (L : ℝ → ℝ → ℝ) (β : ℝ) (p : ScalarParameters h n d) : ℝ :=
  (∑ i, L (scalarPrediction (X i) p) (y i)) +
    β * ∑ j, |p.output j|

/-- Parameter constraints in the scaled problem of Lemma 1. -/
def ScalarParameters.ScaledFeasible {h n d : ℕ} (p : ScalarParameters h n d) : Prop :=
  p.Feasible ∧ ∀ j, norm₂ (p.value j) ≤ 1

/-- The AM–GM step in the paper's proof: the balanced quadratic penalty of
one head dominates the product penalty induced by scaling.
Source: arXiv:2211.11052v1, Appendix A.1, display after `eq:scaling`. -/
theorem head_product_bound {d : ℕ} (v : Vec d) (b : ℝ) :
    norm₂ v * |b| ≤ (normSq v + b ^ 2) / 2 := by
  have hv := norm₂_sq v
  have hb : |b| ^ 2 = b ^ 2 := sq_abs b
  nlinarith [sq_nonneg (norm₂ v - |b|)]

/-- A head prediction depends only on the coordinatewise products of its
value and output weights. -/
theorem scalarHead_eq_product_sum {n d : ℕ} (X : Fin n → Vec d)
    (a : Vec n) (v : Vec d) (b : ℝ) :
    scalarHead X a v b = ∑ k, a k * (∑ q, X k q * (v q * b)) := by
  simp [scalarHead, Finset.mul_sum, mul_left_comm, mul_comm]

/-- Replacing every head by another factorization of the same product vector
leaves all scalar predictions unchanged. -/
theorem scalarPrediction_congr_products {h n d : ℕ} (X : Fin n → Vec d)
    (p p' : ScalarParameters h n d)
    (hatt : p'.attention = p.attention)
    (hprod : ∀ j q, p'.value j q * p'.output j =
      p.value j q * p.output j) :
    scalarPrediction X p' = scalarPrediction X p := by
  unfold scalarPrediction
  apply Finset.sum_congr rfl
  intro j _
  rw [scalarHead_eq_product_sum, scalarHead_eq_product_sum, hatt]
  apply Finset.sum_congr rfl
  intro k _
  congr 1
  apply Finset.sum_congr rfl
  intro q _
  rw [hprod]

/-- The norm of a head's product vector. -/
theorem norm₂_value_product {d : ℕ} (v : Vec d) (b : ℝ) :
    norm₂ (fun q => v q * b) = norm₂ v * |b| := by
  have heq : (fun q => v q * b) = (fun q => b * v q) := by
    funext q
    ring
  rw [heq, norm₂_smul]
  ring

/-- **Lemma 1 (`lemma:scaling`), as a statement about all objective
sublevel sets.**  The two optimizations have the same infimum, including when
it is not attained.  Unlike a bare equality of `sInf`, this formulation does
not require a lower-bound assumption on the arbitrary loss.

The appendix divides by a value norm without discussing zero heads.  The
formal proof includes zero value and zero output heads through explicit
balanced and unit factorizations.

Source: arXiv:2211.11052v1, §3.2, `lemma:scaling`, and Appendix A.1. -/
theorem scaling_equivalence {N h n d : ℕ} (X : Data N n d) (y : Vec N)
    (L : ℝ → ℝ → ℝ) (β : ℝ) (hβ : 0 ≤ β) :
    ∀ r : ℝ,
      (∃ p : ScalarParameters h n d,
        p.Feasible ∧ scalarObjective X y L β p ≤ r) ↔
      (∃ p : ScalarParameters h n d,
        p.ScaledFeasible ∧ scaledScalarObjective X y L β p ≤ r) := by
  intro r
  constructor
  · rintro ⟨p, hp, hcost⟩
    let w : Fin h → Vec d := fun j q => p.value j q * p.output j
    let p' : ScalarParameters h n d :=
      ⟨p.attention, fun j => (unitPair (w j)).1,
        fun j => (unitPair (w j)).2⟩
    have hpred : ∀ i, scalarPrediction (X i) p' =
        scalarPrediction (X i) p := by
      intro i
      exact scalarPrediction_congr_products (X i) p p' rfl
        (fun j q => (unitPair_spec (w j)).1 q)
    have hunit : p'.ScaledFeasible := by
      constructor
      · exact hp
      · intro j
        exact (unitPair_spec (w j)).2.1
    have hreg : ∀ j, |p'.output j| ≤
        (normSq (p.value j) + (p.output j) ^ 2) / 2 := by
      intro j
      change |(unitPair (w j)).2| ≤ _
      rw [(unitPair_spec (w j)).2.2]
      change norm₂ (fun q => p.value j q * p.output j) ≤ _
      rw [norm₂_value_product]
      exact head_product_bound _ _
    have hsum : (∑ j, |p'.output j|) ≤
        ∑ j, (normSq (p.value j) + (p.output j) ^ 2) / 2 :=
      Finset.sum_le_sum (fun j _ => hreg j)
    have hobj : scaledScalarObjective X y L β p' ≤ scalarObjective X y L β p := by
      unfold scaledScalarObjective scalarObjective
      simp only [hpred]
      have hmul := mul_le_mul_of_nonneg_left hsum hβ
      rw [← Finset.sum_div] at hmul
      linarith
    exact ⟨p', hunit, hobj.trans hcost⟩
  · rintro ⟨p, hp, hcost⟩
    let w : Fin h → Vec d := fun j q => p.value j q * p.output j
    let p' : ScalarParameters h n d :=
      ⟨p.attention, fun j => (balancedPair (w j)).1,
        fun j => (balancedPair (w j)).2⟩
    have hpred : ∀ i, scalarPrediction (X i) p' =
        scalarPrediction (X i) p := by
      intro i
      exact scalarPrediction_congr_products (X i) p p' rfl
        (fun j q => (balancedPair_spec (w j)).1 q)
    have hreg : ∀ j,
        (normSq (p'.value j) + (p'.output j) ^ 2) / 2 ≤ |p.output j| := by
      intro j
      change (normSq (balancedPair (w j)).1 +
        (balancedPair (w j)).2 ^ 2) / 2 ≤ _
      rw [(balancedPair_spec (w j)).2]
      change norm₂ (fun q => p.value j q * p.output j) ≤ _
      rw [norm₂_value_product]
      have habs : 0 ≤ |p.output j| := abs_nonneg _
      have hunit := hp.2 j
      nlinarith [mul_nonneg (sub_nonneg.mpr hunit) habs]
    have hsum : (∑ j, (normSq (p'.value j) + (p'.output j) ^ 2) / 2) ≤
        ∑ j, |p.output j| :=
      Finset.sum_le_sum (fun j _ => hreg j)
    have hobj : scalarObjective X y L β p' ≤ scaledScalarObjective X y L β p := by
      unfold scalarObjective scaledScalarObjective
      simp only [hpred]
      have hmul := mul_le_mul_of_nonneg_left hsum hβ
      rw [← Finset.sum_div] at hmul
      nlinarith
    exact ⟨p', hp.1, hobj.trans hcost⟩

/-- The hypotheses of `scaling_equivalence` are satisfiable. -/
example : (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Convexifying
