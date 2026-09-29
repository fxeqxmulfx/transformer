/-
# Recovering vector attention heads

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.3 and Appendix A.2/A.5.
-/

import Transformer.Convexifying.Section3_VectorBridge

open scoped BigOperators

namespace Transformer.Convexifying

/-- Recover one head for each token-output pair and pad with zero heads.
Source: arXiv:2211.11052v1, Appendix A.2, vector extension. -/
noncomputable def vectorHeadsOf {n d c : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin c → Fin n → Vec d) : VectorParameters (n*c + m) n d c where
  attention := Fin.addCases
    (fun j => fun t => if t = (finProdFinEquiv.symm j).1 then 1 else 0)
    (fun _ => fun t => if t = k₀ then 1 else 0)
  value := Fin.addCases
    (fun j => (balancedPair (Z (finProdFinEquiv.symm j).2
      (finProdFinEquiv.symm j).1)).1)
    (fun _ => fun _ => 0)
  output := Fin.addCases
    (fun j => fun l => if l = (finProdFinEquiv.symm j).2 then
      (balancedPair (Z (finProdFinEquiv.symm j).2
        (finProdFinEquiv.symm j).1)).2 else 0)
    (fun _ => fun _ => 0)
/-- The recovered vector heads have simplex attention weights.
Source: arXiv:2211.11052v1, §3.3, equation (11). -/
theorem vectorHeadsOf_feasible {n d c : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin c → Fin n → Vec d) : (vectorHeadsOf m k₀ Z).Feasible := by
  intro j
  refine Fin.addCases (fun j => ?_) (fun j => ?_) j
  · simpa [vectorHeadsOf] using simplex_basis (finProdFinEquiv.symm j).1
  · simpa [vectorHeadsOf] using simplex_basis k₀
/-- A balanced row factor reproduces one token contribution.
Source: arXiv:2211.11052v1, Appendix A.2. -/
theorem scalarHead_basis_balanced {n d : ℕ} (X : Fin n → Vec d)
    (z : Vec d) (k : Fin n) :
    scalarHead X (fun t => if t = k then 1 else 0)
      (balancedPair z).1 (balancedPair z).2 = ∑ q, z q * X k q := by
  rw [scalarHead_eq_product_sum]
  have hprod := (balancedPair_spec z).1
  simp [hprod, mul_comm]
/-- A head assigned to output `l'` contributes only to that output.
Source: arXiv:2211.11052v1, Appendix A.2, vector extension. -/
theorem vectorHead_core {n d c : ℕ} (X : Fin n → Vec d)
    (Z : Fin c → Fin n → Vec d) (k : Fin n) (l l' : Fin c) :
    scalarHead X (fun t => if t = k then 1 else 0)
      (balancedPair (Z l' k)).1
      (if l = l' then (balancedPair (Z l' k)).2 else 0) =
    if l = l' then ∑ q, Z l' k q * X k q else 0 := by
  by_cases h : l = l'
  · simp [h, scalarHead_basis_balanced]
  · simp [h, scalarHead]
/-- Recovered vector heads reproduce all convex predictions.
Source: arXiv:2211.11052v1, §3.3 and Appendix A.2. -/
theorem vectorHeadsOf_prediction {n d c : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin c → Fin n → Vec d) (X : Fin n → Vec d) (l : Fin c) :
    vectorPrediction X (vectorHeadsOf m k₀ Z) l =
      convexPrediction X (Z l) := by
  unfold vectorPrediction
  rw [Fin.sum_univ_add]
  have hpad : (∑ j : Fin m,
      scalarHead X ((vectorHeadsOf m k₀ Z).attention (Fin.natAdd (n*c) j))
        ((vectorHeadsOf m k₀ Z).value (Fin.natAdd (n*c) j))
        ((vectorHeadsOf m k₀ Z).output (Fin.natAdd (n*c) j) l)) = 0 := by
    simp [vectorHeadsOf, scalarHead]
  rw [hpad, add_zero]
  have hactive (j : Fin (n*c)) :
      scalarHead X ((vectorHeadsOf m k₀ Z).attention (Fin.castAdd m j))
        ((vectorHeadsOf m k₀ Z).value (Fin.castAdd m j))
        ((vectorHeadsOf m k₀ Z).output (Fin.castAdd m j) l) =
      if l = (finProdFinEquiv.symm j).2 then
        ∑ q, Z (finProdFinEquiv.symm j).2 (finProdFinEquiv.symm j).1 q *
          X (finProdFinEquiv.symm j).1 q else 0 := by
    simpa [vectorHeadsOf] using vectorHead_core X Z
      (finProdFinEquiv.symm j).1 l (finProdFinEquiv.symm j).2
  simp_rw [hactive]
  calc
    (∑ j : Fin (n*c),
      if l = (finProdFinEquiv.symm j).2 then
        ∑ q, Z (finProdFinEquiv.symm j).2 (finProdFinEquiv.symm j).1 q *
          X (finProdFinEquiv.symm j).1 q else 0)
        = ∑ kl : Fin n × Fin c,
          if l = kl.2 then ∑ q, Z kl.2 kl.1 q * X kl.1 q else 0 := by
          exact (Fintype.sum_equiv finProdFinEquiv
            (fun kl : Fin n × Fin c =>
              if l = kl.2 then ∑ q, Z kl.2 kl.1 q * X kl.1 q else 0)
            (fun j : Fin (n*c) =>
              if l = (finProdFinEquiv.symm j).2 then
                ∑ q, Z (finProdFinEquiv.symm j).2 (finProdFinEquiv.symm j).1 q *
                  X (finProdFinEquiv.symm j).1 q else 0)
            (by intro kl; simp)).symm
    _ = ∑ k, ∑ l' : Fin c,
          if l = l' then ∑ q, Z l' k q * X k q else 0 := by
          rw [Fintype.sum_prod_type]
    _ = convexPrediction X (Z l) := by
          simp [convexPrediction]
/-- The recovered quadratic penalty equals the group norm.
Source: arXiv:2211.11052v1, Appendix A.2. -/
theorem vectorHeadsOf_reg {n d c : ℕ} (m : ℕ) (k₀ : Fin n)
    (Z : Fin c → Fin n → Vec d) :
    (∑ j, (normSq ((vectorHeadsOf m k₀ Z).value j) +
      (norm₁ ((vectorHeadsOf m k₀ Z).output j)) ^ 2) / 2) =
      ∑ l, ∑ k, norm₂ (Z l k) := by
  rw [Fin.sum_univ_add]
  have hpad : (∑ j : Fin m,
      (normSq ((vectorHeadsOf m k₀ Z).value (Fin.natAdd (n*c) j)) +
      (norm₁ ((vectorHeadsOf m k₀ Z).output (Fin.natAdd (n*c) j))) ^ 2) / 2) = 0 := by
    simp [vectorHeadsOf, normSq, norm₁]
  rw [hpad, add_zero]
  have hactive (j : Fin (n*c)) :
      (normSq ((vectorHeadsOf m k₀ Z).value (Fin.castAdd m j)) +
      (norm₁ ((vectorHeadsOf m k₀ Z).output (Fin.castAdd m j))) ^ 2) / 2 =
      norm₂ (Z (finProdFinEquiv.symm j).2 (finProdFinEquiv.symm j).1) := by
    have hnorm1 : norm₁ (fun l : Fin c =>
        if l = (finProdFinEquiv.symm j).2 then
          (balancedPair (Z (finProdFinEquiv.symm j).2
            (finProdFinEquiv.symm j).1)).2 else 0) =
        |(balancedPair (Z (finProdFinEquiv.symm j).2
          (finProdFinEquiv.symm j).1)).2| := by
      simp only [norm₁, abs_ite, abs_zero]
      simp
    simp only [vectorHeadsOf, Fin.addCases_left]
    rw [hnorm1, sq_abs]
    exact (balancedPair_spec (Z (finProdFinEquiv.symm j).2
      (finProdFinEquiv.symm j).1)).2
  simp_rw [hactive]
  calc
    (∑ j : Fin (n*c),
      norm₂ (Z (finProdFinEquiv.symm j).2 (finProdFinEquiv.symm j).1))
      = ∑ kl : Fin n × Fin c, norm₂ (Z kl.2 kl.1) := by
        exact (Fintype.sum_equiv finProdFinEquiv
          (fun kl : Fin n × Fin c => norm₂ (Z kl.2 kl.1))
          (fun j : Fin (n*c) => norm₂ (Z (finProdFinEquiv.symm j).2
            (finProdFinEquiv.symm j).1))
          (by intro kl; simp)).symm
    _ = ∑ k, ∑ l, norm₂ (Z l k) := by rw [Fintype.sum_prod_type]
    _ = ∑ l, ∑ k, norm₂ (Z l k) := Finset.sum_comm
/-- Under separable loss, the recovered vector model has exactly the convex
objective value.
Source: arXiv:2211.11052v1, §3.3, Theorem 2, corrected hypotheses. -/
theorem vectorHeadsOf_objective {N n d c : ℕ} (m : ℕ) (k₀ : Fin n)
    (X : Data N n d) (y : Fin N → Vec c)
    (L : Vec c → Vec c → ℝ) (scalarLoss : ℝ → ℝ → ℝ) (β : ℝ)
    (hL : IsSeparableLoss L scalarLoss)
    (Z : Fin c → Fin n → Vec d) :
    vectorObjective X y L β (vectorHeadsOf m k₀ Z) =
      vectorConvexObjective X y scalarLoss β Z := by
  have hloss : (∑ i, L (vectorPrediction (X i) (vectorHeadsOf m k₀ Z)) (y i)) =
      ∑ i, ∑ l, scalarLoss (convexPrediction (X i) (Z l)) (y i l) := by
    apply Finset.sum_congr rfl
    intro i _
    rw [hL]
    apply Finset.sum_congr rfl
    intro l _
    rw [vectorHeadsOf_prediction]
  unfold vectorObjective vectorConvexObjective
  rw [hloss]
  have hreg := vectorHeadsOf_reg m k₀ Z
  rw [← Finset.sum_div] at hreg
  have hmul := congrArg (fun t : ℝ => β * t) hreg
  nlinarith

end Transformer.Convexifying
