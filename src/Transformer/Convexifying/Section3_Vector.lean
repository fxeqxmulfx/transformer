/-
# Multi-output convex attention

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.3, Theorem 2
(`theo:attn_multihead_convex_vector`), and Appendix A.5.

The left-hand side of the source theorem accepts a vector-valued loss, while
its right-hand side sums scalar losses over outputs.  These are equal only for
separable losses.  Even with a separable squared loss, the claimed equivalence
fails for one head by specializing to one output.
-/

import Transformer.Convexifying.Section3_ScalarCounterexample

open scoped BigOperators

namespace Transformer.Convexifying

/-- The ℓ₁ norm of an output vector in equation (11). -/
def norm₁ {c : ℕ} (v : Vec c) : ℝ := ∑ l, |v l|

/-- Multi-output parameters in equation (11). -/
structure VectorParameters (h n d c : ℕ) where
  attention : Fin h → Vec n
  value : Fin h → Vec d
  output : Fin h → Vec c

/-- Feasibility of the multi-output attention weights. -/
def VectorParameters.Feasible {h n d c : ℕ} (p : VectorParameters h n d c) : Prop :=
  ∀ j, IsSimplex (p.attention j)

/-- The multi-output prediction in equation (11). -/
def vectorPrediction {h n d c : ℕ} (X : Fin n → Vec d)
    (p : VectorParameters h n d c) : Vec c :=
  fun l => ∑ j, scalarHead X (p.attention j) (p.value j) (p.output j l)

/-- The nonconvex objective of equation (11), with the printed squared
Euclidean value norm and squared ℓ₁ output norm. -/
noncomputable def vectorObjective {N h n d c : ℕ} (X : Data N n d)
    (y : Fin N → Vec c) (L : Vec c → Vec c → ℝ) (β : ℝ)
    (p : VectorParameters h n d c) : ℝ :=
  (∑ i, L (vectorPrediction (X i) p) (y i)) +
    β / 2 * ∑ j, (normSq (p.value j) + (norm₁ (p.output j)) ^ 2)

/-- The unrestricted `c` matrices in equation (12).  The paper applies
the same loss symbol to a vector in (11) and to a scalar in (12); here the
scalar loss is given its own type. -/
noncomputable def vectorConvexObjective {N n d c : ℕ} (X : Data N n d)
    (y : Fin N → Vec c) (L : ℝ → ℝ → ℝ) (β : ℝ)
    (Z : Fin c → Fin n → Vec d) : ℝ :=
  (∑ i, ∑ l, L (convexPrediction (X i) (Z l)) (y i l)) +
    β * ∑ l, ∑ k, norm₂ (Z l k)

/-- A vector-valued loss is separable over output coordinates, the condition
needed to compare the objectives of equations (11) and (12). -/
def IsSeparableLoss {c : ℕ} (L : Vec c → Vec c → ℝ)
    (scalarLoss : ℝ → ℝ → ℝ) : Prop :=
  ∀ prediction target, L prediction target =
    ∑ l, scalarLoss (prediction l) (target l)

/-- Specializing a vector model to its only output gives a scalar model. -/
def onlyOutputParameters (p : VectorParameters 1 2 1 1) :
    ScalarParameters 1 2 1 :=
  ⟨p.attention, p.value, fun j => p.output j 0⟩

/-- The single-output separable squared loss. -/
def oneOutputSquareLoss (prediction target : Vec 1) : ℝ :=
  squareLoss (prediction 0) (target 0)

/-- The squared loss example satisfies the loss-separability condition. -/
theorem oneOutputSquareLoss_separable :
    IsSeparableLoss oneOutputSquareLoss squareLoss := by
  intro prediction target
  simp [oneOutputSquareLoss]

/-- Target vectors for the two-sample, one-output counterexample. -/
def separatingVectorTargets : Fin 2 → Vec 1 :=
  fun i _ => separatingTargets i

/-- One output reproduces exactly the scalar objective. -/
private theorem vectorObjective_eq_scalar (p : VectorParameters 1 2 1 1) :
    vectorObjective separatingData separatingVectorTargets oneOutputSquareLoss
      (1 / 8) p =
    scalarObjective separatingData separatingTargets squareLoss (1 / 8)
      (onlyOutputParameters p) := by
  simp [vectorObjective, scalarObjective, vectorPrediction, scalarPrediction,
    oneOutputSquareLoss, separatingVectorTargets, onlyOutputParameters,
    norm₁, sq_abs]

/-- Every feasible one-head vector model in this instance costs at least `1`. -/
theorem oneHead_vector_lower_bound (p : VectorParameters 1 2 1 1)
    (hp : p.Feasible) :
    1 ≤ vectorObjective separatingData separatingVectorTargets
      oneOutputSquareLoss (1 / 8) p := by
  rw [vectorObjective_eq_scalar]
  exact oneHead_original_lower_bound (onlyOutputParameters p) hp

/-- The one-head vector-model feasibility hypothesis is satisfiable. -/
example :
    (⟨fun _ => fun k : Fin 2 => if k = 0 then 1 else 0,
      fun _ _ => 0, fun _ _ => 0⟩ : VectorParameters 1 2 1 1).Feasible := by
  intro _
  exact simplex_basis 0

/-- The vector convex model fits the two opposite targets with cost `1/4`. -/
theorem oneHead_vector_convex_value :
    vectorConvexObjective separatingData separatingVectorTargets
      squareLoss (1 / 8) (fun _ => separatingWeights) = 1 / 4 := by
  simpa [vectorConvexObjective, correctedConvexObjective,
    separatingVectorTargets, Fin.sum_univ_one] using oneHead_convex_value

/-- **Counterexample to Theorem 2 as stated.**  Its left and right programs
have different sublevel sets at `h = 1`, even for separable squared loss.
The appendix constructs `h c` heads from `h` token rows and `c` outputs,
which does not meet the fixed `h` on the theorem's left-hand side.
Source: arXiv:2211.11052v1, §3.3 and Appendix A.2/A.5. -/
theorem vector_equivalence_false_for_one_head :
    ¬ ∀ r : ℝ,
      (∃ p : VectorParameters 1 2 1 1,
        p.Feasible ∧
          vectorObjective separatingData separatingVectorTargets
            oneOutputSquareLoss (1 / 8) p ≤ r) ↔
      (∃ Z : Fin 1 → Fin 2 → Vec 1,
        vectorConvexObjective separatingData separatingVectorTargets
          squareLoss (1 / 8) Z ≤ r) := by
  intro h
  obtain ⟨p, hp, hbound⟩ := (h (1 / 2)).2
    ⟨fun _ => separatingWeights, by rw [oneHead_vector_convex_value]; norm_num⟩
  have hlower := oneHead_vector_lower_bound p hp
  linarith

/-- The simplex and loss hypotheses used in the vector counterexample are
satisfiable (use either basis attention vector). -/
example : IsSimplex (fun k : Fin 2 => if k = 0 then 1 else 0) ∧
    IsSeparableLoss oneOutputSquareLoss squareLoss :=
  ⟨simplex_basis 0, oneOutputSquareLoss_separable⟩

end Transformer.Convexifying
