/-
# Counterexamples to the scalar equivalence as printed

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.2, Theorem 1
(`theo:attn_multihead_convex_scalar`), equations (8)–(9).

The first example isolates the factor `1/2` applied only to the loss in the
printed convex objective.  The second uses one head and two tokens, and still
works after that factor is removed.  Thus a sufficient head count is essential.
-/

import Transformer.Convexifying.Section3_Models

open scoped BigOperators

namespace Transformer.Convexifying

/-- Squared loss, one of the convex losses allowed in §3.2. -/
def squareLoss (prediction target : ℝ) : ℝ := (prediction - target) ^ 2

/-- Squared loss is convex in its prediction argument, so the examples below
lie within the paper's stated class of arbitrary convex losses. -/
theorem squareLoss_convex (x y target t : ℝ)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    squareLoss ((1 - t) * x + t * y) target ≤
      (1 - t) * squareLoss x target + t * squareLoss y target := by
  have hprod : 0 ≤ t * (1 - t) * (x - y) ^ 2 :=
    mul_nonneg (mul_nonneg ht0 (sub_nonneg.mpr ht1)) (sq_nonneg _)
  have hid :
      (1 - t) * squareLoss x target + t * squareLoss y target -
        squareLoss ((1 - t) * x + t * y) target =
      t * (1 - t) * (x - y) ^ 2 := by
    unfold squareLoss
    ring
  linarith

/-- The convexity interval is nonempty. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- One sample, one token, one feature, and a zero input. -/
def zeroData : Data 1 1 1 := fun _ _ _ => 0

/-- The corresponding nonzero target. -/
def unitTarget : Vec 1 := fun _ => 1

/-- Every feasible one-head original model on the zero data has value at
least `1`.  The regularizer is nonnegative. -/
theorem zeroData_original_lower_bound (p : ScalarParameters 1 1 1) :
    1 ≤ scalarObjective zeroData unitTarget squareLoss 1 p := by
  have hs : 0 ≤ normSq (p.value 0) := normSq_nonneg _
  have hb : 0 ≤ (p.output 0) ^ 2 := sq_nonneg _
  simp only [scalarObjective, squareLoss, scalarPrediction, scalarHead,
    zeroData, unitTarget, Fin.sum_univ_one, mul_zero, zero_mul,
    zero_sub, even_two, Even.neg_pow] at *
  nlinarith

/-- The printed convex model has value `1/2` at the zero parameter. -/
theorem zeroData_printed_value :
    printedConvexObjective zeroData unitTarget squareLoss 1
      (fun _ _ => 0 : Fin 1 → Vec 1) = 1 / 2 := by
  norm_num [printedConvexObjective, convexPrediction, zeroData, unitTarget,
    squareLoss, norm₂, normSq, Fin.sum_univ_one]

/-- Theorem 1 as printed is false even when `h = n = 1`: its equations (8)
and (9) assign different values to zero data with a nonzero target.
Source: arXiv:2211.11052v1, §3.2, `theo:attn_multihead_convex_scalar`. -/
theorem printed_scalar_equivalence_false_from_loss_factor :
    ¬ ∀ r : ℝ,
      (∃ p : ScalarParameters 1 1 1,
        p.Feasible ∧ scalarObjective zeroData unitTarget squareLoss 1 p ≤ r) ↔
      (∃ Z : Fin 1 → Vec 1,
        printedConvexObjective zeroData unitTarget squareLoss 1 Z ≤ r) := by
  intro h
  obtain ⟨p, _, hp⟩ := (h (1 / 2)).2
    ⟨fun _ _ => 0, by rw [zeroData_printed_value]⟩
  have hlower := zeroData_original_lower_bound p
  linarith

/-- Two samples with different active tokens, in one feature dimension. -/
def separatingData : Data 2 2 1 :=
  fun i k _ => if i = k then 1 else 0

/-- The two samples demand opposite signs. -/
def separatingTargets : Vec 2 := fun i => if i = 0 then 1 else -1

/-- The unrestricted matrix fits both targets exactly. -/
def separatingWeights : Fin 2 → Vec 1 :=
  fun k _ => if k = 0 then 1 else -1

private theorem separating_prediction_zero (p : ScalarParameters 1 2 1) :
    scalarPrediction (separatingData 0) p =
      p.attention 0 0 * (p.value 0 0 * p.output 0) := by
  simp [scalarPrediction, scalarHead, separatingData]
  ring

private theorem separating_prediction_one (p : ScalarParameters 1 2 1) :
    scalarPrediction (separatingData 1) p =
      p.attention 0 1 * (p.value 0 0 * p.output 0) := by
  simp [scalarPrediction, scalarHead, separatingData]
  ring

/-- One head has the same sign on both samples, so its squared training
loss is at least `1`; this includes the best choice of its simplex weight. -/
theorem oneHead_original_lower_bound (p : ScalarParameters 1 2 1)
    (hp : p.Feasible) :
    1 ≤ scalarObjective separatingData separatingTargets squareLoss (1 / 8) p := by
  have ha0 : 0 ≤ p.attention 0 0 := (hp 0).1 0
  have ha1 : 0 ≤ p.attention 0 1 := (hp 0).1 1
  have hreg : 0 ≤ normSq (p.value 0) + (p.output 0) ^ 2 :=
    add_nonneg (normSq_nonneg _) (sq_nonneg _)
  let t : ℝ := p.value 0 0 * p.output 0
  have hsign : 1 ≤ (p.attention 0 0 * t - 1) ^ 2 +
      (p.attention 0 1 * t + 1) ^ 2 := by
    rcases le_total 0 t with ht | ht
    · have hnonneg : 0 ≤ p.attention 0 1 * t := mul_nonneg ha1 ht
      nlinarith [sq_nonneg (p.attention 0 0 * t - 1)]
    · have hnonpos : p.attention 0 0 * t ≤ 0 := mul_nonpos_of_nonneg_of_nonpos ha0 ht
      nlinarith [sq_nonneg (p.attention 0 1 * t + 1)]
  simp only [scalarObjective, squareLoss, separatingTargets,
    Fin.sum_univ_two, separating_prediction_zero, separating_prediction_one]
  dsimp [t] at hsign
  norm_num at hsign ⊢
  nlinarith

/-- The one-head simplex feasibility hypothesis used above is satisfiable. -/
example :
    (⟨fun _ => fun k : Fin 2 => if k = 0 then 1 else 0,
      fun _ _ => 0, fun _ => 0⟩ : ScalarParameters 1 2 1).Feasible := by
  intro _
  exact simplex_basis 0

/-- The corrected convex objective (without the spurious loss factor) has
value `1/4` at a matrix with two nonzero rows. -/
theorem oneHead_convex_value :
    correctedConvexObjective separatingData separatingTargets squareLoss (1 / 8)
      separatingWeights = 1 / 4 := by
  norm_num [correctedConvexObjective, convexPrediction, separatingData,
    separatingTargets, separatingWeights, squareLoss, norm₂, normSq,
    Fin.sum_univ_two, Fin.sum_univ_one]

/-- Theorem 1 also fails after the loss-factor typo is corrected, if the
number of heads is fixed at `h = 1`.  Here `n = N = 2`, `d = 1`, and the
convex optimum is strictly below every feasible one-head value.
The appendix strong-duality argument explicitly requires `h ≥ h*`, with
`h* ≤ N + 1`, but Theorem 1 omits this hypothesis.
Source: arXiv:2211.11052v1, §3.2 and Appendix A.4. -/
theorem corrected_scalar_equivalence_false_for_one_head :
    ¬ ∀ r : ℝ,
      (∃ p : ScalarParameters 1 2 1,
        p.Feasible ∧
          scalarObjective separatingData separatingTargets squareLoss (1 / 8) p ≤ r) ↔
      (∃ Z : Fin 2 → Vec 1,
        correctedConvexObjective separatingData separatingTargets squareLoss (1 / 8) Z ≤ r) := by
  intro h
  obtain ⟨p, hp, hbound⟩ := (h (1 / 2)).2
    ⟨separatingWeights, by rw [oneHead_convex_value]; norm_num⟩
  have hlower := oneHead_original_lower_bound p hp
  linarith

end Transformer.Convexifying
