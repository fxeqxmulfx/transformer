module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Normalization

/-!
# Score-preserving reconstruction in the original learned family

Source: `review/flat_spectrum.md`, CM104. Weighted rows have variable component
masses. Normalization restores an actual bounded recurrence mixture, with
explicit unused capacity and unchanged complete learned state execution.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair
open scoped BigOperators
variable {J O : Type*} [Fintype J] [Fintype O] [Nonempty O]

/-- Actual joint logits with native weights already absorbed in the rows, CM104. -/
noncomputable def jointScore (f : J → Parameters O) :
    (J → O → State) →ₗ[ℝ] (List O → O → ℝ) := scoreMap 1 (fun _ => 1) f

omit [Fintype O] [Nonempty O] in
/-- Native weight absorption preserves every actual sequence/class score, CM104. -/
theorem absorb_scores (w : J → ℝ) (f : J → Parameters O) (u : J → O → State) :
    jointScore f (fun j c => w j • u j c) = ∑ j, w j • score (replace (f j) (u j)) := by
  funext xs c
  simp only [jointScore, Optimality.scores_pairing, one_mul, pairing_scale,
    Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [score, replace_complex, Optimality.pairing, LinearMap.coe_mk, AddHom.coe_mk]

/-- Normalization exactly preserves the capacity-weighted score, CM104. -/
theorem restore_score (f : Parameters O) (u : O → State) (xs : List O) (c : O) :
    blockSize u * score (replace f (normalized u)) xs c = score (replace f u) xs c := by
  have h : blockSize u • normalized u = u := funext (restore u)
  rw [← score_smul, h]

/-- Explicit normalized masses, including unused capacity, CM104. -/
noncomputable def weights (B : ℝ) (u : J → O → State) : Option J → ℝ
  | none => 1 - size u / B
  | some j => blockSize (u j) / B

/-- All reconstructed masses are nonnegative for a feasible positive budget, CM104. -/
theorem weights_nonneg (B : ℝ) (hB : 0 < B) (u : J → O → State)
    (hu : u ∈ jointDomain B) (k : Option J) : 0 ≤ weights B u k := by
  cases k with
  | none =>
    have h := (div_le_one hB).mpr hu
    change 0 ≤ 1 - size u / B
    linarith
  | some j => exact div_nonneg (block_nonneg (u j)) hB.le

/-- Explicit unused mass makes the total normalized mass exactly one, CM104. -/
theorem weights_sum (B : ℝ) (u : J → O → State) : ∑ k, weights B u k = 1 := by
  rw [Fintype.sum_option]
  simp only [weights, ← Finset.sum_div, size]
  ring

/-- Original learned content with complete normalized readouts and a zero
component for unused capacity, CM104. -/
noncomputable def components (f : J → Parameters O) (u : J → O → State) :
    Option J → Parameters O
  | none => zeroParameters
  | some j => replace (f j) (normalized (u j))

omit [Fintype J] in
/-- Every reconstructed component obeys the original full model bounds, CM104. -/
theorem components_unit (f : J → Parameters O) (hf : ∀ j, Unit (f j))
    (u : J → O → State) (k : Option J) : Unit (components f u k) := by
  cases k with
  | none => exact zero_unit
  | some j => exact replace_unit (f j) (hf j) _ (normalized_bound (u j))

/-- Explicit complete reconstruction agrees on every immutable-object sequence,
including all zero blocks and unused mass, CM104. -/
theorem reconstruct_scores (B : ℝ) (hB : 0 < B) (f : J → Parameters O)
    (u : J → O → State) :
    (∑ k, weights B u k • (B • score (components f u k))) = jointScore f u := by
  funext xs c
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Fintype.sum_option,
    weights, components, zero_score, mul_zero, zero_add, jointScore, scoreMap,
    LinearMap.coe_mk, AddHom.coe_mk, one_mul]
  apply Finset.sum_congr rfl
  intro j hj
  rw [← mul_assoc, div_mul_cancel₀ _ (ne_of_gt hB), restore_score]

/-- The full shared-capacity coordinate domain embeds in the original bounded
learned function family, CM104, not a new prescribed feature architecture. -/
theorem joint_feasible (B : ℝ) (hB : 0 < B) (f : J → Parameters O)
    (hf : ∀ j, Unit (f j)) (u : J → O → State) (hu : u ∈ jointDomain B) :
    jointScore f u ∈ domain O B := by
  have h := finite_feasible B (weights B u) (components f u)
    (weights_nonneg B hB u hu) (weights_sum B u) (components_unit f hf u)
  rwa [reconstruct_scores B hB] at h

example : 0 ≤ weights 16 (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0)) none :=
  weights_nonneg _ (by norm_num) _
    (by change size _ ≤ 16; simp [size, block_const, rowSize]) none

example : Unit (components (fun _ : Fin 1 => positionWitness)
    (fun _ (_ : Bool) => (2 * Complex.I, 0)) (some 0)) :=
  components_unit _ (fun _ => position_unit) _ _

example : (∑ k, weights 16 (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0)) k •
    ((16 : ℝ) • score (components (fun _ => positionWitness)
      (fun _ (_ : Bool) => (Complex.I, 0)) k))) =
    jointScore (fun _ : Fin 1 => positionWitness) (fun _ _ => (Complex.I, 0)) :=
  reconstruct_scores _ (by norm_num) _ _

example : jointScore (fun _ : Fin 1 => positionWitness)
    (fun _ (_ : Bool) => (Complex.I, 0)) ∈ domain Bool 16 :=
  joint_feasible _ (by norm_num) _ (fun _ => position_unit) _
    (by change size _ ≤ 16; simp [size, block_const, rowSize])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
