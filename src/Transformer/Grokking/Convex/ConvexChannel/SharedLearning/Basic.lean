module

public import Transformer.Grokking.Convex.ConvexChannel.Shared
public import Transformer.Grokking.Convex.ConvexChannel.Optimality
public import Transformer.Grokking.Convex.ConvexChannel.Forward.Training

/-!
# Answer loss eliminates shared routing mass through observed collisions

Source: `review/convex/moments.md`, CM73. The domain is the existing
label-independent `Feasible` set on the complete tensor `Channel Unit K V C`.
It has |K||V||C| directly trained coordinates. A fixed input-only alignment
permutes slots before the linear readout; no observed-label anchor is imposed.

The selected slot and decoder below specify representable target witnesses,
not learner inputs or restrictions on the feasible domain. Ordinary answer CE
identifies them only with the observation conditions in the next module.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.SharedLearning

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]
  [Nonempty V] [DecidableEq K] [DecidableEq C]

/-- A representable shared rule; slot and decoder occur only in its parameter
witness. The optimization domain remains the full joint polytope, CM73. -/
def selected (slot : K) (decode : V → C) : Channel Unit K V C :=
  Forward.channelOf (fun () => (slot, decode))

/-- A label-independent uniform initialization, CM73. Every joint coordinate
has the same mass; neither records nor labels are arguments. -/
noncomputable def initial : Channel Unit K V C :=
  (1 / ((Fintype.card K : ℝ) * Fintype.card C)) • (1 : Channel Unit K V C)

omit [Nonempty V] [DecidableEq K] [DecidableEq C] in
/-- The uniform initialization is feasible before any observations, CM73. -/
theorem feasible_initial [Nonempty K] [Nonempty C] :
    Feasible (initial : Channel Unit K V C) := by
  have hk : (Fintype.card K : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hc : (Fintype.card C : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    simp only [initial, Pi.smul_apply, Pi.one_apply, smul_eq_mul, mul_one]
    positivity
  · intro q k v w
    simp [initial]
  · intro q v
    simp [initial, div_eq_mul_inv, mul_inv_rev, hk, hc]

omit [Nonempty V] in
/-- Any selected slot and categorical decoder are feasible, CM73. This is
capacity of the same domain, not an observed-label constraint on training. -/
theorem feasible_selected (slot : K) (decode : V → C) : Feasible (selected slot decode) :=
  Forward.feasible_channelOf _

omit [Fintype C] [Nonempty V] in
/-- Actual linear logits of a selected parameter witness, CM73. -/
theorem logits_selected (slot : K) (decode : V → C) (D : K → V) (c : C) :
    logits (selected slot decode) () D c = if c = decode (D slot) then 1 else 0 :=
  Forward.logits_channelOf _ _ _ _

/-- A realizable shared rule minimizes ordinary CE on the original records,
CM73. Duplicate values keep distinct indices in S. -/
theorem selected_loss_le (data : S → K → V) (label : S → C)
    (slot : K) (decode : V → C) (hlabel : ∀ s, label s = decode (data s slot))
    {A : Channel Unit K V C} (hA : Feasible A) :
    answerLoss (fun _ : S => ()) data label (selected slot decode) ≤
      answerLoss (fun _ : S => ()) data label A := by
  apply Finset.sum_le_sum
  intro s _
  have hz : logits (selected slot decode) () (data s) =
      ConvexRecall.targetLogits (label s) := by
    funext c
    simp only [logits_selected, ConvexRecall.targetLogits, hlabel s]
  rw [hz]
  exact ConvexRecall.target_crossEntropy_le _ (logits_nonneg hA () (data s))
    (logits_le_one hA () (data s) _)

/-- Realizability supplies an attained CE minimum without pre-anchoring any
trained row, CM73. This theorem is not convergence of an iterative optimizer. -/
theorem selected_isMinOn (data : S → K → V) (label : S → C)
    (slot : K) (decode : V → C) (hlabel : ∀ s, label s = decode (data s slot)) :
    IsMinOn (answerLoss (fun _ : S => ()) data label) {A | Feasible A}
      (selected slot decode) := by
  intro A hA
  exact selected_loss_le data label slot decode hlabel hA

/-- Aggregate empirical optimality forces target mass one separately on every
observed record, CM73. Unit target mass is derived from CE, not assumed as a
feasibility equation. The same argument includes repeated observations. -/
theorem observed_mass_one [Nontrivial C] (data : S → K → V) (label : S → C)
    (slot : K) (decode : V → C) (hlabel : ∀ s, label s = decode (data s slot))
    {A : Channel Unit K V C} (hA : Feasible A)
    (hmin : answerLoss (fun _ : S => ()) data label A ≤
      answerLoss (fun _ : S => ()) data label (selected slot decode))
    (s : S) : logits A () (data s) (label s) = 1 := by
  have hz (i : S) : logits (selected slot decode) () (data i) =
      ConvexRecall.targetLogits (label i) := by
    funext c
    simp only [logits_selected, ConvexRecall.targetLogits, hlabel i]
  have hle : ∀ i ∈ (Finset.univ : Finset S),
      crossEntropy (logits (selected slot decode) () (data i)) (label i) ≤
        crossEntropy (logits A () (data i)) (label i) := by
    intro i _
    rw [hz]
    exact ConvexRecall.target_crossEntropy_le _ (logits_nonneg hA () (data i))
      (logits_le_one hA () (data i) _)
  have he := le_antisymm (selected_loss_le data label slot decode hlabel hA) hmin
  have hs := (Finset.sum_eq_sum_iff_of_le hle).mp he s (Finset.mem_univ s)
  rw [hz] at hs
  exact ConvexRecall.target_mass_eq_one _ (logits_nonneg hA () (data s))
    (logits_le_one hA () (data s) _) hs.symm.le

omit [DecidableEq K] in
/-- Equal values with different unit-target answers eliminate a slot's entire
joint block, including unobserved values, CM73. Value-independent routing mass
and nonnegativity propagate the observed row's zero to every other row. -/
theorem collision_zero {A : Channel Unit K V C} (hA : Feasible A)
    (D E : K → V) (t u : C) (hD : logits A () D t = 1)
    (hE : logits A () E u = 1) (k : K) (hvalue : D k = E k) (htu : t ≠ u)
    (v : V) (c : C) : A () k v c = 0 := by
  have hz (d : C) : A () k (D k) d = 0 := by
    by_cases hd : d = t
    · subst d
      rw [hvalue]
      exact atom_eq_zero hA () E u hE k t htu
    · exact atom_eq_zero hA () D t hD k d hd
  have hrow : ∑ d, A () k (D k) d = 0 := Finset.sum_eq_zero fun d _ => hz d
  have hall : ∑ d, A () k v d = 0 := (hA.2.1 () k v (D k)).trans hrow
  exact (Finset.sum_eq_zero_iff_of_nonneg (fun d _ => hA.1 () k v d)).mp
    hall c (Finset.mem_univ c)

example : Feasible (initial : Channel Unit Bool Bool Bool) := feasible_initial

example := selected_loss_le (fun _ : Unit => (id : Bool → Bool)) (fun _ => true)
  false Bool.not (fun _ => rfl) (feasible_initial : Feasible (initial : Channel Unit Bool Bool Bool))

example := selected_isMinOn (fun _ : Unit => (id : Bool → Bool)) (fun _ => true)
  false Bool.not (fun _ => rfl)

example := observed_mass_one (fun _ : Unit => (id : Bool → Bool)) (fun _ => true)
  false Bool.not (fun _ => rfl) (feasible_selected false Bool.not) le_rfl ()

example := collision_zero (feasible_selected false Bool.not) id (fun _ => true)
  true false (by simp [logits_selected]) (by simp [logits_selected]) true rfl
  (by decide) false false

end Transformer.Grokking.Convex.ConvexChannel.SharedLearning
