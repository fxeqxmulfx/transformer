module

public import Transformer.Phasor.TripleTransport

/-!
# Updating the graph when a learned partial map grows

Source: the incremental graph calculation in the “Triple transport” section
of `experiments/phasor_value_transport/README.md`. A surviving trusted pair
can be reused across optimizer updates. When a graph row loses a pair or
ceases to be injective, the implementation rebuilds the graph.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport

variable {I X : Type*}

/-- Source: the learned partial maps in the experiment's “Triple transport”
section. A newly supported triple has at least one correspondence absent
from the same context in the previous bank. -/
def AddedEdge (P Q : I → X → Option X) (s t : Triple X) : Prop :=
  s.distinct ∧ t.distinct ∧ ∃ i,
    (Q i s.first = some t.first ∧ Q i s.second = some t.second ∧ Q i s.third = some t.third) ∧
    ¬ (P i s.first = some t.first ∧ P i s.second = some t.second ∧ P i s.third = some t.third)

/-- Source: “Triple transport”, updating the graph of trusted correspondences.
When all old matches survive, retaining the old components and adjoining
only triples with a new match gives the full new equivalence relation. -/
theorem connected_growing_bank {P Q : I → X → Option X}
    (retained : ∀ i x y, P i x = some y → Q i x = some y) (s t : Triple X) :
    Connected Q s t ↔ Relation.EqvGen (fun a b => Connected P a b ∨ AddedEdge P Q a b) s t := by
  classical
  constructor
  · apply Relation.EqvGen.mono
    intro a b edge
    obtain ⟨ha, hb, i, hQ⟩ := edge
    by_cases old : P i a.first = some b.first ∧ P i a.second = some b.second ∧ P i a.third = some b.third
    · exact Or.inl (.rel _ _ ⟨ha, hb, i, old⟩)
    · exact Or.inr ⟨ha, hb, i, hQ, old⟩
  · intro path
    induction path with
    | rel a b h =>
      rcases h with old | ⟨ha, hb, i, hQ, _⟩
      · apply Relation.EqvGen.mono _ _ _ old
        rintro u v ⟨hu, hv, i, h₁, h₂, h₃⟩
        exact ⟨hu, hv, i, retained _ _ _ h₁, retained _ _ _ h₂, retained _ _ _ h₃⟩
      · exact .rel _ _ ⟨ha, hb, i, hQ⟩
    | refl => exact .refl _
    | symm _ _ _ ih => exact .symm _ _ ih
    | trans _ _ _ _ _ ih₁ ih₂ => exact .trans _ _ _ ih₁ ih₂

private def partialId (_ : Unit) (x : Fin 4) : Option (Fin 4) :=
  if x = 0 then none else some x

example (s t : Triple (Fin 4)) :
    Connected (fun (_ : Unit) x => some x) s t ↔
      Relation.EqvGen (fun a b => Connected partialId a b ∨
        AddedEdge partialId (fun _ x => some x) a b) s t := by
  apply connected_growing_bank
  intro i x y h
  by_cases hx : x = 0 <;> simp [partialId, hx] at h ⊢
  exact h

example : AddedEdge partialId (fun _ x => some x) ⟨1, 2, 0⟩ ⟨1, 2, 0⟩ := by
  refine ⟨by unfold Triple.distinct; decide, by unfold Triple.distinct; decide, (), ⟨rfl, rfl, rfl⟩, ?_⟩
  decide

end Transformer.Phasor.TripleTransport
