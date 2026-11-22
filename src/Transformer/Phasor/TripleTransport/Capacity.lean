module

public import Transformer.Phasor.TripleTransport

/-!
# Capacity of a learned triple-transport dictionary

Source: the dictionary capacity experiment in the “Triple transport” section
of `experiments/phasor_value_transport/README.md`. Adding unattached hidden
directions enlarges the state space of the graph. Projecting its paths back
to the original dictionary shows exactly which old connections it preserves.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport

variable {I X Y : Type*}

/-- Enlarge a fixed dictionary by slots which never occur in a trusted match. -/
def withUnused (P : I → X → Option X) (i : I) : X ⊕ Y → Option (X ⊕ Y)
  | .inl x => (P i x).map Sum.inl
  | .inr _ => none

private def dropUnused (fallback : X) : X ⊕ Y → X := Sum.elim id (fun _ => fallback)

private theorem old_match (P : I → X → Option X) (fallback : X) (i : I) (u v : X ⊕ Y)
    (h : withUnused P i u = some v) :
    u = .inl (dropUnused fallback u) ∧ v = .inl (dropUnused fallback v) ∧
      P i (dropUnused fallback u) = some (dropUnused fallback v) := by
  cases u with
  | inr u => simp [withUnused] at h
  | inl u =>
    cases hp : P i u with
    | none => simp [withUnused, hp] at h
    | some y =>
      simp [withUnused, hp] at h
      subst v
      exact ⟨rfl, rfl, hp⟩

private theorem old_edge (P : I → X → Option X) (fallback : X) (s t : Triple (X ⊕ Y))
    (h : Edge (withUnused P) s t) :
    Edge P (s.map (dropUnused fallback)) (t.map (dropUnused fallback)) := by
  obtain ⟨hs, ht, i, h₁, h₂, h₃⟩ := h
  obtain ⟨s₁, t₁, m₁⟩ := old_match P fallback i _ _ h₁
  obtain ⟨s₂, t₂, m₂⟩ := old_match P fallback i _ _ h₂
  obtain ⟨s₃, t₃, m₃⟩ := old_match P fallback i _ _ h₃
  have ne {u v : X ⊕ Y} (hu : u = .inl (dropUnused fallback u))
      (hv : v = .inl (dropUnused fallback v)) (different : u ≠ v) :
      dropUnused fallback u ≠ dropUnused fallback v :=
    fun equal => different (hu.trans ((congrArg Sum.inl equal).trans hv.symm))
  exact ⟨⟨ne s₁ s₂ hs.1, ne s₁ s₃ hs.2.1, ne s₂ s₃ hs.2.2⟩,
    ⟨ne t₁ t₂ ht.1, ne t₁ t₃ ht.2.1, ne t₂ t₃ ht.2.2⟩, i, m₁, m₂, m₃⟩

/-- Removing unused directions preserves and reflects every triple connection
between represented slots. The statement concerns a fixed bank of trusted
matches. Which directions training will actually use is an empirical question. -/
theorem connected_withUnused_iff (P : I → X → Option X) (s t : Triple X) :
    Connected (withUnused P) (s.map (Sum.inl : X → X ⊕ Y)) (t.map Sum.inl) ↔ Connected P s t := by
  constructor
  · intro path
    have project (u v : Triple (X ⊕ Y)) (h : Connected (withUnused P) u v) :
        Connected P (u.map (dropUnused s.first)) (v.map (dropUnused s.first)) := by
      induction h with
      | rel x y h => exact .rel _ _ (old_edge P s.first x y h)
      | refl => exact .refl _
      | symm _ _ _ ih => exact .symm _ _ ih
      | trans _ _ _ _ _ ih₁ ih₂ => exact .trans _ _ _ ih₁ ih₂
    exact project _ _ path
  · exact rename Sum.inl (fun _ _ h => by simpa using h) (fun _ _ => rfl)

private def cycleThree (_ : Unit) (x : Fin 3) : Option (Fin 3) := some (x + 1)

example : withUnused cycleThree () (Sum.inl 1) = some (Sum.inl (2 : Fin 3) : Fin 3 ⊕ Unit) :=
  rfl

private theorem cycleEdge : Edge cycleThree ⟨0, 1, 2⟩ ⟨1, 2, 0⟩ :=
  ⟨by unfold Triple.distinct; decide, by unfold Triple.distinct; decide, (), rfl, rfl, rfl⟩

example : Connected (withUnused cycleThree) ((Triple.mk 0 1 2).map (Sum.inl : Fin 3 → Fin 3 ⊕ Unit))
    ((Triple.mk 1 2 0).map Sum.inl) :=
  (connected_withUnused_iff cycleThree _ _).2 (.rel _ _ cycleEdge)

end Transformer.Phasor.TripleTransport
