module

public import Transformer.Phasor.TripleTransport

/-!
# Indexing available images of a prototype triple

Source: “Indexing the unowned images” in the triple-transport experiment.
For a fixed row and two anchors, grouping unowned images by connected
component is independent of the current query. Its own trusted image can
be admitted separately.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.CandidateIndex

variable {Y C R : Type*}

/-- A source component has exactly the indicated admissible image. -/
def sole (P : Y → Prop) (v : Y) : Prop := P v ∧ ∀ z, P z → z = v

/-- A row's unowned candidates form a reusable bucket. Its single owned image,
if present and in the component, is disjoint and can be restored at query time. -/
theorem sole_disjoint_iff (P Q : Y → Prop) (h : ∀ y, P y → ¬ Q y) (v : Y) :
    sole (fun y => P y ∨ Q y) v ↔
      (sole P v ∧ ∀ y, ¬ Q y) ∨ ((∀ y, ¬ P y) ∧ sole Q v) := by
  constructor
  · rintro ⟨hv, unique⟩
    rcases hv with pv | qv
    · left
      refine ⟨⟨pv, fun z pz => unique z (.inl pz)⟩, ?_⟩
      intro z qz
      have eq := unique z (.inr qz)
      exact h v pv (eq ▸ qz)
    · right
      refine ⟨?_, ⟨qv, fun z qz => unique z (.inr qz)⟩⟩
      intro z pz
      have eq := unique z (.inl pz)
      exact h v (eq ▸ pz) qv
  · rintro (⟨⟨pv, unique⟩, empty⟩ | ⟨empty, ⟨qv, unique⟩⟩)
    · exact ⟨.inl pv, fun z hz => hz.elim (unique z) (fun h => (empty z h).elim)⟩
    · exact ⟨.inr qv, fun z hz => hz.elim (fun h => (empty z h).elim) (unique z)⟩

/-- A scan with the current query's value admitted selects exactly the unique
unowned image when the query adds none, or the query's matching image when its
bucket is empty. The anchor test may express any restriction on output slots. -/
theorem indexed_read_iff (owner : Y → Option R) (stored : Option Y)
    (occupied : ∀ y, stored = some y → owner y ≠ none)
    (admit : Y → Prop) (comp : Y → C) (key : C) (v : Y) :
    sole (fun y => admit y ∧ comp y = key ∧ (owner y = none ∨ stored = some y)) v ↔
    (sole (fun y => admit y ∧ comp y = key ∧ owner y = none) v ∧
        (∀ y, ¬ (admit y ∧ comp y = key ∧ stored = some y))) ∨
    ((∀ y, ¬ (admit y ∧ comp y = key ∧ owner y = none)) ∧
        sole (fun y => admit y ∧ comp y = key ∧ stored = some y) v) := by
  have split : (fun y => admit y ∧ comp y = key ∧ (owner y = none ∨ stored = some y)) =
      (fun y => (admit y ∧ comp y = key ∧ owner y = none) ∨
        (admit y ∧ comp y = key ∧ stored = some y)) := by
    funext y
    apply propext
    tauto
  rw [split]
  exact sole_disjoint_iff _ _ (fun y a b => occupied y b.2.2 a.2.2) v

example : sole (fun n : Nat => n = 3 ∨ n = 7) 3 ↔ False := by
  constructor
  · intro ⟨_, only⟩
    exact (by decide : (7 : Nat) ≠ 3) (only 7 (.inr rfl))
  · exact False.elim

private def occupied (b : Bool) : Option Unit := if b then some () else none

example : sole (fun y : Bool => y = true ∧ (occupied y = none ∨ some true = some y)) true := by
  have h := (indexed_read_iff occupied (some true) (by intro y eq; cases y <;> simp_all [occupied])
    (fun _ => True) id true true).2 (Or.inr (by
      constructor
      · rintro y ⟨_,hy,free⟩
        simp_all [occupied]
      · constructor
        · simp
        · rintro y ⟨_,hy,_⟩
          exact hy))
  simpa [id] using h

example : sole (fun b : Bool => b = true ∨ b = false) true ↔ False := by
  rw [sole_disjoint_iff _ _ (by intro b h; cases b <;> simp_all)]
  simp [sole]

end Transformer.Phasor.TripleTransport.CandidateIndex
