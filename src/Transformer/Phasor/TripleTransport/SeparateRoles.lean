module

public import Transformer.Phasor.TripleTransport

/-!
# Separate input and image triples

Source: “Separate input and image roles” in the triple-transport experiment.
The two copies of the prototype dictionary can be renamed independently.
A search for the missing third image alternates between them.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.Bipartite

variable {I X Y X' Y' Z : Type*}

/-- The source and image of each learned partial map have separate roles. -/
inductive Crossing (P : I → X → Option Y) : Triple X ⊕ Triple Y → Triple X ⊕ Triple Y → Prop where
  | pair {s : Triple X} {t : Triple Y} (hs : s.distinct) (ht : t.distinct) (i : I)
      (h₁ : P i s.first = some t.first) (h₂ : P i s.second = some t.second)
      (h₃ : P i s.third = some t.third) : Crossing P (.inl s) (.inr t)

abbrev Path (P : I → X → Option Y) := Relation.EqvGen (Crossing P)

/-- Matching source and output invariants propagate along any alternating chain. -/
theorem agrees {P : I → X → Option Y} (A : Triple X → Z) (B : Triple Y → Z)
    (consistent : ∀ {s t}, Crossing P (.inl s) (.inr t) → A s = B t)
    {s : Triple X} {t : Triple Y} (path : Path P (.inl s) (.inr t)) : A s = B t := by
  have allPaths (u v) (p : Path P u v) : Sum.elim A B u = Sum.elim A B v := by
    induction p with
    | rel a b h =>
      cases h with
      | pair hs ht i h₁ h₂ h₃ => exact consistent (.pair hs ht i h₁ h₂ h₃)
    | refl => rfl
    | symm _ _ _ ih => exact ih.symm
    | trans _ _ _ _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  exact allPaths _ _ path

/-- Forgetting the roles takes every alternating chain into the original
graph when both roles share a dictionary. -/
theorem forget {P : I → X → Option X} {u v} (path : Path P u v) :
    Connected P (Sum.elim id id u) (Sum.elim id id v) := by
  induction path with
  | rel a b h =>
    cases h with
    | pair hs ht i h₁ h₂ h₃ => exact .rel _ _ ⟨hs, ht, i, h₁, h₂, h₃⟩
  | refl => exact .refl _
  | symm _ _ _ ih => exact .symm _ _ ih
  | trans _ _ _ _ _ ih₁ ih₂ => exact .trans _ _ _ ih₁ ih₂

/-- Each role can be renamed independently. The labels of the slots
need no shared geometry across the two roles. -/
theorem rename {P : I → X → Option Y} {Q : I → X' → Option Y'}
    (f : X → X') (g : Y → Y') (hf : Function.Injective f) (hg : Function.Injective g)
    (mapped : ∀ i x, Q i (f x) = (P i x).map g) {u v} (path : Path P u v) :
    Path Q (u.map (Triple.map f) (Triple.map g)) (v.map (Triple.map f) (Triple.map g)) := by
  induction path with
  | rel a b h =>
    apply Relation.EqvGen.rel
    cases h with
    | pair hs ht i h₁ h₂ h₃ =>
      exact .pair ⟨hf.ne hs.1, hf.ne hs.2.1, hf.ne hs.2.2⟩
        ⟨hg.ne ht.1, hg.ne ht.2.1, hg.ne ht.2.2⟩ i
        (by simp only [Triple.map, mapped, h₁, Option.map_some])
        (by simp only [Triple.map, mapped, h₂, Option.map_some])
        (by simp only [Triple.map, mapped, h₃, Option.map_some])
  | refl => exact .refl _
  | symm _ _ _ ih => exact .symm _ _ ih
  | trans _ _ _ _ _ ih₁ ih₂ => exact .trans _ _ _ ih₁ ih₂

/-- If all trusted matches implement a single map, every inferred image has
the prescribed third coordinate, including paths that traverse edges backwards. -/
theorem single_image (f : X → Y) {s : Triple X} {t : Triple Y}
    (p : Path (fun (_ : Unit) x => some (f x)) (.inl s) (.inr t)) : f s.third = t.third := by
  apply agrees (fun a => f a.third) Triple.third _ p
  intro a b h
  cases h with
  | pair _ _ _ _ _ h₃ => exact Option.some.inj h₃

private def cycle (x : Fin 5) : Fin 5 := if x = 2 then 3 else if x = 3 then 4 else if x = 4 then 2 else x
private def bank (_ : Unit) (x : Fin 5) := some (cycle x)
private def a : Triple (Fin 5) := ⟨0, 1, 2⟩
private def b : Triple (Fin 5) := ⟨0, 1, 3⟩
private def c : Triple (Fin 5) := ⟨0, 1, 4⟩
private theorem ab : Crossing bank (.inl a) (.inr b) :=
  .pair (by unfold Triple.distinct a; decide) (by unfold Triple.distinct b; decide) () rfl rfl rfl
private theorem bc : Crossing bank (.inl b) (.inr c) :=
  .pair (by unfold Triple.distinct b; decide) (by unfold Triple.distinct c; decide) () rfl rfl rfl

/-- With two fixed anchors, forgetting roles can admit a strictly different
third image. In this example the original graph traverses the same map twice. -/
theorem strict_refinement : ∃ (P : Unit → Fin 5 → Option (Fin 5)) (s t : Triple (Fin 5)),
    Connected P s t ∧ ¬Path P (.inl s) (.inr t) := by
  refine ⟨bank, a, c, .trans _ _ _ (forget (.rel _ _ ab)) (forget (.rel _ _ bc)), fun p => ?_⟩
  have bad := single_image cycle p
  exact (by decide : cycle a.third ≠ c.third) bad

example : Path bank (.inl a) (.inr b) :=
  rename id id (fun _ _ h => h) (fun _ _ h => h) (P := bank) (fun _ _ => rfl) (.rel _ _ ab)


/-- Output-only relabeling preserves and reflects the entire graph search,
including candidates whose membership is tested by a reverse path. -/
theorem relabeled_path_iff (P : I → X → Option Y) (e : Y ≃ Y')
    (s : Triple X) (t : Triple Y) :
    Path (fun i x => (P i x).map e) (.inl s) (.inr (t.map e)) ↔
      Path P (.inl s) (.inr t) := by
  constructor
  · intro path
    have back := rename (P := fun i x => (P i x).map e) (Q := P) id e.symm
      (fun _ _ h => h) e.symm.injective (fun i x => by cases h : P i x <;> simp [h]) path
    cases s
    cases t
    simpa [Triple.map] using back
  · intro path
    have sent := rename (P := P) (Q := fun i x => (P i x).map e) id e
      (fun _ _ h => h) e.injective (fun _ _ => rfl) path
    cases s
    simpa [Triple.map] using sent

/-- With a fixed input triple and two output anchors, existence of a unique
graph candidate is invariant under an arbitrary output dictionary permutation. -/
theorem relabeled_unique_iff (P : I → X → Option Y) (e : Y ≃ Y') (s : Triple X) (a b : Y) :
    (∃! y, Path (fun i x => (P i x).map e) (.inl s) (.inr ⟨e a, e b, y⟩)) ↔
      ∃! y, Path P (.inl s) (.inr ⟨a, b, y⟩) := by
  constructor
  · rintro ⟨y, path, only⟩
    refine ⟨e.symm y, ?_, ?_⟩
    · apply (relabeled_path_iff P e s ⟨a,b,e.symm y⟩).1
      simpa [Triple.map] using path
    · intro z hz
      apply e.injective
      simpa using only (e z) ((relabeled_path_iff P e s ⟨a,b,z⟩).2 hz)
  · rintro ⟨y, path, only⟩
    refine ⟨e y, (relabeled_path_iff P e s ⟨a,b,y⟩).2 path, ?_⟩
    intro z hz
    have : e.symm z = y := only (e.symm z) ((relabeled_path_iff P e s ⟨a,b,e.symm z⟩).1
      (by simpa [Triple.map] using hz))
    simpa using congrArg e this


/-- The single-map hypothesis is realized by a nonidentity permutation. -/
example : cycle a.third = b.third := single_image cycle (.rel _ _ ab)

example : Connected bank a b := forget (.rel _ _ ab)

example : (fun t => cycle t.third) a = b.third := by
  apply agrees (P := bank) (fun t => cycle t.third) Triple.third _ (.rel _ _ ab)
  intro s t h
  cases h with
  | pair _ _ _ _ _ h₃ => exact Option.some.inj h₃

example (e : Fin 5 ≃ Fin 5) :
    Path (fun i x => (bank i x).map e) (.inl a) (.inr (b.map e)) :=
  (relabeled_path_iff bank e a b).2 (.rel _ _ ab)

example (e : Fin 5 ≃ Fin 5) :
    ∃! y, Path (fun i x => (bank i x).map e) (.inl a) (.inr ⟨e 0, e 1, y⟩) := by
  apply (relabeled_unique_iff bank e a 0 1).2
  refine ⟨3, .rel _ _ ab, ?_⟩
  intro y path
  exact (single_image cycle path).symm

end Transformer.Phasor.TripleTransport.Bipartite
