module

public import Transformer.Phasor.TripleTransport.RoleWitnesses

/-!
# Sharing exact signatures in an overlap search

Source: “Sharing exact witness signatures” in the triple-transport experiment.
A signature is the set of observed ordered outputs of two hidden contexts.
Equal eligible signatures can share postings and pooled values. The original
context pairs continue to supply distinct votes.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.RelativeWitnesses

variable {R S V : Type*}

/-- On the eligible context pairs, any graph defined solely by their
signatures can use one representative per distinct signature. Repeated
signatures must still count as distinct contexts when checking support. -/
theorem representative_paths (link : V → V → Prop) (seen : R → V) (stored : S → V)
    (repr : R → S) (exampleOf : S → R)
    (save : ∀ r, stored (repr r) = seen r) (load : ∀ s, seen (exampleOf s) = stored s)
    (eligible : ∀ r, link (seen r) (seen r)) (a b : R) :
    Relation.EqvGen (fun i j => link (seen i) (seen j)) a b ↔
      Relation.EqvGen (fun i j => link (stored i) (stored j)) (repr a) (repr b) := by
  constructor
  · intro h
    induction h with
    | rel i j h => exact .rel _ _ (by simpa [save] using h)
    | refl _ => exact .refl _
    | symm _ _ _ ih => exact .symm _ _ ih
    | trans _ _ _ _ _ left right => exact .trans _ _ _ left right
  · intro h
    have lift {p q : S} (h : Relation.EqvGen (fun i j => link (stored i) (stored j)) p q) :
        Relation.EqvGen (fun i j => link (seen i) (seen j)) (exampleOf p) (exampleOf q) := by
      induction h with
      | rel i j h => exact .rel _ _ (by simpa [load] using h)
      | refl _ => exact .refl _
      | symm _ _ _ ih => exact .symm _ _ ih
      | trans _ _ _ _ _ left right => exact .trans _ _ _ left right
    have joins (r : R) : Relation.EqvGen (fun i j => link (seen i) (seen j)) r (exampleOf (repr r)) :=
      .rel _ _ (by simpa [load, save] using eligible r)
    exact .trans _ _ _ (joins a) (.trans _ _ _ (lift h) (.symm _ _ (joins b)))

/-- The relative and absolute overlap conditions both connect a signature
of sufficient size to itself. Thus exact duplicates may be interned even
when a different observed signature has the same indexing key. -/
theorem linked_self {Y : Type*} [DecidableEq Y] (S : Finset Y) (k : ℕ) (eligible : k ≤ S.card) :
    linked k 2 S S := by
  simp only [linked, Finset.inter_self, min_self]
  exact ⟨eligible, by omega⟩

/-- Searching the representative graph gives the same union of candidate
values at a key. The contents may be any predicate on a signature, so the
claim also applies to an intersection of answers obtained from different
peer lookups. Original context pairs still determine the support count. -/
theorem representative_pool {Y : Type*} (link : V → V → Prop) (seen : R → V) (stored : S → V)
    (repr : R → S) (exampleOf : S → R)
    (save : ∀ r, stored (repr r) = seen r) (load : ∀ s, seen (exampleOf s) = stored s)
    (eligible : ∀ r, link (seen r) (seen r)) (contents : V → Y → Prop) (a : R) (y : Y) :
    (∃ r, Relation.EqvGen (fun i j => link (seen i) (seen j)) a r ∧ contents (seen r) y) ↔
    (∃ s, Relation.EqvGen (fun i j => link (stored i) (stored j)) (repr a) s ∧
      contents (stored s) y) := by
  have paths := representative_paths link seen stored repr exampleOf save load eligible a
  constructor
  · rintro ⟨r, path, member⟩
    exact ⟨repr r, (paths r).1 path, (save r).symm ▸ member⟩
  · rintro ⟨s, path, member⟩
    have last : link (stored s) (stored (repr (exampleOf s))) := by
      simpa only [save, load] using (eligible (exampleOf s))
    exact ⟨exampleOf s, (paths _).2 (.trans _ _ _ path (.rel _ _ last)), (load s).symm ▸ member⟩

private def sample (b : Bool) : Finset ℕ := (Finset.range 7).image (fun i => i + if b then 8 else 0)

example : Relation.EqvGen (fun i j : Bool × Bool => linked 6 2 (sample i.1) (sample j.1))
    (false,true) (false,false) := by
  apply (representative_paths (linked 6 2) (sample ∘ Prod.fst) sample Prod.fst (fun b => (b,true))
    (fun _ => rfl) (fun _ => rfl) _ (false,true) (false,false)).2 (.refl _)
  intro r
  apply linked_self
  cases r with
  | mk a b => cases a <;> cases b <;> decide

example : ∃ r : Bool × Bool,
    Relation.EqvGen (fun i j => linked 6 2 (sample i.1) (sample j.1)) (true, false) r ∧
    11 ∈ sample r.1 := by
  apply (representative_pool (linked 6 2) (sample ∘ Prod.fst) sample Prod.fst (fun b => (b, true))
    (fun _ => rfl) (fun _ => rfl) _ (fun sig y => y ∈ sig) (true, false) 11).2
  · exact ⟨true, .refl _, by decide⟩
  · rintro ⟨a, b⟩
    apply linked_self
    cases a <;> cases b <;> decide

/-- Two separate votes can have the same representative. Checking support
on the retained representatives would lose a valid two-witness answer. -/
example : (true, false) ≠ (true, true) ∧
    (Prod.fst (true, false)) = Prod.fst (true, true) := by decide

end Transformer.Phasor.TripleTransport.RelativeWitnesses
