module

public import Transformer.Phasor.RelationSupport

/-!
# A single value address from closed prefix relations

Source: the closed-relation trial in `experiments/phasor_value_transport/README.md`.
A partial relation with distinct destinations and a continuation from every
written source permutes its written domain. The read uses two steps of such
a permutation when the largest eligible relations agree on one endpoint.
Otherwise it uses the ordinary successor. Both cases have one tied value read.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α : Type*} {p D : ℕ}

/-- Count every written source of a partial prefix relation, including loops. -/
def writtenSources [Fintype α] (R : α → Option α) : Finset α :=
  Finset.univ.filter fun a => (R a).isSome

/-- The streaming count from `continuingSources_update` counts precisely
the written sources whose destination is also written. -/
theorem continuingSources_eq_filter [Fintype α] [DecidableEq α] (R : α → Option α) :
    continuingSources R = (writtenSources R).filter (fun a => (R a).getD a ∈ writtenSources R) := by
  ext a
  cases ha : R a with
  | none => simp [continuingSources, writtenSources, ha]
  | some b => simp [continuingSources, writtenSources, ha]

/-- Equality of both streaming counts is equivalent to a bijection on the
written domain. This lets a reader recognize a finite union of cycles
without following a path around each cycle. -/
theorem closed_counts_iff [DecidableEq α] (s : Finset α) (f : α → α) :
    (s.filter (fun a => f a ∈ s)).card = s.card ∧ (s.image f).card = s.card ↔
      Set.BijOn f (s : Set α) s := by
  constructor
  · rintro ⟨hc, hd⟩
    have closed := Finset.filter_eq_self.mp
      (Finset.eq_of_subset_of_card_le (Finset.filter_subset _ _) hc.ge)
    have eq : s.image f = s := Finset.eq_of_subset_of_card_le (by
      intro b hb
      obtain ⟨a, ha, rfl⟩ := Finset.mem_image.mp hb
      exact closed a ha) hd.ge
    refine ⟨fun a ha => closed a ha, Finset.card_image_iff.mp hd, ?_⟩
    intro b hb
    exact Finset.mem_image.mp (eq.symm ▸ (show b ∈ s from hb))
  · rintro ⟨maps, inj, _⟩
    exact ⟨congrArg Finset.card (Finset.filter_eq_self.mpr (fun _ ha => maps ha)),
      Finset.card_image_iff.mpr inj⟩

/-- At a written query, a closed relation contributes its two-step endpoint.
Other relations return the query and so cannot win a moving read. -/
def closedPathEnd (s : Finset (Fin p)) (f : Fin p → Fin p) (k : Fin p) : Fin p :=
  if k ∈ s ∧ (s.filter (fun a => f a ∈ s)).card = s.card ∧ (s.image f).card = s.card
  then f (f k) else k

/-- Every query in a closed relation gets its actual composed endpoint. -/
theorem closedPathEnd_of_bijOn (s : Finset (Fin p)) (f : Fin p → Fin p) (k : Fin p)
    (hk : k ∈ s) (h : Set.BijOn f (s : Set (Fin p)) s) :
    closedPathEnd s f k = f (f k) := by
  simp [closedPathEnd, hk, (closed_counts_iff s f).mpr h]

/-- Use an endpoint only when the set of distinct proposals has cardinality one. -/
def consensusAddress (fallback : Fin p) (ends : Finset (Fin p)) : Fin p :=
  if h : ends.card = 1 then ends.min' (Finset.card_pos.mp (by omega)) else fallback

/-- A unique proposal changes the address, independently of the fallback. -/
theorem consensusAddress_singleton (fallback r : Fin p) :
    consensusAddress fallback {r} = r := by
  simp [consensusAddress]

/-- No proposal or a disagreement among the largest relations preserves the successor. -/
theorem consensusAddress_fallback (fallback : Fin p) (ends : Finset (Fin p))
    (h : ends.card ≠ 1) : consensusAddress fallback ends = fallback := by
  simp [consensusAddress, h]

/-- The largest closed relations vote by endpoint. A single resolved address
feeds both the learned successor table and the tied embedding displacement. -/
def closedRelationAddress (k fallback : Fin p) (s : Fin p → Finset (Fin p))
    (f : Fin p → Fin p → Fin p) : Fin p :=
  let dest := fun h => closedPathEnd (s h) (f h) k
  consensusAddress fallback ((supportedRelations k dest (fun h => (s h).card)).image dest)

/-- When the strongest eligible relation is unique, the only address read
at this query is its endpoint. The result holds for every fallback value. -/
theorem closedRelationAddress_unique (k fallback winner : Fin p) (s : Fin p → Finset (Fin p))
    (f : Fin p → Fin p → Fin p)
    (hw : winner ∈ movingRelations k (fun h => closedPathEnd (s h) (f h) k))
    (strict : ∀ h ∈ movingRelations k (fun h => closedPathEnd (s h) (f h) k),
      h ≠ winner → (s h).card < (s winner).card) :
    closedRelationAddress k fallback s f = closedPathEnd (s winner) (f winner) k := by
  dsimp only [closedRelationAddress]
  rw [supportedRelations_unique _ winner _ _ hw strict, Finset.image_singleton,
    consensusAddress_singleton]

/-- A three-cycle supplies a nontrivial moving two-step endpoint. -/
example : closedPathEnd (Finset.univ : Finset (Fin 3)) (fun x => x + 1) 0 = 2 ∧
    Set.BijOn (fun x : Fin 3 => x + 1) Set.univ Set.univ := by
  constructor
  · decide
  · rw [← Finset.coe_univ, ← closed_counts_iff]
    decide

/-- Two different endpoints leave even a third, distinct fallback address intact. -/
example : consensusAddress (1 : Fin 3) {0, 2} = 1 :=
  consensusAddress_fallback _ _ (by decide)

/-- Just one relation contains a three-cycle, so its twice-iterated address
wins even when the ordinary successor points to the intermediate token. -/
example :
    let s := fun h : Fin 3 => if h = 0 then Finset.univ else ∅
    let f := fun (_ : Fin 3) (k : Fin 3) => k + 1
    closedRelationAddress 0 1 s f = 2 := by
  decide

end Transformer.Phasor
