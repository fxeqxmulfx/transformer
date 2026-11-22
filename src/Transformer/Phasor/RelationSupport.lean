module

public import Transformer.Phasor.RelationalPaths

/-!
# Support for composition in a causal relation table

Source: the relation-support experiment in `experiments/phasor_value_transport/README.md`.
When several relations offer two-step paths, their support is the number of
distinct starting tokens on which the same relation can be followed twice.
The highest-support moving relations share the hidden update.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {α β : Type*} {p D : ℕ}

/-- Distinct sources from which a partial table gives two consecutive edges. -/
def continuingSources [Fintype α] [DecidableEq α] (R : α → Option α) : Finset α :=
  Finset.univ.filter fun a => ((R a).bind R).isSome

/-- A source is counted precisely when it has a witnessed middle and final
token. A self-loop is a valid witness for both edges. -/
theorem mem_continuingSources [Fintype α] [DecidableEq α] (R : α → Option α) (a : α) :
    a ∈ continuingSources R ↔ ∃ b c, R a = some b ∧ R b = some c := by
  cases hb : R a with
  | none => simp [continuingSources, hb]
  | some b => simp [continuingSources, hb, Option.isSome_iff_exists]

/-- Renaming the vocabulary transports the set of starting points exactly,
so relabeling preserves the cardinality used to rank relations. -/
theorem continuingSources_relabel [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (e : α ≃ β) (R : α → Option α) :
    continuingSources (fun b => (R (e.symm b)).map e) = (continuingSources R).map e.toEmbedding := by
  ext b
  obtain ⟨a, rfl⟩ := e.surjective b
  rw [Finset.mem_map_equiv]
  cases h : R a with
  | none => simp [continuingSources, h]
  | some m => cases h' : R m <;> simp [continuingSources, h, h']

/-- The source count is evaluated on the latest completed writes of each relation. -/
def relationSupport [Fintype α] [DecidableEq α] (xs : ℕ → α) (n : ℕ) (h : α) : ℕ :=
  (continuingSources (fun k => latestRelation xs n k h)).card

/-- Keep every moving relation attaining the largest score among moving reads. -/
def supportedRelations (k : Fin p) (dest : Fin p → Fin p) (score : Fin p → ℕ) : Finset (Fin p) :=
  (movingRelations k dest).filter fun h => ∀ t ∈ movingRelations k dest, score t ≤ score h

/-- If a moving read exists, at least one relation survives selection, including ties. -/
theorem supportedRelations_nonempty (k : Fin p) (dest : Fin p → Fin p) (score : Fin p → ℕ)
    (hn : (movingRelations k dest).Nonempty) : (supportedRelations k dest score).Nonempty := by
  obtain ⟨h, hm, max⟩ := Finset.exists_max_image (movingRelations k dest) score hn
  exact ⟨h, Finset.mem_filter.mpr ⟨hm, max⟩⟩

/-- A strict support advantage leaves exactly one relation, regardless of
the number of competing paths or their endpoints. -/
theorem supportedRelations_unique (k winner : Fin p) (dest : Fin p → Fin p) (score : Fin p → ℕ)
    (hw : winner ∈ movingRelations k dest)
    (strict : ∀ h ∈ movingRelations k dest, h ≠ winner → score h < score winner) :
    supportedRelations k dest score = {winner} := by
  ext h
  rw [Finset.mem_singleton]
  constructor
  · intro hh
    obtain ⟨hm, max⟩ := Finset.mem_filter.mp hh
    by_contra ne
    exact (not_lt_of_ge (max winner hw)) (strict h hm ne)
  · rintro rfl
    refine Finset.mem_filter.mpr ⟨hw, fun t ht => ?_⟩
    by_cases eq : t = h
    · simp [eq]
    · exact (strict t ht eq).le

/-- An ineligible relation returns the current token and contributes zero
to the ordinary mean read. -/
def supportedEnd (k : Fin p) (dest : Fin p → Fin p) (score : Fin p → ℕ) (h : Fin p) : Fin p :=
  if h ∈ supportedRelations k dest score then dest h else k

/-- The moving endpoints of the selected read are precisely its maximizers. -/
theorem moving_supportedEnd (k : Fin p) (dest : Fin p → Fin p) (score : Fin p → ℕ) :
    movingRelations k (supportedEnd k dest score) = supportedRelations k dest score := by
  ext h
  change h ∈ Finset.univ.filter (fun t => supportedEnd k dest score t ≠ k) ↔ _
  rw [Finset.mem_filter]
  by_cases hm : h ∈ supportedRelations k dest score
  · have moved : dest h ≠ k := (Finset.mem_filter.mp (Finset.mem_filter.mp hm).1).2
    simp [supportedEnd, hm, moved]
  · simp [supportedEnd, hm]

/-- With unit amplitudes, a unique maximal relation gives exactly its full
tied displacement for any learned embedding and any vocabulary size. -/
theorem supportMean_unique (k winner : Fin p) (dest : Fin p → Fin p) (score : Fin p → ℕ)
    (E : Fin p → EucSpace D) (hw : winner ∈ movingRelations k dest)
    (strict : ∀ h ∈ movingRelations k dest, h ≠ winner → score h < score winner) :
    relationMeanOffset E k (supportedEnd k dest score) (fun _ => 1) = E (dest winner) - E k := by
  have hs := supportedRelations_unique k winner dest score hw strict
  simp only [relationMeanOffset, moving_supportedEnd, hs]
  simp [supportedEnd, hs]

/-- Here the two writes with middle token zero are 2→3 and 1→2. Only
the latter source can continue for a second edge. -/
example :
    let xs := fun j => (![2,0,3,4,1,0,2,1] : Fin 8 → Fin 5) (Fin.ofNat 8 j)
    relationSupport xs 5 0 = 1 := by decide

example : (supportedRelations (0 : Fin 3) (fun _ => 1) Fin.val).Nonempty :=
  supportedRelations_nonempty _ _ _ (by decide)

example : supportedRelations (0 : Fin 3) (fun _ => 1) Fin.val = {2} :=
  supportedRelations_unique _ _ _ _ (by decide) (by decide)

example (E : Fin 3 → EucSpace D) :
    relationMeanOffset E 0 (supportedEnd 0 (fun _ => 1) Fin.val) (fun _ => 1) = E 1 - E 0 :=
  supportMean_unique _ 2 _ _ _ (by decide) (by decide)

end Transformer.Phasor
