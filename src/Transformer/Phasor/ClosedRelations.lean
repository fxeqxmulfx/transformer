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

open scoped RealInnerProductSpace
open Transformer

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

/-- Apply the common multiplier only when all winning relations select one
distinct address. Source: the closed-relation gain trial in the experiment README. -/
def consensusGain (gain factor : ℝ) (ends : Finset (Fin p)) : ℝ :=
  if ends.card = 1 then factor * gain else gain

/-- Score the normalized hidden read at the selected address through the
original tied output projection. The same address and gain apply to all classes. -/
noncomputable def consensusLogit (E : Fin p → EucSpace D) (base : EucSpace D)
    (read : Fin p → EucSpace D) (gain factor : ℝ) (ends : Finset (Fin p))
    (fallback c : Fin p) : ℝ :=
  ⟪E c, base + consensusGain gain factor ends • read (consensusAddress fallback ends)⟫

/-- On disagreement or an empty vote, the entire score vector equals the
ordinary successor read for every value of every parameter. -/
theorem consensusLogit_fallback (E : Fin p → EucSpace D) (base : EucSpace D)
    (read : Fin p → EucSpace D) (gain factor : ℝ) (ends : Finset (Fin p))
    (fallback c : Fin p) (h : ends.card ≠ 1) :
    consensusLogit E base read gain factor ends fallback c = ⟪E c, base + gain • read fallback⟫ := by
  rw [consensusLogit, consensusAddress_fallback _ _ h]
  simp [consensusGain, h]

/-- For a unique endpoint, increasing the multiplier changes its margin
over a rival by precisely the projection of the normalized read onto the
two tied codes' difference. The contribution can have either sign. -/
theorem consensusLogit_margin_change (E : Fin p → EucSpace D) (base : EucSpace D)
    (read : Fin p → EucSpace D) (gain a b : ℝ) (fallback r c : Fin p) :
    let logit := fun factor => consensusLogit E base read gain factor {r} fallback
    (logit b r - logit b c) - (logit a r - logit a c) =
      (b-a) * gain * ⟪E r - E c, read r⟫ := by
  simp [consensusLogit, consensusGain, consensusAddress_singleton, inner_add_right,
    real_inner_smul_right, inner_sub_left]
  ring

/-- One multiplier suffices on a finite collection of queries and rivals
when singleton reads favor their target and the ordinary gain already wins
the remaining comparisons. No common positive advantage is needed off the gate. -/
theorem exists_consensus_factor {ι : Type*} [Fintype ι] (ends : ι → Finset (Fin p))
    (gap advantage : ι → ℝ) (gain : ℝ)
    (hit : ∀ j, (ends j).card = 1 → 0 < gain * advantage j)
    (ordinary : ∀ j, (ends j).card ≠ 1 → gap j < gain * advantage j) :
    ∃ factor : ℝ, 0 < factor ∧ ∀ j, gap j < consensusGain gain factor (ends j) * advantage j := by
  let κ := {j : ι // (ends j).card = 1}
  obtain ⟨factor, pos, wins⟩ := exists_shared_gain (ι := κ) (fun j => gap j.val)
    (fun j => gain * advantage j.val) (fun j => hit j.val j.property)
  refine ⟨factor, pos, fun j => ?_⟩
  by_cases h : (ends j).card = 1
  · simpa [consensusGain, h, mul_assoc] using wins ⟨j, h⟩
  · simpa [consensusGain, h] using ordinary j h

example (E : Fin 3 → EucSpace 4) (base : EucSpace 4) (read : Fin 3 → EucSpace 4) :
    consensusLogit E base read (3/4) 2 {0, 2} 1 0 = ⟪E 0, base + (3/4 : ℝ) • read 1⟫ :=
  consensusLogit_fallback _ _ _ _ _ _ _ _ (by decide)

/-- The second query has a negative read advantage but already wins.
The first can overcome a positive gap using the same base gain. -/
example : ∃ t : ℝ, 0 < t ∧ ∀ j : Fin 2, (if j = 0 then 3 else -2 : ℝ) <
    consensusGain 1 t (if j = 0 then {0} else ∅ : Finset (Fin 2)) *
      (if j = 0 then 1 else -1) := by
  apply exists_consensus_factor
  · intro j
    fin_cases j <;> norm_num
  · intro j
    fin_cases j <;> norm_num

end Transformer.Phasor
