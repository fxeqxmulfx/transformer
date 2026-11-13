module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences.Basic

/-!
# A finite observed-only row-consequence certifier

Source: `review/flat_spectrum.md`, CM29. A candidate contains only a proposed
class, an actual finite row program and a row position. The certifier compares
every replayed coefficient with the proposed query row. It receives no true
class code, target table, solver rank or cyclicity assumption.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences

open CompositeDiscovery

variable {R n C S : Type*} [CommRing R]
variable [Fintype n] [Fintype C] [DecidableEq n] [DecidableEq C] [DecidableEq S]

/-- Finite proposed data; no correctness premise is hidden in a field, CM29. -/
structure Candidate (S C R : Type*) where
  label : C
  program : List (RowOp S R)
  row : S

/-- The actual replayed coefficient test for a candidate query, CM29. -/
def Passes (cell : S → n × n) (labels : S → C) (i j : n) (p : Candidate S C R) : Prop :=
  ∀ v, reduceRows p.program (observedMatrix cell labels) p.row v = queryRow i j p.label v

variable [DecidableEq R]

instance passesDecidable (cell : S → n × n) (labels : S → C) (i j : n)
    (p : Candidate S C R) : Decidable (Passes cell labels i j p) := by
  unfold Passes
  infer_instance

/-- Choose only by finite observed-matrix replay, otherwise refuse, CM29. -/
def choose (cell : S → n × n) (labels : S → C) (i j : n) :
    List (Candidate S C R) → Option C
  | [] => none
  | p :: rest => if Passes cell labels i j p then some p.label else choose cell labels i j rest

/-- Every emitted answer is supported by an actual passing row program,
CM29. The conclusion concerns certificate data only. -/
theorem choose_certificate {cell : S → n × n} {labels : S → C} {i j : n}
    (bank : List (Candidate S C R)) {c : C} (h : choose cell labels i j bank = some c) :
    ∃ p ∈ bank, p.label = c ∧ Passes cell labels i j p := by
  induction bank with
  | nil => simp [choose] at h
  | cons p rest ih =>
    by_cases hp : Passes cell labels i j p
    · have he : p.label = c := by simpa [choose, hp] using h
      exact ⟨p, by simp, he, hp⟩
    · obtain ⟨q, hq, hc, hpass⟩ := ih (by simpa [choose, hp] using h)
      exact ⟨q, by simp [hq], hc, hpass⟩

/-- A passing candidate anywhere in the bank prevents refusal, CM29. -/
theorem choose_exists {cell : S → n × n} {labels : S → C} {i j : n}
    (bank : List (Candidate S C R)) (h : ∃ p ∈ bank, Passes cell labels i j p) :
    ∃ c, choose cell labels i j bank = some c := by
  induction bank with
  | nil => simp at h
  | cons p rest ih =>
    by_cases hp : Passes cell labels i j p
    · exact ⟨p.label, by simp [choose, hp]⟩
    · obtain ⟨q, hq, hpass⟩ := h
      have hm : q ∈ rest := by
        rcases List.mem_cons.mp hq with hqp | hqr
        · subst q
          exact False.elim (hp hpass)
        · exact hqr
      obtain ⟨c, hc⟩ := ih ⟨q, hm, hpass⟩
      exact ⟨c, by simpa [choose, hp] using hc⟩

/-- Soundness for every finite proposed bank under unknown structural
realizability, CM29. No correct code or cyclicity is supplied to `choose`. -/
theorem choose_sound {cell : S → n × n} {labels : S → C}
    (e : C → R) (hinj : Function.Injective e) (r s : n → R) (y : n → n → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (i j : n) (bank : List (Candidate S C R)) {c : C}
    (h : choose cell labels i j bank = some c) : y i j = c := by
  obtain ⟨p, _, hc, hp⟩ := choose_certificate bank h
  rw [← hc]
  exact decoded_of_program e hinj r s y hreal hobs p.program p.row i j p.label hp

/-- Observable coverage by actual row certificates identifies the full
decoded table without unique code coordinates, CM29. -/
theorem choose_recovers {cell : S → n × n} {labels : S → C}
    (e : C → R) (hinj : Function.Injective e) (r s : n → R) (y : n → n → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (bank : n → n → List (Candidate S C R))
    (hcover : ∀ i j, ∃ p ∈ bank i j, Passes cell labels i j p) :
    ∀ i j, choose cell labels i j (bank i j) = some (y i j) := by
  intro i j
  obtain ⟨c, hc⟩ := choose_exists (bank i j) (hcover i j)
  rw [hc, choose_sound e hinj r s y hreal hobs i j (bank i j) hc]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences
