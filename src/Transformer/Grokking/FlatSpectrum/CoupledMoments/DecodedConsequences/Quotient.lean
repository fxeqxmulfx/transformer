module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences.Discovery

/-!
# Observed row certificates with noninjective class codes

Source: `review/flat_spectrum.md`, CM30. Actual observed row consequences still
identify a latent class code when distinct raw labels share that code. A declared
readout constant on code fibers then has a certified answer. Raw-label equality
does not follow: the finite witness observes every raw class and has two fitting
targets with different raw answers at its genuinely held-out query.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences

open CompositeDiscovery

variable {R n C S D : Type*} [CommRing R]
variable [Fintype n] [Fintype C] [DecidableEq n] [DecidableEq C] [DecidableEq S]

/-- Actual row replay identifies the unknown target's code without assuming
class-code injectivity, CM30. No unobserved label is supplied to the program. -/
theorem code_of_program {cell : S → n × n} {labels : S → C}
    (e : C → R) (r s : n → R) (y : n → n → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (ops : List (RowOp S R)) (slot : S) (i j : n) (c : C)
    (hrow : ∀ v, reduceRows ops (observedMatrix cell labels) slot v = queryRow i j c v) :
    e (y i j) = e c := by
  have hfit : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2 := by
    intro t
    rw [← hobs t]
    exact hreal _ _
  have hz := reduceRows_zero ops (observed_zero e r s hfit) slot
  simp only [rowAction, hrow, query_action, pack, Sum.elim_inl, Sum.elim_inr] at hz
  rw [hreal]
  exact sub_eq_zero.mp hz

variable [DecidableEq R]

/-- The same finite observed-only certifier is sound for codes even when its
emitted raw label is not uniquely determined, CM30. -/
theorem choose_code_sound {cell : S → n × n} {labels : S → C}
    (e : C → R) (r s : n → R) (y : n → n → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (i j : n) (bank : List (Candidate S C R)) {c : C}
    (h : choose cell labels i j bank = some c) : e (y i j) = e c := by
  obtain ⟨p, _, hc, hp⟩ := choose_certificate bank h
  rw [← hc]
  exact code_of_program e r s y hreal hobs p.program p.row i j p.label hp

/-- A fixed output quotient is identified when it is constant on code fibers,
CM30. This is an explicit compatibility hypothesis about the original readout;
it does not replace the original task's metric or ordinary raw-label CE. -/
theorem choose_quotient_sound {cell : S → n × n} {labels : S → C}
    (e : C → R) (q : C → D) (hquot : ∀ a b, e a = e b → q a = q b)
    (r s : n → R) (y : n → n → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (i j : n) (bank : List (Candidate S C R)) {c : C}
    (h : choose cell labels i j bank = some c) : q (y i j) = q c := by
  apply hquot
  exact choose_code_sound e r s y hreal hobs i j bank h

/-- Finite certificate coverage recovers every declared quotient answer under
unknown code realizability, CM30. The certifier still receives no code/readout
correctness oracle and need not recover a unique raw label. -/
theorem choose_quotient_recovers {cell : S → n × n} {labels : S → C}
    (e : C → R) (q : C → D) (hquot : ∀ a b, e a = e b → q a = q b)
    (r s : n → R) (y : n → n → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (bank : n → n → List (Candidate S C R))
    (hcover : ∀ i j, ∃ p ∈ bank i j, Passes cell labels i j p) :
    ∀ i j, (choose cell labels i j (bank i j)).map q = some (q (y i j)) := by
  intro i j
  obtain ⟨c, hc⟩ := choose_exists (bank i j) (hcover i j)
  rw [hc, Option.map_some, choose_quotient_sound e q hquot r s y hreal hobs i j (bank i j) hc]

/-- Eight categorical observations leave the lower-right cell unseen, CM30. -/
def aliasCells : Fin 8 → Fin 3 × Fin 3 :=
  ![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2)]

/-- All four raw class names occur, including both names in each code fiber. -/
def aliasLabels : Fin 8 → ZMod 2 × Bool :=
  ![(0, false), (1, true), (0, false), (1, false),
    (0, true), (1, false), (0, false), (1, true)]

/-- Repeated categorical phases; their numeric names are witness data only. -/
def aliasPhases : Fin 3 → ZMod 2 := ![0, 1, 0]

/-- Two raw targets differ only at the unobserved alias bit, CM30. -/
def aliasTarget (bit : Bool) (i j : Fin 3) : ZMod 2 × Bool :=
  (aliasPhases i + aliasPhases j, if i = 2 ∧ j = 2 then bit else decide (i = 1))

/-- All raw labels are observed, and both targets fit every observed raw label
while admitting the same noninjective additive code representation, CM30. -/
theorem alias_observed :
    (∀ c : ZMod 2 × Bool, ∃ t, aliasLabels t = c) ∧
    (∀ t, aliasCells t ≠ (2, 2)) ∧
    (∀ bit t, aliasTarget bit (aliasCells t).1 (aliasCells t).2 = aliasLabels t) ∧
    (∀ bit i j, (aliasTarget bit i j).1 = aliasPhases i + aliasPhases j) := by
  decide

/-- The witness code really collides while its quotient answers are nonconstant,
CM30; the obstruction is not a constant or uninhabited target family. -/
theorem alias_code_collides :
    ¬ Function.Injective (Prod.fst : ZMod 2 × Bool → ZMod 2) ∧
    (aliasTarget false 0 0).1 ≠ (aliasTarget false 0 1).1 := by
  decide

/-- Add the (0,2) row to (2,0), then subtract the (0,0) row, CM30. -/
def aliasProgram : List (RowOp (Fin 8) (ZMod 2)) := [.add 2 6 1, .add 2 0 (-1)]

/-- The actual original coefficients certify a raw representative of the
unseen code fiber; this check has no target-table argument, CM30. -/
theorem alias_query : ∀ v,
    reduceRows aliasProgram (observedMatrix aliasCells aliasLabels) 2 v =
      queryRow 2 2 (0, false) v := by
  decide

/-- A finite bank of actual row programs, with one genuinely unobserved query. -/
def aliasBank (i j : Fin 3) : List (Candidate (Fin 8) (ZMod 2 × Bool) (ZMod 2)) :=
  if i = 2 ∧ j = 2 then [⟨(0, false), aliasProgram, 2⟩]
  else
    let t : Fin 8 := ⟨(i.val + 3 * j.val) % 8, Nat.mod_lt _ (by decide)⟩
    [⟨aliasLabels t, [], t⟩]

/-- Every categorical query has a checked original row certificate, CM30. -/
theorem alias_coverage : ∀ i j, ∃ p ∈ aliasBank i j, Passes aliasCells aliasLabels i j p := by
  decide

/-- Both targets have exactly the same certified quotient answers at all cells,
CM30, even though the certifier does not see their missing raw alias bit. -/
theorem alias_quotient_recovers : ∀ bit i j,
    (choose aliasCells aliasLabels i j (aliasBank i j)).map Prod.fst =
      some (aliasTarget bit i j).1 := by
  intro bit
  exact choose_quotient_recovers Prod.fst Prod.fst (fun _ _ h => h)
    aliasPhases aliasPhases (aliasTarget bit) (alias_observed.2.2.2 bit)
    (alias_observed.2.2.1 bit) aliasBank alias_coverage

/-- Counterexample to replacing CM30 quotient recovery by raw-label recovery:
all raw names occur in training, both targets fit, and a certified emitted raw
representative disagrees with one target at the unseen cell. -/
theorem alias_raw_obstruction :
    aliasTarget false 2 2 ≠ aliasTarget true 2 2 ∧
    choose aliasCells aliasLabels 2 2 (aliasBank 2 2) = some (aliasTarget false 2 2) ∧
    choose aliasCells aliasLabels 2 2 (aliasBank 2 2) ≠ some (aliasTarget true 2 2) := by
  decide

example : (aliasTarget true 2 2).1 = (0 : ZMod 2) :=
  code_of_program Prod.fst aliasPhases aliasPhases (aliasTarget true)
    (alias_observed.2.2.2 true) (alias_observed.2.2.1 true) aliasProgram 2 2 2 (0, false) alias_query

example : (aliasTarget true 2 2).1 = (0 : ZMod 2) :=
  choose_code_sound Prod.fst aliasPhases aliasPhases (aliasTarget true)
    (alias_observed.2.2.2 true) (alias_observed.2.2.1 true) 2 2 (aliasBank 2 2)
    alias_raw_obstruction.2.1

example : (aliasTarget true 2 2).1 = (0 : ZMod 2) :=
  choose_quotient_sound Prod.fst Prod.fst (fun _ _ h => h) aliasPhases aliasPhases
    (aliasTarget true) (alias_observed.2.2.2 true) (alias_observed.2.2.1 true)
    2 2 (aliasBank 2 2) alias_raw_obstruction.2.1

example : ∀ i j, (choose aliasCells aliasLabels i j (aliasBank i j)).map Prod.fst =
    some (aliasTarget true i j).1 :=
  choose_quotient_recovers Prod.fst Prod.fst (fun _ _ h => h) aliasPhases aliasPhases
    (aliasTarget true) (alias_observed.2.2.2 true) (alias_observed.2.2.1 true)
    aliasBank alias_coverage

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences
