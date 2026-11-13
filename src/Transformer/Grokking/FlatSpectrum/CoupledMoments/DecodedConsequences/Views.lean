module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences.Witness
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Views

/-!
# Decoded row certificates within a fixed categorical view

Source: `review/flat_spectrum.md`, CM29, fixed categorical views for decoded
consequences. Each interior query uses its actual finite row certificate.
The full target is represented by the declared maps and constant strips.
Interior class names map into a separate output type, so the strip constant
can be absent from the interior classes, as it is in the native strip models.
Strip structure is a model hypothesis; interior replay does not establish it.
No cyclicity, unique code coordinates or unknown answers enter the certifier.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences

open CompositeDiscovery

variable {R n m C D S : Type*} [CommRing R]
variable [Fintype m] [Fintype C] [DecidableEq m] [DecidableEq C]
variable [DecidableEq S] [DecidableEq R]

/-- Run the finite observed-only certifier inside the declared view, CM29.
The strip constant is part of the view, not inferred from interior equations. -/
def chooseView (cell : S → m × m) (labels : S → C)
    (bank : m → m → List (Candidate S C R))
    (encode : C → D) (row column : n → Option m) (constant : D) (i j : n) : Option D :=
  match row i, column j with
  | some a, some b => (choose cell labels a b (bank a b)).map encode
  | _, _ => some constant

/-- Every returned full-view answer is sound under unknown injective-code
interior realizability and the declared strip structure, CM29. A partial
certificate bank may refuse some interior queries without changing this result. -/
theorem chooseView_sound {cell : S → m × m} {labels : S → C}
    (e : C → R) (hinj : Function.Injective e) (r s : m → R) (y : m → m → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (bank : m → m → List (Candidate S C R))
    (encode : C → D) (row column : n → Option m) (constant : D) (i j : n) {d : D}
    (h : chooseView cell labels bank encode row column constant i j = some d) :
    viewTable row column constant (fun a b => encode (y a b)) i j = d := by
  cases hr : row i with
  | none =>
    have hc : constant = d := Option.some_inj.mp (by simpa only [chooseView, hr] using h)
    simpa only [viewTable, hr] using hc
  | some a =>
    cases hs : column j with
    | none =>
      have hc : constant = d := Option.some_inj.mp
        (by simpa only [chooseView, hr, hs] using h)
      simpa only [viewTable, hr, hs] using hc
    | some b =>
      have hm : (choose cell labels a b (bank a b)).map encode = some d := by
        simpa only [chooseView, hr, hs] using h
      obtain ⟨c, hc, he⟩ := Option.map_eq_some_iff.mp hm
      have hy : y a b = c := choose_sound e hinj r s y hreal hobs a b (bank a b) hc
      simpa only [viewTable, hr, hs, hy] using he

/-- Actual finite certificate coverage of the interior recovers the full
target represented by this fixed categorical view, CM29. Unknown strip values
are covered by the stated target family, not by an additional observed row. -/
theorem chooseView_recovers {cell : S → m × m} {labels : S → C}
    (e : C → R) (hinj : Function.Injective e) (r s : m → R) (y : m → m → C)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (bank : m → m → List (Candidate S C R))
    (hcover : ∀ i j, ∃ p ∈ bank i j, Passes cell labels i j p)
    (encode : C → D) (row column : n → Option m) (constant : D) :
    ∀ i j, chooseView cell labels bank encode row column constant i j =
      some (viewTable row column constant (fun a b => encode (y a b)) i j) := by
  intro i j
  cases hr : row i with
  | none => simp only [chooseView, viewTable, hr]
  | some a =>
    cases hs : column j with
    | none => simp only [chooseView, viewTable, hr, hs]
    | some b =>
      have h := choose_recovers e hinj r s y hreal hobs bank hcover a b
      simp only [chooseView, viewTable, hr, hs, h, Option.map_some]

/-- The noncyclic witness with a fifth input name and a fifth output label.
`some c` embeds an interior class; `none` is a distinct strip output label. -/
def fourView (i j : Option (Fin 4)) : Option (Option (ZMod 4)) :=
  chooseView fourCells fourLabels fourBank some id id none i j

/-- The fixed-view certifier recovers this full nonconstant target, including
the genuinely held-out interior cell, despite a noncyclic joint module, CM29. -/
theorem four_view_recovers : ∀ i j,
    fourView i j = some (viewTable id id none (fun a b => some (fourTarget a b)) i j) :=
  chooseView_recovers id (fun _ _ h => h) fourRows fourColumns fourTarget
    (fun _ _ => rfl) four_observed.2.2 fourBank four_coverage some id id none

/-- The strip answer and the missing interior answer follow from their
respective view hypothesis and actual row certificate, CM29. -/
theorem four_view_answers :
    (∀ c : ZMod 4, (none : Option (ZMod 4)) ≠ some c) ∧
    (fourView none (some 2), fourView (some 3) (some 3)) =
      (some none, some (some 1)) := by
  refine ⟨?_, ?_⟩
  · intro c h
    cases h
  · rw [four_view_recovers, four_view_recovers]
    decide

example : viewTable (id : Option (Fin 4) → Option (Fin 4)) id none
    (fun a b => some (fourTarget a b)) (some 3) (some 3) = some 1 :=
  chooseView_sound id (fun _ _ h => h) fourRows fourColumns fourTarget
    (fun _ _ => rfl) four_observed.2.2 fourBank some id id none (some 3) (some 3)
    (by change fourView (some 3) (some 3) = some (some 1)
        rw [four_view_recovers]
        decide)

example : ∀ i j : Option (Fin 4),
    chooseView fourCells fourLabels fourBank some id id none i j =
      some (viewTable id id none (fun a b => some (fourTarget a b)) i j) :=
  chooseView_recovers id (fun _ _ h => h) fourRows fourColumns fourTarget
    (fun _ _ => rfl) four_observed.2.2 fourBank four_coverage some id id none

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences
