module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.CrossEntropy

/-!
# Finite phase domains for an arbitrary fixed categorical decoder

Source: `review/flat_spectrum.md`, CM19. Instead of the two-element fibers of CM18,
retain sets of possible phases and prune unsupported values using observed labels.
The definitions use only finite data and decidable comparisons. A fixed number
of sweeps is an explicit finite computation; no completeness of pruning is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Finset

variable {q : ℕ} {n S C : Type*}

/-- Possible latent phases at every categorical row and column. -/
structure Domains (n : Type*) (q : ℕ) where
  row : n → Finset (ZMod q)
  col : n → Finset (ZMod q)

/-- A phase assignment is retained coordinate by coordinate. -/
def Contains (D : Domains n q) (r s : n → ZMod q) : Prop :=
  (∀ i, r i ∈ D.row i) ∧ ∀ j, s j ∈ D.col j

/-- A branch with an empty domain at any variable can be discarded. -/
def Live (D : Domains n q) : Prop :=
  (∀ i, (D.row i).Nonempty) ∧ ∀ j, (D.col j).Nonempty

instance [Fintype n] (D : Domains n q) : Decidable (Live D) :=
  inferInstanceAs (Decidable ((∀ i, (D.row i).Nonempty) ∧ ∀ j, (D.col j).Nonempty))

variable [NeZero q] [DecidableEq n] [DecidableEq C] [Fintype S]

/-- Gauge-fix one observed row to zero and branch on its observed column's phase.
All other variables start with every latent state. -/
def initial (a b : n) (z : ZMod q) : Domains n q where
  row i := if i = a then {0} else univ
  col j := if j = b then {z} else univ

/-- One simultaneous support-pruning sweep. A value must have a compatible
neighbor at every observed incident cell; unobserved labels are never read. -/
def step (d : ZMod q → C) (cell : S → n × n) (labels : S → C)
    (D : Domains n q) : Domains n q where
  row i := (D.row i).filter fun a =>
    ∀ t, (cell t).1 = i → ∃ b ∈ D.col (cell t).2, d (a + b) = labels t
  col j := (D.col j).filter fun b =>
    ∀ t, (cell t).2 = j → ∃ a ∈ D.row (cell t).1, d (a + b) = labels t

/-- Run exactly the given finite number of sweeps; zero is allowed. -/
def run (d : ZMod q → C) (cell : S → n × n) (labels : S → C)
    (steps : ℕ) (D : Domains n q) : Domains n q :=
  (step d cell labels)^[steps] D

/-- All phases compatible with the chosen observed anchor after fixing its row
to zero. Every value of a larger decoder fiber is retained as a branch. -/
def anchors (d : ZMod q → C) (label : C) : Finset (ZMod q) :=
  univ.filter fun z => d z = label

variable [Fintype n]

/-- Possible decoded answers in one surviving branch, using the Cartesian
product of the two domains. This can overestimate globally compatible answers. -/
def outputs (d : ZMod q → C) (D : Domains n q) (i j : n) : Finset C :=
  if Live D then (D.row i).biUnion fun a => (D.col j).image fun b => d (a + b)
  else ∅

/-- Union over every anchor phase; a branch is never selected by its prediction. -/
def answers (d : ZMod q → C) (cell : S → n × n) (labels : S → C)
    (anchor : S) (steps : ℕ) (i j : n) : Finset C :=
  (anchors d (labels anchor)).biUnion fun z =>
    outputs d (run d cell labels steps (initial (cell anchor).1 (cell anchor).2 z)) i j

variable {K : Type*} [Fintype K]

/-- The same conservative union over a fixed finite decoder bank. Identification
must hold across all surviving decoders, not only a preferred fitted decoder. -/
def bankAnswers (decoder : K → ZMod q → C) (cell : S → n × n) (labels : S → C)
    (anchor : S) (steps : ℕ) (i j : n) : Finset C :=
  univ.biUnion fun k => answers (decoder k) cell labels anchor steps i j

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
