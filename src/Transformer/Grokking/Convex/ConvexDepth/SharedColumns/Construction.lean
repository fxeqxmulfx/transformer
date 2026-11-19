module

public import Transformer.Grokking.Convex.ConvexDepth.SharedColumns.Basic

/-!
# Ordered head construction and independent certification passes

Source: `review/convex/moments.md`, CM72. Conditional equality of unknown-row
words permits sharing a head-construction accumulator outside actual anchor
labels. This is a property of a current tensor, not a restriction on the model.
Every payload word and state occurrence remains in its original order.

The same CM71 equality applies to deterministic recodings of complete events,
including interval updates or dense zero-filled coefficient reads, and to an
independent reverse schedule. Both complete vectors are expanded before their
joint readout. No theorem equates real arithmetic with machine operations.

For four-state Fourier construction, the schedule can index frequency/state
pairs. Its payload carries the anchored decoder word; the update reads the
two source words from that index and updates the appropriate real and imaginary
accumulator components. All eight accumulator components may be one value of
`A`. Equal tuples follow from equal payloads without exchanging state additions.

The sufficient hypothesis is checked for each construction block. A single
unknown row whose stored words differ disables this optimization for that
block. Other blocks may still share. Fully observed blocks retain their
original observed-class support; the theorem does not require materializing
inactive classes or treating their skipped operations as arithmetic on zero.

Certification may use more than one event interpretation. An interval pass
keeps inactive skips, while a dense exact pass can read an explicit zero
coefficient in the same position. Equal original events imply equal recodings
within each pass. No equality between the answers of two different passes is
assumed: their discrepancy is still an input to the original certificate test.

The conclusions concern complete functions on the original class type. Any
readout can therefore retain all maximizers and each class's contribution to
cross-entropy. Container capacity, cloned integer storage and runtime speed
are implementation obligations, not consequences of these identities.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.SharedColumns

variable {I C P A B Q R : Type*} [DecidableEq C]

/-- Original anchored expansion at the word level, CM72. Known nonlabels
use the specified zero word; unknown rows retain every supplied class word. -/
def decoderWord (zero : P) (mass : I → P) (labels : I → Option C)
    (unknown : I → C → P) (i : I) (c : C) : P :=
  match labels i with
  | none => unknown i c
  | some y => if c = y then mass i else zero

/-- Outside the actual anchor labels, two expanded decoder words agree when
the current unknown row agrees, CM72. The zero word is not identified with any
other representation of a numerically equal value. -/
theorem decoderWord_eq_outside (zero : P) (mass : I → P) (labels : I → Option C)
    (unknown : I → C → P) (exceptions : Finset C) (i : I) (c d : C)
    (hc : c ∉ exceptions) (hd : d ∉ exceptions)
    (hl : ∀ y, labels i = some y → y ∈ exceptions)
    (hu : labels i = none → unknown i c = unknown i d) :
    decoderWord zero mass labels unknown i c = decoderWord zero mass labels unknown i d := by
  cases hi : labels i with
  | none => simpa [decoderWord, hi] using hu hi
  | some y =>
    have hy := hl y hi
    have hcy : c ≠ y := fun he => hc (he ▸ hy)
    have hdy : d ≠ y := fun he => hd (he ▸ hy)
    simp [decoderWord, hi, hcy, hdy]

/-- A construction event includes the operation index and the entire raw word,
CM72. The index can supply both source components to a tuple-valued update. -/
def constructionEvent (zero : P) (mass : I → P) (labels : I → Option C)
    (unknown : I → C → P) (i : I) (c : C) : Option (I × P) :=
  some (i, decoderWord zero mass labels unknown i c)

/-- Conditional shared head construction preserves every complete class
accumulator, CM72. `A` can hold all real and imaginary component words together.
The known-label set and uniformity condition cover every actual state occurrence;
no property of an unvisited state or a later tensor is assumed. -/
theorem construction_shared_eq (zero : P) (mass : I → P) (labels : I → Option C)
    (unknown : I → C → P) (exceptions : Finset C) (representative : C)
    (hr : representative ∉ exceptions) (order : List I)
    (hl : ∀ i ∈ order, ∀ y, labels i = some y → y ∈ exceptions)
    (hu : ∀ i ∈ order, labels i = none → ∀ c d, unknown i c = unknown i d)
    (step : A → (I × P) → A) (initial : A) :
    shared order step (constructionEvent zero mass labels unknown) initial
        exceptions representative =
      column order step (constructionEvent zero mass labels unknown) initial := by
  apply shared_eq
  intro c hc i hi
  have he := decoderWord_eq_outside zero mass labels unknown exceptions i c
    representative hc hr (hl i hi) (fun hn => hu i hi hn c representative)
  unfold constructionEvent
  rw [he]

/-- A separate pass can deterministically recode a stored event, CM72. It
may turn an absent stored coefficient into an explicit zero payload, preserving
that pass's own original operations instead of borrowing another pass's skips. -/
def recodedEvent (transform : I → Option P → Q) (events : I → C → Option P)
    (i : I) (c : C) : Option Q := some (transform i (events i c))

/-- Shared columns stay equal through a pass's own deterministic decoding,
CM72. The original event equality, not agreement of rounded final answers,
justifies evaluating one representative. -/
theorem recoded_shared_eq (order : List I) (step : A → Q → A)
    (events : I → C → Option P) (transform : I → Option P → Q)
    (initial : A) (exceptions : Finset C) (representative : C)
    (h : ∀ c, c ∉ exceptions → ∀ i ∈ order, events i c = events i representative) :
    shared order step (recodedEvent transform events) initial exceptions representative =
      column order step (recodedEvent transform events) initial := by
  apply shared_eq
  intro c hc i hi
  unfold recodedEvent
  rw [h c hc i hi]

/-- The independently reversed pass keeps its own update order, CM72. Only
equality between classes is used; forward and reverse answers need not agree. -/
theorem reverse_recoded_shared_eq (order : List I) (step : A → Q → A)
    (events : I → C → Option P) (transform : I → Option P → Q)
    (initial : A) (exceptions : Finset C) (representative : C)
    (h : ∀ c, c ∉ exceptions → ∀ i ∈ order, events i c = events i representative) :
    shared order.reverse step (recodedEvent transform events) initial exceptions representative =
      column order.reverse step (recodedEvent transform events) initial := by
  apply recoded_shared_eq
  intro c hc i hi
  exact h c hc i (List.mem_reverse.mp hi)

/-- A joint certification readout receives both original complete vectors,
CM72. This includes residual/enclosure checks and the full list of tied raw
maximizers; it neither drops tied classes nor forces two different passes equal. -/
theorem joint_readout_eq (order : List I) (events : I → C → Option P)
    (forwardStep : A → Q → A) (reverseStep : B → R → B)
    (forwardCode : I → Option P → Q) (reverseCode : I → Option P → R)
    (forwardInitial : A) (reverseInitial : B) (exceptions : Finset C)
    (representative : C) {V : Type*} (readout : (C → A) → (C → B) → V)
    (h : ∀ c, c ∉ exceptions → ∀ i ∈ order, events i c = events i representative) :
    readout (shared order forwardStep (recodedEvent forwardCode events)
        forwardInitial exceptions representative)
        (shared order.reverse reverseStep (recodedEvent reverseCode events)
          reverseInitial exceptions representative) =
      readout (column order forwardStep (recodedEvent forwardCode events) forwardInitial)
        (column order.reverse reverseStep (recodedEvent reverseCode events) reverseInitial) := by
  rw [recoded_shared_eq order forwardStep events forwardCode forwardInitial
      exceptions representative h,
    reverse_recoded_shared_eq order reverseStep events reverseCode reverseInitial
      exceptions representative h]

end Transformer.Grokking.Convex.ConvexDepth.SharedColumns
