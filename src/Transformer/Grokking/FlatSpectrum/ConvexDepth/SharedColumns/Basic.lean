module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RepeatedRows.Complex

/-!
# Sharing complete columns without changing their ordered operations

Source: `review/convex/moments.md`, CM71. Each event contains the entire input
to one deterministic accumulator update, or is absent when the original loop
skips that class. The value type may be exact machine words. No associativity,
commutativity, real arithmetic or floating-point refinement is assumed.

For the native head, an index denotes a program and Fourier term in the
original query loop. A payload includes both source components and both
coefficient components, as exact words. `step` may then be the deterministic
rounded expression used by that implementation. The proof concerns agreement
between two executions of that same operation, not agreement with a real sum.

The exceptional set is global across blocks. It includes all explicit stored
column exceptions and all active classes of every known-only block. Its
complement therefore shares both coefficient words and active flags. The
implementation may use a larger conservative set without changing this proof.

Sharing does not encode new trainable variables, constrain unknown rows or
alter any norm. The complete CM70 head remains available independently of the
evaluation plan. If an update changes a column, the plan must be reconstructed
before using it; equality of earlier parameter values does not suffice.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns

variable {I C P W : Type*}

/-- An absent event preserves the accumulator without performing arithmetic,
CM71. In particular, skipping is not replaced by adding a numeric zero. -/
def advance (step : W → P → W) (value : W) (event : Option P) : W :=
  match event with
  | none => value
  | some payload => step value payload

/-- Evaluate one original class in the given operation order, CM71. Repeated
indices remain repeated updates; every class begins with the same given word. -/
def column (order : List I) (step : W → P → W) (events : I → C → Option P)
    (initial : W) (c : C) : W :=
  order.foldl (fun value i => advance step value (events i c)) initial

/-- Equal active flags and payloads at every actual occurrence give equal
ordered accumulator results, CM71. The proof never rearranges an update. -/
theorem column_eq (order : List I) (step : W → P → W) (events : I → C → Option P)
    (initial : W) (c d : C) (h : ∀ i ∈ order, events i c = events i d) :
    column order step events initial c = column order step events initial d := by
  induction order generalizing initial with
  | nil => rfl
  | cons i tail ih =>
    have hi := h i List.mem_cons_self
    have ht : ∀ j ∈ tail, events j c = events j d :=
      fun j hj => h j (List.mem_cons_of_mem i hj)
    simp only [column, List.foldl_cons]
    rw [hi]
    exact ih _ ht

variable [DecidableEq C]

/-- Preserve every exceptional class independently and expand the one common
result to every other original class, CM71. The result still has domain `C`. -/
def shared (order : List I) (step : W → P → W) (events : I → C → Option P)
    (initial : W) (exceptions : Finset C) (representative : C) : C → W :=
  fun c => column order step events initial
    (if c ∈ exceptions then c else representative)

/-- Expanding shared computations recovers the complete class function,
CM71. Equality is required only on the actual schedule, including its repeats.
All exceptional classes are evaluated separately, with arbitrary payloads. -/
theorem shared_eq (order : List I) (step : W → P → W) (events : I → C → Option P)
    (initial : W) (exceptions : Finset C) (representative : C)
    (h : ∀ c, c ∉ exceptions → ∀ i ∈ order, events i c = events i representative) :
    shared order step events initial exceptions representative =
      column order step events initial := by
  funext c
  by_cases hc : c ∈ exceptions
  · simp only [shared, hc, ↓reduceIte]
  · simp only [shared, hc, ↓reduceIte]
    exact (column_eq order step events initial c representative (h c hc)).symm

/-- Full class support and known-only support of CM70's stored head, CM71.
An explicit exception supplies its entire update payload; otherwise the
baseline is used. Inactive classes have an absent event, regardless of values. -/
def storedEvent (all : I → Bool) (active : I → Finset C) (baseline : I → P)
    (overrides : I → C → Option P) (i : I) (c : C) : Option P :=
  if all i then some ((overrides i c).getD (baseline i))
  else if c ∈ active i then some ((overrides i c).getD (baseline i)) else none

/-- Outside a set containing every explicit exception and every active
known-only class, two classes have the same event, CM71. This conservative
criterion permits arbitrary nonuniform exceptions and different block supports. -/
theorem storedEvent_eq_outside (all : I → Bool) (active : I → Finset C)
    (baseline : I → P) (overrides : I → C → Option P) (exceptions : Finset C)
    (i : I) (c d : C) (hc : c ∉ exceptions) (hd : d ∉ exceptions)
    (ho : ∀ a, a ∉ exceptions → overrides i a = none)
    (ha : all i = false → active i ⊆ exceptions) :
    storedEvent all active baseline overrides i c =
      storedEvent all active baseline overrides i d := by
  cases h : all i with
  | false =>
    have hca : c ∉ active i := fun he => hc (ha h he)
    have hda : d ∉ active i := fun he => hd (ha h he)
    simp [storedEvent, h, hca, hda]
  | true => simp [storedEvent, h, ho c hc, ho d hd]

/-- A global conservative exception union yields exact complete results for
every schedule, CM71. The representative must be outside that union; when
there is no such class the implementation uses the complete original loop. -/
theorem stored_shared_eq (all : I → Bool) (active : I → Finset C)
    (baseline : I → P) (overrides : I → C → Option P) (exceptions : Finset C)
    (representative : C) (hr : representative ∉ exceptions)
    (ho : ∀ i c, c ∉ exceptions → overrides i c = none)
    (ha : ∀ i, all i = false → active i ⊆ exceptions)
    (order : List I) (step : W → P → W) (initial : W) :
    shared order step (storedEvent all active baseline overrides) initial
        exceptions representative =
      column order step (storedEvent all active baseline overrides) initial := by
  apply shared_eq
  intro c hc i _
  exact storedEvent_eq_outside all active baseline overrides exceptions i c
    representative hc hr (ho i) (ha i)

/-- When every class is exceptional, sharing is the original full evaluation,
CM71. No equality or uniformity condition is imposed on any coefficient. -/
theorem full_fallback [Fintype C] (order : List I) (step : W → P → W)
    (events : I → C → Option P) (initial : W) (representative : C) :
    shared order step events initial Finset.univ representative =
      column order step events initial := by
  apply shared_eq
  intro c hc
  simp at hc

/-- Every downstream computation of the full class vector is unchanged,
CM71. This covers ordinary CE in its original summation order and raw argmax;
the readout receives every class, not just the representatives. -/
theorem full_readout {R : Type*} (order : List I) (step : W → P → W)
    (events : I → C → Option P) (initial : W) (exceptions : Finset C)
    (representative : C) (readout : (C → W) → R)
    (h : ∀ c, c ∉ exceptions → ∀ i ∈ order, events i c = events i representative) :
    readout (shared order step events initial exceptions representative) =
      readout (column order step events initial) := by
  have he := shared_eq order step events initial exceptions representative h
  exact congrArg readout he

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns
