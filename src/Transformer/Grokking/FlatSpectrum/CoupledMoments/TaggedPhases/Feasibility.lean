module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases.Basic

/-!
# Convex spectral feasibility and actual CE for tagged phases

Source: `review/flat_spectrum.md`, CM25. The tags change categorical capacity
while retaining radius-N latent Fourier bounds and ordinary output-vocabulary
CE. The declared enlarged prediction hull is convex. All earlier phase
vertices embed in it; no inclusion of every old fractional latent table is claimed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases

open Schatten ObservedRefinement

variable {q : ℕ} [NeZero q] {n C S : Type*}
  [Fintype n] [DecidableEq n] [Fintype C] [DecidableEq C] [Fintype S]

/-- A base categorical phase table with constant flagged input strips, CM25. -/
def labels (constant : C) (d : ZMod q → C) (r s : n → ZMod q)
    (rowFlag columnFlag : n → Bool) : n → n → C :=
  fun i j => if rowFlag i || columnFlag j then constant else d (r i + s j)

omit [Fintype C] in
/-- Every tagged table is a vertex in the declared convex decoder hull, with
the same radius-N latent operator/nuclear budgets, CM25. -/
theorem tagged_member (constant : C) (d : ZMod q → C) (r s : n → ZMod q)
    (rowFlag columnFlag : n → Bool) :
    pointMass (labels constant d r s rowFlag columnFlag) ∈
      DecoderHull (phaseCode (Equiv.refl (ZMod (3 * q))))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n) := by
  unfold labels
  simpa only [decode_lift] using phase_member (decoder constant d)
    (fun i => lift (rowFlag i) (r i)) (fun j => lift (columnFlag j) (s j))

omit [Fintype C] in
/-- No older decoded phase vertex is lost: choose no flagged axes, CM25.
This does not assert inclusion of the entire older fractional density hull. -/
theorem old_vertex_member (constant : C) (d : ZMod q → C) (r s : n → ZMod q) :
    pointMass (fun i j => d (r i + s j)) ∈
      DecoderHull (phaseCode (Equiv.refl (ZMod (3 * q))))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n) := by
  have h := tagged_member constant d r s (fun _ => false) (fun _ => false)
  unfold labels at h
  simpa only [Bool.false_or, Bool.false_eq_true, ↓reduceIte] using h

/-- Actual observed agreement certifies a global ordinary-CE minimum over the
enlarged convex prediction hull, CM25. Agreement is checked, not inferred from
an optimizer, a constant-looking partial row, or spectral flatness. -/
theorem tagged_fit_minimum (constant : C) (d : ZMod q → C) (r s : n → ZMod q)
    (rowFlag columnFlag : n → Bool) (cell : S → n × n) (target : S → C)
    (hfit : ∀ t, labels constant d r s rowFlag columnFlag (cell t).1 (cell t).2 = target t) :
    IsMinOn (loss cell target)
      (DecoderHull (phaseCode (Equiv.refl (ZMod (3 * q))))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n))
      (pointMass (labels constant d r s rowFlag columnFlag)) := by
  unfold labels
  have h := decodedPhase_hull_minimum (decoder constant d)
    (fun i => lift (rowFlag i) (r i)) (fun j => lift (columnFlag j) (s j))
    cell target (fun t => by simpa only [decode_lift, labels] using hfit t)
  simpa only [decode_lift] using h.2

omit [NeZero q] [Fintype n] [DecidableEq n] in
/-- At unit logit scale the flagged observations and the interior contribute
their actual mismatch counts to ordinary CE, CM25. -/
theorem tagged_loss (constant : C) (d : ZMod q → C) (r s : n → ZMod q)
    (rowFlag columnFlag : n → Bool) (cell : S → n × n) (target : S → C) :
    loss cell target (pointMass (labels constant d r s rowFlag columnFlag)) =
      idealLoss target +
        (∑ t, if rowFlag (cell t).1 || columnFlag (cell t).2 then mistake constant (target t) else 0) +
        ∑ t, if rowFlag (cell t).1 || columnFlag (cell t).2 then 0
          else mistake (d (r (cell t).1 + s (cell t).2)) (target t) := by
  rw [pointMass_loss, add_assoc, ← Finset.sum_add_distrib]
  congr 1
  unfold errors
  apply Finset.sum_congr rfl
  intro t ht
  dsimp only [labels]
  split_ifs <;> simp

omit [NeZero q] [Fintype n] [DecidableEq n] in
/-- A verified decrease of the actual interior errors decreases full observed
CE while the strip choices stay fixed, CM25. Native cropped-data refinements
must preserve the original vocabulary and this exact observation partition. -/
theorem tagged_descent (constant : C) (d d' : ZMod q → C) (r s r' s' : n → ZMod q)
    (rowFlag columnFlag : n → Bool) (cell : S → n × n) (target : S → C)
    (hinner : (∑ t, if rowFlag (cell t).1 || columnFlag (cell t).2 then 0
        else mistake (d' (r' (cell t).1 + s' (cell t).2)) (target t)) ≤
      ∑ t, if rowFlag (cell t).1 || columnFlag (cell t).2 then 0
        else mistake (d (r (cell t).1 + s (cell t).2)) (target t)) :
    loss cell target (pointMass (labels constant d' r' s' rowFlag columnFlag)) ≤
      loss cell target (pointMass (labels constant d r s rowFlag columnFlag)) := by
  rw [tagged_loss, tagged_loss]
  linarith

/-! A nonconstant interior and a constant row/column, outside CM24's
surjective-axis family. Input categories have no arithmetic interpretation. -/

/-- The base phases of the closed three-category capacity witness, CM25. -/
def witnessPhase (i : Fin 3) : ZMod 2 := if i = 2 then 1 else 0

/-- The first input category is flagged on each axis, CM25. -/
def witnessFlag (i : Fin 3) : Bool := decide (i = 0)

/-- The witness has a constant first row and column but a nonconstant interior,
CM25, including the intersection of the two constant strips. -/
theorem witness_pattern :
    (∀ j, labels 0 id witnessPhase witnessPhase witnessFlag witnessFlag 0 j = 0) ∧
    (∀ i, labels 0 id witnessPhase witnessPhase witnessFlag witnessFlag i 0 = 0) ∧
    labels 0 id witnessPhase witnessPhase witnessFlag witnessFlag 1 2 = (1 : ZMod 2) ∧
    labels 0 id witnessPhase witnessPhase witnessFlag witnessFlag 1 1 = (0 : ZMod 2) := by
  simp [labels, witnessFlag, witnessPhase]

example := tagged_member (0 : ZMod 2) id witnessPhase witnessPhase witnessFlag witnessFlag
example := old_vertex_member (0 : ZMod 2) id witnessPhase witnessPhase
example := tagged_fit_minimum (0 : ZMod 2) id witnessPhase witnessPhase witnessFlag witnessFlag
  (id : Fin 3 × Fin 3 → Fin 3 × Fin 3)
  (fun ij => labels 0 id witnessPhase witnessPhase witnessFlag witnessFlag ij.1 ij.2) (by intro t; rfl)
example := tagged_loss (0 : ZMod 2) id witnessPhase witnessPhase witnessFlag witnessFlag
  (id : Fin 3 × Fin 3 → Fin 3 × Fin 3) (fun ij => if ij.1 = ij.2 then 0 else 1)
example := tagged_descent (0 : ZMod 2) (fun _ => 0) id
  witnessPhase witnessPhase witnessPhase witnessPhase witnessFlag witnessFlag
  (fun t : Bool => (1, if t then 2 else 1)) (fun t => if t then 1 else 0)
  (by norm_num [witnessFlag, witnessPhase, mistake])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases
