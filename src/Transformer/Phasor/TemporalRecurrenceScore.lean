module

public import Transformer.Phasor.DifferencedDivisionScore

/-!
# Temporal differences and field readout in the GPT recurrence catalog

Source: DiffGPT `recurrence.trajectory_orbits`, `adapters.temporal_records`
and `recurrence_field_sampler.CanonicalFieldSampler` at 5a37c515. Each
window starts with actual trajectory values. Only its last difference is
supervised. Fibonacci LGT pools output probability by the decoded residue.
-/

@[expose] public section

namespace Transformer.Phasor.TemporalRecurrenceScore

open DifferencedDivisionScore
open scoped BigOperators

/-- Both transitions have explicit inverses, so a complete orbit can belong
to a single data split. Division runs on pairs of nonzero residues. -/
def fibonacci {G : Type*} [AddCommGroup G] : (G × G) ≃ (G × G) where
  toFun x := (x.2, x.1 + x.2)
  invFun y := (y.2 - y.1, y.1)
  left_inv := by rintro ⟨a,b⟩; simp
  right_inv := by rintro ⟨a,b⟩; simp

def division {G : Type*} [Group G] : (G × G) ≃ (G × G) where
  toFun x := (x.2, x.1 / x.2)
  invFun y := (y.2 * y.1, y.1)
  left_inv := by rintro ⟨a,b⟩; simp
  right_inv := by rintro ⟨a,b⟩; simp

/-- Real consecutive observations supply the starting level of a window. -/
def delta : List ℤ → List ℤ
  | [] => []
  | a :: xs => difference a xs

theorem delta_append (p : List ℤ) (a b : ℤ) :
    delta (p ++ [a,b]) = delta (p ++ [a]) ++ [b-a] := by
  induction p with
  | nil => rfl
  | cons x xs ih =>
    cases xs with
    | nil => rfl
    | cons y ys => exact congrArg (List.cons (y-x)) ih

def tokens (o : Order) (x : List ℤ) : List ℤ :=
  ((match o with | .first => delta x | .second => delta (delta x))).map (o.offset + ·)

/-- Appending a future value preserves every prefix token. The source's
single last target therefore has a decoder base computed from the past. -/
theorem append_future (o : Order) (p : List ℤ) (a b y : ℤ) :
    tokens o (p ++ [a,b,y]) = tokens o (p ++ [a,b]) ++ [target o a b y] := by
  have end₁ : delta (p ++ [a,b,y]) = delta (p ++ [a,b]) ++ [y-b] := by
    simpa only [List.append_assoc, List.cons_append, List.nil_append] using
      delta_append (p ++ [a]) b y
  cases o with
  | first => simp [tokens, end₁, target]
  | second =>
    have end₂ : delta (delta (p ++ [a,b,y])) =
        delta (delta (p ++ [a,b])) ++ [(y-b)-(b-a)] := by
      rw [end₁, delta_append p a b]
      simpa only [List.append_assoc, List.cons_append, List.nil_append] using
        delta_append (delta (p ++ [a])) (b-a) (y-b)
    simp [tokens, end₂, target]
    ring

/-- On either unit grid, exact decoded-value accuracy from unrestricted
argmax agrees with the unique last target token. -/
theorem decode_iff (o : Order) (a b future token : ℤ) :
    decode o a b token = future ↔ token = target o a b future := by
  cases o <;> simp [decode,target] <;> omega

/-- Pool all vocabulary probabilities that decode to the requested residue. -/
noncomputable def mass {n : ℕ} (p : Fin n → ℝ) (center base r : ZMod 97) : ℝ :=
  ∑ j, if (j.val : ZMod 97) - center + base = r then p j else 0

/-- Moving the decoder base only permutes the pooled classes. One output
distribution serves every known prefix without evaluating the recurrence. -/
theorem mass_shift {n : ℕ} (p : Fin n → ℝ) (center base r : ZMod 97) :
    mass p center base r = mass p center 0 (r-base) := by
  unfold mass
  simp only [add_zero, ← eq_sub_iff_add_eq]

example : fibonacci ((12,15) : ZMod 97 × ZMod 97) = (15,27) := by decide

example : delta [1,4,9,16] = [3,5,7] := by decide

example : tokens .second [8,13,21,34,55] =
    tokens .second [8,13,21,34] ++ [target .second 21 34 55] :=
  append_future .second [8,13] 21 34 55

example : decode .second 21 34 264 = 55 :=
  (decode_iff .second 21 34 55 264).2 (by decide)

end Transformer.Phasor.TemporalRecurrenceScore
