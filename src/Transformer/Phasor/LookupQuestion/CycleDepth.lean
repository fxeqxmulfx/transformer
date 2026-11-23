module

public import Transformer.Phasor.LookupQuestion.DistinctLayers
public import Mathlib.Data.Fin.Basic
public import Mathlib.Data.ZMod.Defs
public import Mathlib.Logic.Function.Iterate

/-! Fixed-alphabet length transfer for the Phasor table study. Five
start positions may share some names across layers of a printed cycle.
Within a lane the displayed endpoint is first reached at its stated
depth when that depth is less than the size of the cycle. -/
@[expose] public section
namespace Transformer.Phasor.LookupQuestion.CycleDepth
open Fin.NatCast

/-- A single successor step in a cycle numbered before renaming its keys. -/
def advance (x : Fin 210) : Fin 210 := x+1

/-- Each step advances the position by one modulo the fixed vocabulary size. -/
theorem position (x : Fin 210) (h : ℕ) : advance^[h] x = x+(h : Fin 210) := by
  induction h with
  | zero => simp
  | succ h ih => rw [Function.iterate_succ_apply', ih]; simp [advance, Nat.cast_succ, add_assoc]

/-- No earlier query state is the endpoint, even after an arbitrary
injective relabeling of the 210 token names. -/
theorem first_arrival {A : Type*} (labels : Fin 210 → A) (separate : Function.Injective labels)
    (x : Fin 210) (k h : ℕ) (before : k < h) (within : h < 210) :
    labels (advance^[k] x) ≠ labels (advance^[h] x) := by
  intro same
  have eq := separate same
  rw [position,position] at eq
  have value := congrArg Fin.val (add_left_cancel eq)
  simp only [Fin.val_natCast, Nat.mod_eq_of_lt (lt_trans before within), Nat.mod_eq_of_lt within] at value
  omega

example : advance^[80] 205 = 75 := by rw [position]; decide
example (x : Fin 210) : advance^[79] x ≠ advance^[80] x :=
  first_arrival id Function.injective_id x 79 80 (by decide) (by decide)
example (x : Fin 210) : advance^[159] x ≠ advance^[160] x :=
  first_arrival id Function.injective_id x 159 160 (by decide) (by decide)

end Transformer.Phasor.LookupQuestion.CycleDepth
