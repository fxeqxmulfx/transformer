module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns.Basic

/-!
# Nonconstant ordered sharing and necessary conditions

Source: `review/convex/moments.md`, CM71. A four-class example retains one
exceptional column, three shared columns, a known-only block, distinct payloads
and a repeated operation. Separate examples show that deleting an exceptional
class, ignoring active support, reordering or deduplicating updates can fail.
Word examples use exact UInt64 values, not an assumed model of IEEE arithmetic.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns

/-- The middle block has known-only support; the other two have full support. -/
def witnessAll (i : Fin 3) : Bool := decide (i ≠ 1)

/-- The known-only block retains class zero; the remaining blocks are full. -/
def witnessActive (i : Fin 3) : Finset (Fin 4) :=
  if i = 1 then {0} else Finset.univ

/-- All payloads differ between the three blocks. -/
def witnessBaseline (i : Fin 3) : Nat := i.val + 2

/-- Class zero has a different complete payload at every block. -/
def witnessOverrides (i : Fin 3) (c : Fin 4) : Option Nat :=
  if c = 0 then some (i.val + 7) else none

/-- Decimal accumulation exposes the operation order without a rounding model. -/
def witnessStep (value payload : Nat) : Nat := 10 * value + payload

/-- The full original events preserve known-only inactive skips. -/
def witnessEvents : Fin 3 → Fin 4 → Option Nat :=
  storedEvent witnessAll witnessActive witnessBaseline witnessOverrides

/-- The final repeated block is retained, CM71. -/
def witnessOrder : List (Fin 3) := [0, 1, 2, 0]

/-- One exceptional class gives a nonconstant complete result, CM71. -/
theorem witness_results :
    column witnessOrder witnessStep witnessEvents 0 =
      fun c => if c = 0 then 7897 else 242 := by
  funext c
  fin_cases c <;> decide

/-- The same nonconstant result is recovered by expanding the shared class,
CM71, while retaining all four logical classes. -/
theorem witness_shared :
    shared witnessOrder witnessStep witnessEvents 0 {0} 1 =
      column witnessOrder witnessStep witnessEvents 0 := by
  apply stored_shared_eq
  · decide
  · intro i c hc
    simp only [Finset.mem_singleton] at hc
    simp [witnessOverrides, hc]
  · intro i
    fin_cases i <;> simp [witnessAll, witnessActive]

-- Each theorem with hypotheses has an instance whose events are nonconstant.
example : column witnessOrder witnessStep witnessEvents 11 2 =
    column witnessOrder witnessStep witnessEvents 11 1 := by
  apply column_eq
  intro i hi
  fin_cases i <;> decide

example : shared witnessOrder witnessStep witnessEvents 0 {0} 1 =
    column witnessOrder witnessStep witnessEvents 0 := by
  apply shared_eq
  intro c hc i hi
  fin_cases c <;> fin_cases i <;> simp_all [witnessEvents, storedEvent,
    witnessAll, witnessActive, witnessBaseline, witnessOverrides]

example (i : Fin 3) :
    storedEvent witnessAll witnessActive witnessBaseline witnessOverrides i 2 =
      storedEvent witnessAll witnessActive witnessBaseline witnessOverrides i 1 := by
  apply storedEvent_eq_outside _ _ _ _ {0}
  · decide
  · decide
  · intro c hc
    simp only [Finset.mem_singleton] at hc
    simp [witnessOverrides, hc]
  · fin_cases i <;> simp [witnessAll, witnessActive]

example : shared witnessOrder witnessStep witnessEvents 0 {0} 1 =
    column witnessOrder witnessStep witnessEvents 0 := by
  exact witness_shared

example :
    (fun z : Fin 4 → Nat => [z 0, z 1, z 2, z 3])
        (shared witnessOrder witnessStep witnessEvents 0 {0} 1) =
      (fun z : Fin 4 → Nat => [z 0, z 1, z 2, z 3])
        (column witnessOrder witnessStep witnessEvents 0) := by
  apply full_readout witnessOrder witnessStep witnessEvents 0 {0} 1
    (fun z : Fin 4 → Nat => [z 0, z 1, z 2, z 3])
  intro c hc i hi
  fin_cases c <;> fin_cases i <;> simp_all [witnessEvents, storedEvent,
    witnessAll, witnessActive, witnessBaseline, witnessOverrides]

/-- Erasing a distinct class changes the result, CM71. The native fallback
therefore keeps every exceptional column even when none can share. -/
theorem witness_wrong_merge :
    shared witnessOrder witnessStep witnessEvents 0 ∅ 1 0 ≠
      column witnessOrder witnessStep witnessEvents 0 0 := by
  decide

example : shared witnessOrder witnessStep witnessEvents 0 Finset.univ 2 =
    column witnessOrder witnessStep witnessEvents 0 := by
  exact full_fallback _ _ _ _ _

/-- Equal numeric payloads alone do not justify ignoring different active
supports, CM71: a skipped update and an active zero update are distinct. -/
theorem witness_skip_matters :
    advance witnessStep 1 none ≠ advance witnessStep 1 (some 0) := by
  decide

/-- A deterministic accumulator need not commute, CM71. -/
theorem witness_order_matters :
    column [1, 0, 2, 0] witnessStep witnessEvents 0 0 ≠
      column witnessOrder witnessStep witnessEvents 0 0 := by
  decide

/-- Multiplicity belongs to the operation sequence, CM71. -/
theorem witness_repeats_matter :
    column [0, 1, 2] witnessStep witnessEvents 0 0 ≠
      column witnessOrder witnessStep witnessEvents 0 0 := by
  decide

/-- A word-level deterministic update, used only to instantiate the generic
theorem. This is xor, not a claim about floating-point addition. -/
def witnessWordStep (value payload : UInt64) : UInt64 := value ^^^ payload

example : advance witnessWordStep (0x8000000000000000 : UInt64) none =
    (0x8000000000000000 : UInt64) := by decide

example : advance witnessWordStep (0 : UInt64) (some 1) = (1 : UInt64) := by decide

example :
    column witnessOrder witnessWordStep
        (fun i c => (witnessEvents i c).map UInt64.ofNat) (0x8000000000000000 : UInt64) 2 =
      column witnessOrder witnessWordStep
        (fun i c => (witnessEvents i c).map UInt64.ofNat) (0x8000000000000000 : UInt64) 1 := by
  apply column_eq
  intro i hi
  fin_cases i <;> decide

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns
