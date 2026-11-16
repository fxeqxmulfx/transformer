module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns.Construction
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns.Witness

/-!
# Nonconstant construction and different certification passes

Source: `review/flat_spectrum.md`, CM72. Two anchored classes, an unknown row
and a repeated operation produce distinct exceptional and shared columns.
Forward and reverse answers deliberately differ; only equality across the
certified shared classes is used. A nonuniform unknown row defeats sharing.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns

/-- A nonconstant three-state dictionary with an unobserved middle state. -/
def buildLabels : Fin 3 → Option (Fin 4) := ![some 0, none, some 2]

/-- Arbitrary mass words are read only in anchored rows. -/
def buildMass (i : Fin 3) : Nat := i.val + 5

/-- A concrete four-class uniform table, without deleting any class. -/
def buildUnknown (i : Fin 3) : Fin 4 → Nat :=
  ![i.val + 3, i.val + 3, i.val + 3, i.val + 3]

/-- Paired component accumulation exposes both index and raw payload. -/
def buildStep (v : Nat × Nat) (p : Fin 3 × Nat) : Nat × Nat :=
  (10 * v.1 + p.2, 10 * v.2 + p.1.val + p.2)

/-- Every original class receives the full paired construction trace. -/
def buildEvents : Fin 3 → Fin 4 → Option (Fin 3 × Nat) :=
  constructionEvent 0 buildMass buildLabels buildUnknown

/-- Two different exceptional columns coexist with the shared complement,
CM72. Both accumulator components and the repeated state remain. -/
theorem witness_constructed_columns :
    column witnessOrder buildStep buildEvents (0, 0) =
      ![(5405, 5525), (400, 520), (470, 590), (400, 520)] := by
  funext c
  fin_cases c <;> decide

/-- The construction condition is inhabited by a nonconstant dictionary,
CM72, and reconstructs every component of all four logical columns. -/
theorem witness_shared_construction :
    shared witnessOrder buildStep buildEvents (0, 0) {0, 2} 1 =
      column witnessOrder buildStep buildEvents (0, 0) := by
  apply construction_shared_eq
  · decide
  · intro i hi y hy
    fin_cases i <;> simp [buildLabels] at hy
    · subst y; simp
    · subst y; simp
  · intro i hi hn c d
    fin_cases c <;> fin_cases d <;> simp [buildUnknown]

example (i : Fin 3) : decoderWord 0 buildMass buildLabels buildUnknown i 1 =
    decoderWord 0 buildMass buildLabels buildUnknown i 3 := by
  apply decoderWord_eq_outside _ _ _ _ {0, 2}
  · decide
  · decide
  · intro y hy
    fin_cases i <;> simp [buildLabels] at hy
    · subst y; simp
    · subst y; simp
  · intro hn
    simp [buildUnknown]

example : shared witnessOrder buildStep buildEvents (0, 0) {0, 2} 1 =
    column witnessOrder buildStep buildEvents (0, 0) := by
  exact witness_shared_construction

/-- Resolve an absent event to a payload, as a dense coefficient reader may.
Its index contribution ensures this is not silently an inactive skip. -/
def certifyCode (i : Fin 3) (event : Option Nat) : Nat := i.val + event.getD 0

example : shared witnessOrder witnessStep (recodedEvent certifyCode witnessEvents)
      0 {0} 1 = column witnessOrder witnessStep (recodedEvent certifyCode witnessEvents) 0 := by
  apply recoded_shared_eq
  intro c hc i hi
  fin_cases c <;> fin_cases i <;> simp_all [witnessEvents, storedEvent,
    witnessAll, witnessActive, witnessBaseline, witnessOverrides]

example : shared witnessOrder.reverse witnessStep (recodedEvent certifyCode witnessEvents)
      0 {0} 1 =
    column witnessOrder.reverse witnessStep (recodedEvent certifyCode witnessEvents) 0 := by
  apply reverse_recoded_shared_eq
  intro c hc i hi
  fin_cases c <;> fin_cases i <;> simp_all [witnessEvents, storedEvent,
    witnessAll, witnessActive, witnessBaseline, witnessOverrides]

/-- The independent order can change the answer, CM72. The sharing theorem
does not erase this discrepancy from a certificate's input. -/
theorem witness_distinct_passes :
    column witnessOrder witnessStep (recodedEvent certifyCode witnessEvents) 0 1 ≠
      column witnessOrder.reverse witnessStep (recodedEvent certifyCode witnessEvents) 0 1 := by
  decide

example :
    (shared witnessOrder witnessStep (recodedEvent certifyCode witnessEvents) 0 {0} 1 2,
      shared witnessOrder.reverse witnessStep (recodedEvent certifyCode witnessEvents) 0 {0} 1 2) =
    (column witnessOrder witnessStep (recodedEvent certifyCode witnessEvents) 0 2,
      column witnessOrder.reverse witnessStep (recodedEvent certifyCode witnessEvents) 0 2) := by
  apply joint_readout_eq witnessOrder witnessEvents witnessStep witnessStep
    certifyCode certifyCode 0 0 {0} 1 (fun a b => (a 2, b 2))
  intro c hc i hi
  fin_cases c <;> fin_cases i <;> simp_all [witnessEvents, storedEvent,
    witnessAll, witnessActive, witnessBaseline, witnessOverrides]

/-- An arbitrary unknown row is permitted by the model, but can invalidate
this optional optimization. Explicit entries remain available in the fallback. -/
def nonuniformBuildUnknown (i : Fin 3) : Fin 4 → Nat :=
  ![i.val + 3, i.val + 3, i.val + 3, i.val + 4]

/-- Sharing the nonuniform unknown row would change a complete column, CM72. -/
theorem witness_nonuniform_construction_refusal :
    shared witnessOrder buildStep
        (constructionEvent 0 buildMass buildLabels nonuniformBuildUnknown) (0, 0) {0, 2} 1 3 ≠
      column witnessOrder buildStep
        (constructionEvent 0 buildMass buildLabels nonuniformBuildUnknown) (0, 0) 3 := by
  decide

example : (0 : UInt64) ≠ (0x8000000000000000 : UInt64) := by decide

/-- An exact paired-word accumulator; xor and integer addition are intentional
fixture operations, not a proposed model of floating-point multiplication. -/
def buildWordStep (v : UInt64 × UInt64) (p : Fin 3 × UInt64) : UInt64 × UInt64 :=
  (v.1 ^^^ p.2, v.2 + p.2 + UInt64.ofNat p.1.val)

example : shared witnessOrder buildWordStep
      (constructionEvent (0 : UInt64) (fun i => UInt64.ofNat (buildMass i)) buildLabels
        (fun i c => UInt64.ofNat (buildUnknown i c))) (0x8000000000000000, 1) {0, 2} 1 =
    column witnessOrder buildWordStep
      (constructionEvent (0 : UInt64) (fun i => UInt64.ofNat (buildMass i)) buildLabels
        (fun i c => UInt64.ofNat (buildUnknown i c))) (0x8000000000000000, 1) := by
  apply construction_shared_eq
  · decide
  · intro i hi y hy
    fin_cases i <;> simp [buildLabels] at hy
    · subst y; simp
    · subst y; simp
  · intro i hi hn c d
    fin_cases c <;> fin_cases d <;> simp [buildUnknown]

example :
    shared witnessOrder buildStep
        (constructionEvent 0 buildMass buildLabels nonuniformBuildUnknown) (0, 0) Finset.univ 1 =
      column witnessOrder buildStep
        (constructionEvent 0 buildMass buildLabels nonuniformBuildUnknown) (0, 0) := by
  exact full_fallback _ _ _ _ _

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedColumns
