module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Identification

/-!
# A direct learner from categorical answer examples

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF1.
For a candidate key, remember the first answer seen with each input value.
A different answer for the same value permanently eliminates that key.
The scan consumes examples from left to right. The selected decoder uses
the recorded answer, or an explicitly supplied default for unseen values.
Neither the target rule nor a gradient is an input to the learner.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

variable {K V C : Type*}

/-- One supplied dictionary and its ordinary answer label (CF1). -/
abbrev Sample (K V C : Type*) := (K → V) × C

/-- A partial value-to-answer table; the outer option marks elimination (CF1). -/
abbrev Record (V C : Type*) := V → Option C

/-- A decoder respects all entries already recorded (CF1). -/
def Extends (record : Record V C) (f : V → C) : Prop :=
  ∀ v c, record v = some c → f v = c

/-- The surviving decoders represented by a candidate's current state (CF1). -/
def Allows (state : Option (Record V C)) (f : V → C) : Prop :=
  match state with
  | none => False
  | some record => Extends record f

/-- Insert one answer, or eliminate a conflicting candidate (CF1). -/
def step [DecidableEq V] [DecidableEq C]
    (state : Option (Record V C)) (v : V) (c : C) : Option (Record V C) :=
  match state with
  | none => none
  | some record =>
    match record v with
    | none => some (Function.update record v (some c))
    | some previous => if previous = c then some record else none

/-- A tail-recursive pass over the examples for one candidate key (CF1). -/
def scan [DecidableEq V] [DecidableEq C] (k : K)
    (samples : List (Sample K V C)) (state : Option (Record V C)) : Option (Record V C) :=
  match samples with
  | [] => state
  | s :: rest => scan k rest (step state (s.1 k) s.2)

/-- Start a candidate with no observed values (CF1). -/
def trainKey [DecidableEq V] [DecidableEq C] (k : K)
    (samples : List (Sample K V C)) : Option (Record V C) :=
  scan k samples (some (fun _ => none))

/-- Complete an observed table using a fixed default on unseen values (CF1). -/
def decode (record : Record V C) (default : C) (v : V) : C :=
  (record v).getD default

/-- Scan candidates in the supplied order and return the first survivor.
Failure is explicit; no prediction is invented if every key conflicts (CF1). -/
def learn [DecidableEq V] [DecidableEq C] (keys : List K)
    (samples : List (Sample K V C)) (default : C) : Option (K × (V → C)) :=
  match keys with
  | [] => none
  | k :: rest =>
    match trainKey k samples with
    | none => learn rest samples default
    | some record => some (k, decode record default)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward
