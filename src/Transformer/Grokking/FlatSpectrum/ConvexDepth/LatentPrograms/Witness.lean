module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms.Transfer

/-!
# A learned many-to-one decoder and an unresolved latent-state witness

Source: `review/flat_spectrum.md`, CM53. Three observations select a latent
input program and learn a nonconstant decoder. A genuinely new input is
identified, while an unseen state admits opposite feasible Fourier heads.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms

open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- Fixed coordinate programs over categorical symbols, with no raw label
or target decoder among their arguments, CM53. -/
def witnessProgram (j : Bool) (x : Fin 4 × Fin 4) : Fin 4 := if j then x.2 else x.1

/-- Original observations identify second-coordinate states0,1,2. -/
def witnessSamples : List ((Fin 4 × Fin 4) × Bool) :=
  [((0, 0), false), ((0, 1), false), ((0, 2), true)]

/-- The unknown target decoder used only to verify realizability, CM53. -/
def witnessDecoder (s : Fin 4) : Bool := decide (2 ≤ s.val)

/-- A genuine many-to-one, nonconstant raw decoder, CM53. -/
theorem witness_many_to_one : witnessDecoder 0 = witnessDecoder 1 ∧
    witnessDecoder 0 ≠ witnessDecoder 2 := by decide

/-- The learned filter itself selects the consistent input program, CM53. -/
theorem witness_retained : retained witnessProgram [false, true] witnessSamples = [true] := by
  decide

/-- A new input with a covered latent state receives its raw label, CM53.
The query pair itself is absent from the observations. -/
theorem witness_new_input :
    predict witnessProgram [false, true] witnessSamples (3, 2) = some true ∧
      (3, 2) ∉ witnessSamples.map Prod.fst := by
  decide

/-- The same learned decoder refuses at latent state3, absent from every
observed second coordinate, CM53. -/
theorem witness_unknown_state :
    predict witnessProgram [false, true] witnessSamples (0, 3) = none ∧
      ∀ t ∈ witnessSamples, witnessProgram true t.1 ≠ 3 := by
  decide

/-- Realizability is inhabited by an unknown noninjective decoder; the
learner never receives this witness index or function, CM53. -/
theorem witness_realizable :
    (∃ j ∈ [false, true], ∃ decoder : Fin 4 → Bool,
      ∀ x, decoder (witnessProgram j x) = witnessDecoder x.2) ∧
      ∀ t ∈ witnessSamples, t.2 = witnessDecoder t.1.2 := by
  constructor
  · exact ⟨true, by simp, witnessDecoder, by simp [witnessProgram]⟩
  · decide

/-- Actual joint coefficients built from the learned dictionary, retaining
an explicit raw default at the one unknown state, CM53. -/
def witnessPoint (default : Bool) : Parameters Bool (Fin 4) Bool :=
  vertex witnessProgram witnessSamples default true

/-- Both possible unknown-state completions are feasible, CM53. -/
theorem witnessPoint_mem (default : Bool) :
    witnessPoint default ∈ domain witnessProgram [false, true] witnessSamples := by
  exact vertex_mem witnessProgram [false, true] witnessSamples default true (by decide)

/-- The executable initializer chooses these coefficients from training,
CM53, rather than being supplied the target program as an input. -/
theorem witness_initialized (default : Bool) :
    initializeTensor witnessProgram [false, true] witnessSamples default =
      some (witnessPoint default) := by
  unfold initializeTensor
  rw [witness_retained]
  rfl

/-- A full four-state Fourier head identifies the new input's raw Boolean
label at margin24, CM53. The state and output vocabulary sizes differ. -/
theorem witness_complex_logits :
    headLogits (features fourier4 witnessProgram) (head fourier4 24 (witnessPoint true))
      (3, 2) = fun c => if c = true then 24 else 0 := by
  apply accepted_logits fourier4 fourier4_unitary witnessProgram [false, true]
    witnessSamples 24 (witnessPoint_mem true) (3, 2) true
  exact witness_new_input.1

/-- The actual complete complex prediction has the ordinary two-class CE,
CM53. This is a new input, not an invented unobserved raw class. -/
theorem witness_complex_ce :
    crossEntropy (headLogits (features fourier4 witnessProgram)
      (head fourier4 24 (witnessPoint true)) (3, 2)) true = Real.log (1 + Real.exp (-24)) := by
  rw [witness_complex_logits]
  have h := fixed_margin_ce true 24
  norm_num at h
  exact h

/-- Every real/imaginary coefficient fits the derived column budget, CM53. -/
theorem witness_budget (default : Bool) : head fourier4 24 (witnessPoint default) ∈
    headBudget 192 := by
  have h := head_budget fourier4 fourier4_unitary witnessProgram [false, true]
    witnessSamples 24 (witnessPoint_mem default)
  norm_num at h
  exact h

/-- Changing only the unknown-state completion leaves every observed full
complex logit unchanged, CM53. Both heads fit the same original raw labels. -/
theorem witness_same_observations (t : (Fin 4 × Fin 4) × Bool) (ht : t ∈ witnessSamples) :
    headLogits (features fourier4 witnessProgram) (head fourier4 24 (witnessPoint false)) t.1 =
      headLogits (features fourier4 witnessProgram) (head fourier4 24 (witnessPoint true)) t.1 := by
  rw [observed_logits fourier4 fourier4_unitary witnessProgram [false, true]
    witnessSamples 24 (witnessPoint_mem false) t ht,
    observed_logits fourier4 fourier4_unitary witnessProgram [false, true]
    witnessSamples 24 (witnessPoint_mem true) t ht]

/-- Two feasible actual complex heads give opposite margins at the
uncovered state, CM53, despite identical source flatness and observed fits. -/
theorem witness_opposite_margins :
    headLogits (features fourier4 witnessProgram) (head fourier4 24 (witnessPoint false))
      (0, 3) false = 24 ∧
    headLogits (features fourier4 witnessProgram) (head fourier4 24 (witnessPoint false))
      (0, 3) true = 0 ∧
    headLogits (features fourier4 witnessProgram) (head fourier4 24 (witnessPoint true))
      (0, 3) false = 0 ∧
    headLogits (features fourier4 witnessProgram) (head fourier4 24 (witnessPoint true))
      (0, 3) true = 24 := by
  norm_num [logits fourier4 fourier4_unitary, witnessPoint, vertex, ofDecoders,
    completion, lookup, witnessSamples, witnessProgram, Fintype.sum_bool]

/-- The actual source and learned head both have nonzero imaginary
coordinates, CM53. Their product contributes6 to a covered raw query. -/
theorem witness_imaginary_coordinates :
    features fourier4 witnessProgram (true, (1, true)) (3, 1) = -(1 / 2 : ℝ) ∧
      head fourier4 24 (witnessPoint true) (true, (1, true)) false = -12 := by
  norm_num [features, head, tableHead, complexTable, complexCoefficients, complexFeatures,
    Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_four, fourier4,
    witnessPoint, vertex, ofDecoders, completion, lookup, witnessSamples, witnessProgram]

example := witness_same_observations ((0, 2), true) (by decide)

example := learned_target fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples true 24 (fun x => witnessDecoder x.2)
  witness_realizable.1 witness_realizable.2 (3, 2) true witness_new_input.1

example := accepted_strict fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples (by norm_num : (0 : ℝ) < 24) (witnessPoint_mem true)
  (3, 1) false (by decide) true (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms
