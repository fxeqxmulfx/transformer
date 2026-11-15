module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent.Confidence
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms.Witness

/-!
# Nonconstant confidence improvement with a genuinely unresolved state

Source: `review/flat_spectrum.md`, CM64. CM53's training-learned many-to-one
dictionary has covered and uncovered query states. Bounded confidence lowers
ordinary CE while the uncovered state's opposite feasible answers remain.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent

open scoped BigOperators
open Landscape.Neural (crossEntropy)
open LatentPrograms (Parameters features head witnessProgram witnessSamples)

/-- Scale the observed dictionary's actual joint tensor, including its
declared completion on the one unobserved state, CM64. -/
def witnessPoint (ρ : ℝ) (default : Bool) : Parameters Bool (Fin 4) Bool :=
  ρ • LatentPrograms.witnessPoint default

/-- The entire confidence interval contains feasible nonconstant heads,
CM64. Both possible completions of the unknown state remain admissible. -/
theorem witnessPoint_mem (ρ : ℝ) (default : Bool) (hρ : 1 ≤ ρ) (hR : ρ ≤ 8) :
    witnessPoint ρ default ∈ domain witnessProgram [false, true] witnessSamples 1 8 := by
  exact scaled_mem _ _ _ (LatentPrograms.witnessPoint_mem default) (by linarith) hρ hR

/-- The actual total confidence is the tensor's declared scale, CM64. -/
theorem witness_mass (ρ : ℝ) (default : Bool) : mass (witnessPoint ρ default) = ρ := by
  unfold witnessPoint
  rw [mass_smul]
  norm_num [mass, LatentPrograms.witnessPoint, LatentPrograms.vertex,
    LatentPrograms.ofDecoders, Fintype.sum_bool]

/-- The observed dictionary determines a genuinely new query input,
CM64; its state is covered although the input pair is absent from training. -/
theorem witness_covered :
    headLogits (features fourier4 witnessProgram) (head fourier4 1 (witnessPoint 4 true))
      (3, 2) = (fun c => if c = true then 4 else 0) ∧
      (3, 2) ∉ witnessSamples.map Prod.fst := by
  constructor
  · rw [accepted_logits fourier4 fourier4_unitary witnessProgram [false, true]
      witnessSamples (witnessPoint_mem 4 true (by norm_num) (by norm_num))
      (3, 2) true LatentPrograms.witness_new_input.1, witness_mass]
  · exact LatentPrograms.witness_new_input.2

/-- A larger feasible confidence strictly lowers actual ordinary observed
CE for this nonconstant many-to-one example, CM64. No unseen target is used. -/
theorem witness_ce_improves :
    crossEntropy (headLogits (features fourier4 witnessProgram)
      (head fourier4 1 (witnessPoint 4 true)) (0, 2)) true <
    crossEntropy (headLogits (features fourier4 witnessProgram)
      (head fourier4 1 (witnessPoint 2 true)) (0, 2)) true := by
  rw [observed_ce fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
      (witnessPoint_mem 4 true (by norm_num) (by norm_num)) ((0, 2), true) (by decide),
    observed_ce fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
      (witnessPoint_mem 2 true (by norm_num) (by norm_num)) ((0, 2), true) (by decide)]
  norm_num [witness_mass]
  apply Real.log_lt_log (by positivity)
  have h : Real.exp (-4) < Real.exp (-2) := Real.exp_lt_exp.mpr (by norm_num)
  linarith

/-- Opposite unobserved completions keep every complete observed logit
identical at the same learned confidence, CM64. -/
theorem witness_same_observations (t : (Fin 4 × Fin 4) × Bool) (ht : t ∈ witnessSamples) :
    headLogits (features fourier4 witnessProgram) (head fourier4 1 (witnessPoint 4 false)) t.1 =
      headLogits (features fourier4 witnessProgram) (head fourier4 1 (witnessPoint 4 true)) t.1 := by
  rw [observed_logits fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
      (witnessPoint_mem 4 false (by norm_num) (by norm_num)) t ht,
    observed_logits fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
      (witnessPoint_mem 4 true (by norm_num) (by norm_num)) t ht,
    witness_mass, witness_mass]

/-- Confidence learning retains opposite actual complex margins at a
genuinely unobserved state, CM64. Flatness and lower observed CE do not decide it. -/
theorem witness_unseen_opposite :
    headLogits (features fourier4 witnessProgram) (head fourier4 1 (witnessPoint 4 false))
      (0, 3) false = 4 ∧
    headLogits (features fourier4 witnessProgram) (head fourier4 1 (witnessPoint 4 false))
      (0, 3) true = 0 ∧
    headLogits (features fourier4 witnessProgram) (head fourier4 1 (witnessPoint 4 true))
      (0, 3) false = 0 ∧
    headLogits (features fourier4 witnessProgram) (head fourier4 1 (witnessPoint 4 true))
      (0, 3) true = 4 := by
  norm_num [LatentPrograms.logits fourier4 fourier4_unitary, witnessPoint,
    LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders,
    LatentPrograms.completion, LatentPrograms.lookup, witnessSamples, witnessProgram,
    Fintype.sum_bool]

/-- Nonzero imaginary source and head coordinates enter this actual
complex readout, CM64; the source and raw output sizes are different. -/
theorem witness_imaginary_coordinates :
    features fourier4 witnessProgram (true, (1, true)) (3, 1) = -(1 / 2 : ℝ) ∧
      head fourier4 1 (witnessPoint 4 true) (true, (1, true)) false = -2 := by
  norm_num [features, head, tableHead, complexTable, complexCoefficients, complexFeatures,
    Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_four, fourier4, witnessPoint,
    LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders,
    LatentPrograms.completion, LatentPrograms.lookup, witnessSamples, witnessProgram]

example := witnessPoint_mem 4 true (by norm_num) (by norm_num)

example := witness_same_observations ((0, 2), true) (by decide)

example := observed_ce fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
  (witnessPoint_mem 4 true (by norm_num) (by norm_num)) ((0, 2), true) (by decide)

example := observed_ce_antitone fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples (witnessPoint_mem 2 true (by norm_num) (by norm_num))
  (witnessPoint_mem 4 true (by norm_num) (by norm_num))
  (by norm_num [witness_mass]) ((0, 2), true) (by decide)

-- Four original record IDs include a repeated value; the singleton is a real tail.
example := observed_batch_ce fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples (witnessPoint_mem 4 true (by norm_num) (by norm_num))
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val))
  Finset.univ (by simp) (by intro i hi; fin_cases i <;> decide)

example := observed_batch_ce fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples (witnessPoint_mem 4 true (by norm_num) (by norm_num))
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val))
  {3} (by simp) (by intro i hi; fin_cases i <;> decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent
