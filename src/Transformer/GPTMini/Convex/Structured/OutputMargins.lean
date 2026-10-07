import Transformer.GPTMini.Convex.Structured.OutputCodes
import Transformer.GPTMini.Convex.Structured.MarkovConfidence

/-!
# Whole-vocabulary margins of actual learned output expectations

Source: the ten-coordinate output code in OutputCodes and the actual
normalized full Markov state/path/channel model at 61226da. A selected
configuration's true mass gives a margin against every other vocabulary
token; competing configurations may carry arbitrary output assignments.
The proof bounds their whole contribution, not only answer-label pairs.

The compact decoder is evaluated directly from actual learned forward
state masses and conditional channel expectations. Its equality with
the complete joint-model expectation is proved, so confidence bounds
apply to the same inference used by the head. These are local readout
results; raw shared weights and actual residual/tied integration remain.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped BigOperators Classical
noncomputable section

variable {R : Type*} [Fintype R]

/-- The exact whole-token score of a normalized latent channel model.
Source: the ten actual signed-axis decoder coordinates; latent channel assignments need not be vocabulary IDs. -/
def weightedOutputScore (weight : R → ℝ) (channels : R → Fin 5 → Fin 4) (target : Fin 1024) : ℝ :=
  ∑ z, weight z * outputCodeScore (channels z) (outputDigit target)

/-- Actual selected configuration mass quantitatively controls every whole-vocabulary score difference.
Source: the correct code's gap at least one, arbitrary code contrast at least minus ten, and genuine probability normalization. -/
theorem weightedOutput_margin (weight : R → ℝ) (channels : R → Fin 5 → Fin 4)
    (selected : R) (target rival : Fin 1024) (hnonneg : ∀ z, 0 ≤ weight z)
    (hnorm : ∑ z, weight z = 1) (hlabel : channels selected = outputDigit target)
    (hne : target ≠ rival) :
    11 * weight selected - 10 ≤ weightedOutputScore weight channels target - weightedOutputScore weight channels rival := by
  have hcorrect : 1 ≤ outputCodeScore (channels selected) (outputDigit target) -
      outputCodeScore (channels selected) (outputDigit rival) := by
    rw [hlabel, outputCodeScore_self]
    have hgap := outputToken_gap target rival hne
    linarith
  have hcontrast : ∀ z, -10 ≤ outputCodeScore (channels z) (outputDigit target) -
      outputCodeScore (channels z) (outputDigit rival) := by
    intro z
    have ha := (outputCodeScore_bounds (channels z) (outputDigit target)).1
    have hb := (outputCodeScore_bounds (channels z) (outputDigit rival)).2
    linarith
  have htail := Finset.sum_le_sum (fun z (_ : z ∈ Finset.univ.erase selected) =>
    mul_le_mul_of_nonneg_left (hcontrast z) (hnonneg z))
  rw [← Finset.sum_mul] at htail
  have hmass := Finset.sum_erase_add Finset.univ weight (Finset.mem_univ selected)
  rw [hnorm] at hmass
  have hsplit := Finset.sum_erase_add Finset.univ (fun z => weight z *
    (outputCodeScore (channels z) (outputDigit target) - outputCodeScore (channels z) (outputDigit rival)))
    (Finset.mem_univ selected)
  have hselected := mul_le_mul_of_nonneg_left hcorrect (hnonneg selected)
  rw [mul_one] at hselected
  have hexpect : weightedOutputScore weight channels target - weightedOutputScore weight channels rival =
      ∑ z, weight z * (outputCodeScore (channels z) (outputDigit target) - outputCodeScore (channels z) (outputDigit rival)) := by
    simp only [weightedOutputScore, mul_sub, Finset.sum_sub_distrib]
  rw [hexpect]
  linarith

example : (∀ z : Fin 2, 0 ≤ (fun _ => (1 / 2 : ℝ)) z) ∧
    (∑ z : Fin 2, (fun _ => (1 / 2 : ℝ)) z) = 1 ∧
    (fun _ : Fin 2 => outputDigit (25 : Fin 1024)) 0 = outputDigit 25 ∧ (25 : Fin 1024) ≠ 17 := by
  constructor
  · intro z; norm_num
  · norm_num [Fin.sum_univ_two]

/-- True selected mass above ten elevenths gives strict integer-token score ordering against every rival.
Source: the actual normalized whole-distribution decoder margin, without a correct-logits premise. -/
theorem weightedOutput_strict (weight : R → ℝ) (channels : R → Fin 5 → Fin 4)
    (selected : R) (target rival : Fin 1024) (hnonneg : ∀ z, 0 ≤ weight z)
    (hnorm : ∑ z, weight z = 1) (hlabel : channels selected = outputDigit target)
    (hne : target ≠ rival) (hconfidence : (10 / 11 : ℝ) < weight selected) :
    weightedOutputScore weight channels rival < weightedOutputScore weight channels target := by
  have h := weightedOutput_margin weight channels selected target rival hnonneg hnorm hlabel hne
  linarith

example : (∀ z : Fin 2, 0 ≤ (if z = 0 then (99 / 100 : ℝ) else 1 / 100)) ∧
    (∑ z : Fin 2, if z = 0 then (99 / 100 : ℝ) else 1 / 100) = 1 ∧
    (fun _ : Fin 2 => outputDigit (25 : Fin 1024)) 0 = outputDigit 25 ∧
    (25 : Fin 1024) ≠ 17 ∧ (10 / 11 : ℝ) < (99 / 100 : ℝ) := by
  constructor
  · intro z; fin_cases z <;> norm_num
  · norm_num [Fin.sum_univ_two]

variable {S A : Type*} [Fintype S] [Nonempty S]

/-- Real compact learned state/value output is scored by the actual ten decoder axes.
Source: MarkovEmissions.markovValueMean and the same signed-axis token code used by full configuration expectations. -/
def markovOutputScore (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A)
    (emission : S → Fin 5 → Fin 4 → ℝ) (target : Fin 1024) : ℝ :=
  ∑ h, (markovValueMean initial transition tokens emission h quarterX * quarterX (outputDigit target h) +
    markovValueMean initial transition tokens emission h quarterY * quarterY (outputDigit target h))

omit [Nonempty S] in
/-- The compact actual ten-coordinate score equals the genuine full normalized joint-model decoder expectation.
Source: exact path/channel contraction and finite expectation linearity, with all initial/transition/value parameters free. -/
theorem markovOutputScore_joint (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A)
    (emission : S → Fin 5 → Fin 4 → ℝ) (target : Fin 1024) :
    markovOutputScore initial transition tokens emission target =
      weightedOutputScore (markovJoint initial transition tokens emission) (fun z => z.2.2) target := by
  unfold weightedOutputScore outputCodeScore
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  unfold markovOutputScore
  apply Finset.sum_congr rfl
  intro h hh
  simp only [outputPairScore, mul_add, Finset.sum_add_distrib, ← mul_assoc, ← Finset.sum_mul]
  have hx : (∑ z : MarkovConfiguration S (Fin 5) (Fin 4) tokens.length,
      markovJoint initial transition tokens emission z * quarterX (z.2.2 h)) =
      markovValueMean initial transition tokens emission h quarterX := by
    convert markovJoint_mean initial transition tokens emission h quarterX
  have hy : (∑ z : MarkovConfiguration S (Fin 5) (Fin 4) tokens.length,
      markovJoint initial transition tokens emission z * quarterY (z.2.2 h)) =
      markovValueMean initial transition tokens emission h quarterY := by
    convert markovJoint_mean initial transition tokens emission h quarterY
  rw [hx, hy]

/-- The actual compact learned output has the derived whole-vocabulary quantitative margin.
Source: genuine joint normalization/positivity and the proved actual forward/decoder contraction identity. -/
theorem markovOutput_margin (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A)
    (emission : S → Fin 5 → Fin 4 → ℝ) (selected : MarkovConfiguration S (Fin 5) (Fin 4) tokens.length)
    (target rival : Fin 1024) (hlabel : selected.2.2 = outputDigit target) (hne : target ≠ rival) :
    11 * markovJoint initial transition tokens emission selected - 10 ≤
      markovOutputScore initial transition tokens emission target - markovOutputScore initial transition tokens emission rival := by
  rw [markovOutputScore_joint, markovOutputScore_joint]
  have hn : ∀ z : MarkovConfiguration S (Fin 5) (Fin 4) tokens.length,
      0 ≤ markovJoint initial transition tokens emission z :=
    fun z => (markovJoint_pos initial transition tokens emission z).le
  have hs : (∑ z : MarkovConfiguration S (Fin 5) (Fin 4) tokens.length,
      markovJoint initial transition tokens emission z) = 1 := by
    convert markovJoint_sum initial transition tokens emission
  exact weightedOutput_margin _ _ selected target rival hn hs hlabel hne

example : ((0 : Fin 6), ((fun _ : Fin 3 => (0 : Fin 6)), outputDigit (25 : Fin 1024))).2.2 = outputDigit 25 ∧
    (25 : Fin 1024) ≠ 17 := by decide

/-- Explicit finite learned tables uniformly give strict actual decoder margins at the largest Basis context.
Source: the true 1-794*exp(-gain) joint bound and ten-axis whole-vocabulary margins; task reference correctness is still separate. -/
theorem sharpReference_output_strict (rule : A → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (tokens : List A) (gain : ℝ) (rival : Fin 1024) (hT : tokens.length ≤ 128)
    (hne : labels (referenceRun rule start tokens) ≠ rival) (hgain : 794 * Real.exp (-gain) < (1 / 11 : ℝ)) :
    markovOutputScore (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens
      (sharpReferenceEmission (fun state => outputDigit (labels state)) gain) rival <
    markovOutputScore (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens
      (sharpReferenceEmission (fun state => outputDigit (labels state)) gain) (labels (referenceRun rule start tokens)) := by
  have h := sharpReference_context_bound rule (fun state => outputDigit (labels state)) start tokens gain hT
  have hm := markovOutput_margin (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens
    (sharpReferenceEmission (fun state => outputDigit (labels state)) gain)
    (start, (referencePath rule start tokens, outputDigit (labels (referenceRun rule start tokens))))
    (labels (referenceRun rule start tokens)) rival rfl hne
  linarith

example : ([0, 1, 2] : List (Fin 3)).length ≤ 128 ∧ (25 : Fin 1024) ≠ 17 ∧
    794 * Real.exp (-(Real.log 100000)) < (1 / 11 : ℝ) := by
  constructor
  · decide
  · constructor
    · decide
    · rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 100000)]
      norm_num

end
end Transformer.GPTMini.Convex.Structured
