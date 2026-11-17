module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval.Binding

/-!
# Nonempty learned binding conditions

Source: `review/flat_spectrum.md`, CM111. Two distinct imaginary vectors and a
nonconstant learned relative potential satisfy the actual content, link and
decoder conditions. This mathematical witness is not a native initializer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval

/-- A nonzero complete complex parameter point, CM111. -/
noncomputable def witness : Parameters Bool ℂ :=
  ⟨fun b => if b then -Complex.I else Complex.I,
    fun b => if b then -Complex.I else Complex.I,
    8, fun d => if d = 1 then 16 else 0⟩

/-- Two distinct original past objects, CM111. -/
def witnessPast : Fin 2 → Bool := ![false, true]

/-- Both learned maps satisfy the full capacity hypotheses, CM111. -/
theorem witness_unit : Unit witness := by
  constructor <;> intro b <;> cases b <;> simp [witness]

/-- The repeated first object has a strict actual learned key-score gap,
CM111, derived through the shared key map. -/
theorem witness_key_gap : ∀ j : Fin 2, j ≠ 0 →
    16 ≤ keyScores witness witnessPast false 0 - keyScores witness witnessPast false j := by
  have h := repeated_key_gap witness witnessPast false 0 2 rfl
    (by norm_num [witness]) (by
      intro j hj
      fin_cases j
      · exact (hj rfl).elim
      · norm_num [witness, witnessPast, RCLike.inner_apply])
  simpa only [show witness.scale * (2 : ℝ) = 16 by norm_num [witness]] using h

/-- The same relative potential supplies a strict actual offset preference,
CM111, without asserting that training discovers it. -/
theorem witness_relative_gap : ∀ j : Fin 2, j ≠ 1 →
    16 ≤ relativeScores (n := 2) witness 0 1 - relativeScores witness 0 j := by
  intro j hj
  fin_cases j
  · norm_num [relativeScores, witness]
  · exact (hj rfl).elim

/-- The shared value/readout vectors separate the linked object from
every original rival, CM111. -/
theorem witness_decoder_gap : ∀ c : Bool, c ≠ witnessPast 1 →
    2 ≤ kernel witness (witnessPast 1) (witnessPast 1) - kernel witness c (witnessPast 1) := by
  intro c hc
  cases c
  · norm_num [kernel, witness, witnessPast, RCLike.inner_apply]
  · exact (hc rfl).elim

/-- Elementary exact-real bounds certify positive derived output margin,
CM111. No numerical oracle or floating-point estimate occurs. -/
theorem witness_positive : 0 < (2 : ℝ) - (2 + 2) * (2 * (Real.exp (-16) + Real.exp (-16))) := by
  have he : Real.exp (-16) ≤ (1 / 17 : ℝ) := by
    rw [Real.exp_neg]
    have hi := (inv_le_inv₀ (Real.exp_pos (16 : ℝ)) (by norm_num : (0 : ℝ) < 17)).mpr
      (by linarith [Real.add_one_le_exp (16 : ℝ)])
    simpa using hi
  linarith

/-- All complete original-class comparisons favor the linked object,
CM111. This inhabits the conditional correctness theorem nontrivially. -/
theorem witness_correct : ∀ c : Bool, c ≠ true →
    logit witness witnessPast false c < logit witness witnessPast false true := by
  have h := correct_object witness witness_unit witnessPast false 0 1 16 16 2 1
    (by norm_num) (by norm_num) witness_key_gap witness_relative_gap witness_decoder_gap
    (by simpa using witness_positive)
  simpa only [witnessPast, Matrix.cons_val_one, Matrix.cons_val_zero, one_mul] using h

example : 1 - Perspective.softmaxWeight (keyScores witness witnessPast false) 0 ≤
    (2 : ℝ) * Real.exp (-16) := by
  simpa using softmax_error (keyScores witness witnessPast false) 0 16 witness_key_gap

example : 1 - pathMass witness witnessPast false (0, 1) ≤
    (2 : ℝ) * (Real.exp (-16) + Real.exp (-16)) := by
  simpa using path_error witness witnessPast false 0 1 16 16 witness_key_gap witness_relative_gap

example : (2 : ℝ) - (2 + 2) * (2 * (Real.exp (-16) + Real.exp (-16))) ≤
    logit witness witnessPast false true - logit witness witnessPast false false := by
  simpa [witnessPast] using logit_margin witness witness_unit witnessPast false false 0 1 16 16 2
    (by norm_num) witness_key_gap witness_relative_gap (witness_decoder_gap false (by decide))

example : (3 : ℝ) * (2 - (2 + 2) * (2 * (Real.exp (-16) + Real.exp (-16)))) ≤
    3 * logit witness witnessPast false true - 3 * logit witness witnessPast false false := by
  simpa [witnessPast] using scaled_margin witness witness_unit witnessPast false false 0 1 16 16 2 3
    (by norm_num) (by norm_num) witness_key_gap witness_relative_gap
    (witness_decoder_gap false (by decide))

example : logit witness witnessPast false false < logit witness witnessPast false true :=
  witness_correct false (by decide)

example : Landscape.Neural.crossEntropy (logit witness witnessPast false) true ≤
    Real.log (1 + Real.exp (-((2 : ℝ) - (2 + 2) *
      (2 * (Real.exp (-16) + Real.exp (-16)))))) := by
  have h := crossEntropy_bound witness witness_unit witnessPast false 0 1 16 16 2 1
    (by norm_num) (by norm_num) witness_key_gap witness_relative_gap witness_decoder_gap
  norm_num [witnessPast] at h ⊢
  exact h

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval
