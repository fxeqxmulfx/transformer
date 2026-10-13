import Transformer.GPTMini.Convex.Structured.TensorPointer

/-!
# Actual mixed compact tensor-head output

Source: MixedHeads' genuine normalized learned mixture at d436526
and TensorState/TensorPointer's exact transfer from prenorm tensors.
The combined head uses the same actual two-way learned softmax and
the inferred state/binding channel means. Raw token IDs, semantic
states, target routes and output labels never enter this inference.
All ten output coordinates stay in [-1,1] for arbitrary finite inputs
and free head weights. Their true tied-code dot product is the actual
compact mixed score, including both positive branches.

For genuine normalized embeddings this score equals the verified raw
model's full normalized joint expectation for every unrestricted
parameter assignment. This establishes tensor-level head inference,
with residual/tied final readout, causal row-prefix selection, actual
complete tensor likelihood and full-stack integration still separate.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped BigOperators Classical
noncomputable section

variable {V C d T : ℕ}

/-- True ten-coordinate mixed tensor output uses learned branch weights and actual inferred values.
Source: MixedHeads' genuine branch mixture applied to the two compact tensor-head coordinate arrays. -/
def tensorMixedCoordinates (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (axis : Fin 10) : ℝ :=
  tensorHeadWeight ψ 0 * tensorStateCoordinates hwidth ψ x axis +
    tensorHeadWeight ψ 1 * tensorPointerCoordinates hwidth ψ x hcap query axis

/-- Genuine candidate-token score from the actual tensor branches and learned mixture logits.
Source: the same inferred ten-coordinate values, without a reference answer or configuration in the forward. -/
def tensorMixedScore (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (target : Fin 1024) : ℝ :=
  tensorHeadWeight ψ 0 * tensorStateScore hwidth ψ x target +
    tensorHeadWeight ψ 1 * tensorPointerScore hwidth ψ x hcap query target

/-- Both real tensor-head weights are strictly positive for every finite learned global parameter assignment.
Source: the actual learned two-way softmax, without a supplied simplex constraint. -/
theorem tensorHeadWeight_pos (ψ : TensorHeadParameters C) (head : Fin 2) :
    0 < tensorHeadWeight ψ head := stateRow_pos _ _

/-- The actual tensor mixture weights sum to exactly one.
Source: the true two-way categorical normalizer, rather than semantic hard selection. -/
theorem tensorHeadWeight_sum (ψ : TensorHeadParameters C) :
    tensorHeadWeight ψ 0 + tensorHeadWeight ψ 1 = 1 := by
  simpa only [tensorHeadWeight, Fin.sum_univ_two] using
    stateRow_sum (fun branch => ψ.1 (.inr (.inr (.inr branch))))

/-- At genuine zero initialization neither tensor head is semantically selected in advance.
Source: the actual two-way softmax, with both learned branches receiving mass one half. -/
theorem tensorHeadWeight_zero (head : Fin 2) :
    tensorHeadWeight (((fun _ => 0), (fun _ => 0)) : TensorHeadParameters C) head = 1 / 2 := by
  norm_num [tensorHeadWeight, stateRow, Fin.sum_univ_two]

/-- Every actual tensor state/value coordinate is bounded at arbitrary prenorm inputs and learned weights.
Source: true normalized causal state/channel expectations of each signed-axis decoder statistic. -/
theorem tensorStateCoordinates_bounds (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (axis : Fin 10) :
    -1 ≤ tensorStateCoordinates hwidth ψ x axis ∧ tensorStateCoordinates hwidth ψ x axis ≤ 1 := by
  unfold tensorStateCoordinates tensorStateMean
  split_ifs
  · apply markovValueMean_bounds
    intro value
    fin_cases value <;> norm_num [quarterX]
  · apply markovValueMean_bounds
    intro value
    fin_cases value <;> norm_num [quarterY]

example : (64 : ℕ) ≤ 128 := by omega

/-- Every actual all-pair tensor pointer coordinate is bounded without semantic routing assumptions.
Source: the true normalized compact value expectation over every visible pair and value channel. -/
theorem tensorPointerCoordinates_bounds (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T) (axis : Fin 10) :
    -1 ≤ tensorPointerCoordinates hwidth ψ x hcap query axis ∧
      tensorPointerCoordinates hwidth ψ x hcap query axis ≤ 1 := by
  let : Nonempty (Fin T) := ⟨query⟩
  unfold tensorPointerCoordinates
  exact pointerOutputCoordinates_bounds _ _ _ _ _

example : (64 : ℕ) ≤ 64 ∧ (4 : ℕ) ≤ 64 := by omega

/-- The genuinely inferred mixed tensor value stays inside the same ten-axis decoder cube.
Source: actual positive normalized head weights and both proved compact branch-coordinate bounds. -/
theorem tensorMixedCoordinates_bounds (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T) (axis : Fin 10) :
    -1 ≤ tensorMixedCoordinates hwidth ψ x hcap query axis ∧
      tensorMixedCoordinates hwidth ψ x hcap query axis ≤ 1 := by
  have hs := tensorStateCoordinates_bounds hwidth ψ x axis
  have hp := tensorPointerCoordinates_bounds hwidth ψ x hcap query axis
  have hsLower := mul_le_mul_of_nonneg_left hs.1 (tensorHeadWeight_pos ψ 0).le
  have hsUpper := mul_le_mul_of_nonneg_left hs.2 (tensorHeadWeight_pos ψ 0).le
  have hpLower := mul_le_mul_of_nonneg_left hp.1 (tensorHeadWeight_pos ψ 1).le
  have hpUpper := mul_le_mul_of_nonneg_left hp.2 (tensorHeadWeight_pos ψ 1).le
  have hm := tensorHeadWeight_sum ψ
  unfold tensorMixedCoordinates
  constructor <;> nlinarith

example : (64 : ℕ) ≤ 128 ∧ (4 : ℕ) ≤ 64 := by omega

/-- The actual compact mixed score is precisely the dot product of its ten tensor output axes with the tied code.
Source: both genuine branch-coordinate score identities and finite mixture linearity. -/
theorem tensorMixedScore_axes (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T) (target : Fin 1024) :
    tensorMixedScore hwidth ψ x hcap query target =
      ∑ axis, tensorMixedCoordinates hwidth ψ x hcap query axis * outputCoordinate (outputDigit target) axis := by
  unfold tensorMixedScore
  rw [tensorStateScore_axes, tensorPointerScore_axes]
  simp only [tensorMixedCoordinates, add_mul, Finset.sum_add_distrib,
    mul_assoc, ← Finset.mul_sum]

example : (64 : ℕ) ≤ 64 ∧ (4 : ℕ) ≤ 64 := by omega

/-- Actual tensor inference equals the verified raw model for every unrestricted shared parameter assignment.
Source: true RMS/state/binding transfer and the identical learned two-way mixture, without correct-output premises. -/
theorem tensorMixedScore_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) :
    tensorMixedScore hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query target =
      mixedScore θ tokens hcap query target := by
  unfold tensorMixedScore
  rw [(tensorHeadParameters_reads θ).2.2.1,
    tensorStateScore_sequence hsize hwidth eps heps,
    tensorPointerScore_sequence hsize hwidth eps heps]
  rfl

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- The real compact tensor score equals the entire normalized joint decoder expectation of the same raw learned model.
Source: MixedHeads' proved full-distribution contraction plus exact actual tensor inference transfer. -/
theorem tensorMixedScore_joint_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) :
    tensorMixedScore hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query target =
      weightedOutputScore (mixedProbability θ tokens hcap query) mixedChannels target := by
  rw [tensorMixedScore_sequence hsize hwidth eps heps]
  exact mixedScore_joint θ tokens hcap query target

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
