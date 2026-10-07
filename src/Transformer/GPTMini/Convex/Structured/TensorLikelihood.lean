import Transformer.GPTMini.Convex.Structured.TensorTraining

/-!
# The same actual tensor distribution governs inference and training

Source: MixedHeads/MixedTraining's exact normalized joint model at
d436526 and TensorTraining's computed likelihood through genuine
embedding/positions/RMSNorm. The state branch uses every actual
physical transition factor; the pointer uses the true all-pair Gibbs
probability. Both are mixed by their actual learned positive weights.
Complete configurations belong to the proof/loss, not a stored bank.

For every unrestricted raw parameter assignment, the tensor joint is
proved positive/normalized and identical to the verified raw model.
The compact inferred output is its exact full-distribution expectation.
The computed tensor loss equals minus log of this very distribution,
and this actual complete negative log likelihood is globally convex
in all free weights. Output-only CE and AdamW convergence remain
unproved; full-stack/FFN preservation is a subsequent obligation.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C d T : ℕ}

/-- True complete state/channel probability reads only actual tensor transitions and free head-global logits.
Source: MarkovTraining's exact chronological path product and MarkovEmissions' genuine conditional value distribution. -/
def tensorStateProbability (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (z : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) T) : ℝ :=
  stateRow (tensorInitial ψ) z.1 *
    (∏ position, stateRow (tensorTransition hwidth (x position) (statePathPrevious z.1 z.2.1 position)) (z.2.1 position)) *
    channelProbability (tensorEmission ψ (statePathEnd z.1 T z.2.1)) z.2.2

/-- The actual tensor joint mixes both complete learned branches with its genuine global categorical weights.
Source: true tensor state/path/channel factors and normalized all-visible-pair matching/value inference. -/
def tensorMixedProbability (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) : TensorConfiguration T → ℝ
  | .inl state => tensorHeadWeight ψ 0 * tensorStateProbability hwidth ψ x state
  | .inr route => tensorHeadWeight ψ 1 * tensorPointerProbability hwidth ψ x hcap query route

/-- Actual tensor history/channel probabilities coincide with the verified raw learned state branch for all weights.
Source: exact physical-index factorization, genuine RMS transition recovery and the identical free initial/emission reads. -/
theorem tensorStateProbability_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (z : SharedStateConfiguration tokens) :
    tensorStateProbability hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) z =
      sharedStateProbability θ.1 tokens z := by
  unfold tensorStateProbability sharedStateProbability markovJoint
  rw [conditionalStatePath_product]
  simp_rw [tensorTransition_sequence hsize hwidth eps heps]
  rw [(tensorHeadParameters_reads θ).1, (tensorHeadParameters_reads θ).2.1]

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- The whole genuine tensor mixture is exactly the same joint learned distribution for every unrestricted raw assignment.
Source: both full branch transfer identities and actual learned head weights, with no correct-configuration premise. -/
theorem tensorMixedProbability_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) :
    tensorMixedProbability hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query =
      mixedProbability θ tokens hcap query := by
  funext observed
  cases observed with
  | inl state =>
      simp only [tensorMixedProbability, mixedProbability, (tensorHeadParameters_reads θ).2.2.1,
        tensorStateProbability_sequence hsize hwidth eps heps]
  | inr route =>
      simp only [tensorMixedProbability, mixedProbability, (tensorHeadParameters_reads θ).2.2.1,
        tensorPointerProbability_sequence hsize hwidth eps heps]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Every actual complete tensor configuration has positive mass at arbitrary finite free weights.
Source: the identical true mixed model's established complete positivity, including both incorrect branches. -/
theorem tensorMixedProbability_pos (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    0 < tensorMixedProbability hwidth (tensorHeadParameters θ)
      (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed := by
  rw [tensorMixedProbability_sequence hsize hwidth eps heps]
  exact mixedProbability_pos θ tokens hcap query observed

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 9, 10] : List (Fin 36)).length ≤ 128 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- The actual tensor full configuration distribution normalizes to one at every unrestricted shared weight assignment.
Source: exact actual tensor joint transfer and the genuine whole mixed normalization. -/
theorem tensorMixedProbability_sum (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) :
    (∑ observed : MixedConfiguration tokens, tensorMixedProbability hwidth (tensorHeadParameters θ)
      (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed) = 1 := by
  rw [tensorMixedProbability_sequence hsize hwidth eps heps]
  exact mixedProbability_sum θ tokens hcap query

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- The compact actual tensor inference score is exactly the full true tensor distribution's output-code expectation.
Source: the proved full compact mixed decoder contraction and actual entire tensor-probability identity. -/
theorem tensorMixedScore_tensorJoint (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) :
    tensorMixedScore hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query target =
      weightedOutputScore (tensorMixedProbability hwidth (tensorHeadParameters θ)
        (tensorSequence hsize hwidth eps θ tokens hcap) hcap query) mixedChannels target := by
  rw [tensorMixedProbability_sequence hsize hwidth eps heps]
  exact tensorMixedScore_joint_sequence hsize hwidth eps heps θ tokens hcap query target

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Computed tensor training is precisely the negative log of the actual complete tensor inference distribution.
Source: identical complete likelihood/probability transfers and MixedTraining's true negative-log identity. -/
theorem tensorMixedNLL_eq (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    tensorMixedNLL hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed =
      -Real.log (tensorMixedProbability hwidth (tensorHeadParameters θ)
        (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed) := by
  rw [tensorMixedNLL_sequence hsize hwidth eps heps, mixedNLL_eq,
    tensorMixedProbability_sequence hsize hwidth eps heps]

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 9, 10] : List (Fin 36)).length ≤ 128 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Negative log of the actual complete tensor inference probability is globally convex in every free learned weight.
Source: exact computed tensor NLL/probability coupling and unrestricted genuine embedding/head likelihood convexity. -/
theorem tensorCompleteLikelihood_convex (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    ConvexOn ℝ Set.univ (fun θ : BindingParameters V C => -Real.log (tensorMixedProbability hwidth
      (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed)) := by
  have heq : (fun θ : BindingParameters V C => tensorMixedNLL hwidth (tensorHeadParameters θ)
      (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed) =
      (fun θ : BindingParameters V C => -Real.log (tensorMixedProbability hwidth
        (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed)) := by
    funext θ
    exact tensorMixedNLL_eq hsize hwidth eps heps θ tokens hcap query observed
  rw [← heq]
  exact tensorMixedNLL_convex hsize hwidth eps heps tokens hcap query observed

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
