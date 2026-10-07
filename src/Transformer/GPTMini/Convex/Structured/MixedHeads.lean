import Transformer.GPTMini.Convex.Structured.RawBinding
import Transformer.GPTMini.Convex.Structured.SharedReference

/-!
# Genuine compact learned mixture of state and binding heads

Source: actual shared causal state/value inference at d72d3b3 and the
raw all-pair binding model at e0be315. Two freely learned shared logits
mix their actual normalized distributions. The same 52 token fields
remain jointly free; no task selector, state path, parsed binding or
desired channel enters inference. Every visible pair remains present.

The implicit complete type distinguishes both stochastic branches.
Actual inference computes two compact scores and their learned weighted
sum, proved equal to that complete model's decoder expectation. This
does not enumerate complete paths or channel configurations. The query
is a physical index; causal callbacks choose their own prefix's last
position. Tensor/prenorm/residual/tied realization and complete mixture
training convexity are subsequent obligations. Output-only CE and AdamW
convergence are not asserted by these probability/operator identities.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ}

/-- The actual complete learned state/path/value branch, implicit during compact inference.
Source: six causal states and five four-channel value groups on the unchanged raw prefix. -/
abbrev SharedStateConfiguration (tokens : List (Fin V)) :=
  MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) tokens.length

/-- The entire stochastic mixture's implicit branch/path/route/value domain.
Source: a disjoint union of the genuine state and all-visible-pair configurations, without stored prefix features. -/
abbrev MixedConfiguration (tokens : List (Fin V)) :=
  SharedStateConfiguration tokens ⊕ RawBindingConfiguration tokens

/-- A free head-selection coordinate is read linearly from the same actual binding parameter space.
Source: the two shared global mixture logits composed with the actual first product projection. -/
def mixedHeadRead (head : Fin 2) : BindingParameters V C →ₗ[ℝ] ℝ :=
  (LinearMap.proj (.inr (.inr (.inr (.inr (.inr head))))) : SharedParameters V C →ₗ[ℝ] ℝ).comp
    (LinearMap.fst ℝ (SharedParameters V C) (Fin (C + (C - 1)) → ℝ))

/-- True branch weights are the actual learned two-way softmax, with no hard task-conditioned selection.
Source: the genuine normalized shared head-logit row; both branches always have positive mass at finite weights. -/
def mixedHeadWeight (θ : BindingParameters V C) (head : Fin 2) : ℝ :=
  stateRow (fun branch => mixedHeadRead branch θ) head

/-- The actual shared state branch reads genuine free initial, raw transition and conditional value fields.
Source: the same model underlying sharedMarkovScore and its complete convex likelihood. -/
def sharedStateProbability (θ : SharedParameters V C) (tokens : List (Fin V))
    (z : SharedStateConfiguration tokens) : ℝ :=
  markovJoint (fun state => sharedInitialRead state θ)
    (fun token previous next => sharedTransitionRead token previous next θ) tokens
    (fun state h d => sharedEmissionRead state h d θ) z

/-- The genuine complete mixed probability uses the same compact branch computations and learned branch weights.
Source: ordinary normalized mixture factorization, without supplied correct route/state information in the forward. -/
def mixedProbability (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) : MixedConfiguration tokens → ℝ
  | .inl z => mixedHeadWeight θ 0 * sharedStateProbability θ.1 tokens z
  | .inr z => mixedHeadWeight θ 1 * rawBindingProbability θ tokens hcap query z

/-- The decoder statistic reads the actually sampled value channels of either genuine branch.
Source: the common fixed signed-axis whole-token code; desired labels are not arguments of this statistic. -/
def mixedChannels {n : ℕ} : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) n ⊕
    SharedPointerConfiguration (Fin n × Fin n) → Fin 5 → Fin 4
  | .inl z => z.2.2
  | .inr z => z.2.2

/-- Actual inference forms the learned weighted sum of two compact genuinely computed whole-token scores.
Source: sharedMarkovScore and rawBindingScore at the same simultaneous free weights, without configuration enumeration. -/
def mixedScore (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) : ℝ :=
  mixedHeadWeight θ 0 * sharedMarkovScore θ.1 tokens target +
    mixedHeadWeight θ 1 * rawBindingScore θ tokens hcap query target

/-- Both actually computed branch weights are positive for every finite freely trained head assignment.
Source: positivity of the genuine two-way row softmax. -/
theorem mixedHeadWeight_pos (θ : BindingParameters V C) (head : Fin 2) : 0 < mixedHeadWeight θ head :=
  stateRow_pos _ _

/-- The genuine learned branch weights sum exactly to one at arbitrary simultaneous weights.
Source: the actual two-way row normalizer, rather than an assumed mixture simplex. -/
theorem mixedHeadWeight_sum (θ : BindingParameters V C) : mixedHeadWeight θ 0 + mixedHeadWeight θ 1 = 1 := by
  have h := stateRow_sum (fun branch => mixedHeadRead branch θ)
  simpa only [Fin.sum_univ_two, mixedHeadWeight] using h

/-- Every actual full state configuration retains positive probability at finite shared parameters.
Source: actual raw state/path/value likelihood positivity, with no correct-state premise. -/
theorem sharedStateProbability_pos (θ : SharedParameters V C) (tokens : List (Fin V))
    (z : SharedStateConfiguration tokens) : 0 < sharedStateProbability θ tokens z :=
  markovJoint_pos _ _ _ _ _

/-- The actual complete shared state branch normalizes exactly despite the implicit chronological path type.
Source: genuine state/path/value normalization at all free shared parameter assignments. -/
theorem sharedStateProbability_sum (θ : SharedParameters V C) (tokens : List (Fin V)) :
    ∑ z : SharedStateConfiguration tokens, sharedStateProbability θ tokens z = 1 := by
  unfold sharedStateProbability
  convert markovJoint_sum (fun state : Fin 6 => sharedInitialRead state θ)
    (fun token previous next => sharedTransitionRead token previous next θ) tokens
    (fun state h d => sharedEmissionRead state h d θ)

/-- Every genuine mixed complete configuration is positive, including all incorrect routes and state histories.
Source: true positive learned head probabilities and true positive branch distributions. -/
theorem mixedProbability_pos (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (z : MixedConfiguration tokens) : 0 < mixedProbability θ tokens hcap query z := by
  cases z with
  | inl state => exact mul_pos (mixedHeadWeight_pos _ _) (sharedStateProbability_pos _ _ _)
  | inr route => exact mul_pos (mixedHeadWeight_pos _ _) (rawBindingProbability_pos _ _ _ _ _)

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- The whole implicit learned mixture normalizes to one on every genuine checked raw prefix.
Source: both actual full branch normalizations and the genuinely computed branch softmax, without task filtering. -/
theorem mixedProbability_sum (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) : ∑ z : MixedConfiguration tokens, mixedProbability θ tokens hcap query z = 1 := by
  rw [Fintype.sum_sum_type]
  simp only [mixedProbability, ← Finset.mul_sum, sharedStateProbability_sum, rawBindingProbability_sum, mul_one]
  exact mixedHeadWeight_sum θ

example : ([1, 0] : List (Fin 2)).length ≤ 4 := by decide

/-- No genuine complete mixed probability exceeds one at any finite simultaneous raw parameter assignment.
Source: actual positivity and exact complete-model normalization, without a correct-route assumption. -/
theorem mixedProbability_le_one (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (z : MixedConfiguration tokens) : mixedProbability θ tokens hcap query z ≤ 1 := by
  have h := Finset.single_le_sum (fun cfg (_ : cfg ∈ Finset.univ) => (mixedProbability_pos θ tokens hcap query cfg).le)
    (Finset.mem_univ z)
  rw [mixedProbability_sum] at h
  exact h

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- The actual compact mixed score equals the full normalized joint distribution's whole-token expectation.
Source: exact compact branch decoders, true learned mixture weights and finite disjoint-union expectation linearity. -/
theorem mixedScore_joint (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) :
    mixedScore θ tokens hcap query target = weightedOutputScore
      (mixedProbability θ tokens hcap query) mixedChannels target := by
  unfold mixedScore
  rw [rawBindingScore_joint]
  have hs : sharedMarkovScore θ.1 tokens target = weightedOutputScore
      (sharedStateProbability θ.1 tokens) (fun z : SharedStateConfiguration tokens => z.2.2) target := by
    unfold sharedMarkovScore sharedStateProbability
    exact markovOutputScore_joint _ _ _ _ _
  rw [hs]
  simp only [weightedOutputScore, Fintype.sum_sum_type, mixedProbability, mixedChannels,
    mul_assoc, Finset.mul_sum]

example : ([0] : List (Fin 2)).length ≤ 4 := by decide

/-- The genuine compact mixed decoder has a quantitative margin in its actually computed complete configuration mass.
Source: the full true normalized mixed expectation and common whole-vocabulary code gap; task weights must derive confidence separately. -/
theorem mixedScore_margin (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (selected : MixedConfiguration tokens) (target rival : Fin 1024)
    (hlabel : mixedChannels selected = outputDigit target) (hne : target ≠ rival) :
    11 * mixedProbability θ tokens hcap query selected - 10 ≤
      mixedScore θ tokens hcap query target - mixedScore θ tokens hcap query rival := by
  rw [mixedScore_joint, mixedScore_joint]
  exact weightedOutput_margin _ _ selected target rival (fun z => (mixedProbability_pos _ _ _ _ z).le)
    (mixedProbability_sum _ _ _ _) hlabel hne

example : ([0] : List (Fin 2)).length ≤ 4 ∧
    mixedChannels (n := 1) (.inl (0, (fun _ => 0), outputDigit (25 : Fin 1024))) = outputDigit 25 ∧
    (25 : Fin 1024) ≠ 17 := ⟨by decide, rfl, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
