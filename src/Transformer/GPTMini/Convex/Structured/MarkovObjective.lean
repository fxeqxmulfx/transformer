import Transformer.GPTMini.Convex.Structured.MarkovEmissions

/-!
# Convex joint initial/transition/value training

Source: the actual compact state/output model at 33e99c8, its genuine
path likelihood at 11899f6 and Structured.Basic's finite affine row CE.
Initial, token-conditioned transition and conditional emission logits
all read linear shared raw parameters. The computed complete NLL is a
sum of small row categorical losses and equals the negative log of the
same full normalized distribution whose compact inference is proved.

All these parameter groups train jointly on the unrestricted real
domain. There is no frozen encoder/state/value table premise. Observed
intermediate states and channel assignments are additional fixed data
targets, absent from inference. No marginal output CE convexity, AdamW
convergence, raw Basis capacity or residual/tied decoding is asserted.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {E S A H D : Type*} [AddCommGroup E] [Module ℝ E]
  [Fintype S] [Nonempty S] [Fintype H] [Fintype D] [Nonempty D]

/-- The actual compact observed conditional-channel objective uses one small row loss per value group.
Source: the proposed free emission potentials, with targets external to inference. -/
def channelNLL (linear : H → D → E →ₗ[ℝ] ℝ) (observed : H → D) (θ : E) : ℝ :=
  ∑ h, stateRowNLL (linear h) (observed h) θ

/-- The compact groupwise loss is exactly the actual joint conditional-output negative log probability.
Source: the true factorial normalizer's product of positive small row sums and the exponential additive channel energy. -/
theorem channelNLL_eq (linear : H → D → E →ₗ[ℝ] ℝ) (observed : H → D) (θ : E) :
    channelNLL linear observed θ =
      -Real.log (channelProbability (fun h d => linear h d θ) observed) := by
  have hp : ∀ h : H, 0 < ∑ d, Real.exp (linear h d θ) := fun _ =>
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hlog : Real.log (channelPartition (fun h d => linear h d θ)) =
      ∑ h, Real.log (∑ d, Real.exp (linear h d θ)) :=
    Real.log_prod (fun h _ => (hp h).ne')
  unfold channelProbability
  rw [Real.log_div (Real.exp_pos _).ne' (channelPartition_pos _).ne', Real.log_exp, hlog]
  simp only [channelNLL, stateRowNLL, channelEnergy, Finset.sum_sub_distrib]
  ring

/-- Conditional output values are jointly convex in every shared raw trainable parameter.
Source: the actual compact sum of affine categorical losses, including all learned emission channels. -/
theorem channelNLL_convex (linear : H → D → E →ₗ[ℝ] ℝ) (observed : H → D) :
    ConvexOn ℝ Set.univ (channelNLL linear observed) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  simp only [channelNLL, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro h _
  exact (stateRowNLL_convex (linear h) (observed h)).2 hx hy ha hb hab

omit [Nonempty D] in
/-- Each actual observed-channel loss is nonnegative for any shared raw parameter assignment.
Source: the true local row likelihood inequalities, without a correct-output assumption. -/
theorem channelNLL_nonneg (linear : H → D → E →ₗ[ℝ] ℝ) (observed : H → D) (θ : E) :
    0 ≤ channelNLL linear observed θ :=
  Finset.sum_nonneg (fun h _ => stateRowNLL_nonneg (linear h) (observed h) θ)

/-- Complete compact training loss, jointly reading initial, token transition and final-state emission parameters.
Source: the true observed initial/path/output factorization; a supplied solved state is never an inference input. -/
def markovNLL (initial : S → E →ₗ[ℝ] ℝ) (transition : A → S → S → E →ₗ[ℝ] ℝ)
    (emission : S → H → D → E →ₗ[ℝ] ℝ) (tokens : List A)
    (observed : MarkovConfiguration S H D tokens.length) (θ : E) : ℝ :=
  stateRowNLL initial observed.1 θ + conditionalStateNLL transition observed.1 tokens observed.2.1 θ +
    channelNLL (emission (statePathEnd observed.1 tokens.length observed.2.1)) observed.2.2 θ

/-- The computed complete loss is exactly the negative log of the actual jointly learned state/path/value model.
Source: genuine initial/transition/channel probabilities and the exact positive log-product identity. -/
theorem markovNLL_eq (initial : S → E →ₗ[ℝ] ℝ) (transition : A → S → S → E →ₗ[ℝ] ℝ)
    (emission : S → H → D → E →ₗ[ℝ] ℝ) (tokens : List A)
    (observed : MarkovConfiguration S H D tokens.length) (θ : E) :
    markovNLL initial transition emission tokens observed θ =
      -Real.log (markovJoint (fun state => initial state θ)
        (fun token previous next => transition token previous next θ) tokens
        (fun state h d => emission state h d θ) observed) := by
  unfold markovNLL markovJoint
  rw [Real.log_mul (mul_pos (stateRow_pos _ _) (conditionalStatePath_pos _ _ _ _)).ne'
    (channelProbability_pos _ _).ne',
    Real.log_mul (stateRow_pos _ _).ne' (conditionalStatePath_pos _ _ _ _).ne',
    stateRowNLL_eq, conditionalStateNLL_eq, channelNLL_eq]
  ring

/-- Complete state/value training is globally convex jointly in unrestricted initial, transition and emission raw weights.
Source: the exact actual complete NLL as a sum of affine row losses, with one shared parameter space and fixed data targets. -/
theorem markovNLL_convex (initial : S → E →ₗ[ℝ] ℝ) (transition : A → S → S → E →ₗ[ℝ] ℝ)
    (emission : S → H → D → E →ₗ[ℝ] ℝ) (tokens : List A)
    (observed : MarkovConfiguration S H D tokens.length) :
    ConvexOn ℝ Set.univ (markovNLL initial transition emission tokens observed) :=
  ((stateRowNLL_convex initial observed.1).add
    (conditionalStateNLL_convex transition observed.1 tokens observed.2.1)).add
      (channelNLL_convex (emission (statePathEnd observed.1 tokens.length observed.2.1)) observed.2.2)

omit [Nonempty S] [Nonempty D] in
/-- The actual complete joint objective is nonnegative at every freely trained parameter assignment.
Source: the genuine initial, path and conditional-emission categorical losses. -/
theorem markovNLL_nonneg (initial : S → E →ₗ[ℝ] ℝ) (transition : A → S → S → E →ₗ[ℝ] ℝ)
    (emission : S → H → D → E →ₗ[ℝ] ℝ) (tokens : List A)
    (observed : MarkovConfiguration S H D tokens.length) (θ : E) :
    0 ≤ markovNLL initial transition emission tokens observed θ :=
  add_nonneg (add_nonneg (stateRowNLL_nonneg _ _ _) (conditionalStateNLL_nonneg _ _ _ _ _))
    (channelNLL_nonneg _ _ _)

/-- A finite minibatch shares all raw initial, token transition and conditional value parameters.
Source: the proposed ordinary-gradient complete-likelihood training, for possibly different raw prefix lengths. -/
def markovBatchNLL {B : Type*} [Fintype B] (initial : S → E →ₗ[ℝ] ℝ)
    (transition : A → S → S → E →ₗ[ℝ] ℝ) (emission : S → H → D → E →ₗ[ℝ] ℝ)
    (tokens : B → List A) (observed : (b : B) → MarkovConfiguration S H D (tokens b).length) (θ : E) : ℝ :=
  ∑ b, markovNLL initial transition emission (tokens b) (observed b) θ

/-- Full shared-table minibatch training stays convex with every initial, transition and value group simultaneously unfrozen.
Source: finite sums of the same actual complete joint likelihood, rather than a fixed correct encoder premise. -/
theorem markovBatchNLL_convex {B : Type*} [Fintype B] (initial : S → E →ₗ[ℝ] ℝ)
    (transition : A → S → S → E →ₗ[ℝ] ℝ) (emission : S → H → D → E →ₗ[ℝ] ℝ)
    (tokens : B → List A) (observed : (b : B) → MarkovConfiguration S H D (tokens b).length) :
    ConvexOn ℝ Set.univ (markovBatchNLL initial transition emission tokens observed) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  simp only [markovBatchNLL, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro sample _
  exact (markovNLL_convex initial transition emission (tokens sample) (observed sample)).2 hx hy ha hb hab

omit [Nonempty S] [Nonempty D] in
/-- The shared minibatch objective is nonnegative at any simultaneous initial, transition and emission weights.
Source: finite sums of the actual complete learned-model negative log probabilities. -/
theorem markovBatchNLL_nonneg {B : Type*} [Fintype B] (initial : S → E →ₗ[ℝ] ℝ)
    (transition : A → S → S → E →ₗ[ℝ] ℝ) (emission : S → H → D → E →ₗ[ℝ] ℝ)
    (tokens : B → List A) (observed : (b : B) → MarkovConfiguration S H D (tokens b).length) (θ : E) :
    0 ≤ markovBatchNLL initial transition emission tokens observed θ := by
  unfold markovBatchNLL
  exact Finset.sum_nonneg (fun sample _ => markovNLL_nonneg initial transition emission
    (tokens sample) (observed sample) θ)

/-- A concrete three-token six-state/five-group loss permits arbitrary joint shared raw parameter lookups.
Source: the proposed compact model's full complete-likelihood objective, with no solved-path premise on the weights. -/
example (initial : Fin 6 → E →ₗ[ℝ] ℝ) (transition : Fin 3 → Fin 6 → Fin 6 → E →ₗ[ℝ] ℝ)
    (emission : Fin 6 → Fin 5 → Fin 4 → E →ₗ[ℝ] ℝ) :
    ConvexOn ℝ Set.univ (markovNLL initial transition emission [0, 1, 2] (0, (![1, 2, 3], fun _ => 0))) := by
  convert markovNLL_convex initial transition emission [0, 1, 2] (0, (![1, 2, 3], fun _ => 0))

end
end Transformer.GPTMini.Convex.Structured
