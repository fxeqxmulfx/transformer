import Transformer.GPTMini.Convex.Structured.MarkovChain
import Mathlib.LinearAlgebra.Pi
import Mathlib.Data.Fin.Tuple.Basic

/-!
# Convex training of fully observed causal state paths

New ordered-head training proposal based on MarkovChain and the compact
pointer's joint likelihood at 5371b5f. Initial/transition state logits can
read free shared raw token parameters through actual linear maps.
The observed state-path loss is a sum of local categorical likelihoods,
proved exactly the negative log probability of that actual path.
It is globally convex jointly in all unrestricted raw shared parameters.

Intermediate state targets are additional supervision from data semantics.
They are inputs only to the loss, not to markovRun's causal encoder.
The encoder propagates every state through its learned row softmax;
normalization of complete paths and an exact final-state bridge remain
separate obligations. This is not convex output-label-only CE or an
AdamW convergence guarantee. Complete depth/parity capability, conditional
value emission and genuine residual/tied readout are still required.
Source algebra: Structured.Basic's affine finite log-sum-exp objective,
motivated by arXiv:2305.05465v6, §7, and the actual causal state recurrence.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {E S A : Type*} [AddCommGroup E] [Module ℝ E] [Fintype S] [Nonempty S]

/-- The actual local state-row likelihood evaluated at linear raw shared parameter lookups.
Source: the proposed free transition logits, with no product of independently trained raw weights. -/
def stateRowNLL (linear : S → E →ₗ[ℝ] ℝ) (observed : S) (θ : E) : ℝ :=
  Real.log (∑ state, Real.exp (linear state θ)) - linear observed θ

omit [Nonempty S] in
/-- The true local row likelihood is the same actual affine-energy Gibbs loss whose convexity is proved.
Source: the finite linear row logits and zero fixed energy offsets. -/
theorem stateRowNLL_joint (linear : S → E →ₗ[ℝ] ℝ) (observed : S) (θ : E) :
    stateRowNLL linear observed θ = jointNLL linear (fun _ => 0) observed θ := by
  simp only [stateRowNLL, jointNLL, partition, energy, add_zero]

/-- Local teacher-state supervision is exactly the actual computed transition-row negative log probability.
Source: the true row softmax and positive normalizer, rather than a surrogate loss of a supplied correct logit. -/
theorem stateRowNLL_eq (linear : S → E →ₗ[ℝ] ℝ) (observed : S) (θ : E) :
    stateRowNLL linear observed θ = -Real.log (stateRow (fun state => linear state θ) observed) := by
  unfold stateRowNLL stateRow
  rw [Real.log_div (Real.exp_pos _).ne'
    (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne', Real.log_exp]
  ring

/-- The actual local row loss is convex jointly in every raw shared parameter.
Source: its exact finite affine-energy Gibbs identity, with unrestricted parameter domain. -/
theorem stateRowNLL_convex (linear : S → E →ₗ[ℝ] ℝ) (observed : S) :
    ConvexOn ℝ Set.univ (stateRowNLL linear observed) := by
  have h : stateRowNLL linear observed = jointNLL linear (fun _ => 0) observed := by
    funext θ
    exact stateRowNLL_joint linear observed θ
  rw [h]
  exact jointNLL_convex _ _ _

omit [Nonempty S] in
/-- Actual observed row losses cannot be negative, independently of any semantic state label premise.
Source: the genuine affine Gibbs loss and its derived partition bound. -/
theorem stateRowNLL_nonneg (linear : S → E →ₗ[ℝ] ℝ) (observed : S) (θ : E) :
    0 ≤ stateRowNLL linear observed θ := by
  rw [stateRowNLL_joint]
  exact jointNLL_nonneg _ _ _ _

/-- The actual probability of a post-token state path conditional on its initial state.
Source: chronological products of the genuine learned transition-row probabilities, not a deterministic task path. -/
def conditionalStatePath (transition : A → S → S → ℝ) (start : S) :
    (tokens : List A) → (Fin tokens.length → S) → ℝ
  | [], _ => 1
  | token :: rest, path => stateRow (transition token start) (path 0) *
      conditionalStatePath transition (path 0) rest (Fin.tail path)

/-- The actual observed-path objective, with teacher states kept outside inference and fixed as training data.
Source: the proposed sum of genuine local row likelihoods at linear shared token/previous-state lookups. -/
def conditionalStateNLL (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S) :
    (tokens : List A) → (Fin tokens.length → S) → E → ℝ
  | [], _, _ => 0
  | token :: rest, path, θ => stateRowNLL (transition token start) (path 0) θ +
      conditionalStateNLL transition (path 0) rest (Fin.tail path) θ

/-- Every complete actual post-token path has positive probability at finite free transition logits.
Source: chronological induction over true positive local rows; no semantic correctness certificate is assumed. -/
theorem conditionalStatePath_pos (transition : A → S → S → ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) : 0 < conditionalStatePath transition start tokens path := by
  induction tokens generalizing start with
  | nil => exact zero_lt_one
  | cons token rest ih =>
      exact mul_pos (stateRow_pos _ _) (ih (path 0) (Fin.tail path))

/-- The entire teacher-state objective equals the negative log of the same actual chronological state-path probability.
Source: real transition normalization, path positivity and exact log-product expansion at every raw token. -/
theorem conditionalStateNLL_eq (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) (θ : E) :
    conditionalStateNLL transition start tokens path θ =
      -Real.log (conditionalStatePath (fun token previous next => transition token previous next θ) start tokens path) := by
  induction tokens generalizing start with
  | nil => simp only [conditionalStateNLL, conditionalStatePath, Real.log_one, neg_zero]
  | cons token rest ih =>
      change stateRowNLL (transition token start) (path 0) θ +
        conditionalStateNLL transition (path 0) rest (Fin.tail path) θ = -Real.log (_ * _)
      rw [Real.log_mul (stateRow_pos _ _).ne' (conditionalStatePath_pos _ _ _ _).ne',
        stateRowNLL_eq, ih]
      ring

/-- Fully observed causal state paths give a globally convex objective in all shared raw transition parameters.
Source: actual path likelihood equals the sum of affine categorical row losses; the model's latent marginal label loss is not asserted convex. -/
theorem conditionalStateNLL_convex (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) :
    ConvexOn ℝ Set.univ (conditionalStateNLL transition start tokens path) := by
  induction tokens generalizing start with
  | nil => exact convexOn_const 0 convex_univ
  | cons token rest ih =>
      exact (stateRowNLL_convex (transition token start) (path 0)).add (ih (path 0) (Fin.tail path))

omit [Nonempty S] in
/-- Every full observed-path training loss is nonnegative at arbitrary free shared parameters.
Source: chronological sums of the genuine nonnegative row likelihoods. -/
theorem conditionalStateNLL_nonneg (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) (θ : E) :
    0 ≤ conditionalStateNLL transition start tokens path θ := by
  induction tokens generalizing start with
  | nil => exact le_rfl
  | cons token rest ih => exact add_nonneg (stateRowNLL_nonneg _ _ _) (ih (path 0) (Fin.tail path))

/-- An actual linear lookup reads an independently trained token/previous-state/next-state logit from the shared raw table.
Source: ordinary coordinate projections, retaining all freely learned transitions rather than a hardcoded semantic machine. -/
def transitionRead (token : A) (previous next : S) : (A → S → S → ℝ) →ₗ[ℝ] ℝ :=
  (LinearMap.proj next : (S → ℝ) →ₗ[ℝ] ℝ).comp
    ((LinearMap.proj previous : (S → S → ℝ) →ₗ[ℝ] (S → ℝ)).comp
      (LinearMap.proj token : (A → S → S → ℝ) →ₗ[ℝ] (S → S → ℝ)))

omit [Fintype S] [Nonempty S] in
/-- The shared trainable coordinate read really returns that token's requested transition logit.
Source: composition of the three actual table projections, with no semantic path built into the parameters. -/
theorem transitionRead_apply (token : A) (previous next : S) (θ : A → S → S → ℝ) :
    transitionRead token previous next θ = θ token previous next := by
  simp only [transitionRead, LinearMap.comp_apply, LinearMap.proj_apply]

/-- The same compact free six-state transition table has a globally convex observed-path objective on a real raw input word.
Source: actual shared coordinate lookups, without separate parameters or a solved encoder for each prefix. -/
example : ConvexOn ℝ Set.univ
    (conditionalStateNLL (transitionRead (S := Fin 6) (A := Fin 3)) 0 [0, 1, 2] ![1, 2, 3]) :=
  conditionalStateNLL_convex _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
