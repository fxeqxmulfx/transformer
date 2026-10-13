import Transformer.GPTMini.Convex.Structured.MixedHeads

/-!
# Globally convex complete training of the genuine learned mixture

Source: shared state/value likelihood at bdbea00, unchanged raw binding
likelihood at e0be315 and the actual normalized mixture in MixedHeads.
Observed branch/path/route/channel labels are fixed data supervision.
The computed loss is the branch categorical NLL plus that actual
branch's compact complete NLL. It equals minus log of the very same
mixed probability whose compact inference/decoder identity is proved.

All shared raw token Q/K/transition/value fields, initial/emission
logits, absolute/relative positions, chronology and mixture logits are
free simultaneously on the complete real product space. The objective
and shared variable-length minibatches are globally convex there; no
freezing, optimizer replacement, projection or head search is involved.
Correct semantic label generation remains data-only. This proves neither
convex output-only CE nor strong convexity, AdamW convergence, numerical
equivalence or a full prenorm/residual/tied tensor-stack implementation.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ}

/-- The actual compact state branch likelihood reads the first component of the same unrestricted mixed parameter domain.
Source: genuine shared initial/token-transition/emission NLL; relative binding weights belong to the other branch. -/
def bindingStateNLL (tokens : List (Fin V)) (observed : SharedStateConfiguration tokens)
    (θ : BindingParameters V C) : ℝ :=
  markovNLL sharedInitialRead sharedTransitionRead sharedEmissionRead tokens observed θ.1

/-- The lifted actual state branch loss is exactly its genuine complete probability's negative log.
Source: sharedMarkovNLL_eq on the actual shared coordinates, without supplied correct hidden states in inference. -/
theorem bindingStateNLL_eq (tokens : List (Fin V)) (observed : SharedStateConfiguration tokens)
    (θ : BindingParameters V C) : bindingStateNLL tokens observed θ = -Real.log (sharedStateProbability θ.1 tokens observed) := by
  unfold bindingStateNLL sharedStateProbability
  exact markovNLL_eq _ _ _ _ _ _

/-- The actual state branch stays globally convex on the entire mixed domain, including freely varying relative-binding coordinates.
Source: genuine shared state NLL convexity under the first product linear projection. -/
theorem bindingStateNLL_convex (tokens : List (Fin V)) (observed : SharedStateConfiguration tokens) :
    ConvexOn ℝ Set.univ (bindingStateNLL (C := C) tokens observed) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  exact (sharedMarkovNLL_convex (C := C) tokens observed).2 hx hy ha hb hab

/-- The actual complete mixture loss is computed from branch CE and that branch's genuine compact likelihood.
Source: learned normalized mixture factorization with loss-only observed branch/path/route/channel supervision. -/
def mixedNLL (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : MixedConfiguration tokens) (θ : BindingParameters V C) : ℝ :=
  match observed with
  | .inl z => stateRowNLL mixedHeadRead (0 : Fin 2) θ + bindingStateNLL tokens z θ
  | .inr z => stateRowNLL mixedHeadRead (1 : Fin 2) θ + rawBindingNLL tokens hcap query z θ

/-- The actually computed complete objective equals negative log of the same true mixed model used by inference.
Source: exact positive branch/head factorization, genuine branch likelihood identities and the real log-product law. -/
theorem mixedNLL_eq (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : MixedConfiguration tokens) (θ : BindingParameters V C) :
    mixedNLL tokens hcap query observed θ = -Real.log (mixedProbability θ tokens hcap query observed) := by
  cases observed with
  | inl state =>
      simp only [mixedNLL, mixedProbability]
      rw [stateRowNLL_eq, bindingStateNLL_eq,
        Real.log_mul (mixedHeadWeight_pos θ 0).ne' (sharedStateProbability_pos θ.1 tokens state).ne']
      unfold mixedHeadWeight
      ring
  | inr route =>
      simp only [mixedNLL, mixedProbability]
      rw [stateRowNLL_eq, rawBindingNLL_eq,
        Real.log_mul (mixedHeadWeight_pos θ 1).ne' (rawBindingProbability_pos θ tokens hcap query route).ne']
      unfold mixedHeadWeight
      ring

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- Joint ordinary-gradient mixture training is globally convex in every unrestricted raw embedding, value, position and head coordinate.
Source: the actually computed complete mixed NLL as a sum of genuine affine categorical/Gibbs losses, not output-only CE. -/
theorem mixedNLL_convex (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : MixedConfiguration tokens) : ConvexOn ℝ Set.univ (mixedNLL tokens hcap query observed) := by
  cases observed with
  | inl state => exact (stateRowNLL_convex mixedHeadRead (0 : Fin 2)).add (bindingStateNLL_convex tokens state)
  | inr route => exact (stateRowNLL_convex mixedHeadRead (1 : Fin 2)).add (rawBindingNLL_convex tokens hcap query route)

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- Every actual complete mixed loss is nonnegative at arbitrary simultaneous finite parameters.
Source: its exact probability identity and the same model's proved positivity/normalization bounds. -/
theorem mixedNLL_nonneg (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : MixedConfiguration tokens) (θ : BindingParameters V C) : 0 ≤ mixedNLL tokens hcap query observed θ := by
  rw [mixedNLL_eq]
  exact neg_nonneg.mpr (Real.log_nonpos (mixedProbability_pos θ tokens hcap query observed).le
    (mixedProbability_le_one θ tokens hcap query observed))

example : ([1] : List (Fin 2)).length ≤ 4 := by decide

/-- A genuine finite minibatch may share every raw parameter across different actual prefix lengths and observed branches.
Source: ordinary summed complete mixed likelihoods; observed configurations remain absent from inference arguments. -/
def mixedBatchNLL {B : Type*} [Fintype B] (tokens : B → List (Fin V)) (hcap : ∀ b, (tokens b).length ≤ C)
    (query : (b : B) → Fin (tokens b).length) (observed : (b : B) → MixedConfiguration (tokens b))
    (θ : BindingParameters V C) : ℝ := ∑ b, mixedNLL (tokens b) (hcap b) (query b) (observed b) θ

/-- Actual variable-length mixed minibatch training is globally convex jointly in the entire shared free parameter space.
Source: finite sums of the very same complete mixed NLL, without per-example embedding or path parameters. -/
theorem mixedBatchNLL_convex {B : Type*} [Fintype B] (tokens : B → List (Fin V))
    (hcap : ∀ b, (tokens b).length ≤ C) (query : (b : B) → Fin (tokens b).length)
    (observed : (b : B) → MixedConfiguration (tokens b)) : ConvexOn ℝ Set.univ (mixedBatchNLL tokens hcap query observed) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  simp only [mixedBatchNLL, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_le_sum (fun i _ => (mixedNLL_convex (tokens i) (hcap i) (query i) (observed i)).2 hx hy ha hb hab)

example : ∀ b : Fin 2, (if b = 0 then [(0 : Fin 2)] else [1, 0]).length ≤ 4 := by
  intro b
  fin_cases b <;> decide

/-- Every actual shared minibatch loss is nonnegative at all finite simultaneous raw parameter assignments.
Source: genuine per-prefix complete likelihood nonnegativity and the actual finite summed objective. -/
theorem mixedBatchNLL_nonneg {B : Type*} [Fintype B] (tokens : B → List (Fin V))
    (hcap : ∀ b, (tokens b).length ≤ C) (query : (b : B) → Fin (tokens b).length)
    (observed : (b : B) → MixedConfiguration (tokens b)) (θ : BindingParameters V C) :
    0 ≤ mixedBatchNLL tokens hcap query observed θ :=
  Finset.sum_nonneg (fun b _ => mixedNLL_nonneg (tokens b) (hcap b) (query b) (observed b) θ)

example : ∀ b : Fin 1, ([b.castLT (by omega), (1 : Fin 2)] : List (Fin 2)).length ≤ 4 := by
  intro b
  norm_num

/-- An untrained raw two-token state-branch example is genuinely convex with all matching/value/head coordinates free.
Source: the actual compact mixed loss, without any correct-answer or successful-training premise. -/
example : ConvexOn ℝ Set.univ (mixedNLL [(0 : Fin 2), 1] (by decide : ([(0 : Fin 2), 1] : List (Fin 2)).length ≤ 4)
    (1 : Fin 2) (.inl (0, (![0, 0], fun _ => 0)))) := mixedNLL_convex _ _ _ _

/-- The other actual branch is genuinely convex on exactly the same unrestricted joint parameter space.
Source: the raw all-pair complete likelihood and learned branch CE, without a prepared binding encoder. -/
example : ConvexOn ℝ Set.univ (mixedNLL [(0 : Fin 2), 1] (by decide : ([(0 : Fin 2), 1] : List (Fin 2)).length ≤ 4)
    (1 : Fin 2) (.inr ((0, 1), (fun _ => 0), (fun _ => 0)))) := mixedNLL_convex _ _ _ _

/-- The exact actual mixed loss is nonnegative for arbitrary free weights in either observed branch.
Source: positive normalized true mixed likelihood, without a successfully trained or correct-output hypothesis. -/
example (θ : BindingParameters 2 4) :
    0 ≤ mixedNLL [(0 : Fin 2), 1] (by decide) (1 : Fin 2)
      (.inr ((0, 1), (fun _ => 0), (fun _ => 0))) θ := mixedNLL_nonneg _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
