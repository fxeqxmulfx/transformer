import Transformer.GPTMini.Convex.Structured.TensorCausalBlock
import Transformer.GPTMini.Convex.Structured.StatePathProduct

/-!
# Complete likelihood computed from actual tensor fields

Source: MixedTraining's genuine complete NLL at d436526, the actual
causal tensor block at 84ba59d, and StatePathProduct's physical-index
unrolling. Observed branch/path/route/channel labels are external
training data. Initial, transition and emission likelihoods use small
categorical rows; the pointer uses its actual contracted partition
minus its observed complete energy. All token-dependent logits and
absolute positions are read from actual prenorm tensors alone.

For every free embedding/head parameter, this computed tensor loss is
proved equal to the verified complete mixed likelihood. Its composition
with genuine token embedding, positional addition and RMSNorm is thus
globally convex on the entire unrestricted BindingParameters domain.
This is neither output-only CE nor an AdamW convergence claim. Complete
tensor probability and full-stack training coupling remain separate.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C d T : ℕ}

/-- A complete observed tensor configuration has the same real physical history and all-pair domains.
Source: MixedHeads' disjoint state/pointer configuration, with sequence length rather than raw token IDs as its index. -/
abbrev TensorConfiguration (T : ℕ) :=
  MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) T ⊕ SharedPointerConfiguration (Fin T × Fin T)

/-- Actual categorical likelihood uses the computed finite row logits.
Source: MarkovTraining.stateRowNLL's exact log-normalizer-minus-observed-score formula. -/
def tensorCategoricalNLL {S : Type*} [Fintype S] (score : S → ℝ) (observed : S) : ℝ :=
  Real.log (∑ state, Real.exp (score state)) - score observed

/-- Actual state/path/value loss reads only tensors, free global logits and fixed observed data.
Source: the real initial categorical row, every physical transition row and every conditional output-channel row. -/
def tensorStateNLL (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (observed : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) T) : ℝ :=
  tensorCategoricalNLL (tensorInitial ψ) observed.1 +
    (∑ position, tensorCategoricalNLL
      (tensorTransition hwidth (x position) (statePathPrevious observed.1 observed.2.1 position)) (observed.2.1 position)) +
    ∑ h, tensorCategoricalNLL (tensorEmission ψ (statePathEnd observed.1 T observed.2.1) h) (observed.2.2 h)

/-- Actual all-pair pointer likelihood uses its genuine compact tensor-conditioned normalizer and observed energy.
Source: Binding.bindingPointerNLL's exact computation, without raw vocabulary-table lookups inside tensor attention. -/
def tensorPointerNLL (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (observed : SharedPointerConfiguration (Fin T × Fin T)) : ℝ :=
  Real.log (pointerPartition (tensorQuery hwidth (x query)) (fun pair => tensorKey hwidth (x pair.1))
    (fun pair => tensorValue hwidth (x pair.2)) (tensorBindingBias hwidth ψ x hcap)) -
  pointerEnergy (tensorQuery hwidth (x query)) (fun pair => tensorKey hwidth (x pair.1))
    (fun pair => tensorValue hwidth (x pair.2)) (tensorBindingBias hwidth ψ x hcap) observed

/-- Complete actual tensor training evaluates the observed branch categorical loss plus its true branch likelihood.
Source: MixedTraining's genuine complete factorization; observation is absent from tensor inference. -/
def tensorMixedNLL (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) : TensorConfiguration T → ℝ
  | .inl state => tensorCategoricalNLL (fun head => ψ.1 (.inr (.inr (.inr head)))) (0 : Fin 2) +
      tensorStateNLL hwidth ψ x state
  | .inr route => tensorCategoricalNLL (fun head => ψ.1 (.inr (.inr (.inr head)))) (1 : Fin 2) +
      tensorPointerNLL hwidth ψ x hcap query route

/-- Tensor categorical rows evaluate the original actual linear-read likelihood at the identical logits.
Source: the unchanged row normalizer and observed-score arithmetic, a coordinate-level computation identity. -/
theorem tensorCategoricalNLL_read {E S : Type*} [AddCommGroup E] [Module ℝ E] [Fintype S]
    (linear : S → E →ₗ[ℝ] ℝ) (observed : S) (θ : E) :
    tensorCategoricalNLL (fun state => linear state θ) observed = stateRowNLL linear observed θ := by
  rfl

/-- The true tensor complete state/value objective is exactly the verified raw shared-state likelihood.
Source: actual RMS transition/global reads and the complete chronological physical-index NLL identity. -/
theorem tensorStateNLL_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (observed : SharedStateConfiguration tokens) :
    tensorStateNLL hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) observed =
      bindingStateNLL tokens observed θ := by
  unfold tensorStateNLL bindingStateNLL markovNLL
  rw [conditionalStateNLL_sum]
  simp_rw [tensorTransition_sequence hsize hwidth eps heps]
  rw [(tensorHeadParameters_reads θ).1, (tensorHeadParameters_reads θ).2.1]
  simp only [tensorCategoricalNLL_read, channelNLL]

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- The actual tensor all-pair objective equals the same genuine affine raw pointer likelihood.
Source: all true prenorm Q/K/value/position reads and the identical compact energy/partition computation. -/
theorem tensorPointerNLL_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (observed : RawBindingConfiguration tokens) :
    tensorPointerNLL hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed =
      rawBindingNLL tokens hcap query observed θ := by
  simp only [tensorPointerNLL,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).1,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).2.1,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).2.2,
    tensorBindingBias_sequence hsize hwidth eps heps, rawBindingNLL, bindingPointerNLL]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Actual free tensor mixture logits yield precisely the original branch likelihood read.
Source: the genuine coordinate split and unchanged two-way categorical arithmetic. -/
theorem tensorHeadNLL_read (θ : BindingParameters V C) (head : Fin 2) :
    tensorCategoricalNLL (fun branch => (tensorHeadParameters θ).1 (.inr (.inr (.inr branch)))) head =
      stateRowNLL mixedHeadRead head θ := by
  rfl

/-- Complete tensor training and the verified raw model compute identical losses for every unrestricted parameter vector.
Source: both actual branch-likelihood transfer identities and the genuinely learned head categorical row. -/
theorem tensorMixedNLL_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    tensorMixedNLL hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed =
      mixedNLL tokens hcap query observed θ := by
  cases observed with
  | inl state =>
      simp only [tensorMixedNLL, mixedNLL, tensorHeadNLL_read,
        tensorStateNLL_sequence hsize hwidth eps heps]
  | inr route =>
      simp only [tensorMixedNLL, mixedNLL, tensorHeadNLL_read,
        tensorPointerNLL_sequence hsize hwidth eps heps]

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 9, 10] : List (Fin 36)).length ≤ 128 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- The actual complete loss through genuine embedding/positions/RMSNorm is globally convex in all jointly free weights.
Source: exact computed tensor likelihood transfer on the entire unrestricted BindingParameters domain. -/
theorem tensorMixedNLL_convex (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    ConvexOn ℝ Set.univ (fun θ : BindingParameters V C => tensorMixedNLL hwidth (tensorHeadParameters θ)
      (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed) := by
  have heq : (fun θ : BindingParameters V C => tensorMixedNLL hwidth (tensorHeadParameters θ)
      (tensorSequence hsize hwidth eps θ tokens hcap) hcap query observed) = mixedNLL tokens hcap query observed := by
    funext θ
    exact tensorMixedNLL_sequence hsize hwidth eps heps θ tokens hcap query observed
  rw [heq]
  exact mixedNLL_convex tokens hcap query observed

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
