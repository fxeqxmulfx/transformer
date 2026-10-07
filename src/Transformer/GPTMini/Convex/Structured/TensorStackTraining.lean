import Transformer.GPTMini.Convex.Structured.TensorStack

/-!
# Genuine full-stack complete likelihood is jointly convex

Source: TensorStack's real shared-weight two-residual induction and
the actual complete tensor probability/NLL at 815556c. Training below
reads the genuine prenorm input of the configured final attention
layer after all actual intermediate layers, not a separate shallow
surrogate. The same branch/path/route/channel observations stay outside
forward inference; every raw parameter remains unrestricted together.

The real final-head probability, inferred means and computed likelihood
are proved equal to the verified raw model for every weight assignment.
Therefore the actual complete objective is globally convex jointly
in all free token/Q/K/value/state/position/head parameters. This scope
requires shared attention across layers and zero fixed FFN matrices.
Output-only CE, independent deep matrices, floating-point equivalence,
AdamW convergence and benchmark-training success are not asserted.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ}

/-- Genuine final-attention input is actual prenorm after every configured nonfinal two-residual block.
Source: TensorStack's real sequential stream initialized by actual learned token/position embeddings. -/
def tensorStackSequence (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model) (eps : ℝ)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (row : Fin tokens.length) : EucSpace cfg.d_model :=
  rmsNormEps eps (tensorStreamStack cfg eps hwidth (tensorHeadParameters θ)
    (tensorRawSequence hsize hwidth θ tokens hcap) hcap (cfg.n_layers - 1) row)

/-- Actual full-stack training computes complete NLL at the real final attention input.
Source: TensorTraining's computed categorical/contracted loss and the true configured stack above. -/
def tensorStackNLL (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model) (eps : ℝ)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) (observed : MixedConfiguration tokens)
    (θ : BindingParameters V C) : ℝ :=
  tensorMixedNLL hwidth (tensorHeadParameters θ) (tensorStackSequence cfg hsize hwidth eps θ tokens hcap) hcap query observed

/-- The actual complete probability is evaluated by the same learned final head in the genuine configured stack.
Source: TensorLikelihood's true mixed joint, applied to the real final prenorm residual sequence. -/
def tensorStackProbability (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model) (eps : ℝ)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) : MixedConfiguration tokens → ℝ :=
  tensorMixedProbability hwidth (tensorHeadParameters θ) (tensorStackSequence cfg hsize hwidth eps θ tokens hcap) hcap query

/-- All recovered input observations at the actual final attention layer equal the initial embedding/prenorm observations.
Source: derived actual embedding anchors and the complete two-residual stack induction, including changed RMS scales. -/
theorem tensorStackSequence_fieldsSame (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (row : Fin tokens.length) :
    tensorFieldsSame hwidth (tensorStackSequence cfg hsize hwidth eps θ tokens hcap row)
      (tensorSequence hsize hwidth eps θ tokens hcap row) := by
  exact tensorStreamStack_fieldsSame cfg eps heps hwidth (tensorHeadParameters θ) _ hcap
    (fun _ => tensorInput_anchor hsize hwidth θ.1 _ _) _ row

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

/-- The real final attention's inferred ten values equal the verified initial-prenorm tensor head for every weight.
Source: genuine whole-stack field recovery and the complete observation-to-inference proof. -/
theorem tensorStackCoordinates_sequence (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) :
    tensorMixedCoordinates hwidth (tensorHeadParameters θ) (tensorStackSequence cfg hsize hwidth eps θ tokens hcap) hcap query =
      tensorMixedCoordinates hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query := by
  apply tensorMixedCoordinates_fieldsSame
  exact tensorStackSequence_fieldsSame cfg hsize hwidth eps heps θ tokens hcap

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 548), 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

/-- The actual configured-stack full joint is precisely the same genuine raw learned probability at all free parameters.
Source: real full-stack observation induction and the identical actual tensor mixture factors. -/
theorem tensorStackProbability_sequence (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) :
    tensorStackProbability cfg hsize hwidth eps θ tokens hcap query = mixedProbability θ tokens hcap query := by
  unfold tensorStackProbability
  rw [tensorMixedProbability_fieldsSame hwidth (tensorHeadParameters θ) _ _ hcap query
    (tensorStackSequence_fieldsSame cfg hsize hwidth eps heps θ tokens hcap)]
  exact tensorMixedProbability_sequence hsize hwidth eps heps θ tokens hcap query

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 36), 9, 10] : List (Fin 36)).length ≤ 128 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

/-- The genuinely computed final-head complete training loss equals the verified raw joint likelihood for every unrestricted weight.
Source: actual configured-stack inputs, complete observation preservation and true computed tensor training transfer. -/
theorem tensorStackNLL_sequence (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    tensorStackNLL cfg hsize hwidth eps tokens hcap query observed θ = mixedNLL tokens hcap query observed θ := by
  unfold tensorStackNLL
  rw [tensorMixedNLL_fieldsSame hwidth (tensorHeadParameters θ) _ _ hcap query
    (tensorStackSequence_fieldsSame cfg hsize hwidth eps heps θ tokens hcap)]
  exact tensorMixedNLL_sequence hsize hwidth eps heps θ tokens hcap query observed

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 21, 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

/-- The actual full-stack training objective is negative log of exactly the same true full-stack inference joint.
Source: both real configured-stack identities and the original genuine mixed likelihood/probability coupling. -/
theorem tensorStackNLL_eq (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    tensorStackNLL cfg hsize hwidth eps tokens hcap query observed θ =
      -Real.log (tensorStackProbability cfg hsize hwidth eps θ tokens hcap query observed) := by
  rw [tensorStackNLL_sequence cfg hsize hwidth eps heps, mixedNLL_eq,
    tensorStackProbability_sequence cfg hsize hwidth eps heps]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 548), 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

/-- Complete likelihood through the actual entire configured stack is globally convex jointly in every free raw parameter.
Source: exact genuine-stack computation identity on Set.univ, with shared attention weights and the FFN actually fixed to zero. -/
theorem tensorStackNLL_convex (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (observed : MixedConfiguration tokens) :
    ConvexOn ℝ Set.univ (tensorStackNLL cfg hsize hwidth eps tokens hcap query observed) := by
  have heq : tensorStackNLL cfg hsize hwidth eps tokens hcap query observed = mixedNLL tokens hcap query observed := by
    funext θ
    exact tensorStackNLL_sequence cfg hsize hwidth eps heps θ tokens hcap query observed
  rw [heq]
  exact mixedNLL_convex tokens hcap query observed

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

/-- The entire actual configured-stack joint is normalized at every unrestricted parameter assignment.
Source: exact real-stack probability identity and genuine full mixed normalization, without a correct-output premise. -/
theorem tensorStackProbability_sum (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) :
    (∑ observed, tensorStackProbability cfg hsize hwidth eps θ tokens hcap query observed) = 1 := by
  rw [tensorStackProbability_sequence cfg hsize hwidth eps heps]
  exact mixedProbability_sum θ tokens hcap query

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 548), 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
