module

public import Transformer.SequenceModel.Attention.Execution.Readout
public import Transformer.SequenceModel.Attention.Stability.Certificate

/-!
# Nonzero operation-error witnesses

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C8.
The final dot products receive a nonzero common offset. This is a synthetic
trace satisfying the operation contracts, not a claim that a CUDA kernel
generates this offset. The classifier is nonconstant on three-letter words.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability

/-- Exact source operations with a controlled error in each final dot product. -/
def shiftedModel (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (δ : ℝ) : ModelTrace cfg T :=
  let t := exactModel cfg θ eps xeps a tokens
  { t with logits := fun i v => t.logits i v + δ }

/-- A readout-only perturbation leaves every recorded residual state unchanged. -/
theorem shiftedModel_state (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (δ : ℝ) (n : ℕ) :
    (shiftedModel cfg θ eps xeps a tokens δ).state n = hidden cfg θ eps xeps a tokens n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [ModelTrace.state, hidden]
      split_ifs with hn
      · exact exactBlock_output _ _ _ _ _ _
      · exact ih

/-- C8: simultaneous local operation hypotheses have witnesses with a
nonzero readout error, at arbitrary source weights, words and phases. -/
theorem shiftedModel_accurate (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (δ τ : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ) :
    ModelAccurate cfg θ eps xeps a tokens (shiftedModel cfg θ eps xeps a tokens δ) δ τ := by
  refine ⟨by intro v; simpa [shiftedModel, exactModel] using hδ,
    by intro i; simpa [shiftedModel, exactModel] using hδ, ?_, ?_, ?_⟩
  · intro l
    rw [shiftedModel_state]
    change BlockAccurate cfg (θ.blocks l) eps xeps (a l)
      (hidden cfg θ eps xeps a tokens l.val)
      (exactBlock cfg (θ.blocks l) eps xeps (a l) (hidden cfg θ eps xeps a tokens l.val)) δ τ
    simp [BlockAccurate, exactBlock, AttentionAccurate, exactAttention,
      AttentionTrace.component, tableRotate_exact, FFNAccurate, exactFFN, hδ, hτ]
  · intro i
    rw [shiftedModel_state]
    change ‖rmsNormEps eps (hidden cfg θ eps xeps a tokens cfg.n_layers i) -
      rmsNormEps eps (hidden cfg θ eps xeps a tokens cfg.n_layers i)‖ ≤ δ
    simpa using hδ
  · intro i v
    simp [shiftedModel, exactModel, abs_of_nonneg hδ]

/-- C8: the recorded logit differs from the exact source by precisely the
specified offset; the witness does not merely have a positive upper bound. -/
theorem shiftedModel_logit_error (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (δ : ℝ) (i : Fin T) (v : Fin cfg.vocab_size) :
    (shiftedModel cfg θ eps xeps a tokens δ).logits i v -
      inner ℝ (rmsNormEps eps (hidden cfg θ eps xeps a tokens cfg.n_layers i)) (θ.embedding v) = δ := by
  change (inner ℝ _ _ + δ) - inner ℝ _ _ = δ
  ring

/-- C8: the complete two-block error recurrence for the synthetic three-class
source model, with a positive operation budget and exact phase tables. -/
theorem example_modelBudget : modelBudget witnessConfig exampleModel (3 / 4) 1 (1 / 1024) 0 2 =
    (5 * rmsLip (3 / 4) + 9) / 1024 := by
  norm_num [modelBudget, exampleModel, exampleBlock, blockLip, ffnGain, ffnLip,
    attentionGain, directionGain, componentLip, valueBound, blockRadius,
    attentionBudget, ffnBudget, witnessConfig]
  ring

/-- C8: the derived error of each tied class logit is strictly below half
the unit source margin, even with all local allowances set to 1/1024. -/
theorem example_logitRadius (v : Fin witnessConfig.vocab_size) :
    logitRadius witnessConfig (3 / 4) (exampleEmbedding v)
      (modelBudget witnessConfig exampleModel (3 / 4) 1 (1 / 1024) 0 2) (1 / 1024) < 1 / 2 := by
  rw [example_modelBudget]
  have hs : 0 < Real.sqrt (3 / 4 : ℝ) := by positivity
  have hsq := Real.sq_sqrt (show (0 : ℝ) ≤ 3 / 4 by norm_num)
  have hL : rmsLip (3 / 4) ≤ 3 := by
    unfold rmsLip
    apply (div_le_iff₀ hs).mpr
    nlinarith
  have hL0 : 0 ≤ rmsLip (3 / 4) := by unfold rmsLip; positivity
  have hL2 : rmsLip (3 / 4) ^ 2 ≤ 9 := by nlinarith
  norm_num [logitRadius, example_embedding_norm, witnessConfig]
  nlinarith

/-- C8: the sufficient execution margin holds on every finite word, with
all three token classes and actual source RoPE enabled. -/
theorem example_execution_margin {T : ℕ} (tokens : Fin T → Fin witnessConfig.vocab_size) (i : Fin T) :
    MarginAbove witnessConfig exampleModel (3 / 4)
      (hidden witnessConfig exampleModel (3 / 4) 1
        (sourceAngles witnessConfig (fun _ => true) (fun j : Fin T => (j.val : ℝ)))
        tokens witnessConfig.n_layers i)
      (modelBudget witnessConfig exampleModel (3 / 4) 1 (1 / 1024) 0 witnessConfig.n_layers)
      (1 / 1024) (tokens i) := by
  intro v hv
  rw [example_hidden, example_embedding_rms]
  change logitRadius witnessConfig (3 / 4) (exampleEmbedding (tokens i)) _ _ +
    logitRadius witnessConfig (3 / 4) (exampleEmbedding v) _ _ <
      inner ℝ (exampleEmbedding (tokens i)) (exampleEmbedding (tokens i)) -
        inner ℝ (exampleEmbedding (tokens i)) (exampleEmbedding v)
  rw [example_embedding_inner, example_embedding_inner,
    ite_eq_left rfl, ite_eq_right (Ne.symm hv)]
  have h1 := example_logitRadius (tokens i)
  have h2 := example_logitRadius v
  change _ + _ < (1 : ℝ) - 0
  linarith

/-- C8: the full error theorem certifies a nonconstant classifier despite
an actual nonzero error of 1/1024 at every recorded logit. -/
theorem example_executed_prediction {T : ℕ} (tokens : Fin T → Fin witnessConfig.vocab_size) (i : Fin T) :
    (shiftedModel witnessConfig exampleModel (3 / 4) 1
      (sourceAngles witnessConfig (fun _ => true) (fun j : Fin T => (j.val : ℝ)))
      tokens (1 / 1024)).predict i (tokens i) = true :=
  predict_true witnessConfig exampleModel (by norm_num) (by norm_num) _ _ _
    (1 / 1024) 0 (by norm_num) (by norm_num)
    (shiftedModel_accurate _ _ _ _ _ _ _ _ (by norm_num) (by norm_num))
    i (tokens i) (example_execution_margin tokens i)

example := shiftedModel_accurate witnessConfig exampleModel (3 / 4) 1
  (sourceAngles witnessConfig (fun _ => true) (fun i : Fin 3 => (i.val : ℝ)))
  (fun i : Fin 3 => i) (1 / 1024) 0 (by norm_num) (by norm_num)
example := predict_true witnessConfig exampleModel (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (fun _ => true) (fun i : Fin 3 => (i.val : ℝ))) (fun i : Fin 3 => i)
  (shiftedModel witnessConfig exampleModel (3 / 4) 1
    (sourceAngles witnessConfig (fun _ => true) (fun i : Fin 3 => (i.val : ℝ))) (fun i : Fin 3 => i) (1 / 1024))
  (1 / 1024) 0 (by norm_num) (by norm_num)
  (shiftedModel_accurate _ _ _ _ _ _ _ _ (by norm_num) (by norm_num)) 2 2 (example_execution_margin _ _)

end Transformer.SequenceModel.Attention.Execution
