module

public import Transformer.SequenceModel.Attention.Stability.Witness
public import Transformer.SequenceModel.Attention.Stability.Classification

/-!
# Classification transfer to the actual nonperiodic counting model

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B8--B9.
The sufficient margin is computed from B7, rather than postulated as a
hidden-state error. The conclusion concerns the existing rounded execution
and its single uniform counting formula. A nonconstant three-class witness
satisfies the conditions on every finite word. It is not an IEEE certificate.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini

/-- A strict source readout margin large enough to absorb both logit errors. -/
def MarginAbove (cfg : Config) (embedding : Fin cfg.vocab_size → EucSpace cfg.d_model)
    (eps : ℝ) (x : EucSpace cfg.d_model) (E : ℝ) (target : Fin cfg.vocab_size) : Prop :=
  ∀ v, v ≠ target → (rmsLip eps * E) * (‖embedding target‖ + ‖embedding v‖) <
    inner ℝ (rmsNormEps eps x) (embedding target) - inner ℝ (rmsNormEps eps x) (embedding v)

/-- B8: an explicit sufficient margin transfers the real source decision to
the actual rounded source model with the same exact RoPE, weights and depth. -/
theorem rounded_predict_true (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) {T : ℕ} (tokens : Fin T → Fin cfg.vocab_size)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ)))
    (hr : TraceInRange cfg θ eps xeps p s enabled tokens) (i : Fin T)
    (target : Fin cfg.vocab_size)
    (hm : MarginAbove cfg θ.embedding eps
      (hidden cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ)))
        tokens cfg.n_layers i) (errorBudget cfg θ eps xeps s cfg.n_layers) target) :
    Rotary.predict cfg θ eps xeps p s enabled (List.ofFn tokens) (i.val + 1) target = true := by
  have hE := stack_error cfg θ heps hxeps p s enabled tokens hp hr cfg.n_layers le_rfl i
  change decide (∀ v, v ≠ target →
    inner ℝ (rmsNormEps eps (Normalized.vector
      (counted cfg θ eps xeps p s enabled tokens cfg.n_layers (i.val + 1)).2)) (θ.embedding v) <
    inner ℝ (rmsNormEps eps (Normalized.vector
      (counted cfg θ eps xeps p s enabled tokens cfg.n_layers (i.val + 1)).2)) (θ.embedding target)) = true
  apply decide_eq_true
  intro v hv
  apply classification_of_hidden_error cfg.d_model_pos heps
    (hidden cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ)))
      tokens cfg.n_layers i)
    (Normalized.vector (counted cfg θ eps xeps p s enabled tokens cfg.n_layers (i.val + 1)).2)
    (θ.embedding target) (θ.embedding v)
    (errorBudget cfg θ eps xeps s cfg.n_layers)
  · rw [norm_sub_rev]
    exact hE
  · exact hm v hv

/-- B8: one fixed formula of depth at most D recognizes every source
decision satisfying this certificate, for any finite input length. The
formula and its exact positional kernels are chosen before the input. -/
theorem uniform_source_certificate (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) (target : Fin cfg.vocab_size)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ))) :
    ∃ φ : Rotary.Form (Fin cfg.vocab_size)
        (Rotary.Key (Rotary.State cfg p s) (Normalized.Coordinate cfg)),
      φ.depth ≤ cfg.n_layers ∧ ∀ (T : ℕ) (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T),
        TraceInRange cfg θ eps xeps p s enabled tokens →
        MarginAbove cfg θ.embedding eps
          (hidden cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ)))
            tokens cfg.n_layers i) (errorBudget cfg θ eps xeps s cfg.n_layers) target →
        φ.sat (Rotary.kernel (Rotary.layers cfg θ eps xeps p s enabled))
          (List.ofFn tokens) (i.val + 1) = true := by
  obtain ⟨φ, hd, hφ⟩ := Rotary.uniform_source_classifier cfg θ eps xeps p s enabled target
  refine ⟨φ, hd, ?_⟩
  intro T tokens i hr hm
  rw [hφ (List.ofFn tokens) (i.val + 1) (by omega)
    (by simp only [List.length_ofFn]; have := i.isLt; omega)]
  exact rounded_predict_true cfg θ heps hxeps p s enabled tokens hp hr i target hm

open CRASP GPTMini

/-- B9: identity residual blocks preserve each normalized token in the real source. -/
theorem example_hidden {T : ℕ} (tokens : Fin T → Fin witnessConfig.vocab_size) (n : ℕ) (i : Fin T) :
    hidden witnessConfig exampleModel (3 / 4) 1
      (sourceAngles witnessConfig (fun _ => true) (fun j : Fin T => (j.val : ℝ))) tokens n i =
      exampleEmbedding (tokens i) := by
  induction n with
  | zero => exact example_embedding_rms _
  | succ n ih =>
      rw [hidden]
      split
      · rw [← Normalized.local_sourceAverage]
        change Normalized.localUpdate witnessConfig exampleBlock _ _ _ _ _ = _
        rw [example_local]
        exact ih
      · exact ih

/-- B9: the computed bound is nonzero even though this particular run rounds exactly. -/
theorem example_budget : errorBudget witnessConfig exampleModel (3 / 4) 1 5 2 = 3 / 16 := by
  norm_num [errorBudget, exampleModel, exampleBlock, blockLip, ffnGain, ffnLip,
    attentionGain, directionGain, componentLip, valueBound, stepRadius, witnessConfig]

/-- B9: different tokens have orthogonal tied embeddings. -/
theorem example_embedding_inner (v w : Fin witnessConfig.vocab_size) :
    inner ℝ (exampleEmbedding v) (exampleEmbedding w) = if v = w then 1 else 0 := by
  rw [EuclideanSpace.inner_eq_star_dotProduct]
  fin_cases v <;> fin_cases w <;>
    norm_num [exampleEmbedding, EuclideanSpace.inner_eq_star_dotProduct, dotProduct,
      Fin.sum_univ_succ, witnessConfig]

/-- B9: each source margin is one, exceeding the complete two-layer budget. -/
theorem example_margin {T : ℕ} (tokens : Fin T → Fin witnessConfig.vocab_size) (i : Fin T) :
    MarginAbove witnessConfig exampleModel.embedding (3 / 4)
      (hidden witnessConfig exampleModel (3 / 4) 1
        (sourceAngles witnessConfig (fun _ => true) (fun j : Fin T => (j.val : ℝ)))
        tokens witnessConfig.n_layers i)
      (errorBudget witnessConfig exampleModel (3 / 4) 1 5 witnessConfig.n_layers) (tokens i) := by
  intro v hv
  change (rmsLip (3 / 4) * errorBudget witnessConfig exampleModel (3 / 4) 1 5 2) *
    (‖exampleEmbedding (tokens i)‖ + ‖exampleEmbedding v‖) < _
  rw [example_budget, example_embedding_norm, example_embedding_norm, example_hidden,
    example_embedding_rms]
  change rmsLip (3 / 4) * (3 / 16) * (1 + 1) <
    inner ℝ (exampleEmbedding (tokens i)) (exampleEmbedding (tokens i)) -
      inner ℝ (exampleEmbedding (tokens i)) (exampleEmbedding v)
  rw [example_embedding_inner, example_embedding_inner,
    ite_eq_left rfl, ite_eq_right (Ne.symm hv)]
  have hs : 0 < Real.sqrt (3 / 4 : ℝ) := by positivity
  have hsq := Real.sq_sqrt (show (0 : ℝ) ≤ 3 / 4 by norm_num)
  have hroot : (3 : ℝ) / 4 < Real.sqrt (3 / 4 : ℝ) := by nlinarith
  unfold rmsLip
  field_simp
  nlinarith

/-- B9: the certificate proves the actual rounded model classifies every
token correctly at every position, on arbitrary finite three-letter words. -/
theorem example_prediction {T : ℕ} (tokens : Fin T → Fin witnessConfig.vocab_size) (i : Fin T) :
    Rotary.predict witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true)
      (List.ofFn tokens) (i.val + 1) (tokens i) = true :=
  rounded_predict_true witnessConfig exampleModel (by norm_num) (by norm_num)
    8 5 (fun _ => true) tokens example_precision (example_trace tokens) i (tokens i)
    (example_margin tokens i)

example := aggregateRadius_nonneg witnessConfig exampleBlock 5 example_precision
example := aggregate_vector_error witnessConfig exampleBlock (eps := 3 / 4) (by norm_num) 1 8 5
  (layerAngles witnessConfig (fun _ => true) 0)
  (counted witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true) (fun i : Fin 3 => i) 0)
  (2 : Fin 3) 0 example_precision (example_aggregate_range _ _ _ _)

example := source_step_error witnessConfig exampleBlock (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num) 8 5 (layerAngles witnessConfig (fun _ => true) 0)
  (counted witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true) (fun i : Fin 3 => i) 0)
  (fun j : Fin 3 => Normalized.vector (counted witnessConfig exampleModel (3 / 4) 1 8 5
    (fun _ => true) (fun i : Fin 3 => i) 0 (j.val + 1)).2) 0 (by intro j; simp) (2 : Fin 3)
  (counted_flag _ _ _ _ _ _ _ _ _ _) example_precision ((example_trace (fun i : Fin 3 => i)).2 0 2)

example := stack_error witnessConfig exampleModel (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num) 8 5 (fun _ => true) (fun i : Fin 3 => i)
  example_precision (example_trace _) 2 (by decide) 2
example := rounded_predict_true witnessConfig exampleModel (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num) 8 5 (fun _ => true) (fun i : Fin 3 => i)
  example_precision (example_trace _) 2 2 (example_margin _ _)
example := uniform_source_certificate witnessConfig exampleModel (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num) 8 5 (fun _ => true) 2 example_precision

end Transformer.SequenceModel.Attention.Stability
