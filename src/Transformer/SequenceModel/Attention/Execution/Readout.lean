module

public import Transformer.SequenceModel.Attention.Execution.Model

/-!
# Execution certificates through the tied readout

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C7.
The final normalizer, shared embedding table and dot-product execution all
contribute. Strict margins certify the recorded logits themselves.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability

/-- The derived error for one class, including its raw embedding norm. -/
def logitRadius (cfg : Config) (eps : ℝ) (e : EucSpace cfg.d_model) (E δ : ℝ) : ℝ :=
  (rmsLip eps * E + δ) * ‖e‖ + (Real.sqrt (cfg.d_model : ℝ) + δ) * δ + δ

/-- Strict prediction by the actual recorded logits, with no tie-breaking assumption. -/
def ModelTrace.predict {cfg : Config} {T : ℕ} (t : ModelTrace cfg T)
    (i : Fin T) (target : Fin cfg.vocab_size) : Bool :=
  decide (∀ v, v ≠ target → t.logits i v < t.logits i target)

/-- C7: full execution error at a tied logit, including final RMSNorm,
embedding quantization and the executed dot product. -/
theorem logit_error (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (t : ModelTrace cfg T)
    (δ τ : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (ht : ModelAccurate cfg θ eps xeps a tokens t δ τ) (i : Fin T) (v : Fin cfg.vocab_size) :
    |t.logits i v - inner ℝ
      (rmsNormEps eps (hidden cfg θ eps xeps a tokens cfg.n_layers i)) (θ.embedding v)| ≤
        logitRadius cfg eps (θ.embedding v) (modelBudget cfg θ eps xeps δ τ cfg.n_layers) δ := by
  have hE := model_error cfg θ heps hxeps a tokens t δ τ hδ hτ ht cfg.n_layers le_rfl i
  have hz := rms_error cfg.d_model_pos heps _ _ (t.normalized i) _ δ hE (ht.2.2.2.1 i)
  have hn := norm_of_error (t.normalized i) (rmsNormEps eps (t.state cfg.n_layers i)) δ
    (Real.sqrt (cfg.d_model : ℝ)) (ht.2.2.2.1 i)
    (rmsNormEps_norm_le eps heps cfg.d_model_pos _)
  let z := rmsNormEps eps (hidden cfg θ eps xeps a tokens cfg.n_layers i)
  have heq : inner ℝ (t.normalized i) (t.embedding v) - inner ℝ z (θ.embedding v) =
      inner ℝ (t.normalized i - z) (θ.embedding v) +
        inner ℝ (t.normalized i) (t.embedding v - θ.embedding v) := by
    rw [inner_sub_left, inner_sub_right]
    ring
  have hinner : |inner ℝ (t.normalized i) (t.embedding v) - inner ℝ z (θ.embedding v)| ≤
      (rmsLip eps * modelBudget cfg θ eps xeps δ τ cfg.n_layers + δ) * ‖θ.embedding v‖ +
        (Real.sqrt (cfg.d_model : ℝ) + δ) * δ := by
    rw [heq]
    refine (abs_add_le _ _).trans ((add_le_add (abs_real_inner_le_norm _ _)
      (abs_real_inner_le_norm _ _)).trans ?_)
    exact add_le_add (mul_le_mul_of_nonneg_right hz (norm_nonneg _))
      (mul_le_mul hn (ht.1 v) (norm_nonneg _) (by positivity))
  have hd := abs_add_le (t.logits i v - inner ℝ (t.normalized i) (t.embedding v))
    (inner ℝ (t.normalized i) (t.embedding v) - inner ℝ z (θ.embedding v))
  rw [sub_add_sub_cancel] at hd
  exact hd.trans ((add_le_add (ht.2.2.2.2 i v) hinner).trans_eq (by unfold logitRadius; ring))

/-- A source margin exceeding both computed class-specific error budgets. -/
def MarginAbove (cfg : Config) (θ : ModelParams cfg) (eps : ℝ)
    (x : EucSpace cfg.d_model) (E δ : ℝ) (target : Fin cfg.vocab_size) : Prop :=
  ∀ v, v ≠ target → logitRadius cfg eps (θ.embedding target) E δ +
    logitRadius cfg eps (θ.embedding v) E δ <
      inner ℝ (rmsNormEps eps x) (θ.embedding target) -
        inner ℝ (rmsNormEps eps x) (θ.embedding v)

/-- C7: primitive operation checks and the derived margin certify the
executed prediction; no block-error or hidden-error premise is assumed. -/
theorem predict_true (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (t : ModelTrace cfg T)
    (δ τ : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (ht : ModelAccurate cfg θ eps xeps a tokens t δ τ) (i : Fin T) (target : Fin cfg.vocab_size)
    (hm : MarginAbove cfg θ eps (hidden cfg θ eps xeps a tokens cfg.n_layers i)
      (modelBudget cfg θ eps xeps δ τ cfg.n_layers) δ target) :
    t.predict i target = true := by
  apply decide_eq_true
  intro v hv
  have h1 := (abs_le.mp (logit_error cfg θ heps hxeps a tokens t δ τ hδ hτ ht i target)).1
  have h2 := (abs_le.mp (logit_error cfg θ heps hxeps a tokens t δ τ hδ hτ ht i v)).2
  have hg := hm v hv
  linarith

example (θ : ModelParams witnessConfig) := logit_error witnessConfig θ
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (fun _ => true) (fun i : Fin 3 => (i.val : ℝ))) (fun i : Fin 3 => i)
  (exactModel witnessConfig θ (1 / 8388608) (1 / 1000000000000)
    (sourceAngles witnessConfig (fun _ => true) (fun i : Fin 3 => (i.val : ℝ))) (fun i : Fin 3 => i))
  0 0 (by norm_num) (by norm_num) (exactModel_accurate _ _ _ _ _ _) 2 2

end Transformer.SequenceModel.Attention.Execution
