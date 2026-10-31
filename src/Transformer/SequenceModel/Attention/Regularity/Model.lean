module

public import Transformer.SequenceModel.Attention.Regularity.Block

/-!
# Local analyticity of the full supplied Transformer

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, R1.
The sole extra activation condition is explicit avoidance of the XSA
clipping spheres. It is checked on actual hidden states at the base weights.
Induction proves regularity of the entire varying finite stack; no network
regularity is inserted as a hypothesis or hidden in a parameterization.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini Landscape.Neural

/-- Actual own values before XSA at every layer, token and head avoid the
clipping sphere. This is a predicate of weights and the supplied input. -/
def Regular (cfg : Config) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (w : Parameters cfg) : Prop :=
  ∀ (d : Fin cfg.n_layers) (j : Fin T) (h : Fin cfg.n_heads),
    ‖layerValue cfg d w
      (rmsNormEps eps (hidden cfg (decode cfg w) eps xeps angle tokens d.val j)) h‖ ≠ xeps

/-- R1: induction through the actual finite stack, with all parameters varying. -/
theorem analyticAt_hidden (cfg : Config) {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (w₀ : Parameters cfg)
    (hreg : Regular cfg eps xeps angle tokens w₀) (n : ℕ) (j : Fin T) (i : Fin cfg.d_model) :
    AnalyticAt ℝ (fun w => hidden cfg (decode cfg w) eps xeps angle tokens n j i) w₀ := by
  induction n generalizing j i with
  | zero =>
      exact analyticAt_rmsNormEps cfg.d_model_pos heps
        (fun c => analyticAt_parameter (I := Index cfg) (.inl (tokens j, c)) w₀) i
  | succ n ih =>
      simp only [hidden]
      split_ifs with hn
      · exact analyticAt_block cfg (d := ⟨n, hn⟩) heps hxeps (angle ⟨n, hn⟩)
          ih (hreg ⟨n, hn⟩) j i
      · exact ih j i

/-- R1: the final RMSNorm and the tied embedding readout both vary. -/
theorem analyticAt_score (cfg : Config) {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (w₀ : Parameters cfg)
    (hreg : Regular cfg eps xeps angle tokens w₀) (j : Fin T) (v : Fin cfg.vocab_size) :
    AnalyticAt ℝ (fun w => score cfg eps xeps angle j w tokens v) w₀ := by
  exact analyticAt_inner
    (analyticAt_rmsNormEps cfg.d_model_pos heps
      (analyticAt_hidden cfg heps hxeps angle tokens w₀ hreg cfg.n_layers j))
    (fun i => analyticAt_parameter (I := Index cfg) (.inl (v, i)) w₀)

/-- R1: the actual full-softmax loss on independently indexed examples is
analytic at regular weights. Repeated contexts retain their multiplicity. -/
theorem analyticAt_loss (cfg : Config) {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} {A : Type*}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (rows : Finset A) (target : A → Fin cfg.vocab_size) (w₀ : Parameters cfg)
    (hreg : ∀ a ∈ rows, Regular cfg eps xeps angle (inputs a) w₀) :
    AnalyticAt ℝ (SequenceModel.loss
      (fun w a v => score cfg eps xeps angle (readout a) w (inputs a) v) rows target) w₀ := by
  exact analyticAt_meanCrossEntropy rows
    (fun a ha v => analyticAt_score cfg heps hxeps angle (inputs a) w₀ (hreg a ha) (readout a) v)
    target

/-- R1: zero weights are strictly inside the clipped region for positive XSA
epsilon, at arbitrary finite dimensions, phases, depths and token arrays. -/
theorem regular_zero (cfg : Config) (eps : ℝ) {xeps : ℝ} (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) : Regular cfg eps xeps angle tokens 0 := by
  intro d j h
  rw [layerValue_zero, norm_zero]
  exact hxeps.ne

private def phases : Fin 2 → Fin 3 → Fin 2 → ℝ :=
  sourceAngles witnessConfig (fun d => d = 0) (fun j => j.val)

example := analyticAt_hidden witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) phases (fun j => j) 0
  (regular_zero _ _ (by norm_num) _ _) 2 2 0
example := analyticAt_score witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) phases (fun j => j) 0
  (regular_zero _ _ (by norm_num) _ _) 2 1
example := analyticAt_loss witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) phases
  (fun a j : Fin 3 => if a = j then 0 else 1) (fun _ => (2 : Fin 3))
  {0, 1} (fun a => a) 0 (fun _ _ => regular_zero _ _ (by norm_num) _ _)

end Transformer.SequenceModel.Attention
