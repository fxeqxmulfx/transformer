module

public import Transformer.SequenceModel.Attention.Execution.Block

/-!
# Complete executed residual stack

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C6--C7;
diff_gpt `b98c665b`, `gpt.py`. Embedding error, input normalization, all
independently weighted blocks and both residual additions are included.
The operation predicates are backend obligations; the resulting state error
is proved here, not included among those obligations.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability

/-- C6: the actual operation checks imply the derived depth recurrence at
every executed layer and input position, with arbitrary exact reference phases. -/
theorem model_error (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (t : ModelTrace cfg T)
    (δ τ : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (ht : ModelAccurate cfg θ eps xeps a tokens t δ τ) (n : ℕ) (hn : n ≤ cfg.n_layers) (i : Fin T) :
    ‖t.state n i - hidden cfg θ eps xeps a tokens n i‖ ≤ modelBudget cfg θ eps xeps δ τ n := by
  rcases ht with ⟨hemb, hinit, hb, hnorm, hlogit⟩
  induction n generalizing i with
  | zero =>
      exact rms_error cfg.d_model_pos heps (θ.embedding (tokens i)) (t.embedding (tokens i))
        (t.initial i) δ δ (hemb (tokens i)) (hinit i)
  | succ n ih =>
      have hlt : n < cfg.n_layers := by omega
      rw [ModelTrace.state, hidden, modelBudget, dite_eq_left hlt, dite_eq_left hlt, dite_eq_left hlt]
      exact block_movement_error cfg (θ.blocks ⟨n, hlt⟩) heps hxeps (a ⟨n, hlt⟩)
        (t.state n) (hidden cfg θ eps xeps a tokens n) (t.blocks ⟨n, hlt⟩)
        δ τ (modelBudget cfg θ eps xeps δ τ n) hδ hτ (hb ⟨n, hlt⟩) (ih (by omega)) i

/-- A complete trace of the existing exact real source, retaining its tied table. -/
def exactModel (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) : ModelTrace cfg T where
  embedding := θ.embedding
  initial := fun i => rmsNormEps eps (θ.embedding (tokens i))
  blocks := fun l => exactBlock cfg (θ.blocks l) eps xeps (a l) (hidden cfg θ eps xeps a tokens l.val)
  normalized := fun i => rmsNormEps eps (hidden cfg θ eps xeps a tokens cfg.n_layers i)
  logits := fun i v => inner ℝ (rmsNormEps eps (hidden cfg θ eps xeps a tokens cfg.n_layers i)) (θ.embedding v)

/-- Structural source correspondence at every depth, including the stopped tail. -/
theorem exactModel_state (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (n : ℕ) :
    (exactModel cfg θ eps xeps a tokens).state n = hidden cfg θ eps xeps a tokens n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [ModelTrace.state, hidden]
      split_ifs with hn
      · exact exactBlock_output _ _ _ _ _ _
      · exact ih

/-- C6: the full operation-contract hypotheses have exact-source witnesses
for arbitrary weights, input words and phase tables. -/
theorem exactModel_accurate (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) :
    ModelAccurate cfg θ eps xeps a tokens (exactModel cfg θ eps xeps a tokens) 0 0 := by
  refine ⟨by intro v; simp [exactModel], by intro i; simp [exactModel], ?_, ?_, ?_⟩
  · intro l
    rw [exactModel_state]
    exact exactBlock_accurate _ _ _ _ _ _
  · intro i
    rw [exactModel_state]
    simp [exactModel]
  · intro i v
    simp [exactModel]

example (θ : ModelParams witnessConfig) := model_error witnessConfig θ
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (fun _ => true) (fun i : Fin 3 => (i.val : ℝ))) (fun i : Fin 3 => i)
  (exactModel witnessConfig θ (1 / 8388608) (1 / 1000000000000)
    (sourceAngles witnessConfig (fun _ => true) (fun i : Fin 3 => (i.val : ℝ))) (fun i : Fin 3 => i))
  0 0 (by norm_num) (by norm_num) (exactModel_accurate _ _ _ _ _ _) 2 (by decide) 2

end Transformer.SequenceModel.Attention.Execution
