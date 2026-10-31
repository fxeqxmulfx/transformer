module

public import Transformer.SequenceModel.Attention.Concentration

/-!
# Full normalized attention in the architecture-independent learning theory

Source: `review/flat_spectrum.md`, *General positional models*, G4--G5.
Examples carry independent IDs, token arrays and readout positions, so repeated
token arrays and losses at every position can be represented without losing
multiplicity. All actual normalized layers and scalar parameters remain.

The source matrix identity holds for any update oracle. The specialization
below uses the classical loss gradient, justified at differentiable queries
by `SequenceModel.oracle_hasGradientAt`. XSA has a clipped normalization;
no global analyticity or equivalence to a chosen autodiff rule at its boundary
is claimed. Neither the response identity nor normalization proves learning.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini ScheduleFree Rotate90 Forcing
open scoped BigOperators

/-- Predict on independently indexed training examples and target positions. -/
def predictor (cfg : Config) (eps xeps : ℝ) {T : ℕ} {A : Type*}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (w : Parameters cfg) (a : A) (v : Fin cfg.vocab_size) : ℝ :=
  score cfg eps xeps angle (readout a) w (inputs a) v

/-- G4: the complete normalized attention model has the exact source history
formula, at arbitrary finite width, depth, context length and sample schedule. -/
theorem attention_history (cfg : Config) (eps xeps : ℝ) {T : ℕ} {A : Type*}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (samples : ℕ → Finset A) (target : A → Fin cfg.vocab_size)
    (p : AdamW.Parameters) (offset : ℕ) (s : AdamW.TrainingState (Index cfg)) (n : ℕ) :
    buffers (trajectory (predictor cfg eps xeps angle inputs readout) samples target p offset s n) =
      propagate (transitions p offset s.history) 0 n (buffers s) +
        ∑ j ∈ Finset.range n, propagate (transitions p offset s.history)
          (j + 1) (n - 1 - j) (forcing p offset
            (sampledOracle (predictor cfg eps xeps angle inputs readout) samples target) s j) := by
  exact history_sum _ samples target p offset s n

/-- G4: even a genuine attention model under zero gradients has only the
initial two-vector parameter family; its nonlinear attention weights can change. -/
theorem attention_zero_plane (cfg : Config) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ) (readout : Fin T)
    (p : AdamW.Parameters) (offset : ℕ) (s : AdamW.TrainingState (Index cfg)) (n : ℕ) :
    ∃ a b : ℝ, ∀ tokens v,
      score cfg eps xeps angle readout (Learning.evaluated p (run p offset (fun _ _ => 0) s n)) tokens v =
        score cfg eps xeps angle readout (WithLp.toLp 2 (a • s.point + b • s.base)) tokens v := by
  exact SequenceModel.zero_scores_plane _ p offset s n

/-- G4: the actual source response changes attention and SwiGLU inside the
normalized nonlinear forward, rather than adding an assumed linear logit term. -/
theorem attention_score_response (cfg : Config) (eps xeps : ℝ) {T : ℕ} {A : Type*}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (samples : ℕ → Finset A) (target : A → Fin cfg.vocab_size)
    (p : AdamW.Parameters) (offset : ℕ) (s : AdamW.TrainingState (Index cfg))
    (n : ℕ) (a : A) (v : Fin cfg.vocab_size) :
    predictor cfg eps xeps angle inputs readout
      (Learning.evaluated p (trajectory (predictor cfg eps xeps angle inputs readout)
        samples target p offset s n)) a v =
      predictor cfg eps xeps angle inputs readout
        (Learning.evaluated p (run p offset (fun _ _ => 0) s n) +
          WithLp.toLp 2 (readEvaluation p (response (transitions p offset s.history)
            (forcing p offset (sampledOracle (predictor cfg eps xeps angle inputs readout)
              samples target) s) n))) a v := by
  exact score_response _ samples target p offset s n a v

example := attention_history witnessConfig (1 / 8388608) (1 / 1000000000000)
  (sourceAngles witnessConfig (fun d => d = 0) (fun i : Fin 5 => i.val))
  (fun a i : Fin 5 => (if a = i then 1 else 2 : Fin 3)) (fun _ => (4 : Fin 5))
  (fun _ => {0, 1}) (fun _ => (1 : Fin 3))
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400
  ⟨fun i => if i.isLeft then 1 else 2, fun _ => 3, ⟨fun _ => 1, 1, 3/1000⟩⟩ 1100

end Transformer.SequenceModel.Attention
