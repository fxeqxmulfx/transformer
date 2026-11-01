module

public import Transformer.SequenceModel.Attention.Stability.Aggregate
public import Transformer.SequenceModel.Attention.Stability.Local

/-!
# Full residual-stack rounding certificate

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B6--B7;
diff_gpt `b98c665b`, `gpt.py` and the rounded execution in `CRASP.Rotary`.
One-step estimates include both coefficient/quotient error and stored
residual rounding. The stack budget uses each executed block's own matrices.
Range premises cover only actual rounding sites; no hidden-state error is assumed.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini

/-- All rounding sites of the actual counted block at this position. -/
def StepInRange (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (a : ℕ → Fin (cfg.head_dim / 2) → ℝ) (q : ℕ → Rotary.State cfg p s) (i : ℕ) : Prop :=
  AggregateInRange cfg B eps p s a q i ∧
    ∀ c, Fx.NoSaturation p s (Normalized.localUpdate cfg B eps xeps (q i).1
      (Normalized.vector (q i).2)
      (fun h => Normalized.vector (fun j =>
        (Rotary.sourceLayer cfg B eps xeps p s a).aggregate (q i) q i (h, j))) c)

/-- New error introduced by this layer's rounding, apart from input error. -/
def stepRadius (cfg : Config) (B : BlockParams cfg) (eps : ℝ) (s : ℕ) : ℝ :=
  ffnGain cfg B eps * attentionGain cfg B eps *
    (2 * Real.sqrt (cfg.head_dim : ℝ) * aggregateRadius cfg B s) +
      Real.sqrt (cfg.d_model : ℝ) * (1 / (2 : ℝ) ^ s)

/-- B6: the actual counted step has error at most L*E+rho relative to the
real source block. L and rho are explicit functions of the model's matrices,
normalization epsilons, dimensions and fractional precision. -/
theorem source_step_error (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (p s : ℕ) {T : ℕ}
    (a : ℕ → Fin (cfg.head_dim / 2) → ℝ) (q : ℕ → Rotary.State cfg p s)
    (x : Fin T → EucSpace cfg.d_model) (E : ℝ)
    (hE : ∀ j, ‖Normalized.vector (q (j.val + 1)).2 - x j‖ ≤ E) (i : Fin T)
    (hflag : (q (i.val + 1)).1 = decide (i.val = 0))
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ)))
    (hr : StepInRange cfg B eps xeps p s a q (i.val + 1)) :
    ‖Normalized.vector ((Rotary.sourceLayer cfg B eps xeps p s a).step q (i.val + 1)).2 -
      block cfg B eps xeps (fun j : Fin T => a j.val) x i‖ ≤
      blockLip cfg B eps xeps * E + stepRadius cfg B eps s := by
  let z := fun j : Fin T => Normalized.vector (q (j.val + 1)).2
  let avg := fun h => Normalized.vector (fun c =>
    (Rotary.sourceLayer cfg B eps xeps p s a).aggregate (q (i.val + 1)) q (i.val + 1) (h, c))
  let ref := Normalized.sourceAverage cfg B eps (fun j : Fin T => a j.val) x i
  let raw := Normalized.localUpdate cfg B eps xeps (q (i.val + 1)).1 (z i) avg
  have havg h : ‖avg h - ref h‖ ≤
      Real.sqrt (cfg.head_dim : ℝ) * aggregateRadius cfg B s + averageLip cfg B eps * E := by
    exact (norm_sub_le_norm_sub_add_norm_sub _
      (Normalized.sourceAverage cfg B eps (fun j : Fin T => a j.val) z i h) _).trans
      (add_le_add (aggregate_vector_error cfg B heps xeps p s a q i h hp hr.1)
        (sourceAverage_error cfg B heps _ z x E hE i h))
  have hlocal : ‖raw - block cfg B eps xeps (fun j : Fin T => a j.val) x i‖ ≤
      ffnGain cfg B eps * (E + attentionGain cfg B eps *
        (2 * (Real.sqrt (cfg.head_dim : ℝ) * aggregateRadius cfg B s + averageLip cfg B eps * E) +
          directionGain cfg B eps xeps * E)) := by
    rw [← Normalized.local_sourceAverage]
    dsimp only [raw]
    rw [hflag]
    exact localUpdate_error cfg B heps hxeps _ (z i) (x i) avg ref E _
      (hE i) havg (sourceAverage_norm cfg B heps _ x i)
  have hround : ‖Normalized.vector
      ((Rotary.sourceLayer cfg B eps xeps p s a).step q (i.val + 1)).2 - raw‖ ≤
      Real.sqrt (cfg.d_model : ℝ) * (1 / (2 : ℝ) ^ s) := vector_round_error p s raw hr.2
  exact (norm_sub_le_norm_sub_add_norm_sub _ raw _).trans
    ((add_le_add hround hlocal).trans_eq (by unfold blockLip stepRadius; ring))

open CRASP GPTMini

/-- The existing uniform counting execution on the finite source token tuple. -/
def counted (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) {T : ℕ} (tokens : Fin T → Fin cfg.vocab_size) :
    ℕ → ℕ → Rotary.State cfg p s :=
  Rotary.run (Rotary.layers cfg θ eps xeps p s enabled)
    (Rotary.input (Rotary.embed cfg θ eps p s) (false, fun _ => 0) (List.ofFn tokens))

/-- Exact source phases for this layer, extended to natural zero-based positions. -/
def layerAngles (cfg : Config) (enabled : Fin cfg.n_layers → Bool)
    (l : Fin cfg.n_layers) (i : ℕ) (c : Fin (cfg.head_dim / 2)) : ℝ :=
  if enabled l then (i : ℝ) * invFreq cfg.head_dim cfg.rope_theta c else 0

/-- A concrete recurrence: embedding rounding, then each layer's L*E+rho.
Like `hidden`, the budget stops at the configured depth. -/
def errorBudget (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (s : ℕ) : ℕ → ℝ
  | 0 => Real.sqrt (cfg.d_model : ℝ) * (1 / (2 : ℝ) ^ s)
  | n + 1 => if h : n < cfg.n_layers then
      blockLip cfg (θ.blocks ⟨n, h⟩) eps xeps * errorBudget cfg θ eps xeps s n +
        stepRadius cfg (θ.blocks ⟨n, h⟩) eps s
    else errorBudget cfg θ eps xeps s n

/-- No clipping at the normalized input, coefficients, quotients and stored
residuals of the actual rounded run. The range can depend on the input word. -/
def TraceInRange (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) {T : ℕ} (tokens : Fin T → Fin cfg.vocab_size) : Prop :=
  (∀ i c, Fx.NoSaturation p s (rmsNormEps eps (θ.embedding (tokens i)) c)) ∧
    ∀ l : Fin cfg.n_layers, ∀ i : Fin T,
      StepInRange cfg (θ.blocks l) eps xeps p s (layerAngles cfg enabled l)
        (counted cfg θ eps xeps p s enabled tokens l.val) (i.val + 1)

/-- B7: the counted run keeps the same first-token XSA choice as the source. -/
theorem counted_flag (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) {T : ℕ} (tokens : Fin T → Fin cfg.vocab_size)
    (n : ℕ) (i : Fin T) :
    (counted cfg θ eps xeps p s enabled tokens n (i.val + 1)).1 = decide (i.val = 0) := by
  unfold counted
  rw [Rotary.first_flag_preserved, input_ofFn]
  rfl

/-- B7: every executed residual layer of the actual uniform counting model
is within the computed budget of the real source, at every input position.
Exact, nonperiodic RoPE and independent block weights are retained. -/
theorem stack_error (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) {T : ℕ} (tokens : Fin T → Fin cfg.vocab_size)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ)))
    (hr : TraceInRange cfg θ eps xeps p s enabled tokens) (n : ℕ) (hn : n ≤ cfg.n_layers)
    (i : Fin T) :
    ‖Normalized.vector (counted cfg θ eps xeps p s enabled tokens n (i.val + 1)).2 -
      hidden cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ))) tokens n i‖ ≤
      errorBudget cfg θ eps xeps s n := by
  induction n generalizing i with
  | zero =>
      change ‖Normalized.vector (Rotary.input (Rotary.embed cfg θ eps p s)
        (false, fun _ => 0) (List.ofFn tokens) (i.val + 1)).2 - _‖ ≤ _
      rw [input_ofFn]
      exact vector_round_error p s _ (hr.1 i)
  | succ n ih =>
      have hlt : n < cfg.n_layers := by omega
      let l : Fin cfg.n_layers := ⟨n, hlt⟩
      have hl := Rotary.layers_at cfg θ eps xeps p s enabled l
      change Rotary.layers cfg θ eps xeps p s enabled n = _ at hl
      change ‖Normalized.vector ((Rotary.layers cfg θ eps xeps p s enabled n).step
        (counted cfg θ eps xeps p s enabled tokens n) (i.val + 1)).2 - _‖ ≤ _
      rw [hl, hidden, dite_eq_left hlt, errorBudget, dite_eq_left hlt]
      exact source_step_error cfg (θ.blocks l) heps hxeps p s (layerAngles cfg enabled l)
        (counted cfg θ eps xeps p s enabled tokens n)
        (hidden cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ))) tokens n)
        (errorBudget cfg θ eps xeps s n) (ih (by omega)) i
        (counted_flag cfg θ eps xeps p s enabled tokens n i) hp (hr.2 l i)

end Transformer.SequenceModel.Attention.Stability
