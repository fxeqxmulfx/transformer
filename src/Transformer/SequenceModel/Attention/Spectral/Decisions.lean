module

public import Transformer.SequenceModel.Attention.Spectral.Numerics

/-!
# Every class decision follows the transferred source margin

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J6--J7. Strict prediction has a unique winner. This extends positive-target
certificates to equality of the complete Boolean class-decision vectors.
The executed source margin is derived from the spectral construction.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open scoped Classical

open GPTMini Stability Execution CRASP

/-- J7: a proved strict winner determines every Boolean class test,
including rejection of every competing class. -/
theorem strict_tests_eq {V : Type*} (s : V → ℝ) (target : V)
    (hs : ∀ v, v ≠ target → s v < s target) (v : V) :
    decide (∀ w, w ≠ v → s w < s v) = decide (v = target) := by
  classical
  by_cases hv : v = target
  · subst v
    rw [decide_eq_true hs, decide_eq_true rfl]
  · rw [decide_eq_false hv]
    apply decide_eq_false
    intro h
    exact lt_asymm (h target (Ne.symm hv)) (hs v hv)

/-- J7: every recorded execution class test is determined by its strict winner. -/
theorem executed_tests_eq (cfg : Config) {T : ℕ} (t : ModelTrace cfg T) (i : Fin T)
    (target : Fin cfg.vocab_size) (ht : t.predict i target = true) (v : Fin cfg.vocab_size) :
    t.predict i v = decide (v = target) := by
  simpa only [ModelTrace.predict, decide_eq_decide] using
    strict_tests_eq (t.logits i) target (of_decide_eq_true ht) v

/-- J7: the same uniqueness holds for the actual rounded counting-model readout. -/
theorem counted_tests_eq (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) (w : List (Fin cfg.vocab_size)) (i : ℕ)
    (target : Fin cfg.vocab_size) (ht : Rotary.predict cfg θ eps xeps p s enabled w i target = true)
    (v : Fin cfg.vocab_size) :
    Rotary.predict cfg θ eps xeps p s enabled w i v = decide (v = target) := by
  unfold Rotary.predict Rotary.readout at ht ⊢
  simpa only [decide_eq_decide] using strict_tests_eq _ target (of_decide_eq_true ht) v

/-- J6: the spectral source margin and the computed operation budget certify
the executed prediction, including all local operations and stored RoPE tables. -/
theorem executed_of_source_margin (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {N : ℕ}
    (x : Fin (N + 1) → Fin cfg.vocab_size) (target : Fin cfg.vocab_size) (γ : ℝ)
    (hg : ∀ v, v ≠ target → γ ≤ sourceLogit cfg θ eps xeps x target - sourceLogit cfg θ eps xeps x v)
    (t : ModelTrace cfg (N + 1)) (δ τ : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (ht : ModelAccurate cfg θ eps xeps
      (sourceAngles cfg (fun _ => true) (fun j : Fin (N + 1) => (j.val : ℝ))) x t δ τ)
    (hb : ExecutionBelow cfg θ eps xeps δ τ γ) : t.predict (Fin.last N) target = true := by
  apply Execution.predict_true cfg θ heps hxeps _ x t δ τ hδ hτ ht (Fin.last N) target
  intro v hv
  have h := (hb target v).trans_le (hg v hv)
  simpa only [sourceLogit, score, decode_encode] using h

example := strict_tests_eq (fun v : Fin 3 => if v = 1 then (2 : ℝ) else -1) 1
  (by intro v hv; norm_num [hv]) 2

end Transformer.SequenceModel.Attention.Spectral
