module

public import Transformer.SequenceModel.Attention.Spectral.Counting

/-!
# Counting transfer with independently certified class radii

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M5.
The spectral construction supplies a source margin. An operation-dependent
execution proof can supply class radii without satisfying the old uniform
ModelAccurate recurrence. The independent fixed-point conditions remain.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Spectral
open GPTMini Stability Execution CRASP

/-- M5: any proved classwise logit error can transfer the spectral source margin. -/
theorem executed_of_logit_bound (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ)
    {N : ℕ} (x : Fin (N + 1) → Fin cfg.vocab_size) (target : Fin cfg.vocab_size) (γ : ℝ)
    (hg : ∀ v, v ≠ target → γ ≤ sourceLogit cfg θ eps xeps x target - sourceLogit cfg θ eps xeps x v)
    (t : ModelTrace cfg (N + 1)) (r : Fin cfg.vocab_size → ℝ)
    (he : ∀ v, |t.logits (Fin.last N) v - sourceLogit cfg θ eps xeps x v| ≤ r v)
    (hr : ∀ v, v ≠ target → r target + r v < γ) : t.predict (Fin.last N) target = true := by
  apply decide_eq_true
  intro v hv
  have h₁ := (abs_le.mp (he target)).1
  have h₂ := (abs_le.mp (he v)).2
  have hm := hg v hv
  have hb := hr v hv
  linarith

/-- M5: the fixed counting family accepts local execution radii as its numerical
interface. Formulae are chosen before all input words and execution traces. -/
theorem uniform_local_certificate (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (p s : ℕ)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ))) {N : ℕ}
    (valid : (Fin (N + 1) → Fin cfg.vocab_size) → Prop)
    (target : (Fin (N + 1) → Fin cfg.vocab_size) → Fin cfg.vocab_size) (γ : ℝ)
    (hg : ∀ x, valid x → ∀ v, v ≠ target x →
      γ ≤ sourceLogit cfg θ eps xeps x (target x) - sourceLogit cfg θ eps xeps x v) :
    ∃ φ : Fin cfg.vocab_size → Rotary.Form (Fin cfg.vocab_size)
        (Rotary.Key (Rotary.State cfg p s) (Normalized.Coordinate cfg)),
      (∀ v, (φ v).depth ≤ cfg.n_layers) ∧
      ∀ (x : Fin (N + 1) → Fin cfg.vocab_size), valid x →
        ∀ (t : ModelTrace cfg (N + 1)) (r : Fin cfg.vocab_size → ℝ),
        (∀ v, |t.logits (Fin.last N) v - sourceLogit cfg θ eps xeps x v| ≤ r v) →
        (∀ v, v ≠ target x → r (target x) + r v < γ) →
        TraceInRange cfg θ eps xeps p s (fun _ => true) x → CountingBelow cfg θ eps xeps s γ →
        ∀ v, (φ v).sat (Rotary.kernel (Rotary.layers cfg θ eps xeps p s (fun _ => true)))
          (List.ofFn x) (N + 1) = decide (v = target x) ∧
          t.predict (Fin.last N) v = decide (v = target x) := by
  classical
  choose φ hd hφ using (fun v => Rotary.uniform_source_classifier cfg θ eps xeps p s (fun _ => true) v)
  refine ⟨φ, hd, ?_⟩
  intro x hx t r he hr htrace hcount
  have hm : Stability.MarginAbove cfg θ.embedding eps
      (hidden cfg θ eps xeps (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ)))
        x cfg.n_layers (Fin.last N)) (errorBudget cfg θ eps xeps s cfg.n_layers) (target x) := by
    intro v hv
    have h := (hcount (target x) v).trans_le (hg x hx v hv)
    simpa only [sourceLogit, score, decode_encode] using h
  have hc := rounded_predict_true cfg θ heps hxeps p s (fun _ => true) x hp htrace
    (Fin.last N) (target x) hm
  have ht := executed_of_logit_bound cfg θ eps xeps x (target x) γ (hg x hx) t r he hr
  intro v
  constructor
  · rw [hφ v (List.ofFn x) (N + 1) (by omega) (by simp)]
    exact counted_tests_eq cfg θ eps xeps p s (fun _ => true) (List.ofFn x) (N + 1) (target x) hc v
  · exact executed_tests_eq cfg t (Fin.last N) (target x) ht v

example := executed_of_logit_bound witnessConfig exampleModel (3 / 4) 1 ![2, 2] 2 1
  (identity_source_margin ![2, 2] rfl)
  (exactModel witnessConfig exampleModel (3 / 4) 1
    (sourceAngles witnessConfig (fun _ => true) (fun j : Fin 2 => (j.val : ℝ))) ![2, 2])
  (fun _ => 0) (by intro v; simp [exactModel, sourceLogit, score, decode_encode])
  (by intro v hv; norm_num)
example := uniform_local_certificate witnessConfig exampleModel (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num) 8 5 example_precision
  (fun x : Fin 2 → Fin witnessConfig.vocab_size => x 0 = x 1) (fun x => x 0) 1
  identity_source_margin

end Transformer.SequenceModel.Attention.Spectral
