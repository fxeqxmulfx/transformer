module

public import Transformer.SequenceModel.Attention.Execution.Witness

/-!
# Executed predictions and one uniform exact-RoPE counting formula

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C7--C8.
Both the execution trace and the existing rounded counting model are
certified against the same exact real source. Agreement is conditional on
their respective numerical certificates, uniformly in the input length.
This supplies no strict depth hierarchy for nonperiodic source kernels.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini CRASP Stability

/-- C7: a formula fixed before the input agrees with every certified
executed target prediction. Its depth is at most the source block count.
The separate fixed-point range/margin and execution margin remain explicit. -/
theorem uniform_execution_certificate (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) (target : Fin cfg.vocab_size)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ))) :
    ∃ φ : Rotary.Form (Fin cfg.vocab_size)
        (Rotary.Key (Rotary.State cfg p s) (Normalized.Coordinate cfg)),
      φ.depth ≤ cfg.n_layers ∧ ∀ (T : ℕ) (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
        (t : ModelTrace cfg T) (δ τ : ℝ), 0 ≤ δ → 0 ≤ τ →
        ModelAccurate cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ)))
          tokens t δ τ →
        TraceInRange cfg θ eps xeps p s enabled tokens →
        Stability.MarginAbove cfg θ.embedding eps
          (hidden cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ)))
            tokens cfg.n_layers i) (errorBudget cfg θ eps xeps s cfg.n_layers) target →
        MarginAbove cfg θ eps
          (hidden cfg θ eps xeps (sourceAngles cfg enabled (fun j : Fin T => (j.val : ℝ)))
            tokens cfg.n_layers i) (modelBudget cfg θ eps xeps δ τ cfg.n_layers) δ target →
        φ.sat (Rotary.kernel (Rotary.layers cfg θ eps xeps p s enabled))
          (List.ofFn tokens) (i.val + 1) = true ∧ t.predict i target = true := by
  obtain ⟨φ, hd, hφ⟩ := uniform_source_certificate cfg θ heps hxeps p s enabled target hp
  refine ⟨φ, hd, ?_⟩
  intro T tokens i t δ τ hδ hτ ht hr hm he
  exact ⟨hφ T tokens i hr hm, predict_true cfg θ heps hxeps _ tokens t δ τ hδ hτ ht i target he⟩

/-- C8: the same fixed counting formula and the nonzero-error execution
recognize their target token at every position of every finite word. -/
theorem example_uniform_execution (target : Fin witnessConfig.vocab_size) :
    ∃ φ : Rotary.Form (Fin witnessConfig.vocab_size)
        (Rotary.Key (Rotary.State witnessConfig 8 5) (Normalized.Coordinate witnessConfig)),
      φ.depth ≤ witnessConfig.n_layers ∧
      ∀ (T : ℕ) (tokens : Fin T → Fin witnessConfig.vocab_size) (i : Fin T), tokens i = target →
        φ.sat (Rotary.kernel (Rotary.layers witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true)))
          (List.ofFn tokens) (i.val + 1) = true ∧
        (shiftedModel witnessConfig exampleModel (3 / 4) 1
          (sourceAngles witnessConfig (fun _ => true) (fun j : Fin T => (j.val : ℝ)))
          tokens (1 / 1024)).predict i target = true := by
  obtain ⟨φ, hd, hφ⟩ := uniform_execution_certificate witnessConfig exampleModel
    (eps := 3 / 4) (xeps := 1) (by norm_num) (by norm_num) 8 5 (fun _ => true) target example_precision
  refine ⟨φ, hd, ?_⟩
  intro T tokens i hi
  apply hφ T tokens i _ (1 / 1024) 0 (by norm_num) (by norm_num)
    (shiftedModel_accurate _ _ _ _ _ _ _ _ (by norm_num) (by norm_num)) (example_trace tokens)
  · rw [← hi]
    exact example_margin tokens i
  · rw [← hi]
    exact example_execution_margin tokens i

example := uniform_execution_certificate witnessConfig exampleModel (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num) 8 5 (fun _ => true) 2 example_precision

end Transformer.SequenceModel.Attention.Execution
