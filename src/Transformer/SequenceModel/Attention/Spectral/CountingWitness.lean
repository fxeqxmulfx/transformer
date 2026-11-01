module

public import Transformer.SequenceModel.Attention.Spectral.Counting

/-!
# Concrete range, rounding and nonzero execution certificates

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J7. These witnesses use the existing three-class source at eps=3/4,
xeps=1, p=8, s=5 and a nonzero logit error 1/1024. Every numerical
premise is discharged. They do not certify a concrete GPU implementation.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Stability Execution CRASP

/-- J7 witness: the fixed-point budget is below the unit margin for all class pairs. -/
theorem example_counting_below : CountingBelow witnessConfig exampleModel (3 / 4) 1 5 1 := by
  intro u v
  change (rmsLip (3 / 4) * errorBudget witnessConfig exampleModel (3 / 4) 1 5 2) *
    (‖exampleEmbedding u‖ + ‖exampleEmbedding v‖) < 1
  have h := example_margin (fun i : Fin 3 => i) 0 1 (by decide)
  rw [example_hidden, example_embedding_rms] at h
  change (rmsLip (3 / 4) * errorBudget witnessConfig exampleModel (3 / 4) 1 5 2) *
    (‖exampleEmbedding 0‖ + ‖exampleEmbedding 1‖) <
      inner ℝ (exampleEmbedding 0) (exampleEmbedding 0) -
      inner ℝ (exampleEmbedding 0) (exampleEmbedding 1) at h
  rw [example_embedding_norm, example_embedding_norm, example_embedding_inner,
    example_embedding_inner] at h
  rw [example_embedding_norm, example_embedding_norm]
  norm_num at h ⊢
  exact h

/-- J7 witness: the executed class radii also fit strictly below the same margin. -/
theorem example_execution_below : ExecutionBelow witnessConfig exampleModel (3 / 4) 1 (1 / 1024) 0 1 := by
  intro u v
  have hu := example_logitRadius u
  have hv := example_logitRadius v
  change logitRadius witnessConfig (3 / 4) (exampleEmbedding u)
    (modelBudget witnessConfig exampleModel (3 / 4) 1 (1 / 1024) 0 2) (1 / 1024) +
    logitRadius witnessConfig (3 / 4) (exampleEmbedding v)
    (modelBudget witnessConfig exampleModel (3 / 4) 1 (1 / 1024) 0 2) (1 / 1024) < 1
  linarith

/-- J7 witness: one fixed formula family gives all three class decisions,
with every range, precision and execution premise explicitly satisfied. -/
theorem example_uniform_decisions :
    ∃ φ : Fin witnessConfig.vocab_size → Rotary.Form (Fin witnessConfig.vocab_size)
        (Rotary.Key (Rotary.State witnessConfig 8 5) (Normalized.Coordinate witnessConfig)),
      (∀ v, (φ v).depth ≤ witnessConfig.n_layers) ∧
      ∀ (x : Fin 2 → Fin witnessConfig.vocab_size), x 0 = x 1 → ∀ v,
        (φ v).sat (Rotary.kernel (Rotary.layers witnessConfig exampleModel (3 / 4) 1 8 5
          (fun _ => true))) (List.ofFn x) 2 = decide (v = x 0) ∧
        (shiftedModel witnessConfig exampleModel (3 / 4) 1
          (sourceAngles witnessConfig (fun _ => true) (fun j : Fin 2 => (j.val : ℝ)))
          x (1 / 1024)).predict 1 v = decide (v = x 0) := by
  obtain ⟨φ, hd, hφ⟩ := uniform_margin_certificate witnessConfig exampleModel
    (eps := 3 / 4) (xeps := 1) (by norm_num) (by norm_num) 8 5 example_precision
    (fun x : Fin 2 → Fin witnessConfig.vocab_size => x 0 = x 1) (fun x => x 0) 1
    identity_source_margin
  refine ⟨φ, hd, ?_⟩
  intro x hx
  exact hφ x hx _ (1 / 1024) 0 (by norm_num) (by norm_num)
    (shiftedModel_accurate _ _ _ _ _ x _ _ (by norm_num) (by norm_num))
    (example_trace x) example_counting_below example_execution_below

example := executed_tests_eq witnessConfig _ 1 1
  (example_executed_prediction (fun i : Fin 3 => i) 1) 0
example := counted_tests_eq witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true)
  (List.ofFn (fun i : Fin 3 => i)) 2 1 (example_prediction (fun i : Fin 3 => i) 1) 0
example := executed_of_source_margin witnessConfig exampleModel
  (eps := 3 / 4) (xeps := 1) (by norm_num) (by norm_num) ![2, 2] 2 1
  (identity_source_margin ![2, 2] rfl) _ (1 / 1024) 0 (by norm_num) (by norm_num)
  (shiftedModel_accurate _ _ _ _ _ _ _ _ (by norm_num) (by norm_num)) example_execution_below

end Transformer.SequenceModel.Attention.Spectral
