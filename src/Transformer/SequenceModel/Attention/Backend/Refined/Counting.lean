module

public import Transformer.SequenceModel.Attention.Backend.Refined.Readout
public import Transformer.SequenceModel.Attention.Spectral.LocalCounting

/-!
# The captured execution satisfies the new counting bridge's execution side

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M5.
The source margin and hardware logit radii are proved, not assumed. Fixed-point
range and rounding conditions concern a different execution and remain explicit.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution Complete Adaptive CRASP

/-- M5: the saved invocation uses the same exact source phases as the spectral bridge. -/
theorem source_angles : angles =
    sourceAngles cfg (fun _ => true) (fun j : Fin 2 => (j.val : ℝ)) := by
  funext l i c
  fin_cases l
  rfl

/-- M5: the operation-graph readout is the actual source score used by flat spectrum. -/
theorem source_score (v : Fin 3) : sourceLogits 1 v =
    Spectral.sourceLogit cfg weights (1 / 8388608) (1 / 1000000000000) tokens v := by
  simp only [sourceLogits, source_hidden, source_angles, Spectral.sourceLogit, score, decode_encode]
  rfl

/-- M5: one fixed formula family accepts the captured trace with radius 1/16
and derived source margin 7/10. Only the separate fixed-point conditions remain. -/
theorem captured_counting_bridge (p s : ℕ)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ))) :
    ∃ φ : Fin cfg.vocab_size → Rotary.Form (Fin cfg.vocab_size)
        (Rotary.Key (Rotary.State cfg p s) (Normalized.Coordinate cfg)),
      (∀ v, (φ v).depth ≤ cfg.n_layers) ∧
      (TraceInRange cfg weights (1 / 8388608) (1 / 1000000000000) p s (fun _ => true) tokens →
        Spectral.CountingBelow cfg weights (1 / 8388608) (1 / 1000000000000) s (7 / 10) →
        ∀ v, (φ v).sat (Rotary.kernel (Rotary.layers cfg weights (1 / 8388608)
          (1 / 1000000000000) p s (fun _ => true))) (List.ofFn tokens) 2 = captured.predict 1 v) := by
  have hg : ∀ x : Fin 2 → Fin cfg.vocab_size, x = tokens → ∀ v, v ≠ (2 : Fin 3) →
      (7 / 10 : ℝ) ≤ Spectral.sourceLogit cfg weights (1 / 8388608) (1 / 1000000000000) x 2 -
        Spectral.sourceLogit cfg weights (1 / 8388608) (1 / 1000000000000) x v := by
    intro x hx v hv
    subst x
    rw [← source_score, ← source_score]
    exact source_margin 1 v hv
  obtain ⟨φ, hd, hφ⟩ := Spectral.uniform_local_certificate cfg weights
    (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
    p s hp (fun x : Fin 2 → Fin cfg.vocab_size => x = tokens) (fun _ => 2) (7 / 10) hg
  refine ⟨φ, hd, ?_⟩
  intro htrace hcount v
  have h := hφ tokens rfl captured (fun _ => 1 / 16)
    (by intro u; rw [← source_score]; exact error_logits 1 u)
    (by intro u hu; norm_num) htrace hcount v
  exact h.1.trans h.2.symm

example := captured_counting_bridge 8 5 example_precision

end Transformer.SequenceModel.Attention.Backend.Refined
