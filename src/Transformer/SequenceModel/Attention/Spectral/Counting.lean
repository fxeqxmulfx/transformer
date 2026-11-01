module

public import Transformer.SequenceModel.Attention.Spectral.Decisions

/-!
# One fixed RoPE counting family agrees with all certified class decisions

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J7. Formulae are chosen before the input word and reproduce the rounded
source model on every length. On the finite task domain with a proved source
margin, numerical certificates imply their full class-decision agreement
with the executed transformer. A strict depth lower bound is not asserted.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Stability Execution CRASP

/-- J7: the source margin yields equality for every class decision under
both numerical certificates. Each formula has counting depth at most D,
and is selected independently of the particular input and execution trace. -/
theorem uniform_margin_certificate (cfg : Config) (θ : ModelParams cfg) {eps xeps : ℝ}
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
        ∀ (t : ModelTrace cfg (N + 1)) (δ τ : ℝ), 0 ≤ δ → 0 ≤ τ →
        ModelAccurate cfg θ eps xeps
          (sourceAngles cfg (fun _ => true) (fun j : Fin (N + 1) => (j.val : ℝ))) x t δ τ →
        TraceInRange cfg θ eps xeps p s (fun _ => true) x →
        CountingBelow cfg θ eps xeps s γ → ExecutionBelow cfg θ eps xeps δ τ γ →
        ∀ v, (φ v).sat (Rotary.kernel (Rotary.layers cfg θ eps xeps p s (fun _ => true)))
          (List.ofFn x) (N + 1) = decide (v = target x) ∧
          t.predict (Fin.last N) v = decide (v = target x) := by
  classical
  choose φ hd hφ using (fun v => Rotary.uniform_source_classifier cfg θ eps xeps p s (fun _ => true) v)
  refine ⟨φ, hd, ?_⟩
  intro x hx t δ τ hδ hτ ht hr hcount hexec
  have hm : Stability.MarginAbove cfg θ.embedding eps
      (hidden cfg θ eps xeps (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ)))
        x cfg.n_layers (Fin.last N)) (errorBudget cfg θ eps xeps s cfg.n_layers) (target x) := by
    intro v hv
    have h := (hcount (target x) v).trans_le (hg x hx v hv)
    simpa only [sourceLogit, score, decode_encode] using h
  have hrounded := rounded_predict_true cfg θ heps hxeps p s (fun _ => true) x hp hr
    (Fin.last N) (target x) hm
  have hexecuted := executed_of_source_margin cfg θ heps hxeps x (target x) γ (hg x hx)
    t δ τ hδ hτ ht hexec
  intro v
  constructor
  · rw [hφ v (List.ofFn x) (N + 1) (by omega) (by simp)]
    exact counted_tests_eq cfg θ eps xeps p s (fun _ => true) (List.ofFn x) (N + 1) (target x) hrounded v
  · exact executed_tests_eq cfg t (Fin.last N) (target x) hexecuted v

/-- J7 witness: the existing nonconstant source has unit margin on equal
two-token inputs, with the target specified by the first original token. -/
theorem identity_source_margin (x : Fin 2 → Fin witnessConfig.vocab_size) (hx : x 0 = x 1)
    (v : Fin witnessConfig.vocab_size) (hv : v ≠ x 0) :
    1 ≤ sourceLogit witnessConfig exampleModel (3 / 4) 1 x (x 0) -
      sourceLogit witnessConfig exampleModel (3 / 4) 1 x v := by
  simp only [sourceLogit, score, decode_encode, example_hidden, example_embedding_rms]
  change 1 ≤ inner ℝ (exampleEmbedding (x 1)) (exampleEmbedding (x 0)) -
    inner ℝ (exampleEmbedding (x 1)) (exampleEmbedding v)
  rw [← hx, example_embedding_inner, example_embedding_inner]
  simp [hv.symm]

example := identity_source_margin ![(2 : Fin 3), 2] rfl 1 (by decide)
example := uniform_margin_certificate witnessConfig exampleModel (eps := 3 / 4) (xeps := 1)
  (by norm_num) (by norm_num) 8 5 example_precision
  (fun x : Fin 2 → Fin witnessConfig.vocab_size => x 0 = x 1) (fun x => x 0) 1
  identity_source_margin

end Transformer.SequenceModel.Attention.Spectral
