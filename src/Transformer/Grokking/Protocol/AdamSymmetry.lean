module

public import Transformer.Grokking.Protocol.Adam

/-!
# The author's Adam run commutes with a reindexing of the parameters

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`, which updates every
entry of every parameter by the same rule from that entry's own gradient and moments;
`experiments/grok_division/PROTOCOL.md`, *The optimizer*. The run is `adamRun` of
`Transformer.Grokking.Protocol.Adam`.

If the gradient oracle commutes with a reindexing `ρ` of the parameters, that is
`grad t (θ ∘ ρ) = grad t θ ∘ ρ`, then so does the run: started from `θ₀ ∘ ρ` its parameters and
both moments after `t` updates are those of the run from `θ₀`, reindexed (`adamRun_comp`). The
weight decay does not matter, since it too acts entry by entry.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {ι κ : Type*} {h : Hyper}

/-- **A reindexing that the gradients commute with commutes with the run.** -/
theorem adamRun_comp (ρ : κ → ι) {grad : ℕ → (ι → ℝ) → ι → ℝ} {grad' : ℕ → (κ → ℝ) → κ → ℝ}
    (hg : ∀ t θ, grad' t (θ ∘ ρ) = grad t θ ∘ ρ) (θ₀ : ι → ℝ) (t : ℕ) :
    (adamRun h grad' (θ₀ ∘ ρ) t).params = (adamRun h grad θ₀ t).params ∘ ρ ∧
      (adamRun h grad' (θ₀ ∘ ρ) t).first = (adamRun h grad θ₀ t).first ∘ ρ ∧
      (adamRun h grad' (θ₀ ∘ ρ) t).second = (adamRun h grad θ₀ t).second ∘ ρ := by
  induction t with
  | zero => exact ⟨rfl, rfl, rfl⟩
  | succ t ih =>
    obtain ⟨hp, hm, hv⟩ := ih
    refine ⟨?_, ?_, ?_⟩ <;> funext i <;>
      simp only [adamRun, adamStep, hp, hm, hv, hg, Function.comp_apply]

/-! The witness: two parameters swapped, and the gradient that returns the parameters, which
commutes with every reindexing. -/

example (θ₀ : Bool → ℝ) (t : ℕ) :=
  adamRun_comp (h := Hyper.source 0) Bool.not (grad := fun _ θ => θ) (grad' := fun _ θ => θ)
    (fun _ _ => rfl) θ₀ t

end Transformer.Grokking.Protocol
