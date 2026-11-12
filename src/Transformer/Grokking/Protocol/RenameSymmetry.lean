module

public import Transformer.Grokking.Protocol.Symmetry

/-!
# A renaming of the tokens that moves the batch

Source: arXiv:2201.02177v1 (Power et al.), section 1: the equations use "distinct abstract
symbols"; openai/grok at 3d64b1d8, `grok/transformer.py`, where a token enters only as a row
of the embedding and leaves only as a row of the readout; `experiments/grok_division/PROTOCOL.md`,
*The model*.

`Symmetry` permutes the gradient with the parameters on a batch that the renaming of the
tokens leaves alone. Here the batch may move. The decoder with the permuted parameters reads
an example as the decoder reads the renamed example (`Example.rename`: every token read and
every target renamed), so the loss with frozen gates at the permuted parameters is that of
the renamed batch (`frozenLoss_comp_paramPerm_rename`), and the gradient of the backward pass
at the permuted parameters is the permuted gradient of the renamed batch
(`sourceGrad_comp_paramPerm_rename`). For a batch that the renaming leaves alone this is
`sourceGrad_comp_paramPerm`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy)

variable {S : Sizes} (τ : Equiv.Perm (Fin S.vocab))

/-- The example with every token read and every target renamed. -/
def Example.rename (e : Example S) : Example S :=
  ⟨τ ∘ e.tokens, e.targets.map fun pc => (pc.1, τ pc.2)⟩

/-- A renaming keeps the number of targets of a batch. -/
lemma targetCount_map_rename (B : List (Example S)) :
    targetCount (B.map (Example.rename τ)) = targetCount B := by
  simp [targetCount, Example.rename, Function.comp_def]

/-- A renaming that leaves an example alone does not change it. -/
lemma Example.rename_of_fixed {e : Example S} (he : e.Fixed τ) : e.rename τ = e := by
  obtain ⟨tokens, targets⟩ := e
  have ht : τ ∘ tokens = tokens := funext he.1
  have hy : targets.map (fun pc => (pc.1, τ pc.2)) = targets := by
    conv_rhs => rw [← List.map_id targets]
    exact List.map_congr_left fun pc hpc => Prod.ext rfl (he.2 pc hpc)
  simp only [Example.rename, ht, hy]

variable {τ} {θ θ₀ : Param S → ℝ}

/-- The loss of an example at the permuted parameters is that of the renamed example. -/
lemma exampleLoss_comp_paramPerm_rename (e : Example S) :
    exampleLoss (θ ∘ paramPerm τ) (gate (θ₀ ∘ paramPerm τ) e.tokens) e =
      exampleLoss θ (gate θ₀ (e.rename τ).tokens) (e.rename τ) := by
  simp only [exampleLoss, gate_comp_paramPerm, Example.rename, List.map_map]
  refine congrArg List.sum (List.map_congr_left fun pc _ => ?_)
  have hz : logits (θ ∘ paramPerm τ) (gate θ₀ (τ ∘ e.tokens)) e.tokens pc.1 =
      logits θ (gate θ₀ (τ ∘ e.tokens)) (τ ∘ e.tokens) pc.1 ∘ τ := by
    funext c
    rw [logits_comp_paramPerm]
    rfl
  rw [Function.comp_apply, hz, crossEntropy_comp_perm]

/-- **The loss with frozen gates at the permuted parameters is that of the renamed batch.** -/
theorem frozenLoss_comp_paramPerm_rename (B : List (Example S)) :
    frozenLoss (θ₀ ∘ paramPerm τ) (θ ∘ paramPerm τ) B =
      frozenLoss θ₀ θ (B.map (Example.rename τ)) := by
  simp only [frozenLoss, targetCount_map_rename, List.map_map]
  rw [List.map_congr_left fun e _ => exampleLoss_comp_paramPerm_rename (θ := θ) (θ₀ := θ₀) e]
  rfl

/-- **The gradient of the backward pass at the permuted parameters is the permuted gradient of
the renamed batch.** -/
theorem sourceGrad_comp_paramPerm_rename (B : List (Example S)) (θ : Param S → ℝ)
    (p : Param S) :
    sourceGrad B (θ ∘ paramPerm τ) p =
      sourceGrad (B.map (Example.rename τ)) θ (paramPerm τ p) := by
  have hupd : ∀ s : ℝ, Function.update (θ ∘ paramPerm τ) p s =
      Function.update θ (paramPerm τ p) s ∘ paramPerm τ := fun s => by
    rw [Function.update_comp_equiv, Equiv.symm_apply_apply]
  simp only [sourceGrad, hupd, frozenLoss_comp_paramPerm_rename, Function.comp_apply]

/-! The witnesses: the identity renaming, which leaves every example alone and does not change
it. `TwinSplit` has a renaming that moves two tokens and two examples. -/

example (e : Example S) : e.rename 1 = e :=
  Example.rename_of_fixed 1 ⟨fun _ => rfl, fun _ _ => rfl⟩

end Transformer.Grokking.Protocol
