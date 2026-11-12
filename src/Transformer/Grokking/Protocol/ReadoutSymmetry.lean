module

public import Transformer.Grokking.Protocol.Symmetry

/-!
# Renaming the classes of the readout alone is a symmetry of the backward pass

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, `Transformer`: the readout
`self.linear` is a matrix of its own, not tied to the embedding, and a token leaves the model
only as a row of it; `grok/training.py`, `TrainableTransformer._step`;
`experiments/grok_division/PROTOCOL.md`, *The model* and *The loss and the accuracy*.

`Symmetry` renames a token everywhere: its row of the embedding and its row of the readout
move together, and an example must neither read nor target the token. Here the rows of the
readout alone are permuted (`outPerm`). No block reads the readout, so the outputs of the
blocks and the gates of the backward pass are unchanged (`decodeTo_comp_outPerm`,
`gate_comp_outPerm`), and the logit of a token is the logit of the renamed token
(`logits_comp_outPerm`). The cross-entropy does not see a renaming of the classes
(`crossEntropy_comp_perm`).

So on a batch none of whose targets is moved (`Example.TargetsFixed`; the tokens read may be
any) the loss with frozen gates is the same at the permuted parameters
(`frozenLoss_comp_outPerm`), and the gradient of the backward pass is permuted with them
(`sourceGrad_comp_outPerm`): exchanging the rows of two tokens that are never targets exchanges
their gradients and changes no loss.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy)

variable {S : Sizes} (τ : Equiv.Perm (Fin S.vocab))

/-- **The permutation of the rows of the readout** that a permutation of the tokens induces;
the embedding and every other parameter stay. -/
def outPerm : Equiv.Perm (Param S) where
  toFun
    | .out token i => .out (τ token) i
    | p => p
  invFun
    | .out token i => .out (τ.symm token) i
    | p => p
  left_inv p := by cases p <;> simp
  right_inv p := by cases p <;> simp

variable {τ} {θ θ₀ : Param S → ℝ} {act : Activation S}

/-- **No block reads the readout**: permuting its rows changes no block's output. -/
theorem decodeTo_comp_outPerm (tokens : Fin S.context → Fin S.vocab) (n : ℕ) :
    decodeTo (θ ∘ outPerm τ) act tokens n = decodeTo θ act tokens n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [decodeTo]
    split_ifs with hn
    · funext t
      rw [ih]
      rfl
    · exact ih

/-- **The logit of a token at permuted rows is the logit of the renamed token**, for the same
tokens read. -/
theorem logits_comp_outPerm (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context)
    (c : Fin S.vocab) :
    logits (θ ∘ outPerm τ) act tokens t c = logits θ act tokens t (τ c) := by
  simp only [logits, decodeTo_comp_outPerm]
  rfl

/-- The gates of the backward pass do not depend on the readout. -/
theorem gate_comp_outPerm (tokens : Fin S.context → Fin S.vocab) :
    gate (θ ∘ outPerm τ) tokens = gate θ tokens := by
  funext l t u v
  simp only [gate, decodeTo_comp_outPerm]
  rfl

variable (τ) in
/-- An example none of whose targets the renaming `τ` moves. -/
def Example.TargetsFixed (e : Example S) : Prop := ∀ pc ∈ e.targets, τ pc.2 = pc.2

lemma exampleLoss_comp_outPerm {e : Example S} (he : e.TargetsFixed τ) :
    exampleLoss (θ ∘ outPerm τ) (gate (θ₀ ∘ outPerm τ) e.tokens) e =
      exampleLoss θ (gate θ₀ e.tokens) e := by
  simp only [exampleLoss, gate_comp_outPerm]
  refine congrArg List.sum (List.map_congr_left fun pc hpc => ?_)
  have hz : logits (θ ∘ outPerm τ) (gate θ₀ e.tokens) e.tokens pc.1 =
      logits θ (gate θ₀ e.tokens) e.tokens pc.1 ∘ τ := by
    funext c
    rw [logits_comp_outPerm]
    rfl
  rw [hz, crossEntropy_comp_perm, he pc hpc]

/-- **On a batch none of whose targets is moved the frozen loss is the same at the permuted
parameters.** -/
theorem frozenLoss_comp_outPerm {B : List (Example S)} (hB : ∀ e ∈ B, e.TargetsFixed τ) :
    frozenLoss (θ₀ ∘ outPerm τ) (θ ∘ outPerm τ) B = frozenLoss θ₀ θ B := by
  simp only [frozenLoss]
  rw [List.map_congr_left fun e he => exampleLoss_comp_outPerm (hB e he)]

/-- **The gradient of the backward pass is permuted with the rows of the readout**, on a batch
none of whose targets is moved. -/
theorem sourceGrad_comp_outPerm {B : List (Example S)} (hB : ∀ e ∈ B, e.TargetsFixed τ)
    (θ : Param S → ℝ) (p : Param S) :
    sourceGrad B (θ ∘ outPerm τ) p = sourceGrad B θ (outPerm τ p) := by
  have hupd : ∀ s : ℝ, Function.update (θ ∘ outPerm τ) p s =
      Function.update θ (outPerm τ p) s ∘ outPerm τ := fun s => by
    rw [Function.update_comp_equiv, Equiv.symm_apply_apply]
  simp only [sourceGrad, hupd, frozenLoss_comp_outPerm hB, Function.comp_apply]

/-! The witnesses: the identity moves no target. `Unlabelled` has a renaming that moves two
tokens and the examples none of whose targets it moves. -/

example (θ₀ θ : Param S → ℝ) (e : Example S) :=
  exampleLoss_comp_outPerm (τ := 1) (θ₀ := θ₀) (θ := θ) (e := e) fun _ _ => rfl

example (θ₀ θ : Param S → ℝ) (B : List (Example S)) :=
  frozenLoss_comp_outPerm (τ := 1) (θ₀ := θ₀) (θ := θ) (B := B) fun _ _ _ _ => rfl

example (θ : Param S → ℝ) (B : List (Example S)) (p : Param S) :=
  sourceGrad_comp_outPerm (τ := 1) (B := B) (fun _ _ _ _ => rfl) θ p

end Transformer.Grokking.Protocol
