module

public import Transformer.Grokking.Protocol.Gradient

/-!
# Renaming the tokens is a symmetry of the decoder and of its backward pass

Source: arXiv:2201.02177v1 (Power et al.), section 1: the equations use "distinct abstract
symbols", so that "the network is not made aware of any internal structure of the elements";
openai/grok at 3d64b1d8, `grok/transformer.py`, where a token enters only as a row of the
embedding and leaves only as a row of the readout; `experiments/grok_division/PROTOCOL.md`,
*The model*.

A permutation `τ` of the tokens permutes the parameters (`paramPerm`): the rows of the
embedding and the rows of the readout, and nothing else. The decoder with the permuted
parameters reads a row of tokens as the decoder reads the renamed row, and its logit of a
token is the logit of the renamed token (`decodeTo_comp_paramPerm`, `logits_comp_paramPerm`);
the gates of the backward pass go along (`gate_comp_paramPerm`), and the cross-entropy does
not see a renaming of the classes (`crossEntropy_comp_perm`).

So on a batch whose examples `τ` leaves alone (`Example.Fixed`: no token read and no target
is moved) the loss with frozen gates is the same at the permuted parameters
(`frozenLoss_comp_paramPerm`), and the gradient of the backward pass is permuted with them
(`sourceGrad_comp_paramPerm`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy)

variable {S : Sizes} (τ : Equiv.Perm (Fin S.vocab))

/-- **The permutation of the parameters that a permutation of the tokens induces**: of the
rows of the embedding and of the readout. -/
def paramPerm : Equiv.Perm (Param S) where
  toFun
    | .emb token i => .emb (τ token) i
    | .out token i => .out (τ token) i
    | p => p
  invFun
    | .emb token i => .emb (τ.symm token) i
    | .out token i => .out (τ.symm token) i
    | p => p
  left_inv p := by cases p <;> simp
  right_inv p := by cases p <;> simp

variable {τ} {θ θ₀ : Param S → ℝ} {act : Activation S}

lemma block_comp_paramPerm (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks)
    (t : Fin S.context) : block (θ ∘ paramPerm τ) act x l t = block θ act x l t :=
  rfl

/-- **The decoder with permuted parameters reads the renamed tokens.** -/
theorem decodeTo_comp_paramPerm (tokens : Fin S.context → Fin S.vocab) (n : ℕ) :
    decodeTo (θ ∘ paramPerm τ) act tokens n = decodeTo θ act (τ ∘ tokens) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [decodeTo]
    split_ifs with hn
    · funext t
      rw [ih, block_comp_paramPerm]
    · exact ih

/-- **Its logit of a token is the decoder's logit of the renamed token.** -/
theorem logits_comp_paramPerm (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context)
    (c : Fin S.vocab) :
    logits (θ ∘ paramPerm τ) act tokens t c = logits θ act (τ ∘ tokens) t (τ c) := by
  simp only [logits, decodeTo_comp_paramPerm]
  rfl

/-- The gates of the backward pass at the permuted parameters are those of the renamed
tokens. -/
theorem gate_comp_paramPerm (tokens : Fin S.context → Fin S.vocab) :
    gate (θ ∘ paramPerm τ) tokens = gate θ (τ ∘ tokens) := by
  funext l t u v
  simp only [gate, decodeTo_comp_paramPerm]
  rfl

omit τ in
/-- **The cross-entropy does not see a renaming of the classes.** -/
theorem crossEntropy_comp_perm {V : Type*} [Fintype V] (σ : Equiv.Perm V) (z : V → ℝ) (y : V) :
    crossEntropy (z ∘ σ) y = crossEntropy z (σ y) := by
  simp only [crossEntropy, Function.comp_apply]
  rw [Equiv.sum_comp σ fun q => Real.exp (z q)]

variable (τ) in
/-- An example that the renaming `τ` leaves alone: it moves no token read and no target. -/
def Example.Fixed (e : Example S) : Prop :=
  (∀ t, τ (e.tokens t) = e.tokens t) ∧ ∀ pc ∈ e.targets, τ pc.2 = pc.2

lemma exampleLoss_comp_paramPerm {e : Example S} (he : e.Fixed τ) :
    exampleLoss (θ ∘ paramPerm τ) (gate (θ₀ ∘ paramPerm τ) e.tokens) e =
      exampleLoss θ (gate θ₀ e.tokens) e := by
  have htok : τ ∘ e.tokens = e.tokens := funext he.1
  simp only [exampleLoss, gate_comp_paramPerm, htok]
  refine congrArg List.sum (List.map_congr_left fun pc hpc => ?_)
  have hz : logits (θ ∘ paramPerm τ) (gate θ₀ e.tokens) e.tokens pc.1 =
      logits θ (gate θ₀ e.tokens) e.tokens pc.1 ∘ τ := by
    funext c
    rw [logits_comp_paramPerm, htok]
    rfl
  rw [hz, crossEntropy_comp_perm, he.2 pc hpc]

/-- **On a batch that the renaming leaves alone the frozen loss is the same at the permuted
parameters.** -/
theorem frozenLoss_comp_paramPerm {B : List (Example S)} (hB : ∀ e ∈ B, e.Fixed τ) :
    frozenLoss (θ₀ ∘ paramPerm τ) (θ ∘ paramPerm τ) B = frozenLoss θ₀ θ B := by
  simp only [frozenLoss]
  rw [List.map_congr_left fun e he => exampleLoss_comp_paramPerm (hB e he)]

/-- **The gradient of the backward pass is permuted with the parameters**, on a batch that the
renaming leaves alone. -/
theorem sourceGrad_comp_paramPerm {B : List (Example S)} (hB : ∀ e ∈ B, e.Fixed τ)
    (θ : Param S → ℝ) (p : Param S) :
    sourceGrad B (θ ∘ paramPerm τ) p = sourceGrad B θ (paramPerm τ p) := by
  have hupd : ∀ s : ℝ, Function.update (θ ∘ paramPerm τ) p s =
      Function.update θ (paramPerm τ p) s ∘ paramPerm τ := fun s => by
    rw [Function.update_comp_equiv, Equiv.symm_apply_apply]
  simp only [sourceGrad, hupd, frozenLoss_comp_paramPerm hB, Function.comp_apply]

/-! The witnesses: the identity renaming leaves every example alone. `Exchange` has a renaming
that moves two tokens and the examples it leaves alone. -/

example (θ₀ θ : Param S → ℝ) (e : Example S) :=
  exampleLoss_comp_paramPerm (τ := 1) (θ₀ := θ₀) (θ := θ) (e := e) ⟨fun _ => rfl, fun _ _ => rfl⟩

example (θ₀ θ : Param S → ℝ) (B : List (Example S)) :=
  frozenLoss_comp_paramPerm (τ := 1) (θ₀ := θ₀) (θ := θ) (B := B)
    fun _ _ => ⟨fun _ => rfl, fun _ _ => rfl⟩

example (θ : Param S → ℝ) (B : List (Example S)) (p : Param S) :=
  sourceGrad_comp_paramPerm (τ := 1) (B := B) (fun _ _ => ⟨fun _ => rfl, fun _ _ => rfl⟩) θ p

end Transformer.Grokking.Protocol
