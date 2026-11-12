module

public import Transformer.Grokking.Protocol.Causal
public import Transformer.Grokking.Protocol.Gradient

/-!
# The decoder reads the embedding in the rows of its tokens only

Source: openai/grok at 3d64b1d8, `grok/transformer.py`: `Transformer.embed` looks up the rows
of the tokens read and nothing else reads the embedding; the readout is another matrix.
`experiments/grok_division/PROTOCOL.md`, *The model*. arXiv:2201.02177v1 (Power et al.),
appendix *Additional experimental details*, states no such fact.

The parameters of the embedding are `Param.emb` (`Param.IsEmb`). A block reads none of them
(`block_congr_param`), and the first layer reads those of the tokens in its row. So two
parameter vectors that agree outside the embedding give the same outputs of every block, and
the same logits, on two rows of tokens whose embedded rows agree (`decodeTo_congr_param`,
`logits_congr_param`). Two consequences:

* a parameter vector with the same embedding for two tokens does not tell them apart
  (`logits_eq_of_emb_eq`);
* the loss of a batch, with whatever gates, does not depend on the embedding of a token that
  no example of the batch reads, so the backward pass returns `0` in every coordinate of that
  row (`sourceGrad_emb_eq_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {S : Sizes} {θ θ' : Param S → ℝ} {act : Activation S}

/-- The parameters of the embedding. -/
def Param.IsEmb : Param S → Prop
  | .emb _ _ => True
  | _ => False

/-- **A block reads no parameter of the embedding.** -/
lemma block_congr_param (h : ∀ p : Param S, ¬p.IsEmb → θ p = θ' p)
    (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks) (t : Fin S.context) :
    block θ act x l t = block θ' act x l t := by
  have hq : ∀ l h r i, θ (.query l h r i) = θ' (.query l h r i) := fun _ _ _ _ => h _ not_false
  have hk : ∀ l h r i, θ (.key l h r i) = θ' (.key l h r i) := fun _ _ _ _ => h _ not_false
  have hv : ∀ l h r i, θ (.value l h r i) = θ' (.value l h r i) := fun _ _ _ _ => h _ not_false
  have hp : ∀ l o h r, θ (.proj l o h r) = θ' (.proj l o h r) := fun _ _ _ _ => h _ not_false
  have hg1 : ∀ l i, θ (.gain1 l i) = θ' (.gain1 l i) := fun _ _ => h _ not_false
  have hb1 : ∀ l i, θ (.bias1 l i) = θ' (.bias1 l i) := fun _ _ => h _ not_false
  have hu : ∀ l u i, θ (.up l u i) = θ' (.up l u i) := fun _ _ _ => h _ not_false
  have hd : ∀ l o u, θ (.down l o u) = θ' (.down l o u) := fun _ _ _ => h _ not_false
  have hg2 : ∀ l i, θ (.gain2 l i) = θ' (.gain2 l i) := fun _ _ => h _ not_false
  have hb2 : ∀ l i, θ (.bias2 l i) = θ' (.bias2 l i) := fun _ _ => h _ not_false
  simp only [block, attended, attention, weight, score, preactivation, hq, hk, hv, hp, hg1, hb1,
    hu, hd, hg2, hb2]

/-- **The decoder reads the embedding in the rows of its tokens only**: parameters that agree
outside the embedding give the same outputs of every block on rows of tokens whose embedded
rows agree. -/
theorem decodeTo_congr_param {tokens tokens' : Fin S.context → Fin S.vocab}
    (h : ∀ p : Param S, ¬p.IsEmb → θ p = θ' p)
    (hemb : ∀ t i, θ (.emb (tokens t) i) = θ' (.emb (tokens' t) i)) (n : ℕ) :
    decodeTo θ act tokens n = decodeTo θ' act tokens' n := by
  induction n with
  | zero =>
    funext t i
    simp only [decodeTo, hemb t i]
  | succ n ih =>
    simp only [decodeTo]
    split_ifs with hn
    · funext t
      rw [ih, block_congr_param h]
    · exact ih

/-- **The logits are the same** for parameters that agree outside the embedding, on rows of
tokens whose embedded rows agree. -/
theorem logits_congr_param {tokens tokens' : Fin S.context → Fin S.vocab}
    (h : ∀ p : Param S, ¬p.IsEmb → θ p = θ' p)
    (hemb : ∀ t i, θ (.emb (tokens t) i) = θ' (.emb (tokens' t) i)) :
    logits θ act tokens = logits θ' act tokens' := by
  funext t c
  simp only [logits, decodeTo_congr_param h hemb, h (.out c _) not_false]

/-- **Tokens with the same embedding are not told apart.** -/
theorem logits_eq_of_emb_eq {tokens tokens' : Fin S.context → Fin S.vocab}
    (hemb : ∀ t i, θ (.emb (tokens t) i) = θ (.emb (tokens' t) i)) :
    logits θ act tokens = logits θ act tokens' :=
  logits_congr_param (fun _ _ => rfl) hemb

/-- A change of one parameter of the embedding leaves the others as they were. -/
lemma update_emb_of_not_isEmb (θ : Param S → ℝ) (τ : Fin S.vocab) (i : Fin S.width) (s : ℝ)
    (p : Param S) (hp : ¬p.IsEmb) : Function.update θ (.emb τ i) s p = θ p :=
  Function.update_of_ne (fun hpe => hp (by rw [hpe]; exact trivial)) _ _

/-- **The backward pass returns `0` in the embedding of a token that the batch does not
read.** -/
theorem sourceGrad_emb_eq_zero {B : List (Example S)} {τ : Fin S.vocab}
    (hB : ∀ e ∈ B, ∀ t, e.tokens t ≠ τ) (θ : Param S → ℝ) (i : Fin S.width) :
    sourceGrad B θ (.emb τ i) = 0 := by
  have hconst : ∀ s : ℝ, frozenLoss θ (Function.update θ (.emb τ i) s) B = frozenLoss θ θ B := by
    intro s
    simp only [frozenLoss]
    rw [List.map_congr_left fun e he => ?_]
    simp only [exampleLoss]
    rw [logits_congr_param (θ := Function.update θ (.emb τ i) s) (θ' := θ)
      (tokens := e.tokens) (tokens' := e.tokens) (update_emb_of_not_isEmb θ τ i s)
      (fun t j => Function.update_of_ne (fun hpe => hB e he t (Param.emb.inj hpe).1) _ _)]
  simp only [sourceGrad, hconst]
  exact deriv_const _ _

/-! The witnesses, in the decoder `tinySizes` with two tokens: parameters changed in the
embedding only; the constant parameters `1`, which give both tokens the same embedding; and
the batch of one example that reads the token `0` twice, in which the embedding of the token
`1` has the gradient `0`. -/

example (θ : Param tinySizes → ℝ) (x : Fin 2 → Fin 1 → ℝ) :
    block (Function.update θ (.emb (0 : Fin 2) (0 : Fin 1)) 5) (relu tinySizes) x (0 : Fin 1)
        (0 : Fin 2) = block θ (relu tinySizes) x (0 : Fin 1) (0 : Fin 2) :=
  block_congr_param (update_emb_of_not_isEmb θ _ _ _) x _ _

example (θ : Param tinySizes → ℝ) :
    decodeTo (Function.update θ (.emb (1 : Fin 2) (0 : Fin 1)) 5) (relu tinySizes)
        (![0, 0] : Fin 2 → Fin 2) 1 = decodeTo θ (relu tinySizes) (![0, 0] : Fin 2 → Fin 2) 1 :=
  decodeTo_congr_param (update_emb_of_not_isEmb θ _ _ _) (fun t i =>
    Function.update_of_ne (fun hpe => by
      have h := (Param.emb.inj hpe).1
      fin_cases t <;> simp at h) _ _) 1

example (θ : Param tinySizes → ℝ) (c : Fin 2) :
    Function.update θ (.emb (1 : Fin 2) (0 : Fin 1)) 5 (.out c (0 : Fin 1)) = θ (.out c 0) :=
  update_emb_of_not_isEmb θ _ _ _ _ not_false

example (θ : Param tinySizes → ℝ) :
    logits (Function.update θ (.emb (1 : Fin 2) (0 : Fin 1)) 5) (relu tinySizes)
        (![0, 0] : Fin 2 → Fin 2) = logits θ (relu tinySizes) (![0, 0] : Fin 2 → Fin 2) :=
  logits_congr_param (update_emb_of_not_isEmb θ _ _ _) fun t i =>
    Function.update_of_ne (fun hpe => by
      have h := (Param.emb.inj hpe).1
      fin_cases t <;> simp at h) _ _

example : logits (fun _ : Param tinySizes => (1 : ℝ)) (relu tinySizes)
      (![0, 0] : Fin 2 → Fin 2) =
    logits (fun _ : Param tinySizes => (1 : ℝ)) (relu tinySizes) (![0, 1] : Fin 2 → Fin 2) :=
  logits_eq_of_emb_eq fun _ _ => rfl

example (θ : Param tinySizes → ℝ) :
    sourceGrad [(⟨(![0, 0] : Fin 2 → Fin 2), [((1 : Fin 2), (0 : Fin 2))]⟩ : Example tinySizes)]
      θ (.emb (1 : Fin 2) (0 : Fin 1)) = 0 :=
  sourceGrad_emb_eq_zero (fun e he t => by
    rw [List.mem_singleton.1 he]
    fin_cases t <;> simp) θ 0

end Transformer.Grokking.Protocol
