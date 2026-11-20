module

public import Transformer.Phasor.Attention
public import Transformer.Phasor.FFN
public import Transformer.MHC.PeriRMS.CurrentModel

/-!
# The forward pass of the main model

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`,
*Architecture*. A model has a tied embedding `E`, the LeRoPE scales `α` and `layer l` for each
layer, each with the routers and parameters of its attention and its feed-forward branch. For a
token sequence `t` and the depth scale `s` (`1/√(2L)` in the model),

    states 0 i       = four copies of R_D(E t_i),
    states (l+1) i   = ffnBlock (attnBlock (states l i)),   attnBlock, ffnBlock = PeriRMS.step,
    logits L i v     = ⟨E v, R_D(mean_s states L i s)⟩.

The attention block at the position `i` reads the states of layer `l` at the positions `j ≤ i`
(`Attention.block`, hence `LeRoPEBlock.block`); the feed-forward block reads the position alone.

* `Layer.apply`: one layer at one position, from the input states of all positions;
* `Model.states_eq_run`: **at each position the model is `PeriRMS.run`** for the sublayers
  `Attention.sublayer` and `FFN.sublayer`, the attention one carrying the heads' averages of the
  context as its raw branch. So every statement about `run` applies to the forward pass;
* `Model.feature`, `Model.logits`: the readout, `feature_eq_finalFeature` the same for
  `PeriRMS.finalFeature`.

Exact real arithmetic; the bounds are in `Transformer.Phasor.Bounds`.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS

variable {D d H F V : ℕ}

/-- The parameters of one layer: the attention with its router, the feed-forward branch with its. -/
structure Layer (D d H F : ℕ) where
  attention : Attention D d H
  attentionRouter : Router D
  ffn : FFN D F
  ffnRouter : Router D

/-- **One layer at the position `i`**, from the layer's input states `x` at all positions: the
attention block with its context, then the feed-forward block, each a `PeriRMS.step` with depth
scale `scale`. -/
noncomputable def Layer.apply (L : Layer D d H F) (eps scale : ℝ) (alpha : Fin (d / 2) → ℝ)
    (x : ℕ → State D) (i : ℕ) : State D :=
  L.attention.block L.attentionRouter eps alpha
    (fun xi heads =>
      step eps scale (L.ffn.sublayer L.ffnRouter eps)
        (step eps scale (L.attention.sublayer L.attentionRouter eps heads) xi)) x i

/-- A model: the tied embedding over a vocabulary of `V` tokens, LeRoPE's scales shared by all
layers, and the layers. -/
structure Model (D d H F V : ℕ) where
  embedding : Fin V → EucSpace D
  alpha : Fin (d / 2) → ℝ
  layer : ℕ → Layer D d H F

/-- **The states of the streams** after `l` layers at every position of the sequence `tokens`:
the normalized embedding copied into four streams, then the layers. -/
noncomputable def Model.states (M : Model D d H F V) (eps scale : ℝ) (tokens : ℕ → Fin V) :
    ℕ → ℕ → State D
  | 0 => fun i => embeddingState eps (M.embedding (tokens i))
  | l + 1 => fun i => (M.layer l).apply eps scale M.alpha (M.states eps scale tokens l) i

/-- The final feature after `L` layers: the parameter-free RMS of the mean of the four streams. -/
noncomputable def Model.feature (M : Model D d H F V) (eps scale : ℝ) (tokens : ℕ → Fin V)
    (L i : ℕ) : EucSpace D :=
  GPTMini.rmsNormEps eps (collapse (M.states eps scale tokens L i))

/-- **The logits** of the next token: the tied embedding applied to the final feature. -/
noncomputable def Model.logits (M : Model D d H F V) (eps scale : ℝ) (tokens : ℕ → Fin V)
    (L i : ℕ) (v : Fin V) : ℝ :=
  inner ℝ (M.embedding v) (M.feature eps scale tokens L i)

/-- A layer at the position `i` is the attention block with the heads' averages of the context,
then the feed-forward block. -/
theorem Layer.apply_eq (L : Layer D d H F) (eps scale : ℝ) (alpha : Fin (d / 2) → ℝ)
    (x : ℕ → State D) (i : ℕ) :
    L.apply eps scale alpha x i =
      step eps scale (L.ffn.sublayer L.ffnRouter eps)
        (step eps scale (L.attention.sublayer L.attentionRouter eps
          (L.attention.heads L.attentionRouter eps alpha x i)) (x i)) :=
  rfl

/-- **At each position the model is `PeriRMS.run`**: after `l` layers the states of position `i` are
the `l`-layer run from the normalized embedding of `t_i`, whose attention sublayers carry the heads'
averages of the context as their raw branch and whose feed-forward sublayers are `FFN.sublayer`. -/
theorem Model.states_eq_run (M : Model D d H F V) (eps scale : ℝ) (tokens : ℕ → Fin V)
    (l i : ℕ) :
    M.states eps scale tokens l i =
      run eps scale
        (fun k => (M.layer k).attention.sublayer (M.layer k).attentionRouter eps
          ((M.layer k).attention.heads (M.layer k).attentionRouter eps M.alpha
            (M.states eps scale tokens k) i))
        (fun k => (M.layer k).ffn.sublayer (M.layer k).ffnRouter eps)
        (embeddingState eps (M.embedding (tokens i))) l := by
  induction l with
  | zero => rfl
  | succ l ih => rw [run, ← ih]; rfl

/-- The final feature is `PeriRMS.finalFeature` of the position's run. -/
theorem Model.feature_eq_finalFeature (M : Model D d H F V) (eps scale : ℝ) (tokens : ℕ → Fin V)
    (L i : ℕ) :
    M.feature eps scale tokens L i =
      finalFeature eps scale
        (fun k => (M.layer k).attention.sublayer (M.layer k).attentionRouter eps
          ((M.layer k).attention.heads (M.layer k).attentionRouter eps M.alpha
            (M.states eps scale tokens k) i))
        (fun k => (M.layer k).ffn.sublayer (M.layer k).ffnRouter eps)
        (M.embedding (tokens i)) L := by
  rw [Model.feature, finalFeature, M.states_eq_run]

end Transformer.Phasor
