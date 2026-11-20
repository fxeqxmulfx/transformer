module

public import Transformer.Phasor.Forward
public import Transformer.Phasor.PlainRouter

/-!
# The ordinary residual is a sub-model of the main model's mHC

R0 of `review/residual_connections.md`, *The ordinary residual inside mHC*. With plain routers
(`Router.IsPlain`: zero scales and phases, read biases `−ln 3`, write biases `0`) every block keeps
four equal streams equal (`step_const_of_isPlain`). So the forward pass of the main model is,
exactly and with no `ε` caveat, the one-stream model with the same embedding, heads and
feed-forward branches (`OneModel`):

    h⁰_i = R_D(E t_i),   h ← h + α R_D(raw(R_D h)) per sublayer,   logits ⟨E v, R_D(h)⟩,

the residual of DiffGPT's former control (label `main`), of diff_gpt and of Rotate90. The
attention of the one-stream model reads `R_D(h_j)` where the main model reads `Router.read`.

* `OneModel.states_withRouters`, `OneModel.logits_withRouters`: the main model with plain routers
  has four copies of the one-stream state at every layer and position, and the same logits;
* `range_oneLogits_subset`: **the logit maps of one-stream models are logit maps of main models**
  of the same widths, heads and vocabulary. So a construction for the ordinary residual is a
  construction for mHC, and a lower bound for mHC holds for the ordinary residual.

Exact real arithmetic, every `ε` and every depth scale `α` (the model's is `1/√(2L)`).
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual
  Transformer.ComplexFFN

variable {D d H F V : ℕ}

/-- One layer of the one-stream model: the attention and the feed-forward branch of a `Layer`,
without routers. -/
structure OneLayer (D d H F : ℕ) where
  attention : Attention D d H
  ffn : FFN D F

/-- The one-stream model: the tied embedding, LeRoPE's scales and the layers. -/
structure OneModel (D d H F V : ℕ) where
  embedding : Fin V → EucSpace D
  alpha : Fin (d / 2) → ℝ
  layer : ℕ → OneLayer D d H F

/-- The logit of head `h` on one stream: `Attention.logit` with `R_D` in place of the router's
read. -/
noncomputable def Attention.oneLogit (A : Attention D d H) (eps : ℝ) (alpha : Fin (d / 2) → ℝ)
    (h : Fin H) : ℕ → EucSpace D → EucSpace D → ℕ → ℝ :=
  LeRoPEBlock.leropeLogit (leropeFrequencies d alpha) (phaseCoefficient (A.coefficient h)) d
    (fun x => headVector eps (A.query h) (GPTMini.rmsNormEps eps x))
    (fun x => headVector eps (A.key h) (GPTMini.rmsNormEps eps x))

/-- The heads' softmax averages over one stream at the position `i`, the values `W_V^h R_D(h_j)`. -/
noncomputable def Attention.oneHeads (A : Attention D d H) (eps : ℝ) (alpha : Fin (d / 2) → ℝ)
    (x : ℕ → EucSpace D) (i : ℕ) : Fin H → EucSpace d :=
  LeRoPEBlock.block (fun _ heads => heads) (A.oneLogit eps alpha)
    (fun h y => A.value h (GPTMini.rmsNormEps eps y)) x i

/-- On four equal streams the heads of an attention with a plain router are the one-stream
heads. -/
theorem Attention.heads_const {R : Router D} (hR : R.IsPlain) (A : Attention D d H) (eps : ℝ)
    (alpha : Fin (d / 2) → ℝ) (x : ℕ → EucSpace D) (i : ℕ) :
    A.heads R eps alpha (fun j _ => x j) i = A.oneHeads eps alpha x i := by
  simp only [Attention.heads, Attention.block, Attention.oneHeads, LeRoPEBlock.block,
    Attention.logit, Attention.oneLogit, LeRoPEBlock.leropeLogit, hR.read_const]

/-- **The ordinary residual**: `h + α R_D(raw(R_D h))`. -/
noncomputable def oneStep (eps scale : ℝ) (raw : EucSpace D → EucSpace D) (h : EucSpace D) :
    EucSpace D :=
  h + scale • GPTMini.rmsNormEps eps (raw (GPTMini.rmsNormEps eps h))

/-- One layer of the one-stream model at the position `i`: the attention residual, then the
feed-forward residual. -/
noncomputable def OneLayer.apply (L : OneLayer D d H F) (eps scale : ℝ)
    (alpha : Fin (d / 2) → ℝ) (x : ℕ → EucSpace D) (i : ℕ) : EucSpace D :=
  oneStep eps scale (plainFFN L.ffn.gate L.ffn.value L.ffn.outRe L.ffn.outIm)
    (oneStep eps scale (fun _ => L.attention.output (L.attention.oneHeads eps alpha x i)) (x i))

/-- The states of the one-stream model: the normalized embedding, then the layers. -/
noncomputable def OneModel.states (M : OneModel D d H F V) (eps scale : ℝ) (tokens : ℕ → Fin V) :
    ℕ → ℕ → EucSpace D
  | 0 => fun i => GPTMini.rmsNormEps eps (M.embedding (tokens i))
  | l + 1 => fun i => (M.layer l).apply eps scale M.alpha (M.states eps scale tokens l) i

/-- The logits of the one-stream model: `⟨E v, R_D(h)⟩`. -/
noncomputable def OneModel.logits (M : OneModel D d H F V) (eps scale : ℝ) (tokens : ℕ → Fin V)
    (L i : ℕ) (v : Fin V) : ℝ :=
  inner ℝ (M.embedding v) (GPTMini.rmsNormEps eps (M.states eps scale tokens L i))

/-- The main model with the parameters of a one-stream model and the routers `attnR l`, `ffnR l`
of layer `l`. -/
noncomputable def OneModel.withRouters (M : OneModel D d H F V) (attnR ffnR : ℕ → Router D) :
    Model D d H F V where
  embedding := M.embedding
  alpha := M.alpha
  layer l := ⟨(M.layer l).attention, attnR l, (M.layer l).ffn, ffnR l⟩

/-- **R0, the states**: with plain routers the main model's four streams are, at every layer and
position, four copies of the one-stream state. -/
theorem OneModel.states_withRouters (M : OneModel D d H F V) {attnR ffnR : ℕ → Router D}
    (hA : ∀ l, (attnR l).IsPlain) (hF : ∀ l, (ffnR l).IsPlain) (eps scale : ℝ)
    (tokens : ℕ → Fin V) (l : ℕ) :
    (M.withRouters attnR ffnR).states eps scale tokens l =
      fun i _ => M.states eps scale tokens l i := by
  induction l with
  | zero => rfl
  | succ l ih =>
    funext i
    change Layer.apply _ eps scale M.alpha ((M.withRouters attnR ffnR).states eps scale tokens l) i
      = _
    rw [Layer.apply_eq, ih]
    change step eps scale ⟨(ffnR l).routing eps, _⟩
      (step eps scale ⟨(attnR l).routing eps, _⟩ (fun _ => M.states eps scale tokens l i)) = _
    rw [step_const_of_isPlain (hA l), step_const_of_isPlain (hF l),
      Attention.heads_const (R := ((M.withRouters attnR ffnR).layer l).attentionRouter) (hA l)]
    rfl

/-- **R0, the logits**: with plain routers the main model's logits are the one-stream model's. -/
theorem OneModel.logits_withRouters (M : OneModel D d H F V) {attnR ffnR : ℕ → Router D}
    (hA : ∀ l, (attnR l).IsPlain) (hF : ∀ l, (ffnR l).IsPlain) (eps scale : ℝ)
    (tokens : ℕ → Fin V) (L i : ℕ) (v : Fin V) :
    (M.withRouters attnR ffnR).logits eps scale tokens L i v = M.logits eps scale tokens L i v := by
  rw [Model.logits, Model.feature, M.states_withRouters hA hF, collapse, weightedSum_const]
  norm_num
  rfl

/-- **R0: the ordinary residual is a sub-model of mHC.** The logit map of every one-stream model is
the logit map of a main model with the same widths, heads and vocabulary, for every `ε` and depth
scale. -/
theorem range_oneLogits_subset (eps scale : ℝ) :
    Set.range (fun M : OneModel D d H F V => M.logits eps scale) ⊆
      Set.range (fun M : Model D d H F V => M.logits eps scale) := by
  rintro _ ⟨M, rfl⟩
  refine ⟨M.withRouters (fun _ => Router.plain D) (fun _ => Router.plain D), ?_⟩
  funext tokens L i v
  exact M.logits_withRouters (fun _ => Router.plain_isPlain D) (fun _ => Router.plain_isPlain D)
    eps scale tokens L i v

/-- The model's sizes, `D = 128`, `d = 32`, `H = 4`, `ε = 2⁻²³`: plain routers give the one-stream
logits. -/
example (M : OneModel 128 32 4 128 97) (tokens : ℕ → Fin 97) (L i : ℕ) (v : Fin 97) :
    (M.withRouters (fun _ => Router.plain 128) (fun _ => Router.plain 128)).logits (1 / 8388608)
        (1 / Real.sqrt 8) tokens L i v =
      M.logits (1 / 8388608) (1 / Real.sqrt 8) tokens L i v :=
  M.logits_withRouters (fun _ => Router.plain_isPlain 128) (fun _ => Router.plain_isPlain 128)
    _ _ tokens L i v

/-- The model's sizes and plain routers: four copies of the one-stream state at every layer. -/
example (M : OneModel 128 32 4 128 97) (tokens : ℕ → Fin 97) (l : ℕ) :
    (M.withRouters (fun _ => Router.plain 128) (fun _ => Router.plain 128)).states (1 / 8388608)
        (1 / Real.sqrt 8) tokens l = fun i _ => M.states (1 / 8388608) (1 / Real.sqrt 8) tokens l i :=
  M.states_withRouters (fun _ => Router.plain_isPlain 128) (fun _ => Router.plain_isPlain 128)
    _ _ tokens l

/-- The heads of a plain router on equal streams, at the model's sizes. -/
example (A : Attention 128 32 4) (alpha : Fin (32 / 2) → ℝ) (x : ℕ → EucSpace 128) (i : ℕ) :
    A.heads (Router.plain 128) (1 / 8388608) alpha (fun j _ => x j) i =
      A.oneHeads (1 / 8388608) alpha x i :=
  A.heads_const (Router.plain_isPlain 128) _ alpha x i

end Transformer.Phasor
