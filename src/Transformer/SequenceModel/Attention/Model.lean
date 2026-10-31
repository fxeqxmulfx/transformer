module

public import Transformer.SequenceModel.Attention.Basic

/-!
# The supplied diff_gpt forward, with normalization kept in its actual places

Source: `review/flat_spectrum.md`, *General positional models*, G4, and
`diff_gpt/domain/model/{gpt,block,mha,feed_forward,rope}.py` at `b98c665b`.
This is the full no-cache training forward in real arithmetic. RMS epsilon
is the dtype epsilon (2^-23 for float32); XSA's clipped L2 epsilon is separate
(1e-12 in the source). The explicit zero at the first position is retained.
KV-cache mechanics and floating-point rounding are outside this model.
The mathematical forward accepts a supplied phase table of any finite length;
correspondence with source execution requires positions within block_size.

Arbitrary angle tables include the source RoPE grid and per-layer disabling
via zero angles. Head layout uses the same fused-QKV split and half-split
rotations. No old L2 QK normalizer, learned temperature or ReLU FFN is used.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini
open scoped BigOperators

/-- Source head logits: RMS-normalize Q/K, rotate, then divide by sqrt(head_dim). -/
def headScore (d : ℕ) (eps : ℝ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) : ℝ :=
  inner (𝕜 := ℝ) (rotateHead d a (rmsNormEps eps q))
    (rotateHead d b (rmsNormEps eps k)) / Real.sqrt (d : ℝ)

/-- Full causal softmax, including the token itself before XSA. -/
def causalWeight {T : ℕ} (scores : Fin T → Fin T → ℝ) (i j : Fin T) : ℝ :=
  if j ≤ i then Real.exp (scores i j) /
    ∑ k ∈ Finset.Iic i, Real.exp (scores i k) else 0

/-- Actual source XSA, with clipped own-value normalization and explicit zero
for the first token. The unnormalized values remain in the attention sum. -/
def headOutput {T d : ℕ} (eps xeps : ℝ) (angle : Fin T → Fin (d / 2) → ℝ)
    (q k v : Fin T → EucSpace d) (i : Fin T) : EucSpace d :=
  let scores := fun a b => headScore d eps (angle a) (angle b) (q a) (k b)
  let y := ∑ j, causalWeight scores i j • v j
  let u := normL2 xeps (v i)
  if i.val = 0 then 0 else y - (inner (𝕜 := ℝ) y u) • u

/-- Fused QKV, actual head reshape, normalized rotary attention, XSA and output projection. -/
def attention (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) : EucSpace cfg.d_model :=
  let qkv := fun j => B.qkv (x j)
  let q := fun h j => headSlice cfg (qkvSlice cfg (qkvQ cfg) (qkv j)) h
  let k := fun h j => headSlice cfg (qkvSlice cfg (qkvK cfg) (qkv j)) h
  let v := fun h j => headSlice cfg (qkvSlice cfg (qkvV cfg) (qkv j)) h
  B.projection (headMerge cfg (fun h => headOutput eps xeps angle (q h) (k h) (v h) i))

/-- Genuine SiLU-gated feed-forward; the source's combined c_fc stores these
gate and value matrices side by side, with no bias. -/
def feedForward (cfg : Config) (B : BlockParams cfg) (x : EucSpace cfg.d_model) :
    EucSpace cfg.d_model :=
  B.output (WithLp.toLp 2 (fun h => Landscape.Neural.swiglu (B.gate x h) (B.value x h)))

/-- Normalize both inputs and outputs of both branches; add them sequentially. -/
def block (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) : Fin T → EucSpace cfg.d_model :=
  let x1 := fun i => x i + rmsNormEps eps
    (attention cfg B eps xeps angle (fun j => rmsNormEps eps (x j)) i)
  fun i => x1 i + rmsNormEps eps (feedForward cfg B (rmsNormEps eps (x1 i)))

/-- Actual normalized embedding and finite residual stack. -/
def hidden (cfg : Config) (p : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) : ℕ → Fin T → EucSpace cfg.d_model
  | 0 => fun i => rmsNormEps eps (p.embedding (tokens i))
  | n + 1 => if h : n < cfg.n_layers then
      block cfg (p.blocks ⟨n, h⟩) eps xeps (angle ⟨n, h⟩) (hidden cfg p eps xeps angle tokens n)
    else hidden cfg p eps xeps angle tokens n

/-- The final normalizer and tied readout of every token, in full parameter coordinates. -/
def score (cfg : Config) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (readout : Fin T) (w : Parameters cfg) (tokens : Fin T → Fin cfg.vocab_size)
    (v : Fin cfg.vocab_size) : ℝ :=
  let p := decode cfg w
  inner (𝕜 := ℝ) (rmsNormEps eps (hidden cfg p eps xeps angle tokens cfg.n_layers readout))
    (p.embedding v)

/-- Source RoPE, with any positive configured base and independent enabled
flags per layer. Other arbitrary phase tables can be supplied directly. -/
def sourceAngles (cfg : Config) {T : ℕ} (enabled : Fin cfg.n_layers → Bool)
    (positions : Fin T → ℝ) (layer : Fin cfg.n_layers) (i : Fin T)
    (pair : Fin (cfg.head_dim / 2)) : ℝ :=
  if enabled layer then positions i * invFreq cfg.head_dim cfg.rope_theta pair else 0

end Transformer.SequenceModel.Attention
