module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Order.Interval.Finset.Fin
public import Mathlib.Tactic

/-!
# The decoder of the author's code: parameters and forward pass

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, "a
standard decoder-only transformer with causal attention masking"; openai/grok at 3d64b1d8,
`grok/transformer.py`: `Transformer`, `DecoderBlock`, `MultiHeadAttention`, `AttentionHead`,
`FFN`, `_position_encoding`; `experiments/grok_division/PROTOCOL.md`, *The model*.

The scalar parameters are indexed by one type (`Param`): the embedding, each block's query,
key and value maps of every head, its map after the heads are concatenated, the gains and
biases of its two normalizations and its two feed-forward maps, and the readout. No linear
map has a bias and the readout is not tied to the embedding. The optimizer acts on a function
`Param S → ℝ`, so every group is one coordinate family of the same run.

The forward pass is in exact reals. A block is post-norm: `a = LayerNorm(x + Attention(x))`
(`attended`), then `LayerNorm(a + W₂ act(W₁ a))` (`block`). A head's scores are
`q · k / √(head width)`; a position weighs the positions up to its own by their softmax
(`weight`, which sums to `1`: `sum_weight`). `LayerNorm` adds `10⁻⁵` to the variance
(`layerNorm`), so its root is positive (`layerNorm_root_pos`). The positions are the sinusoids
of `position`. The logits (`logits`) are the readout of the last block's output.

The activation of a hidden unit is an argument (`Activation`): the model's is `max 0` (`relu`);
`Gradient` freezes the gates of the units to state what the source's backward pass returns.

The mask makes the pass causal (`Causal`). The source computes all this in `f32`; here it is
the function of real parameters that those floats approximate.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- The sizes of a decoder. The source has the width equal to heads times head width, and the
hidden width four times the width. -/
structure Sizes where
  /-- Tokens. -/
  vocab : ℕ
  /-- Width of the residual stream. -/
  width : ℕ
  /-- Attention heads of a block. -/
  heads : ℕ
  /-- Width of a head. -/
  headWidth : ℕ
  /-- Hidden width of a block's feed-forward network. -/
  hidden : ℕ
  /-- Blocks. -/
  blocks : ℕ
  /-- Tokens read. -/
  context : ℕ

/-- **The scalar parameters** of a decoder of sizes `S`. -/
inductive Param (S : Sizes)
  | emb (token : Fin S.vocab) (i : Fin S.width)
  | query (l : Fin S.blocks) (h : Fin S.heads) (r : Fin S.headWidth) (i : Fin S.width)
  | key (l : Fin S.blocks) (h : Fin S.heads) (r : Fin S.headWidth) (i : Fin S.width)
  | value (l : Fin S.blocks) (h : Fin S.heads) (r : Fin S.headWidth) (i : Fin S.width)
  | proj (l : Fin S.blocks) (o : Fin S.width) (h : Fin S.heads) (r : Fin S.headWidth)
  | gain1 (l : Fin S.blocks) (i : Fin S.width)
  | bias1 (l : Fin S.blocks) (i : Fin S.width)
  | up (l : Fin S.blocks) (u : Fin S.hidden) (i : Fin S.width)
  | down (l : Fin S.blocks) (o : Fin S.width) (u : Fin S.hidden)
  | gain2 (l : Fin S.blocks) (i : Fin S.width)
  | bias2 (l : Fin S.blocks) (i : Fin S.width)
  | out (token : Fin S.vocab) (i : Fin S.width)
  deriving DecidableEq, Fintype

/-- The activations of the hidden units: one for each block, position and unit. -/
abbrev Activation (S : Sizes) := Fin S.blocks → Fin S.context → Fin S.hidden → ℝ → ℝ

/-- The source's activation, `max 0`, at every unit. -/
def relu (S : Sizes) : Activation S := fun _ _ _ v => max 0 v

variable {S : Sizes} (θ : Param S → ℝ)

/-- The positional encoding of `Transformer._position_encoding`. -/
noncomputable def position (S : Sizes) (pos : ℕ) (i : Fin S.width) : ℝ :=
  if (i : ℕ) % 2 = 0 then Real.sin (pos / (10000 : ℝ) ^ ((i : ℕ) / (S.width : ℝ)))
  else Real.cos (pos / (10000 : ℝ) ^ (((i : ℕ) - 1 : ℕ) / (S.width : ℝ)))

/-- Layer normalization with gain `g` and bias `b`: the variance is that of the entries, with
`10⁻⁵` added under the root. -/
noncomputable def layerNorm {n : ℕ} (g b z : Fin n → ℝ) (i : Fin n) : ℝ :=
  (z i - (∑ j, z j) / n) / √((∑ j, (z j - (∑ j', z j') / n) ^ 2) / n + 1 / 10 ^ 5) * g i + b i

/-- **Layer normalization divides by a positive number.** -/
lemma layerNorm_root_pos {n : ℕ} (z : Fin n → ℝ) :
    0 < √((∑ j, (z j - (∑ j', z j') / n) ^ 2) / n + 1 / 10 ^ 5) :=
  Real.sqrt_pos.2 (by positivity)

/-- The score of head `h` of block `l` for the query position `t` and the position `j`. -/
noncomputable def score (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks) (h : Fin S.heads)
    (t j : Fin S.context) : ℝ :=
  (∑ r, (∑ i, θ (.query l h r i) * x t i) * (∑ i, θ (.key l h r i) * x j i)) / √S.headWidth

/-- **The attention weight** of the position `t` on the position `j`: the softmax of the
scores over the positions up to `t`, and `0` after it. -/
noncomputable def weight (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks) (h : Fin S.heads)
    (t j : Fin S.context) : ℝ :=
  if j ≤ t then Real.exp (score θ x l h t j) / ∑ j' ∈ Finset.Iic t, Real.exp (score θ x l h t j')
  else 0

/-- The output of the attention of block `l` at the position `t`. -/
noncomputable def attention (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks)
    (t : Fin S.context) (o : Fin S.width) : ℝ :=
  ∑ h, ∑ r, θ (.proj l o h r) *
    ∑ j ∈ Finset.Iic t, weight θ x l h t j * ∑ i, θ (.value l h r i) * x j i

/-- The first half of block `l` at the position `t`: the input plus its attention,
normalized. -/
noncomputable def attended (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks)
    (t : Fin S.context) : Fin S.width → ℝ :=
  layerNorm (fun i => θ (.gain1 l i)) (fun i => θ (.bias1 l i))
    fun i => x t i + attention θ x l t i

/-- The input of the hidden unit `u` of block `l` at the position `t`. -/
noncomputable def preactivation (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks)
    (t : Fin S.context) (u : Fin S.hidden) : ℝ :=
  ∑ i, θ (.up l u i) * attended θ x l t i

/-- **A decoder block** at the position `t`, with the activations `act`. -/
noncomputable def block (act : Activation S) (x : Fin S.context → Fin S.width → ℝ)
    (l : Fin S.blocks) (t : Fin S.context) : Fin S.width → ℝ :=
  layerNorm (fun i => θ (.gain2 l i)) (fun i => θ (.bias2 l i)) fun o =>
    attended θ x l t o + ∑ u, θ (.down l o u) * act l t u (preactivation θ x l t u)

/-- The outputs of the first `n` blocks for the tokens read; the embedding for `n = 0`. -/
noncomputable def decodeTo (act : Activation S) (tokens : Fin S.context → Fin S.vocab) :
    ℕ → Fin S.context → Fin S.width → ℝ
  | 0 => fun t i => θ (.emb (tokens t) i) + position S t i
  | n + 1 => if h : n < S.blocks then fun t => block θ act (decodeTo act tokens n) ⟨n, h⟩ t
      else decodeTo act tokens n

/-- **The logits** at the position `t`: the readout of the last block's output. -/
noncomputable def logits (act : Activation S) (tokens : Fin S.context → Fin S.vocab)
    (t : Fin S.context) (c : Fin S.vocab) : ℝ :=
  ∑ i, θ (.out c i) * decodeTo θ act tokens S.blocks t i

/-- **The attention weights of a position sum to `1`.** -/
theorem sum_weight (θ : Param S → ℝ) (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks)
    (h : Fin S.heads) (t : Fin S.context) : ∑ j ∈ Finset.Iic t, weight θ x l h t j = 1 := by
  have hpos : 0 < ∑ j' ∈ Finset.Iic t, Real.exp (score θ x l h t j') :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨t, Finset.mem_Iic.2 le_rfl⟩
  rw [Finset.sum_congr rfl fun j hj => by rw [weight, ite_eq_left (Finset.mem_Iic.1 hj)],
    ← Finset.sum_div, div_self hpos.ne']

end Transformer.Grokking.Protocol
