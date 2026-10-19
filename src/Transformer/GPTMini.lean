module

public import Transformer.GPTMini.Config
public import Transformer.GPTMini.RMSNorm
public import Transformer.GPTMini.Reshape
public import Transformer.GPTMini.ReshapeDist
public import Transformer.GPTMini.RoPE
public import Transformer.GPTMini.QKNorm
public import Transformer.GPTMini.QKNormLipschitz
public import Transformer.GPTMini.CausalMHA
public import Transformer.GPTMini.AttentionBounds
public import Transformer.GPTMini.TotalVariation
public import Transformer.GPTMini.SoftmaxStability
public import Transformer.GPTMini.AttentionLipschitz
public import Transformer.GPTMini.HeadLipschitz
public import Transformer.GPTMini.AttnSubLayerLipschitz
public import Transformer.GPTMini.ReLU2FFN
public import Transformer.GPTMini.Block
public import Transformer.GPTMini.BlockLipschitz
public import Transformer.GPTMini.Model
public import Transformer.GPTMini.Causality
public import Transformer.GPTMini.TokenInterface
public import Transformer.GPTMini.Semantics
public import Transformer.GPTMini.Properties
public import Transformer.GPTMini.Bridge
public import Transformer.GPTMini.ClusteringTheorem
public import Transformer.GPTMini.TwoTokens
public import Transformer.GPTMini.RateRefutation
public import Transformer.GPTMini.MeanFieldRefutation
public import Transformer.GPTMini.Convex

/-!
# `gpt-mini` Formalization

Top-level module re-exporting the formalization of the `gpt-mini`
transformer architecture defined in `reference/model.py`.

The source is restored byte for byte at
`experiments/archive/gpt_mini/gpt_mini.py`; every citation of
`reference/model.py` in `Transformer.GPTMini` refers to its last version,
commit `f11b6e2`:

  `git show f11b6e27d3cfe6813a2876bdeec38565fc54258c:reference/model.py`

The architecture is:
  - Pre-LN with RMSNorm (no trainable scale)
  - Multi-head causal attention with QK-norm and learnable per-head α
  - XSA: attention output is projected onto `v_i^⊥`
  - Plain RoPE positional encoding
  - ReLU² feed-forward
  - Tied embedding / unembedding
  - 12 unique blocks, no recurrence

| Module | Contents |
| --- | --- |
| `GPTMini.Config` | the hyperparameters, and the grouped-query head split |
| `GPTMini.RMSNorm` | the block normalization, and the sphere it maps onto |
| `GPTMini.Reshape` | the coordinate layout between `d_model` and the heads |
| `GPTMini.ReshapeDist` | and what it does to the distance between two streams |
| `GPTMini.RoPE` | the rotary tables, and that they rotate |
| `GPTMini.QKNorm` | the normalized score, its bounds, and the RMS parameterization |
| `GPTMini.QKNormLipschitz` | and how much that score can move |
| `GPTMini.CausalMHA` | the head itself, masked, with XSA on its output |
| `GPTMini.AttentionBounds` | how large the weights and the head output can get |
| `GPTMini.TotalVariation` | the ℓ¹ distance between two finite probability vectors |
| `GPTMini.SoftmaxStability` | and how far the weight row can move when the scores do |
| `GPTMini.AttentionLipschitz` | how far the output and the XSA projection move with it |
| `GPTMini.HeadLipschitz` | and how far the whole head moves |
| `GPTMini.AttnSubLayerLipschitz` | and how far the sub-layer around it moves |
| `GPTMini.ReLU2FFN` | the feed-forward map |
| `GPTMini.Block` | one pre-norm block, and the residual it adds to |
| `GPTMini.BlockLipschitz` | and how far apart it sends two residual streams |
| `GPTMini.Model` | the stack, the tied unembedding, and the forward pass |
| `GPTMini.Causality` | full-stack causality and prefix preservation at every parameter assignment |
| `GPTMini.TokenInterface` | checked List Int continuations and the exact required Basis logit property |
| `GPTMini.Semantics` | internal count/order/routing guarantees, protected phase and robust code decoding |
| `GPTMini.Properties` | what holds of it at every weight assignment at all |
| `GPTMini.Bridge` | and how it sits inside the setups of the formalized papers |
| `GPTMini.ClusteringTheorem` | and what those setups would then say about its layers |
| `GPTMini.TwoTokens` | two tokens under the Pre-LN head: a ray, a monotone inner product, bounded steps |
| `GPTMini.RateRefutation` | and why no depth rate is uniform in the initial stream |
| `GPTMini.Convex` | a causal sparsemax replacement, its stack, and the remaining FFN training obstruction |

The bridges are the connection to `Transformer.Section1_IPS …
Transformer.MeanField`: `Bridge.SphereResidence` puts the tokens on the sphere
those theorems live on, `Bridge.RoPEAsTimeVarying` shows RoPE is a
pair-dependent key matrix `R(p_j - p_i) K`, which those setups do not allow, `Bridge.CausalConnection` matches the
mask with `Transformer.Causal.CSA`, and `Bridge.XSAEquivalence` identifies the
XSA output at `V = I` with the spherical projection. `ClusteringTheorem`
assembles them into the statement that the representations cluster to one
direction for almost every initial configuration — a composition of bridges,
whose remaining `sorry`-leaves are the deep theorems of the paper
formalizations themselves and not anything about `gpt-mini`.
-/

@[expose] public section
