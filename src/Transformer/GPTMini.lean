/-
# `gpt-mini` Formalization

Top-level module re-exporting the formalization of the `gpt-mini`
transformer architecture defined in `reference/model.py`.

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
| `GPTMini.RoPE` | the rotary tables, and that they rotate |
| `GPTMini.QKNorm` | the normalized score, its bounds, and the RMS parameterization |
| `GPTMini.QKNormLipschitz` | and how much that score can move |
| `GPTMini.CausalMHA` | the head itself, masked, with XSA on its output |
| `GPTMini.AttentionBounds` | how large the weights and the head output can get |
| `GPTMini.SoftmaxStability` | and how far the weight row can move when the scores do |
| `GPTMini.AttentionLipschitz` | how far the output and the XSA projection move with it |
| `GPTMini.ReLU2FFN` | the feed-forward map |
| `GPTMini.Block` | one pre-norm block, and the residual it adds to |
| `GPTMini.BlockLipschitz` | and how far apart it sends two residual streams |
| `GPTMini.Model` | the stack, the tied unembedding, and the forward pass |
| `GPTMini.Properties` | what holds of it at every weight assignment at all |
| `GPTMini.Bridge` | and how it sits inside the setups of the formalized papers |
| `GPTMini.ClusteringTheorem` | and what those setups would then say about its layers |

The bridges are the connection to `Transformer.Section1_IPS …
Transformer.MeanField`: `Bridge.SphereResidence` puts the tokens on the sphere
those theorems live on, `Bridge.RoPEAsTimeVarying` reads RoPE as the
time-varying `Q, K` they already allow, `Bridge.CausalConnection` matches the
mask with `Transformer.Causal.CSA`, and `Bridge.XSAEquivalence` identifies the
XSA output at `V = I` with the spherical projection.  `ClusteringTheorem`
assembles them into the statement that the representations cluster to one
direction for almost every initial configuration — a composition of bridges,
whose remaining `sorry`-leaves are the deep theorems of the paper
formalizations themselves and not anything about `gpt-mini`.
-/

import Transformer.GPTMini.Config
import Transformer.GPTMini.RMSNorm
import Transformer.GPTMini.Reshape
import Transformer.GPTMini.RoPE
import Transformer.GPTMini.QKNorm
import Transformer.GPTMini.QKNormLipschitz
import Transformer.GPTMini.CausalMHA
import Transformer.GPTMini.AttentionBounds
import Transformer.GPTMini.SoftmaxStability
import Transformer.GPTMini.AttentionLipschitz
import Transformer.GPTMini.ReLU2FFN
import Transformer.GPTMini.Block
import Transformer.GPTMini.BlockLipschitz
import Transformer.GPTMini.Model
import Transformer.GPTMini.Properties
import Transformer.GPTMini.Bridge
import Transformer.GPTMini.ClusteringTheorem
