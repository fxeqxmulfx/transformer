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

For the architectural specification and basic universal-in-weights
properties, see the sub-modules.  For clustering / convergence
theorems and bridges to the formalized theory in
`Transformer.Section1_IPS … Transformer.MeanField`, see Phases 5–6
of `todo.md`.
-/

import Transformer.GPTMini.Config
import Transformer.GPTMini.RMSNorm
import Transformer.GPTMini.RoPE
import Transformer.GPTMini.QKNorm
import Transformer.GPTMini.CausalMHA
import Transformer.GPTMini.ReLU2FFN
import Transformer.GPTMini.Block
import Transformer.GPTMini.Model
import Transformer.GPTMini.Properties
import Transformer.GPTMini.Bridge
import Transformer.GPTMini.ClusteringTheorem
