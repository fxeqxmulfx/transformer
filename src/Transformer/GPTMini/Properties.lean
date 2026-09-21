/-
# `gpt-mini` Universal-in-Weights Properties

Re-exports of all proven structural properties of the `gpt-mini` forward
pass that hold for any weight assignment.

Contents:
  - `OutputSimplex`: softmax output is a probability distribution
  - `Causal`: causal-mask structural lemmas
  - `StreamGrowth`, `LipschitzConstants`, `Lipschitz`: the stream's growth and
    its Lipschitz dependence on the input
  - `Entropy`, `EntropyEmbedding`: the output entropy, bounded below by the
    embedding norm alone
-/

import Transformer.GPTMini.Properties.OutputSimplex
import Transformer.GPTMini.Properties.Causal
import Transformer.GPTMini.Properties.StreamGrowth
import Transformer.GPTMini.Properties.LipschitzConstants
import Transformer.GPTMini.Properties.Lipschitz
import Transformer.GPTMini.Properties.Entropy
import Transformer.GPTMini.Properties.EntropyEmbedding
