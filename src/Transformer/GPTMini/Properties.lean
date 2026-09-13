/-
# `gpt-mini` Universal-in-Weights Properties

Re-exports of all proven structural properties of the `gpt-mini` forward
pass that hold for any weight assignment.

Contents:
  - `OutputSimplex`: softmax output is a probability distribution
  - `Causal`: causal-mask structural lemmas
-/

import Transformer.GPTMini.Properties.OutputSimplex
import Transformer.GPTMini.Properties.Causal
import Transformer.GPTMini.Properties.StreamGrowth
import Transformer.GPTMini.Properties.Lipschitz
import Transformer.GPTMini.Properties.Entropy
