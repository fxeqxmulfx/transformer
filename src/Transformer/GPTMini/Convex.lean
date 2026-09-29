/-
# Convex replacements for gpt-mini components

New modifications of `experiments/gpt_mini.py`, restored from commit
f11b6e27d3cfe6813a2876bdeec38565fc54258c. Inference convexity and training
convexity are separate claims. The first replacement is causal sparsemax.
Motivation: Convexifying Transformers, §3.1; not an equivalence to its original
softmax model or a claim that all gpt-mini parameters train jointly convexly.
-/

import Transformer.GPTMini.Convex.Basic
import Transformer.GPTMini.Convex.Attention
import Transformer.GPTMini.Convex.Model
import Transformer.GPTMini.Convex.TrainingBoundary
