/-
# Convex replacements for gpt-mini components

New modifications of `experiments/archive/gpt_mini/gpt_mini.py`, restored
from commit f11b6e27d3cfe6813a2876bdeec38565fc54258c. Inference convexity
and training convexity are separate claims. The first replacement is causal
sparsemax. Motivation: Convexifying Transformers, §3.1; not an equivalence
to its original softmax model or a claim that all gpt-mini parameters train
jointly convexly.

The new direct joint-interaction route changes the attention operator to
an unnormalized causal sum and absorbs the embedding/Q/K/value products.
It gives an unconstrained parameter-space training guarantee, without
changing the optimizer, at cubic vocabulary storage. Its exact finite-head
recovery and single-head compression counterexample state that limit.
Actual two- and three-token outputs identify all interaction coordinates;
exact affine coverage of every linear head therefore has cubic dimension.
That coverage bound leaves nonlinear and restricted-class routes open.
The combined token-to-stream prefix is a candidate interface replacement;
no compact Python drop-in block or full-model convexity is established.
-/

import Transformer.GPTMini.Convex.Basic
import Transformer.GPTMini.Convex.Attention
import Transformer.GPTMini.Convex.Model
import Transformer.GPTMini.Convex.TrainingBoundary
import Transformer.GPTMini.Convex.JointInteractionBoundary
import Transformer.GPTMini.Convex.InteractionCompression
