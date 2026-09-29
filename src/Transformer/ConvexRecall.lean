/-
# Convex training with content-dependent multi-query recall

This is a new cross-paper construction. It extends the convex-training aim
of arXiv:2211.11052v1, §3, and meets the aligned MQAR specification from
arXiv:2312.04927v1, §3 and Appendix `app:synthetic`.

Token embeddings are fixed injective binary codes. The trainable metric is
a weighted Hamming cost. A convex calibration objective, using one-bit
mismatch examples, has a unique global optimum at the unit metric. For each
query, content costs determine a masked simplex quadratic program with a
null slot. Its unique optimum selects the matching earlier key, or null
when none exists. Value mixing then agrees exactly with corrected causal
equality-score attention on every distinct-key input and every query.

This is not the original paper's sample-independent positional matrix, nor
an exact reparameterization of ordinary jointly trained Q/K/V attention.
The binary encoder, local key/value alignment, quadratic regularizer, and
extra matching calibration are explicit changes. Exact functional recall
does not establish parity of end-to-end training, speed, or parameter count
with the two-layer GPT-style experiments in the MQAR paper.
-/

import Transformer.ConvexRecall.Training
import Transformer.ConvexRecall.Simplex
import Transformer.ConvexRecall.Routing
import Transformer.ConvexRecall.Recall
