module

public import Transformer.SequenceModel.Pooling
public import Transformer.SequenceModel.Attention.Training

/-!
# General positional inputs and normalized multi-variable attention

Source: `review/flat_spectrum.md`, *General positional models*, G1--G5.
The sampled source dynamics apply to arbitrary finite parameterized predictors.
Arbitrary rotations, per-pair RoPE frequencies, other positional mixers and
finite input tuples are included; fixed 90 degrees is only one special case.

The concrete attention model follows the user's diff_gpt at b98c665b, retaining
embedding/final RMSNorm, Peri-RMSNorm on both branches, Q/K RMSNorm, arbitrary
rotary phases, causal softmax, clipped XSA, SwiGLU and tied output. Inverse
coordinate maps prove that every trainable matrix entry participates.

The proven normalization bounds depend on head width, residual width and
depth. They do not assert convergence, injective pooling, or a universal
number of variables beyond which attention is required or impossible.
-/
