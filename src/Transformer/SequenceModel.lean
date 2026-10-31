module

public import Transformer.SequenceModel.Pooling
public import Transformer.SequenceModel.RotarySpectrum
public import Transformer.SequenceModel.Transfer.Alignment
public import Transformer.SequenceModel.Attention.FiniteStep
public import Transformer.SequenceModel.Attention.ZeroRun
public import Transformer.SequenceModel.Attention.Expressivity.Existence

/-!
# General positional inputs and normalized multi-variable attention

Source: `review/flat_spectrum.md`, *General positional models*, G1--G5,
and *Transformer learning bridges*, R1, S1--S2, T1--T4, E1--E8.
The sampled source dynamics apply to arbitrary finite parameterized predictors.
Arbitrary rotations, per-pair RoPE frequencies, other positional mixers and
finite input tuples are included; fixed 90 degrees is only one special case.

The concrete attention model follows the user's diff_gpt at b98c665b, retaining
embedding/final RMSNorm, Peri-RMSNorm on both branches, Q/K RMSNorm, arbitrary
rotary phases, causal softmax, clipped XSA, SwiGLU and tied output. Inverse
coordinate maps prove that every trainable matrix entry participates.

Local analyticity is proved from actual XSA activation regularity. Actual
CE gradients and Adam preconditioning derive cross-Jacobian transfer and
finite prediction bounds. A finite source-shaped Transformer represents
every binary finite task, including powered division, with a strict margin.
RoPE spectrum and zero-initialization counterexamples rule out unconditional
spectral invariance and learning guarantees. Larger-context forward and
learning formulas are general; the constructive task theorem uses two tokens.
-/
