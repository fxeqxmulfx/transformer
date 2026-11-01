module

public import Transformer.SequenceModel.Attention.Stability.Classification
public import Transformer.SequenceModel.Attention.Stability.Rounding
public import Transformer.SequenceModel.Attention.Stability.Certificate

/-!
# Source RoPE approximation and rounding certificates

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A1--A6,
and *Full-stack rounding certificates*, B1--B9.
Phase errors propagate through RMS Q/K, causal softmax and XSA. Joint
continuity through the full source stack proves that a strict classifier on
a fixed finite context survives sufficiently accurate periodic frequency
replacement, at unchanged weights, depth and width. The period may depend
on the context: no uniform periodic depth lower bound is transferred.

Explicit quantitative bounds propagate the existing counting model's
rounding through the complete residual stack and final tied readout. They
retain every source normalization, XSA and actual SwiGLU, use each block's
matrix norms, and require range checks at the actual rounding sites. A
sufficient strict source margin transfers classification to the existing
uniform counting formula. This specified fixed-point model is not IEEE
arithmetic, and the sufficient bounds need not be sharp for trained weights.
-/
