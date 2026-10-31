module

public import Transformer.SequenceModel.Attention.Stability.Classification
public import Transformer.SequenceModel.Attention.Stability.Rounding

/-!
# Source RoPE approximation and rounding certificates

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A1--A6.
Phase errors propagate through RMS Q/K, causal softmax and XSA. Joint
continuity through the full source stack proves that a strict classifier on
a fixed finite context survives sufficiently accurate periodic frequency
replacement, at unchanged weights, depth and width. The period may depend
on the context: no uniform periodic depth lower bound is transferred.

Separate quantitative bounds cover tied readout errors and the existing
counting model's rounded attention aggregate, with explicit no-saturation
and positive-denominator conditions. They are not an IEEE error certificate
or a global error estimate for the complete rounded residual stack.
-/
