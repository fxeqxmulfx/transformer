module

public import Transformer.Rotate90.Division

/-!
# Exact division expressivity of the current Rotate90 SwiGLU model

Source: `grokking_abmod/docs/rotate90-swiglu-model.md` and the actual
`scripts/grok-simple.py` forward pass, inspected 2026-10-31.

`Rotate90.exists_division_weights` proves that for every prime p and every
positive RMSNorm epsilon, explicit one-block weights solve a/b on all a and
all nonzero b. Eight active residual coordinates and fourteen hidden units
suffice. Extra channel pairs and hidden units may be padded with zeros.
`exists_division_width16` covers the documented 3616-parameter p=97 model.

The construction retains all four normalizations, the fixed quarter-turn,
residual, genuine SiLU gates, and one shared embedding for input and output.
It uses a faithful multiplicative character plus a zero coordinate. Matching
the branch and input norms makes cancellation exact despite Peri-RMSNorm.
These are expressivity results in real arithmetic, independent of the
flat-spectrum interpolation surrogate and of optimizer convergence.
-/
