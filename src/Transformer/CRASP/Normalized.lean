module

public import Transformer.CRASP.Normalized.Input

/-!
# Normalized multihead source models and counting logic

Extension of arXiv:2506.16055v3, Appendix B.2/F, to the architecture documented
in `review/flat_spectrum.md`, *Depth, width and precision*, R3--R5.
Finite stored activations, parallel heads, arbitrary rotary tables, RMSNorm,
XSA, SwiGLU and tied readout compile into counting logic at the stack's depth.
The end-to-end statement keeps the original input vocabulary at fixed context.
The rounding sites differ from IEEE execution. Exact geometric source RoPE
has no finite positional code over all lengths, so the paper's periodic depth
hierarchy cannot be imported unchanged as a lower bound for this source model.
-/
