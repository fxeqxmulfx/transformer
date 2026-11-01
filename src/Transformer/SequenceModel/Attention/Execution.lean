module

public import Transformer.SequenceModel.Attention.Execution.Certificate
public import Transformer.SequenceModel.Attention.Execution.Rounding

/-!
# Operation-by-operation execution certificates for normalized source RoPE

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C1--C8.
Finite binary32 brackets and explicit sequential/FMA dot products supply
local numerical bounds. Recorded projections, normalization, independent
cosine/sine tables, attention, XSA, SwiGLU, residuals and tied logits propagate
through arbitrary finite width and depth to a strict prediction certificate.
Backend contracts, including SDPA accuracy and finite outputs, remain explicit.
The uniform counting bridge uses the existing exact nonperiodic kernels;
it does not assert a strict depth lower bound or verified CUDA semantics.
-/
