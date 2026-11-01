module

public import Transformer.SequenceModel.Attention.Backend.CapturedCPU32
public import Transformer.SequenceModel.Attention.Backend.CapturedCPU64
public import Transformer.SequenceModel.Attention.Backend.CapturedCUDA32

/-!
# Concrete backend certificates for normalized source primitives

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K1--K6.
Algebraic scale residuals certify RMSNorm and clipped L2; exponential-series
remainders certify SiLU and normalized attention masses; complex-series
remainders certify independently stored RoPE coefficients. The causal
adapter gives the existing execution trace's SDPA-average contract directly.
Exact rational encodings of captured CPU float32/float64 and compiled CUDA
float32 outputs have ordinary Lean proofs of their local error bounds.
These finite cases do not supply ModelAccurate for an arbitrary checkpoint,
a complete captured model trace, or all future executions of a backend.
-/
