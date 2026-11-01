module

public import Transformer.SequenceModel.Attention.Backend.CapturedCPU32
public import Transformer.SequenceModel.Attention.Backend.CapturedCPU64
public import Transformer.SequenceModel.Attention.Backend.CapturedCUDA32
public import Transformer.SequenceModel.Attention.Backend.Complete
public import Transformer.SequenceModel.Attention.Backend.Refined
public import Transformer.SequenceModel.Attention.Adaptive

/-!
# Concrete backend certificates for normalized source execution

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K1--K6, and
*Complete captured execution*, L1--L5, and *Trace-dependent execution bounds*, M1--M5.
Algebraic scale residuals certify RMSNorm and clipped L2; exponential-series
remainders certify SiLU and normalized attention masses; complex-series
remainders certify independently stored RoPE coefficients. The causal
adapter gives the existing execution trace's SDPA-average contract directly.
Exact rational encodings of captured CPU float32/float64 and compiled CUDA
float32 outputs have ordinary Lean proofs of their local error bounds.
The complete captured CUDA source trace also satisfies ModelAccurate, with
every local premise discharged. An actual residual-add error proves that the
existing uniform global margin bound cannot certify this trace, even after
tightening its allowance. Separate local bounds instead certify every captured
class decision against the exact source, with logit radius 1/16 and margin 7/10.
The counting bridge retains its independent fixed-point range/precision conditions.
No guarantee for uncaptured invocations is asserted.
-/
