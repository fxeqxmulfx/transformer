module

public import Transformer.SequenceModel.Attention.Backend.Complete.Obstruction

/-!
# Complete captured normalized RoPE source execution

Source: `review/actual_network/execution.md`, *Complete captured execution*, L1--L5.
An actual compiled CUDA trace satisfies every ModelAccurate operation check.
Its nonzero residual rounding proves that the existing uniform global error
budget cannot certify a decision margin, even if its allowance is tightened.
This establishes a concrete numerical gap in the spectral/counting bridge.
-/
