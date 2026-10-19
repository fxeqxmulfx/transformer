module

public import Transformer.Landscape.Neural.Basic
public import Transformer.Landscape.Neural.Loss
public import Transformer.Landscape.Neural.Residual
public import Transformer.Landscape.Neural.SiLU
public import Transformer.Landscape.Neural.SwiGLU
public import Transformer.Landscape.Neural.Interaction
public import Transformer.Landscape.Neural.AdamW
public import Transformer.Landscape.Neural.MLP
public import Transformer.Landscape.Neural.Products
public import Transformer.Landscape.Neural.Tangent

/-!
# Analytic neural losses for the Landscape framework

`Basic` proves analyticity of SiLU, SwiGLU and positive-epsilon RMSNorm.
`Loss` proves analyticity of finite cross-entropy and scalar squared error,
including the competing classes of the browser model.
`Residual` defines the lab's dense maps, quarter-turn and shared readout.
`SiLU` and `SwiGLU` instantiate the complete residual classifiers in their
actual Euclidean weights and discharge the local Lojasiewicz hypothesis.
`Interaction` proves that the actual normalized tied readout requires a
nonlinear operand interaction to separate a division checkerboard.
`AdamW` derives a compact parameter container for the source-matched
exact-real optimizer recurrence, with its actual inclusive warmup.
`MLP` instantiates both ordinary perceptron readouts in their full biased
parameter spaces and proves the same analytic local-inequality properties.
`Products` compiles exact paired-gate products into the actual SwiGLU
parameters while retaining both normalizations, skip and shared readout.
`Tangent` derives the exact quadratic gradient and normalized prediction
kernel used by the frozen-Jacobian GD diagnostic, separately from AdamW.
Sources and the remaining bridge to the Rotate90 lab are recorded in
`landscape_grokking_plan.md`.
-/

@[expose] public section
