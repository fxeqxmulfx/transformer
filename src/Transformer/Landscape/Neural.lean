/-
# Analytic neural losses for the Landscape framework

`Basic` proves analyticity of SiLU, SwiGLU and positive-epsilon RMSNorm.
`Loss` proves analyticity of finite cross-entropy and scalar squared error,
including the competing classes of the browser model.
Sources and the remaining bridge to the Rotate90 lab are recorded in
`landscape_grokking_plan.md`.
-/

import Transformer.Landscape.Neural.Basic
import Transformer.Landscape.Neural.Loss
