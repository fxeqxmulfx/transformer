module

public import Transformer.Landscape.Neural.MLP.Model
public import Transformer.Landscape.Neural.MLP.Losses

/-!
# Ordinary scalar and vector perceptrons in Landscape

`Model` retains two biased SiLU hidden layers and every free affine output
coordinate, matching the M9.3 controls. `Losses` proves analyticity and the
local Lojasiewicz inequality for their actual scalar/vector half-MSE and
full-class cross-entropy. These are shared regularity properties with the
analytic Rotate90 networks; identifying gradient accessibility requires
additional, architecture-sensitive information.
-/
