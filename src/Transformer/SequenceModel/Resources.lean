module

public import Transformer.SequenceModel.Resources.Parameters

/-!
# Stored parameter budgets for the normalized source model

Source: `review/flat_spectrum.md`, *Depth, width and precision*, R1--R2.
For a fixed parameter alphabet of at most 2^b values, universal strict
classification requires at least one stored parameter bit per possible input
row. Applying this to every actual source matrix yields a joint necessary
budget in depth, width and precision, independently of optimization.
-/
