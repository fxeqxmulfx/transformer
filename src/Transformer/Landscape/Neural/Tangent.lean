module

public import Transformer.Landscape.Neural.Tangent.Basic
public import Transformer.Landscape.Neural.Tangent.Spectral

/-!
# Frozen prediction derivatives as explicit Landscape controls

The quadratic objective, adjoint gradient and normalized prediction kernel
match the offline M9.4 diagnostic in `mod97_representation_controls/access.py`.
An exact eigenmode recurrence and finite-budget loss floor apply to its
actual parameter GD orbit, with a positive-eigenvalue, zero-fit witness.
These concern the frozen linear prediction model. They do not assert that
the real network keeps this derivative or follows this GD orbit under AdamW.
-/
