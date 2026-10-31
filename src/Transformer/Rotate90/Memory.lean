module

public import Transformer.Rotate90.Memory.Prediction

/-!
# Optimizer history and finite-network parameter averaging

Source: `review/flat_spectrum.md`, *Optimizer memory*, M1--M6.
The actual Schedule-Free source continues averaging at zero current rate.
Its exact history-dependent clock drives evaluation weights to the saved base,
and the actual finite model eventually inherits that base's strict comparisons.
Complete-parameter sign symmetry supplies perfect powered-division endpoints
whose midpoint has uniform logits, proving the solver region nonconvex.
These results constrain the learning explanation; they do not guarantee
typical random-start convergence under positive-rate training.
-/
