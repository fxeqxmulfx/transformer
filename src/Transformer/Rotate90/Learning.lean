module

public import Transformer.Rotate90.Learning.Trajectory

/-!
# Actual finite-stack learning transfer

Source: `review/flat_spectrum.md`, *Learning transfer*, T1--T6.
Joint analyticity, exact revised Schedule-Free evaluation displacement,
full-softmax classification calibration, segment-curvature descent and
finite-time bounds are proved. The actual minibatch trajectory differentiates
all shared and branch parameters at its training query. Curvature and favorable
transfer must still be established to obtain a useful rate; sampled diagnostics
do not prove those conditions for future steps or random initializations.
-/
