module

public import Transformer.Rotate90.FeatureMotion.Adam
public import Transformer.Rotate90.FeatureMotion.Source
public import Transformer.Rotate90.FeatureMotion.SharpSource

/-!
# Prospective full-encoder motion for the Rotate90 source graph

Source: `review/flat_spectrum.md`, *Full encoder motion*, EM1–EM5.
Checkpoint parameter radii propagate through shared input embeddings, three
moving real matrices, calibrated complex products, the residual and all four
RMSNorms. Tied readout motion then supplies prospective probability ceilings.
Warm-Adam cones supply exact-real parameter displacement for arbitrary future
gradients. Serialized trainer equivalence and floating-point execution are
separate obligations. PA1–PA2 sharpen old amplification using pair amplitudes
and proved Gram bounds; no convergence or useful measured horizon is asserted.
-/
