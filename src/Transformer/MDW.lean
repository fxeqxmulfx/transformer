module

public import Transformer.MDW.Checkpoints

/-!
# Adam and AMSGrad magnitude/direction optimizers with physical weight decay

Local MDW and Schedule-Free MDW compositions. Exact-real recurrences preserve
the requested fused-weight correction, adaptive histories, and nonzero-base
direction radius. Native Adam and the two Schedule-Free variance conventions
are connected to existing definitions and checked by scalar checkpoints.
No general convergence theorem for these compositions is asserted.
-/
