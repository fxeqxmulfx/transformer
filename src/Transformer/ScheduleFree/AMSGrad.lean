module

public import Transformer.ScheduleFree.AMSGrad.MagnitudeInvariants

/-!
# Schedule-Free AMSGradW and factor-space AMSGradMD

New compositions of the Schedule-Free recurrence, AMSGrad's raw running
maximum and magnitude/direction coordinates. The modules define actual
oracle locations and state updates, prove buffer and averaging invariants,
verify the query-point loss chain rule and prove the base projection norm.
They do not assert a convergence theorem for the combined optimizers.
-/

@[expose] public section
