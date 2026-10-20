module

public import Transformer.MHC.Section4_Regularized

/-!
# Manifold-Constrained Hyper-Connections

Residual sections of Xie et al., arXiv:2512.24880v2, §§1, 4.1–4.2,
with shared HC operator algebra from Zhu et al., arXiv:2409.19606v3, §2.1.
The development proves ideal skip-map stability and conservation,
formalizes the gate and finite Sinkhorn formulas, and separately models
the epsilon-regularized code used by GLM-5.3-Flash and DeepSeek V4.1.
It includes counterexamples to interpreting nonexpansion as an isometry
or as a bound on the complete nonlinear block. No empirical performance
or optimizer claim, and no infinite Sinkhorn convergence theorem, is asserted.

See `review/residual_connections.md` for pinned manuscripts and code.
-/
