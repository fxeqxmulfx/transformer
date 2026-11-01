module

public import Transformer.SequenceModel.Attention.Spectral.RecoveryWitness
public import Transformer.SequenceModel.Attention.Spectral.CountingWitness
public import Transformer.SequenceModel.Attention.Spectral.LocalCounting

/-!
# From recovered spectral features to normalized RoPE transformer decisions

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J1--J7. A separating powered-division character bank recovered under an ideal
nuclear-norm objective supplies an actual source score computation. Finite
SwiGLU interpolation preserves all scores up to positive normalization,
including zero-numerator boundary cases. A positive source margin admits a
positive local execution tolerance and, under explicit fixed-point and
execution contracts, one fixed RoPE-aware counting family agrees with every
class decision. The construction retains the source normalization placement,
XSA, geometric RoPE and tied embeddings. Optimizer convergence, efficient
internal character arithmetic and strict nonperiodic depth hierarchies are
not consequences of this finite-width construction.
-/
