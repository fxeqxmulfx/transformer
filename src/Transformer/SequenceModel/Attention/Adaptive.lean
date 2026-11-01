module

public import Transformer.SequenceModel.Attention.Adaptive.Attention

/-!
# Reusable operation-dependent execution estimates

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M1--M2.
RMS and clipped L2 retain recorded denominators; exact learned matrices,
SiLU, products, arbitrary RoPE, scores, averages and residual additions keep
their independent input radii and local arithmetic residuals.
-/
