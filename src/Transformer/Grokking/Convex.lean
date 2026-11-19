module

public import Transformer.Grokking.Convex.ConvexRecall
public import Transformer.Grokking.Convex.ConvexChannel
public import Transformer.Grokking.Convex.SpectralBall
public import Transformer.Grokking.Convex.CoupledMoments
public import Transformer.Grokking.Convex.ConvexDepth
public import Transformer.Grokking.Convex.ConvexCausal

/-! Convex follow-ups of the flat-spectrum note, built on the definitions and theorems of
`Transformer.Grokking.FlatSpectrum`. From `review/convex/channels.md`: convex answer-loss recall
(`ConvexRecall`), convex channels with flat-spectrum capacity bounds (`ConvexChannel`) and convex
spectral bounds with missing-cell recovery (`SpectralBall`). From `review/convex/moments.md`:
shared class distributions and complex spectral bounds (`CoupledMoments`), which most modules of
`ConvexChannel` and `ConvexDepth` also cite. Nonlinear residual depth with convex joint path
training (`ConvexDepth`) is `review/convex_depth.md`, and a calibrated convex causal model for
finite-token stress tasks (`ConvexCausal`) is `review/convex_causal.md`. -/

@[expose] public section
