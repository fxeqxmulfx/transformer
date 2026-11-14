module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoverageExtension.Witness

/-!
# Conservative coverage extension of bounded complex heads

Source: `review/flat_spectrum.md`, CM39. The supplemental head is used only
where it accepts and the prior refuses. Every other input retains exact prior
logits. Original prior-covered batches retain their ordinary CE and derivatives.
The CM38 convex domain and flat-feature bounds still charge both complex blocks.
A concrete witness has three disagreements but changes only the newly covered
one. Neither structural coverage nor this preservation theorem identifies its
unknown target. Native binding and complete metrics remain separate work.
-/
