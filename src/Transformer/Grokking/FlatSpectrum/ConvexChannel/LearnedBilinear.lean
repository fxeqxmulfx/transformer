module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Model
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Reader
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Sequence
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Geometry
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Witness
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Parity
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Restriction
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Spectral
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates

/-!
# A learned rectangular complex sequence family with exact convex answer loss

Source: `review/flat_spectrum.md`, CM86. Immutable object sequences read learned
query, position and shared adjacent-pair weights; balanced spectral recovery
supplies both encoders and the learned output magnitude. The actual complex
interaction returns probabilities over the original objects. Exact physical
coverage and joint CE convexity are proved, alongside changed encoder geometry,
three-object parity capacity, a four-object parity obstruction and flat-spectrum
width bounds. These results do not provide native training or stress credit.
CM87 proves observed-margin identification with a complete finite certificate,
ordinary CE/error bridges and unseen success/refusal controls. CM88 proves
complete forward-query reuse and exact zero-column storage for native updates.
-/
