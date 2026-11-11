module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Feedback
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.SelfLabel
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Certified
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Examples

/-!
# Learning from observations during inference

Source: `review/flat_spectrum.md`, *Inference-time learning*, CI1-CI4.
Retained candidate state can learn from new observed answers. Unverified
self-labeling leaves the prediction function fixed and can contaminate
the records used for future feedback. An agreement certificate accepts
only answers already justified by every surviving candidate's observations.
-/
