module

public import Transformer.Grokking.Convex.ConvexChannel.Observation.Basic
public import Transformer.Grokking.Convex.ConvexChannel.Observation.Span
public import Transformer.Grokking.Convex.ConvexChannel.Observation.Examples

/-!
# Observable features determine unseen predictions

Source: `review/convex/channels.md`, *Observable features determine unseen
predictions*, O1-O4. Realizable answer labels determine the observed logits
of every global CE optimizer. An input-only linear reconstruction certificate
then determines unseen logits, with an explicit error amplification bound.
The Boolean example infers the unobserved dictionary 11 from 00, 01, and 10.
This supplies an identification condition in addition to spectral capacity.
-/
