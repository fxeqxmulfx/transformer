module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SelectedAnchors.Basic

/-!
# Selected identifying anchors and complete observed CE

Source: `review/flat_spectrum.md`, CM61. Conflicting identical-feature exact
anchors force zero program masses. An explicit selection can restore a
nonempty convex joint domain while the ordinary CE retains every observed
label. Removing an identifying equation also removes the identification it
previously supplied; no clean-label or unseen-target guarantee follows.
-/
