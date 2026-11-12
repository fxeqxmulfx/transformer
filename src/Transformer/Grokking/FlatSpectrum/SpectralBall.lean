module

public import Transformer.Grokking.FlatSpectrum.SpectralBall.Basic
public import Transformer.Grokking.FlatSpectrum.SpectralBall.Recovery
public import Transformer.Grokking.FlatSpectrum.SpectralBall.Ambiguity

/-!
# Convex spectral bounds and missing-cell recovery

Source: `review/flat_spectrum.md`, SB1–SB3. Complex matrix entries, unit entry disks and
an upper L2 operator-norm bound give a convex model with ordinary CE. On XE15's three
observations the tight bound repairs the wrong unseen sign of nuclear-norm training.
Two flat `4 × 4` targets agreeing on twelve observed cells show the repair is not universal.
-/

@[expose] public section
