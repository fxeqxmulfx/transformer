module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Defs
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Identification
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.CrossEntropy
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Witness
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Refusal
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Endpoint

/-!
# Finite propagation for arbitrary fixed latent decoders

Source: `review/flat_spectrum.md`, CM19. An arbitrary categorical decoder on a
nonzero finite cyclic group replaces the opposite-pair restriction of CM18.
Gauge-fix an observed row, enumerate all phases of its anchor label, and prune
unsupported row/column values for any finite number of sweeps. Every fitting
assignment survives in a branch. Only a singleton union of decoded outputs,
across all branches and all members of a declared decoder bank, certifies an
answer. Empty domains and actual disagreements give sound refusals.

The certified one-hot equalities form a convex restriction of the CM15 spectral
decoder hull. Under realizability within the bank, they retain a feasible
phase witness and the global ordinary-cross-entropy optimum. The explicit
two-by-two example uses an even group and a fiber of three latent states;
its missing label is identified while all three hidden states remain possible.

No claim of completeness, efficient decoder discovery, random-split coverage,
numerical stability, optimizer convergence, or original stress-task success.
-/

@[expose] public section
