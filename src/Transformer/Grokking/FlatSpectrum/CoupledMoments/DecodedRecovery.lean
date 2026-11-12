module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Signs
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Certificate
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Rigidity
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Coverage
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Learner
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.PhaseHull
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.CrossEntropy
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Domain
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.ZeroMass
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Obstruction
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Witness

/-!
# Unseen decoded answers behind a many-to-one decoder, from training samples alone

Source: `review/flat_spectrum.md`, CM18. A square table has the labels `d (r i + s j)` for a
fixed known decoder `d` of an odd cyclic group that merges exactly the opposite latent states,
and unknown phases `r`, `s`. A training sample gives its latent state up to a sign, one
unknown sign per cell. A certificate proposes phases and a rectangle of training cells and
reaches further rows and columns by propagation; its conditions are finite statements about
training samples. If it is accepted, every completion of the training samples, the target
among them, has the proposed label at every reached cell (`accepted_decoding`); latent states
stay undetermined up to a reflection and a shift. Conditions on training labels alone
guarantee acceptance for a realizable table (`decoding_of_labels`), and the procedure
`recover` returns the label or one of three refusals.

In the convex programs: the decoded table of fitting phases is a feasible minimizer of the
training cross-entropy over the spectral relaxations of CM15 and CM16 with the target's
answers at the covered cells, but that relaxation has another minimizer that is wrong at a
covered cell (`budget_wrong_minimum`). On the convex hull of decoded phase tables every
minimizer has the target's rows at the covered cells, and so has every minimizer of the
spectral program restricted by the certified linear equalities.

Not claimed: a decoder outside the class, an even number of latent states, cells that the
certificate does not reach, a running time, a way to optimize over the hull, the behaviour of
a numerical solver, or any original stress task.
-/

@[expose] public section
