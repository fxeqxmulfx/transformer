module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Execution
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Domain
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Minibatch

/-!
# Complete readout refitting on already learned recurrent states

Source: `review/flat_spectrum.md`, CM100. Actual complex readout coordinates,
radial row feasibility, unchanged learned state execution, a joint conditional
convex ordinary-CE subproblem and feasible full-family finite queries. No
fitting, optimizer-convergence or unseen-answer guarantee is assumed.
-/
