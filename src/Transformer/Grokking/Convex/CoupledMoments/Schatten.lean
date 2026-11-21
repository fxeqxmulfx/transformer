module

public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Basic
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.NormComparison
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Recovery
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Optimality
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Hull
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Intermediate
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Completion
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Quotient
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Sharpen
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.GraphRecovery
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.PhaseCandidate
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Decoded
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.DecodedPhase
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.DecodedHull
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Paired
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.PairedPhase
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.PairedCoding
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.PairedLimits
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Shift

/-!
# Shared complex moments with singular-value budgets

Source: `review/convex/moments.md`, CM5–CM7, CM9–CM11, CM13–CM16 and CM121. Convex bounds on
the largest singular value
and their sum accommodate different spectral shapes. A concrete rank-one recovery proof
starts with ordinary empirical CE and does not assume correct missing labels or a trained
factorization. The two endpoint budgets miss an explicit intermediate-rank moment.
An uncertain completion can still minimize CE after that capacity gap is repaired;
fixed correct sharing recovers the same witness's missing columns.
Decoded endpoints can be sharpened and checked against both spectral budgets,
with an ordinary-CE excess bound from their common uniform mixing coefficient.
Connected observation graphs with perfect matchings identify the complete table
at ideal CE when a separating code bank has the stated unit-phase rank-one structure.
Recovery outside these hypotheses and numerical solver convergence are not asserted.
A training-compatible categorical phase coding constructs a feasible CE minimizer;
connected matching coverage makes it unique in the fixed-code nuclear budget family.
Fixed many-to-one output decoding preserves convexity and observed CE optimality,
and represents a table excluded by the direct binary rank-one budget. It does not
preserve latent-label identification: two decoded optima can disagree unseen even
with connected matching coverage.
The common hull over all fixed decoders remains convex; a feasible exact decoded
fit certifies global ordinary-CE optimality over that whole prediction family.
Equal probabilities on opposite latent states are linear constraints that cost an even
decoder no prediction; a symmetric phase witness is feasible and CE-optimal when it fits,
the folding decoder admits a table outside the permutation-phase constructions, and two
paired optima still disagree at an unseen cell.
A class shift `c ↦ c + u i + v j` in every cell maps each budget set of character moments onto
itself and observed CE to observed CE with the labels shifted alike.
-/

@[expose] public section
