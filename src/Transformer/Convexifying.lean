/-
# Convexifying Transformers: optimization and attention

Formalization of Ergen, Neyshabur, Mehta — arXiv:2211.11052v1,
*Convexifying Transformers: Improving optimization and understanding of
transformer networks*.

The modules encode the finite-dimensional attention models of §3 and the
matrix-target extension in the appendix.  The softmax/simplex inclusion,
the AM–GM rescaling bound, and finite-width counterexamples are proved.  A
two-sample example also shows that one simplex matrix shared across samples
cannot reproduce sample-dependent self-attention.
For the original and relaxed regularized training objectives themselves, a
one-sample example gives a strict gap, even with positive simplex weights.
With trainable query and key weights, the prediction set of one ordinary
softmax head is nonconvex on a two-sample data set.  A finite atomic mixture
of genuine softmax heads has a convex prediction-budget epigraph under a
positively homogeneous coefficient penalty, but it does not preserve the
original four-matrix weight decay.  The exact prediction-budget epigraph of
ordinary attention is nonconvex even with arbitrarily many finite heads.
Allowing a nonlinear decoder or a changed convex regularizer does not give
an exact epigraph formulation that is convex for every squared-loss target.
An explicit squared loss also separates the original and atomic objectives.
Even equality of all attainable objective sublevels fails for a convex
candidate across every nonnegative squared-loss weight and target; this
last obstruction applies to any real vector space of parameters.
The obstruction persists when objective bounds may be transferred with any
positive tolerance, so it does not rely on attainment.
The source's unqualified scalar, vector, gated, and matrix equivalences fail
for one head.  Its scalar convex objective also halves the loss on only one
side.  Corrected scalar and vector statements with sufficient head counts,
the scaling equivalence, and the sparse-optimum theorem are proved.

The source also leaves `‖·‖_{1,∞}` undefined in the matrix appendix; the
formal matrix convex objective takes this norm as an argument.  Experiments
in §4 are empirical results and have no Lean theorem here.
-/

import Transformer.Convexifying.Section3_Models
import Transformer.Convexifying.Section3_Factorization
import Transformer.Convexifying.Section3_ConvexAttention
import Transformer.Convexifying.Section3_SharedAttention
import Transformer.Convexifying.Section3_OriginalRelaxation
import Transformer.Convexifying.Section3_TrainableScores
import Transformer.Convexifying.Section3_AtomicEpigraph
import Transformer.Convexifying.Section3_SquaredLossGap
import Transformer.Convexifying.Section3_UniversalValues
import Transformer.Convexifying.Section3_ApproxValues
import Transformer.Convexifying.Section3_Scaling
import Transformer.Convexifying.Section3_ScalarBridge
import Transformer.Convexifying.Section3_VectorBridge
import Transformer.Convexifying.Section3_VectorRecovery
import Transformer.Convexifying.Section3_Caratheodory
import Transformer.Convexifying.Section3_ScalarCounterexample
import Transformer.Convexifying.Section3_Mapping
import Transformer.Convexifying.Section3_Vector
import Transformer.Convexifying.Section3_FCN
import Transformer.Convexifying.Section3_Corrected
import Transformer.Convexifying.Section3_Sparsity
import Transformer.Convexifying.Appendix_Matrix
