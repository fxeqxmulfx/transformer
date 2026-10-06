import Transformer.GPTMini.Sparsemax.SeparatedValues
import Transformer.GPTMini.Sparsemax.ValuePlateau
import Transformer.GPTMini.Sparsemax.AnchoredValues
import Transformer.GPTMini.Sparsemax.AnchorTransfer
import Transformer.GPTMini.Sparsemax.TrainableAnchors
import Transformer.GPTMini.Sparsemax.AnchoredSquaredError
import Transformer.GPTMini.Sparsemax.AnchoredCorrection
import Transformer.GPTMini.Sparsemax.QKChart
import Transformer.GPTMini.Sparsemax.QKAnchors
import Transformer.GPTMini.Sparsemax.QKTaskDirections
import Transformer.GPTMini.Sparsemax.QKSquaredError
import Transformer.GPTMini.Sparsemax.QKProjection
import Transformer.GPTMini.Sparsemax.QKProjectedError
import Transformer.GPTMini.Sparsemax.InputDecoder
import Transformer.GPTMini.Sparsemax.ProjectionLift
import Transformer.GPTMini.Sparsemax.ProjectionUpdate
import Transformer.GPTMini.Sparsemax.MixedInputQK
import Transformer.GPTMini.Sparsemax.QKIndependentDirections
import Transformer.GPTMini.Sparsemax.QKIndependentError
import Transformer.GPTMini.Sparsemax.PrefixInputs
import Transformer.GPTMini.Sparsemax.LongContextQK
import Transformer.GPTMini.Sparsemax.QKPrefixMatrix
import Transformer.GPTMini.Sparsemax.QKPrefixDirections
import Transformer.GPTMini.Sparsemax.QKPrefixError
import Transformer.GPTMini.Sparsemax.SupportSegment
import Transformer.GPTMini.Sparsemax.ClippedKeySegment
import Transformer.GPTMini.Sparsemax.SharedRowsExample
import Transformer.GPTMini.Sparsemax.SharedRows
import Transformer.GPTMini.Sparsemax.SquaredSegment
import Transformer.GPTMini.Sparsemax.SharedRowLoss
import Transformer.GPTMini.Sparsemax.SharedRowMinimum
import Transformer.GPTMini.Sparsemax.SharedRowDerivative
import Transformer.GPTMini.Sparsemax.SharedRowCancellation
import Transformer.GPTMini.Sparsemax.EmbeddingGram
import Transformer.GPTMini.Sparsemax.GramRouting
import Transformer.GPTMini.Sparsemax.GramRoutingEnergy
import Transformer.GPTMini.Sparsemax.GramSupport
import Transformer.GPTMini.Sparsemax.GramBoundary
import Transformer.GPTMini.Sparsemax.NormalizedGram
import Transformer.GPTMini.Sparsemax.NormalizedGramExamples
import Transformer.GPTMini.Sparsemax.InvertibleGram
import Transformer.GPTMini.Sparsemax.GramValues
import Transformer.GPTMini.Sparsemax.JointGramValues
import Transformer.GPTMini.Sparsemax.GramValueExampleGrams
import Transformer.GPTMini.Sparsemax.JointGramValueExamples
import Transformer.GPTMini.Sparsemax.MemoryGram
import Transformer.GPTMini.Sparsemax.MemoryValues
import Transformer.GPTMini.Sparsemax.ContextMemory
import Transformer.GPTMini.Sparsemax.PrefixMemoryCodes
import Transformer.GPTMini.Sparsemax.SharedMemoryValues
import Transformer.GPTMini.Sparsemax.SharedMemoryGeometry
import Transformer.GPTMini.Sparsemax.MemoryExampleGrams
import Transformer.GPTMini.Sparsemax.SharedMemoryExamples
import Transformer.GPTMini.Sparsemax.CodeMemoryOutputs
import Transformer.GPTMini.Sparsemax.CausalPrefixKeys
import Transformer.GPTMini.Sparsemax.CausalMemoryUniversality
import Transformer.GPTMini.Sparsemax.PrefixFeatureCodes
import Transformer.GPTMini.Sparsemax.PositionalMemoryTargets
import Transformer.GPTMini.Sparsemax.BoundedGramWidth
import Transformer.GPTMini.Sparsemax.PrototypeKernelCodes
import Transformer.GPTMini.Sparsemax.PrototypeKernelFeatures
import Transformer.GPTMini.Sparsemax.PrototypeKernelTraining
import Transformer.GPTMini.Sparsemax.PrototypePrefixMemory
import Transformer.GPTMini.Sparsemax.PrototypeKernelGeometry
import Transformer.GPTMini.Sparsemax.NearestPrototypeCodes
import Transformer.GPTMini.Sparsemax.PrefixNearestCodes
import Transformer.GPTMini.Sparsemax.NearestMemoryGeneralization
import Transformer.GPTMini.Sparsemax.PermutationMemoryGram
import Transformer.GPTMini.Sparsemax.LocalMemoryFeasibility
import Transformer.GPTMini.Sparsemax.LocalMemorySupport
import Transformer.GPTMini.Sparsemax.LocalJointMemory
import Transformer.GPTMini.Sparsemax.LocalNearestMemory
import Transformer.GPTMini.Sparsemax.LocalMemoryExamples
import Transformer.GPTMini.Sparsemax.LocalMemoryQuadratic
import Transformer.GPTMini.Sparsemax.LocalMemorySelection
import Transformer.GPTMini.Sparsemax.LocalRegularizedMemory
import Transformer.GPTMini.Sparsemax.IncidentNearestMemory
import Transformer.GPTMini.Sparsemax.IncidentMemorySelection
import Transformer.GPTMini.Sparsemax.ObservedMemoryPreferences
import Transformer.GPTMini.Sparsemax.PrefixObservedMemory
import Transformer.GPTMini.Sparsemax.ObservedMemoryExamples
import Transformer.GPTMini.Sparsemax.IncidentRegularizedMemory
import Transformer.GPTMini.Sparsemax.IncidentMemoryDescent
import Transformer.GPTMini.Sparsemax.EnergyMemoryLoss
import Transformer.GPTMini.Sparsemax.PeriodicMemoryRank
import Transformer.GPTMini.Sparsemax.OutputTiedMemoryDescent
import Transformer.GPTMini.Sparsemax.CompactAffineCapacity
import Transformer.GPTMini.Sparsemax.Uniform
import Transformer.GPTMini.Sparsemax.SelfRoute
import Transformer.GPTMini.Sparsemax.Clipping
import Transformer.GPTMini.Sparsemax.Failure
import Transformer.GPTMini.Sparsemax.RoutingLoss
import Transformer.GPTMini.Sparsemax.Certificate.Results

/-!
# Sparsemax training boundaries and convex learned-memory constructions

arXiv:2211.11052v1, §3.1 motivates causal saturation; the finite certificate
is separate from floating-point correctness. arXiv:1602.02068v2, §2.2,
Proposition 1 certifies thresholds and support windows; §3.2–§3.3's supplied
target positions do not establish latent attention learning. XSA,
arXiv:2603.09078v1, §2, is locally zero above epsilon on strict self routes;
the below-epsilon counterexample needs its norm hypothesis.

Derived bounded gaps and QKNorm gains ensure multiple active positions.
Nonzero active-pair directions reach an outer loss only if its derivative
distinguishes their values. Separated or spanning active value differences
prevent cancellation. Bounded anchors implement sparse transfers and correct
wrong squared outputs. Dedicated anchor channels remove the context-length
restriction for independent input directions. Fixed frames and value anchors
remain explicit assumptions; shared-row cancellation is a separate boundary.

Compatible shared-row endpoints with fixed values give an affine attention
path and loss `(1-t)^2 * initialLoss`. A fitting endpoint excludes positive
joint stationary points even at inactive ties. Repeated conflicting inputs
have a nonzero attainable minimum. Joint attainability is essential.

A PSD Gram learns both Q/K families with exact recovery and affine scores.
Unrestricted sparsemax graphs and joint value mixtures are nonconvex.
Probability input scores give affine actual attention, including changed
supports. A fixed small rank can fail at a feasible midpoint.

For one causal context, self-weight floors give a convex inverse domain;
joint Gram/output learning decodes one common value table. Outputs are affine
and convex criteria remain convex. Sharing contexts needs a data encoder.

A shared dictionary handles fixed probability data codes; genuine mixtures
give attention M times memory and a strict floor gives its inverse. A common
value table gives MZ outputs but leaves Gram unidentified. Distinct complete
causal signatures fit consistent arbitrary targets; identical prefixes share
outputs. Full signatures cost `(V+1)^T` slots.

A kernel on P observed distinct prefixes has a training-code right inverse,
from identity plus constant and squared-feature Grams. It fits arbitrary
prototype targets with genuine width 2P and convex joint output criteria.
This chart requires at least R slots for R independent target rows; its free
Gram has quadratic storage and dense routes.

Nearest codes are identity on registered prefixes. Masked Hamming ignores
future tokens. Unseen error is bounded by epsilon plus L times cover radius
under explicit fit, regularity and coverage; unconditional bounds fail.

A compact path learns 3P-1 edge/norm coordinates. Its affine genuine Gram
permits independent Q/K norms; strict floors ensure the inverse. Actual
nearest queries have at most three routes, including unseen witnesses.
Shared values retain MZ outputs, arbitrary prototype fitting, conditional
unseen bounds and convex joint criteria. Possible connections and same-family
orthogonality are fixed. Width and prototype count grow; FFN is deferred.

Separate incident-edge budgets enlarge the compact domain beyond its former
global budget. Nonnegative score-weighted outer products prove PSD without
nonnegative global identity mass. Affine geometry, three-route support,
inverse, variable Q/K norms and conditional unseen bounds survive.

Observed distances and feature energies supply a unique constrained geometry
reference. Positive weight gives finite descent and parameter-error bounds;
prefix masking ignores hidden continuations. This additional criterion
selects attention and norms; output-only error in that chart does not.

A new affine block PSD constraint couples attention B to learned outputs Z.
It is exactly an attention-weighted energy budget on original values B⁻¹Z.
With a prescribed aggregate edge budget, ordinary output squared error
remains jointly convex and selects opposite nonidentity attention supports
for two three-slot answer patterns. Every minimum identifies attention;
the wrong allocation leaves error at least 27/512 after relearning values.
The constraint restricts attainable outputs and leaves Q/K norms free.
This active-energy example is not a language-model guarantee or solver.

Three periodic key classes and a structural local score mask realize every
feasible path at fixed physical width three, with Q/K squared norms at most
three/four. Both Q and K change; actual masked sparsemax, common-value inverse,
joint convex energy domain and categorical forward are proved. The former
normalized Gram requires width at least P; the new physical Gram differs,
and masked attention has full rank despite QK score rank at most three.

Every better feasible endpoint gives joint descent; convex ordinary criteria
have no nonglobal constrained local minima for any finite observation table.
Affine sharing of local edges with learned adjacent output coordinates
removes residual intrinsic freedom. With all prototypes observed, ordinary
squared error is strictly convex for arbitrary targets, with complete
coordinate curvature controlled by `1+4*gain^2`, independent of P. Opposite
nonidentity answer fits and their support-changing midpoint remain feasible.

Nonempty energy domains are compact; arbitrary ordinary answer tables have
an attained unique tied minimum, including positive-error targets. Sharp
suboptimality bounds full parameter distance by `1+4*gain^2` times ordinary
loss gap. Midpoint gain is distance over `4*(1+4*gain^2)`; every other feasible
point descends toward the proved optimum. Its selector is noncomputable;
no numerical solver is implemented.

A generated quadratic edge profile preserves constant and affine position
features. Its original common values need two coefficient rows per output
channel, independent of P, and a query evaluates only its local destinations.
A fixed readout of the learned slope generates geometry. Coefficient caps
and a scalar interval replace the global energy PSD constraint; strict
ordinary curvature, unique attained training and finite descent survive.
This exact affine response family cannot memorize arbitrary answers: every
universal affine scalar decoder needs at least P coefficients. A three-slot
nonlinear target has sharp positive best error 2/3 in the compact model.
-/
