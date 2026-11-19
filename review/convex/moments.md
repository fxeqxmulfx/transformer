# Coupled class moments and learned sequence models

Moved unchanged from `review/flat_spectrum.md` on 2026-11-19: *Coupling complex class
moments* and CM1–CM120. They continue [`channels.md`](channels.md) with shared class
distributions under spectral bounds, categorical decoders and the discovery of their codes,
fixed input programs, and learned sequence models with hinge, recurrent, rotary and retrieval
components. New CM sections go here.

The labels CR, CC, B, MF, CFt, CF, O, CI and SB are in [`channels.md`](channels.md), CCa in
[`../convex_causal.md`](../convex_causal.md) and CD in [`../convex_depth.md`](../convex_depth.md).
Theorem A, Theorem E, R, AR, WD, XE and the other earlier labels are in
[`../flat_spectrum.md`](../flat_spectrum.md).

## Coupling complex class moments

### CM1. One distribution, many spectral bounds

The SB1 pilot could fit observed phase labels and still leave weak, incorrectly
oriented unseen phases. Replace the independent complex entry at each input
cell by a shared class distribution `P_ijc >= 0`, `sum_c P_ijc = 1`. For fixed
complex class codes define `M_f(P)_ij = sum_c P_ijc phi_fc` and impose
`||M_f(P)||op <= R_f`. These are convex upper bounds on linear images, so their
intersection with the probability simplexes is convex. Unit codes imply the
old entry-disk bounds automatically. For centered codes and nonnegative radii,
uniform class probabilities give a feasible starting point.

`CoupledMoments/Basic.lean` proves these statements. This changes the hypothesis
class, rather than asserting that extra optimization resolves SB3's ambiguity.
The radii must still be specified independently of missing labels.

### CM2. Ordinary cross-entropy and an attainable lower bound

Use `P_ijc` as the logits at scale one. The loss is ordinary full-vocabulary
cross-entropy, `log(sum_c exp(P_ijc)) - P_ijy`, not the negative logarithm of a
probability coordinate. It is jointly convex. Among probability-vector logits,
the one-hot vector for the correct label minimizes each observed CE term, with
value `log(exp(1) + V - 1) - 1`. Consequently any feasible table matching all
observed one-hot labels is a global minimum of the same empirical objective.
This can certify a forward-only feasibility method without evaluating gradients.

`CoupledMoments/Loss.lean` proves convexity, the ideal-loss lower bound, the
minimum certificate with explicit feasibility, and the bound
`loss(P) - loss(Q) <= loss(P) - idealLoss` for every feasible competitor Q.
The numeric excess is useful even if a particular iterative solver has no
proved finite-step rate. None of these claims predicts an unobserved label.

### CM3. Full Fourier readings retain labels and exclude inconsistent moments

Choose `phi_fc = exp(2 pi i f c / V)`. The complete Fourier matrix has Gram
matrix `V I`; multiplying by its adjoint recovers all class coordinates. For
probability tables the zero mode is already one, so the nonzero modes suffice.
These fixed codes use only the vocabulary identifiers, never a task operator.
The probability parametrization is thus equivalent to coupled complex Fourier
coordinates, not a set of freely trainable independent complex matrices.

The coupling is a strict restriction. With four equally spaced classes, a
distribution whose first moment equals +1 has all its mass on class zero and
therefore second moment +1. Independent unit disks would also permit the
inconsistent pair (+1,-1). `CoupledMoments/Consistency.lean` proves injectivity,
the Fourier specialization, and this explicit counterexample. A stronger
constraint alone is not a proof of recovery on random half-observed tables.

### CM4. A feasible output from a common mixture

For centered codes, mix a probability table with uniform mass:
`P' = alpha P + (1-alpha) U`. If `0 <= alpha <= 1`, it remains a probability
table and `M_f(P') = alpha M_f(P)`. Thus choosing one alpha satisfying
`alpha ||M_f(P)||op <= R_f` for every f makes the result feasible. If alpha is
positive, all strict class comparisons are unchanged, including ties under a
fixed argmax rule. `CoupledMoments/Blend.lean` proves the scaling, feasibility
and comparison statements. An SVD-based implementation must still report its
numerical error and the actual mixture coefficient; this is not a verification
of floating-point norm estimates or of optimizer convergence.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments
LAKE_ARTIFACT_CACHE=false lake env lean scripts/CoupledMomentsAxioms.lean
```

### CM5. Convex bounds on largest and total singular values

The shared-moment pilot still fails on addition. Proposition 2 gives a reason
to distinguish spectral shapes: the addition phase matrix is rank one with
nuclear norm V, whereas a full flat square matrix has V singular values of
size sqrt(V). A bound on the largest singular value alone can exclude the
former or leave its structure unrestricted. Add the convex upper bounds
`nuclearNorm(M_f(P)) <= B_f` alongside `||M_f(P)||op <= R_f`.

`CoupledMoments/Schatten/Basic.lean` proves convexity of this intersection,
uniform feasibility for centered codes and nonnegative budgets, and the same
global CE-minimum certificate for a feasible exact observed fit. The common
mixture of CM4 scales both norms, so one positive coefficient enforcing both
budgets gives a feasible output without changing strict class comparisons.
The budgets are fixed model parameters; no product of learned factors, rank
equality or task-specific arithmetic operation is introduced.

For prospective tests, two generic square-matrix shapes are `(R,B) =
(sqrt(V), V sqrt(V))` and `(V,V)`. The former adds only a redundant nuclear
budget to the previous operator ball, since each of the V singular values is
at most R. `Schatten/NormComparison.lean` and `budget_at_card_iff` prove this
redundancy. The latter allows a large leading singular value while restricting
their sum. These are alternative constraints, not a claim that every target
belongs to either shape. The nuclear-norm failures in XE15/SB2 remain valid;
a sum bound alone is not a universal replacement for the operator bound.

### CM6. A rank-one missing-entry recovery certificate from actual CE

For `X(z) = [[1,1],[1,z]]`, the established 2x2 formula gives
`nuclearNorm(X(z))^2 = 3 + |z|^2 + 2 |z-1|`. Since
`|z-1| >= 1-Re(z)`, a nuclear budget of two implies
`(Re(z)-1)^2 + Im(z)^2 <= 0`, hence z=1. Conversely X(1) has nuclear norm
two. `Schatten/Recovery.lean` proves this equivalence and the corresponding
recovery of the missing binary probability coordinate from three observed
one-hot labels. There is no rank constraint hidden in the feasible set.

`Schatten/Optimality.lean` removes correct observed labels as an assumption:
each ordinary CE term is at least its one-hot value, so a summed loss no larger
than the ideal sum forces equality term by term. With another class present,
the strict CE inequality of CR3 forces each observed target mass to one. For
the three observed cells `(0,0), (0,1), (1,0)`, the nuclear budget then gives
the unseen target mass at `(1,1)`. The all-true table satisfies every hypothesis.
This is a concrete repair, not a theorem for arbitrary observation patterns
or convergence of the native iterative solver.

### CM7. One convex class for several fixed spectral budgets

Let D_k be the feasible probability tables for each member of a fixed family
of `(R_k,B_k)` budgets. Use `convexHull(union_k D_k)` as the model domain in
the direct probability coordinates. The hull is convex, all its rows remain
probabilities, and the ordinary empirical CE remains convex. This definition
does not train a bilinear product of free mixture weights and free tables.

`Schatten/Hull.lean` proves these properties and a useful optimization
certificate: any feasible exact observed fit from any D_k is a global CE
minimum over the entire hull. Thus a procedure can train candidates with
different budgets and certify a solution using training observations alone.
If none reaches the ideal lower bound, choosing the best candidate does not
prove that the full hull problem has been solved. Nor does an exact fit select
unseen answers uniquely; all previous ambiguity qualifications still apply.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SchattenMomentsAxioms.lean
```

### CM8. Causal and spectral predictions in one convex class

The pair-moment pilot does not by itself cover the sequence tasks. To preserve
the causal count model while admitting shared complex moments, express each
expert as a probability-valued function on the same visible input domain.
Take the convex hull of the fixed expert families in these direct prediction
coordinates. Every original expert remains feasible, every row remains a
probability vector, and ordinary empirical CE remains convex. This is not a
bilinear parameterization by free expert weights and free expert parameters.

All experts for a given dataset use the same fixed scale s >= 0. Their logits
are s times the probability coordinates. The one-hot logits have the common
per-observation lower bound `log(exp(s) + V - 1) - s`. The relative-logit
comparison proving CM2 also proves this scaled bound. An expert that fits all
observed one-hot labels therefore globally minimizes ordinary CE over the
whole prediction hull; for any other feasible competitor, its objective
excess is at most the measured ideal-loss excess. `ScaledLoss.lean` and
`PredictionHull.lean` prove these statements with nonempty examples at s=10.

The empirical objective sums ordinary teacher-forced CE over observed prefixes.
Own-output generation and residue pooling remain evaluation rules, not the
training objective. Input-only eligibility or fixed scale choices do not
train extra parameters. Selecting the first training-certified candidate is
again certificate search: without an ideal fit it need not optimize the hull
interior. Even a certified optimum need not identify missing answers. New
experts, unsupported input shapes, failed fits and compute for every candidate
must all be reported in the full stress suite.

### CM9. The two endpoint budgets leave a capacity gap

CM5 explicitly did not assert that every target belongs to one of its two
example shapes. A four-by-four binary moment makes the gap concrete:

```
H = [[1,1,1,1], [1,1,1,1], [1,1,-1,-1], [1,1,-1,-1]].
```

This is the fixed sign-code moment of a one-hot probability table. Its Gram
matrix satisfies `(H*H)^2 = 8 H*H`, its Frobenius energy is 16, and its rank
is two. Hence its operator norm is `sqrt(8)` and its nuclear norm is
`2 sqrt(8)`. The size-four full-rank budget `(R,B)=(2,8)` excludes it by
operator norm; the rank-one budget `(4,4)` excludes it by nuclear norm.
The fixed intermediate budget `(sqrt(8),2 sqrt(8))` admits the same table.

Mixing the two old spectral families does not repair this example. One-hot
tables are extreme points of the product of simplexes, so the convex hull of
a probability-valued family contains such a table only if that family already
contained it. `Schatten/Vertices.lean` proves this general fact, and the witness
is therefore excluded by CM7's two-endpoint spectral hull too. This does not
exclude the larger CM8 hull when it contains an unrestricted lookup expert.

`Schatten/Intermediate.lean` proves the actual norms and both exclusions,
using a general operator-norm consequence of the Gram-square identity.
The witness has a binary output vocabulary and four possible values of each
input; it is a representation counterexample, not a result on a named stress
dataset. It motivates additional generic fixed spectral budgets under CM5
and CM7. The new bounds remain convex upper bounds, with no learned product,
rank equality, arithmetic rule or missing label in the domain. Inclusion of
a target is necessary capacity, not a proof that CE training will recover it.

### CM10. Feasible uniform completion and a sharing remedy

Repairing CM9's capacity gap does not force missing labels. Observe all rows
of columns zero and two of its table, leaving columns one and three unseen.
Keep the observed one-hot probabilities and put uniform binary probabilities
in the unseen columns. Its moment is H with the unseen columns zeroed. Its
operator and nuclear norms are two and four, below the intermediate bounds
`sqrt(8),2 sqrt(8)`. It fits every observed label and therefore globally minimizes
ordinary CE on that feasible set, although an unseen target coordinate is
only one half. `Schatten/Completion.lean` proves the norms, feasibility, exact
fit and actual minimum together. This explains a failure that extra optimizer
steps alone need not repair; it is not an assertion about a particular random
stress split or a floating-point checkpoint.

One repair is to supply correct shared coordinates. Fix a column representative
map r, and impose `P(i,j)=P(i,r(j))`. These are linear equalities and form a convex
set. At the observed CE lower bound, any row sharing an observed representative
equals that observation's one-hot target. `Schatten/Sharing.lean` proves this
conditional statement. For CM9's table, r maps columns zero/one to zero and
two/three to two; every representative is observed, and the same empirical CE
minimum now determines the entire table. The target satisfies every hypothesis.

The representative map is fixed in this convex problem. Agreement on a few
observations alone does not prove that two input values are interchangeable
on unseen inputs. A future training-only grouping procedure must preserve all
original observation multiplicities, test its proposed consistency, retain
unshared controls, and evaluate missing answers separately. If it searches
over fixed maps, the certificate concerns each conditional convex model or
their direct prediction hull; it is not a convexity theorem for freely learned
products of a map and a table, nor a guarantee that the selected sharing rule
matches unknown ground truth.

More generally, fix row and column group maps rho and kappa, learn Q on their
grouped indices, and predict `P(i,j)=Q(rho(i),kappa(j))`. Impose the same convex
spectral budgets on Q. `Schatten/Quotient.lean` proves that the lifted family
is convex, remains probability-valued, and retains the exact-fit ordinary-CE
minimum certificate. Its loss is still summed over every original observation;
multiple observations mapped to one grouped cell retain their multiplicities.
This is a fixed linear lookup of learned coordinates, not a learned product
of a grouping matrix and Q. The conditional statement does not prove that an
adaptive grouping search returns a correct map or an identifying completion.

For a fixed row/column map, `quotient_observed_group` also derives the complete
one-hot prediction on every input sharing its grouped coordinates with an
observed example, directly from attaining the original CE lower bound. The
inhabited example observes only one of two original columns and recovers the
other through their fixed shared coordinate. A missing-answer audit can thus
separate groups already covered by training from wholly unobserved groups
requiring further completion. Correctness of the copied label on held-out
ground truth is a separate check of the proposed map; the theorem does not
infer it from observed agreement alone.

### CM11. Certify a sharper feasible endpoint without changing decoded answers

Accurate argmax answers can coexist with a small observed CE gap at a numerical
iteration cap. In a larger prediction hull, a memorizing candidate attaining
ideal training CE would then win a training-loss-only comparison. One generic
endpoint check is to take the model's own decoded labels, make their active
grouped coordinates one-hot, and keep inactive padding uniform. This reads no
held-out labels. Decoding itself is not a convex map, and its resulting table
need not satisfy the spectral bounds.

Let H be this decoded table and choose a common `0 < alpha <= 1`. With centered
class codes, CM4 and CM5 give `M_f(blend(alpha,H)) = alpha M_f(H)`. Checking
`alpha ||M_f(H)||op <= R_f` and `alpha ||M_f(H)||_* <= B_f` for every f therefore
certifies the same convex feasible domain. Every active decoded class remains
strictly above the alternatives. The operator and nuclear norms must be checked
after decoding; the old soft table's norms do not suffice.

If the decoded labels agree with every original observed label, H attains ideal
observed CE. Convexity of the actual ordinary CE then gives, against any
probability-valued competitor Q,

```
L(blend(alpha,H)) - L(Q) <= (1-alpha) (L(uniform) - L_ideal).
```

The observation index still includes all original repetitions. `Schatten/Sharpen.lean`
proves probability feasibility, strict decoded comparisons and this certificate,
with a nonzero complex-code example. It does not prove a floating-point norm
computation, that sharpening improves CE, or any missing-label correctness.
The native follow-up must keep the original endpoint when its training CE is
already ideal or the checked candidate does not improve it, record the new
norms and mixture, and verify that all decoded predictions are unchanged.
This checks another point in the same convex family; it does not change the
family, the ordinary loss, or the frozen earlier training runs.

### CM12. Remove dependence on the arbitrary order of output class names

Output relabeling has a different effect from permuting input rows or columns.
For a class permutation sigma define `(T_sigma P)(i,j,c) = P(i,j,sigma⁻¹(c))`.
This real-linear map preserves all probability rows and satisfies

```
CE(scale * T_sigma P(i,j), sigma(y)) = CE(scale * P(i,j), y),
M_phi(T_sigma P) = M_(phi composed with sigma)(P).
```

The first equality preserves the complete vocabulary and softmax denominator.
It also preserves the original empirical CE sum, including repeated observations.
The second equality changes the class code; it need not be a unitary change of
input coordinates. Consequently a spectral bound in a fixed class coding is
not generally invariant under renaming the target labels.

Here is an exact one-code counterexample. Take four class phases
`phi = [1, i, -1, -i]` and the point-mass label table

```
0 0 0 0
0 1 2 3
0 2 0 2
0 3 2 1
```

Its complex moment is twice the normalized four-point Fourier matrix, with Gram
matrix `4 I` and operator norm 2. Swap class names 1 and 2. The new moment applied
to `[1,1,1,1]` is `[4,0,2+2i,0]`; squared input energy is 4 and squared output
energy is 24. An operator bound of 2 would require output energy at most 16.
Both tables remain full class distributions, and their correspondingly renamed
target CE values agree exactly. This example uses **one** centered unit code.
It does not claim feasibility of the first table under every nonconstant
four-class character, nor establish the cause of the prime-vocabulary native
failures. Those are separate empirical and domain-specific questions.

For any original family A define the following domain in direct prediction
coordinates:

```
A_sym = convexHull (union over every class permutation sigma of T_sigma(A)).
```

This domain is convex, contains A, and is invariant under every output relabeling.
If every member of A has probability rows, every member of A_sym does too.
Fixed-scale ordinary CE is convex on A_sym. The existing CM8 ideal-loss and
excess-loss certificates therefore apply with these renamed families as the
generators. The concrete excluded table above belongs to A_sym, proving that
this enlargement can be strict. `CoupledMoments/Relabeling.lean` proves the
general identities and convex-family properties;
`CoupledMoments/RelabelingObstruction.lean` proves the exact counterexample.

Each generator retains the original spectral bounds in its corresponding class
coding. A mixture of generators **need not satisfy those same bounds in one
fixed coding**. This is an enlargement of the feasible family, not an invariance
theorem for the old fixed-code ball. The theorem also supplies neither an
efficient oracle over the factorially many codings nor a jointly convex
parametrization of a trainable permutation. A native follow-up may propose
codings from training observations and compare their actual CE, but finite
proposal search alone does not certify the entire new domain. No conclusion
about missing labels follows without further structure and coverage.

### CM13. Graph coverage and coded structure identify the complete CE optimum

CM6 establishes recovery of a single missing entry in a two-by-two example.
The existing Theorem E gives a general route from the same convex nuclear
budget to recovery on an entire square input table. Let the input dimension be
N and let the observed cells form a bipartite graph between its rows and columns.
Assume that graph is connected and its observed cells contain a perfect matching.

For unit-modulus vectors a and b, Theorem E says that `K = a bᵀ` is the unique
minimum-nuclear-norm interpolant of its observed entries, and `||K||_* = N`.
Consequently any interpolating X with `||X||_* <= N` equals K: K's minimality
places its norm below every other interpolant, while the budget places X's
norm below K's. Thus X is also a minimizer, and uniqueness applies. The learner
needs only the convex upper bound, not a rank equality or trained factors.

Now let `phi : F × C -> complex` be a complete code bank with
`phi* phi = rho I`, `rho != 0`. Suppose a full label table y has the explicit
coded structure

```
phi(f, y(i,j)) = a_f(i) b_f(j),
|a_f(i)| = |b_f(j)| = 1.
```

Learn probability rows P under the bounds `||M_f(P)||_* <= N`. If their actual
ordinary observed CE attains the one-hot lower bound, CM6 forces every observed
target probability to one. Each moment therefore interpolates its observed
entries of `a_f b_fᵀ`. The graph argument recovers every entry of every moment.
CM3's injectivity of the complete code bank then gives

```
P = pointMass(y)
```

on **all** input cells, including unobserved ones. Repeated observations remain
in CE; the graph uses their set of cells. The constant Fourier mode requires
no new constraint: probability rows already make it the all-ones matrix,
whose nuclear norm is N. The native nonconstant-mode budget therefore supplies
the relevant bounds. CM12's coordinate covariance permits applying the result
in a fixed renamed output coding and returning the probabilities to their
original class order.

`CoupledMoments/Schatten/GraphRecovery.lean` proves the budget-to-uniqueness step,
the automatic constant-mode bound and the complete probability-table theorem.
Its inhabited two-class example observes three cells of a nonconstant table,
has a connected graph and an observed perfect matching, and leaves `(1,0)`
unobserved. Both classes occur among its observed targets.

The target's coded separability is a structural hypothesis about the complete
table. Observed phase consistency alone does not prove that unknown ground
truth satisfies it. A training-only graph/coding audit can instead construct a
compatible separable completion, check connectivity and a matching, and compare
saved predictions with that model completion. Agreement between the completion
and held-out ground truth remains an empirical check, never an input to the
proposal. Nor does this exact theorem turn a floating-point CE tolerance into
exact equality: quantitative stability and finite-step solver convergence are
separate claims. Connectivity by itself, without the matching and coded
structure, does not provide this guarantee.

### CM14. A training-compatible phase completion is a feasible CE minimizer

CM13 identifies an ideal endpoint but does not construct one. A discovered
categorical coding supplies a direct candidate in the same convex family.
Let `e : C <-> Z/q` be a bijection of the full class vocabulary, with `q > 0`,
and let `r(i)` and `s(j)` be row and column phase potentials. These potentials
are learned from the observed equations

```
e(y_t) = r(i_t) + s(j_t) mod q.
```

Input symbols enter as categorical row and column indices; no numerical value
or arithmetic operation on an input symbol appears in this construction.
Define the model completion and its full complex code bank by

```
z(i,j) = e^{-1}(r(i) + s(j)),
phi(f,c) = exp(2 pi i f e(c) / q).
```

The one-hot table `P_z` has probability rows. Every coded moment factors as
`a_f b_f^T`, with `a_f(i) = exp(2 pi i f r(i)/q)` and the analogous `b_f(j)`.
Both factors have unit-modulus entries, so its operator norm is at most the
input dimension N and its nuclear norm equals N. Thus `P_z` is feasible for the
existing radius-N, nuclear-budget-N family. The separate `full_rank` control
uses the smaller radius `sqrt(N)` and is not covered by this construction. The
observed phase equations make its actual ordinary CE equal the one-hot lower
bound. It is therefore a global CE minimizer in that fixed-code convex family,
without requiring an iterative optimizer to discover this endpoint.

The complete code bank satisfies `phi* phi = q I`, including after a categorical
renaming. If the observation graph is connected and contains a perfect matching,
CM13 further identifies this candidate as the only ideal probability table under
the nuclear budget. This conclusion concerns the compatible model completion.
It does not prove that unobserved true labels follow the learned phase equations.
Discovery of a coding remains a discrete proposal step, outside the fixed-code
convexity claim. A finite list of such candidates is not a global oracle over
the CM12 hull of all output permutations.

`CoupledMoments/Schatten/PhaseCandidate.lean` proves the renamed Fourier Gram
identity, the explicit moment factorization, feasibility and global CE
optimality. It specializes CM13 to show uniqueness with only the nonconstant
nuclear bounds supplied as hypotheses; probability rows provide the constant
mode. Its nonconstant two-class construction has an inhabited three-observation
CE-optimality example, and the uniqueness theorem has an inhabited complete-graph
example.

A native warm initialization may use this candidate only after checking the
training equations and class bijection, and must retain explicit rejection and
the unchanged cold-initialization controls. Numerical feasibility, original
split metrics, costs of coding and potential discovery, and any subsequent
gradient updates must still be independently audited. The theorem is an
exact-real construction; it does not certify floating-point SVDs or unseen-label
accuracy, and does not change the frozen CM12 or depth cohorts.

### CM15. Decode spectrally bounded latent states into the original vocabulary

The bijection in CM14 excludes some deterministic tables, independently of
the optimizer. If three corners of a rectangle have label `a` and the fourth
has a different label `b`, no injective additive class code can represent it:
the first column forces the two row phases equal, while the second column
would then force the two distinct labels equal. This is a categorical capacity
obstruction, without any interpretation of the input indices as numbers.

Let `L` be a finite latent vocabulary, `C` the original output vocabulary and
`d : L -> C` a fixed decoder, possibly many-to-one. Learn the full latent
probability table `P(i,j,l)` under the existing fixed complex operator/nuclear
bounds, and expose the original-vocabulary probabilities

```
Q(i,j,c) = sum_{l : d(l)=c} P(i,j,l).
```

This map is real-linear and preserves probability rows. Its image of the
latent budget domain is convex, and ordinary original-vocabulary CE is convex
in every latent coordinate. A feasible latent table whose decoded rows fit
the observed labels attains the same one-hot CE lower bound for vocabulary
`C`, even if some original output classes have no latent preimage. CE remains
log-sum-exp over every original class, with logits `Q`; it is not a likelihood
over a reduced observed-class set. The identity decoder gives back the old
model. The decoder and code bank are fixed when taking this convexity claim;
choosing them from data remains a proposal step, not a jointly convex search
or a certified global oracle over all decoders.

The whole prediction class can include those choices convexly: take the convex
hull, in original probability coordinates `Q`, of the decoded feasible sets
over all `d : L -> C`. Every point is still a probability table and ordinary
CE is convex on this hull. Any feasible decoded candidate that reaches the
observed one-hot CE bound is also a global CE minimum over the entire decoder
hull. Thus successful proposal search can have a global prediction-space CE
certificate even though its discrete search is not itself convex. This does
not supply a polynomial-time oracle, guarantee finding such a candidate, or
identify unseen labels. No independent learned decoder/latent factors are
multiplied in the prediction-coordinate optimization claim.

In particular, CM14's latent phase table with any `d : Z/q -> C` is feasible
at radius and nuclear budget `N`. Observed equations
`d(r(i_t)+s(j_t))=y_t` suffice for decoded global CE optimality. The spectral
bounds constrain the latent moments; they must not be presented as unchanged
bounds on the decoded output moments.

This strictly increases the available deterministic phase constructions.
Take two row and column phases `0,1` in `Z/3`, and decode `0,1` as `true` and
`2` as `false`. The decoded table has three `true` corners and a `false`
fourth corner. Its latent moments satisfy the original `N=2` rank-one budgets.
The direct binary sign moment cannot satisfy nuclear budget two, by CM6.

The stronger capacity has an explicit identification limit. With the same
decoder, latent code and budgets, constant zero phases also fit those three
observed `true` labels perfectly but predict `true` at the fourth corner.
Both decoded tables are global ordinary-CE minima. Hence CM13/CM14's recovery
conclusion cannot simply be transferred from observed latent labels to
observed decoded labels. This three-edge observation graph is connected and
has the off-diagonal perfect matching, so graph coverage is not the missing
hypothesis. A successful native proposal still needs original
held-out evaluation; a uniqueness claim requires additional identification
evidence. No native accuracy result or efficient decoder-discovery guarantee
is asserted by this extension.

`CoupledMoments/Schatten/Decoded.lean` proves real-linearity, probability-row
preservation, the pulled-back moment identity, convexity, identity-decoder
equivalence and the actual CE fitting certificate.
`CoupledMoments/Schatten/DecodedPhase.lean` proves the latent phase construction,
the direct-code obstruction, the admitted counterexample and its two distinct
CE-optimal completions, including connected matching coverage.
`CoupledMoments/Schatten/DecodedHull.lean` proves probability preservation,
convex CE and the global fitting certificate across all decoder choices.
It also proves the two ambiguous endpoints remain global CE minima over this
larger hull: convexifying the class does not supply the missing identification.

### CM16. Equal probabilities on opposite latent states

*Source-note extension for board task #195, 2026-11-12. The Lean modules were
drafted by the task's first claimant and completed by its second.*

CM15 leaves the decoder arbitrary. One fixed class of decoders needs no search
over latent signs. Let the latent vocabulary be `Z/q` and let the decoder be
even,

```
d(-z) = d(z).
```

**Reflection.** `(rho P)(i,j,z) = P(i,j,-z)` is real-linear and permutes the
coordinates of every probability row. On the Fourier bank of CM14 it exchanges
frequencies,

```
M_f(rho P) = M_{-f}(P),
```

so operator and nuclear budgets that are the same for every frequency survive
it. The average `pi P = (P + rho P) / 2` then satisfies the same budgets by
convexity, has equal probabilities on opposite states,

```
P(i,j,z) = P(i,j,-z),
```

and decodes to the same table: `decode d (pi P) = decode d P`.

**The paired domain.** Add these linear equalities to the latent domain of
CM15 and decode. The set of predictions is convex, and ordinary CE over the
original vocabulary is convex on it. For an even decoder the equalities remove
no prediction: the paired domain is exactly the decoded domain of CM15 with
the same constant budgets. The symmetry is a restriction of the latent
coordinates, not of the predictions.

**A symmetric phase witness.** For phases `r, s : n -> Z/q` put half of the
probability at `r(i) + s(j)` and half at its negative (all of it if the two
coincide). This table is `pi` of CM14's one-hot table. It is feasible at
radius `N` and nuclear budget `N`: it is the average of two feasible tables,
and no rank-one factorization of its own moments is used. It decodes to the
deterministic table `d(r(i) + s(j))`. If
that table shows the observed labels, it is a global ordinary-CE minimum over
the paired domain, and over the whole decoder hull of CM15. One latent state
per observed label is not chosen anywhere.

**A decoder from the vocabulary alone.** For `k + 1` classes take `q = 2k + 1`
and fold opposite states,

```
d(z) = min(z, q - z)        in {0, ..., k}.
```

It is even, every class has a preimage, and the fiber of the class `c` is
`{c, -c}`. Input side `N`, latent states `q` and output classes `k + 1` are
three separate numbers. Example: `N = 3`, `q = 5`, three classes, row phases
`i` and column phases `-j` give the table

```
0 1 2
1 0 1
2 1 0
```

The permutation-phase constructor of `experiments/convex_stress_decode` takes
as many latent states as the input side and a permutation of them along the
columns. Every row of such a table has the same number of cells of each
class, for every decoder, and convex mixtures keep these equal row masses.
The table above has one cell of class 2 in its first row and none in its
second: it lies outside the hull of those constructions with three states,
and inside the paired domain with five.

**The limit.** The symmetry does not identify unseen labels. Decode `Z/3` by
`z != 0`, and observe `false, true, true` at the cells `(0,0), (0,1), (1,0)`.
The phases `i + j` and `i - j` both fit. Their tables are feasible in the same
paired domain and both are global CE minima, with `true` against `false` at
the unobserved cell `(1,1)`. The observed graph is connected and contains a
perfect matching. Recovery needs the identification conditions of a separate
argument; capacity and an observed fit do not give it.

Not claimed: that an iterative solver reaches such a minimum, any numerical
tolerance, any accuracy on original data, or a way to find the decoder.

`CoupledMoments/Schatten/Paired.lean` proves the reflection identities, the
convexity of the paired domain and its equality with the decoded domain.
`PairedPhase.lean` proves the witness and the CE certificates.
`PairedCoding.lean` proves the properties of the folding decoder and the
three-class example. `PairedLimits.lean` proves the equal-row-mass
obstruction and the ambiguity.

### CM17. The class coding is found from the training rectangles

*Source-note extension for board task #196, 2026-11-12. The number CM16 is
reserved for task #195.*

CM14 builds a feasible CE minimizer from a class coding and CM13 identifies it,
but both stop before two steps: "Discovery of a coding remains a discrete
proposal step", and the conclusion "does not prove that unobserved true labels
follow the learned phase equations". This item proves both steps for one
family of tables and one explicit learner. It is not about a trained network.

**Setting.** A square table `y : n × n -> C` over a finite class vocabulary.
The structural hypothesis, the only assumption about unobserved cells, is

```
y(i,j) = e^{-1}(r(i) + s(j))
```

for some bijection `e : C <-> Z/p`, `p` prime, and phases `r, s : n -> Z/p`.
The learner is given none of `e`, `r`, `s`. It is given training samples
`(i_t, j_t, y_t)` with `y_t = y(i_t, j_t)`, possibly repeated. Rows, columns
and classes are names: no arithmetic on them enters, only arithmetic on the
values of a candidate coding.

**The linear system.** Four samples at the corners of a rectangle, with the
labels `a, b` in one row and `c, d` in another row at the same two columns,
give one homogeneous equation for an unknown coding `z : C -> Z/p`:

```
z(a) + z(d) - z(b) - z(c) = 0.
```

The solutions form a subspace that contains the constants and `e`. Pin one
class, `z(c0) = 0`. The pinned solutions contain `e - e(c0)`, which is not
zero, so the native rejection `no_nonconstant_phase_code` cannot occur for
such a table. The test of the learner is

```
the pinned solution space has dimension one,
```

which by rank–nullity is `rank + 1 = p` for the matrix of the pinned system:
the acceptance test of `coding.rs`. The rectangles through one shared column
of two rows give all rectangles of those rows. The native system keeps, for
two rows `row < other`, only those through their first shared column; it has
the same solutions when a cell has one label. The solution space depends only
on the set of observed (cell, label) pairs: not on the order of the samples,
their repetitions or their grouping into batches.

**What the test gives.** Assume the structural hypothesis and the test. For
every nonzero pinned solution `z`, whichever the elimination returns:

1. `z = α (e - e(c0))` with `α != 0`. The coding is recovered up to the affine
   gauge `e -> α e + β`, which does not change the table.
2. `z` is a bijection of the classes. This is derived, not assumed: the native
   bijection check cannot fail here.
3. `z` has row and column phases on the whole table, hence on the observed
   cells: the native cycle check cannot fail here either.

The test holds exactly when every solution of the rectangle system is
`α e + β`, and it does not depend on the pinned class.

**An observable sufficient coverage.** Take two rows and `p - 1` columns
observed in both, with distinct labels in the first row and one column where
the two rows differ. Then the test holds. Write `w = z ∘ e^{-1}` and let
`δ != 0` be the difference of the phases of the two rows. The rectangles of
the two rows say that `w(u + δ) - w(u)` is one constant `κ` at the `p - 1`
phases `u` of the shared columns. The sum of `w(u + δ) - w(u)` over all `u`
vanishes and `p κ = 0`, so the increment is `κ` at the last phase too. A
function on `Z/p` with a constant increment over a nonzero step is affine,
since the step generates the group. (With only `p - 2` shared labels and no
other rectangle, two increments remain free under one relation; this remark
is not formalized.)

**The completion is the table.** Let the bipartite graph of the observed
cells be connected, and let the test hold. Take any bijection `w : C <-> Z/p`
and potentials `ρ, σ` that fit the training samples,
`w(y_t) = ρ(i_t) + σ(j_t)`; a nonzero pinned solution with potentials found
by a search of the graph is one. Then

```
w^{-1}(ρ(i) + σ(j)) = y(i,j)        on every cell, observed or not.
```

Indeed `w - w(c0)` is a nonzero pinned solution, so `w = α e + β`. The
potentials `ρ, σ` and the gauged phases `α r + β, α s` then have equal sums
on the observed cells, and on a connected graph equal sums everywhere (the
rigidity of the Setting, for potentials in `Z/p`). So under the two
observable conditions the table is the only training-compatible phase
completion in the sense of CM14.

**The certificate.** What the native code returns can be checked without
trusting how it was computed: a bijection `w`; potentials `ρ, σ` with
`w(y_t) = ρ(i_t) + σ(j_t)` at every sample; connectivity of the observed
graph; and `rank + 1 = p` for the pinned system. All four are finite checks
on the training set and the returned objects. They are sound: under the
structural hypothesis they give `w^{-1}(ρ(i) + σ(j)) = y(i,j)` on every cell,
which is the statement above. They are complete for the elimination: under
the hypothesis and the rank test every nonzero pinned solution passes the
bijection check and has such potentials (points 2 and 3).

**The learner.** `discover(cells, labels, c0)` is a function of the training
set and the pinned class alone. In the order of the native checks it answers
`no_code` (the pinned solutions have dimension zero), `ambiguous` (above one),
`nonbijective` (no pinned solution is a bijection), `inconsistent` (no
bijective coding has potentials on the samples), `disconnected`, or else
`accept` with a training-compatible phase completion. For a table of the
family:

- under the test and connectivity it returns `y`, on all cells, and it
  accepts in no other case;
- it is never wrong: it returns `y`, or answers `ambiguous` or
  `disconnected`. The other three refusals certify that the labels belong to
  no table of the family;
- its answer depends only on the set of observed (cell, label) pairs.

Without the structural hypothesis an accepted table is still a
training-compatible phase completion, and nothing more is claimed for it.

**The native checks.** The learner is stated by what the native checks
compute, and it is not the native code. The differences:

- The native elimination keeps the rectangles through the first shared column
  of two rows `row < other`. For every rule that gives two such rows with a
  shared column a column that both show, the kept rectangles have the same
  solutions as all rectangles, when a cell has one label.
- A repeated cell is an error of the native code. The answer of the learner
  does not depend on repetitions.
- The native code refuses a training set that does not show every class
  (`unobserved_output_classes`) before the elimination. The learner has no
  such answer: for `p != 2` such a training set of a table of the family
  fails the rank test, and the answer is `ambiguous`. (For `p = 2` a training
  set that shows one class passes the rank test; with a connected graph its
  table is constant. This remark is not formalized.)
- The native constructor refuses a training set without a perfect matching
  (`NoPerfectMatching`): it admits the case in which CM13 gives one
  minimizer. The learner needs no matching to return the table; the matching
  enters only the uniqueness of the CE minimizer below.
- The native code refuses a composite vocabulary; the modulus of the learner
  is prime.

**The bridge to CE.** Use the coding `w` found above as the code bank of CM14:
probability rows `P`, operator norm and nuclear norm of every moment at most
`N`, ordinary log-sum-exp CE over the full vocabulary on the training samples.
Under the test and connectivity the one-hot table that the learner builds
from `w, ρ, σ` is the one-hot table of `y`, it is feasible, and it minimizes
CE: the minimizer is constructed in closed form, not searched. If the
observed cells also contain a perfect matching, CM13 applies to `w` with its
matching hypothesis as stated and its code separation `φ* φ = p I` proved in
CM14, and the program has exactly one minimizer, `pointMass(y)`. A minimizer
therefore gives probability one to `y(i,j)` at every cell. The same holds
with `w` replaced by any nonzero pinned solution and the bijection proved for
it in point 2.

These are statements about exact minimizers over the reals. A table whose CE
is within the native `1e-8` of the ideal value is not covered: no stability
bound is proved here. No cold or minibatch run of the native solver is
claimed to reach the minimizer.

**Witness.** `p = 5`, classes named `0..4`, and the hidden coding exchanges
the names `0` and `1`; `r(i) = i`, `s(j) = j`. The table is

```
1 0 2 3 4
0 2 3 4 1
2 3 4 1 0
3 4 1 0 2
4 1 0 2 3
```

Observed: rows 0 and 1 at columns 0–3, the cells `(2,2)`, `(3,3)`, `(4,4)`
and `(4,0)`: twelve cells; thirteen are held out, and all five classes occur
on them. Read as numbers, the class names violate the rectangle of rows 0, 1
and columns 0, 1 (`1 + 2 != 0 + 0`), so the given order of the names is not a
coding. Rows 0 and 1 share four columns with the distinct labels `1, 0, 2, 3`
and differ in the first, so the test holds; the graph is connected and the
diagonal is observed. The learner returns the whole table, and the CE program
of the gauge `2 (e - 1)` has the one-hot table as its only minimizer.

**A training set that must be refused.** Observe the row 0 and the column 0
of the table of `i + j` for `p = 5`: nine cells, a connected graph. The
identity coding and the coding above both have potentials that fit them, and
their tables differ at the unobserved cell `(2,2)`: `4` against `3`. No
learner can return both; this one answers `ambiguous`, as it does whenever
two different tables of the family share a connected training set.
Connected coverage alone does not fix the coding, and the rank test is not an
artifact of the method. In the other direction, the witness without its
cells `(4,4)` and `(4,0)` still passes the rank test, but its row 4 is not
observed, and the learner answers `disconnected`.

**What remains open.**

- The structural hypothesis is about the unknown truth and cannot be tested
  on the training set. The answer of the learner does not depend on the
  unobserved labels, so for a table outside the family an accepted completion
  need not be the table. Many-to-one decoders (CM15), composite `p`,
  vocabularies of another size and the question which tasks of the stress
  suite belong to the family are not treated.
- When a split passes the rank test is not analyzed beyond the two-row
  coverage: there is no probability statement over random splits.
- The native elimination and its graph searches are not formalized; of its
  anchoring only that the kept rectangles have the same solutions. The Lean
  statements are about what they compute: the kernel of the pinned system and
  Mathlib's rank of its matrix. Their cost is not analyzed.
- Numerical solutions: the exact-real minimizer against the native tolerance,
  floating-point SVDs, and convergence of any iterative solver.
- Uniqueness is within the fixed-code program of the discovered coding, not
  over the CM12 hull of all output renamings.
- Discovery by training a network is untouched. This learner is an explicit
  algebraic procedure; that AdamW on a transformer (#193) or a paired latent
  decoder (#195) finds such a coding is a separate, open claim. Discovery from
  a stream without keeping the observed rectangles is not claimed either.

`CoupledMoments/CodingDiscovery/Basic.lean` defines the rectangle system.
`TwoRows.lean` proves the sufficient coverage. `Gauge.lean` proves points 1–3
and the equivalent forms of the test, and `Rank.lean` its form `rank + 1 = p`.
`Completion.lean` proves that every training-compatible phase completion is
the table; it uses the rigidity lemma of `PatternGraph.lean`, generalized in
place to potentials in any additive group. `Learner.lean` defines `discover`
and proves the statements about its answers. `CrossEntropy.lean` proves the
bridge, and `Witness.lean` the examples above. `Invariance.lean` proves that
the system depends only on the set of observations. `NativeChecks.lean`
proves the first and the third item of *The native checks*.

### CM18. Unseen decoded answers behind a decoder that merges opposite states

*Source-note extension for board task #197, 2026-11-12. The number CM16 and
the `Paired*.lean` modules belong to task #195.*

CM15 and CM16 each end with two CE minima that disagree at an unseen cell, and
CM16 says: "Recovery needs the identification conditions of a separate
argument; capacity and an observed fit do not give it." This item gives such
conditions for one class of fixed many-to-one decoders, as a certificate that
is checked on training samples alone, and says what they give and do not give
for the convex CE program. CM17 does not apply: its coding is a bijection.

**Setting.** A square table `y : n × n -> C`. The structural hypothesis, the
only assumption about unobserved cells, is

```
y(i,j) = d(r(i) + s(j))
```

for a fixed known decoder `d : Z/q -> C` and unknown phases
`r, s : n -> Z/q`. The decoder class is

```
q odd,        d(z) = d(w)  exactly when  w = z or w = -z.
```

The folding decoder of CM16 is in it for every `k`, and so is the two-class
decoder of `Z/3` of the CM16 limit. Input side, latent states and classes are
three separate numbers, and every fiber but the one of zero has two states.
The learner is given `d` and training samples `(i_t, j_t, y_t)` with
`y_t = y(i_t, j_t)`, and none of `r`, `s`. A sample says only that
`r(i_t) + s(j_t)` is one of two opposite states: one unknown sign per cell.

Two changes of the phases leave every decoded label unchanged: the shift
`(r + g, s - g)` and the reflection `(-r, -s)`. Latent states are therefore
never identified, and are not the target. The target is the decoded label of
an unobserved cell.

**Signs.** Three facts about `Z/q`, `q` odd, carry the argument.

1. Two rows with phase difference `δ` share two columns, where the first row
   has the latent states `u1` and `u2`. In another completion with difference
   `δ'`, the first column leaves `δ' ∈ {δ, -δ, 2 u1 + δ, -(2 u1 + δ)}` and the
   second the same with `u2`. If `u1 != u2` and `u1 + u2 + δ != 0`, the false
   candidates of the two columns are disjoint, and `δ' = δ` or `δ' = -δ`.
2. If `δ != 0`, that sign is the sign of every column shared by the two rows:
   `δ' = ε δ` forces the state `ε u` in the first row.
3. A further row is fixed by two of its cells in columns of different phases
   `a != b`: a state `x'` with `x' + a = ±(x + a)` and `x' + b = ±(x + b)` is
   `x`, because the reflected candidates `-x - 2a` and `-x - 2b` differ.

All three use `2x = 0 => x = 0`. On four states the third fails:
`x = 1, a = 0, b = 2, x' = 3`.

**The certificate.** Proposed phases `ρ, σ : n -> Z/q`, and a rectangle: two
rows `top`, `bot` and two columns `left`, `right`. From the rectangle the
certificate *reaches* rows and columns:

- the rows `top` and `bot`;
- every column with training cells in both;
- a row with two training cells in reached columns `j, j'` with
  `σ(j) != σ(j')`;
- a column with two training cells in reached rows `i, i'` with
  `ρ(i) != ρ(i')`.

It is *accepted* when

```
d(ρ(i_t) + σ(j_t)) = y_t    at every sample of a reached row and reached column,
the four cells of the rectangle are training cells,
ρ(top) != ρ(bot),   σ(left) != σ(right),
ρ(top) + ρ(bot) + σ(left) + σ(right) != 0.
```

Nothing is asked of `ρ, σ` at other samples. Every condition is a finite
statement about the certificate, the decoder and the training samples; a
derivation of "reached" is a finite list of the steps above.

**Soundness.** Let the certificate be accepted, and let `r, s` be any phases
that fit all training samples. Then there are a sign `ε` and a shift `g` with

```
r(i) = ε ρ(i) + g  on reached rows,      s(j) = ε σ(j) - g  on reached columns,
```

so that

```
d(r(i) + s(j)) = d(ρ(i) + σ(j))    for every reached row i and reached column j,
```

whether the cell is observed or not. With `r, s` the phases of the target this
is its label. The proof is the three facts: the rectangle gives the difference
of its rows up to a sign (1), that sign fixes the columns observed in both
rows (2), and every further step is (3). Nothing is assumed about unobserved
labels, about the sign or latent state of any cell, about injectivity, or
about an optimizer. The statement is per cell: it says nothing about a row or
a column that the certificate does not reach.

**Acceptance from labels.** The conditions on `ρ, σ` follow from conditions on
training labels, for any phases that fit the samples:

- the columns `left`, `right` carry the label pairs `(a, b)` and `(c, e)` in
  the rows `top`, `bot`, with `(a, b) != (c, e)` and `(a, b) != (e, c)`; the
  first gives different column phases, the second the sum condition, because
  a vanishing sum exchanges the two pairs;
- some column has different training labels in `top` and `bot`, which gives
  different row phases.

In the same way, two columns have different phases as soon as some row has
different training labels in them, and two rows likewise. Replacing
"different phases" by these witnesses gives a reach relation on training
cells and labels alone. For a realizable table, the phases of the table with
such a rectangle form an accepted certificate, and every cell that the labels
reach is covered. So any two completions of the training samples have the
same label there. These conditions are sufficient; they are not claimed
necessary.

**Why the examples of CM15 and CM16 are excluded.** Their samples are three
cells of a two by two table: a connected graph with a perfect matching, and
no rectangle of training cells. No certificate is accepted on them, for any
decoder and any labels. Identification here is carried by a four-cycle with
label conditions, not by connectivity. The corner decoder `z != 2` of CM15 is
not even, so it is outside the class.

**The procedure.** `recover(d, cells, labels, i, j)` answers `unsupported`
(the decoder is outside the class), `inconsistent` (no phases fit the
samples), `uncertified` (no accepted certificate reaches the row `i` and the
column `j`), or `accept` with the label proposed by such a certificate.

- An accepted label is the label of every completion of the samples at
  `(i, j)`; for a realizable table it is the table's.
- For a realizable table the label conditions above give `accept`.
- Where two completions of the samples disagree, the answer is `uncertified`.

`recover` is a specification by search over phases and certificates. It is
not executable and no running time is proved. A certificate can be found by
propagation (this paragraph is not formalized): choose the rectangle, set the
phase of `top` to zero by a shift, read the two candidate states of each
column of `top` from its label, intersect the four candidates of the row
difference from two columns, choose its sign by a reflection, and continue
with one fiber look-up per training cell. Because acceptance asks for the
labels only on reached rows and columns, this needs no search over the
unreached part. The answer `inconsistent` does: deciding whether any phases
fit all samples is given here only by exhaustive search.

**In the convex programs.** The declared program is ordinary CE over the
whole vocabulary of `decode d P`, with `P` in the spectral domain of CM15 at
the budgets `N`; CM16 adds the equalities of an even latent table. For unit
codes and probability rows the operator bounds hold by themselves (entries of
modulus at most one), and the constant moment has nuclear norm `N` (CM13): at
the budgets `N` the domain is the probability tables whose nonconstant
moments have nuclear norm at most `N`.

1. *A minimizer with the target's answers.* For any phases that fit the
   training samples, the decoded table lies in the paired domain, minimizes
   the training CE over the whole CM15 domain, and has the one-hot row of the
   target's label at every covered cell.
2. *The relaxation has a wrong minimum.* `q = 3`, `d(z) = (z != 0)`, `N = 3`,
   `y(i,j) = (i != j)` with `r(i) = i`, `s(j) = -j`, and every cell observed
   but `(0,0)`. The certificate with the rows `1, 2` and the columns `0, 1`
   is accepted; it reaches the row `0` through the columns `1, 2` and the
   column `0`, and proposes `false` at `(0,0)`. Take the even latent table of
   the target and replace its row at `(0,0)` by the uniform one. Both
   nonconstant moments become the real matrix

   ```
     0   -1/2  -1/2
   -1/2    1   -1/2      =  3/4 v1 v1^T + 1/3 v2 v2^T - 1/12 v3 v3^T,
   -1/2  -1/2    1
   ```

   with `v1 = (0,1,-1)`, `v2 = (1,-1,-1)`, `v3 = (2,1,1)`, of nuclear norm at
   most `3/2 + 1 + 1/2 = 3`. The table is in both domains and fits the eight
   samples, so its decoded table is a global CE minimum, with probability
   `2/3` for the wrong label at a certified cell. Minimizing CE over the
   spectral relaxation does not identify even a certified answer.
3. *The hull of decoded phase tables.* Let `H` be the convex hull of the
   tables `d(r(i) + s(j))` over all phases. It depends on `d` and `N` only
   and lies in both relaxations. For an accepted certificate, a covered cell
   `(i, j)` with proposed label `c`, and every `Q` in `H`,

   ```
   1 - Q(i,j)(c)  <=  sum over samples t of  (1 - Q(i_t,j_t)(y_t)).
   ```

   Both sides are affine in `Q`. At a vertex that fits the samples the left
   side is zero by soundness; at a vertex that does not, the right side is at
   least one. Hence every table of `H` with the ideal training CE, and for a
   realizable table every minimizer of CE over `H`, has the one-hot row of
   the target at every covered cell. Latent probabilities are not identified
   and do not occur in the statement. The wrong table of point 2 is outside
   `H`. No way to project onto `H` or to optimize over it is given.
4. *The certified restriction.* Add to the spectral program the linear
   equalities that fix every covered row to the one-hot row of its proposed
   label. The labels come from the certificate, hence from training samples
   alone. The restricted domain is convex and contains the table of point 1;
   every minimizer over it minimizes CE over the unrestricted domain as well,
   and has the target's rows at the covered cells. The equalities are
   imposed: at the covered cells the answer is the certificate's, by
   construction. What is proved is that it is the target's, and that the
   restriction costs nothing in training CE.

These are statements about exact minimizers over the reals. The changed
families are `H` (point 3) and the restricted domain (point 4); for the
unchanged relaxation only point 1 and the obstruction of point 2 hold.

**Witness.** The folded table of CM16, `N = 3`, `q = 5`, three classes, with
the cell `(2,1)` withheld:

```
0 1 2
1 0 1
2 . 0
```

The rows `0, 1` and the columns `0, 2` carry the pairs `(0,1)` and `(2,1)`,
neither equal nor exchanged, and the column `0` separates the two rows. The
column `1` is observed in both rows. The row `2` has training cells in the
columns `0` and `2`, where the row `0` has the different labels `0` and `2`.
So the labels reach `(2,1)`, and every completion of the eight samples has
the label `1` there; the reflected and shifted phases `(1 - i, j - 1)` are
one, with the latent state `4` in place of `1`. The training cells `(1,0)`
and `(0,1)` have the latent states `1` and `4` and the same label, so the
decoder's collisions are in use. `recover` answers `accept 1`. The folded
table is a feasible CE minimizer, and every minimizer over `H` and over the
restricted domain has the one-hot row of `1` at `(2,1)`.

**Refusals.** On the samples of the CM16 limit `recover` answers
`uncertified` at `(1,1)`, where its two completions disagree, and no accepted
certificate covers that cell. With the corner decoder of CM15 it answers
`unsupported`. One cell with two labels is `inconsistent`.

**For task #195.** The interface is: a certificate `(ρ, σ, top, bot, left,
right)`, its checker (`Accepts`, with the reach relation `Pinned`), and the
soundness statement for every completion (`accepted_decoding`). In the convex
program it enters as the restriction of point 4 (`certified_restriction`); a
solver that fixes the covered rows stays inside the program of CM15 and CM16.
The characterization of the domain at the budgets `N` by the nonconstant
nuclear norms alone (`budget_iff_nuclear`) is the condition that the native
audit of `experiments/convex_stress_paired` checks numerically.

**What remains open.**

- The structural hypothesis is about the unknown truth and cannot be tested
  on the training set. For a table outside the family an accepted label need
  not be the table's.
- The decoder class: a fixed known decoder of an odd cyclic group whose
  fibers are exactly the opposite pairs. Larger fibers, an even `q`, other
  groups, an unknown decoder and the decoder hull of CM15 are not treated.
- Coverage: the label conditions are sufficient ones. There is no statement
  about random splits, and which tasks of the stress suite belong to the
  family or meet the conditions is not examined.
- The procedure is a specification. The propagation and its cost, and any
  efficient test of consistency, are not formalized.
- The minimizers of the spectral relaxation itself are not identified (point
  2). Which minimizer an iterative solver approaches, and whether a wrong
  minimum occurs at the original sizes, is not examined. Optimization over
  `H` is not addressed.
- Exact reals throughout: no tolerance, no stability bound, no convergence of
  the full-batch, minibatch or forward-only iterations.

`CoupledMoments/DecodedRecovery/Signs.lean` proves the three facts.
`Certificate.lean` defines the certificate and proves the rigidity of the
rectangle; `Rigidity.lean` proves soundness. `Coverage.lean` proves
acceptance from labels, and `Learner.lean` the statements about `recover`.
`PhaseHull.lean` proves point 3, `CrossEntropy.lean` points 1 and 4.
`Domain.lean` proves the description of the domain at the budgets `N`.
`ZeroMass.lean` and `Obstruction.lean` prove point 2, and `Witness.lean` the
witness and the refusals.

### CM19. Finite support propagation for general decoder fibers

CM18's opposite-pair restriction is sufficient for its rigidity theorem, not
necessary for every identifiable decoded answer. Here the fixed categorical
decoder `d : ZMod q -> C` is arbitrary and `q > 0` may be even. The unknown
target is still assumed to have the form

```
y(i,j) = d(r(i) + s(j)).
```

Only the training cells and their labels are supplied to the procedure. The
phases are not supplied. A finite bank `d_k` is also allowed: the structural
assumption is that the target uses some member of that declared bank, with
unknown index. Finding a fitted member is not evidence that this assumption
holds for the unseen labels. Flat spectral bounds alone still do not imply it.

**A finite computation.** Choose one observed anchor `(a,b)` with label `c`.
For every `z` with `d(z) = c`, start a branch of row and column phase domains:

```
R_a = {0},  S_b = {z}; all other domains are ZMod q.
```

One simultaneous sweep uses the previous domains on both sides. Retain
`u` in `R_i` exactly when, for every observed incident cell `(i,j)` with
label `c_ij`, there is `v` in `S_j` with `d(u+v) = c_ij`. Retain `v` in
`S_j` by the corresponding condition at every observed row. Repeated
observations are checked separately. There is no constraint for an
unobserved cell and no access to its answer.

Run a specified natural number of sweeps, including possibly zero. Reject
a branch if any of its variable domains is empty. At query `(i,j)` its
output set is `{d(u+v) : u in R_i, v in S_j}`; rejected branches contribute
the empty set. Take the union over **every** anchor phase. For a bank take
the further union over **every** decoder. These operations are finite and
computable in Lean. No fixed-point or efficient implementation theorem is
needed for the stated finite-budget result.

**Preservation and identification.** Every assignment fitting all observed
labels survives in one of these branches at every budget. To prove this,
normalize it to

```
r'(i) = r(i) - r(a),   s'(j) = r(a) + s(j).
```

All sums are unchanged. Its branch has `z = r(a)+s(b)`, which occurs in
the anchor fiber. At each observed cell the assignment's own neighboring
phase supplies the required support, so a simultaneous sweep preserves
every one of its coordinates. Induction gives preservation for any finite
number of sweeps. In particular its domains never become empty, and its
decoded answer occurs in the final output union.

Consequently:

- A singleton union `{c}` certifies label `c` for every compatible phase
  assignment and, in the bank version, every compatible bank member.
- An empty union at any query refutes existence of a fitting assignment
  in the declared family. Pruning need not detect every inconsistency.
- If two actual fitting assignments give different query answers, the
  union is never a singleton at any budget. The same holds when the two
  assignments use different decoder choices.
- A nonsingleton union alone does **not** prove actual ambiguity: the
  Cartesian products may contain combinations with no global completion.

The distinction between decoded and latent identification is essential.
Neither a singleton row domain nor a singleton column domain is required
to certify a label. Pruning is conservative; completeness and coverage on
a particular split are separate questions.

**The convex spectral program.** Let `D` be the spectral decoder hull of
CM15 with Fourier codes on `ZMod q` and both spectral budgets equal to
`N`, the number of rows/columns. Its definition already includes all fixed
decoders; it is convex and has probability-vector rows. Restrict it to

```
D_cert = {Q in D : Q(i,j) = oneHot(c)
                  whenever the bank's output union at (i,j) is {c}}.
```

The labels on the right are computed from training data by the procedure
above. These are linear equalities, so `D_cert` is convex. Use the same
ordinary full-vocabulary log-sum-exp cross-entropy as CM2: the learned
probability coordinates serve as bounded logits. The loss is convex on
the restricted domain; it has not been changed into a latent regression
loss or `-log Q(i,j)(c)`.

If any bank member and phases fit the observations, their decoded one-hot
table belongs to `D_cert`. This follows from CM15's feasible phase witness
and the preservation theorem above. That table already minimizes the
training CE over all of `D`, so it also minimizes it over `D_cert`.
Conversely every minimizer over `D_cert` is a minimizer over `D`: compare
it with this retained witness. Every feasible table in `D_cert` has the
target's rows at certified cells whenever the target belongs to the bank.

The equalities are imposed. The result proves their correctness under the
explicit target-family assumption and that they cost no training CE. It
does not say that minimizing the original spectral relaxation discovers
them, that a preferred fitted decoder is true, or that every decoder in
the larger hull belongs to the declared identification bank.

**Witness outside CM18.** Take `q = 4` and `d(3) = true`, with
`d(0) = d(1) = d(2) = false`. Observe only

```
true   true
false  .
```

Anchor at the false observation `(1,0)`. There are three branches,
`z = 0,1,2`. In each, the first sweep gives `r(0) = 3-z`; the second
gives `s(1) = z`. The missing decoded answer is false in all branches.
All three latent states really remain possible there: the assignments
`r = (3-z,0)`, `s = (z,z)` fit for each `z`. The concrete finite computation
is checked by Lean's kernel using `decide`, and `wide_recovery` proves
agreement of every completion. The decoder is not a `PairedDecoder`:
`0` and `1` collide but are not opposite. Adding `z -> d(z+1)` as a second
bank member preserves the certified answer while allowing a different
latent explanation.

For refusal, replace the three observations by false. The assignments
`r=(0,0), s=(0,0)` and `r=(0,1), s=(0,2)` fit, but the missing answers
are false and true. No number of sweeps certifies a singleton. A bank
containing the original decoder and `z -> (z=1 or z=3)` also has two
compatible members with different answers. Finally, two opposite labels
on the same cell yield an empty union after one sweep, both for the fixed
decoder and for the witness bank.

**Scope.** This extends fixed known decoders to arbitrary fibers, permits
even cyclic groups, gives a computable finite procedure, and quantifies
over a declared finite bank. It does not learn an unrestricted decoder,
prove the target-family assumption from observed data, prove completeness
or a coverage rate, or give numerical error or optimizer convergence
bounds. No original stress-task pass is implied by the finite witness.

`CoupledMoments/DecodedPropagation/Defs.lean` defines the computation;
`Basic.lean` proves preservation, shrinking domains and normalization;
`Identification.lean` proves singleton identification and the refusals;
`CrossEntropy.lean` proves the convex restriction and optimality;
`Witness.lean` and `Refusal.lean` check the explicit examples.

**Endpoint construction for a solver.** A surviving branch also gives a
feasible fallback without solving for a complete training fit. Choose any
one state from each of its nonempty final row and column domains. At every
query its decoded sum occurs in that branch's Cartesian output set, hence
in the bank union. Therefore it agrees with every certified singleton,
whether or not these chosen phases fit the training samples. Its one-hot
latent phase table is spectrally feasible at the budgets `N` by CM14, and
its decoded table lies in both its fixed-decoder domain and `D_cert`.
Every live branch supplies such a point (`branch_endpoint`, `live_endpoint`).

This is a fallback, not necessarily a CE minimum. In the four-state witness
at budget zero, anchoring at the false observation with state zero and
choosing all phases zero keeps the certified anchor but predicts false at
the two observed true cells. The choice `r=(3,0), s=(0,0)` in that same
unpruned branch does fit them. These two feasible decoded tables differ
at an observed, uncertified cell, while agreeing at every certified cell.

If two output tables have the certified rows, every affine mixture
`(1-t) P + t Q` keeps those rows (`certified_rows_blend`). With `0 <= t <= 1`
the simplex and spectral constraints are also preserved when both tables
are feasible, by their existing convexity theorems. When a proposed iterate
is spectrally infeasible, this does not prove a positive mixing weight
feasible: a numerical endpoint routine must check its returned coordinates
and may fall back to weight zero. No floating-point error theorem follows.

A solver using one selected decoder optimizes only its fixed-decoder convex
domain. Its answer constraints must still be computed from the **whole**
declared bank. The bank-level optimality theorem above does not identify a
selected fitted decoder as the true one or prove that its smaller domain
attains the optimum over the full bank hull. This distinction must remain
visible in native candidate selection and reported CE.

`CoupledMoments/DecodedPropagation/Endpoint.lean` proves the endpoint facts
and the partial-coverage witness before the general native solver is added.

**Anchor coverage can change at a fixed point.** On three rows and columns,
use `d : ZMod 3 -> Bool`, with `d(0)=false`, `d(1)=d(2)=true`. Observe every
cell except `(0,0)`, with false on the observed diagonal and true elsewhere.
The phases `r=(0,1,2)`, `s=(0,2,1)` fit these observations. Anchoring at
`(1,1)` has only the phase-zero branch. After one sweep its row and column
domains are both `({1,2},{0},{1,2})`. These domains are a fixed point of the
support operator, and the missing cell's output union remains `{false,true}`
for every positive sweep budget. Increasing that budget does not help.

Anchoring instead at the observed `(0,1)` branches on both phases 1 and 2.
After three sweeps the union at `(0,0)` is `{false}`. Preservation therefore
proves that every compatible completion has that label: the first anchor's
nonsingleton was an overestimate, not actual ambiguity. No extra labels or
preferred branch were used. `DecodedPropagation/Anchor.lean` proves the
finite computations, the fixed point for all budgets and identification.
This example motivates trying more than one observed anchor; it does not
prove completeness of an anchor policy or change the declared-family assumption.

### CM20. Extracting a Fourier direction by constant amplitude

The categorical latent-decoder constructor of CM15 proposes phase orders from
a singular vector of a centered label sketch. A repeated singular value leaves
a choice of directions inside its singular subspace. This section studies a
separate mathematical question: when does constant amplitude inside an exact
Fourier subspace identify a single character? Recovery of that subspace from
the incomplete observed table remains a separate, unproved step.

Let `q > 0`, let `I` be a finite index set and let `k : I -> ZMod q` list
frequencies. Write `chi(t) = exp(2 pi i t/q)` and

```
f(x) = sum_i a_i chi(x k_i).
```

Require **separated differences**: for every `i != j` and every `u,v`,

```
k_j - k_i = k_v - k_u  implies  i = u and j = v.
```

The comparison includes diagonal pairs `u=v`, so this condition also excludes
zero off-diagonal differences. It is a property of the whole proposed exact
subspace, not just the coefficients that happen to be nonzero. It is sufficient;
no necessity or coverage theorem for observed singular subspaces is claimed.

**Extract one direction.** If at least one `a_i` is nonzero and `|f(x)|^2`
is the same real number at every `x`, then exactly one coefficient is nonzero.
Consequently `f(x) = a_i chi(x k_i)` for that index (`flat_extract`). The
modulus need not be prime. A permutation `r : X <-> ZMod q` of the categorical
rows changes neither the hypothesis nor the conclusion: constant amplitude of
`f(r(x))` identifies a single permuted character (`permuted_extract`). The
permutation is a structural hypothesis and is not supplied to the procedure.

If additionally `sum_x f(x) = 0`, then the extracted `k_i` is nonzero
(`centered_extract`). For a prime modulus that character, multiplied by its
nonzero coefficient, takes distinct values at all `q` states
(`centered_injective`). These facts explain the roles of centering and the
prime-modulus restriction when a native proposal needs a phase permutation.

**Proof from the actual amplitude.** Expand

```
|f(x)|^2 = sum_i sum_j conj(a_i) a_j chi(x (k_j-k_i)).
```

Orthogonality of the primitive character gives the unnormalized coefficient

```
sum_x conj(chi(x t)) |f(x)|^2
  = q sum_{i,j : k_j-k_i=t} conj(a_i) a_j.
```

The left side is computed from the squared amplitude, without presupposing
sparse coefficients. If the amplitude is constant, it vanishes for every
`t != 0`. Separation isolates the product `q conj(a_i) a_j` at the difference
of every distinct pair. Since `q != 0`, all these products are zero. Choosing
one nonzero coefficient forces all the others to vanish. Conversely, a vector
supported at one mode has constant squared amplitude `|a_i|^2` (`support_flat`).
Centering rules out the constant character; injectivity in the prime case
follows from injectivity of the primitive character and multiplication by a
nonzero residue. These are proved in `FourierExtraction/Basic.lean` and
`Extraction.lean`, using the existing character orthogonality theorem.

**A quantitative bound in the same exact subspace.** For any real `rho` and
distinct `i,j`, separation also gives

```
q |a_i| |a_j| <= sum_x abs(|f(x)|^2 - rho).
```

Subtracting `rho` does not change a nonconstant Fourier coefficient. Apply
the triangle inequality to that coefficient and use `|chi|=1`. In particular,
if every squared-amplitude error is at most `epsilon`, then
`|a_i| |a_j| <= epsilon` (`pair_le_deviation`, `approximate_pair`). This bounds
remaining cross-mode products. It does not include subspace estimation error,
a floating-point error enclosure or convergence of an extraction iteration.

**Nonempty example and a limit.** For `q=13`, frequencies `1,3,9` have six
distinct nonzero ordered differences (`triple_separated`, checked by `decide`).
Coefficients `(1,0,0)` give a centered, unit-amplitude, nonconstant vector that
distinguishes all thirteen states. Coefficients `(1,1,0)` cannot have constant
amplitude (`mixed_not_flat`). Thus the condition admits a three-dimensional
subspace and excludes a genuine mixture within it.

For `q=2`, take frequencies `0,1` and coefficients `(1+i)/2, (1-i)/2`.
Both coefficients are nonzero, but the vector takes values `1,i`, of unit
amplitude. The two off-diagonal differences coincide. `collision_flat` and
`collision_two_modes` prove the counterexample, so the separation assumption
cannot simply be dropped. `Witness.lean` instantiates every conditional result.

**Convexity limit.** The coefficient set defined by unit amplitude is not
convex (`amplitude_not_convex`). For any listed frequency, its positive and
negative pure coefficient vectors both have unit amplitude. Their midpoint
is zero, which has zero amplitude. Therefore the new criterion cannot be
imposed directly as a convex constraint on the coefficient vector. Alternating
amplitude and subspace projections would be an auxiliary nonconvex proposal,
with no convergence theorem here. A convex formulation needs a separate lift
and proof before it is used to change the model.

**Remaining gap.** A future extraction proposal must keep right phases tied to
the same left direction through the original sketch. Existing observed-label
decoder fitting can score resulting phase orders by ordinary CE, with no
arithmetic meaning supplied for input tokens. This section supplies the
conditional direction-identification result, not that complete learner.

The extraction theorem does not identify the true decoder, prove the target's
membership in a chosen family, prove that a masked sketch's leading subspace
contains the needed character, or identify unseen labels from amplitude alone.
A native experiment must retain every attempted subspace and failed proposal,
its actual work and memory use, and every earlier candidate. Any returned latent
table still needs the CM15 spectral/CE audit; new original-data scores require
a separately frozen protocol and replay. The current frozen cohorts are unchanged.

### CM21. Convex density extraction in a supplied Fourier subspace

CM20's unit-amplitude constraint on vectors is nonconvex. A matrix lift avoids
that obstruction without imposing rank one. This section proves the exact
geometry and a finite policy of two linear objectives. It assumes a supplied
exact subspace; it does not recover that subspace from missing labels.

Let `q > 0`, let `I` be finite, let `r : X <-> ZMod q` be an unknown row
permutation and let `k : I -> ZMod q` have CM20's separated ordered differences.
Use unnormalized Fourier features

```
F[x,i] = chi(r(x) k_i).
```

The supplied feature matrix is `B = F Q`, where `Q` is unitary:
`Q* Q = Q Q* = Id`. Here `*` denotes conjugate transpose. The permutation,
frequencies and unitary are structural witnesses in the theorem, not inputs
to the optimization policy. The chosen normalization gives pure kernel
diagonal one. For a numerically supplied orthonormal basis `U`, the equivalent
scaling is `B = sqrt(q) U`; the exact equality to `F Q` is still a hypothesis,
not a conclusion about an estimated singular subspace.

**Convex matrix domain.** The coefficient variable `M` is Hermitian positive
semidefinite with `trace(M)=1`. Its observable kernel is `K = B M B*`.
Require the affine equalities `K[x,x]=1` for all rows. `Density`, `reading`
and `Feasible` use these actual matrix operations. `density_convex` and
`feasible_convex` prove convexity directly; `reading_mix` proves linearity
of every kernel reading. There is no rank constraint and no presumed vector
factorization in the domain.

In hidden Fourier coordinates `H = Q M Q*`, unitary conjugation preserves
positivity and trace, and the observable kernel equals `F H F*`
(`density_conjugate`, `reading_change`, `feasible_change`). Its diagonal is

```
K[x,x] = sum_i sum_j H[i,j] chi(r(x) (k_i-k_j)).
```

The orientation is `k_i-k_j`, because this is `F H F*`; CM20's squared-vector
formula used conjugated coefficients in the opposite order. Character
orthogonality gives

```
sum_z conj(chi(z t)) (F H F*)[r^-1(z),r^-1(z)]
  = q sum_{i,j : k_i-k_j=t} H[i,j].
```

If the diagonal is constant, its nonzero Fourier coefficients vanish.
Separation isolates `q H[i,j]` for each `i != j`; hence `H[i,j]=0`
(`fourier_diagonal`, `diagonal_coefficient`, `flat_off_diagonal`). This argument
applies to an arbitrary matrix; positivity and rank are not used to obtain
the vanishing cross terms.

Positivity then makes the remaining diagonal entries real and nonnegative;
trace one makes their sum one. Conversely, every such diagonal matrix has
constant readings. Thus `feasible_iff_diagonal` characterizes all feasible
matrices by real probability weights. `feasible_hull` and `rotated_hull`
identify the complete convex sets:

```
{H feasible in F} = convexHull {E_i : i in I},
{M feasible in B} = convexHull {Q* E_i Q : i in I},
```

where `E_i` is the projector onto coordinate `i`. These are equalities of
sets, not an assumption that an optimizer is an extreme point. In particular,
all mixtures remain feasible.

**Two observable linear objectives.** Choose any two distinct categorical
rows `a,b`. First maximize `Re K[a,b]` on the feasible domain. Then maximize
`Im K[a,b]` on the first optimum face. Both objectives are real linear and
the second domain adds one affine equality to a convex set. `LexMaximum`
states exactly this two-stage specification, with comparisons against every
feasible matrix. It does not presume that any numerical iteration satisfies it.
This finite policy avoids assuming an unknown mode-dependent separating
objective or proving a particular rotated-angle tie-break formula.

At a pure mode, the complex score is

```
s_i = chi((r(a)-r(b)) k_i).
```

For prime `q`, distinct rows and separated frequencies give distinct scores
(`score_injective`). For any finite nonempty index set there is an index with
maximum real score and, among the tied indices, maximum imaginary score
(`exists_lex_vertex`). Its pure projector attains both objectives
(`exists_fourier_max`, `exists_rotated_max`), so existence is derived.

For any feasible mixture, equality to the maximal real average forces every
positive-weight component onto the first optimum face. Equality in the second
average forces those components to have the same imaginary score as well
(`weighted_bound`, `weighted_support`). Distinct complex scores therefore
leave only one nonzero weight, equal to one (`lex_weights_pure`). Consequently
every exact two-stage optimizer is a pure mode projector, including in the
supplied rotated basis (`fourier_max_pure`, `lex_change`, `rotated_max_pure`).
No rank-one optimizer or correct unseen answer is supplied as a hypothesis.

**Observable centering.** If every column of `B` sums to zero, multiplying
by `Q*` preserves those zero sums. Orthogonality then rules out `k_i=0` for
every column (`centered_mul`, `centered_nonzero`). A kernel column from an
extracted pure mode is `chi((r(x)-r(c)) k_i)` for its anchor row `c`. With
prime `q` and `k_i != 0` these values distinguish all rows
(`extracted_column_injective`). `supplied_extraction` combines existence and
injectivity using observable centering and the explicit exact-subspace
assumption. It identifies a relative phase representation, not a decoder.

**Examples and boundary.** The separated frequencies `1,3,9` modulo 13 admit
the mixed density `diag(1/2,1/2,0)`, which is none of the pure projectors
(`mixed_feasible`, `mixed_not_pure`). The rational unitary rotation with block
`[[3/5,-4/5],[4/5,3/5]]` mixes two hidden modes and fixes the third. Its supplied
columns are centered; `witness_extraction` proves an optimizer with an
injective phase column for this genuinely rotated three-mode input.

For the colliding frequencies `0,1` modulo two, lift CM20's vector
`a=((1+i)/2,(1-i)/2)` to `H=a a*`. `vector_reading` verifies its diagonal
readings equal the actual squared amplitude. `collision_feasible` proves
positivity, trace one and both constant readings. Its off-diagonal entry is
nonzero (`collision_cross_term`), so it is not a diagonal probability mixture
(`collision_not_diagonal`). Separation cannot simply be removed from the
characterization. Closed examples instantiate every conditional theorem.

**Remaining work before a new experiment.** This proves an exact convex
extraction problem, not a method for recovering its input subspace from a
masked sketch. The separated-difference condition is sufficient and may fail
for a larger chosen spectral subspace. Estimated subspaces, floating-point
feasibility, finite iteration budgets and approximate lexicographic optima
need separate analysis or explicit refusal/tolerance accounting. The proof
does not justify treating an approximate mixed solution as an exact mode.

A bounded Rust/Kernite follow-up must record the chosen subspace, centering,
normalization, PSD/trace/diagonal residuals, both objective stages and every
failed attempt. Right-side phases must be read through the same original
sketch, not extracted independently. Any phase/decoder proposal still needs
ordinary-CE scoring, the existing latent-table spectral audits and replay.
Retain the earlier candidates, the full35 short/long scope, all three update
modes, the conventional-transformer comparison and the 64 GiB process cap.
The current frozen producers and scores are unchanged by this proof.

### CM22. Convex density tolerances and conditional stability

CM21 specifies exact diagonal equalities in an exact supplied Fourier
subspace. The bounded native prototype exposes their sensitivity: its
centered, re-orthonormalized noisy fixture is inconsistent, and all six
estimated subspaces in the 113-observation categorical fixture refuse the
exact affine constraints. Declare a model tolerance separately from numerical
convergence tolerances, and analyze the resulting domain.

**Convex domain.** Write `K_B(M)=B M B*`. For finite coefficient and row sets,

```
D_epsilon(B) = {M PSD : trace(M)=1,
                       |K_B(M)[x,x]-1| <= epsilon for every x}.
```

`Relaxed` uses the norm of the actual complex reading; `relaxed_convex`
proves convexity without a rank constraint. PSD makes diagonal readings real
(`reading_diagonal_real`), so this domain is precisely the real interval
constraint `1-epsilon <= Re K_B(M)[x,x] <= 1+epsilon` (`relaxed_interval`).
Exact feasibility implies membership for every nonnegative tolerance
(`exact_relaxed`), and zero tolerance recovers CM21 (`relaxed_zero`).

For every density, the diagonal real parts are nonnegative and sum to one
(`density_weights`), so each is at most one (`density_diagonal_le_one`).
The two-by-two PSD principal minors then give `|M_ij| <= 1`
(`density_entry_le_one`). These are derived properties of the actual matrix,
not additional purity assumptions.

**Exact-subspace error.** Let `F[x,i]=chi(r(x) k_i)`, where `q>0`,
`r : X <-> ZMod q`, and the ordered frequency differences are separated.
Subtracting the constant target from CM21's Fourier identity gives

```
q M_ij = sum_z conj(chi(z (k_i-k_j))) (K_F(M)[r^-1(z),r^-1(z)]-1), i != j.
```

`error_coefficient` proves this identity. The triangle inequality and
cancellation of `q` give `|M_ij| <= epsilon` from uniform diagonal error
`epsilon` (`off_diagonal_le`); PSD is unnecessary for this implication.
With `d=card(I)`, removing these off-diagonal entries changes each kernel
reading by at most `d^2 epsilon` (`diagonal_reading_error`). The dimension
factor is conservative and explicit. For a density, the diagonal reading is
a probability average of mode scores, so its real objective differs from
that average by at most the same bound (`relaxed_score_error`).

**Feature perturbation.** Suppose a structural reference `C` satisfies

```
|B[x,i]-C[x,i]| <= delta,  |C[x,i]| <= R,  delta,R >= 0.
kappa = d^2 delta (2R+delta).
```

`product_error` bounds each feature-product change by `delta(2R+delta)`.
Summation using the derived density-entry bound proves
`|K_B(M)[x,y]-K_C(M)[x,y]| <= kappa` (`reading_perturbation`). It follows that

```
M in D_epsilon(B)  =>  M in D_(epsilon+kappa)(C),
M exactly feasible in C and kappa <= epsilon  =>  M in D_epsilon(B).
```

These are `relaxed_transfer` and `reference_admitted`. The latter admits
every exact-reference competitor, not a presumed correct mode. Normalization
is not needed for these implications and does not establish their feature
error hypothesis.

**Objective accuracy and mixed mass.** Suppose `M` is a density and

```
Re K_B(N)[a,b] <= Re K_B(M)[a,b] + eta  for every N in D_epsilon(B).
```

When `kappa <= epsilon`, every exact-reference feasible `N` consequently
satisfies `Re K_C(N)[a,b] <= Re K_C(M)[a,b] + eta + 2 kappa`
(`objective_transfer`). This is a conditional optimizer specification;
no numerical iteration is asserted to attain it.

For probability weights `w`, real scores `v`, and a distinguished index `i`,
the gap condition `gamma <= v_i-v_j` for every `j != i` gives
`gamma(1-w_i) <= v_i-sum_j w_j v_j` (`weighted_gap`). No weight is assumed
zero. Combining this with the actual diagonal error gives
`gamma(1-Re M_ii) <= eta+d^2 epsilon` in the exact basis
(`score_concentration`).

For the estimated-basis statement, let `C=F Q`, with both unitary identities
for `Q`, set `H=Q M Q*`, and suppose `M in D_epsilon(B)`. Put
`s_i=chi((r(a)-r(b)) k_i)`. Under frequency separation, the feature bounds
above, `kappa <= epsilon`, the optimizer comparison and
`gamma <= Re s_i-Re s_j` for every `j != i`, `rotated_concentration` proves

```
gamma (1-Re H_ii) <= eta + 2 kappa + d^2 (epsilon+kappa).
```

For positive `gamma`, division bounds the mass outside the best mode.
The unknown permutation, frequencies and unitary are structural witnesses,
not inputs to the observable optimization problem. No recovered mode,
rank-one optimizer, true decoder or unseen answer is assumed. Small gaps or
large errors can make the estimate uninformative. A zero real gap gives no
selection guarantee; CM21's exact imaginary tie-break theorem is not extended
here to approximate second-stage constraints.

**Closed witnesses.** The separated modulo-13 mixture `diag(1/2,1/2,0)`
remains feasible at zero error and is no pure projector (`mixed_relaxed`,
`zero_error_not_pure`). Thus small amplitude error alone cannot prove purity.
Frequencies `0,1` modulo four have separated differences and real scores
`1,0` at rows one and zero (`two_separated`, `character_four_one`, `two_gap`).
Their mixture with weights `3/4,1/4` is exactly feasible and within `1/4` of
every feasible real objective (`two_weights`, `two_feasible`, `two_near`).
A closed example attains the combined bound with positive score gap one.

The supplied matrix `(11/10)F` has actual nonzero entry error `1/10`
(`perturbed_error`). Its conservative bound `kappa=21/25` admits every pure
reference projector (`perturbed_admitted`). This is a perturbation witness,
not a claim about normalized estimated subspaces. Closed examples inhabit
every conditional theorem, including the collision density's entry bound.

**Native follow-up and limits.** A bounded interval solver may use this
convex domain with a model tolerance distinct from numerical residual
thresholds. Check actual PSD, trace, interval violations and objective bounds;
retain every capped/refused attempt and independently replay matrix and dual
states. Justify the interval update explicitly; coordinate clipping is not
specified as a projector onto the matrix domain here. Floating-point
diagnostics remain numerical evidence.

This section does not infer `delta` from missing labels, certify a masked
sketch's subspace, identify an unknown many-to-one decoder, prove approximate
phase sorting correct, or determine unseen labels. Native tolerance choices
need declared training-only policies and empirical evaluation. Preserve exact
and earlier candidates, all35 short/long tasks, full-batch, minibatch and
forward-only modes, the transformer comparison and the 64 GiB limit. New
original-data runs still require a separately frozen protocol.

### CM23. Observed-CE refinement of decoded spectral candidates

*Source-note extension for board task #215, 2026-11-13; Lean before native refinement.*

CM22 removes one exact-subspace obstruction, but the accepted masked-sketch
candidate still has 53 errors on 113 observed labels. Its latent moments meet
their numerical bounds. Keep the exact search, interval search and earlier
candidates; refine their proposed phase assignments using observed labels.

For a deterministic predicted class `p` in the full vocabulary `C`, ordinary
unit-logit cross-entropy satisfies

```
sum_c exp(1[c=p]) = exp(1) + |C| - 1,
CE(onehot(p), y) = CE(onehot(y), y) + 1[p != y].
```

Thus summed observed CE is `idealLoss + errors`; averaged CE excess is the
error fraction. Unobserved output classes remain in the denominator. This
identity concerns deterministic unit-scale logits, not general soft predictions
or variable scales. `ObservedRefinement/Basic.lean` proves the exact denominator,
row identity, table identity, loss-comparison equivalence and observation grouping.

Fix a decoder and column phases. For row `i` and candidate row phase `j`, let
`A_ij` count actual observed mismatches if that row receives `j`. A permutation
`pi` minimizes assignment cost if real potentials `u_i,v_j` satisfy

```
u_i + v_j <= A_ij             for every i,j,
u_i + v_(pi(i)) = A_i,pi(i)   for every i.
```

Every permutation costs at least `sum_i u_i + sum_j v_j`, and `pi` attains that
bound. `Assignment.lean` proves the certificate and connects its grouped matrix
to actual observed prediction error, including repeated observations and unused
axis values. A closed two-row witness strictly improves the old assignment.
The native solver must expose integer dual potentials for an independent checker
to test every inequality and selected equality; an assignment alone is insufficient.

For fixed phases, let `N_lc` count observed label `c` at latent state `l`.
Choosing a count maximizer for each state cannot increase observed error:
`sum_l N_l,d(l) + errors = number of observations`. Empty states contribute zero
for any decoder choice, providing no evidence for their missing labels.
`Decoded.lean` proves this identity and decoder descent, and combines the
assignment certificate with actual CE descent for both row and column updates.

Every resulting table is `onehot(d(r(i)+s(j)))`. CM14 supplies the same radius-N
operator and nuclear upper bounds for its latent moments; CM15 admits its
decoded table into the common convex decoder hull, even if it does not fit.
`phase_member`, `row_step`, `column_step` and `decoder_step` make feasibility and
observed CE descent explicit. Closed examples inhabit all conditional results.
Neither a recovered mode nor the true decoder is assumed; budgets are unchanged.

This certifies local steps, not globally successful discovery. Discrete
row/column/decoder search is not a jointly convex parametrization; the prediction
hull is convex. Ties can cycle and a capped search can stall. CM15's two exact
observed minima still disagree at an unseen cell. Retain earlier candidates on CE
ties, every seed and failure, full states and work ledgers. Native work needs
declared caps, independent integer assignment and raw latent/CE audits,
original-input replay and the 64 GiB cap. All 35 tasks, short/long variants,
three modes and the conventional transformer comparison remain required;
this source note asserts no new benchmark accuracy.

### CM24. Observable row balancing, missing axes and a capacity boundary

*Source-note extension for board task #218, 2026-11-13; Lean before native repair.*

The original nine-task CM23 cohort passes only three tasks. Its 54 exact
attempts refuse and all 162 interval attempts reach their caps. The latter
does not prove infeasibility. Feature row energies vary substantially; column
orthogonality does not give the required unit kernel diagonal. Division also
has one entirely unobserved row, which the earlier phase clock rejects.

For a supplied complex feature matrix `B` with `d>0` columns, measure

```
e_x = sum_i |B[x,i]|^2,
N(B)[x,i] = sqrt(d/e_x) B[x,i],
M_0 = diag(1/d,...,1/d).
```

If a row is nonzero, its energy is positive and the normalized row has energy
exactly `d` (`energy_pos`, `normalized_energy`). The matrix `M_0` is PSD with
trace one (`uniform_density`). Direct matrix multiplication gives
`K_B(M_0)[x,x]=e_x/d` (`uniform_reading`), so `M_0` is feasible for `N(B)`
whenever every row is nonzero (`normalized_feasible`). Consequently the exact
domain and every nonnegative CM22 tolerance domain are nonempty and convex
(`normalized_domain`). These proofs use observed feature entries only; no
unknown reference subspace, Fourier frequencies or correct phase is supplied.

A zero feature row gives kernel diagonal zero for every coefficient matrix,
so it cannot satisfy the original unit-diagonal constraint
(`zero_row_obstruction`). The explicit total construction `completeBalance`
replaces such a row by all ones and normalizes the other rows.
`completed_energy` and `completed_feasible` prove its measured energy and
feasible uniform density. Replacing a row changes the feature domain. It is
a declared convention, not a certificate about the old domain or an inferred
hidden row. A native policy may refuse a zero row with observed entries and
apply completion only when the corresponding input axis is unobserved.

Row normalization is nonlinear preprocessing. For fixed resulting features,
the density domain remains convex. It need not preserve column orthogonality,
centering, proximity to the true Fourier subspace, purity, objective gaps or
correct phase ordering. In particular, a mixed feasible `M_0` is not a
recovered hidden mode. CM21's pure extraction theorem still needs its exact
subspace and centering hypotheses; CM22's stability result still needs its
feature-error and objective-gap hypotheses. The general interval solver can
use these features, with raw matrix audits and every capped attempt retained.

For a sampled input axis `a : S -> n`, define `completeAxis a r f` to equal
`r` on the observed image of `a` and a declared fallback `f` elsewhere.
`completeAxis_observed` proves exact agreement on every observation. More
generally, if row and column phases agree at all sampled coordinates,
`phase_completion` proves equality of their actual ordinary full-vocabulary
CE and membership of the new decoded table in the same CM15 convex hull.
The latent operator and nuclear budgets remain radius `|n|` through CM14/23.
`completed_phase_loss` applies this to the explicit fallback construction.
This permits arbitrary missing-axis completion; it does not identify its
answers. Native permutation completion must separately check its returned
permutation and retain missing-axis metadata. All conditional statements have
closed witnesses, including unequal row energies and a wholly missing axis.

There is also a capacity obstruction in the narrower native permutation
family. In any additive group, if `s` is surjective and
`d(r(i_0)+s(j))=c` for every column, then `d(r(i)+s(j))=c` for every cell.
For a target latent argument, surjectivity supplies a column with phase
`-r(i_0)+target`; the constant row therefore covers every decoder argument.
`constant_row_forces_constant` and `no_constant_row` prove this obstruction
with inhabited examples. The same statement applies to columns after swapping
the axes in an abelian group. Thus a full table with a constant row and other
nonconstant rows cannot be represented by this surjective-axis family.
This is a representation limitation, independent of optimizer depth or budget.
It is not inferred from a partially observed constant-looking row. CM15's
larger hull allows arbitrary phase assignments and latent domains, so this
does not refute the hull. A generic categorical extension of the native
proposal family needs separate construction and checks.

Next native work must retain every prior candidate and failure, distinguish
normalization and fallback choices from convex optimization, replay original
inputs and all states/work, and independently audit raw spectra and CE. Freeze
tested source before another original-data run. Keep the 64 GiB process cap,
all35 short/long tasks, full/mini/forward modes and the transformer comparison.
This section establishes no new original-task score.

### CM25. Categorical constant strips in an enlarged latent domain

*Source-note extension for board task #219, 2026-11-13; Lean before native extension.*

CM24's obstruction concerns surjective phase assignments. A generic categorical
extension can represent constant input strips together with a nonconstant
interior without changing the ordinary output-vocabulary loss or imposing a
nonconvex constraint on the prediction hull. Let a proposed base table be
`d(r(i)+s(j))` with phases in `ZMod q`, `q>0`. Choose categorical row and column
flags and a proposed constant output `c`. These are proposed model components,
not true unseen answers supplied to a learner.

Enlarge the latent cyclic domain to `ZMod (3q)`. With `tag(false)=0` and
`tag(true)=1`, set

```
lift(flag,z) = 3 val(z) + tag(flag) mod 3q,
D(z) = d(val(z)/3 mod q) if val(z) mod 3 = 0, and c otherwise.
```

`lift_sum_val` computes the actual modular sum, including wraparound.
`tag_zero_iff` proves that its residue modulo three is zero exactly when
neither axis is flagged. One flag gives residue one; two give residue two,
so the intersection of the constant strips stays constant. For two unflagged
axes, modular division reduces to their original sum in `ZMod q`.
`decode_lift` consequently proves

```
D(lift(rowFlag(i),r(i)) + lift(columnFlag(j),s(j)))
  = c                            if either axis is flagged,
  = d(r(i)+s(j))                 otherwise.
```

This construction uses arithmetic on internal categorical phase coordinates,
not arithmetic features or an answer rule on the original input names. It
applies to an arbitrary output type, decoder and phase assignments. No flag
discovery, injective decoder, prime modulus or known group coding is assumed.

`tagged_member` places the actual output table in the CM15 decoder hull for
the declared larger latent domain. CM14 gives the same radius `N=|n|` operator
and nuclear bounds on every actual latent Fourier moment. The larger prediction
hull and ordinary CE remain convex by the existing CM15 results. Every earlier
decoded phase vertex embeds with both flags false (`old_vertex_member`);
this statement does not assert inclusion of every fractional latent table
from the older hull. Retaining an existing native candidate does not require
its input phases to be surjective in the larger domain.

`tagged_fit_minimum` proves that actual agreement on observed labels certifies
a global ordinary-CE minimum over the declared enlarged hull. It assumes the
checked observed agreement, not successful search or unseen-label correctness.
`tagged_loss` splits the exact unit-logit CE excess into errors on flagged
observations and errors in the interior. `tagged_descent` then transfers an
actual interior error decrease to full observed-CE descent with the strips
fixed. A cropped native learner must keep the original vocabulary, preserve
that observation partition and independently check the reassembled table.

The three-category closed witness has a constant first row and column, yet
different interior outputs at `(1,1)` and `(1,2)` (`witness_pattern`). The
intersection also has the constant output. This escapes the CM24 capacity
obstruction without asserting recovery of an unknown structure. Closed
examples inhabit every conditional theorem, including strict interior error
improvement and exact-fit optimality.

The native input and latent cardinalities must also be allowed to differ,
and several input categories may share a phase. `Refinement.lean` therefore
defines the actual mismatch cost for assigning each possible latent phase
to an input category. `grouped_errors` equates the sum of those grouped costs
with the original observation error, including duplicates and unused inputs.
`choice_descent` transfers nonincreasing per-input costs to actual observed
error descent. `choice_minimum` certifies the globally best update of one
axis when every possible choice in each cost row is checked. Unlike CM23's
permutation assignment, this update does not require different inputs to
receive distinct phases. `free_row_step` and `free_column_step` combine it
with ordinary CE descent and membership in the existing phase hull. Closed
witnesses assign two input categories the same phase in a three-state latent
domain. Finite independent minimization is a coordinate update, not a joint
global phase/decoder search theorem; empty observations provide no unseen label.

A native follow-up may propose flags from observed categorical agreement,
but a constant-looking partial row need not be constant at unseen entries.
Retain ambiguous, inconsistent and capped proposals. Keep every older candidate
and CE tie, record the enlarged latent domain and its resource bound, reconstruct
the full input/output/latent tables and audit every Fourier frequency. Native
discovery policies and the 64 GiB process cap must be frozen before original
data, preserving all35 short/long tasks, full/mini/forward modes and the
conventional-transformer comparison. No new benchmark score follows from this
capacity construction.

### CM26. Complete categorical fibers give phase-sum constraints

*Source-note extension for board task #220, 2026-11-13; Lean before native discovery.*

An empirical same-label group contains more information than independent scalar
sketches, but its observed size alone does not establish a complete decoder
fiber. Let `G` be a finite additive commutative group, `d:G -> C` an arbitrary
categorical decoder, `r:I -> G` an injective input-phase map, and `b:G` a fixed
opposite-axis phase. Define the actual full fiber

```
fiber(d,c) = {z in G : d(z)=c}.
```

Suppose a finite observed group `F` satisfies `d(r(i)+b)=c` for every `i` in
`F`. Its shifted phase image lies in this fiber (`observed_subset`). Translation
preserves injectivity (`shift_injective`). If the cardinality of `F` also equals
the cardinality of the actual full fiber, the image has the same size as its
containing fiber and therefore equals it (`complete_image`). No latent phase,
full fiber size or completeness is inferred merely from an observed count.

Summing the actual distinct latent phases now yields (`fiber_sum`)

```
sum_(i in F) r(i) + |F| • b = sum_(z in fiber(d,c)) z.
```

Consequently a complete size-`k` observed group in column `j`, whose actual
decoder fiber has mass `mu`, satisfies (`row_equation`)

```
sum_(i in F) r(i) + k • s(j) = mu.
```

With injective column phases, the corresponding complete group in row `i`
satisfies (`column_equation`)

```
k • r(i) + sum_(j in F) s(j) = mu.
```

The two orientations constrain the same phase coordinates. A common mass
across several labels is legitimate only when their actual full fibers all
have that mass. This optional balanced-fiber structure is an additional
categorical model hypothesis, not a consequence of a flat spectrum. The
decoder itself may merge many latent states; it is not supplied with numeric
semantics of the original token names. Closed examples include nonzero shifts,
a many-to-one fiber, and a nonconstant decoder on five latent states with a
four-element fiber of mass zero. `incomplete_fiber_counterexample` exhibits
an observed singleton in a three-element fiber whose shifted phase sum is
different, so dropping completeness would make the equation false.

For a proposed prime cyclic latent domain, these equations are linear over its
finite field. A native learner may guess multiplicities from observed groups,
assemble both orientations, solve a bounded affine system and enumerate a
bounded part of its nullspace. It must retain those guesses as guesses: an
incomplete group, a wrong domain or a nonbalanced decoder can invalidate the
intended structural interpretation. Zero or other nonidentifying solutions
may satisfy the linear equations. Further nullspace freedom need not consist
only of harmless phase shifts and rescaling. Every proposed endpoint therefore
needs independent checks of its original equations, phase injectivity, empirical
decoder coverage, actual observed fit and spectral budgets; passing these does
not by itself prove true unseen labels. No universal identification or efficient
global search theorem is asserted.

This is another categorical proposal mechanism inside the existing decoded
spectral hull. Any final phase table still uses CM23's unconditional phase
membership and actual ordinary-CE/error identity. Complete-group assumptions
explain when this initialization is structurally justified; they do not change
CE, prove a wrong guessed system sound, or replace the raw checks. Native work
must declare finite-field, multiplicity, anchor, nullspace and enumeration
budgets before its own original follow-up, retain the complete task219 parent,
freeze tested source, and preserve all35/modes/transformer scope and64GiB cap.
No new measured benchmark result is claimed by these equations.

### CM27. Composite categorical codes require checked modular lifts

*Source-note extension for board task #222, 2026-11-13; Lean before native discovery.*

Unknown categorical class codes can be proposed from observed relations
`r(i)+s(j)=t(label)` in a cyclic latent domain. Neither input names nor output
names carry arithmetic meaning. The domain may be composite, and repeated
input phases are allowed. A one-dimensional field kernel does not by itself
justify a solution modulo a prime power or identify any unseen label.

For a finite integer matrix `A`, write `A*x` for its actual row sums and
`Fits(m,A,b,x)` for `m | b(i)-(A*x)(i)` at every row. Suppose `x` fits modulo
nonzero `m`. Define the actual integer residual
`e(i)=(b(i)-(A*x)(i))/m`. For any integer digit vector `y`,

```
b - A*(x+m*y) = m*(e-A*y),
Fits(m*p,A,b,x+m*y) iff Fits(p,A,e,y).
```

These statements hold without assuming a field. They justify a checked digit
lift when the residual system is solved; they do not say the residual system
is solvable. The one-variable equation `2*x=0` admits `x=1` modulo2 but no
lift `1+2*y` modulo4. Inconsistency or an unsearched branch must remain explicit.

To recombine components, use actual integers `u,v` satisfying `u*m+v*n=1`.
The vector `z=v*n*x+u*m*y` retains `x` modulo `m` and `y` modulo `n`. If the
two vectors solve the same integer system modulo their respective moduli,
then `z` solves it modulo `m*n`. The Bezout witness is checked; arbitrary
noncoprime moduli do not justify multiplying the modulus. Agreement modulo
each component implies agreement modulo the product under this same witness.

There is also a gauge freedom valid in composite cyclic domains. For a unit
`u` with checked inverse `v`, choose categorical anchor phases `a,b` and set
`r'=u*(r-a)`, `s'=u*(s+a-b)`, and `t'=u*(t-b)`. Every actual observed relation
is preserved. An injective class code stays injective, and the transformed
decoder `d'(z)=d(v*z+b)` preserves every decoded prediction. No primality,
numeric meaning of category identifiers or supplied true decoder is needed for
these identities. Native discovery must still learn and check a code; an
unproved assertion that a kernel is only a unit orbit is not a certificate.

The proposed native extension may guess a cyclic domain from observed class
count, solve and normalize component systems, lift checked residuals, combine
checked residues and test class-code injectivity and every original equation.
Every component, normalization choice, residual, matrix, operation, failed
branch, coverage condition and cap must remain in its report. Actual phase
tables still require CM25's raw latent spectral and ordinary-CE checks. This
does not establish efficient global search, completeness of a capped branch
bank, true unseen labels or any new original-task score. Freeze the tested
native producer before original discovery and preserve all prior candidates,
full35/modes/transformer scope and the64GiB process limit.

### CM28. Identify the joint observed module before resolving CE ties

*Source-note extension for board task #223, 2026-11-13; Lean before changing selection.*

The frozen task222 cohort retains its earlier-first rule for equal training
CE. A newly constructed completion can fit the same observed labels and tie
an earlier completion that gives different unseen answers. Ordinary CE cannot
distinguish these tables from those observations. The following condition
identifies a completion within a specified categorical cyclic model; it does
not identify every member of the larger convex spectral hull.

For a modulus q, consider a vector with coordinates indexed by row names,
column names and class names. Let K be the actual module over ZMod q defined
by every observed equation r(i)+s(j)=t(label), together with r(i0)=0 and
t(c0)=0. These two pins fix shift gauges. Every row and column remains a
variable: an arbitrary zero assignment for an unobserved axis is not an
observed equation and must not enter an identification certificate. No numeric
meaning of input names or class names is assumed.

The observable condition is that K is cyclic: some vector g in K generates
all of K by scalar multiplication. Unlike field dimension one, this condition
is meaningful for composite q. If x in K has a coordinate x(v)=1, then x
itself generates K. Indeed write x=a*g and y=b*g. The equality a*g(v)=1
gives y=(b*g(v))*x by commutative ring arithmetic. This does not divide by an
arbitrary nonzero residue. An actual g with g(v)=1 and a trivial kernel after
adding the coordinate constraint x(v)=0 is a sufficient certificate of
cyclicity. The added constraint is for checking the kernel, not an assumption
that a proposed ground-truth code has that normalization.

Suppose the unknown target has a bijective class code e:C≃ZMod q and arbitrary
row and column phases r,s. Their values may repeat. Shift its fitted vector to
(r-r(i0), s+r(i0)-e(c0), e-e(c0)); it belongs to K. Bijectivity of e supplies
a class coordinate equal to one. Every other bijective phase completion of
the observed samples therefore has a gauged joint vector that is a scalar
multiple of this target vector. Adding its row and column equations and using
its class equation proves equality of decoded labels at every cell. No true
code, phase, unseen label or optimizer is supplied to the learner.

The specified observed-only learner first checks cyclicity of K, then returns
a compatible phase completion if one exists, otherwise refusal. Under unknown
structural realizability and the observable condition it returns the entire
table. Acceptance under this same structural model cannot return a wrong
table. The specification uses classical choice, as CM17 does; it is not an
assertion that a capped native search always finds a certificate or completion.
A genuine missing nonanchor row supports an independent vector with one at
that row and zero elsewhere. The target's bijective code makes this incompatible
with cyclicity, so the learner refuses rather than justifying an arbitrary pin.

The closed witness has seven categorical rows and columns, six latent states,
and a class code exchanging zero and one. Its distinct input names zero and
six have the same phase. Seventeen observed equations generate exactly the
one cyclic joint module; explicit elimination proves the condition. The learner
recovers all49 cells, including32 held-out cells and two distinct held-out
answers. A second, realizable witness shows all six classes in one row but
leaves other rows unobserved; it is correctly refused.

This theorem changes the admissible categorical completion program, not CE.
Discrete code or structural-certificate discovery is not jointly convex. The
old larger hull and its equal-CE ambiguity remain. A subsequent native selection
rule must declare the structural certificate and candidate priority before its
own run, independently check the original equations without missing-axis pins,
retain the frozen task222 parent and failures, and never choose from held-out
scores. No new original benchmark score follows from the proof alone.

**The actual matrix certificate.** Encode each observation as a row with one
at its row coordinate, one at its column coordinate and minus one at its
class coordinate. Append the two shift-gauge coordinate rows and one class
anchor row. A replayable certificate is a finite list of actual row swaps,
scalings and additions, together with a location for each coordinate row in
the final matrix. Every instruction preserves zero evaluations of the original
homogeneous equations, over any commutative ring. If replay produces all the
coordinate rows, an original kernel vector must therefore be zero. No
reported solver rank or solution count substitutes for these checked entries.

This connects directly to K: a vector in the original joint module whose
anchor coordinate is zero satisfies precisely the constructed matrix. If a
checked joint vector g has anchor coordinate one, the replay proves K cyclic.
It also proves uniqueness of any two compatible joint vectors sharing that
anchor value, by applying the certificate to their difference. Missing-axis
pins are absent from the matrix construction. A concrete six-instruction
certificate for a three-cell binary table identifies its genuinely held-out
fourth answer through the learner. A separate nonzero matrix over ZMod 6
starts with nonunit leading entries two and three; row addition exposes the
unit five, and the full four-instruction replay gives identity. No inverse
of two or three is assumed.

**Normalized component and digit uniqueness.** For an integer matrix A let
ZeroKernel(m,A) mean that every vector solving A*x=0 modulo m has all its
coordinates divisible by m. If m is nonzero and both ZeroKernel(m,A) and
ZeroKernel(p,A) hold, then ZeroKernel(m*p,A) holds. Given an actual solution,
the first condition writes x=m*y. CM27's exact residual identity gives A*y=0
modulo p, and the second condition proves each coordinate of y divisible by p.
Induction gives uniqueness at every power of a nonzero base. This proves
uniqueness only; the failed-lift example in CM27 still rules out an existence
claim. Independently, checked Bezout component agreement yields product
agreement. Taking the difference of two solutions proves their congruence at
every coordinate of the actual anchored system. Nontrivial triangular-matrix
witnesses instantiate prime-power and CRT uniqueness with distinct integer
representatives of the same solution.

These are exact Lean statements about replayed coefficients and original
modular equations. A native producer must still implement and independently
check the certificate, retain refusals and caps, and freeze its changed
selection policy before original evaluation. Passing a floating-point spectral
audit does not replace this integer certificate or prove structural truth.

**Scope of categorical strip views.** A fixed view consists of categorical row
and column maps into an interior, with a distinguished `none` value for strips
and a declared constant class there. The full model table uses its interior
when both maps have a value, and the constant otherwise. Applying the observed
interior learner and then this fixed map recovers an unknown full target
represented by that same view, provided its interior is realizable and its
actual joint module is cyclic. The maps may repeat interior names. A closed
example transports the six-state witness and gives both a strip answer and a
genuinely held-out nonconstant interior answer.

The constant-strip part is an explicit structural hypothesis about this full
target family, not a conclusion from the interior equations. Observed constant
support does not prove arbitrary unobserved strip answers constant. Different
proposed views carry different structural model assumptions. A native report
must therefore say that its certificate proves uniqueness within the declared
view; it must not turn an interior certificate into an unconditional statement
about unknown ground truth or the union of all proposed views. Unsupported
padding or missing axes remain refusals under the present certificate policy.

### CM29. Identify decoded answers through actual row consequences

*Source-note extension for board task #224, 2026-11-13; Lean before implementation.*

CM28's cyclic joint module is sufficient, not necessary. Identification of a
decoded answer only requires the corresponding query equation to follow from
the observations. It need not identify every coordinate of an unknown class
code. Keep an explicit structural family: an unknown injective class map e
into a commutative ring and phases r,s represent the target by
e(y(i,j))=r(i)+s(j). Injectivity is required; a general many-to-one latent
decoder is not covered by this representation.

Let Q(i,j,c) be the coefficient row with one at row i, one at column j and
minus one at class c. The original matrix has precisely Q(i,j,label) for each
observed cell. It has no gauge pins, missing-axis pins, rank requirement or
unknown-label rows. A certificate for candidate c is an actual finite row
program and a row position whose replay equals Q(i,j,c) coefficient by
coefficient. Every original fit remains a zero of the replayed matrix, so
r(i)+s(j)=e(c). Structural realizability and injectivity give y(i,j)=c.

A finite observed-only certifier tests these coefficient identities and
returns the first passing candidate, or refusal. Every returned answer is
sound within the stated model, regardless of the other candidates or the
joint module's dimension. If a finite bank contains a passing certificate
for every query, the entire returned table is correct. The candidate bank
contains only labels, row instructions and row positions; it does not store
correctness assumptions or receive the unknown code or target table.

The strict witness uses a four-state composite domain, four row/column names,
repeated row phases and all four observed classes. Fifteen cells are observed;
the lower-right answer is missing. Two row additions recover that query from
three actual observed rows. The gauged observed joint module is noncyclic:
besides its actual fitted vector with class-one coordinate one, it contains
a nonzero vector whose class-one coordinate is zero. Nevertheless every cell
has a checked finite row consequence. This is decoded identification without
unique normalized code coordinates, not an inference from flatness alone.

This extension leaves the frozen CM28 experiment unchanged. A future native
row-consequence producer must separately retain and check its original matrix,
bounded row programs and query identities, and freeze its own rule before an
original evaluation. No new benchmark pass follows from these theorems.

**Fixed categorical views for decoded consequences.** Apply the finite query
certifier to the interior of a declared view. A fixed class-name map sends
interior labels to the full output vocabulary; the strip constant may be outside
its image. If both full input names map to interior names, map that interior
query's certified label to its output name, or refuse; otherwise return the
view's declared constant. Under an unknown injective-code
interior target fitting the observations, every returned label agrees with the
full target represented by that same view. Observable certificate coverage at
all interior queries gives recovery at every full query. No cyclicity or unique
normalized code is needed. The constant-strip structure remains an explicit
target-family hypothesis, as in CM28; an interior row program does not infer
unknown strip labels. The four-state noncyclic witness extends to a fifth
categorical input name and a fifth output label absent from the interior.
It gives both that strip answer and the genuinely held-out interior answer.

### CM30. Noninjective class codes identify quotient answers

CM29 uses an injective class code to turn an observed row consequence into a
unique raw answer. The coefficient argument itself needs no injectivity. For
an unknown code `e : C -> R` and an unknown additive representation
`e(y(i,j)) = r(i) + s(j)` fitting every observed raw label, an actual row program
producing the query row for a proposed label `c` proves
`e(y(i,j)) = e(c)`. The certifier still consumes only the observed matrix and
its finite candidate bank. Neither the unknown code nor the target table is
an input to its coefficient check.

If the task has a declared readout `q : C -> D` constant on the fibers of `e`,
that equality gives `q(y(i,j)) = q(c)`. Finite checked coverage of all queries
therefore recovers all quotient answers through `Option.map q`. This extends
the same observed-only certifier to colliding class codes. Compatibility of
the original readout with the unknown code is an explicit structural premise,
not a learned fact inferred from low CE or flatness. The original raw-label
cross-entropy and the original benchmark metric remain unchanged.

The finite witness has three row names, three column names, four raw classes
`ZMod 2 x Bool`, and eight observations. All four raw classes occur. The
unknown code discards the Boolean alias bit; both axes have phases `(0,1,0)`.
Every cell except `(2,2)` is observed. Two target tables agree on all eight raw
labels and have the same code representation, but differ in the alias bit of
the unseen answer. Adding the observed `(0,2)` row to `(2,0)` and subtracting
`(0,0)` produces the actual query row for representative `(0,false)` at `(2,2)`.
The finite bank certifies every quotient answer for both targets. Its emitted
raw representative is wrong for one target. This is a counterexample to raw
label identification even with every raw class observed and complete query
certificate coverage, once injectivity is removed.

This extension proves a correctness boundary, not discovery of a useful
noninjective code or a native model repair. A future producer must find its
candidate code from training, check the original row identities, preserve
ordinary CE and all raw labels, and demonstrate compatibility with the
unchanged task's readout. No new benchmark pass follows from CM30 alone.

### CM31. Fit raw labels without discarding a learned quotient

There is a separate training-fit obstruction in a head that depends only on
a colliding latent code. If two observed inputs share their logit vector `v`
but have distinct raw labels `c` and `d`, ordinary full-vocabulary CE obeys

```
CE(v,c) + CE(v,d) >= 2 log 2.
```

Indeed, the softmax denominator is at least `exp(v_c)+exp(v_d)`, whose square
is at least `4 exp(v_c+v_d)`. Taking logs proves the bound. The bound is sharp
for two classes with equal logits. The actual empirical sum retains that
floor, since all other CE terms are nonnegative; its mean retains the floor
divided by the number of samples. A deterministic decoder of the shared code
also cannot get both raw labels right. These statements require neither a
budget nor an optimizer. The quotient model's `Model.logits` at `49105730`
has exactly this shared-state restriction. Correct decoded answers from CM30
do not remove the raw-label CE obstruction.

Let `O` be the finite set of distinct observed inputs, `Z` a nonempty finite
latent set, `z:X->Z` a fixed learned latent map, `y:O->C` the consistent recorded
raw labels, and `r:Z->C` a declared
representative decoder. Refine the categorical state to

```
s(x) = inl(x)   if x is observed,
       inr(z(x)) otherwise,
I = O + Z,     |I| = |O| + |Z|.
```

The feature extractor reads the observed input identities and `z`, without
any raw label or held-out target argument. Assign label `y(x)` to `inl(x)`
and `r(t)` to `inr(t)`. Distinct observed inputs now have distinct states,
and observed states cannot collide with defaults. Forgetting the observation
identity recovers `z(x)` exactly, so any old latent logit map has a pullback to
this state set. That representability statement does not promise the same
coefficient budget: the additional states have an explicit cost.

For a declared readout `q:C->D` and fixed code readout `g:Z->D`, assume

```
q(y(x)) = g(z(x)) for observed x,
q(r(t)) = g(t)    for every latent state t.
```

The constructed prediction fits every observed raw label and has decoded
answer `g(z(x))` at every input. To conclude correctness at an unseen input
still requires a separately identified answer `g(z(x))=q(target(x))`, for
example from an applicable CM30 certificate and its structural premises.
Unseen inputs with the same latent code still share a raw prediction. The
refinement does not discover `z`, prove representative compatibility from
low CE, or identify an arbitrary unseen alias.

Reindex the actual normalized `|I|`-state Fourier matrix by the categorical
state names. Both Gram matrices remain the identity, and every singular value
is one. With the assigned raw-label margin table `T`, the actual complex head
`H=U* T` has logits `Re(U H)=T`, using both real and imaginary coordinates.
Thus at every observed input its ordinary CE is exactly
`log(1+(|C|-1) exp(-a))`, with the assigned raw class uniquely preferred for
`a>0`. CFt5 gives squared energy `|I| a^2` and real/imaginary L1 column budget
`2|a||I|`; margin24 fits radius `64|I|`. Every feasible learned head's logits
are bounded by its column budget. The actual CE objective remains convex in
all coefficients for this fixed extractor. Flatness bounds this fixed basis,
not the target table or a learned head's spectrum.

The native vocabulary cap1024 gives `|C| < exp(24)` without a decimal
approximation. Consequently each assigned-class CE is strictly below log2.
For any other readout group, its unnormalized softmax mass is at most `|C|`;
the assigned class's group has mass at least `exp(24)`. Its pooled readout is
therefore strictly preferred, including when the groups have unequal sizes.
The common positive softmax normalizer does not change that ordering.

Ordinary training CE alone leaves default-state logits unconstrained. An
optional stronger domain intersects the convex column budget with affine
equalities fixing every `inr(t)` row's logits to the declared margin24 table.
This domain remains convex, contains the constructed initializer, and retains
convex ordinary CE. Every feasible updated head then preserves those actual
unseen logits and the compatible pooled answer. This is conditional on update
feasibility; no native projection algorithm or convergence result is proved.
It does not assert that unconstrained CE updates preserve the defaults.

The finite witness has eight Boolean input triples, four observations and
four held-out inputs. All four raw classes `Bool x Bool` occur among the
observations; their first bit is the latent code. Two targets agree on every
observation and on the quotient everywhere, but disagree on held-out alias
bits. The unrefined head has a two-example log2 mean-loss floor. The actual
six-state refined Fourier head fits all four observed classes below that
pairwise floor, lies in the per-class budget384 and gives both targets'
correct pooled answer on all inputs. Its raw prediction at a specified
held-out input is strictly wrong for one target. This is a boundary witness,
not a benchmark result or an unconditional discovery theorem.

Lean: `ConvexDepth/QuotientCollision`, `QuotientRefinement`, `QuotientHead`,
`QuotientReadout`, `QuotientDomain` and `QuotientWitness`. The construction
enlarges capacity to `|O|+|Z|` states and has a dense Fourier realization;
practical storage/work limits, actual constrained updates and original-task
evaluation require a separate frozen native implementation. Neither this
extension nor its examples change the original labels, loss or task metrics.

## CM32. Bounded block updates for the refined quotient head

CM31 proves raw interpolation and a convex domain that retains latent-default
logits, but leaves native update feasibility and storage open. The following
construction closes those exact-real obligations. It does not identify a
missing target or establish optimizer convergence.

Let `I` be a finite state set, `C` the original class vocabulary, and `U` any
fixed unitary complex matrix on `I`. For an arbitrary real table `T:I x C -> R`,
set `H(T)=U* complex(T)`. The actual real/imaginary head computes
`Re(U H(T))=T`. Its update transport is linear:
`H(aT+bV)=aH(T)+bH(V)`, including `H(T-eta G)=H(T)-eta H(G)`.
This is an identity for arbitrary directions, not a theorem that a particular
native implementation computes the CE derivative correctly.

Unitary inversion preserves each individual class column's squared energy:

```
sum_k (Re H[k,c])^2 + (Im H[k,c])^2 = sum_s T[s,c]^2.
```

If `|T[s,c]| <= A` and `A >= 0`, Cauchy--Schwarz over the `2N` real
coordinates, `N=|I|`, bounds the squared real/imaginary L1 column by
`2 N^2 A^2`. Thus its L1 column is at most `2 A N`. Unlike a bound using the
whole Frobenius norm across classes, this carries no vocabulary factor.
The native box `[-32,32]` therefore fits the declared `64N` coefficient
budget for every table in the box, not only for a one-hot initializer.

For `I=O+Z`, keep the same observed-identity refinement and latent map as
CM31. Given arbitrary proposed logits, clip each observed-state entry with
`clip_A(t)=max(-A,min(A,t))`; assign each default `inr(z)` its fixed row `d_z`.
Every result lies in the box if the default rows do. Feasible tables with
these defaults are fixed points of this operation. The domain consisting of
the box and the affine default equalities is convex. Ordinary full-vocabulary
CE of `H(T)` is convex in feasible table coordinates, with all original labels
unchanged. The clipping operation is part of the optimizer's constrained
update, not a trainable activation whose composition is asserted convex on
unconstrained proposal coordinates. Neither nearest-point optimality nor a
convergence rate is needed or proved by the feasibility argument.

To avoid a dense global Fourier matrix, assign each refined state to its old
latent code: `b(inl(x))=z(x)` and `b(inr(t))=t`. Every fiber is nonempty
because it contains its default. On each finite fiber, construct its own
categorical Fourier basis. Their direct sum, reindexed back to the original
state names, is exactly unitary; all singular values are one, and entries
between different fibers are zero. This reindexing contains no labels.

Writing `n_z=|b^-1(z)|`, the exact number of stored within-block phases is
`sum_z n_z^2`, while the state count remains `sum_z n_z=|O|+|Z|`. The phase
count is at most `N^2`; if every `n_z <= B`, it is at most `B N`. These are
conditional storage bounds, not an assertion that learned fibers are always
balanced or that any chosen hardware limit is automatically met. Two fibers
of size two provide a concrete eight-phase versus sixteen-phase example.

Use native default rows with24 at the declared representative class and zero
elsewhere. These rows lie in the32-box. For every proposal, the block head
satisfies the64N budget and retains exactly those logits at every unseen
input. If each representative has the declared quotient code and the vocabulary
has at most1024 classes, the compatible pooled readout remains strictly
preferred. If that code separately identifies the target quotient, the unseen
decoded answer remains correct at every projected step. The target occurs
only in this last theorem's identifying premise; it is not an input to the
extractor, constructor or update. Unseen raw aliases remain unidentified.

Initializing observed rows with24 at their recorded raw class and zero
elsewhere is unaffected by clipping. Each observed example has ordinary CE
strictly below log2 under the same vocabulary cap. Concrete examples include
unobserved inputs, negative/non-one-hot trained tables, and arbitrary proposed
entries1000 outside the feasible box. No claim is made that a later arbitrary
proposal retains observed classification; only its feasibility and defaults
are guaranteed. A native implementation still needs actual forward/gradient
checks, floating-point tolerance audits, resource preflights, saved-state
replay and the original success metrics before reporting numerical progress.

Lean: `ConvexDepth/BoxHead`, `BoxUpdate`, `FiberFourier` and `RefinedBox`.
Flatness bounds the fixed categorical basis and the transported head's
capacity; it is not a label oracle or a property imposed on the target table.

## CM33. Ordinary-CE updates preserve the refined observed labels

CM32 allows arbitrary proposed table updates. Such a proposal need not preserve
an observed raw winner. For the actual ordinary-CE derivative, observed-input
separation yields a stronger result: every nonnegative projected step preserves
all previously correct observed raw predictions and cannot increase any observed
row's CE. This is a property of the refined head, not of a colliding latent head
or a general model with coupled observed logits.

Let `O` be the finite set of distinct observed inputs, `y:O->C` their consistent
raw labels, and `Z` the finite set of latent defaults. Fix the actual unitary
complex feature matrix `U` on `I=O+Z`, for example CM32's block Fourier matrix.
With all original classes retained, define

```
p_c(z) = exp(z_c) / sum_j exp(z_j),
g_c(z,y) = p_c(z) - 1[c=y],
CE(z,y) = log(sum_c exp(z_c)) - z_y.
```

Differentiating the finite exponential partition along `z+a*v` gives the actual
directional derivative `sum_c g_c(z,y)*v_c`. For a selected minibatch `B` of
distinct observed inputs, the actual head `H(T)=U^* complex(T)` has batch loss

```
L_B(T) = (1/|B|) sum_(o in B) CE(Re(U H(T))[inl(o),:], y(o)).
```

Unitarity identifies its derivative in every full table direction `V` with
`sum_(s,c) G_B(T)[s,c]*V[s,c]`, where

```
G_B(T)[inl(o),c] = g_c(T[inl(o),:],y(o))/|B|  if o in B,
                  0                          otherwise,
G_B(T)[inr(z),c] = 0.
```

Thus this is the ordinary full-vocabulary CE derivative in real-table
coordinates, through the actual real and imaginary head coefficients. It is
not a derivative hypothesis or a masked-label surrogate. `B=O` gives full-batch
averaging; arbitrary subsets give minibatches, including the final smaller batch
with its actual cardinality.
No held-out input or target is used. The empty-batch boundary has zero gradient
and an identity step on feasible tables; native forward mode executes zero steps.

For any nonnegative rate `eta`, subtract `eta*G_B(T)` and use CM32's box clipping
with fixed defaults. At a selected observed row, the target derivative is
nonpositive because `p_y<=1`; every rival derivative is nonnegative because
`p_c>=0`. Before projection the target score therefore increases or stays fixed
and every rival decreases or stays fixed. Scalar clipping is monotone and fixes
each old feasible coordinate, so both inequalities survive projection. An
unselected row is unchanged. No upper step-size bound is required for this
separable-row preservation result.

Every target margin is consequently nondecreasing. To compare the actual losses,
use the exact relative-gap identity from JC3: the CE change is the logarithm of
the old-softmax-weighted exponential of the rival-gap changes. These changes are
all nonpositive, so the exponential average is at most one and CE cannot rise.
This conclusion holds for every observed row, including rows outside the current
minibatch. It is stronger than a statement only about the selected batch loss.

Define a finite trace by executing these derived gradients and projections from
a feasible initial table. Induction on the actual update recursion proves that
every iterate stays feasible, retains every initial observed margin, and has
nonincreasing observed row CE at each consecutive step. All default rows remain
fixed. The schedule and nonnegative rates may vary at every step; convergence,
fair sampling, or eventual visitation are not assumptions or conclusions.

For the native margin24 initializer, every actual complex forward pass therefore
retains an observed raw margin of at least24. Its observed CE stays below the
initial CE and hence below log2 when the original vocabulary has at most1024
classes. Every finite iterate satisfies the same64N real/imaginary L1 column
budget, `N=|O|+|Z|`, and keeps each unobserved input's declared default logits.
These defaults are not certified answers without the separate quotient-target
identification premise from CM31/CM32. Initializing from all observations is still
an explicit construction cost; these theorems do not turn it into online learning.

The claims are over exact real/complex arithmetic. Native finite-precision
transforms and gradient pullbacks still require their saved-state, derivative,
margin, default and budget audits. The proof supplies neither a floating-point
execution theorem nor a new benchmark success. It changes no frozen model,
training schedule, labels, evaluator or process already running.

Lean: `ConvexDepth/RowCE`, `BatchCE`, `ObservedCE` and `RefinedCE`. Examples include
nonzero complex-head directional derivatives, different full/minibatch
normalizations, alternating minibatches, large rates, and unobserved inputs.

## CM34. Retain the complete observed coefficient kernel

Task248 includes every4704 Fibonacci observation but leaves three coefficient
directions at modulus97. Requiring a unique direction is therefore stronger
than merely requiring an observed fit. Keep the complete compatible space
instead of discarding it because its dimension exceeds one. This extension
proves what its vector codes certify; it does not assert that every query has
a matching observed class or that an arbitrary chosen direction identifies it.

Let `I,J,C` be the independently compacted observed row, column and class
dictionaries. Let `cell:S->I*J` and `labels:S->C` contain the actual observed
equations. Coordinates are `I+(J+C)`, with a proposed query row acting by
`Q(i,j,c)(w)=w_row(i)+w_col(j)-w_class(c)`. Its explicit matrix coefficients
are exactly `+1,+1,-1` at those three different roles. Define `M` to be the
kernel of every observed `Q`, with only the two gauges
`w_row(i0)=0` and `w_class(c0)=0`. No missing axis or unknown answer is pinned.

Every fitting vector has a representative in `M`. Subtract `w_row(i0)` from
all row coordinates, add `w_row(i0)-w_class(c0)` to all column coordinates,
and subtract `w_class(c0)` from all class coordinates. Every query action is
unchanged, the gauges hold, and class-code equalities are unchanged. The
original observed matrix and these two harmless shifts, rather than a solver's
claimed answer or selected nullity, define the mathematical compatible space.

For a complete basis `b:K->M`, define vector codes by reading every direction:

```
r_b(i)[k] = b(k)_row(i),
s_b(j)[k] = b(k)_col(j),
e_b(c)[k] = b(k)_class(c),
z_b(i,j) = r_b(i) + s_b(j).
```

Then `z_b(i,j)=e_b(c)` if and only if `Q(i,j,c)` vanishes on every vector
in `M`. The forward implication uses linearity and the fact that the actual
basis spans `M`; the reverse implication evaluates at each basis element.
Consequently this matching relation is independent of which complete basis
represents `M`. The same argument applies to equality of two class codes.
Every original observation matches its recorded class vector. These statements
do not require a one-dimensional kernel, a nonconstant selected coordinate,
class-code injectivity, or a choice of a particular free variable.

To identify an unknown answer, retain CM30's structural premises. A map
`name:C->A` gives the original raw class name; an unknown code `e:A->R` and
axis functions `r:I->R,s:J->R` satisfy `e(y(i,j))=r(i)+s(j)` for the unknown
target, and the target agrees with every recorded raw label `name(labels(t))`.
Normalize the compatible vector `(r,s,e o name)` into `M`. A checked vector
match therefore gives `e(y(i,j))=e(name(c))`. For an unchanged declared readout
`q:A->D` constant on the fibers of this unknown `e`, the decoded answers agree.
The code and structural target are used in the soundness proof, not supplied
to the matching procedure. No assumption that every unseen raw alias appeared
in the observed class dictionary is needed. Without readout compatibility,
raw-label equality is still unjustified.

A finite ordered bank of observed class names can implement this check by
searching for exact equality of the full vectors. A successful result carries
the checked equality; a refusal is retained. Equality on one direction alone
is insufficient. A concrete two-dimensional diagonal-observation kernel has
class vectors `(0,0),(1,1)` and off-diagonal query vectors `(1,0),(0,1)`.
One coordinate can equate a query with the wrong class while the complete
vector check refuses it. Changing to another complete basis preserves these
decisions. Thus completeness of the basis is a real requirement, and does
not imply completeness of query coverage.

A separate binary-field grid observes raw labels0,1,1,2 at the four distinct
Boolean input pairs. Its equations force the complete vectors of classes0and2
to agree, while a nonzero compatible coefficient vector separates class1.
This alias has a deterministic raw target and a nonconstant additive code;
no raw label is absent from its observation bank. The earlier CM30 alias
witness also certifies the genuinely unobserved(2,2) query but retains two
possible raw answers. Complete-kernel matching does not remove that ambiguity.

For a finite active input domain, use the actual range of `z_b` as the latent
state type. Its cardinality is at most `|I|*|J|`, rather than necessarily
materializing every vector in `R^K`. This finite state map instantiates the
existing categorical/Fourier complex-head construction and CM31--CM33's
observed-input refinement, convex ordinary CE, bounded updates and default
retention. The refined head budget is `64*(|observed|+|range(z_b)|)`. A matching
class can supply a declared default; correctness of that default still uses
the structural premises above. Unknown axes outside the compact dictionaries
and unmatched states need explicit refusal/default policies in a native model.
The state-count and coefficient-budget bounds do not bound the storage of
the basis vectors or the sum of squared Fourier-fiber sizes. Those costs
still need explicit native forecasts and independent execution checks.

The native producer must compute and independently verify a complete basis
of the actual coefficient system, with explicit rank, work and storage caps.
This proof is not a finite-precision execution theorem or an original stress
test result. It changes no frozen232/237/243/244/248 protocol or recorded score.

Lean: `CoupledMoments/KernelCodes/{Defs,Basic,Diagonal,Identification,Witness,
Compact,ComplexHead}`. The complete kernel is defined by original equations;
the basis parameter is a standard spanning `Module.Basis` of that submodule.
Examples instantiate every theorem with hypotheses. The complex-head witness
uses a matched unseen query, noninjective unknown code, full-vocabulary pooled
readout and an actual finite ordinary-CE update trace. Setting trace length0
retains the same forward-only construction, with no gradient assumption.

## CM35. Derive readout-fiber constraints from compatible finite code cardinality

CM34 retains every direction of the original raw-class equations. Some of
those directions can violate the same unknown-code compatibility assumption
used by its soundness theorem. A known, fixed answer readout and finite
cardinality can turn that compatibility assumption into additional linear
class equalities. This extension does not infer a new readout from data or
replace the original raw-label cross-entropy.

Let `A` be the raw vocabulary, `D` a finite declared answer alphabet, and `R`
a finite code alphabet. The declared readout `q:A->D` is surjective. The
unknown code `e:A->R` obeys CM34's existing compatibility premise

```
e(a)=e(a')  implies  q(a)=q(a').
```

If `|D|=|R|`, there exists a bijection `phi:D≃R` such that

```
e(a) = phi(q(a))   for every raw class a.
```

This factorization is a conclusion. To prove it, choose a section `s` of the
surjective `q`. Compatibility makes `d -> e(s(d))` injective. Equal finite
cardinalities make this map surjective too, so every `e(a)` equals some
`e(s(d))`. Compatibility then forces `d=q(a)`. In particular,
`e(a)=e(a')` if and only if `q(a)=q(a')`. Raw aliases may differ, but their
unknown code coordinates cannot differ under these premises. No finiteness
of `A`, additive structure, observed coverage or target value is needed for
this cardinality argument.

Now let `R` also be a commutative ring. Retain CM34's explicit unknown target
family: raw target `y`, arbitrary categorical phases `r,s`,
`e(y(i,j))=r(i)+s(j)`, and exact agreement with every original observed raw
label. Applying the factorization gives

```
phi(q(y(i,j))) = r(i)+s(j).
```

Construct the original observed coefficient matrix with the recorded labels
mapped through the declared `q`, retaining every equation and exactly the two
harmless shift gauges. Keep a complete basis of its kernel. A full vector
match to answer `d` makes `phi(q(y(i,j)))=phi(d)` by CM34, hence `q(y(i,j))=d`.
The same argument supports independently compacted class names: their known
map into `D` must reproduce `q` of each original recorded raw label. This is
an explicit data-binding equality, not an assumption about an unseen answer.
The unknown `e`, `phi` and `y` enter only the soundness proof, never the kernel
constructor or class search. Missing matches still refuse.

This changes construction side information. CM34's frozen native protocols
use the declared readout only for eligibility and evaluation; a future CM35
proposal must explicitly accept the fixed readout partition when building its
observed equations. The partition is a generic map of output classes, not an
arithmetic function of the input names or an answer formula. Its surjectivity
and the code/answer cardinality equality must be checked. No claim is made
that an unknown many-to-one decoder can be discovered this way, or that the
unknown target really has a compatible additive code. Those remain structural
premises; low observed CE and a flat basis do not establish them.

**Strict additional coverage.** Use Boolean row and column names. Observe
three corners with raw labels `(0,false)`, `(1,false)` and `(1,true)`, in that
order at `(false,false)`, `(true,false)` and `(false,true)`. Let the declared
readout be the first coordinate in `ZMod 2`. The raw-class system, with its
two gauges, admits coefficients

```
r(false)=s(false)=e(class0)=0,
r(true)=x, s(true)=z, e(class1)=x, e(class2)=z.
```

Both `x` and `z` may vary independently, so the missing corner's vector
matches none of the three raw-class vectors in a complete basis. The actual
ordered class search refuses. After declared readout grouping, the two
last labels have the same class coordinate; the grouped equations force the
missing corner to code zero in every kernel direction. The query was not
observed. Both raw completions `(0,false)` and `(0,true)` at that corner fit
all observations and the compatible additive binary code. The pooled answer
is identified; the raw alias is not.

**Why the hypotheses matter.** With three code states and two answers, let
the unknown raw code be the identity on `ZMod 3`, and let the readout return
true only on state one. It is surjective and compatible, but states zero and
two have equal readout and different codes. Grouping them would add a false
code equation. Equal nominal cardinalities also do not suffice when the
readout leaves an answer unused: a constant Boolean readout is compatible
with the identity Boolean code, whose aliases have different codes. Finally,
a surjective identity Boolean readout and a constant Boolean code have equal
alphabet sizes but fail compatibility and cannot admit the stated bijective
factorization. The Lean counterexamples retain these distinct failures.

**Original raw CE and the actual complex head.** Use the finite actual range
of the grouped complete vector code as CM34's latent state. Choose a raw
representative of each declared answer before updates, and keep the separate
observed-input rows with their original raw labels. CM31--CM34 then give the
same actual Fourier coefficients, convex ordinary full-vocabulary CE, finite
observed-label preservation and `64*(|observed|+|range|)` coefficient budget.
No grouped-label loss is substituted. For an accepted unseen query, its raw
representative has the identified declared answer. The existing fixed-default
update theorem and pooled margin24 bound preserve that answer under any
finite full-batch or minibatch schedule. The zero-step instance is forward
only. An unsupported state or unknown axis still needs an explicit native
fallback; no correctness claim follows there.

The concrete head examples use all three distinct original raw observations,
show exact observation/head binding, and instantiate full updates, unequal
minibatches, zero updates, the coefficient budget and ordinary raw CE. The
unseen raw alias remains different from the chosen representative even when
its pooled answer is correct. All statements concern exact real/complex
arithmetic. Native quotient construction, raw-code expansion, finite-precision
state replay, resource forecasts and complete original comparisons require a
separately frozen implementation. The running232/252 cohorts and frozen237,
243,244 and253 evidence are unchanged.

Lean: `CoupledMoments/ReadoutKernel/{Factorization,Identification,Witness,
Limits,ComplexHead}`. Every theorem with hypotheses has a satisfying example;
there is no assumed factorization or new axiom.

## CM36. Coverage inclusion and the training-loss tie

CM35's declared-readout grouping supplies a structural ordering of two
complete observed systems. This ordering is separate from empirical CE.
Let the original recorded class names be `C`, the declared answers be `D`,
and let `q : C -> D` be the recorded-name readout. Keep the same observed
cells and row-first gauge, and use corresponding class gauges `c0` and
`q(c0)`. A grouped coefficient vector `(r,s,e)` pulls back to the raw vector
`(r,s,e o q)`. Every grouped observed equation and both gauges then become
exactly an original observed equation or gauge. This linear pullback uses no
unseen target, surjectivity or cardinality premise.

If a query matches raw class `c` in a complete basis of the original kernel,
its signed action vanishes on every original-kernel vector. Apply that fact
to each pulled-back grouped vector. The query therefore matches `q(c)` in
any complete grouped basis. Searching the mapped bank cannot refuse a query
accepted by the original bank. This is inclusion of algebraic match coverage;
it is independent of the particular complete bases. The strict three-corner
example of CM35 shows that the inclusion can be proper. Native construction
caps or observed-group incompatibility checks can still refuse a candidate:
coverage inclusion does not assert that every proposed native model exists.

This algebraic ordering alone does not identify the unknown target. Correct
answers on the enlarged domain still require CM35's additive realizability,
observed fit, globally compatible unknown code, surjective declared readout
and equal code/answer cardinality. It does not justify replacing arbitrary
earlier models with unrelated representations, nor treating a fixed fallback
as a certified match.

The actual complex observed-input refinement exposes a distinct limitation
of training-loss selection. CM33's normalized ordinary-CE step updates an
observed row only from that row, its raw label, its batch membership and the
actual batch size. Two tables with the same initial observed rows therefore
retain identical observed rows through every common finite schedule, even
when their latent types, defaults and Fourier blocks differ. Unitarity turns
this table identity into equality of the actual real/imaginary forward logits.
Starting both heads at their recorded raw margin24 tables yields identical
observed logits and full-vocabulary empirical CE. This includes full batches,
unequal minibatches and zero updates. It does not assert bitwise equality of
native floating-point results; the native comparison uses a declared tolerance.

In the three-corner example, give the ungrouped search the explicit fallback
class1 when it refuses the missing corner. Give the grouped search the
representative of its accepted answer0. Both actual bounded complex heads use
the same three original raw observations and have equal empirical CE at every
common finite schedule. Nevertheless their fixed unseen defaults prefer
different declared answers. The grouped answer agrees with both compatible
raw completions, while the ungrouped fallback is wrong. This is an exact-real
model-class witness with a stated fallback, not a claim about the native
first-class fallback on every task. The existing64N coefficient bounds apply
to both actual heads, with their respective finite state counts.

Consequently an earlier-first CE tie rule may retain a less informative
completion. Coverage inclusion can motivate an explicitly declared structural
prior in a future protocol; it does not change the frozen parent-first260
comparison or certify a new benchmark score. The ongoing252/245 runs and all
frozen sources remain unchanged. A prospective native tie policy must retain
the original comparison, verify its own certificate/data bindings and report
all35 primary outcomes, standalone candidates and every regression.

Lean: `CoupledMoments/ReadoutKernel/{Coverage,LossTies,TieWitness}`. The proofs
use the actual observed kernels, matrix heads and normalized CE updates;
unknown-target premises remain explicit and no new axiom or unproved input is
introduced.

## CM37. Independent gauge anchors preserve complete-kernel coverage

CM36 initially compares corresponding raw and grouped gauges. The current
native implementation keeps each dictionary in first-observation order, and
grouping preserves that observation order. Its two anchors therefore already
correspond. Independent valid certificates could instead choose different row
or class anchors. The stronger coverage result below removes this convention;
it is not a correction of an observed native gauge mismatch.

Use CM34's existing normalization on the pulled-back coefficient vector. For
any fitting raw vector `(r,s,e)`, raw anchors `i0,c0` give
`r'(i)=r(i)-r(i0)`, `s'(j)=s(j)+r(i0)-e(c0)` and
`e'(c)=e(c)-e(c0)`. Every signed query action `r(i)+s(j)-e(c)` is unchanged.
Consequently all original observed equations survive, while the two raw
anchor coefficients become zero. No unseen coordinate is fixed.

Start with a grouped fitting vector having any row anchor and any grouped
class anchor. Pull it back through the declared readout and normalize it to
the raw anchors. This produces an element of the actual raw observed kernel,
with the original grouped query action unchanged. A full raw-basis match
vanishes on that normalized vector, so it also vanishes on every grouped
fitting vector. Thus CM36's raw-to-grouped match and mapped-bank acceptance
inclusions hold for independently chosen anchors.

With the identity readout, applying the argument in both directions gives
exact query-match equivalence between any two gauge choices and complete
bases. Induction on the unchanged ordered class bank then proves equality
of the actual search result, including refusal. Changing the bank's order
is outside this claim. Concrete three-corner examples change both anchors,
retain an observed match and preserve the raw missing-corner refusal.

This removes a coordinate convention from the structural coverage result;
it supplies no unknown-target identification or benchmark success. Native
basis completeness, observed data, declared readout and actual model bindings
still require their numerical checks. CM35's target-family premises are
unchanged, as are the frozen parent-first comparison and all original runs.

Lean: `CoupledMoments/ReadoutKernel/Gauges`, using CM34's actual normalization
and CM36's explicit linear coefficient pullback.

## CM38. Keep a frozen parent outside candidate support

The complete causal comparison in task245 replaces a successful forward-only
associative-divisor parent with a lower-training-CE model whose primary accuracy
is53.86598%. Its report retains2,148 unknown validation prediction calls. That
count is not by itself an attribution of each error. It motivates a separately
declared fallback model; it does not authorize changing the frozen outcome.

Let `g(x)` be a fixed Boolean support predicate obtained from the candidate's
observed structure. Let `phi(k,x)` and `psi(l,x)` be the candidate and parent
features, with coefficient blocks `W` and `P`. Define the combined features on
the disjoint coordinate union by `(phi(x),0)` when `g(x)` is true and by
`(0,psi(x))` otherwise. Store the actual combined coefficients `(W,P)`. Expanding
the finite dot product gives exactly the candidate logits on supported inputs
and the parent logits elsewhere. No answer or held-out score enters this gate.

For complex component matrices, realification gives the same identity for
the real parts of the actual complex matrix products, with both real and
imaginary coefficients free. Every combined real tensor has such a pair of
complex blocks. Both stored blocks are charged: per-class absolute-sum budgets
`Rc,Rp` give combined budget `Rc+Rp`, and the number of real coefficient slots
is `2*(nc+np)*|C|`. A sparse prediction call does not eliminate stored parameters.

If the component codes are unitary, their realified coordinates have absolute
value at most one. Routing only zeroes inactive coordinates, so the same
bound holds for the combined features. Every feasible combined logit is
therefore bounded by the charged coefficient radius. This uses flat component
spectra as a capacity bound; the routed matrix over inputs need not be unitary.
The corresponding nonnegative common feature bound also applies to components
whose bounds are established separately.

Intersect the combined absolute-sum budget with affine coordinate equations
fixing the entire parent block to its saved value. This domain is convex and
contains the combined point whenever both original heads meet their charged
budgets. Ordinary full-vocabulary CE remains convex on this domain because
the complete coefficient tensor still enters the logits linearly. Every
feasible finite sequence preserves the parent's exact logits wherever `g`
is false. Candidate updates, discovery of support, and feasibility of a native
numeric trace remain separate checks; no convergence theorem is asserted.

For an original batch of record IDs, average CE by that batch's cardinality.
Distinct IDs retain repeated input/label observations. If every record in the
batch has `g=true`, the combined batch objective is exactly the candidate's
objective as a function of every candidate coefficient. Its Frechet derivative
therefore agrees, and any actual derivative-existence proof transfers. The
statement retains the full original raw label vocabulary and the actual
normalizer, including unequal final minibatches. An empty batch has zero
objective by the zero-inverse convention; it supplies no learning claim.
Forward-only prediction uses the same initial combined point with no updates.

Support is not a correctness certificate. A concrete three-input witness
observes only input0 with raw label false. The candidate maps inputs0 and1 to
state0 and input2 to state1. Its observed-state support test therefore accepts
the unseen input1 but rejects input2. The parent keeps three distinct states.
Both actual coefficient blocks use the proved four-state flat Fourier matrix,
margin24, and state labels false at0 and true elsewhere; their combined point
fits budget512. They have equal ordinary CE on the observed input. Nevertheless,
for the explicit target false/true/true, the supported unseen input1 changes
from the parent's correct prediction to the candidate's strictly wrong one.
The unsupported input2 retains the exact parent logits. This is a bounded
model-class counterexample, not a claim to learn the parent from one record.

Consequently the fallback supplies exact preservation only on the unsupported
region. Native saved-state loading, observed-support binding, combined resource
accounting, and complete original metrics must still be checked. Any supported
regression remains visible. The245 comparison,252/255 complete-kernel protocol,
260/261 grouped comparison and their CE selectors remain frozen. A native
fallback or structural tie prior requires a separate prospective model and
complete35-task comparison before any improved benchmark count is claimed.

Lean: `ConvexDepth/Fallback/{Basic,Domain,Batch,Witness}`, composed with CD8,
CFt5 and their actual complex-head and ordinary-CE definitions. The fixed route
and feature library are not jointly optimized as raw transformer parameters.

## CM39. Extend coverage while preserving every other prior prediction

CM38 preserves the prior only where the candidate refuses. Its counterexample
shows why accepting every candidate-supported replacement can overwrite a
previously correct answer. A separately declared conservative extension uses
two fixed support predicates: `s(x)` for the prior and `t(x)` for a supplemental
decoder. Neither predicate contains a target label or depends on trained head
coefficients. Set `keepPrior(x) = s(x) or not t(x)`.

Instantiate CM38 with the prior as its free first block and the supplemental
head as its frozen second block. Thus the actual disjoint-union features are
`(phi(x),0)` when `keepPrior` is true and `(0,psi(x))` otherwise. The finite
complex dot product preserves the exact prior logits whenever the prior has
support, including overlap with a disagreeing supplement. It also preserves
the prior wherever both decoders refuse. Only inputs with `s=false,t=true`
use the supplemental logits. Conversely, any actual logit change implies
both of these conditions; newly covered inputs may still have identical
logits by coincidence. No inclusion relation between the supports is needed.

The construction retains CM38's independent real and imaginary parameters,
both stored coefficient blocks, their summed column budgets and their exact
coordinate count. Freezing the supplemental block gives the same affine
convex domain. Flat component matrices provide the same bounded realified
features and feasible-logit bound; the routed input matrix need not be flat.
For any original batch whose records have prior support, the normalized
ordinary raw-vocabulary CE equals the prior's objective for every free prior
coefficient. The Fréchet derivatives therefore agree as well. Repeated inputs
remain separate record IDs, and the actual batch size is unchanged, including
shorter final minibatches. This allows reuse of an already validated prior
trace; it does not prove native optimization, convergence, or trace feasibility.
The identities also apply at the saved zero-update forward point.

A concrete witness uses the proved four-state Fourier code for both heads,
margin24 and combined coefficient budget512. The observed inputs are0 and1,
with respective labels false and true. The prior aliases unseen input2 to
observed input0, but maps unseen inputs3 and4 to unobserved states. The
supplement aliases inputs2 and3 to observed input1 and maps input4 to an
unobserved state. Their support tests are exactly membership in each map's
observed-state image. The supplemental logits strictly disagree with the
prior on all three unseen inputs. The extension preserves input2 because the
prior supports it, changes input3 because only the supplement supports it,
and preserves input4 because both refuse. Full and unequal minibatch examples
retain three original records, including a repeated input0, and their actual CE.

This is preservation of the prior outside the newly covered region, not
identification of targets inside it. Even a supported prior may be wrong,
and a newly accepted answer may be wrong. CM35's explicit target-family
assumptions, or other independently justified identification assumptions,
are still required for correctness. The theorem therefore does not promise
nondecreasing total benchmark accuracy. CM36/37 can establish coverage
inclusion for the related complete-kernel decoders, but cannot turn this
preservation identity into a target certificate by themselves.

Lean: `ConvexDepth/CoverageExtension/{Basic,Witness}`, using the actual CM38
complex realification and raw-CE definitions. A native policy would still need
both saved heads, independently bound support tests, complete cost accounting
and a prospective comparison. The original255/261 strict-CE selectors,267
fallback protocol and all frozen sources and runs remain unchanged.

## CM40. A shared categorical encoder restricts the complete observed kernel

CM34 retains all coefficient directions consistent with the original observed
equations and two shift gauges. CM35 derives additional equalities between
class codes from an explicitly declared compatible readout. Even after that
grouping, separating every observed class need not determine an unseen answer.
A further architectural assumption can restrict the input roles themselves.
It must remain an explicit premise about the unknown target family.

Let `align : T -> I x J` list pairs of row and column names representing the
same categorical input identity, with anchor `t0`. The names in `I` and `J`
remain independent dictionaries. For a coefficient vector `w=(r,s,e)`, add
the actual homogeneous linear equations

```
(r(align(t).row) - s(align(t).column))
  - (r(align(t0).row) - s(align(t0).column)) = 0    for every t.
```

Each row has four signed coordinate coefficients; coinciding coordinates
cancel. The resulting submodule `roleKernel` intersects the original
`observedKernel` to give `sharedKernel`. Neither its matrix nor its membership
predicate contains an unobserved query answer. If the two positions use one
categorical encoder `h` plus independent offsets, `r(i)=h(token(i))+a` and
`s(j)=h(token(j))+b`, aligned token identities give these equations. Their
common row-minus-column offset is `a-b`, which is not fixed to zero.

CM34 normalization sends `r` to `r-r(i0)`, `s` to `s+r(i0)-e(c0)`, and `e`
to `e-e(c0)`. Each aligned difference changes by the same quantity
`-2*r(i0)+e(c0)`. Thus every four-coordinate role action is unchanged, and
role-kernel membership is equivalent before and after normalization. An
original fitting vector satisfying the role equations normalizes into the
actual intersection, without losing either original gauge invariance.

For any complete basis of this intersection, the existing full-vector
query/class match is equivalent to vanishing query action on every member
of the intersection. Every original observed match remains. Every match in
the unrestricted complete kernel also remains, because the intersection is
a subspace of that kernel; new matches are possible. This coverage statement
does not say that distinct observed class codes remain separated.

Suppose an unknown raw target `y` has global additive realization
`e(y(i,j))=r(i)+s(j)`, fits the original observed labels, and its input codes
satisfy the stated shared-role equations. Normalize its packed coefficient
vector. A complete restricted-basis match to an observed class then proves
equality with that class's unknown target code. A compatible many-to-one
readout transfers this equality to the declared answer, while raw aliases
can remain ambiguous. CM35's surjective readout and equal finite code/answer
cardinalities also derive the required answer-code equivalence for independently
compacted grouped class names. The unchanged finite observed-class search is
sound conditional on returning a match. A missing match remains refusal.

The coverage improvement has a concrete matrix witness. Take two Boolean
input dictionaries, observe only `(false,false)` and `(true,true)`, and give
both the same observed class. In the independent-role kernel, the vector
`r=(0,1), s=(0,-1), e=0` satisfies both observed equations and both gauges but
has nonzero action at `(false,true)`. Hence the complete old code refuses
that cross query over the field of three elements. With aligned-role sharing,
write the two diagonal actions as `q00,q11` and the role action as `d`. Then

```
2 * (r(false) + s(true) - e) = q11 - d + q00.
```

The right side vanishes in the actual intersection. In a commutative ring
without zero divisors and with `2 != 0`, the cross action therefore vanishes.
The complete shared code matches the same observed class, so the original
search changes from refusal to acceptance at a genuinely unobserved input.
Over the field of two elements, the opposite vector itself satisfies role
sharing and still has nonzero cross action. The restricted code correctly
refuses there. Generic conditional soundness did not need the nonzero-two
premise; this particular coverage result does.

For a concrete many-to-one realization over the field of three elements,
take raw labels `(code,alias)` and read out the code. Two shared-role targets
have code zero everywhere and agree on the observed diagonal raw labels,
but choose different aliases on the cross cells. The derived decoded answer
is zero in both worlds; the raw answer is unidentified. An independent-role
target using the opposite vector also fits the same observations but has
a different decoded cross answer. Its actual role action is nonzero. This
shows why observed fit cannot justify applying the new family to every task.

A training-only diagnostic on the reviewed task 259 prime-97 grouped code
has complete dimension two. Enumerating all 98 projective directions leaves
97 that separate its 97 observed groups. Requiring one common offset across
all 191 identical token IDs in the two role dictionaries leaves just direction
`[1,36]`, with offset 62 in that saved basis. This motivates constructing a
restricted-kernel candidate. It neither checks an unseen label nor establishes
the unknown target's global shared-role premise. The input pin, generator and
reproduction command are in `experiments/convex_stress_role_sharing/README.md`.

Lean: `CoupledMoments/SharedRoles/{Basic,Identification,Witness,AliasWitness}`.
These results concern the actual linear coefficient
kernel. A future native candidate must construct and audit its complete basis,
check class separation/readout compatibility and resource bounds, then bind
the resulting fixed complex feature library and ordinary cross-entropy traces.
Flatness continues to bound that feature model; it supplies no unseen labels.
The fixed-feature convexity results are unchanged. The existing 255, 261 and
267 comparisons and the frozen CM39 component remain separate and unchanged.

## CM41. Keep the candidate outside a saved predecessor's input domain

Two Boolean cases in the first task277 shared-role comparison fail before
scoring because the chosen count predecessor has a shorter input domain than
an original evaluation split. CM38 assumes total parent features. A saved
finite count alphabet is a genuinely partial parent; zero extension alone
would select zero logits on an unsupported input outside that alphabet. The
failed frozen run remains evidence. A corrected model needs a different,
prospectively declared route, with no change to original labels or metrics.

Let `d : X -> Bool` be the saved parent's fixed input-domain predicate, and
let its actual features have type `psi : L -> {x | d(x)=true} -> R`. Extend
these coordinates by zero outside that subtype. Keep the candidate predicate
`g` separate from domain availability. The combined candidate gate is
`g(x) OR NOT d(x)`. Apply CM38's actual disjoint-coordinate construction to
this gate, the total candidate features, and the zero-extended parent. The
finite dot product has three explicit cases: candidate logits on supported
inputs, parent logits on unsupported inputs in the parent's domain, and
candidate logits on inputs outside the parent's domain. No unavailable parent
coordinate is evaluated, and availability does not assert candidate support.

The actual stored coefficient blocks still have joint absolute-sum budget
`Rc+Rp`. Zero extension preserves any nonnegative coordinate bound established
only on the parent's own domain. The routed coordinates inherit a common
bound `M`, and the combined logits are bounded by `(Rc+Rp)*M`. In particular,
unitary complex component codes give the same bound of one for their realified
coordinates, even when the parent's state map exists only on its accepted
inputs. This uses flat component spectra for a bound; it does not claim that
the routed input matrix is unitary.

The ordinary raw-label CE averaged over a finite batch of original record IDs
is convex on the affine domain freezing the parent coefficients. If every
record is candidate-supported, the actual normalized batch objective equals
the candidate's objective for every candidate coefficient tensor. Its Frechet
derivative agrees, and a candidate derivative-existence proof transfers.
Distinct record IDs retain repeated observations; the full batch and unequal
minibatch tails retain their own denominators. The new routing rule adds no
optimizer step and applies to the same zero-update forward-only state.

A finite nonzero witness distinguishes the policies. Take inputs `0,1,2,3`,
candidate support `x<2`, and parent domain `x<3`. With two coordinates, candidate
features `phi(k,x)=k+x+1`, partial parent features `psi(k,x)=2*k+x+1`, candidate
coefficients one and parent coefficients two, the new routed logit at input3
is9. Routing only by candidate support into the zero extension gives0 instead.
The conditional budget and coordinate bounds have concrete satisfying examples.
No part of this witness or the routing theorem identifies an unseen label.

Lean: `ConvexDepth/PartialFallback/{Basic,Batch}`. A native implementation must
bind the domain to the actual saved parent representation, retain separate
support/availability/route diagnostics, charge both heads, and verify long
prefixes without evaluating a parent outside its domain. The frozen CM38
component, comparisons255/261/267, and the initial shared comparison277 remain
unchanged. A corrected complete comparison remains a separate experiment.

## CM42. Interpolate fixed context features to distinguish raw aliases

The reviewed task281 comparison passes all 35 original primary success
criteria in each training mode and initializer policy at seed1. This does not
mean its raw-label probabilities are accurate. In the recurrence task the
selected endpoint has 4701/4704 correct field answers, but only 3120/4704
correct raw labels and validation CE about 8.0816. Its matched unseen state
always uses the first observed raw label in the latent block. Different
signed differences can share that block. Observed state refinement fits both
aliases without providing a rule to choose between them on unseen inputs.

Let `I` be a finite collection of actual source states, `phi(k,i)` their fixed
realified complex features and `W(k,c)` the complete trained output tensor.
Choose a fixed context map `w(x,i) >= 0` with `sum_i w(x,i)=1`. It may be built
from observed inputs and a declared input metric and latent partition, but
must remain fixed throughout CE optimization and must not inspect an unseen
target. Define

```
phi_w(k,x) = sum_i w(x,i) phi(k,i)
z_W(x,c)  = sum_k W(k,c) phi_w(k,x).
```

Expanding the actual finite products gives
`z_W(x,c) = sum_i w(x,i) z_W_source(i,c)`. This is a linear feature
transformation with the same complete coefficient tensor. It is not a
product of two learned factors. If every source coordinate has absolute
value at most `M`, each interpolated coordinate does too. The original
per-class absolute-sum head budget `R` therefore bounds each actual logit by
`R*M`. Unitary complex source codes give `M=1` for the real/imaginary
coordinates. The context matrix itself need not be unitary. Flatness is
used only for this bound; it does not choose a neighborhood or an answer.

The usual raw-label CE remains convex in every head coefficient on the
original convex budget. Jensen gives the separate query inequality

```
CE(z_W(x), y) <= sum_i w(x,i) CE(z_W_source(i), y).
```

Every term on the right uses the query's raw target `y`. It cannot be replaced
by the neighbor's own observed label when that label differs. Low CE at all
positively weighted source states for this same `y` transfers to the query.
Likewise, a common source margin against a competitor transfers. These are
explicit local raw-label assumptions, not conclusions from spectral flatness
or low observed training loss. With the original conjugate initializer of
margin `a`, the actual logit is exactly `a` times the total context weight
assigned to source states bearing that raw label. A strict raw-label weight
advantage gives a strict prediction margin when `a>0`.

On each original observed input, require a Dirac row at its actual source
state. The complete ordinary CE objective then equals the original one for
every coefficient tensor. This applies to any finite batch of original
record IDs, so repeated examples and short final minibatches retain their
original multiplicities and denominators. Frechet derivatives agree, and
derivative existence transfers wherever established for the source objective.
No extra optimizer step is needed: the same change of query features applies
to full-batch, minibatch and zero-update forward-only endpoints. Observed
loss equality alone provides no training-only preference between different
interpolation maps; any neighborhood selection needs a separately declared
rule, such as a training-only leave-one-out diagnostic.

A concrete witness has two observed source states with raw labels 0 and 2
in a four-class vocabulary. The declared readout is reduction modulo2, so
these raw labels share an answer. Contexts0 and3 are observed. A fixed
neighborhood map sends contexts0,1 to the first source and2,3 to the second.
A normalized two-state complex Fourier code and its actual conjugate head
produce raw margin4 for label0 at unseen context1 and for label2 at unseen
context2. Their ordinary CE is `log(1+3*exp(-4))`. Another possible target
agrees at both observations and every declared readout but exchanges the
two unseen aliases. Thus the decoder can exploit locality when it holds;
the same construction does not establish that locality from the observations.

Lean: `ConvexDepth/ContextInterpolation/{Basic,Loss,Batch,Witness}`. A future
native proposal must bind the fixed context metric and weights, retain exact
observed rows, evaluate actual complex products, charge all stored coefficients
and interpolation costs, and report raw accuracy and ordinary CE alongside
the original success criterion. An ordinal input metric is extra declared
structure and is not invariant under arbitrary token permutations. Missing
latent matches and unsupported axes still require explicit fallback. No
unseen-label guarantee or new numerical improvement is asserted here; the
reviewed task281 endpoints and the running older comparisons remain frozen.

## CM43. Freeze confidence scales before ordinary complex-head CE training

The reviewed CM42 comparison improves Fibonacci raw-label accuracy from
3120/4704 to4574/4704 and ordinary validation CE from about8.0816 to0.2800
in full-batch, minibatch and forward-only modes. Primary field accuracy stays
4701/4704 and all35 tasks pass. The remaining130 raw errors motivate examining
confidence as well as local alias selection. No new numerical gain is assumed.

For any fixed features `phi(k,x)`, freeze an input-dependent scale `t(x)` and
use `phi_t(k,x)=t(x)*phi(k,x)` with the same complete coefficient tensor `W`.
Expanding its actual finite dot product gives
`z_t(x,c)=t(x)*z(x,c)`. If `0 <= t(x) <= 1` and every original coordinate is
bounded in absolute value by a nonnegative `M`, the new coordinate has the
same bound. The original per-class absolute-sum coefficient budget `R` still
bounds the logits by `R*M`. Applied after CM42's stochastic context map, flat
unitary complex source features therefore retain the bound of one. Neither
context interpolation nor scaling makes the query matrix unitary.

At a strictly positive scale, every strict and non-strict raw-logit comparison
is equivalent to the original comparison. Thus raw argmax predictions and
ties are preserved for a fixed context map and tie rule. This does not preserve
comparisons between pooled probabilities for a many-to-one task readout.
Selecting a different context map can also change raw predictions. Both
metrics must be measured in any new comparison.

With `t` fixed, ordinary full-vocabulary CE averaged over any finite batch of
original record IDs remains convex in all coefficients on the same budget.
Separately, for a fixed complete logit vector `z`, `CE(t*z,y)` is convex in
the scalar `t`. Jensen yields

```
CE(t*z,y) <= t*CE(z,y) + (1-t)*CE(0,y),  0 <= t <= 1.
```

This inequality is not monotonicity: shrinking can increase a correct
prediction's CE. No joint convexity in trainable `t` and `W` is asserted.
Any proposed scale and context selection is frozen from original training
inputs and labels, independently of the trained coefficients and held-out
targets. For the actual conjugate complex initializer with margin `a`,
CM42's weighted-label identity becomes `t(x)*a*labelMass(w,labels,x,c)`.
This permits a declared canonical training diagnostic without a new optimizer
step or a table of unseen targets. It is not a generalization certificate.

Require `t(x)=1` on each original observed input. The complete normalized
batch objective then equals the original objective for every coefficient
tensor. Its Frechet derivative agrees, and any actual derivative-existence
proof transfers. Combining this premise with CM42's observed Dirac rows
recovers the original source-state objective. Full batches, repeated record
IDs and unequal minibatch tails retain their original multiplicities and
denominators. The same identities apply before any updates in forward-only
mode. Concrete examples use four-state complex Fourier features, a nonzero
head, repeated observations, and an unseen scale of one half.

Lean: `ConvexDepth/ContextScale/{Basic,Batch}`. A native experiment must freeze
the scale/context bank and training-only selection before reading held-out
metrics, evaluate actual scaled complex products, bind all source tensors
and original records, preserve observed logits exactly, retain the full head
budget, and report all original success criteria alongside raw accuracy and
ordinary CE. CM42's reviewed source, model artifacts and results remain frozen.

## CM44. Realize declared decoder confidence with a complete bounded complex head

The reviewed CM43 comparison retains all 35 primary and auxiliary successes
in full-batch, minibatch and forward-only modes at seed 1. Fibonacci raw
accuracy is 4574/4704 and ordinary validation CE is about 0.16604. A separate
confidence problem remains in the inherited categorical phase heads: several
arithmetic tasks have perfect raw accuracy but CE around 3.59 or 4.27. Their
native predictor puts logit 1 on the declared raw label and zero elsewhere.
For 97 or 193 classes these are exactly the expected correct-margin losses.
Increasing confidence must be represented by an actual charged head, rather
than an unaccounted probability adjustment or a changed success criterion.

Let `state : X -> I` be a fixed declared input-state map, `decoder : I -> C`
the frozen raw-label decoder, and `U` a unitary complex matrix on the finite
source-state type `I`. Neither map needs to be injective. Pull back the actual
realified complex features along `state`. Every coordinate still has absolute
value at most one, even when query states repeat. A complete real/imaginary
coefficient tensor with per-class absolute-sum budget `R` therefore gives
actual logits bounded by `R`. The matrix over original query inputs need not
be unitary. Flatness is used for the bound, not for identifying any raw target.

Use CFt5's actual conjugate initializer with declared state labels `decoder`
and margin `a`. Unitary inversion gives, for every input and raw class,

```
sum_k W_a(k,c) phi(k,x) = if c = decoder(state(x)) then a else 0.
```

Both real and imaginary coefficients are stored and charged. CFt5 bounds
their per-class absolute sum by `2*abs(a)*card(I)`. In particular, every
`0 <= a <= 24` fits the common native head budget `64*card(I)`, including
both margin 1 and margin 24. This is an explicit output-head budget, separate
from an inherited latent-moment spectral bound. A positive margin gives the
declared raw label a strict advantage over every other class. It does not
establish that the declared label is the unknown target. Original pooled
task probabilities still require numerical evaluation after changing confidence.

Write `L_C(a) = log(1+(card(C)-1)*exp(-a))`. For every raw target `y`, ordinary
full-vocabulary CE of the actual head is exactly

```
L_C(a) + if decoder(state(x)) = y then 0 else a.
```

For a nonempty finite batch of original record IDs, its normalized CE is
therefore `L_C(a) + a*errorRate`, where the error rate uses each record's
actual raw target. Repeated inputs retain their separate record IDs and
multiplicities, and each minibatch tail retains its own denominator. A batch
that exactly matches the declared decoder has error rate zero. For a nonempty
vocabulary, increasing `a` cannot increase `L_C(a)`; wrong labels still incur
the separate linear penalty. Thus greater confidence is beneficial on a
correct batch without being an unconditional held-out improvement guarantee.

With the state map and complex code fixed, ordinary normalized CE remains
convex in the entire real/imaginary coefficient tensor on the same budget.
This does not assert joint convexity in learned discrete state assignments
and a decoder. Any confidence selection must use original training data and
be fixed before held-out scoring. The equality also applies to a zero-update
forward-only initializer and to the initial state of ordinary full/minibatch
CE training. Unlike CM43's observed scale-one transformation, raising the
declared margin intentionally changes observed probabilities and CE.

A nonzero four-state complex Fourier witness has a many-to-one Boolean
decoder and six inputs mapped to two source states. Three original record IDs
repeat input 0 twice and use input 4 once. Against a constant-false target,
only the final record is wrong: the full-batch CE at margin 24 is
`log(1+exp(-24))+8`, whereas its one-record final minibatch has
`log(1+exp(-24))+24`. Targets matching the declared decoder instead give the
correct-margin loss on the same repeated inputs. These examples make the
normalization and wrong-label penalty explicit.

Lean: `ConvexDepth/DecoderConfidence/{Basic,Batch}`. A native implementation
must bind the declared source maps and decoder to the saved training result,
materialize and validate the complete conjugate coefficient tensor, charge
its budget and work, and evaluate actual complex products. An original-data
comparison must preserve every task, initializer policy, training mode and
baseline, report raw CE alongside the original task score, and retain any
regressions. The reviewed CM42/CM43 experiments remain frozen.

## CM45. Learn a convex mixture of fixed context maps with the head frozen

The reviewed CM44 comparison preserves all 35 original primary and auxiliary
successes in every mode and both policies. Its stronger complex confidence
heads reduce CE on 13 tasks whose raw labels were already correct. Fibonacci
still has 130 incorrect raw predictions among 4704 validation records, despite
4701 correct field answers. A positive confidence change alone cannot alter
these raw rankings. A different context rule is required to change the raw choice.

Fix a finite bank of actual feature maps `phi_j(k,x)` and the complete complex
head tensor `W(k,c)`. Each bank element may use a different training-only
neighborhood, metric, distance weight or fixed confidence scale. For mixture
coefficients `a_j >= 0` with `sum_j a_j = 1`, define the actual mixed feature
and logit by

```
phi_a(k,x) = sum_j a_j phi_j(k,x)
z_a(x,c)   = sum_k W(k,c) phi_a(k,x).
```

Expanding the finite sums gives `z_a(x,c) = sum_j a_j z_j(x,c)`, where every
`z_j` uses the same complete real/imaginary tensor. A uniform coordinate bound
`abs(phi_j(k,x)) <= M` is retained by the mixture. The original per-class
absolute-sum coefficient budget `R` therefore bounds the actual mixed logits
by `R*M`. In particular, CM42 stochastic context maps over a unitary source
code, followed by CM43 fixed scales in `[0,1]`, retain the bound `M=1`. Repeated
source states and nonunitary query matrices are allowed. The spectrum supplies
this bound; it does not choose the bank or identify a missing target.

The simplex is convex. With `W` and the bank fixed, the actual logits are
linear in all mixture coefficients, so ordinary full-vocabulary raw CE is
convex in the mixture. Jensen compares the mixed CE with the weighted CE of
the bank logits, all evaluated against the same query target. Let `p_a(x,c)`
be the ordinary softmax of the mixed actual logits and define

```
g_j(a,x,y) = sum_c p_a(x,c) z_j(x,c) - z_j(x,y).
```

The exact first-order inequality is

```
CE(z_a(x),y) + sum_j (b_j-a_j) g_j(a,x,y) <= CE(z_b(x),y).
```

It follows from XE1 and the actual finite-product identity, with every class
in the denominator. On a finite batch of original record IDs, the mean CE is
convex and the mean slopes give the corresponding first-order inequality.
Repeated inputs retain separate IDs and weights; a short final minibatch uses
its own cardinality. These statements apply to a fixed training-only
diagnostic feature bank as well as to a prediction bank.

The same complete-head bound controls a mixture change:
`abs(z_b(x,c)-z_a(x,c)) <= R*M*sum_j abs(b_j-a_j)`. Applying XE2 gives an
upper quadratic model with the same explicit linear term and remainder
`(R*M*sum_j abs(b_j-a_j))^2/2`. Thus the spectral feature bound also controls
how far ordinary CE can exceed its linear model; no task-specific answer
formula or uncharged output scaling is used in this estimate.

If every bank element equals the original source feature row on each observed
input, every mass-one mixture has exactly the original observed CE for every
complete tensor. The original complete-head Frechet derivatives also agree.
Consequently, the unchanged observed objective cannot select between mixtures.
A diagnostic such as excluding the query's own observed source state must be
declared separately and use only original training records. This is not a
leave-one-out refit of latent discovery, and it supplies no unseen-correctness
certificate. Convexity in mixture weights with the head frozen is separate
from convexity in the head with a mixture frozen; simultaneous multiplication
of two learned factors is not asserted to be convex.

A nonzero four-state complex Fourier witness uses two raw labels, 0 and 2,
which share a declared readout. Both bank maps preserve observed inputs 0 and
1. At an unseen input, one map uses source state 0 and the other uses state 1.
Mixture weights three quarters and one quarter yield raw logits 18 and 6 at
margin 24; exchanging the weights reverses these logits. The raw-label margin
and the difference between their two target losses are 12. Both mixtures keep
the same original repeated-record CE and one-record tail CE. Matching original
targets have loss `log(1+3*exp(-24))`. Against a constant-zero target, only the
last record is wrong: the full batch adds 8 to that loss, whereas its one-record
tail adds 24. Observed fit and the feature bound therefore do not choose which
unseen raw label is correct.

Lean: `ConvexDepth/ContextMixture/{Basic,Loss,Batch,Witness}`. A native extension
must bind the fixed bank, original records and complete source tensor, account
for all mixture construction and optimization work, and evaluate actual complex
products. It must distinguish ordinary observed CE from the training diagnostic,
preserve full/minibatch/forward metadata, and freeze every learned mixture before
held-out evaluation. All 35 original cases, both policies, unchanged baselines
and any regressions remain required. No numerical improvement from CM45 is
claimed before that separate experiment.

## CM46. Fixed gates and anchored local convex context mixtures

The reviewed CM45 comparison retains 35/35 original successes in each mode
and both policies. Its shared context mixture reduces the remaining Fibonacci
raw errors from 130 to 109 full-batch, 103 minibatch and 100 forward-only.
Full-batch CE is lowest at 0.14011; forward-only raw accuracy is highest at
4604/4704. A shared mixture still applies the same relative context preferences
to every input. The following extension adds local capacity without claiming
that local optimization will improve unseen answers.

Fix the actual bank `phi_j`, complete head `W`, an anchor mixture `b` in the
simplex, a radius `r` in `[0,1]`, and nonnegative input gates `q_g(x)` with
`sum_g q_g(x)=1`. Learn a simplex row `a_g` for each group. Gates may overlap;
the gate map is fixed, and is not a simultaneously trained factor. Define

```
v_j(a,x) = (1-r)*b_j + r*sum_g q_g(x)*a_gj
phi_a(k,x) = sum_j v_j(a,x)*phi_j(k,x)
z_a(x,c) = sum_k W(k,c)*phi_a(k,x).
```

The product of local simplexes is convex. Each `v(a,x)` is in the simplex,
and the actual logits equal
`(1-r)*z_b(x,c) + r*sum_g q_g(x)*z_(a_g)(x,c)`. They are affine in all local
coefficients jointly, with `b,r,q,phi,W` fixed. Thus ordinary full-vocabulary
raw CE is convex in those coefficients. No joint convexity in the head and
local mixture, or in learned gates and local rows, is claimed.

The bound `abs(phi_j(k,x)) <= M` survives every anchored gated mixture.
Every original real and imaginary coefficient remains charged to the same
head budget `R`, so `abs(z_a(x,c)) <= R*M`. CM42 stochastic contexts and CM43
fixed scales in `[0,1]` over a unitary complex source give `M=1`. Flatness
supplies this bound; the repeated or mixed query rows need not be unitary.

Let `g_j(v,x,y)` be CM45's full-softmax slope at the effective weights `v`.
The local slope is `r*q_g(x)*g_j(v(a,x),x,y)`. Its pairing with `a'-a` is
exactly CM45's slope pairing with `v(a',x)-v(a,x)`, since the anchor cancels.
This proves the ordinary first-order supporting inequality in every local
coefficient. Averaging over the actual finite set of original record IDs
gives the corresponding batch inequality and convexity, preserving repeated
inputs and each short tail's own denominator.

For nonnegative radius and gates, the actual change satisfies

```
abs(z_a'(x,c)-z_a(x,c)) <=
  r*R*M*sum_g q_g(x)*sum_j abs(a'_gj-a_gj).
```

XE2 bounds CE by its explicit local linear model plus one half the square
of this right-hand side. Gate normalization and `r<=1` are unnecessary for
this change estimate, whereas they are needed for the unchanged feature
bound. If every bank preserves an original observed feature row, the three
mass-one identities alone preserve that row for any radius. Every original
batch loss and its complete-head Frechet derivative are then unchanged.
This observed objective cannot identify useful local rows. A training-only
source-excluded diagnostic is a separate rule, not a refit of latent discovery
and not a guarantee about missing labels.

An actual nonzero four-state Fourier witness reuses CM45's two maps. Both
maps preserve inputs 0 and 1, and each map uses the same source state at
unseen inputs 2 and 3. Therefore every shared mixture gives identical logits
at those two inputs, making opposite strict raw margins impossible. Fixed
one-hot input gates put inputs 2 and 3 in different groups. With the uniform
anchor, radius one half, and local rows `(3/4,1/4)` and `(1/4,3/4)`, the actual
raw logits on labels `(0,2)` are `(15,9)` at input 2 and `(9,15)` at input 3.
Both strict margins and the corresponding wrong-target CE differences are
six. Labels 0 and 2 still share the same modulo-two readout. The model keeps
the original repeated-record fitted CE `log(1+3*exp(-24))`; a constant-zero
target adds eight on the three-record batch and 24 on its one-record tail.
The input partition is additional model structure. This capacity witness
does not identify the correct unseen label or show that a diagnostic learns it.

Lean: `ConvexDepth/GatedContext/{Basic,Bounds,Loss,Batch,Witness,Capacity}`.
A native implementation must fix and bind gates, anchor, radius, bank and
complete source head before local optimization. Every local row must stay
in its simplex. It must retain original records and mode metadata, including
zero derivative calls for forward-only, and charge all added state and work.
All 35 original cases, both policies, unchanged baselines, raw CE and every
regression remain required. No numerical improvement from CM46 is claimed.

## CM47. Convex raw-margin constraints around a fixed reference

The complete five-gate comparison lowers ordinary held-out CE while losing
some raw-label decisions in minibatch and forward-only modes. Convex CE does
not itself preserve argmax decisions. The following constraint addresses
stability relative to a fixed reference. It does not identify the correct
unseen target or imply an improvement in held-out CE.

Fix reference scores `z_ref(x,c)`, a chosen raw label `l(x)` maximizing those
scores, and a retention fraction `k` in `(0,1]`. For every input in the declared
query domain and every raw competitor class, require

```
z_a(x,l(x)) - z_a(x,c) >= k * (z_ref(x,l(x)) - z_ref(x,c)).
```

These are closed affine halfspaces in the actual score vector. For CM46, fix
the bank, complete head, gate map, anchor and radius. Its actual logits are
affine in all local simplex coefficients. Intersecting those simplexes with
the inverse images of the halfspaces therefore gives a convex coefficient
domain, and ordinary full-vocabulary CE remains convex on it. The constraints
do not change the loss or remove any raw class. The reference coefficient
tensor itself is feasible whenever it is in the original simplexes: its
selected margins are nonnegative and `k<=1`. Convexity of the domain holds for
any fixed real `k`; feasibility uses `k<=1`, weak comparison preservation uses
`k>=0`, and strict comparison preservation uses `k>0`.

A separately fixed diagnostic bank can define the ordinary CE objective while
the constraints continue to use actual inference logits. CM46 proves convexity
for that fixed objective bank, and restriction to the same convex domain
preserves it. This covers source-excluded diagnostic rows without confusing
them with the actual constrained prediction rows. No claim of a leave-one-out
refit or unknown-target correctness follows from that diagnostic.

Every nonnegative reference margin stays nonnegative, and every strictly
positive reference margin stays strictly positive when `k>0`. Define the
first maximizer as a class whose score is at least every class score and is
strictly above the score of every lower-index class. This criterion permits
higher-index ties. The two comparison results preserve this criterion, and
two classes satisfying it must coincide. Thus the deterministic first-argmax
prediction is identical to the reference prediction on every constrained
query. Correct-record counts consequently agree for arbitrary target labels
and any finite batch of original record IDs, including repeated inputs and
short tails. This is prediction stability, not target correctness: a wrong
reference prediction remains wrong.

Use CM46's nonzero four-state Fourier source, complete head, two context maps,
one-hot gates and radius one half. Its reference local rows give logits
`(15,9)` and `(9,15)` on raw labels `(0,2)` at the two unseen queries. Move each
local row halfway toward the uniform anchor. The actual unseen logits become
`(13.5,10.5)` and `(10.5,13.5)`; observed rows remain margin24. All raw-class
halfspaces hold at retention one half, and the raw decisions are unchanged.
If the unseen target at query2 is class2, the retained decision is still the
wrong class0. Its wrong-target CE is three above its predicted-label CE. The
candidate stays within the same feature/head bounds and retains the original
repeated-record batch and tail derivative identities by CM46.

At retention zero, uniform local rows instead give tied unseen logits
`(12,12)`. Every weak margin constraint still holds, but the first maximizer
at query3 changes from raw class2 to raw class0. Strictly positive retention
is therefore necessary for this deterministic stability statement. The
nonzero actual complex product matters: this is not a zero-head example.

Lean: `ConvexDepth/MarginGuard/{Basic,Stability,Witness,Capacity}`. Every
conditional theorem has a concrete inhabited Fourier instance. A native
extension must fix the reference and fraction before evaluation, compute
constraints from its actual complete raw logits without unknown target
labels, retain deterministic tie behavior and charge every construction and
constraint check. A frozen bounded context map can be an additional fixed
bank element under CM45/CM46; it is not a jointly learned factor. Numerical
optimization, finite-precision constraint enforcement and raw/CE tradeoffs
require a separate frozen native experiment. No numerical improvement is
claimed from CM47 alone.

## CM48. Identify covered raw labels using convex input regions

CM47 preserves a fixed reference's raw decisions, including its mistakes.
A repair that changes a missing raw answer needs additional identifying
structure. The following extension uses input geometry. It is independent
of spectral flatness and does not assert that arbitrary benchmark labels
satisfy its structural hypothesis.

Fix observed input points `p_i` in a real vector space and their raw labels
`y_i`. These can be the observations inside one already fixed latent block.
Let the unknown target be `f`, with `f(p_i)=y_i`. Assume each raw-class region
`{x | f(x)=c}` is convex. This is a realizability assumption about the target
on the input space, not a condition established by observed fit. In a blockwise
application it requires an appropriate convex-region extension within each
block's chosen input embedding; it is not asserted globally across blocks.

For each raw class, form the observable set `P_c={p_i | y_i=c}`. If a query
`x` lies in `convexHull(P_c)`, then `f(x)=c`: the target's convex class region
contains `P_c` and therefore its hull. Thus every convex-region target fitting
the observations agrees at that query. If hulls bearing two labels overlap,
the labels must be equal; overlap of differently labelled hulls refutes the
realizability assumption. Coverage and hull overlap concern observed geometry
only. Neither proves the realizability hypothesis on unknown labels.

A finite certificate consists of `w_i>=0`, `sum_i w_i=1`,
`sum_i w_i*p_i=x`, and `w_i!=0 => y_i=c`. These conditions use observed
coordinates and labels and imply the hull membership. The same weights can
construct CM42's actual feature row `phi_w(k,x)=sum_i w_i*phi(k,i)`. With
the complete head frozen, its actual logits are the weighted source logits.
If each positively weighted observed source has raw margin at least `m`
against a competitor, the query has that margin against the same competitor
for its now identified target. If each source's ordinary CE against its own
observed label is at most `epsilon`, the query's ordinary full-vocabulary CE
is at most `epsilon`. Unlike CM42's unrestricted neighborhood transfer, the
query target here follows from geometry and convex-region realizability.

Every source coordinate bound `abs(phi(k,i))<=M` survives these stochastic
weights, and the complete coefficient budget `R` still gives
`abs(z(x,c))<=R*M`. A unitary complex source supplies `M=1`; the interpolated
query rows need not themselves be unitary. The weights are fixed while the
complete real/imaginary head is trained, so ordinary CE stays convex in that
head. Dirac rows at observed inputs preserve every original batch objective
and full-head derivative, including repeated records and a short tail.
There is no jointly learned interpolation/head product or extra loss.

The nonzero witness uses observed coordinates `(-3,-1,1,3)` with raw labels
`(0,0,2,2)`, four-class vocabulary, the actual normalized four-state Fourier
code and its complete conjugate margin-four head. The unseen inputs `-2`
and `2` each average two same-label source points. They receive different
raw answers with exact margin four and ordinary CE `log(1+3*exp(-4))`, although
both raw answers share the modulo-two readout. Observed rows remain Dirac.
The complete head has budget 32, and the same source feature bound applies
to all six inputs. A five-record example repeats the first observation and
checks both the full observed objective and a one-record tail derivative.

Coverage is essential to this argument. The two threshold targets which
return class0 at `x<=0` and at `x<=1/2`, respectively, and class2 otherwise,
both have convex raw-class regions and fit all four observations. They
disagree at `x=1/4`, which is in none of the observed class hulls. Every fixed
prediction there is wrong for at least one compatible target, while every
observed ordinary CE objective is identical for the two targets. This is an
explicit ambiguity example; it does not claim that every uncovered query
in every dataset is ambiguous.

Lean: `ConvexDepth/RawRegions/{Basic,Head,Witness}`. A native proposal must
construct and verify hull certificates from training observations only, bind
them to the exact source block and input coordinates, preserve original
observed rows, and use an explicit fallback when there is no unique covered
class. It must distinguish a conditional geometric certificate from an
unconditional correctness claim, retain ordinary raw CE and all original
records/modes, and charge hull construction and complete complex products.
Numerical inclusion tolerances cannot silently become exact certificates.
A separately frozen complete original comparison remains required; CM48
alone claims no benchmark improvement or discovery of convex target regions.

## CM49. Identify uncovered labels by convex exclusion

CM48 identifies labels inside observed same-class hulls. Outside those hulls,
additional observations can still rule out candidate labels. Fix the same
observed points and raw labels, an unknown target fitting them, and convex
regions for every raw class. For a query `x` and candidate `c`, suppose that
`convexHull(P_c ∪ {x})` intersects `convexHull(P_d)` for some `d != c`.
If `target(x)=c`, the entire first hull lies in class `c`; the second lies in
class `d`. Their intersection is a contradiction. Thus this observable
geometric certificate excludes `c` without supplying the missing answer.
An observed point of class `d` inside the augmented candidate hull suffices.

Now fix an admissible candidate pool containing the unknown query label. If
such certificates exclude every candidate except `a`, the query label is `a`.
A correct many-to-one decoder fiber is one possible pool. Using only locally
observed labels requires a separate closed-world assumption. Neither decoder
correctness nor pool closure follows from flatness or observed fit. Without
a pool restriction, a previously unobserved raw class can remain possible.

For observations `(-3,-1,1,3)` with labels `(0,0,2,2)`, take the pool `{0,2}`.
The query `-4` lies outside every observed class hull. Assigning it class2
would make the segment from `-4` to the observed class2 point `1` contain the
observed class0 point `-3`: weights `(4/5,1/5)` certify this. Hence its label is
class0. The symmetric certificate identifies class2 at `4`. Both conclusions
hold for every compatible convex-region target with the stated candidate pool.

The pool premise matters. A compatible target can introduce class1 on the
half-line `x < -7/2`, retain class0 on `[-7/2,0]` and class2 on `(0,infinity)`.
All class regions are convex and all observations unchanged, but its label at
`-4` differs. Conversely, even with the closed pool, the CM48 targets cutting
at `0` and `1/2` disagree at `1/4`. Candidate closure alone does not identify
every uncovered point; the exclusion certificates do substantive work.

After identification, use a fixed nonnegative unit-mass mixture of observed
source features bearing the surviving label. These prediction weights need
not reproduce the query input coordinate: input geometry has already supplied
a separate identifying certificate. The actual complete head then transfers
source margins and ordinary full-vocabulary CE by CM42. The same flat-source
coordinate bound and complete head budget apply. This does not use signed
affine interpolation weights or assume that observed collinearity determines
unknown class regions. Dirac rows at observed inputs preserve the original
objective and complete-head derivatives, including repeated records and tails.

A native extension must verify the candidate pool, augmented-hull certificates,
source identities, query role and fallback decisions from the frozen inputs;
where pool membership is a structural assumption rather than a proved fact,
that distinction must remain explicit. It must preserve observed rows, charge
all candidate tests and actual complex products, and use ordinary CE. A
separate frozen comparison must retain all earlier results, including failures.
No new benchmark accuracy follows from CM49 alone.

## CM50. Finite certificates for crossings between observed hulls

CM49's general exclusion theorem allows any intersection of the augmented
candidate hull with a differently labelled observed hull. Its first native
implementation uses the sufficient case in which the intersection contains
an original observed point of the other class. This can miss edge crossings.
A stronger exact certificate needs two positive finite mixtures, not a new
assumption about the unknown target or its spectrum.

Let the observed points be `p_i` with labels `y_i`, the query be `x`, and the
candidate and conflicting labels be `c` and `d`, with `d != c`. Take observed
weights `v_i >= 0` of total mass one supported only on label `d`. Take a query
weight `a > 0` and observed weights `w_i >= 0`, supported only on label `c`,
with `a + sum_i w_i = 1`. If the exact input-coordinate identity

`sum_i v_i*p_i = a*x + sum_i w_i*p_i`

holds, both sides give the same point in the two hulls. Thus CM49 excludes
`c`. These are coordinate weights used to prove an intersection. They remain
separate from the positive source-feature weights used for prediction after
a unique surviving label has been identified. Native implementations must
verify both masses, both label supports, source identities, the query role
and the exact common coordinate; floating-point proximity is insufficient.

For a planar witness, use four observations `(-1,-1), (-1,1), (1,-1), (1,1)`
with raw labels `(0,0,2,2)`, and the pool `{0,2}`. The query `(-2,0)` lies
outside every observed class hull. Hypothetically giving it label2 makes
its augmented hull cross the observed label0 segment. The common point
`(-1,-1/3)` is `2/3*(-1,-1) + 1/3*(-1,1)` and also
`2/3*(-2,0) + 1/3*(1,-1)`. Neither observed label0 point lies inside the
augmented label2 triangle: the triangle at horizontal coordinate `-1` has
vertical range `[-1/3,1/3]`. Consequently the earlier observed-point test
misses this valid exclusion. The symmetric query `(2,0)` excludes label0
by the same construction on the other side.

The nonconstant target returning0 when the first coordinate is nonpositive
and2 otherwise fits every observation and has convex class regions. The
exclusions identify the two exterior queries for every compatible convex
raw target with the stated pool, not merely for this particular target.
The actual normalized four-state Fourier code and complete margin-four
head can use the same positive feature mixtures as CM48/CM49: average the
two observed sources of the identified class. Their full-vocabulary CE is
`log(1+3*exp(-4))`, source coordinate bound is one and complete-head budget
is32. Original Dirac rows still preserve repeated-record objectives and
tail derivatives. Pool closure and target convexity remain explicit; the
stronger geometric search does not establish either assumption or guarantee
that every remaining unknown answer is identifiable.

## CM51. Signed context features with bounded amplification

The complete CM50 comparison covers 91 additional Fibonacci validation inputs
in every mode but changes no raw prediction. All of those inputs were already
correct. This motivates testing a different fixed feature map, rather than
inferring further accuracy from increased geometric coverage. No new numerical
gain is asserted by the following extension.

Let `a(x,i)` be arbitrary fixed real weights on the original source states,
including negative weights. Define their observable absolute mass and scale by

```
L(x) = sum_i abs(a(x,i))
k(x) = max(1, L(x))
w(x,i) = a(x,i) / k(x).
```

Use CM42's actual feature construction `phi_w(j,x)=sum_i w(x,i)*phi(j,i)`
and the same complete real/imaginary head `W`. Direct finite expansion gives

```
z_W(x,c) = sum_i a(x,i)*z_W_source(i,c) / k(x).
```

The absolute mass of `w` is `L/k <= 1`. If each source coordinate has absolute
value at most `M>=0`, the unnormalized coordinate is bounded by `L*M` and the
normalized coordinate by `M`. Thus the original absolute-sum head budget `R`
still gives `abs(z_W(x,c))<=R*M`. A unitary complex source supplies `M=1`.
The zero row is included because `k>=1`; the query matrix need not be unitary.
Flatness remains a source-coordinate bound, not a missing-label condition.

CM42's actual-product, complete-head convex-CE and observed-batch identities
do not require positive weights. They therefore apply to these signed maps.
Its positive-weight Jensen and same-label source-margin theorems are not used.
If an original observed row is Dirac, its absolute mass and normalizer are one,
so normalization retains that row exactly. Every original finite batch objective
and complete-head derivative remains unchanged, including repeated record IDs
and a short final minibatch. This includes the state before any forward-only
updates. The map is fixed while the complete head is optimized; a jointly
learned map/head product is outside this convexity claim.

To obtain a correctness condition, separately supply reference source scores
`g(i,c)` and reference query scores `q(c)`. Suppose the actual source scores
differ from `g` by at most `epsilon`, and the reference has query residual

```
abs(sum_i a(x,i)*g(i,c) - q(c)) <= delta
```

for each relevant raw class. The triangle inequality and the actual product
then give

```
abs(z_W(x,c) - q(c)/k(x)) <= (L(x)*epsilon + delta)/k(x).
```

If `q(y)>q(c)+2*(L*epsilon+delta)`, the actual complete head strictly prefers
raw class `y` to `c`. The source errors and reference residual are explicit
premises. Observed class fit alone does not bound a reference query residual;
choosing observable signed weights does not establish the target reference.
Normalization controls magnitude but does not supply that missing information.

For a nonzero complex witness, observe points `(0,0),(0,1),(1,0),(2,0)` with
raw labels `(0,2,0,0)` in a four-class vocabulary. The unseen query `(1,1)` is
the signed affine combination with weights `(-1,1,1,0)`, of total mass one and
absolute mass three. The declared reference has scores `4*(1-v)` for class0,
`4*v` for class2 and zero for the other classes at coordinates `(u,v)`.
It agrees with the complete conjugate Fourier margin-four head at every
source, and its query residual is zero. The normalized signed features give
actual query score `4/3` for class2 and zero for every other class, with ordinary
CE `log(1+3*exp(-4/3))`. The same complete coefficient budget is32.

Two target tables agree on every observed raw label but give class2 and class0,
respectively, at the fifth query. Both raw answers have the same reduction
modulo2 readout. Thus neither the observed data nor the declared readout
establishes the reference at the query. The witness instantiates the conditional
margin theorem; it does not infer arbitrary missing answers from a signed map.
Its five original record IDs repeat the first source and end in a one-record
tail, preserving the actual observed objective and full-head derivatives.

Lean: `ConvexDepth/SignedContext/{Basic,Witness}`. A native proposal may derive
signed weights from a fixed input kernel or descriptor reconstruction. It must
declare that additional input structure, freeze any training-only selection
before scoring, normalize the actual source-feature weights, retain every
complete complex product, and charge construction, inversion, storage and all
diagnostic work. A source-excluded diagnostic is separate from original training
CE and is not a refit of latent discovery. No target residual certificate or
numerical gain is supplied by this proof. A separately frozen original comparison
must retain every task, both policies, all modes, earlier references and costs.

## CM52. Shared input programs and an explicit copy decoder

The additional sequence comparison at `7a2e2ba4`, reviewed at `688f5592`,
exposes a representation problem. Its causal-pair component fits all256 MQAR
training targets but gets only1/128 test targets correct. Of54 test queries
with a recorded selected-feature state,53 are wrong. Lookup reaches its
candidate cap; Dyck and blocks have no consistent candidate. This motivates
query-relative and shared input features. Flatness alone cannot choose them.

Fix a finite bank of input-only programs `p_j : X -> C`, a candidate order,
and a unitary vocabulary code `U`. A program returns a symbol in the original
output vocabulary. Use the complete real/imaginary conjugate head
`H = U* T`, where `T[s,c] = margin * 1[c=s]`. The identity assignment across
all vocabulary states is an explicit copy prior, including states whose raw
classes were never observed as labels. It is not inferred from unitarity.
The program features are the complete row `U[p_j(x), :]`; they do not read
an output label or a task-specific answer function at prediction time.

Learn a simplex mixture `a` of these fixed feature maps while keeping `H`
fixed. Expanding the actual complex products gives

```
z_a(x,c) = sum_j a_j * margin * 1[c = p_j(x)].
```

CM45 proves convex ordinary full-vocabulary CE in all mixture coefficients.
The code bounds each mixed real feature coordinate by1, and CFt5 charges the
entire head to column budget `2*abs(margin)*|C|`. Neither feature construction
nor the shared head is an independently learned factor. If an input renaming
commutes with every program and a vocabulary permutation, the actual mixed
logits commute with that permutation for every mixture. This follows from
the complete product identity; no permutation property of the Fourier entries
themselves is assumed.

The executable learner filters the candidate order by all observed
input/label pairs. `retained` keeps exactly the training-compatible programs.
At a query, `consensus` returns their common symbol when the retained list is
nonempty and unanimous; otherwise it returns `none`. `predict` composes these
two procedures. They receive no unknown target-program index and no held-out
label. Theorems state the additional structural premise explicitly: the true
target is some enumerated program on all inputs, and observed labels are
correct. That program survives, so every accepted consensus is the true raw
answer. This does not assert acceptance on every query or realizability for
every stress task. A missing answer cannot be recovered outside the declared
program family merely because the code is flat.

Observed inconsistencies impose zero-weight constraints on the corresponding
programs. These linear constraints intersect the simplex in a convex face.
Any retained program supplies a feasible forward vertex. Every feasible
mixture has exactly the prescribed finite-margin logits on the observations;
ordinary training CE therefore cannot distinguish surviving programs. At an
accepted query, every compatible mixture has the same exact target logits,
and ordinary CE is `log(1+(|C|-1)*exp(-margin))`. Positive margin gives the
agreed class a strict advantage. The transfer uses the copy prior, program
realizability and checked consensus; none of these is supplied by the spectral
bound. No convergence rate, optimizer success or native accuracy is claimed.

The executable witness supplies two programs on pairs of four-class symbols:
copy the first coordinate or copy the last. The sole observation `(0,1)->1`
eliminates the first program. At `(0,3)`, the resulting compatible vertex and
complete four-state Fourier head place margin24 on raw class3, although3
never occurred as a training label. Its exact CE is `log(1+3*exp(-24))`.
With observation `(0,0)->0` instead, both programs survive and disagree at
the same query. The learner returns `none`; their two feasible vertices give
opposite actual complex margins using the same flat code and shared head.

Lean: `ConvexDepth/SharedPrograms/{Basic,Learner,Transfer}`. A native repair
must specify a generic causal program library, test its training consistency,
retain any failed or ambiguous search, compute the actual complete complex
products, and charge its search, storage and scoring work. Shared copying is
a proposed additional prior for retrieval tasks, not a proof that generic
prefix summaries solve Dyck/blocks or that numerical prediction reaches99%.
The failed sequence baseline and all original35 comparisons remain frozen.

## CM53. Learn shared latent decoders without bilinear factors

The reviewed shared-copy experiment at `c44632bb`, with findings `f868e07f`,
reaches100% raw-token and exact-sequence accuracy on MQAR and two-hop lookup
in all three modes and every short/long split. Dyck and alternating blocks
retain all six no-compatible-copy-program refusals. Their raw labels need not
be symbols that appear in an input. CM44 already realizes a declared many-to-one
decoder with a complete complex head, but does not learn it. CM52 fixes the
identity decoder. CM15's convex decoder hull and CM17–CM18's structured table
identification do not supply the following generic latent-program learner.

Fix input-only programs `p_j : X -> S`, a candidate order, observed pairs
`(x_i,y_i)`, and a unitary code `U` on the finite latent state type `S`.
The raw vocabulary `C` is separate from `S`. For each program, look up the first
observed raw label at a queried latent state. Retain every program for which
this dictionary matches all original labels. This is equivalent to requiring
equal observed raw labels whenever that program gives equal latent states.
No target decoder, target program index, task identity or held-out label enters
the executable filter. Its learned decoder may be many-to-one.

At a new input, require a nonempty retained bank and a known, unanimous raw
dictionary output from every retained program. Unknown state coverage or any
disagreement returns `none`. If the unknown target has the form `d(p_j(x))`
for some enumerated program and some unknown raw decoder, and observations are
correct, that program survives. Every accepted prediction is then the raw
target. This is conditional on structural realizability, not on flatness.
The returned label must occur among the training labels. Thus this learner
can identify new inputs with covered latent states; it does not invent a new
raw class without CM52's copy prior or another identifying assumption.

Learning independent program weights and decoder factors would multiply
learned quantities. Instead use one joint coefficient tensor `gamma[j,s,c]`
and shared masses `w[j]`, with

```
w >= 0,       sum_j w[j] = 1,
gamma >= 0,   sum_c gamma[j,s,c] = w[j] for every j,s,
w[j] = 0 for programs outside the declared order,
gamma[j,p_j(x_i),c] = w[j] * 1[c=y_i] for every original sample and raw class.
```

These are affine equalities and nonnegative coordinate inequalities, so the
whole domain is convex. Two conflicting labels at one latent state force that
program's entire mass to zero. Conversely, every retained candidate supplies a
feasible vertex: use its learned labels on observed states and an explicit
arbitrary default on other states. Selecting the first retained candidate is
an executable initialization; target realizability proves that selection exists,
rather than supplying the selected index to the learner. Convex mixtures of
consistent declared decoders are also feasible points. The joint domain is not
restricted to independently parameterized factors or deterministic completions.

For each program block set `T_j[s,c] = margin * gamma[j,s,c]` and use the actual
complete complex head `H_j = U* T_j`. The input features are the full code row
`U[p_j(x),:]` in every block. Expanding both real and imaginary coordinates gives

```
z_q(x,c) = margin * sum_j gamma[j,p_j(x),c].
```

This is linear in all joint coefficients. The ordinary full-vocabulary
log-sum-exp cross-entropy is convex on the joint domain. Since every entry of
`gamma[j,s,:]` lies between0 and `w[j]`, CM32's arbitrary-table bound gives each
block column budget `2*abs(margin)*w[j]*card(S)`. Summing the actual complete
columns yields `2*abs(margin)*card(S)`. Every source feature coordinate remains
bounded by1. The number of stored coordinates still grows with the number of
programs, latent states and raw classes: the complete realified head has
`2*card(J)*card(S)*card(C)` coordinates, all charged in native work.
The empirical query-feature matrix may repeat states and need not be flat.

Every feasible joint tensor has the declared finite-margin logits on the
observations. At an accepted query, nonzero program masses belong to retained
programs; their observed dictionary entries force the same full raw logit vector
for every feasible joint tensor, including arbitrary unknown-state completions.
It places `margin` on the accepted class and0 on every other raw class. Combining
this with realizability identifies the target, yields a strict winner for
positive margin and ordinary CE `log(1+(card(C)-1)*exp(-margin))`. Further CE
optimization on these exactly fitted observations cannot select among the
remaining unobserved rows. No optimizer convergence rate or new native prefix
accuracy is asserted here.

The witness uses two coordinate programs on pairs of four-state symbols and
a two-class raw vocabulary. Three observations eliminate the first-coordinate
program and learn labels for latent states0,1,2, with0 and1 sharing a raw label.
A new input using state2 receives the observed raw label and its full Fourier
margin. State3 is uncovered: the executable learner refuses, while two feasible
completions produce opposite actual complex margins on that same query. Both
use the same unitary source code and fit every observation. A numerical default
alone therefore cannot be promoted to an identification certificate.

Lean: `ConvexDepth/LatentPrograms`. Native implementation must retain the
training-only boundary, full actual complex products, all raw classes, explicit
defaults and ambiguity, proper minibatch normalization, forward-only accounting,
and bounded search/storage/scoring. It must preserve the reviewed retrieval
passes and failed baselines while measuring any prefix-summary improvement.

## CM54. Accepted latent queries transfer through actual generation

The reviewed task327 Rust data adapter preserves the original copy/reverse/sort
repeat/unique cases, including their EOS targets and length32/64 probes. It
measures no new model accuracy. CM53 identifies an accepted query's actual
complete complex logits. CCa7 proves generation after globally identifying a
causal rule, but CM53 can refuse other inputs. Its query contract therefore
needs a trajectory-specific bridge before a native generative extension.

For a next-token rule `g`, an EOS class, a fuel bound and an initial history,
`queryPrefixes` lists the prefixes actually queried along that rule's rollout.
It includes the prefix whose next output is EOS and stops there; at zero fuel
it is empty. Equality `f(xs)=g(xs)` only on this finite list implies identical
complete generations, including EOS and termination. Induction follows the
actual emitted token into the next history. No agreement is required at an
off-trajectory prefix, and no reference answer is fed back to the model.

Now instantiate the prediction rule with a maximum over all raw classes of
CM53's complete complex logits. Positive margin and executable acceptance
force this maximum to be the accepted raw class for every feasible joint
tensor. Under the existing unknown-program/unknown-decoder realizability
premise and truthful observations, that class equals the target's next token.
If every prefix on the target's finite trajectory is accepted, the complex
model and target therefore generate exactly the same full sequence. Unknown
states and disagreeing retained programs still prevent acceptance; the theorem
does not assume that the original random training pools provide this coverage.

The end-to-end statement also constructs the tensor from the training-only
initializer. Feasibility and the complete column budget
`2 * abs(margin) * card(S)` are conclusions. CM53's convex joint domain,
ordinary full-vocabulary CE, bounded features and accepted-query CE formula
remain unchanged. The result adds no optimizer convergence guarantee, native
refinement certificate or unconditional unseen-label identification. Flatness
continues to supply bounds, not coverage or a choice of target.

The inhabited witness learns two latent parity states from the empty prefix
and a one-token prefix. Starting at the unobserved prefix `[true,true]`, its
actual complex head emits `false`, then reads `[true,true,false]` and emits
EOS=`true`. Both queried histories are absent from training; their latent
states have observed decoder entries. This checks nonconstant generation,
feedback and termination without replacing them by teacher-forced scoring.

Lean: `ConvexDepth/LatentPrograms/Generation.lean`, audited directly by
`scripts/LatentGenerationAxioms.lean`. Task328 must still freeze its native
input-only program bank, learn token roles/decoders exclusively from training,
score actual generated prefixes, retain all failures and previous controls,
and measure per-task99% on every original short/long split. This proof alone
does not establish any additional stress-test pass.

## CM55. Charge the complete head under affine parameter sharing

The task330 comparison does not improve Fibonacci raw accuracy. Sharing
decision geometry across latent blocks is a prospective modeling hypothesis.
CD8 already gives convex ordinary CE after a fixed affine parameter map, on
the preimage of the complete head budget. That exact domain need not have a
cheap projection. A sufficient weighted parameter domain must account for
every coefficient created by expansion, rather than count stored parameters.

Let finite indices `p`, `k` and `c` describe compact parameters, all complete
realified head coordinates (including imaginary slots), and raw classes.
Fix an arbitrary signed expansion `A[p,k,c]` and offset `B[k,c]`, and define

```
H(theta)[k,c] = B[k,c] + sum_p A[p,k,c] * theta[p].
sum_k abs(A[p,k,c]) <= w[p]                         for every p,c,
sum_k abs(B[k,c]) <= T                             for every c,
D(w,R) = {theta | sum_p w[p] * abs(theta[p]) <= R}.
```

For nonnegative weights the domain is convex by the absolute-value triangle
inequality. Expanding the actual complete column, applying the triangle
inequality, interchanging the two finite sums, and using the expansion costs
gives `sum_k abs(H(theta)[k,c]) <= T+R` for every `theta` in that domain.
No positivity of the coefficients or parameters is needed. The bound is
sufficient: cancellation may make the exact affine preimage larger. Zero
costs are allowed only when the stated expansion bound holds; no uncharged
nonzero coefficient is thereby permitted.

For any fixed features with `abs(phi[k,x]) <= M`, `M >= 0`, the actual logits
satisfy `abs(z(theta,x,c)) <= (T+R)*M`. Flat complex codes provide bounded
features; repeated empirical rows need not themselves have a flat spectrum.
Ordinary full-vocabulary log-sum-exp CE is convex in `theta` on `D(w,R)`:
compose CD8's complete-head result with the actual affine map, then restrict
to the proved sufficient domain. For any finite batch of original record IDs,
the same holds after division by its actual cardinality. Repeated inputs remain
distinct observations, and a short final batch uses its own size. Lean also
covers the empty batch through the existing zero-sum convention.

The witness has two shared parameters, two head coordinates and two classes.
Both parameters occur in every coordinate and class, with costs1 and2.
The fixed offset has radius1/2; the point `(1,1/2)` uses parameter budget2
and attains complete radius5/2 in both classes. Nonconstant bounded features,
three records including a repeated input, and a one-record final batch
instantiate all conclusions. This rules out a merely zero or identity example.

Lean: `ConvexDepth/ParameterSharing/{Basic,Witness}.lean`, audited by
`scripts/SharedGeometryAxioms.lean`. This result does not prove a numerical
projection implementation, optimizer convergence, discovery of the fixed
feature/decoder maps, or correctness of any unobserved raw label. Native work
must audit the real expansion, gradients, projection and complete cost bounds;
it must retain training-only selection and measure heldout accuracy separately.

## CM56. Identifying unseen raw classes with supplied class coordinates

CM17 and CM53 leave an unseen raw label ambiguous without additional structure.
Task338 adds an explicit identifying prior: a supplied injective coordinate map
`e : C → ℝ`, a fixed scalar input feature `f`, and an unknown affine decoder
`e(target(x)) = a*f(x) + b`. The coordinates and feature family are assumptions;
flatness does not recover them. In particular, ordinal/numeric label coordinates
would be declared semantic information, not a generic property of categorical
IDs. This extension does not claim that the frequency tasks satisfy the prior
before a native training-only feature-selection experiment establishes evidence.

Two observed pairs at distinct feature values determine `a` and `b`. A shared
positive margin `g` yields scaled parameters `θ = (g*a, g*b, g)` and scores

`score(c,x) = 2*e(c)*(θ₀*f(x)+θ₁) - θ₂*e(c)^2`.

The two observable anchor constraints are linear in these three coordinates.
Subtracting the anchor equations identifies the scaled slope and intercept;
realizability then gives, at every query,

`score(target(x),x) - score(c,x) = θ₂*(e(target(x))-e(c))^2`.

Injectivity and positive margin make every wrong raw class strictly worse,
including when the correct class never appeared among the observed labels.
The constructive `fitted` map uses only the two observed pairs, supplied class
coordinates and chosen margin. The general `identified_prediction` theorem
covers every feasible head under those anchor constraints; it does not assume
correct query logits or optimizer convergence. Renaming classes preserves the
fit and scores when the supplied coordinates are transported with the renaming.
Keeping arbitrary numeric IDs fixed would not preserve this semantic prior.

The actual complex realization uses the fixed unitary four-point Fourier code
and signed row `(f(x),1,0,0)`, divided by `max(1, |f(x)|+1)`. CM51 bounds every
realified feature by one. The complete eight-coordinate head for each class is
an affine image of all three shared parameters. Its source rows are exactly
`(2*e(c)*θ₀, 2*e(c)*θ₁ - θ₂*e(c)^2, 0, 0)`, so its real complex product is the
stated score divided by the same positive, input-only normalizer. Both real
and imaginary head entries participate; this is not an abstract score proxy.

For `|e(c)| ≤ 1`, the complete per-class expansion costs are bounded by `(4,4,2)`.
The sufficient budget `4*|θ₀|+4*|θ₁|+2*|θ₂| ≤ R` charges every coefficient.
Intersecting it with the two linear anchor equations and `θ₂ ≥ δ > 0` remains
convex. CM55 transfers ordinary full-vocabulary cross-entropy convexity to both
the weighted domain and this constrained domain. Minibatches preserve repeated
record identities and their actual denominator, including a short final batch.
The three-coordinate parameterization avoids multiplying independently learned
slope and margin variables. Feature/geometry discovery is outside this convex
stage, and no finite-precision projection or optimization theorem is asserted.

The nonconstant witness uses five raw labels with coordinates `c/4`, features
`(0,1/2,1)`, and target labels `(1,2,3)`. Training records repeat label1 and also
observe label2; the query requires the unseen raw label3. Parameters `(2,1,4)`
and `(1,1/2,2)` are distinct feasible points at radius32 and margin lower bound1.
A genuinely imaginary coefficient is nonzero, and actual complex logits favor
label3 over label4. The countertarget `(1,2,4)` agrees at both anchors and
changes the unseen answer; it cannot satisfy the same affine prior. This is
why supplied geometry plus realizability resolves a gap that flatness alone
cannot. The witness is a theorem about this prior, not evidence of general
unseen-label recovery or success on the pending frequency benchmarks.

Formalization: `ConvexDepth/ClassGeometry/{Basic,Head,Witness}.lean`; direct
standard-axiom audit: `scripts/UnseenClassGeometryAxioms.lean`. Keep generated INDEX
and run direct Lean/global-axiom/native forbidden checks before landing.

## CM57. Joint program masses and coordinate decoders

CM56 and the reviewed frequency experiment339 identify an affine quantity
decoder after selecting an input program. A more general convex stage can
retain a finite, fixed family of input-only programs and learn their masses
and scaled decoder coefficients together. This does not make discovery of the
program family convex. Neither target answers nor learned parameters may be
read when constructing a program's features.

For program p, let h[p,x] have five real coordinates. One intended instance is
quantity times a quantity indicator, that indicator, two structural-role
indicators, and a supplied identity-copy coordinate. The theorem allows any
fixed five-coordinate features. Parameters theta[p,0..4] are the scaled decoder
coefficients themselves, and gamma[p]=theta[p,5] is the nonnegative mass. The
actual component score is

`2*e(c)*sum_j(theta[p,j]*h[p,x,j]) - gamma[p]*e(c)^2`.

There is no product of independently learned mass and decoder. A putative
unscaled decoder beta appears only in a separate semantic realizability
hypothesis. The supplied injective class coordinates e are prior information,
as in CM56, rather than a consequence of spectral flatness.

The feasible set imposes homogeneous linear observed-anchor equations,
`theta[p,4]=gamma[p]` for the explicitly declared identity-copy prior,
nonnegative masses, a positive lower bound on their total, and complete
coefficient budgets. With local cost `(4,4,4,4,4,2)`, impose both a global
weighted absolute budget R and the perspective inequalities
`charge(theta[p]) <= B[p]*gamma[p]`. All these constraints are convex in the
actual learned coordinates. Since every cost is positive, zero mass forces
every coefficient of that program to zero. An incompatible program can
therefore remain in the bank with zero mass without contributing free logits.
The conditions do not assert that a given training pool makes the domain
nonempty, or that an optimizer reaches a desirable point.

The complete complex realization uses two independent four-point Fourier
blocks per program. Their source rows carry `(h0,h1,h2,h3)` and `(h4,1,0,0)`.
Normalize the entire signed row over all programs and both blocks by its
absolute mass or one, whichever is larger. This is one common positive,
input-only denominator N(x). Blockwise unitary feature bounds and CM51 give
absolute bound one for every resulting realified feature. This is not a claim
that repeated training rows or the assembled query Gram matrix are flat.
Expanding the two blocks and retaining every real and imaginary coefficient
gives exactly the sum of component scores divided by N(x). For |e(c)|<=1,
the column costs above charge all sixteen real coefficients per program and
raw class. CM55 then proves the complete budget and ordinary full-vocabulary
CE convexity, including batches of record IDs with the actual tail denominator.

Identification is conditional on observable feature-span coverage and the
semantic prior. Augment each program's observed feature rows with the copy
basis vector, whose coordinate value is explicitly supplied as one. Suppose
the query feature is a linear combination of these rows, an unknown beta
realizes their coordinate labels and the query target, and beta's copy
coefficient is one. Distributing the learned linear center over that same
combination identifies it as gamma times the target coordinate. These are
feature-space and target-realizability premises, not assumptions of correct
query logits. If this condition holds for every positive-mass program, zero
mass contributes nothing and the actual complete-head score gap is

`(sum_p gamma[p]) * (e(target)-e(c))^2 / N(x)`.

A positive total and injective e make every wrong raw class strictly worse,
including a raw class absent from the observations. Mere agreement with
training anchors does not establish query coverage or realizability. In
particular, an unconstrained structural-role direction can admit two feasible
heads with opposite query winners. Such a direction remains uncertified.

The nonconstant witness retains two active programs with different
quantity coordinates and decoders, a third program rejected by inconsistent
anchors, a nonzero imaginary coefficient, repeated training records and an
unseen target class. It also checks a copy direction using the declared prior
and an uncovered role using opposite feasible completions. These are finite
proof witnesses, not new benchmark measurements. Native implementation still
needs a training-only feature bank, an audited projection, complete products
and costs, forward/full/minibatch comparisons, resource caps and independently
repeated heldout measurements. No floating-point refinement or optimization
convergence theorem is asserted here.

Formalization: `ConvexDepth/CoordinatePrograms/{Basic,Head,Identification,Witness,
Coverage}.lean`; direct standard-axiom audit:
`scripts/CoordinateProgramAxioms.lean`. The witness uses class coordinates c/4,
quantity inputs 0, 1/2, 1, two structural roles and a copy input. The active
quantity representations are q and 2q-1, with different affine decoders.
Their masses 4 and 2 give full charge 78 within radius 128. Swapping the masses
to 2 and 4 is another feasible point. The third constant representation is forced
to zero by the two distinct observed coordinate answers. Both full heads have
a nonzero imaginary coefficient and identify the unobserved raw class 3.

## CM58. Observed-prefix equivalence for nested-count search

The CM57 head does not make discovery of a useful fixed input program convex.
The frozen343 experiment fits Count, eight Addition formats and depth1 C-RASP,
but its bounded beam finds no fitting depth2/3 program. The search uses current
truth-signature fit as one criterion for retaining children. The follow-up
investigates a broader search after grouping observed-equivalent children.
The results below specify when such grouping preserves nested counts; they
do not establish completeness of the old beam or convergence of a new search.

For a finite collection of symbol words W, form P(W), the union of all
nonempty prefixes of its words. This construction reads input symbols only;
BOS framing and targets are absent. Its cardinality is at most the sum of
the input lengths. P(W) is closed under nonempty prefixes, contains every
nonempty word in W, is contained in every prefix-closed pool containing W,
and is unchanged by applying the same closure again.

For a Boolean input program p, its inclusive count on w is

`C[p](w) = sum over 0 <= i < length(w) of 1[p(take(i+1,w))]`.

It lies between zero and length(w), regardless of p's internal depth. If p
and q agree on a prefix-closed pool S, then their inclusive counts agree on
S: every summand reads another member of S. Induction then preserves the
observed truth signature under arbitrary finite contexts built from Boolean
constants, current-symbol predicates, negation, conjunction, comparisons of
two counts plus an integer offset, and comparisons with an additional count.
Subprogram holes can be replaced simultaneously by observed-equivalent
programs. This permits one representative per complete observed signature
when the objective is preserving existence of a fit on that observed pool.
Agreement only on a complete word, without its earlier prefixes, is insufficient.

This is not global semantic equivalence or an unseen-answer certificate.
On the closed pool containing [true] and [true,false], the predicates
"current symbol is true" and "this is the first position" agree and are
nonconstant. At [true,true] they disagree, and their inclusive counts are two
and one. Nested comparison contexts can distinguish that unseen word too.
Any search quotient based on observations must retain this limitation; CM57's
separate target-realizability premise remains necessary. A native search also
needs complete signatures, explicit candidate/work/storage caps, independent
interpretation and a newly frozen heldout repeat. The finite closure bound
does not bound the number of distinct program signatures by a polynomial.

Formalization: `ConvexDepth/ObservedPrefixes/{Basic,Contexts}.lean`; direct
standard-axiom audit: `scripts/ObservedPrefixAxioms.lean`. The decoded-fit
theorem permits an arbitrary fixed Boolean-to-raw-label decoder, including
many-to-one decoders. Concrete witnesses use both observed truth values and
two distinct raw labels. Neither the symbolic proof nor its examples are new
heldout benchmark measurements.

## CM59. Necessary latent predicates at observed count transitions

The first CM58 native search retains all first-layer observed signatures and
finds four depth2 programs consistent with the original training pool. No
heldout measurement follows from that result. The depth3 training pool still
has no fitting root at count depth two. A complementary constraint comes from
changes of the observed root truth along adjacent input prefixes.

For Boolean predicates p,q,r on symbol words, put

`D(w) = C[p](w) - C[q](w) - C[r](w)`.

The optional extra count is r; when absent it is constantly false. Appending
one symbol a gives exactly

`D(w ++ [a]) = D(w) + 1[p(w ++ [a])] - 1[q(w ++ [a])] - 1[r(w ++ [a])]`.

This uses actual inclusive counts, not assumed arbitrary latent coordinates.
The integer increment is between -2 and 1, at any internal predicate depth.
For a proposed integer threshold k, crossing from D(w)<k to D(w++[a])>=k
forces the three new bits to be true,false,false. It also forces D(w)=k-1
and D(w++[a])=k. Crossing downward with r(w++[a])=false forces p=false,
q=true and the boundary values k,k-1. With r=true, a downward crossing only
forces p=false OR q=true: (false,false), (true,true), and (false,true) can
all cross down. An r=true increment can never cross upward.

These implications are necessary conditions for a candidate root expression.
A search may condition on its offset, optional extra symbol, negation, gate
and raw-label orientation, then constrain child programs at observed adjacent
prefixes where the root comparison value is known. A gate-false label does
not reveal its hidden comparison value. Missing parent labels do not reveal
a transition. The empty prefix's count difference is exactly zero. Inferred
child bits are branch-specific search constraints, not extra measured labels
for the CE objective. Every final candidate still needs the original complete
training-label and independent-interpreter checks. No completeness, search
convergence, or unseen answer identification follows from these implications.

An exact pair index may use `a < b+e+k` iff `a-e-k+1 <= b`, and the false
comparison iff `b <= a-e-k`. Both integer equivalences are formalized,
including negative thresholds. Intersecting those count constraints narrows
candidate pairs without adding training assumptions or dropping matching pairs.

For a retained left bank with per-prefix count range [lo,hi], a true comparison
requires `right >= lo-extra-offset+1`, and a false one requires
`right <= hi-extra-offset`. Conversely a retained right range requires
`left <= hi+extra+offset-1` for true and `left >= lo+extra+offset` for false.
These necessary bounds are proved with explicit count-range hypotheses. They
preserve candidates that can pair with a member of that retained anchor bank;
they do not establish that a truncated anchor contains the target program.

Formalization: `ObservedPrefixes/Transitions.lean` proves16 theorems, including
the append identities connecting the step to actual prefix counts, threshold
constraints, and the concrete ambiguity with an extra count. Each conditional
theorem has a satisfying example. `scripts/ObservedPrefixAxioms.lean` includes
the direct standard-axiom audit. The joint CM57 CE head is unchanged.

## CM60. Original multiplicities in complete-head cross-entropy

The complete Basis pools prepared by task345 have 5,806,712 supervised training
positions. Exact grouping can reduce storage and repeated computation when
many of those observations have the same complete features and observed raw
label. It must preserve the original empirical distribution. A set of distinct
observations averaged without weights generally changes the objective.

Let B be a finite set of original record IDs, q map IDs to a finite group
type K, and n_B(k) count the IDs in B with q(i)=k. Then

`sum_(i in B) v(q(i)) = sum_k n_B(k) * v(k)`.

The identity holds in any additive commutative monoid, with natural scalar
multiplication; it therefore also holds for continuous linear derivative maps.
The group counts sum to |B|. A group has positive count exactly when some
actual record of B maps to it. Consequently, requiring a predicate on all
original records is equivalent to requiring it on every active group. This
permits deduplicating identical constraints, with no dropped active group.

Normalize both objective sums by the original |B|, using the CM38 zero-inverse
convention for an empty batch. If every original row value equals its group's
value, the grouped and original means are equal. If each active group function
has derivative D_k at the current parameter, the grouped mean has derivative
`(1/|B|) * sum_k n_B(k) * D_k`. This includes derivative existence, rather than
only equality between the library's totalized derivative expressions.

For the actual full-vocabulary head, require equality of every feature
coordinate and equality of the observed raw label between a record and its
representative. The ordinary CE, `log(sum_c exp(logit_c)) - logit_y`, then
agrees at every complete coefficient tensor. All raw classes remain in the
softmax sum. The original finite CE is differentiable at every tensor: logits
are affine and the softmax sum is positive. Equality of functions transfers
its actual Fréchet derivative, and any supplied derivative witness, to the
grouped objective. The grouping is fixed independently of the learned tensor;
matching only current logits is insufficient for this theorem.

Composing this identity with CM57's affine complete head preserves convex
ordinary CE on its existing joint domain. All program masses and decoder
coordinates remain joint variables. The perspective budgets, copy and class
priors, positive total mass and identifying anchors are unchanged. In
particular, grouping does not make inconsistent noisy anchors feasible, nor
does it replace noisy observed labels by unknown clean labels. A noise-tolerant
domain needs its own justification. No new unseen-identification conclusion
follows from grouping, feature equality, or spectral flatness.

Each minibatch uses its own n_B and its actual |B|, including a short final
batch. Global group counts cannot replace batch counts in a minibatch update.
Three distinct record IDs grouped as two false and one true, with values zero
and one, have original mean 1/3; unweighted unique-group averaging gives 1/2.
The one-record tail containing the true group has mean one. A second concrete
witness groups distinct raw inputs zero and two by the complete features
`[1, input mod 2]`, with input one in the other group. It retains nonconstant
features, two raw labels, and every coefficient of the two-class CE. The CM57
convexity example also supplies a previously proved feasible point, so an empty
domain is not the only witness.

Formalization: `ConvexDepth/WeightedObservations/{Basic,Head}.lean`; all16
theorems are included in `scripts/WeightedObservationAxioms.lean`. Conditional
theorems have satisfying examples. These are exact real-arithmetic results;
they do not assert bitwise equality of differently ordered floating-point sums,
native memory bounds, or an empirical success on the new Basis cases. A native
implementation still needs independent objective/gradient checks, complete
record accounting, genuine batch tails, and a frozen repeated evaluation.

## CM61. Selected exact anchors with all noisy observations in CE

CM60 preserves an objective; it does not make the fully anchored CM57 domain
feasible under label noise. For a fixed program's complete feature vector h,
suppose both exact equations `center(theta,h) = gamma*a` and
`center(theta,h) = gamma*b` are imposed with a != b. Subtracting gives
`gamma*(a-b)=0`, hence gamma=0. If every program has such an identical-feature
conflict, all masses vanish. A strictly positive total-mass constraint then
makes the fully anchored domain empty, independently of the optimizer or the
spectral bound. Raising a memory cap or taking more optimization steps cannot
repair this mathematical infeasibility.

Exact center equations are additional identifying assumptions, not part of
the definition of ordinary cross-entropy. Fix a set K of original anchor
indices before the convex stage and impose only those equations. Retain the
CM57 identity-copy prior, perspective charge for every program, nonnegative
masses, positive total mass, and global complete-coefficient budget. This is
CM57's domain with the anchor index restricted to the subtype K. It is convex.
If K is reduced, the domain enlarges; every fully anchored feasible point is
still feasible after selection. The selector itself is outside this convexity
statement and must not consult heldout labels or unknown clean training labels.

The empirical loss is separately indexed by the complete original batch B:
`(1/|B|) * sum_(i in B) CE(actual_head(input_i), observed_label_i)`.
It does not replace B with K. All noisy labels remain observations with their
original multiplicities, and the actual ordinary full/minibatch CE remains
convex on the selected-anchor domain. The theorem permits a different anchor
index type and CE record type, making the distinction explicit. CM60 grouping
can be applied to those complete CE records independently of the fixed anchor
selection. No pseudo-clean label or filtered CE objective is introduced.

A concrete witness has one program, two identical content features `[1,0,0,0,0]`
with different observed coordinates zero and one, and a third copy feature
`[0,0,0,0,1]` with observed coordinate one. The fully anchored positive-mass
domain is empty. Keeping only the third equation admits parameters
`[1/2,0,0,0,1,1]`, with mass one and complete charge eight within local and
global budgets16. The content center is one half and the copy center is one.
The example's convex CE still contains all three records and both raw labels.
It is therefore a nonempty relaxation with genuine noisy observations, not
an empty-domain convexity witness or a claim to fit both labels exactly.

Removing an anchor also removes any identification supported only by that
equation. This result proves neither recovery of unknown clean labels nor
99% heldout accuracy, a noise model, convergence of the numerical optimizer,
or realizability of a newly discovered program. Those remain separate
statistical, semantic and empirical obligations. A native implementation must
record the selected anchors, preserve all original CE records, audit the full
complex head and budgets, and freeze before heldout evaluation. The original
fully anchored baseline remains a distinct comparison.

Formalization: `ConvexDepth/SelectedAnchors/Basic.lean`, nine theorems with
satisfying conditional examples and explicit feasible/infeasible witnesses.
`scripts/SelectedAnchorAxioms.lean` audits all nine directly.

## CM62. Stream one side of the observed count-program join

The complete344 comparison leaves depth3 unfitted. Its training-only evidence
shows13 admissible root configurations: three complete retained-bank joins
have no fit, while ten configurations truncate at least one latent bank at
262144 distinct observed signatures. Several have one complete bank and one
truncated bank. A storage cap on both banks is therefore stronger than needed:
index one fixed anchor bank and stream each query from the other side, without
retaining the whole query bank. No target AST or heldout answer is needed.

For a known root comparison, let a,b,e be the actual inclusive left, right and
extra counts at an observed prefix and k its integer offset. The comparison
a < b+e+k is equivalent to a <= b+e+k-1 and to b >= a-e-k+1. Its negation is
equivalent to a >= b+e+k and to b <= a-e-k. These are exact integer thresholds,
including negative offsets, for either choice of indexed bank. Intersecting
them over every known observed prefix is equivalent to the original count
comparison fit. Inactive gates and missing labels still give no hidden
comparison truth. Original raw-label checks remain necessary after assembly.
Every inclusive count lies between zero and its prefix length. Clamping an
index threshold to [0,length+1] preserves the exact set of admissible counts;
the two endpoints are the all-count and empty-count sentinels.

Fix an anchor set A and a finite family of query chunks B_s. Existence of a
fitting pair in A times the union of all B_s is equivalent to existence in one
of the individual joins A times B_s; the same holds when left and right are
interchanged. Chunks may overlap. An implementation can process one query at
a time, discard it after checking all retained anchors, and stop on a verified
fit. A work-limit stop must remain explicit. This removes the query retention
cap; it does not establish completeness of a truncated anchor, the enumeration
of all grammar terms, or generalization to an unseen word.

Caching a previously tested query requires equality of its entire observed
prefix signature, on a prefix-closed pool containing every constrained word.
CM58 then gives equality of all counts read by the join, so the set of fitting
retained partners is unchanged. However, an AST-size cap adds another condition.
If cached child costs are no larger than the arriving costs, any arriving pair
that fits under the combined child-plus-wrapper cap has a cached equivalent
that also fits. Native costs must equal the actual AST node counts. A cheaper
arrival must be checked again. A bounded cache can evict entries and repeat
work later; eviction must never discard an untested candidate.

Two nonconstant observed labels demonstrate the distinction. On words [true]
and [true,false], compare the elapsed count to the count of a true final symbol
plus one. The constant-true left child fits both observations; a retained bank
containing only the constant-false child does not. Thus a capped bank can miss
an actual fit. On the same words, double negation of the true-symbol predicate
is equivalent to that predicate. The native trees have three and one nodes.
With a three-node constant-true anchor and a five-node count/offset wrapper,
only the cheaper query fits a nine-node cap. An expensive cached equivalent
cannot justify skipping this later fit.

Formalization: `ObservedPrefixes/Streaming/{Basic,Cache}.lean`, with direct
audit `scripts/StreamingSearchAxioms.lean`. The fit predicate uses actual
inclusive counts and all supplied observed comparison values. The size result
requires checked child costs; it is not a formal Rust heap bound. Native
indexing still needs padded-bit checks, exhaustive small-bank comparisons in
both orientations, cache/size counterexamples, explicit work/storage admission,
independent input interpreters, and a newly frozen complete heldout repeat.

## CM63. Exact complete products and genuine unobserved ties

The frozen347 producer recovers twelve depth3 programs fitting all original
observations. Its first heldout evaluation exposes two separate problems:
the two numerical complete-product paths fail their residual/winner check,
and even successfully measured heads miss some longer sequences. A numerical
repair must preserve the actual saved coefficient tensor and must not infer
the correct target from agreement between approximate computations. The347
producer and its refused or unsuccessful outcomes remain immutable; its full
independent repeat is still required.

For a fixed signed input row w, source features S and arbitrary complete head
W, define N_c = sum_k W_kc * sum_i w_i S_ki. Finite distributivity gives
N_c = sum_i w_i * headLogits(S,W,i,c). The existing signed-context construction
therefore has actual logits N_c / max(1,sum_i |w_i|). The denominator is common
to all raw classes and strictly positive. Both strict ordering and exact
equality of raw logits are precisely those of the unnormalized numerators.
This holds for every actual stored coefficient, without assuming it equals
an ideal compact decoder expansion after floating-point construction.

The independent complex contraction is also explicit. For rectangular complex
source U and complex coefficient tensor H, real features are Re(U) and -Im(U),
and real coefficients are Re(H) and Im(H). The complete numerator equals
Re(sum_k (sum_i w_i U_ik) H_kc). Both coordinates and every source/coefficient
entry participate. A signed four-point Fourier example has nonzero imaginary
coefficients and distinct complete numerators four and eight.

Subtracting one class's exact numerator before the common division is exactly
a common shift of the original logits. The underlying coefficient model and
ordinary full-vocabulary CE are unchanged. The reference class can be chosen
from the computed scores; the identity holds for every choice and does not
need a target label. This is a numerical representation of the same objective,
not a new affine parameterization or an unproved floating-point identity.

Valid lower and upper bounds on the exact numerators certify a unique winner
only when its lower bound exceeds every competing upper bound. The theorem
requires the bounds to contain the actual products. Two matching rounded
argmaxes, or an arbitrary tie tolerance, do not supply that premise. The two
vectors [epsilon,0] and [0,epsilon] have residual at most epsilon yet opposite
strict winners for every positive epsilon. A native implementation must
separately establish conservative bounds or evaluate
ambiguous products exactly, account for all coefficients and work, preserve
strict ordering through conversion, and retain any unresolved failure.

Correct arithmetic still does not identify arbitrary unseen labels. A concrete
CM57 witness has two active programs with different quantity scales, different
decoder slopes, mass one each, the copy constraint, sufficient weighted charge
18, local budgets16 and global radius32. Both fit two observed raw labels zero
and one. On a third input their decoded centers are zero and one, so the two
actual complete logits tie exactly; the common normalizer is six. Each program
realizes a different binary target on every input. These target completions
agree on all observations and disagree at the third input. The complete tensor
satisfies the original budget. Flat bounded codes and more accurate products
cannot choose the true completion without an additional identifying premise.

Formalization: `ConvexDepth/ExactScoring/{Basic,Witness}.lean`, eighteen
theorems, with direct audit `scripts/ExactScoringAxioms.lean`. The conditional
interval theorem has a nonconstant satisfying example. These are exact-real
results. They do not prove a Rust arithmetic refinement, empirical99% accuracy,
or that correcting numerical ties repairs the separate program ambiguity.

## CM64. Bounded confidence and exact convex elimination for latent decoders

CM53 learns a possibly many-to-one raw decoder without multiplying learned
factors. Its shared masses sum to one and its external margin is fixed, so
ordinary CE on exactly anchored observations is constant. Changing the program
mixture cannot improve that observed CE or choose unobserved decoder entries.
Task351 first extends the explicit joint domain to learn confidence. This is a
generic input-program construction; no arithmetic target formula is assumed.

Fix a finite input-only family `p_j : X -> S`, a declared candidate order, the
original observed pairs `(x_i,y_i)`, and a unitary source code `U` on `S`.
The learned coordinates are shared masses `w[j]` and the joint raw tensor
`gamma[j,s,c]`, with the following explicit constraints:

```
w >= 0,       delta <= sum_j w[j] <= R,
gamma >= 0,   sum_c gamma[j,s,c] = w[j] for every j,s,
w[j] = 0 for programs outside the declared order,
gamma[j,p_j(x_i),c] = w[j] * 1[c=y_i] for every original sample and raw class.
```

This domain is convex in all of its actual learned coordinates. The decoder
tensor is not parameterized by a product of an independently learned routing
weight and decoder. Every entry lies between zero and its program's mass,
including states absent from training. Zero program mass forces its entire
block to zero. If two observed labels conflict at one latent state, that
program's mass must therefore vanish. Removing exactly these inconsistent
fixed candidates preserves the entire feasible set, in both directions.
This is exact elimination of constrained zero blocks. It does not justify
discarding a compatible candidate, truncating the family, or treating adaptive
grammar discovery as convex. The input-only family must be fixed beforehand.

For `0 < delta <= R`, the bounded domain is nonempty exactly when CM53's observed
filter is nonempty. A retained program and its training-learned dictionary
give a feasible vertex at mass `delta`; an explicit default completes only
the unobserved states. More generally, scaling any proved CM53 feasible tensor
by an admitted nonnegative confidence gives one feasible joint point. This is
a construction within the explicit convex domain, not a factored restriction
of its learned coordinates.

Reuse CM53's actual complete complex head at external margin one:
`H_j = U* gamma_j`. Expanding every real and imaginary coefficient gives

```
z_q(x,c) = sum_j gamma[j,p_j(x),c],
rho(q) = sum_j w[j],
complete per-class coefficient budget <= 2 * R * card(S).
```

The bound follows from the unitary source entry/table bounds and the shared
mass cap. It is independent of the number of programs, but the complete head
still has `2 * card(J) * card(S) * card(C)` real coordinates. Search, storage and
product costs must be accounted for separately. Repeated input states do not
make the empirical feature matrix flat. The spectral source supplies a capacity
bound, not an identifying target representation.

All actual logits are linear in the joint variables. Ordinary full-vocabulary
log-sum-exp CE is convex, as is its average over any finite set of original
record IDs. Repeated values retain distinct IDs, and a short final minibatch
uses its actual cardinality. On an observed record the complete logit vector
is `rho(q)` at its raw label and zero at every other class, so its actual CE is

```
log(1 + (card(C) - 1) * exp(-rho(q))).
```

Every nonempty batch of those observed records has the same function of total
mass. Increasing feasible total mass cannot increase this CE. Thus confidence
can be learned, but this objective still cannot select unknown decoder entries
or distinguish compatible target completions. No optimizer convergence claim
is added here. Native forward-only, full-batch and minibatch trajectories must
be recorded as executed, including zero updates or budget refusals.

At a query, CM53's executable acceptance test requires nonempty retained
support, an observed latent state in every retained program, and unanimous raw
labels. The observed anchor equations and annihilation of zero blocks then
force the same complete raw logit vector for every bounded feasible tensor.
A positive lower total mass makes that accepted class a strict winner. To
identify it with the true target still requires the separately stated prior
that the unknown target is some enumerated input program composed with an
unknown deterministic decoder, and that observations are correct. Neither the
target program nor decoder is a learner input. Uncovered states remain outside
this conclusion, irrespective of confidence or flatness.

The nonconstant witness reuses CM53's four-state coordinate programs and its
training-learned two-class many-to-one dictionary. The new input `(3,2)` is
identified through a covered state. Both confidence two and confidence four
are feasible inside `[1,8]`, and the latter strictly lowers actual observed CE.
At the unobserved state three, two feasible heads still give opposite strict
raw margins while agreeing on every observed complete logit. The actual source
and head have nonzero imaginary coordinates. Repeated-record and singleton-tail
examples instantiate the batch theorem.

Formalization: `ConvexDepth/BoundedLatent`, with direct audit
`scripts/BoundedLatentAxioms.lean`. These are exact-real theorems. They do not
assert a Rust floating-point refinement, discover an unrestricted program
family, or establish any additional stress-test accuracy. A native extension
must freeze its generic input-only family before labels, preserve every
compatible candidate or justify exact parameter elimination, and keep raw
scores separate from coverage certificates and structural realizability.

For the native variable-size state bank, use block-diagonal copies of the
four-point Fourier matrix, indexed by `Fin 4 x B`. Complete block composition
preserves both unitary identities, so this source satisfies the actual unitary
premise for every finite block type. Its entries are exact zeros or signed
real/imaginary halves. The complete column bound specializes to
`8 * R * card(B)`. Zero source entries can be skipped as exact products of zero;
this does not remove any learned coefficient from storage or its column charge.
The nonconstant many-to-one witness inhabits this block-source domain as well.

## CM65. Exact elimination of observed decoder rows and its induced geometry

The complete CM64 implementation stores every joint raw-class row, including
rows already fixed by observed anchors. This duplicates a program's shared mass
at each observed state. The large constant controls retain all 4,767 compatible
input programs, so the complete tensor is expensive even though many of its
entries are constrained. A larger fixed input family needs a representation
that avoids these redundant coordinates without fixing unknown answers.

For each program, let `L_j(s)` be CM53's actual partial dictionary, obtained
only from the original training observations. Define `Unknown_j` as the states
where this lookup is `none`, and `Known_j` as its complement. The reduced
coordinates consist of all shared masses `w[j]` and all raw-class entries
`u[j,s,c]` for `s in Unknown_j`. Their constraints are explicit:

```
w >= 0,    delta <= sum_j w[j] <= R,
w[j] = 0 for programs outside the complete observed-compatible filter,
u >= 0,    sum_c u[j,s,c] = w[j] for every unknown state s.
```

Reconstruct the original tensor by setting an observed row to
`w[j] * 1[c = L_j(s)]`, and using `u[j,s,c]` at every unknown state. This map is
linear in the actual learned coordinates. Conversely, restriction keeps all
masses and every unknown row. Restriction after reconstruction is the identity
on the entire reduced coordinate space. Reconstruction after restriction is
the identity on every feasible original CM64 tensor, including coefficients
never read at a training input. The two explicit feasible domains therefore
correspond bijectively. Inconsistent candidates have zero mass by CM64; the
proof does not truncate the compatible family or choose a decoder completion.

The reduced domain is convex. The reconstructed complete complex head is
exactly the previous head, including all real and imaginary coefficients and
all raw classes. Thus every full or minibatch CE value is unchanged, with the
same original record IDs, multiplicities and short-tail denominator. Ordinary
CE remains convex in the reduced coordinates. The complete column bound is
still `2 * R * card(S)`; no coefficient is exempt merely because it is generated
from a mass instead of stored independently.

Exact feasible-set equivalence does not preserve unweighted Euclidean geometry.
For two reduced points `a,b`, the original squared coordinate distance is

```
sum_j (1 + card(Known_j)) * (a.w[j] - b.w[j])^2
  + sum_j sum_(s in Unknown_j) sum_c (a.u[j,s,c] - b.u[j,s,c])^2.
```

Each distinct observed state contributes one affine copy of the mass. This is
a state count, not the record multiplicity used in empirical CE. Omitting these
weights changes distances and does not justify claiming the same projected
gradient trajectory. The identity describes distances between reconstructed
points; equivalence of a numerical projection from an arbitrary off-domain
gradient proposal still needs its own derivation and numerical validation.

The nonconstant CM53/64 witness has three observed states in its active program
and a fourth, genuinely unknown state. Both opposite raw completions remain
feasible in the reduced domain and give opposite strict complete complex
margins on the unseen query. Between confidence two and four with a fixed
completion, full squared distance is 20 while unweighted reduced distance is 8.
Examples instantiate every conditional theorem, including duplicate records,
a singleton batch tail, and a source/head with nonzero imaginary coordinates.

Formalization: `ConvexDepth/ObservedRows/{Basic,Domain,Geometry,Head,Witness}.lean`;
24 theorems are audited directly by `scripts/ObservedRowAxioms.lean`. These are
exact-real coordinate and geometry results. They establish neither new target
realizability nor a Rust storage bound, floating-point equivalence, optimizer
convergence, or additional heldout accuracy. A native follow-up must preserve
every unknown entry, evaluate and charge the actual complete complex head,
declare its projection geometry, account for expanded storage and work, and
freeze a new producer before its own complete independent comparison. The
already frozen CM64 cohort remains unchanged.

## CM66. Off-domain projection proposals after exact row elimination

CM65 preserves distances between reconstructed feasible tensors. A projected
gradient step also needs to project a proposal that can violate the anchors,
nonnegativity, and the shared row sums. Merely restricting such a proposal is
incorrect: its known-label coordinates contribute to the best shared mass.

Fix the same observed dictionary and an arbitrary full proposal `v`. For each
program write

```
k_j = 1 + card(Known_j),
l_j = v.w[j] + sum_(s in Known_j) v.gamma[j,s,L_j(s)],
t.w[j] = l_j / k_j,
t.u[j,s,c] = v.gamma[j,s,c] for s in Unknown_j.
```

Every `k_j` is strictly positive. The exact original squared projection
objective at any reconstructed reduced point `r` is

```
distance_squared(expand(r), v)
  = sum_j k_j * (r.w[j] - t.w[j])^2
      + sum_j sum_(s in Unknown_j) sum_c (r.u[j,s,c] - t.u[j,s,c])^2
      + K(v),

K(v) = sum_j (v.w[j]^2
         + sum_(s in Known_j) sum_c v.gamma[j,s,c]^2 - l_j^2 / k_j).
```

The constant retains every off-label proposal square. It is independent of
the point being optimized, so it cancels from pairwise comparisons. Together
with CM65's exact domain correspondence, this proves equivalence of the full
and weighted reduced minimizers, with feasible membership explicitly included
on both sides. No assumption that the proposal is itself feasible, or that it
arose from a particular derivative, is needed. This equivalence does not by
itself assert that a numerical solver returns a minimizer or that two floating
implementations have bit-identical trajectories.

The nonconstant dictionary example has mass weights two and four. A concrete
proposal with masses -1 and 2, nonzero off-label coordinates and a nonuniform
unknown row yields reduced centers -3/2 and 5/4. The unknown class entries five
and four are copied without alteration. The original objective at a feasible
confidence-four model is exactly 123. Thus the example covers an actual
off-domain proposal, unequal anchor counts and genuinely free unknown rows.

Formalization: `ObservedRows/{Projection,ProjectionWitness}.lean`, eleven
theorems with direct audit `scripts/ObservedProjectionAxioms.lean`. The native
follow-up must retain all unknown entries, compute the selected proposal sums,
validate its weighted simplex/mass solver independently, and distinguish a
finite-precision KKT residual from this exact-real minimizer theorem. The
complex head, ordinary empirical CE and fixed input-family requirements of
CM64/65 are unchanged; these projection identities add no heldout label or
benchmark success.

## CM67. Widening a fixed input family preserves the full old model class

Before replacing a finite input-program bank by a larger fixed bank, distinguish
representational inclusion from a claim about a new fitted model. Write the
old candidate type as `J` and the added type as `K`, with arbitrary fixed input
maps into the same latent state type. Enumerate the union as `J + K`, retaining
the old order followed by the new order. Equal program functions need not be
deduplicated. Every added compatible candidate has its own full joint block.

Embed an old tensor by keeping every old mass and every raw decoder coordinate,
including all unknown rows, and setting every new mass and decoder coordinate
to zero. This is an injective map of the full parameter space. Its total mass
equals the old total. Nonnegativity, shared row masses, enumeration support and
every observed anchor hold in the combined domain whenever they held before.
No condition on the behavior of added programs is needed for this inclusion.
The exact consistency filter retains an old or new candidate if and only if
that candidate was retained in its own complete branch. Thus widening and
CM64's zero-block elimination do not justify dropping any compatible member.

The actual complete complex products at the embedded tensor agree with the
old logits for every input and every raw class. Consequently every ordinary
indexed batch CE is identical, with all original record multiplicities and the
actual batch denominator. The embedded head still satisfies the full-column
bound `2 * R * card(S)`, charging both real and imaginary coefficients. The
larger class can therefore attain every old feasible loss value at that same
bound. The capacity bound is independent of candidate count; expanded storage,
program evaluation and complex-product work must still be admitted explicitly.

The nonconstant CM64 example is extended by two nonconstant input programs,
the first input coordinate and the sum of the two coordinates. Its nonzero
imaginary feature and head entry survive. Both feasible completions at its
unobserved latent state also survive and still give opposite strict raw answers.
Examples instantiate every conditional theorem using the four-point complex
Fourier code, duplicate records and a singleton minibatch tail.

Formalization: `ConvexDepth/FamilyExtension/{Basic,Complex,Witness}.lean`, with
thirteen direct axiom checks in `scripts/FamilyExtensionAxioms.lean`. These
results justify a prospective broader fixed family; they do not assert target
realizability, identification of unseen labels, unchanged floating trajectories,
or improvement by a finite training schedule. In particular, initializing every
added candidate with positive mass is not the zero-block embedding. Any wider
native bank must retain all compatible members, audit every interpreter and
decoder entry, account for complete memory and work, and undergo a new frozen
comparison before new heldout accuracy is credited.

## CM68. Lossless packed observations preserve the full training problem

The candidate count affects stored observations as well as learned parameters.
An encoded training record contains the state of every retained input-only
program and its original raw label. Keeping those state indices in 64-bit
machine words costs `8 * N * P` bytes before storing the joint decoder or its
complex head. This cost can be reduced without identifying candidates, merging
observations or restricting an unknown decoder row.

Let the state and raw-class counts be `s,c <= 65536`. Represent an original row
as `(J -> Fin s) * Fin c`, and encode each natural value with `UInt16.ofNat`.
For these explicit bounds, every word's `toNat` is exactly its original value.
All encoded words are in their declared domains, decoding returns the entire
original row, and encoding is injective. Conversely, reencoding any valid word
row gives exactly that word row. A checked decoder returns `none` exactly when
a stored state or class lies outside its declared domain. Being representable
in 16 bits alone does not make a value a valid state of a smaller vocabulary.

Precomputing all fixed program outputs at an original observation preserves
the first-observed raw decoder lookup and the complete ordered consistency
filter. It also preserves the exact full bounded joint feasible set: every
anchor equation, shared row mass, nonnegative coordinate, support constraint
and total-mass bound is unchanged. All unknown raw-class coordinates remain
free. Packing and decoding those observations preserves the same set.

Every complete realified complex product is identical at the decoded input,
for an arbitrary complete joint tensor and arbitrary complex source matrix.
No unitary hypothesis is necessary for this representation identity. Every
ordinary indexed batch CE is therefore identical with its actual denominator.
Distinct original IDs can carry equal observations; they are not coalesced.
An ordered ID list retains every occurrence, and a singleton final batch stays
a singleton. No target formula or heldout label enters these identities.

The nonconstant witness has two input-coordinate programs, five original IDs,
duplicate observations, distinct raw classes and a singleton tail. It uses the
four-point Fourier source and a complete head with nonzero imaginary entries.
A separate range witness preserves state 4095 and label 65535. A word with state 4
is rejected when the declared state count is 4. Examples instantiate every
conditional theorem; none assumes a recovered target or a trained optimum.

For the complete bank of 299,551 candidates, 18,528 records and 193 classes with 196 padded
states, the existing conservative core-buffer forecast is at least
110,475,896,672 bytes, even with only the mandatory three padded unknown states
per program. Its observed state matrix alone occupies44,400,647,424 bytes.
Replacing only those indices by UInt16 would save33,300,485,568 bytes; it does not
change the sizes of masses, unknown decoder entries or complex coefficients.
This arithmetic is a prospective storage forecast, not a measured Rust peak.
The native implementation must still account for row-decoding scratch and all
simultaneously live buffers under the 110 GB limit, and stop on either disk's
admission failure.

The completed first task354 training-only probe subsequently reports 1,126,298
unknown rows and 137,825,884 bounded head columns for this control. Applying the
existing forecast to those complete counts gives122,579,278,464 bytes. Replacing
only the state-index storage and adding one decoded row gives89,281,189,304 bytes.
This candidate forecast fits the 96 GB core admission; implementing and testing
the representation, then measuring the original run, are still required. The
probe's original refusal and original audit must not be overwritten.

Formalization: `ConvexDepth/PackedRecords/{Basic,Observations,Complex,Witness}.lean`,
21 theorems audited directly by `scripts/PackedRecordAxioms.lean`. These are
exact integer and exact-real semantic identities. They do not prove Rust
memory safety, IEEE-754 refinement or unchanged optimization convergence.
The native follow-up must compare all decoded states, full domains, complex
coefficients, objective values, actual update ranges and saved parameters with
the existing expanded-record implementation, and preserve the frozen training
probe's original audits separately from any refined memory forecast.

## CM69. Uniform unknown rows at exact projections over the complete domain

CM65 leaves every unobserved decoder coordinate free. A particular projection
can nevertheless have uniform unknown rows when its proposal has that symmetry.
This is a conditional property of a minimizer over the original feasible set;
it is not permission to replace that set by a uniform-row subspace.

Let the finite raw-class type be nonempty, with actual cardinality `n > 0`.
For any real row `x` with sum `m`, and any scalar proposal value `u`,

```
sum_c (x[c] - u)^2
  = sum_c (x[c] - m/n)^2 + n * (m/n - u)^2.
```

The first term is nonnegative and vanishes exactly when every entry is `m/n`.
The identity and fixed-mass uniqueness include `m = 0` and arbitrary real `u`.
Nonnegativity of the row is unnecessary for this identity.

For a feasible CM65 reduced tensor `r`, let `B(r)` keep its masses and replace
each unknown row by `r.w[j]/n`. Since the masses are nonnegative, `B(r)` remains
in the same complete domain, including its support, anchors and mass interval.
For a reduced proposal `v` whose class entries agree within each unknown row,
CM66's weighted distance satisfies the exact decomposition

```
D(r,v) = D(B(r),v)
           + sum_j sum_(s in Unknown_j) sum_c (r.u[j,s,c] - r.w[j]/n)^2,

D(B(r),v) = sum_j [k_j * (r.w[j] - v.w[j])^2
                      + sum_(s in Unknown_j) n * (r.w[j]/n - v.u[j,s,c0])^2],
k_j = 1 + card(Known_j).
```

Here `c0` is any actual raw class. Different unknown proposal rows may have
different values. Every unknown row contributes to the mass-only objective;
the anchored-copy weight `k_j` is retained. Comparing a global minimizer with
its feasible point `B(r)` forces the complete nonnegative remainder to zero.
Therefore every exact minimizer has each unknown entry equal to its mass
divided by `n`. No existence theorem is claimed.

CM66 transfers this result to the original full joint tensor. The full proposal
can violate mass constraints and observed anchors and can have arbitrary
off-label coordinates on known rows. Only its unknown rows must be uniform.
The conclusion concerns a minimizer of the original full Euclidean objective
over the entire CM64 bounded domain. All other feasible completions remain.

The nonconstant two-class witness retains distinct observed labels at three
states. At mass four, its unknown balanced row is `(2,2)`, and the complete
complex head gives logits `(2,2)` at the unseen query. A head coordinate has
imaginary component `-1`, so the example uses the complete complex construction.
The two prior feasible completions `(4,0)` and `(0,4)` remain in exactly the
same domain and still give opposite strict unseen answers. One such nonuniform
tensor is also a global projection of itself, proving that the uniform-proposal
condition matters. Examples cover every conditional theorem, zero mass, full
complex coefficient bounds, duplicate indexed observations and a singleton CE
batch tail.

Formalization: `ConvexDepth/UniformRows/{Basic,Projection,Witness}.lean`, with
27 direct axiom checks in `scripts/UniformRowAxioms.lean`. These are exact-real
projection results. They do not identify unseen answers, prove general update
trajectories uniform, supply a floating-point minimizer, or establish new test
accuracy. Any native storage follow-up must represent arbitrary nonuniform
rows as well, preserve all compatible candidates and original records, charge
the actual complete complex head, and test its numeric behavior separately.
The frozen task355 comparison remains unchanged.

## CM70. Lossless repeated values in the complete decoder and complex head

CM69 gives a conditional symmetry of exact projections over the full domain.
It does not permit replacing that domain by uniform unknown rows. A separate
representation result can reduce storage while still accepting every arbitrary
unknown row, including nonuniform proposals and feasible completions.

For a finite class type `C`, a row `x : C -> A` and a baseline `b : A`, store
`b` together with the finite set `{c | x[c] != b}` and the value at each such
class. Reading an exceptional class returns its stored value; reading every
other class returns `b`. Exact equality suffices: no arithmetic structure on
`A` is needed. Reading after encoding is the identity and encoding at a fixed
baseline is injective. Therefore this also applies to exact 64-bit words,
where positive and negative zero remain distinct words.

Count stored values separately from the native container and index overhead.
A constant row takes one value. If `b = x[c0]` for an actual class `c0`, the
exception set omits `c0`, so baseline plus exceptions use at most `card C`
values. If `x` equals `b` outside a declared finite set `E`, all stored
exceptions lie in `E` and at most `1 + card E` values are stored. A completely
nonuniform row is still represented exactly; no class is removed.

Apply this encoding to every CM65 unknown row and keep every program mass.
Decoding recovers the entire reduced tensor for arbitrary baselines and
arbitrary real coordinates, without nonnegativity, uniformity or positive-mass
assumptions. Expand the decoded tensor with the original observed dictionary.
The image of stored points whose decodings satisfy the original reduced
constraints is exactly the full CM64 feasible set. All compatible programs,
observed anchors, mass bounds and nonuniform unknown completions remain.
Full-coordinate squared distance to an arbitrary proposal is identical; the
known-row weights from CM65/66 and every unknown term are still present.
Storage tags are an implementation choice, not new learned variables on which
convexity or a different Euclidean geometry is asserted.

The same encoding applies to every complete realified complex head row.
After decoding both parameter and head storage, every real and imaginary
coefficient, every query logit and every ordinary indexed CE is unchanged.
Original record IDs, repeated observations and the actual short-batch
denominator are retained. Under the existing unitary and feasible-domain
hypotheses the full logical head still satisfies `2 * R * card S`; repeated
coefficients continue to count in that norm even if their values share storage.

There is a sharper conditional storage bound for the actual head. Fix one
source column `f` supported on a state set `A`. Two raw decoder columns that
agree on `A` give equal conjugate coefficients at `f`, in both components.
When unknown decoder rows are class-uniform, only the known raw labels inside
`A` can differ from a class outside that label set. Each real or imaginary
head row therefore needs at most one shared value plus those exceptional
classes. This is conditional on actual source support, row equality and the
existence of an outside class; otherwise the general lossless encoding remains.
It does not identify an unseen answer or assume that all future proposals
stay uniform.

Nonconstant witnesses retain the original opposite unseen answers `(4,0)`
and `(0,4)`, a balanced unknown row `(2,2)`, and an actual imaginary Fourier
coefficient `-1`. A four-state, five-class table with two observed raw labels
needs at most three values per real or imaginary Fourier row while preserving
all five logical classes. A word-level witness preserves positive zero,
negative zero and the least positive subnormal as distinct original words.
Examples also cover zero mass, duplicated original IDs and a singleton tail.

Formalization: `ConvexDepth/RepeatedRows/{Basic,Model,Complex,Witness}.lean`,
27 direct checks in `scripts/RepeatedRowAxioms.lean`. These are representation
identities and explicit logical value-count bounds. They do not prove a native
container's peak memory, Rust safety, IEEE arithmetic refinement, unchanged
finite training trajectories or new task accuracy. The native follow-up must
retain arbitrary explicit fallback, preserve coefficient and parameter bits
and original operation order, charge every live allocation under 110 GB RAM,
and stop on shortage of either disk. Every frozen running producer stays fixed.

## CM71. Identical complete columns share ordered computations exactly

CM70 saves storage by retaining a baseline and every exceptional head column.
It still evaluates the complete class vocabulary. The same representation can
also identify computations whose results agree exactly, without deleting any
class, coefficient, unknown completion or term of the logical head norm.

Let an ordered list index the original operations, including repeated indices.
For each index and class, an event is either absent (the original loop skips
that class) or contains the complete payload to a deterministic accumulator
update. Start every class with the same initial value. If two classes have the
same event at every actual occurrence, induction on the ordered list proves
that their final accumulators agree. This statement holds for arbitrary value
and payload types and arbitrary deterministic updates. In particular, they may
be exact words; no associativity or commutativity is required.

An inactive skip remains a skip. It is not replaced by adding zero, which need
not preserve a machine accumulator word. Payload equality for the actual
complex contraction includes the source and coefficient components in both
real and imaginary parts. Retain each class's original program/frequency order,
including multiplication, subtraction and accumulation. The proof neither
reorders operations nor replaces repeated additions by multiplication.

A conservative sufficient condition follows from CM70's block storage. Form a
global exceptional class set containing every explicit coefficient exception
and every active class in each known-only block. Any two classes outside this
set receive the same baseline payload in every all-class block and are both
inactive in every known-only block. Their complete event sequences agree for
every query. Evaluate the exceptional classes independently and one outside
representative, then copy the representative's final word to every other
outside class. Expanding gives exactly the entire original class function.
If no outside class exists, the original complete evaluation is the fallback.
Arbitrary nonuniform parameters can cause that fallback; none are excluded.

Every subsequent computation of the complete output vector is unchanged,
including ordinary CE in its original class summation order, raw argmax and
record/minibatch accumulation. Coefficient storage, audits, gradients,
projections and checkpoint interpretation remain separate. The full complex
head and its previously proved norm bound still count every logical class.
An implementation must reconstruct the sharing plan when its head changes.

A nonconstant four-class witness retains one exceptional class, three shared
classes, a known-only block and repeated operations. Decimal accumulation gives
7897 for the exceptional class and 242 for each shared class. Deleting the
exception, ignoring an active zero update, reordering or dropping a repeated
operation gives explicit counterexamples. Exact UInt64 instances also retain
the sign-bit word and the least positive-subnormal word as distinct values;
these examples use xor and do not identify xor with floating-point addition.

Formalization: `ConvexDepth/SharedColumns/{Basic,Witness}.lean`, with 12 direct
checks in `scripts/SharedColumnAxioms.lean`. This proves conditional equality of
ordered executions and complete expansion, not IEEE implementation refinement,
a runtime speedup, convergence, target identification or additional accuracy.
Native controls must compare every logit/CE/update word and original work count,
charge simultaneous plan construction and storage, and measure actual runtime
under the existing RAM and both-disk limits. Frozen producers remain unchanged;
new original measurements require a new source freeze and independent repeat.

## CM72. Shared construction and independent certification preserve complete words

CM71 applies to arbitrary deterministic accumulator types and ordered update
payloads. Two further uses address work outside the ordinary forward logit
contraction. Neither changes the full convex domain or constrains its unknown
rows to remain uniform.

First consider anchored expansion as a function of exact words. At a known
label, its class receives the mass word and every other class the specified zero
word. At an unknown state, every class retains its actual stored word. Fix a
construction block and a set containing every known label in that block. Two
classes outside this set receive the same expanded word at every state if the
current unknown rows are class-uniform as words. This does not identify two
representations of a numerically equal value, such as signed zero words.

The head-construction event contains the operation index and this raw decoder
word. The index can include both frequency and state; the accumulator can hold
all four real and four imaginary words. Apply the original source products and
additions in their original frequency/state order. CM71 then proves equality
of the entire accumulator outside the known-label set. Compute those known
classes and one outside class, then expand the result before storing the full
logical head. Fully known blocks keep their original active-class support.
If any unknown row has unequal words, the complete original class loop remains
available. This conditional shortcut never discards that row or its classes.

Second, a prediction certificate may use multiple interpretations of the saved
coefficient events. An outward interval pass preserves inactive skips. An
independent dense coefficient pass may explicitly read zero at an inactive
class and perform its original exact addition. A deterministic recoding of
equal events remains equal within each pass, even when it maps an absent event
to an explicit payload. The reverse pass retains its own reversed schedule.
Each pass may share equal columns and expand its full class vector afterward.

A joint readout of those complete expanded vectors is unchanged. It can retain
the original residual and enclosure checks, exact integer comparison, every tied
raw maximizer, and ordinary CE over the entire class vocabulary. The theorem
does not require forward and reverse answers to agree; their discrepancy stays
available to the certificate. It does not replace exact dyadic checking with a
rounded argmax or delete equally scoring classes.

A concrete three-state, four-class example has two known labels, one unknown
row and a repeated operation. Paired accumulators give distinct exceptional
columns `(5405,5525)` and `(470,590)`, with shared columns `(400,520)`. A separate
forward/reverse witness deliberately produces different answers and preserves
both. Making the unknown row nonuniform gives a counterexample to the shortcut;
the full fallback still represents it exactly.

Formalization: `SharedColumns/{Construction,ConstructionWitness}.lean`, nine
direct checks in `scripts/SharedConstructionAxioms.lean`. The results prove
conditional ordered identities, not native IEEE refinement, speed, convergence
or target identification. The implementation must test actual word equality,
rebuild plans for changed heads, account for full class vectors and simultaneous
integer clones, and compare every certificate field and training trajectory to
the complete prior model. Bounded synthetic timings are separate from original
fits. Existing frozen producers stay fixed; new original results require a new
freeze, complete training seal, independent repeat and raw replay.

## CM73. Shared categorical interactions learned through ordinary answer loss

CM64's observed-label equations force every observed CE to depend only on
total confidence. CM71-CM72 preserve that model exactly and cannot repair its
identification limitation. Here the observations enter only ordinary CE.
Reuse CC1's complete joint tensor and MF3's input alignment, but train one
shared block `A : K x V x C -> R` on finite nonempty categorical alphabets:

```
A(k,v,c) >= 0,
sum_c A(k,v,c) = rho(k), independent of v,
sum_k rho(k) = 1,
z_A(q,D,c) = sum_k A(k, D(align(q)^(-1)(k)), c).
```

The routing mass is a marginal of A, not a separately multiplied learned
factor. All `|K||V||C|` coordinates are optimized jointly. The feasible set and
uniform initialization `A(k,v,c)=1/(|K||C|)` have no labels or sample-dependent
constraints. CC2 and MF3 prove convexity of this domain and ordinary CE in
every coordinate. A fixed positive logit scale preserves convexity by MF3;
the identification result below uses scale one and does not learn a scale.
At scale one total logit mass is one, so perfect argmax accuracy does not mean
zero CE. Contradictory labels do not make this domain empty; realizability is
a hypothesis of the identification theorem, not a feasibility requirement.

The input encoding is declared categorical slots and a visible query, with
one fixed query-dependent slot permutation. Interaction order is one: the
forward adds slot contributions. The alignment, categorical identities and
sharing across queries are explicit priors. This is a restricted subfamily of
the full query-indexed channel, and is not an unrestricted learned encoder or
a universal model of arbitrary multi-slot interactions. There is no parameter
per complete dictionary and no target program or target formula in the forward.

Suppose an unknown slot r and unknown categorical decoder g realize the actual
observed labels on aligned dictionaries E_s. Assume at least two classes and:

```
y_s = g(E_s(r)),
for every v, some observed s has E_s(r)=v,
for every k != r, some observed s,t have E_s(k)=E_t(k) and y_s!=y_t.
```

These are explicit observation and target-family conditions, with no premise
about successful learned predictions. The selected joint vertex at r with
decoder g is feasible and attains the ideal sample CE. For any feasible A with
no greater total loss, nonnegative sample loss gaps all vanish. CR3's strict
CE inequality therefore forces target mass one on every observed record.

For a wrong slot k, take its collision pair. Each class differs from at least
one of the two answers, so every coordinate of that shared value row is zero.
Value-independent routing mass makes the entire slot's mass zero at every
other value too. Nonnegativity then removes every coordinate of that slot.
Coverage of r determines every remaining decoder row. Thus every global CE
optimizer equals the selected vertex and has the correct strict raw argmax
on every aligned dictionary, including completely unobserved input combinations
and queries. This strengthens CC5 for this shared family by using one observed
collision per wrong slot instead of a separate observation for every wrong atom.

The nonconstant example has three Boolean slots, two classes and twelve joint
coordinates. Its four dictionaries are `000,011,101,110`, with labels
`true,true,false,false`, all at query false. The declared alignment is identity
at false and swaps the first two slots at true. Both value coverage and every
wrong-slot collision are proved. Dictionary `111` and query true are absent,
yet every empirical optimizer answers that new input correctly. Selecting the
wrong slot gives strictly larger observed CE at exactly the same total logit
mass. This is a change of relative predictions, not confidence-only learning.

The ambiguity control observes only `000 -> true` and `111 -> false`.
Every slot has both values and both answer labels occur. Selecting either of
the first two slots with negation is feasible and globally CE-optimal; their
strict answers on `100` are opposite. Their actual signed query-by-dictionary
operators, over both queries and all eight dictionaries, are both flat with
squared singular value eight. Flatness cannot select the missing answer.

For the actual complex implementation, fix a unitary value code U independently
of labels. Slot k's complete head is `H_k=U* A_k`; read its full real and
imaginary coefficients against the source row at the aligned categorical value.
Expanding every product gives exactly z_A. The actual complex CE is convex and
has the same identification theorem. Every source coordinate is bounded by one;
the complete per-class head budget is at most `2|V|`, while storage still holds
`2|K||V||C|` real coordinates. A four-point Fourier witness retains nonzero
imaginary source and head entries. CC6's rank/error bound applies to the actual
represented output operator, without making a target-identification assumption.

Formalization: `ConvexChannel/SharedLearning`, with direct checks in
`scripts/SharedLearningAxioms.lean`. Record indices retain duplicate observations.
No convergence of a native minibatch loop or floating-point refinement is
claimed. MF1's singleton-oracle failure still applies to unsuitable update
rules; MF2's stale-gradient memory alone is not a general convergence proof.
CF3-CF6 supply a possible finite forward-only consistency learner and persistent
minibatch scan under their explicit conditions. Any native implementation must
state its actual observation-dependent parameter updates, retain original
multiplicities and short tails, test relative prediction changes at fixed
confidence, and preserve contradictory-label failures. Compare the anchored
reference, a free decoder on fixed features and the shared learner before any
separately frozen original pilot. This theorem supplies no new catalog credit.

## CM74. Hidden-path supervision, joint repair, and shared-decoder obstruction

The requested interface is a constant object sequence followed by the next
object, with representation, interactions and output mapping all trained from
ordinary answer CE. A proof about fully observed internal paths is insufficient
for that objective. The existing `GPTMini.Convex.Structured.TensorStackTraining`
uses a complete `MixedConfiguration` as its target and explicitly excludes
output-only CE. The following controls identify the mathematical gap without
claiming a new proof about that entire older stack.

First fix one input and learn two scalar parameters a,b. An encoder chooses a
hidden bit h with probabilities sigmoid(a) and sigmoid(-a). A decoder keeps h
with probability sigmoid(b) and flips it otherwise. Every complete pair (h,y)
has positive probability, and the four probabilities sum to one. Its complete
NLL is a sum of two softplus terms, hence convex jointly in a,b. It is also
exactly ordinary CE on the four path categories. A target in that vocabulary
reveals h as well as the actual next object y.

With only y observed, actual output probabilities instead sum the two paths
ending at y. Their normalized log probabilities are the logits supplied to
the usual full-vocabulary log-sum-exp CE. At (a,b)=(log 3,log 3) and
(-log 3,-log 3), P(y=true)=5/8. At the midpoint (0,0), P(y=true)=1/2. The midpoint
CE is therefore strictly larger than the average endpoint CE. This proves
nonconvexity for this learned two-stage answer model, despite its convex
complete-path loss. It does not assert that affine logits are necessary or
rule out every possible nonlinear convex sequence family.

A constructive repair at one input learns a full joint table J(h,c), with

```
J(h,c) > 0,             sum_h sum_c J(h,c) = 1,
E_J(h) = sum_c J(h,c),  D_J(h,c) = J(h,c) / E_J(h),
P_J(c) = sum_h E_J(h) D_J(h,c).
```

The domain is convex and label independent. Neither E_J nor D_J is frozen;
they are conditional distributions coupled through the same learned joint
coordinates. Their actual products equal J(h,c), so P_J is its output marginal.
Every P_J(c) is positive and the full vocabulary normalizes. Consequently
ordinary CE on logits log(P_J(c)) equals -log(P_J(y)) and is jointly convex
in J on the declared domain. No internal-path labels enter this loss.

For a single input this is an exact parameterization of all positive stochastic
encoders E and arbitrary positive conditional decoder rows D: J=E D lies in
the joint domain, and recovering its row marginals and conditionals returns
E and D exactly. This larger family does not impose the earlier counterexample's
keep/flip tie. Strict positivity also means a finite optimum is not guaranteed
for separable data; no optimizer convergence theorem is asserted.

An inhabited two-state/two-class comparison starts from all J entries 1/4.
In false/true row and column order, another feasible point is
[[1/2,1/8],[1/4,1/8]]. The false-state encoder mass changes from 1/2 to 5/8,
its false-output decoder probability changes from 1/2 to 4/5, and the actual
false-output probability increases from 1/2 to 3/4. Ordinary answer CE strictly
decreases. This is a feasible change in both stages, not a proof that an
optimizer finds it or that the loss uniquely identifies latent parameters.

Sharing a learned decoder across distinct inputs is an additional constraint.
For two inputs, two hidden states and two outputs, consider joint tables

```
T(x,h,c) = E(x,h) D(h,c),
E(x,.) and D(h,.) strictly positive normalized rows.
```

At each fixed h, its input/output minor must vanish. Let both stage tables
have diagonal 3/4 and off-diagonal 1/4 at one endpoint; reverse those entries
in both stages at the other endpoint. Both are genuine shared-decoder models
and their actual output probabilities agree at every input. Their joint
midpoint has minor 1/16 at h=false, so no common decoder represents it. Yet
each input of that midpoint remains in the positive joint simplex. Thus this
exact shared-decoder image is nonconvex, even with all entries positive.
Independent joint tables per prefix would silently relax the sharing that is
needed for the proposed sequence model. The identical endpoint answers also
show why internal state labels are not identified by output supervision.

Formalization: `ConvexChannel/HiddenPath/{Basic,Boundary,Joint,Recovery,Witness,Sharing}`;
direct axiom checks: `scripts/HiddenPathAxioms.lean`.
These are exact finite controls with inhabited hypotheses, not a completed
causal sequence architecture, a compact complex implementation, or a native
learning result. The remaining construction must specify an executable shared
representation, prove the intended joint convexity without substituting
complete-path targets, and derive unseen-prefix predictions under explicit
conditions. Flat-spectrum bounds apply to its actual output operator; they
do not remove this parameter-sharing or identification obligation.

## CM75. Learned complex ReLU factors, activation cones, and exact pattern coverage

CM74 rules out transferring complete-path convexity to an arbitrary learned
hidden-state marginal. A different construction uses deterministic learned
ReLU representations and ordinary answer CE. This is an original extension,
not the fixed random gated activation of arXiv:2211.11052v1, §3.4, Theorem 3.
The distinction and the activation-domain restrictions below are essential.

For a vector u in a real normed space, set

```
a(u) = sqrt(norm(u)),       e(u) = u / a(u),
neuron(u,L) = a(u) * max(L(e(u)), 0),
```

using inverse zero equal to zero. Then a(u) e(u)=u, norm(e(u))=a(u), and the
actual neuron equals max(L(u),0) at every input functional L. Conversely, a
physical neuron with representation w and nonnegative amplitude a is exactly
represented by u=a w, on every possible input. Recovery fixes its scaling
gauge; it does not claim convexity in independently optimized a and w. The
balance uses the declared ambient norm. No original weight-decay penalty is
silently transferred to these coordinates.

For complex representations, use the actual componentwise activation
sigma(z)=max(Re z,0)+i max(Im z,0), and learned complex readout a(u)(1+i).
The full product is computed before taking its real part. Positive real
scaling commutes with sigma, giving the exact identity

```
a(u)(1+i) sigma(L(e(u))) = (1+i) sigma(L(u)).
```

Its real logit is max(Re L(u),0)-max(Im L(u),0). Thus imaginary representation
coordinates contribute to the actual answer logit. The readout phase is fixed
architecture, while its two components change with a(u). Its norm is sqrt(2)
times the amplitude used in the balance identity. This is a specified phased
complex ReLU family, not an equivalence to every independently parameterized
complex transformer.

The complete model has vectors u[j,c,s], where j is a hidden slot, c is an
original output object, and s is a positive or negative readout sign. Each
class logit sums the positive complex neuron products minus the negative ones.
All sequences use this same parameter table. Its usual full-vocabulary
softmax is positive and normalized; the target occurs only in ordinary
log-sum-exp CE. No hidden-path target or label-derived input is supplied.

For a declared input region R, impose one real and one imaginary sign mask
per slot and input. An active component requires its preactivation to be
nonnegative; an inactive component requires it to be nonpositive. Zero belongs
to either case. All constraints are linear in u and independent of answer
labels, so their intersection is convex. At an input in R, each actual ReLU
equals its indicated linear branch. The resulting complete complex logits
respect convex parameter combinations, and ordinary CE is jointly convex.
Every indexed minibatch sum has the same property; repeated records and short
tails retain their original indices. This proves neither a projection/update
algorithm nor convergence while changing masks.

The raw interface receives a list of immutable objects with length at most a
declared cap T. An input object at position i addresses a learned complex
coordinate (i,o); one extra coordinate is a learned bias. Inference reads and
sums those actual learned embeddings, applies sigma, multiplies the learned
readouts, and predicts over the same object type. Longer lists cannot inhabit
the input type, rather than being silently truncated. There are
4 |J| |O| (1+T |O|) real joint coordinates before optimizer buffers and recovered
factor caches. Recovering all factor norms costs a full parameter scan; once
cached, a prefix forward reads its visible coordinates in each of the
2 |J| |O| neurons. No arithmetic features or task-specific answer programs are
part of this interface.

An inhabited complex comparison uses two input reads, the two coordinate
projections. Joint points (4,i) and (i,9) have recovered amplitudes two and
three, and representations (2,i/2) and (i/3,3). A positive neuron for one
class with a zero competing logit gives that class's logits (4,-1) at the
first point and (-1,9) at the second. Both points satisfy the same component
cones. With next-object labels (false,true), relative preferences reverse
and both answer CEs strictly decrease. This demonstrates changed representation
direction and readout, not only a positive confidence rescaling. It is a comparison of
feasible points, not a measured or proved optimizer trajectory.

The exactness statement for independent factors is deliberately scoped.
Given physical representations w[j] and arbitrary signed real readouts
a[j,c] with the declared complex phase, use
u[j,c,+]=max(a[j,c],0)w[j] and u[j,c,-]=max(-a[j,c],0)w[j]. The complete complex
prediction is equal at every input. Recovery uses 2 |J| |C| class-specific
neurons; it enlarges the original low-width representation sharing. The
source does not rename that factorization constraint as a convex one.

Each physical representation defines an input-only sign mask. Keeping every
Boolean mask on a finite region X gives one common domain with slots
J times (X times Bool -> Bool). Put each lifted physical neuron in its actual
mask slot and retain every other slot at zero. This point is feasible, and
summing all slots reproduces the entire physical function, even outside X.
The full pattern index has 2^(2 |X|) elements, including redundant assignments.
Recovered width is 2 |J| |C| 2^(2 |X|). Even one binary64 complex coordinate
per pattern for twenty region inputs exceeds 110,000,000,000 bytes. This is
the cost of naive complete enumeration, not a lower bound against every
implicit or sparse exact algorithm. Sampling masks does not inherit the full
coverage theorem.

Taking X to be the entire finite universe of raw object prefixes of length
at most T gives the complete version: one full-pattern domain has convex
next-object CE at every admitted prefix, and every independently specified
physical model in the declared phased family has exactly the same predictions
at every bounded raw prefix after expansion. The real coordinate count becomes
4 |J| |O| (1+T |O|) 2^(2 |Prefix(O,T)|). The global statement retains this full
cost; it is not a training-only result presented as a practical full model.

Fixing a finite mask family still yields a fixed gated linear expression on
R, despite the exact learned-factor interpretation. It does not settle the
user's requirement for an efficiently trained unrestricted sequence path.
An explicit boundary example prevents overstating its guarantee: points
(1,0) and (0,1) satisfy the two observed positive-coordinate cones, but at the
missing difference read L(z)=z1-z2 their actual answer CE violates the midpoint
convexity inequality. Exact factor recovery is global; cone convexity is only
at inputs constrained by the domain. For convexity at every admitted raw prefix,
the constraints must cover that full region rather than only training inputs.

Formalization: `ConvexChannel/LearnedHinge`; direct checks in
`scripts/LearnedHingeAxioms.lean`. No new stress-test credit follows. The next
obligations are a resource-bounded method for the complete learned family,
actual Mini and forward-only Mini updates, unseen-prefix identification under
explicit observations, and bounds for the relevant actual operator. Flatness
does not supply missing labels or justify discarding patterns, coordinates,
or class components. Task 366 and the 99%-accuracy objective remain open.

## CM76. Exact component-pattern aggregation for the full complex sequence family

CM75's Cartesian product of real and imaginary patterns contains an exact
prediction redundancy. It can be removed without sampling patterns or changing
ordinary CE. This is a quotient of predictions on the declared input universe,
not a recovery map for the old individual parameters.

For real learned-neuron coordinates with a common sign pattern, ReLU equals
the indicated linear branch at every region input. Their vector sum therefore
has the same sum of actual neuron predictions there and remains in that cone.
Aggregate separately for each output class and positive/negative slot. Every
old vector enters its group's sum.

The complex raw input reads its real and imaginary coordinates using the same
real linear map L. This follows from the learned bias and position/object
lookups. An arbitrary real linear map into complex numbers need not have this
property; the extension does not assume it for such maps. For full vectors
p[j,c,s], let mR[j] and mI[j] be their component masks on a finite universe X.
For each scalar Boolean pattern m:X->Bool define, coordinatewise,

```
Re q[m,c,s,d] = sum(j with mR[j]=m, Re p[j,c,s,d]),
Im q[m,c,s,d] = sum(j with mI[j]=m, Im p[j,c,s,d]).
```

Both components of q obey pattern m. Before taking a real logit, the whole
complex forward is (1+i) times the complex number formed from the two signed
real-neuron sums. Each sum is preserved by its own aggregation, so the entire
complex prediction is equal at every input in X. Norms, recovered amplitudes
and representations are computed anew; CM75's exact factor identities retain
the actual complex products. All classes and both output signs remain present.
Every indexed minibatch CE is equal, including duplicates and short tails,
and ordinary CE remains jointly convex on the reduced domain. This does not
prove an optimizer trajectory or unrestricted updates across cone boundaries.

The reverse embedding puts q[m] in the old complex pattern whose real and
imaginary masks both equal m, and zero in every other pattern. For any nonempty
old hidden-slot index, put this point in one distinguished old slot and zero
in the others. It satisfies every old constraint and has the same full complex
function at every linear input read. Thus the two prediction families on X
are equal for nonempty old hidden indices. The empty old model only predicts
zero logits, so that qualification cannot be omitted.

For raw sequences take X to be every prefix of length at most T. The encoder
is unchanged. Every feasible CM75 complete point, hence every original phased
physical network covered by CM75, has the same complex prediction at every
admitted raw prefix after reduction. Every reduced point embeds back into the
old complete family. This is stronger than equality on observed examples.
If X were only the training inputs, forward aggregation would guarantee
equality only there; the theorem does not silently extend it outside X.

With K=|X| and d learned complex input coordinates, the original real parameter
count is 4 |J| |C| d 2^(2K), and the reduced count is 4 |C| d 2^K. The recovered
width is 2 |C| 2^K. All scalar patterns remain, including unrealizable ones;
old hidden-slot multiplicity disappears by exact sums. This quotient preserves
ordinary CE, not an arbitrary penalty on old separate factors. It also does
not prove binary64 summation-order identity. No frozen producer is changed.

For fifteen region inputs, two classes, seven complex input coordinates and
one old hidden slot, complete binary64 parameter storage falls from
481,036,337,152 to 14,680,064 bytes. These are finite input counts, not a
stress-cohort admission or a native peak-memory measurement. Optimizer state,
factor caches, cone constraints and evaluation buffers must still be charged.
Even after reduction, one complex binary64 coordinate per scalar pattern at
thirty-three inputs exceeds 110,000,000,000 bytes. Exponential growth remains.

A nonconstant witness uses two vectors, in false/true input-coordinate order,
(-2+7i,5-6i) and (-1-4i,2+3i). Both real masks are the input Boolean, while their
imaginary masks differ. Put both vectors in class true's positive slot and
retain zero vectors in the other class and negative slots. The full complex
class-true outputs before and after reduction are (-7+7i,4+10i); the competing
class has zero output. Relative preferences select different next objects at
the two inputs. This checks nonzero imaginary values and differing masks, not
a fitted learner or identification of missing answers.

Formalization: `ConvexChannel/LearnedHinge/PatternReduction`; direct checks in
`scripts/PatternReductionAxioms.lean`. Remaining work keeps CM75's full scope:
a resource-bounded complete sequence family, executable Mini and forward-only
Mini updates, identification from actual observations, and flat-spectrum
bounds on the actual operator. This removes one concrete storage obstruction;
task 366 and the 99%-accuracy objective remain open, with no new test credit.

## CM77. Missing-answer ambiguity inside the actual learned sequence family

The interface remains a sequence of immutable objects and one next object from
that same set. Parameters used to represent, combine and predict those objects
are learned. CM75–CM76's exact physical coverage and convex CE do not identify
an unobserved answer. This claim now has a constructive proof inside their
actual complex raw-sequence model, including perturbations of any fitted point.

Fix an object set O, cap T and a complete raw prefix v of length T. The learned
bias is 1-T and its position/object weight at (i,o) is one when o=v[i], zero
otherwise. The original raw lookup computes 1-T plus the number of matching
positions. At v this is one; every other admitted prefix, including shorter
ones, gives a nonpositive value. Its ReLU therefore isolates v exactly. The
comparison with v occurs in the proof of this permitted parameter choice;
it is not a task-specific program substituted into the model's forward.

Multiply the whole real vector by 2+i. Both real and imaginary preactivations
are rectified by the unchanged actual complex neuron. The recovered readout
phase 1+i makes its whole complex response 1+3i at v and zero elsewhere.
The output's real part is one. Arbitrary signed real readouts and exact CM76
coverage place this direction in the complete reduced domain, for any chosen
output class and any real amplitude r. No original class or pattern is removed.

The activation domain is a cone. Adding two feasible points preserves every
constraint, and the whole complex prediction adds on every admitted raw prefix.
Thus for any feasible p there is a feasible q with exactly the same complex
prediction everywhere except the chosen prefix/class, where it changes by
r(1+3i). The selected real logit changes by r. Recovered factors may change;
this is an exact forward identity, not a claim about preserving their norms.

If every observed input omits v, every observed logit and ordinary next-object
CE remains equal. Every indexed sum is equal, retaining duplicates and short
tails. For binary objects, choose r relative to p's two missing-prefix logits:
one feasible point then strictly prefers false, another strictly prefers true,
while both keep all of p's observed logits and loss. This holds even when p is
already fitted. It is an identification failure, not an optimizer-divergence
claim or a claim that a particular regularized learner must choose badly.

The hypotheses are inhabited with genuinely different observed answers:
[false,false] has next object false and [false,true] has next object true.
Two permitted neuron directions produce any common positive target margin r.
Each ordinary CE equals log(1+exp(-r)), strictly below log 2. Opposite answers
at the omitted [true,true] preserve that exact nonconstant fit. No hidden-path
labels are used. This witness does not assume a task rule for the missing label.

For the flatness connection, form an operator from actual raw forward calls:
three rows are prefixes [false,false], [false,true], [true,true], and the three
columns are their actual learned singleton neurons. Its complete complex
matrix is (1+3i)I. Retain all three columns but observe only the first two rows,
calling the resulting matrix A. Direct multiplication proves A A* = 10 I2.
For every complex coefficient vector w,

```
||A w||_2^2 + 10 |w[2]|^2 = 10 ||w||_2^2,
||A w||_2^2 <= 10 ||w||_2^2.
```

The retained direction (0,0,1) is nonzero, gives zero observed response and
changes the missing-prefix response by 1+3i. It is real and hence also an
allowed signed-readout direction, although the energy theorem holds on the
larger complex linear coefficient space. This is the observed response
operator at a specific actual learned-neuron configuration. It is not a
flatness theorem about every learned weight tensor or a training trajectory.
The positive observed-row spectrum is exactly flat while an unseen direction
remains; flatness supplies the displayed bound, not the missing label.

Formalization: `ConvexChannel/LearnedHinge/Unseen`; direct axioms command:
`LAKE_ARTIFACT_CACHE=false lake env lean scripts/UnseenHingeAxioms.lean`.
There are no new unproved inputs. Task 366 remains open. A positive repair
needs an explicit relation between observed and unseen answers, as well as a
resource-bounded full model and actual Mini and forward-only Mini updates.
These results add no empirical pass or accuracy credit and change no producer.

## CM78. Conditional unseen transfer through two shared learned complex passes

CM77 leaves a genuine identification gap: independently changing a missing
prefix is allowed by the complete learned family. One positive repair is an
explicit relation between observed and unseen outputs. Here the relation is
an involutive renaming sigma of the original immutable object set. It is an
architecture assumption whose applicability must be checked for the task;
flatness does not establish it or select a suitable renaming.

For an original sequence x, map sigma over every position to obtain sigma x.
Length, order and the object type remain unchanged. With the same learned
parameter point p in both calls, the actual full complex forward is

```
S_p(x,c) = (F_p(x,c) + F_p(sigma x, sigma c)) / 2.
z_p(x,c) = Re S_p(x,c).
```

Both F calls perform the original learned embedding lookup, nonlinear complex
activation and recovered learned readout. The alignment and factor 1/2 are
fixed architecture operations. Neither the target nor an answer rule enters
inference. The loss is the usual full-vocabulary cross-entropy on z; all output
objects retain positive normalized softmax probabilities. The mathematical
predictor returns a maximizing original object, rather than returning a vector
as its external output. Classical choice supplies a mathematical maximum;
native inference still needs a documented tie order and a full class scan.
Strict-winner results are independent of that tie rule and of the fixed witness
object used only to establish a nonempty vocabulary.

Because sigma is an involution, swapping the two calls proves
S_p(sigma x,sigma c)=S_p(x,c) for every p. This is an actual restriction on a
fixed trained function. It is stronger than the usual observation that one
can rename a model's inputs, outputs and parameters simultaneously. If an
original base function already obeys this relation, the average preserves it
exactly. Arbitrary old functions are changed; this construction is not called
an exact replacement of the unrestricted CM75–CM76 prediction family.

Every physical paired network in the declared phased two-layer family is
represented exactly by the CM76 joint lift. Its whole complex prediction is
preserved. The earlier class/sign-specific widening and fixed complex phase
remain; no equivalence with arbitrary free complex transformers is claimed.
Both passes share the complete parameter point, rather than acquiring an
independent table per input or a fixed-feature classifier.

The input-only cone domain includes every admitted bounded prefix, so it
constrains both x and sigma x. Convex parameter combinations pass through both
actual logit computations. The aligned average therefore has jointly convex
ordinary answer CE on that same domain. Every indexed minibatch sum is convex,
retaining duplicates, all output classes and short tails. This is a sufficient
construction, not an assertion that affine logits are necessary for convex CE,
and not a proof of a particular optimizer's feasible updates or convergence.

All classes are permuted bijectively, so ordinary CE is exactly equal at
(x,y) and (sigma x,sigma y). If observed CE at (x,y) is below log 2, every
competitor has a strictly lower logit, and the model returns sigma y at
sigma x. The latter prefix need not be in the observations. If two models
agree on all logits at x, they agree on all logits at sigma x; the independent
CM77 missing-prefix shift is no longer possible within that pair.
Correctness for an actual task requires the explicit data condition
answer(sigma x)=sigma(answer(x)). The answer function occurs only in this
correctness hypothesis and never in the model forward. No statement identifies
prefixes outside the observed renaming pairs or infers this task condition
from spectrum, convexity or empirical fit alone.

There is a necessary limitation. If sigma x=x while sigma y differs from y,
then those two logits are tied and CE for y is at least log 2. Applying a
fixed-point-free object swap to an empty input cannot fit a unique moved
answer. Control objects therefore need appropriate fixed identities when
the data require them. This restriction is checked rather than silently
omitting the empty input, output class or conflicted record.

An inhabited nonconstant example uses O=Option Bool and cap T=1. `none` is an
ordinary fixed control object; `some false` and `some true` swap. The observed
sequences are [some false] with next object some false, and [none] with next
object none. The evaluation sequence [some true] differs from both observations.
Two allowed learned directions supply a base logit four at the first observed
prefix/class and two at the control prefix/class, zero elsewhere. Actual
paired averaging gives target margin two on both observed records. Each
three-class CE is log(1+2 exp(-2)), below log 2 with both competitors retained.
The mathematical predictor consequently returns some true at the unseen
sequence. A head-object rule with default none inhabits the required data
symmetry on every admitted input, including empty inputs; that rule is not
executed by the network. This is a feasible point and a transfer theorem,
not a learned trajectory or a stress-test result.

The architecture uses two full forwards with shared parameters, plus an
O(T) object-renaming pass and O(|O|) output alignment and average. It adds no
trainable coordinates. With K=|Prefix(O,T)| the complete real parameter count
remains 4 |O| (1+T|O|) 2^K. All scalar patterns, original classes and component
coordinates remain. The finite existence proof does not produce a compact
complete optimizer; recovered-factor caches, cone constraints, optimizer state
and both pass buffers still need native resource accounting. The bound from
CM77 remains a bound on its stated actual observed operator, not a claim that
symmetrization preserves every matrix's flat spectrum.

Formalization: `ConvexChannel/LearnedHinge/Renaming`; direct check:
`LAKE_ARTIFACT_CACHE=false lake env lean scripts/RenamingHingeAxioms.lean`.
The positive condition has now been proved on an actual trainable complex
sequence path with ordinary CE and an original-object output. Task 366 remains
open for the complete efficient family and executable Mini and forward-only
Mini. Task symmetry is not assumed for all 165 stress cases. No accuracy
credit follows from these proofs, and the 99% raw/exact target remains.

## CM79. Checkable projection certificates for complete learned-path updates

A usable Mini or forward-only Mini update needs to remain in the declared
activation domain. Convexity alone does not make an arbitrary update feasible.
This extension gives finite arithmetic conditions that certify a projection
onto that domain. It is a proof about a supplied certificate, not an assumed
projection oracle, solver completeness or a completed training trajectory.

Let a[r] be every oriented row of a real activation cone, v the proposed
parameter vector and u a candidate updated vector. Retain every coordinate
and every constraint, including redundant rows. The certificate supplies
one multiplier lambda[r] per row and checks four conditions:

```
lambda[r] >= 0,
a[r] dot u >= 0,
lambda[r] * (a[r] dot u) = 0,
u[d] = v[d] + sum(r, lambda[r] * a[r,d]).
```

These are predicates of the supplied arrays; they do not encode optimality
as a structure field or presumed conclusion. For every other feasible z,
stationarity and complementarity give (u-v) dot (z-u) >= 0. The exact squared
distance identity then proves

```
||v-u||_2^2 + ||u-z||_2^2 <= ||v-z||_2^2.
```

In particular u is a globally nearest feasible point. Checking finitely many
arithmetic conditions therefore certifies a statement over every feasible
competitor. A nonidentity example projects proposal -1 to 0 in the nonnegative
half-line with multiplier one. A feasible point has an identity certificate
with all multipliers zero. These statements do not require that the active
rows be linearly independent or that redundant inequalities be discarded.

For the actual raw model, the bias row coefficient is one. At coordinate
(i,o) it is one exactly when the immutable input has object o at position i,
zero otherwise. Its inner product with learned real weights is proved equal
to the unchanged raw input computation. Multiplying the row by the mask sign
gives exactly the original activation inequality. The proof also connects
both components of the original complex input to their real coordinate reads.
These rows describe constraint geometry; they are not task features supplied
to a replacement inference program.

A certificate for every slot, output object, readout sign and real/imaginary
component proves that the complete updated complex model belongs to the old
domain on every admitted raw prefix. Summing the component distance inequalities
proves the strong projection inequality for the whole array against any other
feasible complex model. Nothing is projected out of the complex forward, and
no output class, pattern, prefix or parameter coordinate is omitted. This is
Euclidean distance in joint coordinates, not preservation of arbitrary
penalties on the recovered factors or their representation-recovery norm.

For a proposal v=x-eta*g from a feasible x, the checked projection u obeys
eta * g dot (u-x) <= -||u-x||_2^2. If the actual chosen loss satisfies the
explicit step bound

```
F(u) <= F(x) + g dot (u-x) + (L/2) ||u-x||_2^2,
```

and eta>0, eta*L<=1, then eta*F(u)<=eta*F(x)-||u-x||_2^2/2. The upper bound is
an explicit hypothesis. CE convexity or flatness alone is not presented as
its proof, nor is an arbitrary forward-query direction called a gradient.
The feasibility and projection certificates apply to proposals from either
update mode independently of how those proposals are formed.

Formalization: `ConvexChannel/LearnedHinge/Projection`; direct command:
`LAKE_ARTIFACT_CACHE=false lake env lean scripts/ProjectionHingeAxioms.lean`.
The next native component should verify these conditions exactly over rational
arrays, refuse failed or over-budget certificates and retain the complete raw
domain. Searching active sets may be exponential; accepting a checked result
does not prove that a search implementation will always find one. No native
training, convergence, new stress coverage or full-suite resource admission is
claimed by this proof. Task 366 and the original 99% target remain open.

## CM80. Complete feasible directions for forward-only queries

CM79 certifies an update after a proposal is supplied. It does not justify
central finite differences outside an activation cone. At a boundary, even
individual coordinate perturbations may leave the domain on which the actual
learned complex forward has convex CE. Some complete patterns have cones with
empty ordinary interior. Discarding these patterns or treating a failed
coordinate probe as proof that its coordinate is irrelevant is unjustified.

There is an exact finite certificate construction for this geometry. For each
original oriented row a[r], obtain a CM79 certificate for y[r], its projection
onto the whole cone C. Stationarity and complementarity imply

```
(y[r] - a[r]) dot y[r] = 0,
a[r] dot y[r] = ||y[r]||_2^2.
```

If y[r]=0, the variational inequality gives a[r] dot z<=0 for every feasible z;
the original constraint gives the reverse inequality. Thus this row vanishes
on all of C. Conversely a row vanishing on C vanishes on its own projection,
so the energy identity forces that projection to be zero. This detects
universal equality from exact certificates, rather than from sampled points.
Every original row, multiplier and coordinate remains present.

Set c=sum(r,y[r]), including zero projected normals. Feasibility is closed
under this sum. For each row its response at c is a sum of nonnegative terms.
A zero sum forces its own projected response to zero, hence its projection
to zero. Consequently

```
a[r] dot c = 0  iff  forall z in C, a[r] dot z = 0.
```

The centre has positive slack exactly on the rows that can have positive
slack anywhere in C. This is a proved statement about the constructed point;
strict feasibility is not assumed as a structure field or new premise.

For any feasible point p and direction v, checking every bound

```
eta >= 0,
eta * abs(a[r] dot v) <= a[r] dot p
```

certifies that both p+eta*v and p-eta*v lie in C. If each zero slack has zero
directional response, a positive common eta exists: start with radius one,
keep it at zero-slack rows, and take its minimum with
slack/(abs(response)+1) at every positive-slack row. This uses the entire
finite constraint collection. At the certified centre such two-sided queries
exist exactly for directions that vanish on the universally forced rows.

An inhabited degenerate example has the two opposed rows (1,-1),(-1,1).
Both projected normals and their centre are zero. Nevertheless v=(1,1)
admits positive two-sided queries: both coordinates move together on the
diagonal. This illustrates why independent coordinate probes must not freeze
all coordinates of a cone with empty ordinary interior. No original constraint
or model parameter is deleted.

After a projected learning step, the current point may again lie on a
boundary. For 0<delta<=1, the complete mixture
p_delta=(1-delta)*p+delta*c stays feasible and has positive slack on every
non-forced row. Every direction respecting the universally forced equalities
then admits two-sided queries at p_delta. Mixing changes the point, and the
proof records both costs explicitly:

```
||p-p_delta||_2^2 = delta^2 * ||p-c||_2^2,
F(p_delta) <= (1-delta)*F(p) + delta*F(c)   when F is convex on C.
```

The objective can increase. No unchanged-trajectory or automatic-descent claim
is made. The actual ordinary CE or indexed Mini CE can supply the convexity
premise on its established full model domain. These geometric results neither
turn a finite difference into an exact gradient nor establish its error bound,
the accuracy of floating complex inference or optimizer convergence.

Formalization: `ConvexChannel/LearnedHinge/Projection/{Geometry,Centre,Queries,Mixture}`;
direct audit: `LAKE_ARTIFACT_CACHE=false lake env lean scripts/QueryGeometryAxioms.lean`.
Native construction must retain and verify the normal-projection certificates,
use all zero-slack equations when constructing query directions, check every
query against the complete domain and charge the work and storage before
allocation. Search failure remains an explicit refusal. This does not resolve
the exponential all-pattern cost or earn any stress accuracy. Actual Mini and
forward-only Mini training, independent producer/replay evidence and the 99%
raw/exact target on all 165 cases remain open under task 366.

## CM81. Checked completeness of the forward-query basis

CM80 identifies the admissible direction space, but an arbitrary native list
of feasible directions need not cover it. To avoid silently restricting
forward-only training, the next certificate proves that a supplied basis spans
the entire kernel of the forced equality rows. The certificate contains finite
coordinate identities, not a presumed spanning or optimizer conclusion.

Let E retain every original row index. It equals the original normal at a
centre-zero row and the zero row elsewhere. This auxiliary equality matrix
does not replace the original cone: all original inequalities remain in every
projection and query-radius check. A proposed basis b[j], selected free
coordinate f[j], and recovery array A[d,r] must satisfy

```
E[r] dot b[j] = 0                         for every j,r,
b[j,f[k]] = [j=k]                        for every j,k,
e[d] - sum(j, b[j,d]*e[f[j]]) = sum(r, A[d,r]*E[r])  for every d.
```

Here e[d] is the original coordinate row. All coordinates of each identity
are checked. Zero or dependent original rows are retained; the certificate
does not require an independent row subset. The basis may be empty when the
full equality kernel is zero.

Dotting the last identity with an arbitrary v gives

```
v[d] - sum(j, b[j,d]*v[f[j]]) = sum(r, A[d,r]*(E[r] dot v)).
```

Therefore every v in the equality kernel is exactly
sum(j,v[f[j]]*b[j]). Conversely every basis combination lies in that kernel
by the first certificate condition. The second condition recovers each
coefficient from its free coordinate, proving uniqueness. These conclusions
range over all vectors, not just a finite collection of numerical probes.

The actual E constructed from CM80's certified centre has a kernel equal to
the directions vanishing on every universally forced original row. Combining
the two proofs shows that the checked basis represents exactly the directions
admitting positive two-sided queries at the centre. Every basis combination
also admits feasible queries after any positive centre mixture of a feasible
iterate. No original model coordinate, activation pattern or inequality is
removed by using coordinates for this direction space.

The opposed diagonal example is inhabited by b=(1,1), one free coordinate,
and recovery of the other coordinate using the retained opposite normal.
All basis and actual-query theorems have instantiated examples. Equal pairings
of two proposed gradient vectors against every certified basis direction imply
equal work against every kernel direction. The example (2,0) versus (1,1)
shows that the vectors themselves need not be equal. This statement has an
explicit exact-pairing hypothesis; finite differences do not establish it
automatically. A nonorthogonal basis also does not justify treating its raw
directional values as Euclidean gradient coefficients.

Formalization: `ConvexChannel/LearnedHinge/Projection/KernelBasis`;
direct audit: `LAKE_ARTIFACT_CACHE=false lake env lean scripts/QueryBasisAxioms.lean`.
A native elimination routine can retain its full row-operation matrix and
produce this finite certificate, which an independent exact checker must verify.
Its generation, coefficient growth, complete storage and operation budgets
remain implementation obligations. The result is not a verified Rust solver,
finite-difference error theorem, convergence result or stress accuracy claim.
Task 366, both required minibatch modes and all 165 cases remain open.

## CM82. Actual complex CE derivatives and complete minibatch directions

CM79--81 certify feasible projection and complete query directions. They do
not by themselves identify the derivative of the actual model loss. This
extension derives that derivative from ordinary full-vocabulary next-object
cross-entropy, through the complete learned complex computation of CM75 and
both shared passes of CM78. Objects and their sequence order remain immutable;
only the learned joint parameter array is perturbed. No target appears in the
forward map, activation constraints or input representation.

For a fixed complete activation domain, write L_x(v) for the difference of
the real and imaginary gated linear forwards applied to a joint direction v.
The equality of the actual recovered complex logits with this expression is
derived on the original domain. In particular, for feasible p and q,

```
z_x(q) - z_x(p) = L_x(q-p).
```

The direction v need not itself be a feasible model. With the actual full-class
residual g[c] = softmax(z_x(p))[c] - [c=y], ordinary CE gives

```
CE_x,y(q) >= CE_x,y(p) + sum(c, g[c] * L_x(q-p)[c]).
```

This supporting inequality holds at boundary points as well. If p and both
p +/- r*v are feasible for r>0, convexity of the unchanged domain supplies
every intermediate query with |t|<=r. On this whole neighbourhood the actual
loss agrees with CE(z_x(p)+t*L_x(v)). Its derivative at t=0 is therefore
sum(c,g[c]*L_x(v)[c]), using the previously proved ordinary CE derivative.
Two feasible isolated probes alone are not used as a differentiability premise.

For raw object sequences, let x[d] be the exact bias/position/object read row
from CM79. The full real-coordinate pullback is

```
G[j,c,s,k,d] = g[c] * sign(s) * sign(k) * gated(mask[j,x,k], x[d]),
sign(true)=1, sign(false)=-1.
```

Here s selects the readout sign, and k selects the real or imaginary component.
The minus sign for the imaginary contribution follows from the actual complex
readout. Every slot, original output object, readout sign, component and input
coordinate is retained. Its pairing with v, summed over all those indices, is
proved equal to the actual CE slope. The raw row describes where an immutable
object reads a learned value; it does not supply fixed task features or freeze
the learned representation. The formula is not claimed to differentiate an
outside-domain unconstrained factorization.

For the shared prediction, define

```
z_pair[c] = (z_x[c] + z_(sigma x)[sigma(c)]) / 2,
g[c] = softmax(z_pair)[c] - [c=y].
```

The first full parameter pullback uses g/2, and the second uses
g[sigma.inverse(c)]/2 at the renamed prefix. Both pullbacks update the same
array. A finite sum reindexing proves that their combined work is the derivative
of the single CE on z_pair; it is not the sum of two separately supervised
losses. This derivative and supporting inequality require no involutivity.
The separate unseen-transfer theorem still has CM78's explicit involutive
renaming and task-equivariance assumptions.

Nonempty minibatches are indexed by Fin(N+1). Their actual mean loss and
complete update array use the same factor 1/(N+1) and the same full indexed
sum. Supporting inequalities and feasible-query derivatives are proved for
both the base and shared-pass mean losses. Equal records are not merged,
and the actual size of a short batch determines its normalization. The
native learner can form a full proposal from this array and apply the already
certified whole-model projection. Forward-only Mini instead evaluates the
actual batch CE at the certified queries and must account for its finite-step
and numerical estimation error; the nonzero-step difference is not asserted
equal to the exact derivative.

Inhabited controls use a learned complex bias 3+3i, zero at the other retained
position/object addresses, in the positive neuron for one of two objects.
Its two actual logits are initially zero. Adding a real bias direction changes
only that object's logit; both radius-one endpoint models are feasible for
every admitted prefix. The true-target CE slope is -1/2 and the false-target
slope is +1/2. An indexed three-record batch with targets true,false,true has
mean slope -1/6. Deduplicating its repeated record would incorrectly give zero.
Every theorem with conditional premises has an instantiated example; a
one-record batch also instantiates the short-tail case. These are proof
controls, not trained benchmark outcomes.

Formalization: `ConvexChannel/LearnedHinge/Learning`; direct audit:
`LAKE_ARTIFACT_CACHE=false lake env lean scripts/LearningHingeAxioms.lean`.
This closes the symbolic loss-to-update link needed for native Mini and
forward-only Mini. It does not prove finite-difference error bounds, a global
optimizer rate, numerical correctness of Rust, arbitrary unseen-label
identification, tractable complete pattern enumeration, or 99% stress accuracy.
Task 366 and the all-165 target remain open.

## CM83. Certified replacement of one learned complex component

Forward-only Mini queries change one full real or imaginary vector in one
slot, original object class and readout sign. Reconstructing and checking
every unchanged vector for each query repeats work. This extension proves
the exact local replacement that an implementation may use while retaining
the original model, every activation pattern and every constraint.

Write `project(k,w)` for the real component when k is true, and the imaginary
component otherwise. The replacement `component(k,u,w)` combines the complete
new real vector u with the unchanged opposite component of w. For either q,

```
project(q, component(k,u,w)) = if q=k then u else project(q,w).
```

Let p satisfy the full raw-prefix complex domain, and replace only its
slot j, object c, sign s component k. Check u against every original oriented
row for j,k, including dependent and zero rows. At the chosen cell these
checks are precisely the original complex activation inequalities, by CM79.
At every other cell the corresponding real vector is unchanged. The whole
replacement model therefore satisfies the same full domain. This argument
does not weaken the domain to observed prefixes or sampled inequalities.

The complete changed neuron must be recovered again. Its norm determines
both representation and readout; changing its real component can change
the recovered imaginary representation as well. Retaining just the old
opposite factor would not implement the joint parameterization. Every other
whole complex vector, its recovered factors and its actual neuron output
are unchanged. The original full complex forward is consequently equal to
the original per-slot sum with just that whole neuron substituted. All class
outputs and both signed terms remain evaluated.

An additional complex identity gives the change in each object's score:

```
forward(replace(p,j,c,s,k,u), x)[b] = forward(p,x)[b]
  + [b=c] * sign(s) * (neuron(component(k,u,p[j,c,s]),x) - neuron(p[j,c,s],x)).
```

Here sign(true)=1 and sign(false)=-1. This identity is over exact complex
numbers. A floating implementation should retain the original summation
order rather than infer that reassociation preserves binary64 results.
For the paired model, use the same replacement in both complete passes
before the original alignment and ordinary full-vocabulary CE.

Nonzero witnesses start from the zero full model with all-positive masks
and replace the chosen object's bias by one. A real replacement gives
complex output 1+i, an imaginary replacement gives -1+i, and a negative
readout sign negates the result. Every admitted prefix satisfies the full
cone, and the other original object retains zero output. Both conditional
theorems have explicit inhabited examples.

Formalization: `ConvexChannel/LearnedHinge/Replacement`; direct audit:
`LAKE_ARTIFACT_CACHE=false lake env lean scripts/ReplacementHingeAxioms.lean`.
This proves exact local replacement, not a Rust refinement or a binary64
error theorem. The native path must independently check dimensions, indices,
arithmetic limits and every selected row, reconstruct the entire changed
neuron and compare every output against full reconstruction. Existing
frozen producers remain unchanged. The exponential pattern cost, general
unseen-answer conditions and all-165 accuracy target remain open.

## CM84. Full-word capacity and limits of exact complete-family compression

CM83 removes repeated reconstruction work without changing the complete
learned complex family. Its remaining cost cannot be understood solely by
counting redundant activation masks. This extension identifies a large class
of functions already present in that family and derives two precise limits
on representations that preserve the entire class. Neither limit is a lower
bound for one structured task or for the requested finite stress suite.

Let W be the full words `Fin(T) -> O`, so `|W|=|O|^T`, and turn each word into
the original immutable prefix by `List.ofFn`. CM77 constructs an actual learned
bias and position/object weight vector for each needle v. The unchanged raw
lookup, complex activation and recovered readout give whole complex response
1+3i at v and zero at every other admitted prefix. These are permitted weight
settings, not a comparison program inserted into inference.

For any real table `a:W x O -> R`, use the needle representations and signed
class readouts a in the original physical model. Its full-word outputs are

```
factoredForward(weights,a,word)[c] = a[word,c] * (1+3i).
```

CM76's exact coverage theorem places this physical network in the same
complete reduced convex domain, which still enforces every bounded prefix.
The real logits therefore equal a at every full word. In particular, for
any two distinct original objects, every real table of their logit differences
is attainable. Setting the target logit to one and every other class to zero
also strictly realizes every possible full-word next-object label table.
This construction describes the existing model's expressivity. Supplying its
needle weights as fixed features, or training an independent table, would not
meet the user's all-trainable-path architecture requirement.

The response matrix A has rows for all full words and columns for all these
actual learned neurons. Its entries, whole complex multiplication and energy
are proved directly from the raw forward:

```
A[word,needle] = (1+3i) * [word=needle],
A A* = 10 I,
A v = (1+3i) v,
sum(word, |(A v)[word]|^2) = 10 * sum(word, |v[word]|^2).
```

Thus this configuration has a perfectly flat positive Gram spectrum at every
dimension |O|^T. The norm identity is the bound supplied by flatness. It does
not make the dimension small, identify labels, or characterize a training
trajectory. CM77 still supplies the observed-row blind-direction counterpart.

First consider an alternative finite-dimensional real parameter space V and
a fixed affine full-word margin readout `L(v)+b`, with L linear. Suppose every
feasible point of the current family has the same selected-object contrast
under some v. The encoding from the old point to v need not be linear,
continuous, unique, or supplied by an algorithm. For any desired table f,
interpolate f+b in the actual old model, then apply the exact-coverage premise.
Its representing v satisfies `L(v)+b=f+b`, hence `L(v)=f`. L is surjective,
and finite-dimensional rank gives

```
|O|^T <= real_dimension(V).
```

For an actual real-coordinate array `D -> R`, this is `|O|^T <= |D|`. It applies
to preserving relative predictions, rather than counting a common softmax
shift as useful information. It does not prohibit more compact representations
of restricted tasks, nor establish a lower bound for arbitrary nonlinear
readouts. CM74's nonlinear convex-CE examples remain valid: affine logits are
not required by convexity of ordinary cross-entropy.

Second, allow any fixed forward score computation whose entire varying model
code is `D -> Fin(2^b)`. The decoder and architecture are fixed before selecting
a label table; inference may be nonlinear and may use exact arithmetic after
decoding. Preserving every strict classification in the current family implies
universal classification of W. Applying the existing R1 information bound gives

```
|O|^T <= b*|D|                when |O| >= 2.
```

Below that budget, there exists one fixed feasible point of the current learned
family and one label table it strictly classifies, for which every available
code fails to preserve that entire strict classification. The counterexample
point and table are quantified before the competing code. This is a stored
information bound, not a bound on peak resident RAM; disk-backed representations
must count their stored information but are not excluded by a RAM-only cap.
The result makes no optimizer-rate or task-specific accuracy assertion.

Every conditional theorem has an inhabited example. Full-dimensional identity
readouts with a nonconstant offset witness the affine premises. A four-word,
four-bit lookup witnesses the information bound at equality; a one-bit code
with two-word-position inputs witnesses the strict-budget counterexample.
These are proof controls, not proposed learners or new benchmark passes.

Formalization: `ConvexChannel/LearnedHinge/Interpolation`; direct audit:
`LAKE_ARTIFACT_CACHE=false lake env lean scripts/InterpolationHingeAxioms.lean`.
The results explain why eliminating unused masks alone does not justify a
compact exact replacement of the whole current family. A next architecture
may impose task-relevant sharing or other explicit restrictions, with their
loss of expressivity, dimensions and execution costs stated and proved. The
immutable object-sequence interface, all-trainable-path requirement, ordinary
CE, both required minibatch modes and all-165 target remain unchanged. No
producer or empirical coverage changes in this extension.

## CM85. Learned quadratic sequence factors with polynomial storage and a depth limit

CM84 prevents an exact compact replacement of the entire complete hinge family.
Here is a different, explicitly restricted family. An input is still a list
of immutable objects from a finite set O, of length at most T, and the output
is one original object from O. Objects address learned weights; their identities
are not arithmetic values or supplied numerical features. Put

```
D = Option (Fin T x O),                 d = 1 + T |O|,
p(c,s) in complex Hermitian PSD(D),     c in O, s in {positive,negative}.
```

Every full matrix is trained jointly. The domain is the product of these PSD
cones and has no labels, observed-prefix constraints, activation masks, fixed
rank, or independent parameter for a complete prefix. Both branches of every
class remain present. The exact zero state is feasible.

For one joint matrix M, recover the positive matrix square root W and the
balanced factors

```
W = sqrt(M),       a = sqrt(||W||_entrywise_max),       E = W / a,
r_h(tokens) = E[h,bias] + sum_visible_positions_i E[h,(i,object_i)],
B_M(tokens) = a^2 (1+i) sum_h |r_h(tokens)|^2.
```

At zero the inverse scale is zero. CM75's recovery identity gives a E = W
and ||E|| = a; here the activation is degree two, so the learned readout is
a^2(1+i). Its phase is fixed architecture. Both real and imaginary source
coordinates enter each squared magnitude. Inference performs the learned raw
lookup, complete complex summation, activation and complex readout. It does
not replace them with a joint quadratic expression in its definition.

For any complex input vector x, `energy_gram`, `network_eq` and `factor_gram`
prove

```
a^2 sum_h |(E x)_h|^2 = sum_h |(W x)_h|^2 = Re(x* M x),
W* W = M.
```

`inputRow_read` identifies the raw lookup with the bias/visible-address row
only for this proof. Signed real logits are
`Re(B_p(c,+)(tokens) - B_p(c,-)(tokens))`. Their usual full-vocabulary softmax
is positive and normalized. `nextObject` returns an original maximizing
object, and every strict winner is returned regardless of tie selection.
Its fixed vocabulary witness establishes nonemptiness and is not a target.
Only ordinary CE receives the observed next object. A native predictor must
specify a deterministic tie order and retain every output class.

The actual logits respect joint parameter combinations on the entire domain
at every admitted prefix. Consequently ordinary answer CE is jointly convex,
as is every indexed mean minibatch, retaining repetitions and short tails.
This is convexity in the coupled matrices, not independent convexity in W, E
and a. No claim that affine logits are necessary for convex CE is used.

The physical coverage is exact for finite squared-response networks. Given
any hidden width H, complex weights V[h,d], and nonnegative independent physical
readouts b[h], absorb each sqrt(b[h]) into its full row. The resulting Gram
matrix is PSD, and its recovered network has exactly the original weighted
sum of squared responses on every input. Signed outputs use the two branches.
Conversely each PSD point recovers an actual physical factor of width d.
This equivalence concerns functions with unrestricted physical width; it does
not preserve a smaller fixed-width bottleneck, a fixed rank, or one hidden
representation tied across every output class. The recovered branches have
class-specific factors shared across all input sequences. Their total width
is 2 |O| d. A native matrix-square-root implementation, numerical residual
bounds and PSD update certification remain required; the exact Lean square
root is not a completed native solver.

The full dense storage count, retaining Hermitian redundancy, is

```
4 |O| (1 + T |O|)^2 real coordinates.
```

This charges both components, signs and all original classes, before factors,
optimizer state and solver workspace. It is polynomial in the declared cap
and vocabulary, not a guarantee of fitting the 110 GB aggregate limit for
every stress case. Given cached recovered factors, each prefix reads its
visible addresses in every retained hidden row. Factor recovery must still
process the complete matrices; no low-rank shortcut or runtime claim follows
from the coordinate count.

A nonconstant raw witness uses O=Bool and T=1. On the two object coordinates
(with the bias row/column zero), positive physical factors are

```
W0 = [[4, i],[-i,1]],       W1 = [[1,i],[-i,9]].
```

Each is the Gram of the row (1,i) plus a nonnegative diagonal, hence PSD; its
joint point is W*W. Their exact entrywise norms are four and nine, so learned
amplitudes are two and three. The first diagonal representation coordinate
changes from 2 to 1/3, while the imaginary coordinate changes from i/2 to i/3;
the representation directions are therefore different. The positive physical
energy at [false] is 17; removing imaginary weights makes it 16, proving that
the complex computation is used. Class false has zero logits. Class true uses
the positive point and negative matrix 4I, giving logits (9,-6) at W0 and
(-6,74) at W1 on [false],[true]. Both relative preferences reverse. The second
point returns the original input objects and strictly lowers both ordinary
answer CEs. These feasible comparisons are not trained outcomes.

The expressivity restriction is proved, not left implicit. Every quadratic
form has zero third mixed difference. The raw one-hot address row for three
Boolean objects is affine in the three bits, so the actual recovered logits
inherit that identity. `parity_not_realizable` proves that no feasible point
strictly classifies all eight three-bit parity prefixes. This is a concrete
function lost relative to the complete hinge family. It is not a conclusion
about one MQAR split or a proof against a deeper architecture. Simply adding
unrestricted independently trained layers does not inherit this CE proof.

Finally, expand the actual complex class-by-prefix output diagnostically
through every ordered address pair. `output_factorization` proves this exact
identity for the recovered complex forward. R3 then bounds its error to a
flat target of squared singular value s>0 by

```
s (min(|O|, |X|) - d^2) <= ||target - actual_complex_output||_F^2.
```

Exact representation of a nonzero flat target requires the corresponding
width. When the lower bound is nonpositive it gives no positive error floor.
These are bounds on the actual operator, not labels or a feature-learning
algorithm. All conditional statements have inhabited examples.

Formalization: `ConvexChannel/LearnedQuadratic/{Basic,Model,Sequence,Witness,
Restriction,Spectral}`; direct audit: `scripts/LearnedQuadraticAxioms.lean`.
This construction supplies a polynomial-size learned quadratic building block,
not a solution of the all-165 99% target. Unseen-answer identification, richer
interactions for the full suite, numerical factor recovery, certified Mini and
forward-only Mini updates, and prospective sealed experiments remain open.
No producer, stress-case coverage or raw accuracy changes in this extension.

## CM86. Rectangular learned encoders, shared pair interactions and the next depth limit

The interface remains a sequence of immutable objects followed by one original
object. No numeric value, key/value role, arithmetic operation or answer rule is
attached to an object by the model. CM85's three-bit obstruction motivates a
richer reader and a rectangular joint lift, without squaring the larger context
dimension. This is an explicitly restricted learned family, not a complete
replacement for every task in the stress suite.

For a prefix of length at most T+1, use its last visible object as the query
and its preceding list as the context of length at most T. The empty prefix
reads learned biases on both sides. With V=|O|, put

```
Q = Option O,                                   q = 1 + V,
P = Fin (if T >= 2 then 1 else 0) x O x O,
D = Option ((Fin T x O) + P),                    d = 1 + T V + e V^2,
e = 1 if T >= 2, otherwise 0,
M_c in C^(Q x D),                               c in O.
```

The full joint domain is the real vector space of every such complex matrix
for every original output class. No rank constraint, label-dependent cone,
observed-pair pruning or independent complete-prefix table is imposed. The
zero-sized pair type when T<2 removes no possible occurrence: there are then
no adjacent positions, as `no_pair_position` proves. Every position/object
coordinate and every ordered object pair is otherwise retained.

For any rectangular M, diagonalize the query-sized positive matrix M M*.
Write U for the full unitary eigenbasis and sigma for the complete list of
singular values of M*. Lemma 1 supplies actual balanced factors

```
L = U diag(sqrt(sigma)),
R = M* U diag(1 / sqrt(sigma)),
L R* = M,
||L||_F^2 = ||R||_F^2 = ||M||_S1 = a.
```

Inverse zero is zero. All q singular directions remain represented, including
zeros; exact recovery holds for zero and rank-deficient matrices. Set
E=L/sqrt(a), F=R/sqrt(a). The readout magnitude a is learned with both factors.
For each hidden coordinate, actual inference reads the learned bias and final
object entry of E*, and the learned bias, every earlier position/object entry,
and every visible adjacent-pair entry of F*. Pair weights are shared across
positions, while overlapping occurrences and repeated pairs contribute separate
summands. In particular the context [false,true,false,true] reads false/true
twice and true/false once. No key/value-specific filtering occurs.

If these learned query and context responses are u and v, the complex class
output is `a (1+i) sum_h conj(u_h) v_h`. Its real part is the logit. Softmax
retains every original object, is positive and normalized, and `nextObject`
returns a maximizing original object. A strict winner is returned for every
tie selection. The vocabulary witness establishes nonemptiness, not a target;
native inference must declare a deterministic full-vocabulary tie order.

For arbitrary complex diagnostic inputs x,y, exact recovery proves

```
forward_M(x,y) = a (1+i) <E* x, F* y> = (1+i) x* M y.
```

The raw reader identities prove that this is the actual sequence computation.
Inference is defined by the learned lookups, encoders, interaction and readout;
the linear identity is a theorem about that path. Thus all joint coordinates
have affine real logits, ordinary original-object CE is convex, and every
indexed mean minibatch is convex with all records, multiplicities and tails.
This says nothing about convexity when L,R,a are varied independently. Their
recovery couples them through M; no claim that affine logits are necessary
for convex CE is used. An executable numerical factor recovery and its error
certificate are not supplied by a noncomputable spectral definition.

Coverage is exact for arbitrary finite-width complex two-encoder interactions:
given physical L0,R0, jointly choose M=L0 R0*. Recovery through q hidden
coordinates reproduces `(1+i)<L0* x,R0* y>` on every complex x,y. An independent
scalar readout can be absorbed into a physical factor. This may widen a smaller
imposed bottleneck; it does not preserve a fixed low rank or one encoder tied
across classes. Class-specific encoders share their learned tables across all
input prefixes. At the minimal three-object cap the permitted pair/query
coefficient table can distinguish every triple, but there is no parameter
indexed by a whole prefix of arbitrary length.

The encoder geometry is not merely a parameter-dependent hidden rotation.
`left_hidden_gram` and `right_hidden_gram` give the identical full hidden Gram
`diag(sigma)`. Consequently

```
L L* = sqrt(M M*),                  R R* = sqrt(M* M).
```

For square positive M, both equal M and both normalized covariances equal
M/||M||_S1. The two positive joint points [[4,i],[-i,1]] and [[1,i],[-i,9]]
have learned readouts 5 and 10. In both actual normalized encoders, the first
object's covariance entry changes from 4/5 to 1/10. These observables do not
depend on an arbitrary eigenbasis choice and cannot change under a hidden
unitary rotation alone. Removing the imaginary entries changes the actual
real output on the off-diagonal basis probe from -1 to 0. These are feasible
comparisons, not an optimizer trajectory or raw stress accuracy.

The complete dense joint storage is exactly

```
2 V (1+V) (1 + T V + e V^2) real coordinates.
```

It includes every class and both complex components, before factors, gradients,
optimizer state, data or solver workspace. The recovered hidden width is q per
class, total Vq. A direct spectral implementation forms q-by-q Gram matrices,
with dense per-class work O(q^2 d + q^3) for Gram/eigendecomposition/recovery;
this is an algorithmic cost estimate, not a measured or Lean-verified runtime.
Caching all L,R costs V(q^2+dq) complex entries before any extra normalized
copies. A cached physical forward reads O(Vq(n+1)) entries for n visible
objects, including overlapping pairs, then scans all V outputs. No claim
that these buffers fit every stress case within 110 GB, or that complete
forward-only coordinate queries are affordable, follows from polynomial size.
The complete case dimensions and all peak buffers must be checked before runs.

The capacity gain and its limit are both explicit. `Parity.parameters` is a
chosen feasible point, not part of the architecture: class false has logit
zero, while class true reads coefficient +1 or -1 at (last object, preceding
ordered pair). Through the actual raw learned path this yields the original
three-bit parity answer as strict winner on all eight triples. In contrast,
fixing the final object to false in a four-object prefix leaves three varying
context bits. Their complete address vector has zero third mixed difference:
singleton entries and adjacent pairs each depend on at most two of those bits.
The actual recovered logits preserve that identity. `parity_not_realizable`
therefore rules out strict classification even on these eight four-bit parity
inputs. More learned composition is still needed; this extension alone cannot
be advertised as the all-165 solution.

Flatness is used only as a bound. Complete complex basis probes of the actual
recovered forward identify M, including its fixed nonzero phase. R3 gives

```
s (min(q,d)-h) <= ||M - L0 R0*||_F^2
```

for an s-flat M and physical encoders of width h. Reproducing all complex
coefficient probes exactly requires h>=min(q,d). These are coefficient probes,
not extra supervised raw sequences; equality of raw real logits alone is not
claimed to identify the full complex matrix. No missing answers are obtained
from flatness. All conditional statements have inhabited examples.

Formalization: `ConvexChannel/LearnedBilinear/{Basic,Model,Reader,Sequence,
Geometry,Witness,Parity,Restriction,Spectral}`; direct audit:
`scripts/LearnedBilinearAxioms.lean`. Remaining work includes richer learned
composition, explicit unseen-prefix identification assumptions, numerical
spectral residual certification, Mini and forward-only Mini updates, memory
admission and prospective independent sealed runs. No native producer or
stress-case credit changes in this extension.

## CM87. Learned capacity and observable unseen-answer certificates

The user initially requested one Lean `sorry` and a new board task on
2026-11-16, then asked to prove it. Task 367 concerns identification in CM86's actual learned sequence
family. It is separate from coding-discovery tasks 196/197 and the fixed-encoding
controls 148/365. The mathematical target is a finite margin-certificate theorem,
not an unconditional claim that a flat spectrum determines missing labels.

Let p range over every complete joint complex matrix in CM86. For a raw prefix
x and original objects a,b, define the actual margin

```
m(p,x,a,b) = sequenceLogits(p,x,a) - sequenceLogits(p,x,b).
```

Given indexed observed prefixes x_i and their original labels y_i, let F be the
set of p with m(p,x_i,y_i,c)>=1 for every i and c!=y_i. Assume F is nonempty.
This is consistency of the observed constraints, not a successful optimizer,
an attained unregularized CE minimum, or a correct unseen answer. A two-class
inhabited example is included independently of the theorem. The central
equivalence below is now proved with its original statement unchanged.

Expose the finite real coordinates z=(imaginary flag,class,query address,
context address), keeping both complex components and every output class.
The diagnostic coefficient vector v(x,a,b) is

```
(1[class=a] - 1[class=b]) * (1 for real, -1 for imaginary)
  * Re(queryRow(x)[query address] * contextRow(past(x))[context address]).
```

The row values here are real counts embedded in C. `margin_expansion` proves
that these finite coefficients give the actual recovered margin. This is not
a new inference definition or an assumption about unknown target weights.
Encoding and decoding the complete real coordinates are mutually inverse.

For any query prefix u, candidate objects a,b and requested bound beta,
`forced_margin_iff_certificate`:

```
[for every p in F, m(p,u,a,b) >= beta]
  iff there exist finite weights w[i,c] such that
    w[i,c] >= 0,                    w[i,y_i] = 0,
    sum_{i,c} w[i,c] >= beta,
    v(u,a,b) = sum_{i,c} w[i,c] v(x_i,y_i,c).
```

The right side has finitely many scalar inequalities and coefficient equalities.
It uses only observed labels and prefixes plus the input u and proposed classes;
it neither reads u's true answer nor assumes that all completions agree. The
nonempty F hypothesis is necessary to avoid false certificates from vacuous
universal statements. The proof first establishes a general finite linear
margin alternative, then applies it to the complete actual coordinate forms.
No executable linear-program solver is supplied by this statement.

The completeness proof has four steps. A forced lower bound is nonnegative
on every feasible recession direction. Homogenizing the constraints with a
scale t>=0 then gives nonnegativity of f(p)-beta*t on the cone
(t, (A_i p-t)_i)>=0, including t=0. A feasible unit-margin point p0 gives
a strictly positive observation vector at (2*p0,1). Mathlib's M. Riesz positive
extension theorem extends the objective from the observation map's range;
its finite coordinate coefficients are nonnegative. Evaluating at (p,0)
gives f=sum_i w_i A_i, and at (0,1) gives beta<=sum_i w_i. Kernel
well-definedness and the extension's cofinality condition are proved explicitly.
No closed-cone assumption, attained CE minimum or optimizer oracle is added.

Self-label comparisons are restored with weight zero. The original record
indices remain, so repeated examples and all competitors retain their sums.
`rationalCoefficient_cast` proves that the signed rational counts equal the
full real coefficients. `checkCertificate_iff` proves correctness of the finite
exact-rational Boolean checker on rational supplied weights and bounds. This
does not assert a certificate-search algorithm or rational completeness of
the real existential theorem. Direct Lean evaluation is a reproducible checker
run; the correctness theorem itself uses only standard Lean axioms.

Positive bounds for every competitor force a unique original-object answer at
u for every model consistent with the observations. To call that answer correct
for a real task still requires explicit realizability in the stated family; the
target rule and its parameters need not be supplied to the certificate checker.
The query need not be held out in the general equivalence, but the positive
acceptance witness must be genuinely unseen with nonconstant observed labels.

A proved witness uses context cap3 and fixed final object false. Observe
[false,false,false,false]->false, [true,false,false,false]->true and
[false,false,true,false]->true. Withhold [true,false,true,false]. The varying
context positions are nonadjacent, so the query's raw address product is the
sum of the two positive-prefix address products minus the negative-prefix one.
Consequently its true-versus-false margin functional is the sum of all three
observed correct-versus-incorrect margin functionals, with nonnegative weights
one and total mass three. Lean proves feasibility, this identity, the
certificate, strict original-object prediction and held-out status. The finite
rational checker accepts this certificate. Explicit parameters have false
logit zero and true logit -1+2*a+r*b+2*c on [a,b,c,false]. Every r fits the
three observed unit margins. On the other unseen prefix [false,true,false,false],
r=0 predicts false and r=2 predicts true. Consequently neither candidate has
any positive certificate there. These are proof
controls, not an original stress-test score or an assumed task formula.

The capacity connection must distinguish rank from directional information.
CM86/R3 controls possible representation width and approximation error. The
observed-margin coefficient matrix describes which directions the data constrain.
Its rank or span alone does not provide the nonnegative certificate or guarantee
an unseen class margin. `observationOperator` evaluates all observed margins;
`observation_rank_bound` bounds its rank by both the complete CM86 real
coordinate count and the indexed competitor count. Labels impose inequalities
on its outputs; they are not measurements of exact real logits. Every certified
query functional annihilates this operator's kernel. Common class matrices and
equal real/imaginary increments are proved null directions of every raw margin.
These coordinates remain present; no quotient is silently substituted for the
learned model. CM86's flat coefficient bounds continue to concern physical
representation width and coefficient-probe approximation. They do not supply
the nonnegative directional identity required by this theorem.

The ordinary CE bridge is explicit: for a competitor c!=y,
CE(z,y)<=log(1+exp(-m)) implies z_y-z_c>=m. Thus the threshold with m=1
on every indexed record implies the feasibility constraints. The same threshold
on a minibatch mean alone is not used. The concrete feasible witness attains
that CE threshold on all three records. If observed margins are only at least
1-e[i,c], `certificate_robust` proves query margin at least
beta-sum_{i,c} w[i,c]*e[i,c]. A uniform error e is charged as e*sum w;
for the mass-three example this gives 3-3*e. Distinguish certificate existence/checking,
structural family discovery, and Mini or forward-only Mini convergence. None
is implied by capacity alone, and no optimizer convergence is requested here.

Formalization starts at `ConvexChannel/LearnedBilinear/Identification.lean`.
Its `Identification/` modules contain the general alternative, exact coordinates,
duality, rational checker, observation geometry, CE/error bridges, and success/
refusal witnesses. The requested `sorry` is removed; the older36 unfinished
statements remain. Conditional results have inhabited examples. Reproduce the
proof checks with `LAKE_ARTIFACT_CACHE=false lake build`, direct Lean compilation
of these modules, `lake env lean scripts/LearnedIdentificationAxioms.lean`, and
`lake env lean scripts/Axioms.lean`; regenerate INDEX with the native generator.
No new native experiment, accuracy credit or change to frozen producers
accompanies this extension. Task-family discovery and Mini/forward-only Mini
convergence remain separate open work; correctness for the unknown target
continues to require explicit family realizability.

## CM88. Complete forward-only queries and exact zero-column storage

The user's request to resume GPT stress testing requires a native CM86 model;
the existing CM75--83 pilot implements a different family. The following exact
identities support the new implementation without discarding coordinates or
using an analytic CE derivative in the forward-only loss-query calculation.

Let r encode every real and imaginary joint coefficient, and let z identify
one coordinate, including its original output class c. The actual logit
coefficient is zero for every other class; for c it is the signed response

```
s(x,z) = (1 for real, -1 for imaginary)
         * Re(queryRow(x)[q]) * Re(contextRow(past(x))[d]).
```

The actual recovered output after changing r_z by t changes only logit c, by
t*s(x,z). The real bias coordinate (c,none,none) has response one on every
prefix, including the empty prefix. Consequently a complete coordinate query
at step t equals the class-bias query at step t*s for this record, before and
after ordinary full-vocabulary CE. This is proved from CM86's actual complex
path through its exact finite coordinate expansion.

Write D_z(h) for the central difference of this record's ordinary CE after
the two complete parameter perturbations plus/minus h at coordinate z. For
nonzero h and nonzero response s, CM88 proves

```
D_z(h/s) = s * D_bias(c)(h).
```

If s=0 then D_z(t)=0 for every t. Thus two bias loss queries for every
original class and every indexed supervised record give the complete vector
of coordinate differences, with a response-dependent coordinate step. They
are not selected-coordinate estimates. The resulting mean uses every record
and its actual batch length, retaining duplicates and short tails. It is the
mean of per-record coordinate differences at those steps, not a claim of one
common coordinate step for the entire batch loss. No exact-derivative,
finite-step error or convergence theorem is asserted. A repeated-pair witness
has response two; a genuinely absent query address supplies a zero response.

For either analytic residuals or measured bias loss differences, define the
complete pullback by N^-1 sum_i s(x_i,z)*scalar[i,c]. If a coordinate has zero
response on every record, its pullback is exactly zero. A zero parameter at
that coordinate stays zero after a complete update. This permits a sparse
storage representation whose absent entries mean exact zeros in the full
unchanged domain. It does not permit deleting small nonzero parameters.

The actual spectral rightFactor has a zero context row whenever the joint
matrix's corresponding context column is zero; rightRepresentation retains
that zero after normalization. The full query Gram matrix M*M* equals the
sum over stored columns whenever every other column is exactly zero. The
remaining singular directions and both original encoders are retained. These
facts justify avoiding arithmetic and storage of zero columns; they do not
justify singular-value thresholding or numerical rank truncation.

Formalization: `LearnedBilinear/Updates/{Basic,Forward,Support}`. Native tests
must separately compare actual recovered complex inference, complete direct
coordinate queries, finite-difference reuse, analytic controls, zero storage,
all original classes and short minibatch tails. Numerical SVD, finite differences
and binary64 CE require measured residual/error reporting; these exact Lean
identities are not a proof that floating-point evaluations are bit-identical.

## CM89. Training-input scaling without reducing the learned family

The initial native CM86 stress control fits short Boolean examples, while
plain SGD at rate0.01 barely reduces the modular-addition training CE within
200 updates. These are training observations, not held-out scores. Rare joint
input responses motivate a separate update-geometry diagnostic.

For N indexed training prefixes define, on every complete real coordinate z,

```
w(z) = (N+1) / (1 + sum_i response(x_i,z)^2).
```

This scale reads training inputs only, retains duplicate records, and squares
the full adjacent-pair multiplicity. Every w is strictly positive. Class-bias
coordinates have w=1; coordinates absent from all training inputs have w=N+1
and still receive zero complete pullback. The scale is the same for every
output class and for the two complex components of a given query/context
address. No small response, class or possible parameter is removed.

The diagonal coordinate map r ↦ w*r is linear and invertible when every w is
nonzero, and complete real coordinates decode linearly to the original
complex joint matrices. Consequently ordinary CE through the actual two
learned encoders remains jointly convex in these scaled coordinates. This
does not replace inference or assert independent-factor convexity.

A preconditioned update uses r_z - rate*w(z)*g_z for the entire pullback g.
For forward-only Mini, w times the CM88 direction is exactly the indexed
mean of w times every coordinate's actual finite CE difference at its
response-scaled step. The original record count and all repetitions remain.
Exact zero-support preservation therefore continues to hold. A repeated-pair
control has response two twice and w=3/(1+8)=1/3, showing that multiplicity
is not replaced with binary presence.

Formalization: `LearnedBilinear/Updates/Scaling`. These are geometry and exact
query identities, not a convergence theorem, a step-size guarantee, a new
identification condition or accuracy credit. Any native comparison must use
a new frozen protocol and leave the original stress-control producer intact.

## CM90. A silent coordinate diagnoses missing-answer ambiguity

Improving the optimizer does not strengthen the observations themselves.
Suppose a complete real coordinate z has zero CM88 response on every indexed
training prefix. Changing that coordinate leaves every actual recovered
training logit unchanged. It therefore preserves the ordinary training loss
and all observed unit margins. If the same coordinate responds nontrivially
to a query and belongs to original class a, its change shifts the a-versus-b
query margin by t*response(query,z) for every b≠a. Starting from any fitting
parameter, choose t to set that margin to any prescribed real number.

Thus no finite lower-margin certificate can exist for that comparison when
the observations are feasible. This is an explicit countermodel construction
in the actual learned family, rather than a count of parameters or a flatness
heuristic. It does not say that every learning algorithm must predict the
wrong answer: additional structure or selection bias may choose a completion.

For ordered two-object inputs [u,v], the real coordinate of class a at query
address `some v` and context address `position0,u` has response one exactly
on that ordered input and zero on every different ordered two-object input.
Every missing pair therefore has a silent responding coordinate. Feasible
training labels on other pairs do not force any finite query-margin bound
in the complete unconstrained CM86 family. The statement uses only immutable
object identities and applies to arbitrary two-object tasks, not an arithmetic
operation. It also explains why zero initialization and complete training-only
pullbacks leave these particular unseen interaction entries zero.

Formalization: `LearnedBilinear/Updates/Unobserved`. Inhabited controls include
the earlier nonconstant three-record CM87 example and a separate nonempty
two-object training example. The result diagnoses an observation/structure gap;
it is not an empirical held-out result, a general impossibility of learning,
or an assertion that increasing optimization speed closes that gap.

## CM91. Capacity pays for a nonzero identification residual

The native CM89 addition diagnostic reaches100% training accuracy and0%
held-out accuracy. CM90 explains why observed labels do not force missing
answers in the unrestricted joint family. A capacity restriction changes the
admissible completions, so it can strengthen identification. This restriction
is additional structure: neither fitting nor a flat coefficient spectrum
establishes the correct budget for an unknown target.

Keep the input/output contract unchanged: a sequence of immutable objects
predicts one next object from the original vocabulary. IDs supply addresses,
not arithmetic values or supplied semantic vectors. Every real and imaginary
joint coordinate remains trainable, and the actual recovered complex encoders
remain the inference path. Define the complete squared size E(p)=sum_z p_z^2
in CM87's real coordinates. The region E(p)<=B is convex, and ordinary CE
through the actual learned forward remains jointly convex on this region.
This does not assert convexity in independently varied physical factors.

Let A_i be observed margin coefficient vectors, lower_i their known lower
bounds, q the proposed query comparison and w_i>=0. Define the full residual
r=q-sum_i w_i A_i and the supplied margin mass m=sum_i w_i lower_i.
The following finite inequalities suffice for query margin at least beta:

```
E(p) <= B,     A_i dot p >= lower_i,
m - beta >= 0,
B * E(r) <= (m - beta)^2.
```

Indeed Cauchy--Schwarz gives (r dot p)^2<=E(r)E(p), hence
r dot p>=-(m-beta). Adding the nonnegative observed combination gives
q dot p>=beta. Squared arithmetic avoids numerical square roots in a rational
check. Original records, duplicate observations and every original competitor
remain in the sums. Explicit lower_i also permit observed-margin errors.
For the sequence certificate, self-label weights are zero and other observed
lower margins are one, as in CM87. Its coefficient reading is proved equal
to the actual learned margin. No query's true label or unknown target
parameters are inputs to the certificate. Positive certificates for all
competitors force one original object; correctness for a real task still
requires that the target belongs to the declared bounded family.

The earlier exact coefficient identity is the zero-residual special case.
No converse or certificate-search/optimizer theorem is asserted here. Unlike
CM87, a nonzero residual can now be tolerated because a silent coefficient
cannot move arbitrarily while remaining inside the capacity region. Rank or
flatness alone does not supply this directional residual bound.

A complete Boolean-object witness demonstrates strictly stronger identification.
Observe [false,false]->false, [false,true]->false, [true,false]->true, and
withhold [true,true]. The labels are nonconstant and the query differs from
every observed input. Let A0,A1,A2 be the observed correct-versus-other margin
forms. The complete real parameter

```
p0 = (5/48) A0 + (1/24) A1 + (1/8) A2
```

fits all three unit margins, has E(p0)=13/48, and has true-versus-false query
margin1/4. With B=13/48, beta=1/8 and weights(20/3,8/3,8), the mass is52/3
and E(r)=3280/3. The squared slack exceeds B E(r) by1/64. Thus every fitting
actual model within this budget predicts true on the missing pair. In
contrast, CM90 proves no finite unrestricted margin certificate exists there.
This is a nonempty bounded example and a real nonzero-residual certificate,
not an original stress-test result or a supplied unseen-answer constraint.

Formalization: `LearnedBilinear/Capacity`. The bound establishes how a
capacity restriction can identify an unseen object, but does not establish
which shared structure or budget solves the failed original GPT tasks.
In particular, Euclidean size alone has not been shown to repair addition,
MQAR or the CM86 higher-context parity restriction. Those remain empirical
and architectural work under the active generalization goal.

## CM92. A bounded shared learned output factor and feasible CE queries

CM86 recovers separate factor pairs for every original output object. A new
candidate uses components with three trainable complex vectors: u over output
objects, v over query addresses and w over context addresses. No arithmetic
meaning, task program, semantic vector or fixed finite feature bank is supplied.
For a raw object sequence x, execute

```
(1+i) u[c] readQuery(x,v) readContext(past(x),w).
```

The complete coefficient tensor is u[c] v[q] w[d]. Summing components with
learned real weights executes all three factors and predicts the next original
object from the complete class scores. This execution equals CM86's complete
joint reading, including complex values before taking real parts, on every
prefix. The output factor couples the two input encoders across all classes.
The raw reader still has CM86's interaction degree; this change alone cannot
remove the proved higher-context parity restriction.

Bound each component's three complete squared vector norms by one. Allow every
complex direction satisfying these bounds, and take the convex hull of the
complete tensors multiplied by a declared scalar budget B. This is a convex
domain in joint tensor coordinates, and ordinary full-vocabulary CE is convex
on it. It is not joint convexity in independently varied physical factor arrays.
Every domain point has a finite representation as a convex combination of
actual bounded factor triples; conversely every such finite representation is
feasible and has the proved executable path. A zero component holds unused
mass. No fixed rank or singular-direction threshold is imposed by the domain.
Existence of a finite representation is not an efficient discovery algorithm.

For one component, the complete real coordinate energy is exactly the product
of its three complete complex squared norms. Thus it is at most one. Convexity
of the CM91 energy ball implies E(p)<=B^2 for every atomic-domain point. This
connects the domain to CM91's sufficient residual certificates. The converse
energy-to-atomic-domain inclusion is not assumed. This bound does not establish
that the original stress targets lie in a chosen budget or that their unseen
answers admit certificates.

A convex component update (1-eta)p+eta*B*atom remains feasible for0<=eta<=1,
for every learned admissible candidate. An approximate component search may
therefore propose valid states, but does not supply an exact global linear
minimization oracle or a global optimization guarantee. Native work must report
this limitation, retain every attempted update and failure, and score held-out
data only under a separately frozen protocol. A nonconstant complex witness
has squared norm1/2 in all three vectors, retains both original objects, and
uses imaginary entries to change real output scores. It gives no stress credit.

Forward-only queries must also correspond to allowed model parameters. Given
current p, use the common centre (1-eta)p. A signed class-bias component has
u=+/-e_c, v=e_bias, w=e_bias, so it is admissible and changes only class c's
real logit. The two complete query points

```
(1-eta)p + eta*B*bias(c,+1),
(1-eta)p + eta*B*bias(c,-1)
```

remain in the same domain. Their real logits equal the centre logits with
class c changed by+/-eta*B. Their ordinary CE evaluations therefore supply
the exact finite central bias difference at step eta*B. CM88's per-record
reuse supplies the complete real/imaginary direction, retaining duplicates,
pair multiplicities, every original class and short batch tails. This does
not identify finite differences with analytic derivatives. Mini and forward-
only Mini must use the same shrunken centre for a matched implementation.

Formalization: `LearnedBilinear/SharedAtoms`. The proofs make a new shared
factor experiment concrete while keeping the input/output contract and the
optimization, identification and depth limitations explicit. They do not
resolve the active generalization goal without new original-split evidence.

## CM93. Positive geometry separates bias capacity from object interactions

The first frozen CM92 native run fails in both minibatch modes: training
accuracy133/4704 and held-out accuracy1/4705, with exact raw-logit replays.
At the Mini endpoint, component-mass-weighted squared query/context bias
energies are0.9780685/0.9783835, excluding the zero component. This is a
measured dominance of always-present input addresses, not a proof that the
model class or every possible optimizer is incapable of generalizing.

Choose fixed nonzero real scales a[q] and b[d], without labels or semantic
meanings for object IDs. Execute the learned query vector a*v and context
vector b*w. On the complete joint real coordinates this is the linear map
D(r)[z]=a[q(z)]*b[d(z)]*r[z], shared across every original class and both
complex components. It has an explicit inverse when every scale is nonzero.
Thus no coordinate direction is removed. The new fixed-budget domain is the
linear image D(CM92-domain(B)), which is convex and still has finite actual
factor representations. Ordinary original-object CE is convex on it. The
bounded image is not asserted equal to the old region; this changes the
capacity geometry, not merely the name of the coordinates.

Scaling commutes with a complete coordinate perturbation, multiplying its
actual raw response by a[q]*b[d]. The complete indexed pullback therefore
receives this factor without losing duplicate records, short tails, possible
coordinates or original classes. If every squared joint scale is at most one,
complete coordinate energy cannot increase, so the image still satisfies the
CM91 bound E(p)<=B^2. This is a sufficient bound, not a capacity-only guarantee
of correct unseen labels.

The proposed dimension-only geometry sets both bias-address scales to1/sqrt(V)
and all object, position and shared pair address scales to one, for positive
original vocabulary size V. Every scale is strictly positive and at most one.
The two bias factors multiply to1/V. Every learned numerical coordinate remains
trainable; no fixed semantic embedding or task-specific answer rule is added.
The input remains an immutable-object sequence and the output one next original
object. The existing higher-context parity limitation remains.

Forward queries must account for this changed bias response. The scaled image
of CM92's signed query point has the centre logits with one class changed by
plus/minus eta*B*a[none]*b[none]. To request the same raw CE step h>0, reserve
eta=h/(B*a[none]*b[none]), requiring h<=B*a[none]*b[none]. Both signed queries
remain feasible. Under dimension scaling this is eta=h*V/B, not h/B. Their
ordinary CE difference is exactly the finite class-bias difference at the
requested step. Both Mini and forward-only Mini must use the same centre,
and native complete responses must include the new coordinate scales.

Formalization: `SharedAtoms/Scaling`. These are geometry, capacity, actual
forward and finite-query identities. They do not prove a better component
oracle, a convergence rate, target-family membership or new stress accuracy.
Any native test uses a separate frozen protocol and preserves the first failed
shared-factor producer and its original-data evidence.

## CM94. Exact zero-row execution for broader shared-factor tests

The first CM93 original addition cohort reaches100% training and held-out
accuracy in both minibatch modes, seed1; full saved-logit replays match. This
is empirical evidence on one original split, not a generalization theorem for
all tasks. Other seeds, short associative recall and longer sequences require
separate prospective protocols. The CM86 higher-context parity limit remains.

The complete gradient array has one real g[i,c] per query/context row i and
every original class c, with the exact imaginary partner from CM88. Its memory
and model dimensions are not reduced. Define A={i | exists c, g[i,c]!=0} using
exact zero equality over all original classes. For every complex multiplier v,

```
sum_{i in A} sum_c g[i,c]*v[i,c] = sum_i sum_c g[i,c]*v[i,c].
```

Every omitted row has all entries zero, so every term vanishes independently
of the learned multiplier. Taking v from the actual complete factor product
gives the unchanged component objective, including the (1+i) output rotation.
The three coordinate contractions are specializations of the same identity.
Skipping these zero rows therefore changes execution cost without introducing
a norm threshold, deleting a parameter, or restricting the learned directions.
All original gradient coordinates and complete factor vectors remain stored.
An explicit nonzero complex control has one zero row and one active row.

Formalization: `SharedAtoms/Support`. The native implementation must select rows
afresh from the complete current gradient, preserve their original order and
compare contractions and objectives with the dense implementation. It must not
apply this rule to merely small values. This supplies no global component-oracle
or optimization theorem and no new held-out accuracy claim by itself.

## CM95. Learned complex recurrent depth on immutable object sequences

CM86's pair/query reader has a proved fixed-interaction restriction: even a
four-object parity slice cannot be strictly classified. Capacity scaling cannot
remove that algebraic identity. A separate candidate gives each original
object a learned complex linear transition on a state space E, with learned
initial state and one learned linear readout for each original output object.
Execute the transitions from left to right on the input list, then read the
complete complex output vector and use its real scores for next-object CE.
For native use E is finite dimensional. No object ID has numeric meaning and
no transition or answer program is supplied to the learner.

Require initial norm, every transition operator norm and every readout operator
norm to be at most one. The actual hidden-state norm then never exceeds the
initial norm, and every complex output has norm at most one, for all lengths.
If transitions are isometries, hidden-state norm is exactly preserved at every
depth. This is the role of a flat operator spectrum: a capacity/execution bound,
not a set of prescribed semantic features or a source of missing answers.
Real and imaginary parameter entries remain part of the executed path.

Take the convex hull of all B-scaled score functions from these learned unit
components. This is a convex joint function domain with finite executed mixture
representations; ordinary full-vocabulary CE is convex there. It is not a claim
that independently varied transition matrices and readout arrays have a convex
parameterization or that component discovery has an efficient global oracle.
The domain is defined on every object list, rather than a table of observed
training prefixes. No held-out target appears in its construction.

Finite CE queries use an actual bias recurrence on the same state space.
Choose v and a linear readout L with norm(v)<=1, norm(L)<=1 and L(v)=1.
Identity transitions keep v unchanged; output c is s*L when c is the queried
class and zero otherwise, with abs(s)<=1. This is admissible and changes only
that class's real score by s on every sequence. Thus the two mixture points
(1-eta)r+eta*B*bias(+1/-1) stay feasible for0<=eta<=1 and have centre logits
with one class shifted by+/-eta*B. Their ordinary full-vocabulary CE values
give the finite bias difference, without equating it to an analytic derivative.
The nonzero state/readout requirement is explicit: in E=complex numbers take
v=1 and L=identity. A zero-dimensional space is not silently assumed to admit it.

The one-dimensional subfamily uses learned complex scalars a,t[o],u[c] and
executes u[c]*a*product_o t[o] in input occurrence order. Scalars commute, so
this subfamily alone is not a general order-dependent sequence architecture;
higher-dimensional transitions retain order. As an expressivity witness only,
take Boolean objects with a=i, t[false]=1,t[true]=-1 and u[false]=-i,u[true]=i.
All parameter norms are one and every transition spectrum is flat. The actual
complex and real scores are+1 for the parity object and-1 for its competitor
at every input length. The same budget-one domain point therefore predicts
arbitrary-length parity with margin2 and is nonconstant. Erasing its imaginary
initial state/readouts destroys this execution. These weights are not supplied
to a learner; this is a witness that the fixed-interaction obstruction has
been removed, not a new stress result, optimizer theorem or identification
guarantee from incomplete data. Native learning remains separate work.

Formalization: `ConvexChannel/LearnedRecurrent`. The input/output contract
remains an immutable-object sequence to one next original object. General
unseen-answer correctness still needs target-family assumptions and training
evidence; bounded recurrent depth alone cannot determine arbitrary labels.

## CM96. Scalar execution is complete but permutation-invariant

The first native recurrent prototype may use the scalar complex state E=complex
numbers. This retains every parameter of that declared state family: every
complex linear operator L on it is exactly multiplication by L(1). Reconstructing
the initial state, all transitions and all class readouts as scalar parameters
therefore changes no operator or output. It does not claim equivalence with a
higher-dimensional recurrent family.

Changing t[o] for an object absent from a raw prefix changes no executed product,
complex output or real logit. If that object is absent from every indexed record
in a minibatch, every corresponding loss is unchanged. A native proposal search
may skip those exactly unobservable changes while storing every original object's
parameter; future batches may observe it. No small-value threshold or semantic
role selection is allowed. Output classes and all original records remain complete.

Scalar multiplication commutes. Every scalar recurrence therefore gives the same
scores to any two permutations of the input list. The identity survives every
finite convex mixture, hence the entire scalar function domain. In particular,
the two Boolean-object prefixes[false,true] and[true,false] cannot both have a
strictly correct first-object prediction. The scalar prototype is consequently
not a general order-sensitive MQAR model, even though CM95 gives an arbitrary-
length parity witness. Higher-dimensional learned operators are a separate step.

Formalization: `LearnedRecurrent/Scalar/Structure`. Test scalar learning first
on unchanged original Boolean and parity records, including their original
answer/end tokens and all length/position probes. Do not feed the witness's
phases or label rules to the learner. Its candidate search uses complete CE
signals and actual learned transitions, with finite loss evaluations at feasible
mixture points; approximate discovery still has no global convergence guarantee.

## CM97. A complete two-state learned operator family retains order

CM96 rules out order distinctions only for the scalar-state subclass. Use
E=complex×complex with the product maximum norm. A learned transition contains
all four complex matrix entries, a learned class readout contains both complex
coefficients, and both initial-state coordinates are learned. Every complex
linear map on this state is reconstructed from its values on the two standard
basis states; these are coordinate identities, not fixed task features.

For a row u=(a,b), the constraint |a|+|b|≤1 implies its operator norm is at most
one. Applying that constraint to both transition rows, to every original-class
readout, and bounding the initial maximum norm by one gives the exact CM95 Unit
hypotheses. Thus its length-independent output bound, convex function domain,
ordinary CE and feasible loss queries apply unchanged. The chosen maximum norm
is explicit: this is not a claim that all bounded matrices have flat Euclidean
singular values or that independently varied matrix factors are jointly convex.

The two transition assignments diag(1,-1) and the coordinate swap are
noncommuting and preserve the state norm. With initial state(i,i), the prefixes
[false,true] and[true,false] end at(-i,i) and(i,-i), respectively. Class readouts
(i,0) for false and(-i,0) for true give the correct first object margin two on
these two swapped prefixes. Actual nextObject therefore distinguishes them,
whereas every scalar-domain function is permutation-invariant by CM96.

This is a nonconstant existence witness inside the learned family, never a
supplied native parameter assignment or proof of MQAR learning. A native learner
must keep every complex initial/matrix/readout coordinate and learn from original
CE observations. The matrix norm constraints control capacity; neither they nor
this order witness select arbitrary unseen answers. Formalization:
`LearnedRecurrent/Pair/Basic` and `LearnedRecurrent/Pair/Order`.

## CM98. Learned rotary positions inside the bounded recurrent family

Use the existing `GPTMini.RoPE.applyRope_isometry` and
`GPTMini.RoPE.applyRope_relative` geometry, with learned frequencies as in the
local `RoPE/LeRoPE/Section3_LearnedFrequencies` formalization. Here both complex
channel frequencies are unrestricted real trainable parameters; no prescribed
frequency grid, token meaning or answer rule is introduced. This is a derived
model extension, not a claim that the cited frequency formula proves learning.

For theta=(theta1,theta2), let phase(theta_j,t)=exp(i*t*theta_j) and let R(t)
be the diagonal complex operator with these phases. R(s)R(t)=R(s+t), R(-t) is
its inverse, and R(t) preserves the maximum norm on the two complex channels.
This is the complex-coordinate form of two independent real rotary pairs.
Relative rotations depend on the position difference. At position t, execute
A_t(o)=R(t) T_o R(-t), where every original-object matrix T_o, initial state and
class readout remains learned. R and T interact inside actual execution.
Conjugation preserves each transition's operator norm; the same Unit budgets
therefore bound every actual output independently of sequence length.

The external input remains List O and the output remains one original O.
An internal position counter starts at zero and increments once per occurrence;
there is no per-prefix parameter table. The joint score domain is the convex
hull of all admissible learned frequencies and content parameters. Physical
frequency/matrix discovery remains approximate and is not jointly convex.
Ordinary full-class CE is convex in the joint scores; zero-frequency identity
components give the same feasible signed class-bias queries as CM95.
Replacing a content matrix for an object absent from the whole raw prefix
changes neither its executed state nor any original-class score. This permits
exact absent-object proposal skips while retaining every learned parameter;
small nonzero responses and original classes cannot be filtered.

In a scalar state, this particular shared rotary conjugation has no effect:
phase(t)*a*phase(-t)*z=a*z. This does not rule out other position-dependent
scalar models. In the two-state family even commuting content matrices can
produce an order distinction through their position-dependent conjugates.
Use a swap for false and identity for true, initial state(1,0), frequencies
(pi/2,0), and readouts(0,1) for false and(0,i) for true. The two swapped prefixes
have states(0,1) and(0,-i), and predict their first original object with margin1.
These are existence witnesses only; no such weights enter native learning.

Preserving each rotation or transition norm does not preserve a token-table
Gram spectrum. The existing `SequenceModel.RotarySpectrum` counterexample
changes eigenvalues(1,1) into(2,0) under valid rowwise RoPE while retaining each
row norm. CM98 asserts operator/output bounds, not that arbitrary positionwise
rotations keep the data Gram matrix flat or identify arbitrary unseen labels.

## CM99. Fixed rotary controls and minibatch CE interference

The user requested a fixed-rotation alternative to CM98's learned frequencies.
Restrict every component to one externally chosen frequency pair theta, retaining
all learned initial, content-matrix and original-class readout coordinates. The
convex hull of these actual functions is a convex subset of the full rotary
function domain, so its output bound and ordinary CE convexity remain valid.
Rotary conjugation of the identity is exactly the identity at every position.
Thus a signed constant-class bias recurrence using the same fixed theta has the
same actual score as CM98's zero-frequency query. Both finite queries remain
inside this restricted domain, not merely inside a larger learned-frequency
family. The proposed fixed control uses the task-independent two-pair schedule
(1,1/100), corresponding to GPTMini.invFreq with real head dimension4 and
base10000. This is an explicit positional control, not a fully trainable
frequency family or a supplied answer rule.

Convex feasibility and decrease on one minibatch do not imply decrease of the
combined training loss. Consider the distinct prefixes[false,true] and
[true,false], with targets false and true respectively. Start from zero scores
and let B=log3. The actual admissible class-bias model with scoreB for false and
zero for true is in the fixed-theta domain for every theta. It decreases the
first example's CE from log2 to log4-log3, but increases the pair's mean CE
from log2 to log4-log3/2, which is strictly greater. The same full rotary family
contains a strict separating model: the zero-frequency CM97 order witness
scaled byB gives correct-class margin2B on both prefixes. This example isolates
the lack of a full-data descent implication; it does not prove that the native
run's failure has only this cause or that a step schedule solves identification.

For zero-based update t, cap the numerical convex line search by2/(t+2).
This cap is positive and at most one. Any nonnegative accepted step below it
therefore satisfies the existing full and fixed-family feasibility theorems.
No global convergence, regret or generalization guarantee is asserted for the
approximate component search. Compare plain, fixed-rotary and learned-rotary
execution with unchanged original data and all classes under this same cap,
retaining the failed unrestricted-step producer and all complete raw replays.

## CM100. Revisit all complex readouts on already learned recurrent states

The native recall runs still fit less than 10% of the training answers after
frequency refinement. Complete state inspection rules out simple state collapse
on those training records: the learned states remain nonconstant with large
norms. This does not establish that the states encode the right recall relation
or are linearly separable. A controlled next step is to revisit every learned
readout across the training stream. The existing component search fits each
readout on its proposal batch and then keeps it unchanged.

Let f_j be already learned CM98 components, with learned initial states,
transitions and frequencies. Let w_j be nonnegative mixture weights summing to
one and let B be the capacity scale. For every component j and original class c,
retain a complete row u_{j,c} in C^2 with

    rowSize(u) = |u_1| + |u_2| <= 1.

Replace only the class readout of f_j by the actual complex linear map
z -> u_{j,c,1} z_1 + u_{j,c,2} z_2. The complete ordered hidden execution is
unchanged. Every old readout can be reconstructed from its two basis values;
no coordinate or imaginary term is omitted. If the content satisfies Unit and
the new rows satisfy their explicit budgets, every replaced component is Unit.

The set of all such rows, across every component and original class, is convex
in their real and imaginary coordinates. The complete executed score map

    S(u)(x,c) = sum_j w_j B Re(row(u_{j,c})(run(f_j,x)))

is real linear. Consequently ordinary full-class cross-entropy and its indexed
mean over any nonempty minibatch are jointly convex on this readout domain.
Indices preserve duplicates and short tails. The resulting functions still
belong to the original learned recurrent domain. This is a conditional
optimization step on already learned content; it does not claim that independent
transition/readout factors become jointly convex or prescribe a fixed feature
map for the architecture.

Use the actual radial feasibility rule

    projectRow(u) = u                       if rowSize(u) <= 1,
                   u / rowSize(u)          otherwise.

Both complex coordinates scale together. It leaves every inside row unchanged
and always returns a row of size at most one. Applying this rule to every
numerically displaced row therefore gives an admissible model. Any accepted
convex interpolation with the previous rows remains admissible as well. This
rule is not asserted to be the Euclidean projection, and feasibility does not
imply descent or convergence.

Both signed original-class CE probes remain actual feasible points of the
full CM98 family, by the existing capacity-reserving bias construction applied
to S(u). Those probes need not belong to the conditional readout slice itself.
Forward-only scalar differences remain finite loss evaluations, not exact CE
derivatives. No readout-separability, unseen-label identification, training-fit,
optimizer-convergence or 99% accuracy guarantee is assumed. Every conditional
statement is instantiated with explicit nonzero complex rows and the existing
nonconstant positional witness.

## CM101. Radial readout normalization can stall away from the CE optimum

The longer CM100 refits improve training accuracy to76.6–79.5%, but gains slow
and novel accuracy remains10.2–15.4%. This does not certify an optimum of the
conditional convex readout problem. Here is an independent ordinary-CE
counterexample to interpreting radial feasibility as an optimizer guarantee.

Use the complete two-state complex family with zero frequencies, initial state
s=(1,1/2), and object transitions T_true=Id and T_false=-Id. Thus the distinct
original prefixes [true] and [false] execute to s and -s. Their next objects are
true and false. These are admissible, nonconstant learned-content parameters
used only as a proof witness, never as benchmark initialization or task features.

Let r=(2/3,1/3), and take class rows u_true=r, u_false=-r. Each complete row has
complex absolute sum one. Actual scores have correct-class logit5/6 and wrong
logit-5/6. For either record b, write q_b for its positive wrong-class softmax
probability. The ordinary-CE row gradient, in real/imaginary coordinates, is

    g_{b,c} = (probability_c - 1[c=b]) conjugate(state_b)
            = -(3/2) q_b u_c.

The proof connects this explicit expression to the actual CE first-order
inequality for arbitrary replacement rows, retaining both complex coordinates.
For every nonnegative learning rate eta,

    u_c - eta g_{b,c} = (1 + (3/2) eta q_b) u_c.

CM100 radial normalization returns u_c exactly, because rowSize(u_c)=1 and the
multiplier is at least one. Every interpolation between the old row and that
proposal is unchanged as well. Both oppositely labeled records have this
failure; it is not caused by conflicting labels on one identical prefix.

Replace the rows instead by v_true=(1,0), v_false=(-1,0). They obey the same
budget and retain exactly the same learned content. Their actual logits are
1 and-1. Ordinary binary CE at correct margin2m is log(1+exp(-2m)), which is
strictly smaller at m=1 than at m=5/6. The conditional loss is convex by CM100,
yet the radial row update is fixed at a strictly worse feasible point.

This is an optimizer implication counterexample, not a claim that the measured
stress endpoints equal this witness, that their raw accuracy can be repaired by
this fact alone, or that unseen labels become identified. The native finite-query
capacity shrink still changes mixture weights; the theorem concerns the row
update at its current centre, not a fixed point of every part of the trainer.
It motivates checking a true constrained optimization step before drawing a
capacity conclusion from the stalled training loss. No new sorry is introduced.

## CM102. Exact Euclidean projection of complete complex readouts

CM101 proves that radial feasibility alone can leave a strictly worse CE point
unchanged. Replace only that numerical row rule; keep the actual complex
readout domain |u1|+|u2|<=1, all content paths and every original class.

For a proposed complete complex row z, let r1=|z1| and r2=|z2|. Compute the
two projected magnitudes s as follows, including the displayed boundary choices:

    r1+r2 <= 1:       s=(r1,r2)
    otherwise r1-r2 >= 1: s=(1,0)
    otherwise r2-r1 >= 1: s=(0,1)
    otherwise:       s=((r1-r2+1)/2,(r2-r1+1)/2).

The projected complex coordinate is (s_i/r_i) z_i. At r_i=0 its magnitude s_i
is zero as well, and the coordinate is zero. An implementation can return an
inside row directly and explicitly guard zero division. No original coordinate
is removed: a zero projected value remains stored and trainable.

The proof first establishes 0<=s_i<=r_i and s1+s2<=1. It then proves that s
minimizes (r1-a)^2+(r2-b)^2 over all a,b>=0 with a+b<=1. In the first dominant
case the excess cost factors as

    (1-a)^2+b^2+2(r1-1-r2)(1-a)+2 r2(1-a-b) >= 0.

The second case is symmetric. When both magnitudes are active, put
tau=(r1+r2-1)/2. The excess is

    (a-s1)^2+(b-s2)^2+2 tau(1-a-b) >= 0.

For arbitrary complex competitors v_i, the reverse triangle inequality gives
(r_i-|v_i|)^2 <= |z_i-v_i|^2. The displayed phase-preserving projection attains
equality with |z_i-p_i|=r_i-s_i. Thus the actual complete complex row p is a
closest point for the Euclidean squared distance
|z1-p1|^2+|z2-p2|^2. This is a proved optimization property of the explicit
formula, not an assumed projection operator. It leaves every feasible row
unchanged and gives admissible complete class/component readout replacements.

For the actual nonconstant CM101 witness, the wrong-class probability is the
same positive q on both original records. Choose the positive training-only
step eta=4/(3q). The ordinary CE displacement sends the class rows to (2,1)
and(-2,-1). The new projection sends these to (1,0) and(-1,0), precisely CM101's
better feasible rows. Actual ordinary CE therefore strictly decreases on both
oppositely labeled prefixes, where radial normalization was fixed.

This proves the replacement and repairs that concrete optimizer counterexample.
It does not assert convergence for noisy minibatches or finite CE differences,
prove that native recall endpoints have the same failure, or identify unseen
answers. Original-split measurements in both training modes remain necessary.

## CM103. A complete training-only conditional readout optimality gap

CM102 repairs the projection counterexample, but the original recall runs
still fail. Falling training CE does not certify the conditional optimum.
Before attributing the failure to learned states or their capacity, derive an
explicit gap for the actual CM100 full-class mean CE on all training records.
This bound holds for any finite learned component family and all its readouts.

For two complete complex vectors u,a, define

    pairing(u,a) = Re(u1*a1 + u2*a2).

The row constraint |u1|+|u2|<=1 implies pairing(u,a)>=-max(|a1|,|a2|).
This lower bound is attained: choose a coordinate of largest magnitude, use
-conjugate(a_i)/|a_i| in that coordinate, and zero in the other. At a tie choose
the first coordinate; when both coefficients are zero return zero. Both
coordinates remain in the domain, even when the minimizing value is zero.
For every component j and class c this gives an explicit feasible minimizing
table, with total linear cost -sum_jc max(|a_jc1|,|a_jc2|).

Let i index all N=n+1 training records, including duplicates and short tails.
Keep the actual learned initial states, full ordered transitions, rotary
frequencies, capacity B and component weights w_j. Let h_ji be their actual
complete complex state on prefix x_i. For current rows U compute all original
logits z_ic=sum_j w_j B pairing(U_jc,h_ji), ordinary softmax p_ic, and
residual r_ic=p_ic-1[c=target_i]. Define the unconjugated coefficient

    A_jc = (1/N) sum_i (w_j B r_ic) h_ji.

This uses every original class, record, component and complex coordinate.
Finite-sum rearrangement derives, rather than assumes, the full contraction
identity

    sum_jc pairing(V_jc-U_jc,A_jc)
      = (1/N) sum_ic r_ic (z_ic(V)-z_ic(U)).

Ordinary full-class CE's first-order inequality then gives the actual mean
loss inequality L(V)>=L(U)+sum_jc pairing(V_jc-U_jc,A_jc). Combining it with
the explicit support minimum defines the computable gap

    G(U) = sum_jc pairing(U_jc,A_jc) + sum_jc max(|A_jc1|,|A_jc2|).

For feasible U, G(U)>=0. For every feasible replacement table V,
L(U)-G(U)<=L(V), hence L(U)-L(V)<=G(U). In particular a zero gap proves
conditional optimality. The gap equals the negative linear change toward
the explicit minimizing table; a positive gap therefore supplies a feasible
direction with strictly negative actual first-order contraction. This does
not claim that an arbitrary finite step decreases CE.

All statements concern exact real arithmetic and the conditional slice on
already learned content. They neither freeze the architecture permanently nor
certify global factor discovery, unseen-label identification or raw accuracy.
A future floating-point gap is a diagnostic until its numerical error is
bounded. An analytic residual diagnostic of a forward-only endpoint is not a
forward-only training update. No validation or test labels enter this bound.

The hypotheses are inhabited using the existing nonconstant CM98 positional
witness, weight one, B=16, and class rows (i,0) and(-i,0). Its state on
[true,false] is (0,-i), so both actual logits are zero. Two indexed copies
with opposite targets have A=0 and G=0; this is a genuine conditional CE
minimum despite the conflicting labels. Retaining just the true-target record
instead gives G=16 and a strict negative first-order direction. These examples
prove the premises are realizable without assuming separability or an optimum.

## CM104. Joint capacity allocation on complete learned readouts

CM103 fixes the mixture weights as well as the learned recurrent content. Its
measured gaps therefore cannot obstruct improvements obtained by changing those
weights. Keep all learned initial states, transitions and frequencies, but
absorb each nonnegative native mixture weight into its complete class rows.
For finite components J and a nonempty finite original vocabulary O, write

    s_j(U) = max_c (|U_jc1| + |U_jc2|),
    K(U) = sum_j s_j(U),                 K(U) <= B.

Every row, class and component remains a variable, even at zero weight. The
score is sum_j Re(row(U_jc,h_j(x))), where h_j is the actual ordered state.
The shared capacity set is convex in all real and imaginary weighted-readout
coordinates. Ordinary full-class CE and every nonempty indexed mean are convex
on it. This is a conditional coordinate formulation, not joint convexity of
independent transition, readout and frequency factors.

For B>0, reconstruct an actual CM98 model explicitly: give component j native
weight s_j(U), replace its readouts by U_jc/s_j(U), and put unused capacity
B-K(U) on the zero component. A zero block has only zero rows; normalization
returns zero and reconstructs it exactly. The normalized weights sum to one
after division by B. All original content and full complex coordinates remain,
and reconstructed scores agree on every immutable-object sequence. Conversely,
any complete unit readouts with nonnegative weights of total native mass at
most B produce a table in this shared capacity domain with exactly their scores.

On indexed training records, form the actual full-class residuals r_ic=p_ic-
1[c=target_i] and coefficients A_jc=(1/N)sum_i r_ic h_ji. Native weights and B
are already in U, so they must not multiply A again. The complete first-order
contraction is derived by CM103 specialized to these weighted rows.

Let d_j(A)=sum_c max(|A_jc1|,|A_jc2|) and D(A)=max_j d_j(A), assuming J is
nonempty for this maximum. For every feasible U, its linear contraction is
at least -B D(A). An explicit minimizer selects a component attaining D,
assigns B times CM103's phase-normalized best row to each of its original
classes, and assigns zero rows to the other components. Every coordinate is
still present; this is one feasible value, not deletion or a reduced family.
Finite maximum attainment proves existence of the selected component rather
than assuming an optimization oracle. At zero coefficients any tied component
works, and all minimizing rows are zero.

The resulting actual mean-CE gap is

    G_joint(U) = sum_jc Re(row(U_jc,A_jc)) + B D(A).

It is nonnegative on the shared capacity domain and bounds every feasible
conditional CE improvement, including arbitrary capacity reallocation among
the retained learned components. A zero gap certifies this enlarged conditional
minimum; positive gap gives a feasible target with negative first-order
contraction. Neither statement supplies unseen labels, a raw-accuracy bound,
global factor discovery, finite-step descent or a floating-point certificate.
Forward-only updates still require actual finite feasible CE queries.

An explicit counterexample uses two retained bounded components: an identity
transition with constant state (1,0), and CM98's ordered component with states
(0,-i) on [true,false] and (0,1) on [false,true]. The two targets are respectively
true and false. Give the constant-state component both class rows (0,i), the
ordered component its original rows (0,i) and (0,1), and native weights 16 and 0.
Every actual logit is zero. Balanced residuals cancel on the constant state;
the zero weight suppresses every old CM103 coefficient of the ordered component.
Consequently the old fixed-weight gap is exactly zero with nonzero complex rows.

Move the same capacity sixteen to the ordered component without changing either
state function. Its correct-class logit becomes sixteen and its other-class
logit zero on both indexed records. The actual target probability rises from
1/2 to exp(16)/(exp(16)+1), so ordinary CE strictly decreases on both records
and their full mean. The joint gap at the initial table is therefore strictly
positive by its proved lower bound. This is a counterexample to interpreting
fixed-weight optimality as joint-capacity optimality, not a contradiction of
CM103 or a benchmark result. For a separate zero-joint-gap witness, retain two
copies of [true,false] with opposite targets: both complete coefficients cancel
and the joint gap is zero, without deleting the conflicting indexed records.

## CM105. Complete coefficient error and minibatch descent failure

The CM104 experiment lowers every updated minibatch's CE but increases the
complete training mean. Convexity of the complete weighted-readout domain does
not equate a current minibatch coefficient with the full training coefficient.
Keep the same finite complete classes/components and shared capacity K(U)<=B.
For any complete coefficient table A, retain the support
D(A)=max_j sum_c max(|A_jc1|,|A_jc2|), with nonempty components and classes.

This support is symmetric and subadditive, so |D(A)-D(H)|<=D(A-H). Every feasible
table satisfies |<U,E>|<=B D(E); two feasible tables therefore satisfy
|<V-U,E>|<=2B D(E). These statements include every original complex coordinate,
including components currently at zero capacity. They need no optimizer oracle.

Define the estimated gap G_H(U)=<U,H>+B D(H), where H may be a current minibatch,
finite-query or retained estimate. At feasible U it is nonnegative for every H.
For actual indexed complete-training coefficient A(U),

    |G_A(U)-G_H(U)| <= 2B D(A(U)-H),
    L(U)-G_H(U)-2B D(A(U)-H) <= L(V)

for every shared-capacity feasible V. If V_H is CM104's explicit finite support
minimizer for H, its actual complete-training first-order contraction obeys

    <V_H-U,A(U)> <= -G_H(U)+2B D(A(U)-H).

Thus G_H(U)>2B D(A(U)-H) supplies a strictly negative actual first-order direction.
A positive estimated gap by itself does not. No bound on estimator error is
assumed free: the error here is its actual full-table discrepancy. The result
does not claim unbiasedness, reduced variance, finite-step CE descent, convergence,
rounding certification, raw accuracy or unseen-label identification. Any subsequent
minibatch implementation must distinguish its estimate from A(U).

The error condition is inhabited nontrivially. Take the CM98 ordered component,
capacity sixteen, full imaginary rows (i,0) and(-i,0), and the true-target record
[true,false]. Scores are zero and the actual coefficient has class rows
(0,i/2) and(0,-i/2). Set H=3A/4: G_H=12, D(A-H)=1/4, and the correction is8.
The error is nonzero, yet the strict-direction condition holds.

A finite-step counterexample keeps both distinct CM104 learned components and
both opposite prefixes with their respective first-object targets. Start at
CM104's parked table, whose actual scores are zero despite imaginary readouts.
Put native capacity sixteen on class rows (1,0) and(-1,0) of its constant-state
component to form a feasible bias table, then take the halfway interpolation
from parked to this bias table. The resulting complete rows retain imaginary
coordinates and obey the same shared budget. Actual logits are8 for true and-8
for false on both prefixes. CE decreases on the minibatch containing training
index0, but the complete two-record mean rises from log2 to
log(exp8+exp(-8))>log2. The full class denominator and both indexed records remain.
The strict comparison uses exp(t)+exp(-t)>2 for every positive contrast t.
The ordered component remains available; CM104 already proves a different
feasible reallocation improves both records. This counterexample concerns a
minibatch descent inference, not impossibility of fitting or generalizing.

## CM106. Retained coefficients while learned content changes

CM105 permits an arbitrary complete estimate but does not justify treating a
cache made under old learned content as a current coefficient. Keep every
original indexed supervised occurrence i, including duplicates. At current
content f let z_ij be its actual ordered state on prefix x_i. A stored state q_ij
may come from a different historical model g_ij for each occurrence/component.
For stored full-class signals s_ic define H_jc = (1/N) sum_i s_ic q_ij.
There is no assumption that s is a current residual or that q is a current state.

For two CM98 models, define the executed transition discrepancy along a prefix
recursively by D_p([])=0 and
D_p(o::xs)=norm(step_f(p,o)-step_g(p,o))+D_(p+1)(xs).
Contractive actual operators and a historical starting state of norm at most
one give
  norm(run_f(p,v,xs)-run_g(p,w,xs)) <= norm(v-w)+D_p(xs).
This follows by splitting each transition difference into the propagated state
error and the changed operator applied to the historical state. Starting at
the learned initial states gives rho(f,g,xs)=norm(init_f-init_g)+D_0(xs).
The operators include both actual rotary conjugations, so changing frequencies,
transitions or initial coordinates is covered. Each original occurrence keeps
its position; no label or prescribed answer feature enters rho.

For arbitrary complete state and signal tables, the mean coefficient is linear
in either table separately. Refreshing an indexed subset replaces its old
signal contribution by its new one, divided by the original N, not the batch
size. Transporting stored signals to changed states adds the complete mean
sum_i s_ic*(z_ij-q_ij)/N. Both identities retain all components and classes.
Identical prefixes at different indices remain separate contributions.
When both states and signals refresh on the minibatch, the increment is the
indexed mean of (newSignal*newState - oldSignal*oldState) on just those records.
Storing historical states therefore permits genuine content updates followed
by ordinary minibatch refreshes without silently reusing the old state as new.

Let r_ic be the actual full-class CE residual at the current complete model.
If norm(z_ij)<=1 and norm(z_ij-q_ij)<=d_i for all j, then the CM105 peak obeys
  peak(A-H) <= E,
  E = (1/N) sum_i [sum_c abs(r_ic-s_ic) + d_i*sum_c abs(s_ic)].
The proof uses
  r*z-s*q = (r-s)*z + s*(z-q)
before summing every class and occurrence. For historical CM98 models take
d_i=max_j rho(f_j,g_ij,x_i). This supplies the state term from actual parameter
changes; the signal term remains explicit. Thus G_H+2B E bounds conditional CE
improvement and G_H>2B E suffices for an actual negative readout direction at the
current content. It does not certify a finite content step or assume successful
optimization. A native learner changing content must either invalidate its
cache, transport every affected coefficient exactly, or account for both errors.

This extension does not turn stored signals into unbiased gradients, bound
their stochastic error for free, prove finite-query rounding or identify unseen
answers. The native CM105 comparison remains partial: retained-small improves
original novel accuracy but is far below 99%. Content learning and generalization
still require actual training and original held-out evaluation.

Nonzero witnesses use the CM98 quarter-turn model and its zero-frequency
counterpart: on [true,false] their actual states are (0,-i) and (0,1), so the
content-change term cannot be omitted. For a separate strict example, keep the
quarter-turn content, CM105 imaginary current rows and a single true target,
but retain class signals (-3/8,3/8) instead of the actual (-1/2,1/2). The full
error budget is 1/4 and the estimated gap is 12, so correction 8 is strictly
smaller. An indexed three-occurrence refresh also retains the denominator 3.

## CM107. Learned norm-preserving mixing and the choice of state norm

The CM106 native comparison correctly accounts for changing content, yet its
finite coordinate revisits do not repair original key-value recall. This does
not prove a capacity obstruction or identify its cause. Check a separate
architectural restriction before another training change: CM97 uses the maximum
norm on two complex states, so its unit transition rows have absolute sum at
most one. This differs from a Hilbert operator-norm bound and can exclude a
perfectly norm-preserving mixing matrix.

Use H = WithLp 2 (complex x complex), with squared norm |z0|^2+|z1|^2.
Keep both complex initial coordinates, all four complex entries of every
original-object transition and both complex coefficients of every original
class readout. Changing the norm changes the admissible set, not the numerical
coordinate representation. Full complex linear maps still reconstruct exactly
from their two columns. Require initial, transition and readout operator norms
at most one in this Hilbert geometry; CM95 then bounds actual ordered execution
and full-class outputs at every length. Readout squared coefficient norm at
most one is a sufficient complete-row capacity condition.

For an exact nonzero witness, take the real-coefficient complex-linear matrix
R=((3/5,-4/5),(4/5,3/5)). It preserves the squared Euclidean norm of every complex
state, including both imaginary coordinates. Consequently all its repeated
executions preserve that norm exactly. But its two absolute row sums are7/5,
and it sends (i,i), of maximum norm1, to (-i/5,7i/5), of maximum norm7/5.
It is excluded by the old maximum-norm transition budget, despite its flat
unit singular values. No prescribed position frequency is used in this example.

Apply the actual CM102 phase-preserving Euclidean row projection to both rows
of R. The result is Q=((2/5,-3/5),(3/5,2/5)). Its squared Hilbert norm is exactly
(13/25) times the input squared norm. Repeating this projected transition n
times gives squared state norm (13/25)^n times its initial value, strictly less
than the unprojected value for every nonzero state after a positive number of
steps. This is an exact example of the geometry's restriction, not a bug in
CM102: Q really is the closest matrix under its original row constraints.
It also distinguishes equal singular values from singular values equal to one.
Flatness without a scale condition does not prevent depth-dependent decay.

The converse inclusion also fails: the broadcast matrix ((1,0),(1,0)) has old
maximum-norm operator bound one but sends the Hilbert unit state (i,0) to
(i,i), whose squared Hilbert norm is two. Preserve the complete old learned-
rotary family rather than replacing its norm and silently dropping parameter
points. Enlarge the joint function domain to
the convex hull of all old CM98 generators and all bounded Hilbert-state CM95
generators. Every old function remains feasible, and every new admissible
Hilbert recurrence is feasible. Its finite representations retain an explicit
branch tag per learned component; no fixed feature table, numeric interpretation
of objects or answer program enters either branch. The uniform output budget B
survives finite mixtures, and ordinary complete-class CE is convex in joint
scores. The already available nonzero original-class bias generators remain
feasible, so finite signed CE queries keep their exact logits and capacity
accounting even when the current model mixes both branches.

This is functional convexity, not joint convexity in independently varied
matrix factors or a global component-search oracle. The witness supplies no
unseen labels and is not a benchmark initialization. The native results do not
yet establish that projection-induced decay is the cause of their failure;
learning under the enlarged geometry needs a separate frozen comparison.

## CM108. A complete numerical Hilbert matrix bound

CM107's operator budget needs an executable full-matrix rule. For every complex
two-by-two transition M, form its complete column Gram entries
  a = |M00|^2 + |M10|^2,
  d = |M01|^2 + |M11|^2,
  b = conjugate(M00)*M01 + conjugate(M10)*M11.
These use all real and imaginary coordinates. Define
  L = (a+d+sqrt((a-d)^2+4|b|^2))/2,  N = sqrt(L).
The actual squared Hilbert execution is
  |M z|^2 = a|z0|^2+d|z1|^2+2 Re(b*conjugate(z0)*z1).
Prove L>=a,d>=0 and (L-a)(L-d)=|b|^2. The resulting weighted quadratic
inequality bounds every actual output by L times the input squared norm.

This is a sharp bound, not an assumed eigenvector certificate. If L=a, then
b=0 and the first basis state attains it. Otherwise L-a>0 and the explicit
nonzero complex state (b,L-a) attains it. Consequently the actual Hilbert
operator norm is N for every full complex matrix, including zero and diagonal
matrices. No coordinate, zero entry, singular direction or class is removed.

Normalize the complete matrix by the real factor 1/max(1,N). Its actual
operator norm is at most one. Every already feasible matrix is unchanged,
including CM107's rational unitary mixing; matrices outside the ball are
scaled uniformly. This is a feasible radial rule, not a closest-point claim
in the matrix Frobenius metric or a global optimizer. It does not replace the
separate old branch or alter its saved parameters, original-class CE, inputs,
outputs or finite-query identities. Native evaluation must still check finite
arithmetic and its measured residuals; a real-number identity is not a
floating-point rounding certificate or an unseen-answer theorem.

## CM109. Exact complete-class reuse for finite output queries

Growing the finite learned mixture should not spend a full vocabulary sum
on every output-coordinate query when only one class score changes. This is
an algebraic reuse of ordinary cross-entropy, not an alternative loss or a
derivative. Let z be the complete original logit vector, target t, changed
class c, S=sum_d exp(z_d), and p_c=exp(z_c)/S. An existing original class
witnesses S>0. Define shift(z,c,delta)_d=z_d+1[d=c]*delta. Then
  S(shift(z,c,delta)) = S + exp(z_c)*(exp(delta)-1)
                    = S*(1+p_c*(exp(delta)-1)).
The factor is strictly positive for every real delta because the updated
full partition is positive. Therefore the actual ordinary full-class loss is
  CE(shift(z,c,delta),t)
    = CE(z,t)+log(1+p_c*(exp(delta)-1))-1[t=c]*delta.
For two actual changes plus and minus, subtract these identities. Their
cached CE centres cancel, leaving the two logarithms and
-1[t=c]*(plus-minus). No symmetry or infinitesimal limit is assumed; feasible
projection of signed parameter perturbations can give asymmetric changes.

Every original class still participates in S and CE. The formula supplies
only the same two finite loss values at the actual feasible query points.
It supplies no analytic CE residual to forward-only training. A native
implementation may use expm1 and log1p for these differences and must compare
them with full ordinary CE evaluations. This exact-real identity gives no
certified floating rounding, stochastic convergence, factor oracle or unseen
answer. Nonzero binary and three-class examples exercise target and competing
class changes. It applies to the complete old and Hilbert branches alike.

## CM110. State width is not the same as the number of plain learned blocks

A separate structural check is possible while the prospective growth cohort
trains, without inspecting its held-out outcomes. A plain two-complex-state
block has arbitrary complete matrices A, B and C shared across input positions.
The two-by-two commutator K=AB-BA is traceless, and its square is scalar.
Consequently C K^2-K^2 C=0. Executing input words from left to right reverses
the order of their matrix products. Define the actual signed score difference
  D(f;a,b,c) = f(ababc)+f(babac)+f(cabba)+f(cbaab)
              -f(abbac)-f(baabc)-f(cabab)-f(cbaba).
Here each letter is an original opaque object, and f is one original class's
real score. Prove D=0 for every complete complex initial state, every full
complex transition matrix and every complete complex readout. No norm,
flatness, balancedness, sign or real-coordinate restriction is needed. The
proof expands actual Pair.operator execution, rather than assuming a matrix
identity as an extra input. The complete coordinate representation extends it
to every complex linear map on the two-state space.

D is a linear functional of actual sequence scores. Thus any finite mixture,
including arbitrary signed weights, duplicate components and zero weights,
still has D=0. If a held position-dependent parent g is included with mass r,
the resulting score has exactly D=r*D(g). No claim that the rotary parent
satisfies the plain identity is made. Training more plain blocks can change
many predictions, but cannot introduce a new independent violation of D.

This is a strict expressivity restriction even under a unit capacity bound.
In three complex coordinates, choose A=E01, B=E10 and C=E12. Their complete
linear maps are (x,y,z)->(y,0,0), (x,y,z)->(0,x,0), and
(x,y,z)->(0,z,0). Start from e2 and read coordinate1 for output class0;
other output classes have zero readout. Every transition and readout is a
contraction in the product norm. Each transition and readout also contracts
the complete squared Euclidean energy |x|^2+|y|^2+|z|^2, with initial energy1.
Actual execution therefore has bounded scores at every length. Nevertheless
K^2=diag(1,1,0), C K^2-K^2 C=-C and the eight actual word scores give D=-1.
No finite mixture of plain two-state blocks equals this bounded three-state
function on all words. The witness is an expressivity counterexample, never
a supplied benchmark program or initialization.

This theorem does not establish that the original MQAR examples violate the
identity, does not lower-bound their classification accuracy, and does not
prove that a larger state can be learned from their training data. It separates
an architectural limit from norm geometry, mixture size and optimization.
The fixed sixteen-run growth comparison must finish and retain all outcomes;
this proof does not justify selecting or changing its protocol after results.

The separation also has a quantitative score consequence. If two sequence
score functions differ uniformly by at most epsilon, their D values differ
by at most8*epsilon, by the eight actual termwise error bounds. Since every
plain two-state mixture has D=0 and the witness has D=-1, any uniform score
approximation to that witness has epsilon>=1/8. The hypothesis is inhabited,
for example by one zero component with unit weight and epsilon1, because all
witness scores have absolute value at most1. This lower bound is independent
of the number of mixture components and does not assert a top-one accuracy
limit on the original stress tests.

## CM111. Shared learned matching with a learned relative-position link

Task402 separates two failures: growth raises original stored-value prediction
to about 80% on novel MQAR examples, but the choice between distinct stored
values remains about half. The following extension changes the sharing of
learned parameters. It does not deduce an answer from a norm bound.

Let K,V map every opaque object into a complex inner-product space, with
norm at most one. Both maps are learned. Query and memory keys share K;
input values and complete output-class readouts share V. A real scale t and
one real relative-position potential r(d), d an integer, are learned too.
For a past prefix x indexed by j and a final query object q, define

    a_j = softmax_j(t Re <K(q),K(x_j)>),
    b_jk = softmax_k(r(k-j)),
    h = sum_jk a_j b_jk V(x_k),
    z_c = Re <V(c),h>.

Every visible position and every original class is present. The runtime takes
only the object sequence; its last object is q and the preceding objects form
x. Empty memory gives zero. There are no supplied key/value roles, chosen
offset, token equality feature, task label program or fixed output vectors.
The sharing is an architectural restriction. It is not independent Q/K/V
parameter convexity. The complex inner products use all coordinates.

The path weights are nonnegative and sum to one for nonempty memory, so
|z_c| <= 1 at every prefix length. Scaled learned heads and a previously
learned bounded base can therefore form a convex hull at the same score
capacity B. Ordinary complete-class CE is convex in that functional state.
Searching for the learned factors remains a separate optimization problem.
This is an output capacity bound, not a finite-sample complexity bound by itself.
The complete vector map v -> v/max(1,||v||) enforces both learned-map norm
restrictions and fixes every feasible vector. It preserves all complex
coordinates. This is an exact-real feasibility result, not a floating-point
or optimizer convergence guarantee.

For explicit binding, suppose j is a past occurrence of q. If its actual
learned key score exceeds every other score by g, then
1-a_j <= n exp(-g). If the learned relative potential at k-j exceeds all its
other visible offsets by h, then 1-b_jk <= n exp(-h). These are bounds derived
from the literal softmax denominators, not hypotheses that attention already
selects the correct path. Thus

    1-a_j b_jk <= e = n(exp(-g)+exp(-h)).

Suppose the learned value/readout map separates the object x_k from each
rival c at that object by m >= 0:
Re <V(x_k),V(x_k)> - Re <V(c),V(x_k)> >= m.
The complete scaled output margin is at least

    B [m-(m+2)e],                         B >= 0.

The good path contributes its decoder separation; every other path has
margin at least -2 by the unit norms. A positive bound identifies x_k against
every original class, and the usual CE bound is
log(1+(|O|-1) exp(-B[m-(m+2)e])). Repeated occurrences of the same value are
allowed and can only make this conservative path argument less sharp.
If a bounded learned base is retained at weight 1-eta and the new head at
eta in [0,1], a new-head margin M gives mixed margin at least
eta M - 2(1-eta)B. The retained base is not assumed to help the new relation.

The same equations apply to unseen combinations of objects when these shared
geometric and relative-position conditions hold. They do not prove that
training establishes the conditions, identifies a unique relative relation,
or that an arbitrary task's correct next object is x_k. A copying relation is
additional structure, not a consequence of flatness. The native follow-up
must learn its relative potential from original training CE and keep all
parameter paths trainable; no offset may be supplied from MQAR roles.
A two-object complex example with separated learned vectors and a nonconstant
relative potential witnesses nonempty positive-margin conditions. It is a
mathematical example, never a benchmark initialization or answer program.

## CM112. A two-object input removes query dependence from CM111

On the original prefix `[a,b]`, the query is `b` and the memory contains just
`a`. Both normalized rows have a single entry. The one path has mass one for
every possible learned key table, key scale and relative potential. The whole
unscaled class-score function is therefore

    score([a,b], c) = Re <V(c), V(a)>.

The native scale multiplies all these scores by the same capacity. For a
fixed first object, every second object gives the same scores and the same
prediction under a fixed deterministic decoder. This holds during fitting
and for every endpoint, including any subsequent change to the weights.
If two such prefixes require different answers, at least one prediction must
be wrong. Distinct Boolean answers and the existing nonzero complex witness
instantiate the contradiction for a deterministic two-class readout.

The original `arithmetic-add` data consists of two operands modulo 97 with
one label at the second position. At a fixed first operand, distinct second
operands have distinct answers. Grouping each original fingerprint-checked
split by first operand bounds the number correct by the sum of the largest
answer counts in those groups. This gives 97/4,704 on train and 97/4,705 on
validation. These are upper bounds on every trained CM111 single head. The
native regression also compares its full raw logit vector across every second
operand in both original splits. A shared architecture that covers arithmetic
needs a learned path from the final object to the output even with one
memory position. Ordinary CE fitting and the flat capacity bound alone
cannot supply that missing dependence.

## CM113. Composed learned retrieval and a two-operand phase interaction

CM112 identifies a lost operand on prefixes of length two. Task 406 also
found a second independent failure. Following two links in a table changes
the object used for the second lookup. The original table can contain an
object both as a key and as a value. A shared learned operator can distinguish
its occurrences through their preceding raw objects and their positions.

For each opaque object, learn three width-$D$ vectors of real angles with
$D>0$. They supply the key $K$, tied value/readout $V$, and rotation $P$. Their complex coordinates
are $e^{i\theta}$. Also learn an initial angle vector, a scalar bias for
each possible predecessor including the boundary, a key scale $t$, a recency
coefficient $s$, a table of signed relative potentials $r$, and depth and
product gates $h,u\in[0,1]$. For memory objects $x_j$ before the final
query $q$, let the means below average over all $D$ coordinates.

```
a(o,j) = softmax_j[t mean cos(K(o) − K(x_j)) + bias(prev_j) + s j]
b(j,k) = softmax_k[r(k−j)]
M(o,k) = sum_j a(o,j) b(j,k)
V1     = sum_k M(q,k) exp(i V(x_k))
V2     = sum_k,l M(q,k) M(x_k,l) exp(i V(x_l))
H      = (1−u) [(1−h) V1 + h V2] + u exp(i P(q)) V1
z_c    = B mean Re[conj(exp(i V(c))) H]
```

Products of coordinate vectors are componentwise. Every position and class
participates. The initial phases supply the empty-memory state. An empty
sequence decodes this state directly and a single object rotates it before
decoding. All predictors share the same computation, with task identity
confined to the data adapter. On `[a,b]`, the product route has actual score
`mean cos(V(a)+P(b)−V(c))`. One admissible width-one setting gives class
`(a+b) mod 97` strictly higher score than every rival on every one of the
9,409 original pairs. Set both tables to $2\pi a/97$ in this existence
proof. Equality of a rival's cosine score to one would require its difference
from the sum to be an integer multiple of 97. Native training initializes the
tables independently of the numerical names of the objects.

Both Markov reads preserve total mass. Their unrotated decoder has
diagonal one for every set of learned angles, including a class that never
appears as a positive training answer. Every actual score has magnitude
at most $B\ge 0$. The convex hull of these complete score functions inherits
this bound and ordinary full-class CE is convex on that functional domain.
The optimizer updates angles, gates and routing coordinates directly.

For one proposed hop from source $i$ to destination $k$, let $g,h_r$ be
lower bounds on the actual key and relative score gaps against every other
visible position. CM111's normalized softmax bound gives
`1−M(o,k) ≤ e = n(exp(−g)+exp(−h_r))` at memory length $n$.
For a second such hop using $x_k$ as its query and ending at $l$, the
missing mass on the joint path is at most $e_1+e_2$. If the tied decoder's
unrotated separation at $x_l$ against every rival is at least $m\ge 0$,
the complete scaled margin is at least

```
B { w [m − (m+2)(e1+e2)] − 2(1−w) },    w = (1−u) h.
```

A positive bound identifies the final object and bounds its ordinary CE
over all classes. In the explicit three-object witness the memory is
`[A,B,B,C]` and the query is `A`. With both gates interior, the first read
prefers `B` to `C`, and the complete model returns `C` against every rival.
The shared predecessor bias distinguishes the two occurrences of `B`. For
occurrences with the same preceding object, the key-score difference equals
the recency coefficient times the position difference. At zero recency those
occurrences tie. The path certificate then needs other evidence or a common
destination for the tied paths.

Perfect matching on observed prefixes alone cannot identify an unrestricted
unseen answer. Two admissible Boolean phase tables agree on every class at
both two-object training inputs ending in false, and select different
answers at `[false,true]`. This is an ambiguity inside the executed family.
The positive addition and two-hop constructions establish representability.
Original held-out performance still requires a separately frozen fit.

## CM114. What the two-hop gate can learn from ordinary cross-entropy

The native CM113 fit initializes every learned relative potential to zero.
More generally, if these potentials share any constant value, the relative
softmax from each source position is uniform. Summing over its normalized
content match gives the same uniform destination law for every query. For
a memory of positive length both unrotated reads therefore return the same
mean value. The complete score still depends on the learned value table,
the final object's rotation and the product gate.

This equality holds for arbitrary content weights and arbitrary depth, with
the other coordinates fixed. Even a simultaneous curve through the key map,
predecessor bias, key scale, recency and depth leaves all logits and hence
the ordinary full-class CE constant. Its directional derivative is zero.
At this initialization the observed labels provide no first-order signal
along those directions. Changing the relative potentials can break the
equality. The claim is specific to spatially uniform relative scores.

For a fixed prefix and all weights except depth held fixed, let z(c)
denote its actual scaled logit at depth h. Write R1(c) and R2(c) for the
unrotated first- and second-read scores. With capacity B and product gate u,
define D(c) = B(1−u)[R2(c)−R1(c)]. At a depth displacement a the new complete
logit is exactly z(c) + a D(c). If p(c) denotes the full softmax at the
current point, the derivative of ordinary CE for observed answer y is

    sum_c p(c) D(c) − D(y).

A target whose direction exceeds the predicted mean favors more depth on
that row. The minibatch averages these signed contributions, retaining
every supervised occurrence. The constrained update also depends on the
feasible interval and optimizer state. These identities supply a training
diagnostic at arbitrary learned parameters.

With the other weights fixed, full-vocabulary CE lies above its tangent
as a function of depth. Summing the tangent bounds over the training
records preserves their multiplicities. At depth zero, a nonnegative summed
training slope therefore certifies that zero minimizes this one-coordinate
training objective over the entire feasible interval. The argument allows
different prefix lengths and leaves joint changes in routing weights open.

If D(c) and D(k) differ, the observed score difference of c and k on this
prefix determines the depth with the other coordinates fixed. The same
difference is their full-softmax log odds. At least one of these two labels
produces nonzero CE slope. CM113's repeated-key example satisfies this
condition. Its unrotated first read favors intermediate object B over C,
its second read favors C over B, and the gate on the product is interior.
One observed correct label supplies the CE expression above. The score
contrast identification result assumes access to its numerical value.

## CM115. The depth gate in log odds

Keep the CM113 classifier and every physical read. Represent the depth by
a finite real coordinate z with h(z) = 1/(1+exp(−z)). This assigns positive
weight to each read inside the depth mixture. In the four-case experiment
the initial z is log 9, giving h = 9/10. The CM114 equality between the
two unrotated reads at uniform relative-position scores still applies, so
the initial raw depth derivative is zero.

For fixed other weights and an observed prefix, let S denote CM114's
full-vocabulary CE slope in physical depth, evaluated at h(z). The actual
CE derivative in z is S h(z)(1−h(z)). The factor is strictly positive and
at most one quarter. At initialization it is 9/100. Every nonzero physical
slope therefore has the same sign in log odds, although its magnitude
is smaller. Adam's subsequent moment normalization also affects the
step taken. Forward-only training measures losses at z+ε and z−ε before
taking a step in this same coordinate.

Between two raw coordinates z and t, class c changes score by
[h(t)−h(z)]D(c), with D the physical class direction from CM114.
The one-row CE lower bound is CE(t) ≥ CE(z) + [h(t)−h(z)]S. Its
displacement is in physical depth. Summing over the original rows shows
that a nonnegative complete training slope at z rules out improvement by
moving to any t ≥ z with all other weights fixed. Convexity of CE in the
physical mixture does not give convexity in the raw log odds.

When two class directions differ on an observed prefix, the measured
score contrast identifies the physical mixture weight and hence its
finite raw coordinate by strict monotonicity. The CM113 chain provides
an instance of that condition. These statements describe the optimization
coordinate and leave the learned routing and held-out accuracy empirical.

## CM116. Bounded output offsets in the two-read successor

For every original class $c$, append an independently learned scalar
$b_c$ to the model. On prefix $x$ the executed logit is
$z_x(c)=B s_p(x,c)+b_c$, where $s_p$ is the complete CM113 physical
score and $|b_c|\le D$. Since $|s_p|\le1$ and $B\ge0$, the absolute logit
is bounded by $B+D$, including at empty memory. The experiment uses
$B=D=16$ and starts every offset at zero.

At fixed sequence weights the loss for a labeled prefix is convex in the
offset vector. Along coordinate $k$ its derivative is the original
softmax probability of $k$ minus the indicator that the target is $k$.
Any offset change $v$ has CE at least the current CE plus the probability
mean of $v$ minus its target entry. Averaging preserves this lower bound
with every training occurrence at its original multiplicity. CM109 gives
the finite single-logit query used by the forward-only optimizer. The
joint fit also moves the sequence weights, where this convexity argument
does not apply.

Suppose only class $k$ moves by $t$ from its present logit. On every
observed row with target $k$, each rival minus target score must be
strictly below $t$. On each other row with target $y$, the difference
$z_y-z_k$ must strictly exceed $t$, and $y$ must already beat every
remaining rival. Together these conditions characterize strict correct
classification on all the rows. Bounded moves additionally require
$|b_k+t|\le D$. For two rows with targets 0 and 1 and respective
base scores $(0,2)$ and $(0,1)$, only the first row is wrong. A shift
of class 0 must exceed 2 on the first row and stay below 1 on the second.

With the sequence weights fixed, the vector of class contrasts on one
observed prefix determines every offset relative to an anchor class.
Adding a common constant leaves those contrasts unchanged on every
prefix. One observed winning label provides inequalities instead of these
numerical contrasts. With all sequence weights trainable, the CM113
example of identical observed scores and opposite unseen winners still
applies by choosing every offset zero.

## CM117. Learned object magnitudes in the product channel

Give every original object $o$ one trainable radius $r_o$ in $[0,1]$.
Use it on both sides of the CM113 product. In a nonempty memory with
first-read weights $w_j$ and final query $q$, the rotated class score becomes

$$
r_q\sum_j w_j r_{x_j}\,\mathrm{decode}(v_{x_j},\phi_q,c).
$$

The two unrotated routes still use the original normalized weights. The
complete logit uses the depth and product mixtures, scale $B\ge0$, and
the CM116 offset $b_c$ with $|b_c|\le D$. An empty prefix or a single
object uses the original boundary rule. Since the normalized read averages
terms of magnitude at most one, every complete logit has absolute value
at most $B+D$. A radius of one for each class recovers the original model.

On a two-object prefix $[a,q]$ both unrotated reads coincide. The product
term equals $r_a r_q$ times its old value. When only the radii change, the
change in an observed logit contrast between classes $c,k$ is the change
in $r_a r_q$ times $B$ times the product gate times the difference of
their rotated decodes. For a nonzero coefficient the numerical contrast
identifies the product. At $a=q$ the nonnegative radius is identified.
Winning labels supply inequalities. For a product-only two-class model,
take phase 0 for false and $\pi$ for true in the value and query tables,
give false an offset of $1/2$, and use $B=1$. Changing the radius of true
from one to zero gives identical training logits on $[\mathrm{false},
\mathrm{false}]$ and reverses the strict winner on $[\mathrm{true},
\mathrm{false}]$.

For a fixed pair, the logits are affine in the scalar product of its two
radii when the other coordinates are fixed, so ordinary CE is convex in
that scalar on $[0,1]$. A radius shared between two occurrences enters
quadratically. Optimizing all radii, phases and routes jointly requires
separate analysis. The native training-only intervention below measures
whether bounded changes could correct the observed original pairs. An
intervention is distinct from learning those values from scratch.

## CM118. Distance decay and the offsets observable from short prefixes

Let both normalized positional reads use the same effective potential
$u(\delta)=v(\delta)-a|\delta|$, where $v$ is a learned signed residual
and $a\geq0$. A model with all residuals trainable has the same finite
score family as CM117. The coefficient changes how a training update can
act on many distances together. The bound $|z_c|\leq B+D$ continues
to hold for every input prefix, including the model's own continuations.

In the positional row starting at $j$, write $w_{jk}$ for the normalized
weight of destination $k$. Its odds relative to staying at $j$ are

$$
w_{jk}=w_{jj}\exp\big(v(k-j)-v(0)-a|k-j|\big).
$$

If a finite set of destinations $F$ has distances at least $L$ and
residual gaps $v(k-j)-v(0)\leq M$, its total weight is at most
$|F|\,w_{jj}\exp(M-aL)$. This estimates the mass of the executed
row. The final winning class also depends on the content weights,
second read, rotation, radii and decoder.

A prefix with at most $H+1$ original objects has at most $H$ memory
positions. Both reads access only offsets $-H<\delta<H$. Agreement
of the effective potentials on that interval, with all other weights
fixed, implies identical complete class logits and CE for every such
training prefix. The restriction can matter on the very next length.
Take two objects with phase values zero and $\pi$, a uniform content
match, unit radii and one unrotated read. Set one signed residual to
$\log 3$ and all others to zero. Choosing $+1$ or $-1$ gives identical
scores on every prefix of length at most two and opposite strict
winners on the prefix $[\mathrm{false},\mathrm{true},\mathrm{false}]$.

Even observations at every length cannot distinguish $(v,a)$ from
$(v+t|\cdot|,a+t)$ when both coefficients are admissible. Their
effective potentials agree everywhere. At fixed residuals, the ordinary
training CE has distance derivative
$-\sum_\delta |\delta|\,\partial\mathrm{CE}/\partial u(\delta)$.
Each original training occurrence contributes with its original weight.
The CM117 conditional convexity result concerns the radius product on a
fixed pair. It supplies no joint convexity claim for these normalized
positional reads. [Training intervention](../../experiments/composed_retrieval/distance_decay/README.md)

## CM119. A signed geometric relative-position profile

The training-only CM118 measurements leave many offset tables compatible
with the original short copy examples. Replace the free table by two learned
adjacent levels $b_-,b_+$ and two nonnegative learned decays $a_-,a_+$.
Fix the central potential to zero. In direction $s$, at distance $h\geq1$,
use $b_s-a_s(h-1)$ in every positional row of both reads. This causal
rule uses only positions in the visible memory. Each object representation,
content comparison, gate, magnitude and readout retains its CM117 meaning.

For any source position $j$ and visible destination $k$, the positional
probability is the probability at $j$ times the exponential of the
effective potential at $k-j$. Moving one additional place into the same
tail multiplies the probability by $\exp(-a_s)$, at every sequence
length. The boundary changes the common row normalization. The complete
class logits retain the absolute bound $B+D$.

Suppose the physical probabilities of a five-position row centered at
its third position are known. The odds at distances $+1$ and $-1$
relative to the center identify the two adjacent levels. The odds at
$+2$ and $-2$ then identify the decays. Two profiles with identical
probabilities on that row coincide on every signed offset. Keeping the
other model weights fixed therefore gives identical complete predictions
on all prefixes. Ordinary training supplies next-object labels and fits
all parameters jointly. This identification result requires the stronger
observation of positional probabilities.

Only coordinates that can affect a legal input are allocated. A two-object
context has none, a three-object context has the two levels, and larger
contexts have all four. Zero initialization gives exactly the CM117 initial
weights on every positional row. The model learns four numbers at the
largest context in place of 257 independent signed-offset entries.

## CM120. Learned matching of preceding objects

In CM117 and CM119, two occurrences of the same candidate object have
the same current-key similarity to any given query. Their positional and
predecessor biases can favor one occurrence. The odds between them are
independent of the query, even when the two occurrences have different
preceding objects. This restriction applies to each normalized content
row before the two reads and positional transitions.

Use the existing learned key phases to compare two preceding objects.
Write $K(a,u)$ for their mean cosine difference, with value zero if
either object is absent. Add $tK(a,u)$ to the original content logit,
where $a$ precedes the query and $u$ precedes the candidate. The final
query uses the last object of its memory as its predecessor. An inner
query at position $i$ uses the object at $i-1$ when it exists. Both
reads share the scalar $t$ and all key phases. Every visible position
participates in normalization.

Let $j,k$ be occurrences of the same symbol with predecessors $u,v$.
For two query rows with predecessors $a,b$, their normalized weights obey

$$
w_{aj}w_{bk}=\exp\!\big(t[K(a,u)-K(a,v)+K(b,v)-K(b,u)]\big)w_{ak}w_{bj}.
$$

All other source-position biases cancel. In the memory $[0,1,1,1]$,
the inner queries at positions one and three both read object one.
With antipodal learned keys for zero and one, the multiplier is
$\exp(4t)$. Two numerical content rows with a nonzero bracket identify
$t$ at fixed keys. Ordinary training observes final next-object labels.

Both normalized reads and the radius-weighted product remain within
their original capacity. The full logit bound is $B+D$ at every input
length. The zero coefficient recovers CM117, and every coefficient gives
the same complete scores as CM117 on all prefixes of length at most two.
On the longer prefix $[0,1,0]$, opposite coefficients can give opposite
strict winning classes at the same other weights. At fixed complete
features ordinary cross-entropy is convex in the output offsets.
The content softmax depends nonlinearly on the jointly trained key
phases and coefficient.

The training-only intervention uses both archived CM117 builds, all
six Rewrite tasks and two-hop lookup. Check all teacher and greedy
training vectors at zero, the complete training-loss derivative at zero,
and the fixed coefficients $-4,-2,-1,0,1,2,4$. These measurements use
the original training examples and their ordinary next-object targets.
