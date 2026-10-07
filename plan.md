# Basis correctness and convex architecture research cycle

Started: 2026-10-06. Status: active, stage 2.

## Objective and constraints

Prove complete Basis correctness in Lean, use its semantics to search for a
convex trainable architecture, and compare any verified candidate with the
original GPTMini softmax under the same measured FLOP budget. Use ordinary
AdamW. The public interface is `f(tokens: list[int]) -> list[int]`, preserving
the input and appending one prediction per call. Parity generates its label
and then EOS in two calls. Train separate parameter assignments for each
task and mode, as the existing benchmark does.

Retain the earlier requirements: compact shared parameters, causal token
processing, compatibility with the existing embedding/attention block
interfaces, and no change of optimizer. Do not count an oracle, a table of
all possible prefixes, an exponential feature bank, a fixed interaction
bank, or an uncharged head-search procedure as the requested solution.
State the exact trainable variables and feasible domain of every convexity
claim. Treat joint values, FFN, readout and loss explicitly: a convex block
does not by itself establish a convex training objective. AdamW compatibility
does not by itself prove convergence or a successful training run.

Work solo. Follow AGENTS.md, use local manuscripts in papers/, and commit
each logical result after its checks pass. Keep INDEX.md generated. New
formal results must not increase the existing sorry count. Keep ANSR stopped.

## 1. Prove complete Basis correctness

Status: complete for the explicit real-arithmetic model family below.

Use Transformer.Basis's actual integer IDs, raw grammars, easy/hard modes,
vocabulary sizes and context caps. The formal target is SolvesTask over
every validated supervised prefix, including depth order and neutral
tokens, raw adjacent MQAR bindings and last overwrites, and both parity
answer/EOS positions. This is stronger than the sampled benchmark's
99-percent stopping rule. Keep that distinction explicit.

Verify the original softmax GPTMini implementation, including embedding,
prenorm, QKNorm, RoPE, mask, finite softmax, XSA, both residuals, ReLU2 FFN,
tied readout and checked List Int adapter. Prove behavior of given model
parameters from explicit local parameter/representation constraints and
layer induction. Correct output logits, SolvesTask, or a whole-prefix
oracle encoding cannot be premises of the substantive correctness proof.
An existence statement for weights alone is not the requested result.
Demonstrate simultaneous satisfiability of any parameter constraints in
the stated architecture; do not silently enlarge the small/large GPTMini.

Current proof boundary at b243ba5: raw last-binding semantics, softmax
routing error bounds, conditional ordered-prefix tests, concrete ONE-count
and phase features, full-stack phase preservation, and internal code/error
conditions implying an actual integer answer are proved. Complete prefix
encoders, reliable retrieval selection, and the parity/EOS decoder remain.

First work item completed: ParityCollision proves a full first-state
collision for legal 4-token/8-token prompts with different parity labels.
DenominatorEmbedding/Head/Block add a simultaneous raw BOS indicator in
the original small GPTMini and prove internal count recovery. ParityFeatures
establishes it for every raw prompt and supplied-answer continuation,
without an external length or RMS multiplier. This repairs a feature
representation; it is not yet a complete parity solver.

CountSpline now proves an exact homogeneous four-hinge count decoder;
CountFFN realizes it with the original W_in/ReLU2/W_out in 68 hidden
units. CompletionFFN adds a simultaneous 69th phase unit for EOS, proves
its exact output formula and preserves phase. All fit the existing width
256. These are explicit original FFN computations, not a task oracle.

ParityConstruction supplies a complete original two-layer parameter family
with distinct tied EVEN/ODD/EOS embeddings. Its projected inputs are proved
unchanged and its full hidden state uses the real decoder FFN.
ParityInputs derives exact raw prompt/answer pre-FFN formulas and phase
values for every bit word, with no prepared encoder-value premise.

ParityCoordinates/Bounds/Readout/Correctness now derive uniform strict
label/EOS margins with ordinary finite FFN weights 1024 and 131072.
The given original small two-layer model's checked integer function
solves every legal parity prefix in either mode and generates label/EOS
in two calls. This covers real arithmetic and the documented shared
epsilon range (0, 1/64], not floating-point equivalence or AdamW success.

AdjacencyRoPE/Routing prove an explicit predecessor gap using the original
head_dim=16/theta=10000 pair seven and the actual clipped QKNorm. Every
wrong causal position has a derived positive gap up to context 128, and
the original finite softmax/XSA copies the predecessor with a derived
error bound when its self-value is zero. This positional routing result
still needs simultaneous raw embedding/QKV realization. A complete recall
encoder must also gate the table region: a post-table filler after an
earlier query must not be interpreted as another key/value write.

RecallCodes/Frequencies/RotaryCode/Matching supply a compact four-base-four-
digit code for all 256 symbols in eight coordinates. In the actual head's
slow pairs four through seven, its exact norm and QKNorm/RoPE score are
proved. At every displacement within context 64, an identical symbol has
normalized score at least 0.93, a different one at most 0.91, and the
content gap is exp(alpha)/50. For equal symbols the actual rotary score
strictly prefers the later visible write. This is simultaneous compact
geometry, not yet a raw encoder or full recall SolvesTask theorem.

RecallSlots/Embedding/EmbeddingFeatures now realize simultaneous raw key
and value codes, a protected constant and three type flags in the original
64-coordinate table. Every one of the 548 actual vocabulary entries has
derived squared norm six, so all first-prenorm projections have one proved
multiplier. Actual slot matrices read faithful codes and zero cross-channels;
the future copy/position/gated-key coordinates are proved initially zero.
The table still contains only token-local information. Fused QKV and the
causal BOS-derived table gate are the next computations to verify.

RecallQKV/HeadValues/RawHeads/RawCopy now give the simultaneous original
64-to-192 matrix and nonzero W_o. The actual raw prenorm/fused projection
yields the predecessor query/key and faithful compact values, plus a
second zero-score marker head. At every actual adjacent key/value pair,
the real softmax/XSA head copies the key with error at most
4*(T-1)*exp(-adjacentGap alpha), and the true attention residual stores
that copy alongside the unchanged raw value and query codes. Its gap and
value bound are derived, not assumed. Exact causal BOS mass, the actual
FFN table gate and robust second-block latest-write/readout remain.

RecallMarkerWeights/RawMarker now derive the exact causal denominator
i+1 and the marker value 1/(i+1) at every later raw position, including
intermediate positions in arrays containing future tokens. The actual
first-head pair and marker run simultaneously; coordinate 26 in the true
attention residual contains that exact mass. Only raw BOS/alphabet IDs
are premises, not a prepared marker or an external position input. Across
the original recall context its value lies in [1/64, 1/2] and strictly
decreases with the position. Next realize a fixed table threshold in the
actual ReLU2 FFN and prove exclusion of post-table false writes.

RecallGateScalar/Matrices/Bounds/Signals/FFN/Block/RawGate now realize
that cutoff with sixteen of the original 256 ReLU2 units. The true first
block's gated-key slot is exactly zero for raw keys, BOS and every later
value after the table; the raw code/type channels remain unchanged.
Raw table values retain the complete real predecessor copy times a
derived positive gate amplitude, including their actual position-dependent
RMS square. No gate-sign, prepared copy or encoded table indicator is a
premise of the raw-input outcomes. Full validated-prefix parser coupling,
robust latest-write selection and second-block/readout margins remain.

RecallLatestGap strengthens equal-key strict ordering to the explicit
uniform normalized margin (1-cos(1/100))/4 whenever records are at least
one raw position apart. It includes all four actual RoPE frequencies
and fits below the different-key content margin, so one positive constant
can separate either kind of competitor before copied-key perturbations.
Finite temperature can use this bound; the real encoder/QKV perturbation
and value/readout margins still need to be connected.

RecallCopyAccuracy/RawBinding now supply an explicit finite shared
first-head temperature for any positive copy tolerance at context 64.
At the fixed tolerance latestMargin/16, the actual full gated table-key
error is bounded by its actual amplitude times that tolerance, and its
norm is strictly positive when the finite gate gain is positive. Raw
query/value codes survive simultaneously. No small-copy-error or nonzero
encoded-key premise is assumed. The conservative temperature is a real
capacity bound, not a numerical-optimality or AdamW claim. Actual second
prenorm/QKV/QKNorm saturation and robust score/readout transport remain.

RecallRotaryInsert realizes the ordinary shared eight-to-sixteen matrix
needed by the second projection. It retains arbitrary real copied-key
coordinates, preserves their exact inner products and norms, and transports
every copy error without amplification through actual RoPE. Its output
on categorical codes is precisely the already verified matching geometry;
zero nonrecords and unused fast coordinates remain zero. The true second
prenorm, shared QKV and clipping saturation still need to be derived.

RecallSaturation proves the exact clipped-QKNorm scale cancellation
condition, including different query/record amplitudes. Above the actual
threshold, faithful insertion and original RoPE convert a compact copy
error eta <= 1 into normalized error at most 2*eta, independently of the
small implementation epsilon or positive amplitude. The base norm lower
bound is derived from the norm-two symbol and its copy distance. These
are local operator laws; raw-state/RMS bounds and one finite shared QKV
gain must still discharge clipping rather than assume it for the model.

RecallStateBounds derives pre-FFN norm in [1,9] from the actual raw
embedding, both simultaneous softmax/XSA heads and their real W_o. The
lower bound comes from the preserved constant. Consequently the true
gate prenorm multiplier is in [1/2,8] for epsilon in [0,1]. The complete
first block also has norm at least one and a positive next-prenorm
multiplier at most eight. These are uniform raw computations, not
prepared representation or common-position-scale premises. A fixed gate
gain, full-block upper bound and sufficient shared QKV scale follow next.

RecallGateGain/EncoderBounds/ProjectionScale now choose the finite shared
table gain 4/tableMargin and derive actual raw table amplitude and bound-key
norm at least one. The full raw block has norm at most M=9+512*gain at
every position, with all gate regions discharged from raw IDs rather than
assumed. Its next RMS multiplier is at least r=8/sqrt(M^2+64*epsilon).
The fixed Q/K gain (1+epsilon)/r therefore times each genuine next RMS
multiplier exceeds epsilon. The raw query's own value remains exactly
zero for XSA; genuine prenorm value norms are in [2*r,16]. These finite
uniform coefficients depend only on table size/epsilon. The complete
fused second QKV and robust selection/readout still need to be connected.

RecallSecondQKV realizes the actual shared 64-to-192 matrix, including
all original query/key/value chunks and sixteen head coordinates. Q reads
the raw query interval, K the real gated-key interval and V the independent
raw value interval. The same fixed ordinary gain scales Q/K; V retains
its own genuine prenorm multiplier. Unassigned head rows, excluded keys
and query self-values are exactly zero. The next step discharges actual
clipping and transports copied-key error through these real projections
to robust latest-write selection and the complete tied integer readout.

RecallNormalizedInputs now discharges actual query and table-key clipping
for these genuine projections. The raw normalized query is exactly its
verified rotary direction; normalized table-key error is at most twice
the fixed copy tolerance. The real query's own V and K are exactly zero.
All statements derive from the full first block, raw token IDs and shared
finite coefficients; no prepared normalized head is supplied.

RecallScoreError evaluates the genuine second Q/K/V through ordinary
forward-pass accessors. It bounds every raw query/table score's error
against its true rotary reference by 2*exp(alpha)*copyTolerance, with no
inverse-epsilon amplification. Raw key/query competitors have exactly
zero score and the genuine query's own value remains zero. The robust
latest-write margin and finite retrieval/readout are the next obligations.

RecallRawScoreGap now derives a strictly positive real gap
exp(alpha)*latestMargin/2 against every different-key or earlier
same-key table competitor, using actual raw adjacency and chronological
positions. Both real score errors fit within the geometric margin;
displacements are derived from the original context bounds. A raw
overwrite witness satisfies every hypothesis. Complete parser coupling
and excluded nonrecord scores must still give the full causal-row gap.

RecallRawExcluded derives exactly zero matching scores for raw BOS and
post-table value fillers through the complete actual gate/prenorm/K.
An imperfect real matching table record's score is at least the positive
retained gap. The real selected V equals its protected raw symbol code
times the genuine next RMS multiplier; all actual V norms are at most
sixteen. These results complete the local competitor/value cases needed
to assemble full-row routing and quantitative finite retrieval.

RecallRawRouting now assembles all genuine score-row cases from raw
table/alphabet/adjacency and chronological last-write predicates. These
predicates mention only raw data, never a hidden state, score or desired
model output. It derives the full-row positive gap, actual causal tail
bound and original softmax/XSA head retrieval error
32*(T-1)*exp(-retainedGap). The six-token raw overwrite control strictly
prefers its later, different-valued write. Full Basis parsing must still
discharge the raw predicates. Next choose a finite shared temperature,
realize the genuine second W_o and connect the complete tied readout.

RecallRetrievalAccuracy chooses an explicit finite shared second-head
log-temperature with true gap log(1+2016/tolerance), evaluating the
finite softmax tail directly. The actual original head achieves every
positive tolerance uniformly. At the fixed global tolerance
secondScaleLower/16 it has the accuracy reserved for tied value readout,
and a finite ordinary output gain has gain*secondScaleLower=2. No small
leakage bound or infinite-temperature hypothesis is supplied. The result
is real-arithmetic capacity, not floating-point or AdamW success.

RecallRetrievalBlock realizes the genuine second original attention/block
parameter records with fixed fused matching QKV, finite temperature and
ordinary nonzero 64-to-64 W_o. Its complete true block writes the actual
compact retrieval to the independent value interval, preserving all
query/type coordinates and retaining original residuals and zero-matrix
FFN. Actual headAt accuracy and output error are connected to this real
block. Complete tied readout and validated integer-parser coupling remain.

RecallFinalState transports the true complete second-block error into a
uniform final-residual error at most 1/8. The selected value retains its
genuine position-dependent prenorm amplitude, proved at least two using
the one fixed ordinary output gain. The real final query code, constant
and all type/reserved readout axes are derived directly from the raw
embedding. No common selected-value scale or desired residual is an input.

RecallReadoutCoordinates evaluates the actual tied input embeddings against
the true final query. Key scores are at most six, reserved scores exactly
one, and selected-value reference scores are one plus their genuine
amplitude times the compact inner product (four for the same symbol,
at most three for any different one). All 548 raw embedding entries amplify
residual error by at most three. These identities precede strict readout.

RecallReadout proves a strict selected-value margin against every one of
the 547 remaining vocabulary entries. Both actual error bounds and the
selected prenorm amplitude are derived from raw data. The true final
RMS multiplier is positive by its protected constant, so the actual
normalized tied readout and greedy decoder return the selected value.
The genuine six-token overwrite control returns the later different value.
These results still require complete ModelParams and parser coupling.

RecallConstruction supplies the complete original ModelParams: the same
raw tied table and the two real blocks, with fixed shared finite weights.
Both true hidden-loop states and full forward logits are connected to the
verified computations at every position. The actual checked List Int
function retains its input and appends the selected value from its true
last-row decoder, including the literal overwrite control. Raw layout
and latest-write applicability must still follow from complete Basis parsing.

Basis.RecallParsed derives the complete key/value alphabet from every
successful raw query/filler scan and retains every record's range checks
in the actual last-write decomposition. Earlier repeated-key writes,
later unrelated writes and arbitrary valid query/filler regions are kept.
No paired model input is supplied by these data-side inversion theorems.

Basis.RecallTablePositions proves actual even/odd serialized key/value
indices, the BOS offset and preservation under any appended query region.
Disjoint validated intervals force every raw table value to its own odd
record position and give its immediate actual key predecessor. No later
matching record can follow the selected last-write decomposition.

Basis.RecallPrefixPositions transports these index facts through actual
BOS, the whole table and arbitrary query/filler regions. It derives true
table-value row recovery, the immediate predecessor key and a bound on
every matching write by the selected raw position. Selected key/value
positions and the successful parser's final validated query index are
exact; every non-BOS position has the true checked key/value alphabet.

RecallIntegerArray connects every actual integer read to the same checked
finite token and embedding index, deriving a bounded position from raw
read success. Key/value interval validation is equivalent to the genuine
finite key/value IDs, and raw BOS recovers actual token one. Reverse
key/value readback preserves exact raw IDs at the same array positions.

RecallParserLayout derives the complete actual RecallRawLayout solely
from successful full Basis parsing, including genuine finite BOS, all
alphabet entries and each true table value's immediate real key predecessor.
It also recovers the genuine final query symbol/position and nonempty
first/final finite rows. No manual raw layout certificate remains necessary.

RecallParserLatest derives the genuine finite selected last write and its
neighboring key solely from complete parser success. Its actual raw key
is proved identical to the final query, and every real matching write is
no later than the selected row. All finite bounds and the answer ID are
derived; no layout, selected-row or latest-write certificate is supplied.

RecallCorrectness proves recallModel_solves_recall for both actual Basis
grammars through the genuine public List Int adapter. Complete successful
parsing alone discharges every raw model condition; all 256 keys/values,
legal query/filler positions and hard-mode last overwrites are covered.
The true eight-write control, sixteen-write overwrite control and swapped
binding controls produce the correct distinct raw answers. This is the
given width-64/two-layer real-arithmetic/shared-epsilon capacity family,
not AdamW success, floating-point/default-epsilon equivalence or a proved
lift to the experiment's width-128/six-layer large model.

Parity and recall are complete within their documented formal model
scope. Next construct and prove the full depth-prefix recurrence in the
actual original blocks, at the existing small/large dimensions.
DepthStep proves a genuine continuous three-ReLU2 saturated presence
step and an ordinary linear raw-type gate. A separated actual prefix
signal yields exactly zero or a positive common plateau; both computations
retain their quadratic scale through real position-dependent prenorm.
For ordered detection the opposite-type self-value must be zero, so XSA
preserves that signal. Arbitrary own-value subtraction is not assumed to
equal removal of just the own softmax summand. Actual matrices, propagated
presence floors, final readout and full depth SolvesTask remain to prove.
DepthPresence connects arbitrary nonconstant real value amplitudes to
the genuine diagonal-inclusive causal softmax/XSA head at zero self-value.
It derives nonnegativity, a length-independent upper cap, exact visible
absence and a uniform occurrence floor L/128. Zero-or-bounded amplitudes
therefore give a separated true prefix signal, even with future array
positions and different prenorm scales. Actual raw matrices and layer
induction must still derive these local representation bounds.
DepthMatrices realizes both type-conditioned three-hinge detectors in
six disjoint ordinary W_in/ReLU2/W_out units, fitting the existing small
256-unit and large 512-unit FFNs. The actual output and all protected
coordinates are proved, including the genuine position-dependent RMS
square. No bias, prepared Boolean output, extra width or external scale
is inserted. Complete raw embeddings, fused attention projections,
amplitude induction and the final depth readout remain to connect.
DepthRecurrence proves alternating ending-pattern recurrence at the
actual original Option Bool positions, retaining every neutral slot.
A next occurrence is exactly the current raw letter plus a strictly
earlier opposite-ending occurrence; the opposite current feature is
absent, which is the semantic zero-self obligation for original XSA.
The two final ending-pattern presences are proved equivalent to the
independent actual E_2/E_4 integer oracle, including wrong-start rejection.
These data-side predicates do not define the model's hidden computation;
actual raw embedding, fused head matrices and state induction still must
realize them, then yield the full tied/greedy integer readout.
DepthEmbedding now supplies ordinary token-local tied entries in the
original small/easy 64x2 and large/hard 128x6 configurations, with four
heads and unchanged FFN widths. Its working coordinates are distinct,
raw constant/type channels are local to the token, all later channels
start at zero and every validated raw input has norm in [1,2]. Labels
use separate tied readout channels and do not occur in the raw input
grammar. Actual fused heads, quantitative state induction and readout
remain; these capacity parameters make no training or convexity claim.
DepthUniformProjection proves the full original head formula without
a zero-current assumption: genuine causal mean times
1-(amplitude/max(abs(amplitude),epsilon))^2 along its unit value axis.
This attenuation lies in [0,1], so all nonnegative feature arrays give
a nonnegative actual probe and head norm below their true amplitude
cap. Final residual readout can retain a positive current occurrence;
earlier-occurrence floors still require its semantic zero-self case.
DepthQKV now realizes both uniform value heads in a single ordinary
fused matrix at each original width. All actual query/key slices are
proved zero, the two V slices read their own residual coordinates,
and the other two original heads have zero values. These identities
hold for arbitrary real residuals with the true chunk/view indices;
output projection, RMS amplitude bounds and state recurrence remain.
DepthNormalization proves positive actual RMS scale, upper scale
sixteen, upper squared scale 128 and normalized norm sixteen. Under
the explicit local norm bound [1,4096] and epsilon in [0,1], every
true scale is at least r=1/4224. One shared ordinary threshold
r^3/256, homogeneous FFN gain 1/(2*threshold^2) and tied label gain
8/r^2 now have proved positive finite compensation identities.
The actual block induction must discharge that local norm bound;
the coefficients alone do not establish floating-point or AdamW success.
DepthAttention now couples the complete actual prenorm/fused-QKV/
RoPE/QKNorm/softmax/XSA/head-merge/W_o sublayer to the two genuine
continuous signals at arbitrary real input arrays. Unwritten output
coordinates are exactly zero; distinct targets read their own signal.
The output matrix and true normalization no longer remain abstract.
Quantitative signal/state bounds, ordered layer induction and full
tied/greedy integer readout still must be completed.
DepthAttentionBounds derives nonnegative genuine V coordinates and
their universal cap sixteen from true normalized residual norms.
Within the explicit bounded state domain, every real feature at
least r^2 gives actual V amplitude at least r^3. Actual attenuated
signals lie in [0,16], so the complete two-head output and W_o add
norm at most 32 and the real first residual is bounded by M+32.
Zero self and visible absence are proved at the actual raw residual
coordinates, including a nonzero future-feature control.
DepthSignalPresence now derives the exact gap 2*threshold in the
actual head from a visible real residual feature at least r^2 and
a zero current feature. In the explicit bounded, separated feature
domain, genuine signal positivity is equivalent to visible feature
presence; the true signal is zero or at least twice the threshold.
A simultaneous ordinary real-vector control satisfies every local
hypothesis. Actual ordered block induction must still establish this
domain and zero-self condition from the validated raw depth word.
DepthDetector supplies the complete original ordinary FFN record and
proves exact zero-or-true-RMS-square output, protected coordinates and
norm at most 256. The signal gap is required only when the raw type
matches: wrong types are suppressed using the cap alone. This removes
an unjustified gap condition on XSA-attenuated wrong-type signals.
A real active-A state satisfies all local hypotheses simultaneously.
Full block recurrence must derive these conditions from raw inputs.
DepthLayout now supplies fixed complete original detector BlockParams
for three possible stages. The actual first source reads the opposite
raw type; later sources are exactly the previous opposite-ending FFN
targets. Both real residual updates preserve the constant, raw types
and every other stage's channels. Signals and feature columns are
distinct in the unchanged widths, with no input-dependent parameters.
DepthResidual computes the actual attention residual: raw types remain
unchanged, fresh signal reads are genuine head outputs, future FFN
targets remain empty, and the true norm is at most M+32 and at least
one when the raw constant is one. The explicit six-condition input
predicate has a simultaneous active-A real-array witness; full raw-word
induction must establish it, without a semantic encoder premise.
DepthTransition now derives all actual pre-FFN constant/type/gap/cap
conditions from those incoming coordinates and true norm bounds. Each
complete block writes exactly zero or its real pre-FFN RMS square, with
positive output exactly at actual type-and-presence success. Including
both residuals, its norm grows by at most 288. The quantitative ordered
raw-state induction and final tied readout remain to be completed.
DepthFeatureBounds derives actual new flags in {0} union [r^2,128]
from incoming norm at most 4064. Their positivity is exactly raw-type
match and a strictly earlier genuine source feature at least r^2;
strict precedence follows from zero opposite self under the original
self-inclusive causal mask. Wrong raw types have exact zero flags.
The raw-word induction must instantiate these results at every stage.
DepthWordInput now derives every first-stage input condition directly
from the genuine token-local embedding of an arbitrary raw letter word.
Neutral positions are retained; both raw types are binary and mutually
exclusive, later stage channels start empty, and input norm lies in
[1,2]. A leading none has the same actual embedding as the true BOS
token, with integer-ID coupling still required at the full model.
DepthLevels/DepthInvariant/DepthEncoding now derive the complete real
detector-prefix invariant by induction from those raw words. All norm,
feature separation, fresh-channel, opposite-self and ordered-occurrence
conditions follow for actual blockForward outputs. Easy uses one detector
and hard three, leaving readout room within the original two/six layers.
Genuine readout, full ModelParams and integer coupling remain required.
DepthReadoutLayout proves genuine readout probes retain current flags
through the residual even under XSA suppression; reserved output axes
stay empty through the real encoder and raw constant/types are protected.
DepthReadoutPresence derives exact visible-pattern positivity, actual
zero-or-twice-threshold probes, cap 144 and pre-FFN norm in [1,898].
These are instantiated by the full raw-word encoder without a supplied
hidden-state invariant or normalized-feature premise.
DepthReadoutMatrices realizes the ordinary seven-unit readout FFN and
proves its complete continuous matrix formula and genuine prenormed
common RMS-square scale. All rows fit the original FFN dimensions.
DepthReadoutFFN derives true zero-or-common-RMS-square indicators,
the exact common coordinate, arbitrary protected channels and a norm
cap 384; simultaneous actual-vector controls satisfy every local premise.
DepthReadoutBlock/DepthWordFinalState now derive the complete actual
readout from raw words, including both semantic indicator coordinates,
common scale in [r^2,128], protected raw types and final norm [1,1282].
No invariant, encoder, route or correct-logit premise is supplied in
these full raw-word results. Tied margins and ModelParams/int coupling
still remain; this is real capacity, not an AdamW or convexity guarantee.
DepthReadoutScores/DepthReadout now prove strict whole-vocabulary depth
decoding: correct tied score at least five, opposite at most minus three,
every other token at most two. Actual final RMS preserves the margin and
greedy returns the independent ordered answer. Full original model-loop
and checked integer-prefix coupling remain the final depth obligations.
DepthModel realizes the complete original two/six-layer parameter record,
proves actual detector/readout/tail hidden-loop equalities and derives
strict full forward logits. Its token-local embedding equality is the
remaining input coupling, to be discharged by raw integer serialization.
Parameters are shared across prefixes and independent of epsilon.
DepthIntegerInput/DepthCorrectness discharge all true BOS/body lookup,
finite indexing, last-position and independent answer coupling. The
actual tokenFunction solves every validated raw E_2/E_4 prefix up to
128 at epsilon in (0,1], retaining original 64x2/128x6 dimensions.
Neutral, wrong-start and equal-bag/different-order controls pass through
the actual integer callback. No new sorrys were introduced.
BasisCorrectness consolidates all three tasks and both modes through the
actual original model family. basisModel_solves covers every validated
raw prefix at common epsilon in (0,1/64]; basisModel_reference_all gives
all six grammars at 1e-5. basisModel_predicts identifies the actual next
token with independent taskNext, and basisModel_parity_twice proves free
label/EOS generation. Widths are explicit: depth easy 64x2/hard 128x6;
recall/parity 64x2 for both full grammars. The large-model lift for those
two tasks, unequal Python epsilons, FP execution and AdamW success are
not certified. Stage 1's semantic exit criterion is satisfied.
Record each remaining assumption and discharge it rather than moving it
into a definition. A failed construction should produce a counterexample
or a precise missing condition, not a weakened correctness target.

Exit criterion: complete task theorems connected to the actual model
function, with no unproved encoder or routing input and no added sorrys.

## 2. Search in Lean for a convex architecture

Status: active. Complete raw semantics and original-model capability are available.

Derive the needed operations from the proven Basis semantics: order,
adjacency, key-conditioned latest-write selection, bounded counting and
completion phase. Search for compact changed operators that preserve them
while giving a proved convex parameter domain and the stated training
objective. Keep learned query-key matching and values in scope. Use the
existing coordinate-independent softmax-head obstruction to reject only
the operator classes it actually covers, not every changed architecture.

For every candidate record its formulas, parameter count and scaling,
causality, integer-function adapter, task guarantees, convexity theorem,
remaining hypotheses and counterexamples. Distinguish global convexity,
conditional convexity and an empirical favorable optimization landscape.

Current direction: a structured Gibbs head with affine complete energies.
Structured.Basic proves the actual positive normalized joint distribution
and global convexity of its complete-configuration NLL in all raw weights.
This is a foundation, not yet an admissible architecture. Latent routes,
state paths and matching/value channels require data-derived supervision
for this objective. Ordinary label-only CE remains outside the guarantee.

The compact proposal under investigation has free token-local Q/K log
potentials (four groups of four channels each) and free value log
potentials (five groups of four), plus a fixed ten-coordinate output
label code and one constant, plus one learned positional potential:
52 free token fields + 11 fixed axes + one free position axis fit width
64 exactly. The fixed output code is a decoder, not an input interaction
bank; all input matching/value potentials are trained. The pointer
energy sums Q(query,channel)+K(key token,channel)+V(value token,channel),
with learned chronology and a free positional potential per context slot.
The current complete-recall construction considers every visible
key/value position pair, with an additional free potential for each
signed relative displacement. Adjacency must be learned, rather than
supplied by a fixed predecessor shift or paired input encoder.
The absolute-position potentials must learn table exclusion from observed routes, without a
fixed table-record mask or an enumerated latent role bank.
Its implicit channels must be summed by products of small sums, never
an enumerated exponential feature table. A free six-state transition
head would handle order/count/phase, with all token transition energies
trainable rather than a hardcoded task interpreter. These are proposed
operators and slot counts; their actual computation, compatibility and
joint capacity are not proved yet. No successful AdamW run is claimed.
Factorial now proves the exact memory/group contraction and genuine
route-weight normalization. PointerTraining realizes shared free token
tables and chronology as actual linear complete energies, identifies the
computed compact objective with jointNLL, and proves global convexity
for both individual examples and shared finite minibatches. The actual
slot type has exactly 52 trainable fields per token; no per-prefix or
exponential parameter table is stored. Output-channel contraction,
semantic capability and the true replacement residual block remain open.
ChannelMarginals/PointerValues now prove that small per-group value means
equal the full joint Gibbs expectation at every raw parameter assignment,
with exact normalization and output coordinate bounds. The distribution
is proved identical to the one trained by contractedPointerNLL. Desired
routes/channels are not inference arguments. This completes the pointer's
compact matching/value contraction, not its raw Basis semantic capability.
PointerControls supplies finite shared learned Q/K/value assignments:
two queries prefer different keys, and actual five-token raw adjacent
bindings with swapped values give means 19/36 versus 17/36 and opposite
strict decoded labels. This is a two-symbol/two-record control, not the
full Basis recall proof. Its unrestricted training objective is convex;
the forward reads actual raw neighbors rather than a paired-key encoder.
Binding now extends this control's operator to fully learned physical
key/value pairing. It proves that the computed all-pair/small-channel
normalizer and inference are the exact affine-energy joint Gibbs model,
whose complete likelihood is convex in all shared Q/K/value, absolute
position, chronology and relative-binding weights simultaneously. The
largest Basis recall layout has 28816 free scalars, including 127 shared
relative offsets; no per-pair learned table or hard adjacency mask is
stored. Naive route evaluation uses quadratically many position pairs
per query and must be charged in the FLOP comparison. Full raw recall
correctness and actual tensor/residual/tied integration remain open.
RecallPositions now derives every physical even table value slot's
actual neighboring key/value IDs, the selected last-write slot and
matching-record chronology from successful complete raw parsing. A real
overwrite plus two distinct queries and a later even-position filler
checks the table boundary. These are data/witness facts, absent from
the freely learned all-pair forward; true finite energy gaps and recall
head correctness are still required.
ChannelGaps proves actual zero/gain channel-energy bounds and a full
gain deficit for wrong output channels or distinct matching vectors.
RecallPotentials constructs one finite actual 52-field shared vocabulary
table, absolute/relative positional potentials, chronology and head
logits for every prefix of a task. Real raw key IDs retain all 256
distinct codes, and every genuine Q/K/value lookup is evaluated exactly.
This is a concrete capacity weight assignment, not a trained model or
complete recall solver; all-pair raw energy gaps and decoding remain.
PointerDecoder proves the actual ten output-coordinate dot product,
exact whole-joint expected score, coordinate/score bounds and the full
11p-10 vocabulary margin. RawBinding realizes true unchanged raw-prefix
query/key/value/position reads over every visible pair, identifies its
computed loss with the same inference probability and proves global
convexity in all free raw binding parameters. Its exact implicit choice
count is T*T*4^9, bounded by 1073741824 at the actual recall cap; these
configurations are summed by contraction, not stored or enumerated by
inference. The finite witness's raw semantic energy gap remains to prove.
RecallEnergy/RecallGap now derive a uniform gain gap against every
actual raw position-pair/channel rival solely from successful complete
parsing and the context cap. Wrong positions/bindings/channels lose
their real finite weight gain; every fully matching physical record is
no later than the actual last overwrite. The correct true configuration
therefore has probability at least 1-1073741824*exp(-gain). The finite
gain log(10^12) puts this full tail plus a two-head allowance below 1/11.
This is a genuine raw head probability bound, not yet the public greedy
recall callback, combined-head capability or full tensor-stack result.

For the ordered head, also investigate a row-softmax six-state Markov
model: free token-conditioned transition logits, normalized causal state
propagation, and learned conditional output channels. MarkovChain now
proves the actual chronological encoder's positivity, normalization and
cached-prefix composition. MarkovTraining proves the complete observed
state-path NLL equals its actual path's negative log probability and is
globally convex in every unrestricted shared transition coordinate. It
sums affine categorical losses, avoiding a global path-partition
computation. MarkovMarginals now proves normalization of every complete
path distribution and exact endpoint-probability/expectation equality
with markovRun's actual compact forward recurrence at arbitrary weights.
Histories exist only in proofs and training targets, not inference memory.
MarkovEmissions adds unrestricted conditional output log potentials and
proves their compact value mean equals the same complete learned
state/path/channel model's exact expectation, with normalization and
coordinate bounds. MarkovObjective now proves the computed compact loss
equals that same joint model's negative log probability and is globally
convex simultaneously in unrestricted initial/transition/emission raw
parameters, including shared finite minibatches. SharedSlots realizes
the actual common raw token coordinates: disjoint 16/16/20 pointer
groups and 36 independent state-transition fields in the same 52 slots.
Its full parameter domain has exactly 52*V+C+129 free real coordinates;
there are six free initial and 120 free conditional value logits.
The state-head objective is proved convex on that whole actual space.
SharedPointer now realizes actual Q/K/value lookups and learned positional
potentials/chronology as one true linear complete energy on that same
parameter domain. Its compact partition, complete NLL and inference
probability are exactly the affine Gibbs model, and joint training is
globally convex with all these parameter groups free. No record mask
or semantic role input is part of inference; full recall capacity remains.
This variant still needs full raw depth/
parity semantic paths and actual residual integration; it is not accepted.
Concentration proves true finite-energy rival/tail bounds for the actual
Gibbs model and actual row softmax, including finite freely trainable
sharp-row witnesses. These are local operator laws only: semantic raw
parameter assignments must derive their gap conditions, and complete
path/output confidence and task capability remain open.
MarkovTeacher now separates data/reference states from actual learned
inference, proves the real teacher path endpoint is its chronological
reference run, and derives full-path confidence loss at most T*epsilon
from actual local row probabilities. The genuine compact encoder's
endpoint mass contains the same initial-state/path probability. Finite
sharp raw transition tables discharge row bounds without hard dynamics;
the task-specific reference rules still need independent Basis proofs.
ParityReference now proves the six-state data rule against the independent
raw Basis parity semantics: real bit tokens, BOS/SEP, count modulo two,
both validated label and supplied-answer/EOS prefixes. It has no hidden
phase/count input. This is a semantic target/capacity-rule proof, not yet
the learned finite-row model's full parity SolvesTask theorem.
MarkovConfidence now bounds the actual compact encoder's reference
endpoint and the complete learned initial/path/value configuration for
explicit finite tables. For six states and five four-channel groups,
every raw length T<=128 has joint mass at least 1-794*exp(-gain).
This derives genuine output confidence, but still requires raw shared
slot realization, proven task labels and actual tied decoder margins.
OutputCodes/OutputMargins now cover every ID below 1024 with ten actual
decoder coordinates. Distinct target codes have gap at least one; the
true full-distribution mean has margin at least 11*p_correct-10 against
every rival. The compact learned Markov score equals that same joint
expectation, so gain log(100000) yields strict whole-vocabulary margins
through context 128. Raw shared witnesses and residual/tied realization
remain open; this is not a complete learned Basis solver.
SharedReference now realizes all finite initial/transition/emission
witnesses in one actual shared 52-field token/global parameter assignment.
Every genuine learned lookup, complete loss and compact decoder score is
proved equal to the same finite stochastic model with the derived margin.
The data rule chooses weights only; it is absent from learned inference.
Next connect raw task encodings and integer decoding, then tensor blocks.
SharedInterface/SharedParity now prove the actual standalone learned
state/value head solves both complete raw parity grammars through a
checked List Int -> List Int callback, including two real free-generation
calls (label then EOS). One finite shared parameter assignment covers all
inputs; no correct encoder/state/logit premise is used. Actual causal
data-derived state/channel targets agree with the independently checked
raw output, and their genuine likelihood is globally convex in all shared
weights. This is head-level capability, not full tensor-block integration.
DepthCompression/DepthScan/DepthReference now prove the six-state data
scan against the independent raw Basis depth parser/subsequence test.
Actual run compression, Bool-chain canonical forms, repeated-letter
idempotence, saturated alternating counts, wrong-start rejection and
neutral/BOS serialization derive the exact label on every valid raw
prefix for E2/E4. This is independently proved data/reference semantics;
the actual learned finite shared depth head must still be connected.
SharedDepth now connects one finite shared raw vocabulary/head weight
table per mode to actual stochastic inference and checked greedy decoding.
The genuine standalone head solves every full raw E2/E4 prefix through
List Int -> List Int, including order-sensitive and conflicting-mode
controls. Its complete raw-data state/channel targets match the independent
Basis answer, and their likelihood is convex in the entire free shared
domain. BindingInterface/SharedRecall now prove full standalone recall
capability in both modes through the actual checked integer-list callback.
The final physical query reads every visible key/value position pair,
using free learned Q/K/value, absolute position, chronology and relative
binding potentials. Complete raw parsing derives the latest correct
record and a strict actual whole-vocabulary margin at finite weights;
the equal-bag swapped-value control has opposite actual predictions.
Thus all three tasks have proved standalone learned-head capability.
MixedHeads/MixedTraining now realize genuine learned two-head mixing on
that same unrestricted parameter space. Both actual branches retain
positive probability; the compact mixed score equals the full normalized
mixture's decoder expectation. The computed branch/path/route/channel
complete NLL equals minus log of that same true mixed probability and is
globally convex jointly in all raw token, value, position and head
coordinates, including variable-length shared minibatches. No task label
or correct state/route enters inference. Complete recall data-target
generation, combined finite-weight task confidence/capability and the
actual tensor-stack/prenorm/residual/tied readout remain required.
Output-only CE and AdamW convergence are not proved convex or successful
by these complete-likelihood/capacity results.

Remaining acceptance tests: prove compact contraction of the latent
partition, learned matching and joint value expressivity, complete Basis
capability from actual raw inputs, the prenorm/residual/tied-readout
coupling, and the precise auxiliary-target generation without inference
oracle use. Charge that generation and the changed loss in FLOP accounting.
Reject or repair this proposal if these obligations fail; stage 3 stays queued.

Exit criterion: a candidate with proved task capability and a proved
convexity claim covering the parameters/objective being trained, usable
with the existing ordinary AdamW implementation and public interface.

## 3. Measure the softmax baseline and compare equal FLOPs

Status: queued; starts when stage 2 produces an admissible candidate.

Use the existing Basis small/large GPTMini softmax recipes and the actual
success criterion. Pin source revision, task/mode, splits, seeds, model,
AdamW settings, schedule, batch, precision and hardware. Reuse archived
measurements only when those details and the required FLOP counts can be
recovered; otherwise run the controlled baseline. Do not report an
unmeasured budget as a measured one.

Count forward and backward arithmetic for all training steps and charge
candidate preprocessing, learned features and any search. Record the FLOP
convention and coverage of the counter, with evaluation costs separately.
Record cumulative training FLOPs at the first successful validation
observation, plus test/OOD results, loss curves and seed variation. Preserve
unfinished and failed baseline runs instead of assigning them success.

Port the verified candidate through the lab's domain/implementation/DSL
interfaces with meaningful tests and Lean/source citations. Use identical
data and ordinary AdamW, and stop each candidate at the corresponding
baseline's cumulative FLOP budget. Compare accuracy and success at those
budgets; elapsed time and update count are secondary measurements. Put
runs, differences and findings in the standard experiment directory and
README registry. Never substitute a much smaller compute budget.

## 4. Repair or replace weaker candidates

Status: queued; repeats after each controlled comparison.

When a candidate performs worse, preserve its run and identify a concrete
semantic or optimization failure. Reproduce it in a small control and
repair its representation/operator in Lean. Reprove correctness and the
applicable convexity claim before rerunning the affected comparison.

If the repair cannot satisfy the compactness, interface, learned-matching,
ordinary-AdamW or convexity constraints, record the failed path and its
mathematical reason, then search for a different architecture in stage 2.
Keep the baseline protocol and measured budgets fixed across candidates.

## 5. Cycle log and verification

This file is the active plan and must be updated with dates, commits,
theorems, remaining obligations, candidate decisions and run artifacts.
EXPERIMENT_PLAN.md points here while retaining earlier completed research.
Each Lean commit passes lake build, make.py audit, index and forbidden.
Each Python commit passes make.py test and the relevant experiment checks.
Do not mark the cycle complete while a required proof or comparison remains.

| Date | Stage | Result | Remaining work |
| --- | --- | --- | --- |
| 2026-10-06 | Setup | Cycle started from b243ba5; prior 77 semantic theorems retained. | Stage 1: normalized count versus exact count, full decoders and encoders. |
| 2026-10-06 | 1 | Proved legal variable-length first-state collision; actual ONE/BOS channels recover the count for all raw parity words in the original 64-wide block. | Bounded ReLU2 parity decoder, EOS/readout, raw MQAR and depth encoders. |
| 2026-10-06 | 1 | Repaired features committed in 2099735. Proved homogeneous parity decoder and realized simultaneous parity/EOS in 69 of the original 256 FFN units. | Full-stack distinct tied embeddings and readout margins; raw MQAR/depth encoders. |
| 2026-10-06 | 1 | Decoder committed in 69b9380. Connected distinct tied embeddings and the actual two-layer hidden state; derived universal raw prompt/answer FFN inputs. | Uniform quantitative readout margins and parity SolvesTask; raw MQAR/depth encoders. |
| 2026-10-06 | 1 | Raw full-model coupling committed in 8adde25. Proved all 68 tied-score comparisons, bounded actual normalization and complete parity SolvesTask, including free generation and the 16-ONE/19-slot boundary. | Full raw MQAR and depth encoders; convex architecture search and controlled FLOP comparisons remain. |
| 2026-10-06 | 1 | Parity correctness committed in f8ea725. Derived an actual original RoPE predecessor gap and finite softmax/XSA copy bound without an assumed positional gap. | Realize simultaneous raw QKV, gate post-table false writes, derive compact content/latest-write matching, then full recall/depth correctness. |
| 2026-10-06 | 1 | Positional copy committed in 10a2c36. Proved compact collision-free 256-symbol codes, all actual low frequencies, normalized content score gap and strict latest-equal-key preference in the original head. | Simultaneous raw embedding/QKV/table gate and finite-copy robustness; full recall readout and depth construction. |
| 2026-10-06 | 1 | Compact geometry committed in a7d5e0f. Realized all raw key/value/type channels in the original embedding table, derived uniform true RMS scaling and proved exact disjoint projections and initially empty encoder channels. | Fused original QKV, simultaneous predecessor/BOS heads, actual table gate, robust latest-write retrieval/readout; full depth construction. |
| 2026-10-06 | 1 | Raw embedding committed in 298d784. Realized simultaneous original QKV/head merge/W_o and derived actual raw adjacency binding error in the first attention residual, preserving value and query codes. | Exact causal BOS marker, actual ReLU2 table gate, finite-copy/latest-write robustness and full recall readout; full depth and convex candidate remain. |
| 2026-10-07 | 1 | Actual raw binding committed in 9b448ed. Derived exact causal BOS mass at every later raw position and transported it into the true original attention residual; proved context bounds and strict position ordering. | Actual ReLU2 table gate and raw-prefix parser integration, robust second-block retrieval/readout, full depth correctness, then convex architecture search. |
| 2026-10-07 | 1 | Exact causal marker committed in 419da8e. Realized the actual sixteen-unit table gate and whole first block; proved exact exclusion of raw keys/BOS/post-table false writes and a derived positive amplitude for true table values. | Complete raw-prefix parser coupling and finite-copy/latest-write robustness, original second-block/tied readout, full depth correctness; convex candidate and FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Actual full-block table gate committed in 62c26f0. Derived a uniform finite-temperature latest-write score margin from the original slow RoPE pairs, simultaneously below the different-key content gap. | Transport the real encoder's copied-key errors through second prenorm/QKNorm/QKV, bind all conditions to validated raw prefixes, then original retrieval/readout and depth correctness. |
| 2026-10-07 | 1 | Uniform latest-write gap committed in b4822b4. Chose a finite shared predecessor temperature and proved uniform raw copy accuracy; the true full block stores positive-norm table keys with relative error latestMargin/16, preserving raw query/value codes. | Realize the actual second matching matrix and prove normalization/saturation/error transport, complete raw-parser/readout coupling, full depth and convex architecture search. |
| 2026-10-07 | 1 | Uniform finite raw binding accuracy committed in 7d7fa9d. Proved an ordinary shared linear insertion of all real copied keys into the original slow RoPE pairs, with exact norm and error preservation and faithful categorical matching geometry. | Derive true second prenorm/QKNorm saturation and robust latest-write routing/readout, complete raw parser coupling, full depth correctness and convex architecture search. |
| 2026-10-07 | 1 | Faithful matching insertion committed in 8dbdabe. Proved actual clipped-QKNorm cancellation of independent positive amplitudes above saturation and epsilon-independent normalized copy error through the true matrix/RoPE. | Derive uniform raw-state/RMS bounds and a finite shared second QKV gain to discharge clipping, then actual robust retrieval/readout, validated parser coupling, full depth and convex architecture search. |
| 2026-10-07 | 1 | Local QKNorm/error transport committed in 42305c4. Derived genuine pre-FFN norm [1,9], gate RMS scale [1/2,8], and a protected positive bounded next-prenorm multiplier after the full first block. | Choose a fixed gate gain and bound the full block to derive one shared saturating QKV scale, then original robust retrieval/readout and full raw parser/depth correctness; convex search remains queued. |
| 2026-10-07 | 1 | Actual state/prenorm bounds committed in 87cf916. Chose finite shared gate/QK gains and proved raw table amplitudes/key norms at least one, a whole raw-block upper bound, and a genuine shared projection/RMS product above epsilon at every position; query self-values stay zero and prenorm values have positive lower/finite upper norms. | Connect the true fused second QKV and normalized copied-key errors to robust latest-write routing/readout, complete validated raw parser coupling and full depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Uniform finite shared gains committed in 072caeb. Realized the actual fused second QKV with simultaneous raw query, real gated copied-key and independent value slots; derived its genuine prenorm formulas and exact zero nonrecord/self-value outcomes. | Discharge clipping for these actual projections, transport normalized copy errors to robust latest-write selection and tied readout, then full validated raw parser and depth correctness; convex search and FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Genuine second QKV committed in 830e91e. Derived exact raw normalized query direction and twice-tolerance table-key error after the actual full encoder, next RMS, fused matrix and QKNorm; query self-values and matching keys are zero. | Robust finite-softmax latest-write selection and tied readout, complete validated raw parser coupling and full depth correctness; convex architecture and FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Actual raw normalized projections committed in 2768f32. Proved actual raw query/table score error at most 2*exp(alpha)*copyTolerance against the categorical rotary reference, without a factor 1/epsilon; raw key competitors have zero score and query self-values stay zero. | Derive robust latest-write gaps, actual finite-softmax retrieval/tied readout, full raw parser and depth correctness; convex architecture and FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Actual raw score perturbations committed in 3f1128d. Derived a positive exp(alpha)*latestMargin/2 gap against real different-key and earlier same-key raw table writes, including both actual copy errors and context bounds. | Exclude nonrecord scores and derive the full causal-row gap, then finite-softmax retrieval/readout, full raw parser and depth correctness; convex architecture and FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Robust raw table gaps committed in 51ec631. Proved exact zero BOS/post-table filler scores, positive lower score for an imperfect real matching record, faithful actual selected V with its genuine RMS multiplier, and uniform actual V norm at most sixteen. | Assemble full-row routing and finite retrieval, connect tied readout and validated raw parser, then full depth correctness; convex architecture and equal-FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Nonrecord exclusion/actual V bounds committed in 377aeaa. Derived a full genuine score-row gap from raw table adjacency and last-write chronology, actual causal-softmax tail and original XSA retrieval error 32*(T-1)*exp(-retainedGap); the real overwrite control strictly prefers its later write. | Choose uniform finite retrieval accuracy, realize actual second W_o/tied readout and discharge raw predicates from complete Basis parsing, then full depth correctness; convex architecture and equal-FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Full raw routing/retrieval committed in 19545b9. Chose a finite logarithmic shared temperature and proved uniform actual head accuracy, including fixed positive secondScaleLower/16 tolerance and finite output-gain product two. | Realize genuine second W_o/tied readout and discharge raw layout/latest-write conditions from complete Basis parsing, then full depth correctness; convex architecture and equal-FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Finite uniform retrieval accuracy committed in 94d0b89. Realized the actual unchanged second attention/block with ordinary nonzero W_o, finite shared matching parameters, complete state formula, protected query/type coordinates and true headAt/output-error coupling. | Strict tied-readout margins and actual integer decoder, full validated raw parser discharge and depth correctness; convex architecture and equal-FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Genuine second block committed in 5e1dc59. Proved full genuine final-state error at most 1/8, selected real value amplitude at least two and faithful final raw query/type/reserved readout channels. | Strict tied readout over all 548 tokens and actual integer decoder, complete validated parser discharge and depth correctness; convex architecture and equal-FLOP comparisons remain queued. |
| 2026-10-07 | 1 | Full final-state/error coupling committed in 486a860. Evaluated actual tied key/reserved scores and selected-value reference scores; proved compact categorical products four versus at most three and uniform raw score-error amplification at most three. | Derive strict all-token readout, connect actual complete ModelParams/decoder, discharge full validated recall parsing, then depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Tied-score identities committed in 1315cac. Proved strict actual answer margin over all 547 competing tokens, genuine final RMS positivity and greedy selected-value correctness; the actual six-token overwrite control decodes its later value. | Complete top-level model and integer adapter coupling, validated-parser discharge and depth correctness; convex architecture and equal-FLOP comparison remain queued. |
| 2026-10-07 | 1 | Strict tied recall decoding committed in d86fa0a. Assembled complete original two-layer ModelParams, connected both actual hidden states and all final forward logits, and proved its checked List Int continuation appends the selected raw value, including the literal overwrite control. | Discharge raw layout/adjacency/latest-write premises from every validated Basis prefix, then full depth correctness; convex architecture and equal-FLOP comparison remain queued. |
| 2026-10-07 | 1 | Complete model/integer coupling committed in a4ca072. Strengthened successful raw Basis parsing to retain all earlier/later table and full query/filler range checks together with chronological last-write decomposition. | Transfer this validated raw decomposition to exact finite-array positions and discharge all routing predicates, then recall SolvesTask and full depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Complete validated parser conditions committed in 52d6321. Proved exact serialized key/value positions, every true table value's immediate raw key predecessor, the actual BOS offset and a last-write upper bound on all matching record indices. | Couple these actual integer positions to the checked finite arrays and discharge raw layout/latest predicates, then recall SolvesTask and full depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Exact table serialization/chronology committed in 5507968. Transported genuine adjacency, row recovery and last-write bounds through full raw BOS/table/query prefixes; derived selected raw positions, the final actual query index and all non-BOS alphabet positions. | Transfer these integer facts to checked finite arrays and prove complete recall SolvesTask, then full depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Full-prefix integer positions committed in 4272535. Coupled genuine raw integer reads to the exact finite input tokens, deriving position bounds and actual key/value/BOS IDs, with reverse exact key/value serialization at the same positions. | Discharge all raw layout/latest-write conditions from successful parsing and prove complete recall SolvesTask, then depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Exact finite/integer coupling committed in 9ea43d1. Derived full actual RecallRawLayout and final compact query solely from successful validated Basis parsing, including finite-position bounds and genuine key/value adjacency; the overwrite control's layout is now parser-derived. | Derive the finite selected last write and its chronology, then complete easy/hard recall SolvesTask and full depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Complete parser-derived raw layout committed in 75c8c94. Derived the actual finite selected write, neighboring key, final matching query, answer ID and chronological upper bound solely from successful full raw parsing, including genuine earlier overwrites. | Complete easy/hard recall SolvesTask through the actual List Int adapter, then full depth correctness; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Parser-derived finite last-write conditions committed in 643f78d. Proved full recall SolvesTask for both Basis grammars through the actual integer model callback, with no layout/selected-write/logit premise. Actual eight-write, sixteen-write and swapped-binding controls are covered. | Complete original depth correctness, retaining all real-arithmetic/shared-epsilon and compact-model scope distinctions; convex architecture/FLOP comparison remain queued. |
| 2026-10-07 | 1 | Full raw recall correctness committed in c30b8c3. Proved an ordinary three-unit homogeneous ReLU2 presence step, raw-type exclusion, exact separated Boolean amplitude and positive prenorm-scaled margin for the depth construction. | Realize and bound genuine prefix detectors in the original attention/FFN matrices; complete depth SolvesTask before the convex architecture/FLOP stages. |
| 2026-10-07 | 1 | Homogeneous depth FFN gate committed in fe30f91. Proved the true uniform softmax/XSA causal mean for nonconstant real value amplitudes, exact absence, upper cap and occurrence floor L/128; derived the separated presence input without unit-amplitude assumptions. | Derive these local bounds from real raw embedding/matrices and propagate the complete ordered depth recurrence, then final SolvesTask. |
| 2026-10-07 | 1 | Actual variable-amplitude causal presence committed in df46fd4. Realized both depth type/presence gates in six ordinary shared FFN units; proved actual matrix coordinates, complete FFN output, protected residual channels and the true RMS quadratic scale at generic dimensions. | Connect raw embedding and fused uniform attention, propagate quantitative ordered-feature bounds and complete depth readout/SolvesTask. |
| 2026-10-07 | 1 | Simultaneous original depth FFN matrices committed in 830e7fc. Proved neutral-preserving original-position alternating occurrence recurrence, opposite-current exclusion and exact independent E_2/E_4 integer answer criteria. | Realize these data predicates by actual raw embedding/head matrices and quantitative hidden-state induction, then full depth SolvesTask. |
| 2026-10-07 | 1 | Ordered depth recurrence committed in 0f9d21c. Defined genuine token-local tied depth embeddings in the original 64x2 and 128x6 configurations; proved working-axis separation, actual raw constant, empty later channels and input norm bounds [1,2]. | Realize fused uniform heads, propagate actual normalized amplitudes and complete depth readout/SolvesTask; no convex candidate or FLOP comparison has started. |
| 2026-10-07 | 1 | Actual original-width depth embeddings committed in 949d2c1. Proved exact original uniform-head/XSA formula for arbitrary current amplitudes, attenuation in [0,1], nonnegative genuine probes and the undoubled value cap. | Couple simultaneous fused heads to the real residual stream, propagate quantitative feature amplitudes and complete depth SolvesTask; convex architecture and measured FLOPs remain conditional. |
| 2026-10-07 | 1 | Full original depth-head attenuation committed in f479eb7. Realized simultaneous ordinary fused QKV at both original widths; proved all actual Q/K slices zero, both active V scalar reads and unused-head zeros using genuine chunk/view indices. | Couple W_o and genuine RMS scaling, prove quantitative hidden-state recurrence and full depth SolvesTask; convex search and equal-FLOP training remain next stages. |
| 2026-10-07 | 1 | Original simultaneous depth QKV committed in afb6840. Proved actual RMS multiplier bounds at both widths and shared finite threshold/FFN/readout compensation; the local [1,4096] norm domain is explicit. | Derive the domain and feature amplitudes through the actual residual blocks, then full depth SolvesTask; no convex or training-success claim follows from these given weights. |
| 2026-10-07 | 1 | Genuine depth RMS/gains committed in 18fff55. Coupled complete original two-head attention, true RMS, real head merge and nonzero W_o; proved exact output signals, zero protected contributions and independent target reads. | Derive quantitative actual signal/state bounds and ordered hidden-state induction, then full depth SolvesTask before convex search and equal-FLOP tests. |
| 2026-10-07 | 1 | Complete original depth attention committed in e40f61f. Derived actual normalized coordinate bounds, r^3 feature floor on the explicit state domain, true signal range [0,16], whole attention norm at most 32 and first residual bound M+32; proved actual zero-self and visible absence. | Derive occurrence/FFN transitions and close the actual hidden-state induction, then tied readout and depth SolvesTask. |
| 2026-10-07 | 1 | Genuine attention/residual bounds committed in d327192. Proved true occurrence signal at least 2*threshold, exact positivity iff visible separated feature, and zero-or-gap input for the ordinary FFN, with simultaneous concrete operator witnesses. | Connect these actual head signals to the true FFN transition, derive all representation conditions by ordered layer induction and complete depth SolvesTask. |
| 2026-10-07 | 1 | Actual signal separation committed in d013328. Proved the complete original six-unit FFN record, exact zero-or-true-RMS-square output, arbitrary protected coordinates and contribution norm at most 256. Corrected the gap condition to matching raw types only; wrong types need the actual cap. | Assemble genuine detector blocks, derive all norm/feature conditions by ordered raw-state induction and complete depth tied readout/SolvesTask. |
| 2026-10-07 | 1 | Complete depth FFN committed in 199706f. Assembled fixed actual detector BlockParams for all three stages, coupled each real source to the previous opposite-ending target, and proved preservation of raw axes and other stages through both residuals. | Derive actual pre-FFN representation and M+288 block bounds, propagate the ordered raw-word features, then complete depth readout/SolvesTask. |
| 2026-10-07 | 1 | Fixed original detector blocks committed in beb4e74. Proved actual pre-FFN raw/signal/feature reads and genuine norm bounds; recorded explicit simultaneous local input conditions with an active-A real-array witness. | Derive FFN conditions and complete M+288 transitions from these inputs, close ordered raw-state induction and full depth readout/SolvesTask. |
| 2026-10-07 | 1 | Actual pre-FFN residual committed in 135a904. Derived all true FFN requirements from incoming states, proved exact complete-block flags and their positive iff condition, and bounded both real residuals by M+288. | Derive separated flag amplitudes and ordered occurrences through all raw stages, then complete tied readout and depth SolvesTask. |
| 2026-10-07 | 1 | Complete detector transitions committed in f8a76c9. Derived real flag separation/cap, threshold iff positivity, exact wrong-type zero and strict earlier-source semantics through the unchanged causal head/FFN. | Derive initial and propagated representation directly from raw words, then complete original full-model depth readout/SolvesTask. |
| 2026-10-07 | 1 | Real flag bounds/strict precedence committed in c753c3b. Derived all first-detector conditions from actual token-local word embeddings, including raw-type exclusion, fresh channels, genuine norm [1,2] and BOS/neutral embedding coupling. | Propagate quantified ordered-occurrence representation through the real hidden stack, then finish depth tied/greedy readout and checked integer SolvesTask. |
| 2026-10-07 | 1 | Raw detector inputs committed in 4e30c42. Proved the complete actual ordered detector encoder by induction from raw words: exact independent occurrences, zero-or-[r^2,128] flags, protected types, fresh future channels and norm at most 866. One detector for easy and three for hard fit the original layer budgets. | Finish genuine readout, full ModelParams/hidden-loop coupling and checked List Int depth SolvesTask; stage 2 remains open and stage 3 conditional. |
| 2026-10-07 | 1 | Actual encoder induction committed in 14a2f0d. Proved upper readout-axis freshness through the full detector prefix and realized genuine residual probes that preserve current flags despite XSA suppression. | Prove quantitative visible-occurrence readout, actual FFN/tied margins, full ModelParams and integer adapter correctness. |
| 2026-10-07 | 1 | Genuine residual readout layout committed in f3e61b2. Derived actual probes zero or at least twice the shared threshold, capped by 144 and positive exactly at visible independent ordered occurrences, with true pre-FFN norm at most 898. | Realize the seven-unit FFN and tied margins, then full ModelParams/checked integer depth correctness. |
| 2026-10-07 | 1 | Actual readout separation committed in 6ba8bea. Realized all seven ordinary readout FFN rows and proved the complete true matrix/prenorm formula with a common RMS-square scale. | Couple exact semantic flags to the full readout block, prove tied margins, then original ModelParams and raw integer correctness. |
| 2026-10-07 | 1 | Original seven-unit matrices committed in a705ed2. Derived exact actual readout FFN binary/common amplitudes, protected all other coordinates and proved contribution norm at most 384 with concrete simultaneous witnesses. | Derive semantic complete-block readout and tied margins, then full ModelParams/checked List Int depth SolvesTask. |
| 2026-10-07 | 1 | Actual FFN amplitudes/bounds committed in 2d14aea. Proved the complete genuine readout block and its raw-word instantiation: independent semantic flags, common true scale [r^2,128], raw protection and final norm [1,1282], without an encoder/invariant premise. | Strict tied 36-token margins, original ModelParams/hidden-loop and raw integer depth SolvesTask remain. |
| 2026-10-07 | 1 | Complete real raw-word readout committed in 59cf964. Proved all 36 tied score comparisons, genuine final RMS margin and greedy semantic answer; linked last-position visibility to full ordered-pattern presence. | Original two/six-layer ModelParams, complete hidden-loop and checked List Int depth correctness remain. |
| 2026-10-07 | 1 | Whole-vocabulary depth decoding committed in 4dec64b. Realized original complete two/six-layer ModelParams, proved every actual detector/readout/tail loop state and full forward strict logits. | Discharge raw BOS/body token-local embedding coupling and prove checked integer depth SolvesTask, then aggregate all Basis semantics. |
| 2026-10-07 | 1 | Complete original depth model committed in d3aae47. Proved exact raw integer serialization and universal depthModel_solves_depth in both full Basis modes, plus actual neutral/wrong-start/order controls and append-one contract. | Consolidate all three tasks/two modes with explicit real-arithmetic, epsilon and width scope; then start stage 2 convex architecture search. |
| 2026-10-07 | 1 -> 2 | Raw depth correctness committed in e1afb3d. Consolidated the actual original family in BasisCorrectness: every task/mode, independent next-token agreement, two-call parity and exact append-one contract. Common epsilon and small/large dimension boundaries are explicit. | Search changed compact embedding/attention operators with jointly trainable matching/values and a proved convex objective; do not start equal-FLOP training until an admissible candidate exists. |
| 2026-10-07 | 2 | Complete Basis family committed in d640a91. Established the actual finite affine-energy joint Gibbs NLL's global convexity as a structured-head foundation. Route/state/channel supervision is explicit, and marginal output CE is not claimed convex. | Prove compact latent contraction and actual changed head capability/interface; assess all acceptance constraints before equal-FLOP training. |
| 2026-10-07 | 2 | Joint likelihood foundation committed in 19ff013. Proved exact compact matching/value partition contraction and global convexity of the actually computed shared-table pointer loss/minibatches, including all raw Q/K/value/chronology weights. Proved the 52-field count fits width 64 with decoder slots. | Prove exact compact value statistics, raw order/adjacency/role handling and complete task capability; instantiate prenorm/residual/tied interface before stage 3. |
| 2026-10-07 | 2 | Compact joint pointer committed in 87ffa1b. Proved exact compact output means equal the same jointly trained Gibbs distribution's actual value expectation, including normalization, coordinate bounds and shared-parameter inference/training coupling. | Check learned content/binding controls, derive role/chronology semantics and ordered-state head capability, then realize the public residual/integer interface. |
| 2026-10-07 | 2 | Exact value contraction committed in 5371b5f. Proved content-dependent routing and raw adjacent-value swap controls for the same finite learned Q/K/value table, with strict actual decoded labels and a globally convex unrestricted complete objective. | Extend from two-record control to full raw recall, prove ordered-state head capability and realize genuine residual/tied/integer architecture. |
| 2026-10-07 | 2 | Raw learned pointer controls committed in aa27386. Proved the actual compact causal state encoder's normalized positive distributions and actual observed-path likelihood's global convexity in the unrestricted shared transition table. | Prove exact complete-path marginal/inference identity, conditional value emissions and full depth/parity paths; full recall and residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Causal encoder/path training committed in 11899f6. Proved exact normalization and endpoint marginal contraction of the same actual full path model into compact forward state propagation, for every unrestricted initial/transition table. | Jointly learned output emission/objective, raw semantic depth/parity paths, complete recall and actual residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Exact causal path contraction committed in f1f6c9c. Added free conditional output-channel potentials and proved their compact inference means equal the actual complete normalized state/path/channel model's value expectation at arbitrary joint parameters. | Full initial/transition/emission training convexity, raw depth/parity capability, complete recall and true residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Exact conditional emissions committed in 33e99c8. Proved the actual compact complete NLL equals that same inference model's joint negative log probability and is globally convex jointly in every initial/transition/value weight, including shared minibatches. | Realize shared compact raw parameter slots and full raw depth/parity capability; full recall and true residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Joint state/value objective committed in 95456c1. Realized actual shared raw slot lookups, disjoint pointer groups, independent 36 transition fields and the convex state objective on a 52*V+C+129-parameter domain; embedding slot counts include a learned position axis. | Full raw depth/parity capability, pointer learned-position/chronology integration and complete recall, then actual prenorm/residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Shared compact coordinates committed in bdbea00. Realized jointly free raw Q/K/value/position/chronology pointer energies, exact compact partition and inference/training identity, and global complete-likelihood convexity on the same actual parameter domain. | Full raw recall/latest-write/filler exclusion, depth/parity capability, two-head mixture and actual prenorm/residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Shared learned-position pointer committed in 7598fbc. Proved actual finite Gibbs/transition-row rival and selected-mass bounds, and explicit finite sharp-row confidence witnesses. | Derive gaps from raw semantic shared parameters and propagate confidence to full depth/parity/recall outputs; mixture and genuine residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Finite confidence laws committed in d68381b. Derived actual full chronological path confidence with linear T*epsilon error, true reference-path endpoints, and inclusion of each initial/path probability in the compact encoder's endpoint mass. | Prove task-specific reference rules against raw Basis, realize their finite shared raw weights and conditional output confidence; recall, mixture and actual residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Actual causal confidence propagation committed in 842cc59. Proved the six-state data/reference rule's counted parity and label/EOS agreement on every actual validated raw Basis parity prefix, with no external phase/count input. | Realize finite learned initial/transition/value witnesses and prove full parity model margins; raw depth/reference rules, complete recall, mixture and genuine residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Full raw parity reference semantics committed in 91299ae. Proved actual compact encoder and full initial/path/value finite-table confidence, including a uniform 1-794*exp(-gain) joint-mass bound up to the largest Basis context 128. | Embed finite witnesses in actual shared raw slots and prove tied whole-vocabulary decoder margins; depth, recall, mixture and true residual/integer integration remain. |
| 2026-10-07 | 2 | Joint causal/output confidence committed in 61226da. Proved ten-coordinate codes for all 1024 output IDs, the quantitative 11*p_correct-10 full-distribution margin, equality to actual compact Markov decoding and strict all-token margins for finite gain log(100000) through length 128. | Realize these finite tables in actual shared raw embedding fields and prove raw parity capability; full depth/recall, mixture and true residual/tied/integer integration remain. |
| 2026-10-07 | 2 | Whole-vocabulary compact decoder committed in d84bd84. Constructed actual finite shared embedding/head weights, evaluated every genuine transition/initial/emission lookup and connected the real shared loss/output score to the same learned finite-row model and strict margin. | Prove the actual shared head's raw integer parity continuation and two-call generation; full depth/recall, mixture and residual/tied block integration remain. |
| 2026-10-07 | 2 | Actual finite shared causal weights committed in d72d3b3. Proved the genuine standalone stochastic state/value head solves every raw parity prefix in both modes through the checked integer-list/greedy callback, including two-call label/EOS generation; actual data targets agree and their full likelihood is convex in the free shared domain. | Prove independent raw depth reference semantics and full learned depth/recall capability; two-head mixing and true prenorm/residual/tied tensor-stack realization remain before stage 3. |
| 2026-10-07 | 2 | Actual raw learned parity head committed in b91036f. Proved the independent raw six-state depth data scan exactly matches Basis E2/E4 via genuine compression/chain/subsequence invariants, physical neutral/BOS transitions and saturated run counts. | Realize full finite shared learned depth capability and data targets; complete recall, two-head mixing and actual tensor-stack/tied/integer integration remain before stage 3. |
| 2026-10-07 | 2 | Independent full raw depth semantics committed in 6c91a98. Proved the genuine finite shared learned head solves all raw depth prefixes in both modes, computes order/mode controls, and has correct raw-data complete labels with a globally convex actual shared likelihood. | Extend the jointly learned pointer to full raw recall/latest-write/filler exclusion, prove finite two-head mixing and actual prenorm/residual/tied tensor-stack realization before stage 3. |
| 2026-10-07 | 2 | Full standalone learned depth capability committed in 3812fda. Replaced the proposed fixed neighboring-key shift by free relative binding over every visible position pair; proved exact compact inference/training coupling and unrestricted joint convexity, with 28816 recall scalars. | Derive the finite all-pair pointer's complete raw recall/latest-write/filler semantics and decoder margin; mixture and full tensor-stack realization remain before stage 3. |
| 2026-10-07 | 2 | Joint learned positional binding committed in f37438a. Derived actual raw even table slots, both unchanged key/value reads and latest-write chronology directly from complete parsing, with genuine overwrite/distinct-query/post-table-filler controls. | Construct finite free shared pointer weights and derive their uniform complete-configuration gap and full-vocabulary recall decoder; combine heads and realize the actual tensor stack before stage 3. |
| 2026-10-07 | 2 | Full raw record/overwrite positional facts committed in 9e6661b. Proved genuine finite learned-channel deficits and constructed/evaluated the complete shared recall token/position/chronology/binding/head weight assignment, with all 256 raw keys distinguished. | Derive the true whole all-pair recall configuration gap and probability/readout margin, then complete raw recall capability, head mixture and actual tensor-stack realization. |
| 2026-10-07 | 2 | Actual finite shared recall fields and channel deficits committed in b127359. Connected true ten-axis pointer decoding to the full normalized joint distribution; realized actual unchanged raw-prefix all-pair inference/loss, proved unrestricted joint convexity and the exact implicit choice-count bound. | Prove uniform raw finite-weight recall energy gaps and actual greedy List Int correctness, then two-head mixing and true prenorm/residual/tied tensor-stack realization before stage 3. |
| 2026-10-07 | 2 | True raw all-pair inference/loss/decoder coupling committed in e0be315. Derived complete raw recall finite-weight energy gaps for every rival and the actual correct configuration's whole-distribution probability bound; proved a finite logarithmic gain covers the latent and two-head tails. | Complete strict whole-vocabulary raw recall greedy/List Int capability and genuine mixture training/inference, then realize the actual tensor/prenorm/residual/tied block before stage 3. |
| 2026-10-07 | 2 | Full raw recall gap/confidence committed in e5e8823. Proved the actual finite learned all-pair head solves both complete raw recall grammars through the checked append-one integer interface, with genuine value-swap control and derived whole-vocabulary greedy margins. | Correct complete recall data targets, true learned two-head mixing and full tensor/prenorm/residual/tied realization remain before stage 3; no candidate training started. |
| 2026-10-07 | 2 | Full standalone raw recall capability committed in 05fb3e2. Realized genuine learned mixing of the actual state and all-pair binding distributions, exact compact mixed decoder expectation, and globally convex actual complete NLL/shared minibatches on the full unrestricted joint parameter space. | Derive complete mixed finite-weight confidence/Basis capability, correct recall data targets and actual tensor-stack/prenorm/residual/tied realization before stage 3; output-only CE and AdamW success remain unproved. |
