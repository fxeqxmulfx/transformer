# Basis correctness and convex architecture research cycle

Started: 2026-10-06. Status: active, stage 1.

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

Status: active.

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

Next connect the fixed ordinary second QKV to robust latest-write retrieval
and readout, discharge full validated-parser conditions, and construct the
depth-prefix recurrence.
Record each remaining assumption and discharge it rather than moving it
into a definition. A failed construction should produce a counterexample
or a precise missing condition, not a weakened correctness target.

Exit criterion: complete task theorems connected to the actual model
function, with no unproved encoder or routing input and no added sorrys.

## 2. Search in Lean for a convex architecture

Status: queued after stage 1; structural analysis may proceed alongside it.

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
