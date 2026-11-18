# Project experiment plan: why sparsemax attention fails, and a repair

## Current full comparison status (2026-11-13)

Use each stress task's declared success criterion, per the user's 2026-11-13
instruction. For these selected35 tasks this is the original primary 99%
metric. Additional held-out probes remain diagnostics and do not revoke a
declared success. The comparison covers the selected25 short and10 long tasks,
not every case in `python_projects/diff-gpt/stress_tests_for_gpt.md`'s165-case
historical catalog. Keep the existing conventional transformer baseline.
DiffGPT's existing selected research-main record reports35/35 for each seed
0/1/2:75short and30long successes,105/105 total. Its
`research/complex_geometry_suite/suite.json` is byte-identical to ours
(SHA256`6d2735e987026959c3cde995fbe866281e4f0b2308cc71be4b9d2edd13840436`).
The committed `research/main_model/model.json` and README identify the selected
`phase-score-lerope` arm and record zero audit failures. Generated run/audit
artifacts are absent from that checkout, so this is an existing recorded result,
not a fresh replay or a new integrated endpoint. No replacement baseline run
is needed to establish what the prior experiment reports.

The reviewed comparison succeeds on34/35 tasks in full-batch and minibatch
modes, and33/35 forward-only. Both initializer policies have these counts and
retain all105 task/mode choices. The stricter all-held-out diagnostic counts
remain33/33/32; frozen reports and their historical flags are unchanged. Task226's
independent replay rehashes 6786 artifacts and reproduces the complete report
byte-for-byte. It compares saved endpoints using observed CE, with all earlier
failures and costs retained; it is not a new joint optimizer or matched-compute
comparison. Tasks175,191,225,226 and227 are approved.

The conventional transformer succeeds on33/35 tasks; its independent review
reproduces all35 endpoints and the complete summary. Its all-held-out diagnostic
count remains32/35. Fibonacci recurrence still fails in the convex comparison,
and forward-only also lacks a branch-amplitudes model. Shifted Boolean AND is
a declared success with a retained out-of-distribution diagnostic failure.
The goal of succeeding on all35 tasks remains open.

Task245's complete causal import and initial independent replay have now closed
with zero process failures. Both policies still pass34/34/33 primary tasks,
and their all-split counts improve to34/34/33. Each selects17 new endpoints:
forward branch amplitudes gains a100% model, while forward associative divisor
regresses from100% to53.86598% despite near-zero raw training CE. Two failing
key-value candidates remain unselected. The full transformer/depth chain and
all refusals/costs remain. Fresh submitted-commit review is pending; see the
[authored comparison](experiments/convex_stress_causal_pairs/cohort/original/README.md).
Its complete summary SHA256 is
`fa07679de40f3662f2ae6177a40f0f8fa67b8504dfa9a83b74a8b8b4884c2c14`.
This completed result changes which forward task fails without increasing the
aggregate count; no full-pass claim follows from the successful new endpoint.

Tasks229–231 implement colliding-code discovery, bounded complex latent states,
ordinary full-vocabulary CE and the complete original adapter. Independent
review passes 18 model tests and 16 adapter tests, including all training modes,
actual saved states, original-row certificates, mutations and full parent scope.
The proved CM30 and CD8 statements support conditional code certification and
convex fixed-feature CE. They do not identify arbitrary unseen raw labels.

Every one of the 55 prime stages is retained. Only codes compatible with the
original declared metric readout enter the observed-CE selector; that filter
never changes features, labels or CE. A synthetic pooled-metric fixture passes
every decoded split while failing its unseen raw token. Task232 has completed
the separately frozen original run at producer `49105730`:23 pair tasks and
1265 stages, all independently replayed, with no process failure. Both complete
parent arms still pass34/35 full and mini and33/35 forward on the declared
primary metric. Eighteen new endpoints per arm replace already-successful
parents, with no accuracy decrease or loss increase. All35 transformer
endpoints,336 depth outcomes and prior costs remain. Of337 accepted codes,
six supply compatible offers;653 stages have no nonconstant direction,39
retain nullity refusals and236 retain2B head-work forecast refusals.
Construction costs8919.143533953 seconds and91664 actual gradient batches;
the complete summary peaks at1291828KiB under64GiB. Its immutable SHA256 is
`ddac2993d6b9e00552daf43519d5ff83c8810632bc9ae0d49dca97659bff9657`.
Fresh review at the submitted commit will rerun every original stage and
rebuild that complete summary before approval. Fibonacci still admits no code.

Reviewed task233 adds CFt4 with nine Lean theorems. Paired visible features
admit a forward decoder when no individual slot suffices, while the channel's
ordinary CE stays convex. Presence-feature invariance and a candidate-order
counterexample make the required selection assumptions explicit. Task234 will
implement and test this generic causal repair before a separate original
evaluation. Its native component passes 15 tests: generic causal discovery,
actual complex Fourier features and a convex full-vocabulary CE head in all
three modes. Independent complex derivatives, finite differences, trajectory
and refusal replay, unseen masks and the candidate-order obstruction retain
the Lean boundary. A full-batch cap preserves available mini/forward models.
The complete original adapter now passes 15 synthetic tests, preserving
both parent arms, all35 tasks, transformer/depth outcomes, process failures,
actual greedy rollout and unknown-prefix masks. Original execution awaits
task232's completed independent review and task234's approval; no fixture
result is counted as a benchmark pass. The running quotient protocol remains
unchanged.

Task235 extends Lean with CFt5's 28 theorems binding the actual complex head
to CD8. Conjugate initialization gives exact recorded-state logits and
ordinary CE; its real/imaginary L1 column budget fits the native margin24
and radius64-times-total-states choices. Arbitrary nonzero Fourier blocks,
including unequal query sizes, have unit singular values, and the fixed
code bounds every feasible head's logits. The unobserved-label and feature
selection limitations remain explicit. This proof extension does not alter
the frozen numerical protocols or advance the benchmark pass counts.

Task236 separates fixed causal construction from the eventual comparison.
After implementation review, a separately frozen training-only stage can build
all 35 unchanged proposals and independently replay every completed actual model,
without a parent selector or held-out metric. Seven additional adapter tests
check fresh/staged equivalence, parent independence, rehashed numerical mutations,
all scope/provenance bindings, failed imports and retained costs. Import still
requires the original reviewed quotient parent and comparison approvals, uses
the same frozen producer/executable, replays the actual state before training-CE
selection, and charges construction once across both complete parent arms.
Every failure and partial resource record remains. The numerical core, feature
order, raw labels, mode policies and caps are unchanged from `78731eb0`.
Task236 is reviewed and task237's original training stage is complete at frozen
producer `b516af90`: all35 process outcomes and independent replays,19 usable
endpoints on nine tasks (full3,mini7,forward9), zero observed-token errors.
It retains23 two-token shape refusals, two depth input-storage refusals, one
recall feature-cache refusal and eight iterative work-cap outcomes. Constructor
time is5.358890285 seconds; peak process RSS is364424KiB under64GiB. Fibonacci
is a two-token exclusion and still needs a separate repair. No held-out metric
or parent selection has run; the reviewed complete comparison counts remain
unchanged. The same frozen executable must import this immutable stage after
the original parent approvals.

Task238 / CM31 proves the next repair before native implementation. A head
that gives two differently labeled observations the same logits has pairwise
mean CE at least log2. Refining the fixed state by observed-input identity
removes that collision while retaining the learned quotient on unseen inputs.
The actual complex Fourier head has exact ordinary CE, an explicit enlarged
budget, convex coefficient training and correct compatible pooled decoding.
Affine constraints can retain the declared default logits at every feasible
iterate. A four-class witness fits every observed raw label but remains wrong
on an unseen alias for one of two indistinguishable targets. Code discovery,
unseen quotient identification, efficient storage and actual constrained
updates remain separate obligations; no benchmark score changes from this proof.

Task239 / CM32 closes the exact-real update and storage gaps. Every bounded
real logit table transports to an actual complex head within64 times the
refined state count. Clipped observed proposals and fixed default rows form
a feasible update in a convex table domain with ordinary CE. Latent-fiber
Fourier blocks remain unitary and need the sum of squared fiber sizes in
phase storage; a maximum fiber sizeB gives a boundB*N. Every projected update
retains the unseen defaults and any separately identified compatible decoded
answer. Native derivative checks, numerical feasibility and capped original
evaluation remain required; no new benchmark success follows from the proof.

Task241 implements CM31/CM32 in `convex_stress_refined_quotient`. Fourteen native
checks pass: actual block Fourier coefficients, observed raw-alias fitting,
unchanged default logits, feasible clipping, independent complex derivatives,
finite differences, unequal weighted batches, full trajectory/JSON replay,
Gram/SVD, altered states and pre-allocation/work-cap refusals. Fixed discovery
precedes convex full-vocabulary CE. The prospective policy is one full update,
one original minibatch epoch or zero forward gradients, each after initializing
from every observation; it does not reuse the old16/256 schedule. The new
32-billion scalar-work allowance and64GiB memory limit are explicit. Original
construction and comparison require a separately reviewed adapter and freeze;
the232/237 artifacts and all existing benchmark counts remain unchanged.

Task241 is reviewed at `c55fe210`. Task242 adds a separate training-only adapter
over the immutable232 discoveries, with seven native adapter checks. It verifies
the frozen producer/source/binary, complete55-stage ledgers and original input
and minibatch order before refinement, retaining every source refusal and cost.
Actual states and CE trajectories replay independently, including mutations with
recomputed file/ledger hashes. Every23-task process outcome and35-task metadata
remain; partial process failures keep their files and resource records. This
stage has no held-out evaluator or parent selector. After its own review and
the source catalogue closing, a separate original freeze may construct it;
comparison still requires the reviewed original parents. No benchmark count
changes from adapter fixtures or training-only availability.

Task243's frozen original queue has now started after all232 source outcomes
and the complete summary closed successfully. It imports the immutable
discoveries, preserves every source refusal and constructs no new code.
Its original training, replay and review remain in progress; no held-out
evaluation or new comparison score is implied by starting this stage.

Task244 implements the separate refined comparison with seven native checks.
It imports the complete reviewed causal parent and the separately reviewed243
training stage, verifies actual complex states before raw-CE offers, and seals
both105-choice selections before metrics. The complete35-task fixture retains
unequal probability groups, unknown raw aliases, real mode refusals, process
failures, masks, all prior costs,336 depth outcomes and35 transformer endpoints.
Primary success controls `full35_pass`; auxiliary splits remain diagnostics.
The new32B per-endpoint metric forecast and64GiB process limit are prospective.
Original evaluation still requires the closed reviewed232/243 evidence and
task245's full causal comparison using the preserved237 stage. No fixture
result changes the reviewed benchmark counts or DiffGPT's recorded105/105.
Task246 tracks the separate original evaluation after243,244 and245 approval,
including every original process outcome and the complete independent replay.

Task247 / CM33 derives the full-vocabulary ordinary-CE direction through the
actual complex head and its normalized batches of distinct observed inputs.
For feasible rows, any nonnegative projected step raises or retains the
target score, lowers or retains every rival, and cannot increase actual CE.
Every finite full/minibatch trace preserves the margin24 observed winners,
the64N coefficient budget and fixed unseen defaults. Concrete nonzero
derivatives distinguish full-batch and minibatch normalization. This exact
arithmetic preservation result supplies neither unseen-label identification
nor a convergence or floating-point execution theorem. Frozen232/237/243/244
implementations, active runs and benchmark pass counts remain unchanged.

Task248 isolates an observed-equation coverage limitation before changing the
model family. The frozen Fibonacci catalogue admits no code: its modulus97
system has five free directions, and the4096-row discovery prefix omits raw
class33 from the4704 training observations. A separately frozen training-only
pilot raises only the supported equation allowance to8190. It reuses the
reviewed executable, all55 primes, work/nullity limits, original raw CE and
mode policies. CM30 already permits arbitrary observed row sets; it supplies
no claim that this larger system has a unique useful direction. The pilot
preserves every refusal and independently replays the catalogue, without a
held-out evaluator or parent selector. Original232/237/243/244 runs stay fixed.
The pilot is now complete: all55 stages replay, zero codes are admitted and
no CE gradients execute. At modulus97, full observed coverage reduces nullity
from five to three. Thirty-nine stages retain nullity refusals;16 retain the
existing2B head-work preflight refusal. Construction costs182.576 seconds;
run/replay peak RSS is91160KiB. The one-direction restriction remains a real
obstacle after removing the prefix omission. Task249 tracks the Lean-first
extension to complete multidimensional coefficient kernels; no benchmark
success or unknown-label identification follows from this diagnostic.

Task249 / CM34 removes the mathematical one-direction restriction. The
complete kernel of the original observed equations, with only two shift
gauges, gives vector codes across every basis direction. Matches and class
aliases are basis independent. A checked match identifies an unknown target
code under explicit additive realizability; decoded correctness additionally
requires the original readout to be constant on that unknown code's fibers.
Concrete two-dimensional and noninjective witnesses retain refusals and raw
alias ambiguity. The finite actual code range has at most the active input
product's cardinality and instantiates the existing bounded complex head,
convex ordinary CE, full/minibatch preservation and conditional unseen pooled
decoding. Code discovery remains separate from convex head training. Native
complete-basis construction and its storage/work checks are still required;
no frozen numerical source, protocol or benchmark result changes here.

Task250 implements CM34 in the separate `convex_stress_kernel` component.
It retains all original equations and every kernel direction, independently
checks original-system replay and basis completeness, and uses only the actual
active-pair vector range for the bounded complex head. Unknown axes and
unmatched states retain explicit masks. Twenty native tests cover multidimensional
and sheared kernels, false scalar matches, raw aliases, actual complex forward
and CE derivatives, all three modes, mutations and resource refusals. Full and
mini updates preserve fixed defaults; forward computes no CE gradients.
The prospective head forecast reserves replay and serialization buffers below
56GiB with a hard64GiB process limit. The current original protocols and
benchmark counts remain unchanged. Review this component before a separately
frozen original adapter/cohort; runtime code matches still do not establish
the unknown-target or readout hypotheses of CM34.

Task243's frozen original refined training has now closed: all23 cases and1265
stages complete and independently replay, with zero process failures. Six
modulus97 stages provide full/mini/forward endpoints, each with zero observed
raw errors and CE about3.62e-9. The ledger retains331 Fourier shape refusals,
692 stages with no source code and236 source-stage errors. Fibonacci remains
unavailable. Construction costs185.750755602 seconds and120 gradient batches;
peak run/replay RSS is272068KiB under64GiB. No new discovery, held-out metric
or parent selection occurs. The existing34/34/33 primary counts are unchanged;
the task246 comparison still requires the reviewed parent chain. Task251
tracks a separate prospective original-data adapter for the new complete-kernel
component; it cannot reuse the older one-direction discovery as its full basis.

The task251 native adapter now binds that complete-kernel core to all23 original
pair tasks and all55 supported primes with the original observation order and
minibatches. Seven fixture checks cover multidimensional/aliased codes, all
actual complex states and numerical replay, input/scope mutations, corrupted
bases/parameters after rehashing, all-equation refusals and complete process
failure records. Its summary retains all35-task metadata and separates new
discovery from head costs. No original run starts until implementation review
and a separate source/binary/request/protocol freeze; held-out comparison and
both105-choice parent arms remain a later gated task.

Task252's original complete-kernel training is now closed: all23 cases and1265
stages complete and pass their initial independent numerical replay. Of1045
complete kernels,653 have dimension0,337 dimension1 and55 dimension3;220 stages
retain all-equation-cap refusals. Fourier shape caps refuse988 initializations,
memory/audit forecasts refuse2, and six minibatch schedules exceed their work
cap. The55full/49mini/55forward endpoints all fit the observed raw labels with
zero errors. Six tasks admit prime97; Fibonacci admits49 full/forward primes
and43 mini primes. Discovery costs792.728345770s and head work1992.023800371s
with599 gradient batches; peak run/replay/summary RSS is550028KiB. No held-out
metric, parent selection or benchmark count changes. Authored findings and the
read-only report generator live in `convex_stress_kernel/cohort/original/`;
the frozen component/adapter sources and complete training evidence stay intact.

Task253 implements that separate comparison in `convex_stress_kernel_compare`.
It imports all complete-kernel stages and the future reviewed refined parent,
checks actual bases, complex heads and original data, and seals parent-first
raw-CE selections before any held-out metric. Eight native fixtures retain
both105-choice arms, all35 tasks, depth/transformer and every earlier cost.
Unknown axes, unmatched active codes and unobserved exact inputs remain distinct.
Primary99percent success is unchanged by auxiliary diagnostic failures.
Original comparison waits for reviewed252 training and246 parent evidence;
fixtures establish no new benchmark score or unknown-target identification.

Task256 / CM35 derives additional constraints from the existing unknown-code
readout compatibility premise. When the finite code and surjective declared
answer alphabets have equal cardinality, the unknown code factors through that
readout via a bijection. Complete kernels of the grouped original observations
can therefore certify answers refused by the ungrouped kernel. A strict
unseen-corner witness preserves distinct raw aliases; cardinality, unused-answer
and incompatibility counterexamples delimit the claim. The actual bounded
complex head retains original raw-label CE, full/minibatch and forward examples,
and a concrete wrong-raw/correct-pooled prediction. A new native proposal would
explicitly take the declared readout partition as construction side information;
it cannot silently alter the frozen252/253 protocols or claim to learn an
unknown decoder. Native implementation and original evaluation remain separate.

Task257 implements the CM35 proposal in `convex_stress_readout_kernel`.
Fifteen native checks retain the genuine grouped complete-basis certificate
separately from the expanded raw-label vectors. The original bounded complex
head, all raw CE labels, one full update, one minibatch epoch with its shorter
tail and zero-gradient forward mode remain unchanged. The strict unseen-corner
witness now matches its declared answer group while preserving raw ambiguity;
label/group/input transports, rehashed mutations, incompatible observed groups
and all resource/cardinality refusals are checked. This introduces declared
readout metadata explicitly, leaves unknown-target premises conditional and
does not change frozen252/253. An original-data adapter and later complete
comparison are required before claiming any new benchmark pass.

Task258 now binds the reviewed CM35 component to all23 original pair tasks and
all55 primes. The adapter explicitly records the original metric's complete
output partition as new construction information; its raw CE labels and
minibatches stay unchanged. Ten native checks cover genuine grouped bases,
raw expansions and actual complex replay, field-readout aliases, complete
cardinality/refusal/failure accounting, rehashed mutations and a disk-reserve
transition that continues after recording a refused case. Its source fingerprint
also pins the unchanged CM34 numeric dependency. Original execution requires
review and a separate freeze; full35 comparison and unknown-target hypotheses
remain separate from these fixture checks.

Task259's original CM35 training cohort has now closed under frozen producer
`21e86e7c`: all23 cases and1265 stages complete with initial independent replay
and zero process failures. There are19 complete grouped bases,12 observed-group
incompatibility refusals,7 admitted raw expansions,1242 cardinality refusals and
4 all-observation equation refusals. All seven admitted stages supply full,
mini and forward endpoints with zero observed raw errors, including a new
two-direction Fibonacci model. Its unknown-axis and unmatched-active masks
remain explicit. Discovery/head construction totals74.235849478seconds and
131 gradient batches; peak run/replay RSS372048KiB under64GiB. Fresh review has
reproduced every original case and the exact complete summary at6419ecd2.
The full parent/transformer comparison remains separate; training fit changes
no benchmark count. Only authored findings are committed; generated evidence
remains in the root's ignored `cohort/runs/training-v1/` directory.

Task260 adds the prospective CM35 comparison adapter. It binds the entire
declared output partition, genuine grouped basis, separate raw expansion and
original raw-CE state before offers. Both105-choice arms, all35 tasks and all
transformer/depth/parent evidence remain. Observed-CE selection is unchanged;
standalone candidate metrics and selected regressions are reported separately.
Ten native fixtures cover actual pooled-correct/raw-wrong heads, primary
success despite auxiliary failure, independently rehashed mutations and
disk-refusal continuation. Original execution requires reviewed255,259 and260.
New builds and run artifacts use the mounted `storage` filesystem under
`/run/media/misha/storage/transformer-runs`, with the64GiB RAM cap unchanged.

Task262 / CM36 proves the next selection boundary before changing a native
policy. Pulling grouped coefficients back through the recorded-name readout
preserves every original equation and both gauges. Every original full-kernel
match therefore transfers to the grouped system, and mapped-bank search cannot
lose an accepted query. This algebraic coverage statement needs no unseen
target premise; target correctness and native admission still do. Separately,
the actual refined complex heads retain equal observed logits and ordinary CE
through common full, mini and zero-update schedules even with different latent
spaces/defaults. A bounded three-corner witness has tied training loss and
opposite unseen pooled predictions. Frozen260 retains its parent-first tie
rule; any structural priority must be a separately frozen, fully compared
proposal. No native benchmark count follows from these exact-real proofs.

Task263 / CM37 removes the matching-gauge restriction from that coverage
result. The current native first-observation order already aligns the anchors;
the stronger result also covers independently normalized certificates.
Normalizing the actual pulled-back coefficient vector preserves every query
action and satisfies the raw anchors, regardless of the grouped anchors.
Complete matches and mapped-bank acceptances still transfer; with unchanged
labels and bank order, both gauge choices return exactly the same class or
refusal. Concrete examples change both anchors. No native selection rule,
unknown-target premise, original run or benchmark count changes.

Task264 / CM38 supplies the Lean-first fallback repair. Fixed support routes
actual candidate logits inside support and saved parent logits outside it.
Both complex coefficient blocks and their storage are charged, and flat
component codes supply the retained feature/logit bound. The affine domain
freezing the parent and ordinary normalized CE remain convex. Covered original
batches have exactly the candidate's objective and derivatives, with their
raw labels and original multiplicity. An actual bounded Fourier-head witness
still regresses on a supported unseen alias while preserving an unsupported
input. Native binding and full35 evaluation remain necessary; frozen original
protocols, selectors and benchmark counts do not change from this proof.

Task265 implements CM38 as a separate bounded complex fallback component.
It binds actual observed query/value support, candidate tensors and a saved
causal or count predecessor. The count parent is an explicit sparse complex
embedding of its fixed integer joint coefficients, with zero imaginary part;
its indicator features are bounded, without a unitary-code claim. Both stored
heads, logical coordinates, column radii and buffers are charged. Eight native
checks cover bit-identical count forwarding, supported/unsupported routes,
full/mini/zero modes, actual finite-difference CE derivatives with a shorter
tail, forged states and a supported unseen regression. The prospective protocol
retains the earlier head selection and tests a separately declared structural
fallback transformation on the complete35-task comparison. Because observed
CE is identical, this cannot be described as a strict-CE improvement. A reviewed
adapter and original complete evidence remain necessary; no benchmark count
or frozen245/246/252/255/261 source or selection rule changes here.

Task268 corrects the wrapper's long-prefix boundary before original integration.
The candidate's stored context defines its fixed feature library, while the
original causal forward accepts lengths through4096. A causal parent retains
that limit even with a shorter training library; a count parent retains its
saved count-value domain. The wrapper now uses the common prediction limit,
with actual supported/unsupported tests at6,65 and4096 tokens. Ten native checks
retain all prior CE, parameter, memory and mode guarantees. Protocolv1 evidence
stays preserved; v2 changes no original benchmark result because the fallback
has not yet been evaluated on the original suite.

## Recovered comparison queues (2026-11-13)

Task 282 formalizes CM42 before the next native model change, following the
user's priority of improving raw-label accuracy and ordinary cross-entropy.
Fixed stochastic context weights interpolate actual complex source features
with the unchanged coefficient budget. Exact observed rows preserve all
original CE batches and derivatives; local raw-label information remains an
explicit premise. The two-alias witness also preserves the nonidentifiability
counterexample. Native context metrics, any training-only neighborhood
selection, and held-out evaluation require a separate prospective protocol.
The reviewed task281 model and the older queued comparisons stay frozen.

Task 283 implements the reviewed CM42 fixed context map in
`experiments/convex_stress_context`. A declared ordinal input metric and a
canonical margin24 training-only diagnostic choose among the original default
and 1/3/5/9 nearest observed states. The query state is excluded during the
diagnostic; original record multiplicities remain. Selection is independent
of trained coefficients, so full/mini/forward modes share the same fixed map.
Actual complex feature products retain all head coefficients, original observed
rows and raw-CE derivatives. Native fixtures and mutation checks precede any
original-data evaluation; the 35/35 benchmark result is unchanged at this stage.

Task 284 prepares a separate prospective original comparison of the reviewed
context component. All35 tasks, both105-choice policies and all35 conventional
plus336 depth outcomes remain. Only stored selected raw/shared-extension pair
heads are transformed; task281 has three such Fibonacci endpoints. Selection
uses the reviewed canonical training-only rule before metrics. Actual tensors,
original data fingerprints, raw CE and accuracy must replay independently;
all unchanged choices remain explicitly inherited. Source and executable are
frozen before execution, with a64GiB process limit and storage-backed outputs.

The original task284 run completes and replays all35 cases without process
failure. Fibonacci raw accuracy improves from66.3265% to97.2364% in all three
modes, while ordinary validation CE falls from8.08163266 to0.27997472.
Primary field accuracy stays99.9362%, and both policies retain35/35 primary
and auxiliary success in every mode. Three distinct changed heads supply six
policy choices;102 choices per policy remain explicitly inherited. All35
transformer and336 depth endpoints and prior costs remain. Canonical training
CE chooses five neighbors for both heads, independently of trained parameters;
no extra optimizer updates occur. Peak original RSS is1,662,128KiB under64GiB.
Authored findings live in`convex_stress_context_compare/original`; exact-commit
review at `d173da9f` and a fresh end-to-end reproduction pass, including identical
model tensors, selections, metrics, summaries, reports and329 original hashes.
Task284 is approved.

Task285 proves CM43 before any confidence experiment: a frozen per-input scale
in[0,1] preserves the actual feature/logit bound and positive scaling preserves
raw-class ordering. Ordinary CE remains convex in the complete coefficient
tensor. Scalar CE convexity at fixed logits is a separate statement; no joint
convexity in scale and tensor or monotone CE improvement is claimed. Scale one
on every original observed record preserves exact batch objectives and full
derivatives, including repetitions, minibatch tails and forward-only states.
Any native context/scale bank and training-only selector require a separate
prospective protocol; pooled task metrics must still be measured afresh.

Task286 implements the reviewed CM43 theorem in a separate native component,
`convex_stress_context_scale`. It retains the reviewed CM42 neighborhoods and
selects a positive confidence scale from1/0.75/0.5/0.375/0.25/0.125 using the
same canonical training-only raw-label diagnostic. Only previously interpolated
unseen rows are scaled; observed and other unchanged rows stay exact. Complete
source tensors and budgets remain, with no extra optimizer updates. All original
held-out metrics require a separately frozen comparison; no new result is assumed.

Task287 freezes the separate original comparison of reviewed286 confidence
calibration against reviewed284. It imports every stored selected CM42 context
head, validates the original record IDs and actual complete coefficient tensors,
and seals training-only scale selection before fresh metrics. Both105-choice
policies retain all35 tasks and all35 conventional plus336 depth endpoints.
Other choices remain explicitly inherited. Every changed head must reproduce
its model bytes and complete metrics; original success, raw accuracy and CE
are reported together, with all costs and regressions retained under64GiB.

The task287 original run completes all35 cases and numerical replays with
zero process failures. Training-only selection chooses scale0.5 in every
prior/supplement head. Fibonacci validation CE improves0.27997472 to0.16604154;
raw accuracy97.2364% and primary field accuracy99.9362% are unchanged in all
three modes. Both policies retain35/35 primary and auxiliary success, with
3fresh/102inherited choices each. Peak RSS780,884KiB remains below64GiB.
The retained transformer has95.1318% raw accuracy and CE0.23454365 on this
endpoint. Authored findings live in`convex_stress_context_scale_compare/original`;
the exact-commit fresh reproduction and review follow before completion.

Task287 is now approved at `c5ba15c6`: a fresh all35-case reproduction and
every numerical replay match model, metric, summary and report bytes, and
all329 original artifact hashes verify. Task288 extends Lean first for the
remaining high-CE declared categorical heads. CM44 realizes their fixed state
map and raw decoder with an actual conjugate complex tensor and a common
64-times-state-count budget for margins through24. Its exact ordinary batch
CE separates the correct-margin loss from margin times raw error, retaining
repeated record IDs and short minibatch tails. Native confidence changes and
original held-out evaluation remain separate prospective work.

CM44 task 288 is approved at `a8f0020a`, including a repeated full build,
axiom audit and generated INDEX check. Task 289 implements the actual complete
complex confidence head for frozen categorical decoders. It selects a positive
margin using only original training records, preserves their multiplicities
and minibatch tails, and explicitly adds zero optimizer updates in every mode.
Original-data metrics will follow only after the native component and its
prospective comparison protocol are reviewed and committed.

Task 289 is approved at `9ddda0f8`: six native tests, complete complex
coefficient gradients, source/mutation checks and repeated repository checks
pass. Task 290 compares its confidence heads with the reviewed task 287 parent
on all 35 original cases. Eligibility uses the saved identified, consequence
or fiber model representation. Exact source tensors and original source metrics
must reproduce their reviewed hashes and results. All other representations,
both policies, all modes and every baseline remain explicit; no held-out gain
determines which transformed heads are reported.

Task 290 producer `d5ec25b6` completes all 35 original cases and 35 replays.
The 35 transformed heads span 13 tasks: 13 full-batch, nine minibatch and 13
forward-only endpoints. Their raw accuracy remains 100%, while CE falls from
3.59227016 or 4.27155381 to below 8e-9. Every primary and auxiliary success
remains 35/35 in both policies and all modes. Source import peaks at 8.10 GiB
under the 64 GiB cap. All prior choices and baselines remain; Fibonacci retains
97.2364% raw accuracy and CE 0.16604, leaving its 130 raw errors unresolved.
Authored findings are in `convex_stress_decoder_compare/original`. The separate
complete reproduction and exact-commit review follow before task completion.

Task 290 is approved at `347f29e1`: a separate complete preparation, all 35
cases and all replays reproduce every JSON artifact and compact report.
Task 291 extends Lean first for the remaining raw-label errors. CM45 mixes
fixed context feature maps on a simplex with the complete head frozen. It
retains the complex feature/head bound and gives convex ordinary diagnostic
CE, explicit mean slopes and a quadratic upper model. Exact observed rows
preserve original repeated and tail batches, so that unchanged objective cannot
choose the mixture. The training-only diagnostic is a separate declared rule.
Nonzero Fourier witnesses retain observed fit while choosing opposite unseen
raw aliases. Native mixture optimization and original evaluation follow only
after the proof and its review.

Task 291 is approved at `2f0dc9ba` after the full build, axiom/source audit,
native generated INDEX and exact-commit review. Task 292 implements the fixed
91-map context bank in Rust. Ordinary source-excluded diagnostic CE uses every
original raw record and the full vocabulary denominator. Full/minibatch modes
optimize only simplex weights, with explicit repeated/tail batches; forward
selects a vertex with no derivatives. Actual inference uses the unchanged
complete complex source tensor. Allocation/work forecasts and reconstruction
bind all added state. The original 35-task comparison follows component review;
no CM45 original-data improvement is claimed by the component fixtures.

Task 292 is approved at `14b8ad6e`. Its synthetic fixture exposes a CE/raw-label
tradeoff, so task 293 reports both quantities separately from original success.
The prospective comparison pins task290's complete choices and task287's actual
context model bytes. Eligibility is by current representation/reference/hash,
not task name or held-out scores. It retains all35 tasks, both policies, all
baselines and prior costs. Original records and batch sizes are checked; every
mixture is sealed before metrics, old metrics are reproduced, and all new
gradients/updates are counted. Complete original execution and independent
reproduction follow the committed adapter/protocol.

Task293 producer `714d6d69` completes all35 original cases and replays with
zero failures. Both policies retain35/35 primary/auxiliary successes in every
mode. Fibonacci raw errors fall130 to109 full,103 mini and100 forward; CE falls
0.16604 to0.14011,0.14196 and0.14904 respectively. Field accuracy stays4701/4704
and observed CE stays9.62659e-9. Full has the lowest CE; forward has the best raw
count, with no new gradients. The six prior/supplement components execute288
gradient batches and172 nonzero updates in total; all work and baselines remain.
Peak observed replay RSS is1.99GiB under64GiB. Authored findings are in
`convex_stress_context_mixture_compare/original`; complete independent
reproduction and exact-commit review follow before approval.

Task 293 is approved at `f0a19727`: a separate complete run reproduces all
115 JSON artifacts and the compact report exactly. Task 294 extends Lean
first with CM46 fixed gates and anchored local context mixtures. The local
simplex rows retain the same source/head bound, with affine actual logits,
convex ordinary CE and explicit local slope and change bounds. Original
record and head-derivative identities remain exact. A nonzero Fourier example
shows two opposite unseen raw margins available to local mixtures but
impossible for a shared mixture on the same bank. This is a capacity result;
training-only diagnostics and actual held-out gains still require a separate
native implementation and complete comparison after proof review.

Task 294 is approved at `c60816fa`: 33 new theorems, full Lean build, unchanged
36 sorry/zero extra axioms, generated INDEX and exact-commit source review.
Task 295 implements fixed latent-block gates and radius-one-half anchoring
to the reviewed CM45 model in Rust. Its original-record diagnostic keeps the
full vocabulary and original minibatch tails. Local initialization uses only
forward CE; iterative modes add local conditional gradients, leaving absent
gates unchanged. Actual predictions retain the unchanged complete complex
head. The synthetic fixture lowers CE but loses one raw prediction in mini
and forward, so the separate complete original comparison must retain both
metrics and every regression after component review.

Task 295 is approved at `a8deaa63`, with eight native tests and exact-commit
checks. Task 296's prospective adapter pins the complete task293 parent and
every current CM45 model byte. Eligibility uses only representation and hash;
original modes, references, records and minibatches must match. Every local
mixture is sealed before held-out scoring. All 35 tasks, both policies, original
success thresholds, raw CE/accuracy, unchanged baselines and inherited costs
remain. The original execution and independent complete reproduction follow
the committed adapter and frozen protocol before any original CM46 gain claim.

Task296 producer `1b42f411` completes all35 cases and35replays with zero
numerical process failures. Both policies retain35/35primary/auxiliary successes
in every mode. Full raw errors fall109to101; mini103to100; forward stays100.
Validation CE regresses in all modes to0.17261/0.17511/0.17728, despite lower
source-excluded diagnostics; the earlier CM45 mixture remains better on CE.
Observed loss and field accuracy stay unchanged. The six components record288
gradient batches,182updates and7887local-row updates; forward has zero gradients.
Peak replay RSS is3.10GiB under64GiB. Authored findings retain every tradeoff,
all baseline/cost evidence and the interrupted pre-evaluation launcher. Complete
independent reproduction and exact-commit review follow before approval.

Task296 is approved at `96aee0fc`: fresh complete preparation,35cases and
35replays reproduce115JSON artifacts and the report exactly. Task297 tests
stronger coefficient sharing under the existing generic CM46 proof. Five
fixed class-count profiles replace separate source-block gates; count profiles
exclude the query's unique observed source state and ignore raw-label numeric
names. The CM45 anchor,91-map bank,radius0.5,head and ordinary CE optimizer stay
fixed. Ten native tests cover boundaries,exclusion/repeats,renaming,shared actual
complex blocks and original modes/gradients. A separate full original comparison
must retain CM46's negative CE result and all baselines after component review.

Task297 is approved at `5e89c616`. Task298 freezes a complete native comparison
of its five shared gates against reviewed CM45. Its separate pinned task296
reference retains every previous choice, raw metric, CE regression and local
cost, without participating in selection. The source bank/head/anchor and all
35 tasks, original modes/batches, success rules and transformer/depth baselines
remain. Full original evaluation and independent reproduction follow the
adapter commit and executable freeze. The earlier original results motivated
this model change, so they are not independent architecture-selection holdouts.

Task 255 completes the original raw-kernel comparison with 35/35 full-batch,
35/35 minibatch and 34/35 forward-only primary successes in both policies.
All 35 cases replayed with zero process failures. Only three choices per
policy change, all Fibonacci at prime97, reaching 99.8724% primary field
accuracy. Its native-token accuracy is 66.2840% and validation CE rises to
8.0918; forward associative divisor remains at 53.866%. The full159 evaluated
endpoints,111 eligible offers,3636 refused mode outcomes and all constructor
costs remain. Fresh exact-commit review is required before queued task 261.

Task 281 completes the original seed-1 comparison with 35/35 primary and
auxiliary successes in full batch, minibatch and forward only, for both
initializer policies. All 35 cases replayed with zero process failures.
Each policy has 19 changed choices and four newly successful task/mode choices;
35 distinct fresh endpoints and the inherited reviewed metrics remain explicit.
Fibonacci reaches 99.936% and forward associative divisor 100%. Fibonacci raw
validation CE rises to 8.0816, a retained diagnostic regression. Actual calls
never reach an unavailable parent: the seven old failures were preflight/domain
validation failures. Available-parent fallback handles 2,148 associative-divisor
calls per policy. The findings commit requires fresh exact-commit review.

Task 277 completed its original attempt: 28/35 primary passes in each mode and
both initializer policies, with seven saved-parent domain process failures.
The successful fresh states include Fibonacci at 99.936% in all modes and
forward-only associative divisor at 100%. All completed cases replayed; the
findings commit requires fresh review. Failed cases keep their missing scores,
logs and partial artifacts. Task 280's proved route correction is a separate
prospective comparison, with no substituted parent successes in task 277.

Task 280 reuses the reviewed 276 comparison source with the reviewed CM41
partial-parent wrapper. The original selector and same-field shared-role gate
remain fixed; only causal parent availability changes the route. Actual masks
separate support, parent availability and the source used. A complete runner
fixture reproduces the old short-parent refusal, checks long prediction and
saved-state replay, and keeps construction independent of held-out labels.
All original sources, both initializer arms and baseline scope remain. A new
original run follows review; task 277 retains its original domain failures.

Task 279 implements the reviewed CM41 route around the original checked saved
heads. It preserves the candidate through length 4096 when a count predecessor
is unavailable, without expanding that parent's finite feature alphabet.
Support, availability and actual routing are distinct diagnostics. Both heads,
original record multiplicity and batch derivatives retain their old bindings
and bounds. Task 280 will integrate the reviewed component into a separate
complete comparison; the frozen task 277 failures remain unchanged.

Task 278 extends the Lean fallback theory to saved parents with restricted
input domains after three early task 277 cases refused original long inputs.
The fixed gate keeps the candidate on supported inputs and outside the parent's
domain, using the parent only on unsupported inputs where it is available.
Actual zero-extended parent coordinates retain both budgets, convex ordinary
CE and supported original-batch objective and derivative identities. The
frozen run and its failures remain unchanged; native implementation and a new
complete comparison follow only after the proof is reviewed.

Task 276 builds a separate full comparison from reviewed 246, 252, 237 and 274
evidence. It keeps the unchanged raw-CE selector, applies the reviewed CM39/40
coverage gate to selected raw kernels and the CM38 predecessor fallback to
selected causal heads. Choices and actual tensors are sealed before scoring.
Changed models receive fresh original metrics; unchanged models retain explicit
approved-parent provenance. Both 105-choice arms, all 35 primary results and
transformer/depth baselines remain, with constructors charged once and failed
cases retained. The frozen 255/261/267 queues are unchanged. Original execution
requires a reviewed adapter and a separate complete findings/replay task.

Task 275 applies the proved CM39 conservative gate to the actual CM40 shared-role
certificate. Both heads and certificates remain checked and charged; explicit
original IDs retain order and repeated-record CE weights. The supplement acts
only on newly supported inputs, preserving prior logits and all original
training derivatives. Original histories and all 35 benchmark results require
a separate comparison adapter after the reviewed training cohort. Existing
frozen comparisons and their selectors remain unchanged.

Task 274 completed all 23 original CM40 training cases and all 1265 prime stages
on the storage disk. Five tasks have endpoints in every mode, with zero observed
raw-label errors; fourteen restricted bases refuse incompatible answer groups.
The Fibonacci construction has dimension one and zero unmatched active pairs,
while its unknown-axis count stays unchanged. This is training and support
evidence, with no held-out success count. Authored findings and native report
commands live in `experiments/convex_stress_shared_roles/cohort/original`; fresh
review replays every completed original case from the frozen runner.

Task 273 prepares the separate CM40 original training cohort after the reviewed
272 component. It retains the same 23 cases, all 55 primes and all 35 suite
metadata. Original record IDs, order and repetitions are saved separately from
the distinct algebraic equations, with full raw CE and the original batch size.
Replay checks the complete augmented matrix, raw expansion, complex states,
every update and all recorded refusals. Original execution needs the reviewed
adapter and a separate findings/replay task. It does not evaluate held-out
labels or change the frozen comparison selectors and baseline counts.

Task 272 ports the reviewed CM40 restriction into a separate native component.
Its certificate contains the actual augmented observed matrix, all role rows
and both gauges, with independently checked complete basis and grouped-code
separation. Actual Fourier heads retain raw CE; separately bound original IDs
preserve repetitions and short minibatch tails. Nine checks include exhaustive
small systems, nonzero common offsets, basis/gauge changes, raw aliases, the
characteristic-two limit, actual complex gradients and mode replay, long
prefixes, certificate mutations and resource refusals. The combined head,
certificate and record forecasts must fit 56 GiB under a 64 GiB process cap.
These are component results; original training and complete comparison still
require separate frozen adapters. Existing 255/261/267/270 evidence is unchanged.

Task 271 / CM40 adds actual shared-role equations to the complete observed
kernel. Aligned categorical identities have one free row-minus-column offset,
which preserves both original gauges. Full-basis matches identify codes and
declared many-to-one answers under explicit global additive, shared-role and
readout premises. Actual matrix witnesses show strict extra coverage over
the field of three elements, remaining raw aliases, an independent-role
counterexample to inferring the new premise, and refusal in characteristic
two. A reproducible training-only diagnostic on task 259's saved prime-97
grouped code finds 97 observed-separating projective directions, of which
exactly one has a common offset across all 191 aligned input tokens. No
held-out query is evaluated. Native restricted-kernel construction, model
binding and a complete original comparison are separate later steps. The
existing frozen comparisons and task 270 component remain unchanged.

Task269 / CM39 proves a prospective conservative coverage extension. A
supplemental complex head acts only where the prior refuses and the supplement
accepts. Every other input retains exact prior logits, even if the two heads
disagree. Prior-covered original batches retain their normalized raw CE and
Fréchet derivatives; both coefficient blocks remain charged in the CM38
convex domain. A flat Fourier witness disagrees on three unseen inputs but
changes only the newly covered one. It asserts no unknown-target correctness
or benchmark gain. Native binding and a separately declared full comparison
remain necessary; frozen255/261 selectors and267 fallback are unchanged.

Task270 implements that gate for actual CM34 raw and CM35 grouped complex
heads over the same original observations and field. The separate component
checks complete codes, observed dictionaries, numerical complex audits, both
stored budgets and bound record multiplicity. It preserves prior logits on
prior support and joint refusal, and supplies the unchanged raw-CE derivative
in prior coordinates. Six native checks include a strict new prediction,
full/mini/forward states, nonzero finite differences with repeated records,
short minibatch tails, long prefixes, mutations and resource refusals. Its
prospective transformed arm requires a same-prime/same-mode grouped endpoint,
positive added coverage and unchanged training CE; a separate reviewed adapter
and complete original evaluation after267 are still required. No original
selector, frozen evidence or benchmark count changes from these fixtures.

Task266 supplies the complete CM38 adapter after the268 long-domain correction.
It preserves both task261 initializer arms, then declares separate transformed
arms using each selected causal head's exact immutable predecessor. Actual
causal histories, original record order/multiplicity and raw CE are replayed;
every supported input keeps exact candidate logits. Native tests retain all35
tasks/105choices, both policies, primary success despite auxiliary failure,
long prefixes, original Counts bytes, absolute/relative checkpoint bindings,
corruption, refused imports and failed processes. Both heads and the retained
history receive memory forecasts; all original costs and per-call support and
exact-input masks remain. Task267 waits for reviewed261 and266, then freezes
source, executable, request and protocol before any original fallback metric.
No benchmark pass or raw-CE selector change follows from these native fixtures.

Task266 review also exposed a resource-policy mismatch before original metrics:
the combined candidate/Count-parent forecast can exceed the inherited2B bare
causal allowance on already saved inputs. Protocolv2 declares a uniform32B
evaluation allowance, matching the refined comparison, with unchanged64GiB RAM,
models and selection. A ninth test covers both forecast boundaries. The passed
v1 checks and frozen source/binary remain preserved; original267 uses the fresh
reviewed revision after all parent gates close.

Task246 has completed all35 original refined-comparison cases and their first
independent replays without process failures. Its18 available prime97 endpoints
all pass original primary and auxiliary metrics, but none improves the selected
parent's raw CE by the declared tolerance. Both105-choice initializer policies
remain identical to245:34/35 full,34/35 mini and33/35 forward. Fibonacci remains
unsuccessful in all modes and associative divisor remains the second forward
failure. Construction185.750755602s and120 gradient batches are retained once;
peak process RSS2361828KiB stays below64GiB. The separate authored findings and
read-only report preserve the frozen adapter; fresh findings-commit review
must finish before255 imports this complete parent.

Task #221 preserves a completed parent-chain failure caused solely by nine
relative/absolute phase model-path strings. Committed native recovery stages
canonical views of the same actual phase, paired and certified artifacts;
every other summary field remains exact. All eight decoded cases then replay
with byte-identical models/proposals and unchanged scores under the frozen
producer. The complete decoded comparison keeps15/24 passes. The general-bank
integration now passes21/27 choices,7/9 per mode, with the entire105-choice
hull selection independently replayed and retained. Sum-cube and sum-power-four
still fail. All original failed gates remain. The unchanged35-task transformer
cohort is running serially; final full35/depth/fiber integration and transformer
comparison remain open.

Task #217 now has a native full105 integration harness for the completed
general comparison and all density/balanced/tagged/fiber candidates. It retains
every parent and all35 choices by mode, uses earlier-first observed-CE ties,
checks the original split/vocabulary/scale bindings and charges each forward
construction once. Synthetic metadata tests cover failures, absent models,
held-out invariance, both raw-audit vocabulary schemas and22 mutations. The
completed78fb450c integration now passes26/35 all-split cases in full/mini and
25/35 forward-only; primary counts are27/35,27/35,26/35. All nine recently
extended pair tasks pass. The other failures, three absent forward models and
the initial schema gate remain recorded. Depth, composite discovery and the
fresh transformer remain explicit follow-ups.

## Composite categorical codes (2026-11-13)

Task #222 / CM27 extends the theory beyond prime latent domains before native
discovery. Thirteen Lean theorems give the actual divided-residual identity,
an exact digit-lifting equivalence, checked Bezout recombination and unit gauge
normalization of observed class codes. Closed examples use composite moduli;
counterexamples show that a base modular solution may have no lift and that
noncoprime component agreement need not extend to the product modulus. These
claims certify algebraic steps, not solvability, unique discovery or unseen
labels. A future observed-only native code learner must retain all component
systems, normalization choices, lifts, recombination witnesses, failures and
caps; verify original equations, code injectivity and actual spectral/CE
bounds; then freeze separately before original discovery. All prior candidates,
full35/modes/transformer scope and the64GiB limit remain required.

The native integer core now passes five checks, with97 exhaustive scalar lift
systems and over2000 CRT residue pairs. Independent i128 audits check the i64
producer's original equations, exact residual identities, all field operations,
actual representatives and Bezout witnesses. Nonunique digit spaces, failed
lifts and caps remain explicit. Seven further native checks now exercise the
observed-only categorical learner: masked domains 2 through16, repeated phases,
class renaming, unused vocabulary, every original equation, caps and certificate
tampering. It requires one-dimensional normalized component kernels and unique
digit lifts, checks CRT and class injectivity, then audits actual raw spectra
and ordinary CE. A missing-axis example explicitly fits observations while
misidentifying unseen answers. The complete suite has twelve checks. The
categorical strip wrapper adds three checks for composite interiors, unequal
axis padding, class renaming, caps and changed state. All fifteen native checks
pass. The parent harness now adds five checks, covers all23 original eligible pair-table
tasks and preserves the complete105-choice parent. The protocol now pins
task217's checked78fb450c comparison, digest a2a3c4af874defb15b0f27af1b5115079b08738ec4b2efacefd7f0b79ac0096a.
Producer f661532e now completed and replayed all23 cases with zero process
failures. Sixteen new candidates have100% accuracy on every original split;
seven retain explicit refusals. Full105 selection passes26/35 full,30/35 mini
and25/35 forward. Four power-division cases improve minibatch, while earlier
equal-CE failed completions remain selected in full/forward. All parents and
failures remain; construction cost574.527s is charged once, with69 references
and zero CE gradients. Peak recorded run/replay RSS is2,501,016KiB, under64GiB.
See the modular README for the immutable summary digest and full findings.

Task #223 / CM28 now has17 checked Lean theorems at4fee3d09. A cyclic full
joint observed module identifies a table with an unknown bijective class code
over a composite domain, permitting repeated axes. Seventeen observations
recover a49-cell six-state witness with32 held-out cells. A missing row leaves
an independent kernel direction and forces refusal. This is identification
inside the stated categorical model, not across every exact fit in the larger
convex hull. Checked native elimination/component certificates and a separate
structural selection freeze remain next; do not overwrite task222's CE ties.

CM28 now has16 further theorems at9cd95708: replayed actual row operations
certify the original anchored matrix, imply joint cyclicity and normalized
vector uniqueness, and give kernel uniqueness through modular digits and CRT.
The first native certificate core in `convex_stress_identified` exhaustively
checks2274 small matrices against their finite kernels, plus the exact Lean
nonunit-pivot example, larger domains, work caps, state/control mutations and
invalid inputs. It uses independent i128 row replay and retains every refusal.
Categorical/model binding, explicit strip-model scope and a separate original
selection freeze remain next. No new original score is claimed by this core.

The categorical certificate now binds the actual observed matrix, all original
code equations and shifted joint vector without missing-axis pins. Five further
checks certify masked domains2–16 and the exact six-state Lean witness, preserve
class renaming, and reject missing-axis ambiguity, collisions, omitted-equation
conflicts and changed certificates. A realizable noncyclic joint-module example
remains an explicit limitation. All ten native checks pass. Four additional
Lean theorems at6a8fb49e state the fixed-view/constant-strip scope, bringing CM28
to37 new checked theorems. The c1a6720e adapter binds actual parent models,
phase/code points, every full proposed prediction and raw spectral/CE audits;
its five tests retain missing-axis refusals and verify held-out score changes
cannot affect selection. The declared rule prefers smaller ordinary CE, then
a verified fixed-view certificate within a1e-12 tie, then earlier order.
The separate original harness adds six tests, pins the entire completed task222
comparison and preserves all23 eligible tasks and all105 mode choices. All21
native checks pass. Frozen producer a2ae1a35 now completed all23 original cases
and full replays with zero process failures. Sixteen fixed-view certificates
preserve100% original split metrics. The declared CE/certificate rule repairs
four power-division ties in full/forward, giving30/35 full,30/35 mini,29/35
forward all-split passes across all105 choices. Task222 remains immutable.
Certification/checking takes553.734s once,69 mode references,zero new CE
gradients; recorded peak prepare/run/replay RSS4,256,420KiB. Summary digest
5c13439e55af498340977c6b27c2492ccc840d4bfb9940568e7ad73fe4cf311d.
Fresh review passed all 23 case replays, all 21 native tests and the complete
byte-identical summary at 0cddb722; task223 is done. Retained depth integration
and the fresh transformer are pending.
Full pass has not been achieved; no held-out scores changed the frozen rule.

Task #224 / CM29 extends the theory independently of the frozen task223 run.
Fourteen Lean theorems now certify decoded answers from actual observed row
consequences, without cyclicity or unique normalized class codes. The finite
certifier tests every replayed coefficient and returns a passing proposed
class or refusal. Unknown injective-code additive realizability stays explicit;
neither the code nor unknown target is an input to the certifier. A four-state
composite witness has all classes observed and a genuinely held-out cell.
Two row additions identify that answer, and every cell is recovered, while
the actual gauged joint module is provably noncyclic. This removes an overly
strong sufficient-condition restriction; it does not certify arbitrary decoder
fibers or make the bounded code search complete. Native row-consequence checking
and bounded discovery through nonunique code spaces remain future work with
their own original freeze. All existing original runs and model boundaries stay.

Task #225 begins with bounded native enumeration of actual checked affine
spaces in `convex_stress_consequences`. The producer retains all emitted
parameter tuples, points, exact work and cap/cardinality outcomes. Independent
i128 replay checks every point against every original equation. Five tests
pass, including all793 two-variable affine systems modulo2 and3, nonzero
particular solutions, several free directions, every cap, state mutations and
a257-to-the-eighth space whose bounded prefix is not claimed complete.
Composite branch propagation and CM29 decoded query certificates are still
needed before any separately frozen original experiment. Task223 stays frozen.

The native digit layer now lifts every emitted affine digit through CM27's
exact residual identity, including nonunique banks, and independently checks
all original next-modulus equations with i128. Four additional tests cover
1,456 direct exhaustive comparisons, failed lifts, actual branch order, caps
and mutations. All nine native checks pass. Complete component/CRT branching,
decoded query certification and original model integration are still pending.

The native component tree now retains bounded nonunique branches across every
prime-power level. It rotates among normalized base codes and explores each
root in depth, preventing shallow branching from exhausting all 16 default
stages before a valid high-power endpoint appears. Independent replay verifies
actual normalization, original equations, class multiplicity filters, queue
order, every refusal/cap and exact work. Five more tests (14 total) cover
noncyclic witnesses through modulus 64, identical decoding from distinct codes,
missing-row ambiguity, all caps and malformed reports. The component forecast
must fit 32 GiB inside the 64 GiB process limit. CRT branching and actual CM29
query certificates remain required; these fixtures change no original score.

CRT now combines a bounded mixed-radix prefix of emitted component banks with
actual checked Bezout witnesses and final original equations. Every tuple,
collision and cap is retained; balanced component marginals alone cannot admit
a colliding joint code. Independent i128 replay verifies the original systems,
component residues and declared prefix/work. Five added tests (19 total)
cover nonunique codes across multiple prime factors, missing-row ambiguity,
joint collisions and all caps/mutations. The default permits 64 tuples and
500 million CRT/endpoint row products; the retained-bank forecast must fit
48 GiB. CM29 query certification and original model integration remain pending.

The native CM29 certifier now constructs a shared actual row program from the
retained observations and a checked suffix for every proposed decoded query.
Independent i128 replay verifies the original matrix and exact query
coefficients; full replay additionally binds search decisions and work. The
constructor supports divisor pivots using unit normalization and actual
integer divisibility, with explicit refusals rather than a completeness claim.
Nine new tests bring the suite to 28: all 2,274 small matrix spans, the noncyclic
Lean witness, masked domains through 32, all 432 binary realizable query cases,
missing-axis/wrong-label refusals, program mutations and caps. The certificate
forecast must fit 16 GiB. Unknown injective-code realizability over the declared
observed class set remains required. Original model/raw-spectrum and fixed-view
binding are next; the original score and all frozen parents remain unchanged.

Four further CM29 Lean theorems now transport decoded certificates through a
fixed categorical view. An explicit class-name map sends interior classes to
the full output type, allowing a strip constant absent from the interior.
Soundness holds for every returned query; finite interior certificate coverage
gives full-view recovery. The noncyclic witness has a fifth strip input name
and a fifth output label, plus the certified held-out interior answer. The
unknown target must have the declared view/strip structure; no interior
certificate infers it. This matches the native tagged decoder's output types
before the new model integration.

The native learner now joins observed code discovery, actual complex phase
models, raw spectral/ordinary-CE audits and decoded certificates. It rechecks
all original observations after bounded-prefix discovery, retains conflicts,
duplicates/caps and numerical refusals, and binds every certified query to
the actual model table. Selection is observed CE, certificate on a 1e-12 tie,
then earlier order. Eight added tests bring the total to 36. Half-observed
96- and 192-state fixtures recover and certify all 9,216 and 36,864 cells,
respectively, under unchanged default budgets. Unit-column cleanup cuts the
96-state query arithmetic from 349.84 million to 15.58 million products.
The eight learner checks/replays take 190.89s with peak RSS 3,310,156 KiB;
the other 28 tests, fmt/clippy and all Lean audits pass. There are no CE gradient
steps in these forward proposals. Fixed-view native binding and the separate
original cohort freeze remain required; all previous scores/parents stay fixed.

The native fixed-view wrapper now applies CM29 per-query soundness to the actual
full table, with a separate output class map and explicit strip assumption.
Every mapped answer retains its interior query certificate; unmapped answers
retain their assumed constant. Unused padding does not require a certificate.
Five more tests (41 total) check raw spectra/CE, missing axes, outside strip
labels, duplicate tables, caps/refusals and mutated evidence. A deliberately
ambiguous unseen row demonstrates why this is not unconditional identification.
One dense report per view permits serial disk-backed cohort processing under
the 60 GiB view forecast and 64 GiB process limit. The separate original harness
and freeze remain next; no previous score or parent changes.

The separate task225 original harness now has eight passing native tests (49
with the core). It pins the complete task223 summary, preserves all23 eligible
pair inputs and all105 choices, serializes one dense view at a time, and seals
observed-CE/certificate selection before original held-out metrics. Full replay
checks all native programs, raw models and metrics; summaries bind every dense
artifact by size and SHA256. Existing certified parent choices retain priority
on ties; all process failures retain their prior rows. The tested source/binary/
protocol must be archived before original preparation. No new score yet.

Producer `70cf9ab5` is now committed and archived with the binary, protocol,
source fingerprint and complete task223 parent hash. All23 original constructions
and replays completed under64GiB with zero process failures. Seventeen cases
admit certified100%-accurate models; square-times adds one all-split pass in each
mode. The complete105 comparison is31/35 full,31/35 mini,30/35 forward, preserving
all previous parents and costs. Constructor time is788.9996s with zero CE
gradients; peak model run/replay RSS4.68GiB and complete-summary RSS13.45GiB.
Summary SHA256 is89a14738d36461e42db2faa2e9611b331ea6ea1fad01da5f16cd2dc34e2a0b63.
Independent review at e32f4489 passes all49 native checks, all23 original
constructor replays and a byte-identical complete105 summary; task225 is
approved. Depth and transformer integration remains required.

The independent task175 depth review is complete at `d1c4f860`: all31 native
checks and Lean audits pass, and all288 original plus48 retained outcomes and
both summaries replay byte-identically. Task175 is approved. Task226 records
the remaining full105 integration after task225, preserving both retained
initializer policies, all prior costs/failures and the pending conventional
transformer comparison. The depth successes are not yet added to the score.

Task227 prepares `convex_stress_paths_compare` with synthetic evidence only.
Its eight native checks preserve105 prior choices, all336 depth outcomes and35
transformer endpoints, including failures/refusals. Two separate initializer
arms choose only by observed CE and earlier order; complete parent snapshots,
actual model/cache/checkpoint hashes and numerical replay gates are required.
This is a saved-endpoint comparison with zero new training or prediction
evaluation. Task226 may evaluate the originals only after175/191/225/227 close.

Task228 / CM30 extends the observed-row proof to noninjective class codes.
Actual row replay identifies the target code; a declared readout constant on
code fibers therefore has a certified quotient answer. The same finite bank
recovers every quotient answer under checked coverage. A nonconstant3-by3
witness observes all four raw names and certifies its held-out quotient while
two fitting targets disagree on the raw answer. Ordinary raw-label CE and
original metrics remain unchanged. This is a theory boundary for a future
native alias-aware producer, not an additional benchmark pass.

Task229 starts that native component in `convex_stress_quotient`. Its fixed
prime catalogue is independent of data; each stage compacts observed categorical
names, solves original observed equations with two shift gauges and retains
colliding class codes. Independent signed128-bit checks replay every original
equation and preserve caps, ambiguity and conflicts; unused axes stay unknown.
Five synthetic checks pass, including a seven-state/14-raw-label hidden-alias
table, categorical renaming and seven altered reports. Actual complex model,
bounded ordinary-CE readout, CM30 query certificates and a separate original
freeze remain necessary before any new benchmark result.

## Complete categorical fiber constraints (2026-11-13)

Task #220 / CM26 starts from actual same-label observation groups. Lean proves
that injective shifted phases and a matching actual decoder-fiber cardinality
make such a group complete, then derives its row/column phase-sum equation.
Seven theorems include a counterexample to dropping completeness; closed
witnesses include nonconstant axes and a nonconstant many-to-one decoder.
A common fiber mass is an explicit additional hypothesis. Observed counts
alone do not justify it. A future native finite-field proposal must retain
every guessed multiplicity, equation, nullspace choice and cap, independently
check actual endpoints and preserve all earlier candidates. Freeze it only
after its own tests; no new original result follows from this theory. The
bounded native finite-system core now passes four checks, including exact
solution-set agreement with all793 binary/ternary two-variable systems.
Independent full-row replay verifies the retained invertible operations,
original equations, affine basis and explicit work/nullity caps. Six additional
categorical checks now pass: observed-only equations and bounded affine
enumeration recover masked pair/triple/four-fiber fixtures with permuted input
phases and renamed output classes. Independent actual-fiber/coverage/injectivity
and raw spectral/CE checks reject invalid candidates. A separate exact-fit
fixture retains two different unseen completions, with no identification claim.
The parent-preserving original harness now passes four checks, retaining every
tagged-family candidate and earlier-first CE tie, with all27 mode references
and rehashed tampering checks across balanced/tagged/fiber reports. Frozen
producer0e0c2d66 now completes and replays all nine original cases without
failures, retaining byte-identical inputs and complete tagged parents. The
new fiber models recover sum-cube and fourth-power completely, improving this
family from3/9 to5/9 all-split passes. They add two models,9507 coefficient
trials and194 raw-frequency audits; all CE gradients remain zero. Complete
construction including older parents takes903.440s, with171.3MiB peak process
RSS. The full105 comparison must retain stronger earlier families; this is
not yet a full35 result.

## Categorical constant strips (2026-11-13)

Task #219 / CM25 uses a latent domain three times larger to mark constant
categorical input rows/columns. Their intersection remains constant while the
unflagged interior retains its arbitrary base phase decoder. Lean proves the
actual modular construction, membership in the enlarged convex prediction
hull, unchanged radius-N latent bounds, embedding of prior phase vertices,
ordinary-CE optimality from checked observed fit, and transfer of actual
interior error descent to full observed CE. A closed nonconstant witness
escapes CM24's surjective-axis capacity obstruction. Native discovery must
still propose and verify flags from training data; a constant-looking partial
row does not identify its unseen labels. Lean `84cacc46` also proves ordinary-CE
descent for independently chosen phases when input categories share phases and
input/latent cardinalities differ. The native construction/refinement core now
passes eight tests, including exhaustive checks of all512 binary3x3 target
tables, a masked recovery and fully covered but ambiguous unseen rows. Raw
all-frequency SVD/CE and integer minima are checked; full input/state/work replay
is required. Bounded generic flag/phase discovery now passes six further checks:
observed categorical constant strips, angular phase quantization, repeated
phases and empirical decoders recover a separate masked fixture without its
latent coordinates. Every power/refinement state, partition, missing axis and
cap is retained. The original harness now passes four checks, preserving all
balanced parents and earlier-first CE ties and rejecting rehashed modifications
to both old and new ledgers. Frozen producer`69c83182` now completes and replays
all nine original tasks: still3/9, zero process failures. Division reaches
55.670% and blind absolute difference61.403%; both still fail. Sum-cube's lower
observed CE selects a candidate with worse validation (41.318% versus42.912%).
All parent artifacts and observed-only choices remain intact. The search uses
112 seeds and1962 certified axis steps, with every refinement ending at a
repeated state; peak RSS150.2MiB and zero CE gradients. CM26's complete-fiber
constraints are the next repair. Full35/modes/transformer integration stays open.

## Observable feature balancing (2026-11-13)

Task #218 / CM24 proves a feasible uniform PSD trace-one density after
normalizing each actual nonzero feature row to squared norm equal to its
column count. Explicit zero-row completion also gives a feasible domain;
it changes those features and identifies no hidden phases. Completing only
unobserved phase-axis entries preserves ordinary observed CE and the same
decoded spectral hull. A separate Lean obstruction shows that a surjective
phase axis and a constant opposite row force the entire decoded table to be
constant. The native permutation proposal family therefore has a capacity
limit beyond its numerical feature problem. The bounded Rust balanced and
missing-axis proposals now retain all prior candidates/failures, raw matrix
and latent audits, and certified refinement states. Five additional native
tests pass; the masked fixture has a correct rank-one uniform candidate while
other candidates remain wrong. The frozen producer `4510ee7d` has now completed
and replayed the unchanged nine original tasks: still3/9, zero process failures.
Division gains a candidate at38.810% validation accuracy and blind absolute
difference improves to50.223%; neither passes. All parent reports are unchanged.
The new search uses316542 interval iterations,144 audited seeds and2180
certified assignments; peak RSS85.7MiB. Address the capacity restriction next
in task219 with a generic categorical construction. Freeze tested source before
new original discovery. Full35/modes/transformer success remains open.

## Observed-label refinement (2026-11-13)

Task #215 / CM23 proves the exact one-hot ordinary-CE/error identity, an
independently checkable assignment dual certificate, decoder count descent
and feasible row/column/decoder updates in the existing spectral decoder hull.
Native refinement now reduces the masked fixture from 53/113 observed errors
to zero with two certified assignments and three decoder refits; all eight
withheld fixture cells are correct. It retains the inaccurate interval candidate
and every earlier candidate, exposes integer dual certificates and full states,
and preserves the earlier correct model on CE ties. All 26 native checks pass;
all four retained comparisons replay. Peak RSS is 69,896 KiB under the 64 GiB
cap. Task #216 now freezes the same policy for the nine original pair tasks
in `experiments/convex_stress_density/cohort`, retaining the earlier CM15
proposal and every exact/interval/refinement failure. It seals training-only
selection before split metrics and provides independent full replay under a
64 GiB cap. The frozen original run now completes and replays all nine tasks:
3/9 pass all splits, with no improvement over the earlier decoded proposal.
All 54 exact attempts refuse, all 162 interval attempts hit their caps, and
division's earlier constructor refuses a missing input axis. Peak RSS is
47,000 KiB. Retain all failures and the full35/modes/transformer scope; investigate
feature balancing and uncovered axes next, with Lean before model changes.

## Convex density tolerances (2026-11-13)

Task #213 / CM22 proves a convex interval tolerance for density diagonals,
explicit hidden off-diagonal and feature-perturbation bounds, and concentration
under a declared real-objective gap. A mixed approximate optimizer attains the
bound in a closed example. The feature-error hypothesis remains structural;
missing labels do not supply it. Task #214 now has a bounded Rust interval
core with a separate model tolerance, raw support-box dual audits and complete
replay. Its six checks retain mixed real ties and distinguish genuine feature
perturbation from numerical convergence tolerances. A centered, normalized
noisy subspace refused by exact constraints is admitted at model tolerance
`0.1`. Its categorical bridge retains all exact and interval proposals and
earlier decoded candidates. The masked fixture's initial 65,536-iteration cap
skipped two slots; a 73,728 cap covers all 18, admits one candidate, but that
candidate has 53/113 observed errors and 6/8 hidden errors. The earlier correct
decoder stays selected by observed CE. All 21 native checks and full raw/state
replay pass; neither numerical convergence nor the latent flat-spectrum audit
identifies the missing answers. Investigate observed-label-guided phase and
decoder recovery next, preserving the failed proposals and original scope.
Approximate real ties and unseen-label identification remain limitations.

## Convex matrix extraction from a supplied subspace (2026-11-12)

The first task #212 native step is `convex_stress_density`: a bounded PSD
trace-one/affine ADMM extractor with two linear objectives, full stage traces,
independent raw matrix/dual audits and saved-input replay. Exact complex
rotated/permuted fixtures pass, including a real-score tie that stays mixed
until the imaginary stage. A centered, re-orthonormalized noisy fixture has
inconsistent affine equalities and is explicitly refused. This is evidence
for the exact-subspace gap, not a masked-label recovery result. Connect the
extractor to declared training-only sketches without changing frozen parents;
retain failures and require a separate protocol before any original-data run.
The categorical bridge now passes the complete 121-cell fixture but refuses
all six subspace proposals when eight cells are removed. It ties right phases
to the same original sketch, audits latent spectra and ordinary CE, preserves
the successful earlier CM15 candidate and selects it by observed CE. Twelve
native checks include renaming, replay, tampering and full work accounting.
This measured masked-subspace failure motivated the Lean-first quantitative
CM22 bounds and the measured interval follow-up above.

Task #211 / CM21 proves the PSD trace-one lift suggested by CM20. Constant
diagonal readings force a diagonal hidden Fourier density when frequency
differences are separated. The complete feasible set is the convex hull of
pure mode projectors, including after an unknown unitary basis rotation.
Mixed matrices stay feasible; there is no rank constraint. Two observable
linear objectives, real then imaginary reading of two distinct rows, have a
pure optimizer and force every optimizer to be pure at a prime modulus.
Observable column centering removes the constant mode, giving an injective
relative-phase column. The rationally rotated three-mode modulo-13 example
inhabits the full extraction theorem. The modulo-2 collision has a feasible
off-diagonal density and prevents dropping separation.

The exact-subspace assumption remains the main gap. This is not recovery
from masked labels or a numerical convergence theorem. Before original-data
runs, build a separate bounded Rust/Kernite density extractor and audit its
matrix, both objective stages, residuals and failed attempts. Use exact
rotated-subspace fixtures first, then declared perturbations and categorical
masked sketches without true latent coordinates. Keep noisy-subspace success
empirical unless separately proved. Right-side phases must use the same
original sketch; every proposed latent table still requires ordinary CE and
the existing spectral/replay checks. Retain all earlier candidates, every
failure, full35 short/long scope, full/mini/forward modes, the transformer
comparison and the 64 GiB process cap. No new numerical producer or original
score is introduced by this proof commit; current frozen jobs continue.

## Constant-amplitude extraction from a Fourier subspace (2026-11-12)

Task #210 extends the theory before changing the frozen many-to-one constructor.
CM20 isolates every cross-mode coefficient product when ordered frequency
differences are separated. A nonzero vector of constant amplitude in that exact
subspace has one Fourier mode; centering removes the constant mode and a prime
modulus makes its phases distinct. Total amplitude deviation also bounds the
cross-mode products. The three-mode modulo-13 witness is inhabited and
nonconstant; a two-state collision counterexample prevents dropping separation.

The unit-amplitude coefficient set itself is proved nonconvex. Do not impose
it directly on the convex model. Before native work, investigate a positive
semidefinite, trace-one matrix lift with affine amplitude readings and a linear
selection objective. Prove or refute whether its feasible extreme points select
single directions under the same separation condition. No rank-one constraint
may be hidden in that convex domain. Right phases must then stay tied to the
same left direction, with ordinary-CE decoder fitting and the previous spectral
audits. A later native prototype must remain separate from frozen producers,
use bounded subspace sizes and work caps, and retain every attempted candidate
under the 64 GiB limit. No original-data follow-up is frozen by this proof step.
The proof neither recovers the subspace from masked observations nor proves
projection convergence, true-decoder identification or an original-task pass.
Keep the full35 goal and the unchanged live comparisons.

## General decoder propagation (2026-11-12)

CM19 at `0807609b` extends the Lean result to arbitrary fixed decoder fibers,
even cyclic groups and finite decoder banks. Every compatible assignment
survives finite support pruning in a normalized anchor branch; singleton unions
over every branch and bank member certify answers. The resulting linear
equalities preserve convexity and the realizable ordinary-CE optimum. A
three-state fiber supplies a concrete unseen answer with an ambiguous latent
state. No unknown-decoder discovery, coverage rate or optimizer convergence is
proved.

Task #207 implements this finite component in `convex_stress_propagation`.
Fourteen native tests include 11,840 small observation/budget cases against all
compatible completions, separate dense replay, a five-state witness with no
singleton fibers, tamper rejection and explicit resource refusals. New work
must preserve all decoder branches in a certificate, the 64 GiB cap, every
previous candidate and refusal, and the full35 short/long goal.

Task #208 integrates the general-fiber solver in `convex_stress_general` after
the Lean endpoint proof at `7d299b04`. Twelve native checks cover all three
modes, partial-coverage learning, independent DFT/SVD audits, exact saved-state
replay, repetitions, renaming and full-bank refusals. A three-by-three fixture
reaches CE 0.551444714 from 1.030216342 in full/mini and from 0.678028932 in
forward, which evaluates no gradients. One solver optimizes one fixed decoder's
convex subdomain; it does not optimize the full hull. Lean `aef9d966` also
records a permanent poor-anchor stall and recovery with another observed anchor.
Task #209 supplies the declared bank, anchor discovery and original-data protocol.
The solver fixtures alone do not update original-task scores.

Task #209 adds bounded observed-anchor discovery in `convex_stress_general_bank`.
The native search selects one complete independently checked whole-bank
certificate, keeps all attempts/refusals and budgets total producer comparisons;
it does not merge incompatible fallback constraints. Five checks cover the
Lean permanent-stall example, larger fibers in a four-member declared bank,
genuine cross-decoder disagreement, replay and resource/input refusals.
The vocabulary-only bank contains four coarsenings of a cyclic fold. It is an
explicit prior, not unrestricted decoder learning. Before original discovery,
freeze the complete harness/protocol, source and binary. The harness now fixes
all eight prior pair tasks plus the long blind task, all three modes, original
metrics and batches, CE-only stops/selection and the 64 GiB cap. Its 27 cases
retain all 108 decoder slots. Twelve native checks pass, including complete
state/update replay, larger fibers, complex raw spectra, spectral rejection,
rehashed artifact tampering and process-failure accounting. Only checked
endpoints may enter summaries; preserve partial files and every failed case.
The protocol fixes 16 anchors, 16 sweeps, one billion producer comparisons,
128/512/128 update caps and 12 endpoint repair rounds. Producer `eb65aa73` now
finishes all 27 cases with no process/replay failure: six returned models and
102 explicit decoder refusals. Absolute difference and its blind long variant
each reach 100% on all 4,705 held-out pairs in full/mini/forward; the other seven
tasks refute this bank in every mode. Successful cases certify all 9,409 cells
at their first anchor. Full state replay and raw decoder/DFT/SVD/CE audits pass;
peak RSS is 331,628 KiB. Their update-zero certificate fallback already gives
ideal training CE. Each stops at update 19; the blind endpoint uses fallback
weight zero, so no convergence benefit is claimed. A separate native comparison
harness preserves all older candidates, tests CE-only choices and refuses
partial parents. Actual integration remains pending the complete certified/
decoded chain and recovered full-suite baseline. Keep #209 open, the full35
goal, the transformer comparison and all earlier failures.

## Certified paired-decoder answers (2026-11-12)

CM18 at `5c0fe9bc` proves training-only identification under the paired-phase
family and coverage conditions, and proves that the unchanged spectral CE
relaxation can still have a wrong global minimum at a certified cell. Task #201
implements the proved linear restriction in a separate `convex_stress_certified`
crate. Its exact integer checker, bounded proposals and constrained full-batch,
minibatch and forward-only solver pass twenty native fixture checks, including
all compatible three-state phase completions, partial coverage, exact state
replay and raw spectral audits of endpoints that preserve certified equalities.
The protocol now fixes the eight original pair tasks and all three modes, the
4096/16384 discovery caps and 12-round endpoint repair; full replay and raw audits
are tested on fixtures. Producer `d7164082` now completes all 24 original cases:
absolute difference reaches 100% on all 4,705 held-out pairs in every mode;
the other seven tasks are uncertified in every mode at the frozen search cap.
Every result replays; the three models pass raw coordinate and spectral audits.
Peak process RSS is 134,556 KiB. The certificate identifies the successful answers
before optimization. Integration stays pending the complete unchanged paired and
decoded parents; keep #201 open and do not infer a full-suite pass.
Do not alter the frozen #195 source or live cohorts. Retain previous candidates
and all refusals; this follow-up does not replace the all-35 short/long goal.
The user's new memory limit is 64 GiB per run, with pre-allocation buffer
checks and a native process cap. Existing frozen cohorts keep their limits.

## Calibrated convex causal readout (2026-11-12)

The user authorized a fresh short/long test after the Lean-only model work.
CCa1–CCa2 at `281fb852` supplies the calibrated log-probability CE proof before
native work. Task #199 freezes both scaled and calibrated readouts on the
unchanged compact count-channel domain: 70 original full-batch cells, seed zero,
64 updates, training-only CE/gap stops and a 120 second between-update budget.
Each process has an 8 GiB memory cap and a 600 second hard wall limit. Evaluate
the original splits only after freezing the endpoint; retain every failure and
replay every returned model. Keep the other agents' spectral/depth campaigns
unchanged. This independent loss ablation does not wait for or select among
their endpoints, and is not a full claim of the universal feature construction
or its conditional unseen-answer identification hypotheses.

## Fixed decoding of bounded latent complex states (2026-11-12)

CM15 in `review/flat_spectrum.md` extends the model before native work. A fixed
many-to-one categorical decoder maps latent probability tables into the full
original output vocabulary, preserving probability rows and convex ordinary
CE. The hull over all fixed decoders is also convex; any feasible exact decoded
fit globally minimizes CE over that common prediction class. A concrete table
excluded by the direct binary rank-one budget is admitted by three latent
states. Two decoded global minima still disagree on its unseen corner despite
connected matching coverage. Native proposals must therefore retain numerical
latent-budget checks, original held-out metrics and the unchanged predecessors;
capacity and ideal observed CE alone are not recovery certificates. Decoder
proposal search has no efficient global-oracle claim. No new native result or
all-35 pass is asserted by the proof extension.

The native `convex_stress_decode` prototype now proposes latent phases from
centered categorical complex sketches and alternating assignments, followed by
an observed-count decoder. Six fixture checks pass, including missing output
classes, class renaming, raw latent spectral audits and explicit search failures.
Every attempted sketch and its work counts are retained. Original-data runs,
continuous full/mini optimization and comparison with frozen predecessors
remain separate work; the toy recovery test is not a stress-suite result.

`convex_stress_decode/protocol.json` now freezes eight original-data constructors
with the tested 16/64/24 trial/power/refinement limits. Preserve all imperfect
and rejected proposals and every internal trial. Ten native checks pass,
including saved-table/trace/metric replay and tamper detection. Gate execution
on the complete unchanged phase cohort and its cold/graph parents. Compare each
new endpoint once with all three previous training modes, selecting by original
CE alone and retaining the previous candidates and costs. This is not three new
gradient runs or a full35 result. Keep the serial queue and memory caps.

## Bounded conventional transformer comparator (2026-11-12)

The fresh comparison now has a native backend in `convex_stress_transformer`.
Extend the existing validated causal softmax engine to the original one/two/six
layers, original variable masks and streamed token-weighted CE gradients.
Six checks pass against the independent f64 reference and the unchanged
two-layer implementation. All 35 original datasets reconstruct with checked
hashes; the largest conservative model buffer bound is 342 MiB and data
reconstruction peaks at 244 MiB RSS under an 8 GiB process cap. No original
training or prediction has been performed by this preflight. Freeze a separate
training/observation/replay protocol before using this backend for comparison;
record the inherited AdamW warmup difference from the historical stress setup.
Keep the running convex cohorts and their serial queue unchanged.

The comparator now also has immutable streamed f32 checkpoints, with bounded
metadata/payload reads and a case/source/data binding. Nine native checks pass:
the three added checks cover bit-exact continuation of full/mini fixtures at
one/two/six layers and rejection of damaged or invalid state. This avoids
whole-file tensor copies while preserving Adam moments and sampling RNG.
No original-data training is started by these checks; freeze the remaining
training, observation, stopping and explicit-resume protocol before a cohort.

`convex_stress_transformer/protocol.json` now freezes all 35 original tasks at
seed 1, retaining their original 20k/30k/100k update caps and minibatch sizes.
Observe every 250 updates; stop at the first positive-update original primary
validation score of 99%, otherwise the original cap. Evaluate all original
splits at that endpoint. Preserve the comparator's documented AdamW/sampling
conventions and every checkpoint pair. Twelve native checks pass, including
every-observation prediction replay and an exact final-update reconstruction;
the latter is not a complete independent training rerun. All 35 additional
replay-buffer forecasts pass, at most 387,244,032 bytes; data-only preflight
peaks at 250,656 KiB RSS. Gate the serial cohort on the decoded cohort finishing
and preserve partial attempts without automatic resume. No original training,
transformer result or full35 convex success is asserted at the freeze.

The frozen `414431c7` transformer cohort is now complete: all35 original
seed-1 tasks and endpoint replays pass provenance, original-metric and exact
final-update checks. The complete summary verifies every observed checkpoint
hash. It has33 primary passes and32 all-held-out passes. Easy depth stops at
250 updates after primary success but scores97.0703% on length128; sum fourth
power ends at100,000 updates with92.4336% validation; long MQAR ends at20,000
with0% exact held-out sequences. The recurrence and shifted-AND tasks pass.
All617,750 updates and failures remain. Summed run time is15,173.915s excluding
replay; peak RSS269,964KiB. All12 native tests/fmt/clippy pass again. Summary
SHA25670f26e21fa756fd54416f8ac14116d63aa67950439a6d4048236dae904080057.
This CPU comparator retains its own documented sampling/optimizer conventions;
no matched-compute speed claim is made. Full105 integration with task225 and
reviewed task175 remains task226; full convex recovery is still unfinished.

## Prediction-preserving spectral sharpening (2026-11-12)

Lean CM11 landed at `b3c2e8ae` before the native follow-up. The
`convex_stress_sharpen` pilot reads all 36 frozen sharing endpoints, decodes their
own active predictions, restores the original spectral bounds with a common
uniform mixture and checks both norms again. Select a proposal only from
original training CE and preserve already-ideal coordinates exactly. Every
active argmax and every split correctness count must remain unchanged.
Keep all 36 proposals and 18 budget choices; independently audit raw tables,
the CE bound and parent comparisons. No training restarts or new gradients.
All 36 checks and replays, 72 raw-table audits, 36 parent comparisons and
18-choice replay pass. Exactly the three square-times full-rank endpoints
improve, from CE excess 0.00093613/0.00014259/0.00174838 in full/mini/forward
to 2.31e-12, with every prediction unchanged. The other 33 retain exact parent
coordinates. All 18 choices still pass and now attain the numerical ideal-CE
tolerance. This addresses the square-times CE gap before hull selection;
it cannot repair wrong unseen answers. The user additionally
requested depth: a separate follow-up must compare complete nonlinear paths
and independent routing under CD5-CD7 after this endpoint check.

## Missing-answer ambiguity and fixed sharing (2026-11-12)

Lean CM10 at `78a6743d` proves a spectrally feasible, globally CE-optimal
completion can remain uncertain on unseen answers. Correct fixed sharing
repairs the witness. The [sharing pilot](experiments/convex_stress_sharing/protocol.json)
proposes input groups using only training-label overlap, rejects conflicting
merges, and retains every original observation's weight. Conditional grouped
prediction families remain convex; the discrete map search is not claimed
jointly convex or guaranteed correct. Freeze 36 candidates in all three modes,
keep the unshared failure baselines, and require all 18 original success-task
endpoint controls to match exactly. Preserve the earlier cohorts unchanged.

Completed: all 36 candidate replays, independent audits and coverage analyses
pass, all 18 original controls match exactly, and all 18 training-only choices
pass the original 99% metric. Square-times hits its caps without ideal CE;
the remaining k=3 errors lie in wholly unobserved groups. These six-task results
do not establish full-suite recovery. Keep the original all-35 cohort separate.

## Intermediate spectral capacity (2026-11-12)

Lean CM9 at `efaf0d93` proves an actual rank-two binary probability moment is
excluded by both endpoint budgets and their spectral convex hull, and admitted
by a fixed intermediate budget. The [budget follow-up](experiments/convex_stress_budgets/protocol.json)
adds two generic vocabulary-only bounds, retaining the imported solver and
ordinary CE. Freeze 72 candidates on three observed failure diagnostics and
three original success controls, in all three modes. Keep every outcome and
compare all 18 old endpoint-budget controls exactly. This is an adaptive
follow-up to the independently frozen all-35 run; do not alter that cohort,
its candidate ordering, budgets or limits. Admission alone is not recovery.

Completed: all 72 candidates and 18 choices replay and all 72 independent
audits pass; all 18 old endpoint controls match exactly. Neither new budget
passes any of its 18 tests. Exact training CE still accompanies failures on
missing answers. Keep these negative results and proceed with the separately
frozen sharing follow-up; do not select a budget using held-out accuracy.

## Flat-spectrum connections with convex depth (2026-11-12)

CD8 at `32a9b668` additionally proves convex ordinary CE for a learned combined
path/feature/class tensor with per-class absolute-sum budgets. The native
`convex_stress_depth` follow-up implements fixed causal path blocks, a bounded
combined head, an allocation preflight and training-only full/mini/forward
checks. No held-out stress accuracy is claimed. Original easy/hard depth,
long MQAR and short binding are its initial
diagnostics. Check buffers against 2 GiB and run under an 8 GiB process cap.
The implemented Fourier/identity pair transports both skip and branch updates;
it is an expected coordinate-invariance control, not a claim that changing only
the skip improves accuracy. The first diagnostic passed on short binding at
depth one in all three modes, with training and replay below 100 MiB peak RSS.
The 2 GiB guard also rejected depth-4 MQAR before cache/head allocation.
All modes count their common
training-label initializer; full/mini also perform projected CE refinement.
Independent routing, additive controls and their checked budget embeddings are
implemented. `convex_stress_depth/protocol.json` freezes 288 original-data cells
over four depths, three architectures, two transports and three training modes.
Run and replay the entire inventory before its 72 training-only depth choices;
retain rejected allocations and failed performance. No cohort result is claimed
at the freeze stage, and a fresh matched transformer baseline remains separate.

The frozen producer `032762e5` has now completed and replayed all 72 binding
candidates. Depth one passes all modes and controls; final-only depth two
joint/independent heads fit training CE near zero but score 98.63%, and depth
four is worse. Keep the complete cohort running unchanged. Lean CD9 at
`9136a122` proves same-budget retention of all shorter path features. A separate
cumulative bank and streamed feature cache implement that capacity extension
under the existing memory limits. Native checks compare dense and streamed
CE/gradients exactly; retained-path stress training needs its own frozen
protocol, including any training-only preference for the first shallow channel
that attains the CE threshold. No retention result is claimed yet.

`retained_protocol.json` now fixes that 48-case follow-up. Both initializer
policies remain in the report; maximum depth comes only from the allocation
preflight (four for binding/depth, two for MQAR). Lean `fixed_margin_ce` at
`79f779ba` calibrates the first-channel cutoff to the pure margin-24 CE plus
`1e-8`, since a 548-class pure proposal itself exceeds `1e-8`. Optimizer stopping
and caps remain unchanged. The launcher requires the frozen 288-case cohort to
close before starting, enforces serial capped execution and disk reserves,
and preserves every streamed cache for exact original-prefix/boundary replay.
The original288 slots are now complete and replayed:270 models,18 memory
refusals and60/72 passing training-only depth choices. Joint and independent
families each pass24/24; additive passes12/24 and fails the two depth tasks.
The retained producer `9da1c677` completes all48 cases with100% sequence
accuracy on every original held-out split, in both transports, both initializers
and all three modes. Forward-only already achieves the same predictions.
Full/mini use one update on binding/depth; MQAR reaches16/256 update caps because
its margin-24 CE exceeds1e-8, with at most1.78e-16 CE change. Peak RSS1.99GiB
stays within the frozen8GiB and user64GiB limits. Preserve all original failures
and both retained policies when integrating; this is four-task evidence, not
the full35 or transformer result.

The user requested a theory extension to depth, residuals and the newer mHC
in the local DiffGPT repository, and authorized a new connection. The
[theory note](review/convex_depth.md) specifies ordinary residual connections
and fixed phase-conjugated Fourier connections with nonlinear branches that
read the evolving hidden state. Train a joint tensor over complete paths
through a finite block library. Lean proves ordinary-CE convexity in that
tensor, exact embedding of deterministic residual stacks, additive bounds
for bounded branches, and unit Fourier transport singular values. A two-block
parity witness retains a feature interaction that independent routing with
the same layer marginals erases. The additive fixed-feature baseline remains
as a comparison; a strict residual/mHC CE counterexample explains why raw
independently trained block parameters do not inherit convexity.

CD5-CD7 fix the library and decoder while cross-depth joint weights train;
CD8 allows a combined learned output tensor with a convex coefficient budget.
Full nonlinear path execution is retained, but dense
coordinates cost `|J|^depth`; this is not convex training of arbitrary raw
transformer weights. The native training-only check above supplies no new
held-out stress result, and no earlier full-suite outcome is changed.
Experiments must compare joint
routing, independent routing, and plain residual versus Fourier connections
under explicit library, data and compute budgets, retaining the original
short/long tasks and held-out metrics.

## Common convex predictions on all stress tasks (2026-11-12)

The first cohort has 311/315 endpoint records. On 2026-11-12 at 07:30:05 UTC,
two unbounded dense count auditors exhausted host memory; four training jobs
later ended with SIGTERM and no resumable state. Preserve their partial traces
and the completed outputs. The auditor now processes one query and sixteen
feature keys at a time, retaining the independent full-vocabulary oracle.
The large saved full-batch MQAR audit passes at 240 MiB peak RSS with unchanged
CE and a gap difference of 8.14e-14. Future audits run serially with an 8 GiB
address-space cap; training defaults to one worker with a 16 GiB cap. No whole
suite completion is claimed until the missing outcome and audit scope is resolved.

Task #204 completes the separate recovery of all four missing endpoints with
the unchanged producer, batches, caps and stopping under 64 GiB. All four replay
and pass independent audits; the original 311 outcomes and partial traces stay
immutable. The complete 315-slot, 105-choice baseline now passes selection and
replay. Every-split passes are 21/35 for full-batch and minibatch, 20/35 forward-only.
The recovered minibatch recurrence rank-one candidate reaches 99.766% validation,
but the unchanged training-CE selector chooses the count model with 0% validation.
Peak fresh-training RSS is 15.2 GiB. Retain all 240 models, three fit failures and 72
ineligible slots. This exposes a selection failure and remains below the full35
goal; it does not replace the separate depth, decoder or transformer cohorts.

Lean CM8 at `d7e1e6aa` proves fixed-scale ordinary CE convexity, common
probability feasibility and observed-fit/excess certificates for a prediction
hull containing causal and spectral families. The [full-suite protocol](experiments/convex_stress_hull/protocol.json)
fixes all 35 original tasks, three modes and three candidate slots: 315
records and 105 training-only choices. Keep every unsupported pair layout and
failed forward fit; remove held-out stopping from the count wrapper. Retain
the original solvers and diagnostic caps, check all data hashes and replay
every endpoint. Six integration tests pass before stress training. A fresh
full-suite result, including all failures, is required before claiming gains
over the earlier 10/35 count model or aggregating the three-task pilot's gains.

## Largest and total singular-value budgets (2026-11-12)

CM5–CM7 extends the coupled complex-moment class with convex operator and
nuclear bounds. Lean comes first: redundancy of the full-rank control, a
rank-one missing-answer theorem from ordinary CE, and a common convex-hull
certificate are proved. The [Schatten pilot](experiments/convex_stress_schatten/protocol.json)
fixes 18 cells on the original addition/division/multiplication splits with
both generic budgets and all three modes. Nine subsequent budget choices use
only training CE; if neither candidate reaches the ideal loss, report that
the full hull problem is not certified solved. Retain both candidates and all
failures. Eight native checks precede training. Preserve the previous six
successful full-rank outcomes and compare their saved coordinates directly.
This extension does not yet replace the 35-task objective or its transformer
comparison.

Completed: all 18 cells replay and pass independent direct-DFT/plain-CE audits.
Training-only selection passes all nine task/mode choices. Addition improves
to 100% in every mode with the rank-one budget; full-rank division/multiplication
retain 100% in full batch and forward-only, and 99.9785%/99.9575% in minibatches.
All six original positive controls match every saved coordinate exactly. All
nine opposite-budget failures remain at 0% at their caps. The selected models
meet the numerical ideal-CE tolerance, but the remaining minibatch errors and
the untouched sequence-task failures prevent a full-suite or unique-completion
claim. Keep the frozen pilot separate from any broader extension.

## Coupled complex moments (2026-11-12)

The single-phase spectral pilot fitted observed division/multiplication phases
with full-batch residuals near 3e-13 yet failed on unseen labels. CM1–CM4 now
proves convexity, consistent shared class codes, an observed CE lower bound and
uniform-mixture feasibility. The [coupled-moments protocol](experiments/convex_stress_moments/protocol.json)
fixes 18 cells: three original pair tasks, two radii and full-batch, minibatch
and forward-only arms. Native checks precede stress training. Keep all failures
and numerical defects, and replay outputs and diagnostic state. This pilot
does not replace the 35-task objective or claim a matched transformer result.

All 18 coupled-moment cells now replay and pass independent saved-table audits.
With R=sqrt(V), division and multiplication reach 100% held-out accuracy in
full batch and forward-only, and 99.9785%/99.9575% with minibatches. The latter
miss one/two answers despite exact observed fitting and tiny endpoint residuals.
All six successful cells attain the numerical observed-CE certificate. Addition
and the loose-radius controls fail; retain these results and the original caps.
This repairs two failures, not the entire 35-task objective. Next extensions
must preserve these results while addressing the incompatible addition bound
and the remaining task families without task-specific answer rules.

## Convex stress repair after formal failure witnesses (2026-11-11)

Continuation on 2026-11-12: the shared pilot repairs both recall tasks and, at
scale 10, branch generation, but mandatory sharing regresses on Boolean AND.
Lean MF4 now proves a sharing-capacity obstruction and an unrestricted oracle
preference. The [adaptive protocol](experiments/convex_stress_adaptive/protocol.json)
prespecifies all 35 tasks, three arms, two scales and seed zero (210 cells), with
the same diagnostic caps. Six native checks pass. Retain the shared-only pilot;
no general full-suite success is inferred from the regression witnesses.

MF5 further proves that pooled completion of unobserved rows preserves the
exact oracle cost. The [completion follow-up](experiments/convex_stress_completion/protocol.json)
fixes four tasks and all three arms at both scales (24 cells). Eight native
checks pass, including a missing-pair repair when a global shared fit is
impossible. Keep the adaptive cohort and its binary unchanged; this is a
separate targeted test, not a full-suite claim.

Completion results: all 24 prespecified cells replay, with 18 primary passes
and six ordinary-division failures. A separately frozen, adaptively selected
six-cell associative follow-up uses the same binary and all six pass. The
scale-1 minibatch associative failure improves from 53.866% at update 512 to
100% at update 19. All ten forward controls match exactly. Completion also
reduces the observed update count for short scale-10 and long scale-1 recall.
Ordinary division remains at 0.9450%; the full-suite objective remains open.

The adaptive all-task cohort is now complete: all 210 outcomes replay,
including 12 explicit forward-only failures. At scales 1 / 10, full batch passes
6 / 7 of 35 primary tasks, minibatches 5 / 7, and forward-only 6 / 6. No old
primary pass is lost; Boolean OOD failures remain. The independent uncentered
saved-model auditor agrees with the reported gaps within `3.5e-13` on three
failed full-batch endpoints and the completed associative minibatch endpoint.
Best observed all-task performance is 7/35; no full pass or universal recovery
of missing labels has been established. All frozen cohorts remain distinct.

Task #160 follows the user's instruction to prove the failure first and repair
full-batch failures too. The Lean statements in `ConvexChannel/Minibatch/` and
`ConvexChannel/Shared.lean` are landed before native experiments: singleton
oracle ties can preserve a constant route; gradient memory breaks the two-input
witness; aligned query sharing and fixed-scale ordinary CE remain convex.
These are algebraic guarantees, not general stochastic convergence or unseen
label identification. The 204 completed and six user-stopped old minibatch
cells remain separate and fully accounted for.

The [native repair protocol](experiments/convex_stress_repair/protocol.json)
prespecifies eight short/long tasks, three learning arms and two fixed scales,
with seed zero: 48 diagnostic cells. Seven native checks pass, including a
full-batch transfer witness and an ambiguous negative control. Freeze sources,
retain all failures, replay every endpoint, and distinguish fresh training gaps
from held-out errors. Pilot caps are 64 full-batch and 512 minibatch updates;
this is not an original-budget benchmark or a fresh transformer comparison.

Completed shared-only pilot: all 48 cells replay, including 12 explicit
forward-only failures. All three modes reach 100% short and long recall; at
scale 10 both CE modes also reach 100% branch sequences. Sixteen cells pass the
original primary threshold, but two are Boolean primary-only passes with
50.78% training accuracy, retained as regressions. Other task families remain
unsolved. The adaptive all-task continuation addresses the capacity regression.

## Prospective probability control (2026-11-09 active goal)

The user asks to close the gap between flat-spectrum guarantees and actual
training, and authorizes recreating the missing native training evidence.
The [probability-motion protocol](experiments/probability_tube/PROTOCOL.md)
fixes all three cohorts, eight seeds and 144 checkpoint slots before new
measurements. Derive future unused-class probability ceilings from checkpoint
logits and full parameter/feature motion, then feed those ceilings into the
actual-CE step condition. Retain tied encoder motion, class-dependent AdamW,
execution errors and every failed condition. A useful probability forecast
is one necessary step; full-model persistence, alignment and discovery remain
part of the active goal. Recreated evidence has separate provenance and does
not replace historical measurements.

The formal probability bridge is complete: [17 new Lean theorems](experiments/probability_tube/README.md)
propagate checkpoint ceilings through contrast motion and supply the CE
coefficient. Full build, axioms and native bound checks pass. The original
Rotate90 trainer at b70811e, pinned Kernite and all `diff-gpt` optimizer inputs
are now recovered and pass their exact source checks. The user's exact-source
prerequisite is satisfied. The reproduction runner is frozen at `9b336cb1`,
the first probability constructor/verifier at `26159c82`, and the matched
pairwise/Gram audit at `d3893996`. All 24 recreated trajectories complete,
retaining 240000 updates, 744 geometry probes and 1680 full checkpoints with
separate hardware provenance. Both frozen forecast constructions retain all
144 anchors and 1800 endpoints. Neither gives a positive raw CE stability
horizon at any of the 120 moving anchors, in either fixed metric. Full-model
control and the overall goal remain open.

Both future-state audits and the complete matched comparison pass, with no
missing cases or numerical failures. The original audit takes 462.71 seconds;
the refined construction, verification and comparison take 637.62, each under
its own 1800-second cap. The sharper per-query bounds never worsen the first
forecast, while executed probabilities, potentials and next-step accounting
match exactly. Actual old-probability ceilings are exceeded at 396 later
moving endpoints. The recreated raw full potential still increases at 31/30
next updates despite current-minibatch CE condition successes in 58/44 cases.

The full-encoder continuation derives the feature-motion radius through all
four RMSNorms, the moving real matrices, calibrated complex product and
residual. Its 27 additional Lean results connect that graph to the existing
tied source, propagate parameter caps to all-rival probability ceilings and
give checkpoint-centered warm-Adam caps. Native checks cover every stage and
parameter block. The user-supplied `ae801acc` named the trainer revision;
the refreshed `grokking` history supplied its verified bytes. Synthetic
validation is not a recreated checkpoint result.

The independent pairwise refinement adds 14 Lean theorems and four native
tests. It uses old logical-pair amplitudes in the complex product and proved
Gram caps for the same source matrices, intersecting the result with the
original enclosure. It preserves the frozen first audit. No trained
probability outcome is used to choose these formulas, and synthetic
improvements do not establish useful real-training horizons. In the recreated
comparison the sharper bound lowers 16 positive-horizon probability ceilings
below one, all in late settling, but still fails the CE coefficient condition.
At settling anchor 9500 its one-update maximum feature radius falls from
22.627 to 2.400–3.645; its smallest isotropic coefficient is still 158.60
against a required maximum of two. The next mathematical target is stronger
control of actual future gradients or their directional effects, alongside
less conservative CE aggregation. The observed prediction/geometry gap is
retained: final mean validation accuracy is 99.57–99.63%, while mean rank-32
certificate coverage is only 97.97–98.32%.

The query-weighted continuation now proves 27 further Lean theorems retaining
per-query probabilities and the full feature covariance, including finite
motion, arbitrary saved momentum, changing batch sizes and explicit GPU-to-real
CE execution allowances. Its [native library](experiments/probability_tube/gram)
passes five independent synthetic tests, including both source batch sizes
and a finite-difference check of actual CE. The new
[comparison protocol](experiments/probability_tube/gram_audit/PROTOCOL.md)
fixes both source forecast families, three coefficients, the full inventory,
inclusive future batches, execution conversions and a cumulative 7200-second
budget. Implement and freeze its old-only constructor and separate verifier
before evaluating recreated checkpoints. The original studies remain
immutable; no numerical gain or full-model stability is yet claimed for the
new coefficient.

## Checkpoint-fixed feature metric (2026-11-09 continuation)

Task #207 follows the user's continuation. Freeze the
[feature-metric protocol](experiments/ce_metric/PROTOCOL.md) before analysis.
Absorb class-shared feature scaling in the actual-CE momentum theorem and
freeze each metric from the saved state before reading the next update.
Compare all 144 previous cases against an isotropic control, retaining
current versus prospective step conditions, every adverse full update,
and the ordinary-energy penalty from metric conditioning. Full unknown
future control and discovery must not be inferred from finite diagnoses.

Completed: [21 Lean theorems and all 144 fixed comparisons](experiments/ce_metric/README.md)
absorb class-shared feature scaling, but the diagnostic gain is small:
full raw increases change from 31 to 30, and the ordinary-energy bound
becomes looser. The prospective q=1 condition fails on all 120 moving cases
in both metrics. All 32 later settling samples still decrease, without a
full-hold guarantee. Future small-probability control, class-dependent
adaptivity and tied encoder forcing remain unresolved; coverage is unchanged.

## Loss-specific momentum control (2026-11-09 continuation)

Task #206 follows the user's continuation. Freeze the
[CE momentum protocol](experiments/ce_lyapunov/PROTOCOL.md) before measuring
new quantities. Derive quantitative unused-class CE damping and an exact
main-beta momentum potential that absorbs saved memory. Replay all 144
declared next minibatches to measure the full model's decay, adaptive,
encoder and execution perturbations. Preserve failed stability conditions
and every actual potential increase. A partial-model theorem and finite
replay diagnosis do not establish future full-model control or discovery.

Completed: [24 Lean theorems and 144 verified native replays](experiments/ce_lyapunov/README.md)
remove saved-memory alignment from the partial-CE potential bound. All
120 moving partial references decrease, but the full raw potential rises
in 31 cases, including 13 meeting the sufficient reference condition.
The stable reference still has 36 full-model increases. All 32 sampled
settling updates from anchor 4000 onward decrease under both potentials;
these statewise diagnoses do not certify the full hold or reliable discovery.
Coordinate-dependent scaling has a large positive mean credit on the long
plateaus and a favorable mean credit after settling. Future signed control
remains the missing full-model step; no query certificate is changed.

## Future forcing from current moments (2026-11-09 continuation)

Task #205 addresses the user's remaining discovery/future-gradient gap.
Freeze the [checkpoint-only protocol](experiments/future_forcing/PROTOCOL.md)
before measuring it. Derive a warm-state cone for every future gradient
history under the actual .9/.95 moments and known schedule; quantify its
useful horizon and its limitations for directional alignment. Freeze each
native forecast before reading any later trace. Include initialization,
early learning and late cooldown; preserve all failures. Exact-real future
control, FP32 guard checks and reliable discovery remain distinct claims.

Completed: [25 Lean theorems and the complete native audit](experiments/future_forcing/README.md)
remove the future-forcing observation premise in exact arithmetic. All 144
frozen windows and 444,000 future prefixes pass the declared numerical
guards. At step 9500, every settling seed has a future active-share floor
above 67.83% through all 500 remaining moving updates; the uniform control
already clears 2/3 there. At step 4000, current moments add only three steps
to the 60% guarantee, and the full 4000-step hold remains uncertified.
Reliable discovery and long useful directional control are still unproved.

## Certify finite-window persistence (2026-11-09 continuation)

Task #204 follows the user's continuation. Freeze the
[observed-persistence protocol](experiments/geometry_persistence/PROTOCOL.md)
before new measurements. Replace the loose uniform direction estimate by
explicit realized forcing budgets with common decay handled exactly. Audit
every hold update, then transport old certificates over 192 fixed finite
parameter displacements, including the nonlinear feature remainder. Keep
every control and failure. These are conditional maintenance results; they
must not be described as discovery or prediction of unknown future updates.
The [completed study](experiments/geometry_persistence/README.md) proves ten
new Lean theorems and verifies all 192 windows and 96000 hold prefixes.
Early settling now has an active-readout share lower bound of 61.76–65.88%
through every moving hold update in all eight seeds. Transported rank-33
validation coverage is 98.510% at 250 updates and 98.045% at 1000, but only
0.244% for a single 4000-update displacement. The higher-rate controls remain
mostly inconclusive. There are zero false certificates or bound violations;
discovery and future-gradient alignment remain open.

## Bound optimizer motion and improve certificates (2026-11-09 user direction)

Task #203 follows the user's request to pursue the main proof gap, improve
certificates and examine persistence under continuing low-rate updates.
Freeze the [optimizer-motion and certificate protocol](experiments/geometry_dynamics/PROTOCOL.md)
before measurements. Derive the .9/.95 moment invariant and a finite-rate
motion guarantee with explicit assumptions. Audit all three existing cohorts.
Test active readouts centered at the unused-class mean, retaining all rivals,
and report a separate envelope with its extra basis cost. Do not present a
conditional maintenance theorem as proof of discovery or hide loose bounds.
The [completed study](experiments/geometry_dynamics/README.md) adds 21 Lean
theorems with standard axioms only. All 240000 readout updates and 744
certificate states pass. The gradient-independent motion bound remains too
loose: 56.58–56.98 versus 0.76–0.88 measured displacement in the low-rate hold.
For early settling, the new single rank-32 basis falls slightly; rank 33
raises mean coverage to 98.565%. A three-basis rank-33 envelope reaches
98.626%, reducing correct-but-uncertified predictions from 403 to 357 with
unchanged accuracy. All extra basis costs and failed sufficient conditions
remain explicit. Discovery, feature alignment and global convergence remain open.

## Close the geometry gap (2026-11-09 user direction)

Task #202 follows the user's request to close the remaining gap. Freeze the
[early-settling and active-answer protocol](experiments/geometry_gap/PROTOCOL.md)
before new measurements. Keep the copied AdamW kernels and all eight seeds;
reduce .01 to .0001 during updates 1000–4000, hold to 8000, then cool to
1e-8. Independently compare a basis fitted to observed answer classes on
all saved states of the .001, .01 and new cohorts. Every rival class stays
in the bound. Preserve all original numerical allowances, late probes,
budgets and failed slots. Both comparisons are complete and independently
verified: all eight training seeds and all 744 certificate states pass,
with zero false certificates or bound violations. Together, early settling
and the active-answer basis raise per-seed late rank-32 coverage minima to
98.11–98.66%, with every sampled late accuracy at least 99.51%. Mean final
coverage is 98.504%, versus 98.015% for the long .01 plateau; correct but
uncertified endpoint predictions fall from 608 to 403. Mean final accuracy
is 99.575%, versus 99.631%, and incorrect predictions rise from 139 to 160.
All 32000 hold updates retain nonzero class-contrast motion. Postprocessing
and its independent verification take 1499.62 seconds within the frozen cap.
No endpoint reaches 99% coverage, and no convergence claim follows.

## Test higher WSD learning rates (2026-11-09 user direction)

Task #201 follows the user's suggestion to increase learning rate. The
[protocol](experiments/class_geometry_lr/PROTOCOL.md), frozen at `637d3a1e`,
tests peaks .003 and .01 against task #200's .001, keeping every other
recipe setting, all eight seeds, 10000 updates and all probes fixed. Each
arm has its own fixed 1800-second cap and runs sequentially after #200.
Both native builds and 32 GPU preflight cases pass with exact FP32 agreement.
The user then explicitly prioritized immediate .01 training. Under the
[recorded amendment](experiments/class_geometry_lr/USER_PRIORITY.md), .003
stops with four complete seeds, one partial and three unrun. The
[completed .01 cohort](experiments/class_geometry_lr/README.md) passes all
80000 updates, 248 probes and independent margin checks in 877.96 seconds.
Mean final accuracy improves from 99.567% to 99.631%, with all eight pairs
improving. Mean final rank-32 coverage barely changes (97.970% to 98.015%),
while every late coverage minimum falls: 16.07–22.61% versus 89.95–94.69%.
First 90% coverage now occurs during cooldown. The higher peak helps final
accuracy but does not close the discovery/persistence gap. Preserve the
partial .003 arm without treating its missing outcomes as measurements.

## Transfer the main model's AdamW/WSD (2026-11-09 user direction)

Task #200 uses the user's requested native optimizer from `python_projects/gpt`.
The [protocol](experiments/class_geometry_wsd/PROTOCOL.md), frozen at `443138ab`,
copies the current selected FP32 kernels and the 2% warmup / 20% cosine-decay
schedule, with beta2=.95, decay=.1 and independent-coordinate clipping at 1.
Its formula is scaled to the existing 10000-update budget. Compare all eight
task #199 seeds and all fixed probes, retaining the baseline and every failure.
The [completed full-recipe transfer](experiments/class_geometry_wsd/README.md)
passes all 80000 updates and 248 probes, with zero false certificates. All
eight endpoint accuracies exceed 99%, versus six previously, but rank-32
coverage falls in six seeds and reaches 99% in none. Every paired late
coverage minimum decreases. No effect is attributed to WSD alone; beta2,
decay and clipping also changed. Six complete runs were reused unchanged
after a JSON-ratio comparison blocked two seeds at initialization; the
two missing runs and original attempt total 920.89 seconds within the cap.

## Pursue discovery and persistence (2026-11-09 user direction)

Task #199 freezes the [class-geometry mechanism study](experiments/class_geometry/PROTOCOL.md)
at `9692d9a0`, following the user's explicit instruction to pursue the remaining
learning question. Replay every update of both original seeds and six fixed
replication seeds. Separate 32 active-answer contrast directions, 63 unused
class directions, and one between-group direction. Audit actual adaptive,
tied-encoder and memory effects over every prefix, and test geometry at all
31 declared probes per seed. No monotone or universal discovery claim is
licensed by a local CE derivative. The [completed study](experiments/class_geometry/README.md)
passes all 80000 updates and 248 probes, with zero false certificates and
20 new Lean theorems. Active-class energy share rises from about 33% to
70–76%. Signed readout contributions build that share; encoder and memory
contributions oppose them. All six replications retain at least 93.99%
rank-32 coverage at every late probe, but none sustains 99% throughout that
window. Concentration fluctuates and is insufficient for alignment or a
correct prediction. Preserve these failures and the open global theorem.

## Extend flat spectrum to observable geometry (2026-11-09 user direction)

Task #198 freezes the [readout-aware comparison and update attribution](experiments/readout_geometry/PROTOCOL.md)
at `db7b55b3`. The [completed study](experiments/readout_geometry/README.md)
extends the geometry to class-centered readout directions. Nineteen Lean
theorems prove observable Gram/margin identities and signed update attribution.
Both original trajectories and all 26 probes pass, preserving every #197
hidden-baseline bound byte for byte, with zero false certificates.

At equal rank 32, endpoint coverage rises from 96.22%/97.45% to
99.21%/99.53%, removing 89.85% of the gap to direct forecasts. First majority
coverage moves to step 500 for both seeds, before the first sampled 99%
accuracy at 1500/1000. The endpoint gap remains 20/7 direct certificates;
28 correct endpoint queries remain uncertified in total. Rank-128 results
are not uniformly improved, and all later regressions remain recorded.

The actual AdamW decomposition separates decay, readout/encoder credit,
sampling, moment memory, rounding, projection motion and the mixed term.
Parameter, tangent and logit identities close, but readout credit agrees with
the total predicted validation-margin sign on only 44.12–77.24% of queries
across probes. This supplies a precise local mechanism audit; long-run
discovery and maintenance of the useful span remain unproved. Source
training, model accuracy and the separate task #189 efficiency study are unchanged.

## Connect training geometry to update forecasts (2026-11-09 user direction)

Task #197 freezes the [training-to-forecast bridge](experiments/transport_forecast/PROTOCOL.md)
at `07c328f1`, then replays the two original plain-product trajectories at 26
declared probe states. Fifteen new Lean theorems retain signed training
reconstruction, readout/feature movement, mixed terms and forecast remainders,
with margins, probability/CE consequences and actual-source adapters.

The [completed audit](experiments/transport_forecast/README.md) has zero false
certificates or enclosure violations; all six original checkpoint hashes and
six original next-update hashes match. Rank-32 endpoint coverage is
96.22%/97.45%. Its first majority coverage occurs at the same sampled step
as the first 99% accuracy for each seed, and later coverage can regress while
accuracy remains high. Thus the local conditional bridge is useful, but an
early prediction of the learning transition and optimizer discovery remain
unestablished. Full-rank reconstruction is a separate control. The task is
`(a+b)^3 mod 97`, not a new result for the original `a^3+ab` obstruction.
Task #189 remains the separate inference-efficiency study.

## Tighten pre-update certificates (2026-11-09 user direction)

Task #196 freezes the [forecast-tightening comparison](experiments/forecast_tightening/PROTOCOL.md)
before outcomes. Diagnose layerwise inflation and compare Gram amplification,
a local regularized-normalization secant bound, and signed class-contrast
intervals before the final RMS against the original 24 task #188 forecasts.
Prove the new bounds and audit all cases with unchanged numerical allowances
and source updates. Retain full CE/backpropagation; no local-learning restart.
Measure coverage and preparation cost separately. Task #189 remains the
separate inference-efficiency study; no runtime gain follows from coverage.

The [closed comparison](experiments/forecast_tightening/README.md) passes all
24 cases with unchanged source updates and numerical allowances. Eighteen
new Lean theorems support Gram/RMS propagation and signed final readout.
Trained-validation coverage rises from 68040/75280 (90.38%) to 74727/75280
(99.27%), against 74952 actually correct; difficult-subset coverage rises
from 73.71% to 93.17%. Zero false certificates or enclosure violations occur.
The bound also certifies 10166 validation CE decreases and 2090 increases;
63024 directions remain inconclusive. Combined Gram/RMS provides most of the
coverage gain, with 42 extra certificates from the directional intersection.
The complete multi-track forecast costs 1.258 seconds per 9409-query state,
so serving efficiency remains a separate implementation/measurement task.

Task #189 now implements its original fresh-input cached-weight and repeated
query reuse controls, plus the [premeasurement tightening extension](experiments/inference_efficiency/TIGHTENING_PROTOCOL.md).
The fourth method applies Gram/local-RMS displacement bounds without tangent
matrix products. All serving paths receive operand pairs only; compare
complete host-result latency, setup, memory and exact fallback for the same
24 states and fixed batches. Keep the original 900-second cap and retain
all outcomes. The 99.27% ground-truth certificate coverage does not predict
label-free reuse coverage or inference speed.

## Three efficiency targets (2026-11-08 user direction)

**Later user direction (2026-11-08): skip no-gradient training.** Stop further
Forward–Forward/local-learning candidate and credit-control experiments.
Retain completed evidence and pending reviews; continue optimizer efficiency
with standard full backpropagation. The proposed post-#195 credit comparison
below is superseded by this direction and will not be run.

The user requested all three targets below. They are planned as separate
native comparisons; task #188's valid feature forecasts establish no speedup.
Efficiency means end-to-end wall time and memory at matched quality, with
all computation charged to the candidate. Freeze each candidate and budget
before measurement, start with an instrumented pilot, and use its evidence
to justify any longer training run. Keep the current source, data, parameter
budget and seeds as controls. No Python or generated result commits.

| Target | Board task | Required comparison |
| --- | --- | --- |
| Faster inference | #189 | Latency, throughput and memory at matched predictions/quality; fresh-input inference and repeated-query inference across parameter updates are separate workloads. |
| Training without backpropagation | #190 | Further work skipped by later user direction; retain completed local-learning evidence and reviews. |
| More efficient optimizer | #191 | Optimizer update cost, complete step cost, memory and wall time to matched quality versus the actual source AdamW. |

For inference, count cache construction, storage, prediction bounds and
fallback evaluations. Any serving-time certificate must work without labels:
it certifies agreement with an exact model evaluation, not unknown ground-truth
correctness. The current tangent forecast performs extra dense products and
is not already a faster substitute for a normal forward pass. Savings require
measured reuse, structure, or an improved implementation.

For forward-only training, a directional derivative is not generally the
full gradient. Declare the number and source of directions, estimator variance
and total function evaluations. Do not count directions or useful initialization
obtained by earlier backpropagation as free. Separate learning from scratch
from continuation at a trained checkpoint; retain all seeds and failures.

Forward–Forward is a distinct candidate, following Hinton, arXiv:2212.13345,
§§2–3.3 (read locally in `papers/arXiv-2212.13345/paper.txt`). Its local
goodness objectives use positive and negative data. Measure negative-example
construction, local derivatives/updates and activation storage; removal of
end-to-end backpropagation alone is not a speed result. The supervised
label-conditioned inference in §3.3 evaluates each candidate label, so count
all 97 evaluations if that route is used here. A neutral-label learned readout
is a separate candidate with its own training and accuracy costs. Declare any
changes to the tied complex architecture before the pilot, including where
goodness is measured relative to RMS normalization. Preserve independent
seeds and compare total time to the baseline's accuracy, not just 90%.

Task #189 starts with the [frozen inference protocol](experiments/inference_efficiency/PROTOCOL.md):
exact cached weight transposes for fresh inputs, plus old-feature reuse with
label-free certificates and exact fallback across one actual parameter update.
The 90.38% figure from #188 is certificate coverage, not classifier accuracy.
Require 100% final prediction agreement with full inference on audited inputs;
report coverage separately from accuracy and speed.

The user's next direction is to reformulate Forward–Forward before the
inference implementation. Task #192 freezes a [local-loss diagnostic](experiments/forward_forward/PROTOCOL.md):
directional goodness from corrected class distances, all-class pairwise
softplus margins, and analytic local gradients. This avoids a separate
encoder evaluation for every negative label. It tests the loss and the
difference between detached readout updates and the actual tied network.
The complete training comparison is task #190 below. Task #189's frozen
protocol remains queued.

The [loss reformulation](experiments/forward_forward/README.md) now has 19 Lean
theorems and a native implementation. All 24 source cases pass: 225816 loss
checks and 21904152 logit-gradient checks, plus independent partial-readout
gradients. One small E-only step lowers the computed frozen-feature and
actual tied-network batch loss in all cases, but trained-state changes are
tiny. This verifies the loss implementation, not useful accuracy gains or
efficiency. Task #190 supplies local objectives for every block and a complete
shared-parameter update.

Task #190 froze the [complete native pilot](experiments/forward_training/PROTOCOL.md)
at `34da0951`, before outcomes:
32 from-scratch runs, four original arms and two seeds, 4000 updates per run.
CE backpropagation, pairwise backpropagation, full auxiliary-loss
backpropagation and local pairwise credit are distinct controls. Fixed
vocabulary/hidden heads add token reconstruction and task-label objectives;
the shared embedding receives all applicable contributions before one AdamW
update. The local variant does not propagate upper-block errors into earlier
blocks. Costs, numerical checks, caps and matched-quality criteria are frozen
before outcomes. Forward-mode/finite-difference training remains an untested
alternative; finite differences in this pilot are correctness checks only.

The [completed pilot](experiments/forward_training/README.md) retains all
32 runs and reports a negative result. Local pairwise credit reaches only
2.23–10.50% endpoint validation accuracy, versus 99.28–99.87% for CE; neither
the local nor full-auxiliary method ever matches paired CE endpoint accuracy
and loss together. Pair BP reaches that joint criterion in 2/8 cases, after
the corresponding CE control. All 224 gradient checks and 328 saved-metric
replays pass. Foreign GPU training workloads invalidate clean-device timing
claims; the measured data are retained. Do not extend or tune this frozen
pilot. Next diagnose local-update alignment with terminal credit, including
the tied embedding, before defining another candidate. That diagnosis would
use backpropagation as measured analysis, not hidden free training.

Task #193 freezes the [alignment diagnostic](experiments/forward_alignment/PROTOCOL.md):
40 initial/endpoint states, seven one-step counterfactuals per state, three
data scopes and independent complex-coordinate pairings. Separate current
gradient alignment from actual AdamW motion, inspect both tied-E sources,
and compare infinitesimal predictions with finite loss changes. All
diagnostic backward passes are counted; no new training trajectory is run.

The [completed alignment study](experiments/forward_alignment/README.md)
passes all 40 states, 1120 finite differences and 6720 coordinate pairings.
At the eight failed local endpoints, A lowers its own local objective in
all eight probes but increases full-training CE in all eight and validation
CE in seven. Its current gradient has positive validation CE alignment in
all eight, while the actual AdamW direction has negative alignment in five;
finite-step residuals overturn two further descent predictions. B-only
updates lower full-training and validation CE in all eight. Removing
vocabulary E worsens the complete step's validation CE in all eight.
Prioritize an A-only direction/step-size
diagnostic before changing loss and optimizer together. This is a proposed
next study, not an accuracy repair or a new training result; freeze it
separately and keep diagnostic full gradients out of training information.

Task #194 freezes the [A-only direction/scale diagnostic](experiments/forward_step/PROTOCOL.md):
the same 40 states, actual AdamW versus a norm-matched negative local-gradient
direction, and five fixed scales from 1 to 1/256. All 400 probes hold E/B
fixed and start from the saved state. Compare realized directions, local
loss, terminal loss and finite-step remainders; do not select only favorable
scales or extend training. Any later training repair needs its own protocol.

The [completed direction/scale study](experiments/forward_step/README.md)
passes all 400 probes and 320 directional checks. At local endpoints the
current-gradient A direction lowers training/validation CE in all eight
cases at every scale. Shrinking AdamW recovers two finite-step overshoots,
but leaves five validation ascent cases even at 1/256. Every probe across
all 40 states lowers its local objective; terminal outcomes at initialization
and the other origins remain mixed. Accuracy changes are mixed too.
Next freeze a from-scratch comparison with a standalone current-gradient A
rule, a locally computable step-size policy, original-local and CE controls,
fresh confirmation seeds and a fixed budget. The diagnostic borrows AdamW's
radius, so it is not itself a cheaper optimizer. No accuracy repair or speed
claim follows; keep all terminal gradients as charged analysis only.

Task #195 freezes the [standalone normalized-A training comparison](experiments/forward_normalized/PROTOCOL.md):
replace only A's optimizer with its current local-gradient direction and a
fixed 1%-of-A-norm step with ten-update warmup, without A moments or decay.
E/B and all objectives retain their original local rules. Compare against
original local-pair and source CE from scratch on seeds 17/23, all four arms,
4000 updates each (24 runs). No radius search or validation-directed updates;
all accuracy, cost, numerical failures and caps remain in the evidence.

The [completed normalized-A comparison](experiments/forward_normalized/README.md)
retains all 24 completed runs. Candidate endpoint validation accuracy is
2.23–11.63% versus 99.45–99.87% for CE; compared with original local-pair,
validation CE is worse in all eight cases and accuracy in six. No local run
matches paired CE quality. All numerical, state and source checks pass.
The fixed 1% relative rule therefore does not turn the earlier one-step
direction result into a training repair. The previously proposed control
using the same A optimizer with full-gradient credit is now skipped by the
later user direction above; no such cohort was run. Do not attribute failure
solely to the head or radius, tune this completed cohort, or claim a speedup;
all ownership snapshots include the desktop.

For optimizer efficiency, separate cheaper arithmetic per update from reaching
the same quality in less total time. Preserve exact baseline semantics, measure
the optimizer's share of complete step time, and validate the candidate's
update equations before attributing any difference to optimization quality.

Task #191 first froze a [source AdamW cost profile](experiments/optimizer_efficiency/PROFILE_PROTOCOL.md):
all 24 original states, fixed-gradient update-only throughput versus complete
CE steps, synchronized wall and CUDA-event timing, actual memory and process
ownership. The candidate was chosen and frozen after this baseline measurement;
no speed claim follows from optimizer operation counts alone.

The [source profile](experiments/optimizer_efficiency/README.md) passes all
24 cases and retains 576 samples. Update-only/full-step wall ratios range
from 2.47% to 5.52%, with another GPU training process present throughout.
Complex moment storage duplicates tied coordinates: 2194432 bytes versus
1146880 for independent moments. This motivated compact moment storage and
uniform-scalar precomputation with identical updates and trajectories;
the shared-device ratios are not an exclusive-device speed result.

Task #191 froze the [compact-moment candidate](experiments/optimizer_efficiency/PROTOCOL.md):
store complex moments once per independent coordinate and precompute uniform
FP32 AdamW scalars, preserving source update rounding. Require bitwise state
equality at 24 saved-state cases and all paired workloads, then on 16 fresh
CE/backpropagation trajectories (seeds 29/31, 4000 updates). Measure update,
complete-step and matched-quality costs; retain interference and slowdowns.

The [completed comparison](experiments/optimizer_efficiency/README.md) passes
all 24 saved states, 160 warmup updates, eight zero/tiny-gradient cases,
576 paired benchmark workloads and 328 paired training observations.
All 16 full-CE runs finish 4000 updates with identical paired states/metrics
and endpoint validation accuracy 99.45–99.87%. Complex moment storage falls
47.74% (1047552 bytes), or 3.68% of resident training buffer payload. This is
a validated memory improvement. Shared-device matched-quality active ratios
range 0.946–1.012, with four of eight below one; no reliable training speedup
or convergence improvement is established. Keep full backpropagation and
the measured storage change; larger runtime gains need a forward/backward
or inference comparison, and speed claims need uncontaminated timing.

Task #188 also leaves a theoretical target: tighter directional error transport
through the block and readout. Such bounds may support the three comparisons,
but a proof of a local bound alone is not evidence of runtime savings.

## Feature forecasts before updated-network evaluation (2026-11-08)

Task #188 completed the [frozen forecast protocol](experiments/single_update/FORECAST_PROTOCOL.md):
22 Lean theorems and all 24 native probes pass. The old-network tangent and
layerwise radii are persisted before any candidate-network evaluation. All
225816 feature balls and 21678336 competing-class bounds hold; fixed FP32
allowances have zero violations, and resulting states still match task #186.

Forecasts certify correctness on 68040/75280 trained validation observations
(90.38%), but **no loss-change direction**. Mean validation margin interval
width is 4.947498 versus the retrospective 0.001595. Plain-product endpoint
mean radii exceed mean actual forecast errors by factors 109.10/196.37.
The [authored results](experiments/single_update/README.md#feature-forecasts-before-the-updated-forward-pass)
retain all cases and the loss of coverage. The next target is directional
error propagation through the block/readout, avoiding repeated scalar-ball
and Frobenius inflation. No architecture or numerical allowance was changed.

## Joint control of difficult-query margins (2026-11-08)

Task #187 completed the [frozen extension](experiments/single_update/MARGIN_PROTOCOL.md):
35 proved Lean theorems and all 24 native probes pass, checking 225816 query
updates and 21678336 competitor pairs with no false certificates. Signed
parameter/feature contributions, mixed terms and rounding remain explicit;
all resulting states match task #186. On trained validation states, the joint
bounds certify 74677/75280 loss-change directions. The fixed old-margin
bottom decile still has worsening mean CE at both plain-product endpoints,
even though its mean margin increases; the loss bounds certify the regression.

The [authored results](experiments/single_update/README.md#joint-margins-and-difficult-query-losses)
retain all cases and inconclusive queries. This is conditional local control
using observed feature displacement. Predicting that displacement from
pre-update training quantities remains the next obligation; no convergence
or statistical generalization guarantee follows from retrospective coverage.

## Actual single-update audit (2026-11-08)

Task #186 completed the [frozen audit](experiments/single_update/README.md):
24 actual source AdamW updates all match ordinary `train_step` bitwise, with
the original compiled CUDA hash and unchanged input files. All 48 directional
gradient checks and 9360 projected decompositions pass. Encoder, minibatch
and moment-memory contributions have substantial signed cancellation;
feature/basis movement is explicit. Step-10000 probes remain outside the
original training budget and do not revise the architecture ranking.

At both plain-product endpoints, mean validation margin increases but
validation CE worsens and accuracy is unchanged. On 1082/1163 queries, margin
improves while the residual budget grows. Five of the 16 trained probes
increase mean validation margin while worsening validation CE. Thus the
next obligation is a condition on worst competitor margins and difficult
queries under joint forcing and feature/basis movement, retaining cancellation.
The report is retrospective; no new convergence theorem is claimed. Generated
outputs remain local, and the existing Lean statements and INDEX.md are unchanged.

## Adaptive forcing and signed interaction control (2026-11-08)

Task #185 adds 23 proved theorems linking actual buffered Adam updates to
explicit diagonal mixing, saved-moment discrepancy and moving interaction
budgets. The [derivation](review/flat_spectrum.md#adaptive-interaction-control)
retains distinct preconditioners for competitor and target class rows.
Sufficient contraction is proved from explicit forcing/drift control, not
assumed from CE descent. No new `sorry` is introduced.

The [fixed native audit](experiments/adaptive_interaction/README.md) preserves
all 24 states, 16 intervals, both groupings and ranks 16/32/64. All 6,240
positive-step adaptive checks pass; initial states retain geometry and an
explicit no-historical-direction status. Moment discrepancy is substantial,
and the objective-based bound is millions of times above actual projected
direction energy at the plain-product endpoints. There, rank-32 orbit
validation mean absolute interaction is only 10.76%/10.42% of the norm-product
bound. Nevertheless it increases in 6 of 12 CE-descent intervals across arms.

The current cubic endpoint ranking remains diagonal > plain > radial on
two-seed mean validation accuracy and CE. This is not a universal architecture
ranking, and complex multiplication is unchanged. The next obligation is
same-update control of the full-gradient/moment discrepancy and feature/basis
drift, retaining signed contributions relative to the classification margin.
Generated reports remain local; INDEX.md retains its committed exception.

## Residual learning dynamics (2026-11-08)

Task #184 derives the actual fixed-feature readout CE gradient, a loss-controlled
projected-gradient energy bound, and a sufficient Euclidean descent/decay
mechanism. The [Lean extension](review/flat_spectrum.md#residual-learning-dynamics)
adds 23 proved theorems, including a changing-subspace identity and an actual
Adam first-step counterexample: CE falls while a training-nullspace residual
appears. No residual contraction is assumed, and no new `sorry` is introduced.

The [fixed native diagnostic](experiments/residual_dynamics/README.md) retains all
24 original states and 16 adjacent intervals, both groupings, and ranks 16/32/64.
All 9,360 projected-gradient bounds pass. Among the 12 intervals with falling
training CE, rank-32 orbit readout residual energy rises in 11 and the mean
validation residual-norm product rises in 3. Changes in the basis can reverse
the sign of a change measured with the old basis held fixed.

The next obligation is to control the residual interaction under actual
adaptive updates and changing features. Small partial readout gradients do
not control current residuals or encoder drift. The sufficient fixed-feature
Euclidean theorem does not directly describe these tied AdamW trajectories.
Complex multiplication, training and the checkpoints are unchanged. Generated
reports remain local under the new AGENTS rule; `INDEX.md` retains its explicit
generated-and-committed exception.

## Signed directional transport (2026-11-08)

Task #183 preserves signed training-score reconstruction and bounds only the
remaining interaction between query-state and readout residuals. The
[Lean derivation](review/flat_spectrum.md#signed-directional-transport) retains
an explicit orthogonality defect and the actual normalized source execution
allowance. It also proves the zero-residual re-expression limit and a nonzero
extrapolation witness. Sixteen new theorems are proved, with no new `sorry`.

The [fixed Rust diagnostic](experiments/signed_transport/README.md) retains all
24 original states, both groupings and all nine requested ranks with paired
signed/directional bounds. At the plain-product endpoints, rank-32 orbit
probability coverage is 70.67%/72.50%; argmax coverage is 96.60%/97.43%.
Signed reconstruction with the full readout norm gives only 29.99%/32.86%
probability coverage at the same rank. All validation orbit bases achieve
rank 32 of 128, with substantial query residuals still present. Numerical
bound discrepancies stay below the fixed tolerance; no positive condition
classifies an input incorrectly.

This is useful conditional coverage from a separate training subspace per
answer orbit. Full-rank reconstruction can simply re-express the original
classifier. Coverage can fall with further training, as the SwiGLU seed-7
trajectory shows. The next obligation is to derive learning conditions that
make these residual budgets small. No model training, architecture change,
independent statistical test or optimizer-discovery claim was introduced.

## Probability transfer through normalized geometry (2026-11-08)

Task #182 extends the theory from training probability control to conditional
query probability floors and loss ceilings. The
[Lean derivation](review/flat_spectrum.md#probability-transfer) uses training
log odds and unlabeled query-state distances, retains each actual final RMS
denominator, and carries an explicit numerical execution allowance. It adds
20 proved theorems and concrete witnesses, with no new `sorry`.

The [fixed Rust diagnostic](experiments/probability_transfer/README.md) retains
all 24 original checkpoints and all five declared methods. All probability,
loss and gap bounds pass the numerical audit. At the plain-product endpoints,
the combined orbit envelope guarantees target probability above one half on
only 13/4705 and 43/4705 inputs (0.28%/0.91%), despite source accuracies of
99.66%/99.68%. The weaker argmax condition covers 18.21%/28.71%. Diagonal-arm
probability coverage is 48.20%/46.01%. Mean CE ceilings for the plain product
are 3.5043/3.2261 versus actual 0.01974/0.01872, so this bound is still too loose.

The next obligation is tighter training-derived control in the relevant
class-contrast directions. This is a conditional geometric result, not a
learning theorem or independent statistical test. No model training or
architecture change was made; complex multiplication remains unchanged.

## Geometry controlled by the objective (2026-11-08)

Task #181 asks which part of the required geometry follows from training loss.
The [Lean extension](review/flat_spectrum.md#loss-controlled-geometry) proves
that mean within-answer probability scatter is at most twice mean CE on the
same rows. It also gives an actual shared-scalar gradient-descent counterexample:
CE decreases while raw contrast scatter increases. No unconditional raw-state
or raw-contrast contraction is assumed, and no new `sorry` was added.

The [fixed Rust diagnostic](experiments/loss_geometry/README.md) retains all
24 original checkpoints. All numerical objective bounds and centering
identities pass. Among 16 adjacent training intervals, 12 lower CE and two
of those increase raw contrast scatter: both diagonal seeds, 5000 to 10000.
At the plain-product endpoints, training probability scatter is
3.428e-5/1.173e-6, while validation scatter is 0.005069/0.004630. Both validation
values exceed twice training CE, so the training inequality cannot simply be
reused as a held-out bound.

The next obligation is a transfer bound from training evidence to unseen
prediction probabilities, or sufficient learning conditions for the raw
contrast transport terms. Small training CE does not by itself provide either.
No model training, checkpoint changes or architecture changes were made.

## Training-state transport (2026-11-08)

Task #180 extends the theory to actual class contrasts using training states,
query states and an explicit reconstruction residual. The
[Lean derivation](review/flat_spectrum.md#training-state-transport) proves the
Euclidean baseline, weighted transport bound, valid per-competitor choice,
actual plain-product source bridge and cubic routes for transferring observed
training labels. It includes a counterexample to omitting the residual and a
positive nonzero-residual witness, with no new `sorry`.

The [fixed Rust diagnostic](experiments/contrast_transport/README.md) retains
all 24 original checkpoints. Bound construction has no query-label, query-logit
or target-error argument; a separate audit checks the frozen construction.
Best orbit conditions cover 991/4705 and 1406/4705 held-out inputs at the plain
product endpoints (21.06%/29.88%), with no incorrect positive cases. The
Euclidean training-center baseline already covers 989/1404, so covariance
weighting adds only two queries per endpoint. All initial states have zero
coverage; later coverage is not monotone with classifier accuracy.

This gives partial deterministic conditions from training evidence and query
geometry, not a learning theorem, independent statistical test or verified
floating-point certificate. Explaining how training establishes useful
contrast geometry, improving coverage and controlling execution error remain
open. No new training or architecture change was made; all earlier reports
remain intact.

## Combined class contrasts (2026-11-08)

Task #179 follows the user's approval to connect the theory to actual combined
class-score differences. The [Lean extension](review/flat_spectrum.md#combined-answer-contrasts)
proves answer pooling, cancellation, a sharp centered contrast bound and the
source execution bridge with separate competitor gaps and explicit alignment.
It includes a nonzero-error source witness and no new `sorry`.

The [fixed Rust diagnostic](experiments/answer_contrasts/README.md) retains all
24 original cubic checkpoints. At the plain-product endpoints, exact-sum probe
accuracy is 45.40%/46.97%, while pooled-answer accuracy is 99.53%/99.66%.
Measured combined budgets are positive on 99.30%/99.43% of held-out queries.
Independent quotient fitting equals pooling the full probe, as predicted by
linearity; this is not a second learned representation. The uniform half-range
L1 bound and weight-norm alignment bound both give zero positive held-out budgets.
These are retrospective conditions, not predictive generalization certificates.

The next open obligation is to bound the relevant error directions and source
alignment without measuring held-out target errors. No new model training,
architecture change or continuation cohort was opened. The earlier spectral
diagnostic and its negative result remain intact.

## Quantitative recovery and answer equivalence (2026-11-08)

Task #178 extends nonunique recovery to noisy fitting, finite nuclear-norm
objective excess, explicit readout conditioning and approximate source alignment.
The actual final RMS denominator stays in the execution budget. The user's
follow-up adds answer-fiber readouts and the exact cubic equivalence
`s^3=t^3 iff s=t or s=35t or s=61t` in `ZMod 97`, including a concrete
normalized Fourier witness. See [the derivation](review/flat_spectrum.md#approximate-nonunique-recovery).

The [Rust diagnostic](experiments/flat_approx/README.md) is complete: all 24
original cubic checkpoints, all arms and both seeds, no new training. Every
stored correctness count reproduces. The declared componentwise character-bank
spectral bound leaves zero positive held-out budgets, even at zero assumed
execution error. Direct measured contrast budgets are positive for most trained
queries. Plain-product endpoints remain 99.66% and 99.68%; this result supplies
no reason to change the block. The theory should next bound the actual combined
class contrasts while allowing answer-preserving feature ambiguity. The new
fiber theorems do not yet identify the trained code or prove optimizer discovery.

The current Rust/Lean-only override in `AGENTS.md` supersedes the older Python
execution instructions below. This read-only diagnostic neither reopens a
learning cohort nor spends any earlier training authorization again.

## Active research target and handoff (2026-11-02)

The current model recommendation is in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).
It retains the strongest completed baseline, the measured fixed-scale option,
and the separate experimental learned-gate extension.

The current user target is to connect flat spectrum, the actual normalized
RoPE transformer, and C-RASP through **efficient structured computation and
its learnability**. Preserve this target across sessions. Real transformers
reuse local computations; the exponential width of a universal table
interpolator is not a necessary width for arithmetic or natural tasks.
At fixed width each token still has a separate state: context supplies
distributed working memory. Attention scores select weighted earlier pieces,
and the feed-forward/residual blocks combine them over depth. Correct
selection must be proved or measured, not assumed from a high attention score.

User constraints: work solo; Python3.15/Torch2.15; CUDA max-autotune;
**Schedule-Free AdamW only** for new learning runs. Preserve all source
normalizations, exact geometric RoPE, XSA, SwiGLU and tied readout. Grokking
can occur very late: the 2000-update screen is an early endpoint, not an
impossibility test. The separately authorized **long phase has a two-hour
wall-time cap**, including compilation and evaluation; this is not a cap
on the earlier setup/screen. Profile before beginning that phase, using
Python3.15's sampling profiler where available and CUDA-aware PyTorch traces.

| Board task | Current status | Required outcome |
| --- | --- | --- |
| #154 fixed-scale continuation | Complete; self-reviewed at 3978d528 | Both reach 99% validation 1,000 updates after 23k, versus ordinary 32k/37k; final test 99.227%/98.711%; original reports and parents preserved |
| #155 remaining gradient paths and direct model repair | Complete: 48 cases; unit-scale pre-RMS recovery 0/4 native versus 4/4 corrected; direct GPT repair at7314dec | Independent scalar/FP64 derivatives, native forward equality, known-solution recovery and runtime; verified repair gets an actual diff_gpt regression and direct integration, with no new architecture flag |
| #156 remove legacy architecture flags | Complete: canonical scaled GPT atdf1e9f7; legacy residual/XSA/RoPE switches removed; migrated callers and diagram checks pass | Remove existing residual-mixing, legacy XSA and per-layer RoPE architecture switches; one current model, migrated callers/tests and updated site; historical variants live in Git |
| #157 existing Muon allocation failures | Queued, separate baseline regression | Reproduced on unchanged main; AdamW grokking work does not use Muon |
| #158 repeated gradient search loop | Done: synthetic mechanism verified; arithmetic3 wins/1 tie did not meet the promotion rule | Complete evidence and rejected actual-GPT patch at93ffa071; production stays df1e9f7; broader checks continue in160 and the next mechanism in161 |
| #159 batch-transfer profiling follow-up | Done; self-reviewed atb9f2a4a2; candidate rejected | Same2KB copy waits0.024ms idle versus6.745ms with queued work; exact CUDA state checks fail even between control repeats; no input-cache change retained |
| #160 conditional gains on Basis | Done; same-assistant review at23154349 | Large parity improves with gains in both seeds; depth and small recall/parity regress; canonical hard-recall seed1 reaches first99% at123,550; reserved tests opened only after closure |
| #161 conditional-gain transfer diagnosis | Done; same-assistant review atd652d62d | Explain slower length128 generalization despite fitting short lengths; distinguish saturation, optimizer width scaling and learned length dependence before a specific new repair |
| #162 matched original-GPTMini control | Closed; temperature candidate rejected under the new cost rule | Historical GPTMini solves one seed at12,200; temperature seeds have no exact validation sequences at24,400; actual later states at100,000/62,900 retained and final tests unopened |
| #164 powered-division and Basis confirmation | Queued after #162 for a viable isolated candidate | Match Schedule-Free controls for P97/P193, K1/K2/K5 and both seeds; verify K0 and zero-input edges separately; require Basis, direct flag-free GPT tests and same-assistant review before adoption |
| #165 intermediate gate trajectories | Deferred; untrained draft preserved | No depth repeats until an analytical prediction and cheap diagnostic justify them; apply the user's twice-best cost cap before any launch |
| #166 Schedule-Free direction diagnosis | Complete CPU evidence; no repair supported | Averaging opposes less than0.008% of fresh predicted descent in each stalled endpoint; common-cancellation hypothesis rejected without learning; first-order loss predictions alone do not diagnose grokking |
| #167 normalization and projection initialization | Analytical follow-up; no training queued | Derive finite-epsilon RMS scale identities and Schedule-Free parameter/decay transformations; require a cheap discriminating witness before changing depth-dependent initialization |
| #163 CUDA repeatability before caching | Queued after159 | Characterize unchanged-control drift and declare a new execution-comparison protocol before revisiting the rejected cache candidate |
| #153 local architecture site | Complete; self-reviewed at709b6f70; root server on127.0.0.1:8765 | Interactive source diagram, residual options, measured curves, config and training recipe; loopback server remains running after worktree cleanup |
| #138 source model and early mixed-arithmetic screen | Self-reviewed; all48 replayed; fresh old-model compiled count drift retained separately | All cases retained, final-checkpoint replay, exact split/source hashes, train/heldout/longer accuracy and baselines;2000 updates alone cannot rule out grokking |
| #140 local attention theory | Self-reviewed; seven theorems at730c0155 rechecked | Useful-set mass, weighted retrieval error and normalized head-width bounds; arbitrary rotary phases; distinguish a local constraint from a task lower bound |
| #142 training profile | Self-reviewed after57935e11; historical missing rates explicit; fresh retained logs and wide/deep profiles pass | Warmed W256D1/W128D4 Python sampling plus CUDA traces; separate startup, compilation, optimizer, kernels, synchronization and evaluation; retain profiler overhead |
| #141 long grokking phase | Self-reviewed; all10 completed100000 updates in6283.38s; full states and compiled follow-up rechecked | Freeze selected cases and checkpoint hashes before continuation; retain query/base/moments/averaging clock/sampler; target100000 absolute updates within two hours; save incomplete runs as resumable |
| #143 theory falsification ledger | Self-reviewed; complete for this cohort; valid spectral theorems retained | Seek reproducible counterexamples to current learning interpretations; record prediction, premises, outcome and correction; update Lean if a formal claim is false |
| #144 residual mechanism audit | Self-reviewed; all8 early/long pair diagnostics reproduce identically | Connect existing AttnRes/mHC proofs to fixed-checkpoint measurements of branch/residual norms, cancellation, depth readouts and gradients; compare early and long states |
| #147 numerical XSA correction | Self-reviewed;17 Lean theorems and fresh compiled checks pass; all6 division checkpoint reports reproduce identically | Preserve clipping/backward conventions; all failures retained; actual-source checks pass and original long trajectories stay frozen |
| #149 repeated-token length collisions | Backlog after#147; structural follow-up | Check guarded constant-prefix invariance across lengths, the P97 chain labels3 versus31 on repeated token1, and an explicit boundary-token repair; do not turn a rare conditional collision into an average-error explanation |
| #148 paired deep numerical correction | Backlog after#147; separate budget required | Compare legacy and guarded XSA with matched seeds, gradients and sufficiently long learning curves; numerical fidelity alone is not improved generalization |
| #145 residual training comparison | Complete; self-reviewed at 79a6f443; 16 cases retained, GPU phase closed at 7165.75s | Freeze matched seeds and sufficiently long budgets for ordinary residual, branch-scale control, source-faithful AttnRes and the separately labeled conditioned Q/K variant from#150; preserve source branches and disclose added parameters/compute |
| #146 normalized depth-attention verification | Self-reviewed; eight Lean theorems rechecked;21-case synthetic report reproduces identically | Prove Q/K score and sensitivity bounds with explicit gain/temperature assumptions; reuse diff_gpt; run a known-solution synthetic task and inspect gradients before full-model changes; commit both repositories |
| #150 depth-query coordinate repair | Self-reviewed; all30 cases complete and replay byte-identically;6/6 huge-query repairs versus0/6 controls | Preserve finite-epsilon Q/K outputs and complete Schedule-Free state under coordinate changes; prove identities and bounds, test all30 fresh/resumed cases, retain failures, then self-review |
| #151 normalization and residual gradient benchmark | Complete; 36 CPU cases, proved mechanisms and self-reviewed GPT regression | Known-solution synthetic tasks for gate placement around RMSNorm and depth-dependent gradient transport; independent derivatives, paired Schedule-Free recovery, held-out inputs, and diff_gpt regressions for verified repairs |
| #139 polynomial-resource construction | Backlog; core theory target | Route local arithmetic states and compose real source blocks; aim for fixed width at fixed prime with growing depth, otherwise explicit polynomial bounds; include precision and margins |
| #133 optimizer-to-recovery bridge | Backlog after#131 | Derive useful routing or separating feature recovery from actual sampled CE/Schedule-Free dynamics under witnessed conditions; retain alias/memorization counterexamples |
| #134 structured depth/width tradeoffs | Backlog after#131 | Task-specific bounds for exact nonperiodic RoPE; do not substitute arbitrary-table counting or periodic C-RASP lower bounds |
| #137 backend range/precision certificate | Existing queued task after#136 | Complete the captured invocation's numerical certificate; this single invocation does not prove modular learning |

The user explicitly requested self-review by the same assistant. The
[self-review record](review/mixed_arithmetic_self_review.md) retains checks,
findings and fresh replay discrepancies; it is not independent review.
The long phase is finished. Its two-hour budget must not be reused for
the separate deep numerical or residual cohorts.

**Current research policy, 2026-11-02:** the user's later instructions supersede
the open-ended "until99%" continuation policy: stop at twice the best comparable
run, diagnose the failure, and think through the next intervention before
spending more laptop time. A new run needs a written mathematical mechanism,
a falsifiable prediction, a competing explanation and a cheap diagnostic.
Declare the matched reference, both update and elapsed-time caps, evaluation
cadence and a bounded preparation allowance before launch. Stop at first99%
or either cost cap. Report compilation and diagnostic time separately rather
than hiding them as free overhead. If no comparable baseline exists, declare
a small baseline budget first. A cost rejection does not prove eventual
grokking impossible or establish a defect in the model.

For the temperature screen the matched Schedule-Free best is12,200 updates,
so the new update cap is24,400. The instruction arrived after the retained
100,000/62,900-update states had been trained; preserve those actual outcomes
and do not relabel them as checkpoints saved at the new cap. Training stopped
on receipt. Both runs had zero exact validation sequences at24,400. The later
seed1 state reaches81.25%, which remains below the99% target. Final tests stay
unopened. Task165's queued depth repeats are deferred. Task164 still requires
powered division and Basis, but only after a candidate survives this reasoning
and cost gate. No production architecture change is justified yet.

**Task154 authorization, 2026-11-02:** the user requested rechecking fixed
scaling in longer runs and specified "until 99%". Continue both original P97
scaled seeds from23000 updates and stop each at its first canonical validation
accuracy>=99%. This is a new target-stopped continuation, not a reopening of the
closed task145 budget. Preserve the original source states and report the
previously published test split as reused. The recipe is unchanged; the runner's
first-crossing stop replaces the original three-observation99.9% rule. Start with
a100000-update checkpoint horizon and retain complete states if it is insufficient.

Execution order: finish and audit the frozen early screen while completing
local retrieval lemmas; profile representative warmed updates; commit a
separate long-run protocol; run it under the two-hour cap; audit every saved
endpoint and update the empirical/theory comparison. Then use the evidence
for the polynomial construction and conditional learning bridge. Do not
silently retune an existing cohort or discard failed seeds. Any performance
change must preserve the algorithm and pass independent numerical checks;
an arithmetic or optimizer change needs a separately declared cohort.

Keep five statements separate: mathematical labels are well-defined;
weights can represent them; attention retrieves the required information;
training discovers the computation (possibly after long grokking); execution
preserves the answer margin. A flat whole-table kernel settles none of the
last four by itself. Existing spectral theorems are about specified feature
readouts and recovery objectives, not arbitrary transformer impossibility.
The purpose of the experiments is to find where the account fails, not to
collect confirmation. Candidate claims to challenge: flat final kernels
prevent transformer generalization; rank-one character availability guarantees
learning; task spectrum predicts depth/width benefit; an early plateau means
permanent failure. Each needs a declared endpoint and checked premises.
Exact nuclear-norm theorems are not contradicted merely because a transformer
uses a different computation or objective. Correct the inference at the point
where evidence fails, retaining the stronger proved statements when valid.

The user's residual hypothesis is now a separate competing explanation for
depth effects. The source uses unit-coefficient residual addition after both
Peri-RMSNorm branches. Existing `Residual`, `AttentionResiduals` and `MHC`
modules describe alternative mixing and its limits; none proves that a
particular residual rule optimizes arithmetic learning. First measure saved
states and continue the unchanged deeper model long enough. A subsequent
residual ablation must be a separately declared cohort, with its own budget,
source correspondence and multiple seeds. Do not silently replace the model
mid-trajectory or spend the two-hour phase budget twice.

Persistent locations: `experiments/modular_composition/` holds the frozen
screen, hypotheses, source/compile checks and analysis. Ignored runs are
linked to the root's `experiments/modular_composition/runs/`. Task agent: `mixed_arithmetic`; completed results are landed in the root.
Its temporary worktree is removed after the final audit. Board evidence belongs in `.kanban/artifacts/<task>/`.
The screen's execution log is `/tmp/modular-composition-screen.log` and is
copied into board evidence before cleanup. Source fixture CPU initialization,
logits and gradients are bitwise equal; CUDA errors are7.45e-8/3.50e-7.
Latest CPU-visible validation:446 lab tests and104 optimizer tests pass,
with13 and14 CUDA-dependent skips. Earlier CUDA runs retain two documented
ScheduleFreeMuon allocation failures; Schedule-Free AdamW passed. CPU-only
success does not erase that CUDA baseline.

**2026-11-01 mixed-arithmetic progression, requested by the user.** Active:
board #138 adds the exact normalized source RoPE model to the lab and tests
`a^3+ab mod P` against powered-division controls, then final-answer chains
`z <- z^3+b*z+c mod P`. The frozen 48-case protocol is in
[modular_composition](experiments/modular_composition/README.md), with
P17/97/193, approximate parameter-matched width/depth comparisons,
initialization and optimizer controls, full-table fitting controls and
final-only doubled-length evaluation. Every training case uses Schedule-Free
AdamW, Python3.15/Torch2.15 and CUDA max-autotune. No convergence or
task-specific depth lower bound follows from the local flat spectrum.
Board #139 tracks a polynomial-resource actual-source construction using
active depth. The user clarified that exponential whole-table interpolation
must not be interpreted as necessary width for structured tasks. Aim for
width independent of chain length at fixed prime, with explicit routing,
depth and precision costs; this remains a target, not an established result.
Tasks #133/#134 retain the optimizer and exact-RoPE lower-bound gaps.

**2026-10-31 sampled-gradient theory extension, requested by the user.**
The existing minibatch symmetry controls expose the next gap: a symmetric
full-table gradient does not imply a symmetric sampled update. The 25 theorems
in `Rotate90.Training.Sampling` now compute the actual tied embedding gradient
at a collapsed table from empirical label counts, prove its mean squared
row-separating signal, and carry a valid singleton through the source
Schedule-Free AdamW update. An explicit nonzero width-16, hidden-43 model at
every positive finite depth has zero full-table division embedding gradient,
yet the sampled update breaks aliases in both query and evaluation buffers.
The arithmetic example a=b=1 works at every natural exponent; the general
escape theorem covers every prime with a nontrivial subgroup and a positive
hidden coordinate. This is a local escape mechanism, not a convergence or
generalization theorem for random starts or partially collapsed checkpoints.
No new training run or optimizer change was needed; the completed controls
remain the empirical source. See `review/flat_spectrum.md`, *Sampled updates
can break aliases*, for the exact scope and the remaining learning gap.

**2026-10-31 completed learning-gap cycle, requested by the user.**
The [alias-escape study](experiments/rotate90_alias_escape/README.md) completed
50 Schedule-Free interventions with hypotheses recorded before training. The
actual full-table model preserves subgroup aliases when the exponent is coprime
to the subgroup order; 48 new Lean theorems carry this through the actual
Schedule-Free recurrence, evaluation averages, a log-cardinality loss floor,
and pointwise limits. No arbitrary-initialization convergence assertion survives
this counterexample. Exact full-table symmetry is not imposed on partial data.

The user's subsequent duration/size question produced an explicitly adaptive
[24-run follow-up](experiments/rotate90_alias_budget/README.md): matched width 16
restarts at 20000 updates and decay 1/.1/0, plus fresh width 64 controls at 5000.
The two P193 zero-decay runs rise from 95.81/98.39% to 99.74/99.989%; all six width 64
runs finish at 99.96--100%. Some width 16 starts still fail, including every P97
restart in this follow-up. Thus finite plateaus need not be permanent, and the
width effect concerns training/generalization despite sufficient proved capacity.
All 74 runs use Schedule-Free AdamW, Python 3.15/Torch 2.15, CUDA max-autotune;
every failure and split/provenance record is retained. Predicting random-start
basin selection and the separate P17 memorization failure remain open.

**2026-10-31 completed comparison cycle, requested by the user:** test the current
Rotate90 Peri-RMSNorm SwiGLU on `(a/b)^K mod P` across primes, widths, depths
and optimizer settings, using Python 3.15, PyTorch 2.15 and max-autotune
compilation. [The powered-division study](experiments/rotate90_power_learning/README.md)
defines matched half-table learning screens and full-table fitting controls.
Arithmetic uniqueness is proved in `Transformer.Rotate90.existsUnique_div_pow`;
`Transformer.Rotate90.Power.exists_weights_width16_any_depth` now proves actual
neural representability for every prime and natural K, including zero numerators,
at width 16, hidden size 43 and every positive finite depth. The user's next direction
is to compare the flat-spectrum account with what the model actually learns,
revise the theory where needed and test again. The first revision distinguishes
recoverable rank-one characters from a jointly separating answer code; its
fresh-seed retry and Schedule-Free controls are specified before training in
the study's `hypotheses.json`. Training outcomes will be reported with their
finite budgets and held-out accuracies, including failures of the new rule.
The user's subsequent request tests larger models (widths 256 and 512, up to
four blocks) with three seeds and matched smaller controls. All these new
runs use Schedule-Free AdamW exclusively, per the user's latest restriction;
prior completed AdamW records remain available for comparison. All 175 definitions
are complete (48 Schedule-Free), and all checkpoints are verified. All 27 larger
models on the main P97/P193 tasks finish above 99%; P17 still fits the training
table without strong held-out generalization at width 512. The exact character
separation and actual-readout margin arguments correct the scope of the spectral
explanation. The approximate 95%-energy rule passes 16 of 17 applicable cases;
its counterexample is retained, and no convergence theorem is claimed. See the
study README and portable results for the complete three-seed comparisons.
The completed Lean extension constructs actual SwiGLU matrices, retains every
RMSNorm, and allows arbitrary finite width/hidden padding. Thus the small-model
failures are compatible with sufficient capacity; the remaining gap concerns
optimization and generalization. Quantification over finite architectures avoids
assuming that a particular infinite-width or infinite-depth limit contains their
learned functions. See the note's *Actual network expressivity for every exponent*.

Updated on 2026-10-19 UTC. **Active, resumed at the user's request on 2026-10-19.
Current direction: [Landscape explanation of modular
arithmetic generalization](landscape_grokking_plan.md), requested on 2026-10-19:
connect the published Rotate90 network and analytic activation variants to the
Landscape framework, explain why an expressible scalar MLP solution is hard
to reach by training, then investigate encoders of numbers.** Matched controls
must distinguish fitting the full table from held-out generalization. The
[scalar MLP study](experiments/mod97_scalar_mlp/README.md) is the existing
baseline; its completed results remain below. The earlier MLP snapshot has 11
completed runs, one checkpoint at 25,000 updates and 24 not started; see
[the retained progress](experiments/mod97_representation_controls/progress.json).
Those raw ignored checkpoints are absent from the resumed workspace. Task
#74 completed all 36 cases afresh on the available GTX 1050 and locked runtime,
preserving the earlier partial snapshot as a separate cohort. The experiment
definitions, seeds, splits, optimizer and original budgets remain fixed.
All 36 cases pass exact final-checkpoint replay and source/split audits;
all twelve scalar-output cases also have initial/final Jacobian diagnostics.
The [complete findings](experiments/mod97_representation_controls/README.md)
include three one-hot/vector MSE successes without decay, three memorizing
CE failures without decay, and one of three CE successes with decay one.
The [tangent control](review/mod97_tangent.md) proves a frozen-quadratic GD
speed limit with a positive-mode, attainable-fit witness. It does not
identify the actual AdamW attraction or held-out generalization mechanism.

Previous direction: [Grokking: how it works and when it fails](grokking_plan.md),
rebuilt on 2026-10-08 at the user's request around three measured boundaries
(whether the rule is expressible at the model's size, memorization capacity and
the critical data size) on RASP and C-RASP tasks, on ordinary softmax
transformers with unchanged AdamW. The earlier [route list](grokking_lean_plan.md)
is closed and kept as the ledger of its results. The user changed
direction on 2026-10-07. The [Basis correctness and convex architecture](plan.md)
cycle is paused and unfinished in stages 3 and 4; its checkpoints and original
FLOP budgets are retained. Complete real-model Basis semantics and the explicit
tensor prototype remain verified in Lean, with the empirical recall failures
preserved. The earlier investigation below is
complete: **steps 0 to 6 and the requested QKNorm follow-up.** The investigation cycle
was started on 2026-10-04 at the user's request. On 2026-10-04 the user asked for
this plan: find out why sparsemax attention fails, and try to repair it, on
the basis benchmark. It replaces the plan of 2026-10-03 for the mod-193
normalizer pair, which the user had paused. That plan's forward/backward
intervention and its Lean step continue here as steps 3 and 6, and mod 193
returns in step 5 to confirm a repair. The steps run in order; each ends in a
commit, and this file marks it done. Detailed historical evidence remains in
[the synthetic trainer handoff](experiments/archive/synthetic_trainers/HANDOFF.md).

## Objective

Find what makes sparsemax attention fail where softmax succeeds, on
[`experiments/basis`](experiments/basis/README.md); repair it; prove in Lean
what the experiments support. A repair succeeds when sparsemax attention with
it passes the basis wherever softmax passes, and still attends sparsely.

## What is known

Sparsemax replaces softmax by the Euclidean projection of a row's scores onto
the simplex (arXiv:1602.02068v2, §2), whose weights are exactly zero
outside a support.

- Modular division ([`mod193_stability`](experiments/mod193_stability/README.md)):
  GPTMini, whose attention normalizes queries and keys (QKNorm) and removes
  from each position's output its own value (XSA), trained on 25% of the
  37,056 equations `x / y mod 193`. Under sparsemax it fits every training
  equation, memorizes for 206,000 updates and never generalizes: held-out
  accuracy ends at 9,465/27,792 = 34.06% and peaks at 35.48%. Under softmax
  it ends at 100% on both splits.
- Associative recall ([`mqar_sparsemax`](experiments/mqar_sparsemax/README.md)):
  a transformer of one head, scaled dot-product scores and no XSA. Sparsemax
  reached 99% in 11 of 12 archived runs, at lengths 64 to 256 and four rates;
  softmax in 2.
- Tiny Shakespeare ([`shakespeare_amsgradw`](experiments/shakespeare_amsgradw/README.md)):
  GPTMini again; best test cross-entropy 1.638935 ± 0.016796 under sparsemax
  against 1.625375 ± 0.001731 under softmax, over three seeds.
- Lean, commit `0cf0524` (`Transformer.GPTMini.Sparsemax`): when one visible
  score of a causal row exceeds every other by more than 1, the sparsemax row
  is that position's basis vector on a neighborhood
  (`sparseWeights_eventually_eq_basis`), so its derivative, and that of any
  loss of it, is zero (`sparseWeights_hasFDerivAt_zero`,
  `rowLoss_hasFDerivAt_zero`); a wrong route can be such a point, of positive
  loss (`wrong_route_positive_stationary_point`). These concern one row: a
  possible obstruction, not the cause. Lean also certifies the accuracy of
  all 37,056 final CPU prediction records of the mod-193 pair
  ([report](experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-mechanism-and-final-certificate.md)).

So sparsemax does not fail everywhere: it wins on recall without QKNorm and
XSA, nearly ties on language with them, and fails to generalize on modular
division with them. The basis tells tasks apart: depth, recall and parity,
which a transformer solves by different means, on two sizes of GPTMini, each
calibrated under softmax ([Found](experiments/basis/README.md#found), at
bd63e50). Its 30 runs take 20 minutes on 28 cores, the small model's 15 about
nine.

## Hypotheses

Steps 1 to 3 decide each by its criterion.

**H1. Locked routes.** A sparsemax row passes no gradient to the scores
outside its support, and none at all once a gap of more than 1 isolates one
position (the Lean results above). A route the task needs that starts outside
the support is found only through other rows. *Predicted:* in a failing run
the supports of fixed rows stop changing while the run is below target, and
on wrong recall answers the position holding the answer lies outside every
head's support. *Refuted if* the supports keep changing until the run ends,
or the answer's position is in a support on the wrong answers.

**H2. The starting scale.** QKNorm multiplies the cosine of a query and a key
by a learned e^α that starts at √(head width): 4 on the small model, about
5.7 on the large. A row's scores then spread about 1, whatever the
initialization, and sparsemax keeps few positions: were a row's 64 scores
independent standard normals, it would keep 3.3 on average, and one alone in
6% of rows. Scaled dot-product scores under the initialization at 0.02 spread
about 0.02² × width, 0.026 on the small model and 0.051 on the large; as
independent normals, a row of 64 keeps about 41 and 27. *Predicted:* at
update 0, QKNorm rows keep a few positions, and a smaller starting e^α or
scaled dot-product scores remove sparsemax's failures. *Refuted if* QKNorm
rows start wide, or neither change helps where sparsemax fails.

**H3. Counting needs ties.** Depth is definable by counting
(`Transformer.CRASP.definablePos_altPlusNeutral`), and parity of up to 16
bits can be read off the count of ones; a count is an average over
positions. A sparsemax row that keeps all T visible positions weighs each by
1/T plus its score's deviation from the row's mean, so averaging within a
relative error ε needs every score within ε/T of the mean; softmax needs
about ε, whatever T. At length 64 sparsemax needs ties 64 times finer.
*Predicted:* sparsemax fails or slows on depth and parity more than on
recall; where softmax passes, some heads average many positions nearly
uniformly, while under sparsemax the same layers keep few positions, or their
e^α falls far below its start. *Refuted if* sparsemax passes depth and parity
as softmax does.

**H4. XSA erases self-routes.** XSA subtracts from a head's output its
component along the position's own value. A sparsemax row whose support is
the position itself outputs exactly that value, so the head outputs zero
there, and with a gap above 1 the row passes no gradient to its query, keys
or values. Softmax gives one position all of a row's weight only in a row of
one position. *Predicted:* failing sparsemax runs have many such rows, and
dropping XSA helps sparsemax more than softmax. *Refuted if* such rows are
rare, or dropping XSA leaves sparsemax's outcome unchanged.

**H5. The recipe.** The basis recipes were set under softmax. A sparsemax
failure that a neighboring rate of `RATES` removes is the recipe's, and the
mechanism study leaves it.

Step 6 corrects H4's implementation-level wording: exact erasure requires
the self-value norm at least epsilon, and the locally-zero result uses a
strict norm bound above epsilon. A clipped nonzero value can survive a
strict self route; a Lean counterexample below makes that correction
explicit. It does not change the empirical rarity conclusion of step 2.

## 0. Run sparsemax on the basis

**Done on 2026-10-04.** [basis_sparsemax](experiments/basis_sparsemax/README.md)
defines the 60 benchmark runs and six short timing runs, with the existing
diagnostics recording training batch loss and head scales. Sparsemax's
projection, backward, model logits and every parameter gradient agree with
eager evaluation within rounding under full-graph compilation. Diagnostics
preserves one-thread compiled observation metrics and the complete training
state under both weights, and a one-thread sparsemax run resumes onto its
own records. On large depth and recall at two and four threads, measurements
leave the actual training state and their inputs exactly unchanged; separate
large multi-threaded runs can round differently even without diagnostics,
so exact trajectory repetition is not guaranteed there. The measured
sparsemax/softmax update-time ratios are
1.681 (depth), 1.708 (recall) and 1.238 (parity), below the factor-of-two
ceiling; no implementation repair was needed. The short softmax controls
repeat every common archived basis validation record exactly; step 1 checks
the full trajectories. The small model's 15 runs under each weight come
first, then the large model's, then neighboring rates for failures.

Write `experiments/basis_sparsemax` (`experiment.py`, `README.md` and a row
in `experiments/README.md`). Its runs are the basis's 30 benchmark runs under
each of the two weights, labeled
`<weights>-<mode>-<model>-<task>-seed<seed>`: `softmax` repeats the basis,
and `sparsemax` is `basis(substitute(model, Softmax, Sparsemax()), mode,
seed)`, both on the basis's thread counts. Both record `Diagnostics` at
every observation: the training batch's loss, which tells a failure to fit
from mod 193's failure to generalize, and the e^α of every head. First
check:

- that `Sparsemax`, a sort, a cumulative sum and a custom backward, compiles
  under `Compiled`, and that compiled and eager sparsemax give the same
  logits and gradients to rounding (a test in `python/tests`);
- that diagnostics leave a compiled run's records unchanged, and that the
  softmax runs repeat bd63e50's; if not, the two arms compare at this commit
  alone;
- the time of an update of each small task under both weights. If sparsemax
  does not compile, or costs more than twice softmax, fix that in the lab
  first.

## 1. Find where sparsemax fails

**Done on 2026-10-04.** All 60 original runs and 12 required adjacent-rate
checks are complete. The 15 small softmax controls repeat all 661 archived
non-timing observations and their model, optimizer and sampler checkpoints
exactly. Fourteen of the 15 large controls' full histories differ, with the
same pass/fail set, so the large comparison uses the current arms as step 0
allows. Separate large runs without diagnostics also drift; direct tests
verify that measurements leave the actual training state unchanged.

H5 removes both small parity failures at 1e-4, small easy recall seed 2 at
3e-4, and large easy recall seed 2 at either 3e-4 or 3e-3. There is no
persistent matched-seed small-model failure. On the large model, sparsemax
passes every depth and parity; hard depth passes earlier than softmax,
parity later. Only `(hard, large, recall)` remains a mechanism target:
softmax passes seeds 1 and 2 while sparsemax's best accuracies at the recipe
are 98.63% and 97.66%, and neither adjacent rate passes either seed. Step 2
measures this cell from all three seeds under both weights. The small-model
target ablation is empty. Tables, final training batch losses, head scales
and source hashes are in [basis_sparsemax](experiments/basis_sparsemax/README.md).

Train the small model's 15 runs under both weights, then the large model's
15. Record, as the basis's Found does, the update of each pass or the best
selection accuracy, and the last training loss. Where sparsemax fails from a
seed that softmax passes from, train that seed at the two rates of `RATES`
beside the recipe's (H5).

Predicted before the runs: by H3, depth and parity fail or slow; by
`mqar_sparsemax`, recall passes at least as often as under softmax, unless
H2 or H4 interferes.

The cells (mode, model, task) where sparsemax fails at every rate tried and
softmax passes are the targets of steps 2 to 4. If there are none, the basis
does not show the failure: record that, move step 2's measurement to mod 193,
and keep the basis to check that a repair costs nothing there.

## 2. Measure the mechanism

**Done on 2026-10-04 UTC.** The target is large hard recall.
`AttentionDiagnostics` retains fixed validation examples and previous
causal support masks across checkpoints. Focused tests check hand-made
statistics, latest-write query routes, CPU/CUDA live-state preservation,
compiler guards, one-thread compiled noninterference and resumed turnover.
The 24 initial QKNorm/scaled-dot measurements cover both sizes and weights
from three seeds; the six target runs retain every canonical measurement.
All 155 lab tests pass. The 24 initial runs are complete: first-layer
query support averages 3.43/3.34 positions under QKNorm on the small/large
models, against 35.47/24.88 under ScaledDot, with otherwise identical
initial parameters. This supports H2's initialization prediction, not
its causal or repair conclusion. The six target repeats are complete:
softmax passes all seeds, sparsemax none. The measured sparsemax trajectories
also fit poorly, unlike step 1's fitted near-passes; separate threaded runs
diverge despite identical initial metrics. Direct real-shape, four-thread
checks preserve the entire live training state around attention observation.
H1 is refuted by continued turnover through the budget and by answer
membership on 90--97% of final wrong queries. H4 is refuted by rare self-only
routes, excluding BOS (at most 1.87% averaged over layer/head pairs at any
observation). H3 has no persistent depth/parity failure after H5; parity
can still take longer. H2's causal prediction awaits the score/scale
interventions. There is no small-model target for the small factorial
ablation. Detailed trajectories, per-head statistics, qualifications and
source hashes are in [basis_sparsemax](experiments/basis_sparsemax/README.md).

Add to `Diagnostics` a measurement of attention (domain, infrastructure,
test). At every observation, an uncompiled forward of 256 fixed validation
rows, which draws no random numbers, records per layer and head:

- the size of the support, in positions and as a share of the visible ones;
- the share of rows that keep one position, and of rows that keep only
  their own;
- the share of rows with a gap above 1, which are locally constant;
- the turnover: the share of (row, position) pairs whose membership in the
  support changed since the previous observation;
- under softmax, the same for the sparsemax of its scores, and the
  normalized entropy of its weights;
- on recall, at each query, the weight on the position holding the answer,
  the largest over the layer's heads, and whether that position is in any
  support.

Tests check each statistic on hand-made rows, and that a run with the
measurement records the same losses as one without. Then:

1. At update 0, measure both models under QKNorm and under scaled
   dot-product scores (H2's estimate).
2. Train the target cells under both weights from seeds 0 to 2 with the
   measurement.
3. On the small model's target cells, train {QKNorm, ScaledDot} ×
   {XSA, none} × {softmax, sparsemax} from seeds 0 to 2 (H2, H4).

Decide each hypothesis by its criterion, and write what was found in the
experiment's README.

## 3. Separate forward routing from backward sensitivity

**Done on 2026-10-05 UTC.** `SurrogateWeights` and six mixed target
labels are implemented. Identical pairs use the ordinary block; mixed pairs
retain exact forward probabilities and ask autograd for the other map's
score Jacobian. Tests cover bit-identical diagonal logits and gradients,
independent `torch.func.vjp` references, inactive scores and compilation.
All six focused tests pass, as do all 161 lab tests. The previous 156 run
descriptions remain identical; the six mixed targets differ only in the
explicit forward/backward weights block. All six mixed targets completed
4,800 updates. Sparse forward with softmax backward fails from every seed
(best sequence accuracy 0%); softmax forward with sparse backward learns
much more but also misses the target (96.88%, 96.29%, 98.63%). Both ordinary
and surrogate backward fail with sparse routing, so the forward map remains
binding in this intervention; this does not confirm H3's counting or H4's
self-erasure mechanism, already refuted by their criteria. The inverse
surrogate's failures also leave a role for backward sensitivity or an
unsuitable surrogate. No gradient-only repair is found. The conditional
inactive-score-only case is not triggered: softmax backward does not rescue
sparse forward. H2's initialization prediction remains the next justified
repair test. Results, per-head statistics and source hashes are in
[basis_sparsemax](experiments/basis_sparsemax/README.md).

The intervention planned for mod 193, moved onto the basis:

| Forward weights | Backward score map | Purpose |
| --- | --- | --- |
| Softmax | Softmax Jacobian | Control |
| Sparsemax | Sparsemax support Jacobian | Candidate |
| Sparsemax | Softmax Jacobian | Keep sparse routing; change score gradients |
| Softmax | Sparsemax support Jacobian | Keep dense routing; change score gradients |

A new `Weights` block computes one normalizer forward and the other's
Jacobian backward, at the same scores: a declared surrogate gradient, not
the derivative of the loss. CPU tests: with both normalizers the same, it
equals the plain block bit for bit, logits and every gradient; mixed, its
weights are those of its forward normalizer, and its backward equals an
independent vector-Jacobian product of its backward normalizer
(`torch.func.vjp`), inactive entries included. Train the two mixed cases on
the target cells from seeds 0 to 2; steps 1 and 2 hold the diagonal ones.

If sparse forward with the softmax backward passes where sparsemax fails,
and dense forward with the sparsemax backward fails where softmax passes,
the backward map is implicated (H1). The swap changes both the support and
the size of score gradients; a focused case, the sparsemax backward with
softmax's gradients for the inactive scores alone, attributes the effect to
their zeros. If sparse forward fails under either
backward, the forward is implicated (H3, H4). Other patterns indicate an
interaction or an unsuitable surrogate.

## 4. Repair

**Done on 2026-10-05 UTC.** H2 justifies the two score interventions:
QKNorm initialized at scale one and ScaledDot, each with the ordinary
sparsemax backward and unchanged basis recipes. `QKNorm.initial_scale`
defaults to the original square-root-of-head-width initialization and keeps
the scale learned. Five focused tests pass: validation, old-description
compatibility, original rounding, identical non-score parameters and score
gradients with a learned scale. Sixty repair labels cover both candidates
on the small model's 15 runs and then the large model's 15, with fixed-row
attention measurements. All 166 lab tests pass. All 30 small-model runs are
complete: each candidate matches 6 of the 9 softmax passes, neither passes
the six hard runs, and every passing run retains exact visible zeros in
all eight heads after BOS. QKNorm-one fails parity seeds 0 and 2 and easy
recall seed 2; ScaledDot fails every easy recall seed. Thus neither is a
complete basis repair at these recipes. All 30 large-model runs are now
complete too: QKNorm-one matches 11 of 13 required large softmax passes,
ScaledDot 6 of 13. QKNorm-one passes the original persistent hard-recall
seed 2 at update 3,550 but not seed 1, and fails large parity seed 2.
ScaledDot fails every large parity and hard-recall seed. Every final head
of all 60 runs retains exact visible zeros after BOS. Across both sizes,
QKNorm-one matches 17 of 22 required softmax passes and has 19 total passes;
ScaledDot matches 12 of 22 and has 13 total passes. Neither is a complete
repair. QKNorm-one is the best attempted intervention for step 5, ranked
by sparse matched softmax passes, then persistent target passes, then all
sparse passes. H2's narrow-start prediction holds and changing the start
helps one persistent seed, but its failure-removal prediction is not met.
The complete source/description/cadence and 300 raw-file hash checks pass,
as do all 166 lab tests (938.589 seconds).
The matched-seed table, final losses, full per-head supports and source
hashes are in [basis_sparsemax](experiments/basis_sparsemax/README.md).
The failed surrogate is not screened as a repair; H4's rarity criterion and
H3's lack of persistent failures do not justify no-XSA or entmax trials here.

Try the candidates whose hypothesis survived, each a block or a `swap` or
`substitute` variant:

- (H2) QKNorm starting at e^α = 1, a new field of `QKNorm`; scaled
  dot-product scores;
- (H4) no XSA;
- (H1, H3) α-entmax at α = 1.5 (arXiv:1905.05702v2, §3.2 to 3.4): still
  exact zeros, but its support reaches 2 below the top score, not 1, and
  averaging within ε needs scores within about ε/√T of the mean, not ε/T;
  or α = 1 + sigmoid(a) learned per head, in (1, 2) (arXiv:1909.00015v2,
  §3 and 4), so that a head that counts can stay dense; there the heads
  first grew denser, and some sparser later (§5.1);
- (H1) sparse forward with the softmax backward, from step 3.

Screen each on the small model's 15 runs: in the easy mode a failure counts
against it, in the hard mode a pass counts for it. Then the large model's 15.
A repair succeeds if it passes every cell from every seed that softmax
passes from. Record its updates to pass beside softmax's, seed by seed, and
how sparse its attention ends: the mean support and the share of exact
zeros, per layer and head. A repair that passes only by making its rows
dense explains the failure; it does not repair sparse attention.

### QKNorm follow-up requested during step 5

On 2026-10-05 the user asked whether sparsemax needs QKNorm and requested
continued investigation. The completed ScaledDot screen removed both L2
normalization and the learned gain, and changed the initial score scale.
A supplementary CPU study, [basis_qknorm](experiments/basis_qknorm/README.md),
therefore removes L2 normalization while keeping a learned head gain.
It compares gain one against the initialization-only gain estimate
`1 / (width * init_std^2 * sqrt(head width))`, with fresh QKNorm-one,
ScaledDot and softmax controls in the large hard-recall target cell.
The estimated scale is fixed before measurements; actual initial score
dispersion and support must be measured, not assumed equal. All 150
descriptions check and five focused tests pass. Thirty initial models
preserve every non-gain parameter and the observer's live state; all 90
control descriptions equal their originals with the same observer. Six
actual-shape pairs preserve ScaledDot's initial logits and every non-gain
gradient exactly at gain one, while the gain receives nonzero gradients.
On first-layer queries, the estimated-gain raw-dot rows keep 8.70/11.00
positions on the small/large model, against QKNorm-one's 8.58/10.76,
with similar measured score dispersion. These are initialization checks,
not evidence that removing normalization repairs training. Their full
observations and source hashes are retained in
[preparation.json](experiments/basis_qknorm/preparation.json).

The full CPU lab gate checks 171 tests in 1172.811 seconds: 165 pass and six
CUDA-only checks are skipped while the GPU runs mod193. This follow-up starts
with independent derivative/compilation and initialization checks and
that full lab gate, then trains both new arms
and fresh softmax controls on the small model's 15 runs before the large
model's 15, plus the six QKNorm-one/ScaledDot target controls. All 96
runs use the same lab sources. It collects the same pass, batch-loss
and per-head sparsity evidence as step 4; untrained declared controls
support no conclusion. Preparation and runs
use a separate checkout until the pinned step-5 series ends. The existing
mod-193 jobs continue, and steps 5 and 6 remain required in their original
order. This supplements the completed screen rather than changing its
criteria or calling either old intervention a repair.

All 45 small follow-up runs are complete. Fresh softmax passes nine cells;
both learned raw-dot starts match six, with visible exact zeros in all
120 final non-BOS heads per arm. Gain one passes all three parity seeds
but loses all three easy-recall seeds. The estimated starting gain passes
two recall seeds but loses two parity seeds; recall seed 0 reaches 98.44%
without passing. Neither is a complete small-model repair. All 51 large
runs have now finished with unchanged sources and budgets. The full small reduction,
per-head statistics, gains, pass comparisons and raw file hashes are in
[small_results.json](experiments/basis_qknorm/small_results.json).
All 15 fresh small softmax controls repeat step 1's 661 canonical
observations and every non-timing model/optimizer/sampler/best checkpoint
field exactly, excluding the new attention-observer state. All 225 raw
reduction hashes verify; the independent comparison is retained in
[small_control_repetition.json](experiments/basis_qknorm/small_control_repetition.json).
The findings' required `./run.py test` gate checks 171 tests in 818.584
seconds: 165 pass and the same six CUDA-only checks are skipped. Runtime
and experiment sources remained unchanged during the large runs.

The complete 96-run screen is done. Each new arm matches 6 of 14 large
softmax passes and 12 of 23 across both sizes. All 360 final large-model
non-BOS heads per arm retain visible exact zeros. Neither new arm passes
large hard recall from any seed; current softmax passes 3/3, QKNorm-one
2/3 and ScaledDot 1/3. The earlier step-4 global counts use different
multi-threaded trajectories; only the three current target seeds compare
all five arms. There is no better candidate warranting another mod-193
transfer. [results.json](experiments/basis_qknorm/results.json) verifies
480 raw hashes and all runtime/source/description/fixed-row pins and links
the detailed large and small reductions. Before the complete-findings
commit, all 171 tests are checked again (808.529 seconds): 165 pass and six
CUDA-only tests are skipped. Steps 5 and 6 retain their original scope.
The source-validation mismatch after the completed small training was
corrected to follow Lab's 78-file runtime provenance, excluding its three
command-line files; all completed small sources match and none is retrained.
After all four pinned mod193 runs finished, the complete study was merged
into the main workspace. The full CPU/CUDA lab gate then passed all 171
tests without skips (911.172 seconds). Raw study runs are retained in the
main workspace's ignored `experiments/basis_qknorm/runs/` directory.

## 5. Confirm on mod 193

**Done on 2026-10-05 UTC.** Step 4 selects QKNorm-one as the best
partial attempt; neither screened intervention passes the complete basis
repair criterion. `mod193_stability` declares `repair-qknorm-one` beside
fresh `base` and `sparsemax` controls, with the same ordinary backward,
recipes and seeds, and fixed attention on 256 held-out examples at every
observation. A separate seed-1 label changes both model and data seeds and
is trained only if the first attempt confirms. The seven run descriptions
and CUDA live-state preservation are checked before the full training.
Confirmation uses 20 consecutive canonical evaluations with both
accuracies at least 99% and ends at 100% on both splits; neighbor probes
are excluded. Memorization and the 201 final-window checks remain separate
report columns, not extra confirmation requirements. All seven
descriptions check and the three actual-shape CUDA observer checks pass:
parameters, buffers, gradients, optimizer, sampler, CPU/CUDA RNG, modes,
hooks and inputs remain exactly unchanged at updates 0, 10 and 20. These
temporary short runs are setup checks, not generalization evidence.
Before/after descriptions and the check's source/output are retained in
[preparation.json](experiments/mod193_stability/preparation.json).
The full lab check passes all 166 tests (936.095 seconds) before the setup
commits and full runs.

All four scheduled lab runs completed 300,000 updates at the pinned sources.
The fresh softmax and ordinary sparsemax controls repeat all 1,201 archived
canonical observations exactly, excluding timing. Softmax ends at 100% on
both splits, with two final-window failures; ordinary sparsemax ends at
100%/34.06%, never confirms and fails all 201 final-window checks. QKNorm-one
confirms at update 121,000, ends at 100%/100%, and passes all 201 final-window
checks (worst held-out 99.62%). Its fresh model/data-seed-1 repeat confirms
at 186,250 and ends at 100%/100%; two final-window held-out checks fall below
99% (worst 98.93%). Both meet the original step-5 confirmation criterion,
but strict final-window persistence holds only for the first seed. Their
eight final heads retain visible exact zeros, averaging 36.52% and 46.56%
of visible non-BOS pairs respectively. This confirms the selected partial
intervention on mod193 from two seeds; it does not repair the complete basis
or establish general success or persistence on unseen seeds.

[confirmation_results.json](experiments/mod193_stability/confirmation_results.json)
retains all canonical/attention curves, initial/final per-head statistics,
component losses, gains, 32 raw file hashes and embedded reduction sources.
Each run has 1,201 canonical evaluations, 2,400 neighbor probes, 3,601
fixed-256-example attention records and 300,000 verified finite gradient
updates. [control_repetition.json](experiments/mod193_stability/control_repetition.json)
independently verifies both archived histories, all four canonical reductions,
matched descriptions, fresh seeds and all 78 runtime source hashes.
The complete-findings `./run.py test` gate passes all 166 tests, including
CUDA checks, in 886.968 seconds.

Train the best repair as a label of `experiments/mod193_stability` beside
`sparsemax`, changing nothing else, with step 2's measurement, for its
300,000 updates; the archived pair took 2.6 and 3.5 hours on the GPU. The
original table held archived runs of the historical trainers, so train
`base` and `sparsemax` afresh too. Record the archived
table's columns: memorized, confirmed, failures in the last 50,000 updates
and the worst held-out accuracy there, final accuracies. The repair is
confirmed if it generalizes as `base` does: confirmed for 20 evaluations,
ending at 100% on both splits. One seed cannot establish general
applicability: repeat a confirmed repair from fresh model and data seeds
before claiming it.

A preliminary CPU inspection of the archived pair's final checkpoints has
finished; its observations are under
`experiments/archive/synthetic_trainers/runs/adamw_stability_20261002/bootstrap/sparsemax-generalization-inspection-20261003/`
(not tracked). The inspector that wrote them is in commit `441f47b`, at
`experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/inspect_sparsemax_generalization.py`,
and left the tree with the other archive scripts; its SHA256 is the
`program_sha256` the observations record. It imports the removed synthetic
trainers and reads `experiments/runs/`, so it does not run as it is. Review
the source, independently check the observations, and archive them before
treating them as final.

This review is complete. The exact source from `441f47b`, original
observations, independent verification source/output and input hashes are
archived in [sparsemax-final-review](experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-final-review/README.md).
The independent program rebuilt every equation and strictly loaded the
unchanged final parameters in the lab at `d03f594`; all four fixed-weight
forward swaps and the selected local derivative probes match exactly,
with no rounding differences or optimizer updates. Both raw checkpoint
hashes and every embedded lab source hash match their pinned inputs.
This establishes the fixed-checkpoint observations, not a training cause
or the result of the new 300,000-update lab trials.

## 6. Prove the supported statements in Lean

**Done on 2026-10-05 UTC.** Five modules beside
`Transformer.GPTMini.Sparsemax` add 25 proved theorems, with examples for
every explicit hypothesis and no new `sorry`:

- [ClosedForm.lean](src/Transformer/GPTMini/Sparsemax/ClosedForm.lean)
  proves a normalized clipped candidate minimizes the existing variational
  objective, proves a threshold exists, and identifies the actual row with it.
- [SupportWindow.lean](src/Transformer/GPTMini/Sparsemax/SupportWindow.lean)
  proves H2's strict unit support window and the `exp(-alpha)` window for
  QKNorm's normalized similarities, including epsilon clipping.
- [Uniform.lean](src/Transformer/GPTMini/Sparsemax/Uniform.lean) proves H3's
  full-support weight formula, its exact relative-uniformity/score-tie
  equivalence at `epsilon/N`, and the resulting attention average.
- [SelfRoute.lean](src/Transformer/GPTMini/Sparsemax/SelfRoute.lean) proves
  H4's corrected locally-zero output and score/value/parameter derivative
  with strict score and epsilon-norm bounds; continuity of parameter maps
  is an explicit premise.
- [Clipping.lean](src/Transformer/GPTMini/Sparsemax/Clipping.lean) proves
  the residual below epsilon and a strict self-route counterexample:
  self-value norm `1/2`, epsilon `1`, output norm `3/8`.

Sources were reread beside the statements: arXiv:1602.02068v2, §2.2,
Proposition 1, and arXiv:2603.09078v1, §2, equation `xsa` and Algorithm 1,
plus the epsilon-clipped lab code at `73f8a0b`. All modules are reachable
from `src/Transformer.lean`. The full `lake build`, `./run.py audit`,
`./run.py index` and `./run.py forbidden` checks pass: 157 existing
`sorry`, zero resting on them, zero extra axioms, zero vacuous statements
and zero placeholders. The generated index has 1,836 modules and 7,556
theorems. New modules have no warnings. H1's intervention did not isolate
a whole-training cause, so no such theorem is claimed.

Prove only what the experiments support, under AGENTS.md: statements checked
against their sources, an example for every theorem with hypotheses, no axiom
or new `sorry`; then `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden`. Distinguish a real-arithmetic theorem, a
floating-point measurement and a certificate of supplied data; a theorem
about one row says nothing about a whole training run. Candidates, beside
`Transformer.GPTMini.Sparsemax`:

- (H2) every position of a sparsemax row's support scores within 1 of the
  row's maximum, by the closed form of arXiv:1602.02068v2, §2.2, so under
  QKNorm its cosine is within e^(−α) of the largest;
- (H3) with a full support the weights are 1/T plus each score's deviation
  from the mean, so a near-uniform average needs ties T times finer than
  under softmax;
- (H4) a row whose support is its own position, by a gap above 1, has an XSA
  output of zero on a neighborhood, hence zero derivative in its query, keys
  and values (from `sparseWeights_eventually_eq_basis`);
- (H1) what the intervention attributes.

## Cycle outcome

The prescribed cycle and the QKNorm follow-up are complete. The selected
initial-scale intervention confirms sparse Mod193 generalization on two
fresh model/data seeds, but the complete-basis repair criterion remains
unmet: QKNorm-one matched 17/22 step-4 softmax passes; each learned raw-dot
start matched 12/23 contemporary follow-up passes. Final-window persistence
holds for the first Mod193 repair seed and fails twice for the repeat.
These are qualified experimental conclusions; the row-level Lean results
do not prove a global training cause. The completion manifest is retained in
[cycle_audit.json](experiments/basis_sparsemax/cycle_audit.json).

## Mathematical follow-up: supervised score loss

**Done on 2026-10-05 UTC.** The loss in arXiv:1602.02068v2, §3.2,
`sparsemax_loss`, depends directly on the target score as well as on the
projection's quadratic potential. Its score gradient is `sparsemax(z) - y`
for a supplied routing target `y`. This objective can supply a corrective
signal on a wrong singleton even though every outer function of that
locally constant probability row has zero score derivative.

[RoutingLoss.lean](src/Transformer/GPTMini/Sparsemax/RoutingLoss.lean)
uses the existing causal variational projection. It proves the potential
identity, feasible-route lower bound, visible-target nonnegativity, and
full score derivative on strict singleton regions. Adding the score loss
to any outer probability loss supplies the same derivative multiplied by
its weight. On the previous counterexample `(2, 0)` with target slot 1,
the target-coordinate derivative is `-1`; a score-gradient step of size
`3/2` produces `(1/2, 3/2)`, the exact correct route and zero score loss.
No new `sorry` is used.

This repairs the supervised row counterexample, not the complete basis.
Attention routes are latent: the experiment's output labels do not directly
specify the desired position in every head. An auxiliary routing loss would
need supplied targets, a declared teacher or another source of score
supervision. No such training intervention has been run, and the previous
benchmark findings and completed-cycle manifest remain unchanged.

## Mathematical follow-up: constraints without routing targets

**Done on 2026-10-05 UTC.** Five modules add 34 proved theorems about
the existing causal variational projection, with inhabited examples for
every explicit hypothesis and no new `sorry`. These are derived results
from arXiv:1602.02068v2, §2.2 and §2.5, and the score/value code at
`73f8a0b`; they are not results of a new training intervention.

| Module | Constraint or result | Scope |
| --- | --- | --- |
| [NonSaturation.lean](src/Transformer/GPTMini/Sparsemax/NonSaturation.lean) | A singleton occurs exactly at a visible winner's unit gap; a top-two gap below one ensures two positive weights. | Exact zeros at other visible slots remain possible: `(2/5, 2/5, -2/5)` projects to `(1/2, 1/2, 0)`. |
| [ActiveDirection.lean](src/Transformer/GPTMini/Sparsemax/ActiveDirection.lean) | Transferring scores between two active slots transfers the same probability mass for small steps. | The score-to-weight map cannot have a zero full derivative; this does not assert existence of a full derivative at support boundaries. |
| [BoundedGain.lean](src/Transformer/GPTMini/Sparsemax/BoundedGain.lean) | QKNorm's persistent gain `g < 1/2` suffices; the new law `g = c/(1 + exp(-a))`, `0 < c < 1/2`, enforces it and has a positive derivative at every finite `a`. | Epsilon is nonnegative and at least two slots are visible. The guarantee concerns raw score derivatives, not query/key/gain parameter derivatives. |
| [OuterSensitivity.lean](src/Transformer/GPTMini/Sparsemax/OuterSensitivity.lean) | A task derivative distinguishes an active pair exactly when `output_gradient(value_j - value_k)` is nonzero. | This uses the ordinary task loss, with frozen values. Equal values cancel; distinct values can also lie in the gradient's kernel. |
| [ValuePlateau.lean](src/Transformer/GPTMini/Sparsemax/ValuePlateau.lean) | With values `(0, 0, 1)` and scalar output target `1/2`, squared error stays `1/4` on an open score neighborhood of the sparse bounded example. | Both this positive stationary point and a zero-error alternative obey `abs(score) <= 2/5`. The projection itself still has a nonzero active direction. |

The gain restriction removes singleton saturation without a teacher or
attention-position labels. It does not remove every flat outer loss or
prove convergence on Shakespeare or the basis. The value condition is a
conditional guarantee, not a rule proved enforceable for every task input.
No Python implementation or training run of the bounded gain has been
added. The original cycle remains complete with its qualified negative
complete-basis outcome and its pinned manifest.

Validation: the full `lake build`, `./run.py audit`, `./run.py index`
and `./run.py forbidden` pass. The generated index has 1,842 modules,
7,599 theorems and the same 157 existing `sorry`; there are zero results
resting on them, zero extra axioms, zero vacuous statements and zero
placeholders. The five new Lean modules have no warnings.
The required `./run.py test` also passes all 171 Python tests.

## Pure sparse follow-up: active value span

**Done on 2026-10-05 UTC.** Two further modules add ten proved theorems
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the frozen linear value readout at `73f8a0b`.

[ValueSpan.lean](src/Transformer/GPTMini/Sparsemax/ValueSpan.lean)
proves a structural sufficient condition independent of the task target:
the differences of active values span the output space. Their span then
cannot lie in the kernel of a nonzero output derivative, so some active
pair supplies a nonzero raw score direction for the ordinary task loss.
Two distinct active scalars satisfy the condition. The earlier collapsed
values `(0, 0, 1)` on the actual two-active row are proved to violate it.

[SeparatedValues.lean](src/Transformer/GPTMini/Sparsemax/SeparatedValues.lean)
computes the actual derivative for ordinary scalar squared output error:
`2 * (output - target) * (value_j - value_k)`. A wrong scalar output and
distinct active values therefore exclude a zero score derivative. The
constructed values `(-4, 4, 1)` retain the previous initial output, target
`1/2`, scores `(2/5, 2/5, -2/5)` and weights `(1/2, 1/2, 0)`. The
active-pair derivative is `-8`; transferring scores by `1/16` gives
weights `(7/16, 9/16, 0)` and zero output error. Both score endpoints
stay strictly below the same absolute cap `12/25 < 1/2`.

This repairs the constructed output plateau after changing the values to
satisfy the separation premise. It does not repair the collapsed values
through a locally unchanged support, supply an enforcement rule for the
span in every head/input, prove query/key parameter accessibility of the
score direction, or establish whole-model convergence. The ordinary
output target is retained; no attention-position labels are introduced.
These additions are Lean results only. The Python code and previous
experimental outcomes are unchanged.

Validation: the full `lake build`, `./run.py audit`, `./run.py index`
and `./run.py forbidden` pass. The new modules have no warnings. The
index now has 1,844 modules and 7,609 theorems, with the same 157 existing
`sorry`, zero results resting on them, zero extra axioms, zero vacuous
statements and zero placeholders. Python is unchanged since the passing
171-test run recorded above.

## Pure sparse follow-up: enforcing the value span

**Done on 2026-10-05 UTC.** Three modules add sixteen proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the frozen linear value readout at `73f8a0b`.

[BoundedCoordinates.lean](src/Transformer/GPTMini/Sparsemax/BoundedCoordinates.lean)
proves strict bounds, a positive derivative and a differentiable inverse
for independent score coordinates `c * (exp(a) - 1) / (exp(a) + 1)`.
These modify the scores; the weights remain the actual causal variational
sparsemax projection.

[AnchoredScores.lean](src/Transformer/GPTMini/Sparsemax/AnchoredScores.lean)
prepends `A` bounded anchor scores to ordinary scores below `-c`.
If `c > 0`, `2 * A * c < 1` and the anchors are visible, every anchor
has positive weight at every finite parameter assignment. Ordinary slots
can remain exactly inactive, including the checked weights `(1/2, 1/2, 0)`.

[AnchoredValues.lean](src/Transformer/GPTMini/Sparsemax/AnchoredValues.lean)
uses `d + 1` anchors in a `d`-dimensional output: a trainable common base
and that base plus each fixed basis direction multiplied by a positive
learned exponential scale. Ordinary values are arbitrary. All active value
differences automatically span the output, for arbitrary changes of the
base, scales, score parameters and ordinary values. Thus preservation
under finite parameter updates follows from the parameterization; the
span is no longer a hypothesis about learned values. A nonzero ordinary
output derivative always distinguishes two anchors and cannot disappear
in the raw score path.

This is an explicit anchored architecture modification, requiring `d + 1`
visible prefix slots and independent anchor scores. It does not prove that
the existing query/key architecture satisfies the same condition, or that
a zero output derivative or whole-model nonconvexity is repaired. No route
target, teacher, dense weight branch, Python implementation or new training
run is introduced. The completed benchmark cycle remains unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden`; the new modules have no warnings. Python is unchanged
since the passing 171-test run recorded above.

## Pure sparse follow-up: reaching trainable parameters

**Done on 2026-10-05 UTC.** Four modules and a readout evaluation lemma
add twenty-three proved theorems, with no new `sorry`, derived from
arXiv:1602.02068v2, §2.2 and §2.5, and the ordinary value sum at `73f8a0b`.

[AnchorTransfer.lean](src/Transformer/GPTMini/Sparsemax/AnchorTransfer.lean)
lifts an exact active-anchor score transfer through the bounded coordinate
inverse. The curve starts at the actual parameters, is differentiable,
and realizes the same raw score transfer on a neighborhood. Ordinary
scores and values stay frozen along this particular curve.

[TrainableAnchors.lean](src/Transformer/GPTMini/Sparsemax/TrainableAnchors.lean)
proves that the ordinary task derivative along this parameter curve is
`output_gradient(value_j - value_k)`. The scaled-basis anchors always
supply a separating pair for a nonzero output derivative. Thus the
earlier raw-score guarantee now reaches actual trainable anchor parameters,
for every finite assignment; no supplied attention route is required.

[AnchoredSquaredError.lean](src/Transformer/GPTMini/Sparsemax/AnchoredSquaredError.lean)
proves the actual derivative of squared distance to an ordinary output
target in a real inner-product space. It vanishes exactly at the target.
For the anchored architecture, every wrong finite output has a nonzero
trainable score direction and is not a local minimum of this row loss.
The local-minimum proof uses the differentiable parameter curve rather
than assuming a full sparsemax derivative at support boundaries.

[AnchoredCorrection.lean](src/Transformer/GPTMini/Sparsemax/AnchoredCorrection.lean)
checks a complete finite correction. Values are `(-1/16, 15/16, 7)` and
the ordinary output target remains `1/2`. Zero initial anchor parameters
give weights `(1/2, 1/2, 0)`, output `7/16` and loss `1/256`. The lifted
active-pair derivative is `-1/8`. A raw score transfer of `1/16` is exactly
realized by the finite parameters `(-log 3, log 3)`, yields weights
`(7/16, 9/16, 0)` and reaches zero ordinary output error.

The closure is for the explicit anchored row architecture. It does not
transfer automatically to the existing query/key factorization, rule out
gradient cancellations between rows sharing parameters, supply a uniform
gradient lower bound, or prove global learning on the basis or Shakespeare.
Fixed value hulls can still make a target unattainable. All results use
the actual causal variational sparsemax with exact zeros and ordinary
output losses. Python and the completed benchmark cycle are unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings. No new
`sorry`, extra axiom, result resting on a `sorry`, vacuous statement or
placeholder is introduced. Python is unchanged since its passing 171-test
run recorded above.

## Pure sparse follow-up: reaching query/key projections

**Done on 2026-10-05 UTC.** Five modules add twenty-five proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the linear projections, normalization and value sum at `73f8a0b`.

[QKChart.lean](src/Transformer/GPTMini/Sparsemax/QKChart.lean)
constructs actual unit keys in an orthogonal unit frame. A desired score
`s` in `(-g, g)` is represented by
`(s/g) * query + sqrt(1 - (s/g)^2) * transverse`. The existing normalized
dot product, with temperature `log g` and epsilon at most one, is proved
equal to `s`. The vector chart is differentiable on its strict interval.

[QKAnchors.lean](src/Transformer/GPTMini/Sparsemax/QKAnchors.lean)
sets `g = c + 1 + sum exp ordinary`, which strictly dominates every
anchor and ordinary score at all finite parameters. The actual QKNorm
scores therefore equal the earlier anchored scores. Every key has norm
one; the key array is differentiable in the learned anchor parameters.
The actual sparse example retains weights `(1/2, 1/2, 0)`.

[QKTaskDirections.lean](src/Transformer/GPTMini/Sparsemax/QKTaskDirections.lean)
proves the full active-value span on these actual Q/K rows and excludes
a zero ordinary task derivative both in anchor parameters and in all
actual key-vector directions. The chain rule uses the genuine differentiable
key chart; no sparsemax derivative is assigned or assumed at support boundaries.

[QKSquaredError.lean](src/Transformer/GPTMini/Sparsemax/QKSquaredError.lean)
removes the nonzero output-gradient premise for ordinary squared output
error: any wrong output has a nonzero actual K direction and is not a
local minimum of the key-vector loss. The finite correction's loss
`1/256` to `0` is now verified through the existing QKNorm operator.

[QKProjection.lean](src/Transformer/GPTMini/Sparsemax/QKProjection.lean)
evaluates actual shared linear Q/K projection matrices on standard-basis
inputs. Each input selects one column, so the nonzero task direction
reaches the jointly trained projection parameters. Equal inputs are
proved to force equal scores under every shared projection, explaining
why arbitrary input embeddings do not inherit this freedom.

[QKProjectedError.lean](src/Transformer/GPTMini/Sparsemax/QKProjectedError.lean)
adds four further proved theorems. Wrong squared-error outputs have a
nonzero direction in the jointly learned Q/K projection matrices and
cannot be local minima of that projection loss. The complete finite
correction, now evaluated through the shared linear projections,
normalization and actual sparsemax, changes only the key matrix and
reduces ordinary output loss from `1/256` to `0`.

This closes the score-to-Q/K transfer for an explicit restricted row
architecture. It requires an orthogonal unit query frame, chart-parameterized
keys, a gain depending on this row's ordinary score parameters, `d + 1`
visible value anchors and independent input coordinates for the projection
result. These are modifications, not consequences of the existing unrestricted
transformer. Shared-row gradient cancellation, target attainability with fixed
values and global convergence remain open; no basis or Shakespeare training
claim is made. Python and the completed experiment cycle are unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Pure sparse follow-up: removing standard-basis inputs

**Done on 2026-10-05 UTC.** Six modules add forty proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the shared projections and normalization at `73f8a0b`.

[InputDecoder.lean](src/Transformer/GPTMini/Sparsemax/InputDecoder.lean)
proves that a finite input family's linear independence is equivalent to
the existence of a continuous linear coordinate decoder. Its existence
comes from a proved left inverse of the synthesis map; no decoder oracle
or unproved accessibility assumption is used. Inputs need not be standard
basis vectors, orthogonal, or span the ambient space. The rank restriction
is explicit: the full family must have size at most the input width, and
a longer context cannot admit a full coordinate decoder.

[ProjectionLift.lean](src/Transformer/GPTMini/Sparsemax/ProjectionLift.lean)
constructs the actual matrix-column lift and proves its evaluation is a
right inverse. [ProjectionUpdate.lean](src/Transformer/GPTMini/Sparsemax/ProjectionUpdate.lean)
turns it into an affine update from an arbitrary current key matrix. The
update starts at that matrix, is differentiable, realizes every requested
key row, preserves the matrix's action on decoder-kernel inputs, and
composes correctly with subsequent updates. This closes the gap between
a specially constructed matrix and a local perturbation at actual parameters.

[QKIndependentDirections.lean](src/Transformer/GPTMini/Sparsemax/QKIndependentDirections.lean)
transports a nonzero ordinary output derivative into actual shared key
columns and joint Q/K columns on any independent input family.
[QKIndependentError.lean](src/Transformer/GPTMini/Sparsemax/QKIndependentError.lean)
proves that wrong squared-error outputs cannot have a zero joint matrix
derivative and cannot be local minima of that matrix loss. The latter
uses continuity of the affine lift and holds without assuming sparsemax
differentiable at inactive support boundaries.

[MixedInputQK.lean](src/Transformer/GPTMini/Sparsemax/MixedInputQK.lean)
supplies concrete inhabited premises with input vectors `(2, 1, 0)`,
`(1, 2, 0)` and `(0, 0, 1)`. Actual shared matrices and QKNorm realize the
anchored scores. A finite change of the key matrix lowers ordinary output
error from `1/256` to `0`, retaining the third weight exactly zero.

The unit query frame, chart-restricted projected keys, fixed values,
visible value anchors and single unrotated row remain explicit conditions.
Full-family independence is still stronger than arbitrary learned input
embeddings; long contexts, shared-row cancellation and whole-model
convergence remain open. No new basis or Shakespeare training claim is
made. The completed experiment cycle and Python implementation are unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Pure sparse follow-up: partial anchor access on long contexts

**Done on 2026-10-05 UTC.** Five modules add twenty-seven proved
theorems with no new `sorry`, derived from arXiv:1602.02068v2,
§2.2 and §2.5, and shared Q/K projections and QKNorm at `73f8a0b`.
The full-context independence and resulting context-width bound are
removed from the task-direction and local-minimum results below.

[PrefixInputs.lean](src/Transformer/GPTMini/Sparsemax/PrefixInputs.lean)
constructs dedicated anchor channels followed by arbitrary ordinary
embedding channels. Its proved continuous linear decoder identifies
each anchor and kills every ordinary input, regardless of repetitions,
rank or context length. A context longer than the input width is proved
dependent and still satisfies the partial decoder identities. This
architecture uses `A + B` input coordinates for `A + N` tokens, where
`N` is unrestricted. The ordinary embedding channels are not decoded.

[QKPrefixMatrix.lean](src/Transformer/GPTMini/Sparsemax/QKPrefixMatrix.lean)
builds a differentiable path through the actual current matrix that
changes anchor keys and preserves all ordinary keys. Because the ordinary
keys in the anchored family do not depend on anchor parameters, the
actual shared projection realizes the entire desired family along that
path. A special initial matrix with zero unseen components is not required.

[QKPrefixDirections.lean](src/Transformer/GPTMini/Sparsemax/QKPrefixDirections.lean)
proves that a nonzero ordinary output derivative survives in the shared
key matrix and joint Q/K matrices under only the partial decoder condition.
[QKPrefixError.lean](src/Transformer/GPTMini/Sparsemax/QKPrefixError.lean)
excludes zero joint matrix derivatives and erroneous local minima for
ordinary squared output error. The local-minimum proof uses the continuous
anchor path and does not assume full sparsemax differentiability at
inactive support boundaries.

[LongContextQK.lean](src/Transformer/GPTMini/Sparsemax/LongContextQK.lean)
provides explicit shared matrices for two anchors and arbitrarily many
repeated nonzero ordinary inputs, using input width three and head width
two. The actual normalized scores realize every finite anchor assignment.
The initial actual sparse row has weights `1/2` on both anchors and
exactly zero on every ordinary token. With anchor values zero and one,
ordinary values seven and output target zero, its incorrect output is
`1/2`. The general matrix theorems have concrete inhabited examples at
one hundred ordinary tokens, where full independence is proved impossible.

The partial decoder is an enforced input-channel modification, with a
proved implementation, rather than a consequence of arbitrary embeddings.
The current projected key family, unit query frame, active value anchors,
fixed values and single unrotated readout before XSA/output projection
remain explicit conditions. Shared-row cancellation, uniform gradient
bounds and whole-model convergence remain open. The completed experiment
cycle and Python implementation are unchanged; no new dataset training
claim is made.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Pure sparse follow-up: transfer to a shared multi-row objective

**Done on 2026-10-05 UTC.** Nine modules add fifty-eight proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and actual shared Q/K projections, QKNorm and value sums at `73f8a0b`.
This is a conditional transfer of the single-row result to the sum;
unconditional transfer is refuted by an actual shared-matrix counterexample.

[SupportSegment.lean](src/Transformer/GPTMini/Sparsemax/SupportSegment.lean)
proves that matching actual endpoint zero patterns make sparsemax affine
on the score segment, including inactive threshold ties. It constructs a
normalized threshold for the actual simplex projection and retains exact
inactive zeros; no attention targets, dense mixture or replacement
projection are introduced.

[ClippedKeySegment.lean](src/Transformer/GPTMini/Sparsemax/ClippedKeySegment.lean)
connects this score segment to one actual shared key-matrix segment,
with the query matrix fixed. The additional restriction is explicit:
projected keys at both endpoints have norm at most epsilon. Their whole
convex segment remains within that ball, where the actual QKNorm maximum
denominator is constant and normalized scores are affine for every row.
This replaces the preceding curved unit-key chart for this transfer;
arbitrary normalized unit-key segments need not be affine. The example
uses epsilon one. At the usual epsilon `1e-6`, these premises require
projected keys within that much smaller ball; they are not an automatic
property of the existing initialization or an experiment-backed repair.

[SharedRows.lean](src/Transformer/GPTMini/Sparsemax/SharedRows.lean)
defines actual multi-row outputs and their summed ordinary squared loss.
Different examples can have different inputs and frozen values, while all
use one pair of actual Q/K matrices. Endpoint norm and support conditions
give simultaneous affine outputs and exact inactive zeros without query
independence, an input decoder, row-specific parameters or a context-width
bound. [SharedRowLoss.lean](src/Transformer/GPTMini/Sparsemax/SharedRowLoss.lean)
proves convexity of the sum along that common matrix segment. A shared
endpoint fitting all ordinary targets gives exact loss `(1-t)^2 * L(0)`.
[SquaredSegment.lean](src/Transformer/GPTMini/Sparsemax/SquaredSegment.lean)
supplies the ordinary squared-loss and neighborhood arguments.

[SharedRowMinimum.lean](src/Transformer/GPTMini/Sparsemax/SharedRowMinimum.lean)
excludes a local minimum in jointly learned Q/K whenever a better
compatible shared key endpoint exists, even without an exact target fit.
Consequently, a joint local minimum is optimal against all clipped-key
endpoints with the same row supports and fixed Q. If a common admissible
endpoint fits the ordinary targets, positive-loss local minima are
excluded. [SharedRowDerivative.lean](src/Transformer/GPTMini/Sparsemax/SharedRowDerivative.lean)
proves that the actual common path has right derivative `-2 * L(0)`.
At positive loss, this rules out a zero full joint Q/K derivative without
assuming ambient sparsemax differentiability across support boundaries.

[SharedRowsExample.lean](src/Transformer/GPTMini/Sparsemax/SharedRowsExample.lean)
provides actual shared matrices for two different causal queries in a
four-position context. Both initial outputs are `1/2`, ordinary targets
are `11/16` and `5/16`, and the initial errors have opposite signs. One
shared key correction attains both: the sum drops from `9/128` to zero,
is `9/512` halfway along the actual matrix path, and has initial right
derivative `-9/64`. Both anchors stay active and ordinary weights stay zero.
Every conditional theorem has a concrete inhabited example.

[SharedRowCancellation.lean](src/Transformer/GPTMini/Sparsemax/SharedRowCancellation.lean)
refutes unconditional transfer. Two identical actual observations with
ordinary targets zero and one necessarily have the same output for every
shared parameter point. A certified sparse output `1/2` is a global and
local matrix minimum with summed loss `1/2`, despite both rows being wrong.
It has two active anchors and exact ordinary zeros, rather than a saturated
one-hot route. No common target fit exists. Ordinary target attainability
cannot be inferred from separate single-row correction theorems.

Inputs, values, gain and epsilon remain fixed, and readouts are unrotated
and before XSA/output projection. Joint attainability, compatible endpoint
supports and epsilon-clipped keys remain restrictions. Uniform gradient
bounds, unrestricted support changes and whole-model convergence remain
open. The completed experiment cycle and Python implementation are unchanged;
no new basis or Shakespeare training claim is made.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Convex embedding and attention foundation

**Done on 2026-10-05 UTC.** At the user's request, the next architectural
stage focuses on learned embeddings and attention; FFN and task-loss design
are deferred. Seven modules add fifty-five proved theorems without new
`sorry`, derived from sparsemax arXiv:1602.02068v2, Eq. (1) and Proposition 1.
This is a new restricted architecture, not a completed transformer repair
or an equivalence to the QKNorm implementation at `73f8a0b`.

[EmbeddingGram.lean](src/Transformer/GPTMini/Sparsemax/EmbeddingGram.lean)
learns one positive semidefinite Gram matrix on the query and key copies of
a finite vocabulary. Its full bounded domain is proved convex; every feasible
matrix has exact finite embedding coordinates. Query and key families both
remain trainable, and no fixed small feature width is assumed. Unnormalized
Gram scores replace QKNorm, RoPE and the learned multiplicative gain.

[GramRouting.lean](src/Transformer/GPTMini/Sparsemax/GramRouting.lean)
uses this single matrix across every specified context and row. Shared token
identities select affine content scores, so identical visible token codes
receive identical weights; occurrence information requires richer codes.
The joint embedding/causal-weight domain is convex with no prescribed support.
[GramRoutingEnergy.lean](src/Transformer/GPTMini/Sparsemax/GramRoutingEnergy.lean)
proves joint convexity and the exact Jensen gap of the squared projection
energy. Actual sparsemax conditionally minimizes it at any fixed learned
Gram. The score-square term is retained: it is variable during embedding
learning. A future objective on free attention rows can change their
conditional optimizer, so this energy alone is not an exact task-training
reformulation.

[GramSupport.lean](src/Transformer/GPTMini/Sparsemax/GramSupport.lean)
proves that a Gram entry cap below one half guarantees two positive weights
on every row with two visible positions, for every feasible learned matrix.
A scalar embedding example has an exact visible zero whereas the feasible
zero Gram has full support. No routing targets or frozen Q/K are needed.
[GramBoundary.lean](src/Transformer/GPTMini/Sparsemax/GramBoundary.lean)
proves that the unrestricted exact sparsemax graph remains nonconvex even
inside this bounded PSD domain. Omitting the score-square term has an actual
Jensen violation. Ordinary value mixing has a nonconvex exact output graph
even on an unchanged full two-slot support with bounded scores and values;
this obstruction precedes any task loss or FFN.

[NormalizedGram.lean](src/Transformer/GPTMini/Sparsemax/NormalizedGram.lean)
provides an exact positive restriction: require each specified Gram score
row to be nonnegative, sum to one and vanish at future positions. These are
linear constraints on the learned matrix. The normalized domain is proved
convex, actual variational sparsemax fixes every such row, and the exact
embedding/attention graph is convex. Supports can acquire or lose exact
zeros. Normalization is an architectural constraint; it is not inferred
from PSD or an ordinary norm bound, and arbitrary-context feasibility is
not assumed.

[NormalizedGramExamples.lean](src/Transformer/GPTMini/Sparsemax/NormalizedGramExamples.lean)
inhabits the exact domain at cap `3/8`. Two genuine one-feature embedding
families have different Q and K squared norms. The actual row changes from
`(1/4, 1/4, 1/4, 1/4)` to `(1/3, 1/3, 1/3, 0)`; its true Gram midpoint
gives `(7/24, 7/24, 7/24, 1/8)`, exactly the mean endpoint attention.
A positive two-by-two minor proves that this midpoint cannot be recovered
with only one feature, recording the cost of a fixed small embedding width.

The foundation covers Q/K embeddings and attention weights on the specified
finite contexts. Jointly learned value mixtures, arbitrary-context
normalization, a practical embedding-width bound, FFN and task loss remain
outside the guarantee. No new basis or Shakespeare training is claimed.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Exact convex coordinates for jointly learned values

**Done on 2026-10-05 UTC.** The user's joint-values extension adds five
modules and forty-six proved theorems without new `sorry`. FFN and task-loss
design remain deferred. This is an exact change of coordinates on an explicit
finite-context architecture, not a convexity claim in the original value
parameters or an unrestricted language-model guarantee.

[InvertibleGram.lean](src/Transformer/GPTMini/Sparsemax/InvertibleGram.lean)
uses one distinct-token context with every causal row and the normalized
learned Gram from the previous stage. A positive diagonal floor `eta` adds
linear constraints. The domain remains convex, actual sparsemax attention
is lower triangular, and its determinant is positive for every feasible
learned Gram. Off-diagonal supports can change. The entry cap is explicit;
the forced first causal row cannot satisfy the earlier sub-half cap.

[GramValues.lean](src/Transformer/GPTMini/Sparsemax/GramValues.lean)
replaces a learned shared value table `V` by its output table `Z = A(G) * V`.
It decodes `V = A(G) inverse * Z`, with both inverse identities, uniqueness,
injectivity and surjectivity proved for the actual sparsemax/value product.
Changing the Gram also admits an exact output-preserving value transport.
No original values are frozen, private row values or routing targets used.

[JointGramValues.lean](src/Transformer/GPTMini/Sparsemax/JointGramValues.lean)
proves convexity of the joint Gram/output domain and affinity of the actual
decoded forward map. Any future convex objective on outputs is convex in
these new learned coordinates. Every feasible original Gram/value pair is
recovered exactly, and every output table is attainable once the Gram domain
is inhabited. This is a bijective coordinate change, not a rank relaxation.
An output-only objective therefore leaves the Gram undetermined: learned
values can compensate any feasible attention change on this context.

[GramValueExampleGrams.lean](src/Transformer/GPTMini/Sparsemax/GramValueExampleGrams.lean)
and [JointGramValueExamples.lean](src/Transformer/GPTMini/Sparsemax/JointGramValueExamples.lean)
give two genuine Q/K embedding tables in the same cap-four, floor-one-half
domain. Actual attention changes from identity to `[[1, 0], [1/2, 1/2]]`,
and one value table shared across both rows changes from `(1, 0)` to `(0, 2)`.
Outputs change from `(1, 0)` to `(0, 1)`. At the true joint midpoint, actual
attention is `[[1, 0], [1/4, 3/4]]`, recovered values and outputs are both
`(1/2, 1/2)`, and the determinant is `3/4`. Literal mean original values
instead give second output `7/8`, witnessing the nonlinear value decoder.

The guarantee covers this one distinct-token context, before XSA and output
projection. Repeated-token or arbitrary multi-context value sharing is not
proved. Original value penalties or norm bounds and a fixed small embedding
width are not preserved, and no uniform inverse conditioning bound is proved.
No new basis or Shakespeare training is claimed.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Convex shared memory across causal data contexts

**Done on 2026-10-05 UTC.** The user's continuation adds eight modules and
sixty-eight proved theorems without new `sorry`. One learned dictionary
and one learned value table now serve arbitrarily many context queries,
including repeated tokens. The positive construction changes ordinary
input-token self-attention into attention to learned parameter memory.
Fixed causal probability codes from data mix learned query embeddings.
FFN and task-loss design remain deferred.

[MemoryGram.lean](src/Transformer/GPTMini/Sparsemax/MemoryGram.lean)
normalizes every learned dictionary score row linearly and bounds its
diagonal below by a floor greater than one half. Together with a bounded
PSD Gram, these constraints define a convex domain. Actual variational
sparsemax equals its normalized genuine Q/K scores. Unit row mass gives
strict diagonal dominance, and Gershgorin proves nonsingularity without
a triangular support restriction. Both directional supports can change.

[MemoryValues.lean](src/Transformer/GPTMini/Sparsemax/MemoryValues.lean)
uses the exact global coordinates `Z = B(G) * V`, where `B` is dictionary
attention. It decodes one table `V = B(G) inverse * Z`, independent of the
context. Both inverse identities and uniqueness are proved. Original values
and both embedding families remain variable in the convex joint domain.

[ContextMemory.lean](src/Transformer/GPTMini/Sparsemax/ContextMemory.lean)
proves that scores for a probability code `M` are actual inner products of
mixed learned Q and the common learned K. Actual sparsemax equals `M * B`.
Data codes are inputs, not teacher routes or attention targets. All learned
memory slots are visible parameters preceding the query; they do not contain
future observations. Codes are fixed during optimization, while query and
key embedding coordinates remain trainable through the Gram.

[PrefixMemoryCodes.lean](src/Transformer/GPTMini/Sparsemax/PrefixMemoryCodes.lean)
constructs such codes from arbitrary causal token prefixes. Zero-score
sparsemax is proved exactly uniform on every visible prefix, and its mass
is aggregated by token identity. Every length and repetition pattern gives
a probability code. Future-token changes leave codes unchanged, and absent
visible tokens receive exactly zero mass. This concrete encoder retains
prefix frequencies and loses word order; other fixed causal probability
codes can be supplied to the general memory results.

[SharedMemoryValues.lean](src/Transformer/GPTMini/Sparsemax/SharedMemoryValues.lean)
proves simultaneous exact original-value recovery and actual forward affinity
for all context rows. The shared decoded output is `M * Z`, computed through
the real sparsemax/value operation. Prefix-code forwards are causal even
outside the structural Gram domain.
[SharedMemoryGeometry.lean](src/Transformer/GPTMini/Sparsemax/SharedMemoryGeometry.lean)
proves that every future convex criterion on the whole context-output table
is convex in joint Gram/output coordinates. The exact prediction class is
the linear image `{M * Z}` and is convex; arbitrary context targets need not
be attainable. A common decoder compensates feasible Gram changes across
all contexts, so an output-only objective leaves the Gram undetermined.

[MemoryExampleGrams.lean](src/Transformer/GPTMini/Sparsemax/MemoryExampleGrams.lean)
gives a genuine feasible identity embedding Gram for every dictionary size.
Its two-slot witness changes Q/K squared norms and actual memory attention
from identity to `[[3/4,1/4],[1/4,3/4]]`, acquiring both off-diagonal supports.
[SharedMemoryExamples.lean](src/Transformer/GPTMini/Sparsemax/SharedMemoryExamples.lean)
uses actual repeated-token prefixes `(1,1,1)`, `(0,1,1)` and `(0,0,1)`.
One original value table changes from `(1,0)` to `(-1/2,3/2)`, while global
output coordinates change from `(1,0)` to `(0,1)`. The three actual context
outputs change from `(0,1/3,2/3)` to `(1,2/3,1/3)`. At the true midpoint,
one shared decoded table `(1/2,1/2)` gives all three outputs one half.
Literal mean original values instead give first-context output `11/16`.
Every actual output obeys `Y2 = 2 * Y1 - Y0`; the target triple `(0,0,1)`
is proved unattainable for every feasible Gram and common value table.

The guarantee covers the stated memory architecture and fixed data codes.
It does not recover unrestricted occurrence self-attention, trainable codes,
a fixed small embedding width, original value penalties or inverse norm bounds.
Outputs precede XSA and output projection. No new basis or Shakespeare training
is claimed, and the completed experiment cycle is unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Richer causal codes remove the three-target obstruction

**Done on 2026-10-05 UTC.** The user's challenge adds five modules and
thirty-eight proved theorems without new `sorry`. The previous exclusion
of `(0,0,1)` was specific to the rank-two token-frequency code. It remains
valid for that encoder; it is not a universal impossibility for shared
values or the convex memory architecture. Richer codes are derived from
observations alone, and the learned Gram, values and supports remain variable.

[PrefixFeatureCodes.lean](src/Transformer/GPTMini/Sparsemax/PrefixFeatureCodes.lean)
pushes causal zero-score sparsemax probabilities through arbitrary fixed
position/token features. Probability feasibility and future insensitivity
are proved, as is the actual shared-memory output formula. The same three
prefixes use six shared `(position, token)` slots; their code rows occupy
`(1,3,5)`, `(0,3,5)` and `(0,2,5)`, each with weights one third.

[PositionalMemoryTargets.lean](src/Transformer/GPTMini/Sparsemax/PositionalMemoryTargets.lean)
constructs a fixed right inverse `C` with `M * C = I`. For every finite
vector-valued target table `Y` and every feasible learned Gram, one common
original value table `V = B(G) inverse * C * Y` gives actual outputs `Y`.
Scalar global coordinates for `(a,b,c)` put `3*(a-b)`, `3*(c-b)` and `3*b`
in slots 1, 2 and 5. The exact actual prediction class is the entire target
space. From every current joint point, the update `Z += C * delta` produces
any desired output correction `delta`; no task loss or attention target is
needed for this forward-map statement. Explicit original values with only
slot 2 equal to three give the formerly excluded triple `(0,0,1)`.

[CodeMemoryOutputs.lean](src/Transformer/GPTMini/Sparsemax/CodeMemoryOutputs.lean)
proves a general categorical-code characterization. One global table extends
every target table constant on equal-code fibers. Actual shared-memory
attainment is equivalent to this consistency, for every feasible Gram.
The encoder is chosen from data before the target table is supplied.

[CausalPrefixKeys.lean](src/Transformer/GPTMini/Sparsemax/CausalPrefixKeys.lean)
encodes complete visible prefixes, masking every future position with `none`.
Equality of dictionary keys is proved equivalent to equality of query positions
and every visible token. Word order, repetitions and prefix length are retained.
The full target-independent dictionary has exactly `(V + 1)^T` slots for
vocabulary size `V` and window length `T`, including unused signatures.

[CausalMemoryUniversality.lean](src/Transformer/GPTMini/Sparsemax/CausalMemoryUniversality.lean)
proves that the exact actual prediction class is all output tables consistent
on identical observed causal prefixes. Distinct prefixes permit arbitrary
finite vector targets; repeated observations with contradictory targets remain
impossible for a deterministic causal model. The joint actual forward is causal,
the shared prediction class is convex, and the earlier convex-objective theorem
applies. Original values are decoded globally, independently of context.

The six-slot result removes the specific example obstruction at modest cost.
Complete-prefix universality pays an exponential dictionary cost and unbounded
feature width; it does not prove an efficient compact ordinary transformer.
Fixed data codes and the learned-memory architecture remain explicit. An
output-only objective still leaves the Gram undetermined. FFN, task-loss choice
and new training remain deferred; the completed experiment cycle is unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Compact observed-prefix kernel memory

**Done on 2026-10-06 UTC.** The requested compact mathematical construction
adds six modules and forty-two proved theorems without new `sorry`. Register
`P` distinct observed causal prefixes before target fitting and use `P`
learned memory slots. This replaces the exponential complete-prefix dictionary
by a data-relative memory. The learned Gram and the common value table remain
variable on the earlier exact convex joint domain.

[PrototypeKernelCodes.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelCodes.lean)
defines the fixed data kernel
`k(q,j) = indicator(prefix(q) = prefix(j)) + 1 + dot(phi(q), phi(j))^2`.
Every entry is at least one, giving a positive normalization denominator
even on unseen queries. Its normalized profiles are probability codes.
Evaluation uses dot products and a normalized weighted sum of the global
output table, without enumerating all signatures or ordered feature pairs.

[PrototypeKernelFeatures.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelFeatures.lean)
proves that the squared dot product is the Gram of ordered pair features.
On distinct registered observations, the training kernel is identity plus
the constant Gram plus this genuine pair-feature Gram. It is positive
definite and nonsingular for arbitrary real feature rows, even when those
rows are zero or rank deficient. The two-prototype example with scalar
features `(1,2)` gives kernel `[[3,5],[5,18]]` and determinant 29.

[PrototypeKernelTraining.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelTraining.lean)
constructs the data-derived right inverse `C = kernel inverse * diagonal(row mass)`
of the normalized training codes `M`, proving `M * C = I`. Every vector
target table `Y` has global coordinates `Z = C * Y` and one common original
value table `V = B(G) inverse * C * Y`. Actual sparsemax/value outputs are
exactly `Y` for every feasible learned Gram. Any current point admits every
prototype-output correction through `Z += C * delta`; attainability is a
proved consequence of the encoder, not an assumed target-compatibility input.

[BoundedGramWidth.lean](src/Transformer/GPTMini/Sparsemax/BoundedGramWidth.lean)
retains the explicit dimension in Mathlib's positive spectral decomposition.
Every PSD Gram on `2P` query/key indices is recovered exactly with Q/K width
`2P`, preserving the squared norm cap. The same feature table realizes all
context scores and actual normalized attention. Context count does not
increase this width, and no nonconvex rank restriction is added to the domain.

[PrototypePrefixMemory.lean](src/Transformer/GPTMini/Sparsemax/PrototypePrefixMemory.lean)
constructs the kernel profiles from actual masked prefix signatures and
causal position/token features. Future changes in either queries or prototype
records leave codes and actual outputs unchanged. Its final theorem combines
arbitrary prototype-target recovery, one common original value table and
Q/K width `2P`, for every feasible learned Gram. The actual three-prefix
example fits `(0,0,1)` with three memory slots, versus six position/token
slots or twenty-seven complete-signature slots in the earlier constructions.

[PrototypeKernelGeometry.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelGeometry.lean)
proves convexity of every future convex output criterion on the same joint
domain, together with the actual normalized-kernel forward formula. Matrix
rank gives a lower bound: this exact fixed-code affine chart needs at least
`R` memory slots to realize every independently specifiable vector target
table on `R` contexts. The construction attains that slot bound on its
registered distinct observations. This is a bound for this architecture,
not an impossibility for every transformer or structured target class.

The guarantees cover finite registered observations. Learned Gram parameters
still grow quadratically with `P`, the width bound grows linearly, and fixed
features and prototype selection remain explicit. Positive profiles and the
diagonal floor force dense actual query attention; memory supports can still
change. Output-only optimization still leaves the Gram undetermined under
the common value decoder. Unseen queries have a well-defined prediction,
without a generalization guarantee or an independently specifiable target.
FFN, task-loss choice and new training remain deferred; the completed sparsemax
experiment cycle is unchanged. A sample-independent compact construction for
structured text targets remains a mathematical question.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Sparse nearest observations and conditional unseen guarantees

**Done on 2026-10-06 UTC.** Three modules add twenty-one proved theorems
without new `sorry`. The nearest variant uses the same variable learned
memory and globally decoded shared values as the kernel construction.

[NearestPrototypeCodes.lean](src/Transformer/GPTMini/Sparsemax/NearestPrototypeCodes.lean)
selects a minimum-distance registered observation, independently of targets.
Its data code has one active slot. Nonnegative separating distances, zero
self-distance and distinct prototypes imply identity training codes. Actual
query attention is the selected learned memory row; this does not yet bound
the sparsity of that memory row.

[PrefixNearestCodes.lean](src/Transformer/GPTMini/Sparsemax/PrefixNearestCodes.lean)
uses Hamming distance between masked causal prefix signatures. Distance is
nonnegative, symmetric, separating and satisfies the triangle inequality.
Future changes in queries or prototype records leave codes unchanged.
Distinct registered prefixes have identity codes, so arbitrary consistent
prototype targets remain attainable with the existing common-value decoder.

[NearestMemoryGeneralization.lean](src/Transformer/GPTMini/Sparsemax/NearestMemoryGeneralization.lean)
proves the actual forward selects the corresponding global output row.
Prototype fitting error at most `epsilon`, a target Lipschitz constant `L`
and data coverage radius `delta` imply unseen coordinate error at most
`epsilon + L * delta`. These are explicit assumptions with nonconstant,
unseen-query examples, not consequences of interpolation or Shakespeare
claims. Two target functions agree on both registered observations and
disagree at the third; every prediction has error at least one half for one
of these indistinguishable targets there.

The result removes dense data codes and replaces the absence of any unseen
bound by a conditional one. Gram parameter count, actual attention sparsity
and output-only Gram nonidentifiability remain to address. Prototype selection
and distance stay fixed data choices. FFN, task-loss selection and new
training remain deferred; the completed experiment cycle is unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the modules have no warnings, new `sorry`, extra
axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Linear-parameter Gram and at most three actual routes

**Done on 2026-10-06 UTC.** Nine modules add sixty-eight proved theorems
without new `sorry`. The compact variant retains joint convex embeddings,
actual sparsemax attention and one common learned value table. It restricts
the earlier unrestricted Gram family to a fixed path of possible connections.

[PermutationMemoryGram.lean](src/Transformer/GPTMini/Sparsemax/PermutationMemoryGram.lean)
constructs genuine finite-coordinate Gram atoms from basis queries and
permuted basis keys. Identity and adjacent swaps are bounded PSD memories
with probability cross-score rows. Both same-family blocks are identity.

[LocalMemoryWeights.lean](src/Transformer/GPTMini/Sparsemax/LocalMemoryWeights.lean)
learns nonnegative path-edge weights `t`, with `sum(t) <= 1 - floor`.
Identity receives residual mass `1 - sum(t)`, at least the required diagonal
floor. The domain is convex and inhabited exactly when `floor <= 1`.
[LocalMemoryCore.lean](src/Transformer/GPTMini/Sparsemax/LocalMemoryCore.lean)
proves the affine mixture of these actual atoms is a bounded PSD memory
with the required score normalization and floor, without assuming a feature
factorization or imposing a nonconvex rank constraint.

[LocalMemoryParameters.lean](src/Transformer/GPTMini/Sparsemax/LocalMemoryParameters.lean)
adds `2P` independently stored norm coordinates `u`, with `0 <= u <= cap-1`.
The full affine Gram is `core(t) + diagonal(u)`. Its cross scores are unchanged
by these additions and genuine Q/K squared norms are `1 + u`. The complete
coordinate encoding is injective and has exactly `3P-1` stored scalars.
The parameter domain is convex and inhabited exactly for `cap >= 1` and
`floor <= 1`. This is storage count, not the dimension of a boundary domain.

[LocalMemoryFeasibility.lean](src/Transformer/GPTMini/Sparsemax/LocalMemoryFeasibility.lean)
proves all bounded PSD and normalized memory constraints from the linear
parameter bounds. Actual variational sparsemax equals the path scores.
A floor strictly above one half guarantees its inverse, one global original
value table, and genuine Q/K feature width at most `2P`. Actual attention
is affine throughout the convex parameter domain, including support changes.

[LocalMemorySupport.lean](src/Transformer/GPTMini/Sparsemax/LocalMemorySupport.lean)
proves exact zeros outside a query slot and its two immediate neighbors.
Every actual nearest-data query has at most three active sparsemax routes,
including unseen queries; the bound concerns attention itself, not just codes.
[LocalJointMemory.lean](src/Transformer/GPTMini/Sparsemax/LocalJointMemory.lean)
retains the exact common-value forward `MZ` and a convex joint domain for
the compact parameters and global output table. Every future convex output
criterion is convex jointly. Every output-only criterion is also invariant
under feasible attention-parameter changes compensated by the common values.

[LocalNearestMemory.lean](src/Transformer/GPTMini/Sparsemax/LocalNearestMemory.lean)
combines this family with actual causal nearest-prefix codes. Distinct
registered prefixes fit arbitrary vector targets. Future query tokens do not
affect outputs. Explicit prototype fitting, target regularity and coverage
give actual unseen coordinate error at most `epsilon + L * delta` here too.
[LocalMemoryExamples.lean](src/Transformer/GPTMini/Sparsemax/LocalMemoryExamples.lean)
changes both Q/K squared norms from one to two and actual two-slot support
from identity to weights `(3/4,1/4)` in both directions. A four-slot interior
row has weights `(1/10,4/5,1/10,0)` and exactly three active routes, also for
the unseen real query `5/4`. The forward map in compact attention parameters
is proved noninjective for every fixed output table and probability encoder.

Linear Gram storage and bounded actual route count are now proved. Their
price is the fixed path, fixed same-family orthogonality and a global edge
budget stronger than separate row budgets. Prototype count, feature width
and nearest-search cost can still grow with data. Output-only Gram freedom
remains intrinsic to this exact value chart; a separate criterion is needed
to select parameters. Unseen regularity and coverage have not been established
for Shakespeare or arbitrary text. FFN, task-loss selection and new training
remain deferred; the completed sparsemax experiment cycle is unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the modules have no warnings, new `sorry`, extra
axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Unique compact attention selection with an additional convex criterion

**Done on 2026-10-06 UTC.** Three modules add twenty-two proved theorems
without new `sorry`. They address the compact chart's remaining output-only
Gram freedom by an explicit extra criterion, while preserving joint convexity.
This is a proved optional mathematical extension, not a selected model task
loss, a solver implementation or evidence of useful learned semantic routes.

[LocalMemoryQuadratic.lean](src/Transformer/GPTMini/Sparsemax/LocalMemoryQuadratic.lean)
sums squared differences from a supplied reference over all `3P-1` compact
edge and Q/K-norm coordinates. It is nonnegative, zero exactly at the
reference and strictly convex. Its exact affine gap is
`a * R(reference,p) + b * R(reference,q) - R(reference,a*p+b*q)
= a*b*R(q,p)` for `a+b=1`. Two distinct constrained parameter minima cannot
exist on any convex domain; no embedding family is omitted from the criterion.

[LocalMemorySelection.lean](src/Transformer/GPTMini/Sparsemax/LocalMemorySelection.lean)
proves the actual compact parameter domain is compact: a finite coordinate
box intersected with a closed total-edge budget. The criterion is continuous,
so `cap >= 1` and `floor <= 1` imply a unique constrained minimum for every
reference, including infeasible references. A concrete reference violates
both edge and norm bounds and still has a unique admissible minimum. The
noncomputable selected-point definition records this proved optimization
result, not an executable numerical solver. A floor above one half gives
genuine width-`2P` embeddings and an inverse for the selected actual attention.

[LocalRegularizedMemory.lean](src/Transformer/GPTMini/Sparsemax/LocalRegularizedMemory.lean)
adds `coefficient * R(reference, parameters)` to any convex output criterion.
A nonnegative coefficient preserves convexity of the entire compact
parameter/global-output learning task. A strictly positive coefficient forces
every attained joint minimum to minimize the parameter criterion. All joint
minima have identical attention parameters, even if their global output
tables differ; these parameters equal the proved unique constrained projection.
Nonconstant squared-output witnesses attain the minimum with nonidentity
attention, changed Q/K norms and different nonconstant output tables.

The selection reference supplies additional information or an architectural
preference; it may be computed from observations, but no such text-derived
reference is claimed here. The unchanged output-only criterion remains
nonidentifying. This extension chooses among its equivalent attention/value
representations; it does not demonstrate learning useful routes from output
targets alone. The full joint minimum's existence still depends on the
output objective; parameter-projection existence is proved. Text regularity,
coverage and useful numerical generalization constants remain to establish.
FFN, task-loss selection and new training remain deferred; the completed
sparsemax experiment cycle is unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass; the modules have no warnings, new `sorry`, extra
axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Compact memory with separate incident-edge budgets

**Done on 2026-10-06 UTC.** Eight modules add 64 proved theorems without
new `sorry`. The global edge budget was sufficient but unnecessarily coupled
remote parts of the dictionary. The new convex domain requires nonnegative
edges and, separately at every slot, incident edge mass at most `1-floor`.
It retains the same `3P-1` stored edge and independent Q/K-norm coordinates.

[LocalIncidentWeights.lean](src/Transformer/GPTMini/Sparsemax/LocalIncidentWeights.lean)
proves domain convexity, bounds and strict enlargement. Three edges of `1/8`
at floor `3/4` satisfy every separate budget and fail the former global one.
[LocalMemoryScoreIdentities.lean](src/Transformer/GPTMini/Sparsemax/LocalMemoryScoreIdentities.lean)
proves symmetric unit-mass rows, diagonal `1-incidentMass`, nonnegative scores
and entry bounds. These identities hold for the unchanged affine path Gram.

[BipartiteMemoryGram.lean](src/Transformer/GPTMini/Sparsemax/BipartiteMemoryGram.lean)
constructs an explicit outer-product lift of any nonnegative cross-score table.
Unit row and column sums give identity Q/Q and K/K blocks. The cross block
recovers the score table exactly. [IncidentMemoryCore.lean](src/Transformer/GPTMini/Sparsemax/IncidentMemoryCore.lean)
proves the existing path Gram equals this lift, hence is PSD without a
nonnegative global identity coefficient. Local budgets imply the complete
bounded memory domain; actual sparsemax equals the path scores, is affine
and has an inverse for floor above one half. Width-`2P` recovery and exact
common-value decoding remain conclusions of the construction.

[IncidentMemoryParameters.lean](src/Transformer/GPTMini/Sparsemax/IncidentMemoryParameters.lean)
proves convexity, compactness and unique squared-criterion minimization on the
enlarged edge/norm domain. [IncidentMemoryFeasibility.lean](src/Transformer/GPTMini/Sparsemax/IncidentMemoryFeasibility.lean)
proves actual learned norm bounds, PSD, width recovery, inverse, common-value
fit and at most three routes. The new witness has an actual row
`(1/8, 3/4, 1/8, 0)` and lies outside the former compact parameter domain.

[IncidentJointMemory.lean](src/Transformer/GPTMini/Sparsemax/IncidentJointMemory.lean)
and [IncidentNearestMemory.lean](src/Transformer/GPTMini/Sparsemax/IncidentNearestMemory.lean)
reuse the actual forward unchanged and prove its MZ identity, joint affinity,
convex output criteria, arbitrary registered causal-prefix fitting, three-route
nearest attention and conditional `epsilon + L*radius` unseen error bounds.
Output-only geometry remains unidentified. The path, same-family orthogonality
and input metric remain fixed restrictions; generalization assumptions for
Shakespeare and a bounded prototype count/width remain unproved. Data-derived
parameter selection is the next extension. FFN and task-loss selection remain
deferred; the completed experimental training cycle is unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Observation-derived convex geometry selection and finite descent

**Done on 2026-10-06 UTC.** Six modules add thirty-nine proved theorems
without new `sorry`. The additional selection reference is now explicitly
computed from observations rather than supplied as an arbitrary table.
The separate-budget compact domain, actual sparsemax and common-value
forward are reused unchanged. Task-loss choice and FFN remain deferred.

[ObservedMemoryPreferences.lean](src/Transformer/GPTMini/Sparsemax/ObservedMemoryPreferences.lean)
uses `scale/(1+distance)` on adjacent observed prototypes, and independent
fixed query/key data-feature energies as preferred norm additions. It needs
only adjacent distances and per-prototype energies, with `3P-1` stored
reference coordinates, rather than a dense reference Gram or attention
labels. Nonnegative distances give nonnegative, scale-bounded preferences;
positive scale gives positive preferences and strict decrease with increasing
distance. Coupled constraints can alter their projected ordering. The metric,
feature maps, prototype order and scale are explicit fixed choices during
optimization; joint convex learning of these choices is not claimed.

[IncidentMemorySelection.lean](src/Transformer/GPTMini/Sparsemax/IncidentMemorySelection.lean)
selects the proved unique constrained projection, including infeasible
references. Its geometry has genuine width-`2P` embeddings, an actual
attention inverse, at most three routes and exact common-value decoding.
For any convex feasible set and an attained minimum q, midpoint comparison
proves `R(q,p) <= 2 * (R(reference,p)-R(reference,q))`. An epsilon-suboptimal
geometry therefore has squared complete-coordinate error at most `2*epsilon`.
The factor two is explicit; this bound concerns parameter error, not unseen
prediction error. The selector records a proved optimization result and is
noncomputable; no numerical solver is implemented.

[PrefixObservedMemory.lean](src/Transformer/GPTMini/Sparsemax/PrefixObservedMemory.lean)
computes the reference from masked prefix Hamming distances and features
of those same masked records. Hidden prototype continuations cannot change
the reference, selected parameters, or the full actual nearest-code/common-
value forward, even when geometry is recomputed from the altered dataset.
Distinct registered causal prefixes still fit arbitrary vector targets with
one globally decoded value table. These fit statements do not establish
text-target regularity or coverage for new text.

[ObservedMemoryExamples.lean](src/Transformer/GPTMini/Sparsemax/ObservedMemoryExamples.lean)
compares prototype tables `(0,1)` and `(0,3)` under identical metric, order,
scale and nonconstant separate features. Both references are feasible and
are recovered exactly by unique projection. Their actual selected sparsemax
edge weights are `1/16` and `1/32`; actual query and key squared norms both
change. The attention inverses are proved without assuming original embedding
coordinates or target routes. Observation dependence survives constrained
selection, rather than being asserted only for an unprojected formula.

[IncidentRegularizedMemory.lean](src/Transformer/GPTMini/Sparsemax/IncidentRegularizedMemory.lean)
proves joint convexity of the additional quadratic criterion plus any convex
output criterion on the enlarged compact domain. With positive coefficient,
every attained joint minimum uses the same proved parameter projection,
even at different output tables. The output-only criterion remains unable
to identify geometry. This extra criterion chooses among equivalent exact
attention/value representations; its semantic usefulness is not established.

[IncidentMemoryDescent.lean](src/Transformer/GPTMini/Sparsemax/IncidentMemoryDescent.lean)
proves a feasible midpoint toward a geometry minimum lowers the actual
regularized criterion by at least `coefficient/4 * R(q,p)`, including support
boundaries. Positive coefficient makes this a strict finite decrease whenever
p differs from q, without any convexity requirement on the fixed-output
criterion. A regularized fixed-output gap of epsilon bounds weighted squared
parameter error by `2*epsilon`. An attained output-coordinate minimum and
the proved geometry projection together supply an actual joint minimum;
output-minimum attainment is the remaining explicit objective assumption.

Possible path connections, same-family orthogonality, fixed observation metric
and feature maps remain restrictions. Prototype count and embedding width
still grow, and nearest search may scan the dictionary. Useful regularity,
coverage and constants for Shakespeare remain to establish. FFN, model task
loss selection and new training remain deferred; the completed experiment
cycle is unchanged. These results remove the supplied-reference gap and
geometric flatness of the additional criterion, not output-only semantic
learning or every unrestricted transformer obstruction.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Ordinary output error selects attention through a convex value-energy coupling

**Done on 2026-10-06 UTC.** Five modules add thirty-five proved theorems
without new `sorry`. Unlike the earlier additional geometry criterion, the
new restriction makes attainable ordinary output error depend on attention,
while every original shared value remains learned. No target attention routes
or embedding reference are supplied to the criterion.

[EnergyCoupledMemory.lean](src/Transformer/GPTMini/Sparsemax/EnergyCoupledMemory.lean)
adds the affine block PSD constraint `[[B,Z],[Z^T,energy I]] >= 0`, together
with the incident parameter bounds and a prescribed aggregate edge budget.
All these constraints are jointly convex in the compact Q/K and output
coordinates. B is the actual sparsemax attention on the domain. The genuine
shared-value forward is reused and remains MZ; every convex output criterion
remains jointly convex. Its unconstrained partial formula at fixed Z still
does not distinguish geometry: geometry now changes the feasible output set.

[EnergyMemoryValues.lean](src/Transformer/GPTMini/Sparsemax/EnergyMemoryValues.lean)
derives positive definiteness of actual attention from the coupling and
proved strict self-weight floor. The Schur complement gives an exact
equivalence, for arbitrary original common values V and Z=BV, with
`energy I - V^T B V >= 0`. Recovered shared values satisfy this bound in
every output direction. This is an attention-weighted energy budget, not a
Euclidean value-norm cap; its proof assumes no unproved inverse or factorization.

[EnergyMemoryExamples.lean](src/Transformer/GPTMini/Sparsemax/EnergyMemoryExamples.lean)
supplies two genuine three-slot examples in the same domain: floor `3/4`,
norm cap 4, total edge weight `1/8` and energy 6. Ordinary answers `(1,1,-2)`
fit with first edge `1/8` and second edge zero; answers `(-2,1,1)` fit with
the opposite allocation. Explicit PSD decompositions certify both examples,
and the globally decoded original value tables equal their nonconstant
answer tables. All values are trained; the examples do not use private values
per query or frozen decoder entries.

[EnergyMemoryIdentification.lean](src/Transformer/GPTMini/Sparsemax/EnergyMemoryIdentification.lean)
tests the PSD matrix on `(target,-1)`. Exact ordinary output fitting forces
the discordant edge to vanish; the positive aggregate budget then determines
the other edge. Both selected actual attention matrices are nonidentity and
different. Q/K squared norms remain a proved freedom at fixed edges: the
energy matrix and feasibility permit every otherwise feasible norm change.

[EnergyMemoryLoss.lean](src/Transformer/GPTMini/Sparsemax/EnergyMemoryLoss.lean)
uses only sum squared error of the actual shared-memory predictions. It
proves joint convexity, attained zero minima and attention identification at
every attained minimum. Fixing the opposite edge allocation leaves error at
least `27/512` for every feasible relearned common value table. This bound
is on the sum of squared errors, without averaging or a factor of one half.
The lower-bound premises have a concrete feasible nonconstant witness.

The aggregate budget excludes identity attention, which otherwise permits
every energy-feasible output of this symmetric path family and weakly
dominates mixing. It fixes only the total weight, not its allocation or
support. Selection here uses an active energy boundary and two finite
regression answer patterns. Unrestricted arbitrary-output fitting is traded
for the convex energy coupling. General task-conditioned identifiability,
interior flatness, useful language-model objectives, a numerical solver and
Shakespeare generalization remain open. Prototype count and feature width
still grow; FFN and new model training remain deferred.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Constant physical width for locally masked learned attention

**Done on 2026-10-06 UTC.** Five modules add thirty-three proved theorems
without new `sorry`. The physical Q/K head dimension is now exactly three
for arbitrary prototype count. This changes the old normalized-Gram
architecture by applying an explicit structural score mask; learned local
weights and actual sparse supports still vary.

[PeriodicMemoryKeys.lean](src/Transformer/GPTMini/Sparsemax/PeriodicMemoryKeys.lean)
assigns each prototype its index modulo three. Every possible local
destination triple has distinct classes, including clamped boundaries.
Each genuine key uses its class coordinate with learned scale equal to one
plus its incident edge mass. Scales lie in `[1,2-floor]`, and squared key
norms are at most four, independently of prototype count.

[PeriodicMemoryQueries.lean](src/Transformer/GPTMini/Sparsemax/PeriodicMemoryQueries.lean)
constructs genuine queries by placing each local score divided by its
destination's learned key scale in the corresponding coordinate. Actual
dot products recover every possible local score. Query coordinates lie in
`[0,1]`, and squared norms are at most three. Explicit simultaneous edge
changes alter both query and key families; neither family is frozen.

[PeriodicMemoryAttention.lean](src/Transformer/GPTMini/Sparsemax/PeriodicMemoryAttention.lean)
keeps genuine dot products on the local mask and assigns score -1 outside.
The original variational sparsemax has proved threshold zero and equals
the old affine path attention, including support boundaries. A self-weight
floor above one half retains the actual inverse. The real feature Gram is
PSD, has width three and obeys the uniform norm cap. It differs from the
old normalized Gram; the old affine lift is only a proof certificate here.

[PeriodicMemoryForward.lean](src/Transformer/GPTMini/Sparsemax/PeriodicMemoryForward.lean)
uses only learned edges and output coordinates, eliminating independent
norm additions. The earlier block PSD value-energy coupling and aggregate
edge budget remain jointly convex. Actual categorical queries multiply
one globally decoded original value table; registered and nearest codes
give the selected rows of Z. Joint affinity and convex output criteria are
proved for any finite dictionary and output dimension. Both earlier
nonidentity, nonconstant task examples remain feasible at physical width
three. Mixtures of differently masked queries are not asserted to commute
with the new mask.

[PeriodicMemoryRank.lean](src/Transformer/GPTMini/Sparsemax/PeriodicMemoryRank.lean)
proves the old unmasked normalized chart necessarily needs width at least
P: an invertible P-slot QK product cannot factor through fewer coordinates.
In the new construction the genuine unmasked score matrix has rank at
most three, while actual masked sparsemax attention has rank P. For P>3
these matrices and the former physical Gram provably differ. Thus the
fixed-width result does not silently assume a low-rank invertible QK product.

Stored prototype records, values, edge parameters and nearest search still
grow with prototype count. Possible path connections and categorical input
encoding remain restrictions; the physical Gram is not affine in its
three-coordinate realization, while the intrinsic optimization chart is
convex. General output-conditioned identifiability and removal of intrinsic
flat directions are the next extension. FFN, new model training and useful
Shakespeare generalization remain deferred.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Ordinary output curvature through affine parameter sharing

**Done on 2026-10-06 UTC.** Six modules add forty-six proved theorems
without new `sorry`. General feasible descent now applies to arbitrary
finite observation tables. An additional explicit architectural restriction
removes intrinsic flat segments of ordinary squared output error, without
an added geometry criterion or supervised attention routes.

[MatrixOutputError.lean](src/Transformer/GPTMini/Sparsemax/MatrixOutputError.lean)
defines only sum squared error against ordinary vector answers. It proves
zero-error equivalence, strict output curvature, the exact affine gap,
exact target-segment decay and a constrained output-distance growth bound.
There is no averaging, factor one half or embedding penalty.

[PeriodicMemoryDescent.lean](src/Transformer/GPTMini/Sparsemax/PeriodicMemoryDescent.lean)
connects those facts to the actual masked sparsemax/common-value forward.
For any convex output criterion and any better feasible joint endpoint,
all positive segment times up to one give strict descent. Every feasible
constrained local minimum is global, without an exact-fit or fixed-support
assumption. A feasible exact fit gives exact `(1-time)^2` ordinary-error
decay for arbitrary output dimension. Independent edge freedom at fixed
outputs remains in this untied chart.

[OutputTiedMemory.lean](src/Transformer/GPTMini/Sparsemax/OutputTiedMemory.lean)
intersects the true joint domain with the affine equality
`t[e] = offset[e] + gain*(Z[e,channel]+Z[e+1,channel])`. Z is the learned
output-coordinate table, not a supplied attention target. The fixed offset,
gain and channel implement parameter sharing before optimization. This
preserves joint convexity and the original physical width-three Q/K,
variational sparsemax and globally decoded common values. With every
registered prototype observed, actual predictions determine all intrinsic
parameters. Independently adjustable norm/frame variables are absent from
this canonical physical chart.

[OutputTiedMemoryExamples.lean](src/Transformer/GPTMini/Sparsemax/OutputTiedMemoryExamples.lean)
uses fixed offset and gain both 1/24 on the same earlier domain: floor 3/4,
edge budget 1/8 and energy 6. Ordinary answers `(1,1,-2)` and `(-2,1,1)`
remain exact fits with opposite nonidentity attention supports. Both physical
queries and keys change, and original common values remain nonconstant and
learned. The same domain contains both distinct points and their midpoint;
the midpoint has both local edges 1/16. The restriction is not a singleton.

[OutputTiedMemoryBounds.lean](src/Transformer/GPTMini/Sparsemax/OutputTiedMemoryBounds.lean)
proves the adjacent affine readout's squared Lipschitz bound `4*gain^2`,
independently of dictionary size. Each output coordinate occurs in at most
two adjacent edges. Full intrinsic edge/output squared distance is at most
`(1+4*gain^2)` times output-table squared distance. This distance is used
only to state guarantees and is never added to the optimized criterion.

[OutputTiedMemoryCurvature.lean](src/Transformer/GPTMini/Sparsemax/OutputTiedMemoryCurvature.lean)
proves strict convexity of ordinary squared prediction error in every tied
intrinsic direction for arbitrary answer tables, including unattainable
ones. Its quantitative affine gap controls full coordinate distance with
the same size-independent factor. Every constrained local minimum is global
and every attained minimum is unique in all intrinsic learned coordinates.
Thus intrinsic affine flat segments are removed in this restricted block,
rather than only separated at two special exact-fit examples.

Full prototype observation, categorical encoding, structural local mask,
strict self-weight floor and the fixed affine parameter-sharing rule are
explicit restrictions. The output criterion is ordinary squared regression;
no claim of strict curvature for arbitrary convex criteria or free physical
Q/K/value parameterizations is made. The rule restricts attainable answers,
and its language-model usefulness is unproved. Unobserved prototype freedoms
are not covered. Minimum attainment for arbitrary targets and quantitative
descent to an attained minimum are the next extension. Prototype records
and values still grow with P; FFN and new model training remain deferred.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Attained training and complete parameter guarantees

**Done on 2026-10-06 UTC.** Five modules add thirty-one proved theorems
without new `sorry`. Arbitrary ordinary answer tables now have an attained
unique minimum on every nonempty tied domain. Exact fitting and a supplied
optimal point are not assumptions of the final guarantees.

[PeriodicEnergyTopology.lean](src/Transformer/GPTMini/Sparsemax/PeriodicEnergyTopology.lean)
proves closedness of the actual joint energy domain and its affine
parameter-sharing restriction. Local score coefficients, the auxiliary
attention/output PSD block and the affine readout are continuous. Ordinary
prediction error is continuous on the genuine inverse domain through the
proved actual forward identity; continuity at singular inverses is not
assumed.

[PeriodicEnergyCompact.lean](src/Transformer/GPTMini/Sparsemax/PeriodicEnergyCompact.lean)
derives `Z[i,d]^2 <= energy` from a principal two-by-two PSD minor and the
attention diagonal bound. The incident edge bounds and these output bounds
make every finite joint domain compact. Any continuous ordinary output
error attains a minimum on a nonempty domain, for arbitrary finite coded
observations and target answers.

[OutputTiedMemoryMinimum.lean](src/Transformer/GPTMini/Sparsemax/OutputTiedMemoryMinimum.lean)
combines compactness with strict ordinary-loss convexity to prove existence
and uniqueness in all intrinsic trained coordinates. A noncomputable
selector records the proved point, its feasibility, its true attention
inverse and its optimality. The earlier nonsingleton domain also gives an
unattainable zero answer table with a proved positive-error minimum; the
two nonconstant exact-fit targets retain their unique learned solutions.

[OutputTiedMemoryGrowth.lean](src/Transformer/GPTMini/Sparsemax/OutputTiedMemoryGrowth.lean)
compares all feasible interpolation times to obtain the sharp bound
`output_distance^2 <= loss - minimum_loss`. Full intrinsic edge/output
squared distance is at most `(1+4*gain^2)*(loss-minimum_loss)`, independently
of prototype count. Epsilon-suboptimal training therefore controls all
intrinsic coordinates with that same constant. These are consequences of
the ordinary criterion, without an added parameter penalty or an exact-fit
assumption.

[OutputTiedMemoryDescent.lean](src/Transformer/GPTMini/Sparsemax/OutputTiedMemoryDescent.lean)
proves that the feasible midpoint toward the attained minimum decreases
ordinary error by at least full squared parameter distance divided by
`4*(1+4*gain^2)`. Every other feasible point strictly descends toward the
proved minimum at every positive segment time up to one. The statements
allow sparse support changes and positive optimal error; the optimal
endpoint is proved to exist rather than supplied as an oracle premise.
The noncomputable selector is not a numerical optimization algorithm.

The guarantees retain the explicit local mask, categorical input codes,
full prototype observation, fixed affine edge/output sharing, strict
self-weight floor and energy constraints. They remove intrinsic affine
flat segments and nonglobal constrained minima throughout this restricted
constant-width block. They do not establish convexity for independently
free physical Q/K/value tables, a language-model criterion or an FFN.
Physical Q/K width is three for arbitrary P; stored prototype records,
values and nearest search still grow with P. A numerical solver, useful
Shakespeare generalization and new model training remain deferred.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Constant-size generated original common values

**Done on 2026-10-06 UTC.** Eight modules add fifty-seven proved theorems
without new `sorry`. A new restricted jointly learned block has two
coefficient rows per output channel for arbitrary prototype count. Original
common values are evaluated from a generated function; no per-prototype
value or fixed basis rows are stored in its mathematical representation.

[AffineValueProfile.lean](src/Transformer/GPTMini/Sparsemax/AffineValueProfile.lean)
generates all local path edges from one scalar as `c*(e+1)*(N+1-e)`.
The actual path score action preserves constants and affine position at
both boundaries, giving eigenvalues 1 and `1-2*c` on these two features.

[AffineValueBasis.lean](src/Transformer/GPTMini/Sparsemax/AffineValueBasis.lean)
generates normalized positions in `[-1,1]` and affine responses from a
`2 x D` coefficient table W. Endpoint observations recover all coefficients
and bound their squared distance by half the full output distance,
independently of dictionary size. The feature is a position formula,
rather than a supplied dictionary-sized basis matrix.

[AffineValueDecoder.lean](src/Transformer/GPTMini/Sparsemax/AffineValueDecoder.lean)
proves equality to the original global inverse value decoder: the constant
coefficient stays unchanged, and the slope is divided by `1-2*c`.
The strict self-weight floor bounds that denominator below by
`2*floor-1`. Thus the exact original value at slot i is
`W[0] + position(i)*W[1]/(1-2*c)`, requiring only two coefficient rows.
The actual variational sparsemax forward gives the affine response exactly.

[AffineValueFeasibility.lean](src/Transformer/GPTMini/Sparsemax/AffineValueFeasibility.lean)
derives the full incident mass formula. Normalizing `c=rho/(N+2)^2`
makes `0 <= rho <= 1-floor` sufficient for every local budget, without
a learned edge vector. Genuine bounded width-three Q/K therefore remain
valid for every dictionary size. A query needs values only at its at most
three possible local destinations; the other original value rows need not
be materialized for that query.

[CompactAffineMemory.lean](src/Transformer/GPTMini/Sparsemax/CompactAffineMemory.lean)
ties rho to one learned slope coefficient by a fixed affine readout.
The entire trained state is W, with `2*D` scalar coefficients; geometry and
values are both generated from it. Coefficient caps and the scalar interval
give a convex compact domain that is itself independent of prototype count.
This explicitly replaces the earlier global PSD value-energy constraint,
using coefficient bounds and sufficient local feasibility budgets instead.
Both opposite nonconstant response witnesses are feasible on one domain.

[CompactAffineTraining.lean](src/Transformer/GPTMini/Sparsemax/CompactAffineTraining.lean)
connects actual predictions to ordinary sum squared output error. For
arbitrary full prototype answer tables, error is strictly convex in all
learned coefficients. The exact affine gap controls coefficient distance
with factor two, independently of prototype count. There are no route
labels, additional geometric criterion or independently free frame/norm
coordinates. Exact fitting holds for the feasible generated response family.

[CompactAffineMinimum.lean](src/Transformer/GPTMini/Sparsemax/CompactAffineMinimum.lean)
proves an attained unique minimum on every nonempty domain for arbitrary
targets. Every constrained local minimum is global. Ordinary suboptimality
bounds complete coefficient squared distance with constant one; a midpoint
lowers error by at least half that distance. Every different feasible point
strictly descends toward the minimum at every positive segment time up to
one. Existence does not assume an exact fit or a supplied minimum.

[CompactAffineCapacity.lean](src/Transformer/GPTMini/Sparsemax/CompactAffineCapacity.lean)
proves that universal exact fitting through a fixed coefficient-affine
scalar decoder requires at least P scalar coefficients. The compact
two-feature response family therefore trades arbitrary memorization for
constant parameter storage. Answers `(0,1,0)` have sharp best ordinary
error `2/3`, attained by a genuine feasible compact point. Separate physical
witnesses prove that both Q and K change and actual sparsemax support goes
from zero to positive on a local edge. Original common values are nonconstant
and genuinely inverse-adjusted, rather than frozen or equal to coefficients.

Only original value and learned geometry storage are constant here. Stored
prototype records and nearest search still grow with P. The positional
response family, categorical input encoding, fixed local mask, fixed scalar
readout, coefficient caps and full observation are explicit restrictions.
No numerical implementation, useful language-model criterion, FFN or
Shakespeare generalization is established. New model training remains deferred.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Expressive causal content encoding and joint sparsemax memory

**Done on 2026-10-06 UTC.** Eleven modules add seventy-eight proved
theorems without new `sorry`. The new encoder preserves observed token
content, order, repetitions and prefix length. The new block learns
coefficients of selected nonlinear content interactions instead of only
constant and affine prototype-position responses.

[CausalContentBits.lean](src/Transformer/GPTMini/Sparsemax/CausalContentBits.lean)
encodes each position's presence and B fixed binary token bits. For a
vocabulary of size V, `V <= 2^B` makes the canonical binary code injective.
The prefix representation is lossless on visible tokens, and a future token
change leaves it unchanged. The bit dimension is `r = T*(B+1)` for a maximum
length T. Basis recall's current 548-token vocabulary fits in ten token bits;
these are data codes, not supervised attention destinations.

[CubeContentBits.lean](src/Transformer/GPTMini/Sparsemax/CubeContentBits.lean)
and [CubeContentFeatures.lean](src/Transformer/GPTMini/Sparsemax/CubeContentFeatures.lean)
generate sign products `chi_S(x) = product(i in S, (-1)^x[i])` for selected
coordinate subsets S. They include constants, individual bits, bit matching
and interactions across positions; a selected high-order product represents
parity. The response is `Phi_W(x,d) = sum(k, chi_S[k](x)*W[k,d])`, with K*D
learned coefficients and fixed feature descriptions.

[CubeContentRouting.lean](src/Transformer/GPTMini/Sparsemax/CubeContentRouting.lean)
uses generated virtual bit states rather than stored text prototypes.
Nonnegative flip masses t have total at most `1-floor`. A query routes to
itself with mass `1-sum(t)` and to the state with bit i flipped with mass
`t[i]`. This action has eigenvalue `lambda_S = 1-2*sum(i in S, t[i])` on
each selected feature, uniformly bounded below by `2*floor-1 > 0` when
`floor > 1/2`.

[CubeContentQK.lean](src/Transformer/GPTMini/Sparsemax/CubeContentQK.lean)
and [CubeContentAttention.lean](src/Transformer/GPTMini/Sparsemax/CubeContentAttention.lean)
realize those weights through genuine Q/K dot products of width r+1, a fixed
structural local mask with outside score -1, and the original variational
sparsemax operator. The normalization threshold is zero. At most r+1
generated destinations can receive positive mass among `2^r` virtual states;
supports may change during training. This is content-addressed virtual
memory, not GPTMini attention over input-token occurrences. The causal
guarantee comes from encoding only the observed prefix.

[CubeContentValues.lean](src/Transformer/GPTMini/Sparsemax/CubeContentValues.lean)
generates original common values as
`V(x,d) = sum(k, chi_S[k](x)*W[k,d]/lambda_S[k])`. Actual sparsemax times
these values equals `Phi_W` exactly; that equality is proved from the
physical forward, not used as its definition. The inverse factor is at
most `1/(2*floor-1)`. A query evaluates only its generated local destinations.
No stored per-prototype values, bit-state feature table or nearest-prototype
search is needed in this representation. Besides K*D learned coefficients,
the fixed selected subsets and readout maps also require storage.

[CubeContentLearning.lean](src/Transformer/GPTMini/Sparsemax/CubeContentLearning.lean)
sets `t[i](W) = offset[i] + gain[i]*W[pick[i],channel]`. The coefficient caps
and route budgets define a convex joint domain. Both Q and K change, and
original values are learned through their actual inverse-adjusted formula.
The genuine block forward is affine in every learned coefficient throughout
the domain, including across support changes. Convexity is in this shared
state W; independently free physical Q/K/value matrices are not covered.

[CubeContentBudget.lean](src/Transformer/GPTMini/Sparsemax/CubeContentBudget.lean)
normalizes every offset and gain to `(1-floor)/(2*r)`. For `r > 0` and
`1/2 < floor <= 1`, the complete domain is exactly the nonempty compact box
`-1 <= W[k,d] <= 1`. Coefficient clipping therefore suffices for the real
arithmetic feasibility and value-denominator guarantees. No dictionary-sized
PSD constraint or learned routing table is required. Optimizer convergence
and floating-point accuracy are not established.

[CubeContentExpressivity.lean](src/Transformer/GPTMini/Sparsemax/CubeContentExpressivity.lean)
proves that a single two-bit interaction fits XOR sign, while a constant
plus individual-bit response cannot. The full subset family spans every
finite bit-state response through an explicit kernel inverse. For targets
in [-1,1], all resulting coefficients lie in the unit box, so the jointly
learned physical sparsemax/common-value block has a feasible exact fit.
Full capacity needs `2^r*D` coefficients. A universal coefficient-linear
scalar decoder needs at least `2^r` coefficients; selecting K features trades
that capacity for K*D learned storage rather than evading this bound.

[CausalContentBlock.lean](src/Transformer/GPTMini/Sparsemax/CausalContentBlock.lean)
connects the physical block to causal token data. It proves future
independence at any parameter assignment, joint forward affinity on the
convex domain and convexity of every supplied convex criterion in the block
outputs. Ordinary squared prediction error supplies a nonconstant example.
It does not assert strict curvature or remove freedoms from unobserved or
linearly dependent selected features.

The fixed binary token encoding, selected feature identities, affine readout
maps, local virtual-memory mask and strict self-weight floor are explicit
restrictions. Q/K width grows with the prefix bit dimension, and the full
interaction family is exponential. Neither a small selected family solving
all Basis tasks nor generalization to new text is proved. Free learned token
codes, normalization, a jointly trained layer stack, FFN and a language-model
head remain outside the guarantee. The Python port and a ConvexGPT versus
GPTMini-softmax Basis comparison have not started; new model training remains
deferred.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## ANSR training on ordinary softmax GPTMini

**In progress on 2026-10-06, explicitly requested by the user.** The user
provided https://github.com/fxeqxmulfx/ansr and requested a clone in `/tmp`
and Basis training on ordinary softmax GPTMini. This authorizes the present
empirical attempt before any further convex-block port. The reference is
pinned to 9cb98c12b3184368c80fea72f3b81432123d96dd.

`experiments/basis_ansr` keeps the small, easy Basis GPTMini and all three
tasks. Low and high p_self ANSR arms get 100,000 generations each (about
12.8 million loss evaluations); AdamW controls retain calibrated task budgets.
The model, data splits, seeds and batch
sizes are unchanged. Execution is eager CUDA. The ANSR port refreshes stored
attractor fitness on each new training batch and counts those evaluations,
preserves tied weights, and checkpoints its population and RNG. This is a
training attempt, not a claim that optimizer behavior proves convexity.

The nine 256-generation preparation/control runs completed and are archived
in `experiments/basis_ansr/pilot_summary.json` and `pilot_report.json`. AdamW
passed easy depth, recall and parity at 200, 1,700 and 4,800 updates. The short
ANSR runs do not settle its efficacy. The user explicitly required a much
larger budget; all six ANSR runs continue from checkpoint toward 100,000
generations each, without resetting their population or random generators.
The user then explicitly returned to Lean and convex training. The background
ANSR queue continued until the user explicitly stopped it on 2026-10-06 UTC.
Its verified depth checkpoint is generation 22,560; the remaining queue was
cancelled. Best validation loss was 0.5689167010 at generation 22,528. Checkpoints
and raw records are preserved; ANSR must not restart without a new user request.

## Freely learned query-key matching through a convex atomic state

**Done on 2026-10-06 UTC for the stated mathematical architecture.** Seven
modules add forty-six proved theorems without new `sorry`. The user's
correction is accepted: the previous selected bit-product response and its
inverse-adjusted values do not establish independently learned attention
matching. That construction is not sufficient evidence for the requested
ConvexGPT block; its Python port has not started.

The new architecture retains genuine attention over causal input-token
occurrences. An atom contains three independent learned tables: Q and K
of shape V*H and original values of shape V*D. Token identities are observed
inputs; every context uses the same tables of each head. All coordinates
are bounded by a numerical cap C. The complete eligible atom family ranges
over every such table triple, with no selected feature products, prescribed
routes, prototype bank or fixed sparsemax support. Effective Q/K token
embeddings are free; a shared factorization into E*W_Q and E*W_K is not asserted.

[AtomicMatchingHead.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingHead.lean)
uses literal Q(token_i) dot K(token_j), the repository's original variational
causal sparsemax, and its product with independent common original values.
It proves physical causality and output bounds. Changing both Q and K from
zero to one in the two-token witness changes the output from 1/2 to 1 while
the original values remain identical. No inverse decoder cancels matching.
Position features may be included in input identities; RoPE and normalization
are not implemented by this theorem.

[AtomicMatchingMixture.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingMixture.lean)
stores a finitely supported nonnegative distribution with total mass one
over the complete continuously parameterized atom family. Its domain is
convex: combining states may select previously absent Q/K/value heads.
The forward is the actual weighted sum of all active attention-head outputs.
Absorbing each coefficient into that head's original values gives ordinary
physical multihead evaluation. Convexity is in the distributional state,
not in independently interpolated raw matrices at a fixed head count.

[AtomicMatchingHull.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingHull.lean)
proves that this physical prediction class is exactly the convex hull of
all eligible heads' observed responses, reconstructing finite head families
in both directions. Convexity is not stipulated as the definition of the
physical forward. Head identities may be chosen from the data, while a
fixed finite interaction dictionary is not part of the architecture.

[AtomicMatchingCompression.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingCompression.lean)
proves that R observed scalar outputs have an equivalent feasible state
with at most R+1 active heads. Every criterion of those exact outputs is
preserved. Learned storage is at most `(R+1)*(2*V*H+V*D+1)` real scalars;
V is the input vocabulary and H,D are per-head dimensions. This counts the
stored Q/K/value tables and weights, not caches or a numerical optimizer.
If an attained optimum is supplied, there is an equally optimal compressed
state. Attainment for arbitrary continuous criteria is not proved here.
Compression concerns the chosen observations and may change unseen responses.

[AtomicMatchingOptimality.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingOptimality.lean)
proves convexity for every supplied convex output criterion. It evaluates
the true pricing objective `sum(r, g[r]*head_output[r])`. A supporting
functional g and a certified lower price c over every eligible head give
a global criterion-gap bound `pairing(g, current_output)-c`. The hypothesis
ranges over all Q/K/value triples, including heads outside the active support.
A nonzero-gradient squared-error example satisfies the complete certificate
and has a genuine constrained optimum. No efficient global pricing oracle
or convergence of raw Q/K gradient descent is claimed.

[AtomicMatchingSelection.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingSelection.lean)
proves that adding positive mass to a different matching head strictly
improves the ordinary squared answer error, with identical original values
and no route labels. The actual midpoint obtained by interpolating Q/K
of one raw head differs from the convex mixture midpoint. Atom pricing is
proved nonconvex even on an affine raw-Q/K path with unchanged values.
Any normalized number of repeated uniform heads remains suboptimal against
an eligible nonuniform head; restricting optimization to an unchanging bank loses the
full problem's guarantee.

[AtomicMatchingContext.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingContext.lean)
uses two contexts with the same token multiset in different orders. The
ordinary final-position answers differ. A zero-query uniform head has total
squared error at least 1/8 even with freely relearned keys and original
values; a constant value table attains this bound. A bounded genuine matching
head attains zero error. Every exact-fitting feasible mixture must select an
actual head that distinguishes the orders. Answer-only data can therefore
require effective matching even when values are trainable; no route labels
or inverse values are supplied.

The remaining computation is substantial: globally pricing a new genuine
head still contains nonlinear learned matching. The numerical atom box,
unit mixture mass and variable head count are explicit restrictions of this
new architecture; it is not an equivalence with the paper's four-matrix
weight decay or a fixed-width GPTMini. Multiple jointly trained nonlinear
layers, FFN, language-head loss, constant storage, arbitrary exact fitting,
generalization on new text and a successful Basis comparison are not proved.
Python is unchanged and new model training remains deferred.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, warnings in the new modules,
extra axioms, results resting on `sorry`, vacuous statements or placeholders.


## Attained convex atomic matching training

**Done on 2026-10-06 UTC for the bounded genuine-head architecture.** Three
modules add twenty-one proved theorems without new `sorry`. The user explicitly
returned to Lean and convex training. This removes the previous need to supply
an attained optimum before obtaining a finite optimal physical model.

[AtomicMatchingContinuity.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingContinuity.lean)
proves continuity of the repository's original variational causal sparsemax
at every score assignment, including changes of active support. Its optimality
graph is closed in the compact causal simplex. Genuine shared Q/K scores,
independent original values and their actual head outputs are continuous.
The complete numerical Q/K/value box is compact, and a nonnegative cap makes
it nonempty. No score-gap, fixed-support or attention-inverse premise is used.

[AtomicMatchingFamily.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingFamily.lean)
gives a compact N-slot chart of freely selected physical heads and probability
weights, with a continuous actual forward. Any feasible finite mixture with
at most N active heads has an exactly matching chart representation: unused
slots receive zero mass and a feasible zero head. Conversely, each valid chart
reconstructs a feasible finite distributional state. Slots are storage locations,
not a fixed interaction bank. The criterion in raw chart coordinates need not
be convex; convexity remains in the distributional state and observed outputs.

[AtomicMatchingAttainment.lean](src/Transformer/GPTMini/Sparsemax/AtomicMatchingAttainment.lean)
proves that the entire genuine observed prediction set is exactly the image
of the compact R+1-slot chart. It is therefore nonempty and compact for a
nonnegative cap. Every continuous criterion of R observed scalar outputs has
an attained global minimum with at most R+1 learned Q/K/value heads, including
criteria with unavoidable positive optimal error. Neither a fitting target
nor an optimal mixture is an existence hypothesis. The earlier convexity
theorem applies whenever the supplied output criterion is convex.

The true physical-head pricing objective is also continuous and attains a
minimum over every eligible Q/K/value table. A supporting output functional
therefore has an attained finite global objective-gap certificate; the pricing
bound is derived from an existing global minimizing head. This proves existence,
not an efficient global head search or convergence of a practical optimizer.
Fixed-width raw matrices, FFN training and unseen-text generalization remain
outside these statements. Numerical bounds and unit total mass are unchanged.

Validation: full `lake build`, `./run.py audit`, `./run.py index` and
`./run.py forbidden` pass. No new `sorry`, new-module warnings, extra axioms,
results resting on `sorry`, vacuous conclusions or placeholder definitions.
Python sources are unchanged; the previously passing 178-test suite applies.

## Numerical check of genuine convex atomic training

**Done on 2026-10-06 UTC.** The user explicitly requested an experiment of
the attained convex architecture. [convex_atomic](experiments/convex_atomic)
implements actual shared bounded Q/K/value heads, causal sparsemax and
probability mixtures, with no route targets, predefined head bank, FFN,
normalization, residual or output projection. Thirty-eight runs finished.

Exact physical-head pricing on the two Lean order contexts attains the
universal price lower bound. A small corrective convex simplex QP escapes
both the zero-Q/K state and a strictly one-route saturated state: error
0.125 falls to 2.0954e-31 at update two, with a numerical global gap bound
3.3307e-16. Each result persists through 1,000 updates and eight batch orders.
Raw AdamW stays at 0.125 on both starts. An out-of-box target reaches its
positive optimum two with gap zero, without assuming an exact fit.

The independent numerical head search also escapes both states: random
projected Q/K Adam and exact conditional original values reach error
1.1752e-21 after 128 outer updates, with a box gap bound 3.1196e-12. No
analytic order pricing is used there. The oracle remains nonconvex and
approximate; twelve active heads are retained without numerical compression.

Basis uses the unchanged easy data and answer loss, with 128 outer updates
and 66,560 pricing forwards per task. Best test sequence accuracies are
40.23% depth, 0% recall and 54.69% parity. The current encoder is content-only:
query token plus visible multiset determine every real response. Depth has
observed signature conflicts (test token-accuracy ceiling 83.44%). Two valid
MQAR inputs obtained by exchanging written values have different oracle
answers and identical model signatures/logits. This separates an encoder
capacity restriction from the earlier raw-gradient plateaus. Parity has no
such finite-split collision certificate. A fresh-batch corrective solver
does not establish convergence for the full training objective.

Validation: all 189 lab tests pass, including eleven new tests of physical
forward/causality, flat gradients, exact and approximate certificates,
positive minima, original values, synthetic answer training and checkpoint
continuation. Lean sources are unchanged from the checked attainment commit.
The user immediately authorized trying to repair the binding encoder; that
follow-up must retain these baseline results and convex mixture training.

## Preserve key/value binding in convex matching

**Encoder repair and experiment completed on 2026-10-06 UTC; full Basis
training remains unsolved.** Explicitly authorized by the user. The content-only
recall witness exchanges two written values while preserving the visible
multiset and query. Every old physical head and mixture is invariant to that
exchange. This is an encoder obstruction separate from sparsemax plateaus.

[convex_binding](experiments/convex_binding) introduces `PairedMatching`:
current-token queries and original values, with current-token and predecessor
roles in two freely learned key halves. No pair dictionary, route labels or
fixed head bank is supplied. Commit `4ef16e5` proves the actual causal forward,
box bounds, continuity, convex mixture criterion, attained head pricing,
the old sharp error floor and its repaired original-value witness. Compact
four-channel integer key codes give one-head exact recall with total width
eight and cap four under a unique matching-predecessor premise.
Commit `a155424` also proves that any feasible mixture fitting the swapped
answer labels contains an actual head that distinguishes those bindings.

All 18 runs finished. The four paired exact and four paired numerical scalar
runs reach squared error below `1.5e-29`; both four-seed content controls
remain at the sharp floor `0.5`. On easy Basis, both width-eight models
finish 512 outer updates and 266,240 physical pricing forwards. Their test
token accuracies are 0.415% (content) and 0.732% (paired), with zero sequence
accuracy and identical logits on the binding-swap witness in both cases.
A separately constructed feasible head at the same width and cap attains
100% on both held-out splits; it is not used in learning or pricing.

The archived ordinary softmax GPTMini reaches the 99% validation sequence
criterion in 1,650–2,450 AdamW updates and 207–287 CPU seconds over three
seeds, against 1,622/1,843 CUDA seconds for the column models. Split hashes
match exactly. At update 500 it still has zero sequence accuracy, with
11.06–11.72% validation token accuracy. Architectures differ and equal
FLOPs have not been measured. Detailed results and qualifications are in
[the experiment report](experiments/convex_binding/README.md).

Validation passes: 196 Python tests, all 18 experiment definitions, and
unchanged lab-source hashes for every run. Lean passes the full build,
audit, index and forbidden checks with 157 existing `sorry`, zero rests,
zero extra axioms, zero vacuous claims and zero placeholders.

The structural encoder obstruction is removed; numerical pricing and
fresh-batch correction still require investigation before claiming a
trained convex Basis model. Keep all prior baseline results. ANSR remains
stopped by the user's request.

## Drop-in joint embedding/attention scope

**Status: In progress; compact admissible block not yet constructed.**
The user's 2026-10-06 requirements supersede the sparsemax-specific search:
first make embedding plus self-attention jointly convex, and consider only
replacements of the existing blocks that keep the training optimizer.
Keep the ordinary forward/backward/update loop and the existing data,
token-to-stream shapes, FFN and downstream loss. Changing the attention
operator or its internal parameterization is allowed. Keep ANSR stopped.

The guarantee must concern the jointly trained embedding/attention
parameters. Convex inference at fixed scores, convexity in network inputs,
separate convexity of two trained modules, and a convex outer mixture with
nonconvex head pricing do not meet this requirement. A feasible direct
parameter update must not require an SDP solver, rank projection, atomic
column optimizer, or inner search for a new head. Keeping AdamW does not
by itself supply a convergence theorem, even for a convex objective.

The first algebraic control is the complete unfactorized interaction tensor
in `GPTMini.Convex.JointInteraction`, `JointInteractionFactorization` and
`JointInteractionBoundary`. A free residual embedding plus coefficients
`C[query token, predecessor key token, current value token, channel]` maps
tokens to a causal residual stream. The forward is linear in all these
parameters and every convex output criterion remains jointly convex.
Ordinary answer squared error fits the two swapped binding tables exactly;
its convexity and actual global minimum are proved, with no route targets.

This control changes softmax/sparsemax to a signed unnormalized interaction
sum and absorbs all learned embedding/projection/value products into C.
It is a mathematical candidate for replacing the combined token-to-stream
prefix, not a wired replacement of each individual Python module. Every
tensor has an exact representation by V² width-one dot-product heads with
original-token values; one shared width-V embedding realizes the recovered
tables. None of these width, head-count or penalty choices preserves the
original fixed-small GPTMini parameter class. At the actual Basis recall
vocabulary V=548 and output width 64, the naive full tensor stores
10,532,261,888 scalars, about 42.1 GB for float32 parameters alone, before
gradients or AdamW states. This is the complete tensor's size, not a lower
bound for every restricted-data or compact replacement.

Requiring that tensor to come from one original head is not a convex
compression, even if the head's Q/K width is unrestricted: two attainable
query/value diagonal tensors have a midpoint with nonzero rank-one minor.
This counterexample rules out that particular compression only. It does
not rule out a different compact operator or parameterization.

The new `InteractionObservations` and `InteractionCompression` modules
strengthen the scope of that storage result. All true final-row outputs on
two- and three-token contexts linearly identify every tensor coordinate and
the free residual embedding. If an affine prediction family covers every
width-one physical head of this changed unnormalized operator on all these
contexts, its parameter dimension is at least V³D. The proof does not assume
that the family explicitly stores a tensor: recover the coefficients from
its actual predictions, then use the original heads' full tensor span.
Thus exact affine compression of this complete class is ruled out. This is
not a bound for a finite training sample, normalized softmax, a restricted
matching class, or arbitrary nonlinear parameterizations. The separate
residual embedding has VD further observable coordinates.

Next, search for a compact, unconstrained joint parameterization with the
existing module interfaces, ordinary optimizer and learned content matching.
Check that joint embedding training preserves its convexity; a fixed-input
attention proof is insufficient. Fixed tangent features would provide an
approximation control, not the requested unrestricted learned matching.
Finite selected head/feature banks and SDP drafts remain outside the active
drop-in route. No new training result is claimed from the tensor control,
and nonlinear downstream layers or the tied readout can still destroy
convexity of the complete model.
The affine forward is a sufficient control for every convex output
criterion, not a necessary condition for a particular cross-entropy loss.
Continue investigating compact nonlinear probabilities with convex
negative log likelihoods, checking simultaneous embedding, matching and
value training rather than coordinatewise output convexity alone.

The new `GPTMini.Convex.Likelihood` modules provide a constructive nonlinear
control. Two unrestricted real utilities define three sequential Bernoulli
outcome probabilities. They are strictly positive and normalize exactly.
Their log probabilities are actual categorical logits; unchanged log-sum-exp
cross entropy equals a sum of softplus terms and is jointly convex in both
utilities, including any joint affine parameterization of those utilities.
A strict midpoint gap in a class-logit difference proves that this is not
an affine logit model disguised by a common shift. It has no token encoder,
Q/K matching or learned attention values and is not a drop-in block.

A separate necessary condition covers nonlinear probabilities: if every
observed-category negative log likelihood is convex, any two parameter
assignments with identical normalized positive predictions must retain that
prediction on their entire segment. All likelihoods are flat along such a
fiber. Consequently an even family on an unrestricted parameter space must
be prediction-constant. This concerns a guarantee for all possible labels,
not convexity of one particular dataset's aggregate loss. Replacing one
utility by a free scalar query times a free scalar key already produces an
explicit ordinary-CE midpoint counterexample. A new normalizer by itself
does not remove that factorization's opposite-parameter symmetry.

Keep the nonlinear route active only with an admissible matching/value
parameterization that passes these tests and the existing module interfaces.
An unused outcome that absorbs probability mass could bypass normalization
over observed categories and encode a curvature penalty, but it requires a
changed output head; no compatible embedding/attention-only construction
has been found. Do not present that idea as an admitted drop-in prototype.
No optimizer was changed, no new experiment was run, and ANSR remains stopped.

The new GPTMini.Convex.Reparameterization modules test the coordinate
escape on an unchanged physical softmax head class. Two query rows share
three keys, scalar Q/K factors are unrestricted, and fixed one-hot values
expose the attention probabilities. Every physical head has a rank-one
log-odds matrix against the third category, hence zero two-by-two minor.
One head predicts (1/2, 1/4, 1/4) and a uniform second row; another predicts
a uniform first row and (1/4, 1/2, 1/4). Convex likelihoods for every category
would require each midpoint entry to dominate its endpoint geometric mean.
For every positive normalized table meeting these inequalities, rational
probability bounds force a strictly positive log-odds minor. The arithmetic
mean witnesses consistency of those hypotheses, but no such table belongs
to the original scalar-head class.

Thus no parameterization into this class covering these two actual heads
has globally convex negative log likelihoods for every category. The map
may be nonlinear, redundant, and of arbitrary dimension; no continuity,
smoothness, injectivity, or affine-logit assumption is used. This rules out
coordinate changes for this finite class, beyond the original Q/K sign
symmetry. It is not a theorem about all head widths, joint learned values,
a full causal GPTMini, or one dataset's aggregate loss. The control uses
log probabilities as logits; its ordinary cross-entropy identity is proved,
without identifying that readout with the model's learned downstream head.

There is also a constructive finite completion. Two unconstrained
sequential Bernoulli utilities per query row (four scalars total) attain
every positive normalized two-by-three probability table, including every
original scalar head and the arithmetic midpoint outside that class.
All category likelihoods and the explicit log-probability cross entropies
are jointly convex. Therefore enlarging the operator's prediction class
can bypass the obstruction without increasing the parameter count in this
tiny example. Its utilities are independent per query, category decision
order and one-hot values are fixed, and a direct generalization grows with
the number of query/category pairs. It is a finite mathematical control,
not an admissible shared token encoder, learned-value block, or drop-in
replacement. Continue seeking a compact changed operator with shared
learned matching and values; do not treat this completion as that result.

## Basis integer-list formalization (2026-10-06)

At the user's request, `Transformer.Basis` now gives depth E_2/E_4,
easy/hard raw MQAR and 1..16-bit parity the common `List Int -> List Int`
answer-append interface. Every call preserves the input and increases its
length by one. Depth uses the independently proved alternating-subsequence
criterion for E_k, including neutral symbols. MQAR parses actual adjacent
key/value tokens, validates its query/filler region and selects the latest
write. Parity requires BOS/bits/SEP and generates the label and EOS in two
successive calls. The integer IDs, mode parameters, vocabulary sizes and
context caps come from the current Basis generators; invalid inputs use an
explicit PAD fallback. The validated prefix domains contain the sampled
supervised prefixes but omit the sampler's distribution-specific constraints.
Depth/recall labels are appended only for this mathematical interface: they
do not replace the benchmark's following random input tokens.

`Transformer.GPTMini.TokenInterface` wraps the actual existing Lean stack,
checks integer-to-Fin vocabulary conversion and the Python context cap,
uses positions 0..T-1, and greedily decodes the last-position logits. The
softmax and logit decoders agree, including the smallest-ID tie rule. The
task oracle is not called by the model function. Exact maximum/tie and
sufficient strict-margin certificates connect this forward to the raw task
labels. Every certificate hypothesis has a satisfiable example, including
explicit weights in the small two-layer, four-head, width-64 architecture.
Those control weights correctly reject BOS,a and fail on BOS,a,b, so a
passing example does not become a whole-task capacity theorem.

The user's clarified target is verification of the required model
properties, rather than an existence theorem for weights. Full-stack
causality is now proved for every parameter assignment, including the
masked denominator, all head reshapes, RoPE, QKNorm, XSA, residuals, FFN
and final tied readout. Arbitrary future tokens preserve all old logits,
also across different sequence lengths. Teacher-forced row evaluation and
the List Int prefix continuation therefore predict exactly the same token.

All validated task prefixes are proved nonempty, within the task context,
and encodable in the actual vocabulary. Given a compatible model shape,
`solvesTask_iff_logit_property` characterizes full correctness exactly by
its actual maximum/tie conditions on every supervised prefix. No weight
existence or assumed whole-task success is used in this equivalence.
Independent necessary answer distinctions require depth order, adjacent
MQAR binding, chronological overwrite order and parity bit content;
bag-invariant next-answer computations provably fail actual easy recall.
These semantic requirements are stated for the real model too, but their
satisfaction by an unrestricted parameter assignment does not follow
from causality. The control model proves that implication false.

`SolvesTask` covers every validated supervised prefix, stronger than the
sampled benchmark's 99 percent stopping rule. There are no new sorrys.
The adapter inherits the existing Lean stack's shared epsilon, whereas
Python defaults use 1e-5 for RMSNorm and 1e-6 for QKNorm/XSA. General
floating-point equality and AdamW convergence are not certified. No
experiment or optimizer was changed, and ANSR remains stopped.

## Basis internal semantic guarantees (2026-10-06)

The next formalization step is `Transformer.GPTMini.Semantics`. Raw MQAR
parser inversion identifies the last matching adjacent write, including
earlier overwrites and unrelated later records. For an actual retrieval
head, a post-RoPE score gap bounds the omitted softmax mass and the value
error. The complete XSA head has error at most
`2 * ((T - 1) * exp(-gap) * B + eta)` when the target is orthogonal to the
normalized self-value, value diameter is at most B, and the selected value
has encoding error at most eta. Approximate key copying reduces an ideal
matching gap by at most `4 * exp(alpha) / eps * eta`. These are conditional
operator guarantees; they do not assume correct final task logits.

Internal head and FFN codes propagate through the actual head merge, W_o
and both residuals. A maximal-norm answer embedding separated from every
competitor by delta decodes correctly when the final state error is less
than `scale * delta / 2`, for positive scale and epsilon. The final-block
theorem derives the full original model's List Int continuation from these
local codes and their combined error budget. Correct logits are a
consequence, rather than a premise. Examples realize every hypothesis.

For parity, explicit parameters in the original 64-wide, two-layer,
four-head, 68-token model compute raw ONE counts from the actual embeddings,
RMSNorm and fused QKV. The count reaches the actual first residual block.
The ONE and ZERO one-bit inputs have provably distinct first hidden states
at every positive epsilon. A single further parameter assignment computes
the count alongside a protected phase coordinate with a nonzero attention
output projection. Its two legal six-token contexts needing EVEN and EOS
have equal count signals and opposite phase signals. Local kernel equations
for W_o and W_out preserve that phase through every layer of the real
stack. The explicit controls verify internal features; their final
EVEN/ODD/EOS decoder is not a full-task correctness result.

For depth, an actual finite-softmax head detects ordered subsequences under
a faithful earlier-prefix feature encoding, and distinguishes raw aabb
from abab. Positivity and a QKNorm-dependent signal floor are proved for
arbitrary Q/K arrays and learned finite temperatures. The earlier-prefix
encoder and raw adjacent-key copying/routing for every Basis prefix remain
unproved model invariants. Consequently the work does not yet establish
complete Basis correctness or learned-checkpoint correctness. All new
results have proofs, with no additional sorrys. No optimizer, experiment
or running training job was changed.

## Abandoned schedule pair

The frozen constant/cosine schedule pair is incomplete and will not be
resumed. Trainer PID 168237 (session 11503) and archive worker PID 168332
(session 60201), stopped with `SIGSTOP` at 07:48:53 UTC, were terminated with
`SIGKILL` at 08:03 UTC on 2026-10-03 by the user's decision, before the Python
code moved out of `experiments/`. Neither process ran again after the pause,
so the raw state below is exactly the paused state.

The constant case's last canonical observation is update 288,000 and its last
complete gradient record is 288,250. Its latest canonical train and held-out
accuracies are 100%, but failures at 274,000 and 275,500 already violate the
strict final-window condition. Only 153 of 201 final-window checks exist.
The durable checkpoint is update 285,000, SHA-256
`14706a5662e124119fef1caf882c6adae1ba8b8a8dfd7ed5cf14e6080a371380`,
rechecked after termination. The cosine case never started. Raw state is under
`experiments/archive/synthetic_trainers/runs/adamw_stability_20261002/schedule_mod193_fraction25_lr0003_budget300k/`
(`experiments/runs/` until 2026-10-03).
The incomplete pair supports no conclusion about either schedule. Its frozen
plan pins sources at their `experiments/` paths; replaying it requires a
checkout of commit `698d904`.

## Mod 97 size search (2026-10-11)

Status: active, requested by the user. Search the dense GPTMini first in
`experiments/mod97_size_search`, then Mesh GPT in the Rust repository. Every
run is capped at 10,000 complete passes over its fixed training split. Use
exhaustive held-out evaluation, record delayed grokking separately from
immediate generalization and transient peaks, and commit tested setup and
result milestones promptly. Preserve the archived runs and unrelated work.

Dense stage complete: seven fresh 10,000-epoch runs. The smallest tested
delayed-grokking model has width 32, two layers, four heads and 32,232
parameters with learning rate 0.001 and weight decay 0.2. Generalization
began at epoch 7,675 and was confirmed at 8,150; final training and held-out
accuracy are both 100%, and the final 1,000-epoch window stays above 99%.
Widths 24 and 16 with the same decay failed within the cap. The Mesh stage
is active; retain the fixed architecture-specific data and report the
vocabulary, loss and split differences. See the experiment README and the
Rust repository's portable histories for the complete outcomes.

## Rotate90 learning transfer (2026-10-31)

Status: completed, requested by the user; board #119. After the finite sampled
alias-escape proof, test the gap between static character capacity and actual
learning. `experiments/rotate90_transfer` declares 15 fresh Schedule-Free AdamW
runs at P17/97/193, widths16/64, depth1 and initialization seeds31/37/41, keeping
data and batch seeds fixed at11. Train 1500 updates with CUDA max-autotune;
retain initial and every50-update model/optimizer/sampler state using the new
`CheckpointHistory` block. Ordinary checkpoint recipes retain their fingerprints.

Before training, freeze the exact evaluation-update identity, next-step heldout
loss sign test, step400 extrapolation and matched width test in hypotheses.json.
Replay independently with all parameter derivatives and the original RMS epsilon.
Do not feed heldout gradients to training. Formalize actual joint analyticity,
the source optimizer's evaluation displacement and conditional finite-time
loss/accuracy bounds. Report every failed forecast. Sampled remainders are not
segment curvature certificates; these tests cannot establish random-start
convergence or a success probability.

Outcome: all 15 runs finished. The six P97/P193 width64 runs reach 99.87--99.97%
heldout accuracy; width16 remains at 1.96--17.93% after 1500 updates. P17 fits
training perfectly and reaches only 7.35--9.56% heldout accuracy. All 120 source
update identities and next-step loss signs pass. The frozen early forecast
predicts only one of six successes; its sufficient CE threshold is conservative,
and the three P193 large runs accelerate after weak early progress. The matched
early width hypothesis fails (three of six pairs, below the four-pair threshold).
Seventeen Lean theorems now connect all actual parameters and the source
optimizer to conditional finite-time loss/classification bounds. The unresolved
step is deriving sustained favorable transfer and useful curvature bounds from
random initialization and the observed training sample. See the study README
for all seeds, false negatives and numerical limitations.

## Rotate90 optimizer memory (2026-10-31)

Status: completed, requested by the user; board #120. Extend the completed transfer
study by separating its dominant averaging term from new gradient-driven base
updates. Existing seed31 x/y/z exploration is disclosed in memory_exploration.json:
high-accuracy evaluation parameters can coexist with poor base parameters.

The frozen `experiments/rotate90_transfer/memory_hypotheses.json` selects all 15
sources at updates 200/400/750. Preserve each source model, second moment, base,
scalar history and sampler. Check 32 actual zero-rate Schedule-Free updates against
the exact averaging formula, then evaluate its declared longer horizons using
compiled CUDA max-autotune. Include all failures, P17 and narrow controls. Test
the full-network sign symmetry and its zero midpoint independently. No new fit,
optimizer or tuned forecast is introduced. Prove the source clock and the actual
finite-network averaging obstruction in Lean; distinguish it from typical
random-start learning. Retain the two original transfer-study failures unchanged.

Outcome: setup `daab04d6` froze all controls before their evaluation. All 45
source identities match within 6.661e-16; bases stay fixed and second moments
change. All 45 evaluation vectors move despite zero rate. All six successful
wide checkpoints initially improve loss, then fall from 99.76--99.98% to
4.06--7.28% accuracy at the declared 50000-update closed-form point. All 45
negation controls have identical logits and tied midpoints. Narrow/P17 controls
remain unsuccessful. The 27 checked Lean theorems prove the exact memory
clock, convergence of actual predictions/loss to the saved base, and a
nonconvex whole-table solver region. Remaining: explain and predict the
favorable path generated by continued positive-rate training from random
starts. The stopped-base control does not isolate gradient from decay updates.

## Rotate90 gradient forcing (2026-10-31)

Status: completed; board #121, setup `9f769888`. The completed memory test
stopped both gradient and decay movement. Separate them with all fifteen
transfer-study states at update 400. Thirty declared full-state continuations
retain all optimizer and sampler history and run to absolute update 1500:
original decay 1 and decay 0, using only Schedule-Free AdamW. Fixed-horizon
zero-gradient decay and zero-rate diagnostics accompany every source. The
definitions and predictions are in `experiments/rotate90_forcing` and must be
committed before observing these outcomes. Preserve narrow/P17 controls and
every failed prediction. Prove the actual-source linear response and its
initial-span constraint without assuming generalization from new directions.

Outcome: all thirty continuations and all fixed diagnostics completed. All
six wide P97/P193 no-decay continuations exceed 99%, while their zero-gradient
controls finish at 2.95--5.15%. Independent actual-source recurrence errors
are at most 1.11e-15 over 32 steps. All fifteen full trajectories leave their
initial parameter span, including unsuccessful narrow/P17 controls, so span
escape is insufficient to predict learning. H1 replay fails on all three
P17 controls; the twelve P97/P193 cases pass. All source histories and matched
samples are preserved. The unchanged retry of one CUDA graph-capture failure
is retained with the initial attempt. H2--H5 pass their declared checks.
Twenty-seven Lean theorems prove the finite actual-model response and
zero-gradient span invariance. The remaining gap is to derive useful
normalized-gradient directions and their accumulation from sampled data;
neither fresh-start no-decay success nor general convergence is established.

## General positional inputs and normalized attention (2026-10-31)

Status: completed theory and source validation; requested by the user; board
#122. Extend beyond fixed quarter turns and two variables, with arbitrary
positional mixers and a concrete attention model. The user supplied diff_gpt
at `b98c665b`; retain its embedding/final RMSNorm, Peri-RMSNorm on attention
and SwiGLU branches, Q/K RMSNorm, arbitrary rotary phases, causal softmax, XSA
clipping and explicit first-position zero. Source hashes and numerical checks
are in `review/sequence_model_source*.json`.

Outcome: the source-response theory now accepts arbitrary input types and
full finite predictors. Lean proves arbitrary-angle rotation geometry, the
relative-position law, exact recovery of the current two-input Rotate90
forward, and an impossibility example specific to repeated-quarter-turn
pooling of five variables. Full coordinate coverage and normalization bounds
are proved for the supplied attention architecture, including the bound
exp(2*sqrt(head_width))/visible_positions on one attention weight. Three
CPU float64 and one compiled CUDA source comparisons pass for all logits and
trainable gradients. This is a formal-model extension and correspondence
check, not a new learning experiment or a convergence/generalization claim.

## Scalar MLP on mod 97 (2026-10-18)

Status: active, requested by the user. The runs are done (#56,
2026-10-18): every gradient-descent run ends in H4, which chooses no Lean
step. Board tasks: #54 landed the lab blocks, #55 the experiment, #56 ran
it, and #57 cites M8.3, which #52 proved. The Lean side is M8 of
`landscape_plan.md`, tasks #50 to #53.

A perceptron reading x and y as two real numbers and answering with one
represents x / y mod 97 exactly at depth 2 and width 2p + 1 = 195 (`lookup`
in `python/tests/test_regression.py` builds the weights), yet gradient
descent does not find such weights. Ask why, by experiment, before any Lean
step.

A perceptron's loss has a Lipschitz gradient on bounded sets only. M8 asks
instead for a condition that a run of gradient descent with step η checks
along its own orbit x_k:

> (S) ‖∇²f‖ ≤ L on every segment [x_k, x_{k+1}], with η L < 1.

Proved (M8.1): under (S) the loss decreases along the orbit
(`Transformer.Landscape.SaddleAvoidance.iterate_succ_le_sub_of_segment`).
Proved (M8.3): for an analytic loss, which satisfies the Łojasiewicz
inequality at every point
(`Transformer.Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticOnNhd`),
almost every start whose orbit satisfies (S), under any distribution of
starts with a density, has an orbit that converges to a critical point
that is not a strict saddle, or ‖x_k‖ → ∞
(`Transformer.Landscape.SaddleAvoidance.ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd`).
A GELU perceptron's loss is analytic, though Lean does not prove it. A run
that does not fit the table therefore ends in one of four ways:

- H1, a spurious minimum or a degenerate saddle: it converges with positive
  loss. The gradient norm falls to near 0, and λ_min ≥ 0 up to the Lanczos
  error. For gradient descent under (S) this is the first alternative of
  M8.3, with a limit of positive loss.
- H2, slowness: it still descends when the budget ends, with negative
  final-decade slopes. M8.3 leaves this open, for it bounds no time. Under
  (S), M8.4 bounds the loss above the limit by C b^k or
  C (k + 1)^(-1/(2a - 1)), by the Łojasiewicz exponent a there
  (`Transformer.Landscape.SaddleAvoidance.exists_le_mul_pow_of_isLojasiewiczAt_of_segment`,
  `Transformer.Landscape.SaddleAvoidance.exists_le_mul_rpow_of_isLojasiewiczAt_of_segment`),
  with constants it does not give. The residual's share at high
  frequencies of the transform over Z/97 x Z/97 shows where the fit lags:
  x / y keeps most of its energy there and x + y little.
- H3, divergence: the parameter norm grows without bound. For gradient
  descent under (S) this is the second alternative of M8.3, ‖x_k‖ → ∞.
- H4, a step past the curvature: (S), the hypothesis of M8.3, fails,
  η·max(|λ_min|, λ_max) ≥ 1 at a recorded update of a gradient-descent run.
  The runs sample it every 1,000 updates, not on every segment.

AdamW runs are controls: M8.3 concerns gradient descent, so for them H1 to
H3 name only what a run does, and H4 does not apply.

[`experiments/mod97_scalar_mlp`](experiments/mod97_scalar_mlp/README.md)
fits two-hidden-layer perceptrons to the whole table with full batches:
gradient descent at a constant step of about half the inverse initial
sharpness, and AdamW as a control. It compares scalar with one-hot operands
and x / y with x + y, and varies the width (64, 1024) of the scalar
division run. Every perceptron uses the analytic GELU; under ReLU the loss
has no second derivative on its kinks, so the theorems do not apply. Each
observation records the gradient norm, the extreme Hessian eigenvalues, the
parameter norm and the residual's high-frequency shares. For every
gradient-descent run the report adds η·max(|λ_min|, λ_max) at every
observation, its maximum, and the first update at which it reaches 1.

The outcome chooses the Lean step:

- H1: a trap theorem, that a non-global local minimizer attracts starts of
  positive measure, applied where a spurious minimum is proved.
- H2: a lower bound on the loss of gradient descent on a quadratic model by
  its spectrum, with the flat spectrum of division.
- H3: the second alternative of M8.3; no landscape theorem says more.
- H4: none; M8 says nothing where (S) fails.

Outcome (#56, 2026-10-18): every gradient-descent run ends in H4. In each,
η·max(|λ_min|, λ_max) first reaches 1 between update 1,000 and 28,000 and
never falls below it again: λ_max climbs to about 2/η and stays there. No
gradient-descent run fits its table, and the list chooses H4: none. AdamW
fits three of the four tables; on scalar division its parameter norm grows,
H3, while its loss falls slowly. The numbers are in the Findings of the
experiment's README.

The explicit representation can itself be proved in Lean as a statement
about two-hidden-layer ReLU networks on a finite grid.


## Residual transformer comparison — task145, 2026-11-01

Active target: determine whether the verified query-coordinate repair improves
full-transformer generalization. The user authorized up to **two new GPU hours**,
including profiling, compilation, evaluation and continuations, and requested
early completion if generalization is stable. The old two-hour cohort is closed.
Use Python3.15, Torch2.15, max-autotune and Schedule-Free AdamW only. Preserve
Q/K normalization, centered XSA, its constant-prefix guard and all source
Peri/RoPE/SwiGLU/tied-readout branches in every comparison.

Implement ordinary addition, fixed branch scaling, paper key-only AttnRes,
raw Q/K depth attention and complete-state conditioned Q/K depth attention.
Freeze seeds, cases, data hashes, source hashes, ordered runtime allocation
and validation stopping before learning. Reserve a disjoint final test set;
never select a checkpoint or stop from its scores. A stable validation stop
keeps the current endpoint. Report incomplete runs as budget-censored.
Source equation/gradient checks and save/resume tests precede GPU learning.
The same assistant will review the final commits and measured evidence.

User follow-up for task145: any architectural fix or generalization improvement
found in this comparison must get a regression test in the diff_gpt repository.
A numerical/equation test alone must not be described as evidence of improved
held-out learning. Keep the useful failure mechanism and learning protocol in
the test or its explicitly slow integration fixture.


## Normalization and residual gradient benchmark — task151, 2026-11-02

The user asks whether another residual extension, gradient path, or normalization
may interfere with grokking, and explicitly requests a new synthetic benchmark
inspired by this Lean project. Keep ordinary residual addition as the working
transformer baseline while task145 closes. A candidate extension is one bounded
learned strength per completed branch, `alpha = 2*sigmoid(s)`, initialized at
`s=0` so the initial function is the ordinary residual model. Place this strength
after the existing branch output RMSNorm. Its identity path is not a proof of
well-conditioned gradients or improved generalization.

Task151 starts after the frozen task145 cohort, replay, and self-review. Its
small mechanism probes and learning runs use CPU, Python3.15/Torch2.15 and
Schedule-Free AdamW; the existing GPU deadline is not renewed. Freeze the
concrete cohort after independent formula checks and disposable compilation
profiling, before any scored learning. CPU learning should use max-autotune.

Mathematical anchors already inspected: `GPTMini.rmsNorm_pos_homog` removes
positive scaling for the ideal zero-epsilon norm; `GPTMini.rmsNormEps_lipschitz`
gives an upper sensitivity bound with positive epsilon, not a lower gradient
bound; `Residual.residualRun` and `residualRun_unroll` retain the actual branch
recursion. The complete coordinate-state identities in `AttentionResiduals`
also do not imply convergence. The refreshed discrtree index still reports
unbuilt newer modules, so read the relevant statements in their sources and
refresh after any required Lean build.

Proposed gate-placement probe: supply independent Rademacher branch directions
(their RMS is exactly one), and fit known coefficients in a residual sum.
Compare a bounded scalar gate before versus after finite-epsilon branch RMSNorm.
Both variants start with the same outputs and both have explicit finite optimal
gates. Sweep branch scale and branch count with paired data and batch seeds.
For a unit-RMS direction, the normalized coefficient is
`a/sqrt(a*a+eps)` and its derivative is `eps/(a*a+eps)^(3/2)`.
A gate before normalization can have a tiny radial derivative even while the
function is representable. A gate after normalization directly controls branch
amplitude. Check scalar derivatives with high-precision independent arithmetic,
FP64 autograd, and actual compiled FP32 gradients; record precision loss as such.
Use zero weight decay in this mechanism isolation so shrinking a large parameter
is not mistaken for learning from a useful gradient. This is still Schedule-Free
AdamW, with a disclosed recipe distinct from task145.

Proposed depth probe: frozen linear residual branches give a known transport
Jacobian `(1+c)^depth`. Contrast repeated contraction with depth-scaled branches;
match the initial input-output function and use an unbounded scalar readout so
both families can represent the same teacher. Measure parameter scale, true
transport derivatives, finite-precision update size, and coefficient recovery.
A tiny unnormalized MSE is not success when the target signal is tiny: score
relative error and the recovered coefficient. Include a depth-one control where
the compared transports coincide. This isolates a mechanism and does not model
the full source transformer's Peri/RoPE/XSA computation.

All learning cases keep disjoint train/validation/test directions and full
trajectories. Final test opens only after learning closes. Use coefficient error
as well as held-out prediction error; the known coefficient oracle allows an
all-input error bound through `Residual.weightedSum_norm_le`. Do not describe
these synthetic regressions as evidence of improved modular CE grokking. Any
verified architecture/placement repair must get a meaningful diff_gpt regression
that rejects a broken control, followed by same-assistant review.


Task145 measured outcome (2026-11-02): all four ordinary residual controls
passed stable validation and scored99.96–100% on final test. Raw/conditioned
Q/K depth mixing stayed near chance within37–40k updates; one conditioned P193
case failed at initial CUDA capture. Query conditioning never activated.
Fixed scaling at23k updates improved paired validation from34.97%/27.23% to
97.60%/98.37%; final test was97.77%/98.37%, still short of the stable99.9% target.
One key-only control reached20k at2.79% test; the other was globally censored
during setup. No continuation fit. Keep these censoring limits in any claim.

All449,000 gradient records are finite, and CPU/CUDA counts and fixed-epsilon
FP32/FP64 predictions agree. Local normalization gains occur in successful
models too; this does not establish a general normalization defect. The new
known-solution task151 must distinguish coefficient/coordinate conditioning
from missing features and actual modular generalization. Both prior GPU phase
budgets are now closed. GPT5d71786 adds two held-out regression fixtures for
scaling, with a disabled-scaling control falling to36.21%/43.77%.


Task151 measured outcome (2026-11-02): all 36 CPU cases reached 10,000 updates
(360,000 total), in 171.11 scored seconds. Post-RMS bounded gates recovered
12/12 cases; pre-RMS gates recovered 4/12, only at scale .001. Depth-scaled
transport recovered 6/6; ordinary contracting transport recovered 2/6 (depth
one). Known optima and independent derivatives establish equal representability.
FP32 radial cancellation and depth-dependent parameter conditioning explain
the controlled failures; depth-eight ordinary transport still moves and is
not an impossibility result. Nine Lean statements in Residual.GradientPaths
formalize the exact mechanisms without a convergence claim. GPT379a9f2 adds
an optional post-RMS gain and meaningful recovery/negative-control tests.
The next full-transformer learning comparison needs a new explicit GPU budget;
the previous phase remains closed. Preserve ordinary and fixed-scale controls,
QK norms, source RoPE/XSA/Peri branches, paired seeds and long continuations.


## Repeated gradient search — task158, 2026-11-02

The user explicitly requested a loop: identify a remaining gradient/normalization
limitation, check it with a known-solution synthetic task inspired by Lean, and
integrate an evidenced improvement directly into the actual diff_gpt model with
regression tests and no architecture flag. Rejected candidates remain recorded.
Do the review in this same assistant, without delegation.

The first completed pass was task155's corrected finite-epsilon RMS backward.
Task156 removed the old architecture switches; current production GPT is
`df1e9f7`. Every new full-model comparison must include this corrected fixed-scale
baseline, not the older native-backward baseline.

Next hypothesis: a learned input-dependent gain *after* branch RMS can expose
conditional amplitude gradients which normalization attenuates inside a branch.
Use gains `2*sigmoid(W_gate * RMS(branch_input))`, with zero gate weights and no
bias. Multiply the existing `1/sqrt(2L)` coefficient by that gain. Initial forward
values and common parameter gradients must match the current model exactly.
This extends the scalar task151 mechanism to conditional amplitude; it does not
assume a transformer grokking improvement.

First freeze a supplied-feature benchmark with two independent gate features
(constant 1 and an independent random sign) and analytic conditional target coefficients.
Both before-RMS and after-RMS parameterizations must have explicit finite optima,
the same initial outputs, parameter count, data, batch order and Schedule-Free
AdamW recipe. Use the corrected RMS backward in both so the benchmark cannot
rediscover the already repaired native cancellation. Sweep branch scale and
branch count; retain full held-out recovery, per-condition coefficient errors,
independent derivatives and compiled-state continuation checks.

Only after those checks, compare the actual fixed baseline and the
candidate architecture on modular division, matched common initialization and
splits. Source snapshots/hashes must pin both; no runtime architecture toggle
enters production. The user's new stop instruction is **first 99% validation for
each promising transformer run**; report final test independently. The previous
two-hour budgets are closed and are not reused. A synthetic win alone does not
prove faster grokking or justify calling the whole model improved. Keep the
current production architecture if full-model evidence fails to support a change.

Other candidate mechanisms must first be checked against the real forward pass:
Q/K temperature calibration before a post-attention RMS may simply move a gain
that the output RMS then attenuates. A synthetic attention-only benchmark would
not establish that this fixes the complete transformer. Repeated-token length
collisions remain task149, with their clipping guard; do not silently turn that
conditional source invariant into an unconditional impossibility claim.

Task158 synthetic outcome: all24 cases completed10,000 updates. Conditional
post-RMS gains recovered12/12; pre-RMS gains recovered4/12 (scale.001 only),
despite explicit finite optima for both. The largest post-RMS coefficient error
was3.52e-6. The failure at larger scales is conditioning within the measured
updates, not an expressivity impossibility or a refutation of a flat-spectrum
rank theorem. Full transformer validation remains necessary.

Before GPU learning, an actual-size source audit found a last-bit backward
difference in the historical lab attention port: it shares the centered value
subtraction where GPT computes it twice. Its logits matched exactly, but seed11
had maximum FP32 gradient difference1.49e-8 (seed17:7.45e-9). A preflight using
GPT's exact subtraction/first-position assignment order removed both differences.
Preserve the already-scored synthetic source pin; commit that evidence before
updating the new canonical port. Archive the unused initial transformer protocol
as unscored, then freeze the source-exact GPU comparison. Do not mutate a frozen
protocol or treat this numerical audit as a learning result.

The synthetic evidence and initial unused protocol are committed at `63c2d22a`.
The source-exact GPU protocol extends the comparison to P97 and P193, with
two model seeds per prime and reversed variant order in each second pair.
Before any GPU outcomes, promotion requires an earlier first99% crossing in
all four pairs. Record per-parameter gradient/update diagnostics every1,000
steps and a10-second Python3.15 parent-owned sampling window per warmed case.
No system tracing-policy changes are needed. The final test remains sealed
until the entire learning phase closes.

The arithmetic cohort has now closed: canonical/candidate first crossings
are26k/21k and24k/20k on P97, and8k/8k and15k/13k on P193. All eight reserved
final tests exceed99%; those results did not select the architecture. Three
wins and a tie do not satisfy the frozen all-four-earlier promotion rule.
Production remains GPT `df1e9f7`; retain the actual tested candidate as a
replayable patch. The engine pin stays at598979b6. Runner amendment23d16a5e
fixes only a one-ULP CPU/CUDA metadata diagnostic mismatch before P193
training and preserves exact split hashes and the original protocol.


## User-requested Basis confirmation — task160

The user explicitly requested testing improvement on Basis during task158.
Keep the active P97/P193 source pin at598979b6 intact and prepare this work in
an isolated checkout. The arithmetic cohort and its final-test boundary stay
unchanged. A modular-arithmetic gain alone will not be described as a broad
model improvement; the production integration decision also needs this check.

Compare CanonicalResidual and AdaptiveResidual in the standard diagonal Basis
settings: small/easy (width64, depth2) and large/hard (width128, depth6), four
heads, all three tasks in each setting, model seeds0/1. This is24 runs, not the
full crossed-size historical matrix. Preserve the original task generators,
20,000 training rows,512 validation/test rows, data seed1, batch order convention,
training batches, task learning rates, warmup50 and3,200-example observation
cadence. Use only Schedule-Free AdamW (betas .9/.98, decay .1), Python3.15,
Torch2.15, and CUDA fullgraph max-autotune with math attention. Evaluate in
batches64 to fit the4GiB GPU; disclose this execution deviation from historical
Basis. No comparison to old AdamW wall times is causal.

Use exactly Basis sequence accuracy, including free generation for parity;
for hard depth, the stopping distribution is length128. Stop at the first99%
canonical validation crossing and retain that endpoint. Final test and easy
length-transfer test remain sealed until learning closes. Report all cases,
including numerical failures and unresolved continuation checkpoints. Initial
100,000-update chunks are not proof of impossibility; continue promising runs.
Original Basis samples can repeat inputs across splits, especially short parity
strings: report overlap/novel-input test results separately without changing the
original distributions or using them to select a model.

Register the source-GPT synthetic adapter as a proper lab block and retain the
legacy Synthetic/GPTMini behavior. Check exact original split fingerprints,
full-logit versus supervised-position loss/gradients, free-generation semantics,
complete Schedule-Free continuation and source correspondence. Keep the model
candidate flag-free. Review this work in the same assistant.

The completed cohort has21 first99% crossings and three explicit hard-recall
censorings. Conditional gains improve large parity in both seeds but slow hard
depth, small recall and small parity. Canonical hard-recall seed1 required a
long continuation:99.02% validation at123,550 updates and98.44% reserved test.
Its extension resumed the complete102,750-update checkpoint after interruption;
the fixed200,000-update horizon and all original learning sources were preserved.

Every endpoint and650,300 finite nonzero gradient records are audited. CPU
replay agrees with all canonical validation prediction counts. Parity's original
sampled split contains repeated training inputs: only184/512 test prompts are
novel. Large-parity novel-test scores are96.74%/97.83% for the controls and
92.93%/100% for gains. Easy-depth length128 transfer is also mixed across seeds.
No architecture is promoted; final tests were opened only after learning closed.

Task162 is the urgent next architecture comparison: original GPTMini under
Schedule-Free AdamW on the unchanged hard-recall problem. Historical AdamW
successes already establish that the Basis task was solved; they do not locate
the cause of today's architecture/optimizer gap. Keep model seeds, batches,
rates and first99% stopping matched, and reserve new final tests. If the gap
persists, isolate one architectural change before any flag-free production fix.
Task161 remains the subsequent diagnosis of conditional-gain transfer.

Task162's concrete control is `experiments/basis_control`: historical GPTMini,
width128/depth6/four heads, model seeds0/1, unchanged original hard-recall data
and task160's exact Schedule-Free recipe. The initial boundary is100,000 updates,
with first99% sequence stopping and complete-state continuation for promising
cases. Current-model references are the already closed task160 endpoints,
not rerun or reselected using final test. The control keeps the original
optimized readout and has nearly the same parameter count. Freeze all sources,
runtime, data and reference hashes before updates; no component-level causal
claim or production change follows from this whole-architecture control alone.

Task162's isolated follow-up is `experiments/attention_temperature`: keep the
current canonical architecture and every common initial parameter, and add
one zero-initialized log-gain per head after Q/K normalization and RoPE. The
positive gain multiplies queries entering SDPA. This adds 24 parameters without
consuming randomness, changing epsilon, or moving residual normalization.
The two seeds use task160's exact hard-recall data and Schedule-Free recipe.
The complete frozen sources precede GPU learning. First99% validation selects
the endpoint; unresolved 100,000-update boundaries require explicit decisions.
Final tests stay sealed until both cases close. Two hard-recall runs are a
screen, not sufficient evidence for a production replacement. A useful result
requires broader Basis and modular-division confirmation plus direct flag-free
GPT tests. Larger temperatures in both successful and unsuccessful historical
controls motivate this comparison but do not establish a cause.

The matched historical GPTMini cohort has closed: seed0 crosses first99% at
12,200 updates (97.27% reserved-test exact sequences), while seed1 is explicitly
censored at 100,000 with no exact validation sequences and 11.25% token
accuracy. Its last 50,000 updates remain flat. The full state is retained;
this validation-only censoring does not rule out later grokking. Both the
historical and current architectures solve one of the two matched seeds, with
different successful seeds and unequal continuation horizons. All 112,200
gradient records and CPU/CUDA endpoint prediction counts pass the independent
audit. The separately frozen learned-temperature screen addresses one
component; historical AdamW success is not a causal normalization diagnosis.

The active user goal explicitly requires `(a / b)^K mod P` as well as Basis.
Task164 records that requirement so a K=1-only arithmetic success cannot stand
in for the requested power family. Its declared matrix will include P97/P193,
K1/K2/K5 and model seeds11/17, with b nonzero and independent label/oracle
checks including zero inputs and K0. A candidate must first survive the
isolated screen; broader training, length/novel-input checks and direct GPT
tests precede any flag-free production update. The current model is retained
when a candidate regresses. Task161's separate gate diagnosis reads closed
states and does not alter either active or historical learning trajectories.

Task161's endpoint probes and finite-solution width oracle do not justify a
repair: hard-depth gate saturation is seed-dependent, and inference removal
of learned conditioning harms the already learned function. The original
per-parameter norms are available, but intermediate weights and gradient
vectors are not. Task165 records the needed matched repeats with full state
history; all outcomes and differences from historical CUDA trajectories must
remain visible. These diagnostics do not alter task162's temperature screen.

The later cost policy closes task162's temperature screen and defers task165.
Task165's five untrained draft sources and focused-test evidence are preserved
under `.kanban/artifacts/165/`; the archive was verified byte-for-byte before
removing the worktree. The board task is released with an explicit deferred
title; its CLI retains the unclaimed in-progress status.

Task166 derives the actual Schedule-Free working-weight increment before
reading one next training batch from each of six closed hard-recall states.
The declared common averaging-opposition signature is absent in all three
zero-exact-validation endpoints, globally and for every parameter tensor.
The constant-direction and reversal witnesses distinguish averaging dynamics
from the fresh-gradient coefficient. Four first-order predictions exceed the
nonnegative training loss itself, including a successful control; they cannot
be used as exact next-loss predictions or grounds for a larger-rate sweep.
See [the diagnosis](review/schedule_free_directions.md). No research training
or new final-test prediction is made, and production GPT stays unchanged.

## Convex answer-CE recall comparison (2026-11-11)

**Status: completed all 24 fixed-budget native runs; seven native checks pass.**
The user requested a comparison of the newly proved convex matching model
with a usual transformer on MQAR and `a / b mod p`, after the Lean proof
`0fcc0d1b`. `experiments/convex_recall_compare/README.md` fixes three seeds,
8-key/16-value MQAR, division mod 17/97, endpoints and reporting rules before
training. It reuses the validated native Kernite transformer engine with
longer attention scratch buffers. No Python is used.

All three seeds of matching and the transformer finish at 100% MQAR test
accuracy. Matching first shows 100% at step 5 and exact identity at step 10;
the transformer first shows 100% at steps 1000/2000/1000. Division copy-only
adaptations finish at mean test accuracy 9.07% (mod 17) and 1.62% (mod 97),
versus 11.76% and 16.26% for the width-32 transformer at 20,000 updates.
Independent convex answer tables fit every training pair but retain uniform
held-out logits. The transformer has not solved division in this pilot;
mod 17 is memorized and mod 97 remains poorly fit even on training data.
No endpoint was extended or selected by test results. Generated evidence is
preserved in the root experiment's ignored runs directory.

This tests the narrower fixed-copy architecture proved in
`Grokking.FlatSpectrum.ConvexRecall`. Its explicit copy adaptation and a
separate convex table control expose why bare arithmetic is different from
supplied-dictionary recall. It does not resolve the earlier compact joint
embedding/attention replacement requirement or claim that its hypotheses are
satisfied. No answer dictionary is provided to the arithmetic forward pass.

## Convex generated channels and flat-spectrum bounds (2026-11-11)

**Status: Lean construction landed at `dbb3a4cb`; all 18 fixed native runs complete.**
The user requested extending the convex model, then clarified to use flat spectrum
as a bound rather than an arithmetic-specific architecture. The joint tensor
learns routing and output classes from categorical bindings with ordinary CE.
Lean proves the whole feasible domain and loss convex, strict extension of
copying, identification from separating answer examples, necessary rank/error
bounds on actual outputs, a flat ambiguity counterexample and an exact finite
linear oracle. The full build and axiom audits pass without new unfinished claims.

The authorized native follow-up is `experiments/convex_channels`: three seeds of
ordinary and recoded four-key MQAR, comparing the new channel, fixed copying and
the existing conventional Kernite transformer. The README fixes all 18 endpoints,
data, optimizer budgets and reporting before training. A full finite spectral
diagnostic checks the sharp rank/error bound. This does not reopen the unrelated
compact joint-embedding drop-in requirement or claim generalization from flatness
alone. No arithmetic feature map is introduced.

Every channel and transformer run reaches 100% held-out accuracy on both tasks;
fixed copying reaches 100% on ordinary MQAR and 0% on recoded outputs. The channel
recovers the exact parameter witness after one full-batch update in every seed.
The saved tensor has zero gap, constraint violation and full-domain spectral
error on all 4096 dictionaries. Removing target rank directions gives exactly
the proved `4096(4-d)` squared-error bound. Five native tests, all 18 endpoint
checks and re-evaluation of saved channels pass. Confidence scales, batch sizes
and update budgets differ, so no equal-budget speed comparison is claimed.

## Forward-only versus iterative convex channels on MQAR (2026-11-11)

**Status: all six comparisons complete; replay and native checks pass.** The user requested
testing the learner proved at `a7c246fe` against the usual convex channel.
`experiments/convex_channel_forward/README.md` fixes the same ordinary and
recoded MQAR data for seeds 0,1,2, exact correctness checks, and repeated
single-CPU timings. The primary baseline is one Frank-Wolfe update because
the prior experiment already showed exact recovery there. The original
1000-update endpoint checks consistency; it is not used to inflate a speedup.
A losslessly compacted baseline separates lookup representation from training
method. Native Rust only; no changes to the existing frozen experiment.

Both methods learn exactly the same channel, with 100% accuracy on training,
held-out and all 4096 dictionaries in every case; the 1000-update endpoint
is unchanged. The direct learner fits in 17.21-17.85 us versus 468.89-473.57 us
for one usual FW update (26.38-27.30x faster). Lookup inference is about 21x
faster than the dense channel, but exact compaction gives the usual channel
the same lookup speed and 288-byte payload. This distinguishes the measured
training benefit from the representation benefit shared by both methods.
All timing observations and saved models remain local and ignored.

## Convex channels on the GPT short/long stress suite (2026-11-11)

**Status: all 210 cells complete; saved-outcome replay passes.**
The user requested the existing short/long suite and explicitly added the
forward-only learner. `experiments/convex_stress/protocol.json` fixes both
arms, all 35 tasks and seeds 0,1,2, the shared causal adapter, CE optimizer
budget, direct consistency search, complete-sequence metrics and explicit
nonrealizability outcomes. Original data generators and fingerprints are
copied unchanged from diff-gpt `3555299`. The full-batch observation schedule
and optimizer differ from the original transformer; no matched speed or
promotion claim is made. No feature tuning or budget extensions after runs.
All generated evidence remains local and ignored under the root's `runs/`.

Both learners pass 9/75 short and 0/30 long cells. The 87 realizable cells
per arm fit training exactly, yet most fail the held-out threshold. The
direct learner explicitly rejects 18 inconsistent cells; iterative training
continues and retains every numerical stop or exhausted budget. Short recall
is 86.13% accurate; long recall is 91.82% per token and 50% per sequence.
Saved models confirm correct lookup routes but missing query/value decoder
records. A separate minibatch comparison was requested after these runs;
the full-batch cohort remains frozen.

## Persistent forward-only minibatches on MQAR (2026-11-11)

**Status: Lean equivalence landed at `e9b0260c`; all 42 native comparisons and replay pass.**
The user requested checking minibatch support. CF6 retains every candidate's
record across batches and compares the exact computed result with the original
complete-data learner. `experiments/convex_channel_minibatch/README.md` fixes seven
batch sizes across the same six MQAR task/seed combinations, prefix and late
conflict checks, and paired CPU timing. The prior forward/FW experiment is
unchanged. Only the same ordered examples are assumed by the Lean equivalence.

Every batch size learns the identical full-data model, with 100% accuracy on
training, held-out and all 4096 dictionaries. All 2316 prefix snapshots agree
with full-prefix fitting. Single-dictionary batches cost 8.9-13.3% more than
the original forward fit; larger batches are close in speed. Persistent state
uses 2064 bytes, independent of data count; the emitted model uses 288 bytes.
All six native tests, formatting, clippy and source/data replay pass. Complete
timing observations and model evidence remain local and ignored.

## Convex stress suite with original minibatches (2026-11-11)

**Status: 204 outcomes completed and replayed; six remaining runs stopped by user.**
The user additionally requested minibatches. The separate
`experiments/convex_stress_minibatch/protocol.json` uses all 35 original
stress tasks, three seeds and both learners. Frank-Wolfe updates the joint
tensor on every original minibatch, retaining unvisited queries. Forward-only
keeps every candidate's state across boundaries and completes one epoch,
then compares the entire tensor or explicit failure with complete fitting.
CM1 proves feasibility of independent query steps; CF6 proves persistent
batch equality. Neither claims stochastic convergence or unseen-label
identification. Original maximum updates and observation cadence are kept;
all endpoint source splits and explicit failures must replay. The earlier
full-batch cohort stays frozen. A separate posthoc branch-amplitude witness
shows that current lag features can classify that task perfectly, despite
higher unit-scale CE than the v1 trained model; no benchmark arm uses these
hand-written task-specific weights.

The user stopped the remaining near-zero runs on 2026-11-11: three long-recall
seeds at 0/512 exact sequences and three recurrence seeds at 53/4704. All 210
planned cells remain in the partial comparison; interrupted cells have no
endpoint score. All 105 forward outcomes exactly match complete-data fitting.
Both arms pass 9/75 short cells; forward passes 0/30 long, and FW has 0/24
completed long passes plus six interrupted cells. The original source fingerprint
and full-budget protocol remain frozen. `report_stopped.sh` and `compare.jq`
regenerate the ignored evidence without dropping interrupted cells.

Task298 producer `448f034d` finishes all35 original cases and35replays with
zero process failures. Both policies retain35/35primary/auxiliary success in
all modes. Five-gate Fibonacci CE improves to0.13366full/0.13222mini/0.13013forward,
but raw errors change109to105,103to107,and100to105. Prior block-local results
remain in full, including their better raw counts and worse CE. Work is288
gradient batches,176updates and797local-row updates; forward has zero gradients.
Peak replay RSS is3.02GiB under64GiB. Authored findings retain these tradeoffs;
independent complete reproduction at the findings commit precedes approval.

Task298 is approved at `06c19178`: independent complete preparation,35cases,
35replays and all116JSON artifacts/report match exactly. Task299 formalizes
convex raw-margin constraints around a fixed reference before a new native
model. Positive retention preserves strict comparisons and first-index argmax,
while a maximizing reference remains feasible and ordinary CE stays convex.
Nonzero Fourier witnesses retain wrong labels and expose the zero-retention
tie failure. This stabilizes predictions; it does not prove unknown correctness
or a numerical CE gain. A separate native implementation/comparison follows.

Task261's frozen CM35 readout comparison completes all35 original cases and
35replays with zero failures. Both policies retain35/35full,35/35mini and
34/35forward primary/auxiliary successes, with zero replacements. All21
available readout endpoints pass; they cover7tasks in each mode, with3774
refused outcomes and all1265stages retained. Fibonacci's standalone field/raw
counts improve by one but observed rawCE ties preserve the earlier selected
parent. The sole forward associative-divisor failure remains53.866%. These
older branch results stay distinct from later all-mode35/35 context models.
Exact-commit complete replay/summary review precedes the task267 barrier.

Task300 implements CM47 after its Lean review: the fixed block-local CM46
reference and a half-scale copy precede the existing91context maps. Five
source-excluded count gates use radius0.5 and retain half of every actual raw
margin over every eligible finite input/class. Every accepted capped segment
is verified with complete complex products and the exact metric argmax;
final stored inference rows reproduce the certificate hash. Full64steps,
mini8original-order epochs and forward0gradients retain original head
derivatives and budgets. Ten native fixtures pass; mini/forward noisy-fixture
diagnostic CE falls0.325069508to0.302528484 with no raw prediction changes.
Exact-commit component review precedes separate original evaluation, which
must retain296raw and298CE references and test pooled field metrics too.

Task300 passed exact-commit review at70fb09bb. Task301 prepares the separate
original35-task comparison against296, retaining298as a complete secondary
reference. All endpoints are sealed sequentially to disk, released, then
reconstructed individually for scoring after the complete seal. This keeps
large temporary guard caches within the64GiB address limit. Replay reserves
allthree construction passes in its conservative work forecast. Original
primary success, raw accuracy and CE remain separate, with every inherited
baseline and cost retained. No new original numerical result is claimed yet.

Task301 v1 attempted all35 cases:34runs and34replays pass; Fibonacci stops
after sealing forward because full/mini work forecasts are9.0-11.4T, above
the8T cap. PeakRSS is6.53GiB, with no out-of-memory event and no new original
held-out scores. Task302 ports the frozen numerical model into a separate
package with an inspectable16Twork limit; the memory and cost formulas stay
unchanged. Both frozen forward prior/supplement trajectories reproduce exactly
apart from the protocol digest. Review precedes a separately frozen full rerun;
the failed v1 sources,binary,model,logs and all35attempts remain preserved.

Task301's separate margin_budget_compare rerun binds component26af04ec.
It preserves the failed-v1 manifest as a third independent history input,
reconstructing all35oldprocess statuses and the partial forward model before
new preparation. Every old file is checked; missing replay after refusal is
explicit. The16Tcomponent cap changes no model trajectory or memory limit.
The new source,binary andrequest will be frozen separately before original
metrics, with no reinterpretation of the failed v1 as a completed comparison.

Task301 v2 finishes all 35 original cases and 35 replays with exit0 at frozen
producer `e971b722`. Both policies retain 35/35 primary and auxiliary successes
in every mode. Fibonacci raw errors stay 101/100/100 while CE improves from
0.172615/0.175106/0.177280 to 0.170831/0.173737/0.176289. The complete task298
reference retains its lower CE and higher raw errors. All-query actual margin
checks preserve every reference raw decision; they cannot repair its remaining
mistakes. Work is 288 gradient batches, 164 updates and 770 local-row updates,
with zero new gradients in forward-only. Peak RSS is 8.01 GiB under 64 GiB.
The 334-file completed manifest retains the complete v1 work-cap refusal.
Exact-commit review must independently reproduce all 117 JSON artifacts and
the report before approval; no original numerical source is changed.

Task 303 is approved at `71d96c93`. CM48 adds 25 Lean theorems: convex raw-class
regions and observable same-label input-hull certificates identify covered
missing labels and transfer actual complex-head margins and ordinary CE.
Nonzero Fourier witnesses preserve observed rows and bounds; two compatible
targets still disagree at an uncovered query. Convex target regions remain an
extra assumption, not a consequence of flatness or observed fit. The native
INDEX has 4812 modules and 25187 theorems, with 36 sorry and no extra axioms.

Task 304 implements this geometry over the reviewed CM46 source. Integer hull
membership and rational coordinate certificates replace only uniquely covered
unseen contexts; intersecting class hulls, missing coverage and unknown axes
keep the parent. Complete complex products and original head derivatives remain.
All full/minibatch/forward metadata and repeated/tail records are retained;
the new fixed context construction adds no gradient or optimizer update.
Component fixtures and exact review precede a separate frozen all 35 comparison,
which must retain tasks 296, 298 and 301 results and every earlier failure/baseline.

Task 304 is reviewed at `0fee1a9d`: all 11 native tests, exact binary/source
checks, full Lean, INDEX and axiom checks pass. Training-only geometry inspection
covers 3,878 prior and 3,932 supplement unseen inputs without overlapping
observed class hulls. This is coverage, not held-out correctness.

Task 305 freezes the separate original comparison. The source remains task296;
the complete task301 guarded CE result and its nested task298 tradeoff are
retained, alongside every earlier failure and all conventional/depth baselines.
All 35 tasks and both 105-choice policies remain in scope. Geometry is fixed
before scoring, with confidence one and zero new optimization in every mode.
Raw accuracy and ordinary CE are compared to both source and guarded references;
regressions cannot be hidden by choosing a weaker reference after evaluation.

Task 305 original evaluation completed at frozen producer `64efc9f8`: all 35
cases and replays pass, with 35/35 primary and auxiliary successes in every
mode for both policies. Fibonacci raw errors improve from 101/100/100 to
96/95/95; CE improves against both task296 and task301 to
0.165498/0.168674/0.171923. Task298 still has lower CE with more raw errors and
remains retained in full. Original training CE and field accuracy are unchanged;
new gradients and optimizer updates are zero. Peak RSS is 3.4943 GiB.
The complete 334-file manifest preserves every reference and failed attempt.
Exact findings-commit independent reproduction is required before approval.

Task 305 is approved at `5dd63c6e`: independent fresh preparation, all 35 cases
and replays reproduce all 117 JSON artifacts and the report byte-for-byte.
The full-suite primary raw totals are 279856/279952 full, 279855/279952 mini,
and 279857/279952 forward. The remaining errors are Fibonacci 96/95/95 and
two inherited minibatch sum-square errors.

Task 306 audits these totals and reconstructs the actual Fibonacci predictions
without changing the model. It classifies coverage and observed class geometry
to decide the next Lean question. Observed line structure alone cannot justify
unknown-label extrapolation; no new prediction rule is part of this audit.

Task 306 frozen audit `5a59a971` completed twice with byte-identical reports.
All 3878 covered Fibonacci validation queries are correct: five fixes and zero
regressions in each mode. Remaining errors split into 91/90/90 uncovered and
five without observed class support. Eight uncovered true classes are sampled
on a line but their error query lies off that line, so blanket affine extension
is not justified. The next proof proposal is convex exclusion with an explicit
closed candidate pool or a correct decoder fiber, followed by a separate native
comparison. Audit peak RSS was 3.31 GiB; the model remains unchanged.

Task 307 is approved at `26ec2faa`. CM49 proves observable candidate exclusion,
conditional identification beyond observed hull coverage, actual complete-head
margin/CE transfer and bounded positive feature mixtures. Its 33 Lean theorems
include nonzero Fourier exterior examples, a missing-class counterexample and
an ambiguous gap with a closed pool. Candidate closure and target convexity
remain explicit hypotheses; the proof does not change benchmark accuracy.

Task 308 implements this exclusion rule in a separate native component. Its
candidate pool contains every raw label observed in the routed block, with
closure explicitly unproved. Exact rational conflicts rule out alternatives;
only one survivor enables a new positive same-label feature mixture. Every
other parent row, original record and full-head derivative is preserved.
Component checks and exact review precede a frozen complete original comparison.

Task 308 is approved at `9455f516`: 12 native tests and full Lean/INDEX/axiom
checks pass. Training-only inspection identifies 11904 additional prior inputs
and 12311 supplement inputs in every mode, with no inconsistent candidate pool.
This is conditional coverage, not held-out accuracy. Peak inspection RSS was
2.32 GiB under the 64 GiB process cap; all original states remain unchanged.

Task 309 integrates the reviewed CM49 component into a separate complete
comparison against task305, retaining all earlier references and baselines.
It seals full actual states before scoring, preserves original observed metric
objects exactly, and audits every fresh raw prediction change. Both 105-choice
policies and all 35 raw denominators remain. Task310 will execute and independently
replay the frozen original comparison only after adapter review.

Task267's earlier CM38 fallback branch has now completed all35 original cases,
all35 numerical replays and its recursive complete summary. Both policies
reach35/35 primary and auxiliary successes in all three modes: forward-only
associative divisor improves from2508/4656 to4656/4656, while validation CE
falls from11.07216 to0.00531025. The176 unchanged,32 wrapped and2 refused import
outcomes remain;30 wrappers refuse their prediction domain and retain earlier
results. PeakRSS4.883GiB, no new gradients or discovery. This is a separate
historical branch, not a new raw-exclusion result. Exact findings review remains.

Task309 is approved at ee4b6a44. Its frozen task310 original comparison and
all35 replays now pass. Both policies retain35/35 primary and auxiliary
successes in every mode. Fibonacci improves one raw answer per mode, with
95/94/94 remaining errors andCE0.164323/0.167717/0.171430. The169 new validation
contexts are all correct; only one prediction changes per mode. All35 primary
raw errors are95/96/94, including two inherited minibatch sum-square errors.
Full observed metric objects remain exact. All previous references and costs
are retained; peakRSS4.43GiB. Full findings-commit independent review is next.

Task310 is approved at df79e005: the independent fresh run reproduces all117
JSON artifacts and the complete report exactly. Its one raw fix per mode and
all35 declared successes are now verified. The next identified implementation
gap is geometric: CM49's general theorem permits intersections between hulls,
but its first native search only checks opposite-label observed points.
Task311 formalizes CM50 finite two-sided barycentric certificates and a planar
crossing example missed by that sufficient test. The actual complete Fourier
head, ordinary raw CE, bounds and observed-record derivatives stay connected.
Tasks312–314 will implement, integrate and separately evaluate the stronger
search; no additional benchmark gain follows from the proof alone.

Task311 is approved at `2b17dc86`, with 24 new Lean theorems and unchanged
36sorry/0rests/0extra-axiom counts. Task312 implements those two-sided rational
certificates in a separate Rust component over the exact reviewed CM49 parent.
It searches remaining eligible inputs, retains all earlier rows and verifies
both mixture supports, masses, tagged query role and exact coordinates.
Complete source/head products, ordinary CE and original repeated-record
derivatives are preserved. Native and training-only resource checks precede
exact review; tasks313–314 will integrate and measure the complete original
suite separately. No new benchmark gain is claimed from the fixture.

Task312's 13 native tests and full Lean/INDEX/axiom checks pass. Training-only
construction on all six task310 sources identifies 6350 additional prior
inputs and 6540 supplement inputs in every mode; 7762/7999 remain ambiguous
and no candidate pool is inconsistent. Peak RSS is 2.94GiB, the maximum buffer
forecast 9.11GiB and work forecast178.76B, within unchanged caps. Exact review
and source/executable freezing precede task313; held-out metrics are unopened.

Task312 is approved at `a6be8048`: exact review reproduces the executable,
all550 frozen source files and all six training-only coverage reports. Task313
integrates that component into a separate complete comparison against reviewed
task310, retaining task305 and task301 comparisons and every older reference,
failure and cost. Its prospective4T case allowance covers the3.125T forecast
from training-only construction, reconstruction and full metric/audit work;
the64GiB process and56GiB buffer limits remain. Exact adapter review precedes
task314's frozen all35 original evaluation and independent fresh reproduction.

Task313's18 native tests,format,clippy,release executable,fullLean13917jobs,
nativeINDEX andaxiom audit pass with36sorry/0rests/0extra unchanged. The
compiled parent fingerprint and every task310 artifact reproduce. Exact
landed-commit review is the remaining gate before original task314 scoring.

Task267 is now approved at `1cdc7128`: its independent complete review repeats
all35 saved-state replays and the recursive summary byte-for-byte, with fresh
native/Lean/INDEX/axiom and frozen-hash checks. This closes the historical CM38
branch without changing any later raw-label comparison.

Task313 is approved at `9373e853`. The frozen task314 original evaluation now
completes all35 cases and all35 replays. Both policies retain35/35 primary and
auxiliary successes in every mode, but the stronger hull search fixes no new
raw answer. All91 newly covered validation queries were already correct.
Fibonacci remains95/94/94 errors with slightly lowerCE
0.164271/0.167656/0.171369; all35 raw errors remain95/96/94 over279952 targets.
Every observed metric object is unchanged. Actual peakRSS5.83GiB stays below
the64GiB limit. All references, including task298's lower-CE tradeoff, remain.
Authored findings record the unchanged accuracy; exact findings-commit fresh
prepare,35cases,35replays and complete JSON/report reproduction follow.

Task315 follows the unchanged raw accuracy with a different feature proposal.
CM51 bounds fixed signed source combinations by their absolute mass and
normalizes them to retain the original complex-source feature and head bounds.
Observed Dirac rows preserve original CE and complete-head derivatives in all
modes. Query score error and strict raw margins require an explicit reference
reconstruction residual; flatness and observed fit do not establish it. A
nonzero Fourier witness and two compatible disagreeing targets retain that
distinction. Native signed-kernel construction and a frozen original comparison
remain separate work; no new raw-label gain is claimed from the proof.

Task315 is approved at `803958cc`: 22 new Lean theorems establish bounded
signed source combinations and conditional raw margins, with the existing
36 sorry and no extra axioms unchanged. Task316 implements a fixed bank of
ordinal input kernels over the exact reviewed task314 complex heads. Original
source-excluded records select a convex mixture with ordinary raw CE; full,
minibatch and forward modes retain their actual source histories and charge
new diagnostic work separately. Eight native tests pass, including independent
kernel solves, signed complex products, full/tail gradients and replay.
Training-only inspection and exact component review precede any separate
original all35 comparison. No new held-out accuracy result is claimed.

Task316's six original-source training-only inspections now complete. Every
mode improves its source-excluded diagnostic CE with no fit refusals. Peak
RSS is 3.37 GiB; the largest buffer forecast is 10.55 GiB and work forecast
237.01 billion units. Full/minibatch charge 64/80 new diagnostic gradient
batches per role; forward uses zero. The minibatch prior retains its initial
vertex while accounting for all attempted updates. These diagnostics do not
establish a new raw-label result. Exact component review comes next.

The user's 2026-11-14 extension adds the full `stress_tests_for_gpt.md` catalog:
165 historical cases, including 130 outside the current 35-task suite. The new
goal is at least 99% held-out accuracy per task. Retain raw-token accuracy,
exact-sequence accuracy, ordinary CE and each original declared success flag
as distinct fields. Fibonacci's current approximately 98% raw accuracy remains
below this new goal despite its original success flag. Expand first to MQAR,
two-hop lookup and other sequence tasks, preserving original splits, longer
probes, controls, budgets, baselines and full/mini/forward modes. Use Rust/Kernite
only, retain the 64 GiB memory cap, and derive no answer rule from task names.

Task316 is approved at `99a42652`: all native and Lean checks pass, the589-file
source freeze matches and all six training-only diagnostic records reproduce
byte-for-byte. Tasks317/319 will integrate and independently evaluate the signed
model on the original35 suite, preserving the new per-task99% goals separately
from the original success flags.

Task318 implements the native165-case reconciliation and freezes the additional
cohort: MQAR, two-hop lookup, Dyck prefixes and three alternating blocks. The
user clarified99% accuracy: a sequence case must meet99% raw-token and99% exact
sequence accuracy, with both fields and ordinary CE retained. Original95%
thresholds and non-accuracy targets remain separate. All165 original plan and
history identities must match the catalog;130 cases are additional, with no
new-model measurements inferred from historical passes. Task320 will port the
first four original data families and run full/mini/forward with native tooling
under64GiB, preserving longer probes and all bounded misses.

Task318 is approved at `07a39ac2`: fresh native inventory generation reproduces
all165 cases byte-for-byte, all39 frozen reference inputs match and native/Lean
checks pass. Task320's initial Rust adapter now generates MQAR, two-hop lookup,
Dyck prefixes and alternating blocks, retaining clean/novel and longer splits.
Four native tests pass, including ten existing35 golden split hashes and
independent target/mask checks. The new cases' individual historical hashes
and checkpoints are unavailable, so source-derived fingerprints are explicitly
distinguished from historical artifact reproduction. No new model is scored
by data preparation. Training-only construction and frozen all-mode comparison
remain, with99% raw-token and99% exact-sequence goals reported per task.

The data adapter is landed at `a3283cd7`. The next task320 change binds the
unchanged CFt4/CFt5 causal-pair component with full16, cyclic-mini256 and
forward0 updates as a baseline for these four additional cases. Eight native
tests verify training-only construction, all-mode full complex predictions,
independent replay, refusal retention and separate raw/exact goals. Freeze the
committed source, release executable and protocol before all four training
stages, then score and replay every short/long split. The165 historical GPT
snapshot and original35 selected-model/transformer reports remain separate;
this component is not the full original35 ensemble. No new score is claimed
until that frozen comparison completes.

Task320's frozen producer `7a2e2ba4` and a fresh complete run now reproduce
all23 JSON artifacts byte-for-byte (report `65620f07…9f7ff7b`). Of12 outcomes,
MQAR full/forward score1/128 test tokens and0/64 exact sequences despite
256/256 training tokens; mini hits the existing2B work cap. Lookup exhausts
250000 discovery attempts, while Dyck/blocks have no consistent candidate in
the bounded single/pair feature library. All10 refusals and all longer probes
are retained, with no process failure or OOM (peak53388KiB RSS). No additional
case passes95% or99%. The next repair needs shared query-relative input
programs and generic prefix summaries, with the copy prior and unseen-label
assumptions proved explicitly before implementation. Preserve this baseline,
the previous original35 ensemble and tasks317/319's signed-kernel work.

Task320 is approved at `688f5592`. Task321 extends Lean first with CM52: a
shared canonical copy head over fixed input programs, actual full complex
products, convex ordinary-CE mixtures and executable training-only filtering.
Consensus determines accepted unseen symbols only under explicit bank
realizability. A four-state witness predicts a class absent from training
labels; a nonseparating observation leaves two flat-code predictions ambiguous.
All26 new theorems compile without warnings in a13924-job full build. Their
axiom audit is standard; the global count remains36sorry/0rests/0extra, and
generated INDEX has4827 modules and25292 theorems. Independent review must
finish before task322 implements and evaluates a native shared-program repair.

Task321 is approved at `62b675fc`. Task322 implements the native CM52 proposal:
fixed bounded causal copy walks, training-only elimination, a complete complex
vocabulary copy head and an ordinary-CE convex mixture. Full16 and mini4 epochs
with32-prefix short-tail batches charge actual derivatives; forward uses zero.
Consensus and unobserved-label coverage are diagnostics, not an assertion of
unknown-target realizability. Ten native checks pass; the adapter preserves
all four original sequence datasets, all12 outcomes, historical GPT rows and the
failed task320 baseline. Seal all training states before held-out scoring and
independently replay every prediction, metric and refusal. Original35 results
and pending signed-kernel tasks317/319 remain unchanged. No new cohort accuracy
is claimed until the committed source/binary/protocol comparison completes.

Task322 producer `c44632bb` now repairs both retrieval cases. MQAR and two-hop
lookup reach100% raw-token and exact-sequence accuracy in full, mini and forward
on every original short/long split, including lengths48/96. Each primary test is
128/128 tokens and64/64 sequences; CE is approximately6.15e-9 and3.74e-9. Unseen
test labels are correctly copied (MQAR1/1, lookup10/10), with unanimous programs
and no missing edges. Dyck/blocks retain six no-compatible-program refusals;
all12 outcomes remain. Every mode has2/4 both99% successes on all held-out splits.
The complete original and fresh review reproduce23 artifacts byte-for-byte
(report `550e9d86…01eb791`), with zero process failures/OOM and peak31,444KiB RSS.
Full/mini compute all derivatives but make zero effective weight changes on the
compatible face; the gain is shared representation and the explicit copy prior.
Finish independent review before the next generic prefix-summary/decoder repair.
This result does not re-evaluate the original35 or the remaining126 new cases.

Task322 is approved at `f868e07f`. Task323 extends Lean first with CM53:
fixed input programs produce latent states, and training observations learn
possibly many-to-one raw decoders. The executable learner retains every
consistent program and accepts only covered, unanimous dictionary outputs.
Joint program masses and decoder coefficients form one convex affine domain;
the actual complete complex head has convex ordinary full-vocabulary CE and
a charged coordinate count and column budget. Explicit target realizability
implies correct accepted raw predictions. A Fourier witness predicts a new
input and exhibits opposite feasible completions at an uncovered latent state.
The learner cannot identify a raw class absent from training without a copy
prior or another identifying assumption. No native prefix accuracy is inferred.
All55 new theorems pass a warning-free13932-job full build and standard axiom
audit; generated INDEX has4835 modules/25347 theorems, with the global
36sorry/0rests/0extra unchanged. Review323 before task324 implements and freezes
the generic prefix repair. Preserve both retrieval passes and tasks317/319.

Task323 is approved at `8c7b8933`: all55 new declarations and the full Lean
validation pass in the independent root review. Task324 now implements the
native CM53 affine slice with fixed generic signed-prefix/run summaries and
training-learned partial decoders. All vocabulary roles and caps are enumerated;
unknown rows and conflicts stay explicit. Training-only CM52 copy priority
preserves its existing models. Freeze the checked source and binary before
all four training stages, then score/replay every short/long split and all12
outcomes. Keep raw99/exact99/original95/CE separate; no new original accuracy
is inferred from the proof or synthetic component tests.

Task324 producer `3badaa30` now passes all four additional cases at100% raw
and exact accuracy in every mode and every original short/long split. Dyck
scores834/834 test tokens and64/64 sequences; blocks716/716 and64/64. Both have
CE1.32130e-9. MQAR/lookup complete copy states and results match task322 exactly.
All12 outcomes are usable with no unknown/ambiguous scored state, process
failure or OOM. Original and fresh review runs reproduce23 artifacts exactly
(report `1607addd…07c8a97`). Peak original RSS is171276KiB. Full/mini compute
real derivatives; the only mass changes are roundoff and observed CE stays
constant. The gain comes from the new shared summaries and learned decoders.
Finish independent review, then freeze further catalog cohorts separately.
Original35 Fibonacci remains approximately98%; tasks317/319 remain pending,
and126 additional cases have not been evaluated. No combined39-task pass is
claimed from this component result.

Task324 is approved at `3fde00b5`: independent native/Lean checks and every
mode replay pass, with all23 JSON artifacts exact and all four cases100% on
every split. Task325 prepares the next two original cases, `suite-dyck-2` and
`suite-dyck-3`, in Rust with unchanged seed1 lengths12–16 and32/64 probes,
128/32/64 pools, masks and clean/novel selection. Preserve historical GPT rows
and the complete task324 baseline; data preparation supplies no model result.
Task326 will audit CM53's generic contract and add bounded input-only nested
summaries with all token roles inferred by enumeration, retaining every
compatible learned decoder, all failures and the existing four passes. Keep
the per-task99% raw/exact goals and original95% flags separate. Original35
Fibonacci and tasks317/319 remain pending.

Task325 is approved at `47ab2f50`; the two new datasets and historical baselines
are frozen. Task326 audits CM53's existing generic input-program contract first
and implements exhaustive partial directed matchings of observed input symbols
with up to three nested role pairs. Raw decoders are learned from training only.
The unchanged task324 binder has training-only priority, preserving its four
passes. Complete complex heads and affine joint masses retain ordinary CE,
full16/mini4x32/forward0 and all bounded failures. Freeze this new six-case
producer before scoring, keep all 18 outcomes and every short/long split, and
require exact independent replay. No original accuracy is inferred from the
proof or synthetic checks; original35 and tasks317/319 remain separate.

Task326 producer `cd850e22` now reaches 100% raw and exact accuracy on both typed
bracket cases, in full/mini/forward and all 11 short/long splits. Original tests
score 842/842 and 850/850 tokens, each 64/64 sequences; CE is 1.39680e-9 and
1.47230e-9. All four controls also remain perfect, with complete parent states
and results exactly equal to task324. All 18 outcomes succeed without omitted
failures, missing states or OOM. The original and fresh runs reproduce all 33
JSON artifacts exactly (report `9fcfd6ae…29996f9`). Peak RSS is 391.7 MiB.
Each new case identifies one nested-role program and learns its three raw
labels; full/mini compute derivatives but do not change that single mass.
Depths extend from training maxima 8/6 to held-out 9/11. This is a structural
representation gain, not an observed CE optimization gain. Complete independent
review before further integration. Original35 Fibonacci remains near 98%,
tasks317/319 remain pending, and 124 additional catalog cases remain unscored.

Task326 is approved at `50d3e258`, with six cases at 100% on every split/mode,
33 exact artifacts and preserved controls/baselines. Task317 now integrates the
already reviewed CM51 signed-kernel component316 into the complete original35
comparison. Reread and recheck the existing signed bound first; keep its query
reference and residual assumptions explicit. Preserve all original success flags,
both policies, complete histories, raw changes and every older CE tradeoff.
Report native raw and native exact-sequence 99% separately from field readout.
The six training-only inspections imply a 4.213T complete-case forecast, so fix
a prospective 5T allowance while retaining 56 GiB buffers and 64 GiB processes.
Review and freeze the adapter before task319 scores any original signed model.
No new Fibonacci accuracy is inferred from the proof, forecast or synthetic tests.

Task317 is approved at `1bbbe27f`. Task319's first complete attempt preserves
34 successful unchanged runs/replays and one Fibonacci refusal. Before any new
model seal or held-out score, the real raw/field/audit forecast requires
101,001,277,440 units per endpoint, exceeding the reserved64B. The prior
prospective command had checked the reservation without verifying that input
layout. Correct it to share the runtime call counter and check all56,448 calls
per endpoint from the frozen generator. Reserve128B per endpoint prospectively;
4,323,846,694,320 actual forecast and4,404,842,862,000 reserved units still fit
the original5T case allowance. Preserve all324 failed-attempt artifacts and its
complete35-case record in new reports. Freeze and independently check this cost
repair before rerunning; kernels, model selection, head coefficients and the
64GiB memory limit do not change. No new signed accuracy is available yet.

The corrected producer `9c7370a5` is independently reviewed and frozen before
task319's `original-v2`. All35 runs and35 replays complete with exact saved-state
and metric reconstruction. Both policies retain35/35 original success flags in
all modes; raw99/native exact99 are34/35 in each mode and across all held-out
splits. Fibonacci remains the sole miss: full4608/4704 (97.959184%, CE0.115829730),
mini and forward4611/4704 (98.022959%, CE0.101307444). Full fixes2/regresses3;
mini and forward fix14/regress13. Lower CE does not establish the99% target.
Original observed CE and every earlier baseline/tradeoff remain exact. The
34 unchanged cases explicitly inherit314 metrics; only three Fibonacci states
are freshly changed. Peak RSS6.69GiB; complete work forecast4.324T below5T.
Report SHA2567632e96c882483211d31a2fc882990ffbe0d3fd77144424b353dc1bed2b18ebc.
Independent review at `b99c845b` now reproduces all118 JSON artifacts and the
report exactly, with22 native checks and every direct Lean/global axiom/INDEX
check passing. The fresh repeat takes14m50.38s and peaks at6.70GiB RSS under
64GiB. Task319 is approved as a reproducible result; Fibonacci still misses99%.

Tasks329–330 will audit the fixed-feature Lean results and test a freshly trained
bounded complete complex head on generic shared input features, using ordinary
CE for the actual observed prediction logits. Keep target assumptions explicit,
forward initialization work visible and the entire signed result as a reference.
In parallel with the ongoing evaluation workflow, only planning has been done
for tasks327–328: native ports of the six original copy/reverse/sort repeat/unique
cases, then a Lean-first causal-program extension with real EOS generation,
longer probes and the six existing100% sequence cases as frozen controls.
No new catalog case is scored by this planning. Original35 and the separate
six-case component remain distinct;124 additional cases remain unscored.

Task327 now ports the six original copy/reverse/sort repeat/unique cases in
native Rust after319's exact independent repeat. Preserve the326 six-case report
and all historical GPT endpoints. These tasks require actual greedy generation
from raw prompts through EOS, with shifted teacher-forced targets and CE kept
separate. Seed1 has128/32/64 pools, lengths8–16 and32/64 probes, clean/novel splits
and100 raw classes. All six native test-target counts match the historical
772/842/842/842/828/860 counts; longer tests each contain2112 and4160 targets.
Seven source-conformance/scoring tests pass. Commit and freeze before production
inventory generation, then independently reproduce all bytes under64GiB.
No learned model or additional catalog accuracy is measured in this data step.

Task327 producer `d7f078ad` is frozen with411 source inputs before inventory
generation. Original and fresh repeat match the complete inventory exactly,
SHA2562778983591a49bcde79dbcd9952b4999fa0afb914b9e37f4826834116a146e6f.
All six historical denominator checks and66 split identities are retained;
the complete326 reference is structurally exact. Both runs exit0 at a peak
639.2MiB RSS under64GiB. Seven native and all direct Lean/axiom/INDEX checks
pass. Review the landed findings and repeat the inventory before328's model
work. This is data/scoring preparation, with zero new model evaluations.

Task327 is approved at `552b9738`: independent native/directLean checks and a
third inventory reproduce all bytes, retaining the full326 baseline. Task328
starts with CM54's missing trajectory bridge. `LatentPrograms/Generation`
proves actual complex argmax agreement and complete own-prefix/EOS generation
from accepted queries on the target's finite path. The training-only initializer
supplies a feasible bounded tensor; target realizability, truthful observations
and path coverage remain explicit. A nonconstant unseen-prefix witness emits
one token then EOS. Audit and land this Lean extension before native model work.
No new original35 or generative-case accuracy is claimed by the proof.

CM54 landed at `4c95deeb` after full Lean13933 jobs, standard-axiom checks and
native INDEX regeneration (4836 modules,25352 theorems,36 existing sorry).
The initial full check hit thread creation during concurrent search indexing;
sequential checks with `LEAN_NUM_THREADS=2` pass under64GiB. Task328 now has
a native fixed positional/rank bank, learned delimiter/decoder/EOS roles,
128-state complete complex products and true generated-prefix auditing.
Unknown states retain explicit diagnostics even with a declared identity-copy
numerical completion. Nine synthetic checks pass. Freeze and seal all12 states
before any original scoring; retain36 outcomes and exact326 control states.
Original35 raw99 remains34/35, and no new catalog pass is claimed yet.

Task328's first frozen run at `3eedd618` completes all36 native stage/score/
replay jobs: six preserved controls plus four new copy/sort tasks reach100%
raw/exact on all11 splits in full/mini/forward modes. Two reverse tasks retain
four compatible programs and refuse scoring at the128B cap; totals remain
10/12 with two refusals. The running driver was edited prematurely and exits2
at summary; a separate frozen-binary summary exits0. Both outcomes are retained.
Report SHA256637fe5c225dda9bbbd9c0cce9b9b303be252976ddfe2060dcff3a72173e2d0d4.
The reverse dimension-only forecasts are130.65B/131.84B. Before measuring their
predictions, raise the scoring cap to160B, freeze the revision and require all12
state hashes and all previously measured endpoints to match exactly. Preserve
the complete initial report, run the immutable driver, then independently repeat
every case under64GiB. No model or training change and no new original35 result.

While task328's immutable rerun executes, task329 audits the fresh shared-feature
head. CM42's batch convexity and CM51's normalized feature bounds apply without
observed Dirac rows; CD8 charges the complete learned coefficient tensor. The
old observed-CE identities do not apply and will not be claimed. A prospective
fixed ordinal/quadratic8-state basis will share features on observed and unseen
inputs, with a new ordinary-CE head and explicit training-only initialization.
Check the non-Dirac Lean witnesses and axiom audit before native implementation.

Task328 producer `b3723b82` completes all12 cases and36 mode outcomes with100%
raw/exact on every original short/long split. All12 states and ten earlier
measured endpoints remain exact; both previously refused reverse tasks now
score100%. All36 native jobs and the immutable driver/summary exit0. The run
takes17m58.44s, peaks at3.06GiB, and preserves76 JSON artifacts. Report SHA256
c84f3f86b049c476809d13cf96f7b7ab46bab6b44a0c92c7dbed9eb5f7db11cc.
Land authored findings and independently reproduce the complete result before
approving328. The original35 raw99 count remains34/35;118 catalog cases remain.

Task328 is approved at `5c2f2817`: independent native/directLean checks and a
fresh complete run reproduce all76 JSON artifacts, including every state,
metric, refusal reference and report. All36 native jobs exit0; the repeat
takes17m46.56s and3.06GiB peak RSS under64GiB. Evidence is preserved under the
v2 freeze's `findings-review`; the completed328 worktree is removed. Original35
remains34/35 raw99 while the separate329 head undergoes training-only inspection.

Task329's Lean audit `5570b115` covers the fresh complete head without Dirac
observed rows. Native producer `57a2d61a` is frozen with624 source inputs.
All six original training-role inspections and exact state replays exit0;
full/mini perform128/304 effective CE updates per role, and forward performs
zero gradients/updates. Primary CE changes0.001336682346→0.001310457498 in
full and→0.001330829691 in mini. These are improvements within the new model,
not below the parent's observed CE9.63e-9. No heldout score has been computed.
All nine native/directLean checks pass. Inspection takes2m25.80s/3.60GiB;
26 JSON artifacts have summary SHA256
e3bcf2ac9a27e41c7af37c199db10e65e46654861e9fe11fe662bc8856f62729.
Independent review at7d5029ee reproduces all26 JSON artifacts and all native/
directLean checks. The repeat takes2m26.74s/3.60GiB, exit0;329 is approved.
Task330 now audits CM38's existing fixed coverage gate with shared non-Dirac
observed rows, repeated record IDs, a real tail batch and nonzero unsupported
parent logits. The primary's291 empty training blocks retain the earlier
predictor; all new training logits and CE changes remain reported. No new
theorem is required, and no new original35 heldout result is inferred.

Task330's native adapter binds the entire331-artifact319 run, all six reviewed329
outcomes and the separate historical DiffGPT result. Seven checks exercise both
roles in full/mini/forward, real unsupported fallback, complete raw-logit replay,
call accounting and105-choice regression preservation. Full metrics and paired
old/new raw logits are always computed for changed states, including observed
CE. The prospective complete-case forecast is7,730,302,869,792 kernel units;
all constructors, role training/replay, field calls and independent audits are
included before scoring, with a10T cap,56GiB explicit buffers and64GiB processes.
Freeze and run this reviewed adapter before reporting new99% attainment.

Task330 producer3653b58e corrects a summary/report checksum mixup before any
fitting or scoring, retaining the refused preparation. The full original run
and all35 replays pass with150 JSON and6 complete raw-logit JSONL artifacts.
Both policies retain35/35 original flags, but raw99/exact99 remain34/35.
Fibonacci full/mini produce4607/4704 raw hits (97.9379%), forward4608/4704
(97.9592%); validation CE is0.1511245/0.1511673/0.1511809, worse than319.
All4704 inputs use training-covered blocks. Of forward's96 errors,91 have a
true label already present in the routed training block. Freshly training the
complete head therefore does not solve the unseen-boundary problem. Preserve
this failed improvement; finish independent review, then test training-only
geometry sharing across blocks after a Lean audit of its linear parameter map.
The original run takes9m21.72s/6.05GiB under64GiB; complete summary774b6bea.

Task331 starts with a Lean audit of fixed affine sharing across complete
complex heads. CD8/CFt5/CM38 and `ConvexOn.comp_affineMap` cover actual ordinary
CE on the expanded-budget preimage. The explicit example shares parameters
across two complex blocks and both classes, includes a nonzero fixed offset,
checks a feasible nonzero point and retains repeated IDs and short batches.
No new theorem or `sorry` is needed. Native construction must use training-only
geometry and categorical decoders, account for every expanded coefficient,
and validate actual gradients/projection/replay before task332's all35 run.
The latter depends on both330 and331 and must retain319 and330 as references.

Task330 is approved at ec89d684 after all native/Lean checks and an independent
full run/replay reproduce all156 JSON/JSONL artifacts and the diagnostic.
Summary774b6bea is unchanged. The repeat takes9m20.70s/6.05GiB and exits0;
this validates the regression finding, not attainment of99%. Task331's explicit
affine-sharing audit passes with eight standard-axiom declarations and no
new theorem or `sorry`; native training-only work is next.

Task331 extends the exact preimage audit with CM55's sufficient weighted
parameter budget. Every occurrence in the complete raw-class tensor and the
nonzero fixed offset is charged. Lean proves convexity, the expansion budget,
the actual logit bound and ordinary full/minibatch CE convexity; a nonzero
example attains the bound. This permits a weighted parameter projection in
native training without treating compact storage as the expanded head norm.
Geometry/decoder discovery remains training-only and outside the convex claim.

Task331's Rust component now shares complete complex coefficients across
binary latent blocks, retaining separate intercepts and categorical decoders.
The declared quadratic/hinge library,16 directions, three covariance ridges
and four gains use three original-training exclusion folds. The inherited
route was learned on all original training, so this is explicitly a conditional
head diagnostic. All432 fold states and1728 candidate scores, three329 local
controls, final refit and ordinary full/mini updates are retained; forward
uses zero CE gradients. The complete radius1048576 is charged through CM55,
not presented as a matched capacity with329's65536. Seven native checks pass,
including actual full products, gradients, unequal batches, projection KKT,
label permutations, excluded-label independence and complete replay. Freeze
the source and full resource forecast before the six original training-role
inspections. No new original heldout measurement or99% claim is made here.

Task331 producer0a55b83f's six training-only inspections and complete internal
replays pass in5m03.71s/3.60GiB. Summary06590309 retains the failure to fit
80–86 primary training labels; full/mini CE reduces0.0433867 to0.0420775/
0.0429323. Primary conditional fold hits4534/4704 trail the local329 control's
4606/4704. No original heldout record has been scored. V2 preserves all v1
moment candidates and adds training-only projected interval calibration plus
16 fixed shared linear directions, with all overlaps and fallback intercepts
reported. The new1344 configurations,1008 fold states and4032 gain scores
must pass updated resource/replay checks and original training inspection before
task332. CM55 still covers the unchanged affine head and complete budget.

Task331 v2 producerb19b3bc9 is approved: all six original training inspections,
exact internal replays and an independent repeat reproduce all32 artifacts
(summary826b4fa7). Both roles/all modes fit4704/4704 training labels; conditional
folds4623/4704 improve on local329's4606/4704 but remain below99%. Every mode
selects fixed direction[0,1] with calibrated intercepts, quadratic features
and gain128; all93/94 binary blocks separate. Full/mini reduce ordinary CE
slightly with128/304 effective updates per role; forward uses zero gradients.
Original/repeat times8m51.92s/8m47.46s and3.60GiB peak remain under64GiB. All
eight native/directLean checks pass and every v1 candidate/control is retained
exactly. Task332 is claimed for the separately frozen all35 comparison; no new
original heldout accuracy has been measured. Tasks333/334 queue a native port
and generic latent-program extension for four additional parity scratchpads,
preserving the12 reviewed additional passes and original short/long probes.

Task332 first audits CM38 after fixed affine sharing in
`scripts/SharedGeometryFallbackAxioms.lean`. The combined parent-frozen domain,
ordinary normalized CE, equality as a function of shared parameters and the
covered-batch derivatives pass, retaining non-Dirac rows, repeated IDs, short
tails and nonzero fallback logits. CM55 supplies the separate sufficient
expansion budget. The native all35 adapter is under construction and must keep
both complete319 and330 references, with fresh metrics for every new state.

The task332 Rust adapter preserves all35 cases and both105-choice policies,
all conventional/depth/history references, and complete319/330 snapshots.
Fresh331 training must match the reviewed six full outcomes exactly, then all
three endpoint states are sealed before scoring. Full raw audits join retained
330 vectors by original record and verify319 logits/targets/roles exactly.
Seven integration checks pass, including both references and corrupt-record
refusals. Each run/replay invocation forecasts the entire search, actual shared
head, original records, projections and calls, with10T case work,56GiB explicit
buffers and64GiB processes. Summary parsing is a separate bounded phase.
Freeze the source/protocol, run all35 and independently repeat before reporting
any new heldout99% attainment; current original results remain34/35 raw99.

Task333 ports the four original parity scratchpads to Rust. Six checks preserve
format-independent bit draws, the separate128-bit hint stream, paired ordering,
initial EVEN, sparse original indices, shifted EOS supervision and own-output
generation. All historical short/32/64 token and transition denominators match.
The prospective inventory retains the entire reviewed328 report and all12
previous passes. No new model score is produced;118 additional cases remain
unmeasured. Commit and freeze before producing and independently repeating the
complete inventory. Task334 must handle229 hinted classes explicitly rather
than silently using328's126-class limit.

Task333 producer9d9216b0 is approved. Both frozen/native inventories reproduce
the full four-case data and12-case baseline byte-for-byte (SHA8b5508d9), with
44 splits and matching historical short/long counts. Original/repeat use
1m19.10s/1m17.67s and3.05GiB peak under64GiB. All six native checks, source pins,
directLean/global audit and native INDEX pass. Task334 is now unblocked; this
data result still adds zero model evaluations and leaves118 additional cases
unmeasured.

Task332 producer312ff793 completes all35 original cases and their replays;
an independent fresh run reproduces all163 JSON/JSONL artifacts (summary
d0f83b52). Both105-choice policies/all modes retain35/35 original success flags
but34/35 raw99/native exact99/all-heldout99. Fibonacci forward/full/mini produce
4612/4610/4605 of4704 correct (98.044/98.002/97.895%), with CE around0.263,
worse than319 and330. All training labels fit. Complete head7,180,800 real
coordinates and518 new shared parameters are charged; forward has zero CE
updates. Each full driver takes22m23s with6.49GiB peak under64GiB. Seven native
checks and directLean/audits/INDEX pass. Preserve the regression: no99% claim.
Tasks335/336 queue a bounded input-geometry hypothesis and a separate frozen
all35 comparison;335 is available to claim. Task334 continues the four parity
model evaluations with the reviewed12 controls.

Task334 first audits a generic count-state program and a learned categorical
decoder in `scripts/PrefixCountingGenerationAxioms.lean`. Four observations
eliminate the wrong count symbol, identify a new covered input through the
actual complex Fourier head, and leave a removed state's query unidentified.
CM53 supplies initialization, complete coefficient bounds and convex ordinary
CE; CM54 supplies conditional own-prefix generation including EOS. The direct
audit passes without new source theorems or axioms.

The prospective native bank learns delimiters and frequent count symbols from
minimal observed prefixes, enumerates stride/offset, moduli2..8, final/running/
selected emissions and adjacent-token copying, and learns raw labels only from
training dictionaries. Retain every conflict and compatible dictionary; choose
the first compatible program in the fixed order, as CM53's vertex initializer.
This is a singleton affine training slice after discrete discovery. All48 mode
endpoints remain required, with unchanged328 controls. The new complex code has
256 states, reserving raw copy states below240, count states240..247, and two
structural states whose raw outputs are learned. No target oracle or task name
enters a model. Unknown copies use an explicit prior; other unknown states have
no identification certificate. Forecast complete work before fitting/scoring;
the prospective new evaluation cap is512B scalar units per case/mode, with
56GiB buffers and64GiB process limits. No new heldout result yet.

The task334 native implementation now passes eight checks, including a full
generic count-three binding and longer own-output generation in all modes,
categorical decoder renaming, independent interpreters/products, real mini
tails, source/refusal preservation and prospective original dimensions.
The largest new full-batch forecast is66.4904B scalar units, so the new training
cap is96B before production; search remains64B and evaluation512B. Full Lean,
the new audit, global36sorry/0rests/0extra, native INDEX, fmt/clippy/release and
shell checks pass. Freeze committed source and run all16 states before any
score, then independently repeat. Original35 remains34/35 raw99; no new parity
model pass is claimed from these implementation checks.

Task335 starts with `scripts/BoundarySharingAxioms.lean`, before any native fit.
The audit instantiates CM55 with global complex coefficients and boundary
parameters tied through signed input-only block descriptors. It charges all
expanded coefficients and a nonzero categorical offset, and combines CM51
normalization with full/mini ordinary-CE convexity. The finite four-state
witness is not a proof of floating-point refinement or unseen-label accuracy.
The proposed native constructor will compare generic route extrema inside the
whole grid and the nonmodal-route bounding rectangle, using categorical class
orientations and training-only conditional folds. It will replace independent
block offsets by a global affine descriptor map. Retain331/332, including
their original heldout CE regression. Task336 remains the separate original35
comparison; task335 may not use its query labels to choose a model.

Task335 native source now passes five checks. Its fixed map uses axis/diagonal
block extrema inside full-grid/nonmodal-box frames, global uncentered ridge
moments or shared interval-boundary regression, and three conditional
cell-exclusion folds. The3072 configurations retain1152 fold states and9216
gain/common evaluations. Complete complex budgets charge at most25 stored
parameters per role; full/mini/forward retain original records and final
iterates. Commit/freeze before six original training-only inspections and
independent repetition. No new original heldout result follows from these checks.

Task334 is now reviewed at producer2612b05f. Original25m21.56s and independent
repeat25m12.05s both finish; all100 JSON artifacts are exactly equal and the12
earlier control states/data/results remain exact. All16 additional cases pass
original success and raw/exact99 in every mode and short/long split; all four
new parity cases reach100%. Peak RAM7.53GiB. The learned finite programs fix
singleton convex slices, so full/mini compute true gradients but make zero
effective updates. Ten missing-state scoring events per mode in one hinted
case use an explicit copy prior, with duplicate novel subsets retained.
These empirical successes do not identify arbitrary unseen labels. Evidence
and review are preserved in the immutable prefix-programs-v1 freeze. There
remain114 additional cases unmeasured; original35 raw99 remains34/35.

Task335 producerc09c06bd is frozen with669 source inputs. Original and independent
six-inspection drivers are running from immutable copies, with step0.1 chosen
prospectively for the stronger shared-gradient coupling. Task336 remains blocked
on the component review and has not scored any new original heldout query.

The first335 primary inspection regresses:4494/4704 training labels and
4512/4704 conditional folds. The best raw result anywhere in its3072-config
bank is only4513/4704. Its full-batch CE decreases from0.25046 to0.24785,
so this failure is not an exploding optimizer. A training-only comparison of
local331 boundaries with input descriptors exposes discontinuities that one
global affine map cannot represent. Keep both complete six-inspection runs.
Before any revised native fit, `scripts/PartitionedBoundaryAxioms.lean` audits
a fixed descriptor partition with multiple blocks sharing each leaf's
parameters, full coefficient costs and ordinary convex CE. A proposed shallow
partition is learned only from training boundary diagnostics and then fixed
during CE. It is not a jointly convex tree-discovery procedure or an unseen-label
identification theorem. No revised native result exists yet.

The full335 affine-descriptor failure now reproduces exactly across28 JSON
artifacts. Six inspections finish in16m27.27s and independent repeat16m26.27s,
peak3.60GiB. Primary training4494/4513/4501 and supplement4300/4305/4310 for
forward/full/mini all trail331. Full128 and mini304 steps lower final CE;
forward0. Preserve complete summary1b40f427e1ec6607d7e49fcccc88ebc42d6b6d350eac466b9975eb7aaceaf18d
and all original/independent evidence in the boundary-geometry-v1 freeze.

`convex_stress_partitioned_boundary` extends that component after Lean-first
47ba037f. The declared library adds boundary regression trees of maximum depth1
or2, fixed during CE, with all attempted descriptor splits retained. At most
four regions and52 compact parameters share the complete complex head. Seven
native checks pass; the6144 configurations,2304 fold states and18432 scores
are fixed before production. Full-vocabulary CE scores use exact absent-class
multiplicity algebra, checked against actual full vectors; gradients retain
full vectors. The prospective12T work cap includes all split searches and
replays; buffers remain56GiB and processes64GiB. Freeze sources and run six
training-only inspections plus an independent repeat, retaining all failed335
data/state/score artifacts. Original heldout comparison336 remains separate.

Task335 is reviewed at partitioned producer2d5ac6ce. All56 JSON artifacts
match the independent repeat. All six mode/role fits classify4704/4704 training
records; the selected conditional folds reach4686/4704 and4676/4704. Both roles
choose a two-leaf boundary map with34 parameters. Full128 and mini304 effective
updates reduce CE; forward0. Original/repeat17m56.65s/17m55.24s, peak3.60GiB,
all native/Lean checks pass. Complete failed affine evidence remains retained.
These are conditional training diagnostics, not original heldout scores.
Task336 is claimed to freeze and evaluate this reviewed component on all35
original cases, retaining319/330/332, both105-choice policies and all baselines.
Its forecast must account for sequential fitting/replay lifetimes and complete
partition-search work under the unchanged64GiB process limit.

Task336 Lean-first audit4e25eef2 passes before native fitting:52 shared
coordinates, fixed coverage and complete fallback preserve original CE and
batch derivatives. The new partitioned comparison retains319/330/332 and all
baselines. It authenticates the complete reviewed335 outcomes and seals all
three endpoints before scores. Prospective80T/case includes every partition
search and replay. Sequential phase memory accounting replaces simultaneous
three-constructor buffers; immutable outcome byte sizes and all old resident
models remain charged under56GiB buffers/64GiB process caps. Test and freeze
the adapter before production; no new original heldout result is claimed yet.

Task336 producer3186b9c3 is frozen with1302 source inputs. The independent
rebuild is byte-identical and all eight native/Lean checks pass. Original and
independent drivers run from immutable copies, with65.581T complete case work
forecast and maximum58,600,283,680 buffer bytes, under56GiB/64GiB limits.
All modes are sealed before scoring. No new heldout result is recorded yet.

Task337 ports seven histogram/mode/frequency cases while336 runs. The native
port matches all21 historical token/transition denominators; all16 reviewed334
controls remain embedded. It retains formatting RNG semantics, first-occurrence
tie order, mode tie repair, own-output generation, EOS and final-versus-exact
metrics. Longer outputs include raw count labels absent from short training;
record them as an explicit decoder-coverage gap. Data-only preparation adds no
pass and leaves114 additional cases unmeasured. Freeze before original/repeat
inventories under64GiB; source and scoring rules stay outside learned predictors.

Task337 is reviewed at952b37ca. Both complete inventories match SHA256
21d0c176efdfe08b8bc04bdf83b62c91987e542d2e799160905dbea05d32c5ae;
the whole334 report remains equal. All native/source/Lean checks pass. Runs
3m22.07s/3m09.32s, peak7.52GiB. Numeric label coverage is a substantial gap:
each histogram length64 test has2792 targets absent from training outputs.
Task338 is claimed for an explicit class-coordinate identifying prior, with
training-only two-anchor fitting and affine complete-head CE. Flatness alone
still cannot identify arbitrary unseen class names. Seven new model evaluations
remain outstanding;114 additional catalog cases remain unmeasured.

Tasks336 and338 are now reviewed. Frozen336 producer3186b9c3 and findings
23cdec93 establish35/35 original tasks at raw99/exact99 in every mode, every
reported heldout split and both105-choice policies. Fibonacci is4695/4704
(99.8087%) in all modes; the other34 cases explicitly retain319 results.
All165 JSON/JSONL artifacts repeat exactly. Aggregate primary raw accuracy is
99.996785% forward/full and99.996071% mini. Complete work is65.581T, measured
peak RAM about7GiB under64GiB. All transformer/history baselines remain retained.

CM56 atb5f1c9d7 proves an additional identifying prior, not an implication of
flatness: supplied injective class coordinates, an unknown affine decoder of
a fixed scalar feature, and two distinct observed feature anchors. It derives
the actual complete complex head's strict unseen-class winner, charged budget
and convex ordinary full/minibatch CE. The nonconstant witness has two feasible
margins and exposes ambiguity without the prior. Full/global/INDEX checks pass
with36 existing sorry,0 rests and0 extra axioms; no native convergence theorem.

Task339 implements the frequency-model extension after this proof. First add
the complete CM56 decoder with observed-anchor fitting, a positive learned
margin, actual all-class products and full/minibatch/forward CE. Then add an
input-only finite group/order/aggregate program bank, training-only decoder
selection, declared coordinate/copy priors and a categorical-only control.
Freeze the complete runner before any heldout measurement and independently
repeat all seven337 cases while preserving all16 reviewed334 controls and the
separate35 original results. No new frequency model result is available yet;
114 additional catalog cases remain unmeasured.

Task339 is measured and independently reproduced at producer34ea5801. All
seven frequency cases achieve100% raw and exact accuracy on all11 splits in
forward-only, full-batch and minibatch modes. Every mode gets32,696/32,696 test
tokens,1,344/1,344 test sequences and7,410/7,410 unseen-label occurrences correct.
Weighted test CE is0.00619416 forward,0.000703828 full and0.000452416 mini.
The explicit class-coordinate prior completes the learned quantity decoder;
flatness alone does not identify it. The categorical control reaches7/7 primary
successes but1/7 all-heldout; both arms select the same programs on this cohort.
Discrete program selection remains outside the convex CE stage.

Both complete drivers exit0 and all123 JSON/JSONL artifacts match, manifest
f8fa1558162949f0677e494bf8eba7f36f509dfe9413631146d4c9be1fd744f6.
All13 native checks, full Lean/global axioms, INDEX and forbidden scans pass;
36 existing sorry remain, with0 rests and0 extra axioms. The original/repeat
take9:35.63/9:25.60 and peak about8.2GiB under64GiB. The whole16-case control
report and separate35-task comparison remain unchanged, including transformer
histories. Thus23 additional cases pass all-heldout99, and58/165 catalog cases
are measured;107 remain unmeasured. Task340 prepares the remaining13 synthetic
datasets: Count, eight Addition variants, three C-RASP depths and the explicit
random-label control. Data preparation alone must add no model-success claim.

Task340 now has a native data/scoring port for all13 remaining synthetic
cases. Seven independent checks pass, including digit-wise carry arithmetic,
all original masks/limits, format-equivalent RNG streams, inclusive C-RASP
prefix counts checked by a separate postorder recount, and random-control
novelty by full sampled sequence. All63 historical test example/target/change
denominators match. Freeze and independently repeat the full inventory before
model work; preserve whole339 and the separate original35 reference. No new
model is measured, so58/165 remains the evaluated scope and107 are unmeasured.

Task340's frozen producer28dc5c94 now completes both inventories, with an
independently rebuilt byte-identical binary. The full inventories match SHA256
a89e22f8896c693306c7eb19cdfa7ce9b960488285478e3cced4fb76d461d9b6;
original35 copies match too. Native readback rehashes the complete nested339
report, regenerates all13 case metadata records and checks every full historical
record. All215 splits are retained, including forced carry probes and random
payload novelty. Original/repeat24.87s/24.68s, peak7.60GiB under64GiB. This
completes data preparation only:58 cases measured,107 still unmeasured.

Task341 extends CM56 with joint program masses and coordinate decoders before
new native model work. CM57 uses six learned coordinates per fixed input-only
program, two actual Fourier blocks, complete costs (4,4,4,4,4,2), homogeneous
training anchors and an explicit identity-copy prior. Perspective budgets force
all coefficients to vanish at zero mass. Observable feature spans plus a
separate semantic realizability prior identify covered queries and derive
strict raw-class gaps through the actual normalized complex head. Ordinary
full/minibatch CE stays convex in every learned coordinate. Two active distinct
decoders, a rejected constant program, repeated records, unseen class 3 and an
unobserved copy input have nonzero witnesses; an uncovered role admits opposite
feasible raw winners. Direct Lean/audit/index checks must pass and the proof
must land before native joint-program runs. These proof witnesses add no
benchmark results; measured coverage remains 58/165, with 107 still unmeasured.

Task341's 40 new theorems now build and pass direct standard-axiom checks.
The full build has 13,946 jobs; global audit retains 36 existing sorry with
0 rests and 0 extra axioms. The native INDEX has 4,849 modules and 25,427
theorems, with 0 vacuous and 0 placeholder entries. Task342 implements and
audits the native joint head, gradients and projection; task343 then measures
the remaining 13 synthetic cases through a committed frozen driver and an
independent repeat. The independent-label control remains an explicit negative
control, and unknown-program realizability remains a separate assumption.

Task342 now implements the CM57 core in native Rust. All six coordinates per
program are updated through ordinary CE and a bounded Dykstra projection onto
observed anchors, the declared copy prior, perspective cones, a positive total
mass and the complete weighted budget. Prediction evaluates all expanded
complex coefficients; independent complex multiplication and finite differences
check the product and every gradient coordinate. Original anchors are checked
separately from the numerical row basis, and projection reports normalized
feasibility/change residuals or returns an explicit failure at its cycle cap.
Meaningful mode-replay, tail-batch, variational-projection, missing-direction
and contradictory-label controls accompany the core. All eight native checks,
formatting, clippy with warnings denied and release build pass. Full Lean,
direct CM57/global audits, native INDEX and forbidden scans also pass, retaining
36 existing sorry and 0 rests/extra axioms. Task343 will freeze the full
data/program driver before heldout measurements. No catalog measurement is
added here.

Task343 now connects fixed input-only range/radix and nested-count families
to the reviewed CM57 joint head. The302 numeric programs have a separate
interpreter; prefix discovery uses a bounded training-only beam and retains
distinct fitting syntaxes. The coordinate prior is compared with a frozen
observed-label dictionary on the same retained bank. Seven native checks cover
unseen-label completion, missing-role uncertainty, label transport, independent
products and all modes including the true minibatch tail. The complete driver
and protocol must be committed and frozen before heldout measurement. These
implementation checks add no benchmark result:58/165 remains the verified
scope,107 cases remain unmeasured, and the independent-label control may fail.

Task343's complete Rust driver now separates training-only discovery from
original prefix/own-output evaluation. It binds pinned340/339/original35
references, forecasts the complete search/train/reconstruct/score workload,
seals all13 cases and six endpoints before heldout scoring, and streams every
complete raw-class vector with independent interpreter/product checks. Ten
native checks cover the runner and inherited modes. Training-only development
fixed BOS exclusion from counted words and tested a128-entry prefix beam.
Depth1 C-RASP fits; depth2/3 discovery still refuses, and RandomLM remains an
explicit negative control. The upcoming frozen benchmark must retain those
failures and distinguish measured cases from successes. Do not change the
verified58/165 scope until frozen original and independent repeat evidence exists.

Task343's frozen producer17877185 now completes both full runs. Count, eight
Addition variants and depth1 C-RASP score100% raw/exact on all182 measured
splits in every mode; each mode gets63,700/63,700 test tokens and3,456/3,456
test sequences correct. Coordinate weighted test CE is0.3263473621 forward,
0.0030031147 full and0.0024034406 mini. The categorical control also reaches
100%; all896 unseen-label test occurrences use the supplied identity-copy
prior for hints. Depth2/3 C-RASP and RandomLM remain explicit fit refusals,
with null measurements and no success credit. All13 attempts stay in the
denominator. The verified scope is now68/165;97 cases remain unmeasured.

All189 original/repeat artifacts match, manifest56f68a3141418222f652a4796dc35933eda9713443d7603cead3875f962aeb6f.
The complete report has SHAd7714f2d2d261b9680aa4818b9877501a6c2586b09f5762d4c6c402687b4139d.
Both independent readback audits authenticate the whole340/339 histories,
the separate original35 comparison and120 complete raw files per run. Frozen
production binary equality survives the separate read-only auditor addition.
Runs11:12.83/10:52.54 peak at7.62GiB under64GiB; no generated output is committed.
Task344 must improve training-only deeper count-program discovery while keeping
the full343 report and every earlier baseline, before freezing a new benchmark.

Task344 starts with CM58 in Lean. Closing observed symbol words under nonempty
prefixes has cardinality at most their total length and preserves inclusive
counts when subprogram signatures agree on that closure. Structural induction
extends this to nested Boolean/count contexts and decoded observed-label fits.
Nonconstant witnesses show both why missing prefixes invalidate that argument
and why the resulting observed equivalence still says nothing about a new word.
All16 theorems pass the direct standard-axiom audit and the full13949-job Lean
build. The global audit retains36 existing sorry, zero dependent results and
zero extra axioms; native INDEX reports4852 modules and25443 theorems, with
zero vacuous or placeholder declarations. Keep the frozen343 producer and its68/165 coverage unchanged;
new algorithmic progress requires a new frozen independent benchmark repeat.

Training-only CM58 development retains1926 first-layer signature classes for
C-RASP2 and finds four compatible programs in1.40s/7.6MiB; depth3 still refuses
a depth2 fit. These are no new heldout scores. Before using observed label
transitions to narrow deeper latent programs, CM59 proves exact count-append
identities and necessary threshold-crossing child bits in Lean. It explicitly
retains ambiguity for downward transitions with an extra count increment.

Task344's native search and complete13-case comparison now follow CM58/CM59.
The proof extension totals32 theorems, with4853 modules/25459 theorems and the
same36 existing sorry, zero extra axioms/vacuous/placeholders. Depth2 training
fit is recovered; the first five depth3 development attempts retain their
explicit failures and caps. Conditional opposite-bank count bounds and an
exact indexed pair search now precede the unchanged joint CE head. Twelve
native checks pass, including independent positive and negative pair cases,
count-filter equivalence, true mask/tail semantics, refusals and saved seals.
The full driver retains all343 history and original35, forecasts every original
short/long/probe/carry split within64GiB, and must be committed/frozen before
any new heldout score. Reviewed coverage remains68/165 until that repeat.

Task345 independently prepares the remaining eight Basis cases while344's
frozen repeat runs. The Rust-only port preserves the full20000/512/512 recall
and language pools,128-length language selection, and512/128/256 disjoint noisy
parity pools with fixed noise seed2. Basis parity retains its default68 raw
classes. Independent oracles check all generated labels; seven native checks
include six immutable original35 fingerprints and the hard-recall historical
512/4096/4082 example/target/transition counts. Pending historical language and
noise runs have no invented measurements. The complete344 history and original35
comparison will remain exact separate artifacts. This is data preparation,
not added accuracy. Task346 must first justify bounded processing of all
observations in Lean and handle noisy-label constraints explicitly. Task347
retains the unresolved depth3 search beyond truncated latent banks.

Task345 producer67863307 now independently reproduces all eight data cases
and45 splits. All5806712 original training targets remain; noisy parity flips
103/512 content labels and no EOS targets. Four artifacts, including complete344,
original35 and the full source history, match exactly. Inventory hash8c02d2dc
and manifestaa3276ad bind the evidence; both seven-check native suites pass.
Preparation/repeat take27.59/26.67s and peak below0.89GiB. No model is evaluated,
so this adds no accuracy credit. Task346 can develop exact weighted or streamed
CE handling from these complete pools while344 finishes its model review.

Task344 now completes the independent13-case comparison. The eleven fitted
cases reach100% raw/exact on all193 measured splits in forward/full/minibatch
and categorical controls. Depth2 improves from refusal to704/704 ordinary-test
labels plus all6016 long-test labels; depth3 and RandomLM remain explicit
refusals. Aggregate coordinate test CE is0.2991924427/0.0027374829/0.0021887531.
All201 artifacts match, manifesteed7c94d65644bae0de432455b3fcb3dd7f2c383bd8d7c0008e018c8785b9dec;
report65bf8c53f330d76f254503cfcdc2b7d1b1de3c018bc44cc74a7e430559ce86fb.
Readback audits verify132 complete raw files per run and all prior histories.
The complete runs take23:51.60/24:05.96, peak7.63GiB, and remain under64GiB.
Verified scope is now69/165, with96 cases still unmeasured by the extended
model; data preparation345 adds no accuracy credit. These are seed1 results
in three modes. Task346 handles complete large pools and noisy constraints;
task347 retains the depth3 truncation gap. No end-to-end convexity of discrete
program discovery or unknown-target identification from flatness is claimed.

Task346 begins with CM60: exact multiplicity grouping preserves the complete
ordinary CE and its actual derivatives, using every original record and each
minibatch's own counts and denominator. Sixteen Lean theorems include active
constraint equivalence, a nonconstant full-head witness, the 1/3 versus 1/2
unweighted-deduplication counterexample and a one-record tail. The joint CM57
domain remains convex; noisy anchors remain a separate feasibility problem.
Full build13953 jobs and the direct standard-axiom audit pass; the global audit
retains36 existing sorry,0 rests and0 extra axioms. Native INDEX reports4856
modules,25475 theorems,0 vacuous and0 placeholder. Native weighted training
and the eight-case model comparison are still pending; no accuracy is added.

The CM60 native library now retains each original ID and evaluates complete
feature/observed-label fibers with their actual batch multiplicities. Six
checks compare all three modes with the retained ungrouped trainer, including
every joint gradient and independent complex-product finite differences.
A200003-record stream keeps all IDs and crosses the old200000 cap; two full
updates represent400006 gradient records through six weighted group visits.
The127-record schedule retains its105-record tail. Conflicting noisy labels
stay separate and incompatible exact anchors are refused. These are numerical
implementation checks, not new catalog measurements. Complete Basis feature
construction, noisy-domain treatment and the frozen eight-case repeat remain.

CM61 now separates exact identifying anchors from the complete observed CE
pool. Conflicting identical-feature anchors force zero masses and can make
the fully anchored positive-mass domain empty. A fixed selected-anchor domain
retains all CM57 budgets and convex full/minibatch CE on every observed label.
Its concrete noisy witness is feasible after selection and infeasible with
all anchors. Nine new Lean theorems pass the full13955-job build and direct
standard-axiom audit; global36 sorry,0 rests and0 extra axioms remain. Native
INDEX reports4858 modules25484 theorems,0 vacuous and0 placeholder. This is
a constraint repair, not a clean-label or heldout-accuracy guarantee.

Native CM61 now accepts a fixed explicit anchor-group selection while retaining
every original observed label in CE. The all-anchor path still uses the old
domain. Empty selections retain only the copy prior and budgets, with no fake
training rows. Nine numerical checks and clippy pass: noisy-witness feasibility,
complete record/gradient accounting, CE decrease in full/minibatch modes,
independent projection variational checks, and exact equality of projections,
heads and traces when every clean group is selected. The original200003-record
and complete-gradient checks still pass. Basis feature discovery, its explicit
anchor policy and the frozen complete eight-case repeat remain pending.

The weighted trainer additionally accepts a declared finite joint starting
tensor and projects it onto the same CM57/CM61 domain. Its masses must match
the policy; the entire proposal is saved and replayed. This allows observed
training statistics to initialize a noisy decoder without adding pseudo-clean
anchors or changing ordinary CE. All ten native checks and clippy pass; the
original mass-only entry points retain their previous behavior.

The complete eight-case Basis producer now uses observed-label moments and a
bounded, training-only discrete input grammar before fixing the joint convex
domain. Every original supervised position remains in ordinary CE. Exact fits
retain all anchors; noisy fits retain only consistently observed structural
roles, while all noisy labels still contribute to full and minibatch gradients.
The categorical control uses the same selected program. Six native tests cover
original masks/IDs, complete noisy pools, true tails, independent interpreters,
raw readback, tampered CE and failed own-output/EOS trajectories. Training-only
probes fit all six endpoints on recall-hard, languageL6 and noisy parity. L6
already saturates at initialization, so its executed gradients are not evidence
of meaningful parameter learning. The protocol freezes all eight cases before
heldout scoring; every case is trained/sealed before the first heldout query.
Complete independently rebuilt comparison remains pending. Coverage is still
69/165, with no accuracy credit from development probes or the new implementation.

Task346's frozen producerc8cc576d now completes the eight-case comparison.
All48 endpoints fit every original training record; all modes and categorical
controls reach100% raw/exact on the36 original heldout splits. Across14 ordinary
and long tests, each mode gets538734/538734 labels and6912/6912 sequences.
Coordinate weighted test CE is0.0040970567/0.0006544111/0.0003199512; the
language majority already saturates at initialization. Hard-recall CE improves
from0.4995287420 to0.0467284254/full and0.0027377025/mini. Noisy parity has
100% clean heldout accuracy while retaining89.94140625% raw accuracy on its
original corrupted training labels; all103 corruptions remain in every epoch.

All71 artifacts, including raw vectors, complete training traces and unchanged
earlier histories, independently repeat byte for byte. Report SHA256:
5fd5a0fd16f2d37a359c9bf70276b41866fd867ff2352be1ca0d8e8331cac34e.
Manifest SHA256:33260f0eac6489b845c8f0d89b220841a3b6b2ffbeba0fffcf4855584cba1b55.
Fresh builds pass six experiment and ten weighted-core checks. The formal
review repeats the13955-job build, all25 new standard-axiom checks and the
global36 sorry/0 rests/0 extra-axiom audit. Original stage times total15:31.47;
the repeat takes15:09.48 and peaks at393.03MiB RSS under64GiB. The final paired
verifier rehashes both complete manifests. Cumulative coverage is77/165 on
data seed1, with88 still unmeasured or not passing. This is a union of separate
catalog cohorts, not a shared all-task checkpoint. The supplied grammar and
discrete selection remain outside the joint convex CE claim. Task347 retains
the depth3 truncated-bank failure; no convergence or unseen-target guarantee
is inferred from these eight empirical successes.

Task347 starts with CM62's streamed count-program join. The344 evidence has
three complete incompatible joins and ten branches with at least one capped
bank. A fixed anchor index can process the other bank one candidate at a time,
preserving the actual observed-count fit while avoiding a query-retention cap.
Seventeen Lean theorems prove both exact index orientations, sentinel clipping,
chunked existence, complete-signature cache equivalence and reuse of refusals
only for no-larger cached syntax. Concrete nonconstant witnesses show both
truncated-bank failure and why a cheaper equivalent arrival must be retested.
The13958-job build and direct standard-axiom audit pass; global36 sorry,
0 rests and0 extra axioms remain. Native INDEX reports4861 modules25501
theorems,0 vacuous and0 placeholder. Native streaming search and its frozen
heldout repeat remain pending; verified catalog coverage stays77/165.

CM62 now has a separate native streaming implementation. Both exact index
orientations agree with independent exhaustive interpretation, including a
73-anchor padded bitset, negative offsets, negation and both label orientations.
Six checks also cover inactive gates, complete small-grammar signatures and
least syntax costs, a fit past truncation, work/storage admission, unchanged
shallow results, and the actual127-node size/cache counterexample. The bounded
cache retests cheaper equivalents and repeats evicted work. A complete anchor
is preferred; truncated anchors remain explicitly conditional. No heldout data
has entered this repair. The training-only depth3 probe and the complete frozen
13-case repeat remain pending; catalog coverage is unchanged at77/165.

The complete347 driver now preserves all original13 cases, the six CM57 CE
endpoints, frozen344/346 histories and the original35 reference. All cases seal
before any heldout query; independent runs must match every raw file, full
training state and replay. Coverage uses distinct case identities and reports
regressions explicitly. Twelve native checks pass, including six-endpoint CE
reconstruction, true minibatch tails and integer95/99 thresholds. The largest
composed buffer forecast is47561448432 bytes under56GiB, with the64GiB process
cap unchanged. A schema-specific JSON bound admits the complete independent
pair at about97.4GiB disk without removing raw evidence. The first committed
training-only streaming probe is still running; no new heldout credit is taken.

The ee289380 training-only probe now recovers12 complete observed fits for
depth3. Every1370 original label is checked with both program interpreters.
The successful12th branch retains a capped262144-class left anchor and streams
the opposite grammar past its old cap; this is a fit, not exhaustive coverage.
All17223763 indexed queries and8722818 cache hits are recorded;858393 cheaper
equivalents are correctly retested. Work37393912927376 remains below40T.
The committed snapshot and full JSON/log/resource evidence persist on storage.
Runtime16:59.56, peak1596700KiB (1.52GiB), no swaps. Heldout generalization and
the six CE endpoints are not yet measured; frozen full-cohort verification is
next, with cumulative accuracy credit still77/165.

The full producer1371ae4f is frozen with5408 source paths. A fresh target build
reproduces executable009df4e6616fe08a516eeac5fc45e9d76122208e7b73f0127b9740b062e89927
and passes all12 native checks. Repeated Lean/global/index audits remain green
with the existing36 sorry. The original13-case run is active. A separate audit
crate now rederives raw decisions, independent complex products, ordinary CE,
original transitions and complete EOS/limit trajectories from immutable output.
Its two tests reject rehashed CE tampering and altered counts and preserve
actual early-EOS, extra-output and categorical unseen-label failures. It does
not alter the frozen producer; the heldout comparison remains pending.

Task348 prepares the next catalog cohort while347's immutable comparison
continues. The54 arithmetic cases comprise22 existing cases and32 additions.
The native port verifies both historical tensor fingerprints for48 cases,
including26 additions, and independently matches the old22-case oracle.
Six fixed random-table/set helpers have unavailable source and remain explicit
data refusals with all old results and hashes retained. Model input is limited
to vocabulary and ordered raw training rows; formulas are data oracles only.
Five native checks cover masks, integer/constant semantics, source mutations,
disjoint splits and rehashed raw-label tampering. The complete frozen data
generation and independent repeat are next. This adds no model accuracy credit
and does not change the full producer1371ae4f or verified77/165 coverage.

Task348's data producer cf05378c now completes original, fresh rebuild/repeat
and independent source recount. All52 artifacts and the executable are exact;
48 datasets reproduce96 original split fingerprints, while all six unavailable
oracles retain explicit refusals. The old22 controls match independently, and
26 additions are ready for future model tests. Five tests, fmt and clippy pass.
Original/repeat take4.43/5.03 seconds; peak memory stays below11MiB. The manifest
is30be56ecda6b77edaea1958f91e5e15197c348be551d9b673633a2e55d5492bc.
These are data results only; the77/165 verified model count is unchanged.

Task349 starts Lean first for the numerical consistency failures exposed by
the immutable347 original scores. CM63's18 theorems give exact complete
real/complex contractions before common positive normalization, preservation
of raw ordering/ties and common-shift ordinary CE, plus an interval winner
certificate with actual enclosure premises. A feasible two-program witness
fits nonconstant observations but has an exact unseen tie and two different
realizable target completions. Arbitrarily small residuals permit opposite
strict winners. This distinguishes numerical precision from
program identification. The13961-job full build and all18 direct standard-axiom
checks pass. Global audit remains36 sorry,0 rests and0 extra axioms. Generated
INDEX has4864 modules25519 theorems,0 vacuous and0 placeholder; the native
forbidden scan is empty. A separate native correction follows this proof.
No347 producer is changed.

### 2026-11-15: task349 certified numerical component after CM63

CM63 at18308305 passed the full Lean build,18 theorem axiom checks, the global
axiom audit and native index regeneration, preserving36 existing sorry statements.
The separate `convex_stress_certified_head` component evaluates the actual complete
saved coefficient tensor with outward numerator enclosures and independent exact
rational real/complex products on ambiguous queries. Common-shift normalization
preserves ordinary CE; exact ties remain explicit and lost strict float gaps are
refusals. Nine native component checks and strict Clippy pass, including original
forward/full/minibatch policies and an unsupported categorical entry. No target
label or task identity enters inference. The frozen347 producer remains intact;
its independent13-case repeat is running. No new accuracy credit:77/165 remains
the independently verified cohort union. Next: freeze the component, diagnose
numerical failures without target labels, then freeze/repeat the full comparison.

### 2026-11-15: task349 full numerical comparison implementation

The certified component atd1b10e2e resolves five old real/complex winner
inconsistencies on74 explicit input-only diagnostic queries. All query outputs
independently repeat from the frozen source; no target labels or accuracy are
consumed. Findings are recorded atc3903117.

`convex_stress_certified_compare` now implements the complete13-case, six-endpoint
comparison with unchanged347 discovery/training, verified against original state
hashes. It separates all-heldout raw99 from exact99 and recomputes numerical
certificates and original-row scores from raw outputs. Twelve native tests,
strict Clippy, and release builds pass. A data/shape-only preflight admits all13
cases; maximum planned buffer48136330224 bytes stays below56GiB and the process
cap remains64GiB. Exact fallback has an explicit4096-successful-query budget per
endpoint plus one accounted refusal query. Ordinary CE and existing modes remain.

The independent347 repeat is still running; its verification hash is explicitly
pending, and the freeze script rejects this incomplete protocol. After that
actual verification exists, pin it, rerun checks, commit and freeze the final
producer, then execute the original comparison and an independent rebuild/repeat.
No new349 heldout score or catalog credit is claimed; verified coverage stays77/165.

### 2026-11-15: task350 training-only search continuation

The current depth3 bank contains12 fitting programs of56--64 AST nodes. Its
first-fit stopping rule is independent of the numerical evaluator; continuing
search is a separate candidate-discovery experiment. Before implementation, the
unchanged CM62 streaming/count-fit and negative-cache Lean modules were rebuilt
and their standard-axiom audit rerun. No new pruning/equivalence or unseen-target
identification theorem is claimed.

`convex_stress_search_continuation` continues successful query streams, retains
up to32 distinct completed fits by actual syntax size/order, and preserves them
when the40T search budget is exhausted. Final two-interpreter checks are reserved
inside the budget; all truncated anchors and stopped streams remain explicit.
Four native controls pass, including a real interrupted stream retaining its
completed fit, later simpler syntax, negative-cache retests and fatal-error
classification. Freeze and independently repeat the training-only1370-record
probe next. This changes no frozen347/349 producer and adds no accuracy credit.

### 2026-11-15: task347 complete independent verification

Both frozen13-case runs complete, and the verifier independently rehashes all218
artifacts with exact agreement. Verification SHA256 is
`4f40b9f20d216f5d1564ff76683733adfd5e3c08d681b0f44b116c82b95ff002`.
The first11 cases remain100% raw/exact on every heldout split and all six
endpoints. Depth3 Coordinates Full reaches98.9415% worst-split raw and92.1875%
exact; Forward/Mini retain independent-product winner refusals. Categorical
controls cross raw99 but fail exact99. The random case retains its training-fit
refusal. Catalog coverage remains77/165 with no new case or regression.
Original/repeat wall times are1:04:13 and1:04:37; peak RSS stays below4GiB.
Task349 can now bind this actual verification before its own producer freeze.

Task349's final protocol pins that actual347 verification hash. Native checks
are rerun before committing and freezing this complete producer. The subsequent
13-case/6-endpoint original and independently rebuilt repeat must preserve all
original training hashes and report raw99 separately from exact99. No corrected
accuracy result is assumed from the successful input-only numerical diagnosis.

### 2026-11-15: task351 bounded joint latent confidence, Lean first

CM64 extends CM53's fixed total mass to an explicit interval while preserving
joint decoder coefficients. Its33 proved theorems establish exact elimination
of inconsistent fixed candidates, bounded-domain feasibility, complete complex
column bounds, convex ordinary full/minibatch CE, and accepted-query transfer.
The observed CE is a decreasing function of learned total mass. A nonconstant
many-to-one witness strictly improves CE at larger confidence while retaining
two opposite feasible answers at a genuinely unseen state. Structural
realizability and observed coverage remain separate from flat-spectrum bounds.

The full13968-job Lean build and all33 standard-axiom checks pass. Global audit
remains36 sorry,0 rests,0 extra axioms; the native generated INDEX reports4871
modules,25552 theorems,0 vacuous and0 placeholder, and passes its check. Every
new hypothesis has a concrete example; forbidden and whitespace scans pass.
The next native extension must fix its input-only candidate family before
labels and preserve every compatible candidate or prove exact elimination.
Task348 supplies48 original datasets:22 old controls and26 additional cases,
including23 nonconstant cases and3 constant power-zero controls. Six cases
retain unavailable-source refusals. No new model score or catalog pass yet.

### 2026-11-15: task350 continuation probe independently reproduced

Frozenf5fb5f43 and a fresh offline source rebuild produce identical full reports,
SHA25607617c92e2814bb9c0e93aad5bda8653ccac675f5609ef5c8fee8f555ec012f5.
Five successful queries offer60 completed fitting roots;32 distinct ASTs remain.
All1370 observed labels fit both interpreters and the joint decoder compatibility
check. The minimum syntax stays56 nodes: this probe finds no smaller program.
Search stops inside branch12 at the work cap with completed fits preserved;
final charged work39999999938512 includes the reserved final audit. Original
and repeat take19:36.52/19:58.66 with about1.52GiB RSS and an8GiB address cap.
The fresh build passes four native tests and strict Clippy. Different source
checkout paths yield different binary bytes, with both hashes retained; the
complete behavioral reports match exactly. No heldout data or new score was
used, and the77/165 verified catalog count is unchanged.

Task351's native source code is now justified for arbitrary padded state sizes:
CM64 adds complete block-diagonal unitarity, replicated four-point Fourier
unitarity, and its8R-per-block complete column bound. A nonconstant many-to-one
witness satisfies the instantiated domain. All36 CM64 axiom checks and the
13969-job full build pass; the global audit remains36 sorry/0 rests/0 extras.
Native INDEX generation/check reports4872 modules and25555 theorems. The Rust
core will retain the full joint variables and explicitly audit its numerical
projection; these new results do not claim a floating-point projection theorem.

### 2026-11-15: task351 full joint native decoder component

`convex_stress_bounded_latent` implements every shared mass and every latent/raw
decoder coefficient, after the36 CM64 Lean results. Actual full/minibatch CE
gradients are checked against complete complex-head finite differences in
every coordinate. Projection handles nonuniform unknown rows and global mass
bounds, with independent numerical feasibility/KKT gates. Forward makes zero
updates. Original duplicate records and the short minibatch tail retain their
actual weights. Eleven native controls pass, including exact complete-head
ambiguity evaluation and two indistinguishable observed heads with opposite
unseen answers. This supplies no projection refinement or target-realizability
certificate. The fixed generic candidate family and complete48-case original
cohort remain pending; verified catalog coverage stays77/165.

The native pair-family component fixes4,767 input-only programs before original
labels: bounded Laurent monomials/two-term sums and ordinary integer primitives.
All compatible candidates remain; every zero block has a two-record collision
witness checked by a second interpreter. Complete retained-state encoding is
independently checked too. Five component controls pass, including all-survivor
retention, forged-subset refusal, label renaming, noninjective decoding and
actual forward/full/minibatch training. Freeze this source and independently
repeat a training-only48-pool probe before the separate complete model runner.
The structural family is distinct from the proved flat-spectrum capacity bound;
neither native compatibility nor Lean CM64 supplies unseen target realizability.

The frozen `e12200b2` pair probe independently reproduces all99 artifacts.
All48 original training pools are attempted;41 encode, including all22 old
controls and19 additions (16 nonconstant and three constant). Seven additional
cases have empty support, with every collision witness retained. All constant
controls keep4,767 candidates. The largest full-model buffer forecast is
24,863,859,328 bytes; the training-only probe peaks at691MiB. It reads zero
heldout pools and adds no model accuracy or catalog pass. Complete trained-state
sealing and independently repeated original heldout scoring remain next.

### 2026-11-15: updated user resource limits

The user raises the RAM limit to110GB: subsequent experiment processes may use
at most110,000,000,000 bytes. Existing frozen producers retain their stricter
64GiB caps. The user explicitly identifies this as RAM, not a disk quota.
Check both the workspace filesystem and the storage filesystem before admitting
new runs, accounting for original outputs and the independent repeat. Stop if
either filesystem cannot accommodate the required work. Do not delete existing
evidence to make a run fit. Native temporary buffers and concurrent run reserves
must still be included in admission.

### 2026-11-15: user-authorized cleanup of old experiments

The user explicitly requests removal of old experiments' generated outputs,
keeping code and summaries. This supersedes the earlier blanket retention of
old raw artifacts. Prepare and verify compact numerical summaries, historical
verification records and source snapshots before deleting inactive outputs.
Keep running comparisons and their actual dependencies until they finish.
Retained code and authored experiment READMEs remain in the repository; the
native `scripts/retain_experiment_summaries` tool archives additional source
snapshots and compact findings without deleting anything itself. Do not present
historical verification records as a fresh raw-artifact audit after cleanup.

Cleanup completed for all217 checked inactive output directories on both
filesystems, after preserving and independently hashing102896 source-file
entries,11145 compact result summaries and12510 provenance records. Earlier
inactive build caches were also removed. Original35 per-task accuracy and the
complete77-case report/verification match their pre-cleanup values. The archive
index, deletion receipts and readable summaries live in
`/run/media/misha/storage/transformer-runs/retained-summaries/2026-11-15/`.
Active349, its frozen source/executable, current348 data and351 pair-probe inputs
remain available. Continue351 from these inputs and retained historical
summaries; old raw-artifact paths in prior protocols need reproduction before
they can support a fresh replay. The verified catalog count remains77/165.

### 2026-11-15: task351 complete bounded comparison runner

The native runner preserves the complete CM64 joint tensors in bit-exact
binary64 storage and binds all54 original cases, including six unavailable
sources and seven observed empty-support cases. All training outcomes are
sealed before any original heldout scoring. Full, mini and forward policies
remain the frozen pair-family policies; raw readback recomputes all logits,
original target counts and ordinary CE. Ten native controls cover actual
updates, tails, wrong labels, the99% boundary, rehashed tampering, refused-model
credit, incomplete seals and both-disk stops. Historical77/35-case numerical
summaries and transformer references survive the authorized cleanup with
explicit provenance. The original/repeat output forecast is60,494,708,126 bytes
plus reserves; freeze this tested producer, execute every case and independently
repeat before accepting any additional accuracy or catalog coverage.

### 2026-11-15: task349 certified comparison independently reproduced

Frozene68747ca and its byte-identical independent executable reproduce all221
complete artifacts. Verification SHA256 is
719945303ab0f741d6452dbe0090c508d4e1655cd36eb3d39365c9e754588a9b.
The depth3 numerical refusals disappear in all six endpoints; each uses57
exact fallback queries with zero genuine ties. Coordinates worst raw accuracy
is1963/1984 (98.9415%) in forward/full and983/992 (99.0927%) in mini. Worst exact
sequence accuracy remains92.1875% and93.75%, respectively. Normal test CE is
0.04732585/0.00342399/0.00319021. The categorical control reaches raw99 in all
three modes but not exact99 on every heldout split. Every mode retains12/13
original primary successes, with RandomLM still a fit refusal. No new catalog
case or regression is credited: coverage stays77/165. Original/repeat finish
in1:14:11/1:13:54 under64GiB; peak3.86/1.54GiB. Task351's complete bounded
comparison is now frozen atca6a4977 and running; its results remain pending.

### 2026-11-15: task352 exact observed-row reduction before a wider family

CM65 proves a bijective affine coordinate reduction of the full CM64 domain:
known decoder rows are reconstructed from their shared masses, while every
unknown raw-class row remains a variable. All complete complex coefficients,
ordinary full/minibatch CE and the spectral capacity bound are preserved.
The exact induced Euclidean metric weights each mass by one plus its number of
observed states; an unweighted reduced optimizer must not be presented as the
same trajectory. The nonconstant witness retains opposite unseen answers and
has full/reduced squared distances 20/8. This proof supplies a basis for testing
a larger fixed input family within the RAM limit. It supplies no new catalog
pass. Keep the frozen task351 producer and its independent repeat unchanged;
any native compressed representation needs its own checks and future freeze.

### 2026-11-15: task353 exact reduced projection before native storage changes

CM66 extends the observed-row reduction to arbitrary full gradient proposals.
Completing all squares gives weighted reduced centers and a point-independent
remainder containing every eliminated off-label square. Exact minimizers
correspond on the full and reduced domains. The nonconstant example has unequal
weights, a negative proposal mass and nonuniform unknown entries. Implement the
Rust representation and independently validate dense/reduced projections and
complete complex products before any wider-family data run. Numerical KKT
checks are not a Lean floating-point refinement. The frozen task351 comparison
and its queued independent repeat remain unchanged.

### 2026-11-15: task353 reduced native component

The Rust component stores every unknown decoder coefficient and reconstructs
anchored rows exactly. Its sparse block-Fourier head omits only exact zero
columns; all logical coefficients remain available and included in the audit.
CM66 weighted projection retains selected proposal sums and complete unknown
simplex rows. Twelve controls compare full coefficients, logits, derivatives,
arbitrary projections, three actual training modes and checkpoint integrity
with the dense implementation and exercise both-disk stops. Buffer admission
is 96 GB within the 110 GB process limit. Guarded checkpoint writes check both
disks before creation, during writing and at completion, reserving the supplied
remaining cohort forecast and its repeat; the outer runner applies the process cap.
Keep the current task351 freeze intact. Extend the fixed input-only family and
freeze a new full comparison only after this component's independent review;
these synthetic controls add no stress-task coverage.

### 2026-11-15: task354 fixed-family inclusion before a wider native bank

CM67 embeds every old complete joint tensor into a disjoint union of fixed
input programs. All new coordinates are zero at that embedded point, while
the enlarged convex domain keeps them free. The actual mass, every complex
logit and all original indexed batch CE values are preserved at the same
spectral bound. Exact filtering retains every compatible old and new member.
Nonconstant examples preserve imaginary coefficients and opposite unseen
answers; no optimizer or new-label guarantee is inferred. After the Lean
checks, implement the complete three-term extension of the existing monomial
family and independently probe all original training sets with reduced-buffer
admission. Freeze before reading those training rows. Keep the task351 producer
and its queued repeat unchanged; a later full wider-family comparison requires
its own freeze and all training outcomes sealed before heldout scoring.

### 2026-11-15: task354 complete three-term native family

The new fixed bank retains the old 4,767-member prefix and all 294,784 declared
three-term additions. Eight native controls pass, including complete family
enumeration, independent interpreters, pairwise conflicts, label renaming,
duplicates, forged-subset refusal, full reduced-domain reconstruction and three
actual training modes. The probe streams every observed state and complete
dictionary before forecasting expanded model buffers; it can retain a complete
audit and refuse an oversized model without allocating it. Both-disk admission
reserves original and repeat outputs. Freeze this source before probing the
48 original training extracts; preserve all 54 inventory outcomes and give no
heldout accuracy credit. A separately frozen complete comparison remains
required after independent probe reproduction.

### 2026-11-16: task355 reduced wide-family comparison adapter

CM65/66/67 now have a complete native comparison adapter for the original
54-case cohort. It preserves every unknown coordinate, reconstructs and audits
the full complex head from bound binary64 checkpoints, uses the original
full/minibatch/forward schedules and enforces the pre-heldout training seal.
Memory refusals retain their complete support and original denominators.
Thirteen boundary controls pass, including corruption, tails, raw99 thresholds,
wrong scoring labels, incomplete seals and either-disk exhaustion. Both runs
are forecast before writing under the 110 GB process cap. The production
protocol remains disabled until task354's independently repeated training-only
probe supplies its final immutable digests. Commit and freeze that finalized
protocol before fitting any original case; no new heldout accuracy or catalog
credit follows from this component work. The existing task351/354 freezes stay
unchanged.

### 2026-11-16: task356 packed observations, Lean before native memory changes

The complete large constant control's conservative expanded-record forecast
already exceeds 110 GB at its minimum padding cost. CM68 proves lossless UInt16
encoding under explicit range bounds, valid-word rejection, identical observed
dictionaries and retained candidates, the same full joint domain, every complex
logit and ordinary indexed batch CE. It retains original IDs, duplicates and
tails, with nonconstant complex and boundary witnesses. Implement packed
observations next, preserving all free decoder coordinates and actual numeric
policies. Compare against the expanded record path, account for decoded row
scratch, and keep the original task354 audit distinct from refined comparison
admission. No existing frozen run changes and no benchmark credit follows from
these prospective memory identities.

### 2026-11-16: task356 native packing and separate comparison admission

The Rust path now stores every observed state and label as a checked UInt16
word and decodes at most one original row per visit. Fifteen core controls and
eight wider-family controls preserve complete domains, all coefficient bits,
ordinary CE, actual update traces and saved parameters across expanded and
packed storage. The original state and dictionary digests, selected support
and legacy memory refusal remain unchanged. Thirteen comparison controls
exercise separate packed admission while retaining oversized-unknown refusals
and all original denominators. The large constant control's forecast becomes
89,281,189,304 bytes; no actual peak or new accuracy has been measured. Keep
production disabled until this change is independently rebuilt and reviewed
and the task354 probe repeat is verified. Then pin those exact results, commit
and freeze before any original fitting. Existing task351/354 producers remain
immutable. Both disks must continue to fit original/repeat outputs under the
110 GB process RAM limit; otherwise stop and retain partial evidence.

### 2026-11-16: task354 independently verified; finalize task355 protocol

The frozen wider-family original and independent source rebuild/repeat match
all102 artifacts. All48 available training extracts completed:43 nonempty
supports versus41 previously,42 legacy memory admissions,5 empty supports and
6 unavailable sources within the54-case inventory. The added compatible cases
are arithmetic-cubic-mixed and arithmetic-quadratic. No heldout labels, training
updates or accuracy measurements were used. The probe's p193 memory refusal
remains recorded even though the separately reviewed packed representation
admits its complete model. Task356 is done at e5ef378a after independent native
and Lean checks. The task355 protocol now pins the exact verified probe digests.
Check and commit it, freeze from clean source, then execute all54 cases and all
three modes with the complete pre-heldout training seal. Rebuild frozen source
independently and repeat/replay everything before catalog credit. Keep the
running351 producer and its queued repeat unchanged.

### 2026-11-16: task357 uniform projections over the unchanged full domain

CM69 proves the exact row variance decomposition and its consequence for every
full Euclidean minimizer with class-uniform unknown proposal rows. It retains
the entire CM64 domain and CM66 weighted mass objective. Nonconstant complex
witnesses exhibit an unseen tie and surviving opposite nonuniform completions;
a nonuniform self-projection shows why the proposal condition is necessary.
The 27 theorems include zero mass and actual raw-class cardinality. Check the
direct axioms, full build, native index and global audit before landing. Any
future native storage change must retain arbitrary nonuniform coordinates and
receive its own controls and producer freeze. No running comparison changes,
general-iterate uniformity claim or new catalog credit follows from this proof.

### 2026-11-16: task358 fixed compositions for remaining empty supports

Reapply CM67's proved inclusion theorem before native work: its thirteen axiom
checks pass at ac16339c. The new crate retains the entire 299,551-member family
and adds complete finite grammars for conditional inputs, residue composition,
integer digit extraction and four-term nonnegative-exponent polynomials, for
962,806 candidates. Selection and packed materialization reuse the reviewed
engine with this new family; no decoder coordinate or record is removed. Check
independent interpretation, complete enumeration, every collision, actual
three-mode complex training and resource refusal before committing. Then freeze
and independently repeat the full 48-pool training-only probe, retaining all
54 inventory outcomes. No new heldout result follows from compatibility. Keep
the running 351 and 355 producers and queued repeats unchanged. A future full
comparison requires another freeze and a complete training seal.

### 2026-11-16: task359 complete composition comparison adapter

Recheck CM67 before applying the original comparison to all 962,806 fixed
candidates. Reuse the reviewed complete model, checkpoint, resource, scoring
and summary modules; retain every compatible candidate and arbitrary unknown
coordinate. The new strict typed probe reader preserves all report fields and
checks the complete candidate partition without cloning a generic JSON tree.
Fifteen native controls cover the larger report limit, tampering, all three
actual training modes, original duplicates and tails, genuine unseen ambiguity,
raw99, training seals and both-disk stops. Complete oversized supports remain
refusals; a larger input family alone cannot guarantee memory admission.
Keep production disabled until task358's full independent probe is verified
and pinned. Then finalize, check, commit and freeze before original fitting,
seal every training endpoint before scoring, and independently rebuild/repeat
the complete cohort. The running351/355/358 sources and outputs stay immutable.
No new heldout result or catalog credit follows from this adapter work.

### 2026-11-16: task360 lossless repeated-row theory before native storage

The full composed p97 constant probe retains all962806 candidates but exceeds
the existing complete-buffer forecast. CM70 proves generic baseline/exception
row reconstruction, injectivity and explicit value-count bounds before changing
the numerical implementation. Apply the codec to every unknown decoder row and
every complete real and imaginary head row: the full CM64 domain, projection
distance, original logits, logical budget and indexed ordinary CE are unchanged.
Uniform unknown rows give a conditional head-storage bound based on the known
labels in an actual source column's support; arbitrary nonuniform rows remain
fully representable. Nonconstant, opposite-answer, imaginary-coefficient,
word-level signed-zero/subnormal, duplicate-record and short-tail witnesses
accompany all27 theorems. Check the complete Lean build, axioms and generated
INDEX before committing. Native bitwise equivalence, actual memory admission,
independent review and a new frozen original/repeat remain required. Existing
351/355/358/359 sources and outcomes stay immutable; no accuracy credit follows
from representation theory alone.

### 2026-11-16: task360 native row storage foundation

CM70 is committed at cc6931a4 with27 direct standard-axiom checks and no
warnings. The Rust parameter codec now shares exactly equal binary64 words
and retains a full explicit fallback for every nonuniform row. It preserves
original row IDs and visits every logical class coordinate in original order.
Local admission includes container capacities, scratch and both old and new
dense buffers during growth. Six controls pass, including signed zero,
subnormals, arbitrary nonuniform fallback, stateful mapped updates, repeated
floating sums, allocation refusal and full prior-model complex reconstruction.
Commit this bounded storage component after native and mandatory Lean checks.
Continue with complete head storage and actual optimizer/checkpoint integration;
aggregate all live allocations under110 GB before admitting original models.
The full three-mode comparison and independent repeat are still required for
task360, and verified catalog coverage remains77/165.

### 2026-11-16: task360 complete compact complex head

After the CM70 proofs and row codec, implement actual compact parameter
initialization, anchored expansion, feasibility sums and the full CM65 distance.
The complete head stores one eight-word baseline per Fourier block and every
exceptional raw class; it preserves the prior logical active support and all
original floating operations. Interval separation and exact dyadic predictions
use every logical column. Fifteen native controls compare every coefficient,
all64 mixed-domain queries, ordinary CE, certificates and operation charges,
including nonuniform rows, zero mass, subnormals and arbitrary proposals.
The 128-state/1024-class uniform control stores256 head words rather than262144
while visiting every class. Both old and new buffers count during growth.
Complete the required native/Lean checks and commit this coherent head change.
The current legacy dictionary admission remains unchanged; do not claim that
larger original models fit. Next integrate actual optimizer and checkpoint
paths, aggregate all live memory, then independently rebuild and freeze the
complete three-mode original/repeat. No new accuracy credit follows; task360
stays open and all live351/355/358 producers stay immutable.

### 2026-11-16: task360 compact projection and actual CE training

Port the complete CM66 weighted proposal, scalar projection, numerical KKT audit
and all three ordinary CE policies onto CM70 storage. Keep every arbitrary
unknown coordinate and the original arithmetic. Entirely uniform programs
retain repeated-add prefix totals; mixed programs keep all sorted/prefix arrays.
Charge the full fallback, concurrent multiplier vectors and class scratch.
Training reserves dictionary reconstruction, packed record expansion and every
step before initialization, then subtracts all simultaneously live parameters,
gradients, proposals and old/new heads at each nested allocation.

Twenty-three native controls cover48 complete training combinations with
original duplicates, singleton/three-record minibatch tails, both record
representations, uniform/nonuniform starts and learning rates3 and1000000.
Final parameters, complex coefficients, step metrics, CE, operations and queries
match the prior reduced implementation bit for bit. Separate actual-step replay
checks every intermediate parameter and head. Projection controls preserve
all multiplier regimes and rounding-sensitive uniform prefix sums and reject
oversized fallback without changing the proposal. Commit after required native
and Lean checks, then rebuild independently. Whole-process input/runtime memory
admission, persistence and a separately frozen original/repeat remain required;
no new original-data accuracy is claimed and task360 remains open.

### 2026-11-16: task360 compatible compact checkpoint streaming

After compact training at47db6ff5, keep the exact RDLT6401 checkpoint format
while streaming every logical parameter into and out of compact storage.
Admission includes the complete read dictionary, masses, class scratch and
arbitrary nonuniform fallback. Verify the declared unknown-row count before
allocating derived indices. Keep the existing both-disk guard, independent
repeat reserve, exact-length check and refusal to overwrite.

Twenty-eight native controls now pass: serialized bytes and hashes match the
prior complete model; both readers reconstruct every word and complex product.
Corrupt dimensions/counts/hashes, rehashed invalid parameters, incomplete data,
oversized complete fallback and injected write/flush failures are rejected.
The uniform1024-class stream remains over1MB on disk while loading under24KiB
of local output/scratch; input bytes belong to the caller. Commit after native
and Lean checks and rebuild independently. The conservative legacy domain
admission still blocks larger original models; complete input/runtime/process
admission and a frozen original/repeat remain required before accuracy credit.

### 2026-11-16: verified probes and the current two-mode requirement

The user permits new work to require minibatch and forward-only minibatch,
with full batch optional. Task 362 is independently verified at producer
`1c67b444`, with findings at `31c209ed`: five available original probes reach
100% raw and exact accuracy in both modes on every original split. All 66
artifacts match the complete independent repeat and raw replay. Routing retains
its original teacher-data refusal in task 361. Conditional amplitudes and order
discrimination add two cases, giving a historical union of 79/165 across
separate seed 1 cohorts under the two-mode requirement; the older three-mode
total remains 77. This is not a unified model or an additional seed. Preserve
all existing Transformer records and their original success criteria.

### 2026-11-16: task 363, exact ordered column computation

CM71 is proved first at `91281572`: identical complete event sequences yield
identical accumulator words without reordering updates or replacing inactive
skips by zero additions. A conservative global union of coefficient exceptions
and known-only active classes supplies a sufficient sharing condition. The
complete class vector, arbitrary fallback, coefficient tensor and norm remain.
The full Lean build, 12 direct axiom checks, audit and generated INDEX pass with
no new sorry or extra axioms.

The corresponding bounded native forward contraction passes 46 controls,
preserving every logit/CE/update/checkpoint word and original work charge against
the full model. Actual shared nonuniform training and memory refusals are
covered. Bounded synthetic constant-shaped contractions improve by 23–44 times;
this is not an end-to-end training speedup. Commit after native checks and retain
the generated evidence outside the worktree. Existing frozen 351/355/360/362
producers and outputs stay immutable. New original measurements require a new
source freeze, complete training seal and independent repeat/replay. Continue
under aggregate 110 GB RAM and stop on shortage of either disk; no new accuracy
credit follows from a proof or synthetic timing result.

### 2026-11-16: task 364, complete construction and certified prediction

CM72 is proved first at `1279ab89`, following the fixed phase profiler at
`90a2d178`. Nine additional theorems cover word-level anchored construction,
conditional uniform-row sharing, independently recoded/reversed certificate
passes and complete joint readouts. Nonuniform unknown rows retain a witnessed
fallback. Full build, direct axiom checks, audit and generated INDEX pass with
36 existing sorry, zero resting declarations and zero extra axioms.

The native extension keeps every coefficient and class while sharing only the
proved equal computations. Forty-nine controls compare complex words, interval
endpoints, exact integers, all tied maximizers, original training trajectories
and checkpoints. Seven alternating before/after profiles on one CPU retain
every fixture pair and bind all endpoint words and work counts. Synthetic
complete forward runs improve by 1.30–1.34× and minibatch runs by 1.24–1.26×;
construction and certification gains are larger but are reported separately.
Projection and full audits remain costs. Independent review at `240cb9b4`
passes all 49 controls in both profiles, the 32 dependent controls, strict
Clippy, full Lean build and audits. Another 28 paired fixtures preserve all
words and work, with complete forward ratios 1.28–1.31× and minibatch ratios
1.24–1.28×. A new complete original freeze/repeat remains necessary for new
accuracy credit. Existing frozen producers remain fixed,
the verified historical two-mode union remains 79/165, and routing's data gap
remains task 361. No synthetic timing is counted as a stress-test pass.

### 2026-11-16: learn shared convex interactions from ordinary answer loss

Status: planned, following the user's request to move beyond fixed program
discovery and reread the flat-spectrum results. The task is to develop a shared
convex learner whose answer loss learns routing and output interactions from
categorical inputs. Begin with a nonconstant identification example and native
controls before a separately frozen original-task pilot. Existing comparison
producers, recorded failures and the historical 77/165 coverage remain intact.

The mathematical starting points are [flat spectrum](review/flat_spectrum.md),
CR1-CR7, CC1-CC8, MF1-MF5, SB1-SB3 and CM64-CM70. Theorem A concerns
minimum-nuclear-norm recovery of a specified flat kernel; it does not prohibit
all convex learning. CC1-CC2 already learn the complete joint tensor
`W(q,k,v,c)`, with logits `sum_k W(q,k,D(k),c)` and a label-independent convex
domain. [CC5](src/Transformer/Grokking/FlatSpectrum/ConvexChannel/Identification.lean)
identifies routing and generation under explicit realizability and separation
conditions. Those hypotheses and the supplied input representation must remain
visible. Flatness and sufficient rank alone do not identify unseen answers.

The current CM64 model imposes
`gamma[j,p_j(x_i),c] = w[j] * 1[c=y_i]` before CE optimization. Its theorem
[`observed_batch_ce`](src/Transformer/Grokking/FlatSpectrum/ConvexDepth/BoundedLatent/Confidence.lean)
therefore makes every nonempty observed batch loss exactly
`log(1 + (V - 1) * exp(-sum_j w[j]))`. The loss distinguishes total confidence,
but neither compatible programs nor unknown decoder completions at equal
confidence. Increasing epochs, widening the grammar or compressing its storage
does not change this identification limitation. Projection can move additional
coordinates without making the empirical loss informative about them.

The proposed learning problem must meet these requirements:

1. Specify a common categorical or causal input encoding, its interaction order
   and resource bounds before fitting. Task identifiers, target formulas,
   selected answer programs and held-out labels must not enter its forward map.
   Any fixed positional, pairing, copying or class-coordinate prior must be
   declared. Removing label anchors while retaining the existing program bank
   is a fixed-feature control, not completion of this task.
2. Learn shared joint interaction and output coordinates with ordinary original
   answer CE. Replace hard observed-label anchor equations by supervision in the
   loss; use a label-independent feasible domain and initialization. Prove
   convexity in every optimized coordinate and preserve the actual complex-head
   computation where used. Products of separately learned factors need their
   own justification; the current joint-coordinate theorem does not cover them.
3. Establish sufficient representational capacity without assuming that a
   spectral bound determines the answer. Use the flat-spectrum rank/error
   bounds on the actual represented operator. Any spectral radius, sharing map
   or regularizer must be specified without missing labels. A table with an
   independent parameter for every input is only a capacity control until a
   sharing and identification argument connects observed and unseen inputs.
4. Prove a small nonconstant learning result: state the target family and an
   explicit observation/separation condition, then derive correct predictions
   on previously unobserved inputs from empirical optimality. Include an example
   satisfying every premise and retain an ambiguous-data counterexample. Keep
   parameter identification, decoded-answer identification and convergence of
   the actual update rule distinct. Fixed categorical input features do not
   establish unrestricted learned-encoder or full-catalog generalization.
5. Implement the proved model and its training in Rust/Kernite. Minibatch CE
   must update routing/output interactions rather than only total confidence;
   preserve original record multiplicities and short tails. Address MF1's
   singleton-oracle failure explicitly; stale gradient memory alone is not a
   convergence proof. A forward-only method must specify how observations
   change its learned parameters. Full-batch is an optional reference. Report
   any unresolved learning or convergence obligation as such.

Acceptance proceeds in this order:

- Theory: reread all new statements beside the cited sections; build the new
  modules and full library with `LAKE_ARTIFACT_CACHE=false`; run direct and
  global axiom checks, native forbidden checks, and generate `INDEX.md` for
  every `src/` change. Preserve the repository's existing unfinished results.
- Native controls: check derivatives or the claimed forward-only update,
  independent predictions, feasibility, actual parameter movement, initialization
  independence from labels, contradictory-label behavior, duplicate/tail
  handling, checkpoint replay and complete memory/work accounting. Demonstrate
  a change in relative class predictions from training at fixed confidence,
  rather than treating a lower confidence-only CE as representation learning.
- Learning controls: compare the anchored fixed-program reference, a free
  decoder on fixed features, and the proposed shared learner on the proved
  example and its ambiguous counterpart. Separate fit, optimization gap and
  unseen accuracy, and keep every refusal and failed control.
- Original pilot: after theory and native review, freeze the producer and a
  small original-task protocol with declared seeds, budgets and unchanged
  inputs, splits, full-vocabulary CE and evaluation criteria. Require minibatch
  and forward-only endpoints; retain full-batch as an optional reference.
  Rebuild independently and repeat raw scoring before crediting new coverage.
  Preserve each task's original success flag and separately report raw99 and
  exact99 across held-out splits, with the earlier three-mode union identified
  separately. Keep generated evidence local and authored findings in the new
  experiment README.

The first deliverable is the explicit shared domain, its convex answer loss
and the inhabited identification theorem. A compact general learner for all
165 cases is a later objective, not an assumed consequence of this task.

### 2026-11-16: task 365, shared answer-loss identification in Lean

CM73 formalizes the first theory deliverable using the complete joint
`Channel Unit K V C` domain and a declared input-only slot alignment. Labels
enter ordinary CE, with no observed-label anchors in feasibility or uniform
initialization. Empirical optimality, realizability, value coverage and an
observed collision at each wrong slot identify every routing/output coordinate.
The nonconstant four-record example predicts genuinely unobserved dictionaries
and queries; wrong routing has strictly worse CE at the same total confidence.
The ambiguous counterpart has two global optima with opposite unseen answers,
both producing flat output operators with squared singular value eight.

The actual full complex head has the same logits, convex CE and identification
theorem. Every real and imaginary coordinate is retained and its norm/storage
bound is stated. The full 14006-job build, 37 direct axiom checks and global
audit pass: 36 existing sorry, zero resting declarations and zero extra axioms.
Generated INDEX has 4909 modules and 25735 theorems, with no vacuous or placeholder
entries. Independent root review of `c9304c8f` repeats the full build, directly
elaborates all five new implementation modules without warnings, and passes
all 37 direct axiom checks and the global audit. Task 365 is approved. No native
optimizer convergence, learned raw-token encoder or additional catalog accuracy
follows from this result.

The subsequent user clarification below changes the next architecture target;
resolve the entire trainable path before selecting its native implementation.
Retain CM73 for the planned restricted controls. Any implementation must use
genuine minibatch CE updates, and forward-only minibatch must update learned
parameters from observations. Prove new update identities first; retain MF1's
failure and avoid treating stale-gradient memory as a general convergence
proof. Run the planned controls before a separately frozen original pilot.
The broader 165-case objective remains unchanged.

### 2026-11-16: constant object sequences with the entire model path trained

The user clarified the interface and learning requirement: the model receives
an immutable sequence of objects and predicts the next object. The input and
target objects are constants with respect to optimization. Their numerical
representation does not require a frozen feature extractor. Representation,
sequence interactions and the output mapping must all participate in training
from ordinary next-object cross-entropy. Treat CM73 as a proved restricted
control, not as completion of this requirement. Do not implement its fixed
categorical encoding as the proposed final architecture.

The next theory task must define the entire trainable computation and its
parameter domain before choosing a native optimizer. It must identify what is
fixed by the architecture or object serialization, what is learned, and how
every learned component affects actual next-object probabilities. Merely
renaming a fixed feature bank or optimizing only its output coefficients does
not satisfy the clarified requirement. A joint lifting must state whether it
exactly represents the layered family or relaxes it; a prediction-space convex
hull alone does not prove an efficient trainable representation or inference
procedure. Keep parameter count and execution cost explicit.

Do not infer joint convexity from immutable inputs or convexity of individual
layers with other weights held fixed. Also do not impose affine logits as a
necessary condition: the existing
`GPTMini.Convex.Likelihood.threeRouteCrossEntropy_convex` and
`threeRouteLogits_non_affine_witness` prove a normalized nonlinear categorical
example with jointly convex ordinary CE. This example has no sequence inputs
or learned embedding/interaction coupling. Its accompanying
`productRouteCrossEntropy_not_convex` proves that introducing two independent
query/key factors can destroy convexity. These positive and negative controls
guide the new construction; neither settles the requested architecture.

Preserve the rest of the acceptance path: Lean first, an inhabited nonconstant
identification result with explicit assumptions and an ambiguous counterpart,
flat-spectrum bounds on the actual operator, then Rust/Kernite minibatch and
forward-only minibatch controls. Use the original next-object targets only in
the loss, retain original multiplicities and short tails, and show that training
changes internal representations and relative predictions. No new catalog
credit follows from this clarification or the restricted CM73 theorem. Existing
frozen comparisons, the 99% raw/exact target and the full 165-case objective
remain unchanged.

### 2026-11-16: task 351 complete comparison and named coverage audit

The original and independent repeat at frozen producer `ca6a4977` have finished.
All 686 deterministic artifacts and complete raw replays agree. Each of
Forward, Full and Mini passes 41 of the 54 retained cases: 22 old controls and
19 additions, with no old pass-status regression. Six source refusals and seven
empty-support refusals remain; there are no further training or numerical
prediction refusals. The 19 additions comprise 16 nonconstant cases and three
constant power-zero controls. The worst passing accuracy is 99.9362% on
`arithmetic-max` (three errors in 4,705 validation targets).

The scored aggregate is 238,927 / 238,930 validation targets, or 99.9987%; counting
unanswered available targets as wrong gives 87.8844%. Weighted validation CE is
0.0436100 in Forward and 0.0000765866 in Full/Mini. The largest recorded process
RSS across both runs is 11.41 GiB. Complete source/binary hashes, 26 native
controls, strict Clippy, direct CM64 axioms and a fresh full-manifest verifier
are rechecked before task completion. Authored findings and exact regeneration
commands are in `experiments/convex_stress_bounded_compare/README.md`.

The native named-case audit reconstructs the old 77 from its preserved cohort
summaries, checks every required mode and heldout pass flag, and finds no overlap
with the 19 additions. Coverage becomes 96/165 for all three historical modes,
or 98/165 under the current Mini and Forward-only requirement after including
the two independently verified new probes. There are 67 cases without that
credit. The audit rejects missing modes/splits, overlapping additions and raw99
failures even when original success is true. This is a union of separate seed-1
cohorts; historical raw files deleted at the user's request are not reverified.

CM64 remains a fixed-family reference whose observed CE learns confidence.
These benchmark results do not close task 366's entire-trainable-path requirement.
The old tensor-stack likelihood proof also needs observed internal paths, so
it does not supply the requested ordinary next-object CE theorem. Continue with
the complete causal representation/interaction/readout construction in Lean;
do not substitute either restricted model for the clarified architecture.

### 2026-11-16: hidden-path likelihood and the shared-decoder constraint

CM74 supplies finite controls for task 366. With two learned stochastic stages,
complete-path CE is jointly convex, but ordinary next-object CE after summing
the hidden paths is not: two endpoints give probability 5/8 for the same answer,
while their midpoint gives 1/2. This prevents transferring the older
fully-observed-path theorem to the requested answer-only objective.

At one fixed input, learning a strictly positive joint hidden/output table gives
an exact parameterization of all positive untied stochastic encoder/decoder
pairs. Both recovered stages can change while ordinary answer CE decreases,
and that CE is convex in the joint coordinates. The construction has no claim
of optimizer convergence or identification of the hidden-state names. It also
does not preserve the first counterexample's keep/flip tie.

A two-input counterexample proves the remaining sharing issue. Two positive
models with a common decoder have a joint midpoint whose input/output minor is
1/16, whereas every shared-decoder factorization requires zero. The midpoint
still belongs to the independent joint simplex at each input. Thus independent
joint tables per prefix relax parameter sharing; they do not complete the
shared learned sequence architecture. Flatness does not remove that constraint.

The new source is `ConvexChannel/HiddenPath`, with direct checks in
`scripts/HiddenPathAxioms.lean` and source discussion in `review/flat_spectrum.md`
CM74. Task 366 remains open: construct the actual shared causal computation,
establish its ordinary CE geometry, and connect observed training sequences to
genuinely unseen prefixes under explicit conditions before native learning.
These controls add no stress-test accuracy credit.

### 2026-11-16: exact learned complex ReLU recovery and activation scope

CM75 constructs a shared complex ReLU family from raw bounded object prefixes.
Joint neuron coordinates recover both learned representation vectors and
readout amplitudes, with an exact full complex forward identity at every
input. Ordinary next-object CE is jointly convex on explicit input-only
activation cones. A nonconstant witness changes representation direction,
readout amplitude and relative predictions while both desired CEs decrease.
Indexed minibatch objectives retain duplicate records and short tails.

The lifting preserves every original phased two-layer prediction but widens
the recovered network to class-specific signed neurons. All-pattern expansion
places every such physical model in one common convex domain for the declared
input region and retains its predictions outside that region. It requires
2^(2N) Boolean patterns for N region inputs before any exact redundancy
elimination. Naive enumeration exceeds the 110 GB RAM cap at N=20 even with
only one complex coordinate per pattern. No dense original run is admitted
from this construction; no sampled mask family receives its coverage claim.

The complete raw-prefix theorem takes the region to be every object list of
length at most the cap. Its single domain has convex CE at every such prefix
and exact physical-model coverage there, with the full exponential coordinate
count retained. This global finite construction does not prove compact storage
or an efficient optimizer.

The restriction remains material: on a fixed cone the logits are a fixed gated
linear expression. A checked two-input example violates answer-CE convexity
at a missing input, despite exact factor recovery there. This is constructive
progress toward all-path training, not a completed efficient architecture or
an unseen-answer identification theorem. Keep task 366 open. The next work
must handle the complete learned family within resource limits, specify real
Mini and forward-only Mini updates, and derive the required identification
and actual-operator bounds before a separately frozen native pilot. Preserve
all 165 cases, the 99% target and the existing frozen comparison pipelines.

### 2026-11-16: exact reduction of the full complex pattern family

CM76 removes the Cartesian real/imaginary pattern redundancy and duplicate
hidden slots by exact sums. At every admitted bounded raw prefix, the entire
complex prediction is unchanged. A reverse embedding proves that the new
family does not relax the old complete prediction class when its hidden index
is nonempty. All classes, signs, components and scalar patterns are retained;
indexed ordinary CE preserves every duplicate and short tail.

Real coordinates fall from 4 |J| |O| (1+T |O|) 2^(2K) to
4 |O| (1+T |O|) 2^K, where K counts every bounded raw prefix. A finite example
with fifteen inputs, two classes and seven complex coordinates needs
14,680,064 binary64 parameter bytes instead of 481,036,337,152. This is not a
native peak-memory result or a stress admission. The remaining exponent,
cone constraints, optimizer state and updates are still unresolved for the
full suite. Parameter penalties and floating-point summation order need not
be preserved; do not retrofit this into frozen comparisons.

Keep task 366 open and the 99% raw/exact target on all 165 cases. Native Mini
and forward-only Mini still need executable feasible updates and complete
resource accounting. A representation witness neither replaces those duties
nor identifies unseen answers. Flatness must bound the actual learned operator.

### 2026-11-16: actual learned-prefix ambiguity and a flat observed operator

CM77 proves the missing-answer gap inside the CM75–CM76 complex network. A
permitted learned bias and position/object vector isolates one complete prefix.
Both complex components enter the original raw forward. Conic addition changes
one chosen missing-prefix logit by any real amount while preserving the whole
complex output at every other prefix. For any already fitted feasible point,
all original observed logits and indexed ordinary CE remain exactly equal.
Two alternative binary answers at the omitted prefix are therefore possible.

A nonconstant two-label witness has observed CE strictly below log 2 for both
labels and still permits the opposite unseen preferences. An operator made
from three actual neuron forwards has observed-row Gram matrix 10I, an exact
energy identity and a retained nonzero direction invisible to the observed
rows. Thus flatness is an actual operator bound here, not an identification
principle or permission to remove the missing coordinate.

No frozen pipeline or empirical coverage changes. Keep task 366 open. Next,
test explicit sharing conditions that transfer observed answers to genuinely
unseen inputs without freezing the representations or output mapping. Any
such condition must state which tasks satisfy it and its cost; it cannot be
assumed for all 165 cases. Executable complete-family Mini and forward-only
Mini updates under 110 GB, the 99% target and full raw replay remain required.

### 2026-11-16: conditional transfer with two shared learned passes

CM78 defines an actual paired complex forward from an immutable object sequence
to a next object from the same set. Both calls share every learned coordinate;
the second consistently renames input and output objects. Ordinary full-class
CE and its indexed minibatch sum are jointly convex on the full-prefix domain.
The mathematical predictor returns a maximizing original object and keeps all
classes, duplicates and tails.

For an involutive renaming, an observed CE below log 2 forces the renamed next
object at the renamed prefix, including genuinely unseen prefixes. This is
correct for a task only when its answer rule respects that renaming. A
three-object example fits two different observed labels and identifies a third
object at an unseen sequence. A separate theorem shows why fixed inputs cannot
have unique moved targets. Control identities and the symmetry assumption must
match the data; no answer rule is inserted into inference.

This explicitly restricts the original function family. Exact lifting holds
for the declared paired physical model; arbitrary original functions are not
claimed unchanged. Two actual forwards share the original exponential complete
parameter domain. This closes a conditional identification gap, not practical
complete-family training or universal task compatibility. Keep task 366 open,
all 165 cases and the 99% raw/exact target. Native Mini and forward-only Mini,
complete resource admission, frozen training/evaluation separation, independent
repeat and full raw replay are still required before any empirical credit.

### 2026-11-16: certified feasible updates for the complete learned model

CM79 gives exact finite projection certificates: primal feasibility, dual
nonnegativity, complementarity and stationarity imply global nearest-point
optimality. The rows are proved equal to the actual raw bias and position/
object lookup constraints. Component certificates enforce every class, sign,
pattern and complex coordinate on the entire bounded-prefix domain, and their
distance inequalities sum to the whole-model projection guarantee.

This is an update primitive, not an optimizer oracle or a trained-model result.
A separate projected-step descent theorem keeps its actual-loss quadratic bound
as an explicit hypothesis. Native Mini and forward-only Mini need their own
proposal rules, full-domain checks, actual forwards and measured outcomes.
Next implement an exact rational certificate checker with explicit resource
and failed-search refusals. Small complete-domain controls must retain every
pattern; they do not replace the all-165-case objective or earn accuracy credit.
Do not omit constraints or accept floating-point feasibility tolerances as
proof of the Lean premises. Existing frozen comparison producers stay unchanged.

The native `experiments/convex_learned_sequence` component now implements CM79's
exact rational certificate checks and finite active-set candidate generation.
Ten controls pass, including every128pattern of the complete two-object,
two-position domain (two proposals each), every16pattern of the three-object,
one-position domain, nonbinary rational outputs, degenerate constraints and
corrupted certificates. All original rows remain in every verification. Domain
and search budgets refuse the whole request rather than sampling constraints.
Strict Clippy and release build pass. This is a component, not a model fit:
the whole complex forward and both minibatch proposal rules are still pending.
Follow the experiment README's commands and preserve the complete family and
resource limits before a separately frozen, independently repeated pilot.

The native model now executes the original recovered complex representation,
componentwise activation and learned complex readout. It retains every joint
coordinate and checks the complete raw domain. CM76 physical lifting and CM78
paired inference preserve their stated scope. Stable full-class ordinary CE
uses targets only in the loss, and prediction returns an original object index.
Atomic whole-model projection retains and independently replays every component
certificate; failure leaves the old model intact. Rational parameters remain
exact while recovered factors and inference are qualified binary64 approximations.
Nineteen controls cover full complex reference outputs, paired equivariance,
signed readouts, empty inputs, complete certificate replay and numerical/resource
refusals. These are constructed feasible points, not model training. Next derive
and implement actual Mini and forward-only Mini proposal rules. Forward queries
must stay feasible even on lower-dimensional cones; no boundary derivative from
an outside-domain gated surrogate is authorized as a replacement.

CM80 extends the Lean update foundation with certified forward-query geometry.
Project every original constraint normal and sum those projections. The centre's
zero-slack rows are proved to be precisely the universally forced equalities.
Their joint nullspace characterizes two-sided feasible query directions, including
degenerate cones where individual coordinate probes fail. Positive centre mixtures
admit every such direction and retain explicit displacement and convex-loss costs.
The native implementation must keep all normal certificates and original rows,
account for complete basis work/storage, and query the actual complex forward.
This is domain safety, not a finite-difference error or convergence theorem.
Both actual minibatch proposal rules and the independent training pilot remain
pending; no stress accuracy follows from CM80 or its constructed controls.

CM81 adds finite certificates for complete feasible-query bases. Kernel checks,
free-coordinate identity and explicit coordinate reconstruction from every
retained equality row prove full coverage and unique coefficients. The actual
centre-derived matrix is connected to CM80's two-sided query directions and
positive iterate mixtures. A native basis generator must retain a recovery
certificate and pass a separate exact verifier; sampled successful directions
are not a complete replacement. Exact pairing agreement is explicitly required
for its work identity and is not inferred from numerical finite differences.

The native model now constructs and independently checks CM80--CM81 cone geometry.
All original normal projections, complete multipliers, equality rows, basis
coordinates and recovery certificates remain present. Exact rational queries
check every original inequality and absolute-response bound; positive centre
mixtures handle non-forced boundaries. Twenty-eight controls pass, including all
144 small-domain patterns with 960 certified normal projections, missing-basis
rejection, rational nonorthogonal directions and actual complex model queries.
This supplies executable domain-safe queries, not either completed minibatch
optimizer. Next recover update directions from all measured basis pairings,
retain every record/multiplicity/tail, and connect both proposal modes to atomic
whole-model projection. Full-family resource admission and the separately frozen,
independently repeated training pilot remain required before accuracy credit.

Native basis-pairing recovery now solves the complete exact Gram system and
independently verifies the resulting full vector against every supplied pairing
and kernel reconstruction. Off-diagonal entries are retained. Thirty controls
pass, including nonorthogonal and empty-kernel cases and complete recovery/query
checks for all 144 small-domain patterns. This supplies a direction from provided
numerical values; it does not prove those values are true CE derivatives. Actual
Mini/forward-only Mini proposals, whole-family budgets and a sealed learning
pilot are still next, with no new empirical accuracy credit.

### 2026-11-16: actual CE updates and prospective learned-sequence pilot

CM82 (`ba4b4b93`) now proves the actual ordinary CE slope, supporting inequality
and feasible-query derivative for every raw joint coordinate, both shared passes
and nonempty indexed mean minibatches. A nonzero complex witness and repeated
record/tail examples inhabit every conditional theorem. The full build, 38 direct
axiom checks, global audit and native INDEX check pass independently; existing
36 `sorry` declarations remain. No finite-step or numerical error guarantee is inferred.

Native Mini (`282f4ff1`) and forward-only Mini (`3fdba35e`) now implement complete
proposals and atomic certified projection. Forward-only code uses actual complex
loss queries along every certified direction and full Gram recovery, with no
analytic gradient path. It records the centre-mixture cost and supports complete
independent query/update replay. Total-family work/storage bounds retain all
patterns, classes, components, coordinates, record multiplicities and short tails.
Thirty-nine controls pass in fresh independent builds, including a nonorthogonal
forced-equality cell and forged receipt/certificate rejection.

The next authorized run is the separate small prospective pilot in
`experiments/convex_learned_sequence`: complete O=3,T=1, both required minibatch
modes, shared involution [0,2,1], eight epochs and batches of two, including a
repeated record and one-record tail. Commit/freeze producer and fixtures first;
all training and complete persisted update replay precede any evaluation-data
read. Independently rebuild, repeat, replay all raw outputs/CE, and retain every
failure before interpreting outcomes. The unseen example relies explicitly on
task equivariance. It earns no stress-case credit and does not replace the
all-165 raw/exact 99% goal. Complete-family exponential cost and applicability
beyond that restricted toy remain open. Older frozen producers stay unchanged.

### 2026-11-16: proved local replacement for full complex loss queries

CM83 (`ce6970ab`) proves exact replacement of one learned real/imaginary
component, preservation of every original complex-domain constraint, and the
full complex forward with only the entire changed neuron reconstructed. Both
signs and components have nonzero witnesses. All twelve new theorems, the full
build, global audit and generated INDEX pass an independent main-checkout review;
existing 36 `sorry` declarations remain unchanged.

The native local query checks the complete new vector and all its original
rows, reconstructs the whole affected neuron, and evaluates every neuron in
the original summation order. Both paired passes, all classes, ordinary CE,
indexed records and tails remain unchanged. Forty-four library controls and
five protocol controls pass, including bitwise full-reconstruction comparison
at every three-object cell and in a degenerate nonorthogonal two-position cone.
The original conservative work/storage limits remain in force.

The frozen `7881eb5d` original/repeat workflow continues unchanged. After a
separate checked commit and independent native review, freeze a new optimized
producer, independently repeat it and replay every raw output. Also compare
every full training-step artifact against the old producer, explicitly retaining
and accounting for the different source identities. Local equivalence controls
alone give no accuracy or speed claim. The all-165 target, complete-family
exponential cost and general unseen-label conditions remain open.

### 2026-11-16: independently verified learned pilot and identical faster trajectory

Both frozen producers (`7881eb5d` full reconstruction and `719ba30b` certified
local replacement) completed original/repeat training, all-training seals before
evaluation reads, and independent full trajectory/raw replay. Each matches all
34 deterministic training artifacts within its source version. Both Mini and
forward-only Mini score 3/3 original objects in both repeats; mean evaluation
CE is respectively 0.2491072676434611 and 0.1530963328629868. The unseen prefix
is covered only under the declared task-equivariant involution.

Across source versions, all 32 full step traces are byte-identical, and every
raw prediction, complex score, probability and CE bit agrees. Only the documented
source/configuration identity fields and corresponding seal identity differ.
The full identities, hashes and normalized comparisons are preserved locally;
the experiment README records findings and exact reproduction commands.
Training plus sealing, replay and scoring fell from 808.66/817.93 seconds for
the original/repeat to 28.13/27.98 seconds. These local pilot observations do not
provide a general performance guarantee.

This pilot adds zero stress cases. Keep task 366 and the all-165 raw/exact 99%
goal open. Next address complete-pattern growth while retaining the original
object-sequence interface and the stated all-trainable path requirement. Prove
any exact sharing/compression, or explicitly name a new family's restrictions
and loss of expressivity, before treating it as a scalable replacement. Continue
both required minibatch modes and the 110 GB aggregate RAM/two-disk guards.

### 2026-11-16: complete-family capacity explains the compression limit

CM84 (`4b67e11b`) extends CM77's actual learned singleton neurons to arbitrary
full-word logit tables, placed by CM76 in the same complete feasible family.
The complete actual response operator has Gram matrix `10 I` and exact energy
gain ten. Thus flatness supplies a bound while its dimension can be `|O|^T`.

Exact affine compression of every class contrast in this family requires at
least `|O|^T` real dimensions, even if its encoder is arbitrary and nonlinear.
For a fixed finite-code forward, preserving every strict classification requires
at least `|O|^T` stored model bits; below that budget one fixed feasible point's
classification defeats every code. The existing R1 information bound supplies
this second implication. Each conditional theorem has an inhabited example.
These are whole-family limits, not task-specific lower bounds, a RAM-only
obstruction, or an impossibility result for nonlinear convex-CE architectures.

All fifteen new theorems pass independent main-checkout validation: full build,
direct source elaboration, direct/global axioms, native generated INDEX and
forbidden-source checks. Existing 36 `sorry` declarations remain. Evidence stays
local under `.kanban/artifacts/366/interpolation-hinge-*`; no producer changed.
The latest upstream fetch remains `fa130a5e`.

Keep task 366 and all-165 raw/exact99% open. The next positive construction
must provide compact learned sharing suitable for the stress tasks, with any
restriction of the current universal family stated explicitly. Selecting a
smaller fixed feature bank, an independent whole-prefix table, or an unimplemented
pricing oracle still does not meet the architecture requirement. No additional
test accuracy or stress-case coverage follows from CM84.

### 2026-11-16: a learned quadratic block with explicit polynomial cost and depth limit

CM85 supplies a positive alternative to complete activation-mask enumeration.
The new `LearnedQuadratic` family jointly learns complete complex PSD matrices,
recovers actual balanced representations and readouts, processes immutable raw
sequences through squared complex responses, and returns an original object.
Ordinary answer CE and indexed mean minibatches are convex on its entire domain.
Exact physical coverage allows finite squared-response networks, with recovered
width `1 + T|O|` per signed class branch. This does not preserve a smaller fixed
rank or one representation tied across all classes. Full dense real storage is
`4|O|(1+T|O|)^2`, before recovery and optimizer memory.

A nonzero raw witness changes representation direction and readout, uses its
imaginary weights, reverses object preferences and lowers ordinary CEs. Flat
spectrum remains only a bound on the actual output operator. The price of the
restriction is formal: every logit has zero third mixed difference, and no
feasible point strictly classifies all eight three-bit parity prefixes.
This is a learned building block, not an adequate full-suite architecture.

Keep task 366 and the all-165 raw/exact 99% objective open. Next prove richer
shared interactions with their actual convexity, recovery and cost; do not
infer convexity merely by stacking learned quadratic factors. The new family's
identification conditions, numerical PSD/factor certification and both native
minibatch modes are not supplied by this proof. No new training producer or
stress-case credit is authorized by proof witnesses alone. Existing frozen
producers, independent repeats and raw-replay requirements stay unchanged.

### 2026-11-16: rectangular learned pair/query interaction and its depth limit

CM86 uses a query-sized balanced factorization of full rectangular complex
matrices. Both actual raw encoders and the readout are recovered jointly;
ordinary original-object CE and indexed minibatch means are convex. Learned
position/object and shared adjacent-pair entries retain all occurrences and
all output classes. With total prefix cap T+1 and vocabulary V, the dense
joint count is 2V(1+V)(1+TV+eV²), e=1 for T>=2. Recovery width is 1+V per
class; factor, optimizer and numerical workspace are additional.

The actual recovered encoder geometries both change at nonzero complex
parameter points. The raw model represents three-bit parity but fails on a
fixed-query slice of four-bit parity, so it remains an explicitly restricted
building block. Complete complex coefficient probes identify the joint matrix,
allowing R3 width/error bounds; raw real labels alone do not identify that
matrix or unseen answers. There is no native or stress accuracy increment.

The user now requests a separate Lean sorry and board task connecting capacity
to observable identification of unseen original objects. Prepare that handoff,
then wait for review. Preserve the existing frozen producers and the completed
proof work; do not launch broader training on this restricted family.

### 2026-11-16: new identification goal and task 367 handoff

The user cleared the earlier goal and requested a new goal connecting
flat-spectrum capacity to training-only unseen next-object identification.
CM87 now contains one explicitly requested open Lean statement,
`LearnedBilinear.Identification.forced_margin_iff_certificate`. It characterizes
forced original-object margins by finite nonnegative combinations of observed
margin coefficients in the complete actual learned family. Feasibility excludes
inconsistent-data vacuity; a separate two-class example inhabits that premise.

Task367 owns removing this sorry and proving the coefficient identity, finite
checker/duality, capacity-versus-observation connection, CE and robust margin
bridges, a nonconstant held-out witness and an ambiguous refusal. The full-suite
goal is no longer active; task366 itself is not marked complete. The new theory
goal initially requested a wait after this handoff. The later user request
`try proof it` supersedes that wait. Do not launch training for this proof task.

### 2026-11-16: task 367 identification proof

The unchanged CM87 equivalence is now proved through a finite linear margin
alternative and the actual learned complex coordinate identity. The extension
adds exact rational checking, observation-rank/kernel accounting, ordinary-CE
and weighted-error margin transfer, and nonconstant unseen success/refusal
controls. Preserve all36 older sorry and complete direct/global axiom and native
INDEX checks before landing. No numerical producer or stress credit changes.
The remaining model-development gap is discovering sufficient task structure
and reaching certified margins with Mini or forward-only Mini, not a theorem
that width or flatness alone supplies unseen labels.

The user subsequently requested renewed GPT stress testing. After task367's
proof review, resume native implementation and prospective stress experiments
with Mini and forward-only Mini, keeping the transformer baseline and original
splits/metrics. The CM86 bilinear model has not yet been ported to the native
pilot; the existing CM75--83 pilot is a different exponential family. Preserve
all full joint coordinates, complex factors, classes and minibatch tails, and
report the known four-object parity restriction alongside actual measurements.
Keep the 110 GB aggregate RAM limit, both-disk guards and frozen producers.

### 2026-11-16: complete CM88 query reuse before native stress runs

CM88 proves coordinate changes of the actual learned logits and exact reuse
of per-record class-bias CE queries for every real/imaginary coordinate. The
forward-only Mini rule averages all per-record coordinate differences using
response-dependent steps. It does not call an analytic CE derivative and does
not assert finite-difference convergence. Exact zero support persists under the
complete pullback; zero joint columns also give zero learned context rows and
zero Gram contributions. Implement implicit zeros, retain all nonzero entries
and singular directions, and report numerical factor/replay residuals. Verify
these identities and native update kernels before prospective stress scoring.

### 2026-11-16: native CM86–88 original stress pilot

Implement `convex_bilinear_sequence` with the complete joint complex domain,
actual recovered learned encoders, every original class, exact zero support,
and complete Mini/forward-only Mini pullbacks. The source protocol fixes five
original cases and both modes at seed1 before scoring. Use original batches,
target masks and short tails; finish all ten endpoints before any held-out
measurement. Preserve source success flags and separate raw/exact99 fields.
Replay every raw logit and metric from each saved endpoint in a separate
process. This initial diagnostic is not an independent training repeat or a
completed35/165 claim. The long parity control retains the proved family
restriction. Keep all frozen producers unchanged and enforce the 110 GB
aggregate RAM and both-disk admission limits without further cleanup.

### 2026-11-16: CM89 input-only update scaling

The frozen native control now fits both short Boolean training sets, while
addition's training CE barely changes in200 updates. Before changing native
updates, prove positive training-response scaling on every complete real and
imaginary coordinate, preservation of the actual joint-convex family, complete
finite-query means, duplicate/pair multiplicities and zero support. Implement
any subsequent optimization diagnostic as a separate prospective cohort;
the original control continues unchanged. Training-only improvements do not
establish unseen-answer identification or original-task success.

CM89 is now proved and independently rechecked. The native scaled diagnostic
uses `scaled_protocol.json`, a new frozen binary and separate output directory.
It keeps the addition control's seed,200 updates,rate0.01 and original batches
unchanged. Eight native controls include all complete coordinate changes under
both update modes, repeated responses and a one-record tail. Score both sealed
endpoints and replay every raw logit separately; retain the unscaled producer.

### 2026-11-16: CM90 silent-coordinate diagnosis

The scaled addition training curve improves while the original pilot continues.
Before interpreting future held-out scores, formalize an actual parameter
countermodel: a coordinate silent on every training input can move a responding
query margin arbitrarily without changing training logits. Every unseen raw
ordered two-object input has such a coordinate in the complete CM86 family.
This distinguishes optimization progress from identification; it does not prove
that no algorithm or additional inductive bias can learn the task. Retain
nonempty feasible examples and all original stress outcomes.

### 2026-11-16: learned bilinear stress pilot completed and replayed

All ten original cells and both scaled addition cells have sealed endpoints
and exact separate-process raw-logit/metric replays. Both mode queues exit0.
The original five-case seed1 cohort passes2/5 primary tests in each mode:
both Boolean AND cases100% raw/exact; addition1/4705, key-value binding155/512,
parity242/314 raw tokens and85/157 exact sequences. Long Boolean probe exact
accuracy ranges28.125–68.75%, and random-AND ordinary test exact is95.3125%.
Preserve original success flags while reporting these transfer failures.

CM89 scaling keeps the addition control's rate,batches,seed and200 updates.
It raises training accuracy4.42177%→100% and lowers full training CE4.572614→
2.604803, while held-out accuracy falls1/4705→0/4705 and CE4.575985→4.654385.
The observation confirms that fixing fitting is insufficient for this tested
update rule. CM90's silent-coordinate theorem diagnoses missing constraints,
without asserting that every possible learning algorithm must fail.

Historical transformer results remain unchanged and passed all five primary
cases at their original first99 observations. The new run is a fixed-budget
diagnostic, not a matched-budget or matched-time baseline comparison; parity
visits only0.1024 epochs. There is no three-seed,35/165-completion or independent
training-repeat claim. Save authored findings and commands in the experiment
README; keep generated evidence in external storage. Before broader runs on a
new family, establish how shared trainable structure constrains unseen answers
and how its context interactions address the proved parity restriction.

### 2026-11-17: generalization is the active goal

Task368 starts from the verified fitting/generalization separation, not a
fresh accuracy claim. Preserve the immutable-object-sequence to next-original-
object interface, every class and complete trainable complex path. First
prove CM91 capacity-constrained identification and its nonconstant missing-
pair control, with unrestricted ambiguity alongside it. Then investigate
shared learned structure and native Mini/forward-only Mini diagnostics on
unchanged original splits. A capacity theorem or training-only fit does not
complete the99% held-out goal. Keep the110GB aggregate RAM cap, both-disk
guards, transformer baseline and all frozen producers; use no Python.

CM92 supplies a shared learned output/query/context component family with
complete joint-domain convexity, finite executable representations and feasible
signed class-bias queries. Use the same capacity-reserving shrink in Mini and
forward-only Mini. Component discovery may use an explicitly approximate
training-only search; do not call it an exact convex oracle or claim optimizer
convergence. First validate full complex execution and complete finite-query
updates in Rust, then freeze a separate original-split diagnostic. Preserve
the old failed cohorts and report the unchanged higher-context parity limit.

The first frozen CM92 native cohort (`3bb8159a`, `shared-atoms-v1`) is complete.
Both modes fit 133/4704 training answers and predict 1/4705 held-out answers
correctly; all full raw-logit replays match. Both source success and raw/exact99
are false. The active-mass-weighted squared query/context bias energies are
0.9781/0.9784 in Mini. Treat this as a failed optimization/generalization trial.
Next prove a positive coordinate rescaling that retains all learned directions
and feasible forward queries before testing whether less bias dominance helps.
Preserve this producer, every failed result and the unresolved goal.

CM93 (`c7118dc1`) now proves the positive geometry and its corrected loss-query
reservation. Freeze a separate dimension-bias cohort: both input bias addresses
scale by1/sqrt(V), all other original coordinates by1. Keep both modes' common
centre, all original records, seed1,1000 updates and the same capacity scalar.
Validate full complex responses and actual feasible queries before training;
seal both endpoints before held-out scoring and replay every raw logit. This
geometry changes the bounded domain and supplies no optimizer/generalization
guarantee. The input/output contract remains opaque objects throughout.

The scaled cohort (`66b08989`, `shared-atoms-balanced-v1`) is complete: both
modes reach4704/4704 training and4705/4705 held-out raw/exact accuracy, with
held-out CE0.00112526. Both original success and raw/exact99 are true. Every
saved raw logit replays exactly. Mini query/context bias energies fall to
0.00004850/0.00005369. This is empirical generalization on the original addition
split, seed1, with all classes retained and no arithmetic rule in the model.
The active goal remains open for other seeds, MQAR and long sequence families;
the CM86 parity restriction and approximate-oracle limitation still apply.

Task369 extends the CM93 experiment to three training seeds on the unchanged
data seed1 splits: both AND tasks and addition1000updates, short MQAR200,
parity16128, in both minibatch modes. CM94 proves the exact zero-row contraction
optimization required for these larger complete tensors. Retain all stored
coordinates, all30 endpoints and every original split; seal the whole cohort
before scoring. Compare the repeated addition seed1 raw logits with the prior
successful dense producer. Keep the known parity limitation and training budget
visible, and do not convert a diagnostic failure into a reduced task definition.

Task370 formalizes CM95 learned complex recurrence while task369's frozen
cohort trains unchanged. Every input object indexes a learned complex operator;
initial state and all original-class readouts are learned. Operator capacity
bounds actual outputs independently of length, with exact state-norm preservation
for isometric transitions. The convex joint function domain has finite executed
mixtures, ordinary CE convexity and actual feasible finite bias queries.
A nonconstant complex parameter witness realizes parity at every length with
margin2, removing CM86's fixed-interaction obstruction. It is not supplied to
training and provides no new stress score or optimizer-discovery guarantee.
Implement and evaluate learning only after the complete Lean checks pass.

Task371 investigates a separate training-only optimization issue while the
three-seed producer remains unchanged. At large candidate scales the24-iteration
finite CE line search can return zero despite a strict interior improvement;
an independent mixed-label control verifies this against its known CE optimum.
Freeze a separate original AND-shift diagnostic with64 iterations,100 updates,
seed1 and both modes. Keep the same complete learned domain, all original data
and the110GB/both-disk guards; seal both endpoints before scoring and replay.
This does not address the old reader's parity expressivity limit.

The refined diagnostic (`41cffd66`, `shared-atoms-refined-v1`) completes and all
raw logits replay. Both modes reach100% raw/exact on train, main validation and
ordinary test in100 updates; source success and primary raw/exact99 are true.
Long length32/64 probes still have28.125%–59.375% exact accuracy, and validation
length64 CE exceeds126. Keep these transfer failures visible. The result supports
the line-resolution diagnosis for short fitting while reinforcing the need to
train the separately proved recurrent family. The original30-run cohort remains
frozen and continues unchanged.

Task372 implements CM95/96 after the complete Lean checks pass at680dffb0.
Use the complete scalar complex recurrent family with learned initial state,
every original-object transition and every class readout. CM96 proves its
permutation limitation; this prototype makes no general MQAR claim. Freeze an
independent original AND shift/random128-update and parity16512-update cohort,
seed1, both minibatch modes, fixedcapacity16. Component discovery uses complete
CE scalar signals and finite loss-only numerical search, without supplying
parity or Boolean weights. Keep every original target, duplicate and batch tail.
Seal all six endpoints before any held-out scoring, replay all raw logits and
report all short/long outcomes. Preserve the previous producers, baseline and
resource limits; a bound or constructed parity witness alone is not success.

Task369's frozen30-cell comparison is complete. All90 train/score/replay commands
exit0 and every complete raw-logit replay matches. Addition succeeds in all six
cells with4705/4705 held-out answers; seed1 is exactly equal to the earlier dense
producer in both modes. Both modes pass2/5 primary tasks at each training seed:
addition always, AND shift only seed3, AND random only seeds1/2. Total12/30.
Key-value binding stays2.930–5.078% raw/exact and parity16 stays44.586–54.140%
exact. Preserve all failures and24-iteration settings; these are not replaced
by the separate line refinement or recurrent results. Broader generalization
remains unresolved, while three-seed addition repeatability is established.

Task372's first recurrence cohort at00d9936e completes with all18 commands
successful and all six complete raw-logit/metric replays exact. Every original
split of AND shift, AND random and parity16 reaches100% raw/exact in both modes,
including Boolean length32/64 and position shifts. Main held-out counts are
14/14,16/16 and314/314 tokens. Parity is evaluated at the original16-bit task
length; no longer-parity dataset is added here. Maximum CE across all splits is
7.59429e-7. This repairs the tested Boolean length-transfer failures and learns
held-out parity without supplied witness weights. Keep the scalar order limit,
one-seed scope and unresolved broader goal explicit. Next validate three seeds
and more original parity lengths, then address order-sensitive recall through
the separately proved general learned-operator family.

Task373 follows the approved task372 review atafa4d859. Freeze the unchanged
scalar recurrence across three training seeds on both original Boolean tasks
and original parity16/20/24:30cells, fixed data seed1,128 Boolean updates and512
parity updates. The named parity cases retain their original variable-length
distributions up to16/20/24 bits; they are separately trained tasks. Seal all
endpoints before scoring all original splits and replay every raw logit. Compare
the repeated seed1 endpoints to the first recurrence producer exactly. Keep all
failures, the scalar order restriction and the110GB/both-disk limits explicit.

Task374 proves CM97's complete two-state complex matrix/readout representation,
explicit row-sum operator bounds under the product maximum norm, and a nonzero
imaginary-state witness that predicts opposite labels on swapped input orders.
Its two transitions preserve state norm but do not commute. The19 new theorems
pass full build, direct/global axioms, generated INDEX and forbidden checks;
no new sorry or extra axiom is introduced. This is an expressivity and capacity
result, not learned MQAR accuracy. The user next suggested RoPE: task375 will
connect learned position-dependent rotations to actual bounded execution.
Existing RotarySpectrum already shows that per-row RoPE need not preserve a
flat Gram spectrum; preserve the operator bound without claiming that equality.

Task373's expanded frozen cohort at1786526f is complete: all30 endpoints pass
original success and raw/exact99 with100% raw/exact on every original split.
All90 commands exit0 and all30 raw-logit/metric replays are identical. Both
original Boolean tasks and parity16/20/24 pass in both modes at all three
training seeds on the unchanged data seed1 splits. The six repeated seed1 split
reports exactly equal the first producer, including full-logit hashes. Worst CE
across the entire cohort is4.89207e-5 on a long Boolean probe. This is five of35
selected tasks, not a full-suite result; separately trained parity lengths are
not a parity16 extrapolation claim. Preserve the scalar order limitation and
continue with the learned rotary/matrix extension for order-sensitive tasks.

Task375 extends the recurrence with CM98 learned real rotary frequencies and
actual position-dependent conjugation of the complete two-state content
matrices. All52 new theorems build: rotations preserve the actual operator
norm, the output bound is independent of sequence length, and full-class CE
queries remain feasible in the joint convex score domain. A nonconstant
swapped-prefix witness uses commuting content matrices, so its order signal
comes from rotary positions. Scalar shared conjugation cancels; the result
does not exclude other position-dependent scalar models. No Gram-flatness or
unseen-label guarantee is inferred. Complete Lean audits precede task376's
native comparison of plain and learned rotary matrices on original binding.

Task376 follows the approved CM98 proof review at56ab91ca. Implement the full
complex two-state matrix family and learned rotary conjugation in native Rust,
retaining every original object/class/record and all numerical coordinates.
Freeze a four-cell original short binding diagnostic: plain and rotary models,
Mini and forward Mini, seed1,128 updates, capacity16 and original batch32.
All endpoints seal before any scoring. Preserve every original split, failure,
complete raw-logit replay and query count. Compare matched content starts and
batches; frequency search adds work, so equal updates are not equal compute.
The two-state order witness is not a supplied training rule, convergence result
or guarantee of99% recall. The transformer and earlier frozen producers remain
unchanged, with110GB aggregate RAM and both-disk free thresholds enforced.

Task376's frozen128-update binding comparison at15b2bd3e completes with all12
commands successful and all four raw-logit/metric replays identical. All original
success/raw99 flags fail. Novel validation raw/exact is2.734%/2.148% for plain
Mini/forward and3.711%/2.930% for learned rotary. Full training is only3.027%–
4.492%, and CE exceeds the uniform initial loss. Mean mixture steps0.892–0.904
replace most existing mass on each32-example batch despite low after-update
batch CE. This suggests an update-stability issue, without proving that fixing
it suffices for recall. Preserve the failures. Task377 will prove the relevant
CE interference and feasible step bounds, then compare conservative updates for
plain, user-requested fixed-rotary and learned-rotary models on unchanged data.

CM99 now formalizes the fixed-frequency control as a convex subset of the full
rotary score family, including actual same-frequency identity bias recurrences
and feasible finite CE queries. Its concrete two-prefix counterexample proves
that one minibatch's CE can decrease while the combined mean CE increases,
although the full family contains a strict separating model. It also proves
that2/(t+2) lies in(0,1] and any accepted nonnegative step below this ceiling
preserves either the full or fixed-frequency domain for arbitrary object types.
These26 new theorems establish feasibility and a failure mechanism, not an
optimizer convergence or unseen-answer guarantee. Full Lean checks precede
native fixed/plain/learned-rotary capped-step comparisons.

After CM99 passes all checks and lands at18c7409b, the native cautious protocol
compares plain, fixed(1,0.01), and learned rotary recurrence at128 original
updates, both modes and seed1. Fixed and learned candidate starts share the
same content and initial frequencies; only learned frequencies can change.
Accepted convex steps are capped2/(zero-based t+2), with the original component
comparison unchanged. Fixed-family bias queries use the same fixed frequencies.
Record stored versus trainable coordinates separately. The plain comparison
isolates the cap versus the first producer; the learned comparison also changes
its initialization and is described accordingly. Seal all six endpoints before
scoring/replay; preserve every failure and resource limit.

Task377's cautious six-cell comparison atff00f921 completes and all18 commands
succeed; every full raw-logit replay matches. All accepted steps respect the
prospective2/(t+2) cap. Full train improves to8.105%–8.789%, CE3.459–3.484,
but novel recall remains1.953%–5.273%, and every original/raw99 flag fails.
Fixed and learned rotations have exactly identical raw split reports, mixture
weights and nonzero components in each mode. Learned frequencies stay at their
matched(1,0.01) starts despite9192 frequency trials per run. The matrices retain
many nonzero off-diagonals, so diagonal collapse does not explain this equality.
Preserve all failures and distinguish improved update stability from solving
fitting or generalization. Further work needs to diagnose component discovery,
geometry and budget rather than assume another rotary formula supplies labels.

Task378 adds a training-only diagnosis of the unchanged frequencies. Reconstruct
the frozen cautious learned-rotary trajectories exactly, requiring all128 batch
hashes/receipts and final checkpoint bytes to match, then reproduce the exact
last component criterion. Compare the complete original frequency grid with
all finer dyadic signed steps through2^-20, retaining all88 directions, actual
state responses, fixed/refitted readout CE and independently executed feasible
mixture CE. Use no held-out labels, accept no new model, and give no accuracy
credit. Existing CM98 bounds cover arbitrary finite frequencies; any future
architecture or optimizer guarantee still requires its own proof and test.

Task378's frozen diagnosis at1da2f96c exactly replays both cautious learned-rotary
trajectories:128 batch hashes/receipts each and byte-identical endpoints. It also
reproduces the final component and CE criterion exactly. All original frequency
steps worsen that final criterion, but+0.0078125 improves Mini by1.4600753e-4
and-0.0009765625 improves forward by4.3273533e-6 after the complete readout refit.
Actual feasible mixture execution confirms those gains well beyond rounding.
All88 directions per mode are retained; no held-out labels or accepted update.
Task379 will extend frequency resolution while preserving all content paths,
the original data and capped-step protocol. Existing CM98/99 proofs cover every
finite frequency and feasible convex step; no new convergence claim is made.

Task379 implements the training-only resolution repair within CM98/99: after
unchanged six full coordinate sweeps, each learned-rotary start receives16 finer
signed frequency levels1/32 through2^-20, with complete original-class readout
refits and actual feasible CE comparisons. Keep every content parameter, original
record,128-update budget, matched initialization and capped step unchanged.
Zero extra levels must reproduce both cautious training trajectories exactly.
Freeze a new two-mode producer, seal both endpoints before original scoring,
retain every failure and raw replay. No new architecture or convergence claim.

Task379's frozen repair at534a258f completes both original runs and raw replays.
All128 nonzero components now change rotary frequencies. Nevertheless novel
accuracy is3.125%/2.148%, full-training8.594%/9.570%, and both training/novel CE
worsen against cautious learned/fixed controls. All original success/raw99 flags
remain false; original batches, records and capped steps remain intact. Each
run records33704 frequency trials and73001664 component CE calls. This fixes the
measured resolution defect without solving fitting or generalization. Next
measure actual state retention and component fitting on training data before
choosing a memory-capacity or operator-geometry change.

Task380 diagnoses the frozen cautious/refined learned-rotary checkpoints on all
original training records, both modes. Trace every component/input occurrence;
measure state norms by position and length, within-length variance and exact
collisions of all weight-scaled real/imaginary coordinates. Verify every endpoint
against actual Atom.state and independent real4 execution, and reconstruct every
original-class logit against Model.logits. No held-out labels, selected features,
accepted updates or accuracy credit. Freeze and independently reproduce all four
reports before choosing a state-geometry or component-fitting intervention.

Task380's frozen training-state diagnosis at2356b0f1 completes all four runs.
Mass-weighted final state norm means14.204–14.271 remain near initial15.996;
only the explicit zero component is constant within every length. All2048
complete training-state vectors are distinct, with no exact mixed-label
collision. Independent real execution discrepancies are at most1.11e-15 and
complete logits agree exactly. This excludes simple state collapse on the
training set, not a capacity/separability or unseen-label obstruction. Next
prove and test complete readout refitting across the training stream, keeping
the trained content and all original coordinates. Treat it as a controlled
optimizer investigation, not a supplied answer rule or generalization theorem.

Task381 extends Lean with CM100 before native readout refitting. The complete
row domain varies every real/imaginary class coefficient on already learned
content; actual recurrent states stay unchanged, while full logits are linear
and indexed mean CE is convex. Prove the explicit radial row update, complete
model feasibility and full-family signed CE-query feasibility. Keep this
conditional subproblem distinct from full physical-factor convexity, and assert
no descent, separability or unseen-label guarantee. Native task382 follows only
after module/full builds, direct/global axioms, source warning, INDEX and
forbidden checks and review pass.

CM100's22 new theorems pass the full14156-job Lean build, all six touched-source
warning checks,22 direct standard-axiom reports, global36sorry/0rests/0extra audit,
native INDEX generation/check and forbidden checks. INDEX now records5059 modules
and26502 theorems, with0vacuous/0placeholder. The conditional readout step is
proved feasible and convex in its complete coefficients; no new fitting or
unseen-answer conclusion. Task382 will measure whether revisiting all learned
readouts across original minibatches improves the retained recall failures.

Task382 follows approved CM100 at16276e1c. The Rust experiment
convex_recurrent_readouts freezes four conditional refits: cautious/refined
learned rotary sources, both modes, same seed1 original minibatches,512 updates
(eight extra epochs), learning rate4 and64-step CE interpolation on[0,1]. Every
original class and complex readout coordinate receives the actual weighted
state contraction, radial feasibility rule and checked full-model update.
Retain all already learned content and verify its complete hash unchanged.
Use no supplied features or answer rule. Seal all four endpoints before any new
held-out score, preserve every failure, independently replay all raw logits,
and retain110GB aggregate RAM and both-disk guards. Training fit alone is not
an unseen-answer result.

Task382's frozen CM100 refits at6cc8d452 complete all twelve commands and four
raw-logit replays. With all learned content hashes unchanged, train accuracy
rises to68.018–71.240% and novel validation to10.547–13.672%; every original
success/raw99 flag still fails. Each run adds512 original minibatches/eight
epochs and retains all129 components/classes. All accepted readout steps are1;
actual complete logits match interpolation exactly. This supports revisiting
learned readouts as a useful fitting repair, while the train/held-out gap and
unseen-answer problem remain. Preserve every old failed result and separate
additional refit compute from source-content training and transformer budgets.

Task383 changes only CM100 readout-refit duration from512 to2048 updates, keeping
all four original sources, minibatches, learning rate4, finite queries and row
updates unchanged. Validate every original first512 deterministic receipt and
batch/count field and require the checkpoint bytes to match before continuing.
All four prefix replay records gate final scoring; no held-out label enters the
check. Freeze a new producer and retain every original split/failure/raw replay.
The training-only epoch CE trajectory justifies checking further fitting, not
claiming that greater training accuracy establishes generalization.

Task383's longer cohort atb78d4661 reproduces all four original512-update
trajectories and checkpoint bytes, then completes2048 updates per cell and all
four complete raw replays. Train accuracy reaches76.611–79.541%, novel
validation10.156–15.430%; every source-success/raw99 flag still fails. Training
CE gains diminish by epoch32 while all content stays unchanged. This does not
certify conditional optimality or a state-capacity obstruction. Investigate
radial normalization as an optimizer limitation in Lean before another native
repair; do not infer an unseen-answer guarantee from better fitting.

Task384 proves a CM101 optimizer counterexample before choosing a replacement
for radial normalization. Actual bounded recurrent content maps two different
original-object prefixes to opposite complex states with opposite next labels.
The complete CE residual gradient is checked against actual CE's first-order
inequality. At feasible boundary rows every nonnegative gradient step, followed
by radial normalization and interpolation, leaves those rows unchanged, while
different feasible rows on the same learned content have strictly smaller CE
for both records. This tests a convergence implication, not the specific native
endpoint or a claim that query-shrink weights are fixed. Complete Lean/source,
axiom, native INDEX and forbidden checks precede any new optimizer experiment.

CM101's22 theorems pass the full14160-job build, all five touched-source warning
checks and22 direct standard-axiom reports. The global audit remains36sorry,
0rests,0extra axioms. Native INDEX generation/check records5063 modules and26524
theorems, with0vacuous/0placeholder. The radial-CE counterexample is established;
the next repair must use a constrained step with a justified optimization
property, then be tested in both Mini and forward Mini on the unchanged recall
cohort. This result alone gives no additional stress-test accuracy credit.

Task385 replaces the radial rule mathematically with CM102's explicit complete
complex Euclidean projection. Prove all magnitude and zero/boundary cases,
phase-preserving distance, closest-point optimality against every feasible
complex row, original model capacity and identity on inside rows. On CM101's
actual nonconstant CE witness, a concrete positive training-probability step
must reach the better feasible rows and strictly improve both labeled records.
Full Lean/source/axiom/INDEX/forbidden checks and review precede native changes.
The next controlled Rust cohort changes only the projection rule, retaining the
four learned sources, all original coordinates/classes/batches, both training
modes,512 updates and unchanged CE settings. It must replay the old radial
trajectory exactly as a regression control before interpreting new outcomes.

CM102's25 new theorems pass the full14165-job build, all six touched-source
warning checks and25 direct standard-axiom reports. The global audit remains
36sorry/0rests/0extra; native INDEX records5068 modules and26549 theorems,
0vacuous/0placeholder. The actual four-real-coordinate closest-point property,
full model capacity and strict CE repair of CM101's example are proved. This
does not establish a native recall improvement or an unseen-answer guarantee.

Task386 follows CM102's approved proof at9d1ce0fb. Add an explicit Euclidean
projection option to the existing Rust readout-refit experiment, preserving
the radial default and immutable earlier producers. Freeze the same four
original sources,512 updates, batch stream, rate4, query1e-4 and64-step CE line
search. Test complete complex projection geometry and actual both-mode escape
from CM101, then require exact legacy trajectories and original raw reports.
Only after those controls pass interpret the prospective projected cohort,
with all four endpoints sealed before scoring, every original failure/raw
replay retained and the existing RAM/disk limits. Source/configuration and
authored findings are committed; generated evidence stays outside the worktree.

Task386's frozen producer65eb373d reproduces all four old512-update radial
trajectories, checkpoint bytes and original raw reports, then completes the
four projected refits and all eight raw replays. At the matched eight-epoch
budget, every projected training CE improves, but train raw falls to
64.648–68.750%; novel11.523–14.844% changes in both directions and every success
flag still fails. Full content, coordinates, classes and batches remain intact.
The corrected projection repairs CM101 without solving original recall.
Before blaming state capacity or accepting conditional optimality, prove and
measure a training-only complete readout optimality gap; then use that evidence
to choose further optimization or revisiting the shared learned content.

Task387 formalizes CM103 before a numerical optimality diagnostic. Derive the
exact support minimum for each complete complex row and all original classes
and components. Contract the actual indexed mean CE residuals with the learned
ordered states, then bound every feasible conditional CE improvement by the
computed gap. Explicit nonzero imaginary examples at capacity16 realize both
zero and positive gaps. This is a training-only conditional theorem; it does
not certify finite precision, global factor optimization or unseen answers.
The next measurement will stream all original training records in minibatches
through saved endpoints, retaining every coordinate and using no held-out labels.

Task387 is proved and reviewed at6383026a:29 new theorems, full14171-job build,
seven warning-free source checks, direct/global axiom checks and generated INDEX
all pass. No new sorry or extra axiom. Task388 now pins all twelve retained
readout endpoints and will measure the complete floating gap using only original
training minibatches. Check every numerical coefficient against actual CE before
the frozen run, evaluate eight fixed feasible directions without an accepted
update, and independently replay all reports. Analytic diagnostics of saved
forward-only endpoints do not count as forward-only training. No held-out labels
or new accuracy credit enter this diagnostic.

Task388 producer2e1e1aa6 measures all twelve pinned endpoints. Estimated gaps are
0.4925–0.5500 after512 updates and0.2733–0.3060 after2048. An actual fixed feasible
probe lowers training CE in every case by0.01218–0.04751; no endpoint is declared
optimal. The lowest-CE probe raises training correct counts in nine cases and
lowers them in three. No update is accepted and no held-out label is used.
Complete coefficients, classes, coordinates and records are retained; endpoint
bytes are unchanged. Original training correct counts match exactly and CE to
6.67e-16. These floating conditional bounds fix both states and mixture weights.
Task389 will first prove a joint weighted-readout capacity domain and full CE gap
that allow capacity redistribution among learned components. This remaining
conditional restriction must not be mistaken for a state-capacity obstruction,
a raw-accuracy bound or identification of unseen labels.

Task389 is proved and reviewed at33511122:62 new theorems, full14181-job build,
eleven warning-free direct sources, all direct/global axiom checks and native
INDEX pass. The shared weighted-readout capacity is convex and reconstructs
the original complete learned family exactly. Its support and actual training
CE gap include all capacity reallocations. An explicit nonzero complex model
has zero old fixed-weight gap yet strict actual CE improvement after capacity
moves to a retained ordered component. No new sorry or extra axiom is introduced.

Task390 tests this restriction with convex_joint_readouts. Freeze source,
binary, protocol and all twelve reviewed input identities before execution.
First compare both conditional gaps and fixed probes on all twelve original
training streams. Then run512 additional conditional weighted-readout updates
on the four projected-v1 endpoints, minibatch32, sampler seed10001, both modes,
query reserve1e-4, twelve line iterations and step ceiling1/16. All components,
classes, coordinates, masks and records remain. Forward training uses finite
feasible CE queries. Seal all four models before original held-out scoring;
replay all diagnostics and raw reports. Preserve learned content and the
transformer baseline,110GB RAM and both-disk limits. This tests conditional
allocation, without assuming unseen identification or a global factor optimum.

Task390 producerf5ae34ef completes the fixed comparison. All twelve enlarged
gaps exceed the old fixed-weight gaps (1.180–1.619 versus0.273–0.550), and all
have actual training-only probes lowering CE. But all four512-update minibatch
continuations fail: train raw3.71–4.05%, novel1.95–3.52%, all success flags false.
Every accepted step lowers its own minibatch CE and takes eta1/16; full-training
CE instead rises above4.0. All original raw replays match, all129 components and
learned content remain, and native capacity/reconstruction checks pass. This
recipe is finished and is not extended. Next prove how error between a training
coefficient and a minibatch or retained estimate affects the CM104 direction;
then test a minibatch estimate/update with less stochastic variation. No global
factor or unseen-answer conclusion follows from these conditional results.

Task391 proves CM105 before another minibatch repair: complete support
perturbation, corrected actual CE gaps and actual direction bounds with the
explicit2B D(A-H) coefficient-error term. A nonzero complex witness has estimated
gap12 and correction8, so strict actual descent does not require exact equality
of the estimate and training coefficient. A separate finite feasible halfway
step lowers CE on original training index0 while raising the complete mean on
two distinct opposite prefixes. Both imaginary class rows and the learned
ordered component remain. All41 new theorems compile; full14187-job build,
direct/global axiom checks and native INDEX pass with no new sorry or extra axiom.
The next native comparison will retain coefficients by original record index
and separate estimate retention from step-size reduction. It must measure actual
coefficient discrepancy and all original raw outcomes, without claiming a
stochastic-error certificate or generalization guarantee from convexity alone.

Task392 implements that prospective native comparison after reviewed CM105 at
`b87fa714`. Twelve new512-update endpoints compare instantaneous-small,
retained-large and retained-small across cautious/refined inputs and both mini
modes. Four pinned f5ae34ef instantaneous-large results complete the comparison;
the predeclared step ceilings are1/16 and1/1024. Every retained signal is keyed
by original supervised occurrence, with zero initial coverage and explicit stale
coefficients. Content changes invalidate the cache. Five native controls cover
cache reconstruction, original masks/duplicates/tails, both query modes and the
actual nonzero CM105 witnesses. All endpoints must seal before scoring; complete
training coefficient-error diagnostics are read-only and all original raw scores
must replay. This conditional repair does not establish an all-path optimum,
stochastic convergence or unseen generalization. Execute only the frozen
`convex_retained_readouts` protocol and retain evidence outside the worktree.

Task392 producer `5380ac97` completes all twelve endpoints and raw replays.
Retained-small improves all four paired starting models: train raw74.56–77.78%,
novel15.23–18.95%, test14.45–16.60%. Every original success and99% flag remains
false. Instantaneous-small and retained-large both worsen training accuracy.
All four retained-large estimates propose positive actual training directional
changes despite their small estimated gaps; their CM105 corrections6.15–9.34
exceed those gaps. Retained-small gives negative actual directions in all four
cells, actual gaps0.454–0.470, and the sufficient corrected condition in the two
cautious cells. Cache reconstruction errors are below2.95e-16, separating the
staleness issue from numerical accumulation. All129 components and learned
content remain. The optimizer repair is partial; learning shared content and
unseen generalization remain unresolved. Preserve the frozen results and extend
the theory before changing how retained estimates interact with content updates.

Task393 proves CM106's 30 new theorems before that change: actual ordered state
perturbation from learned initial/transported operators; complete signal-only,
state-only and simultaneous indexed refresh identities; separate signal/state
error in every coefficient; corrected current-content CE and direction bounds.
The historical model may differ by original record and component. Examples
include a changed learned frequency with nonzero actual state discrepancy, an
indexed three-record refresh, and strict descent with nonzero retained-signal
error. Full14195-job build, ten warning-free direct sources, all30 standard
axiom checks and the global audit pass; native INDEX has5098 modules and26711
theorems, still36 sorry with0 rests, extra axioms, vacuous or placeholder entries.
Task394 will compare a fixed-content continuation with actual learned-content
revisits, retaining historical states and using the exact old/new-product cache
update. Signals remain stale between visits, and finite content search remains
approximate. Neither CM106 nor the proposed experiment is an unseen-answer or
global-factor convergence guarantee; all original held-out outcomes still matter.

Task394 freezes `convex_content_revisit` before training: eight endpoints compare
control/revisit across all four retained-small inputs, 512 minibatches of32,
sampler20001 and unchanged readout step ceiling1/1024. Revisit cycles through
all128 learned components, testing every responding complex initial/matrix
coordinate and both frequencies against actual CE. Stored historical states
and signals refresh by the exact CM106 old/new-product identity, and every
cached state's version must replay from recorded content history. All eight
endpoints seal before scoring. Training-only diagnostics measure separate
signal/state discrepancy and the actual CM105 coefficient error. Original
success/raw/exact/CE/99% reports and complete raw replays are required; preserve
the transformer baseline and prior failures. This is approximate shared-content
learning, not a certified convex oracle in the independent factors.

Task394 producer `48571783` completes the eight fixed runs. Revisit novel raw
17.97–20.31% exceeds its matched control in all four cells, but training raw
75.15–76.76% is below every control and all four training CEs worsen; test
accuracy improves twice, ties once and declines once. Every original success
and99% flag fails. All2,048 content changes preserve feasibility and lower or
preserve their current minibatch CE. All16,384 cached original-record states
replay bit for bit, and independent product sums agree within3.47e-16. All
eight complete raw replays agree. Actual estimated readout directions remain
negative in every cell, although the CM105/106 sufficient error bounds are
inconclusive. These checks validate the changing-state estimator, not this
content optimizer or generalization. Do not extend this finite coordinate
recipe merely because its visited-batch CE decreases; shared-content learning
and the requested99% original held-out target remain unresolved.

Task395 proves CM107's geometry extension before another native model change.
The complete Hilbert-coordinate operators/readouts reconstruct all complex
linear maps. Rational mixing R=((3/5,-4/5),(4/5,3/5)) preserves every complex
state's Euclidean norm but exceeds the old maximum-norm budget. The actual
CM102 row projection gives Q=((2/5,-3/5),(3/5,2/5)), whose repeated squared
norm is exactly(13/25)^length times the initial value. Conversely, an old
admissible broadcast matrix exceeds the Hilbert budget, so changing the norm
cannot silently replace the old family. The enlarged joint domain retains
every old rotary function and adds every bounded Hilbert recurrence, with the
same output budget, ordinary CE convexity and actual feasible finite queries.
All34 new theorems, full14201-job build, seven direct sources, standard-axiom
checks and native INDEX pass;36 existing sorry remain with0 rests or extra
axioms. This proves a concrete geometric restriction, not that it caused the
native failure or that the added family learns unseen answers. The next native
comparison needs a checked full-matrix Hilbert feasibility rule, preserved old
models, learned content and the unchanged original stress labels.

Task396 proves CM108's executable full-matrix rule. All four complex entries
form the actual column Gram coefficients; its closed larger-energy value is
both an upper bound and attained by an explicitly constructed nonzero state.
The resulting square root equals the actual Hilbert operator norm, including
zero/diagonal cases. Uniform normalization by max(1,norm) preserves every unit
matrix and gives a feasible full transition; no singular direction is removed
and no Frobenius closest-point or optimizer claim is made. All28 new theorems,
full14206-job build, six warning-free direct sources, direct/global axiom checks
and generated native INDEX pass. INDEX has5109modules26773theorems,36existing
sorry and0rests/extra/vacuous/placeholder. Native comparison can now enforce
the added geometry without silently shrinking its isometries or replacing the
old family. Original held-out99% accuracy remains unresolved and must be tested.

Task397 freezes `convex_hilbert_content` before native training: all four394
control endpoints, selected on training evidence, retain their complete old
parent at7/8 mass and add one full learned plain complex block at mass2.
Old max-norm versus CM108 Hilbert geometry uses identical initial predictions,
the same512 original minibatches of32 and both Mini/forward-only modes. All
1204 real coordinates of the new block remain trainable. Complete feasible
coordinate queries feed per-occurrence retained estimates; all parameters
update simultaneously, replacing the earlier greedy coordinate acceptance.
The retained parent stays learned but held during this new-block phase. This
is a fixed empirical pilot, not jointly convex factor training or a proof of
stochastic convergence. Every cached estimate must replay from its original
record and actual historical parameters. All eight endpoints must seal before
original raw/exact/CE/success/99 scoring and raw replay. Preserve all earlier
failures and the transformer, with native tooling and110GB/both-disk guards.

Task397 producer `c620019a` completes all eight frozen runs. All improve novel
and test raw accuracy over their starting parents; novel is21.68–25.39% and
test22.27–25.39%, with all original success/99% flags false. Hilbert training
raw80.22–82.32% exceeds every matched old arm, but held-out raw wins only two
of four pairs on each split and CE is worse in all four pairs on every split.
All16384 cached estimates and all original raw logits replay; retained means
agree within3.34e-16, and aggregate RSS stays below15.090GB. Learned Hilbert
states still contract: allowing isometries does not force the optimizer to
find them. Do not infer a generalization solution from norm feasibility or
positive numerical estimate alignment. Before expanding the recipe, measure
whether the new block's gains require input-dependent states beyond a learned
class bias. Preserve the prospective eight-cell results and both geometries.

Task398 declares a post-hoc dependence diagnostic over all eight397 endpoints,
with no retraining or selection: intact, zero new state, training-mean constant
new state, and new-block-only at its original mass. The complete complex mean
uses original supervised training prefixes without reading target labels.
All original held-out examples, labels and metrics remain. Intact raw logits
must match saved scores exactly, every ablation must satisfy its full numerical
geometry, and the complete report must repeat independently. These ablations
test dependence on input information beyond a class bias; they are not newly
trained endpoints or stress-suite successes. Keep sources and pins frozen,
all prior results intact, native checks and110GB/both-disk guards active.

Task398 producer `1978c688` completes all32 declared ablations with zero updates.
All eight intact raw reports match c620019a exactly, and a fresh review repeats
all reports byte for byte. Mean-state ablation lowers novel accuracy by1.95–6.84
points and test by4.10–6.84 points in every cell, with worse held-out CE. Current
input-dependent content contributes beyond its average class bias. The new
block alone remains at8.40–13.67% novel and9.96–13.87% test; every original
success/99% flag fails. This does not prove that a separately optimized bias
cannot match the gains, that query-specific binding was learned, or that wider
states are necessary. Three native controls and two immutable-input fault
refusals pass. The next learning step must strengthen shared retrieval content;
do not treat normalization, positive estimate alignment or this ablation as an
unseen-answer guarantee. Original endpoints and transformer remain unchanged.

Task399 proves CM109's full-class CE reuse before the larger native content
comparison. A complete one-class logit change updates the original partition
by its exact changed summand. Its positive factor gives the exact ordinary CE
query using the actual full-vocabulary softmax share, and subtraction gives
the two-query difference even when projected parameter queries are asymmetric.
This changes no labels or loss and supplies no analytic residual to forward
training. Native output-row queries can reuse the partition after comparison
with complete CE evaluations; real identities are not rounding certificates.
The intended follow-up grows learned mixtures at the existing CM107 budget
and compares them with equal-update continuation of one learned block.
All seven new theorems pass the full14207-job build, warning-free direct
checks, standard-axiom audit and native INDEX regeneration. INDEX now has5110
modules26780theorems,36existing sorry and0rests/extra/vacuous/placeholder.

Task400 compares continued learning of the existing plain block with adding
complete learned blocks under the same CM107 budget16. All eight c620019a
endpoints enter both arms, yielding16 prospective runs. Eight stages of512
original minibatches32 use identical stage samplers, fresh occurrence caches
and complete1204-coordinate updates. Growth scales all prior weights by7/8
and adds mass2, preserving every stored component. CM109 full-class CE reuse
is checked against actual finite CE queries during training; forward-only
learning never uses analytic CE residuals. Every phase history and frozen base
must reconstruct. Seal all128 stages before original scoring at0,1,2,4,8,
retain all outcomes and replay all80 reports. No new task program, label subset,
RoPE, full-batch training or success assumption is introduced. Freeze native
source/binary/protocol/inputs before training and enforce110GB/both-disk limits.

Task401 independently examines a structural limit while the fixed400 cohort
runs. CM110 proves that all complete plain two-complex-state recurrences
satisfy an eight-word squared-commutator identity. Every finite mixture retains
it, regardless of signed weights, learned entries or normalization. A held
rotary parent's contribution is explicitly its scaled obstruction, not zero
by assumption. A complete three-coordinate contraction has actual obstruction
-1, with unit initial state, bounded outputs and nonincreasing Euclidean
energy. No finite plain two-state mixture equals it; uniform score error is
at least1/8. This distinguishes state width from mixture size, without claiming
that the original MQAR labels violate the identity or that width alone learns
unseen answers. Do not change the ongoing400 protocol or filter its outcomes.
CM110 validation passes the full14210-job build, warning-free direct checks
and all23 standard-axiom checks. Generated INDEX has5113modules26803theorems,
with36existing sorry and0rests/extra/vacuous/placeholder. The nonempty singleton
zero-component example satisfies the uniform-error hypothesis at epsilon1.

Task400 producer `f5d317a6` completes all16 runs,128 stages and65,536 original
minibatch updates. At the fixed final stage8, growth reaches37.70–42.77% novel
raw and33.98–44.53% test; all eight cells beat equal-update continuation on both
held-out raw metrics and every split CE. Old-geometry growth averages41.26%
novel/42.43% test, versus23.19%/21.73% continuation; Hilbert growth averages
38.67%/36.13%, versus21.39%/23.24%. Training raw is nearly unchanged or lower,
so this is not just increased training fit. All80 declared checkpoint success
and99% flags fail. The largest logged novel observation,43.75% at stage4, is
retained without selecting it as an endpoint. All original classes, examples
and the transformer remain unchanged; only one original task and seed are covered.

All262,144 cached occurrences replay at their actual historical versions;
maximum mean-recomputation error is1.458e-15. The56,492,032 compact/full CE
checks agree within1.777e-15, actual/cache logits agree exactly, norm excess
is2.221e-16, and aggregate RSS peaks at34.73GB. No rising minibatch loss is
filtered. Fresh tests from the frozen source, all80 raw report replays,128
history/current diagnostic files and seven deliberate corruption refusals pass.
The old-geometry continuation caches have sizeable current-estimate errors,
so bookkeeping consistency does not imply optimizer convergence. Growth also
redistributes old mass; its gain does not isolate added content from calibration.
Task402 will classify remaining errors using original examples: other stored
value versus outside both stored values, two-value margins, selected pair and
query position. Those task roles enter read-only diagnosis, never training.
CM110 supplies a separate width obstruction, not an explanation of the MQAR
errors or a99% guarantee. Generalization remains unresolved.

Task402 now pins all eight initial and sixteen final inputs before evaluation.
It preserves original full metrics and independently reproduces their logits,
while partitioning errors into the other stored value and outside both values.
Distinct-value choice, complete probability mass, CE decomposition, both selected
pairs and all query positions are auxiliary descriptive reports. Repeated-value
examples retain their original weight. No roles enter the predictor or training,
and no new success flag is introduced. Six native controls and fresh complete
reports plus changed-input refusals precede interpretation. Enforce110GB aggregate
RAM and both disks; freeze all source, binaries, protocol and pins first.

Task402 producer `3a509853` completes and replays all 24 diagnostic reports
without changing any complete original score. Across the four old-geometry
growth cells, novel/test stored-value prediction is 80.13%/78.71%, while
two-distinct-value choice is 49.90%/51.79% versus 82.46% training. Hilbert
growth gives 70.61%/67.63% stored-value prediction and 52.91%/48.98% choice.
Old growth's CE reduction is entirely from better stored-value mass, with
conditional stored-value CE slightly worse. A first-pair preference remains:
novel choice is 67.63% for the first pair versus 33.27% for the second in old
growth. Even with no filler, stored-value prediction is 100% but choice is
54.01%. At the longest original query prefix, stored-value prediction drops
to 69.74%. Query binding and longer-prefix retention are distinct problems.
All repeated values, original classes and masks remain; validation aliases
novel and is not an extra dataset. Six frozen-source native controls and four
changed-input refusals pass; fresh reports are identical. This is one original
task/seed, zero updates and no new success. Prioritize shared learned matching
across unseen combinations, while retaining CM110's width result as a separate
expressivity fact rather than evidence that width alone solves this failure.

### 2026-11-17 GPT stress-test goal for the fully trainable next-object model

Use the November 16 constant-object interface across all 165 cases in
`stress_tests_for_gpt.md`. At every prediction the model reads an immutable
prefix and returns one object from the same vocabulary. Fit a fresh instance
of the shared architecture per case. Ordinary full-vocabulary next-object CE
trains the representation, sequence interactions and output mapping using the
original training records. Object IDs provide addresses only. For the learned
complex model, state the domain on which ordinary CE is jointly convex and
which coordinates the native optimizer updates. CM95, for example, proves
convexity over finite mixtures of score functions. Training the factors of a
component requires its own analysis. Use flat-spectrum results to bound the
executed operator and check which unseen predictions the observations identify
under the stated assumptions (CC6–7, CM77, CM87, CM91).

Run both minibatch and forward-only minibatch in Rust/Kernite. Freeze each
protocol and seal its endpoints before scoring, then independently repeat and
replay. Retain every original split, metric and success flag and report the
separate 99% raw-object and exact-sequence targets wherever defined. The
98/165 historical union spans separate seed-1 cohorts. Track the new
architecture's results case by case. Include negative controls and refusals
in the denominator. On `random-lm-control` the deterministic exact-score
ceiling on the primary split is 1/64. The native check and its assumptions
are in `experiments/convex_stress_remaining_synthetic_data/README.md`.

The first milestone is `gpt-probe-key-value-binding`. Board task 403 covers
the learned retrieval margin. Task 404 tests whether training learns the
matching and readout on the original 2,048 training examples and generalizes
to the held-out queries in both modes. Use full-vocabulary accuracy for the
pass decision. Report choice between the two distinct stored values as the
diagnostic from task 402.

Task 403 establishes the conditional retrieval margin in flat-spectrum CM111.
One learned complex embedding supplies queries and keys, and a second supplies
values and all original-class readouts. A learned relative-position potential
connects every pair of past positions. The normalized path weights give a
length-independent score bound. Actual gaps in the two learned softmax scores
and separation in the value embedding imply a margin against every competing
class. The two-object complex example satisfies these assumptions. Whether
ordinary next-object training discovers them on the original probe is the
question for task 404.

Task 404 prospectively fits a single CM111 head on the original binding probe.
Use the original data seed 1, training seeds 0, 1 and 2, 100 output classes
and 1,024 minibatches of 32. Every offset starts at zero. Learn all complex
key and value/readout coordinates, the relative positions and the key scale.
Mini uses the analytic CE gradient. Forward-only Mini uses paired finite
queries of complete minibatch CE on feasible models and cycles through both
entire embedding tables. Freeze both builds and seal all endpoints before
scoring. Compare all original splits, the numerical CM111 margin and the
queries whose supervised or stored key/value pair was absent during fitting.
The two definitions of unseen pairs apply only to evaluation.

Task 404 producer `339ba587` learned the original binding probe in both
minibatch modes at seeds 0, 1 and 2. Every endpoint scored 512/512 on novel
validation and 512/512 on test, including 65/65 test queries with a previously
unsupervised key/value pair and 5/5 with a pair absent from all training
memories. All initializers and trained raw outputs replayed byte for byte
across independent builds. Forward-only training learned offsets that give
the correct visible link from the original two key positions, while larger
potentials at inaccessible negative offsets remain. Evaluate other positions,
overwrite retrieval and two-hop lookup on their original data. This is one
measured case for the current protocol. The historical 98-case union already
includes this case.

Task 406 applies the same model and fixed 1,024-update protocol to
`suite-mqar` and `suite-lookup`. Both original tasks use two supervised
queries, batches of 32 and data seed 1. Size the input context from the
published maximum length 96. Train only on original length-24 records and
score every original length-24/48/96 split after both independent cohorts are
sealed. Task 407 separately covers eight-query Basis recall and overwritten keys.

Task 406 producer `3a69f86b` reached the original multi-query recall target
at all seeds in both modes. Forward-only Mini achieved 100% raw and exact on
all 11 splits, including the length-96 probe. Mini scored 31/32 exact on
novel validation and 63/64, 58/64 and 62/64 on the length-24/48/96 tests.
Two-hop lookup missed its original target in both modes. At length 24 the
test exact counts were 10–12/64 in Mini and 14–15/64 in forward-only Mini.
Across the three originals evaluated with this architecture, one passes the
99% raw and exact goal at all model seeds in both modes. Keep the lookup
failure and the mode difference on recall visible in subsequent comparisons.

The retained multi-query recall logits identify one missing supervised output.
Object 132 occurs 33 times in training inputs. All three Mini endpoints fail
every held-out query for this object and succeed on every other answer.
All three forward-only endpoints also succeed on object 132. This observation
gives a training-exposure test for the next comparison.

Task 408 and flat-spectrum CM112 identify a separate obstacle to the
165-case goal. On any two-object prefix the current single retrieval head
discards all query dependence through its singleton softmaxes. The original
modular-addition validation score is bounded by 97/4,705 for every set of
weights. A successor must retain a trainable interaction with the final
object at this prefix length and must also be tested on two-hop composition.

Task 409 and flat-spectrum CM113 define a shared successor with two learned
content-dependent reads, a learned predecessor bias and a query rotation
that survives a one-object memory. Its complete bounded score has an
admissible witness for all 9,409 original modular sums and a separate
repeated-key witness where the first and second reads favor different
objects. In the same family two admissible states agree on complete training
scores and predict opposite objects at an unobserved query.

Freeze the four-case native comparison in
[composed_retrieval](experiments/composed_retrieval/README.md). Use both
training modes, three seeds and 1,024 updates per cell. Seal all 48 fits from
two independent builds before opening held-out scores. Compare with the
retained CM111 cohort, including every missed unseen answer and every length
probe. Keep the full 165-case goal and its negative controls in view.

Task 410 producer `0f0aadbd` completed the four frozen original cases.
All 48 independently trained endpoints sealed before scoring, and all 672
initial and final raw files passed independent decoding. At all three seeds
in both modes, binding and multi-query recall reached their original target,
and binding passed the 99% raw and exact requirement everywhere. The other
161 cases have no measured result for this model. The separate historical
98/165 union retains its original scope.

Standard minibatch learned every one of the 4,705 held-out modular sums at
every seed. Forward-only minibatch learned 79, 64 and 82. It visited each
value and rotation coordinate only 21 or 22 times. On two-hop lookup all
six trained models assigned zero weight to the second read at the endpoint.
Board tasks 411 and 412 distinguish two follow-up questions. Test forward-only loss
queries with matched coordinate visits, and determine what the observed CE
gradient does to the two-hop gate and routing parameters. Freeze each new
comparison and retain the four original-case controls.

Task 411 compares every-coordinate forward-only loss queries against standard
minibatch on the same four original cases. Freeze the comparison in
[the matched-visit protocol](experiments/composed_retrieval/dense/protocol.json).
Each of the 48 fits uses the original seed, batch order and fixed 1,024-update
endpoint. Record numerical query counts and measured resource costs. Check the
standard minibatch runs against the previous weights and optimizer evidence.

Producer `0c8436c1` recovered all 4,705 held-out sums with forward-only
training at all three seeds. The standard minibatch control reproduced
every earlier endpoint and optimizer trace. Across both learned modes all
46,659 scored positions gave the same next object. With this shared dense
protocol, binding and addition pass every 99% raw and exact criterion.
Multi-query recall also passes its original target. The 161 remaining cases
are unmeasured. Two-hop lookup still underperforms in both modes.

Task 412 derives the ordinary training signal for the learned depth gate in
flat-spectrum CM114. With a constant relative potential all queries route
to the uniform distribution over visible destinations. The first and second
unrotated read coincide, and the CE derivative along any joint content and
depth curve is zero. For general weights the actual depth derivative is
the difference between the full predictive mean direction and the observed
target's direction. Replay the original sparse-protocol training trajectories
to measure when the symmetry breaks and what moves the gate to its boundary.

Task 413 applies the matched-coordinate CM113 protocol to all 54 original
two-operand arithmetic and powered-division cases. Recover the original
sources and verify both seed-1 split fingerprints before each fit. Preserve
all six unavailable cases and the three constant controls. Seal both builds
before evaluating any held-out row, then compare their raw outputs and the
overlapping addition endpoints with the preceding four-case experiment.

Training-only replay of the six original sparse lookup fits reproduced
every endpoint and training trace. The two reads became distinguishable on
minibatch two in every fit. The first minibatch starting at depth zero was
47–52, and 5,852 of 5,853 subsequent boundary minibatches favored keeping
it there. At each endpoint the training CE slope was positive, +0.283 to
+0.370, and forcing the gate to one reduced correct objects from 119–242
to 52–81 out of 256. CM114 explains this unilateral depth optimum with
the learned routing and readout held fixed.

Matched-visit CM113 now has 51 measured original cases. Six satisfy
their original criterion and five satisfy all raw and exact 99% targets
at all three seeds in both training modes. The arithmetic cohort includes
48 original datasets with 4 passes. Addition and subtraction scored
4,705/4,705 at every seed. Division at modulus 97 and one constant-output
control also crossed 99%. Six fixed-random source datasets are unavailable
and 108 cases are outside the measured cohorts. Addition is counted once
after checking every overlapping model, receipt and raw file.

Task 414 keeps the four original controls, both dense minibatch modes,
three seeds and the 1,024-update budget. Change only the depth coordinate
and its initialization. Train log odds initialized to log 9, giving the
second read 90% of the physical depth mixture. CM115 derives the actual
raw CE gradient and the relation between raw displacements and physical
score contrasts. Compare trained gates and full training slopes along
with every original and strict held-out criterion.

Frozen log-odds depth produced the first full two-hop result here. All
six fits scored every original training, novel validation and held-out
test object and sequence correctly, through length 96. At the trained
weights, disabling the second read reduced training accuracy to 0/256
while depth one gave 256/256. Binding and modular addition remained
perfect. Multi-query recall reached 31/32, 32/32 and 30/32 on novel
validation at the three seeds and lost its original all-seed pass. This
protocol has three strict passes in four measured cases. Its coverage is
reported separately from the 51-case physical-depth experiment.

Task 415 extends the frozen log-odds training rule to the same original
54-case arithmetic inventory. CM113 proves that either read selects the
sole memory object in these prefix pairs. Independently refit every
available case in both modes and compare actual raw decisions. Verify the
overlap against the log-odds four-case evidence before merging its ledger.

The complete log-odds comparison passed six original and six strict cases
among 51 measured. Binding and two-hop lookup joined four arithmetic
passes. All 48 available arithmetic outcomes and per-seed correctness
counts were identical to the physical-depth reference. Only one raw
decision changed across 3,262,182 final arithmetic predictions, a wrong
validation answer in both models. Six original sources remain unavailable
and 108 cases fall outside this cohort. The ordinary and forward-only
fits matched every strict case decision.

Task 416 tests a bounded trainable offset on every original output class.
On division, multiplication and affine product at modulus 97, all observed
training mistakes in the physical-depth reference shared the same target
at all three seeds. CM116 gives the full-logit bound, the conditional CE
derivative, and exact fixed-score inequalities for repairing a class with
one offset. Jointly train all representation, sequence and output
coordinates from zero-initialized offsets. The two independent cohorts
each contain all 51 cases with one fit per case, mode and seed. Compare
every held-out outcome and retain all six arithmetic source refusals.

Task 417 freezes the same output-offset model on all nine synthetic cases
that score the visible input prefixes. The seven additional cases cover
brackets, alternating blocks and three past-counting formula depths. Before
combining coverage, repeat multi-query recall and two-hop lookup and compare
both builds' complete models, training receipts and raw splits to task 416.
Use the independently retained native data fingerprints and the original
plan's batch sizes and accuracy criteria.

The CM116 joint-offset cohort passed nine original and eight strict cases
in 51 measured, with matching accuracy counts at every split and seed in
both modes. The constant controls at moduli 17, 97 and 193 became exact.
Division modulo 193 gained a strict pass while division modulo 97 missed
at seed 1. Multi-query recall regained the original target. Two-hop lookup
and binding remained exact. On the frozen previous scores the three
modulus-97 operations with errors concentrated at zero all required
incompatible single-class displacements. Producer `e60c0c43` sealed 612
fits before scoring and replayed 2,928 raw files across two native builds.

The synthetic-prefix extension verified 58 distinct original cases under
CM116. Nine met their original criterion and eight met the all-split 99%
criterion. None of the seven additional cases crossed its all-seed
threshold. The depth-1 counting formula fit every training position but
only 18, 38 and 36 of 64 exact sequences on the longest test. The two
retrieval overlaps reproduced all 528 original raw files, model weights
and optimizer receipts across both builds. The source-unavailable count
remains six, with 101 cases outside these two cohorts.

Task 418 freezes all six original copy, reverse and sort cases and repeats
two-hop lookup under CM116. The closed-loop scorer starts with the original
prompt, appends each model-written object and stops at EOS or the original
input-derived limit. Compare generated and teacher-forced accuracy on every
split after both builds seal. Verify the repeated lookup before adding the
six cases to the original 165-case ledger.

Task 418 measured one further original pass. Distinct-symbol copy reached
32/32 exact novel validation and 64/64 ordinary test sequences at every
seed and in both training modes. At each longer test length it produced
zero of 64 sequences exactly. On length 64 its teacher-forced correct
object counts were 0, 1 and 1 of 4,160. The other five Rewrite cases missed
the original threshold. The matched CM116 total is 10 original and eight
strict passes in 64 measured cases, with six unavailable and 95 outside.

Task 419 tests whether CM117's learned object magnitudes can address the
observed product errors. Its original-training-only intervention reproduced
all raw logits of 288 arithmetic models. At modulus 97, setting the selected
input radius to zero made multiplication exact on training pairs at two
seeds and repaired 45 of 55 errors at the third. Most fitted models still
had errors under all five tested radii. Compare a fresh fully trainable
model starting with all radii at one against the frozen CM116 cohort.

Task 420 freezes that joint fit on all 64 available cases from the three
CM116 cohorts. The initial object radii reproduce every old logit. Both
native optimizers update all old coordinates and each new radius on the
same original minibatches. Seal both independent 384-endpoint builds
before scoring. Preserve original failure, control and source-unavailable
rows when comparing the two architectures.

Task 421 applies the same trained-radius protocol to the 23 remaining
original synthetic generation cases, repeating copy as a control. Verify
all 345 splits, including novel Boolean probes, exact generation with
scratchpads and the one-prompt random language control. Seal both
144-endpoint builds before scoring. Combine the two CM117 studies only
after the repeated models, receipts, metrics and raw files agree.

Task 422 bounds the executed distance-decayed link and proves which
offsets short prefixes can observe. The training-only grid reproduced all
42 original Rewrite-cohort fits. Distinct-symbol copy remained exact at
every distance setting, with 192 residuals unobserved. Separate slope
and residual parameters can compensate at every length. Test a smaller
relative function whose near-distance observations constrain its long
tail, retaining ordinary next-object supervision on the original records.
