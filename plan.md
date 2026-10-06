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

Next construct the simultaneous raw adjacency/table-gating/last-write encoder
and depth-prefix recurrence.
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
