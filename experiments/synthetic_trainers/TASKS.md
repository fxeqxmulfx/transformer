# Task selection from RASP and C-RASP

The proposed initial pair is composed associative lookup and causal prefix
computation. Together they test content selection, composition, aggregation,
comparisons, and sequence order. MQAR remains a reference measurement. This
was the initial selection; the complete suite now includes 17 tasks and 37
default comparison variants. The generators and common training
protocol are described in [README.md](README.md). Difficulty calibration and
architecture experiments are pending.
The separate [memorization and double-descent protocol](STUDIES.md) adds a random
sequence control and measures fitting, noise memorization, and delayed algorithmic
transfer across these tasks. The random control is not a new RASP mechanism task.

| Trainer | Initial tasks | Independent difficulty controls |
| --- | --- | --- |
| Composed lookup | Several queries asking for the endpoint of a lookup chain | Chain length, table size, distractors, query count, record order, query distance |
| Prefix computation | Dyck-1 prefix status and neutral-letter languages `E_k` | Count scale, block count, block sizes, neutral gaps, input length |

## Composed lookup

An example contains a table of uniquely keyed directed links, followed by
queries. For a fixed chain length `h`, a query `q` asks for `f^h(q)`, where
`f` is the table's lookup function. For example, `a -> b`, `b -> c`, and
`c -> d` make the answer to a three-step query from `a` equal to `d`.
All required links and distractors occur before the queries. Selected paths
have no early termination or repeated vertices.

Start with `h = 1, 2, 4`, fixed per experiment, and several independent
queries. Serialize records with explicit key/value roles and preserve the
same identity when a value becomes the next lookup key. The current MQAR
generator uses disjoint key and value vocabularies, so it cannot supply this
composition unchanged. Use ordinary tokens and the existing mini GPT output
head; supervise answer positions without putting answers or intermediate
chain states into the input.

Randomly relabel identities and shuffle records in every example. Include
counterfactual pairs that permute values between records, preserving token
frequencies while changing the queried endpoint. Vary chain length separately
from table size and distance; increasing the number of queries must not make
answers available through earlier supplied answers.

RASP motivates this task through repeated content-based `select` and
`aggregate` operations. The compilation rules give a schedule for the chosen
program, rather than a minimum layer count for every possible solution.
Key/value binding in the serialized input also has a cost. Architectural
hypotheses to test include attention selectivity, the interleaving of
attention and feed-forward operations, and extra or reused computation steps.
Count all reused steps in the compute and timing comparison.

Sources: [RASP compilation](../../src/Transformer/RASP/Compilation.lean),
arXiv:2106.06981v2, Sections 3.1 and 4; the existing
[MQAR data contract](../convex_mqar/src/convex_mqar/data.py).

## Prefix computation

Use two separately scored modes within the second trainer. At each input
position, predict a label determined by that prefix. Labels are supervision,
not additional input tokens, so this remains a causal task.

**Dyck-1 prefix status.** Predict whether the prefix is balanced and valid,
valid but still open, or already invalid. Count opening and closing
parentheses and check that no earlier prefix had more closes than opens.
For example, `(())()` is valid at the end, while `())(()` is invalid despite
having the same length, symbol counts, first symbol, and last symbol.
Include neutral tokens that leave the logical state unchanged. Vary maximum
balance and the location of the first violation separately from input length.
This supplies a concrete test of aggregation and a second computation over
earlier comparison results. The formalized Dyck predicate belongs to C-RASP's
past-counting fragment at depth two; this is an upper bound.

**Neutral-letter block recognition.** After deleting every `_`, test whether
the prefix contains exactly `k` nonempty alternating blocks of `a` and `b`,
starting with `a`. This is the formalized language `E_k`. For `k = 3`,
`a _ a b _ b a` is accepted and `a _ b a _ b a` is rejected. These examples
have identical symbol counts, length, endpoints, and neutral-token positions.
Start with `k = 1, 2, 3, 4`. Vary block sizes and neutral gaps independently,
including long gaps and uneven blocks. Neutrals prevent an adjacent-symbol
transition count from being sufficient. Score full prefix membership,
including negative examples; this differs from the paper's next-token task
restricted to prefixes of positive strings.

The proved `E_k` hierarchy gives this mode stronger theoretical grounding
than merely making a RASP expression longer. In the specified rounded model,
`E_k` is recognized at depth `k` and not at smaller depths. In each formalized
positional encoding family, the positive construction chooses encoding
parameters; the lower bound covers every parameter choice satisfying its
angle assumptions. The theorem concerns exact recognition on all lengths.
Its rounding rules and rational phase assumptions differ from mini GPT's
implementation, so
the practical depth requirement and learning speed remain experimental.

Sources: [Dyck example](../../src/Transformer/CRASP/Basic.lean),
arXiv:2506.16055v3, Example 2.3 and Appendix A.2;
[`E_k` definition](../../src/Transformer/CRASP/PositionalDepth.lean) and
[encoding hierarchy](../../src/Transformer/CRASP/EncodingHierarchy.lean),
Appendix F and Section 4.5. The ordinary hierarchy is in
[Transformers.lean](../../src/Transformer/CRASP/Transformers.lean).

## Evaluation and first experiments

Keep the task fixed while extending length. For lookup, enlarge the table or
query distance at fixed `h`; report larger `h` as a separate composition
test. For prefix computation, extend blocks and neutral gaps at fixed `k`;
report larger `k` separately. Proposed length checks are twice and four times
the training maximum, within the model's configured context capacity.

Report answer accuracy and correctness of every answer in an example for
lookup. For prefix tasks, report class-balanced accuracy, correctness at
state changes, and correctness of the complete label sequence. Long runs of
unchanged labels must not dominate the result. Compare each mode and
difficulty group separately before reporting any aggregate score.

The RASP-L diversity condition motivates matched examples and independent
difficulty controls: training should expose failures of simple frequency,
endpoint, fixed-position, and fixed-distance shortcuts. This is an empirical
design principle from a conjecture, not a proved learning guarantee.
Source: arXiv:2310.16028v1, Section 3, recorded in
[Conjecture.lean](../../src/Transformer/RASPL/Conjecture.lean).

First calibrate the baseline on these modes, using validation data to choose
informative difficulty groups and quality targets. Then test architecture
changes with the protocol in [PLAN.md](PLAN.md). A more selective attention
mechanism could help lookup while making distributed aggregation harder;
measure both rather than assuming that a gain transfers. Evaluate additional
depth or reused layers at comparable budgets. These are experimental
hypotheses, not established outcomes. Record time to each predefined target,
and explicitly mark targets that are not reached.

Sorting, reversal, and scratchpad output are now implemented alongside this
first pair. In the original RASP sorting experiment,
a bounded alphabet admitted a bucket-counting shortcut, illustrating why
program length alone is an unreliable difficulty measure. Scratchpads change
the available computation and therefore need a separate comparison budget.

Source: arXiv:2106.06981v2, Section 5, in the local
[experiment section](../../papers/arXiv-2106.06981v2/05_experiments.tex).

## Complete RASP-family coverage

| Tasks added | Mechanism and comparison controls |
| --- | --- |
| Histogram, double-histogram | Exact symbol counts; frequency classes of distinct symbols; histogram BOS control |
| Mode, Most-Freq | Unique argmax; distinct symbols ranked by frequency with first-occurrence ties; sorted and itemized count scratchpads |
| Copy, reverse, sort | Ordered full answers, repeated-symbol ambiguity, and unique-symbol controls |
| Dyck-2/3 | Match bracket types as well as balance; neutral gaps; incorrect closer pairs with unchanged counts |
| Count | Inclusive interval generation with atomic integer tokens and EOS |
| Addition | Decimal digits; forward/reverse outputs; index hints; standard/balanced sampling; independently prescribed and hard carry chains |
| Parity | Direct XOR; full running-state scratchpad; paper-style indexed one-bit scratchpad |
| Boolean-AND | Transfer of the decisive zero to unseen positions at unchanged length; random-position control |
| Sampled C-RASP formulas | Fixed programs with varied inclusive count nesting, comparisons, integer arithmetic, and Boolean composition |

Sources: [RASP Section 5](../../papers/arXiv-2106.06981v2/05_experiments.tex),
[RASP-L experimental task definitions](../../papers/arXiv-2310.16028v1/appendix.tex),
[RASP-L causal interpretation](../../papers/arXiv-2310.16028v1/rasp.tex),
[addition and scratchpads, Section 5](../../papers/arXiv-2310.16028v1/scratchpads.tex),
and [C-RASP syntax, Section 2.3](../../papers/arXiv-2506.16055v3/neurips2025.tex).

These are mechanism benchmarks derived from the papers rather than replicas
of every published experimental setting. Global RASP outputs follow the full
prompt in the causal decoder. Count/histogram/count-scratchpad numbers are atomic
tokens; the paper's decimal mode scratchpad is therefore a different encoding.
Addition keeps one padding digit on both operands and the answer. The balanced
sampler constructs exactly one carry chain of the sampled length; digits outside
that chain cannot propagate another carry. Its distribution differs from uniform
sampling conditional on a carry profile. AND uses disjoint early/late regions,
with the final quarter reserved for its position-transfer test.

The temporal sampler selects a nonconstant formula on a finite witness bank,
not uniformly from the entire logic. Programs are fixed per run, rather than
supplied as instructions in each example. Count nesting is a syntactic control
and can exceed the minimum equivalent formula depth. Other formula seeds allow
evaluation across programs while keeping each program fixed during length tests.

Free rollout supplies the external iteration discussed in RASP-L. Quality targets
use generated answers rather than answers evaluated with correct preceding outputs.
Mode/parity final-answer accuracy is separate from full scratchpad exactness.
Compare both quality and the cost of the complete generated trace; neither a
scratchpad nor a longer RASP expression guarantees a learning advantage.
