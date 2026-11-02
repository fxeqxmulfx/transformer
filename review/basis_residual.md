# Basis comparison: same-assistant review

Task160 is the user's requested broader check of the conditional-gain candidate.
Task158 recorded three earlier arithmetic crossings and one tie; its frozen
all-four-earlier promotion rule failed, so production GPT remains `df1e9f7`.
Basis results must not be inferred from those arithmetic results.

The24 recipes are the original small/easy and large/hard Basis settings, each
with canonical/adaptive GPT and model seeds0/1. All six full problem/settings
have exactly the original split fingerprints, including every training row,
validation/test row, length distribution and original label. The generators and
their oracles are unchanged. The optimizer deviation is explicit: only
Schedule-Free AdamW, with the original Basis rates/batches/warmup. Compilation
uses CUDA fullgraph max-autotune and math SDPA, with evaluation batch64.

The source adapter gathers supervised logits from the actual GPT's full output.
Independent full-position cross-entropy and every parameter gradient agree for
depth, recall and parity. Original free-generation scoring is inherited for
parity; token accuracy cannot replace sequence accuracy. Hard depth stops on
length128 validation. Tests exercise first99% stopping without opening test or
selecting a best state. Compiled continuation is exact across the final partial
batch of an epoch, including model, optimizer, sampler and gradient trace.

Full Python verification initially ran495 lab tests in825.09 seconds with13
CUDA skips and one error: the new first-crossing test used `Eager()`'s default
CUDA device despite being intended for CPU. This was a test-fixture error, not
a model or engine failure. Explicit `Eager(device="cpu")` fixes it; all six
focused Basis tests pass with CUDA hidden in15.52 seconds. The separately
completed optimizer phase passes104 tests with14 CUDA skips in12.86 seconds.
The existing CUDA ScheduleFreeMuon allocation failures remain task157 and are
not claimed fixed by running CPU checks. This cohort uses Schedule-Free AdamW.

The complete corrected rerun at18112982 is green:495 lab tests in776.99 seconds
with13 skips, and104 optimizer tests in12.58 seconds with14 skips. CUDA was
hidden for this CPU suite so the Basis cohort remained the only GPU learning
workload. Focused fullgraph CUDA preparation is checked separately below.

Original Basis is sampled with replacement. All depth and recall validation
examples are novel against training, but only157/512 parity validation examples
are novel; its final sampled test contains184/512 novel examples. No predictions
or test accuracies were consulted to obtain those input-overlap counts. The
report preserves the original score and adds a separate novel-input score using
the generator's input key, which is the prompt for generated answers.

The source/data/runtime protocol is frozen before learning. GPU preparation
checks take zero training updates and compare the complete model/optimizer frame
before and afterward. Their timings are preparation measurements, not learning
or isolated throughput results. All three large-model checks passed, including
parity free generation and both recall training batch sizes. Peak allocated CUDA
memory was161.14MiB for depth,403.03MiB for recall and107.85MiB for parity. These
checks populated compilation caches; preparation times in the scored cohort
therefore cannot be treated as an isolated model-speed comparison. The scored
learning cohort is in progress with every final-test split still sealed.

The same assistant performs this review; no delegated reviewer was used.

The first continuation was interrupted after its102,750-update checkpoint.
An elevated process query confirmed the trainer absent and the GPU idle apart
from11MiB. All102,750 gradient records and the canonical endpoint are present;
no final test was opened. `analysis/recovery-1.json` pins the checkpoint,
interrupted phase and logs. The recovery runner preserves those files before
resuming the frozen engine, with the same200,000-update horizon and first99%
criterion. An actual compiled interruption/resume fixture checks the complete
state (excluding elapsed clocks), validation and gradient records against an
uninterrupted run, and retains trailing unsaved logs separately. Changed
horizons, stale evidence, completed cases and final-test disclosure are rejected.
The complete recovery-version CPU verification passes500 lab tests in790.99
seconds with13 CUDA skips and104 optimizer tests in12.09 seconds with14 skips.
The five focused continuation/recovery checks also pass in30.04 seconds.

The first twelve endpoints include both small recall pairs: canonical/gain
crossings are25,100/38,350 and11,600/71,100 updates. Together with the depth
results, these supply no Basis improvement so far. Every final test remains
sealed. The complete cohort still needs hard recall and both parity sizes.

A copied26,100-update canonical large recall checkpoint was replayed at both
Schedule-Free states. The first512 training examples reach502/512 exact
sequences at working weights and512/512 at averaged weights, while validation
is0/512 at both. Validation token counts are718/4096 and712/4096. All28 scalar
measurements reproduce the initial diagnostic exactly. This identifies a
train/validation gap at this checkpoint; it is not an averaging-mode artifact,
a causal normalization diagnosis, or a proof that further training cannot help.
The diagnostic script is separate from the frozen training sources. Its source,
protocol, copied checkpoint and complete raw-result hashes are in the compact
evidence under `experiments/basis_residual/analysis/`.

Three continuation checks pass in25.09 seconds. They reject missing or
duplicate cases, changed endpoints, invalid budgets, re-opened successes
and final-test disclosure. A real compiled six-update trajectory agrees
exactly with a two-update checkpoint followed by the new continuation driver:
parameters, optimizer, sampler, epoch remainder, update counts and every
gradient record. Censoring the unresolved fixture preserves its checkpoint
and budget-exit reason; it does not mark it successful. The full frozen
protocol and training source files are unchanged.

The report's storage reduction was checked on a temporary fixture: the
retained raw file is byte-identical, every aggregate survives, and the
endpoint per-class table is unchanged. Repeating reduction is rejected.
This is a storage check, not a final-test evaluation of the active cohort.

The subsequent full CPU verification passes495 lab tests in1208.03 seconds
with13 skips and104 optimizer tests in12.28 seconds with14 skips. The three
new continuation tests were added after that suite's discovery and pass
separately as recorded above. CPU compilation used a fresh disk cache; the
active Basis run kept its original source and CUDA cache throughout.

Canonical large hard recall seed0 reached the initial100,000-update boundary
without crossing:0/512 validation sequences,1,213/4,096 correct tokens.
Its copied-state CPU probe matches all six canonical validation counts and
accuracies exactly. Both the source and copied-state hashes verify. Working
weights give0/512 validation sequences and1,193/4,096 tokens, while the same
training subset is493/512 exact at working weights and512/512 at averages.
The phase records a budget boundary, not success or impossibility. The paired
candidate and the remaining initial chunks continue before any final test
or explicit extension/censoring decisions.

The initial24-run phase now has20 first99% crossings and four unresolved
hard-recall endpoints at100,000 updates. Large parity is the candidate's
measured Basis improvement:8,400 versus14,000 updates in seed0 and10,200
versus14,000 in seed1. Small parity instead slows:5,000 versus4,600 and6,200
versus5,400. All data and stopping semantics remain the frozen ones.

Continuation1 uses validation only: canonical hard-recall seed1 progressed
from51.27% to82.52% token accuracy in its last25,000 updates, and20.70% of
complete sequences are correct. It continues to first99% within a200,000
update chunk. The other three cases retain their complete100,000-update
states and are explicitly censored. This is a recorded research decision
after observing the initial phase, not a predeclared numerical threshold.
Different horizons and unobserved crossings must remain explicit in the
final report. No final-test predictions entered this decision.

The user's historical check confirms that original GPTMini solved the hard
mode in all three seeds at bd63e509. Its hard-recall crossings were
3,350/2,850/2,600; a later softmax repetition at b7d6e673 gives
2,950/2,150/2,750. The committed README and structured repetition record
agree. The original ignored run directory is absent here, so these are
historical records, not a new checkpoint replay. Their hashes and source
commits are retained in `analysis/historical_baseline.json`. Both optimizer
and architecture changed: learned per-head QK temperature, PreNorm versus
PeriRMS, activation, embedding normalization, epsilon and initialization
must not be conflated. Task162 is now urgent after this cohort.
