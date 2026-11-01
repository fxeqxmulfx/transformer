# Conditional gain research loop: same-assistant review

Task158 continues the user's requested gradient/normalization search. The current
production baseline is GPT `df1e9f7`; the task155 accurate RMS backward and task156
architecture-switch removal are already landed. This review is performed by the
same assistant, without subagents.

The new supplied-feature task asks a distinct question: whether the model can
learn different branch amplitudes for two input conditions. Both before-RMS and
after-RMS gates have two parameters per branch, identical initial functions, and
explicit finite optima. Both use the corrected local RMS backward. Independent
scalar derivatives and finite-optimum checks cover both placements and all three
scales. Compiled complete-state continuation matches uninterrupted execution.

The frozen 24-case cohort completed all 240,000 updates. After-RMS recovered the
teacher in 12/12 cases; before-RMS in 4/12. The latter succeeds only at scale .001.
The criterion combines held-out relative MSE and conditional coefficient error,
so small target energy or one fitted condition cannot conceal failure. Test was
opened only after every case finished. All failures and curves remain in the
report. Nonzero pre-RMS gradients mean finite-budget failure, not impossibility.

The transformer extension has 1,024 additional parameters at width128/depth4,
about .128% over its 798,336-parameter baseline. Both variants start with identical
common tensors, logits, common gradients and generator state. No bias or
production architecture flag is proposed. The candidate is not yet accepted:
the frozen rule requires earlier first99% validation in both paired seeds.

An independent frozen `diff_gpt` source copy checks baseline correspondence.
Small FP64 source tests passed, but a stricter actual-size FP32 audit found a
last-bit backward discrepancy in the inherited lab attention graph: maximum
absolute differences were 1.49e-8 and 7.45e-9 for seeds11/17. Logits were exact.
GPT computes centered-value subtractions separately and zeros the first output
by assignment; the historical lab port reused one subtraction and concatenated
a zero. A preflight matching GPT's order eliminated both gradient discrepancies.
The already-scored synthetic pin must remain unchanged; update the new canonical
port and refreeze the unused transformer protocol before GPU learning.

The existing Lean gain derivative ratio predicts pointwise radial attenuation.
It does not predict optimizer convergence or held-out accuracy. This experiment
therefore tests a missing learning claim, rather than refuting an established
flat-spectrum rank theorem. Integration and final review remain pending the
matched transformer results.

Checks for the initial synthetic/protocol commit: 19 focused tests passed,
including compiled CPU/CUDA forward/backward, exact resume, frozen source
correspondence and first99% validation-only stopping. The full `./run.py test`
lab phase passed 487 tests with 3 skips in 815.82s. The optimizer phase passed
102/104; the same two ScheduleFreeMuon CUDA allocation failures recorded in
task157 remain (75 allocations of 8,519,680 bytes in FP32 and FP64). No optimizer
source changed, and this cohort uses Schedule-Free AdamW. The overall command
therefore remains nonzero; the unrelated failures are not reported as green.

The follow-up source port now preserves GPT's exact attention graph. Eight
focused tests pass, including bit-identical FP32 logits and every parameter
gradient for both seeds, primes97/193, and lengths2/4; compiled CPU/CUDA checks
and complete-state resume also pass. The original unused protocol is archived
unchanged in `preflight/`. Before any GPU outcome, the new protocol adds P193
and per-parameter diagnostics and requires earlier crossing in all four pairs.
Python3.15's parent-owned sampler was independently checked on its child:
1,000 samples in one second, zero errors, without changing tracing permissions.

The eight-case transformer comparison is complete: P97 seed11 crosses at
26k/21k updates and seed17 at24k/20k (canonical/candidate); P193 crosses at
8k/8k and15k/13k. Every case reaches the requested99% validation stop.
The candidate fails the predeclared requirement to improve all four pairs.
Retain production GPT `df1e9f7`; do not relabel three wins and a tie as a
passed promotion criterion. The independently reserved final tests are all
above99%, but do not choose the architecture or stopping step. CPU replay
reproduces every canonical CUDA validation count at the endpoints.

P193 initially failed a metadata comparison before learning. CPU and CUDA
float64 means for the descriptive copy-last statistic differed by one ULP;
all labels, row hashes, input hashes and counts were exact. Commit23d16a5e
retains the original protocol and completed checkpoint hashes, pins the
replacement runner in an explicit amendment, and permits only that diagnostic
difference. A regression rejects changed rows, counts, other floats, NaN and
a two-ULP deviation. Preparation failures now leave an explicit phase record.
No engine or training source changed, so completed P97 trajectories are intact.

All135k measured gradients are finite and nonzero. All eight Python3.15
parent-owned profiling windows completed within their intended case. They
show63–74% main-thread wall residency at the blocking index upload, with
roughly5–6% sampler errors. This is not a CUDA timing proof: queued compute
can account for an upload wait. Task159 requires an isolated timing/control
experiment before changing that path. Training clocks and full histories
are retained; concurrent CPU checks and cache differences limit throughput
interpretation.

The proposed actual GPT implementation was also tested in an isolated
checkout. Initial CPU model/grokking coverage passed302 cases, skipped19,
and exposed two test-reference defects: the frozen reference's import-sort
hash round trip and a constant-prefix oracle missing the new gain gradient.
Both were corrected;52 focused/application cases passed. Eight actual GPT
source, cache/state and fullgraph CPU/CUDA tests then passed, including
checkpointed backward and nonzero gains. These checks establish that the
candidate was implemented faithfully; they do not override its failed
promotion criterion. The user-requested Basis follow-up is tracked by160.
