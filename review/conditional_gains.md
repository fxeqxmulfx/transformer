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
