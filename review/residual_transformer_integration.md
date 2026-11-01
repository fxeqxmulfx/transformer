# Full-transformer residual integration: task 145 self-review

The same assistant implemented and reviewed this change, as requested. The source
integration is GPT `ad76812`; the query-gradient regressions are `9d16e39` and
`cf1d7c8`; both frozen scaled-model generalization regressions are `5d71786`.
Lab implementation/protocol commits are `3d811802`, `0ff3615b`, `e4edb9f5`.

The port preserves source initialization, exact RoPE, token QK normalization,
centered/clipped XSA including its persistent prefix guard, SwiGLU, Peri norms
and tied readout. Independent source fixtures compare values and every parameter
gradient for all five modes, with unclipped and clipped inputs. AttnRes uses the
paper's key-only scores, zero queries and learned key gain; retaining host output
norms is an explicit deviation from arXiv:2603.15031v1 §3.1 Eqs. 2–4 and §5.
Q/K depth attention is separately labeled. Full Schedule-Free state, sampler,
query coordinates and conditioning state survive resume. No singleton query or
extra ordinary residual is inserted into the depth-attention recurrence.

The frozen 16-case cohort used 7,165.75 of 7,200 GPU wall seconds, including all
profiling, compilation, failures and disposable final capture diagnostics. Four
ordinary controls reached stable validation and 99.957–100% final test. Raw and
conditioned depth attention stayed near chance within their allowances. Recorded
coordinate scales were always one: the oversized-query repair was inactive.
Both fixed-scale controls generalized earlier at update 23,000, but did not reach
the stability target before their shorter caps. One key-only case trained for
20,000 updates; another and one conditioned case never trained. Both failures
remain in the report. No unused old GPU budget or replacement seed was used.

All 449,000 update-gradient norms are finite. Final test was opened after learning
closed. CPU FP32 reproduces all available CUDA train/validation counts, and FP64
with the same epsilon has zero prediction disagreements on all three splits.
The endpoint VJP audit separates each RMSNorm edge from other residual paths and
checks its formula against FP64 autograd. Nonzero query gradients and substantial
normalization gains also in successful controls do not identify a universal bug.

The selected inner-graph math SDPA backend passed four CUDA value/gradient probes
within 2.9e-6/3.9e-7 and improved warmed throughput. Key-only profiling did not
complete. A disposable padded evaluation candidate passed repeated initial and
tail-batch checks without optimizer steps; it remains a diagnostic, not a landed
production capture fix. The original reproducer timed out. The sampling trace's
errors and malformed records are explicitly retained in profiling.json.

The final full CPU run passed 463 lab tests (13 skipped) and 104 optimizer tests
(14 skipped). The affected GPT run passed 173 tests (15 CUDA-only skips). Source,
optimizer-equation, complete-state resume, stopping, reserved-test isolation and
wall-time-limit tests pass. The huge-query regression rejects a no-op conditioner.
Both frozen scaled endpoints reproduce every saved prediction in actual GPT;
removing scaling lowers held-out accuracy from 97.77/98.37% to 36.21/43.77%.
These are learned-function regressions, not a guarantee of stochastic training speed.

The final reporting-only plot edit handles a failed case with empty history;
learning and scoring sources were unchanged. Raw evidence and command output are
under `.kanban/artifacts/145/`, including final-full-cpu-suite.log,
gpt-generalization-regressions.log, rms-local-vjp-check.log, and post-commit checks.
Results, mechanism probes, source hashes, profiling limits and all curves are in
`experiments/residual_generalization/`. Existing Lean identities concern current
states and score bounds; none asserts optimizer convergence. Task 151 now tests
known-solution gradient mechanisms before any further transformer architecture change.
