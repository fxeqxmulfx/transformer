# Complete original training of the CM34 complex model

Retaining the complete observed kernel produces models for seven of the 23
original pair tasks, including Fibonacci. All 23 cases and 1,265 prime stages
finished, and every case passed its initial independent numerical replay.
There were no process, discovery or head execution errors. These are training
results; no held-out evaluation or parent selection occurs in this cohort.

The frozen producer is `166bebc3d1d1d206b1169e8c33b02bde60549afd`, using the
reviewed CM34 core at `9b10d880`. All 55 supported primes from 2 through 257
were attempted in increasing order for every case. Every original observation
and raw label was retained. There was no target-derived prime choice or readout
compatibility filter. Discovery is separate from fixed-feature convex head CE.

| Mode | Original CE schedule | Available endpoints |
| --- | --- | ---: |
| Full | One full-batch update, rate 1 | 55 |
| Mini | One original-order epoch, including the actual tail, rate 1 | 49 |
| Forward | Initial complex head, no CE derivatives | 55 |

All 159 available endpoints have zero observed raw-label errors. CE evaluated
through their actual complex heads ranges from `3.624130329e-9` to
`9.626594307e-9`. This observed fit does not identify unseen answers.

| Original stage outcome | Count | Consequence |
| --- | ---: | --- |
| Complete kernel, dimension 0 | 653 | No direction is invented to make a head available |
| Complete kernel, dimension 1 | 337 | The complete basis remains checked |
| Complete kernel, dimension 3 | 55 | All are Fibonacci stages |
| All-observation equation-cap refusal | 220 | No truncated equation set replaces the input |

Of the 1,045 complete kernels, 988 fail the declared Fourier vocabulary/group/
state caps and two fail the memory or audit-work forecast. The 220 discovery
refusals also retain explicit initialization refusals. The remaining 55 stages
initialize actual complex heads; six of those decline the minibatch schedule
under its 32-billion scalar-work forecast. Every refusal remains in the ledger.

Six tasks each admit one stage at prime 97: cube-plus, subtraction, addition,
square-sum, half-sum and permuted addition. Fibonacci admits 49 full and forward
stages, at every supported prime from 17 upward, and 43 minibatch stages. The six
minibatch refusals are primes 17, 19, 23, 29, 31 and 37. There is no benchmark
selection among these alternatives in this training-only stage.

At prime 97, Fibonacci has a complete three-dimensional kernel and a 5,093-state
complex head. Its retained construction audit counts 29,055 unknown-axis cells
and 568 unmatched active cells over the categorical grid. These are support
diagnostics, not held-out error counts. All three modes fit every observed raw
answer. Whether the latent code and the task's declared output mapping identify
unseen answers still needs the separate complete comparison and its premises.

The original construction records 792.728345770 seconds of complete-kernel
discovery, 1,992.023800371 seconds of head work and 599 gradient batches. These
costs are charged once when imported into both later comparison arms. Preparation,
replay, summary work and their resource records remain separate. The maximum
recorded original run/replay/summary RSS is 550,028 KiB (about 537 MiB), below
the 64 GiB process cap. This is not a matched-compute transformer experiment.

The existing 34/35 full, 34/35 mini and 33/35 forward primary comparison counts
do not change from these training results. The conventional transformer stays
33/35 primary. The separately recorded DiffGPT result remains 35/35 on each of
seeds 0, 1 and 2; this cohort is neither its replay nor a new three-seed result.
Task 255 retains the prospective parent-first raw-CE selector and complete
35-task, two-policy comparison after the parent evidence is reviewed.

The source fingerprint is
`5c748aeb6697d76ceafbc8ef99aee03eb24ee9933ec22c0fdebfc729a6bf2954`;
the exact executable SHA256 is
`f23d149db1aa331a3c7921e4c2a8183871942ed69b3986bec643de86b2836268`.
The complete summary SHA256 is
`08224d2041a51b3189782bc08384976b115077a04505532324144480e66107f1`;
the inventory SHA256 is
`46c66e5b88c40c4a2b5bf968f2f8cdb8f56abfa2a418f8f9661acaa0a40f463a`.
Generated evidence stays outside worktrees in the ignored root
`experiments/convex_stress_kernel/cohort/runs/training-v1/` directory and
`.kanban/artifacts/252/`. New review outputs use the storage disk. The frozen
component and adapter READMEs stay unchanged because they belong to the source
inventory; this separate authored report does not alter that fingerprint.

From the repository root, regenerate the accounting with:

```sh
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_kernel/cohort/original/report.jq \
  experiments/convex_stress_kernel/cohort/runs/training-v1/summary.json
```

The frozen executable at
`.kanban/artifacts/252/freeze-v1/kernel-training-v1` supports `replay-case ROOT
TASK` for each of the 23 `inventory.json` planned cases, and `summarize ROOT`.
Replay checks original inputs, every complete basis and equation, actual complex
coefficients, ordinary CE trajectories and masks. Fresh review repeats all cases
and requires byte-identical replay seals and summary, alongside native core/
adapter tests, format and Clippy, direct Lean builds and the global axiom audit.
The unchanged adapter README gives source-build and prospective-freeze commands.
