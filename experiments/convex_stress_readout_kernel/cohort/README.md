# Original training with declared output partitions

This adapter tests the reviewed CM35 component at `4bd03816` on the original
stress-suite training pairs. Unlike the frozen CM34 cohort, it explicitly
supplies the declared output partition to discovery. It retains every original
raw label for the complex cross-entropy head. It has no held-out evaluator or
parent selector; native fixtures establish no new benchmark score.

`protocol.json` fixes all 23 original pair tasks, all 55 supported primes and
the complete 35-task suite. The unchanged native generator verifies original
fingerprints. Task names select the original data and artifact location only;
the component receives categorical inputs, raw labels, the fixed partition and
the original minibatch size. Every observed equation is retained.

The full output partition is the original metric's public readout. For field
metrics, each raw output class `c` maps to `(c - floor(vocabulary/2)) mod 97`;
otherwise it maps to itself. This computation applies to output names only.
It supplies no input arithmetic rule, proposed answer or held-out target.
The adapter checks uniform metric kind on training data and surjectivity over
the complete vocabulary. Each original input record and `readout.json` saves
all partition entries and the answer count. The same partition is supplied at
every prime; cardinality mismatch is a recorded pre-elimination refusal.

| Stage | Original information retained | New operation |
| --- | --- | --- |
| Discovery | Every observed categorical pair and raw label | Apply the declared output partition; compute and independently check the complete grouped kernel |
| Expansion | Original observed raw class order | Copy grouped vectors to raw classes; reject incompatible observed group vectors |
| Full CE | Original raw vocabulary and labels | One full update at rate 1 |
| Mini CE | Original example order and minibatch size | One epoch with its shorter final batch |
| Forward | Same complex initialization | Zero CE gradients |

No raw-system complete-basis report is fabricated. Each stage stores the genuine
grouped report, the separately checked raw expansion and the actual complex
states, gradients, default constraints, Gram/SVD audits and CE trajectories.
Summaries distinguish complete grouped kernels from admitted raw expansions;
both are distinct from available mode endpoints or benchmark passes. An
observed collision proves this candidate family incompatible, while observed
separation does not prove an unknown target's global compatibility or additive
realizability. Those CM35 hypotheses remain unproved in the numerical reports.

Independent replay rebuilds original inputs and the declared partition,
checks all equations/basis directions and raw bindings, then replays actual
real/imaginary state, derivatives, projected updates, masks and work counters.
It also reruns discovery and construction with the identical executable.
The ledger binds every artifact by size and hash, but numerical correctness
does not follow merely from matching those hashes. Every error and refusal
must replay. Historical durations remain measured observations, not proofs.

Memory and work limits are unchanged: 64 GiB address-space limit, prospective
56 GiB head-buffer allowance, bounded finite elimination and 32 billion
scalar-work forecasts. One prime is held at a time, each artifact has an 8 GiB
limit and disk reserve is checked before each case and prime stage. A case
refused before starting records exit 125 and zero new construction; the runner
continues to account for all subsequent cases and the final summary. Partial
files and process/resource records remain when a later operation fails.

Ten adapter tests check all 55 stages, genuine multidimensional kernels,
cardinality and equation refusals, all three original-CE modes, raw aliases
under a field readout, complete failed-case accounting, changed held-out labels,
rehashed partition/basis/tensor/trajectory mutations and source/scope binding.
A shell fixture makes disk space fall between preparation and the first case,
then recover for the second, proving that the first refusal is retained and
the second case and summary still run. Initial disk refusal also stops cleanly.
The 15 reviewed component tests supply the CM35 unseen-answer witness.

From the repository root:

```sh
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path experiments/convex_stress_readout_kernel/cohort/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --manifest-path experiments/convex_stress_readout_kernel/cohort/Cargo.toml --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --manifest-path experiments/convex_stress_readout_kernel/cohort/Cargo.toml --offline --locked --all-targets -- -D warnings
bash -n experiments/convex_stress_readout_kernel/cohort/run_training.sh experiments/convex_stress_readout_kernel/cohort/freeze_training.sh
prlimit --as=68719476736:68719476736 -- cargo build --manifest-path experiments/convex_stress_readout_kernel/cohort/Cargo.toml --offline --locked --release
```

After tasks 257 and 258 are reviewed, invoke `freeze_training.sh REPOSITORY_ROOT
PRODUCER_COMMIT NEW_REQUEST` from the clean submitted checkout. Preserve its
source fingerprint, exact executable, request, protocol and script hashes
outside the temporary worktree. The compiled fingerprint covers this adapter,
CM35 core, the unchanged CM34 numeric core and the inherited dependency chain.
Set `READOUT_KERNEL_COHORT_BINARY` to the frozen executable, then run
`bash run_training.sh REQUEST FRESH_ROOT`. Use that identical executable for
`replay-case ROOT TASK` and `summarize ROOT`. Generated evidence belongs in
the root experiment's ignored `cohort/runs/` directory.

This is an attributed port of `convex_stress_kernel/cohort` at `166bebc3`;
the declared output mapping is the same as `convex_stress_kernel_compare`
at `d51aa422`, now explicit construction side information. The shell failure
test ports the comparison runner at that commit. All existing frozen protocols
and states stay unchanged. The complete comparison with the existing baseline
and any full-pass claim require subsequent held-out evidence.

The original training cohort is now closed under producer
`21e86e7c0054ce16397998726b715155be1cbebd`. All 23 cases, 1,265 stages and
initial independent actual-state replays completed with zero process failures.
The source fingerprint is
`6af7080d925c81800c843efa4ef16fe924468136eb7c4ae11186492929070172`;
the frozen executable SHA256 is
`abb22d6e853be31442dd18d2cd8d5efc91c6874623f7620aef5c780bda3fa480`.
Generated evidence stays in `runs/training-v1/` outside the worktree.

| Original outcome | Count | Interpretation |
| --- | ---: | --- |
| Answer/prime cardinality mismatch | 1,242 | Recorded before elimination; no hidden prime selection |
| All-observation equation cap | 4 | The complete original data exceed the declared bound; no truncation |
| Complete grouped kernel with incompatible observed groups | 12 | Basis completeness alone does not establish compatible target structure |
| Admitted grouped kernel and raw expansion | 7 | Each provides all three actual complex endpoints |

Nineteen stages have complete grouped bases; only seven admit raw expansions.
The six tasks cube-plus, subtraction, addition, square-sum, half-sum and
permuted addition each have a one-dimensional modulus-97 kernel. Fibonacci
has two independent directions at modulus 97 and now also provides all three
modes. Every available endpoint has zero observed raw-label errors; ordinary
CE is about `3.62413e-9` on the first six and `9.62659e-9` on Fibonacci.
No held-out label, score or parent-selection rule was used.

Fibonacci's actual range has 291 vector states and 4,995 head states after
retaining its 4,704 distinct observed inputs. Of the complete 256-by-256
categorical grid, 29,055 cells have an unknown axis and 380 active cells have
no matching observed class vector. These are construction masks, not counts
of errors or held-out coverage. The matched region still carries CM35's
conditional unknown-target assumptions. An available low-CE model is not a
claim of a new stress-suite pass.

Recorded construction took `14.773404514` seconds for discovery and
`59.462444964` seconds for heads, with 131 gradient batches in total. Forward
endpoints performed zero gradients. Peak run/replay RSS was 372,048 KiB
(about 363 MiB), under the 64 GiB cap. Preparation, source/data checking,
independent replay and summary costs remain separately recorded; no matched
speed comparison is asserted. The complete summary SHA256 is
`08eefbe7758513273f0353161af2027eeefc5fe15d244a2cad08ffcb7e8cd7e4`.

Fresh submitted-commit review and the complete parent/baseline comparison are
separate steps. Existing benchmark counts remain unchanged until that comparison.
