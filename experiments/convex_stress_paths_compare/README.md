# Reviewed depth and transformer comparison

Does adding the already reviewed retained-path heads repair the full35
comparison while preserving every prior result and both initializer policies?
This experiment compares saved endpoints using ordinary training cross-entropy.
It performs no new training, feature discovery or prediction evaluation.

| Parent | Required coverage | Use |
| --- | ---: | --- |
| CM29 decoded consequences, task225 | 105 choices and23 constructor outcomes | Unchanged first candidate for each task/mode; whole earlier comparison retained |
| Original depth, task175 | 288 outcomes | Preserve every architecture, depth, refusal and cost |
| Retained depth, task175 | 48 outcomes | Four tasks, three modes, two transports, two separate initializer policies |
| Conventional transformer, task191 | 35 seed-one endpoints | Original success criteria, caps, checkpoints and costs; one run referenced across modes |

Each initializer policy has its own105-choice arm. For an eligible task/mode,
the order is the prior endpoint, retained Residual, retained Fourier. A later
endpoint replaces the earlier one only if training CE improves by more than
`1e-12`. The additional comparison has no certificate tie-breaker. The earlier
parent's own choices and certificate policy remain preserved. Held-out metrics
cannot enter the score selector, and the two initializer policies are never
selected against each other. The other31 tasks keep their earlier choices.

The theory is unchanged: `review/flat_spectrum.md`, CM5–CM8 and CM29, and
`review/convex_depth.md`, CD5–CD9, distinguish convex prediction-space CE and
fixed-feature heads from discrete proposal discovery. Fixed nonlinear paths
supply inputs to a convex head. This endpoint
comparison is not a new joint optimizer or a proof of unconditional recovery.
Numerical spectral audits are not exact-real Lean certificates.

Complete parent summaries and inventories are copied byte-for-byte. Their
immutable references retain all older parents, failures and costs. Every depth
result, replay, actual model and retained feature cache is hashed; the ledger
includes refused slots. The transformer adapter binds every observation
checkpoint, final endpoint and numerical replay. Consequence model bytes are
bound through the original sealed bank and dense-view hashes; typed model
hashes and numerical correctness inherit the producer's independent replay.
Unchanged earlier105 model bindings inherit their exact reviewed parent.

Integration replay rehashes the retained evidence, reconstructs the observed-CE
selection and compares the full report. It uses the pinned numerical replay
metrics; it does not rerun predictions or training trajectories. Every parent
is charged once. Both policy arms and all historical attempts remain visible;
these are not equal-compute or speed comparisons.

Eight native tests construct complete synthetic artifact trees without reading
original run outputs. They cover105 prior choices,336 depth outcomes,35
transformers, a failed constructor,18 allocation refusals, a transformer cap,
and a missing forward model. Checks reject changed models, caches, data, replay
seals, complete parents and rehashed selection/report mutations. Duplicate or
missing cases and nonfinite CE are rejected. Changing held-out fields cannot
alter selection; an earlier CE tie wins separately within each policy.

```sh
cd experiments/convex_stress_paths_compare
cargo fmt --check
prlimit --as=68719476736:68719476736 -- cargo test --locked -- --test-threads=1
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release
```

Original integration belongs to task226. It must wait for tasks175,191,225 and227
to be approved. The launch scripts enforce those board gates. Each input also
requires a successful numerical review exit, reviewed commit and a review
summary identical to the parent summary. Freeze the committed source, binary,
protocol and generated request before invoking the comparison:

```sh
# From the repository/worktree root, after the four review gates close:
bash experiments/convex_stress_paths_compare/freeze_request.sh \
  /home/misha/lean_projects/transformer "$(git rev-parse HEAD)" REQUEST.json
export PATHS_COMPARE_BINARY=/absolute/path/to/frozen-comparison-binary
bash experiments/convex_stress_paths_compare/run.sh REQUEST.json FRESH_RUN_ROOT
```

The request pins both complete parent and inventory byte hashes. Its review
paths refer to the retained independent checks under `.kanban/artifacts`.
Generated requests, manifests, reports and logs remain local; source and the
generator are committed. Hashing streams64KiB blocks, JSON files are bounded to
8GiB, individual binary/cache artifacts to16GiB, and both comparison processes
run under the user's64GiB address-space limit.

## Original integration (2026-11-13)

The frozen producer `ad272c62c44dfb9ca184b379bc6f017efd473e7c` completed
preparation and integration replay under `runs/original-v1`. Both initializer
arms have the same counts:

| Mode | Every held-out split passes | Primary split passes | No model |
| --- | ---: | ---: | ---: |
| Full batch | 33/35 | 34/35 | 0 |
| Minibatch | 33/35 | 34/35 | 0 |
| Forward only | 32/35 | 33/35 | 1 |
| Conventional transformer | 32/35 | 33/35 | 0 |

Each arm selects twelve retained Residual heads: depth4 on binding and both
depth tasks, and depth2 on MQAR, across all three modes. Shifted Boolean AND
still fails its out-of-distribution split, and the Fibonacci recurrence task
still fails. Forward-only also lacks a branch-amplitudes model. The transformer
row is the independently reviewed task191 cohort, referenced once across modes.
These are observed-CE choices among retained models with their original costs;
they do not establish matched compute or a new training algorithm.

Integration replay verifies all6786 retained artifacts, both105-choice arms,
all336 depth outcomes and all35 transformer endpoints. Its summary hash is
`23922c1c5c1c731a6af565b881543416907498f0bec94129bf2bc99c2bfdb956`.
Preparation took10m32s and replay20m44s, with peak resident memory6.9GiB.
The replay rechecks artifact bindings and selection using the independently
reviewed numerical endpoints; it does not retrain or reevaluate predictions.
Task226's separate review is required before this result becomes a new parent.
