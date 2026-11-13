# Serial catalogue preparation

This native component runs the complete fixed55-prime catalogue, one binding
report at a time. It is the first part of task231. The training-score selector
now retains both complete original parent arms; artifact/data/metric wiring
for the original adapter remains unfinished. No original cohort is evaluated.

Each stage retains its original request, discovery outcome, actual complex
model, all three CE endpoints, audits and original-row certificates. Returned
errors are saved explicitly and cannot be treated as model results. A compact
ledger holds hashes, work counters, every refusal and elapsed stage times; it
never keeps all dense reports resident. Observed-CE selection uses increasing
prime order and earlier ties separately for full/mini/forward. It cannot read
held-out targets. Construction, training and checking time is charged once per
stage; measured time is preserved rather than recomputed during replay.

The native source inventory includes the component, its immediate numerical
dependencies, the runner and protocol, plus both existing shared-source
fingerprints. Metadata also pins the executable, configuration and producer
commit. Replay checks all55 original reports, including errors/caps, recomputes
actual saved-state semantics and reconstructs the compact ledger and CE choice.
A changed model still fails after its outer hashes are rewritten.

Five tests cover the entire55-stage catalogue. A six-class fixture admits both
two-state and three-state quotient codes; all modes select the three-state
head by observed raw-label CE, while retaining the other successful proposal
and all53 refusals. Separate tests cover all55 returned policy errors,
elimination caps, changed states after rehashing, removed stages, altered work,
source/configuration changes, oversized files, and earlier CE ties regardless
of certificates or held-out metadata.

Three further tests exercise the parent selector, for eight tests in total.
Each of the two initializer policies keeps its own105 choices across35 tasks
and three modes. New offers use increasing prime order and replace that arm's
parent only on a training-CE decrease greater than1e-12. The selector preserves
missing models, checks336 retained/original depth slots and35 transformer
endpoints, and rejects missing, duplicate, unknown or invalid score records.
Held-out fields and certificate flags cannot affect its choices. These are
synthetic parent tests; they do not validate original artifact bytes or metrics.

```sh
cd experiments/convex_stress_quotient/cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked -- --test-threads=1
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release
```

The generic CLI accepts a serialized `Configuration`; the shell runner caps
both construction and replay at64GiB. It creates new output files only. Files
are streamed with an8GiB limit. Generated artifacts remain local and ignored.
The complete original adapter must still bind the reviewed task226 parent,
preserve both105-choice initializer arms and all35 transformer endpoints, and
freeze tested source/binary/protocol before original discovery. Successful
synthetic code tests do not establish raw-label or original benchmark passes.
