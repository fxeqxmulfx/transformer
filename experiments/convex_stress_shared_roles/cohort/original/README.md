# Original shared-role training results

The original CM40 training cohort completed all 23 planned cases and all 1,265
prime stages on 2026-11-14. Five tasks produced actual complex heads in every
mode, with zero observed raw-label errors. This is training and certificate
evidence; it contains no held-out score or new benchmark success count.

The fixed shared-role restriction is useful for some observed systems and
incompatible with others. All 55 primes per case remain in the evidence.
There are 19 complete restricted bases: eight of dimension zero and eleven of
dimension one. Fourteen do not separate the observed answer groups and are
refused. Five yield admissible raw expansions and heads, all at prime 97.
There are also 1,242 answer/prime cardinality refusals and four augmented
equation-cap refusals. No constructor exception or process failure was omitted.

| Training mode | Available original endpoints | Observed errors | CE derivative batches |
| --- | ---: | ---: | ---: |
| Full | 5 | 0 | 5 |
| Mini | 5 | 0 | 86 |
| Forward | 5 | 0 | 0 |

The available tasks are `arithmetic-add`, `arithmetic-square-sum`,
`arithmetic-half-sum`, `arithmetic-perm-add` and
`recurrence-fibonacci-k1-c2-lgt`. Across all 15 endpoints, ordinary raw-vocabulary
record CE ranges from `3.6241303291717486e-9` to `9.626594307035853e-9`.
Original IDs, record order, raw labels and batch sizes remain bound to the
saved data. Repetitions would remain separate records even when identical
algebraic equations are coalesced. Every original case was independently
replayed by the frozen runner before the complete summary was produced.

Fibonacci has 4,704 original records, 191 aligned token identities and a
`4897 × 479` augmented matrix. Its complete restricted basis has dimension
one. The actual initializer has 4,801 states: 4,704 observed table states and
97 default latent states. Every pair among the 191 encoded token identities
has a matched answer code, leaving zero unmatched active pairs. The previous
CM35 training construction had 380 unmatched active pairs and 4,995 states.
The unknown-axis count remains 29,055 over the complete `256 × 256` categorical
range. These are support counts derived from training codes; they are not
counts of held-out mistakes. The shared-role family and compatible unknown
readout remain explicit assumptions, and raw aliases remain unidentified.

The saved Fibonacci full, mini and forward endpoints each have zero observed
errors and CE `9.626594307035853e-9`. They use one full update, ten original
minibatches including the actual tail, and zero updates respectively. The
constructor did not access an arithmetic input formula or a held-out answer.

Discovery took `28.351324533999986` seconds and head construction/training took
`81.51276832299995` seconds across all original stages. The 91 derivative batches
are charged once across later comparison arms. The largest measured RSS among
preparation, case execution, replay and summary processes was 380,508 KiB
(about 372 MiB). These times exclude separate replay/serialization overhead,
which remains in the process resource files. Every numerical process had a
64 GiB address-space cap; certificate, head and record forecasts retained the
56 GiB combined cap. All outputs live on the storage disk.

The immutable producer is `5c2931adeb62071721b6a426d27554e9551c7ae9`, approved as
task 273 before execution. The original run is board task 274. Its pins are:

| Artifact | SHA-256 |
| --- | --- |
| Compiled source fingerprint | `f7fc185f4a9f11eb7228aaeae1dc72e867f31c9818363e470ded1671d8e4b96a` |
| Frozen release executable | `31922d4360f7bb4b31eeb8b45bb680e465fd5e4218bfe7164cb97d9182fd2a00` |
| Original summary | `41f8a9b8f7ddbb68352cead575061a5e85b0dd5871c869c8dfa786b55a03e71e` |
| Original inventory | `ea017408af3409b009dd87c258700568665b90dab172e4d2c3a58809ba9e82f8` |

The frozen source graph and executable are in
`/run/media/misha/storage/transformer-runs/freezes/shared-roles-cohort-v1`.
The frozen original request and runner are in the neighboring
`shared-roles-original-v1`. The stable original output root is
`/run/media/misha/storage/transformer-runs/convex_stress_shared_roles/training-v1`.
Keep these paths unchanged because the summary binds absolute artifact paths.
Source, scripts and request are preserved; generated JSON, models, logs and
resource measurements remain local and uncommitted.

Regenerate the compact findings from the complete original summary:

```sh
out=/run/media/misha/storage/transformer-runs/convex_stress_shared_roles/training-v1
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_shared_roles/cohort/original/report.jq \
  "$out/summary.json"
awk '/Maximum resident set size/ {if ($6>m) m=$6} END {print m " KiB"}' "$out/"*.resources
```

For fresh numerical review, use a new directory on the storage disk, leaving
all original files unchanged:

```sh
freeze=/run/media/misha/storage/transformer-runs/freezes/shared-roles-original-v1
review=/run/media/misha/storage/transformer-runs/shared-role-training-review
mkdir "$review"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r task; do
  /usr/bin/time -v -o "$review/$task.resources" \
    prlimit --as=68719476736:68719476736 -- "$freeze/shared-roles-training-v1" \
    replay-case "$out" "$task" > "$review/$task.json"
  cmp "$review/$task.json" "$out/$task.replay.json"
done < <(jq -r '.planned[]' "$out/inventory.json")
prlimit --as=68719476736:68719476736 -- "$freeze/shared-roles-training-v1" \
  summarize "$out" > "$review/summary.json"
cmp "$review/summary.json" "$out/summary.json"
```

Fresh review must additionally verify all source, executable, protocol and
artifact hashes and rerun the native and direct Lean/global/index checks at
the submitted findings commit. The full comparison still needs both original
105-choice arms, all 35 primary outcomes and the unchanged transformer/depth
baselines. The recorded DiffGPT three-seed result remains separate evidence.
