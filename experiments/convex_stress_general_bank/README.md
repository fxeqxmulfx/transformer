# Bounded anchor discovery for general decoder banks

Lean `aef9d966` proves that one observed anchor can remain at a support fixed
point forever while another identifies the same missing answer. This crate
tries a finite, deterministic list of observed anchors before constructing a
`convex_stress_general` solver. Twelve native fixture tests pass. The frozen
original-data cohort now completes all 27 task/mode cases and all 108 decoder
slots. Absolute difference and its long blind variant pass in every mode;
the other seven tasks refute this particular declared decoder family.

The mathematical source is `review/flat_spectrum.md`, CM19. Soundness uses its
theorem for any observed anchor and any finite budget. Choosing one certificate
by its coverage does not need a new preservation theorem. Combining equalities
from different anchors would require an endpoint satisfying their intersection;
this implementation retains one complete certificate and its proved fallback.

## Declared bank and search

`decoder_bank(V)` declares four fixed maps on `q=2V-1` latent states:
`d_w(z)=floor(min(z,q-z)/w)`, for widths `w=1,2,3,4`. The map depends on the
vocabulary alone and is shared by every task of that vocabulary. It reads no
input token, task name, training label or missing answer. This is an explicit
structural prior on the latent decoder; arbitrary class renaming requires
renaming the supplied decoder maps as well. It does not learn an unrestricted
decoder or prove that a target belongs to this family. The search API also
accepts an independently supplied finite decoder bank.

Distinct observed triples determine incident degrees. Anchors are sorted by
descending row-degree plus column-degree, then ascending total anchor-fiber size
across the whole bank, then lexicographic observation. Repeated observations
remain in propagation and its costs. Arrival order and batch boundaries do not
change the canonical search.

Every attempted anchor runs the fixed number of simultaneous sweeps through
**all** bank members and all their anchor states. Only a complete union passes
to the independent dense verifier. Among complete certificates, retain the one
with the most singleton cells, breaking ties by first discovery. Stop at full
coverage, a sound inconsistency certificate, the anchor cap, or a resource
refusal. A cap refusal is not a proof of inconsistency. A previously completed
certificate can remain usable when a later attempt runs out of budget.

Every attempt records its configuration, costs, refusal or complete coverage,
live-branch count and certificate digest. The selected certificate is retained
in full. `replay` repeats every attempt and the choice, rejecting any changed
record. The shared work cap counts producer decoder comparisons across all
attempts. The independent checker repeats these comparisons for completed
certificates, with its costs separately reported. Neither count is a wall-clock
bound or a complete count of loop/index work.

## Fixture results

| Case | Result |
| --- | --- |
| Lean three-by-three stall | One anchor refuses the missing label; trying later observed anchors identifies it |
| Five-state larger fibers | The declared four-map bank eliminates incompatible members and certifies the withheld label using the member with fibers of sizes three and two; all three solver modes accept its whole-bank certificate |
| Two genuinely conflicting decoders | Every anchor retains both possible missing answers; no fitted member is chosen as truth |
| Work and memory caps | An unfinished union supplies no constraints; the earlier checked certificate and every refusal are retained |
| Evidence and input handling | Reordered/batched input, repetitions, output renaming with unused classes, contradiction detection, serialization and tamper rejection pass |

Memory checks precede degree arrays and propagation. Each attempt forecasts
both its complete checked result and the previous best certificate, plus search
and serialization allowance. Limits cannot exceed 64 GiB, and an OS cap remains
necessary for allocations outside those explicit buffers.

```sh
cargo fmt --manifest-path experiments/convex_stress_general_bank/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 prlimit --as=68719476736:68719476736 -- cargo test --offline --manifest-path experiments/convex_stress_general_bank/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_general_bank/Cargo.toml -- -D warnings
```

Source fixtures are the reproducibility inputs; generated evidence stays local
in `.kanban/artifacts/209` and the root's `runs/` directory.

## Frozen original-data follow-up

`protocol.json` includes the eight prior pair tasks and the long blind
absolute-difference case, all three modes, unchanged source splits and the
original minibatch order, sizes and tails. It declares the same four-member
vocabulary-only bank for every case. The search permits 16 anchors, 16 sweeps
each and one billion producer decoder comparisons in total; independent
checking is charged separately. No original data enters the fixture tests.

| Setting | Fixed value |
| --- | --- |
| Full / mini / forward update caps | 128 / 512 / 128 |
| Observation cadence | Original task cadence, capped at batches per epoch |
| CE stop | Positive scheduled update with ideal-CE excess at most `1e-8` |
| Feasible endpoint repair | At most 12 checked bisections |
| Candidate choice | First within CE tolerance, otherwise smallest training CE |
| Pass threshold | 99% exact sequences on the primary and every held-out split |
| Memory | Explicit preflight and 64 GiB address-space cap |

Discovery reads all training observations before numerical minibatch updates.
This preprocessing is repeated and charged once per task/mode case, then shared
among its four candidates. Every decoder receives a recorded refusal or a full
numerical fit. The runner finishes all candidate endpoints and chooses by CE
before evaluating any held-out split. It retains nonselected models too.
There is no online-learning or equal-compute comparison claim.

Replay reconstructs discovery, every update, the complete state, endpoint,
training trace, counters and CE-only stop/choice. A separate auditor recomputes
decoder sums, every nonconstant complex DFT matrix, SVD norms and ordinary CE
directly from saved latent coordinates. It accepts a genuinely complex phase
endpoint and rejects a probability table with correct certified labels that
violates the nuclear budget. Additional tests cover rehashed artifact tampering,
larger-fiber candidate selection, all-bank inconsistency, minibatch tails and
complete accounting of explicit process failures. Shared generators and LAPACK
remain explicit; numerical tolerance is not an interval certificate.

After committing and freezing the producer, build and launch from the repository
root, with `GENERAL_BANK_PRODUCER_COMMIT` set to that full commit hash:

```sh
prlimit --as=68719476736:68719476736 -- cargo build --offline --release --manifest-path experiments/convex_stress_general_bank/Cargo.toml
bash experiments/convex_stress_general_bank/run_cohort.sh experiments/convex_stress_general_bank/runs/cohort-v1
```

`GENERAL_BANK_BINARY` can name the frozen copied producer. The launcher requires
a fresh destination and a 16 GiB disk reserve, records training/replay exit codes,
keeps partial files on failure, and continues every remaining planned case.
The summary requires all 27 process slots. Only successfully replayed, audited
endpoints have scores; failures remain explicit with no score. Scripts and the
producer are frozen together before launch; a running script is never edited.

## Original results

Producer `eb65aa73` finishes all 27 cases with zero process or replay failures.
All six returned models pass complete update/state replay and independent raw
coordinate, decoder, DFT/SVD, CE and original-split audits. The other 102 decoder
slots are explicit refusals. Recomputing the sealed summary gives the same file
hash, `bcc5e6d9bd17022f7ab45652272f596f654f398dc744010c3c57c83036f496a7`.

| Original task | Selected decoder | Certified cells | Held-out exact sequences, full / mini / forward |
| --- | --- | --- | --- |
| Absolute difference | Width 1 | 9,409 / 9,409 | 4,705 / 4,705 in every mode |
| Blind absolute difference, long tier | Width 2 | 9,409 / 9,409 | 4,705 / 4,705 in every mode |
| Permuted addition, addition, subtraction, division, multiplication, sum-cube, sum-power-four | None | Whole-bank inconsistency | No endpoint score; all 21 case refusals retained |

The two successful tasks use the first scheduled anchor; the original results
therefore do not measure an advantage from trying multiple anchors. Discovery
uses 153,335,629 producer decoder comparisons per ordinary case and 779,272,915
per blind case, with the independent checker repeating the same number. This
setup is charged in every mode. Peak RSS over preparation, fitting and replay is
331,628 KiB (about 324 MiB), within the 64 GiB process cap.

Every successful fit stops at its first positive observation, update 19, with
CE excess about `-3.4e-13` from floating-point roundoff. Full/mini/forward gradient
counts are 89,376 / 4,704 / 0 per successful task. The certificate and feasible
fallback already attain ideal observed CE at update zero. Ordinary absolute
difference retains iterate weight one; the blind case returns fallback weight
zero in all modes after the checked repair. Thus these results demonstrate
training-only identification under the declared family and a valid endpoint,
not numerical convergence or a benefit from the 19 updates. The blind raw audit
checks 192 nonconstant frequencies and genuinely complex moment entries.

The seven refuted tasks show that this vocabulary-only bank is not a universal
prior. Their refusals neither refute the broader decoder hull nor replace earlier
models that fit those tasks. Every old candidate remains part of the comparison.
The separate [`comparison/`](comparison/) harness implements CE-only integration,
including the added long case and complete full-suite baseline; its metadata
tests pass. Actual integration awaits complete audited predecessors. This is
an adaptive follow-up; it does not complete the full35 goal or transformer comparison.
