# Bounded anchor discovery for general decoder banks

Lean `aef9d966` proves that one observed anchor can remain at a support fixed
point forever while another identifies the same missing answer. This crate
tries a finite, deterministic list of observed anchors before constructing a
`convex_stress_general` solver. Five native fixture tests pass. The original-data
cohort harness and protocol are not yet frozen, and no original score is claimed.

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
in `.kanban/artifacts/209`. The next protocol must include all eight prior pair
tasks and the long blind absolute-difference case, all three modes, original
splits/batches, training-only CE stops and candidate selection, and raw-coordinate
spectral/CE replay. All earlier candidates and failures remain. This is an
adaptive follow-up; it does not complete the full35 goal or transformer comparison.
