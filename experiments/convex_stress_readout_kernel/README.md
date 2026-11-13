# Declared readout constraints with original complex CE

Can the additional equations proved in flat-spectrum CM35 identify a grouped
answer that the complete original raw-label kernel leaves undetermined?
This separate Rust library implements the theorem after Lean commit `ff6313e1`.
It changes the information supplied to discovery: the caller declares a fixed
surjective partition of the **entire raw vocabulary**. This is explicit output
metadata, not a learned unknown decoder. Input names remain categorical; there
is no arithmetic answer feature or held-out input to construction.

CM35 proves that a compatible unknown scalar code factors bijectively through
the declared readout when the answer count equals the prime-field cardinality.
The native component verifies that cardinality and the declared partition,
then applies the unchanged CM34 complete-kernel solver/checker to every grouped
observation. It rejects observed group-code collisions. Neither a complete
basis nor a collision-free observed bank establishes the unknown target's
additive realizability or global readout compatibility; those Lean hypotheses
remain conditional in every saved report.

The complete-basis report is genuinely about grouped observations. A separate
checked expansion copies each grouped vector to the original observed raw
class names. It does not fabricate a certificate for the larger raw-label
kernel. Original raw labels, observation order, vocabulary and multiplicities
are retained by the complex model and its ordinary cross-entropy loss. The
default representative is the first matching observed raw class, fixed before
training. Actual observed states still distinguish raw aliases within a group.

| Mode | Fixed features and labels | Updates |
| --- | --- | --- |
| Full | Expanded vectors; original raw vocabulary and labels | One full-data CE gradient update |
| Mini | Same model; original ordered examples | One epoch, including the final shorter batch |
| Forward | Same initialized complex model | Zero CE gradients |

`construct.rs` and `binding.rs` port the small CM34 wrappers at `166bebc3` to
accept the newly checked expansion. The actual model, Fourier transform,
projection, derivatives, training replay and complex-arithmetic/SVD audit are
the unchanged `convex_stress_kernel` dependency. The private preflight formula
is copied with attribution and compared against the native trainer's recorded
forecast. Convexity concerns the bounded head with these features fixed;
discrete basis discovery is not a jointly convex learned encoder.

Fifteen native checks include the exact CM35 three-corner witness. The raw
complete kernel has two independent directions and refuses the fourth corner;
the grouped complete kernel has one and matches it. All three actual complex
endpoints predict its correct group, yet can prefer the wrong raw alias.
Their observed loss uses all four raw logits, not two grouped logits. Tests
also transport labels, groups and input names, preserve unequal readout fibers,
and distinguish unobserved exact inputs, unmatched active vectors and unknown
axes. Rehashed basis, partition, coefficient, trajectory and work mutations
are rejected by independent checks.

All 55 supported primes retain outcomes: unequal cardinality refuses before
elimination, while feasible requests preserve the original full equation,
nullity, scalar-work and row-operation caps and their costs. The Fourier head
retains at most 513 states per block, four million phase entries and 128 million
real coefficients. Its prospective buffer allowance is below 56 GiB, including
the existing 1 GiB reserve for bounded encodings and field certificates; the
new partition/raw expansion fit within that reserve. Commands have a hard
64 GiB address-space limit. Forecasts are allocation guards, not measured RSS
or formal allocator bounds. Refused initialization and individual modes remain
recorded without discarding valid outcomes.

Reproduce from the repository root without Python:

```sh
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path experiments/convex_stress_readout_kernel/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --manifest-path experiments/convex_stress_readout_kernel/Cargo.toml --offline --locked -- --test-threads=2
prlimit --as=68719476736:68719476736 -- cargo clippy --manifest-path experiments/convex_stress_readout_kernel/Cargo.toml --offline --locked --all-targets -- -D warnings
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- lake build
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
```

These are finite component witnesses, not an original stress-suite result.
The separate [original-data adapter](cohort/README.md) now has ten native checks
for explicit readout metadata, all 55 stages, original CE and complete failures.
Its frozen construction cohort and complete comparison remain subsequent tasks.
The running CM34 cohort and its comparison protocol stay unchanged. No new
benchmark score or full-pass claim follows from this library or adapter.
