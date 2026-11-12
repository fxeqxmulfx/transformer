# General decoder constraints in the convex spectral solver

Can the CM19 certificate constrain actual spectral updates when a categorical
decoder has arbitrary fibers? This crate connects the independently checked
propagation procedure to full-batch, persistent-minibatch and forward-only
updates. It uses full latent distributions, including odd and even cyclic
groups. Twelve native checks pass; no original stress-task run is claimed.

Lean came first: `0807609b` proves sound identification across a declared decoder
bank and the convex restriction; `7d299b04` proves a feasible endpoint from any
live branch, including partial coverage. `aef9d966` proves that anchor choice
can change coverage even after propagation has reached a fixed point.
The source is `review/flat_spectrum.md`, CM19. The numerical iteration extends
the CM18 solver present at `6f2efed1`. FFTW, LAPACK, simplex and singular-value
budget machinery come from the moment/Schatten experiments at `7d299b04`.
This Rust code is not extracted from Lean or proved to converge.

## Domain and updates

Input side `N`, latent states `q`, and original output vocabulary `V` are
independent dimensions. Each cell has a probability vector `P(i,j,z)` over all
`q` states. Every nonconstant complex Fourier moment has operator and nuclear
norm at most `N`. A fixed decoder sums probabilities over its class fibers;
these `V` coordinates are the logits for ordinary log-sum-exp cross-entropy.
All output classes, including unused ones, remain in its denominator.

One solver uses one member of a declared bank. Its output constraints come from
the union over **every** bank member and anchor phase, verified by the separate
dense checker in `convex_stress_propagation`. A singleton answer restricts that
cell to the appropriate latent fiber simplex. Larger fibers retain their free
probabilities; certified rows are not replaced with a chosen hidden state.

Each update uses `x=P_A(z)`, `y=P_B(2x-z-step*g)`, then `z=z+y-x`. `A` imposes
probability rows and the output equalities; `B` imposes all Fourier spectral
bounds. The step is fixed at the reciprocal of the largest decoder-fiber size
times the largest observed-cell multiplicity. This is a tested numerical
recipe, without a convergence or finite-budget theorem.

| Mode | Gradient and observation treatment |
| --- | --- |
| Full | Refresh every sample's original-vocabulary CE gradient each update |
| Mini | Refresh selected sample memories; retain the others and their sum |
| Forward | Evaluate zero derivatives; additionally project observed cells onto their label fibers |

Repeated observations keep their empirical weight. Minibatch indices refer to
the canonical observation multiset in the saved certificate; refreshing an
index twice charges two evaluations but replaces its memory, without adding
the same gradient twice. Updating with all indices agrees with the full mode.
The gradient uses the summed empirical loss, which has the same minimizers as
its mean. Update ledgers count row projections, spectral projections, SVDs and
gradient evaluations. Endpoint checks have a separate norm-evaluation count.

The row-projected iterate need not be spectrally feasible. Endpoint construction
measures every nonconstant Fourier matrix separately. If needed, it bisects a
mixture with the one-hot phase table selected from a live branch. CM19 proves
that this fallback preserves every certified output even if it fails some
training observations. The actual returned latent and decoded coordinates are
checked; neither is repaired by replacing predicted labels. Weight zero is
allowed, and forward-only endpoints can temporarily lose uncovered training
equalities when mixed. No positive-weight or interval-arithmetic guarantee is
claimed. Endpoint spectral tolerance is `N * (1 + 1e-10)`.

## Findings and checks

The partial-coverage fixture withholds one cell of a three-by-three table and
uses zero propagation sweeps, so the certificate does not solve the training
problem. After 240 updates:

| Mode | Initial endpoint CE | Final endpoint CE | Gradient evaluations |
| --- | ---: | ---: | ---: |
| Full | 1.030216342 | 0.551444714 | 1,920 |
| Mini, one sample per update | 1.030216342 | 0.551444714 | 240 |
| Forward | 0.678028932 | 0.551444714 | 0 |

All three returned endpoints have iterate weight one and reach the fixture's
ordinary-CE lower bound numerically. This small fixture establishes actual
learning beyond certified rows; it gives no original-task success rate.

Other checks cover direct DFT versus FFTW for `q=2,3,4,5,8`, independent raw
all-frequency SVD audits, finite-difference decoder gradients with unused
classes, the Lean four-state wide-fiber witness, and the separately enumerated
five-state witness whose decoder has no singleton fiber. A mixture of distinct
live-branch phase tables keeps its ambiguous latent probabilities. Arbitrary
row, column and class renaming commutes with updates. Two fitted bank members
that disagree on a missing label remain separate feasible candidates; the
whole-bank certificate refuses to identify a winner.

The original wrong-minimum fixture initially failed the test's assumed
certificate: anchoring at a diagonal cell stalls forever. Lean now proves
both that permanent stall and recovery from an off-diagonal anchor after three
sweeps. The regression retains the poor-anchor refusals. The stronger anchor
rejects an explicitly constructed, spectrally feasible, wrong-label global CE
minimum of the unrestricted domain. This is a coverage correction, not a
claim that more sweeps or every anchor suffice.

State serialization retains the entire checked certificate, splitting state,
sample gradient memories, their aggregate, counters and projected shadow.
JSON roundtrip and continued updates replay exactly in every mode. Restore
checks dimensions, the instance, finite values, aggregate consistency, the
projection and ledger constraints; it does not authenticate an arbitrary
history. Invalid batches leave the state unchanged. Dimension, memory, decoder,
repair-budget and numerical-overflow refusals are also checked.

## Resources and reproduction

Before allocating the solver, checked size arithmetic combines the certificate
forecast, full latent/output/FFT buffers, gradient memories, state copies and
serialization allowance. Supported dimensions are `2<=N<=256`, `2<=q<=512`,
`2<=V<=1024`, with at most one million observations. The explicit forecast must
fit 64 GiB. An OS address-space cap is still required for library overhead.
FFTW planner operations are serialized within this crate; callers must not
construct another crate's FFTW plans concurrently using a different lock.

From the repository root:

```sh
cargo fmt --manifest-path experiments/convex_stress_general/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 prlimit --as=68719476736:68719476736 -- cargo test --offline --manifest-path experiments/convex_stress_general/Cargo.toml -- --nocapture --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_general/Cargo.toml -- -D warnings
```

The source fixtures are reproducibility inputs. Logs and resource measurements
remain ignored under `.kanban/artifacts/208`. No Python is used.

This optimizes a fixed-decoder convex subdomain, not the whole decoder hull.
Training fit cannot establish that the unknown target belongs to the declared
bank. Useful bank construction, broader anchor discovery, the original-data
protocol, all-35 short/long evaluation and transformer comparison remain open.
Earlier candidates, refusals and running frozen cohorts remain retained.
