# Categorical strips and repeated latent phases

Can a broader family of feasible phase vertices escape the native permutation
capacity obstruction while retaining ordinary cross-entropy and the convex
spectral bounds? CM25, formalized before this implementation in Lean commits
`d8efeb36` and `84cacc46`, supplies two constructions. These use categorical
training data; no arithmetic rule on input token names is supplied.

| Component | Construction and guarantee |
| --- | --- |
| Repeated phases | Each input category may receive any phase in a declared cyclic latent domain. Input count and latent size may differ. |
| Constant strips | Lift an interior phase `z` to `3z` and a flagged category to `1`, in a domain of size `3q`. Decoder residues one and two give the proposed constant label, including the intersection of two strips. |
| Free-axis refinement | Check every possible phase for each input category and choose a minimum observed mismatch cost. Keep the previous phase on ties. Integer certificates prove an optimal update of that axis and nonincreasing observed CE. |
| Decoder refinement | Choose an empirical count maximum with fixed phases, preserving earlier-label ties and the original output vocabulary. |
| Endpoint admission | Require every base latent state to occur among observations. Uncovered states, capped searches and unsuccessful states remain explicit. This coverage requirement does not prove unseen answers. |

The convex prediction hull and ordinary CE are the mathematical model. Native
phase/decoder proposals and their discrete coordinate search are heuristics
inside that hull; the implementation does not claim joint convex optimization
in these coordinates. Lean proves spectral membership and coordinate descent,
not successful discovery of the unknown decoder or phases.

`Model` materializes the actual latent and output tables. Its independent audit
reads those tables, constructs every latent Fourier matrix and measures its
operator and nuclear norms using LAPACK SVD. Each budget is the number of input
categories. It also recomputes full-vocabulary CE and its exact theoretical
identity with unit-logit prediction errors. Floating-point measurements are
not exact real-arithmetic certificates.

The free-axis checker rebuilds each integer cost directly from grouped original
observations, checks all alternatives and the saved minima, verifies the fixed
coordinates and actual prediction errors, and reconstructs the work ledger.
Full deterministic replay additionally binds input, state order, stopping
reason, all states and costs. Equal phases and empty observed axes are allowed;
an empty axis preserves its earlier phase on ties.

Fourteen tests pass, including all 512 binary three-by-three target tables. A
masked four-category fixture with three latent states is repaired to the full
correct table through repeated phase assignments. The Lean three-category
constant-strip witness is checked, including the strip intersection and all
six actual latent frequencies. Tests retain genuine failure and ambiguity:
uncovered latent states are refused, and two fully covered exact observed fits
satisfy the spectral bounds but disagree on an unseen input row. Input, state,
minimum-certificate and work-ledger tampering is rejected.

Generic discovery now proposes constant strips only when an observed row or
column has at least eight examples, all of one class. It explicitly flags
unobserved axes in those proposals, pads unequal interior axis sizes with
unobserved categories, and retains the original vocabulary. These guesses
remain hypotheses about unseen entries. Every proposed interior and the
unchanged full table receive centered random categorical label sketches.
Power iterations propose complex phases; angular quantization allows repeated
phases. The latent-domain bank consists of the input-category count and the
number of observed output classes, with out-of-range sizes explicitly refused.
Neither the task name nor a numeric relationship between input tokens is used.

The default policy is four views, eight sketches per view,64 power steps,
32 seeds,24 refinement rounds and1024 total axis updates. Iteration is by
trial, then view and latent domain, to share early work across proposed views.
All power states, codes, matrices, partitions, missing axes, coordinate states,
integer certificates, measured spectra, refusals and budget stops are retained.
The aggregate report/replay forecast is checked against8GiB before search.

A separate masked four-category fixture now recovers both constant strips
and its three-state interior from observations alone; all withheld outputs
are correct. Its support threshold is explicitly lowered to three for this
small test. Output-class renaming preserves discovery. A zero sketch and
uncovered states remain refused; partial constant evidence has no uniqueness
claim. The separate [`cohort`](cohort/README.md) harness now passes four checks:
unchanged protocols/data, complete fixture replay, earlier-candidate CE ties,
rehashing of altered parent/new ledgers, refused inputs and all nine process
failures with27 retained mode references. The frozen original producer
`69c83182` now completes and replays all nine unchanged tasks, with zero process
failures and still3/9 primary and all-split passes.

| Original task | Observed-CE selection | Observed errors | Validation accuracy |
| --- | --- | ---: | ---: |
| Permuted addition | earlier prior-0 | 0/4704 | 100% |
| Addition | earlier prior-0 | 0/4704 | 100% |
| Subtraction | earlier prior-0 | 0/4704 | 100% |
| Division | tagged-3 | 1793/4656 | 55.670% |
| Multiplication | earlier prior-0 | 1198/4704 | 72.604% |
| Sum-cube | tagged-12 | 2372/4704 | 41.318% |
| Sum-power-four | earlier prior-0 | 3758/4704 | 7.396% |
| Absolute difference | earlier prior-0 | 4061/4704 | 3.783% |
| Blind absolute difference | tagged-14 | 1671/4704 | 61.403% |

Division and blind absolute difference improve from38.810% and50.223%, but
remain failures. Sum-cube improves observed CE while validation falls from
42.912% to41.318%; the frozen observed-only choice is preserved. All earlier
inputs, proposals and balanced parent reports remain unchanged. Stronger
earlier full-rank division/multiplication and certified absolute-difference
families remain required in the overall comparison.

The search uses11 views,88 sketches,112 seeds and1962 certified axis steps.
All112 refinements stop at a repeated state. Their5017 coordinate states,
799051200 axis-observation comparisons and every failed candidate remain in
the reports. There are zero CE gradients. Construction including all parents
takes864.535 seconds summed over the nine cases; peak run/replay RSS is153796
KiB (150.2MiB). These are nine forward constructions referenced by27 mode
records, not independent full-batch/minibatch/forward optimization runs.
All35 short/long tasks, full batch, minibatch, forward-only and the conventional
transformer comparison remain required. The next Lean-first repair is CM26's
complete-fiber constraint proposal, with its structural hypotheses explicit.

The declared dimensions are `2 <= n <= 257`, `2 <= q <= 771` and
`2 <= vocabulary <= 1024`; base refinement restricts `q <= 257`, at most128
rounds and256 axis updates. Fourier audits stream one matrix at a time. The
conservative single-model/refinement buffer bound stays below1GiB; aggregate
search storage and work use the separate bounds above.
All numerical runs use the user's64GiB process cap:

```sh
cd experiments/convex_stress_tagged
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Source: [`review/flat_spectrum.md`, CM25](../../review/flat_spectrum.md#cm25-categorical-constant-strips-in-an-enlarged-latent-domain).
The model/audit layout and bounded refinement scaffolding are adapted from
`convex_stress_decode` and `convex_stress_density` at `4510ee7d`; the changed
cardinality, noninjective choices and explicit tag lift are described above.
Generated evidence remains ignored and rooted in the main checkout.
