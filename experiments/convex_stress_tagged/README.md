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

Eight tests pass, including all 512 binary three-by-three target tables. A
masked four-category fixture with three latent states is repaired to the full
correct table through repeated phase assignments. The Lean three-category
constant-strip witness is checked, including the strip intersection and all
six actual latent frequencies. Tests retain genuine failure and ambiguity:
uncovered latent states are refused, and two fully covered exact observed fits
satisfy the spectral bounds but disagree on an unseen input row. Input, state,
minimum-certificate and work-ledger tampering is rejected.

This is a tested construction/refinement core. Generic bounded flag and phase
discovery, its original-data harness and a separately frozen original follow-up
remain to be implemented in task 219. The balanced parent still passes3/9.
All35 short/long tasks, full batch, minibatch, forward-only and the conventional
transformer comparison remain required; fixture success is no original score.

The declared dimensions are `2 <= n <= 257`, `2 <= q <= 771` and
`2 <= vocabulary <= 1024`; base refinement restricts `q <= 257`, at most128
rounds and256 axis updates. Fourier audits stream one matrix at a time. The
conservative single-model/refinement buffer bound stays below1GiB. A future
multi-candidate search must separately bound its aggregate storage and work.
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
