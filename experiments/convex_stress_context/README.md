# Fixed context interpolation for raw labels

Can a fixed input neighborhood resolve raw aliases left ambiguous by a learned
latent code, while retaining the existing convex complex head and ordinary CE?
CM42 was proved and reviewed first at `81f4e46e`. Its Lean modules are
`ConvexDepth/ContextInterpolation/{Basic,Loss,Batch,Witness}`. They establish
the actual feature/logit identity, the unchanged coefficient bound, Jensen's
raw-CE bound and exact original-batch objective and derivative identities.
Local raw-label information remains an explicit additional premise.

The existing kernel head maps every matched unseen input to one default raw
label. This component instead averages the actual complex feature rows of
nearby observed inputs in the same latent block when that block contains
multiple observed raw labels. The complete original real/imaginary output
tensor is used in the finite dot product. Observed inputs, unique-label blocks,
unmatched cells and unknown axes retain their original logits exactly.

| Choice | Construction and use |
| --- | --- |
| Metric | Squared Euclidean distance between the two original token IDs |
| Bank | Original default, then uniform 1, 3, 5 or 9 nearest observed states |
| Distance ties | Ascending original cell ID |
| Selection | Minimum canonical raw CE over original training record IDs, excluding the queried input state from its neighbor bank |
| Canonical head | Margin 24 on each recorded raw label, independent of trained coefficients |
| Objective ties | Retain the earlier bank entry unless CE improves by more than `1e-10` |
| Full, mini, forward | Same fixed choice, same stored source tensors and original updates; zero additional optimization |

The ordinal metric is declared extra structure. It is not invariant under
arbitrary token renaming and it is not an arithmetic answer function. Selection
uses all original training record multiplicities. The complete latent discovery
is already fitted to that training data, so the selection diagnostic is not a
leave-one-out refit of discovery. It excludes the queried source state but does
not certify generalization. No held-out label or trained coefficient enters
the neighbor-size selection. The canonical vote formula is CM42's actual
conjugate-head identity; predictions use the saved head's real complex products.

The serialized model binds source coefficients, original records and the full
protocol. Loading reconstructs selection, all neighbor rows and resource costs;
altered bindings, records, weights or budgets are rejected. The caller must
provide the externally pinned source hash. Importing a benchmark endpoint also
requires its original discovery/training evidence and frozen artifact pins.
The source's full coefficient budget remains charged, with additional buffer
and scalar-work forecasts for interpolation. Run under the user's 64 GiB cap.

The native tests use a small observed grid with two raw aliases and three
unseen inputs. They compare actual feature products with weighted source
logits, check the raw-CE Jensen bound, verify both real and imaginary derivatives
numerically, retain repeated records and a short minibatch tail in all three
original modes, and test serialization/domain/binding failures. These fixtures
are not an original stress-suite result. The reviewed 35/35 endpoints and all
older comparison runs remain unchanged.

Reproduce with native tooling from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_context/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_context/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_context/Cargo.toml --all-targets -- -D warnings
```

Original-data evaluation needs a separate prospective adapter and protocol.
Report raw accuracy and ordinary CE alongside the original success criterion;
passing that criterion is still success.
