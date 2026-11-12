# Nonlinear causal paths with a convex output tensor

Can deeper generic causal features repair the remaining depth and recall
failures while ordinary cross-entropy stays convex in the learned coordinates?
Lean comes first: CD5-CD7 preserve complete nonlinear paths; CD8 at `32a9b668`
allows a learned combined path/feature/class tensor under per-class absolute-sum
budgets. The blocks remain fixed. Separate learned routing and decoder factors
would introduce a product and are not covered by that joint-convexity claim.

This directory currently contains the path component and an allocation
preflight. No output head has been trained and no stress accuracy is claimed.
The next stage must implement and freeze the full/mini/forward-only head
comparison and independent-routing/additive baselines before scoring them.
The interrupted earlier 35-task cohort remains unchanged.

| Component | Fixed construction | Information used |
| --- | --- | --- |
| Input views | All symbols, plus every pair of the four most frequent training symbols | Training input tokens only |
| Previous-distinct block | Move the current complex position code to the latest earlier distinct symbol in its view | Visible prefix and previous path state |
| Completed-successor block | Find the latest completed occurrence of the selected symbol and move to its next visible symbol | Visible prefix and previous path state |
| Stream transport | Identity or fixed phase-conjugated normalized Fourier-4 | Layer index only |
| Terminal features | One-hot final symbol and fixed FNV history bucket | Actual executed path history |

Positions are represented by four complex amplitudes at each prefix location,
including an explicit missing-position slot. Routing reads stream energy from
the actual preceding hidden state. A branch clips that state's norm at one,
moves the selected content, and transports its displacement. Its norm is at
most two. The full block computes the transported skip plus that bounded
displacement; the numerical implementation checks its norm on every step.
No block reads a target, task identity or arithmetic operator.

The identity/Fourier pair is a coordinate-invariance control. Both the skip
and the branch displacement are transported, and position readout uses energy;
their exact predictions should agree. This is not the same experiment as
changing only a skip while holding a coordinate-sensitive branch fixed, and
does not claim a Fourier accuracy advantage. The unitary skip is the flat
spectral bound. A bounded feature passed to the CD8 head has a logit bound
from its coefficient budget; no whole-network Jacobian flatness is asserted.

There are at most seven views and `2^depth` paths per view. Depth is capped at
four, prefixes at 1024 tokens, vocabulary at 1024, and history slots at 4096.
The preflight uses the original frozen easy/hard depth, long MQAR and short
binding datasets, checks their hashes through the existing native constructor,
and estimates coefficient, gradient and u32 feature-cache allocations for
depths 0, 1, 2 and 4. Explicit buffers above 2 GiB must be rejected before
allocation. An 8 GiB OS address-space limit must also cover data, libraries,
temporary buffers and serialization. Hash collisions remain part of the model.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexDepth
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexDepthAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_depth/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_depth/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_depth/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_depth/Cargo.toml
bash experiments/convex_stress_hull/limited.sh 8 experiments/convex_stress_depth/target/release/convex-stress-depth prepare
```

Six component checks cover complex norm preservation, genuinely composed
history, causal completed successors, matched-coordinate invariance, early
resource-limit rejection and independence from labels/task names. These are
implementation checks, not benchmark outcomes or a learning convergence proof.

The allocation preflight on the original four datasets completed in 2.26 seconds
at 244.5 MiB peak RSS under the 8 GiB process limit. Depth-4 MQAR would require
4,704,824,320 bytes for the explicit buffers and is rejected by the 2 GiB budget;
its depths 0, 1 and 2 fit. All four depth choices fit for the two depth tasks
and short binding. These are allocation estimates, not measured training peaks.
The preflight, resource log and component-check logs are retained outside the
worktree in `.kanban/artifacts/175/`.
