# Generic causal pairs with an explicit invariant prior

Can a forward scan learn a decoder of two causal features when no single
feature explains the observed labels, and can a declared preference for
position invariance repair a spurious positional choice?

Lean CFt4 at `7126517f`, in `ConvexChannel/CausalPairs.lean`, proves the paired
representation, actual forward existence and conditional prediction from
coverage of the selected pair. It also proves presence-feature permutation
invariance with the query fixed, and exhibits two candidate orders that fit
all observations but disagree on an unseen input. Candidate priority is a
structural prior; consistency and flatness alone do not justify it.

The first native component implements only the categorical search. No original
task is evaluated here, and the complex readout/training adapter is still being
implemented. Ordinary raw labels remain unchanged. An observation contains a
visible token prefix and its supervised label; its final token is the query.
There is no task identifier or target formula in the extractor or request.

| Stage | Fixed rule |
| --- | --- |
| Features | Token presence, token counts, length, relative lags, then latest completed successors at offsets one and two |
| Singles | Presence first, then the remaining feature order |
| Pairs | Lag pairs by increasing maximum lag, then every other unordered pair; streamed without a pair-by-example tensor |
| Decoder | Record each observed feature value and raw label; reject a candidate on the first conflicting raw label |
| Choice | First surviving candidate per observed query; no held-out data used |
| Unknown values | No recorded label; the discovery decoder returns `None` |
| Refusals | Cache, trace, visit and state caps remain explicit, including partial query results |

The defaults cap the feature cache at 32 million cells, the search at 200
million observation visits, the retained candidate trace at 250,000 attempts
and each query's state dictionary at 512 observed values. Inputs have at most
1024 tokens in their vocabulary, context 4096, 200,000 supervised prefixes and
20 million stored prefix tokens. The memory forecast includes input copies,
cache, traces and serialization. The complete process retains the user's
64 GiB address-space cap. These are bounded proposals, not a completeness
claim about all multi-feature rules.

Five native tests pass. The four-class Boolean-pair fixture eliminates every
individual feature and learns a lag pair. The position-shift fixture selects
presence and keeps its answer after the input positions change. Exhaustive
short-prefix tests compare cached features with an independent scalar
interpretation. Further tests check full lazy catalogue coverage, original
label contradictions, unknown pair values, serialization, caps, contradictory
duplicate observations and changed state/work/ordering evidence. Deterministic
replay checks the entire trace; a separate checker reads original prefixes to
validate every retained contradiction and every selected decoder's raw labels.

```sh
cd experiments/convex_stress_causal_pairs
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked -- --test-threads=1
cargo clippy --locked --all-targets -- -D warnings
```

Task234 still requires actual complex feature/readout binding, full-batch,
minibatch and forward-only endpoints, ordinary full-vocabulary CE checks and
the complete parent-preserving evaluation adapter. A separately reviewed and
frozen original cohort will retain task232, both initializer arms, all 35 tasks,
the transformer and every earlier cost and failure. No original pass is inferred
from these fixtures. Generated evidence remains ignored and local.
