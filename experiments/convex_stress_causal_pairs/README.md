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

The native component implements categorical search, actual complex features
and full-batch, minibatch and forward-only readouts. No original task is
evaluated here; the [complete comparison adapter](cohort/README.md) passes
eight synthetic checks and awaits completion/review of its quotient parent.
Ordinary raw labels remain unchanged. An observation contains a
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

For a selected query with `m` observed values, the actual saved code is the
normalized complex DFT on `m+1` categorical states. The last state represents
unknown values and initially chooses class zero; an absent query has zero
logits. A separate mask distinguishes observed feature coverage from either
fallback. Coverage of a selected feature is not proof that this was the right
feature for an unseen prefix. The code is numerically audited by complex SVD;
flatness refers to this fixed basis, not the empirical target matrix.

The real logits are `Re(sum_f H[f,c] phi[s,f])` over the original vocabulary.
The trainable tensor contains both real and imaginary head coordinates, with
per-class L1 radius `64 * total_states`. CD8 makes this a convex fixed-feature
domain with ordinary CE. Discrete feature discovery is a separate stage. The
forward initializer applies the conjugate Fourier transform to observed-state
logits of 24 at each recorded raw label, and zero at other classes. It uses no
CE derivative. An independent inversion check reads every saved state.

Lean CFt5 now supplies the exact bridge in `ConvexDepth/ComplexHead.lean`,
`ComplexHeadBudget.lean` and `FourierCode.lean`: both actual complex
coordinates represent the full real head domain, conjugate initialization
reproduces the assigned logits and ordinary CE, and every initialized class
column fits the chosen radius. It proves the flat normalized Fourier basis
for any nonzero state count, including unequal query blocks. These 28
theorems also bound the logits of every feasible learned head. They certify
the recorded assignments and explicit default; they do not identify unseen
labels or prove that feature discovery is jointly convex.

Full-batch training performs 16 projected CE steps at learning rate 1;
minibatch performs 256 cyclic steps of 64 observations at rate 0.1. Forward
keeps the initializer unchanged. Each iterative step retains pre/post batch
CE and derivative norm; full-data CE is computed at both endpoints, avoiding
a full rescore at every minibatch. Iterative mode refusals retain the other
available modes. Work includes separate derivative examples, derivative terms,
score-only examples, coefficient updates and projection columns. Independent
replay uses complex multiplication, a different accumulation order and
bisection projection; exact producer replay additionally checks saved bits.

The head caps total states at 65,536, complex code entries at four million and
real coefficients at 16 million. Construction/inversion, original-data audit
and per-mode training have explicit work forecasts; an audit or training
forecast above two billion scalar operations is retained as a refusal. These
are declared estimates, not measured FLOP counts. Complete replay/serialization
buffers are forecast before discovery, with a 56 GiB upper gate inside the
64 GiB process limit. The source adapter retains measured time and RSS.

Fifteen native tests pass. The four-class Boolean-pair fixture eliminates every
individual feature and learns a lag pair. The position-shift fixture selects
presence and keeps its answer after the input positions change. Exhaustive
short-prefix tests compare cached features with an independent scalar
interpretation. Further tests check full lazy catalogue coverage, original
label contradictions, unknown pair values, serialization, caps, contradictory
duplicate observations and changed state/work/ordering evidence. Deterministic
replay checks the entire trace; a separate checker reads original prefixes to
validate every retained contradiction and every selected decoder's raw labels.
Complex tests check Fourier inversion, singular values, full-vocabulary CE,
both coefficient components, longer-prefix presence invariance and unknown
masks. Finite differences check every real/imaginary derivative away from the
interpolating initializer; weighted minibatches reproduce the full gradient.
Actual updates reduce CE and replay independently, while forward has zero
gradient/update counts. A full-batch work refusal preserves minibatch and
forward outcomes. Two flat, training-consistent heads also disagree on a new
prefix whose feature values were both observed, retaining the CFt4 boundary.

```sh
cd experiments/convex_stress_causal_pairs
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked -- --test-threads=1
cargo clippy --locked --all-targets -- -D warnings
```

The complete parent-preserving evaluation adapter is implemented. Task234
remains open until its parent gate and independent review are satisfied.
A separately reviewed and
frozen original cohort will retain task232, both initializer arms, all 35 tasks,
the transformer and every earlier cost and failure. No original pass is inferred
from these fixtures. Generated evidence remains ignored and local.
