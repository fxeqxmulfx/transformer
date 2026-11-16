# Causal features for the complete convex complex model

Can a generic fixed family of causal tuple and relation features extend the
complete convex decoder to more original GPT mechanism probes? This component
connects the input-only boundary of `convex_stress_probe_data` to the full CM70
model. Source formulas, task identities, numeric teachers, evaluation labels and
numeric output codecs are absent from the model boundary. The feature family is
an explicit structural prior. A flat-spectrum bound does not supply this prior
or guarantee unseen answers.

The first implementation has no original-data production entry point or new
accuracy result. The prospective comparison must include all six source cases,
retain the unresolved routing teacher refusal, freeze the producer, seal every
training endpoint before heldout scoring, and independently repeat and replay
the full cohort. The three older probe controls remain in that comparison.
Raw-token99%, exact-sequence99% and the original numeric MSE criterion must be
reported separately. Synthetic controls cannot add stress-test coverage.

| Part | Fixed behavior |
| --- | --- |
| Aligned tuples |144 programs:1–8 chunks, contiguous or interleaved layout, no context or one of the first8 input positions |
| Dense features |8 token lags;16 first/latest successor walks with1–2 hops and offsets1–4;6 first/last or last-two relations; length and constant |
| Input alphabet |Sorted training input symbols only; separated rows exclude teacher-forced answers, dense rows stop at their final supervised prefix |
| Finite encoding |512 states for every retained program; undefined0, end1, categorical signatures2–510, padding511 retained |
| Signature |Alphabet of at most3 symbols uses categorical indices plus an unseen-symbol digit; otherwise raw token IDs. Base is alphabet size+1 or vocabulary+1; fold modulo509 |
| Pruning |Only an actual same-state/different-label training conflict; two original supervised IDs witness each rejection |
| Retention |Every compatible program, including duplicate functions; every original row and supervised position, with multiplicity and order |
| Decoder |All512 states and every raw class, including arbitrary nonuniform unknown completions; no support pruning or learned discrete program selection |
| Forward |Zero updates, initial total mass8, ordinary CE in original127-record minibatches with the actual global denominator |
| Mini |One epoch of actual projected ordinary CE, batch127, learning rate1000000, original final tail |
| Complex head |CM70 full block-Fourier coefficients, complete products, inverse and column-L1 audits; total mass in[1,64] |
| Admission |Bounded original input and observation counts; caller must reserve full owning capacities and other live buffers; nested CM70 budgets include model inputs and updates |

The public input codec recognizes BOS1 and SEP18. A chunk feature takes the
output position from the visible prefix length and learns its end-state raw
label from training. There is no hard-coded EOS answer. Undefined layouts and
missing context are regular states rather than reasons to discard a program.
The 512-state map is fixed before labels; no apparently unreachable state is
removed. Hash collisions and unknown states remain possible. Relation features
compare input token IDs; learned raw labels are not identified with comparison
values. Successor walks only inspect completed earlier token pairs inside the
visible prefix.

The independent interpreter uses a separate tuple-selection loop and reversed
successor scan. Preparation compares both interpreters for every program on
every original supervised prefix, including rows after the first conflict.
The retained dictionary is then reconstructed from all original anchors by
the CM70 trainer. Prediction contracts the complete complex head; its separate
coverage diagnostic cannot replace the model output.

## Lean contract checked before implementation

At `2e9c1bb9`, the current manuscript remained at CM70. The following declarations
in `Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent` were reread
beside the manuscript and their module and axiom checks passed before native
implementation:

- `cone_convex` and `domain_convex`: joint nonnegative program masses and decoder
  coefficients, each decoder row summing to its program mass, original observed
  one-hot anchors and an explicit total-mass interval. Independently learned
  factors are not multiplied together.
- `retained_domain_eq`: an inconsistent program has exactly zero coefficients;
  restriction to every consistent program preserves the entire feasible set.
- `head_budget`, `ce_convex` and `batchLoss_convex`: a fixed unitary source,
  complete complex head, column budget2R·|S| and ordinary CE on the actual
  indexed observations. The new family is simply a finite choice of the theorem's
  arbitrary fixed input-to-state programs.
- `accepted_logits` and `accepted_strict`: every retained program must reach
  an observed state with the same label; positive total mass then gives the
  model winner. Uncovered or disagreeing states are not identified.
- `target_logits`: a correct unseen target additionally requires target
  realizability in the fixed bank and correct observed labels. Neither a
  spectral bound nor passing training examples establishes that hypothesis.

CM70 `RepeatedRows` proves exact encoding, complete-domain coverage, complex
logit/CE preservation and projection-distance equality. The native component
uses its already-tested full arbitrary-row fallback; no extra uniformity
constraint is introduced. Floating-point projection and finite observations
do not certify exact real projection, a learning convergence rate or target
realizability. No new mathematical statement is needed merely to instantiate
these arbitrary finite-program results with the declared family.

Eight native controls pass in default and release profiles, with strict Clippy.
Both interpreters agree on all176 programs across varied visible prefixes;
hand-computed examples check chunk alignment, context, end states and successor
walks independently. The130-record ordering control retains all four compatible
relation programs, including duplicate functions, and all2040 unknown rows.
Forward makes no updates; mini executes the127-record batch and three-record
tail, changes parameters and lowers CE. Every logical coefficient is audited,
and complete checkpoints preserve all parameter words. A padding-row control
accepts a nonuniform unknown completion and produces a nonzero imaginary
coefficient. Contradictory duplicate prefixes empty the entire family; invalid
IDs, tokens, missing memory reservations and exhausted work are rejected.

The full Lean build also passes:13995 jobs,36 existing direct sorries, zero
results resting on a sorry and zero extra axioms. The unchanged native index
checks4898 modules and25678 theorems with zero vacuous or placeholder entries.
These checks establish the component contract, not original-probe success.

Reproduce the native controls without Python:

```sh
cargo test --offline --locked --manifest-path experiments/convex_stress_probe_model/Cargo.toml
cargo test --offline --locked --release --manifest-path experiments/convex_stress_probe_model/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_probe_model/Cargo.toml --check
cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_stress_probe_model/Cargo.toml -- -D warnings
LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2 lake build Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent Transformer.Grokking.FlatSpectrum.ConvexDepth.RepeatedRows
LAKE_ARTIFACT_CACHE=false lake env lean -j2 -M40960 scripts/BoundedLatentAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean -j2 -M40960 scripts/RepeatedRowAxioms.lean
```

The prospective runner must enforce110000000000 bytes across live processes,
check both disks, stop on shortage, account for input/scorer/checkpoint buffers
and preserve all refusals. A component allocation cap alone is not a process
or whole-cohort admission. Generated outputs remain local and ignored.
