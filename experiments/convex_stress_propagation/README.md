# Finite propagation for general latent decoders

Can categorical training constraints identify an unseen label when its decoder
fiber contains more than two states? Lean CM19 at `0807609b` proves a sufficient
finite procedure before this Rust implementation. It supports any fixed decoder
on `ZMod q`, including even `q`, and a finite bank of decoders on the same group.
The target must belong to that declared phase/decoder family. A fit to the
observations does not establish this assumption for unseen labels.

`propagate(&Problem, Config)` returns either a complete untrusted `Certificate`
or an explicit budget `Refused` outcome. A problem contains only dimensions,
the fixed categorical decoder maps and observed row/column/label triples.
The configuration names an observed anchor and an exact sweep count. No task
name, input arithmetic rule, missing labels, preferred decoder or fitted
phases enter the computation.

Every anchor-fiber state starts a branch with its row fixed to zero. Each
simultaneous sweep retains a phase only if every incident training cell has
a compatible phase in the old neighboring domain. A branch with any empty
domain contributes nothing. Otherwise all Cartesian row/column pairs
contribute decoded labels. The final union includes every branch and every
bank member. Only a singleton certifies an answer; an empty union refutes
this declared family. A nonsingleton is a refusal, not a proof of ambiguity
unless two actual global completions are provided.

`verify` recomputes the procedure using a separate dense Boolean membership
implementation. It checks the canonical observation multiset, full decoder
bank, configuration, every branch, every final phase domain, sweep deletion
counts, empty-domain decisions, output unions and operation counts. Only it
constructs `Verified`, whose `answer` method exposes singleton outputs.
The producer uses sorted state vectors and precomputed incidence lists;
the checker scans observations separately and never calls the producer's sweep.
The Rust code is not formally extracted or verified by Lean.

## Checked cases

| Case | Difference and result |
| --- | --- |
| Lean CM19 witness | Four states, a three-state false fiber, one missing cell: two sweeps certify false while all three hidden states remain possible |
| Shifted two-decoder bank | Six live anchor branches agree on the same missing label |
| No singleton decoder fiber | A four-by-four, five-state table with fibers of sizes three and two; the withheld label is certified, and all 78,125 normalized phase assignments are independently checked |
| Ambiguous observations | Two actual completions give opposite unseen answers; more sweeps retain both |
| Conflicting decoder conclusions | Two individually identifying decoders predict different labels; their bank union refuses to choose one |
| Contradictory observations | Repeated cell with opposite labels produces an empty union after one sweep |
| Budget exhaustion | Insufficient branch, decoder-comparison or memory budget returns no answer sets, including after some branches have finished |
| Altered evidence | Missing branches, changed domains, labels, bank index, sweep counts, costs and output unions are rejected |

Fourteen native tests pass. The independent completion oracle covers every
binary decoder on four states, all 80 nonempty labeled observation patterns
of a two-by-two table and budgets 0–3: 5,120 cases. It also covers every pair
of distinct binary decoders on three states, the same patterns and budgets
0, 1 and 3: 6,720 cases. For every compatible completion it checks retention
of the correctly normalized phases and inclusion of all actual answers.
Every certified singleton must agree with every completion; every empty
union must have no fitting completion. It does not demand completeness.

Additional tests check categorical row/column permutations, arbitrary output
renaming with unused vocabulary classes, the one-state group, reordered
observations, every batch size of the fixture, retained repetitions, exact
serialization and independent replay. Batch tests concern the accumulated
observations; this crate does not implement an optimizer.

## Resources and reproduction

Preflight validates dimensions, vocabulary, decoder arrays, observations and
the anchor before allocating domains or output tables. Checked arithmetic
counts complete branch traces, working domains, decoder/input copies, answer
sets, vector capacity and allocation allowances. The cap is at most 64 GiB.
The comparison budget counts decoder evaluations in support searches and
output unions; it is not a wall-clock bound or a count of loop/index overhead.
Repetitions remain visible in the saved inputs and costs. No branch is omitted
to meet a budget: an unfinished union is never a certificate.

From the repository root, using only native tooling:

```sh
cargo fmt --manifest-path experiments/convex_stress_propagation/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --manifest-path experiments/convex_stress_propagation/Cargo.toml
cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_propagation/Cargo.toml -- -D warnings
```

The source fixtures are the reproducibility inputs. Generated logs and resource
measurements remain in `.kanban/artifacts/207`, outside version control.

CM19 also proves that the resulting answer equalities give a convex restriction
of the spectral decoder hull and preserve its realizable ordinary-CE optimum.
This crate supplies the finite identification/checking component. Connecting
general-fiber constraints to full-batch, minibatch and forward-only spectral
updates, choosing a useful declared bank without task-specific answer rules,
and freezing an original-data comparison remain separate work. There is no
new original stress score, full35 pass or transformer comparison here.
