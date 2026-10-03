# AGENTS.md

Repository instructions for Codex. `CLAUDE.md` contains the fuller rationale and
examples behind these rules.

The current experimental research plan and the state of the last run are in the
root [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). The user paused the experimental
cycle on 2026-10-03; resume it only after an explicit instruction to continue.

## Paper fidelity

- Formalize the manuscripts under `papers/` in Lean files under `src/`. The
  papers are gitignored; `INDEX.md` is the generated inventory of declarations
  and proof debt.
- The papers and reference source files for this project are available locally
  under `papers/`. Search and read those local files when checking a statement
  or proof; do not browse the web for their contents.
- Recheck every theorem against its paper before calling it complete or making
  a requested commit: hypotheses, quantifiers, constants, indices, and
  conclusion. A closing Lean proof does not establish fidelity to the paper.
- If a paper statement needs a fixable correction, state the corrected theorem
  and document the source wording and correction in its docstring. If the claim
  is false, prove a counterexample and identify the refuted claim in the
  docstring. Never weaken a claim while retaining the paper's name.
- Cite the paper section or equation in every statement's docstring. Write
  declarations, docstrings, comments, filenames, and commit messages in English.

## Statements and proof integrity

- Represent every paper claim, including conjectures and open problems, as a
  `theorem`; use `sorry` for an unproved claim. Do not hide claims in a
  `def _ : Prop`, `axiom`, or structure field. A genuine predicate of its
  arguments may be a `def _ : Prop`.
- For each theorem with hypotheses, include an `example` showing that the
  hypotheses are satisfiable. Delete unused binders or fix the incomplete
  proof; do not silence them by renaming to `_h`.
- Do not use decorative proofs or definitions: conclusions of `True`, constant
  bodies that ignore substantive arguments, placeholder `Prop` definitions,
  or `sorry` outside proof position. A substantive theorem closed by `rfl`,
  `trivial`, or one-line `simp` warrants checking its definitions.
- Do not add `axiom`, `native_decide`, or `set_option linter.* false`.
  `autoImplicit` is disabled in `lakefile.toml`; declare variables explicitly.
  Use explicit hypotheses for assumptions instead of axioms.
- A theorem using a sorried theorem can appear proved in `INDEX.md`. Run
  `lake env lean scripts/Axioms.lean` to audit the full tree. `rests` (proved
  declarations depending on `sorryAx`) and extra `axiom` must both be zero;
  only `propext`, `Classical.choice`, and `Quot.sound` are accepted. Carry an
  unproved premise as an explicit hypothesis rather than hiding a dependency.
- Checking every external-library declaration used by a proof is mandatory,
  including declarations from Mathlib. Inspect its statement, hypotheses,
  relevant definitions, typeclass assumptions, and transitive proof
  dependencies. Verify that it expresses the intended mathematical claim and
  contains no placeholders, vacuous reformulations, or hidden unproved claims.
  Audit its transitive axioms with `#print axioms` and the full-tree audit;
  only `propext`, `Classical.choice`, and `Quot.sound` are accepted. Successful
  import, compilation, or a library's own claim of verification is insufficient.
- The `sorry` count must not increase except when adding a new paper's
  statements under `src/` and importing its aggregator from
  `src/Transformer.lean` in the same change. Vacuous statements and
  placeholder definitions must not increase. Read the current counts from
  `INDEX.md`, not from a hard-coded number. A Lean file without `sorry`
  should have no warnings.

## Modules and navigation

- Keep Lean files around 150–200 lines; split at about 200 unless a proof is
  indivisible. Paper formalizations mirror manuscript sections as
  `Section*_*.lean`.
- For other subjects, organize by import depth rather than declaration kind:
  `Foo/Defs.lean` (definitions) → `Foo/Basic.lean` (immediate API) →
  `Foo/<Topic>.lean` (theorems) → `Foo.lean` (aggregator with imports and
  docstring, no proofs). Merge `Defs` into `Basic` while small.
- Use namespace `Transformer.` followed by the file's directory path. Root
  files use their own name (`XSA.lean` → `Transformer.XSA`); `Basic.lean` uses
  `Transformer`. Every module must be reachable from `src/Transformer.lean`
  so that `lake build` covers it.
- Python experiments live in `python/` (the `lab` package, see
  `python/README.md`); `experiments/` holds a folder per experiment, its
  `experiment.py` beside a `README.md`, and in `archive/` the records of
  past runs. `./make.py` runs every task, Lean and Python.
- Read `INDEX.md` to locate declarations. Regenerate it with
  `python3 scripts/index.py` after any change under `src/`, and include it in
  the same requested commit as the source change. Do not edit it by hand.
- Use `dt find`, `dt show`, and `dt deps` to search Mathlib declarations when
  a lemma is hard to name or may already exist. Do not grep Mathlib sources
  first. `.discrtree/`, `discrtree.toml`, and the discrtree skill belong to a
  separate line of work; do not stage them for Lean changes.
- Search [Reservoir](https://reservoir.lean-lang.org/) for Lean packages when
  useful results are unavailable in the current dependencies. Additional
  packages needed for proofs may be installed. Check compatibility with the
  project's Lean and Mathlib versions. The mandatory checks for every imported
  result and its proof dependencies apply before using a package's results.
- Keep external dependencies to a minimum. When practical, prefer copying only
  the necessary definitions and proofs into this repository over adding a whole
  package. Preserve source attribution and license notices. Copied code must
  pass the same statement, definition, assumption, and proof-dependency checks
  as imported library results.

## Checks and workflow

- Work directly in one session; do not use subagents or parallel agent work.
- During Lean development, build the affected module with
  `lake build Transformer.X`. Before finishing a Lean source change, run
  `lake build`, `lake env lean scripts/Axioms.lean`, and
  `python3 scripts/index.py`; inspect the resulting counts and warnings. Use
  `#print axioms F` for an individual theorem when needed.
- Search `src/` for forbidden constructs such as `native_decide`, `axiom`,
  and disabled linter options. Fix Mathlib deprecations in the change that
  updates Mathlib.
- Commit completed, verified logical changes regularly during task work.
  Do not wait until an entire long-running experiment campaign is finished.
  Keep one logical change per commit, run the checks appropriate to that
  change, and recheck any changed theorem statements against the papers first.
  Leave unfinished work and unrelated changes out of each commit.
