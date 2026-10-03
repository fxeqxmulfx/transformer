# AGENTS.md

Lean in `src/` formalizes the manuscripts in `papers/` (gitignored; `./make.py papers`
fetches missing ones; read them there, not on the web). The lab in `python/` runs the
experiments in `experiments/`. `./make.py` runs every task.

## Always

- Work solo: no subagents, no parallel agents.
- English in code, docstrings, comments, filenames, commits.
- One commit per logical change, as soon as its checks pass.
- Never stage `.discrtree/`, `discrtree.toml`, `.claude/skills/discrtree/`.

## Lean

- Every paper claim, open ones included, is a `theorem`; unproved = `sorry` in proof
  position. Never a claim as `def _ : Prop`, `axiom` or structure field; a
  `def _ : Prop` is only a predicate of its arguments.
- Before each commit, reread every touched statement beside its source: hypotheses,
  quantifiers, constants, indices, conclusion. Docstring cites paper and § or equation.
- Source fixable (missing hypothesis, wrong constant, index): prove the corrected
  statement; docstring gives the source's wording and the change. Source false: prove
  a counterexample; docstring names the claim. Never weaken a statement under the
  paper's name.
- Every theorem with hypotheses gets an `example` satisfying them.
- Stuck: compare the statement and its definitions with the paper before stronger
  tactics. Never move a difficulty into a definition.
- Forbidden: `axiom`, `native_decide`, `set_option linter.* false`, `autoImplicit`, a
  conclusion `True`, a definition ignoring an argument, an unused binder renamed `_h`
  (delete it). `rfl`/`trivial`/one-line `simp` closing a substantive theorem: suspect
  the definitions. A file without `sorry` has no warnings.
- `sorry` in `INDEX.md` only falls, except for a new paper's statements entering
  `src/Transformer.lean` in that commit. `vacuous` and `placeholder` stay 0.
- `./make.py audit`: 0 `rests`, 0 extra axioms (only `propext`, `Classical.choice`,
  `Quot.sound`). A result needing an unproved input takes it as a hypothesis.
- Read the statement, hypotheses and instances of every library lemma used. Find
  lemmas with `dt find '<pattern>'`, `dt find --name <part>`, `dt show <name>`, not
  grep or memory. A Reservoir package only if compatible; prefer copying the needed
  proofs, attributed.
- A paper: `Section<§>_<Topic>.lean` files mirroring the manuscript. Otherwise by
  import depth, never by declaration kind: `Foo/Defs.lean` → `Foo/Basic.lean` →
  `Foo/<Topic>.lean` → `Foo.lean` (imports and docstring only).
- Namespace `Transformer.` + directory: `Transformer/Metastability/Staircase.lean` →
  `Transformer.Metastability`; `Transformer/XSA.lean` → `Transformer.XSA`;
  `Transformer/Basic.lean` → `Transformer`.
- Files of 150–200 lines. Every module reachable from `src/Transformer.lean`.
- `INDEX.md` locates declarations. Never edit it; regenerate it in the commit of
  every `src/` change.
- Iterate with `lake build Transformer.<Module>`. Before committing: `lake build`,
  `./make.py audit`, `./make.py index`, `./make.py forbidden`.
- A Mathlib bump fixes its deprecations in the same commit.

## Python

- Experiment: `experiments/<name>/experiment.py` and `README.md`, and a row in
  `experiments/README.md`. The file: `from lab.dsl import *`, then
  `experiments = {label: Experiment(...)}`; variants by `swap(base, "path", value)`,
  `substitute(base, Block, block)`, `grid`, never by flags. `./make.py blocks` lists
  the words.
- README: the question, a table of how the runs differ, what they found.
- `./make.py check experiments/<name>`, `show … <label>`, `run … [labels]`,
  `report … [labels]`. `run` trains into `runs/<label>/` beside the file (gitignored)
  and resumes on rerun; `report` prints JSON.
- `EXPERIMENT_PLAN.md` is the plan; obey its status.
- New block: spec in `python/src/lab/domain/`, PyTorch in `infrastructure/`, word in
  `dsl.py`, test in `python/tests/`. Layers domain → application → infrastructure →
  interfaces import only their own or earlier ones; third-party packages only in the
  last two.
- A block's docstring cites its source (paper §, Lean declaration, ported module and
  commit) and every deviation from it.
- uv only through `./make.py` or plain `uv run --locked` inside `python/`; never
  `--active`, never at the root.
- Before committing: `./make.py test`.
