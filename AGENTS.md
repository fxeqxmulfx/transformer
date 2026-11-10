# AGENTS.md

Lean in `src/` formalizes the manuscripts in `papers/` (gitignored; `./run.py papers`
fetches missing ones; read them there, not on the web). The lab in `python/` runs the
experiments in `experiments/`. `./run.py` runs every task.

## Current user override (2026-11-07)

- Do not use Python for future work, including experiments, scripts, wrappers or
  regression suites. Use Rust/Kernite for experiments and native tooling.
- Invoke Lean directly with `LAKE_ARTIFACT_CACHE=false`: `lake build`,
  `lake env lean scripts/Axioms.lean`, and the applicable Lean checks.
- This instruction supersedes the `./run.py`, `uv` and Python-suite requirements
  below. Existing Python sources remain reference material. Keep `INDEX.md`
  generated; use or port a native generator when regeneration is next needed.

## Always

- Work solo: no subagents, no parallel agents of your own; the user's other agents
  share the work only through the board.
- English in code, docstrings, comments, filenames, commits.
- One commit per logical change, as soon as its checks pass.
- Never stage `.discrtree/`, `discrtree.toml`, `.claude/skills/discrtree/`,
  `.kanban/`.

## Generated files (2026-11-08 user override)

- Do not commit reproducible generated outputs: reports, rendered summaries,
  plots, logs, caches, or build artifacts. Keep them ignored and local.
- **Exception: `INDEX.md` must still be generated and committed with every
  `src/` change.** Never edit it by hand.
- Commit the source, generators, configuration, required reproducibility inputs,
  and documented commands needed to regenerate outputs. Keep authored findings
  and interpretation in the experiment README.
- Preserve generated evidence needed for review outside a temporary worktree
  before removing it. This rule supersedes older instructions to commit other
  generated files; it does not supersede the `INDEX.md` requirement.

## Local compute

- An NVIDIA GPU is available: GeForce RTX 3050 Laptop GPU, 4 GiB VRAM,
  verified on 2026-10-19. Use it for suitable experiments.
- A sandboxed `nvidia-smi` can fail to communicate with the driver. Check
  with an authorized elevated query before concluding that CUDA is unavailable;
  GPU training may need the same execution context.

## Board

- `agent-kanban` keeps the work in `.kanban/board.db` at the root, untracked: one
  task per `sorry` of `INDEX.md`. Run `agent-kanban guide` first.
- Register with `agent-kanban agent register <name> --role developer|reviewer`;
  pass `--agent <name>` to the claim, release, submit and review commands.
- The roles are the board's, not a fixed split of agents: an agent need not be
  only a developer or only a reviewer. Do whichever the board needs, developing
  and reviewing in one session when both are open, with a registered name per
  role if the board requires it. An agent may review its own submission.
- Take a `sorry` only by claiming its task: `claim-next`, or `claim ID`. After an
  hour another agent may take it over: claim again to renew, `release ID` to give up.
- Each agent works in its own worktree, made when it starts and removed with its
  branch when it stops: `git worktree add -b <name> .claude/worktrees/<name> main`.
  One that trains nothing may leave out the large archived outputs: add
  `--no-checkout`, then run
  `git sparse-checkout set --no-cone '/*' '!/experiments/archive/'` and
  `git checkout` in it. Keep active experiment sources local: Python tests import
  them. Do not replace `experiments/` with a link to another checkout.
  `agent-kanban` and `dt` find their files from there.
- A worktree's ignored files die with it: link its `papers`, `.lake/packages` and
  the `runs` of every experiment it trains to the root's. Its first `lake build`
  links into `.lake/build`, read-only, every module that Lake's artifact cache
  (`lakefile.toml`) holds: all of `main`, in seconds. `lake` runs with
  `LAKE_ARTIFACT_CACHE=false` (`~/.profile`, `run.py`), which keeps Mathlib out of
  the cache. `lake update`, `lake clean`, `lake exe cache` and a build with a
  `lake-manifest.json` other than the root's change the shared packages: never in a
  worktree. A Mathlib bump is made in the root while none exists, after
  `lake cache clean` and `chmod -R u+w .lake/packages`.
- The cache keeps every module version ever built, a removed worktree's too. When
  the disk runs short and no `lake` runs: `lake cache clean`, then `lake build` in
  the root to refill it.
- Land each commit on `main` at once: `git rebase main` (regenerate a conflicting
  `INDEX.md`), rerun the checks if `main` had moved, and fast-forward the root:
  `git -C <root> merge --ff-only <name>`. The root holds no uncommitted work.
- Done: once landed, `submit-review ID` with `--pass` or `--fail IDX EVIDENCE` for
  every test; evidence cites the commit and the command output. Fix a reviewer's
  `changes` in a new commit and submit again.
- Review: rerun every test at the submitted commit; `approve ID` if all hold, else
  `request-changes ID --notes`.
- A `sorry` entering `INDEX.md` gets its task when committed: tags `sorry`, arXiv id,
  namespace (`open-problem` and priority `low` if the source leaves it open); tests:
  module build, `#print axioms`, statement beside the source, pre-commit checks;
  `--after` every task whose result the source's proof uses.

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
- `./run.py audit`: 0 `rests`, 0 extra axioms (only `propext`, `Classical.choice`,
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
  `./run.py audit`, `./run.py index`, `./run.py forbidden`.
  `./run.py verify` runs these mandatory checks and the Python suite.
- Strict lint is optional by the user's later instruction on 2026-10-19 and
  does not run in `verify`. `./run.py lint` retains the manual command
  `lake --wfail lint --builtin-lint --linters=linter.all --code-quality`.
  Its previous failures remain recorded in `review/dependency_update.md`.
- `./run.py kernel` (`lake check`) is optional while the existing 39 unfinished
  statements remain. The user explicitly retained them on 2026-10-19; do not
  remove them or present them as proved to obtain a green kernel check.
- A Mathlib bump fixes its deprecations in the same commit.

## Python

- Experiment: `experiments/<name>/experiment.py` and `README.md`, and a row in
  `experiments/README.md`. The file: `from lab.dsl import *`, then
  `experiments = {label: Experiment(...)}`; variants by `swap(base, "path", value)`,
  `substitute(base, Block, block)`, `grid`, never by flags. `./run.py blocks` lists
  the words.
- README: the question, a table of how the runs differ, what they found.
- `./run.py check experiments/<name>`, `show … <label>`, `run … [labels]`,
  `report … [labels]`, `profile … <label>`. `run` trains into `runs/<label>/` beside
  the file (gitignored) and resumes on rerun; `report` prints JSON; `profile` trains
  one run afresh under Scalene.
- `EXPERIMENT_PLAN.md` is the plan; obey its status.
- New block: spec in `python/src/lab/domain/`, PyTorch in `infrastructure/`, word in
  `dsl.py`, test in `python/tests/`. Layers domain → application → infrastructure →
  interfaces import only their own or earlier ones; third-party packages only in the
  last two.
- A block's docstring cites its source (paper §, Lean declaration, ported module and
  commit) and every deviation from it.
- uv only through `./run.py` or plain `uv run --locked` inside `python/`; never
  `--active`, never at the root.
- Before committing: `./run.py test`.
- Historical numerical references are bundled in `python/tests/references/sources/`;
  measured benchmark sources are frozen in `scripts/fixtures/optimizer_benchmark/`.
  `./run.py references` checks their hashes without Git or another checkout.
  Only an explicit `./run.py references --rebuild` needs the pinned Git history;
  never regenerate references from the implementation under test.
