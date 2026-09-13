# Conventions

- English only: declarations, docstrings, comments, commit messages, filenames.
- Lean files ~150-200 lines; split at ~200 unless it is one indivisible proof.
  (File = compilation unit: a big file is slow to elaborate and blocks everything
  downstream.)

## Working agreement

- **Work solo.** No subagents, no workflows, no parallel fan-out — one session,
  one line of work, every file read and edited directly.
- **Commit every change.** One commit per logical change, made as soon as
  `lake build` is green for it. Never batch unrelated edits into one commit, and
  never leave finished work uncommitted.

## Layout

`src/` Lean (lakefile `srcDir`) · `reference/` Python, not built ·
`papers/`, `transformer-vm/` gitignored.

Paper formalizations mirror the manuscript (`Section*_*.lean`). Everything else:
subject dir layered **by import depth, never by declaration kind** —

`Foo/Defs.lean` (defs, notation, minimal imports) → `Foo/Basic.lean` (simp lemmas,
immediate API) → `Foo/<Topic>.lean` (hard theorems, one topic each) → `Foo.lean`
(aggregator: imports + docstring, no proofs). Merge `Defs` into `Basic` while small.

Grouping axioms/lemmas/theorems into separate files is an anti-pattern: it splits a
def from its API, forces consumers to import all layers, and cycles as soon as two
subjects interact.

Every module must be reachable from `src/Transformer.lean` or it is not built.

## No decorative proofs

Forbidden:

- `def _ : Prop := True` / `∀ _, True` — invisible to the sorry count and to
  `#print axioms`; write `:= sorry`.
- Concluding `True`.
- A def that ignores any argument (`:= 0`, `:= ∅`, `:= ⊥` under a substantive
  docstring); write `:= sorry`.
- `sorry` outside proof position.
- `set_option linter.* false`.
- `native_decide` (adds `Lean.ofReduceBool`); plain `decide` is fine.
- `axiom` — bundle assumptions as a `structure`/`class` or section `variable`s so
  they stay hypotheses.
- `autoImplicit` — off in `lakefile.toml`; keep it off, declare every variable.

Required:

- Every theorem with hypotheses: an `example` witnessing that they are satisfiable
  (contradictory hypotheses make a theorem provable and worthless).
- Every statement: source cited in its docstring (paper, § or equation).
- Unused binder → delete it, or fix the `sorry` that has not reached it. Never
  rename to `_h`, never silence. Invariant: **no `sorry` in a file ⇒ no warnings
  in it.**
- `rfl` / `trivial` / one-line `simp` closing a substantive theorem ⇒ suspect a
  placeholder definition beneath it.

## Build

```
lake build [Transformer.X]
#print axioms F                  -- expect [propext, Classical.choice, Quot.sound]
grep -rn ':= True\|: True :=\|native_decide\|^axiom \|linter\..* false' src
```

Mathlib-bump deprecations are fixed in the same change as the bump.
