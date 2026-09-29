# Next task: prove the proxy Kac–Rice bound on `T'`

Prove `Transformer.Modes.main_int_phi_T'` in
`src/Transformer/Modes/Section2_MainIntPhi.lean`. It is the second bound of
`lem:main-int-phi` in `papers/arXiv-2412.09080v3/paper.tex`, §2.3: the proxy
Kac–Rice integral over `T'` is finite and is `O(√β)` throughout the stated
regime. The current baseline in `INDEX.md` is 175 theorems using `sorry`.

Before editing, compare the Lean statement with the paper: the regime,
definition of `T'`, integrand, finiteness, and asymptotic bound must match.
Preserve the theorem's mathematical content. If it is false, prove a concrete
counterexample and document the source wording and correction in a docstring
as required by `AGENTS.md`.

Useful starting points:

- `sq_le_phiRate` and `integral_exp_phiRate_T'` in the target file already prove
  the Gaussian bound for the model rate on `T'`.
- `Section2_ProxyKRInner.lean` converts the inner `lintegral` to a real Gaussian
  moment. `Section2_ProxyKRScale.lean` controls the determinant and curvature
  factor on `T`.
- `Section2_PhiTDelta.lean` explains why `int_phi_final` requires the extra
  assumption `n = O(β^(5/2))`. `IsRegime` does not supply that assumption for
  every `c > 0`. Handle the Gaussian shift directly on `T'`; do not silently
  import that assumption or use a sorried theorem.
- Check the import graph before moving helpers: `Section2_IntPhiB.lean`
  imports the target file. Split the target module if the proof makes it much
  longer than the repository's 150–200 line guideline.

Done when the target theorem has a genuine proof with no new `sorry`, its
statement and docstring have been checked against §2.3, and the proof tree has
zero declarations resting on `sorryAx` and zero extra axioms. Build the
affected module, then run `lake build`, `lake env lean scripts/Axioms.lean`,
and `python3 scripts/index.py`. Inspect warnings and the new `INDEX.md` counts.
Do not commit unless the user explicitly asks for a commit.
