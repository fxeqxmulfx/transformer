# Rotate90 interaction obstruction

The new `Transformer.Landscape.Neural.Interaction` module proves a necessary
condition for fitting division with the actual shared readout. It does not
assume an embedding geometry, a correct character representation or future
training behavior.

The architecture comes from `web/own-channel/README.md` in `grokking_abmod`
at `f49969393a7482490d45483d8a4fd3046b1adede`. The four-row argument also
appears in section 2.1 of that repository's working
`docs/rotate90-mathematics.md`, inspected on 2026-10-19. That note was
untracked; its other claims are not attributed to the recorded revision or
treated as formal results.

With the residual branch deleted, the unnormalized logit is

```text
<E[a], E[q]> + <J E[b], E[q]>.
```

RMSNorm divides every competing logit by the same strictly positive number
at each input. It therefore preserves weak and strict comparisons, including
at a zero hidden state. It can change the input-dependent scale but cannot
repair the decision obstruction.

For two additive score functions, the four differences between classes q
and r on a rectangle sum to zero with alternating signs. Requiring q on
the diagonal and r off the diagonal contradicts this identity. Two weak
and two strict comparisons suffice, so ties favoring q are included.

The concrete theorem retains vocabulary 97, width 128 and epsilon 1e-5.
The rows `(2,1)` and `(95,96)` have quotient 2; `(2,96)` and `(95,1)` have
quotient 95. Since 2 precedes 95, the theorem's comparisons are necessary
for the browser's smallest-index argmax to classify all four correctly.
These four rows need not all belong to a particular training half. An
independent count on the source-matched splits finds 134, 144 and 122
complete division rectangles for seeds 11, 17 and 23, respectively, among
2,304 distinct rectangles.

The checkpoint bridge proves that an actually zero dense matrix in the
complete SiLU model gives this exact no-branch logit, with E still shared.
It does not assert that SiLU training preserves a zero matrix. No ReLU
autodifferentiation convention, AdamW convergence, typical initialization,
positive attraction basin or scalar-MLP obstruction is inferred here.

The module and the full library build passed. `#print axioms` on all eight
new theorems reports only `propext`, `Classical.choice` and `Quot.sound`.
The full audit reports 39 direct unfinished statements, zero rests and zero
extra axioms. The regenerated index has 3,675 modules, 18,883 theorems,
39 `sorry`, zero vacuous statements and zero placeholders. The forbidden
check passes.

At this commit the strict all-linters gate was mandatory and had its existing
configuration blocker; see [dependency_update.md](dependency_update.md) for
the later user instruction making it optional.
The unchanged full command was attempted again with a 60-second process
limit and returned timeout status 124. It is not reported as a lint pass.
The ordinary build of the new module has no warnings. No linter has been
disabled to obtain that result.
