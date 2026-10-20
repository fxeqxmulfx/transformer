# Leading-subspace convergence for two causal tokens

Source checked locally: `papers/arXiv-2411.04990v2/main_arxiv.tex`,
Sections 3–4, `lemma1` and Conjecture `thm2`.

`Causal.subspace_cluster_twoTokens` proves the complete two-token restriction
of `thm2`. Both actual tokens converge almost surely to the normalized leading
projection of the original leader, whenever that projection is nonzero. The
query and key matrices are arbitrary. The value matrix need not be symmetric.
The strict transverse quadratic condition and invariance of the orthogonal
complement are exactly those in `Causal.subspace_cluster`.

The source leaves `thm2` as a conjecture. These proofs establish its two-token
case; the original theorem keeps its exact statement and its remaining `sorry`
for three or more tokens. The existing correction of the source's inconsistent
`P_(L-perp)` notation remains documented beside that theorem: its words and
surrounding argument specify projection onto `L`. Its existing nonzero-vector
qualification on the strict transverse inequality is also explicit there:
the printed strict inequality cannot hold at the zero vector.

## Statement and proof correspondence

| Module | Proved step |
| --- | --- |
| `Section4_SubspaceCovectors` | The invariant orthogonal splitting makes every leading vector a left eigenvector too. The Rayleigh defect is exactly its transverse defect. The full Rayleigh quotient is at most the leading eigenvalue. The actual scalar overlap equation and predecessor error estimate follow. |
| `Section4_TwoTokenLeading` | The leader's scaled leading component is fixed. A leading covector initially orthogonal to it stays orthogonal. Its actual follower component solves a homogeneous scalar equation and preserves a positive initial sign. |
| `Section4_TwoTokenPositiveEntry` | Dividing longitudinal overlap by this positive transverse component cancels the original nonlinear radial term. A positive lower bound on the actual leader forcing makes this quotient diverge and the follower overlap eventually positive. |
| `Section4_TwoTokenPositiveLimit` | The actual self contribution is nonnegative in the positive hemisphere. Actual leader convergence leaves uniformly positive drift on every closed overlap strip below one. Scalar escape and barriers give overlap one and hence convergence of the actual follower. |
| `Section4_SubspaceTransverse` | A nonzero polynomial residual supplies the required initial covector. Dimension at least two supplies a nonzero leading normal for every exceptional fiber. |
| `Section4_SubspaceTwoTokens` | The explicit exceptional set is Borel, each relevant fiber lies in a null great sphere, and the original product law gives the almost-everywhere result by Fubini. The pointwise dynamics and this measure statement are combined without assuming a follower limit. |

For the actual second token `x`, let `p` denote the actual leader and let
`a10, a11` denote its original masked attention weights. For every `v` in `L`,

```text
d/dt <v,x> = lambda*a10*<v,p> + h*<v,x>
h = lambda*a11 - <x, csaAverage(original row)>.
```

The radial term is the original weighted average, including both original
tokens. It is not replaced by the self Rayleigh quotient. With unit leading
limit `u`, longitudinal overlap `q=<u,x>`, and a transverse covector `v`,
the homogeneous component `r=<v,x>` is positive and bounded by `norm(v)`.
The exact quotient derivative is

```text
(q/r)' = lambda*a10*<u,p>/r.
```

After the leader approaches `u`, this is bounded below by a positive
constant. Once `q` is nonnegative, the actual field satisfies

```text
q' >= lambda*csaWeightFloor*(1-q^2) - norm(V)*norm(p-u).
```

The last term tends to zero. No convergence, positivity, or measure-zero
assertion for the follower is supplied as an unproved input to the final
two-token theorem.

## Validation

The module build, full `lake build`, `./run.py audit`, `./run.py index`, and
`./run.py forbidden` are the required checks. Every new theorem is also
inspected with `#print axioms` by
`lake env lean scripts/CausalSubspaceAxioms.lean`. Witness examples include actual orthogonal
two-token flows, actual consensus, and collinear exceptional initial data.
The overall count remains 36 unfinished statements; the new two-token results
have no `sorry` dependencies.
