# Leading-subspace convergence for causal tokens

Source checked locally: `papers/arXiv-2411.04990v2/main_arxiv.tex`,
Sections 3–4, `lemma1` and Conjecture `thm2`.

The proved restrictions of `thm2` are:

- `Causal.subspace_cluster_twoTokens`: the complete two-token case.
- `Causal.subspace_cluster_of_positive_leading_covector`: any nonempty
  original tuple whose initial tokens admit a common positive leading covector.
- `Causal.subspace_cluster_of_count_le_finrank`: the complete original
  almost-everywhere conclusion whenever the token count is at most `dim L`.

Every actual token converges to the normalized leading projection of the
original leader. The query and key matrices are arbitrary. The value matrix
need not be symmetric. The strict transverse quadratic condition and
invariance of the orthogonal complement are exactly those in
`Causal.subspace_cluster`. The positive-covector condition and the count
restriction are additional, explicitly stated sufficient conditions.

The source leaves `thm2` as a conjecture. The original theorem keeps its exact
statement and its remaining `sorry` when the token count exceeds `dim L`.
No unrestricted result for those longer tuples is claimed.
The existing correction of the source's inconsistent
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
| `Section4_LeadingComponents` | Every actual last leading component has its exact predecessor forcing and shared radial coefficient. A common positive initial leading covector is preserved on every token by causal induction and an integrating factor. |
| `Section4_PositiveReference` | With a positive reference component, its nonnegative predecessor forcing helps the quotient argument while longitudinal overlap is negative. Positive longitudinal entry and its subsequent barrier are proved for any original prefix size. |
| `Section4_SubspaceFollowerDrift` | The original last-token force has positive leading drift minus the full original prefix error, retaining the actual self term and all original predecessor weights. |
| `Section4_SubspaceFollowerLimits` | Every actual follower with eventually nonnegative overlap inherits its actual prefix's leading limit. The positive initial covector supplies the entry event. |
| `Section4_ProjectedGenericity` | At counts up to `dim L`, leading projections are independent almost surely under the original product law. A projected dependent last token lies in the inverse image of a smaller projected span, a proper ambient subspace of zero original marginal measure. Independence supplies a common positive leading covector. |
| `Section4_SubspaceConeConsensus` | Actual prefix induction proves the pointwise arbitrary-count result, with its leader limit supplied by the proved original leader ODE. The preceding initialization theorem gives the original almost-everywhere conclusion for counts up to `dim L`. |

The projected-independence proof adapts
`Perspective.ae_linearIndependent_pi` from
`src/Transformer/Perspective/UniformHemisphere.lean`, which formalizes the
auxiliary hemisphere argument in arXiv:2312.10794v5, Section 6.1.
It does not assume a uniform law or an independence property for normalized
projected tokens.

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
constant. For an arbitrary prefix, the common positive initial covector
instead gives a positive bounded reference `r` with nonnegative original
forcing `g`:

```text
r' = g + h*r,    q' = f + h*q,    g >= 0.
(q/r)' = (f*r - q*g)/r^2 >= f/r    whenever q <= 0.
```

Actual prefix convergence bounds `f` below by a positive constant on a
tail. The quotient cannot remain negative forever, and the actual
longitudinal boundary derivative prevents a later crossing below zero.
Once `q` is nonnegative, the actual field satisfies

```text
q' >= lambda*csaWeightFloor*(1-q^2)
      - norm(V)*sum_j_in_prefix norm(x_j-u).
```

The last term tends to zero. No convergence, positivity, or measure-zero
assertion for the follower is supplied as an unproved input to the final
pointwise or count-restricted almost-everywhere theorem.

## Validation

The module build, full `lake build`, `./run.py audit`, `./run.py index`, and
`./run.py forbidden` are the required checks. Every new theorem is also
inspected with `#print axioms` by
`lake env lean scripts/CausalSubspaceAxioms.lean` (48 results, including the
original two-token development). Witness examples include actual orthogonal
two-token and three-token flows, actual consensus, the original shell law,
and collinear exceptional initial data.
The overall count remains 36 unfinished statements; the new two-token results
and their extensions have no `sorry` dependencies.
