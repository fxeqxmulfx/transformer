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
The new `Causal.csaLast_tendsto_or_negative_direction` has no count or
hemisphere restriction: given an actually converging original prefix, the
actual last token either converges to its leading limit `u`, or its leading
projection stays away from zero and its normalized direction converges to
`-u`. This does not assert convergence of the full exceptional token or
measure zero of the exceptional basin.
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
| `Section4_ExpandingScalar` | A bounded scalar with a uniformly expanding coefficient and vanishing forcing tends to zero, with derivatives required only on a tail. Positive longitudinal forcing and a bounded diagonal coefficient give eventual nonnegative overlap or a uniform negative gap. |
| `Section4_LeadingForcing` | The actual last-row leading coefficient has magnitude at most `abs(lambda) + norm(V)`. The original converging prefix supplies a uniformly positive longitudinal forcing and a vanishing forcing against every leading covector orthogonal to its limit. No convergence of the attention weights is assumed. |
| `Section4_TransverseQuotient` | If longitudinal overlap stays uniformly negative, its quotient with each transverse leading component has a uniformly expanding coefficient and vanishing forcing. Every transverse leading component tends to zero. |
| `Section4_LeadingDirection` | Fixed-covector convergence gives norm convergence of the off-axis leading projection by a finite coordinate test. Uniformly negative overlap bounds the leading projection away from zero and controls its normalized distance from `-u`. |
| `Section4_NegativeLeadingBranch` | Every actual last follower of a converging original leading prefix either converges to `u`, or stays uniformly negative with normalized leading projection tending to `-u`. There is no bound on prefix length or additional initialization condition. |
| `Section4_FrozenFollowerField` | The actual last-row field at a leading consensus prefix retains every original weight and self term. Its exact leading component is the positive original predecessor mass plus the original common radial coefficient. Negative critical overlap makes that coefficient positive. |
| `Section4_TransverseLinearization` | Differentiating an exact transverse scalar factorization keeps the derivative of the coefficient and yields a left eigenvector when the actual transverse component is zero. Finite dimension gives a right eigenvector without symmetry. Projection retains a positive eigenvalue in a genuine tangent direction, and the derivative at a critical projected field maps tangent directions into the tangent space. |
| `Section4_LeadingCriticalInstability` | At every actual negative critical follower of a leading consensus prefix, leading dimension at least two supplies a nonzero transverse left eigenvector of the full original derivative. Its positive right eigenvalue has a nonzero original tangent eigenvector. No symmetry of the value matrix or omission of attention derivatives is used. |
| `Section4_SubspaceLimitInstability` | Actual original token limits give the original limiting velocity. Boundedness forces that limiting velocity to vanish. An actual non-consensus follower limit is therefore a negative critical point with a positive actual tangent eigenvalue. Existence of the full follower limit is an explicit premise of this necessary-condition theorem. |

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

## The remaining exceptional branch

For any original prefix converging to a unit `u` in `L`, set `q=<u,x>`
for its actual last token, and `r=<v,x>` for any fixed `v` in `L` with
`<v,u>=0`. The exact original scalar equations are

```text
q' = f + h*q,    r' = g + h*r,
f >= c > 0 eventually,    g -> 0,
abs(h) <= abs(lambda) + norm(V).
```

The positive forcing gives a positive drift across a fixed strip just
below zero. Either the actual overlap enters the nonnegative hemisphere
and the already proved follower theorem applies, or `q <= -delta < 0`
on a tail. On that tail `z=r/q` solves

```text
z' = (-f/q)*z + g/q,    -f/q >= c > 0,    g/q -> 0.
```

The denominator is bounded away from zero, so `z` is bounded. On each
positive level its derivative is eventually positive, and on each negative
level it is eventually negative. Scalar barriers and boundedness force
`z -> 0`, hence `r -> 0`. This uses the actual nonlinear coefficient; it
does not replace it by a frozen-prefix or self-only coefficient.

Consequently `P_L x - <u,x>u -> 0`. Negative overlap also gives
`norm(P_L x) >= delta`, and the repository normalization estimate gives

```text
norm(P_L x / norm(P_L x) + u)
    <= 2*norm(P_L x - <u,x>u)/delta -> 0.
```

The original stationary antipodal CSA pair witnesses this exceptional
alternative and satisfies the original spectral hypotheses. A nonzero
decaying scalar example also verifies that transverse forcing tending to
zero gives convergence, not exact vanishing at finite time. Neither
example is a counterexample to an almost-everywhere statement.

The missing step is still conditional measure zero of this exceptional
basin for arbitrary converging original prefixes. Approaching an
asymptotic lower-dimensional set does not by itself prove that the basin
has measure zero. The original `subspace_cluster` statement has not been
weakened, and its remaining `sorry` is retained.

## Exact tangent eigenvalues at actual non-consensus limits

For `n` original predecessors all equal to a leading unit vector `u`, let
`F(x)` be the original last-row field, `a(x)` its original self weight,
and `A(x)` its full original value average. The auxiliary limiting field
uses the same `n+1` tokens in the denominator; it is not substituted for
an arbitrary moving-prefix trajectory. For every original leading `v`,

```text
<v,F(x)> = lambda*(1-a(x))*<v,u> + h(x)*<v,x>,
h(x) = lambda*a(x) - <x,A(x)>.
```

At a critical point with `q=<u,x><0`, the exact longitudinal identity is
`0=lambda*(1-a)+h*q`. Nonempty original predecessor mass is strictly
positive, so `h>0`. For every `v` in `L` perpendicular to `u`, the same
identity gives `<v,x>=0` and, upon differentiating the full field,

```text
<v,DF(x)w> = Dh(x)w*<v,x> + h(x)*<v,w> = h(x)*<v,w>.
```

The attention derivatives are present in `Dh`; their contribution here
vanishes because the actual transverse component is zero. It is not a
frozen-weight approximation. Leading dimension at least two supplies a
nonzero `v`, tangent to the original unit sphere at `x`.

The left eigenvector makes `DF-h I` non-surjective. In finite dimension
it consequently has a nonzero kernel. Applying the same argument to the
derivative projected on both sides onto the original tangent space gives
a nonzero tangent right eigenvector. Differentiating the actual projected
field at an actual critical point also proves that `DF` maps tangent
directions to tangent directions. Thus `h>0` belongs to the actual
tangent derivative, without a symmetry assumption on `V`.

For a moving original prefix converging to `u`, existence of a full last
limit `x` makes the actual velocity tend to `F(x)`. A nonzero `F(x)` would
give a uniformly positive eventual derivative of `<F(x),x_last(t)>`,
contradicting its boundedness on the original sphere. Hence `F(x)=0`.
If `x` differs from `u`, the previous exact negative-branch dichotomy
makes `<u,x>` strictly negative, so the tangent eigenvalue result applies.

This is a local algebraic instability result and a necessary condition on
convergent exceptional trajectories. It does not supply measure zero of
their original basins, nor does it establish full follower convergence
from convergence of the normalized leading projection. Those steps must
not be inferred from the positive eigenvalue alone.

## Validation

The module build, full `lake build`, `./run.py audit`, `./run.py index`, and
`./run.py forbidden` are the required checks. Every new theorem is also
inspected with `#print axioms` by
`lake env lean scripts/CausalSubspaceAxioms.lean` (86 results, including the
original two-token development). Witness examples include actual orthogonal
two-token and three-token flows, actual consensus, the original shell law,
and collinear exceptional initial data.
The overall count remains 36 unfinished statements; the new two-token results
and their extensions, including the unrestricted follower dichotomy and
the exact critical-limit tangent eigenvalues, have no `sorry` dependencies.
