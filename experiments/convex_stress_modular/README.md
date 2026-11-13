# Categorical codes in composite latent domains

Can observed categorical equations learn a phase code when the proposed cyclic
domain is composite? CM27 / Lean `98b12abc` proves exact residual lifting,
checked Bezout recombination and unit gauge identities before this native work.
It also proves that a valid base solution may have no lift, and that arbitrary
noncoprime moduli do not justify multiplying the modulus. These results do not
identify unseen labels or guarantee successful discovery.

The native integer core now passes five checks. It retains the original matrix
and right side, canonical base representatives, exact divided residual, reduced
prime-field system, every elimination operation and its complete affine space,
selected digits, next representatives, final original residuals and work.
When the affine space is nontrivial, it keeps every direction and explicitly
chooses its free-zero representative; that is not a uniqueness claim. Caps and
inconsistency leave no selected next point.

The independent checker uses i128 original-row arithmetic while the producer
uses bounded i64 arithmetic. It verifies coefficient reduction, full invertible
row operations, actual digit/residual identities and every final congruence.
CRT recombination retains and checks the actual Bezout witness, each component
residue, the explicit representative and all original equations. Unsupported
noncoprime components remain refusals. Deterministic replay additionally binds
every input, intermediate state, stopping decision and work counter.

Tests compare all 97 small scalar lift systems modulo four/nine against direct
enumeration of original integer solutions, and over 2000 small CRT residue pairs
against exhaustive product-domain search. They include CM27's nonzero matrix
example, a nonunique digit space, its failed base lift, each solver cap, invalid
representatives/domains, JSON round trips and tampering with residuals, matrices,
chosen points, Bezout witnesses and work. No original task was used in these
checks, and no categorical code learner or original score is implemented yet.

The next layer will use observed relations `r[i]+s[j]=t[label]`, with unknown
class codes and repeated input phases allowed. It may propose the latent domain
from observed class count, solve normalized component systems, lift residuals,
recombine components, and test actual code injectivity and every observation.
Every guess and rejected/capped branch must be retained. Generic categorical
strip views and every older family remain available; no numeric answer rule on
the names of input or output tokens is supplied.

The core restricts integer matrices to 1024 variables and 8192 rows, coefficients
of absolute value at most 1024, right sides at most 2^30, and total modulus at
most 257. The lift forecast covers report, checker, replay and serialization
and must fit within 8 GiB. Numerical processes use the user's 64 GiB cap:

```sh
cd experiments/convex_stress_modular
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Source: [`review/flat_spectrum.md`, CM27](../../review/flat_spectrum.md#cm27-composite-categorical-codes-require-checked-modular-lifts).
Prime-field row operations reuse the checked finite-system core at `05b89afc`;
integer residual lifting, original-system checks and CRT witnesses are new.
All35 short/long tasks, full batch, minibatch, forward-only, retained stronger
families and the conventional transformer comparison remain required. Any
original follow-up needs its own tested and committed producer freeze.
