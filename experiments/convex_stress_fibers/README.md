# Categorical phase proposals from complete label fibers

Can observed equal-label groups initialize a phase model when scalar spectral
sketches mix several modes? CM26 / Lean `13711438` proves the relevant equation
before this native work. An injective phase image inside an actual decoder
fiber is complete when its cardinality matches that full fiber. Its phase sum
then equals the fiber's actual mass. Both row and column orientations give
linear equations. A checked counterexample shows that incomplete observations
do not justify the equation. Common mass across different labels is an extra
model hypothesis, not a consequence of a flat spectrum.

The exact finite-system core passes four native checks. Exhaustive enumeration
of all 793 binary and ternary two-variable, two-equation systems independently
matches its complete affine solution sets, including inconsistent systems.
Other checks cover all three row operations, JSON round trips, every cap,
invalid domains, and tampering with inputs, operations, matrices, solutions and
work. It retains its original matrix,
right side, every row operation, reduced matrix, pivots, free coordinates,
particular solution and nullspace directions, including incomplete work.
The producer updates suffixes; the independent checker verifies zero prefixes
and replays full invertible row operations. It tests solutions against original
equations and checks the free-coordinate basis. A separate deterministic replay
binds all states, stopping decisions and work. These are integer certificates
for the supplied equations, not certificates that guessed fibers are complete.

The proposed next construction is fixed before any original-data discovery:

1. Use only categorical training cell/label pairs. Group equal labels within
   each row and column. Guess a small bank of group sizes between2 and8, keeping
   each guess and every group chosen under a deterministic constraint cap.
2. For a prime latent domain of size equal to the input category count, use
   unknown coordinates `r[0..n)`, `s[0..n)` and one proposed common mass `mu`.
   A size-k same-label group in column j supplies
   `sum r[i] + k*s[j] - mu = 0`; a group in row i supplies
   `k*r[i] + sum s[j] - mu = 0`. These use category indices as variable names,
   not arithmetic features of original tokens.
3. Pin one row and one column to zero and a second observed row to one. Treat
   inconsistent systems, missing axes and unsupported domains as explicit
   refusals. Check every original equation independently after elimination.
4. Bound nullity and enumerate a declared number of free-coordinate values.
   Retain every coefficient tuple and its outcome. Check both phase maps are
   permutations, recording concrete collision witnesses for rejected tuples.
5. Fit the decoder from the original observations. Refuse conflicts or
   uncovered latent states. For an admitted candidate, verify actual full
   decoder-fiber cardinalities and masses for every guessed group; this makes
   the candidate itself satisfy the Lean premises. It does not establish that
   the unknown true table belongs to that structural family.
6. Reconstruct the actual latent and output tables and audit every Fourier
   frequency, SVD budget and ordinary full-vocabulary CE. Keep all earlier
   task219 candidates and earlier-first CE ties. Freeze a separately tested
   producer before evaluating unchanged original splits.

Nullspaces may contain more freedom than shifts and rescaling. No uniqueness
or unseen-answer claim follows from a nonzero solution, a permutation, or
training fit alone. Capped enumeration must not be called a proof that the
model class is impossible. Any later identification certificate needs to
account for every feasible decoder completion, including uncovered states.

Planned fixtures use permuted phase coordinates, renamed output categories
and explicitly declared balanced fibers. They include incomplete fibers,
inconsistent guesses, ambiguous solutions, missing axes and tampering. They
are separate from the original nine tasks. Full35 short/long, full batch,
minibatch, forward-only and conventional transformer scope remains open.

The linear-algebra core restricts primes to2..257, variables to1024 and rows
to8192. Its policy caps scalar updates, row operations and nullity separately.
No solver cap silently produces an accepted phase model. Native checks run
under the user's64GiB process cap:

```sh
cd experiments/convex_stress_fibers
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The source is [`review/flat_spectrum.md`, CM26](../../review/flat_spectrum.md#cm26-complete-categorical-fibers-give-phase-sum-constraints).
Prime-field normalization builds on the native coding learner at `4510ee7d`;
general affine systems, row-operation certificates, arbitrary variable counts
and explicit nullspace bounds are new. Final spectral/CE audits reuse the
CM25 model at `69c83182`. Generated evidence remains ignored and rooted.

A possible later extension is an unknown injective output code over a composite
cyclic domain: solve observed equations `r[i]+s[j]=t[label]` with gauge pins,
factor the proposed domain into prime powers, lift solutions digit by digit,
and combine checked residues. The domain would come from observed class count,
with repeated input phases allowed. CM17's current identification theorem
assumes a prime domain; such a composite construction needs its own Lean
boundary and native checks. It is not implemented or claimed here.
