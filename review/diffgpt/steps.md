# DiffGPT: a recurrence written as steps

Moved unchanged from [`../flat_spectrum.md`](../flat_spectrum.md) on 2026-11-19: the sections
*The wrap of a step is a comparison* through *The decoder holds the level*. They study the
DiffGPT task *Fibonacci mod 97*, a recurrence modulo `p` that the model reads and writes as
integer steps: the wrap of a step is a comparison (CW), the step is the phase of a sum (CH), the
decay of two training recipes (RR), what the training cells decide for a learner with free
scores (CS), the tokens as numbers (NT), steps against levels (EN), the carry of the codec (CC)
and the level the decoder holds (DL). The Lean is mostly in `Transformer.Grokking.FlatSpectrum`;
each *Lean* section names its modules.

These labels are this file's; its CC1–CC7 are not the CC1–CC8 of
[`../convex/channels.md`](../convex/channels.md). M, Lemma 1, Proposition 2, Theorems A to G,
WD and OC are those of [`../flat_spectrum.md`](../flat_spectrum.md), and AD1 and AD3 those of
[`../actual_network/dynamics.md`](../actual_network/dynamics.md).

## The wrap of a step is a comparison (added)

Added on 2026-11-10 and proved in Lean. A difference encoding writes a recurrence
`x ↦ (b + x) mod p` by its integer steps: from three consecutive values `a`, `b`,
`x = (a + b) mod p` the model reads `d₁ = b − a` and `d₂ = x − b`, and it has to write
`d₃ = y − x` for `y = (b + x) mod p`. This is the DiffGPT task *Fibonacci mod 97* with one
feature; its decoded answer is right only for the exact integer `d₃`. This section says what M
does with it.

The step is a residue and a wrap. The residue has a kernel of rank one, which M recovers
(Theorems D, E, G). The wrap is a comparison of the two addends. Its kernel is `1 + A` with
`A` skew; M does not recover it from a table that misses a cell of its boundary, and with one
such cell unseen it answers with the opposite sign. An input token that training never showed
is a zero row of every minimum. Read through the phase of the sum that is not reduced, the
same bit is a half-plane of a kernel of rank one.

### CW1. A step is a residue and a wrap

For `0 ≤ b, x < p` let `s(b, x) = ((b + x) mod p) − x` be the step from `x` when the value
before it is `b`. Then

```
s(b, x) = b        if b + x < p,
s(b, x) = b − p    if b + x ≥ p.
```

So `s(b, x) ≡ b (mod p)`, and `s(b, x) < 0` exactly when the sum wraps. Along `a`, `b`,
`x = (a + b) mod p` the steps are `d₁ = b − a`, `d₂ = s(a, b)` and `d₃ = s(b, x)`, and

```
d₃ ≡ b ≡ d₁ + d₂   (mod p).
```

Hence `ψ(d₃) = ψ(d₁) ψ(d₂)` for every additive character `ψ` of `ℤ/p`: read through `ψ`, the
step is a kernel `u vᵀ` of modulus one in the two steps before it, the kernel of Theorems D, E
and G. The decoded integer needs one more bit, `[b + x ≥ p]`.

### CW2. The comparison kernel loses its boundary

The comparison kernel of `n` ordered values is `C_ij = 1` for `i ≤ j` and `C_ij = −1` for
`j < i`. Its boundary is the diagonal and the cells `(i + 1, i)` below it: the pairs of
neighbours, between which the sign changes.

`C + Cᵀ = 2`, so `C = 1 + A` with `A` skew. In general let `K + K* = 2c` with `c > 0`.

- `K` is invertible: `Re ⟨v, K v⟩ = c ‖v‖²`.
- `K* = 2c − K` commutes with `K`. So `S = (K* K)^{-1/2}` commutes with `K`, and the polar
  factor `W = K S` has `W + W* = (K + K*) S = 2c S`.
- `S` is positive definite, so `Re W_ii = c S_ii > 0`.

By Theorem A such a `K` is not a minimum of `‖X‖_S1` over `{X = K on Ω}` once `Ω` misses a
diagonal cell. The identity is the case `A = 0`.

For `C`, row `i + 1` is row `i` minus `2 e_i`. So `W_{i+1,i} = W_ii − 2 S_ii` and
`Re W_{i+1,i} = −S_ii < 0`. By Theorem A, `C` is not a minimum once `Ω` misses a cell of its
boundary.

The spectrum of `C` is not flat. For `q^n = −1` the vector `v_j = q^j` has
`C v = (2 / (1 − q)) v`, and `C` is normal, so its singular values are
`1 / sin((2k + 1) π / 2n)`, `k = 0, …, n − 1`. The two largest are about `2n/π` and hold
about `8/π² = 81%` of `‖C‖_S2² = n²`, and `‖C‖_S1 = (2n/π)(ln n + O(1))`. This paragraph is
not in Lean.

### CW3. A kink between two projections

Let `P` and `Q` be orthogonal projections with `P K = K = K Q`, and let

```
‖P E Q‖_S1 ≤ ‖(1 − P) E (1 − Q)‖_S1.
```

Then `‖K‖_S1 ≤ ‖K + x E‖_S1` for every `x ∈ ℂ`.

Proof. For contractions `Z`, `Z'` the matrix `Y = P Z Q + (1 − P) Z' (1 − Q)` is a
contraction: `1 − Y* Y = Q (1 − Z* P Z) Q + (1 − Q)(1 − Z'* (1 − P) Z')(1 − Q)`, and
`1 − Z* P Z = (1 − Z* Z) + ((1 − P) Z)* ((1 − P) Z)`. Take `Z` with `tr(Z* K) = ‖K‖_S1` and
`Z'` with `tr(Z'* x (1 − P) E (1 − Q)) = |x| ‖(1 − P) E (1 − Q)‖_S1`. As
`P (K + x E) Q = K + x P E Q` and `(1 − P)(K + x E)(1 − Q) = x (1 − P) E (1 − Q)`,

```
tr(Y* (K + x E)) = ‖K‖_S1 + x tr(Z* P E Q) + |x| ‖(1 − P) E (1 − Q)‖_S1,
```

and `|tr(Z* P E Q)| ≤ ‖P E Q‖_S1`. So the real part of the left side is at least `‖K‖_S1`,
and it is at most `‖K + x E‖_S1`. ∎

The norm need not be constant along `E`. The part of `E` outside both ranges costs
`|x| ‖(1 − P) E (1 − Q)‖_S1`, and the part inside saves at most `|x| ‖P E Q‖_S1`: a kink
at `x = 0`.

### CW4. Merged levels

Let rows `i ≠ i'` of `K` be equal and columns `j ≠ j'` be equal. Then `K` is a minimum of
`‖X‖_S1` among the matrices equal to it off the cell `(i, j)`.

Proof. `P = 1 − ½ (e_i − e_i')(e_i − e_i')ᵀ` is the projection onto the vectors with equal
coordinates `i` and `i'`, and `Q` is the same for `j` and `j'`; `P K = K = K Q`. A projection
`A` has `Σ_r |A_ri|² = A_ii`, so `‖A e_i e_jᵀ B‖_S1 = √(A_ii B_jj)` for projections `A`, `B`.
As `P_ii = Q_jj = ½`, both `‖P e_i e_jᵀ Q‖_S1` and `‖(1 − P) e_i e_jᵀ (1 − Q)‖_S1` are `½`,
and CW3 holds for `E = e_i e_jᵀ`, with equality in its hypothesis. ∎

So when one value of a cell merges its row with another row and its column with another
column, that value is a fill of M.

### CW5. One unseen cell of the boundary

Let `(i, i)` with `0 < i < n − 1` be the only unseen cell of `C`. With `−1` in it, row `i` is
row `i + 1` and column `i` is column `i − 1`. By CW4 this matrix is a minimum of M. By CW2,
`C` is not, and a minimum with the true `1` in the cell would be `C`. So no minimum holds the
true value.

Let `(i + 1, i)` be the only unseen cell. With `1` in it, rows `i` and `i + 1` are equal and
columns `i` and `i + 1` are equal. So this matrix is a minimum of M, and no minimum holds the
true `−1`.

M answers a comparison of two neighbours it has not seen against the truth: it merges them.
OC5 gives the fill `−θ κ` for a kernel with orthogonal rows; here the rows are not
orthogonal, and the fill of CW4 is `−κ`.

### CW6. The wrap in the coordinates of the addends

Take the rows `b` and the columns `x`. The sign of the step, `−1` where `b + x` wraps and `1`
where it does not, is `C_{b, p−1−x}`: `b + x < p` reads `b ≤ p − 1 − x`. Reversing the columns
keeps the nuclear norm and carries minima to minima. The boundary of the wrap is the pairs
with `b + x = p − 1`, the last sums that do not wrap, and with `b + x = p`, the first that
do. So:

- the sign of the step is not a minimum of M once `Ω` misses a pair of the boundary;
- with one pair of the boundary unseen, no minimum holds the true sign there;
- with the pair `(b, x)`, `b + x = p − 1`, `0 < b < p − 1`, unseen, the table with a wrap at
  that pair is a minimum: M writes `b − p` for `b`;
- with the pair `(b, x)`, `b + x = p`, unseen, the table without a wrap there is a minimum: M
  writes `b` for `b − p`.

### CW7. A half-plane of a phase of rank one

The model reads the steps, not the addends. In the steps the sum that is not reduced is an
integer form:

```
b + x = 2 d₁ + 3 d₂ + 2p [d₂ < 0],
```

from `d₂ = a − p [d₂ < 0]`, `b = a + d₁` and `x = b + d₂`. So the wrap compares `2 d₁` with
`p − 3 d₂ − 2p [d₂ < 0]`, and

```
exp(iπ (b + x) / p) = exp(2πi d₁ / p) · exp(3πi d₂ / p):
```

the phase of order `2p` of the sum is a kernel of rank one and modulus one in the two steps.
For `0 ≤ n < 2p`, `sin(π (n + ½) / p) < 0` exactly when `n ≥ p`. Hence

```
d₃ < 0    ⟺    Im( exp(iπ / 2p) · exp(2πi d₁ / p) · exp(3πi d₂ / p) ) < 0.
```

The bit has two readouts. As a sign it is a comparison, which M does not complete at the
boundary (CW2, CW5, CW6). As the phase of the sum it is a kernel of rank one, which M
recovers under the conditions of Theorem G, and the wrap is a fixed half-plane of the
recovered value. The phase is not a function of the answer alone: `b + x = 2 (d₃ mod p) + d₂`.
It is a function of the answer and of the last step read, so a training example shows it.

### CW8. An unseen token

For an orthogonal projection `P` of the rows,

```
‖P X‖_S1² + ‖(1 − P) X‖_S1² ≤ ‖X‖_S1².
```

Proof. For contractions `Z`, `Z'` and real `c`, `s` with `c² + s² ≤ 1` the matrix
`Y = c P Z + s (1 − P) Z'` is a contraction:
`1 − Y* Y = c² (1 − Z* P Z) + s² (1 − Z'* (1 − P) Z') + (1 − c² − s²)`. Pair `X` with `Y` for
the `Z`, `Z'` that attain the norms of the two blocks:
`c ‖P X‖_S1 + s ‖(1 − P) X‖_S1 ≤ ‖X‖_S1`. Take `(c, s)` proportional to the two norms. ∎

So removing a block of rows that is not zero lowers the norm. If no cell of row `i` is in
`Ω`, removing the row stays among the matrices equal to `K` on `Ω`. Hence every minimum of M
has a zero row `i`, and by the adjoint (OC1) a zero column `j` when no cell of column `j` is
in `Ω`.

A token that does not occur in training at an input position is such a row or column of
every kernel of the task. M answers `0` there, which decodes to no value: for M the share of
held-out inputs with an unseen token is a floor of the error.

### CW9. The tables of the task (numerical)

These numbers are computed, not proved. The task is `p = 97` with one feature and MASH-half
tokens: 9408 states in 48 cycles of 196, of which 24 cycles (4704 states) are held out, for
the seeds 11, 17 and 23 of the split. The two input tokens index a table of `194 × 370`
cells, one cell for a state. M is computed by ADMM from the 4704 seen cells and decoded at
the 4704 held-out ones. The program is `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository (Rust, CPU).

| Kernel | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| Residue `ψ(d₃)`, decoded right | 99.745% | 99.298% | 98.278% |
| Sign of `d₃`, right | 91.008% | 91.837% | 90.795% |
| Held-out cells with an unseen input token | 12 | 31 | 85 |
| of these, residue right | 0 | 1 | 4 |
| Residue wrong, both tokens seen | 0 | 3 | 0 |
| Sign right at `b + x ∈ {p − 1, p}` | 40 / 91 | 41 / 94 | 54 / 110 |
| Sign right at `b + x ≤ p − 8` or `≥ p + 7` | 3889 / 4034 | 3938 / 4058 | 3853 / 4016 |
| Half-plane of the phase of CW7 right, both tokens seen | 4692 / 4692 | 4671 / 4673 | 4619 / 4619 |
| Half-plane of the phase of CW7 right, an unseen token | 6 / 12 | 16 / 31 | 41 / 85 |
| Residue and half-plane both right | 99.745% | 99.298% | 98.278% |

- The residue is recovered where both tokens were seen, and lost at an unseen token (CW8):
  0.26%, 0.66% and 1.81% of the held-out cells.
- The sign of the step is at chance at the boundary and wrong in 9% of the held-out cells.
- The phase `exp(iπ (b + x) / p)` of CW7, completed by M as a second kernel of rank one in
  place of the sign, gives the wrap by its half-plane at every held-out cell where the
  residue is right. At an unseen token its row is zero and the half-plane is a coin. So M
  with two kernels of rank one answers every held-out state with both tokens seen, but three
  on seed 17; on seed 23 that is 98.278%, below 99%.
- One unseen diagonal cell of `C`, the others seen: the fill is `−1.000000` for
  `n = 2, 3, 4, 8, 16, 25, 97` (CW5). One unseen cell at the distance `1, 2, 3, 5, 8` above
  the diagonal, `n = 97`: the fill is `0.240`, `0.535`, `0.666`, `0.787`, `0.861` of the
  truth.
- The singular values of `C` are those of CW2 to `5 · 10⁻¹³` for these `n`.

### CW10. Scope

Proved: CW1 to CW8 as stated, without the paragraph on the spectrum in CW2.

Not covered:

- The table of the tokens. CW2, CW5 and CW6 are for the full table of the two addends with
  one cell unseen. The model reads steps, and in the coordinates of the steps a quarter of
  the cells are states. The task hides half of the cycles. For it there are only the numbers
  of CW9.
- Uniqueness. CW5 and CW6 say that the flipped table is a minimum and that no minimum holds
  the truth. That every minimum is the flipped table is not proved; in CW9 the computed fill
  is `−1.000000`.
- The corners. For the cells `(0, 0)` and `(n − 1, n − 1)` CW2 says that `C` is not a
  minimum; CW4 does not apply, as one neighbour is missing.
- Readouts of the answer other than its sign and its characters. A readout of the integer
  step that is a function of the answer alone and has a kernel of rank one is not exhibited;
  the phase of CW7 depends on the last step read as well.
- Trained models. CW5, CW6 and CW8 are about the minimum of M, the object of the modeling step
  of Lemma 1. What a fit trained at a fixed weight decay answers at the boundary of the wrap
  is not derived.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, modules `StepWrap`, `PolarDiagonal`, `Compare`,
`ProjectionKink`, `MergedLevels`, `CompareFill`, `WrapFill`, `StepHalfPlane` and `UnseenRow`,
namespace `Transformer.Grokking.FlatSpectrum`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| CW1 | `modStep`, `modStep_eq`, `modStep_neg_iff`, `intCast_modStep`, `modStep_modEq_add`, `addChar_modStep` | `modStep p b x : ℤ` for natural `b`, `x`; the two cases are `b - if p ≤ b + x then p else 0`. The congruences are stated for all natural `a`, `b`, `x`; the cases and the sign ask for `b, x < p`. `ψ` is any `AddChar (ZMod p) ℂ`. That the kernel is the one of Theorems D, E and G is not a Lean statement. |
| CW2 | `gramInvSqrt`, `polarFactor_eq_mul_gramInvSqrt`, `gramInvSqrt_eq`, `conjTranspose_gramInvSqrt`, `gramInvSqrt_apply_self_pos`, `det_ne_zero_of_add_conjTranspose`, `polarFactor_add_conjTranspose`, `re_polarFactor_apply_self`, `not_isMinOn_of_add_conjTranspose`, `compareKernel`, `compareKernel_add_conjTranspose`, `compareKernel_succ_row`, `compareKernel_succ_col`, `not_isMinOn_compareKernel_diag`, `re_polarFactor_compareKernel_succ`, `not_isMinOn_compareKernel_succ` | `K + Kᴴ = ((2 * c : ℝ) : ℂ) • 1`. `S` is `gramInvSqrt K`, the functional calculus of `x ↦ (√x)⁻¹` at `Kᴴ * K`, and its diagonal is given as a positive real. `compareKernel n` is over `Fin n`; the neighbour is `i'` with `(i : ℕ) + 1 = i'`. The spectrum of `C` is not in Lean. |
| CW3 | `one_sub_mul_one_sub`, `IsContraction.posSemidef_one_sub`, `IsContraction.proj_add_proj`, `trace_conjTranspose_proj_mul`, `nuclearNorm_le_add_smul_of_proj`, `isMinOn_compl_singleton_of_proj` | A projection is `Pᴴ = P` and `P * P = P`. The last statement is the case `E = e_i e_jᵀ` as a minimum among the matrices equal to `K` on `({(i, j)}ᶜ : Set (m × n))`. |
| CW4 | `diffVec`, `mergeProj`, `star_diffVec`, `diffVec_dotProduct_self`, `conjTranspose_mergeProj`, `mergeProj_mul_mergeProj`, `mergeProj_apply_self`, `mergeProj_mul_of_row_eq`, `mul_mergeProj_of_col_eq`, `sum_norm_sq_col_of_proj`, `sum_norm_sq_row_of_proj`, `nuclearNorm_mul_single_mul_of_proj`, `isMinOn_of_row_eq_of_col_eq` | `mergeProj i i'` is `1 − ½ (e_i − e_i')(e_i − e_i')ᴴ`. The norm of `A e_i e_jᵀ B` is `√(A i i).re * √(B j j).re`. |
| CW5 | `add_smul_single_one_apply`, `mem_add_smul_single_of_mem`, `isMinOn_compareKernel_flip_diag`, `isMinOn_compareKernel_flip_succ`, `ne_of_isMinOn_compareKernel_diag`, `ne_of_isMinOn_compareKernel_succ` | The flipped tables are `compareKernel n + (-2) • single i i 1` and `compareKernel n + 2 • single i₁ i 1`. "No minimum holds the true value" is `X i i ≠ 1` and `X i₁ i ≠ -1` for a minimum `X` equal to the kernel at the other cells, and holds at the corners as well. The comparison with OC5 is not a Lean statement. |
| CW6 | `wrapKernel`, `wrapKernel_eq_submatrix`, `wrapKernel_submatrix`, `isMinOn_submatrix`, `not_isMinOn_wrapKernel`, `ne_of_isMinOn_wrapKernel`, `isMinOn_wrapKernel_flip`, `isMinOn_wrapKernel_flip_succ` | `wrapKernel p` is `-1` where `modStep p b x < 0` and `1` elsewhere, over `Fin p`. The boundary is `(b : ℕ) + x + 1 = p ∨ (b : ℕ) + x = p`. `isMinOn_submatrix` is for any two permutations of rows and columns. "M writes `b − p` for `b`" is the flipped sign, not a statement about `modStep`. |
| CW7 | `add_eq_steps`, `modStep_neg_iff_steps`, `exp_sum_eq_mul`, `sin_neg_iff_le`, `modStep_neg_iff_im_neg` | `d₁` is `(b : ℤ) - a` and `d₂` is `modStep p a b`, for `a, b < p`; `[d₂ < 0]` is an `if`. The last paragraph of the item is not a Lean statement. |
| CW8 | `IsContraction.smul_proj_add`, `nuclearNorm_proj_sq_add_sq_le`, `nuclearNorm_proj_lt`, `row_eq_zero_of_isMinOn`, `col_eq_zero_of_isMinOn` | "Removing a block lowers the norm" is `nuclearNorm (P * X) < nuclearNorm X` for `(1 - P) * X ≠ 0`. The row has `∀ j, (i, j) ∉ Ω`, and that `X` equals `K` on `Ω` is a separate hypothesis. The reading for tokens is not a Lean statement. |

## The step is the phase of a sum of two phases (added)

Added on 2026-11-10 and proved in Lean. The section before left the wrap of a step with two
readouts: its sign, a comparison that M does not complete at the boundary (CW5, CW6), and a
half-plane of the phase of the sum, a kernel of rank one (CW7). A half-plane is still a
threshold on a smooth value. This section removes the threshold. The phase of order `2p` of
the step is the phase of a sum of one phase of each of the two steps read, times a kernel of
rank one; the comparison is the branch of a half angle, and a normalization of the modulus
takes it.

### CH1. The sum of two phases

```
exp(iα) + exp(iβ) = 2 cos((α − β) / 2) · exp(i (α + β) / 2).
```

So the sum of two phases, with its modulus normalized, is the product of the two half phases
times the sign of `cos((α − β) / 2)`: a product, and a comparison of the two angles.

### CH2. The chord of the sum

Along `a`, `b`, `x = (a + b) mod p` let `n = b + x`, and let
`z = exp(iπ n / p) = exp(2πi d₁ / p) · exp(3πi d₂ / p)` be the phase of the sum (CW7). The
chord from `−exp(−iπ / 2p)` to `z` is

```
S = z + exp(−iπ / 2p) = 2 cos φ · exp(i (φ − π / 2p)),        φ = π (n + ½) / 2p,
```

by CH1 for `α = π n / p` and `β = −π / 2p`. As `0 ≤ n ≤ 2p − 2`, the angle `φ` lies in
`(0, π)`, and `φ < π/2` exactly when `n < p`. So `cos φ > 0` without a wrap and `cos φ < 0`
with one, `S ≠ 0`, and `|S| = 2 |cos φ|`. The phase of `S` halves the angle of `z` and turns
by `π` at the wrap. The point `−exp(−iπ / 2p)` lies between the phases of the sums `p − 1`
and `p`, where no sum has its phase.

### CH3. The step is the phase of the chord

```
exp(iπ d₃ / p) = exp(iπ / 4p) · exp(−iπ d₂ / 2p) · S / |S|.
```

Proof. `d₃ = b − p [n ≥ p]` (CW1), so `exp(iπ d₃ / p)` is `exp(iπ b / p)` times the sign of
`cos φ` (CH2). And `n = 2b + d₂`, so `φ − π / 2p = π b / p + π d₂ / 2p − π / 4p`. ∎

The phase of order `2p` separates the integers `−p < d < p`: the steps `b` and `b − p` have
opposite phases. With CW1, for a character `ψ` of `ℤ/p`,

```
ψ(d₃) · exp(iπ d₃ / p) = ψ(d₁) ψ(d₂) · exp(iπ / 4p) · exp(−iπ d₂ / 2p) · S / |S|.
```

The characters of `ℤ/2p` of even order are the characters of `ℤ/p`, the kernels of CW1; those
of odd order are `ψ(d) exp(iπ d / p)`. So every character of `ℤ/2p` of the answer is a kernel
of rank one and modulus one in the two steps read, times, for an odd order, the phase of `S`;
and the characters of `ℤ/2p` determine the integer step.

### CH4. A sum of one phase of each step

```
T = exp(iπ / 2p) · exp(3πi d₂ / p) + exp(−2πi d₁ / p) = exp(iπ / 2p) · exp(−2πi d₁ / p) · S.
```

So `|T| = |S|`, and

```
exp(iπ d₃ / p) = exp(−iπ / 4p) · exp(2πi d₁ / p) · exp(−iπ d₂ / 2p) · T / |T|.
```

`T` is the sum of a function of the last step read and a function of the step before it, both
of modulus one: what a residual connection adds, when one channel holds a phase of each of the
two tokens.

### CH5. What a normalization of the modulus changes

This item is a reading and a hypothesis, not a theorem.

- `S` and `T` have rank at most two and are made of kernels of rank one and modulus one, the
  kernels that M recovers. They are not of modulus one: `|S| = 2 |cos φ|`.
- A readout linear in `S` holds the wrap as the half-plane of CW7:
  `Im(exp(iπ / 2p) z) = sin(π (n + ½) / p)`, which is `sin(π / 2p)` at `n = p − 1` and
  `−sin(π / 2p)` at `n = p`. For `p = 97` this is 1.6% of its range. A sharp answer from it
  needs a threshold, and a threshold is fitted from the cells next to it: the comparison of
  CW5 and CW6.
- With the modulus normalized, `S / |S|` has opposite values on the two sides of the boundary.
  The boundary is placed by `z`, which all cells fix, and not by the cells next to it.
- A normalization of the mean square over the whole width divides every channel by one number
  and keeps `S` up to a scale. The normalization that CH3 and CH4 use acts on one complex
  channel, that is on one pair of real coordinates. It is a refinement: when every pair has
  mean square one, the vector has.

The hypothesis for a trained model: with the mean square normalized over each pair of
coordinates, in place of the whole width, the answers at the boundary of the wrap are fixed
by the cells away from it. The experiment is in `research/diffgpt_flat_spectrum/` of the
`diff-gpt` repository.

### CH6. The sign of the step is a real kernel of rank two, normalized

The half-plane value of CW7 is the real number

```
ℓ = Im( exp(iπ / 2p) · exp(2πi d₁ / p) · exp(3πi d₂ / p) ) = sin(π (n + ½) / p).
```

With `u = exp(iπ / 2p) exp(2πi d₁ / p)` and `v = exp(3πi d₂ / p)` it is
`Re(u) Im(v) + Im(u) Re(v)`: a sum of two products of a real function of `d₁` and a real
function of `d₂`. So `ℓ` is a real kernel of rank at most two in the two steps, one real
channel of a bilinear map of them. It is never zero: positive without a wrap, negative with
one. Hence

```
ℓ / |ℓ| = −1   if d₃ < 0,        ℓ / |ℓ| = 1   otherwise:
```

the sign of the step, the kernel of CW6, is `ℓ` with its modulus normalized.

### CH7. One angle for all cells

Move the base point of the chord of CH2 by an angle `2γ`:

```
S_γ = z + exp(−iπ / 2p) · exp(2iγ) = 2 cos(φ − γ) · exp(i (φ − π / 2p + γ)).
```

For `|γ| < π / 4p` the moved point is still between the phases of the sums `p − 1` and `p`.
Then `cos(φ − γ)` has the sign of `cos φ` at every cell, and

```
S_γ / |S_γ| = exp(iγ) · S / |S|.
```

Every answer turns by the same angle and none changes its branch. In the readout of CH3 the
wrap of all cells is placed by one angle, with a tolerance of `π / 2p` for the base point.
Conversely, by the same formula, a base point outside this arc gives the wrong branch at the
sum `p − 1` or at the sum `p`; so one seen cell of each kind confines the angle. This converse
is not in Lean.

In CW5 and CW6 an unseen cell of the boundary is decided on its own, and against the truth.
Here no cell of the boundary decides alone. The price is that `z` has to be the character:
`z = exp(2πi d₁ / p) exp(3πi d₂ / p)` at every cell, with the phases of the tokens at equal
distances. A model whose features are free at each token has an angle of its own for each
token, and then the argument gives nothing beyond CW6.

### CH8. The trained model (numerical)

These numbers are measured, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust and Kernite, FP32 on a GTX 1050. The task is that of CW9. The model is the
historical real transformer of that repository: width 128, one layer, one head, a tied
embedding and readout `E`, parameter-free normalizations `N` of the mean square, its
historical optimizer with weight decay 1, 5000 updates of 512 examples. At the last position it holds

```
mid = N(E(t₂)) + N(W_o (v(t₁) ⊥ v(t₂))),    out = mid + N(FFN(N(mid))),    logits = E N(out);
```

the attention probabilities cancel in `N`. Seeds 11, 17, 23 reach 90% of the held-out states
after 1300, 1500, 1150 updates and end at 94.98%, 93.24%, 92.35%, the numbers recorded for
this model; all 4704 training states are answered right. A held-out answer is *wrap wrong*
when it is off by exactly `p`, and *residue wrong* otherwise.

| Held-out answers, of 4704 | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| Right | 4468 | 4386 | 4344 |
| Wrap wrong | 133 | 119 | 165 |
| of these, `\|b + x − p\| ≤ 3` | 115 | 100 | 119 |
| Residue wrong | 103 | 199 | 195 |
| of these, an input token with at most 5 training cells | 90 | 153 | 184 |
| Right at an input token unseen in training | 0 of 12 | 0 of 31 | 0 of 85 |

- The wrap errors lie at the boundary: of the held-out answers with `|b + x − p| ≤ 3` a third
  is off by `p` (115 of 330, 100 of 320, 119 of 350). Both kinds of error are flat from 2000
  updates on.
- The residue errors lie at rare tokens. MASH-half writes a step by one of two tokens, chosen
  by a carry of the step before, and the steps near `±p` occur in few states. By CW9, M loses
  the residue only at an unseen token; the prediction that the trained model does the same
  failed.
- 99% is at most 47 errors. It needs both kinds removed, and on seed 23 also 38 of the 85
  answers at an unseen token, where CW8 gives M none.

The hypothesis of CH5 was tested by normalizing the mean square over each pair of coordinates
`(2k, 2k + 1)`, in place of the whole width, at chosen normalizations of this model; nothing
else changes. The variants were fixed before training, in two rounds: first the final
normalization and the sites before the product block, and then the two places where the
theory has modulus one, the feature of one token (the embedding) and the output of the
product block (the FFN output; CH3, CH6).

| Pairs at | Right, seeds 11, 17, 23 | Wrap wrong at `\|b + x − p\| ≤ 3` | Residue wrong |
| --- | --- | --- | --- |
| no site (historical) | 4468, 4386, 4344 | 115, 100, 119 | 103, 199, 195 |
| the final normalization | 4404, 4406, 4306 | 118, 125, 145 | 157, 144, 207 |
| the FFN input | 4155, 3341, 4132 | 83, 50, 102 | 447, 1299, 439 |
| the embedding, the attention output and the FFN input | 3638, 3637, 3675 | 67, 71, 91 | 975, 968, 888 |
| every site | 115, 144, 143 | 0, 0, 0 | 4589, 4560, 4560 |
| the FFN output | 4417, 4418, 4292 | 114, 116, 141 | 151, 146, 236 |
| the embedding | 4407, 4461, 4263 | 104, 115, 110 | 178, 106, 306 |
| the embedding and the FFN output | 4291, 4385, 4224 | 87, 93, 114 | 309, 206, 330 |
| the FFN output and the final normalization | 4225, 2761, 4101 | 106, 43, 100 | 344, 1889, 463 |
| the embedding, the FFN output and the final normalization | 1844, 1545, 1995 | 27, 23, 36 | 2825, 3128, 2657 |

The hypothesis failed. With pairs at the final normalization, at the embedding or at the FFN
output alone the model stays above 90% on every seed, within two points of the historical
one, and keeps its wrap errors; it reaches 90% later. With pairs before the bilinear block it
loses the residue and falls below 90%, and so it does with two or three of the later sites
together; with pairs at every site it does not fit the training set. No variant has fewer
than half of the wrap errors at the boundary on every seed, and none has fewer residue errors
over the three seeds. A normalized sum of a feature of each token is a comparison in every
channel (CH1), and the block after it no longer gets kernels of rank one. That the answer can
be written with one normalized modulus (CH3, CH4, CH6) did not make this training find it:
the features of this model are free at every token, and then by CH7 one angle places
nothing.

### CH9. Scope

Proved: CH1 to CH4, CH6 and CH7 without its converse.

Not covered:

- That M or a trained model takes this readout. With `S / |S|` the fit is not a completion of
  a table: the modulus of `S` is free at every cell, seen or not. CH8 is one model and one
  recipe, in which it did not.
- That the characters of `ℤ/2p` determine the step, and which of them are of odd order: stated
  in CH3, not in Lean.
- The tokens. CH3, CH4, CH6 and CH7 are in the steps `d₁`, `d₂`. A MASH-half token of the
  second step also holds a carry of the first.
- CH5, and the converse in CH7.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, modules `StepChord`, `StepPhase`, `StepSign` and
`StepPole`, namespace `Transformer.Grokking.FlatSpectrum`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| CH1 | `exp_mul_I_add_exp_mul_I` | For real `α`, `β`. The reading for a normalized sum is not a Lean statement. |
| CH2 | `halfAngle`, `stepChord`, `stepChord_eq`, `cos_halfAngle_pos`, `cos_halfAngle_neg`, `stepChord_ne_zero`, `norm_stepChord` | `halfAngle p a b` is `φ` and `stepChord p a b` is `S`, for natural `a`, `b`; `d₁` is `(b : ℂ) - a` and `d₂` is `modStep p a b`. No wrap is `b + (a + b) % p < p`; a wrap is `p ≤ b + (a + b) % p` with `b < p`. The remark on the point between two phases is not a Lean statement. |
| CH3 | `exp_modStep_eq_stepChord`, `addChar_mul_exp_modStep` | For `a, b < p`; `ψ` is any `AddChar (ZMod p) ℂ`. `S / \|S\|` is `stepChord p a b / (‖stepChord p a b‖ : ℂ)`. The two remarks on the characters of `ℤ/2p` are not Lean statements. |
| CH4 | `stepPhaseSum`, `stepPhaseSum_eq`, `norm_stepPhaseSum`, `exp_modStep_eq_stepPhaseSum` | `stepPhaseSum p a b` is `T`. The first two statements hold for all natural `p`, `a`, `b`; the last asks for `a, b < p`. The reading for a residual connection is not a Lean statement. |
| CH6 | `stepHalfPlane`, `stepHalfPlane_eq_add`, `stepHalfPlane_eq_sin`, `sin_pos_of_lt`, `stepHalfPlane_pos_iff`, `stepHalfPlane_ne_zero`, `stepHalfPlane_div_abs` | `stepHalfPlane p a b` is `ℓ`. The two products are those of real and imaginary parts of `exp(iπ / 2p) exp(2πi d₁ / p)` and `exp(3πi d₂ / p)`, for all natural `p`, `a`, `b`; the other statements ask for `a, b < p`. The sign is `if modStep p b ((a + b) % p) < 0 then -1 else 1`, the entry of `wrapKernel`. |
| CH7 | `stepChordAt`, `stepChordAt_zero`, `halfAngle_eq_add`, `stepChordAt_eq`, `cos_halfAngle_sub_pos`, `cos_halfAngle_sub_neg`, `norm_stepChordAt`, `stepChordAt_div_norm` | `stepChordAt p a b γ` is `S_γ`. The bound is `\|γ\| < π / (2 * p) / 2`. The converse and the last paragraph of the item are not Lean statements. |

## The rate of a recipe (added)

Added on 2026-11-10 and proved in Lean. In WD1 a step of the decayed descent is
`W ← (1 − η M) W`: what the gradient of the loss does not feed shrinks by `1 − η λ` a step,
and that is how a run leaves the fits that are not the decayed minimum of WD2. This section
counts these factors for two recipes of 5000 updates on the task of CH8. The historical real
transformer of CH8 is trained by its historical optimizer with the rate 0.03 and the weight
decay 1, and reaches 90% of the held-out states. The selected main of the `diff-gpt`
repository is trained by AdamW with rates up to 0.001 and the weight decay 0.1, and does not.
In the second recipe every update reaches the end of the run with a weight of at least one
half, so its decay removes less than half of anything. On a loss without a scale a recipe
depends on its rate and its decay through their product, and at the rate 0.001 the historical
recipe is the weight decay 30.

### RR1. What the decay does along a run

The AdamW of the main model is the recurrence of AD3 in
[`../actual_network/dynamics.md`](../actual_network/dynamics.md) for every coordinate,

```
p_{t+1} = (1 − η_t ρ) p_t − η_t u_t,
```

with `u_t` the bias-corrected direction of the two moments of AD1. Unrolled,

```
p_t = p_0 · ∏_{s<t} (1 − η_s ρ)  −  Σ_{s<t} η_s u_s · ∏_{s<r<t} (1 − η_r ρ):
```

the initial value and every update, each times the decay factors of the steps after it.

If the gradient of the coordinate is zero at every update, both moments stay zero, `u_t = 0`,
and `p_t = p_0 · ∏_{s<t} (1 − η_s ρ)`. In closed loop: let the gradient be a field `G` on the
weights, taken at the current weights, with `G(w)_i = 0` at every `w`. Then the coordinate
`i` of the run is this product, whatever the other coordinates do.

### RR2. Two recipes

For `0 ≤ a_s ≤ 1`, by induction, `1 − Σ a_s ≤ ∏ (1 − a_s)`. For `x ≤ 1`,
`(1 − x)^n ≤ exp(−n x)`.

- The selected main has rates in `[0, 0.001]`, the decay `ρ = 0.1` and 5000 updates. So
  `Σ η_s ρ ≤ ½` over any of its updates, and every product of RR1 is between `½` and `1`:
  the initial value and each update reach the end of the run with at least half of their
  size, and at least half of a coordinate without gradient remains. Its schedule, 100
  updates of warm-up, 3900 at the peak and 1000 of a cosine to `10⁻⁸`, has `Σ η_s = 4.4495`,
  and the product over the run is 0.64.
- A constant rate 0.03 with the decay 1, the two numbers of the historical recipe: after 5000
  updates at most `exp(−150)` of a coordinate without gradient remains.

So in the recipe of the selected main the weights at the end are the initial weights and the
sum of the updates, up to these factors. What the run removes, it removes by gradients.

### RR3. A recipe on a loss without a scale

Let `c > 0`, and let the gradient field be homogeneous of degree `−1` at the scale `c`:

```
G(c w) = G(w) / c,
```

as the gradient of a loss with `L(c w) = L(w)` is. Then the AdamW of the main model with the
rates `c η_t`, the decay `ρ / c` and the epsilon `ε / c`, started from `c p_0`, is the run
with `η_t`, `ρ`, `ε` from `p_0` at the scale `c`: its weights are `c` times those, its first
moments are those over `c`, its second moments those over `c²`.

Proof. In one update the gradient is over `c`, so the moments are over `c` and `c²`. With `β`
and `β'` the two bias corrections,

```
(c η / β) · (m / c) / (√(v / c²) / β' + ε / c) = c · (η / β) m / (√v / β' + ε),
```

and `c η · ρ / c = η ρ`. Induction on the updates. ∎

The same holds for the historical optimizer, with the learning rate `c · lr`, the decay
`decay / c` and the epsilon `ε / c` from `c` times the first point: every point of its run is
`c` times that of the other run, its second moments are over `c²`, and its averaging weights
are the same.

So on a loss without a scale a recipe is the products `η ρ` and `η ε` and the ratio
`p_0 / η`.

### RR4. The historical recipe at the rate of the selected main

Take `c = 1/30`. The historical recipe, with the rate 0.03, the decay 1 and the epsilon
`10⁻⁸`, is the rate 0.001 with the decay 30 and the epsilon `3 · 10⁻⁷`, from a first point
thirty times smaller. The selected main has the decay 0.1 at that rate. The product `η ρ` is
0.03 in the historical recipe and `10⁻⁴` in that of the selected main, 300 times less.

### RR5. The reading for the two models

This item is a reading and a hypothesis, not a theorem.

- The two models differ in architecture and in recipe at once. RR2 and RR4 say that the
  recipes differ by more than their numbers show: the decay of the selected main does not
  act within its run, and that of the historical recipe acts within tens of updates.
- By WD2 to WD10 the weight decay is what selects the fit of M among the fits of the training
  set. A run whose decay does not act ends at the fit that its gradients reach first.

The hypothesis: the miss of the selected main on this task follows its recipe. The historical
real model, trained by the AdamW of the selected main with its rates and its decay, misses
90%; with the decay raised toward 30 at the same rates it reaches 90% again; and the selected
main gains from a raised decay. It was written down with its predictions before the runs;
RR7 holds what they found.

### RR6. Scope

Proved: RR1 to RR4.

Not covered:

- That a model has a loss without a scale. In the model of CH8 a matrix followed by a
  normalization `N` has none. The tied matrix `E` is also read without one, in
  `logits = E N(out)`, and the gate of the FFN is not homogeneous. For the selected main
  nothing is stated.
- The clip of the gradient norm in the recipe of the selected main. It has a scale of its
  own: with `G(c w) = G(w) / c` the norm of the gradient moves against a fixed threshold.
- Which coordinates of a trained model receive no gradient. RR1 and RR2 are about the decay
  term alone.
- The historical optimizer in RR1 and RR2. It takes its decay at one point and evaluates the
  model at an average of earlier points, so a coordinate without gradient does not follow
  the product. Computed for the historical recipe, not proved: 0.61 of it remains after 33
  updates, 0.14 after 100 and `8 · 10⁻⁶` after 5000, with changes of sign.
- The two optimizers against each other. RR3 is within one optimizer. That AdamW at the
  product 0.03 trains as the historical optimizer at that product is not claimed.
- RR5, beyond the numbers of RR7: three seeds of one task.

### RR7. The two models under both recipes (numerical)

These numbers are measured, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust and Kernite, FP32 on a GTX 1050. Task, split and seeds are those of CH8; a
run is 5000 updates of 512 examples, observed every 50 updates. The predictions of RR5 were
fixed before the runs.

The selected main computes its products in FP16 and TF32 on tensor cores, which this GPU does
not have. It ran in an FP32 build of the same sources, checked against the FP64 reference of
that model. Its recorded results on this task are 82.14%, 89.33%, 76.68% of the held-out
states for the seeds 11, 17, 23 with axial positions and 81.97%, 88.84%, 77.38% with sequence
positions; the FP32 build ends at 79.32%, 88.35%, 77.87% and 81.44%, 89.39%, 78.25%. The
precision of the products is not the cause of the miss.

In the historical real model the AdamW of the selected main replaces the historical optimizer,
with its schedule, betas and clip, and with the weight decay `ρ` on every coordinate. In the
selected main `ρ` is the decay of its complex matrices; its real coordinates keep 0.1 and its
frequencies 0.

| Model | Recipe | Right of 4704, seeds 11, 17, 23 | Share | First update at 90% | Training loss at the end |
| --- | --- | --- | --- | --- | --- |
| Historical | historical optimizer, rate 0.03, decay 1 | 4468, 4386, 4344 | 94.98%, 93.24%, 92.35% | 1300, 1500, 1150 | 0.039, 0.035, 0.036 |
| Historical | AdamW of the selected main, `ρ = 0.1` | 362, 464, 1419 | 7.70%, 9.86%, 30.17% | none | `1.3 · 10⁻⁶`, `1.3 · 10⁻⁶`, `1.1 · 10⁻⁶` |
| Historical | the same, `ρ = 1` | 4168, 4315, 4235 | 88.61%, 91.73%, 90.03% | none, 3700, 4500 | `7.3 · 10⁻⁵`, `9.9 · 10⁻⁵`, `1.4 · 10⁻⁴` |
| Historical | the same, `ρ = 3` | 4341, 4336, 4276 | 92.28%, 92.18%, 90.90% | 2000, 1700, 2100 | 0.0022, 0.0026, 0.0023 |
| Historical | the same, `ρ = 10` | 4436, 4439, 4310 | 94.30%, 94.37%, 91.62% | 2400, 2400, 3150 | 0.552, 0.531, 0.497 |
| Historical | the same, `ρ = 30` | 873, 846, 813 | 18.56%, 17.98%, 17.28% | none | 3.635, 3.639, 3.613 |
| Selected main | its recipe, `ρ = 0.1` | 3731, 4156, 3663 | 79.32%, 88.35%, 77.87% | none | `1.5 · 10⁻⁷`, `1.0 · 10⁻⁷`, `1.5 · 10⁻⁷` |
| Selected main | the same, `ρ = 1` | 4418, 4420, 4265 | 93.92%, 93.96%, 90.67% | 2600, 1850, 3400 | `7.0 · 10⁻⁵`, `7.2 · 10⁻⁵`, `1.0 · 10⁻⁴` |
| Selected main | the same, `ρ = 3` | 4381, 4396, 4280 | 93.13%, 93.45%, 90.99% | 1800, 1350, 1750 | 0.0012, 0.0015, 0.0016 |
| Selected main | the same, `ρ = 10` | 4478, 4459, 4356 | 95.20%, 94.79%, 92.60% | 1750, 1550, 2000 | 0.352, 0.362, 0.312 |
| Selected main | the same, `ρ = 30` | 1350, 2439, 1224 | 28.70%, 51.85%, 26.02% | none | 3.379, 3.244, 3.365 |

- The hypothesis of RR5 held on these runs. Under the recipe of the selected main the
  historical model fits the training set and answers 7.70%, 9.86%, 30.17% of the held-out
  states. With `ρ = 3` or `ρ = 10` it reaches 90% on every seed. The selected main reaches
  90% on every seed with `ρ = 1`, 3 or 10, and with `ρ = 10` it ends above the historical
  model under the historical recipe on every seed.
- At every `ρ` from 0.1 to 10 the selected main ends above the historical model on every
  seed: by 48 to 78 points at `ρ = 0.1`, by 0.4 to 1.0 at `ρ = 10`. The miss follows the
  recipe, not the architecture.
- `ρ = 30` fails in both models, with a training loss above 3. This was predicted before the
  runs from AD3: the tied matrix `E` is read without a normalization, and a coordinate of
  AdamW stays within `B / ρ`. RR4 does not apply to it (RR6).
- The first observation with more than half of the held-out answers right, seeds 11, 17, 23:
  in the historical model none at `ρ = 0.1`, then 2650, 2350, 1650 at `ρ = 1`, 1100, 1050,
  900 at `ρ = 3` and 400, 500, 400 at `ρ = 10`; in the selected main 2250, 1600, 2250 at
  `ρ = 0.1`, then 1200, 950, 1200, then 1050, 850, 1000, then 500, 550, 500. At `ρ = 3` the
  training set is fitted in full only after it (at 1750, 1750, 1300 and 1400, 1300, 1300),
  and at `ρ = 10` never (4649 to 4698 of 4704 at the end).
- One second moment for the two parts of a complex coordinate, at `ρ = 0.1`: 81.70%, 88.82%,
  84.08%, that is 2.38, 0.47 and 6.21 points above its control, and 90% on no seed.

What remains. No run has 99% at any observation. The selected main with `ρ = 10` has 57, 62,
120 residue errors, 46, 56, 106 of them at an input token with at most 5 training cells; 169,
183, 228 wrap errors, 132, 135, 152 of them at `|b + x − p| ≤ 3`; and 0, 2, 14 right answers
at the 12, 31, 85 states with an unseen token. The wrap errors do not fall with the decay: the
historical model has 133, 119, 165 under its recipe and 121, 129, 137 at `ρ = 10`. Of the wrap
errors of a run with `ρ` from 1 to 10, in either model, 57% to 74% lie at cells where the sign
fill of M (CW9) is wrong. M is wrong there at 423, 384, 433 of the 4704 cells, and at 168 of
330, 176 of 320, 167 of 350 of those with `|b + x − p| ≤ 3`.

These are three seeds of one task in an FP32 build. Whether the other tasks of the selected
main keep their results under a raised decay was not measured.

### Lean

`src/Transformer/Grokking/GradientEvidence/`, modules `RecipeDecay` and `RecipeRescale`,
namespace `Transformer.Grokking.GradientEvidence`; and
`src/Transformer/ScheduleFree/AdamW/Rescale.lean`, namespace
`Transformer.ScheduleFree.AdamW`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| RR1 | `mainAdamPath_eq_sum`, `mainMoments_zero`, `mainDirection_zero`, `mainAdamPath_zero`; in closed loop `mainAdamStep`, `mainAdamRun`, `mainAdamGradients`, `mainAdamRun_moments`, `mainAdamRun_fst`, `mainAdamRun_of_zero` | The path is `mainAdamPath` of AD3, with betas 0.9 and 0.95; the product after the update `s` is over `Finset.Ico (s + 1) t`. The statements hold for every real `ρ`, `ε` and every rates; with zero moments the direction is `0 / (0 + ε)`, which is `0` in Lean for `ε = 0` as well. `mainAdamRun` holds the weights and the two moments of all coordinates, and the gradient is taken at the weights. |
| RR2 | `one_sub_sum_le_prod`, `half_le_prod_main`, `half_le_prod_Ico_main`, `one_sub_pow_le_exp`, `historical_remainder` | The first is for a finite set of indices. The recipe of the selected main is `0 ≤ η s ≤ 1/1000` at every step with the decay `1/10`; the bounds hold over every set of at most 5000 steps, and over `Finset.Ico (s + 1) t` for `t ≤ 5000`. The numbers 4.4495 and 0.64 of its schedule are computed, not in Lean. The last statement is `mainAdamPath` with the constant rate `3/100` and the decay `1`: AdamW, not the historical optimizer. |
| RR3 | `atScale`, `mainAdamStep_atScale`, `mainAdamRun_atScale`; `scaled`, `rate_scaled`, `IsScaled`, `isScaled_initial`, `query_scaled`, `step_isScaled`, `run_isScaled`, `run_scaled` | Homogeneity is asked at the one scale `c`: `G (fun j => c * w j) i = G w i / c` for all `w`, and for the gradient oracle of the historical optimizer at every step. That a loss without a scale has such a gradient is not a Lean statement. The historical optimizer is `Transformer.ScheduleFree.AdamW.run` and needs `0 ≤ lr`. `IsScaled c p s s'` relates two states; the maximal rates are compared after a nonnegative rate has entered. The last sentence of the item is not a Lean statement. |
| RR4 | `historical`, `scaled_historical` | `historical` has ten warm-up steps, betas 0.9 and 0.98 and the default averaging powers 0 and 2. The first point thirty times smaller is the start of `run_scaled`. The comparison with the selected main is not a Lean statement. |

## A comparison of two free scores (added)

Added on 2026-11-10 and proved in Lean. CW5 and CW6 are about M: with one unseen cell at the
boundary of the wrap, the table with that cell flipped is a minimum. This section asks what
the training cells determine of the wrap for any learner that has one free score for each
token, whatever it minimizes. A table of comparisons stays one when the scores of a set of
tokens closed upwards are raised. So a held-out cell is determined exactly when a chain of
training cells links its two tokens, and at the boundary a chain is the cell itself. On the
task of CH8 the wrap of 22% to 24% of the held-out states is open, and the wrap errors of the
trained models of RR7 are among these states.

### CS1. A table of comparisons

Let the rows `i` have real scores `f_i` and the columns `j` real scores `h_j`. The cell
`(i, j)` *wraps* when `h_j ≤ f_i`. By CW7, `b + x = 2 d₁ + 3 d₂ + 2p [d₂ < 0]`, so the next
step wraps exactly when

```
p − 3 d₂ − 2p [d₂ < 0] ≤ 2 d₁:
```

the wrap is a table of comparisons in the two steps read, with the score `2 d₁` of the first
and `p − 3 d₂ − 2p [d₂ < 0]` of the second. A token determines its step, so it is a table of
comparisons in the two input tokens, and the difference of the two scores is `b + x − p`.

A learner *with free scores* knows that the answer is such a table and does not know the
scores. A seen cell gives it one inequality, `h_j ≤ f_i` or `f_i < h_j`. Scores `f'`, `h'`
*fit* the seen cells `Ω` when they compare at every cell of `Ω` as `f`, `h` do.

### CS2. Raising a set closed upwards

A set `A` of rows with a set `B` of columns is *closed upwards* along `Ω` when for every seen
cell `(i, j)`: if `h_j ≤ f_i` and `j ∈ B` then `i ∈ A`, and if `f_i < h_j` and `i ∈ A` then
`j ∈ B`. For such sets and `c ≥ 0` the scores

```
f'_i = f_i + c [i ∈ A],        h'_j = h_j + c [j ∈ B]
```

fit `Ω`.

Proof. Take a seen cell. If its row and its column are both raised, or neither, the
comparison is unchanged. If only the row is raised, the cell does not have `f_i < h_j`, as
then `j ∈ B`; so `h_j ≤ f_i ≤ f_i + c`. If only the column is raised, the cell does not have
`h_j ≤ f_i`, as then `i ∈ A`; so `f_i < h_j ≤ h_j + c`. ∎

### CS3. An open cell

Let `A`, `B` be closed upwards and `(i, j)` a cell. If `i ∉ A` and `j ∈ B`, the scores of CS2
with `c > f_i − h_j` fit `Ω` and have `f'_i < h'_j`: no wrap. If `i ∈ A` and `j ∉ B`, those
with `c ≥ h_j − f_i` fit `Ω` and have `h'_j ≤ f'_i`: a wrap. A cell at which fitting scores
give both answers is *open*: the seen cells do not determine it for a learner with free
scores.

### CS4. Chains

A *link* goes from a column `j` up to a row `i` when `(i, j)` is seen and `h_j ≤ f_i`, and
from a row `i` up to a column `j` when `(i, j)` is seen and `f_i < h_j`. A *chain* is a
sequence of links. Fitting scores do not decrease along a link, and they increase along a
link into a column. So

- a chain from the column `j` to the row `i` gives `h'_j ≤ f'_i` for all fitting scores;
- a chain from the row `i` to the column `j` gives `f'_i < h'_j` for all fitting scores.

The rows and the columns that the chains reach from one row or column are closed upwards.
With CS3:

```
some fitting scores have f'_i < h'_j    ⟺    no chain goes from the column j to the row i,
some fitting scores have h'_j ≤ f'_i    ⟺    no chain goes from the row i to the column j.
```

As `f`, `h` fit, no chain goes against the truth. So a cell is open exactly when no chain
links its two tokens.

### CS5. The boundary is open

A chain of more than one link from the column `j` to the row `i` passes a link into a
column: `h_j ≤ f_{i₁} < h_{j₂} ≤ … ≤ f_i`. So if `f_i = h_j`, the only chain from `j` to `i` is
the cell `(i, j)`, seen. A chain of more than one link from the row `i` to the column `j` is
`f_i < h_{j₁} ≤ f_{i₂} < … < h_j`, with the score of the column `j₁` strictly between `f_i`
and `h_j`. So if no score of a column lies strictly between the two, as for integer scores
with `f_i + 1 = h_j`, the only chain from `i` to `j` is the cell itself.

Hence an unseen cell with `f_i = h_j` is open, and so is an unseen cell with integer scores
and `f_i + 1 = h_j`. In the task these are the held-out states with `b + x = p` and with
`b + x = p − 1`, the two layers of CW6. There the minimum of M answers against the truth; a
learner with free scores has no reason for either answer.

What decides these cells is the distances of the scores. In CH7 the scores are the phases
`2π d₁ / p` and `3π d₂ / p` of two characters, and one angle places the wrap of all cells.

### CS6. The task (numerical)

These numbers are computed and measured, not proved: `research/diffgpt_flat_spectrum/` of the
`diff-gpt` repository, Rust on the CPU for the chains, and the runs of CH8 and RR7. The task
is that of CW9: 4704 training cells and 4704 held-out cells in a table of `194 × 370` tokens.
The chains are computed by reachability. For every open cell the two fits of CS3 are built
from the tokens that the chains reach and checked at all 4704 training cells.

| Held-out cells, of 4704 | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| The wrap open | 1077 | 1043 | 1147 |
| share | 22.90% | 22.17% | 24.38% |
| open, of those with `b + x − p ∈ {−1, 0}` | 91 of 91 | 94 of 94 | 110 of 110 |
| open, of those with `\|b + x − p\| ≤ 3` | 321 of 330 | 316 of 320 | 344 of 350 |
| open, of those with `\|b + x − p\| ≥ 12` | 257 of 3648 | 252 of 3646 | 277 of 3614 |
| The two tokens in one class of the training graph | 4692 | 4670 | 4619 |
| An input token in no training example | 12 | 28 | 70 |
| The target in no training example | 0 | 0 | 20 |

A class is a connected class of the graph that joins a first token and a second token of a
training cell; by Theorems E and G a kernel of rank one with entries of modulus one is
determined inside a class and by nothing between two classes. The completion of M in CW9
answers 4692, 4670, 4619 of the states with both tokens seen: the same counts. A token in no
training example occurs in it neither as an input at any position nor as a target.

The wrap errors of a run, and how many of them are at open cells:

| Run | Seed 11 | Seed 17 | Seed 23 |
| --- | --- | --- | --- |
| Historical model, historical recipe | 126 of 133 | 118 of 119 | 164 of 165 |
| Historical model, AdamW with `ρ = 1` | 188 of 203 | 162 of 166 | 149 of 153 |
| Historical model, `ρ = 3` | 181 of 195 | 141 of 144 | 159 of 162 |
| Historical model, `ρ = 10` | 118 of 121 | 129 of 129 | 137 of 137 |
| Selected main, `ρ = 0.1` | 139 of 161 | 136 of 142 | 170 of 192 |
| Selected main, `ρ = 1` | 151 of 155 | 187 of 187 | 233 of 243 |
| Selected main, `ρ = 3` | 215 of 226 | 197 of 200 | 249 of 259 |
| Selected main, `ρ = 10` | 165 of 169 | 182 of 183 | 224 of 228 |

- In every run of the table but the selected main with `ρ = 0.1`, 93% to 100% of the wrap
  errors are at open cells, and the 3627, 3661, 3557 cells with a determined wrap hold 0 to 15
  wrap errors.
- A run answers most open cells right: it has the wrap wrong at 11% to 22% of them, the
  selected main with `ρ = 10` at 165, 182, 224 of the 1077, 1043, 1147.
- 99% of the held-out states is at most 47 errors. The two layers of CS5 alone hold 91, 94,
  110 states, and the wrap is open at more than a thousand.
- On seed 23, 70 held-out states have an input token that no training example holds, and 20 a
  target that none holds. A model with a free row for each token has seen nothing of such a
  token, and the runs of the table answer none or one of the 70. That is more than the 47
  errors of 99%, whatever the wrap.

So the answer to why the runs stay below 99% is in the data and in the free features, not in
the optimizer: the training cells determine the wrap at three quarters of the held-out
states, the runs are right there, and what remains is decided by the distances of the scores,
which a model with a free feature for each token does not have.

### CS7. Scope

Proved: CS1 to CS5.

Not covered:

- That a trained network is a learner with free scores. CS6 measures that its wrap errors lie
  at the open cells; nothing here derives it.
- A prior on the fitting scores. Open means that both answers fit, not that they are equally
  likely; the runs have the wrap right at most open cells.
- The residue. Tokens in one class determine it, and the runs still have residue errors at
  tokens with few training cells (CH8, RR7). CS does not explain these.
- Weights tied across positions. A token unseen at one position may be seen at another or as
  a target, and the runs answer a few such states (RR7).
- A learner that is given the numbers of its tokens: *The numbers of the tokens* below.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, modules `ScoreShift` and `ScoreChain`, namespace
`Transformer.Grokking.FlatSpectrum`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| CS1 | `modStep_neg_iff_score_le`, `ScoreFit`, `scoreFit_self` | For `a, b < p`, with `d₁ = (b : ℤ) - a` and `d₂ = modStep p a b`; the two scores are integers. `ScoreFit Ω f h f' h'` is `h' j ≤ f' i ↔ h j ≤ f i` at every cell of `Ω`, for real scores on any types of rows and columns. That a token determines its step is not a Lean statement. |
| CS2 | `ScoreUpClosed`, `scoreFit_indicator` | The raised scores are `f + A.indicator fun _ => c` and `h + B.indicator fun _ => c`. |
| CS3 | `exists_scoreFit_lt`, `exists_scoreFit_le` | The constants are `\|f i - h j\| + 1` and `\|h j - f i\|`. "Open" is not a Lean definition; the two statements give the two fits. |
| CS4 | `scoreChain`, `ScoreFit.le_of_chain`, `ScoreFit.lt_of_chain`, `scoreUpClosed_chain`, `exists_scoreFit_lt_iff`, `exists_scoreFit_le_iff` | A link is a relation on `m ⊕ n` and a chain is its `Relation.ReflTransGen`, which holds the empty chain. `ScoreFit.le_of_chain` is for the scores `Sum.elim f' h'` between any two tokens. The two equivalences hold for every `Ω`, `f`, `h`, `i`, `j`. |
| CS5 | `mem_of_chain_of_eq`, `mem_of_chain_of_gap`, `exists_scoreFit_lt_of_eq`, `exists_scoreFit_le_of_add_one` | The second asks that no `h j'` lies strictly between `f i` and `h j`. The last is for integer scores `F`, `H` with `F i + 1 = H j`, cast to the reals. The reading for the states with `b + x = p` and `b + x = p − 1` is not a Lean statement. |

## The numbers of the tokens (added)

Added on 2026-11-10 and proved in Lean. In CS a learner has one free score for each token, and
the training cells leave the wrap open at 22% to 24% of the held-out states. But a token of
the task is not a free symbol. It writes a step, an integer: its *number*. And the wrap is a
half-space of the numbers of the two tokens read. This section takes the numbers as given. At
every frequency the phase of the sum is a product of one function of each number, and up to
the frequency `π / p` the wrap is a half-plane of it. Scores that are linear in the numbers
and fit the training cells answer rightly at every cell in the convex hull of the training
cells of one answer; on the task of CS6 that is all the held-out states of one split and all
but 19 and 75 of the other two. And a decay toward a fixed table leaves at that table what the
gradients do not feed. NT8 has a model trained with such a table: it is better and earlier on
every seed, the same rows at other tokens ruin it, and a token that training does not show is
still not answered.

### NT1. The phase of the sum at any frequency

By CW7, `b + x = 2 d₁ + 3 d₂ + 2p [d₂ < 0]`. So for every real `ω`

```
exp(iω (b + x)) = exp(2iω d₁) · exp(3iω d₂) · c(d₂),
c(d) = exp(2iω p) for d < 0,   c(d) = 1 otherwise.
```

In CH4 the frequency is `π / p`, where `c = 1`. At another frequency the second factor is
the phase `exp(3iω d₂)` of the number `d₂` times one of two constants, by the sign of `d₂`.
Let the *code* of a number `d` at a frequency `ν` be the pair
`(exp(iν d) [d ≥ 0], exp(iν d) [d < 0])`: its phase on the half-line of its sign, and zero on
the other. Then the phase of the sum is a product of a linear function of the code of `d₁` at
`2ω`, with the coefficients `(1, 1)`, and a linear function of the code of `d₂` at `3ω`, with
the coefficients `(1, exp(2iω p))`: it is bilinear in the codes of the two numbers, with the
same coefficients for all tokens.

### NT2. The wrap is a half-plane of it

Let `0 < ω` with `ω p ≤ π`, and let `n < 2p` be a natural number. Then

```
0 < sin(ω (n + ½ − p))    ⟺    p ≤ n.
```

Proof. For `p ≤ n ≤ 2p − 1` the angle is in `[ω / 2, ω (p − ½)]`, inside `(0, π)`. For
`n ≤ p − 1` it is in `[−ω (p − ½), −ω / 2]`, inside `(−π, 0)`. ∎

With `n = b + x`: the next step wraps exactly when `0 < sin(ω (b + x + ½ − p))`, the
half-plane `0 < Im(exp(−iω (p − ½)) z)` of the phase `z` of NT1. So at every frequency up to
`π / p` the wrap is the sign of one bilinear form in the codes of the two numbers, the same
form for all cells. At the two layers of CS5 the phase is at the distance `sin(ω / 2)` from
the edge of the half-plane.

### NT3. A decay toward a fixed table

In RR1 a coordinate follows `p ← (1 − η ρ) p − η u`. Let `c` be a fixed value for it, and add
`η ρ c` after every update:

```
p ← (1 − η ρ) p − η u + η ρ c = p − η ρ (p − c) − η u.
```

With the gradients given, `p − c` follows the recurrence of RR1 from `p₀ − c`. In closed
loop, with a gradient field `G` on all coordinates and a table `c` of fixed values: the run is
`c` plus the run, with the decay toward zero, of the field `q ↦ G(c + q)` from `p₀ − c`, and
the moments of the two runs are equal. So it is the run of the model whose weights are
`c + q`, with `q` decayed toward zero.

A coordinate whose gradient is zero at every point is

```
p_t = c + (p₀ − c) · ∏_{s<t} (1 − η_s ρ).
```

At the constant rate 0.001 with the decay 10 it is within `e⁻⁵⁰ |p₀ − c|` of `c` after 5000
updates. With a decay that acts within the run, as in RR7, what the gradients do not feed is
left at its fixed value and not at zero: the row of a table for a token that training shows
rarely is near its fixed row.

### NT4. Scores linear in the numbers

Give every row `i` a vector `φ_i` and every column `j` a vector `ψ_j` of real vector spaces:
their *numbers*. A learner *with scores linear in the numbers* has `f'_i = u(φ_i)` and
`h'_j = θ − w(ψ_j)`, with linear forms `u`, `w` and a real `θ`, all free. If its scores fit
the seen cells `Ω`, then

- at a cell whose pair `(φ_i, ψ_j)` is in the convex hull of the pairs of the seen cells with
  `h ≤ f`, it has `h'_j ≤ f'_i`;
- at a cell whose pair is in the convex hull of the pairs of the seen cells with `f < h`, it
  has `f'_i < h'_j`.

Proof. `h'_j ≤ f'_i` is `θ ≤ u(φ_i) + w(ψ_j)`, a half-space of the pair, and `f'_i < h'_j`
is the open half-space of the other side. A half-space is convex, so one that holds the seen
cells of an answer holds their convex hull. ∎

When `f` and `h` are themselves linear in the numbers, they are among these fits. So the hull
of the seen cells of one answer holds only cells of that answer, and every linear fit answers
rightly there: such a cell is not open, with or without a chain between its two tokens. The
scores of CS1 are linear in the numbers: `2 d₁` in the number `d₁` of the first token, and
`p − 3 d₂ − 2p [d₂ < 0]` in the pair `(d₂, [d₂ < 0])` of the second.

At the two layers of CS5: the states with `b + x = p` and one sign of `d₂` lie on one line of
the plane of `(d₁, d₂)`, the edge of the wrap, and those with `b + x = p − 1` on the next
parallel line, the edge of the other answer. A held-out state of such a line is in the hull
exactly when two training states of the line have it between them.

### NT5. The task (numerical)

These numbers are computed, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust on the CPU. No model is trained here. They were not predicted in writing
before they were computed. The task is that of CS6.

**The number of a token.** In the codec of the task a token `τ` of the 512 writes the step
`round((τ − 255.5) · 192 / 511)`. For the 9408 states of every split this is `d₁` at the first
input, `d₂` at the second and the step to write at the target. A step is 2.66 tokens wide, and
the tokens of one number share it.

**The hull.** The indicator `[d₂ < 0]` is 0 or 1, so a state is in the hull of NT4 exactly
when `(d₁, d₂)` is in the hull of the training states with its answer and its sign of `d₂`:
four hulls of points of the plane, computed in integers. No state, seen or not, is in a hull
of the other answer.

| Held-out states in the hull of their answer | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| of all 4704 | 4685 | 4704 | 4629 |
| of those with the wrap open in CS6 | 1058 of 1077 | 1043 of 1043 | 1098 of 1147 |
| of those with `\|b + x − p\| ≤ 3` | 315 of 330 | 320 of 320 | 314 of 350 |
| of those with an input token unseen at its position | 12 of 12 | 31 of 31 | 53 of 85 |

So with scores linear in the numbers the training cells leave open at most 19, 0, 75 of the
held-out states, against 1077, 1043, 1147 with free scores. A state outside the hulls may
still be determined; that is not computed.

**A learner that is not told the half-space.** Let the code of a number `d` hold its phases
`exp(i m ω₀ d)` for `m = 1, …, 32` and `ω₀ = 2π / 512`, on the half-line of its sign: 128
real coordinates, half of them zero. By NT1 and NT2 with `ω = ω₀`, the wrap is the sign of a
bilinear form in the codes of the two tokens, through the harmonic 2 of the first and 3 of the
second. The learner is the bilinear form of least Frobenius norm that is `+1` at the training
cells with a wrap and `−1` at the others: a kernel regression with the product of the two Gram
matrices, solved with a ridge of `10⁻⁶` of its mean diagonal. The sign of the form at the
held-out states:

| The wrap right | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| of all 4704 | 4664 (99.150%) | 4670 (99.277%) | 4642 (98.682%) |
| of those with the wrap open in CS6 | 1039 of 1077 | 1009 of 1043 | 1085 of 1147 |
| of those with `\|b + x − p\| ≤ 3` | 292 of 330 | 287 of 320 | 290 of 350 |
| of those with an input token unseen at its position | 12 of 12 | 31 of 31 | 74 of 85 |

Nine other codes and ridges, all that were computed. The highest frequency of the code is in
turns per number.

| Code | Highest frequency | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: | ---: |
| 32 harmonics of one turn in 194 numbers | 1/6 | 4591 | 4615 | 4567 |
| 64 harmonics of one turn in 512 | 1/8 | 4633 | 4650 | 4603 |
| 32 harmonics of one turn in 256 | 1/8 | 4633 | 4650 | 4603 |
| the code above with the ridge `10⁻⁹` | 1/16 | 4662 | 4668 | 4624 |
| the code above with the ridge `10⁻³` | 1/16 | 4663 | 4678 | 4644 |
| the code above without the half-lines | 1/16 | 4669 | 4667 | 4649 |
| 16 harmonics of one turn in 512 | 1/32 | 4667 | 4683 | 4649 |
| 32 harmonics of one turn in 1024 | 1/32 | 4666 | 4682 | 4649 |
| 4 harmonics of one turn in 512 | 1/128 | 4670 | 4666 | 4657 |

- The models of RR7 that generalize have the wrap wrong at 119 to 259 held-out states, and
  the best run, the selected main with `ρ = 10`, at 169, 183, 228. This learner has it wrong
  at 40, 34, 62, and at 21 to 137 over all the codes.
- It has the wrap right at the states with a token unseen at its position: at all of them on
  two splits and at 74 of 85 on the third. A model of RR7 answers 0 to 14 of these states, most
  of the others with a wrong residue, which this learner does not answer.
- Over the three splits the codes with the highest frequencies 1/6 and 1/8 answer least, and
  those with 1/32 and 1/128 most.

### NT6. The reading for a model

This item is a reading and a hypothesis, not a theorem.

The models of RR7 have one free row of the tied table `E` for each token, decayed toward
zero. By NT3 a decay toward a fixed table `Q` leaves the row of a rare token near its row of
`Q`, and the run is that of the model with the table `Q + Δ`, `Δ` decayed toward zero. Let the
row of `Q` for a token hold the code of NT5 of its number. By NT1 and NT2 the wrap is then the
sign of one bilinear form in the fixed parts of the two rows read.

The hypothesis: the historical model of RR7, trained by the AdamW of the selected main with
`ρ = 3` and `ρ = 10` and with its tied table decayed toward this `Q`, has fewer wrap errors at
the open cells of CS6 than with the decay toward zero, and answers states with a token unseen
at its position. With the rows of `Q` permuted among the tokens it does neither. The runs, the
table and the predictions were written down before the runs as stage 7 of
`research/diffgpt_flat_spectrum/PLAN.txt`; NT8 holds what they found.

A second table, added on 2026-11-10 before any run of stage 7. The code of NT5 has one turn of
its first harmonic in 512 numbers. Its harmonics are no characters of `ℤ/p`: it holds the
numbers and not the modulus. Let the turn be `2p` instead, so that the harmonic `m` of a number
`d` is the phase `exp(i π m d / p)`. For even `m` this is a character of the residue of the
step: a step and its other wrap, `d` and `d − p`, have one phase (CW1). For odd `m` the two
have opposite phases; `m = 1` is the half angle of CH1, the frequency `π / p` at which NT2
still gives the wrap as a half-plane. This table holds the modulus as well as the numbers.

The second hypothesis: with the tied table decayed toward this table, the same model under the
same recipes has fewer residue errors than with the decay toward zero, because the row of a
token that training shows rarely is then near characters that the frequent tokens share; and
it answers states with a token unseen at its position. With the rows permuted it does neither.
It was written down with its predictions in the same stage 7, before the runs.

### NT7. Scope

Proved: NT1 to NT4.

Not covered:

- That a trained network with a table of codes is a learner with scores linear in the
  numbers, or the learner of NT5. NT4 is about one family of fits; a network has others.
- The cells outside the hulls. NT4 does not say that they are open; the count of NT5 is an
  upper bound of the open cells.
- The residue and the token to write. The residue of the step to write is that of
  `d₁ + d₂`, a product of one character of each number at the frequency `2π / p`. Nothing here
  is stated or computed for a learner of the residue with the numbers, and the learner of NT5
  answers the wrap alone.
- The codec. That a token writes its number is computed for the task (NT5); it is not a Lean
  statement.
- The clip of the gradient norm of the recipe, as in RR6: it is not in the run of NT3.
- NT6, two hypotheses.

### NT8. The models with a table of numbers (numerical)

These numbers are measured, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust and Kernite, FP32 on a GTX 1050. The model is the historical model of RR7
under the AdamW of the selected main with `ρ = 3` and `ρ = 10`. The update of NT3 is applied
to its tied table and to nothing else; the initialization, the sampled examples and the budget
are those of RR7. A row of `Q` is 0.224 for `ρ = 3`, or 0.116 for `ρ = 10`, times the code of
the number of its token: 32 harmonics on the half-line of the sign. The predictions were fixed
before the runs.

| Table `Q` | `ρ = 3`: right of 4704, seeds 11, 17, 23 | Share | `ρ = 10`: right | Share |
| --- | --- | --- | --- | --- |
| Zero (RR7) | 4341, 4336, 4276 | 92.28%, 92.18%, 90.90% | 4436, 4439, 4310 | 94.30%, 94.37%, 91.62% |
| The code of NT5, turn 512 | 4409, 4361, 4354 | 93.73%, 92.71%, 92.56% | 4546, 4533, 4411 | 96.64%, 96.36%, 93.77% |
| The code with the turn `2p` | 4549, 4551, 4431 | 96.70%, 96.75%, 94.20% | 4469, 4565, 4351 | 95.00%, 97.05%, 92.50% |
| The rows of the first at other tokens | 3445, 3745, 3395 | 73.24%, 79.61%, 72.17% | 680, 511, 641 | 14.46%, 10.86%, 13.63% |
| The rows of the second at other tokens | 3247, 3317, 3038 | 69.03%, 70.51%, 64.58% | 1503, 2769, 1200 | 31.95%, 58.86%, 25.51% |

The errors of the first three rows. A rare token is here an input token with 1 to 5 training
cells at its position, and the last column counts the answered states among the 12, 31, 85
with a token unseen at its position.

| Table `Q` | Wrap errors at the open cells of CS6, `ρ = 3` | `ρ = 10` | Residue errors at rare tokens, `ρ = 3` | `ρ = 10` | Unseen tokens answered, `ρ = 3` and `ρ = 10` |
| --- | --- | --- | --- | --- | --- |
| Zero | 181, 141, 159 | 118, 129, 137 | 129, 151, 170 | 72, 71, 123 | 0, 0, 1 and 0, 0, 0 |
| Turn 512 | 135, 129, 136 | 121, 111, 144 | 65, 74, 95 | 17, 22, 56 | 0, 1, 4 and 0, 3, 1 |
| Turn `2p` | 97, 80, 95 | 72, 76, 81 | 24, 30, 61 | 68, 22, 82 | 0, 1, 5 and 1, 2, 2 |

- The table acts as numbers. With its rows at their tokens the model ends above the decay
  toward zero in every arm and on every seed, by 25 to 215 answers. With the same rows at
  other tokens it ends 591 to 3928 answers below. It first has 90% at 500 to 950 updates, in
  place of 1700 to 3150.
- The first hypothesis of NT6 held in half. With the code of NT5 the wrap errors at the open
  cells are fewer at `ρ = 3` on every seed, and at `ρ = 10` on one seed of three. The states
  with a token unseen at its position are not answered: 0 to 4 of them.
- The second held in half too. With the turn `2p` the residue errors at `ρ = 3` are 57, 73,
  178 against 168, 224, 266, and those at rare tokens fall by 64% to 81%. At `ρ = 10` they
  are fewer on one seed of three; on the other two, 108 of 163 and 108 of 272 are off by one,
  the neighbouring step. The wrap errors at the open cells fall by 39% to 46% at both decays.
  The states with an unseen token are not answered: 0 to 5 of them.
- So the model does not compute from the code alone. A token with a few training cells gains
  from the table and a token with none does not: the model needs what the gradients add to a
  row. NT4 is about scores linear in the numbers, and these networks are not such learners.
- No run reaches 99%; the most is 97.05%. The best arm, the turn `2p` with `ρ = 3`, ends at
  96.70%, 96.75%, 94.20%, above every run of RR7, with 98, 80, 95 wrap errors.

What this does not show: the selected main, whose table is complex and has another layout; a
table on a grid of scale 2 (CC7) or with level tokens; and another amplitude, number of
harmonics or decay of the table, which were fixed before the runs and not tuned.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, modules `StepNumbers` and `ScoreNumbers`, namespace
`Transformer.Grokking.FlatSpectrum`, and `src/Transformer/Grokking/GradientEvidence/`, module
`RecipeToward`, namespace `Transformer.Grokking.GradientEvidence`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| NT1 | `exp_sum_eq_mul_halfLine` | For `a, b < p` and a real `ω`, with `d₁ = (b : ℂ) - a` and `d₂ = modStep p a b`; the constant is an `if` on `modStep p a b < 0`. The code of a number and "bilinear in the codes" are not Lean definitions. |
| NT2 | `sin_pos_iff_le`, `modStep_neg_iff_sin_pos` | For `n < 2 * p`, `0 < ω` and `ω * p ≤ π`. The second is the step after `x = (a + b) % p`, for `b < p` and any `a`. The half-plane of the phase is not a Lean statement. |
| NT3 | `mainAdamPathToward`, `mainAdamPathToward_succ`, `mainAdamPathToward_eq`, `mainAdamPathToward_zero_value`, `mainAdamPathToward_zero`, `mainAdamPathToward_remainder`, `mainAdamStepToward`, `mainAdamRunToward`, `mainAdamRunToward_eq`, `mainAdamRunToward_of_zero` | The path is one coordinate with given gradients, as `mainAdamPath`; the run is all coordinates with a gradient field, as `mainAdamRun` of RR3, with a fixed value `c i` for every coordinate. `mainAdamRunToward_eq` gives the weights as `c i +` those of the run of `fun q => G fun j => c j + q j` from `p0 - c`, and the two moments as equal. The remainder is for the constant rate `1 / 1000`, `ρ = 10`, 5000 updates and zero gradients. |
| NT4 | `seenLePairs`, `seenLtPairs`, `ScoreFit.le_of_mem_convexHull`, `ScoreFit.lt_of_mem_convexHull`, `scoreFit_linear_iff` | `φ : m → E` and `ψ : n → F` into real vector spaces, `u : E →ₗ[ℝ] ℝ`, `w : F →ₗ[ℝ] ℝ`; the hull is `convexHull ℝ` in `E × F` of the image of the seen cells of one answer. `scoreFit_linear_iff` is for scores `u₀ (φ i)` and `θ₀ - w₀ (ψ j)` and a cell in the hull of either answer. That the scores of CS1 are of this form, and the reading for the two layers, are not Lean statements. |

## The encoder: steps and levels (added)

Added on 2026-11-10 and proved in Lean. The task of CH8 is an additive recurrence modulo `p`
seen through a difference encoding: the model reads the steps between the values and writes
a step. CW, CS and NT are about what this asks of a model: a residue, a wrap, and tokens that
are numbers. This section sets beside it a level encoding of the same task, in which the
model reads the values and writes the value. A level is determined by its residue, and a step
is not. The level task is the kernel of `a + b` of Proposition 2 and nothing else, and M
recovers it under Theorem E. A level is read at `p` states, and a step `d` at `p − |d|`. On
the task of CS6 the training cells of the level encoding satisfy Theorem E on all three
splits. Trained on level tokens, the models of RR7 reach 99% of the held-out states under
every recipe, and five of six end at 99.96% or above (EN6).

### EN1. A level is its residue; a step is not

Let `a, b < p` be two values, `x = (a + b) % p` the next and `(b + x) % p` the one to write.
With levels the model reads `b` and `x` and writes `y = (b + x) % p`. With the character `ψ`
of Proposition 2,

```
ψ(y) = ψ(b + x) = addKernel_{b, x},
```

the kernel of `a + b`, and no other level `y' < p` has this phase, as `ψ` is injective on the
residues. So the level task is one kernel of rank one with entries of modulus one.

With steps the model writes `d₃ = modStep p b x`, and its residue is that of `b` (CW1). After
a value `0 < b < p` the step is `b` from `x = 0` and `b − p` from `x = p − 1`: two steps with
one residue. So the step task has, beside its kernel of rank one, the wrap of CW6, and the
wrap is the comparison of CS1.

### EN2. How many states read a token

Among the `p²` pairs `(a, b)` of values below `p`:

- the first step `b − a` is `d` at `p − |d|` pairs;
- the second step `modStep p a b` is `d` at `p − |d|` pairs;
- the two levels read, `b` and `a + b`, run over all pairs of levels once, so a level is the
  first level read at `p` pairs and the second at `p` pairs.

A step with `|d| = p − k` is read at `k` states, at each of the two positions: the steps near
`±p` are rare tokens of any difference encoding, whatever the split. Every level is read
equally often.

### EN3. The task (numerical)

These numbers are computed, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust on the CPU. No model is trained here. The states, the split and the value to
write are those of CS6. The step tokens are those of the task. The level tokens are those of
the same codec with the difference operator of order 0 in place of order 1: a level `x` is the
token `256 + floor(s x)` with the scale `s` the binary64 number below `8 / 3`, so the levels
`0, 1, 2, 3, …, 96` are the tokens `256, 258, 261, 263, …, 511`. The example of a state reads
the tokens of `b` and `x` and writes that of `(b + x) % p`.

| Of a split | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| Step tokens at the first and the second input | 194, 370 | 194, 370 | 194, 370 |
| of them with at most 5 training cells at their position | 142 | 144 | 150 |
| Held-out states with such a step token | 392 | 387 | 501 |
| Held-out states with a step token unseen at its position | 12 | 31 | 85 |
| Held-out states with the wrap open (CS6) | 1077 | 1043 | 1147 |
| Held-out states with both step tokens in one class (CS6) | 4692 | 4670 | 4619 |
| Level tokens at each input and as the target | 97 | 97 | 97 |
| Least training cells of a level token at an input | 42 | 41 | 36 |
| Classes of the training graph of the level tokens | 1 | 1 | 1 |
| Held-out states with both level tokens in one class | 4704 | 4704 | 4704 |
| The training cells hold a permutation of the 97 levels | yes | yes | yes |

The codec of the task writes a step on a grid that alternates by half a token, so a step at
the second input has two tokens (370 for 193 steps), which halves the cells of a token there.

On every split the training cells of the level encoding connect the 97 rows with the 97
columns and hold the graph of a permutation: the two hypotheses of Theorem E. So M recovers
the kernel of the level task from the training cells, and by EN1 the level to write at every
held-out state. With steps the kernel of the residue is determined inside a class, at 4692,
4670, 4619 held-out states, and the wrap is a second question, open at more than a thousand.

### EN4. The reading for a model

This item is a reading and a hypothesis, not a theorem.

By CS6 and RR7 the models with step tokens stay below 99% by wrap errors at the open cells
and by residue errors at rare tokens. EN1 and EN2 say that both come with the encoding: a
level has no wrap, and no level is rare. The hypothesis: the models of RR7 whose decay acts
within the run, trained in the same way on the level tokens of EN3, reach 99% of the held-out
states. It was written down with its predictions before the runs, as stage 8 of
`research/diffgpt_flat_spectrum/PLAN.txt`; EN6 holds what they found.

### EN5. Scope

Proved: EN1 and EN2.

Not covered:

- That a trained network is M. Theorem E is about the minimum of the nuclear norm; EN4 is a
  hypothesis about networks.
- The token of a level and of a step. EN1 and EN2 are about the values; that the tokens of
  EN3 are one for a level, and one or two for a step, is computed for the task.
- Series that are not residues. For a series with a trend a difference encoding has a reason
  that this task does not show; nothing here compares the two encodings there.
- The split. EN2 counts states, not training cells; the counts of EN3 are for three splits by
  cycles of the recurrence.

### EN6. The models on level tokens (numerical)

These numbers are measured, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust and Kernite, FP32 on a GTX 1050. The models, recipes, initializations, sampled
examples and budget are those of RR7; the tokens are the level tokens of EN3, and nothing else
is changed. The predictions were fixed before the runs.

| Model | Recipe | Right of 4704, seeds 11, 17, 23 | Share | First update at 90% | First at 99% | Share with steps (RR7) |
| --- | --- | --- | --- | --- | --- | --- |
| Historical | historical optimizer, rate 0.03, decay 1 | 4666, 4682, 4625 | 99.19%, 99.53%, 98.32% | 350, 200, 400 | 850, 550, 500 | 94.98%, 93.24%, 92.35% |
| Historical | AdamW of the selected main, `ρ = 0.1` | 4704, 4704, 4704 | 100%, 100%, 100% | 300, 300, 300 | 350, 350, 350 | 7.70%, 9.86%, 30.17% |
| Historical | the same, `ρ = 3` | 4702, 4703, 4702 | 99.96%, 99.98%, 99.96% | 250, 250, 250 | 300, 300, 300 | 92.28%, 92.18%, 90.90% |
| Historical | the same, `ρ = 10` | 4702, 4704, 4704 | 99.96%, 100%, 100% | 250, 200, 250 | 250, 250, 1100 | 94.30%, 94.37%, 91.62% |
| Selected main | its recipe, `ρ = 0.1` | 4704, 4704, 4704 | 100%, 100%, 100% | 300, 300, 300 | 300, 350, 300 | 79.32%, 88.35%, 77.87% |
| Selected main | the same, `ρ = 10` | 4704, 4704, 4704 | 100%, 100%, 100% | 350, 300, 250 | 500, 300, 300 | 95.20%, 94.79%, 92.60% |

- The hypothesis of EN4 held for the three arms of AdamW whose decay acts: 99% on every seed,
  and 99.96% or more at the end. It failed for the historical recipe on one seed. That recipe
  has 99% on every seed, at 850, 550, 500 updates, and ends at 99.19%, 99.53%, 98.32% with
  4683, 4689, 4677 of the 4704 training states right: it loses its training fit. By RR4 it
  is the decay 30, at which the step runs of RR7 do not fit their training set.
- A prediction made with the hypothesis failed: that the recipe would still matter. With
  level tokens `ρ = 0.1` ends at 100% on every seed in both models. The selected main under
  its own recipe has all 4704 held-out states right at every observation from its first 99%,
  at 300 to 350 updates, to the end. So the miss of RR7 under `ρ = 0.1` needs the step tokens:
  it is the encoding and the weak decay together.
- No answer of any run, at any of its observations, is off by `p`: a level has no wrap.
- In every run half of the held-out answers are right at 200 to 300 updates, and no run fits
  its training set before that. With step tokens and `ρ = 0.1` the historical model fits its
  training set at 700 to 2250 updates and has 362, 464, 1419 held-out answers right at 5000.
- The runs are not monotone: after its first 99% a run falls as low as 4533 of 4704 (the
  selected main with `ρ = 10`, seed 11) before it ends at 4704.

What this does not show: which part of the step encoding keeps the models of RR7 below 99%,
the comparison of CS1, the steps that few states read (EN2), or the two tokens of a step.
*The carry of the codec* and *The decoder holds the level* below separate them.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, module `StepLevel`, namespace
`Transformer.Grokking.FlatSpectrum`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| EN1 | `addKernel_natCast`, `stdAddChar_eq_addKernel_iff`, `modStep_zero_and_pred` | The levels are naturals cast to `ZMod p`, and `ψ` is `ZMod.stdAddChar`; the second is for `y < p` and any `b`, `x`. `modStep_zero_and_pred` is for `0 < b < p`; the residue of both steps is `intCast_modStep` of CW1. That M recovers the kernel is `forall_isMinOn_addKernel_iff_connected` of Theorem E, under its two hypotheses. |
| EN2 | `card_filter_sub_eq`, `card_filter_modStep_eq`, `levelsRead`, `card_levelsRead_fst`, `card_levelsRead_snd` | The pairs are `Finset.range p ×ˢ Finset.range p`, and the count is `p - d.natAbs` in the naturals, zero for `\|d\| ≥ p`. `levelsRead p` is the equivalence `(a, b) ↦ (b, a + b)` of `ZMod p × ZMod p`; the two counts are the cardinalities of its fibers over a first and a second level. |

## The carry of the codec (added)

Added on 2026-11-10 and proved in Lean. EN3 counts 370 tokens at the second input of the task
for 193 steps. This section says where the second token of a step comes from. A codec that
writes steps on a grid, and keeps its reconstruction on the grid, carries its rounding error
to the next row. Then the token of a step is one of two, and which of the two is a bit of the
steps before it. On a grid that holds the values nothing is carried and a step is one token.
The grid of the task has 2.66 tokens for a step; a grid of scale 2 is a change of one number
of the codec, its domain. On the task it removes two thirds of the rare tokens and one third
of the open wraps of CS6. What remains is the difference encoding itself, EN1 and EN2. CC7 has
the models trained on such a grid. With the tokenizer of the task two of three arms have fewer
residue errors on every seed; the selected main under its own recipe passes 90%; no run
reaches 99%.

### CC1. The tokens of a codec that carries

Let `s > 0` be the scale of the grid, in tokens per unit of the series. The codec anchors the
grid at the value before the window. After steps that sum to `n` the series is `n` above the
anchor, and its level on the grid is `⌊s n + θ⌋`, where `θ ∈ [0, 1)` is the offset of the
grid at that row. The token of the next step `d` is, up to a constant, the difference of two
levels:

```
carriedStep s θ θ' n d = ⌊s (n + d) + θ'⌋ − ⌊s n + θ⌋.
```

So the reconstruction stays on the grid, within one level of the series at every row. A codec
that rounded every step by itself would write one token for a step, and its errors would add
up along the rows.

The codec of the task shifts its grid by half a token at every other row: `θ = 1/2` at the
anchor and at the even rows, `θ = 0` at the odd rows. Its three tokens at a state with the
steps `d₁, d₂, d₃` are

```
τ₁ = 256 + carriedStep s (1/2) 0 0 d₁           = 256 + ⌊s d₁⌋,
τ₂ = 255 + carriedStep s 0 (1/2) d₁ d₂,
τ₃ = 256 + carriedStep s (1/2) 0 (d₁ + d₂) d₃,
```

with `s = 511 / 192`. This is read from the source of the codec and checked at all 9408 states
of the task (CC4); it is not proved about the codec.

### CC2. One step, two tokens, one bit

For all real `s, θ, θ'` and integers `n, d`,

```
carriedStep s θ θ' n d = ⌊s d + θ' − θ⌋ + [1 ≤ fract(s n + θ) + fract(s d + θ' − θ)].
```

So a step has at most two tokens, `⌊s d + θ' − θ⌋` and the next, and the choice is one bit of
`n`, the sum of the steps before it: whether the two fractional parts reach `1` together.
The bit is not a function of the step, and it is not what the task asks for: the residue of
the step to write is that of `d₁ + d₂` (CW1), and its wrap is a comparison of `2 d₁` with a
function of `d₂` (CW7).

### CC3. A grid that holds the values

If `s n` and `s d` are integers, then for any offsets `θ, θ' ∈ [0, 1)`

```
carriedStep s θ θ' n d = s d.
```

So on a grid that holds the values of the series nothing is carried: a step is one token at
its row, whatever came before. An integer scale is such a grid for every integer series.

The scale of the task is not: with `s = 511 / 192` and the offsets `0` and `1/2` of the second
token, the step `1` is written `3` after `n = 0` and `4` after `n = 3`.

### CC4. The task (numerical)

These numbers are computed, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust on the CPU. No model is trained here. They were not predicted in writing
before they were computed. The states, the split and the value to write are those of CS6.

On the grid of the task the three formulas of CC1 give the tokens of all 9408 states; at a tie
of the grid, `s n` an integer under a floor or a half-integer under a round, the binary64
arithmetic of the codec decides, and either neighbour is accepted. Of the 193 second steps 177
have two tokens and 16 have one; the first step has one token, except `0`, which has two by
such a tie.

A domain with the bound 127.75 in place of 96 gives the same tokenizer the scale 2. Then a
step `d` is the token `256 + 2 d` at the first input and at the target and `255 + 2 d` at the
second input, at every state. With the tokenizer that has one grid at every row and the bound
127.5 it is `256 + 2 d` at every position.

| Of a split | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| **The grid of the task, 2.66 tokens for a step** | | | |
| Tokens at the first input, at the second, as the target | 194, 370, 204 | 194, 370, 204 | 194, 370, 204 |
| Input tokens with at most 5 training cells at their position | 142 | 144 | 150 |
| Held-out states with such a token | 392 | 387 | 501 |
| Held-out states with a token unseen at its position | 12 | 31 | 85 |
| Held-out states with the wrap open (CS6) | 1077 | 1043 | 1147 |
| Held-out states with both tokens in one class | 4692 | 4670 | 4619 |
| **A grid of scale 2** | | | |
| Tokens at the first input, at the second, as the target | 193, 193, 193 | 193, 193, 193 | 193, 193, 193 |
| Input tokens with at most 5 training cells at their position | 44 | 48 | 56 |
| Held-out states with such a token | 112 | 147 | 222 |
| Held-out states with a token unseen at its position | 0 | 0 | 30 |
| Held-out states with the wrap open (as in CS6) | 703 | 731 | 799 |
| Held-out states with both tokens in one class | 4704 | 4703 | 4674 |

The two tokenizers of scale 2 have the same counts; they differ in the tokens that the three
positions share, 386 tokens in all against 193.

So the carry of the task's grid costs about a hundred rare tokens, 240 to 280 held-out states
that read one, and 312 to 374 open wraps. Without it the wrap is still open at 15% to 17% of
the held-out states for a learner with a free score for each token, and 112 to 222 held-out
states still read a token with at most 5 training cells: these are the steps near `±p` of EN2
and the comparison of CS1, which come with any difference encoding of the task.

### CC5. The reading for a model

This item is a reading and a hypothesis, not a theorem.

By RR7 the residue errors of the models are at tokens with few training cells, and by CC4 two
thirds of those tokens are there by the carry. The hypothesis: the models of RR7 whose decay
acts within the run, trained in the same way on the steps of a grid of scale 2, make fewer
residue errors than on the grid of the task, on every seed. Under the registered decode they
are not expected to reach 99%: the wrap is still open at 703, 731, 799 held-out states. With
the decoder of *The decoder holds the level* it is another question (DL4). The runs and the
predictions were written down before the runs as stage 10 of
`research/diffgpt_flat_spectrum/PLAN.txt`; CC7 holds what they found.

### CC6. Scope

Proved: CC2 and CC3.

Not covered:

- The codec. CC1 describes it, and the description is checked at the states of the task; the
  binary64 arithmetic of the codec at a tie of the grid is not described.
- That both tokens of a step occur. CC2 bounds the tokens of a step by two. That 177 of the
  193 second steps of the task have two is computed (CC4); CC3 proves it for one step.
- Series that are not on a grid. A scale `u / v` in lowest terms holds the multiples of `v`,
  so a series of all integers is on the grid only for `v = 1`; real values are on no grid, and
  some rounding is then carried by every codec of this kind. Nothing here says what the two
  tokens of a step cost a model on such a series.
- A model. CC4 counts tokens and cells; CC5 is a hypothesis about networks.

### CC7. The models on a grid of scale 2 (numerical)

These numbers are measured, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust and Kernite, FP32 on a GTX 1050. The models, recipes, initializations, sampled
examples and budget are those of RR7; the tokens are the steps on the two grids of scale 2 of
CC4, and nothing else is changed. The predictions were fixed before the runs. The decode is
the registered one; DL6 has the same runs decoded with the domain.

| Model | Recipe | Share on the grid of the task (RR7) | Scale 2, the tokenizer of the task | Scale 2, one grid at every row |
| --- | --- | --- | --- | --- |
| Historical | historical optimizer, rate 0.03, decay 1 | 94.98%, 93.24%, 92.35% | 96.64%, 96.79%, 94.52% | 94.45%, 94.03%, 95.30% |
| Historical | AdamW of the selected main, `ρ = 0.1` | 7.70%, 9.86%, 30.17% | 69.62%, 69.86%, 33.23% | 71.47%, 84.06%, 81.16% |
| Historical | the same, `ρ = 10` | 94.30%, 94.37%, 91.62% | 92.73%, 86.84%, 92.05% | 81.68%, 79.32%, 82.48% |
| Selected main | its recipe, `ρ = 0.1` | 79.32%, 88.35%, 77.87% | 90.67%, 91.92%, 93.24% | 95.79%, 96.15%, 93.54% |
| Selected main | the same, `ρ = 10` | 95.20%, 94.79%, 92.60% | 97.07%, 96.49%, 95.09% | 94.90%, 96.43%, 95.47% |

The wrap errors and, after "and", the residue errors of the same runs, seeds 11, 17, 23:

| Model and recipe | Grid of the task | Scale 2, the tokenizer of the task | Scale 2, one grid at every row |
| --- | --- | --- | --- |
| Historical, its recipe | 133, 119, 165 and 103, 199, 195 | 138, 124, 185 and 20, 27, 73 | 142, 131, 159 and 119, 150, 62 |
| Historical, `ρ = 10` | 121, 129, 137 and 147, 136, 257 | 74, 50, 87 and 268, 569, 287 | 24, 19, 34 and 838, 954, 790 |
| Selected main, `ρ = 0.1` | 161, 142, 192 and 812, 406, 849 | 195, 178, 166 and 244, 202, 152 | 152, 141, 207 and 46, 40, 97 |
| Selected main, `ρ = 10` | 169, 183, 228 and 57, 62, 120 | 126, 110, 162 and 12, 55, 69 | 84, 93, 156 and 156, 75, 57 |

- The hypothesis of CC5 held with the tokenizer of the task for two of its three arms. The
  historical recipe has 20, 27, 73 residue errors against 103, 199, 195, and the selected main
  with `ρ = 10` has 12, 55, 69 against 57, 62, 120. Both end 1.7 to 3.6 points higher on every
  seed.
- It failed for the historical model under AdamW with `ρ = 10`, on both grids. That run no
  longer fits its training set: 4596, 4495, 4614 and 4382, 4234, 4417 of its 4704 training
  states are right at its end. And 67% to 93% of its residue errors are off by one, the token
  of the neighbouring step; on the grid of the task that is 5% to 11%. With one grid at every
  row the hypothesis failed in three more cells of six.
- Not predicted: the selected main under its own recipe, `ρ = 0.1`, ends above 90% of the
  held-out states on every seed on either grid of scale 2, with no change of the model or of
  the recipe. It first has 90% at 3150 to 4250 updates with the tokenizer of the task and at
  1850 to 2050 with one grid at every row. So 3.6 to 16.5 points of its miss in RR7 need the
  grid of the task.
- As CC5 expects, no run reaches 99%; the most is 97.07%. The wrap errors stay. The three best
  arms are the historical recipe and the selected main with `ρ = 10`, both with the tokenizer
  of the task, and the selected main with `ρ = 0.1` and one grid at every row. They have 110
  to 207 wrap errors, against 119 to 228 on the grid of the task. On seeds 11 and 17 their
  right answers and wrap errors together are 98.8% to 99.7% of the held-out states: the
  residue is held there, and what is missing is the wrap.
- On seed 23, 30 held-out states read a step that training does not show at its position
  (CC4). The runs answer none of them, except one run that answers one.
- Neither tokenizer of scale 2 is better in every arm.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, module `StepCarry`, namespace
`Transformer.Grokking.FlatSpectrum`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| CC1 | `carriedStep` | The definition, for real `s`, `θ`, `θ'` and integers `n`, `d`. That the tokens of the codec are `256 + carriedStep s (1/2) 0 0 d₁`, `255 + carriedStep s 0 (1/2) d₁ d₂` and `256 + carriedStep s (1/2) 0 (d₁ + d₂) d₃` is not a Lean statement. |
| CC2 | `floor_add_eq_ite`, `carriedStep_eq`, `carriedStep_eq_or` | The first is `⌊x + y⌋ = ⌊x⌋ + ⌊y⌋ + [1 ≤ fract x + fract y]` for real `x`, `y`. The bit is `if 1 ≤ Int.fract (s * n + θ) + Int.fract (s * d + θ' - θ) then 1 else 0`. |
| CC3 | `carriedStep_of_int`, `carriedStep_intCast`, `carriedStep_two` | The first takes integers `j`, `k` with `s * n = j` and `s * d = k`, and `0 ≤ θ, θ' < 1`; the second is an integer scale `m` and gives `m * d`. The third is `s = 511 / 192`, `θ = 0`, `θ' = 1 / 2`, `d = 1`, with `n = 0` and `n = 3`. |

## The decoder holds the level (added)

Added on 2026-11-10 and proved in Lean. CW6 makes the wrap of the step to write a comparison,
CS counts where the training cells leave it open, and RR7 finds most of the errors of the best
run there. This section observes that the wrap is not the model's to learn. The decoder adds
the step to the level before it, and it holds that level; the model reads steps and does not.
Of the integers with the residue of the step, one keeps the series in its domain `[0, p)`, and
it is the step. So a decoder that admits only values of the domain asks the model for the
residue, and it makes no wrap error. On the task every wrap error of every run with step
tokens decodes outside the domain. Decoded with the domain, the runs have 63% of their wrap
errors right, and none reaches 99% on the grid of the task (DL6).

### DL1. The wrap is the domain

Let `x` be the level before the row to write and `b` the value before it, so that the step to
write is `modStep p b x` (CW1). For every integer `d`,

```
d = modStep p b x   ⟺   d ≡ b (mod p)  and  0 ≤ x + d < p.
```

So the step is the only integer of its residue that keeps the series in `[0, p)`. The others,
`modStep p b x + k p` with `k ≠ 0`, lead outside: a wrong wrap is `k = ±1`. The wrap of CW6,
`p ≤ b + x`, is thus a function of the residue `b` and of the level `x`. The model of a
difference encoding does not read `x`. It reads the steps `d₁` and `d₂`, and in them the wrap
is the comparison of CW7. The decoder holds `x`.

### DL2. A decoder with a domain

A decoder with the domain `[0, p)` admits a token when its value, the level plus the step
that the token writes, is in the domain, and it answers with the best admitted token. By DL1:

- An admitted answer whose step has the residue of the truth is the truth. This decoder makes
  no wrap error, whatever the model.
- What it asks of the model is the residue of the step. Every character of that residue is a
  kernel of rank one in the two steps read, with entries of modulus one (CW1), and such a
  kernel is determined by the training cells inside a class of their graph (Theorems E and G).

The tokens, the model and its training are as before; what changes is how its scores are read.
A level encoding has this for nothing, as a level cannot leave the domain (EN1). The domain is
a declaration about the series, as its grid is. It is not read from the held-out states.

### DL3. The task (numerical)

These numbers are computed from the recorded answers of the runs of RR7 and of the earlier
runs of the same study, `research/diffgpt_flat_spectrum/` of the `diff-gpt` repository; no
model is run here. The domain is `[0, 96]`.

- The 66 runs with step tokens have 8251 wrap errors at their ends. Every one decodes outside
  the domain, and no right answer does.
- Of the residue errors some decode outside the domain as well. So a run decoded with the
  domain has at most its right answers, its wrap errors and these right:

| Model and recipe (RR7) | Right, seeds 11, 17, 23 | Wrap errors | Residue errors | of them outside the domain | At most right with the domain |
| --- | --- | --- | --- | --- | --- |
| Historical, its recipe | 4468, 4386, 4344 | 133, 119, 165 | 103, 199, 195 | 26, 42, 61 | 4627, 4547, 4570 |
| Historical, `ρ = 10` | 4436, 4439, 4310 | 121, 129, 137 | 147, 136, 257 | 24, 34, 91 | 4581, 4602, 4538 |
| Selected main, `ρ = 10` | 4478, 4459, 4356 | 169, 183, 228 | 57, 62, 120 | 10, 20, 37 | 4657, 4662, 4621 |

  For the selected main with `ρ = 10` that is at most 99.00%, 99.11%, 98.24% of the 4704
  held-out states. How many of these answers a run has right with the domain is not computed
  from the recorded answers: it needs the scores of the model.
- The residue of the step is determined at the held-out states with both tokens in one class
  of the training graph: 4692, 4670, 4619 on the grid of the task, that is 99.74%, 99.28%,
  98.19%, and 4704, 4703, 4674 on a grid of scale 2 (CC4), that is 100%, 99.98%, 99.36%.

### DL4. The reading for a model

This item is a reading and two hypotheses, not a theorem.

By RR7 the best run has 169, 183, 228 wrap errors and 57, 62, 120 residue errors, and 99% is
at most 47 errors. By DL2 the wrap errors are not errors of a decoder with the domain. The
residue errors are at tokens with few training cells, and by CC4 the carry of the grid makes
two thirds of those tokens.

The first hypothesis is about the recorded checkpoints, with no training. Decoded with the
domain, a run of RR7 that generalizes has most of its wrap errors right, and its errors are
then its residue errors.

The second is about the encoder and the decoder together. On a grid of scale 2 and with the
decoder of DL2, the models of RR7 whose decay acts within the run reach 99% of the held-out
states.

Both were written down with their predictions before the runs, as stages 9 and 10 of
`research/diffgpt_flat_spectrum/PLAN.txt`; DL6 holds what they found.

### DL5. Scope

Proved: DL1.

Not covered:

- That the best admitted token of a model has the right residue. DL2 says what the decoder
  cannot get wrong, not what a model gets right.
- That a trained network is M. Theorems E and G are about the minimum of the nuclear norm.
- The registered metric of the task, which decodes the best token with no domain. A decode
  with the domain is another metric; RR7, CS6 and EN6 are about the registered one.
- A series with no domain. If no bounds of the series are declared in advance there is
  nothing to admit by, and for a series far from its bounds the domain admits every token.
- The training. The model is still trained to write the exact token, wrap included. A loss
  over the admitted tokens alone is not considered here.

### DL6. The runs decoded with the domain (numerical)

These numbers are measured, not proved: `research/diffgpt_flat_spectrum/` of the `diff-gpt`
repository, Rust and Kernite, FP32 on a GTX 1050. The first part trains no model. The final
checkpoints of the 66 runs with step tokens, those of RR7 and the earlier ones, are loaded
again, replay their recorded metrics exactly, and are decoded twice from the same scores: by
the registered decode, and by the decoder of DL2 with the domain `[0, 96]`. The second part
decodes the runs of CC7 in the same way. The predictions were fixed before the decodes and
before the runs.

| Model and recipe (RR7) | Right, registered decode | Wrap errors | of them right with the domain | Right with the domain | Share |
| --- | --- | --- | --- | --- | --- |
| Historical, its recipe | 4468, 4386, 4344 | 133, 119, 165 | 108, 98, 124 | 4586, 4488, 4484 | 97.49%, 95.41%, 95.32% |
| Historical, `ρ = 3` | 4341, 4336, 4276 | 195, 144, 162 | 158, 111, 124 | 4504, 4454, 4412 | 95.75%, 94.69%, 93.79% |
| Historical, `ρ = 10` | 4436, 4439, 4310 | 121, 129, 137 | 58, 63, 60 | 4498, 4506, 4386 | 95.62%, 95.79%, 93.24% |
| Selected main, `ρ = 1` | 4418, 4420, 4265 | 155, 187, 243 | 123, 155, 192 | 4545, 4578, 4464 | 96.62%, 97.32%, 94.90% |
| Selected main, `ρ = 3` | 4381, 4396, 4280 | 226, 200, 259 | 180, 172, 214 | 4569, 4571, 4500 | 97.13%, 97.17%, 95.66% |
| Selected main, `ρ = 10` | 4478, 4459, 4356 | 169, 183, 228 | 107, 115, 151 | 4590, 4577, 4508 | 97.58%, 97.30%, 95.83% |

- With the domain no answer of any run is off by `p`, and every right answer of the registered
  decode stays right, as DL2 says. The runs with level tokens of EN6 are unchanged: none of
  their answers is outside the domain.
- The first hypothesis of DL4 held in part. Over the 66 runs the domain turns 5196 of the 8251
  wrap errors right, 63%. The prediction made with it, three quarters in every run with 88%
  right, failed in 23 of the 38 such runs. In the rows of the table it is 75% to 86% for all
  but the two with `ρ = 10`, and 44% to 66% for those: the best arms of AdamW under the
  registered decode.
- A wrap error is of two kinds. Where the domain turns it right, the best score of a token of
  the truth is 0.5 to 2.8 below the best score of a token of the other wrap, on average over
  a run: a close call. Where it does not, it is 1.9 to 7.7 below, and the best admitted token
  is not a token of the truth. Under `ρ = 10` its value is 3 to 7 from the value of the wrong
  wrap in 58% to 83% of these errors, and 68% to 92% of them are at states with
  `94 ≤ b + x ≤ 99`, beside the comparison of CW6. There the model is on the other side of
  the comparison, and a token of the truth is not its second choice.
- No run with step tokens on the grid of the task has 99% with the domain. The most is 4590
  of 4704, 97.58%, below the range that was expected for it.
- No decoder gives 99% to these runs. A decoder that keeps the residue of the best token, one
  that is told the modulus for example, has at most the right answers and the wrap errors
  right. The residue errors alone are more than the 47 that 99% allows in every one of the 66
  runs; the least is 57.

The runs of CC7 decoded with the domain, as shares of the 4704 held-out states:

| Model and recipe | Grid of the task | Scale 2, the tokenizer of the task | Scale 2, one grid at every row |
| --- | --- | --- | --- |
| Historical, its recipe | 97.49%, 95.41%, 95.32% | 99.26%, 98.83%, 97.58% | 97.02%, 96.36%, 97.87% |
| Historical, `ρ = 10` | 95.62%, 95.79%, 93.24% | 93.49%, 87.35%, 92.86% | 81.80%, 79.42%, 82.74% |
| Selected main, `ρ = 0.1` | 82.36%, 91.05%, 80.46% | 94.03%, 94.83%, 95.88% | 98.75%, 98.79%, 96.94% |
| Selected main, `ρ = 10` | 97.58%, 97.30%, 95.83% | 99.55%, 97.92%, 96.73% | 95.68%, 97.70%, 97.96% |

- The second hypothesis of DL4 failed. Of the 18 runs of the three arms whose decay acts, two
  have 99% with the domain: the historical recipe and the selected main with `ρ = 10`, both
  with the tokenizer of the task on seed 11, at 99.26% and 99.55%. On seed 17 these two end at
  98.83% and 97.92%, and on seed 23 no run has 99%.
- The grid and the decoder are both parts of the miss. Together they take the historical
  recipe from 94.98%, 93.24%, 92.35% to 99.26%, 98.83%, 97.58%, and the selected main with
  `ρ = 10` from 95.20%, 94.79%, 92.60% to 99.55%, 97.92%, 96.73%.
- Much of what stays wrong is at the steps that few states read (EN2). Take these two arms and
  the selected main with `ρ = 0.1` and one grid at every row. Of the errors that they keep
  with the domain, 43% to 86% are at states with a token of at most 5 training cells. On seed
  23 the 30 states with a step unseen at its position are 28 to 30 of them.

So a decoder with the domain repairs a model that hesitates between the two wraps, and not one
that is on the wrong side of the comparison. On this task a difference encoding with a grid
that holds the values and a decoder with the domain ends at 96.7% to 99.6% in these three
arms. With the level encoding of EN6 the same arms end at 98.32% to 99.53% under the historical
recipe and at 100% in the selected main.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, module `StepDomain`, namespace
`Transformer.Grokking.FlatSpectrum`; proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| DL1 | `eq_modStep_iff`, `modStep_add_mul_not_mem` | For `0 < p` and all naturals `b`, `x`; neither needs `b < p` or `x < p`. The residue is the cast of `d` to `ZMod p`, and the domain is `0 ≤ (x : ℤ) + d ∧ (x : ℤ) + d < p`. The second is for an integer `k ≠ 0`. |
