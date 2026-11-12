# Why x³ + xy resists: a flat spectrum

Source note for `Transformer.Grokking.FlatSpectrum`, received from the user on
2026-10-28 and kept verbatim below. Section names (Setting, Lemma 1,
Proposition 2, Theorem A, Corollary, Theorem B, Readout-free version) are the
labels the Lean docstrings cite. The section *Torch check* after the note keeps
another session's recomputation of it, and *Lean status* maps each claim to its
declarations.

## How to say it in math

Written as an operator, the answer table of x³+xy has a completely flat spectrum. A minimum-norm learner fills unseen entries of a flat operator with the opposite answer. And the labels give the gradient almost no pull toward the intermediate that would split the task into easy steps.

### Setting

G = ℤ/p and ψ(x) = e^{2πix/p}.
For a task f and a readout φ: G → ℂ, the task kernel K_φ(a,b) = φ(f(a,b)) is an operator on ℓ²(G). Ω is the training set.
‖·‖_S1 is the nuclear norm and ‖·‖_S2 the Frobenius norm. The effective rank is r_eff = (‖K‖_S1/‖K‖_S2)².

### Lemma 1 (classical)

‖K‖_S1 = min{ ½(‖U‖² + ‖V‖²) : K = UV* }.

Weight decay on a bilinear layer therefore acts as a nuclear-norm penalty.
Training with weight decay then behaves like minimum-nuclear-norm interpolation from Ω.
This is the modeling step. It is exact for a pure factorization; for the real network it is an assumption, supported by the earlier finding that trained blocks are bilinear.

### Proposition 2 (exact)

- a+b: K = ψψᵀ, rank 1, ‖K‖_S1 = p.
- a³+ab: K = D·F, with D = diag ψ(a³) and F = [ψ(ab)], where FF* = pI. So KK* = pI: all singular values equal √p, and ‖K‖_S1 = p^{3/2}.
- For entries of modulus 1, p ≤ ‖K‖_S1 ≤ p^{3/2}. Addition is the minimum and x³+xy the maximum.
- Relabeling the vocabulary permutes rows and columns, so the spectrum doesn't change.

### Theorem A (proven)

Let K be invertible with polar factor W = K(K*K)^{-1/2}. Then K minimizes ‖X‖_S1 over {X = K on Ω} if and only if W = 0 on every unseen entry.

Proof: the nuclear norm is convex and differentiable at an invertible K with gradient W. Feasible directions are the matrices supported off Ω. ∎

For x³+xy, W = K/√p has every entry of modulus p^{-1/2}. So K is never the minimum-norm interpolant, for any proper Ω.

### Corollary (one missing entry, exact)

If one entry of a flat K is missing, the minimum-norm fill is X_ij = −K_ij. That is the opposite phase, which decodes to f + (p±1)/2 and is always wrong. For a+b the fill is +K_ij, which is right.

Proof sketch: write the fill as K_ij(1−s). Then X = √p·W(I − s·W_ij·(W*e_i)e_jᵀ).
This changes a single 2×2 block, whose determinant is 1 − s/p and squared Frobenius norm is 2 − 2s/p + s²/p.
So ‖X‖_S1 = √p[(p−2) + √(4 − 4s/p + s²/p)], minimized at s = 2. A complex s only makes it larger.

Numerical check at p = 97:

- One missing entry: the fill/truth ratio is −1.000000 for x³+xy and +1.000000 for a+b.
- Many missing entries (ADMM): the mean fill/truth ratio is −0.16, −0.78, −0.87 and −0.96 at 50, 90, 95 and 99% seen.
- Decoded accuracy is 0.7% at 50% seen and 0% at 90–95%, below the 1% chance level.
- a+b decodes 100% at every fraction, as completion theory predicts for rank-1 incoherent matrices (Candès–Recht 2009, Gross 2011).

### Theorem B (the intermediate is invisible, proven)

Split the task as v = ab, then f = a³ + v.

Each step is rank 1, but in different bases: χ(ab) = χ(a)χ(b) in multiplicative characters, and ψ(a³+v) = ψ(a³)ψ(v) in additive ones.
These two bases are mutually unbiased: their inner products are Gauss sums of modulus √p.
For r ≠ 0:

    ‖E[ψ_r(ab) | f]‖² = |S_r − 1|²/p² + |p − 1 + conj(S_r)|² / (p²(2p − 1)),   S_r = Σ_x ψ_r(x³)

Proof: on the fiber f = c ≠ 0, a runs once over the nonzero residues and ab = c − a³. The fiber f = 0 also contains the line a = 0.
For p ≡ 1 mod 3, S_r is a sum of two cubic Gauss sums, so |S_r| ≤ 2√p. For p ≡ 2 mod 3, S_r = 0. ∎

So the quantity is at most 4.5/p + O(p^{-3/2}).
By Cauchy–Schwarz, no unit-norm function of the label correlates with ψ_r(ab) by more than O(p^{-1/2}). That caps how hard the loss can pull a hidden unit toward ab.

Numerical check:

- The maximum over r is 0.044 at p = 97 (bound 0.053) and 0.020 at p = 199.
- For comparison, the sampling noise of a half split is 2/p = 0.021 at p = 97, and (a+b)² with v = a+b gives ½.

Multiplicative characters of ab are the same order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p.
The b/a² route is no better. Since 1 + b/a² = f/a³, the only multiplicative characters of 1 + b/a² that the label reveals are the cubic ones. Those equal χ₃(f), whose kernel is itself spread out (effective rank 48 of 97).

### Readout-free version (conjecture, numerical)

E chooses the readout, so the right invariant is the minimum of r_eff(φ∘f) over readouts φ.

- For (a+b)², gradient search finds 2.0.
- For x³+xy, the best readout found is the indicator of answer 0, with r_eff ≈ p/4 + √(p/2): 31.5 at p = 97 and 60.0 at p = 199. The other one-hot readouts give 0.56p–0.67p.
- Heuristic: a rank-r p×p matrix has r(2p−r) parameters, so a fraction ρ of the table pins down at most r ≤ (1−√(1−ρ))p.
- With half the data that is 28.4 at p = 97, just below the best 31.5. At 80% data it rises to 0.55p, and p = 199 did partly generalize there (50.5% val).

### How it fits together (the link to training is a heuristic)

- Weight decay favors minimum-norm solutions. Theorem A says that solution actively prefers wrong answers on unseen pairs, which is the memorization basin.
- Theorem B says gradients have no first-order path to the intermediate that would make the task two rank-1 steps. This is the modular version of the staircase/leap condition (Abbe, Boix-Adserà, Misiakiewicz 2022–23).
- Giving the model that intermediate as an extra target removes exactly this obstruction. b/a² reaches 99.8% and survives after the target is dropped; a+b and random targets give nothing.
- Theorem A depends only on the objective and Theorem B only on the labels. That fits AdamW and ScheduleFree failing the same way.

Proven: Lemma 1, Proposition 2, Theorem A with its corollary, and Theorem B. They are finite-dimensional linear algebra plus Gauss sums, so they could go into Lean when you say so. Nothing is written there. Still open: the readout-free bound, and whether the real network (norms, SiLU) behaves like the factorization model.

---

## Torch check

Received from another session on 2026-10-28; its claims and completion checks are kept
verbatim below. Its training runs, scripts and setup questions are left out.

The numbers in the readout-free paragraph all check out in torch, but the conclusion doesn't. The readout with the lowest effective rank, the indicator of 0, can't be completed at any data fraction. The real Fourier readouts can, because f is odd in a: f(−a, b) = −f(a, b). The trained networks generalize through that same symmetry, but only partly and only with 85% data or more.

**The claims**

| Claim | Torch result |
|---|---|
| (a+b)²: search finds 2.0 | 2.00 at p = 47, 61, 97 |
| Best readout is the indicator of answer 0 | Yes, after centering: every random start converges to it. r_eff = 0.359p, 0.345p, 0.325p, 0.302p at p = 47, 61, 97, 199 |
| 31.5 at p=97, 60.0 at p=199 | These are the centered values. Uncentered, the exact value is (√(p+1)+1+(p−3)/√2)²/(2p−1) = 31.01 and 59.53; the spectrum matches to 1e-14. p/4+√(p/2) gives 31.21 and 59.72 |
| Other one-hots 0.56p–0.67p | 0.556–0.670p at p=97, 0.575–0.649p at p=199 |
| r ≤ (1−√(1−ρ))p | The arithmetic is right: 28.4 at ½, 0.553p at 80% |

One caveat on the best readout. If you remove the additive part g(a)+h(b), which E[a] + J·E[b] gives the network for free, the indicator of 0 rises to about 0.48p. The minimum is then 0.38–0.41p and comes from a different readout.

**Where it breaks: effective rank doesn't decide whether a kernel can be completed** (min-nuclear-norm completion, p=97)
- **One-hot class kernels [f = c]**, including the indicator of 0: not a single hidden 1 is recovered, at any fraction up to 99%. Each row holds a single 1, which is the worst case for completion.
- **Random matrices of the same rank** do follow the counting rule, but need about 0.1–0.2 more data than it says. At rank 49 the rule says 0.76; completion error is 0.47, 0.08 and 0.00 at 80%, 90% and 95%.
- **Real Fourier kernels:** cos(2πr·f/p) has rank (p+1)/2 and sin has rank (p−1)/2, with every singular value exactly √p. Rows a and −a are equal for cos and opposite for sin, so a hidden entry can be copied from the pair (−a, b). Completing every cos and sin kernel and decoding by argmax:

| Data | Hidden pairs decoded | (−a, b) seen | (−a, b) hidden |
|---|---|---|---|
| 50% | 48.5% | 99% | 0% |
| 70% | 70.2% | 100% | 0% |
| 80% | 79.9% | 100% | 0% |
| 90% | 89.4% | 100% | 0% |

So the flat-spectrum argument (Prop 2 and Theorem A) covers the complex kernel ψ(f) only. Its real and imaginary parts, which are what a real network outputs, have half rank, and Theorem A needs an invertible kernel.

---

## Lean status

Added after the note; the note above is unchanged. All declarations are in
`Transformer.Grokking.FlatSpectrum` (`src/Transformer/Grokking/FlatSpectrum/`) and are
proved without `sorry`. `ψ` is `ZMod.stdAddChar`, `K_φ` is `taskKernel f φ`, and
`addKernel p`, `cubicKernel p` are the kernels of `ψ(a + b)` and `ψ(a³ + a b)`.

| Claim | Lean | Differences from the note |
| --- | --- | --- |
| Setting | `nuclearNorm`, `frobeniusSq` (`‖·‖²_S2`), `effectiveRank`, `taskKernel` | |
| Setting, a task whose arguments have different types | `Recall.recallKernel` | Added, *Associative recall has a flat spectrum*: the kernel `φ(π q)` of a query `q` and a dictionary `π`, with `m` rows and `c^m` columns. |
| Lemma 1 | `nuclearNorm_isLeast`, `nuclearNorm_le_of_eq_mul` | The minimum runs over every inner dimension. |
| Lemma 1, "exact for a pure factorization" | `IsOptimalFit`, `isOptimalFit_iff`, `setOf_isOptimalFit_eq`, `forall_isMinOn_iff_iff_isOptimalFit`, `not_exists_isOptimalFit_of_singularValues_eq`, `isOptimalFit_vecMulVec_iff_connected` | Added. A fit of a set `S` of matrices is a factorization `U Vᴴ ∈ S` of any width `r`; it is optimal when its weight decay `½(‖U‖² + ‖V‖²)` is the least among the fits of every width. A fit is optimal exactly when its product minimizes `‖·‖_S1` on `S` and its weight decay equals that norm; the products of the optimal fits are exactly the minimizers of `‖·‖_S1` on `S`. So for `K ∈ S`, some fit is optimal and every optimal fit multiplies to `K` exactly when M recovers `K`, in D.2's form. Read through this, Theorem A for flat kernels and Theorem E hold for weight decay: no optimal fit multiplies to a flat `K` without zero entries on a proper `Ω`, and for `u vᵀ` with `|u_i v_j| = 1` some fit is optimal and all multiply to `u vᵀ` exactly when `Ω` is connected and contains a permutation. That training reaches an optimal fit is the modeling step; for the pure factorization it is proved for every convergent run, with Theorems A and E for the trained fits as λ → 0⁺, in *Training reaches the optimal fit* (added). |
| Prop. 2, `a + b` | `addKernel_eq_vecMulVec`, `rank_addKernel`, `nuclearNorm_addKernel`, `effectiveRank_addKernel` | |
| Prop. 2, `a³ + a b` | `cubicKernel_eq_diagonal_mul`, `stdAddChar_mul_conjTranspose`, `cubicKernel_mul_conjTranspose`, `singularValues_cubicKernel`, `nuclearNorm_cubicKernel`, `effectiveRank_cubicKernel` | |
| Prop. 2, extremes | `card_le_nuclearNorm_le_of_unimodular`, `nuclearNorm_unimodular_extremes` | The bounds hold for every unimodular `N × N` matrix. |
| Prop. 2, relabeling | `singularValues_taskKernel_relabel`, `nuclearNorm_submatrix` | |
| Theorem A | `isMinOn_nuclearNorm_iff` | Invertible means `K.det ≠ 0`. |
| Theorem A, `x³ + x y` | `polarFactor_cubicKernel`, `norm_polarFactor_cubicKernel`, `not_isMinOn_cubicKernel` | |
| Theorem A, every flat kernel | `polarFactor_of_singularValues_eq`, `det_ne_zero_of_singularValues_eq`, `not_isMinOn_of_singularValues_eq`, `not_forall_isMinOn_iff_of_singularValues_eq` | Added: the source states it for `x³ + x y`. For every square `K` whose singular values are all equal and whose entries are all nonzero, and every proper `Ω`, `K` does not minimize `‖·‖_S1` among the matrices equal to it on `Ω`: some such matrix has a smaller nuclear norm. So M recovers `K` from no proper `Ω`, in D.2's form; `∑ σ_i² = ‖K‖_S2² > 0` makes the common value nonzero and `K` invertible. The example is `x³ + x y` at `p = 3` with `Ω = ∅`. |
| Theorem A, flat kernels of any shape | `IsFlat`, `isMinOn_iff_of_full_row_rank`, `IsFlat.isMinOn_iff`, `IsFlat.not_isMinOn` | Added, *Flat kernels of any shape*: an `m × n` kernel with `Kᴴ K = c I` or `K Kᴴ = c I` is a minimum of M from `Ω` exactly when every unseen cell is a zero of `K`. |
| Corollary, the fill | `nuclearNorm_flat_fill`, `nuclearNorm_flat_fill_real`, `isMinOn_flat_fill_iff` | Proved for every `K` with `K* K = N I`, `N ≥ 2` and `|K_ij| = 1`. The formula `√p[(p−2) + √(4 − 4s/p + s²/p)]` needs `s ≤ p`. `nuclearNorm_flat_fill` gives the value for every complex `s`. |
| Corollary, decoding | `isMinOn_cubic_fill_iff`, `isMinOn_decode_cubic_fill_iff`, `not_isMinOn_decode_cubic_fill` | Nearest `ψ(c)` is exactly `c = f + (p ± 1)/2` for odd `p`. It is never `f` for `p ≥ 2`. |
| Corollary, `a + b` | `isMinOn_add_fill_iff`, `isMinOn_decode_add_fill_iff` | |
| Corollary, numerical check, one missing entry | `isMinOn_cubic_fill_iff`, `isMinOn_add_fill_iff` | The fill/truth ratios `−1.000000` and `+1.000000` are exact: the only minimizing fill is `−K_ij` for `x³ + x y` and `K_ij` for `a + b`. |
| Theorem B, rank-1 steps | `taskKernel_mul_mulChar`, `rank_taskKernel_mul_mulChar`, `taskKernel_cube_add`, `rank_taskKernel_cube_add` | `[χ(a b)] = χ χᵀ` for every multiplicative character `χ`; `[ψ_r(a³ + v)] = ψ_r(a³) ψ_r(v)ᵀ` for every `r`. |
| Theorem B, "in different bases" | `conjTranspose_mul_taskKernel_mul`, `singularValues_taskKernel_mul`, `conjTranspose_mul_taskKernel_add`, `rank_taskKernel_add`, `singularValues_taskKernel_add`, `effectiveRank_taskKernel_add` | Added: read in the other basis, each step is flat. For `r ≠ 0` every singular value of `[ψ_r(a b)]` is `√p`; for nontrivial `χ`, `[χ(u + v)]` has rank `p − 1`, its nonzero singular values are `√p`, and `r_eff = p − 1`. |
| Theorem B, mutually unbiased | `sum_star_stdAddChar_mul`, `norm_gaussSum_mulShift`, `gaussSum_one_mulShift`, `gaussSum_mulShift_zero` | `∑_x conj ψ_r(x) χ(x) = g(χ, ψ_{−r})`, and `|g(χ, ψ_r)| = √p` for nontrivial `χ` and `r ≠ 0`. The other pairs give `g(1, ψ_r) = −1` and `g(χ, ψ_0) = 0`. The `p − 1` multiplicative characters vanish at `0`, so they are not a basis of all of `ℓ²(G)`. |
| Theorem B, `S_r` | `cubicSum`, `norm_cubicSum_le`, `cubicSum_eq_zero` | `|S_r| ≤ 2√p` holds for every prime `p` and `r ≠ 0`. The proof counts the solutions of `x³ = y³`; the next row is the source's route. |
| Theorem B, cubic Gauss sums | `exists_orderOf_eq_three`, `card_pow_three_eq_eq`, `cubicSum_eq_gaussSum_add`, `norm_cubicSum_le_of_orderOf` | For `p ≡ 1 mod 3` a character `χ` of order 3 exists, `#{x³ = y} = 1 + χ(y) + χ²(y)`, and `S_r = g(χ, ψ_r) + g(χ², ψ_r)` for `r ≠ 0`. |
| Theorem B, formula | `meanSq_condMean_cube_add_mul` | Prime `p`, every `r`, `r = 0` included. |
| Theorem B, "at most 4.5/p + O(p^{-3/2})" | `meanSq_condMean_cube_add_mul_le` | `≤ 9/(2p) + 8/p^{3/2}` for `r ≠ 0`. |
| Theorem B, Cauchy–Schwarz | `norm_mean_star_mul_cube_add_mul_le` | `|E[conj φ(f) ψ_r(a b)]| ≤ ‖φ(f)‖ √(9/(2p) + 8/p^{3/2})` for every `φ : ℤ/p → ℂ`. |
| Theorem B, numerical check, "bound 0.053" | `meanSq_condMean_cube_add_mul_le_sqrt`, `cube_bound_97`, `meanSq_condMean_cube_add_mul_lt_97` | The bound is the formula with `\|S_r\| ≤ 2√p`. At `p = 97` it is `(84681 + 1156√97)/1815937`, between `0.0528` and `0.053`. |
| Theorem B, numerical check, "(a+b)² with v = a+b gives ½" | `sum_prod_add`, `condMean_sq_add`, `meanSq_condMean_sq_add` | Exact for odd prime `p` and `r ≠ 0`: the fiber of `(a + b)²` through `v` is `a + b = ±v`, so `E[ψ_r(a + b) \| f] = (ψ_r(v) + ψ_r(−v))/2`. |
| Theorem B, Weil's bound | `norm_jacobiSum`, `norm_jacobiSum_le`, `sum_mulChar_sub_cube_eq`, `sum_mulChar_sub_cube_eq_zero`, `norm_sum_mulChar_sub_cube_le` | Nontrivial `χ` and `c ≠ 0`. The source cites Weil's theorem; the proof uses Jacobi sums `J(χ₁, χ₂) = ∑_t χ₁(t) χ₂(1 − t)`. For `p ≡ 1 mod 3` and `χ₃` of order 3, `∑_a χ(c − a³) = χ₃(c) χ(c) J(χ₃, χ) + χ₃²(c) χ(c) J(χ₃², χ)` with each `|J| ≤ √p`; otherwise cubing is a bijection and the sum is `0`. |
| Theorem B, multiplicative characters "the same order" | `sum_fiber_cube_add_mul_mulChar`, `meanSq_condMean_cube_add_mul_mulChar_le`, `norm_mean_star_mul_cube_add_mul_mulChar_le` | For nontrivial `χ`, `‖E[χ(a b) \| f]‖² ≤ 9/(2p) + 8/p^{3/2}`, the bound proved for `ψ_r(a b)`, and the same Cauchy–Schwarz corollary. |
| Theorem B for any `P` | `poly_add_mul_eq_iff`, `sum_row_poly_add_mul`, `sum_fiber_poly_add_mul`, `card_fiber_poly_add_mul`, `meanSq_condMean_poly_add_mul`, `meanSq_condMean_poly_add_mul_le` | Added; the source takes `P(a) = a³`. For every function `P` of `a` and every `r`, `‖E[ψ_r(a b) \| P(a) + a b]‖² = \|T\|²/p² + \|ψ_r(P(0)) T + p\|²/(p² (2p − 1))` with `T = ∑_{a ≠ 0} ψ(−r P(a))`, and `\|∑_x ψ(r P(x))\| ≤ B` gives `≤ (B + 1)²/p² + (p + B + 1)²/(p² (2p − 1))`. For `P(a) = a³`, `B = 2√p`. |
| Theorem B for any `P`, multiplicative characters | `sum_fiber_poly_add_mul_mulChar`, `norm_sum_erase_mulChar_le`, `meanSq_condMean_poly_add_mul_mulChar_le`, `meanSq_condMean_poly_add_mul_mulChar_le_of_finset` | Added. If `\|∑_a χ(c − P(a))\| ≤ B` for every `c ≠ P(0)`, then `‖E[χ(a b) \| f]‖² ≤ (B + 1)²/p² + (p − 1)²/(p² (2p − 1))`. A set `E` of fibers where the bound fails adds at most `\|E\| (p − 1)/p²`. |
| Theorem B for a polynomial `P`, the Legendre symbol | `norm_sum_quadraticChar_sub_poly_le`, `meanSq_condMean_poly_add_mul_quadraticChar_le`, `eq_of_C_sub_eq_C_mul_sq`, `meanSq_condMean_poly_add_mul_quadraticChar_le_of_natDegree_pos`; Weil: `norm_sum_quadraticChar_eval_le_of_ne`, `norm_sum_quadraticChar_eval_le`, `norm_lPowerSum_quadraticChar_le`, `quadraticChar_norm`, `sum_mulChar_norm_aeval`, `lCoeff_mulWeight_quadraticChar_eq_zero_of_ne`, `exists_isCoprime_quadraticChar_resultant`, `lCoeff_mulWeight_eq_zero_of_isCoprime`, `resultant_mulMod`, `norm_lPowerSum_one_le`; Stepanov: `mul_card_pow_eq_le`, `card_pow_eq_le_sqrt`, `abs_sum_quadraticChar_le`, `abs_card_sq_eq_sub_le`, `exists_monic_irreducible_odd_pow_mul`, `eq_zero_of_C_mul_sq_eq_of_ne_C_mul_sq` | Added: the source cites Weil only for `a³`. For odd `p` and `P` of degree `m ≥ 1`, Weil's bound `\|∑_a χ₂(c − P(a))\| ≤ (m − 1)√p` holds whenever `c − P` is not a constant times a square. For odd `m` that is every `c`, and `‖E[χ₂(a b) \| P(a) + a b]‖² ≤ ((m − 1)√p + 1)²/p² + (p − 1)²/(p² (2p − 1))`, of order `m²/p`; for `a³` the constant is the source's `2√p`. For even `m` at most one `c` fails (for `P = a²` and `c = 0` the sum is `±(p − 1)`), and the bound gains `(p − 1)/p²`. Weil's bound is proved in `Transformer.Weil` from the mathematics: Stepanov's bound `7m√q` for `y² = f(x)` over every `𝔽_{p^d}`, where `f` keeps a root of odd multiplicity and `χ_L = χ₂ ∘ N_{L/𝔽_p}`; the `L`-function of the weight `χ₂(Res(g, f))`, a polynomial of degree `< m` through a unit `u` modulo `f` with `χ₂(Res(f, u)) ≠ χ₂(λ)^N`, with `S_d` the Legendre sum over `𝔽_{p^d}`; and the transfer to `\|ω\| ≤ √p`. Characters of order `> 2` are in the next row; `ψ_r(a b)` follows it. |
| Theorem B for a polynomial `P`, a character of order prime to `deg P` | `norm_sum_mulChar_sub_poly_le_of_coprime`, `meanSq_condMean_poly_add_mul_mulChar_le_of_coprime`; Weil: `norm_sum_mulChar_eval_le_of_coprime_orderOf`, `norm_lPowerSum_mulChar_le_of_coprime`, `normMulChar`, `pow_eq_one_of_normMulChar_eq_one`, `pow_eq_one_of_mulChar_eq_one`, `lCoeff_mulWeight_eq_zero`; Stepanov: `eq_zero_of_sum_mul_pow_pow_mul_X_pow_eq_zero`, `eval_pow_mul_hasseDeriv_powAux`, `mul_card_pow_eq_le_of_coprime`, `pow_stepanov_params`, `card_pow_eq_le_of_coprime`, `norm_sum_le_of_card_fiber_le`, `norm_sum_mulChar_eval_le_of_coprime` | Added: the source cites Weil only for `a³`. For `χ ≠ 1` of order `d` and `P` of degree `m ≥ 1` with `d` prime to `m`, Weil's bound `\|∑_a χ(c − P(a))\| ≤ (m − 1)√p` holds for every `c`, and `‖E[χ(a b) \| P(a) + a b]‖² ≤ ((m − 1)√p + 1)²/p² + (p − 1)²/(p² (2p − 1))`, of order `m²/p`; for `d = 2` this is the odd-degree Legendre case. Weil's bound is proved in `Transformer.Weil` from the mathematics: Stepanov's method for `y^d = f(x)` with `R = ∑_{i < d, j < J} r_{ij} f^{c i} X^{q j}`, `d c + 1 = q`, gives `#{x : f(x)^c = a} ≤ q/d + 4m√q` and `\|∑_x χ(f(x))\| ≤ 5dm√q` over `𝔽_q`, `q = p^k ≥ max(64, 8dm, d²)`, with `χ_L = χ ∘ N_{L/𝔽_p}`; `χ^m ≠ 1` makes the `L`-function a polynomial of degree `< m`; and the transfer gives `\|ω\| ≤ √p`. An order `d` with a common factor with `m`: next row. |
| Theorem B for a polynomial `P`, every character | `norm_sum_mulChar_sub_poly_le_of_not_forall`, `norm_sum_mulChar_sub_poly_le_of_derivative_eq_zero`, `exists_forall_ne_norm_sum_mulChar_sub_poly_le`, `meanSq_condMean_poly_add_mul_mulChar_le_of_natDegree_pos`, `dvd_derivative_sq_of_forall_sq_dvd`, `eq_of_forall_sq_dvd_C_sub`; Weil: `norm_sum_mulChar_eval_le_of_pow_mul`, `norm_lPowerSum_mulChar_le_of_pow_mul`, `norm_sum_mulChar_eval_le_of_root`, `sum_mulChar_eval_reflect_comp`, `natDegree_reflect_comp_X_add_C`, `exists_isCoprime_mulChar_resultant`, `lCoeff_mulWeight_eq_zero_of_pow_mul`, `exists_map_pow_mul_eq_X_sub_C_pow_mul`, `norm_lPowerSum_one_le_of_mul` | Added. For `χ ≠ 1` of any order `d` and `P` of degree `m ≥ 1`, `d` and `m` may share a factor: Weil's bound `\|∑_a χ(c − P(a))\| ≤ (m − 1)√p` holds for every `c` but one, and `‖E[χ(a b) \| P(a) + a b]‖² ≤ ((m − 1)√p + 1)²/p² + (p − 1)/p² + (p − 1)²/(p² (2p − 1))`, the bound of the previous row plus `(p − 1)/p²`. Weil's bound is proved for `f = R^e Q` with `R` monic irreducible, `R ∤ Q` and `e` prime to `d`: over `𝔽_{p^k}` with `deg R ∣ k`, `R` has a simple root `α`, and `x = α + 1/y` turns `f` into a polynomial of degree `N − e`, prime to `d` for `d ∣ N > m`, to which the Stepanov bound applies; a unit modulo `f` built from the surjective norm of `𝔽_p[X]/(R)` gives `c_n = 0` for `n ≥ m`; the transfer runs along the `k` divisible by `deg R`. A `c` for which `c − P` has an irreducible factor `R` with `R² ∤ c − P` takes `e = 1`. If every factor of `c − P` is squared, then `c − P ∣ P'²`, so for `P' ≠ 0` at most one `c` is left, two coprime such `c − P` having total degree `2m > deg P'²`; `P' = 0` gives `p ∣ m` and the trivial bound `p ≤ (m − 1)√p`. The fiber left can fail Weil's bound for `d ∣ m`: `P = (a − 1)^d` and `c = 0` give `\|∑_a χ(−(a − 1)^d)\| = p − 1`. Weil's bound for `c − P` without a factor of multiplicity prime to `d`, such as `d = 6` and multiplicities `2, 3, 3`, is not proved; Theorem B does not need it. |
| Theorem B for a polynomial `P`, `ψ_r(a b)` | `norm_sum_stdAddChar_mul_poly_le`, `meanSq_condMean_poly_add_mul_stdAddChar_le`, `norm_mean_star_mul_poly_add_mul_le`; Weil: `norm_sum_addChar_eval_le`, `norm_le_sqrt_of_mem_roots`, `lPowerSum_eq_neg_sum_roots`, `sum_addChar_trace_aeval`, `natCast_mul_lCoeff`, `lCoeff_eq_zero`, `norm_sum_addChar_trace_le` | Added: the source bounds `S_r` only for `a³`. For `P` of degree `m ≥ 1` prime to `p` (every `1 ≤ m < p`) and `r ≠ 0`, Weil's bound `\|∑_x ψ(r P(x))\| ≤ (m − 1)√p` gives `‖E[ψ_r(a b) \| P(a) + a b]‖² ≤ ((m − 1)√p + 1)²/p² + (p + (m − 1)√p + 1)²/(p² (2p − 1))`, of order `m²/p`, and the Cauchy–Schwarz corollary. For `a³` and `p ≠ 3` the constant is the source's `2√p`. Weil's bound is proved in `Transformer.Weil` from the mathematics: Stepanov's bound over every `𝔽_{p^d}`, the `L`-function of `y^p − y = f(x)` (a polynomial of degree `< m`, with `S_d = −∑ ω^d` over the `m − 1` roots `ω` of its reverse), and the transfer to `\|ω\| ≤ √p`. |
| Theorem B, the `b/a²` route | `one_add_div_sq_eq`, `mulChar_one_add_div_sq_of_pow_three`, `condMean_mulChar_one_add_div_sq_of_pow_three`, `condMean_mulChar_one_add_div_sq_of_pow_three_ne` | `b/a²` needs `a ≠ 0`, so `(a, b)` is uniform on `(ℤ/p)ˣ × ℤ/p`. There `E[χ(1 + b/a²) \| f] = χ(f)` if `χ³ = 1` and `0` otherwise. |
| Theorem B, "effective rank 48 of 97" | `cubicKernelMulChar`, `cubicKernelMulChar_mul_conjTranspose_apply`, `cubicKernelMulChar_mul_conjTranspose_cube`, `eigenvalues_mem_of_cubic`, `card_eigenvalues_cubicKernelMulChar`, `singularValues_cubicKernelMulChar_mem`, `rank_cubicKernelMulChar`, `nuclearNorm_cubicKernelMulChar`, `effectiveRank_cubicKernelMulChar`, `rank_cubicKernelMulChar_97`, `effectiveRank_cubicKernelMulChar_97` | Proved for every nontrivial `χ` and odd `p`, not only `χ₃`. `K K*` has eigenvalue `p + 1` once, `2p` `(p − 3)/2` times and `0` otherwise, so `rank K = (p − 1)/2` and `r_eff = (√(p + 1) + (p − 3)/2 · √(2p))²/(p − 1)²`. At `p = 97`: rank 48 and `47.9 < r_eff < 48`, so the note's 48 is `r_eff` rounded. |
| Readout-free version, "For (a+b)², gradient search finds 2.0"; torch check "2.00 at p = 47, 61, 97" | `fourierMatrix`, `taskKernel_add_eq_fourier`, `conjTranspose_mul_taskKernel_add_eq_fourier`, `charpoly_conjTranspose_mul_taskKernel_add`, `nuclearNorm_taskKernel_add_eq_sum_dft`, `frobeniusSq_taskKernel_add_eq_sum_dft`, `effectiveRank_taskKernel_add_eq_dft`, `dft_cos`, `two_mul_sum_sq_le_sq_sum`, `two_le_effectiveRank_taskKernel_add_of_even`, `two_le_effectiveRank_sq_add`, `effectiveRank_taskKernel_add_cos`, `cosSqReadout`, `effectiveRank_sq_add_cosSqReadout`, `effectiveRank_taskKernel_add_const`, `effectiveRank_sq_add_const`, `isLeast_effectiveRank_sq_add` | Exact for every odd prime `p`. Changed: the minimum runs over the centered readouts, `∑_s φ(s²) = 0` with `φ` not zero on the squares, since a constant readout gives `r_eff = 1`. The kernel `φ((a + b)²)` is `g(a + b)` with `g(s) = φ(s²)`, and the singular values of `[g(a + b)]` are the `|ĝ(k)|` of the discrete Fourier transform. A centered even `g` has `ĝ` even with `ĝ(0) = 0`, so pairing `k` with `−k` gives `(∑_k |ĝ(k)|)² ≥ 2 ∑_k |ĝ(k)|²`, that is `r_eff ≥ 2`. A readout with `φ(s²) = cos(2π r s/p)`, `r ≠ 0`, has `r_eff = 2`. |
| Readout-free version, counting rule | `mul_two_mul_sub_le_iff`, `counting_rule_half_97`, `counting_rule_eighty`, `counting_rule_rank_49` | Only the arithmetic: for `r ≤ p`, `r(2p − r) ≤ ρp²` iff `r ≤ (1 − √(1 − ρ))p`; `28.4 < (1 − √½)·97 < 28.45`, `0.5525 < 1 − √0.2 < 0.553`, and the torch check's "at rank 49 the rule says 0.76" is `49 · 145/97² = 7105/9409 ≈ 0.7551`. That a rank-`r` matrix has `r(2p − r)` parameters, and what a fraction of the table pins down, are not formalized. |
| Torch check, one-hot kernels | `classKernel`, `nuclearNorm_updateRow_zero_lt`, `eq_zero_of_isMinOn_of_row`, `eq_zero_of_isMinOn_classKernel`, `eq_zero_of_isMinOn_classKernel_cube_add_mul`, `isMinOn_updateRow_zero` | Exact for every training set: a hidden 1 in a row `a ≠ 0` of `[a³ + a b = c]` is `0` in every minimum-nuclear-norm completion. More generally, a row whose seen entries are all `0` is all `0` in every completion. The all-ones row `a = 0` of `[f = 0]` is not covered. A minimum is shown to exist only for one hidden cell. |
| Torch check, real Fourier kernels | `cosKernel`, `sinKernel`, `cosKernel_neg`, `sinKernel_neg`, `det_cosKernel`, `det_sinKernel`, `rank_cosKernel`, `rank_sinKernel`, `singularValues_cosKernel`, `singularValues_sinKernel`, `nuclearNorm_effectiveRank_cosKernel`, `nuclearNorm_effectiveRank_sinKernel` | Odd prime `p` and `r ≠ 0`; the cos kernel needs both. "Every singular value exactly √p" holds for the nonzero ones; the others are `0`. Also `‖·‖_S1 = √p (p ± 1)/2` and `r_eff = (p ± 1)/2`. |
| Torch check, the indicator of 0, "Uncentered, the exact value is (√(p+1)+1+(p−3)/√2)²/(2p−1) = 31.01 and 59.53" | `zeroIndicatorKernel`, `eigenvalues_zeroIndicatorKernel_mem`, `trace_zeroIndicatorKernel_mul_conjTranspose`, `trace_zeroIndicatorKernel_mul_conjTranspose_sq`, `card_eigenvalues_zeroIndicatorKernel`, `singularValues_zeroIndicatorKernel_mem`, `rank_zeroIndicatorKernel`, `nuclearNorm_zeroIndicatorKernel`, `frobeniusSq_zeroIndicatorKernel`, `effectiveRank_zeroIndicatorKernel`, `effectiveRank_zeroIndicatorKernel_97`, `effectiveRank_zeroIndicatorKernel_199`, `quarter_add_sqrt_half_97`, `quarter_add_sqrt_half_199` | Exact for every odd prime `p`. `K K*` has the eigenvalues `p + 1` and `1` once each, `2` `(p − 3)/2` times and `0` otherwise, so the singular values are `√(p + 1)`, `1`, `√2` and `0`, `rank K = (p + 1)/2`, `‖K‖_S1 = √(p + 1) + 1 + (p − 3)/2 · √2` and `‖K‖²_S2 = 2p − 1`. `r_eff` lies in `(31.014, 31.015)` at `p = 97` and in `(59.532, 59.533)` at `p = 199`, so it rounds to `31.01` and `59.53`; the note's `p/4 + √(p/2)` lies in `(31.214, 31.215)` and `(59.724, 59.725)`, rounding to `31.21` and `59.72`. The centered values are the next row; that this readout is the best one is not formalized. |
| Torch check, the centered indicator of 0, "31.5 at p=97, 60.0 at p=199 \| These are the centered values"; "r_eff = 0.359p, 0.345p, 0.325p, 0.302p at p = 47, 61, 97, 199" | `nuclearNorm_of_gram_eq_proj_add`, `nuclearNorm_of_gram_eq_proj_add_diagonal`, `frobeniusSq_of_gram_eq_add`, `effectiveRank_of_gram_eq_proj_add_diagonal`, `shiftedZeroIndicatorKernel`, `centeredZeroIndicatorKernel`, `shiftedZeroIndicatorKernel_apply`, `negSqIndicator`, `negSqIndicator_mul_self`, `sum_neg_sq_eq`, `sum_negSqIndicator`, `ite_cube_add_mul_eq`, `sum_ite_cube_add_mul`, `sum_ite_cube_add_mul_mul`, `conjTranspose_mul_shiftedZeroIndicatorKernel_apply`, `sum_shiftedZeroIndicatorKernel`, `sum_centeredZeroIndicatorKernel`, `negSqProjection`, `negSqFrame`, `shiftedGramBlock`, `dotProduct_negSqIndicator`, `negSqProjection_conjTranspose`, `negSqProjection_mul_self`, `negSqFrame_conjTranspose_mul`, `negSqProjection_mul_negSqFrame`, `trace_negSqProjection`, `conjTranspose_mul_shiftedZeroIndicatorKernel`, `trace_shiftedGramBlock_mul_diagonal`, `det_shiftedGramBlock`, `conjTranspose_mul_shiftedZeroIndicatorKernel_eq_sq`, `nuclearNorm_shiftedZeroIndicatorKernel`, `frobeniusSq_shiftedZeroIndicatorKernel`, `effectiveRank_shiftedZeroIndicatorKernel`, `effectiveRank_centeredZeroIndicatorKernel`, `effectiveRank_centeredZeroIndicatorKernel_97`, `effectiveRank_centeredZeroIndicatorKernel_199`, `effectiveRank_centeredZeroIndicatorKernel_div_47`, `effectiveRank_centeredZeroIndicatorKernel_div_61`, `effectiveRank_centeredZeroIndicatorKernel_div_97`, `effectiveRank_centeredZeroIndicatorKernel_div_199` | Exact for every odd prime `p`. The source does not define the centering; subtracting the mean `μ = (2p − 1)/p²` of `[a³ + a b = 0]` over the `p²` pairs, so that the entries of `K_μ = [a³ + a b = 0] − μ` sum to `0`, reproduces all six numbers. For every `μ`, `K_μ* K_μ = 2E + U M U*`: here `u_b = 1` if `b = −a²` for some `a ≠ 0`, `E = diag u − 2u uᵀ/(p − 1)` is the projection onto the vectors supported on `{u = 1}` with sum `0`, `U = [u, 1 − u]` has `U* U = diag((p − 1)/2, (p + 1)/2)` and `E U = 0`, and `M` is a `2 × 2` block. If `X* X = c²E + W B W*` with `E` a projection, `W* W = 1` and `E W = 0`, then `‖X‖_S1 = c tr E + √(tr B + 2√\|det B\|)`, so `‖K_μ‖_S1 = (p − 3)/√2 + √(p + 2 − 2(2p − 1)μ + p²μ² + 2\|1 − μ\|√(p + 1))` and `‖K_μ‖²_S2 = 2p − 1 − 2(2p − 1)μ + p²μ²`; `μ = 0` gives the previous row. Centered, `r_eff = (p(p − 3)/√2 + √(p³ − 2p² + 4p − 1 + 2(p − 1)²√(p + 1)))²/((2p − 1)(p − 1)²)`. It lies in `(31.498, 31.499)` at `p = 97` and in `(60.024, 60.025)` at `p = 199`, rounding to `31.5` and `60.0`, and `r_eff/p` lies in `(0.3586, 0.3587)`, `(0.3449, 0.3450)`, `(0.3247, 0.3248)` and `(0.3016, 0.3017)` at `p = 47, 61, 97, 199`, rounding to `0.359`, `0.345`, `0.325` and `0.302`. That this readout is the best one, and that every random start converges to it, are not formalized. |
| Torch check, "Other one-hots 0.56p–0.67p" | `sum_comp_eigenvalues_of_charpoly`, `conjTranspose_mul_classKernel`, `nuclearNorm_classKernel`, `frobeniusSq_classKernel`, `effectiveRank_classKernel`, `frobeniusSq_classKernel_cube_add_mul`, `effectiveRank_classKernel_cube_add_mul` | Only the formula. If every row of `[f = c]` holds at most one 1, `K* K` is diagonal with the column counts `N_b = #{a : f(a, b) = c}`, so `r_eff = (∑_b √N_b)²/∑_b N_b`. For `a³ + a b` and `c ≠ 0`, `∑_b N_b = p − 1` and `N_b` is the number of roots of `X³ + b X − c`. The measured range `0.556–0.670p` over `c` is that of the centered kernels (a numpy check gives `0.555–0.671p` uncentered) and is not formalized. |

Not formalized:

- the measured numbers: the ADMM fill ratios and decoded accuracies, `a + b` decoding at
  every fraction, the maxima `0.044` and `0.020` over `r`, the sampling noise `2/p`, and the
  torch completion rates;
- the results of the readout searches for `x³ + x y`: that the indicator of 0 is the best
  readout, the range `0.56p–0.67p` of the other one-hots, and the caveat's `0.48p` and
  `0.38–0.41p`;
- Lemma 1's modeling step for the real network (norms, SiLU, AdamW). For the pure
  factorization it is proved: at an optimal fit weight decay is exactly
  minimum-nuclear-norm interpolation (row *Lemma 1, "exact for a pure factorization"*), a
  convergent run of gradient descent with weight decay reaches the optimum, and on the
  squared loss the run does converge, by Łojasiewicz's theorem (*Training reaches the
  optimal fit*, added). On the cross-entropy the run converges as well, and its limit fills
  the unseen cells by M (*Training with cross-entropy*, added). For AdamW the step fails on
  the pure factorization: a limit of a full-batch run is not a minimizer of the
  weight-decayed loss (item WD13 there);
- the readout-free version for `x³ + x y`, which the note calls a conjecture;
- "How it fits together", which the note calls a heuristic.

Scope: Proposition 2, Theorem A and the Corollary are about the complex kernel `ψ(f)`. The real
readouts `cos(2πr f/p)` and `sin(2πr f/p)` of `f = a³ + a b` are singular, as rows `a` and `−a`
are equal or opposite, so Theorem A does not apply to them. They have rank `(p + 1)/2` and
`(p − 1)/2`, and every nonzero singular value is `√p` (row *Torch check, real Fourier kernels*);
the proof uses `conj K = P K`, where `P` is the row flip `a ↦ −a`. The completion rates of the
real kernels are not formalized. Of the readout numbers, the minimum `2.0` for `(a+b)²`, over
centered readouts, the exact uncentered and centered values for the indicator of 0 and the
formula for the other one-hots are formalized (rows *Readout-free version, (a+b)²*, *Torch
check, the indicator of 0*, *the centered indicator of 0* and *Other one-hots*); that the search
for `x³ + x y` finds the indicator of 0, and the values of the other readouts it found, are not.
Theorem B covers real readouts: `φ` is any complex function of the label.

---

## Readout-free bound for P(a) + a b (added)

Added on 2026-10-28 and proved in Lean. The note fixes the readout `ψ` in Proposition 2 and
Theorem A. This section lets the readout be any function, for the family
`f(a, b) = P(a) + a b` with a polynomial `P` over `𝔽_p`.

### Setting

- `ρ, g, h : ℤ/p → ℂ`, with `‖h‖² = ∑_b |h(b)|²`.
- The readout kernel acts by `(K_ρ h)(a) = ∑_b ρ(f(a, b)) h(b)`.
- `h` is centered when `∑_b h(b) = 0`. Centering removes what `b` alone predicts.
- `m = deg P` and `κ = (m − 1)(2p − 1)`.

### Theorem C (proven)

1. For every `P`, every `ρ` and every centered `h`: `‖K_ρ h‖² ≤ (p − 1) ‖ρ‖² ‖h‖²`.
2. For `deg P ≤ 1` this bound is reached. Take `r = 1 − p δ₀`, `ρ(c) = r(c − P(0))` and
   `h(b) = r(b + P'(0))`; then `‖K_ρ h‖² = (p − 1) ‖ρ‖² ‖h‖²`.
3. For `deg P ≥ 2`, every `ρ`, every `g` and every centered `h`:
   `|∑_{a,b} ρ(f(a, b)) g(a) h(b)|² ≤ (1 + √κ) ‖ρ‖² ‖g‖² ‖h‖²`. Hence
   `‖K_ρ h‖² ≤ (1 + √κ) ‖ρ‖² ‖h‖²`.
4. Dichotomy: once `κ < (p − 2)²`, some `ρ` and centered `h` with `‖ρ‖ ‖h‖ > 0` reach
   item 1's bound if and only if `deg P ≤ 1`.
5. Phase readouts, `|ρ| = 1`: `|∑ ρ(f) g h|² / (p² ‖g‖² ‖h‖²) ≤ (1 + √κ)/p`.

The normalized peak `‖K_ρ h‖² / ((p − 1) ‖ρ‖² ‖h‖²)` lies in `[0, 1]`. It is `1` for affine
`P`. For `deg P ≥ 2` it is at most `(1 + √κ)/(p − 1) ≈ √(2(m − 1)/p)`. For `a³ + a b` at
`p = 97` that is `0.215`. So no readout of `P(a) + a b` with `deg P ≥ 2` has one centered
mode carrying more than `O(p^{−1/2})` of the kernel. This holds for every readout, not only `ψ`.

### Proof

1. Row `a = 0` is constant, so it vanishes against a centered `h`. For `a ≠ 0` the map
   `b ↦ P(a) + a b` is a bijection. Hence `∑_{a,b} ρ(f) g h = ∑_c ρ(c) G(c)`, with
   `G(c) = ∑_{a ≠ 0} g(a) h((c − P(a))/a)`, and Cauchy–Schwarz gives
   `|·|² ≤ ‖ρ‖² ‖G‖²`.
2. Substitute `c = P(a) + a x` in the Gram entries of `G`. This gives
   `‖G‖² = ∑_{a' ≠ 0} ∑_{λ ≠ 0} g(λ a') conj g(a') E_h(λ, μ_λ(a'))`. Here
   `E_h(λ, μ) = ∑_x h(x) conj h(λ x + μ)` and `μ_λ(a') = (P(λ a') − P(a'))/a'`.
3. The term `λ = 1` is at most `‖g‖² ‖h‖²`, since `|E_h(λ, μ)| ≤ ‖h‖²`. Cauchy–Schwarz over
   the pairs `(a', λ)` with `λ ≠ 1` bounds the rest by `‖g‖²` times the square root of the
   off-diagonal energy.
4. Affine energy: for a centered `h`, `∑_{λ,μ} |E_h(λ, μ)|² = p ‖h‖⁴`.
5. Call `λ` degenerate when `P(λ t) − P(t)` is linear in `t`.
   - For a nondegenerate `λ`, each value of `μ_λ` is taken by at most `m − 1` nonzero `a'`.
     They are roots of `P(λ t) − P(t) − μ t` other than `0`.
   - A degenerate `λ ≠ 0` has `λ^m = 1`, from the coefficient of `t^m`. So at most `m − 1` of
     them differ from `1`.
6. The off-diagonal energy is therefore at most `(m − 1) p ‖h‖⁴ + (m − 1)(p − 1) ‖h‖⁴ = κ ‖h‖⁴`.

Items 1 and 2 are direct. In item 2, `f − P(0) = a (b + P'(0))` and `r(a u) = r(u)` for
`a ≠ 0`, so every row `a ≠ 0` of `K_ρ h` equals `‖r‖²`. Item 4 follows from items 1–3,
because `√κ < p − 2` makes `1 + √κ < p − 1`. Item 5 is item 3 with `‖ρ‖² = p`.

### Numerical check (torch, 2026-10-28)

Alternating maximization of `‖K_ρ h‖² / (‖ρ‖² ‖h‖²)` over `ρ` and centered `h`, in
complex128. Each round sets `h` to the top centered right singular vector of `K_ρ`, then `ρ`
to the top singular vector of `ρ ↦ K_ρ h`. 24 random starts of 40 rounds each. The values
found are lower estimates of the maximum.

| `p` | `P` | `m` | found | `1 + √κ` | `p − 1` |
| --- | --- | --- | --- | --- | --- |
| 61, 97, 199 | `a`, `3 + 5a` | 1 | `p − 1` exactly | — | 60, 96, 198 |
| 61 | `a²`, `a³`, `a⁴`, `a⁵` | 2–5 | 8.011, 10.159, 11.836, 12.586 | 12.000, 16.556, 20.053, 23.000 | 60 |
| 97 | `a²`, `a³`, `a⁴`, `a⁵` | 2–5 | 8.515, 11.009, 14.433, 11.050 | 14.892, 20.647, 25.062, 28.785 | 96 |
| 199 | `a²`, `a³`, `a⁴`, `a⁵` | 2–5 | 10.630, 12.710, 12.893, 8.830 | 20.925, 29.178, 35.511, 40.850 | 198 |
| 61, 97, 199 | `a³ + a²` | 3 | 6.869, 7.123, 7.483 | 16.556, 20.647, 29.178 | |
| 61, 97, 199 | `2 + a + 4a² + a⁴` | 4 | 8.615, 9.010, 9.546 | 20.053, 25.062, 35.511 | |

Every value found is below the bound, and the affine cases reach `p − 1`. The bound is not
sharp. The values found stay between 6.8 and 14.5 while `1 + √κ` grows like `√(2(m − 1)p)`. The true
growth is open.

### Lean status

All declarations are in `Transformer.Grokking.FlatSpectrum` and are proved without `sorry`.

| Claim | Lean | Differences |
| --- | --- | --- |
| Proof, item 4, affine energy | `affineCorr`, `sum_mul_add_eq`, `norm_affineCorr_le`, `sum_star_mul_shift`, `sum_autocorr_mul`, `affineCorr_mul_star`, `sum_sum_affineCorr_mul_star`, `sum_sum_norm_sq_affineCorr` | The sum includes `λ = 0`, where `E_h(0, μ) = 0` for centered `h`. |
| Proof, item 5, slopes and degenerate dilations | `dilationDiff`, `eval_dilationDiff`, `dilationSlope`, `IsDegenerate`, `sum_comp_le_of_card_fiber_le`, `pow_eq_one_of_isDegenerate`, `card_slope_fiber_le`, `card_le_of_forall_isDegenerate` | |
| Proof, item 6, off-diagonal energy | `sum_sum_norm_sq_affineCorr_slope_le` | |
| Proof, items 1–2, transport | `fiberSum`, `sum_sum_poly_add_mul_eq`, `sum_fiber_mul_star`, `sum_fiberSum_mul_star` | |
| Theorem C.1 | `sum_norm_sq_sum_poly_add_mul_le_sub_one` | |
| Theorem C.2 | `centeredDelta`, `centeredDelta_mul`, `sum_centeredDelta`, `sum_sq_centeredDelta_pos`, `sum_norm_sq_centeredDelta_add`, `sum_norm_sq_sum_affine_add_mul_eq` | `ρ(c)` is written `r(c + −P(0))`. `P'(0)` is `P.coeff 1`. |
| Theorem C.3 | `norm_sum_sum_mul_le`, `sum_norm_sq_fiberSum_le`, `norm_sq_sum_poly_add_mul_le`, `sum_norm_sq_sum_poly_add_mul_le` | |
| Theorem C.4 | `exists_sum_norm_sq_eq_iff_natDegree_le_one` | |
| Theorem C.5 | `norm_sq_sum_poly_add_mul_div_le` | When `‖g‖ ‖h‖ = 0` the left side is `0`. |

Not formalized: the numerical maxima above; that training of the real network behaves like the factorization model
(Lemma 1's modeling step; for the pure factorization see *Training reaches the optimal fit*); the note's readout-free conjecture about the least `r_eff`. Theorem C
bounds the largest centered mode, which is a different quantity.

Scope:
- Real readouts are covered, since `ρ` is any complex function of the label.
- Theorem C does not rule out exact structure spread over many modes. For odd `P`, rows `a`
  and `−a` of `cos(2πr f/p)` are equal (*Torch check*). That kernel has rank `(p + 1)/2` and
  is still flat in every mode.

## Exact recovery of a rank-one kernel (added)

Added on 2026-10-28 and proved in Lean. Theorem A says no proper `Ω` recovers `ψ(a³ + a b)`.
This section gives the positive side for `ψ(a + b)`. The model of Lemma 1 recovers it exactly
from `2p` cells on two diagonals, and not from the `p` cells of one diagonal. Items 7–10 mark
the limits: a pattern must be rigid, a rigid pattern that determines `ψ(a + b)` can still
fail, and a large missed block leaves item 2 without a certificate. Theorem E: M recovers
`ψ(a + b)` from `Ω` exactly when `Ω` is rigid and contains a permutation graph, with or
without a certificate. Theorem F replaces the permutation by a transport plan between the
weights `|a_i|²` and `|b_j|²`, for any `a bᵀ` without zero entries. Theorem G extends it to any
`a ∈ ℂ^m` and `b ∈ ℂ^n`, with rigidity needed only on the supports.

### Setting

- Model M: given the values of `K` on `Ω ⊆ G × G`, return a matrix of least nuclear norm among
  those equal to `K` on `Ω`. M recovers `K` from `Ω` when `K` is the only such matrix.
- `J` is the all-ones `N × N` matrix, `Q = J/N` the projection onto the constant vectors and
  `Π = 1 − Q`. A matrix `W` is a contraction when `‖W v‖ ≤ ‖v‖` for every `v`.
- The diagonal of offset `s` is `{(a, a + s)}`. It has `p` cells and is the graph of the
  permutation `a ↦ a + s`.
- `Ω` is rigid when `r_i + s_j = 0` on `Ω` forces `r_i + s_j = 0` everywhere. This holds
  exactly when the bipartite graph on rows and columns, with an edge for each cell of `Ω`, is
  connected.
- The cross of `(i₀, j₀)` is row `i₀` together with column `j₀`: `2N − 1` cells.

### Theorem D (proven)

1. Certificate bound. Let `Y` vanish off `Ω` and have unit row and column sums. Let `c (Y − Q)`
   be a contraction for some `c > 0`. Then every `H` vanishing on `Ω` has
   `N + (1 − 1/c) ‖Π H Π‖_S1 ≤ ‖J + H‖_S1`.
2. Uniqueness. Let also `c > 1`, and let `Ω` be rigid: `r_i + s_j = 0` on `Ω` only if
   `r_i + s_j = 0` everywhere. Then M recovers `J` from `Ω`.
3. Two permutations. Let `Ω` contain the graphs of permutations `σ` and `τ`, with `τ σ⁻¹` a
   single `N`-cycle. Then `Y = (P_σ + P_τ)/2` has `‖(Y − Q) v‖² ≤ (1 − 1/(4N²)) ‖v‖²`, so item 1
   holds with `c = √(1 + 1/(4N²))`. Such an `Ω` is rigid, so M recovers `J` from it.
4. Transfer. If M recovers `J` from `Ω`, it recovers every `K = a bᵀ` with
   `|a_i| = |b_j| = 1` from the same `Ω`. In particular item 3's `Ω` does.
5. Two diagonals. Let `Ω` contain the diagonals of offsets `s₁` and `s₂`, with `s₂ − s₁` a unit
   of `ℤ/p`. For prime `p`, that means `s₁ ≠ s₂`. Then M recovers `ψ(a + b)` from `Ω`. The two
   diagonals are `2p` of the `p²` cells.
6. One diagonal is not enough. Let `Ω` be exactly the graph of one permutation `σ`, with
   `N ≥ 2`. Then `K = a bᵀ` and the different matrix `Z = diag(a) P_σ diag(b)` both have the
   least nuclear norm, `N`. So one diagonal does not recover `ψ(a + b)`.
7. Rigidity is necessary. Let `r_i + s_j = 0` on `Ω` with `r_{i₀} + s_{j₀} ≠ 0`, and let
   `|a_i| = |b_j| = 1`. Then a different `K' = a' b'ᵀ`, with `|a'_i| = |b'_j| = 1`, equals
   `K = a bᵀ` on `Ω`. No method that sees only the values on `Ω` can return both, and M
   recovers `K` from no such `Ω`.
8. Two diagonals, sharp. Let `Ω` be exactly the diagonals of offsets `s₁` and `s₂`. Then M
   recovers `ψ(a + b)` from `Ω` if and only if `s₂ − s₁` is a unit of `ℤ/p`.
9. The cross. Let `Ω` be the cross of `(i₀, j₀)`, `N ≥ 2` and `|a_i| = |b_j| = 1`. Then `Ω` is
   rigid, and every rank-one `a' b'ᵀ` equal to `K = a bᵀ` on `Ω` is `K`. Yet the least nuclear
   norm of the matrices equal to `K` on `Ω` is `2√(N − 1)`. For `N ≥ 3` this is below
   `N = ‖K‖_S1`, so `K` is not even a minimum. For `ψ(a + b)` and `p ≥ 3`, column 0 holds
   `ψ(a)` and row 0 holds `ψ(b)`. They fix `ψ(a + b) = ψ(a) ψ(b)`, yet M returns a matrix of
   norm `2√(p − 1) < p`.
10. Missed blocks. Let `Ω` miss `R × C`, with `k = |R|` and `l = |C|`. Let `Y` vanish off `Ω`
    with unit row and column sums, and let `c (Y − Q)` be a contraction. Then
    `c² k l ≤ (N − k)(N − l)`. So if `R` and `C` are nonempty and `k + l ≥ N`, no `c > 1`
    works and item 2 has no certificate. This happens when a row meets `Ω` in a single cell:
    `k = 1`, `l = N − 1`.
11. The sharp constant. In item 3, let `N ≥ 2`, and number the cycle by a bijection
    `e : ℤ/N → n` with `τ(σ⁻¹(e l)) = e(l + 1)`. Then `‖Y − Q‖_op = cos(π/N)`, reached at
    `v(i) = ψ(e⁻¹(i))`. So `c (Y − Q)` is a contraction exactly when `|c| cos(π/N) ≤ 1`. For
    `N ≥ 3`, item 1 holds with `c = 1/cos(π/N)`: `N + (1 − cos(π/N)) ‖Π H Π‖_S1 ≤ ‖J + H‖_S1`,
    and no larger `c` makes `c (Y − Q)` a contraction. In particular, for `N ≥ 2` and offsets
    `s₁`, `s₂` of `ℤ/N` with `s₂ − s₁` a unit, the certificate of the two diagonals has
    `‖Y − Q‖_op = cos(π/N)`.

### Proof

1. Item 1. Take a contraction `Z₀` with `tr(Z₀ᴴ Π H Π) = ‖Π H Π‖_S1`. Then `Q + Π Z₀ Π` is a
   contraction, and its pairing with `J + H` is `N + tr(Q H) + ‖Π H Π‖_S1`.
   - `H` vanishes on `Ω` and `Y` vanishes off it, so `tr(Yᴴ H) = 0`.
   - The unit row and column sums give `Π (Y − Q)ᴴ Π = (Y − Q)ᴴ`.
   - Hence `tr(Q H) = −tr((Y − Q)ᴴ Π H Π)`, of modulus at most `‖Π H Π‖_S1 / c`.
2. Item 2. Let `X` be a minimum and `H = X − J`. Then `‖X‖_S1 ≤ ‖J‖_S1 = N`, so item 1 gives
   `Π H Π = 0`. So `H = Q H + Π H Q` has entries `r_i + s_j`, and rigidity forces `H = 0`.
3. Item 3. Write `w = v − mean(v)`. Then `(Y − Q) v = (w∘σ + w∘τ)/2`.
   - The parallelogram law gives `‖(Y − Q) v‖² = ‖w‖² − D/4`, with
     `D = ∑_i |w(σ i) − w(τ i)|²`.
   - A Poincaré inequality along the cycle gives `‖w‖² ≤ N (N − 1) D` for a centered `w`.
   - Hence `‖(Y − Q) v‖² ≤ (1 − 1/(4N²)) ‖w‖² ≤ (1 − 1/(4N²)) ‖v‖²`.
   - Rigidity: `r_i + s(σ i) = 0 = r_i + s(τ i)` makes `s` constant along the cycle.
4. Item 4. The map `X ↦ diag(a) X diag(b)` keeps the nuclear norm and sends `J` to `K`. It also
   maps the matrices equal to `J` on `Ω` onto those equal to `K` on `Ω`.
5. Item 5. Take `σ = (· + s₁)` and `τ = (· + s₂)`. Then `τ σ⁻¹ = (· + (s₂ − s₁))` is one
   `p`-cycle, through `e(l) = l (s₂ − s₁)`.
6. Item 6. `Z` is unitary, so it is a contraction and `‖Z‖_S1 = N`.
   - Each `X` equal to `K` on the graph has `tr(Zᴴ X) = ∑_i |a_i b_{σ i}|² = N`, so
     `‖X‖_S1 ≥ N`.
   - Both `K` and `Z` reach `N`. They differ because `Z` is zero off the graph.
7. Item 7. Divide `r` and `s` by `w = r_{i₀} + s_{j₀}` and take real parts `ρ`, `σ`.
   - Then `ρ_i + σ_j = 0` on `Ω` and `ρ_{i₀} + σ_{j₀} = 1`.
   - The phases `u_i = e^{iπρ_i}` and `v_j = e^{iπσ_j}` have `u_i v_j = 1` on `Ω` and
     `u_{i₀} v_{j₀} = −1`.
   - So `K' = (u a)(v b)ᵀ` equals `K` on `Ω` and differs from it at `(i₀, j₀)`.
   - Both have nuclear norm `N`. If M recovered `K`, then `K'` would be a second minimum.
8. Item 8. If `g = s₂ − s₁` is a unit, item 5 applies. Otherwise:
   - The multiples `H` of `g` miss `1`, since `k g = 1` would make `g` a unit.
   - `r = 1_H` and `s = −1_{s₁ + H}` vanish on both diagonals, because `a + g ∈ H` exactly
     when `a ∈ H`.
   - But `r_0 + s_{s₁ + 1} = 1`, so item 7 applies.
9. Item 9.
   - Rigidity: `r_i + s_j = (r_i + s_{j₀}) + (r_{i₀} + s_j) − (r_{i₀} + s_{j₀})`.
   - Rank one: `a'_i b'_j · a'_{i₀} b'_{j₀} = a'_i b'_{j₀} · a'_{i₀} b'_j`, and
     `a_{i₀} b_{j₀} ≠ 0` can be cancelled.
   - Lower bound: let `m = √(N − 1)` and let `Z` be `1/m` on the cross except at `(i₀, j₀)`,
     zero elsewhere. Cauchy–Schwarz gives
     `‖Z v‖² = |∑_{j ≠ j₀} v_j|²/(N − 1) + |v_{j₀}|² ≤ ‖v‖²`. `Z` pairs with every `X` equal
     to `J` on the cross to `2(N − 1)/m = 2m`.
   - Upper bound: `X₀`, equal to 1 on the cross and `1/(N − 1)` elsewhere, is `U Vᴴ`, with
     `U = [α f, β g]` and `V = [α f, −β g]`. Here `f = e + 1'/m`, `g = e − 1'/m`,
     `α² = (m + 1)/2` and `β² = (m − 1)/2`, where `1'` is the indicator of the other indices.
     Both factors have `‖·‖²_F = 2m`, so `‖X₀‖_S1 ≤ 2m`.
   - The map of item 4 carries both bounds to `K`. Finally, `2√(N − 1) < N` exactly when
     `(N − 2)² > 0`.
10. Item 10. Let `u = 1_R − k/N` and `v = 1_C − l/N`, so `‖u‖² = k (N − k)/N` and
    `‖v‖² = l (N − l)/N`.
    - `Q v = 0`, and the unit row sums give `(Y v)_i = ∑_{j ∈ C} Y_ij − l/N`.
    - `∑_i u_i = 0`, `Y` vanishes on `R × C`, and the unit column sums give
      `∑_i ∑_{j ∈ C} Y_ij = l`. Hence `⟨u, (Y − Q) v⟩ = −k l/N`.
    - Cauchy–Schwarz and the contraction give `|c| k l/N ≤ ‖u‖ ‖v‖`. Squaring and dividing by
      `k l` gives the bound; it is trivial when `k l = 0`.
    - When `k + l ≥ N`, `(N − k)(N − l) = N (N − k − l) + k l ≤ k l`, so `c² ≤ 1`.
11. Item 11. Along the cycle, `((Y − Q) v)(σ⁻¹(e l)) = ((Y₀ − Q₀)(v ∘ e))(l)`, where on `ℤ/N`
    `(Y₀ w)(l) = (w(l) + w(l + 1))/2` and `Q₀ = J/N`. Both reindexings keep the norm.
    - The columns `l ↦ ψ(k l)` of the Fourier matrix `F` are eigenvectors of `Y₀ − Q₀`, with
      eigenvalues `λ_k = (1 + ψ(k))/2` for `k ≠ 0` and `λ_0 = 0`. As `F* F = F F* = N I`,
      `‖(Y₀ − Q₀) w‖ ≤ max_k |λ_k| ‖w‖`.
    - `|λ_k| = |cos(π k/N)|` for `k ≠ 0`, at most `cos(π/N)`, with equality at `k = 1`. So the
      bound is `cos(π/N)`, reached at `w = ψ`, that is `v = ψ ∘ e⁻¹`.
    - `‖c (Y − Q) v‖ ≤ ‖v‖` for every `v` exactly when `|c| cos(π/N) ≤ 1`, and item 1 applies
      with `c = 1/cos(π/N)`, positive for `N ≥ 3`.

### Numerical check (torch, 2026-10-28)

The check runs ADMM for model M (singular value thresholding, step `1/√p`) in complex128. It
stops when both residuals fall below `10⁻⁹`, or after 20000 iterations. The error is
`‖X − K‖/‖K‖` on the unseen cells.

| `Ω` for `ψ(a + b)` | cells | `p = 7` | `p = 13` | `p = 31` |
| --- | --- | --- | --- | --- |
| diagonals `{0, 1}` | `2p` | 2.4e-10 | 1.3e-10 | 1.1e-10 |
| diagonals `{0, 3}` | `2p` | 2.4e-10 | 1.3e-10 | 1.1e-10 |
| diagonal `{0}` | `p` | 1.0 | 1.0 | 1.0 |
| diagonals `{0, 1}` minus one cell | `2p − 1` | 2.0e-9 | 2.5e-9 | 2.1e-4, not converged |
| row 0 and column 0 | `2p − 1` | 0.83 | 0.92 | 0.97 |
| random | `2p` | 0.45 | 0.58 | 0.66 |
| random | `4p` | 4.6e-10 | 6.7e-10 | 0.19 |
| random | `p²/2` | 2.7e-5 | 1.6e-10 | 5.8e-11 |

- Recovered completions reach `‖X‖_S1 = p = ‖K‖_S1`.
- On one diagonal, ADMM returns the matrix `Z` of item 6, with the same norm `p`.
- On row 0 and column 0, it returns a matrix of norm `2√(p − 1) < p`, so `K` is not even a
  minimum there. The `2p` random cells also give norms below `p`: 6.47, 11.07 and 26.33.
- `ψ(a³ + a b)` has `‖K‖_S1 = p^{3/2}`. Its error is 1.0 on one diagonal and between 1.001
  and 1.405 on every other pattern above, half the table included.
- The certificate of item 3 for the diagonals `{0, 1}` has `‖Y − Q‖_op` 0.900969, 0.970942
  and 0.994869, the values of `cos(π/p)` (item 11). Item 3's bound `√(1 − 1/(4p²))` is
  0.997446, 0.999260 and 0.999870.
- At `p = 61` and `p = 97` the cap was raised to 200000 iterations.
  - The diagonals `{0, 1}`, `{0, 5}` and `{3, (p − 1)/2}` give errors 8.8e-11 and 6.6e-11,
    after 6813 and 20197 iterations.
  - The three pairs give the same numbers, since relabeling and rescaling map one problem onto
    another.
  - One diagonal gives error 1.0: ADMM returns the matrix `Z` of item 6, zero off `Ω`.
  - `‖Y − Q‖_op = cos(π/p)` is 0.998674 and 0.999476 there.

### Patterns (torch, 2026-10-28)

Which patterns recover `ψ(a + b)`? ADMM is run as above, with a cap of 20000 iterations, for
`p = 31, 61, 97`. For each `Ω` the run reports:
- whether `Ω` is rigid;
- `‖Y − Q‖_op` for the least-Frobenius `Y` that vanishes off `Ω` with unit row and column sums
  (`Y_ij = u_i + v_j` on `Ω`). Below 1 it is item 2's certificate, with `c = 1/‖Y − Q‖_op`.
  "None" means no such `Y` exists;
- the error off `Ω`, and `‖X‖_S1`.

The training split of `grok-simple.py` is reproduced exactly:
- the rows `(a, b)` are listed `a`-major;
- they are shuffled by SplitMix64 seeded with `seed ^ 0x73706C6974730002`;
- the first `⌊p²/2⌋` rows are for training.

Every operation uses the same `Ω` for a given seed. The random patterns keep each cell with the
given probability, three draws each.

| `Ω` at `p = 97` | cells | rigid | `‖Y − Q‖_op` | error | `‖X‖_S1` |
| --- | --- | --- | --- | --- | --- |
| diagonals `{0, 1}` | 194 | yes | 0.9995 | 4.5e-10 | 97.000 |
| row 0 and column 0 | 193 | yes | 96 | 0.99 | 19.596 |
| training split, seed 7 | 4704 | yes | 0.2091 | 1.8e-11 | 97.000 |
| training split, seed 11 | 4704 | yes | 0.2087 | 1.4e-11 | 97.000 |
| random 2% | 185–193 | no | none | 0.61–0.70 | 82.4–84.8 |
| random 4% | 373–380 | no | none | 0.21–0.26 | 93.8–94.9 |
| random 6%, draws 0 and 1 | 546, 555 | no | none | 0.15 | 96.0 |
| random 6%, draw 2 | 536 | yes | 1.1169 | 5.4e-5, not converged | 97.000 |
| random 8% | 717–736 | yes | 1.003–1.008 | 2.2e-5–2.4e-5, not converged | 97.000 |
| random 12% | 1083–1127 | yes | 0.72–0.73 | 7e-12–7e-11 | 97.000 |
| random 20% | 1802–1921 | yes | 0.42–0.44 | 2.3e-11–3.5e-11 | 97.000 |
| random 30% | 2716–2888 | yes | 0.31–0.33 | 3e-12–2e-11 | 97.000 |

Over the three primes (63 random draws):
- The training splits are rigid. Their `‖Y − Q‖_op` is 0.37 and 0.38 at `p = 31`, 0.26 and
  0.25 at `p = 61`, 0.21 and 0.21 at `p = 97`.
  - So item 2 applies to the network's own training data.
  - ADMM recovers `ψ(a + b)` from them with error below 7e-11.
- On the same splits `ψ(a³ + a b)` fails, as Theorem A says. The error is 1.29–1.30 and
  `‖X‖_S1 ≈ 0.55 p^{3/2}` for all six splits.
- All 32 non-rigid draws fail, with error 0.13 to 1.0.
- All 21 rigid draws with `‖Y − Q‖_op < 1` are recovered, with error at most 1.1e-10 within
  310 iterations.
- The other 10 rigid draws split by Hall's condition:
  - One, `p = 61` at 8%, draw 1, contains no permutation graph: its maximum matching has 60
    cells. It fails, with `‖X‖_S1 = 60.912 < 61` and error 0.056.
  - The other nine contain a permutation graph. Each has a row or a column with a single cell.
    ADMM stops at the cap with `‖X‖_S1 = p` to three decimals and error 2.2e-5–5.4e-5.
  - Projected subgradient descent over all admissible `Y` finds 1.0000 for all nine and 1.5566
    for the failure, never below 1.
- All 21 recovered draws and all six training splits are elementary: connected, with every cell
  on a permutation graph inside `Ω`. Checked by a matching and strong connectivity.

The block bound, item 10, and the witness of Theorem E.3 explain the split:
- Let `Ω` miss a block `R × C` with `|R| = k` and `|C| = l`, both nonempty and proper.
  - Take any `Y` vanishing off `Ω` with unit row and column sums.
  - Pair `Y − Q` with the centered indicators of `R` and `C`. This gives
    `1_Rᵀ (Y − Q) 1_C = −k l/N`.
  - Hence `‖Y − Q‖_op ≥ √(k l / ((N − k)(N − l)))`.
  - This is at least 1 when `k + l ≥ N`, and above 1 when `k + l > N`.
  - Such a block with `k + l > N` exists exactly when Hall's condition fails, that is, when
    `Ω` contains no permutation graph. Lean proves the direction Theorem E.3 uses.
- A row with a single cell is the case `k = 1`, `l = N − 1`, so no certificate is below 1.
  This also covers two diagonals minus a cell.
- Two rows with a single cell in the same column give `k = 2`, `l = N − 1`. That is the
  failure, with bound 1.4261.
- When `k + l > N`, the matrix `X = J − t 1_R 1_Cᵀ` equals `J` on `Ω` and has rank 2.
  Theorem E.3 proves `‖X‖_S1 < N` for `t = 1/(k + 1)²`. Lean gives the exact norm:
  `‖X‖²_S1 = N² − 2t (k l − g) + t² k l` for `t ≥ 0`, with `g = √(k l (N − k)(N − l))`
  (`nuclearNorm_onesSubBlock_sq_of_nonneg`).
  - The best `t`, `1 − g/(k l)`, gives `‖X‖²_S1 = N² − (k l − g)²/(k l) < N²`, that is
    `‖X‖_S1 = √(k (N − l)) + √(l (N − k))` (`nuclearNorm_onesSubBlock_best_sq`,
    `nuclearNorm_onesSubBlock_best`, `isLeast_nuclearNorm_onesSubBlock`).
  - Some `t` gives `‖X‖_S1 < N` exactly when `k + l > N`
    (`exists_nuclearNorm_onesSubBlock_lt_card_iff`).
  - At `k = 2`, `l = 60`, `N = 61` the best norm is `√2 + √3540`, between 60.912 and 60.913
    (`isLeast_nuclearNorm_onesSubBlock_sixtyOne`). ADMM found 60.912, the same to three
    decimals.
  - For the cross, `k = l = N − 1`, it is item 9's `2√(N − 1)`
    (`isLeast_nuclearNorm_onesSubBlock_cross`).

Between the two cases, Theorem D has no certificate. Longer runs at `p = 31` keep lowering the
error, while `‖X‖_S1 − p` stays positive and shrinks:
- 12%, draw 1: 3.4e-5, 8.2e-6 and 3.1e-6 after 20000, 100000 and 300000 iterations;
- 20%, draw 0: 2.5e-5, 5.1e-6 and 1.7e-6;
- two diagonals minus a cell: 2.1e-4 after 20000 iterations, then 3.7e-9 when both residuals
  fall below `10⁻⁹` after 67491.

This looks like slow convergence to `K`, not a second minimum. Theorem E shows that M does
recover these patterns: each is rigid and contains a permutation graph. For the random draws
this is checked numerically; for two diagonals minus a cell it is proved (row *Two diagonals
minus a cell* of the Lean status).

### Theorem E (proven)

1. Lower bound. Let `Ω` contain the graph of a permutation `σ`. Then every `X` equal to `J` on
   `Ω` has `‖X‖_S1 ≥ N`, so `J` is a minimum.
2. Uniqueness. Let also `Ω` be rigid. Then M recovers `J` from `Ω`. By item D.4 it recovers
   every `K = a bᵀ` with `|a_i| = |b_j| = 1`, and for `G = ℤ/p` it recovers `ψ(a + b)`.
3. Witness. Let `Ω` miss a block `R × C` with `|R| = k`, `|C| = l` and `k + l > N`. Then
   `X = J − s² 1_R 1_Cᵀ` equals `J` on `Ω`, and `‖X‖_S1 < N` for `s = 1/(k + 1)`. If `Ω`
   contains no permutation graph, such a block exists, so `J` is not a minimum.
4. Characterization. Let `|a_i| = |b_j| = 1`. M recovers `a bᵀ` from `Ω` if and only if `Ω`
   is rigid and contains the graph of a permutation. This covers `J` and, for `G = ℤ/p`,
   `ψ(a + b)`.

Item 2 needs no certificate with `c > 1`. So it covers the patterns of item D.10, where every
admissible `Y` has `‖Y − Q‖_op ≥ 1`, such as a rigid pattern with a single-cell row. The
smallest is `{(0, 0), (0, 1), (1, 1)}` for `N = 2`.

### Proof of Theorem E

1. Item 1. By Lemma 1, `X = U Vᴴ` with `½ (‖U‖² + ‖V‖²) = ‖X‖_S1`. Let `u_i` and `v_j` be the
   rows of `U` and `V`, so `X_ij = ⟨u_i, v_j⟩ = ∑_k U_ik conj(V_jk)`.
   - `‖u − v‖² = ‖u‖² + ‖v‖² − 2 Re⟨u, v⟩`, and `⟨u_i, v_{σ i}⟩ = X_{i, σ i} = 1`.
   - Summing over `i`, `∑_i ‖u_i − v_{σ i}‖² = ‖U‖² + ‖V‖² − 2N = 2 (‖X‖_S1 − N)`. This is
     `≥ 0`.
2. Item 2. Let `X` be a minimum, so `‖X‖_S1 ≤ ‖J‖_S1 = N`.
   - The sum above is then `0`, so `u_i = v_{σ i}` for every `i`.
   - Hence `1 = X_{i, σ i} = ‖u_i‖²`, so every `u_i` and every `v_j` is a unit vector.
   - On a cell `(i, j)` of `Ω`, `‖u_i − v_j‖² = 1 + 1 − 2 = 0`, so `u_i = v_j`.
   - Rigidity, applied to each coordinate `k` with `r_i = U_ik` and `s_j = −V_jk`, gives
     `u_i = v_j` for all `i` and `j`.
   - So `X_ij = ‖u_i‖² = 1`, that is, `X = J`.
3. Item 3. Let `U` have the rows `(1 − s², s)` on `R` and `(1, 0)` elsewhere, and `V` the rows
   `(1, 0)` on `C` and `(1, s)` elsewhere.
   - Then `U Vᴴ = J − s² 1_R 1_Cᵀ`. It equals `J` on `Ω`, since `Ω` misses `R × C`.
   - `‖U‖² = N − k s² + k s⁴` and `‖V‖² = N + (N − l) s²`.
   - By Lemma 1, `‖X‖_S1 ≤ ½ (‖U‖² + ‖V‖²) = N − ½ s² (k + l − N − k s²)`.
   - For `s = 1/(k + 1)`, `k s² < 1 ≤ k + l − N`, so `‖X‖_S1 < N`.
   - If `Ω` contains no permutation graph, Hall's theorem gives rows `R` with fewer than `k`
     neighbouring columns. The other columns form `C`, with `l > N − k`.
4. Item 4. Sufficiency is item 2. If `Ω` is not rigid, item D.7 gives a second minimum. If
   `Ω` contains no permutation graph, take `Y` from item 3. Then `diag(a) Y diag(b)` equals
   `a bᵀ` on `Ω` and has the norm of `Y` (item D.4), below `N = ‖a bᵀ‖_S1`.

### The failure margin (proven)

Theorem E.3 gives a completion of `J` shorter than `N` when `Ω` contains no permutation graph.
How much shorter, sharply:

1. Least norm off a block. In an `m × n` table, let `R` be `k` rows and `C` be `l` columns.
   Every `X` equal to 1 on `R × Cᶜ` and on `Rᶜ × C` has
   `‖X‖_S1 ≥ √(k (n − l)) + √(l (m − k))`. For nonempty `R` and `C` with `m n ≤ k n + l m`,
   the best witness `J − t 1_R 1_Cᵀ` of *Patterns* has this norm. So it is the least norm of
   the matrices equal to `J` off `R × C`.
2. The margin. Let `Ω ⊂ N × N` contain no permutation graph. Then `Ω` misses a block `R × C`
   with `k + l = N + 1`, and some `X` equal to `J` on `Ω` has
   `‖X‖_S1 = √(k (N − l)) + √(l (N − k)) ≤ √(N² − 1) < N − 1/(2N)`. So the least norm on `Ω`
   is below `‖J‖_S1 = N` by more than `1/(2N)`.
3. Sharpness. Let `k + l > N`. The cells off a `k × l` block contain no permutation graph, and
   by item 1 the least norm on them is `√(k (N − l)) + √(l (N − k))`. With item 2, the largest
   least norm over the patterns without a permutation graph is the largest of these values with
   `k + l = N + 1`. It is reached where `k` and `l` differ by at most one: for `N ≥ 1` and
   `a + b = N + 1` with `a ≤ b ≤ a + 1`, it is `√(a (a − 1)) + √(b (b − 1))`.
   - Odd `N`: `√(N² − 1)`, at `a = b = (N + 1)/2`.
   - `N = 2m`: `√(m (m − 1)) + √(m (m + 1))`; for `N = 4`, `√2 + √6`, between 3.8637 and 3.8638.
   - Both are `N − 1/(2N) − O(1/N³)`, so the bound `N − 1/(2N)` of item 2 is sharp to first
     order.
4. Every table of rank one with entries of modulus one. Let `N ≥ 1` and `K = u vᵀ` be `N × N`
   with every `|u_i v_j| = 1`; then `‖K‖_S1 = N`. Items 2 and 3 hold with `K` in place of `J`.
   If `Ω` contains no permutation graph, some `X` equal to `K` on `Ω` has `‖X‖_S1 ≤ √(N² − 1)`
   and `‖X‖_S1 < ‖K‖_S1 − 1/(2N)`. Every such `Ω` has a completion of `K` of norm at most
   `√(a (a − 1)) + √(b (b − 1))`, and some such `Ω` has it as least norm: `√(N² − 1)` for odd
   `N`, `√2 + √6` for `N = 4`.

### Proof of the failure margin

1. Item 1. Let `Z` be `c₁ = 1/√(k (n − l))` on `R × Cᶜ`, `c₂ = 1/√(l (m − k))` on `Rᶜ × C`,
   and 0 elsewhere; a `c` is 0 when its product is 0.
   - `(Z v)_i` is `c₁ ∑_{Cᶜ} v_j` for `i ∈ R` and `c₂ ∑_C v_j` otherwise. By Cauchy–Schwarz,
     `‖Z v‖² ≤ c₁² k (n − l) ∑_{Cᶜ} |v_j|² + c₂² l (m − k) ∑_C |v_j|² ≤ ‖v‖²`. So `Z` is a
     contraction.
   - `X` is 1 wherever `Z` is not 0, so `tr(Zᴴ X) = c₁ k (n − l) + c₂ l (m − k)`. This is
     `√(k (n − l)) + √(l (m − k))`.
   - `|tr(Zᴴ X)| ≤ ‖X‖_S1` for a contraction `Z` gives the bound. The best witness reaches it
     (*Patterns*).
   - Item D.9's certificate is the case `R = {i₀}ᶜ`, `C = {j₀}ᶜ`, `c₁ = c₂`.
2. Item 2. Hall's theorem gives rows `R` that meet fewer than `k = |R|` columns. The other
   columns number more than `N − k`; keep `N + 1 − k` of them as `C`.
   - The best witness for `R × C` equals `J` on `Ω` and has the norm above.
   - By Cauchy–Schwarz, `√(x₁ y₁) + √(x₂ y₂) ≤ √((x₁ + x₂)(y₁ + y₂))` for `x, y ≥ 0`. With
     `x₁ = k`, `y₁ = N − l`, `x₂ = l`, `y₂ = N − k` the right side is
     `√(N² − (k + l − N)²) = √(N² − 1)`.
   - `(N − 1/(2N))² = N² − 1 + 1/(4N²)`.
3. Item 3. A permutation `σ` with every `(i, σ i)` off `R × C` maps `R` into `Cᶜ`, so
   `k ≤ N − l`. The least norm is item 1 with `m = n = N`, whose condition `N² ≤ (k + l) N`
   holds.
   - For `k + l = N + 1` the value is `√(k (k − 1)) + √(l (l − 1))`. Its square is
     `N (N + 1) − 2p + 2 √(p (p − N))` with `p = k l`, and `p ≥ N` since
     `(k − 1)(l − 1) ≥ 0`.
   - `√(p (p − N)) − p` grows with `p ≥ N`: for `N ≤ p ≤ q`,
     `q (q − N) − (√(p (p − N)) + q − p)² = (q − p)(2p − N − 2 √(p (p − N)))`, and
     `2 √(p (p − N)) ≤ 2p − N`.
   - `a b − k l = (a − k)(b − k) ≥ 0`, since no integer lies strictly between `a` and `b`.
   - Odd `N`: `2 √(a (a − 1)) = √((2a − 1)² − 1) = √(N² − 1)`. `N = 4`: `a = 2`, `b = 3`, and
     `√2 + √6` lies between 3.8637 and 3.8638.
   - The expansions: `N − √(N² − 1) = 1/(2N) + 1/(8N³) + …` and, for `N = 2m`,
     `N − √(m (m − 1)) − √(m (m + 1)) = 1/(2N) + 5/(8N³) + …`.
4. Item 4. For a row `i₀`, `u/|u_{i₀}|` and `|u_{i₀}| v` have the product `K` and entries of
   modulus one, since `|u_i| |v_j| = 1 = |u_{i₀}| |v_j|`. So let `|u_i| = |v_j| = 1`.
   - `X ↦ diag(u) X diag(v)` keeps the nuclear norm (item D.4). It maps the matrices equal to
     `J` on `Ω` onto those equal to `K` on `Ω`, with inverse `X ↦ diag(ū) X diag(v̄)`. So both
     sets have the same nuclear norms, and items 2 and 3 carry over.
   - `‖K‖_S1 = ‖diag(u) J diag(v)‖_S1 = ‖J‖_S1 = N`.

### Exhaustive check (torch, 2026-10-28)

Every proper pattern `Ω ⊂ {0, 1, 2}²`, 511 in all, is classified by rigidity and by Hall's
condition. Batched ADMM on `J` then runs 20000 iterations for each. The error is the RMS of
`X − J` off `Ω`.

| `N = 3` | patterns | `‖X‖_S1 − N` | error |
| --- | --- | --- | --- |
| rigid, permutation graph | 195 | within 5e-15 of 0 | at most 4.8e-14 |
| rigid, no permutation graph | 9 | −0.172 | 0.5 |
| not rigid, permutation graph | 51 | within 9e-16 of 0 | 0.89–1.00 |
| not rigid, no permutation graph | 256 | −3.00 to −0.172 | 0.77–1.00 |

The four rows are Theorem E.2, E.3, E.1 with D.7, and E.3. The nine rigid patterns without a
permutation graph are the crosses, and `−0.172 = 2√2 − 3` is item D.9's least norm. It is also
`√8 − 3`, the largest least norm without a permutation graph (*The failure margin*, odd `N`).

For `N = 4`, all 65 535 proper patterns, 20000 iterations each, on the GPU:

| `N = 4` | patterns | `‖X‖_S1 − N` | error |
| --- | --- | --- | --- |
| rigid, permutation graph | 33 852 | within 7e-7 of 0 | at most 8.5e-5 |
| rigid, no permutation graph | 2464 | −0.536 to −0.136 | 0.345–0.667 |
| not rigid, permutation graph | 3970 | within 2e-14 of 0 | 0.775–1.00 |
| not rigid, no permutation graph | 25 249 | −4.00 to −0.136 | 0.632–1.00 |

- The ends −0.536 and −0.136 agree to three decimals with `2√3 − 4`, item D.9's least norm,
  and `√2 + √6 − 4`, the largest least norm without a permutation graph (*The failure
  margin*, `N = 4`). The patterns that are not rigid reach the same end: for `k = 2`, `l = 3`
  the union of `R × Cᶜ` and `Rᶜ × C` is not rigid, and its least norm is again `√2 + √6`,
  since item 1 needs `X = 1` only there.
- Theorem E.2 makes `J` the only solution on the first row, so its error up to 8.5e-5 is
  ADMM's slowness. It comes from the 27 068 of these patterns in which a proper set of rows
  meets exactly as many columns as it has rows; on the other 6784 the error is below 1e-13, a
  rounding error.
- The port's rerun (`experiments/recovery` in the grokking repository, `slow 4`, 2026-10-29)
  leaves 4464 of the 27 068 above 1e-6 after 20000 iterations. Run on, their error stays below
  3.5e-5, 1.8e-5, 8.9e-6 and 4.5e-6 after 50000, 100000, 200000 and 400000 iterations, about
  half each time the iterations double. Their gap stays positive, below 7.9e-9 at the end.

### Margin check (torch, 2026-10-29)

`recovery margin 20000 --sizes 3 4 5 6 7 --count 2000` (`experiments/recovery` in the grokking
repository) runs ADMM on `J`, 20000 iterations, for `N = 3` to `7`.

- Off each `k × l` block with `k + l > N`, `‖Z‖_S1` is item 1's least norm
  `√(k (N − l)) + √(l (N − k))` to within 5e-15, a rounding error.
- 2000 random patterns of each size without a permutation graph: each draws a density uniform
  in [0.3, 0.9] and keeps each cell with it (seed 0). A pattern's least block value is the
  least of these values over the blocks with `k + l > N` it misses; item 1's best witness is a
  completion with this norm. `‖Z‖_S1` is never more than 7.7e-7 above it.
- In each size the largest gap `‖Z‖_S1 − N`, rigid and other patterns apart, is to six
  decimals the largest least block value of the drawn patterns minus `N`, so at most item 3's
  largest least norm minus `N`. For `N = 3` to `6` both kinds reach it: `√8 − 3`,
  `√2 + √6 − 4`, `√24 − 5` and `√6 + √12 − 6`, that is −0.171573, −0.136297, −0.101021 and
  −0.086409. For `N = 7` the rigid patterns reach `√48 − 7`, −0.071797, and the others the
  `3 × 5` block's −0.078374.

### Theorem F (proven)

Let no entry of `a` or `b` vanish. Scaling `a` by `t > 0` and `b` by `1/t` keeps `a bᵀ`, so
let `‖a‖ = ‖b‖`; then `‖a bᵀ‖_S1 = ‖a‖ ‖b‖ = ‖a‖²`. Let `α_i = |a_i|²` and `β_j = |b_j|²`. A
transport plan on `Ω` from `α` to `β` is a matrix `T ≥ 0` that vanishes off `Ω`, with row sums
`α_i` and column sums `β_j`.

1. Lower bound. If `Ω` carries a plan, every `X` equal to `a bᵀ` on `Ω` has
   `‖X‖_S1 ≥ ‖a‖²`, so `a bᵀ` is a minimum.
2. Uniqueness. If also `Ω` is rigid, M recovers `a bᵀ` from `Ω`.
3. Converse. M recovers `a bᵀ` from `Ω` only if `Ω` is rigid and carries a plan. By the
   supply–demand theorem (Gale, 1957), a plan exists if and only if
   `∑_{i ∈ R} α_i ≤ ∑_{j ∈ N(R)} β_j` for every set `R` of rows, where `N(R)` is the set of
   columns that meet `R` in `Ω`. Without the scaling the condition reads
   `‖b‖² ∑_R |a_i|² ≤ ‖a‖² ∑_{N(R)} |b_j|²`.
4. The cross. Let `Ω` be row `i₀` and column `j₀`. M recovers `a bᵀ` from `Ω` if and only if
   `|a_{i₀}|²/‖a‖² + |b_{j₀}|²/‖b‖² ≥ 1`.

For `|a_i| = |b_j| = 1` the permutation matrix of a graph inside `Ω` is a plan, so item 2
contains Theorem E.2. The condition of item 3 is then Hall's `|R| ≤ |N(R)|`, and items 2–3
give Theorem E.4. Item 4 then reads `2/N ≥ 1`, that is `N ≤ 2`, as item D.9 found.

### Proof of Theorem F

1. Item 1. Write `X = U Vᴴ` with `½ (‖U‖² + ‖V‖²) = ‖X‖_S1` (Lemma 1), and let
   `x_i = u_i / a_i` and `y_j = v_j / conj(b_j)`. Then `⟨x_i, y_j⟩ = X_ij / (a_i b_j)`, which
   is `1` on `Ω`.
   - Expanding the squares and using the row and column sums of `T`,
     `∑_ij T_ij ‖x_i − y_j‖² = ∑_i α_i ‖x_i‖² + ∑_j β_j ‖y_j‖² − 2 ∑_ij T_ij Re⟨x_i, y_j⟩`.
   - `α_i ‖x_i‖² = ‖u_i‖²` and `β_j ‖y_j‖² = ‖v_j‖²`. `T` vanishes off `Ω`, so
     `T_ij Re⟨x_i, y_j⟩ = T_ij`.
   - So the sum is `‖U‖² + ‖V‖² − 2 ‖a‖² = 2 (‖X‖_S1 − ‖a‖²)`, and it is `≥ 0`.
2. Item 2. Let `X` be a minimum, so `‖X‖_S1 ≤ ‖a bᵀ‖_S1 = ‖a‖²` and the sum above is `0`.
   - `x_i = y_j` wherever `T_ij > 0`.
   - Row `i` carries the mass `α_i > 0`, so `T_ij > 0` for some `j`, and
     `‖x_i‖² = ⟨x_i, y_j⟩ = 1`. Likewise every `y_j` is a unit vector.
   - On a cell `(i, j)` of `Ω`, `‖x_i − y_j‖² = 1 + 1 − 2 = 0`, so `x_i = y_j`.
   - Rigidity, applied to each coordinate as in Theorem E.2, makes all `x_i` and `y_j` equal.
     So `X_ij = a_i ⟨x_i, y_j⟩ b_j = a_i b_j`.
3. Item 3.
   - If `Ω` is not rigid, the phases of item D.7, `u_i v_j = 1` on `Ω` and
     `u_{i₀} v_{j₀} = −1`, give `(u a)(v b)ᵀ`. It equals `a bᵀ` on `Ω` and differs at
     `(i₀, j₀)`. It has the moduli of `a bᵀ`, so the same norm: `‖a bᵀ‖_S1 = ‖a‖ ‖b‖`, since
     `P = (‖a‖/‖b‖) conj(b) bᵀ` is positive semidefinite with `P² = (a bᵀ)ᴴ (a bᵀ)`.
   - If the condition fails at `R`, let `C = N(R)ᶜ`. Then `Ω` misses `R × C` and
     `δ = ∑_R α + ∑_C β − ‖a‖² > 0`. With `U₀`, `V₀` of item E.3, `U = diag(a) U₀` and
     `V = diag(conj b) V₀` give `U Vᴴ = a bᵀ − s² a_R b_Cᵀ`, equal to `a bᵀ` on `Ω`, with
     `½ (‖U‖² + ‖V‖²) = ‖a‖² − ½ s² (δ − s² ∑_R α)`. This is below `‖a‖²` for
     `s = δ / (∑_R α + δ)`, since then `s² ∑_R α < δ`.
   - The supply–demand theorem. A plan sends all of `∑_R α` into `N(R)`, so the condition is
     necessary. For the converse, allow column sums `≤ β`; when `∑ α = ∑ β`, they are `β`.
     Induct on the number of cells; with none, the condition at `{i}` gives `α_i = 0`. Pick a
     cell `(i, j)` and ship along it the largest `t ≤ min(α_i, β_j)` that is at most the slack
     `∑_{N(R)} β − ∑_R α` of every `R ∌ i` with `j ∈ N(R)`. A set `R ∋ i` loses `t` on both
     sides, so the condition still holds. Either row `i` or column `j` has no mass left, and
     the cell goes; or some `R₀ ∌ i` with `j ∈ N(R₀)` is tight, `∑_{N(R₀)} β = ∑_{R₀} α`. Then
     the rows `R₀` keep the condition on their own cells. The other rows keep it on the cells
     outside `N(R₀)`: the condition at `R ∪ R₀` minus the equality at `R₀` gives
     `∑_{R ∖ R₀} α ≤ ∑_{N(R) ∖ N(R₀)} β`. Neither pattern holds `(i, j)`, so both are smaller,
     and their plans add up.
4. Item 4. With `‖a‖ = ‖b‖` the condition reads `‖a‖² ≤ α_{i₀} + β_{j₀}`.
   - If it holds, let each row `i ≠ i₀` send `α_i` to column `j₀`, let row `i₀` send `β_j` to
     each column `j ≠ j₀`, and leave `α_{i₀} + β_{j₀} − ‖a‖² ≥ 0` at `(i₀, j₀)`. This is a
     plan on the cross, and the cross is rigid (item D.9), so item 2 applies.
   - If it fails, the rows `R` other than `i₀` meet only column `j₀`, and
     `∑_R α = ‖a‖² − α_{i₀} > β_{j₀}`. Item 3's witness, with `C` the columns other than
     `j₀`, is shorter than `a bᵀ`.

### Weighted check (torch, 2026-10-28)

For `N = 3`, all 510 nonempty proper patterns, batched ADMM on `K = a bᵀ` for 6000
iterations. The gap is `‖X‖_S1 − ‖a‖ ‖b‖`; the error is the relative norm of `X − K` off `Ω`.
"Hall" is the condition of item 3.

Random weights: moduli uniform in `[0.3, 2]`, random phases, three draws. Each draw gave the
same counts.

| class | patterns | gap | error |
| --- | --- | --- | --- |
| rigid, Hall | 90 | within 7e-15 of 0 | at most 1.8e-15 |
| rigid, no Hall | 114 | −1.63 to −3e-6 | 1.7e-3 to 0.88 |
| not rigid, no Hall | 306 | −6.16 to −3e-6 | 0.44 to 1.00 |

No pattern satisfies Hall without being rigid. For a component with rows `R`, the condition
for `R` and for its complement would force `∑_R α = ∑_{N(R)} β`, which generic weights avoid.

Tight weights: shares `|a_i|²/‖a‖²` and `|b_j|²/‖b‖²` of `(½, ¼, ¼)` for both (A), and of
`(⅗, ⅕, ⅕)` and `(⅕, ⅖, ⅖)` (C). Permuting the shares of `b` (B) changed no count and, to the
digits shown, no range but one: the gap of the non-rigid Hall patterns is within 4e-15 of 0.

| weights | class | patterns | gap | error |
| --- | --- | --- | --- | --- |
| A | rigid, Hall | 144 | 0 to 3.1e-6 | at most 2.4e-4 |
| A | not rigid, Hall | 24 | within 2e-15 of 0 | 0.87 to 1.00 |
| A | rigid, no Hall | 60 | −0.54 to −0.14 | 0.37 to 0.67 |
| A | not rigid, no Hall | 282 | −3.00 to −0.14 | 0.69 to 1.00 |
| C | rigid, Hall | 110 | 0 to 3.9e-6 | at most 2.4e-4 |
| C | not rigid, Hall | 8 | within 3e-15 of 0 | 0.89 to 1.00 |
| C | rigid, no Hall | 94 | −1.00 to −0.10 | 0.29 to 0.75 |
| C | not rigid, no Hall | 298 | −4.00 to −0.10 | 0.63 to 1.00 |

Rigid Hall patterns are recovered (item 2). A Hall pattern that is not rigid reaches
`‖a‖ ‖b‖` (item 1) at another matrix (item 3's phases). Without Hall the least norm falls
below `‖a‖ ‖b‖` (item 3's witness). A rigid Hall pattern at tight weights converges slowly.
For the worst one at A, `{(0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}`, the error
is 2.4e-4, 5.0e-5 and 1.0e-5 after 6000, 30000 and 150000 iterations. Item 2 proves that `K`
is the only minimum there, so the slowness is ADMM's.

### Cross check (torch, 2026-10-28)

Item 4 with `Ω` the row 0 and the column 0, for `N = 3, 4, 5, 6`, 400 draws each. The moduli
are uniform in `[0.3, 2]` with random phases; then `|a_0|` and `|b_0|` are set so that the
shares `α₀ = |a_0|²/‖a‖²` and `β₀ = |b_0|²/‖b‖²` are uniform in `(0.05, 0.95)`. Batched ADMM,
100000 iterations, `ρ = 1/√N`; gap and error as in the weighted check. Item 4 says that M
recovers `a bᵀ` exactly when the margin `m = α₀ + β₀ − 1` is `≥ 0`.

| margin | draws | gap | error |
| --- | --- | --- | --- |
| `m ≤ −0.05` | 723 | −3.88 to −2.6e-3 | 0.099 to 0.94 |
| `−0.05 < m ≤ 0` | 72 | −4.4e-2 to −3.7e-6 | 2.5e-3 to 0.23 |
| `0 < m ≤ 0.05` | 73 | within 1.5e-14 of 0 | at most 1.8e-15 |
| `m > 0.05` | 732 | within 4.3e-14 of 0 | at most 1.6e-15 |

Every draw with `m > 0` is recovered, and every draw with `m ≤ 0` has a completion shorter than
`a bᵀ`. Each `N` splits the same way; the largest gap with `m ≤ 0` is −3.7e-6, −1.4e-4,
−7.0e-5 and −2.5e-5 for `N = 3, 4, 5, 6`.

### Theorem G (proven)

Theorem F for any `a ∈ ℂ^m` and `b ∈ ℂ^n`: the kernel may be rectangular and `a`, `b` may
have zero entries. Let `I = {i : a_i ≠ 0}` and `J = {j : b_j ≠ 0}` be the supports. `Ω` is
rigid on the supports when `r_i + s_j = 0` on the cells of `Ω` inside `I × J` forces
`r_i + s_j = 0` on all of `I × J`. Plans and `N(R)` are as in Theorem F.

1. Lower bound. If `Ω` carries a plan from `|a_i|²` to `|b_j|²`, every `X` equal to `a bᵀ` on
   `Ω` has `‖X‖_S1 ≥ ‖a‖²`.
2. Uniqueness. If also `Ω` is rigid on the supports, M recovers `a bᵀ` from `Ω`. Without the
   scaling, a plan from `‖b‖² |a_i|²` to `‖a‖² |b_j|²` serves.
3. Converse. M recovers `a bᵀ` from `Ω` only if `Ω` is rigid on the supports and every set
   of rows `R` has `‖b‖² ∑_R |a_i|² ≤ ‖a‖² ∑_{N(R)} |b_j|²`.

By the supply–demand theorem the condition of item 3 gives the plan of item 2. So M recovers
`a bᵀ` from `Ω` exactly when `Ω` is rigid on the supports and every set of rows `R` has
`‖b‖² ∑_R |a_i|² ≤ ‖a‖² ∑_{N(R)} |b_j|²`. A zero row or column of `a bᵀ` needs no cell: its
mass is `0`. From its entry `1` alone, M completes the row `(1, ?)` to `(1, 0)`, but not
`(1, ?)` to `(1, 1)`. For `I` and `J` all of the rows and columns, rigidity on the supports is
rigidity, and Theorem G is Theorem F.

### Proof of Theorem G

1. Item 1. As in Theorem F, with `x_i = u_i / a_i` and `y_j = v_j / conj(b_j)`, set to `0`
   where `a_i = 0` or `b_j = 0`. A plan sends nothing from a row or into a column of mass `0`,
   so the expansion reads `∑_ij T_ij ‖x_i − y_j‖² + ∑_{a_i = 0} ‖u_i‖² + ∑_{b_j = 0} ‖v_j‖²
   = ‖U‖² + ‖V‖² − 2 ‖a‖² = 2 (‖X‖_S1 − ‖a‖²)`. The three sums are `≥ 0`.
2. Item 2. At a minimum the three sums vanish.
   - `u_i = 0` where `a_i = 0` and `v_j = 0` where `b_j = 0`, so `X` vanishes on these rows
     and columns, as `a bᵀ` does.
   - Every row of `I` and column of `J` carries a positive mass, so on `I × J` the argument of
     Theorem F runs unchanged: `x_i` and `y_j` are unit vectors, `x_i = y_j` on the cells of
     `Ω` inside `I × J`, and rigidity on the supports makes them all equal.
   - Without the scaling: if `a = 0` or `b = 0`, the kernel is `0 0ᵀ`, and the plan `0`
     recovers it; otherwise scale as in Theorem F.
3. Item 3.
   - If `Ω` is not rigid on the supports, take the phases of Theorem F's proof from the cells
     of `Ω` inside `I × J`. On the other cells of `Ω` both `(u a)(v b)ᵀ` and `a bᵀ` vanish, so
     they agree on `Ω`; they differ at a cell `(i₀, j₀)` of `I × J`.
   - If the condition fails at `R`, Theorem F's witness applies as it stands: it uses neither
     the shape of the table nor nonzero entries.

### Theorem G for moduli one (proven)

Let `|a_i| = 1` on the support `I` of `a` and `|b_j| = 1` on the support `J` of `b`, with
`#I = #J`. M recovers `a bᵀ` from `Ω` exactly when the cells of `Ω` inside `I × J` are rigid on
`I × J` and contain the graph of a bijection `I → J`. Rows outside `I` and columns outside `J`
need no cell. For `I` and `J` all of a square table this is Theorem E.

A multiplicative character `χ` of `ℤ/p`, with `χ(0) = 0`, has `|χ(x)| = 1` for `x ≠ 0`. Every
character qualifies, the principal one included.

- `χ(ab) = χ(a) χ(b)`, with `I = J = {x ≠ 0}`. M recovers `[χ(ab)]` from `Ω` exactly when the
  cells of `Ω` with `a, b ≠ 0` are rigid on `{a ≠ 0} × {b ≠ 0}` and contain the graph of a
  permutation of the nonzero residues. For `p = 3` the three cells `(1, 1)`, `(1, 2)`, `(2, 2)`
  of the nine suffice; `(1, 1)` and `(2, 1)` do not, as both nonzero rows meet only column 1.
- `χ(a + ab) = χ(a) χ(1 + b)`, with `I = {a ≠ 0}` and `J = {b ≠ −1}`. M recovers
  `[χ(a + ab)]` from `Ω` exactly when the cells of `Ω` inside `I × J` are rigid on `I × J` and
  contain the graph of a bijection `I → J`. Row `0` and column `−1` need no cell.

Proof. Here `∑_R |a_i|² = #(R ∩ I)` and `‖a‖² = ‖b‖² = #I`, so the condition of Theorem G
reads `#(R ∩ I) ≤ #(N(R) ∩ J)` for every set of rows `R`. If `#I = 0`, both sides hold. Else,
for `R ⊆ I` this is Hall's condition for the cells of `Ω` inside `I × J`, which gives an
injection `I → J` inside `Ω`, a bijection as `#I = #J`. Conversely a bijection `σ` inside `Ω`
maps `R ∩ I` injectively into `N(R) ∩ J`. Rigidity on the supports is rigidity of the pattern
cut down to `I × J`: extend `r` and `s` by `0` off `I` and `J`.

### Rigidity is connectivity (proven)

Let the table have at least one row and one column. `Ω` is rigid exactly when the bipartite
graph on the rows and the columns, with an edge from row `i` to column `j` for each cell
`(i, j) ∈ Ω`, is connected. A rigid pattern then has at least `#rows + #columns − 1` cells,
`2N − 1` for `N × N`. So every rigidity condition above reads as connectivity: M recovers
`a bᵀ` with `|a_i| = |b_j| = 1` exactly when the cells of `Ω` connect the rows with the columns
and contain the graph of a permutation, and `[χ(ab)]` exactly when the cells with `a, b ≠ 0`
connect the nonzero rows and columns and contain the graph of a permutation of them.

Proof. If the graph is connected and `r_i + s_j = 0` on `Ω`, the values `r_i` at the rows and
`−s_j` at the columns agree across every edge, hence along every path, so `r_i = −s_j` for all
`i`, `j`. If `Ω` is rigid, fix a row `i₀` and set `r_i = 1`, `s_j = −1` on the rows and columns
reachable from `i₀`, and `0` elsewhere. An edge joins two reachable vertices or two unreachable
ones, so `r_i + s_j = 0` on `Ω`; rigidity gives it everywhere. Then `r_{i₀} + s_j = 0` makes
every column reachable, and `r_i + s_{j₀} = 0` every row. A connected graph on `V` vertices has
at least `V − 1` edges. The source's statement needs the row and the column: without rows every
pattern is rigid, and the graph on the columns alone has no edges.

### The fewest cells (proven)

Every `N × N` matrix `u vᵀ` with `|u_i v_j| = 1` is recovered by M from the `2N − 1` cells
`(i, i)` and `(j, j + 1)`, `j < N − 1`: the diagonals of offsets `0` and `1` without the cell
`(N − 1, 0)`. No set of fewer cells recovers even one of these matrices. So `2N − 1` is the
least number of cells from which M recovers all of them; at `N = 97` that is 193 of 9409.

For `[χ(ab)]`, `χ` a multiplicative character, the zero row and column need no cell, and the same
count runs over the `p − 1` nonzero rows and columns. The `2p − 3` cells `(a, a)` and `(1, b)`,
`a, b ≠ 0`, recover it: every nonzero column meets row `1`, and every nonzero row meets its own
column. No set of fewer cells recovers it, as its cells with `a, b ≠ 0` connect `2p − 2`
vertices. At `p = 97` that is 191 of 9409.

Proof. The cells join row `0`, column `1`, row `1`, column `2`, …, row `N − 1`, and column `0`
to row `0`, so their graph is connected; they contain the diagonal, the graph of the identity.
Rigidity is connectivity, so the criterion above holds. Conversely, a recovering set is
connected, and a connected graph on `2N` vertices has at least `2N − 1` edges.

### The readout decides (proven)

The same task `a b` has two tables. Read through `ψ_r`, `r ≠ 0`, `[ψ_r(ab)]` has every singular
value `√p` and entries of modulus one, so on every proper `Ω` some matrix equal to it there has a
smaller nuclear norm: M never returns it from an incomplete set. Read through a multiplicative
character, `[χ(ab)] = χ χᵀ` has rank one and is recovered from the sets of the criterion above.
Some `χ` is injective, `χ(g^k) = e^{2πik/(p−1)}` for a generator `g` and `χ(0) = 0`, and through
an injective readout the table determines the task. So M recovers `a b` from those sets through
`χ`, and from no incomplete set through `ψ_r`.

### Zero-entry check (torch, 2026-10-28)

Theorem G on every pattern of an `m × n` table except the full one, batched ADMM on `K = a bᵀ`
for 30000 iterations with `ρ = 1/√max(m, n)`; gap and error as in the weighted check.
"Recovered" means an error below 1e-6, and "rigid" means rigid on the supports.

Generic weights: moduli uniform in `[0.3, 2]` with random phases, and zeros at fixed places:
`2 × 3` with `a_0 = 0` or `b_1 = 0`; `3 × 3` with `a_2 = 0`, `a_0 = b_0 = 0`, `b_1 = b_2 = 0` or
none; `3 × 2` with `a_1 = b_0 = 0`; `2 × 4` with `b_3 = 0`; `3 × 4` with `a_1 = b_2 = 0`.

| class | patterns | recovered | gap | error |
| --- | --- | --- | --- | --- |
| rigid, condition | 790 | 790 | within 3.6e-15 of 0 | at most 4.8e-16 |
| rigid, no condition | 1118 | 0 | −2.86 to −3.5e-4 | 0.017 to 0.90 |
| not rigid, no condition | 4675 | 0 | −5.11 to −3.5e-4 | 0.15 to 1.00 |

Tight weights, with random phases: `|a| = (1, 1, 0)` with `|b| = (1, 1, 0)` or `(1, 1, 1)`;
`|a| = (2, 1, 0)` with `|b| = (1, 1, √3, 0)`; `|a| = (1, √2, 0)` with `|b| = (1, 0, √2)`.

| class | patterns | recovered | gap | error |
| --- | --- | --- | --- | --- |
| rigid, condition | 1220 | 1220 | within 1.8e-15 of 0 | at most 7.1e-16 |
| not rigid, condition | 224 | 0 | within 1.8e-15 of 0 | 0.57 to 0.71 |
| rigid, no condition | 464 | 0 | −1.00 to −0.035 | 0.17 to 0.60 |
| not rigid, no condition | 3720 | 0 | −5.00 to −0.035 | 0.45 to 1.00 |

Every class behaves as Theorem G says. The rigid patterns that meet the condition are recovered.
A pattern that meets the condition without rigidity reaches `‖a‖ ‖b‖` at another matrix, and
without the condition some completion is shorter. As in the weighted check, no generic draw
meets the condition without rigidity.

### Character check (torch, 2026-10-28)

The training halves of `grok-simple.py` (seeds 7 and 11) for `χ(ab)` and `χ(a + ab)`, with `χ`
a faithful character, `χ(g^k) = e^{2πik/(p−1)}` for a primitive root `g`: 12 splits for
`p = 31, 61, 97`. Rigidity on the supports is checked as connectivity of the cells inside
`I × J`, the bijection by augmenting paths; ADMM as in the patterns check, 20000 iterations at
most; the error is `‖X − K‖ / ‖K‖` on the cells off `Ω`, both seeds. A decoded cell is the
residue whose `χ` is nearest to the entry of `X`.

| p | kernel | rigid | bijection | relative error off `Ω` | on the zero row and column | decoded | `‖X‖_S1` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | `χ(ab)` | yes | yes | 3.0e-11, 6.9e-11 | 0 | 100 % | 30 |
| 31 | `χ(a + ab)` | yes | yes | 7.1e-11, 7.5e-11 | below 5e-15 | 100 % | 30 |
| 61 | `χ(ab)` | yes | yes | 2.0e-11, 3.2e-11 | 0 | 100 % | 60 |
| 61 | `χ(a + ab)` | yes | yes | 1.5e-11, 2.7e-11 | below 5e-15 | 100 % | 60 |
| 97 | `χ(ab)` | yes | yes | 1.5e-11, 1.9e-11 | 0 | 100 % | 96 |
| 97 | `χ(a + ab)` | yes | yes | 1.6e-11, 1.9e-11 | below 5e-15 | 100 % | 96 |

Every split meets both conditions, so by the corollary M recovers the kernel. ADMM agrees:
`‖X‖_S1 = p − 1 = ‖K‖_S1` to three decimals, and its entries on the zero row and column are 0
for `χ(ab)` and below 5e-15 for `χ(a + ab)`: rounding errors, which change with the number of
threads from run to run.

### Lean status

All declarations are in `Transformer.Grokking.FlatSpectrum` and are proved without `sorry`.

| Claim | Lean | Differences |
| --- | --- | --- |
| Theorem D.1 | `onesMatrix`, `meanProj`, `IsContraction.proj_add`, `le_nuclearNorm_onesMatrix_add` | A contraction is `IsContraction W`: `1 − Wᴴ W` is positive semidefinite. |
| Theorem D.2 | `eq_zero_of_nuclearNorm_eq_zero`, `nuclearNorm_onesMatrix`, `isMinOn_onesMatrix_iff` | Stated as `IsMinOn nuclearNorm S X ↔ X = J` for every `X` equal to `J` on `Ω`. |
| Theorem D.3 | `isContraction_of_forall_sum_norm_sq_le`, `sum_norm_sq_add_const`, `norm_sq_sub_le_cycle`, `sum_norm_sq_le_cycle`, `permAvg`, `add_eq_zero_of_cycle`, `sum_norm_sq_mulVec_permAvg_sub_le`, `isContraction_permAvg`, `isMinOn_onesMatrix_iff_of_cycle` | The cycle is a bijection `e : ℤ/N ≃ n` with `τ(σ⁻¹(e l)) = e(l + 1)`. |
| Theorem D.4 | `nuclearNorm_diagonal_mul_mul_diagonal`, `diagonal_mul_onesMatrix_mul_diagonal`, `nuclearNorm_vecMulVec_of_norm_eq_one`, `isMinOn_vecMulVec_iff_of_onesMatrix`, `isMinOn_vecMulVec_iff_of_cycle` | |
| Theorem D.5 | `isMinOn_addKernel_iff_of_diagonals`, `isMinOn_addKernel_iff_of_ne` | `Ω` may hold more cells than the two diagonals. |
| Theorem D.6 | `graphMatrix`, `conjTranspose_graphMatrix_mul`, `isMinOn_vecMulVec_and_graphMatrix`, `exists_ne_isMinOn_addKernel_of_diagonal` | `Ω` is exactly one diagonal. |
| Theorem D.7 | `exists_vecMulVec_ne_of_not_rigid`, `not_forall_isMinOn_iff_of_not_rigid` | The failure of rigidity is given as explicit `r`, `s`, `i₀`, `j₀`. "M recovers no such `Ω`" is the negation of D.2's conclusion. |
| Theorem D.8 | `forall_isMinOn_addKernel_iff_isUnit` | `Ω` is exactly the two diagonals. |
| Theorem D.9 | `rigid_cross`, `vecMulVec_eq_of_cross`, `crossCert`, `card_compl_singleton_cast`, `isContraction_crossCert`, `two_mul_sqrt_le_nuclearNorm_of_cross`, `crossWitness`, `crossFactor`, `crossWitness_eq_mul`, `frobeniusSq_crossFactor`, `two_mul_sqrt_lt`, `isLeast_nuclearNorm_vecMulVec_cross`, `vecMulVec_cross_determined_not_isMinOn`, `isLeast_nuclearNorm_addKernel_cross` | The least norm is `IsLeast` of the set of nuclear norms. Rank-one determination needs only `a_{i₀} b_{j₀} ≠ 0`. |
| Theorem D.10 | `IsContraction.sum_norm_sq_mulVec_le`, `centeredIndicator`, `sum_centeredIndicator`, `sum_sq_centeredIndicator`, `sum_centeredIndicator_mul_mulVec`, `sq_mul_le_of_block`, `not_isContraction_of_block`, `not_isContraction_of_row` | `c` is any real number. The single-cell row is `∀ j ≠ j₀, (i₀, j) ∉ Ω`, with `N ≥ 2`. |
| Theorem D.11 | `sum_norm_sq_mulVec_of_conjTranspose_mul`, `cycleAvg_mul_fourierMatrix`, `norm_sq_one_add_stdAddChar_div_two`, `cos_sq_mul_div_le`, `cos_pi_div_nonneg`, `norm_sq_cycleEigen_le`, `sum_norm_sq_mulVec_cycleAvg_le`, `cycleAvg_mulVec_stdAddChar`, `sum_norm_sq_mulVec_cycleAvg_stdAddChar`, `permAvg_sub_meanProj_mulVec_symm_apply`, `sum_norm_sq_mulVec_permAvg_sub_le_cos`, `sum_norm_sq_mulVec_permAvg_sub_stdAddChar`, `l2_opNorm_permAvg_sub_meanProj`, `isContraction_smul_permAvg_sub_iff`, `le_nuclearNorm_onesMatrix_add_of_cycle`, `l2_opNorm_permAvg_addRight_sub` | Added; the Numerical check states the value for the diagonals `{0, 1}` of `ℤ/p`, generalized to every pair of item 3. `‖·‖_op` is Mathlib's L2 operator norm on matrices (`Matrix.Norms.L2Operator`), and `c` is any complex number. The cycle is D.3's bijection `e`. Item 1 with `c = 1/cos(π/N)` needs `N ≥ 3`; for `N = 2`, `cos(π/2) = 0` and every `c` makes `c (Y − Q)` a contraction. The diagonals are those of item 5, for `N ≥ 2` and `s₂ − s₁` a unit of `ℤ/N`. |
| Theorem E.1 | `sum_norm_sq_sub_eq`, `sum_sum_norm_sq_sub_perm`, `card_le_nuclearNorm_of_perm` | The permutation is an `Equiv.Perm` with `(i, σ i) ∈ Ω`. |
| Theorem E.2 | `eq_onesMatrix_of_perm`, `isMinOn_onesMatrix_iff_of_perm`, `isMinOn_vecMulVec_iff_of_perm`, `isMinOn_addKernel_iff_of_perm`, `rigid_ne_one_zero` | The examples use the rigid pattern `{(0, 0), (0, 1), (1, 1)}`. |
| Theorem E.3 | `blockFactorL`, `blockFactorR`, `blockWitness_apply`, `sum_ite_mem_univ`, `frobeniusSq_blockFactorL`, `frobeniusSq_blockFactorR`, `nuclearNorm_blockWitness_le`, `exists_nuclearNorm_lt_of_block`, `exists_nuclearNorm_lt_of_not_perm`, `not_isMinOn_onesMatrix_of_not_perm` | Hall's theorem is Mathlib's `Finset.all_card_le_biUnion_card_iff_exists_injective`. `sum_ite_mem_univ` holds in any commutative ring. |
| The exact witness (*Patterns*) | `onesSubBlock`, `splitColumns`, `splitColumns_gram`, `gram_onesSubBlock`, `nuclearNorm_onesSubBlock`, `nuclearNorm_onesSubBlock_sq_of_nonneg`, `radicand_ge`, `radicand_best`, `radicand_ge_of_le`, `sq_sqrt_add_sqrt`, `sqrt_le_mul_of_le`, `sqrt_lt_mul_of_lt`, `mul_le_sqrt_of_le`, `isLeast_nuclearNorm_onesSubBlock`, `isLeast_nuclearNorm_onesSubBlock_of_le`, `nuclearNorm_onesSubBlock_best_sq`, `nuclearNorm_onesSubBlock_best`, `exists_nuclearNorm_onesSubBlock_lt_iff`, `exists_nuclearNorm_onesSubBlock_lt_card_iff`, `isLeast_nuclearNorm_onesSubBlock_sixtyOne`, `isLeast_nuclearNorm_onesSubBlock_cross` | Generalized to `m × n` and every real `t`: `‖J − t 1_R 1_Cᵀ‖²_S1 = m n − 2t k l + t² k l + 2|t| g` with `g = √(k l (m − k)(n − l))`, from `Xᴴ X = U M Uᴴ` with the orthogonal columns `U = [1_C, 1_{Cᶜ}]`. For nonempty `R`, `C` with `m n ≤ k n + l m`, the least norm over `t` is `√(k (n − l)) + √(l (m − k))` (`IsLeast`), reached at `t = 1 − g/(k l)`, where `‖X‖²_S1 = m n − (k l − g)²/(k l)`; when `k n + l m ≤ m n` it is `√(m n) = ‖J‖_S1`. So some `t` beats `J` iff `m n < k n + l m`, for `N × N` iff `N < k + l`. At `N = 61`, `k = 2`, `l = 60` the least is `√2 + √3540`, between 60.912 and 60.913; for `k = l = N − 1` and `N ≥ 2` it is `2√(N − 1)`. |
| The failure margin | `blockCert`, `blockCert_mulVec`, `card_compl_cast`, `blockCert_compl_singleton`, `isContraction_blockCert`, `inv_sqrt_sq_mul_le_one`, `trace_blockCert_mul`, `sqrt_add_sqrt_le_nuclearNorm_of_block`, `isLeast_nuclearNorm_compl_block`, `sqrt_mul_add_sqrt_mul_le`, `sqrt_add_sqrt_le_sqrt_sq_sub`, `sqrt_add_sqrt_le_sqrt_sq_sub_one`, `sqrt_sq_sub_one_lt`, `exists_block_of_not_perm`, `exists_nuclearNorm_eq_of_block`, `exists_nuclearNorm_eq_of_not_perm`, `exists_nuclearNorm_le_sqrt_of_not_perm`, `not_exists_perm_of_block`, `isLeast_nuclearNorm_compl_block_card`, `isLeast_nuclearNorm_compl_half_block`, `exists_compl_block_isLeast`, `failureMargin_odd`, `sqrt_two_add_sqrt_six_mem`, `failureMargin_four`, `sqrt_mul_sub_sub_le`, `sq_sqrt_add_sqrt_mul_sub_one`, `sqrt_add_sqrt_le_of_mul_le`, `mul_le_mul_of_add_eq`, `failureMargin`, `image_nuclearNorm_vecMulVec`, `image_nuclearNorm_vecMulVec_of_norm`, `margin_vecMulVec_of_onesMatrix`, `exists_nuclearNorm_le_sqrt_vecMulVec_of_not_perm`, `failureMargin_vecMulVec`, `failureMargin_odd_vecMulVec`, `failureMargin_four_vecMulVec` | Added. Item 1 is for `m × n`. Its lower bound needs `X = 1` only on `R × Cᶜ` and `Rᶜ × C`; the least norm off `R × C` is `IsLeast` of the nuclear norms of the matrices with `X i j = 1` whenever `¬(i ∈ R ∧ j ∈ C)`. Item 2's block has `#R + #C = N + 1`, and both `≤ √(N² − 1)` and `< N − 1/(2N)` are stated. In item 3 the pattern is `{ij \| ¬(ij.1 ∈ R ∧ ij.2 ∈ C)}`, and "no permutation graph" is `¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω`. For all `k, l ≤ N` with `N < k + l`, `exists_compl_block_isLeast` gives `R` and `C` of these sizes. `failureMargin` takes `N ≥ 1` and `a b : ℕ` with `a + b = N + 1`, `a ≤ b ≤ a + 1`, and states both halves: every pattern without a permutation graph has a completion of `J` of norm at most `√(a (a − 1)) + √(b (b − 1))`, and some such pattern has it as least norm; `failureMargin_odd` and `failureMargin_four` are the same for odd `N` and for `N = 4`, with the values `√(N² − 1)` and `√2 + √6`. Item 4 takes `∀ i j, ‖vecMulVec u v i j‖ = 1`. `image_nuclearNorm_vecMulVec` states that the nuclear norms of the matrices equal to `a bᵀ` on `Ω` and of those equal to `J` on `Ω` are the same set, for `|a i| = |b j| = 1`; `image_nuclearNorm_vecMulVec_of_norm` states it for `u vᵀ`, an empty `n` included; `margin_vecMulVec_of_onesMatrix` carries a bound `c` on a completion of `J` from every pattern without a permutation graph, together with a pattern without one whose least norm is `c`, to `u vᵀ`; `exists_nuclearNorm_le_sqrt_vecMulVec_of_not_perm` states `‖X‖_S1 ≤ √(N² − 1)` and `‖X‖_S1 < ‖u vᵀ‖_S1 − 1/(2N)`; `failureMargin_vecMulVec` is `failureMargin` for `u vᵀ`, with a nonempty `n` in place of `N ≥ 1`; `failureMargin_odd_vecMulVec` and `failureMargin_four_vecMulVec` are `failureMargin_odd` and `failureMargin_four` for `u vᵀ`; `‖u vᵀ‖_S1 = N` is `nuclearNorm_vecMulVec_of_norm_eq_one` with `exists_norm_eq_one_of_vecMulVec`. The expansions in `1/N` are not formalized. |
| Theorem E.4 | `forall_isMinOn_vecMulVec_iff`, `forall_isMinOn_onesMatrix_iff`, `forall_isMinOn_addKernel_iff` | "M recovers `K` from `Ω`" is D.2's form, `∀ X ∈ S, IsMinOn nuclearNorm S X ↔ X = K`. An example derives the failure of the cross for `N = 3`. |
| Theorem F.1 | `IsTransportPlan`, `IsTransportPlan.sum_eq`, `isTransportPlan_perm`, `sum_sum_mul_norm_sq_sub`, `rowDiv_mul_apply`, and G.1 | The plan has the marginals `‖a i‖²` and `‖b j‖²`, so it carries the scaling `‖a‖ = ‖b‖`. |
| Theorem F.2 | G.2 with `rigid_support_of_ne_zero` | Stated as `IsMinOn nuclearNorm S X ↔ X = a bᵀ` for every `X` equal to `a bᵀ` on `Ω`. |
| Theorem F.3 | `nbhd`, `mem_nbhd`, `nbhd_mono`, `nbhd_union`, `IsTransportPlan.sum_le_sum_nbhd`, `isTransportPlan_univ`, `nbhd_diag_zero`, and G.3 | The witness needs no scaling: `U` carries a factor `p` and `V` a factor `p⁻¹`, with `p² = ‖b‖/‖a‖`. A failed condition gives `¬ IsMinOn`, stronger than non-recovery. "Only if `Ω` carries a plan" is in `forall_isMinOn_vecMulVec_iff_exists_transport`, below. |
| Theorem F.3, supply–demand | `IsSubPlan`, `IsTransportPlan.isSubPlan`, `IsSubPlan.mono`, `IsSubPlan.ship`, `IsSubPlan.add_split`, `sum_le_sum_nbhd_ship`, `nbhd_diag`, `sum_le_sum_nbhd_diff_of_row`, `sum_le_sum_nbhd_diff_of_col`, `sum_le_sum_nbhd_inside`, `sum_le_sum_nbhd_outside`, `exists_ship`, `exists_isSubPlan`, `exists_isTransportPlan_iff` | For any nonnegative real masses on finite index types. The equivalence adds `∑ α = ∑ β`, which the scaling `‖a‖ = ‖b‖` gives in item 3. `exists_isSubPlan` is the form with capacities: the condition alone gives row sums `α` and column sums `≤ β`. The induction runs on `Ω.ncard`. |
| Theorem F.4 | `crossPlan`, `isTransportPlan_cross`, `forall_isMinOn_vecMulVec_cross_iff` | Stated as `A B ≤ ‖a i₀‖² B + A ‖b j₀‖²` with `A = ‖a‖²` and `B = ‖b‖²`. The proof scales `a` by `c` and `b` by `c⁻¹` with `c⁴ = B/A`. |
| Theorem F, items 2–3 | `forall_isMinOn_vecMulVec_iff_of_ne_zero`, `forall_isMinOn_vecMulVec_iff_exists_transport`, `nbhd_ne_one_zero` | Corollaries of Theorem G's criteria. Without the scaling: the plan goes from `‖b‖² |a_i|²` to `‖a‖² |b_j|²`, and "M recovers" is D.2's form. The examples: the triangle `{(0, 0), (0, 1), (1, 1)}` recovers `(2, 1)(1, 2)ᵀ`, through the plan `[[5, 15], [0, 5]]`, and not `(1, 2)(2, 1)ᵀ`. |
| Theorem G.1 | `IsTransportPlan.eq_zero_of_row`, `IsTransportPlan.eq_zero_of_col`, `sum_sum_mul_norm_sq_sub_of_transport`, `sum_norm_sq_le_nuclearNorm_of_transport`, `nuclearNorm_vecMulVec_le`, `nuclearNorm_vecMulVec_of_transport` | `a : m → ℂ` and `b : n → ℂ` on any finite types. `x_i = a_i⁻¹ u_i` uses Lean's `0⁻¹ = 0`. |
| Theorem G.2 | `eq_vecMulVec_of_transport`, `rigid_support_of_ne_zero`, `isMinOn_vecMulVec_iff_of_transport`, `isMinOn_vecMulVec_iff_of_transport_mul` | Rigidity on the supports is `∀ r s, (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) → ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0`. The examples: the triangle recovers `(2, 1)(1, 2)ᵀ` through the plan `[[1, 3], [0, 1]]`, and `{(0, 0)}` recovers `(1, 0)`. |
| Theorem G.3 | `star_dotProduct_self_eq_sum_norm_sq`, `nuclearNorm_vecMulVec`, `exists_vecMulVec_ne_of_not_rigid_of_ne_zero`, `not_forall_isMinOn_vecMulVec_of_not_rigid`, `weightedFactorL`, `weightedFactorR`, `weightedWitness_apply`, `sum_mul_ite_mem`, `frobeniusSq_weightedFactorL`, `frobeniusSq_weightedFactorR`, `nuclearNorm_weightedWitness_le`, `exists_weighted_bound_lt`, `exists_nuclearNorm_lt_of_weighted_block`, `exists_nuclearNorm_lt_of_nbhd`, `not_isMinOn_vecMulVec_of_nbhd` | The failure of rigidity on the supports is given as explicit `r`, `s`, `i₀`, `j₀` with `a i₀ b j₀ ≠ 0`; `r i + s j = 0` is required only on the cells of `Ω` with `a i b j ≠ 0`. A failed condition gives `¬ IsMinOn`. The examples: `{(0, 0), (1, 2)}` does not recover `(1, 1)(1, 1, 0)ᵀ`, and `(1)(1, 1)ᵀ` has a shorter completion from its cell `(0, 0)`. |
| Theorem G, items 2–3 | `forall_isMinOn_vecMulVec_iff_support`, `forall_isMinOn_vecMulVec_iff_exists_transport_support`, `rigid_support_iff_of_ne_zero` | For any `a : m → ℂ` and `b : n → ℂ`; "M recovers" is D.2's form. Theorem F's criteria follow by `rigid_support_iff_of_ne_zero`. The examples: in the `1 × 2` table the cell `(0, 0)` recovers `(1)(1, 0)ᵀ`, through the plan `[[1, 0]]`, and not `(1)(1, 1)ᵀ`. |
| Theorem G for moduli one | `rigid_support_iff_subtype`, `sum_norm_sq_eq_card`, `forall_sum_le_iff_exists_equiv`, `forall_isMinOn_vecMulVec_iff_equiv_support` | The supports are the subtypes `{i // a i ≠ 0}` and `{j // b j ≠ 0}`, with `Nat.card` equal; the bijection is an `Equiv` between them. Hall's theorem is Mathlib's. The example: in the `2 × 2` table the cell `(0, 0)` recovers `(1, 0)(1, 0)ᵀ`. |
| Rigidity is connectivity (Setting) | `patternGraph`, `patternGraph_adj_inl_inr`, `patternGraph_adj_inr_inl`, `not_patternGraph_adj_inl_inl`, `not_patternGraph_adj_inr_inr`, `rigid_of_preconnected`, `connected_of_rigid`, `rigid_iff_connected`, `rigid_iff_connected'`, `card_add_card_le_ncard_add_one_of_rigid` | The graph is Mathlib's `SimpleGraph.fromRel` on `m ⊕ n`, row `i` adjacent to column `j` when `(i, j) ∈ Ω`. Corrected: the equivalence needs a row and a column, `[Nonempty m] [Nonempty n]`, which the source's `N × N` tables with `N ≥ 1` have; without rows every pattern is rigid. The count is `#m + #n ≤ #Ω + 1`, from Mathlib's `Connected.card_vert_le_card_edgeSet_add_one`; for `N × N`, the source's `2N − 1`. The example is `{(0, 0), (0, 1), (1, 1)}`. |
| Theorems E and G with graphs | `forall_isMinOn_vecMulVec_iff_connected`, `exists_norm_eq_one_of_vecMulVec`, `forall_isMinOn_vecMulVec_iff_connected_of_entries`, `forall_isMinOn_addKernel_iff_connected`, `rigid_support_iff_connected`, `forall_isMinOn_taskKernel_mul_mulChar_iff_connected`, `forall_isMinOn_taskKernel_add_mul_mulChar_iff_connected` | Rigidity replaced by `(patternGraph Ω).Connected`; on the supports, by connectivity of the graph of the cells inside them, on the subtypes. Added: a rank-one `u vᵀ` with `|u_i v_j| = 1` rescales to `|a_i| = |b_j| = 1`, so the criterion holds for every rank-one matrix with entries of modulus one. The examples: `{(0, 0), (0, 1), (1, 1)}` recovers `(1, 1)(1, 1)ᵀ` and `(2, 2)(½, ½)ᵀ`, all of `ℤ/3 × ℤ/3` recovers `ψ(a + b)`, and for `p = 3` the cells with `a, b ≠ 0`, and those with `a ≠ 0`, `b ≠ −1`, recover `[χ(ab)]` and `[χ(a + ab)]`. |
| `χ(ab)`, `χ(a + ab)` | `mulChar_ne_zero_iff`, `forall_isMinOn_taskKernel_mul_mulChar_iff`, `taskKernel_add_mul_mulChar`, `forall_isMinOn_taskKernel_add_mul_mulChar_iff`, `nonzero_zmod_three` | For every `χ : MulChar (ZMod p) ℂ`, `p` prime. The nonzero residues are `{x : ZMod p // x ≠ 0}`, the columns of `χ(a + ab)` `{x // x ≠ −1}`. The examples, for `p = 3` and the principal character: `{(1, 1), (1, 2), (2, 2)}` recovers `[χ(ab)]` and `{(1, 1), (2, 1)}` does not; the cells off row `0` and column `−1` recover `[χ(a + ab)]`. |
| The fewest cells | `staircase`, `ncard_staircase`, `staircase_connected`, `forall_isMinOn_vecMulVec_staircase`, `two_mul_card_le_ncard_add_one_of_forall_isMinOn`, `isLeast_ncard_forall_isMinOn_vecMulVec` | Added. On `Fin N`, `N = k + 1`, the staircase `{(i, i)} ∪ {(j, j + 1) : j < N − 1}` is the diagonals `{0, 1}` without the cell `(N − 1, 0)`, `2N − 1` cells. It is connected and contains the diagonal, so M recovers from it every `N × N` matrix `u vᵀ` with `|u_i v_j| = 1`. A set from which M recovers one such matrix has at least `2N − 1` cells. So `2N − 1` is the least size of a set from which M recovers all of them (`IsLeast`). |
| Two diagonals minus a cell | `twoDiagonals`, `twoDiagonals_comm`, `mk_add_ne_of_ne`, `connected_of_diagonals_diff`, `connected_and_perm_of_diagonals_diff`, `forall_isMinOn_vecMulVec_of_diagonals_diff`, `forall_isMinOn_addKernel_of_diagonals_diff`, `forall_isMinOn_addKernel_of_diagonals_diff_of_ne`, `ncard_twoDiagonals`, `ncard_twoDiagonals_diff` | Added; the Numerical check removes the cell `(0, 1)` from the diagonals `{0, 1}`. For `N ≥ 2` and offsets `s₁`, `s₂` of `ℤ/N` with `s₂ − s₁` a unit, every `Ω` that holds the diagonals `b = a + s₁` and `b = a + s₂` but at most one cell connects the rows with the columns and contains the graph of a permutation: one diagonal survives whole, and from the row of the removed cell the steps `a ↦ a + (s₂ − s₁)` reach every row. So M recovers from `Ω` every `N × N` matrix `u vᵀ` with `|u_i v_j| = 1`, `ψ(a + b)` among them; for prime `p`, from any two distinct diagonals. The two diagonals have `2N` cells, `2N − 1` without one of them. `Ω` may hold more cells. The examples: `{0, 1}` minus `(2, 0)` and minus `(1, 1)` in `ℤ/3`, the all-ones `2 × 2` matrix from `{0, 1}` minus `(0, 0)`, `ψ(a + b)` from `{0, 1}` minus `(0, 1)` for `p = 4` and from `{0, 2}` minus `(1, 1)` for `p = 3`. |
| The table of `a b` from `2p − 3` cells | `diagRow`, `diagRow_connected`, `ncard_diagRow`, `nonzeroCell`, `nonzeroCell_injective`, `productCells`, `mem_productCells`, `ncard_productCells`, `forall_isMinOn_taskKernel_mul_mulChar_productCells`, `two_mul_sub_three_le_ncard_of_forall_isMinOn`, `isLeast_ncard_forall_isMinOn_taskKernel_mul_mulChar`, `exists_injective_mulChar_isLeast` | Added. The diagonal and a row, `(i, i)` and `(i₀, j)`, are `2N − 1` connected cells. For prime `p` and every `χ : MulChar (ZMod p) ℂ`, the cells `(a, a)` and `(1, b)` with `a, b ≠ 0`, `2p − 3` of them, recover `[χ(ab)]`, and a set from which M recovers `[χ(ab)]` has at least `2p − 3` cells: its cells with `a, b ≠ 0` connect the `p − 1` nonzero rows with the `p − 1` nonzero columns. So `2p − 3` is the least size (`IsLeast`), for an injective `χ` too. |
| The readout decides | `eq_of_taskKernel_eq`, `not_isMinOn_taskKernel_mul`, `exists_injective_mulChar` | Added. For prime `p`, `r ≠ 0` and every proper `Ω`, `[ψ_r(ab)]` does not minimize `‖·‖_S1` among the matrices equal to it on `Ω` (its singular values all equal `√p`, its entries have modulus one). Some `χ : MulChar (ZMod p) ℂ` is injective: `χ(g^k) = e^{2πik/(p−1)}` for a generator `g` of `(ZMod p)ˣ`, `χ(0) = 0`. Through an injective readout `φ`, `[φ(f(a, b))] = [φ(g(a, b))]` forces `f = g`. |

Not formalized:
- the numerical tables, including the certificates found for random patterns and for the
  training splits;
- on the diagonals `{0, 1}` minus a cell, that the only `Y` with unit row and column sums is a
  permutation matrix, with `‖Y − Q‖_op = 1`. That no `Y` has `c > 1` is item D.10 for the row
  of the removed cell, which keeps a single cell (`not_isContraction_of_row`), and recovery is
  proved (row *Two diagonals minus a cell*);
- the failures on `2p` random cells;
- the weighted and cross tables;
- that training of the real network behaves like model M (Lemma 1's modeling step; for the
  pure factorization see *Training reaches the optimal fit*).

Scope:
- The spectrum alone does not decide recovery; the pattern does. `ψ(a + b)` has the best
  spectrum, rank one. It is recovered from two diagonals, `2p` cells, and from the 50% training
  splits, which are rigid and contain a permutation graph (checked numerically), so Theorem E
  applies. It is not recovered from one diagonal, from two diagonals whose offset
  is not a unit, from row 0 and column 0, or from `2p` random cells.
- Rigidity is necessary (item 7): otherwise a phase on one component changes `K` off `Ω`. A
  rigid pattern is connected, so it has at least `2p − 1` cells, and `2p − 1` cells suffice
  (*The fewest cells*), two diagonals minus any one cell among them (*Two diagonals minus a
  cell*); two diagonals use one more.
- Rigidity is not sufficient. The cross is rigid and determines `ψ(a + b)` among rank-one
  matrices, yet M misses it (item 9). The cross holds no permutation graph once `N ≥ 3`.
- A rigid pattern that contains a permutation graph is recovered (Theorem E.2). Theorem D's
  certificate is not needed:
  - every elementary pattern in the numerics had a certificate below 1;
  - a single-cell row or column pins every certificate at `‖Y − Q‖_op ≥ 1` (item D.10), yet
    such patterns are recovered too.
- A pattern with no permutation graph inside is not recovered, like the cross (Theorem E.3).
  So M recovers `ψ(a + b)` from `Ω` exactly when `Ω` is rigid and contains a permutation
  graph (Theorem E.4). Every pattern in the numerics agrees.
- The 50% training splits are deep in the elementary regime, with `‖Y − Q‖_op ≈ 0.2`.
- `ψ(a³ + a b)` is recovered from no proper `Ω` (Theorem A). Numerically it fails on every
  pattern above.

---

## The discontinuity of `ab` (added)

Source: the course's home page (`~/rust_projects/grokking`, `#/en/home/full`). It reads the
table of a task at `x = a/p`, `y = b/p` as `F_p(x, y) = f(a, b)/p`. For `f = a b mod p` this
is `frac(p x y)`, where `frac t = t − ⌊t⌋`.

### Statement (proven, except item 5)

1. If `2α` is an integer, `frac(pα) = frac α` for every odd `p`, so `frac(pα) → frac α` as
   `p → ∞` through the primes.
2. (Dirichlet) Let `α` be rational with denominator `q` in lowest terms. For every `k < q`
   prime to `q`, infinitely many primes `p` have `frac(pα) = k/q`.
3. For rational `α`, `frac(pα)` converges as `p → ∞` through the primes exactly when `2α` is
   an integer.
4. For `x, y ∈ [0, 1)`, `2xy` is an integer exactly when `x = 0`, `y = 0` or `xy = ½`. In the
   plane, the points where `2xy` is an integer form a set of area zero.
5. (Vinogradov) For irrational `α`, the values `frac(pα)` over the primes are equidistributed
   in `[0, 1)`; in particular they have no limit.

So on `[0, 1)²`, `frac(pxy)` has a limit through the primes exactly on the axes and the
hyperbola `xy = ½`, a null set. At the points with rational `xy` this is items 3 and 4; at
those with irrational `xy`, item 5.

### Proof

1. Write `p = 2j + 1`. Then `pα = j(2α) + α`, and `j(2α)` is an integer.
2. Let `α = n/q`. Then `n` is a unit mod `q`. By Dirichlet's theorem infinitely many primes
   have `p ≡ k n⁻¹ (mod q)`. Each has `pn = k + qt` for an integer `t`, so `pα = k/q + t` and
   `frac(pα) = k/q`.
3. If `2α` is an integer, item 1 applies. Otherwise `q ≥ 3`, and `1` and `q − 1` are prime to
   `q`. By item 2, `1/q` and `(q − 1)/q` both occur infinitely often, and they differ, so there
   is no limit.
4. For `x, y ∈ [0, 1)`, `2xy ∈ [0, 2)`, so the integer is 0 or 1. For the area: the set is
   closed, and its section at each `x ≠ 0` is the countable set `{m/(2x) : m ∈ ℤ}`.

### Lean status

All declarations are in `Transformer.Grokking.FlatSpectrum` and are proved without `sorry`.

| Claim | Lean | Differences |
| --- | --- | --- |
| Item 1 | `fract_mul_of_odd`, `tendsto_fract_mul_of_two_mul_eq` | "Through the primes" is the filter `atTop ⊓ 𝓟 {p \| p.Prime}` on `ℕ`. `α` is any real number with `2α = m` for an integer `m`. |
| Item 2 | `infinite_setOf_prime_fract_mul_eq` | `α : ℚ`, whose `den` is its denominator in lowest terms. Dirichlet's theorem is Mathlib's `Nat.forall_exists_prime_gt_and_eq_mod`. |
| Item 3 | `exists_tendsto_fract_mul_iff` | Stated as `(∃ L, Tendsto …) ↔ ∃ m : ℤ, 2α = m` for `α : ℚ`. |
| Item 4 | `two_mul_mul_mem_range_intCast_iff`, `volume_setOf_two_mul_mul_mem_range_intCast` | The area is the Lebesgue measure on `ℝ × ℝ` of `{z \| 2 z.1 z.2 ∈ range Int.cast}`. |

Not formalized: item 5. Mathlib has no equidistribution theorem over the primes.

---

## Rank above one (added)

Source: the course's plan (`~/rust_projects/grokking`, `todo.md`, *Open*): when does M recover
a table of rank `r`? Theorem G is `r = 1`. Added on 2026-10-29 and proved in Lean. Item 1
carries Theorem D's certificate to a table of any rank. Item 2 is a necessary condition for any
table. Item 3 settles the smallest case past rank one, a `2 × 2` table seen on three cells. Its
examples show that from rank two on, no criterion in `Ω`, the support of `K`, the moduli of its
entries and its rank decides recovery. Item 4 shows that item 1's condition on `U Aᴴ + B Vᴴ` is
not necessary. Item 5 decides when `K` is a minimum, at every rank: Theorem A without its
invertibility. Item 6 shows that item 1.2's hypotheses hold exactly when the nuclear norm grows
linearly away from `K`, which at full rank fails as soon as one cell is missing.

### Setting

- `K` is an `m × n` complex matrix and `Ω` a set of cells. M recovers `K` from `Ω` when `K` is
  the only matrix of least nuclear norm among those equal to `K` on `Ω`, as in *Exact recovery
  of a rank-one kernel*.
- `U` is `m × r` and `V` is `n × r`, with orthonormal columns: `Uᴴ U = Vᴴ V = 1`. Write
  `Π = 1 − U Uᴴ` and `Π' = 1 − V Vᴴ`.
- The graph of a set of cells is the bipartite graph on the rows and the columns with an edge
  from row `i` to column `j` for each of its cells `(i, j)`, as in *Rigidity is connectivity*.

### Statement (proven)

1. Certificate. Let `U Uᴴ K = K` and `tr((U Vᴴ)ᴴ K) = ‖K‖_S1`. A compact singular value
   decomposition `K = U diag(s) Vᴴ` with `s ≥ 0` gives both, and `‖K‖_S1 = ∑_k s_k`.
   1. Bound. Let `Y = U Vᴴ + W` vanish off `Ω`, with `Uᴴ W = 0`, `W V = 0` and `c W` a
      contraction for a real `c > 0`. Then every `H` vanishing on `Ω` has
      `‖K‖_S1 + (1 − 1/c) ‖Π H Π'‖_S1 ≤ ‖K + H‖_S1`.
   2. Uniqueness. Let also `c > 1`, and let no nonzero `U Aᴴ + B Vᴴ`, with `A` an `n × r` and
      `B` an `m × r` matrix, vanish on `Ω`. Then M recovers `K` from `Ω`.

   For `K = J`, `U = V = 𝟙/√N` (one column) and `W = Y − Q` these are Theorem D's items 1 and 2:
   `Uᴴ W = 0` and `W V = 0` say that `Y` has unit column and row sums, and `U Aᴴ + B Vᴴ` is the
   general matrix `r_i + s_j`.
2. Necessary condition. Let `r_i + s_j = 0` on the cells of `Ω` where `K` is nonzero, and
   `r_{i₀} + s_{j₀} ≠ 0` at a nonzero entry `K_{i₀ j₀}`, for complex `r` and `s`. Then M does not
   recover `K` from `Ω`. In graph form: if M recovers `K` from `Ω`, the graph of the cells of
   `Ω` where `K` is nonzero joins row `i₀` to column `j₀` for every nonzero entry `K_{i₀ j₀}`.
   For `K = a bᵀ` these cells are those of `Ω` inside the supports `I × J`, and the graph form
   says what the rigidity half of Theorem G's item 3 says (*Rigidity is connectivity*).
3. Three cells. Let `K` be `2 × 2` and `Ω` the three cells other than `(0, 1)`. Write
   `b = k₁₀`, `c = k₀₀ k₁₁` and `x* = conj(b) c / max(|b|², |c|)`, with `x* = 0` when
   `b = c = 0`.
   1. Every `X` equal to `K` on `Ω`, with `x = X₀₁`, has
      `‖X‖²_S1 = |k₀₀|² + |k₁₀|² + |k₁₁|² + |x|² + 2 |c − x b|`.
   2. Every `x ∈ ℂ` has `|x*|² + 2 |c − x* b| + |x − x*|² ≤ |x|² + 2 |c − x b|`.
   3. Among the matrices equal to `K` on `Ω`, the nuclear norm is least exactly at the one with
      `x*` at `(0, 1)`. So M has exactly one minimum, and it recovers `K` from `Ω` exactly when
      `k₀₁ = x*`.
   4. Rank at most one, `det K = 0`: M recovers `K` exactly when `|k₀₁| ≤ |k₁₀|`. For
      `|k₀₁| > |k₁₀|`, `x* = |k₁₀| k₀₁ / |k₀₁|`.
   5. Rank two, `det K ≠ 0`: M recovers `K` exactly when `k₀₁ = conj(k₁₀) det K / |det K|`.
   6. Moduli one, `|k_ij| = 1` for all `i`, `j`: `x* = k₀₀ k₁₁ / k₁₀`, the rank-one completion,
      and M recovers `K` exactly when `det K = 0`, so at rank two never.

   Examples. `[[2, 1], [1, 1]]`, of determinant `1`, is recovered. `[[2, 1], [1, −1]]`, of
   determinant `−3`, has the same support, moduli and rank and is not: M's unique minimum is
   `[[2, −1], [1, −1]]`. At rank one `[[2, 1], [2, 1]]` is recovered and `[[1, 2], [1, 2]]` is
   not. The Hadamard matrix `[[1, 1], [1, −1]]` is not.
4. Item 1.2's condition is sufficient, not necessary. In item 3, let `det K ≠ 0`. Every `U`
   with `U Uᴴ K = K`, as item 1 asks, has `U Uᴴ = 1`. So the matrix unit `E` at `(0, 1)` equals
   `U (Uᴴ E)`, a nonzero `U Aᴴ + B Vᴴ` vanishing on the three cells, for every `V`. Yet M
   recovers `[[2, 1], [1, 1]]` from them, and every table of item 3.5.
5. Minimum.
   1. Dual certificate. A matrix `K` is a minimum of the nuclear norm among the matrices equal
      to it on `Ω` exactly when some contraction `Y` vanishing off `Ω` has
      `tr(Yᴴ K) = ‖K‖_S1`.
   2. Let `K = U diag(s) Vᴴ` with `s > 0`. The contractions `Y` with `tr(Yᴴ K) = ‖K‖_S1` are
      exactly the matrices `U Vᴴ + W` with `Uᴴ W = 0`, `W V = 0` and `W` a contraction.
   3. Theorem A at every rank. With `K` as in item 5.2, `K` is a minimum exactly when some
      contraction `W` with `Uᴴ W = 0` and `W V = 0` makes `U Vᴴ + W` vanish off `Ω`: item 1.1's
      condition with `c = 1`.
   4. Full column rank. If also `V` is square, `K` is a minimum exactly when `U Vᴴ` vanishes off
      `Ω`. For an invertible `K`, `U Vᴴ` is the polar factor `K (Kᴴ K)^{−1/2}`, and this is
      Theorem A (that identification is a remark, not in Lean).

   Item 5 decides whether `K` is a minimum, not whether it is the only one. Lemmas of the proof:
   the nuclear norm is subadditive, and a matrix `Y` is a contraction exactly when
   `Re tr(Yᴴ X) ≤ ‖X‖_S1` for every `X`, and then so is `Yᴴ`.
6. Linear growth. Let `K = U diag(s) Vᴴ` with `s > 0`. Linear growth means that some real
   `c > 0` has `‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`.
   1. Linear growth holds exactly when item 1.2's hypotheses do: some `W` with `Uᴴ W = 0`,
      `W V = 0` and `c' W` a contraction for a real `c' > 1` makes `U Vᴴ + W` vanish off `Ω`,
      and no nonzero `U Aᴴ + B Vᴴ` vanishes on `Ω`.
   2. Linear growth makes `K` the only minimum. So the tables that M recovers and item 1.2
      misses are exactly the tables that M recovers without linear growth.
   3. Full rank. If `V` or `U` is square and some cell is not in `Ω`, linear growth fails. So
      with such `U` and `V`, item 1.2's hypotheses fail on every `Ω` that misses a cell, while
      item 5.4 still decides whether `K` is a minimum.
   4. Example. In item 3, `K = [[2, 1], [1, 1]]` has `‖K + h E₀₁‖_S1 = √(9 + h²)` for real
      `h ≤ 1`, with `E₀₁` the matrix unit at `(0, 1)`. M recovers `K` from the three cells
      (item 3), and linear growth fails: along `E₀₁` the nuclear norm grows quadratically.
   5. Dimension count. If no nonzero `U Aᴴ + B Vᴴ` vanishes on `Ω`, then
      `|Ω| ≥ r (m + n − r)`. So item 1.2 and linear growth need at least `r (m + n − r)` cells;
      at `r = 1` this is `m + n − 1`, the count of a rigid pattern (*Rigidity is
      connectivity*). Recovery needs no such count: M recovers `[[2, 1], [1, 1]]`, of rank two,
      from 3 cells, fewer than `2 (2 + 2 − 2) = 4` (item 6.4).

   Lemmas of the proof: no entry exceeds the nuclear norm; products
   of contractions, `Π` and `Π'` are contractions, and multiplying by contractions does not
   increase the nuclear norm; every real-linear map between matrix spaces is bounded in the
   nuclear norm; and, for every `H`, some `C` has
   `‖K + t H‖_S1 ≤ ‖K‖_S1 + t Re tr((U Vᴴ)ᴴ H) + |t| ‖Π H Π'‖_S1 + C t²` for every real `t`.

### Proof

1. Item 1.1. Take a contraction `Z₀` with `tr(Z₀ᴴ Π H Π') = ‖Π H Π'‖_S1`, and let
   `Z = U Vᴴ + Π Z₀ Π'`.
   - `Z` is a contraction: `1 − Zᴴ Z = Π' ((1 − Z₀ᴴ Z₀) + (Uᴴ Z₀)ᴴ (Uᴴ Z₀)) Π'` is positive
     semidefinite.
   - `Π K = 0`, so the pairing `tr(Zᴴ (K + H))` is `‖K‖_S1 + tr((U Vᴴ)ᴴ H) + ‖Π H Π'‖_S1`.
   - `H` vanishes on `Ω` and `Y` off it, so `tr(Yᴴ H) = 0`. With `W = Π W Π'` this gives
     `tr((U Vᴴ)ᴴ H) = −tr(Wᴴ Π H Π')`, of modulus at most `‖Π H Π'‖_S1 / c`.
   - The pairing has modulus at most `‖K + H‖_S1`. Its real part gives the bound.

   The decomposition: `U Uᴴ U = U`, and `tr((U Vᴴ)ᴴ U diag(s) Vᴴ) = tr(diag(s) Vᴴ V) = ∑_k s_k`.
   As `U` and `V` are isometries, `‖U diag(s) Vᴴ‖_S1 = ‖diag(s)‖_S1 = ∑_k s_k`.
2. Item 1.2. By item 1.1 with `1 − 1/c > 0`, `K` is a minimum. Let `K + H` be a minimum. Then
   `‖K + H‖_S1 ≤ ‖K‖_S1`, so item 1.1 gives `Π H Π' = 0`. Hence
   `H = U (Hᴴ U)ᴴ + (Π H V) Vᴴ`, a matrix `U Aᴴ + B Vᴴ` vanishing on `Ω`, and `H = 0`.
3. Item 2. Let `w = r_{i₀} + s_{j₀}`, `u_i = exp(iπ Re(r_i / w))` and
   `v_j = exp(iπ Re(s_j / w))`.
   - `K' = diag(u) K diag(v)` has `‖K'‖_S1 = ‖K‖_S1`, as the diagonal matrices are unitary.
   - On the cells of `Ω` where `K ≠ 0`, `u_i v_j = exp(iπ Re((r_i + s_j)/w)) = 1`; where
     `K = 0`, both vanish. So `K'` equals `K` on `Ω`.
   - `u_{i₀} v_{j₀} = exp(iπ) = −1`, so `K'_{i₀ j₀} = −K_{i₀ j₀} ≠ K_{i₀ j₀}`. If M recovered
     `K`, `K'` would be a second minimum.
   - Graph form. If row `i₀` is not joined to column `j₀`, let `r_i = 1` on the rows joined to
     row `i₀`, `s_j = −1` on the columns joined to it, and both `0` elsewhere. The two ends of an
     edge are both joined to row `i₀` or both not, so `r_i + s_j = 0` on the cells, while
     `r_{i₀} + s_{j₀} = 1`.
4. Item 3.
   - 3.1. A `2 × 2` matrix has `‖X‖²_S1 = (σ₁ + σ₂)² = ‖X‖²_S2 + 2 |det X|`, and
     `det X = k₀₀ k₁₁ − x k₁₀`.
   - 3.2. Let `g = c / max(|b|², |c|)`, and `g = 0` when the max is `0`. Then `|g| ≤ 1`,
     `x* = g conj(b)`, and `c − x* b = c (1 − |b|² / max(|b|², |c|))` is a nonnegative multiple
     of `c`, so `Re(conj(g) (c − x* b)) = |c − x* b|`. For every `x`,
     `2 |c − x b| ≥ 2 Re(conj(g) (c − x b)) = 2 |c − x* b| − 2 Re(conj(g) b (x − x*))`, and
     `|x|² = |x*|² + 2 Re(conj(g) b (x − x*)) + |x − x*|²`, as `conj(x*) = conj(g) b`. Add.
   - 3.3. By 3.1 the nuclear norm is an increasing function of `|x|² + 2 |c − x b|`, which by
     3.2 is least exactly at `x = x*`.
   - 3.4. `det K = 0` gives `c = k₀₁ b`, so `x* = k₀₁ |b|² / max(|b|², |k₀₁| |b|)`. This is
     `k₀₁` when `|k₀₁| ≤ |b|`, and `|b| k₀₁ / |k₀₁|`, of modulus `|b| < |k₀₁|`, otherwise.
   - 3.5. Let `det K = c − k₀₁ b ≠ 0`. If `k₀₁ = x*`, then `det K = c (1 − |b|² / max(|b|², |c|))`
     is nonzero, so the max is `|c|`, `det K` is a positive multiple of `c` and
     `k₀₁ = conj(b) c / |c| = conj(b) det K / |det K|`. Conversely, if
     `k₀₁ = conj(b) det K / |det K|`, then `g = det K / |det K|` satisfies 3.2's conditions with
     `k₀₁` in place of `x*`. So `k₀₁` is also a least point with margin `|x − k₀₁|²`, and two
     such points coincide.
   - 3.6. `|b| = |c| = 1` gives `x* = conj(b) c = c / b`, and `k₀₁ = c / b` exactly when
     `det K = c − k₀₁ b = 0`.
5. Item 4. `U Uᴴ = U Uᴴ K K⁻¹ = K K⁻¹ = 1`, so `U (Uᴴ E) = E`, which vanishes off `(0, 1)`.
6. Item 5.
   - The lemmas. `vᴴ (1 − Yᴴ Y) v = ‖v‖² − ‖Y v‖²`, so `Y` is a contraction exactly when
     `‖Y v‖ ≤ ‖v‖` for every `v`. If `Re tr(Yᴴ X) ≤ ‖X‖_S1` for every `X`, take
     `X = (Y v) vᴴ`: `‖Y v‖² ≤ ‖Y v‖ ‖v‖`, so `‖Y v‖ ≤ ‖v‖`. The converse is Theorem A's bound
     `|tr(Zᴴ X)| ≤ ‖X‖_S1`. For `w = Yᴴ x`, `‖w‖² = xᴴ (Y w) ≤ ‖x‖ ‖Y w‖ ≤ ‖x‖ ‖w‖`, so `Yᴴ` is a
     contraction. Subadditivity: a contraction `Z` with `tr(Zᴴ (X + X')) = ‖X + X'‖_S1` has
     `Re tr(Zᴴ X) ≤ ‖X‖_S1` and `Re tr(Zᴴ X') ≤ ‖X'‖_S1`.
   - 5.1, if. Every `X` equal to `K` on `Ω` has `tr(Yᴴ X) = tr(Yᴴ K) = ‖K‖_S1`, of modulus at
     most `‖X‖_S1`.
   - 5.1, only if. Let `L` be the real vector space of the matrices vanishing on `Ω`. If
     `K ∈ L`, then `0` equals `K` on `Ω`, so `‖K‖_S1 = 0` and `Y = 0` serves. Otherwise let
     `f(H + t K) = t ‖K‖_S1` on the span of `L` and `K`. For `t > 0`, `K + H/t` equals `K` on
     `Ω`, so `‖H + t K‖_S1 = t ‖K + H/t‖_S1 ≥ t ‖K‖_S1`; for `t ≤ 0`, `f ≤ 0`. The nuclear norm
     is positively homogeneous and subadditive, so by the Hahn–Banach theorem over `ℝ`, `f`
     extends to a real-linear `g ≤ ‖·‖_S1` on all matrices. With
     `Y_ij = g(E_ij) + i g(i E_ij)`, `g(X) = Re tr(Yᴴ X)` for every `X`, so `Y` is a
     contraction by the lemmas. Off `Ω`, `E_ij` and `i E_ij` lie in `L`, so `Y_ij = 0`. Last,
     `Re tr(Yᴴ K) = g(K) = ‖K‖_S1 ≥ |tr(Yᴴ K)|`, so `tr(Yᴴ K) = ‖K‖_S1`.
   - 5.2, if. `Uᴴ W = 0` and `W V = 0` give `W = Π W Π'`, so `U Vᴴ + W` is a contraction by
     item 1.1's first step with `Z₀ = W`. `Wᴴ U = 0` gives `Wᴴ K = Wᴴ U Uᴴ K = 0`, so
     `tr((U Vᴴ + W)ᴴ K) = ‖K‖_S1`.
   - 5.2, only if. Take `0 < t ≤ min_k s_k`. Then `K − t U Vᴴ = U diag(s − t) Vᴴ` has nuclear
     norm `‖K‖_S1 − t r`, so `‖K‖_S1 − t Re tr(Yᴴ U Vᴴ) ≤ ‖K‖_S1 − t r`, while
     `Re tr(Yᴴ U Vᴴ) ≤ ‖U Vᴴ‖_S1 = r`. Hence `Re tr(Uᴴ Y V) = Re tr(Yᴴ U Vᴴ) = r`, and, as
     `Vᴴ (1 − Yᴴ Y) V` is positive semidefinite,
     `‖Y V − U‖²_S2 = tr(Vᴴ Yᴴ Y V) − 2 Re tr(Uᴴ Y V) + r ≤ r − 2r + r = 0`. So `Y V = U`. The
     same argument for the contraction `Yᴴ` gives `Yᴴ U = V`. Then `W = Y − U Vᴴ` has
     `Uᴴ W = Vᴴ − Vᴴ = 0`, `W V = U − U = 0` and `1 − Wᴴ W = (1 − Yᴴ Y) + V Vᴴ`, positive
     semidefinite.
   - 5.3 is 5.1 with 5.2. 5.4: a square `V` with `Vᴴ V = 1` has `V Vᴴ = 1`, so
     `W = W V Vᴴ = 0`.
7. Item 6.
   - The lemmas. The matrix unit `E_ij` is a contraction, so `|X_ij| = |tr(E_ijᴴ X)| ≤ ‖X‖_S1`.
     A product of contractions is one, as `‖A B v‖ ≤ ‖B v‖ ≤ ‖v‖`, and so is `Π`, as
     `1 − Πᴴ Π = U Uᴴ`. For contractions `P` and `Q`, a contraction `Z` with
     `tr(Zᴴ P X Q) = ‖P X Q‖_S1` gives the contraction `Pᴴ Z Qᴴ`, which pairs with `X` to the
     same value, so `‖P X Q‖_S1 ≤ ‖X‖_S1`. A real-linear `g` has, from
     `X = ∑_ij (Re X_ij E_ij + Im X_ij (i E_ij))`, `‖g X‖_S1 ≤ C ‖X‖_S1` with
     `C = ∑_ij (‖g E_ij‖_S1 + ‖g (i E_ij)‖_S1)`.
   - The first-order bound. Let `T = U Aᴴ + B Vᴴ`, `R = diag(√s)` and `R' = diag(1/√s)`. The
     factors `P = U R + t B R'` and `Q = V R + t A R'` have `P Qᴴ = K + t T + t² B R'² Aᴴ`, and
     Lemma 1 bounds `‖P Qᴴ‖_S1` by `(‖P‖²_S2 + ‖Q‖²_S2)/2`, which is
     `∑_k s_k + t Re tr((U Vᴴ)ᴴ T) + O(t²)`. So some `C` has
     `‖K + t T‖_S1 ≤ ‖K‖_S1 + t Re tr((U Vᴴ)ᴴ T) + C t²` for every real `t`. Every `H` is such a
     `T`, with `A = Hᴴ U` and `B = Π H V`, plus `Π H Π'`, and `tr((U Vᴴ)ᴴ Π H Π') = 0`; the part
     `t Π H Π'` adds at most `|t| ‖Π H Π'‖_S1`.
   - Item 1.2's hypotheses give linear growth. If no nonzero `U Aᴴ + B Vᴴ` vanishes on `Ω`, the
     real-linear map `H ↦ (Π H Π', H on Ω)` is injective, by the split of `H` above. A left
     inverse is bounded in the nuclear norm, so some `C` has `‖H‖_S1 ≤ C ‖Π H Π'‖_S1` for every
     `H` vanishing on `Ω`. With item 1.1, `c' = (1 − 1/c) / (|C| + 1)` gives linear growth.
   - Linear growth leaves no nonzero `T = U Aᴴ + B Vᴴ` vanishing on `Ω`: for real `t`, `t T`
     vanishes on `Ω`, so `‖K‖_S1 + c |t| ‖T‖_S1 ≤ ‖K + t T‖_S1 ≤ ‖K‖_S1 + t Re tr((U Vᴴ)ᴴ T) + C t²`.
     The bounds at `t` and `−t` add to `c ‖T‖_S1 ≤ C t` for every `t > 0`, so `T = 0`.
   - Linear growth gives the certificate. For `H` vanishing on `Ω` and `t > 0`, the first-order
     bound at `t H` gives `c ‖H‖_S1 ≤ Re tr((U Vᴴ)ᴴ H) + ‖Π H Π'‖_S1 + C t`, so, letting
     `t → 0`, `c ‖H‖_S1 ≤ Re tr((U Vᴴ)ᴴ H) + ‖Π H Π'‖_S1`. Let `a = max(1 − c, 1/2)`, so
     `0 < a < 1`, and `p(X) = Re tr((U Vᴴ)ᴴ X) + a ‖Π X Π'‖_S1`, positively homogeneous and
     subadditive. For `H` vanishing on `Ω`, as `‖Π H Π'‖_S1 ≤ ‖H‖_S1`,
     `p(H) ≥ c ‖H‖_S1 − (1 − a) ‖Π H Π'‖_S1 ≥ (c − 1 + a) ‖H‖_S1 ≥ 0`. By the Hahn–Banach
     theorem the functional `0` on these `H` extends to a real-linear `g ≤ p`, and as in item
     5.1, `g = Re tr(Yᴴ ·)` with `Y` vanishing off `Ω`. Let `W = Y − U Vᴴ`. Then
     `Re tr(Wᴴ X) = g(X) − Re tr((U Vᴴ)ᴴ X) ≤ a ‖Π X Π'‖_S1` for every `X`. At `X = U (Uᴴ W)`
     and at `X = (W V) Vᴴ`, where `Π X Π' = 0`, this says `‖Uᴴ W‖²_S2 ≤ 0` and
     `‖W V‖²_S2 ≤ 0`. With `‖Π X Π'‖_S1 ≤ ‖X‖_S1` it makes `W / a` a contraction, by item 5's
     lemma, and `c' = 1/a > 1`.
   - 6.2. For `X` equal to `K` on `Ω`, `X − K` vanishes on `Ω`, so
     `‖K‖_S1 + c ‖X − K‖_S1 ≤ ‖X‖_S1`. A minimum `X` has `‖X‖_S1 ≤ ‖K‖_S1`, so `X = K`.
   - 6.3. A square `V` with `Vᴴ V = 1` has `V Vᴴ = 1`, so the matrix unit `E` at the missing
     cell is `(E V) Vᴴ`, a nonzero `U Aᴴ + B Vᴴ` vanishing on `Ω`, and linear growth leaves none.
     For a square `U`, `E = U (Eᴴ U)ᴴ`.
   - 6.4. By item 3.1 with `x = 1 + h`,
     `‖K + h E₀₁‖²_S1 = 4 + 1 + 1 + (1 + h)² + 2 |1 − h| = 9 + h²` for `h ≤ 1`. If `c > 0` gave
     linear growth, let `t = min(1, c)`. As `‖t E₀₁‖_S1 ≥ t`, `3 + c t ≤ √(9 + t²)`, so
     `6 c t + c² t² ≤ t² ≤ c t`, which is false for `c t > 0`.
   - 6.5. The complex-linear map `Ψ(C, B) = (U C + B Vᴴ)` on `Ω`, on pairs of an `r × n` and an
     `m × r` matrix, has `r n + m r = rank Ψ + dim ker Ψ` and `rank Ψ ≤ |Ω|`. For `(C, B)` in
     `ker Ψ`, the hypothesis with `A = Cᴴ` gives `U C + B Vᴴ = 0`. Multiplying by `V` on the
     right gives `B = −U (C V)`, and by `Uᴴ` on the left `C = −Uᴴ B Vᴴ = (C V) Vᴴ`. So `ker Ψ`
     lies in the image of `D ↦ (D Vᴴ, −U D)`, `D` an `r × r` matrix, of dimension at most `r²`.

### Lean status

All declarations are in `Transformer.Grokking.FlatSpectrum` and are proved without `sorry`.

| Claim | Lean | Differences |
| --- | --- | --- |
| Item 1, the decomposition | `svd_hypotheses` | `r` is any finite type, `s : r → ℝ` with `s ≥ 0`. Orthonormal columns are `Uᴴ * U = 1`, `Vᴴ * V = 1`. |
| Item 1.1 | `isContraction_mul_conjTranspose_add`, `isContraction_mul_conjTranspose`, `le_nuclearNorm_add_of_certificate` | `Y` vanishing off `Ω` is `∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0`; `c W` a contraction is `IsContraction ((c : ℂ) • W)` for `c : ℝ`. |
| Item 1.2 | `isMinOn_iff_of_certificate` | Stated for each `X` equal to `K` on `Ω`: `IsMinOn nuclearNorm S X ↔ X = K`, `S` the matrices equal to `K` on `Ω`. The condition quantifies over `A : Matrix n r ℂ`, `B : Matrix m r ℂ`. |
| Item 2 | `not_forall_isMinOn_of_not_rigid_support`, `reachable_of_forall_isMinOn` | "M recovers `K` from `Ω`" is `∀ X ∈ S, IsMinOn nuclearNorm S X ↔ X = K`. The graph is `patternGraph {ij \| ij ∈ Ω ∧ K ij.1 ij.2 ≠ 0}` on `m ⊕ n`; joined is `Reachable (.inl i₀) (.inr j₀)`. |
| Item 3.1 | `nuclearNorm_threeCells` | Stated as `‖X‖_S1 = √(…)`, with `Ω = {ij \| ij ≠ (0, 1)}` in `Fin 2 × Fin 2`. |
| Item 3.2 | `threeCellFill`, `sub_threeCellFill_mul`, `sq_norm_add_le_of_subgradient`, `sq_norm_threeCellFill_add_le` | `x* = threeCellFill b c`; Lean's `x / 0 = 0` gives `x* = 0` when `b = c = 0`. `sq_norm_add_le_of_subgradient` is the inequality for any `g`, `y` with `‖g‖ ≤ 1`, `Re(conj(g) (c − y b)) = ‖c − y b‖` and `y = g conj(b)`. |
| Item 3.3 | `threeCells_completion`, `isMinOn_threeCells_iff`, `forall_isMinOn_threeCells_iff` | |
| Item 3.4 | `threeCellFill_mul`, `eq_threeCellFill_mul_iff`, `threeCellFill_mul_of_lt`, `forall_isMinOn_threeCells_iff_of_det_eq_zero` | Rank at most one is `K.det = 0`. |
| Item 3.5 | `eq_threeCellFill_iff_of_sub_ne_zero`, `forall_isMinOn_threeCells_iff_of_det_ne_zero` | Rank two is `K.det ≠ 0`. |
| Item 3.6 | `threeCellFill_of_norm_eq_one`, `eq_threeCellFill_iff_of_norm_eq_one`, `forall_isMinOn_threeCells_iff_det_eq_zero` | Moduli one is `∀ i j, ‖K i j‖ = 1`. |
| Item 3, examples | `example`s after `isMinOn_threeCells_iff`, `forall_isMinOn_threeCells_iff` and the three theorems of 3.4–3.6 | The matrices are `!![2, 1; 1, 1]`, `!![2, 1; 1, -1]` with minimum `!![2, -1; 1, -1]`, `!![2, 1; 2, 1]`, `!![1, 2; 1, 2]` and `!![1, 1; 1, -1]`. |
| Item 4 | `exists_ne_zero_threeCells_of_det_ne_zero` | `U : Matrix (Fin 2) r ℂ` for any finite type `r`, with only `U * Uᴴ * K = K` assumed; `V` is arbitrary. The matrix is `!![0, 1; 0, 0]`. Its `example` is `!![2, 1; 1, 1]` with `U = V = 1`. |
| Item 5, the lemmas | `isContraction_iff_sum_norm_sq_le`, `isContraction_of_re_trace_le`, `norm_trace_le_nuclearNorm`, `IsContraction.conjTranspose`, `nuclearNorm_add_le`, `nuclearNorm_real_smul` | `‖Y v‖ ≤ ‖v‖` is `∑ i, ‖(Y *ᵥ v) i‖ ^ 2 ≤ ∑ j, ‖v j‖ ^ 2`. The converse direction of "exactly when" is `norm_trace_le_nuclearNorm` with `Complex.re_le_norm`. |
| Item 5.1 | `rieszMatrix`, `re_trace_rieszMatrix`, `exists_isContraction_of_isMinOn`, `isMinOn_iff_exists_isContraction` | "A minimum" is `IsMinOn nuclearNorm S K`, `S` the matrices equal to `K` on `Ω`. The Hahn–Banach theorem is Mathlib's `exists_extension_of_le_sublinear`, on `Matrix m n ℂ` as a real vector space. |
| Item 5.2 | `mul_eq_of_isContraction`, `isContraction_trace_eq_iff` | `K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ` with `∀ k, 0 < s k`. |
| Item 5.3 | `isMinOn_iff_exists_certificate` | |
| Item 5.4 | `isMinOn_iff_of_full_rank` | `V : Matrix n n ℂ`, `s : n → ℝ`. Its `example` is `diag(1, 2)` from its diagonal. That `U Vᴴ` is `polarFactor K` for invertible `K` is not proved; `isMinOn_nuclearNorm_iff` is Theorem A itself. |
| Item 6.1 | `exists_pos_le_nuclearNorm_add_iff` | Linear growth is `∃ c > 0, ∀ H, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) → nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H)`, with `K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ` and `∀ k, 0 < s k`; the certificate is written as in item 1.1, with `1 < c`. |
| Item 6.1, its three parts | `exists_pos_le_nuclearNorm_add_of_certificate`, `eq_zero_of_exists_pos_le_nuclearNorm_add`, `exists_certificate_of_exists_pos_le_nuclearNorm_add` | The first takes item 1.2's hypotheses as `isMinOn_iff_of_certificate` does, with `U Uᴴ K = K` and `tr((U Vᴴ)ᴴ K) = ‖K‖_S1` in place of the decomposition. The Hahn–Banach theorem is `exists_extension_of_le_sublinear`, as in item 5.1. |
| Item 6, the lemmas | `isContraction_single_one`, `norm_apply_le_nuclearNorm`, `IsContraction.mul`, `isContraction_one_sub_mul_conjTranspose`, `nuclearNorm_mul_mul_le`, `nuclearNorm_real_smul_eq_abs`, `eq_sum_re_smul_single_add`, `exists_nuclearNorm_map_le`, `exists_nuclearNorm_add_smul_tangent_le`, `exists_nuclearNorm_add_smul_le`, `exists_nuclearNorm_le_of_tangent` | `t` is real and enters as `(t : ℂ) • H`. The left inverse is Mathlib's `LinearMap.exists_leftInverse_of_injective`. Lemma 1 is `nuclearNorm_le_of_eq_mul`. |
| Item 6.2 | `isMinOn_iff_of_exists_pos_le_nuclearNorm_add` | Stated for each `X` equal to `K` on `Ω`, as `isMinOn_iff_of_certificate` is; it needs no decomposition of `K`. |
| Item 6.3 | `not_exists_pos_le_nuclearNorm_add_of_full_column_rank`, `not_exists_pos_le_nuclearNorm_add_of_full_row_rank`, `single_apply_of_notMem` | `V : Matrix n n ℂ` with `s : n → ℝ`, or `U : Matrix m m ℂ` with `s : m → ℝ`; the missing cell is `(i, j) ∉ Ω`. |
| Item 6.4 | `nuclearNorm_add_smul_single_threeCells`, `not_exists_pos_le_nuclearNorm_add_threeCells` | `h E₀₁` is `(h : ℂ) • single 0 1 1`; `K` is `!![(2 : ℂ), 1; 1, 1]` and `Ω` is `{ij \| ij ≠ (0, 1)}`, with no decomposition of `K`. Recovery is the `example` after `forall_isMinOn_threeCells_iff`. |
| Item 6.5 | `card_mul_add_le_ncard_of_tangent`, `card_mul_add_le_ncard_of_exists_pos_le_nuclearNorm_add` | Stated without subtraction: `card r * card n + card m * card r ≤ Ω.ncard + card r * card r`. The rank–nullity theorem is Mathlib's `LinearMap.finrank_range_add_finrank_ker`. The count at `r = 1` and the comparison `3 < 4` are arithmetic, not stated in Lean. |

### Rank check (torch, 2026-10-29)

`recovery three-cells 20000` and `recovery rank 20000` (`experiments/recovery` in the grokking
repository, `e5cfbe6`) run ADMM, 20000 iterations, on the CPU, seed 0. Entries and factor
entries have moduli uniform in [0.3, 2] and uniform phases.

- Item 3: 1000 tables of each of five kinds: random; rank one, `u vᵀ`; rank two with
  `k₀₁ = x*`; moduli one; moduli one and rank one. On every table ADMM's entry at `(0, 1)` is
  `x*` to within `2.67e-15 ‖K‖_S2`, and ADMM returns `K` (to within `1e-6 ‖K‖_S2` at `(0, 1)`)
  exactly when `k₀₁ = x*` (to within `1e-12 ‖K‖_S2`): 0, 521, 1000, 0 and 1000 tables. The five
  examples of item 3 come out as stated.
- Items 1 and 2: 300 random patterns for each `N × N` with `N = 4, 6, 8`, each rank
  `r = 2, 3` and two kinds of table: `A Bᵀ` with `N × r` factors, and `r` diagonal blocks
  `a bᵀ` of nearly equal sizes with zeros off them. Each pattern keeps every cell with one
  density uniform in [0.3, 0.95]. Joined is item 2's condition. Injective means that the least
  singular value, on `Ω`, of an orthonormal basis of the matrices `U Aᴴ + B Vᴴ` exceeds `1e-9`.
  The certificate is the least `‖W‖_op` that least squares and then 2000 steps of projected
  subgradient descent found among the `W` of item 1. Recovered means
  `‖Z − K‖_S2 < 1e-6 ‖K‖_S2`.

  | class | patterns | recovered | `‖Z‖_S1 − ‖K‖_S1` |
  | --- | --- | --- | --- |
  | not joined | 1446 | 0 | −18.5 to −2.71e-5 |
  | joined, not injective | 1037 | 117 | −15.4 to 8.88e-15 |
  | injective, certificate below 1 | 468 | 468 | −1.42e-14 to 3.55e-14 |
  | injective, no certificate below 1 found | 649 | 0 | −4.18 to −5.50e-6 |

  Each of the twelve runs agrees with items 1.2 and 2. In the last class ADMM finds a completion
  of smaller nuclear norm than `K` every time. The 117 recovered patterns that are not
  injective are all block tables: as in item 4, recovery does not need the injectivity.

## Powered division and the current Rotate90 model (2026-10-31)

This extension addresses the user's request to compare predictions with the current
Peri-RMSNorm SwiGLU model, revise the explanation, and test it again. The earlier source
note is retained verbatim. The new algebra below has elementary proofs and exhaustive
numerical checks. The later *Joint feature recovery and classification* section formalizes
the generic bank separation and robustness results, the powered-division character route,
and its connection to ideal recovery in Lean. The frequency-gcd criterion and the bound
for the actual normalized neural readout below remain mathematical arguments rather than
Lean theorems. Arithmetic uniqueness and the K=1 neural existence result are in
`Transformer.Rotate90`.

### What the existing theorems predict

Theorem A and the later rank-general certificate characterize **minimum-nuclear-norm
interpolation of a specified matrix**. They do not assert that a neural classifier reaches
that interpolant. Their concrete flat example is `a³+ab`, not `(a/b)^K`. A successful neural
run cannot refute those theorems without satisfying their hypotheses. Conversely, a
rank-one kernel alone does not promise that training finds it, that its code identifies
the answer, or that 5000 updates suffice. The pattern still needs the recovery conditions.

The actual model learns a tied embedding, passes through input and branch RMSNorms,
multiplies SiLU gates by values, adds residuals, and minimizes cross-entropy. These are
different constraints from matching one fixed scalar readout on the training cells.
In particular, a low classification loss fixes inequalities between logits, not their
values. With positive epsilon RMSNorm is only approximately invariant to positive
rescaling away from zero. Its parameter penalty cannot be identified with the nuclear
norm of the answer kernel by the pure-factorization lemma. AdamW and Schedule-Free AdamW
also supply no theorem here that their finite iterates find a global penalty minimum.
The missing connection is an assumption to measure, not a consequence of flatness.

For this comparison the existing Theorem E gives an exact, testable ideal-learner
prediction. On the nonzero operand table each character kernel has unit-modulus
entries. If the training bipartite graph is connected and contains a perfect matching,
the minimum-nuclear-norm interpolant uniquely recovers that character kernel. The
experiment checks these two finite conditions on every prime/seed pattern and saves
the matching witness. The conclusion concerns training on the fixed transformed targets
`χ((a/b)^K)` and solving the completion problem. Discovering `χ` from class labels and
reaching such a solution with the actual optimizer are additional tasks.

The finite check passes for all nine combinations of `p=17,97,193` and data seed
`11,17,23`: every graph is connected and every saved matching is a permutation
whose edges are observed. The same operand masks serve every exponent at a fixed
prime/seed. Thus, on these nonzero tables, ideal recovery of a chosen faithful
character predicts exact decoding even in configurations where the network fails.
That discrepancy tests the modeling connection, not the proved completion criterion.

### Rank-one features and a separating answer code

Let `p` be any prime, `n=p−1`, `K` a natural number, `d=gcd(K,n)` and `m=n/d`.
On nonzero inputs the attainable answers form the subgroup `H={q^K}` of order `m`.
For a generator `u` of the nonzero residues, define
`χ_r(u^t)=exp(2πi r t/n)`. Then

```
χ_r((a/b)^K) = χ_(rK)(a) · conjugate(χ_(rK)(b)).
```

This is an outer product, hence a rank-one complex matrix for **every** `K`.
Its real and imaginary parts each have rank at most two. Thus powered division has
an available low-rank route regardless of the size of the integer exponent. The
nonzero task depends on `K mod (p−1)`. When `K>0`, zero numerators are handled by the
additional rank-one feature `[a=0]`; for `K=0`, every answer is 1, including `0^0=1`.

Rank is only part of the question. Write `z=u^d`, a generator of `H`, and retain a
nonempty finite bank of output-character frequencies `R`. Its answer code is

```
C_R(z^t) = (exp(2πi r t/m))_(r∈R),       g = gcd(m, R).
```

**Exact separation criterion.** Two answers `z^t` and `z^s` have the same code exactly
when `t−s` is a multiple of `m/g`. Every code fiber has size `g`; the bank separates
all attainable nonzero answers exactly when `g=1`. Equal codes also imply
`(z^t)^g=(z^s)^g`.

Proof. Code equality says `m | r(t−s)` for every `r∈R`. Bezout's identity for the
integers `m` and the frequencies makes this equivalent to `m | g(t−s)`, and then to
`m/g | t−s`. There are exactly `g` such residues modulo `m`. Raising their ratio to
power `g` gives 1. The outer-product identity follows directly from multiplication
and division of character values. A single faithful complex character (`r=1`) already
separates `H`; several nonfaithful ones can jointly do so, for example frequencies
2 and 3. A constant character has rank one and separates nothing when `m>1`.

**Conditional decoding bound.** Under uniform nonzero inputs, all answers in `H` have
the same number of preimages. Any decoder receiving only `C_R(c)` is correct on at
most `1/g` of those inputs. This bound does not apply to a model with additional
features that distinguish the fibers. In particular, discarding a small fraction of
embedding energy can discard the entire distinction between answers.

For an injective finite vector code with at least two labels let
`γ=min_(c≠c') ||C(c)−C(c')||>0`. If a predicted
code differs from the true one by less than `γ/2`, nearest-code decoding is correct:
the triangle inequality makes every other code farther away. This is a sufficient
condition for that decoder, not an assertion that the trained tied-readout network
uses nearest-code decoding. Margin and recovery error are needed as well as rank.

### A bound for the actual normalized, tied readout

For the current model, write `L_c(a,b)=<h(a,b),E_c>`, where `h` is the final RMSNorm
output, of width `W` and positive epsilon. Directly from its definition,
`||h||²=||x||²/(||x||²/W+epsilon)≤W`. Thus for every pair of candidate labels,

```
|L_c(a,b) − L_c'(a,b)| ≤ sqrt(W) ||E_c − E_c'||.
```

This follows from Cauchy--Schwarz and applies to every depth and all branch weights.
If a projection gives identical rows to the two labels and `E=E_projected+R`, the
right-hand side is `sqrt(W)||R_c−R_c'||`. An exact collision forbids a strict margin
between them. A small residual only bounds the margin; it does **not** bound accuracy.
This is a direct statement about the current forward pass, unlike the assumed
identification of weight decay with nuclear-norm completion.

There is no universal energy threshold below which a mode is irrelevant to accuracy,
even after centering. For four cyclic labels `c=0,1,2,3`, put
`v_c=(cos(pi*c/2),sin(pi*c/2))` and `E_c=(A*(-1)^c,0,epsilon*v_c)` with
`epsilon>0`. Both the parity coordinate and the faithful circle code are centered.
States `h_c=(0,0,v_c)` have norm 1 and give the correct label strictly highest score.
The separating energy fraction is `epsilon²/(A²+epsilon²)`, arbitrarily small as
`A` grows. These states are compatible with width-4 RMSNorm's norm bound (and can
be its outputs for suitable inputs). This readout example does not claim that the
full network learns those states; it disproves an accuracy conclusion from energy
alone. The experiment records the pairwise bound and actual gaps between aliases.

### Comparison and a prospective retry

The [powered-division experiment](../experiments/rotate90_power_learning/README.md)
records matched train/heldout tables, actual checkpoints and an independent arithmetic
oracle. In the discovery seed, width 64/depth 1 learns the P=97 and P=193 tasks almost
perfectly, while narrow or deeper variants sometimes learn only a quotient of the
answer group. Examples from the initial audit:

| Case | Exact heldout accuracy | Accuracy after powering both prediction and target |
| --- | ---: | ---: |
| P=97, K=2, W=16, D=1 | 32.11% | 99.98% after power 3 |
| P=193, K=2, W=64, D=4 | 31.67% | 99.70% after power 3 |
| P=193, K=3, W=16, D=1 | 11.95% | 100% after power 8 |

These are descriptive measurements, not proofs of the optimization mechanism. The
original flat-spectrum account alone does not distinguish them from the successful
variants: each task admits separating rank-one characters. The revised account asks
which modes are learned, whether their joint code separates the answer image, and how
the small remaining modes affect margins. It retains the exact completion theorems
and rejects treating the minimum effective rank of one readout as a learning criterion.

The retry is specified before new training in
[`hypotheses.json`](../experiments/rotate90_power_learning/hypotheses.json): repeat the
three aliasing cases and three controls with seeds 17 and 23; compare Schedule-Free
AdamW at three rates in the aliasing cases. Fourier analysis uses the reachable
nonzero **output** labels, not an assumed full label group. The approximate rule calls
an alias of size `g` when at least 95% of centered output-embedding energy lies at
frequencies divisible by `g`. It predicts over 95% accuracy after raising both outputs
to `g` and exact nonzero accuracy within ten percentage points of `1/g`. Sensitivity
at 90%, 99% and 99.9% is recorded. This threshold rule is explicitly falsifiable;
the exact separation theorem does not justify it for approximate codes. `g=1` supplies
no convergence prediction. Results, including counterexamples, belong in the experiment
report and must not be used to silently retune the rule.

## Joint feature recovery and classification (2026-10-31)

This section answers the user's request to generalize the Lean theory beyond a single
rank-one readout. It builds on *Rank above one*: matrix recovery already covers arbitrary
rank. The new results connect recovered feature banks to classification, distinguish the
ranks of feature, score and indicator matrices, and quantify robustness. All seven items
below are formalized in `Transformer.Grokking.FlatSpectrum.FeatureCodes`; there are no
additional axioms or unfinished proofs in these modules.

### Setting and statements

Let `A` be an answer type, `R` a finite feature index, `C : A → R → ℂ` an answer code,
and `w : R → ℝ` nonnegative weights. Assume `|C(a,r)|² = 1` for all answers and features.
For a computed feature vector `z`, define the real class score and the ideal code score by

```
S(z,c) = ∑_r w_r Re(z_r conjugate(C(c,r))),
Q(a,c) = S(C(a),c),
D²(a,c) = ∑_r w_r |C(a,r) − C(c,r)|².
```

1. **Exact gap.** `Q(a,a) = ∑_r w_r` and
   `2(Q(a,a) − Q(a,c)) = D²(a,c)`. Thus `Q(a,c) ≤ Q(a,a)`. The identity itself needs
   no sign condition on the weights; nonnegativity is used for the inequality.
2. **Joint separation.** `Q(a,c) < Q(a,a)` exactly when there is an `r` with
   `w_r > 0` and `C(a,r) ≠ C(c,r)`. When every weight is positive, unique classification
   of every answer is equivalent to injectivity of the entire map `C`. Individual features
   may collide. For example, on the four binary pairs `(a,b)`, the two features
   `(-1)^a` and `(-1)^b` jointly separate all four answers, although each feature alone
   identifies pairs of answers. With weights `1/2`, self-scores are one.
3. **Finite margin and score errors.** If `A` is finite and the active features separate
   each distinct pair, there is a `δ > 0` such that
   `δ ≤ Q(a,a) − Q(a,c)` whenever `a ≠ c`. More generally, for arbitrary ideal real
   scores `s` and approximate scores `L`, if the true answer `a` satisfies
   `s(a) − s(c) ≥ δ` for every competitor, `|L(c) − s(c)| ≤ ε` for every class including
   the true one, and `2ε < δ`, then `L(c) < L(a)` for every competitor. Equality in the
   error threshold is deliberately excluded because it can permit a tie.
4. **Feature errors.** For a unit-modulus answer code,
   `|S(z,c) − Q(a,c)| ≤ ∑_r w_r |z_r − C(a,r)|` for every class `c`. Consequently,
   weighted feature error at most `ε` preserves classification when `2ε < δ`.
   If `∑_r w_r = 1`, coordinate errors at most `ε` imply this score-error bound.
   Fixing the total weight fixes the scale of this comparison.
5. **Joint ideal recovery at arbitrary rank.** Let `K_r` be complex matrices on finite
   operand sets and let `Ω` be the observed cells. Suppose each `K_r` is the unique
   minimum-nuclear-norm completion of its values on `Ω`. Then the bank `K` is exactly
   the minimum of

   ```
   Φ(X) = ∑_r ||X_r||_S1,
   subject to X_r(i,j) = K_r(i,j) for all r and (i,j) ∈ Ω.
   ```

   The existing arbitrary-rank dual certificates imply these hypotheses component by
   component; each component can use its own singular vectors and certificate. In the
   explicit certificate corollary, the auxiliary singular-vector index and strict
   contraction factor are shared across the bank. No rank-one assumption is imposed.
   If `K_r(i,j) = C(f(i,j),r)` and the active answer code separates labels, every minimizing
   bank classifies every input correctly, including unobserved cells. The statement
   explicitly assumes observed agreement and global optimality of this objective.
6. **Powered division for every prime and natural exponent.** Work on the unit group
   `G = (ZMod p)ˣ`. For a multiplicative character `χ`, put
   `K_(χ,k)(a,b) = χ((a/b)^k)`. Then

   ```
   K_(χ,k)(a,b) = χ(a)^k conjugate(χ(b))^k,
   rank_ℂ K_(χ,k) = 1.
   ```

   This includes `k=0`. Each character supplies an explicit separable computation.
   If the observation graph on `G × G` is connected and contains a perfect matching,
   every character kernel is uniquely recovered by the ideal learner. The same two
   mask conditions suffice for all natural exponents. A jointly separating bank then
   classifies `(a/b)^k` after joint recovery by item 5. For every prime there is also a
   faithful character and a positive margin working for all exponents: choose a faithful
   character on the whole unit group and take its finite code margin from item 3.
   The conclusion uses the explicit operand formula, not an assumed computed target.
7. **Different ranks.** For arbitrary complex operand factors `u_r(i), v_r(j)`, the
   real class-score matrix

   ```
   L_c(i,j) = ∑_r w_r Re(u_r(i) conjugate(v_r(j)) conjugate(C(c,r)))
   ```

   has rank at most `2|R|`. Indeed, write `v_r(j) C(c,r)` in real and imaginary parts:
   its score contribution is the sum of the two outer products
   `w_r Re(u_r) Re(v_r C_c)` and `w_r Im(u_r) Im(v_r C_c)`. The left factors are shared
   across classes. This bound needs no positivity, unit norms, or separation assumption.
   By contrast, the division indicator `T(a,b) = [a/b = 1]` is the identity matrix on
   any finite group and has full rank `|G|`. Therefore, for every prime, one faithful
   complex character classifies every powered-division task with real score ranks at
   most two; for ordinary division (`k=1`), the identity-class indicator has rank `p−1`.
   The Lean examples include rank 192 at `p=193`. This is not a claim that the trained
   network's logits have rank two or that additional learned modes are unnecessary for
   robust optimization.

### Proofs

For item 1, expand `|z−v|² = |z|² + |v|² − 2 Re(z conjugate(v))` coordinate by
coordinate. Each summand of `D²` is nonnegative, and it is positive precisely when its
weight is positive and its two code values differ. A finite sum of nonnegative terms is
positive precisely when one is positive, proving item 2.

For item 3, take the minimum of the distinct-pair score gaps together with the sentinel
value one. Every entry is positive; including one handles empty and singleton answer
sets without an extra hypothesis. For the perturbation bound,
`L(a)−L(c) ≥ s(a)−s(c)−2ε ≥ δ−2ε > 0`. Item 4 follows from the triangle inequality,
`|Re z| ≤ |z|`, and `|C(c,r)|=1`.

For item 5, every feasible component has nuclear norm at least that of `K_r`. A bank
minimum cannot have a larger total norm than `K`, so the nonnegative component gaps sum
to zero. Each gap is therefore zero, and componentwise uniqueness gives `X_r=K_r`.
The reverse implication follows by summing the component lower bounds. Applying the
arbitrary-rank certificate to each component supplies a direct certificate corollary;
substitution into item 2 proves classification.

For item 6, characters preserve powers and multiplication, and conjugation is inversion
on nonzero character values. The outer product gives rank at most one and the entry at
`(1,1)` is one, proving the lower bound. Unit-modulus factors permit the existing
connected-graph and perfect-matching recovery theorem to be applied unchanged. A faithful
character exists by `exists_injective_mulChar`; restricted to units it supplies the
separating code needed for a uniform margin. The factorization and identity-matrix
calculations in item 7 prove its two rank statements.

### Lean inventory and scope

All declarations below are in `Transformer.Grokking.FlatSpectrum.FeatureCodes`.

| Module | Main declarations |
| --- | --- |
| `Basic` | `twice_score_gap`, `codeScore_lt_self_iff`, `unique_classification_iff_injective`, `binaryCode_injective`, `binaryCode_coordinate_not_injective` |
| `Margins` | `exists_positive_margin`, `strict_winner_of_score_error`, `score_error_le`, `score_error_le_of_coordinate_error`, `strict_winner_of_feature_error` |
| `Recovery` | `bank_minimum_iff`, `bank_minimum_iff_of_certificates`, `classification_of_bank_minimum` |
| `PoweredDivision` | `rank_powerKernel`, `power_scores_correct`, `powerKernel_minimum_iff`, `power_classification_of_minimum`, `exists_power_division_margin` |
| `RankSeparation` | `score_matrix_factorization`, `rank_score_matrix_le`, `rank_divisionIndicator`, `rank_power_scores_le`, `exists_low_rank_power_classifier` |

This generalization proves representation, exact recovery under explicit hypotheses, and
conditional error tolerance. The neural network's objective is still cross-entropy, its
readout is tied, and its state is normalized and nonlinear. No theorem identifies its
parameter penalty with `Φ`, proves convergence of Schedule-Free AdamW, or guarantees a
width/depth-dependent success rate. The new margin results can be applied to actual logits
once their error relative to an ideal score construction is bounded. A small discarded
spectral-energy fraction alone does not supply that uniform error bound.

The cyclic frequency-gcd formula, the general powered-division indicator rank
`(p−1)/gcd(k,p−1)`, and optimal margins for full Fourier banks are outside this Lean
extension. Item 7 formalizes the full-rank indicator comparison at `k=1`; it does not
silently treat the more general formula as proved in Lean. Nonzero operands are explicit
throughout item 6; the arithmetic theorem covers zero numerators. The actual-network
extension below covers them at every natural exponent.

## Actual network expressivity for every exponent

**2026-10-31, new constructive result.** The previous feature-code theorem supplies
ideal scores for every exponent. The earlier actual-network proof only constructs
weights for ordinary division. This section closes that gap: actual tied Rotate90
SwiGLU weights solve `(a/b)^k` for every prime p and natural k, including zero
numerators and k=0. Width 16 with the source hidden size 43 suffices, at every
positive finite depth. All statements concern exact real arithmetic and positive
RMSNorm epsilon. No training-convergence claim is added.

### Statement and quantifiers

Fix a prime p, k in the natural numbers, epsilon > 0, any finite sets of padding
pairs and hidden units, and any depth D >= 1. There exist one embedding table E
and D sets of three branch matrices such that

```
for every a in ZMod p, every b != 0, and every q != (a/b)^k:
    logit(E, blocks, a, b, q) < logit(E, blocks, a, b, (a/b)^k).
```

The width is 14 plus twice the number of padding pairs; hidden size is 20 plus
the number of padding units. Thus the source sizes W=14/H=37, W=16/H=43 and
W=128/H=341 all contain the construction. These are sufficient sizes, with no
assertion of minimality. A one-block W=16 witness has 16p+2064 parameters, or
3616 at p=97. Weights may depend on p, k, epsilon and architecture, but never on
the queried operand pair. The exponent is fixed for each task. There is no
extra input specifying k, and the construction stores no operand-pair lookup table.

The forward pass is the source's shared embedding, quarter-turn, initial RMSNorm,
residual SwiGLU blocks with both branch normalizers, final RMSNorm and tied readout.
`Power.stackLogit_singleton` proves agreement with the existing one-block
`Rotate90.logit`; the stack is the actual loop in `PeriRotate90.forward`.

### The embedding and residual geometry

Choose a faithful complex multiplicative character chi of ZMod p, extended by
chi(0)=0. Define the real answer code

```
C(a) = (Re chi(a), Im chi(a), [a=0]).
```

Every code has squared norm one, including C(0). Faithfulness makes the codes
distinct. Their inner product is one on the diagonal and strictly below one
off the diagonal. Use seven adjacent channel pairs, with embedding

```
E(a) even slots = (1, C(a), C(a^k)),    E(a) odd slots = 0.
```

All padding is zero. One row has squared norm three. The quarter-turn places
b's features in odd slots, so the raw input R=E(a)+J E(b) has squared norm six
for every a,b,k. In particular, C(a^0)=C(1), even when a=0.

Write t=(a/b)^k. Construct the following branch output B, before its RMSNorm:

| Pair block | Even slots | Odd slots |
| --- | --- | --- |
| Constant | 0 | 0 |
| Faithful code, three pairs | C(t)-C(a) | C(t)+C(a) |
| Powered code, three pairs | -C(a^k) | C(a^k) |

The parallelogram identity gives

```
norm(B)^2 = 2 norm(C(t))^2 + 2 norm(C(a))^2 + 2 norm(C(a^k))^2 = 6.
```

Let W be the full width, including padding, and define

```
alpha = 1 / sqrt(epsilon + 6/W)
beta  = alpha / sqrt(epsilon + alpha^2 * 6/W).
```

The actual block input is alpha R, its normalized branch input is beta R,
and its normalized branch output is alpha B. Equality of the two raw norms
makes residual cancellation exact, even with positive epsilon. The state is
alpha(R+B). Its even slots are alpha times `(1,C(t),0,0,0)`; every powered
lookup coordinate cancels out of the tied readout. Odd slots may be nonzero
and remain in the final RMSNorm denominator. Consequently, for a positive
common scalar s depending on the input,

```
logit(a,b,q) = s * (1 + dot(C(t), C(q))).
```

The faithful-code strict inequality makes t the unique winning residue,
including comparisons with zero and with nonzero labels unattainable under k.

### Realizing the branch with actual SwiGLU matrices

The displayed target-dependent B is a specification to be proved, not a new
model operation. It is implemented by fixed matrices as follows. Write
chi(a^k)=x+iy and chi(b^k)=u+iv. Since b is nonzero,

```
chi(t) = chi(a^k) conjugate(chi(b^k)) = (xu+yv) + i(yu-xv),
[t=0] = [a^k=0].
```

Four products compute xu, yv, yu, xv. Six further products multiply the
constant input coordinate by the six coordinates of C(a) and C(a^k). Their
sparse output directions supply the cancellations and copy the powered zero
flag into the target-code slot. This is ten products in total.

Each product is implemented by two genuine SwiGLU units using
`SiLU(z)-SiLU(-z)=z`. Their gates are opposite coordinate projections, their
values select the same second coordinate, and their output directions are
opposite. Since the actual branch input is beta R, output coefficients are
multiplied by beta^-2. The resulting pair computes the unscaled product
exactly. All twenty active hidden units are ordinary trainable units; no
activation is replaced in the forward definition. Unused hidden units are zero.

`Power.branch_weights` checks all sparse coordinate formulas. Gate, value and
output matrices have no dependence on k: only the shared embedding's powered
lookup block changes with the fixed task exponent. The construction includes
p=2, a=0 and k=0; these cases also have explicit Lean examples.

### Depth, unbounded size and what remains unexplained

Set all matrices of each extra residual block to zero. Its output before and
after branch-output RMSNorm is zero, so the block is exactly the identity.
Induction proves that any finite number of these blocks preserves all logits.
Thus for every positive finite depth there are correct weights, with the same
width and hidden size per block.

Quantifying over all finite sizes is enough to state a universal finite-model
impossibility theorem: it would have to exclude every allowed size and every
weight choice. Failure of one infinite-width or infinite-depth limiting model
only implies such impossibility after an additional inclusion argument showing
that all finite networks' functions are covered by its failure theorem. In
particular, failure of a frozen-kernel training limit alone supplies no such
argument about finite feature-learning networks. No infinite limit is needed
for the positive expressivity theorem here.

The W16/D1 failures in the Schedule-Free study therefore cannot be explained
by the absence of a correct function in that architecture. The larger models'
success remains evidence about training under the measured data, initialization
and optimizer settings. This theorem does not establish convergence, a basin
size, finite-precision accuracy for arbitrary p, or a generalization guarantee
from a partial table. It separates that remaining optimization question from
the now-proved capacity result.

### Lean inventory

All declarations are in `Transformer.Rotate90.Power`, imported by
`Transformer.Rotate90` and the project root module.

| Module | Main declarations |
| --- | --- |
| `Defs` | `width_eq`, `width_add_pairs`, `rmsNorm_of_sum_sq`, `rmsNorm_twice_of_sum_sq` |
| `Geometry` | `character_div_pow`, `zeroFlag_div_pow`, `embedding_sq`, `raw_sq`, `branchVector_sq`, `readout_raw_add_branchVector`, `residual_even` |
| `Products` | `weights`, `branch_weights`, `gateMatrix_weights_eq`, `valueMatrix_weights_eq`, `outputMatrix_weights_eq` |
| `Expressivity` | `state_weights`, `logit_weights`, `weights_correct`, `exists_weights`, `exists_weights_width14`, `exists_weights_width16`, `exists_weights_width128`, `width16_parameter_count` |
| `Depth` | `stackLogit_singleton`, `block_zero`, `stackLogit_padded`, `exists_weights_any_depth`, `exists_weights_width16_any_depth` |

## Alias symmetry and training

The expressivity result above removes insufficient representational capacity as
an explanation for the width-16 failures. It does not say which weights training
will find. The next result concerns the actual normalized, tied network and the
actual Schedule-Free AdamW update, at arbitrary finite width and depth.

### Full-table symmetry of the actual loss

Fix a prime p, natural exponent k, positive RMSNorm epsilon and a finite list of
branch matrices B. Write F(A,Bin,C) for the mean cross-entropy on all
`a in ZMod p`, `b in (ZMod p)^units`, with independent names for the first input
embedding, second input embedding and output embedding. The actual tied loss is
`L(E) = F(E,E,E)`. This auxiliary notation changes no layer: it retains the fixed
quarter-turn, every RMSNorm, every residual SwiGLU block and the full softmax.

For units r,s, define row pullback by `T_r E(q)=E(r*q)`. The arithmetic identity

```
((r*a)/(s*b))^k = (r/s)^k * (a/b)^k
```

gives the global identity

```
F(T_r A, T_s Bin, T_((r/s)^k) C) = F(A,Bin,C).
```

Indeed the output transformation reindexes every competing softmax class and
its target. Reindexing a by r and b by s then permutes the complete dataset.
This also holds at a=0 and k=0. It generally fails for a sampled half-table or
one minibatch, because the transformed pair need not belong to that sample.
`Training.fullLoss_permute` proves this identity for the existing arbitrary-depth
forward, rather than postulating symmetry of a surrogate loss.

### The embedding gradient preserves exact aliases

Let S be a multiplicative subgroup of nonzero residues. An embedding is
S-aliased when `E(h*q)=E(q)` for every h in S and every q. Zero is a fixed row;
it need not equal any nonzero row. Assume `gcd(k, |S|)=1`.

At the tied point (E,E,E), each of the three table transformations above fixes
the parameters whenever r,s belong to S. Differentiate the global identity,
first in a direction supported only in A, then only in Bin, then only in C.
The A and Bin partial gradients are constant on S-orbits. The C partial is
constant under multiplication by h^k; coprimality makes these kth powers cover
S. The actual tied derivative is the sum of all three partial derivatives.
Consequently,

```
E(h*q)=E(q) for all h in S  ==>  grad L(E)(h*q)=grad L(E)(q).
```

Every derivative used here exists: the finite network and its cross-entropy are
real analytic in the embedding when RMSNorm epsilon is positive. No generic
position assumption excludes repeated rows or zero activations. The proof does
not assume fixed branch matrices throughout training: the result holds for each
possible matrix choice at each step. It covers simultaneous learning of all
branches. `gradient_sharedLoss_aliased` proves the gradient equality;
`alias_difference_derivative_zero` proves the corresponding zero first-order
signal in a direction that separates two aliased rows.

This is an invariant subspace, not a stationary point theorem. Common embedding
rows and branch matrices can keep moving. It also is not a Hessian, attraction,
or random-initialization probability theorem. Coprimality is a real hypothesis:
without it, the output gradient need not preserve the whole subgroup.

### The source Schedule-Free update preserves the invariant

Use the revised optimizer recurrence in `ScheduleFree.AdamW.Defs`, corresponding
to the lab's `lean_optim/schedule_free_adamw.py`: query y, base z, raw second
moment v, group averaging coefficient c, scalar rate eta and query decay lambda.
In each embedding coordinate the update is

```
v_new = beta2*v + (1-beta2)*g^2
d     = g / (sqrt(v_new/(1-beta2^(t+1))) + eps_opt) + lambda*y
z_new = z - eta*d
y_new = y + c*(z-y) + eta*(beta1*(1-c)-1)*d.
```

The evaluation embedding is `x=y+(1-1/beta1)*(z-y)`. The source's warmup,
maximum-rate history, bias correction and weighted averaging determine scalar
eta and c equally for all coordinates. If two rows of y,z,v and g agree, these
formulas preserve their equality. Initialization sets y=z=E and v=0. Combining
this coordinate fact with the actual-gradient theorem proves that every query,
base, second moment and evaluation embedding stays S-aliased forever.

`trainingRun_aliased` proves this by induction on the independently defined
optimizer recurrence. `trajectory_aliased` permits arbitrary external scalar
parameter schedules satisfying the same source update, including the lab's
external warmup. It also permits the lab's initial maximum-rate history of 0,
where the upstream recurrence starts at -1: no scalar-history condition enters
`StateAliased`. The branch trajectory remains arbitrary. The statements are
real-arithmetic results; implementation roundoff and missing/sparse gradients
are not silently included in the recurrence. The experiment uses ordinary valid
parameters beta1=.9, beta2=.98 and positive epsilons.

### An incorrect model can remain incorrect despite enough capacity

Tied readout gives `logit(h*q)=logit(q)` whenever E is S-aliased, for every input
and hidden state. If S contains h unequal to 1, the query a=b=1 has target 1
for every k, yet class h ties its score. Thus the model does not have the unique
correct winner on the complete task. `trainingRun_not_solves` proves this for
every finite update count, at every finite width/depth, regardless of the branch
trajectory or the chosen evaluation branch matrices.

There is also a quantitative obstruction. For a nonzero target q, its |S|
distinct orbit classes all have the target's score, so

```
sum_c exp(logit(c)) >= |S| * exp(logit(q))
cross_entropy(logit,q) >= log(|S|).
```

`crossEntropy_ge_log_subgroup` and `trainingRun_loss_floor` prove this bound.
`poweredLoss_floor` applies it directly to every nonzero numerator and unit
denominator, including exponent zero. The zero target is excluded from the
orbit-cardinality claim because its orbit is a singleton.

Nonzero witnesses satisfy all hypotheses: at p=3, k=1, take S to be the entire
unit group, put 2 in every coordinate of the zero embedding row and 1 in every
coordinate of each nonzero row. The examples use positive rate and epsilon,
width 16, hidden size 43 and actual Schedule-Free states. Branch matrices are
unrestricted by the theorem. These witnesses refute unconditional convergence
from every initialization; they do not establish failure for almost every
initialization or for the measured random starts.

`aliased_limit` proves that any pointwise limit of aliased embeddings remains
aliased, and `limit_not_solves` retains the readout obstruction at that limit.
Together with the trajectory theorem, even unlimited training time cannot split
these exact aliases if a limit exists. No convergence or infinite-width/depth
limit is assumed. Wider finite networks still admit these bad starts, while the
preceding expressivity theorem supplies good weights at width 16 already.

### Tests of the mechanism and its limits

The [50-run intervention study](../experiments/rotate90_alias_escape/README.md)
records its hypotheses and source checkpoint hashes before training. All runs
use Schedule-Free AdamW, Python 3.15.2, Torch 2.15.0+cu132 and CUDA compile
max-autotune. Each restart resets optimizer histories and batch order from a
pinned original training query; unchanged restarts provide matched controls.
The original averaged checkpoints and original 175-run study remain intact.

The portable [gradient checks](../experiments/rotate90_alias_escape/gradient_checks.json)
use CPU float64 while explicitly retaining the original FP32 RMSNorm epsilon.
At seed 11's embeddings projected onto the observed subgroup, the transverse
fraction of the actual embedding gradient is:

| Task | Subgroup order | Complete table | Original half-table | Noncoprime exponent, complete table |
| --- | ---: | ---: | ---: | ---: |
| P97/K2 | 3 | 2.41e-16 | .22285 | .39131 with K3 |
| P193/K3 | 8 | 3.15e-16 | .22040 | .63062 with K8 |

The complete-table controls satisfy the exact theorem to float64 roundoff.
The other columns demonstrate why its data and exponent hypotheses matter.
After 200 compiled FP32 full-batch steps from exactly projected embeddings,
relative embedding asymmetry is about 7.3e-8 and 7.6e-8 in the evaluation averages.
Minibatches on the same complete table produce about .00888 and .01074 instead.
All eight symmetry-control runs have 100% quotient accuracy, which merges the
aliased classes; their actual task accuracy is only about 34% or 13%. The
nonzero-target entropy floors are log 3 and log 8. FP32 tie-breaking also changes
argmax accuracy between compiled CUDA and eager CPU replay by up to .226
percentage points in the projected controls; this drift is retained separately.

On the half-table interventions, no run crosses the prespecified 99% final
heldout threshold within 5000 new updates. At P193, seeds 17 and 23 improve from
58.14% and 45.09% originally to 72.62% and 74.31% under unchanged restarts, and to
95.81% and 98.39% at zero decay. These two embeddings are far from the seed 11
order-eight alias subspace: their initial relative projection residuals are .94
and .91. The exact symmetry theorem is not an explanation for their remaining
errors. At P193 seed 11, zero decay instead reduces heldout accuracy from 11.79%
to 8.01%; at P97 it also initially improves fitting without reliable generalization.
Freezing either parameter family or injecting the tested noise does not produce
a 99% solution. Every seed and failed intervention remains in the report.

The user's follow-up asks whether more updates or width repair the remaining
errors. [The adaptive budget study](../experiments/rotate90_alias_budget/README.md)
compares 20000-update restarts at decay 1/.1/0 and fresh width 64 controls, with its
hypotheses recorded before those additional runs. This is a finite empirical
test of duration and architecture, not a proof that any failed random start is
trapped forever. All 24 additional runs finished. At zero decay, the two P193
runs above reach 99.74% and 99.989% after 20000 updates; even seed 11 rises to 92.12%.
All six fresh width 64 controls reach 99.96--100% at 5000 updates, with their first
measured 99% points at 750. The P97 width 16 restarts remain below 56% in these
conditions after 20000 updates. The recorded specifications and split hashes
confirm that the duration controls change only the budget, and the fresh width
controls change only width (with hidden size following it). These outcomes
refute a permanent-trap interpretation of the earlier half-table plateaus.
They support an optimization effect of width and decay in this finite study,
while leaving a theorem about typical random-start learning open.

The corrected theory now separates an actual capacity construction from an
actual optimizer obstruction. Predicting which approximate quotient codes
random partial-table training selects, the stability of their neighborhoods,
and the separate P17 memorization failures remain unresolved. Increasing
width or changing decay can help measured training without falsifying the exact
invariant theorem; its hypotheses must be checked before applying it.

### Lean inventory for training

All declarations are in `Transformer.Rotate90.Training`, reachable from the
project root. Every theorem has hypothesis witnesses; no new unfinished claim
or extra axiom is introduced.

| Module | Main declarations |
| --- | --- |
| `Defs` | `untiedLogit_diagonal`, `crossEntropy_permute`, `aliased_iff`, `aliased_zero_nonzero` |
| `Basic` | Analyticity and differentiability of the actual untied and tied full-table loss |
| `Symmetry` | `fullLoss_permute`, `differential_fixed`, `differential_fullLoss_fixed` |
| `Gradient` | Three partial-gradient equalities, `gradient_sharedLoss_aliased`, `alias_difference_derivative_zero` |
| `Optimizer` | `trainingStep_equal_coordinates`, `trainingRun_aliased`, `trajectory_aliased`, `evaluation_trainingRun_aliased` |
| `Readout` | `crossEntropy_ge_log_subgroup`, `poweredLoss_floor`, `not_solves_of_aliased` |
| `Failure` | `aliased_limit`, `limit_not_solves`, `trainingRun_not_solves`, `trainingRun_loss_floor`, `trajectory_not_solves` |

## Sampled updates can break aliases

The next gap is demonstrated by the existing symmetry controls: minibatches on
the complete table produce unequal embedding rows even where a full-table update
preserves their equality. The theorem above concerns the loss used in each step,
not merely the dataset from which that step was sampled. A symmetric mean gradient
does not make each sampled gradient symmetric. The following local calculation
gives an exact mechanism in the actual model, with no change to the architecture.
It treats a completely collapsed embedding; the measured controls merge smaller
subgroup orbits. Thus this is a rigorous escape mechanism and counterexample to
extending the full-table invariant to every batch, not a quantitative prediction
for those particular checkpoints.

### The tied derivative and empirical label counts

Let all embedding rows initially equal a vector v, which need not be zero. For
arbitrary finite branch weights B and positive RMSNorm epsilon, every pair then
has the same actual final hidden vector h = hidden(eps,B,v,v). A perturbation D
of the tied embedding changes each logit by

```
(S1) d logit_q[D] = <D_q,h> + <v, d hidden[D]>.
```

The second term includes both operand lookups, all residual blocks, SiLU and
every normalization. It is common to every output class. Equal initial logits
give uniform softmax, so this indirect derivative cancels in cross-entropy:

```
(S2) d CE_t[D] = (1/p) sum_q <D_q,h> - <D_t,h>.
(S3) grad_E CE_t(q,i) = (1/p - indicator[t=q]) * h_i.
```

This is a derivative at the collapsed point in the full parameter space. It is
not a derivative restricted to moving all rows together or a frozen-feature
approximation. The branch weights are arbitrary, and their input-dependent
derivative is retained until its exact cancellation.

For any nonempty finite indexed batch J, let f_J(q) be its empirical label
frequency, counting repetitions. Ordinary mean cross-entropy therefore satisfies

```
(S4) grad_E L_J(q,i) = (1/p - f_J(q)) * h_i.
(S5) grad_E L_J(q,i) - grad_E L_J(r,i) = (f_J(r)-f_J(q)) * h_i.
```

Different label counts and a nonzero hidden coordinate give a transverse signal.
A label-balanced batch has zero embedding gradient at this point. Sampling with
replacement and fixed subsets are both covered by the indexed-family definition.
The formula does not claim a whole dataset's labels are uniform for every K:
powering can merge distinct quotients, and K=0 has only target 1.

For K=1, fixing a unit denominator makes division a permutation of all p
residues. The full table therefore has frequency exactly 1/p for every label,
including zero. Its embedding gradient at the collapsed point is zero. This
cancellation coexists with a nonzero sampled signal: for two distinct classes
q,r and a uniformly sampled target t, equation (S3) gives

```
(S7) mean_t (grad CE_t(q,i) - grad CE_t(r,i))^2 = (2/p) * h_i^2.
```

Only t=q and t=r contribute, each with squared magnitude h_i^2. The formal
statement is a finite average over labels with arbitrary chosen operand pairs;
the division witness uses a=t, b=1. It also describes uniform sampling from
the complete division table because every label has the same multiplicity and
the collapsed hidden vector is input independent. No independence assumption
about successive optimizer steps or stochastic convergence theorem is used.

### An actual Schedule-Free escape step

Start the source optimizer with y=z=E and zero second moment. At a positive first
rate eta, the averaging coefficient is one, for all real averaging powers. With
beta2<1, the first bias correction cancels 1-beta2, giving

```
(S6) y_new = z_new = E - eta * (g / (|g| + eps_opt) + decay*E).
```

The evaluation embedding agrees with these buffers. If E(q,i)=E(r,i) and
g(q,i)<0<g(r,i), positive eta and eps_opt imply
`y_new(q,i)>y_new(r,i)`. Equal weight-decay terms cancel. This uses the source
AdamW normalization and averaging, not a substitute SGD step. The legal numerical
witness uses beta1=.9, beta2=.98, rate=.003, decay=1, eps_opt=1e-8, with no warmup.
The theorem concerns a fresh first step with positive rate; it does not silently
identify this with the lab's initial zero-rate external warmup step.

For every prime p and natural K, the valid example a=b=1 has target 1. At a
nonzero collapsed hidden coordinate, its gradient has negative target and
positive competitor components. Consequently the optimizer can separate rows
that full-table training preserves under its symmetry hypotheses. Breaking an
alias is necessary progress away from that particular equality constraint; it
does not imply correct predictions on unseen examples, loss convergence, or
reliable escape from every partially collapsed state.

The nonzero witness is explicit: v is the all-ones vector, width is 16, hidden
size is 43, and there are d+1 zero-matrix residual blocks for any natural d.
The rotated input has even coordinates zero and odd coordinates two. Positive
epsilon makes both outer normalization factors positive, so every odd hidden
coordinate is positive. At p=3, classes 1 and 2 share the initial row but separate
after the singleton update. `full_zero_sample_escapes` proves in one statement
that the full-table embedding gradient is zero, the initial table is aliased,
and both the next query and evaluation tables are no longer aliased. The general
`rowStep_not_aliased` covers any prime, exponent and subgroup containing a
nonidentity unit, with the stated positive hidden-coordinate hypothesis.

### Lean inventory for sampled training

The 25 theorems in `Transformer.Rotate90.Training.Sampling` add no unfinished
claims. Every hypothesis has a satisfying example; all modules are reachable
through `Transformer.Rotate90.Training`. The proof uses the existing forward
and optimizer definitions, with no new numerical experiment or optimizer change.

| Module | Main declarations |
| --- | --- |
| `Sampling.Basic` | Actual row-loss analyticity, `fderiv_tiedLoss_constant`, `hidden_ones_pos` |
| `Sampling.Gradient` | `gradient_batchLoss_constant`, `gradient_row_difference`, `singleton_transverse_energy`, `row_gradient_signs` |
| `Sampling.Optimizer` | Source initial-step identities, `initial_step_separates`, `rowStep_separates` |
| `Sampling.Escape` | `frequency_full_division`, `gradient_sharedLoss_division_constant`, `rowStep_not_aliased`, `full_zero_sample_escapes` |

The next unresolved gap is the behavior after this local escape: whether a
sampled trajectory develops a separating answer code and generalizes, with a
prediction of success probability or training time. The new theorem does not
derive those outcomes from breaking one equality. The measured width and decay
effects, partially collapsed checkpoints and P17 memorization remain separate
questions for that next comparison cycle.

Validation on 2026-10-31: full `lake build` passed (13070 jobs); all 25 new
`#print axioms` checks contain only `propext`, `Classical.choice`, and `Quot.sound`.
`./run.py audit` reports 36 existing unfinished statements, zero results resting
on them, and zero extra axioms. Regenerated `INDEX.md` has zero vacuous or
placeholder entries; `./run.py forbidden` passed. No Python code changed.

## Learning transfer

The next comparison replaces the static rank condition with the actual effect
of a sampled optimizer update on unseen equations. Static character capacity
and fixed-code matrix completion do not explain why width 16 and width 64 train
differently, or why P17 fits its training split without generalizing. This is a
conditional dynamics extension, not a proof that a neural optimizer solves the
nuclear-norm interpolation problem of Theorem A.

Let theta contain every embedding, gate, value and output entry, with its usual
Euclidean metric. For any finite ordered stack, let F(theta) be the mean full
softmax cross-entropy on a fixed evaluation split. All normalizations, both
input lookups and the tied output embedding remain in the forward pass.

```
(T1) epsilon > 0 implies F is analytic, hence C^2, jointly in all parameters.
```

At update t, Schedule-Free stores query y, base z, raw variance v and scalar
averaging history. It evaluates at x = y + (1 - 1/beta1)(z-y). The training
gradient g is the gradient of the actual minibatch loss at y. Set

```
v_next = beta2*v + (1-beta2)*g^2,
P_i = 1 / (sqrt(v_next_i / (1-beta2^(t+1))) + epsilon_optimizer),
c = weight_next / (weight_total + weight_next),
eta = current warmup rate.

(T2) delta = x_next-x = c*(z-x) - c*eta*P*g - c*eta*lambda*y.
(T3) <grad F(x),delta>
     = c*<grad F(x),z-x> - c*eta*<grad F(x),P*g>
       - c*eta*lambda*<grad F(x),y>.
```

T2 follows by expanding the independently defined source `trainingStep` and
`evaluation`, with beta1 nonzero. The gradient at y and evaluation derivative
at x generally differ. The preconditioner depends on the actual sampled
gradient and stored second moment; it is not a freely chosen positive matrix.
The two extra terms show why training/heldout gradient alignment alone is an
incomplete test. T3 includes the change caused by averaging and weight decay.

For a bound L_t on the operator norm of D(grad F) throughout the entire segment
from x_t to x_(t+1), the existing mean-value descent lemma proves

```
(T4) F(x_(t+1)) <= F(x_t) + <grad F(x_t),delta_t> + L_t*||delta_t||^2.
(T5) failed_strict_winner_fraction <= F(x_t) / log(2).
(T6) F(x_N) <= F(x_0) + sum_(t<N) [<grad F(x_t),delta_t> + L_t*||delta_t||^2].
```

T5 includes ties as failures: any competitor at least as large as the correct
logit contributes enough softmax mass to force row loss at least log2. It counts
all answer classes. The conservative Taylor coefficient in T4 is L, not L/2.
Joint C^2 regularity and compactness prove that a finite L_t exists for every
finite segment. They do not compute a useful certified value.

If for every t<N the bracket in T6 is at most -mu*F(x_t), with 0<mu<=1,
`geometric_loss_le` proves F(x_N)<=(1-mu)^N F(x_0). Combined with T5, a value
below .01*log2 certifies at least 99% strict correct winners. The Lean rate lemma
only needs mu<=1 algebraically; positive mu is what makes the rate useful.
Its hypotheses have a nonzero, strictly improving witness at every finite
horizon. The actual-model additive theorem has a positive-depth, nonempty-data,
positive-rate five-update witness, with curvature existence proved rather than
assumed. Neither witness asserts positive heldout transfer from random starts.

### Formal scope and prospective test

`Transformer.Rotate90.Learning.runFrom` uses the real revised Schedule-Free
update and the full gradient of the actual minibatch network. Its initial
history is explicit, including the lab's zero initial maximum. Its samples may
change at every step and may have short epoch tails. The model and optimizer
are not replaced with a linearized kernel or infinite-width limit.

| Module | Result |
| --- | --- |
| `Learning.Defs` | Full finite-stack parameterization, actual parameter count, and T1 |
| `Learning.Basic` | Exact evaluation displacement and directional effect, T2--T3; segment-curvature existence and T4; full-softmax calibration T5 |
| `Learning.Trajectory` | Additive and geometric T6; actual minibatch path and finite-time failure bound using source transfer terms |

`experiments/rotate90_transfer/hypotheses.json` declares fifteen fresh runs
before training: P97K2 and P193K3, widths16/64; P17K1 width 64; depth1 and model
seeds 31/37/41. Data and batch seeds stay fixed at 11. Only Schedule-Free AdamW is
used, with Python 3.15/Torch 2.15 and compiled CUDA max-autotune. Every 50-update
checkpoint retains all model, optimizer and sampler state.

Independent CPU float64 copies preserve the trained float32 RMS epsilon and
replay the next actual batch. H1 checks T2 against the source optimizer. H2
tests whether T3 predicts the next-step heldout-loss sign in at least 90% of
eligible probes. H3 freezes a step 400 forecast using relative first-order
progress at 100/200/400; every false positive and false negative is retained.
H4 compares matched widths. Heldout gradients never enter training.

A measured finite-step remainder is not a certified segment-curvature bound.
The H3 forecast extrapolates future progress without proving it. The unresolved
claim is whether typical random starts maintain enough favorable transfer,
with controlled curvature, to obtain a success probability or training-time
prediction. This study tests that missing premise instead of assuming it.

Validation before training (2026-10-31): full `lake build` passed (13074 jobs).
All 17 new theorem axiom sets contain only `propext`, `Classical.choice` and
`Quot.sound`. `./run.py audit` retains the 36 existing unfinished statements,
with zero results resting on them and zero extra axioms. The source check passes.
`./run.py test` passed 410 lab tests (13 skipped) and 104 optimizer tests
(14 skipped), including bit-exact retained-checkpoint/resume comparisons,
independent source-update replays and a full-network finite-difference derivative
check. All 15 experiment recipes pass `./run.py check`.

### Outcome of the frozen transfer test

All fifteen runs completed their declared 1500 updates at setup commit
`d47112d5`, with unchanged definitions and hypotheses. Every P97/P193 width 64
run reaches 99.87--99.97% heldout accuracy, crossing 99% at updates 600--650.
The six width 16 runs remain at 1.96--17.93% within this budget. All P17 controls
fit training perfectly but reach only 7.35--9.56% heldout accuracy. These results
are scoped to the declared update cap and fixed split, not a capacity bound.

H1 and H2 pass all 120 probes. Maximum evaluation-displacement coordinate error
is 2.042e-16 in the independent float64 source replay. The complete directional
derivative predicts 98 improving and 22 worsening next-step heldout losses, with
no sign errors. Initial-state and checkpoint provenance, identical matched
sample fingerprints and all failed forecasts are retained in the study artifacts.
Before-update replay accuracy exactly matches the recorded CUDA evaluation;
the largest mean-loss difference is 4.040e-7. These are local numerical checks,
not a certified supremum of curvature over each segment.

H3 predicts one of the six final successes, with five misses and no false
positives. This has two causes which should not be conflated. The log2
calibration is sufficient, not necessary: every successful run's final CE is
above 0.01*log2. P97 seed31's predicted loss 0.009553 nearly matches its actual
0.009550, yet neither certifies its observed 99.8711% accuracy. The dynamics
forecast is also inaccurate for P193: width 64 seed31 is forecast at loss 1.977,
but reaches 0.01189 after its relative progress increases later in training.

H4 is rejected: width 64 improves the early statistic in only three of six pairs,
below the preset four-pair threshold. All three P193 early comparisons favor
width 16 even though all three wider models later generalize. At P193 width 64
seed31, relative first-order progress rises from about 0.000275 at update 200 to
0.00440 at 400 and 0.00990 at 750. The measured early statistic is not an invariant
ordering of model sizes or a uniform lower bound on future progress.

Thus the finite-time Lean bound must retain the evolving full update, and its
geometric specialization must quantify over every step in the horizon. T1--T6
do exactly that. They close the algebraic connection from this actual model
and optimizer to evaluation loss; they do not explain why the late acceleration
arises, prove a random-start success probability, or turn a few early derivative
probes into future guarantees. That remains the next gap, alongside obtaining
useful accuracy bounds when mean cross-entropy calibration is conservative.

## Optimizer memory

The next comparison separates new base optimization from the effect of stored
Schedule-Free averaging history. This is an extension of the actual-network
learning-transfer account, not a change to the flat-spectrum matrix theorem.
The source is `optimizers/src/lean_optim/schedule_free_adamw.py`, `step`, `train`
and `eval`, with default averaging powers 0 and 2. Write x for the evaluation
weights, y for the query, z for the base, W for the stored averaging total and
M for the stored maximum learning rate. The source relation is
y = beta1*x + (1-beta1)*z, with beta1 positive in the executable API.

**M1, zero-rate continuation.** Setting the current rate to zero, while keeping
the saved state, fixes z and M but advances W by M² per update. Raw second
moments still use each supplied gradient. With M nonnegative, the exact source
evaluation update is x' = (1-c)*x + c*z, where c = M²/(W+M²). Thus a zero
learning rate does not freeze the evaluated model when historical weights
remain positive. This is a continuation, not reinitialization of the optimizer.

**M2, exact clock.** For W > 0, induction through the source recurrence gives

```
z_n = z_0,  M_n = M_0,  W_n = W_0 + n*M_0²,
x_n = z_0 + W_0/(W_0 + n*M_0²) * (x_0-z_0).
```

The Euclidean distance to z_0 is multiplied by that exact scalar. If M_0 > 0,
x_n tends to z_0 regardless of future gradients, decay and changing second
moments. The time scale is W_0/M_0². The real-arithmetic theorem is checked
independently against actual finite-precision source steps; it does not assert
bitwise equality of floating-point repeated averaging and the closed form.

**M3, complete-parameter sign symmetry.** At every finite ordered depth,
`score(-w, a, b, q) = score(w, a, b, q)`. All trainable entries change sign.
The input and normalized hidden states change sign; simultaneously negating
gate and value matrices keeps their preactivations equal; negating the output
matrix reverses the branch; the negative tied embedding reverses the final
readout again. No oddness of SiLU is assumed. Full-softmax losses agree too.

**M4, nonconvex solver region.** Zero weights have zero logits on every row,
and per-row cross-entropy log(p). The existing powered-division construction
supplies a w solving the entire valid table for every prime p and natural k.
By M3, -w solves it too, while their midpoint is zero. Consequently the set
of parameters strictly solving the entire task is not convex, already at
width 16 and one block, with arbitrary additional width/hidden padding allowed.
The witness a=b=1, competing answer 0 works even at k=0. M3 itself covers all
finite depths. This is about parameter averaging; it does not assert a loss
convexity theorem or describe a typical training trajectory.

**M5, an actual source averaging counterexample.** An admissible saved state
with evaluation w, base -w, W=M²=1 and source query y=(2*beta1-1)*w has c=1/2.
One zero-rate source update has evaluation zero for every supplied gradient.
Combining M4 gives perfect endpoints and complete loss of strict winners at
the evaluation midpoint. Admissibility does not prove that random-start
training reaches this particular state.

**M6, classification and loss limit.** For positive RMS epsilon, the actual
all-parameter finite network is analytic. M2 and continuity give convergence
of every logit and every fixed split's full-softmax loss to its base value.
Every strict base logit comparison eventually holds along the stopped path;
finiteness gives one eventual time for all strict comparisons on the whole
table. This includes incorrect base winners. Ties at the base are excluded,
and continuity alone supplies no useful numerical time bound.

The declaration `experiments/rotate90_transfer/memory_hypotheses.json` pins
45 existing checkpoints: all fifteen runs at updates 200, 400 and 750. It
retains narrow models and P17 controls alongside all successful wide models.
The prior seed31 inspection is disclosed in `memory_exploration.json`: strong
evaluation predictions coexisted with poor base predictions, so the local
averaging benefit was not simply delayed arrival at an already solved base.
The counterfactual horizons, source-replay tolerance and transient-benefit
prediction are fixed before their evaluation. The 50000-update point uses
M2 directly; 32 actual source updates independently check the formula using
real gradients and the saved sampler. Compiled CUDA evaluation retains Python
3.15, Torch 2.15 and max-autotune. No new optimizer or fitting recipe is used.

These results distinguish historical averaging from continuing base learning.
They do not yet explain the favorable parameter-path geometry that emerges
under positive-rate Schedule-Free training, predict its late acceleration,
or prove a success probability from random starts.

Setup validation (2026-10-31): all 27 `Rotate90.Memory` theorems pass the full
Lean build (13080 jobs) and individual `#print axioms`; only `propext`,
`Classical.choice` and `Quot.sound` occur. Audit retains 36 existing unfinished
statements, zero results resting on them and zero extra axioms. The regenerated
index has zero vacuous/placeholder declarations, and the forbidden-source check
passes. The full Python suite passes 412 lab tests (13 skipped) and 104 optimizer
tests (14 skipped), including independent real-gradient continuation and a
depth-three sign-symmetry check. The compiled CUDA smoke check has exactly zero
negation logit error and no strict winner at the zero midpoint. All fifteen
source experiment recipes pass their definition check.

### Outcome of the declared memory controls

The analysis at setup `daab04d6` completed every pinned checkpoint and all ten
declared horizons. H1's maximum coordinate error against 32 actual source
updates is 6.661e-16. All 45 bases remain bitwise unchanged, while every raw
second moment changes and every evaluation point moves. H4's maximum negation
logit error is exactly zero; every zero midpoint has zero strict-winner fraction.
All 450 curve points and all controls are retained in `memory_results.json`.

All six wide P97/P193 update-750 checkpoints satisfy every H3 condition. Their
first stopped-base update improves loss, but their 50000-update closed-form
points drop from initial accuracies 99.76--99.98% to 4.06--7.28%, approaching
the poor base classifiers as M2 and M6 describe. Their lowest loss among the
declared horizons occurs at 100 or 200 additional updates, and every final
loss is above its initial value. Thus locally favorable averaging is not a
uniform benefit over the future path. The genuine trained models exhibit the
behavioral failure; the antipodal theorem independently excludes a general
convex solver-region argument, without claiming those trained paths are
antipodal or cross the zero model.

The intervention removes both gradient and decay movement of the base, while
retaining the actual historical clock and second moments. It therefore tests
stopped base optimization, not the separate causal roles of gradient and decay.
Narrow/P17 controls remain unsuccessful, and the earlier transfer forecast
failures remain recorded. These observations narrow the learning gap to how
ongoing base optimization shapes a favorable nonlinear averaging path; the
present theorems do not derive that geometry or its probability from random
initialization. Exact eventual comparison transfer also excludes ties at the
base and supplies no calibrated finite stopping time.

## Gradient forcing

The memory control changed the learning rate to zero, which removes both
normalized-gradient movement and query decay. The next test varies decay
alone in matched positive-rate continuations and uses a zero-gradient decay
control. It preserves each source's query, base, raw second moment, averaging
total, maximum rate, absolute optimizer time and next minibatch. Only the
declared rate/decay may change; the beta, epsilon and warmup settings agree.

**F1, actual two-buffer source update.** Let c be the current averaging
coefficient, eta the current rate, beta the first beta, d the query decay,
h = beta*(1-c)-1, and v the normalized gradient using the updated raw second
moment and its actual bias correction. The source update is exactly

```
[ y' ]   [ 1-c+eta*d*h    c ] [ y ]   [ eta*h*v ]
[ z' ] = [ -eta*d         1 ] [ z ] + [ -eta*v  ].
```

This is an identity for the existing `AdamW.trainingStep`, with the old query
in decay. The gradient input and its changing second moment are retained.
The scalar averaging history depends on rates and time, not the gradient
values. Therefore the same coefficient matrices govern every parameter and
both the original and zero-gradient trajectories from one saved history.

**F2, homogeneous evolution and forcing.** Denote those matrices by A_t and
the normalized-gradient terms by b_t. Independently define the matrix product
Phi from ordered source transitions and R_0=0, R_(t+1)=A_t R_t+b_t. Induction
through the actual source proves buffers_t = Phi_(0,t)*buffers_0 + R_t.
The response also equals the finite sum of past b_j transported by all
subsequent matrices. The b_j remain dependent on the actual trajectory and
variance; this is not a linear model of the neural network or a forecast
that replaces the unknown future gradients.

**F3, the gradient-free parameter plane.** With zero gradient tensors, b_t=0
even while the raw second moments decay from their saved values. Query and
base, and hence evaluation parameters, stay in the linear span of the initial
query and base. More generally any subspace containing the initial buffers
and all normalized-gradient inputs contains the entire finite trajectory.
Escaping it therefore requires at least one normalized-gradient input outside
it. Coordinatewise Adam normalization matters: this statement concerns v,
not merely the unnormalized minibatch gradient.

**F4, actual finite network.** The source identities permit the oracle to be
the gradient of the full finite Rotate90 sampled loss. The evaluation response
uses the source linear recovery x=y+(1-1/beta)*(z-y), and logits are evaluated
by the actual nonlinear tied network. An escape from the initial parameter
plane supplies a new direction; it does not prove correct labels, heldout
generalization, or favorable future transfer from random initialization.

`experiments/rotate90_forcing/hypotheses.json` pins fifteen step-400 sources
and thirty continuations, plus fixed zero-gradient diagnostic horizons. It
declares source replay, gradient-removal and decay-removal predictions before
their outcomes are observed. The initial new observation and compilation
preserve the loaded query exactly; later evaluations use the ordinary engine.
Full and no-decay arms continue to absolute update 1500 with the same samples.
The zero-gradient long-horizon evaluation uses F1's scalar recurrence, checked
against 32 actual source steps with zero tensors. All arithmetic diagnostics
and controls use the original finite architecture and Schedule-Free AdamW.

Setup validation (2026-10-31): all 27 `Rotate90.Forcing` theorems pass the
full build (13085 jobs) and individual axiom checks, using only `propext`,
`Classical.choice` and `Quot.sound`. The audit retains 36 existing unfinished
statements, zero dependencies on them and zero extra axioms; index and source
checks report zero vacuous/placeholder declarations and no forbidden constructs.
All 418 lab tests (13 skipped) and 104 optimizer tests (14 skipped) pass. The
new tests verify preserved source buffers, moments and sampler, exact CPU
continuation, changed decay behavior and independent forced/homogeneous source
identities. An independent P7 CUDA max-autotune smoke test also preserves the
initial query and reproduces the final query bit for bit. All thirty declared
recipes pass their definition check before the scientific continuations run.

The declared study at setup `9f769888` completed all thirty continuations and
all 150 zero-gradient diagnostic points. H2's maximum coordinate error over
32 actual source updates is 1.11e-15, and its maximum zero-gradient relative
span residual is 2.00e-14. All saved raw moments change under zero gradients;
they were not reset. All six wide P97/P193 full and no-decay continuations
exceed 99% heldout accuracy at update 1500. Their zero-gradient decay controls
finish at 2.95--5.15%; their best accuracies among the fixed horizons are
48.50--94.63%, all below 99%. H3 and H4 pass. This separates continued
gradient information from subsequent decay for those step-400 states; it
does not test no-decay training from random initialization.

Every full trajectory leaves its initial parameter span, with relative
distance 0.395--0.764, so H5 passes even for the failed narrow/P17 controls.
New directions alone do not explain generalization. The remaining theory
gap is the task alignment and accumulation of the normalized gradients that
drive F2's response. Its exact finite history still depends on those actual
gradients; it cannot be used as a learning forecast without controlling them.

H1 fails for all three P17 controls: CE differs from the original endpoint
by 0.34223, 0.01828 and 0.14770, and seed 37 also differs in accuracy by
0.7353 percentage points. All twelve P97/P193 cases pass the declared
tolerances. Initial state equality and matched samples hold in every case.
Post-outcome original/continuation comparisons at four saved steps confirm
identical samplers and scalar optimizer states; P17 parameter discrepancies
grow from about 1e-8 at update 450 to order one by update 1500. Their cause
is not established by that check, and H1 remains failed. One narrow P193
CUDA capture failure completed on an unchanged retry and passed replay.
All results, controls, hashes and attempt records are retained in
`experiments/rotate90_forcing`; no threshold or recipe was adjusted.

## General positional models

The next extension follows the user's request for arbitrary rotations, more
than two input variables, other positional mixers and attention. The supplied
normalization reference is
`/home/misha/python_projects/gpt/src/diff_gpt/domain/model`, at commit
`b98c665bd8faa30729fd368d257a13b6d58712a1`. The five inspected source-file hashes
and the exact scope are retained in `review/sequence_model_source.json`.
All results below concern real arithmetic at finite architecture sizes.

**G1, arbitrary inputs and full predictors.** `SequenceModel` takes an input
type X, a finite set of scalar trainable coordinates I, a finite label type V,
and the complete score function `forward : R^I -> X -> V -> R`. X can be
`Fin N -> ZMod P`, a token tuple, or a variable-length sequence. Sample IDs
can be independent of their data values, retaining multiplicity of repeated
examples and losses at distinct token positions. The source Schedule-Free
recurrence is stated independently, then identified with F1--F4's run.
Its finite matrix-product/normalized-gradient decomposition and zero-gradient
parameter-plane invariant therefore apply without a two-operand assumption.
The nonlinear forward is evaluated after adding the parameter response; logits
and attention probabilities are not assumed to depend linearly on it.

The source identities hold for arbitrary update oracles. The sampled classical
loss gradient is justified at differentiable queries by an explicit theorem
with that hypothesis. An arbitrary forward is not declared differentiable by
definition, and the clipped XSA normalizer does not acquire a global smoothness
guarantee. At a clipping boundary, relating a chosen autodiff convention to an
update oracle is an additional step, not an implicit assumption of this theory.

**G2, arbitrary rotations and positional mixing.** The rotary operator has an
arbitrary real angle for every channel pair. Lean proves angle addition under
composition, the inverse rotation, preservation of squared length and norm,
and the relative-position dot-product identity for every frequency vector:

```
<R(s*omega) x, R(t*omega) y> = <x, R((t-s)*omega) y>.
```

The half-split layout agrees with the existing actual RoPE operator. The
geometric frequency grid is an instance, and pi/2 is another instance; neither
is required by the general theorem. A positional mixer can also transform
each token embedding by an arbitrary position-dependent map before summing.
The ensuing predictor retains the shared embedding, input/final RMSNorm and
the full Peri-RMSNorm SwiGLU stack. The two-input zero/pi-over-two instance
has exactly the previous Rotate90 scores. For arbitrary fixed rotary phase
tables and positive epsilon, the pooled forward is analytic jointly in all
its trainable weights, at every finite tuple length, width and depth.

**G3, information lost in pooling.** An invertible rotation of each vector
does not make a sum of rotated vectors injective. If the positional operators
repeat under a permutation, the pooled representation is invariant under that
permutation, for every embedding. In the special sequence of fixed quarter
turns, positions 0 and 4 use the same rotation. Consequently
`[a,b,c,d,e]` and `[e,b,c,d,a]` have identical pooled representations and all
downstream logits. Lean proves that no finite width, depth or weights of that
special pooled network can strictly predict the first of five binary variables
on every input. This is a limitation of the repeated-phase pooling rule;
the theorem does not impose it on arbitrary RoPE or attention operating on
the unpooled token stream. It supplies no universal variable-count threshold
at which attention becomes necessary.

**G4, the supplied normalized attention model.** `SequenceModel.Attention`
formalizes the supplied diff_gpt no-cache training forward with the following
operations in their source order:

1. Normalize the learned token embeddings by parameter-free RMSNorm.
2. In every block, normalize the attention input; project fused QKV; reshape
   heads; RMS-normalize each Q/K head; apply its specified rotary phases;
   use causal softmax with scale `1/sqrt(head_dim)` on unnormalized values.
3. Apply XSA with clipped L2 normalization of the own value, and explicitly
   zero the first position's attention output. Merge heads and project.
   RMS-normalize that branch output before adding it to the residual stream.
4. Normalize the updated stream; apply bias-free SwiGLU; normalize its output;
   add that branch to the residual stream.
5. Apply final RMSNorm and the original shared embedding as tied readout.

RMS epsilon remains explicit: it is 2^-23 for float32 in the supplied source.
XSA's L2 epsilon is separately 1e-12. The proof does not replace either
normalizer by an exactly unit-length vector. RoPE can be disabled independently
per layer by zero angles, or the forward can take arbitrary angle tables.
The source uses hidden size H=(8W+1)//3; other finite H are a stated extension.
The coordinate encoder and decoder are inverse and cover every source weight,
with parameter count `V*W + D*(4*W^2 + 3*W*H)`. This includes embedding, QKV,
attention output and all three SwiGLU matrices; there are no learned norm
gains, head temperatures or biases. The Schedule-Free theorems now instantiate
this complete model for arbitrary finite input length. KV-cache execution and
floating-point rounding are outside the formal forward's scope. The mathematical
forward receives its phase table directly; correspondence with Python execution
uses positions within the configured block_size. Larger finite contexts are
a mathematical extension or require correspondingly larger source tables.

**G5, normalization changes the quantitative theory.** With positive RMS
epsilon and positive width, Lean proves, for every choice of weights and phases:

```
|head_score| <= sqrt(d_head)
||block(x)_i|| <= ||x_i|| + 2*sqrt(W)
||hidden_L(i)|| <= (1 + 2*L)*sqrt(W)
|logit(i,v)| <= sqrt(W)*||E[v]||.
```

Both normalized branch outputs are responsible for the residual bound; it
does not require a bound on the attention or FFN matrices. The final logit
bound still contains the tied embedding row norm. From the actual normalized
Q/K score bound and the positive causal softmax denominator, every attention
weight with N visible positions satisfies

```
attention(i,j) <= exp(2*sqrt(d_head)) / N,   N = i+1.
```

The bound can exceed one for short contexts. It limits concentration of one
attention row; it does not bound the computational power of the entire stack
or prove failure on all tasks with many variables. Conversely, these bounds
and the source-response identity do not guarantee learning. The static
flat-spectrum/nuclear-norm surrogate still does not describe the complete
normalized nonlinear network's optimizer or generalization.

Source correspondence was checked with `review/sequence_model_source_check.py`
against the pinned Python files, using Python 3.15.2 and Torch 2.15.0+cu132.
Three CPU float64 cases cover width 8/16, depth 1/2/3, length 1/5/7 and all,
mixed or disabled RoPE layers. Logits agree within 2.78e-16 and every trainable
gradient within 6.89e-15. A CUDA float32 max-autotune/fullgraph/static case
agrees within 5.97e-8 in logits and 1.50e-6 in gradients. The equation checker
uses the source's rounded RoPE tables; it is a numerical source check, not
Lean extraction or a floating-point proof. No training outcome is inferred.
All configurations, parameter counts, errors and hashes are retained in
`review/sequence_model_source_check.json`.

Validation adds 37 proved theorems. The complete `lake build` passes (13096
jobs); individual `#print axioms` checks use only the three allowed standard
axioms. Audit remains at 36 existing `sorry`, with zero results resting on
them and zero extra axioms. The regenerated index has 3999 modules and 20682
theorems, with zero vacuous or placeholder declarations; `./run.py forbidden`
passes. Every new module is reachable from `Transformer.lean`.

The 40 lab tests selected by `./run.py test rotary compiled schedule_free`
pass on an unchanged retry after a CUDA graph-capture error in the existing
power-division resume test. The following 31 optimizer tests report two
failures in the float32/float64 Schedule-Free Muon allocation subtests. Both
failures reproduce on unchanged main `2006c06b` with
`./run.py test schedule_free_warmed_cuda_steps_allocate_no_device_storage`:
each records 75 positive allocations of 8519680 bytes. The AdamW allocation
subtests pass, and `./run.py test schedule_free_adamw` separately passes all
six selected tests. No production Python or optimizer code changes here.
The unsuccessful attempts and final checks are retained in the root board's
`.kanban/artifacts/122/`; the broader Python command is not reported green.

## Transformer learning bridges

This extension addresses the review at `cabf2119`. Static matrix completion,
representational capacity, and learning under the actual optimizer have
different hypotheses; the bridge results keep those hypotheses explicit.

**R1, actual local regularity.** With positive RMS and XSA epsilons, every
score of the supplied finite Transformer is analytic jointly in every
embedding and matrix entry at weights where each actual own-value norm
avoids the XSA clipping threshold. `Attention.Regular` spells out this
condition on the values computed from the actual normalized hidden states,
at each layer, position and head. This is a sufficient condition; it is not
claimed necessary at exceptional boundary points where cancellation occurs.

The proof follows scalar coordinates through fused QKV, head reshaping,
Q/K RMSNorm, arbitrary fixed rotary phases, causal softmax, clipped XSA,
attention projection, Peri-RMSNorm, SwiGLU, final RMSNorm and tied readout.
Both open sides of the clipping sphere are covered. Zero value vectors are
inside the clipped region and are included, not excluded as singular. The
causal softmax denominator is proved positive. Induction through all layers
then proves score analyticity and analyticity of the full sampled softmax
cross-entropy, with independent example IDs retaining repeated examples.

The condition is witnessed by zero weights at every positive finite source
configuration, context, rotary phase table and positive clipping epsilon.
The component proofs also have nonconstant and nonzero witnesses on both
sides of the clipping sphere. This discharges differentiability of the
actual loss at regular query points; it does not assign a classical gradient
to arbitrary clipping-boundary points or assert that a training trajectory
always avoids those points.

**S1, the rotary spectrum correction.** Norm preservation of individual
tokens does not preserve the spectrum of their Gram table when different
positions use different rotations. `SequenceModel.RotarySpectrum` proves
an exact counterexample using the genuine rotary operator: the two unit
basis vectors have Gram matrix `[[1,0],[0,1]]`; RoPE frequency pi/2 at
positions 0 and 1 changes it to `[[1,-1],[-1,1]]`. The independent vectors
`[1,1]` and `[1,-1]` change eigenvalues from `(1,1)` to `(0,2)`. Both Gram
diagonal entries remain one. The same lost mode persists after positive
common normalization/scaling. This corrects a possible extrapolation from
orthogonal invariance; it does not change the existing valid common-rotation
or relative-position identities. The counterexample concerns pre-softmax
Gram scores, not a claim that softmax has those same eigenvalues.

**T1--T2, the actual learning operator.** Let `F` be the complete finite
logit table, indexed by independent example IDs and classes, and let `ell`
be ordinary sampled softmax cross-entropy in those logit coordinates. At a
regular query `y`, the proved chain rule gives

```
g = J_train(y)^T r,      r = gradient ell(F_train(y)).
P_t = diag(1 / actual_Adam_denominator(t, history, g)).
K_eval,train = J_eval(x) P_t J_train(y)^T.
```

Here `y` is the source optimizer's query and `x` its evaluation average.
Every embedding and matrix coordinate participates in both Jacobians.
`P_t` includes the actual second-moment history, current-gradient update,
bias correction and optimizer epsilon. Positive optimizer epsilon proves
its nonnegative quadratic form. The same-Jacobian operator `J P_t J^T`
is positive semidefinite. Across splits or parameter points, symmetry and
positivity are not automatic. The full table can include held-out examples;
only the sampled IDs contribute to `ell` and its residual `r`.

**T3, finite updates, not just infinitesimal identities.** Write `c_t` for
the source averaging coefficient, `eta_t` for its rate and `lambda` for
decay. Its exact evaluation displacement is

```
delta = drift - c_t eta_t P_t g,
drift = c_t (base - x) - c_t eta_t lambda y.
predicted_change = J_eval(x) drift - c_t eta_t K_eval,train r.
```

The derivation requires nonzero source beta1 to recover the evaluation
average. It includes the source's drift and query decay; replacing it with
plain gradient descent would give a different formula. If the actual
network is regular along `[x,x+delta]` and
`||J_eval(w)-J_eval(x)|| <= C` on that segment, the mean-value theorem gives

```
||F_eval(x+delta) - F_eval(x) - predicted_change|| <= C ||delta||.
```

`Attention.actual_finite_prediction_error` discharges differentiability and
the gradient factorization for the full supplied forward from its actual
activation regularity. A predicted correct-class margin exceeding
`2 C ||delta||` therefore implies true finite-step strict classification on
every indexed example (`Transfer.source_step_correct`). A scalar version
proves true loss decrease when `D(delta) + C ||delta|| < 0`.
These conditions are certificates to establish or measure; no theorem says
they hold automatically throughout a run, or assigns a useful numerical C
solely from width, depth or flatness. The margin theorem is distribution-free
over its explicit finite row set, not a claim about unseen distributions.

**T4, alignment cannot be replaced by a spectrum.** Negating an evaluation
Jacobian leaves its entire self Gram operator `J P J^T` unchanged but
reverses `J P K^T` against a fixed training Jacobian K. Lean proves both
identities and opposite signs of residual transfer with the same self Gram.
Thus even knowing its whole self spectrum does not determine whether a
held-out residual improves. The necessary generalization of the static
spectral view is to retain cross-split alignment, changing Jacobians, the
actual preconditioner and finite nonlinear error. No equivalence between
nonlinear CE training and fixed-matrix nuclear-norm minimization is asserted.
The existing flat-spectrum completion theorems retain their original scope.

**E1--E8, constructive full-Transformer expressivity.** For every nonempty
finite vocabulary of size p and every binary function `f(a,b)`, the actual
source-shaped model has one fixed set of weights giving the correct label
a strictly larger logit than every rival, on every ordered pair. A sufficient
family is

```
W = 2 (p+1)^2 + 2 padding,
H = floor((8 W + 1)/3),
heads = 1,      depth = extraDepth + 1,      context length = 2.
```

Both padding and extraDepth are arbitrary naturals. This is a finite,
deliberately generous construction, not a minimal-width or efficient-learning
claim. RMS epsilon must be positive and XSA epsilon at most one; the actual
float32 source values `2^-23` and `1e-12` satisfy these conditions. All
embedding/final, Q/K and branch normalizers remain in their actual positions.

The proof has eight concrete steps, each connected to the actual forward:

1. **E1:** positive-epsilon RMSNorm has the explicit inverse
   `sqrt(W eps / (W-||v||^2)) v` for `||v||^2 < W`.
2. **E2:** two opposed actual SwiGLU gates compute one exact product, by
   `swiglu(g,u) - swiglu(-g,u) = g u`, with optional zero padding.
3. **E3:** normalized embeddings are unit token vectors. Zero Q/K gives
   uniform causal scores at any RoPE phases. With unit values, XSA keeps
   half of the earlier token when operands differ and returns zero when
   they agree. The current token's residual is retained.
4. **E4:** a fixed projection and its output RMSNorm produce the ordered
   code `x_ab = e_b + [a!=b] e'_a` in two disjoint banks. Its squared norm is
   one or two. Squares identify diagonal pairs, and context/current products
   identify unequal pairs, canceling the square contribution. Fixed output
   weights compensate both exact RMSNorm input scales.
5. **E5:** the source's fused-QKV cuts and head reshape are proved to realize
   that construction. Attention values, XSA and the context projection are
   active. The arbitrary RoPE table is present, but the chosen zero Q/K
   means this existence witness does not need position-dependent Q/K scores.
6. **E6:** the source H contains all `2(p+p^2)` opposed hidden features;
   residual width contains both token banks.
7. **E7:** the FFN interpolates the inverse-normalized displacement
   `e_f(a,b) - x_ab`, whose squared norm is at most three. Its output RMSNorm
   and residual therefore give exactly `e_f(a,b)`.
8. **E8:** extra blocks are identity residuals. The final normalizer and
   tied raw embedding give logits `L * 1[q=f(a,b)]` with `L>0`.

The main theorem is `Attention.Expressivity.exists_binary_classifier`.
`exists_power_division_classifier` instantiates it with `(a/b)^k` over
`ZMod p`, for every prime p and every natural k. It covers zero numerators,
exponent zero and all nonzero denominators; its stronger statement also
specifies Lean's totalized value at denominator zero. The weights depend
on the task, never on the queried operands. Concrete Lean examples include
p=2,3,97,193, exponents 0,5,17, varied depth and padding. Neither an infinite
width nor an infinite depth limit is needed.

The forward, regularity and learning results cover arbitrary finite context
lengths, heads, widths and depths. The constructive arbitrary-task theorem
above specifically covers binary tasks with one head and two token positions.
The binary theorem alone does not prove arbitrary-arity universality. The
F1--F6 extension below supplies a finite-arity theorem with a different,
explicit readout protocol and constructed active rotary phases. Neither
construction requires a learned model to use its particular circuit or
predicts the required training set.

**S2, representability does not force optimizer convergence.** At all-zero
parameters the full model's hidden state and tied embedding both vanish.
Lean differentiates the actual forward and proves every logit derivative,
the full logit Jacobian and the ordinary CE parameter gradient are zero.
This point is inside the clipped-XSA region and is locally analytic, so the
result is not an artifact of a total derivative at a singularity. Starting
with zero query/base buffers, actual Schedule-Free AdamW keeps both buffers
zero for every finite step, even with averaging, weight decay and evolving
Adam history. All logits remain zero. With multiple classes this cannot
strictly classify any example. The result holds at every positive finite
architecture size: more width, depth, epochs or altered optimizer parameters
alone cannot establish a universal learning guarantee.

Together, the exact successful weights and the genuine stationary run close
the logical gap between capacity and guaranteed learning: capacity is proved,
unconditional convergence is refuted, and finite learning conclusions require
the explicit transfer, regularity and remainder conditions above. Whether
typical random initializations satisfy sufficiently strong conditions on the
experiment tasks remains a mathematical and empirical question.

Validation of this extension: the full `lake build` passes (13121 jobs).
There are 97 new public theorems, plus two private witness lemmas, across
25 new modules; every module is reachable from `Transformer.lean`.
Individual `#print axioms` checks for all 97 public theorems use only the
allowed standard axioms. `./run.py audit` reports 36 existing `sorry`, zero
results resting on them and zero extra axioms. The regenerated index has
4024 modules and 20781 theorems, with zero vacuous or placeholder declarations.
`./run.py forbidden` and `git diff --check` pass. Every theorem with explicit
mathematical hypotheses has a concrete Lean witness, including both sides
of clipping, ordered/equal token pairs, real source epsilons, and nonempty CE
training rows. `./run.py test schedule_free_adamw` passes all six selected
optimizer tests on Python 3.15.2. No production Python code changes or new
training run is part of this proof extension; the previously recorded wider
Muon allocation failures are not claimed fixed. Logs and the theorem-level
axiom script are retained in `.kanban/artifacts/123/`.

## Finite sequence capacity and information limits

**N1, what the actual learner observes.** For an arbitrary parameterized
predictor, two target functions agreeing on every row sampled during the
first n updates induce the same CE function at each update, hence the same
full parameter gradient and the same complete Schedule-Free AdamW state.
This includes query/base buffers, moments, averaging, weight decay, any
initial state and a nonzero starting step. A label at an unobserved row
can therefore be replaced without changing the trained weights. Two
different replacements cannot both be strictly classified. The Lean
theorems are `Information.trajectory_eq_of_labels`,
`unseen_trajectory_eq`, `unseen_correct_incompatible` and
`exists_unseen_failure`. The sampling plan is fixed independently of the
replacement; no claim about label-adaptive sampling is hidden here.

**N2, a target independent of the random seed.** Let a finite seed space
carry arbitrary nonnegative masses summing to one. Each seed may select
its own initialization and sampling plan, provided the same designated row
is unobserved throughout the finite training horizon. There is one fixed
replacement label v such that

```
card(V) * Pr_seed[strictly correct on that row when its target is v] <= 1.
```

The order is `exists v, probability bound over seeds`, not a different
adversarial target chosen after observing each seed. For any seed, at most
one label can strictly win; summing those winner masses and averaging over
labels proves the bound. `Information.exists_unseen_low_success` connects
this counting argument to the actual CE/SFAdamW trajectory using N1.
Ties count as failures, and an equally weighted two-seed/two-label Lean
example attains the bound. Other examples use nonempty observed rows,
nonzero initial weights, the source optimizer parameters and 100 updates.

N2 covers finite distributions, including finite ensembles of PRNG seeds;
it does not assert a measure-theoretic result for a continuous Gaussian.
More importantly, these are **unconstrained target extensions**: changing
one label can leave a known modular-arithmetic family. The result does not
refute learning that family from few examples. It identifies the missing
assumption in a guarantee for arbitrary tasks: information about unseen
labels must come from additional observations or a justified restriction
on the target family. More width, depth, epochs or random starts alone
cannot supply it. Positive transfer certificates T1--T4 retain their stated
scope; convergence from typical starts on the algebraic experiments remains
unproved.

Validation of N1--N2: all 11 new public theorems have individual axiom
checks using only `propext`, `Classical.choice` and `Quot.sound`. The full
build passes (13123 jobs), with no warnings in the new modules. Audit
remains at 36 existing `sorry`, zero dependent results and zero extra
axioms. Index regeneration reports 4026 modules, 20792 theorems and zero
vacuous or placeholder declarations. Forbidden checks and all six selected
Schedule-Free AdamW Python tests pass. No new training run is claimed.
Logs and axiom checks are retained in `.kanban/artifacts/124/`.

**F1--F6, finite capacity for arbitrary arity.** For every nonempty finite
vocabulary of size p, every natural N, and every table
`f : (Fin N -> Fin p) -> Fin p`, there are fixed finite Transformer weights
and a fixed positional rotary phase table strictly classifying every input.
The input is the N ordered operands **followed by the fixed token 0 for
readout**. That token already belongs to the vocabulary. Its use is part of
the theorem's input protocol; this is not an unstated claim about the
unaugmented N-token protocol. A sufficient source-shaped family is

```
W = 2 (p+1)^2 + 6 p^N + 2 padding,
H = floor((8 W + 1)/3),
heads = 1,      depth = extraDepth + 1,      context length = N + 1.
```

Padding and extraDepth are arbitrary naturals, RMS epsilon is positive, and
XSA epsilon is at most one. The main theorem is
`Attention.Finite.exists_finite_sequence_classifier`. Unlike E8, its query
and key are nonzero and its rotary attention weights depend on position.
It constructs a phase table; that theorem alone does **not** quantify over
prescribed tables. P1--P6 below establish capacity for the usual source
integer-position geometric grid, using the original input tokens.

1. **F1, bounded separation:** any finite collection of distinct real
   vectors can be separated by one linear functional chosen in any open
   ball. A finite union of proper hyperplanes cannot contain an open ball.
   Applied to all zero/one subsets of N positions, this gives coefficients
   `1 < w_j < exp(1/sqrt(W))` with distinct subset sums. They can be chosen
   arbitrarily close to uniform. The proof does not demand unbounded
   attention temperatures or increasingly concentrated softmax weights.
2. **F2, actual finite SwiGLU interpolation:** positive common-channel
   values u_i and distinct projective input directions admit a linear
   measurement with distinct ratios alpha_i. Choose a positive common
   separation delta. Three actual SiLU gates at each center, using the
   common channel for offsets, converge after gain rescaling to a triangular
   feature. The evaluation matrices converge to a diagonal matrix with
   entries `u_i^2 delta > 0`. Continuity of the determinant gives an
   invertible matrix at a **finite** positive gain. Its inverse supplies
   fixed output weights interpolating arbitrary vector targets exactly.
   The model activation remains SiLU throughout; ReLU occurs only in the
   auxiliary limit argument. There are three hidden channels per input row.
3. **F3, actual normalization and branch reachability:** these features
   compile into the source's gate, value and output matrices with zero
   padding. Positive-epsilon RMSNorm is injective, and projective coordinates
   relative to a positive common channel survive its row-dependent scale.
   For a finite context table, one fixed positive projection scale makes all
   normalized contexts have norm below one. A unit readout residual and unit
   answer then differ by norm below three. Width above nine places every
   desired displacement inside the true branch-output ball; E1 supplies
   its exact inverse-normalized raw target. Both FFN normalizers remain.
4. **F4, ordered context after XSA:** a disjoint context bank stores
   `sum_j w_j e_(x_j)` with the readout label's coordinate removed. Distinct
   subset sums recover the positions of each other label; remaining
   positions necessarily contain the readout label. Thus this code is
   injective on ordered sequences, including repeated tokens and any finite
   N. The readout residual supplies a fixed common coordinate equal to one.
5. **F5, connection to actual rotary attention:** normalized embeddings are
   unit token vectors. A fixed linear sum of their coordinates creates the
   same nonzero query/key at each position, normalized to a unit vector in
   one actual half-split rotary pair. Set the readout phase to zero and
   operand phase j to `acos(log(w_j) sqrt(W))`. The normalized score
   exponentiates to w_j. The actual causal denominator is
   `D = sum_j w_j + exp(1/sqrt(W))`, including self attention. Unit values
   are above the XSA clip at the source epsilon. Actual own-value subtraction
   deletes exactly the readout coordinate, and a fixed projection copies
   the retained code to the workspace. The explicit zero output at the first
   position handles N=0. Neither a modified mask nor an assumed context
   representation enters the forward.
6. **F6, full source model:** the exact FFN maps each normalized context to
   the inverse-normalized displacement to the answer's unit token vector.
   The actual residual reaches that answer. Additional blocks have zero
   projections and are identity residuals. Final RMSNorm and the tied raw
   embedding give logits `L * 1[q=f(x)]` with `L>0`, so every rival has a
   strictly smaller score. The source hidden-width ratio has room for all
   `3 p^N` channels, and its context window contains the appended token.

This closes the finite-arity **capacity** gap for the stated input protocol
and freely chosen rotary phases. The sufficient size is exponential in N,
and the real-valued generic coefficients and finite gate gain need not be
well conditioned or easy to discover numerically. No useful parameter-size
optimum, floating-point error bound, new training result, or typical-start
convergence claim follows. In particular, N1--N2 still separate available
label information from expressivity; T1--T4 still state the explicit
conditions needed for finite optimizer progress. P1--P6 below close the
prescribed-source-phase and unaugmented-input capacity questions. Learning
the algebraic task family from typical random starts remains a separate question.

Validation of F1--F6: the full `lake build` passes (13138 jobs), with no
warnings in the 15 new modules. All 55 new public theorems have individual
`#print axioms` checks using only the three standard allowed axioms; seven
private witness lemmas are also covered by the whole-project audit. Audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4041 modules and 20854 theorems, with zero vacuous
or placeholder declarations. `./run.py forbidden` and `git diff --check`
pass. Every conditional construction has a concrete Lean witness, including
nonzero Q/K, positive-epsilon normalization, nonempty inputs, N=0, p=2,3,97,193,
three/four-variable tasks, source hidden widths, padding and varied depth.
No production Python change or training run is included. Logs and the
theorem-level axiom script are retained in `.kanban/artifacts/124/`.

## Prescribed RoPE capacity

The source's actual integer-position geometric grid suffices. For every
nonempty finite vocabulary of size p, every **positive** finite arity m,
and every table `f : (Fin m -> Fin p) -> Fin p`, Lean proves that one finite
set of model weights strictly classifies all original input tuples at the
last original token. Positions are `0,...,m-1`, every layer enables RoPE,
and its angles are exactly `position * base^(-2*pair/head_dim)`.
No positional table is selected by the proof and no readout token is added.
The full forward retains embedding/final RMSNorm, both Peri-RMSNorm branches,
Q/K RMSNorm, causal softmax, clipped XSA, bias-free SwiGLU and tied readout.

The public theorem is `Attention.Grid.exists_finite_sequence_classifier`.
Its Lean index is `N=m-1`, so the original input type is `Fin (N+1) -> Fin p`.
A sufficient source-shaped family, with arbitrary natural padding and extraDepth, is

```
W = 2 (p+1)^2 + 6 p^m + 2 padding,
H = floor((8 W + 1)/3),
heads = 1,      depth = extraDepth + 1,      context length = m.
```

RMS epsilon is positive and XSA epsilon is at most one; the source values
`2^-23` and `1e-12` satisfy these hypotheses. The displayed family uses
base 10000. `Grid.exists_classifier` works for any valid configured positive
base, with one head, width above nine, disjoint token/context banks, and
`3 p^m <= H`. The family theorem configures enough source table positions
for the entire input. As in G4, the generic mathematical forward can accept
longer directly supplied phase tables; Python execution requires them to
fit the configured context window.

1. **P1, fixed integer offsets:** source rotary pair zero has frequency one,
   independently of the base or width. Its final-row cosine values are
   `cos(N-j)` for prefix positions `j<N`. Cosines of distinct nonnegative
   integers are distinct: equality would make an integer difference or sum
   equal to `2 k pi`; irrationality of pi forces `k=0`, then equality of the
   original integers. Thus the fixed slopes
   `a_j = cos(N-j)/sqrt(W)` are injective. This proof concerns the actual
   integer positions, not an assumption that arbitrary phases separate inputs.
2. **P2, a bounded ordinary query gain:** for distinct slopes, the functions
   `s -> exp(a_j s)` are linearly independent by Dedekind's theorem for
   multiplicative characters of the additive reals. Every difference of two
   unequal positional subset sums is therefore nonzero somewhere and is
   globally real analytic. Analytic uniqueness makes its nonzero set dense
   and open. There are finitely many subset pairs, so their nonzero sets
   intersect every open gain interval. In particular, some `0<s<1` gives
   weights `w_j=exp(a_j s)` with all subset sums distinct. The proof neither
   assumes independent learned features nor adjusts the prescribed RoPE phases.
3. **P3, genuine fixed-grid attention:** normalize the query to `s e_0` and
   key to `e_0` in the first actual half-split rotary pair. Both are inside
   the positive-epsilon RMSNorm image and have exact finite raw preimages.
   Fixed fused-QKV weights generate these same nonzero queries/keys at every
   token; values are unit token vectors. Source scores are
   `s cos(N-j)/sqrt(W)`, and the self score is `s/sqrt(W)`. The real causal
   denominator is `D=sum_j w_j + exp(s/sqrt(W))`. Actual clipped XSA removes
   the final label's value coordinate. A fixed projection multiplies by D
   and copies the rest to the disjoint context bank. For m=1 the source's
   explicit first-position attention zero gives the empty prefix context.
4. **P4, original last-token readout:** the final label is an input variable,
   not a fixed extra token. Its unit residual has zero coordinates in the
   context bank, while the normalized context has zero coordinates in the
   entire residual token bank. Equality of resulting states first forces
   equality of final labels. Distinct subset sums then recover every prefix
   token, including repeated occurrences of the final label. Positive-epsilon
   RMSNorm preserves this injectivity. The sum of residual token coordinates
   is one on every state, giving the common positive FFN channel even when
   the last label varies. `split_input` proves the prefix/last decomposition
   reassembles the exact original token tuple.
5. **P5, complete normalized model:** choose one positive projection scale
   making the finitely many normalized contexts have norm below one. The
   F2--F3 finite SwiGLU interpolator uses `3 p^m` channels to reach each
   requested unit answer by the actual normalized residual branch. Every
   matrix is fixed across all inputs. Additional identity blocks preserve
   that answer at any positive finite depth. Final RMSNorm and tied raw
   embeddings give logits `L * 1[q=f(x)]`, with `L>0`. The conclusion uses
   `sourceAngles cfg (fun _ => true) (fun j => j.val)` directly; only
   `weights`, not phases or input tokens, is existentially quantified.
6. **P6, finite source dimensions:** the displayed even width contains both
   token banks, has width above nine, and its source FFN ratio contains all
   interpolation channels. Its context window is exactly m, with no hidden
   position beyond the original input. Concrete Lean instances cover m=1,
   two-, three- and four-variable tasks, p=2,3,97,193, padding and varied depth.

The modules follow proof dependencies: `Grid.Separation` proves P1--P2,
`Grid.Head` the actual P3 scores and XSA code, `Grid.Encoder` the fixed matrix
and P4 injectivity statements, and `Grid.Model` P5--P6 for the full forward.
The source remains diff_gpt commit `b98c665bd8faa30729fd368d257a13b6d58712a1`;
its geometric table, half-split layout, source normalization order and
context-window conditions were reread beside these statements.

This closes two capacity gaps left by F1--F6: a prescribed source grid and
the original input protocol. It does not change the earlier flat-spectrum
matrix-completion theorem into a learning impossibility theorem for this
nonlinear model. The sufficient width grows exponentially in arity. The
argument supplies finite real weights but no useful bound on feature
separation, gate size, numerical conditioning or floating-point error.
It also makes no claim that every smaller architecture can implement every
table. Typical-random-start Schedule-Free AdamW convergence, efficient
learning of the modular family, and generalization from partial labels
remain unproved; N1--N2's information limits and T1--T4's conditional
progress statements retain their existing scope. No new training result
is asserted by this capacity proof.

Validation of P1--P6: `lake build` passes (13142 jobs), with no warnings in
the four new modules. All 29 new public theorems have individual
`#print axioms` checks using only `propext`, `Classical.choice` and
`Quot.sound`; the private concrete-witness lemma is covered by the project
audit. Audit remains at 36 existing `sorry`, zero dependent results and
zero extra axioms. The generated index has 4045 modules, 20884 theorems,
zero vacuous declarations and zero placeholders. Forbidden checks and
`git diff --check` pass. Every conditional theorem has a concrete Lean
instance, including a nonempty prefix, both possible final labels, the
source epsilons, a bounded nonzero query gain, and the finite source-shaped
width/depth examples above. All five source hashes still match the pinned
manifest. Logs and the theorem-level axiom script are retained in
`.kanban/artifacts/125/`.

The required `./run.py test` run uses Python 3.15.2. All 418 lab tests pass
with three skips. The standalone optimizer suite runs 104 tests and reports
the two previously recorded Schedule-Free Muon CUDA allocation failures
(float32 and float64); each reports 75 allocations of 8519680 bytes in the
warmed-step allocation assertion. Schedule-Free AdamW tests pass. Thus the
complete Python command exits 1, rather than being reported as green. No
Python or optimizer implementation changed in this proof extension, and
those existing Muon failures are not claimed fixed. The complete log is
retained as `.kanban/artifacts/125/tests.log`.

## Depth, width and precision

The earlier capacity constructions establish existence at a sufficient finite
width. Their additional blocks are identities, so they do not establish a
benefit from active depth or a necessary depth/width tradeoff. A flat spectrum
alone does not supply either conclusion. The following results add explicit
resource accounting and a counting-logic model of the normalized architecture.
They distinguish stored parameter bits, stored activation bits, counting depth,
fixed context length, and optimization success.

### R1: finite parameter information

Fix the architecture, input set X, label set V, and a parameter alphabet before
choosing a target table. One setting of the parameters can strictly classify
at most one complete table X -> V: two different labels on the same row would
require contradictory strict score inequalities. Therefore, universality needs

```
|V|^|X| <= number of parameter settings.
```

With b bits in each of M parameter coordinates and at least two labels, this
implies `|X| <= b*M`. Below that budget there is one fixed table which every
allowed weight setting fails to classify strictly. The target is quantified
before the weights. This argument permits any nonlinear forward and even
exact real arithmetic after decoding the finite weight alphabet. It does not
require finite-precision activations, a kernel approximation, or a rank bound.

### R2: the actual source parameter count

For vocabulary p, residual width W, FFN width H and D blocks, the full source
parameter count is

```
M = p*W + D*(4*W^2 + 3*W*H).
```

The tied embedding is counted once; fused QKV, output projection and all three
SwiGLU matrices are included. For the source ratio `H = (8*W+1)/3`, integer
division gives `M <= p*W + D*(12*W^2+W)`. Consequently, universality on the
complete length-N input space requires

```
p^N <= b*[p*W + D*(4*W^2+3*W*H)]
     <= b*[p*W + D*(12*W^2+W)].
```

The implemented source obstruction uses its actual RMSNorm/RoPE/XSA/SwiGLU
scores, arbitrary fixed phases and any fixed b-bit value codebook. A concrete
instance uses p=3, W=4, H=11, D=2, N=8 and b=8: M=404, hence 3232 stored bits
are fewer than the 6561 input rows. Some complete task cannot be represented
with that budget, even with exact forward evaluation.

This is a worst-case statement about representing **all** tables. It is not
a necessary budget for `(a/b)^K mod P`, whose algebraic structure can make it
much cheaper, nor a claim that increasing the budget guarantees learning.
The precise table-count inequality in R1 is stronger than the one-bit-per-row
consequence displayed here. Inputs restricted to nonzero denominators use
their own finite input set in R1.

### R3: extend the counting model to parallel heads and local nonlinearities

The construction extends the finite-state argument of arXiv:2506.16055v3,
Appendix B.2 and Appendix F, rather than identifying the source block with the
paper's different transformer definition. A counted layer has a denominator
and numerator table for each head coordinate, followed by an arbitrary local
finite-state map. Quotients are compiled using the existing proved rounded
count-cell formulas. Enumerating the query state and the rounded head vector
then implements the local map by Boolean tests.

All parallel heads add **at most one counting level per layer**. The number of
heads and width affect the number of coordinates and enumerated states, not
the number of nested counting operations. The induction proves exact state
simulation for every finite stack and every positive prefix. Initial finite
embedding and final local readout add no counting level.

This compiler is not claimed efficient. With activation precision r>=1 bits,
W coordinates, T possible position tags and a separate sentinel, the state
count is exactly `T * 2^(r*W) + 1`. The proof exposes that width/precision cost
rather than treating arbitrary finite local maps as a bounded-size circuit.

### R4: normalized source adapter and unchanged token inputs

The concrete counted layer uses the source fused-QKV split, head reshape,
RMS-normalized Q/K, arbitrary half-split rotary phases, scaled logits, clipped
XSA and its explicit zero at the first source position. It preserves both
branches' input and output RMSNorm, both residual additions and SwiGLU.
`local_sourceAverage` proves that substituting the unrounded source averages
into this local computation gives exactly the existing real `Attention.block`.
The source exponential establishes the denominator nonnegativity required by
the counting-cell proof.

The **rounded variant** has explicit numerical semantics: round exponential
coefficients and weighted value coordinates before summing; accumulate their
integer mantissas without overflow; round the quotient; then round the stored
residual after each complete block. Round the normalized embedding once on
entry. A zero rounded denominator produces zero attention. The local maps and
final normalized tied scores use real arithmetic. This differs from the
paper's uniform-attention zero-denominator fallback and from IEEE float32
execution, including its accumulation and intermediate rounding.

For a fixed context bound T, residues modulo T+1 identify every positive
position on words of length at most T. The compiler uses those predicates to
recover zero-based source position tags without changing the input vocabulary
or inserting a source token. Its sentinel contributes zero to attention sums.
Thus `source_classifier_mem` and `source_classifier_correct` give an
end-to-end depth-D counting formula on the **original token words**, including
embedding normalization, all rounded blocks, final RMSNorm and tied readout.
The readout tests strict classification; ties are rejected, consistently with
the capacity results. There is no hypothesis assuming that trained logits
already solve the target task.

### R5: exact source RoPE is outside the existing periodic hierarchy

Appendix F, `sec:rope_pes`, assumes rational multiples of pi so that RoPE is
periodic in the integer position. The actual geometric source grid has first
frequency exactly one radian, regardless of its base. Lean proves that this
is not a rational multiple of pi, that its first-pair cosine has no positive
integer period, and that **no finite positional code** can reproduce that
cosine at every natural position. The latter follows from cosine injectivity
on natural numbers and the finite pigeonhole principle.

The bounded-context compiler in R4 is therefore not a uniform simulation for
unbounded context: its modulo T+1 predicates and state space change with T.
It does not transfer the paper's all-length depth hierarchy unchanged to the
exact source model. This obstruction concerns exact real rotations; it makes
no assertion about the eventual behavior of a particular float32 RoPE routine.

| Lean module | Results |
| --- | --- |
| `SequenceModel.Resources.Parameters` | unique strict target, table/bit counting, complete parameter count, source depth/width/precision obstruction |
| `CRASP.Normalized.Counting` | parallel attention-cell compilation and arbitrary local output tests |
| `CRASP.Normalized.Program` | stack simulation, depth-zero embedding, exact RoPE positional obstruction |
| `CRASP.Normalized.Source` | actual source local computation, rounded adapter, stack simulation, activation-state count |
| `CRASP.Normalized.Input` | normalized tied readout and complete bounded-context compiler on original tokens |

The remaining questions are sharper: a task-specific active-depth advantage,
useful tight width/depth lower bounds for powered modular division, a uniform
hierarchy allowing the source's nonperiodic positions, and a quantitative
connection between this rounding model and IEEE execution. None is replaced
by identity padding or inferred from the flat spectrum. Training convergence
and optimal Schedule-Free AdamW settings also remain separate questions.

Validation of R1--R5: the full `lake build` passes (13149 jobs), with no
warnings in the new modules. Individual `#print axioms` checks cover all
31 new public theorems and use only `propext`, `Classical.choice` and
`Quot.sound`; the private lookup witness is also covered by the project
audit. Audit remains at 36 existing `sorry`, zero results resting on a
`sorry`, and zero extra axioms. The regenerated index has 4052 modules,
20916 theorems, zero vacuous declarations and zero placeholders. Forbidden
checks and `git diff --check` pass. Concrete witnesses cover the resource
obstruction, finite state counts, initial-state predicates, positive readable
prefixes, and the actual first geometric frequency. All five source hashes
match the pinned manifest. Proof and validation logs are retained in
`.kanban/artifacts/126/`.

The required Python 3.15.2 regression run passes all 418 lab tests (three
skips). Its 104-test optimizer suite reproduces the two pre-existing
Schedule-Free Muon warmed CUDA allocation failures, float32 and float64,
each reporting 75 allocations of 8519680 bytes. Schedule-Free AdamW tests
pass. Thus `./run.py test` exits unsuccessfully, as at the parent commit;
this Lean-only extension does not fix those unrelated optimizer failures.
The full log is `.kanban/artifacts/126/tests.log`.

## Uniform nonperiodic RoPE

The model's RoPE does not need to be made periodic. The theory needs a
position representation that can express the actual source coefficients.
R5 rules out an exact finite periodic code; it does not rule out a counting
language that directly uses a query/source position pair. U1--U4 provide that
extension and remove the context-bound restriction from the simulation theorem.

### U1: position-dependent weighted counts

The new syntax contains token tests, a first-position test, Boolean operations,
integer constants, addition, integer scaling, comparisons, and the operation

```
Count_K[phi](w, i) = sum_{1 <= j <= i} K(i,j) * 1[phi(w,j)].
```

K is fixed independently of the input word. The count retains the query
position i while evaluating phi at each source position j. A nested count in
phi uses j as its own query position. The constant-one kernel recovers ordinary
causal counting; `Term.ordinary_count` proves this identity.

The grammar does not contain attention, a transformer forward, rounded
division, or arbitrary predicates of the whole word. Rounded quotients are
compiled into integer comparisons of numerator and denominator sums. For
positive D, the unsaturated cell for mantissa y is

```
y*D <= 2^s*N < (y+1)*D.
```

The lower or upper comparison is omitted at the corresponding saturation
endpoint. A zero denominator returns zero, matching R4. Lean proves these
endpoint-aware cells from `Fx.round_eq_iff`, including signed numerators and
complete denominator underflow. Thus division adds no counting level.

### U2: one uniform formula for a finite-state stack

Each coefficient name records its layer, numerator/denominator choice,
fixed query state, head coordinate and fixed source state. Its interpretation
is the corresponding integer mantissa at the position pair (i,j). It cannot
inspect the input word. The actual query state is selected by its previous-layer
formula, and source states are selected inside the count. Reordering finite
sums proves that this enumeration recovers the actual attention sums.

Parallel heads use parallel coordinate formulas. The arbitrary local finite-state
map is then a Boolean test of the current state and rounded aggregate vector.
The induction proves exact simulation with one additional counting level per
block. Tokenwise embedding, the first-position flag and final local readout
add no counting level.

The order of quantifiers is essential: for fixed layers, embedding, output test
and depth D, construct **one formula**, then quantify over **all finite token
words and positive readable positions**. Neither its syntax nor its kernels
depend on a word's length. There is no context bound or annotated token alphabet.
`classifier_correct` proves this statement; `classifier_depth` bounds its
counting depth by D.

Each positional coefficient remains a fixed-point mantissa in
`[-2^(r-1), 2^(r-1))`, even when its dependence on positions is nonperiodic.
This is proved by `kernel_bounds`. Prefix sums still use unbounded integers,
as in the source paper's idealized `sumdiv`; this is not an IEEE accumulation
guarantee.

### U3: construct the kernels from the supplied RoPE architecture

The concrete kernels evaluate the source head score: pre-attention RMSNorm,
fused-QKV projection and reshape, head Q/K RMSNorm, exact half-split rotary
phases, division by sqrt(head_dim), and exponentiation. Numerator kernels
also multiply by the source's unnormalized value coordinate before rounding.
The source phase is exactly

```
angle(layer, position, pair) =
  position * theta^(-2*pair/head_dim)
```

at enabled layers, and zero at disabled layers. Position is an unrestricted
natural number, with no modulus or rational-angle assumption. The logic's
one-based positions are converted to the source's zero-based positions before
computing these phases. `sourceAngles_restrict` identifies this formula with
the existing finite source angle table.

The local map retains clipped XSA, its exact first-position zero, output
projection, both branches' input/output RMSNorm, the two residual additions
and SwiGLU. Only the first-position bit is retained in the state; the absolute
position is supplied to the coefficient computation. That bit is proved
invariant under the stack. For r>=1 activation bits and width W, the finite
state count is `2 * 2^(r*W)`, independent of input length.

`denominator_restrict`, `numerator_restrict` and `finish_restrict` identify
the new operators with the earlier rounded source model at every finite
position. `coefficient_sum` then proves equality of the complete prefix sums.
`aggregate_restrict` covers the rounded attention quotient, including the
zero-denominator case, and `step_restrict` covers the whole normalized block.
The final normalized tied readout is also unchanged. These are exact
agreement results under R4's specified rounding semantics.

### U4: the complete normalized model at every finite length

`uniform_source_classifier` instantiates the compiler with the complete
source parameter table, the exact geometric RoPE grid, arbitrary per-layer
enable flags, normalized/rounded token embeddings, all configured blocks,
and the final RMSNorm and tied embedding readout. It proves

```
for every fixed model and target label,
  there exists one formula phi with depth(phi) <= model.n_layers,
    such that for every finite original-token word w and 0 < i <= length(w),
      phi(w,i) = the rounded source model's strict classification test at i.
```

The formula is chosen before the word, and no premise asserts that trained
logits already solve a task. The stored residual state and formula no longer
grow with a prescribed context bound. This closes the nonperiodic-position
coverage gap in the rounded model's counting simulation. It leaves the
actual RoPE frequencies unchanged and preserves the earlier finite-code
impossibility result.

This is an extension of the language, not a proof that the periodic language's
strict lower-bound hierarchy also holds in the extension. Its source-generated
kernel family must be analyzed to establish new lower bounds; arbitrary kernel
families can have greater power. The upper simulation bound alone does not
prove that every extra layer increases expressivity. The exact real phase
calculation and the fixed-point/real-local-map rounding semantics also remain
distinct from actual float32 execution. No new training or convergence result
is asserted, and the mathematical all-position grid does not enlarge the
Python implementation's configured context storage or KV cache.

| Lean module | Contents |
| --- | --- |
| `CRASP.Rotary.Logic` | explicit syntax, weighted-count semantics, ordinary-count special case |
| `CRASP.Rotary.Cells` | exact quotient cells and position-sensitive state-sum identity |
| `CRASP.Rotary.Layer` | bounded coefficient alphabet and one-layer compilation |
| `CRASP.Rotary.Program` | uniform stack/input/output simulation and counting-depth bound |
| `CRASP.Rotary.Source` | source-generated kernels, exact phases, finite states and tied readout |
| `CRASP.Rotary.Model` | complete prefix/block agreement and the all-length source classifier |

Validation of U1--U4: full `lake build` passes (13156 jobs), with no warnings
in the new modules. All 43 public theorems have individual `#print axioms`
checks using only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4059 modules, 20959 theorems, zero vacuous results
and zero placeholders. Forbidden checks and `git diff --check` pass. All
five pinned source-file hashes still match. Concrete witnesses cover Boolean
state predicates, positive and zero rounding denominators, nonempty prefix
sums, finite activation-state counts and the full source classifier.

The required Python 3.15.2 lab run completed 418 tests with three skips and
one CUDA graph-capture error in
`test_compiled_training_resumes_with_float32_tolerance`. Rerunning that test
alone in a fresh process passed (one test, 8.544 seconds); the original full
run is still recorded as unsuccessful. The separately completed 104-test
optimizer suite reproduces the two existing Schedule-Free Muon allocation
failures, float32 and float64, each with 75 allocations of 8519680 bytes.
Schedule-Free AdamW tests pass. No Python implementation was changed.
The theorem script, source checks and all original/retry validation logs
are retained in `.kanban/artifacts/127/`.

## Quantitative RoPE approximation

This extension keeps exact source RoPE in the model and asks what changing
its frequencies or rounding its attention coefficients can change. The
source is still `diff_gpt/domain/model/{gpt,block,mha,feed_forward,rope}.py`
at `b98c665b`, with the hashes in `sequence_model_source.json`. The periodic
comparison is with arXiv:2506.16055v3, Appendix F, `sec:rope_pes`, read from
`papers/`. As in the earlier correction, periodic frequencies are rational
multiples of pi, rather than rational numbers of radians.

### A1. Quantitative phase and frequency errors

For the actual half-split rotation, including an unchanged odd leftover
coordinate in the mathematical generalization,

\[
 \|R_a x-R_b x\|_2\le\delta\|x\|_2
 \quad\text{if}\quad |a_c-b_c|\le\delta\text{ for every pair }c.
\]

The proof bounds the squared chord between two unit-circle phases by the
squared angle difference and sums over the source channel pairs. With
positive RMS epsilon and positive head width, the normalized source logit
therefore obeys

\[
 |s(a,b;q,k)-s(a',b';q,k)|
 \le (\delta_q+\delta_k)\sqrt d.
\]

This holds for arbitrary query/key magnitudes. For frequencies f and g
with coordinate errors at most eta, the phase error at position t is at
most `|t| eta`. A fixed frequency error does not supply a fixed phase-error
budget at arbitrarily large positions. These statements bound exact plane
rotations; separately rounded sine/cosine entries need not form an exact
plane rotation and are not automatically covered by a phase-error premise.

### A2. Causal softmax and the actual XSA head

If all unmasked logits at a query move by at most r, the L1 change in the
entire causal softmax row is at most `8 r`. The constant is conservative.
For the same value stream with `norm(v_j) <= B`, its average changes by at
most `8 r B`. Applying the source XSA correction with nonnegative clipped-L2
epsilon gives a whole-head phase-error bound

\[
 \|H_a(q,k,v)_i-H_b(q,k,v)_i\|_2
 \le 32\delta\sqrt d B.
\]

The explicit first-position zero is retained. The constants do not grow
with the number of keys when the phase-error and value-norm premises are
fixed. This theorem holds Q/K/V fixed; it is not by itself a stack bound,
since changing early phases changes later residuals and hence later Q/K/V.

### A3. Continuity through the complete source stack

Joint continuity is proved through every actual operation: epsilon RMSNorm,
half-split rotations, fused QKV and head reshaping, causal softmax, clipped
XSA, projection, all four branch normalizers, SwiGLU, residual additions,
normalized embedding, final RMSNorm and tied readout. Weights are arbitrary
and fixed, depth and width are arbitrary valid finite source dimensions,
and both normalizer epsilons are positive.

This includes the XSA clipping sphere. Continuity needs no exclusion of
that sphere, although the earlier analyticity/gradient theorem still does.
The entire residual stream is recomputed in every subsequent block. There
is no premise assuming continuity of a black-box model or assuming a bound
on the final change in logits.

### A4. What a finite-context periodic replacement preserves

`frequencyAngles` varies the shared frequency vector, retaining the source
per-layer RoPE flags and positions. Substitution of `invFreq` is exactly
`sourceAngles`. Finite vectors of rational multiples of pi are dense; for
any rational vector q the actual half-split rotations have a common positive
integer period `2 * product(q_c.den)`.

For every fixed finite T, fixed weights and positive error tolerance eta,
one such rational-pi frequency vector approximates **every logit, every
readout position and every input tuple of length T** to error less than
eta. This is uniform over the finite input domain, not just selected
training examples. A strict source classifier at this fixed T consequently
has a periodic frequency replacement with the same weights, width, depth
and strict decisions. A nonconstant two-token binary sum task witnesses
the hypotheses using the existing exact-source capacity theorem.

The replacement frequencies may cease to be a geometric grid. The theorem
does not claim that merely changing the source's single base parameter can
produce the replacement. Its period and tolerance may depend on T, the
weights, the epsilons and the strict margins. Ties have no such certificate.
The frequency tolerance is existential; this proof does not numerically
compute it for a trained model.

The quantifiers are crucial for the depth question:

\[
 \forall T\ \exists q_T\ \text{(same finite-context decisions)},
 \qquad\text{not established:}\qquad
 \exists q\ \forall T\ \text{(same language)}.
\]

Thus the finite-context replacement preserves the given depth, but does
not transfer the periodic language's strict depth lower bounds to exact
source RoPE. The earlier impossibility of a finite exact position code
remains true. The weighted-count compiler in U1--U4 still supplies the
uniform upper bound of one count nesting per source block. A strict lower
hierarchy for its particular source-generated nonperiodic kernels remains
unproved; the present approximation argument does not remove that gap.

### A5. Quantitative readout and classification certificates

For a final residual error `norm(x-y) <= E`, positive RMS epsilon eps and
a tied embedding row e, the logit error is bounded by

\[
 \frac{2}{\sqrt{\mathrm{eps}}}E\|e\|_2.
\]

A target-versus-competitor decision is preserved if the original strict
margin exceeds

\[
 \frac{2}{\sqrt{\mathrm{eps}}}E
 (\|e_{\rm target}\|_2+\|e_{\rm competitor}\|_2).
\]

The raw tied embedding norms are necessary; final RMSNorm does not bound
them. This certificate requires a separately established final residual
error E. It does not silently treat the local head estimate as that error.

### A6. Rounding certificates for the existing nonperiodic counting head

`Fx.NoSaturation p s x` checks that `floor(x * 2^s)` lies in the format's
mantissa interval. Under that condition, the existing floor-and-saturate
`Fx.round` has absolute error strictly smaller than `u = 2^-s`. There is
also a theorem that no finite uniform absolute error bound exists without
a range restriction, and a concrete saturated counterexample at p=8,s=0.

The source normalized logits give every unrounded exponential coefficient
a floor `a = exp(-sqrt(d))`. If `u < a` and every exponential coefficient
avoids saturation, the sum of the rounded coefficients is strictly positive
at every nonempty prefix. This rules out the counting model's zero-denominator
fallback under these assumptions, rather than assuming division is safe.

For a value coordinate bounded by B, if all exponential coefficients,
weighted numerator coefficients **and the final quotient** avoid saturation,
then the actual `CRASP.Rotary.sourceLayer` attention aggregate has error

\[
 \left|\widehat y-y\right|
 \le \frac{u(1+B)}{\exp(-\sqrt d)-u}+u.
\]

The comparison y is the exact real softmax average on the **same stored
residual stream**. The proof relates the implementation's integer-mantissa
quotient to the ratio of interpreted coefficients, bounds both sums, and
cancels their common prefix-length factor. The last u accounts for the
existing final quotient rounding. Concrete source-layer witnesses retain
the source epsilons and nonzero positions.

This remains the specified counting model: exact integer sums, rounded
coefficients and quotient, real local nonlinearities, and separately rounded
stored residuals. It is not IEEE round-to-nearest. In particular, fixed-point
u is not float32 machine epsilon. B1--B9 below propagate this local estimate
through the whole specified rounded residual stack. Hardware error bounds
for projections, RMSNorm, SwiGLU and RoPE table generation remain outside
these results. No new training or optimizer convergence guarantee follows.

| Lean module | Result |
| --- | --- |
| `SequenceModel.RotaryStability` | phase chord, half-split rotation, normalized logit and frequency-error bounds |
| `Attention.Stability.Softmax` | causal row, average and source XSA phase-error bounds |
| `Attention.Stability.Continuity`, `Model` | joint phase/residual continuity of the complete source |
| `Attention.Stability.Periodic` | genuine rational-pi periods and simultaneous finite logit approximation |
| `Attention.Stability.Classification` | finite-context same-depth classifier replacement and quantitative tied readout certificate |
| `CRASP.FixedError` | conditional quantum error, no-underflow criterion and saturation counterexample |
| `Attention.Stability.Quotient`, `Rounding` | length-independent coefficient error, exact source-aggregate connection and denominator positivity |

Validation of A1--A6: full `lake build` passes (13166 jobs), with no warnings
in the new modules. Individual `#print axioms` checks for all 41 new public
theorems use only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4069 modules, 21004 theorems, zero vacuous results
and zero placeholders. Forbidden checks and `git diff --check` pass; all
five pinned source hashes still match. Witnesses include nonzero phases,
nonzero rounding errors, saturation, source epsilon values, positive rounded
denominators and strict decisions on all four binary input pairs.

The required Python regression run passed all 418 lab tests with three skips.
The 104-test optimizer suite reproduced the two existing Schedule-Free Muon
allocation failures (float32 and float64), each with 75 allocations of
8519680 bytes. Schedule-Free AdamW tests pass. No Python implementation was
changed. All validation logs, source review and the theorem axiom-check
script are retained in `.kanban/artifacts/128/`.

## Full-stack rounding certificates

The earlier A6 result compared one rounded aggregate with a real aggregate
on the same stored residuals. B1--B9 close the propagation gap: they bound
the **actual `CRASP.Rotary.run`** against the complete real source `hidden`,
then transfer strict predictions to `CRASP.Rotary.predict` and its existing
uniform weighted-count formula. There is no assumed final residual error,
assumed block Lipschitz constant, or substituted activation.

The source remains diff_gpt `b98c665b`, with the five pinned files listed in
`review/sequence_model_source.json`. Both normalization epsilons are positive.
The exact source RoPE frequencies and per-layer enable flags are unchanged;
the block estimates also allow arbitrary identical phase tables on both
sides. Dimensions are arbitrary admissible finite dimensions. As before,
actual Python execution requires the source FFN size and positions within
`block_size`; the mathematical forward and counting language also admit
longer directly supplied phase tables.

### B1. Pre-normalization, projections and vector rounding

Write W for residual width, d for head dimension, H for the number of heads,
F for FFN width, epsilon for RMS epsilon, and xi for XSA's clipped-L2 epsilon.
For one block let Q, O, G, V and U be its fused-QKV, attention-output,
gate, value and FFN-output linear maps, respectively. All matrix norms below
are Euclidean operator norms. Define

\[
 L_R=2/\sqrt\epsilon,\qquad M=\|Q\|\sqrt W,\qquad C=\|Q\|L_R.
\]

Every actual pre-normalized Q/K/V head has norm at most M, and a residual
change E changes each such head by at most CE. This needs no bound on the
incoming residual itself. Keeping the same exact RoPE phases preserves the
norm of vector differences. Rounding W coordinates without clipping adds
at most `sqrt(W) u`, where `u=2^-s` is the fixed-point quantum.

### B2. Actual SwiGLU, with independent gate and value matrices

The derivative of sigmoid yields a global Lipschitz bound of one. From this
and `abs(silu(t)) <= abs(t)`, the actual SiLU product is bounded on the
pre-normalized input ball. A sufficient FFN constant is

\[
 L_F=\|U\|\sqrt F\left[
   \|V\|\sqrt W(1+\|G\|\sqrt W)\|G\|
   +\|G\|\sqrt W\|V\|\right].
\]

For inputs of norm at most `sqrt(W)`, the real `feedForward` changes by at
most `L_F` times their distance. The two FFN normalizers make its whole
residual map globally Lipschitz with constant

\[
 R=1+L_R L_F L_R.
\]

The proof uses the source SwiGLU formula. It does not reuse the old ReLU
block's constants or assume that the gate and value projections coincide.

### B3. Both the attention distribution and values change

If every input residual changes by at most E, the source Q/K logits change
by at most `2 L_R C E`. Causal softmax row sensitivity and the changing
value vectors therefore give

\[
 A=(16 L_R M+1)C,\qquad
 \|\operatorname{average}(x)-\operatorname{average}(y)\|\le AE.
\]

The reference average has norm at most M by convexity. These constants do
not depend on the number of keys. Unlike A2, this result changes Q, K and V
together as the residual stream moves.

### B4. XSA, all branch normalizers and both residual additions

Define the projection and moving-direction constants

\[
 P=L_R\|O\|\sqrt H,\qquad X=2M(2C/\xi).
\]

If the query residual error is E and each head-average error is at most B,
while the reference averages have norm at most M, the actual local source
update differs by at most

\[
 R\{E+P(2B+XE)\}.
\]

This includes movement of XSA's clipped own-value direction, its explicit
first-position zero, head merging, learned projection, all four branch
normalizers, both residual additions and SwiGLU. Substituting B=AE gives
the full source block constant

\[
 L_{\rm block}=R\{1+P(2A+X)\}.
\]

The bounds hold at the XSA clipping sphere; no differentiability premise
is needed. Small normalization epsilons can make these sufficient global
constants very large. The theorem does not assert that they predict typical
error amplification accurately.

### B5. Actual rounded prefixes and their vector error

One-based natural counting positions are explicitly reindexed to the
zero-based finite source tensor. Every causal prefix term, including the
query's own key/value, is retained. The real ratio in A6 becomes exactly
the coordinate of `Normalized.sourceAverage` on the decoded counting stream.

Let `a=exp(-sqrt(d))` and require `u<a`. Pre-normalization supplies M as the
value-coordinate bound, so the per-coordinate new aggregate error is

\[
 r=\frac{u(1+M)}{a-u}+u.
\]

The head-vector error is at most `sqrt(d) r`. `AggregateInRange` checks
`Fx.NoSaturation` for every actual exponential coefficient, weighted-value
coefficient and pre-rounding quotient. It assumes no error bound or output
correctness. The positive-denominator theorem of A6 applies to these same
premises.

### B6. One actual counted block against the real source

`StepInRange` additionally checks every coordinate of the actual local
update before stored-residual rounding. Combining B3--B5 proves

\[
 E_{\rm next}\le L_{\rm block} E+\rho,\qquad
 \rho=RP\,2\sqrt d\,r+\sqrt W\,u.
\]

The comparison is the existing `Rotary.sourceLayer.step`, with its integer
prefix sums and existing rounding sites, against the complete real source
`block`. It is not a newly defined approximate forward pass. Both the
changed input stream and the new rounding errors are included.

### B7. Full stack with each layer's own weights

`errorBudget` starts from `E_0=sqrt(W) u` and recursively applies B6 using
the actual independently learned block at each layer:

\[
 E_{l+1}=L_l E_l+\rho_l,\qquad 0\le l<D.
\]

`TraceInRange` checks the normalized input and each executed layer's actual
rounding sites, on the actual input word. `stack_error` proves, for every
executed layer n and every source position i,

\[
 \|\operatorname{decode}(\operatorname{counted}_n(i))
       -\operatorname{hidden}_n(i)\|\le E_n.
\]

The normalized embedding, first-token flag, exact source RoPE phases and
entire residual recurrence are connected to the actual implementations.
The constants and recurrence explicitly expose width, head count, FFN
width, depth, learned matrix norms and fractional precision. Total bit
width p controls the range premises. No finite-context cardinality factor
appears in the numerical bound, although whether the range and margin
conditions hold can depend on the input and its length.

### B8. Strict classification and the uniform counting formula

For a target and each competitor v, a sufficient real-source margin is

\[
 L_R E_D(\|e_{\rm target}\|+\|e_v\|)
 <\operatorname{logit}_{\rm target}-\operatorname{logit}_v.
\]

`rounded_predict_true` derives the actual rounded model's strict prediction
from this margin, positive epsilons, `u<a` and `TraceInRange`. `E_D` here is
the computed B7 recurrence, rather than the unproved residual-error premise
of A5. Final RMSNorm and the raw tied embedding norms remain explicit.

`uniform_source_certificate` chooses one formula of counting depth at
most D, with the same exact nonperiodic kernels, before any input word.
It preserves every source decision on every finite word that satisfies
the numerical certificate. This is conditional language agreement with a
uniform formula. It supplies no strict depth lower bound for those kernels,
and does not turn a finite-context periodic replacement into a uniform one.

### B9. Nonconstant concrete witness and limits

The witness has W=4, d=4, H=1, F=11, D=2 and three distinct unit embeddings.
Both residual blocks are identities, and source RoPE is enabled. A synthetic
RMS epsilon 3/4 makes those embeddings fixed points of normalization;
XSA epsilon is one and the fixed-point format is p=8,s=5. These example
epsilons are explicitly different from the source float32 defaults.

Every rounding-range premise holds on every finite three-letter word.
The computed final budget is `3/16`; every correct logit margin is one,
and its sufficient threshold is `sqrt(3)/2 < 1`. The classification theorem
therefore proves that the actual rounded model predicts the token at each
position, for every word. Predictions vary across all three classes. The
actual residual error in this witness is zero: the nonzero computed budget
demonstrates sufficiency and is not claimed sharp. Separate B1 examples
exercise nonzero coordinate rounding, and B2--B4 examples admit arbitrary
learned block matrices at the actual source normalization epsilons.

| Lean module | Result |
| --- | --- |
| `Attention.Stability.Vector`, `Gates` | projection/vector rounding and actual SwiGLU constants |
| `Attention.Stability.Averages`, `Local` | moving logits, values and XSA; full source block sensitivity |
| `Attention.Stability.Aggregate` | actual natural prefix/finite tensor connection and rounded head error |
| `Attention.Stability.Stack` | actual step and full residual-stack error budgets |
| `Attention.Stability.Certificate` | strict prediction and uniform weighted-count formula transfer |
| `Attention.Stability.Witness`, `Certificate` | complete nonconstant witnesses for range and margin hypotheses |

The closed gap is propagation through the specified fixed-point counting
model. Local operations still use exact reals, integer sums are exact, and
the final readout is real. Float32 projections, nonlinearities, reductions,
overflow/underflow, quantized weights, and generated cosine/sine tables need
their own execution-error analysis. A strict hierarchy for source-generated
nonperiodic kernels and any optimizer/generalization guarantee also remain
unproved. No training experiment or Python implementation changes in B1--B9.

Validation of B1--B9: full `lake build` passes (13174 jobs), with no warnings
in the new modules. All 46 new named theorems pass individual `#print axioms`
checks using only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4077 modules, 21050 theorems, zero vacuous results
and zero placeholders. Forbidden checks and `git diff --check` pass, and
all five pinned source hashes match after rereading the actual forward.

The required Python run passes all 418 lab tests with three skips. Its
104-test optimizer suite reproduces the two previously recorded Muon
allocation failures (float32 and float64, 75 allocations of 8519680 bytes
each); Schedule-Free AdamW passes. These are unchanged from A1--A6's
validation. Logs, the individual theorem axiom-check script and source
review are retained in `.kanban/artifacts/129/`.


## Primitive execution certificates

C1--C8 extend B1--B9 from exact local operations in a fixed-point counting
model to a finite tensor trace with explicit operation errors. The reference
is still the pinned `diff_gpt` no-cache forward at `b98c665b`. The conclusion
is conditional on those local errors; it is not a proof that an arbitrary
PyTorch/CUDA backend satisfies them. All widths, block matrices, finite
depths, input words and reference phase tables remain parameters.

### C1. Finite binary32 and explicit arithmetic order

`binary32Value` decodes a sign, exponent field `e<255` and fraction field
`m<2^23`; `binary32_value_mem` proves membership in the existing finite
`Precision.grid 8 23`, including subnormals. For a nearest map Q and two
represented endpoints bracketing x, `IsNearest.error_of_bracket` proves

\[
 |Q(x)-x|\le (b-a)/2.
\]

`binary32_bracket_error` instantiates this with actual binary32 patterns.
Inside a positive normal binade, consecutive fraction fields give absolute
error at most `2^(e-150)/2`, and `binary32_normal_relative_error` gives
`2^-24 |x|`. The normal theorem requires `0<e<255` and `m+1<2^23`;
it does not silently include binade boundaries. Positive subnormal
intervals instead give the absolute bound `2^-149/2`, including intervals
next to zero. Negative values and boundary-crossing intervals can use the
general represented-endpoint theorem. There is no universal relative-error
claim near zero, or an overflow/NaN/flush-to-zero emulator.

`roundedDot mulRound addRound a b n` specifies the sequential recurrence
`z[0]=0`, `z[k+1]=addRound(z[k]+mulRound(a[k]*b[k]))`. The proved error is
at most the sum of all local product and accumulator errors, each evaluated
at the actual rounded partial sum. Uniform bounds P and A give `n(P+A)`.
Represented brackets discharge those premises. Identity product rounding
with nearest accumulator rounding models one correctly rounded FMA per
step; two nearest maps model separate multiply/add. No selected GPU kernel
is asserted to use this order or these formats. `binary32_quarter_ulp`
proves that nearest binary32 rounding of `1+2^-25` is one, giving an actual
nonzero rounding witness.

### C2. Stored RoPE tables need not be exact rotations

`tableRotate` retains the source half-split arithmetic with independently
stored cosine and sine entries. Given coordinate errors at most tau from
`cos(a)` and `sin(a)`, `tableRotate_error` proves

\[
 \|\widehat R x-R_a x\|\le 2\tau\|x\|.
\]

It assumes neither a unit-circle identity for the stored entries nor that
they represent any exact phase. Thus frequency/position computation, table
generation and table-storage errors can be included in tau. Arithmetic
applying those entries is checked separately. An explicit example uses
`c=1+1/1024`, `s=1/1024` at reference phase zero. Odd leftover coordinates
are unchanged in the mathematical helper; the source configuration has
even head width. Disabled RoPE uses the identity phase/table and exact copy.

### C3. Finite traces and local contracts

`AttentionTrace`, `FFNTrace`, `BlockTrace` and `ModelTrace` contain numerical
data only. `AttentionAccurate`, `FFNAccurate`, `BlockAccurate` and
`ModelAccurate` are predicates checking the recorded operations against
their own recorded inputs. They do not assume a final state, block or
prediction error. A common nonnegative delta bounds Euclidean vector errors
and absolute scalar errors; tau bounds table entries independently.

The checks include input RMSNorm, fused QKV, head Q/K RMSNorm, stored table
application, scaled scores, causal weighted averaging, clipped L2
normalization, XSA and the explicit first-position zero, head merge/output
projection, branch RMSNorm, gate/value projections, SiLU, multiplication,
FFN projection, both residual additions, embedding-table error, final
RMSNorm and tied logit dot products. Reshape and slices are exact indexing.
The fused FFN matrix's two halves are represented by their independent
gate/value maps, preserving its exact mathematical function.

`roundedVector_error` turns coordinate brackets of half-width at most eta
into Euclidean error `sqrt(d)*eta`. `roundedMatrix_error` turns explicit
rowwise product/add bounds into `sqrt(m)*n*(P+A)` for an m-by-n reduction.
These supply operation-contract bounds when their rounding/order premises
hold. Weight quantization can be included in the matrix operation error
relative to the reference matrix; the shared embedding-table perturbation
is represented separately and used at both ends of the model.

RMSNorm, L2 normalization and SiLU accuracy remain primitive implementation
contracts. In particular, the causal average check is an explicit SDPA
kernel contract, covering softmax and the weighted reduction at the
recorded scores and values. A fused implementation may use a logical score
trace instead of materializing that tensor. This is not yet a scalar IEEE
proof of its exponential, reciprocal, scaling, summation or reduction
order. Finite real traces also require excluding backend nonfinite results.
A single delta is convenient notation, not a backend-independent error
constant; it may depend on width, context, input range and kernel choice.

### C4. Attention and the approximately normalized XSA direction

Write `L_R=2/sqrt(eps)`, `M=||Q||sqrt(W)` and let H be the head count,
d the head width. `attentionBudget` derives, in execution order,

\[
\begin{aligned}
 e_Q&=\|Q\|\delta+\delta,&
 e_N&=L_R e_Q+\delta,\\
 e_R&=2\tau(\sqrt d+e_N)+e_N+\delta,&
 e_S&=(2\sqrt d+e_R)e_R/\sqrt d+\delta,\\
 e_A&=8e_S M+e_Q+\delta,&
 e_U&=(2/\epsilon_X)e_Q+\delta,\\
 e_H&=[1+(1+e_U)^2]e_A+M(2+e_U)e_U+\delta,\\
 e_P&=\|O\|\sqrt H\,e_H+\delta,&
 e_{\rm attn}&=L_R e_P+\delta.
\end{aligned}
\]

`attention_scores_error` includes the bilinear cross term from perturbing
both Q and K. `attention_error` derives the complete branch bound from
the local checks. The exact clipped direction has norm at most one, but
the computed direction can have norm as large as `1+e_U`.
`xsa_pair_error` proves the corresponding bound, instead of incorrectly
reusing the earlier theorem requiring both directions to have norm at
most one. Its witness has a computed direction of norm two.

### C5. SwiGLU including activation and product execution

Set `e_g=||G||delta+delta`, `e_v=||V||delta+delta`,
`R_g=||G||sqrt(W)` and `R_v=||V||sqrt(W)+e_v`. Then `ffnBudget` gives

\[
 e_{\rm product}=\sqrt F\,[\delta+\delta R_v+
                   R_v(1+R_g)e_g+R_g e_v],\qquad
 e_{\rm ffn}=L_R(\|U\|e_{\rm product}+\delta)+\delta.
\]

`ffn_error` derives this from the actual independent gate and value
projections, the SiLU error, multiplication error, output projection and
post-normalizer. The approximate pre-normalized input need not satisfy
the exact source norm bound; its error is propagated through both matrices.
Examples admit arbitrary block weights at the source float32 epsilons.

### C6. Both residuals and the complete depth recurrence

With the actual B3 FFN residual gain `f` and B4 full-block Lipschitz
constant `L`, the new block radius is

\[
 r=f(e_{\rm attn}+\delta)+e_{\rm ffn}+\delta.
\]

`block_error` and `block_movement_error` prove the same-input and moving-input
bounds. `model_error` derives the complete layer recurrence

\[
 E_0=L_R\delta+\delta,\qquad E_{\ell+1}=L_\ell E_\ell+r_\ell.
\]

The base includes the shared embedding-table perturbation and input
normalizer; each independently weighted block supplies its own constants.
The result bounds every recorded residual state at every depth up to D.
Width affects matrix norms, head/FFN sizes and local arithmetic bounds;
depth affects repeated amplification and accumulated operation error.
This is a sufficient stability bound, not a sharp width/depth lower bound.

### C7. Executed logits and the uniform counting bridge

For raw reference class embedding e, `logitRadius` is

\[
 r_e=(L_R E_D+\delta)\|e\|+(\sqrt W+\delta)\delta+\delta.
\]

`logit_error` proves this bound for the recorded logit, including final
RMSNorm error, the perturbed tied embedding and the executed dot product.
`predict_true` proves a strict computed target prediction when each exact
source target-versus-competitor margin exceeds `r_target+r_competitor`.
Neither an assumed hidden-state error nor an assumed final-logit error
appears among the primitive contracts.

`uniform_execution_certificate` chooses the existing nonperiodic RoPE
counting formula before any input word, with counting depth at most D.
When both B8's fixed-point range/precision/margin certificate and C7's
execution certificate hold against the same exact source, the formula
accepts the target and the executed trace predicts it. These are explicit
conditional target agreements for all finite words. This does not assert
unrestricted equivalence for arbitrary hardware, a context-independent
floating-point error budget, or a strict hierarchy for exact RoPE kernels.

### C8. Witnesses, remaining gap and validation

Every new theorem with hypotheses has a concrete Lean `example`. Exact
source traces witness all operation contracts for arbitrary weights and
phases. `shiftedModel` additionally perturbs each final dot product by
`delta`; `shiftedModel_accurate` proves every local check, and
`shiftedModel_logit_error` proves that its actual logit error equals delta.
This synthetic trace is not attributed to a CUDA implementation.

For B9's W=4, H=1, F=11, D=2, three unit embeddings and zero block matrices,
use `eps=3/4`, `epsilon_X=1`, `delta=1/1024`, `tau=0`, with source RoPE
active. The complete derived budget is `(5 L_R+9)/1024`; each logit bound
is below 1/2, while each correct source margin is one. Thus
`example_executed_prediction` certifies the nonconstant three-class
classifier on every finite word despite its nonzero logit errors.
`example_uniform_execution` also instantiates the counting bridge with
p=8,s=5. The synthetic normalization epsilons are not the source defaults.

| Lean module | Proved connection |
| --- | --- |
| `Precision.Binary32Error`, `ArithmeticError` | finite represented brackets, normal/subnormal bounds, explicit multiply/add or FMA accumulation |
| `Attention.Execution.Tables`, `Rounding` | independent stored RoPE coefficients and scalar-to-tensor operation bounds |
| `Attention.Execution.Trace`, `Basic`, `Budget` | concrete numerical traces, local predicates and derived constants |
| `Attention.Execution.Scores`, `Attention`, `FFN` | complete branch errors, including non-unit computed XSA direction |
| `Attention.Execution.Block`, `Model`, `Readout` | both residuals, arbitrary finite depth, tied logits and strict prediction |
| `Attention.Execution.Witness`, `Certificate` | nonzero-error classifier and conditional uniform exact-RoPE counting transfer |

The closed gap is the propagation of local execution errors, including
independent stored RoPE tables, through the full normalized source stack.
Connecting a selected PyTorch/CUDA kernel to the local contracts remains:
its dtypes and reduction order, transcendental/normalization accuracy,
SDPA error, table-generation bounds and finite-range behavior must still
be established. The scalar/vector adapters are reusable pieces of that
connection, not a completed backend proof. A strict exact-RoPE depth
hierarchy and optimizer/generalization guarantees remain unproved. No
training experiments or Python implementations change in C1--C8.


Validation of C1--C8: full `lake build` passes (13190 jobs); new modules
have no warnings. All 46 new named theorems pass individual `#print axioms`
checks using only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 pre-existing `sorry`, zero dependent results and zero extra
axioms. The regenerated index has 4093 modules, 21096 theorems, zero vacuous
results and zero placeholders. Forbidden checks and `git diff --check`
pass, and all five pinned source hashes match after reviewing the forward.

All 418 lab tests pass with three skips. The 104-test optimizer suite
reproduces the two existing ScheduleFreeMuon allocation failures, float32
and float64, each with 75 allocations of 8519680 bytes. Schedule-Free AdamW
passes. These failures match B1--B9's baseline. Build/test logs, individual
axiom checks and the source review are retained in `.kanban/artifacts/130/`.

## Unified spectral transformer certificate

`Attention.Spectral.exists_power_certificate` connects feature recovery,
the actual normalized source transformer and exact-RoPE counting logic in
one theorem. It constructs weights from the recovered real score table and
proves a source margin. Under explicit numerical contracts, every class
decision agrees with the task and one fixed family of counting formulae.
The task is (a/b)^K for every prime P, natural K, numerator a and nonzero b.
This is finite representation and conditional execution; the ideal feature
optimization premise is separate from training with Schedule-Free AdamW.
The counting language is the existing exact-RoPE extension, not the periodic
hierarchy.

### J1. Bounded vector outputs in the actual source

P5's prescribed-RoPE construction extends from unit answer vectors to any
finite table t(x) with norm at most one. The actual state after attention is
base + context, where base has norm one and the normalized context has norm
less than one. Thus

\[
 \|t(x)-(\mathrm{base}+\mathrm{context})\|<3<\sqrt W
 \quad\text{when }W>9.
\]

The existing inverse of output RMSNorm reaches the residual displacement.
The source RoPE encoder distinguishes the whole input word projectively,
and the actual bias-free SwiGLU interpolates this finite vector table.
`exists_bounded_block` and `exists_bounded_model` retain attention, Q/K
normalization, XSA, both Peri-RMSNorm branches, embedding normalization and
tied final readout. The original final input token is the readout position.
Extra layers are identity residual blocks; they do not prove an advantage
from active depth.

### J2. Preserve every recovered score

For any real score table S on V^(N+1) by V, let e_v be orthonormal token
coordinates, A the positive raw embedding amplitude, and

\[
 M=1+\sum_x\sum_v |S(x,v)|,\qquad
 t(x)=M^{-1}\sum_v S(x,v)e_v.
\]

The triangle inequality bounds norm t(x) by one. J1 realizes t(x) exactly
before final RMSNorm. Since the tied raw embedding is A e_v, every actual
source logit satisfies

\[
 L_v(x)=g(x)S(x,v),\qquad
 g(x)=\frac{A\sqrt W}{M\sqrt{\|t(x)\|^2+W\,\mathrm{eps}}}
 \ \ge\ g_0=\frac{A\sqrt W}{M\sqrt{1+W\,\mathrm{eps}}}>0.
\]

`exists_score_model` proves this for the complete table; `minimumGain` is
g_0. A spectral score gap m transfers to a source gap at least g_0 m. The
construction uses S, not an assumed table of correct predictions. Positive
row scaling preserves ordering, but need not preserve softmax probabilities.

### J3. Supply the scores by spectral recovery

Let chi_r be a finite bank of multiplicative characters on F_P, with
nonnegative weights w_r summing to one. Require joint separation on the
unit group: each distinct a,b has a positively weighted character with
chi_r(a) != chi_r(b). Individual characters need not each separate answers.
The true feature kernel is

\[
 K_r(a,b)=\chi_r((a/b)^K),\qquad a,b\in\mathbb F_P^\times.
\]

`Fits` requires agreement on an observation pattern Omega. `Minimizes`
requires a global minimum of the sum of coordinate nuclear norms under
those same feature observations. If the bipartite observation graph is
connected and contains a perfect matching, `recovered_bank` proves X_r=K_r
on every pair, including pairs outside Omega. This composes the existing
unique kernel and bank recovery theorems. `recovered_unit_margin` gives a
uniform positive margin for the recovered bank's weighted character readout.

`true_bank_minimizes` supplies witnesses to the ideal premise. A faithful
character exists for every prime, so a singleton bank with weight one
satisfies separation; larger jointly separating banks are also admitted.
Full observation gives explicit witnesses in `RecoveryWitness`; the general
theorem permits any pattern satisfying the two graph conditions. Nothing
here identifies transformer cross-entropy with feature nuclear norm or
asserts that Schedule-Free AdamW finds its minimum.

### J4. Original residues, zero numerators and exponent zero

The source vocabulary is all of F_P, not just its units. On nonzero input
pairs, `powerScores` reads the recovered coordinates and takes their fixed
character-code score for each nonzero candidate; candidate zero receives
score -1. The true unit answer scores one. For a=0, the scores are +1 at
the elementary boundary answer and -1 elsewhere: the boundary answer is
one when K=0 and zero when K>0, using the natural-power convention 0^0=1.

If the unit-bank margin is m>0, the full residue margin is at least
min(1,m)>0. Denominator-zero rows are totalized for interpolation and
excluded from every powered-division correctness claim. No answer oracle
is used on nonzero pairs. `powerLabel_residue` and `tokenUnit_power` prove
that the unit and original-token labels describe the same field computation.

### J5. Finite resources and actual recovered-score computation

`exists_recovered_power_model` composes J2--J4. For two original operands,
its sufficient source family has

\[
 V=P,\quad H=1,\quad W=2(P+1)^2+6P^2+2\,\mathrm{padding},\quad
 F=\lfloor(8W+1)/3\rfloor,\quad D=\mathrm{extraDepth}+1.
\]

It uses the geometric integer-position RoPE grid at the configured source
base, enabled at every layer, and context length two within the configured
block size. It permits eps>0 and 0<xeps<=1. Examples use the source float32
RMS epsilon 2^-23 and XSA epsilon 10^-12. A prime vocabulary can use the
source constructor with `pad_vocab_size_to=1`.

For a general fixed length N+1, `exists_finite_score_model` replaces 6P^2
by 6P^(N+1). This is finite interpolation of recovered scores, not efficient
character arithmetic inside the transformer. These sufficient bounds are
not minimal. Additional depth is identity padding; neither a sharp
width/depth tradeoff nor useful weight-conditioning bounds follow.

### J6. A positive execution allowance exists

Fix the constructed finite weights and positive margin gamma=g_0 m. C6's
recurrence E_n(delta,tau) is continuous along delta=tau=eta and vanishes
at eta=0 for every finite n. The same holds for C7's logit radii. All
budgets and radii are nonnegative for nonnegative local errors. The finite
sum of twice each class radius bounds every pair of class radii. Continuity
therefore yields eta>0 such that

\[
 r_u(\eta,\eta)+r_v(\eta,\eta)<\gamma
 \quad\text{for all classes }u,v.
\]

`exists_execution_tolerance` proves this for any fixed finite model with
positive margin; it does not assume a small final-state error.
`exists_robust_power_source` instantiates all spectral premises for every
prime and K and applies the allowance to C8's synthetic shifted trace.
Its actual logit error equals eta>0, all primitive contracts hold, and
every class decision is correct. Examples include P=3,K=0 and P=97,K=5.
This gives no useful lower bound on eta and does not assert that a selected
hardware format or kernel satisfies the allowance.

### J7. Fixed counting formulae and complete decision agreement

For each format (bits,frac), choose the existing RoPE-aware formula phi_v
for each class before the input word. Each has counting depth at most D
and represents the rounded source classifier. `uniform_margin_certificate`
strengthens the earlier positive-target connection to every class, including
rejection of competitors. `exists_power_certificate` composes it with J5:

\[
 \varphi_v(x)=\widehat{\mathrm{predict}}_v(x)
 =\mathbf 1\{v=(a/b)^K\}.
\]

The conditions remain explicit: positive normalization epsilons;
2^-frac < exp(-sqrt(head_dim)); the rounded trace's `TraceInRange`; its
derived budget below gamma (`CountingBelow`); the executed trace's local
`ModelAccurate` contracts; and its derived pairwise radii below gamma
(`ExecutionBelow`). Both numerical executions are compared through the
same exact source. A format meeting all counting conditions for the
compiled powered-division weights is not constructed here. The formulae
are uniform over words; this task guarantee is for two-token inputs, not
an unbounded arithmetic language or a strict exact-RoPE depth hierarchy.

`example_uniform_decisions` discharges every numerical premise for the
existing nonconstant three-class source: W=4,D=2,eps=3/4,xeps=1,bits=8,frac=5,
delta=1/1024,tau=0. All class decisions agree on equal two-token words
despite nonzero executed logit errors. This witnesses the joint certificate
independently of any unverified backend contract.

| Lean module under `Attention.Spectral` | Proved connection |
| --- | --- |
| `Realization`, `Geometry`, `Scores` | bounded vector interpolation and all positively scaled source logits |
| `Recovery`, `Residues`, `PoweredScores`, `Powered` | ideal bank recovery, zero boundary and source margin |
| `Numerics`, `Decisions`, `Counting` | positive error allowance and every-class numerical/counting agreement |
| `Certificate`, `Existence` | composed theorem and all-prime perturbed execution witnesses |
| `RecoveryWitness`, `CountingWitness` | concrete spectral and numerical premise checks |

Board task #131 covers this composition. Tasks #132--#134 separately track
concrete backend operation contracts, Schedule-Free AdamW reaching the
spectral conditions, and task-specific depth/width lower bounds for exact
nonperiodic RoPE. Those remain questions, not corollaries of this certificate.
No training experiment or Python implementation changes in J1--J7.

Validation of J1--J7: full `lake build` passes (13205 jobs), with no warnings
in the new modules. Individual `#print axioms` checks for all 45 new public
theorems use only `propext`, `Classical.choice` and `Quot.sound`. The global
audit remains at 36 pre-existing `sorry`, zero dependent results and zero
extra axioms. The regenerated index has 4108 modules and 21143 theorems,
with zero vacuous results and zero placeholders. Forbidden checks and
`git diff --check` pass; all five pinned source hashes still match after
reviewing the forward and every new statement against J1--J7.

All 418 lab tests pass with three skips. The 104-test optimizer suite
reproduces C1--C8's two ScheduleFreeMuon allocation failures (float32 and
float64, 75 allocations of 8519680 bytes each). Schedule-Free AdamW passes;
the overall Python command exits unsuccessfully because of those existing
Muon failures. Logs, axiom checks and the source crosswalk are retained
in `.kanban/artifacts/131/`.

## Concrete backend contracts

K1--K6 replace several opaque primitive-error assumptions with algebraic
and polynomial checks and prove those checks for actual captured PyTorch
outputs. This is a finite backend conformance result: the captured CPU and
CUDA numbers have ordinary Lean proofs of their local error bounds. It
does not establish `ModelAccurate` for a complete captured transformer,
every checkpoint, or every future execution of these kernels. That
composition is addressed by task #135 in L1--L5 below.

### K1. Certify normalization scales without trusting sqrt

For a nonnegative scalar t,

\[
 |t-1|\le |t^2-1|.
\]

Let x be an operation's actual input, z its stored output, b>0 its exact
normalizing denominator, and a>=0 a rational certificate scale. Suppose

\[
 |a^2b^2-1|\le\rho,\qquad
 \|z-ax\|\le\beta,\qquad \|x/b\|\le B.
\]

`normalization_residual` proves

\[
 \|z-x/b\|\le\beta+\rho B.
\]

The proof factors the scale error through the normalized vector. The scale
a is a certificate witness; it need not be the hidden intermediate used
by a fused kernel. The product residual is checked at the stored input and
output. No sqrt/rsqrt accuracy claim about a library or device is assumed.

### K2. Source RMSNorm and clipped L2

For RMSNorm at width d and eps>0,

\[
 b^2=(\|x\|^2+d\,\mathrm{eps})/d,\qquad B=\sqrt d.
\]

Thus `rms_residual_error` checks the polynomial residual
`abs(a^2*(norm(x)^2+d*eps)/d-1)` and the observed multiplication residual.
Its local error bound is beta+rho*sqrt(d), retaining the actual epsilon.
It does not multiply this local residual by inverse sqrt(eps). This is a
local execution certificate, not a replacement for the moving-input
Lipschitz bound in C6.

For clipped L2, b=max(norm(x),xeps)>0 and B=1. `l2_residual_error` gives
beta+rho, keeping the clipping boundary. On a known side of the boundary,
b^2 is either the exact sum of coordinate squares or xeps^2, so the scale
check is rational for rational stored inputs. For any width, a coordinate
product-residual bound eta gives beta=sqrt(d)*eta by B1.

### K3. Exponential and SiLU certificates

Define the finite polynomial P_n(x)=sum_{m<n} x^m/m!. For abs(x)<=1 and
n>0, Mathlib's proved exponential-series remainder yields

\[
 |\exp x-P_n(x)|\le |x|^n\frac{n+1}{n!\,n}.
\]

`exponential_error` adds a checked rational residual between a witness e
and P_n(x). For SiLU, check e>=0, an exponential certificate
abs(e-exp(-x))<=E, and the rational division residual
abs(z-x/(1+e))<=beta. `silu_residual_error` gives

\[
 |z-\mathrm{SiLU}(x)|\le\beta+|x|E.
\]

`silu_polynomial_error` composes these checks, so neither exp nor SiLU
accuracy is assumed. This polynomial certificate has an explicit input
range and order; it is not a claim about the accuracy of an arbitrary
CUDA exponential over the whole real line.

### K4. Source causal SDPA from checked masses and a reduction residual

Let a_j be approximate masses with A=sum a_j>0 and b_j the exact
nonnegative masses with B=sum b_j>0. If sum abs(a_j-b_j)<=E,
`normalized_weights_error` proves

\[
 \sum_j\left|a_j/A-b_j/B\right|\le 2E/A.
\]

For stored values with norm at most M, a checked residual beta between
the stored output and the rational weighted sum sum_j (a_j/A)v_j gives
`average_residual_error`:

\[
 \left\|z-\sum_j(b_j/B)v_j\right\|\le\beta+(2E/A)M.
\]

`causalMass` retains exactly the original visible-prefix mask, including
the current token. `causalMass_weight` identifies its normalized masses
with the existing source `causalWeight`. `causal_average_residual` therefore
fills the SDPA-average field of `Execution.AttentionAccurate` directly.
Polynomial witnesses supply mass errors; the reduction residual compares
the actual stored output with an exact rational sum. A fused kernel need
not expose its hidden softmax weights or use the sequential reduction
order of C3's different adapter. These checks still have to be discharged
at each operation in a complete model trace.

### K5. Stored RoPE coefficients

The complex exponential polynomial at ix supplies a common real/imaginary
reference. If abs(x)/(n+1)<=1/2, its proved norm remainder is

\[
 \left\|e^{ix}-\sum_{m<n}(ix)^m/m!\right\|
 \le 2|x|^n/n!.
\]

`rotation_polynomial_error` transfers this to cosine and sine separately.
`stored_rotation_error` adds the rational residual of each stored table
entry. It assumes neither unit-circle identities for those entries nor
correctness of a second floating-point trigonometric implementation.
`head4_phases` proves that the actual base-10000 source frequencies at head
width four are 1 and 1/100. The captured position-one checks therefore
include frequency computation and table generation against exact source
phases, not against reused rounded tables.

### K6. Identified executions and exact-value Lean replay

`./run.py backend-contracts` captures seven fixed operation sites using
Python 3.15.2, PyTorch 2.15.0+cu132 and CUDA 13.2 on the RTX 3050 Laptop GPU:
CPU float32, CPU float64, and CUDA float32 compiled with
`max-autotune/fullgraph/static`. SDPA explicitly selects the MATH backend;
matrix float32 precision is IEEE, with TF32 disabled. Kernel event names
are retained in `review/backend_contract_check.json`. The CUDA run records
fused Triton RMS/L2, softmax, RoPE and SiLU kernels. The GPU reports too few
SMs for GEMM max-autotuning, so the requested compile mode uses its fallback
there. Reduction order remains backend-managed and is not assumed by the
output-residual certificates.

The RMS and L2 input is [1,2,3,4]. All three cases use RMS eps=2^-23 and
L2 eps=10^-12; in particular, the float64 comparison intentionally uses
the same RMS epsilon rather than the source's float64 default. SiLU uses
input 1/2. The last row of the two-token causal SDPA has actual scaled
scores [0,1/2] and distinct unit value vectors. RoPE uses the source
float32 table-construction dtype, even in the float64 case. The matrix
fixture is a nonzero 2-by-4 dyadic product.

`backend_contract_arithmetic.py` decodes every finite recorded binary
value exactly with rational arithmetic. The normalizer scales, order-12
polynomials, coordinate boxes of radius 10^-6 and exact matrix result are
fixed independently of those outputs. It rejects nonfinite values, changed
operation sets, wrong shapes and any violated check. All 21 site checks
pass. The rational exponential witness at -1/2 is
49583642701/81749606400; its analytic error is proved below 10^-10.

`./run.py backend-certificates` exports the actual binary values as rational
literals to `CapturedCPU32`, `CapturedCPU64` and `CapturedCUDA32`. Their
18 named theorems prove the following bounds with ordinary Lean tactics:

| Captured operation | Proved error bound |
| --- | --- |
| RMSNorm | Euclidean error <= 4e-6 |
| Clipped L2 | Euclidean error <= 3e-6 |
| SiLU at 1/2 | absolute error <= 2e-6 |
| Two-token causal SDPA | Euclidean error <= 3e-6 |
| Each source RoPE cosine/sine | absolute error <= 2e-6 |
| Dyadic matrix product | exact equality |

The tests also require the exported Lean literals to equal the frozen
capture, check capture/checker hashes and source hashes, and reject corrupted
coordinates, invalid layouts, missing sites and nonfinite results. The
analytic consequences are proved in Lean rather than inferred from
agreement with a higher-precision floating-point calculation.

Task #132 supplies these concrete primitive certificates. Task #135 below
collects and discharges them throughout one complete, nontrivial transformer
trace and checks the derived model budget against its margin. The fixed
probes above do not prove that a whole model satisfies
`ModelAccurate`, that the bound is useful for the constructed powered-division
weights, or that every future invocation of these kernels is accurate.
Learning conditions (#133) and exact-RoPE depth/width lower bounds (#134)
remain separate. No training or optimizer implementation changes here.

Validation: the full Lean build passes (13217 jobs). Individual axiom checks
cover all 42 new public theorems and use only propext, Classical.choice and
Quot.sound. The global audit reports 36 unchanged sorry, zero results resting
on a sorry and zero extra axioms. The regenerated index has 4120 modules and
21185 theorems, with zero vacuous results and zero placeholders; forbidden
and whitespace checks pass. All new modules build without warnings.

The 424-test lab suite passes with three skips. A final seven-test certificate
run also passes, including the export-consistency test added after full-suite
discovery. The 104-test optimizer suite reproduces the two existing
ScheduleFreeMuon allocation failures (float32/float64, 75 allocations of
8519680 bytes each); Schedule-Free AdamW passes. Consequently the overall
Python command exits unsuccessfully, matching the baseline recorded for
#130 and #131. Logs and the statement/source crosswalk are retained in
`.kanban/artifacts/132/`.

## Complete captured execution

L1--L5 discharge every operation premise for one complete compiled CUDA
invocation of the pinned source transformer. They also prove a limitation
of the current global error budget: it cannot certify this invocation's
decision margin, for any valid uniform operation allowance. This is a
proved obstruction in the sufficient numerical bound, not evidence that
the model computes the wrong decision. Task #135 records this result;
task #136 tracks a replacement bound using the observed norm ranges and
individual operation errors.

### L1. Capture the actual source forward

`./run.py model-trace-capture` imports the unchanged diff_gpt `GPT.forward`
at commit b98c665bd8faa30729fd368d257a13b6d58712a1, checking all five pinned
file hashes first. It uses width four, one head, one block, the source
SwiGLU width eleven, three tied token embeddings, context length two and
the input word [0,1]. The stored weights are deterministic dyadic values;
all five branch matrices are nonzero. Both attention and SwiGLU change the
second residual, as proved by `active_branches`. This is a finite execution
certificate, not a training run or a modular-division accuracy experiment.

The invocation uses Python 3.15.2, PyTorch 2.15.0+cu132, CUDA 13.2 and the
RTX 3050 Laptop GPU. Compilation requests max-autotune, fullgraph and static
shapes. SDPA selects MATH; float32 matrix precision is IEEE. RMS epsilon
is the source float32 value 2^-23, and clipped L2 retains epsilon 10^-12.
The original source constructs its RoPE tables on CPU in float32 before
the model moves to CUDA. Their exact mathematical phases are checked
independently, including the source base-10000 frequency calculation.

Observational hooks and functional wrappers retain 24 tensors from the
actual source call. The wrappers return each original operation's result.
The trace includes all RMS calls, the fused QKV projection, source reshapes,
Q/K rotations, SDPA output, clipped own-value direction, first-position
zero, head merge, both learned output projections, all SwiGLU operations,
both residual additions, final RMSNorm and tied logits. The shared embedding
and cosine/sine tables are also saved. Auxiliary scores are explicitly
computed from the stored Q/K values; they are not claimed to expose the
SDPA kernel's internal scores. The average certificate checks its output
against those auxiliary scores and the captured values.

Returning intermediates can affect compiler fusion. The claim concerns
this instrumented compiled invocation. Its six final logit bits match the
separately compiled uninstrumented forward in this capture; that numerical
comparison is retained as evidence, not used as a Lean premise. Observed
kernel names are retained in `review/model_trace_capture.json`. The GPU's
GEMM autotuning fallback is permitted; no particular reduction order is
assumed by the output-residual checks.

### L2. Check all local premises with exact arithmetic

`model_trace_arithmetic.py` decodes every finite captured value as its exact
binary rational. It checks the complete operation/parameter/table sets,
shapes, source dimensions, input word and epsilons. It rejects nonfinite
values, Boolean or string substitutes, missing sites and corrupt outputs.
There are 52 checked vector/scalar sites, in addition to the exact stored
embedding identification.

Each normalization has its own rational scale witness. Decimal square roots
only propose scales; exact rational squared residuals and coordinate boxes
verify them. Therefore no Decimal sqrt correctness is assumed. Matrix
products, source rotations, scaled dot products, elementwise products, XSA,
residual additions and final logits are checked against their actual stored
inputs using rational sums. Source QKV/head/half-split layouts are retained.

Order-12 polynomial witnesses certify all 22 SiLU coordinates and both
causal average rows. On abs(x)<=1, `exp_checked` bounds the analytic
exponential error by 10^-8 after a rational polynomial residual at most
10^-10. `silu_checked` and `average_checked` transfer those certificates
to the source activation and causal average. The approximate mass sum is
checked above 1/2 and each stored value coordinate below one in magnitude.
The actual source input ranges, rather than a uniform CUDA exp guarantee,
discharge these requirements.

Most rational coordinate boxes have radius 10^-6; widths up to sixteen
convert them to the common Euclidean operation allowance delta=10^-4.
Normalizer and nonlinear analytic remainders are included before that
conversion. The independently checked cosine/sine allowance is tau=2e-6.
These convenient common allowances are upper bounds, not measurements of
each primitive's maximum error.

### L3. A complete ordinary Lean certificate

`./run.py model-trace-certificates` rechecks the rational premises, saves
`review/model_trace_certificates.json`, and exports exact rational literals
and ordinary Lean proof scripts. Lean independently proves all numeric
premises and their analytic consequences. The exporter introduces no
axioms, native decision procedures or unchecked Python-derived equalities.
Tests require its output to equal the committed Lean source exactly.

`Complete.captured` is a full `Execution.ModelTrace`, linked at every
operation boundary to its actual stored input. `attention_accurate`,
`ffn_accurate` and `block_accurate` assemble the checked primitive results.
`captured_accurate` proves

```
ModelAccurate cfg weights (1/8388608) (1/1000000000000)
  angles tokens captured (1/10000) (2/1000000).
```

There is no remaining operation-accuracy hypothesis in this theorem.
`captured_logit_error` applies the existing complete-stack composition
theorem to the actual saved logits. `captured_decisions` proves that the
first token's strict winner is class zero and the second token's strict
winner is class two. These are statements about the captured outputs;
agreement with the exact-real source's decisions requires a useful
comparison bound, which the existing global one does not provide.

The capture process, PyTorch runtime, driver and hardware are not themselves
verified in Lean. The mathematical statements concern the exact encoded
data and the pinned source equations. The scope is one checkpoint and word,
not all weights, future invocations, context lengths or modular tasks.

### L4. An unavoidable obstruction in the existing uniform budget

The final residual addition at token one, coordinate two has exact error

\[
 |\widehat x_{\mathrm{out},1,2}
   -(\widehat x_{\mathrm{mid},1,2}+\widehat f_{1,2})|=2^{-24}.
\]

`residual_error_exact` checks this equality directly on the captured binary
rationals. A coordinate's absolute error is at most its Euclidean error,
so `allowance_floor` proves that **every** `ModelAccurate` certificate for
these same tensors must have delta>=2^-24. Thus the obstruction below cannot
be removed merely by choosing a tighter common allowance than 10^-4.

Let L=2/sqrt(eps), the existing `rmsLip`. `budget_ge_initial` proves that
the old depth recurrence always satisfies

\[
 E_D\ge E_0=L\,\delta+\delta.
\]

Each residual block gain is at least one and its new error charge is
nonnegative. At eps=2^-23, L^2=2^25. Each tied embedding has norm one, so
the existing final logit radius satisfies

\[
 R_v > L^2\delta\ge 2^{25}2^{-24}=2.
\]

This uses only the unavoidable residual error and the old recurrence's
uniform initial/final normalization charge. It holds before considering
any additional inflation from XSA's small clipping epsilon or learned
matrix norms. The true source final RMSNorm has norm at most two; hence
every exact source logit is between -2 and 2 and every class margin is
at most four. Two class radii strictly exceed four.

`no_global_margin` consequently proves that, for these weights and captured
tensors, no valid nonnegative table allowance and uniform operation
allowance can satisfy `MarginAbove`, at any residual vector or target class.
The earlier error theorem remains sound; its sufficient bound is too coarse.
The captured decisions and active branches remain proved.

### L5. Consequence for the flat-spectrum / transformer / C-RASP connection

`no_execution_below` connects the obstruction to the exact
`Spectral.ExecutionBelow` premise used by J6--J7. For every margin gamma<=4,
it is impossible to discharge this premise for the captured trace using
the old uniform budget. Increasing the independent counting precision
does not remove that execution-budget obstruction. No spectral recovery
assumption or optimizer convergence claim has been added.

The required change is numerical: use verified norm intervals for
normalization sensitivity, individual operation errors and exact stored
embedding identification, then propagate those bounds through the actual
source stack. Section M supplies this improved execution certificate for
task #136. The obstruction in L4 remains valid for the old uniform budget;
the new certificate uses a different, operation-dependent bound. Its
counting transfer retains the independent fixed-point conditions.

Validation: the full Lean build passes (13233 jobs), without warnings in the
new modules. Individual axiom checks cover all 56 new public theorems and
use only propext, Classical.choice and Quot.sound. The audit reports 36
unchanged sorry, zero results resting on a sorry and zero extra axioms.
The regenerated index has 4136 modules and 21241 theorems, with zero
vacuous results and zero placeholders. Forbidden and whitespace checks pass.
The certificate command reproduces its saved witnesses and Lean sources
exactly, and all five pinned source hashes still match.

All seven certificate tests pass. The final 432-test lab suite passes with
three skips; the 104-test optimizer suite reproduces the two existing
ScheduleFreeMuon allocation failures (float32/float64, 75 allocations of
8519680 bytes each). Schedule-Free AdamW passes. The overall Python command
therefore exits unsuccessfully for the same baseline failures recorded in
#130--#132. Validation logs and the source/statement review are preserved in
`.kanban/artifacts/135/`.

## Trace-dependent execution bounds

This extension closes the execution-bound gap exhibited in L4 for the same
frozen width-four, one-head, one-block, two-token CUDA invocation. It changes
neither the checkpoint nor the captured tensors, source epsilons or exact
RoPE phases. The new result proves agreement of all captured class decisions
with exact-real source arithmetic. It does not assert that a specific
fixed-point counting execution has already satisfied its separate range and
rounding conditions, or that this checkpoint solves modular division.

### M1. Normalization sensitivity at a recorded input

For dimension d>0, eps>0 and arbitrary real vectors x,y, write

\[
 R_\epsilon(x)=\frac{\sqrt d\,x}{\sqrt{\|x\|^2+d\epsilon}}.
\]

The asymmetric, globally valid secant estimate is

\[
 \|R_\epsilon(x)-R_\epsilon(y)\|
 \le \frac{2\sqrt d}{\sqrt{\|x\|^2+d\epsilon}}\|x-y\|.
\]

Here x is the recorded input. No lower norm bound on the unknown exact
input y is assumed. To prove this, put a=||x||^2+d eps and
b=||y||^2+d eps. The difference of square roots satisfies

\[
 |\sqrt b-\sqrt a|\,\|y\|\le\|x-y\|\sqrt b.
\]

Splitting the normalized difference into the scaled displacement and the
change of scale gives two copies of sqrt(d)/sqrt(a). The old global proof
substituted sqrt(a)>=sqrt(d eps) at this point; the new proof retains a.
`Adaptive.rms_secant` formalizes the estimate. A rational gain L>=0 is valid
whenever 4d<=L^2(||x||^2+d eps), an exact polynomial check on the recorded
coordinates. `rms_step` then gives L E+delta, where E is the previously
proved input error and delta is this normalizer's local residual.

For clipped normalization U_eps(x)=x/max(||x||,eps), eps>0,

\[
 \|U_\epsilon(x)-U_\epsilon(y)\|
 \le\frac{2}{\max(\|x\|,\epsilon)}\|x-y\|.
\]

`l2_secant` retains the first denominator in the original proof and uses
that max with a constant is 1-Lipschitz. The estimate covers zero inputs and
crossings of the clipping boundary. `l2_step` accepts the recorded check
2<=L max(||x||,eps). These are proved analytic estimates, not Jacobians
assumed to remain constant in a neighborhood.

### M2. Composition with distinct residuals and stored weights

Each operation keeps its own error allowance, verified at its own recorded
inputs. For an exact stored matrix A with ||A||<=L, the next error is
L E+delta. `matrix_checked` derives L from the squared Frobenius norm of
its entries by Cauchy--Schwarz. Q, K and V use separate submatrices of the
same fused source matrix, with exact selector identities. The captured
embedding table equals the reference table exactly.

The generic operation rules retain the following terms:

- RMS and clipped L2: the recorded gain from M1 times input error, plus
  the particular normalizer residual.
- Arbitrary RoPE: 2 tau R+E+delta, where R bounds the recorded normalized
  vector, tau bounds both stored coefficients against the exact phase,
  and delta bounds arithmetic applying the stored table.
- Scaled dot product: (2R+E)E/sqrt(d)+delta when both recorded vectors have
  norm at most R and both input errors are at most E.
- Causal average: 8 E_score R_value+E_value+delta_SDPA. The value norm is
  checked on the recorded tensor; the mask includes the current token.
- SiLU: (1+R_gate)E_gate+delta_activation using recorded coordinate bounds.
- Hadamard product: V E_activation+(A+E_activation)E_value+delta_product,
  where V and A bound recorded value and activation coordinates. The cross
  term remains. Squared-coordinate aggregation avoids repeating a
  sqrt(FFN width) loss for every product.
- Residual addition: the two input radii plus its own addition residual.

XSA uses the existing two-input estimate, including the excess norm of a
computed direction; its exact direction has norm at most one. Source head
merging and learned projection are retained. At position zero, the source
and captured attention branches are both exactly zero. That case is proved
separately, so its small post-normalization denominator cannot amplify an
error from another position. The rules hold at arbitrary finite dimensions
and can be composed repeatedly; the fully instantiated certificate below
concerns the actual one-block capture, not every depth or checkpoint.

### M3. Rechecking the frozen operations

`./run.py model-trace-refined` rechecks all 52 original residual sites with
separate allowances, the squared scale residuals at tolerance 10^-12, and
separate Q/K/V arithmetic. Seven generated Lean modules import the original
stored literals unchanged and independently prove the sharper contracts.
The SiLU, softmax and RoPE checks retain their analytic Taylor remainders.
No floating transcendental calculation is accepted as an exact reference.

Representative certified constants are:

| Operation or recorded quantity | Bound |
| --- | ---: |
| Stored Q and K matrix norms, separately | 1/8000 |
| Stored V matrix norm | 101/100 |
| Stored attention projection norm | 13/50 |
| Stored gate, value and FFN output norms | 21/50 each |
| Attention and FFN input RMS gains | 21/10 |
| Q/K RMS gains | 6000 |
| Own-value clipped L2 gain | 2 |
| Active attention output RMS gain | 60 |
| FFN output RMS gain, both tokens | 1500 |
| Final RMS gain | 3/2 |
| Stored RoPE coefficient error | 10^-7 |
| Q/K slice arithmetic error, separately | 4*10^-12 |
| V slice arithmetic error | 4*10^-8 |
| Initial normalized embedding error | 1/4000000 |
| First residual addition error | 0 |
| Second residual addition error | 12*10^-8 |

The actual late addition still has a coordinate error of 2^-24, exactly as
in L4. It is charged at that addition alone. It is not an assumed error in
the stored embedding, either Q/K projection, or an earlier normalizer.
Rejection tests include perturbations small enough to pass the old 10^-6
coordinate checks but fail the new pre-RMS, Q-projection or residual checks.

### M4. Source agreement and a derived numerical margin

`Refined.Source` exposes the exact-real operations of the pinned source.
`source_hidden` proves that this graph is exactly its actual hidden-state
forward, including both residual additions and all normalization placements.
The propagation theorems have no unproved operation, hidden-state error,
source-margin or decision hypotheses. The following conservative radii hold
at both captured token positions:

| Stage | Euclidean error bound |
| --- | ---: |
| Attention input RMS | 53/100000000 |
| Each normalized Q/K | 48/100000000 |
| Each rotated query/key | 6/10000000 |
| Causal average | 31/10000000 |
| Own-value direction | 135/100000000 |
| XSA head | 93/10000000 |
| Attention branch output | 146/1000000 |
| First residual | 147/1000000 |
| FFN pre-RMS | 31/100000 |
| Gate and value, separately | 131/1000000 |
| Activated gate | 164/1000000 |
| Hadamard product | 575/10000000 |
| FFN output projection | 242/10000000 |
| FFN output RMS | 364/10000 |
| Final residual | 366/10000 |
| Final normalized state | 55/1000 |

The tied embeddings have norm one and the recorded final dot products have
zero local residual. Thus `error_logits` proves, for every token i and class v,

\[
 |\widehat\ell_{i,v}-\ell_{i,v}^{\rm source}|\le1/16.
\]

The strict captured winners are 0 and 2. Every competing captured class is
at least 17/20 below its winner, verified directly on the stored rationals.
Subtracting the two logit radii gives at least 17/20-2/16=29/40; in particular
`source_margin` proves the uniform source margin 7/10. `decisions_agree`
proves equality of the complete Boolean class-test vectors, including every
rejected class. The earlier `no_global_margin` and `no_execution_below`
theorems are unchanged: they concern the old sufficient bound, not the
classifier or these new operation-dependent radii.

### M5. Reconnecting to flat spectrum and the counting model

`Spectral.executed_of_logit_bound` accepts any independently proved
classwise execution radii whose pairwise sums fit below the source margin.
`uniform_local_certificate` uses this interface in the existing J7 bridge.
The source margin can therefore come from the same spectral construction;
its hypotheses about recovered, separating characters have not changed.
The formula family is fixed before all input words and traces, and retains
the depth upper bound n_layers and exact source RoPE kernels. This result
asserts no strict nonperiodic depth hierarchy or optimization convergence.

`Refined.source_score` identifies the captured invocation's exact readout
with `Spectral.sourceLogit`. `captured_counting_bridge` supplies its proved
radius 1/16 and source margin 7/10 to the generic theorem. It no longer
requires the impossible old `ExecutionBelow` premise. The formula agrees
with every captured final-token class decision provided the independently
rounded counting execution satisfies `TraceInRange` and `CountingBelow`;
the softmax precision inequality is also explicit. No concrete integer
width/fractional precision for this active checkpoint is certified here.
Those fixed-point premises are the remaining numerical work for an
unconditional captured-to-counting instance. The exact source/captured
agreement in M4 is already unconditional for the recorded tensors.

Validation: the full Lean build passes (13257 jobs), with no warnings in the
new modules. All 119 new public theorems use only propext, Classical.choice
and Quot.sound. The repository audit reports the unchanged 36 sorry, zero
results resting on a sorry and zero extra axioms. The regenerated index has
4160 modules and 21360 theorems, with zero vacuous results and placeholders.
All eleven original/refined trace tests pass, including exact export replay
and the tighter residual rejection cases. All five pinned physical source
hashes still match. Logs and the statement/source review are preserved in
`.kanban/artifacts/136/`.

The final 436-test lab suite passes with three skips. The 104-test optimizer
suite reproduces the two existing ScheduleFreeMuon allocation failures in
float32/float64 (75 allocations of 8519680 bytes each); Schedule-Free AdamW
passes. Consequently the overall `./run.py test` exit status is unsuccessful
for the same baseline failures as L5, while all eleven trace certificate
and rejection tests pass. Task #137 records the remaining fixed-point range
and precision proofs for this active captured checkpoint.

## Local attention routing (2026-11-01)

The user's proposed mechanism is per-position retrieval and composition:
each query forms a weighted sum of relevant earlier value vectors, and
successive feed-forward/residual blocks combine these pieces. A context of
N tokens has N separate states of width W. A whole-table interpolation
construction with exponential width is a sufficient universality result;
it is not a necessary width for structured arithmetic. The next construction
target is polynomial resources, ideally width independent of chain length
at fixed prime with growing active depth. That construction and its learning
guarantee remain open in tasks#139 and#133.

The following local bounds are formalized in
`Transformer.SequenceModel.Attention.Retrieval`. They use the existing actual
source `causalWeight` and `headScore`, including positive RMS epsilon and
arbitrary rotary phases. Causal softmax includes the current position before
the source XSA operation; a useful set may consist solely of earlier tokens.

### LR1. Weighted pieces and retrieval error

Fix one causal row i, coefficients alpha_j, values v_j and target t.
The source coefficients are nonnegative and sum to one. If
`norm(v_j-t) <= e_j` at every position, then

```
norm(sum_j alpha_j*v_j - t) <= sum_j alpha_j*e_j.
```

For a useful set S, let alpha=sum_(j in S) alpha_j. If useful values have
error at most g and remaining values error at most b, the error is at most
`alpha*g + (1-alpha)*b`. Multiple positions may support the desired feature;
no unique best position is assumed. These are bounds on the raw attention
average, before XSA, output projection and Peri-RMSNorm. Propagation through
those operations needs the existing source stability/execution bounds; a
small raw error alone is not a final classification certificate.

### LR2. Useful mass under actual Q/K normalization

Let N=i+1 be the visible prefix size and r=card(S). For scores bounded in
absolute value by B, the useful mass is at most `r*exp(2B)/N`. For the actual
source normalized rotary head of positive width d and positive epsilon,
the proved score bound gives B=sqrt(d). Hence

```
sum_(j in S) alpha_j <= r*exp(2*sqrt(d))/N.
```

The upper bound can exceed one. Future positions contribute zero, so the
bound remains valid without assuming every member of S is visible. A larger
useful set changes the bound: selecting one distinguished token and
aggregating a broad set are different local requirements.

### LR3--LR4. A necessary local score/width budget

Suppose S is nonempty and reaches mass at least mu>0. Then

```
log(N*mu/r) <= 2*sqrt(d),
max(0, log(N*mu/r))^2 <= 4*d.
```

For fixed r and mu this necessary single-head concentration condition grows
as log(N)^2, not exponentially in N. It is not a sufficient construction,
a task-specific width lower bound, or a multi-layer impossibility theorem.
It does not assert that training creates the required query/key scores.
In particular, it supplies no unconditional constant-width routing guarantee
for arbitrarily long contexts in this source with bounded normalized scores.

| Claim | Lean declaration |
| --- | --- |
| Nonnegative coefficients | `causalWeight_nonneg` |
| Weighted retrieval error | `retrieval_error_le` |
| Useful-set error | `retrieval_error_of_useful_set` |
| Bounded-score useful mass | `useful_mass_le` |
| Actual normalized useful mass | `normalized_useful_mass_le` |
| Necessary log score budget | `retrieval_log_budget` |
| Necessary squared-log width budget | `retrieval_width_budget` |

Each theorem with hypotheses has a concrete witness. These local statements
refine what attention must accomplish; they neither assume nor establish an
optimizer convergence theorem. The source-model experiments are intended to
find counterexamples to current learning interpretations. The frozen early
screen, subsequent profiling and bounded long-grokking phase are tracked in
`EXPERIMENT_PLAN.md`, tasks#138/#140--#143. A finite plateau is not permanent
failure, and a transformer succeeding at a flat task would invalidate an
impossibility inference, not the exact nuclear-norm completion theorem.

Validation of LR1--LR4: `lake build` passes13258 jobs. The seven new
theorems use only propext, Classical.choice and Quot.sound. The audit has
the unchanged36 sorry, zero results resting on sorry and zero extra axioms.
The regenerated index contains4161 modules and21367 theorems, with zero
vacuous statements and placeholders; the forbidden-pattern check passes.


## Experimental corrections to learning implications (2026-11-01)

The actual normalized RoPE source transformer was tested with Schedule-Free
AdamW on division and `a^3 + ab mod P`, including approximately parameter-matched
W256D1/W128D4 models. A frozen 48-case screen, profiles, and a separate ten-case
full-state continuation are recorded in `experiments/modular_composition/`
and `modular_composition_long/`. The continuation preserves query, base,
second moments, averaging clock, sampler, labels and splits. Python 3.15,
Torch 2.15 and CUDA max-autotune are used throughout learning.

All four long division pairs finish at 100% heldout accuracy. P97 W256D1
first reaches 99% at 12,500 updates (657.93 sampled-table epochs); W128D4
first reaches it at 27,500 (1,447.38 epochs), relapses below it at 30,000,
and returns at 32,000. P193 W256D1 first crosses at 750 updates (10.28
epochs); W128D4 crosses at 12,250 (167.82 epochs). Every run continues to
100,000; no checkpoint is selected by heldout performance. The P97 deep
relapse is retained. The actual nonzero-input division observation graphs
have verified connectedness and perfect matchings, so an absent ideal rank-one
recovery premise does not explain the early failures.

These observations reject the **learning interpretations** that rank-one
character availability predicts discovery within 2,000 updates, or that an
early memorization plateau is permanent. They do not contradict the exact
kernel-completion theorems. Width/depth changes leave the target spectrum
unchanged but strongly change the observed discovery time. This comparison
also changes head dimension and compute, so it is not an isolated causal
estimate of depth. A source-faithful optimizer-to-feature recovery theorem
remains missing; neither unlimited representational capacity nor an ideal
nuclear-norm optimum supplies it.

All four mixed pair models fit the observed training half at 100%, but their
100,000-update heldout accuracies are only 1.360%/1.445% at P97 and
0.816%/0.854% at P193 (wide/deep respectively). Their exact fixed character
kernels remain flat. This supplies no successful heldout counterexample to
the proposed flat-task barrier, and also no theorem that transformers cannot
learn such tasks. The full-table P193 fitting control reaches 100% training
accuracy after its early 58.62% endpoint; fitting with every label supplied
is a separate statement from heldout recovery.

The residual audit identified a second gap: an exact real identity does not
ensure stable floating-point execution or backward accumulation. Original
XSA cancellation on repeated values is amplified by Peri-RMSNorm over depth.
Centering improves outputs but alone can make shared-parameter FP32 gradients
much less accurate. The final guarded common-prefix implementation passes an
actual-source FP32/FP64 audit. Seventeen Lean theorems prove centered/clipped
source equivalence, common prefix states through arbitrary finite depth and
the tied readout, for arbitrary rotary phases. Those equalities justify the
implementation; the numerical errors are measured separately. The frozen long
runs retain the old arithmetic, so their late grokking is not attributed to
this correction. See `review/residual_connections.md` for every failure stage.

Finally, the known-solution depth-attention toy rejects an unconditional
trainability inference from bounded Q/K scores: unit-scale normalized queries
learn all three declared data seeds, whereas magnitude 1e6 can leave tiny
query gradients and fail within the same 2,000 updates. Key-only queries can
saturate to exact zero FP32 gradients; their zero-initialized controls succeed.
The normalized score and Lipschitz theorems remain valid. Initialization,
parameter scaling, optimizer dynamics and execution fidelity are additional
premises, not consequences of flat or rank-one spectra. Full-model residual
ablation and paired deep numerical retraining remain tasks #145 and #148;
polynomial-resource source constructions and learning guarantees remain #139
and #133/#134.


The complete continuation audit finishes all ten cases within 6,283.38 seconds
of the 7,200-second cap. Every source-to-initial full state matches bitwise;
all final classifications survive checkpoint replay. Two deep mixed CPU
one-answer differences disappear on the original compiled CUDA path within
the unchanged loss tolerance; both measurements remain recorded. The chain
case finishes at 1.270% heldout and 0.684% at doubled length, so it supplies
no successful length-generalization counterexample in this cohort. These
finite outcomes leave structured composition and optimizer recovery open.

## Classification without unique recovery

Added on the user's request, 2026-11-07. These are conditional statements about
the ideal nuclear-norm learner and fixed real linear readouts. They do not
identify neural weight decay with nuclear-norm minimization, establish global
optimality of AdamW, or assert these conditions for the cubic experiment.

1. **Recover only the readout.** Let `U,V` have orthonormal columns, with
   `U U* K = K` and `tr((U V*)* K) = ||K||_*`. Suppose `Y = U V* + W` is
   supported on the observations `Omega`, `U* W = 0`, `W V = 0`, and `c W`
   is a contraction for some `c > 1`. Write
   `T = {U A* + B V*}` and `N = {H : H|Omega = 0}`. A real linear
   functional `L` satisfying `L(H) = 0` for every `H` in `T intersect N`
   has `L(X) = L(K)` for every feasible global nuclear-norm minimum `X`.
   Proof: the existing certificate inequality and optimality give
   `(1 - U U*) (X - K) (1 - V V*) = 0`. Consequently `X - K` lies in
   `T intersect N`, where `L` vanishes. Tangent injectivity, which would
   force this entire intersection to be zero, is not assumed.

2. **Every optimum classifies correctly.** Let `s(q,a)` be real linear
   functionals of the feature matrix, and let `f(q)` be the correct class.
   Suppose `s(q,f(q))(K) > s(q,a)(K)` for every incorrect `a`. Under item1's
   certificate, require only the score contrast `s(q,f(q)) - s(q,a)` to
   vanish on `T intersect N`. Item1 then preserves every strict correct-class
   gap for every feasible global minimum. Absolute scores may change by a
   common offset. This is a sufficient condition for classification without
   unique feature recovery; the strict certificate is retained. A non-strict
   certificate alone only proves that `K` is a minimum and cannot guarantee
   that every optimal solution gives the desired labels.

3. **Nonunique witness.** Take `K = diag(1,-1)`, observe only its diagonal,
   and take `Z = [[1,1],[-1,-1]]`. Both have nuclear norm2 and are global
   minima: the supported contraction `Y = K` certifies the lower bound2.
   For query `q in {0,1}`, use features `(X[q,q], X[0,1])` and fixed class
   codes `(1,1)` and `(-1,1)`. Thus
   `s(q,0)(X) = Re X[q,q] + Re X[0,1]` and
   `s(q,1)(X) = -Re X[q,q] + Re X[0,1]`.
   Every feasible global minimum predicts class0 for query0 and class1 for
   query1, with gap2. The two displayed minima are distinct and give different
   absolute scores. The certificate has `U = K`, `V = I`, `W = 0`, `c = 2`;
   its unobserved tangent directions have zero diagonal, so every contrast
   vanishes on them. This witness checks nonunique feature recovery with
   unchanged labels, not generalization on unobserved modular examples.

| Claim | Lean declaration |
| --- | --- |
| Recover a real linear readout without tangent injectivity | `FeatureCodes.linear_readout_eq_of_certificate` |
| Every optimum has the correct strict winner | `FeatureCodes.classification_of_minimum_without_unique_recovery` |
| Distinct optimal features and scores, identical correct labels | `FeatureCodes.nonunique_recovery_correct_classification` |

Each general theorem has a concrete nonunique instantiation in
`FeatureCodes/NonuniqueWitness.lean`. No unique completion assumption is hidden
in the readout condition: the displayed target and alternative completion
violate uniqueness while satisfying that condition.

## Complex-product Rotate90 bridge

Source model: `rust_projects/grokking/experiments/rotate90_complex`,
implementation/protocol commit `b70811e`, results commit `a9a2231`, inspected
2026-11-07. This section derives the bridge for the plain complex-product arm,
not the radial-gated or SwiGLU arm. The corresponding Lean namespace is
`Transformer.Rotate90.ComplexProduct`.

### CP1. Exact source model and its real expansion

Let `E : labels -> C^k` be the shared embedding and let `G,U,O` be the three
complex-linear, input-major matrices. Define

```
D_eps(z) = sqrt(eps + sum_j |z_j|^2 / (2 k))
N_eps(z) = z / D_eps(z)
P(g,u)_j = g_j u_j / sqrt(2)
B(z) = P(z G, z U) O
x(a,b) = N_eps(E[a] + i E[b])
h(a,b) = x(a,b) + N_eps(B(N_eps(x(a,b))))
L(a,b,c) = Re sum_j N_eps(h(a,b))_j conjugate(E[c]_j).
```

The factor `2 k` is the actual real-coordinate width, not `k`. Epsilon is
retained in every normalization; the experiment uses float32 epsilon.
There is one residual branch with coefficient one. All four normalizations,
the fixed quarter-turn and the tied readout are present. No class answer or
intermediate sum is supplied to the source model.

Independently expand each complex matrix `A+iB` into real blocks with the
signs implementing `(x+i y)(A+iB) = (xA-yB)+i(xB+yA)`. Define the real product
using its four scalar products and the same `sqrt(2)` divisor. Pairwise
`J(x,y)=(-y,x)` implements multiplication by `i`; the real dot product is the
real part of the Hermitian dot product. Summing the two real squares in each
pair gives the same RMS denominator. These identities propagate through
the entire residual computation, proving equality of the independently
defined real and complex logits for arbitrary parameter arrays.

The Lean indices are logical pairs. The native residual uses adjacent pairs,
whereas its hidden buffer stores all real components followed by all imaginary
components; the pairing describes their logical coordinates. The theorem is
an exact-arithmetic model correspondence, not a proof of buffer indexing,
PTX execution, rounding or training. The Rust forward/gradient/replay tests
are separate numerical evidence.

### CP2. Final normalization and approximate separating features

Write the actual raw tied score as `R_c = Re sum_j h_j conjugate(E[c]_j)`.
For `eps > 0`, `D = D_eps(h) > 0` and `L_c = R_c / D`. Therefore the exact
winner is unchanged by the final RMSNorm. Its numerical margin is divided
by `D`; it is incorrect to drop this factor in a rounding-error bound.

One sufficient feature condition uses unit-modulus embeddings, with ideal
scores `S_t(c) = Re sum_j E[t]_j conjugate(E[c]_j)`. If the correct ideal
code has gap at least `delta` and
`sum_j |h_j - E[t]_j| <= eta`, then every raw score differs from its ideal
score by at most `eta`. Thus `2 eta < delta` implies correct real-source
classification. The feature error is on the computed residual state above;
it is not a replacement definition of the network. Unit-modulus embeddings,
code separation and the feature approximation are explicit hypotheses,
not inferred properties of the trained embedding.

### CP3. Executed classification from a raw source margin

Suppose `R_t - R_c >= delta` for every incorrect class and an execution
returns scores `Lhat` with `|Lhat_c - L_c| <= eta` for every class. Then

```
Lhat_t - Lhat_c >= delta / D - 2 eta.
```

Consequently `2 eta D < delta` certifies a strict correct winner for the
execution. This bound includes the errors of both competing scores and
the actual source denominator. It allows any embedding and matrices; it
does not require the unit-code condition used in CP2. The execution-error
bound must be established separately for the checkpoint/backend in question.

### CP4. Rank-r nonunique recovery to the actual source classifier

Use the rank-r certificate of *Classification without unique recovery*,
item 1: orthonormal `U,V`, the stated target polar-factor identities,
supported `UV*+W`, `U*W=0`, `WV=0`, and contractive `cW` for some `c>1`.
Let `X` fit the observed target matrix `K` and globally minimize nuclear
norm. For a query `q`, fix real linear matrix scores `s(q,c)` and answer
`f(q)`. Require each contrast `s(q,f(q))-s(q,c)` to vanish on the unobserved
tangent directions `T intersect ker(P_Omega)`. Tangent injectivity is not
required, so multiple optimal feature matrices remain possible.

The previous recovery theorem gives equality of every correct-versus-wrong
score contrast at `X` and `K`. A **separate source-alignment premise** is

```
R_f(q)(inputs(q)) - R_c(inputs(q)) = s(q,f(q))(X) - s(q,c)(X),
```

where `R` is the actual complex-product residual state's tied readout from
CP1. Absolute scores may differ by a common offset; only correct-versus-wrong
contrasts need align. If the target matrix gives raw margin `delta(q)`, the
actual source therefore inherits that raw margin. Applying CP3 proves correct executed
labels when each execution error is bounded by `eta(q)` and
`2 eta(q) D_eps(h(inputs(q))) < delta(q)`.

This is a conditional bridge for arbitrary finite rank, not a theorem that
the normalized neural training objective is the nuclear-norm objective.
Observed fitting, global optimality, score-contrast preservation, alignment
and execution accuracy are separate obligations. AdamW convergence,
flatness of learned spectra and satisfaction of these obligations by the
cubic-task checkpoints have not been proved. The result does not assert
that the complex product is optimal among possible blocks or optimizers.

### CP5. Jointly satisfiable, nonunique example

Use two labels with one complex embedding coordinate: `E[0]=1`, `E[1]=-1`,
epsilon one, both input projections one, and output projection zero. For
queries `q=0,1`, use input `(q,0)` and answer `q`. The actual residual state
is `(sigma(q)+i)/sqrt(2)`, with `sigma(0)=1`, `sigma(1)=-1`; its raw score
is `sigma(c) sigma(q)/sqrt(2)`, giving positive gap `2/sqrt(2)` for both
different labels. The ideal code gap is two and the actual feature error
is `sqrt(2-sqrt(2)) < 4/5`, so CP2 applies with `delta=2`, `eta=4/5`.

For the spectral bridge take `K=diag(1,-1)`, observe only the diagonal and
use the distinct completion `Z=[[1,1],[-1,-1]]`. Both are global minima of
nuclear norm with value two. Set `U=K`, `V=I`, `W=0`, `c=2`, and fix
`s(q,c)(X)=sigma(c) Re X[q,q]/sqrt(2)`. Fitting the diagonal proves the
source alignment for every optimum; every contrast annihilates unobserved
tangent directions. CP4 therefore applies to both distinct matrices with
exact execution (`eta=0`) and classifies both inputs correctly.

This witness deliberately uses the residual path; it proves satisfiability
with distinct labels and distinct feature optima, not modular generalization
or the usefulness of a nonzero branch. Separate nonzero complex parameter
instances exercise every real/complex algebraic correspondence.

| Claim | Lean declaration in `Rotate90.ComplexProduct` |
| --- | --- |
| Entire real and complex source models agree | `realLogit_eq_logit` |
| Final normalization preserves strict comparisons | `realLogit_lt_iff` |
| Approximate separating features classify | `realLogit_correct_of_feature_error` |
| Raw margin and bounded execution error certify labels | `executed_correct_of_raw_margin` |
| Nonunique rank-r optimum transfers to executed labels | `executed_correct_of_nonunique_minimum` |
| Two distinct optima satisfy the concrete source bridge | `copy_nonunique_bridge` |

## Approximate nonunique recovery

Added 2026-11-08, following the user's request for a quantitative extension.
These are deterministic inequalities for finite complex matrices. They do not
identify the neural objective with nuclear-norm minimization. The empirical
application must establish each premise for its chosen features and readout.

### QR1. Condition the readout, allowing invisible ambiguity

Write `N = {H : H|Omega = 0}`, `T = {U A* + B V*}` and
`Pnormal(H) = (I-UU*) H (I-VV*)`. If a real-linear functional `L` vanishes
on `T intersect N`, then there is a finite `kappa >= 0` such that

```
|L(H)| <= kappa ||Pnormal(H)||_*   for every H in N.
```

Proof: the kernel of `Pnormal` restricted to `N` is contained in the kernel
of `L` restricted to `N`, by the explicit tangent/normal decomposition.
The induced functional on the projection's range extends linearly to all
matrices. Every finite-dimensional matrix functional has a nuclear-norm bound:
one explicit bound is the sum over cells of `|L(e_ij)| + |L(i e_ij)|`.
The proof uses that bound for the extended functional. It does not invert
the full matrix or require that `T intersect N` be zero. Existence is not a
small-constant guarantee; a useful certificate must bound or compute `kappa`.

### QR2. Noisy fitting and finite objective excess

Retain CP4's strict rank-r certificate: orthonormal `U,V`, the target polar
identities, supported `UV*+W`, `U*W=0`, `WV=0` and contractive `cW` for `c>1`.
Put `s = 1-1/c > 0`. Suppose a correction `E` makes `X-E` agree with `K`
on the observations, `||E||_* <= rho`, and
`||X||_* <= ||K||_* + epsilon`. The correction can be the observed residual
with zeros elsewhere. Then

```
||Pnormal(X-E-K)||_* <= (epsilon + rho)/s.
```

Indeed, the old certificate inequality bounds its left side times `s` by
`||X-E||_* - ||K||_*`, which is at most `epsilon + rho` by the triangle
inequality. `epsilon` may be a signed objective excess; the premises imply
`epsilon + rho >= 0`. No exact fitting by `X` or exact global optimum is needed.

### QR3. Recover the score contrast approximately

In addition to QR1's hidden conditioning, let `lambda >= 0` satisfy
`|L(M)| <= lambda ||M||_*` for every matrix `M`. The explicit bound of QR1
always supplies such a `lambda`, though a tighter one is preferable. Splitting
`X-K = (X-E-K)+E` gives

```
|L(X)-L(K)| <= kappa*(epsilon+rho)/s + lambda*rho.
```

This is applied directly to a correct-versus-wrong score contrast. It does
not demand accurate absolute scores or a uniquely determined feature matrix.
If actual raw source contrasts differ from their recovered values by at most
`alpha(q,v)`, their error relative to the target is bounded by

```
B(q,v) = alpha(q,v) + kappa(q,v)*(epsilon+rho)/s + lambda(q,v)*rho.
```

The alignment error here is already a contrast error; it is not doubled.

### QR4. Preserve the actual source's executed decision

Let the target raw gap be at least `delta(q)` for every wrong class. With
the actual CP1 residual state `h`, denominator `D_eps(h)`, and execution
error at most `eta(q)` per logit, every executed class is correct if

```
B(q,v) + 2*eta(q)*D_eps(h(inputs(q))) < delta(q)
```

for every wrong class. The denominator is positive for positive epsilon.
The exact source gap is at least `delta-B`; dividing by its denominator and
allowing the two execution errors proves the strict comparison. Both the
direct contrast-error version and the full spectral-to-source composition
are proved. Neither version predicts optimizer convergence or supplies a
backend execution bound.

### QR5. Nonzero noise with nonunique underlying optima

Keep CP5's `K=diag(1,-1)`, alternative optimum `Z=[[1,1],[-1,-1]]`,
diagonal observations and two-label source. Put `X=Z+E` with `||E||_*<=1/8`.
The fitting correction is `E`, and the objective excess is at most `1/8`.
For CP5's contrast, `kappa=0` and `lambda=2/sqrt(2)`: unobserved diagonal
contrasts are zero, and each entry's modulus is bounded by the nuclear norm.
Its alignment error is at most `lambda/8`. With `eta=0`, the combined bound
is `lambda/4`, strictly below the raw gap `delta=lambda`. The explicit
perturbation `E=K/16` is nonzero and has nuclear norm exactly `1/8`.
This instantiates all general bounds. It is still a residual-path consistency
witness, not a modular generalization or active-product construction.

| Claim | Lean declaration |
| --- | --- |
| Explicit finite functional bound | `FeatureCodes.readoutBound_nonneg`, `FeatureCodes.abs_readout_le` |
| Finite conditioning without tangent injectivity | `FeatureCodes.exists_readout_conditioning` |
| Normal component bound from noise and objective excess | `FeatureCodes.normal_error_le_of_approximate_minimum` |
| Approximate recovery of a linear contrast | `FeatureCodes.linear_readout_error_le_of_approximate_minimum` |
| Add recovery and alignment errors | `FeatureCodes.contrast_error_le_of_alignment` |
| Direct contrast-error execution bound | `ComplexProduct.executed_correct_of_contrast_error` |
| Full quantitative spectral-to-source bridge | `ComplexProduct.executed_correct_of_approximate_recovery` |
| Nonzero noisy witness | `ComplexProduct.witnessNoise_norm`, `ComplexProduct.witnessNoise_ne_zero`, `ComplexProduct.copy_approximate_spectral_classifies` |

## Answer equivalence

Added 2026-11-08 after the user asked whether the failed diagnostic called for
changing the complex product or extending the theory. The extension concerns
which distinctions a class readout must preserve. It does not modify the model.

### QE1. Read out a fiber in any isometric feature basis

Let `S` be a finite latent set, `f : S -> A` the answer map, and `F*F=I`.
For each possible answer `a`, let `v_a(s)=[f(s)=a]`, `C_a=F v_a` and
`R_a(z)=Re <C_a,z>`, using the Hermitian inner product. Direct calculation gives

```
R_a(F x) = sum_{s: f(s)=a} Re x_s.
R_a(F e_s) = [f(s)=a].
```

Here `e_s` is a coordinate vector. The readout has exact gap one against every
incorrect answer, including answers outside the image of `f`. The answer codes
need not have unit coordinates or equal norms. This avoids imposing the earlier
unit-modulus code hypothesis on fiber indicators transformed by Fourier analysis.

### QE2. Different features with the same complete score vector

If `f(s)=f(t)`, every class score agrees at `F e_s` and `F e_t`. If `s!=t`,
the feature vectors are nevertheless distinct, since applying `F*` recovers
the distinct coordinate vectors. More generally, if an error `e` satisfies
`sum_{s:f(s)=a} Re e_s=0` for each answer, then

```
R_a(F(x+e)) = R_a(F x)   for every a.
```

There is no bound on the size of the individual error coordinates. The concrete
four-latent-state witness uses `S=Bool*Bool`, answer the first bit, and errors
`+100` and `-100` on the two states of each answer. All class scores are unchanged.
Thus bounding every coordinate error separately can discard cancellation that
is essential to an accurate classification-level analysis.

### QE3. Bound the combined contrast after summing

For target latent state `s`, suppose, for every wrong answer `a`,

```
|sum_{t:f(t)=f(s)} Re e_t - sum_{t:f(t)=a} Re e_t| <= epsilon < 1.
```

Then `R_{f(s)}(F(e_s+e)) > R_a(F(e_s+e))`. The lower bound on the actual
contrast is `1-epsilon`. This is a bound on the complete contrast after summing
errors within each answer fiber. It is not a sum of separate absolute errors.
To apply it to a learned network, one must still identify its readout with this
construction or bound the source alignment error, as in QR4.

### QE4. The cubic task's exact equivalence

In `ZMod 97` the polynomial identity

```
(s-t)(s-35t)(s-61t) = s^3-t^3
```

implies `s^3=t^3` exactly when `s=t`, `s=35t`, or `s=61t`. The formula
includes zero; it also applies to the sums of the original two operands.
In particular, sums 1 and 35 have the same cube. The columns for those sums
are distinct in every isometric basis, but all cubic fiber readouts agree.
The explicit normalized basis `fourierMatrix 97 / sqrt(97)` is proved isometric
using the existing Fourier Gram identity, so this conclusion applies to the
actual additive-character features, not just an unspecified change of coordinates.

These theorems justify allowing answer-preserving feature ambiguity. They do
not show that a trained checkpoint learns exactly this code, give efficient
tied-embedding source weights, or prove a rate of optimizer discovery. The
failed full-character diagnostic must not be interpreted as a defect in
complex multiplication.

| Claim | Lean declaration in `FeatureCodes` |
| --- | --- |
| Hermitian readout equals the fiber sum | `fiber_readout_mulVec`, `fiber_readout_column` |
| Distinct features can have identical complete scores | `fiber_readouts_eq_of_same_answer`, `feature_columns_ne` |
| Balanced errors are invisible | `fiber_readouts_ignore_balanced_error` |
| A combined contrast error below one preserves classification | `fiber_classifies_of_contrast_error` |
| Exact modulus-97 cubic equivalence | `cubic97_factor`, `cubic97_eq_iff`, `cubic97_answers_eq_iff` |
| Concrete additive-Fourier witness | `cubic97_distinct_features_same_readouts`, `cubicFourier97_isometry`, `cubic97_fourier_features_same_readouts` |

## Combined answer contrasts

Added 2026-11-08, task #179. The next step after QE1–QE4 is an arbitrary
learned decoder on answer coordinates, with explicit error and source alignment.
These are deterministic conditional bounds. A probe fitted on training inputs
does not, by itself, establish its error on held-out inputs.

### AC1. Pool the observable error

For a finite latent set `S` and answer map `f:S->A`, put
`P_f x(a) = sum_{s:f(s)=a} x(s)`. Coordinates are real and may be signed.
Pooling preserves total mass and contracts absolute error:

```
sum_a P_f x(a) = sum_s x(s)
sum_a |P_f e(a)| <= sum_s |e(s)|.
```

It also commutes with every linear prediction operator:
`P_f(sum_i w_i x_i) = sum_i w_i P_f(x_i)`.
Consequently, fixed-design ridge regression with the same penalty commutes
with pooling its output targets. A smaller answer probe is not an independent
discovery of a different representation. This identity is checked independently
in the native diagnostic, including the affine mass correction below.

### AC2. An arbitrary answer decoder and invisible offsets

Let `C(v,a)` be any real answer decoder, including scores for output labels
outside the answer image. Its score is `s_v(p)=sum_a C(v,a)p(a)`.
If a latent decoder has weights `C(v,f(s))+b(s)`, its correct-versus-wrong
contrast on any `x` equals `s_t(P_f x)-s_v(P_f x)`. The common latent offset
`b(s)` cancels exactly, regardless of its magnitude or the total mass of `x`.
Errors with zero mass in every fiber have no effect on any pooled score.
The witness has errors `M,-M` within each of two fibers for arbitrary real `M`.

### AC3. Center the contrast coefficients

Given a reference `r` with total mass one, replace a prediction `p` by
`p'(a)=p(a)+(1-sum_b p(b))*r(a)`. Then `sum_a p'(a)=1`, even if its entries
are negative. Use the pushforward reference `P_f r` when pooling; linearity
and conservation of total mass make correction commute with pooling too.

For any zero-total-mass error `e`, coefficient vector `d`, and constant `c`,

```
sum_a d(a)e(a) = sum_a (d(a)-c)e(a).
|sum_a d(a)e(a)| <= max_a |d(a)-c| * sum_a |e(a)|.
```

If `lo<=d(a)<=hi`, choose `c=(lo+hi)/2`. The multiplier is `(hi-lo)/2`.
The bound is sharp: `d=(b+1,b-1)` and `e=(M,-M)` give equality for any
`b,M`. Thus a shared coefficient offset cannot inflate this bound, and
balanced latent errors disappear before the absolute error is computed.

### AC4. The source's class contrasts

For answer coordinate `a`, output target `t`, competitor `v`, and unit-mass
prediction `p`, set `e=p-e_a` and `d_v(j)=C(t,j)-C(v,j)`. The decoder's
ideal gap is `delta_v=C(t,a)-C(v,a)`. AC3 gives

```
|(s_t(p)-s_v(p))-delta_v| <= ((hi_v-lo_v)/2) * ||e||_1.
```

Let `alpha_v` bound the difference between the actual raw source contrast
and this decoded contrast. If the pooled error is at most `epsilon`, the
actual final RMS denominator is `D`, and each executed logit has error at
most `eta`, a sufficient condition for every executed comparison is

```
alpha_v + ((hi_v-lo_v)/2)*epsilon + 2*eta*D < delta_v.
```

Each competitor keeps its own gap and bound. Combining the smallest gap
with the largest error from a different competitor needlessly loses precision.
The Lean source theorem retains all input/branch/final normalizations and
tied weights. It does not identify either `epsilon` or `alpha_v` from training.
QR3 can supply a bound on an observable error when its spectral premises are
proved; AC1–AC4 do not replace those premises with measured accuracy.

### AC5. Nonzero-error source witness

For CP5's two-label source, use its raw score columns as `C` and put mass
`7/8` on the correct answer and `1/8` on the other. The L1 error is `1/4`,
the coefficient half-range is `g=2/sqrt(2)`, and the source alignment error
is `g/4`. With zero execution error, their sum `g/2` is below the ideal
gap `g`. Both labels classify. This witnesses nonzero recovery and alignment
errors; it remains a residual-path example, not a learned active product.

| Claim | Lean declaration |
| --- | --- |
| Pooling preserves mass, contracts error and commutes with linear prediction | `FeatureCodes.sum_answerMass`, `answerMass_abs_sum_le`, `answerMass_linear_prediction` |
| Fiber decoder and invisible common offsets/errors | `FeatureCodes.answerScore_pool`, `answer_contrast_common_offset`, `answerScore_ignore_balanced` |
| Exact unit-mass correction, commuting with pooling | `FeatureCodes.sum_unitMass`, `answerMass_unitMass` |
| Centered and sharp range bounds | `FeatureCodes.centered_contrast_error_le`, `range_contrast_error_le`, `range_bound_attained` |
| Combined learned-decoder error | `FeatureCodes.answer_contrast_error_le` |
| Actual executed source and nonzero-error witness | `ComplexProduct.executed_correct_of_answer_mass`, `copy_answer_mass_classifies` |

## Training-state transport

Added 2026-11-08, task #180. AC4 still required a measured error against an
ideal answer code and a source-alignment term. Here the bound works directly
with the actual tied readout. Its inputs are observed training labels/states,
the trained readout weights, raw query operands and the unlabeled query state.
It is an input-dependent deterministic condition, not an optimizer theorem,
a statistical coverage claim, or a prediction made without running the model.

### TT1. Reconstruct a query displacement

Let `T_i` be real training states in a group that preserves the task answer,
`mu` a center computed from them, and `x` the query state. The diagnostic uses
the training mean as `mu`. For arbitrary real coefficients `a_i`, define

```
H_i = T_i - mu
r = x - mu - sum_i a_i H_i.
```

The residual is calculated from states and is never assumed to vanish. For
any linear contrast `L_w(z)=sum_k w_k z_k`, exact algebra gives

```
L_w(x)-L_w(mu) = sum_i a_i (L_w(T_i)-L_w(mu)) + L_w(r).
```

Neither a fitted ideal decoder nor its alignment with the source is needed.

### TT2. Joint training/residual energy

For nonzero real `s`, put `lambda=s^2>0` and

```
G = sum_i a_i^2 + ||r||_2^2/lambda
B_w = sum_i (L_w(T_i)-L_w(mu))^2 + lambda*||w||_2^2.
```

Apply Cauchy–Schwarz to the concatenated vectors
`(a_i, r_k/s)` and `(L_w(T_i)-L_w(mu), s*w_k)`. This proves

```
|L_w(x)-L_w(mu)| <= sqrt(G*B_w).
```

All terms on the right use training contrasts, known weights and query-state
geometry. No query target error occurs. Omitting the residual term is false:
zero training states and center, with scalar query and readout both one, have
zero training variation and query contrast error one. This counterexample is
proved explicitly.

A separate Euclidean-center baseline is
`|L_w(x)-L_w(mu)| <= ||x-mu||_2*||w||_2`. It uses displacement from a
training center, which can be much smaller than the complete query state.

The numerical diagnostic chooses `a` by regularized linear reconstruction:
`z=(H^T H+lambda I)^(-1)(x-mu)`, `a=H z`, `r=x-mu-H^T a`. The theorem
does not depend on inversion correctness or optimality: any coefficients with
their actual residual satisfy its inequality. Floating-point evaluation of
the coefficients, residual and inequalities still needs arithmetic certification.

### TT3. Actual source classification and valid bound selection

For target label `t` supplied by training and competitor `v`, take
`w_v=realify(E_t)-realify(E_v)`, the difference of the actual tied embedding
rows. The existing real/complex correspondence proves
`L_wv(realify(state)) = rawLogit_t-rawLogit_v` exactly. Put `g_v=L_wv(mu)`.
The source classifier's executed comparison is correct whenever

```
sqrt(G_v*B_v) + 2*eta*D < g_v
```

for every wrong class, where `D` is the actual final RMS denominator and
`eta` bounds each executed logit's error. A separate theorem supplies the
Euclidean baseline. Another permits selecting a different valid upper bound
for each competitor: every candidate must bound that contrast error before
selection. This justifies the diagnostic's fixed scale grid and its mixture
with the Euclidean baseline without using query target errors to select scales.
There is no theorem that AdamW discovers small training variation or transport
energy, nor a claim that every query satisfies the strict budget.

### TT4. Transfer a training label along a task-preserving route

In `ZMod 97`, each `u` in `{1,35,61}` satisfies `u^3=1`. The route

```
(a,b) -> (u*a+t, u*b-t)
```

therefore preserves `(a+b)^3`. Conversely, a checked sum relation
`c+d=u*(a+b)` supplies the route with `t=c-u*a`. A theorem composes this
identity with an observed training label. The numerical constructor groups
equal sums, or sums in the same three-element orbit, and checks an explicit
allowed multiplier and translation for every member. Groups without training
anchors abstain. The constructor does not cube query sums to obtain labels.
Task symmetry is known here; no equivariance of the trained network is assumed.

### TT5. Concrete witnesses and practical scope

Take two real training states `(1,1),(1,-1)`, center `(1,0)`, query
`(9/8,1/4)`, contrast weights `(2,0)`, coefficients `(1/8,-1/8)` and
`s=1/2`. The residual is `(1/8,0)`, and `G*B=3/32`, so its square root
is strictly below the center margin two. Lean proves both the nonzero residual
and the positive budget. The source theorem also has CP5's exact-center witness.
That witness remains a residual-path example, not a learned active product.

The [native diagnostic](../experiments/contrast_transport/README.md) constructs
and hashes all candidate labels and bounds before a separate audit accesses
query labels and logits. Positive FP64 budgets are useful evidence about these
saved states, but they do not constitute verified backend arithmetic or an
independent statistical test. The source-specific Lean bridge covers plain
complex product; the generic linear-readout inequalities apply to all four arms.

| Claim | Lean declaration |
| --- | --- |
| Center-displacement baseline | `FeatureCodes.center_contrast_error_le` |
| Exact training/residual decomposition | `FeatureCodes.linearScore_transport` |
| Weighted Cauchy–Schwarz and query error | `FeatureCodes.weighted_transport_sq_le`, `training_transport_error_le` |
| Necessary residual and nonzero positive witness | `FeatureCodes.residual_penalty_necessary`, `training_transport_nonzero_witness` |
| Exact cubic route and observed-label transfer | `FeatureCodes.cubic97_route`, `cubic97_route_of_sum`, `cubic97_transfer_label` |
| Actual source contrast identity | `ComplexProduct.linearScore_contrastRow` |
| Source decision from valid bound choices | `ComplexProduct.executed_correct_of_center_bounds` |
| Euclidean and training-transport source budgets | `ComplexProduct.executed_correct_of_euclidean_center`, `executed_correct_of_training_transport` |
| Source witness | `ComplexProduct.copy_training_transport_classifies` |

## Loss-controlled geometry

Added 2026-11-08, task #181. TT1–TT5 bound query contrasts from training
geometry but do not say that the objective reduces raw state or contrast
scatter. Before assuming such a mechanism, distinguish the quantity that
small cross-entropy actually controls. The results below apply to the real
finite-class CE used by the source models, including all competing labels.

### LG1. The objective controls missing target probability

For arbitrary finite logits `z`, write

```
p_c = exp(z_c) / sum_j exp(z_j)
ell(z,t) = log(sum_j exp(z_j)) - z_t = -log(p_t).
```

A target index witnesses that the denominator is positive. Thus every `p_c`
is positive, each is at most one, and the probabilities sum to one.
The elementary inequality `log(x) <= x-1` for `x>0` yields

```
1-p_t <= ell(z,t).
```

This is a theorem about the actual objective, with no correct-classification,
large-margin, optimizer or feature-collapse premise. The finite softmax in
`Geometry.probability` agrees with the source's log-sum-exp CE definition in
`Landscape.Neural.Loss`.

### LG2. Small loss controls squared probability error

Let `e_t` be the one-hot target vector. Since `sum_c p_c^2 <= sum_c p_c = 1`,

```
||p-e_t||_2^2 = sum_c p_c^2 + 1 - 2*p_t
              <= 2*(1-p_t) <= 2*ell(z,t).
```

Lean derives this from the finite sums. The constant two is a convenient
universal bound; no sharpness is claimed. This does not bound raw logits or
hidden states: softmax is nonlinear and does not have a uniform inverse
Lipschitz constant in the high-confidence regime.

### LG3. Training probability scatter follows from training loss

For a finite same-answer group of rows `S`, define the computed probability
mean `pbar = sum_{i in S} p_i / |S|`. Coordinatewise least-squares optimality
of that mean, already proved by `Geometry.mean_minimizes_error`, gives

```
sum_i ||p_i-pbar||_2^2 <= sum_i ||p_i-e_t||_2^2 <= 2*sum_i ell_i.
```

Consequently mean CE at most `epsilon` implies mean within-group probability
scatter at most `2*epsilon`. Every same-answer training group obeys this;
summing the group inequalities gives the corresponding weighted split bound.
The exact centering identity also separates the two terms:

```
mean_i ||p_i-e_t||_2^2 = mean_i ||p_i-pbar||_2^2 + ||pbar-e_t||_2^2.
```

The mean could be confidently wrong if only its within-group scatter were
examined. CE controls the error to the correct target as well. These statements
apply to exactly the rows where CE is measured. Small training CE gives no
held-out loss bound. A decreasing loss lowers this upper bound; it need not
make the actual scatter decrease at every step. No optimizer convergence rate
or existence of a small-loss iterate is asserted.

### LG4. Actual shared-parameter descent can expand raw contrasts

Consider two same-label binary examples with logits `(theta,0)` and
`(2*theta,0)`, both targeting the first class. Their one shared trainable
scalar has the actual objective

```
F(theta) = (CE(theta,0) + CE(2*theta,0))/2
F'(theta) = ((sigmoid(theta)-1) + 2*(sigmoid(2*theta)-1))/2 < 0.
```

Their raw class contrasts are `theta` and `2*theta`; subtracting the actual
mean `3*theta/2` leaves total centered energy `theta^2/2`. For every `eta>0`
and `theta>=0`, the genuine gradient step
`theta_next = theta - eta*F'(theta)` increases theta, strictly lowers F and
strictly increases the raw contrast energy. Lean proves the derivative and
strict decrease rather than inserting a prescribed gradient in a definition.
There is no step-size restriction here because this particular scalar loss
is strictly decreasing on the entire real line.

This refutes unconditional raw-contrast contraction from CE descent, even with
shared parameters and no label conflict. It is a two-example classifier,
not a claim that the native transformer realizes this exact scalar trajectory.
The normalized source has additional parameter coupling and state geometry;
those must be established separately to explain its training dynamics.

### LG5. Fixed-cohort comparison and remaining obligation

The [native diagnostic](../experiments/loss_geometry/README.md) measures both
geometries on all 24 original saved states, grouping by equal sums or cubic
answer orbits. Training and validation means are computed independently.
The protocol fixes the cohort, metrics and numerical tolerances before the
measurement. Adjacent checkpoints are finite training intervals, not gradient
steps; their geometry changes cannot identify a causal optimizer mechanism.

This extension establishes which output geometry follows from the observed
objective. It does not upgrade TT3 to a generalization theorem: its raw contrast
transport terms still need control, or a separate probability-space transfer
bound must connect unobserved inputs to training evidence. Validation versions
of LG1–LG3 use validation CE and are retrospective checks only. All numerical
calculations are FP64 evidence, not certified execution arithmetic.

| Claim | Lean declaration |
| --- | --- |
| Probability simplex and actual CE identity | `Geometry.probability_pos`, `sum_probability`, `probability_le_one`, `crossEntropy_eq_neg_log_probability` |
| Objective controls missing probability and one-hot error | `Geometry.missing_probability_le_loss`, `probability_error_le_twice_loss` |
| Group scatter and its loss ceiling | `Geometry.probability_scatter_le_twice_loss`, `mean_probability_scatter_le` |
| Actual shared-scalar derivative and sign | `GradientEvidence.sharedContrastLoss_hasDerivAt`, `sharedContrastLoss_deriv_neg` |
| Strict loss decrease and computed raw scatter | `GradientEvidence.sharedContrastLoss_strictAnti`, `sharedContrast_scatter` |
| Actual descent with expanding raw contrasts | `GradientEvidence.descent_increases_contrast_scatter` |

## Probability transfer

Added 2026-11-08, task #182. LG1–LG4 bound probability geometry only on rows
whose loss is known. The next step is a pointwise transfer inequality with
training probabilities and unlabeled query-state geometry on its right side.
It remains conditional on the measured query geometry; it is not a theorem
that training reaches small loss or that future inputs have small distance.

### PT1. Express confidence through competitor odds

For finite logits `z` and target `t`, ordinary softmax gives

```
p_c/p_t = exp(z_c-z_t)
R_true = sum_(c != t) exp(z_c-z_t)
p_t = 1/(1+R_true),       CE(z,t) = log(1+R_true).
```

All competing classes are included. Lean derives these identities from the
same `Geometry.probability` and actual CE definition used in LG1. A target
index ensures a nonempty class set; no finite logits have zero softmax
denominator. A one-class task has zero wrong odds and is covered by the
identities, although the numerical experiment uses all 97 classes.

### PT2. Valid envelopes transfer probability and loss

Suppose `z_c-z_t <= U_c` for every wrong class. Define
`R = sum_(c != t) exp(U_c)`. Monotonicity of exp, reciprocal and log gives

```
p_t >= 1/(1+R),       CE(z,t) <= log(1+R).
```

Average the latter over any declared finite query set to obtain a mean CE
ceiling; every query stays in the average, including failed classification
conditions. These are proved pointwise and finite-mean inequalities, not a
statistical confidence interval or a claim about an unseen distribution.

The condition `R<1` implies target probability above one half and a strict
target winner. The weaker condition `U_c<0` for every competitor also proves
the target wins, but need not put more than half the probability mass there.
Both decision implications are proved and reported separately.

### PT3. Derive the envelope from normalized state geometry

Let `E_c` be the actual tied readout rows, `h_i` normalized training states,
and `h` the normalized query state. For a selected training reference `i(c)`,
put

```
d_i = ||h-h_i||_2
L_c = ||E_c-E_t||_2
U_c = log(p_i(c),c / p_i(c),t) + d_i(c)*L_c.
```

The training log odds are evaluated stably as training logit differences.
The earlier Cauchy–Schwarz contrast bound proves

```
z_c(h)-z_t(h) <= z_c(h_i)-z_t(h_i) + d_i*L_c = U_c.
```

Thus the probability and loss bounds are derived from actual training scores,
readout weights and state distances. They do not require a measured query
target error or a query class-score evaluation on the right side. Different
competitors may select different valid references. Taking a finite minimum
of such valid bounds uses no query targets and preserves the inequality.

A computed mean of normalized training states is also a valid reference.
Its linear logits are the means of training logits, so its log odds are
mean training log odds. Its softmax probabilities are not the arithmetic
means of training probabilities. The generic geometric theorem allows this
reference without claiming that the center is itself a network input.

### PT4. Actual source normalization and numerical execution

For the plain complex-product source, the reference coordinates are
`realify(state_i)/D_i` and the query coordinates are `realify(state)/D`, with
each actual final RMS denominator retained. Lean proves that linear readout
of these coordinates equals the independently expanded source's real logits.
This includes all four original normalizations; replacing the denominators
by a fixed radius or by a mean denominator is not part of the result.

If every executed query logit differs from the exact source by at most `eta`,
replace each `U_c` by `U_c+2*eta`. The source probability and CE theorems
retain that hypothesis explicitly. For positive `R`, the probability
condition has available error budget `eta < -log(R)/2`, because the total
bounded wrong odds become `exp(2*eta)*R`. The diagnostic initially sets eta
to zero and reports this allowance; it does not establish backend arithmetic
accuracy or verify the saved floating-point computation in Lean.

Task symmetry supplies the candidate answer separately. The existing cubic
routes `(a,b) -> (u*a+t,u*b-t)`, `u in {1,35,61}`, transfer an observed
training label. No equivariance of the learned network is assumed. The
generic linear-readout result applies to every numerical arm; the source
bridge is specific to the formalized plain complex product.

### PT5. Witnesses, fixed diagnostic and limits

In the odds witness the training target logit is `log 4` with other logit
zero; the query target logit drops by `log 2`. The wrong-class odds envelope
is exactly `1/2`, giving target probability at least `2/3` and a strict winner.
The geometric witness uses scalar readout rows two and zero, training state
two and query state `3/2`. Its nonzero distance gives relative-logit bound
`-3` and total wrong odds below one. Actual-source examples instantiate the
bridge with nonzero complex branch parameters and distinct input pairs.

The [fixed Rust diagnostic](../experiments/probability_transfer/README.md)
uses all 24 original checkpoints. It constructs both same-sum and cubic-orbit
bounds from training labels/scores and normalized query states, hashes them,
then separately audits query labels, probabilities, losses and logits.
The model/oracle evaluator does compute logits separately; the constructor
does not receive them. Training queries exclude themselves from both the
anchor family and center. Missing references cause abstention.

The protocol compares the training center, nearest anchor, best complete
anchor, per-competitor anchor envelope, and the envelope including the center.
All methods and failed conditions remain in the report. Query evaluation and
known task symmetry are required. Conditional correctness at these saved
states does not imply an optimizer-discovery theorem, independent test-set
coverage, or a priori generalization without running the query model.

| Claim | Lean declaration |
| --- | --- |
| Exact probability ratios and odds identities | `Geometry.probability_ratio`, `probability_eq_one_div_odds`, `crossEntropy_eq_log_odds` |
| Envelope nonnegativity and comparison | `Geometry.oddsEnvelope_nonneg`, `odds_le_envelope` |
| Query probability floor and CE ceiling | `Geometry.probability_ge_of_contrast_bounds`, `crossEntropy_le_of_contrast_bounds` |
| Mean query loss ceiling | `Geometry.meanCrossEntropy_le_of_contrast_bounds` |
| Stronger probability and weaker argmax conditions | `Geometry.strict_winner_of_odds_envelope`, `strict_winner_of_negative_contrast_bounds` |
| Nonzero odds witness | `Geometry.nonzero_probability_transfer_witness` |
| Training-center linearity and geometric bound | `FeatureCodes.linearScore_mean`, `relative_logit_le_anchor` |
| Probability and loss from training-state geometry | `FeatureCodes.probability_ge_anchor_bound`, `crossEntropy_le_anchor_bound` |
| Nonzero geometric witness | `FeatureCodes.nonzero_anchor_geometry_witness` |
| Exact normalized source readout | `ComplexProduct.readoutState_logit` |
| Execution allowance and source bounds | `ComplexProduct.executed_relative_le_anchor`, `executed_probability_ge_anchor`, `executed_crossEntropy_le_anchor` |

## Signed directional transport

Added 2026-11-08, task #183. PT3 uses a norm penalty for the entire displacement
from a training reference. TT1 already identifies an exact signed contribution
from reconstructed training directions. This extension preserves that term
and bounds only the remaining state/readout interaction. It proves a pointwise
bound, not a claim that training discovers a useful low-dimensional subspace.

### ST1. Preserve the signed training contribution

For normalized training states `T_i`, a center `mu`, query `x`, readout contrast
`w=E_c-E_t` and coefficients `a_i`, define

```
r = x - mu - sum_i a_i*(T_i-mu)
prediction = <w,mu> + sum_i a_i*(<w,T_i>-<w,mu>)
<w,x> = prediction + <w,r>.
```

The equality follows from TT1's exact transport decomposition. The coefficients
may have either sign, and reconstruction may extrapolate beyond the training
states. No convexity assumption or omitted residual is introduced. In the
diagnostic `mu` is the mean of normalized training states, so linearity gives
mean training log odds for `<w,mu>`; the weighted terms use only training
log-odds deviations. The baseline bound is

```
abs(<w,x> - prediction) <= ||r||_2 * ||w||_2.
```

Unlike TT2's covariance bound, the reconstructed training contribution is kept
signed. It is not replaced by an upper bound on its absolute value.

### ST2. Separate the remaining readout directions

For any finite vectors `B_j` and coefficients `b_j`, let

```
u = w - sum_j b_j*B_j
<w,r> = <u,r> + sum_j b_j*<B_j,r>
abs(<w,r>) <= ||r||_2*||u||_2 + sum_j abs(b_j)*abs(<B_j,r>).
```

Lean proves the decomposition and bound without assuming the vectors are
orthonormal. The second term is an explicit geometric orthogonality defect.
If every `<B_j,r>=0`, the two residual norms suffice, as a separate theorem
shows. This hypothesis cannot silently be imposed on floating-point SVD.
The counterexample `B=r=w=b=1` has zero readout residual but interaction one,
so omitting the defect without orthogonality is false even for a unit basis.

The theorem allows arbitrary vectors; the diagnostic restricts `B` to a basis
computed solely from centered training states, without query states or readout
weights. It uses `b_j=<B_j,w>`. The constructor computes query scores against
this training basis, but never the original query class readout. The remaining
readout norm is computed from weights and the training basis.

### ST3. Transfer probabilities and retain actual source execution

Set `U_c=prediction_c+residual_bound_c` for each wrong class. ST1–ST2 give
`z_c(x)-z_t(x)<=U_c`; PT2 then supplies

```
R = sum_(c!=t) exp(U_c)
p_t >= 1/(1+R),       CE <= log(1+R).
```

The same finite-mean theorem includes every query, including failed decision
conditions. `R<1` and `max U_c<0` retain their distinct probability-majority
and argmax meanings. Task-preserving cubic routes supply the candidate answer
from observed training labels; no learned equivariance is assumed.

The plain-product source bridge uses its actual normalized `readoutState`,
including all four RMS normalizations. With per-logit execution error at most
`eta`, each bound increases by `2*eta`. The numerical diagnostic starts at
eta zero and reports an available allowance. It does not prove an FP64 error
bound for its decomposition, training readouts or source evaluator.

### ST4. Rank control and the full-reconstruction limit

The [frozen native protocol](../experiments/signed_transport/PROTOCOL.md) uses
thin SVD of centered normalized training states, with singular cutoff
`1e-10*s_max` and fixed ranks `0,1,2,4,8,16,32,64,128`. Each rank is clamped to
the available training rank. It uses the state projection to construct signed
training coefficients, then explicitly recomputes the residual from those
coefficients and training states. No exact SVD identity is assumed by the bound.

Rank zero gives the previous normalized training-center baseline. At the other
extreme, Lean proves `r=0` implies the training prediction equals the original
score. Full-dimensional reconstruction can therefore re-express the source's
classification without explaining its learning or compression. The diagnostic
retains every rank; successful full-rank reconstruction must not be reported
as evidence of a useful low-rank representation. Finite training rank and the
cutoff can also leave a residual at the requested ambient rank 128.

Validation queries use all training references in their group. Training queries
use the opposite half of alternating sorted group training indices, avoiding
self reconstruction. Their reference sets differ from validation's. Both
complete constructions are hashed before a separate target/score audit.

### ST5. Nonzero witness and scope

Take normalized-coordinate training states `(1/2,0)` and `(3/2,0)`, center
`(1,0)`, query `(3,1/4)`, contrast `w=(-2,1)`, and coefficients `(-2,2)`.
The signed prediction is `-6`; the residual `(0,1/4)` is genuinely nonzero.
The training basis `(1,0)` with readout coefficient `-2` leaves readout residual
`(0,1)`. Its defect is zero, and the upper contrast bound is exactly `-23/4`.
Both observed training contrasts are negative. Lean proves the nonzero residual
and exact upper bound; the native test also checks the resulting probability.

Known task symmetry and query evaluation remain necessary. A successful saved
state bound is neither optimizer convergence nor an independent statistical
generalization result. The generic linear-readout bound applies to every native
arm; the formal source bridge is specific to the plain complex product.

| Claim | Lean declaration |
| --- | --- |
| Exact signed reconstruction and baseline error | `FeatureCodes.linearScore_signed_residual`, `signed_transport_norm_le` |
| Readout residual identity and directional bound | `FeatureCodes.linearScore_readout_residual`, `directional_residual_abs_le` |
| Signed transport error and upper bound | `FeatureCodes.signed_transport_abs_le`, `signed_transport_upper` |
| Zero-residual re-expression limit | `FeatureCodes.exact_transport_of_zero_residual` |
| Orthogonal specialization and necessary defect | `FeatureCodes.orthogonal_residual_abs_le`, `orthogonality_defect_necessary` |
| Relative logit, probability and loss transfer | `FeatureCodes.relative_logit_le_signed`, `probability_ge_signed_bound`, `crossEntropy_le_signed_bound` |
| Nonzero extrapolation witness | `FeatureCodes.signed_transport_nonzero_witness` |
| Actual source with execution allowance | `ComplexProduct.executed_relative_le_signed`, `executed_probability_ge_signed`, `executed_crossEntropy_le_signed` |

## Residual learning dynamics

This project extension, task #184, asks what learning mechanism could reduce
ST2's directional residuals. It derives a fixed-feature readout result and
separately checks the limitations imposed by adaptive updates and moving
features. Complex multiplication is unchanged.

### RD1. Differentiate the actual readout objective

For normalized training states T_i, labels y_i, readout rows E_c and
L(E) = mean_i CE((<E_c,T_i>)_c,y_i), differentiate a scalar perturbation of
each coordinate of E. The exact result is

```
G_c = mean_i (p_i,c - 1[c=y_i]) T_i.
```

Lean defines G using the actual derivative and proves this equality; it does
not assume an update direction. This is the partial final-readout derivative
with states frozen. When the embedding is tied to the encoder, backpropagation
has another path. Neither the full tied gradient nor AdamW's update is G.
The line-derivative proof follows the local finite-logit calculation in
`Imbalance.hasDerivAt_crossEntropy_line` (arXiv:2402.19449v2, §3.2).

### RD2. Bound gradient forcing using training CE

For any fixed linear map Q, Cauchy–Schwarz and LG1's actual probability-error
inequality give

```
||QG||_F² <= mean_i sum_c (p_i,c - 1[c=y_i])² * mean_i ||QT_i||²
          <= 2 L(E) * mean_i ||QT_i||².
```

All training rows enter both means. Choosing Q from an answer group's
centered PCA does not remove other groups from the feature factor. No
orthogonality assumption on Q is needed for this inequality. A small loss
controls gradient forcing relative to projected feature scale, not the
current readout residual, an adaptive normalized direction, or future drift.

### RD3. The sufficient Euclidean descent mechanism

For fixed T, consider the explicitly defined Euclidean update with decay
E+ = (1-eta*rho)E - eta*G. Linearity gives the exact projected update.
If Q annihilates **every** training feature, QG=0, and induction yields
QE_n=(1-eta*rho)^n QE_0. With 0<eta*rho<1 the residual decays;
rho=0 gives invariance. The theorem allows arbitrary rates, so its identity
does not assert contraction outside the stated rate range.

For eta>=0, rho>0 and eta*rho<=1, the scalar square identity

```
(1-eta*rho)u² + (eta/rho)g² - ((1-eta*rho)u-eta*g)²
  = (eta/rho)(1-eta*rho)(rho*u+g)² >= 0
```

and RD2 imply

```
||QE+||_F² <= (1-eta*rho)||QE||_F²
             + (2eta/rho)L(E) mean_i ||QT_i||².
```

This is a derived decay-plus-forcing estimate. Small forcing relative to
the current energy is a sufficient one-step contraction condition. It is
not assumed to hold for the saved models. Concrete Lean examples use
nonzero training features, nonzero initial residuals and positive rates.

### RD4. Account for movement of the learned subspace

For arbitrary old/new residual maps and vectors, linearity gives

```
Q1 w1 = Q0 w0 + Q0(w1-w0) + (Q1-Q0)w1.
```

The corresponding energy difference can be split exactly by inserting
||Q0 w1||²: first change w with Q frozen, then change Q at the new w.
The same identity applies to readout contrasts and normalized query
displacements h-mu. The frozen native protocol retains both terms for all
adjacent saved intervals. These are retrospective changes over 5000 steps,
not an individual update or an identified causal effect of weight decay.

### RD5. An actual adaptive step can create a nullspace residual

One fixed feature is (1,2), with binary logits (u+2v,0), first-class target,
and zero initial readout. Differentiating actual CE gives (-1/2,-1).
Euclidean descent yields (eta/2,eta), which stays in the feature span.
The existing bias-corrected Adam first-update formula, with betas 0.9/0.999,
epsilon 1 and decay 0, instead gives (eta/3,eta/2). Its squared orthogonal
residual is

```
(2u-v)²/5 = eta²/180 > 0  for eta>0.
```

The correct margin strictly increases, so CE strictly decreases. Lean proves
the actual derivative, Adam formula, residual projection identity and strict
loss decrease. Epsilon is positive and deliberately fixed to one for this
counterexample; this is not the checkpoint training recipe or a replay of
that trajectory. It refutes transferring the Euclidean feature-span
invariance/decay argument unchanged to coordinatewise adaptive updates.

| Claim | Lean declaration in `GradientEvidence` |
| --- | --- |
| Actual coordinate derivative | `readoutLoss_coordinate_hasDerivAt`, `readoutGradient_eq` |
| Objective-controlled projected gradient | `projected_readoutGradient_energy_le` |
| Annihilated directions and iterated decay | `projected_gradient_zero`, `projected_readoutRun` |
| Derived decay and loss-controlled forcing | `decay_forcing_sq_le`, `readout_step_energy_le` |
| Changing weights and basis | `moving_residual_identity` |
| Actual adaptive counterexample | `axisReadoutGradient_initial`, `axisAdamStep_eq`, `adam_creates_readout_residual` |

## Adaptive interaction control

Task #185 extends RD1–RD5 to coordinate-adaptive forcing and moving residual
interactions. This is a sufficient-control theory with explicit missing terms;
it does not conclude that the optimizer makes those terms small.

### AI1. Retain the actual buffered Adam direction

The source is the local Rust/Kernite rotate90_complex implementation at source
hash 14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da:
`engine.rs::update`, `product.rs::moment`, `ops.rs::adamw`, and the buffer
layout in `storage.rs::state_bytes/decode`. The checked cohort uses coordinate
AdamW, betas 0.9/0.98, epsilon 1e-8, decay 1 and rate 0.001 after ten warmup
updates. Embedding coordinates are independent in all four arms; the complex
projection constraints do not change this embedding update.

Let m_t,v_t denote the buffers *after* incorporating update t's gradient.
In exact real arithmetic the source factors as

```
M_t = m_t/(1-beta1^t)
D_t,k = 1/(sqrt(v_t,k)/sqrt(1-beta2^t)+epsilon)
w_new = (1-eta*rho) w_old - eta D_t M_t.
```

Lean proves the factorization and projected update without identifying M_t
with a current reference gradient. The source uses t>=1 and nonnegative
second moments; the initial checkpoint has no historical bias-corrected
direction. The native audit retains that case with an explicit unavailable
status. It evaluates real-form equations in FP64 on saved FP32 buffers, not
a bitwise replay of the FP32 CUDA kernel. In a trained checkpoint, the
historical moments summarize minibatch gradients at earlier parameters.

### AI2. Coordinate scaling does not preserve an arbitrary subspace

For a residual map Q and diagonal scale D, define

```
[Q,D]g = Q(Dg) - D(Qg)
([Q,D]g)_k = sum_l Q_kl (D_l-D_k) g_l.
```

Constant D commutes with every Q. For any reference gradient G and moment M,

```
QDM = DQG + [Q,D]G + QD(M-G).
```

With RD1's actual current full-training partial readout derivative G, the
three terms isolate scaled projected CE forcing, diagonal mixing, and the
combined discrepancy from the saved moment. The last term includes encoder,
minibatch, temporal and momentum effects; the decomposition does not identify
them separately. In particular M-G is not just a momentum error or the current
encoder gradient. The historical direction is not a predicted next update.

The RD5 example makes the mixing term concrete: Q is orthogonal projection
perpendicular to (1,2), G=(-1/2,-1), D=(2/3,1/2). Then QG=0 but [Q,D]G has
squared norm 1/180. Lean proves this directly. The native test also checks
zero commutator under a constant diagonal scale.

### AI3. Bound the part controlled by CE and keep the remainder

For class-dependent D_c,k with D_c,k²<=dmax², finite sums give

```
||D Q G||_F² <= dmax² ||QG||_F²
             <= dmax² 2 L_train mean_all_training ||Qh_i||².
```

This uses every training row, as RD2 requires. Coordinatewise
(a+b+c)²<=3(a²+b²+c²) gives the sufficient adaptive-forcing bound

```
||QDM||_F² <= 3*(dmax² 2 L_train mean_all_training ||Qh_i||²
                 + ||[Q,D]G||_F² + ||QD(M-G)||_F²).
```

No commutator or discrepancy is assumed zero. Large inverse scales can
amplify the CE contribution, and cross terms can make the sum of individual
energies very conservative. The native audit retains those cross terms,
the direct three-term bound and the objective-based bound separately.

### AI4. Control the signed interaction, not just separate energies

Let J=<u,r>, where u=Qw and r=Qx; x is normalized query displacement
relative to its training center. For arbitrary new vectors define
du=u_new-u and dr=r_new-r. Bilinearity yields

```
J_new-J = <du,r> + <u,dr> + <du,dr>
|J_new-J| <= ||du||||r|| + ||u||||dr|| + ||du||||dr||.
```

The mixed term is retained. An old bound |J|<=B transfers to the new state
with this explicit drift allowance. This is a retrospective finite-change
statement; it does not identify a 5000-step checkpoint interval with an update.
Changing Q is already included in du and dr.

A nonzero witness has u=r=(1,0), u_new=(0,2), r_new=(3,0). The interaction
falls from 1 to 0 while the norm product rises from 1 to 6. Signed changes
are -1, +2 and -2. This shows why tracking signs and cancellation can be
more informative than requiring both residual norms to contract.

For computed Q, <Qw,Qx> differs from the source residual <w,Qx> by a
projection defect unless the required projection identity holds. The audit
retains that defect. Neither interaction alone replaces ST2's certificate.

### AI5. A sufficient condition for an actual moving adaptive update

Write a=1-eta*rho, f=DM, w'=a w-eta f, u=Qw, r=Qx, u'=Q'w', r'=Q'x'.
This buffered direction describes one readout row. For a class contrast
w=E_c-E_target, use f=D_c M_c-D_target M_target: the two preconditioners
need not coincide. Lean proves this actual contrast update and its moving
interaction bound separately; it never replaces f by D*(M_c-M_target)
without a shared-scale assumption.
The readout residual satisfies

```
u' = a*u - eta*Qf + b,          b=(Q'-Q)w'
<u',r'> = a<u,r> - eta<Qf,r> + <b,r> + <u',r'-r>.
```

Consequently,

```
|<u',r'>| <= |a| |<u,r>| + Gamma
Gamma = |eta| ||Qf||||r|| + ||b||||r|| + ||u'||||r'-r||.
```

Lean derives this bound, instantiates it with the actual buffered Adam
equations, and proves strict contraction when
Gamma < (1-|a|)|<u,r>|. The hypothesis controls concrete forcing and drift
terms, not the desired new interaction. A concrete example has Q=Q'=1,
a=eta=1/2, w=4, f=1, x=x'=3: nonzero forcing and residuals satisfy it.
AI3 supplies control for one part of Qf; it supplies no control over the
other two parts or over feature/basis movement. The condition is therefore
sufficient, not a claim of optimizer discovery or a consequence of CE alone.

| Claim | Lean declaration in `GradientEvidence` |
| --- | --- |
| Source update and projection | `bufferedAdamUpdate_factor`, `projected_bufferedAdamUpdate` |
| Mixing and three forcing terms | `diagonalCommutator_eq_sum`, `adaptive_forcing_decomposition`, `diagonal_mixing_witness` |
| Loss-controlled contribution and full bound | `scaled_readoutGradient_energy_le`, `adaptive_forcing_energy_le` |
| Exact interaction change and drift control | `interaction_change`, `interaction_change_abs_le`, `interaction_bound_transport` |
| Cancellation witness | `interaction_cancellation_witness` |
| Actual update with moving geometry | `moving_update_interaction`, `buffered_interaction_le` |
| Row-specific preconditioners in actual class contrasts | `buffered_contrast_eq_directionStep`, `buffered_contrast_interaction_le` |
| Sufficient strict contraction | `moving_update_interaction_contracts` |

## Actual single-update evidence

Task #186 implements the [fixed native audit](../experiments/single_update/README.md)
of AI1–AI5 on all 24 original checkpoints. Each next scheduled AdamW update
replays bitwise against the pinned source trainer. All 48 independent scalar
directional derivative checks and 9360 projected decompositions pass; the
original 73 input files remain unchanged. Eight endpoint probes extend the
sampler by one post-budget step and do not change the original ranking.

At the old parameters let R be the full-training readout embedding gradient,
F the full-training total embedding gradient and B the next-minibatch total
gradient. M and D are taken from the new moment buffers, with bias correction.
The measured decomposition is

```
M = R + (F-R) + (B-F) + (M-B)
QDM = DQR + (QDR-DQR) + QD(F-R) + QD(B-F) + QD(M-B).
```

This distinguishes the tied encoder, sampling and moment-memory terms
without treating them as independent causal interventions. It retains each
class row's own D before subtraction. For the plain-product post-budget
probes at rank-32 orbit bases, individual forcing energies sum to
57923.68/4407.75, whereas actual energies are 3786.84/885.70. Cross terms
contribute -54136.84/-3522.05. These sums run over 33 different fitted bases;
they are not a single global model energy. A sum of separate norm bounds
discards substantial cancellation.

For actual FP32 movement write `w'=a*w-eta*f+rho`, retaining the measured
rounding remainder rho. The audit evaluates AI5's interaction identity with
that extra projected remainder, and further splits
`r'-r = Q(x'-x)+(Q'-Q)x'`. Both x and x' are normalized, training-centered
feature displacements. Feature and basis contributions can be much larger
than individual optimizer terms, but their absolute magnitudes do not add.

The margin diagnostic keeps the signed training-affine reconstruction
`s=<w,mu+(I-Q)(x-mu)>`, interaction `J=<Qw,Q(x-mu)>`, projection defect
`e=<w,Q(x-mu)>-J` and measured executed-logit discrepancy. Each executed
competitor gap is reconstructed as `s+J+e+execution`. A sufficient condition
uses an absolute residual/defect/execution allowance for every competitor.
This is a retrospective measurement of the actual interaction, not a new
training-only certificate or an improvement to ST's norm-bound coverage.

Both plain-product endpoint probes increase mean validation margin, while
validation CE rises and correctness stays 4689/4705 and 4690/4705. On
1082/1163 queries margin improves despite a growing maximum residual budget.
Across the cohort, 17/24 probes lower training CE and five of those increase
rank-32 orbit validation mean absolute interaction; among trained states
the counts are 9/16 and 1/9. Five trained probes improve mean validation
margin while worsening validation CE.

The remaining theoretical obligation is control of worst competitor margins
and difficult-query losses under joint forcing and moving features/bases.
Residual contraction and average-margin improvement alone do not supply it.
This evidence motivates that extension; it introduces no additional Lean
claim of convergence or discovery and leaves complex multiplication unchanged.

## Joint difficult-query control

Task #187 extends AI's interaction analysis to the actual competing-class
score gaps and individual query losses. Source equations are the native
coordinate AdamW update at 14cff3b5 and the finite-class softmax CE in
`Transformer.Landscape.Neural.Loss`. The mathematical statements below are
exact-real identities and sufficient bounds; their numerical audit retains
explicit parameter rounding and executed-logit discrepancies.

### JC1. Keep signed parameter and feature changes together

For a competitor-minus-target readout w and a normalized feature x, write
the new readout as `w'=w+p+rho` and the new feature as `x'=x+dx`. The intended
increment p and rounding remainder rho are distinct. Then

```
L = <p,x> + <w,dx>
gap' - gap = L + <p,dx> + <rho,x'> + (e'-e)
R = ||p|| ||dx|| + ||rho|| ||x'|| + |e'-e|
gap + L - R <= gap' <= gap + L + R.
```

The two first-order terms are combined before taking any absolute value.
Cauchy–Schwarz bounds the mixed term and rounding term; a triangle inequality
adds the explicit execution change. The separated allowance
`S=||p||||x||+||w||||dx||+R` also bounds the full change in both directions.
The signed upper allowance L+R is at most S. This does not assert small
feature movement or small forcing as a consequence of training CE.

The nonzero scalar witness w=2, p=1, x=2, dx=-1 has L=0, R=1 and S=5.
The score changes by -1 through the mixed term. Thus omitting that term is
incorrect even when the two first-order contributions cancel exactly.

### JC2. Control every competing class under the actual update

For each competitor c and target t, use w=E_c-E_t and its own increment
contrast. Define `U_c=gap_c+L_c+R_c` and `V_c=gap_c+L_c-R_c` from JC1.
Every new gap lies in `[V_c,U_c]`. Hence if all U_c are at most -m, the
new target has margin at least m against every competitor. In particular,
all U_c<0 suffice for a strict correct prediction. Checking only the old
worst competitor would miss changes in its identity.

The source Adam increment is proved to be

```
p_c = -eta * (decay*E_c + D_c*M_c)
p_c-p_t = -eta * (decay*(E_c-E_t) + D_c*M_c-D_t*M_t).
```

M and D come from the already updated moment buffers, retaining bias
correction and epsilon. Each class has its own diagonal scale. Adding p
reconstructs `bufferedAdamUpdate`; adding rho connects the joint bound to
the actual rounded parameters. A concrete margin witness has nonzero
moments, decay, parameter movement and feature movement, with different
preconditioners for the two class rows.

### JC3. Control actual query loss with old probability weights

For arbitrary old/new logits z,z', define
`delta_c=(z'_c-z'_t)-(z_c-z_t)` and the actual old softmax p_c. Lean proves

```
CE(z',t)-CE(z,t) = log(sum_c p_c * exp(delta_c)).
```

The target increment is zero. If every delta_c is bounded by B_c, the
same expression with B gives an upper bound on loss change. A weighted
exponential moment at most one is sufficient for loss nonincrease. This
condition permits signed compensation across classes but does not replace
their old probabilities by uniform weights or replace CE by a mean margin.

Taking `B_c=L_c+R_c` instantiates this bound with the actual joint step.
Equivalently, the target contributes its fixed zero gap, and the wrong-class
upper/lower gaps U/V give upper/lower CE bounds through log-sum-exp. Both
directions are proved, allowing detection of worsening as well as improving
query losses. Parameter and feature contributions remain signed inside L.

### JC4. Preserve difficult queries and retain the limitation

Independently valid query loss ceilings average over any fixed subset.
The native protocol freezes the lowest-margin decile of each old split
before evaluating the new state, and retains old errors as another subset.
It does not select difficult rows using the new outcome.

A proved two-query binary counterexample uses old margins `(log 8,log 8)`
and new margins `(log 256,log 2)`. Both new predictions remain strictly
correct and the mean margin increases, but mean actual CE increases:
the old loss product is `(9/8)^2=81/64`, whereas the new one is
`(257/256)*(3/2)=771/512 > 81/64`. The second query's margin deteriorates.
This formally rules out an average-margin argument for loss improvement.

These are conditional local guarantees. The audit measures dx and numerical
execution error after the update. Deriving a useful bound on feature movement
from pre-update training quantities, and validating it on further seeds/tasks,
remain separate obligations. No convergence or optimizer-discovery theorem
is inferred. The new bounds use ambient normalized features and need no PCA
basis, avoiding a dependence on how fitted subspace coordinates move.

| Claim | Lean declarations in `GradientEvidence` |
| --- | --- |
| Exact finite step and joint interval | `score_step_identity`, `score_step_error_bound`, `score_step_interval` |
| Comparison with separated norms | `joint_drift_abs_le`, `score_step_separate_bound`, `joint_upper_le_separate` |
| Nonzero mixed-term witness | `joint_score_cancellation_witness` |
| All-competitor bounds and margins | `gap_step_error_bound`, `new_gap_le_joint_upper`, `joint_lower_le_new_gap`, `all_competitor_margin`, `strict_winner_of_joint_step` |
| Actual Adam and row-specific scales | `adam_increment_update`, `adam_increment_contrast`, `adam_gap_le_joint_upper`, `adam_margin_of_joint_condition` |
| Exact probability-weighted CE change | `crossEntropy_relative_change`, `loss_change_le_relative_bound` |
| Joint query and subset loss bounds | `loss_change_le_joint_step`, `loss_nonincrease_of_joint_step`, `crossEntropy_le_joint_step`, `crossEntropy_ge_joint_step`, `meanCrossEntropy_le_joint_step` |
| Mean-margin counterexample | `mean_margin_loss_counterexample` |

### Native audit of JC1–JC4

The [frozen #187 extension](../experiments/single_update/MARGIN_PROTOCOL.md)
passes all 24 original probes and reproduces every #186 new-state hash.
Across 225816 query updates and 21678336 competing-class pairs, all joint
and separated gap, margin and CE intervals hold, with zero false certificates.
Maximum scaled identity and weighted-CE errors are below 4.39e-15 and
2.79e-15 respectively. The earlier source, derivative and geometry checks
also remain green; original checkpoint inputs are unchanged.

For the 16 trained-state validation probes, joint conditions certify the
direction of CE change on 74677/75280 observations; 603 remain inconclusive.
The fixed old-margin bottom deciles give 7489/7536, with 47 inconclusive.
At both plain-product post-budget endpoints, even the fixed difficult set's
mean margin rises while its mean CE increases (0.172718 → 0.174981 and
0.167636 → 0.168661). Averaging the lower CE bounds certifies both increases.
Thus an average-margin argument remains insufficient after restricting to
difficult queries; JC3's individual probability-weighted control is material.

The [full authored findings](../experiments/single_update/README.md#joint-margins-and-difficult-query-losses)
retain every arm, seed and inconclusive case. The good coverage uses observed
feature displacement and is conditional, not a pre-update prediction. A
useful bound on that displacement from training quantities remains open.

## Feature forecasts

Task #188 replaces JC's observed feature displacement with a forecast made
from the old parameters and the proposed rounded Adam step. The source is
the same native trainer (14cff3b5); the
[frozen protocol](../experiments/single_update/FORECAST_PROTOCOL.md) specifies
every propagation rule and numerical allowance before measuring outcomes.

### FF1. Propagate the old tangent and a finite error ball

At each node let a be the old value, v its old directional derivative along
the proposed parameter step, and r a radius for the unknown true new value
around a+v. The embedding gather is affine, so its radius starts at zero.
Residual addition adds radii. For a linear layer with old matrix A and
increment D, the predicted tangent is Av+Da and

```
(A+D)y - (Aa+Av+Da) = (A+D)(y-a-v) + Dv
radius_out <= ||A+D|| * r + ||Dv||.
```

The mixed term Dv is known before evaluating the future network. Native
code uses the Frobenius norm as a conservative operator bound. For bilinear
products, expand around the two nominal centers, retaining the known product
of tangents, both center/error products and the error/error product. Coordinate
and complex block maximum norms bound those linear factors; the complex
source keeps its 1/sqrt(2) multiplier. No product term is silently discarded.

For a nonlinear map f, a proposed tangent u and an anchored sensitivity L at
a+v give the finite bound

```
||f(y)-(f(a)+u)|| <= L*r + ||f(a+v)-f(a)-u||.
```

The nominal defect on the right is computable from a and v. Its value is
used only in the scalar error budget; the next node receives f(a) and u,
not f(a+v). Thus the center remains the old-network tangent. The certificate
is valid for any u; derivative correctness is independently checked against
central differences of the scalar source oracle.

### FF2. Use anchored nonlinear sensitivities

Define `softNorm(eps,x)=sqrt(||x||^2+eps)` and
`softNormalize(eps,x)=x/softNorm(eps,x)`. Positive epsilon gives

```
|softNorm(eps,y)-softNorm(eps,x)| <= ||y-x||
||softNormalize(eps,y)-softNormalize(eps,x)||
    <= 2/softNorm(eps,x) * ||y-x||.
```

The first bound follows by factoring the difference of squared regularized
lengths and using the reverse triangle inequality. For the second, split
the vector difference and the reciprocal-denominator difference; the
regularized denominator dominates the other vector's norm. Scaling by
sqrt(width) and using eps=width*epsilon gives the source RMS coefficient
2/s(x). Anchor x at the known nominal input, avoiding the global inverse
sqrt(epsilon) factor. This works without an assumed small relative step.

The radial complex gate uses the first bound with two real coordinates.
The sigmoid has global derivative magnitude at most 1/4; its nominal defect
therefore transports incoming error with coefficient 1/4. Real SwiGLU,
complex diagonal, radial product and plain product retain their source
equations and every normalization site.

### FF3. Read out a forecast ball

The proposed rounded readout W is known before future feature evaluation.
If center z has certified feature radius r and each executed readout logit
has an additional error at most delta, then the competitor gap lies within

```
<W_c-W_t,z> +/- (||W_c-W_t||*r + 2*delta).
```

Checking every upper gap controls worst-competitor margins even when that
competitor changes. Log-sum-exp of lower/upper gaps encloses actual query
CE; independently valid upper bounds average over fixed subsets. Errors and
inconclusive conditions remain. These are actual-loss bounds, not an
inference from increasing mean margins.

The native forecast adds a fixed 1e-3 Euclidean feature allowance and 1e-3
per-class readout-logit allowance. They are separately audited numerical
assumptions, not proved CUDA error bounds. Forecasts are persisted before
the candidate network is evaluated. The real graph and its tangent are
checked against source computations; Lean proves the propagation lemmas
and feature-to-outcome bridge, not a verified implementation of the Rust/CUDA
graph. This is local control given a proposed step, not optimizer discovery,
convergence or a statistical generalization theorem.

| Claim | Lean declarations in `GradientEvidence` |
| --- | --- |
| Finite nominal-defect transport | `forecast_transport`, `linear_forecast`, `residual_forecast` |
| Mixed product terms | `product_forecast_identity`, `product_forecast_bound`, `product_forecast_mixed_witness` |
| Source sigmoid | `sigmoid_quarter_lipschitz`, `sigmoid_forecast` |
| Radial and normalization sensitivity | `softNorm_error`, `softNorm_forecast`, `softNormalize_error`, `scaled_softNormalize_error`, `softNormalize_forecast` |
| Explicit execution allowance | `forecast_execution_ball` |
| All-competitor control | `forecast_contrast_error`, `forecast_gap_interval`, `forecast_margin` |
| Actual query/subset loss | `forecast_loss_upper`, `forecast_loss_lower`, `forecast_mean_loss_upper` |

### Native audit of FF1–FF3

All 24 frozen forecasts pass, with unchanged #186 state hashes and original
inputs. All 225816 feature balls and 21678336 competitor bounds enclose the
later outcomes. Fixed FP32 allowances have zero violations; independent
scalar tangent checks have maximum scaled error 5.47e-9. The native report
retains all per-query errors, radii, intervals and frozen feature centers.

Forecast correctness coverage is 68040/75280 (90.38%) on trained validation
observations, but no loss-change direction is certified anywhere in the
24-case cohort. Mean validation margin interval width is 4.947498, versus
0.001595 for JC's retrospective interval. Plain-product endpoint ratios of
mean radius to mean actual forecast error are 109.10 and 196.37. The
[authored findings](../experiments/single_update/README.md#feature-forecasts-before-the-updated-forward-pass)
show all arms and seeds. Future feature measurements are no longer needed
for these bounds, but repeated scalar-ball propagation is too conservative
for loss control. The next obligation is directional error transport through
the actual block and readout, not an architecture change or a fitted allowance.

## Forecast tightening

Task #196 keeps FF1–FF3's old values, directional tangents and explicit mixed
terms. The question is whether sharper, computable error bounds can close the
gap between 90.38% trained-validation certificate coverage and the 99.56%
observed correctness of the same updates. These are different quantities:
neither a certificate failure nor an inconclusive interval is a wrong answer.
The [frozen protocol](../experiments/forecast_tightening/PROTOCOL.md) fixes the
24 source states and all ablations before reading their candidate outcomes.

### FT1. A regularized secant retains radial cancellation

For a real inner-product space, let

\[
s_x=\sqrt{\|x\|^2+\varepsilon},\qquad
F(x)=x/s_x,\qquad \varepsilon>0.
\]

Expanding the two squared distances gives the exact identity

\[
\|F(y)-F(x)\|^2(s_xs_y)^2
 +(s_x-s_y)^2(s_xs_y+\varepsilon)
 =\|y-x\|^2s_xs_y.
\tag{FT1}
\]

Indeed, multiply the normalized difference by the positive product
\(s_xs_y\), expand \(\|s_xy-s_yx\|^2\), and substitute
\(\|x\|^2=s_x^2-\varepsilon\) and its counterpart for y. The second
term is nonnegative, so the secant coefficient is at most
\(1/\sqrt{s_xs_y}\). If a known ball gives \(\|y-x\|\le r<s_x\),
the already proved regularized reverse triangle bound gives
\(s_y\ge s_x-r>0\). Consequently,

\[
\|F(y)-F(x)\|\le {r\over\sqrt{s_x(s_x-r)}}.
\]

For source RMS at width n, set the regularization above to n times the
source epsilon and multiply F by \(\sqrt n\). Use the smaller of this
coefficient and FF2's \(2\sqrt n/s_x\). If \(r\ge s_x\), retain
the old global bound; no unknown future denominator is substituted. At a
forecast center \(x=a+v\), add the unchanged nominal nonlinear defect
\(\|F(a+v)-(F(a)+u)\|\). Computing this defect does not make F(a+v)
the next layer's feature: the center remains the old value plus tangent.
The Lean witnesses include opposite nonzero vectors and a nonzero radius.

### FT2. Gram energy bounds linear amplification

For any finite real rectangular matrix M with its Euclidean operator norm,

\[
\|M\|_{\rm op}
 \le \sqrt{\|M^\top M\|_F}
 =\left(\sum_{ij}(M^\top M)_{ij}^2\right)^{1/4}.
\tag{FT2}
\]

Rowwise Cauchy–Schwarz proves \(\|G\|_{\rm op}\le\|G\|_F\).
The Euclidean C-star identity
\(\|M^\top M\|_{\rm op}=\|M\|_{\rm op}^2\) then proves FT2.
Applying the same argument to the transpose gives the row Gram version.
The implementation chooses the smaller Gram dimension and takes the minimum
with the independent Frobenius upper bound. There is no estimated singular
vector, spectral flatness assumption or trained parameter change.

For the updated matrix M=A+D, FF1's radius becomes

\[
B r+\|Dv\|,\qquad B\ge\|A+D\|_{\rm op}.
\]

The actual mixed vector Dv remains. Lean also proves the conditional
statement that any U above the exact Gram squared Frobenius norm gives the
bound \(U^{1/4}\). This exposes the numerical obligation. The Rust routine
enlarges every FP64 product, sum and square root outward, including signed
Gram dot products; the whole Rust/BLAS/CUDA program is not formally verified.
Nonzero square and rectangular examples inhabit the Lean statements; native
tests also cover identity, rank-one and transposed layouts.

### FT3. Preserve signed contrasts across the final RMS

Let \(\|y-x\|\le r\) describe the feature before the final normalization,
and let w be one updated readout contrast. Cauchy–Schwarz gives
\(a\le\langle w,y\rangle\le b\), where
\(a=\langle w,x\rangle-\|w\|r\) and
\(b=\langle w,x\rangle+\|w\|r\). Independently,

\[
0<l=\sqrt{\max(0,\|x\|-r)^2+\varepsilon}
 \le s_y\le
u=\sqrt{(\|x\|+r)^2+\varepsilon}.
\]

The signed quotient interval is

\[
\begin{split}
L&=\begin{cases}a/u&a\ge0,\\a/l&a<0,\end{cases}\\
U&=\begin{cases}b/l&b\ge0,\\b/u&b<0,\end{cases}\qquad
L\le\langle w,F(y)\rangle\le U.
\end{split}
\tag{FT3}
\]

The sign cases follow by multiplying through positive denominators; using
the same denominator endpoint for both signs would be wrong. Regularization
keeps l positive even if the ball contains the origin. Source RMS contributes
the factor \(\sqrt n\), or equivalently divides the denominator interval
by \(\sqrt n\). The native experiment then adds the independently audited
execution allowance \(\|w\|10^{-3}+2\cdot10^{-3}\).

Intersect FT3 with the independent combined final-feature interval by taking
the maximum lower endpoint and minimum upper endpoint. Both components are
audited separately. A strictly negative upper competitor gap certifies that
the target beats that competitor; applying the existing all-competitor
margin and log-sum-exp bounds gives correctness and CE intervals. No observed
future feature, score or error selects a radius, denominator or class direction.
Strictness uses only the old score and frozen endpoints; future scores are
used to check the resulting certificates.

`ForecastSecant.lean`, `ForecastGram.lean` and `ForecastDirectional.lean`
contain 18 proved theorems, each conditional theorem with an explicit
satisfying example. `scripts/ForecastTighteningAxioms.lean` checks all of them.
These extend the forecast bounds, not the convergence theory: they neither
raise model accuracy nor establish an efficient forward-only training method.
The [experiment findings](../experiments/forecast_tightening/README.md) retain
all predeclared ablations, execution checks and preparation costs.

## Training-to-forecast bridge

Task #197 joins ST1–ST3's signed training reconstruction to FT1–FT3's
pre-update forecasts. The [frozen protocol](../experiments/transport_forecast/PROTOCOL.md)
uses the actual plain complex-product source on `(a+b)^3 mod 97`. This is
the next conditional connection in this development, not a replacement for
the failed componentwise spectral diagnostic or a solution of `a^3+ab`.

### TF1. Retain training geometry across the proposed step

Fix old normalized training states `T`, their center `mu`, old query state
`x`, coefficients `a`, and a computed basis `B`. For the competitor-minus-target
readout `w`, ST gives an old signed prediction `P=transportPrediction` and
directional remainder `R=directionalResidualBound` with

```
<w,x> <= P + R.
```

Neither the basis nor the coefficients need be exactly orthogonal. The
existing defect term stays in `R`. Keep this old geometry fixed across one
actual parameter update. Let `w'` be the known rounded updated readout,
`v` the old-network directional feature tangent, and suppose FT propagation
supplies `||x'-(x+v)|| <= r`, with `r>=0`. Then the exact finite identity is

```
<w',x'> = <w,x> + Delta + <w',x'-(x+v)>
Delta = <w'-w,x> + <w',v>.
```

The second term includes `<w'-w,v>`; it is not the first-order readout
`<w,v>` alone. Cauchy–Schwarz gives the complete bridge

```
<w',x'> <= P + R + Delta + ||w'||*r.
```

The finite-coordinate Lean theorem takes the equivalent squared-ball premise.
The native implementation uses FT1/FT2's combined radius and retains the
fixed feature/logit execution allowances. A two-coordinate witness has
training states `(1/2,0),(3/2,0)`, center `(1,0)`, query `(3,1/4)`, old
readout `(-2,1)`, new readout `(-3,0)`, tangent `(1/4,0)`, and future state
`(27/8,1/4)`. Both the old residual and forecast remainder are nonzero.
Its future score is `-81/8`, below the bridge endpoint `-75/8`.

### TF2. Transport the directional endpoint without hiding its slack

If FT3 gives an independently valid future gap endpoint `U`, its signed
drift allowance is `U-<w,x>`. The corresponding training bridge is

```
U_bridge = P + R + (U-<w,x>)
         = U + (P+R-<w,x>).
```

Lean proves both the bound and `U<=U_bridge`. Training reconstruction adds
nonnegative slack; this composition cannot improve the independent forecast's
coverage. Its value is testing how much of that coverage can also carry a
training-to-query explanation. Reporting `min(U,U_bridge)` as the bridge
would erase that question. The diagnostic reports both separately.

### TF3. Preserve every competitor and the actual loss

Apply TF1 to every class contrast with the same actual future state. If
executed scores differ from their exact linear readouts by at most `eta`,
add `2*eta` to each endpoint. Negative upper endpoints for all wrong classes
give a strict correct winner, and endpoints at most `-m` give margin `m`.
The existing odds-envelope lemmas give probability floor
`1/(1+sum exp(U_c))` and CE ceiling `log(1+sum exp(U_c))`, summing wrong
classes. The worst competitor may change. These are per-query sufficient
conditions; no average-margin argument or future error measurement enters
their construction.

### TF4. Actual source and temporal scope

`ComplexProduct.executed_gap_le_transport_forecast` instantiates TF1/TF3
with the old and new `readoutState`, the actual tied embeddings and all four
source normalizations. The endpoint adapter instantiates TF2. Feature-ball
and numerical-execution premises remain explicit, just as in FF and FT;
this does not formally verify the full Rust/BLAS/CUDA program.

The execution adapter also separates a squared feature discrepancy `delta^2`
from per-logit arithmetic error `eta`. It derives the native contrast allowance
`||w'||*delta+2*eta`, rather than silently treating the feature error as a
per-logit error. The feature radius becomes `r+delta`; both nonnegativity and
both error premises have explicit witnesses.

The frozen native study replays both original seeds, preserving their full
parameter/moment hashes at 0, 5000 and 10000, and inserts 26 declared probes
through the trajectory. Training-only answer-orbit bases transfer observed
labels via the proved cubic routes. Rank 32 is primary; rank 0 and rank 128
are center and reconstruction-limit controls. Every class endpoint is
persisted before the updated-network evaluation. Existing seeds and the
previously examined validation split provide a temporal diagnostic, not an
independent statistical generalization test. A correctness certificate cannot
precede the correctness it certifies. The test asks whether useful coverage
appears before the first sampled 99% accuracy, with no eventual-convergence
claim. The original flat-spectrum learning obstruction remains open.

| Claim | Lean declaration |
| --- | --- |
| Signed finite update with mixed term | `FeatureCodes.linearScore_forecast_identity` |
| Euclidean remainder and training composition | `FeatureCodes.linearScore_forecast_error_le`, `signed_transport_forecast_upper` |
| Directional endpoint and unavoidable old slack | `FeatureCodes.signed_transport_endpoint_upper`, `endpoint_le_transportEndpointUpper` |
| Nonzero update/residual witness | `FeatureCodes.transport_forecast_nonzero_witness` |
| Executed gap, margin and strict winner | `FeatureCodes.executed_gap_le_transportForecast`, `margin_of_transportForecast`, `strict_winner_of_transportForecast` |
| Actual probability and CE | `FeatureCodes.probability_ge_transportForecast`, `crossEntropy_le_transportForecast` |
| Complete source adapters | `ComplexProduct.executed_gap_le_transport_forecast`, `executed_gap_le_transport_endpoint` |
| Separate feature and arithmetic execution errors | `FeatureCodes.signed_transport_forecast_execution_upper`, `ComplexProduct.executed_gap_le_transport_forecast_with_feature_error` |

## Observable spectral geometry

Task #198 extends the geometry after the user's request to close both the
compressed-certificate and learning-mechanism gaps. The extension concerns
the directions that change class comparisons. It does not assert that the
full hidden feature spectrum becomes flat. The
[protocol](../experiments/readout_geometry/PROTOCOL.md), committed before
candidate measurements, fixes both trajectories, all 26 probes, ranks,
execution allowances and the actual AdamW attribution.

### OG1. Remove common readout offsets

For finitely many classes with readout rows `E_c`, define

```
mean(E) = (sum_c E_c) / number_of_classes
C_c = E_c - mean(E).
```

Every pairwise score difference is unchanged:
`<C_c,x>-<C_t,x>=<E_c,x>-<E_t,x>`. A common added readout row also cancels
from every comparison. Linear scoring commutes with the class mean. Thus
the classification geometry should not demand recovery of a shared logit
offset. The finite Lean results also cover empty class sets, for which the
contrast claims are vacuous and the sums are zero.

### OG2. The observable Gram form

For a feature error `e`, define

```
Q = sum_c C_c C_c^T
observable_energy(E,e) = sum_c <C_c,e>^2 = e^T Q e.
```

The equality is an exact finite-sum identity. This nonnegative form may be
singular: zero energy is equivalent to `<E_c,e>=<E_t,e>` for every pair
of classes. Such errors can change every absolute score while preserving
every margin. No inverse, positive smallest eigenvalue, or exact feature
recovery is assumed. This is the natural readout-dependent extension of
the earlier recovery geometry; it does not replace a missing optimizer
argument with a definition.

### OG3. A sharp finite margin budget

For distinct classes, their two centered squared score errors are at most
the total energy. The scalar inequality `(a-b)^2 <= 2(a^2+b^2)` gives

```
|<E_c,e>-<E_t,e>| <= sqrt(2*observable_energy(E,e)).
margin_t,c(x+e) >= margin_t,c(x) - sqrt(2*observable_energy(E,e)).
```

The bound also holds when the classes coincide. If every old target margin
exceeds this budget, the target remains the strict winner. These are
conditional theorems: a prediction must obtain its error budget before
observing the future error. The native experiment changes basis selection
using known old training states and readouts; it retains the existing
signed transport residual and orthogonality-defect bounds. It does not
insert a measured future observable energy into a forecast.

The readout-aware constructor first projects all centered readout rows into
the available centered-training-state span, then takes the leading right
singular vectors. It uses the training pseudoinverse only to construct
coefficients; the actual reconstruction residual and basis defects remain
in the certificate. Thus small training singular values, rank truncation
and imperfect arithmetic cannot be silently treated as exact recovery.
Rank 32 has the same budget as the hidden-variance baseline. The existing
signed-transport theorem accepts either computed basis; optimality of a
floating-point SVD is not a premise of correctness.

### OG4. Nonzero and sharp witnesses

Take two readout rows `(-1,3),(1,3)` and error `e=(0,7)`. Both scores shift
by 21, although observable energy is zero and the hidden error is nonzero.
Every margin is preserved. For rows `-1,1` in one dimension and `e=1`,
the squared contrast error is 4 and observable energy is 2, attaining the
factor two in OG3. Lean proves both witnesses and instantiates the strict
winner theorem with a nonzero invisible error and positive old margins.

### OG5. Attribute the actual adaptive update

The fixed source has weight decay one. Write `g_R` for the full-training
partial readout derivative, `g_F` for the full tied-embedding derivative,
`g_B` for the actual minibatch derivative, and `m_hat` for the updated
bias-corrected first moment. Keep each coordinate's actual diagonal
preconditioner `D`. The known rounded embedding increment obeys

```
Delta E = -eta E - eta D g_R - eta D(g_F-g_R)
          - eta D(g_B-g_F) - eta D(m_hat-g_B) + rounding.
```

The rounding term is defined by the difference between the executed FP32
increment and this real-valued decomposition, and is explicitly reported.
It includes any discrepancy in the reconstructed adaptive scaling. The
seventh parameter component contains the actual projection-matrix changes.
The instrumented update must match ordinary source `train_step` byte for
byte. Full-training partial readout derivatives receive an independent
analytic check, and the complete tied derivative receives source finite
difference checks.

Propagate each parameter component through the old source Jacobian. If
`Delta w=sum_i dw_i` and `v=sum_i v_i`, finite algebra gives

```
Delta forecast_score
  = sum_i (<dw_i,x> + <w,v_i>) + <Delta w,v>.
new_score = old_score + Delta forecast_score + <w',x'-(x+v)>.
```

All signed terms, the mixed term and the nonlinear remainder are retained.
Lean proves these identities for arbitrary finite decompositions, with
explicit satisfying examples. It also gives a witness where one positive
component coexists with a negative total update. In the native study the
parameter, tangent and logit sums must close, and the independent verifier
reconstructs the total predicted change from frozen features and independent
old logits. Attribution fixes the old worst competitor; correctness bounds
still cover every competitor.

These are contributions to one actual update, not seven alternative trained
trajectories or a proof of long-run causal necessity. Readout-only CE credit
cannot stand in for adaptive tied-weight dynamics. The
[findings](../experiments/readout_geometry/README.md) distinguish reduced
certificate slack, measured update contributions and the remaining discovery
and convergence question.

| Claim | Lean declaration |
| --- | --- |
| Class-centering and common shifts | `FeatureCodes.centeredReadout_contrast`, `common_readout_shift_preserves_contrast` |
| Observable quadratic form and nonnegativity | `FeatureCodes.observableEnergy_eq_gram`, `observableEnergy_nonneg` |
| Exact invisible-error characterization | `FeatureCodes.observableEnergy_eq_zero_iff_centered`, `observableEnergy_eq_zero_iff_contrasts` |
| Sharp contrast and finite margin bounds | `FeatureCodes.observable_contrast_sq_le`, `observable_contrast_abs_le`, `observable_margin_lower` |
| Invisible errors and strict winners | `FeatureCodes.invisible_error_preserves_contrast`, `strict_winner_of_observable_error` |
| Nonzero and sharp witnesses | `FeatureCodes.nonzero_invisible_error_witness`, `observable_contrast_sharp_witness` |
| Signed component and remainder identities | `FeatureCodes.forecast_component_identity`, `finite_update_component_identity` |
| Positive component, negative total | `FeatureCodes.positive_component_negative_update_witness` |

## Class-contrast discovery and persistence

Task #199 follows the user's request to pursue the remaining learning
question. The [fixed protocol](../experiments/class_geometry/PROTOCOL.md)
replays two original trajectories and six replication seeds for all 10000
updates. The [findings](../experiments/class_geometry/README.md) distinguish
a proved supervised damping mechanism, exact attribution of realized
trajectories, and the still missing convergence and alignment argument.
These claims extend the local CE derivative and observable geometry above;
they are authored derivations, not claims attributed to another manuscript.

### CG1. A supervised dissipative direction

Let `T_i` be the current features, `y_i` the current minibatch targets, and
`z_ic=<E_c,T_i>`. The existing derivative theorem `readoutGradient_eq`
identifies the actual partial derivative of mean cross-entropy:

```
g_R,c = mean_i ((p_ic - 1[c=y_i]) T_i),
p_ic = exp(z_ic) / sum_v exp(z_iv).
```

For two classes `c,d` absent from the minibatch targets, their label terms
vanish. Expanding the finite sums gives the exact identity

```
<E_c-E_d, g_R,c-g_R,d>
  = mean_i ((z_ic-z_id)(p_ic-p_id)) >= 0.
```

The sign follows from monotonicity of the exponential and a positive
softmax denominator. Features are fixed only for this partial derivative;
the theorem can be applied anew after each feature or minibatch change.
For any nonempty group U containing no targets, the pairwise identity

```
sum_u <E_u-mean_U(E), g_R,u-mean_U(g_R)>
  = (1/(2*|U|)) sum_u,v <E_u-E_v, g_R,u-g_R,v> >= 0
```

turns pair damping into damping of the whole centered unused group. In
particular, for nonnegative step scales `eta_t,alpha_t`, its cumulative
isotropic radial contribution `sum_t -2*eta_t*alpha_t*<Q_U E_t,Q_U g_R,t>`
is nonpositive for every finite prefix, even with changing features and
minibatches. This is a theorem about one signed contribution. It does not
identify the complete tied, adaptive optimizer with isotropic readout descent.

### CG2. Keep the finite step

For any finite vectors `w,g` and real `eta`,

```
||w-eta*g||^2 = ||w||^2 - 2*eta*<w,g> + eta^2*||g||^2.
```

Thus `eta>=0` and the current-state condition
`eta*||g||^2 <= 2*<w,g>` imply nonincrease. A dissipative derivative alone
does not imply contraction for every learning rate. Neither this condition
nor isotropic dynamics is assumed for the complete source AdamW step.
The native audit checks its separate CG1 contribution on every update and
keeps the complete finite quadratic term in CG4. A nonzero two-coordinate
witness satisfies the finite-step condition with positive learning rate.

### CG3. Why 32 class directions appear

Split the readout rows into nonempty groups A and U. With the overall mean
`mu`, and their computed group means `mu_A,mu_U`, finite centering gives

```
||E-centered||_F^2
 = sum_a ||E_a-mu_A||^2 + sum_u ||E_u-mu_U||^2
   + |A|*||mu_A-mu||^2 + |U|*||mu_U-mu||^2.
```

The three terms are active within-group contrasts, unused within-group
contrasts, and between-group energy. Cross terms vanish because each
within-group centered sum is zero. The Lean scalar identity holds for any
chosen `mu`; summing it over coordinates and choosing the overall mean
gives this Frobenius identity.

For `n+1` rows, centering also has an explicit reconstruction:

```
E_c-mean(E)
 = sum_(i=1..n) (1[c=i]-1/(n+1)) (E_i-E_0).
```

Hence at most `n` generators are needed; linear independence is not assumed.
In this experiment the training labels have 33 distinct answers and 64
unused classes. Active contrasts therefore need at most 32 generators;
the two group means add at most one more direction, since their weighted
centered sum is zero. This dimensional fact holds at initialization too.
Learned concentration in these directions is a separate assertion, and a
rank-33 SVD need not select exactly the between-group direction.

### CG4. Attribute the actual trajectory

Here `g_R` is the partial derivative on the **actual minibatch** and `g_B`
its full tied-embedding derivative. This differs deliberately from OG5's
full-training/sampling split. With actual updated moment `m_hat`, actual
coordinatewise preconditioner `D`, and `alpha=min_coordinate(D)>0`, write

```
Delta E = -eta E - eta alpha g_R - eta(D-alpha I)g_R
          -eta D(g_B-g_R) -eta D(m_hat-g_B) + rounding.
```

The six terms are decay, isotropic readout, adaptive readout correction,
tied-encoder credit, moment memory, and the measured FP32 remainder. The
remainder is bounded and reported; no exact floating-point implementation
theorem is claimed. Class centering is applied after each coordinatewise
increment: `D` need not commute with the class projections.

For any finite component family `d_i`, the real identity is

```
||x+sum_i d_i||^2-||x||^2
  = sum_i 2<x,d_i> + ||sum_i d_i||^2.
```

The last square includes all cross-component products. If every realized
step satisfies `x_(t+1)=x_t+sum_i d_(t,i)`, summing these contributions
telescopes to `||x_n||^2-||x_0||^2` for every prefix n. Apply this separately
to each fixed class projection. Lean proves both identities with a nonzero
trajectory example; the experiment checks parameter closure, all projected
energy identities, and every cumulative prefix without dropping terms.

### CG5. Relative concentration and persistence

Absolute active energy is insufficient to explain its share. Let `S` be
active energy and `V` unused plus between-group energy. For nonzero old and
new total energies, algebra gives

```
(S+ds)/(S+V+ds+dv) - S/(S+V)
  = (V*ds-S*dv) / ((S+V)*(S+V+ds+dv)).
```

Substitute all CG4 radial terms and the total quadratic term into `ds,dv`.
This yields an exact signed attribution of share change, which can be
summed over every prefix. With positive denominators, `S*dv <= V*ds` is
the explicit improvement condition. The native trajectory does not satisfy
monotone improvement at every step; these conditions are not asserted to
follow from training loss or the local dissipative sign.

Common nonzero scaling preserves `S/(S+V)`. Pure weight decay scales both
energies by the same square and cannot by itself select active contrasts.
Its radial share numerator is zero. Mixed-step cross products still remain
in the total quadratic term, so decay is not removed from the full optimizer.
Signed attribution explains one realized trajectory, not the outcome of
retraining after deleting an optimizer component.

### CG6. Concentration still needs alignment

There is a simple obstruction to identifying class concentration with
successful prediction. In one feature dimension take active readout rows
`+1,-1`, unused row `0`, and query feature `-1` with target the `+1` row.
All centered energy is in the active contrast space, yet the other active
class wins strictly. Lean proves the two energies equal 2 and the strict
wrong-class comparison. Thus concentration needs a separate query alignment
and positive-margin argument such as OG3 and signed transport.

The fixed study finds a reproducible increase in active-class share and
useful sampled late geometry, with signed opposition from tied-encoder and
memory terms. It also retains fluctuations, persistent residual readout
energy, and failures to sustain 99% coverage. The remaining theorem must
control coupled feature learning, adaptive scales and moments strongly
enough to obtain query alignment and margins over time. CG1–CG5 alone do
not prove that theorem or universal optimizer discovery.

| Claim | Lean declarations |
| --- | --- |
| Actual CE pair identity and sign | `GradientEvidence.unused_pair_gradient_identity`, `unused_pair_gradient_damping`, `logit_probability_difference_nonneg` |
| Whole-group and cumulative damping | `GradientEvidence.unused_group_gradient_damping`, `cumulative_unused_readout_credit_nonpos` |
| Finite isotropic identity and condition | `GradientEvidence.isotropic_pair_energy_identity`, `isotropic_pair_energy_nonincrease` |
| Computed class means and energy split | `FeatureCodes.class_center_sum_zero`, `class_translation_energy`, `class_split_energy` |
| Pairwise-to-centered product identities | `FeatureCodes.zero_sum_pair_product`, `centered_pair_product` |
| Explicit centered-row generators | `FeatureCodes.centered_rows_from_anchor` |
| Finite and cumulative component identity | `GradientEvidence.component_energy_identity`, `trajectory_component_energy_identity` |
| Relative share identity and improvement | `GradientEvidence.class_share_change`, `class_share_component_identity`, `class_share_nondecrease` |
| Common scaling preserves share | `GradientEvidence.common_scaling_preserves_class_share` |
| Full concentration with a wrong prediction | `FeatureCodes.active_concentration_insufficient_witness` |

## Forward–Forward reformulation

User direction, 2026-11-08: reformulate Forward–Forward and modify its loss.
The reference is Geoffrey Hinton, *The Forward-Forward Algorithm: Some
Preliminary Investigations*, arXiv:2212.13345, §§2–3.3, read locally in
`papers/arXiv-2212.13345/paper.txt`. The following is an explicitly changed
candidate, not a claim made by that paper or a new convergence guarantee.
The [fixed native diagnostic](../experiments/forward_forward/PROTOCOL.md)
precedes any full training comparison.

### LF1. Factor the label out of the encoder

Compute a label-independent local feature `h=f(x)` once. For class vector
`e_c`, define corrected distance goodness

`G(h,e_c) = (||e_c||² - ||h-e_c||²)/2 = Re<h,e_c> - ||h||²/2`.

The class-norm correction matters: uncorrected nearest-prototype distance
would add a class-dependent bias and change the source classifier. The
common feature-norm term cancels in every positive/negative difference:
`G(h,e_y)-G(h,e_c) = Re<h,e_y-e_c>`. Thus the existing real dot-product
readout, or the real part of a Hermitian product in complex coordinates,
computes all goodness comparisons. Inference remains the same classifier.
Complex multiplication inside the encoder is unchanged.

Hinton's supervised label-conditioned encoder can require one encoder
evaluation per candidate label (§3.3); this factorization uses one encoder
evaluation and a class readout. It gives up label-conditioned feature
extraction. It does not eliminate readout work or prove an overall speedup.
Putting these goodness scores into a softmax recovers ordinary CE exactly;
that relabeling alone would not change the learning algorithm.

### LF2. Do not use normalized energy as the only goodness

Hinton measures squared activities **before** normalization (§2). In the
actual source model, a post-RMS squared norm is
`W*q/(q+W*epsilon)`, where `q=||x||²`. It depends only on radius and saturates
near W. At zero epsilon it is exactly W for every nonzero input. Its
derivative with respect to q is `W²*epsilon/(q+W*epsilon)²`.
Equal-energy features can have opposite label preferences. LF1 retains
that directional information and avoids treating post-normalization
magnitude as a rich classification signal. No change to RMS is required.

### LF3. Change the local loss to paired directional margins

For local scores `s_c=Re<h,e_c>`, let

`L_pair = sum_{c != y} softplus(mu+s_c-s_y)/(C-1)`.

The fixed candidate uses mu=1 and all C-1=96 negative labels. No label
requires a second encoder evaluation. Unlike Hinton's separate positive
and negative binary objectives, this loss directly compares each negative
with its positive partner. Its logit derivatives are

`q_c = sigmoid(mu+s_c-s_y)/(C-1)` for c != y,
`q_y = -sum_{c != y} q_c`.

The local linear-head derivatives are `dL/dh=sum_c q_c e_c` and
`dL/de_c=q_c h`. Complex derivatives mean the corresponding real-coordinate
derivatives, with the conjugation in the Hermitian product retained. At
mu=0 and two classes this is CE. With multiple competing classes its
gradient need not be a scalar multiple of the CE gradient. The distinct
competitor weights are a real change to the objective.

The **sum** of pair penalties below softplus(mu) guarantees positive margin
against every competitor. The native **mean** therefore needs a threshold
softplus(mu)/(C-1). A small average penalty or a 90% certificate coverage
must not be presented as 100% model accuracy. This is a per-query labeled
loss implication; it gives no held-out generalization guarantee.

### LF4. Specify local credit, not just a different loss

At a boundary, cache the incoming feature u and optimize the next block's
local loss with u fixed. Local Jacobians, outer products and optimizer
updates still require computation. No derivative from an upper block may
be silently propagated into an earlier block and called Forward–Forward.
For the two-layer scalar example `u=a*x`, `v=b*u`, the local first-layer
loss softplus(-a) has derivative -1/2 at a=0, x=1. The terminal loss
softplus(-2*a) has derivative -1 there. The two update rules are different.

The source E is shared by the input embedding and output readout. The
local readout derivative is only one contribution to this shared parameter;
changing E also changes the next encoder evaluation. The native one-step
probe therefore reports both frozen-feature and actual tied-network loss
changes. It is not a complete local-learning trainer. Before task #190,
specify losses/prototypes for every trainable block, detachment boundaries,
and accumulation of all local contributions before one shared-parameter
update. Count their storage and computation, and compare training from
scratch at matched validation quality. A loss reformulation alone supplies
neither a faster optimizer nor a solution to the flat-spectrum obstruction.

| Claim | Lean declarations in `ForwardForward` |
| --- | --- |
| Corrected distance, class order and CE identity | `goodness_eq_inner`, `goodness_difference`, `goodness_order`, `goodness_crossEntropy` |
| Normalized energy and its sensitivity | `normalized_energy`, `normalized_energy_zero_eps`, `normalized_energy_derivative`, `equal_energy_opposite_preferences` |
| Actual pairwise derivative and invariances | `softplus_derivative`, `pairLoss_directional`, `pairLoss_shift`, `pairLoss_goodness` |
| Per-query strict correctness threshold | `pair_term_le`, `winner_of_pairLoss_lt` |
| Local versus terminal derivatives | `local_linear_derivative`, `terminal_first_derivative`, `local_global_derivative_witness` |

### LF5. Complete local training exposes the missing link

The [frozen task #190 pilot](../experiments/forward_training/README.md)
specifies local objectives for all blocks and sums both direct contributions
to the tied E before one original AdamW update. Its controls separately
change terminal loss, add fixed auxiliary heads and stop cross-block credit.
All 224 directional checks pass, but local training ends at 2.23–10.50%
validation accuracy across eight cases, compared with 99.28–99.87% for CE.
Full auxiliary backpropagation is also worse than CE. This is empirical
evidence against that particular objective/boundary choice, not a
counterexample to the local derivative theorems or to all Forward–Forward
methods. Complex multiplication is unchanged.

A useful extension must connect a local parameter displacement to terminal
task improvement, with all tied-parameter contributions included. Local loss
descent alone supplies no such implication. The next diagnostic is their
directional alignment under the actual optimizer, before selecting new local
heads or boundaries. Any full-gradient measurements used for diagnosis must
remain separate from the information available to a proposed local trainer.
The pilot found no efficiency gain, and simultaneous foreign GPU training
also prevents a clean-device speed comparison from its retained timings.

### LF6. Local gradient, optimizer displacement and finite step differ

The [task #193 diagnostic](../experiments/forward_alignment/README.md)
measures all three at 40 saved states. Complex directional pairings use
the signed pullback to independent real coordinates. Validation and
full-training gradients are charged diagnostic information, never free
signals for a proposed local trainer.

At all eight failed local endpoints, the hidden block A's current gradient
aligns with terminal validation CE, but its actual AdamW displacement is
an ascent direction in five. Positive finite-change remainders turn two
other first-order descent predictions into actual increases. A lowers its
own frozen local loss in all eight probes while raising full-training CE
in all eight and validation CE in seven. B-only updates decrease full-training
and validation CE in every local endpoint, while validation pair loss falls
in five of eight. The vocabulary-E removal worsens complete
validation CE in all eight, so the present probes do not support removing
that term as the immediate repair.

These numerical observations distinguish three obligations for a useful
local method: task alignment of the available gradient, preservation of
useful direction by the optimizer, and control of finite-step error. They
do not prove a trajectory-level cause or a generalization guarantee. The
next diagnostic should isolate A's direction and scale before changing
objective and optimizer together. No new training outcome or speed claim
follows from these backward-assisted measurements.

### LF7. Smaller steps do not remove a direction defect

The [task #194 follow-up](../experiments/forward_step/README.md) compares the
actual A displacement with its norm-matched negative current local gradient
at five fixed scales, retaining all 40 states and 400 probes. At the eight
local endpoints, every gradient-direction probe reduces full-training and
validation CE. AdamW improves validation CE in one case at full scale and
three at scales 1/64 and 1/256: shrinking the step recovers the two overturned
descent predictions but retains five ascent cases. This separates observed
finite-step overshoot from an observed direction defect without assigning
that defect solely to moment carryover, adaptive scaling or decay.

All 400 probes reduce their local loss, yet the other state origins still
have mixed terminal outcomes. Accuracy remains mixed even where CE falls.
These are finite numerical observations, not a theorem about an entire
training trajectory. They support testing a current-gradient A update from
scratch while holding the local objective and other block rules fixed.
Its step-size policy must be specified independently: borrowing AdamW's
radius in this diagnostic establishes neither a standalone optimizer nor
an efficiency advantage. Keep validation derivatives out of the proposed
trainer and use a separately frozen confirmation cohort. No complex-product
definition or formal theorem changes in this study.

### LF8. A useful one-step direction need not yield useful local training

The [task #195 comparison](../experiments/forward_normalized/README.md)
replaces only A's optimizer with its normalized current local gradient,
using a fixed 1%-of-A-norm radius and ten-update warmup. It leaves local
objectives, E/B AdamW and the classifier unchanged. All 24 fresh-seed runs
complete 4000 updates, yet the candidate reaches only 2.23–11.63% validation
accuracy versus 99.45–99.87% CE. Its endpoint validation CE is worse than
original local-pair in all eight cases; accuracy improves in two and falls
in six. Numerical and source checks pass, so this is a negative learning
result for the specified rule and budget, not a failed coordinate check.

This does not contradict LF7's finite same-state observations: the radius
and subsequent trajectory differ. Neither local descent nor favorable
terminal alignment at selected states proves trajectory-wide alignment,
finite-step control or generalization. A next empirical control should
cross local/full credit with original/normalized A updates while keeping
the auxiliary objective set fixed. This separates a possible step-policy
failure from stopped credit without blaming the head or optimizer from
this study alone. The result supplies no new theorem, accuracy repair,
memory saving or clean-device speed advantage.

## Main-model AdamW motion

Task #203 pursues the remaining discovery/persistence question under the
[fixed protocol](../experiments/geometry_dynamics/PROTOCOL.md). These are
authored derivations for the main-model AdamW copied from GPT commit
`6bb9d03a`, with beta1=.9, beta2=.95 and decay .1. Exact-real arithmetic and
the numerical FP32 audit remain distinct. The gradient sequence below may
come from the full coupled model, a minibatch, clipping, or any other rule.

### AD1. A moment invariant independent of the gradient direction

For one coordinate, write

```
m+ = (9/10)m + (1/10)g,
v+ = (19/20)v + (1/20)g^2.
```

Expanding the actual recurrences gives the identity

```
(19/14)v+ - (m+)^2
  = (19/20)((19/14)v - m^2) + (14m-9g)^2/1400.
```

Thus `m^2 <= (19/14)v` is preserved for every real gradient g. Zero
initial moments satisfy it, so induction proves it at every time without
assuming an improving loss, fixed features, isotropic scales, or a favorable
gradient sign. In particular, v is nonnegative. This applies separately to
each independent coordinate before restoring the complex parameter ties.

If the actual buffer updates instead add errors `em,ev`, their excess obeys

```
(m+ + em)^2 - (19/14)(v+ + ev)
 <= (19/20)(m^2-(19/14)v)
    + 2|m+||em| + em^2 + (19/14)|ev|.
```

The square in the first identity is dropped only with its proved sign.
The errors are explicit; this is not a proof that the FP32 recurrence
satisfies the unperturbed cone. The native audit checks every saved moment
state using the protocol's fixed numerical allowance.

### AD2. Bias-corrected adaptive directions are bounded

For a nonnegative variance, `m^2 <= K^2 v`, `K>=0`, positive bias factors
`b1,b2`, and positive epsilon, elementary square-root algebra yields

```
| (m/b1) / (sqrt(v)/b2 + eps) | <= K*b2/b1.
```

This includes v=0. At zero-based update t, AD1 supplies the explicit bound

```
B_t = sqrt(19/14) * sqrt(1-(19/20)^(t+1)) / (1-(9/10)^(t+1)).
```

This is a bound on the actual buffered adaptive direction, including its
two different bias corrections. It does not replace momentum by a current
gradient. At late times B_t approaches `sqrt(19/14)`, about 1.165.

### AD3. Finite-window motion with continuing updates

For a coordinate update `p+ = (1-eta*rho)p - eta*u + e`, assume `eta>=0`
and `eta*rho<=1`. If `|p|<=R` and `|u|<=B`, then

```
|p+| <= (1-eta*rho)R + eta*B + |e|.
```

If also `rho>=0`, displacement obeys

```
|p+-p| <= eta*(rho*|p|+B) + |e|.
```

The invariant box `|p|<=R` is preserved by exact steps when `B<=rho*R`.
For a sharper finite interval starting at an arbitrary saved time s, set
`R_0=|p_s|` and `R_(j+1)=(1-eta_(s+j)*rho)R_j+eta_(s+j)*B_(s+j)`.
Induction and the triangle inequality give

```
|p_(s+n)-p_s| <= sum_(j<n) eta_(s+j)*(rho*R_j+B_(s+j)).
```

Lean proves the result for the actual exact-real main-model moment and
parameter recurrences, with arbitrary gradients and any finite window.
The recurrence is explicitly identified with `bufferedAdamUpdate`.
It permits nonzero updates throughout the window. The native FP32 trace
audit retains its recorded coordinate rounding remainder in both radius
and displacement recurrences and checks all recorded readout updates.
It does not prove a worst-case FP32 error bound for all future executions.

### AD4. What motion control must still supply

Let w be a target-minus-rival readout contrast and x a query feature.
For coordinate bounds `|w'_k-w_k|<=r_k` and `|x'_k-x_k|<=s_k`,

```
|<w',x'>-<w,x>| <= sum_k (|w_k|s_k + |x_k|r_k + r_k*s_k).
```

The mixed term remains. A positive old margin lower bound larger than this
budget guarantees a positive new contrast. Apply the statement to every
rival; a contrast's readout budget is the sum of its two row budgets.
This turns useful motion bounds into a maintenance guarantee, including
feature movement explicitly. AD3 alone does not control the nonlinear
feature map tightly enough, nor does it establish the starting margins.

Consequently these theorems add gradient-independent control of exact
AdamW step sizes, but do not prove discovery of the useful geometry.
Small individual steps can accumulate a large displacement. A constant
positive learning rate over arbitrarily many steps does not provide a
small total budget. The study reports whether the explicit bound is useful
on the declared hold interval; a loose bound is not counted as certification.

### AD5. Valid certificate envelopes

The new single basis uses the rows `E_a-mu_U` for training-observed labels a,
where mu_U is the mean of unused rows. Each equals
`(E_a-mu_A)+(mu_A-mu_U)`: the CG3 active contrasts plus the between-group
direction. There are at most 33 generators here. Rank 32 still truncates;
rank 33 is a distinct control. All rival classes and all residual terms
remain in the arbitrary-basis signed-transport bound.

If a,b,c are three valid lower bounds for the same margin M, then
`max(a,max(b,c))<=M`. A positive constituent remains positive in this
envelope, and a positive envelope guarantees `M>0`. Lean proves validity,
retention and soundness. No query outcome is required to take the maximum.
Preparing three bases has a separate cost: this is not one rank-32 basis
or a speed claim. The independent artifact verifier checks the maximum
and every emitted bound directly.

The formal modules are `GradientEvidence.MainAdamMoments`, `MainAdamMotion`
and `MainAdamCertificates`; `scripts/MainAdamAxioms.lean` lists all 21
theorems for an explicit axiom audit. Each theorem with hypotheses has a
nonzero admissible example. The [experiment README](../experiments/geometry_dynamics/README.md)
records the full numerical comparison and the remaining limitations.

## Observed finite-window persistence

Task #204 follows the failure of AD3's uniform coordinate bound to certify
the observed low-rate hold. The following refinement is conditional on
realized forcing budgets. It does not predict unknown future gradients,
prove their alignment, or explain discovery from random initialization.
The [frozen study](../experiments/geometry_persistence/PROTOCOL.md) checks
every hold update and separately transports predictions over finite windows.

### DP1. Damped forcing tubes

Let P be a fixed linear projection on the readout matrix, including an
active-class contrast, unused-class contrast, or between-group projection
from CG3. For an actual AdamW update with all rounding in e,

```
E+ = (1-eta*rho) E - eta*u + e,
P E+ - (1-eta*rho) P E = -eta*P u + P e.
```

Thus the forcing norm is at most `eta*||P u|| + ||P e||` for nonnegative
eta. The adaptive direction u includes the saved moments and the complete
tied-parameter gradient. No partial CE term is substituted. Alternatively
the realized residual `F_t = P E_(t+1) - a_t P E_t` can be measured directly;
the native checker includes projection arithmetic defects in its radius.

For any path x in a real normed space with nonnegative scales a_t and
`||x_(t+1)-a_t*x_t|| <= b_t`, define

```
q_0 = 1,     q_(t+1) = a_t*q_t,
R_0 = 0,     R_(t+1) = a_t*R_t + b_t.
```

Induction and the triangle inequality give

```
||x_n-q_n*x_0|| <= R_n,
max(0,q_n*||x_0||-R_n) <= ||x_n|| <= q_n*||x_0||+R_n.
```

The inductive residual is the sum of the next forcing and a_t times the
previous residual; this retains common decay exactly. Reindex a window at
zero to obtain every prefix statement from an arbitrary anchor. This works
with nonzero movement at every step. A uniform balance condition
`b_t <= (1-a_t)*B` further implies `R_n <= (1-q_n)*B`. The algebraic statement
holds for any real B and nonnegative a; the usual bounded-radius reading
also requires B nonnegative and scales at most one. Establishing that balance
for future adaptive forcing is an additional problem, not a consequence of
small individual learning rates.

The native audit can identify forcing from already executed training steps.
Those are observed-path bounds, even though they use no query outcomes.
This differs from AD1–AD3's gradient-independent guarantee available before
the window. The stronger hypothesis buys a potentially much smaller radius.

### DP2. Readout share certificates

For actual nonnegative signal and competing energies S,V, lower and upper
bounds `0<=L<=S`, `V<=U` give

```
L/(L+U) <= S/(S+V)
```

when both denominators are positive. Multiplying by the denominators reduces
the claim to `L*V<=S*U`. Apply this to squared active norms and the sum of
squared unused/between norms, using the lower and upper DP1 radii. CG3's
orthogonal decomposition identifies the denominator with centered readout
energy. When the lower active norm is exhausted the implementation emits
zero; it does not count nonnegative energy as a useful persistence result.

An energy-share guarantee does not establish query classification, a flat
nonzero singular spectrum, or feature alignment. Those limitations persist
even when the radius is small. The native study also checks prediction
certificates with the whole nonlinear feature map included.

### DP3. Transporting certificates across a finite displacement

Let M be a saved margin lower bound for a training-derived target and let
g_c denote the old rival-minus-target gap. Suppose the full future model's
gap g'_c is enclosed above by U_c. If `U_c-g_c<=D` for every rival, then

```
M <= -g_c,   g'_c <= U_c  implies  M-D <= -g'_c.
```

Consequently `D<M` preserves the target prediction against every rival. The
native D is the maximum of all 96 changes `U_c-g_c`, with no omitted unused
class. The starting M comes from task #203's stored envelope; rank 128 is a
separate control. A direct forecast without the old geometry slack is also
reported separately. A claim of retained certification additionally requires
the old bound to have been positive.

The FT1–FT3 forecast equations apply to an arbitrary finite parameter delta,
so the native study uses the difference between two checkpoints. It retains
the old-feature tangent, mixed products, nonlinear remainder, and unchanged
execution allowances. Endpoint parameters are known to the constructor;
endpoint evaluation and labels are not used to construct its intervals.
This certifies an endpoint after many updates. It says nothing about the
intermediate query predictions, which are not recorded at every update.

The modules `GradientEvidence.DampedPersistence` and
`GradientEvidence.PersistenceCertificates` formalize DP1–DP3;
`scripts/PersistenceAxioms.lean` audits their ten theorems. Native numerical
checking remains separate from Lean's real-arithmetic proof.

## Future forcing from checkpoint moments

Task #205 asks what can be guaranteed before later gradients or updates are
read. DP1's tube is useful with observed forcing, but that observation is an
input to its earlier experiment. The following exact-real bounds remove that
input for main-model AdamW. They do not assert reliable discovery, favorable
gradient alignment, or a flat nonzero singular spectrum.

### FF1. A sharp scalar cone update

At a checkpoint let `v>=0`, `r>=0`, and `m^2<=r*v`. For any next real gradient
`g`, retain the copied main-model recurrences

```
m' = .9*m + .1*g,     v' = .95*v + .05*g^2.
Q = 81/95,           C = 19/14.
((Q*r+1/5)*v' - m'^2)*v
  = Q*(r*v-m^2)*v' + (19*v-9*m*g)^2/1900.
```

When `v>0`, nonnegativity of the right side proves
`m'^2<=(Q*r+1/5)*v'`. When `v=0`, the initial cone forces `m=0`, and the same
inequality follows directly for all `g`, including zero. For `v>0` and
`m!=0`, set `r=m^2/v` and `g=19*v/(9*m)`; the square vanishes and the next
ratio attains `Q*r+1/5`. This sharpness concerns one scalar step over
unrestricted real gradients. It does not claim that the tied model's CE
gradient, or a globally clipped sequence, attains every worst-case step.
There is a nonzero clipped-compatible example: `m=.1`, `v=.01`, `g=19/90<1`.

Iteration yields the checkpoint cone

```
K_h(r) = C + (r-C)*Q^h,    m_h^2 <= K_h(r)*v_h.
K_(h+1)(r) = Q*K_h(r)+1/5.
K_h(r)-K_h(s) = (r-s)*Q^h.
```

These statements require no independence, fixed-feature, small-gradient,
loss-decrease, or favorable-direction assumption. They permit arbitrary
adaptation to the past. The ratio is zero when both buffers are zero; positive
variance permits the exact checkpoint ratio. The formula is nonnegative for
every nonnegative initial ratio. Its rapidly vanishing memory describes the
worst-case magnitude envelope, not a theorem that the learned geometry itself
forgets its initialization.

### FF2. Actual resumed moments and absolute-time normalization

`warmMoments` continues the same recurrences from saved buffers.
`main_moments_resume` identifies this continuation with the original history,
and `main_direction_resume` does the same for the normalized direction. For
the h-th future update (`h>=1`) after absolute checkpoint time t, write

```
b(t,h) = sqrt(1-.95^(t+h))/(1-.9^(t+h)),
u_h = (m_h/(1-.9^(t+h))) /
      (sqrt(v_h)/sqrt(1-.95^(t+h)) + eps).
```

For `eps>0`, AD2 and FF1 give `|u_h|<=sqrt(K_h(r))*b(t,h)`, including zero
variance. Epsilon remains in the actual update; dropping it only enlarges
the upper bound. The Lean function uses zero-based update index `h`, hence
its next buffer and bias exponent are `h+1` and `t+h+1`.

Simultaneously reversing the initial first moment and every future gradient
negates every first moment while preserving every second moment. Thus the
same moment magnitudes and cone can support opposite directions. This
symmetry limits what magnitude-only information implies; it is not a
counterexample to high-probability discovery under the actual training loss.

### FF3. Readout blocks and class centering

For a coordinate block J of size d, with checkpoint ratios r_k,

```
sum_(k in J) K_h(r_k) = d*C + (sum_(k in J) r_k-d*C)*Q^h,
||u_(J,h)||_2 <= b(t,h)*sqrt(sum_(k in J) K_h(r_k)).
```

This gives separate budgets for the whole readout and its observed-answer
row block. CG3's mean-subtraction identity at center zero proves
`sum_c (x_c-mean(x))^2 <= sum_c x_c^2`; summing across feature coordinates
therefore bounds the active class-contrast projection by the active-row
budget. The complementary centered projection has norm at most the whole
readout norm by CG3's orthogonal decomposition. Neither step assumes that
the actual gradient lies in the active subspace.

### FF4. A checkpoint-only finite tube and its limits

For the exact-real continuation `p_(h+1)=a_h*p_h-eta_h*u_(h+1)`, with
`a_h=1-eta_h*rho>=0` and `eta_h>=0`, set `q_0=1`, `R_0=0` and

```
q_(h+1) = a_h*q_h,
R_(h+1) = a_h*R_h + eta_h*B_(h+1),
B_h = b(t,h)*sqrt(sum_k K_h(r_k)).
```

DP1 now gives `||p_h-q_h*p_0||<=R_h` with B computed entirely from the
checkpoint and known schedule. `warm_adam_source` identifies each coordinate
with the existing buffered AdamW definition, and `main_adam_resume` identifies
the whole continuation with the original parameter history. `warm_adam_tube` has no premise
bounding a future gradient or observed forcing. Its norm bounds compose with
DP2 to bound active-energy share using the active-row and whole-readout
radii; `warm_adam_projected_tube` gives the tube for any fixed contractive
linear projection. A zero lower share is an exhausted certificate, not preserved useful
geometry.

The native forecast adds the protocol's predeclared numerical allowances:
relative direction allowance 1e-6 and a per-coordinate rounding cap
`8*eps32*(1+M_h+eta_h*Bmax_h*(1+1e-6))`. The coordinate cap M is itself
propagated from the checkpoint, schedule, and these predicted bounds. It
never substitutes observed future rounding into a forecast. These allowances
are audited against every saved FP32 update; their universal IEEE/CUDA
validity is not a Lean theorem. Exact-real proofs and finite numerical
verification remain distinct.

The missing discovery argument is still about the actual loss: why typical
initialization and minibatch sequences create the relevant class contrasts
and align the changing features with them. Long useful persistence also needs
more than a magnitude bound when the prospective radius exhausts the active
norm. The sign symmetry and the measured failures motivate controlling that
directional interaction; they do not prove such control impossible.

## Loss-specific momentum control

This continuation strengthens CG1's sign of unused-class CE damping to a
quantitative inequality, then absorbs arbitrary saved first moments in a
quadratic potential. These are authored consequences of the actual partial
CE derivative and the main model's beta1=.9 recurrence. They are not claims
that the flat-spectrum manuscript proves full tied AdamW training dynamics.
The corresponding files are `GradientEvidence/CECocoercivity.lean`,
`MomentumPotential.lean`, and `CEMomentumControl.lean`.

### CL1. Quantitative damping of the actual partial CE gradient

For a finite class set, let p be softmax(z). The exponential tangent gives
`exp(a)-exp(b) <= (a-b)*exp(a)`, with either ordering of a,b. Divide by the
common positive softmax denominator. When p_c>=p_d, multiply the resulting
inequality by p_c-p_d and use p_c<=q; when p_d>=p_c, swap the endpoints.
CG1's sign `(z_c-z_d)*(p_c-p_d)>=0` justifies both cases. Thus

```
(p_c-p_d)^2 <= q*(z_c-z_d)*(p_c-p_d),   p_c<=q, p_d<=q.
```

For minibatch features T_i and readout rows E_c, logits are `<E_c,T_i>`.
If neither class c nor d occurs as a target, their actual partial gradient
difference is `G_c-G_d = mean_i ((p_ic-p_id)*T_i)`. Holding T fixed here
specifies the derivative, not its behavior at later training steps.
Finite Cauchy–Schwarz bounds the squared norm of this mean by the product of
mean squared probability difference and mean feature energy. The displayed
softmax bound and CG1's gradient identity then give

```
||G_c-G_d||^2 <= L*<E_c-E_d,G_c-G_d>,
L = q*mean_i ||T_i||^2.
```

Sum over all pairs of unused classes and divide by twice their count. CG3's
computed centering identity yields `||G_U||^2 <= L*<E_U,G_U>` for the whole
within-unused contrast block. Its radial product is nonnegative by CG1.
No fixed feature table, fixed minibatch, probability independence, or
spectral-flatness assumption enters this pointwise statement. The Lean
statements also permit repeated class indices (the same weighted finite
pair identity) and empty batches (the existing zero-mean convention);
the experiment uses 64 distinct unused classes and nonempty minibatches.
For clipping a>=0, g=a*G_U has coefficient aL and nonnegative radial product.
The probability ceiling q=1 is always valid. A tighter measured q only
bounds the current minibatch unless a separate future bound is available.

### CL2. Saved momentum is dissipative in a suitable potential

Write w=E_U and let m be any saved within-unused first moment, including
arbitrary encoder/minibatch history. For a scalar effective step s define

```
m_R = .9*m+.1*g,       w_R = w-s*m_R,
V_s(w,m) = ||w-9*s*m||^2+90*s^2*||m||^2.
V_s(w_R,m_R)-V_s(w,m) =
  -2*s*<w,g>+19*s^2*||g||^2-(171/10)*s^2*||m-g||^2.
```

This is a polynomial identity with the main model's actual first-moment
coefficient. If s>=0, `<w,g>>=0`, `||g||^2<=L*<w,g>` and `19*s*L<=2`,
the first two terms sum to at most zero; the last is nonpositive for every
m. No alignment or smallness premise on the saved moment is needed.
The exact identity

```
19*V_s(w,m)-10*||w||^2 = 9*||w-19*s*m||^2
```

also gives `||w||^2 <= (19/10)*V_s(w,m)`. The potential bounds ordinary
unused-class contrast energy, not just a transformed position coordinate.
Induction yields nonincrease over every finite prefix of the declared
fixed-s recurrence. The CE specialization allows features, targets and
clipping to change at each step, provided the classes remain unused and
the explicit probability/feature-energy step condition holds each time.
The generic version instead takes a common L. Nonzero examples accompany
the conditional statements; the CE example has distinct unused readout
rows and nonzero saved memory. The sequence example demonstrates consistent
premises; it is not a discovery example.

This reference omits decay, coordinate-dependent scaling and the tied
encoder's current gradient. It does not show that unused energy itself is
monotone, that active contrast grows, or that prediction margins improve.
Changing s also changes V_s; one-step results with different s cannot be
chained without a bound on that change of potential.

### CL3. Exact full-model perturbations

Let g_R be clipped partial CE and g_E the clipped encoder contribution.
On a full readout coordinate, retain the actual new second moment v and
absolute-time index t+1. Write `b1=1-.9^(t+1)` and
`D=1/(sqrt(v)/sqrt(1-.95^(t+1))+eps)`. Let e_m and e_p denote moment
and parameter execution residuals relative to these exact-real expressions:

```
m_new = .9*m+.1*(g_R+g_E)+e_m,
w_new = (1-eta*rho)*w-eta*D*m_new/b1+e_p.
dm = m_new-m_R = .1*g_E+e_m,
dw = w_new-w_R = -eta*rho*w
     +(s-eta*D/b1)*m_new-s*.1*g_E+(e_p-s*e_m).
```

The scalar Lean identities use the existing buffered AdamW definitions,
with beta1=.9, beta2=.95 and both bias corrections. They are valid without
assuming the residuals small. Project the four position terms onto the
unused contrast block; projection commutes with scalar multiplication and
addition, but **not** with coordinate-dependent D. The native replay
therefore computes the adaptive term as `s*P_U(m_new)-eta*P_U(D*m_new)/b1`,
not by multiplying separately centered arrays. Encoder and execution terms
also carry their respective moment perturbations; decay and adaptive
scaling have zero moment perturbation in this decomposition.

For any reference state (w_R,m_R) and total perturbation (dw,dm),

```
V_s(w_R+dw,m_R+dm)-V_s(w_R,m_R) =
  2*<w_R-9*s*m_R,dw-9*s*dm>+180*s^2*<m_R,dm>+V_s(dw,dm).
```

The signed linear credits add over the four sources. The quadratic term
uses their **total** perturbation, preserving all cross terms. Adding CL2's
reference change gives the exact full-update potential identity. A concrete
one-dimensional example has a decreasing reference and an increasing
perturbed potential; it demonstrates that algebra alone does not sign the
perturbation, not a failure of the actual CE or GPT trajectory. A prospective
full-model certificate must bound these terms before the unknown update.

### CL4. Scope of the numerical diagnosis

The frozen `experiments/ce_lyapunov/PROTOCOL.md` uses the original next
minibatch at 144 saved states, preserving the actual model and optimizer.
An independent FP64 calculation supplies partial CE from executed features;
GPU/CPU derivative disagreement joins the execution residual. The raw
reference uses `s=eta*min(D)/b1` over the full readout. A separate stable
reference caps s by `1/(19*L)` when L>0, without changing training. These
statewise quantities use the actual next second moment and current q;
they are diagnoses, not forecasts. The declared tolerances test finite
executions and are not universal floating-point theorems. Results and
complete reproduction commands belong in the experiment README. Neither
these replays nor CL1–CL3 prove reliable discovery, full-model future
gradient alignment, or a useful active-share floor over the entire hold.

## Checkpoint-fixed feature metrics

The scalar reference in CL2 treats every coordinate-scale difference as a
perturbation. The following authored extension absorbs feature-dependent
scaling that is shared across unused classes. It leaves class-dependent
scaling and tied encoder updates explicit. The files are
`GradientEvidence/FeatureMetricCE.lean`, `FeatureMetricMomentum.lean`, and
`FeatureMetricControl.lean`; these do not assert a new manuscript theorem.

### FM1. Weighted damping from actual partial CE

Fix nonnegative feature weights r_k. Apply finite Cauchy–Schwarz to each
coordinate of `mean_i A_i*T_ik`, multiply by r_k, and sum. This yields

```
sum_k r_k*(mean_i A_i*T_ik)^2 <=
  (mean_i A_i^2)*(mean_i sum_k r_k*T_ik^2).
```

Use CL1's pair probability inequality with A_i=p_ic-p_id. For two classes
that never appear as targets, the actual partial-gradient pair obeys

```
sum_k r_k*(G_ck-G_dk)^2 <= L_r*<E_c-E_d,G_c-G_d>,
L_r = q*mean_i sum_k r_k*T_ik^2.
```

The radial product is the same unweighted nonnegative product as CG1.
Multiply the centered pair-product identity at feature k by r_k, then sum
over features. Applying the pair inequality to every unused pair gives
`sum_(u,k) r_k*G_U(u,k)^2 <= L_r*<E_U,G_U>`. The class means are computed;
the feature weights do not depend on the unused class index. This is why
scaling and centering commute. No corresponding commutation is asserted
for the actual row-dependent Adam diagonal. Empty batches use the existing
zero-mean convention; the experiment uses nonempty batches and 64 distinct
unused classes. As in CL1, repeated class indices are also permitted formally.

Common clipping a>=0 replaces G_U by g=a*G_U and L_r by aL_r.
If r_k<=R and mean feature energy is at most M, weighted mean feature energy
is at most R*M. Hence q=1 and 0<=a<=1 give a coefficient at most R*M without
observing a future probability or clipping scalar. This still requires the
declared feature-energy bound. In particular, a future sequence needs it
at each step; a single measured feature norm does not supply that premise.

### FM2. A fixed metric absorbs feature scaling and arbitrary memory

Require r_k>0. For centered unused readout w and arbitrary saved first
moment m, use the actual main beta1=.9 and the scaled reference

```
m_R = .9*m+.1*g,        w_R = w-s*r*m_R,
V_(r,s)(w,m) = sum_(u,k) ((w-9*s*r*m)^2/r+90*s^2*r*m^2).
```

Coordinatewise expansion and cancellation of positive r gives the identity

```
V_(r,s)(w_R,m_R)-V_(r,s)(w,m) =
  -2*s*<w,g>+19*s^2*sum r*g^2-(171/10)*s^2*sum r*(m-g)^2.
```

Thus s>=0 and `19*s*L_r<=2`, with the clipped coefficient, prove
nonincrease for every saved moment. No alignment, zero-memory or isotropic
feature assumption is needed. The universal-probability corollary instead
uses `19*s*R*M<=2` and the stated metric, feature and clipping bounds.
Induction proves the fixed-r, fixed-s result over any finite prefix of the
declared partial-CE recurrence. Features, targets and clipping may change,
with unused-class and coefficient hypotheses made explicit at every step.

The exact coercivity identity is

```
19*V_(r,s)(w,m)-10*sum w^2/r = 9*sum (w-19*s*r*m)^2/r.
```

Consequently, for 0<=R and every r_k<=R,
`||w||^2 <= (19/10)*R*V_(r,s)(w,m)`. The largest-weight penalty cannot be
omitted when comparing with ordinary geometric energy. Nonzero CE and
memory examples accompany the conditional statements. A sequence example
shows their premises are consistent, not that discovery occurs. The metric
and scalar step cannot change silently when chaining this theorem.

### FM3. Class-dependent and encoder remainders remain explicit

Apply CL3's buffered AdamW identity with scalar reference step s*r_k at
each full readout coordinate. For absolute time t+1, actual updated v and
`D=1/(sqrt(v)/sqrt(1-.95^(t+1))+eps)`, retain

```
dm = .1*g_E+e_m,
dw = -eta*rho*w +(s*r-eta*D/(1-.9^(t+1)))*m_new
     -s*r*.1*g_E+(e_p-s*r*e_m).
```

Project the four full-coordinate position terms onto unused class
contrasts. Feature scaling r commutes with that projection; D generally
does not. The adaptive remainder is therefore exactly
`s*r*P_U(m_new)-eta*P_U(D*m_new)/(1-.9^(t+1))`. It includes both differences
between class rows and changes since the checkpoint metric was frozen.
The encoder and execution sources retain their moment perturbations.

For reference (w_R,m_R) and total perturbation (dw,dm), direct expansion gives

```
V_(r,s)(w_R+dw,m_R+dm)-V_(r,s)(w_R,m_R) =
  sum [2*(w_R-9*s*r*m_R)*(dw-9*s*r*dm)/r+180*s^2*r*m_R*dm]
  +V_(r,s)(dw,dm).
```

The multi-source theorem distributes only the signed linear term, keeping
the quadratic energy of the total perturbation. The full-step theorem adds
FM2's exact reference change. These are identities, not advance estimates
on the signs or magnitudes of unknown encoder and adaptive remainders.

### FM4. Checkpoint construction and scope of the comparison

Task #207 fixes each metric from the old checkpoint before its separate
verifier opens any #206 next-update evidence. For t>0, compute the old
bias-corrected D_old; at the zero state take D_old=1/eps. Let alpha be its
minimum over the complete readout and set
`r_k=min_(unused c) D_old(c,k)/alpha`. The isotropic control has r=1 and
uses the same checkpoint scalar `s=eta(t)*alpha/(1-.9^(t+1))`. Thus the
comparison does not use the next second moment to choose its metric.

The next minibatch's exact-real partial CE is independently computed in
FP64 from the executed features. Its measured q supplies a diagnostic L_r;
the separately reported conservative condition uses q=1, clip<=1 and the
declared numerical feature cap. A stable reference caps s using the current
L_r, so it is explicitly a diagnostic rather than a forecast. All four
metric/step tracks retain every case, source, full increase and coefficient
failure. Since different metrics define different potentials, changes in
their numerical values are not effects of intervening on the optimizer.
The exact-real Lean results and the finite numerical allowances remain
separate. Full-model future control, discovery and prediction certificates
are not consequences of these comparisons.

## Prospective probability control

This continuation addresses FM4's future-probability premise. These are
authored consequences of the actual finite softmax and bilinear readout,
not new claims of the original flat-spectrum manuscript. The files are
`Geometry/ProbabilityMotion.lean`, `GradientEvidence/ProbabilityMotion.lean`
and `GradientEvidence/ProspectiveCE.lean`.

### PM1. Keep the class's own denominator term

For actual old and future logits z,z', fix a class u. Suppose every rival
satisfies `(z'_u-z'_c)-(z_u-z_c)<=D_c`. Rearranging and exponentiating gives
`exp(z'_c-z'_u)>=exp(z_c-z_u-D_c)`. Therefore

```
p'_u = 1/(1+sum_(c!=u) exp(z'_c-z'_u))
     <= 1/(1+sum_(c!=u) exp(z_c-z_u-D_c)).
```

No rival is discarded, and the self term is exactly one. When every D_c=D,
the old odds sum is `(1-p_u)/p_u`. Substitution gives the sharp ceiling

```
F(p_u,D) = p_u*exp(D)/(1-p_u+p_u*exp(D)).
```

For `0<=p<=q<=1`, direct cross multiplication gives `F(p,D)<=F(q,D)`;
the denominator is positive and `0<=F(q,D)<=1`, for either sign of D.
For D>=0 this also implies the less precise exponential bound `p*exp(D)`.
For negative D the self term matters: that simpler exponential bound need
not hold. The binary witness z=(log 4,0), z'=(log 4,log 2) has old
probability 1/5 and new probability 1/3 for the second class; F attains 1/3.

### PM2. Accumulate motion before using the ceiling

If each increment of `z_u-z_c` is at most D_t, telescoping gives total
deterioration at most `sum_(t<n) D_t` at every finite horizon n. PM1 then
bounds its future probability using only the initial probability ceiling
and those declared budgets. Budgets may be signed; this identity does not
assert that the actual probabilities or contrasts change monotonically.
To cover an arbitrary future training minibatch, these bounds must hold
for each input in the training population. The maximum probability on one
observed minibatch is not a ceiling over that population.

### PM3. Account for full readout and feature motion

Write the old class contrast as w=E_u-E_c and the feature as x, including
final normalization. At a later state they are w+dw and x+dx. The exact
score change is `<dw,x>+<w,dx>+<dw,dx>`. If

```
||w||<=A, ||x||<=X, ||dw||<=R, ||dx||<=S,
```

Cauchy–Schwarz gives absolute change at most `R*X+A*S+R*S`. The Lean
statements use the equivalent squared-energy premises and nonnegative
caps. For executed logits, an explicit allowance on the change of the
execution contrast adds e. Use that complete budget as D_c in PM1.

The readout and feature changes can be coupled: independence is unnecessary,
and the mixed term is retained. Class-dependent scaling is not assumed to
commute with centering. The parameter and feature radii must come from a
valid source-motion analysis; this theorem does not infer them from small
loss, a measured future feature displacement or a single checkpoint norm.
In particular, the feature radius must cover every moving encoder parameter
and normalization. Existing warm-moment Adam tubes address parameter motion;
their usefulness after propagation through the nonlinear model remains to
be measured in the new experiment.

### PM4. Supply the actual CE coefficient prospectively

Suppose old logits on every possible training input give unused-class
probabilities at most q. A uniform valid contrast-motion budget D gives
the new ceiling F(q,D) by PM1. If the feature weights satisfy r_k<=R,
the new mean ordinary feature energy is at most M, and clipping is in [0,1],
the weighted partial-CE cocoercivity coefficient is at most `F(q,D)*R*M`.
Hence

```
19*s*F(q,D)*R*M <= 2
```

implies the weighted reference potential's nonincrease at that future state,
for arbitrary saved first moment. This composes PM1 with the actual
partial-CE theorem FM2. The future probability is no longer a separate
smallness assumption. The motion and feature-energy premises are explicit,
and their prospective validity is still required.

The reference conclusion does not control the signed full-step perturbations
of FM3. Nor do statewise conditions with different s or r automatically chain
into one potential over time. The complete task still requires useful motion
and forcing bounds, a geometry/margin bridge and a discovery argument. The
new [experiment](../experiments/probability_tube/README.md) records source
recovery and verification status separately from those open obligations.

## Full encoder motion

This continuation discharges PM3's separate feature-motion premise from
checkpoint parameter caps. Its reference graph is the plain arm of the
committed `experiments/flat_approx/fixtures/reference.rs` and `src/model.rs`
at 62638521. The Lean namespace is `Transformer.Rotate90.FeatureMotion`.
The original trainer's exact bytes and execution remain prerequisites for
recreating the training evidence; a reference forward equation is insufficient
to recover its initializer, sampler and full training process.
The trainer has now been recovered at b70811e with its exact pinned digest;
its matrix aliases, buffer layout, four normalizations and product kernel
agree with the reference graph inspected here. New execution evidence is pending.

### EM1. Moving inputs and linear layers

Let each shared embedding row have displacement at most R_c. The fixed
quarter-turn J is an isometry, so the input E[a]+J E[b] moves by at most
R_a+R_b, also when a=b. A real linear layer with checkpoint matrix A,
operator cap L>=||A||, matrix displacement ||A'-A||<=R and input displacement
||x'-x||<=r satisfies

```
||A'x'-Ax|| <= (L+R)*r + R*||x||.
```

Expand the difference as `A*dx+dA*x+dA*dx`; the final product is essential.
All terms on the right are checkpoint quantities or declared displacement
caps. In contrast, a one-update forecast that computes ||A'|| from the known
proposed parameters cannot be used unchanged for unknown future updates.

`matrixMap_apply` identifies the continuous real map with ordinary dense
matrix multiplication; `matrixMap_sub` identifies its displacement.
`matrixMap_norm_le` supplies the explicit Frobenius cap via FT2. The native
constructor uses these Frobenius caps, including every repeated entry of an
expanded tied matrix. It does not substitute an estimated leading singular
value. The expanded input width is 128, hidden width 682 and parameter-buffer
length 274304. The existing `optimizer_efficiency/layout.rs` records signed
matrix/first-moment aliases and unsigned second-moment aliases; counting
expanded coordinates conservatively does not assert that they are independent
trainable owners.

### EM2. Every normalization retains its checkpoint denominator

For real width d>0 and epsilon>0, put s(x)=sqrt(||x||^2+d*epsilon).
The exact source RMS is `sqrt(d)*x/s(x)`, as `rms_eq_source` proves.
Given an incoming displacement cap r, intersect three valid output caps:

```
2*sqrt(d)                                      (range diameter),
2*sqrt(d)*r/s(x)                               (FF2),
sqrt(d)*r/sqrt(s(x)*(s(x)-r))   when r<s(x)      (FT1).
```

When r>=s(x), omit the local secant candidate. No future denominator enters.
The first bound follows from ||RMS(x)||<=sqrt(d); the final feature therefore
has squared energy at most d at every parameter state. The same argument
applies to the input, branch-input, branch-output and final normalizations.
Near-zero inputs retain epsilon rather than assuming a nonzero raw norm.

### EM3. Complex products retain both moving factors

For the actual paired kernel
`P(g,u)_j=((g_re*u_re-g_im*u_im),(g_re*u_im+g_im*u_re))/sqrt(2)`,
the scalar modulus identity gives

```
||P(g,u)||^2 = sum_j |g_j|^2*|u_j|^2 / 2
            <= ||g||^2*||u||^2 / 2.
```

Expanding the finite change into `P(dg,u)+P(g,du)+P(dg,du)` gives
`(r_g*||u||+||g||*r_u+r_g*r_u)/sqrt(2)`. The last term is required even
when both old factors vanish. The input and output linear maps may be
arbitrary real maps; source complex-linear ties are a valid special case.
Logical pairs replace the native hidden buffer's separated real/imaginary
halves, without changing its Euclidean norm or calibrated product.

### EM4. Compose the whole encoder and its tied readout

Evaluate the checkpoint graph once. Transport `R_a+R_b` through the first
two RMS bounds, then separately through the moving gate and value maps.
Apply EM3, the moving output map and branch RMS. Add the first RMS radius
for the residual, then apply the final RMS bound. `feature_motion` proves
that the resulting explicit S bounds the final feature displacement. Its
only motion premises are parameter caps. `feature_eq_source` and
`logit_eq_source` prove equality with the previously defined expanded
complex-linear source graph, including all normalizations and ties.

For readout classes u,c, R_u+R_c bounds their contrast-row displacement.
PM3 then gives

```
D_uc = (R_u+R_c)*||x|| + ||E_u-E_c||*S + (R_u+R_c)*S.
```

`probability_le_encoder_motion` feeds these bounds into PM1 for every rival.
The old executed logits are the old exact scores plus their execution
residuals. A separately bounded CHANGE of the execution contrast is added
to D_uc. Dropping the old residual while calling a future residual a change
would be incorrect; the native interface accepts both explicitly.
`feature_energy_le` also supplies PM4's ordinary energy cap d, rather than
assuming it of an unknown future feature. These results still control the
partial-CE reference condition, not the signed full-update perturbations FM3.

### EM5. Obtain full-block parameter caps from saved Adam moments

For the selected main-model adapter's betas (0.9,0.95), retain checkpoint
p,m,v for any complete coordinate block, with v>=0 and
m_k^2<=r_k*v_k. FF4 gives an arbitrary-gradient future tube about q_n*p,
where q_n is the known product of weight-decay factors. Recenter it at p:

```
||p_n-p|| <= |q_n-1|*||p|| + forcingRadius_n.
forcingRadius_(h+1) = (1-eta_h*rho)*forcingRadius_h
                   + eta_h*bias(start+h+1)*sqrt(sum_k futureCone(r_k,h+1)).
futureCone(r,h) = (81/95)^h*r + (1-(81/95)^h)*(19/14).
```

`warm_adam_checkpoint_motion` proves this for every finite prefix and any
future gradients, including absolute-time bias correction and arbitrary
valid saved moments. Applying it separately to each embedding row and each
full matrix yields the native parameter caps. The constructor never receives
future parameters, moments, gradients or feature values. Bounds count the
whole model, rather than the earlier readout-only subset. A zero variance
with nonzero first moment is rejected. No observed future direction norm is
smuggled into a hypothesis.

These are exact-real optimizer bounds evaluated in FP64 by the native code.
The floating-point optimizer recurrence, serialized source correspondence,
outward rounding and execution-error allowances remain separate obligations.
The tests use explicitly synthetic parameter vectors with the source ties;
they do not invoke the original initializer or train a replacement baseline.
They check each intermediate stage, original dimensions, near-zero inputs,
all class contrasts and probabilities, and 24 arbitrary-gradient warm-Adam
prefixes across every parameter block. Eight native tests pass, including
the four earlier PM tests. This establishes implementation checks, not a
useful horizon on any of the 144 frozen training checkpoints.

## Pairwise encoder amplification

### PA1. Use the pointwise structure of the complex product

EM3's full-vector amplification is conservative when gate and value energy
is distributed over many logical pairs. For the unchanged calibrated source
product, the complex-modulus identity gives

```
||product(g,u)||² = (1/2) sum_j |g_j|² |u_j|².
```

If every old gate pair has amplitude at most G and every old value pair has
amplitude at most U, the two linear terms in finite product motion obey

```
||product(g+dg,u+du)-product(g,u)||
  <= (U*||dg|| + G*||du|| + ||dg||*||du||)/sqrt(2).
```

G and U depend only on the checkpoint activations. The mixed term cannot
use a small old amplitude: both factors are unknown perturbations. Apply
this bound inside EM4, keeping all RMS normalizations, moving matrices and
residual terms. Intersect the resulting feature bound with EM4's original
one. Every checkpoint operator cap still requires its own proof; the
existing FT2 Gram bound is an available alternative to Frobenius caps.
No estimated singular value is treated as a certified operator norm.

This continuation is independent of the frozen `1fb060b9` training audit.
It does not change that audit's formulas or numerical allowances. A sharper
local formula alone is not a useful training horizon or a bound on the
full-update potential; those remain separate experimental and mathematical
obligations.

### PA2. Use a proved Gram cap for the same moving maps

For an old real matrix A, FT2 bounds its operator norm by
`(sum_ij (Aᵀ*A)_ij²)^(1/4)`; the row Gram gives the same valid bound with
the smaller matrix dimension when appropriate. The source wrapper
`matrixMap` represents exactly that real matrix action. PA2 exposes this
correspondence for both orientations and for an upper bound on Gram energy.
The native continuation reuses FT2's outward-expanded scalar Gram calculation,
including its checks against identity, rank-one and transposed matrices.
This certifies no whole forward execution: the remaining activation/radius
arithmetic and CUDA execution retain their separate numerical obligations.

## Query-weighted feature covariance

The complete recreated comparison in `experiments/probability_tube` validates
both encoder forecasts but yields zero useful positive raw CE horizons. This
continuation examines another loss of information: replacing every example's
probability by the maximum, then replacing a feature operator by its trace.
The mathematical reference remains actual partial CE with arbitrary saved
momentum; full tied AdamW perturbations remain separate.

### QG1. Retain individual probabilities and the feature operator

For a batch of n examples, let T_i be its actual feature vectors, q_i>0 a
ceiling on every unused-class probability for that example, and r_k>0 the
fixed feature metric. Define the matrix with feature rows and example columns

```
W_ki = sqrt(r_k)*T_ik*sqrt(q_i),
L_op = ||W||_op^2/n,
L_trace = mean_i q_i*sum_k r_k*T_ik^2.
```

For arbitrary real a_i, apply W to a_i/sqrt(q_i). The probability weights
cancel from the true mean, yielding

```
sum_k r_k*(mean_i a_i*T_ik)^2 <= L_op*mean_i (a_i^2/q_i).
L_op <= L_trace <= Q*mean_i sum_k r_k*T_ik^2,   q_i <= Q.
```

The first inequality is the L2 operator bound; the second is the Frobenius
bound, and the third is nonnegative weighted summation. Two orthogonal unit
features with q_i=r_k=1 give L_op=1/2 and L_trace=1. A flat feature covariance
can therefore improve this coefficient without assuming that training
discovers such a covariance. The Lean convention also permits empty batches,
for which means and coefficients are zero; the source uses nonempty batches.

### QG2. Apply the operator to the actual CE derivative

For two unused classes c,d, write a_i=p_ic-p_id. CL1 gives
`a_i^2 <= q_i*(z_ic-z_id)*a_i`. Each q_i is positive because it bounds an
actual finite-logit probability. Divide before summing, then use QG1 and
CG1's exact unused-pair gradient identity:

```
sum_k r_k*(G_ck-G_dk)^2 <= L_op*<E_c-E_d,G_c-G_d>.
```

Sum the same coefficient over every unused pair and apply the computed
centering identity. Thus the whole centered unused gradient satisfies
`sum r*g_U^2 <= L_op*<E_U,g_U>`. Common clipping a>=0 changes the coefficient
to a*L_op. FM2's exact momentum identity then uses `19*s*a*L_op <= 2` in the
same fixed metric, with no alignment premise on saved memory. Changing the
coefficient does not change the executed optimizer or remove its encoder,
class-dependent scaling, decay and execution terms.

### QG3. Transport the coefficient through full feature motion

Use the same q_i and r_k in an old and a future table. Suppose r_k<=R,
R>=0 and `||T'_i-T_i|| <= S_i`. The weighted matrix difference has squared
Frobenius norm at most `R*sum_i q_i*S_i^2`. For any certified old operator
cap B, the triangle inequality gives

```
L_op(T') <= (B + sqrt(R*sum_i q_i*S_i^2))^2/n.
```

FT2 supplies B from the complete row or column Gram, including correlations.
In the smaller feature Gram, define `C_kl=sum_i q_i*T_ik*T_il`; its squared
Frobenius energy is `sum_kl r_k*r_l*C_kl^2`. An upper bound U on that explicit
energy gives `B=U^(1/4)`. A pointwise future feature-energy cap M also gives
`L_op(T') <= R*M*mean_i q_i`; intersecting both bounds retains the trace
control. These formulas use every probability and every feature radius.

The feature radius must cover executed features, not just the ideal encoder.
The previous ideal radius plus both endpoint execution errors supplies this
radius. Likewise, a GPU probability ceiling and a readout dot error at most
e give a real-CE ceiling via PM1 with an additional contrast allowance 2e.
The new numerical comparison must retain this conversion explicitly. It must
not infer a universal floating-point theorem from endpoint checks.

### QG4. Known batches and the remaining full-model obligation

The original source plan uses batches of 512, with a shorter final batch in
each epoch. Apply each coefficient to its actual finite batch and take the
maximum over the known batch sequence in the declared window. Varying batch
sizes, changing features and changing clipping do not change the fixed-r,
fixed-s momentum argument. Bounds and conditions must hold at every prefix;
an endpoint diagnosis alone cannot supply them.

The frozen native continuation must keep all 144 anchors, all 1800 horizons,
both source probability forecasts and both metrics. A useful partial-CE
condition still needs separate control of full adaptive/encoder perturbations
before it becomes a training stability result. No new numerical gain is
asserted by this derivation.

## Training reaches the optimal fit (added)

Lemma 1 ends with a modeling step: "Training with weight decay then behaves like
minimum-nuclear-norm interpolation from Ω. This is the modeling step. It is exact for a
pure factorization; for the real network it is an assumption." The Lean status row
*Lemma 1, "exact for a pure factorization"* proves the static half: weight decay at its
optimum is M. It leaves open that training reaches that optimum. This section proves it for
the pure factorization in the following form: a run of gradient descent with weight decay
that converges, converges to the optimum. The objective is not convex in the factors. It
has saddles, the origin among them whenever the loss gradient there exceeds λ, so the claim
is that a run started at full rank cannot end at one. Full rank needs width `m + n`; WD10
lowers the width to `min(m, n)` for almost every start and shows that it cannot go lower.

Notation: `⟨A, B⟩ = Re tr(Aᴴ B)` is the real inner product of complex matrices and
`‖A‖² = ⟨A, A⟩ = ‖A‖_S2²`. The loss `L` is a real function of an `m × n` matrix together
with a field `G` that satisfies the first-order inequality

```
L(X) + ⟨G(X), Y − X⟩ ≤ L(Y)    for all X, Y.
```

This says that `L` is convex and `G` is its gradient (WD3). For the squared loss on Ω,
`L(X) = ½ Σ_Ω |X_ij − K_ij|²` and `G(X) = P_Ω(X − K)`.

### WD1. The model and its lifted form

The model is `X = U Vᴴ` with `U` of size `m × k` and `V` of size `n × k`. Its weight-decayed
loss and gradient descent with step η are

```
F(U, V) = L(U Vᴴ) + λ · ½(‖U‖² + ‖V‖²),
U ← U − η (G V + λ U),    V ← V − η (Gᴴ U + λ V),    G = G(U Vᴴ).
```

Stack the factors, `W = [U; V]`, and put `M = [[λ, G], [Gᴴ, λ]]`, a Hermitian matrix of
size `m + n`. Then `M W = [G V + λ U; Gᴴ U + λ V]`, so a step is `W ← (1 − η M) W`, and

```
⟨W, M W⟩ = λ (‖U‖² + ‖V‖²) + 2 ⟨G, U Vᴴ⟩.
```

Since `2 |⟨G, U Vᴴ⟩| ≤ ‖G‖_S2 (‖U‖² + ‖V‖²)`, `M` is positive semidefinite when
`‖G‖_S2 ≤ λ`. The sharp condition has the operator norm of `G`; the examples need only this
one.

### WD2. A certificate for the global minimum

If `M W = 0` and `M` is positive semidefinite at `(U, V)`, then `F(U, V) ≤ F(U', V')` for
the factorizations of every width. Indeed `0 ≤ ⟨W', M W'⟩ = λ ‖W'‖² + 2 ⟨G, U' V'ᴴ⟩`, with
equality at `W`, and the first-order inequality at `X = U Vᴴ`, `Y = U' V'ᴴ` gives

```
F(U', V') − F(U, V) ≥ ⟨G, Y − X⟩ + (λ/2)(‖W'‖² − ‖W‖²) = ½⟨W', M W'⟩ − ½⟨W, M W⟩ ≥ 0.
```

Call a factorization with the least `F` among those of every width a decayed minimum. By
Lemma 1 it is the penalized form of M: for λ ≥ 0 its product minimizes `L(X) + λ ‖X‖_S1`
over all matrices, for λ > 0 its weight decay `½(‖U‖² + ‖V‖²)` equals `‖U Vᴴ‖_S1`, and for
λ > 0 these two properties characterize it.

### WD3. The update is the gradient

The other items use `G` only through the first-order inequality. If `G` is continuous, that
inequality makes it the gradient of `L`: along a curve `X + t H(t)`, applying it at both
ends,

```
t ⟨G(X), H(t)⟩ ≤ L(X + t H(t)) − L(X) ≤ t ⟨G(X + t H(t)), H(t)⟩,
```

and both bounds have slope `⟨G(X), H(0)⟩` at `t = 0`. The squared Frobenius norm satisfies
the same inequality with `G(X) = 2X`. With
`(U + tA)(V + tB)ᴴ = U Vᴴ + t (A Vᴴ + U Bᴴ + t A Bᴴ)` this gives

```
d/dt F(U + tA, V + tB) at t = 0  =  ⟨G V + λ U, A⟩ + ⟨Gᴴ U + λ V, B⟩.
```

So the update of WD1 is gradient descent on `F`, not a definition of it.

### WD4. No negative direction survives at the limit

Let `W_{t+1} = (1 − η M_t) W_t` with η > 0, every `1 − η M_t` positive definite,
`M_t → M∞` Hermitian, `W_t` bounded and `W_0` of full row rank. Then `M∞` is positive
semidefinite.

Suppose `xᴴ M∞ x < 0`. The characteristic polynomial has finitely many roots, so some small
`c > 0` makes `S = M∞ + c` invertible with `xᴴ S x < 0` still. Put `T = 1 − η M∞ = a − η S`
with `a = 1 + ηc > 1`; `T` is positive semidefinite as a limit of the steps. Let
`Q(Z) = −⟨Z, S Z⟩`. For one perturbed step `Z' = (T − η E) Z`, with `Y = S Z`,

```
Q(Z') = a² Q(Z) + a η ‖Y‖² + η ⟨Y, T Y⟩ + 2η ⟨T Y, E Z⟩ − η² ⟨E Z, S E Z⟩.
```

As `‖Z‖ ≤ ‖S⁻¹‖ ‖Y‖`, the last two terms are at least
`−η ‖Y‖² (2 ‖T‖ ‖E‖ ‖S⁻¹‖ + η ‖S‖ ‖E‖² ‖S⁻¹‖²)`, so `Q(Z') ≥ a² Q(Z)` once `‖E‖ ≤ ε`,
for an ε that depends on `S`, `a` and η only. Now `W_0` has a right inverse `R`, and
`W_t R` is a product of the invertible steps, hence invertible. Take `t₁` with
`‖M_t − M∞‖ ≤ ε` from `t₁` on and `z = R (W_{t₁} R)⁻¹ x`. The columns `Z_j = W_{t₁+j} z`
start at `Z_0 = x`, solve `Z_{j+1} = (T − η (M_{t₁+j} − M∞)) Z_j` and are bounded, yet
`Q(Z_j) ≥ a^{2j} Q(x)` grows without bound while `Q(Z_j) ≤ ‖S‖ ‖Z_j‖²`. Contradiction.

Full row rank is what lets a column of the run point along `x`. It needs width
`k ≥ m + n` and then holds for almost every start. Without it the claim fails: `W_0 = 0`
stays at `0` whatever `M` is.

### WD5. A convergent run ends at a global minimum

Let `L` satisfy the first-order inequality with a continuous `G`, let η > 0, let
`[U_0; V_0]` have full row rank, and let every step matrix `1 − η M_t` be positive definite.
If the run of WD1 converges, `(U_t, V_t) → (U, V)`, then `(U, V)` is a decayed minimum:
`F(U, V) ≤ F(U', V')` for the factorizations of every width. So for λ ≥ 0 the trained
product `U Vᴴ` minimizes `L(X) + λ ‖X‖_S1` over all matrices, and for λ > 0 the trained
weight decay equals `‖U Vᴴ‖_S1`.

Proof: `M_t → M∞ = M(U, V)` since `G` is continuous, and a convergent run is bounded, so
`M∞` is positive semidefinite by WD4. The limit of `W_{t+1} = (1 − η M_t) W_t` is
`W = (1 − η M∞) W`, that is `M∞ W = 0`. WD2 applies.

A run that meets every hypothesis and has a nonzero limit: `m = n = 1`, `L(X) = −Re X`,
`G = −1`, λ = 1, η = 1/4, `U_0 = [1 0]`, `V_0 = [0 1]`. A step keeps `U + V` and halves
`U − V`, so the run converges to `U = V = [½ ½]`, where `F = 0`. This is the minimum, as
`F(u, v) = ½ |u − v|²` for vectors `u`, `v`.

Three hypotheses remain. The run converges: WD8 proves this for a real-analytic loss
controlled in a ball, by Łojasiewicz's theorem. The start has full row rank, which needs width
at least `m + n`; WD10 replaces it by width `min(m, n)` and almost every start. The steps are
admissible, which WD6 reduces to a bound on η. The second half of the proof uses none of
them: the limit of a convergent run of any width and rank is stationary, `M∞ W = 0`.

### WD6. Admissible steps

`1 − η M` is positive definite when `η (|λ| + ‖G‖_S2) < 1`, since by WD1
`⟨W, (1 − η M) W⟩ = ‖W‖² − η ⟨W, M W⟩ ≥ (1 − η (|λ| + ‖G‖_S2)) ‖W‖²`.

The bound on η is explicit when the gradient is controlled in a ball. Let λ > 0, let ρ
satisfy `‖U_0‖² + ‖V_0‖² ≤ ρ`, and suppose that at the matrices with `‖X‖ ≤ ρ/2`

```
‖G(X)‖ ≤ γ    and    ⟨G(X), X⟩ ≥ −b,    where 4b ≤ λ ρ.
```

In the ball `‖W‖² ≤ ρ`, `‖U Vᴴ‖ ≤ ‖W‖²/2 ≤ ρ/2`, so both bounds hold at `X = U Vᴴ`.
Suppose

```
2η (λ² + γ²) ≤ λ.
```

Then `η (λ + γ) < 1`, so every step in the ball is admissible. A step also keeps the ball.
With `w = ‖W‖²`,

```
‖(1 − η M) W‖² = w − 2η ⟨W, M W⟩ + η² ‖M W‖² ≤ (1 − η λ) w + 4η b ≤ ρ,
```

because `⟨W, M W⟩ = λ w + 2 ⟨G, U Vᴴ⟩ ≥ λ w − 2b` and `‖M W‖² ≤ 2 (λ² + γ²) w`. So a run
from the ball stays in it with admissible steps, and WD5 needs only full row rank, this η,
and convergence.

For the squared loss on Ω with `‖K‖ ≤ κ` the constants are `γ = ρ/2 + κ`, as
`‖G‖ ≤ ‖X − K‖`, and `b = κ²/4`, as `⟨G, X⟩ = ‖G‖² + ⟨G, K⟩ ≥ −‖K‖²/4` for
`G = P_Ω(X − K)`. The conditions read `κ² ≤ λ ρ` and `2η (λ² + (ρ/2 + κ)²) ≤ λ`.

### WD7. Vanishing weight decay: the optimal fits of M

Take the squared loss on the seen cells, `L(X) = ½ ‖P_Ω(X − K)‖²` with
`G(X) = P_Ω(X − K)`. It satisfies the first-order inequality, the two sides differing by
`½ ‖P_Ω(Y − X)‖²`; `G` is continuous; and `L(X) = 0` exactly when `X = K` on Ω. Call a
decayed minimum of this loss a trained fit. By WD5 a convergent run ends at one, and by WD8
the run converges.

Let `X'` be any matrix equal to `K` on Ω. WD2 at `Y = X'` gives, for λ ≥ 0,

```
L(U Vᴴ) + λ ‖U Vᴴ‖_S1 ≤ λ ‖X'‖_S1.
```

So for λ > 0 a trained fit has `‖U Vᴴ‖_S1 ≤ ‖X'‖_S1` and `L(U Vᴴ) ≤ λ ‖X'‖_S1`: no more
nuclear norm than any exact fit, and a loss that vanishes with λ. As λ → 0⁺ this gives
three statements.

1. Limits are optimal fits. If trained fits `(U_j, V_j)` for `λ_j → 0⁺` converge to
   `(U, V)`, then `U Vᴴ = K` on Ω, and `(U, V)` has the least weight decay among the exact
   fits of every width: for an exact fit of weight decay `w'`, `λ_j w_j ≤ λ_j w'`. So the
   limit is an optimal fit of the Lean status row *Lemma 1, "exact for a pure
   factorization"*, and `U Vᴴ` is a minimizer of M. Such limits exist. A trained fit has
   `‖U‖² + ‖V‖² = 2 ‖U Vᴴ‖_S1 ≤ 2 ‖K‖_S1`, by WD2 and the bound above at `X' = K`, so the
   factors of trained fits of one width lie in a compact set. Hence every sequence of trained
   fits of one width for `λ_j → 0⁺` has a subsequence whose factors converge, to an optimal
   fit. Trained fits exist at every width `r ≥ min(m, n)` (WD10), so an optimal fit of every
   such width exists.
2. Theorem A for training. If `K` is not a minimizer of `‖·‖_S1` among the matrices equal
   to it on Ω, take `X'` with `‖X'‖_S1 < ‖K‖_S1`. Every trained fit, for every λ > 0 and
   every width, has `‖U Vᴴ‖_S1 ≤ ‖K‖_S1 − δ` with `δ = ‖K‖_S1 − ‖X'‖_S1`. By Theorem A
   this holds for every square `K` with equal singular values and no zero entry, on every
   proper Ω: training with weight decay does not return such a kernel, however small λ.
3. Theorem E for training. If `K` is the only minimizer of M, the trained products
   converge to `K` as λ → 0⁺; the factors need not converge. The products are bounded by
   `‖K‖_S1` and their loss tends to `0`, so a limit point `X` equals `K` on Ω, and
   `‖X‖_S1 ≤ ‖K‖_S1` because `‖X‖_S1 = Re tr(Zᴴ X)` for a contraction `Z` while
   `Re tr(Zᴴ U_j V_jᴴ) ≤ ‖K‖_S1`. Hence `X = K`. By Theorem E this covers `u vᵀ` with
   `|u_i v_j| = 1` when Ω is connected and contains a permutation.

Example: for the `1 × 1` kernel `K = 1` seen on its cell and `0 ≤ s ≤ 1`, `U = V = 1 − s`
is a trained fit for `λ = s (2 − s)`: `G = (1 − s)² − 1 = −λ` and
`M = λ [[1, −1], [−1, 1]]`. The trained product is `1 − λ`. Weight decay shrinks the
singular value by λ, and the product tends to `K`.

The factors need not converge, and an optimal fit is not unique. The weight-decayed loss
does not change under `(U, V) ↦ (−U, −V)`: the product and the two norms are the same. So
the sign change of a trained fit is a trained fit, for every loss, and that of an optimal
fit is an optimal fit; an optimal fit with `U ≠ 0` is not the only one. If trained fits
`(U_j, V_j)` converge to factors with `U ≠ 0`, the fits `((−1)^j U_j, (−1)^j V_j)` are
trained for the same `λ_j` and have the same products, and their factors do not converge:
the even ones tend to `U` and the odd ones to `−U`. In the example, with `s_j = 1 / (j + 1)`,
the factors `(−1)^j (1 − s_j)` do not converge while the products tend to `K = 1`, and
`(1, 1)` and `(−1, −1)` are optimal fits.

### WD8. The run converges

WD5 assumes that the run converges. For a real-analytic loss controlled in a ball this is a
theorem: the Łojasiewicz inequality turns a cluster point of a descent into its limit. The
squared loss, a polynomial, is the first case.

A function `F` on a real inner product space satisfies the Łojasiewicz inequality at `p`
with exponent `0 ≤ a < 1` if for some `c > 0`

```
c |F(y) − F(p)|^a ≤ ‖∇F(y)‖    for y near p with F(y) slightly above F(p).
```

Łojasiewicz's theorem: a real-analytic function on a finite-dimensional space satisfies it
at every point, degenerate critical points included. The theorem is formalized in this
repository (`Transformer.Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`),
and so is the convergence theorem of Absil, Mahony and Andrews (2005, Theorem 3.2) in this
form: if every step of gradient descent with step η lowers `F` by at least
`η c' ‖∇F(x_t)‖²`, `c' > 0`, then the run converges to any cluster point at which `F` is
continuous and the inequality holds (`tendsto_of_isLojasiewiczAt_of_le_sub`). Near `p` the
inequality and the concavity of `s ↦ s^(1−a)` bound each step by the drop of
`(F(x_t) − F(p))^(1−a)`; these bounds telescope, so the run cannot leave a small ball
around `p`.

Three facts put the run of WD1 under these theorems. Let `L` be real-analytic as a function
of the real and imaginary parts of the entries of its matrix.

1. The factor space. List the entries of `W = [U; V]` as a vector `x` of `ℂ^((m+n)k)`, a
   real inner product space with `‖x‖² = ‖U‖² + ‖V‖²` and `⟨x, y⟩ = ⟨U, U'⟩ + ⟨V, V'⟩`. The
   entries of `U Vᴴ` and the weight decay are polynomials in the real and imaginary parts
   of the coordinates, so `F` is analytic on it, and by WD3 its gradient is the vector that
   lists `D = M W`. So a gradient step on this space is the step of WD1, and
   `‖∇F‖ = ‖M W‖`.
2. Sufficient decrease. Suppose that at the matrices with `‖X‖, ‖Y‖ ≤ ρ/2` the loss lies
   below its quadratic model with a constant `s ≥ 0`:

   ```
   L(Y) ≤ L(X) + ⟨G(X), Y − X⟩ + (s/2) ‖Y − X‖².
   ```

   With `G = G(U Vᴴ)` frozen at the current point and both `W`, `W'` in the ball of WD6,

   ```
   F(W') − F(W) ≤ ½ (⟨W', M W'⟩ − ⟨W, M W⟩) + (s/2) ‖U' V'ᴴ − U Vᴴ‖².
   ```

   For the step `W' = W − η D` the first term is `−η ‖D‖² + (η²/2) ⟨D, M D⟩` with
   `⟨D, M D⟩ ≤ (λ + ‖G‖) ‖D‖²`, and `U' V'ᴴ − U Vᴴ = −η (D_U V'ᴴ + U D_Vᴴ)` has squared
   norm at most `2 η² ρ ‖D‖²` when `‖U‖², ‖V'‖² ≤ ρ`. In the ball of WD6, where
   `‖G‖ ≤ γ`, the condition `η (λ + γ + 2 s ρ) ≤ 1` gives `F(W') + (η/2) ‖D‖² ≤ F(W)`.
3. A cluster point. The run stays in the ball `‖W‖² ≤ ρ` (WD6), and the ball is compact.

Result. Let `L` be real-analytic in the entries and satisfy the first-order inequality with
a continuous `G`. Let λ > 0 and ρ be such that at the matrices with `‖X‖, ‖Y‖ ≤ ρ/2` the
bounds `‖G(X)‖ ≤ γ` and `⟨G(X), X⟩ ≥ −b` of WD6 hold with `4b ≤ λ ρ`, and the quadratic
model holds with `s`. Take a step η > 0 with

```
2η (λ² + γ²) ≤ λ    and    η (λ + γ + 2 s ρ) ≤ 1.
```

Then the run from every start in the ball `‖W_0‖² ≤ ρ` converges, whatever its width and
rank, and its limit is stationary, `M W = 0`. From a start of full row rank the limit is a
decayed minimum (WD5 with WD6). No hypothesis on convergence remains.

The sign of λ enters this proof once, through WD6 in fact 3. Facts 1 and 2 hold for every
real λ. So for every real λ, with `‖G(X)‖ ≤ γ` and the quadratic model for
`‖X‖, ‖Y‖ ≤ ρ/2` and a step `η (λ + γ + 2 s ρ) ≤ 1`: a run that stays in the ball
`‖W_t‖² ≤ ρ` converges, and its limit is stationary. If moreover `η (|λ| + γ) < 1` and the
start has full row rank, the limit is a decayed minimum for that λ (WD5, whose proof does
not use the sign of λ either, with the step matrices of WD6). So where no decayed minimum
exists, such a run leaves the ball. XE17 applies this at λ = 0.

The squared loss on Ω with `‖K‖ ≤ κ` has `γ = ρ/2 + κ`, `b = κ²/4` and `s = 1`: it exceeds
its linear model by `½ ‖P_Ω(Y − X)‖² ≤ ½ ‖Y − X‖²`. So for it the conditions are
`κ² ≤ λ ρ` and

```
2η (λ² + γ²) ≤ λ    and    η (λ + γ + 2ρ) ≤ 1,    γ = ρ/2 + κ.
```

Example: the `1 × 1` kernel `1` seen on its cell, λ = 1, η = 1/10, `U_0 = [1 0]`,
`V_0 = [0 1]`. The bounds hold with κ = 1, ρ = 2: `2η (λ² + γ²) = 1` and
`η (λ + γ + 2ρ) = 7/10`. So this run converges to a decayed minimum. The width-one start
`U_0 = V_0 = 1` lies in the same ball and has no full row rank; its run converges to a
stationary point.

Corollary: trained fits exist. For every λ > 0 the start `U_0 = [1 0]`, `V_0 = [0 1]` of
width `m + n` has full row rank, and the result applies with `κ = ‖K‖² + 1`, a large ρ and a
small η. So for every λ > 0 the squared loss on Ω has a decayed minimum of width `m + n`,
and the statements of WD7 are about a nonempty class.

Corollary: training does not return a flat kernel. Let `K` be square with equal singular
values and no zero entry, and let Ω be proper. With the δ of WD7, statement 2, for every
λ > 0 every run under the bounds above from a start of full row rank converges, to factors
with `‖U Vᴴ‖_S1 ≤ ‖K‖_S1 − δ`. This is Theorem A for the run itself: nothing is assumed about
its limit.

The Łojasiewicz inequality cannot be replaced here by a nondegeneracy argument. For `K = 1`
with every cell seen and λ < 1 the partial fits `(1 − λ) P`, `P` a projection of a fixed
rank, form a connected family of critical points along which the gradient of the loss
varies: the critical points are not isolated.

### WD9. Small steps for every convex analytic loss

WD8 asks for three constants in a ball. With no constant at all, small steps still
converge.

Let `L` be real-analytic in the entries and satisfy the first-order inequality with a
continuous `G`, and let it be bounded below, `L ≥ ℓ_0`. Let λ > 0 and fix a level `c`. On
the factor space of WD8 the sublevel set `{F ≤ c}` is compact, because `F(W) ≤ c` gives
`λ ‖W‖²/2 ≤ c − ℓ_0`. For an analytic function with a compact sublevel set the theorems of
`Transformer.Landscape.SaddleAvoidance` give a step `η_0 > 0` such that for every
`0 < η < η_0`

- gradient descent maps `{F ≤ c}` into itself: near a compact set the gradient and the
  Hessian are bounded, so a small step moves a point a little and does not raise `F`;
- every run from `{F ≤ c}` converges to a critical point in it: the run is that of a
  function with a Lipschitz gradient that agrees with `F` near `{F ≤ c}`, and the
  Łojasiewicz inequality holds at every point;
- the starts in `{F ≤ c}` whose run converges to a strict saddle form a null set.

`G` is bounded on the products `U Vᴴ` of the points of `{F ≤ c}`, so after lowering `η_0`
every step matrix `1 − η M` there is positive definite (WD6).

Result. For such a loss, λ > 0 and a level `c` there is `η_0 > 0` such that for every
`0 < η < η_0` the run from every start with `F(U_0, V_0) ≤ c` converges, whatever its width
and rank, to a stationary point; and from a start of full row rank it converges to a decayed
minimum (WD5). The threshold `η_0` comes from compactness and is not explicit. WD8 gives
explicit steps when its constants are known.

Corollary: decayed minima exist. Every such loss has, for every λ > 0, a decayed minimum of
width `m + n`: start at `U_0 = [1 0]`, `V_0 = [0 1]`, take `c = F(U_0, V_0)` and a small
step. So `L(X) + λ ‖X‖_S1` attains its minimum over all matrices. WD10 lowers the width to
`min(m, n)`.

Example: the exponential loss `L(X) = Σ_ij exp(Re X_ij)` with `G(X)_ij = exp(Re X_ij)`. It
is analytic and positive, and the first-order inequality is
`exp(x) + exp(x)(y − x) ≤ exp(y)` in every entry. The constants of WD8 are not computed for
it. By the result, small steps converge for it, from full row rank to a decayed minimum, and
decayed minima exist for every λ > 0.

### WD10. Narrow factors: width min(m, n)

Full row rank of the start needs width `k ≥ m + n`. For almost every start the width
`k ≥ min(m, n)` suffices, and below it the claim fails.

Let `L` be real-analytic in the entries and satisfy the first-order inequality with a
continuous `G`. On the factor space of WD8 a strict saddle of `F` is a critical point at
which the Hessian of `F` has a negative eigenvalue. Let `W = [U; V]` be a critical point,
`M W = 0`.

1. Dependent columns. Let `W z = 0` for a vector `z ≠ 0` of the width `k`: `U z = 0` and
   `V z = 0`. For `a ∈ ℂ^m`, `b ∈ ℂ^n` put `S = [a; b]`, and let `M_t` be the matrix `M` at
   `W + t S zᴴ`. The gradient of `F` there is `M_t (W + t S zᴴ)`, and
   `⟨S zᴴ, M_t W⟩ = ⟨S, M_t W z⟩ = 0`, so

   ```
   ⟨S zᴴ, ∇F(W + t S zᴴ)⟩ = t ‖z‖² ⟨S, M_t S⟩    exactly, for every t.
   ```

   So the Hessian of `F` at `W` along `S zᴴ` is `‖z‖² ⟨S, M S⟩`: a negative direction of
   `M` is a negative direction of the Hessian. A critical point with dependent columns is
   therefore a strict saddle, or `M` is positive semidefinite and the point is a decayed
   minimum (WD2).

2. Independent columns at width `k ≥ min(m, n)`, with λ > 0. `M W = 0` reads
   `G V = −λ U` and `Gᴴ U = −λ V`. So `U z = 0` gives `V z = 0`, and the columns of `U` are
   independent, as are those of `V`. If `k ≥ m`, then `U` is square and invertible, and
   `G Gᴴ U = −λ G V = λ² U` gives `G Gᴴ = λ²`: `‖Gᴴ a‖ = λ ‖a‖`. Then
   `|⟨G, a bᴴ⟩| = |⟨Gᴴ a, b⟩| ≤ λ ‖a‖ ‖b‖`, and by WD1

   ```
   ⟨S, M S⟩ = λ (‖a‖² + ‖b‖²) + 2 ⟨G, a bᴴ⟩ ≥ λ (‖a‖ − ‖b‖)² ≥ 0.
   ```

   The case `k ≥ n` is the same with `V` and `Gᴴ G = λ²`. So such a point is a decayed
   minimum, whatever its Hessian.

Result 1: no spurious minimum. A critical point of `F` that is not a strict saddle is a
decayed minimum when its columns are dependent, and, for λ > 0, at every width
`k ≥ min(m, n)`. A local minimum of `F` among the pairs of its width is a critical point and
not a strict saddle. So a local minimum with dependent columns, and at width
`k ≥ min(m, n)` every local minimum, is a decayed minimum: it is least among the
factorizations of every width.

Result 2: almost every start. Let moreover `L` be bounded below and λ > 0, and fix a level
`c`. By the third item of WD9, for every `0 < η < η_0` the starts in `{F ≤ c}` whose run
converges to a strict saddle form a null set. So for almost every start in `{F ≤ c}` the
run converges to a stationary point that

- is a decayed minimum if its columns are dependent, whatever the width;
- is a decayed minimum at every width `k ≥ min(m, n)`.

"Almost every" is for the Lebesgue measure of the factor space. No rank condition on the
start appears, and some exceptional set is needed: the start `W_0 = 0` stays at `0`, a
strict saddle whenever `‖G(0)‖_op > λ`.

Corollary: every such loss has, for every λ > 0, a decayed minimum of every width
`k ≥ min(m, n)`, as the open set `{F < F(0) + 1}` is not null.

Theorem A for runs of width `p`. Let `K` be a `p × p` kernel with equal singular values and
no zero entry, and let Ω be proper. With the δ of WD7, statement 2, for every λ > 0, every
level `c` and every width `k ≥ p`, small steps on the squared loss on Ω from almost every
start in `{F ≤ c}` converge to factors with `‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1`. WD8 has this for
every start of full row rank, at width `2p`.

The width cannot be lowered. Let `K = 1` of size `2 × 2` with every cell seen and
`0 ≤ λ < 1`. For factors `U`, `V` of width one put `q = ⟨U, V⟩`. Then
`‖U Vᴴ‖² = ‖U‖² ‖V‖²`, `⟨U Vᴴ, 1⟩ = q` and `|q| ≤ ‖U‖ ‖V‖ ≤ ½ (‖U‖² + ‖V‖²)`, so

```
F(U, V) = ½ (‖U‖² ‖V‖² − 2 q + 2) + ½ λ (‖U‖² + ‖V‖²)
        ≥ ½ |q|² − (1 − λ) |q| + 1 ≥ 1 − ½ (1 − λ)².
```

The factors `U = V = √(1 − λ) · 1` of width two have `F = λ² + 2 λ (1 − λ) = 2 λ − λ²`,
which is smaller by `½ (1 − λ)²`. So no factorization of width one is a decayed minimum.
The runs of width one converge all the same (WD9), to stationary points such as the partial
fits `(1 − λ) P` of WD8, and none of these limits is a decayed minimum. At width two a
decayed minimum exists and almost every run with a small step reaches one.

### WD11. Scope

Proved, for the pure factorization `X = U Vᴴ` trained by gradient descent with weight decay
on a convex loss with continuous gradient:

- the update is the gradient step of the weight-decayed loss (WD3);
- a run from stacked factors of full row rank with admissible steps that converges ends at
  a global minimum of the weight-decayed loss over the factorizations of every width, and
  its product minimizes `L + λ ‖·‖_S1` (WD5);
- for a real-analytic loss with the bounds of WD6 and WD8 in a ball and an explicit step, a
  run from the ball converges to a stationary point, for every width and rank; from stacked
  factors of full row rank its limit is such a minimum; no convergence is assumed (WD8);
- the squared loss on Ω meets these bounds with explicit constants, and for every λ > 0 it
  has such a minimum of width `m + n` (WD8);
- for every loss that is real-analytic in the entries and bounded below, small steps
  converge from a sublevel set of the weight-decayed loss, for every width and rank, to a
  stationary point, and from stacked factors of full row rank to such a minimum; such a
  minimum of width `m + n` exists for every λ > 0 (WD9);
- for such a loss a critical point of the weight-decayed loss that is not a strict saddle is
  such a minimum when its columns are dependent, and for λ > 0 at every width
  `k ≥ min(m, n)`; at these widths no local minimum is spurious, small steps from almost
  every start of a sublevel set converge to such a minimum, and such a minimum of width
  `min(m, n)` exists; for the `2 × 2` identity no factorization of width one is such a
  minimum (WD10);
- for a square kernel with equal singular values and no zero entry on a proper Ω, every
  such run converges at least a fixed δ below `‖K‖_S1`, for every λ > 0 (WD8), and so does
  almost every run of width `p` with a small step (WD10); the same holds for a flat kernel
  of any shape when some nonzero cell is unseen, at width `min(m, n)` (R4 of *Flat kernels
  of any shape*);
- for the squared loss on Ω these limits compare with every exact fit as M does, their
  limits as λ → 0⁺ are optimal fits, every sequence of them of one width has a subsequence
  that converges to an optimal fit, flat kernels are not returned, and kernels that M
  recovers are approached; the sign change of both factors keeps a trained fit and an
  optimal fit, so the factors of trained fits need not converge and an optimal fit with a
  nonzero factor is not unique (WD7).
- for the linear loss `−Re tr X`, which is convex and real-analytic and not bounded below,
  with a weight decay λ < 1: the run from every start with `U_0 + V_0 ≠ 0` has factors whose
  norm tends to infinity, at every step size, so it has no limit, and no factorization is a
  decayed minimum (WD12);
- a run does not raise the rank of its stack, so a start whose stack has a rank below that
  of every trained product reaches no decayed minimum, at any width; for the `2 × 2`
  identity these are the starts whose stack has rank at most one, and under the bounds of
  WD8 their runs converge to stationary points that are not decayed minima; the run from a
  start with zero columns is the narrower run; a stationary point with λ ≠ 0 is balanced,
  `Uᴴ U = Vᴴ V`, its weight decay is the nuclear norm of its product, and for λ > 0 it is a
  decayed minimum exactly when its product is a trained product; for the `2 × 2` identity
  the run from `V_0 = −U_0` reaches no decayed minimum, though its stack can have rank two
  (WD14);
- for the `1 × 1` identity under the bounds of WD8, at every width, width one included, the
  run from a start in the ball converges; for `0 < λ < 1` its limit is a decayed minimum
  exactly when `U_0 + V_0 ≠ 0`, and for `λ ≥ 1` it is one for every start (WD15);
- for the identity of any size and λ > 0, a pair is stationary exactly when its factors are
  equal and `U Uᴴ U = (1 − λ) U`, so the run of WD8 converges, at every width, to a pair
  `(U, U)` whose product is `1 − λ` times an orthogonal projection; for `0 ≤ λ ≤ 1` the only
  trained product is `(1 − λ) · 1`, a stationary point is a decayed minimum exactly when it
  has that product, and for λ < 1 no factorization of a width below `n` is one; for λ ≥ 1
  the run converges to the origin, and for `0 < λ < 1` it does so exactly from the starts
  with `U_0 + V_0 = 0`; from a start `V_0 = U_0` the product of the limit is `1 − λ` times
  the orthogonal projection onto the column space of `U_0`, so the limit is a decayed
  minimum exactly when `U_0` has full row rank (WD16).

Assumed or not covered:

- Losses outside WD8 and WD9. For a loss that is not bounded below the run can diverge with
  the other hypotheses of WD5 in force: the linear loss `−Re tr X` with λ < 1 (WD12). For a
  loss that is bounded below and not real-analytic, convergence stays the hypothesis of WD5,
  and so does the condition `η (|λ| + ‖G_t‖_S2) < 1` along the run: no proof and no
  counterexample is given.
- Explicit steps: the constants of WD8 are verified here for the squared loss, and in
  *Training with cross-entropy* (added) for the cross-entropy. For another loss WD9 gives a
  step threshold that is not explicit.
- Width: a statement for every start needs full row rank of `[U_0; V_0]`, so `k ≥ m + n`
  (WD5). At the widths `min(m, n) ≤ k < m + n` the limit is a decayed minimum for almost
  every start (WD10). The exceptional null set is described in part: it contains every start
  whose stack has a rank below that of every trained product, such as the starts of rank at
  most one for the `2 × 2` identity, and for that kernel also the starts `V_0 = −U_0`, whose
  stacks have rank up to two, so the rank of the stack does not describe it (WD14). An exact
  description is given only for the `1 × 1` identity: there the exceptional starts are those
  with `U_0 + V_0 = 0` when λ < 1, there are none when λ ≥ 1, and the step is the explicit
  one of WD8 (WD15). For a larger kernel the step of that statement is the non-explicit one
  of WD9, not the explicit one of WD8. Below `min(m, n)` a decayed minimum of that width
  need not exist (WD10); the run still converges to a stationary point. Such a limit is
  balanced, `Uᴴ U = Vᴴ V`, its weight decay is the nuclear norm of its product, and it is a
  decayed minimum exactly when its product is a trained product; these limits are also
  limits of runs of every larger width, from starts with zero columns (WD14). Which
  stationary products a narrow run reaches is not determined in general; for the identity of
  any size they are `1 − λ` times orthogonal projections of rank at most `k`, and which
  projection is determined only for the starts `V_0 = U_0` (WD16).
- In WD7 the limits as λ → 0⁺ are taken along sequences. Statement 1 gives a convergent
  subsequence of the factors of trained fits of one width, and every such limit is an optimal
  fit; the factors themselves need not converge, and an optimal fit with a nonzero factor is
  not unique, by the sign change of WD7.
- The model and the optimizer: for the real network (norms, SiLU, attention) the modeling
  step of Lemma 1 remains an assumption, as the note says. For AdamW it does not hold in the
  form of WD5: a limit of a full-batch run satisfies `wd · θ = − g / (|g| + ε)` in every
  coordinate, and for a `1 × 1` factorization no run converges to a minimizer of the
  weight-decayed loss (WD13). What a run of AdamW does reach is not treated.

### WD12. Without a lower bound the run can diverge

WD5 assumes that the run converges, and WD8 and WD9 prove it for losses that are bounded
below. A loss that is not bounded below can have a run that diverges, with the two other
hypotheses of WD5 in force.

Take the linear loss `L(X) = −Re tr X` of a square matrix. It satisfies the first-order
inequality with the constant gradient `G = −1`, with equality. It is real-analytic, and it is
not bounded below: `L(s · 1) = −s m`. With a weight decay λ a step of WD1 is

```
U ← (1 − ηλ) U + η V,    V ← (1 − ηλ) V + η U,
```

so the sum and the difference of the factors are multiplied by constants:

```
U_t + V_t = (1 + η (1 − λ))^t (U_0 + V_0),    U_t − V_t = (1 − η (1 + λ))^t (U_0 − V_0).
```

Let `λ < 1` and `η > 0`. The first constant exceeds one. So from every start with
`U_0 + V_0 ≠ 0`, of any width,

```
‖U_t‖² + ‖V_t‖² ≥ ½ ‖U_t + V_t‖² = ½ (1 + η (1 − λ))^{2t} ‖U_0 + V_0‖² → ∞,
```

and the run has no limit. The step size plays no role. The weight-decayed loss is not bounded
below either: at `(s U, s U)` it is `−(1 − λ) s² ‖U‖²`. So no factorization of any width is a
decayed minimum.

Example: the `1 × 1` loss `−Re X` with `λ = 1/2` and `η = 1/4`, from `U_0 = [1 0]`,
`V_0 = [0 1]`. The start has full row rank, and the step matrix `1 − η M`, the same at every
step, has the eigenvalues `1 − η (λ ∓ 1) = 9/8` and `5/8`: it is positive definite. These are
the two other hypotheses of WD5, and the run has no limit. At `λ = 1` the same run converges
(the example of WD5); there the weight-decayed loss is bounded below.

So convergence does not follow from the other hypotheses of WD5, even for a convex
real-analytic loss with a constant gradient. What fails is the lower bound that WD9 asks for:
with it the sublevel sets of the weight-decayed loss are compact.

### WD13. AdamW is not gradient descent with weight decay: where its runs can end

WD1–WD12 are about gradient descent with weight decay. The models of the note are trained
with AdamW. In the code of the paper's experiments
(`experiments/grok_division/PROTOCOL.md`, *The optimizer*) update `t` of a parameter `p` at
the gradient `g` and the learning rate `lr` is

```
p ← p (1 − lr · wd),    m ← β₁ m + (1 − β₁) g,    v ← β₂ v + (1 − β₂) g²,
p ← p − lr / (1 − β₁ᵗ) · m / (√v / √(1 − β₂ᵗ) + ε).
```

Let every update see the gradient of one continuous field `g`, the full batch, with
`0 ≤ β₁, β₂ < 1`, `ε > 0` and a learning rate that is a nonzero constant after its warmup.

1. The limit. If the parameters of a run converge to θ, then in every coordinate

   ```
   wd · θ = − g(θ) / (|g(θ)| + ε).
   ```

   The gradients seen converge to `g(θ)`. A moving average of a convergent sequence converges
   to its limit, so `m → g(θ)` and `v → g(θ)²`, and the bias corrections tend to `1`. So the
   right side of the update has a limit, and it is θ:
   `θ = (1 − lr · wd) θ − lr · g(θ) / (|g(θ)| + ε)`.

2. Consequences. With `wd > 0` every coordinate of a limit has `|θ| < 1 / wd`. Without weight
   decay a limit is a critical point, `g(θ) = 0`. A limit is stationary for the loss with the
   penalty `(wd/2) ‖θ‖²`, that is `g(θ) + wd · θ = 0`, in a coordinate only if `g(θ) = 0` or
   `|g(θ)| = 1 − ε` there.

3. The pure factorization. Take real factors `u`, `v` of a `1 × 1` product, the squared loss
   `½ (u v − 3)²` and λ = 1. The weight-decayed loss of WD2 is

   ```
   ½ (u v − 3)² + ½ (u² + v²) = 5/2 + ½ (u v − 2)² + ½ (u − v)²,
   ```

   so its minimizers are `u = v = ±√2`, with the product `2`. At them the gradient of the
   loss is `∓√2` in both coordinates, and `√2` is neither `0` nor `1 − ε`. So with the
   source's constants and weight decay `1` no run of AdamW converges to a minimizer of the
   weight-decayed loss, from any start. The same holds for one parameter and the loss
   `½ (p − 3)²`: the minimizer `3/2` of the penalized loss is not a limit.

So the modeling step of Lemma 1, "training with weight decay then behaves like
minimum-nuclear-norm interpolation", does not pass to AdamW through its limits. Statement 1
takes the place of the stationarity `M W = 0` of WD3. Where `g(θ) ≠ 0` and as ε → 0 it reads
`wd · θ = − sign g(θ)`: the coordinate is at the wall of the box `‖θ‖_∞ ≤ 1 / wd` to which
the gradient pushes it, as in the first-order condition of minimizing the loss in that box
(Xie and Li, 2024, *Implicit bias of AdamW: ℓ∞-norm constrained optimization*, study the
convergence to such points; not formalized here).

Not covered: that a run of AdamW converges; minibatches, where the field changes with the
update; which of the points of statement 1 a run reaches, and what replaces M for it; the
real network (normalizations, SiLU, attention), to which statement 1 applies as to every
model with a continuous gradient, and for which nothing about M is proved.

### WD14. Exceptional starts: a run does not raise the rank of its stack

WD10 leaves an exceptional null set of starts at the widths `min(m, n) ≤ k < m + n`, and
names only the start `W_0 = 0`. This item describes a part of that set, at every width.
Statements 1, 2 and 4 hold for every gradient field, step and weight decay.

A step multiplies the stack `W = [U; V]` on the left: `W_{t+1} = (1 − η M_t) W_t` (WD1).

1. The kernel is kept. If `W_0 z = 0` for a vector `z` of the width `k`, then `W_t z = 0` at
   every step, and `W z = 0` for a limit `W` of the run.

2. The rank does not grow. `rank W_t ≤ rank W_0` at every step, and a limit `(U, V)` of the
   run has `rank (U Vᴴ) ≤ rank U ≤ rank W_0`. By 1 the kernel of `W_0` lies in the kernel of
   `W_t` and in that of `U`, and the rank is the width minus the dimension of the kernel.

3. Exceptional starts. Let λ ≥ 0. The product of a decayed minimum is a trained product, a
   minimizer of `L + λ ‖·‖_S1` (WD2). So if every trained product has a rank above
   `rank [U_0; V_0]`, no limit of the run from `(U_0, V_0)` is a decayed minimum. With `r*`
   the least rank of a trained product, the starts with `rank [U_0; V_0] < r*` are
   exceptional, at every width.

4. A narrow run inside a wide one. For factors `(u_0, v_0)` of width `k₁`, the run from
   `([u_0 0], [v_0 0])` of width `k₁ + k₂` is the run from `(u_0, v_0)` with zero columns
   appended, at every step: zero columns change neither the product nor the weight decay. If
   the narrow run converges to `(u, v)`, the wide one converges to `([u 0], [v 0])`, and that
   is a decayed minimum exactly when `(u, v)` is. The stack of a padded pair has rank at most
   `k₁`. So the limits of the runs of a width below `min(m, n)`, which WD10 does not
   characterize, are limits of runs of every larger width.

5. The `2 × 2` identity. Let `K = 1` of size `2 × 2` with every cell seen and `0 ≤ λ < 1`, as
   in WD10. A trained product has rank two: a matrix `X` of rank at most one has
   `‖X‖_S1 = ‖X‖_S2`, call it `f`, and `Re tr X ≤ ‖X‖_S1`, so

   ```
   ½ ‖X − 1‖² + λ ‖X‖_S1 ≥ ½ f² − (1 − λ) f + 1 ≥ 1 − ½ (1 − λ)²,
   ```

   above the value `2 λ − λ² = 1 − (1 − λ)²` of the product `(1 − λ) · 1` (WD10). By 3 a run
   from a start whose stack has rank at most one does not converge to a decayed minimum, at
   any width and with any step. For λ > 0 and the explicit bounds of WD8 such a run
   converges: its limit is stationary and is not a decayed minimum.

6. Stationary points are balanced. Let `W = [U; V]` be stationary, `M W = 0`, with λ ≠ 0: the
   limits of the runs of WD8 and WD9 are such points, at every width. Then `G V = −λ U` and
   `Gᴴ U = −λ V`, and both give `Uᴴ G V`: `−λ Uᴴ U = −λ Vᴴ V`, so

   ```
   Uᴴ U = Vᴴ V.
   ```

   So `‖U‖ = ‖V‖`, and `(U Vᴴ)ᴴ (U Vᴴ) = V Uᴴ U Vᴴ = (V Vᴴ)²`: the positive semidefinite
   root of the Gram matrix of the product is `V Vᴴ`, and

   ```
   ‖U Vᴴ‖_S1 = tr V Vᴴ = ‖V‖² = ½ (‖U‖² + ‖V‖²).
   ```

   The equality of Lemma 1 holds at every stationary point, not only at the decayed minima
   of WD2. The product has the rank of each factor. For λ > 0 the weight-decayed loss of a
   stationary point is `L(X) + λ ‖X‖_S1` at its product `X`, so a stationary point, and a
   limit of a run, is a decayed minimum exactly when its product is a trained product.

7. A start of full rank can be exceptional. Let the gradient field map Hermitian matrices to
   Hermitian matrices, as that of the squared loss on all cells of the identity does.
   A step from a pair `(U, −U)` gives a pair `(U', −U')`: the product `−U Uᴴ` is Hermitian,
   so `G` is, and the updates of the two factors differ by the sign. So the run from
   `V_0 = −U_0` has `V_t = −U_t` at every step, and a limit of it is a pair `(U, −U)`. The
   stack `[U_0; −U_0]` has the rank of `U_0`.

   For the `2 × 2` identity with every cell seen and `0 ≤ λ < 1` the product `−U Uᴴ` of such
   a pair is not a trained product: `Re tr (−U Uᴴ) = −‖U‖² ≤ 0`, so

   ```
   ½ ‖−U Uᴴ − 1‖² + λ ‖U Uᴴ‖_S1 ≥ 1 > 2 λ − λ²,
   ```

   the value of `(1 − λ) · 1`. So the run from `V_0 = −U_0` does not converge to a decayed
   minimum, at any width and with any step, and for λ > 0 under the bounds of WD8 it
   converges to a stationary point that is not one. With `U_0 = 1` of width two the stack
   `[1; −1]` has rank two, the rank of the trained products (5): the exceptional set of WD10
   is larger than the set of 3, and the rank of the stack does not describe it.

So at width two, which is `min(m, n)` and at which almost every run with a small step ends at
a decayed minimum (WD10), the exceptional set contains every start whose stack has rank at
most one: the start `0`, the starts with a zero column, whose runs are the runs of width one
(4), and all starts `[U_0; V_0] = S wᴴ`. These starts satisfy polynomial equations, the
vanishing of the `2 × 2` minors of the stack, so they form a null set; that is not
formalized. The exceptional set also contains the starts `V_0 = −U_0` (7), a proper linear
subspace of the starts, whose stacks have rank up to two.

Example: with λ = ½, κ = 2, ρ = 8 and η = 1/200, the run of width two from
`U_0 = V_0 = [[1, 0], [1, 0]]` converges to a stationary point that is not a decayed
minimum, and it is the run of width one from `u_0 = v_0 = (1, 1)` with a zero column. With
the same numbers the run of width two from `U_0 = 1`, `V_0 = −1`, whose stack has rank two,
converges to a stationary point that is not a decayed minimum.

Not covered. An exact description of the exceptional set of WD10 for a kernel larger than
`1 × 1`: it contains the set of 3 and the starts of 7, and the rank of the stack does not
describe it; an explicit step for the statement "almost every start" of WD10 for such a
kernel; the limits of the runs below `min(m, n)` beyond 4 and 6, that is which stationary
products such a run reaches. For the `1 × 1` identity the exceptional starts are exactly
those with `U_0 + V_0 = 0`, with the explicit step of WD8 (WD15). For the identity of any
size the limit of every run of WD8 is a pair `(U, U)` whose product is `1 − λ` times an
orthogonal projection, and the run from `V_0 = −U_0` of 7 converges to the origin (WD16);
which projection a run reaches is determined only for the starts `V_0 = U_0`.

### WD15. The 1 × 1 identity: the exceptional starts, exactly

WD10 gives the decayed minimum for almost every start at the widths
`min(m, n) ≤ k < m + n`, with a step that is not explicit, and by WD14 the rank of the stack
does not describe the exceptional starts. For the smallest kernel the exceptional starts and
the step are explicit.

Let `K = 1` of size `1 × 1` with its cell seen, the squared loss, and λ > 0. The factors `U`
and `V` are rows of any width; width one is `min(m, n)`, below `m + n = 2`. The product is
the number `x = U Vᴴ`, the gradient is `G = x − 1`, and for λ < 1 the trained product is
`1 − λ`.

1. The sum of the factors. Put `P = U + V` and `Q = U − V`. Then
   `Re x = (‖P‖² − ‖Q‖²) / 4` and `Im x = −½ ⟨P, i Q⟩`, and a step gives

   ```
   P' = a P + η Im x · i Q,        a = 1 − η (λ − 1 + Re x).
   ```

   With `p = ‖P‖²`, `q = ‖Q‖²` and `c = ⟨P, i Q⟩`, where `c² ≤ p q`, for `a ≥ 0` and `η ≥ 0`

   ```
   ‖P'‖² = a² p − a η c² + ¼ η² c² q ≥ a (a − η q) p.
   ```

   Here `a − η q = 1 − η (λ − 1 + (p + 3 q) / 4)`, and
   `(p + 3 q) / 4 ≤ (3/2) (‖U‖² + ‖V‖²)`.

2. In the ball. Let `η > 0` and `η (λ + ρ/2 + κ + 2 ρ) ≤ 1` with `κ ≥ 0`, the second bound
   of WD8, and let `‖U‖² + ‖V‖² ≤ ρ`. Then `a ≥ a − η q ≥ η`, so `‖P'‖² ≥ η² ‖P‖²`: the sum
   of the factors stays nonzero. Where also `‖U‖² + ‖V‖² ≤ ⅔ (1 − λ)`, `a ≥ a − η q ≥ 1`, so
   `‖P'‖ ≥ ‖P‖`: near the origin the sum does not decrease.

3. The limit is not the origin. Under the bounds of WD8 with `κ ≥ 1` the run from a start in
   the ball stays in the ball and converges to a stationary point (WD6, WD8). Let λ < 1 and
   `U_0 + V_0 ≠ 0`. If the limit were `(0, 0)`, the norms would tend to `0`, so by 2 from
   some step on `‖P_t‖` would not decrease, and it is positive at every step: it would not
   tend to `0`.

4. The stationary points. At a stationary point `G V = −λ U` and `Ḡ U = −λ V`, so
   `|G|² ‖V‖² = λ² ‖U‖²` and `|G|² ‖U‖² = λ² ‖V‖²`. If `(U, V) ≠ (0, 0)`, then `|G| = λ`, the
   lifted matrix `M` is positive semidefinite (WD1), and the point is a decayed minimum
   (WD2). So every stationary point other than the origin is a decayed minimum.

5. The exceptional starts, for λ < 1. If `U_0 + V_0 = 0`, the run has `V_t = −U_t` at every
   step and its limit is a pair `(U, −U)` (WD14, statement 7). Its product `−‖U‖²` is not
   positive, so its weight-decayed loss is at least `½`, above the value `λ − ½ λ²` of the
   factorization `√(1 − λ) · √(1 − λ)` of the trained product: it is not a decayed minimum.

6. Weight decay λ ≥ 1. At the origin `G = −1` and `|G| = 1 ≤ λ`, so the origin is a decayed
   minimum (WD1, WD2), and with 4 every stationary point is one.

Theorem. Under the bounds of WD8 with `κ ≥ 1`, at every width, the run from a start in the
ball `‖U_0‖² + ‖V_0‖² ≤ ρ` converges. For `0 < λ < 1` its limit is a decayed minimum if and
only if `U_0 + V_0 ≠ 0`; for `λ ≥ 1` it is one for every start.

Every start lies in such a ball, with a step that depends on ρ. So for this kernel the
exceptional starts are exactly those of the subspace `U_0 + V_0 = 0` when λ < 1, and there
are none when λ ≥ 1; the step is the explicit one of WD8; and width one is included: for
λ < 1 the narrow run reaches the trained product from every start outside that subspace.

Example: κ = 1, ρ = 4 and η = 1/100, at width one. With λ = ½ the run from `u_0 = v_0 = 1`
converges to a decayed minimum, and the run from `u_0 = 1`, `v_0 = −1` converges to a point
that is not one. With λ = 1 the run from `u_0 = 1`, `v_0 = −1` converges to a decayed
minimum.

Not covered. Kernels larger than `1 × 1`. There 4 fails: a stationary point other than the
origin need not be a decayed minimum (WD14, statements 5 and 7), so it does not suffice that
the run avoids the origin, and the exceptional starts are not described. WD16 gives the
stationary points, the form of the limits and the starts attracted to the origin for the
identity of any size.

### WD16. The identity of any size: stationary points, the trained product, the limits

WD10 and WD14 treat the `2 × 2` identity and WD15 the `1 × 1` identity. For the identity of
any size the stationary points, the trained product and the form of the limit of every run
are explicit, and so are the starts whose run ends at the origin; the other exceptional
starts are not.

Let `K = 1` of size `n × n` with every cell seen, the squared loss, and factors `U`, `V` of
any width `k`. The gradient at the product `X = U Vᴴ` is `G = X − 1`.

1. Stationary points have equal factors. Let λ > 0. A pair `(U, V)` is stationary if and
   only if

   ```
   V = U    and    U Uᴴ U = (1 − λ) U.
   ```

   The equations `G V = −λ U` and `Gᴴ U = −λ V` read `V = U Vᴴ V + λ U` and
   `U = V Uᴴ U + λ V`. A stationary point is balanced, `Uᴴ U = Vᴴ V` (WD14, statement 6);
   call this matrix `B`. Then `V = U (B + λ)` and `U = V (B + λ)`, so `D = U − V` has
   `D (B + 1 + λ) = 0`. Pairing with `D` gives `‖D Uᴴ‖² + (1 + λ) ‖D‖² = 0`, so `D = 0`. With
   `V = U` the first equation is `U Uᴴ U = (1 − λ) U`. Conversely, for such a pair
   `G U + λ U = U Uᴴ U − U + λ U = 0`, and `G` is Hermitian.

2. Their products. So the product `X = U Uᴴ` of a stationary point is positive semidefinite
   and `X² = (1 − λ) X`: for λ < 1 it is `1 − λ` times an orthogonal projection, of the rank
   of `U`. For λ ≥ 1 the only stationary point is the origin: `‖U Uᴴ‖² = (1 − λ) ‖U‖² ≤ 0`
   gives `U Uᴴ = 0` and `‖U‖² = Re tr (U Uᴴ) = 0`.

3. The trained product is unique. Let `0 ≤ λ ≤ 1`. By `Re tr X ≤ ‖X‖_S1`, for every `X`

   ```
   ½ ‖X − 1‖² + λ ‖X‖_S1 ≥ ½ ‖X − 1‖² + λ Re tr X = ½ ‖X − (1 − λ) 1‖² + n (λ − ½ λ²),
   ```

   and `n (λ − ½ λ²)` is at least the value at `(1 − λ) · 1`, by the factorization
   `√(1 − λ) 1 · √(1 − λ) 1` (Lemma 1). So the only minimizer of `L + λ ‖·‖_S1` is
   `(1 − λ) · 1`, the product of a decayed minimum of any width is `(1 − λ) · 1` (WD2), and
   for λ < 1 no factorization of a width `k < n` is a decayed minimum: a product of rank at
   most `k` is not a nonzero multiple of the identity. For `n = 2` these are the statement of
   WD10 on width one and statement 5 of WD14.

4. Which stationary points are decayed minima. Let `0 ≤ λ ≤ 1`. A stationary point is a
   decayed minimum if and only if its product is `(1 − λ) · 1`: one direction is 3; for the
   other, `G = −λ · 1`, the lifted matrix `M = λ [[1, −1], [−1, 1]]` is positive
   semidefinite, and WD2 applies. In general `[[λ, c], [c, λ]]` with a real `c` is positive
   semidefinite when `|c| ≤ λ`: its form at `[A; B]` is
   `½ (λ + c) ‖A + B‖² + ½ (λ − c) ‖A − B‖²`. With `c = −1` this shows that for λ ≥ 1 the
   origin is a decayed minimum.

5. The limits of the runs. A limit of a run with a step `η > 0` is stationary (WD5), so by 1
   it is a pair `(U, U)` with `U Uᴴ U = (1 − λ) U`, and for λ ≤ 1 it is a decayed minimum
   exactly when `U Uᴴ = (1 − λ) · 1` (4).

6. The origin. Put `P = U + V` and `Q = U − V`. A step gives

   ```
   P' = (1 + η (1 − λ)) P − η (U Vᴴ V + V Uᴴ U),
   4 (U Vᴴ V + V Uᴴ U) = P (Pᴴ P + Qᴴ Q) − Q (Pᴴ Q + Qᴴ P).
   ```

   In `4 ⟨P, U Vᴴ V + V Uᴴ U⟩` the two terms with `P` on the left are `‖Pᴴ P‖² ≤ ‖P‖⁴` and
   `⟨Qᴴ Q, Pᴴ P⟩ ≤ ‖P‖² ‖Q‖²`, and the two with `Q` on the left are subtracted and sum to
   `⟨Pᴴ Q, Qᴴ P⟩ + ‖Qᴴ P‖² ≥ 0`. So `4 ⟨P, U Vᴴ V + V Uᴴ U⟩ ≤ ‖P‖² (‖P‖² + ‖Q‖²)`, and with
   `a = 1 + η (1 − λ) ≥ 0`, `η ≥ 0` and `‖P‖² + ‖Q‖² = 2 (‖U‖² + ‖V‖²)`

   ```
   ‖P'‖² ≥ a (a − η (‖U‖² + ‖V‖²)) ‖P‖².
   ```

   In the ball of WD8 the factor is at least `η²`: the sum of the factors stays nonzero.
   Where `‖U‖² + ‖V‖² ≤ 1 − λ` the factor is at least `1`: near the origin the sum does not
   decrease. So, as in WD15, for `0 < λ < 1` a run of WD8 from `U_0 + V_0 ≠ 0` does not
   converge to the origin.

7. Equal factors. A step from a pair `(U, U)` gives the pair `(U', U')` with

   ```
   U' = (1 + η (1 − λ)) U − η U Uᴴ U = M U,        M = (1 + η (1 − λ)) − η U Uᴴ,
   ```

   so the run from `V_0 = U_0` has `V_t = U_t`, and `X_t = U_t U_tᴴ` follows `X' = M X M`.
   What commutes with `X_0` commutes with every `X_t`; so `X_0` does, and so does the
   product `X` of the limit `(U, U)`. Let `0 < λ < 1` and `E = 1 − X / (1 − λ)`. By 2 it is a
   Hermitian idempotent, and `E U = 0`. It commutes with every `X_t`, so
   `E U' = (1 + η (1 − λ)) E U − η (E U) (E U)ᴴ (E U)`: the matrices `E U_t` are the run from
   the pair `(E U_0, E U_0)`, whose norm is at most that of `(U_0, U_0)`. That run converges
   to `E U = 0`, so by 6 `E U_0 = 0`:

   ```
   X U_0 = (1 − λ) U_0.
   ```

   The product of the limit is `1 − λ` on the columns of `U_0`, so its column space contains
   that of `U_0`. The stack `[U_0; U_0]` has the rank of `U_0`, so `rank X ≤ rank U_0` (WD14,
   statement 2): the two column spaces are equal, and by 2 `X` is `1 − λ` times the
   orthogonal projection onto the column space of `U_0`. So the limit is a decayed minimum
   exactly when `U_0` has full row rank, `U_0 R = 1` for some `R`: then `X = (1 − λ) · 1`,
   and without full row rank `rank X < n`.

Theorem. Let λ > 0 and the bounds of WD8 hold with `‖1‖² ≤ κ²`. At every width the run from
a start in the ball converges to a pair `(U, U)` with `U Uᴴ U = (1 − λ) U`: the product of
the limit is `1 − λ` times an orthogonal projection, of a rank at most `rank [U_0; V_0]`
(WD14, statement 2). For λ < 1 the limit is a decayed minimum exactly when that projection
is the identity, and at a width `k < n` it is not one. For λ ≥ 1 the limit is the origin, a
decayed minimum. The run from `V_0 = −U_0` (WD14, statement 7) converges to the origin, for
every λ > 0: its limit is a pair `(U, −U)` and a pair `(U, U)`. For `0 < λ < 1` these are the
only such runs: the limit is the origin if and only if `U_0 + V_0 = 0` (6). For `0 < λ < 1`
and a start `V_0 = U_0`, the product of the limit is `1 − λ` times the orthogonal projection
onto the column space of `U_0`, and the limit is a decayed minimum if and only if `U_0` has
full row rank (7).

So for the identity the question of WD14, which stationary products a narrow run reaches,
has the answer: `1 − λ` times an orthogonal projection of rank at most `k`. At width one,
for `0 < λ < 1`, the rank is one exactly from the starts with `u_0 + v_0 ≠ 0`. The starts
attracted to the origin form the subspace `U_0 + V_0 = 0`, at every size and width: for the
`1 × 1` identity this is the exceptional set of WD15. Among the starts `V_0 = U_0` the
exceptional ones are exactly those with `rank U_0 < n`, at every width. At the width `k = n`
the pairs `(U_0, U_0)` with `U_0` invertible are starts that reach the decayed minimum with
the explicit step of WD8; their stack has rank `n`, half of the full row rank that WD8 asks
for.

Example: λ = 3/4, so `1 − λ = 1/4`, and `n = 2`. The pair `U = V = ½ · 1` is a decayed
minimum. The pair `U = V = ½ · diag(1, 0)` is stationary, its product `¼ · diag(1, 0)` is
not `¼ · 1`, and it is not a decayed minimum. With κ = 2, ρ = 8 and η = 1/200: for λ = ½ the
run of width one from `u_0 = v_0 = (1, 1)` converges to a pair `(u, u)` with
`u uᴴ u = ½ u` and `u ≠ 0`, which is not a decayed minimum; for λ = 1 it converges to the
origin; for λ = ½ the run of width two from `U_0 = 1`, `V_0 = −1` converges to the origin;
for λ = ½ the run of width two from `U_0 = V_0 = [[1, 1], [0, 1]]` converges to a decayed
minimum; and for λ = ½ the run of width two from `U_0 = V_0 = diag(1, 0)` converges to a
pair that is not one.

Not covered. Which projection the limit of a run is, beyond the starts `V_0 = U_0` (7). For
λ < 1 its rank is zero exactly from the starts with `U_0 + V_0 = 0` (6). By WD10 it is `n`
for almost every start at a width `k ≥ n` and a small step, and the starts named here with
that property are those of full row rank at `k ≥ 2 n` (WD8), the pairs `(U_0, U_0)` with
`U_0` of full row rank (7), and those with `U_0 + V_0 ≠ 0` of the `1 × 1` identity (WD15). The
starts whose limit has a rank from `1` to `n − 1` are described only among the pairs
`(U_0, U_0)`, where that rank is `rank U_0` (7): the exceptional starts of the identity of a
size above one are not described in general. Kernels other than the identity: there a
stationary point need not have equal factors.

### Lean

`src/Transformer/Grokking/FlatSpectrum/Training/`, namespace
`Transformer.Grokking.FlatSpectrum.Training`, proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| WD1 | `pairing`, `decayedLoss`, `decayStep`, `decayRun`, `stack`, `lift`, `lift_mul_stack`, `stack_decayStep`, `stack_decayRun_succ`, `pairing_stack_lift`, `two_mul_abs_pairing_mul_le`, `lift_posSemidef_of_frobeniusSq_le` | Scalars η, λ are real and act on complex matrices. |
| WD2 | `IsDecayedMin`, `isDecayedMin_of_posSemidef`, `IsDecayedMin.add_nuclearNorm_le`, `IsDecayedMin.weightDecay_eq_nuclearNorm`, `isDecayedMin_iff`, `isDecayedMin_linear_one` | The competitors have every finite width `r`; the factorization itself may have any finite column type. |
| WD3 | `mul_pairing_le_sub_and_le`, `frobeniusSq_first_order`, `hasDerivAt_comp_add_smul`, `add_smul_mul_conjTranspose`, `hasDerivAt_decayedLoss` | The derivative is taken along every line `(U + tA, V + tB)`. |
| WD4 | `gain_identity`, `gain_step`, `gain_iterate`, `pairing_nonneg_of_bounded`, `exists_pos_forall_pairing_nonneg`, `exists_pos_lt_det_add_smul_ne_zero`, `posSemidef_of_tendsto` | Norms are `‖·‖_S2`. Full row rank is `Function.Surjective (W 0).mulVec`; bounded is `∀ t, ‖W_t‖² ≤ B`. |
| WD5 | `lift_mul_stack_eq_zero_of_tendsto_decayRun`, `lift_posSemidef_of_tendsto_decayRun`, `isDecayedMin_of_tendsto_decayRun`, `add_nuclearNorm_le_of_tendsto_decayRun`, `weightDecay_eq_nuclearNorm_of_tendsto_decayRun`, `decayRun_linear`, `tendsto_decayRun_linear` | Convergence is `Tendsto (decayRun G η λ U₀ V₀) atTop (𝓝 (U, V))`. The example is the run above, with the start `stack_fromCols_one`. |
| WD6 | `one_sub_smul_lift_posDef`, `frobeniusSq_mul_conjTranspose_le`, `frobeniusSq_lift_mul_stack_le`, `frobeniusSq_decayStep_le`, `frobeniusSq_decayRun_le`, `one_sub_smul_lift_posDef_of_ball`, `isDecayedMin_of_tendsto_decayRun_of_ball`; squared loss: `frobeniusSq_mask_sub_le`, `neg_le_pairing_mask_sub`, `neg_le_pairing_mask_sub_of_le`, `isDecayedMin_fitLoss_of_tendsto_decayRun`, `decayRun_mask_empty` | The bounds are stated at the matrices with `frobeniusSq X ≤ (ρ / 2) ^ 2`; `‖G X‖ ≤ γ` is `frobeniusSq (G X) ≤ γ ^ 2` with `γ ≥ 0`, and `‖K‖ ≤ κ` is `frobeniusSq K ≤ κ ^ 2` with `κ ≥ 0`. The example run of the two last theorems has no cell seen, where the run is `(1 − η λ)^t (U_0, V_0)`. |
| WD7 | `mask`, `fitLoss`, `fitLoss_first_order`, `fitLoss_eq_zero_iff`, `continuous_mask_sub`, `IsDecayedMin.fitLoss_add_le`, `IsDecayedMin.nuclearNorm_le`, `IsDecayedMin.fitLoss_le`, `isOptimalFit_of_tendsto`, `exists_nuclearNorm_add_le_of_not_isMinOn`, `exists_nuclearNorm_add_le_of_singularValues_eq`, `tendsto_of_fitLoss_tendsto_zero`, `tendsto_mul_conjTranspose_of_forall_isMinOn`, `tendsto_mul_conjTranspose_vecMulVec_of_connected`, `isDecayedMin_one_sub`, `isDecayedMin_shrink`; the subsequence of 1: `exists_subseq_tendsto_of_frobeniusSq_le`, `IsDecayedMin.weightDecay_le_nuclearNorm`, `exists_subseq_tendsto_isOptimalFit`, `exists_isOptimalFit`; the sign change: `frobeniusSq_neg`, `eq_zero_of_neg_eq`, `decayedLoss_neg`, `IsDecayedMin.neg`, `isOptimalFit_neg`, `exists_isOptimalFit_ne`, `not_tendsto_ite_even_neg`, `exists_seq_isDecayedMin_not_tendsto`, `isOptimalFit_one`, `exists_seq_isDecayedMin_one_not_tendsto` | "The only minimizer of M" is D.2's form, `∀ X ∈ S, (IsMinOn ‖·‖_S1 S X ↔ X = K)`. In 1 the factors have a fixed width `r`; in 2 the bound holds for every width; in 3 they have any fixed finite column type. Limits run over sequences `λ_j → 0⁺`. The subsequence is a strictly increasing `φ : ℕ → ℕ`, and the compact set is that of the factors with entries of modulus at most `√(2 ‖K‖_S1)`. "Every width `r ≥ min(m, n)`" is `Fintype.card m ≤ r ∨ Fintype.card n ≤ r`. The fits `((−1)^j U_j, (−1)^j V_j)` are `if Even j then Us j else -Us j` and the same for `Vs`; "do not converge" is `∀ W, ¬ Tendsto Us' atTop (𝓝 W)`, for the first factor; "not the only one" gives an optimal fit with `U' ≠ U`. `IsDecayedMin.neg` holds for every loss and every weight decay. |
| WD8 | `factors`, `flatten`, `factors_flatten`, `flatten_factors`, `continuous_factors`, `continuous_flatten`, `norm_sq_eq_frobeniusSq_factors`, `real_inner_eq_pairing_factors`, `analyticAt_norm_sq`, `flatLoss`, `productVector`, `analyticAt_productVector`, `analyticAt_flatLoss`, `hasGradientAt_flatLoss`, `gradient_flatLoss`, `factors_gradientStep`, `norm_gradient_flatLoss_sq`, `decayedLoss_sub_le`, `pairing_one_sub_smul`, `frobeniusSq_step_mul_sub_le`, `decayedLoss_decayStep_add_le`, `decayedLoss_decayStep_add_le_of_ball`, `decayedLoss_decayRun_succ_add_le`, `norm_apply_le_sqrt_frobeniusSq`, `isCompact_setOf_frobeniusSq_add_le`, `factors_iterate_gradientStep`, `exists_tendsto_decayRun`, `exists_tendsto_decayRun_isDecayedMin`; every real λ: `decayedLoss_decayRun_succ_add_le_of_forall_le`, `exists_tendsto_decayRun_of_forall_le`, `exists_tendsto_decayRun_isDecayedMin_of_forall_le`; squared loss: `fitLoss_eq_add`, `fitLoss_le_add`, `analyticAt_fitLoss`, `exists_tendsto_decayRun_fitLoss`, `exists_tendsto_decayRun_fitLoss_isDecayedMin`, `exists_isDecayedMin_fitLoss`, `exists_tendsto_decayRun_nuclearNorm_add_le` | The factor space is `EuclideanSpace ℂ ((m ⊕ n) × k)` with its real inner product. Real-analytic in the entries is `∀ y, AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y`; the three bounds are stated at the matrices with `frobeniusSq X ≤ (ρ / 2) ^ 2`. Łojasiewicz's theorem and the convergence theorem are `exists_isLojasiewiczAt_of_analyticAt` and `tendsto_of_isLojasiewiczAt_of_le_sub` of `Transformer.Landscape.SaddleAvoidance`, used with `κ = 1/2`. The statements `…_of_forall_le` (modules `BoundedRun` and `Convergent`) have no hypothesis on `lam`; "the run stays in the ball" is `∀ t, frobeniusSq (decayRun G η lam U₀ V₀ t).1 + frobeniusSq (decayRun G η lam U₀ V₀ t).2 ≤ ρ`, and `exists_tendsto_decayRun` is their instance with the ball of WD6. In the existence corollary `κ = ‖K‖_S2² + 1`, `ρ = max (‖W_0‖²) (κ²/λ)` and η is the smaller of the two bounds. |
| WD9 | `isCompact_setOf_flatLoss_le`, `exists_pos_forall_sublevel`, `exists_pos_tendsto_decayRun`, `exists_pos_tendsto_decayRun_isDecayedMin`, `exists_isDecayedMin`; example: `expLoss`, `expGrad`, `pairing_expGrad`, `expLoss_first_order`, `continuous_expGrad`, `analyticAt_expLoss`, `expLoss_nonneg`, `exists_pos_tendsto_decayRun_expLoss_isDecayedMin`, `exists_isDecayedMin_expLoss` | Bounded below is `∀ X, l₀ ≤ L X`. The level bounds the start, `decayedLoss L λ U₀ V₀ ≤ c`, and `η₀` depends on `L`, `G`, λ, `c` and the width. `exists_pos_forall_sublevel` is stated on the factor space, with the null set for Lebesgue measure `volume`; it needs no convexity and uses `exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`, `exists_pos_tendsto_of_mapsTo` and `exists_isLojasiewiczAt_of_analyticOnNhd` of `Transformer.Landscape.SaddleAvoidance`. |
| WD10 | `conjTranspose_mul_self_unit`, `pairing_mul_conjTranspose_unit`, `inner_fderiv_gradient_flatLoss`, `lift_mul_stack_eq_zero_of_gradient_eq_zero`, `surjective_mulVec_of_injective`, `frobeniusSq_conjTranspose_mul_of_surjective`, `lift_posSemidef_of_pairing_sq_le`, `lift_posSemidef_of_injective`, `IsDecayedMin.isLocalMin`, `gradient_eq_zero_and_not_isStrictSaddle_of_isLocalMin`, `isDecayedMin_of_not_isStrictSaddle`, `isDecayedMin_of_not_isStrictSaddle_of_card_le`, `isDecayedMin_of_isLocalMin`, `isDecayedMin_of_isLocalMin_of_card_le`, `exists_pos_ae_tendsto_not_isStrictSaddle`, `exists_pos_ae_tendsto_decayRun`, `exists_pos_ae_tendsto_decayRun_isDecayedMin`, `exists_isDecayedMin_of_card_le`, `exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le`; counterexample: `frobeniusSq_one_fin_two`, `le_decayedLoss_width_one`, `decayedLoss_sqrt_smul_one`, `not_isDecayedMin_width_one` | Dependent columns is `¬ Function.Injective (stack U V).mulVec`; width `k ≥ min(m, n)` is `Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k`. A strict saddle is `IsStrictSaddle (flatLoss L λ) x` of `Transformer.Landscape.SaddleAvoidance`, where `not_isStrictSaddle_of_isLocalMin` is added. The Hessian identity is `⟪S Zᴴ, ∇²F(W) (S Zᴴ)⟫ = ‖Z‖² ⟨S, M S⟩` for one-column `Z`, `A`, `B` and `S = [A; B]`, read off the exact formula for the gradient along the line. `lift_posSemidef_of_injective` holds for any matrix `G` with `M [U; V] = 0`, not only `G (U Vᴴ)`. A local minimum is `IsLocalMin (fun p => decayedLoss L λ p.1 p.2) (U, V)` on the pairs of matrices of one width. Almost every is `∀ᵐ x` for `volume` on the factor space, with the run started at `factors x`; `η₀` is that of `exists_pos_forall_sublevel`. The lower bound for width one holds for every λ ≥ 0 and the value at width two for every λ ≤ 1. |
| WD12 | `linear_first_order`, `pairing_neg_one_smul_one`, `exists_linear_lt`, `analyticAt_linear`, `decayRun_linear_decay`, `decayRun_linear_add`, `tendsto_frobeniusSq_decayRun_linear_atTop`, `not_tendsto_decayRun_linear`, `decayedLoss_linear_smul`, `not_isDecayedMin_linear` | Module `Scope/LinearDiverge`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The loss is `fun X => pairing (-1) X` on `Matrix m m ℂ` and its gradient is the constant `fun _ => -1`. "Not bounded below" is `∀ c, ∃ X, pairing (-1) X < c`, for a nonempty `m`. "The norm tends to infinity" is `Tendsto (fun t => frobeniusSq U_t + frobeniusSq V_t) atTop atTop`, and "no limit" is `¬ Tendsto … (𝓝 W)` for every pair `W`. Real-analytic is in the form of WD8. The start of the example is `fromCols 1 0`, `fromCols 0 1`; its full row rank and the positive definite step matrix are two further examples. |
| WD13 | `tendsto_movingAverage`, `adamRun_succ_first`, `adamRun_succ_second`, `adamRun_succ_params`, `tendsto_adamRun_first`, `tendsto_adamRun_second`, `Hyper.Regular`, `Hyper.source_regular`, `decay_mul_eq_of_tendsto_adamRun`, `abs_decay_mul_lt_one_of_tendsto_adamRun`, `eq_zero_of_tendsto_adamRun`, `eq_zero_or_abs_eq_of_tendsto_adamRun`; witnesses: `adamRun_zero`, `scalarField`, `continuous_scalarField`, `hasDerivAt_scalarLoss`, `not_tendsto_adamRun_scalar`, `factorField`, `continuous_factorField`, `hasDerivAt_factorLoss_left`, `hasDerivAt_factorLoss_right`, `factorLoss`, `factorLoss_eq`, `five_halves_le_factorLoss`, `factorLoss_le_iff`, `not_tendsto_adamRun_factor` | Modules `src/Transformer/Grokking/Protocol/AdamLimit` and `AdamLimitWitness`, namespace `Transformer.Grokking.Protocol`, not the namespace of this section: the update is `adamStep` and the run `adamRun` of the protocol of the division experiment, with the constants `Hyper` (`Hyper.source wd` for the source's). "Every update sees one field" is the oracle `fun _ => g`; the parameters are a function `ι → ℝ` and convergence is in the product topology. The hypotheses on the constants are `Hyper.Regular h`, and the learning rate is `Hyper.rateAt`, constant after the warmup. `|θ| < 1 / wd` is stated as `|h.decay * θ i| < 1`. The factors are `θ false` and `θ true`; `factorField` is proved to be the gradient of `½ (u v − 3)²` coordinate by coordinate, and "a minimizer" is the hypothesis `∀ u v, factorLoss (θ false) (θ true) ≤ factorLoss u v`. The remark on ε → 0 and the citation are not formalized. |
| WD14 | `stack_decayRun_mulVec_eq_zero`, `stack_mulVec_eq_zero_of_tendsto`, `rank_le_of_forall_mulVec_eq_zero`, `rank_stack_decayRun_le`, `rank_mul_conjTranspose_le_of_tendsto`, `not_isDecayedMin_of_rank_lt`; the `2 × 2` identity: `one_lt_rank_of_isNuclearMin_one`, `not_isDecayedMin_one_of_rank_le_one`, `exists_tendsto_decayRun_one_not_isDecayedMin`; padded pairs: `fromCols_zero_mul_conjTranspose`, `frobeniusSq_fromCols_zero`, `decayStep_fromCols_zero`, `decayRun_succ`, `decayRun_fromCols_zero`, `tendsto_decayRun_fromCols_zero`, `decayedLoss_fromCols_zero`, `isDecayedMin_fromCols_zero_iff`, `rank_stack_fromCols_zero_le`; stationary points: `conjTranspose_mul_self_eq_of_stationary`, `frobeniusSq_eq_of_stationary`, `gram_mul_conjTranspose_of_stationary`, `nuclearNorm_mul_conjTranspose_of_stationary`, `rank_mul_conjTranspose_of_stationary`, `isDecayedMin_iff_isNuclearMin_of_stationary`, `isDecayedMin_iff_isNuclearMin_of_tendsto`; the starts `V₀ = −U₀`: `decayStep_neg`, `decayRun_neg`, `eq_neg_of_tendsto_decayRun_neg`, `rank_stack_neg`, `not_isNuclearMin_one_mul_neg`, `isHermitian_mask_sub_one`, `not_isDecayedMin_one_of_neg`, `exists_tendsto_decayRun_one_neg_not_isDecayedMin` | Modules `Scope/StackRank`, `Scope/RankStart`, `Scope/PaddedRun`, `Scope/Balanced` and `Scope/NegStart`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. `W z = 0` is `stack U V *ᵥ z = 0` and the rank is `Matrix.rank`; statements 1, 2 and 4 take any `G`, `η` and `λ`, and a limit is `Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))`. Statement 2 for a limit is `(U * Vᴴ).rank ≤ (stack U₀ V₀).rank`. "Every trained product has a rank above" is `∀ X, IsNuclearMin L lam X → (stack U₀ V₀).rank < X.rank`; `r*` is not defined. `[u 0]` is `fromCols u 0` with the column type `k₁ ⊕ k₂`. For the identity the gradient field is `fun X => mask Set.univ (X - 1)`, "rank two" is `1 < X.rank`, the start has `(stack U₀ V₀).rank ≤ 1` at any finite column type, and the bounds of WD8 are those of `exists_tendsto_decayRun_fitLoss` with `2 ≤ κ ^ 2`. That the starts of rank at most one are the `S wᴴ`, satisfy the minor equations and form a null set is not formalized. The start of the example is `fromCols (of fun _ _ => 1) 0` for both factors. In statement 6 "stationary" is `lift lam G * stack U V = 0` for any matrix `G`; the Gram identity is stated as `(U * Vᴴ)ᴴ * (U * Vᴴ) = (V * Vᴴ)ᴴ * (V * Vᴴ)`, the trace is taken by `nuclearNorm_eq_trace_of_mul_self`, the ranks are the conjunction `(U * Vᴴ).rank = U.rank ∧ (U * Vᴴ).rank = V.rank`, and the statement for a limit takes a continuous `G` and `η > 0`. In statement 7 "maps Hermitian matrices to Hermitian matrices" is `∀ X, X.IsHermitian → (G X).IsHermitian`, the start is `(U₀, -U₀)` at any finite column type, the conclusions are `(decayRun G η lam U₀ (-U₀) t).2 = -(decayRun G η lam U₀ (-U₀) t).1` and `V = -U` at a limit, and `(stack U (-U)).rank = U.rank`; for the identity the product is `U * (-U)ᴴ`, the hypothesis on the ball is `2 * frobeniusSq U₀ ≤ ρ`, and the example is the start `(1, -1)` of width two, whose stack has rank `2`, with the numbers of the example of statement 5. |
| WD15 | `frobeniusSq_one_fin_one`, `frobeniusSq_complex_smul`, `fin_one_mul`, `decayStep_one_add`, `im_mul_conjTranspose_apply`, `frobeniusSq_I_smul`, `re_mul_conjTranspose_apply`, `le_frobeniusSq_decayStep_one_add`, `frobeniusSq_add_decayStep_one`; the run: `frobeniusSq_decayRun_one_le`, `frobeniusSq_add_decayRun_one_pos`, `frobeniusSq_pos_of_tendsto_decayRun_one`, `one_add_one_fin_one_ne_zero`; the limit: `isDecayedMin_one_of_stationary`, `not_isDecayedMin_one_fin_one_neg`, `exists_tendsto_decayRun_one_isDecayedMin_iff`, `isDecayedMin_one_of_stationary_of_one_le`, `exists_tendsto_decayRun_one_isDecayedMin_of_one_le` | Modules `Scope/OneByOne`, `Scope/OneByOneRun` and `Scope/OneByOneLimit`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The factors are `Matrix (Fin 1) k ℂ` for any finite `k`, the gradient field is `fun X => mask Set.univ (X - 1)` and the loss is `fitLoss Set.univ 1`. `Re x` and `Im x` are `((U * Vᴴ) 0 0).re` and `.im`, `⟨P, i Q⟩` is `pairing (U + V) (Complex.I • (U - V))`, and the squared norms are `frobeniusSq`. The inequality of statement 1 has `Re x` written as `(frobeniusSq (U + V) - frobeniusSq (U - V)) / 4`: its hypothesis is `0 ≤ a`, its conclusion `a * (a − η q) * p ≤ ‖P'‖²` with `a − η q` written as `1 - η * (lam - 1 + (p + 3 * q) / 4)`; the equality for `‖P'‖²` is inside its proof. Statement 2 is one lemma, the conjunction of `η ^ 2 * ‖P‖² ≤ ‖P'‖²` and the implication from `‖U‖² + ‖V‖² ≤ 2 * (1 - lam) / 3`; it takes `0 < η`, `0 ≤ κ`, the second bound of WD8 and the ball. "The bounds of WD8 with `κ ≥ 1`" are the hypotheses of `exists_tendsto_decayRun_fitLoss` with `0 ≤ κ` and `1 ≤ κ ^ 2`. Statement 3 is `0 < frobeniusSq U + frobeniusSq V` for a limit `(U, V)` of the run. In statements 4 and 6 "stationary" is `lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0`, and "other than the origin" is `0 < frobeniusSq U + frobeniusSq V`. The theorem for `0 < λ < 1` is `∃ U V, Tendsto (decayRun _ η lam U₀ V₀) atTop (𝓝 (U, V)) ∧ (IsDecayedMin (fitLoss Set.univ 1) lam U V ↔ U₀ + V₀ ≠ 0)`, and for `1 ≤ λ` the same with `IsDecayedMin` alone. That every start lies in a ball of WD8 is not formalized. The examples are those of the item, of width one; the witness of statement 4 is the limit of the run from `(1, 1)`, and that of statement 6 is the origin with λ = 1. |
| WD16 | `mask_univ`, `eq_add_of_stationary_identity`, `eq_and_mul_of_stationary_identity`, `stationary_identity_of_mul`, `stationary_identity_iff`, `pairing_mul_conjTranspose_one`, `frobeniusSq_mul_conjTranspose_of_mul`, `mul_self_of_stationary_identity`, `eq_zero_of_stationary_identity`, `smul_one_mul_conjTranspose`, `half_smul_one_mul`, `halfDiag`, `halfDiag_mul`; the minima: `lift_smul_one_posSemidef`, `eq_smul_one_of_isNuclearMin_identity`, `mul_conjTranspose_eq_of_isDecayedMin_identity`, `mul_conjTranspose_ne_smul_one`, `not_isDecayedMin_identity_of_card_lt`, `isDecayedMin_identity_iff_of_stationary`, `isNuclearMin_identity_iff`, `isDecayedMin_identity_zero`, `halfDiag_mul_conjTranspose_ne`; the runs: `eq_and_mul_of_tendsto_decayRun_identity`, `isDecayedMin_identity_iff_of_tendsto`, `exists_tendsto_decayRun_identity`, `tendsto_decayRun_identity_zero`, `tendsto_decayRun_identity_neg`; the origin: `decayStep_identity_add`, `four_smul_cubic`, `four_mul_pairing_add_cubic_le`, `le_frobeniusSq_decayStep_identity_add`, `frobeniusSq_add_decayStep_identity`, `frobeniusSq_decayRun_identity_le`, `frobeniusSq_add_decayRun_identity_pos`, `frobeniusSq_pos_of_tendsto_decayRun_identity`, `tendsto_decayRun_identity_zero_iff`, `exists_tendsto_decayRun_identity_eq_zero_iff`, `ones_add_ones_ne_zero`, `frobeniusSq_ones_add_le`; equal factors: `symStep`, `decayStep_identity_self`, `decayRun_identity_self`, `mul_symStep`, `commute_symStep`, `commute_iterate_symStep`, `mul_iterate_symStep`, `frobeniusSq_mul_le_of_idempotent`, `shearFactor`, `shearFactor_mul`, `frobeniusSq_shearFactor`, `exists_tendsto_decayRun_identity_self`, `exists_tendsto_decayRun_identity_self_isDecayedMin`, `rank_stack_self_le`, `exists_mul_eq_one_of_tendsto_decayRun_self`, `range_mul_conjTranspose_of_tendsto_decayRun_identity_self`, `isDecayedMin_iff_of_tendsto_decayRun_identity_self` | Modules `Scope/IdentityStationary`, `Scope/IdentityMin`, `Scope/IdentityLimit`, `Scope/IdentityOrigin`, `Scope/IdentityOriginRun`, `Scope/IdentitySymmetric`, `Scope/IdentitySymmetricLimit` and `Scope/IdentitySymmetricRange`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The identity is `(1 : Matrix m m ℂ)` for any finite `m`, the factors are `Matrix m k ℂ` for any finite `k`, the gradient field is `fun X => mask Set.univ (X - 1)` and the loss is `fitLoss Set.univ 1`. "Stationary" is `lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0`. Statement 2 is `(U * Vᴴ).PosSemidef ∧ (U * Vᴴ) * (U * Vᴴ) = (1 - lam) • (U * Vᴴ)`; "orthogonal projection" is not defined, and that the product has the rank of `U` is `rank_mul_conjTranspose_of_stationary` of WD14. The trained product is `IsNuclearMin (fitLoss Set.univ 1) lam X ↔ X = (1 - lam) • 1`; the displayed inequality is inside the proof of `eq_smul_one_of_isNuclearMin_identity`, with `frobeniusSq 1` in place of `n`. "A width `k < n`" is `Fintype.card k < Fintype.card m`, and the statement holds for every pair `U V : Matrix m k ℂ`. `[[λ, c], [c, λ]]` is `lift lam (c • 1)` with `c : ℝ`; its form is inside the proof. In statement 5 a limit is `Tendsto (decayRun _ η lam U₀ V₀) atTop (𝓝 (U, V))` with `0 < η`, and no bound on the step is assumed. In the theorem the bounds of WD8 are the hypotheses of `exists_tendsto_decayRun_fitLoss` with `frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2`; the conclusion is `∃ U, Tendsto _ atTop (𝓝 (U, U)) ∧ U * Uᴴ * U = (1 - lam) • U`, for `1 ≤ λ` it is `Tendsto _ atTop (𝓝 (0, 0))`, and for the start `(U₀, -U₀)` the hypothesis on the ball is `2 * frobeniusSq U₀ ≤ ρ`. The rank bound of the theorem is `rank_mul_conjTranspose_le_of_tendsto` of WD14 and is not restated; that the limit at a width `k < n` is not a decayed minimum is `not_isDecayedMin_identity_of_card_lt`. In statement 6 the inequality is `a * (a - η * (frobeniusSq U + frobeniusSq V)) * frobeniusSq (U + V) ≤ frobeniusSq P'` with `a` written out as `1 + η * (1 - lam)`, under `0 ≤ η` and `0 ≤ a`; the statement in the ball is one lemma, the conjunction of `η ^ 2 * ‖P‖² ≤ ‖P'‖²` and the implication from `‖U‖² + ‖V‖² ≤ 1 - lam`, under `0 < η`, `0 ≤ κ`, the second bound of WD8 and the ball. "Does not converge to the origin" is `0 < frobeniusSq U + frobeniusSq V` for a limit `(U, V)`. The equivalence of the theorem is `Tendsto _ atTop (𝓝 (0, 0)) ↔ U₀ + V₀ = 0` and, with the limit, `∃ U, Tendsto _ atTop (𝓝 (U, U)) ∧ U * Uᴴ * U = (1 - lam) • U ∧ (U = 0 ↔ U₀ + V₀ = 0)`; that the product of a nonzero limit of width one has rank one is not stated. In statement 7 the step is `symStep η lam U`, the run from `(U₀, U₀)` is `(symStep η lam)^[t] U₀` in both components, "commutes" is `Commute`, and a Hermitian idempotent is `E * E = E` with `Eᴴ = E`; `E` and the run of `E U_0` are inside the proof. The conclusion is `∃ U, Tendsto _ atTop (𝓝 (U, U)) ∧ U * Uᴴ * U = (1 - lam) • U ∧ U * Uᴴ * U₀ = (1 - lam) • U₀`, with the ball as `2 * frobeniusSq U₀ ≤ ρ`; "full row rank" is a right inverse, `U₀ * R = 1`. That the product is the projection onto the column space of `U_0` is `LinearMap.range (U * Vᴴ).mulVecLin = LinearMap.range U₀.mulVecLin` for a limit `(U, V)`, beside statement 2; the rank of the stack is `(stack U U).rank ≤ U.rank`. The equivalence is `IsDecayedMin (fitLoss Set.univ 1) lam U V ↔ ∃ R, U₀ * R = 1` for a limit `(U, V)` under the bounds of WD8; its half "only if" is also `exists_mul_eq_one_of_tendsto_decayRun_self`, for every gradient field, step and weight decay and a limit with the product `c • 1`, `c ≠ 0`. `shearFactor` is `!![1, 1; 0, 1]`. `halfDiag` is `(1 / 2 : ℝ) • Matrix.diagonal ![1, 0]`. The examples are those of the item. |

## Flat kernels of any shape (added)

The note calls a kernel flat when all its singular values are equal, and states
Proposition 2, Theorem A and the Corollary for a square kernel without zero entries. Two
cases fall outside: a kernel whose rows and columns are indexed by different sets, such as
the recall kernel of the next section, and a flat kernel with zeros, such as the identity.
This section defines flat for an `m × n` kernel and extends the statements that use
flatness alone: the norms, Theorem A, the cost of low rank, and what training returns.

### R1. The definition and the norms

An `m × n` kernel `K` is flat with squared singular value `c > 0` when

```
Kᴴ K = c I    (orthogonal columns of squared length c)
or    K Kᴴ = c I    (orthogonal rows of squared length c).
```

Equal singular values `s` give `Kᴴ K = s² I`, and `Kᴴ K = c I` gives singular values `√c`,
so for a square kernel this is the notion of the note with `c = s²`. In both cases
`(Kᴴ K)² = c Kᴴ K`, so `K / √c` is a partial isometry, and a contraction. From
`I = (Kᴴ / c) K` or `I = K (Kᴴ / c)` the rank is `min(m, n)`. Hence

```
rank K = min(m, n),    ‖K‖_S1 = min(m, n) √c,    ‖K‖_S2² = min(m, n) c,    r_eff K = min(m, n).
```

Every `m × n` matrix has `r_eff ≤ rank ≤ min(m, n)` (R3), so a flat kernel has the largest
effective rank of its shape. For `p × p` entries of modulus one this is the upper bound
`‖K‖_S1 ≤ p^{3/2}` of Proposition 2.

### R2. Theorem A by the support

Item 5 of *Rank above one*: `K = U diag(s) Vᴴ`, with orthonormal columns in `U` and `V` and
`s > 0`, is a minimum of M from Ω exactly when some contraction `W` with `Uᴴ W = 0` and
`W V = 0` makes `U Vᴴ + W` vanish off Ω. If `V` is square, `W V = 0` forces `W = 0`
(item 5.4). If `U` is square, `Uᴴ W = 0` forces `W = 0` in the same way. So at full row
rank, as at full column rank, `K` is a minimum of M exactly when `U Vᴴ` vanishes off Ω.

A flat kernel has `U Vᴴ = K / √c`: take `U = K / √c`, `V = I` if `Kᴴ K = c I`, and `U = I`,
`V = Kᴴ / √c` if `K Kᴴ = c I`. Therefore

```
a flat K is a minimum of M from Ω    ⟺    K_ij = 0 at every unseen cell (i, j).
```

The seen cells must contain the support of `K`.

- A flat kernel is not a minimum of M when some nonzero cell is unseen.
- A flat kernel without zero entries is a minimum of M from no proper Ω, and M recovers it
  from none. For a square kernel this is the last paragraph of Theorem A.
- The zeros of a flat kernel may stay unseen: the identity is a minimum of M from every Ω
  that contains its diagonal. This says that `K` is a minimum. Whether it is the only one
  is the recovery question of Theorems D to G and is not answered here.

### R3. No cheap directions

The nonzero singular values of an `m × n` matrix `X` number `rank X`, so by Cauchy–Schwarz

```
‖X‖_S1² ≤ rank X · ‖X‖_S2²,    hence    r_eff X ≤ rank X ≤ min(m, n).
```

Let `K` be flat with squared singular value `c`, and `r₀ = min(m, n)`. Since `K / √c` is a
contraction, `⟨K, X⟩ ≤ √c ‖X‖_S1`, the dual pairing of Theorem A. With `‖K‖_S2² = c r₀`,

```
‖K − X‖_S2² = c r₀ − 2 ⟨K, X⟩ + ‖X‖_S2²
            ≥ c r₀ − 2 √c √(rank X) ‖X‖_S2 + ‖X‖_S2²
            ≥ c (r₀ − rank X).
```

So a product `A B` through width `k` misses a flat kernel by at least `c (r₀ − k)`, the
fraction `1 − k / r₀` of `‖K‖_S2²`, and an exact factorization has `k ≥ r₀`. Each unit of
width buys at most the same share `1 / r₀`: no direction of a flat kernel is cheaper to fit
than another.

### R4. Training

Theorem A enters *Training reaches the optimal fit* through one fact: `K` is not a minimum
of M from Ω. Items WD7, WD8 and WD10 then say that the trained fits, the limits of the runs
of WD8 and the limits of almost every run of width at least `min(m, n)` stay a fixed
`δ > 0` below `‖K‖_S1`. They are stated there for a square kernel with equal singular values
and no zero entry. The proofs use only that fact, so by R2 the statements hold for a flat
kernel of any shape whenever some nonzero cell is unseen:

- no optimal fit to the values of `K` on Ω multiplies to `K`;
- some `δ > 0` has `‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1` for the trained fits of every weight decay
  `λ > 0` and every width;
- every run of WD8 from stacked factors of full row rank converges to factors with that
  bound, and so does almost every run of width `k ≥ min(m, n)` with a small step (WD10).

By R3, with every cell seen, a factorization `U Vᴴ` of width `k` keeps the squared loss
`½ ‖U Vᴴ − K‖_S2² ≥ c (min(m, n) − k) / 2`.

### Lean

`FlatKernel`, `FlatSupport`, `FlatCapacity` and `Training/AnyShape` of
`src/Transformer/Grokking/FlatSpectrum/`, namespace `Transformer.Grokking.FlatSpectrum`,
proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| R1 | `IsFlat`, `IsFlat.conjTranspose`, `isFlat_of_singularValues_eq`, `singularValues_eq_of_conjTranspose_mul_self`, `IsFlat.gram_mul_gram`, `rank_eq_card_of_conjTranspose_mul_self`, `rank_eq_card_of_mul_conjTranspose_self`, `IsFlat.rank_eq`, `IsFlat.effectiveRank_eq`, `IsFlat.frobeniusSq_eq`, `IsFlat.nuclearNorm_eq`, `IsFlat.isContraction`; instances `isFlat_one`, `isFlat_cubicKernel`, `isFlat_row_ones` | `IsFlat K c` is the disjunction, with `c` real; the theorems take `c > 0` or `c ≠ 0` as a hypothesis. `singularValues` is indexed by the columns, so for `K Kᴴ = c I` with `m < n` it has `n − m` zeros; the norms are stated with `min (card m) (card n)`. |
| R2 | `isMinOn_iff_of_full_row_rank`, `IsFlat.isMinOn_iff`, `IsFlat.not_isMinOn_of_ne_zero`, `IsFlat.not_isMinOn`, `IsFlat.not_forall_isMinOn_iff` | A minimum of M from Ω is `IsMinOn nuclearNorm S K` for the set `S` of the matrices equal to `K` on Ω, as in Theorem A. |
| R3 | `nuclearNorm_sq_le_rank_mul_frobeniusSq`, `effectiveRank_le_rank`, `effectiveRank_le_min`, `IsFlat.pairing_le`, `IsFlat.mul_sub_rank_le_frobeniusSq_sub`, `IsFlat.mul_sub_card_le_frobeniusSq_sub`, `IsFlat.card_le_of_eq_mul` | `⟨K, X⟩` is `Training.pairing`. The width `k` is the cardinality of a finite column type. Whether a product through width `k` attains `c (r₀ − k)` is not formalized. |
| R4 | `Training.exists_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn`, `Training.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn`, `IsFlat.not_exists_isOptimalFit`, `IsFlat.exists_nuclearNorm_add_le`, `IsFlat.exists_tendsto_decayRun_nuclearNorm_add_le`, `IsFlat.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le`, `IsFlat.mul_sub_card_le_fitLoss_univ` | The two first theorems take "`K` is not a minimum of M from Ω" as their hypothesis, for any kernel. "Some nonzero cell is unseen" is `(i, j) ∉ Ω` with `K i j ≠ 0`. The width `k ≥ min(m, n)` is `card m ≤ card k ∨ card n ≤ card k`, and the constants of the runs are those of WD8 and WD10. The squared loss with every cell seen is `fitLoss Set.univ K (U * Vᴴ)`. |

## Associative recall has a flat spectrum (added)

Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*,
arXiv:2312.04927v1, explain the gap between attention and gated convolutions by one task,
associative recall (Appendix `app:retnet-proof`): "The AR problem takes key-value pairs
`{k_i, v_i}_{i=0}^{n−1}` along with a query `q` appended at the end as input and the goal is
to output `v_i` if `q = k_i` for some `i ∈ [0, N − 1]`." Their Claim 1 (§4) is measured:
"Gated-convolution models with two layers require model dimension to scale at least linearly
in sequence length in order to solve associative recall, while attention models can solve it
with near-constant dimensionality." Their Corollary `cor: space-ar` bounds the RetNet by
counting bits.

This section puts the task in the setting of this note. Recall is a task kernel. The kernel
is flat, and flatness prices the width: a readout through which `d` numbers cross from the
context to the query misses at least the fraction `1 − d / m` of recall with `m` bindings. One
gated convolution, the RetNet output and attention without softmax are such readouts.

### AR1. The kernel

Keys `κ`, `m = |κ|`; values `ν`, `c = |ν|`. A context binds each key to a value: a
dictionary `π : κ → ν`. The answer to the query `q` is `π(q)`. For a readout `φ : ν → ℂ` of
the answer the recall kernel is

```
K(q, π) = φ(π(q)),      m rows,  c^m columns.
```

It is the task kernel `φ(f(a, b))` of the Setting for the operation `f(q, π) = π(q)`,
evaluation, whose two arguments have different types. The readout is centered when
`Σ_v φ(v) = 0`.

### AR2. Flatness

A dictionary is the value of one key and, independently, the values of the other keys. So
for functions `f`, `g` of a value and keys `q' ≠ q`,

```
Σ_π f(π q) = c^{m−1} Σ_v f(v),      Σ_π f(π q) g(π q') = (Σ_v f(v)) · Σ_ρ g(ρ q'),
```

with `ρ` over the dictionaries of the keys other than `q`. For a centered readout the second
sum vanishes with `f = φ`, `g = conj φ`, so the rows of `K` are orthogonal:

```
K Kᴴ = σ² I,      σ² = c^{m−1} Σ_v |φ(v)|².
```

The recall kernel of a centered readout that is not zero is flat (R1), with

```
rank K = m,      r_eff K = m.
```

Every key is a direction of the same weight. The table of `a + b` has effective rank one and
that of `a³ + ab` has effective rank `p`, its number of rows; recall is at the same end as
`a³ + ab`.

### AR3. Theorem A: unseen pairs

By R2, for a centered readout that is not zero,

```
K is a minimum of M from Ω    ⟺    φ(π q) = 0 at every unseen pair (q, π).
```

For a readout without zeros, such as `±1` on two values, the recall kernel is a minimum of M
from no proper set of pairs.

### AR4. Width: the regression bound

A model reads the context and then the query. Its readout `F(q, x)` of the query `q` after
the context `x` has cut width `d` when

```
F(q, x) = g(q) + Σ_{r<d} a_r(q) b_r(x).
```

Apart from a term of the query alone, `d` numbers `b_r(x)` are all that crosses from the
context to the query, and the query combines them linearly. The functions `g`, `a_r`, `b_r`
are arbitrary.

Let `F` have cut width `d` on the dictionaries and let `φ` be real with mean `μ`. Then

```
Σ_{q, π} (φ(π q) − F(q, π))²  ≥  (m − d) c^{m−1} Σ_v (φ(v) − μ)².
```

There are `m c^m` pairs, so the mean squared error is at least `(1 − d / m) Var φ`.

Proof. Subtract from each `b_r` its mean `β_r` over the dictionaries:
`F(q, π) = γ(q) + X(q, π)` with `X(q, π) = Σ_r a_r(q) (b_r(π) − β_r)`. For each `q` the
numbers `φ(π q) − μ − X(q, π)` sum to `0` over `π` (AR2), so removing the constant
`γ(q) − μ` from them does not lower their sum of squares. What remains is `‖K − A B‖_S2²` for
the recall kernel `K` of `φ − μ` and a product through width `d`, at least `(m − d) σ²` by
R3. ∎

For a readout `φ` that is not constant:

- exact recall, `F(q, π) = φ(π q)` for all `q` and `π`, has `d ≥ m`;
- one-hot keys attain it: `a_r(q) = [r = q]` and `b_r(π) = φ(π r)` have cut width `m`.

So recall has cut width `d` exactly when `d ≥ m`.

### AR5. Width: a sign or an argmax

AR4 bounds a regression. A classifier is right when a sign or an argmax is right, with any
margin. The count of keys survives.

Two values. If `F` has cut width `d` and `F(q, π) > 0` exactly when `π(q)` is the second
value, for every dictionary, then `d ≥ m`.

Proof. If `m > d`, the vectors `a(q) ∈ ℝ^d` are dependent: `Σ_q λ_q a(q) = 0` with `λ ≠ 0`,
so `Σ_q λ_q F(q, π) = Σ_q λ_q g(q)` for every `π`. For the dictionary with `π(q)` the second
value exactly when `λ_q > 0`, every term `λ_q F(q, π)` is `≥ 0`. For the one with `π(q)` the
second value exactly when `λ_q < 0`, every term is `≤ 0`. At a key with `λ_q ≠ 0` one of the
two is strict. So `Σ_q λ_q g(q)` is both `≥ 0` and `≤ 0`, with one inequality strict. ∎

Any number of values. If the largest of the logits `L_v(q, π)` is always that of the bound
value and the difference `L_{v₁} − L_{v₀}` of two of them has cut width `d`, then `d ≥ m`: on
the dictionaries with values in `{v₀, v₁}` that difference is the sign readout above.

### AR6. One gated convolution

The Coyote layer of the paper (§4, equation `eq: coyote-recursion`; printed as BaseConv) is

```
y = (u W + b₁) ⊙ (h ∗ u + b₂)
```

on a sequence `u` of tokens of dimension `d`, with a causal convolution in each of the `d`
channels. Let the prompt be `n` context tokens and then the query token, and read the layer
at the query. The first factor is a function of the query token. The convolution is additive
in `u`, so the second factor is a function of the query token plus a function of the context
tokens, one number for each channel. Hence every linear readout `Σ_c w_c y_c` at the query
has cut width `d`, for every encoding of the dictionary in the context tokens and of the key
in the query token and every `W`, `h`, `b₁`, `b₂`, `w`. The circular convolution that Appendix
`sec:simplified-hyena-append` also allows sums the same terms at the last position, so that
layer is the same at the query. By AR4 and AR5:

- the squared error of the layer on recall is at least `(m − d) c^{m−1} Σ_v (φ(v) − μ)²`;
- exact recall of a readout that is not constant has `d ≥ m`;
- logits `Σ_c R_{cv} y_c` whose largest is always that of the bound value have `d ≥ m`;
- a layer of dimension `m` recalls `m` bindings: one-hot keys, the value of the key `j` in
  every channel of the token `j`, and the filter that shifts the token `c` to the query in
  channel `c`. Its context lists the values in the order of the keys. For a context that
  lists its bindings in another order the three bounds hold as they stand, and no layer that
  attains them is constructed.

The number of bindings that a context can hold grows linearly with its length. So for one
layer read linearly the first half of Claim 1 is a theorem: the dimension is at least the
number of bindings.

### AR7. The RetNet output

The RetNet of the paper (Appendix `app:retnet-proof`) keeps a `d × d` state
`z^n = γ z^{n−1} + A[n,:]ᵀ V[n,:]` and outputs `Out[N,:] = C[N,:] z^N` at the query. Corollary
`cor: space-ar` reads: "The RetNet model requires Ω(N)-bits to solve AR for d ≤ √N". Its proof
counts the bits of the state; `exact_index_requires_bits` is the finite-state form, and a
state of real numbers has no fixed number of bits.

The state holds `d²` numbers, but the query reads it through the `d` numbers `C[N,:]`:

```
Σ_c w_c Out[N,c] = g(q) + Σ_r C_r(q) · γ Σ_c w_c z^{N−1}[r,c],
```

and `z^{N−1}` depends on the context alone. So every linear readout of the output has cut
width `d`, at any precision, and the three bounds of AR6 hold with `d` the dimension of the
state: recall of `m` bindings has `d ≥ m`. For `√N < d < N` the count of bits does not
exclude recall of `N` bindings; this bound does.

### AR8. Attention without softmax

Proposition `prop: attention-ar` of the paper: "Attention (even without using soft-max)
solves MQAR for `u` using O(c²) parameters", with one-hot tokens of dimension `3c` for a
vocabulary of size `c` (Appendix Proposition `prop: app-attention`). Attention without
softmax returns `Σ_j ⟨Q q, K x_j⟩ V x_j` at the query. The score of the token `j` is bilinear
in the projections `Q q` and `K x_j` of dimension `d`, so every linear readout of the output
has cut width `d`, when `Q q` depends on the query alone, and the three bounds of AR6 hold
with `d` the dimension of the projections. For such a layer the dimension is at least the
number of bound keys; the construction of the paper has dimension `3c`.

The appendix adds linear biases to the scores (Algorithm `ALiBi-without-softmax`):
`O = (Q Kᵀ + B) V` with a fixed matrix `B` of positions. The row of `B` at the query adds
`Σ_j B_j V x_j`, a function of the context alone: one more number crosses, the cut width is
`d + 1`, and the three bounds hold with `d + 1` in place of `d`. The unit is used: with the
bias, two keys are recalled at dimension one. The second layer of the paper's construction
has `B = 0`, and its first layer leaves the query token as it is, so that construction is of
the first form.

### AR9. Training

The pure factorization `U Vᴴ` of *Training reaches the optimal fit*, with a row of `U` for
each key and a row of `V` for each dictionary, is the bilinear model of recall. For a
centered readout, by R4:

- with every pair seen, width `k` keeps a squared loss of at least `σ² (m − k) / 2`;
- for a readout without zeros and a proper set Ω of pairs, the trained fits of every weight
  decay and every width stay a fixed `δ > 0` below `‖K‖_S1`, and almost every run of width
  at least `m` with a small step converges to such a fit.

### AR10. Scope

Proved: AR1 to AR9 as stated, for every finite set of keys and of values.

Not covered:

- Depth. One layer reads the context at the query. The context tokens may be any function of
  the dictionary, so layers that act on the context alone are covered. A stack in which two
  layers read the context at the query is not of the form of AR4. *Polynomial readouts and
  two layers* below treats two gated convolutions with a linear readout: exact recall has
  `m ≤ 10 d`. Claim 1 of the paper, for two layers with MLPs and a decoded answer, stays a
  measurement.
- Nonlinear readouts. An MLP after the layer is not a linear readout; AR4 and AR5 do not
  bound it. The same section bounds an MLP with a polynomial activation.
- Softmax. Softmax divides the scores by a function of the query and the context together.
  That output is not of the form of AR4. *Softmax attention escapes the cut* below treats one
  layer of it: it recalls at projection dimension two and pays in the scale of its scores.
- The family of contexts. The bounds range over every dictionary of the `m` keys. A model
  that meets a smaller family of dictionaries is not constrained by them. *Families of
  dictionaries and the multi-query task* below gives the regression and the exact bound on
  every pairwise uniform family, such as the `p²` affine dictionaries.
- The multi-query task of the paper (`def: general-AR`), with repeated keys and queries
  without a match: here every key is bound once and one query follows the context. The same
  section shows that these inputs are inputs of that task, so the lower bounds transfer.
- AR3 and AR9 say that the recall kernel is not a minimum of M and is not returned by
  training from a part of its table. They do not say what the minimum is or how far its
  decoded answers are from recall. *One unseen cell* below gives the minimum when one pair is
  unseen: it answers that pair with the opposite phase, at a modulus of the order `m / |ν|^m`
  of the right one for a readout of modulus one.

### Lean

`src/Transformer/Grokking/FlatSpectrum/Recall/`, namespace
`Transformer.Grokking.FlatSpectrum.Recall`, proved without `sorry`. The models are those of
`Transformer.Zoology` (`coyoteLayer`, `retNetRun`).

| Item | Lean | Differences |
| --- | --- | --- |
| AR1 | `recallKernel`, `HasCutWidth` | The kernel is `Matrix κ (κ → ν) ℂ`; the keys and the values are finite types. `HasCutWidth d F` asks for `g`, `a`, `b` with values in `ℝ` and features indexed by `Fin d`. |
| AR2 | `sum_apply`, `sum_apply_mul_apply`, `recallKernel_mul_conjTranspose`, `isFlat_recallKernel`, `scale_pos`, `rank_recallKernel`, `effectiveRank_recallKernel` | The two sums hold in every commutative semiring. "Not zero" is `φ v ≠ 0` for some `v`, which makes `σ² > 0`. |
| AR3 | `isMinOn_recallKernel_iff`, `not_isMinOn_recallKernel` | "Without zeros" is `∀ v, φ v ≠ 0`. |
| AR4 | `sum_sq_le_sum_sub_sq`, `exists_centering`, `HasCutWidth.mul_le_sum_sq`, `HasCutWidth.card_le_of_recall`, `hasCutWidth_recall`, `hasCutWidth_recall_iff` | The mean `μ` is any number with `Σ_v (φ v − μ) = 0`. "Not constant" is `φ v ≠ φ v'` for some `v`, `v'`. The bound is stated for the sum over all pairs, not divided by their number. |
| AR5 | `HasCutWidth.comp_context`, `HasCutWidth.card_le_of_sign`, `card_le_of_argmax` | The two values are `Bool`, the second one `true`. A right argmax is `L v q π < L (π q) q π` for every `v ≠ π q`. |
| AR6 | `prompt`, `prompt_eq_add`, `causalConvolution_add`, `coyoteLayerCyclic_last`, `hasCutWidth_coyoteLayer`, `coyoteLayer_recall_error`, `card_le_of_coyoteLayer_recall`, `card_le_of_coyoteLayer_argmax`, `exists_coyoteLayer_recall` | The prompt is `Fin.snoc` of the `n` context tokens and the query token; the layer is read at `Fin.last n`. The encodings `ctx` and `qry` are arbitrary functions. The bounds are stated for `coyoteLayer`, with the causal convolution; `coyoteLayerCyclic_last` rewrites the layer with the circular one into it. The layer that recalls has `n = d = m` and keys `Fin m`, and its context lists the values in the order of the keys. |
| AR7 | `hasCutWidth_retNet`, `retNet_recall_error`, `card_le_of_retNet_recall`, `card_le_of_retNet_argmax` | The context is a list of projected pairs `(A[n,:], V[n,:])`, the query appends its own pair, and `out q` is `C[N,:]`. |
| AR8 | `hasCutWidth_linearAttention`, `linearAttention_recall_error`, `card_le_of_linearAttention_recall`, `card_le_of_linearAttention_argmax`, `hasCutWidth_biasedAttention`, `biasedAttention_recall_error`, `card_le_of_biasedAttention_recall`, `card_le_of_biasedAttention_argmax` | `qf q` is `Q q`, `kf x j` is `K x_j`, `vf x j` a linear readout of `V x_j`, and `self q` the term of the query attending to itself. The projections are arbitrary functions of the query and of the context. `bias j` is the entry of `B` at the query and the context token `j`; it does not depend on the query. |
| AR9 | `mul_sub_card_le_fitLoss_recallKernel`, `exists_nuclearNorm_add_le_recallKernel`, `exists_pos_ae_tendsto_decayRun_recallKernel` | The runs and the constants are those of WD10; the width is `card κ ≤ card k`. |

## Softmax attention escapes the cut (added)

The transformers of Yang, Cadilhac and Chiang, *Knee-Deep in C-RASP: A Transformer Depth
Hierarchy*, arXiv:2506.16055v3, use softmax attention (§2.1):

```
sumdiv(a, b) = (Σ_j a_j) / (Σ_j b_j),      att(q, K, v) = sumdiv((exp q Kᵀ) ∘ v, exp q Kᵀ).
```

AR8 bounds attention without the exponential, and AR10 left softmax open. This section
treats one layer of it on recall. Softmax attention is not a readout of bounded cut width: it
recalls at projection dimension two. The price is the scale of its scores. Attention whose
scores ignore the query, the kind that the paper builds from its formulas, stays within the
cut.

### SA1. A leading score

Over `n` tokens with scores `s_j` and values `v_j`, `att = Σ_j e^{s_j} v_j / Σ_j e^{s_j}`.
If `s_j + g ≤ s_{j₀}` for every `j ≠ j₀` and `|v_j| ≤ B` for every `j`, then

```
|att − v_{j₀}| ≤ 2 B (n − 1) e^{−g}.
```

Proof. `att − v_{j₀} = Σ_{j ≠ j₀} e^{s_j} (v_j − v_{j₀}) / Σ_j e^{s_j}`. Each term of the
numerator is at most `e^{s_{j₀}} e^{−g} · 2B` in absolute value, and the denominator is at
least `e^{s_{j₀}}`. ∎

### SA2. Recall at dimension two

Keys `0, …, m − 1`. The token `j` of the context carries the key `j` and the value `φ(π j)`.
Project the query `q` to `β (q, 1)` and the key `j` to `(j, −j²/2)`, both of dimension two.
The score is

```
s_j = β (q j − j²/2) = β (q² − (q − j)²) / 2,
```

largest at `j = q` and smaller by at least `β/2` elsewhere. By SA1, for `|φ| ≤ B`,

```
|att − φ(π q)| ≤ 2 B (m − 1) e^{−β/2}.
```

- For every `m` and every `ε > 0`, some projections of dimension two recall to accuracy `ε`.
- For values `±1` the sign of the attention is the bound value, for every `m`.
- Under the future mask of the paper the query attends to its own token too. Give that token
  the key projection `(0, −1/2)`: its score is `−β/2`, and the error is at most
  `2 B m e^{−β/2}`, whatever value that token carries.

Without the exponential the same scores give a readout of cut width two, which recalls at
most two bindings (AR8). So for one layer in exact arithmetic the second half of Claim 1 of
the Zoology paper, "attention models can solve it with near-constant dimensionality", holds
with constant dimension; its first half for one layer is AR6.

### SA3. The price is the scale of the scores

In SA2 the scale `β` grows with `m`. It must. Let the scores of `n` tokens be bounded,
`|s_j| ≤ B`, and let `ρ = (n − 1) / (e^{2B} + n − 1)`. Each of the other weights is at least
`e^{−B}` and the weight of a token `j₀` is at most `e^{B}`, so the other tokens hold at least
the fraction `ρ` of the sum of the weights. With `a` the average of the other tokens under
their weights,

```
att = w v_{j₀} + (1 − w) a,   1 − w ≥ ρ,      so      |v_{j₀} − att| ≥ ρ |v_{j₀} − a|.
```

On recall take a token `(k, π k)` for each key, with a score `s(q, k, π k)` that is any
function of the query, of the key of the token and of its value, bounded by `B`, and with
the value `φ(π k)`. Changing `π q` changes neither the weights nor the values of the other
tokens. So `a` does not depend on `π q`, and the error at each value `v` of `π q` is at least
`ρ |φ(v) − a|`. The sum of `(φ(v) − a)²` over `v` is smallest at the mean `μ` of `φ`. Summing
over `v`, over the values of the other keys and over `q`, with `ρ = (m − 1) / (e^{2B} + m − 1)`:

```
Σ_{q, π} (φ(π q) − att)²  ≥  m c^{m−1} ρ² Σ_v (φ(v) − μ)².
```

The mean squared error is at least `ρ² Var φ`. For values `±1` recalled to accuracy `ε` at
every pair, `ρ ≤ ε`:

```
(m − 1)(1 − ε) ≤ ε e^{2B},      B ≥ ½ log((m − 1)(1 − ε) / ε).
```

One-hot keys reach this scale. With the score `β/2` at the queried key and `−β/2` elsewhere,
`B = β/2` and by SA1 the error is at most `2 (m − 1) e^{−β}` for values `±1`, so
`e^{2B} = 2 (m − 1) / ε` suffices: the bound is sharp up to the factor `2 / (1 − ε)`. The
scores of SA2 are of order `β m²`: dimension two is paid by a larger scale.

The transformers of the paper round to fixed precision outside `sumdiv` (Appendix
`app:transformers`, Equation `eq:att`): their weights are `round(exp s)`, which vanish below
the last bit. That is how a layer of width 5 with `O(log c)` bits recalls there
(`Transformer.CRASP.matcher_answers`). SA3 is for the exact exponential, whose weights never
vanish.

### SA4. Scores that ignore the query

The transformer that Proposition `thm:TLCl_to_rtfr` of the paper builds from a formula of
`TL[◁#]` uses uniform attention: no score depends on the query
(`Transformer.CRASP.RTfr.queryFree_model`). At the query, such an attention over the context
`x` and the query's own token is

```
att_r = (N_r(x) + α_r(q)) / (D(x) + δ(q)),      D(x) + δ(q) > 0,
```

with the sums `N_r(x) = Σ_j e^{s_j} v_{jr}` and `D(x) = Σ_j e^{s_j}` over the context. A
readout `g(q) + Σ_{r<d} w_r(q) att_r` has the sign of

```
g(q) (D(x) + δ(q)) + Σ_r w_r(q) (N_r(x) + α_r(q)),
```

a readout of cut width `d + 1`: the `d` sums and the normalization cross the cut. By AR5,
read by its sign it recalls at most `d + 1` bindings. So the exponential leaves the cut only
through scores that pair the query with the key.

`Transformer.CRASP.two_pow_le_of_queryFree` counts bits, for the rounded transformer and
every readout. SA4 counts numbers, for exact arithmetic and a linear readout.

### SA5. Scope

Proved: SA1 to SA4 as stated.

Not covered:

- SA2 places a binding in one token. A context that lists the key and the value in separate
  tokens needs a layer that brings them together first; no such stack is constructed here.
- SA3 takes the value of a token to be the readout of its bound value, and its score to be a
  function of the query and of that token. Scores that an earlier layer computes from several
  tokens are not bounded.
- SA4 is for a linear readout of one layer, read by its sign. A regression bound for it and a
  nonlinear readout after it are not proved.

### Lean

`src/Transformer/Grokking/FlatSpectrum/Recall/`, modules `Softmax`, `SoftmaxRecall`,
`SoftmaxScale` and `QueryFree`, namespace `Transformer.Grokking.FlatSpectrum.Recall`, proved
without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| SA1 | `softmaxAttention`, `abs_softmaxAttention_sub_le` | The tokens are any finite type; scores and values are real. |
| SA2 | `queryProj`, `keyProj`, `selfProj`, `sum_queryProj_mul_keyProj`, `sum_queryProj_mul_keyProj_add_le`, `softmaxAttention_recall_error`, `softmaxAttention_self_recall_error`, `exists_softmaxAttention_recall`, `exists_softmaxAttention_recall_sign` | The keys are `Fin m` and the token `j` carries the key `j`. The existence statements take `B = Σ_v |φ(v)|` and `β = 4 B m / ε + 2`, from `e^{−x} ≤ 1/x`; they do not claim a logarithmic scale. The query's own token is `none : Option (Fin m)`. |
| SA3 | `mul_sum_exp_le_sum_erase`, `mul_abs_sub_le_abs_sub_softmaxAttention`, `sum_sum_update`, `exists_forall_mul_abs_sub_le_softmaxAttention`, `mul_le_sum_softmaxAttention_sq`, `sub_one_mul_le_of_softmaxAttention_recall`, `softmaxAttention_oneHot_recall_error` | The average `a` of the other tokens is a quotient of sums over `univ.erase j₀`. Changing `π q` is `Function.update π q v`. The mean `μ` is any number with `Σ_v (φ v − μ) = 0`. The bound for accuracy `ε` is for values `Bool` read as `±1` and at least one key. |
| SA4 | `card_le_of_normalized_sign`, `softmaxAttention_option`, `card_le_of_queryFreeAttention_sign` | The context is a dictionary with values in `Bool`; its tokens are any finite type, with scores `s π j` and values `v π j r` that are any functions of the dictionary. The query's own token is `none`, with the score `σ q` and the values `u q r`. |

## The block languages across a cut (added)

The depth hierarchy of arXiv:2506.16055v3 is witnessed by one family of languages (§2.4,
Lemma `lem:piecewise_testable`): "For all `k ≥ 0`, define `L_k` to be the language of strings
with alternating *blocks* of `a`'s and `b`'s, starting with `a`", `L_k = (a⁺b⁺)^{k/2}` for
even `k` and `(a⁺b⁺)^{(k−1)/2} a⁺` for odd `k`. Theorem `thm:rtfr_depth_hierarchy` (§4): "A
depth-`(k+1)` fixed-precision transformer can recognize `L_{k+1}`, but no depth-`k`
fixed-precision transformer can", at every width and precision. A transformer accepts by a
sign: "We say that `T` accepts `w` if `T(w) > 0`" (Appendix `app:transformers`).

The recall section prices the width of a model by what crosses a cut between the context and
the query. This section asks the same of `L_k`. Cut a word into a prefix `u` and a suffix
`v`, and let a readout `F(v, u) = g(v) + Σ_{r<d} a_r(v) b_r(u)` of cut width `d` (AR4) tell
whether `u v ∈ L_k`. The answer separates two resources. By a sign, five numbers suffice for
every `k`. The indicator itself holds a flat kernel of size `⌊k/2⌋`, so its regression costs
width. The hierarchy of the paper is in depth, at bounded cut width.

### BL1. Blocks add across a cut

`blocks w` is the number of maximal blocks of equal letters of `w`. For a suffix `v` that is
not empty,

```
blocks(u v) + [last u = first v] = blocks u + blocks v :
```

the last block of `u` and the first block of `v` are one block of `u v` exactly when their
letters agree. Write `ℓ(u) = [u ends with b]` and `f(v) = [v starts with b]`. Then
`[last u = first v] = 1 − ℓ − f + 2 ℓ f`, so for a prefix that is not empty

```
blocks(u v) = blocks u + shift v + ℓ(u) · slope v,
shift v = blocks v − 1 + f(v),      slope v = 1 − 2 f(v),
```

with `shift v = slope v = 0` for the empty suffix. The suffix acts on two numbers of the
prefix: its number of blocks and its last letter.

### BL2. `L_k` is a first letter and a number of blocks

A word that is not empty is in `L_k` exactly when it starts with `a` and has `k` blocks.
`L_0` holds the empty word alone.

Proof. A word that starts with `s` is a run of `s` and then a word that does not start with
`s`, and the run is one block. Induction on `k` along `L_{k+1}(s) = s⁺ L_k(¬s)`. ∎

### BL3. Five numbers decide every `L_k` by a sign

Let `A = [u starts with a]`, `P = blocks u`, `ℓ = ℓ(u)`, and for the suffix `G = shift v − k`,
`H = slope v`. By BL1 and BL2, for a prefix that is not empty, `u v ∈ L_k` exactly when
`A = 1` and the integer `ρ = P + G + ℓ H` is `0`, that is, exactly when

```
A (3/2 − ρ²) − 1 > 0.
```

As `ℓ² = ℓ`,

```
A (3/2 − ρ²) − 1 = −1 + (3/2 − G²) A − A P² − (H² + 2 G H) A ℓ − 2 G A P − 2 H A P ℓ.
```

The five numbers `A, A P², A ℓ, A P, A P ℓ` of the prefix cross the cut, and the suffix
combines them linearly. For the empty prefix take the five numbers `1, 0, 3/4, 1/4, 1/8`. The
readout is then `1/2 − (blocks v − k)²` on a suffix that starts with `a`, `G (1 − G) ≤ 0` with
the integer `G = blocks v − k` on one that starts with `b`, and `1/2 − k² + k/2` on the empty
suffix: positive exactly when `v ∈ L_k`. So

- for every `k`, a readout of cut width 5 is positive exactly on the pairs `(u, v)` with
  `u v ∈ L_k`, at every cut of every word.

The count does not grow with `k`. With the empty suffix the readout tells whether the prefix
`u` is in `L_k`, which is the answer that the next-token prediction problem of the paper
(`def:prediction_task`) asks for at the last position of `u`.

### BL4. A flat kernel inside a task prices the cut width

AR4 is stated for recall. Its mechanism is general. Let `F` have cut width `d`, choose queries
`q_i`, `i ∈ m`, and contexts `x_j`, `j ∈ n`, and let `K` be a real flat kernel on `m × n`
with squared singular value `c > 0`. The table `F(q_i, x_j)` is a product through width
`d + 1`, the term `g` being one more column. By R3,

```
Σ_{i,j} (K_ij − F(q_i, x_j))²  ≥  c (min(|m|, |n|) − (d + 1)),
```

and `d` replaces `d + 1` when the rows of `K` sum to zero, by the centering of AR4. A readout
that equals `K` on the chosen pairs has `min(|m|, |n|) ≤ d + 1`, and `≤ d` for rows of sum
zero. AR4 is the case of all keys, all dictionaries and the recall kernel of `φ − μ`.

The indicator of `L_k` holds such a kernel. The prefix `u_i = (ab)^i a` has `2i + 1` blocks
and ends with `a`. Let `v_j` be the alternating word that starts with `b` and has
`k − (2j + 1)` blocks, for `2j + 1 < k`. No blocks join at the cut, so by BL1 and BL2

```
u_i v_j ∈ L_k    ⟺    i = j.
```

On these `⌊k/2⌋` prefixes and suffixes the indicator is the identity. Hence for a readout `F`
of cut width `d`:

- `Σ_{i,j<⌊k/2⌋} (1[u_i v_j ∈ L_k] − F(v_j, u_i))² ≥ ⌊k/2⌋ − (d + 1)`;
- if `F(v, u) = 1[u v ∈ L_k]` for all `u` and `v`, then `d ≥ ⌊k/2⌋ − 1`.

Regression of the indicator costs a width that grows with `k`. Its sign does not (BL3).

### BL5. Depth is not cut width

For every `k`: no transformer of depth `k`, of any width and precision, recognizes `L_{k+1}`
(Theorem `thm:rtfr_depth_hierarchy`, `Transformer.CRASP.rtfr_depth_hierarchy`), and a readout
of cut width 5 decides `L_{k+1}` by its sign at every cut (BL3).

The five numbers are functions of the prefix: of its first letter, its last letter and its
number of blocks. Cut width prices what crosses the cut. It does not price computing those
functions. The hierarchy of the paper is the cost of computing the number of blocks by
counting without positions, and a flat spectrum does not see it. This is the sentence of
*Depth, width and precision* above, "A flat spectrum alone does not supply either
conclusion", for the paper's own languages.

### BL6. Scope

Proved: BL1 to BL5 as stated.

Not covered:

- The exact cut width of the indicator of `L_k`. BL4 gives `d ≥ ⌊k/2⌋ − 1`. On the classes of
  prefixes and suffixes the kernel of the indicator is bidiagonal, not flat; its rank and its
  spectrum are not computed here.
- A lower bound on the cut width of `L_k` by a sign. BL3 is an upper bound, 5.
- Spectra. No task kernel of the paper is shown to be flat, and Theorem A is not applied to
  its languages.
- The cut is one. A transformer reads the prefix at every later position; nothing here bounds
  a model by its width across several cuts at once.

### Lean

`src/Transformer/Grokking/FlatSpectrum/BlockCut/`, namespace
`Transformer.Grokking.FlatSpectrum.BlockCut`, and
`src/Transformer/Grokking/FlatSpectrum/Recall/Subkernel.lean`, proved without `sorry`. `L_k`
is `Transformer.CRASP.altPlus false k`, over the alphabet `Bool` with `false` for `a`.

| Item | Lean | Differences |
| --- | --- | --- |
| BL1 | `blocks`, `blocks_cons`, `blocks_pos`, `blocks_eq_zero`, `blocks_append`, `blocks_replicate_append`, `exists_replicate_append`, `lastB`, `shift`, `slope`, `cast_blocks_append` | `[last u = first v]` is `u.getLast? = v.head?`. The formula in numbers is over `ℝ`. |
| BL2 | `head?_of_mem_altPlus`, `mem_altPlus_succ_iff`, `mem_altPlus_iff`, `blocks_altList`, `getLast?_altList_odd` | Stated for both first letters `s`; `mem_altPlus_succ_iff` is for `k + 1` and every word. |
| BL3 | `headA`, `prefixFeature`, `suffixCoeff`, `blockReadout`, `hasCutWidth_blockReadout`, `one_le_sq_sub`, `sub_mul_one_sub_nonpos`, `blockReadout_of_ne_nil`, `blockReadout_pos_iff_of_ne_nil`, `blockReadout_nil_pos_iff`, `blockReadout_pos_iff`, `exists_hasCutWidth_altPlus_sign` | The query is the suffix and the context is the prefix: `F v u`, with `HasCutWidth 5 F` on `List Bool × List Bool`. |
| BL4 | `Recall.HasCutWidth.mul_le_sum_sq_of_isFlat`, `Recall.HasCutWidth.mul_le_sum_sq_of_isFlat_of_sum_eq_zero`, `Recall.HasCutWidth.card_le_of_isFlat`, `Recall.HasCutWidth.card_le_of_isFlat_of_sum_eq_zero`, `oddPrefix`, `evenSuffix`, `oddPrefix_append_mem_iff`, `div_two_sub_le_sum_sq_altPlus`, `div_two_le_of_altPlus` | `K` is a real matrix that is flat as a complex one. `⌊k/2⌋` is `k / 2` in `ℕ`, the pairs are indexed by `Fin (k / 2)`, and the indicator is `Set.indicator (altPlus false k) 1`. |
| BL5 | `not_recognizes_and_hasCutWidth_five` | The transformers are `Transformer.CRASP.RTfr (Option Bool) p s d k`, for all `p`, `s`, `d`. |

## Polynomial readouts and two layers (added)

AR10 leaves two readouts open: "A stack in which two layers read the context at the query",
the setting of Claim 1 of arXiv:2312.04927v1 (§4: "Gated-convolution models with two layers
require model dimension to scale at least linearly in sequence length in order to solve
associative recall"), and "An MLP after the layer". In both the query combines the numbers
that cross the cut by a polynomial, not by a linear form. This section bounds such readouts.
The bound is again a flat spectrum, that of the kernel of all parities of the bindings.

### PR1. The parity kernel

Take two values, read as `±1`. For a set `S` of keys and a dictionary `β`, the parity is
`χ_S(β) = Π_{q∈S} ±1(β q)`. The parity kernel `P(S, β) = χ_S(β)` has a row for each of the
`2^m` sets of keys and a column for each of the `2^m` dictionaries. Its rows are the
characters of `(ℤ/2)^m`:

```
Σ_β χ_S(β) χ_T(β) = 2^m [S = T],        P Pᴴ = 2^m · 1.
```

So `P` is flat of rank `2^m`, and by Theorem A (R2) it is a minimum of M from no proper set
of its entries. The recall kernel of AR1 for values `±1` is its `m` rows of the single keys.

### PR2. Polynomial readouts

`D` numbers `z(x)` cross from the context, and the readout is `F(q, x) = P_q(z(x))` with
`P_q` a polynomial of total degree at most `Δ` whose coefficients depend on the query. Sums
keep the degree and products add the degrees. Degree one is a readout of cut width `D` with
features `z` (AR4). A polynomial of degree at most `Δ` is a combination of the `C(D + Δ, D)`
monomials of degree at most `Δ`, so `F` has cut width at most `C(D + Δ, D)`, and AR4 and AR5
hold with this width. At degree two, written
`g(q) + Σ_i a_i(q) z_i + Σ_{i,j} A_{ij}(q) z_i z_j`, the cut width is at most `D + D²`. The
product of two readouts of degree one is of degree two.

### PR3. Exact recall: the count is linear

Let `F` of degree at most `Δ` in `D` numbers return `±1` of the bound value for every query
and every dictionary of `m` keys. For a set `S` of keys

```
χ_S(β) = Π_{q∈S} F(q, β) = (Π_{q∈S} P_q)(z(β)),
```

a polynomial of degree at most `Δ m` in `z(β)`. So every row of the parity kernel is a
combination of the `C(D + Δ m, D)` monomials of degree at most `Δ m` at `z(β)`: the kernel is
a product through that width. It is flat of rank `2^m` (PR1), and an exact factorization of a
flat kernel has width at least its rank (R3):

```
2^m ≤ C(D + Δ m, D).
```

A readout `φ` that is not constant gives the same bound: restrict the dictionaries to two
values with different `φ` and rescale. For `Δ = D = m = 1` it is an equality.

The count is linear in `m`. One term of the binomial expansion is
`C(D + N, D) t^N ≤ (t + 1)^{D + N}`. With `N = Δ m` and `t = 2Δ`, after squaring, and with
`(t + 1)^t ≤ 3 t^t` from `(1 + 1/t)^t ≤ e`:

```
(4/3)^m ≤ (2Δ + 1)^{2D},        m ≤ 2 D log(2Δ + 1) / log(4/3).
```

At degree two, `t = 8` gives `128^m ≤ 9^D 81^m`, and `2^35 > 3^22` gives

```
m ≤ 5 D.
```

### PR4. An MLP with a polynomial activation

A hidden layer `Σ_h u_h (Σ_j V_{hj} Y_j + b_h)^p` over readouts `Y_j` of degree one in `z` is
a readout of degree `p`, for every number of hidden units. By PR3 its exact recall has
`(4/3)^m ≤ (2p + 1)^{2D}`: the hidden width does not enter, and the degree enters by its
logarithm. By PR2 its cut width is at most `C(D + p, D)`.

### PR5. Two gated convolutions

Stack two Coyote layers (AR6) on the prompt of `n` context tokens and the query token `e`,
and read the second at the query, the position `N`. A layer is causal: before the query its
output is
that of the context alone, so the output of a layer on a prompt is a prompt. At the query a
layer is

```
y_c = (Σ_k W_{kc} e_k + b₁) (B_c + h[0]_c e_c + b₂),      B_c = Σ_{k≥1} h[k]_c u_{N−k,c}.
```

So the first layer hands the query `e'`, affine in the `d` numbers `B` of the context. The
second layer returns `(Σ_k W'_{kc} e'_k + b'₁)(G_c + h'[0]_c e'_c + b'₂)`, where `G` is its
convolution of the outputs of the first layer before the query, `d` more numbers of the
context. A linear readout of it is of degree two in the `2d` numbers `(B, G)`, with
coefficients that depend on the query. So for `m` keys, for every encoding of the context
and of the query:

- exact recall of a `φ` that is not constant has `m ≤ 10 d` (PR3);
- dimension `d = m` recalls `m` bindings, by the layer of AR6 and a second layer that
  returns its input, so the least dimension of exact recall is between `m / 10` and `m`;
- the cut width is at most `2d + 4d²` (PR2), so the squared error on recall is at least
  `(m − 2d − 4d²) c^{m−1} Σ_v (φ(v) − μ)²` (AR4), and a right argmax has `m ≤ 2d + 4d²`
  (AR5).

The first item is Claim 1 for exact recall by two layers and a linear readout: the dimension
is at least linear in the number of bindings.

### PR6. Scope

Proved: PR1 to PR5 as stated.

Not covered:

- A sign, an argmax or a squared error at dimension linear in `m`. For two layers these
  bounds are `m ≤ 2d + 4d²`. PR3 multiplies the values of the readout, and the product is a
  parity only for values that are exactly `±1`. A linear bound for the sign patterns of
  polynomials is not formalized here.
- Activations that are not polynomials, such as the MLPs of the paper, and an MLP between
  the two layers.
- More than two layers.
- Whether `m ≤ 5 D` and `m ≤ 10 d` hold with a smaller constant. At degree one the constant
  is 1 (AR4).

### Lean

`src/Transformer/Grokking/FlatSpectrum/Recall/`, modules `Parity`, `Monomials`, `Polynomial`,
`PolynomialCount`, `Quadratic`, `TwoLayers` and `TwoLayersRecall`, namespace
`Transformer.Grokking.FlatSpectrum.Recall`, proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| PR1 | `parityKernel`, `sum_prod_mul_prod`, `parityKernel_mul_conjTranspose`, `isFlat_parityKernel`, `rank_parityKernel`, `not_isMinOn_parityKernel` | The rows are `Finset κ` and the columns `κ → Bool`, with `true` read as `1`. The kernel is complex, as `IsFlat` asks. |
| PR2 | `IsPolyIn`, `HasPolyCut`, `isPolyIn_query`, `isPolyIn_coord`, `IsPolyIn.mono`, `IsPolyIn.add`, `IsPolyIn.mul`, `IsPolyIn.sum`, `IsPolyIn.pow`, `IsPolyIn.comp_context`, `hasCutWidth_of_fintype`, `IsPolyIn.hasCutWidth`, `IsAffineIn`, `IsQuadIn`, `isAffineIn_query`, `isAffineIn_coord`, `IsAffineIn.hasCutWidth`, `IsAffineIn.add`, `IsAffineIn.smul`, `IsAffineIn.sum`, `affine_mul_affine`, `IsAffineIn.mul`, `IsQuadIn.sum`, `IsQuadIn.hasCutWidth`, `IsAffineIn.isPolyIn`, `IsQuadIn.isPolyIn` | `P_q` is an `MvPolynomial (Fin D) ℝ` with `totalDegree ≤ Δ`. `IsPolyIn z Δ F` fixes the numbers `z`; `HasPolyCut D Δ F` asks for some. `D + D²` is `D + D * D`: the products `z_i z_j` are indexed by ordered pairs. |
| PR3 | `monomialExponents`, `card_monomialExponents`, `eval_eq_sum_monomialExponents`, `choose_mul_pow_le`, `succ_pow_le_three_mul_pow`, `le_five_mul_of_pow_le`, `le_five_mul_of_le_choose`, `four_pow_le_of_le_choose`, `IsPolyIn.two_pow_le_choose_of_sign`, `HasPolyCut.two_pow_le_choose_of_recall`, `HasPolyCut.four_pow_le_of_recall`, `HasPolyCut.card_le_five_mul_of_recall`, `hasPolyCut_one_key` | A monomial of degree at most `N` is a tuple of `D + 1` naturals with sum `N`, the last one the slack. `(4/3)^m ≤ (2Δ + 1)^{2D}` is `4 ^ m ≤ 3 ^ m * (2 * Δ + 1) ^ (2 * D)` in `ℕ`; its form with logarithms is not stated. |
| PR4 | `IsPolyIn.powerHead` | `H` hidden units over `r` readouts `Y_j` of degree one. |
| PR5 | `prompt_of_ne_last`, `causalConvolution_prompt_zero_last`, `coyoteLayer_prompt_last`, `coyoteLayer_prompt_castSucc`, `coyoteLayer_prompt`, `firstCross`, `secondCross`, `twoLayerCross`, `coyoteLayer_coyoteLayer_prompt_last`, `isQuadIn_twoCoyoteLayers`, `hasCutWidth_twoCoyoteLayers`, `card_le_of_twoCoyoteLayers_recall`, `twoCoyoteLayers_recall_error`, `card_le_of_twoCoyoteLayers_argmax`, `coyoteLayer_id`, `exists_twoCoyoteLayers_recall` | The two layers have the same dimension `d` and length `n + 1`, with the causal convolution. The `2d` numbers are `Fin (d + d) → ℝ`, and `2d + 4d²` is `d + d + (d + d) * (d + d)`. The layers that recall have `n = d = m`. |

## Families of dictionaries and the multi-query task (added)

AR4 to AR9 range over every dictionary of the `m` keys, `c^m` contexts. AR10 leaves open a
model that meets fewer of them, and the multi-query task of arXiv:2312.04927v1. This section
answers both for lower bounds. A family in which every two keys are independent is as hard
as all dictionaries, and recall with one query is a special case of the multi-query task.

### FD1. Pairwise uniform families

A family `D_i`, `i ∈ ι`, of dictionaries is pairwise uniform of index `λ` when every key
takes every value on `λ c` of them and every two different keys take every pair of values on
`λ` of them: an orthogonal array of strength two. For such a family and all functions `f`,
`g` of a value

```
Σ_i f(D_i q) = λ c Σ_v f(v),      Σ_i f(D_i q) g(D_i q') = λ (Σ_v f(v)) (Σ_v g(v))    (q ≠ q').
```

The four dictionaries of two keys with two values are such a family, with `λ = 1`.

### FD2. The kernel of the family is flat

For a centered readout the kernel `K(q, i) = φ(D_i q)`, with `m` rows and `|ι|` columns, has
by FD1

```
K Kᴴ = λ c Σ_v φ(v)² · 1.
```

So `K` is flat. For `λ ≥ 1` and a `φ` without zeros it is, by Theorem A (R2), a minimum of M
from no proper set of its entries.

### FD3. Width on the family

By BL4 for rows of sum zero, a readout `F` of cut width `d` has, with `μ` the mean of `φ`,

```
Σ_q Σ_i (φ(D_i q) − F(q, D_i))² ≥ λ c Σ_v (φ(v) − μ)² (min(m, |ι|) − d),
```

and for `λ ≥ 1` exact recall on the family of a `φ` that is not constant has
`d ≥ min(m, |ι|)`.

### FD4. The affine dictionaries

Let the keys and the values be a field of `p` elements, and `D_{a,b}(q) = a q + b`. Through
two points with different keys passes one line, so these `p²` dictionaries are pairwise
uniform of index 1. A readout of cut width `d` misses recall on them by at least
`p Σ_v (φ(v) − μ)² (p − d)`, over `p · p²` pairs: per pair the fraction `1 − d / p` of the
variance of `φ`, as in AR4 on all `p^p` dictionaries. Exact recall on them has `d ≥ p`.

Uniformity on pairs of keys is needed. The constant dictionaries are recalled at cut width
one for every number of keys: one number crosses, the readout of the common value.

### FD5. The multi-query task

The general task of the paper (Appendix `sec: intro-general-ar`, Setup) reads triples
`(k_i, v_i, q_i)` over a vocabulary of `c` tokens: "Our goal is then to check, for each
`1 ≤ i ≤ N/3 − 1`, whether there exists `0 ≤ j < i` such that `q_i ≡ k_j`, and if so, output
`v_j`." A dictionary `π` of `m < c` keys and a query `q` are an input of `m + 1` triples: the
triple `j < m` is `(j, π j, j)` and the last one is `(m, m, q)`. Its keys are distinct. The
first `m` triples depend on `π` alone and ask for keys that no earlier triple binds, the last
triple depends on `q` alone, and the answer at the last triple is `π q`.

So a model that answers the last query of every input with distinct keys recalls every
dictionary, and AR4 applies to its answer at the last triple. If that answer has cut width
`d` across the cut before the last triple, its squared error over these inputs is at least
`(m − d) c^{m−1} Σ_v (φ(v) − μ)²`, and exact answers have `m ≤ d`. Repeated keys and more
queries add inputs: a bound on these inputs is a bound on all.

### FD6. Scope

Proved: FD1 to FD5 as stated.

Not covered:

- A sign or an argmax on a family. AR5 uses every dictionary with two values; on a family the
  signs of the kernel may have a smaller rank, and no bound is proved.
- Families that are not pairwise uniform, apart from the two extremes of FD4. Which families
  of `p²` dictionaries exist for keys and values of different numbers is the existence of
  orthogonal arrays, not treated here.
- Upper bounds for the multi-query task. FD5 transfers lower bounds only; a model must also
  answer the queries at earlier positions and decline those without a match.

### Lean

`src/Transformer/Grokking/FlatSpectrum/Recall/`, modules `Family`, `AffineFamily` and
`MultiQuery`, namespace `Transformer.Grokking.FlatSpectrum.Recall`, proved without `sorry`.
The task is that of `Transformer.Zoology` (`MQARInstance`, `PriorAnswer`, `UniqueKeys`).

| Item | Lean | Differences |
| --- | --- | --- |
| FD1 | `IsPairwiseUniform`, `IsPairwiseUniform.sum_comp`, `IsPairwiseUniform.sum_comp_mul_comp`, `isPairwiseUniform_two` | The family is `D : ι → κ → ν` over a finite type `ι`, and the counts are cardinalities of filters. Both conditions are fields: the first follows from the second only for two or more keys. |
| FD2 | `familyKernel`, `IsPairwiseUniform.sum_mul`, `isFlat_familyKernel`, `not_isMinOn_familyKernel`, `signValue` | The kernel is real, and it is flat as a complex matrix. "Centered" is `Σ_v φ v = 0`. |
| FD3 | `HasCutWidth.mul_le_sum_sq_of_isPairwiseUniform`, `HasCutWidth.card_le_of_recall_of_isPairwiseUniform` | The readout is `F : κ → (κ → ν) → ℝ`, evaluated at `D i`. `min(m, |ι|)` is kept: `m ≤ |ι|` is not derived. |
| FD4 | `affineDict`, `isPairwiseUniform_affineDict`, `HasCutWidth.mul_le_sum_sq_affineDict`, `HasCutWidth.card_le_of_recall_affineDict`, `exists_hasCutWidth_one_recall_const` | The field is any finite field `𝔽`, the family is indexed by `𝔽 × 𝔽`, and `p` is `Fintype.card 𝔽`. |
| FD5 | `mqarOfDictionary`, `castLE_ne_mk`, `uniqueKeys_mqarOfDictionary`, `priorAnswer_mqarOfDictionary`, `not_priorAnswer_mqarOfDictionary`, `card_le_of_mqar`, `mqar_recall_error`, `hasCutWidth_firstValue` | Tokens are `Fin c`, the key `j` is `Fin.castLE` of `j : Fin m`, and the new key and value of the last triple are the token `m`. The model's answer at the last triple is any `G : MQARInstance (m + 1) c → ℝ`; "answers" is `G x = φ u` for the `u` with `PriorAnswer x (Fin.last m) u`, on inputs with `UniqueKeys`. |

## One unseen cell (added)

The last item of AR10 says: "AR3 and AR9 say that the recall kernel is not a minimum of M and
is not returned by training from a part of its table. They do not say what the minimum is or
how far its decoded answers are from recall." The Corollary of the note answers for a square
kernel with `K* K = N I`, `N ≥ 2`, and one missing entry of modulus one: the fill is `−K_ij`
(`isMinOn_flat_fill_iff`). The recall kernel is not square, and a readout need not have
modulus one. This section finds the minimum of M when one cell of a kernel with orthogonal
rows is unseen. It exists, and it fills the cell with `−θ K_ij` for a `θ > 0` bounded on both
sides.

### OC1. Two tools

The nuclear norm of the adjoint is that of the matrix, `‖Xᴴ‖_S1 = ‖X‖_S1`: pair `X` with the
adjoint of the contraction that attains `‖Xᴴ‖_S1`.

Let `Xᴴ X = c 1 + U M Uᴴ` with `c ≥ 0`, where `U` has two orthogonal columns of squared
lengths `d₀, d₁ > 0` and `M` is a `2 × 2` block. With `B = M + c diag(1/d₀, 1/d₁)` and `n`
the number of columns of `X`,

```
‖X‖_S1 = √c (n − 2) + √(Re tr(B diag d) + 2 √(|det B| d₀ d₁)).
```

Proof: `P = U diag(1/d) Uᴴ` is the projection onto the span of `U`, and
`Xᴴ X = c (1 − P) + U B Uᴴ` with `tr (1 − P) = n − 2`. This is the form of the block formula
in the row *the centered indicator of 0* of the Lean status
(`nuclearNorm_of_gram_eq_proj_add_diagonal`).

### OC2. The Gram matrix of a changed cell

Let `K Kᴴ = c 1`, `κ = K_ij`, and change the cell `(i, j)` by `t`: `X = K + t e_i e_jᵀ`. Write
`k_j` for the column `j` of `K` and `u = k_j − κ e_i` for that column outside the row `i`, of
squared length `g = Σ_{p ≠ i} |K_pj|²`. Then

```
X Xᴴ = c 1 + t e_i k_jᴴ + t̄ k_j e_iᴴ + |t|² e_i e_iᴴ
     = c 1 + U [[t κ̄ + t̄ κ + |t|², t], [t̄, 0]] Uᴴ,        U = [e_i, u],
```

and `Uᴴ U = diag(1, g)`.

### OC3. The norm

For `g > 0`, OC1 for `Xᴴ` and OC2 give, with `τ = t κ̄ + t̄ κ + |t|² = |κ + t|² − |κ|²` and `m`
the number of rows,

```
‖K + t e_i e_jᵀ‖_S1 = √c (m − 2) + √(2c + τ + 2 √|c (c + τ) − g |t|²|).
```

Here `B = [[τ + c, t], [t̄, c/g]]`, `tr(B diag(1, g)) = 2c + τ` and
`g det B = c (c + τ) − g |t|²`.

The column `j` of `K` has squared length at most `c`, as `K / √c` is a contraction:
`|κ|² + g ≤ c`.

### OC4. The value of the cell

Write `x = κ + t` for the value of the cell, `s = |κ| > 0`, and

```
N(x) = √(2c + |x|² − s² + 2 √|c (c + |x|² − s²) − g |x − κ|²|),
```

so that the norm of OC3 is `√c (m − 2) + N(x)`. Let `g > 0` and `s² + g ≤ c`.

`N` is continuous and `N(x) ≥ |x|`, so some `x` minimizes it. Every minimizing `x` is

```
x = −θ κ      with      g / (2c − g) ≤ θ ≤ g / (c − g).
```

Proof. On the ray `x = −(r/s) κ`, `r ≥ 0`, opposite to `κ`: `|x| = r`, `|x − κ| = r + s`, and
the inner square root is of `Δ(r) = c (c + r² − s²) − g (r + s)²`. This is not negative:
`(c − g) Δ(r) = c² (c − g − s²) + ((c − g) r − g s)²` and `c − g ≥ s² > 0`. So on the ray
`N² = 2c − s² + r² + 2 √Δ(r)`.

- Off the closed ray, `|x − κ| < |x| + s`, since `ℂ` is strictly convex. Then
  `|c (c + |x|² − s²) − g |x − κ|²| > Δ(|x|)`, and the point of the ray with the modulus `|x|`
  has a smaller `N`.
- Near zero `N` decreases along the ray. With `w = √Δ(0)`, `0 < w ≤ c`,
  `Δ(r) = w² + ((c − g) r − 2 g s) r`. Let `r₁ = g s / (w + c − g)` and `0 ≤ r < r₁`. Then
  `√Δ(r) ≤ w`, `√Δ(r₁) ≤ w` and
  `Δ(r) − Δ(r₁) = (r₁ − r)(2 g s − (c − g)(r + r₁)) > (r₁ − r)(r₁ + r) w`, so
  `2 (√Δ(r) − √Δ(r₁)) > r₁² − r²`. As `w ≤ c`, `r₁ ≥ g s / (2c − g)`: a minimizing modulus is
  at least `g s / (2c − g)`, and `x = 0` is not a minimum.
- Beyond `ρ = g s / (c − g)` it increases: `Δ(r) = Δ(ρ) + (c − g)(r − ρ)²`, so for `r > ρ`
  both `r²` and `Δ(r)` are larger than at `ρ`.

### OC5. One unseen cell

Let `K Kᴴ = c 1` with `c > 0`, and let `(i, j)` be the only unseen cell, with `κ = K_ij ≠ 0`
and `g = Σ_{p ≠ i} |K_pj|² > 0`. A matrix equal to `K` at the other cells is
`K + (x − κ) e_i e_jᵀ` for its value `x` at the cell. So by OC3 and OC4:

- a minimum of M from the other cells exists;
- every minimum `X` has `X_ij = −θ κ` with `g / (2c − g) ≤ θ ≤ g / (c − g)`: the unseen cell
  is filled with the opposite phase.

The upper bound is attained. For the Hadamard matrix `[[1, 1], [1, −1]]` and the cell
`(0, 0)`: `c = 2`, `κ = 1`, `g = 1`. The fill `[[−1, 1], [1, −1]]` has norm `2`, and the swap
`[[0, 1], [1, 0]]` is a contraction that pairs every fill to `2`, so this fill is the minimum
and `θ = 1 = g / (c − g)`.

For a square kernel with `K* K = N I` and `|K_ij| = 1`, the case of the Corollary, `c = N`
and `g = N − 1`, and the Corollary has `θ = 1`: between the bounds `(N − 1) / (N + 1)` and
`N − 1` of this item.

By OC1 the adjoint carries the fills of `K` without `(i, j)` to the fills of `Kᴴ` without
`(j, i)`, and minima to minima. So for orthogonal columns, `Kᴴ K = c 1`, the same two
statements hold with `g' = Σ_{q ≠ j} |K_iq|²`, the rest of the row of the cell, in the place
of `g`. For a flat kernel of either shape with `κ ≠ 0`, `g > 0` and `g' > 0`: a minimum
exists, and every minimum fills the cell with `−θ κ`, `θ > 0`.

### OC6. Recall with one unseen pair

The recall kernel of a centered readout has `K Kᴴ = c 1` with `c = |ν|^{m−1} Σ_v |φ(v)|²`
(AR2). For the pair `(q, π)` its cell is `φ(π q)`, and `g = Σ_{p ≠ q} |φ(π p)|²` is the sum of
the squared readouts of the other keys of the dictionary: positive for a readout without
zeros and two keys or more. So, with every pair but `(q, π)` seen:

- a minimum of M exists;
- every minimum answers the unseen pair with `−θ φ(π q)`, `g / (2c − g) ≤ θ ≤ g / (c − g)`.

For a readout of modulus one, `g = m − 1` and `c = |ν|^m`:

```
(m − 1) / (2 |ν|^m − (m − 1)) ≤ θ ≤ (m − 1) / (|ν|^m − (m − 1)).
```

The answer at the unseen pair has the opposite phase, for a real readout the wrong sign, and
its modulus is of the order `m / |ν|^m` of the right one. For two keys with values `±1`,
`1/7 ≤ θ ≤ 1/3`.

### OC7. Scope

Proved: OC1 to OC6 as stated.

Not covered:

- More than one unseen cell. With `k` unseen cells the Gram matrix of OC2 changes on a block
  of size up to `2k`, and no closed form is derived. AR3 still says that recall is not the
  minimum. The numerical check of the note, "Many missing entries (ADMM): the mean
  fill/truth ratio is −0.16, −0.78, −0.87 and −0.96 at 50, 90, 95 and 99% seen", is for the
  square kernel of `x³ + x y` and is not a theorem.
- The value of `θ`. On the ray it minimizes `r² + 2 √Δ(r)`. OC4 bounds it on both sides,
  within the factor `2 + g / (c − g)`; neither bound is exact in general.
- `g = 0`: a cell that is alone in its column, or alone in its row for orthogonal columns.
  The Torch check row of the one-hot kernels treats a row whose seen entries are all zero.
- A decoded answer. With more than two values the answer is decoded from several readouts;
  OC6 is for one readout `φ`.
- Trained models. OC5 and OC6 are about the minimum of M, the object of the modeling step
  of Lemma 1. What a fit trained at a fixed weight decay (R4, AR9) answers at the unseen pair
  is not derived.

### Lean

`src/Transformer/Grokking/FlatSpectrum/`, modules `OneCell`, `OneCellNorm`, `OneCellRay`,
`OneCellPhase`, `OneCellMin` and `OneCellFlat`, namespace `Transformer.Grokking.FlatSpectrum`,
and `Recall/Unseen`, namespace `Transformer.Grokking.FlatSpectrum.Recall`; proved without
`sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| OC1 | `nuclearNorm_conjTranspose_le`, `nuclearNorm_conjTranspose`, `nuclearNorm_of_gram_eq_smul_one_add` | `d : Fin 2 → ℝ` with `Uᴴ U = diagonal d`; the trace term is the real part of `tr ((M + c • diagonal d⁻¹) * diagonal d)`; `n` is `Fintype.card n` as a real number. |
| OC2 | `mul_conjTranspose_smul_single_apply`, `smul_single_mul_conjTranspose_apply`, `cellFrame`, `restSq`, `cellFrame_gram`, `mul_conjTranspose_add_smul_single` | `e_i e_jᵀ` is `Matrix.single i j 1`. `restSq K i j` is `∑ p, if i = p then 0 else ‖K p j‖ ^ 2`, and `cellFrame K i j` has the columns `e_i` and `u`. |
| OC3 | `restSq_nonneg`, `corner_eq`, `nuclearNorm_add_smul_single`, `norm_sq_add_restSq`, `sum_norm_sq_le_of_mul_conjTranspose`, `norm_sq_add_restSq_le`, `hadamard_mul_conjTranspose`, `restSq_hadamard` | `τ` is written `‖K i j + t‖ ^ 2 - ‖K i j‖ ^ 2`. `m − 2` is a subtraction of real numbers; `g > 0` gives two rows. The formula asks for `c ≥ 0`, the bound on the column for `c > 0`. |
| OC4 | `ray_nonneg`, `ray_outer_nonneg`, `exists_ray_lt`, `cellNorm`, `cellNorm_ray`, `norm_le_cellNorm`, `exists_forall_cellNorm_le`, `le_mul_of_forall_cellNorm_le`, `mul_sub_le_of_forall_cellNorm_le`, `exists_neg_mul_of_forall_cellNorm_le` | The bounds are `g ≤ θ * (2 * c - g)` and `θ * (c - g) ≤ g`, with `0 < θ` stated as well. `exists_ray_lt` asks for `r * (2 * c - g) < g * s` and returns some `r₁ > 0`. Existence asks only for `‖κ‖ ^ 2 ≤ 2 * c`. |
| OC5 | `add_smul_single_apply_of_mem`, `eq_add_single_of_mem`, `nuclearNorm_eq_cellNorm`, `exists_isMinOn_compl_singleton`, `exists_neg_mul_of_isMinOn_compl_singleton`, `isContraction_swap`, `hadamard_fill_mem`, `isMinOn_hadamard_fill`, `conjTranspose_mem`, `isMinOn_conjTranspose`, `exists_isMinOn_compl_singleton_of_conjTranspose_mul`, `exists_neg_mul_of_isMinOn_compl_singleton_of_conjTranspose_mul`, `IsFlat.exists_isMinOn_cell`, `IsFlat.exists_neg_mul_of_isMinOn_cell`, `hadamard_conjTranspose`, `hadamard_conjTranspose_mul`, `restSq_hadamard_conjTranspose` | The seen cells are `({(i, j)}ᶜ : Set (m × n))`, as in the row of the one-hot kernels. A minimum is `IsMinOn nuclearNorm` on the matrices equal to `K` there, and that `X` is one of them is a separate hypothesis. `add_smul_single_apply_of_mem` and `eq_add_single_of_mem` are the lemmas of `RankOneFill`, now stated for rectangular matrices. For orthogonal columns `g'` is `restSq Kᴴ j i`. The statement for either shape asks for `g > 0` and `g' > 0` and keeps only `θ > 0`. The comparison with the Corollary is not a Lean statement. |
| OC6 | `restSq_recallKernel_pos`, `restSq_recallKernel_of_norm_eq_one`, `scale_of_norm_eq_one`, `exists_isMinOn_recallKernel`, `exists_neg_mul_of_isMinOn_recallKernel`, `exists_neg_mul_of_isMinOn_recallKernel_of_norm_eq_one`, `sum_sign_eq_zero`, `norm_sign_eq_one`, `sign_ne_zero` | "Two keys or more" is `Nontrivial κ`. The readout is complex, and "modulus one" is `‖φ v‖ = 1` for every `v`. "Of the order `m / \|ν\|^m`" is the two bounds, not a separate statement. |

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
embedding and readout `E`, parameter-free normalizations `N` of the mean square, ScheduleFree
AdamW with weight decay 1, 5000 updates of 512 examples. At the last position it holds

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
transformer of CH8 is trained by ScheduleFree AdamW with the rate 0.03 and the weight
decay 1, and reaches 90% of the held-out states. The selected main of the `diff-gpt`
repository is trained by AdamW with rates up to 0.001 and the weight decay 0.1, and does not.
In the second recipe every update reaches the end of the run with a weight of at least one
half, so its decay removes less than half of anything. On a loss without a scale a recipe
depends on its rate and its decay through their product, and at the rate 0.001 the historical
recipe is the weight decay 30.

### RR1. What the decay does along a run

The AdamW of the main model is the recurrence of AD3 for every coordinate,

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

The same holds for the revised ScheduleFree AdamW, the optimizer of the historical recipe,
with the learning rate `c · lr`, the decay `decay / c` and the epsilon `ε / c` from `c` times
the first point: the evaluation, base and query points of the run are `c` times those of the
other run. Its second moments are over `c²`, its averaging total is times `c^ratePower`, and
the averaging coefficients are the same; the first maximal rate is the sentinel `−1`, which
does not scale and which no rate sees.

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
- ScheduleFree AdamW in RR1 and RR2. Its decay is taken at the query point and its evaluation
  point is an average of the base points, so a coordinate without gradient does not follow
  the product. Computed for the historical recipe, not proved: 0.61 of it remains after 33
  updates, 0.14 after 100 and `8 · 10⁻⁶` after 5000, with changes of sign.
- The two optimizers against each other. RR3 is within one optimizer. That AdamW at the
  product 0.03 trains as ScheduleFree AdamW at that product is not claimed.
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

In the historical real model the AdamW of the selected main replaces ScheduleFree AdamW, with
its schedule, betas and clip, and with the weight decay `ρ` on every coordinate. In the
selected main `ρ` is the decay of its complex matrices; its real coordinates keep 0.1 and its
frequencies 0.

| Model | Recipe | Right of 4704, seeds 11, 17, 23 | Share | First update at 90% | Training loss at the end |
| --- | --- | --- | --- | --- | --- |
| Historical | ScheduleFree AdamW, rate 0.03, decay 1 | 4468, 4386, 4344 | 94.98%, 93.24%, 92.35% | 1300, 1500, 1150 | 0.039, 0.035, 0.036 |
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
| RR2 | `one_sub_sum_le_prod`, `half_le_prod_main`, `half_le_prod_Ico_main`, `one_sub_pow_le_exp`, `historical_remainder` | The first is for a finite set of indices. The recipe of the selected main is `0 ≤ η s ≤ 1/1000` at every step with the decay `1/10`; the bounds hold over every set of at most 5000 steps, and over `Finset.Ico (s + 1) t` for `t ≤ 5000`. The numbers 4.4495 and 0.64 of its schedule are computed, not in Lean. The last statement is `mainAdamPath` with the constant rate `3/100` and the decay `1`: AdamW, not ScheduleFree AdamW. |
| RR3 | `atScale`, `mainAdamStep_atScale`, `mainAdamRun_atScale`; `scaled`, `rate_scaled`, `IsScaled`, `isScaled_initial`, `query_scaled`, `step_isScaled`, `run_isScaled`, `run_scaled` | Homogeneity is asked at the one scale `c`: `G (fun j => c * w j) i = G w i / c` for all `w`, and for the oracle of ScheduleFree AdamW at every step. That a loss without a scale has such a gradient is not a Lean statement. ScheduleFree AdamW is `Transformer.ScheduleFree.AdamW.run`, the revised update of facebookresearch/schedule_free, and needs `0 ≤ lr`. `IsScaled c p s s'` relates two states; the maximal rates are compared after a nonnegative rate has entered. The last sentence of the item is not a Lean statement. |
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
| Historical | ScheduleFree AdamW, rate 0.03, decay 1 | 4666, 4682, 4625 | 99.19%, 99.53%, 98.32% | 350, 200, 400 | 850, 550, 500 | 94.98%, 93.24%, 92.35% |
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
| Historical | ScheduleFree AdamW, rate 0.03, decay 1 | 94.98%, 93.24%, 92.35% | 96.64%, 96.79%, 94.52% | 94.45%, 94.03%, 95.30% |
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

## Training with cross-entropy (added)

*Training reaches the optimal fit* proves that gradient descent with weight decay on the pure
factorization converges, and to a global minimum, with an explicit step for the squared loss
(WD8). The models of the note are trained with the cross-entropy. This section generalizes
the proof. The cross-entropy of logits that are linear in the product meets every hypothesis
of WD1–WD10, and it has the three constants of WD8 at every matrix, so its run converges
under an explicit bound on the step (XE1–XE3). What the run converges to changes with the
loss. The squared loss has exact fits, and as λ → 0⁺ its trained fits become the exact fits
of least nuclear norm (WD7). The cross-entropy of separable data has no minimum: as λ → 0⁺
its trained fits grow without bound, and they have the largest margin for their nuclear norm
(XE4–XE6). At every λ > 0 the unseen cells are still filled by M (XE7), so a flat kernel is
not returned, at a fixed λ or in the limit (XE7–XE9). The limit points of the direction of the
trained fits are the products of largest margin (XE10). With every cell seen and the answer
code of the note, the flat kernel is the only such product, so the trained fits converge in
direction to the kernel exactly when every cell is seen (XE11). A row or a column that is
never seen is zero in every trained product (XE12).

Notation as in that section: `⟨A, B⟩ = Re tr(Aᴴ B)`, `‖·‖ = ‖·‖_S2`, the model `X = U Vᴴ`
with the run of WD1, and a decayed minimum is a factorization of least weight-decayed loss
among those of every width (WD2).

### XE1. The loss

There are samples `s ∈ S` with labels `y_s` in a set `C` of classes, both finite. A sample
reads the product through probes, matrices `A_{s,c}` of the size of `X`, one for each class.
Its logits are

```
z_{s,c}(X) = ⟨A_{s,c}, X⟩,
```

real-linear in `X`. The loss is the cross-entropy of the labels, and `G` is its gradient:

```
L(X) = Σ_s ( log Σ_c exp z_{s,c}(X) − z_{s,y_s}(X) ),
G(X) = Σ_s ( Σ_c p_{s,c}(X) A_{s,c} − A_{s,y_s} ),    p_{s,c} = exp z_{s,c} / Σ_c' exp z_{s,c'}.
```

Two cases. The probes `A_{s,c} = φ(c) e_ij` of a cell `(i, j)` and an answer code `φ` read
`z_{s,c} = Re(conj φ(c) X_ij)`: the kernel of the note with its decoder (XE8). The probes
`A_{s,c} = e_ic`, with one column for each class, read `z_{s,c} = Re X_ic`: the product is the
table of the logits. These probes have norm one and vanish off the rows that the samples read.
So XE3 holds for them with `a = 1`, and a row that no sample reads is zero in every trained
table (XE12).

This loss has what WD1–WD5, WD9 and WD10 ask of a loss.

- The first-order inequality. For one sample let `d = z' − z` be a change of its logits and
  `p` the softmax of `z`. Then
  `log Σ_c exp z'_c − log Σ_c exp z_c = log Σ_c p_c exp d_c ≥ Σ_c p_c d_c` by Jensen's
  inequality, so the loss of the sample grows by at least `Σ_c p_c d_c − d_y`. With
  `d_c = ⟨A_{s,c}, Y − X⟩`, the sum over the samples is `L(X) + ⟨G(X), Y − X⟩ ≤ L(Y)`.
- `G` is continuous, and `L ≥ 0`, as `Σ_c exp z_c ≥ exp z_y`.
- `L` is real-analytic in the entries: the logits are real-linear, and the sum under the
  logarithm is positive.
- `L(0) = |S| log |C|`.

So for the cross-entropy the update of WD1 is the gradient step (WD3), and a convergent run
from full row rank with admissible steps ends at a decayed minimum (WD5).

### XE2. The three constants

WD6 and WD8 control the run through three bounds on the loss in a ball. Let the probes have
norm at most `a`, `‖A_{s,c}‖ ≤ a`, and write `N = |S|`. A logit moves by at most `a ‖H‖` when
the product moves by `H`. The cross-entropy has the three bounds at every matrix:

```
‖G(X)‖ ≤ 2 a N,    ⟨G(X), X⟩ ≥ −N log |C|,
L(Y) ≤ L(X) + ⟨G(X), Y − X⟩ + (N a² / 2) ‖Y − X‖².
```

- The gradient. `⟨G(X), H⟩ = Σ_s ( Σ_c p_{s,c} ⟨A_{s,c}, H⟩ − ⟨A_{s,y_s}, H⟩ )`: for each
  sample a mean of logit increments minus one increment, at most `2 a ‖H‖`. So
  `⟨G(X), H⟩ ≤ 2 a N ‖H‖`; take `H = G(X)`.
- The first-order term. A convex loss with `L ≥ ℓ_0` has `⟨G(X), X⟩ ≥ −(L(0) − ℓ_0)`: this is
  the first-order inequality from `X` to `0`. Here `ℓ_0 = 0` and `L(0) = N log |C|`.
- The quadratic model. For one sample, with `d` and `p` as in XE1 and `d̄ = Σ_c p_c d_c`, the
  loss exceeds its linear model by `log Σ_c p_c exp(d_c − d̄)`. If `|d_c| ≤ r` for every `c`,
  Hoeffding's lemma for a variable with values in `[−r, r]` gives
  `Σ_c p_c exp(d_c − d̄) ≤ exp((2r)² / 8)`, so the excess is at most `r² / 2`. Here
  `r = a ‖Y − X‖`.

So `γ = 2 a N`, `b = N log |C|` and `s = N a²`, with no radius. For the squared loss `γ` grows
with the radius of the ball (WD6); the gradient of the cross-entropy is bounded on the whole
space.

### XE3. The run converges

WD8 with these constants. Let λ > 0, let ρ satisfy `4 N log |C| ≤ λ ρ`, and take a step
η > 0 with

```
2η (λ² + (2 a N)²) ≤ λ    and    η (λ + 2 a N + 2 N a² ρ) ≤ 1.
```

Then the run of gradient descent with weight decay on the cross-entropy, from every start
with `‖U_0‖² + ‖V_0‖² ≤ ρ`, converges, whatever its width and rank, and its limit is
stationary; from a start of full row rank the limit is a decayed minimum. Convergence is a
conclusion, by Łojasiewicz's theorem as in WD8. Every start lies in such a ball: ρ may be as
large as needed, at the price of a smaller step.

WD9 and WD10 apply as they stand, the loss being real-analytic and bounded below: at a
width `k ≥ min(m, n)` small steps from almost every start of a sublevel set converge to a
decayed minimum, and decayed minima of that width exist for every λ > 0. Call a decayed
minimum of the cross-entropy a trained fit.

The mean. The models of the note are trained on the mean `L / N` of the losses of the
samples, whose gradient is `G / N`. For a real `c ≠ 0`, a step of WD1 on `c G` with the step η
and the weight decay λ is a step on `G` with the step `c η` and the weight decay `λ / c`:

```
U − η (c G V + λ U) = U − c η (G V + (λ / c) U),
```

and likewise for `V`. The weight-decayed loss of `c L` at λ is `c` times that of `L` at
`λ / c`, so for `c > 0` the decayed minima are the same. So training on the mean with `(η, λ)`
is training on the sum with `(η / N, N λ)`, and every statement of this section holds for the
mean with these values. The bounds above become those of one sample,

```
4 log |C| ≤ λ ρ,    2η (λ² + (2 a)²) ≤ λ    and    η (λ + 2 a + 2 a² ρ) ≤ 1:
```

for the mean they do not depend on the number of samples.

Example: one sample of a `1 × 1` product with the probes `+1` and `−1` and the label of `+1`,
so that `L(X) = log(1 + exp(−2 Re X))`. Here `a = 1`, `γ = 2`, `b = log 2` and `s = 1`. With
λ = 1, ρ = 3 and η = 1/10: `4 log 2 ≤ 3`, `2η (λ² + γ²) = 1` and `η (λ + γ + 2 s ρ) = 9/10`.
From `U_0 = [1 0]`, `V_0 = [0 1]` the run converges to a trained fit. From `U_0 = V_0 = 1`,
which has no full row rank, it converges to a stationary point.

### XE4. Margins

For a sample `s` and a class `c ≠ y_s` the gap is

```
g_{s,c}(X) = z_{s,y_s}(X) − z_{s,c}(X),
```

linear in `X`. The margin `μ(X)` is the least gap over the samples and their wrong classes;
there are a sample and two classes. The product classifies every sample when `μ(X) > 0`. The
margin is positively homogeneous, `μ(t X) = t μ(X)` for `t ≥ 0`, and every gap has
`g_{s,c}(X) ≤ 2 a ‖X‖ ≤ 2 a ‖X‖_S1`.

The loss of one sample is `log(1 + Σ_{c ≠ y} exp(−g_c))`, so the cross-entropy lies between
two functions of the gaps:

```
log(1 + exp(−g_{s,c}(X))) ≤ L(X)                                 for every gap,
L(X) ≤ N log(1 + (|C| − 1) exp(−μ)) ≤ N (|C| − 1) exp(−μ)      if every gap is at least μ.
```

### XE5. A trained fit against a competitor

Let `(U, V)` be a trained fit for a weight decay λ ≥ 0, `X = U Vᴴ` and `B = ‖X‖_S1`. Let `X'`
be a competitor with `‖X'‖_S1 ≤ 1` whose gaps are all at least `μ'`. WD2 at `Y = t X'`,
`t ≥ 0`, gives

```
L(X) + λ B ≤ L(t X') + λ t,
```

and at `t = B` it gives `L(X) ≤ L(B X')`: a trained fit has no more loss than the competitor
scaled to its own norm. With XE4 and `κ = N (|C| − 1)`, every gap `g` of `X` has

```
log(1 + exp(−g)) ≤ L(X) ≤ L(B X') ≤ κ exp(−B μ').
```

If `log(2κ) ≤ B μ'`, then `ε = κ exp(−B μ') ≤ 1/2`, and `1 + exp(−g) ≤ exp ε ≤ 1/(1 − ε)` gives
`exp(−g) ≤ 2ε`. So

```
g ≥ B μ' − log(2κ)    for every gap g of X,    that is    μ(X) ≥ ‖X‖_S1 μ' − log(2κ).
```

The margin of a trained fit is within the constant `log(2κ)` of the margin that the
competitor has at the same nuclear norm.

Trained products. Only the product enters this item and XE6, XE9, XE10 and XE11, and the
width of the factors plays no role. Call `X` a trained product for λ when it minimizes
`L + λ ‖·‖_S1` over all matrices. For λ ≥ 0:

- the product of a trained fit of any width is a trained product, by WD2;
- a trained product is the product of a trained fit of width `n`, the number of columns. By
  Lemma 1 it is `U Vᴴ` with `½ (‖U‖² + ‖V‖²) = ‖X‖_S1` at that width. Then for factors
  `(U', V')` of any width, `L(X) + λ ‖X‖_S1 ≤ L(U' V'ᴴ) + λ ‖U' V'ᴴ‖_S1`, and
  `‖U' V'ᴴ‖_S1 ≤ ½ (‖U'‖² + ‖V'‖²)`: so `(U, V)` is a trained fit.

So the trained products are the products of the trained fits of every width, and of the width
`n` alone. A statement about the products of all sequences of trained fits of the width `n` is
a statement about all sequences of trained products. This item and the limits of XE6, XE9,
XE10 and XE11 hold for every sequence of trained products `X_j` for `λ_j`: the fits behind
them may have any widths, and the widths may change along the sequence. For λ > 0 trained
products of the cross-entropy exist (XE3 with WD10); for λ = 0 and samples that some product
classifies there is none (XE6, statement 1).

### XE6. Vanishing weight decay: the largest margin

Let `λ_j → 0⁺`, let `(U_j, V_j)` be trained fits for `λ_j`, `X_j = U_j V_jᴴ`, and suppose
that some product has a positive margin. Scaled to `‖X'‖_S1 ≤ 1` it is a competitor with a
margin `μ' > 0`.

1. The norm grows: `‖X_j‖_S1 → ∞`. By XE5, `L(X_j) ≤ κ exp(−t μ') + λ_j t` for every `t ≥ 0`,
   so `L(X_j) → 0`: choose `t` for the first term, then `j` for the second. But a product
   with `‖X‖_S1 < b` has a gap at most `2 a b`, and so `L(X) ≥ log(1 + exp(−2 a b))` (XE4).
   So `‖X_j‖_S1 ≥ b` in the end, for every `b`. The cross-entropy of separable data has no
   minimum: it is positive at every product, as `L(X) ≥ log(1 + exp(−g)) > 0` for a gap `g`
   (XE4), and `L(t X') ≤ κ exp(−t μ') → 0` as `t → ∞`. Its infimum `0` is not attained, and
   the trained fits leave every bounded set. So for λ = 0 there is no trained product and no
   trained fit of any width: XE5 at λ = 0 is about samples that no product classifies.
2. The margin passes that of every competitor. If `‖X'‖_S1 ≤ 1` and `0 ≤ θ < μ(X')`, then
   `θ < μ(X_j) / ‖X_j‖_S1` in the end: by XE5, `μ(X_j) / B_j ≥ μ(X') − log(2κ) / B_j` with
   `B_j = ‖X_j‖_S1 → ∞`.
3. The limit. Let `μ* = sup {μ(X) : ‖X‖_S1 ≤ 1}`, finite as `μ(X) ≤ 2 a ‖X‖_S1`. A nonzero
   product has `μ(X) / ‖X‖_S1 ≤ μ*` by homogeneity. With 2,

   ```
   μ(X_j) / ‖X_j‖_S1 → μ*.
   ```
4. The second form. Here `μ* > 0`, as `μ* ≥ μ(X') > 0`. The least nuclear norm of a product
   with margin at least one is `1 / μ*`. If `μ(X) ≥ 1`, then `X ≠ 0` and
   `1 ≤ μ(X) ≤ μ* ‖X‖_S1` by 3, so `‖X‖_S1 ≥ 1 / μ*`. The largest margin is attained at some
   `Z` of the unit ball (XE10, statement 1); `Z / μ*` has margin one and nuclear norm at most
   `1 / μ*`, so it has the least norm. The two forms have the same solutions up to the scale
   `μ*`: a product `Z` of the unit ball has `μ(Z) = μ*` exactly when `Z / μ*` has margin at
   least one and nuclear norm `1 / μ*`. In one direction this was just shown. In the other,
   `‖Z‖_S1 = 1` and `μ(Z) ≥ μ*`, and `μ(Z) ≤ μ*` in the unit ball. For the trained fits,
   inverting 3,

   ```
   ‖X_j‖_S1 / μ(X_j) → 1 / μ*:
   ```

   scaled to margin one, their nuclear norm tends to the least one.

So with a vanishing weight decay the cross-entropy solves the margin problem of the nuclear
norm: the largest margin at a given `‖X‖_S1`, which is the least `‖X‖_S1` at a given margin
(4). For linear classifiers with an `ℓ_p` penalty this is the theorem of Rosset, Zhu and Hastie
(2004), and for homogeneous networks with weight decay that of Wei, Lee, Liu and Ma (2019).
Here the penalty on the factors is the nuclear norm of the product, by Lemma 1 and WD2. The
squared loss has M in this limit, the exact fit of least nuclear norm (WD7). The
cross-entropy replaces "equal to `K` on Ω" by "classify the samples with margin one".

### XE7. The unseen cells are filled by M

Say that a loss reads only the seen cells Ω when `L(X) = L(Y)` for all `X`, `Y` equal on Ω.
The squared loss on Ω does. The cross-entropy does when its probes vanish off Ω, since then
its logits read only Ω.

Let `L` read only Ω, let λ > 0, and let `(U, V)` be a decayed minimum of `L`, `X = U Vᴴ`. For
`Y` equal to `X` on Ω, WD2 gives `L(X) + λ ‖X‖_S1 ≤ L(Y) + λ ‖Y‖_S1 = L(X) + λ ‖Y‖_S1`, so

```
‖X‖_S1 ≤ ‖Y‖_S1    for every Y equal to X on Ω.
```

The product of a trained fit is a minimum of M from Ω for its own values. The loss decides
the seen cells, and weight decay fills the unseen ones by M, whatever the loss.

If `K` is not a minimum of M from Ω, neither is `t K` for a real `t ≠ 0`: a matrix `Y` equal
to `K` on Ω with a smaller nuclear norm gives `t Y`. Therefore:

- For λ > 0 no trained product is `t K` with `t ≠ 0`, at any width. By R2 this holds for a
  flat `K` with a nonzero unseen cell. The multiple `t = 0` is not excluded: for a large λ
  the trained product is `0`.
- Under the bounds of XE3, with probes that vanish off Ω, the run from a start of full row
  rank converges, and the product of its limit is no nonzero multiple of such a `K`. This is
  Theorem A for the run under the cross-entropy; nothing is assumed about its limit.

The multiple `t = 0` does occur, for every convex loss with a gradient `G`. At the zero
factorization the lifted gradient of WD1 is `[[λ, G(0)], [G(0)ᴴ, λ]]`, positive semidefinite
when `‖G(0)‖ ≤ λ`. So for `λ ≥ ‖G(0)‖` the zero factorization is a decayed minimum (WD2), at
every width. If `‖G(0)‖ ≤ γ < λ`, every decayed minimum has the product `X = 0`: WD2 at
`Y = 0` and the first-order inequality at `0` give

```
λ ‖X‖_S1 ≤ L(0) − L(X) ≤ −⟨G(0), X⟩ ≤ γ ‖X‖ ≤ γ ‖X‖_S1.
```

The cross-entropy has `γ = 2 a N` (XE2): for `λ > 2 a N` the only trained product is `0`. The
squared loss on Ω has `G(0) = −P_Ω K` and `γ = ‖K‖`. The sample of XE3 has the trained
product `0` for `λ > 2`, and the `1 × 1` kernel `1` for `λ > 1`; below that it has `1 − λ`
(WD7).

For the squared loss WD7 says more: the trained products stay a fixed δ below `‖K‖_S1`.
Under the cross-entropy the norm of the trained products is unbounded as λ → 0⁺ (XE6), and
the comparison is made at the scale of the product (XE9).

### XE8. The answer code

The kernel of the note shows the answer `c` of the task at each cell as `φ(c)` for an answer
code `φ : C → ℂ`, with `φ = ψ` on `ℤ/p`, and it decodes a product by the nearest codeword.
For a unimodular code `|x − φ(c)|² = |x|² + 1 − 2 Re(conj φ(c) x)`, so the codeword nearest
to `x = X_ij` is that of the class with the largest

```
z_c = Re(conj φ(c) X_ij) = ⟨φ(c) e_ij, X⟩.
```

So let the sample `s` see the cell `(i_s, j_s)`, and take the probes
`A_{s,c} = φ(c) e_{i_s j_s}`. They have the norm `|φ(c)| = 1`, so `a = 1` in XE2 and XE3, and
they vanish off the seen cells `Ω = {(i_s, j_s)}`, so XE7 applies. The softmax of these
logits is the decoder of the note made smooth.

Let `K` show the labels on the seen cells, `K_{i_s j_s} = φ(y_s)`. Then

```
g_{s,c}(K) = 1 − Re(conj φ(c) φ(y_s)) = |φ(y_s) − φ(c)|² / 2.
```

With an injective code every gap is positive, so `μ(K) > 0`: the kernel classifies every
sample, by half the squared distance of two codewords.

The margin is about the decoder of the note. By the identity above the gap `g_{s,c}(X)` is
positive exactly when `φ(y_s)` is nearer to `X_{i_s j_s}` than `φ(c)`. So a product with
`μ(X) > 0` is decoded to the label at every sample: the codeword of the label is the nearest.
By XE6, on samples that some product classifies the trained fits are decoded to the labels
in the end, as λ → 0⁺.

### XE9. Vanishing weight decay does not return a flat kernel

Let the probes vanish off Ω, so that the gaps and the margin read only Ω. Let `K` have a
positive margin and not be a minimum of M from Ω: some `Y` equal to `K` on Ω has
`‖Y‖_S1 < ‖K‖_S1`. Then `μ(Y) = μ(K)`, and `Y` has the larger margin for its norm:

```
μ(K) / ‖K‖_S1 < μ(K) / ‖Y‖_S1 = μ(Y) / ‖Y‖_S1.
```

Let δ be half the difference. By XE6, statement 2, with the competitor `Y / ‖Y‖_S1`, the
trained fits of every sequence `λ_j → 0⁺` have

```
μ(X_j) / ‖X_j‖_S1 ≥ μ(K) / ‖K‖_S1 + δ    in the end.
```

δ depends on `K`, Ω and the probes, and not on the sequence. Consequently `X_j / ‖X_j‖_S1`
does not converge to `K / ‖K‖_S1`. Some gap `g_{s,c}` of `K` is its margin. It is linear, so
if the normalized products converged, `g_{s,c}(X_j) / ‖X_j‖_S1` would tend to
`μ(K) / ‖K‖_S1`; but `g_{s,c}(X_j) ≥ μ(X_j)` keeps it δ above.

By R2 and XE8 this holds for the kernels of the note. Let `K` be flat, let it show the labels
on the seen cells with an injective unimodular code, and let a nonzero cell of `K` be unseen.
Then, as the weight decay vanishes, the trained fits of the cross-entropy with the probes of
XE8 grow without bound, classify the seen cells with a margin for their norm that ends δ
above that of `K`, and do not converge in direction to `K`. The kernel `ψ(a³ + a b)` with the
labels `a³ + a b` on a nonempty proper set of cells is an instance, for every `p ≥ 2`.

This is statement 2 of WD7 for the cross-entropy. There the trained products stay δ below
`‖K‖_S1`. Here their norm is unbounded, and they beat `K` by δ in margin for their norm.

### XE10. The direction of the trained fits

XE6 is about one number, the margin of a trained fit for its norm. The direction of a trained
fit is the product `X_j / ‖X_j‖_S1`, a point of the unit ball `{‖X‖_S1 ≤ 1}`. Its margin is
`μ(X_j) / ‖X_j‖_S1`, by homogeneity.

1. `μ*` is attained. The unit ball is bounded, as `|X_ij| ≤ ‖X‖_S1`. The nuclear norm is
   lower semicontinuous: for a limit `Z` of matrices with `‖Z_j‖_S1 ≤ b` take a contraction
   `W` with `⟨W, Z⟩ = ‖Z‖_S1`, by the dual form of the norm; then `⟨W, Z_j⟩ ≤ ‖Z_j‖_S1 ≤ b`,
   and in the limit `‖Z‖_S1 ≤ b`. The gaps are linear, so they are continuous. Now let
   `Z_j → Z` with `‖Z_j‖_S1 ≤ 1` and `μ(Z_j) → μ*`. Then `‖Z‖_S1 ≤ 1`, and each gap of `Z` is
   the limit of gaps `g(Z_j) ≥ μ(Z_j)`, so it is at least `μ*`: `μ(Z) = μ*`. Products of the
   unit ball with `μ(Z_j) > μ* − 1/(j + 1)` have a convergent subsequence. So some product of
   the unit ball has the margin `μ*`.
2. The limit points. Let some product have a positive margin, let `λ_j → 0⁺`, and let
   `X_j / ‖X_j‖_S1 → Z`. By XE6 the margins of these products of the unit ball tend to `μ*`,
   so `‖Z‖_S1 ≤ 1` and `μ(Z) = μ*` by 1. Here `μ* > 0`, and `μ* = μ(Z) ≤ μ* ‖Z‖_S1` by
   homogeneity. So

   ```
   ‖Z‖_S1 = 1    and    μ(Z) = μ*.
   ```

   A subsequence of trained fits is a sequence of trained fits, and their directions lie in
   a compact set. So every subsequence of the directions has a limit point, and every limit
   point is a product of nuclear norm one with the largest margin.
3. Convergence. If one product `Z_0` of the unit ball has the margin `μ*`, then

   ```
   X_j / ‖X_j‖_S1 → Z_0:
   ```

   every subsequence has a further subsequence that converges, by 2 to `Z_0`.
4. Margin one. Let `X_j / ‖X_j‖_S1 → Z`. In the end `μ(X_j) > 0`, by XE6, statement 2, and

   ```
   X_j / μ(X_j) = (‖X_j‖_S1 / μ(X_j)) · X_j / ‖X_j‖_S1 → Z / μ*,
   ```

   by XE6, statement 4. The limit has margin one and the least nuclear norm, `1 / μ*`. So
   with a unique `Z_0` the trained fits scaled to margin one converge to `Z_0 / μ*`.

So the direction of the trained fits approaches the solutions of the margin problem of XE6,
and it converges when the solution is unique. In the terms of this item, XE9 says that
`K / ‖K‖_S1` is no solution when `K` is not a minimum of M from Ω.

Example: the sample of XE3, a `1 × 1` product with the probes `+1` and `−1`. Its gap is
`2 Re X`, so `μ* = 2`, and in the unit disk only `Z = 1` has `2 Re Z = 2`. The trained fits
converge in direction to `1`.

### XE11. Every cell seen: the flat kernel is the direction

By XE9 the trained fits do not converge in direction to a flat kernel with a nonzero unseen
cell. With every cell seen they do, for the answer codes of the note.

Say that a unimodular code `φ` is mirrored at the gap `g` when, for every class `t`,

```
|φ(t) − φ(c)|² ≥ 2g  for every c ≠ t,    and    φ(c) + φ(c') = 2 (1 − g) φ(t)  for some c, c' ≠ t.
```

Call `c` and `c'` the neighbours of `t`. The code `ψ` of `ℤ/p`, `p ≥ 2`, is mirrored at
`g = 1 − cos(2π/p)`, with the neighbours `t ± 1`: `|ψ(t) − ψ(c)|² = 2 − 2 cos(2π (t − c)/p)`
is least at `c = t ± 1`, and `ψ(t + 1) + ψ(t − 1) = 2 cos(2π/p) ψ(t)`. Its gap is positive,
and `g < 2` for `p ≥ 3`. For `p = 2` the code is `±1` and `g = 2`.

Take the probes of XE8 for a code mirrored at `g`.

1. The bound at a sample. Let the sample `s` have the label `t = y_s` and the cell `(i, j)`.
   Its gaps against the neighbours of `t` add up to

   ```
   g_{s,c}(X) + g_{s,c'}(X) = 2 Re(conj φ(t) X_ij) − Re(conj (φ(c) + φ(c')) X_ij)
                            = 2g Re(conj φ(t) X_ij),
   ```

   and the margin is at most each of them. So `μ(X) ≤ g Re(conj φ(y_s) X_ij)` at every sample.
2. The kernel. Let `K` show the labels on the seen cells. Its gaps are
   `|φ(y_s) − φ(c)|² / 2 ≥ g` (XE8), and 1 at `X = K` gives `μ(K) ≤ g`. So `μ(K) = g`.
3. Every cell seen. Let every cell be seen, so that `K_ij = φ(y_s)` at every cell, and let `K`
   be flat with the squared singular value `c`. The sum of 1 over the cells is

   ```
   |m| |n| μ(X) ≤ g Σ_ij Re(conj K_ij X_ij) = g ⟨K, X⟩ ≤ g √c ‖X‖_S1,
   ```

   the last step by R3. The entries of `K` are unimodular, so
   `|m| |n| = ‖K‖² = min(|m|, |n|) c = √c ‖K‖_S1` by R1. Therefore

   ```
   μ(X) ‖K‖_S1 ≤ g ‖X‖_S1 = μ(K) ‖X‖_S1    for every X.
   ```

   No product has a larger margin for its norm than `K`: `μ* = g / ‖K‖_S1`, the margin of
   `K / ‖K‖_S1`.
4. The only one, if `0 < g < 2`. Let `‖Z‖_S1 ≤ 1` and `μ(Z) = μ*`. Then 3 holds with equality
   for `Z`, so 1 holds with equality at every cell, and both gaps of 1 equal `μ(Z)`. For `K`
   the two gaps are at least `g` and add up to `2g`, so both equal `g`. With `x = Z_ij`,
   `a = K_ij` and a neighbour `b = φ(c)` of `a` this reads

   ```
   Re(conj b a) = 1 − g,    Re(conj a x) = 1 / ‖K‖_S1,    Re(conj b x) = (1 − g) / ‖K‖_S1.
   ```

   As `−1 < 1 − g < 1`, the unit numbers `a` and `b` are independent over ℝ, and the two
   readings of `x` along them fix it: `x = a / ‖K‖_S1`. So `Z = K / ‖K‖_S1`.

By XE10, statement 3, for a code mirrored at a gap `0 < g < 2`, every cell seen, and a flat
`K` that shows the labels,

```
X_j / ‖X_j‖_S1 → K / ‖K‖_S1    as λ_j → 0⁺.
```

Scaled to margin one they converge to the kernel divided by the gap of the code. By XE10,
statement 4, the limit is `Z_0 / μ*` with `Z_0 = K / ‖K‖_S1` and `μ* = g / ‖K‖_S1`:

```
X_j / μ(X_j) → K / g.
```

For `ψ` on `ℤ/p` this is `K / (1 − cos(2π/p))`.

With XE9: if the flat `K` has no zero entry and shows the labels of the seen cells, the trained
fits converge in direction to `K` exactly when every cell is seen. For the kernel
`ψ(a³ + a b)` with the code `ψ` and `p ≥ 3`: under the cross-entropy a vanishing weight decay
returns the kernel, up to its scale, from the whole table, and from no proper part of it.

The bound `g < 2` leaves out the code `±1`. There `φ(c) = φ(c') = −φ(t)` and the two readings
of 4 are one. Statements 1–3 hold, and 4 fails: the direction of the trained fits depends on
their choice (5–7). For trained fits with real factors it is that of `K` (8, 9).

The logits of the code `±1` at a cell are `±Re X_ij`, so the loss and the margin read only
the real parts of the product. Let every cell be seen once, and let the flat `K` show the
labels; its entries are `±1`.

5. The multiples of `K`. At `X = t K` every sample has the logit `t` at its label and `−t` at
   the other class, so the softmax weights are `1 / (1 + e^{−2t})` and `1 / (1 + e^{2t})`, and

   ```
   G(t K) = −β K,    β = 2 / (1 + e^{2t}).
   ```

   By R3, `⟨K, Y⟩ ≤ √c ‖Y‖_S1`, and `⟨K, K⟩ = ‖K‖² = √c ‖K‖_S1` by R1. For `t ≥ 0` and
   `λ = β √c` the first-order inequality of XE1 gives, for every `Y`,

   ```
   L(Y) + λ ‖Y‖_S1 ≥ L(t K) − β ⟨K, Y⟩ + β t ⟨K, K⟩ + β √c ‖Y‖_S1 ≥ L(t K) + λ ‖t K‖_S1.
   ```

   So `t K` is a trained product for the weight decay `λ = 2 √c / (1 + e^{2t})`, which is
   `√c` at `t = 0` and tends to `0` as `t → ∞`.
6. A second matrix. Let `K' ≠ K` have the real parts of `K` and `‖K'‖_S1 ≤ ‖K‖_S1`. Then
   `μ(K') = μ(K) = 2`, so `‖K'‖_S1 = ‖K‖_S1` by 3, and `K' / ‖K‖_S1` is a second product of
   the unit ball with the margin `μ*`. And `t K'` has the loss and the norm of `t K`: it is a
   trained product for the same λ.
7. No limit. Let `t_j = j + 1` and `λ_j = 2 √c / (1 + e^{2 t_j}) → 0⁺`. The products
   `X_j = t_j K` are trained, with the direction `K / ‖K‖_S1`. With `X_j = t_j K'` at the odd
   `j` instead they are trained too, and products of trained fits (XE5); their directions
   alternate between `K / ‖K‖_S1` and `K' / ‖K‖_S1`, so `X_j / ‖X_j‖_S1` does not converge.
8. The real parts. Two trained products `X` and `Y` for one λ ≥ 0 have the same real parts
   at every seen cell, whatever the cells and the labels. A sample whose label has the sign
   `σ`, at a cell with the entry `x`, has the loss `log(1 + e^{−2σ Re x})`, and
   `u ↦ log(1 + e^u)` is strictly midpoint convex: for `a ≠ b`, `2 e^{(a+b)/2} < e^a + e^b`,
   so `(1 + e^{(a+b)/2})² < (1 + e^a)(1 + e^b)`. If the real parts differed at a seen cell,
   then `L((X + Y)/2) < (L(X) + L(Y))/2`, while `‖(X + Y)/2‖_S1 ≤ (‖X‖_S1 + ‖Y‖_S1)/2`: the
   midpoint would have less of `L + λ ‖·‖_S1` than `X` and `Y`.
9. Real factors. For `0 < λ ≤ √c` put `t(λ) = ½ log(2 √c / λ − 1) ≥ 0`, so that
   `λ = 2 √c / (1 + e^{2 t(λ)})`. By 5 and 8 every trained product for λ has the real parts
   of `t(λ) K`, and one with real entries is `t(λ) K`. The product of real factors has real
   entries. So trained fits with real factors, of any widths, have the direction
   `K / ‖K‖_S1` once `λ < √c`, and for them `X_j / ‖X_j‖_S1 → K / ‖K‖_S1` as `λ_j → 0⁺`.
   The probes `±e_ij` are real, so the gradient has real entries, a step of the run keeps
   real factors, and so does the limit of the run. Hence under the bounds of XE3 the run
   from real stacked factors of full row rank converges to a factorization of `t(λ) K`. From
   `U_0 = [1 0]`, `V_0 = [0 1]` these bounds can be met for every λ > 0, so trained fits with
   real factors exist at the width `|m| + |n|`.

A pair `K`, `K'` as in 6 is

```
K = [[1, 1], [−1, 1]],    K' = [[1 − i, 1 + i], [−(1 + i), 1 − i]] = (1, −i)ᵀ (1 − i, 1 + i).
```

Here `Kᴴ K = 2`, so `K` is flat with `‖K‖_S1 = 2 √2`, and `K'` has rank one and `‖K'‖² = 8`,
so `‖K'‖_S1 ≤ √8` by R3.

The mirror is needed. Take two classes with the code `φ = (1, i)`. It is unimodular and
injective, and it is mirrored at no gap: the only other codeword of `1` is `i`, and `2i` is no
real multiple of `1`. Let one sample read a `1 × 1` product `x` with the label of the codeword
`1`. Its gap is `Re x − Re(conj(i) x) = Re x − Im x`. The kernel `K = 1` is flat and shows the
label, its cell is seen, and `μ(K) = 1`. In the unit disk `Re x − Im x ≤ √2 |x| ≤ √2`, with
equality only at `x = (1 − i) / √2`. So `μ* = √2`, attained once, and by XE10, statement 3,

```
X_j / ‖X_j‖_S1 → (1 − i) / √2 ≠ K / ‖K‖_S1.
```

The direction turns away from the other codeword. It is still decoded to the label (XE8).

### XE12. An unseen row or column

By XE7 the product `X` of a trained fit is a minimum of M from Ω for its own values, for
λ > 0 and a loss that reads only Ω. By CW8 a minimum of M from Ω is zero on every row without
a cell of Ω, and on every such column: removing the row keeps the seen values and lowers the
nuclear norm. So

```
X_ij = 0    at every cell (i, j) whose row has no cell in Ω, or whose column has none.
```

This fixes the unseen cells of such rows and columns, for every such loss and every λ > 0,
whatever the seen values are. A token that never occurs in training is such a row or column
(CW8).

For the answer code of XE8 let no sample see the row `i`, or let none see the column `j`.
Then every logit `Re(conj φ(c) X_ij)` at the cell `(i, j)` is zero: the classes tie and the
softmax is uniform. For held-out samples `S'` at such cells, with any labels, the
cross-entropy is that of chance,

```
L'(X) = |S'| log |C|.
```

Under the bounds of XE3, with probes that vanish off Ω, the run from a start of full row rank
converges, and the product of its limit is zero on these rows and columns.

Example: a `2 × 1` product, the code `±1`, and one sample at the cell `(0, 0)`. The row `1` is
unseen. The trained product has `X_10 = 0`, and a held-out sample at `(1, 0)` has the loss
`log 2`.

### XE13. Scope

Proved, for the pure factorization `X = U Vᴴ` trained by gradient descent with weight decay
on the cross-entropy of logits that are real-linear in the product:

- the loss is convex with a continuous gradient, nonnegative and real-analytic in the
  entries, so WD1–WD5, WD9 and WD10 apply to it (XE1);
- for probes of norm at most `a` it has the constants of WD6 and WD8 at every matrix (XE2),
  and under two explicit bounds on the step the run from every start converges to a
  stationary point, and from stacked factors of full row rank to a decayed minimum; no
  convergence is assumed; training on the mean of the losses is training on their sum with
  the step `η / N` and the weight decay `N λ`, under bounds without `N` (XE3);
- for λ ≥ 0 the gaps of a trained fit are within `log(2κ)` of the margin that a competitor
  of unit nuclear norm has at the norm of the fit; the trained products, the minimizers of
  `L + λ ‖·‖_S1`, are the products of the trained fits of every width, so this and the limits
  below hold for fits of any widths, which may change along a sequence (XE5);
- as λ → 0⁺ on samples that some product classifies, the nuclear norm of the trained fits
  tends to infinity and their margin for their norm tends to the largest margin `μ*` of the
  unit ball of the nuclear norm; the least nuclear norm at margin one is `1 / μ*`, with the
  same solutions up to scale, and the norm of the trained fits for their margin tends to it
  (XE6);
- for λ > 0 and a loss that reads only the seen cells, a trained product is a minimum of M
  from Ω for its own values, and it is no nonzero multiple of a flat kernel with a nonzero
  unseen cell; the run of XE3 converges to no such multiple; for a weight decay above the
  norm of the gradient at `0`, so above `2 a N`, the trained product is `0` (XE7);
- for an injective unimodular answer code and such a flat kernel that shows the labels, the
  trained fits pass its margin for its norm by a fixed δ as λ → 0⁺, and they do not converge
  in direction to it (XE8, XE9);
- the largest margin of the unit ball is attained; as λ → 0⁺ every limit point of the
  direction `X_j / ‖X_j‖_S1` of the trained fits has nuclear norm one and the largest margin,
  limit points exist, and the direction converges when one product of the unit ball has the
  largest margin; where the direction converges to `Z`, the trained fits scaled to margin
  one converge to `Z / μ*` (XE10);
- for an answer code mirrored at a gap `0 < g < 2`, such as `ψ` for `p ≥ 3`, with every cell
  seen, a flat kernel that shows the labels has the largest margin for its norm and is the
  only such product of the unit ball up to its scale, so the trained fits converge in
  direction to it, and scaled to margin one to `K / g`; a flat kernel without zero entries
  is the limit of the direction exactly when every cell is seen; for the code `(1, i)`, which
  is not mirrored, the direction on one seen cell converges to `(1 − i) / √2`, not to the
  kernel; for the code `±1`, where `g = 2`, with every cell seen once, the nonnegative
  multiples of a flat kernel that shows the labels are trained products, and for the kernel
  `[[1, 1], [−1, 1]]` two products of the unit ball have the largest margin and some trained
  fits do not converge in direction; trained products for one weight decay have the same
  real parts, trained fits with real factors converge in direction to the kernel, and the
  run of XE3 from real stacked factors of full row rank converges to a factorization of the
  multiple `t(λ) K` (XE11);
- for λ > 0 and a loss that reads only the seen cells, a trained product is zero on every row
  and every column without a seen cell; with an answer code the held-out cross-entropy at
  such cells is that of chance, and the run of XE3 ends at such a product (XE12).
- without convexity, for a loss that is real-analytic in the entries and bounded below with
  a continuous gradient, small steps from a sublevel set converge to a stationary point; for
  the logits `±(Re x)²` of a `1 × 1` product, a weight decay `0 < λ ≤ 1/4` and a step with
  `ηλ ≤ ½`, every run from the ball `‖U_0‖² + ‖V_0‖² ≤ λ/4` converges to the zero
  factorization, which is not a decayed minimum (XE14);
- for the kernel of signs `[[1, 1], [−1, 1]]`, which is the kernel `ψ(a³ + a b)` for `p = 2`,
  with the code `±1` and every cell seen but `(1, 1)`: the largest margin of the unit ball
  is `1`, only `½ [[1, 1], [−1, −1]]` has it, and the trained products converge in direction
  to it; for every small weight decay the trained products, and the limits of the runs of
  XE3 from a start of full row rank, decode the unseen cell to the class of `−1`, where the
  kernel shows the class of `+1` (XE15);
- for the code `±1`, a flat kernel that shows the labels with every cell seen once and
  `0 < λ ≤ √c`: a matrix is a trained product exactly when it has the real parts of `t(λ) K`
  and at most its nuclear norm; for the kernel `[[1, 1], [−1, 1]]` the trained products are
  the segment `t(λ) K + i s [[1, −1], [1, 1]]`, `|s| ≤ t(λ)`, and the run of XE3 from a start
  of full row rank ends at a factorization of one of its points (XE16);
- without weight decay, on samples that some product classifies with a positive margin and
  from stacked factors of full row rank: the run has no limit when `2 a N η < 1`, and it is
  outside every ball `‖U‖² + ‖V‖² ≤ ρ` with `η (2 a N + 2 N a² ρ) < 1` at infinitely many
  steps (XE17);
- for every positive sequence of weight decays `λ_j → 0`, one run of gradient descent that
  passes through them in phases, each with the step of XE3 for the ball of its start, has
  for some lengths of the phases, at the phase ends, a nuclear norm that tends to infinity
  and a margin for its norm that tends to the largest; its products then converge in
  direction to the flat kernel when every cell is seen, for a mirrored code of gap
  `0 < g < 2`, and they answer the unseen cell of the kernel of signs with the wrong class
  (XE18);
- on one sample of two classes with the probes `±1` of a `1 × 1` product the run without
  weight decay is followed at every step `η > 0`: from every start with `U_0 + V_0 ≠ 0`, at
  every width, the norm of the factors tends to infinity, the loss tends to `0`, at least
  like `1/t`, the margin for the nuclear norm tends to the largest margin and the direction
  of the product to the product of largest margin; from the other starts the loss stays at
  least `log 2` (XE19);
- on samples of two classes with the code `±1` at cells no two of which share a row or a
  column, the rows of each sample follow the run of XE19: for every step η > 0, at every
  width, the norm of the factors tends to infinity when the rows of some sample do not
  cancel, and the loss tends to `0`, with a margin that tends to infinity, exactly when the
  rows of no sample cancel (XE20).

Assumed or not covered:

- No weight decay. For λ = 0 the cross-entropy of separable data has no minimum and no
  trained fit (XE6, statement 1), and the sublevel sets of WD9 are not compact: without weight
  decay a sublevel set on the factor space that contains a factorization of `0` contains
  `(U, 0)` for every `U`. These three facts are proved. Of the run without weight decay XE17
  proves that it has no limit and that it does not settle in a ball where its step is
  admissible. That its norm tends to infinity for one fixed step, that its loss tends to
  `0`, and its implicit bias, the direction of the factors, are proved for one sample of two
  classes, at every step (XE19). For more than one sample the first two are proved when no
  two samples share a row or a column (XE20); the direction, and samples that share a row or
  a column, are not treated. The limit λ → 0⁺ of XE6 and XE9 runs over trained fits. One run
  whose weight decay decreases to `0` reaches it at the ends of phases whose lengths are not
  explicit (XE18); a schedule written before the run, the steps inside a phase and one fixed
  step are not treated.
- The direction when several products of the unit ball have the largest margin. XE10 gives
  the limit points of `X_j / ‖X_j‖_S1`; that it converges is proved only when the product of
  largest margin is unique. XE11 proves that for a mirrored code of gap `0 < g < 2` with
  every cell seen. For a code that is not mirrored the limit can be another product: for the
  code `(1, i)` on one seen cell it is `(1 − i) / √2`, not the kernel (XE11). For the code
  `±1`, where `g = 2`, the direction need not converge: for the kernel `[[1, 1], [−1, 1]]`
  with every cell seen once two products have the largest margin, and some trained fits
  alternate between them (XE11). Trained fits with real factors converge in direction to the
  kernel, and the runs of XE3 from real factors end at its multiples (XE11). For factors
  that are not real the trained products for one weight decay are those of XE16: their real
  parts and their nuclear norm are determined, and for the kernel `[[1, 1], [−1, 1]]` the
  limit of a run is a point of a segment. Which point a run from a start that is not real
  reaches is not determined here.
- The unseen cells. By XE7 they are filled by M from the seen values of the trained product.
  Those values are not the values of `K`, so the fill of the Corollary and its decoded
  accuracy do not transfer. Three cases are settled: with every cell seen there is nothing to
  fill (XE11), a row or a column without a seen cell is zero (XE12), and the one unseen cell
  of the `2 × 2` kernel of signs is decoded to the wrong class for every small weight decay
  (XE15). Which class an unseen cell of a seen row and a seen column decodes to in a larger
  kernel, such as `ψ(a³ + a b)` for `p ≥ 3`, is not determined here.
- Logits that are not linear in the product, such as a network between the product and the
  logits. Then the loss need not be convex. Small steps still converge to a stationary point
  when the loss is real-analytic and bounded below, and runs from a set of starts with
  interior points, among them starts of full row rank, can end at a factorization that is
  not a decayed minimum: the logits `±(Re x)²` (XE14). Which readouts that are not linear
  keep the conclusion of XE3 is not determined.
- Losses other than the squared loss, the cross-entropy and those of WD9.
- The model and the optimizer: for the real network the modeling step of Lemma 1 remains an
  assumption, as the note says. For AdamW the limits of a run are not those of gradient
  descent with weight decay (WD13).

### XE14. Logits that are not linear in the product

XE1–XE3 take logits that are real-linear in the product. Linearity enters twice. It makes the
loss convex, and the first-order inequality makes the field `G` the gradient of the loss
(WD3). And WD2 and WD5 use the inequality again: a stationary point with a positive
semidefinite lifted gradient is a decayed minimum. The first use does not need convexity. The
second does.

1. Small steps converge. Let `L` be real-analytic in the entries and bounded below, and let a
   continuous field `G` be its gradient: `d/dt L(X + t H) = ⟨G(X), H⟩` at `t = 0` for all `X`
   and `H`. A real-analytic function is differentiable, so along a curve `X + t H(t)` with a
   continuous `H` the derivative at `t = 0` is `⟨G(X), H(0)⟩`, the conclusion of WD3. So the
   update of WD1 is the gradient step of the weight-decayed loss on the factor space, and WD9
   applies with its proof: for λ > 0 and a level `c` there is `η_0 > 0` such that for
   `0 < η < η_0` the run from every start with weight-decayed loss at most `c`, of any width
   and rank, converges, and its limit is stationary, `M [U; V] = 0`. The cross-entropy of
   logits that are real-analytic in the product is such a loss: it is nonnegative.
2. The limit need not be a decayed minimum. Let two classes read a `1 × 1` product `x`
   through the logits `±(Re x)²`, with the label of `+`. The loss and its gradient are

   ```
   L(X) = log(1 + e^{−2 r²}),    G(X) = −4 r w(r),    w(r) = e^{−2 r²} / (1 + e^{−2 r²}),    r = Re x.
   ```

   The loss is nonnegative and real-analytic, and `G` is continuous, so 1 applies. It is not
   convex: `L(0) = log 2` is its largest value and `G(0) = 0`, so the first-order inequality
   fails at `X = 0`.

   Near the zero factorization the gradient is small. As `0 ≤ w ≤ ½`,

   ```
   ‖G(X)‖ ≤ 2 |r|,    ⟨G(X), X⟩ = −4 r² w(r) ≥ −2 r²,    |r| ≤ ‖U Vᴴ‖ ≤ ½ (‖U‖² + ‖V‖²).
   ```

   Write `ω = ‖U‖² + ‖V‖²`, so that `‖G‖ ≤ ω` and `⟨G, X⟩ ≥ −ω²/2`. A step of WD1 is
   `W ← (1 − η M) W`, and with `⟨W, M W⟩ = λ ω + 2 ⟨G, X⟩` and `‖M W‖² ≤ 2 (λ² + ‖G‖²) ω`
   (WD1, WD6),

   ```
   ω' = ω − 2η ⟨W, M W⟩ + η² ‖M W‖² ≤ ω (1 − 2ηλ + 2η ω + 2η² λ² + 2η² ω²).
   ```

   Let `ω ≤ λ/4` and `ηλ ≤ ½`. Then `2η ω ≤ ηλ/2`, `2η² λ² ≤ ηλ` and `2η² ω² ≤ ηλ/16`, so

   ```
   ω' ≤ (1 − ηλ/4) ω.
   ```

   So for `0 < η` every run from the ball `‖U_0‖² + ‖V_0‖² ≤ λ/4`, of any width and rank,
   stays in it and converges to the zero factorization. In the ball
   `η (λ + ‖G‖) ≤ (5/4) ηλ < 1`, so the step matrices are positive definite, as WD5 asks.

   The zero factorization is not a decayed minimum when `λ ≤ 1/4`. Its weight-decayed loss
   is `log 2 > 0.693`. The factorization `(3/2) · (4/3)` of the product `2`, of width one,
   has

   ```
   log(1 + e^{−8}) + λ (9/4 + 16/9) / 2 ≤ 1/9 + 145/288 < 0.615,
   ```

   as `log(1 + u) ≤ u` and `e^8 ≥ 9`.

So for `0 < λ ≤ 1/4` and `0 < η` with `ηλ ≤ ½`, every run from the ball converges, and its
limit is not a decayed minimum. The ball has interior points, and it contains starts of full
row rank: for λ = 1/4 the start `U_0 = [1/8 0]`, `V_0 = [1/8 1/8]` has
`‖U_0‖² + ‖V_0‖² = 3/64 ≤ 1/16`, full row rank and the product `1/64`. So the conclusions of
WD5 ("a convergent run from full row rank with admissible steps ends at a decayed minimum")
and of WD10 ("small steps from almost every start converge to a decayed minimum") fail for
this loss, which meets their hypotheses except convexity. With the linear logits `±Re x` in
place of `±(Re x)²` this is the sample of XE3, whose run from a start of full row rank ends
at a trained fit.

What 2 uses is that these logits have no linear term at `0`: the gradient vanishes there, and
the weight decay holds the factors at `0`, a local minimum of the weight-decayed loss that is
not global. Which readouts that are not linear keep the conclusion of XE3 is not determined.

### XE15. One unseen cell of a seen row and a seen column

XE13 left open which class an unseen cell of a seen row and a seen column decodes to. For the
smallest flat kernel without zeros it is the wrong class.

Let `K = [[1, 1], [−1, 1]]`, the kernel of signs of XE11. It is flat, `Kᴴ K = 2 · 1`, and it
is the kernel `ψ(a³ + a b)` for `p = 2`: there `ψ(0) = 1`, `ψ(1) = −1`, and `a³ + a b` is odd
only at `(a, b) = (1, 0)`. The code is `±1`. The cells `(0, 0)`, `(0, 1)` and `(1, 0)` are
seen once each, with the labels that `K` shows. The cell `(1, 1)`, where `K` shows the class
of `+1`, is unseen; its row and its column are seen.

1. The gaps. The gap of a product `X` at a seen cell against the other class is twice the
   real part of the cell, with the sign of the label: `2 Re X_00`, `2 Re X_01` and
   `−2 Re X_10`. The margin is the least of the three.

2. A bound for `2 × 2` matrices. For `Z` with the rows `(a, b)` and `(c, d)`,

   ```
   ‖Z‖_S2² + 2 |det Z| ≤ ‖Z‖_S1².
   ```

   With the singular values `σ₁`, `σ₂` both sides are `(σ₁ + σ₂)²`; only the inequality is
   used, and it has a proof by a certificate. Let `ω` be a unit number with
   `conj ω · det Z = |det Z|`, and

   ```
   C = Z + ω adj(Z)ᴴ = [[a + ω conj d, b − ω conj c], [c − ω conj b, d + ω conj a]].
   ```

   Then `Cᴴ C = s · 1` and `⟨C, Z⟩ = s` for `s = ‖Z‖_S2² + 2 |det Z|`. So `C` is flat, and R3
   gives `s = ⟨C, Z⟩ ≤ √s ‖Z‖_S1`. With `|det Z| ≥ |b| |c| − |a| |d|` the bound reads

   ```
   (|a| − |d|)² + (|b| + |c|)² ≤ ‖Z‖_S1².
   ```

3. The largest margin. By 1 the margin of `X` is at most `Re X_01 − Re X_10`, so by 2 at
   most `|X_01| + |X_10| ≤ ‖X‖_S1`. The matrix

   ```
   Z₀ = ½ [[1, 1], [−1, −1]]
   ```

   has rank one, `‖Z₀‖_S1 ≤ 1`, and its three gaps are `1`. So `μ* = 1`. No other product of
   the unit ball has the margin `1`: for such a product the three seen cells have modulus at
   least `½`, the bound of 2 gives `|X_01| = |X_10| = ½` and `|X_00| = |X_11|`, and the first
   form of the bound gives `|X_00| = ½` and `det X = 0`. The seen cells are then real, `½`,
   `½` and `−½`, and `det X = 0` leaves `X_11 = X_10 X_01 / X_00 = −½`.

4. The direction. By XE10, for every sequence of trained products `X_j` at weight decays
   `λ_j → 0⁺`,

   ```
   X_j / ‖X_j‖_S1 → ½ [[1, 1], [−1, −1]].
   ```

5. The unseen cell. The limit has `−½` at `(1, 1)`. Since 4 holds for every sequence, there
   is a `λ₀ > 0` such that for every weight decay `0 < λ < λ₀` every trained product has
   `Re X_11 < 0`: the codeword `−1` is nearer to the unseen cell than `+1`, and the cell is
   decoded to the class of `−1`. The kernel shows the class of `+1` there.

6. The run. For `0 < λ < λ₀`, under the bounds of XE3 with `a = 1`, `N = 3` and `|C| = 2`,
   the run from stacked factors of full row rank converges, and the product of its limit is
   a trained product, so it answers the unseen cell with the wrong class. Such a run exists:
   the start `U₀ = [1 0]`, `V₀ = [0 1]` of width four, with a weight decay below `λ₀` and a
   small step.

The trained product completes the three seen signs to a matrix of rank one, and the rank-one
completion of the signs `(+, +)`, `(−, ·)` has `−` in the fourth cell. The kernel has `+`
there, and that sign is what makes it flat. For the squared loss OC5 finds the same opposite
sign at one unseen cell; here it is found for the cross-entropy, at the largest margin.

Not covered: the threshold `λ₀` in closed form; kernels larger than `2 × 2`, among them
`ψ(a³ + a b)` for `p ≥ 3` with its code of `p` classes; several unseen cells.

### XE16. The code ±1: all trained products for one weight decay

For the code `±1` XE11 gives one trained product for each weight decay, `t(λ) K`, and the
real parts of every other one. XE13 left open what the weight decay leaves free. Let a flat
kernel `K` show the labels, with every cell seen once, and let `0 < λ ≤ √c`.

1. The set. A matrix `X` is a trained product for `λ` if and only if

   ```
   Re X = t(λ) K    and    ‖X‖_S1 ≤ t(λ) ‖K‖_S1.
   ```

   The loss reads only the real parts (XE11), so a matrix with the real parts of `t(λ) K` has
   its loss, and it minimizes `L + λ ‖·‖_S1` exactly when it has at most its nuclear norm.
   Every trained product then has the nuclear norm `t(λ) ‖K‖_S1`.

2. The kernel of signs. For `K = [[1, 1], [−1, 1]]`, where `c = 2`, let
   `J = [[1, −1], [1, 1]]`. Then

   ```
   X is a trained product for λ    ⟺    X = t(λ) K + i s J  for a real s with |s| ≤ t(λ).
   ```

   The trained products are a segment. Its midpoint is `t(λ) K`. Its ends `t(λ) (K ∓ i J)`
   have rank one, and `K − i J` is the second product of XE11.

   "⟸": `K + i J` and `K − i J` have rank one and `‖·‖_S2² = 8`, so their nuclear norm is at
   most `2 √2 = ‖K‖_S1`, and `t K + i s J` is a convex combination of `t (K + i J)` and
   `t (K − i J)`.

   "⟹": write `X = t K + i B` with a real `B`. The bound of XE15,
   `‖X‖_S2² + 2 |det X| ≤ ‖X‖_S1² ≤ 8 t²`, with `‖X‖_S2² = 4 t² + ‖B‖²` and
   `Re det X = 2 t² − det B`, gives `‖B‖² ≤ 2 det B`, so `B = [[a, −b], [b, a]]`. Then
   `|det X| = Re det X = 2 t² − a² − b²`, so `Im det X = 2 t (a − b)` is zero. For `t > 0`
   this is `b = a`, so `B = a J` and `0 ≤ Re det X = 2 t² − 2 a²`. For `t = 0` the
   determinant `−a² − b² ≥ 0` gives `B = 0`.

3. The run. For `0 < λ ≤ √2`, under the bounds of XE3 with the probe norm `1`, `N = 4` and
   `|C| = 2`, the run from stacked factors of full row rank converges, and the product of its
   limit is a point of the segment, `t(λ) K + i s J` with `|s| ≤ t(λ)`. So the limit of a run
   is determined up to one real number. From real factors `s = 0` (XE11).

Not covered: which point of the segment a run from factors that are not real reaches; the
set of statement 1 in closed form for a flat kernel larger than `2 × 2`.

### XE17. The run without weight decay

By XE6, for λ = 0 the cross-entropy of samples that some product classifies with a positive
margin has no minimum, and no factorization is a trained fit. This item is about the run of
gradient descent on the factors with λ = 0, `W_{t+1} = (1 − η M_t) W_t` with
`M_t = [[0, G_t], [G_tᴴ, 0]]`. Let some product have a positive margin, let the probes have
norm at most `a`, let `N` be the number of samples, and let the start be stacked factors of
full row rank.

1. No limit. If `2 a N η < 1`, the run converges to no pair of factors.

   By XE2 `‖G_t‖ ≤ 2 a N`, so the step matrices `1 − η M_t` are positive definite (WD6, where
   the condition is `η (|λ| + ‖G‖) < 1`). The proof of WD5 does not use the sign of λ: a
   limit of the run would be a decayed minimum for λ = 0, and there is none (XE6).

2. A run that stays in a ball ends at a trained fit (WD8, for every real λ). With the
   constants `γ = 2 a N` and `s = N a²` of XE2, which hold at every matrix: if
   `η (2 a N + 2 N a² ρ) ≤ 1`, a run that stays in the ball `‖U‖² + ‖V‖² ≤ ρ` converges, and
   under statement 1's condition its limit is a decayed minimum for λ = 0.

3. Growth. If `η (2 a N + 2 N a² ρ) < 1`, then `‖U_t‖² + ‖V_t‖² > ρ` at some step, and at
   infinitely many steps.

   A run that stayed in the ball would end at a trained fit by statement 2, and there is
   none. For infinitely many steps: every step multiplies the stack by the invertible
   matrix `1 − η M_t`, so the run keeps full row rank, and its tail from any step is the run
   from the factors of that step; the first part applies to it.

   So for a step η the run does not settle in the ball of any squared radius

   ```
   ρ < ρ*(η) = (1/η − 2 a N) / (2 N a²),
   ```

   and `lim sup ‖W_t‖² ≥ ρ*(η)`, a bound that grows as `1/η` when the step shrinks.

Example: one sample of two classes with the probes `±1` of a `1 × 1` product (the sample of
XE3), where `a = 1`, `N = 1` and the product `1` has the margin `2`. With η = 1/10 the run
from `U_0 = [1 0]`, `V_0 = [0 1]` has no limit, and `‖W_t‖² > 2` at infinitely many steps:
`(1/10) (2 + 2 · 2) < 1`. Here `ρ*(η) = 4`.

Not covered. For one fixed step: that the norm of the factors tends to infinity, that the
loss tends to `0`, and the direction of the factors, the implicit bias. XE19 proves the
three for the sample of the example above, at every step, and XE20 the first two for samples
no two of which share a row or a column; for other sets of samples they are open. Outside
the ball of squared radius `ρ*(η)` the quadratic model of XE2 no longer gives a decrease of
the loss:
its constant on the factor space grows with the norm of the factors, and the proof of a
decrease there needs the smaller curvature of the cross-entropy where its loss is small.
For the gradient flow of a homogeneous model from a point that classifies the samples, the
growth of the norm and the convergence of the direction to a stationary point of the margin
problem in the parameters are theorems of Lyu and Li (2020) and of Ji and Telgarsky (2020);
the pure factorization is homogeneous of degree two in its factors. They are not formalized
here, and they are not about the largest margin for the nuclear norm of XE6.

### XE18. One run with a decreasing weight decay

The limits of XE6, XE9, XE10 and XE11 run over trained fits, one for each weight decay. By
XE17 the run with λ = 0 has no limit. This item is about one run of gradient descent whose
weight decay decreases to `0`. Let some product have a positive margin, let the probes have
norm at most `a`, let `N` be the number of samples and `γ = 2 a N`, `b = N log |C|`,
`s = N a²` the constants of XE2, and write `‖W‖² = ‖U‖² + ‖V‖²` for a pair of factors
`W = (U, V)`. Let `λ_0, λ_1, …` be positive weight decays that tend to `0`; they need not
decrease monotonically.

The run in phases. It starts at `W_0`. Phase `j` makes `T_j` steps of gradient descent with
the weight decay `λ_j`, from the factors `W_j` at which phase `j − 1` ended, with the step

```
η_j = min( λ_j / (2 (λ_j² + γ²)),  1 / (λ_j + γ + 2 s ρ_j) ),    ρ_j = ‖W_j‖² + 4 b / λ_j.
```

`X_j = U_j V_jᴴ` is the product at the start of phase `j`. The weight decays are given and
the steps are determined by the run; only the lengths `T_j` are free.

1. One phase. Let λ > 0 and let `W` be stacked factors of full row rank. The run from `W`
   with the weight decay λ and the step `η(λ, W)` of the display converges to a trained fit
   for λ, and it has full row rank at every step.

   The ball of squared radius `ρ = ‖W‖² + 4 b / λ` contains `W` and has `4 b ≤ λ ρ`, and
   `η(λ, W)` is the largest step that the two bounds of XE3 admit for it. So XE3 applies:
   the run stays in the ball and converges, and its limit is a decayed minimum. Every step
   multiplies the stack by the positive definite `1 − η M_t` (WD6), so the rank is kept.

2. The run. From stacked factors `W_0` of full row rank there are lengths `T_j` with

   ```
   ‖X_j‖_S1 → ∞    and    μ(X_j) / ‖X_j‖_S1 → μ*.
   ```

   The lengths are chosen one after another. Phase `j` starts at factors of full row rank,
   so by 1 its run converges to a trained fit for `λ_j`, with a product `X*_j`. The nuclear
   norm is continuous, as `|‖X‖_S1 − ‖Y‖_S1| ≤ ‖X − Y‖_S1 ≤ √m ‖X − Y‖_S2` for `m` rows, and
   the margin is the least of finitely many gaps, which are continuous; so the margin for
   the nuclear norm is continuous at every nonzero matrix. The phase is stopped at a step
   `T_j` where the nuclear norm of the product is within `1` of `‖X*_j‖_S1` and, if
   `X*_j ≠ 0`, its margin for the nuclear norm is within `λ_j` of the value at `X*_j`. The
   factors at that step have full row rank by 1. By XE6 `‖X*_j‖_S1 → ∞` and
   `μ(X*_j) / ‖X*_j‖_S1 → μ*`, so `X*_j ≠ 0` from some phase on, and both limits pass to the
   phase ends.

3. The direction. If one product `Z_0` of the unit ball has the margin `μ*`, then for the
   lengths of 2

   ```
   X_j / ‖X_j‖_S1 → Z_0.
   ```

   XE10 uses of the trained fits only the second limit of 2: for every sequence of matrices
   with `μ(X_j) / ‖X_j‖_S1 → μ*`, a limit point of the direction is a product of the unit
   ball with the margin `μ*`, and the directions lie in a compact set. Two cases:

   - every cell seen, an answer code mirrored at a gap `0 < g < 2` and a flat kernel `K` that
     shows the labels (XE11): `X_j / ‖X_j‖_S1 → K / ‖K‖_S1`. For the kernel `ψ(a³ + a b)` on
     `ℤ/p`, `p ≥ 3`, with the code `ψ`, the run returns the kernel in direction;
   - the kernel of signs `[[1, 1], [−1, 1]]` with the code `±1` and every cell seen but
     `(1, 1)` (XE15): `X_j / ‖X_j‖_S1 → ½ [[1, 1], [−1, −1]]`, so at every late phase end the
     unseen cell is decoded to the class of `−1`, where the kernel shows `+1`.

So the limit λ → 0⁺ of XE6 is reached along one run: hold each weight decay until the run
has settled near its trained fit, then lower it. The products at the phase ends grow without
bound, so the run has no limit; what converges is the margin for the norm, and the direction
where the product of largest margin is unique.

Example: one sample of two classes with the probes `±1` of a `1 × 1` product (the sample of
XE3), where `a = 1`, `N = 1`, `μ* = 2` and `1` is the only product of largest margin (XE10).
From `U_0 = [1 0]`, `V_0 = [0 1]` with `λ_j = 1 / (j + 1)` there are lengths of the phases
with `‖X_j‖_S1 → ∞`, `μ(X_j) / ‖X_j‖_S1 → 2` and `X_j / ‖X_j‖_S1 → 1`. The first phase has
`ρ_0 = 2 + 4 log 2` and the step `η_0 = 1 / (7 + 8 log 2) ≈ 0.08`.

Not covered. The lengths `T_j` are not explicit: each comes from the convergence of its
phase, for which WD8 gives no rate, so this is not a schedule that can be written before the
run. Inside a phase the margin is not followed. The steps shrink with the weight decay,
`η_j ≤ λ_j / (8 s b)`; a run with one fixed step, or with a weight decay that changes at
every step by a given rule, is not treated, and no schedule is shown to fail.

### XE19. The run without weight decay on one sample

XE17 leaves three questions about the run with λ = 0 and one fixed step: whether the norm of
the factors tends to infinity, whether the loss tends to `0`, and the direction of the
factors. This item answers them for the sample of XE3 and of the example of XE17: one sample
of two classes with the probes `±1` of a `1 × 1` product, labelled with the class of the
probe `1`. The factors `U` and `V` are rows of any width, the product is the number
`x = U Vᴴ`, the logits are `±Re x`, the margin is `μ(x) = 2 Re x`, and `μ* = 2`, attained
only at the product `1` (XE10).

The gradient is a real number, `G(x) = −w(Re x)` with the weight

```
w(r) = 2 / (1 + e^{2r}),
```

twice the probability of the wrong class. It is positive, at most `2` and decreasing, and
`w(0) = 1`. A step with λ = 0 is `U' = U + η w V`, `V' = V + η w U`, so the sum and the
difference of the factors are multiplied by numbers:

```
U' + V' = (1 + η w) (U + V),        U' − V' = (1 − η w) (U − V).
```

Write `a = ‖U + V‖²` and `b = ‖U − V‖²`. Then `Re x = (a − b) / 4`,
`‖U‖² + ‖V‖² = (a + b) / 2`, and

```
a' = (1 + η w)² a,        b' = (1 − η w)² b,        w = w((a − b) / 4).
```

Let η > 0 be any step, and let `U_0 + V_0 ≠ 0`, that is `a_0 > 0`.

1. The norm tends to infinity: `a_t → ∞`, so `‖U_t‖² + ‖V_t‖² ≥ a_t / 2 → ∞`.

   `a` does not decrease. If it stayed below a bound `l`, then `Re x_t ≤ a_t / 4 ≤ l / 4`
   and `w_t ≥ w(l / 4) > 0`, so `a_{t+1} ≥ (1 + η w(l / 4))² a_t` at every step, and `a`
   would grow geometrically.

2. The loss tends to `0`: `a_t − b_t → ∞`, so `Re x_t → ∞`, the margin `2 Re x_t → ∞`, and
   the cross-entropy, which is at most `e^{−μ(x_t)}` (XE4), tends to `0`.

   The ratio `a / b` does not decrease, since `(1 − s)² ≤ (1 + s)²` for `s = η w ≥ 0`:
   `a_{t₀} b_t ≤ a_t b_{t₀}` for `t ≥ t₀`. While `a_t ≤ b_t` the real part of the product is
   not positive, so `1 ≤ w_t ≤ 2` and `s_t` lies in `[η, 2η]`, where

   ```
   (1 + s)² ≥ (1 + 4η / (1 + 4η²)) (1 − s)².
   ```

   So the ratio grows geometrically, and `a_{t₀} > b_{t₀}` at some step `t₀`. From then on
   `a_t − b_t ≥ (1 − b_{t₀} / a_{t₀}) a_t`, which tends to infinity by 1.

3. The margin and the direction. Once `Re x_t ≥ η / 2`, `η w_t ≤ 2` and `(1 − η w_t)² ≤ 1`:
   from some step `T` on `b` does not grow, `b_t ≤ b_T`. So the margin for the weight norm

   ```
   μ(x_t) / (½ (‖U_t‖² + ‖V_t‖²)) = 2 (a_t − b_t) / (a_t + b_t) ≥ 2 − 4 b_T / a_t
   ```

   tends to `2`. Since `‖x‖_S1 ≤ ½ (‖U‖² + ‖V‖²)` (Lemma 1) and `μ(x) / ‖x‖_S1 ≤ μ* = 2`,

   ```
   μ(x_t) / ‖x_t‖_S1 → μ*,        x_t / ‖x_t‖_S1 → 1:
   ```

   the margin for the nuclear norm tends to the largest margin, the limit of XE6, and the
   direction of the product to the product of largest margin (XE10). The weight norm
   `½ (‖U_t‖² + ‖V_t‖²)` of the factors is asymptotically the nuclear norm of their product.

4. The exceptional starts. If `U_0 + V_0 = 0`, then `U_t + V_t = 0` at every step, for every
   real η: `Re x_t = −b_t / 4 ≤ 0`, and the loss is at least `log 2`, its value at the
   product `0`. So for η > 0 the loss tends to `0` exactly from the starts with
   `U_0 + V_0 ≠ 0`.

5. A rate. Let `E = e^{(a − b)/2}`, the exponential of the margin `μ(x) = (a − b) / 2`, so
   that `w = 2 / (1 + E)`. At a step with `a − b ≥ 2η` one has `η w ≤ 2`, so `b' ≤ b`, and
   `E ≥ 1`, so `E η w = 2 η E / (1 + E) ≥ η`. Then, by `a' − a ≥ 2 η w a` and `e^x ≥ 1 + x`,

   ```
   E' ≥ e^{(a' − b)/2} = E e^{(a' − a)/2} ≥ E (1 + (a' − a) / 2) ≥ E + E η w a ≥ E + η a.
   ```

   By 2 there is a step `T` from which `a − b ≥ 2η`, and `a` does not decrease, so for
   `t ≥ T`

   ```
   e^{μ(x_t)} ≥ e^{μ(x_T)} + η a_T (t − T) ≥ 1 + η ‖U_T + V_T‖² (t − T).
   ```

   So the cross-entropy, at most `e^{−μ(x_t)}` (XE4), is at most
   `1 / (1 + η ‖U_T + V_T‖² (t − T))`: it falls at least like `1 / t`. And
   `‖U_t‖² + ‖V_t‖² = (a_t + b_t) / 2 ≥ μ(x_t) ≥ log (1 + η ‖U_T + V_T‖² (t − T))`: the norm
   grows at least like `log t`.

No bound on the step is used, where statement 1 of XE17 needs `2 a N η < 1`, and the start
need not have full row rank: at width one the condition is `u_0 + v_0 ≠ 0`. Here
`−M_t = w_t [[0, 1], [1, 0]]`, and the exceptional starts are the stacks `[U_0; V_0]` that
the eigenvector `(1, 1)` of its positive eigenvalue annihilates from the left.

Example: with η = 1/10 from `U_0 = [1 0]`, `V_0 = [0 1]`, the example of XE17,
`‖U_t‖² + ‖V_t‖² → ∞`, the loss tends to `0`, `μ(x_t) / ‖x_t‖_S1 → 2` and
`x_t / ‖x_t‖_S1 → 1`, and from some step `T` on the loss is at most
`1 / (1 + ‖U_T + V_T‖² (t − T) / 10)`. From `u_0 = 1`, `v_0 = −1` of width one the loss is at
least `log 2` at every step.

Not covered. More than one sample: the gradient is then not a multiple of one matrix, the
sum and the difference of the factors do not decouple, and the three questions of XE17
remain open, except for samples no two of which share a row or a column (XE20). Rates beyond
5: a bound on the norm from above and on the loss from below, and the step `T`. One sample
with probes `±A` for a matrix `A` that is not `1 × 1`.

### XE20. The run without weight decay on samples in distinct rows and columns

XE19 follows the run with λ = 0 on one sample. This item follows it on several samples, no
two of which share a row or a column: there the samples do not interact, and each of them
is a copy of the sample of XE19.

Let the samples `s` of two classes lie at the cells `(i_s, j_s)` of an `m × n` product, read
with the code `±1` (XE8) and labelled `y_s = ±1`: the logits of the sample `s` are
`± Re X_{i_s j_s}`. Let no two samples lie in one row, and no two in one column. The factors
have any width.

The term of the sample `s` in the gradient is `−y_s w(y_s Re X_{i_s j_s}) E_{i_s j_s}`, with
the weight `w` of XE19 and the matrix unit `E_ij`, so

```
G(X) = − Σ_s y_s w_s E_{i_s j_s},        w_s = w(y_s Re X_{i_s j_s}).
```

The rows `i_s` are distinct, so the row `i_s` of `G V` is `−y_s w_s` times the row `j_s` of
`V`. The columns `j_s` are distinct, so the row `j_s` of `Gᴴ U` is `−y_s w_s` times the row
`i_s` of `U`. Write `u_s` for the row `i_s` of `U` and `v_s` for `y_s` times the row `j_s` of
`V`.

1. The rows of a sample follow the sign sample. `Re (u_s v_sᴴ) = y_s Re X_{i_s j_s}`, and a
   step with λ = 0 gives, by `y_s² = 1`,

   ```
   u_s' = u_s + η w_s v_s,        v_s' = v_s + η w_s u_s,        w_s = w(Re (u_s v_sᴴ)):
   ```

   the step of XE19 on the pair `(u_s, v_s)`. So along the run the pair `(u_s, v_s)` is the
   run of XE19 from its start, whatever the other rows are. The loss of the sample `s` is
   the loss of the sign sample at the product `u_s v_sᴴ`, so the loss along the run is the
   sum of the losses of these runs.

2. The norm tends to infinity. Let η > 0 be any step. If `u_s + v_s ≠ 0` at the start for
   some sample `s`, then `‖u_s‖² + ‖v_s‖² → ∞` (XE19, statement 1), and
   `‖U‖² + ‖V‖² ≥ ‖u_s‖² + ‖v_s‖²`.

3. The loss tends to `0` exactly when no pair cancels. If `u_s + v_s ≠ 0` at the start for
   every sample, every term of the loss tends to `0` and every gap `2 Re (u_s v_sᴴ)` tends
   to infinity (XE19, statement 2): the loss tends to `0`, and the margin, the least gap,
   tends to infinity. If `u_s + v_s = 0` at the start for one sample, its term is at least
   `log 2` at every step, for every real η (XE19, statement 4), and so is the loss.

4. A rate. If `u_s + v_s ≠ 0` at the start for every sample, the term of the sample `s` is
   at most `1 / (1 + η a_s (t − T_s))` from its step `T_s` on, with `a_s = ‖u_s + v_s‖² > 0`
   at that step (XE19, statement 5). With `T` the latest of the steps `T_s` and
   `c = η min_s a_s`, the loss is at most `|S| / (1 + c (t − T))` for `t ≥ T`: it falls at
   least like `1 / t`.

Theorem. Let samples of two classes with the code `±1` lie at cells no two of which share a
row or a column, and let η > 0 be any step. At every width, the run without weight decay
has `‖U_t‖² + ‖V_t‖² → ∞` from a start with `U_0[i_s] + y_s V_0[j_s] ≠ 0` for some sample.
Its cross-entropy tends to `0` if and only if `U_0[i_s] + y_s V_0[j_s] ≠ 0` for every sample;
then its margin tends to infinity, and from some step `T` on the loss is at most
`|S| / (1 + c (t − T))` for a `c > 0`. Otherwise the loss is at least `log 2` at every step.

So for these samples two of the three questions of XE17 are answered for one fixed step,
with the exceptional starts, and no bound on the step is used. For the `n` samples at the
diagonal cells of an `n × n` product, all labelled `+`, the condition is that no row of
`U_0 + V_0` vanishes.

Example: the two samples at the diagonal cells of a `2 × 2` product, with η = 1/10. With the
labels `(+, −)`, from `U_0 = 1`, `V_0 = 0` the norm of the factors tends to infinity, the
loss tends to `0` and the margin tends to infinity. With the labels `(+, +)`, from `U_0 = 1`,
`V_0 = −1` the loss is at least `log 2` at every step.

Not covered. The direction of the product and the margin for the nuclear norm. The modes
`‖u_s + v_s‖²` of the samples tend to infinity one by one, and the direction of the product
depends on their ratios, that is on the rates, which XE19 does not give. If the ratios tend
to `1`, as the growth `≈ 2 log t` of one mode suggests, the product of the samples at the
diagonal cells, divided by its nuclear norm, tends to `1/n` times the Gram matrix of the
normalized rows of `U_0 + V_0`, a product of largest margin; this is not proved. Samples
that share a row or a column: there the rows of different samples interact, and the three
questions remain open. Codes of more than two classes.

### Lean

`src/Transformer/Grokking/FlatSpectrum/Training/`, modules `Logits`, `CrossEntropy`,
`CrossEntropyBounds`, `CrossEntropyRun`, `Margin`, `MaxMargin`, `MaxMarginLimit`, `SeenCells`,
`AnswerCode`, `SeenMargin`, `UnitBall`, `MarginDirection`, `MirroredCode`, `CharacterCode`,
`MirroredUnique`, `SeenDirection`, `UnseenRows`, `LargeDecay`, `NearestCodeword`,
`ClassProbes`, `MeanLoss`, `LeastNorm`, `NuclearMin`, `NuclearMinLimit`, `NoDecay`,
`QuarterCode`, `SignPath`, `SignKernel`, `SignDirection`, `SignUnique`, `RealEntries`,
`SignReal`, `SignRealRun` and `NuclearMinDirection`, namespace
`Transformer.Grokking.FlatSpectrum.Training`, proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| XE1 | `ceLoss`, `ceGrad`, `pairing_sum_left`, `pairing_ceGrad`, `ceLoss_first_order`, `continuous_probability_pairing`, `continuous_ceGrad`, `ceLoss_nonneg`, `ceLoss_zero`, `analyticAt_pairing`, `analyticAt_ceLoss`; one sample: `sum_probability_mul_le_log_weightedExp`, `crossEntropy_first_order`, `crossEntropy_zero` | The samples and the classes are finite types `S`, `C`; the probes are `A : S → C → Matrix m n ℂ` and the labels `y : S → C`. `crossEntropy` is that of `Transformer.Landscape.Neural`, and `probability` and `weightedExp` are those of `Transformer.Grokking.Geometry` and `Transformer.Grokking.GradientEvidence`. Real-analytic in the entries is the hypothesis of WD8. The first case of probes is `codeProbe` (XE8). The second is `classProbe` (module `ClassProbes`), with `pairing_classProbe`, `frobeniusSq_classProbe`, `classProbe_apply_of_ne`, `gap_classProbe`, `IsDecayedMin.row_eq_zero_of_classProbe` and `exists_tendsto_decayRun_classProbe`: the samples read the rows `row : S → m`, the classes are the column type `n`, and `e_ic` is `Matrix.single i c 1`. |
| XE2 | `abs_pairing_le_mul_sqrt`, `exists_forall_frobeniusSq_le`, `frobeniusSq_le_sq_of_le_mul_sqrt`, `pairing_ceGrad_le`, `frobeniusSq_ceGrad_le`, `neg_le_pairing_of_first_order`, `neg_le_pairing_ceGrad`, `ceLoss_le_add`; one sample: `sum_mul_exp_sub_le_of_mem_Icc`, `log_weightedExp_le`, `crossEntropy_le_add` | `‖A s c‖ ≤ a` is `frobeniusSq (A s c) ≤ a ^ 2` with `a ≥ 0`, and `‖G X‖ ≤ 2 a N` is `frobeniusSq (G X) ≤ (2 a N) ^ 2`. Hoeffding's lemma is Mathlib's `ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc`, applied to the weights `p` as a `PMF` on the classes; it is stated for values in `[a, b]` with the constant `(b − a)² / 8`. |
| XE3 | `exists_tendsto_decayRun_ceLoss`, `exists_tendsto_decayRun_ceLoss_isDecayedMin`, `exists_pos_ae_tendsto_decayRun_ceLoss_isDecayedMin`, `exists_isDecayedMin_ceLoss`, `exists_seq_isDecayedMin_ceLoss`; the mean: `decayStep_smul`, `decayRun_smul`, `decayedLoss_const_mul`, `isDecayedMin_const_mul_iff`, `exists_tendsto_decayRun_mean_ceLoss_isDecayedMin`; example: `signProbe`, `frobeniusSq_signProbe_le`, `signWitness_radius` | Instances of `exists_tendsto_decayRun`, `exists_tendsto_decayRun_isDecayedMin`, `exists_pos_ae_tendsto_decayRun_isDecayedMin` and `exists_isDecayedMin_of_card_le`, with their forms of the hypotheses. `N` and `\|C\|` are `Fintype.card S` and `Fintype.card C`. The mean is the loss `fun X => (card S)⁻¹ * ceLoss A y X` with the gradient `fun X => (card S)⁻¹ • ceGrad A y X`, for a nonempty `S`; of the statements of the section only the convergence of the run is restated for it, the others follow by `isDecayedMin_const_mul_iff` and `decayRun_smul`. |
| XE4 | `gap`, `margin`, `gap_smul`, `margin_smul`, `margin_le_gap`, `nonempty_wrong`, `le_margin`, `exists_gap_eq_margin`, `frobeniusSq_le_nuclearNorm_sq`, `gap_le_mul_nuclearNorm`, `crossEntropy_le_of_le_gap`, `ceLoss_le_of_le_gap`, `ceLoss_le_mul_exp`, `log_le_ceLoss` | `margin` is the infimum over the pairs `(s, c)` with `c ≠ y s`; the statements that use it as a minimum take `[Nonempty S] [Nontrivial C]`. The upper bounds are stated for a number `μ` below every gap. |
| XE5 | `IsDecayedMin.ceLoss_add_le`, `IsDecayedMin.ceLoss_le_smul`, `IsDecayedMin.log_le_mul_exp`, `IsDecayedMin.sub_log_le_gap`; trained products: `IsNuclearMin`, `exists_mul_eq_weightDecay_eq`, `IsDecayedMin.isNuclearMin`, `IsNuclearMin.exists_isDecayedMin`, `isNuclearMin_iff`, `forall_seq_isNuclearMin`, `exists_isNuclearMin_ceLoss`, `exists_seq_isNuclearMin_ceLoss`, `IsNuclearMin.ceLoss_add_le`, `IsNuclearMin.ceLoss_le_smul`, `IsNuclearMin.log_le_mul_exp`, `IsNuclearMin.sub_log_le_gap`; example: `nuclearNorm_one_fin_one`, `exists_isDecayedMin_signProbe`, `exists_isDecayedMin_signProbe_log_le` | The margin of the competitor enters as a number `μ'` below its gaps, and the conclusion is stated for each gap. `κ` is written `card S * (card C − 1)`. The last bound asks `log(2κ) ≤ ‖X‖_S1 μ'`; its example is a trained fit of the sample of XE3 at one of the weight decays `1 / (j + 1)`, found in `MaxMarginLimit` after the norms are shown to grow (`exists_isDecayedMin_signProbe_log_le`). A trained product is `IsNuclearMin L λ X`, that is `∀ Y, L X + λ * nuclearNorm X ≤ L Y + λ * nuclearNorm Y`. Its trained fit of width `n` has the column type `n` itself, `U : Matrix m n ℂ` and `V : Matrix n n ℂ`. `forall_seq_isNuclearMin` takes the statement as a predicate `P` of the sequence of products. |
| XE6 | `nuclearNorm_pos_of_margin_pos`, `tendsto_nuclearNorm_atTop`, `eventually_lt_margin_div`, `maxMargin`, `margin_le_maxMargin`, `margin_div_le_maxMargin`, `tendsto_margin_div_nuclearNorm`; the second form: `maxMargin_pos`, `inv_maxMargin_le_nuclearNorm`, `margin_eq_maxMargin_iff_inv_smul`, `isLeast_nuclearNorm_of_one_le_margin`, `tendsto_nuclearNorm_div_margin`; trained products: `tendsto_nuclearNorm_atTop_of_isNuclearMin`, `eventually_lt_margin_div_of_isNuclearMin`, `tendsto_margin_div_nuclearNorm_of_isNuclearMin`, `tendsto_nuclearNorm_div_margin_of_isNuclearMin`, `tendsto_margin_div_nuclearNorm_of_widths`; no weight decay: `ceLoss_pos`, `tendsto_ceLoss_smul_atTop`, `isGLB_range_ceLoss`, `not_exists_forall_ceLoss_le`, `not_isNuclearMin_ceLoss_zero`, `not_isDecayedMin_ceLoss_zero`, `not_isCompact_setOf_flatLoss_zero_le`; example: `exists_seq_isDecayedMin_signProbe`, `margin_signProbe_one_pos` | The trained fits of a sequence have one finite column type; the statements `…_of_isNuclearMin` are about sequences of trained products and have no width, and `tendsto_margin_div_nuclearNorm_of_widths` is statement 3 for fits `U_j : Matrix m (Fin (r j)) ℂ` of the widths `r j`. `maxMargin` is the supremum `sSup` of the margins on `{‖X‖_S1 ≤ 1}`; that it is attained is `exists_margin_eq_maxMargin` (XE10). `margin_div_le_maxMargin` holds at `X = 0` too, where Lean's quotient is `0`. "Some product has a positive margin" is the hypothesis `0 < margin A y X₀`. The second form is `IsLeast (nuclearNorm '' {X \| 1 ≤ margin A y X}) (maxMargin A y)⁻¹`; `Z / μ*` is `(maxMargin A y)⁻¹ • Z`, and the equivalence of the solutions takes `0 < maxMargin A y`. `‖X_j‖_S1 / μ(X_j)` is Lean's quotient, `0` where the margin is `0`. "No minimum" is `¬ ∃ X, ∀ Y, ceLoss A y X ≤ ceLoss A y Y`, with the infimum as `IsGLB (Set.range (ceLoss A y)) 0`. `not_isCompact_setOf_flatLoss_zero_le` is for every loss `L` and level `c ≥ L 0`, on `EuclideanSpace ℂ ((m ⊕ n) × k)` with `m` and `k` nonempty. The two citations are not formalized. |
| XE7 | `fitLoss_eq_of_seen`, `pairing_eq_of_seen`, `ceLoss_eq_of_seen`, `IsDecayedMin.isMinOn_nuclearNorm`, `not_isMinOn_smul`, `IsDecayedMin.ne_smul_of_not_isMinOn`, `IsDecayedMin.ne_smul_of_isFlat`, `exists_tendsto_decayRun_ceLoss_ne_smul`; a large λ: `isDecayedMin_zero`, `IsDecayedMin.mul_eq_zero_of_lt`, `isDecayedMin_zero_ceLoss`, `IsDecayedMin.mul_eq_zero_of_ceLoss`, `frobeniusSq_mask_zero_sub_le`, `isDecayedMin_zero_fitLoss`, `IsDecayedMin.mul_eq_zero_of_fitLoss` | "Reads only Ω" is the hypothesis `∀ X Y, (∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) → L X = L Y`, and "the probes vanish off Ω" is `∀ s c i j, (i, j) ∉ Ω → A s c i j = 0`. A minimum of M from Ω is `IsMinOn nuclearNorm S K` as in R2. For a large λ, `‖G 0‖ ≤ γ` is `frobeniusSq (G 0) ≤ γ ^ 2` with `0 ≤ γ`, the norm `‖·‖_S2`; the sharp threshold has the operator norm, as in WD1. The zero factorization is stated at every finite column type. |
| XE8 | `codeProbe`, `pairing_single`, `frobeniusSq_single`, `pairing_codeProbe`, `codeProbe_apply_of_notMem`, `frobeniusSq_codeProbe`, `gap_codeProbe`, `margin_codeProbe_pos`; the decoder: `norm_sub_sq_of_norm_eq_one`, `re_conj_mul_le_iff`, `re_conj_mul_lt_iff`, `pairing_codeProbe_le_iff`, `gap_codeProbe_pos_iff`, `norm_sub_lt_of_margin_pos`, `eventually_norm_sub_lt`; examples: `signCode`, `norm_signCode`, `injective_signCode` | The cells are `cell : S → m × n` and the seen cells are `Set.range cell`; `φ(c) e_ij` is `Matrix.single i j (φ c)`. "The class with the largest logit has the nearest codeword" is stated for two classes: the logit of `t` is at least that of `c` exactly when `φ t` is at least as near to the cell as `φ c`. "Decoded to the label" is that the codeword of the label is strictly nearer than every other. |
| XE9 | `gap_eq_of_seen`, `margin_eq_of_seen`, `exists_pos_eventually_add_le_margin_div`, `not_tendsto_inv_nuclearNorm_smul`, `exists_pos_eventually_add_le_margin_div_of_codeProbe`, `not_tendsto_inv_nuclearNorm_smul_of_codeProbe`; trained products: `exists_pos_eventually_add_le_margin_div_of_isNuclearMin`, `not_tendsto_inv_nuclearNorm_smul_of_isNuclearMin`, `not_tendsto_inv_nuclearNorm_smul_codeProbe_of_isNuclearMin` | δ is chosen before the sequence, for one finite column type of the factors, or for trained products in the statements `…_of_isNuclearMin`. The normalized product is `(nuclearNorm X_j)⁻¹ • X_j`. That the trained fits grow without bound is `tendsto_nuclearNorm_atTop` (XE6). The examples are the `1 × 2` kernel `(1, 1)` with the code `±1` on its first cell, and `cubicKernel 3` with `ψ` on the cell `(0, 0)`. |
| XE10 | `continuous_gap`, `nuclearNorm_le_of_tendsto`, `nuclearNorm_inv_smul_le`, `margin_inv_smul`, `exists_subseq_tendsto_inv_smul`, `margin_eq_maxMargin_of_tendsto`, `exists_margin_eq_maxMargin`, `margin_eq_maxMargin_of_tendsto_inv_smul`, `tendsto_inv_smul_of_unique`, `tendsto_inv_margin_smul`, `tendsto_inv_margin_smul_of_unique`; trained products: `margin_eq_maxMargin_of_tendsto_inv_smul_of_isNuclearMin`, `tendsto_inv_smul_of_unique_of_isNuclearMin`, `tendsto_inv_margin_smul_of_isNuclearMin`; example: `gap_signProbe`, `maxMargin_signProbe`, `eq_one_of_margin_signProbe` | The direction is `(nuclearNorm X)⁻¹ • X`, which is `0` for `X = 0`. The compact set is `{X \| ∀ i j, ‖X i j‖ ≤ 1}` (`isCompact_setOf_norm_apply_le`), which contains the unit ball, and the dual form of the norm is `exists_isContraction_trace_eq`. Statement 2 is for a convergent sequence of directions; its subsequences are taken by `exists_subseq_tendsto_inv_smul`, which holds for every sequence of matrices. "One product has the margin `μ*`" is the hypothesis `∀ Z, nuclearNorm Z ≤ 1 → margin A y Z = maxMargin A y → Z = Z₀`. In statement 4 the scaled fit is `(margin A y X)⁻¹ • X`, which is `0` where the margin is `0`. |
| XE11 | `IsMirroredCode`, `IsMirroredCode.injective`, `pairing_eq_sum_re`, `gap_codeProbe_add_gap`, `margin_codeProbe_le_mul_re`, `margin_codeProbe_eq`, `norm_apply_eq_one_of_surjective`, `margin_codeProbe_le_mul_re_apply`, `card_mul_margin_codeProbe_le`, `card_mul_card_eq_of_isFlat`, `margin_codeProbe_mul_nuclearNorm_le`, `maxMargin_codeProbe_eq`, `margin_inv_smul_codeProbe`, `eq_mul_of_re_conj_mul`, `mul_re_eq_margin_of_margin_eq`, `apply_cell_eq_of_margin_eq`, `eq_inv_smul_of_margin_codeProbe_eq`, `tendsto_inv_smul_codeProbe`, `tendsto_inv_margin_smul_codeProbe`, `tendsto_inv_smul_codeProbe_iff`; the code `ψ`: `cos_two_pi_mul_div_le`, `one_sub_cos_two_pi_div_pos`, `one_sub_cos_two_pi_div_lt_two`, `norm_stdAddChar_sub_sq`, `isMirroredCode_stdAddChar`, `tendsto_inv_smul_cubicKernel_iff`; trained products: `tendsto_inv_smul_codeProbe_iff_of_isNuclearMin`, `tendsto_inv_margin_smul_codeProbe_of_isNuclearMin`, `tendsto_inv_smul_cubicKernel_iff_of_isNuclearMin`; not mirrored: `quarterCode`, `norm_quarterCode`, `injective_quarterCode`, `not_isMirroredCode_quarterCode`, `gap_quarterCode`, `margin_quarterCode`, `quarterLimit`, `quarterLimit_apply`, `nuclearNorm_quarterLimit_le`, `margin_quarterLimit`, `margin_quarterCode_le`, `maxMargin_quarterCode`, `eq_quarterLimit_of_margin`, `tendsto_inv_smul_quarterCode`, `not_tendsto_inv_smul_quarterCode`; the code `±1`: `isNuclearMin_of_pairing`, `IsNuclearMin.of_le`, `inv_nuclearNorm_smul_smul`, `pairing_codeProbe_signCode`, `ceLoss_signCode_congr`, `margin_signCode_congr`, `probability_signCode_sub`, `pairing_ceGrad_smul_signCode`, `isNuclearMin_smul_signCode`, `isNuclearMin_smul_of_re_eq`, `tendsto_signDecay`, `nuclearNorm_eq_of_re_eq`, `nuclearNorm_pos_of_signCode`, `exists_ne_margin_eq_maxMargin_signCode`, `exists_seq_isNuclearMin_tendsto_signCode`, `exists_seq_isNuclearMin_not_tendsto_signCode`, `exists_seq_isDecayedMin_not_tendsto_signCode`; its pair: `signKernel`, `signLabel`, `signTwin`, `signKernel_apply`, `isFlat_signKernel`, `nuclearNorm_signKernel`, `signTwin_re`, `signTwin_ne`, `signTwin_eq_vecMulVec`, `frobeniusSq_signTwin`, `nuclearNorm_signTwin_le`, `exists_ne_margin_eq_maxMargin_signKernel`, `exists_seq_isDecayedMin_not_tendsto_signKernel`; the real parts: `log_one_add_exp_midpoint_lt`, `log_one_add_exp_midpoint_le`, `crossEntropy_signCode`, `ceLoss_signCode_eq`, `ceLoss_signCode_midpoint_lt`, `IsNuclearMin.re_apply_eq_of_signCode`; real factors: `HasRealEntries` with `.zero`, `.one`, `.add`, `.sub`, `.smul`, `.conjTranspose`, `.mul`, `.sum`, `.fromCols`, `.of_tendsto`, `.decayStep`, `hasRealEntries_decayRun`, `hasRealEntries_of_tendsto_decayRun`, `signScale`, `signDecay_signScale`, `signScale_nonneg`, `signScale_pos`, `isNuclearMin_signScale_smul`, `IsNuclearMin.re_eq_signScale_mul`, `IsNuclearMin.eq_signScale_smul`, `tendsto_inv_smul_signCode_of_real`, `one_le_sqrt_two`, `hasRealEntries_signKernel`, `hasRealEntries_ceGrad_signCode`, `exists_tendsto_decayRun_signCode_of_real`, `exists_isDecayedMin_signCode_real`, `tendsto_inv_smul_signCode_of_real_factors`, `exists_seq_isDecayedMin_real_tendsto_signCode`; example: `isMirroredCode_signCode` | `IsMirroredCode φ g` is the conjunction of `∀ c, ‖φ c‖ = 1`, `∀ t c, c ≠ t → 2 * g ≤ ‖φ t - φ c‖ ^ 2` and `∀ t, ∃ c c', c ≠ t ∧ c' ≠ t ∧ φ c + φ c' = ((2 * (1 - g) : ℝ) : ℂ) * φ t`. "Every cell is seen" is `Function.Surjective cell`, and "`K` has no zero entry" is `∀ i j, K i j ≠ 0`. The flat kernel is `IsFlat K c₀` with `0 < c₀`. `ψ` is `ZMod.stdAddChar` as a function on `ZMod p`, with the gap `1 - cos (2 * π / p)`. In `tendsto_inv_smul_cubicKernel_iff` the samples are a finite nonempty type with cells `cell : S → ZMod p × ZMod p` and the labels `(cell s).1 ^ 3 + (cell s).1 * (cell s).2`. Statement 4 is proved cell by cell in `apply_cell_eq_of_margin_eq`. The limit at margin one is `g⁻¹ • K`. The code `(1, i)` is `quarterCode` on `Bool`, with `true` for the codeword `1`; `(1 − i) / √2` is `quarterLimit`, the `1 × 1` matrix `(√2)⁻¹ • (1 − i)`; the kernel is the `1 × 1` matrix `1`, and the limit excluded is `(nuclearNorm 1)⁻¹ • 1`. For the code `±1`, "every cell is seen once" is `Function.Bijective cell`; `K'` is a matrix with `(K' i j).re = (K i j).re`, `nuclearNorm K' ≤ nuclearNorm K` and `K' ≠ K`; the second product of 6 is `(nuclearNorm K)⁻¹ • K'`; "does not converge" is `∀ Z, ¬ Tendsto … (𝓝 Z)`; the trained fits of 7 have the width of the columns; the certificate `isNuclearMin_of_pairing` is stated for every convex loss with its gradient; `G(t K) = −β K` is stated as the pairing with every `H`, a sum over the samples; the labels of `signKernel` are `signLabel`, `false` at the cell `(1, 0)`. "Real entries" is `HasRealEntries X`, `∀ i j, (X i j).im = 0`; `t(λ)` is `signScale c₀ lam`; statement 8 is stated at every sample `s`, for the cell `cell s`; `tendsto_inv_smul_signCode_of_real` is about trained products with real entries, so about fits of any widths, and `tendsto_inv_smul_signCode_of_real_factors` about fits of one column type; the run of 9 has the bounds of `exists_tendsto_decayRun_ceLoss_isDecayedMin` at `a = 1`, and the start of `exists_isDecayedMin_signCode_real` is `fromCols 1 0`, `fromCols 0 1`. The examples are the code `±1` with the `1 × 1` kernel `1`, and `cubicKernel 3` with all nine cells seen and the code `ψ`. |
| XE12 | `IsDecayedMin.row_eq_zero`, `IsDecayedMin.col_eq_zero`, `IsDecayedMin.apply_eq_zero_of_codeProbe`, `IsDecayedMin.pairing_codeProbe_eq_zero`, `IsDecayedMin.ceLoss_heldOut_eq`, `exists_tendsto_decayRun_ceLoss_eq_zero` | The row has `∀ j, (i, j) ∉ Ω` as in CW8, whose `row_eq_zero_of_isMinOn` and `col_eq_zero_of_isMinOn` are applied to the minimum of XE7. For the answer code the seen cells are `Set.range cell`, and "no sample sees the row `i`" is `∀ s, (cell s).1 ≠ i`. The held-out samples are a finite type `S'` with cells `cell' : S' → m × n` and any labels `y'`. "The softmax is uniform" is the statement that every logit is zero. |
| XE14 | `hasDerivAt_comp_add_smul_of_analytic`, `gradient_flatLoss_of_analytic`, `factors_iterate_gradientStep_of_analytic`, `exists_pos_tendsto_decayRun_of_analytic`; the logits `±(Re x)²`: `sqWeight`, `sqWeight_nonneg`, `sqWeight_le_half`, `sqLogitLoss`, `sqLogitGrad`, `sqLogitLoss_eq_crossEntropy`, `sqLogitLoss_nonneg`, `sqLogitLoss_zero`, `sqLogitGrad_zero`, `pairing_sqLogitGrad`, `hasDerivAt_sqLogitLoss`, `continuous_sqLogitGrad`, `analyticAt_sqLogitLoss`, `exists_pos_tendsto_decayRun_sqLogit`, `sqLogitLoss_lt_zero`, `not_first_order_sqLogit`, `frobeniusSq_sqLogitGrad_le`, `neg_le_pairing_sqLogitGrad`; the ball: `tendsto_zero_of_tendsto_frobeniusSq`, `re_sq_le_frobeniusSq`, `frobeniusSq_sqLogitGrad_mul_le`, `frobeniusSq_decayStep_sqLogit_le`, `frobeniusSq_decayRun_sqLogit_le`, `tendsto_decayRun_sqLogit_zero`, `one_sub_smul_lift_posDef_sqLogit`, `not_isDecayedMin_sqLogit_zero`, `exists_tendsto_decayRun_sqLogit_not_isDecayedMin`; in `Training`: `hasDerivAt_decayedLoss_of_hasDerivAt`, `hasGradientAt_flatLoss_of_hasDerivAt` | Modules `Scope/SmoothLoss`, `Scope/SquaredLogit`, `Scope/SquaredLogitBall` and `Scope/SquaredLogitRun`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. "`G` is the gradient of `L`" is the hypothesis `∀ X H, HasDerivAt (fun t : ℝ => L (X + t • H)) (pairing (G X) H) 0`; real-analytic and bounded below are the hypotheses of WD9, and the conclusion of statement 1 is that of `exists_pos_tendsto_decayRun`. The two lemmas in `Training` are the parts of `hasDerivAt_decayedLoss` and `hasGradientAt_flatLoss` that do not use convexity. The loss is `sqLogitLoss X = log (1 + exp (-(2 * (X 0 0).re ^ 2)))` on `Matrix (Fin 1) (Fin 1) ℂ`, equal to `crossEntropy (fun c : Bool => (signCode c).re * (X 0 0).re ^ 2) true`, and `sqLogitGrad X = (-(4 * r * sqWeight r)) • 1`. "Not convex" is `¬ ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y`. The factors are `Matrix (Fin 1) k ℂ` for a finite `k`; "converges to the zero factorization" is `Tendsto … (𝓝 (0, 0))`; "not a decayed minimum" is `¬ IsDecayedMin sqLogitLoss lam 0 0` for `lam ≤ 1/4`, at every `k`. That the ball has interior points and the comparison with WD10 are not formalized. The start of the examples is `!![1/8, 0]`, `!![1/8, 1/8]` with `lam = 1/4` and `η = 1`. |
| XE15 | `exists_unit_conj_mul`, `detCertificate`, `frobeniusSq_fin_two`, `isFlat_detCertificate`, `pairing_detCertificate`, `frobeniusSq_add_two_mul_norm_det_le`, `norm_mul_sub_le_norm_det`, `sq_sub_add_sq_add_le`, `eq_ofReal_of_norm_le_of_le_re`; the three cells: `threeCells`, `threeLabels`, `signFill`, `gap_threeCells`, `margin_threeCells_le`, `le_margin_threeCells`, `margin_threeCells_le_nuclearNorm`, `nuclearNorm_signFill_le`, `margin_signFill`, `maxMargin_threeCells`, `eq_signFill_of_margin`; the limit: `cubicKernel_two`, `tendsto_inv_smul_threeCells`, `eventually_re_unseen_neg`, `norm_sub_signCode_lt`, `eventually_unseen_wrong`; every small weight decay and the run: `exists_pos_forall_re_unseen_neg`, `exists_pos_forall_unseen_wrong`, `exists_pos_forall_tendsto_decayRun_unseen`, `exists_tendsto_decayRun_unseen_wrong` | Modules `Scope/TwoByTwo`, `Scope/UnseenCell`, `Scope/UnseenCellLimit` and `Scope/UnseenCellRun`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The samples are `Fin 3` with the cells `threeCells = ![(0, 0), (0, 1), (1, 0)]` and the labels `threeLabels s = signLabel (threeCells s)`, those that `signKernel` shows; `cubicKernel_two` is `cubicKernel 2 = signKernel`. "Decoded to the class of `−1`" is `‖X 1 1 − signCode false‖ < ‖X 1 1 − signCode (signLabel (1, 1))‖`, where `signLabel (1, 1) = true`. The statements of 4 take a sequence of trained products `IsNuclearMin`; the threshold `λ₀` of 5 is obtained from them by contradiction and is not explicit. The bounds of XE3 in 6 are written with their numbers: `12 * log 2 ≤ λ ρ`, `2 η (λ² + 36) ≤ λ` and `η (λ + 6 + 6 ρ) ≤ 1`; the width is any finite type. The run that exists has the step `min (λ / (2 (λ² + 36))) (1 / (λ + 6 + 6 ρ))` for `λ = λ₀ / 2` and `ρ = 4 + 12 log 2 / λ`. The bound of 2 is proved as an inequality; the equality is not needed. |
| XE16 | `isNuclearMin_signCode_iff`, `nuclearNorm_eq_signScale_mul_of_isNuclearMin`; the kernel of signs: `signRot`, `im_eq_of_det_bounds`, `exists_eq_add_smul_signRot`, `signCotwin`, `nuclearNorm_signCotwin_le`, `nuclearNorm_add_smul_signRot_le`, `isNuclearMin_signKernel_iff`; the run: `exists_tendsto_decayRun_signKernel` | Modules `Scope/SignSet`, `Scope/SignSegmentCore` and `Scope/SignSegment`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. `t(λ)` is `signScale c₀ lam`, and "`Re X = t(λ) K`" is `∀ i j, (X i j).re = signScale c₀ lam * (K i j).re`; the hypotheses are those of XE11 (`Function.Bijective cell`, `K` shows the labels, `IsFlat K c₀`). For the kernel of signs the cells are `id` on `Fin 2 × Fin 2` with the labels `signLabel`, `J` is `signRot`, `K − i J` is `signTwin` and `K + i J` is `signCotwin`; the number `s` of the segment is `a` in Lean, and the point is `signScale 2 lam • signKernel + ((a : ℂ) * Complex.I) • signRot`. `im_eq_of_det_bounds` is the real arithmetic of "⟹", in the imaginary parts and `N = ‖det X‖`. The bounds of XE3 are written with their numbers: `16 * log 2 ≤ λ ρ`, `2 η (λ² + 64) ≤ λ` and `η (λ + 8 + 8 ρ) ≤ 1`; the width is any finite type. |
| XE17 | `not_tendsto_decayRun_ceLoss_zero`, `exists_lt_frobeniusSq_decayRun_ceLoss_zero`, `decayRun_add`, `surjective_stack_decayRun`, `frequently_lt_frobeniusSq_decayRun_ceLoss_zero`; every real λ (WD8): `exists_tendsto_decayRun_of_forall_le`, `exists_tendsto_decayRun_isDecayedMin_of_forall_le` | Module `Scope/NoDecayRun`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`; the statements for every λ are in `BoundedRun` and `Convergent`, namespace `Transformer.Grokking.FlatSpectrum.Training`. "No limit" is `¬ Tendsto (decayRun (ceGrad A y) η 0 U₀ V₀) atTop (𝓝 (U, V))` for every pair `(U, V)`. "Some product has a positive margin" is `0 < margin A y X₀`, with `[Nonempty S] [Nontrivial C]`; `N` is `Fintype.card S`, and the conditions are `η * (2 * a * card S) < 1` and `η * (2 * a * card S + 2 * (card S * a ^ 2) * ρ) < 1`. Statement 3 is `∃ t, ρ < …` and `∃ᶠ t in atTop, ρ < …`; `ρ*(η)` and the `lim sup` are not stated. The two citations are not formalized. |
| XE18 | `abs_nuclearNorm_sub_le`, `continuous_nuclearNorm`, `continuous_margin`, `continuousAt_margin_div_nuclearNorm`, `tendsto_inv_smul_of_tendsto_margin_div`; one phase: `phaseStep`, `phaseStep_pos`, `exists_tendsto_decayRun_phaseStep`, `exists_seq_of_forall_exists`; the run: `phaseRun`, `phaseProduct`, `exists_tendsto_margin_div_phaseProduct`; the direction: `exists_tendsto_inv_smul_phaseProduct`, `exists_tendsto_inv_smul_phaseProduct_codeProbe`, `exists_tendsto_inv_smul_phaseProduct_cubicKernel`, `exists_eventually_phaseProduct_unseen_wrong` | Modules `Scope/MarginContinuity`, `Scope/PhaseStep`, `Scope/PhasedRun` and `Scope/PhasedDirection`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. `phaseStep S C a lam W` is `η(λ, W)` with `N` and `\|C\|` as `Fintype.card S` and `Fintype.card C`. `phaseRun A y a lam T U₀ V₀ j` is the pair `W_j`, defined by recursion on `j` with the step `phaseStep S C a (lam j) W_j`, and `phaseProduct A y a lam T U₀ V₀ j` is `X_j`. The weight decays are any `lam : ℕ → ℝ` with `∀ j, 0 < lam j` and `Tendsto lam atTop (𝓝 0)`, and "there are lengths" is `∃ T : ℕ → ℕ`. Statement 2 is the conjunction of `Tendsto (fun j => nuclearNorm (X_j)) atTop atTop` and the limit of `margin A y (X_j) / nuclearNorm (X_j)`, Lean's quotient, which is `0` where the product is `0`. The Lipschitz bound is `\|‖X‖_S1 − ‖Y‖_S1\| ≤ √(card m * frobeniusSq (X − Y))`. `tendsto_inv_smul_of_tendsto_margin_div` is the statement about every sequence of matrices; its hypothesis on `Z₀` is that of XE10. For the code probes `a = 1`. "Decoded to the class of `−1`" is written as in XE15, at the cell `(1, 1)` of `phaseProduct`. `exists_seq_of_forall_exists` is the dependent choice of the phases. The bound `η_j ≤ λ_j / (8 s b)`, the numbers of the example's first phase and "the run has no limit" are not stated; the examples are the sample of XE3, `cubicKernel 3` with all nine cells seen, and the three seen cells of the kernel of signs, along `λ_j = 1 / (j + 1)` from the start `fromCols 1 0`, `fromCols 0 1`. |
| XE19 | `signWeight`, `signWeight_pos`, `signWeight_le_two`, `signWeight_anti`, `signWeight_zero`, `mul_signWeight_le_two`; the system: `IsSignSystem`, `IsSignSystem.snd_nonneg`, `IsSignSystem.fst_pos`, `IsSignSystem.fst_mono`, `IsSignSystem.tendsto_fst_atTop`, `IsSignSystem.mul_le_mul_of_le`, `IsSignSystem.exists_lt`, `IsSignSystem.tendsto_sub_atTop`, `IsSignSystem.exists_forall_snd_le`; the run: `ceGrad_signProbe`, `decayStep_signProbe`, `pairing_one_mul_conjTranspose`, `frobeniusSq_add_frobeniusSq`, `signProduct`, `decayRun_signProbe_add`, `decayRun_signProbe_sub`, `isSignSystem_decayRun_signProbe`, `margin_signProbe`, `margin_signProduct`, `tendsto_modes_decayRun_signProbe`, `tendsto_frobeniusSq_decayRun_signProbe_atTop`; the limits: `tendsto_margin_signProduct_atTop`, `tendsto_ceLoss_signProduct`, `tendsto_margin_div_frobeniusSq_signProduct`, `tendsto_margin_div_signProduct`, `tendsto_inv_smul_signProduct`, `log_two_le_ceLoss_signProduct`, `tendsto_ceLoss_signProduct_iff`; the rate: `IsSignSystem.exp_add_le_exp_succ`, `IsSignSystem.exists_exp_ge`, `exists_ceLoss_signProduct_le`, `exists_log_le_frobeniusSq_decayRun_signProbe` | Modules `Scope/SignSystem`, `Scope/SignGap`, `Scope/SignRun`, `Scope/SignRunLimit` and `Scope/SignRate`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The sample is `signProbe` with the label `fun _ => true`; the factors are `U₀ V₀ : Matrix (Fin 1) k ℂ` for any finite `k`, the run is `decayRun (ceGrad signProbe fun _ => true) η 0 U₀ V₀`, and `signProduct η U₀ V₀ t` is `x_t` as a `1 × 1` matrix. `Re x` is `pairing 1 X`. `IsSignSystem η a b` states the two recursions for sequences `a b : ℕ → ℝ`; the run satisfies it with `a t = frobeniusSq (U_t + V_t)` and `b t = frobeniusSq (U_t − V_t)`. Statements 1 to 3 have the hypotheses `0 < η` and `U₀ + V₀ ≠ 0`; statement 4 is `log_two_le_ceLoss_signProduct` for every real `η` and the equivalence `tendsto_ceLoss_signProduct_iff` for `0 < η`. The quotient `margin / nuclearNorm` is Lean's, which is `0` where the product is `0`. The geometric growth of the ratio while `a ≤ b` is a private lemma. Statement 5 for the system is `∃ T, b T ≤ a T ∧ ∀ t, T ≤ t → exp ((a T - b T) / 2) + η * a T * (t - T) ≤ exp ((a t - b t) / 2)`; for the run it is `∃ T, 0 < a_T ∧ ∀ t, T ≤ t → ceLoss _ _ (signProduct η U₀ V₀ t) ≤ 1 / (1 + η * a_T * (t - T))` and the same with `Real.log (1 + η * a_T * (t - T)) ≤ frobeniusSq U_t + frobeniusSq V_t`, where `a_T` is `frobeniusSq (U_T + V_T)`. The remark on the eigenvector `(1, 1)` is not stated. The examples are the start `fromCols 1 0`, `fromCols 0 1` with `η = 1/10` and the starts `(1, −1)` and `(1, 1)` of width one; the witness of the system is `signWitness`, the first mode from `a_0 = 1`, `b = 0` with the step `1`. |
| XE20 | `classSign`, `classSign_mul_self`, `ceGrad_term_signCode`, `ceGrad_signCode`, `ceGrad_signCode_mul_apply`, `conjTranspose_ceGrad_signCode_mul_apply`, `diagCell`, `injective_diagCell_fst`, `injective_diagCell_snd`; the rows: `rowOf`, `pairing_one_rowOf`, `rowOf_decayStep_signCode`, `rowOf_decayRun_signCode`, `frobeniusSq_rowOf_le`, `rowOf_add_ne_zero_iff`, `pairing_neg_left`, `crossEntropy_signCode_mul`, `ceLoss_decayRun_signCode`; the run: `tendsto_frobeniusSq_decayRun_signCode_atTop`, `log_two_le_ceLoss_decayRun_signCode`, `tendsto_ceLoss_decayRun_signCode_iff`, `tendsto_margin_decayRun_signCode_atTop`, `diagCell_one_add_zero_ne`, `diagCell_one_add_neg_one_eq`, `exists_ceLoss_decayRun_signCode_le` | Modules `Scope/SeparateCells`, `Scope/SeparateCellsRows`, `Scope/SeparateCellsRun` and `Scope/SeparateCellsRate`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The samples are `codeProbe signCode cell` with `cell : S → m × n` and the labels `y : S → Bool`; `y_s = ±1` is `classSign (y s)`. "No two samples in one row" is `Function.Injective fun s => (cell s).1`, and the same with `.2` for the columns. `E_ij` is `Matrix.single i j 1`; the gradient is `ceGrad_signCode`, and the two statements on its rows are stated for the entries of `G X * V` and of `(G X)ᴴ * U`. The row `i` of a factor is `rowOf i U`, a matrix with one row: `u_s` is `rowOf (cell s).1 U` and `v_s` is `classSign (y s) • rowOf (cell s).2 V`. Statement 1 is `decayStep` and `decayRun` of the sign sample of XE19 at these rows, equal to the rows of the step and of the run; the loss of a sample is `crossEntropy_signCode_mul`, an equality with `ceLoss signProbe`, and the sum is `ceLoss_decayRun_signCode`, with `signProduct` of XE19. "`u_s + v_s ≠ 0` at the start" is `∃ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b ≠ 0`, and its negation is `∀ b, _ = 0`. The run is `decayRun (ceGrad (codeProbe signCode cell) y) η 0 U₀ V₀`; the loss and the margin are those of the product of its two factors. The statement on the margin assumes `Nonempty S`. Statement 4 is `∃ T c, 0 < c ∧ ∀ t, T ≤ t → ceLoss _ _ _ ≤ Fintype.card S / (1 + c * (t - T))`, under `Nonempty S`; the step `T` and the constant `c` of its proof are not in the statement. The remark on the diagonal cells and the text of "Not covered" on the direction are not stated. `diagCell s` is `(s, s)` for `s : Fin 2`. The examples are those of the item. |

## Convex answer-loss recall

Added 2026-11-11. This construction separates the flat-spectrum obstruction
from a blanket obstruction to convex learning. R2 and AR3 concern nuclear-norm
completion of an answer table. They do not quantify over all convex models
that receive the dictionary as input. The construction below uses ordinary
answer cross-entropy, not an auxiliary key-matching or calibration loss.

### CR1. Model and objective

Let `κ` be a finite key set, `ν` a finite value set, and `S` the finite training
sample set. The learned parameters are **all entries** of `A ∈ ℝ^(κ×κ)`,
constrained by `A(q,k) ≥ 0` and `Σ_k A(q,k) = 1`. For an input query `q` and
input dictionary `D : κ → ν`, the class logits are

```
z_A(q,D,c) = Σ_k A(q,k) 1[D(k) = c].
L(A) = Σ_s (log Σ_c exp z_A(q_s,D_s,c) − z_A(q_s,D_s,D_s(q_s))).
```

The dictionary's values are input features; the label is its queried value.
The forward pass never receives that label separately. Categorical value
copying and the key identities are fixed. The matrix is learned directly,
without a factorization, learned embedding, attention softmax, or nuclear-norm
penalty. Its `m²` entries, for `m = |κ|`, share across all `|ν|^m` dictionaries;
there is no learned parameter per dictionary. This is an explicit, restricted
architecture, not a convex replacement for an arbitrary transformer.

Every logit is nonnegative and at most one. At `A = I` the logits are the
one-hot vector of `D(q)`. The constraints bound confidence: the best correct
softmax probability is `e / (e + |ν| − 1)`. Perfect argmax accuracy is possible;
zero cross-entropy is not claimed. Unlike unbounded separable logits in XE6,
this compact feasible set admits a finite minimum without weight decay.

### CR2. Joint convexity

The row-simplex domain is convex. For every fixed input, the complete logit
vector is linear in **all** entries of `A`. Ordinary softmax cross-entropy is
convex in its logits, by the first-order inequality of XE1. Therefore `L` is
jointly convex, even on the unrestricted matrix space. Its restriction to
the stated convex domain is a convex optimization problem, including the
usual cross-entropy loss.

### CR3. What answer-only training identifies

Assume `|ν| ≥ 2`. Say the input dataset **separates keys** when

```
∀ q k, k ≠ q → ∃ s, q_s = q and D_s(k) ≠ D_s(q).
```

This is a checkable condition on supplied inputs. It mentions no learned
parameters, matching labels, test outputs, or successful training.

For any nonnegative logits `z` with `z_t ≤ 1`, each competitor's relative
logit `z_c − z_t` is at least that of the ideal one-hot vector `e_t`.
Consequently `CE(e_t,t) ≤ CE(z,t)`. If `z_t < 1`, a competitor exists and the
inequality is strict. Applying this to every sample gives `L(I) ≤ L(A)` for
every feasible `A`. Thus `I` actually attains the minimum.

If `L(A) ≤ L(I)`, every sample has the ideal loss and its target logit is one.
Nonnegative row mass then vanishes on any key whose input value differs from
the target. Separation supplies such a sample for every off-diagonal entry,
so all off-diagonal entries vanish. Row sums force `A(q,q) = 1`, hence `A = I`.
The optimum is unique, and every optimizer recalls **every** test dictionary,
including those never supplied during training, with a strict correct argmax.

### CR4. Small separating data and a negative control

An injective code `code : κ → (ι → ν)` constructs separating input data:
use dictionary `D_r(k) = code(k,r)` at each coordinate `r`, queried at every
key. The code is only a way to construct data; it is not an additional input
to the model. In the formal example, four keys are Boolean pairs, and the two
coordinate dictionaries give eight training samples. The theorem applies to
all sixteen Boolean dictionaries, not only those two. More generally binary
codes allow `m b` samples when `m` keys have distinct `b`-bit codes.

Without separation, convexity does not guarantee generalization. For two
keys, let sample `s ∈ Bool` query key `s` in the constant dictionary `D_s = s`.
Both labels occur. The matrix that swaps the two keys is feasible and attains
the global minimum of the same ordinary answer CE. On the test dictionary
`D = id`, it gives a strictly wrong argmax. This is a proved counterexample,
not a failed optimization experiment.

### CR5. Causal MQAR

For the aligned pairs of Zoology §4 / Appendix, use

```
z_A(x,i,v) = Σ_{j<i} A(query_i,key_j) 1[value_j = v].
```

This is also linear in the complete parameter matrix. After CR3's training,
`A = I`, and it equals the existing paired causal attention operator exactly.
The repository's `paired_attention_solves_mqar` theorem therefore proves that
it produces every prior-binding answer on every distinct-key stream, at
any length. It returns the zero vector when no key matches, as specified by
`expectedPairedAnswer`; no learned null-class decision is claimed. Training
and stream values may use different alphabets because only key matching is
learned. The theorem assumes the key/value pairing shift and token identities
are already supplied. Repeated-key overwrite semantics and raw-token parsing
are outside this result. The training theorem is for complete dictionary
inputs; arbitrary partially observed MQAR training streams are not assumed
to satisfy it.

### CR6. Flat spectrum and the scope of the earlier obstruction

For a fixed scalar readout `φ : ν → ℂ`, define the model's output table by
`K_A(q,D) = Σ_c z_A(q,D,c) φ(c)`. CR3 implies `K_A(q,D) = φ(D(q))` at every
optimizer. Thus AR2 applies: if `Σ_c φ(c) = 0`,

```
K_A K_Aᴴ = (|ν|^(|κ|−1) Σ_c |φ(c)|²) I.
```

For a nonzero readout it has rank `|κ|` and a positive flat singular scale.
The model keeps the capacity needed to carry every key's binding, consistent
with AR4's width lower bound. It does not force a low-rank context summary.
For `φ = ±1` and two Boolean keys, the squared singular value is four.

If every `φ(c)` is nonzero, AR3 still says this very same trained kernel is
**not** a nuclear-norm minimum from any proper observation set. Both the
positive convex-learning theorem and the negative nuclear-completion theorem
are proved together. The nuclear-norm results have not been disproved; using
them to exclude all convex dictionary-aware learners would exceed their
hypotheses. The old calibration-based `Transformer.ConvexRecall` construction
also did not establish ordinary answer-CE training. The present construction
supplies that narrower missing result. It makes no claim to learn a general
flat arithmetic task such as the cubic kernel from two bare numeric tokens.

### CR7. What convergence is proved

The feasible matrix set is closed and contained in the finite coordinate
cube `[0,1]^(κ×κ)`, hence compact. `L` is continuous. Under CR3's separation
condition, **every feasible sequence `A_n` with `L(A_n) → L(I)` satisfies
`A_n → I`**: continuity makes each cluster point an optimizer, and uniqueness
forces every cluster point to be `I`. Since the limiting logit gap is one and
the input and class sets are finite, eventually **all** dictionary queries
have a strict correct argmax, uniformly over test dictionaries.

This is a theorem about minimizing sequences. It does not prove that a
specific projected-gradient, mirror-descent, SGD, or AdamW implementation has
convergent loss, nor give a convergence rate. That algorithmic obligation
remains separate; no optimizer run or empirical result is reported here.

### Lean

The modules under `src/Transformer/Grokking/FlatSpectrum/ConvexRecall/` prove
these statements with ordinary `Landscape.Neural.crossEntropy`:

| Item | Principal declarations |
| --- | --- |
| CR1–CR2 | `IsStochastic`, `logits`, `answerLoss`, `convex_stochastic`, `answerLoss_convex` |
| CR3 | `Separates`, `isMinOn_answerLoss_one`, `eq_one_of_answerLoss_le`, `trained_logits`, `trained_strict_argmax` |
| CR4 | `separates_of_injective_code`, `twoBitCode_injective`, `wrong_global_minimum_without_separation` |
| CR5 | `streamLogits_linear`, `trained_solves_mqar` |
| CR6 | `trained_outputKernel`, `trained_isFlat`, `trained_rank`, `trained_not_nuclear_minimum` |
| CR7 | `isCompact_stochastic`, `continuous_answerLoss`, `tendsto_one_of_loss_tendsto`, `eventually_all_strict_argmax` |

Reproduce with native Lean tooling:

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexRecall
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexRecallAxioms.lean
```

## Convex channels and capacity

The next extension learns generated outputs as well as routing. Flat spectrum
is used as a necessary capacity bound. It is not an arithmetic feature map,
an answer oracle, or a sufficient condition for generalization. The inputs
remain categorical query identities and supplied key/value bindings. The
earlier compact joint-embedding transformer replacement remains a separate,
unresolved requirement.

### CC1. A joint parameter domain

For finite query, key, value and output sets `Q,K,V,C`, directly train
`W(q,k,v,c)` subject to

```
W(q,k,v,c) ≥ 0,
Σ_c W(q,k,v,c) = ρ(q,k), independent of v,
Σ_k ρ(q,k) = 1.

z_W(q,D,c) = Σ_k W(q,k,D(k),c),
L(W) = Σ_s CE(z_W(q_s,D_s), y_s).
```

`ρ` is the output marginal of `W`, not a separately multiplied parameter.
The implementation stores the joint tensor, with `|Q||K||V||C|` entries.
The forward receives input bindings and a query, never the answer or the
rule used to construct labels. The loss is ordinary log-sum-exp cross-entropy.
Summing class logits gives one for every dictionary: the common routing
mass lets each key use its own observed value. Logits lie in `[0,1]`, so the
confidence limitation of CR1 remains: ideal CE is `log(e+|C|−1)−1`.

### CC2. Joint convexity

Nonnegativity and affine equalities make the entire feasible set convex.
For fixed inputs, `z_W` is linear in every trained entry. XE1's logit
convexity therefore makes ordinary answer CE jointly convex in routing and
output generation. A product of separately learned routing and decoding
matrices is neither needed nor used.

### CC3. Copying is a strict subfamily

Every stochastic matcher `A` from CR1 embeds as
`W(q,k,v,c) = A(q,k) 1[c=v]`, with exactly the old forward logits. The enlarged
family also contains, for every categorical rule `g : K×V → C`,

```
W_g(q,k,v,c) = 1[k=q] 1[c=g(q,v)],
z_Wg(q,D,c) = 1[c=g(q,D(q))].
```

This constructs a parameter witness; `g` is not part of the model's input
features or forward code. For two Boolean keys, `g(q,v)=not v` produces
`true` on an all-false dictionary. No old copying matrix can do so.

### CC4. General categorical two-input capacity

With a single context slot, `W_f(q,(),v,c)=1[c=f(q,v)]` represents every
finite two-input classification table. This is a representation theorem.
It does not infer the labels of unobserved pairs. With several slots the
family sums their contributions; it is not a universal model of arbitrary
multi-slot interactions. New interactions would require additional input
features and a fresh capacity/identification analysis.

### CC5. What answer-only training can identify

Assume `|C|≥2`, labels `y_s=g(q_s,D_s(q_s))`, and the following finite-data
condition, called `SeparatesAtoms`:

```
∀ q,k,v,c, (k≠q or c≠g(q,v)) →
  ∃ s, q_s=q and D_s(k)=v and y_s≠c.
```

Every incorrect routing/output atom must be active in some training input
whose ordinary answer contradicts it. This condition refers to the actual
data and target rule, not to learned parameters or training success. It
requires value coverage beyond the simpler copy-only condition CR3.

As in CR3, `W_g` attains the ideal CE sample by sample. Equal aggregate
loss forces target mass one at each observed sample. Any active atom
predicting another class has zero weight. Separation eliminates all
incorrect atoms; unit mass forces the remaining entries of `W` to equal
`W_g`. Therefore every global optimizer generates `g(q,D(q))` with a strict
correct argmax on every test dictionary. The Lean witness uses all Boolean
two-key dictionaries with negated answer labels and verifies separation.
This theorem is conditional on reaching a global optimum; it is not a
proof of convergence of the native optimization loop.

### CC6. Use flat spectrum as the bound

For a fixed class readout `φ:C→ℂ` and any supplied family `D_x`, the actual
model output has the exact factorization

```
F_W(q,x) = Σ_c z_W(q,D_x,c) φ(c) = A_W B,
A_W(q,(k,v)) = Σ_c W(q,k,v,c) φ(c),
B((k,v),x) = 1[D_x(k)=v].
```

These are task-independent categorical binding features. For any flat
target `T` of squared singular value `σ²>0` and rank `r=min(|Q|,|X|)`, R3 gives

```
‖T−F_W‖_F² ≥ σ² (r−|K||V|).
```

Exact representation requires `|K||V|≥r`. This is an upper feature count,
not a claim that all those features are independent. A further factorization
through width `d` has the stronger corresponding bound `σ²(r−d)`. The bound
concerns scalar-output approximation, not CE or classification error; AR5
is the separate sign/argmax result. It need not impose a positive error
floor once capacity exceeds rank. The feasible one-slot Boolean AND table,
read out as `false↦1,true↦−1`, is `[[1,1],[1,−1]]`: a full-rank flat kernel
with `σ²=2`, realized at width two. No arithmetic features are supplied.

### CC7. Flat capacity does not identify missing labels

The two tables with rows `[[1,1],[1,−1]]` and `[[1,1],[−1,1]]` are both
flat with `σ²=2`. Their categorical labels agree on the entire first query
row and disagree on the second. Both feasible channel tensors globally
minimize the same ordinary CE on observations from the first row. Thus even
known flatness, full rank and sufficient capacity leave two incompatible
held-out predictions. The Lean counterexample proves feasibility, both
flatness equations, both global-optimality statements, observed agreement
and held-out disagreement together. It does not rule out learning with
additional structure or identifying examples such as CC5.

### CC8. An exact training oracle

For a gradient tensor `G`, choose `decode(q,k,v)` minimizing `G(q,k,v,c)` over
classes. Choose `key(q)` minimizing `Σ_v min_c G(q,k,v,c)` over keys. The
vertex

```
S(q,k,v,c) = 1[k=key(q)] 1[c=decode(q,k,v)]
```

globally minimizes `Σ_qkvc G(q,k,v,c) W(q,k,v,c)` on the feasible set. Indeed,
each key's cost is at least its class-minimum sum times its routing mass;
the routing masses are nonnegative and sum to one. This bound is attained
by the selected vertex. The finite search costs `O(|Q||K||V||C|)` and does
not fit a nonconvex inner model. Convex updates `(1−η)W+ηS`, `0≤η≤1`, remain
feasible. These facts justify the native Frank-Wolfe step; no convergence
rate for that implementation is asserted by the Lean result.

### Lean

The eight modules under `Grokking/FlatSpectrum/ConvexChannel/` prove CC1–CC8.
Principal results are `convex_feasible`, `answerLoss_convex`,
`generates_absent_value`, `eq_generator_of_answerLoss_le`, `trained_logits`,
`flat_error_bound`, `flat_width_necessary`, `flat_capacity_does_not_identify`
and `linear_oracle`. Every theorem with premises has a satisfying example.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexChannel
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
```

## Balancedness as a capacity certificate (added)

This extension isolates what the balancedness update in WD14.6 supplies for
the convex-model question. Let `U : C^(m × k)`, `V : C^(n × k)` and assume
`Uᴴ U = Vᴴ V`. No loss, stationary point, optimizer or successful training is
assumed in B1–B2. The statement concerns a factorization of a matrix; applying
it to a model requires identifying the matrix that the model represents.

### B1. Identities from balance alone

The trace of the common Gram matrix gives `||U||_F² = ||V||_F²`. Also

```
(U Vᴴ)ᴴ (U Vᴴ) = (V Vᴴ)²,
||U Vᴴ||_* = tr(V Vᴴ) = (||U||_F² + ||V||_F²)/2,
rank(U Vᴴ) = rank U = rank V = rank [U; V].
```

The positive semidefinite matrix `V Vᴴ` is a root of the product's Gram
matrix, which proves the nuclear-norm identity. The first three ranks follow
from equality of Gram matrices; the stack has Gram matrix `2 Uᴴ U`, which
has the rank of `U`.

If `n > 0`, the stack rank is at most `m < m+n`. Thus an exactly balanced
start cannot satisfy WD5's full-stack row-rank hypothesis. This only limits
that sufficient theorem; it does not imply that all balanced starts fail or
that discrete gradient steps preserve balance. The stationary points in
WD14 satisfy the hypothesis by its separate stationarity argument.

### B2. Use flatness as a bound on actual rank

For `IsFlat K c`, `c > 0`, put `r₀ = min(m,n)`. R3 and B1 give

```
c (r₀ - rank U) ≤ ||K - U Vᴴ||_F².
```

The subtraction here is in the real numbers. Allocated width can exceed
the actual rank: adding zero columns does not remove this obstruction.
Equivalently, an error budget `||K-U Vᴴ||_F² ≤ ε` requires
`rank U ≥ r₀ - ε/c`; this is a necessary capacity condition.
The theorem does not identify unseen labels, certify optimization, or
guarantee that a particular representation attains the lower bound.

### B3. Balance is neither a convex constraint nor an optimum certificate

Even for real scalars, the balanced set `{(u,v) : u² = v²}` is not convex:
`(1,1)` and `(1,-1)` belong to it, but their midpoint `(1,0)` does not.

For a nonzero stationary example with a convex loss, take
`K = (3/2) I₂`, `L(X) = ||X-K||_F²/2`, `λ = 1/2` and
`U = V = diag(1,0)`. The actual product gradient is
`G = U Vᴴ-K = diag(-1/2,-3/2)`. Both factor gradients vanish:
`G V + λ U = 0` and `Gᴴ U + λ V = 0`. The factors are balanced and
have rank one. The target is flat with `c = 9/4`, so B2 gives an error
lower bound of `9/4`. Nevertheless their decayed loss is `7/4`, strictly
above the `5/4` attained by the balanced same-width competitor `U = V = I₂`.
Thus even nonzero balanced stationarity does not certify optimality.
This squared-loss witness refutes a loss-independent inference; it is not
a cross-entropy experiment or a replacement for the existing ordinary CE.

The existing convex-channel learner optimizes the joint coefficients directly.
It has no separately trained `U,V` whose balancing would repair its predictions.
An exact refactorization preserving the represented matrix also preserves
every linear readout of that matrix. B1–B3 support checking effective capacity
and the hypotheses of a factorized optimizer. The minibatch oracle bias,
query-specific missing records and fixed logit scale discussed in the
experiment README need their own remedies and proofs.

Lean: `BalancedFactorization.lean` proves B1–B2, including the stack-rank
identity and full-row-rank obstruction. `Training/Scope/BalanceObstruction.lean`
proves B3 and checks the numerical matrix witness, its actual rank, flat target,
stationarity, both losses and the resulting failure of `IsDecayedMin`.

## Repairing minibatch and full-batch failures (added)

### MF1. Exact singleton CE oracles can fail forever

Take two Boolean inputs `D_b(false) = false`, `D_b(true) = b`, with answer
`b`. For any current joint channel `W`, one sample's ordinary CE gradient is
`G_b(k,v,c) = 1[v=D_b(k)] (softmax(logits_W(D_b))_c - 1[c=b])`.
The Lean first-order inequality checks this tensor against ordinary CE.
At the observed value, class `b` minimizes the residual. All other value
rows have zero gradient, so every key has the same best singleton cost.
The first-key tie rule is therefore an exact linear minimization oracle.

After its initial step of size one, this rule places all routing mass on
the constant slot. Every subsequent convex update with another first-key
vertex leaves the separating slot at zero, regardless of label order and
retained choices on unobserved rows. Both inputs receive the same logits.
Any deterministic Boolean readout is exactly 50% accurate on the two inputs.
Yet the feasible vertex routing to the second slot with decoder `id` assigns
the correct class logit one and the other class zero on both inputs.

`ConvexChannel/Minibatch/Gradient.lean` and `Failure.lean` prove this exact
two-key/two-class instance. The native frozen regression uses three classes
and the larger causal adapter and exhibits the same constant-key failure.
No universal stochastic-convergence conclusion follows from an exact oracle
for each noisy minibatch objective.

### MF2. Retain sample gradients, not only decoder choices

Store one gradient per observed training example and replace it on revisit.
Replacing entry `g_i` by `g'_i` updates the aggregate by `g'_i-g_i`; repeated
visits do not increase an example's weight. If all entries are refreshed at
one model state, this is its full empirical gradient. With minibatches,
entries may instead describe different earlier states.

For MF1, even two arbitrary evaluation states `W_0,W_1` suffice: the
separating vertex minimizes `G_false(W_0)+G_true(W_1)`. Every constant-answer
vertex has strictly larger cost, because the probability assigned to the
other class is strictly positive. Thus this memory removes the demonstrated
singleton tie. `Minibatch/Memory.lean` proves this statement and the exact
replacement identity. It does not prove general convergence with stale gradients.

For a sparse implementation, subtract the class-independent softmax background
from each sample gradient. On a nonempty value type, a tensor
`B(k,v,c)=1[v=D(k)] beta` has linear cost exactly `beta` against every feasible
joint channel. Therefore subtracting this baseline preserves oracle comparisons
and FW gaps. The constant-cost identity is also checked in `Memory.lean`.

### MF3. Shared joint parameters address a separate full-batch failure

The earlier full-batch recall model fitted observed labels, but each query had
an independent decoder. An unseen query/value pair could keep its default even
when another query had learned that value. This is a structural identification
problem, not a nonzero empirical optimization gap.

Fix an input-only slot permutation `align(q)` and train one joint tensor `A`.
Its expansion is `W(q,k,v,c)=A(align(q)(k),v,c)`. Equivalently it reads the
aligned dictionary `D(align(q)^(-1)(j))`. The expansion preserves all feasibility
constraints and is linear in `A`. Ordinary CE remains convex, including after
multiplication of logits by any fixed scale. Equal aligned dictionaries have
equal predictions under different queries. `ConvexChannel/Shared.lean` proves
these statements. A positive fixed scale preserves a model's argmax ordering;
learning the scale jointly is not covered by this convexity result.

The intended causal alignment leaves lag slots fixed and relabels each
token-indexed successor/presence block relative to the visible final query.
All output classes and value decodings remain learned. This explicitly shares
information across queries; flat spectrum alone does not imply that sharing
is appropriate for an arbitrary task. B1–B3 remain capacity and scope checks,
and do not supply optimization or missing-label identification by themselves.

### MF4. Prefer sharing without discarding query-specific capacity

Mandatory sharing is a restriction, not a general repair. For two Boolean input
slots `D(a,b)=(a,b)`, every shared additive logit satisfies the rectangle identity
`z(0,0)+z(1,1)=z(0,1)+z(1,0)`. Applying it to the difference between two class
logits rules out strict XOR classification on all four inputs, even without
feasibility constraints. The full query-indexed domain, with visible query `a`,
represents XOR by selecting the second slot and learning `decode(a,b)=a xor b`.
This is a capacity witness, not an arithmetic feature in the stress model.

Retain the original convex domain and compute its exact query-wise linear
oracle `V`. An aligned shared feasible candidate `S` can replace it whenever
`<G,S> <= <G,V>`. Since `V` minimizes the linear objective on the full domain,
the replacement is also a minimizer. Convex combinations with either candidate
remain feasible, and the ordinary CE forward map stays linear in all joint
coordinates. `ConvexChannel/Adaptive.lean` proves the XOR obstruction, the
unrestricted witness, and this oracle preference. It does not guarantee that
the resulting optimizer discovers every rule or identifies arbitrary unseen
labels. A forward-only learner may similarly use a shared consistent solution
when one exists and otherwise retain the query-specific consistency family.

### MF5. Complete unseen rows without changing the oracle cost

The unrestricted fallback can still keep a default on an unobserved query/value
row even when other queries have learned that value. If `G(q,k,v,c)=0` for every
class on such a row, replace its decoder choice by a pooled choice. This leaves
the linear cost of every routing vertex unchanged, so an exact oracle remains
exact. The completed vertex and its convex updates remain feasible in the
original domain. No product of separately trained factors is introduced.

For any finite retained residual sum
`G(q,k,v,c)=sum_s 1[q_s=q and D_s(k)=v] r_s(c)`, a row absent from every consumed
sample is zero, regardless of the states at which residuals were evaluated.
`ConvexChannel/Completion.lean` proves this support fact, cost invariance and
oracle preservation. The missing-row mask must be derived from consumed
training inputs. It must not mark an observed row just because its current
gradient happens to be small. The shared decoder's correctness on an unseen
row is additional structure; this theorem supplies no arbitrary-label guarantee.

The native completion uses the pooled oracle's decoder for the same aligned
key, only for values absent from that query's retained observations. It does
not replace observed choices, alter oracle costs or read held-out answers.

## Causal features and convex capacity

### CFt1. Add features without discarding the old model

For any additional input-only slots `L`, embed a channel on `K` into one on
`K + L` by setting the new slots' joint weights to zero. Nonnegativity and
value-independent routing mass remain true; the total routing mass stays one.
If the value alphabet also grows, first pull back the old rows through a fixed
retraction: old values retain their rows, and each new unused value reads any
fixed old row. Feasibility and predictions on the embedded old inputs are
preserved. The key extension then applies on this common enlarged alphabet.
On every input, the new slots contribute zero, so the old logits and ordinary
CE are exactly preserved. The enlarged domain remains convex and the logits
remain linear in every learned coordinate, for any fixed feature extractor.
Consequently the optimal empirical loss cannot increase. This is a capacity
claim; a particular optimizer or tie rule can still generalize differently.

### CFt2. Query-specific coordinates do not remove additive obstructions

Fix one query and expose two independent Boolean slots. Even an unconstrained
query-specific channel obeys the rectangle identity of MF4 on that query.
Therefore it cannot strictly classify XOR on all four inputs. Adding more
iterations or increasing a fixed positive logit scale cannot remove that
identity. This small counterexample is not claimed to be a proof about every
feature of the native stress adapter; the latter has additional causal slots.

### CFt3. Counts add interactions without learned-factor products

Append the occurrence count of each token in the visible prefix. The extractor
receives only that prefix and the vocabulary, with no labels, arithmetic
operation or task identifier. A count is between zero and the context bound,
so it is a finite categorical value in a bounded-context experiment. The
usual joint tensor can select that slot and learn any output rule of its count.
In particular, the count of `true` represents two-bit XOR with a unit margin,
strictly extending the separated-slot model of CFt2. Parity appears only in
the mathematical parameter witness; the executable learns its decoder from
ordinary labels and does not compute a parity answer in its feature map.

`ConvexChannel/Counts.lean` proves the embedding, exact predictions and loss,
convex CE, fixed-query obstruction, context bound and count-rule witness.
The generic finite categorical flat-spectrum bound CC6 still applies to the
enlarged feature alphabet. Counts do not identify labels at unseen counts,
and do not assert sufficiency for order-sensitive tasks or modular division.

## Forward-only channel learning

This extension gives a finite categorical learner for the realizable
generated-recall family of CC3 and CC5. It uses ordinary answer labels and
constructs a global minimum of the same answer cross-entropy as CC1. No
gradient, backward pass, contrastive surrogate or optimizer convergence
assumption is needed. It does not solve arbitrary noisy or multi-key
classification in the larger convex channel family.

### CF1. Record answers and eliminate conflicting keys

Group the training dictionaries and labels by query. For each query, try
the candidate keys in a supplied finite order. For one candidate `k`, start
with an empty partial table from values to classes, then scan its examples
from left to right:

1. Read the input value `v=D(k)` and the ordinary answer `y`.
2. If `v` has not been seen, record `v -> y`.
3. If its recorded answer is `y`, retain the table.
4. Otherwise eliminate `k` permanently.

Return the first surviving key and its recorded decoder. Complete unseen
values with a fixed, supplied default class. If every candidate is
eliminated, return `none`. The target key and rule are never algorithm
inputs. `learn` scans the data once per attempted key; it does not require
keeping every candidate's table live simultaneously.

The Lean state is `Option (V -> Option C)`: the outer `none` is an
eliminated candidate; the inner `none` is an unobserved value. `step`,
`scan`, `trainKey` and `learn` are executable definitions. Functional
updates are a mathematical table representation, not a claim of constant
time memory access in the compiled Lean implementation.

### CF2. Exact scan invariant and explicit failure

Let a decoder extend a table when it agrees with every recorded entry.
After each observation, the surviving decoders are exactly the previous
ones that predict this answer from this candidate value:

```
Allows(step(state,v,y), f) <-> Allows(state,f) and f(v)=y.
Allows(trainKey(k,samples), f) <-> every (D,y) in samples has f(D(k))=y.
```

Thus any realizable candidate survives; every returned decoder fits all
observations. Two samples with the same candidate value and different
labels eliminate that candidate. The learner returns `none` exactly when
no supplied key has a decoder fitting every observation. This is failure
of exact single-key realizability, not infeasibility of the convex domain
or proof that its best approximate classifier is poor.

### CF3. A constructed global ordinary-CE optimum

Suppose every query's row is realizable by some enumerated key and decoder.
The computed learner returns `(k_q,f_q)` and hence the feasible channel

```
W(q,k,v,c) = 1[k=k_q] 1[c=f_q(v)].
```

All training samples have unit target logits. CC5's sample lower bound
then proves global minimality of ordinary answer CE over **all** feasible
joint channels, including nondeterministic ones. The minimum CE retains
the bounded-logit value `log(e+|C|-1)-1` per example, which is positive
when there are at least two classes.
This is finite completion of the specified learner, rather than an
assumption that an iterative optimizer reached a minimum.

`learnedChannel` uses an explicit fallback to make the mathematical tensor
total. Its optimality theorem assumes realizability and proves a successful
return on every row, so the fallback is unused. The executable `learn`
continues to report failure explicitly. The finite CE sample family must
be included in the rows actually supplied to the learner.

### CF4. Conditions identifying held-out answers

For the target `g(q,D(q))`, assume:

* the correct key is enumerated and all supplied labels follow this rule;
* every value at the correct key appears in the query's training row;
* for every wrong candidate key, some two observations have the same value
  at that key and different answer labels.

One collision per wrong key eliminates it. Coverage at the true key then
fixes its complete decoder. The learner therefore returns exactly
`(q,g(q,·))`, and the assembled channel equals `generator g`. It answers
every dictionary, including unobserved combinations, correctly. The
theorem does not infer labels for uncovered true-key values.

The Boolean witness trains negation from two of the four dictionaries
per query: the first has `false` at the queried key and `true` elsewhere;
the second has `true` at both keys. Their labels are `true` and `false`.
Each wrong key is constant across these examples and therefore conflicts.
The learned model produces `true` on the held-out all-false dictionary,
although that value is absent from the input. Reading its actual outputs
with `false -> 1, true -> -1` over all four dictionaries gives a flat
rank-two kernel with squared singular value four (`bool_learned_flat`).
Separate inconsistent data produce explicit failure. Flat spectrum supplies
no extra identifying information to this procedure.

### CF5. Counted work and scope of the efficiency claim

`countedScan` increments a counter for every executed call to `step`.
`countedLearn` accumulates these counters while running the actual key
selection. Its returned result equals `learn`, and its count is at most

```
keys.length * samples.length
```

for each query row. If each key is listed once, this proves an abstract
`O(|K| N)` bound on candidate observations over `N` total training examples.
Each successful result can
be represented by one key and a categorical decoder, with no dense
`|Q||K||V||C|` floating-point parameter tensor. An array-backed port could
perform constant-time record accesses and retain at most one candidate
table per query during the sequential search. Those runtime and storage
claims need a separate native implementation and measurements; the Lean
counter does not model allocations, lookup cost or wall-clock time.

The theorems under `ConvexChannel/Forward/` establish the invariant,
success/failure characterization, exact CE optimality, identification,
held-out Boolean witness and operation count. No claim is made here about
the proposed soft multiplicative update for inconsistent data.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
```

### CF6. Persistent minibatches and exact boundary invariance

Retain a partial record and an elimination marker for every `(query,key)`
candidate. Each batch updates every candidate using CF1's `scan`. A snapshot
selects the first surviving key in the fixed candidate order and supplies the
same default on unseen values, without modifying any records. In particular,
neither selecting a key nor predicting a default discards information needed
by later batches.

For any initial candidate state, scanning `xs ++ ys` equals scanning `xs` and
then continuing on `ys`. Induction over batches proves exact state equality
with scanning their concatenation. The executable minibatch learner therefore
returns exactly the same key and decoder, or the same failure, as `learn` on
the flattened observations. This holds for empty batches, unequal batch sizes,
partial coverage and inconsistent labels. It requires the same ordered examples,
candidate order and default; it does not assume realizability. Assembled channel
tensors are equal too, so CF3's CE optimality and CF4's identification conclusions
transfer when their hypotheses hold for the combined observations.

A concrete Boolean example exposes premature selection: the first batch has
dictionary `(false,false)` and label `true`; a snapshot selects key `false`.
The next nonempty batch has dictionary `(false,true)` and label `false`.
Key `false` is now contradictory. Key `true` survives and decodes negation,
using the first batch's `false -> true` entry as well as the new entry.
Keeping only the first snapshot would lose that alternative's earlier evidence.

An array implementation needs `|Q||K||V|` optional categorical entries and
`|Q||K|` elimination flags, independent of the number of examples. It visits
all candidates, rather than CF5's early stopping between candidates. Its
runtime and memory constants are measured separately in
`experiments/convex_channel_minibatch`; batch-boundary invariance alone is
not a claim that streaming is faster or that an arbitrary prefix generalizes.

## Observable features determine unseen predictions

This extension supplies a sufficient identification condition in addition to
CC6's capacity bound. It neither removes the CC7 counterexample nor assumes
that a flat spectrum determines unobserved labels. All channels and losses
are the unfactorized model and ordinary cross-entropy of CC1-CC2.

### O1. Training labels determine observed logits

Let the finite training data be `(q_s,D_s,y_s)`. Assume a feasible reference
channel `A` realizes these labels at unit logit scale:

```
z_A(q_s,D_s,c) = 1[c=y_s]                 for every observed s and class c.
```

There is no condition on unobserved answers in this hypothesis. Every
feasible channel has nonnegative logits summing to one. Each example's CE
is therefore bounded below by the CE of its target vector, and `A` attains
all those bounds. Equality of total CE forces equality on every sample.
For at least two classes, equality forces target mass one. Nonnegativity
and unit mass then force all other coordinates to zero. Consequently every
feasible global CE optimizer agrees with `A` on the complete observed
logit vectors, not only on their argmax. This proves identification on the
training observations; it does not assert that an optimization algorithm
has reached that optimum.

### O2. An input-only certificate determines an unseen prediction

Define categorical features `x(D)_(k,v)=1[D(k)=v]`. For a proposed unseen
input `(q,D)`, supply finite real coefficients `a_s` satisfying

```
a_s = 0 whenever q_s != q,
x(D) = Σ_s a_s x(D_s).
```

This is an explicit finite system of linear equations involving only
observed inputs and the proposed test input. Neither training answers,
test answers nor trained parameters occur in the certificate. Coefficients
may be negative. Direct expansion of the actual channel forward gives

```
z_W(q,D,c) = Σ_s a_s z_W(q_s,D_s,c)        for every channel W and class c.
```

Thus agreement on observed logits implies agreement on this unseen input.
Together with O1, every feasible global optimizer has the same unseen
logits as the feasible reference channel. Equivalently, the inferred vector
is the explicit combination `Σ_s a_s 1[c=y_s]` of observed target vectors.
To call its argmax a correct ground-truth answer requires that the target
rule on the test input is represented by that reference channel. Training
data cannot exclude an arbitrary ground-truth rule outside the model.

The certificate is sufficient, not necessary. Nonnegativity and the shared
routing constraints can identify additional predictions, as in CC5. Failure
to find this certificate alone is not an impossibility theorem. A query with
no observed samples has no nonzero categorical test vector in the indicated
span, which is consistent with CC7's unseen-query ambiguity.

### O3. Error amplification is measurable

For any two channels, any class and any real error bound `ε`, suppose
the absolute difference of the observed logits is at most `ε` per sample.
The same input certificate and the triangle inequality give

```
|z_W(q,D,c) - z_A(q,D,c)| <= κ ε,       κ = Σ_s |a_s|.
```

This is a deterministic logit error bound. It is not a CE convergence rate
or a statistical sampling guarantee. Large coefficients can amplify error;
finding a small-norm reconstruction is therefore useful beyond merely
checking that a reconstruction exists. If a reference answer has a logit
margin greater than `2 κ ε`, this bound preserves its strict argmax (this
last elementary margin consequence is explanatory, not a separate Lean
theorem in this extension).

### O4. A genuinely unobserved input

Use two Boolean dictionary slots and a fixed query selecting the second.
Observe inputs `00,01,10` with labels `false,true,false`. The all-true input
`11` is absent. Coordinate by coordinate,

```
x(11) = -x(00) + x(01) + x(10).
```

The copying channel is a feasible witness for the observed labels. O1-O2
therefore force every feasible global CE optimizer to predict `true` on
`11`. This is a new input, not an added training example. The certificate's
amplification factor is three. Lean proves the absence of `11` from the
observations, the feature identity, realizability, the factor three and the
conclusion for every feasible optimizer. This example uses shared input
structure; flatness remains a separate rank/capacity bound.

The three `ConvexChannel/Observation/` modules prove O1-O4 and provide
satisfying examples for every conditional theorem. Principal declarations
are `observed_logits_eq`, `logits_reconstruct`, `unseen_logits_of_min`,
`logit_error_of_reconstruct` and `every_optimizer_predicts_unseen`.

## Convex channel minibatch updates

### CM1. Independent query steps preserve the domain

For any feasible joint tensors `A,B` and query-dependent step sizes
`0 <= eta(q) <= 1`, define

```
W(q,k,v,c) = (1-eta(q)) A(q,k,v,c) + eta(q) B(q,k,v,c).
```

Nonnegativity follows term by term. Summing over classes preserves each
key's value-independent routing marginal. Summing over keys then gives
`(1-eta(q))*1 + eta(q)*1 = 1`. Hence `W` is feasible even when different
query blocks use different step sizes. Its logits undergo the same convex
combination for that query. When `eta(q)=0`, every coefficient of that query
is retained. This matters for a minibatch containing only some queries:
the absent ones must not be reset to an arbitrary zero-gradient oracle.

`ConvexChannel/Batch.lean` proves these facts as `feasible_mixRows`,
`logits_mixRows` and `mixRows_zero`, with Boolean witnesses. Combined with
CC8's feasible oracle vertices, they justify blockwise minibatch feasibility.
CC2 already supplies convexity of ordinary CE on each finite batch. These
statements do not assert stochastic optimization convergence or transfer of
unseen labels. CF6 separately proves persistent forward-only batch equality.

## Inference-time learning

### CI1. Retained state can learn from later observations

After main training, retain `trainedState(history)`, the complete candidate
records of CF6. When a correct answer becomes available, whether through
feedback or an observable target in the incoming stream, pass that labeled
example to the same `advance` operation. For arbitrary history and feedback,

```
advance(trainedState(history), feedback) = trainedState(history ++ feedback).
finish(keys, advance(trainedState(history), feedback), default)
  = learn(keys, history ++ feedback, default).
```

These are exact computed-state and model equalities, including failure.
If one enumerated key and decoder realize both phases, the next snapshot
succeeds and fits every old and new observation. The update receives the
labeled data, not that target key or decoder. This is continued categorical
learning without backpropagation; it is not a theorem that arbitrary unlabeled
inputs reveal their correct answers. A task-specific observable-target adapter
must justify the labels it supplies. The theorem guarantees fitting the combined
observations; it asserts no general monotonicity of held-out accuracy. The finite
example in CI4 demonstrates correction of a previously wrong prediction.

### CI2. Self-labeling has a fixed prediction function

Given the current successful snapshot `(k,f)`, label an arbitrary list of
inputs by `(D,f(D(k)))` and really apply `advance` to all candidates. Every
earlier candidate remains eliminated. The selected key survives: at a recorded
value its prediction is the recorded label; at an unobserved value it inserts
the same default that its decoder already returned. Its entire decoded function
therefore stays equal to `f`. Thus the next snapshot is exactly `(k,f)`.

`inferBatch` performs this unverified update; if no candidate survives, it
retains the failed state. `inferBatches` refreshes the snapshot between batches.
For every initial state and every finite list of arbitrary input batches,

```
finish(keys, inferBatches(keys, batches, state, default), default)
  = finish(keys, state, default).
```

The result includes exact key and function equality, so all predictions and
the corresponding deterministic channel remain unchanged. Refitting the original
learner after appending its own hard predictions gives the same result too.
This is a claim about this deterministic consistency learner with fixed keys,
default and hypothesis family, not about all uses of self-supervision.

The candidate **state** need not stay unchanged: unobserved cells can acquire
default guesses, and alternatives can be eliminated. Those mutations can harm
future learning even though the current predictions have not improved.

### CI3. Fix unverified updates with a conservative agreement certificate

`certify(keys,state,D)` returns an answer only when at least one candidate
survives, every surviving enumerated candidate has already observed its value
`D(k)`, and all those observed entries have the same class. It returns `none`
on an unknown value or disagreement. It uses neither a completed decoder's
default nor a confidence threshold. This sufficient certificate is deliberately
conservative; no completeness claim is made.

If the certificate returns `c`, every enumerated surviving key and decoder
consistent with its observations predicts `c` on `D`. In particular, the answer
is correct if the true key is enumerated and its decoder is still consistent.
Those hypotheses matter: corrupted prior labels or a target outside the
single-key family are not repaired by agreement among the remaining candidates.

`guardedObserve` updates only on a certificate and otherwise abstains. For
every enumerated candidate its post-update record is exactly its old record.
If all keys are enumerated, the entire state is unchanged. Accepted entries
were already known, so this guard prevents contamination but supplies no new
training information. Keep new unverified guesses out of the trusted records;
use independently observed answers to add information through CI1. Records
already contaminated by guesses must be rebuilt from trusted observations.

### CI4. Useful feedback, poisoned defaults and an ambiguity witness

Take Boolean values/classes, enumerate key `false`, and train on dictionary
`id` with label `false`. The value at that key is `false`; the true decoder
may be `id`, but the default completion predicts `false` everywhere.

* Revealing label `true` on the all-true dictionary changes the learned decoder
  to `id`, correcting the previously wrong prediction at the new value.
* Self-labeling that dictionary with the current prediction `false` first
  leaves the prediction function unchanged, but records the guess. The same
  genuine feedback then eliminates the only enumerated key and learning fails.
* The certificate abstains at this unobserved value. Guarding that inference
  observation preserves the record, and the genuine feedback still recovers `id`.

Separate examples show acceptance of an observed answer and abstention when
two surviving candidates have observed but conflicting answers on a new input.
Finally, for **any** proposed Boolean answer at value `true`, there is a Boolean
decoder fitting the same observed `false -> false` label that disagrees with
that answer. Both possible rules belong to the single-key family. Thus no
procedure using only the common observations can guarantee that unseen label
without additional evidence or restrictions on the rule. This ambiguity
statement does not rule out a valid task-specific self-supervised signal.

The `ConvexChannel/Forward/Inference/` modules prove CI1-CI4. They formalize the
existing categorical update and the new conservative guard; no neural teacher,
noisy-label optimizer or Rust deployment of the guard is asserted here.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
```

## Convex spectral upper bounds

### SB1. Direct complex coordinates with ordinary cross-entropy

Use the matrix entries `X ∈ ℂ^(m×n)` as the learned coordinates, with
`|X_ij| ≤ 1` and `||X||op ≤ R`. Both constraints are convex over the reals.
Fixed probes give logits `Re tr(A_scᴴ X)`, so ordinary empirical
cross-entropy is jointly convex as well. The spectral condition is an
upper bound, not equality of singular values, orthogonality, or a product
of trained factors. For code probes the logits are
`Re(conj(phi(c)) X[cell(s)])`. A flat unit-modulus target of squared
singular value `c` meets the spectral bound `R = sqrt(c)`.

`SpectralBall/Basic.lean` proves domain convexity, zero feasibility,
convexity of the usual probe CE, and the squared vector-energy bound from
the actual L2 operator norm. The size of a useful radius is additional
information; flatness does not select the missing labels by itself.

### SB2. Repair of the XE15 unseen-sign counterexample

For XE15's `K = [[1,1],[-1,1]]`, use `R = sqrt(2)` and observe only
`(0,0), (0,1), (1,0)`, with their binary labels. The target is feasible.
An entry in the unit disk has signed real part at most one, so the target
minimizes each observed ordinary binary CE term. Strict convexity in
the observed real parts, together with the disks, forces every global
CE minimum to equal the three observed complex entries of K.

The spectral bound says `2I - XᴴX` is positive semidefinite. Its `(0,0)`
entry vanishes because the first column is `(1,-1)`. A zero diagonal
entry in a positive-semidefinite matrix forces its column to vanish;
the off-diagonal equation is `1 - X_11 = 0`. Thus every constrained CE
minimum is K, including the unobserved `+1`. `SpectralBall/Recovery.lean`
proves this from the norm constraint and an empirical-loss inequality,
with K as a satisfying example. No assumption of correct unseen
predictions is used. This changes the model constraint; XE15's negative
claim about nuclear-norm training remains valid.

This witness is not a theorem for random half-observed modular tables,
looser radii, arbitrary labels, long contexts, or a specific finite-step
optimizer. Those require separate evidence. Numerical projections and
finite-difference updates are floating-point algorithms, not Lean-certified
solutions of the constrained optimization problem.

### SB3. A tight bound can still leave multiple optimal unseen answers

Let

```
H = [ 1  1  1  1 ]      H' = [ -1  1 -1  1 ]
    [ 1 -1  1 -1 ]           [ -1 -1 -1 -1 ]
    [ 1  1 -1 -1 ]           [  1  1 -1 -1 ]
    [ 1 -1 -1  1 ]           [  1 -1 -1  1 ].
```

Both satisfy `HᴴH = 4I`; H' exchanges and negates the first two rows.
Observe every cell except rows `{0,1}` crossed with columns `{0,2}`.
The twelve observations are distinct, cover every row and column, and
have identical labels under H and H'. Each table belongs to the same
tight domain `|X_ij| ≤ 1`, `||X||op ≤ 2`. Each also minimizes every
observed ordinary binary CE term over the unit disks. Thus both are
global CE minima over that convex domain, yet their unseen `(0,0)`
labels are opposite. `SpectralBall/Ambiguity.lean` proves these claims,
including the observation coverage and both Gram identities.

This refutes a universal extension of SB2 even at 75% observation,
with genuinely flat targets and all rows and columns represented.
Additional assumptions must eliminate observationally indistinguishable
targets; changing batch size or solving the same empirical objective more
accurately cannot supply such information. This example is not a theorem
that the particular modular stress instances are unrecoverable.

## Coupling complex class moments

### CM1. One distribution, many spectral bounds

The SB1 pilot could fit observed phase labels and still leave weak, incorrectly
oriented unseen phases. Replace the independent complex entry at each input
cell by a shared class distribution `P_ijc >= 0`, `sum_c P_ijc = 1`. For fixed
complex class codes define `M_f(P)_ij = sum_c P_ijc phi_fc` and impose
`||M_f(P)||op <= R_f`. These are convex upper bounds on linear images, so their
intersection with the probability simplexes is convex. Unit codes imply the
old entry-disk bounds automatically. For centered codes and nonnegative radii,
uniform class probabilities give a feasible starting point.

`CoupledMoments/Basic.lean` proves these statements. This changes the hypothesis
class, rather than asserting that extra optimization resolves SB3's ambiguity.
The radii must still be specified independently of missing labels.

### CM2. Ordinary cross-entropy and an attainable lower bound

Use `P_ijc` as the logits at scale one. The loss is ordinary full-vocabulary
cross-entropy, `log(sum_c exp(P_ijc)) - P_ijy`, not the negative logarithm of a
probability coordinate. It is jointly convex. Among probability-vector logits,
the one-hot vector for the correct label minimizes each observed CE term, with
value `log(exp(1) + V - 1) - 1`. Consequently any feasible table matching all
observed one-hot labels is a global minimum of the same empirical objective.
This can certify a forward-only feasibility method without evaluating gradients.

`CoupledMoments/Loss.lean` proves convexity, the ideal-loss lower bound, the
minimum certificate with explicit feasibility, and the bound
`loss(P) - loss(Q) <= loss(P) - idealLoss` for every feasible competitor Q.
The numeric excess is useful even if a particular iterative solver has no
proved finite-step rate. None of these claims predicts an unobserved label.

### CM3. Full Fourier readings retain labels and exclude inconsistent moments

Choose `phi_fc = exp(2 pi i f c / V)`. The complete Fourier matrix has Gram
matrix `V I`; multiplying by its adjoint recovers all class coordinates. For
probability tables the zero mode is already one, so the nonzero modes suffice.
These fixed codes use only the vocabulary identifiers, never a task operator.
The probability parametrization is thus equivalent to coupled complex Fourier
coordinates, not a set of freely trainable independent complex matrices.

The coupling is a strict restriction. With four equally spaced classes, a
distribution whose first moment equals +1 has all its mass on class zero and
therefore second moment +1. Independent unit disks would also permit the
inconsistent pair (+1,-1). `CoupledMoments/Consistency.lean` proves injectivity,
the Fourier specialization, and this explicit counterexample. A stronger
constraint alone is not a proof of recovery on random half-observed tables.

### CM4. A feasible output from a common mixture

For centered codes, mix a probability table with uniform mass:
`P' = alpha P + (1-alpha) U`. If `0 <= alpha <= 1`, it remains a probability
table and `M_f(P') = alpha M_f(P)`. Thus choosing one alpha satisfying
`alpha ||M_f(P)||op <= R_f` for every f makes the result feasible. If alpha is
positive, all strict class comparisons are unchanged, including ties under a
fixed argmax rule. `CoupledMoments/Blend.lean` proves the scaling, feasibility
and comparison statements. An SVD-based implementation must still report its
numerical error and the actual mixture coefficient; this is not a verification
of floating-point norm estimates or of optimizer convergence.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments
LAKE_ARTIFACT_CACHE=false lake env lean scripts/CoupledMomentsAxioms.lean
```

### CM5. Convex bounds on largest and total singular values

The shared-moment pilot still fails on addition. Proposition 2 gives a reason
to distinguish spectral shapes: the addition phase matrix is rank one with
nuclear norm V, whereas a full flat square matrix has V singular values of
size sqrt(V). A bound on the largest singular value alone can exclude the
former or leave its structure unrestricted. Add the convex upper bounds
`nuclearNorm(M_f(P)) <= B_f` alongside `||M_f(P)||op <= R_f`.

`CoupledMoments/Schatten/Basic.lean` proves convexity of this intersection,
uniform feasibility for centered codes and nonnegative budgets, and the same
global CE-minimum certificate for a feasible exact observed fit. The common
mixture of CM4 scales both norms, so one positive coefficient enforcing both
budgets gives a feasible output without changing strict class comparisons.
The budgets are fixed model parameters; no product of learned factors, rank
equality or task-specific arithmetic operation is introduced.

For prospective tests, two generic square-matrix shapes are `(R,B) =
(sqrt(V), V sqrt(V))` and `(V,V)`. The former adds only a redundant nuclear
budget to the previous operator ball, since each of the V singular values is
at most R. `Schatten/NormComparison.lean` and `budget_at_card_iff` prove this
redundancy. The latter allows a large leading singular value while restricting
their sum. These are alternative constraints, not a claim that every target
belongs to either shape. The nuclear-norm failures in XE15/SB2 remain valid;
a sum bound alone is not a universal replacement for the operator bound.

### CM6. A rank-one missing-entry recovery certificate from actual CE

For `X(z) = [[1,1],[1,z]]`, the established 2x2 formula gives
`nuclearNorm(X(z))^2 = 3 + |z|^2 + 2 |z-1|`. Since
`|z-1| >= 1-Re(z)`, a nuclear budget of two implies
`(Re(z)-1)^2 + Im(z)^2 <= 0`, hence z=1. Conversely X(1) has nuclear norm
two. `Schatten/Recovery.lean` proves this equivalence and the corresponding
recovery of the missing binary probability coordinate from three observed
one-hot labels. There is no rank constraint hidden in the feasible set.

`Schatten/Optimality.lean` removes correct observed labels as an assumption:
each ordinary CE term is at least its one-hot value, so a summed loss no larger
than the ideal sum forces equality term by term. With another class present,
the strict CE inequality of CR3 forces each observed target mass to one. For
the three observed cells `(0,0), (0,1), (1,0)`, the nuclear budget then gives
the unseen target mass at `(1,1)`. The all-true table satisfies every hypothesis.
This is a concrete repair, not a theorem for arbitrary observation patterns
or convergence of the native iterative solver.

### CM7. One convex class for several fixed spectral budgets

Let D_k be the feasible probability tables for each member of a fixed family
of `(R_k,B_k)` budgets. Use `convexHull(union_k D_k)` as the model domain in
the direct probability coordinates. The hull is convex, all its rows remain
probabilities, and the ordinary empirical CE remains convex. This definition
does not train a bilinear product of free mixture weights and free tables.

`Schatten/Hull.lean` proves these properties and a useful optimization
certificate: any feasible exact observed fit from any D_k is a global CE
minimum over the entire hull. Thus a procedure can train candidates with
different budgets and certify a solution using training observations alone.
If none reaches the ideal lower bound, choosing the best candidate does not
prove that the full hull problem has been solved. Nor does an exact fit select
unseen answers uniquely; all previous ambiguity qualifications still apply.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SchattenMomentsAxioms.lean
```

### CM8. Causal and spectral predictions in one convex class

The pair-moment pilot does not by itself cover the sequence tasks. To preserve
the causal count model while admitting shared complex moments, express each
expert as a probability-valued function on the same visible input domain.
Take the convex hull of the fixed expert families in these direct prediction
coordinates. Every original expert remains feasible, every row remains a
probability vector, and ordinary empirical CE remains convex. This is not a
bilinear parameterization by free expert weights and free expert parameters.

All experts for a given dataset use the same fixed scale s >= 0. Their logits
are s times the probability coordinates. The one-hot logits have the common
per-observation lower bound `log(exp(s) + V - 1) - s`. The relative-logit
comparison proving CM2 also proves this scaled bound. An expert that fits all
observed one-hot labels therefore globally minimizes ordinary CE over the
whole prediction hull; for any other feasible competitor, its objective
excess is at most the measured ideal-loss excess. `ScaledLoss.lean` and
`PredictionHull.lean` prove these statements with nonempty examples at s=10.

The empirical objective sums ordinary teacher-forced CE over observed prefixes.
Own-output generation and residue pooling remain evaluation rules, not the
training objective. Input-only eligibility or fixed scale choices do not
train extra parameters. Selecting the first training-certified candidate is
again certificate search: without an ideal fit it need not optimize the hull
interior. Even a certified optimum need not identify missing answers. New
experts, unsupported input shapes, failed fits and compute for every candidate
must all be reported in the full stress suite.

### CM9. The two endpoint budgets leave a capacity gap

CM5 explicitly did not assert that every target belongs to one of its two
example shapes. A four-by-four binary moment makes the gap concrete:

```
H = [[1,1,1,1], [1,1,1,1], [1,1,-1,-1], [1,1,-1,-1]].
```

This is the fixed sign-code moment of a one-hot probability table. Its Gram
matrix satisfies `(H*H)^2 = 8 H*H`, its Frobenius energy is 16, and its rank
is two. Hence its operator norm is `sqrt(8)` and its nuclear norm is
`2 sqrt(8)`. The size-four full-rank budget `(R,B)=(2,8)` excludes it by
operator norm; the rank-one budget `(4,4)` excludes it by nuclear norm.
The fixed intermediate budget `(sqrt(8),2 sqrt(8))` admits the same table.

Mixing the two old spectral families does not repair this example. One-hot
tables are extreme points of the product of simplexes, so the convex hull of
a probability-valued family contains such a table only if that family already
contained it. `Schatten/Vertices.lean` proves this general fact, and the witness
is therefore excluded by CM7's two-endpoint spectral hull too. This does not
exclude the larger CM8 hull when it contains an unrestricted lookup expert.

`Schatten/Intermediate.lean` proves the actual norms and both exclusions,
using a general operator-norm consequence of the Gram-square identity.
The witness has a binary output vocabulary and four possible values of each
input; it is a representation counterexample, not a result on a named stress
dataset. It motivates additional generic fixed spectral budgets under CM5
and CM7. The new bounds remain convex upper bounds, with no learned product,
rank equality, arithmetic rule or missing label in the domain. Inclusion of
a target is necessary capacity, not a proof that CE training will recover it.

### CM10. Feasible uniform completion and a sharing remedy

Repairing CM9's capacity gap does not force missing labels. Observe all rows
of columns zero and two of its table, leaving columns one and three unseen.
Keep the observed one-hot probabilities and put uniform binary probabilities
in the unseen columns. Its moment is H with the unseen columns zeroed. Its
operator and nuclear norms are two and four, below the intermediate bounds
`sqrt(8),2 sqrt(8)`. It fits every observed label and therefore globally minimizes
ordinary CE on that feasible set, although an unseen target coordinate is
only one half. `Schatten/Completion.lean` proves the norms, feasibility, exact
fit and actual minimum together. This explains a failure that extra optimizer
steps alone need not repair; it is not an assertion about a particular random
stress split or a floating-point checkpoint.

One repair is to supply correct shared coordinates. Fix a column representative
map r, and impose `P(i,j)=P(i,r(j))`. These are linear equalities and form a convex
set. At the observed CE lower bound, any row sharing an observed representative
equals that observation's one-hot target. `Schatten/Sharing.lean` proves this
conditional statement. For CM9's table, r maps columns zero/one to zero and
two/three to two; every representative is observed, and the same empirical CE
minimum now determines the entire table. The target satisfies every hypothesis.

The representative map is fixed in this convex problem. Agreement on a few
observations alone does not prove that two input values are interchangeable
on unseen inputs. A future training-only grouping procedure must preserve all
original observation multiplicities, test its proposed consistency, retain
unshared controls, and evaluate missing answers separately. If it searches
over fixed maps, the certificate concerns each conditional convex model or
their direct prediction hull; it is not a convexity theorem for freely learned
products of a map and a table, nor a guarantee that the selected sharing rule
matches unknown ground truth.

More generally, fix row and column group maps rho and kappa, learn Q on their
grouped indices, and predict `P(i,j)=Q(rho(i),kappa(j))`. Impose the same convex
spectral budgets on Q. `Schatten/Quotient.lean` proves that the lifted family
is convex, remains probability-valued, and retains the exact-fit ordinary-CE
minimum certificate. Its loss is still summed over every original observation;
multiple observations mapped to one grouped cell retain their multiplicities.
This is a fixed linear lookup of learned coordinates, not a learned product
of a grouping matrix and Q. The conditional statement does not prove that an
adaptive grouping search returns a correct map or an identifying completion.

For a fixed row/column map, `quotient_observed_group` also derives the complete
one-hot prediction on every input sharing its grouped coordinates with an
observed example, directly from attaining the original CE lower bound. The
inhabited example observes only one of two original columns and recovers the
other through their fixed shared coordinate. A missing-answer audit can thus
separate groups already covered by training from wholly unobserved groups
requiring further completion. Correctness of the copied label on held-out
ground truth is a separate check of the proposed map; the theorem does not
infer it from observed agreement alone.

### CM11. Certify a sharper feasible endpoint without changing decoded answers

Accurate argmax answers can coexist with a small observed CE gap at a numerical
iteration cap. In a larger prediction hull, a memorizing candidate attaining
ideal training CE would then win a training-loss-only comparison. One generic
endpoint check is to take the model's own decoded labels, make their active
grouped coordinates one-hot, and keep inactive padding uniform. This reads no
held-out labels. Decoding itself is not a convex map, and its resulting table
need not satisfy the spectral bounds.

Let H be this decoded table and choose a common `0 < alpha <= 1`. With centered
class codes, CM4 and CM5 give `M_f(blend(alpha,H)) = alpha M_f(H)`. Checking
`alpha ||M_f(H)||op <= R_f` and `alpha ||M_f(H)||_* <= B_f` for every f therefore
certifies the same convex feasible domain. Every active decoded class remains
strictly above the alternatives. The operator and nuclear norms must be checked
after decoding; the old soft table's norms do not suffice.

If the decoded labels agree with every original observed label, H attains ideal
observed CE. Convexity of the actual ordinary CE then gives, against any
probability-valued competitor Q,

```
L(blend(alpha,H)) - L(Q) <= (1-alpha) (L(uniform) - L_ideal).
```

The observation index still includes all original repetitions. `Schatten/Sharpen.lean`
proves probability feasibility, strict decoded comparisons and this certificate,
with a nonzero complex-code example. It does not prove a floating-point norm
computation, that sharpening improves CE, or any missing-label correctness.
The native follow-up must keep the original endpoint when its training CE is
already ideal or the checked candidate does not improve it, record the new
norms and mixture, and verify that all decoded predictions are unchanged.
This checks another point in the same convex family; it does not change the
family, the ordinary loss, or the frozen earlier training runs.

### CM12. Remove dependence on the arbitrary order of output class names

Output relabeling has a different effect from permuting input rows or columns.
For a class permutation sigma define `(T_sigma P)(i,j,c) = P(i,j,sigma⁻¹(c))`.
This real-linear map preserves all probability rows and satisfies

```
CE(scale * T_sigma P(i,j), sigma(y)) = CE(scale * P(i,j), y),
M_phi(T_sigma P) = M_(phi composed with sigma)(P).
```

The first equality preserves the complete vocabulary and softmax denominator.
It also preserves the original empirical CE sum, including repeated observations.
The second equality changes the class code; it need not be a unitary change of
input coordinates. Consequently a spectral bound in a fixed class coding is
not generally invariant under renaming the target labels.

Here is an exact one-code counterexample. Take four class phases
`phi = [1, i, -1, -i]` and the point-mass label table

```
0 0 0 0
0 1 2 3
0 2 0 2
0 3 2 1
```

Its complex moment is twice the normalized four-point Fourier matrix, with Gram
matrix `4 I` and operator norm 2. Swap class names 1 and 2. The new moment applied
to `[1,1,1,1]` is `[4,0,2+2i,0]`; squared input energy is 4 and squared output
energy is 24. An operator bound of 2 would require output energy at most 16.
Both tables remain full class distributions, and their correspondingly renamed
target CE values agree exactly. This example uses **one** centered unit code.
It does not claim feasibility of the first table under every nonconstant
four-class character, nor establish the cause of the prime-vocabulary native
failures. Those are separate empirical and domain-specific questions.

For any original family A define the following domain in direct prediction
coordinates:

```
A_sym = convexHull (union over every class permutation sigma of T_sigma(A)).
```

This domain is convex, contains A, and is invariant under every output relabeling.
If every member of A has probability rows, every member of A_sym does too.
Fixed-scale ordinary CE is convex on A_sym. The existing CM8 ideal-loss and
excess-loss certificates therefore apply with these renamed families as the
generators. The concrete excluded table above belongs to A_sym, proving that
this enlargement can be strict. `CoupledMoments/Relabeling.lean` proves the
general identities and convex-family properties;
`CoupledMoments/RelabelingObstruction.lean` proves the exact counterexample.

Each generator retains the original spectral bounds in its corresponding class
coding. A mixture of generators **need not satisfy those same bounds in one
fixed coding**. This is an enlargement of the feasible family, not an invariance
theorem for the old fixed-code ball. The theorem also supplies neither an
efficient oracle over the factorially many codings nor a jointly convex
parametrization of a trainable permutation. A native follow-up may propose
codings from training observations and compare their actual CE, but finite
proposal search alone does not certify the entire new domain. No conclusion
about missing labels follows without further structure and coverage.

### CM13. Graph coverage and coded structure identify the complete CE optimum

CM6 establishes recovery of a single missing entry in a two-by-two example.
The existing Theorem E gives a general route from the same convex nuclear
budget to recovery on an entire square input table. Let the input dimension be
N and let the observed cells form a bipartite graph between its rows and columns.
Assume that graph is connected and its observed cells contain a perfect matching.

For unit-modulus vectors a and b, Theorem E says that `K = a bᵀ` is the unique
minimum-nuclear-norm interpolant of its observed entries, and `||K||_* = N`.
Consequently any interpolating X with `||X||_* <= N` equals K: K's minimality
places its norm below every other interpolant, while the budget places X's
norm below K's. Thus X is also a minimizer, and uniqueness applies. The learner
needs only the convex upper bound, not a rank equality or trained factors.

Now let `phi : F × C -> complex` be a complete code bank with
`phi* phi = rho I`, `rho != 0`. Suppose a full label table y has the explicit
coded structure

```
phi(f, y(i,j)) = a_f(i) b_f(j),
|a_f(i)| = |b_f(j)| = 1.
```

Learn probability rows P under the bounds `||M_f(P)||_* <= N`. If their actual
ordinary observed CE attains the one-hot lower bound, CM6 forces every observed
target probability to one. Each moment therefore interpolates its observed
entries of `a_f b_fᵀ`. The graph argument recovers every entry of every moment.
CM3's injectivity of the complete code bank then gives

```
P = pointMass(y)
```

on **all** input cells, including unobserved ones. Repeated observations remain
in CE; the graph uses their set of cells. The constant Fourier mode requires
no new constraint: probability rows already make it the all-ones matrix,
whose nuclear norm is N. The native nonconstant-mode budget therefore supplies
the relevant bounds. CM12's coordinate covariance permits applying the result
in a fixed renamed output coding and returning the probabilities to their
original class order.

`CoupledMoments/Schatten/GraphRecovery.lean` proves the budget-to-uniqueness step,
the automatic constant-mode bound and the complete probability-table theorem.
Its inhabited two-class example observes three cells of a nonconstant table,
has a connected graph and an observed perfect matching, and leaves `(1,0)`
unobserved. Both classes occur among its observed targets.

The target's coded separability is a structural hypothesis about the complete
table. Observed phase consistency alone does not prove that unknown ground
truth satisfies it. A training-only graph/coding audit can instead construct a
compatible separable completion, check connectivity and a matching, and compare
saved predictions with that model completion. Agreement between the completion
and held-out ground truth remains an empirical check, never an input to the
proposal. Nor does this exact theorem turn a floating-point CE tolerance into
exact equality: quantitative stability and finite-step solver convergence are
separate claims. Connectivity by itself, without the matching and coded
structure, does not provide this guarantee.

### CM14. A training-compatible phase completion is a feasible CE minimizer

CM13 identifies an ideal endpoint but does not construct one. A discovered
categorical coding supplies a direct candidate in the same convex family.
Let `e : C <-> Z/q` be a bijection of the full class vocabulary, with `q > 0`,
and let `r(i)` and `s(j)` be row and column phase potentials. These potentials
are learned from the observed equations

```
e(y_t) = r(i_t) + s(j_t) mod q.
```

Input symbols enter as categorical row and column indices; no numerical value
or arithmetic operation on an input symbol appears in this construction.
Define the model completion and its full complex code bank by

```
z(i,j) = e^{-1}(r(i) + s(j)),
phi(f,c) = exp(2 pi i f e(c) / q).
```

The one-hot table `P_z` has probability rows. Every coded moment factors as
`a_f b_f^T`, with `a_f(i) = exp(2 pi i f r(i)/q)` and the analogous `b_f(j)`.
Both factors have unit-modulus entries, so its operator norm is at most the
input dimension N and its nuclear norm equals N. Thus `P_z` is feasible for the
existing radius-N, nuclear-budget-N family. The separate `full_rank` control
uses the smaller radius `sqrt(N)` and is not covered by this construction. The
observed phase equations make its actual ordinary CE equal the one-hot lower
bound. It is therefore a global CE minimizer in that fixed-code convex family,
without requiring an iterative optimizer to discover this endpoint.

The complete code bank satisfies `phi* phi = q I`, including after a categorical
renaming. If the observation graph is connected and contains a perfect matching,
CM13 further identifies this candidate as the only ideal probability table under
the nuclear budget. This conclusion concerns the compatible model completion.
It does not prove that unobserved true labels follow the learned phase equations.
Discovery of a coding remains a discrete proposal step, outside the fixed-code
convexity claim. A finite list of such candidates is not a global oracle over
the CM12 hull of all output permutations.

`CoupledMoments/Schatten/PhaseCandidate.lean` proves the renamed Fourier Gram
identity, the explicit moment factorization, feasibility and global CE
optimality. It specializes CM13 to show uniqueness with only the nonconstant
nuclear bounds supplied as hypotheses; probability rows provide the constant
mode. Its nonconstant two-class construction has an inhabited three-observation
CE-optimality example, and the uniqueness theorem has an inhabited complete-graph
example.

A native warm initialization may use this candidate only after checking the
training equations and class bijection, and must retain explicit rejection and
the unchanged cold-initialization controls. Numerical feasibility, original
split metrics, costs of coding and potential discovery, and any subsequent
gradient updates must still be independently audited. The theorem is an
exact-real construction; it does not certify floating-point SVDs or unseen-label
accuracy, and does not change the frozen CM12 or depth cohorts.

### CM15. Decode spectrally bounded latent states into the original vocabulary

The bijection in CM14 excludes some deterministic tables, independently of
the optimizer. If three corners of a rectangle have label `a` and the fourth
has a different label `b`, no injective additive class code can represent it:
the first column forces the two row phases equal, while the second column
would then force the two distinct labels equal. This is a categorical capacity
obstruction, without any interpretation of the input indices as numbers.

Let `L` be a finite latent vocabulary, `C` the original output vocabulary and
`d : L -> C` a fixed decoder, possibly many-to-one. Learn the full latent
probability table `P(i,j,l)` under the existing fixed complex operator/nuclear
bounds, and expose the original-vocabulary probabilities

```
Q(i,j,c) = sum_{l : d(l)=c} P(i,j,l).
```

This map is real-linear and preserves probability rows. Its image of the
latent budget domain is convex, and ordinary original-vocabulary CE is convex
in every latent coordinate. A feasible latent table whose decoded rows fit
the observed labels attains the same one-hot CE lower bound for vocabulary
`C`, even if some original output classes have no latent preimage. CE remains
log-sum-exp over every original class, with logits `Q`; it is not a likelihood
over a reduced observed-class set. The identity decoder gives back the old
model. The decoder and code bank are fixed when taking this convexity claim;
choosing them from data remains a proposal step, not a jointly convex search
or a certified global oracle over all decoders.

The whole prediction class can include those choices convexly: take the convex
hull, in original probability coordinates `Q`, of the decoded feasible sets
over all `d : L -> C`. Every point is still a probability table and ordinary
CE is convex on this hull. Any feasible decoded candidate that reaches the
observed one-hot CE bound is also a global CE minimum over the entire decoder
hull. Thus successful proposal search can have a global prediction-space CE
certificate even though its discrete search is not itself convex. This does
not supply a polynomial-time oracle, guarantee finding such a candidate, or
identify unseen labels. No independent learned decoder/latent factors are
multiplied in the prediction-coordinate optimization claim.

In particular, CM14's latent phase table with any `d : Z/q -> C` is feasible
at radius and nuclear budget `N`. Observed equations
`d(r(i_t)+s(j_t))=y_t` suffice for decoded global CE optimality. The spectral
bounds constrain the latent moments; they must not be presented as unchanged
bounds on the decoded output moments.

This strictly increases the available deterministic phase constructions.
Take two row and column phases `0,1` in `Z/3`, and decode `0,1` as `true` and
`2` as `false`. The decoded table has three `true` corners and a `false`
fourth corner. Its latent moments satisfy the original `N=2` rank-one budgets.
The direct binary sign moment cannot satisfy nuclear budget two, by CM6.

The stronger capacity has an explicit identification limit. With the same
decoder, latent code and budgets, constant zero phases also fit those three
observed `true` labels perfectly but predict `true` at the fourth corner.
Both decoded tables are global ordinary-CE minima. Hence CM13/CM14's recovery
conclusion cannot simply be transferred from observed latent labels to
observed decoded labels. This three-edge observation graph is connected and
has the off-diagonal perfect matching, so graph coverage is not the missing
hypothesis. A successful native proposal still needs original
held-out evaluation; a uniqueness claim requires additional identification
evidence. No native accuracy result or efficient decoder-discovery guarantee
is asserted by this extension.

`CoupledMoments/Schatten/Decoded.lean` proves real-linearity, probability-row
preservation, the pulled-back moment identity, convexity, identity-decoder
equivalence and the actual CE fitting certificate.
`CoupledMoments/Schatten/DecodedPhase.lean` proves the latent phase construction,
the direct-code obstruction, the admitted counterexample and its two distinct
CE-optimal completions, including connected matching coverage.
`CoupledMoments/Schatten/DecodedHull.lean` proves probability preservation,
convex CE and the global fitting certificate across all decoder choices.
It also proves the two ambiguous endpoints remain global CE minima over this
larger hull: convexifying the class does not supply the missing identification.

### CM16. Equal probabilities on opposite latent states

*Source-note extension for board task #195, 2026-11-12. The Lean modules were
drafted by the task's first claimant and completed by its second.*

CM15 leaves the decoder arbitrary. One fixed class of decoders needs no search
over latent signs. Let the latent vocabulary be `Z/q` and let the decoder be
even,

```
d(-z) = d(z).
```

**Reflection.** `(rho P)(i,j,z) = P(i,j,-z)` is real-linear and permutes the
coordinates of every probability row. On the Fourier bank of CM14 it exchanges
frequencies,

```
M_f(rho P) = M_{-f}(P),
```

so operator and nuclear budgets that are the same for every frequency survive
it. The average `pi P = (P + rho P) / 2` then satisfies the same budgets by
convexity, has equal probabilities on opposite states,

```
P(i,j,z) = P(i,j,-z),
```

and decodes to the same table: `decode d (pi P) = decode d P`.

**The paired domain.** Add these linear equalities to the latent domain of
CM15 and decode. The set of predictions is convex, and ordinary CE over the
original vocabulary is convex on it. For an even decoder the equalities remove
no prediction: the paired domain is exactly the decoded domain of CM15 with
the same constant budgets. The symmetry is a restriction of the latent
coordinates, not of the predictions.

**A symmetric phase witness.** For phases `r, s : n -> Z/q` put half of the
probability at `r(i) + s(j)` and half at its negative (all of it if the two
coincide). This table is `pi` of CM14's one-hot table. It is feasible at
radius `N` and nuclear budget `N`: it is the average of two feasible tables,
and no rank-one factorization of its own moments is used. It decodes to the
deterministic table `d(r(i) + s(j))`. If
that table shows the observed labels, it is a global ordinary-CE minimum over
the paired domain, and over the whole decoder hull of CM15. One latent state
per observed label is not chosen anywhere.

**A decoder from the vocabulary alone.** For `k + 1` classes take `q = 2k + 1`
and fold opposite states,

```
d(z) = min(z, q - z)        in {0, ..., k}.
```

It is even, every class has a preimage, and the fiber of the class `c` is
`{c, -c}`. Input side `N`, latent states `q` and output classes `k + 1` are
three separate numbers. Example: `N = 3`, `q = 5`, three classes, row phases
`i` and column phases `-j` give the table

```
0 1 2
1 0 1
2 1 0
```

The permutation-phase constructor of `experiments/convex_stress_decode` takes
as many latent states as the input side and a permutation of them along the
columns. Every row of such a table has the same number of cells of each
class, for every decoder, and convex mixtures keep these equal row masses.
The table above has one cell of class 2 in its first row and none in its
second: it lies outside the hull of those constructions with three states,
and inside the paired domain with five.

**The limit.** The symmetry does not identify unseen labels. Decode `Z/3` by
`z != 0`, and observe `false, true, true` at the cells `(0,0), (0,1), (1,0)`.
The phases `i + j` and `i - j` both fit. Their tables are feasible in the same
paired domain and both are global CE minima, with `true` against `false` at
the unobserved cell `(1,1)`. The observed graph is connected and contains a
perfect matching. Recovery needs the identification conditions of a separate
argument; capacity and an observed fit do not give it.

Not claimed: that an iterative solver reaches such a minimum, any numerical
tolerance, any accuracy on original data, or a way to find the decoder.

`CoupledMoments/Schatten/Paired.lean` proves the reflection identities, the
convexity of the paired domain and its equality with the decoded domain.
`PairedPhase.lean` proves the witness and the CE certificates.
`PairedCoding.lean` proves the properties of the folding decoder and the
three-class example. `PairedLimits.lean` proves the equal-row-mass
obstruction and the ambiguity.

### CM17. The class coding is found from the training rectangles

*Source-note extension for board task #196, 2026-11-12. The number CM16 is
reserved for task #195.*

CM14 builds a feasible CE minimizer from a class coding and CM13 identifies it,
but both stop before two steps: "Discovery of a coding remains a discrete
proposal step", and the conclusion "does not prove that unobserved true labels
follow the learned phase equations". This item proves both steps for one
family of tables and one explicit learner. It is not about a trained network.

**Setting.** A square table `y : n × n -> C` over a finite class vocabulary.
The structural hypothesis, the only assumption about unobserved cells, is

```
y(i,j) = e^{-1}(r(i) + s(j))
```

for some bijection `e : C <-> Z/p`, `p` prime, and phases `r, s : n -> Z/p`.
The learner is given none of `e`, `r`, `s`. It is given training samples
`(i_t, j_t, y_t)` with `y_t = y(i_t, j_t)`, possibly repeated. Rows, columns
and classes are names: no arithmetic on them enters, only arithmetic on the
values of a candidate coding.

**The linear system.** Four samples at the corners of a rectangle, with the
labels `a, b` in one row and `c, d` in another row at the same two columns,
give one homogeneous equation for an unknown coding `z : C -> Z/p`:

```
z(a) + z(d) - z(b) - z(c) = 0.
```

The solutions form a subspace that contains the constants and `e`. Pin one
class, `z(c0) = 0`. The pinned solutions contain `e - e(c0)`, which is not
zero, so the native rejection `no_nonconstant_phase_code` cannot occur for
such a table. The test of the learner is

```
the pinned solution space has dimension one,
```

which by rank–nullity is `rank + 1 = p` for the matrix of the pinned system:
the acceptance test of `coding.rs`. The rectangles through one shared column
of two rows give all rectangles of those rows. The native system keeps, for
two rows `row < other`, only those through their first shared column; it has
the same solutions when a cell has one label. The solution space depends only
on the set of observed (cell, label) pairs: not on the order of the samples,
their repetitions or their grouping into batches.

**What the test gives.** Assume the structural hypothesis and the test. For
every nonzero pinned solution `z`, whichever the elimination returns:

1. `z = α (e - e(c0))` with `α != 0`. The coding is recovered up to the affine
   gauge `e -> α e + β`, which does not change the table.
2. `z` is a bijection of the classes. This is derived, not assumed: the native
   bijection check cannot fail here.
3. `z` has row and column phases on the whole table, hence on the observed
   cells: the native cycle check cannot fail here either.

The test holds exactly when every solution of the rectangle system is
`α e + β`, and it does not depend on the pinned class.

**An observable sufficient coverage.** Take two rows and `p - 1` columns
observed in both, with distinct labels in the first row and one column where
the two rows differ. Then the test holds. Write `w = z ∘ e^{-1}` and let
`δ != 0` be the difference of the phases of the two rows. The rectangles of
the two rows say that `w(u + δ) - w(u)` is one constant `κ` at the `p - 1`
phases `u` of the shared columns. The sum of `w(u + δ) - w(u)` over all `u`
vanishes and `p κ = 0`, so the increment is `κ` at the last phase too. A
function on `Z/p` with a constant increment over a nonzero step is affine,
since the step generates the group. (With only `p - 2` shared labels and no
other rectangle, two increments remain free under one relation; this remark
is not formalized.)

**The completion is the table.** Let the bipartite graph of the observed
cells be connected, and let the test hold. Take any bijection `w : C <-> Z/p`
and potentials `ρ, σ` that fit the training samples,
`w(y_t) = ρ(i_t) + σ(j_t)`; a nonzero pinned solution with potentials found
by a search of the graph is one. Then

```
w^{-1}(ρ(i) + σ(j)) = y(i,j)        on every cell, observed or not.
```

Indeed `w - w(c0)` is a nonzero pinned solution, so `w = α e + β`. The
potentials `ρ, σ` and the gauged phases `α r + β, α s` then have equal sums
on the observed cells, and on a connected graph equal sums everywhere (the
rigidity of the Setting, for potentials in `Z/p`). So under the two
observable conditions the table is the only training-compatible phase
completion in the sense of CM14.

**The certificate.** What the native code returns can be checked without
trusting how it was computed: a bijection `w`; potentials `ρ, σ` with
`w(y_t) = ρ(i_t) + σ(j_t)` at every sample; connectivity of the observed
graph; and `rank + 1 = p` for the pinned system. All four are finite checks
on the training set and the returned objects. They are sound: under the
structural hypothesis they give `w^{-1}(ρ(i) + σ(j)) = y(i,j)` on every cell,
which is the statement above. They are complete for the elimination: under
the hypothesis and the rank test every nonzero pinned solution passes the
bijection check and has such potentials (points 2 and 3).

**The learner.** `discover(cells, labels, c0)` is a function of the training
set and the pinned class alone. In the order of the native checks it answers
`no_code` (the pinned solutions have dimension zero), `ambiguous` (above one),
`nonbijective` (no pinned solution is a bijection), `inconsistent` (no
bijective coding has potentials on the samples), `disconnected`, or else
`accept` with a training-compatible phase completion. For a table of the
family:

- under the test and connectivity it returns `y`, on all cells, and it
  accepts in no other case;
- it is never wrong: it returns `y`, or answers `ambiguous` or
  `disconnected`. The other three refusals certify that the labels belong to
  no table of the family;
- its answer depends only on the set of observed (cell, label) pairs.

Without the structural hypothesis an accepted table is still a
training-compatible phase completion, and nothing more is claimed for it.

**The native checks.** The learner is stated by what the native checks
compute, and it is not the native code. The differences:

- The native elimination keeps the rectangles through the first shared column
  of two rows `row < other`. For every rule that gives two such rows with a
  shared column a column that both show, the kept rectangles have the same
  solutions as all rectangles, when a cell has one label.
- A repeated cell is an error of the native code. The answer of the learner
  does not depend on repetitions.
- The native code refuses a training set that does not show every class
  (`unobserved_output_classes`) before the elimination. The learner has no
  such answer: for `p != 2` such a training set of a table of the family
  fails the rank test, and the answer is `ambiguous`. (For `p = 2` a training
  set that shows one class passes the rank test; with a connected graph its
  table is constant. This remark is not formalized.)
- The native constructor refuses a training set without a perfect matching
  (`NoPerfectMatching`): it admits the case in which CM13 gives one
  minimizer. The learner needs no matching to return the table; the matching
  enters only the uniqueness of the CE minimizer below.
- The native code refuses a composite vocabulary; the modulus of the learner
  is prime.

**The bridge to CE.** Use the coding `w` found above as the code bank of CM14:
probability rows `P`, operator norm and nuclear norm of every moment at most
`N`, ordinary log-sum-exp CE over the full vocabulary on the training samples.
Under the test and connectivity the one-hot table that the learner builds
from `w, ρ, σ` is the one-hot table of `y`, it is feasible, and it minimizes
CE: the minimizer is constructed in closed form, not searched. If the
observed cells also contain a perfect matching, CM13 applies to `w` with its
matching hypothesis as stated and its code separation `φ* φ = p I` proved in
CM14, and the program has exactly one minimizer, `pointMass(y)`. A minimizer
therefore gives probability one to `y(i,j)` at every cell. The same holds
with `w` replaced by any nonzero pinned solution and the bijection proved for
it in point 2.

These are statements about exact minimizers over the reals. A table whose CE
is within the native `1e-8` of the ideal value is not covered: no stability
bound is proved here. No cold or minibatch run of the native solver is
claimed to reach the minimizer.

**Witness.** `p = 5`, classes named `0..4`, and the hidden coding exchanges
the names `0` and `1`; `r(i) = i`, `s(j) = j`. The table is

```
1 0 2 3 4
0 2 3 4 1
2 3 4 1 0
3 4 1 0 2
4 1 0 2 3
```

Observed: rows 0 and 1 at columns 0–3, the cells `(2,2)`, `(3,3)`, `(4,4)`
and `(4,0)`: twelve cells; thirteen are held out, and all five classes occur
on them. Read as numbers, the class names violate the rectangle of rows 0, 1
and columns 0, 1 (`1 + 2 != 0 + 0`), so the given order of the names is not a
coding. Rows 0 and 1 share four columns with the distinct labels `1, 0, 2, 3`
and differ in the first, so the test holds; the graph is connected and the
diagonal is observed. The learner returns the whole table, and the CE program
of the gauge `2 (e - 1)` has the one-hot table as its only minimizer.

**A training set that must be refused.** Observe the row 0 and the column 0
of the table of `i + j` for `p = 5`: nine cells, a connected graph. The
identity coding and the coding above both have potentials that fit them, and
their tables differ at the unobserved cell `(2,2)`: `4` against `3`. No
learner can return both; this one answers `ambiguous`, as it does whenever
two different tables of the family share a connected training set.
Connected coverage alone does not fix the coding, and the rank test is not an
artifact of the method. In the other direction, the witness without its
cells `(4,4)` and `(4,0)` still passes the rank test, but its row 4 is not
observed, and the learner answers `disconnected`.

**What remains open.**

- The structural hypothesis is about the unknown truth and cannot be tested
  on the training set. The answer of the learner does not depend on the
  unobserved labels, so for a table outside the family an accepted completion
  need not be the table. Many-to-one decoders (CM15), composite `p`,
  vocabularies of another size and the question which tasks of the stress
  suite belong to the family are not treated.
- When a split passes the rank test is not analyzed beyond the two-row
  coverage: there is no probability statement over random splits.
- The native elimination and its graph searches are not formalized; of its
  anchoring only that the kept rectangles have the same solutions. The Lean
  statements are about what they compute: the kernel of the pinned system and
  Mathlib's rank of its matrix. Their cost is not analyzed.
- Numerical solutions: the exact-real minimizer against the native tolerance,
  floating-point SVDs, and convergence of any iterative solver.
- Uniqueness is within the fixed-code program of the discovered coding, not
  over the CM12 hull of all output renamings.
- Discovery by training a network is untouched. This learner is an explicit
  algebraic procedure; that AdamW on a transformer (#193) or a paired latent
  decoder (#195) finds such a coding is a separate, open claim. Discovery from
  a stream without keeping the observed rectangles is not claimed either.

`CoupledMoments/CodingDiscovery/Basic.lean` defines the rectangle system.
`TwoRows.lean` proves the sufficient coverage. `Gauge.lean` proves points 1–3
and the equivalent forms of the test, and `Rank.lean` its form `rank + 1 = p`.
`Completion.lean` proves that every training-compatible phase completion is
the table; it uses the rigidity lemma of `PatternGraph.lean`, generalized in
place to potentials in any additive group. `Learner.lean` defines `discover`
and proves the statements about its answers. `CrossEntropy.lean` proves the
bridge, and `Witness.lean` the examples above. `Invariance.lean` proves that
the system depends only on the set of observations. `NativeChecks.lean`
proves the first and the third item of *The native checks*.

### CM18. Unseen decoded answers behind a decoder that merges opposite states

*Source-note extension for board task #197, 2026-11-12. The number CM16 and
the `Paired*.lean` modules belong to task #195.*

CM15 and CM16 each end with two CE minima that disagree at an unseen cell, and
CM16 says: "Recovery needs the identification conditions of a separate
argument; capacity and an observed fit do not give it." This item gives such
conditions for one class of fixed many-to-one decoders, as a certificate that
is checked on training samples alone, and says what they give and do not give
for the convex CE program. CM17 does not apply: its coding is a bijection.

**Setting.** A square table `y : n × n -> C`. The structural hypothesis, the
only assumption about unobserved cells, is

```
y(i,j) = d(r(i) + s(j))
```

for a fixed known decoder `d : Z/q -> C` and unknown phases
`r, s : n -> Z/q`. The decoder class is

```
q odd,        d(z) = d(w)  exactly when  w = z or w = -z.
```

The folding decoder of CM16 is in it for every `k`, and so is the two-class
decoder of `Z/3` of the CM16 limit. Input side, latent states and classes are
three separate numbers, and every fiber but the one of zero has two states.
The learner is given `d` and training samples `(i_t, j_t, y_t)` with
`y_t = y(i_t, j_t)`, and none of `r`, `s`. A sample says only that
`r(i_t) + s(j_t)` is one of two opposite states: one unknown sign per cell.

Two changes of the phases leave every decoded label unchanged: the shift
`(r + g, s - g)` and the reflection `(-r, -s)`. Latent states are therefore
never identified, and are not the target. The target is the decoded label of
an unobserved cell.

**Signs.** Three facts about `Z/q`, `q` odd, carry the argument.

1. Two rows with phase difference `δ` share two columns, where the first row
   has the latent states `u1` and `u2`. In another completion with difference
   `δ'`, the first column leaves `δ' ∈ {δ, -δ, 2 u1 + δ, -(2 u1 + δ)}` and the
   second the same with `u2`. If `u1 != u2` and `u1 + u2 + δ != 0`, the false
   candidates of the two columns are disjoint, and `δ' = δ` or `δ' = -δ`.
2. If `δ != 0`, that sign is the sign of every column shared by the two rows:
   `δ' = ε δ` forces the state `ε u` in the first row.
3. A further row is fixed by two of its cells in columns of different phases
   `a != b`: a state `x'` with `x' + a = ±(x + a)` and `x' + b = ±(x + b)` is
   `x`, because the reflected candidates `-x - 2a` and `-x - 2b` differ.

All three use `2x = 0 => x = 0`. On four states the third fails:
`x = 1, a = 0, b = 2, x' = 3`.

**The certificate.** Proposed phases `ρ, σ : n -> Z/q`, and a rectangle: two
rows `top`, `bot` and two columns `left`, `right`. From the rectangle the
certificate *reaches* rows and columns:

- the rows `top` and `bot`;
- every column with training cells in both;
- a row with two training cells in reached columns `j, j'` with
  `σ(j) != σ(j')`;
- a column with two training cells in reached rows `i, i'` with
  `ρ(i) != ρ(i')`.

It is *accepted* when

```
d(ρ(i_t) + σ(j_t)) = y_t    at every sample of a reached row and reached column,
the four cells of the rectangle are training cells,
ρ(top) != ρ(bot),   σ(left) != σ(right),
ρ(top) + ρ(bot) + σ(left) + σ(right) != 0.
```

Nothing is asked of `ρ, σ` at other samples. Every condition is a finite
statement about the certificate, the decoder and the training samples; a
derivation of "reached" is a finite list of the steps above.

**Soundness.** Let the certificate be accepted, and let `r, s` be any phases
that fit all training samples. Then there are a sign `ε` and a shift `g` with

```
r(i) = ε ρ(i) + g  on reached rows,      s(j) = ε σ(j) - g  on reached columns,
```

so that

```
d(r(i) + s(j)) = d(ρ(i) + σ(j))    for every reached row i and reached column j,
```

whether the cell is observed or not. With `r, s` the phases of the target this
is its label. The proof is the three facts: the rectangle gives the difference
of its rows up to a sign (1), that sign fixes the columns observed in both
rows (2), and every further step is (3). Nothing is assumed about unobserved
labels, about the sign or latent state of any cell, about injectivity, or
about an optimizer. The statement is per cell: it says nothing about a row or
a column that the certificate does not reach.

**Acceptance from labels.** The conditions on `ρ, σ` follow from conditions on
training labels, for any phases that fit the samples:

- the columns `left`, `right` carry the label pairs `(a, b)` and `(c, e)` in
  the rows `top`, `bot`, with `(a, b) != (c, e)` and `(a, b) != (e, c)`; the
  first gives different column phases, the second the sum condition, because
  a vanishing sum exchanges the two pairs;
- some column has different training labels in `top` and `bot`, which gives
  different row phases.

In the same way, two columns have different phases as soon as some row has
different training labels in them, and two rows likewise. Replacing
"different phases" by these witnesses gives a reach relation on training
cells and labels alone. For a realizable table, the phases of the table with
such a rectangle form an accepted certificate, and every cell that the labels
reach is covered. So any two completions of the training samples have the
same label there. These conditions are sufficient; they are not claimed
necessary.

**Why the examples of CM15 and CM16 are excluded.** Their samples are three
cells of a two by two table: a connected graph with a perfect matching, and
no rectangle of training cells. No certificate is accepted on them, for any
decoder and any labels. Identification here is carried by a four-cycle with
label conditions, not by connectivity. The corner decoder `z != 2` of CM15 is
not even, so it is outside the class.

**The procedure.** `recover(d, cells, labels, i, j)` answers `unsupported`
(the decoder is outside the class), `inconsistent` (no phases fit the
samples), `uncertified` (no accepted certificate reaches the row `i` and the
column `j`), or `accept` with the label proposed by such a certificate.

- An accepted label is the label of every completion of the samples at
  `(i, j)`; for a realizable table it is the table's.
- For a realizable table the label conditions above give `accept`.
- Where two completions of the samples disagree, the answer is `uncertified`.

`recover` is a specification by search over phases and certificates. It is
not executable and no running time is proved. A certificate can be found by
propagation (this paragraph is not formalized): choose the rectangle, set the
phase of `top` to zero by a shift, read the two candidate states of each
column of `top` from its label, intersect the four candidates of the row
difference from two columns, choose its sign by a reflection, and continue
with one fiber look-up per training cell. Because acceptance asks for the
labels only on reached rows and columns, this needs no search over the
unreached part. The answer `inconsistent` does: deciding whether any phases
fit all samples is given here only by exhaustive search.

**In the convex programs.** The declared program is ordinary CE over the
whole vocabulary of `decode d P`, with `P` in the spectral domain of CM15 at
the budgets `N`; CM16 adds the equalities of an even latent table. For unit
codes and probability rows the operator bounds hold by themselves (entries of
modulus at most one), and the constant moment has nuclear norm `N` (CM13): at
the budgets `N` the domain is the probability tables whose nonconstant
moments have nuclear norm at most `N`.

1. *A minimizer with the target's answers.* For any phases that fit the
   training samples, the decoded table lies in the paired domain, minimizes
   the training CE over the whole CM15 domain, and has the one-hot row of the
   target's label at every covered cell.
2. *The relaxation has a wrong minimum.* `q = 3`, `d(z) = (z != 0)`, `N = 3`,
   `y(i,j) = (i != j)` with `r(i) = i`, `s(j) = -j`, and every cell observed
   but `(0,0)`. The certificate with the rows `1, 2` and the columns `0, 1`
   is accepted; it reaches the row `0` through the columns `1, 2` and the
   column `0`, and proposes `false` at `(0,0)`. Take the even latent table of
   the target and replace its row at `(0,0)` by the uniform one. Both
   nonconstant moments become the real matrix

   ```
     0   -1/2  -1/2
   -1/2    1   -1/2      =  3/4 v1 v1^T + 1/3 v2 v2^T - 1/12 v3 v3^T,
   -1/2  -1/2    1
   ```

   with `v1 = (0,1,-1)`, `v2 = (1,-1,-1)`, `v3 = (2,1,1)`, of nuclear norm at
   most `3/2 + 1 + 1/2 = 3`. The table is in both domains and fits the eight
   samples, so its decoded table is a global CE minimum, with probability
   `2/3` for the wrong label at a certified cell. Minimizing CE over the
   spectral relaxation does not identify even a certified answer.
3. *The hull of decoded phase tables.* Let `H` be the convex hull of the
   tables `d(r(i) + s(j))` over all phases. It depends on `d` and `N` only
   and lies in both relaxations. For an accepted certificate, a covered cell
   `(i, j)` with proposed label `c`, and every `Q` in `H`,

   ```
   1 - Q(i,j)(c)  <=  sum over samples t of  (1 - Q(i_t,j_t)(y_t)).
   ```

   Both sides are affine in `Q`. At a vertex that fits the samples the left
   side is zero by soundness; at a vertex that does not, the right side is at
   least one. Hence every table of `H` with the ideal training CE, and for a
   realizable table every minimizer of CE over `H`, has the one-hot row of
   the target at every covered cell. Latent probabilities are not identified
   and do not occur in the statement. The wrong table of point 2 is outside
   `H`. No way to project onto `H` or to optimize over it is given.
4. *The certified restriction.* Add to the spectral program the linear
   equalities that fix every covered row to the one-hot row of its proposed
   label. The labels come from the certificate, hence from training samples
   alone. The restricted domain is convex and contains the table of point 1;
   every minimizer over it minimizes CE over the unrestricted domain as well,
   and has the target's rows at the covered cells. The equalities are
   imposed: at the covered cells the answer is the certificate's, by
   construction. What is proved is that it is the target's, and that the
   restriction costs nothing in training CE.

These are statements about exact minimizers over the reals. The changed
families are `H` (point 3) and the restricted domain (point 4); for the
unchanged relaxation only point 1 and the obstruction of point 2 hold.

**Witness.** The folded table of CM16, `N = 3`, `q = 5`, three classes, with
the cell `(2,1)` withheld:

```
0 1 2
1 0 1
2 . 0
```

The rows `0, 1` and the columns `0, 2` carry the pairs `(0,1)` and `(2,1)`,
neither equal nor exchanged, and the column `0` separates the two rows. The
column `1` is observed in both rows. The row `2` has training cells in the
columns `0` and `2`, where the row `0` has the different labels `0` and `2`.
So the labels reach `(2,1)`, and every completion of the eight samples has
the label `1` there; the reflected and shifted phases `(1 - i, j - 1)` are
one, with the latent state `4` in place of `1`. The training cells `(1,0)`
and `(0,1)` have the latent states `1` and `4` and the same label, so the
decoder's collisions are in use. `recover` answers `accept 1`. The folded
table is a feasible CE minimizer, and every minimizer over `H` and over the
restricted domain has the one-hot row of `1` at `(2,1)`.

**Refusals.** On the samples of the CM16 limit `recover` answers
`uncertified` at `(1,1)`, where its two completions disagree, and no accepted
certificate covers that cell. With the corner decoder of CM15 it answers
`unsupported`. One cell with two labels is `inconsistent`.

**For task #195.** The interface is: a certificate `(ρ, σ, top, bot, left,
right)`, its checker (`Accepts`, with the reach relation `Pinned`), and the
soundness statement for every completion (`accepted_decoding`). In the convex
program it enters as the restriction of point 4 (`certified_restriction`); a
solver that fixes the covered rows stays inside the program of CM15 and CM16.
The characterization of the domain at the budgets `N` by the nonconstant
nuclear norms alone (`budget_iff_nuclear`) is the condition that the native
audit of `experiments/convex_stress_paired` checks numerically.

**What remains open.**

- The structural hypothesis is about the unknown truth and cannot be tested
  on the training set. For a table outside the family an accepted label need
  not be the table's.
- The decoder class: a fixed known decoder of an odd cyclic group whose
  fibers are exactly the opposite pairs. Larger fibers, an even `q`, other
  groups, an unknown decoder and the decoder hull of CM15 are not treated.
- Coverage: the label conditions are sufficient ones. There is no statement
  about random splits, and which tasks of the stress suite belong to the
  family or meet the conditions is not examined.
- The procedure is a specification. The propagation and its cost, and any
  efficient test of consistency, are not formalized.
- The minimizers of the spectral relaxation itself are not identified (point
  2). Which minimizer an iterative solver approaches, and whether a wrong
  minimum occurs at the original sizes, is not examined. Optimization over
  `H` is not addressed.
- Exact reals throughout: no tolerance, no stability bound, no convergence of
  the full-batch, minibatch or forward-only iterations.

`CoupledMoments/DecodedRecovery/Signs.lean` proves the three facts.
`Certificate.lean` defines the certificate and proves the rigidity of the
rectangle; `Rigidity.lean` proves soundness. `Coverage.lean` proves
acceptance from labels, and `Learner.lean` the statements about `recover`.
`PhaseHull.lean` proves point 3, `CrossEntropy.lean` points 1 and 4.
`Domain.lean` proves the description of the domain at the budgets `N`.
`ZeroMass.lean` and `Obstruction.lean` prove point 2, and `Witness.lean` the
witness and the refusals.

### CM19. Finite support propagation for general decoder fibers

CM18's opposite-pair restriction is sufficient for its rigidity theorem, not
necessary for every identifiable decoded answer. Here the fixed categorical
decoder `d : ZMod q -> C` is arbitrary and `q > 0` may be even. The unknown
target is still assumed to have the form

```
y(i,j) = d(r(i) + s(j)).
```

Only the training cells and their labels are supplied to the procedure. The
phases are not supplied. A finite bank `d_k` is also allowed: the structural
assumption is that the target uses some member of that declared bank, with
unknown index. Finding a fitted member is not evidence that this assumption
holds for the unseen labels. Flat spectral bounds alone still do not imply it.

**A finite computation.** Choose one observed anchor `(a,b)` with label `c`.
For every `z` with `d(z) = c`, start a branch of row and column phase domains:

```
R_a = {0},  S_b = {z}; all other domains are ZMod q.
```

One simultaneous sweep uses the previous domains on both sides. Retain
`u` in `R_i` exactly when, for every observed incident cell `(i,j)` with
label `c_ij`, there is `v` in `S_j` with `d(u+v) = c_ij`. Retain `v` in
`S_j` by the corresponding condition at every observed row. Repeated
observations are checked separately. There is no constraint for an
unobserved cell and no access to its answer.

Run a specified natural number of sweeps, including possibly zero. Reject
a branch if any of its variable domains is empty. At query `(i,j)` its
output set is `{d(u+v) : u in R_i, v in S_j}`; rejected branches contribute
the empty set. Take the union over **every** anchor phase. For a bank take
the further union over **every** decoder. These operations are finite and
computable in Lean. No fixed-point or efficient implementation theorem is
needed for the stated finite-budget result.

**Preservation and identification.** Every assignment fitting all observed
labels survives in one of these branches at every budget. To prove this,
normalize it to

```
r'(i) = r(i) - r(a),   s'(j) = r(a) + s(j).
```

All sums are unchanged. Its branch has `z = r(a)+s(b)`, which occurs in
the anchor fiber. At each observed cell the assignment's own neighboring
phase supplies the required support, so a simultaneous sweep preserves
every one of its coordinates. Induction gives preservation for any finite
number of sweeps. In particular its domains never become empty, and its
decoded answer occurs in the final output union.

Consequently:

- A singleton union `{c}` certifies label `c` for every compatible phase
  assignment and, in the bank version, every compatible bank member.
- An empty union at any query refutes existence of a fitting assignment
  in the declared family. Pruning need not detect every inconsistency.
- If two actual fitting assignments give different query answers, the
  union is never a singleton at any budget. The same holds when the two
  assignments use different decoder choices.
- A nonsingleton union alone does **not** prove actual ambiguity: the
  Cartesian products may contain combinations with no global completion.

The distinction between decoded and latent identification is essential.
Neither a singleton row domain nor a singleton column domain is required
to certify a label. Pruning is conservative; completeness and coverage on
a particular split are separate questions.

**The convex spectral program.** Let `D` be the spectral decoder hull of
CM15 with Fourier codes on `ZMod q` and both spectral budgets equal to
`N`, the number of rows/columns. Its definition already includes all fixed
decoders; it is convex and has probability-vector rows. Restrict it to

```
D_cert = {Q in D : Q(i,j) = oneHot(c)
                  whenever the bank's output union at (i,j) is {c}}.
```

The labels on the right are computed from training data by the procedure
above. These are linear equalities, so `D_cert` is convex. Use the same
ordinary full-vocabulary log-sum-exp cross-entropy as CM2: the learned
probability coordinates serve as bounded logits. The loss is convex on
the restricted domain; it has not been changed into a latent regression
loss or `-log Q(i,j)(c)`.

If any bank member and phases fit the observations, their decoded one-hot
table belongs to `D_cert`. This follows from CM15's feasible phase witness
and the preservation theorem above. That table already minimizes the
training CE over all of `D`, so it also minimizes it over `D_cert`.
Conversely every minimizer over `D_cert` is a minimizer over `D`: compare
it with this retained witness. Every feasible table in `D_cert` has the
target's rows at certified cells whenever the target belongs to the bank.

The equalities are imposed. The result proves their correctness under the
explicit target-family assumption and that they cost no training CE. It
does not say that minimizing the original spectral relaxation discovers
them, that a preferred fitted decoder is true, or that every decoder in
the larger hull belongs to the declared identification bank.

**Witness outside CM18.** Take `q = 4` and `d(3) = true`, with
`d(0) = d(1) = d(2) = false`. Observe only

```
true   true
false  .
```

Anchor at the false observation `(1,0)`. There are three branches,
`z = 0,1,2`. In each, the first sweep gives `r(0) = 3-z`; the second
gives `s(1) = z`. The missing decoded answer is false in all branches.
All three latent states really remain possible there: the assignments
`r = (3-z,0)`, `s = (z,z)` fit for each `z`. The concrete finite computation
is checked by Lean's kernel using `decide`, and `wide_recovery` proves
agreement of every completion. The decoder is not a `PairedDecoder`:
`0` and `1` collide but are not opposite. Adding `z -> d(z+1)` as a second
bank member preserves the certified answer while allowing a different
latent explanation.

For refusal, replace the three observations by false. The assignments
`r=(0,0), s=(0,0)` and `r=(0,1), s=(0,2)` fit, but the missing answers
are false and true. No number of sweeps certifies a singleton. A bank
containing the original decoder and `z -> (z=1 or z=3)` also has two
compatible members with different answers. Finally, two opposite labels
on the same cell yield an empty union after one sweep, both for the fixed
decoder and for the witness bank.

**Scope.** This extends fixed known decoders to arbitrary fibers, permits
even cyclic groups, gives a computable finite procedure, and quantifies
over a declared finite bank. It does not learn an unrestricted decoder,
prove the target-family assumption from observed data, prove completeness
or a coverage rate, or give numerical error or optimizer convergence
bounds. No original stress-task pass is implied by the finite witness.

`CoupledMoments/DecodedPropagation/Defs.lean` defines the computation;
`Basic.lean` proves preservation, shrinking domains and normalization;
`Identification.lean` proves singleton identification and the refusals;
`CrossEntropy.lean` proves the convex restriction and optimality;
`Witness.lean` and `Refusal.lean` check the explicit examples.

**Endpoint construction for a solver.** A surviving branch also gives a
feasible fallback without solving for a complete training fit. Choose any
one state from each of its nonempty final row and column domains. At every
query its decoded sum occurs in that branch's Cartesian output set, hence
in the bank union. Therefore it agrees with every certified singleton,
whether or not these chosen phases fit the training samples. Its one-hot
latent phase table is spectrally feasible at the budgets `N` by CM14, and
its decoded table lies in both its fixed-decoder domain and `D_cert`.
Every live branch supplies such a point (`branch_endpoint`, `live_endpoint`).

This is a fallback, not necessarily a CE minimum. In the four-state witness
at budget zero, anchoring at the false observation with state zero and
choosing all phases zero keeps the certified anchor but predicts false at
the two observed true cells. The choice `r=(3,0), s=(0,0)` in that same
unpruned branch does fit them. These two feasible decoded tables differ
at an observed, uncertified cell, while agreeing at every certified cell.

If two output tables have the certified rows, every affine mixture
`(1-t) P + t Q` keeps those rows (`certified_rows_blend`). With `0 <= t <= 1`
the simplex and spectral constraints are also preserved when both tables
are feasible, by their existing convexity theorems. When a proposed iterate
is spectrally infeasible, this does not prove a positive mixing weight
feasible: a numerical endpoint routine must check its returned coordinates
and may fall back to weight zero. No floating-point error theorem follows.

A solver using one selected decoder optimizes only its fixed-decoder convex
domain. Its answer constraints must still be computed from the **whole**
declared bank. The bank-level optimality theorem above does not identify a
selected fitted decoder as the true one or prove that its smaller domain
attains the optimum over the full bank hull. This distinction must remain
visible in native candidate selection and reported CE.

`CoupledMoments/DecodedPropagation/Endpoint.lean` proves the endpoint facts
and the partial-coverage witness before the general native solver is added.

**Anchor coverage can change at a fixed point.** On three rows and columns,
use `d : ZMod 3 -> Bool`, with `d(0)=false`, `d(1)=d(2)=true`. Observe every
cell except `(0,0)`, with false on the observed diagonal and true elsewhere.
The phases `r=(0,1,2)`, `s=(0,2,1)` fit these observations. Anchoring at
`(1,1)` has only the phase-zero branch. After one sweep its row and column
domains are both `({1,2},{0},{1,2})`. These domains are a fixed point of the
support operator, and the missing cell's output union remains `{false,true}`
for every positive sweep budget. Increasing that budget does not help.

Anchoring instead at the observed `(0,1)` branches on both phases 1 and 2.
After three sweeps the union at `(0,0)` is `{false}`. Preservation therefore
proves that every compatible completion has that label: the first anchor's
nonsingleton was an overestimate, not actual ambiguity. No extra labels or
preferred branch were used. `DecodedPropagation/Anchor.lean` proves the
finite computations, the fixed point for all budgets and identification.
This example motivates trying more than one observed anchor; it does not
prove completeness of an anchor policy or change the declared-family assumption.

### CM20. Extracting a Fourier direction by constant amplitude

The categorical latent-decoder constructor of CM15 proposes phase orders from
a singular vector of a centered label sketch. A repeated singular value leaves
a choice of directions inside its singular subspace. This section studies a
separate mathematical question: when does constant amplitude inside an exact
Fourier subspace identify a single character? Recovery of that subspace from
the incomplete observed table remains a separate, unproved step.

Let `q > 0`, let `I` be a finite index set and let `k : I -> ZMod q` list
frequencies. Write `chi(t) = exp(2 pi i t/q)` and

```
f(x) = sum_i a_i chi(x k_i).
```

Require **separated differences**: for every `i != j` and every `u,v`,

```
k_j - k_i = k_v - k_u  implies  i = u and j = v.
```

The comparison includes diagonal pairs `u=v`, so this condition also excludes
zero off-diagonal differences. It is a property of the whole proposed exact
subspace, not just the coefficients that happen to be nonzero. It is sufficient;
no necessity or coverage theorem for observed singular subspaces is claimed.

**Extract one direction.** If at least one `a_i` is nonzero and `|f(x)|^2`
is the same real number at every `x`, then exactly one coefficient is nonzero.
Consequently `f(x) = a_i chi(x k_i)` for that index (`flat_extract`). The
modulus need not be prime. A permutation `r : X <-> ZMod q` of the categorical
rows changes neither the hypothesis nor the conclusion: constant amplitude of
`f(r(x))` identifies a single permuted character (`permuted_extract`). The
permutation is a structural hypothesis and is not supplied to the procedure.

If additionally `sum_x f(x) = 0`, then the extracted `k_i` is nonzero
(`centered_extract`). For a prime modulus that character, multiplied by its
nonzero coefficient, takes distinct values at all `q` states
(`centered_injective`). These facts explain the roles of centering and the
prime-modulus restriction when a native proposal needs a phase permutation.

**Proof from the actual amplitude.** Expand

```
|f(x)|^2 = sum_i sum_j conj(a_i) a_j chi(x (k_j-k_i)).
```

Orthogonality of the primitive character gives the unnormalized coefficient

```
sum_x conj(chi(x t)) |f(x)|^2
  = q sum_{i,j : k_j-k_i=t} conj(a_i) a_j.
```

The left side is computed from the squared amplitude, without presupposing
sparse coefficients. If the amplitude is constant, it vanishes for every
`t != 0`. Separation isolates the product `q conj(a_i) a_j` at the difference
of every distinct pair. Since `q != 0`, all these products are zero. Choosing
one nonzero coefficient forces all the others to vanish. Conversely, a vector
supported at one mode has constant squared amplitude `|a_i|^2` (`support_flat`).
Centering rules out the constant character; injectivity in the prime case
follows from injectivity of the primitive character and multiplication by a
nonzero residue. These are proved in `FourierExtraction/Basic.lean` and
`Extraction.lean`, using the existing character orthogonality theorem.

**A quantitative bound in the same exact subspace.** For any real `rho` and
distinct `i,j`, separation also gives

```
q |a_i| |a_j| <= sum_x abs(|f(x)|^2 - rho).
```

Subtracting `rho` does not change a nonconstant Fourier coefficient. Apply
the triangle inequality to that coefficient and use `|chi|=1`. In particular,
if every squared-amplitude error is at most `epsilon`, then
`|a_i| |a_j| <= epsilon` (`pair_le_deviation`, `approximate_pair`). This bounds
remaining cross-mode products. It does not include subspace estimation error,
a floating-point error enclosure or convergence of an extraction iteration.

**Nonempty example and a limit.** For `q=13`, frequencies `1,3,9` have six
distinct nonzero ordered differences (`triple_separated`, checked by `decide`).
Coefficients `(1,0,0)` give a centered, unit-amplitude, nonconstant vector that
distinguishes all thirteen states. Coefficients `(1,1,0)` cannot have constant
amplitude (`mixed_not_flat`). Thus the condition admits a three-dimensional
subspace and excludes a genuine mixture within it.

For `q=2`, take frequencies `0,1` and coefficients `(1+i)/2, (1-i)/2`.
Both coefficients are nonzero, but the vector takes values `1,i`, of unit
amplitude. The two off-diagonal differences coincide. `collision_flat` and
`collision_two_modes` prove the counterexample, so the separation assumption
cannot simply be dropped. `Witness.lean` instantiates every conditional result.

**Convexity limit.** The coefficient set defined by unit amplitude is not
convex (`amplitude_not_convex`). For any listed frequency, its positive and
negative pure coefficient vectors both have unit amplitude. Their midpoint
is zero, which has zero amplitude. Therefore the new criterion cannot be
imposed directly as a convex constraint on the coefficient vector. Alternating
amplitude and subspace projections would be an auxiliary nonconvex proposal,
with no convergence theorem here. A convex formulation needs a separate lift
and proof before it is used to change the model.

**Remaining gap.** A future extraction proposal must keep right phases tied to
the same left direction through the original sketch. Existing observed-label
decoder fitting can score resulting phase orders by ordinary CE, with no
arithmetic meaning supplied for input tokens. This section supplies the
conditional direction-identification result, not that complete learner.

The extraction theorem does not identify the true decoder, prove the target's
membership in a chosen family, prove that a masked sketch's leading subspace
contains the needed character, or identify unseen labels from amplitude alone.
A native experiment must retain every attempted subspace and failed proposal,
its actual work and memory use, and every earlier candidate. Any returned latent
table still needs the CM15 spectral/CE audit; new original-data scores require
a separately frozen protocol and replay. The current frozen cohorts are unchanged.

### CM21. Convex density extraction in a supplied Fourier subspace

CM20's unit-amplitude constraint on vectors is nonconvex. A matrix lift avoids
that obstruction without imposing rank one. This section proves the exact
geometry and a finite policy of two linear objectives. It assumes a supplied
exact subspace; it does not recover that subspace from missing labels.

Let `q > 0`, let `I` be finite, let `r : X <-> ZMod q` be an unknown row
permutation and let `k : I -> ZMod q` have CM20's separated ordered differences.
Use unnormalized Fourier features

```
F[x,i] = chi(r(x) k_i).
```

The supplied feature matrix is `B = F Q`, where `Q` is unitary:
`Q* Q = Q Q* = Id`. Here `*` denotes conjugate transpose. The permutation,
frequencies and unitary are structural witnesses in the theorem, not inputs
to the optimization policy. The chosen normalization gives pure kernel
diagonal one. For a numerically supplied orthonormal basis `U`, the equivalent
scaling is `B = sqrt(q) U`; the exact equality to `F Q` is still a hypothesis,
not a conclusion about an estimated singular subspace.

**Convex matrix domain.** The coefficient variable `M` is Hermitian positive
semidefinite with `trace(M)=1`. Its observable kernel is `K = B M B*`.
Require the affine equalities `K[x,x]=1` for all rows. `Density`, `reading`
and `Feasible` use these actual matrix operations. `density_convex` and
`feasible_convex` prove convexity directly; `reading_mix` proves linearity
of every kernel reading. There is no rank constraint and no presumed vector
factorization in the domain.

In hidden Fourier coordinates `H = Q M Q*`, unitary conjugation preserves
positivity and trace, and the observable kernel equals `F H F*`
(`density_conjugate`, `reading_change`, `feasible_change`). Its diagonal is

```
K[x,x] = sum_i sum_j H[i,j] chi(r(x) (k_i-k_j)).
```

The orientation is `k_i-k_j`, because this is `F H F*`; CM20's squared-vector
formula used conjugated coefficients in the opposite order. Character
orthogonality gives

```
sum_z conj(chi(z t)) (F H F*)[r^-1(z),r^-1(z)]
  = q sum_{i,j : k_i-k_j=t} H[i,j].
```

If the diagonal is constant, its nonzero Fourier coefficients vanish.
Separation isolates `q H[i,j]` for each `i != j`; hence `H[i,j]=0`
(`fourier_diagonal`, `diagonal_coefficient`, `flat_off_diagonal`). This argument
applies to an arbitrary matrix; positivity and rank are not used to obtain
the vanishing cross terms.

Positivity then makes the remaining diagonal entries real and nonnegative;
trace one makes their sum one. Conversely, every such diagonal matrix has
constant readings. Thus `feasible_iff_diagonal` characterizes all feasible
matrices by real probability weights. `feasible_hull` and `rotated_hull`
identify the complete convex sets:

```
{H feasible in F} = convexHull {E_i : i in I},
{M feasible in B} = convexHull {Q* E_i Q : i in I},
```

where `E_i` is the projector onto coordinate `i`. These are equalities of
sets, not an assumption that an optimizer is an extreme point. In particular,
all mixtures remain feasible.

**Two observable linear objectives.** Choose any two distinct categorical
rows `a,b`. First maximize `Re K[a,b]` on the feasible domain. Then maximize
`Im K[a,b]` on the first optimum face. Both objectives are real linear and
the second domain adds one affine equality to a convex set. `LexMaximum`
states exactly this two-stage specification, with comparisons against every
feasible matrix. It does not presume that any numerical iteration satisfies it.
This finite policy avoids assuming an unknown mode-dependent separating
objective or proving a particular rotated-angle tie-break formula.

At a pure mode, the complex score is

```
s_i = chi((r(a)-r(b)) k_i).
```

For prime `q`, distinct rows and separated frequencies give distinct scores
(`score_injective`). For any finite nonempty index set there is an index with
maximum real score and, among the tied indices, maximum imaginary score
(`exists_lex_vertex`). Its pure projector attains both objectives
(`exists_fourier_max`, `exists_rotated_max`), so existence is derived.

For any feasible mixture, equality to the maximal real average forces every
positive-weight component onto the first optimum face. Equality in the second
average forces those components to have the same imaginary score as well
(`weighted_bound`, `weighted_support`). Distinct complex scores therefore
leave only one nonzero weight, equal to one (`lex_weights_pure`). Consequently
every exact two-stage optimizer is a pure mode projector, including in the
supplied rotated basis (`fourier_max_pure`, `lex_change`, `rotated_max_pure`).
No rank-one optimizer or correct unseen answer is supplied as a hypothesis.

**Observable centering.** If every column of `B` sums to zero, multiplying
by `Q*` preserves those zero sums. Orthogonality then rules out `k_i=0` for
every column (`centered_mul`, `centered_nonzero`). A kernel column from an
extracted pure mode is `chi((r(x)-r(c)) k_i)` for its anchor row `c`. With
prime `q` and `k_i != 0` these values distinguish all rows
(`extracted_column_injective`). `supplied_extraction` combines existence and
injectivity using observable centering and the explicit exact-subspace
assumption. It identifies a relative phase representation, not a decoder.

**Examples and boundary.** The separated frequencies `1,3,9` modulo 13 admit
the mixed density `diag(1/2,1/2,0)`, which is none of the pure projectors
(`mixed_feasible`, `mixed_not_pure`). The rational unitary rotation with block
`[[3/5,-4/5],[4/5,3/5]]` mixes two hidden modes and fixes the third. Its supplied
columns are centered; `witness_extraction` proves an optimizer with an
injective phase column for this genuinely rotated three-mode input.

For the colliding frequencies `0,1` modulo two, lift CM20's vector
`a=((1+i)/2,(1-i)/2)` to `H=a a*`. `vector_reading` verifies its diagonal
readings equal the actual squared amplitude. `collision_feasible` proves
positivity, trace one and both constant readings. Its off-diagonal entry is
nonzero (`collision_cross_term`), so it is not a diagonal probability mixture
(`collision_not_diagonal`). Separation cannot simply be removed from the
characterization. Closed examples instantiate every conditional theorem.

**Remaining work before a new experiment.** This proves an exact convex
extraction problem, not a method for recovering its input subspace from a
masked sketch. The separated-difference condition is sufficient and may fail
for a larger chosen spectral subspace. Estimated subspaces, floating-point
feasibility, finite iteration budgets and approximate lexicographic optima
need separate analysis or explicit refusal/tolerance accounting. The proof
does not justify treating an approximate mixed solution as an exact mode.

A bounded Rust/Kernite follow-up must record the chosen subspace, centering,
normalization, PSD/trace/diagonal residuals, both objective stages and every
failed attempt. Right-side phases must be read through the same original
sketch, not extracted independently. Any phase/decoder proposal still needs
ordinary-CE scoring, the existing latent-table spectral audits and replay.
Retain the earlier candidates, the full35 short/long scope, all three update
modes, the conventional-transformer comparison and the 64 GiB process cap.
The current frozen producers and scores are unchanged by this proof.
