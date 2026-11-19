# Why x³ + xy resists: a flat spectrum

Source note for `Transformer.Grokking.FlatSpectrum`, received from the user on
2026-10-28 and kept verbatim below, except for one optimizer name taken out of *How it
fits together* on 2026-11-19. Section names (Setting, Lemma 1, Proposition 2, Theorem A,
Corollary, Theorem B, Readout-free version) are the labels the Lean docstrings cite. The
section *Torch check* after the note keeps another session's recomputation of it, and
*Lean status* maps each claim to its declarations.

Sections on other subjects moved out of this note on 2026-11-19.
[`flat_spectrum_moved.md`](flat_spectrum_moved.md) lists them, with their labels and the files
that hold them now.

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
- Theorem A depends only on the objective and Theorem B only on the labels. That fits AdamW and a second optimizer failing the same way.

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
norm of the answer kernel by the pure-factorization lemma. The optimizers that train it
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
three aliasing cases and three controls with seeds 17 and 23; compare three optimizer
rates in the aliasing cases. Fourier analysis uses the reachable
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
parameter penalty with `Φ`, proves convergence of its optimizer, or guarantees a
width/depth-dependent success rate. The new margin results can be applied to actual logits
once their error relative to an ideal score construction is bounded. A small discarded
spectral-energy fraction alone does not supply that uniform error bound.

The cyclic frequency-gcd formula, the general powered-division indicator rank
`(p−1)/gcd(k,p−1)`, and optimal margins for full Fourier banks are outside this Lean
extension. Item 7 formalizes the full-rank indicator comparison at `k=1`; it does not
silently treat the more general formula as proved in Lean. Nonzero operands are explicit
throughout item 6; the arithmetic theorem covers zero numerators. The actual-network
extension, *Actual network expressivity for every exponent* in
[`actual_network/sequence_models.md`](actual_network/sequence_models.md), covers them at every
natural exponent.

## Experimental corrections to learning implications (2026-11-01)

The actual normalized RoPE source transformer was trained on division and
`a^3 + ab mod P`, including approximately parameter-matched
W256D1/W128D4 models. A frozen 48-case screen, profiles, and a separate ten-case
full-state continuation are recorded in `experiments/modular_composition/`
and `modular_composition_long/`. The continuation preserves the whole optimizer
state, the sampler, labels and splits. Python 3.15,
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

The residual audit and the known-solution depth-attention toy of the same day are in
[`residual_connections.md`](residual_connections.md), *Repeated-input numerical audit* and
*Normalized depth attention*; the two paragraphs that summarized them here are at its end.
Their point for this note: initialization, parameter scaling, optimizer dynamics and execution
fidelity are further premises of learning, not consequences of flat or rank-one spectra.

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
most two bindings (AR8). With it, at a scale `β` at which its sign recalls, the attention has
no cut width `d < m`: by AR5 a readout of cut width `d` whose sign recalls `m` bindings has
`m ≤ d`. No `d` numbers formed from the dictionary before the query, read linearly by it,
give the attention; it forms its summary of the context at the query. So for one layer in
exact arithmetic the second half of Claim 1 of the Zoology paper, "attention models can solve
it with near-constant dimensionality", holds with constant dimension; its first half for one
layer is AR6.

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

The average over dictionaries halves what the worst one costs. Bind the queried key to `1` and
every other key to `−1`. Then `a = −1`, and with `R` the weight of the other tokens and `Z` the
sum of the weights, `1 − att = 2R / Z ≥ 2ρ`. Recall of values `±1` to accuracy `ε` at every
pair has `2ρ ≤ ε`:

```
(m − 1)(2 − ε) ≤ ε e^{2B},
```

and recall by the sign, an error below `1` everywhere, has `m − 1 < e^{2B}`. Against this bound
the one-hot scale `e^{2B} = 2 (m − 1) / ε` is sharp up to the factor `2 / (2 − ε)`. At a fixed
`B` the error of the worst dictionary tends to `2` as `m` grows: for every `ε < 2`, from some
`m` on, every query has a dictionary answered with an error above `ε`. The answer tends to the
opposite value.

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

### SA5. A scale that grows with the length

SA3 asks the scale of the scores to grow with `m`. Some long-context models grow it with the
length by construction. arXiv:2512.01868v4, §6 (*Long context Transformers*), names Qwen,
SSMax and SWAN-GPT among the systems with the scale `β_n = γ log n` over `n` tokens, so that

```
A_ij = n^{γ ⟨x_i, x_j⟩} / Σ_k n^{γ ⟨x_i, x_k⟩}.
```

Write `att_s(z, v)` for softmax attention with the scores `s log n · z_j`, from raw scores
`z_j`. At one length it is softmax attention, and SA1 to SA4 hold for it. Across lengths the
scale grows by itself. If `z_j + g ≤ z_{j₀}` for every `j ≠ j₀` and `|v_j| ≤ C`, SA1 with the
gap `s g log n` gives

```
|att_s − v_{j₀}| ≤ 2 C (n − 1) n^{−s g}.
```

For `s g > 1` the right side tends to `0`. On recall with `m + 1` keys, raw scores `z(q, k)`
that put the queried key ahead of every other key by `g` recall values bounded by `C` to every
accuracy from some `m` on, with one `s` and one `g` at every length. The growth of the scale
that SA3 asks for comes from `log n`; the raw scores need only a fixed gap.

Below the threshold the scale does not suffice. Raw scores bounded by `B` give scores bounded
by `s B log n`, so the worst dictionary of SA3 has an error of at least

```
2 (n − 1) / (n^{2 s B} + n − 1),
```

which tends to `2` when `2 s B < 1`: for every `ε < 2`, from some `m` on, every query has a
dictionary answered with an error above `ε`. One-hot raw scores, `B` at the queried key and
`−B` elsewhere, lead by `g = 2B`. They fade for `2 s B < 1` and recall for `2 s B > 1`. This is
the threshold `s g = 1` of the equiangular model of that section, whose gap is `1 − ρ` and whose
threshold is `γ (1 − ρ) = 1`.

For one-hot raw scores the worst error is exact. Softmax attention misses the value of a token
by at most `2 C` times the weight of the other tokens, for values bounded by `C`, and the
indicator dictionary misses it by exactly twice that weight. Over `n = m + 1` keys the other
keys hold the weight `m / (n^{2 s B} + m)`, so the greatest error over the dictionaries of
values `±1` is

```
2m / (n^{2 s B} + m)
```

at every `s`, `B` and `m`. It tends to `2`, `1` and `0` as `2 s B` is below, at and above `1`.
At the threshold `2 s B = 1` it is `2m / (2m + 1)`. That is below `1`, so every dictionary is
answered with the right sign at every length; and from some `m` on it exceeds every `ε < 1`, so
no accuracy better than `1` holds at every length. The queried key keeps the weight
`n / (2n − 1)`, which tends to `1/2`, as the diagonal weight of the equiangular model does at
`γ (1 − ρ) = 1`.

The gap that the recall statement asks for is limited by the geometry of the raw scores. Let
the raw score of the query of the key `i` at the key `j` be `⟨u_i, v_j⟩`, with `‖u_i‖ ≤ a` and
`‖v_j‖ ≤ b`. If every key leads every other key by more than `a b` at its own query, the keys
are affinely independent, so in dimension `d` there are at most `d + 1` of them. Proof: Radon's
theorem splits affinely dependent keys into two parts whose convex hulls share a point `x`; a
key of the first part leads `x` by more than `a b`, so `‖v_i − x‖ > b ≥ ‖v_i‖` and
`⟨x, v_i⟩ < ‖x‖² / 2`, and that convex half-space then holds `x` itself, which is absurd. The
diff-gpt model of `Transformer.MHC.PeriRMS` takes `s = 1` and the raw score `(1/d) ⟨q̂, k̂⟩`
with `‖q̂‖, ‖k̂‖ ≤ √d` after Q/K RMS and rotary phases, for the head width `d`. So the raw gap
above `1` that it needs holds for at most `d + 1` keys at once, whatever the training; at
`d = 1` two keys reach the bound.

A binding may take two tokens, the key and then its value. Two layers recall it, as an
induction head does: layer 1 attends at every token to the token before it and copies the key
channel, and layer 2 attends at the query with the coordinate of the query in the copied key
channel and reads the value channel. The error of layer 1 enters the raw scores of layer 2. If
layer 1 leads by `g₁` and the matches, bounded by `H`, lead by `g₂`, layer 1 is within
`δ = 2 H (n − 1) n^{−s₁ g₁}` of its target, layer 2 leads by `g₂ − 2δ`, and the error is at
most `2 C (n − 1) n^{−s₂ (g₂ − 2δ)}`. One-hot keys and positions give `g₁ = g₂ = H = 1`, so with
`s₁ > 1` and `s₂ > 1` two layers recall at every large number of keys. The context is read
cyclically, and attention is not masked.

Layer 1 has one-hot positional raw scores and copies one-hot key channels, so both layers can be
computed exactly. Raw scores `a` at one token and `b` at the other `n − 1` give that token the
weight `n^{s (a − b)}` against `1` for each other token. Layer 1 copies the key `q` to its value
token with the weight `n^{s₁} / (n^{s₁} + n − 1)` and to every other token with
`1 / (n^{s₁} + n − 1)`. So the raw scores of layer 2 lead by `a = (n^{s₁} − 1) / (n^{s₁} + n − 1)`,
and the induction head is one-hot attention at the scale `s₂ a`. Its error is
`(n φ(π q) − Σ_i φ(π i)) / (n^{s₂ a} + n − 1)`, at most `(3m + 1) C / (n^{s₂ a} + 2m + 1)` for
values bounded by `C`, and with `C = 1` that bound is the worst error over the dictionaries of
values `±1`, attained by the indicator dictionary. The key tokens carry the value `0`, so the
bound tends to `3/2`, not `2`, when `n^{s₂ a}` grows slower than `n`.

So the limit of the lead decides the head. Since `a = 1 − 1 / (n^{s₁ − 1} + 1 − 1/n)`, the lead
tends to `0`, `1/2` or `1` as `s₁` is below, at or above `1`: the limits of the diagonal weight of
the equiangular model at `ρ = 0`. If the lead tends to `a`, the head recalls when `s₂ a > 1`.
When `s₂ a < 1`, from some number of keys on, every query has a dictionary of values `±1` that it
misses by more than any given `ε < 3/2`. So `s₁ > 1` is not needed at layer 1: at `s₁ = 1` the
copy keeps half its lead, and `s₂ > 2` recalls. But `s₁ ≥ 1` is needed: below it the copy loses
its lead, and no `s₂` recalls. At `s₁ = 1`, `s₂ < 2` fades, and above it `s₂ < 1` does.

With a causal mask the token at the position `p` reads the `p + 1` positions up to it, at the
scale `s log (p + 1)`. Over `b` tokens without a key (a beginning-of-sequence token when `b = 1`)
and then `k₀ v₀ k₁ v₁ …`, layer 1 copies the key `c` to its value token with the weight
`L^{s₁} / (L^{s₁} + L − 1)`, where `L = b + 2c + 2` is the length of its prefix: the copy of the
cyclic head over `L` tokens. So the copy of a key is set by its position, not by the length of the
context, and over the keys it tends to `0`, `1/2` or `1` as `s₁` is below, at or above `1`.
Above `s₁ = 1` the late keys copy almost perfectly and the first only at
`(b + 2)^{s₁} / ((b + 2)^{s₁} + b + 1)`. The key token of `c` copies itself with
`1 / (K^{s₁} + K − 1)` over its `K = b + 2c + 1` positions, the tokens after the value token copy
it with `1 / ((p + 1)^{s₁} + p)`, and the tokens before the key token do not see it.

Layer 2 at the query `q` reads the `n = b + 2m + 2` tokens at the scale `s₂ log n`, with the
copies of `q` as raw scores. When the target has the value `1` and the other tokens at most `0`,
attention misses it by at least the weight of the other tokens. Here the other `n − 1` tokens have
raw scores at least `0`, so the indicator dictionary is answered at `q` with an error at least
`(n − 1) / (n^{s₂ w_q} + n − 1)`, where `w_q` is the copy of `q` at its value token. So the head
fades at every key with `s₂ w_q < 1`: for every `ε < 1`, from some number of keys on, every key
with `s₂ w_q ≤ σ < 1` has a dictionary answered with an error above `ε`. The copy is set by the
position of the key, so the condition holds or fails key by key. Below `s₁ = 1` the copies of the
late keys tend to `0`, and at every `s₂` the head fades at every key from some key on.

Conversely, at `s₁ ≥ 0` no token but the value token copies `q` more than its key token, which
copies it with `r_q = 1 / ((b + 2q + 1)^{s₁} + b + 2q)`, and the token at the position `p` after
the value token copies it with at most `1 / p`. Splitting the tokens at a position `P` at or
after the value token, the head answers `q` within
`2 C ((P + 1) n^{−s₂ (w_q − r_q)} + n^{−(s₂ w_q − 1 − s₂ / P)})` for values bounded by `C`. So
if `r_q < w_q` and `s₂ w_q > 1`, the key `q` is recalled to every accuracy from some number of
keys on. After a beginning-of-sequence token at `s₁ = 2` and `s₂ = 11/10`, the fourth key is
recalled and the first fades.

The second condition is needed: if the key token copies `q` with `r_q > w_q`, the error at `q` is
at least `1 / (n^{−s₂ (r_q − w_q)} + 1)`, which tends to `1` at every `s₂ > 0`. Without a
beginning-of-sequence token this happens to the first key at every `s₁`: its key token reads only
itself and copies the key with `1`. After one such token the key token of the first key copies it
with `1 / (2^{s₁} + 1)`, less than the value token does if and only if `6^{s₁} > 2`.

Over the keys, below `s₂ = 1` every key fades at every `s₁`, since `w_q ≤ 1`. At `s₁ = 1` every key
has `w_q = L / (2L − 1) > 1/2`, and from `s₁ = 1` on the value token outcopies the key token at
every key but the first without a beginning-of-sequence token; so `s₂ ≥ 2` recalls all of these,
and below it the late keys fade. Above `s₁ = 1` and `s₂ = 1` the late keys are recalled. The early
keys are not always the ones lost: after a beginning-of-sequence token, at `s₁ = 1` and
`s₂ = 17/10` the first key is recalled while the second fades.

### SA6. Scope

Proved: SA1 to SA5 as stated.

Not covered:

- SA2 places a binding in one token; SA5 gives two layers for a binding split over two tokens,
  with the context read cyclically and no causal mask. With a causal mask layer 2 recalls a key
  with `s₂ w_q > 1` that its key token copies less than its value token does, and fades at a key
  with `s₂ w_q < 1` or one that its key token copies more; the recall is proved key by key, not
  uniformly over the keys. The boundaries `s₂ a = 1`, `s₂ w_q = 1` and `r_q = w_q` between recall
  and fading are not treated.
- SA3 takes the value of a token to be the readout of its bound value, and its score to be a
  function of the query and of that token. Scores that an earlier layer computes from several
  tokens are not bounded.
- SA4 is for a linear readout of one layer, read by its sign. A regression bound for it and a
  nonlinear readout after it are not proved.
- SA5 bounds the number of keys with a raw gap above `a b`, not the error once there are more:
  how many other keys come within that gap of a queried key, and so whether recall fails, is not
  treated. Whether a trained model keeps a fixed gap as the length grows is an empirical question.

### Lean

`src/Transformer/Grokking/FlatSpectrum/Recall/`, modules `Softmax`, `SoftmaxRecall`,
`SoftmaxScale`, `Fading`, `QueryFree`, `LogScale`, `Threshold`, `RawGap`, `Induction`,
`InductionExact`, `InductionPhases`, `CausalCopy`, `CausalFading`, `CausalRecall`,
`CausalKeyToken` and `CausalPhases`, namespace `Transformer.Grokking.FlatSpectrum.Recall`, proved
without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| SA1 | `softmaxAttention`, `abs_softmaxAttention_sub_le` | The tokens are any finite type; scores and values are real. |
| SA2 | `queryProj`, `keyProj`, `selfProj`, `sum_queryProj_mul_keyProj`, `sum_queryProj_mul_keyProj_add_le`, `softmaxAttention_recall_error`, `softmaxAttention_self_recall_error`, `exists_softmaxAttention_recall`, `exists_softmaxAttention_recall_sign`, `exists_softmaxAttention_not_hasCutWidth` | The keys are `Fin m` and the token `j` carries the key `j`. The existence statements take `B = Σ_v |φ(v)|` and `β = 4 B m / ε + 2`, from `e^{−x} ≤ 1/x`; they do not claim a logarithmic scale. The query's own token is `none : Option (Fin m)`. |
| SA3 | `mul_sum_exp_le_sum_erase`, `mul_abs_sub_le_abs_sub_softmaxAttention`, `sum_sum_update`, `exists_forall_mul_abs_sub_le_softmaxAttention`, `mul_le_sum_softmaxAttention_sq`, `sub_one_mul_le_of_softmaxAttention_recall`, `softmaxAttention_oneHot_recall_error`, `one_sub_softmaxAttention_indicator`, `two_mul_le_one_sub_softmaxAttention_indicator`, `sub_one_mul_two_sub_le_of_softmaxAttention_recall`, `sub_one_lt_exp_of_softmaxAttention_sign`, `exists_lt_softmaxAttention_recall_error` | The average `a` of the other tokens is a quotient of sums over `univ.erase j₀`. Changing `π q` is `Function.update π q v`. The mean `μ` is any number with `Σ_v (φ v − μ) = 0`. The bound for accuracy `ε` is for values `Bool` read as `±1` and at least one key. The worst dictionary is `π k = decide (k = q)`. The statement for growing `m` takes `m + 1` keys, `Fin (m + 1)`, and scores `s m q k b` with one bound `B` for every `m`. |
| SA4 | `card_le_of_normalized_sign`, `softmaxAttention_option`, `card_le_of_queryFreeAttention_sign` | The context is a dictionary with values in `Bool`; its tokens are any finite type, with scores `s π j` and values `v π j r` that are any functions of the dictionary. The query's own token is `none`, with the score `σ q` and the values `u q r`. |
| SA5 | `logScaleAttention`, `logScaleAttention_eq_rpow`, `abs_logScaleAttention_sub_le`, `exists_forall_logScaleAttention_recall`, `exists_lt_logScaleAttention_recall_error`, `oneHot_logScaleAttention_threshold`, `abs_sub_softmaxAttention_le`, `oneHot_erase_div_sum`, `isGreatest_oneHot_logScaleAttention_error`, `oneHot_logScaleAttention_at_threshold`, `convex_inner_lt`, `affineIndependent_of_inner_gap`, `card_le_finrank_add_one_of_inner_gap`, `card_le_of_rawScore_gap`, `rotateHead_one`, `abs_logScaleAttention_comp_sub_le`, `prevToken`, `prevToken_eq_iff`, `keyChannel`, `prevTokenHead`, `inductionHead`, `abs_inductionHead_sub_le`, `exists_forall_inductionHead_recall`, `logScaleAttention_ite`, `card_keyValueTokens`, `prevTokenHead_eq`, `copyLead`, `inductionHead_eq_logScaleAttention`, `sub_inductionHead_eq`, `abs_sub_inductionHead_le`, `isGreatest_inductionHead_error`, `tendsto_two_mul_add_two`, `copyLead_eq`, `tendsto_copyLead_of_tendsto`, `tendsto_copyLead_of_lt`, `tendsto_copyLead_one`, `tendsto_copyLead_of_gt`, `exists_forall_inductionHead_recall_of_tendsto`, `exists_lt_inductionHead_error_of_tendsto`, `exists_forall_inductionHead_one_recall`, `exists_lt_inductionHead_error_of_lt_one`, `causalCopy`, `causalCopy_eq`, `copyWeight`, `prevTokenHead_value`, `causalCopy_value`, `causalCopy_value_eq_prevTokenHead`, `causalCopy_key`, `causalCopy_of_lt`, `causalCopy_of_gt`, `causalCopy_nonneg`, `copyWeight_eq`, `tendsto_copyWeight_of_tendsto`, `tendsto_valuePrefix`, `tendsto_causalCopy_value_of_lt`, `tendsto_causalCopy_value_one`, `tendsto_causalCopy_value_of_gt`, `causalPos`, `card_causalTokens`, `causalInductionHead`, `sum_erase_div_le_one_sub_logScaleAttention`, `le_one_sub_logScaleAttention`, `le_one_sub_causalInductionHead`, `exists_lt_causalInductionHead_error`, `exists_lt_causalInductionHead_error_of_lt_one`, `abs_sub_logScaleAttention_le_sum_erase`, `causalPos_injective`, `causalCopy_le_key`, `causalCopy_le_inv`, `abs_sub_causalInductionHead_le`, `exists_forall_causalInductionHead_recall`, `rpow_div_le_one_sub_logScaleAttention`, `le_one_sub_causalInductionHead_of_key`, `exists_lt_causalInductionHead_error_of_key`, `exists_lt_causalInductionHead_error_first`, `causalCopy_key_lt_value_iff`, `causalCopy_key_lt_copyWeight`, `copyWeight_le_one`, `exists_lt_causalInductionHead_error_of_scale_lt_one`, `exists_forall_causalInductionHead_recall_one`, `exists_lt_causalInductionHead_error_one`, `exists_forall_causalInductionHead_recall_of_gt` | `n` is the number of tokens, `Fintype.card`, and `s ≥ 0`. The recall statement takes `m + 1` keys, `Fin (m + 1)`, raw scores `z m q k` of the length, the query and the key, and values `φ (π k)` with `|φ| ≤ C`. The fading statement takes raw scores `z m q k b` that may depend on the value `b` of the key, bounded by one `B`, and values `±1`. The one-hot statement takes `B ≥ 0`. The worst one-hot error is `IsGreatest` of the range of the errors over `π`, at every real `s` and `B`; the statement at the threshold takes `2 * s * B = 1` and no sign condition. The gap statements take queries and keys in any real inner product space, finite-dimensional for the count. The diff-gpt statement takes the rotary phases of every query and every key as given, with `d > 0` and `eps > 0`; its example has the head width `1`, where the rotary map is the identity. The two-layer statement takes any finite tokens, a partner of every token, and the leads `g₁` and `g₂`; the induction head takes the tokens `Fin (m + 1) × Bool`, a key token and a value token for every key, read cyclically, with one-hot key channels. Its closed form holds at every real `s₁` and `s₂`, with `n = 2m + 2` written out. The phase statements take the limit `a` of the lead as the hypothesis `Tendsto (copyLead s₁) atTop (𝓝 a)` and need no sign of `s₁` or `s₂`; the fading statements take values `±1`. The causal copy takes positions and keys in `ℕ`; the copy weight `copyWeight s x` takes a real number `x` of tokens. The causal head reads the `n` tokens of the context at the scale `s₂ log n`, as the cyclic head does, and not the query token itself; its fading statements take the indicator dictionary of values `±1`. The causal recall statement takes a fixed key `q : ℕ` and the queries `k : Fin (m + 1)` with `k = q`. |

## A positional bias at the scale `s log n` (added)

ALiBi adds the bias `−m |j − i|` to the score of the key `j` at the query `i`, with a slope
`m > 0` fixed before training; as a prior over the positions it is a Laplace distribution of
scale `1/m` (Bianchessi, Aguirre, Barros, Kupssinskü, *Bayesian Attention Mechanism: A
Probabilistic Framework for Positional Encoding and Context Length Extrapolation*,
arXiv:2505.22842v4, §2.2, Lemma 2; `Transformer.BAM.alibi`). SSMax multiplies the scores by
`s log n` over `n` tokens (Nakanishi, *Scalable-Softmax Is Superior for Attention*,
arXiv:2501.19399v1, §2). This section asks which slope recall needs, and what the scale
`s log n` changes. A bias `b` of the positions adds `b_j` to the score of the token `j`; at the
scale `s log n` the weights are `n^{s z_j} e^{b_j}`, and ALiBi has `b_r = −m r` at the distance
`r` from the query.

### PB1. No bias helps the worst position

Let `j₀` be a token of least bias, `b_{j₀} ≤ b_j` for every `j`, and let `v` be `1` at `j₀` and
`−1` elsewhere. For all scores `t`, the scores `t + b` answer `v` at least as badly as `t`:

```
1 − att(t, v) ≤ 1 − att(t + b, v).
```

Proof. `1 − att(t, v) = 2X / (A + X)` with `A = e^{t_{j₀}}` and `X = Σ_{j ≠ j₀} e^{t_j}` (SA3).
With the bias, `A' = e^{b_{j₀}} A` and `X' = Σ_{j ≠ j₀} e^{t_j + b_j} ≥ e^{b_{j₀}} X`, so
`X A' ≤ X' A`, which is `2X / (A + X) ≤ 2X' / (A' + X')`. ∎

Every bias has a token of least bias; for ALiBi it is the farthest token. With one-hot raw
scores, `B` at the queried key and `−B` at the other `m` keys, the worst error over the
dictionaries of values `±1` without a bias is `2m / (n^{2sB} + m)` at every position (SA5). A
key at a position of least bias is answered with at least that error. So over the positions of
the key and the dictionaries no bias lowers the worst error, and the bias `0` attains it: for a
key that may sit at any position the best slope is `0`, at every scale, and at the scale
`s log n` the threshold `2 s B = 1` stays.

### PB2. A fixed slope leaves a window of radius `(2sB/m) log n`

Two bounds hold for any bias `b` at the scale `s log n`, `s ≥ 0`. If the raw score of `j₀` leads
every other by `g` and `|v_j| ≤ C`, then

```
|v_{j₀} − att| ≤ 2C n^{−sg} Σ_{j ≠ j₀} e^{b_j − b_{j₀}},
```

since the weight of `j` against that of `j₀` is `n^{s(z_j − z_{j₀})} e^{b_j − b_{j₀}}` (SA5). If
the raw scores are bounded by `B`, the value `1` at `j₀` and `−1` elsewhere miss by at least
`2 / (1 + n^{2sB} e^{b_{j₀} − b_{j₁}})` for every other token `j₁`, since `X ≥ e^{t_{j₁}}` in
`2X / (A + X)` (PB1).

Index the `n` tokens that the query reads by their distance `r = 0, …, n − 1` from it, with the
ALiBi bias `−m r`, `m > 0`. The prior weights sum to `Σ_r e^{−mr} ≤ 1 / (1 − e^{−m})` at every
length. So a key at the distance `R` with the lead `g` is answered within
`2C n^{−sg} e^{mR} / (1 − e^{−m})`, and with raw scores bounded by `B` a key at a distance
`R ≥ 1` misses by at least `2 / (1 + n^{2sB} e^{−mR})`, against the query's own token at `r = 0`.
At `R ≤ D + ρ log n` the first bound is at most `2C e^{mD} n^{−(sg − mρ)} / (1 − e^{−m})`, and
at `R ≥ ρ log n` the second is at least `2 / (1 + n^{−(mρ − 2sB)})`. So

- keys within `D + ρ log n` of the query with `m ρ < s g` are recalled from some length on;
- keys beyond `ρ log n` with `m ρ > 2 s B` miss by an error that tends to `2`, the opposite
  value.

For one-hot raw scores `±B` the lead is `g = 2B` and the two radii meet: the slope `m` leaves a
window of radius `(2sB/m) log n`, a vanishing fraction of the context, and every key at a
distance proportional to `n` is eventually lost. Inside the window the threshold `2 s B = 1` of
SA5 is gone: at `ρ = 0` a key at a bounded distance is recalled at every lead with `s g > 0`.
Without a bias the other tokens weigh `n − 1` against `n^{sg}`; with the slope they weigh at most
`e^{mR} / (1 − e^{−m})`, a constant. At a fixed scale the window is bounded instead
(`Transformer.BAM.alibi_local`). The scale `s log n` widens it only logarithmically.

### PB3. The nearer tokens bound every decreasing bias

Let the raw scores be bounded by `B`, and let `S` be a set of `k` tokens other than `j₀`, each
of bias at least `b_{j₀}`. The weight of `j₀` is at most `n^{sB} e^{b_{j₀}}` and the weights of
`S` sum to at least `k n^{−sB} e^{b_{j₀}}`, so in `1 − att = 2X / (A + X)` (PB1)

```
1 − att ≥ 2k / (k + n^{2sB}).
```

A bias that does not increase with the distance, as ALiBi's `−m r` at every `m ≥ 0` or
`−α log(1 + r)` at every `α ≥ 0`, gives the key at the distance `R` its `R` nearer tokens as
such a set. The key misses by at least `2R / (R + n^{2sB})`, the worst error of SA5 with `R`
keys in place of the `n − 1` other tokens. At `R ≥ n^ρ` this is at least
`2 / (1 + n^{2sB − ρ})`, which tends to `2` when `2 s B < ρ`, uniformly in the bias, which may
change with the length. So no slope, fixed or chosen with the length, recalls a key at the
distance `n^ρ` below the threshold `2 s B = ρ`: a decreasing bias can at best lower the threshold
`2 s B = 1` of SA5 to the exponent `ρ` of the distance. It cannot lower it for the keys at a
distance proportional to `n`, which lie beyond `n^ρ` for every `ρ < 1` from some length on.

### PB4. The slope fitted to a span reaches the threshold

ALiBi's bias `−m r` is the logarithm of the geometric prior `(1 − e^{−m}) e^{−mr}` of the
distance (BAM §2.2, Lemma 2), whose mean is `D` at `m = log(1 + 1/D)`: the slope fitted to the
span `D`. At it `e^m = 1 + 1/D` and `1 − e^{−m} = 1/(D + 1)`, so by PB2 a key at a distance
`R ≤ D` with the lead `g` is answered within

```
2C n^{−sg} e^{mR} / (1 − e^{−m}) = 2C n^{−sg} (1 + 1/D)^R (D + 1) ≤ 2C e (D + 1) n^{−sg}.
```

For a span `D ≤ n^ρ` this tends to `0` when `s g > ρ`, uniformly in the span and in the key
within it. For one-hot raw scores `±B` the lead is `g = 2B`, so with PB3 the threshold
`2 s B = ρ` is sharp, and the fitted slope reaches it for the whole span.

A slope fitted to a span `D` below the distance `R` of the key is too steep. Since
`log(1 + 1/D) ≥ 1/(D + 1)`, the key misses by at least `2 / (1 + n^{2sB} e^{−R/(D+1)})`, and the
keys beyond `κ (D + 1) log n` are lost for every `κ > 2 s B`: the loss grows exponentially in
`R / D`. A fixed slope is the case of a fixed `D`, the window of PB2. A slope fitted to a span
`D` above `R` is too gentle: PB4 bounds the rivals of the key by `e (D + 1)`, against the `R` of
PB3.

### PB5. The bias `−log(1 + r)` reaches the threshold at every distance

A bias `b` weighs the distances against a key at the distance `R` by `W_R = ∑_r e^{b_r − b_R}`,
the inverse of the prior `e^{b_R} / ∑_r e^{b_r}` of `R`. By PB2 a key with the lead `g` is
answered within `2C n^{−sg} (W_R − 1)`, and by PB3 a decreasing bias has `W_R ≥ R + 1`. The
inverses `1 / W_R` sum to `1`, so the `W_R` cannot all be below `(R + 1) H_n`, where
`H_n = ∑_{r<n} 1/(r + 1)`: if they were, `1 = ∑_R 1/W_R > ∑_R 1/((R + 1) H_n) = 1`. Every bias
has a distance `R` with `W_R ≥ (R + 1) H_n`, and the bias `−log(1 + r)`, the prior `1/(r + 1)`
normalized by `H_n`, has `W_R = (R + 1) H_n` at every distance. It is the bias of least
`max_R W_R / (R + 1)`, the weight against the `R + 1` that no decreasing bias avoids, and serving
all distances at once costs the factor `H_n ≤ 1 + log n`.

With it a key at the distance `R` with the lead `g` is answered within
`2C (R + 1) (1 + log n) n^{−sg}`. For `R + 1 ≤ n^ρ` this is at most
`2C (1 + log n) n^{−(sg − ρ)}`, which tends to `0` when `s g > ρ`, and by PB3 the keys at
`R ≥ n^ρ` are lost when `2 s B < ρ`. So for one-hot raw scores the one bias `−log(1 + r)` has the
threshold `2 s B = ρ` at every distance `n^ρ` at once, the least that any decreasing bias has
there. A fitted ALiBi slope has it only on its span (PB4), and a fixed slope only at bounded
distances (PB2). Derived from the threshold, the positional bias that serves every distance is
logarithmic in the distance, not linear.

### Scope

Proved: PB1 to PB5. Not covered: a lower bound of order `D` on the rivals of a key well inside
the span `D` of a fitted slope, the coefficients `α ≠ 1` of the bias `−α log(1 + r)`, and whether
training finds such a bias.

### Lean

`src/Transformer/Grokking/FlatSpectrum/Recall/`, modules `PositionBias`, `ALiBiWindow`,
`NearerTokens`, `FittedSlope` and `LogBias`, namespace `Transformer.Grokking.FlatSpectrum.Recall`,
proved without `sorry`.

| Item | Lean | Differences |
| --- | --- | --- |
| PB1 | `biasedLogScaleAttention`, `biasedLogScaleAttention_zero`, `one_sub_softmaxAttention_indicator_le_add`, `exists_le_oneHot_biasedLogScaleAttention_error`, `oneHot_logScaleAttention_error_le_biased` | The bias is any real function of the tokens, not only of the distance to the query. The first statement takes any finite type of tokens, the others the `m + 1` keys `Fin (m + 1)` with one-hot raw scores. "No bias lowers the worst error" is: for every bias, every key and every dictionary, the error without the bias is at most the error with it at some key and some dictionary. |
| PB2 | `abs_sub_biasedLogScaleAttention_le`, `le_one_sub_biasedLogScaleAttention`, `sum_exp_neg_mul_le`, `abs_sub_alibi_le`, `le_one_sub_alibi`, `exists_forall_alibi_recall`, `exists_lt_alibi_error` | The two general bounds take any finite type of tokens and any real bias. For ALiBi the tokens are `Fin n`, the token `r` at the distance `r` from the query, with the bias `fun r => -(m * r)`; the slope of `le_one_sub_alibi` may be any real. The recall statement takes the keys within `D + ρ log n` for any real `D`, so that `ρ = 0` gives the keys at a bounded distance; the fading statement takes the indicator dictionary of values `±1`. "Eventually lost at a distance proportional to `n`" is not stated apart from the fading statement. |
| PB3 | `le_one_sub_biasedLogScaleAttention_of_subset`, `le_one_sub_biasedLogScaleAttention_of_nearer`, `exists_lt_error_of_nearer`, `exists_lt_alibi_error_of_rpow_le` | The bias need only be at the distance `R` at most its value at every nearer distance, not decreasing everywhere. The fading statements take the indicator dictionary of values `±1` and the keys at the distances `R ≥ n^ρ`; at `ρ ≥ 1` they are vacuous, so the keys at a distance proportional to `n` are reached through a `ρ` between `2 s B` and `1`. |
| PB4 | `abs_sub_fittedSlope_le`, `exists_forall_fittedSlope_recall`, `le_one_sub_fittedSlope`, `exists_lt_fittedSlope_error` | The span is a natural number `D ≥ 1` and the slope `Real.log (1 + 1 / D)`; the recall statement takes the spans `D ≤ n^ρ` and the keys at the distances `R ≤ D`, the fading statement every span `D ≥ 1`. The sharp threshold for one-hot raw scores is this item with PB3, not one statement. |
| PB5 | `exists_mul_harmonic_le_sum_exp`, `sum_exp_logBias`, `sum_one_div_succ_le`, `abs_sub_logBias_le`, `exists_forall_logBias_recall`, `exists_lt_logBias_error` | `H_n` is the sum `∑ r : Fin n, 1 / ((r : ℝ) + 1)`, bounded through Mathlib's `harmonic_le_one_add_log`; the bias is `fun r : Fin n => -Real.log ((r : ℝ) + 1)`. The recall statement takes the keys with `R + 1 ≤ n^ρ`, the fading statement those with `R ≥ n^ρ`. The least `max_R W_R / (R + 1)` is the pair of the first two statements, not one statement. |

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
*Depth, width and precision* in
[`actual_network/sequence_models.md`](actual_network/sequence_models.md), "A flat spectrum
alone does not supply either conclusion", for the paper's own languages.

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
λ > 0 and a loss that reads only Ω. By CW8 in [`diffgpt/steps.md`](diffgpt/steps.md), a
minimum of M from Ω is zero on every row without a cell of Ω, and on every such column:
removing the row keeps the seen values and lowers the nuclear norm. So

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
| XE12 | `IsDecayedMin.row_eq_zero`, `IsDecayedMin.col_eq_zero`, `IsDecayedMin.apply_eq_zero_of_codeProbe`, `IsDecayedMin.pairing_codeProbe_eq_zero`, `IsDecayedMin.ceLoss_heldOut_eq`, `exists_tendsto_decayRun_ceLoss_eq_zero` | The row has `∀ j, (i, j) ∉ Ω` as in CW8 of `diffgpt/steps.md`, whose `row_eq_zero_of_isMinOn` and `col_eq_zero_of_isMinOn` are applied to the minimum of XE7. For the answer code the seen cells are `Set.range cell`, and "no sample sees the row `i`" is `∀ s, (cell s).1 ≠ i`. The held-out samples are a finite type `S'` with cells `cell' : S' → m × n` and any labels `y'`. "The softmax is uniform" is the statement that every logit is zero. |
| XE14 | `hasDerivAt_comp_add_smul_of_analytic`, `gradient_flatLoss_of_analytic`, `factors_iterate_gradientStep_of_analytic`, `exists_pos_tendsto_decayRun_of_analytic`; the logits `±(Re x)²`: `sqWeight`, `sqWeight_nonneg`, `sqWeight_le_half`, `sqLogitLoss`, `sqLogitGrad`, `sqLogitLoss_eq_crossEntropy`, `sqLogitLoss_nonneg`, `sqLogitLoss_zero`, `sqLogitGrad_zero`, `pairing_sqLogitGrad`, `hasDerivAt_sqLogitLoss`, `continuous_sqLogitGrad`, `analyticAt_sqLogitLoss`, `exists_pos_tendsto_decayRun_sqLogit`, `sqLogitLoss_lt_zero`, `not_first_order_sqLogit`, `frobeniusSq_sqLogitGrad_le`, `neg_le_pairing_sqLogitGrad`; the ball: `tendsto_zero_of_tendsto_frobeniusSq`, `re_sq_le_frobeniusSq`, `frobeniusSq_sqLogitGrad_mul_le`, `frobeniusSq_decayStep_sqLogit_le`, `frobeniusSq_decayRun_sqLogit_le`, `tendsto_decayRun_sqLogit_zero`, `one_sub_smul_lift_posDef_sqLogit`, `not_isDecayedMin_sqLogit_zero`, `exists_tendsto_decayRun_sqLogit_not_isDecayedMin`; in `Training`: `hasDerivAt_decayedLoss_of_hasDerivAt`, `hasGradientAt_flatLoss_of_hasDerivAt` | Modules `Scope/SmoothLoss`, `Scope/SquaredLogit`, `Scope/SquaredLogitBall` and `Scope/SquaredLogitRun`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. "`G` is the gradient of `L`" is the hypothesis `∀ X H, HasDerivAt (fun t : ℝ => L (X + t • H)) (pairing (G X) H) 0`; real-analytic and bounded below are the hypotheses of WD9, and the conclusion of statement 1 is that of `exists_pos_tendsto_decayRun`. The two lemmas in `Training` are the parts of `hasDerivAt_decayedLoss` and `hasGradientAt_flatLoss` that do not use convexity. The loss is `sqLogitLoss X = log (1 + exp (-(2 * (X 0 0).re ^ 2)))` on `Matrix (Fin 1) (Fin 1) ℂ`, equal to `crossEntropy (fun c : Bool => (signCode c).re * (X 0 0).re ^ 2) true`, and `sqLogitGrad X = (-(4 * r * sqWeight r)) • 1`. "Not convex" is `¬ ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y`. The factors are `Matrix (Fin 1) k ℂ` for a finite `k`; "converges to the zero factorization" is `Tendsto … (𝓝 (0, 0))`; "not a decayed minimum" is `¬ IsDecayedMin sqLogitLoss lam 0 0` for `lam ≤ 1/4`, at every `k`. That the ball has interior points and the comparison with WD10 are not formalized. The start of the examples is `!![1/8, 0]`, `!![1/8, 1/8]` with `lam = 1/4` and `η = 1`. |
| XE15 | `exists_unit_conj_mul`, `detCertificate`, `frobeniusSq_fin_two`, `isFlat_detCertificate`, `pairing_detCertificate`, `frobeniusSq_add_two_mul_norm_det_le`, `norm_mul_sub_le_norm_det`, `sq_sub_add_sq_add_le`, `eq_ofReal_of_norm_le_of_le_re`; the three cells: `threeCells`, `threeLabels`, `signFill`, `gap_threeCells`, `margin_threeCells_le`, `le_margin_threeCells`, `margin_threeCells_le_nuclearNorm`, `nuclearNorm_signFill_le`, `margin_signFill`, `maxMargin_threeCells`, `eq_signFill_of_margin`; the limit: `cubicKernel_two`, `tendsto_inv_smul_threeCells`, `eventually_re_unseen_neg`, `norm_sub_signCode_lt`, `eventually_unseen_wrong`; every small weight decay and the run: `exists_pos_forall_re_unseen_neg`, `exists_pos_forall_unseen_wrong`, `exists_pos_forall_tendsto_decayRun_unseen`, `exists_tendsto_decayRun_unseen_wrong` | Modules `Scope/TwoByTwo`, `Scope/UnseenCell`, `Scope/UnseenCellLimit` and `Scope/UnseenCellRun`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The samples are `Fin 3` with the cells `threeCells = ![(0, 0), (0, 1), (1, 0)]` and the labels `threeLabels s = signLabel (threeCells s)`, those that `signKernel` shows; `cubicKernel_two` is `cubicKernel 2 = signKernel`. "Decoded to the class of `−1`" is `‖X 1 1 − signCode false‖ < ‖X 1 1 − signCode (signLabel (1, 1))‖`, where `signLabel (1, 1) = true`. The statements of 4 take a sequence of trained products `IsNuclearMin`; the threshold `λ₀` of 5 is obtained from them by contradiction and is not explicit. The bounds of XE3 in 6 are written with their numbers: `12 * log 2 ≤ λ ρ`, `2 η (λ² + 36) ≤ λ` and `η (λ + 6 + 6 ρ) ≤ 1`; the width is any finite type. The run that exists has the step `min (λ / (2 (λ² + 36))) (1 / (λ + 6 + 6 ρ))` for `λ = λ₀ / 2` and `ρ = 4 + 12 log 2 / λ`. The bound of 2 is proved as an inequality; the equality is not needed. |
| XE16 | `isNuclearMin_signCode_iff`, `nuclearNorm_eq_signScale_mul_of_isNuclearMin`; the kernel of signs: `signRot`, `im_eq_of_det_bounds`, `exists_eq_add_smul_signRot`, `signCotwin`, `nuclearNorm_signCotwin_le`, `nuclearNorm_add_smul_signRot_le`, `isNuclearMin_signKernel_iff`; the run: `exists_tendsto_decayRun_signKernel` | Modules `Scope/SignSet`, `Scope/SignSegmentCore` and `Scope/SignSegment`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. `t(λ)` is `signScale c₀ lam`, and "`Re X = t(λ) K`" is `∀ i j, (X i j).re = signScale c₀ lam * (K i j).re`; the hypotheses are those of XE11 (`Function.Bijective cell`, `K` shows the labels, `IsFlat K c₀`). For the kernel of signs the cells are `id` on `Fin 2 × Fin 2` with the labels `signLabel`, `J` is `signRot`, `K − i J` is `signTwin` and `K + i J` is `signCotwin`; the number `s` of the segment is `a` in Lean, and the point is `signScale 2 lam • signKernel + ((a : ℂ) * Complex.I) • signRot`. `im_eq_of_det_bounds` is the real arithmetic of "⟹", in the imaginary parts and `N = ‖det X‖`. The bounds of XE3 are written with their numbers: `16 * log 2 ≤ λ ρ`, `2 η (λ² + 64) ≤ λ` and `η (λ + 8 + 8 ρ) ≤ 1`; the width is any finite type. |
| XE17 | `not_tendsto_decayRun_ceLoss_zero`, `exists_lt_frobeniusSq_decayRun_ceLoss_zero`, `decayRun_add`, `surjective_stack_decayRun`, `frequently_lt_frobeniusSq_decayRun_ceLoss_zero`; every real λ (WD8): `exists_tendsto_decayRun_of_forall_le`, `exists_tendsto_decayRun_isDecayedMin_of_forall_le` | Module `Scope/NoDecayRun`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`; the statements for every λ are in `BoundedRun` and `Convergent`, namespace `Transformer.Grokking.FlatSpectrum.Training`. "No limit" is `¬ Tendsto (decayRun (ceGrad A y) η 0 U₀ V₀) atTop (𝓝 (U, V))` for every pair `(U, V)`. "Some product has a positive margin" is `0 < margin A y X₀`, with `[Nonempty S] [Nontrivial C]`; `N` is `Fintype.card S`, and the conditions are `η * (2 * a * card S) < 1` and `η * (2 * a * card S + 2 * (card S * a ^ 2) * ρ) < 1`. Statement 3 is `∃ t, ρ < …` and `∃ᶠ t in atTop, ρ < …`; `ρ*(η)` and the `lim sup` are not stated. The two citations are not formalized. |
| XE18 | `abs_nuclearNorm_sub_le`, `continuous_nuclearNorm`, `continuous_margin`, `continuousAt_margin_div_nuclearNorm`, `tendsto_inv_smul_of_tendsto_margin_div`; one phase: `phaseStep`, `phaseStep_pos`, `exists_tendsto_decayRun_phaseStep`, `exists_seq_of_forall_exists`; the run: `phaseRun`, `phaseProduct`, `exists_tendsto_margin_div_phaseProduct`; the direction: `exists_tendsto_inv_smul_phaseProduct`, `exists_tendsto_inv_smul_phaseProduct_codeProbe`, `exists_tendsto_inv_smul_phaseProduct_cubicKernel`, `exists_eventually_phaseProduct_unseen_wrong` | Modules `Scope/MarginContinuity`, `Scope/PhaseStep`, `Scope/PhasedRun` and `Scope/PhasedDirection`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. `phaseStep S C a lam W` is `η(λ, W)` with `N` and `\|C\|` as `Fintype.card S` and `Fintype.card C`. `phaseRun A y a lam T U₀ V₀ j` is the pair `W_j`, defined by recursion on `j` with the step `phaseStep S C a (lam j) W_j`, and `phaseProduct A y a lam T U₀ V₀ j` is `X_j`. The weight decays are any `lam : ℕ → ℝ` with `∀ j, 0 < lam j` and `Tendsto lam atTop (𝓝 0)`, and "there are lengths" is `∃ T : ℕ → ℕ`. Statement 2 is the conjunction of `Tendsto (fun j => nuclearNorm (X_j)) atTop atTop` and the limit of `margin A y (X_j) / nuclearNorm (X_j)`, Lean's quotient, which is `0` where the product is `0`. The Lipschitz bound is `\|‖X‖_S1 − ‖Y‖_S1\| ≤ √(card m * frobeniusSq (X − Y))`. `tendsto_inv_smul_of_tendsto_margin_div` is the statement about every sequence of matrices; its hypothesis on `Z₀` is that of XE10. For the code probes `a = 1`. "Decoded to the class of `−1`" is written as in XE15, at the cell `(1, 1)` of `phaseProduct`. `exists_seq_of_forall_exists` is the dependent choice of the phases. The bound `η_j ≤ λ_j / (8 s b)`, the numbers of the example's first phase and "the run has no limit" are not stated; the examples are the sample of XE3, `cubicKernel 3` with all nine cells seen, and the three seen cells of the kernel of signs, along `λ_j = 1 / (j + 1)` from the start `fromCols 1 0`, `fromCols 0 1`. |
| XE19 | `signWeight`, `signWeight_pos`, `signWeight_le_two`, `signWeight_anti`, `signWeight_zero`, `mul_signWeight_le_two`; the system: `IsSignSystem`, `IsSignSystem.snd_nonneg`, `IsSignSystem.fst_pos`, `IsSignSystem.fst_mono`, `IsSignSystem.tendsto_fst_atTop`, `IsSignSystem.mul_le_mul_of_le`, `IsSignSystem.exists_lt`, `IsSignSystem.tendsto_sub_atTop`, `IsSignSystem.exists_forall_snd_le`; the run: `ceGrad_signProbe`, `decayStep_signProbe`, `pairing_one_mul_conjTranspose`, `frobeniusSq_add_frobeniusSq`, `signProduct`, `decayRun_signProbe_add`, `decayRun_signProbe_sub`, `isSignSystem_decayRun_signProbe`, `margin_signProbe`, `margin_signProduct`, `tendsto_modes_decayRun_signProbe`, `tendsto_frobeniusSq_decayRun_signProbe_atTop`; the limits: `tendsto_margin_signProduct_atTop`, `tendsto_ceLoss_signProduct`, `tendsto_margin_div_frobeniusSq_signProduct`, `tendsto_margin_div_signProduct`, `tendsto_inv_smul_signProduct`, `log_two_le_ceLoss_signProduct`, `tendsto_ceLoss_signProduct_iff`; the rate: `IsSignSystem.exp_add_le_exp_succ`, `IsSignSystem.exists_exp_ge`, `exists_ceLoss_signProduct_le`, `exists_log_le_frobeniusSq_decayRun_signProbe` | Modules `Scope/SignSystem`, `Scope/SignGap`, `Scope/SignRun`, `Scope/SignRunLimit` and `Scope/SignRate`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The sample is `signProbe` with the label `fun _ => true`; the factors are `U₀ V₀ : Matrix (Fin 1) k ℂ` for any finite `k`, the run is `decayRun (ceGrad signProbe fun _ => true) η 0 U₀ V₀`, and `signProduct η U₀ V₀ t` is `x_t` as a `1 × 1` matrix. `Re x` is `pairing 1 X`. `IsSignSystem η a b` states the two recursions for sequences `a b : ℕ → ℝ`; the run satisfies it with `a t = frobeniusSq (U_t + V_t)` and `b t = frobeniusSq (U_t − V_t)`. Statements 1 to 3 have the hypotheses `0 < η` and `U₀ + V₀ ≠ 0`; statement 4 is `log_two_le_ceLoss_signProduct` for every real `η` and the equivalence `tendsto_ceLoss_signProduct_iff` for `0 < η`. The quotient `margin / nuclearNorm` is Lean's, which is `0` where the product is `0`. The geometric growth of the ratio while `a ≤ b` is a private lemma. Statement 5 for the system is `∃ T, b T ≤ a T ∧ ∀ t, T ≤ t → exp ((a T - b T) / 2) + η * a T * (t - T) ≤ exp ((a t - b t) / 2)`; for the run it is `∃ T, 0 < a_T ∧ ∀ t, T ≤ t → ceLoss _ _ (signProduct η U₀ V₀ t) ≤ 1 / (1 + η * a_T * (t - T))` and the same with `Real.log (1 + η * a_T * (t - T)) ≤ frobeniusSq U_t + frobeniusSq V_t`, where `a_T` is `frobeniusSq (U_T + V_T)`. The remark on the eigenvector `(1, 1)` is not stated. The examples are the start `fromCols 1 0`, `fromCols 0 1` with `η = 1/10` and the starts `(1, −1)` and `(1, 1)` of width one; the witness of the system is `signWitness`, the first mode from `a_0 = 1`, `b = 0` with the step `1`. |
| XE20 | `classSign`, `classSign_mul_self`, `ceGrad_term_signCode`, `ceGrad_signCode`, `ceGrad_signCode_mul_apply`, `conjTranspose_ceGrad_signCode_mul_apply`, `diagCell`, `injective_diagCell_fst`, `injective_diagCell_snd`; the rows: `rowOf`, `pairing_one_rowOf`, `rowOf_decayStep_signCode`, `rowOf_decayRun_signCode`, `frobeniusSq_rowOf_le`, `rowOf_add_ne_zero_iff`, `pairing_neg_left`, `crossEntropy_signCode_mul`, `ceLoss_decayRun_signCode`; the run: `tendsto_frobeniusSq_decayRun_signCode_atTop`, `log_two_le_ceLoss_decayRun_signCode`, `tendsto_ceLoss_decayRun_signCode_iff`, `tendsto_margin_decayRun_signCode_atTop`, `diagCell_one_add_zero_ne`, `diagCell_one_add_neg_one_eq`, `exists_ceLoss_decayRun_signCode_le` | Modules `Scope/SeparateCells`, `Scope/SeparateCellsRows`, `Scope/SeparateCellsRun` and `Scope/SeparateCellsRate`, namespace `Transformer.Grokking.FlatSpectrum.Training.Scope`. The samples are `codeProbe signCode cell` with `cell : S → m × n` and the labels `y : S → Bool`; `y_s = ±1` is `classSign (y s)`. "No two samples in one row" is `Function.Injective fun s => (cell s).1`, and the same with `.2` for the columns. `E_ij` is `Matrix.single i j 1`; the gradient is `ceGrad_signCode`, and the two statements on its rows are stated for the entries of `G X * V` and of `(G X)ᴴ * U`. The row `i` of a factor is `rowOf i U`, a matrix with one row: `u_s` is `rowOf (cell s).1 U` and `v_s` is `classSign (y s) • rowOf (cell s).2 V`. Statement 1 is `decayStep` and `decayRun` of the sign sample of XE19 at these rows, equal to the rows of the step and of the run; the loss of a sample is `crossEntropy_signCode_mul`, an equality with `ceLoss signProbe`, and the sum is `ceLoss_decayRun_signCode`, with `signProduct` of XE19. "`u_s + v_s ≠ 0` at the start" is `∃ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b ≠ 0`, and its negation is `∀ b, _ = 0`. The run is `decayRun (ceGrad (codeProbe signCode cell) y) η 0 U₀ V₀`; the loss and the margin are those of the product of its two factors. The statement on the margin assumes `Nonempty S`. Statement 4 is `∃ T c, 0 < c ∧ ∀ t, T ≤ t → ceLoss _ _ _ ≤ Fintype.card S / (1 + c * (t - T))`, under `Nonempty S`; the step `T` and the constant `c` of its proof are not in the statement. The remark on the diagonal cells and the text of "Not covered" on the direction are not stated. `diagCell s` is `(s, s)` for `s : Fin 2`. The examples are those of the item. |
