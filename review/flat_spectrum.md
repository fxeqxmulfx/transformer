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
| Lemma 1 | `nuclearNorm_isLeast`, `nuclearNorm_le_of_eq_mul` | The minimum runs over every inner dimension. |
| Lemma 1, "exact for a pure factorization" | `IsOptimalFit`, `isOptimalFit_iff`, `setOf_isOptimalFit_eq`, `forall_isMinOn_iff_iff_isOptimalFit`, `not_exists_isOptimalFit_of_singularValues_eq`, `isOptimalFit_vecMulVec_iff_connected` | Added. A fit of a set `S` of matrices is a factorization `U Vᴴ ∈ S` of any width `r`; it is optimal when its weight decay `½(‖U‖² + ‖V‖²)` is the least among the fits of every width. A fit is optimal exactly when its product minimizes `‖·‖_S1` on `S` and its weight decay equals that norm; the products of the optimal fits are exactly the minimizers of `‖·‖_S1` on `S`. So for `K ∈ S`, some fit is optimal and every optimal fit multiplies to `K` exactly when M recovers `K`, in D.2's form. Read through this, Theorem A for flat kernels and Theorem E hold for weight decay: no optimal fit multiplies to a flat `K` without zero entries on a proper `Ω`, and for `u vᵀ` with `|u_i v_j| = 1` some fit is optimal and all multiply to `u vᵀ` exactly when `Ω` is connected and contains a permutation. That training reaches an optimal fit is the modeling step, not proved. |
| Prop. 2, `a + b` | `addKernel_eq_vecMulVec`, `rank_addKernel`, `nuclearNorm_addKernel`, `effectiveRank_addKernel` | |
| Prop. 2, `a³ + a b` | `cubicKernel_eq_diagonal_mul`, `stdAddChar_mul_conjTranspose`, `cubicKernel_mul_conjTranspose`, `singularValues_cubicKernel`, `nuclearNorm_cubicKernel`, `effectiveRank_cubicKernel` | |
| Prop. 2, extremes | `card_le_nuclearNorm_le_of_unimodular`, `nuclearNorm_unimodular_extremes` | The bounds hold for every unimodular `N × N` matrix. |
| Prop. 2, relabeling | `singularValues_taskKernel_relabel`, `nuclearNorm_submatrix` | |
| Theorem A | `isMinOn_nuclearNorm_iff` | Invertible means `K.det ≠ 0`. |
| Theorem A, `x³ + x y` | `polarFactor_cubicKernel`, `norm_polarFactor_cubicKernel`, `not_isMinOn_cubicKernel` | |
| Theorem A, every flat kernel | `polarFactor_of_singularValues_eq`, `det_ne_zero_of_singularValues_eq`, `not_isMinOn_of_singularValues_eq`, `not_forall_isMinOn_iff_of_singularValues_eq` | Added: the source states it for `x³ + x y`. For every square `K` whose singular values are all equal and whose entries are all nonzero, and every proper `Ω`, `K` does not minimize `‖·‖_S1` among the matrices equal to it on `Ω`: some such matrix has a smaller nuclear norm. So M recovers `K` from no proper `Ω`, in D.2's form; `∑ σ_i² = ‖K‖_S2² > 0` makes the common value nonzero and `K` invertible. The example is `x³ + x y` at `p = 3` with `Ω = ∅`. |
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
- Lemma 1's modeling step: that training with weight decay reaches an optimal fit; at an
  optimal fit it is exactly minimum-nuclear-norm interpolation (row *Lemma 1, "exact for a pure
  factorization"*);
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

Not formalized: the numerical maxima above; that training behaves like the factorization model
(Lemma 1's modeling step); the note's readout-free conjecture about the least `r_eff`. Theorem C
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
- that training behaves like model M (Lemma 1's modeling step).

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
