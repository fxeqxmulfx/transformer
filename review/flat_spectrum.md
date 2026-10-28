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
| Prop. 2, `a + b` | `addKernel_eq_vecMulVec`, `rank_addKernel`, `nuclearNorm_addKernel`, `effectiveRank_addKernel` | |
| Prop. 2, `a³ + a b` | `cubicKernel_eq_diagonal_mul`, `stdAddChar_mul_conjTranspose`, `cubicKernel_mul_conjTranspose`, `singularValues_cubicKernel`, `nuclearNorm_cubicKernel`, `effectiveRank_cubicKernel` | |
| Prop. 2, extremes | `card_le_nuclearNorm_le_of_unimodular`, `nuclearNorm_unimodular_extremes` | The bounds hold for every unimodular `N × N` matrix. |
| Prop. 2, relabeling | `singularValues_taskKernel_relabel`, `nuclearNorm_submatrix` | |
| Theorem A | `isMinOn_nuclearNorm_iff` | Invertible means `K.det ≠ 0`. |
| Theorem A, `x³ + x y` | `polarFactor_cubicKernel`, `norm_polarFactor_cubicKernel`, `not_isMinOn_cubicKernel` | |
| Theorem A, every flat kernel | `polarFactor_of_singularValues_eq`, `det_ne_zero_of_singularValues_eq`, `not_forall_isMinOn_iff_of_singularValues_eq` | Added: the source states it for `x³ + x y`. For every square `K` whose singular values are all equal and whose entries are all nonzero, M recovers `K` from no proper `Ω`, in D.2's form; `∑ σ_i² = ‖K‖_S2² > 0` makes the common value nonzero and `K` invertible. The example is `x³ + x y` at `p = 3` with `Ω = ∅`. |
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
- Lemma 1's modeling step: training with weight decay behaves like minimum-nuclear-norm
  interpolation;
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
- `ψ(a³ + a b)` has `‖K‖_S1 = p^{3/2}`. Its error is between 1.0 and 1.4 for every pattern
  above, half the table included.
- The certificate of item 3 for the diagonals `{0, 1}` has `‖Y − Q‖_op = cos(π/p)`: 0.900969,
  0.970942 and 0.994869. The proved bound `√(1 − 1/(4p²))` is 0.997446, 0.999260 and 0.999870.
- At `p = 61` and `p = 97` the cap was raised to 200000 iterations.
  - The diagonals `{0, 1}`, `{0, 5}` and `{3, (p − 1)/2}` give errors 8.8e-11 and 6.6e-11,
    after 6812 and 20196 iterations.
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
| random 20% | 1802–1921 | yes | 0.42–0.44 | 2e-11–4e-11 | 97.000 |
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
  Theorem E.3 proves `‖X‖_S1 < N` for `t = 1/(k + 1)²`. By hand, not formalized:
  `‖X‖²_S1 = N² − 2t (k l − g) + t² k l` for `t ≥ 0`, with `g = √(k l (N − k)(N − l))`.
  - The best `t` gives `‖X‖²_S1 = N² − (k l − g)²/(k l) < N²`.
  - At `k = 2`, `l = 60`, `N = 61` this is `60.912²`, the norm ADMM found.
  - For the cross, `k = l = N − 1`, it is item 9's `2√(N − 1)`.

Between the two cases, Theorem D has no certificate. Longer runs at `p = 31` keep lowering the
error, while `‖X‖_S1 − p` stays positive and shrinks:
- 12%, draw 1: 3.4e-5, 8.2e-6 and 3.1e-6 after 20000, 100000 and 300000 iterations;
- 20%, draw 0: 2.5e-5, 5.1e-6 and 1.7e-6;
- two diagonals minus a cell: 2.1e-4 after 20000 iterations, then 3.7e-9 when both residuals
  fall below `10⁻⁹` after 67491.

This looks like slow convergence to `K`, not a second minimum. Theorem E shows that M does
recover these patterns: each is rigid and contains a permutation graph.

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

### Exhaustive check (torch, 2026-10-28)

Every proper pattern `Ω ⊂ {0, 1, 2}²`, 511 in all, is classified by rigidity and by Hall's
condition. Batched ADMM on `J` then runs 20000 iterations for each. The error is the RMS of
`X − J` off `Ω`.

| `N = 3` | patterns | `‖X‖_S1 − N` | error |
| --- | --- | --- | --- |
| rigid, permutation graph | 195 | within 4e-15 of 0 | at most 4.7e-14 |
| rigid, no permutation graph | 9 | −0.172 | 0.5 |
| not rigid, permutation graph | 51 | within 9e-16 of 0 | 0.89–1.00 |
| not rigid, no permutation graph | 256 | −3.00 to −0.172 | 0.78–1.00 |

The four rows are Theorem E.2, E.3, E.1 with D.7, and E.3. The nine rigid patterns without a
permutation graph are the crosses, and `−0.172 = 2√2 − 3` is item D.9's least norm.

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
`(⅗, ⅕, ⅕)` and `(⅕, ⅖, ⅖)` (C). Permuting the shares of `b` (B) changed no count or range.

| weights | class | patterns | gap | error |
| --- | --- | --- | --- | --- |
| A | rigid, Hall | 144 | 0 to 3.1e-6 | at most 2.4e-4 |
| A | not rigid, Hall | 24 | within 2e-15 of 0 | 0.87 to 1.00 |
| A | rigid, no Hall | 60 | −0.54 to −0.14 | 0.37 to 0.67 |
| A | not rigid, no Hall | 282 | −3.00 to −0.14 | 0.69 to 1.00 |
| C | rigid, Hall | 110 | 0 to 3.9e-6 | at most 2.3e-4 |
| C | not rigid, Hall | 8 | within 2e-15 of 0 | 0.89 to 1.00 |
| C | rigid, no Hall | 94 | −1.00 to −0.10 | 0.29 to 0.75 |
| C | not rigid, no Hall | 298 | −4.00 to −0.10 | 0.63 to 1.00 |

Rigid Hall patterns are recovered (item 2). A Hall pattern that is not rigid reaches
`‖a‖ ‖b‖` (item 1) at another matrix (item 3's phases). Without Hall the least norm falls
below `‖a‖ ‖b‖` (item 3's witness). A rigid Hall pattern at tight weights converges slowly.
For the worst one at A, `{(0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}`, the error
is 2.4e-4, 5.0e-5 and 1.0e-5 after 6000, 30000 and 150000 iterations. Item 2 proves that the
limit is `K`.

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
`a bᵀ`. Each `N` splits the same way; the smallest gap below the threshold is −3.7e-6, −1.5e-4,
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
| rigid, no condition | 1118 | 0 | −2.86 to −3.5e-4 | 0.017 to 0.91 |
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
| 31 | `χ(a + ab)` | yes | yes | 7.1e-11, 7.5e-11 | at most 3.0e-15 | 100 % | 30 |
| 61 | `χ(ab)` | yes | yes | 2.0e-11, 3.2e-11 | 0 | 100 % | 60 |
| 61 | `χ(a + ab)` | yes | yes | 1.5e-11, 2.7e-11 | at most 4.3e-15 | 100 % | 60 |
| 97 | `χ(ab)` | yes | yes | 1.5e-11, 1.9e-11 | 0 | 100 % | 96 |
| 97 | `χ(a + ab)` | yes | yes | 1.6e-11, 1.9e-11 | at most 3.3e-15 | 100 % | 96 |

Every split meets both conditions and is recovered, as the corollary says: M returns the
kernel, `‖X‖_S1 = p − 1 = ‖K‖_S1`, and puts nothing on the zero row and column.

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
| Theorem E.1 | `sum_norm_sq_sub_eq`, `sum_sum_norm_sq_sub_perm`, `card_le_nuclearNorm_of_perm` | The permutation is an `Equiv.Perm` with `(i, σ i) ∈ Ω`. |
| Theorem E.2 | `eq_onesMatrix_of_perm`, `isMinOn_onesMatrix_iff_of_perm`, `isMinOn_vecMulVec_iff_of_perm`, `isMinOn_addKernel_iff_of_perm`, `rigid_ne_one_zero` | The examples use the rigid pattern `{(0, 0), (0, 1), (1, 1)}`. |
| Theorem E.3 | `blockFactorL`, `blockFactorR`, `blockWitness_apply`, `sum_ite_mem_univ`, `frobeniusSq_blockFactorL`, `frobeniusSq_blockFactorR`, `nuclearNorm_blockWitness_le`, `exists_nuclearNorm_lt_of_block`, `exists_nuclearNorm_lt_of_not_perm`, `not_isMinOn_onesMatrix_of_not_perm` | Hall's theorem is Mathlib's `Finset.all_card_le_biUnion_card_iff_exists_injective`. |
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

Not formalized:
- the numerical tables, including the certificates found for random patterns and for the
  training splits;
- the sharp value `‖Y − Q‖_op = cos(π/p)`;
- that two diagonals minus one cell are rigid. With Theorem E.2 this gives recovery from those
  `2p − 1` cells, where item D.1 gives no `c > 1`: the only `Y` with unit row and column sums
  is a permutation matrix, with `‖Y − Q‖_op = 1`;
- the failures on `2p` random cells;
- the exact norm of the witness `J − t 1_R 1_Cᵀ` and its best `t` (*Patterns*); Theorem E.3
  proves only `‖X‖_S1 < N`;
- the weighted and cross tables;
- that training behaves like model M (Lemma 1's modeling step).

Scope:
- The spectrum alone does not decide recovery; the pattern does. `ψ(a + b)` has the best
  spectrum, rank one. It is recovered from two diagonals, `2p` cells, and from the 50% training
  splits, which are rigid and contain a permutation graph (checked numerically), so Theorem E
  applies. It is not recovered from one diagonal, from two diagonals whose offset
  is not a unit, from row 0 and column 0, or from `2p` random cells.
- Rigidity is necessary (item 7): otherwise a phase on one component changes `K` off `Ω`. A
  rigid pattern is connected, so it has at least `2p − 1` cells; two diagonals use one more.
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
