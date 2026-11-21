# What each character code of a task needs from one layer

Source: `review/flat_spectrum.md`, NV9. One layer of the main model, with positional heads and
the scalars of `R_D` set aside, writes only tables `X` with `dim(col X + row X) ≤ D_ℂ + 3 = 67`.
This experiment computes that dimension for the codes of seven tasks over `ℤ/97`, together with
the cost of each code in one product layer and a first-order test on a half split.

## What is measured

The codes are the injective characters: `ψ_r(f) = exp(2πi r f / p)` for `r ≠ 0`, 96 of them, and
`χ^j(f) = exp(2πi j log f / (p − 1))` with `χ^j(0) = 0` for `gcd(j, p − 1) = 1`, 32 of them. For
each task and each code, on the `p × p` table `X = code(f(a, b))`:

- **rank** is the rank of `X`, and **dim** is `dim(col X + row X)`, the rank of `[X | Xᵀ]`. Both are
  computed exactly over `F_ℓ` for two primes `ℓ ≡ 1 mod p(p − 1)`, 1073757409 and 1073962273. The
  reduction of `ℤ[ζ_{p(p−1)}]` to `F_ℓ` is a ring homomorphism, so each is a certified lower bound
  on the rank over `ℂ`. A one-sided Jacobi SVD in double precision, with a threshold of `10⁻⁹` of
  the largest singular value, gives the same ranks; for `dim` it is run on the cheapest code of
  each family.
- **cost** is `(3/2) Σ σᵢ^{2/3}`, the least value of `½(‖G‖² + ‖o‖² + ‖W‖²)` over the factorizations
  `X = G diag(o) Wᵀ` of one product layer. The table gives its least and largest value in the
  family. The singular values, unlike the rank, can change between Galois conjugates.
- **`|R|/|G|`** is the first-order test at the cheapest code. With `X = U S Vᴴ` of rank `r`, let
  `G = U S^{−1/3} Vᴴ`, and let `R` be the component of `G` in the directions tangent to the rank-`r`
  matrices at `X` that vanish on the training cells. Moving along `−R` keeps the training cells
  and lowers the cost at rate `|R|²`, so `R = 0` is necessary for `X` to be a local minimum among
  the interpolants. `R` comes from `P_Ω P_T P_Ω w = P_Ω G` by conjugate gradients, `R = G − P_T w`.
  The training cells are the first `⌊p²/2⌋ = 4704` of a shuffle of the `p²` cells by SplitMix64.
  Seed 0 is the split reported, and seeds 1 to 4 show the spread.

## Result

`cargo run --release -- 97 5`, about 4 s on six threads. In every family the exact rank and `dim`
are the same for all codes and both primes, and the floating-point ranks agree. On the training
cells `|R|` is at most `7·10⁻¹¹ |G|`.

| `f` | code | rank | dim | cost, least | cost, largest | `\|R\|/\|G\|`, seed 0 | seeds 1–4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `a³ + ab` | `ψ` | 97 | 97 | 668.5 | 668.5 | 0.707 | 0.707 |
| `a³ + ab` | `χ` | 48 | 72 | 415.0 | 415.0 | 0.586 | 0.584–0.588 |
| `ab` | `ψ` | 97 | 97 | 668.5 | 668.5 | 0.707 | 0.707 |
| `ab` | `χ` | 1 | 1 | 31.4 | 31.4 | 0.000 | 0.000 |
| `a²b` | `ψ` | 49 | 73 | 423.7 | 423.7 | 0.593 | 0.588–0.593 |
| `a²b` | `χ` | 1 | 2 | 31.4 | 31.4 | 0.000 | 0.000 |
| `a + ab` | `ψ` | 97 | 97 | 668.5 | 668.5 | 0.707 | 0.707 |
| `a + ab` | `χ` | 1 | 2 | 31.4 | 31.4 | 0.000 | 0.000 |
| `a² + ab` | `ψ` | 97 | 97 | 668.5 | 668.5 | 0.707 | 0.707 |
| `a² + ab` | `χ` | 96 | 97 | 656.2 | 656.2 | 0.706 | 0.706–0.708 |
| `a² + ab + b²` | `ψ` | 97 | 97 | 668.5 | 668.5 | 0.707 | 0.707 |
| `a² + ab + b²` | `χ` | 96 | 96 | 542.4 | 585.7 | 0.709 | 0.705–0.708 |
| `(a + b)²` | `ψ` | 97 | 97 | 668.5 | 668.5 | 0.707 | 0.707 |
| `(a + b)²` | `χ` | 96 | 96 | 661.6 | 661.6 | 0.707 | 0.707 |

Against 67, no code of `a³ + ab`, `a² + ab` or `a² + ab + b²` fits one layer, and the codes
`χ(ab)`, `χ(a²b)` and `χ(a + ab)` do. The margin on `χ(a³ + ab)` is five dimensions, so a state of
`D_ℂ ≥ 69` passes the count.

## What this does not show

The lower bounds on `dim` are exact. Equality with the floating-point rank is numerical. The count
concerns a code written into a value, not the task: no code of `(a + b)²` fits, and main passes
that task in one layer. The first-order test is on the code itself, so `R ≠ 0` says that this code
is not a local minimum among interpolants, and says nothing about other features.
