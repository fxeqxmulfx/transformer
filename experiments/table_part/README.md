# What a part of a table determines

Source: `review/flat_spectrum.md`, section *What a part of the table determines*, PT1 to PT7.
Theorem A says that M returns a flat kernel from no proper `Ω`. That leaves open whether `Ω` still
holds the table. This experiment checks the answer inside the class of NV1,

```
H = { g(u) + h(v) + uᵀ M v },    u ∈ F_p^k,  v ∈ F_p^l,    dim H = n − 1 + k l,    n = p^k + p^l.
```

By PT3 a part `Ω` puts `n − c + s` independent conditions on `H`. Here `c` is the number of
components of its bipartite graph, and `s` is the dimension of the span of the twists
`Σᵢ (uᵢ − uᵢ₊₁) vᵢᵀ` of its closed walks. `Ω` determines `H` exactly when `c = 1` and `s = k l`.

## What is measured

1. **The count against the rank.** The rank of the evaluation matrix `[1_u | 1_v | vec(u vᵀ)]` on
   `Ω`, by elimination over `F_p`, is compared with `n − c + s`, from a weighted union-find that
   never forms that matrix. Six tables, of 4 to 25 cells, are run on every subset. The others take
   three families of parts in turn, by SplitMix64: uniform parts of `n − 1` to `3n` cells, the cross
   with up to `2 k l` random cells, and parts inside the zero set of a random function with
   `M ≠ 0`, which must not come out determined.
2. **The least part.** The rank on the cross with the cells `(e_i, e_j)`.
3. **The least support.** The fewest nonzero cells of a nonzero function, over all `p^{dim H}`
   functions.
4. **A cycle is not enough.** The six cells off the line `a + b = 2` at `p = 3`.
5. **The worst part** at `p = 97`: the table without one row, then with one of its cells back.
6. **Random parts** at `p = 97`: 5000 uniform parts of each size.
7. **The frozen half splits.** The 48 tasks with a table in `convex_stress_arithmetic_data`, on the
   split by SplitMix64(1) that E6 trains on. A union-find with offsets affine in the unknown `m`
   solves `g(a) + h(b) + m a b = y` on the training cells in `O(|Ω| log p)`. It reports that no
   function of the class fits, that several do, or the one that does. That one is scored on the
   held-out cells, which the solve does not see.

## Result

`cargo run --release`: 5 s on 12 threads, 9 MB. The output is the same on 3 threads.

The count equals the rank on all 33,921,608 parts, and every branch of the criterion is reached in
every table.

| `p k l` | parts | disconnected | connected, `s < k l` | determined | mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2 1 1`, all | 16 | 11 | 4 | 1 | 0 |
| `2 2 1`, all | 256 | 191 | 56 | 9 | 0 |
| `2 1 2`, all | 256 | 191 | 56 | 9 | 0 |
| `3 1 1`, all | 512 | 307 | 87 | 118 | 0 |
| `2 2 2`, all | 65,536 | 29,219 | 31,252 | 5,065 | 0 |
| `5 1 1`, all | 33,554,432 | 9,874,531 | 511,765 | 23,168,136 | 0 |
| `7 1 1`, sampled | 60,000 | 23,509 | 8,995 | 27,496 | 0 |
| `11 1 1`, sampled | 30,000 | 12,893 | 4,050 | 13,057 | 0 |
| `3 2 1`, sampled | 60,000 | 25,881 | 14,696 | 19,423 | 0 |
| `3 1 2`, sampled | 60,000 | 25,882 | 14,698 | 19,420 | 0 |
| `3 2 2`, sampled | 30,000 | 8,110 | 11,208 | 10,682 | 0 |
| `5 2 1`, sampled | 30,000 | 15,783 | 5,675 | 8,542 | 0 |
| `2 3 2`, sampled | 30,000 | 7,957 | 14,236 | 7,807 | 0 |
| `97 1 1`, sampled | 600 | 367 | 59 | 174 | 0 |

The least part has `dim H` cells and full rank at `(p, k, l)` = `(3, 1, 1)`, `(5, 1, 1)`,
`(97, 1, 1)`, `(3, 2, 2)` and `(5, 2, 1)`. Over all functions the least support is `min(p^k, p^l)`
on nine shapes, and 1 at `p = 2`, `k = l = 1`, the one exception of PT4. The six cells at `p = 3`
have `c = 1`, `s = 0` and rank 5 of 6. At `p = 97` the table without row 0 has 9312 cells and rank
193 of 194, and one more cell gives 194.

Random parts at `p = 97`, 5000 of each size:

| cells | determine `H` | no empty row or column |
| ---: | ---: | ---: |
| 194 | 0.0000 | 0.0000 |
| 300 | 0.0000 | 0.0002 |
| 400 | 0.0506 | 0.0572 |
| 500 | 0.3694 | 0.3754 |
| 600 | 0.7322 | 0.7336 |
| 700 | 0.9070 | 0.9072 |
| 800 | 0.9640 | 0.9640 |
| 900 | 0.9894 | 0.9894 |
| 1000 | 0.9964 | 0.9964 |
| 1200 | 1.0000 | 1.0000 |
| 4704 | 1.0000 | 1.0000 |

On the frozen splits, 15 of the 48 tasks have a function of the class on their training cells, and
the solve returns every held-out label of all 15. Twelve are at `p = 97` on the full table, 4705 of
4705 held-out cells each: `a + b`, `a − b`, `(a + b)/2`, `a b`, `a + a b`, `a² + a b`,
`a² + a b + b²`, `a² + a b + b² + a`, `a² + b²`, `(a + b)²`, `a³ + b` and `a³ + a b`. The other three
are the constant `(a/b)⁰` at `p = 17`, `97` and `193`, whose tables have no column `b = 0`. All six
tasks of E6 are among the 15. For the other 33 tasks no function of the class fits the training
cells, and the solve says so.

## What this does not show

PT1 to PT4, the first-moment bound of PT5, the general claim of PT6, and the real half of E6 at
`p = 97` with its twelve tasks are proved in Lean (`TablePart` of
`src/Transformer/Grokking/FlatSpectrum/`). `cargo test --release` ties them to this data: the FNV-1a
hash of the order of the 9409 cells, which Lean computes from its port of the shuffle, is the hash
of the split of all 33 tasks with the full table at `p = 97`, and each of the twelve formulas in
Lean is the table of exactly one task. This check covers the tables listed, and its other counts,
including the table of random parts, are measurements rather than proofs. The class is given to the
solve: a part determines a task only against the alternatives the class allows, so the solve is not
a learner, and this says nothing about whether a model finds `H`. Labels are exact.
