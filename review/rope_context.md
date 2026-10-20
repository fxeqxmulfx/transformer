# RoPE parameters, context length, and endpoint dependence

The development covers the mathematical results listed below from eight
downloaded manuscripts. It distinguishes the rotary dimension, the number of
tokens, the largest relative distance, the frequency grid, and the learned
content-dependent score. All source files are reachable through
`Transformer.RoPE` and `Transformer`.

## Local manuscripts

Read and checked on 2026-10-20. PDFs and extracted text are in the root's
gitignored `papers/`; worktrees link that directory. Versioned arXiv paths
resolve to the downloaded PDF of the version printed in its manuscript.
The ICLR proceedings PDF is pinned by its proceedings identifier and hash.

| Manuscript | Frozen source | Local PDF |
| --- | --- | --- |
| Liu et al., *Scaling Laws of RoPE-based Extrapolation* | [arXiv:2310.05209v2](https://arxiv.org/abs/2310.05209v2) | `papers/arXiv-2310.05209v2/paper.pdf` |
| Peng et al., *YaRN* | [arXiv:2309.00071v3](https://arxiv.org/abs/2309.00071v3) | `papers/arXiv-2309.00071v3/paper.pdf` |
| Men et al., *Base of RoPE Bounds Context Length* | [arXiv:2405.14591v1](https://arxiv.org/abs/2405.14591v1) | `papers/arXiv-2405.14591v1/paper.pdf` |
| Shang et al., *LongRoPE2* | [arXiv:2502.20082v1](https://arxiv.org/abs/2502.20082v1) | `papers/arXiv-2502.20082v1/paper.pdf` |
| Oka et al., *Frequency Bands in RoPE* | [ICLR 2026, PR1PPxvG9Q](https://proceedings.iclr.cc/paper_files/paper/2026/hash/993dbaec0418a6449090f1debbcb8844-Abstract-Conference.html) | `papers/iclr-2026-frequency-bands-rope/paper.pdf` |
| Wu, Liu and Jadbabaie, *How Data Shapes RoPE Frequency Usage* | [arXiv:2607.07678v1](https://arxiv.org/abs/2607.07678v1) | `papers/arXiv-2607.07678v1/paper.pdf` |
| Karypis et al., *LeRoPE* | [arXiv:2607.10134v1](https://arxiv.org/abs/2607.10134v1) | `papers/arXiv-2607.10134v1/paper.pdf` |
| Wu, Du and Peng, *RoPE at the End of Its Rope?* | [arXiv:2609.39929v1](https://arxiv.org/abs/2609.39929v1) | `papers/arXiv-2609.39929v1/paper.pdf` |

PDF SHA-256:

```text
2310.05209v2 c1bc6b0e916cfd8d8aa5f046b174f91b7305e367c3de70d5fab19a6d6784df17
2309.00071v3 e7c0268a796138460c6ba2f67a7cba5bd922c401aea94c1fcd09c4b76b883c85
2405.14591v1 e710ed0f83c9bc1905c474ab8e1e2a45129dd5218f36c083984231566f4c9e57
2502.20082v1 46182adc037c46cdaeafe9eac886e4ec70c7c34f8e4adc99f858fdf6139119a6
ICLR-PR1PPxvG9Q dcd030428f6b36ef6f089b80677f41a6868e33d33cc93953503f7e84b5f36bf8
2607.07678v1 1757378003f19cf0b1b7743d9c6fd724f3f616555c0c86e0f7537186ca5efe39
2607.10134v1 a5ff7cdcbde2196dce3919659ca5252e0f9246e657b0fb47affc74ab13d05c1e
2609.39929v1 709c51e6c8aa0fe8fde7b83604b1c68a6598381f2420ff94a92e6be16ab77162
```

## Statement-to-source coverage

Paths below are relative to `src/Transformer/RoPE/`. Every named theorem is
included in `scripts/RoPEAxioms.lean`; statements with assumptions have concrete
witness examples beside them.

| Source | Modules | Proved content |
| --- | --- | --- |
| Shared notation and existing model | `Basic.lean`, `Scores.lean` | Agreement with `GPTMini.invFreq`; actual last coordinate of an even head; monotone frequency grid; field/period rescaling; bounded integrable contrast; real query/key score expansion; realization of arbitrary score coefficients; amplitude bounds. |
| Liu §3.2, Eq. (4) | `ScalingLaws/Section3_Periods.lean` | Full periods and the exact integer cutoff between complete and incomplete periods, with a strict inequality matching the boundary convention. |
| YaRN §3.2 and A.2, Eqs. (19)–(22) | `YaRN/SectionA2_NTK.lean`, `YaRN/Section3_Bands.lean` | Coordinatewise NTK factors; preservation of the highest frequency; matching of the lowest-frequency phase; positivity and bounds for the actual interpolation blend. |
| Men §4.2–4.3, Appendix A | `BaseBounds/Section4_ExpectedScores.lean` | The integral identity for the expected score gap under explicit coefficient moments; its nonzero-variance sign criterion; a counterexample to the omitted zero-mean/cross-moment implication. |
| LongRoPE2 §2.2, Eqs. (8)–(10), and §3.3 | `LongRoPE2/Section2_Rescaling.lean` | The scaling-factor constraint is necessary and sufficient for endpoint phase coverage; it bounds every in-window phase; period scaling and the mixed-context rule preserve the original short-context phases. |
| Oka §5.1, Eqs. (4)–(9) | `FrequencyBands/Section5_Variance.lean` | The variance formula from actual integrals; the exact continuous band-coordinate identity; a counterexample to the global-unimodality assertion in Step 2. |
| Wu/Liu/Jadbabaie §3, Theorem 1 | `FrequencyMatching/Section3_Optimality.lean` | Strict increase of the actual integrated contrast and the unique maximizer `pi / width`, including atomic distributions and the essential positive-mass assumption. |
| Wu/Liu/Jadbabaie §4, Theorem 2 and Corollary 1 | `FrequencyMatching/Section4_Dilation.lean` | Exact pushforward-measure utility invariance and maximizer transport; a counterexample when frequencies are scaled without dilating the task; no positive frequency covers unbounded widths. |
| Derived endpoint design results | `FrequencyMatching/Section3_WorstCase.lean`, `FrequencyMatching/Section3_Retention.lean` | Actual maximum token distance; exact slow-frequency base threshold; guarantees at all nearer distances; opposite directions of retention and contrast; a unique contrast optimum under a fixed retention floor; a realizable attention-score counterexample. |
| LeRoPE §3.1, Eq. (5), and §3.2 | `LeRoPE/Section3_LearnedFrequencies.lean` | Positive frequencies, zero initialization, representation of every positive target, multiplicative increments, and the exact local log-frequency score derivative. |
| Wu/Du/Peng A.2, Proposition 2, and A.4/C.6 | `Scores.lean`, `Diagnostics/SectionA4_Response.lean` | Real score representation and its converse; exact adjacent-position response; deterministic frequency/amplitude bounds for arbitrary coefficients and complete heads. |

## Worst-case design and the meaning of optimality

For `L` tokens the largest possible relative distance is `W = L - 1`.
The implementation uses `length + 1` tokens when constructing their `Fin`
indices, so `token_distance_bound` proves this endpoint distance without
silently identifying token count with maximum relative distance.

For Wu/Liu/Jadbabaie's scalar objective,

```text
D(omega, r) = 1 - cos(omega * r)
U(mu, omega) = integral D(omega, r) dmu(r)
Admissible(W, omega) = (omega > 0 and omega * W <= pi).
```

The unique optimizer is `omega = pi / W` when `W > 0`, the dependency
distribution is a probability measure supported in `[0,W]`, and it assigns
positive mass to `(0,W]`. A Dirac mass at zero shows why the last assumption
cannot be removed. This is the optimum of the stated contrast objective.

For an even rotary dimension `d > 2`, the actual slowest frequency is
`base ^ (-(d-2)/d)`. Its half-period covers `[0,W]` exactly when

```text
base >= (W / pi) ^ (d / (d - 2)).
```

This geometric threshold differs from Men's expected-score discrimination
criterion. It does not guarantee that a trained head uses the slowest
coordinate or that the head can retrieve the first token.

If the design instead prescribes a cosine-retention floor `-1 < rho < 1`,
the derived result chooses

```text
omega_best = arccos(rho) / W.
```

Among admissible frequencies preserving endpoint retention at least `rho`,
this frequency uniquely maximizes contrast at every positive in-field
distance simultaneously. It also guarantees retention at least `rho` at
every nearer distance. The retention metric at nearer distances is at least
its endpoint value; the contrast metric has the opposite ordering. A general
realizable content-dependent score need not attain its minimum at the endpoint.
Consequently, this result must not be called dominance across all attention,
retrieval, perplexity, and positional-resolution metrics.

## Corrections and limits of the source claims

1. **Oka §5.1 Step 2: global unimodality is false.** The proof evaluates the
   actual variance integrals at three ordered positive arguments:
   `V(pi) = V(2*pi) = 1/2` and `V(3*pi/2) < 1/2`. No nondecreasing-then-
   nonincreasing function can have this dip. This counterexample refutes the
   global-unimodality assertion; it does not refute the reported first maximum
   or the empirical frequency-band observations. The approximate numeric
   optimizer and fitted band multiplier are not claimed as proved.
2. **Men Theorem 1 and Appendix A need cross-moment control.** Zero-mean noise
   alone does not imply zero correlation with the query: on two equally likely
   signs, take `noise = -query`. Both means are zero and the cross moment is
   `-1`. `expected_score_of_moments` is the exact expectation lemma used in
   the calculation, with integrability and every required moment explicit.
   It is not advertised as a proof that the source's stated i.i.d. assumptions
   alone imply those moments for the unspecified perturbation.
3. **Liu's performance scaling laws are empirically supported model claims.**
   The formal results here prove the underlying periodic geometry and integer
   cutoff, without treating perplexity improvement or an extrapolation limit
   for arbitrary trained weights as a consequence of that geometry.
4. **LongRoPE2 and LeRoPE have empirical training results.** Search quality,
   needle retrieval scores, compute improvements, and generalization beyond
   training lengths are not inferred from their frequency formulas. The
   LeRoPE derivative is the local score contribution with Q/K coefficients
   held fixed, which is what the chain-rule calculation uses.
5. **The later diagnostic paper's calibrated probability results are separate.**
   Its Fourier-frame moment estimates, Gaussian threshold calibration, and
   conditional context-length theorem are outside the deterministic response
   results covered here. No unproved probability estimate is introduced as an
   axiom or hidden in a structure field.

## Validation

```sh
lake build Transformer.RoPE
lake env lean scripts/RoPEAxioms.lean
lake build
./run.py audit
./run.py index
./run.py forbidden
```

No new `sorry` is introduced. The repository's existing unfinished statements
are retained and are not dependencies of these results. The recorded kernel
axioms must be limited to `propext`, `Classical.choice`, and `Quot.sound`.
