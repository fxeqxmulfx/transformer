# RoPE parameters, context length, and endpoint dependence

The development covers the mathematical results listed below from the
downloaded manuscripts and posts. It distinguishes the rotary dimension, the number of
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
| Ding et al., *LongRoPE* | [arXiv:2402.13753v1](https://arxiv.org/abs/2402.13753v1) | `papers/arXiv-2402.13753v1/main.tex` and the section files it inputs (TeX source) |
| Shang et al., *LongRoPE2* | [arXiv:2502.20082v1](https://arxiv.org/abs/2502.20082v1) | `papers/arXiv-2502.20082v1/paper.pdf` |
| Oka et al., *Frequency Bands in RoPE* | [ICLR 2026, PR1PPxvG9Q](https://proceedings.iclr.cc/paper_files/paper/2026/hash/993dbaec0418a6449090f1debbcb8844-Abstract-Conference.html) | `papers/iclr-2026-frequency-bands-rope/paper.pdf` |
| Wu, Liu and Jadbabaie, *How Data Shapes RoPE Frequency Usage* | [arXiv:2607.07678v1](https://arxiv.org/abs/2607.07678v1) | `papers/arXiv-2607.07678v1/paper.pdf` |
| Karypis et al., *LeRoPE* | [arXiv:2607.10134v1](https://arxiv.org/abs/2607.10134v1) | `papers/arXiv-2607.10134v1/paper.pdf` |
| Wu, Du and Peng, *RoPE at the End of Its Rope?* | [arXiv:2609.39929v1](https://arxiv.org/abs/2609.39929v1) | `papers/arXiv-2609.39929v1/paper.pdf` |
| Su et al., *RoFormer* | [arXiv:2104.09864v5](https://arxiv.org/abs/2104.09864v5) | `papers/arXiv-2104.09864v5/roformer_arxiv.tex` (TeX source) |
| Su, *Transformer升级之路：12、无限外推的ReRoPE？* (ReRoPE, Leaky ReRoPE), 2023-08-07 | [kexue.fm/archives/9708](https://web.archive.org/web/2023id_/https://kexue.fm/archives/9708) (2023 archive snapshot); code [bojone/rerope@acfe788](https://github.com/bojone/rerope/tree/acfe78840357a94d0bb504cb8865eeaa752df1a6) | `papers/kexue-9708-rerope/page.html`, `page.txt`; `papers/github-bojone-rerope/` |
| bloc97, *NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation*, r/LocalLLaMA, 2023-06-29, with its comment thread | [reddit.com/r/LocalLLaMA/comments/14lz7j5](https://web.archive.org/web/2023id_/https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) (2023 archive snapshot) | `papers/reddit-14lz7j5-ntk-aware/post.html`, `post.txt` |
| Su, *Transformer升级之路：10、RoPE是一种β进制编码* (RoPE is a β-base encoding), 2023-07-06 | [kexue.fm/archives/9675](https://web.archive.org/web/2023id_/https://kexue.fm/archives/9675) (2023 archive snapshot) | `papers/kexue-9675-ntk-beta/page.html`, `page.txt` |

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

TeX source SHA-256:

```text
2104.09864v5 roformer_arxiv.tex f296bf76e65f308d07af20823375f42eb7806352cd5647d140e55c274c0302cc
2402.13753v1 main.tex 575b65578b57dd52b0cd55c1beb6a78eadfc194ef915a6e5538e237ded1787d9
2402.13753v1 abstract.tex a00216e8c116ee1783feaef2e0f2a913d93a8f7dc7bcae5a079e55e9d923fcf4
2402.13753v1 intro.tex 0b46e3ef04b0b3a8b3c73b12753a1d925223b004f4c22e963dc297f47a70c3d1
2402.13753v1 analysis.tex 259cbe4513e930a0f9fb3c437ff01cacd4aa6ebb9018182d3693d32011b9ec8e
2402.13753v1 method.tex b3d461e6b4124e5ad85281907ed1e7aa1c7f3b82c86167da81e5cd3df66be526
2402.13753v1 eval.tex 05ae587078c26549c30f55a0613b26ad1a70952a7748e499508960eb0a9caa30
2402.13753v1 appendix.tex 79708aaeb077da0572c36b74f0443d3103ba8c093de984877b1709a7a8c963cc
```

Post and code SHA-256:

```text
kexue-9708 page.html 0f1c270c1cd47b40e77227827d877509e691460f78f8621941a67d6b689854b9
kexue-9708 page.txt 9867d0f8d8f79a6b0f8759c5437a16b4f923e200a3754d80509f139e08536f1d
rerope@acfe788 rerope_patch.py 5c68798f2af0086b008c4dd297137e36bd72d96ce13199fcf4491635c0d4dfd8
rerope@acfe788 leaky_rerope_patch.py 9364b88d75bf53caef2dcb18eac0c73de5cfd4ea2c497f4dbe72204870b3ae96
reddit-14lz7j5 post.html 83cbd181e2c93741d9be5a13f41c91ddc129be07a4464c9f7c526a38ce02423f
reddit-14lz7j5 post.txt 7299392a982b22ee36b8945debdce615df7ac654c04aea4f53e7058e42f91398
kexue-9675 page.html 26cc7d7b3692b6c14a3bf10c36d23d7417d0198f3948a87c3531a2ba66b175a7
kexue-9675 page.txt 420a438947be5d2f715759a012f933bf8b364bc48d653ae4c16172ad6d5194fc
```

## Statement-to-source coverage

Paths below are relative to `src/Transformer/RoPE/`. Every named theorem is
included in `scripts/RoPEAxioms.lean`; statements with assumptions have concrete
witness examples beside them.

| Source | Modules | Proved content |
| --- | --- | --- |
| Shared notation and existing model | `Basic.lean`, `Scores.lean` | Agreement with `GPTMini.invFreq`; actual last coordinate of an even head; monotone frequency grid; field/period rescaling; bounded integrable contrast; real query/key score expansion; realization of arbitrary score coefficients; amplitude bounds. |
| Liu §3.2, Eq. (4) | `ScalingLaws/Section3_Periods.lean` | Full periods and the exact integer cutoff between complete and incomplete periods, with a strict inequality matching the boundary convention. |
| YaRN §3.2, footnote ‡, and A.2, Definition 3 and Eqs. (17)–(22) | `YaRN/SectionA2_NTK.lean`, `YaRN/Section3_Bands.lean` | Coordinatewise NTK factors; preservation of the highest frequency; matching of the lowest-frequency phase; every pair but the last beyond its pretrained phase over the extended window, the "out-of-bound" values; the pair `d` extended by `s^{2d/(|D|−2)}` only, so a nominal scale above the target; the scale from the two bases; positivity and bounds for the actual interpolation blend; footnote ‡'s harmonic mean as linear interpolation of wavelengths, between PI and the blend. |
| YaRN §3.2–3.4, Eqs. (10)–(15), Definitions 1–2, and A.3, Eq. (23) | `YaRN/Section3_Ramp.lean`, `YaRN/Section3_Temperature.lean`, `YaRN/Section3_Algorithm.lean` | The actual rotation-count ramp in all three regimes; context-derived scale; positive compressed frequencies; the squared Q/K temperature gain; exact softmax odds and preservation of logit order; the complete score construction, identity without extension, and fixed-activation slope and margin bounds; the gain `1.20794…` of Eq. (23) at `s = 8`. |
| Derived investigation of YaRN §3.3, Eq. (14), using existing ALM and Imbalance softmax results | `YaRN/Retention.lean`, `YaRN/Calibration.lean`, `YaRN/Approximation.lean` | Exact probability-preserving temperature under an explicit score model; a worst-case bound for arbitrary scores with a certified positive gap; changed-gap calibration; impossibility of a length-only rule even with positive margins; the local linear coefficient and exact quadratic approximation error. |
| YaRN §§2.1–2.3, Eqs. (3)–(10) and (16), A.1 | `YaRN/Section2_Background.lean` | The inner product as the real part of the Hermitian product; Eq. (5) as multiplication by `e^{imθ_l}`; the score through `n − m` only; Eq. (7) for PI as the frequencies `θ/s`; Eq. (16)'s `m/s` keeps the extended window inside the pretrained one, the printed Eq. (9)'s `s·m` does not; Eq. (8)'s wavelength as the least period of the rotation; Eq. (10) in closed form. |
| YaRN §§3.1–3.2, footnote † | `YaRN/Section3_Pressure.lean` | PI's frequencies at most `1/s`; the NTK-aware divisors growing from 1 to `s`; strictly increasing wavelengths, which for LLaMA (`b = 10000`, `|D| = 128`, `L = 2048`) exceed `L` exactly at the pairs `d ≥ 41`; distinct rotations at the positions below a wavelength. |
| YaRN §3.4 | `YaRN/Section3_Dynamic.lean` | A rotation moves when a pair's phase moves by less than a turn; each Dynamic Scaling step past `L` moves the rotation of every token at `0 < n ≤ l'`, under PI and under Dynamic NTK, and never the token at 0; a key cached after RoPE gives a stale score. |
| YaRN §§4.4–4.5, Tables 3, 4 and 9 | `YaRN/Section4_Experiments.lean` | The mean drop of `0.4875` points behind "0.49%", a pooled relative drop of `0.88%`; Code Llama's NTK-aware scale `100^{(|D|−2)/|D|}`, `93.06` at `|D| = 128` and `86.60` at `|D| = 64`, not `88.6`; Together's PI scale `8`, not Table 9's `4`. |
| Men §4.2–4.3, Appendix A | `BaseBounds/Section4_ExpectedScores.lean` | The integral identity for the expected score gap under explicit coefficient moments; its nonzero-variance sign criterion; a counterexample to the omitted zero-mean/cross-moment implication. |
| Men §2.2, Eq. (6) | `BaseBounds/Section2_OOD.lean` | The NTK base identity in the source's notation; every period of base 500 is below 4096. |
| Men §4.1, Eq. (7) | `BaseBounds/Section4_LongTermDecay.lean` | Summation by parts; the bound with RoFormer's complex pair products and `h_{d/2} = 0`; a counterexample to the printed real products. |
| Men §4.2–4.3, Eqs. (10)–(11) | `BaseBounds/Section4_CosineSums.lean`, `BaseBounds/Section4_ContextLength.lean`, `BaseBounds/Section4_BaseBound.lean` | Bounded partial sums of the cosine kernel; it turns negative for all distinct frequencies in `(0, pi)` and is not antitone; the attained supremum of Eq. (10) for every base above 1; nonempty Eq. (11) set with at least two pairs; the lower-bound property and monotonicity of `base_L`; context length 1 for one pair. |
| RoFormer §3.2, Eqs. (12)–(16), and §3.4.2, Eq. (34) | `RoFormer/Section3_Rotary.lean` | The 2D solution and its inner product; Eq. (13) as multiplication by `e^{imθ}`; the block-diagonal rotary matrix, its group law, transpose and orthogonality; Eq. (16) with the left factor `x_mᵀ W_qᵀ`; the sparse realization. |
| RoFormer §3.4.1, Eqs. (20)–(33) | `RoFormer/Section3_Derivation.lean` | The rotations solve the relative-position equations; with the source's radial solution they are the only solutions, and the initial condition forces `γ = 0`; without that choice, a solution that is not a rotation. |
| RoFormer §3.3 and §3.4.3, Eqs. (19) and (35)–(37) | `RoFormer/Section3_Properties.lean` | The complex form of the score; the summation-by-parts bound with `h_{d/2} = 0`; relative weights and a query-independent key–value summary in linear attention; a negative numerator weight with nonnegative features. |
| ReRoPE post, 重温 and 融合, matrices (rope), (leaky-rerope), (rerope) | `ReRoPE/Section2_Fusion.lean` | Leaky ReRoPE keeps every position inside the window, and for a window below the training length some `k` keeps all positions of any length below it; Position Interpolation bounds the positions but moves every nonzero one; the limit `k → ∞` is ReRoPE, whose positions stay in `[0, w]`, while every finite `k` exceeds any training length on long inputs; `k = 1` is RoPE, and the positions are RoPE's through a Leaky ReLU and a ReLU. |
| ReRoPE post, 计算 | `ReRoPE/Section3_Computation.lean`, `ReRoPE/Section3_SinglePass.lean` | Leaky ReRoPE's second attention matrix; ReRoPE's second matrix with `R^{−w}`, and a counterexample to the printed `R^w`; the merged matrices are RoPE scores at the rectified positions; the one-matrix decoding forms, the leaky one for `k ≥ 1`; one RoPE pass with any absolute positions has relative positions linear up to whole turns, so it gives neither ReRoPE (`w ≥ 1`) nor Leaky ReRoPE (`k > 1`) at a frequency in `(0, 2π)`. |
| ReRoPE post, 实验, and `bojone/rerope` at acfe788 | `ReRoPE/Section4_Experiments.lean` | The query scale `max(1, log_T n)` is 1 up to the training length and above 1 beyond; the code's offset `w(k − 1)` gives the source's query position; its decoding table holds Leaky ReRoPE's positions and its `−sin` rotation is `R^{−t}`, so both decoding paths give the merged scores. |
| NTK-aware post (bloc97) | `NTKAware/Section1_Post.lean` | A rotated vector moves between positions `t` apart by twice the sum of its pairs' contrasts; under Position Interpolation by `s` the displacement between adjacent positions tends to 0; the NTK-aware base keeps the first frequency 1, so the displacement is at least `(8/9)|x_0|²` for every `α`, and a frequency-1 pair tells all integer positions apart because `π` is irrational. |
| NTK-aware post, comment thread | `NTKAware/Section2_Comments.lean` | The last pair's frequency divided by `α` exactly; with the base `base·α` the slowest phase over `αT` is `α^{2/d}` times the trained one, beyond it for `α > 1`; the clock readings tell apart exactly the times below the slowest period, `43.2k` and `129.6k` seconds. |
| β-base post, 进制表示 to 进制转换 | `BetaBase/Section1_Representation.lean` | Below `β^L` the digits determine the number; smaller bases need more digits; a reserved digit is 0 exactly below `β^L`; after division by `k ≥ 1` the higher digits step by 0 or 1; the highest differing digit decides the order in every base; `L` digits in base `b` cover `N` numbers iff `b ≥ ⌈N^{1/L}⌉`, and `⌈∛2000⌉ = 13` is the least base. |
| β-base post, 位置编码 and 追根溯源, Eqs. (1)–(3) | `BetaBase/Section5_Encoding.lean` | Eq. (1) as a floor; the digit is periodic in the quotient; RoPE's angle is `n/β^i`; `d/2` digits reach `k` times as far iff `β' ≥ β k^{2/d}`, and the base `base·k` reaches exactly `k` times; Eq. (3) iff `λ = k^{2/(d−2)}`, whose base is the NTK-aware `base·k^{d/(d−2)}`; `λ → 1` and `k^{2/(d−2)}/k^{2/d} → 1`. |
| LongRoPE §2.1, Eqs. (1)–(2) | `LongRoPE/Section2_Unified.lean` | `β^i = 1/θ_i`, so Eq. (2) is Eq. (1) with `θ_i/λ_i`; PI as the position `n/s`; the NTK-aware base as `λ_i = s^{2i/(d−2)}`; the printed `λ = s^i` as the base `θ·s^{d/2}`, above `s^{2i/(d−2)}` for `d > 4` and `i ≥ 1`, where the NTK-aware factor of the last pair is `s`; YaRN's frequencies as `λ_i = 1/((1−γ_i)/s + γ_i)`, 1 where `γ_i = 1`, `s` where `γ_i = 0`, in `[1, s]` and growing with `i`. |
| LongRoPE §§3.1–3.2, Eq. (3) and Table 4 | `LongRoPE/Section3_Search.lean` | Eq. (3) as RoPE below `n̂` and Eq. (2) from it on; `n̂ = 0` as PI and NTK; the factor idle at exactly the `n̂` positions `0, …, n̂−1`; the thresholds 0 and 1 give the same angles; `400^{64}·14` between `4.76·10^{167}` and `4.77·10^{167}`, and 401 grid values; the count `o(c^s)` for every `c > 1`; the PI, NTK and YaRN factors admissible under the monotone constraint; the printed `s^i` inadmissible for `s ≥ 2`; `4^{1/63}` strictly between the grid values 1.02 and 1.03. |
| LongRoPE §1, §3.3, §§4.2–4.3 and Appendix A | `LongRoPE/Section3_Progressive.lean`, `LongRoPE/Section4_Experiments.lean` | More than 99.5% new positions beyond `10^6`; the ratios `32×`, `64×`, `8×`, `16×` and `512× = 64·8`; `2048k > 2·10^6`; PI at `512×` keeps the original window's angles below `8θ_i`; causal attention's `n(n+1)/2` scores are `o(c^n)`; TruthfulQA's `+0.5` points and Table 10's averages 52.9 and 50.8 as means of Table 8. |
| LongRoPE2 §2.2, Eqs. (8)–(10), and §3.3 | `LongRoPE2/Section2_Rescaling.lean` | The scaling-factor constraint is necessary and sufficient for endpoint phase coverage; it bounds every in-window phase; period scaling and the mixed-context rule preserve the original short-context phases. |
| Oka §5.1, Eqs. (4)–(9) | `FrequencyBands/Section5_Variance.lean` | The variance formula from actual integrals; the exact continuous band-coordinate identity; a counterexample to the global-unimodality assertion in Step 2. |
| Wu/Liu/Jadbabaie §3, Theorem 1 | `FrequencyMatching/Section3_Optimality.lean` | Strict increase of the actual integrated contrast and the unique maximizer `pi / width`, including atomic distributions and the essential positive-mass assumption. |
| Wu/Liu/Jadbabaie §4, Theorem 2 and Corollary 1 | `FrequencyMatching/Section4_Dilation.lean` | Exact pushforward-measure utility invariance and maximizer transport; a counterexample when frequencies are scaled without dilating the task; no positive frequency covers unbounded widths. |
| Derived endpoint design results | `FrequencyMatching/Section3_WorstCase.lean`, `FrequencyMatching/Section3_Retention.lean` | Actual maximum token distance; exact slow-frequency base threshold; guarantees at all nearer distances; opposite directions of retention and contrast; a unique contrast optimum under a fixed retention floor; a realizable attention-score counterexample. |
| LeRoPE §3.1, Eq. (5), and §3.2 | `LeRoPE/Section3_LearnedFrequencies.lean` | Positive frequencies, zero initialization, representation of every positive target, multiplicative increments, and the exact local log-frequency score derivative. |
| Wu/Du/Peng A.2, Proposition 2, and A.4/C.6 | `Scores.lean`, `Diagnostics/SectionA4_Response.lean` | Real score representation and its converse; exact adjacent-position response; deterministic frequency/amplitude bounds for arbitrary coefficients and complete heads. |
| Wu/Du/Peng C.6 generalized to arbitrary shifts; derived margin guarantees | `Diagnostics/SectionC6_Stability.lean`, `Diagnostics/SectionC6_Margins.lean` | Exact response for any displacement; spectral Lipschitz budget; common query shifts with different baseline key phases; pairwise and whole-window ranking preservation under explicit reference margins; unbounded score changes without amplitude restrictions. |

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
6. **Men Eq. (7) needs RoFormer's complex products.** With the printed real
   products `h_l = q[2l:2l+1]^T k[2l:2l+1]`, one pair with `q = (1, 0)`,
   `k = (0, 1)` has `h_0 = 0` and a right side of `0`, while the score at
   `m = 1` is `sin 1`. The complex products `h_l = q_l conj(k_l)` with
   `h_{d/2} = 0` give the bound by summation by parts.
7. **Men §4.2–4.3: every base bounds the context length.** The source
   observes a negative kernel "for a very small base value". For all distinct
   frequencies in `(0, pi)`, hence for every RoPE base above 1, the kernel has
   bounded partial sums and turns negative, so `L_theta` is finite and
   attained. The kernel is not antitone in the distance, contrary to the
   literal "decreases". Eq. (11) needs the implicit range `base > 1` (at
   `base = 0` the kernel is `cos m + d/2 - 1 >= 0`) and at least two pairs; it
   gives a necessary bound only, and Table 1 stays numerical.
8. **RoFormer Eq. (16) and §3.4.1.** Eq. (16) prints
   `x^T W_q R_{n-m} W_k x_n`; its left factor is `(W_q x_m)^T = x_m^T W_q^T`.
   The derivation picks the radial solution `R_q(x_q, m) = ||q||` ("a
   straightforward solution"): `q 2^m` and `k 2^{-n}` also solve Eqs. (21)–(22).
   With that choice the rotations are the only solutions, and the initial
   condition already forces `gamma = 0`. The decay of the averaged `|S_j|` is
   read off Figure 2 and is not formalized.
9. **ReRoPE post, 计算: ReRoPE's second matrix has `R^{-w}`.** The post prints
   `a^(2)_{i,j} = (R^w q_i)^T k_j = q_i^T R^w k_j`; the rotation moves to the
   key transposed, so the second form is `q_i^T R^{-w} k_j`, as in the post's
   decoding form `R^{max(j-i,-w)}`. With one pair, `theta = pi/2`, `w = 1`,
   `q = e_1` and `k = e_2`, the printed sides are `1` and `-1`. The leaky
   decoding form `R^{max(j-i,(j-i+w)/k-w)}` needs the implicit `k >= 1`; the
   code's decoding table needs only `k != 0`. "Realizing the matrices takes
   two attention matrices" holds for one RoPE pass with arbitrary absolute
   positions at a frequency in `(0, 2pi)`, with `w >= 1` and `k > 1`. The
   accuracy and loss tables are measurements and are not formalized.
10. **NTK-aware scaling: the post, its thread and the β-base post.** bloc97's
   "all the positions are perfectly distinguishable" does not separate the
   NTK-aware base from Position Interpolation by exact injectivity: a rational
   scale keeps integer positions apart too. What separates them is the
   displacement between adjacent positions, which interpolation sends to 0 and
   the NTK-aware base bounds below. The clock analogy holds when the hours are
   slowed by 2 against the new 90-minute hours, the reading under which
   `129.6k` follows; slowed by 2 against the seconds they would give `86.4k`.
   The clock is "a RoPE with a base of 60" only up to the hour hand, which
   turns once per 12 turns of the minute hand. Su's β-base post calls `n/β`
   the highest-frequency term; the highest is `n/β^0 = n`, which no change of
   base touches, and `n/β` is the second. After Eq. (1) the text divides by
   `β^{k-1}` for `β^{m-1}`. The perplexity graphs and the accuracy table are
   measurements and are not formalized.
11. **YaRN: Eq. (9), A.2, §3.4 and the tables.** Eq. (9) prints Position
   Interpolation as `g(m) = s·m`; Eq. (16) and the rest of the paper use
   `g(m) = mL/L' = m/s`, and `s·m` sends the position `L` of the extended
   window out of the pretrained one. A.2's "the last dimension `d ∈ D` is
   `|D| − 2`" counts real coordinates; in the pair index of Eqs. (6) and (18)
   the last pair is `|D|/2 − 1`, and Eq. (21)'s exponent `(|D|−2)/|D|` is that
   pair's in both readings. §3.4's "the RoPE of every token changes when `s`
   changes" leaves out the token at position 0, whose rotation is the identity
   at every scale; for every other token it holds at each Dynamic Scaling step,
   under PI and under Dynamic NTK. §4.4's "0.49% drop" is the mean drop in
   percentage points, 0.4875; relative to the `s = 16` scores the pooled drop
   is 0.88%. Tables 4 and 9 give Code Llama, base `10^4 → 10^6`, the NTK-aware
   scale 88.6; Eqs. (19)–(22) give `100^{(|D|−2)/|D|}`, 93.06 at LLaMA's
   `|D| = 128` and 86.60 at `|D| = 64`. Table 4's 44.2 for Xiong et al. is
   `50^{62/64}` if their base is `5·10^5`, which the manuscript does not state;
   under that `|D| = 64`, 88.6 may be a typo for 86.6. Table 9 gives Together's
   32k PI model the scale 4; Table 4's `4k × 8` and Eq. (20) give 8. The
   perplexities, passkey accuracies, benchmark scores and training times are
   measurements and are not formalized.
12. **LongRoPE: §2.1's NTK factor, Eq. (3)'s thresholds and the growth
   claims.** §2.1 gives NTK's rescale factor as `λ = s^i`. The NTK-aware base
   `θ·s^{d/(d−2)}` gives `λ_i = s^{2i/(d−2)}`, from 1 to `s`; `s^i` is the
   base `θ·s^{d/2}`, which divides the last pair's frequency by `s^{d/2−1}`
   and leaves Table 4's range `[1, 1.25s]` from `i = 2` on for `s ≥ 2`, so
   it cannot be the NTK individual of the initial population. §2.1 says
   YaRN's middle band "employs the NTK"; YaRN blends the PI and the native
   frequencies there. §3.1's "initial `n̂-1` token positions" names the last
   of the `n̂` positions `0, …, n̂−1` that keep RoPE, which §2.2 and §3.2
   count as `n̂`. Position 0 has the angle 0 under every factor, so Table 4's
   thresholds 0 and 1 give the same RoPE. Eq. (3)'s argmin is written over
   the documents `x ∈ X`; what the search minimizes over is the rescale
   factors and `n̂`. §3.2's `400^{128/2}×14` is `4.76…·10^{167}`, printed
   `4×10^{167}`, and the grid `(1.0, 5.0, 0.01)` has 401 values, which give
   more than `5.5·10^{167}` choices. The count is a polynomial of degree
   `d/2` in `s` and `o(c^s)` for every `c > 1`, not exponential in the
   extension ratio as §1 and §3.2 say; the `n(n+1)/2` scores of causal
   attention are likewise `o(c^n)`, against Appendix A.1's "GPU memory and
   computation time increase exponentially with the sequence length". The
   NTK-aware factors are off the 0.01 grid: `4^{1/63}` lies between 1.02 and
   1.03. The perplexities, passkey accuracies, benchmark scores and costs are
   measurements and are not formalized.

## Arbitrary coefficients and learned weights

For fixed query/key activations, any rotary head has the real score form

```text
S(r) = sum_i (a_i*cos(omega_i*r) + b_i*sin(omega_i*r)).
A_i = sqrt(a_i^2 + b_i^2).
C = sum_i |omega_i| * A_i.
```

The development proves, without distributional assumptions on those coefficients,

```text
|S(r + delta) - S(r)| <= 2 * sum_i A_i * |sin(omega_i * delta / 2)|
                    <= |delta| * C.
```

For a target and competitor, use their coefficient differences to compute
`C_margin`. If the reference margin exceeds `|delta| * C_margin`, their order
is preserved. If every competitor's reference margin exceeds `W * C_margin`,
the target remains the unique highest-scoring key everywhere in `[0,W]`.
Different keys' baseline positional phases can be absorbed into their
coefficients, so the same parameter may represent a common query shift.

These are sufficient bounds on variations, even when actual scores are not
monotone. They do not say that the actual score at the far endpoint is always
the smallest. An explicit construction shows that without a bound on coefficient
magnitudes, a fixed nonzero in-field phase change can cause arbitrarily large
score changes for realizable query/key vectors.

Training may create favorable spectral amplitudes, phases, and margins. Such
properties must be proved from a learning model or supplied as explicit
measured/certified hypotheses; they do not follow from a parameter vector being
trained. Recomputing Q/K when an input or preceding layer changes requires
additional control of those activation changes. The generic bounds here freeze
the activations and never hide that distinction in a performance claim.

## Complete YaRN algorithm and empirical defaults

The implementation of Definition 2 now includes every mathematical operation
in fixed-context YaRN, not just NTK-by-parts frequency blending:

```text
s = max(1, target_context / original_training_context)
r_i = original_training_context * omega_i / (2*pi)
gamma_i = clip((r_i - 1) / (32 - 1), 0, 1)
omega'_i = (1 - gamma_i) * omega_i / s + gamma_i * omega_i
m = 1 + log(s) / 10
Q' = m * rotate(Q, omega')
K' = m * rotate(K, omega')
logits = Q' * transpose(K') / sqrt(head_dimension)
```

The ramp is linear in **rotation count**, exactly as manuscript Eq. (11).
It is not replaced by a ramp linear in frequency index, which is used by
some implementations. The original training context is required provenance;
it cannot be inferred from the configured target context. For shorter targets
the clamping convention gives native RoPE and unit temperature gain. For
extensions it is precisely §3.4's fixed configured-context rule. Dynamic
per-sequence rescaling and its KV-cache implications are not implemented here.

The constants `alpha=1`, `beta=32`, and `m=1+0.1*log(s)` are **empirical
recommendations for LLaMA**, not proved optima. Section 3.3 Eq. (15) fits the
temperature giving the lowest perplexity at different extension factors for
unfinetuned LLaMA 7B, 13B, 33B, and 65B; it reports transfer to Llama 2. No
theorem here asserts that these constants minimize an arbitrary model's loss.
The complete algorithm is formalized; the paper's empirical evaluation is not
a mathematical theorem.

The temperature scales logits by `m^2`, preserves their ranking, and multiplies
softmax odds by changing `exp(margin)` into `exp(m^2*margin)`. For positive
frequencies, the interpolated slope before temperature lies in `[C/s,C]`.
After temperature its budget is `m^2*C_interpolated`. Reference margins are
also multiplied by `m^2`, so temperature cannot improve the margin/slope
certificate. All statements about score changes keep Q/K activations fixed.

## Deriving temperature instead of fitting it

The repository already supplies useful results:
`Transformer.ALM.softmax_winner_sharp` bounds arbitrary-score softmax mass
through the competing exponentials; `Transformer.Imbalance.sum_exp_twoLevel`
and `softmax_twoLevel` calculate the exact equal-tail softmax. The new proofs
reuse these results rather than treating a fitted temperature as an optimum.

An explicit objective is necessary. For retention of one desired token's
attention probability, let `N=L-1` be its original competitor count, `M=L'-1`
the new count, and let the desired token exceed every equal losing logit by
`Delta`. Logits include the ordinary division by `sqrt(head_dimension)`.
At logit gain `g=m^2`, its exact softmax probability is

```text
P(N, Delta, g) = 1 / (1 + N * exp(-g * Delta)).
```

For positive competitor counts and a nonzero gap, solving
`P(M,Delta,g)=P(N,Delta,1)` gives the **unique** answer

```text
g = 1 + log(M/N) / Delta
m = sqrt(1 + log(M/N) / Delta).
```

If RoPE rescaling or activation changes alter the gap, the exact equation is
instead `g=(Delta_old+log(M/N))/Delta_new`, with `Delta_new != 0`.
A physically positive temperature requires a positive resulting gain.
For a positive gap and `M>=N`, the unchanged-gap gain is at least one.
These equations optimize no unspecified loss: they solve the stated
probability-retention constraint in the equal-tail model.

For **arbitrary** competing logits, a certified lower gap `Delta>0` gives
`p_target >= P(M,Delta,g)`. The derived gain therefore preserves the old
**worst-case lower bound** `P(N,Delta,1)`. It does not in general preserve
the actual old probability, which can be larger than that bound. The gap
must hold for the scores at the new context and frequencies. Taking the
largest logit as the target certifies concentration on the existing winner,
not that this winner is the desired first token.

Two impossibility results rule out a universal default based only on length:

* With tied logits, extending two visible keys to three changes a token's
  probability from `1/2` to `1/3` at every finite temperature.
* Even with strictly positive winner gaps, extending one competitor to two
  needs `g=1+log(2)` at gap one, and `g=1+log(2)/2` at gap two. These are
  unequal. `no_length_only_retention` rules out every function of the two
  context lengths with retention for all such rows.

This is not a refutation of the measured LLaMA perplexities in YaRN. It
refutes promotion of any length-only fit to a universal retention law.

Write `u=log(M/N)`. The exact conditional Q/K gain is
`sqrt(1+u/Delta)` and its derivative at `u=0` is `1/(2*Delta)`.
Consequently a coefficient `0.1` occurs at **gap five** as a local tangent,
not as a universal constant. The squared tangent has the exact excess

```text
(1 + u/(2*Delta))^2 - (sqrt(1+u/Delta))^2 = u^2/(4*Delta^2).
```

This excess is strictly positive for positive growth. Also `M/N` is a
competitor-count ratio, while the empirical formula uses the token-count
ratio `s=L'/L`; they are close only for sufficiently large `L`.
The following are algebraic illustrations at `L=4096`, not language-model
experiments or perplexity predictions:

| Token extension `s` | YaRN fitted Q/K gain | Exact retention, gap 1 | Gap 5 | Gap 10 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1.138629 | 1.544823 | 1.130175 | 1.067074 |
| 16 | 1.277259 | 1.942374 | 1.246821 | 1.130169 |
| 128 | 1.485203 | 2.419147 | 1.403729 | 1.218699 |

The Python lab ports these results as `YaRN` and `extend_context` (see
`python/README.md`). The current row's top-two visible logits supply the gap;
there is no fitted-temperature constant or manually supplied gap. This is a
derived variant, with the winner-concentration objective and numerical guards
explicit. It is not paper YaRN's empirical temperature or an automatic quality
guarantee for arbitrary trained attention. Reference-gap changes between the
old and new model are not estimated by this implementation.

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
