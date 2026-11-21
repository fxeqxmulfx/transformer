module

public import Transformer.Basic
public import Transformer.Basis
public import Transformer.Wasserstein
public import Transformer.SmoothApprox
public import Transformer.ContinuityEq
public import Transformer.Wiener
public import Transformer.ALM
public import Transformer.RASP
public import Transformer.RASPL
public import Transformer.CRASP
public import Transformer.Perspective
public import Transformer.Metastability
public import Transformer.Interpolation
public import Transformer.Causal
public import Transformer.Normalization
public import Transformer.MeanField
public import Transformer.FrankWolfe
public import Transformer.Kinetic
public import Transformer.Homogenized
public import Transformer.Perceptron
public import Transformer.Clusters
public import Transformer.Modes
public import Transformer.AMSGrad
public import Transformer.AMSGradW
public import Transformer.AdamBeyond
public import Transformer.XSA
public import Transformer.Quartet
public import Transformer.GPTMini
public import Transformer.GPTMini.Sparsemax
public import Transformer.Precision
public import Transformer.Convexifying
public import Transformer.Zoology
public import Transformer.ConvexRecall
public import Transformer.CausalRecall
public import Transformer.MRCR
public import Transformer.Muon
public import Transformer.DASH
public import Transformer.AdaFisher
public import Transformer.Optimization
public import Transformer.ICEoT
public import Transformer.OptimizerBenchmark
public import Transformer.Magma
public import Transformer.Imbalance
public import Transformer.BatchSize
public import Transformer.DoubleDescent
public import Transformer.Memorization
public import Transformer.MagnitudeDirection
public import Transformer.MDW
public import Transformer.RecurrentGradients
public import Transformer.NoiseScale
public import Transformer.Surge
public import Transformer.ScheduleFree
public import Transformer.Grokking
public import Transformer.Landscape
public import Transformer.MHC
public import Transformer.Residual.RMSBackward
public import Transformer.AttentionResiduals
public import Transformer.DeepSeekV41
public import Transformer.RoPE
public import Transformer.BAM
public import Transformer.KERPLE
public import Transformer.Sandwich
public import Transformer.Weil
public import Transformer.Rotate90
public import Transformer.ComplexFFN
public import Transformer.ComplexProjection
public import Transformer.ReRoPEBlock
public import Transformer.LeRoPEBlock
public import Transformer.Phasor
public import Transformer.SequenceModel
public import Transformer.GainMatroid

/-!
Formalizations of papers on the mathematics and architecture of Transformers:

1. Weiss, Goldberg, Yahav — arXiv:2106.06981v2
   "Thinking Like Transformers".

2. Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio, Nakkiran —
   arXiv:2310.16028v1
   "What Algorithms can Transformers Learn? A Study in Length
   Generalization".

3. Yang, Huang, Chiang — arXiv:2506.16055v3
   "Knee-Deep in C-RASP: A Transformer Depth Hierarchy".

4. Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5
   "A mathematical perspective on Transformers".

5. Geshkovski, Koubbi, Polyanskiy, Rigollet — arXiv:2410.06833v1
   "Dynamic metastability in the self-attention model".

6. Geshkovski, Karagodin, Polyanskiy, Rigollet — arXiv:2411.04551v3
   "Measure-to-measure interpolation using Transformers".

7. Karagodin, Polyanskiy, Rigollet — arXiv:2411.04990v2
   "Clustering in Causal Attention Masking".

8. Karagodin, Polyanskiy, Rigollet — arXiv:2510.22026v2
   "Normalization in Attention Dynamics".

9. Geshkovski, Polyanskiy, Rigollet — arXiv:2512.01868v4
   "The Mean-Field Dynamics of Transformers" (survey).

10. Zhai — arXiv:2603.09078v1
   "Exclusive Self Attention".

11. Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1
   "Attention's forward pass and Frank-Wolfe".

12. Duerinckx, Geshkovski, Rossi — arXiv:2605.09213v1
   "Kinetic theory for Transformers and the lost-in-the-middle phenomenon".

13. Panferov, Schultheis, Tabesh, Alistarh — arXiv:2601.22813v2
   "Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased
   Gradient Estimation".

14. Geshkovski, Koubbi, Rigollet — arXiv:2604.01978v1
   "Homogenized Transformers".

15. Álvarez-López, Geshkovski, Ruiz-Balet — arXiv:2601.21366v2
   "Perceptrons and localization of attention's mean-field landscape".

16. Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2305.05465v6
   "The emergence of clusters in self-attention dynamics".

17. Geshkovski, Rigollet, Sun — arXiv:2412.09080v3
   "On the number of modes of Gaussian kernel density estimators".

18. Tran, Le — arXiv:1904.03590v4
   "On the Convergence Proof of AMSGrad and a New Version".

19. Reddi, Kale, Kumar — arXiv:1904.09237
   "On the Convergence of Adam and Beyond".

20. Ergen, Neyshabur, Mehta — arXiv:2211.11052v1
   "Convexifying Transformers: Improving optimization and understanding of
   transformer networks".

21. Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré —
   arXiv:2312.04927v1
   "Zoology: Measuring and Improving Recall in Efficient Language Models".

22. Liu et al. — arXiv:2502.16982
   "Muon is Scalable for LLM Training".

23. Modoranu et al. — arXiv:2602.02016v2
   "DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers".

24. Gomes, Zhang, Belilovsky, Wolf, Hosseini — arXiv:2405.16397v3
   "AdaFisher: Adaptive Second Order Optimization via Fisher Information".

25. Xu, Zhi, Jiang — arXiv:2603.22095v2
   "Input Convex Encoder-Only Transformer for Computationally Efficient
   Model Predictive Control in Building Demand Response".

26. Joo, Xia, Kim, Zhang, Ie — arXiv:2602.15322v1
   "On Surprising Effectiveness of Masking Updates in Adaptive Optimizers".

27. Kunstner, Yadav, Milligan, Schmidt, Bietti — arXiv:2402.19449v2
   "Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent
   on Language Models".

28. Sreckovic, Geiping, Orvieto — arXiv:2506.12543v1
   "Is your batch size the problem? Revisiting the Adam-SGD gap in language
   modeling".

29. Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever — arXiv:1912.02292v1
   "Deep Double Descent: Where Bigger Models and More Data Hurt".

30. Morris, Sitawarin, Guo, Kokhlikyan, Suh, Rush, Chaudhuri, Mahloujifar —
   arXiv:2505.24832v3
   "How much do language models memorize?".

31. Hägele, Hernández-Cano, Kosson, Jaggi — arXiv:2606.25971v2
   "Improving Neural Network Training by Decoupling the Magnitude and
   Direction of Weight Vectors".

32. Pascanu, Mikolov, Bengio — arXiv:1211.5063
   "On the difficulty of training Recurrent Neural Networks".

33. McCandlish, Kaplan, Amodei, OpenAI Dota Team — arXiv:1812.06162
   "An Empirical Model of Large-Batch Training".

34. Li, Zhao, Zhang, Sun, Wu, Jiao, Wang, Liu, Fang, Xue, Tao, Cui, Wang —
   arXiv:2405.14578v5
   "Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling".

35. Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky — arXiv:2405.15682v4
   "The Road Less Scheduled".

36. Zhu et al. — arXiv:2409.19606v3, "Hyper-Connections".
   The residual operator and its standard-residual specialization.

37. Xie et al. — arXiv:2512.24880v2, "mHC: Manifold-Constrained Hyper-Connections".
   Residual sections §§1, 4.1–4.2, with ideal and finite normalization kept distinct.

38. Kimi Team — arXiv:2603.15031v1, "Attention Residuals".
   Residual sections §§3–4.2: Full/Block AttnRes and exact two-phase merging.

39. DeepSeek-AI — "DeepSeek-V4.1-Flash Technical Report", §2.4.1.
   Single-Pass mHC and its activation traffic model.

40. Kimi Team — "Kimi K3: Open Frontier Intelligence", §2.2.
   The AttnRes residual mechanism, shared with item 38, and concrete block counts.

41. Liu et al. — arXiv:2310.05209v2, "Scaling Laws of RoPE-based Extrapolation".
   Exact periods and the integer critical-coordinate cutoff, §3.2.

42. Peng et al. — arXiv:2309.00071v3, "YaRN: Efficient Context Window Extension
   of Large Language Models". RoPE in complex coordinates, wavelengths and Position
   Interpolation (§2); the pressure of PI and the NTK-aware base, the wavelengths
   beyond the context and footnotes † and ‡ (§§3.1–3.2); bandwise interpolation and
   the temperature (§§3.2–3.3); Dynamic Scaling, Dynamic NTK and the key–value cache
   (§3.4); the arithmetic of Tables 3, 4 and 9 (§4); the out-of-bound phases of the
   NTK-aware base (A.2) and Eq. (23) (A.3). Eq. (9) prints PI as `g(m) = s·m`; the
   token at position 0 keeps its rotation; Table 4's Code Llama scale 88.6 and
   Table 9's Together scale 4 follow from neither Eqs. (19)–(22) nor Eq. (20).

43. Men et al. — arXiv:2405.14591v1, "Base of RoPE Bounds Context Length".
   Expected-score discrimination with explicit corrected moments, the NTK
   identity and base-500 periods, the attention-score bound of Eq. (7), the
   context length and base bound of Eqs. (10)–(11): §2.2, §4, Appendix A.
   Eq. (7) holds with RoFormer's complex pair products, not the printed real
   ones; the kernel turns negative for every base above 1, so every base
   bounds the context length, and it is not antitone.

44. Shang et al. — arXiv:2502.20082v1, "LongRoPE2: Near-Lossless LLM Context
   Window Scaling". Coordinatewise phase constraints and mixed windows
   (§2.2, §3.3); Eq. (7)'s critical dimension and Phi3-mini's periods
   (§§2.1–2.2); Liu's critical NTK base, the NTK-aware factor and YaRN against
   Eq. (10) (§§2.3–2.4). Eq. (7) counts real coordinates, twice the pair index
   that §2.4 calls `d_tcd`; YaRN's middle-band factor lies below the linear
   blend of 1 and `s`. The consequences of §3.1's hypothesis; Algorithm 1's
   candidates for `d_rcd`, 19–31 for Phi3-mini and 24–35 for LLaMA3-8B; the
   NTK lower group, the monotone factor vector and Eq. (10) (§3.2); mixed
   windows, the one-time switch and the attention cost (§3.3, Fig. 10,
   Appendix B). Algorithm 1's line 6 breaks the monotone constraint; Fig. 10's
   inverse frequencies are not Eq. (5)'s; attention cost is quadratic, not
   exponential. The arithmetic of the abstract, §1, §4 and Appendix B over
   Tables 1–10: the data mix, the averages, the retention, RULER, the
   real-world benchmarks and the ablations. §1's NTK MMLU drop is 4.35, not
   4.34; two printed averages are not their rows' means; from the means
   LLaMA3-8B retains 98.45%, below the abstract's 98.5%; "all evaluation
   lengths" and "all benchmarks" fail on Phi3-mini at 4k, SciFact and LCC.

45. Oka et al. — "Frequency Bands in RoPE", ICLR 2026, §5.
   The variance formula, the band-coordinate identity, and a counterexample
   to the source's global-unimodality assertion.

46. Wu, Liu and Jadbabaie — arXiv:2607.07678v1, "How Data Shapes RoPE Frequency
   Usage". Optimal contrast frequency and matched dependency dilation, §§3–4.

47. Karypis et al. — arXiv:2607.10134v1, "LeRoPE: Learnable RoPE Frequencies
   Improve Language Modeling". Log-frequency parametrization and derivative, §3.

48. Wu, Du and Peng — arXiv:2609.39929v1, "RoPE at the End of Its Rope?".
   Score realization and deterministic adjacent-response bounds, A.2/A.4/C.6.

The RoPE development formalizes the mathematical results specified in
`review/rope_context.md`; empirical accuracy, training behavior, and calibrated
probability bounds outside that coverage are not claimed as proved.

49. Bianchessi, Aguirre, Barros, Kupssinskü — arXiv:2505.22842v4, "Bayesian
   Attention Mechanism: A Probabilistic Framework for Positional Encoding and
   Context Length Extrapolation". The content–position factorization, the
   uniform, Laplace and GGD priors, locality, the learnable bias and its
   initializations, and the normalizer `Z`: §2 and Appendices A–C, H.1, K.
   Lemma 2 needs its normalizer, Lemmas 3–4 bounded content, Theorem 3 `μ = 0`;
   Theorem 4 fails for bounded content and holds relative to NoPE; Appendix K's
   `Z` is the reciprocal of the overlap it is equated with.

50. Chi, Fan, Ramadge, Rudnicky — arXiv:2205.09921v2, "KERPLE: Kernelized
   Relative Positional Embedding for Length Extrapolation". PD and CPD kernels
   and the facts that build CPD kernels, the attention and its composite kernel,
   T5's buckets, the complicated kernels, PD functions, the Gaussian-like kernels
   and the effective lengths: §3, §4, §5.3 and Appendices A.1–A.3, A.5.
   Lemma 1 (the CPD shift lemma) is false and holds on finite sets for strictly
   CPD kernels; no bias kernel `c + k̃` of §4, §5.3 or of ALiBi is PD; T5's
   printed buckets are not monotone; Fact 6 needs `ψ` CPD, not PD; Proposition
   2's first part is false; A.3's power constraint is not needed on integer
   positions; A.5's attenuation `exp(−2)` is `exp(−1/4)` under Eq. 1 at `d = 64`.

51. Chi, Fan, Rudnicky, Ramadge — arXiv:2212.10356v2, "Dissecting Transformer
   Length Extrapolation via the Lens of Receptive Field Analysis". The causal
   transformer with positional biases, ALiBi and windowed attention, large
   slopes as an implicit window, receptive fields and extrapolation, the
   cumulative normalized gradient, Sinusoidal and Rotary, Sandwich and its
   connection to KERPLE and T5, the cost of inference with a cache window,
   Sandwich's reference code, parity past the receptive field: §3, §4.2, §5,
   §6.1–6.3, Limitations, Appendices B and E.
   The theoretical receptive field is `R(w − 1) + 1`, not `wR`; `w = 100` gives
   `1189`; the printed `ERF = min {m | c_m > 0.99}` is always the first
   position; the smoothed Sandwich is KERPLE with `r₁ = 0.825 √k`, not `0.825`,
   and the exact Sandwich is bounded, so no logarithmic KERPLE bias; the
   reference code's frequencies start at `1` where Eq. (6)'s end at `1/10000`,
   which changes the attention.

52. Su et al. — arXiv:2104.09864v5, "RoFormer: Enhanced Transformer with Rotary
   Position Embedding". The 2D solution and its rotation matrix, the rotary
   matrix, its group law and orthogonality, relative scores, the sparse
   realization, the 2D derivation, the long-term decay bound, and RoPE with
   linear attention: §3.2–3.4, Eqs. (12)–(19) and (20)–(37).
   Eq. (16)'s left factor is `x_mᵀ W_qᵀ`, not `xᵀ W_q`; the derivation's
   radial solution is a choice, and with it the rotations are the only
   solutions.

53. Su — kexue.fm/archives/9708 (2023), "The road to upgrading the
   Transformer, 12: ReRoPE for infinite extrapolation?", with its code,
   github.com/bojone/rerope at acfe788. Leaky ReRoPE's and ReRoPE's relative
   positions, locality and bounded positions, the limit `k → ∞`, the two
   attention matrices and their merge, decoding with one matrix, why one RoPE
   pass does not suffice, the `log n` query scale and the reference code:
   the sections Review, Fusion, Computation and Experiments.
   ReRoPE's second matrix is `q_iᵀ R^{−w} k_j`, not `q_iᵀ R^w k_j`; the leaky
   decoding form needs `k ≥ 1`.
54. bloc97 — r/LocalLLaMA post 14lz7j5 (2023), "NTK-Aware Scaled RoPE allows
   LLaMA models to have extended (8k+) context size without any fine-tuning
   and minimal perplexity degradation", with its comment thread. Position
   Interpolation's displacement between adjacent tokens vanishes; the
   NTK-aware base `base·α^{d/(d−2)}` keeps a lower bound for every `α` and
   tells all integer positions apart; the last pair's factor `1/α`; the base
   `base·α` overshoots the slowest phase by `α^{2/d}`; the clock analogy.
55. Su — kexue.fm/archives/9675 (2023), "The road to upgrading the
   Transformer, 10: RoPE is a β-base encoding". Digits, extrapolation,
   interpolation and base conversion (`⌈∛2000⌉ = 13`), Eqs. (1)–(3), the
   range `β^{d/2}`, `λ = k^{2/(d−2)}` and the NTK-aware base, and the limits
   `λ → 1` and `k^{2/(d−2)}/k^{2/d} → 1`. The highest-frequency term is
   `n/β^0`, not `n/β`.
56. Ding et al. — arXiv:2402.13753v1, "LongRoPE: Extending LLM Context Window
   Beyond 2 Million Tokens". PI, NTK and YaRN as the rescale factors of Eq. (2)
   (§2.1); Eq. (3)'s threshold `n̂`, the count of the search space and the
   monotone constraint (§§3.1–3.2); the ratios of the progressive extension and
   PI's crowding at `512×` (§3.3); the arithmetic of Tables 8 and 10 (§4). §2.1
   prints NTK's factor as `λ = s^i`, the base `θ·s^{d/2}`, not `s^{2i/(d−2)}`;
   the thresholds `n̂ = 0` and `n̂ = 1` give the same RoPE; the search space and
   the attention cost grow polynomially, not exponentially.
57. Su — kexue.fm/archives/9728 (2023), "The road to upgrading the
   Transformer, 13: Leaky ReRoPE used in reverse". Leaky ReRoPE's step `1/k`,
   its monotone positions and the Review's rule that they stay within half the
   training length; the composition of the leaky maps, by which RoPE at
   inference is Leaky ReRoPE with `k⁻¹` in training distances; why training is
   no single RoPE pass and its two merged matrices; the rule `k = 1/(2s)`,
   `w = N/4` and the comparisons of the table: the sections Review, Inversion,
   Experiments and Summary. At `w = N/2` no finite `k` meets the Review's rule,
   the runs' inference positions break it, and "below ReRoPE/Leaky ReRoPE"
   holds like for like only.
58. Su — kexue.fm/archives/9731 (2023), "The road to upgrading the
   Transformer, 14: when HWFA meets ReRoPE", with HWFA's definition from
   kexue.fm/archives/9603. RoPE window heads are local and shift-invariant, and
   `L` stacked window layers see `(w̃ − 1)L + 1` tokens; HWFA's constraint
   `(w̃ − 1)(L − 1) + 1 = αN`; ReRoPE in the full layers keeps every inference
   position below `N`; the cost in scores against ReRoPE and against full RoPE
   layers; the comparisons of the table. At `L = 24` and `N = 512` the
   constraint allows only `w̃ ≤ 23`; the author's "even faster" needs
   `11w̃ < 5(n + 1)`; the bold `49.50` is the largest of the runs, not of the
   table.

59. Funk, Pivotto, Slilaty — arXiv:1609.05574v5, "Matrix representations of
   frame and lifted-graphic matroids correspond to gain functions". Projective
   equivalence as sequences of the source's elementary operations and
   Proposition 1.1; canonical matrices, switching and scaling; walks, cycles,
   thetas, biased graphs and the balanced cycles of a gain function; deletion,
   contraction, link minors and induced gains; connectivity and the balance
   classes of biased graphs; frame and lift matroids, matroid connectivity and
   uniform matroids;
   Propositions 2.1, 2.2, 2.3 and 4.1, Lemmas 4.2 to 4.12 and Theorem 5.1;
   which biased 2C₃, tubes and K₄ are properly unbalanced; 𝖴₂ and 𝖴₃ as
   representations of U_{2,4}; counterexamples to Conjectures 1 and 2 as
   literally stated and to Theorem 1 (i) as printed.

The latest GLM-5.3-Flash residual is instantiated from its pinned official
configuration and Transformers implementation. The older GLM-5 report
(arXiv:2602.15763v2) was downloaded for context; its scope is not silently
extended to cover the Flash model's later architectural changes.

Alongside these, `Transformer.ALM` formalizes the paraboloid-lifted lookup
used by the append-only lookup machine (Percepta, transformer-vm), and
`Transformer.Precision` proves the limits finite precision puts on attention:
finite formats blind a head past a finite length, and a sequential sum in `p`
significant bits is capped at `2^{p+1}` times its increments.
`Transformer.Landscape` holds the landscape framework and everything it rests on.
Its `SaddleAvoidance` proves that gradient descent almost never
converges to a strict saddle (arXiv:1710.07406, §5.1; arXiv:1602.04915), with
a cone estimate in place of the center-stable manifold theorem, and that under
the Łojasiewicz inequality at the critical points almost every bounded orbit
converges, to a local minimizer under the strict saddle property; for a
periodic function with nondegenerate critical points, almost every orbit
converges to a local minimizer. Its `PhaseModel` applies this to
phase energies `Σ_r (1 - cos ⟪m_r, ψ⟫)` with integer rows, such as the
relations `ψ_x - ψ_y - ψ_{x-y}` of division in a cyclic group: every limit
point up to periods is critical, and almost every orbit with a nondegenerate
limit point converges to a local minimizer; on the full division table, and
on every set of rows generating it, the zeros are the phases of the
characters up to periods, all nondegenerate; for `n ≥ 4`, below `n²/8` the
local minimizers of the full table's energy are these zeros and its other
critical points are strict saddles.
`Transformer.Rotate90` proves that the current tied SwiGLU architecture, including
all four RMSNorm calls, represents modular division at every prime with eight
active residual channels and fourteen hidden units; padding gives the documented
width-16 and width-128 models. This is an existence theorem for weights.
`Transformer.ComplexFFN` compares the complex-product feed-forward blocks of DiffGPT's
complex model with its bias-free SwiGLU control: a plain complex-product FFN with `h`
units is a SwiGLU FFN with `6h` units, none of the three complex products
approximates a single SwiGLU unit at any width, and no SwiGLU FFN approximates a
single radial or diagonal unit; one SwiGLU unit misses one plain unit. After the branch
RMS the complex-structure gap of plain and radial FFNs remains, and one SwiGLU unit still
misses one plain unit; with `eps = 0` the RMS erases the one-unit evenness and
parallelogram witnesses, and with two units each both separations hold again.
`Transformer.ComplexProjection` compares the complex-linear projections of that model
with real ones: a real projection splits uniquely into a complex-linear part and a
conjugate-linear part, the block storage `[[A,-B],[B,A]]` holds exactly the first, and a
map commuting with the complex structure errs on a real projection by at least its
conjugate-linear part, which the complex part attains; after the RMS it still misses a
normalized conjugate-linear target by its full norm.
At equal parameters the two classes are incomparable.
`Transformer.ReRoPEBlock` compares a block of that model's main arm with LeRoPE against the same
block with ReRoPE. Inside the window they are one function. Beyond it every ReRoPE block,
whatever its per-position maps, heads and frequencies, is unchanged when the far keys are
permuted, so it misses LeRoPE's π head, which sees their order, by at least
`(1 − N^{−2κ}) / (2N)` times the distance of two values, `N = w + 2`. Leaky ReRoPE forgets
nothing: with free scores its blocks are LeRoPE's, and ReRoPE misses its `kπ` head likewise.
`Transformer.LeRoPEBlock` compares that block with LeRoPE against the same block with other
positional schemes, independently of ReRoPE. A block with no positional encoding does not see
the order of its keys, whatever its per-position maps, heads and projections, while LeRoPE's π
head, one head of the main model with one channel of frequency `π`, does: every block with no
positional encoding misses it by at least `(1 − N^{−2κ}) / (2N)` times the distance of two
values, `N` keys. Conversely every such block is a LeRoPE block with the same projections on every
input, with the frequency `2π` on each plane: it is unrotated at natural distances, as RoPE's
unrotated planes are, and the learned scales reach it.
`Transformer.Phasor` writes that model's forward pass: the dynamic routers, the LeRoPE heads with
the squared-imaginary phase score, the complex-product feed-forward branches, the layers and the
tied readout. At each position it is `Transformer.MHC.PeriRMS.run`, so for any parameters after `L`
layers every stream and their mean have RMS at most `1 + 2√(2L)`, the final feature has RMS at
most one, and every attention logit at the query position `i` is at most `(5/4) ln (i + 1)`.
`Transformer.Weil` proves Stepanov's form of the Riemann hypothesis for the
curves `y² = f(x)`: over a finite field of odd order `q ≥ 64`, for `f` of odd
degree `m` with `8m ≤ q`, `|∑_x χ₂(f(x))| ≤ 7 m √q`, and the affine curve has
`q` points up to `7 m √q`; and for the curves `y^p − y = f(x)`: over a finite
field of characteristic `p` and order `q ≥ 5`, for `f` of degree `m ≥ 1` prime
to `p` and a nontrivial additive character `ψ` of `𝔽_p`,
`|∑_x ψ(Tr f(x))| ≤ 4 p m √q`. Through the `L`-function of `y^p − y = f(x)`,
this bound over every extension of `𝔽_p` gives Weil's bound
`|∑_{x ∈ 𝔽_p} ψ(f(x))| ≤ (m − 1) √p`. Through the `L`-function of `χ₂(f(x))`,
the bound for `y² = f(x)` over every extension gives Weil's bound
`|∑_{x ∈ 𝔽_p} χ₂(f(x))| ≤ (m − 1) √p` for odd `p` and odd `m`.

This top-level module re-exports all included formalizations. Per-paper
subdirectories follow the manuscript sections, splitting larger arguments
into additional modules. Some analytic proofs still use `sorry`.
The Zoology development is partial; its aggregator records the remaining
source claims and resource gaps. `Transformer.ConvexRecall` is a new
cross-paper construction with explicit changes to the original models.
`Transformer.CausalRecall` bounds the recall of causal models that read the
context through a summary formed before the query. `Transformer.MRCR` takes
the long-context benchmark MRCR-V2 apart into components and states each one.
-/

@[expose] public section
