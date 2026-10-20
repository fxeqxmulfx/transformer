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
public import Transformer.AttentionResiduals
public import Transformer.DeepSeekV41
public import Transformer.RoPE

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
   of Large Language Models". Bandwise interpolation and NTK scaling, §3/A.2.

43. Men et al. — arXiv:2405.14591v1, "Base of RoPE Bounds Context Length".
   Expected-score discrimination with explicit corrected moments, §4/Appendix A.

44. Shang et al. — arXiv:2502.20082v1, "LongRoPE2: Near-Lossless LLM Context
   Window Scaling". Coordinatewise phase constraints and mixed windows, §2/§3.3.

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

This top-level module re-exports all included formalizations. Per-paper
subdirectories follow the manuscript sections, splitting larger arguments
into additional modules. Some analytic proofs still use `sorry`.
The Zoology development is partial; its aggregator records the remaining
source claims and resource gaps. `Transformer.ConvexRecall` is a new
cross-paper construction with explicit changes to the original models.
-/

@[expose] public section
