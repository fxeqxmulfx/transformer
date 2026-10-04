/-
Formalization of thirty-two papers on the mathematics of Transformers:

1. Weiss, Goldberg, Yahav — arXiv:2106.06981v2
   "Thinking Like Transformers".

2. Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio, Nakkiran —
   arXiv:2310.16028v1
   "What Algorithms can Transformers Learn?  A Study in Length
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

Alongside these, `Transformer.ALM` formalizes the paraboloid-lifted lookup
used by the append-only lookup machine (Percepta, transformer-vm), and
`Transformer.Precision` proves the limits finite precision puts on attention:
finite formats blind a head past a finite length, and a sequential sum in `p`
significant bits is capped at `2^{p+1}` times its increments.

This top-level module re-exports all included formalizations. Per-paper
subdirectories follow the manuscript sections, splitting larger arguments
into additional modules. Some analytic proofs still use `sorry`.
The Zoology development is partial; its aggregator records the remaining
source claims and resource gaps. `Transformer.ConvexRecall` is a new
cross-paper construction with explicit changes to the original models.
-/

import Transformer.Basic
import Transformer.Wasserstein
import Transformer.GlobalFlow
import Transformer.ALM
import Transformer.RASP
import Transformer.RASPL
import Transformer.CRASP
import Transformer.Perspective
import Transformer.Metastability
import Transformer.Interpolation
import Transformer.Causal
import Transformer.Normalization
import Transformer.MeanField
import Transformer.FrankWolfe
import Transformer.Kinetic
import Transformer.Homogenized
import Transformer.Perceptron
import Transformer.Clusters
import Transformer.Modes
import Transformer.AMSGrad
import Transformer.AMSGradW
import Transformer.AdamBeyond
import Transformer.XSA
import Transformer.Quartet
import Transformer.GPTMini
import Transformer.GPTMini.Sparsemax
import Transformer.Precision
import Transformer.Convexifying
import Transformer.Zoology
import Transformer.ConvexRecall
import Transformer.Muon
import Transformer.DASH
import Transformer.AdaFisher
import Transformer.Optimization
import Transformer.ICEoT
import Transformer.OptimizerBenchmark
import Transformer.Magma
import Transformer.Imbalance
import Transformer.BatchSize
import Transformer.DoubleDescent
import Transformer.Memorization
import Transformer.MagnitudeDirection
import Transformer.RecurrentGradients
