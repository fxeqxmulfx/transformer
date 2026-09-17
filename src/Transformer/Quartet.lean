/-
# Quartet II: accurate LLM pre-training in NVFP4

Formalization of Panferov, Schultheis, Tabesh, Alistarh — arXiv:2601.22813v2,
"Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient
Estimation" (ICML 2026).

NVFP4 is the 4-bit micro-scaling format of the NVIDIA Blackwell GPUs: one E2M1
number per entry, one E4M3 scale per `16` entries, and one FP32 scale per
tensor.  Quantized pre-training needs the *backward* pass to be unbiased, and
every scheme before this paper buys that with element-wise stochastic rounding,
at a cost of roughly `2.5×` in mean-square error.  The paper instead carries
the bias correction of EDEN — a randomized rotation, then a per-group rescaling
`S` — over to micro-scaling, by folding `S` into the E4M3 group scales through
stochastic rounding.  That is `MS-EDEN`, and `Quartet II` is the linear-layer
scheme built on it.

| Module | Contents |
| --- | --- |
| `Quartet.Section3_Grids` | the E2M1 and E4M3 grids, `RTN`, and `SR` with its coin |
| `Quartet.Section3_NVFP4` | the two quantizers: unbiased `Q_SR`, and the clipping `Q_RTN` of `MS-EDEN` |
| `Quartet.Section3_Eden` | the randomized Hadamard transform, the EDEN correction, and `MS-EDEN` |
| `Quartet.Section4_FourOverSix` | the two-branch grid choice, unbiased branch by branch and biased together |

Most of the paper is experimental: the pre-training loss gaps of §5, the
kernel benchmarks of §6 and the concentration plots of Appendix A are
measurements, not statements, and are not formalized.  What is formalized is
the arithmetic the guarantees rest on.
-/

import Transformer.Quartet.Section3_Grids
import Transformer.Quartet.Section3_NVFP4
import Transformer.Quartet.Section3_Eden
import Transformer.Quartet.Section4_FourOverSix
