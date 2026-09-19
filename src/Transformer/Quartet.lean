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
| `Quartet.Walsh` | the Walsh characters on `(ℤ/2)^n` and their orthogonality |
| `Quartet.Hadamard` | the randomized Hadamard transform: orthonormal, involutive, inner-product preserving |
| `Quartet.Section3_Grids` | the E2M1 and E4M3 grids, `RTN`, and `SR` with its coin |
| `Quartet.Fp8Grid` | how far `RTN_FP8` can move a group scale: the factor `16/17` of §3.1 |
| `Quartet.Section3_NVFP4` | the two quantizers: unbiased `Q_SR`, and the clipping `Q_RTN` of `MS-EDEN` |
| `Quartet.Section3_NonClipping` | why neither quantizer asks E2M1 for a value outside `[-6, 6]` |
| `Quartet.Section3_Unbiased` | `E_ω Q_SR(x) = x`, one entry at a time |
| `Quartet.Section3_Eden` | the EDEN correction and `MS-EDEN` |
| `Quartet.Section4_FourOverSix` | the two-branch grid choice, unbiased branch by branch |
| `Quartet.Section4_Witness` | the tensor that catches the bias, and its two scales |
| `Quartet.Section4_Rounding` | what Four Over Six returns on it, coin by coin |
| `Quartet.Section4_Bias` | the mean over the coins is `65/64`, not `1`: the scheme is biased |
| `Quartet.AppendixA_Concentration` | what the concentration plot measures: the `1/B` slope, and the plateau of a bias |

Most of the paper is experimental: the pre-training loss gaps of §5 and the
kernel benchmarks of §6 are measurements, not statements, and are not
formalized.  What is formalized is the arithmetic the guarantees rest on, and
— for Appendix A — what its plot would have to show.
-/

import Transformer.Quartet.SeedSums
import Transformer.Quartet.Walsh
import Transformer.Quartet.Hadamard
import Transformer.Quartet.Section3_Grids
import Transformer.Quartet.Fp8Grid
import Transformer.Quartet.Section3_NVFP4
import Transformer.Quartet.Section3_NonClipping
import Transformer.Quartet.Section3_Unbiased
import Transformer.Quartet.Section3_Eden
import Transformer.Quartet.Section4_FourOverSix
import Transformer.Quartet.Section4_Witness
import Transformer.Quartet.Section4_Rounding
import Transformer.Quartet.Section4_Bias
import Transformer.Quartet.AppendixA_Concentration
