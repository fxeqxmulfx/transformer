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
| `Quartet.Section3_SRCoinCube` | one independent FP4 SR coin per entry and unbiasedness with fixed non-clipping scales |
| `Quartet.Section3_SRRotation` | exact vector-level SR unbiasedness after a fixed RHT, including a scale choice valid for every input |
| `Quartet.Section3_SRConcrete` | the §3.1 `Q_SR` scales preserve that result under the necessary FP8 group-range condition |
| `Quartet.Section3_SRCeilScale` | E4M3 group scales rounded upward give exact SR unbiasedness for every input |
| `Quartet.Section3_Eden` | the EDEN correction and `MS-EDEN` |
| `Quartet.Section3_EdenFalse` | the Corollary of §3.3 fails for large `s` |
| `Quartet.Fp4Grid` | `RTN_FP4` at the top of the E2M1 grid: above `5` to `6`, between `4` and `5` to `4` |
| `Quartet.Section3_EdenPair` | the vector `e₀ + e₁/10` and its rotation, which takes two values |
| `Quartet.Section3_EdenTwoValued` | `MS-EDEN` on a two-valued rotation, in closed form |
| `Quartet.Section3_EdenCoins` | integrating one stochastic-rounding coin out of the cube of coins |
| `Quartet.Section3_EdenMean` | the mean of `MS-EDEN` over the coins on a two-valued rotation |
| `Quartet.Section3_EdenBias` | the Corollary of §3.3 fails at every dimension, inside the paper's window of `s` |
| `Quartet.Section3_EdenVectorCore` | integration and a two-coordinate RHT formula for the full-vector mean |
| `Quartet.Section3_EdenVectorBias` | the exact expected vector of the counterexample |
| `Quartet.Section3_EdenVectorNorm` | its relative bias and preserved projection |
| `Quartet.Section3_EdenUnitOne` | the second operand `e₁` is reproduced in expectation |
| `Quartet.Section3_EdenGemm` | a biased inner product with a common RHT and independent FP8 coins |
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
import Transformer.Quartet.Section3_SRCoinCube
import Transformer.Quartet.Section3_SRRotation
import Transformer.Quartet.Section3_SRConcrete
import Transformer.Quartet.Section3_SRCeilScale
import Transformer.Quartet.Section3_Eden
import Transformer.Quartet.Section3_EdenFalse
import Transformer.Quartet.Fp4Grid
import Transformer.Quartet.Section3_EdenPair
import Transformer.Quartet.Section3_EdenTwoValued
import Transformer.Quartet.Section3_EdenCoins
import Transformer.Quartet.Section3_EdenMean
import Transformer.Quartet.Section3_EdenBias
import Transformer.Quartet.Section3_EdenVectorCore
import Transformer.Quartet.Section3_EdenVectorBias
import Transformer.Quartet.Section3_EdenVectorNorm
import Transformer.Quartet.Section3_EdenUnitOne
import Transformer.Quartet.Section3_EdenGemm
import Transformer.Quartet.Section4_FourOverSix
import Transformer.Quartet.Section4_Witness
import Transformer.Quartet.Section4_Rounding
import Transformer.Quartet.Section4_Bias
import Transformer.Quartet.AppendixA_Concentration
