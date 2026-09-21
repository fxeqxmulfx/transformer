/-
# GGUF tensor formats

The tensor types llama.cpp stores model weights in, decoded from their bytes as
`ggml/src/ggml-quants.c` decodes them (llama.cpp master, commit `335b21f`).

| Module | Contents |
| --- | --- |
| `GGUF.Basic` | bytes and bit fields; IEEE binary16, bfloat16, binary32 |
| `GGUF.Nearest` | the dead zone of round-to-nearest, and the context length of softmax weights stored in a float |
| `GGUF.Legacy` | the 32-weight blocks `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`; what `Q8_0` keeps of a block |
| `GGUF.KAffine` | the K-quants with a min: `Q2_K`, `Q4_K`, `Q5_K` |
| `GGUF.KSymmetric` | the K-quants without: `Q3_K`, `Q6_K` |
| `GGUF.IQ4` | the codebook formats `IQ4_NL`, `IQ4_XS`; `IQ4_NL` has no level `0` |
| `GGUF.Grid.*` | the lattice codebooks `iq1s_grid`, `iq2xxs_grid`, `iq2xs_grid`, `iq3xxs_grid`, and `ksigns_iq2xs` |
| `GGUF.IQ2` | `IQ2_XXS`, `IQ2_XS` |
| `GGUF.IQ3` | `IQ3_XXS` |
| `GGUF.IQ1` | `IQ1_S` |

The symmetric integer formats (`Q4_0`, `Q5_0`, `Q8_0`, `Q3_K`, `Q6_K`) have
`0` among their levels, so small weights vanish: in `Q8_0` everything below
`amax/254` of its block (`exists_Q8_0`).  The affine ones (`Q4_1`, `Q5_1`,
`Q2_K`, `Q4_K`, `Q5_K`) have `0` as a level only when the min lines up with it.  The codebook formats (`IQ4_NL`, `IQ2_XXS`, `IQ2_XS`,
`IQ3_XXS`, `IQ1_S`) have no level `0`: in a block with a nonzero scale, no
weight is stored as `0` (`IQ4_NL_ne_zero`, `IQ2_XXS_ne_zero`, …).
-/

import Transformer.GGUF.Basic
import Transformer.GGUF.Nearest
import Transformer.GGUF.Legacy
import Transformer.GGUF.KAffine
import Transformer.GGUF.KSymmetric
import Transformer.GGUF.IQ4
import Transformer.GGUF.IQ2
import Transformer.GGUF.IQ3
import Transformer.GGUF.IQ1
