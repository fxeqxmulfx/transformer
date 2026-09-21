/-
# Numerical precision and context length

How the format attention weights are stored in bounds the context a head can
still read.

| Module | Contents |
| --- | --- |
| `Precision.Basic` | softmax dispersion under bounded scores, the fixed-point rounding and its dead zone |
| `Precision.Nearest` | round-to-nearest in any format: the dead zone and absorption |
| `Precision.Blind` | any finite format: past some length the head outputs `c · Σ v`, independent of the scores |
| `Precision.Accumulate` | a sequential sum in `p` significant bits never exceeds `2^{p+1}` times its increments; a balanced tree does not stall |
| `Precision.IEEE` | IEEE 754 binary formats: finite, contain `0`, `M + 1` significant bits |
| `Precision.Float` | weights rounded to binary16, E5M2, E4M3 vanish past `2^{25}`, `2^{17}`, `2^{10}` times `e^D` tokens |
| `Precision.FloatSum` | the sequential online-softmax sum is wrong past `2^{M+2} e^D` keys: binary32 `2^{25}`, binary16 `2^{12}`, bfloat16 `2^9`, E4M3 `2^5`, E5M2 `2^4` |
| `Precision.Tail` | FP8 flash attention storing `exp(s - max)` in E4M3 drops every key `ln 2^{10}` below the top; the output shrinks by `1 + (n - 1) e^{-D}` (below `1/40` at `n = 2^{17}`, `D = 8`) |
| `Precision.ContextLength` | past `2^{b+1} e^D` tokens a `b`-bit head outputs `0`; below `2^{b+1} e^{-D}` it reads every token |
-/

import Transformer.Precision.Basic
import Transformer.Precision.ContextLength
import Transformer.Precision.Nearest
import Transformer.Precision.Blind
import Transformer.Precision.Accumulate
import Transformer.Precision.IEEE
import Transformer.Precision.Float
import Transformer.Precision.FloatSum
import Transformer.Precision.Tail
