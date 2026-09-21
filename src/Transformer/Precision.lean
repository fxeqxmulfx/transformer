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
| `Precision.ContextLength` | past `2^{b+1} e^D` tokens a `b`-bit head outputs `0`; below `2^{b+1} e^{-D}` it reads every token |
-/

import Transformer.Precision.Basic
import Transformer.Precision.ContextLength
import Transformer.Precision.Nearest
import Transformer.Precision.Blind
import Transformer.Precision.Accumulate
