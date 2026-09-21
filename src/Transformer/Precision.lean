/-
# Numerical precision and context length

How the format attention weights are stored in bounds the context a head can
still read.

| Module | Contents |
| --- | --- |
| `Precision.Basic` | softmax dispersion under bounded scores, the fixed-point rounding and its dead zone |
| `Precision.ContextLength` | past `2^{b+1} e^D` tokens a `b`-bit head outputs `0`; below `2^{b+1} e^{-D}` it reads every token |
-/

import Transformer.Precision.Basic
import Transformer.Precision.ContextLength
