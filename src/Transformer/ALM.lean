/-
# Append-only Lookup Machine

Formalization of the attention-head primitives of the *Append-only Lookup
Machine* (Percepta, `Percepta-Core/transformer-vm`).

| Module | Contents |
| --- | --- |
| `ALM.Defs` | the paraboloid embedding `score`, and its scalar case `sScore` |
| `ALM.Basic` | exact hardmax lookup in arbitrary key dimension |
| `ALM.Softmax` | softmax-to-hardmax error bounds for an abstract score gap |
| `ALM.Lattice` | the geometric-series bound that removes the factor `n` |
| `ALM.ScalarInt` | the length-free bound for distinct integer scalar keys |
-/

import Transformer.ALM.Defs
import Transformer.ALM.Basic
import Transformer.ALM.Softmax
import Transformer.ALM.Lattice
import Transformer.ALM.ScalarInt
