/-
# Append-only Lookup Machine

Formalization of the attention-head primitives of the *Append-only Lookup
Machine* (Percepta, `Percepta-Core/transformer-vm`).

| Module | Contents |
| --- | --- |
| `ALM.Defs` | the paraboloid embedding `score`, and its scalar case `sScore` |
| `ALM.Duality` | the 1D reduction the convex-hull implementation rests on |
| `ALM.Envelope` | when the hull may discard a line, and why erasing it is safe |
| `ALM.Query` | why one binary search over the breakpoints finds the maximum |
| `ALM.TieBreak` | where score ties live, and what the two tie-break modes do |
| `ALM.Basic` | exact hardmax lookup in arbitrary key dimension |
| `ALM.Softmax` | softmax-to-hardmax error bounds for an abstract score gap |
| `ALM.Lattice` | the tail bounds that remove the factor `n` |
| `ALM.ScalarInt` | the length-free bound for distinct integer scalar keys |
| `ALM.ScalarSharp` | the strictly better bound the quadratic gap gives |
-/

import Transformer.ALM.Defs
import Transformer.ALM.Duality
import Transformer.ALM.Envelope
import Transformer.ALM.Query
import Transformer.ALM.Basic
import Transformer.ALM.TieBreak
import Transformer.ALM.Softmax
import Transformer.ALM.Lattice
import Transformer.ALM.ScalarInt
import Transformer.ALM.ScalarSharp
