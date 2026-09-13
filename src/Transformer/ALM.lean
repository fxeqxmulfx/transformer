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
| `ALM.OrthVectors` | one lookup decides an Orthogonal Vectors instance |
| `ALM.SAT` | satisfiability *is* an Orthogonal Vectors question |
| `ALM.LookupIndex` | an exact index, and the reduction run through it |
| `ALM.Hardness` | the conditional lower bound that reduction yields |
| `ALM.SETH` | and the same bound with SETH as the only conjecture |
| `ALM.Sparsification` | SETH without a density restriction, plus the sparsification |
| `ALM.SparseModel` | a model where those two hypotheses hold together |
| `ALM.Polylog` | the same barrier from SETH alone, at polylog dimension |
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
import Transformer.ALM.OrthVectors
import Transformer.ALM.SAT
import Transformer.ALM.LookupIndex
import Transformer.ALM.Hardness
import Transformer.ALM.SETH
import Transformer.ALM.Sparsification
import Transformer.ALM.SparseModel
import Transformer.ALM.Polylog
