/-
# Append-only Lookup Machine

Formalization of the attention-head primitives of the *Append-only Lookup
Machine* (Percepta, `Percepta-Core/transformer-vm`).

| Module | Contents |
| --- | --- |
| `ALM.Defs` | the paraboloid embedding `score`, and its scalar case `sScore` |
| `ALM.Duality` | the 1D reduction the convex-hull implementation rests on |
| `ALM.Envelope` | when the hull may discard a line, and why erasing it is safe |
| `ALM.HullErase` | and that erasing it changes no answer, at any query at all |
| `ALM.Query` | why one binary search over the breakpoints finds the maximum |
| `ALM.Hull` | and why that search answers the lookup: the two halves joined |
| `ALM.BinSearch` | `lower_bound` written out, and its comparison count bounded |
| `ALM.KeyOrder` | the sorted, deduplicated keys that search is run over |
| `ALM.HullScan` | and why the tie-merge walk after it is a constant-time step |
| `ALM.HullResolve` | and what that walk hands back, in either tie-break mode |
| `ALM.HullBuild` | and what building it costs, the erase loops amortized away |
| `ALM.HullCost` | and what one query costs, and returns, search and walk together |
| `ALM.FloatHull` | and why finite precision does not change either answer |
| `ALM.FloatIndex` | and that the index itself is the one the running code answers |
| `ALM.TieBreak` | where score ties live, and what the two tie-break modes do |
| `ALM.Basic` | exact hardmax lookup in arbitrary key dimension |
| `ALM.Softmax` | softmax-to-hardmax error bounds for an abstract score gap |
| `ALM.Lattice` | the tail bounds that remove the factor `n` |
| `ALM.ScalarInt` | the length-free bound for distinct integer scalar keys |
| `ALM.ScalarSharp` | the strictly better bound the quadratic gap gives |
| `ALM.Theta` | counting lattice points by the theta function instead |
| `ALM.VectorInt` | which gives the same length-free bound in every dimension |
| `ALM.TieHyperplane` | and the tie locus in every dimension: a bisecting hyperplane |
| `ALM.OrthVectors` | one lookup decides an Orthogonal Vectors instance |
| `ALM.SAT` | satisfiability *is* an Orthogonal Vectors question |
| `ALM.LookupIndex` | an exact index, and the reduction run through it |
| `ALM.Hardness` | the conditional lower bound that reduction yields |
| `ALM.HullIndex` | the machine itself as an index, and why the barrier misses it |
| `ALM.SoftmaxIndex` | and that the softmax head returns what that index answers |
| `ALM.SoftmaxValue` | and that its output vector is the value stored at that answer |
| `ALM.SoftmaxTie` | and that on a tie it returns what the tie-break returns |
| `ALM.HullHead` | and that the tie it averages is the one the hull walk found |
| `ALM.FixedDim` | why that bound needs a growing dimension, unconditionally |
| `ALM.SETH` | and the same bound with SETH as the only conjecture |
| `ALM.Sparsification` | SETH without a density restriction, plus the sparsification |
| `ALM.SparseModel` | a model where those two hypotheses hold together |
| `ALM.Polylog` | the same barrier from SETH alone, at polylog dimension |
| `ALM.Independence` | and why that last conjecture cannot be dropped either |
| `ALM.Probe` | algorithms that can only evaluate the formula |
| `ALM.QueryModel` | where SETH stops being a conjecture and is proved |
| `ALM.OVProbe` | the same restriction on the Orthogonal Vectors side |
| `ALM.Unconditional` | and the whole chain with nothing assumed at all |
-/

import Transformer.ALM.Defs
import Transformer.ALM.Duality
import Transformer.ALM.Envelope
import Transformer.ALM.HullErase
import Transformer.ALM.Query
import Transformer.ALM.Basic
import Transformer.ALM.Hull
import Transformer.ALM.BinSearch
import Transformer.ALM.KeyOrder
import Transformer.ALM.HullScan
import Transformer.ALM.HullResolve
import Transformer.ALM.HullBuild
import Transformer.ALM.HullCost
import Transformer.ALM.FloatHull
import Transformer.ALM.FloatIndex
import Transformer.ALM.TieBreak
import Transformer.ALM.Softmax
import Transformer.ALM.Lattice
import Transformer.ALM.ScalarInt
import Transformer.ALM.ScalarSharp
import Transformer.ALM.Theta
import Transformer.ALM.VectorInt
import Transformer.ALM.TieHyperplane
import Transformer.ALM.OrthVectors
import Transformer.ALM.SAT
import Transformer.ALM.LookupIndex
import Transformer.ALM.Hardness
import Transformer.ALM.HullIndex
import Transformer.ALM.SoftmaxIndex
import Transformer.ALM.SoftmaxValue
import Transformer.ALM.SoftmaxTie
import Transformer.ALM.HullHead
import Transformer.ALM.FixedDim
import Transformer.ALM.SETH
import Transformer.ALM.Sparsification
import Transformer.ALM.SparseModel
import Transformer.ALM.Polylog
import Transformer.ALM.Independence
import Transformer.ALM.Probe
import Transformer.ALM.QueryModel
import Transformer.ALM.OVProbe
import Transformer.ALM.Unconditional
