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
| `ALM.HullPrune` | and that a whole build of them changes none either |
| `ALM.TieSet` | and the winner it can drop while keeping the value |
| `ALM.GeneralPosition` | the hypothesis that stops it, which the lift happens to meet |
| `ALM.Query` | why one binary search over the breakpoints finds the maximum |
| `ALM.Hull` | and why that search answers the lookup: the two halves joined |
| `ALM.BinSearch` | `lower_bound` written out, and its comparison count bounded |
| `ALM.HullLines` | the same search over the arbitrary lines the code really stores |
| `ALM.HullBranch` | and the three branches of `query`, one of which can never tie |
| `ALM.KeyOrder` | the sorted, deduplicated keys that search is run over |
| `ALM.HullScan` | and why the tie-merge walk after it is a constant-time step |
| `ALM.HullResolve` | and what that walk hands back, in either tie-break mode |
| `ALM.HullBuild` | and what building it costs, the erase loops amortized away |
| `ALM.HullCover` | and that it holds one line per key, so the search covers them all |
| `ALM.BuildOrder` | and that no arrival order can cost it more than one erase per key |
| `ALM.BuildFinger` | and that this port pays half of that, and on sorted keys no logarithm |
| `ALM.HullLift` | and that on the paraboloid no erase rule whatsoever can fire |
| `ALM.HullMark` | and that the recency term the compiler adds does not revive one |
| `ALM.HullClear` | while the clear marker does, which is how a cleared entry leaves |
| `ALM.HullWall` | and that past the wall the recency term is not there to begin with |
| `ALM.HullNear` | so a live head answers with the key nearest the query, and only that |
| `ALM.HullSep` | provided its keys clear two thirds of a step, which is where that breaks |
| `ALM.HullTwin` | and where the shipped model breaks it, it is one key rounded twice |
| `ALM.HullLower` | while the other half of the head keeps two lines at any length |
| `ALM.HullSpace` | and no container at all keeps fewer than one line per key |
| `ALM.HullCost` | and what one query costs, and returns, search and walk together |
| `ALM.HullValue` | and what it hands back, at any maximizer and at the search's own |
| `ALM.FloatHull` | and why finite precision does not change either answer |
| `ALM.FloatIndex` | and that the index itself is the one the running code answers |
| `ALM.IntGrid` | integer keys on a grid, and their breakpoints on a half-grid |
| `ALM.FloatLattice` | and that on integer data no separation is needed at all |
| `ALM.FloatTie` | and that the `==` the merge loops branch on is the real tie |
| `ALM.FloatWalk` | and the whole query, search and walk, in the arithmetic that runs |
| `ALM.FloatGrid` | and that the exactness that walk needs is a property of the grid |
| `ALM.FloatResolve` | and the value `resolve` writes out, in the arithmetic that runs |
| `ALM.TieBreak` | where score ties live, and what the two tie-break modes do |
| `ALM.LatestWindow` | and what orders the writes the released weights never tie |
| `ALM.LatestClose` | and how long that lasts, past which rounding decides instead |
| `ALM.ScoreWall` | and that the wall is the stored coordinate, not the arithmetic |
| `ALM.ScoreGuard` | and the query-time test for it, which the scale nearly cancels out of |
| `ALM.GuardSep` | and what passing that test buys: the separation the comparison needs |
| `ALM.DotError` | except that the score costs three roundings and they do not cancel |
| `ALM.Expansion` | and the exact sum the arbiter accumulates when they cannot decide |
| `ALM.ExpansionSign` | and the one component it reads that sum's sign off |
| `ALM.ExactDot` | and the eight terms that make that sign the comparison's |
| `ALM.CrossFilter` | and the breakpoint test, whose fast path answers the same |
| `ALM.ScoreGap` | and the gap the runtime reports, which is that separation counted out |
| `ALM.GridWitness` | and the whole run's verdict, which merging the heads does not soften |
| `ALM.QueryScale` | and the scale all of that is stated at, which one division removes |
| `ALM.CumSum` | the counter the average has to divide out, and cannot |
| `ALM.DriftMargin` | and the margin that replaces the exactness it costs |
| `ALM.ClearKey` | and the cleared entry, which wins by no arithmetic at all |
| `ALM.Basic` | exact hardmax lookup in arbitrary key dimension |
| `ALM.Softmax` | softmax-to-hardmax error bounds for an abstract score gap |
| `ALM.Lattice` | the tail bounds that remove the factor `n` |
| `ALM.ScalarInt` | the length-free bound for distinct integer scalar keys |
| `ALM.ScalarSharp` | the strictly better bound the quadratic gap gives |
| `ALM.Theta` | counting lattice points by the theta function instead |
| `ALM.VectorInt` | which gives the same length-free bound in every dimension |
| `ALM.TieHyperplane` | and the tie locus in every dimension: a bisecting hyperplane |
| `ALM.TieMeasure` | and that that locus is null, so the merge path is almost never taken |
| `ALM.OrthVectors` | one lookup decides an Orthogonal Vectors instance |
| `ALM.SAT` | satisfiability *is* an Orthogonal Vectors question |
| `ALM.LookupIndex` | an exact index, and the reduction run through it |
| `ALM.Hardness` | the conditional lower bound that reduction yields |
| `ALM.HullIndex` | the machine itself as an index, and why the barrier misses it |
| `ALM.SoftmaxIndex` | and that the softmax head returns what that index answers |
| `ALM.SoftmaxValue` | and that its output vector is the value stored at that answer |
| `ALM.SoftmaxTie` | and that on a tie it returns what the tie-break returns |
| `ALM.SparseSoftmax` | and what truncating it to the retained keys costs, exactly |
| `ALM.SoftmaxMass` | and where the mass that bound assumes actually comes from |
| `ALM.SoftmaxTieInt` | and that on the stored integer keys it needs no gap at all |
| `ALM.SoftmaxLatest` | and how far it is from the other tie-break mode, exactly |
| `ALM.SoftmaxLatestMass` | and that that mode is priced by the keys too, with no gap assumed |
| `ALM.HullHead` | and that the tie it averages is the one the hull walk found |
| `ALM.HullHeadLatest` | and what the other tie-break mode costs it, exactly |
| `ALM.FloatHead` | and the value the head returns at the line that code lands on |
| `ALM.FloatHeadTie` | and at the tie that code merges, on the arithmetic it merges in |
| `ALM.SAHead` | and that the head is an ordinary attention head, at two projections |
| `ALM.SAHeadValue` | and so that every bound above is a bound on standard attention |
| `ALM.PlanarHead` | and that one hull query answers any planar head, not only the lift |
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
import Transformer.ALM.HullPrune
import Transformer.ALM.TieSet
import Transformer.ALM.GeneralPosition
import Transformer.ALM.Query
import Transformer.ALM.Basic
import Transformer.ALM.Hull
import Transformer.ALM.BinSearch
import Transformer.ALM.HullLines
import Transformer.ALM.HullBranch
import Transformer.ALM.KeyOrder
import Transformer.ALM.HullScan
import Transformer.ALM.HullResolve
import Transformer.ALM.HullBuild
import Transformer.ALM.HullCover
import Transformer.ALM.BuildOrder
import Transformer.ALM.BuildFinger
import Transformer.ALM.HullLift
import Transformer.ALM.HullMark
import Transformer.ALM.HullClear
import Transformer.ALM.HullWall
import Transformer.ALM.HullNear
import Transformer.ALM.HullSep
import Transformer.ALM.HullTwin
import Transformer.ALM.HullLower
import Transformer.ALM.HullSpace
import Transformer.ALM.HullCost
import Transformer.ALM.HullValue
import Transformer.ALM.FloatHull
import Transformer.ALM.FloatIndex
import Transformer.ALM.IntGrid
import Transformer.ALM.FloatLattice
import Transformer.ALM.FloatTie
import Transformer.ALM.FloatWalk
import Transformer.ALM.FloatGrid
import Transformer.ALM.FloatResolve
import Transformer.ALM.TieBreak
import Transformer.ALM.LatestWindow
import Transformer.ALM.LatestClose
import Transformer.ALM.ScoreWall
import Transformer.ALM.ScoreGuard
import Transformer.ALM.CumSum
import Transformer.ALM.DriftMargin
import Transformer.ALM.ClearKey
import Transformer.ALM.GuardSep
import Transformer.ALM.DotError
import Transformer.ALM.Expansion
import Transformer.ALM.ExpansionSign
import Transformer.ALM.ExactDot
import Transformer.ALM.CrossFilter
import Transformer.ALM.ScoreGap
import Transformer.ALM.GridWitness
import Transformer.ALM.QueryScale
import Transformer.ALM.Softmax
import Transformer.ALM.Lattice
import Transformer.ALM.ScalarInt
import Transformer.ALM.ScalarSharp
import Transformer.ALM.Theta
import Transformer.ALM.VectorInt
import Transformer.ALM.TieHyperplane
import Transformer.ALM.TieMeasure
import Transformer.ALM.OrthVectors
import Transformer.ALM.SAT
import Transformer.ALM.LookupIndex
import Transformer.ALM.Hardness
import Transformer.ALM.HullIndex
import Transformer.ALM.SoftmaxIndex
import Transformer.ALM.SoftmaxValue
import Transformer.ALM.SoftmaxTie
import Transformer.ALM.SparseSoftmax
import Transformer.ALM.SoftmaxMass
import Transformer.ALM.SoftmaxTieInt
import Transformer.ALM.SoftmaxLatest
import Transformer.ALM.SoftmaxLatestMass
import Transformer.ALM.HullHead
import Transformer.ALM.HullHeadLatest
import Transformer.ALM.FloatHead
import Transformer.ALM.FloatHeadTie
import Transformer.ALM.SAHead
import Transformer.ALM.SAHeadValue
import Transformer.ALM.PlanarHead
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
