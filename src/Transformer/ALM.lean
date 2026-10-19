module

public import Transformer.ALM.Defs
public import Transformer.ALM.Duality
public import Transformer.ALM.Envelope
public import Transformer.ALM.HullErase
public import Transformer.ALM.HullPrune
public import Transformer.ALM.HullCache
public import Transformer.ALM.HullMono
public import Transformer.ALM.HullSplice
public import Transformer.ALM.TieSet
public import Transformer.ALM.GeneralPosition
public import Transformer.ALM.Query
public import Transformer.ALM.Basic
public import Transformer.ALM.Hull
public import Transformer.ALM.BinSearch
public import Transformer.ALM.HullLines
public import Transformer.ALM.HullBranch
public import Transformer.ALM.KeyOrder
public import Transformer.ALM.HullScan
public import Transformer.ALM.HullResolve
public import Transformer.ALM.HullBuild
public import Transformer.ALM.HullCover
public import Transformer.ALM.BuildOrder
public import Transformer.ALM.BuildFinger
public import Transformer.ALM.HullLift
public import Transformer.ALM.HullMark
public import Transformer.ALM.MarkedPosition
public import Transformer.ALM.HullClear
public import Transformer.ALM.HullWall
public import Transformer.ALM.HullNear
public import Transformer.ALM.HullSep
public import Transformer.ALM.HullTwin
public import Transformer.ALM.HullLower
public import Transformer.ALM.HullSpace
public import Transformer.ALM.HullCost
public import Transformer.ALM.HullValue
public import Transformer.ALM.FloatHull
public import Transformer.ALM.FloatIndex
public import Transformer.ALM.IntGrid
public import Transformer.ALM.FloatLattice
public import Transformer.ALM.FloatTie
public import Transformer.ALM.FloatWalk
public import Transformer.ALM.FloatGrid
public import Transformer.ALM.FloatResolve
public import Transformer.ALM.TieBreak
public import Transformer.ALM.LatestWindow
public import Transformer.ALM.LatestClose
public import Transformer.ALM.ScoreWall
public import Transformer.ALM.ScoreGuard
public import Transformer.ALM.CumSum
public import Transformer.ALM.DriftMargin
public import Transformer.ALM.ClearKey
public import Transformer.ALM.ClearQuery
public import Transformer.ALM.LiftCompare
public import Transformer.ALM.LiftResidual
public import Transformer.ALM.GuardSep
public import Transformer.ALM.DotError
public import Transformer.ALM.Expansion
public import Transformer.ALM.ExpansionSign
public import Transformer.ALM.ExactDot
public import Transformer.ALM.CrossFilter
public import Transformer.ALM.TreeBalance
public import Transformer.ALM.TreeQuery
public import Transformer.ALM.Dense
public import Transformer.ALM.DensePad
public import Transformer.ALM.GateGrid
public import Transformer.ALM.SparseHead
public import Transformer.ALM.ScoreGap
public import Transformer.ALM.GridWitness
public import Transformer.ALM.QueryScale
public import Transformer.ALM.Softmax
public import Transformer.ALM.Lattice
public import Transformer.ALM.ScalarInt
public import Transformer.ALM.ScalarSharp
public import Transformer.ALM.Theta
public import Transformer.ALM.VectorInt
public import Transformer.ALM.TieHyperplane
public import Transformer.ALM.TieMeasure
public import Transformer.ALM.OrthVectors
public import Transformer.ALM.LookupIndex
public import Transformer.ALM.HullIndex
public import Transformer.ALM.SoftmaxIndex
public import Transformer.ALM.SoftmaxValue
public import Transformer.ALM.SoftmaxTie
public import Transformer.ALM.SparseSoftmax
public import Transformer.ALM.SoftmaxMass
public import Transformer.ALM.SoftmaxTieInt
public import Transformer.ALM.SoftmaxLatest
public import Transformer.ALM.SoftmaxLatestMass
public import Transformer.ALM.HullHead
public import Transformer.ALM.HullHeadLatest
public import Transformer.ALM.FloatHead
public import Transformer.ALM.FloatHeadTie
public import Transformer.ALM.SAHead
public import Transformer.ALM.SAHeadValue
public import Transformer.ALM.PlanarHead

/-!
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
| `ALM.HullCache` | and that the cache the port keeps beside them needs one write |
| `ALM.HullMono` | and that those cached breakpoints increase, which is what is searched |
| `ALM.HullSplice` | and that the splice restores that order with three comparisons |
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
| `ALM.MarkedPosition` | and that it does not cost the general position either, which is what the hull head assumed |
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
| `ALM.TreeQuery` | and the container the query descends, which is not an array |
| `ALM.TreeBalance` | and its rebalancing, which moves nodes and not their order |
| `ALM.Dense` | and the layout the rest of the forward pass reads its weights from |
| `ALM.GateGrid` | and the gate between two of them, which squares the stream's range |
| `ALM.DensePad` | and that the padded rows it drops are the only ones it drops |
| `ALM.SparseHead` | and the head that skips its zeros, ties resolved as the C++ resolves them |
| `ALM.ScoreGap` | and the gap the runtime reports, which is that separation counted out |
| `ALM.GridWitness` | and the whole run's verdict, which merging the heads does not soften |
| `ALM.QueryScale` | and the scale all of that is stated at, which one division removes |
| `ALM.CumSum` | the counter the average has to divide out, and cannot |
| `ALM.DriftMargin` | and the margin that replaces the exactness it costs |
| `ALM.ClearKey` | and the cleared entry, which wins by no arithmetic at all |
| `ALM.ClearQuery` | the same guard when the query does not normalise its ordinate |
| `ALM.LiftCompare` | and the live one, whose abscissa the wall never reached |
| `ALM.LiftResidual` | the same comparison when the query misses the grid by a residual |
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
| `ALM.LookupIndex` | an exact index, and the reduction run through it |
| `ALM.HullIndex` | the machine itself as an index |
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
-/

@[expose] public section
