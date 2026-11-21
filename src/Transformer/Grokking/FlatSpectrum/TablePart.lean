module

public import Transformer.Grokking.FlatSpectrum.TablePart.AgainstM
public import Transformer.Grokking.FlatSpectrum.TablePart.Defs
public import Transformer.Grokking.FlatSpectrum.TablePart.Degree
public import Transformer.Grokking.FlatSpectrum.TablePart.DegreeHalf
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Half
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Marks
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Shuffle
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatBest
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatClass
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCount
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCycles
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatRank
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatThreshold
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatWeight
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatWorst
public import Transformer.Grokking.FlatSpectrum.TablePart.OffLine
public import Transformer.Grokking.FlatSpectrum.TablePart.RandomCount
public import Transformer.Grokking.FlatSpectrum.TablePart.RandomFlat
public import Transformer.Grokking.FlatSpectrum.TablePart.Threshold
public import Transformer.Grokking.FlatSpectrum.TablePart.Twist

/-!
# What a part of the table determines

Source: `review/flat_spectrum.md`, *What a part of the table determines*. A part `Ω` of the cells
determines a class `H` of tables when two members of `H` that agree on `Ω` are equal
(`TablePart.Defs`). Every part of `m` cells determines `H` exactly when `m > N − δ(H)`, and for a
linear class `δ(H)` is the least weight of a nonzero member (PT1, `TablePart.Threshold`). For
polynomials of degree at most `d < q`, `δ = (q − d) q^{k−1}` (PT2, `TablePart.Degree`), which at
`q = 97` puts the half of the table between degrees 48 and 49 (`TablePart.DegreeHalf`).

For the flat class `g(u) + h(v) + uᵀ M v` (`TablePart.FlatClass`), the members that vanish on `Ω`
have dimension `(c − 1) + (k l − s)`, with `c` the components of the graph of `Ω` and `s` the
dimension of the span of the twists of its closed walks (`TablePart.Twist`, `TablePart.FlatCount`).
So `Ω` determines the class exactly when its graph is connected and its twists span all matrices
(PT3, `TablePart.FlatRank`). A four-cycle gives a twist of rank one (`TablePart.FlatCycles`), and a
connected cycle with twist zero does not determine the class (`TablePart.OffLine`).

The least part that determines the flat class is the cross with `k l` more cells, of size `dim H`
(PT4, `TablePart.FlatBest`). Except at `q = 2`, `k = l = 1`, `δ = min(q^k, q^l)`
(`TablePart.FlatWeight`, `TablePart.FlatWorst`), so at `p = 97` every 9313 cells determine it and
the 9312 outside one row do not (`TablePart.FlatThreshold`).

A part with an empty row or column does not determine the class. Over the parts of `m` cells at
`k = l = 1` the empty lines average at most `2 p e^{−m/p}`, and at `p = 97` this is below 1 exactly
from `m = 511` (PT5, `TablePart.RandomCount`, `TablePart.RandomFlat`).

On a proper part with a connected graph and twists that span all matrices, the data leave one
function `g + h + uᵀ M v` of the class, and for injective `w ↦ wᵀ M` the minimum of M does not return
its kernel (PT6, `TablePart.AgainstM`). The training half of E6 at `p = 97` is such a part: the
kernel runs the SplitMix64(1) shuffle of the 9409 cells (`TablePart.E6Shuffle`), and its first 4704
entries are different cells (`TablePart.E6Marks`) that determine the class (`TablePart.E6Half`).
-/
