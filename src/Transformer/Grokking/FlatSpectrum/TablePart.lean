module

public import Transformer.Grokking.FlatSpectrum.TablePart.Defs
public import Transformer.Grokking.FlatSpectrum.TablePart.Degree
public import Transformer.Grokking.FlatSpectrum.TablePart.DegreeHalf
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatClass
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCount
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCycles
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatRank
public import Transformer.Grokking.FlatSpectrum.TablePart.OffLine
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
-/
