module

public import Transformer.Grokking.FlatSpectrum.TablePart.Defs
public import Transformer.Grokking.FlatSpectrum.TablePart.Degree
public import Transformer.Grokking.FlatSpectrum.TablePart.DegreeHalf
public import Transformer.Grokking.FlatSpectrum.TablePart.Threshold

/-!
# What a part of the table determines

Source: `review/flat_spectrum.md`, *What a part of the table determines*. A part `Ω` of the cells
determines a class `H` of tables when two members of `H` that agree on `Ω` are equal
(`TablePart.Defs`). Every part of `m` cells determines `H` exactly when `m > N − δ(H)`, and for a
linear class `δ(H)` is the least weight of a nonzero member (PT1, `TablePart.Threshold`). For
polynomials of degree at most `d < q`, `δ = (q − d) q^{k−1}` (PT2, `TablePart.Degree`), which at
`q = 97` puts the half of the table between degrees 48 and 49 (`TablePart.DegreeHalf`).
-/
