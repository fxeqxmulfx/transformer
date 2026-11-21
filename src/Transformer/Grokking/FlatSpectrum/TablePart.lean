module

public import Transformer.Grokking.FlatSpectrum.TablePart.Defs
public import Transformer.Grokking.FlatSpectrum.TablePart.Threshold

/-!
# What a part of the table determines

Source: `review/flat_spectrum.md`, *What a part of the table determines*. A part `Ω` of the cells
determines a class `H` of tables when two members of `H` that agree on `Ω` are equal
(`TablePart.Defs`). Every part of `m` cells determines `H` exactly when `m > N − δ(H)`, and for a
linear class `δ(H)` is the least weight of a nonzero member (PT1, `TablePart.Threshold`).
-/
