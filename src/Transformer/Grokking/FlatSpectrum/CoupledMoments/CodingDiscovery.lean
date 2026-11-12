module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Invariance
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.TwoRows
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Gauge
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Rank
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Completion
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Learner
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.CrossEntropy
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Witness

/-!
# The class coding of a categorical table is found from its training rectangles

Source: `review/flat_spectrum.md`, CM17. A square table has the labels `e⁻¹ (r i + s j)` for a
bijection `e` of its classes onto `ZMod p`, `p` prime, and phases `r`, `s`; a learner is given
training cells with their labels, and none of `e`, `r`, `s`. The training rectangles give a
linear system over `ZMod p` for a coding of the classes. If its pinned solutions have dimension
one, every nonzero one is `e` up to its affine gauge, a bijection with phases; if the graph of
the observed cells is connected as well, every training-compatible phase completion is the
table on all cells, and the learner `discover` returns it. On a table of this kind the learner
is never wrong: it returns the table or refuses, and a training set shared by two such tables
is refused. With a perfect matching among the observed cells, the convex cross-entropy program
of the coding found has one minimizer, the one-hot table. That the table has such a coding is
assumed; that a trained network finds it, or that a numerical solver reaches the minimizer, is
not claimed.
-/

@[expose] public section
