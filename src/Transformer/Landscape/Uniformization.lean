module

public import Transformer.Landscape.Uniformization.Induction
public import Transformer.Landscape.Uniformization.BasicExamples
public import Transformer.Landscape.Uniformization.NewtonExamples
public import Transformer.Landscape.Uniformization.Examples

/-!
# Local uniformization of real analytic functions

Every real analytic function on `ℝⁿ` that does not vanish identically near a point has
finitely many analytic charts with compact domains, whose images cover a neighbourhood
of the point, in each of which it is an analytic unit times a coordinate monomial at
every point (`Landscape.Uniformization.hasMonomialCharts_of_analyticAt`). The proof is by
induction on the dimension and on the order in the last variable, through Weierstrass
preparation, ramified Newton charts, and the principalization of `(y, t^p)` by monomial
maps. This is the geometric input of the gradient inequality of Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. The `*Examples` modules satisfy the hypotheses of
every lemma on the way.
-/

@[expose] public section
