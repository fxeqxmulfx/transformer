module

public import Transformer.Normalization.Basic
public import Transformer.Normalization.Section2_RadialInvariant
public import Transformer.Normalization.Section2_SphereInvariant
public import Transformer.Normalization.Section3_RadialBounds
public import Transformer.Normalization.Clustering
public import Transformer.Normalization.ClusteringLine
public import Transformer.Normalization.Section3_NGPTZeroStep
public import Transformer.Normalization.ClusterSpeed
public import Transformer.Normalization.Convergence
public import Transformer.Normalization.Line
public import Transformer.Landscape.Lojasiewicz.Modulated
public import Transformer.Landscape.Lojasiewicz.LojasiewiczInnerProduct
public import Transformer.Normalization.TimeChange
public import Transformer.Normalization.Section3_LogarithmicClock
public import Transformer.Normalization.Section3_PreLimit
public import Transformer.Normalization.Radial
public import Transformer.Normalization.Rates
public import Transformer.Normalization.Symmetric
public import Transformer.Normalization.UnstableProduct
public import Transformer.Normalization.Velocities

/-!
Formalization of:
  Karagodin, Polyanskiy, Rigollet,
  "Normalization in Attention Dynamics", arXiv:2510.22026v2.

Lemma `lem: loj` of Appendix D.1 and the analytic geometry behind it hold for any
analytic energy; they live in `Transformer.Landscape.Lojasiewicz`.
-/

@[expose] public section
