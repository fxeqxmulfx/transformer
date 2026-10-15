/-
Formalization of:
  Karagodin, Polyanskiy, Rigollet,
  "Clustering in Causal Attention Masking", arXiv:2411.04990v2.
-/

import Transformer.Causal.Basic
import Transformer.Causal.SingleToken
import Transformer.Causal.MainTheorem
import Transformer.Causal.Interaction
import Transformer.Causal.InteractionBounds
import Transformer.Causal.InteractionNumerics
import Transformer.Causal.InteractionPeak
import Transformer.Causal.InteractionWindow
import Transformer.Causal.InteractionRemark
import Transformer.Causal.Packing
import Transformer.Causal.Metastability
import Transformer.Causal.Section5_CircleGeometry
import Transformer.Causal.Section5_InteractionDecay
import Transformer.Causal.Section5_CausalWeights
import Transformer.Causal.Section5_ClosestDrift
import Transformer.Causal.Section5_GapBarrier
import Transformer.Causal.Section5_StrongGap
import Transformer.Causal.Section5_StrongCenterWitness
import Transformer.Causal.FixedCenters
import Transformer.Causal.SequentialFlow
import Transformer.Causal.AppendixA_Cusp
import Transformer.Causal.AppendixA_CuspEnergy
import Transformer.Causal.AppendixA_CuspCritical
import Transformer.Causal.AppendixA_CuspJacobian
import Transformer.Causal.AppendixA_CuspHyperbolicity
import Transformer.Causal.AppendixA_CuspClock
import Transformer.Causal.ParkingCount
import Transformer.Causal.CapCone
import Transformer.Causal.CapPolar
import Transformer.Causal.CapHalfPlane
import Transformer.Causal.CapSpace
import Transformer.Causal.CapMass
import Transformer.Causal.UniformAtoms
