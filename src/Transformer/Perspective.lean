/-
# A mathematical perspective on Transformers

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5.

One module per section of the manuscript (`Section1_IPS … Section9_Approximation`),
followed by the four appendices.  Everything lives in the namespace
`Transformer.Perspective`.  `MinCurve` carries no section of its own: it is
the one-sided calculus behind step 1 of §6.1, `PartitionGradient` is the
differentiation of `Z_{β,μ}` under the integral sign that §3.3 rests on, and
`RussianTrick` is the linear-algebra identity `e:russiantrick` of Appendix A
with the rank-two construction that proves it.  `InnerAsymptotics` and
`PeanoTaylor` are the two ledgers of real analysis Appendix B's second-order
expansion runs on; neither mentions the survey.
-/

import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap
import Transformer.Perspective.Section2_EnergyKernel
import Transformer.Perspective.Section2_EnergyMax
import Transformer.Perspective.PartitionGradient
import Transformer.Perspective.Section2_GradientFlow
import Transformer.Perspective.Section2_ParticleFlow
import Transformer.Perspective.SphereInvariant
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.UniformAtomless
import Transformer.Perspective.Section5_InvariantMeasure
import Transformer.Perspective.Beta0Field
import Transformer.Perspective.Softmax
import Transformer.Perspective.Beta0Gronwall
import Transformer.Perspective.SAField
import Transformer.Perspective.SALipschitz
import Transformer.Perspective.Section3_Gronwall
import Transformer.Perspective.Section4_LargeBeta
import Transformer.Perspective.MinCurve
import Transformer.Perspective.Section5_HighD
import Transformer.Perspective.Section5_ConeCollapse
import Transformer.Perspective.Section5_Hemisphere
import Transformer.Perspective.Section5_HemisphereCone
import Transformer.Perspective.Section5_Exceptional
import Transformer.Perspective.Section5_HighDCurve
import Transformer.Perspective.Section5_Vanishing
import Transformer.Perspective.Section6_Circle
import Transformer.Perspective.Section7_BBGKY
import Transformer.Perspective.Section8_General
import Transformer.Perspective.Section8_CohnKumar
import Transformer.Perspective.Section9_Approximation
import Transformer.Perspective.RussianTrick
import Transformer.Perspective.RussianPairs
import Transformer.Perspective.AppendixA_Beta0
import Transformer.Perspective.AppendixA_Hessian
import Transformer.Perspective.AppendixA_Rotation
import Transformer.Perspective.AppendixA_Saddle
import Transformer.Perspective.AppendixB_BetaInterval
import Transformer.Perspective.AppendixB_Taylor
import Transformer.Perspective.StrictSaddle
import Transformer.Perspective.InnerAsymptotics
import Transformer.Perspective.PeanoTaylor
import Transformer.Perspective.AppendixB_EBeta
import Transformer.Perspective.AppendixB_Expansion
import Transformer.Perspective.AppendixB_HessBeta
import Transformer.Perspective.AppendixB_HighD
import Transformer.Perspective.AppendixB_Intrinsic
import Transformer.Perspective.AppendixB_ClaimYury
import Transformer.Perspective.AppendixB_MetricGrad
import Transformer.Perspective.AppendixB_MetricHess
import Transformer.Perspective.DoubleSum
import Transformer.Perspective.AppendixC_BetaTiny
import Transformer.Perspective.Gronwall
import Transformer.Perspective.AppendixD_Stability
import Transformer.Perspective.AppendixD_PhaseTransition
import Transformer.Perspective.AppendixD_Alpha
import Transformer.Perspective.AppendixD_AlphaDeriv
import Transformer.Perspective.AppendixD_Product
import Transformer.Perspective.AppendixD_Ybeta
import Transformer.Perspective.AppendixD_YbetaUSA
import Transformer.Perspective.AppendixD_Assembly
