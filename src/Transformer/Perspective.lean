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
expansion runs on; neither mentions the survey.  `MaxCurve`, `ConeChart`,
`ConeWidth` and `ConeLimit` are the proof of cone collapse (§6.1) for any flow
`ẋ_i = Proj_{x_i}(Σ_j a_ij x_j)` with weights in `[m, M]`, `m > 0`, through
the chart `x ↦ x / ⟨x, w⟩` of a hemisphere, and `ConeFields` puts the
attention dynamics in that form.  `SphereHyperplane` and `UniformHemisphere`
are the footnote to `thm: d.infty`: a rotation-invariant law does not charge
great spheres, so `n ≤ d` uniform points are almost surely linearly
independent and lie in an open hemisphere.  `SphereMoments` and
`PositiveDefinite` make the kernel `e^{β x·y}` positive definite on the
moments of a measure: a probability measure on the sphere is determined by its
moments, hence by its partition function
`Z_{β,μ}`, `β > 0`.  `InjectedFlow`, `InjectedSpread`, `InjectedAttention` and
`InjectedConsensus` are not in the survey: the flow
`ẋ_i = Proj_{x_i}(Σ_j a_ij x_j + z_i)`, with constant vectors `z_i` mixed back
into the drive, does not collapse when two of the `z_i` are linearly
independent, uniformly in weights of bounded row sums — for softmax attention,
in `β`, `Q` and `K` — while at `z = 0` softmax attention with constant `Q`,
`K` collapses from any open hemisphere (§6.1).  `DrivenFlow` and
`DrivenSpread` let the drive be anything and ask for it on one window of time
only: along `ẋ_i = Proj_{x_i}(v_i(t))`, if two drive differences `v_k - v_l`,
`v_m - v_p` stay near two independent vectors while the tokens are close, the
tokens do not stay close through any window of positive length.
`BlockDrive` and `BlockSpread` apply it to a residual block — several heads
with any values, a feed-forward term Lipschitz on the sphere, an injection
`z_i(t)` — and to a stack of different blocks: if two differences of
injections `z_k - z_l`, `z_m - z_p` are independent in every block, one `δ`
works for the whole stack.  `BlockConsensus` shows that differences are what
counts: when all `z_i - z_{i₀}` are collinear, a block with a rank-one value
holds every token at one point, so two independent injections alone — the
hypothesis for `V = I_d` — do not rule collapse out.  `RawStream` and
`RawGrowth` drop the sphere: behind the RMS norm of a pre-norm block the
directions follow the sphere model with the drive of each token divided by its
length, and for `mix[0] = 1 + c > 1` the stream outgrows its drive, so every
direction stays within `2 M / (c ‖x_i(t₀)‖)` of where it was at `t₀` and
`blockDrive_spread` fails before the norm.  `BlockMLP` and `BlockMLPSpread`
take in the feed-forward term of parameter-golf, `W₂ σ(W₁ x)²` with a leaky
ReLU `σ`: on the sphere it is bounded and Lipschitz with constants read off
the weights, so `blockDrive_spread` holds for a block with that term, with
nothing assumed of it but bounds on `‖W₁‖` and `‖W₂‖`.  `XSAProj`, `BlockXSA`,
`BlockXSASpread` and `BlockXSARecord` make the heads exclusive (`eq:xsa`,
arXiv:2603.09078v1): a head with its own value projected away is `O(ε)` by
itself on a cluster, not only in a difference, so the rows of a head no longer
need one common sum and a gate per head and token — which the projection turns
into a factor on the row — needs only to be bounded.  That is the block of the
record, XSA on every layer under a sigmoid gate, with softmax rows, values and
output maps carrying `attn_scale` and the grouping of queries, and the
leaky-ReLU² feed-forward term: it does not collapse either
(`gatedXSA_block_spread`).  `RawStack` leaves continuous time behind: a stack
`x_{k+1,i} = λ_{k,i} x_{k,i} + g_{k,i}` of blocks that are all different, with
the scalar residual gains of modded-nanogpt, is its gauge `∏_{j<k} λ_{j,i}`
times the same stack with the gains divided out, so the directions never see
them (`normalize_rawStack`), and in `RawStackFrozen` gains above `1 + c` freeze
every direction within `2 M / (c ‖x_{0,i}‖)` of its start, at every depth and
whatever the blocks compute (`rawStack_frozen`) — a bound that the record's own
gains make empty (`lt_rawStack_frozen_bound`), which is what the exactness of
the factorisation is worth.  `RawStackSkip` adds the one update left over, the
skip that adds an earlier state of the stream instead of a bounded output: it is
gauge-covariant too (`ungauged_rec_skip`), with its gate divided by the gain
accumulated over the depth it jumps, and `RawStackSkipDamp` reads that division
from both sides — exponential in the depth skipped (`abs_skipWeight_le`), but
bounded below by the same exponential (`le_abs_skipWeight`), so skips at a fixed
distance keep one weight at every depth and `rawStack_frozen` does not reach
them; at the record's own gains more than three quarters of the gate survives
(`lt_inv_pow_record`).
-/

import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap
import Transformer.Perspective.Section2_EnergyKernel
import Transformer.Perspective.Section2_EnergyMoments
import Transformer.Perspective.Section2_EnergySeries
import Transformer.Perspective.SphereMoments
import Transformer.Perspective.PositiveDefinite
import Transformer.Perspective.Section2_EnergyDerivative
import Transformer.Perspective.Section2_Dissipation
import Transformer.Perspective.Section2_EnergyMax
import Transformer.Perspective.Section2_EnergyConvex
import Transformer.Perspective.Section2_EnergyMin
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
import Transformer.Perspective.MaxCurve
import Transformer.Perspective.ConeChart
import Transformer.Perspective.ConeWidth
import Transformer.Perspective.ConeLimit
import Transformer.Perspective.ConeFields
import Transformer.Perspective.Section5_HighD
import Transformer.Perspective.SphereHyperplane
import Transformer.Perspective.UniformHemisphere
import Transformer.Perspective.Section5_ExpRate
import Transformer.Perspective.InjectedFlow
import Transformer.Perspective.InjectedSpread
import Transformer.Perspective.InjectedAttention
import Transformer.Perspective.InjectedConsensus
import Transformer.Perspective.DrivenFlow
import Transformer.Perspective.DrivenSpread
import Transformer.Perspective.BlockDrive
import Transformer.Perspective.BlockSpread
import Transformer.Perspective.BlockConsensus
import Transformer.Perspective.RawStream
import Transformer.Perspective.RawGrowth
import Transformer.Perspective.RawStack
import Transformer.Perspective.RawStackFrozen
import Transformer.Perspective.RawStackSkip
import Transformer.Perspective.RawStackSkipDamp
import Transformer.Perspective.BlockMLP
import Transformer.Perspective.BlockMLPSpread
import Transformer.Perspective.XSAProj
import Transformer.Perspective.BlockXSA
import Transformer.Perspective.BlockXSASpread
import Transformer.Perspective.BlockXSARecord
import Transformer.Perspective.Section5_ConeCollapse
import Transformer.Perspective.Section5_Hemisphere
import Transformer.Perspective.Section5_HemisphereCone
import Transformer.Perspective.Section5_HemisphereRate
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
import Transformer.Perspective.AppendixD_DotAlpha
import Transformer.Perspective.AppendixD_AlphaDeriv
import Transformer.Perspective.AppendixD_Product
import Transformer.Perspective.AppendixD_YbetaInvariance
import Transformer.Perspective.AppendixD_Ybeta
import Transformer.Perspective.AppendixD_YbetaUSA
import Transformer.Perspective.AppendixD_Assembly
