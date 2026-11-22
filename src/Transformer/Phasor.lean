module

public import Transformer.Phasor.Router
public import Transformer.Phasor.Attention
public import Transformer.Phasor.FFN
public import Transformer.Phasor.Forward
public import Transformer.Phasor.LearnedStreamRead
public import Transformer.Phasor.EmbeddingInput
public import Transformer.Phasor.BoundedMemoryRead
public import Transformer.Phasor.CausalProductInput
public import Transformer.Phasor.CompactFFN
public import Transformer.Phasor.PhasePrefixRead
public import Transformer.Phasor.InputPhaseRead
public import Transformer.Phasor.AffinePhaseRead
public import Transformer.Phasor.TransportedValues
public import Transformer.Phasor.PairValues
public import Transformer.Phasor.PackedReadBuffers
public import Transformer.Phasor.PowerAddresses
public import Transformer.Phasor.SharedValues
public import Transformer.Phasor.GeometricMemory
public import Transformer.Phasor.AffineProjectionMemory
public import Transformer.Phasor.TiedGradients
public import Transformer.Phasor.PrefixValues
public import Transformer.Phasor.SuccessorValues
public import Transformer.Phasor.SuccessorDisplacement
public import Transformer.Phasor.TransportCoefficients
public import Transformer.Phasor.TransitionValues
public import Transformer.Phasor.TransitionTable
public import Transformer.Phasor.SaturatedTransitions
public import Transformer.Phasor.InclusiveTransitions
public import Transformer.Phasor.StutterTransfer
public import Transformer.Phasor.ModulatedTransitions
public import Transformer.Phasor.DisplacementGain
public import Transformer.Phasor.SuccessorAgreement
public import Transformer.Phasor.PrefixFrequency
public import Transformer.Phasor.RelationalPaths
public import Transformer.Phasor.RelationSupport
public import Transformer.Phasor.ClosedRelations
public import Transformer.Phasor.RelationRows
public import Transformer.Phasor.ClosedReadBounds
public import Transformer.Phasor.SingleWrite
public import Transformer.Phasor.ContextBackoff
public import Transformer.Phasor.Bounds
public import Transformer.Phasor.PlainRouter
public import Transformer.Phasor.OneStream
public import Transformer.Phasor.TwoCopies
public import Transformer.Phasor.StreamSum
public import Transformer.Phasor.MeanStream
public import Transformer.Phasor.ZeroSumPart
public import Transformer.Phasor.PositiveReads
public import Transformer.Phasor.PhaseHiding
public import Transformer.Phasor.HideSkip
public import Transformer.Phasor.HideRouter

/-!
# The main model of DiffGPT as a forward pass

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`,
`model.json` and `mhc.json`: the complex Peri-RMS transformer of arm `phase-score-lerope`, four mHC
streams with a phase-conjugated skip, a complex product FFN, LeRoPE with SSMax and the
squared-imaginary phase score, a tied embedding. The abstract blocks and their bounds are in
`Transformer.MHC.PeriRMS`; this folder gives the model's own router, heads and feed-forward branch
as concrete maps and plugs them into it, in exact real arithmetic.

* `Router`: the parameter-free RMS of the four streams, the three learned scales and 24 biases of
  the read, write and skip logits, and the phase coordinates of streams `1, 2, 3`, as a `Routing`;
  `Router.read`, what the sublayer reads.
* `Attention`: the LeRoPE heads with the squared-imaginary phase score as a
  `Transformer.LeRoPEBlock.block` over the streams of each position, causal, and the bound
  `(5/4) ln (i + 1)` of every logit.
* `FFN`: the complex-product feed-forward branch `W_F (W_g z ⊙_ℂ W_u z) / √2` as a `Sublayer`.
* `Forward`: layers and models, `Model.states`, `Model.logits`, and `Model.states_eq_run`: at each
  position the forward pass is `PeriRMS.run` of these sublayers, so its theorems apply to it.
* `TransportedValues` studies the NV11 variant. First-token attention reads learned embeddings
  at product-shifted addresses in the answer state. Its tied readout translates the winner for
  orthonormal embeddings and admits a sufficient margin bound for arbitrary learned embeddings.
* `PrefixValues` gives two causal scans for a shared learned table, their chunk-composition
  laws at arbitrary lengths, and their agreement with the pair residual on two-token inputs.
* `Bounds`: after `L` layers every stream and their mean have RMS at most `1 + 2√(2L)`, the final
  feature RMS at most one, every logit at most `√D ‖E v‖`, and every attention logit at most
  `(5/4) ln (i + 1)` at the query position `i`, for any parameters and any token sequence.
* `PlainRouter`, `OneStream`: routers with zero scales and phases, read biases `−ln 3` and write
  biases `0` keep four equal streams equal, so the model is then the one-stream model
  `h ← h + α R_D(raw(R_D h))`: the ordinary residual is a sub-model of mHC (R0).
* `TwoCopies`: two copies `q₀(x₀, y₀) + q₁(x₁, y₁)` of any tasks over `ℤ/p` computed exactly, in the
  one-stream model and so in this one, with real-linear projections, `D ≥ 4p + 7`, four heads,
  `F ≥ 2p` units and two layers, for every `ε ≥ 0` (T3c).
* `StreamSum`, `MeanStream`, `ZeroSumPart`: routers with zero scales and phases, read biases
  `−ln 3`, write gates summing to four and balanced skip biases make the mean of the streams the
  one-stream state, though the streams differ; a part of the streams that sums to zero is carried
  by the skip maps alone and never read (T3d).
* `PositiveReads`: with zero phases a block keeps every stream a combination of the contents
  written so far with positive coefficients, so no read misses a content exactly (T3d).
* `PhaseHiding`, `HideSkip`, `HideRouter`: with phases a router of main hides four equal streams
  from their sum while stream `0` keeps them, and a later read with unequal gates sees them (T3d).

The parameters are arbitrary: real-linear projections (the model's are the complex-linear ones, a
subclass), any routers, phase coordinates and score coefficients. The model pairs the coordinates
of a stream inside each head block; the half-split pairing of the whole stream used here is the
same after a fixed permutation of the coordinates, which arbitrary projections absorb.

What differs from the model, none of it used by the bounds: the arithmetic is exact real, not FP16
operands with FP32 accumulation, and the skip is the paper's twenty exact Sinkhorn cycles, not the
compensated log-space first cycle; the projections are not required to be complex-linear; the
width `D`, the head width `d` and the number of heads `H` are independent, the model has `D = H d`.
-/
