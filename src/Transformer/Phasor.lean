module

public import Transformer.Phasor.Router
public import Transformer.Phasor.Attention
public import Transformer.Phasor.FFN
public import Transformer.Phasor.Forward
public import Transformer.Phasor.Bounds
public import Transformer.Phasor.PlainRouter
public import Transformer.Phasor.OneStream

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
* `Bounds`: after `L` layers every stream and their mean have RMS at most `1 + 2√(2L)`, the final
  feature RMS at most one, every logit at most `√D ‖E v‖`, and every attention logit at most
  `(5/4) ln (i + 1)` at the query position `i`, for any parameters and any token sequence.
* `PlainRouter`, `OneStream`: routers with zero scales and phases, read biases `−ln 3` and write
  biases `0` keep four equal streams equal, so the model is then the one-stream model
  `h ← h + α R_D(raw(R_D h))`: the ordinary residual is a sub-model of mHC (R0).

The parameters are arbitrary: real-linear projections (the model's are the complex-linear ones, a
subclass), any routers, phase coordinates and score coefficients. The model pairs the coordinates
of a stream inside each head block; the half-split pairing of the whole stream used here is the
same after a fixed permutation of the coordinates, which arbitrary projections absorb.

What differs from the model, none of it used by the bounds: the arithmetic is exact real, not FP16
operands with FP32 accumulation, and the skip is the paper's twenty exact Sinkhorn cycles, not the
compensated log-space first cycle; the projections are not required to be complex-linear; the
width `D`, the head width `d` and the number of heads `H` are independent, the model has `D = H d`.
-/
