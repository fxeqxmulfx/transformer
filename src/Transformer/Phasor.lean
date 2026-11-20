module

public import Transformer.Phasor.Router
public import Transformer.Phasor.Attention

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

The parameters are arbitrary: real-linear projections (the model's are the complex-linear ones, a
subclass), any routers, phase coordinates and score coefficients. The model pairs the coordinates
of a stream inside each head block; the half-split pairing of the whole stream used here is the
same after a fixed permutation of the coordinates, which arbitrary projections absorb.
-/
