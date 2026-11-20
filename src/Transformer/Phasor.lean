module

public import Transformer.Phasor.Router

/-!
# The main model of DiffGPT as a forward pass

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`,
`model.json` and `mhc.json`: the complex Peri-RMS transformer of arm `phase-score-lerope`, four mHC
streams with a phase-conjugated skip, a complex product FFN, LeRoPE with SSMax and the
squared-imaginary phase score, a tied embedding. The abstract blocks and their bounds are in
`Transformer.MHC.PeriRMS`; this folder gives the model's own router, heads and feed-forward branch
as concrete maps and plugs them into it, in exact real arithmetic.

* `Router`: the parameter-free RMS of the four streams, the three learned scales and 24 biases of
  the read, write and skip logits, and the phase coordinates of streams `1, 2, 3`, as a `Routing`.
-/
