module

public import Transformer.Phasor.TwoCopies.Slots
public import Transformer.Phasor.TwoCopies.States
public import Transformer.Phasor.TwoCopies.Pairs
public import Transformer.Phasor.TwoCopies.Heads
public import Transformer.Phasor.TwoCopies.Read
public import Transformer.Phasor.TwoCopies.Forms
public import Transformer.Phasor.TwoCopies.Product
public import Transformer.Phasor.TwoCopies.Constants
public import Transformer.Phasor.TwoCopies.Forward
public import Transformer.Phasor.TwoCopies.Logits
public import Transformer.Phasor.TwoCopies.Margin
public import Transformer.Phasor.TwoCopies.Theorem

/-!
# Two copies of any task through main's layers

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8, task T3c.

Two copies `q₀(x₀, y₀) + q₁(x₁, y₁)` over `ℤ/p`, the operands at the positions `0, …, 3`, the
answer read at the position `3`: an exact construction in the one-stream model of `OneStream`
with `D ≥ 4p + 7`, `d ≥ max(p, 4)`, four heads, `F ≥ 2p` units and two layers, and by R0 in main's
model with mHC, for every `ε ≥ 0` and every depth scale `s > 0`.

* `Slots`, `States`: `4p + 7` orthonormal directions in the stream, the RMS of a combination of
  them, the embedding and the states that the position `3` passes through.
* `Pairs`, `Heads`, `Read`: the first layer's heads, three DFT heads of RS5 and one uniform head,
  and the output projection that decodes them, which writes each other operand into a code of its
  own.
* `Forms`, `Product`: the two feed-forward branches. The first computes `ψ(q_c(x_c, y_c))` with
  `p` units per copy; the second multiplies the two phases with one unit.
* `Constants`, `Forward`, `Logits`: the RMS factors, the two layers, the zero layers after them
  and the logits at every depth `L ≥ 2`.
* `Margin`, `Theorem`: the bias and the gain that make the answer's logit the largest.
-/
