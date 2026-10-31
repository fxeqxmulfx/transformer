module

public import Transformer.Rotate90.Power.Depth

/-!
# Powered division in the actual normalized Rotate90 model

Source: `review/flat_spectrum.md`, *Actual network expressivity for every exponent*.
For every prime, natural exponent and positive RMSNorm epsilon, explicit tied
weights strictly classify every numerator and nonzero denominator. The construction
uses fourteen active residual coordinates and twenty genuine SwiGLU units. Arbitrary
finite padding is allowed, including the source width-16, hidden-43 configuration.
Zero residual blocks extend the same result to every positive finite depth.

`Power.exists_weights` and `Power.exists_weights_any_depth` state the two general
results. `Power.exists_weights_width16_any_depth` is the tested small architecture.
The proofs include zero numerators and exponent zero, all normalizations, one shared
embedding, and exact SiLU gates. They concern real-arithmetic expressivity, not the
convergence of training or the behavior of an infinite-width limit.
-/
