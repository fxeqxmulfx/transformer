/-
# DoReFa-Net: low bitwidth weights, activations and gradients

Formalization of Zhou, Wu, Ni, Zhou, Wen, Zou — arXiv:1606.06160v3,
"DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low
Bitwidth Gradients".

A DoReFa-Net runs its convolutions on `W`-bit weights and `A`-bit activations
and back-propagates `G`-bit gradients, so that every expensive operation of
training — `forward`, `backward_input`, `backward_weight` — is a dot product of
fixed-point integers and therefore a sum of `bitcount(and(·, ·))` kernels.
Weights and activations can be quantized deterministically; gradients cannot,
and the paper's contribution is the additive dither that makes a `2`-bit
gradient usable at all.

| Module | Contents |
| --- | --- |
| `DoReFa.Section2_BitConv` | the bitwise dot product, its `M · K` bit planes, and a sign slip in footnote 2 |
| `DoReFa.Section2_Quantize` | `quantize_k`: the grid it rounds onto, its error, and its monotonicity |

Everything rests on one operator, `quantize_k`, and the paper's claims about
it are claims about rounding: what grid the output lands on, how far it is from
the input, and that it is monotone.  §3 is a table of accuracies and §4 a
discussion; neither is formalized.
-/

import Transformer.DoReFa.Section2_BitConv
import Transformer.DoReFa.Section2_Quantize
