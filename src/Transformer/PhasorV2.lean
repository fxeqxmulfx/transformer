module

public import Transformer.PhasorV2.SSMax
public import Transformer.PhasorV2.SSMaxLength

/-!
# Phasor v2

A second version of Phasor (`Transformer.Phasor`, the main model of DiffGPT), built only from
the elements chosen for it one at a time.

* Scalable-Softmax (Nakanishi, arXiv:2501.19399v1, §2): `Transformer.PhasorV2.SSMax` defines the
  weights and proves the bounds of §2.2; `Transformer.PhasorV2.SSMaxLength` lets the number of
  keys grow. A lead of `g > 1/s` keeps a key's weight near one at every length, while under
  softmax every fixed lead fades.
-/
