module

public import Transformer.PhasorV2.SSMax
public import Transformer.PhasorV2.SSMaxLength
public import Transformer.PhasorV2.LeRoPE
public import Transformer.PhasorV2.LeRoPEBands
public import Transformer.PhasorV2.PeriNorm
public import Transformer.PhasorV2.QKNorm

/-!
# Phasor v2

A second version of Phasor (`Transformer.Phasor`, the main model of DiffGPT), built only from
the elements chosen for it one at a time.

* Scalable-Softmax (Nakanishi, arXiv:2501.19399v1, §2): `Transformer.PhasorV2.SSMax` defines the
  weights and proves the bounds of §2.2; `Transformer.PhasorV2.SSMaxLength` lets the number of
  keys grow. A lead of `g > 1/s` keeps a key's weight near one at every length, while under
  softmax every fixed lead fades.
* LeRoPE (Karypis et al., arXiv:2607.10134v1, §3.1): `Transformer.PhasorV2.LeRoPE` defines the
  learned frequencies and turns the complex bands of a head by them; the score depends only on
  the offset and keeps the bound of the unturned vectors. `Transformer.PhasorV2.LeRoPEBands`
  shows that every Partial RoPE head, NoPE's included, is a LeRoPE head at integer positions,
  while a band at `π` sees the parity of the offset, which NoPE cannot.
* Peri-norm (DiffGPT's main model, `research/main_model/README.txt`):
  `Transformer.PhasorV2.PeriNorm` writes the model's vectors as `c` complex channels, `D = 2c`
  real coordinates, with the parameter-free `R_D`, and normalizes the input and the output of
  every branch. The output has length at most `√D` whatever the branch, and a causal branch stays
  causal.
* Q/K RMS (DiffGPT's main model): `Transformer.PhasorV2.QKNorm` normalizes the bands of queries
  and keys over the real head width `d`. A normalized query and key score at most `d` at all
  positions, so under the gain `1/d` every SSMax logit at the length `N` is at most `ln N` and no
  key outweighs another by more than the factor `N²`, for all weights.
-/
