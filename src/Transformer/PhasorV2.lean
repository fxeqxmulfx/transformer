module

public import Transformer.PhasorV2.SSMax
public import Transformer.PhasorV2.SSMaxLength
public import Transformer.PhasorV2.LeRoPE
public import Transformer.PhasorV2.LeRoPEBands
public import Transformer.PhasorV2.PeriNorm

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
* The Peri-norm residual (DiffGPT's main model, `research/main_model/README.txt`):
  `Transformer.PhasorV2.PeriNorm` normalizes the input and the output of every branch and scales
  the output by `α`. Every update has length at most `|α| √D` whatever the branch, so from a
  normalized embedding the stream has RMS at most `1 + n |α|` after `n` sublayers, and causal
  branches give a causal stack.
-/
