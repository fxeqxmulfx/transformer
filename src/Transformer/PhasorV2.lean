module

public import Transformer.PhasorV2.SSMax
public import Transformer.PhasorV2.SSMaxLength
public import Transformer.PhasorV2.LeRoPE
public import Transformer.PhasorV2.LeRoPEBands
public import Transformer.PhasorV2.PeriNorm
public import Transformer.PhasorV2.QKNorm
public import Transformer.PhasorV2.MHCPhase
public import Transformer.PhasorV2.MHCRouter
public import Transformer.PhasorV2.MHC
public import Transformer.PhasorV2.Attention
public import Transformer.PhasorV2.FFN
public import Transformer.PhasorV2.Readout

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
  real coordinates, with a parameter-free RMS over the complex channels where the source's `R_D`
  takes it over the real coordinates, and normalizes the input and the output of every branch.
  The output has length at most `√c` whatever the branch, and a causal branch stays causal.
* Q/K RMS (DiffGPT's main model): `Transformer.PhasorV2.QKNorm` normalizes queries and keys over
  their `p` complex bands. A normalized query and key score at most `p` at all positions, so
  under the gain `1/p` every SSMax logit at the length `N` is at most `ln N` and no key outweighs
  another by more than the factor `N²`, for all weights.
* mHC with phases (Xie et al., arXiv:2512.24880v2, §4.2, as DiffGPT's main model trains it):
  four streams, the router of the paper, and the skip map `D P D*` with the stream phases
  `(1 + i b)/√(1 + b²)` in the open right half-plane. `Transformer.PhasorV2.MHCPhase` shows that
  this skip turns the streams back, mixes them as the paper's skip does and turns them forward,
  so it keeps the paper's bounds and the streams aligned with the phases.
  `Transformer.PhasorV2.MHCRouter` is the router:
  sigmoid read gates, twice-sigmoid write gates, twenty Sinkhorn cycles. `Transformer.PhasorV2.MHC`
  is the sublayer. From the normalized embedding every stream has RMS at most `1 + 2 n |α|` after
  `n` sublayers, `1 + 2√(2L)` at the source's `α = 1/√(2L)`, for all weights, and causal branches
  give a causal stack.
* Attention (DiffGPT's main model): `Transformer.PhasorV2.Attention` puts these elements together
  from the embedding to the output of attention, complex throughout: the embedding table holds
  complex vectors, the projections are complex-linear, and a head is `p` complex bands. For all
  weights the score depends on the positions only through their offset, every logit at the
  position `i` is at most `ln(i + 1)`, no key outweighs another by more than `(i + 1)²`, and the
  branch is causal.
* The complex FFN (DiffGPT's main model): `Transformer.PhasorV2.FFN` is the plain complex product
  of two complex-linear projections, mapped back by a third. It is a quadratic form, so it doubles
  the phase of its input and is even. An attention sublayer and a feed-forward one make a layer,
  and from the embedding the streams at a position depend on the tokens up to it only.
* Tied readout (DiffGPT's main model) takes the unweighted mean of the four streams, applies
  RMS over complex channels and scores each token with the real part of the inner product
  against its learned embedding row. The mean retains at least a quarter of the length of a
  phase-aligned mode. `Transformer.PhasorV2.Readout` gives the causal forward pass through all
  layers with a shared table for the normalized input lookup and the output projection.
-/
