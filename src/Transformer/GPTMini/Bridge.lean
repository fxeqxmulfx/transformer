/-
# Bridges from `GPTMini` to the formalized theory

This module collects the structural lemmas linking the `gpt-mini`
forward pass to the canonical setups of the seven formalized papers
under `Transformer.Section1_IPS … Transformer.MeanField`.

The bridges enable the following inferential chain:

  `gpt-mini` forward
   ↓ `Bridge.SphereResidence`  (RMSNorm → tokens on √d-sphere)
  spherical IPS with QK-normed scores
   ↓ `Bridge.XSAEquivalence`    (XSA at V=I → spherical projection)
  spherical SA from `Section1_IPS`
   ✗ `Bridge.RoPEAsTimeVarying` (RoPE = pair-dependent keys R(p_j - p_i) K,
     not the depth-varying Q(t), K(t) of the survey: the chain breaks here
     unless RoPE is off; `Bridge.RoPENoClustering`: even the misread chain
     clusters almost every initial sequence, never every one)
   ↓ `Bridge.CausalConnection`  (mask matches eq: csa)
  causal SA from `Causal.Basic`
   ↓ `Causal.MainTheorem.thm1`  (clustering, modulo extension)

The substantive contents of each bridge are proven (where the underlying
algebra suffices) or stated with `sorry` (where the deep theorem from the
target paper is itself sorry-leaf).

`Bridge.ALMLookup` runs the other way: it asks whether this architecture can
host the exact lookup head of `Transformer.ALM`, and answers in two halves --
QK-norm makes the head blind to the length of a key, so the paraboloid head is
not an instance of it at any temperature; but on a common sphere the two rank
the keys identically, so only the lattice gap, a constant, is lost.
-/

import Transformer.GPTMini.Bridge.SphereResidence
import Transformer.GPTMini.Bridge.XSAEquivalence
import Transformer.GPTMini.Bridge.CausalConnection
import Transformer.GPTMini.Bridge.RoPEAsTimeVarying
import Transformer.GPTMini.Bridge.RoPENoClustering
import Transformer.GPTMini.Bridge.ALMLookup
