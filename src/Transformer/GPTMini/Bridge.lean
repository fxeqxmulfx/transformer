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
   ↓ `Bridge.RoPEAsTimeVarying` (RoPE = time-varying Q, K)
  time-varying-Q,K spherical SA, `V = I_d`
   ↓ `Bridge.CausalConnection`  (mask matches eq: csa)
  causal SA from `Causal.Basic`
   ↓ `Causal.MainTheorem.thm1`  (clustering, modulo extension)

The substantive contents of each bridge are proven (where the underlying
algebra suffices) or stated with `sorry` (where the deep theorem from the
target paper is itself sorry-leaf).
-/

import Transformer.GPTMini.Bridge.SphereResidence
import Transformer.GPTMini.Bridge.XSAEquivalence
import Transformer.GPTMini.Bridge.CausalConnection
import Transformer.GPTMini.Bridge.RoPEAsTimeVarying
