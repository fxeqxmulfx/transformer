/-
# Input Convex Encoder-Only Transformer

Kaipeng Xu, Zhuo Zhi, Keyue Jiang, arXiv:2603.22095v2:
"Input Convex Encoder-Only Transformer for Computationally Efficient
Model Predictive Control in Building Demand Response".

The PDF, extracted text and complete LaTeX archive are stored locally in
`papers/arXiv-2603.22095v2/`. This development has no `sorry`.

Source coverage:

* §2.1 and §2.2.4: ordinary dot-product/softmax attention on two scalar
  tokens is proved non-convex (`Section2_AttentionFalse`).
* §2.2.1, Eq. (1): the ICNN recurrence at varying layer widths, convexity
  with unrestricted passthrough weights, and monotonicity with non-negative
  passthrough weights (`Section2_FeedForward`). The targeted state-only
  restriction, with unrestricted control weights, is proved separately in
  `Section2_FeedForwardSelective`.
* §2.2.2, Eqs. (2), (3): all six ICRNN matrices, the real recurrence/output,
  and convexity of both expanded and original sequences (`Section2_Recurrent`).
* §2.2.3, Eqs. (4)–(11): the actual one-layer IC-LSTM, its shared matrices,
  four diagonal scales, gate/cell/hidden updates, dense skip and readout.
  The claimed sufficient conditions for sequence convexity are FALSE:
  `iclstm_paper_convexity_false` provides an explicit three-step counterexample
  with strictly positive convex monotone internal activations. The midpoint
  output is 16, exceeding the endpoint average `(17+13)/2 = 15`.
* §3.2–3.3, Eqs. (12)–(27): explicit embedding, positional encoding, queries,
  keys, shared latent, diagonal gate/value product, all-source average,
  concatenation/projection, both residuals, feed-forward sublayer, block stack,
  last-token readout, full signed expansion and positive de-standardization.
  Every structural requirement is an explicit predicate on actual parameters.
  The layer-normalization obstruction after Eq. (25) has an explicit
  two-channel counterexample (`Section3_Normalization`).
* §3.3, Definition 1 / Lemma 1: `componentwiseConvex_iff`, `affine_convex`,
  `affine_monotone`, `composition_properties`, `activation_properties`.
* §3.3, Lemma 2: `gateValue_nonnegative`, `gateValue_monotone`,
  `gateValue_convex`, with a derivative-free Jensen proof covering ReLU.
* §3.3, Proposition 1: `attention_properties` for Eqs. (15)–(22).
* §3.3, Proposition 2: `encoderBlock_properties` for Eqs. (23)–(25).
* §3.3, Theorem 1: `predict_properties` for every encoder depth.
* §3.3, Corollary 1: `predictOriginal_convex`; `Section3_Expansion` also
  certifies signed affine expressivity and a decreasing original-input example.
* §3.4, Eqs. (28)–(30): the literal toy surfaces, with proved Jensen
  counterexamples to convexity (`Section3_Toy`). The ambiguous real fractional
  power in Eq. (30) is documented; it is not changed to the standard polynomial.
* §4.1, Eqs. (31)–(43): finite controls/slacks, selective control expansion,
  real history shifts, recursively generated predictions, all control/comfort/
  slack bounds and the complete electricity-plus-quadratic-slack objective.
* §4.1, Corollary 2: `soft_constrained_mpc_convex`; the induction requires
  monotonicity only in recursively predicted columns, exactly as stated.
  `selectivePredict_conditions` connects this premise to the proved IC-EoT.
* §4.1, following Corollary 2: `mpc_local_minimum_global` for feasible local
  minima. Neither solver convergence nor true-plant optimality is assumed.
* §3.4 and §4.3–4.4, Tables 1–7: exact reported model dimensions, parameter
  counts, predictive metrics, all 24 training S/R/A/I cells, all 64 solver S/A/M
  cells, aggregate counts/rates, rounded time reductions/speedups, and full-day
  cost/comfort comparisons. Adopted tariffs and horizon durations are checked.

The numerical theorems are consequences of TRANSCRIBED published values,
not proofs that training or EnergyPlus/IPOPT runs were reproduced. The source
archive contains no raw training data, fitted models or simulation traces.
Qualitative empirical comparisons are represented by their exact reported
inequalities/differences rather than by a universal performance guarantee.
§5 repeats these results and proposes future experiments, not conjectures.

Two additional source discrepancies are recorded in theorem docstrings:
`reported_office_r2_gap_corrected` gives 0.0059 from Table 4, versus the
prose's 0.0060; `equation_parameter_counts_disagree` counts the independently
parameterized written architecture as 31,179/33,104 versus Table 3's
23,563/25,488. Unreported implementation-specific tying could explain the
second discrepancy. Both distinctions remain explicit and auditable.

All mathematically conditional results include examples witnessing their
assumptions. Claims proven false are refuted, rather than weakened or sorried.
-/

import Transformer.ICEoT.Section2_FeedForward
import Transformer.ICEoT.Section2_FeedForwardSelective
import Transformer.ICEoT.Section2_AttentionFalse
import Transformer.ICEoT.Section2_Recurrent
import Transformer.ICEoT.Section2_LSTMFalse
import Transformer.ICEoT.Section3_Expansion
import Transformer.ICEoT.Section3_Toy
import Transformer.ICEoT.Section3_Normalization
import Transformer.ICEoT.Section4_EncoderMPC
import Transformer.ICEoT.Section4_RecursionLimits
import Transformer.ICEoT.Section4_Parameters
import Transformer.ICEoT.Section4_Tariffs
