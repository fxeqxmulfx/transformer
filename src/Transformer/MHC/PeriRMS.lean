module

public import Transformer.MHC.PeriRMS.SqrtGain

/-!
# Finite mHC with scaled and unscaled Peri-RMS branches

Local extension of Xie et al., arXiv:2512.24880v2, §§4.1–4.2, to the
Q/K-normalized SSMax model from diff-gpt c6c517b. Both attention gains,
`1/sqrt(head_width)` and `1/head_width`, are covered with and without
post-RMS depth scaling and with the original twice-sigmoid write gates.
The development includes the independent score bounds and
concrete nonzero examples; it assumes no exact finite double stochasticity.
See `review/mhc_peri_rms/validation.json` for migration and check evidence.
-/
