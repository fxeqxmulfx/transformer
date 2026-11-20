module

public import Transformer.MHC.PeriRMS.SqrtGain
public import Transformer.MHC.PeriRMS.PhaseScore
public import Transformer.MHC.PeriRMS.PhaseScoreHead

/-!
# Finite mHC with scaled and unscaled Peri-RMS branches

Local extension of Xie et al., arXiv:2512.24880v2, §§4.1–4.2, to the
Q/K-normalized SSMax model from diff-gpt c6c517b. Both attention gains,
`1/sqrt(head_width)` and `1/head_width`, are covered with and without
post-RMS depth scaling and with the original twice-sigmoid write gates.
The development includes the independent score bounds and
concrete nonzero examples; it assumes no exact finite double stochasticity.
`PhaseScore` and `PhaseScoreHead` bound the squared-imaginary phase score of the complex heads
by `(5/4) ln N`, after the same Q/K RMS and arbitrary rotary phases. The skip maps are
conjugated by stream phases, `D P D*` (`Basic`, `MHC.PhaseStep`), and `CurrentModel` states the
bounds of DiffGPT's main model, both scores included.
See `review/mhc_peri_rms/validation.json` for migration and check evidence.
-/
