module

public import Transformer.SequenceModel.Attention.Backend.Refined.Counting

/-!
# A trace-dependent source execution certificate

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M1--M5.
Recorded normalization denominators, separate matrix blocks and local residuals
certify a 1/16 logit radius and 7/10 source margin for the complete CUDA trace.
Every class decision agrees with exact source arithmetic. The spectral/counting
adapter accepts this certificate while retaining independent fixed-point conditions.
-/
