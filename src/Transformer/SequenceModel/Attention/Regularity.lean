module

public import Transformer.SequenceModel.Attention.Regularity.Model

/-!
# Regularity of the actual normalized Transformer

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, R1.
Every score and finite sampled cross-entropy loss is analytic jointly in
all actual weights wherever the actual XSA values avoid the clipping sphere.
The proof retains positive RMS epsilon, arbitrary fixed rotary phases,
full causal softmax, both normalized residual branches and the tied readout.
Zero weights witness the conditions for every positive clipping epsilon.
-/
