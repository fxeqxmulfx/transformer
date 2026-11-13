module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Witness

/-!
# Fixed support routing with a frozen complex parent

Source: `review/flat_spectrum.md`, CM38. Actual gated complex features preserve
the parent outside candidate support, with both coefficient blocks charged.
The affine parent-frozen domain and ordinary normalized CE are convex. Covered
observations have exactly the candidate's loss function and derivatives in
candidate coordinates. A bounded flat-code witness retains equal training CE
yet regresses on a covered unseen input, while preserving an unsupported input.
Native support checks, state loading and complete evaluation remain separate.
-/
