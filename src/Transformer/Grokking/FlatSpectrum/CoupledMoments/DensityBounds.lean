module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds.Examples

/-!
# Convex density tolerances and conditional stability

Source: `review/flat_spectrum.md`, CM22. Norm-bounded diagonal error preserves
convexity and is an actual real interval constraint on PSD kernels. Separated
Fourier differences bound hidden cross terms; explicit feature perturbations
bound kernel errors. Approximate objective accuracy plus a strict scalar score
gap controls mixed mass. These bounds do not certify subspace discovery from
missing labels, approximate lexicographic tie breaking, or unseen answers.
-/

@[expose] public section
