module

public import Transformer.Grokking.FlatSpectrum.Kernels

/-!
# Finite Fourier mixtures and their amplitude spectrum

Source: `review/flat_spectrum.md`, CM20. These definitions describe a vector
inside an exact Fourier subspace. The input indices may later be permuted;
no arithmetic meaning of an observed input token is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction

open Finset

variable {q : ℕ} [NeZero q] {ι : Type*}

/-- Every ordered off-diagonal frequency difference has exactly one source
pair, CM20. Comparing with diagonal pairs also excludes zero differences. -/
def Separated (k : ι → ZMod q) : Prop :=
  ∀ i j, i ≠ j → ∀ u v, k j - k i = k v - k u → i = u ∧ j = v

/-- A vector in the declared Fourier subspace, CM20. -/
noncomputable def wave [Fintype ι] (k : ι → ZMod q) (a : ι → ℂ) (x : ZMod q) : ℂ :=
  ∑ i, a i * ZMod.stdAddChar (x * k i)

/-- The unnormalized Fourier coefficient of the actual squared amplitude,
CM20. It is not defined from a presumed sparse coefficient representation. -/
noncomputable def powerCoefficient [Fintype ι]
    (k : ι → ZMod q) (a : ι → ℂ) (t : ZMod q) : ℂ :=
  ∑ x, star (ZMod.stdAddChar (x * t)) * (Complex.normSq (wave k a x) : ℂ)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction
