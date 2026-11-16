module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Model

/-!
# Complete complex inference with learned representations and readouts

Source: `review/flat_spectrum.md`, CM75. Each complex preactivation is
rectified in both real and imaginary components. Its learned readout is
`amplitude * (1 + i)`, so both components affect the real output logit.
The phase and class/sign directions are fixed architecture, while every
representation coordinate and readout amplitude depends on training.
The phase adds a constant factor to the readout norm; no claim of ordinary
independent complex factor convexity or unchanged weight decay is made.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge

open scoped BigOperators

variable {E J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Componentwise complex ReLU, CM75. Both components are evaluated. -/
def splitHinge (z : ℂ) : ℂ := ⟨hinge z.re, hinge z.im⟩

/-- Positive real scaling commutes with the actual complex activation,
CM75. Arbitrary complex rotations do not satisfy this identity. -/
theorem splitHinge_smul (a : ℝ) (ha : 0 ≤ a) (z : ℂ) :
    splitHinge (a • z) = a • splitHinge z := by
  apply Complex.ext <;>
    simp [splitHinge, hinge, mul_max_of_nonneg _ _ ha]

/-- The learned complex readout, CM75. Its real and imaginary parts both
change with the recovered amplitude. -/
noncomputable def complexReadout (u : E) : ℂ := (amplitude u : ℂ) * (1 + Complex.I)

/-- Actual complex embedding, split activation, and full product with the
learned complex readout, CM75. Nothing is projected out before multiplication. -/
noncomputable def complexNeuron (u : E) (input : E →ₗ[ℝ] ℂ) : ℂ :=
  complexReadout u * splitHinge (input (representation u))

/-- Exact full complex recovery holds at every input, CM75, including
unobserved inputs and coordinates outside the activation-cone constraints. -/
theorem complexNeuron_eq (u : E) (input : E →ₗ[ℝ] ℂ) :
    complexNeuron u input = (1 + Complex.I) * splitHinge (input u) := by
  have h : amplitude u • splitHinge (input (representation u)) = splitHinge (input u) := by
    rw [← splitHinge_smul _ (amplitude_nonneg u), ← input.map_smul, recover]
  calc
    complexNeuron u input = (1 + Complex.I) *
        (amplitude u • splitHinge (input (representation u))) := by
      simp only [complexNeuron, complexReadout, Complex.real_smul]
      ring
    _ = (1 + Complex.I) * splitHinge (input u) := by rw [h]

/-- Every physical neuron with nonnegative amplitude and the declared
complex phase is represented exactly on all inputs, CM75. The recovered
factors can have a different scale but the complete complex output is equal. -/
theorem complexNeuron_of_scaled (a : ℝ) (ha : 0 ≤ a) (w : E) (input : E →ₗ[ℝ] ℂ) :
    complexNeuron (a • w) input = (a : ℂ) * (1 + Complex.I) * splitHinge (input w) := by
  rw [complexNeuron_eq, map_smul, splitHinge_smul _ ha, Complex.real_smul]
  ring

/-- The real component of the actual complex product uses both learned
preactivation components with their correct signs, CM75. -/
theorem complexNeuron_re (u : E) (input : E →ₗ[ℝ] ℂ) :
    (complexNeuron u input).re =
      neuron u (Complex.reLm.comp input) - neuron u (Complex.imLm.comp input) := by
  rw [complexNeuron_eq, neuron_eq, neuron_eq]
  simp [splitHinge, Complex.mul_re]

/-- Complete complex inference across every neuron and sign slot, CM75. -/
noncomputable def complexForward [Fintype J] (p : Parameters J C E)
    (input : E →ₗ[ℝ] ℂ) (c : C) : ℂ :=
  ∑ j, (complexNeuron (p j c true) input - complexNeuron (p j c false) input)

/-- The classifier reads real logits after the full complex computation,
CM75. Imaginary representation values already contributed in each product. -/
noncomputable def complexLogits [Fintype J] (p : Parameters J C E)
    (input : E →ₗ[ℝ] ℂ) (c : C) : ℝ := (complexForward p input c).re

/-- Expanding all complex products gives the difference of two actual
learned real-neuron computations, CM75. Both use the same joint parameters. -/
theorem complexLogits_eq [Fintype J] (p : Parameters J C E) (input : E →ₗ[ℝ] ℂ) :
    complexLogits p input =
      forward p (Complex.reLm.comp input) - forward p (Complex.imLm.comp input) := by
  funext c
  change Complex.reLm (∑ j, (complexNeuron (p j c true) input -
    complexNeuron (p j c false) input)) = _
  rw [map_sum]
  change (∑ j, (complexNeuron (p j c true) input - complexNeuron (p j c false) input).re) = _
  simp only [Complex.sub_re, complexNeuron_re, forward, Pi.sub_apply, Finset.sum_sub_distrib]
  ring

example := splitHinge_smul 2 (by norm_num) (1 + Complex.I)
example := complexNeuron_of_scaled 2 (by norm_num) (1 + Complex.I)
  (LinearMap.id : ℂ →ₗ[ℝ] ℂ)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge
