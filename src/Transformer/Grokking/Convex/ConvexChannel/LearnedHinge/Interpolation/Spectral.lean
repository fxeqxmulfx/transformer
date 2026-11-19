module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Interpolation.Basic
public import Mathlib.LinearAlgebra.Matrix.ConjTranspose

/-!
# The actual full-word interpolation operator has a flat Gram matrix

Source: `review/convex/moments.md`, CM84. Every column is evaluated through
the learned raw complex neuron. Its complete Gram matrix is 10I and its
energy is exactly ten times input energy. Flatness does not make the number
of independent full-word coordinates small. This concerns a specific learned
configuration, not every parameter point or a training trajectory.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Interpolation

open scoped BigOperators Matrix

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- The complete response operator of actual learned neurons, CM84. -/
noncomputable def wordOperator : Matrix (Fin T → O) (Fin T → O) ℂ :=
  fun word needle => complexNeuron (Unseen.complexWeights needle) (rawInput (fullPrefix word))

/-- Actual raw inference gives the same nonzero complex gain at each
singleton coordinate, CM84, retaining every full word. -/
theorem wordOperator_entry (word needle : Fin T → O) :
    wordOperator word needle = if word = needle then (⟨1, 3⟩ : ℂ) else 0 :=
  neuron_entry word needle

/-- The full Gram matrix has the same strictly positive diagonal entry
on every word, CM84. Its dimension still grows as |O|^T. -/
theorem wordOperator_gram :
    wordOperator (O := O) (T := T) * wordOperator.conjTranspose =
      (10 : ℂ) • (1 : Matrix (Fin T → O) (Fin T → O) ℂ) := by
  ext word other
  simp only [Matrix.mul_apply, Matrix.conjTranspose_apply, wordOperator_entry,
    Matrix.smul_apply, Matrix.one_apply]
  by_cases h : word = other
  · subst other
    simp
    apply Complex.ext <;> norm_num
  · have ho : other ≠ word := fun he => h he.symm
    simp [h, ho]

/-- The actual operator preserves every complex input coordinate, up to
the common nonzero gain, CM84. -/
theorem wordOperator_mulVec (v : (Fin T → O) → ℂ) :
    wordOperator *ᵥ v = fun word => (⟨1, 3⟩ : ℂ) * v word := by
  funext word
  simp only [Matrix.mulVec, dotProduct, wordOperator_entry, ite_mul, zero_mul]
  exact Fintype.sum_ite_eq word _

/-- A flat spectrum supplies an exact norm bound on the full operator,
CM84, with no removed coordinate or inferred missing answer. -/
theorem wordOperator_energy (v : (Fin T → O) → ℂ) :
    (∑ word, Complex.normSq ((wordOperator *ᵥ v) word)) =
      10 * ∑ word, Complex.normSq (v word) := by
  rw [wordOperator_mulVec]
  have hgain : Complex.normSq (⟨1, 3⟩ : ℂ) = 10 := by norm_num [Complex.normSq]
  simp only [Complex.normSq_mul, hgain, Finset.mul_sum]

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Interpolation
