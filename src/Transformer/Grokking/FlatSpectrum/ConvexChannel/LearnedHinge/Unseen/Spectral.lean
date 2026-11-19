module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Witness
public import Mathlib.LinearAlgebra.Matrix.ConjTranspose

/-!
# A flat observed operator from actual learned complex neurons still has a blind direction

Source: `review/convex/moments.md`, CM77. Three original raw prefixes select
three actual complex neurons. The first two observed rows have Gram matrix
10I and a sharp energy bound. The retained third coordinate is invisible to
those rows but changes the missing-prefix response. Thus this actual flat
operator supplies a bound, not the unobserved label or identifiability.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Spectral

open scoped BigOperators Matrix

noncomputable section

/-- Three different complete object prefixes for the actual operator. -/
def needles (j : Fin 3) : Fin 2 → Bool := ![![false, false], ![false, true], ![true, true]] j

/-- The raw prefix passed to the unchanged learned network. -/
def tokens (i : Fin 3) : Prefix Bool 2 := ⟨List.ofFn (needles i), by simp⟩

/-- Columns are whole actual complex neuron outputs, not assumed features. -/
def operator : Matrix (Fin 3) (Fin 3) ℂ :=
  fun i j => complexNeuron (complexWeights (needles j)) (rawInput (tokens i))

/-- The first two observed rows retain every original neuron column. -/
def observedOperator : Matrix (Fin 2) (Fin 3) ℂ := fun i j => operator i.castSucc j

/-- The complete complex operator is the literal gain times the identity,
CM77, derived from the raw sequence computation. -/
theorem operator_entry (i j : Fin 3) :
    operator i j = if i = j then (⟨1, 3⟩ : ℂ) else 0 := by
  fin_cases i <;> fin_cases j <;>
    norm_num [operator, neuron_singleton, tokens, needles, List.ofFn_succ]

/-- Every nonzero observed-row Gram eigenvalue is represented by the same
literal diagonal value, CM77. No zero-column deletion is needed for this equality. -/
theorem observed_gram : observedOperator * observedOperator.conjTranspose =
    (10 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [Matrix.mul_apply, Matrix.conjTranspose_apply, observedOperator,
      operator_entry, Fin.sum_univ_three, Matrix.one_apply] <;>
    apply Complex.ext <;> norm_num

/-- The full observed multiplication reads its two visible coordinates,
CM77, while the third original input coordinate remains in the domain. -/
theorem observed_mulVec (v : Fin 3 → ℂ) :
    observedOperator *ᵥ v = fun i => (⟨1, 3⟩ : ℂ) * v i.castSucc := by
  funext i
  fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, observedOperator, operator_entry]

/-- Squared Euclidean energy of every retained complex coordinate. -/
def energy {K : Type*} [Fintype K] (v : K → ℂ) : ℝ := ∑ k, Complex.normSq (v k)

/-- An exact actual-operator energy identity keeps the invisible coordinate
explicitly, CM77. The coefficient 10 is the squared complex gain. -/
theorem energy_identity (v : Fin 3 → ℂ) :
    energy (observedOperator *ᵥ v) + 10 * Complex.normSq (v 2) = 10 * energy v := by
  rw [observed_mulVec]
  have hg : Complex.normSq (⟨1, 3⟩ : ℂ) = 10 := by norm_num [Complex.normSq]
  simp only [energy, Fin.sum_univ_two, Fin.sum_univ_three, Complex.normSq_mul, hg]
  simp only [show (0 : Fin 2).castSucc = (0 : Fin 3) from rfl,
    show (1 : Fin 2).castSucc = (1 : Fin 3) from rfl]
  ring

/-- Flatness gives the actual norm bound for every complex direction,
CM77. It does not remove the unobserved null direction. -/
theorem energy_bound (v : Fin 3 → ℂ) : energy (observedOperator *ᵥ v) ≤ 10 * energy v := by
  have h := energy_identity v
  have hn := Complex.normSq_nonneg (v 2)
  linarith

/-- The original missing-prefix coefficient direction. -/
def blind : Fin 3 → ℂ := ![0, 0, 1]

/-- The invisible direction is genuinely nonzero, CM77. -/
theorem blind_ne_zero : blind ≠ 0 := by
  intro h
  have he := congrFun h 2
  norm_num [blind] at he

/-- This retained nonzero direction changes no observed complex output,
CM77, despite the exactly flat positive observed-row Gram matrix. -/
theorem blind_observed : observedOperator *ᵥ blind = 0 := by
  rw [observed_mulVec]
  funext i
  fin_cases i <;> simp [blind]

/-- The same direction changes the actual missing-prefix complex output,
CM77. Its real logit contribution is one. -/
theorem blind_unseen : (operator *ᵥ blind) 2 = (⟨1, 3⟩ : ℂ) := by
  simp [Matrix.mulVec, dotProduct, operator_entry, blind]

example := energy_bound ![1, Complex.I, 2]

end

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Spectral
