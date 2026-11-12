module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction.Stability

/-!
# A separated three-mode witness and a collision counterexample

Source: `review/flat_spectrum.md`, CM20, examples. Frequencies 1, 3 and 9 modulo
13 have unique ordered differences. In contrast, the two differences modulo 2
collide, and a genuine two-mode mixture has unit amplitude at every state.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction

open Finset

/-- A three-dimensional latent subspace with separated frequency differences. -/
def tripleFrequency : Fin 3 → ZMod 13 := ![1, 3, 9]

/-- All six ordered differences are distinct and nonzero, CM20. -/
theorem triple_separated : Separated tripleFrequency := by
  unfold Separated tripleFrequency
  decide

/-- A nonzero pure direction in that subspace. -/
def pureCoefficient : Fin 3 → ℂ := ![1, 0, 0]

/-- The witness coefficient is supported at its first mode, CM20. -/
theorem pure_support (j : Fin 3) (hj : j ≠ 0) : pureCoefficient j = 0 := by
  fin_cases j <;> simp_all [pureCoefficient]

/-- Its amplitude is one at every latent state, CM20. -/
theorem pure_flat (x : ZMod 13) : Complex.normSq (wave tripleFrequency pureCoefficient x) = 1 := by
  simpa [pureCoefficient] using support_flat tripleFrequency pureCoefficient 0 pure_support x

/-- This unit-amplitude witness is centered, CM20. -/
theorem pure_centered : ∑ x, wave tripleFrequency pureCoefficient x = 0 := by
  simp_rw [wave_of_support tripleFrequency pureCoefficient 0 pure_support]
  simpa [pureCoefficient, tripleFrequency, show (1 : ZMod 13) ≠ 0 by decide]
    using sum_stdAddChar_mul 13 (1 : ZMod 13)

/-- The extracted direction distinguishes all thirteen states, CM20. -/
theorem pure_injective : Function.Injective (wave tripleFrequency pureCoefficient) := by
  have : Fact (Nat.Prime 13) := ⟨by decide⟩
  exact centered_injective triple_separated pureCoefficient ⟨0, by norm_num [pureCoefficient]⟩
    pure_flat pure_centered

/-- The positive witness is nonconstant, CM20. -/
theorem pure_nonconstant : wave tripleFrequency pureCoefficient 0 ≠
    wave tripleFrequency pureCoefficient 1 := by
  intro he
  have hz := pure_injective he
  exact (by decide : (0 : ZMod 13) ≠ 1) hz

/-- A mixture of two genuine directions in this subspace cannot pass the
constant-amplitude condition, CM20. -/
theorem mixed_not_flat :
    ¬ ∃ ρ, ∀ x, Complex.normSq (wave tripleFrequency ![1, 1, 0] x) = ρ := by
  rintro ⟨ρ, hρ⟩
  have hz := flat_pair_zero triple_separated ![1, 1, 0] hρ
    (by decide : (0 : Fin 3) ≠ 1)
  norm_num at hz

/-- The two frequencies in the collision example. -/
def collisionFrequency : Fin 2 → ZMod 2 := ![0, 1]

/-- Both coefficients are nonzero; their sum is 1 and difference is `I`. -/
noncomputable def collisionCoefficient : Fin 2 → ℂ :=
  ![1 / 2 + Complex.I / 2, 1 / 2 - Complex.I / 2]

/-- Opposite differences coincide modulo two, CM20. -/
theorem collision_not_separated : ¬ Separated collisionFrequency := by
  intro h
  have he := h 0 1 (by decide) 1 0 (by decide)
  exact (by decide : (0 : Fin 2) ≠ 1) he.1

/-- The primitive character at the nonzero state of `ZMod 2`, CM20. -/
private theorem character_two : ZMod.stdAddChar (1 : ZMod 2) = -1 := by
  have hs := sum_stdAddChar_mul 2 (1 : ZMod 2)
  simp only [show (1 : ZMod 2) ≠ 0 by decide, ite_false, mul_one] at hs
  change (∑ x : Fin 2, ZMod.stdAddChar (show ZMod 2 from x)) = 0 at hs
  rw [Fin.sum_univ_two] at hs
  change ZMod.stdAddChar (0 : ZMod 2) + ZMod.stdAddChar (1 : ZMod 2) = 0 at hs
  simp only [AddChar.map_zero_eq_one] at hs
  linear_combination hs

/-- Constant amplitude alone does not select one character: this two-mode
mixture has squared amplitude one at both states, CM20. -/
theorem collision_flat (x : ZMod 2) :
    Complex.normSq (wave collisionFrequency collisionCoefficient x) = 1 := by
  have hx : x = 0 ∨ x = 1 := by
    change (x : Fin 2) = (0 : Fin 2) ∨ x = (1 : Fin 2)
    have hall : ∀ z : Fin 2, z = 0 ∨ z = 1 :=
      Fin.forall_fin_two.mpr ⟨Or.inl rfl, Or.inr rfl⟩
    exact hall x
  rcases hx with rfl | rfl <;>
    norm_num [wave, collisionFrequency, collisionCoefficient, Fin.sum_univ_two,
      character_two, Complex.normSq_apply]

/-- The flat counterexample has two nonzero coefficient magnitudes, CM20. -/
theorem collision_two_modes : collisionCoefficient 0 ≠ 0 ∧ collisionCoefficient 1 ≠ 0 := by
  constructor <;> norm_num [collisionCoefficient, Complex.ext_iff]

/-! Closed examples instantiate every conditional result. -/

example := wave_power tripleFrequency pureCoefficient (1 : ZMod 13)
example := powerCoefficient_eq tripleFrequency pureCoefficient (1 : ZMod 13)
example := pure_support 1 (by decide)
example := powerCoefficient_const tripleFrequency pureCoefficient pure_flat
  (by decide : (1 : ZMod 13) ≠ 0)
example := triple_separated.difference_ne_zero (by decide : (0 : Fin 3) ≠ 1)
example := triple_separated.coefficient pureCoefficient (by decide : (0 : Fin 3) ≠ 1)
example := flat_pair_zero triple_separated pureCoefficient pure_flat
  (by decide : (0 : Fin 3) ≠ 1)
example := wave_of_support tripleFrequency pureCoefficient 0 pure_support (1 : ZMod 13)
example := flat_extract triple_separated pureCoefficient
  ⟨0, by norm_num [pureCoefficient]⟩ pure_flat
example := centered_extract triple_separated pureCoefficient
  ⟨0, by norm_num [pureCoefficient]⟩ pure_flat pure_centered
example : Function.Injective (wave tripleFrequency pureCoefficient) := by
  have : Fact (Nat.Prime 13) := ⟨by decide⟩
  exact centered_injective triple_separated pureCoefficient
    ⟨0, by norm_num [pureCoefficient]⟩ pure_flat pure_centered
example := permuted_extract (Equiv.refl (ZMod 13)) triple_separated pureCoefficient
  ⟨0, by norm_num [pureCoefficient]⟩ pure_flat
example := support_flat tripleFrequency pureCoefficient 0 pure_support (1 : ZMod 13)
example := amplitude_not_convex tripleFrequency 0
example := powerCoefficient_sub tripleFrequency pureCoefficient 1
  (by decide : (1 : ZMod 13) ≠ 0)
example := pair_le_deviation triple_separated pureCoefficient 1
  (by decide : (0 : Fin 3) ≠ 1)
example := approximate_pair triple_separated pureCoefficient 1 0
  (by intro x; simp [pure_flat]) (by decide : (0 : Fin 3) ≠ 1)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction
