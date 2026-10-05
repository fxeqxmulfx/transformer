import Transformer.GPTMini.Sparsemax.QKProjection
import Mathlib.LinearAlgebra.Basis.VectorSpace
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# Input independence gives a proved linear coordinate decoder

Derived accessibility condition for arXiv:1602.02068v2, §2.5, and
the shared linear query/key projections in `Attention.forward` at
`73f8a0b`. Standard-basis inputs are sufficient but unnecessary.
Any finite linearly independent family admits a continuous linear
decoder taking its members to the corresponding coordinate vectors.
Conversely such a decoder proves the family's independence.

The existence statement is proved by a linear left inverse of the
actual synthesis map, not assumed as an architectural oracle. The
construction does not require the input vectors to be orthogonal or
to span their ambient input space. It still requires independence
of the whole considered family; this is a finite-row restriction.
It is not a claim that arbitrary learned embeddings have that rank.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall
open scoped BigOperators

/-- Independent inputs admit an actual continuous linear decoder.
Source: the derived input restriction for §2.5 of arXiv:1602.02068v2
and shared projections at `73f8a0b`; no basis or orthogonality is assumed. -/
theorem inputDecoder_exists {F T : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (hi : LinearIndependent ℝ inputs) :
    ∃ decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ), ∀ n, decoder (inputs n) = basis n := by
  classical
  let synthesis := Fintype.linearCombination ℝ inputs
  have hinj : synthesis.ker = ⊥ :=
    LinearMap.ker_eq_bot.mpr hi.fintypeLinearCombination_injective
  obtain ⟨inverse, hinverse⟩ := synthesis.exists_leftInverse_of_injective hinj
  refine ⟨LinearMap.toContinuousLinearMap inverse, ?_⟩
  intro n
  have hb : synthesis (basis n) = inputs n := by
    rw [Fintype.linearCombination_apply]
    simp [basis]
  have he := congrArg (fun f : (Fin T → ℝ) →ₗ[ℝ] (Fin T → ℝ) => f (basis n)) hinverse
  change inverse (synthesis (basis n)) = basis n at he
  rw [hb] at he
  exact he

/-- A decoder that separates all input coordinates proves independence.
Source: the same derived accessibility restriction for §2.5 of
arXiv:1602.02068v2 and the linear Q/K projections at `73f8a0b`. -/
theorem inputIndependent_of_decoder {F T : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (hd : ∀ n, decoder (inputs n) = basis n) : LinearIndependent ℝ inputs := by
  classical
  have hb : (fun n : Fin T => basis n) = Pi.basisFun ℝ (Fin T) := by
    funext n j
    simp [basis, Pi.basisFun_apply, Pi.single_apply]
  apply LinearIndependent.of_comp decoder.toLinearMap
  have he : (decoder.toLinearMap ∘ inputs) = fun n => basis n := by
    funext n
    exact hd n
  rw [he, hb]
  exact (Pi.basisFun ℝ (Fin T)).linearIndependent

/-- Input independence is exactly the existence of a linear coordinate decoder.
Source: the derived matrix-accessibility condition for §2.5 of
arXiv:1602.02068v2, rather than a hypothesis that the inputs are a basis. -/
theorem inputIndependent_iff_decoder {F T : ℕ} (inputs : Fin T → (Fin F → ℝ)) :
    LinearIndependent ℝ inputs ↔
      ∃ decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ), ∀ n, decoder (inputs n) = basis n := by
  constructor
  · exact inputDecoder_exists inputs
  · rintro ⟨decoder, hd⟩
    exact inputIndependent_of_decoder inputs decoder hd

/-- The whole-row independence condition requires enough input coordinates.
Source: the derived rank restriction for §2.5 of arXiv:1602.02068v2;
this records the limitation instead of hiding it inside decoder existence. -/
theorem inputIndependent_width_bound {F T : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (hi : LinearIndependent ℝ inputs) : T ≤ F := by
  simpa only [Fintype.card_fin, Module.finrank_pi] using hi.fintype_card_le_finrank

/-- A context longer than the input width cannot have a full coordinate decoder.
Source: the derived shared-projection rank condition for §2.5 of
arXiv:1602.02068v2; no statement about partial anchor decoders is made. -/
theorem inputDecoder_not_exists_of_width_lt {F T : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (hw : F < T) :
    ¬ ∃ decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ), ∀ n, decoder (inputs n) = basis n := by
  rintro ⟨decoder, hd⟩
  have hi := inputIndependent_of_decoder inputs decoder hd
  exact (not_le_of_gt hw) (inputIndependent_width_bound inputs hi)

/-- The width obstruction is inhabited by two nonzero inputs of width one.
Source context: §2.5's derived matrix accessibility restriction. -/
example : ¬ ∃ decoder : (Fin 1 → ℝ) →L[ℝ] (Fin 2 → ℝ),
    ∀ n, decoder (fun _ => (1 : ℝ)) = basis n :=
  inputDecoder_not_exists_of_width_lt (fun _ : Fin 2 => fun _ : Fin 1 => 1) (by norm_num)

/-- Nonorthogonal, nonstandard inputs with two independent coordinates.
Source context: the derived input restriction for §2.5 of arXiv:1602.02068v2. -/
def mixedInputs : Fin 2 → (Fin 2 → ℝ) :=
  fun n j => if n = j then 2 else 1

/-- The inverse of the concrete input matrix, as a continuous linear map.
Source context: linear Q/K projections at `73f8a0b`; the coefficients
invert the input matrix `[[2, 1], [1, 2]]`, whose determinant is three. -/
def mixedInputDecoder : (Fin 2 → ℝ) →L[ℝ] (Fin 2 → ℝ) :=
  ContinuousLinearMap.pi fun n =>
    (2 / 3 : ℝ) • ContinuousLinearMap.proj n -
      (1 / 3 : ℝ) • ContinuousLinearMap.proj (1 - n)

/-- The concrete inverse recovers both coordinate vectors exactly.
Source: the derived nonstandard input example for §2.5 of
arXiv:1602.02068v2 and the shared projection matrix at `73f8a0b`. -/
theorem mixedInputDecoder_coordinates : ∀ n, mixedInputDecoder (mixedInputs n) = basis n := by
  intro n
  funext j
  fin_cases n <;> fin_cases j <;>
    norm_num [mixedInputDecoder, mixedInputs, basis, ContinuousLinearMap.pi_apply,
      sub_apply, smul_apply, ContinuousLinearMap.proj_apply]

/-- The concrete inputs are independent without being standard basis vectors.
Source context: the derived matrix accessibility condition for §2.5 of
arXiv:1602.02068v2, witnessed by the explicit inverse above. -/
theorem mixedInputs_independent : LinearIndependent ℝ mixedInputs := by
  exact inputIndependent_of_decoder mixedInputs mixedInputDecoder mixedInputDecoder_coordinates

/-- The existence theorem's independence hypothesis has a nonstandard instance.
Source context: §2.5's derived shared-projection accessibility condition. -/
example : ∃ decoder : (Fin 2 → ℝ) →L[ℝ] (Fin 2 → ℝ),
    ∀ n, decoder (mixedInputs n) = basis n :=
  inputDecoder_exists mixedInputs mixedInputs_independent

/-- The converse's coordinate-decoding hypothesis is inhabited by the same inverse.
Source context: §2.5's derived shared-projection accessibility condition. -/
example : LinearIndependent ℝ mixedInputs :=
  inputIndependent_of_decoder mixedInputs mixedInputDecoder mixedInputDecoder_coordinates

/-- The independence width bound has the concrete nonstandard instance.
Source context: §2.5's derived projection rank restriction. -/
example : (2 : ℕ) ≤ 2 := inputIndependent_width_bound mixedInputs mixedInputs_independent

/-- Every concrete input differs from its standard coordinate vector.
Source context: the derived nonstandard example for §2.5 of
arXiv:1602.02068v2; the diagonal input entry is two rather than one. -/
theorem mixedInputs_ne_basis (n : Fin 2) : mixedInputs n ≠ basis n := by
  intro he
  have hh := congrFun he n
  norm_num [mixedInputs, basis] at hh

/-- The concrete inputs are also nonorthogonal in the ordinary coordinate dot product.
Source context: the derived §2.5 example; no orthogonality condition is hidden
in the decoder's independence premise. -/
theorem mixedInputs_dot : ∑ j : Fin 2, mixedInputs 0 j * mixedInputs 1 j = 4 := by
  norm_num [mixedInputs, Fin.sum_univ_two]

end Transformer.GPTMini.Sparsemax
