/-
# Homogenized Transformers — immediate API of the model

Simp lemmas and elementary consequences of the definitions of
`Transformer.Homogenized.Defs`, all for arXiv:2604.01978v1,
*Homogenized Transformers* (Geshkovski, Koubbi, Rigollet).

The last section is the witness the statements downstream use: at the head
`θ = (V, A) = (0, 0)` the attention field vanishes, so under the law
`ρ* = δ_0` every field of the model is identically zero and every tuple of
unit vectors is a stationary configuration of `eq:update_tokens`.
-/

import Transformer.Homogenized.Defs

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- At `β = 0` attention is uniform and the field is the plain average
`(1/n) Σ_k V x_k` of the values.  (At `n = 0` both sides are `0`, so no
positivity hypothesis is needed.) -/
theorem attnField_zero_beta {d n : ℕ} (θ : HeadParam d)
    (x : Idx n → EucSpace d) (z : EucSpace d) :
    attnField 0 θ x z = ((n : ℝ))⁻¹ • ∑ k : Idx n, valueMap θ (x k) := by
  have hw : ∀ k : Idx n, attnWeight 0 θ z (x k) = 1 := by
    intro k; simp [attnWeight]
  simp only [attnField, hw, one_smul, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, mul_one]
/-- `N` lands on the sphere away from the origin. -/
theorem norm_normalizeLayer {d : ℕ} {v : EucSpace d} (hv : v ≠ 0) :
    ‖normalizeLayer v‖ = 1 :=
  norm_smul_inv_norm (𝕜 := ℝ) hv

/-- The hypothesis of `norm_normalizeLayer` is satisfiable. -/
example : (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ≠ 0 := by
  intro h
  have : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by simp [PiLp.norm_single]
  rw [h] at this
  simp at this
/-- `N` fixes the sphere. -/
@[simp]
theorem normalizeLayer_of_norm_one {d : ℕ} {v : EucSpace d} (hv : ‖v‖ = 1) :
    normalizeLayer v = v := by
  simp [normalizeLayer, hv]

/-- The hypothesis of `normalizeLayer_of_norm_one` is satisfiable. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by simp [PiLp.norm_single]
/-- At a grid time `t = ℓη` the interpolation `X^η(t) = X^{⌊t/η⌋}` of
`eq:update_tokens` is the `ℓ`-th layer of the chain. -/
@[simp]
theorem interpChain_natCast_mul {d n : ℕ} {η : ℝ} (hη : 0 < η)
    (X : ℕ → Idx n → EucSpace d) (l : ℕ) :
    interpChain η X ((l : ℝ) * η) = X l := by
  rw [interpChain, mul_div_assoc, div_self hη.ne', mul_one, Nat.floor_natCast]

/-- The hypothesis of `interpChain_natCast_mul` is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

/-- The variance proxy is unique: two of them agree. -/
theorem IsVarianceProxy.unique {d n : ℕ} {β : ℝ} {ρ : Measure (HeadParam d)} {s s' : ℝ}
    (h : IsVarianceProxy d n β ρ s) (h' : IsVarianceProxy d n β ρ s') : s = s' := by
  have hsq : s ^ 2 = s' ^ 2 := h.2.unique h'.2
  nlinarith [h.1, h'.1]
/-! ### The trivial head

At `θ = (V, A) = (0, 0)` the attention field vanishes, so the law `ρ* = δ_0`
makes every field of the model vanish identically.  This is the witness that
the model's hypotheses are satisfiable — a Transformer whose weights are all
zero moves no token — and it is used as such by the statements downstream. -/

@[simp]
theorem valueMap_zero {d : ℕ} (y : EucSpace d) : valueMap (0 : HeadParam d) y = 0 := by
  simp [valueMap]

@[simp]
theorem attnField_zero_param {d n : ℕ} (β : ℝ) (x : Idx n → EucSpace d) (z : EucSpace d) :
    attnField β (0 : HeadParam d) x z = 0 := by
  simp [attnField]

@[simp]
theorem meanField_dirac_zero {d n : ℕ} (β : ℝ) (x : Idx n → EucSpace d) (z : EucSpace d) :
    meanField β (Measure.dirac (0 : HeadParam d)) x z = 0 := by
  simp [meanField]

@[simp]
theorem fluct_dirac_zero {d n : ℕ} (β : ℝ) (x : Idx n → EucSpace d) (z : EucSpace d) :
    fluct β (Measure.dirac (0 : HeadParam d)) (0 : HeadParam d) x z = 0 := by
  simp [fluct]

@[simp]
theorem bField_dirac_zero {d n : ℕ} (β : ℝ) (x : Idx n → EucSpace d) (i : Idx n) :
    bField β (Measure.dirac (0 : HeadParam d)) x i = 0 := by
  simp [bField, proj]

@[simp]
theorem Gfield_dirac_zero {d n : ℕ} (β : ℝ) (x : Idx n → EucSpace d) (i : Idx n) :
    Gfield β (Measure.dirac (0 : HeadParam d)) x (0 : HeadParam d) i = 0 := by
  simp [Gfield, proj]

/-- `σ = 0` is the variance proxy of the trivial head. -/
theorem isVarianceProxy_dirac_zero (d n : ℕ) (β : ℝ) (x₀ : Idx (n + 1) → EucSpace (d + 1))
    (hx₀ : ∀ i, ‖x₀ i‖ = 1) :
    IsVarianceProxy (d + 1) (n + 1) β (Measure.dirac (0 : HeadParam (d + 1))) 0 := by
  refine ⟨le_rfl, ⟨⟨x₀, hx₀, ⟨0, ?_⟩⟩, ?_⟩⟩
  · simp
  · rintro v ⟨x, -, j, rfl⟩
    simp

/-- The hypothesis of `isVarianceProxy_dirac_zero` is satisfiable: the constant
tuple at `basePoint d`. -/
example (d : ℕ) : ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 := by
  simp [basePoint, PiLp.norm_single]

end Homogenized
end Transformer
