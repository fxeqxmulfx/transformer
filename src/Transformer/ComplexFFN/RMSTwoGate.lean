module

public import Transformer.ComplexFFN.RMSGap

/-!
# Two gates turn the output direction

Source: the blocks of `Defs` (DiffGPT `5a37c51`).

Let `φ x = ψ y = 1` and `φ y = ψ x = 0`. Two radial or diagonal units with gates `φ, ψ`
and values `ψ, φ` have the same product `φ z ψ z` and different gates. At `z = p x + q y` they
write `κ p q (s(p) e + s(q) e')` through the real columns `e, e'` of `W_F`
(`radialFFN_twoGate` with `κ = 1/√2`, `diagonalFFN_twoGate` with `κ = 1`). The gate
ratio `s(p)/s(q)` turns the direction of the output, and the branch RMS keeps
directions. For orthonormal `e, e'` and `s(β) < s(α)`, the normalized output read along
`e - e'` has the signs `+, -, -, +` at `αx + βy`, `αx - βy`, `βx + αy`, `βx - αy`. So
`swigluFFN_rms_gap` gives an explicit gap (`swigluFFN_rms_gap_twoGate`), positive for
`0 < β < α` (`twoGate_bound_pos`).

Two SwiGLU units `SiLU(ρz)(ρz) e + SiLU(-ρz)(-ρz) e'` are
`(ρz)² (σ(ρz) e + σ(-ρz) e')` (`swigluFFN_pair`): `z ↦ -z` swaps the coefficients.
-/

@[expose] public section

open scoped BigOperators RealInnerProductSpace

namespace Transformer.ComplexFFN

variable {d : ℕ} {V ι : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]

section Values

variable {W : Type*} [AddCommGroup W] [Module ℝ W] {φ ψ : V →ₗ[ℝ] ℂ} {x y : V}

/-- A form with `φ x = 1`, `φ y = 0` reads the first coordinate of `p x + q y`. -/
theorem map_twoGate (hx : φ x = 1) (hy : φ y = 0) (p q : ℝ) : φ (p • x + q • y) = p := by
  rw [map_add, map_smul, map_smul, hx, hy, smul_zero, add_zero, Complex.real_smul, mul_one]

example (p q : ℝ) := map_twoGate (φ := LinearMap.fst ℝ ℂ ℂ) (x := ((1 : ℂ), (0 : ℂ)))
  (y := (0, 1)) rfl rfl p q

/-- Two radial units, gates `φ, ψ`, values `ψ, φ`: at `p x + q y` the hidden vector is
real and the output is `(pq/√2)(s(p) c₀ + s(q) c₁)`. -/
theorem radialFFN_twoGate (ε : ℝ) (hφx : φ x = 1) (hφy : φ y = 0) (hψx : ψ x = 0)
    (hψy : ψ y = 1) (c c' : Fin 2 → W) (p q : ℝ) :
    radialFFN ε ![φ, ψ] ![ψ, φ] c c' (p • x + q • y) =
      ((Real.sqrt 2)⁻¹ * p * q * radialGate ε p) • c 0 +
        ((Real.sqrt 2)⁻¹ * p * q * radialGate ε q) • c 1 := by
  have h₂ : ψ (p • x + q • y) = q := by
    rw [add_comm]
    exact map_twoGate hψy hψx q p
  have hr : ∀ r u v : ℝ, (r : ℂ) * ((u : ℂ) * v) / (Real.sqrt 2 : ℂ) =
      (((Real.sqrt 2)⁻¹ * u * v * r : ℝ) : ℂ) := fun r u v => by
    push_cast
    ring
  simp only [radialFFN, complexReadout, Fin.sum_univ_two, radialHidden, Matrix.cons_val_zero,
    Matrix.cons_val_one, map_twoGate hφx hφy, h₂, hr, Complex.ofReal_re, Complex.ofReal_im,
    zero_smul, add_zero]
  module

/-- The same two units with the diagonal product: `pq (s(p) c₀ + s(q) c₁)`. -/
theorem diagonalFFN_twoGate (ε : ℝ) (hφx : φ x = 1) (hφy : φ y = 0) (hψx : ψ x = 0)
    (hψy : ψ y = 1) (c c' : Fin 2 → W) (p q : ℝ) :
    diagonalFFN ε ![φ, ψ] ![ψ, φ] c c' (p • x + q • y) =
      (p * q * radialGate ε p) • c 0 + (p * q * radialGate ε q) • c 1 := by
  have h₂ : ψ (p • x + q • y) = q := by
    rw [add_comm]
    exact map_twoGate hψy hψx q p
  simp only [diagonalFFN, complexReadout, Fin.sum_univ_two, diagonalHidden,
    Matrix.cons_val_zero, Matrix.cons_val_one, map_twoGate hφx hφy, h₂, Complex.ofReal_re,
    Complex.ofReal_im, Complex.re_ofReal_mul, Complex.im_ofReal_mul, mul_zero, zero_smul,
    add_zero]
  module

/-- Two SwiGLU units whose gate and value change sign together:
`SiLU(ρz)(ρz) e + SiLU(-ρz)(-ρz) e' = (ρz)² (σ(ρz) e + σ(-ρz) e')`. -/
theorem swigluFFN_pair (ρ : V →ₗ[ℝ] ℝ) (e e' : W) (z : V) :
    swigluFFN ![ρ, -ρ] ![ρ, -ρ] ![e, e'] z =
      (ρ z ^ 2 * Real.sigmoid (ρ z)) • e + (ρ z ^ 2 * Real.sigmoid (-ρ z)) • e' := by
  simp only [swigluFFN, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    LinearMap.neg_apply, silu_eq_mul_sigmoid]
  module

example (z : ℂ × ℂ) := radialFFN_twoGate (W := ℂ) 0 (φ := LinearMap.fst ℝ ℂ ℂ)
  (ψ := LinearMap.snd ℝ ℂ ℂ) (x := (1, 0)) (y := (0, 1)) rfl rfl rfl rfl ![1, 1] ![0, 0]
  z.1.re z.2.re
example (z : ℂ × ℂ) := diagonalFFN_twoGate (W := ℂ) 0 (φ := LinearMap.fst ℝ ℂ ℂ)
  (ψ := LinearMap.snd ℝ ℂ ℂ) (x := (1, 0)) (y := (0, 1)) rfl rfl rfl rfl ![1, 1] ![0, 0]
  z.1.re z.2.re

end Values

/-- The common step: an even `T` that writes `κ p q (s(p) e + s(q) e')` at `p x + q y`,
with an even gate `s`, is missed after `R_D` by every bias-free SwiGLU FFN by the
returned bound over `√2`. -/
theorem swigluFFN_rms_gap_twoGate (eps κ : ℝ) {s : ℝ → ℝ} (hs : ∀ t, s (-t) = s t)
    {T : V → EucSpace d} (hT : ∀ z, T (-z) = T z) {x y : V} {e e' : EucSpace d}
    (he : ‖e‖ = 1) (he' : ‖e'‖ = 1) (hee : ⟪e, e'⟫ = 0)
    (hTv : ∀ p q : ℝ, T (p • x + q • y) = (κ * p * q * s p) • e + (κ * p * q * s q) • e')
    (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → EucSpace d) {K : Set V} (hK : ∀ z ∈ K, -z ∈ K) {δ : ℝ}
    (hδ : ∀ z ∈ K,
      ‖GPTMini.rmsNormEps eps (swigluFFN a b c z) - GPTMini.rmsNormEps eps (T z)‖ ≤ δ)
    {α β : ℝ} (h₁ : α • x + β • y ∈ K) (h₂ : α • x - β • y ∈ K) (h₃ : β • x + α • y ∈ K)
    (h₄ : β • x - α • y ∈ K) :
    Real.sqrt d * (κ * α * β * (s α - s β)) /
      Real.sqrt ((κ * α * β) ^ 2 * (s α ^ 2 + s β ^ 2) + d * eps) ≤ δ * Real.sqrt 2 := by
  have hv : ∀ p q : ℝ, ⟪e - e', GPTMini.rmsNormEps eps (T (p • x + q • y))⟫ =
      Real.sqrt d * (κ * p * q * (s p - s q)) /
        Real.sqrt ((κ * p * q) ^ 2 * (s p ^ 2 + s q ^ 2) + d * eps) := fun p q => by
    rw [hTv, inner_sub_rmsNormEps_pair eps he he' hee,
      show κ * p * q * s p - κ * p * q * s q = κ * p * q * (s p - s q) by ring,
      show (κ * p * q * s p) ^ 2 + (κ * p * q * s q) ^ 2 =
        (κ * p * q) ^ 2 * (s p ^ 2 + s q ^ 2) by ring]
  have e₂ : α • x - β • y = α • x + (-β) • y := by rw [neg_smul, sub_eq_add_neg]
  have e₄ : β • x - α • y = β • x + (-α) • y := by rw [neg_smul, sub_eq_add_neg]
  have hR : ∀ u v : ℝ, (κ * u * v) ^ 2 * (s u ^ 2 + s v ^ 2) =
      (κ * v * u) ^ 2 * (s v ^ 2 + s u ^ 2) := fun u v => by ring
  rw [← norm_sub_of_orthonormal he he' hee]
  refine swigluFFN_rms_gap (w := fun z => GPTMini.rmsNormEps eps (T z))
    (fun z => congrArg _ (hT z)) eps a b c hK hδ (e - e') h₁ h₂ h₃ h₄ (hv α β).ge ?_ ?_ ?_
  · show _ ≤ -⟪e - e', GPTMini.rmsNormEps eps (T (α • x - β • y))⟫
    rw [e₂, hv, hs, show (κ * α * -β) ^ 2 = (κ * α * β) ^ 2 by ring]
    exact le_of_eq (by ring)
  · show _ ≤ -⟪e - e', GPTMini.rmsNormEps eps (T (β • x + α • y))⟫
    rw [hv, hR β α]
    exact le_of_eq (by ring)
  · show _ ≤ ⟪e - e', GPTMini.rmsNormEps eps (T (β • x - α • y))⟫
    rw [e₄, hv, hs, show (κ * β * -α) ^ 2 * (s β ^ 2 + s α ^ 2) =
      (κ * α * β) ^ 2 * (s α ^ 2 + s β ^ 2) by ring]
    exact le_of_eq (by ring)

/-- The product target `z₁ z₂ (e₀ + e₁)` on `ℝ²` with constant `s`, `δ = 2√2`: there the
bound is `0`. -/
example (a b : Fin 1 → ℝ × ℝ →ₗ[ℝ] ℝ) (c : Fin 1 → EucSpace 2) (α β : ℝ) :=
  swigluFFN_rms_gap_twoGate 1 1 (s := fun _ => 1) (fun _ => rfl)
    (T := fun z : ℝ × ℝ =>
      (z.1 * z.2) • (EuclideanSpace.single 0 1 + EuclideanSpace.single 1 1))
    (fun z => by simp) (x := (1, 0)) (y := (0, 1)) single_orthonormal.1
    single_orthonormal.2.1 single_orthonormal.2.2 (fun p q => by simp [smul_add]) a b c
    (K := Set.univ) (fun _ _ => trivial)
    (fun _ _ => norm_rmsNormEps_sub_le one_pos two_pos _ _) (α := α) (β := β)
    trivial trivial trivial trivial

/-- The bound of `swigluFFN_rms_gap_twoGate` is positive when the gate at `α` exceeds
the gate at `β > 0`. -/
theorem twoGate_bound_pos {eps κ α β sα sβ : ℝ} (hd : 0 < d) (heps : 0 ≤ eps) (hκ : 0 < κ)
    (hα : 0 < α) (hβ : 0 < β) (hsβ : 0 < sβ) (hs : sβ < sα) :
    0 < Real.sqrt d * (κ * α * β * (sα - sβ)) /
      Real.sqrt ((κ * α * β) ^ 2 * (sα ^ 2 + sβ ^ 2) + d * eps) := by
  have h1 : 0 < κ * α * β := by positivity
  have h2 : 0 < (κ * α * β) ^ 2 * (sα ^ 2 + sβ ^ 2) := by positivity
  have h3 : 0 ≤ (d : ℝ) * eps := by positivity
  exact div_pos (mul_pos (Real.sqrt_pos.2 (Nat.cast_pos.2 hd)) (mul_pos h1 (sub_pos.2 hs)))
    (Real.sqrt_pos.2 (by linarith))

example : 0 < Real.sqrt (2 : ℕ) * (1 * 2 * 1 * (2 - 1)) /
    Real.sqrt ((1 * 2 * 1) ^ 2 * (2 ^ 2 + 1 ^ 2) + (2 : ℕ) * 0) :=
  twoGate_bound_pos two_pos le_rfl one_pos two_pos one_pos one_pos one_lt_two

end Transformer.ComplexFFN
