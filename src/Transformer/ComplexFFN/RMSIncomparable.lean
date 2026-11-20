module

public import Transformer.ComplexFFN.RMSTwoGate

/-!
# SwiGLU and the gated products stay incomparable after the branch RMS

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`): the branch
`f = R_D(W_F (g ⊙ u)/√2)` of `research/main_model/README.txt`, with the parameter-free
`R_D = GPTMini.rmsNormEps`, and the products of `Defs`.

With RMS `eps = 0` the branch RMS erases the one-unit witnesses of
`plainFFN_gap_swigluUnit`, `swigluFFN_gap_radial` and `swigluFFN_gap_diagonal`
(`RMSErased`): one SwiGLU unit after `R_D` is even, and a positive gate is a positive
scalar, so one radial or diagonal unit after `R_D` is its ungated product after `R_D`, a
SwiGLU block. Two units on each side bring the separations back.

* No bias-free SwiGLU FFN of any width, after `R_D`, approximates the two radial units of
  `radialFFN_twoGate` after `R_D` with error below the bound of
  `swigluFFN_rms_gap_radial` over `√2`, nor the two diagonal units below that of
  `swigluFFN_rms_gap_diagonal`. Both bounds are positive for `0 < β < α`, gate `ε ≥ 0` and
  RMS `eps ≥ 0` (`radial_rms_bound_pos`, `diagonal_rms_bound_pos`).
* No even map, in particular none of the plain, radial and diagonal FFNs after `R_D`,
  approximates the two SwiGLU units of `swigluFFN_pair` after `R_D` with error below the
  bound of `even_rms_gap_swigluPair` over `√2`, positive for `ρ z > 0`
  (`swigluPair_rms_bound_pos`).

The radial and diagonal targets have complex-linear gate and value projections when
`φ, ψ` are complex-linear, and any imaginary columns `c'`. The four points `αx ± βy`,
`βx ± αy` lie on one sphere when `x ⊥ y` and `‖x‖ = ‖y‖` (`four_points_mem_sphere`), as
for coordinate vectors of `ℂ^n`.
-/

@[expose] public section

open scoped BigOperators RealInnerProductSpace

namespace Transformer.ComplexFFN

variable {d : ℕ} {V ι : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]
variable {φ ψ : V →ₗ[ℝ] ℂ} {x y : V} {e e' : EucSpace d}

/-- The radial gate is even in a real argument. -/
theorem radialGate_neg_ofReal (ε t : ℝ) : radialGate ε ((-t : ℝ) : ℂ) = radialGate ε t :=
  radialGate_of_normSq_eq ε (by simp)

/-- After `R_D`, no bias-free SwiGLU FFN of any width approximates the two radial units
of `radialFFN_twoGate`, with orthonormal real columns `e, e'`, with error below this
bound over `√2`. -/
theorem swigluFFN_rms_gap_radial (eps ε : ℝ) (hφx : φ x = 1) (hφy : φ y = 0)
    (hψx : ψ x = 0) (hψy : ψ y = 1) (he : ‖e‖ = 1) (he' : ‖e'‖ = 1) (hee : ⟪e, e'⟫ = 0)
    (c' : Fin 2 → EucSpace d) (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → EucSpace d) {K : Set V}
    (hK : ∀ z ∈ K, -z ∈ K) {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖GPTMini.rmsNormEps eps (swigluFFN a b c z) -
      GPTMini.rmsNormEps eps (radialFFN ε ![φ, ψ] ![ψ, φ] ![e, e'] c' z)‖ ≤ δ)
    {α β : ℝ} (h₁ : α • x + β • y ∈ K) (h₂ : α • x - β • y ∈ K) (h₃ : β • x + α • y ∈ K)
    (h₄ : β • x - α • y ∈ K) :
    Real.sqrt d * ((Real.sqrt 2)⁻¹ * α * β * (radialGate ε α - radialGate ε β)) /
      Real.sqrt (((Real.sqrt 2)⁻¹ * α * β) ^ 2 * (radialGate ε α ^ 2 + radialGate ε β ^ 2) +
        d * eps) ≤ δ * Real.sqrt 2 :=
  swigluFFN_rms_gap_twoGate eps _ (s := fun t => radialGate ε t) (radialGate_neg_ofReal ε)
    (radialFFN_neg ε _ _ _ _) he he' hee
    (fun p q => (radialFFN_twoGate ε hφx hφy hψx hψy _ c' p q).trans
      (by simp only [Matrix.cons_val_zero, Matrix.cons_val_one]))
    a b c hK hδ h₁ h₂ h₃ h₄

/-- The same for the two diagonal units of `diagonalFFN_twoGate`. -/
theorem swigluFFN_rms_gap_diagonal (eps ε : ℝ) (hφx : φ x = 1) (hφy : φ y = 0)
    (hψx : ψ x = 0) (hψy : ψ y = 1) (he : ‖e‖ = 1) (he' : ‖e'‖ = 1) (hee : ⟪e, e'⟫ = 0)
    (c' : Fin 2 → EucSpace d) (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → EucSpace d) {K : Set V}
    (hK : ∀ z ∈ K, -z ∈ K) {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖GPTMini.rmsNormEps eps (swigluFFN a b c z) -
      GPTMini.rmsNormEps eps (diagonalFFN ε ![φ, ψ] ![ψ, φ] ![e, e'] c' z)‖ ≤ δ)
    {α β : ℝ} (h₁ : α • x + β • y ∈ K) (h₂ : α • x - β • y ∈ K) (h₃ : β • x + α • y ∈ K)
    (h₄ : β • x - α • y ∈ K) :
    Real.sqrt d * (α * β * (radialGate ε α - radialGate ε β)) /
      Real.sqrt ((α * β) ^ 2 * (radialGate ε α ^ 2 + radialGate ε β ^ 2) + d * eps) ≤
        δ * Real.sqrt 2 := by
  have h := swigluFFN_rms_gap_twoGate eps 1 (s := fun t => radialGate ε t)
    (radialGate_neg_ofReal ε) (diagonalFFN_neg ε _ _ _ _) he he' hee
    (fun p q => (diagonalFFN_twoGate ε hφx hφy hψx hψy _ c' p q).trans
      (by simp only [Matrix.cons_val_zero, Matrix.cons_val_one, one_mul]))
    a b c hK hδ h₁ h₂ h₃ h₄
  simpa only [one_mul] using h

/-- The radial bound is positive for `0 < β < α`, gate `ε ≥ 0` and RMS `eps ≥ 0`. -/
theorem radial_rms_bound_pos (eps ε : ℝ) (hd : 0 < d) (heps : 0 ≤ eps) (hε : 0 ≤ ε)
    {α β : ℝ} (hβ : 0 < β) (hαβ : β < α) :
    0 < Real.sqrt d * ((Real.sqrt 2)⁻¹ * α * β * (radialGate ε α - radialGate ε β)) /
      Real.sqrt (((Real.sqrt 2)⁻¹ * α * β) ^ 2 * (radialGate ε α ^ 2 + radialGate ε β ^ 2) +
        d * eps) :=
  twoGate_bound_pos hd heps (inv_pos.2 (Real.sqrt_pos.2 two_pos)) (hβ.trans hαβ) hβ
    (Real.sigmoid_pos _) (radialGate_lt ε hε (by simpa using mul_self_lt_mul_self hβ.le hαβ))

/-- The diagonal bound is positive under the same conditions. -/
theorem diagonal_rms_bound_pos (eps ε : ℝ) (hd : 0 < d) (heps : 0 ≤ eps) (hε : 0 ≤ ε)
    {α β : ℝ} (hβ : 0 < β) (hαβ : β < α) :
    0 < Real.sqrt d * (α * β * (radialGate ε α - radialGate ε β)) /
      Real.sqrt ((α * β) ^ 2 * (radialGate ε α ^ 2 + radialGate ε β ^ 2) + d * eps) := by
  simpa only [one_mul] using twoGate_bound_pos (κ := 1) (sα := radialGate ε α)
    (sβ := radialGate ε β) hd heps one_pos (hβ.trans hαβ) hβ
    (Real.sigmoid_pos _) (radialGate_lt ε hε (by simpa using mul_self_lt_mul_self hβ.le hαβ))

/-- After `R_D`, every even map misses the two SwiGLU units of `swigluFFN_pair` on a set
holding `±z` by this bound over `√2`. The plain, radial and diagonal FFNs after `R_D` are
even (`plainFFN_neg`, `radialFFN_neg`, `diagonalFFN_neg`). -/
theorem even_rms_gap_swigluPair (eps : ℝ) {f : V → EucSpace d} (hf : ∀ z, f (-z) = f z)
    (ρ : V →ₗ[ℝ] ℝ) (he : ‖e‖ = 1) (he' : ‖e'‖ = 1) (hee : ⟪e, e'⟫ = 0) {K : Set V}
    {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖f z - GPTMini.rmsNormEps eps (swigluFFN ![ρ, -ρ] ![ρ, -ρ] ![e, e'] z)‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hz' : -z ∈ K) :
    Real.sqrt d * (ρ z ^ 2 * (Real.sigmoid (ρ z) - Real.sigmoid (-ρ z))) /
      Real.sqrt ((ρ z ^ 2) ^ 2 * (Real.sigmoid (ρ z) ^ 2 + Real.sigmoid (-ρ z) ^ 2) +
        d * eps) ≤ δ * Real.sqrt 2 := by
  have hg := gap_of_invariant (σ := Neg.neg) hf hδ hz hz'
  have h1 := (real_inner_le_norm (e - e') _).trans
    (mul_le_mul_of_nonneg_left hg (norm_nonneg (e - e')))
  rw [norm_sub_of_orthonormal he he' hee, inner_sub_right, swigluFFN_pair, swigluFFN_pair,
    map_neg ρ z, neg_neg, neg_sq, inner_sub_rmsNormEps_pair eps he he' hee,
    inner_sub_rmsNormEps_pair eps he he' hee,
    show (ρ z ^ 2 * Real.sigmoid (-ρ z)) ^ 2 + (ρ z ^ 2 * Real.sigmoid (ρ z)) ^ 2 =
      (ρ z ^ 2) ^ 2 * (Real.sigmoid (ρ z) ^ 2 + Real.sigmoid (-ρ z) ^ 2) by ring,
    show (ρ z ^ 2 * Real.sigmoid (ρ z)) ^ 2 + (ρ z ^ 2 * Real.sigmoid (-ρ z)) ^ 2 =
      (ρ z ^ 2) ^ 2 * (Real.sigmoid (ρ z) ^ 2 + Real.sigmoid (-ρ z) ^ 2) by ring] at h1
  have h2 : Real.sqrt d * (ρ z ^ 2 * Real.sigmoid (ρ z) - ρ z ^ 2 * Real.sigmoid (-ρ z)) /
        Real.sqrt ((ρ z ^ 2) ^ 2 * (Real.sigmoid (ρ z) ^ 2 + Real.sigmoid (-ρ z) ^ 2) +
          d * eps) -
      Real.sqrt d * (ρ z ^ 2 * Real.sigmoid (-ρ z) - ρ z ^ 2 * Real.sigmoid (ρ z)) /
        Real.sqrt ((ρ z ^ 2) ^ 2 * (Real.sigmoid (ρ z) ^ 2 + Real.sigmoid (-ρ z) ^ 2) +
          d * eps) =
      2 * (Real.sqrt d * (ρ z ^ 2 * (Real.sigmoid (ρ z) - Real.sigmoid (-ρ z))) /
        Real.sqrt ((ρ z ^ 2) ^ 2 * (Real.sigmoid (ρ z) ^ 2 + Real.sigmoid (-ρ z) ^ 2) +
          d * eps)) := by ring
  linarith

/-- The SwiGLU-pair bound is positive for `ρ z > 0` and `eps ≥ 0`. -/
theorem swigluPair_rms_bound_pos (eps : ℝ) (hd : 0 < d) (heps : 0 ≤ eps) {t : ℝ}
    (ht : 0 < t) :
    0 < Real.sqrt d * (t ^ 2 * (Real.sigmoid t - Real.sigmoid (-t))) /
      Real.sqrt ((t ^ 2) ^ 2 * (Real.sigmoid t ^ 2 + Real.sigmoid (-t) ^ 2) + d * eps) := by
  have h1 := Real.sigmoid_pos t
  have h2 : Real.sigmoid (-t) < Real.sigmoid t := Real.sigmoid_lt (by linarith)
  have h3 : 0 < (t ^ 2) ^ 2 * (Real.sigmoid t ^ 2 + Real.sigmoid (-t) ^ 2) := by positivity
  have h4 : 0 ≤ (d : ℝ) * eps := by positivity
  exact div_pos (mul_pos (Real.sqrt_pos.2 (Nat.cast_pos.2 hd))
    (mul_pos (by positivity) (sub_pos.2 h2))) (Real.sqrt_pos.2 (by linarith))

section Examples

/-- The coordinates of `ℂ²` as gate and value forms, the unit vectors of `ℝ²` as `W_F`,
an arbitrary one-unit SwiGLU FFN, `δ = 2√2` from `norm_rmsNormEps_sub_le`. -/
example (a b : ℂ × ℂ →ₗ[ℝ] ℝ) (w : EucSpace 2) :=
  swigluFFN_rms_gap_radial 1 (2 ^ (-23 : ℤ)) (φ := LinearMap.fst ℝ ℂ ℂ)
    (ψ := LinearMap.snd ℝ ℂ ℂ) (x := (1, 0)) (y := (0, 1)) rfl rfl rfl rfl
    single_orthonormal.1 single_orthonormal.2.1 single_orthonormal.2.2 ![0, 0]
    (fun _ : Fin 1 => a) (fun _ => b) (fun _ => w) (K := Set.univ) (fun _ _ => trivial)
    (fun _ _ => norm_rmsNormEps_sub_le one_pos two_pos _ _) (α := 2) (β := 1)
    trivial trivial trivial trivial

example (a b : ℂ × ℂ →ₗ[ℝ] ℝ) (w : EucSpace 2) :=
  swigluFFN_rms_gap_diagonal 1 (2 ^ (-23 : ℤ)) (φ := LinearMap.fst ℝ ℂ ℂ)
    (ψ := LinearMap.snd ℝ ℂ ℂ) (x := (1, 0)) (y := (0, 1)) rfl rfl rfl rfl
    single_orthonormal.1 single_orthonormal.2.1 single_orthonormal.2.2 ![0, 0]
    (fun _ : Fin 1 => a) (fun _ => b) (fun _ => w) (K := Set.univ) (fun _ _ => trivial)
    (fun _ _ => norm_rmsNormEps_sub_le one_pos two_pos _ _) (α := 2) (β := 1)
    trivial trivial trivial trivial

/-- The kernels' `ε = 2^-23` and the README's RMS `eps = 2^-23`, at `D = 128`. -/
example := radial_rms_bound_pos (d := 128) (2 ^ (-23 : ℤ)) (2 ^ (-23 : ℤ)) (by norm_num)
  (by positivity) (by positivity) one_pos one_lt_two

example := diagonal_rms_bound_pos (d := 128) (2 ^ (-23 : ℤ)) (2 ^ (-23 : ℤ)) (by norm_num)
  (by positivity) (by positivity) one_pos one_lt_two

/-- The radial FFN after `R_D` against the SwiGLU pair on `ℝ`, `δ = 2√2`. -/
example (g u : Fin 3 → ℝ →ₗ[ℝ] ℂ) (c₁ c₂ : Fin 3 → EucSpace 2) :=
  even_rms_gap_swigluPair 1 (f := fun z => GPTMini.rmsNormEps 1 (radialFFN 0 g u c₁ c₂ z))
    (fun z => congrArg _ (radialFFN_neg 0 g u c₁ c₂ z)) LinearMap.id single_orthonormal.1
    single_orthonormal.2.1 single_orthonormal.2.2 (K := Set.univ)
    (fun _ _ => norm_rmsNormEps_sub_le one_pos two_pos _ _) (z := 1) trivial trivial

example := swigluPair_rms_bound_pos (d := 128) (2 ^ (-23 : ℤ)) (by norm_num) (by positivity)
  one_pos

end Examples

end Transformer.ComplexFFN
