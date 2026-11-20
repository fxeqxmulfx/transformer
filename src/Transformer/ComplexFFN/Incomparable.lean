module

public import Transformer.ComplexFFN.EvenPart

/-!
# SwiGLU and the gated complex products are incomparable

Source: the blocks of `Defs` (DiffGPT `5a37c51`).

The radial and diagonal products are even, but their gate breaks the parallelogram law
that the even part of every bias-free SwiGLU FFN obeys (`EvenPart`). At `φ x = 1`,
`φ y = 0` the defect of one unit is `(s(1) - s(1/√2))/√2`, respectively
`s(1) - s(1/√2)`, positive for `ε ≥ 0` (`radialGate_inv_sqrt_two_lt`). So no SwiGLU FFN
of any width approximates one radial unit with error below a quarter of the first,
nor one diagonal unit below a quarter of the second (`swigluFFN_gap_radial`,
`swigluFFN_gap_diagonal`); for the kernels' `ε = 2^-23` these quarters evaluate
numerically to about `0.0108` and `0.0153`. With `Separation`, each class misses a
single unit of the other.

The plain product is not dominated at equal parameters either. One plain unit is
`φ₁ z · φ₂ z` exactly (`plainFFN_unit`), while one SwiGLU unit errs by at least `1`
wherever that product takes every value of modulus one (`swigluUnit_gap_product`). On
the torus of `ℂ²` the two units have ten real parameters each, and the SwiGLU unit is
no better than the zero map (`swigluUnit_gap_plainUnit_torus`).
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.ComplexFFN

variable {V ι : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]

/-- No bias-free SwiGLU FFN, of any width, approximates one radial unit with error
below `(s(1) - s(1/√2)) / (4√2)`, which is positive for `ε ≥ 0`. -/
theorem swigluFFN_gap_radial (ε : ℝ) (φ : V →ₗ[ℝ] ℂ) {x y : V} (hx : φ x = 1) (hy : φ y = 0)
    (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → ℂ) {K : Set V} {δ : ℝ} (hK : ∀ z ∈ K, -z ∈ K)
    (hδ : ∀ z ∈ K, ‖swigluFFN a b c z - radialFFN ε (fun _ : Unit => φ) (fun _ => φ)
      (fun _ => (1 : ℂ)) (fun _ => Complex.I) z‖ ≤ δ)
    (hxK : x ∈ K) (hyK : y ∈ K) (hpK : (Real.sqrt 2)⁻¹ • (x + y) ∈ K)
    (hmK : (Real.sqrt 2)⁻¹ • (x - y) ∈ K) :
    (radialGate ε 1 - radialGate ε ((Real.sqrt 2)⁻¹ : ℝ)) / Real.sqrt 2 ≤ 4 * δ := by
  have h := swigluFFN_gap_even (radialFFN_neg ε _ _ _ _) a b c hK hδ two_mul_inv_sqrt_two_sq
    hxK hyK hpK hmK
  obtain ⟨hp, hm⟩ := map_inv_sqrt_two_smul φ hx hy
  rw [radialFFN_unit, radialFFN_unit, radialFFN_unit, radialFFN_unit, hx, hy, hp, hm,
    inv_sqrt_two_mul_self] at h
  have hd : (radialGate ε 1 : ℂ) * (1 * 1) / (Real.sqrt 2 : ℂ) +
      (radialGate ε 0 : ℂ) * (0 * 0) / (Real.sqrt 2 : ℂ) -
      (radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) : ℂ) * (1 / 2) / (Real.sqrt 2 : ℂ) -
      (radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) : ℂ) * (1 / 2) / (Real.sqrt 2 : ℂ) =
      (((radialGate ε 1 - radialGate ε ((Real.sqrt 2)⁻¹ : ℝ)) / Real.sqrt 2 : ℝ) : ℂ) := by
    push_cast
    ring
  rw [hd, Complex.norm_real, Real.norm_eq_abs] at h
  exact (le_abs_self _).trans h

/-- The zero FFN on the slab `|z₁| ≤ 1` of `ℂ²`, with `φ = fst`, `x = (1, 0)`,
`y = (0, 1)` and `δ = 1`. -/
example (ε : ℝ) :
    (radialGate ε 1 - radialGate ε ((Real.sqrt 2)⁻¹ : ℝ)) / Real.sqrt 2 ≤ 4 * 1 := by
  obtain ⟨hp, hm⟩ := map_inv_sqrt_two_smul (LinearMap.fst ℝ ℂ ℂ)
    (x := ((1 : ℂ), (0 : ℂ))) (y := (0, 1)) rfl rfl
  refine swigluFFN_gap_radial ε (LinearMap.fst ℝ ℂ ℂ) (x := (1, 0)) (y := (0, 1)) rfl rfl
    (fun _ : Unit => 0) (fun _ => 0) (fun _ => 0) (K := {z | ‖LinearMap.fst ℝ ℂ ℂ z‖ ≤ 1})
    (δ := 1) (fun z hz => ?_) (fun z hz => ?_) (by simp) (by simp) ?_ ?_
  · have h : ‖LinearMap.fst ℝ ℂ ℂ z‖ ≤ 1 := hz
    rwa [Set.mem_ofPred_eq, map_neg, norm_neg]
  · simp only [swigluFFN, LinearMap.zero_apply, mul_zero, smul_zero, Finset.sum_const_zero,
      zero_sub, norm_neg]
    exact (norm_radialFFN_unit_le ε _ z).trans (pow_le_one₀ (norm_nonneg _) hz)
  · rw [Set.mem_ofPred_eq, hp]
    exact norm_inv_sqrt_two_le_one
  · rw [Set.mem_ofPred_eq, hm]
    exact norm_inv_sqrt_two_le_one

/-- No bias-free SwiGLU FFN, of any width, approximates one diagonal unit with error
below `(s(1) - s(1/√2)) / 4`, which is positive for `ε ≥ 0`. -/
theorem swigluFFN_gap_diagonal (ε : ℝ) (φ : V →ₗ[ℝ] ℂ) {x y : V} (hx : φ x = 1)
    (hy : φ y = 0) (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → ℂ) {K : Set V} {δ : ℝ}
    (hK : ∀ z ∈ K, -z ∈ K)
    (hδ : ∀ z ∈ K, ‖swigluFFN a b c z - diagonalFFN ε (fun _ : Unit => φ) (fun _ => φ)
      (fun _ => (1 : ℂ)) (fun _ => Complex.I) z‖ ≤ δ)
    (hxK : x ∈ K) (hyK : y ∈ K) (hpK : (Real.sqrt 2)⁻¹ • (x + y) ∈ K)
    (hmK : (Real.sqrt 2)⁻¹ • (x - y) ∈ K) :
    radialGate ε 1 - radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) ≤ 4 * δ := by
  have h := swigluFFN_gap_even (diagonalFFN_neg ε _ _ _ _) a b c hK hδ two_mul_inv_sqrt_two_sq
    hxK hyK hpK hmK
  obtain ⟨hp, hm⟩ := map_inv_sqrt_two_smul φ hx hy
  rw [diagonalFFN_unit, diagonalFFN_unit, diagonalFFN_unit, diagonalFFN_unit, hx, hy, hp, hm] at h
  have ht : (Real.sqrt 2)⁻¹ * (Real.sqrt 2)⁻¹ = 1 / 2 := by
    linarith [two_mul_inv_sqrt_two_sq, sq ((Real.sqrt 2)⁻¹)]
  have hd : (radialGate ε 1 : ℂ) * ⟨(1 : ℂ).re * (1 : ℂ).re, (1 : ℂ).im * (1 : ℂ).im⟩ +
      (radialGate ε 0 : ℂ) * ⟨(0 : ℂ).re * (0 : ℂ).re, (0 : ℂ).im * (0 : ℂ).im⟩ -
      (radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) : ℂ) *
        ⟨(((Real.sqrt 2)⁻¹ : ℝ) : ℂ).re * (((Real.sqrt 2)⁻¹ : ℝ) : ℂ).re,
          (((Real.sqrt 2)⁻¹ : ℝ) : ℂ).im * (((Real.sqrt 2)⁻¹ : ℝ) : ℂ).im⟩ -
      (radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) : ℂ) *
        ⟨(((Real.sqrt 2)⁻¹ : ℝ) : ℂ).re * (((Real.sqrt 2)⁻¹ : ℝ) : ℂ).re,
          (((Real.sqrt 2)⁻¹ : ℝ) : ℂ).im * (((Real.sqrt 2)⁻¹ : ℝ) : ℂ).im⟩ =
      ((radialGate ε 1 - radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) : ℝ) : ℂ) := by
    apply Complex.ext <;>
      simp only [Complex.add_re, Complex.sub_re, Complex.mul_re, Complex.add_im,
        Complex.sub_im, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im, Complex.one_re,
        Complex.one_im, Complex.zero_re, Complex.zero_im, ht] <;> ring
  rw [hd, Complex.norm_real, Real.norm_eq_abs] at h
  exact (le_abs_self _).trans h

/-- The zero FFN on the slab `|z₁| ≤ 1` of `ℂ²`, as for the radial unit. -/
example (ε : ℝ) : radialGate ε 1 - radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) ≤ 4 * 1 := by
  obtain ⟨hp, hm⟩ := map_inv_sqrt_two_smul (LinearMap.fst ℝ ℂ ℂ)
    (x := ((1 : ℂ), (0 : ℂ))) (y := (0, 1)) rfl rfl
  refine swigluFFN_gap_diagonal ε (LinearMap.fst ℝ ℂ ℂ) (x := (1, 0)) (y := (0, 1)) rfl rfl
    (fun _ : Unit => 0) (fun _ => 0) (fun _ => 0) (K := {z | ‖LinearMap.fst ℝ ℂ ℂ z‖ ≤ 1})
    (δ := 1) (fun z hz => ?_) (fun z hz => ?_) (by simp) (by simp) ?_ ?_
  · have h : ‖LinearMap.fst ℝ ℂ ℂ z‖ ≤ 1 := hz
    rwa [Set.mem_ofPred_eq, map_neg, norm_neg]
  · simp only [swigluFFN, LinearMap.zero_apply, mul_zero, smul_zero, Finset.sum_const_zero,
      zero_sub, norm_neg]
    exact (norm_diagonalFFN_unit_le ε _ z).trans (pow_le_one₀ (norm_nonneg _) hz)
  · rw [Set.mem_ofPred_eq, hp]
    exact norm_inv_sqrt_two_le_one
  · rw [Set.mem_ofPred_eq, hm]
    exact norm_inv_sqrt_two_le_one

/-- One SwiGLU unit misses a product `φ₁ z · φ₂ z` by at least `1` wherever the
product takes every value of modulus one; one plain unit is exact (`plainFFN_unit`). -/
theorem swigluUnit_gap_product (φ₁ φ₂ : V →ₗ[ℝ] ℂ) (a b : V →ₗ[ℝ] ℝ) (c₀ : ℂ)
    {K : Set V} {δ : ℝ} (hK : ∀ w : ℂ, ‖w‖ = 1 → ∃ z ∈ K, φ₁ z * φ₂ z = w)
    (hδ : ∀ z ∈ K, ‖swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => c₀) z -
      φ₁ z * φ₂ z‖ ≤ δ) : 1 ≤ δ := by
  have hS : ∀ z, swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => c₀) z =
      ((Transformer.Landscape.Neural.silu (a z) * b z : ℝ) : ℂ) * c₀ := fun z => by
    simp only [swigluFFN, Fintype.sum_unique, Complex.real_smul]
  by_cases hc : c₀ = 0
  · obtain ⟨z, hz, hzw⟩ := hK 1 (by simp)
    have h := hδ z hz
    rwa [hS, hzw, hc, mul_zero, zero_sub, norm_neg, norm_one] at h
  · have hn : 0 < ‖c₀‖ := norm_pos_iff.2 hc
    obtain ⟨z, hz, hzw⟩ := hK (Complex.I * c₀ / (‖c₀‖ : ℂ)) (by
      rw [norm_div, Complex.norm_mul, Complex.norm_I, one_mul, Complex.norm_real,
        Real.norm_eq_abs, abs_of_pos hn, div_self hn.ne'])
    have h := hδ z hz
    set r := Transformer.Landscape.Neural.silu (a z) * b z
    have hfac : (r : ℂ) * c₀ - Complex.I * c₀ / (‖c₀‖ : ℂ) =
        c₀ * ((r : ℂ) - Complex.I / (‖c₀‖ : ℂ)) := by ring
    rw [hS, hzw, hfac, Complex.norm_mul] at h
    have him := Complex.abs_im_le_norm ((r : ℂ) - Complex.I / (‖c₀‖ : ℂ))
    rw [Complex.sub_im, Complex.ofReal_im, Complex.div_ofReal_im, Complex.I_im, zero_sub,
      abs_neg, abs_of_pos (by positivity)] at him
    calc 1 = ‖c₀‖ * (1 / ‖c₀‖) := by field_simp
      _ ≤ ‖c₀‖ * ‖(r : ℂ) - Complex.I / (‖c₀‖ : ℂ)‖ := by gcongr
      _ ≤ δ := h

/-- The zero SwiGLU unit against `z₁ z₂` on the torus of `ℂ²`, with `δ = 1`. -/
example : (1 : ℝ) ≤ 1 :=
  swigluUnit_gap_product (LinearMap.fst ℝ ℂ ℂ) (LinearMap.snd ℝ ℂ ℂ) 0 0 0
    (K := {z | ‖z.1‖ = 1 ∧ ‖z.2‖ = 1}) (δ := 1)
    (fun w hw => ⟨(w, 1), ⟨hw, by simp⟩, by simp⟩) (fun z hz => by simp [swigluFFN, hz.1, hz.2])

/-- One plain unit, `(z₁, z₂) ↦ z₁ z₂` (`plainFFN_unit`), against one SwiGLU unit. Both
have ten real parameters: the plain unit's complex-linear rows `fst`, `snd` and column
`√2` have `4 + 4 + 2`, as do the SwiGLU unit's real rows `a`, `b` and column `c₀`. On the
torus `|z₁| = |z₂| = 1` the SwiGLU unit errs by at least `1`, as the zero map does. -/
theorem swigluUnit_gap_plainUnit_torus (a b : ℂ × ℂ →ₗ[ℝ] ℝ) (c₀ : ℂ) {δ : ℝ}
    (hδ : ∀ z : ℂ × ℂ, ‖z.1‖ = 1 → ‖z.2‖ = 1 →
      ‖swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => c₀) z -
        plainFFN (fun _ : Unit => LinearMap.fst ℝ ℂ ℂ) (fun _ => LinearMap.snd ℝ ℂ ℂ)
          (fun _ => (Real.sqrt 2 : ℂ)) (fun _ => (Real.sqrt 2 : ℂ) * Complex.I) z‖ ≤ δ) :
    1 ≤ δ :=
  swigluUnit_gap_product (LinearMap.fst ℝ ℂ ℂ) (LinearMap.snd ℝ ℂ ℂ) a b c₀
    (K := {z | ‖z.1‖ = 1 ∧ ‖z.2‖ = 1}) (fun w hw => ⟨(w, 1), ⟨hw, by simp⟩, by simp⟩)
    (fun z hz => by simpa only [plainFFN_unit] using hδ z hz.1 hz.2)

example : (1 : ℝ) ≤ 1 :=
  swigluUnit_gap_plainUnit_torus 0 0 0 fun z h1 h2 => by
    simp [swigluFFN, plainFFN_unit, h1, h2]

end Transformer.ComplexFFN
