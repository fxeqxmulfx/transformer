module

public import Transformer.ComplexFFN.Separation

/-!
# The even part of a bias-free SwiGLU FFN is a quadratic form

Source: the blocks of `Defs` (DiffGPT `5a37c51`).

A bias-free SwiGLU FFN `S` has even part `(S z + S(-z))/2 = P z / 2`, where `P` is the
product network `Σ_j (a_j z)(b_j z) c_j` (`swigluFFN_add_neg`). Product networks obey
the parallelogram law (`productFFN_parallelogram`). So at any width the uniform error of
`S` on an even target `T` is at least a quarter of the target's parallelogram defect
(`swigluFFN_gap_even`). The radial and diagonal units are even and break the law;
`Incomparable` evaluates their defect at the points `x`, `y`, `(x ± y)/√2` with
`φ x = 1` and `φ y = 0`, which the lemmas below prepare.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.ComplexFFN

variable {V E ι : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]
variable [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- `t = 1/√2` satisfies `2t² = 1`. -/
theorem two_mul_inv_sqrt_two_sq : 2 * ((Real.sqrt 2)⁻¹) ^ 2 = 1 := by
  rw [inv_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  norm_num

/-- `(1/√2)² = 1/2` in `ℂ`. -/
theorem inv_sqrt_two_mul_self : (((Real.sqrt 2)⁻¹ : ℝ) : ℂ) * ((Real.sqrt 2)⁻¹ : ℝ) = 1 / 2 := by
  have h : (Real.sqrt 2)⁻¹ * (Real.sqrt 2)⁻¹ = 1 / 2 := by
    linarith [two_mul_inv_sqrt_two_sq, sq ((Real.sqrt 2)⁻¹)]
  rw [← Complex.ofReal_mul, h]
  push_cast
  ring

/-- `1/√2` has modulus at most `1`. -/
theorem norm_inv_sqrt_two_le_one : ‖(((Real.sqrt 2)⁻¹ : ℝ) : ℂ)‖ ≤ 1 := by
  rw [Complex.norm_real, Real.norm_of_nonneg (inv_nonneg.2 (Real.sqrt_nonneg 2))]
  exact inv_le_one_of_one_le₀ (Real.one_le_sqrt.2 one_le_two)

/-- The parallelogram law for product networks, in the scaled form that keeps the
four points on one sphere: with `2t² = 1`, `P(t(x+y)) + P(t(x-y)) = P x + P y`. -/
theorem productFFN_parallelogram (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → E) {t : ℝ}
    (ht : 2 * t ^ 2 = 1) (x y : V) :
    productFFN a b c (t • (x + y)) + productFFN a b c (t • (x - y)) =
      productFFN a b c x + productFFN a b c y := by
  simp only [productFFN, ← Finset.sum_add_distrib, ← add_smul, map_smul, map_add, map_sub,
    smul_eq_mul]
  refine Finset.sum_congr rfl fun j _ => ?_
  congr 1
  linear_combination (a j x * b j x + a j y * b j y) * ht

example (a b : Fin 3 → ℂ →ₗ[ℝ] ℝ) (c : Fin 3 → ℂ) (x y : ℂ) :
    productFFN a b c ((Real.sqrt 2)⁻¹ • (x + y)) +
        productFFN a b c ((Real.sqrt 2)⁻¹ • (x - y)) =
      productFFN a b c x + productFFN a b c y :=
  productFFN_parallelogram a b c two_mul_inv_sqrt_two_sq x y

/-- A SwiGLU FFN of any width approximates an even target no better than a quarter
of the target's parallelogram defect, on a set closed under negation. -/
theorem swigluFFN_gap_even {T : V → E} (hT : ∀ z, T (-z) = T z) (a b : ι → V →ₗ[ℝ] ℝ)
    (c : ι → E) {K : Set V} {δ : ℝ} (hK : ∀ z ∈ K, -z ∈ K)
    (hδ : ∀ z ∈ K, ‖swigluFFN a b c z - T z‖ ≤ δ) {t : ℝ} (ht : 2 * t ^ 2 = 1) {x y : V}
    (hx : x ∈ K) (hy : y ∈ K) (hp : t • (x + y) ∈ K) (hm : t • (x - y) ∈ K) :
    ‖T x + T y - T (t • (x + y)) - T (t • (x - y))‖ ≤ 4 * δ := by
  set Q : V → E := fun z => (2 : ℝ)⁻¹ • productFFN a b c z with hQdef
  have hQ : ∀ z ∈ K, ‖Q z - T z‖ ≤ δ := by
    intro z hz
    have h1 := hδ z hz
    have h2 := hδ (-z) (hK z hz)
    rw [hT] at h2
    have he : Q z - T z =
        (2 : ℝ)⁻¹ • ((swigluFFN a b c z - T z) + (swigluFFN a b c (-z) - T z)) := by
      simp only [hQdef, ← swigluFFN_add_neg]
      module
    rw [he, norm_smul, Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 2⁻¹)]
    nlinarith [norm_add_le (swigluFFN a b c z - T z) (swigluFFN a b c (-z) - T z)]
  have hsum : Q (t • (x + y)) = Q x + Q y - Q (t • (x - y)) := by
    simp only [hQdef, ← smul_add, ← smul_sub, ← productFFN_parallelogram a b c ht x y,
      add_sub_cancel_right]
  have he : T x + T y - T (t • (x + y)) - T (t • (x - y)) =
      (Q (t • (x + y)) - T (t • (x + y))) + (Q (t • (x - y)) - T (t • (x - y))) -
        (Q x - T x) - (Q y - T y) := by
    rw [hsum]
    abel
  rw [he]
  have := norm_sub_le ((Q (t • (x + y)) - T (t • (x + y))) +
    (Q (t • (x - y)) - T (t • (x - y))) - (Q x - T x)) (Q y - T y)
  have := norm_sub_le ((Q (t • (x + y)) - T (t • (x + y))) +
    (Q (t • (x - y)) - T (t • (x - y)))) (Q x - T x)
  have := norm_add_le (Q (t • (x + y)) - T (t • (x + y))) (Q (t • (x - y)) - T (t • (x - y)))
  linarith [hQ _ hx, hQ _ hy, hQ _ hp, hQ _ hm]

/-- The zero FFN against the even target `cos` on `ℝ`, with `δ = 1`. -/
example (x y : ℝ) :
    ‖Real.cos x + Real.cos y - Real.cos ((Real.sqrt 2)⁻¹ • (x + y)) -
      Real.cos ((Real.sqrt 2)⁻¹ • (x - y))‖ ≤ 4 * 1 :=
  swigluFFN_gap_even (T := Real.cos) (fun z => by simp) (fun _ : Unit => 0) (fun _ => 0)
    (fun _ => (0 : ℝ)) (K := Set.univ) (δ := 1) (fun _ _ => trivial)
    (fun z _ => by simpa [swigluFFN] using Real.abs_cos_le_one (x := z))
    two_mul_inv_sqrt_two_sq trivial trivial trivial trivial

/-- For `ε ≥ 0` the radial gate at modulus `1/√2` is below the gate at modulus `1`. -/
theorem radialGate_inv_sqrt_two_lt (ε : ℝ) (hε : 0 ≤ ε) :
    radialGate ε ((Real.sqrt 2)⁻¹ : ℝ) < radialGate ε 1 := by
  apply radialGate_lt ε hε
  rw [Complex.normSq_ofReal, Complex.normSq_one, ← sq, inv_pow,
    Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  norm_num

/-- The kernels' `ε = 2^-23`. -/
example : radialGate (2 ^ (-23 : ℤ)) ((Real.sqrt 2)⁻¹ : ℝ) < radialGate (2 ^ (-23 : ℤ)) 1 :=
  radialGate_inv_sqrt_two_lt _ (by positivity)

/-- The points `t(x ± y)` of the parallelogram, for `φ x = 1` and `φ y = 0`. -/
theorem map_inv_sqrt_two_smul (φ : V →ₗ[ℝ] ℂ) {x y : V} (hx : φ x = 1) (hy : φ y = 0) :
    φ ((Real.sqrt 2)⁻¹ • (x + y)) = ((Real.sqrt 2)⁻¹ : ℝ) ∧
      φ ((Real.sqrt 2)⁻¹ • (x - y)) = ((Real.sqrt 2)⁻¹ : ℝ) := by
  simp only [map_smul, map_add, map_sub, hx, hy, add_zero, sub_zero, Complex.real_smul, mul_one,
    and_self]

/-- The first coordinate of `ℂ²` at `x = (1, 0)` and `y = (0, 1)`. -/
example := map_inv_sqrt_two_smul (LinearMap.fst ℝ ℂ ℂ) (x := ((1 : ℂ), (0 : ℂ))) (y := (0, 1))
  rfl rfl

/-- One radial unit into `ℂ` with `W_F = 1`: `s(φ z) (φ z)² / √2`. -/
theorem radialFFN_unit (ε : ℝ) (φ : V →ₗ[ℝ] ℂ) (z : V) :
    radialFFN ε (fun _ : Unit => φ) (fun _ => φ) (fun _ => (1 : ℂ)) (fun _ => Complex.I) z =
      (radialGate ε (φ z) : ℂ) * (φ z * φ z) / (Real.sqrt 2 : ℂ) := by
  rw [radialFFN, complexReadout_one_I, Fintype.sum_unique]
  rfl

/-- One diagonal unit into `ℂ` with `W_F = 1`: `s(φ z) ((φ z)_R² + i (φ z)_I²)`. -/
theorem diagonalFFN_unit (ε : ℝ) (φ : V →ₗ[ℝ] ℂ) (z : V) :
    diagonalFFN ε (fun _ : Unit => φ) (fun _ => φ) (fun _ => (1 : ℂ)) (fun _ => Complex.I) z =
      (radialGate ε (φ z) : ℂ) * ⟨(φ z).re * (φ z).re, (φ z).im * (φ z).im⟩ := by
  rw [diagonalFFN, complexReadout_one_I, Fintype.sum_unique]
  rfl

/-- One radial unit is at most `|φ z|²` in modulus. -/
theorem norm_radialFFN_unit_le (ε : ℝ) (φ : V →ₗ[ℝ] ℂ) (z : V) :
    ‖radialFFN ε (fun _ : Unit => φ) (fun _ => φ) (fun _ => (1 : ℂ)) (fun _ => Complex.I) z‖ ≤
      ‖φ z‖ ^ 2 := by
  rw [radialFFN_unit, norm_div, norm_mul, norm_mul, Complex.norm_real, Complex.norm_real,
    Real.norm_of_nonneg (radialGate_nonneg ε _), Real.norm_of_nonneg (Real.sqrt_nonneg 2),
    div_le_iff₀ (Real.sqrt_pos.2 two_pos)]
  nlinarith [mul_nonneg (sub_nonneg.2 (radialGate_le_one ε (φ z))) (mul_self_nonneg ‖φ z‖),
    mul_nonneg (mul_self_nonneg ‖φ z‖) (sub_nonneg.2 (Real.one_le_sqrt.2 one_le_two))]

/-- One diagonal unit is at most `|φ z|²` in modulus. -/
theorem norm_diagonalFFN_unit_le (ε : ℝ) (φ : V →ₗ[ℝ] ℂ) (z : V) :
    ‖diagonalFFN ε (fun _ : Unit => φ) (fun _ => φ) (fun _ => (1 : ℂ)) (fun _ => Complex.I) z‖ ≤
      ‖φ z‖ ^ 2 := by
  rw [diagonalFFN_unit, norm_mul, Complex.norm_real, Real.norm_of_nonneg (radialGate_nonneg ε _)]
  have h : ‖(⟨(φ z).re * (φ z).re, (φ z).im * (φ z).im⟩ : ℂ)‖ ≤
      |(φ z).re * (φ z).re| + |(φ z).im * (φ z).im| := Complex.norm_le_abs_re_add_abs_im _
  rw [abs_of_nonneg (mul_self_nonneg _), abs_of_nonneg (mul_self_nonneg _),
    ← Complex.normSq_apply, Complex.normSq_eq_norm_sq] at h
  nlinarith [mul_nonneg (sub_nonneg.2 (radialGate_le_one ε (φ z)))
    (norm_nonneg (⟨(φ z).re * (φ z).re, (φ z).im * (φ z).im⟩ : ℂ))]

end Transformer.ComplexFFN
