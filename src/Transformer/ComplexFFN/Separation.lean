module

public import Transformer.ComplexFFN.Embedding

/-!
# SwiGLU blocks that no complex-product FFN approximates

Source: the blocks of `Defs` (DiffGPT `5a37c51`) and the symmetries of `Basic`.

A single SwiGLU unit `SiLU(ℓ z) (ℓ z) w` has a nonzero odd part
`(ℓ z)² (2σ(ℓ z) - 1) w / 2`. All three complex products are even, so at any width
their error on a set closed under `z ↦ -z` is at least that odd part
(`plainFFN_gap_swigluUnit` and its radial and diagonal versions).

`|φ z|² w`, for a complex-linear `φ`, is four SwiGLU units (`normSq_eq_swigluFFN`).
It is unchanged by `J`, while plain and radial FFNs with complex-linear projections
are negated by it. So their error is at least `|φ z|² ‖w‖` wherever `z` and `J z`
lie in the set (`plainFFN_gap_normSq`): on the unit sphere of `ℂ^n` no plain FFN of
any width approximates `|z₁|²` better than the zero map does.

With `Embedding`, the plain FFN class is strictly inside the bias-free SwiGLU class:
equal to SwiGLU blocks of six times its width, and missing SwiGLU blocks of width
one. The claim that the plain product approximates better than SiLU is false.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.ComplexFFN

open Transformer.Landscape.Neural (silu)

variable {V E κ : Type*} [Fintype κ] [AddCommGroup V] [Module ℝ V]
variable [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The odd part of one SwiGLU unit with equal gate and value rows:
`T z - T (-z) = (ℓ z)² (2σ(ℓ z) - 1) w`. -/
theorem swigluFFN_unit_sub_neg (ℓ : V →ₗ[ℝ] ℝ) (w : E) (z : V) :
    swigluFFN (fun _ : Unit => ℓ) (fun _ => ℓ) (fun _ => w) z -
        swigluFFN (fun _ : Unit => ℓ) (fun _ => ℓ) (fun _ => w) (-z) =
      (ℓ z ^ 2 * (2 * Real.sigmoid (ℓ z) - 1)) • w := by
  simp only [swigluFFN, Fintype.sum_unique, map_neg, ← sub_smul]
  rw [silu_eq_mul_sigmoid, silu_eq_mul_sigmoid, Real.sigmoid_neg]
  ring_nf

/-- An even approximant of one SwiGLU unit errs by at least half its odd part. -/
theorem even_gap_swigluUnit {f : V → E} (hf : ∀ z, f (-z) = f z) (ℓ : V →ₗ[ℝ] ℝ) (w : E)
    {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖f z - swigluFFN (fun _ : Unit => ℓ) (fun _ => ℓ) (fun _ => w) z‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hnz : -z ∈ K) :
    ℓ z ^ 2 * |2 * Real.sigmoid (ℓ z) - 1| * ‖w‖ ≤ 2 * δ := by
  have h := gap_of_invariant (σ := Neg.neg) hf hδ hz hnz
  rwa [swigluFFN_unit_sub_neg, norm_smul, Real.norm_eq_abs, abs_mul,
    abs_of_nonneg (sq_nonneg _)] at h

/-- No plain FFN, of any width, approximates one SwiGLU unit better than half its
odd part. -/
theorem plainFFN_gap_swigluUnit (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → E) (ℓ : V →ₗ[ℝ] ℝ)
    (w : E) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖plainFFN g u c c' z -
      swigluFFN (fun _ : Unit => ℓ) (fun _ => ℓ) (fun _ => w) z‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hnz : -z ∈ K) :
    ℓ z ^ 2 * |2 * Real.sigmoid (ℓ z) - 1| * ‖w‖ ≤ 2 * δ :=
  even_gap_swigluUnit (plainFFN_neg g u c c') ℓ w hδ hz hnz

/-- The same gap for the radial product, with real or complex projections. -/
theorem radialFFN_gap_swigluUnit (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → E)
    (ℓ : V →ₗ[ℝ] ℝ) (w : E) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖radialFFN ε g u c c' z -
      swigluFFN (fun _ : Unit => ℓ) (fun _ => ℓ) (fun _ => w) z‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hnz : -z ∈ K) :
    ℓ z ^ 2 * |2 * Real.sigmoid (ℓ z) - 1| * ‖w‖ ≤ 2 * δ :=
  even_gap_swigluUnit (radialFFN_neg ε g u c c') ℓ w hδ hz hnz

/-- The same gap for the diagonal (coordinate) product. -/
theorem diagonalFFN_gap_swigluUnit (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → E)
    (ℓ : V →ₗ[ℝ] ℝ) (w : E) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖diagonalFFN ε g u c c' z -
      swigluFFN (fun _ : Unit => ℓ) (fun _ => ℓ) (fun _ => w) z‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hnz : -z ∈ K) :
    ℓ z ^ 2 * |2 * Real.sigmoid (ℓ z) - 1| * ‖w‖ ≤ 2 * δ :=
  even_gap_swigluUnit (diagonalFFN_neg ε g u c c') ℓ w hδ hz hnz

/-- `|φ z|² w` is a product network with two terms. -/
theorem normSq_eq_productFFN (φ : V →ₗ[ℝ] ℂ) (w : E) :
    (fun z => Complex.normSq (φ z) • w) =
      productFFN ![Complex.reLm ∘ₗ φ, Complex.imLm ∘ₗ φ]
        ![Complex.reLm ∘ₗ φ, Complex.imLm ∘ₗ φ] ![w, w] := by
  funext z
  simp only [productFFN, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    LinearMap.comp_apply, Complex.reLm_coe, Complex.imLm_coe, Complex.normSq_apply, add_smul]

/-- `|φ z|² w` is a bias-free SwiGLU block with four units. -/
theorem normSq_eq_swigluFFN (φ : V →ₗ[ℝ] ℂ) (w : E) :
    (fun z => Complex.normSq (φ z) • w) =
      swigluFFN (Sum.elim ![Complex.reLm ∘ₗ φ, Complex.imLm ∘ₗ φ]
          (-![Complex.reLm ∘ₗ φ, Complex.imLm ∘ₗ φ]))
        (Sum.elim ![Complex.reLm ∘ₗ φ, Complex.imLm ∘ₗ φ]
          ![Complex.reLm ∘ₗ φ, Complex.imLm ∘ₗ φ])
        (Sum.elim ![w, w] (-![w, w])) := by
  rw [normSq_eq_productFFN, productFFN_eq_swigluFFN]

/-- An approximant negated by `J` errs by at least `|φ z|² ‖w‖` on the target
`|φ z|² w`, when `φ` is complex-linear and both `z` and `J z` are in the set. -/
theorem anti_gap_normSq {f : V → E} {J : V → V} (hf : ∀ z, f (J z) = -f z)
    (φ : V →ₗ[ℝ] ℂ) (hφ : ∀ z, φ (J z) = Complex.I * φ z) (w : E) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖f z - Complex.normSq (φ z) • w‖ ≤ δ) {z : V} (hz : z ∈ K)
    (hJz : J z ∈ K) : Complex.normSq (φ z) * ‖w‖ ≤ δ := by
  have h := gap_of_anti (T := fun z => Complex.normSq (φ z) • w) hf hδ hz hJz
  simp only [hφ, Complex.normSq_mul, Complex.normSq_I, one_mul, ← add_smul, norm_smul,
    Real.norm_eq_abs] at h
  rw [abs_of_nonneg (add_nonneg (Complex.normSq_nonneg _) (Complex.normSq_nonneg _))] at h
  linarith

/-- No plain FFN with complex-linear projections, of any width, approximates the
four-unit SwiGLU block `|φ z|² w` with error below `|φ z|² ‖w‖`. -/
theorem plainFFN_gap_normSq (J : V → V) (g u : κ → V →ₗ[ℝ] ℂ)
    (hg : ∀ k z, g k (J z) = Complex.I * g k z) (hu : ∀ k z, u k (J z) = Complex.I * u k z)
    (c c' : κ → E) (φ : V →ₗ[ℝ] ℂ) (hφ : ∀ z, φ (J z) = Complex.I * φ z) (w : E)
    {K : Set V} {δ : ℝ} (hδ : ∀ z ∈ K, ‖plainFFN g u c c' z - Complex.normSq (φ z) • w‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hJz : J z ∈ K) : Complex.normSq (φ z) * ‖w‖ ≤ δ :=
  anti_gap_normSq (plainFFN_anti J g u hg hu c c') φ hφ w hδ hz hJz

/-- The same gap for the radial product with complex-linear projections. -/
theorem radialFFN_gap_normSq (ε : ℝ) (J : V → V) (g u : κ → V →ₗ[ℝ] ℂ)
    (hg : ∀ k z, g k (J z) = Complex.I * g k z) (hu : ∀ k z, u k (J z) = Complex.I * u k z)
    (c c' : κ → E) (φ : V →ₗ[ℝ] ℂ) (hφ : ∀ z, φ (J z) = Complex.I * φ z) (w : E)
    {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖radialFFN ε g u c c' z - Complex.normSq (φ z) • w‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hJz : J z ∈ K) : Complex.normSq (φ z) * ‖w‖ ≤ δ :=
  anti_gap_normSq (radialFFN_anti ε J g u hg hu c c') φ hφ w hδ hz hJz

/-- On the unit circle the zero network attains the bound: no plain FFN beats it. -/
example : Complex.normSq ((LinearMap.id : ℂ →ₗ[ℝ] ℂ) 1) * ‖(1 : ℂ)‖ ≤ 1 :=
  plainFFN_gap_normSq (Complex.I * ·) (fun _ : Unit => LinearMap.id) (fun _ => LinearMap.id)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ => (0 : ℂ)) (fun _ => 0) LinearMap.id
    (fun _ => rfl) 1 (K := Metric.sphere 0 1) (fun z hz => by
      have hz' : ‖z‖ = 1 := by simpa using hz
      simp [plainFFN, complexReadout, Complex.normSq_eq_norm_sq, hz'])
    (by simp) (by simp)

/-- The zero plain FFN against one SwiGLU unit on `{1, -1} ⊆ ℂ`. -/
example :
    Complex.reLm 1 ^ 2 * |2 * Real.sigmoid (Complex.reLm 1) - 1| * ‖(1 : ℂ)‖ ≤
      2 * (‖swigluFFN (fun _ : Unit => Complex.reLm) (fun _ => Complex.reLm)
          (fun _ => (1 : ℂ)) 1‖ +
        ‖swigluFFN (fun _ : Unit => Complex.reLm) (fun _ => Complex.reLm)
          (fun _ => (1 : ℂ)) (-1)‖) := by
  refine plainFFN_gap_swigluUnit (fun _ : Unit => LinearMap.id) (fun _ => LinearMap.id)
    (fun _ => (0 : ℂ)) (fun _ => 0) Complex.reLm 1 (K := {1, -1}) ?_ (by simp) (by simp)
  intro z hz
  have h0 : plainFFN (fun _ : Unit => (LinearMap.id : ℂ →ₗ[ℝ] ℂ)) (fun _ => LinearMap.id)
      (fun _ => (0 : ℂ)) (fun _ => 0) z = 0 := by
    simp [plainFFN, complexReadout]
  rw [h0, zero_sub, norm_neg]
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hz
  rcases hz with rfl | rfl
  · exact le_add_of_nonneg_right (norm_nonneg _)
  · exact le_add_of_nonneg_left (norm_nonneg _)

end Transformer.ComplexFFN
