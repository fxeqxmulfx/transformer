module

public import Transformer.ComplexProjection.Basic
public import Mathlib.Analysis.InnerProductSpace.Projection.Submodule

/-!
# Conjugation of a complex line

Source: the splitting of `Defs` (DiffGPT `5a37c51`), on a real inner product space with
a complex structure `J` that keeps the norm, as multiplication by `i` does on the model's
stream. Such a `J` keeps inner products and is skew, `⟪x, J y⟫ = -⟪J x, y⟫`
(`inner_map_right`), so `J x ⊥ x`.

For a unit `u`, `lineConj J u z = ⟪u, z⟫ u - ⟪J u, z⟫ J u` is complex conjugation on the
line spanned by `u` and `J u`, and zero on its orthogonal complement. It is
conjugate-linear (`lineConj_anticomm`), maps into any subspace holding `u` and `J u`
(`lineConj_mem`), vanishes on its complement (`lineConj_eq_zero`) and shrinks norms
(`norm_lineConj_le`). Every map commuting with `J`, linear or not, misses it by `1` on
the unit sphere, as much as the zero map (`one_le_of_approx_lineConj`).
-/

@[expose] public section

namespace Transformer.ComplexProjection

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {J : E →ₗ[ℝ] E}

/-- A linear map that keeps the norm keeps inner products. -/
theorem inner_map_map_of_norm (hn : ∀ z, ‖J z‖ = ‖z‖) (x y : E) : ⟪J x, J y⟫ = ⟪x, y⟫ :=
  LinearIsometry.inner_map_map ⟨J, hn⟩ x y

/-- A complex structure that keeps the norm is skew. -/
theorem inner_map_right (hJ : ∀ z, J (J z) = -z) (hn : ∀ z, ‖J z‖ = ‖z‖) (x y : E) :
    ⟪x, J y⟫ = -⟪J x, y⟫ := by
  have h := inner_map_map_of_norm hn (J x) y
  rw [hJ, inner_neg_left] at h
  linarith

/-- In particular `J x ⊥ x`. -/
theorem inner_map_self (hJ : ∀ z, J (J z) = -z) (hn : ∀ z, ‖J z‖ = ‖z‖) (x : E) :
    ⟪J x, x⟫ = 0 := by
  have h := inner_map_right hJ hn x x
  have h' := real_inner_comm x (J x)
  linarith

example (x y : ℂ) := inner_map_map_of_norm norm_mulLeft_I x y
example (x y : ℂ) := inner_map_right mulLeft_I_mulLeft_I norm_mulLeft_I x y
example (x : ℂ) := inner_map_self mulLeft_I_mulLeft_I norm_mulLeft_I x

/-- Conjugation of the complex line through `u`: `z ↦ ⟪u, z⟫ u - ⟪J u, z⟫ J u`. -/
noncomputable def lineConj (J : E →ₗ[ℝ] E) (u : E) : E →ₗ[ℝ] E :=
  (innerₛₗ ℝ u).smulRight u - (innerₛₗ ℝ (J u)).smulRight (J u)

theorem lineConj_apply (J : E →ₗ[ℝ] E) (u z : E) :
    lineConj J u z = ⟪u, z⟫ • u - ⟪J u, z⟫ • J u := by
  simp only [lineConj, LinearMap.sub_apply, LinearMap.smulRight_apply, innerₛₗ_apply_apply]

/-- It is conjugate-linear. -/
theorem lineConj_anticomm (hJ : ∀ z, J (J z) = -z) (hn : ∀ z, ‖J z‖ = ‖z‖) (u z : E) :
    lineConj J u (J z) = -J (lineConj J u z) := by
  rw [lineConj_apply, lineConj_apply, inner_map_map_of_norm hn, inner_map_right hJ hn,
    map_sub, map_smul, map_smul, hJ]
  module

/-- It fixes `u` up to `‖u‖²`. -/
theorem lineConj_self (hJ : ∀ z, J (J z) = -z) (hn : ∀ z, ‖J z‖ = ‖z‖) (u : E) :
    lineConj J u u = ‖u‖ ^ 2 • u := by
  rw [lineConj_apply, inner_map_self hJ hn, zero_smul, sub_zero, real_inner_self_eq_norm_sq]

example (z : ℂ) := lineConj_anticomm mulLeft_I_mulLeft_I norm_mulLeft_I 1 z
example := lineConj_self mulLeft_I_mulLeft_I norm_mulLeft_I (1 : ℂ)

/-- It maps into every subspace holding `u` and `J u`. -/
theorem lineConj_mem {S : Submodule ℝ E} {u : E} (hu : u ∈ S) (hJu : J u ∈ S) (z : E) :
    lineConj J u z ∈ S := by
  rw [lineConj_apply]
  exact S.sub_mem (S.smul_mem _ hu) (S.smul_mem _ hJu)

/-- It vanishes on the orthogonal complement of such a subspace. -/
theorem lineConj_eq_zero {S : Submodule ℝ E} {u z : E} (hu : u ∈ S) (hJu : J u ∈ S)
    (hz : z ∈ Sᗮ) : lineConj J u z = 0 := by
  rw [lineConj_apply, Submodule.inner_right_of_mem_orthogonal hu hz,
    Submodule.inner_right_of_mem_orthogonal hJu hz, zero_smul, zero_smul, sub_zero]

example (z : ℂ) := lineConj_mem (J := LinearMap.mulLeft ℝ Complex.I) (S := ⊤) (u := 1)
  Submodule.mem_top Submodule.mem_top z
example := lineConj_eq_zero (J := LinearMap.mulLeft ℝ Complex.I) (S := ⊤) (u := 1) (z := 0)
  Submodule.mem_top Submodule.mem_top (Submodule.zero_mem _)

/-- On a unit `u` it shrinks norms: Bessel's inequality for the pair `u, J u`. -/
theorem norm_lineConj_le (hJ : ∀ z, J (J z) = -z) (hn : ∀ z, ‖J z‖ = ‖z‖) {u : E}
    (hu : ‖u‖ = 1) (z : E) : ‖lineConj J u z‖ ≤ ‖z‖ := by
  have hJu : ‖J u‖ = 1 := by rw [hn, hu]
  have h0 : ⟪u, J u⟫ = 0 := by
    rw [real_inner_comm]
    exact inner_map_self hJ hn u
  have h1 : ‖lineConj J u z‖ ^ 2 = ⟪u, z⟫ ^ 2 + ⟪J u, z⟫ ^ 2 := by
    simp only [lineConj_apply, norm_sub_sq_real, norm_smul, hu, hJu, real_inner_smul_left,
      real_inner_smul_right, h0, Real.norm_eq_abs, mul_one, mul_zero, sq_abs]
    ring
  have h2 : ‖z - (⟪u, z⟫ • u + ⟪J u, z⟫ • J u)‖ ^ 2 =
      ‖z‖ ^ 2 - (⟪u, z⟫ ^ 2 + ⟪J u, z⟫ ^ 2) := by
    simp only [norm_sub_sq_real, norm_add_sq_real, inner_add_right, real_inner_smul_left,
      real_inner_smul_right, norm_smul, hu, hJu, h0, Real.norm_eq_abs, mul_one, mul_zero,
      sq_abs]
    rw [real_inner_comm u z, real_inner_comm (J u) z]
    ring
  have h3 := sq_nonneg ‖z - (⟪u, z⟫ • u + ⟪J u, z⟫ • J u)‖
  exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).1 (by rw [h1]; linarith)

example (z : ℂ) := norm_lineConj_le mulLeft_I_mulLeft_I norm_mulLeft_I norm_one z

/-- Every map commuting with `J` misses the conjugation of a complex line by `1` on the
unit sphere: as much as the zero map, by `norm_lineConj_le`. -/
theorem one_le_of_approx_lineConj (hJ : ∀ z, J (J z) = -z) (hn : ∀ z, ‖J z‖ = ‖z‖) {u : E}
    (hu : ‖u‖ = 1) {f : E → E} (hf : ∀ z, f (J z) = J (f z)) {δ : ℝ}
    (hδ : ∀ z, ‖z‖ = 1 → ‖f z - lineConj J u z‖ ≤ δ) : 1 ≤ δ := by
  have h := norm_le_of_anticomm (J := J) (J' := J) hn (T := lineConj J u) hf
    (lineConj_anticomm hJ hn u) (K := {z | ‖z‖ = 1}) hδ (z := u) hu
    (show ‖J u‖ = 1 by rw [hn, hu])
  rwa [lineConj_self hJ hn, hu, one_pow, one_smul, hu] at h

example : (1 : ℝ) ≤ 1 :=
  one_le_of_approx_lineConj mulLeft_I_mulLeft_I norm_mulLeft_I norm_one (f := fun _ => 0)
    (fun z => by simp) fun z hz => by
      rw [zero_sub, norm_neg]
      exact (norm_lineConj_le mulLeft_I_mulLeft_I norm_mulLeft_I norm_one z).trans hz.le

end Transformer.ComplexProjection
