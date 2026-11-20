module

public import Transformer.ComplexProjection.Line
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.LinearAlgebra.Complex.FiniteDimensional
public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
public import Mathlib.LinearAlgebra.FreeModule.Finite.Matrix

/-!
# Complex and real projections at equal parameters

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`). A complex-linear projection of the
model's stream `ℂ^n` has `2n²` real parameters (`finrank_complexLinear`), a real
projection of width `r` has `r²` (`finrank_realLinear`). Equal parameters give
`r = √2 n`, strictly between `n` and the `2n` real dimensions of `ℂ^n`
(`finrank_real_euclideanSpace`); at `D = 128` that is `n = 64` and `r ≈ 90.5`.

The real stream sits in the complex one as a real subspace `S`: the real model's
projections land in `S` and vanish on `Sᗮ`, the complex model's commute with
multiplication by `i`. For `n < dim S < 2n` neither class contains the other
(`complex_real_incomparable`):

* every map landing in a proper subspace, of any form, misses the identity by `1` at a
  unit vector orthogonal to it (`exists_unit_mem_orthogonal`, `one_le_norm_sub_self`);
  the identity is complex-linear;
* a subspace of more than half the dimension holds a complex line `u, J u`
  (`exists_mem_complexLine`), and every map commuting with `J` misses the conjugation of
  that line, a real projection of `S`, by `1` on the unit sphere
  (`one_le_of_approx_lineConj`), as much as the zero map.

Both misses are as large as the zero map's. The width count is the arithmetic above;
the theorems hold for every `S` in the range, the equal-parameter width among them.
-/

@[expose] public section

namespace Transformer.ComplexProjection

open Module
open scoped RealInnerProductSpace

section General

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {J : E →ₗ[ℝ] E}

/-- A map landing in `S` misses the identity by `1` at a unit vector orthogonal to `S`,
whatever its form. -/
theorem one_le_norm_sub_self {S : Submodule ℝ E} {z : E} (hz : z ∈ Sᗮ) (hz1 : ‖z‖ = 1)
    {f : E → E} (hf : ∀ x, f x ∈ S) : 1 ≤ ‖f z - z‖ := by
  have h := norm_sub_sq_real (f z) z
  rw [Submodule.inner_right_of_mem_orthogonal (hf z) hz, hz1] at h
  nlinarith [norm_nonneg (f z - z), sq_nonneg ‖f z‖]

example : (1 : ℝ) ≤ ‖(fun _ : ℂ => (0 : ℂ)) 1 - 1‖ :=
  one_le_norm_sub_self (S := ⊥) (by simp) norm_one fun _ => Submodule.zero_mem _

/-- A proper subspace of a finite-dimensional space has a unit vector orthogonal to it. -/
theorem exists_unit_mem_orthogonal [FiniteDimensional ℝ E] {S : Submodule ℝ E} (hS : S ≠ ⊤) :
    ∃ z ∈ Sᗮ, ‖z‖ = 1 := by
  have h : Sᗮ ≠ ⊥ := by rwa [Ne, Submodule.orthogonal_eq_bot_iff]
  obtain ⟨v, hv, hv0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot h
  exact ⟨‖v‖⁻¹ • v, Sᗮ.smul_mem _ hv, by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hv0)]⟩

example := exists_unit_mem_orthogonal (S := (⊥ : Submodule ℝ ℂ)) bot_ne_top

/-- A real subspace of more than half the dimension holds a complex line: `S ∩ J S` has
dimension at least `2 dim S - dim E > 0`. -/
theorem exists_mem_complexLine [FiniteDimensional ℝ E] (hJ : ∀ z, J (J z) = -z)
    {S : Submodule ℝ E} (hS : finrank ℝ E < 2 * finrank ℝ S) :
    ∃ u ∈ S, J u ∈ S ∧ ‖u‖ = 1 := by
  have hinj : Function.Injective J := fun x y hxy => by
    have h := congrArg J hxy
    rwa [hJ, hJ, neg_inj] at h
  have hmap : finrank ℝ (S.map J) = finrank ℝ S :=
    (LinearEquiv.finrank_eq (Submodule.equivMapOfInjective J hinj S)).symm
  have hsum := Submodule.finrank_sup_add_finrank_inf_eq S (S.map J)
  have hle := Submodule.finrank_le (S ⊔ S.map J)
  have hne : S ⊓ S.map J ≠ ⊥ := by
    intro h
    rw [h, finrank_bot] at hsum
    omega
  obtain ⟨v, hv, hv0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hne
  obtain ⟨hvS, hvJ⟩ := Submodule.mem_inf.mp hv
  obtain ⟨w, hwS, rfl⟩ := Submodule.mem_map.mp hvJ
  refine ⟨‖J w‖⁻¹ • J w, S.smul_mem _ hvS, ?_, ?_⟩
  · rw [map_smul, hJ]
    exact S.smul_mem _ (S.neg_mem hwS)
  · rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hv0)]

example := exists_mem_complexLine (J := LinearMap.mulLeft ℝ Complex.I) mulLeft_I_mulLeft_I
  (S := ⊤) (by simp)

/-- Neither class contains the other when `S` has more than half and less than all of the
dimensions: every map landing in `S` misses the complex-linear identity by `1` at some unit
vector, and some real projection of `S` (into `S`, zero on `Sᗮ`, norm at most one) is
missed by `1` on the unit sphere by every map commuting with `J`. -/
theorem complex_real_incomparable [FiniteDimensional ℝ E] (hJ : ∀ z, J (J z) = -z)
    (hn : ∀ z, ‖J z‖ = ‖z‖) {S : Submodule ℝ E} (hlo : finrank ℝ E < 2 * finrank ℝ S)
    (hhi : finrank ℝ S < finrank ℝ E) :
    (∃ z : E, ‖z‖ = 1 ∧ ∀ f : E → E, (∀ x, f x ∈ S) → 1 ≤ ‖f z - z‖) ∧
      ∃ T : E →ₗ[ℝ] E, (∀ x, T x ∈ S) ∧ (∀ x ∈ Sᗮ, T x = 0) ∧ (∀ x, ‖T x‖ ≤ ‖x‖) ∧
        ∀ f : E → E, (∀ z, f (J z) = J (f z)) →
          ∀ δ : ℝ, (∀ z, ‖z‖ = 1 → ‖f z - T z‖ ≤ δ) → 1 ≤ δ := by
  have hS : S ≠ ⊤ := fun h => by
    rw [h, finrank_top] at hhi
    exact lt_irrefl _ hhi
  obtain ⟨z, hz, hz1⟩ := exists_unit_mem_orthogonal hS
  obtain ⟨u, hu, hJu, hu1⟩ := exists_mem_complexLine hJ hlo
  exact ⟨⟨z, hz1, fun f hf => one_le_norm_sub_self hz hz1 hf⟩, lineConj J u,
    lineConj_mem hu hJu, fun x hx => lineConj_eq_zero hu hJu hx, norm_lineConj_le hJ hn hu1,
    fun f hf δ hδ => one_le_of_approx_lineConj hJ hn hu1 hf hδ⟩

end General

section Stream

/-- Multiplication by `i` on the model's stream `ℂ^n`, as a real map. -/
noncomputable def mulI (n : ℕ) : EuclideanSpace ℂ (Fin n) →ₗ[ℝ] EuclideanSpace ℂ (Fin n) :=
  (Complex.I • LinearMap.id : EuclideanSpace ℂ (Fin n) →ₗ[ℂ] _).restrictScalars ℝ

theorem mulI_mulI (n : ℕ) (z : EuclideanSpace ℂ (Fin n)) : mulI n (mulI n z) = -z := by
  simp [mulI, smul_smul]

theorem norm_mulI (n : ℕ) (z : EuclideanSpace ℂ (Fin n)) : ‖mulI n z‖ = ‖z‖ := by
  simp [mulI, norm_smul]

/-- `ℂ^n` has `2n` real dimensions. -/
theorem finrank_real_euclideanSpace (n : ℕ) :
    finrank ℝ (EuclideanSpace ℂ (Fin n)) = 2 * n := by
  rw [finrank_real_of_complex, finrank_euclideanSpace_fin]

/-- A complex-linear projection of `ℂ^n` has `2n²` real parameters. -/
theorem finrank_complexLinear (n : ℕ) :
    finrank ℝ (EuclideanSpace ℂ (Fin n) →ₗ[ℂ] EuclideanSpace ℂ (Fin n)) = 2 * n ^ 2 := by
  rw [Module.finrank_linearMap, finrank_real_of_complex, finrank_euclideanSpace_fin]
  ring

/-- A real projection of width `r` has `r²`. -/
theorem finrank_realLinear (r : ℕ) :
    finrank ℝ (EuclideanSpace ℝ (Fin r) →ₗ[ℝ] EuclideanSpace ℝ (Fin r)) = r ^ 2 := by
  rw [Module.finrank_linearMap, finrank_euclideanSpace_fin]
  ring

/-- On the model's stream: for a real stream `S ⊆ ℂ^n` with `n < dim S < 2n`, which holds
at equal parameters, complex and real projections are incomparable. -/
theorem complex_real_incomparable_stream {n : ℕ}
    {S : Submodule ℝ (EuclideanSpace ℂ (Fin n))} (hlo : n < finrank ℝ S)
    (hhi : finrank ℝ S < 2 * n) :
    (∃ z : EuclideanSpace ℂ (Fin n), ‖z‖ = 1 ∧
        ∀ f : EuclideanSpace ℂ (Fin n) → EuclideanSpace ℂ (Fin n), (∀ x, f x ∈ S) →
          1 ≤ ‖f z - z‖) ∧
      ∃ T : EuclideanSpace ℂ (Fin n) →ₗ[ℝ] EuclideanSpace ℂ (Fin n), (∀ x, T x ∈ S) ∧
        (∀ x ∈ Sᗮ, T x = 0) ∧ (∀ x, ‖T x‖ ≤ ‖x‖) ∧
        ∀ f : EuclideanSpace ℂ (Fin n) → EuclideanSpace ℂ (Fin n),
          (∀ z, f (mulI n z) = mulI n (f z)) →
          ∀ δ : ℝ, (∀ z, ‖z‖ = 1 → ‖f z - T z‖ ≤ δ) → 1 ≤ δ :=
  complex_real_incomparable (mulI_mulI n) (norm_mulI n)
    (by rw [finrank_real_euclideanSpace]; omega) (by rw [finrank_real_euclideanSpace]; exact hhi)

/-- A real stream of three dimensions in `ℂ²`: the complement of a real line. -/
theorem finrank_orthogonal_single :
    finrank ℝ (ℝ ∙ (EuclideanSpace.single 0 1 : EuclideanSpace ℂ (Fin 2)))ᗮ = 3 := by
  have h := Submodule.finrank_add_finrank_orthogonal
    (ℝ ∙ (EuclideanSpace.single 0 1 : EuclideanSpace ℂ (Fin 2)))
  rw [finrank_span_singleton (by simp), finrank_real_euclideanSpace] at h
  omega

example := complex_real_incomparable_stream (n := 2)
  (by rw [finrank_orthogonal_single]; norm_num) (by rw [finrank_orthogonal_single]; norm_num)

end Stream

end Transformer.ComplexProjection
