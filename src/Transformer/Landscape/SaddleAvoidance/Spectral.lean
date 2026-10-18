/-
# Saddle avoidance — the expanding splitting of a symmetric map

A symmetric linear map `A` of a finite-dimensional real inner product space
with an eigenvalue of modulus greater than one has an expanding splitting
(`exists_expandingSplitting`).  In an orthonormal basis of eigenvectors `bᵢ`
of `A` (the spectral theorem, `LinearMap.IsSymmetric.eigenvectorBasis`), `A`
multiplies the coordinate `⟪bᵢ, v⟫` by the eigenvalue `λᵢ`.  The map `P` keeps
the coordinates with `|λᵢ| > 1` and drops the others, and `c` is the least of
those `|λᵢ|`; a basis vector with `|λᵢ| > 1` is a direction fixed by `P`.

For gradient descent `A = 1 - η ∇²f(z)` at a strict saddle `z`, the kept
coordinates are those of the negative eigenvalues of the Hessian, the
directions along which the step `x ↦ x - η ∇f(x)` pushes away from `z`.
-/

import Transformer.Landscape.SaddleAvoidance.Defs
import Mathlib.Analysis.InnerProductSpace.Spectrum

open Finset

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- The coordinates in an orthonormal basis determine a vector. -/
theorem eq_of_forall_inner_basis_eq {ι : Type*} [Fintype ι] (b : OrthonormalBasis ι ℝ E)
    {w w' : E} (h : ∀ i, ⟪b i, w⟫ = ⟪b i, w'⟫) : w = w' := by
  rw [← b.sum_repr' w, ← b.sum_repr' w']
  simp_rw [h]

/-- Parseval's identity: the squared norm is the sum of the squared
coordinates in an orthonormal basis. -/
theorem norm_sq_eq_sum_inner_sq {ι : Type*} [Fintype ι] (b : OrthonormalBasis ι ℝ E) (w : E) :
    ‖w‖ ^ 2 = ∑ i, ⟪b i, w⟫ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, ← b.sum_inner_mul_inner w w]
  refine sum_congr rfl fun i _ => ?_
  rw [sq, real_inner_comm (b i) w]

/-- The hypothesis of `eq_of_forall_inner_basis_eq` holds for a vector and
itself, in the standard basis of the plane. -/
example (w : EuclideanSpace ℝ (Fin 2)) :
    ∀ i, ⟪EuclideanSpace.basisFun (Fin 2) ℝ i, w⟫ = ⟪EuclideanSpace.basisFun (Fin 2) ℝ i, w⟫ :=
  fun _ => rfl

variable [FiniteDimensional ℝ E]

/-- **A symmetric map with an eigenvalue of modulus greater than one has an
expanding splitting**, with constant `c > 1` and a nonzero vector `e` fixed by
the splitting map `P`. -/
theorem exists_expandingSplitting {A : E →ₗ[ℝ] E} (hA : A.IsSymmetric) {μ : ℝ} (hμ : 1 < |μ|)
    (hev : Module.End.HasEigenvalue A μ) :
    ∃ (P : E →ₗ[ℝ] E) (c : ℝ) (e : E), IsExpandingSplitting A P c ∧ 1 < c ∧ P e = e ∧ e ≠ 0 := by
  set b := hA.eigenvectorBasis rfl with hb
  set lam := hA.eigenvalues rfl with hlam
  have hAb : ∀ w i, ⟪b i, A w⟫ = lam i * ⟪b i, w⟫ := by
    intro w i
    rw [← hA (b i) w, hb, hA.apply_eigenvectorBasis, real_inner_smul_left]
    simp [hlam]
  set U := univ.filter (fun i => 1 < |lam i|) with hU
  let P : E →ₗ[ℝ] E :=
    { toFun := fun w => ∑ i ∈ U, ⟪b i, w⟫ • b i
      map_add' := fun w w' => by simp only [inner_add_right, add_smul, sum_add_distrib]
      map_smul' := fun a w => by
        simp only [real_inner_smul_right, mul_smul, smul_sum, RingHom.id_apply] }
  have hPb : ∀ w j, ⟪b j, P w⟫ = if j ∈ U then ⟪b j, w⟫ else 0 := by
    intro w j
    simp only [P, LinearMap.coe_mk, AddHom.coe_mk, inner_sum, real_inner_smul_right,
      orthonormal_iff_ite.1 b.orthonormal j, mul_ite, mul_one, mul_zero, sum_ite_eq]
  have hQb : ∀ w j, ⟪b j, w - P w⟫ = if j ∈ U then 0 else ⟪b j, w⟫ := by
    intro w j
    rw [inner_sub_right, hPb]
    split_ifs <;> ring
  have hnorm := norm_sq_eq_sum_inner_sq b
  obtain ⟨x, hx⟩ := hev.exists_hasEigenvector
  obtain ⟨j, hj⟩ : ∃ j, ⟪b j, x⟫ ≠ 0 := by
    by_contra h
    push Not at h
    exact hx.2 (eq_of_forall_inner_basis_eq b fun i => by rw [h i, inner_zero_right])
  have hjU : j ∈ U := by
    have h := hAb x j
    rw [hx.apply_eq_smul, real_inner_smul_right] at h
    rw [hU, mem_filter, ← mul_right_cancel₀ hj h]
    exact ⟨mem_univ j, hμ⟩
  have hUne : U.Nonempty := ⟨j, hjU⟩
  set c := U.inf' hUne (fun i => |lam i|) with hc
  have hc1 : 1 < c := (lt_inf'_iff hUne).2 fun i hi => (mem_filter.1 hi).2
  have hcle : ∀ i ∈ U, c ≤ |lam i| := fun i hi => inf'_le _ hi
  have hsq {a a' : ℝ} (ha' : 0 ≤ a') (h : a ^ 2 ≤ a' ^ 2) : a ≤ a' :=
    abs_le_of_sq_le_sq' h ha' |>.2
  refine ⟨P, c, b j, ⟨fun v => ?_, fun v => ?_, fun v => ?_, fun v => ?_, fun v => ?_⟩, hc1,
    ?_, b.orthonormal.ne_zero j⟩
  · refine eq_of_forall_inner_basis_eq b fun i => ?_
    simp only [hPb, hAb]
    split_ifs <;> ring
  · refine hsq (norm_nonneg _) ?_
    rw [hnorm (P v), hnorm v]
    refine sum_le_sum fun i _ => ?_
    rw [hPb]
    split_ifs
    · exact le_rfl
    · rw [sq (0 : ℝ), mul_zero]
      exact sq_nonneg _
  · refine hsq (norm_nonneg _) ?_
    rw [hnorm (v - P v), hnorm v]
    refine sum_le_sum fun i _ => ?_
    rw [hQb]
    split_ifs
    · rw [sq (0 : ℝ), mul_zero]
      exact sq_nonneg _
    · exact le_rfl
  · refine hsq (norm_nonneg _) ?_
    rw [mul_pow, hnorm (P v), hnorm (A (P v)), mul_sum]
    refine sum_le_sum fun i _ => ?_
    rw [hAb, hPb, mul_pow]
    split_ifs with hi
    · refine mul_le_mul_of_nonneg_right ?_ (sq_nonneg _)
      rw [← sq_abs (lam i)]
      exact pow_le_pow_left₀ (by linarith) (hcle i hi) 2
    · simp
  · refine hsq (norm_nonneg _) ?_
    rw [hnorm (A (v - P v)), hnorm (v - P v)]
    refine sum_le_sum fun i _ => ?_
    rw [hAb, hQb, mul_pow]
    split_ifs with hi
    · simp
    · have hl : |lam i| ≤ 1 := by
        simpa [hU] using hi
      refine (mul_le_mul_of_nonneg_right ?_ (sq_nonneg _)).trans (one_mul _).le
      rw [← sq_abs (lam i)]
      nlinarith [abs_nonneg (lam i)]
  · refine eq_of_forall_inner_basis_eq b fun i => ?_
    rw [hPb, orthonormal_iff_ite.1 b.orthonormal i j]
    split_ifs with h1 h2 <;> simp_all

/-- The hypotheses of `exists_expandingSplitting` hold for twice the identity
of the line, which is symmetric with the eigenvalue `2`. -/
example : ((2 : ℝ) • LinearMap.id : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] _).IsSymmetric ∧
    1 < |(2 : ℝ)| ∧
    Module.End.HasEigenvalue ((2 : ℝ) • LinearMap.id : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] _) 2 := by
  refine ⟨fun u v => ?_, by norm_num, ?_⟩
  · simp [real_inner_smul_left, real_inner_smul_right]
  · refine Module.End.hasEigenvalue_of_hasEigenvector (x := EuclideanSpace.single 0 1) ⟨?_, ?_⟩
    · simp
    · simp

end Transformer.Landscape.SaddleAvoidance
