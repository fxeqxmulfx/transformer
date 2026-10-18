/-
# Saddle avoidance — a positive definite map times a symmetric map

Let `P` be a symmetric positive definite map and `H` a symmetric map of a
finite-dimensional real inner product space.

* `P` has a symmetric invertible square root `S` (`exists_sqrt`).  In an
  orthonormal basis `b` of eigenvectors of `P` with eigenvalues `λ_i > 0`, the
  maps `D_c v = Σ_i c_i ⟪b_i, v⟫ b_i` are symmetric, multiply `b_i` by `c_i`
  and compose as the products of their coefficients; `P = D_λ`, and
  `S = D_{√λ}` has the inverse `D_{1/√λ}`.
* `S⁻¹ (P H) S = S H S` is symmetric (`isSymmetric_comp_comp`), so `P H` is
  conjugate to a symmetric map.  A negative eigenvalue `λ` of `H`, with
  eigenvector `v`, gives one of `P H`
  (`exists_conj_isSymmetric_and_hasEigenvalue_neg`): for `w = S⁻¹ v`,
  `⟪S H S w, w⟫ = ⟪H v, v⟫ = λ ‖v‖² < 0`, so the symmetric `S H S` has a
  negative eigenvalue (`exists_hasEigenvalue_neg_of_inner_neg`), and `S`
  carries its eigenvectors to eigenvectors of `P H`.
* Conjugation commutes with the exponential (`conj_exp`).

For `P = G(z)` and `H = ∇²f(z)` this makes the derivative `exp (-P H)` of the
time-one map of `ẋ = -G(x) ∇f(x)` at a strict saddle `z` conjugate to a
symmetric map, with an eigenvalue greater than one
(`SaddleAvoidance.MetricFlow`).
-/

import Transformer.SaddleAvoidance.LinearFlow
import Mathlib.Analysis.InnerProductSpace.Spectrum

open NormedSpace

namespace Transformer.SaddleAvoidance

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Conjugation commutes with the exponential**:
`T exp(A) T⁻¹ = exp (T A T⁻¹)`. -/
theorem conj_exp [CompleteSpace E] (T : E ≃L[ℝ] E) (A : E →L[ℝ] E) :
    (T : E →L[ℝ] E) ∘L exp A ∘L (T.symm : E →L[ℝ] E) =
      exp ((T : E →L[ℝ] E) ∘L A ∘L (T.symm : E →L[ℝ] E)) := by
  let f : (E →L[ℝ] E) →+* (E →L[ℝ] E) :=
    { toFun := fun B => (T : E →L[ℝ] E) ∘L B ∘L (T.symm : E →L[ℝ] E)
      map_one' := by ext; simp
      map_mul' := fun B C => by ext; simp [ContinuousLinearMap.mul_def]
      map_zero' := by ext; simp
      map_add' := fun B C => by ext; simp }
  exact map_exp_of_mem_ball (𝕂 := ℝ) f
    (continuous_const.clm_comp (continuous_id.clm_comp continuous_const)) A
    ((expSeries_radius_eq_top ℝ (E →L[ℝ] E)).symm ▸ edist_lt_top _ _)

/-- `S H S` is symmetric when `S` and `H` are. -/
theorem isSymmetric_comp_comp {S H : E →L[ℝ] E} (hS : (S : E →ₗ[ℝ] E).IsSymmetric)
    (hH : (H : E →ₗ[ℝ] E).IsSymmetric) :
    ((S ∘L H ∘L S : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric := fun u v => by
  have hS' : ∀ x y, ⟪S x, y⟫ = ⟪x, S y⟫ := hS
  have hH' : ∀ x y, ⟪H x, y⟫ = ⟪x, H y⟫ := hH
  change ⟪S (H (S u)), v⟫ = ⟪u, S (H (S v))⟫
  rw [hS', hH', hS']

variable [FiniteDimensional ℝ E]

/-- **A symmetric map with a direction of negative curvature has a negative
eigenvalue.**  Otherwise, in an orthonormal basis `b` of eigenvectors with
eigenvalues `λ_i ≥ 0`, `⟪A h, h⟫ = Σ_i λ_i ⟪b_i, h⟫² ≥ 0`. -/
theorem exists_hasEigenvalue_neg_of_inner_neg {A : E →ₗ[ℝ] E} (hA : A.IsSymmetric) {h : E}
    (hh : ⟪A h, h⟫ < 0) : ∃ μ < 0, Module.End.HasEigenvalue A μ := by
  set b := hA.eigenvectorBasis rfl with hb
  set lam := hA.eigenvalues rfl with hlam
  by_contra hneg
  push Not at hneg
  have hpos : ∀ i, 0 ≤ lam i := fun i =>
    not_lt.1 fun hi => hneg (lam i) hi (hA.hasEigenvalue_eigenvalues rfl i)
  have hAb : ∀ i, ⟪A h, b i⟫ = lam i * ⟪b i, h⟫ := by
    intro i
    rw [real_inner_comm, ← hA (b i) h, hb, hA.apply_eigenvectorBasis, real_inner_smul_left]
    rfl
  have h0 : 0 ≤ ⟪A h, h⟫ := by
    rw [← b.sum_inner_mul_inner (A h) h]
    refine Finset.sum_nonneg fun i _ => ?_
    rw [hAb i, mul_assoc]
    exact mul_nonneg (hpos i) (mul_self_nonneg _)
  linarith

/-- **A symmetric positive definite map has a symmetric invertible square
root**: `S` multiplies the `i`-th vector of an orthonormal basis of
eigenvectors of `P` by the square root of its eigenvalue. -/
theorem exists_sqrt {P : E →L[ℝ] E} (hPs : (P : E →ₗ[ℝ] E).IsSymmetric)
    (hPp : ∀ v, v ≠ 0 → 0 < ⟪P v, v⟫) :
    ∃ S : E ≃L[ℝ] E, ((S : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric ∧
      (S : E →L[ℝ] E) ∘L (S : E →L[ℝ] E) = P := by
  set b := hPs.eigenvectorBasis rfl
  set lam := hPs.eigenvalues rfl
  have hPb : ∀ i, P (b i) = lam i • b i := fun i => hPs.apply_eigenvectorBasis rfl i
  have hlam : ∀ i, 0 < lam i := fun i => by
    have h := hPp (b i) (b.orthonormal.ne_zero i)
    rwa [hPb, real_inner_smul_left, real_inner_self_eq_norm_sq, b.orthonormal.1 i, one_pow,
      mul_one] at h
  set D : (Fin (Module.finrank ℝ E) → ℝ) → E →L[ℝ] E :=
    fun c => ∑ i, c i • (innerSL ℝ (b i)).smulRight (b i) with hD
  have hDb : ∀ c j, D c (b j) = c j • b j := fun c j => by
    simp [hD, orthonormal_iff_ite.1 b.orthonormal, ite_smul, smul_ite]
  have hext : ∀ A B : E →L[ℝ] E, (∀ i, A (b i) = B (b i)) → A = B := fun A B h =>
    ContinuousLinearMap.coe_injective (b.toBasis.ext fun i => by simpa using h i)
  have hDD : ∀ c d, D c ∘L D d = D (c * d) := fun c d => hext _ _ fun i => by
    rw [ContinuousLinearMap.comp_apply, hDb, map_smul, hDb, hDb, smul_smul, Pi.mul_apply,
      mul_comm]
  have hD1 : D 1 = ContinuousLinearMap.id ℝ E := hext _ _ fun i => by
    rw [hDb, Pi.one_apply, one_smul, ContinuousLinearMap.id_apply]
  have hDs : ∀ c, ((D c : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric := fun c u v => by
    change ⟪D c u, v⟫ = ⟪u, D c v⟫
    simp only [hD, sum_apply, smul_apply,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, sum_inner, inner_sum,
      real_inner_smul_left, real_inner_smul_right, real_inner_comm u]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hinv : ∀ i, √(lam i) * (√(lam i))⁻¹ = 1 := fun i =>
    mul_inv_cancel₀ (Real.sqrt_pos.2 (hlam i)).ne'
  have h1 : D (fun i => √(lam i)) ∘L D (fun i => (√(lam i))⁻¹) = ContinuousLinearMap.id ℝ E := by
    rw [hDD, ← hD1]
    exact congrArg D (funext hinv)
  have h2 : D (fun i => (√(lam i))⁻¹) ∘L D (fun i => √(lam i)) = ContinuousLinearMap.id ℝ E := by
    rw [hDD, ← hD1]
    exact congrArg D (funext fun i => by rw [Pi.mul_apply, mul_comm]; exact hinv i)
  refine ⟨ContinuousLinearEquiv.equivOfInverse (D fun i => √(lam i)) (D fun i => (√(lam i))⁻¹)
    (fun v => by simpa using DFunLike.congr_fun h2 v)
    (fun v => by simpa using DFunLike.congr_fun h1 v), hDs _, ?_⟩
  change D (fun i => √(lam i)) ∘L D (fun i => √(lam i)) = P
  rw [hDD]
  exact hext _ _ fun i => by rw [hDb, hPb, Pi.mul_apply, Real.mul_self_sqrt (hlam i).le]

/-- **A positive definite map times a symmetric map with a negative
eigenvalue.**  For `P` symmetric positive definite and `H` symmetric with a
negative eigenvalue, some continuous linear automorphism `T` makes
`T (P H) T⁻¹` symmetric — `T = S⁻¹` for the square root `S` of `P` gives
`S H S` — and `P H` has a negative eigenvalue. -/
theorem exists_conj_isSymmetric_and_hasEigenvalue_neg {P H : E →L[ℝ] E}
    (hPs : (P : E →ₗ[ℝ] E).IsSymmetric) (hPp : ∀ v, v ≠ 0 → 0 < ⟪P v, v⟫)
    (hH : (H : E →ₗ[ℝ] E).IsSymmetric) {lam : ℝ} (hlam : lam < 0)
    (hev : Module.End.HasEigenvalue (H : E →ₗ[ℝ] E) lam) :
    ∃ T : E ≃L[ℝ] E,
      ((T : E →L[ℝ] E) ∘L (P ∘L H) ∘L (T.symm : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric ∧
        ∃ κ < 0, Module.End.HasEigenvalue ((P ∘L H : E →L[ℝ] E) : E →ₗ[ℝ] E) κ := by
  obtain ⟨S, hS, hSS⟩ := exists_sqrt hPs hPp
  set K : E →L[ℝ] E := (S : E →L[ℝ] E) ∘L H ∘L (S : E →L[ℝ] E) with hK
  have hKs : (K : E →ₗ[ℝ] E).IsSymmetric := isSymmetric_comp_comp hS hH
  have hconj : (S.symm : E →L[ℝ] E) ∘L (P ∘L H) ∘L (S.symm.symm : E →L[ℝ] E) = K := by
    rw [← hSS, S.symm_symm]
    ext x
    simp [hK]
  refine ⟨S.symm, hconj ▸ hKs, ?_⟩
  obtain ⟨v, hv⟩ := hev.exists_hasEigenvector
  have hHv : H v = lam • v := hv.apply_eq_smul
  have hS' : ∀ x y, ⟪S x, y⟫ = ⟪x, S y⟫ := hS
  have hneg : ⟪K (S.symm v), S.symm v⟫ < 0 := by
    have hKv : K (S.symm v) = S (lam • v) := by simp [hK, hHv]
    rw [hKv, hS', S.apply_symm_apply, real_inner_smul_left, real_inner_self_eq_norm_sq]
    exact mul_neg_of_neg_of_pos hlam (pow_pos (norm_pos_iff.2 hv.2) 2)
  obtain ⟨κ, hκ, hevκ⟩ := exists_hasEigenvalue_neg_of_inner_neg hKs hneg
  obtain ⟨u, hu⟩ := hevκ.exists_hasEigenvector
  have hKu : K u = κ • u := hu.apply_eq_smul
  refine ⟨κ, hκ, Module.End.hasEigenvalue_of_hasEigenvector (x := S u) ⟨?_, by simpa using hu.2⟩⟩
  rw [Module.End.mem_eigenspace_iff, ContinuousLinearMap.coe_coe, ← hSS]
  have h := congrArg S hKu
  simp only [hK, ContinuousLinearMap.comp_apply, ContinuousLinearEquiv.coe_coe, map_smul] at h ⊢
  exact h

/-- The hypotheses of the theorems above hold on the line for `P = 2`,
symmetric and positive definite, and `H = -1`, symmetric with the eigenvalue
`-1 < 0` and `⟪H h, h⟫ = -1 < 0` at `h = 1`. -/
example : let P : EuclideanSpace ℝ (Fin 1) →L[ℝ] EuclideanSpace ℝ (Fin 1) :=
      (2 : ℝ) • ContinuousLinearMap.id ℝ _
    let H : EuclideanSpace ℝ (Fin 1) →L[ℝ] EuclideanSpace ℝ (Fin 1) := -ContinuousLinearMap.id ℝ _
    (P : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)).IsSymmetric ∧
      (∀ v, v ≠ 0 → 0 < ⟪P v, v⟫) ∧
      (H : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)).IsSymmetric ∧
      ⟪H (EuclideanSpace.single 0 1), EuclideanSpace.single 0 1⟫ < 0 ∧ (-1 : ℝ) < 0 ∧
      Module.End.HasEigenvalue
        (H : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)) (-1) := by
  intro P H
  refine ⟨fun u v => ?_, fun v hv => ?_, fun u v => ?_, by simp [H], by norm_num, ?_⟩
  · simp [P, real_inner_smul_left, real_inner_smul_right]
  · change 0 < ⟪(2 : ℝ) • v, v⟫
    rw [real_inner_smul_left, real_inner_self_eq_norm_sq]
    have := norm_pos_iff.2 hv
    positivity
  · simp [H]
  · refine Module.End.hasEigenvalue_of_hasEigenvector (x := EuclideanSpace.single 0 1) ⟨?_, by simp⟩
    simp [H]

end Transformer.SaddleAvoidance
