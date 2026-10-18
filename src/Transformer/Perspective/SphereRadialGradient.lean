/-
# The radial extensions of `USA` and `SA` are rescaled Euclidean gradients

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Let `g(x) = 𝖤_β(x/‖x‖)` on tuples of vectors of `ℝ^d`, with
`𝖤_β(X) = (2β)⁻¹ Σ_i Σ_j e^{β⟨x_i, x_j⟩}` (`selfEnergy`).  For `β ≠ 0` and
`x_i ≠ 0`, with `u = x/‖x‖`, its gradient in the `i`-th vector is the round
gradient of `𝖤_β` at `u` divided by `‖x_i‖`:
`∇_i g(x) = ‖x_i‖⁻¹ Proj_{u_i} Σ_j e^{β⟨u_i, u_j⟩} u_j`
(`hasGradientAt_selfEnergy_update`, through the derivative `‖y‖⁻¹ Proj_{y/‖y‖}`
of `y ↦ y/‖y‖`, `hasFDerivAt_normalize`).  So the radial extensions of
`Perspective.SphereRadial` are rescaled gradient fields:

* for `USA`, `F_i(x) = n⁻¹ ‖x_i‖² ∇_i g(x)` (`radialField_usaField_eq`);
* for `SA`, `F_i(x) = ‖x_i‖² Z_i⁻¹ ∇_i g(x)`, `Z_i = Σ_j e^{β⟨u_i, u_j⟩}`
  (`radialField_saField_eq`), the weight of the metric `e:scalarproduct`.

`USA` and `SA` are the slices `‖x_i‖ = r_i` of these flows (`usa_iff_gradient`,
`sa_iff_gradient`): M5.3 of `landscape_plan.md`, with the factors `n⁻¹`, `Z_i⁻¹`.

Source: arXiv:2312.10794v5, §3.3, `rem: particle.gf` (`e:dynonX`: `USA` is `n`
times the round gradient of `𝖤_β`, there normalized by `n⁻²`), and §3.4
(`e:scalarproduct`: `SA` is the gradient of `𝖤_β` for `Σ_i Z_{β,i} ⟨a_i, b_i⟩`).
-/

import Transformer.Perspective.SphereRadial
import Transformer.Perspective.AppendixB_EBeta

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The derivative of `y ↦ y/‖y‖`** at `y ≠ 0` is `‖y‖⁻¹ Proj_{y/‖y‖}`:
`h ↦ ‖y‖⁻¹ (h - ⟨y/‖y‖, h⟩ y/‖y‖)`. -/
theorem hasFDerivAt_normalize {y : EucSpace d} (hy : y ≠ 0) :
    HasFDerivAt (fun z : EucSpace d => ‖z‖⁻¹ • z)
      (‖y‖⁻¹ • (ContinuousLinearMap.id ℝ (EucSpace d) -
        (innerSL ℝ (‖y‖⁻¹ • y)).smulRight (‖y‖⁻¹ • y))) y := by
  have hn : ‖y‖ ≠ 0 := norm_ne_zero_iff.2 hy
  have hnorm : HasFDerivAt (fun z : EucSpace d => ‖z‖)
      ((1 / (2 * √(‖y‖ ^ 2))) • (2 • innerSL ℝ y)) y := by
    convert (hasStrictFDerivAt_norm_sq y).hasFDerivAt.sqrt (pow_ne_zero 2 hn) using 1
    funext z; exact (Real.sqrt_sq (norm_nonneg z)).symm
  refine (((hasFDerivAt_inv hn).comp y hnorm).fun_smul (hasFDerivAt_id y)).congr_fderiv ?_
  ext h
  simp [Real.sqrt_sq (norm_nonneg y), smul_smul, smul_sub]
  ring

/-- **The gradient of `𝖤_β ∘ normalize` in one vector.**  For `β ≠ 0` and
`x_i ≠ 0`, with `u = x/‖x‖`,

  `∇_{x_i} 𝖤_β(x/‖x‖) = ‖x_i‖⁻¹ Proj_{u_i} Σ_j e^{β⟨u_i, u_j⟩} u_j`.

Near `x_i` only the `i`-th direction moves, by `hasFDerivAt_normalize`; the
terms `(i, j)` and `(j, i)` of the double sum contribute equally.  At `β = 0`
the factor `(2β)⁻¹` makes `selfEnergy` vanish, hence `β ≠ 0`. -/
theorem hasGradientAt_selfEnergy_update (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0)
    {x : Idx n → EucSpace d} {i : Idx n} (hx : x i ≠ 0) :
    HasGradientAt (fun y => selfEnergy d n β (tupleNormalize hd (Function.update x i y)))
      (‖x i‖⁻¹ • proj d (tupleNormalize hd x i) (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (tupleNormalize hd x i : EucSpace d)
          (tupleNormalize hd x j : EucSpace d)) • (tupleNormalize hd x j : EucSpace d))) (x i) := by
  set u : Idx n → EucSpace d := fun j => (tupleNormalize hd x j : EucSpace d) with hu
  have hui : u i = ‖x i‖⁻¹ • x i := coe_sphereNormalize hd hx
  set N' := ‖x i‖⁻¹ • (ContinuousLinearMap.id ℝ (EucSpace d) -
    (innerSL ℝ (‖x i‖⁻¹ • x i)).smulRight (‖x i‖⁻¹ • x i)) with hN'
  set w : EucSpace d → Idx n → EucSpace d := fun y => Function.update u i (‖y‖⁻¹ • y) with hw
  have hw0 : w (x i) = u := by simp only [hw, ← hui, Function.update_eq_self]
  have hD : ∀ k, HasFDerivAt (fun y => w y k) (if k = i then N' else 0) (x i) := by
    intro k
    by_cases hk : k = i
    · subst hk
      simp only [hw, Function.update_self, ite_true]
      exact hasFDerivAt_normalize hx
    · simp only [hw, Function.update_of_ne hk, hk, ite_false]
      exact hasFDerivAt_const _ _
  have hE := (HasFDerivAt.fun_sum fun k (_ : k ∈ Finset.univ) => HasFDerivAt.fun_sum
    fun l (_ : l ∈ Finset.univ) => (((hD k).inner ℝ (hD l)).const_mul β).exp).const_mul (2 * β)⁻¹
  have heq : (fun y => selfEnergy d n β (tupleNormalize hd (Function.update x i y))) =ᶠ[nhds (x i)]
      fun y => (2 * β)⁻¹ * ∑ k : Idx n, ∑ l : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (w y k) (w y l)) := by
    filter_upwards [eventually_ne_nhds hx] with y hy
    have hc : ∀ k, (tupleNormalize hd (Function.update x i y) k : EucSpace d) = w y k := by
      intro k
      by_cases hk : k = i
      · subst hk
        show (sphereNormalize hd (Function.update x k y k) : EucSpace d) = _
        simp only [hw, Function.update_self]
        exact coe_sphereNormalize hd hy
      · show (sphereNormalize hd (Function.update x i y k) : EucSpace d) = _
        simp only [hw, Function.update_of_ne hk, hu]
        rfl
    simp only [selfEnergy, particleEnergy, ContinuousLinearMap.id_apply, hc]
  show HasGradientAt _ (‖x i‖⁻¹ • proj d (u i)
    (∑ j, Real.exp (β * inner (𝕜 := ℝ) (u i) (u j)) • u j)) (x i)
  rw [hasGradientAt_iff_hasFDerivAt]
  refine (hE.congr_of_eventuallyEq heq).congr_fderiv (ContinuousLinearMap.ext fun h => ?_)
  simp only [hw0]
  have hDh : ∀ k, (if k = i then N' else 0) h = if k = i then N' h else 0 := by
    intro k; split_ifs <;> rfl
  simp only [FunLike.coe_smul, FunLike.coe_sum, Pi.smul_apply,
    Finset.sum_apply, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply,
    fderivInnerCLM_apply, InnerProductSpace.toDual_apply_apply, smul_eq_mul, hDh]
  have hsplit : ∀ r b c : ℝ, r * (β * (b + c)) = β * (r * b) + β * (r * c) := by intros; ring
  have hswap : ∑ k, ∑ l, Real.exp (β * inner (𝕜 := ℝ) (u k) (u l)) *
      inner (𝕜 := ℝ) (u k) (if l = i then N' h else 0) =
      ∑ k, ∑ l, Real.exp (β * inner (𝕜 := ℝ) (u k) (u l)) *
        inner (𝕜 := ℝ) (if k = i then N' h else 0) (u l) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
    rw [real_inner_comm (u k) (u l), real_inner_comm _ (u l)]
  have hsingle : ∑ k, ∑ l, Real.exp (β * inner (𝕜 := ℝ) (u k) (u l)) *
      inner (𝕜 := ℝ) (if k = i then N' h else 0) (u l) =
      inner (𝕜 := ℝ) (N' h) (∑ l, Real.exp (β * inner (𝕜 := ℝ) (u i) (u l)) • u l) := by
    rw [Finset.sum_eq_single i (fun k _ hk => by simp [hk]) (by simp), inner_sum]
    simp only [↓reduceIte, real_inner_smul_right]
  simp_rw [hsplit, Finset.sum_add_distrib, ← Finset.mul_sum, hswap, hsingle, hN', ← hui]
  simp [proj, inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right,
    real_inner_comm h]
  field_simp
  ring

/-- **The radial extension of `USA` is `n⁻¹ ‖x_i‖²` times the gradient** of
`𝖤_β ∘ normalize` in `x_i`. -/
theorem radialField_usaField_eq (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0) {x : Idx n → EucSpace d}
    {i : Idx n} (hx : x i ≠ 0) :
    radialField hd (usaField β) x i = ((n : ℝ)⁻¹ * ‖x i‖ ^ 2) •
      gradient (fun y => selfEnergy d n β (tupleNormalize hd (Function.update x i y))) (x i) := by
  rw [(hasGradientAt_selfEnergy_update hd hβ hx).gradient, radialField, usaField, proj_smul,
    smul_smul, smul_smul]
  congr 1
  field_simp [norm_ne_zero_iff.2 hx]

/-- **The radial extension of `SA` is `‖x_i‖² / Z_i` times the gradient** of
`𝖤_β ∘ normalize` in `x_i`, `Z_i = Σ_j e^{β⟨u_i, u_j⟩}` at `u = x/‖x‖`. -/
theorem radialField_saField_eq (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0) {x : Idx n → EucSpace d}
    {i : Idx n} (hx : x i ≠ 0) :
    radialField hd (fun U => saField d n β fun j => (U j : EucSpace d)) x i =
      (‖x i‖ ^ 2 / particlePartition d n β (ContinuousLinearMap.id ℝ (EucSpace d))
        (tupleNormalize hd x) i) •
      gradient (fun y => selfEnergy d n β (tupleNormalize hd (Function.update x i y))) (x i) := by
  rw [(hasGradientAt_selfEnergy_update hd hβ hx).gradient, radialField]
  have hS : ∑ j, softmaxWeight (saScore d n β (fun k => (tupleNormalize hd x k : EucSpace d)) i) j •
      (tupleNormalize hd x j : EucSpace d) = (particlePartition d n β
        (ContinuousLinearMap.id ℝ (EucSpace d)) (tupleNormalize hd x) i)⁻¹ •
      ∑ j, Real.exp (β * inner (𝕜 := ℝ) (tupleNormalize hd x i : EucSpace d)
        (tupleNormalize hd x j : EucSpace d)) • (tupleNormalize hd x j : EucSpace d) := by
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [softmaxWeight, div_eq_inv_mul, mul_smul]
    rfl
  simp only [saField]
  rw [hS, proj_smul, smul_smul, smul_smul]
  congr 1
  field_simp [norm_ne_zero_iff.2 hx]

/-- The hypotheses of `hasFDerivAt_normalize`, `hasGradientAt_selfEnergy_update`,
`radialField_usaField_eq` and `radialField_saField_eq` hold for two copies of a
unit vector of `ℝ³` and `β = 1`. -/
example : 1 ≤ 3 ∧ (1 : ℝ) ≠ 0 ∧ (fun _ : Idx 2 => (basePoint 2 : EucSpace 3)) 0 ≠ 0 :=
  ⟨by norm_num, one_ne_zero, ne_zero_of_mem_unit_sphere _⟩

/-- **`USA` is the slice `‖x_i‖ = r_i` of the flow `ẋ_i = n⁻¹ ‖x_i‖² ∇_i g(x)`.** -/
theorem usa_iff_gradient (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0) {r : Idx n → ℝ} (hr : ∀ i, 0 < r i)
    (X : ℝ → SphereTuple d n) :
    USA d n β X ↔ ∀ t i, HasDerivAt (fun s => r i • (X s i : EucSpace d))
      (((n : ℝ)⁻¹ * r i ^ 2) • gradient (fun y => selfEnergy d n β (tupleNormalize hd
        (Function.update (fun j => r j • (X t j : EucSpace d)) i y)))
        (r i • (X t i : EucSpace d))) t := by
  rw [usa_iff_radialField hd β hr X]
  refine forall_congr' fun t => forall_congr' fun i => ?_
  rw [radialField_usaField_eq hd hβ (x := fun j => r j • (X t j : EucSpace d))
    (smul_ne_zero (hr i).ne' (ne_zero_of_mem_unit_sphere (X t i))), norm_smul,
    mem_sphere_zero_iff_norm.mp (X t i).2, Real.norm_eq_abs, abs_of_pos (hr i), mul_one]

/-- **`SA` is the slice `‖x_i‖ = r_i` of the flow `ẋ_i = ‖x_i‖² Z_i⁻¹ ∇_i g(x)`.** -/
theorem sa_iff_gradient (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0) {r : Idx n → ℝ} (hr : ∀ i, 0 < r i)
    (X : ℝ → SphereTuple d n) :
    SA d n β X ↔ ∀ t i, HasDerivAt (fun s => r i • (X s i : EucSpace d))
      ((r i ^ 2 / particlePartition d n β (ContinuousLinearMap.id ℝ (EucSpace d)) (X t) i) •
        gradient (fun y => selfEnergy d n β (tupleNormalize hd (Function.update
          (fun j => r j • (X t j : EucSpace d)) i y))) (r i • (X t i : EucSpace d))) t := by
  rw [sa_iff_radialField hd β hr X]
  refine forall_congr' fun t => forall_congr' fun i => ?_
  rw [radialField_saField_eq hd hβ (x := fun j => r j • (X t j : EucSpace d))
    (smul_ne_zero (hr i).ne' (ne_zero_of_mem_unit_sphere (X t i))), tupleNormalize_smul hd hr,
    norm_smul, mem_sphere_zero_iff_norm.mp (X t i).2, Real.norm_eq_abs, abs_of_pos (hr i), mul_one]

/-- The hypotheses of `usa_iff_gradient` and `sa_iff_gradient` hold for `d = 3`,
`β = 1` and unit radii. -/
example : 1 ≤ 3 ∧ (1 : ℝ) ≠ 0 ∧ ∀ _ : Idx 2, (0 : ℝ) < 1 :=
  ⟨by norm_num, one_ne_zero, fun _ => one_pos⟩

end Perspective
end Transformer
