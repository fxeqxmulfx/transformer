/-
# Every solution of `USA` and of `SA` converges to a critical point of `𝖤_β`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

At a tuple of unit vectors the gradient of `g = 𝖤_β ∘ normalize` is the round gradient
`(Proj_{x_i} Σ_j e^{β⟨x_i, x_j⟩} x_j)_i` (`gradient_tupleEnergy_toLp_apply`).  So
`USA` is `ẏ = n⁻¹ ∇g(y)` and `SA` is `ẏ_i = Z_i(t)⁻¹ ∇_i g(y)`, with the partition
function `Z_i = Σ_j e^{β⟨x_i, x_j⟩}` between `n e^{-|β|}` and `n e^{|β|}`
(`partitionSA_mem_Icc`).  Both are block scalings of `∇g` by numbers in a fixed
interval of `(0, ∞)`, and Łojasiewicz convergence on the product of spheres
(`exists_tendsto_of_hasDerivAt_blockScale`) gives, for `β ≠ 0`, that every solution
converges to a critical point of `𝖤_β` (`exists_tendsto_isCriticalEBeta_of_USA`,
`exists_tendsto_isCriticalEBeta_of_SA`).  At `β = 0` the energy `(2β)⁻¹ Σ e^{β⟨·,·⟩}`
is `0` in Lean and both flows are `e:Snonres0`, treated in `Perspective.AppendixA_Saddle`.

Source: arXiv:2312.10794v5, §5 (the outline after `thm: beta.interval`) and §3.3–§3.4,
where `USA` and `SA` are gradient flows of `𝖤_β`.
-/

import Transformer.Perspective.SphereConvergence

open Filter
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- Radial projection fixes a tuple of unit vectors. -/
theorem tupleNormalize_coe (hd : 1 ≤ d) (X : SphereTuple d n) :
    tupleNormalize hd (fun i => (X i : EucSpace d)) = X := by
  simpa using tupleNormalize_smul hd (fun _ => one_pos) X

/-- **At unit vectors the gradient of `g` is the round gradient of `𝖤_β`:**
`(∇g(x))_i = Proj_{x_i} Σ_j e^{β⟨x_i, x_j⟩} x_j`. -/
theorem gradient_tupleEnergy_toLp_apply (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0)
    (X : SphereTuple d n) (i : Idx n) :
    (gradient (tupleEnergy hd β) (WithLp.toLp 2 fun j => (X j : EucSpace d))).ofLp i =
      proj d (X i) (∑ j, Real.exp (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) •
        (X j : EucSpace d)) := by
  rw [gradient_tupleEnergy_apply hd hβ (fun j => ne_zero_of_mem_unit_sphere (X j)) i,
    tupleNormalize_coe hd X, mem_sphere_zero_iff_norm.mp (X i).2, inv_one, one_smul]

/-- **Scaled round gradient flows converge to critical points:** if every vector
moves along `c_i(t)` times the round gradient of `𝖤_β`, with `c_i(t) ∈ [a, b]`,
`0 < a ≤ b`, and `β ≠ 0`, the configuration converges to a critical point of `𝖤_β`. -/
theorem exists_tendsto_isCriticalEBeta_of_hasDerivAt (hd : 1 ≤ d) {β a b : ℝ} (hβ : β ≠ 0)
    (ha : 0 < a) (hab : a ≤ b) {c : ℝ → Idx n → ℝ} (hc : ∀ t i, a ≤ c t i ∧ c t i ≤ b)
    {X : ℝ → SphereTuple d n}
    (hX : ∀ t i, HasDerivAt (fun s => (X s i : EucSpace d)) (c t i • proj d (X t i)
      (∑ j, Real.exp (β * inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d)) •
        (X t j : EucSpace d))) t) :
    ∃ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar ∧
      ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)) := by
  obtain ⟨ystar, h1, hcrit, hlim⟩ := exists_tendsto_of_hasDerivAt_blockScale hd (β := β) ha hab
    hc (y := fun t => WithLp.toLp 2 fun i => (X t i : EucSpace d))
    (fun t i => mem_sphere_zero_iff_norm.mp (X t i).2)
    (fun t => hasDerivAt_toLp_tuple_iff.2 fun i => by
      rw [blockScale_apply, gradient_tupleEnergy_toLp_apply hd hβ]
      exact hX t i)
  set Xstar : SphereTuple d n := fun i => ⟨ystar.ofLp i, mem_sphere_zero_iff_norm.2 (h1 i)⟩
  have hy : ystar = WithLp.toLp 2 fun i => (Xstar i : EucSpace d) := rfl
  refine ⟨Xstar, fun i => ?_, fun i => ((PiLp.continuous_apply 2 _ i).tendsto ystar).comp hlim⟩
  have hi := congrArg (fun z : TupleSpace d n => z.ofLp i) hcrit
  simp only [hy, gradient_tupleEnergy_toLp_apply hd hβ, PiLp.zero_apply] at hi
  exact hi

/-- The hypotheses of `exists_tendsto_isCriticalEBeta_of_hasDerivAt` hold for one token
at rest, `β = 1` and the scaling by `1`: the round gradient vanishes. -/
example : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (∀ (_ : ℝ) (_ : Idx 1), (1 : ℝ) ≤ 1 ∧
    (1 : ℝ) ≤ 1) ∧ let X : ℝ → SphereTuple 3 1 := fun _ _ => basePoint 2
    ∀ t i, HasDerivAt (fun s => (X s i : EucSpace 3)) ((1 : ℝ) • proj 3 (X t i)
      (∑ j, Real.exp (1 * inner (𝕜 := ℝ) (X t i : EucSpace 3) (X t j : EucSpace 3)) •
        (X t j : EucSpace 3))) t := by
  refine ⟨one_ne_zero, one_pos, le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩, fun t i => ?_⟩
  rw [Fin.sum_univ_one, proj_smul_self (mem_sphere_zero_iff_norm.mp (basePoint 2).2), smul_zero]
  exact hasDerivAt_const t _

/-- **Every solution of `USA` converges to a critical point of `𝖤_β`,** for `β ≠ 0`:
`USA` moves each vector along `n⁻¹` times the round gradient. -/
theorem exists_tendsto_isCriticalEBeta_of_USA (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0)
    {X : ℝ → SphereTuple d n} (hX : USA d n β X) :
    ∃ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar ∧
      ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)) := by
  obtain ⟨a, b, ha, hab, hc⟩ : ∃ a b : ℝ, 0 < a ∧ a ≤ b ∧
      ∀ (_ : ℝ) (_ : Idx n), a ≤ (n : ℝ)⁻¹ ∧ (n : ℝ)⁻¹ ≤ b := by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · exact ⟨1, 1, one_pos, le_rfl, fun _ i => i.elim0⟩
    · exact ⟨(n : ℝ)⁻¹, (n : ℝ)⁻¹, by positivity, le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩⟩
  exact exists_tendsto_isCriticalEBeta_of_hasDerivAt hd hβ ha hab hc fun t i => by
    rw [← proj_smul]
    exact hX t i

/-- **The partition function of `SA` between its bounds:**
`n e^{-|β|} ≤ Z_i ≤ n e^{|β|}`, since `|⟨x_i, x_j⟩| ≤ 1`. -/
theorem partitionSA_mem_Icc (β : ℝ) (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) :
    n * Real.exp (-|β|) ≤ partitionSA d n β X t i ∧ partitionSA d n β X t i ≤ n * Real.exp |β| := by
  have hb : ∀ k, Real.exp (-|β|) ≤ Real.exp (β * inner (𝕜 := ℝ) (X t i : EucSpace d) (X t k)) ∧
      Real.exp (β * inner (𝕜 := ℝ) (X t i : EucSpace d) (X t k)) ≤ Real.exp |β| := by
    intro k
    have h1 : |inner (𝕜 := ℝ) (X t i : EucSpace d) (X t k)| ≤ 1 := by
      have h := abs_real_inner_le_norm (X t i : EucSpace d) (X t k)
      rwa [mem_sphere_zero_iff_norm.mp (X t i).2, mem_sphere_zero_iff_norm.mp (X t k).2,
        mul_one] at h
    have h2 : |β * inner (𝕜 := ℝ) (X t i : EucSpace d) (X t k)| ≤ |β| := by
      rw [abs_mul]
      exact mul_le_of_le_one_right (abs_nonneg _) h1
    exact ⟨Real.exp_le_exp.2 (by linarith [neg_abs_le (β * inner (𝕜 := ℝ)
      (X t i : EucSpace d) (X t k))]), Real.exp_le_exp.2 ((le_abs_self _).trans h2)⟩
  unfold partitionSA
  refine ⟨?_, ?_⟩
  · calc (n : ℝ) * Real.exp (-|β|) = ∑ _k : Idx n, Real.exp (-|β|) := by simp
      _ ≤ _ := Finset.sum_le_sum fun k _ => (hb k).1
  · calc _ ≤ ∑ _k : Idx n, Real.exp |β| := Finset.sum_le_sum fun k _ => (hb k).2
      _ = _ := by simp

/-- **Every solution of `SA` converges to a critical point of `𝖤_β`,** for `β ≠ 0`:
`SA` moves each vector along `Z_i(t)⁻¹` times the round gradient, and
`Z_i(t)⁻¹ ∈ [(n e^{|β|})⁻¹, (n e^{-|β|})⁻¹]`. -/
theorem exists_tendsto_isCriticalEBeta_of_SA (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0)
    {X : ℝ → SphereTuple d n} (hX : SA d n β X) :
    ∃ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar ∧
      ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)) := by
  obtain ⟨a, b, ha, hab, hc⟩ : ∃ a b : ℝ, 0 < a ∧ a ≤ b ∧ ∀ t i,
      a ≤ (partitionSA d n β X t i)⁻¹ ∧ (partitionSA d n β X t i)⁻¹ ≤ b := by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · exact ⟨1, 1, one_pos, le_rfl, fun _ i => i.elim0⟩
    have hlo : (0 : ℝ) < n * Real.exp (-|β|) := by positivity
    refine ⟨(n * Real.exp |β|)⁻¹, (n * Real.exp (-|β|))⁻¹, by positivity,
      inv_anti₀ hlo (mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (by linarith [abs_nonneg β]))
        (Nat.cast_nonneg n)), fun t i => ?_⟩
    have h := partitionSA_mem_Icc β X t i
    exact ⟨inv_anti₀ (hlo.trans_le h.1) h.2, inv_anti₀ hlo h.1⟩
  exact exists_tendsto_isCriticalEBeta_of_hasDerivAt hd hβ ha hab hc fun t i => by
    rw [← proj_smul]
    exact hX t i

/-- The hypothesis `β ≠ 0` of `exists_tendsto_isCriticalEBeta_of_USA` and
`exists_tendsto_isCriticalEBeta_of_SA` holds for `β = 1`, and one token at rest solves
both flows. -/
example : (1 : ℝ) ≠ 0 ∧ SA 3 1 1 (fun _ _ => basePoint 2) :=
  ⟨one_ne_zero, SA_const_consensus 3 1 one_pos 1 _⟩

end Perspective
end Transformer
