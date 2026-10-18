/-
# Almost every start clusters, for `d ≥ 3` and `β > 0`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

On the radii `r_i = ‖x_i‖`, `USA` is `ẏ_i = n⁻¹ ‖y_i‖² ∇_i g(y)` and `SA` is
`ẏ_i = ‖y_i‖² Z_i(y/‖y‖)⁻¹ ∇_i g(y)` (`usa_iff_gradient`, `sa_iff_gradient`): block
scalings of `∇g` by positive scales, smooth off the zero vectors (the partition function
`Z_i` of the normalized tuple is analytic there, `analyticAt_coe_tupleNormalize`).  So
`ae_not_tendsto_nonTrivial_of_blockScale` applies to both: for `d ≥ 3`, `β > 0` and
almost every `x`, the solution from `x/‖x‖` converges to no critical point of `𝖤_β`
with two distinct tokens (`ae_USA_not_tendsto_nonTrivial`,
`ae_SA_not_tendsto_nonTrivial`).

Every solution converges to a critical point of `𝖤_β`
(`exists_tendsto_isCriticalEBeta_of_SA`, `exists_tendsto_isCriticalEBeta_of_USA`: the
Łojasiewicz theorem), so for almost every start the limit is a consensus; the solution
being unique (`SA_unique`, `USA_unique`), the start lies in `𝒮_β` and in its `USA`
counterpart (`ae_mem_clusteringSet_of_pos`).  The step from limits to clusters holds for
any dynamics with unique solutions (`exists_cluster_of_not_tendsto_nonTrivial`).

Source: arXiv:2312.10794v5, §6, `thm: boumal`, and the outline of the gradient-flow
proofs at the end of §5; arXiv:2405.18273, `thm:spheres`, for the landscape.
-/

import Transformer.Perspective.SphereSaddleAvoidance
import Transformer.Perspective.CircleClustering

open Filter MeasureTheory
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The normalized vectors are analytic off the zero vectors:** near a tuple of
nonzero vectors, `y ↦ y_k/‖y_k‖`. -/
theorem analyticAt_coe_tupleNormalize (hd : 1 ≤ d) {x : TupleSpace d n}
    (hx : ∀ i, x.ofLp i ≠ 0) (k : Idx n) :
    AnalyticAt ℝ (fun y : TupleSpace d n => (tupleNormalize hd y.ofLp k : EucSpace d)) x := by
  have hp : AnalyticAt ℝ (fun y : TupleSpace d n => y.ofLp k) x :=
    (PiLp.proj 2 (𝕜 := ℝ) (fun _ : Idx n => EucSpace d) k).analyticAt x
  refine ((AnalyticAt.comp (f := fun y : TupleSpace d n => y.ofLp k)
    (analyticAt_norm_inv (hx k)) hp).fun_smul hp).congr ?_
  filter_upwards [isOpen_setOf_ofLp_ne_zero.mem_nhds hx] with y hy
  exact (coe_sphereNormalize hd (hy k)).symm

/-- The hypothesis of `analyticAt_coe_tupleNormalize` holds for two copies of a unit
vector of `ℝ³`. -/
example : ∀ i, (WithLp.toLp 2 fun _ : Idx 2 => (basePoint 2 : EucSpace 3) :
    TupleSpace 3 2).ofLp i ≠ 0 := fun _ => ne_zero_of_mem_unit_sphere _

/-- **Almost no solution of `USA` converges to a critical point with two distinct
tokens**, for `d ≥ 3` and `β > 0`, started from `x/‖x‖` for almost every `x`. -/
theorem ae_USA_not_tendsto_nonTrivial (hd1 : 1 ≤ d) (hd : 3 ≤ d) {β : ℝ} (hβ : 0 < β) :
    ∀ᵐ x ∂(volume : Measure (Idx n → EucSpace d)), ∀ X : ℝ → SphereTuple d n,
      X 0 = tupleNormalize hd1 x → USA d n β X →
      ∀ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar → NonTrivialTuple d n Xstar →
        ¬∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)) := by
  have hw : ∀ y : TupleSpace d n, (∀ i, y.ofLp i ≠ 0) →
      ContDiffAt ℝ 1 (fun y : TupleSpace d n => fun i => (n : ℝ)⁻¹ * ‖y.ofLp i‖ ^ 2) y :=
    fun y _ => contDiffAt_pi.2 fun i => contDiffAt_const.mul ((contDiff_norm_sq ℝ).comp
      (PiLp.proj 2 (𝕜 := ℝ) (fun _ : Idx n => EucSpace d) i).contDiff).contDiffAt
  filter_upwards [ae_not_tendsto_nonTrivial_of_blockScale hd1 hd hβ hw
    (fun y hy i => mul_pos (inv_pos.2 (Nat.cast_pos.2 i.pos)) (pow_pos (norm_pos_iff.2 (hy i)) 2)),
    ae_forall_ne_zero hd1] with x hx hx0
  intro X hX0 hX
  have hr : ∀ i, 0 < ‖x i‖ := fun i => norm_pos_iff.2 (hx0 i)
  refine hx X hX0 fun t => hasDerivAt_toLp_tuple_iff.2 fun i => ?_
  rw [blockScale_apply, gradient_ofLp_apply ((analyticOnNhd_tupleEnergy hd1 β _ fun j =>
    smul_ne_zero (hr j).ne' (ne_zero_of_mem_unit_sphere _)).differentiableAt) i]
  convert (usa_iff_gradient hd1 hβ.ne' hr X).1 hX t i using 2
  · rw [norm_smul, norm_norm, mem_sphere_zero_iff_norm.mp (X t i).2, mul_one]
  · rfl

/-- The hypotheses of `ae_USA_not_tendsto_nonTrivial` hold for `d = 3`, `β = 1`. -/
example := ae_USA_not_tendsto_nonTrivial (d := 3) (n := 2) (by norm_num) le_rfl one_pos

/-- **Almost no solution of `SA` converges to a critical point with two distinct
tokens**, for `d ≥ 3` and `β > 0`, started from `x/‖x‖` for almost every `x`. -/
theorem ae_SA_not_tendsto_nonTrivial (hd1 : 1 ≤ d) (hd : 3 ≤ d) {β : ℝ} (hβ : 0 < β) :
    ∀ᵐ x ∂(volume : Measure (Idx n → EucSpace d)), ∀ X : ℝ → SphereTuple d n,
      X 0 = tupleNormalize hd1 x → SA d n β X →
      ∀ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar → NonTrivialTuple d n Xstar →
        ¬∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)) := by
  set Z : TupleSpace d n → Idx n → ℝ := fun y i =>
    particlePartition d n β (ContinuousLinearMap.id ℝ (EucSpace d)) (tupleNormalize hd1 y.ofLp) i
  have hZ : ∀ y i, 0 < Z y i := fun y i =>
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩
  have hw : ∀ y : TupleSpace d n, (∀ i, y.ofLp i ≠ 0) →
      ContDiffAt ℝ 1 (fun y : TupleSpace d n => fun i => ‖y.ofLp i‖ ^ 2 / Z y i) y := by
    refine fun y hy => contDiffAt_pi.2 fun i => ContDiffAt.div ((contDiff_norm_sq ℝ).comp
      (PiLp.proj 2 (𝕜 := ℝ) (fun _ : Idx n => EucSpace d) i).contDiff).contDiffAt ?_ (hZ y i).ne'
    have hN := analyticAt_coe_tupleNormalize hd1 hy
    refine (Finset.analyticAt_fun_sum _ fun j _ => (analyticAt_const.fun_mul
      (((innerSL ℝ : EucSpace d →L[ℝ] EucSpace d →L[ℝ] ℝ).analyticAt_bilinear _).comp
        ((hN i).prod (hN j)))).rexp').contDiffAt
  filter_upwards [ae_not_tendsto_nonTrivial_of_blockScale hd1 hd hβ hw
    (fun y hy i => div_pos (pow_pos (norm_pos_iff.2 (hy i)) 2) (hZ y i)),
    ae_forall_ne_zero hd1] with x hx hx0
  intro X hX0 hX
  have hr : ∀ i, 0 < ‖x i‖ := fun i => norm_pos_iff.2 (hx0 i)
  refine hx X hX0 fun t => hasDerivAt_toLp_tuple_iff.2 fun i => ?_
  rw [blockScale_apply, gradient_ofLp_apply ((analyticOnNhd_tupleEnergy hd1 β _ fun j =>
    smul_ne_zero (hr j).ne' (ne_zero_of_mem_unit_sphere _)).differentiableAt) i]
  convert (sa_iff_gradient hd1 hβ.ne' hr X).1 hX t i using 2
  · change ‖‖x i‖ • (X t i : EucSpace d)‖ ^ 2 / particlePartition d n β _
      (tupleNormalize hd1 fun j => ‖x j‖ • (X t j : EucSpace d)) i = _
    rw [tupleNormalize_smul hd1 hr, norm_smul, norm_norm, mem_sphere_zero_iff_norm.mp (X t i).2,
      mul_one]
  · rfl

/-- The hypotheses of `ae_SA_not_tendsto_nonTrivial` hold for `d = 3`, `β = 1`. -/
example := ae_SA_not_tendsto_nonTrivial (d := 3) (n := 2) (by norm_num) le_rfl one_pos


/-- **No unsynchronized limit means one cluster.**  For dynamics `D` with unique
solutions that converge to critical points of `𝖤_β`: if no solution from `X₀` converges
to a critical point with two distinct tokens, then every solution from `X₀` converges to
one point `x⋆` of the sphere. -/
theorem exists_cluster_of_not_tendsto_nonTrivial (hn : 0 < n) {β : ℝ}
    {D : (ℝ → SphereTuple d n) → Prop} (hD : ∀ X Y, D X → D Y → X 0 = Y 0 → X = Y)
    (hlim : ∀ X, D X → ∃ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar ∧
      ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)))
    {X₀ : SphereTuple d n}
    (hX₀ : ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → D X →
      ∀ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar → NonTrivialTuple d n Xstar →
        ¬∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d))) :
    ∃ x_star : SSphere d, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → D X →
      ∀ i, Tendsto (fun t => (X t i : EucSpace d) - x_star) atTop (𝓝 0) := by
  by_cases hex : ∃ X, X 0 = X₀ ∧ D X
  · obtain ⟨X, hX0, hX⟩ := hex
    obtain ⟨Xstar, hcrit, hconv⟩ := hlim X hX
    have htriv : ∀ i j, (Xstar i : EucSpace d) = Xstar j := by
      by_contra h
      push Not at h
      exact hX₀ X hX0 hX Xstar hcrit h hconv
    refine ⟨Xstar ⟨0, hn⟩, fun Y hY0 hY i => ?_⟩
    rw [← hD X Y hX hY (hX0.trans hY0.symm), htriv ⟨0, hn⟩ i]
    exact tendsto_sub_nhds_zero_iff.2 (hconv i)
  · push Not at hex
    exact ⟨X₀ ⟨0, hn⟩, fun X hX0 hX => absurd hX (hex X hX0)⟩

/-- The hypotheses of `exists_cluster_of_not_tendsto_nonTrivial` hold for two tokens of
`𝕊²` under dynamics with no solution. -/
example := exists_cluster_of_not_tendsto_nonTrivial (d := 3) (n := 2) (β := 1) two_pos
  (D := fun _ => False) (fun _ _ h => h.elim) (fun _ h => h.elim)
  (X₀ := fun _ => basePoint 2) fun _ _ h => h.elim

/-- **Almost every start clusters, for `d ≥ 3` and `β > 0`:** under the uniform law,
almost every initial sequence lies in `𝒮_β` and in its `USA` counterpart. -/
theorem ae_mem_clusteringSet_of_pos (hd : 3 ≤ d) (hn : 0 < n) {β : ℝ} (hβ : 0 < β)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P) :
    ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β ∧ X₀ ∈ clusteringSetUSA d n β := by
  have hd1 : 1 ≤ d := by omega
  refine (ae_iff_ae_volume hd1 hP).2 ?_
  filter_upwards [ae_SA_not_tendsto_nonTrivial hd1 hd hβ,
    ae_USA_not_tendsto_nonTrivial hd1 hd hβ] with x hSA hUSA
  exact ⟨exists_cluster_of_not_tendsto_nonTrivial hn (SA_unique β)
      (fun _ hX => exists_tendsto_isCriticalEBeta_of_SA hd1 hβ.ne' hX) hSA,
    exists_cluster_of_not_tendsto_nonTrivial hn (USA_unique β)
      (fun _ hX => exists_tendsto_isCriticalEBeta_of_USA hd1 hβ.ne' hX) hUSA⟩

/-- The hypotheses of `ae_mem_clusteringSet_of_pos` hold for two tokens of `𝕊²`, `β = 1`
and the product of shell laws. -/
example := ae_mem_clusteringSet_of_pos (d := 3) (n := 2) le_rfl two_pos one_pos _
  (uniformTuple_shellLaw (d := 3) (by norm_num))

end Perspective
end Transformer
