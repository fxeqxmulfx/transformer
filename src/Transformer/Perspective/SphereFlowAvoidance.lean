/-
# `USA` and `SA` almost never converge to an unsynchronized critical point

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

On the radii `r_i = ‖x_i‖`, `USA` is `ẏ_i = n⁻¹ ‖y_i‖² ∇_i g(y)` and `SA` is
`ẏ_i = ‖y_i‖² Z_i(y/‖y‖)⁻¹ ∇_i g(y)` (`usa_iff_gradient`, `sa_iff_gradient`): block
scalings of `∇g` by positive scales, smooth off the zero vectors (the partition function
`Z_i` of the normalized tuple is analytic there, `analyticAt_coe_tupleNormalize`).  So
`ae_not_tendsto_nonTrivial_of_blockScale` applies to both: for `d ≥ 3`, `β > 0` and
almost every `x`, the solution from `x/‖x‖` converges to no critical point of `𝖤_β`
with two distinct tokens (`ae_USA_not_tendsto_nonTrivial`,
`ae_SA_not_tendsto_nonTrivial`).  The field of `USA` extends to a smooth field of
`(ℝ^d)^n` (`usaAmbientField`), so its solutions are unique (`USA_unique`), as those of
`SA` are (`SA_unique`).

Source: arXiv:2312.10794v5, §6, `thm: boumal`, and §2 for the
well-posedness of the Cauchy problems.
-/

import Transformer.Perspective.SphereSaddleAvoidance
import Transformer.GlobalFlow.Existence

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

/-- **The ambient field of `USA`** on all of `(ℝ^d)^n`:
`x ↦ (Proj_{x_i} n⁻¹ Σ_j e^{β⟨x_i, x_j⟩} x_j)_i`. -/
noncomputable def usaAmbientField (β : ℝ) (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  fun i => proj d (Y i) ((n : ℝ)⁻¹ • ∑ j, Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j)) • Y j)

theorem contDiff_usaAmbientField (β : ℝ) :
    ContDiff ℝ 1 (usaAmbientField (d := d) (n := n) β) := by
  refine contDiff_pi.2 fun i => ?_
  have hc : ∀ j, ContDiff ℝ 1 fun Y : Idx n → EucSpace d => Y j := fun j => by fun_prop
  have he : ∀ j, ContDiff ℝ 1 fun Y : Idx n → EucSpace d =>
      Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j)) := fun j =>
    (contDiff_const.mul ((hc i).inner ℝ (hc j))).exp
  have hsum : ContDiff ℝ 1 fun Y : Idx n → EucSpace d =>
      ∑ j, Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j)) • Y j :=
    ContDiff.sum fun j _ => (he j).smul (hc j)
  have hS : ContDiff ℝ 1 fun Y : Idx n → EucSpace d =>
      (n : ℝ)⁻¹ • ∑ j, Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j)) • Y j :=
    hsum.const_smul (n : ℝ)⁻¹
  exact hS.sub (((hc i).inner ℝ hS).smul (hc i))

/-- **The solution of `USA` is unique:** two solutions with the same initial
configuration coincide, both following the smooth ambient field. -/
theorem USA_unique (β : ℝ) {X Y : ℝ → SphereTuple d n} (hX : USA d n β X)
    (hY : USA d n β Y) (h0 : X 0 = Y 0) : X = Y := by
  have hpath : ∀ W : ℝ → SphereTuple d n, USA d n β W → ∀ t,
      HasDerivAt (fun s i => (W s i : EucSpace d)) (usaAmbientField β fun i => (W t i : EucSpace d))
        t := fun W hW t => hasDerivAt_pi.2 fun i => hW t i
  have heq := GlobalFlow.eq_of_hasDerivAt (contDiff_usaAmbientField β).locallyLipschitz
    (hpath X hX) (hpath Y hY) (by rw [h0])
  funext t i
  exact Subtype.ext (congrFun (congrFun heq t) i)

/-- The hypotheses of `USA_unique` hold for one particle at rest, twice. -/
example : USA 1 1 0 (fun _ _ => basePoint 0) := fun t i => by
  simpa [proj, Finset.univ_unique, proj_smul_self] using hasDerivAt_const t _

end Perspective
end Transformer
