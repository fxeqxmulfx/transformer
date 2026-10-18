/-
# The energy of normalized tuples on `(ℝ^d)^n` with the `ℓ²` norm

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The gradient flows of `Perspective.SphereRadialGradient` live on tuples of vectors of
`ℝ^d`; with the `ℓ²` norm (`TupleSpace`) these form an inner product space, the setting
of the Łojasiewicz theorem and of saddle avoidance.  On it, `g(x) = 𝖤_β(x/‖x‖)`
(`tupleEnergy`) is real-analytic wherever no vector vanishes
(`analyticOnNhd_tupleEnergy`), since `‖y‖⁻¹ = exp(-½ log⟨y, y⟩)` is
(`analyticAt_norm_inv`).  The gradient of a differentiable function of `ℓ²` tuples is
the tuple of its partial gradients (`gradient_ofLp_apply`), so `∇g` has the vectors
`‖x_i‖⁻¹ Proj_{u_i} Σ_j e^{β⟨u_i, u_j⟩} u_j` of `hasGradientAt_selfEnergy_update`
(`gradient_tupleEnergy_apply`), and vanishes at a tuple of unit vectors exactly at the
critical points of `𝖤_β` (`gradient_tupleEnergy_eq_zero_iff`).

Source: arXiv:2312.10794v5, §3.3 and Appendix A, where `𝖤_β` is real-analytic on
`(𝕊^{d-1})^n`.
-/

import Transformer.Perspective.SphereRadialGradient
import Mathlib.Analysis.Calculus.FDeriv.Analytic
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Calculus.FDeriv.WithLp
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic

open Filter
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The gradient of a function of `ℓ²` tuples is the tuple of partial gradients:**
where `f` is differentiable, the `i`-th vector of `∇f(x)` is the gradient of `f` in
the `i`-th vector alone. -/
theorem gradient_ofLp_apply {ι : Type*} [Fintype ι] [DecidableEq ι] {E : ι → Type*}
    [∀ i, NormedAddCommGroup (E i)] [∀ i, InnerProductSpace ℝ (E i)]
    [∀ i, CompleteSpace (E i)] {f : PiLp 2 E → ℝ} {x : PiLp 2 E}
    (hf : DifferentiableAt ℝ f x) (i : ι) :
    (gradient f x).ofLp i =
      gradient (fun v => f (WithLp.toLp 2 (Function.update x.ofLp i v))) (x.ofLp i) := by
  have hφ : HasFDerivAt (fun v => WithLp.toLp 2 (Function.update x.ofLp i v))
      ((PiLp.continuousLinearEquiv 2 ℝ E).symm.toContinuousLinearMap.comp
        (.pi (Pi.single i (.id ℝ (E i))))) (x.ofLp i) :=
    (PiLp.hasFDerivAt_toLp 2 _).comp (x.ofLp i) (hasFDerivAt_update x.ofLp (x.ofLp i))
  have hf' : HasFDerivAt f (fderiv ℝ f x)
      (WithLp.toLp 2 (Function.update x.ofLp i (x.ofLp i))) := by
    simpa using hf.hasFDerivAt
  refine (HasGradientAt.gradient ?_).symm
  rw [hasGradientAt_iff_hasFDerivAt]
  refine (hf'.comp (x.ofLp i) hφ).congr_fderiv (ContinuousLinearMap.ext fun w => ?_)
  rw [ContinuousLinearMap.comp_apply, ← toDual_gradient, InnerProductSpace.toDual_apply_apply,
    InnerProductSpace.toDual_apply_apply, PiLp.inner_apply, Finset.sum_eq_single i]
  · simp
  · intro j _ hj
    simp [hj]
  · simp

/-- The hypothesis of `gradient_ofLp_apply` holds for a constant function. -/
example : DifferentiableAt ℝ (fun _ : PiLp 2 (fun _ : Fin 2 => ℝ) => (0 : ℝ)) 0 :=
  differentiableAt_const _

/-- Tuples of `n` vectors of `ℝ^d` with the `ℓ²` norm, an inner product space. -/
abbrev TupleSpace (d n : ℕ) : Type := PiLp 2 fun _ : Idx n => EucSpace d

/-- **The energy of normalized tuples:** `g(x) = 𝖤_β(x/‖x‖)`. -/
noncomputable def tupleEnergy (hd : 1 ≤ d) (β : ℝ) (x : TupleSpace d n) : ℝ :=
  selfEnergy d n β (tupleNormalize hd x.ofLp)

/-- The tuples of nonzero vectors form an open set. -/
theorem isOpen_setOf_ofLp_ne_zero :
    IsOpen {x : TupleSpace d n | ∀ i, x.ofLp i ≠ 0} := by
  rw [Set.ofPred_forall]
  exact isOpen_iInter_of_finite fun i =>
    isOpen_ne_fun (PiLp.continuous_apply 2 _ i) continuous_const

/-- **`y ↦ ‖y‖⁻¹` is real-analytic off `0`:** it is `exp(-½ log⟨y, y⟩)` there. -/
theorem analyticAt_norm_inv {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {z : E} (hz : z ≠ 0) : AnalyticAt ℝ (fun w : E => ‖w‖⁻¹) z := by
  have hq : AnalyticAt ℝ (fun w : E => inner (𝕜 := ℝ) w w) z :=
    ((innerSL ℝ : E →L[ℝ] E →L[ℝ] ℝ).analyticAt_bilinear (z, z)).comp
      (f := fun w => (w, w)) (analyticAt_id.prod analyticAt_id)
  have hpos : 0 < inner (𝕜 := ℝ) z z := real_inner_self_pos.2 hz
  have hl : AnalyticAt ℝ (fun w : E => Real.log (inner (𝕜 := ℝ) w w)) z :=
    AnalyticAt.comp (f := fun w : E => inner (𝕜 := ℝ) w w) (analyticAt_log hpos) hq
  refine ((analyticAt_const (v := -(1 / 2 : ℝ))).fun_mul hl).rexp'.congr ?_
  filter_upwards [eventually_ne_nhds hz] with w hw
  rw [real_inner_self_eq_norm_sq, Real.log_pow, Nat.cast_ofNat,
    show -(1 / 2 : ℝ) * (2 * Real.log ‖w‖) = -Real.log ‖w‖ by ring, Real.exp_neg,
    Real.exp_log (norm_pos_iff.2 hw)]

/-- The hypothesis of `analyticAt_norm_inv` holds for a basis vector of `ℝ²`. -/
example : (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) ≠ 0 := by simp

/-- **`g` is real-analytic on the tuples of nonzero vectors:** there it is
`(2β)⁻¹ Σ_k Σ_l exp(β ⟨x_k/‖x_k‖, x_l/‖x_l‖⟩)`. -/
theorem analyticOnNhd_tupleEnergy (hd : 1 ≤ d) (β : ℝ) :
    AnalyticOnNhd ℝ (tupleEnergy (n := n) hd β) {x | ∀ i, x.ofLp i ≠ 0} := by
  intro x hx
  set N : TupleSpace d n → Idx n → EucSpace d := fun y k => ‖y.ofLp k‖⁻¹ • y.ofLp k
  have hN : ∀ k, AnalyticAt ℝ (fun y => N y k) x := fun k => by
    have hp : AnalyticAt ℝ (fun y : TupleSpace d n => y.ofLp k) x :=
      (PiLp.proj 2 (𝕜 := ℝ) (fun _ : Idx n => EucSpace d) k).analyticAt x
    exact (AnalyticAt.comp (f := fun y : TupleSpace d n => y.ofLp k)
      (analyticAt_norm_inv (hx k)) hp).fun_smul hp
  have hinner : ∀ k l, AnalyticAt ℝ (fun y => inner (𝕜 := ℝ) (N y k) (N y l)) x :=
    fun k l => ((innerSL ℝ : EucSpace d →L[ℝ] EucSpace d →L[ℝ] ℝ).analyticAt_bilinear
      (N x k, N x l)).comp (f := fun y => (N y k, N y l)) ((hN k).prod (hN l))
  have hsum : AnalyticAt ℝ (fun y => (2 * β)⁻¹ * ∑ k, ∑ l,
      Real.exp (β * inner (𝕜 := ℝ) (N y k) (N y l))) x :=
    analyticAt_const.fun_mul (Finset.analyticAt_fun_sum _ fun k _ =>
      Finset.analyticAt_fun_sum _ fun l _ => (analyticAt_const.fun_mul (hinner k l)).rexp')
  refine hsum.congr ?_
  filter_upwards [isOpen_setOf_ofLp_ne_zero.mem_nhds hx] with y hy
  simp only [tupleEnergy, selfEnergy, particleEnergy, ContinuousLinearMap.id_apply, N]
  congr 1
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
  rw [show (tupleNormalize hd y.ofLp k : EucSpace d) = _ from coe_sphereNormalize hd (hy k),
    show (tupleNormalize hd y.ofLp l : EucSpace d) = _ from coe_sphereNormalize hd (hy l)]

/-- **The gradient of `g`:** for `β ≠ 0` and nonzero vectors, with `u = x/‖x‖`,
`(∇g(x))_i = ‖x_i‖⁻¹ Proj_{u_i} Σ_j e^{β⟨u_i, u_j⟩} u_j`. -/
theorem gradient_tupleEnergy_apply (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0) {x : TupleSpace d n}
    (hx : ∀ i, x.ofLp i ≠ 0) (i : Idx n) :
    (gradient (tupleEnergy hd β) x).ofLp i =
      ‖x.ofLp i‖⁻¹ • proj d (tupleNormalize hd x.ofLp i) (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (tupleNormalize hd x.ofLp i : EucSpace d)
          (tupleNormalize hd x.ofLp j : EucSpace d)) •
            (tupleNormalize hd x.ofLp j : EucSpace d)) := by
  rw [gradient_ofLp_apply (analyticOnNhd_tupleEnergy hd β x hx).differentiableAt i]
  exact (hasGradientAt_selfEnergy_update hd hβ (hx i)).gradient

/-- **The critical points of `g` on nonzero tuples are those of `𝖤_β`:** `∇g(x) = 0`
exactly when `x/‖x‖` is a critical point of `𝖤_β` on `(𝕊^{d-1})^n`. -/
theorem gradient_tupleEnergy_eq_zero_iff (hd : 1 ≤ d) {β : ℝ} (hβ : β ≠ 0)
    {x : TupleSpace d n} (hx : ∀ i, x.ofLp i ≠ 0) :
    gradient (tupleEnergy hd β) x = 0 ↔ IsCriticalEBeta d n β (tupleNormalize hd x.ofLp) := by
  refine ⟨fun h i => ?_, fun h => PiLp.ext fun i => ?_⟩
  · have hi := congrArg (fun y : TupleSpace d n => y.ofLp i) h
    simp only [gradient_tupleEnergy_apply hd hβ hx i, PiLp.zero_apply] at hi
    exact (smul_eq_zero.1 hi).resolve_left (inv_ne_zero (norm_ne_zero_iff.2 (hx i)))
  · rw [gradient_tupleEnergy_apply hd hβ hx i, PiLp.zero_apply, h i, smul_zero]

/-- The hypotheses of `analyticOnNhd_tupleEnergy`, `gradient_tupleEnergy_apply` and
`gradient_tupleEnergy_eq_zero_iff` hold for two copies of a unit vector of `ℝ³` and
`β = 1`. -/
example : 1 ≤ 3 ∧ (1 : ℝ) ≠ 0 ∧ ∀ i, (WithLp.toLp 2 fun _ : Idx 2 =>
    (basePoint 2 : EucSpace 3) : TupleSpace 3 2).ofLp i ≠ 0 :=
  ⟨by norm_num, one_ne_zero, fun _ => ne_zero_of_mem_unit_sphere _⟩

end Perspective
end Transformer
