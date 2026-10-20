module

public import Transformer.Perspective.SphereTupleEnergy
public import Transformer.Landscape.Lojasiewicz.LojasiewiczInnerProduct
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Łojasiewicz convergence on the product of unit spheres

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Written on `ℓ²` tuples, a solution of `USA` or `SA` solves `ẏ = M(t) ∇g(y)` for
`g = 𝖤_β ∘ normalize`, where `M(t)` scales the `i`-th vector (`blockScale`) by a
number in a fixed interval `[a, b]`, `0 < a` (`Perspective.SphereFlowLimit`). Such a
scaling is symmetric with `a I ≼ M(t) ≼ b I` (`inner_blockScale_self_mem_Icc`), `g` is
real-analytic off the tuples with a zero vector, and the trajectory stays on the compact
product of unit spheres, so the Łojasiewicz theorem applies: the trajectory converges
to a tuple of unit vectors where `∇g` vanishes
(`exists_tendsto_of_hasDerivAt_blockScale`).

Source: arXiv:2312.10794v5, §5 (the outline after `thm: beta.interval`: "the
celebrated Łojasiewicz theorem implies that for any initial condition, the gradient
flow converges to some critical point of `𝖤_β`"), with `lem: loj` of
arXiv:2510.22026v2 for the theorem itself.
-/

@[expose] public section

open Filter
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Block scaling** `(v_i)_i ↦ (c_i v_i)_i` of `ℓ²` tuples. -/
noncomputable def blockScale (c : Idx n → ℝ) : TupleSpace d n →L[ℝ] TupleSpace d n :=
  LinearMap.toContinuousLinearMap
    { toFun := fun v => WithLp.toLp 2 fun i => c i • v.ofLp i
      map_add' := fun v w => by ext i; simp [smul_add]
      map_smul' := fun r v => by ext i; simp [smul_comm (c i) r] }

@[simp] theorem blockScale_apply (c : Idx n → ℝ) (v : TupleSpace d n) (i : Idx n) :
    (blockScale c v).ofLp i = c i • v.ofLp i := rfl

theorem inner_blockScale (c : Idx n → ℝ) (v w : TupleSpace d n) :
    inner (𝕜 := ℝ) (blockScale c v) w = ∑ i, c i * inner (𝕜 := ℝ) (v.ofLp i) (w.ofLp i) := by
  rw [PiLp.inner_apply]
  exact Finset.sum_congr rfl fun i _ => by rw [blockScale_apply, real_inner_smul_left]

/-- Scalings by numbers in `[a, b]` lie between `a` and `b` in the quadratic form. -/
theorem inner_blockScale_self_mem_Icc {a b : ℝ} {c : Idx n → ℝ} (hc : ∀ i, a ≤ c i ∧ c i ≤ b)
    (v : TupleSpace d n) :
    a * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) (blockScale c v) v ∧
      inner (𝕜 := ℝ) (blockScale c v) v ≤ b * ‖v‖ ^ 2 := by
  rw [inner_blockScale, PiLp.norm_sq_eq_of_L2 (fun _ : Idx n => EucSpace d) v, Finset.mul_sum,
    Finset.mul_sum]
  simp only [real_inner_self_eq_norm_sq]
  exact ⟨Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hc i).1 (sq_nonneg _),
    Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hc i).2 (sq_nonneg _)⟩

/-- A curve of `ℓ²` tuples has a derivative exactly when each of its vectors has. -/
theorem hasDerivAt_toLp_tuple_iff {x : ℝ → Idx n → EucSpace d} {v : TupleSpace d n} {t : ℝ} :
    HasDerivAt (fun s => (WithLp.toLp 2 (x s) : TupleSpace d n)) v t ↔
      ∀ i, HasDerivAt (fun s => x s i) (v.ofLp i) t := by
  rw [← hasDerivAt_pi]
  let e := PiLp.continuousLinearEquiv 2 ℝ fun _ : Idx n => EucSpace d
  exact ⟨fun h => e.hasFDerivAt.comp_hasDerivAt t h, fun h => e.symm.hasFDerivAt.comp_hasDerivAt t h⟩

/-- **Forward Łojasiewicz on the product of unit spheres:** a curve with
`ẏ = M(t) ∇g(y)` and positive bounded block scalings converges to a critical
unit tuple. The equations and bounds are needed only for `t ≥ 0`, as in
arXiv:2510.22026v2, Appendix D, `lem:loj` and Step 2. -/
theorem exists_tendsto_of_hasDerivAt_blockScale_forward (hd : 1 ≤ d) {β a b : ℝ} (ha : 0 < a)
    (hab : a ≤ b) {c : ℝ → Idx n → ℝ} (hc : ∀ t, 0 ≤ t → ∀ i, a ≤ c t i ∧ c t i ≤ b)
    {y : ℝ → TupleSpace d n} (hy1 : ∀ t, 0 ≤ t → ∀ i, ‖(y t).ofLp i‖ = 1)
    (hy : ∀ t, 0 ≤ t → HasDerivAt y (blockScale (c t) (gradient (tupleEnergy hd β) (y t))) t) :
    ∃ ystar : TupleSpace d n, (∀ i, ‖ystar.ofLp i‖ = 1) ∧
      gradient (tupleEnergy hd β) ystar = 0 ∧ Tendsto y atTop (𝓝 ystar) := by
  set K : Set (TupleSpace d n) := {x | ∀ i, ‖x.ofLp i‖ = 1}
  have hKc : IsClosed K := by
    simp only [K]
    rw [Set.ofPred_forall]
    exact isClosed_iInter fun i =>
      isClosed_eq (continuous_norm.comp (PiLp.continuous_apply 2 _ i)) continuous_const
  have hK : IsCompact K := by
    refine Metric.isCompact_of_isClosed_isBounded hKc
      ((Metric.isBounded_closedBall (x := 0) (r := n + 1)).subset fun x hx => ?_)
    have hx2 : ‖x‖ ^ 2 = n := by
      rw [PiLp.norm_sq_eq_of_L2 (fun _ : Idx n => EucSpace d) x]
      simp [show ∀ i, ‖x.ofLp i‖ = 1 from hx]
    rw [mem_closedBall_zero_iff]
    nlinarith [norm_nonneg x]
  have hgrad : ∀ x, gradient (-tupleEnergy hd β) x = -gradient (tupleEnergy (n := n) hd β) x :=
    fun x => by simp only [gradient, fderiv_neg, map_neg]
  have hb : 0 < b := ha.trans_le hab
  obtain ⟨ystar, -, hlim, hcrit⟩ := Landscape.Lojasiewicz.lojasiewicz_modulated_of_finiteDimensional
    {x | ∀ i, x.ofLp i ≠ 0} isOpen_setOf_ofLp_ne_zero (-tupleEnergy hd β)
    (analyticOnNhd_tupleEnergy hd β).neg (4 * b / a) (fun _ => a / 2) (fun t => blockScale (c t))
    (fun _ _ => half_pos ha)
    (fun t _ v w => by
      rw [inner_blockScale, real_inner_comm, inner_blockScale]
      exact Finset.sum_congr rfl fun i _ => by rw [real_inner_comm])
    (fun t ht v hv => by
      have h := (inner_blockScale_self_mem_Icc (hc t ht) v).1
      have hv2 : 0 < a * ‖v‖ ^ 2 := mul_pos ha (by positivity)
      linarith)
    (fun t ht v hv => by
      have h := (inner_blockScale_self_mem_Icc (hc t ht) v).2
      have hv2 : 0 < b * ‖v‖ ^ 2 := mul_pos hb (by positivity)
      have : 4 * b / a * (a / 2) * ‖v‖ ^ 2 = 2 * (b * ‖v‖ ^ 2) := by
        field_simp [ha.ne']
        ring
      linarith)
    (by
      have heq : (fun T : ℝ => ∫ _s in (0 : ℝ)..T, a / 2) = fun T => T * (a / 2) := by
        ext T
        simp [intervalIntegral.integral_const]
        ring
      rw [heq]
      exact tendsto_id.atTop_mul_const (half_pos ha))
    y K hK (fun x hx i h => by
      have h1 : ‖x.ofLp i‖ = 1 := hx i
      rw [h, norm_zero] at h1
      exact zero_ne_one h1)
    (fun t ht => hy1 t ht) (fun t ht => by rw [hgrad, map_neg, neg_neg]; exact hy t ht)
  refine ⟨ystar, hKc.mem_of_tendsto hlim (eventually_atTop.2 ⟨0, hy1⟩), ?_, hlim⟩
  rw [hgrad, neg_eq_zero] at hcrit
  exact hcrit

/-- A stationary unit tuple and the scaling by one satisfy all forward
hypotheses. At `β = 0`, the extended energy is constant. -/
example : ∃ ystar : TupleSpace 2 1, (∀ i, ‖ystar.ofLp i‖ = 1) ∧
    gradient (tupleEnergy (by decide : 1 ≤ 2) 0) ystar = 0 ∧
      Tendsto (fun _ : ℝ => (WithLp.toLp 2 fun _ : Idx 1 => (basePoint 1 : EucSpace 2)))
        atTop (𝓝 ystar) := by
  have hg : tupleEnergy (n := 1) (by decide : 1 ≤ 2) 0 = fun _ => 0 := by
    funext x
    simp [tupleEnergy, selfEnergy, particleEnergy]
  apply exists_tendsto_of_hasDerivAt_blockScale_forward (d := 2) (n := 1) (β := 0)
    (by decide) (a := 1) (b := 1)
    one_pos le_rfl (c := fun _ _ => 1) (fun _ _ _ => ⟨le_rfl, le_rfl⟩)
    (fun _ _ _ => by simp [basePoint, PiLp.norm_single])
  intro t ht
  rw [hg, gradient_fun_const, map_zero]
  exact hasDerivAt_const t _

/-- **Łojasiewicz on the product of unit spheres:** the original all-time
statement follows from the forward theorem. Source: arXiv:2312.10794v5,
§5, gradient-flow outline, and arXiv:2510.22026v2, Appendix D, `lem:loj`. -/
theorem exists_tendsto_of_hasDerivAt_blockScale (hd : 1 ≤ d) {β a b : ℝ} (ha : 0 < a)
    (hab : a ≤ b) {c : ℝ → Idx n → ℝ} (hc : ∀ t i, a ≤ c t i ∧ c t i ≤ b)
    {y : ℝ → TupleSpace d n} (hy1 : ∀ t i, ‖(y t).ofLp i‖ = 1)
    (hy : ∀ t, HasDerivAt y (blockScale (c t) (gradient (tupleEnergy hd β) (y t))) t) :
    ∃ ystar : TupleSpace d n, (∀ i, ‖ystar.ofLp i‖ = 1) ∧
      gradient (tupleEnergy hd β) ystar = 0 ∧ Tendsto y atTop (𝓝 ystar) :=
  exists_tendsto_of_hasDerivAt_blockScale_forward hd ha hab (fun t _ => hc t)
    (fun t _ => hy1 t) (fun t _ => hy t)

/-- The hypotheses of `exists_tendsto_of_hasDerivAt_blockScale` hold for one token at
rest and the scaling by `1`: with a single unit vector `∇g = 0`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ ∀ (_ : ℝ) (_ : Idx 1), (1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 :=
  ⟨one_pos, le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩⟩

end Perspective
end Transformer
