/-
# The emergence of clusters in self-attention dynamics — the Lagrangian
  field of the continuity equation

The existence half of `c:wellposedtransformers` is proved here by another
route than the source's.  The source freezes the field on time steps `T/2^k`
and passes to the limit by Arzelà–Ascoli; here the solution is sought as
`μ_t = (Y_t)_# μ₀`, where `Y_t` is a bounded continuous map of `ℝ^d` moving
each point along the field that the measure it transports generates:

  `Ẏ_t = F(Y_t)`,  `F(Y)(x) = 𝒳[Y_# μ₀](Y x)`  (`lagrangianField`).

This is an autonomous ODE in the Banach space `ℝ^d →ᵇ ℝ^d` of bounded
continuous maps.  `F` has linear growth, `‖F(Y)‖ ≤ ‖V‖ ‖Y‖`, since `Y_# μ₀`
is carried by `B̄(0, ‖Y‖)` (`e:bddinx`); and `F` is Lipschitz on every ball
(`lipschitzOnWith_lagrangianField`), since `𝒳` is jointly Lipschitz in the
point and in the measure (`exists_norm_attentionKernel_sub_le`) and
`W₂(Y_# μ₀, Z_# μ₀) ≤ ‖Y - Z‖` through the coupling `(Y, Z)_# μ₀`
(`W2_map_le_norm_sub`).  So the ODE has a solution on all of `ℝ`
(`Landscape.GlobalFlow.exists_global_of_lipschitzOnWith`), from which
`Section6_ContEq` builds the solution of `e:conteq`.

Source: arXiv:2305.05465v6, `c:wellposedtransformers`; the Lagrangian form
`μ(t) = (Φ^t_{𝒳[μ(t)]})_# μ₀` is the one its proof arrives at.
-/

import Transformer.Clusters.Section6_KernelField
import Transformer.ContinuityEq.Stability

open Set Metric Filter Topology MeasureTheory
open scoped NNReal BoundedContinuousFunction

namespace Transformer
namespace Clusters

open Wasserstein

variable {d : ℕ}

/-- The image of a probability measure under a bounded continuous map is a
probability measure. -/
instance isProbabilityMeasure_map_boundedContinuous (μ₀ : Measure (EucSpace d))
    [IsProbabilityMeasure μ₀] (Y : EucSpace d →ᵇ EucSpace d) : IsProbabilityMeasure (μ₀.map Y) :=
  (Measure.isProbabilityMeasure_map_iff Y.continuous.aemeasurable).mpr inferInstance

/-- The image of a measure under a map bounded by `r` is carried by `B̄(0, r)`. -/
theorem isCarriedBy_map (μ₀ : Measure (EucSpace d)) (Y : EucSpace d →ᵇ EucSpace d) {r : ℝ}
    (hr : ‖Y‖ ≤ r) : IsCarriedBy (μ₀.map Y) r := by
  have h : ⇑Y ⁻¹' (closedBall (0 : EucSpace d) r)ᶜ = ∅ := eq_empty_of_forall_notMem fun x hx =>
    hx (mem_closedBall_zero_iff.2 ((Y.norm_coe_le_norm x).trans hr))
  rw [IsCarriedBy, Measure.map_apply Y.continuous.measurable measurableSet_closedBall.compl, h,
    measure_empty]

/-- **Pushing one measure by two maps**: `W₂(Y_# μ₀, Z_# μ₀) ≤ ‖Y - Z‖`,
through the coupling `(Y, Z)_# μ₀`. -/
theorem W2_map_le_norm_sub (μ₀ : Measure (EucSpace d)) [IsProbabilityMeasure μ₀]
    (Y Z : EucSpace d →ᵇ EucSpace d) : W2 (μ₀.map Y) (μ₀.map Z) ≤ ‖Y - Z‖ := by
  have hdiag : Measurable fun x : EucSpace d => (x, x) := by fun_prop
  have hcoup : IsCoupling μ₀ μ₀ (μ₀.map fun x => (x, x)) := by
    constructor <;> rw [Measure.map_map (by fun_prop) hdiag] <;> simp [Function.comp_def]
  refine (ContinuityEq.W2_map_le_sqrt hcoup Y.continuous Z.continuous).trans ?_
  rw [integral_map (f := fun p : EucSpace d × EucSpace d => ‖Y p.1 - Z p.2‖ ^ 2)
    hdiag.aemeasurable (((Y.continuous.comp continuous_fst).sub
      (Z.continuous.comp continuous_snd)).norm.pow 2).aestronglyMeasurable]
  calc Real.sqrt (∫ x, ‖Y x - Z x‖ ^ 2 ∂μ₀) ≤ Real.sqrt (∫ _x, ‖Y - Z‖ ^ 2 ∂μ₀) :=
        Real.sqrt_le_sqrt (integral_mono_of_nonneg (ae_of_all _ fun _ => by positivity)
          (integrable_const _) (ae_of_all _ fun x => by
            have := (Y - Z).norm_coe_le_norm x
            rw [BoundedContinuousFunction.coe_sub, Pi.sub_apply] at this
            show ‖Y x - Z x‖ ^ 2 ≤ ‖Y - Z‖ ^ 2
            gcongr))
    _ = ‖Y - Z‖ := by simp

variable (Q K V : ParamMatrix d) (μ₀ : Measure (EucSpace d)) [IsProbabilityMeasure μ₀]

/-- **The Lagrangian field of `e:conteq`** on bounded continuous maps:
`F(Y)(x) = 𝒳[Y_# μ₀](Y x)`, bounded by `‖V‖ ‖Y‖` (`e:bddinx`). -/
noncomputable def lagrangianField (Y : EucSpace d →ᵇ EucSpace d) : EucSpace d →ᵇ EucSpace d :=
  BoundedContinuousFunction.ofNormedAddCommGroup
    (fun x => attentionKernel Q K V (μ₀.map Y) (Y x))
    ((lipschitzWith_attentionKernel Q K V (R := ‖Y‖ + 1) (by positivity) (μ₀.map Y)
      (isCarriedBy_map μ₀ Y (le_add_of_nonneg_right zero_le_one))).continuous.comp Y.continuous)
    (‖V‖ * ‖Y‖)
    fun x => attentionKernel_norm_le Q K V ‖Y‖ (norm_nonneg _) _ inferInstance
      (isCarriedBy_map μ₀ Y le_rfl) _

theorem lagrangianField_apply (Y : EucSpace d →ᵇ EucSpace d) (x : EucSpace d) :
    lagrangianField Q K V μ₀ Y x = attentionKernel Q K V (μ₀.map Y) (Y x) :=
  rfl

/-- **Linear growth**: `‖F(Y)‖ ≤ ‖V‖ ‖Y‖`. -/
theorem norm_lagrangianField_le (Y : EucSpace d →ᵇ EucSpace d) :
    ‖lagrangianField Q K V μ₀ Y‖ ≤ ‖V‖ * ‖Y‖ :=
  BoundedContinuousFunction.norm_ofNormedAddCommGroup_le _ (by positivity) _

/-- **The Lagrangian field is Lipschitz on every ball** of `ℝ^d →ᵇ ℝ^d`: on
`B̄(0, ρ)`, with `ρ' = |ρ| + 1`,
`‖F(Y) - F(Z)‖ ≤ (2 ‖QᵀK‖ ‖V‖ ρ'² + C) ‖Y - Z‖`, `C` from `e:lipinmu`. -/
theorem lipschitzOnWith_lagrangianField (ρ : ℝ) :
    ∃ L : ℝ≥0, LipschitzOnWith L (lagrangianField Q K V μ₀) (closedBall 0 ρ) := by
  obtain ⟨C, hC, hCμ⟩ := exists_norm_attentionKernel_sub_le Q K V (R := |ρ| + 1) (by positivity)
  set L := 2 * ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖V‖ * (|ρ| + 1) ^ 2
  refine ⟨(L + C).toNNReal, LipschitzOnWith.of_dist_le_mul fun Y hY Z hZ => ?_⟩
  have hρ : ∀ Y : EucSpace d →ᵇ EucSpace d, Y ∈ closedBall 0 ρ → ‖Y‖ ≤ |ρ| + 1 := fun Y hY => by
    linarith [mem_closedBall_zero_iff.1 hY, le_abs_self ρ]
  rw [Real.coe_toNNReal _ (by positivity), dist_eq_norm, dist_eq_norm]
  refine (BoundedContinuousFunction.norm_le (by positivity)).2 fun x => ?_
  have hYZ := (Y - Z).norm_coe_le_norm x
  rw [BoundedContinuousFunction.coe_sub, Pi.sub_apply] at hYZ ⊢
  calc ‖lagrangianField Q K V μ₀ Y x - lagrangianField Q K V μ₀ Z x‖
      ≤ L * ‖Y x - Z x‖ + C * W2 (μ₀.map Y) (μ₀.map Z) :=
        hCμ _ _ inferInstance inferInstance (isCarriedBy_map μ₀ Y (hρ Y hY))
          (isCarriedBy_map μ₀ Z (hρ Z hZ)) (Y x) (Z x)
          (mem_closedBall_zero_iff.2 ((Z.norm_coe_le_norm x).trans (hρ Z hZ)))
    _ ≤ L * ‖Y - Z‖ + C * ‖Y - Z‖ :=
        add_le_add (mul_le_mul_of_nonneg_left hYZ (by positivity))
          (mul_le_mul_of_nonneg_left (W2_map_le_norm_sub μ₀ Y Z) hC.le)
    _ = (L + C) * ‖Y - Z‖ := by ring

/-- The hypotheses of the results above are satisfiable: the Dirac mass at the
origin is a probability measure, and the zero map is bounded by `0`. -/
example : IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
    ‖(0 : EucSpace d →ᵇ EucSpace d)‖ ≤ 0 :=
  ⟨inferInstance, by simp⟩

end Clusters
end Transformer
