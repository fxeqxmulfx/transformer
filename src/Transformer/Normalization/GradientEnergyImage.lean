/-
# The analytic energy-gradient image and its level minima

For an analytic energy the map `x ↦ (E(x), ‖∇E(x)‖²)` is analytic.
Its image of a compact source set is compact, and each nonempty energy
fiber has a point of minimum gradient norm. These are the elementary
inputs to the curve-selection route for Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.ModulatedCritical
import Mathlib.Analysis.Analytic.Linear
import Mathlib.Topology.Order.Compact

open Set
open scoped BigOperators

namespace Transformer.Normalization

/-- The real gradient of an analytic energy is analytic, by the analytic
Fréchet derivative and the Riesz isometry. Auxiliary regularity input for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_gradient_analyticAt {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (hE : AnalyticAt ℝ E z) :
    AnalyticAt ℝ (gradient E) z := by
  have hR := (InnerProductSpace.toDual ℝ (EucSpace N)).symm.analyticAt (fderiv ℝ E z)
  change AnalyticAt ℝ (fun x =>
    (InnerProductSpace.toDual ℝ (EucSpace N)).symm (fderiv ℝ E x)) z
  exact hR.comp hE.fderiv

/-- The analytic map whose lower edge records gradient minima on energy
levels. Squaring the norm preserves analyticity at zero. Auxiliary
definition for Appendix D.1 of arXiv:2510.22026v2. -/
noncomputable def energyGradientMap {N : ℕ}
    (E : EucSpace N → ℝ) (x : EucSpace N) : ℝ × ℝ :=
  (E x, ‖gradient E x‖ ^ 2)

/-- The energy and squared gradient norm form an analytic map at every
analytic base point. Auxiliary image regularity for the analytic step of
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem energy_gradient_map_analyticAt {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (hE : AnalyticAt ℝ E z) :
    AnalyticAt ℝ (energyGradientMap E) z := by
  have hG := analytic_gradient_analyticAt E z hE
  have hsq : AnalyticAt ℝ (fun y => ∑ i : Fin N, ((gradient E y).ofLp i) ^ 2) z := by
    apply Finset.univ.analyticAt_fun_sum
    intro i hi
    exact (((EuclideanSpace.proj i : EucSpace N →L[ℝ] ℝ).analyticAt
      (gradient E z)).comp hG).fun_pow 2
  have hnorm : AnalyticAt ℝ (fun y => ‖gradient E y‖ ^ 2) z := by
    simpa only [EuclideanSpace.real_norm_sq_eq] using hsq
  exact hE.prod hnorm

/-- Squared gradient norm, whose minima on energy fibers describe the
lower edge of `energyGradientMap`. Auxiliary definition for Appendix D.1
of arXiv:2510.22026v2. -/
noncomputable def squaredGradientNorm {N : ℕ}
    (E : EucSpace N → ℝ) (x : EucSpace N) : ℝ := ‖gradient E x‖ ^ 2

/-- The squared gradient norm is analytic, including at critical points.
Auxiliary regularity fact for the energy-level curve-selection route to
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem squared_gradient_norm_analyticAt {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (hE : AnalyticAt ℝ E z) :
    AnalyticAt ℝ (squaredGradientNorm E) z := by
  change AnalyticAt ℝ (fun y => (energyGradientMap E y).2) z
  exact ((ContinuousLinearMap.snd ℝ ℝ ℝ).analyticAt (energyGradientMap E z)).comp
    (energy_gradient_map_analyticAt E z hE)

/-- The energy-gradient image of a compact set on which the energy is
analytic is compact. Auxiliary image fact for Appendix D.1 of
arXiv:2510.22026v2. -/
theorem energy_gradient_map_compact_image {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (hK : IsCompact K) (hE : AnalyticOnNhd ℝ E K) :
    IsCompact (energyGradientMap E '' K) :=
  hK.image_of_continuousOn (fun x hx =>
    (energy_gradient_map_analyticAt E x (hE x hx)).continuousAt.continuousWithinAt)

/-- Every nonempty energy level in a compact analytic domain has a point
with minimum gradient norm on that level. This establishes the compact
minimization step of the analytic-curve route to Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2, without assuming that the minimizers have an analytic
selection. -/
theorem compact_gradient_minimum_on_energy_fiber {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (hK : IsCompact K) (hE : AnalyticOnNhd ℝ E K)
    (c : ℝ) (hc : c ∈ E '' K) :
    ∃ w ∈ K, E w = c ∧ ∀ y ∈ K, E y = c → ‖gradient E w‖ ≤ ‖gradient E y‖ := by
  let T : Set (EucSpace N) := {y | y ∈ K ∧ E y = c}
  have hT : IsCompact T := hK.of_isClosed_subset
    (hK.isClosed.isClosed_eq hE.continuousOn continuousOn_const) (fun _ hy => hy.1)
  have hT0 : T.Nonempty := by
    obtain ⟨w, hw, hwc⟩ := hc
    exact ⟨w, hw, hwc⟩
  have hG : ContinuousOn (gradient E) T := fun y hy =>
    (analytic_gradient_analyticAt E y (hE y hy.1)).continuousAt.continuousWithinAt
  obtain ⟨w, hw, hmin⟩ := hT.exists_isMinOn hT0 hG.norm
  exact ⟨w, hw.1, hw.2, fun y hy hyc => hmin ⟨hy, hyc⟩⟩

/-- The analytic energy `x²+y⁴` on a closed radius-two ball has a
nonempty level `1/16`; hence all regularity and compact-minimization
hypotheses are jointly satisfiable. Auxiliary example for Appendix D.1
of arXiv:2510.22026v2. -/
example : ∃ w ∈ Metric.closedBall (0 : EucSpace 2) 2,
    (w 0) ^ 2 + (w 1) ^ 4 = (1 / 16 : ℝ) ∧
      ∀ y ∈ Metric.closedBall (0 : EucSpace 2) 2,
        (y 0) ^ 2 + (y 1) ^ 4 = (1 / 16 : ℝ) →
          ‖gradient (fun t : EucSpace 2 => (t 0) ^ 2 + (t 1) ^ 4) w‖ ≤
            ‖gradient (fun t : EucSpace 2 => (t 0) ^ 2 + (t 1) ^ 4) y‖ := by
  apply compact_gradient_minimum_on_energy_fiber
    (fun y : EucSpace 2 => (y 0) ^ 2 + (y 1) ^ 4)
    (Metric.closedBall 0 2) (isCompact_closedBall _ _)
  · intro y hy
    exact (((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt y).fun_pow 2).add
      (((EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt y).fun_pow 4)
  · refine ⟨PiLp.single 2 (0 : Fin 2) (1 / 4), ?_, ?_⟩
    · simp [Metric.mem_closedBall]
      norm_num
    · norm_num

end Transformer.Normalization
