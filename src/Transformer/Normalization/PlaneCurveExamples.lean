/-
# A singular example of prepared plane curve selection

The cusp with a positive distinguished variable satisfies jointly the
analytic preparation, accumulation, orientation, and curve-selection
hypotheses. Its multiple root is not replaced by a simplicity assumption.
-/

import Transformer.Normalization.PlaneCurveSelection

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- The positive cusp `y³ = z₀²` has a prepared plane analytic curve
through its accumulation point. The concrete accumulating arc `(s³,s²)`
also exercises the orientation lemma's hypotheses directly. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let F : Ambient 1 → ℝ := fun x => x.2 ^ 3 - (x.1 0) ^ 2
    let C : Fin 1 → Ambient 1 → ℝ := fun _ x => x.2
    let requirement : Fin 1 → AnalyticSignRequirement := fun _ => .positive
    (∃ sign : ℝ, (sign = 1 ∨ sign = -1) ∧
      ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
        F ((fun _ => sign * x.1), x.2) = 0 ∧
          ∀ i, (requirement i).Holds (C i ((fun _ => sign * x.1), x.2))) ∧
    ∃ (curve : ℝ → Ambient 1) (epsilon : ℝ), 0 < epsilon ∧
      AnalyticAt ℝ curve 0 ∧ curve 0 = 0 ∧
        ∀ t ∈ Ioo 0 epsilon, curve t ∈ preparedPlaneSignSet F C requirement := by
  intro F C requirement
  let L : Ambient 1 →L[ℝ] ℝ :=
    (ContinuousLinearMap.proj (0 : Fin 1)).comp (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
  have hF : AnalyticAt ℝ F 0 := (analyticAt_snd.fun_pow 3).fun_sub ((L.analyticAt 0).fun_pow 2)
  have horder : ExactOrderInLastVariable F 3 := by
    have hslice : lastSlice F = fun t : ℝ => t ^ 3 := by funext t; simp [lastSlice, F]
    rw [ExactOrderInLastVariable, hslice]
    have hord : analyticOrderAt (fun t : ℝ => t ^ 3) 0 = 3 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 3
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 3)).mp hord
  have hacc : (0 : Ambient 1) ∈ closure (preparedPlaneSignSet F C requirement) := by
    have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    let path : ℝ → Ambient 1 := fun t => ((fun _ => t ^ 3), t ^ 2)
    have haPath : AnalyticAt ℝ path 0 :=
      (AnalyticAt.pi (fun _ => analyticAt_id.fun_pow 3)).prod (analyticAt_id.fun_pow 2)
    have hpath0 : path 0 = 0 := by
      apply Prod.ext
      · funext i
        simp [path]
      · simp [path]
    have ht : Tendsto path (nhdsWithin 0 (Ioi 0)) (nhds (0 : Ambient 1)) := by
      simpa only [hpath0] using haPath.continuousAt.tendsto.mono_left nhdsWithin_le_nhds
    have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply mem_closure_iff_frequently.mpr
    apply ht.frequently
    apply hpositive.frequently.mono
    intro t htp
    refine ⟨?_, ?_, fun i => sq_pos_of_pos htp⟩
    · change (t ^ 2) ^ 3 - (t ^ 3) ^ 2 = 0
      ring
    · intro heq
      have hz : t ^ 2 = 0 := congrArg Prod.snd heq
      exact (sq_pos_of_pos htp).ne' hz
  exact ⟨prepared_plane_positive_accumulation F hF horder C requirement hacc,
    prepared_plane_curve_selection F hF horder C (fun _ => analyticAt_snd) requirement hacc⟩

end Transformer.Normalization
