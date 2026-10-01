/-
# Joint witnesses for analytic energy-atlas arithmetic comparisons

The two charts have opposite distinguished directions and the objective
is the actual squared gradient norm of a degenerate energy.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticAtlasGradientQuery

namespace Transformer.Normalization

open AnalyticPreparation

/-- Two opposite quartic energy charts, an analytic moving energy
coordinate, and exponential constraints simultaneously witness every
analyticity and exact-order hypothesis of the atlas comparison and
gradient-minimizer theorems. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : let E : EucSpace 1 → ℝ := fun x => (x 0) ^ 4
    let phi : Fin 2 → Ambient 1 → EucSpace 1 := fun j x =>
      (if j = 0 then x.2 else -x.2) • PiLp.single 2 (0 : Fin 1) 1
    let level : Fin 2 → Base 1 → ℝ := fun _ p => p 0
    let G : Fin 2 → Fin 1 → Ambient 1 → ℝ := fun j i x =>
      Real.exp x.2 + (j : ℝ) + (i : ℝ)
    (∀ j, AnalyticAt ℝ (phi j) 0) ∧
    (∀ j, AnalyticAt ℝ E (phi j 0)) ∧
    (∀ j, AnalyticAt ℝ (squaredGradientNorm E) (phi j 0)) ∧
    (∀ j, AnalyticAt ℝ (level j) 0) ∧
    (∀ j, ExactOrderInLastVariable (fun x => E (phi j x) - level j x.1) 4) ∧
    (∀ j i, AnalyticAt ℝ (G j i) 0) := by
  intro E phi level G
  have hE : AnalyticAt ℝ E 0 :=
    ((EuclideanSpace.proj 0 : EucSpace 1 →L[ℝ] ℝ).analyticAt 0).fun_pow 4
  have hphi0 (j : Fin 2) : phi j 0 = 0 := by
    simp [phi]
  refine ⟨?_, (fun j => by simpa only [hphi0] using hE),
    (fun j => by simpa only [hphi0] using squared_gradient_norm_analyticAt E 0 hE),
    (fun _ => (ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0),
    ?_, fun _ _ => ((analyticAt_rexp.comp analyticAt_snd).add analyticAt_const).add
      analyticAt_const⟩
  · intro j
    fin_cases j
    · exact analyticAt_snd.smul analyticAt_const
    · exact analyticAt_snd.neg.smul analyticAt_const
  · intro j
    have hslice : lastSlice (fun x => E (phi j x) - level j x.1) =
        fun t : ℝ => t ^ 4 := by
      funext t
      fin_cases j
      · simp [lastSlice, E, phi, level]
      · simp [lastSlice, E, phi, level]
        ring
    rw [ExactOrderInLastVariable, hslice]
    apply (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 4)).mp
    simpa [Pi.pow_def] using analyticOrderAt_pow
      (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 4

end Transformer.Normalization
