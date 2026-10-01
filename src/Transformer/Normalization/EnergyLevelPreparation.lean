/-
# Finite projection of the local analytic energy graph

The level value is included among the Weierstrass parameters. Preparation
therefore gives finite distinguished-coordinate fibers at every nearby
energy, including the singular central level. This is a general step toward
lifting comparison curves, independent of any Hessian condition.
-/

import Transformer.Normalization.WeierstrassCoordinates

open Filter

noncomputable section

namespace Transformer.Normalization

open AnalyticPreparation

/-- The moving energy level is the last base parameter. All other base
parameters and the distinguished variable enter the analytic function.
Auxiliary construction for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def energyLevelFamily {n : ℕ} (f : Ambient n → ℝ) : Ambient (n + 1) → ℝ :=
  fun x => f ((fun i => x.1 i.castSucc), x.2) - x.1 (Fin.last n)

/-- A function regular in the distinguished variable has finite local
fibers on all nearby energy levels, after including the energy as a base
parameter in real analytic preparation. Auxiliary for the general
curve-selection step in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_energy_level_fibers_finite {n d : ℕ}
    (f : Ambient n → ℝ) (hf : AnalyticAt ℝ f 0)
    (horder : ExactOrderInLastVariable f d) :
    ∃ r > 0, ∀ p : Base (n + 1),
      Set.Finite {w : ℝ | (p, w) ∈ Metric.ball (0 : Ambient (n + 1)) r ∧
        f ((fun i => p i.castSucc), w) = p (Fin.last n)} := by
  let base : Base (n + 1) →L[ℝ] Base n :=
    ContinuousLinearMap.pi (fun i => ContinuousLinearMap.proj i.castSucc)
  let project : Ambient (n + 1) →L[ℝ] Ambient n :=
    (base.comp (ContinuousLinearMap.fst ℝ (Base (n + 1)) ℝ)).prod
      (ContinuousLinearMap.snd ℝ (Base (n + 1)) ℝ)
  let level : Ambient (n + 1) →L[ℝ] ℝ :=
    (ContinuousLinearMap.proj (Fin.last n)).comp
      (ContinuousLinearMap.fst ℝ (Base (n + 1)) ℝ)
  have hfp : AnalyticAt ℝ f (project 0) := by simpa only [map_zero] using hf
  have hfamily : AnalyticAt ℝ (energyLevelFamily f) 0 :=
    (hfp.comp (project.analyticAt 0)).sub (level.analyticAt 0)
  have hslice : lastSlice (energyLevelFamily f) = lastSlice f := by
    funext t
    simp [lastSlice, energyLevelFamily]
    rfl
  have hfamilyOrder : ExactOrderInLastVariable (energyLevelFamily f) d := by
    simpa only [ExactOrderInLastVariable, hslice] using horder
  obtain ⟨a, u, hprep⟩ := exists_isWeierstrassPreparation hfamily hfamilyOrder
  obtain ⟨r, hr, hfinite⟩ := prepared_zero_fibers_finite (energyLevelFamily f) a u hprep
  exact ⟨r, hr, fun p => by simpa only [energyLevelFamily, sub_eq_zero] using hfinite p⟩

/-- Every nonconstant real analytic energy germ in a positive dimension
has invertible coordinates in which projection of its local energy graph
has finite fibers. The energy value is a parameter, so the conclusion
applies to nearby noncritical levels as well as the central critical level.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_finite_fiber_energy_coordinates {n : ℕ}
    (E : EucSpace (n + 1) → ℝ) (z : EucSpace (n + 1))
    (hE : AnalyticAt ℝ E z) (hnonconstant : ¬ ∀ᶠ y in nhds z, E y = E z) :
    ∃ (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (r : ℝ), 0 < r ∧
      ∀ p : Base (n + 1),
        Set.Finite {w : ℝ | (p, w) ∈ Metric.ball (0 : Ambient (n + 1)) r ∧
          E (z + L ((fun i => p i.castSucc), w)) - E z = p (Fin.last n)} := by
  obtain ⟨L, d, _, hf, horder⟩ := exists_regular_energy_coordinates E z hE hnonconstant
  obtain ⟨r, hr, hfinite⟩ := analytic_energy_level_fibers_finite _ hf horder
  exact ⟨L, r, hr, hfinite⟩

/-- A quadratic distinguished variable has exact order two, and the
energy parameter yields actual finite fibers on nearby levels. Auxiliary
example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ r > 0, ∀ p : Base 1,
    Set.Finite {w : ℝ | (p, w) ∈ Metric.ball (0 : Ambient 1) r ∧ w ^ 2 = p 0} := by
  let f : Ambient 0 → ℝ := fun x => x.2 ^ 2
  have hf : AnalyticAt ℝ f 0 := analyticAt_snd.fun_pow 2
  have horder : ExactOrderInLastVariable f 2 := by
    have hord : analyticOrderAt (fun t : ℝ => t ^ 2) 0 = 2 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := 0)) 2
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero
      (analyticAt_id.fun_pow 2)).mp hord
  exact analytic_energy_level_fibers_finite f hf horder

/-- A degenerate two-dimensional quadratic energy satisfies the joint
hypotheses of finite energy-graph projection. Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : ∃ (L : Ambient 1 ≃L[ℝ] EucSpace 2) (r : ℝ), 0 < r ∧
    ∀ p : Base 2,
      Set.Finite {w : ℝ | (p, w) ∈ Metric.ball (0 : Ambient 2) r ∧
        ((L ((fun i => p i.castSucc), w)) 0) ^ 2 = p 1} := by
  let E : EucSpace 2 → ℝ := fun x => (x 0) ^ 2
  have hE : AnalyticAt ℝ E 0 :=
    ((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2
  have hnonconstant : ¬ ∀ᶠ y in nhds (0 : EucSpace 2), E y = E 0 := by
    intro hzero
    let gamma : ℝ → EucSpace 2 := fun t =>
      t • (PiLp.single 2 (0 : Fin 2) 1 : EucSpace 2)
    have ht : Tendsto gamma (nhds 0) (nhds 0) := by
      have hc : ContinuousAt gamma 0 := continuousAt_id.smul continuousAt_const
      simpa [gamma] using hc.tendsto
    have htzero : ∀ᶠ t in nhds (0 : ℝ), t ^ 2 = 0 := by
      simpa [E, gamma] using ht.eventually hzero
    obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp htzero
    have hz : (r / 2) ^ 2 = 0 := hball (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hr)] using half_lt_self hr)
    exact (sq_pos_of_pos (half_pos hr)).ne' hz
  simpa [E] using exists_finite_fiber_energy_coordinates E 0 hE hnonconstant

end Transformer.Normalization
