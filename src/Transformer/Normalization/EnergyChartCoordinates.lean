/-
# Energy-graph coordinates and their actual inverse on the source

The energy value is a free base parameter. Inverse coordinates set that
parameter to the actual energy, retaining every nearby source point.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.WeierstrassCoordinates
import Transformer.Normalization.CompactEnergyAtlasCover
import Transformer.Normalization.QuadraticEnergy

noncomputable section
open Filter Topology

namespace Transformer.Normalization

open AnalyticPreparation

/-- Add an energy parameter to a linear spatial chart; the chart image
ignores that extra parameter. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
def energyChartMap {n : ℕ} (w : EucSpace (n + 1))
    (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) : Ambient (n + 1) → EucSpace (n + 1) :=
  fun x => w + L ((fun i => x.1 i.castSucc), x.2)

/-- The actual moving energy level in the added base coordinate.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def energyChartLevel {n : ℕ} (E : EucSpace (n + 1) → ℝ) (w : EucSpace (n + 1)) :
    Base (n + 1) → ℝ := fun z => E w + z (Fin.last n)

/-- The spatial inverse together with the actual shifted energy.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def inverseEnergyChart {n : ℕ} (E : EucSpace (n + 1) → ℝ) (w : EucSpace (n + 1))
    (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (x : EucSpace (n + 1)) : Ambient (n + 1) :=
  (Fin.lastCases (E x - E w) (L.symm (x - w)).1, (L.symm (x - w)).2)

/-- Forgetting the added energy coordinate recovers every actual
source point. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem energyChartMap_inverse {n : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w x : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) :
    energyChartMap w L (inverseEnergyChart E w L x) = x := by
  simp only [energyChartMap, inverseEnergyChart, Fin.lastCases_castSucc]
  change w + L (L.symm (x - w)) = x
  rw [L.apply_symm_apply]
  abel

/-- The inverse energy parameter is exactly the source energy.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem energyChartLevel_inverse {n : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w x : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) :
    energyChartLevel E w (inverseEnergyChart E w L x).1 = E x := by
  simp only [energyChartLevel, inverseEnergyChart, Fin.lastCases_last]
  ring

/-- The center has zero coordinates in the energy graph.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem inverseEnergyChart_center {n : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) :
    inverseEnergyChart E w L w = 0 := by
  apply Prod.ext
  · ext i
    refine Fin.lastCases ?_ (fun j => ?_) i <;> simp [inverseEnergyChart]
  · simp [inverseEnergyChart]

/-- The coordinate map is analytic and has the required center.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem energyChartMap_analytic {n : ℕ} (w : EucSpace (n + 1))
    (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) :
    AnalyticAt ℝ (energyChartMap w L) 0 ∧ energyChartMap w L 0 = w := by
  let base : Base (n + 1) →L[ℝ] Base n :=
    ContinuousLinearMap.pi (fun i => ContinuousLinearMap.proj i.castSucc)
  let project : Ambient (n + 1) →L[ℝ] Ambient n :=
    (base.comp (ContinuousLinearMap.fst ℝ (Base (n + 1)) ℝ)).prod
      (ContinuousLinearMap.snd ℝ (Base (n + 1)) ℝ)
  exact ⟨analyticAt_const.add ((L.analyticAt (project 0)).comp (project.analyticAt 0)),
    by change w + L ((0 : Base n), 0) = w; simp⟩

/-- The moving level is analytic in every base coordinate.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem energyChartLevel_analytic {n : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w : EucSpace (n + 1)) : AnalyticAt ℝ (energyChartLevel E w) 0 :=
  analyticAt_const.add
    ((ContinuousLinearMap.proj (Fin.last n) : Base (n + 1) →L[ℝ] ℝ).analyticAt 0)

/-- The actual source inverse is analytic at every analytic energy
center. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem inverseEnergyChart_analytic {n : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1))
    (hE : AnalyticAt ℝ E w) : AnalyticAt ℝ (inverseEnergyChart E w L) w := by
  have hcoords : AnalyticAt ℝ (fun x : EucSpace (n + 1) => L.symm (x - w)) w :=
    (by simpa using L.symm.analyticAt 0 : AnalyticAt ℝ L.symm (w - w)).comp
      (f := fun x : EucSpace (n + 1) => x - w) (x := w)
      (analyticAt_id.fun_sub analyticAt_const)
  have hbase : AnalyticAt ℝ (fun x : EucSpace (n + 1) => (L.symm (x - w)).1) w :=
    ((ContinuousLinearMap.fst ℝ (Base n) ℝ).analyticAt (L.symm (w - w))).comp
      (f := fun x : EucSpace (n + 1) => L.symm (x - w)) (x := w) hcoords
  have hscalar : AnalyticAt ℝ (fun x : EucSpace (n + 1) => (L.symm (x - w)).2) w :=
    ((ContinuousLinearMap.snd ℝ (Base n) ℝ).analyticAt (L.symm (w - w))).comp
      (f := fun x : EucSpace (n + 1) => L.symm (x - w)) (x := w) hcoords
  apply AnalyticAt.prod _ hscalar
  apply AnalyticAt.pi
  intro i
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simpa only [Fin.lastCases_last] using hE.fun_sub (analyticAt_const (v := E w))
  · simpa only [Fin.lastCases_castSucc, Function.comp_def, ContinuousLinearMap.proj_apply] using
      ((ContinuousLinearMap.proj j : Base n →L[ℝ] ℝ).analyticAt ((L.symm (w - w)).1)).comp
        (f := fun x : EucSpace (n + 1) => (L.symm (x - w)).1) (x := w) hbase

/-- Adding the energy parameter preserves the exact distinguished
order of the original regular energy chart. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem energyChart_exact_order {n d : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1))
    (horder : ExactOrderInLastVariable (fun x => E (w + L x) - E w) d) :
    ExactOrderInLastVariable
      (fun x => E (energyChartMap w L x) - energyChartLevel E w x.1) d := by
  have hslice : lastSlice (fun x => E (energyChartMap w L x) - energyChartLevel E w x.1) =
      lastSlice (fun x : Ambient n => E (w + L x) - E w) := by
    funext t
    simp only [lastSlice, energyChartMap, energyChartLevel, Pi.zero_apply, add_zero]
    rfl
  simpa only [ExactOrderInLastVariable, hslice] using horder

/-- The nonconstant quadratic energy supplies genuine regular spatial
coordinates and an analytic inverse energy graph. This witnesses the
analyticity and exact-order hypotheses jointly. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let E : EucSpace 1 → ℝ := fun x => ‖x‖ ^ 2
    ∃ (L : Ambient 0 ≃L[ℝ] EucSpace 1) (d : ℕ), 0 < d ∧
      ExactOrderInLastVariable
        (fun x => E (energyChartMap 0 L x) - energyChartLevel E 0 x.1) d ∧
      AnalyticAt ℝ (inverseEnergyChart E 0 L) 0 := by
  intro E
  have hE : AnalyticAt ℝ E 0 := quadratic_energy_analytic 0 (Set.mem_univ _)
  have hnonconstant := nonflatEnergyLevel_nonconstant E (Metric.closedBall 0 1) 0
    zero_mem_nonflat_quadratic_level
  obtain ⟨L, d, hd, _, horder⟩ := exists_regular_energy_coordinates E 0 hE hnonconstant
  exact ⟨L, d, hd, energyChart_exact_order E 0 L horder,
    inverseEnergyChart_analytic E 0 L hE⟩

end Transformer.Normalization
