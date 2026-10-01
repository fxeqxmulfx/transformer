/-
# Lifting projected curves to global energy-level gradient minima

A prescribed analytic base curve with approaching global-minimum
witnesses lifts to an analytic global-minimum curve. Finite root minima
remove the last fiber quantifier, including comparisons with points
outside the coordinate neighborhood.
-/

import Transformer.Normalization.EnergyGradientMinimumLifting
import Transformer.Normalization.ApproachingFiberWitnesses

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- A projected analytic parameter curve with approaching global
gradient-minimum witnesses has an analytic lift that is globally
minimizing on the same energy levels. The source set is described locally
by finite analytic sign constraints; the comparison still covers every
point of the original set. Only pointwise witnesses above the given base
curve are assumed, with no analytic selection of their last coordinates.
Selecting such a base curve remains the geometric input for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_global_gradient_minimum_lifting {n N d l : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (phi : Ambient n → EucSpace N) (hphi : AnalyticAt ℝ phi 0) (hphi0 : phi 0 = z)
    (level : Base n → ℝ) (hlevel : AnalyticAt ℝ level 0)
    (horder : ExactOrderInLastVariable (fun x => E (phi x) - level x.1) d)
    (gamma : ℝ → Base n) (hgamma : AnalyticAt ℝ gamma 0) (hgamma0 : gamma 0 = 0)
    (K : Set (EucSpace N)) (C : Fin l → Ambient n → ℝ)
    (hC : ∀ i, AnalyticAt ℝ (C i) 0) (requirement : Fin l → AnalyticSignRequirement)
    (hK : ∀ᶠ x in nhds (0 : Ambient n),
      phi x ∈ K ↔ ∀ i, (requirement i).Holds (C i x))
    (hprojected : ∀ delta : ℝ, 0 < delta → ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, |y| < delta ∧ E (phi (gamma t, y)) = level (gamma t) ∧
        phi (gamma t, y) ∈ K ∧ ∀ w ∈ K, E w = level (gamma t) →
          ‖gradient E (phi (gamma t, y))‖ ≤ ‖gradient E w‖) :
    ∃ (q : ℕ) (curve : ℝ → EucSpace N), 0 < q ∧ AnalyticAt ℝ curve 0 ∧ curve 0 = z ∧
      (∀ᶠ s in nhds (0 : ℝ), E (curve s) = level (gamma (s ^ q))) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), curve s ∈ K ∧
        ∀ w ∈ K, E w = level (gamma (s ^ q)) → ‖gradient E (curve s)‖ ≤ ‖gradient E w‖ := by
  let F : Ambient n → ℝ := fun x => E (phi x) - level x.1
  have hEphi : AnalyticAt ℝ E (phi 0) := by simpa only [hphi0] using hE
  have hF : AnalyticAt ℝ F 0 :=
    (hEphi.comp (f := phi) (x := 0) hphi).fun_sub
      (hlevel.comp (f := fun x : Ambient n => x.1) (x := 0) analyticAt_fst)
  let V : Ambient n → ℝ := fun x => squaredGradientNorm E (phi x)
  have hG : AnalyticAt ℝ (squaredGradientNorm E) (phi 0) := by
    simpa only [hphi0] using squared_gradient_norm_analyticAt E z hE
  have hV : AnalyticAt ℝ V 0 := hG.comp (f := phi) (x := 0) hphi
  let slice : ℝ × ℝ → Ambient n := fun x => (gamma x.1, x.2)
  have hslice : AnalyticAt ℝ slice 0 :=
    (hgamma.comp (f := fun x : ℝ × ℝ => x.1) (x := 0) analyticAt_fst).prod analyticAt_snd
  have hslice0 : slice 0 = 0 := by simp [slice, hgamma0, Prod.mk_zero_zero]
  have htSlice : Tendsto slice (nhds 0) (nhds 0) := by
    simpa only [hslice0] using hslice.continuousAt.tendsto
  have hKslice : ∀ᶠ x in nhds (0 : ℝ × ℝ),
      phi (gamma x.1, x.2) ∈ K ↔ ∀ i, (requirement i).Holds (C i (gamma x.1, x.2)) :=
    htSlice.eventually hK
  obtain ⟨eta, heta, hKball⟩ := Metric.eventually_nhds_iff.mp hKslice
  have hKbox (t y : ℝ) (ht : |t| < eta) (hy : |y| < eta) :
      phi (gamma t, y) ∈ K ↔ ∀ i, (requirement i).Holds (C i (gamma t, y)) :=
    hKball (y := (t, y)) (by
      simpa only [dist_zero_right, Prod.norm_def, Real.norm_eq_abs] using max_lt ht hy)
  have haccK : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
      E (phi (gamma x.1, x.2)) = level (gamma x.1) ∧ phi (gamma x.1, x.2) ∈ K := by
    apply frequent_joint_of_approaching_fiber_witnesses
    intro delta hd
    exact (hprojected delta hd).mono (fun t ht => by
      obtain ⟨y, hy, hEq, hyK, _⟩ := ht
      exact ⟨y, hy, hEq, hyK⟩)
  have hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ F (gamma x.1, x.2) = 0 ∧
      ∀ i, (requirement i).Holds (C i (gamma x.1, x.2)) :=
    (haccK.and_eventually hKslice).mono (fun x hx =>
      ⟨hx.1.1, sub_eq_zero.mpr hx.1.2.1, hx.2.mp hx.1.2.2⟩)
  obtain ⟨q, g, r, hq, hr, hg, hg0, heq, hminimum⟩ :=
    analytic_fiber_minimum_lifting F hF horder gamma hgamma hgamma0 V hV C hC requirement hacc
  have hpower : AnalyticAt ℝ (fun s : ℝ => s ^ q) 0 := analyticAt_id.fun_pow q
  have hgammaPower : AnalyticAt ℝ (fun s : ℝ => gamma (s ^ q)) 0 :=
    (by simpa [hq.ne'] using hgamma : AnalyticAt ℝ gamma ((0 : ℝ) ^ q)).comp
      (f := fun s : ℝ => s ^ q) (x := 0) hpower
  let path : ℝ → Ambient n := fun s => (gamma (s ^ q), g s)
  have hpath : AnalyticAt ℝ path 0 := hgammaPower.prod hg
  have hpath0 : path 0 = 0 := by simp [path, hq.ne', hgamma0, hg0, Prod.mk_zero_zero]
  let curve : ℝ → EucSpace N := fun s => phi (path s)
  have hcurve : AnalyticAt ℝ curve 0 :=
    (by simpa only [hpath0] using hphi : AnalyticAt ℝ phi (path 0)).comp
      (f := path) (x := 0) hpath
  have htPath : Tendsto path (nhds 0) (nhds 0) := by
    simpa only [hpath0] using hpath.continuousAt.tendsto
  have htPower : Tendsto (fun s : ℝ => s ^ q) (nhdsWithin 0 (Ioi 0)) (nhdsWithin 0 (Ioi 0)) := by
    apply tendsto_nhdsWithin_iff.mpr
    refine ⟨?_, ?_⟩
    · simpa [hq.ne'] using hpower.continuousAt.tendsto.mono_left nhdsWithin_le_nhds
    · have hp : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), 0 < s := self_mem_nhdsWithin
      exact hp.mono (fun s hs => pow_pos hs q)
  have hsmallPower : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), |s ^ q| < eta :=
    htPower.eventually ((continuousAt_id.abs.eventually
      (Iio_mem_nhds (by simpa using heta))).filter_mono nhdsWithin_le_nhds)
  have hdelta : 0 < min r eta := lt_min hr heta
  refine ⟨q, curve, hq, hcurve, by simp only [curve, hpath0, hphi0],
    heq.mono (fun s hs => sub_eq_zero.mp hs), ?_⟩
  filter_upwards [hminimum, hsmallPower,
    (htPath.eventually hK).filter_mono nhdsWithin_le_nhds,
    htPower.eventually (hprojected (min r eta) hdelta)] with s hs hsp hks hproj
  obtain ⟨y, hy, hyE, hyK, hglobal⟩ := hproj
  have hyeta : |y| < eta := hy.trans_le (min_le_right _ _)
  have hyr : |y| < r := hy.trans_le (min_le_left _ _)
  have hCy := (hKbox (s ^ q) y hsp hyeta).mp hyK
  have hsq := hs.2.2 y hyr (sub_eq_zero.mpr hyE) hCy
  have hnorm : ‖gradient E (curve s)‖ ≤ ‖gradient E (phi (gamma (s ^ q), y))‖ :=
    (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hsq
  exact ⟨hks.mpr hs.2.1, fun w hw hwE => hnorm.trans (hglobal w hw hwE)⟩

end Transformer.Normalization
