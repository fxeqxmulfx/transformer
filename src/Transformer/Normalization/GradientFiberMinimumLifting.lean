/-
# Analytic selection of gradient minima on finite energy fibers

The squared gradient norm is an analytic objective. Minimum lifting for
prepared fibers therefore gives gradient-norm minima on all nearby roots
above a prescribed analytic parameter curve.
-/

import Transformer.Normalization.AnalyticFiberMinimumLifting
import Transformer.Normalization.GradientEnergyImage

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- An analytic coordinate family regular in its distinguished variable
admits analytic gradient-norm minima on its scalar energy fibers over any
prescribed analytic base curve. The level can vary analytically with the
base parameters. After ramification the selected point compares with
every nearby root on that same fiber. No gradient-minimizer selection is
assumed. This proves the finite-fiber minimization step in arbitrary
dimension for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2; comparison
across different base points still requires the general projection step. -/
theorem analytic_gradient_fiber_minimum_lifting {n N d : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (phi : Ambient n → EucSpace N) (hphi : AnalyticAt ℝ phi 0) (hphi0 : phi 0 = z)
    (level : Base n → ℝ) (hlevel : AnalyticAt ℝ level 0)
    (horder : ExactOrderInLastVariable (fun x => E (phi x) - level x.1) d)
    (gamma : ℝ → Base n) (hgamma : AnalyticAt ℝ gamma 0) (hgamma0 : gamma 0 = 0)
    (hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
      E (phi (gamma x.1, x.2)) = level (gamma x.1)) :
    ∃ (q : ℕ) (g : ℝ → ℝ) (r : ℝ), 0 < q ∧ 0 < r ∧
      AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), E (phi (gamma (s ^ q), g s)) = level (gamma (s ^ q))) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), |g s| < r ∧
        ∀ y : ℝ, |y| < r → E (phi (gamma (s ^ q), y)) = level (gamma (s ^ q)) →
          ‖gradient E (phi (gamma (s ^ q), g s))‖ ≤ ‖gradient E (phi (gamma (s ^ q), y))‖ := by
  let F : Ambient n → ℝ := fun x => E (phi x) - level x.1
  have hEphi : AnalyticAt ℝ E (phi 0) := by simpa only [hphi0] using hE
  have hF : AnalyticAt ℝ F 0 :=
    (hEphi.comp (f := phi) (x := 0) hphi).fun_sub
      (hlevel.comp (f := fun x : Ambient n => x.1) (x := 0) analyticAt_fst)
  let V : Ambient n → ℝ := fun x => squaredGradientNorm E (phi x)
  have hG : AnalyticAt ℝ (squaredGradientNorm E) (phi 0) := by
    simpa only [hphi0] using squared_gradient_norm_analyticAt E z hE
  have hV : AnalyticAt ℝ V 0 := hG.comp (f := phi) (x := 0) hphi
  have haccF : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ F (gamma x.1, x.2) = 0 ∧
      ∀ i : Fin 0, (Fin.elim0 i : AnalyticSignRequirement).Holds
        ((Fin.elim0 i : Ambient n → ℝ) (gamma x.1, x.2)) :=
    hacc.mono (fun x hx => ⟨hx.1, sub_eq_zero.mpr hx.2, fun i => Fin.elim0 i⟩)
  obtain ⟨q, g, r, hq, hr, hg, hg0, heq, hminimum⟩ :=
    analytic_fiber_minimum_lifting F hF horder gamma hgamma hgamma0 V hV
      (fun i : Fin 0 => Fin.elim0 i) (fun i => Fin.elim0 i) Fin.elim0 haccF
  refine ⟨q, g, r, hq, hr, hg, hg0, heq.mono (fun s hs => sub_eq_zero.mp hs), ?_⟩
  filter_upwards [hminimum] with s hs
  refine ⟨hs.1, ?_⟩
  intro y hy hyE
  have hsq := hs.2.2 y hy (sub_eq_zero.mpr hyE) (fun i => Fin.elim0 i)
  exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hsq

/-- The degenerate energy `x⁴`, a linear distinguished-variable map, and
the moving level `t` have roots accumulating via `(t,y)=(s⁴,s)`. All
analyticity, exact-order and selection hypotheses hold simultaneously.
Auxiliary example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let E : EucSpace 1 → ℝ := fun x => (x 0) ^ 4
    let phi : Ambient 1 → EucSpace 1 := fun x => x.2 • PiLp.single 2 (0 : Fin 1) 1
    let gamma : ℝ → Base 1 := fun t _ => t
    ∃ (q : ℕ) (g : ℝ → ℝ) (r : ℝ), 0 < q ∧ 0 < r ∧
      AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), E (phi (gamma (s ^ q), g s)) = s ^ q) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), |g s| < r ∧
        ∀ y : ℝ, |y| < r → E (phi (gamma (s ^ q), y)) = s ^ q →
          ‖gradient E (phi (gamma (s ^ q), g s))‖ ≤ ‖gradient E (phi (gamma (s ^ q), y))‖ := by
  intro E phi gamma
  have hE : AnalyticAt ℝ E 0 :=
    ((EuclideanSpace.proj 0 : EucSpace 1 →L[ℝ] ℝ).analyticAt 0).fun_pow 4
  have hphi : AnalyticAt ℝ phi 0 := analyticAt_snd.smul analyticAt_const
  let level : Base 1 → ℝ := fun p => p 0
  have hlevel : AnalyticAt ℝ level 0 :=
    (ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0
  have horder : ExactOrderInLastVariable (fun x => E (phi x) - level x.1) 4 := by
    have hslice : lastSlice (fun x => E (phi x) - level x.1) = fun t : ℝ => t ^ 4 := by
      funext t
      simp [lastSlice, E, phi, level]
    rw [ExactOrderInLastVariable, hslice]
    have hord : analyticOrderAt (fun t : ℝ => t ^ 4) 0 = 4 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 4
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 4)).mp hord
  have hgamma : AnalyticAt ℝ gamma 0 := AnalyticAt.pi (fun _ => analyticAt_id)
  have hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
      E (phi (gamma x.1, x.2)) = level (gamma x.1) := by
    have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    let path : ℝ → ℝ × ℝ := fun t => (t ^ 4, t)
    have ht : Tendsto path (nhdsWithin 0 (Ioi 0)) (nhds (0 : ℝ × ℝ)) := by
      have hc : ContinuousAt path 0 := by fun_prop
      simpa [path, Prod.mk_zero_zero] using hc.tendsto.mono_left nhdsWithin_le_nhds
    have hp : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply ht.frequently
    exact hp.frequently.mono (fun t ht => ⟨pow_pos ht 4, by simp [E, phi, gamma, level, path]⟩)
  exact analytic_gradient_fiber_minimum_lifting E 0 hE phi hphi (by simp [phi])
    level hlevel horder gamma hgamma rfl hacc

end Transformer.Normalization
