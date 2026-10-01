/-
# A moving quadratic level witnesses global minimum lifting

The projected curve is the level parameter. Square-root witnesses
approach zero, and every point on a quadratic level has the same gradient
norm. The lifted analytic curve retains comparison with the whole space.
-/

import Transformer.Normalization.ProjectedGradientMinimumLifting
import Transformer.Normalization.QuadraticEnergy

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- A quadratic energy and its moving positive levels satisfy all global
minimum-lifting hypotheses, with comparison over the entire ambient
space. The source set is unconstrained, while the roots have order two
and the pointwise square-root witnesses are not analytic before
ramification. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : ∃ (q : ℕ) (curve : ℝ → EucSpace 1), 0 < q ∧
    AnalyticAt ℝ curve 0 ∧ curve 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), ‖curve s‖ ^ 2 = s ^ q) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∀ w : EucSpace 1,
        ‖w‖ ^ 2 = s ^ q →
          ‖gradient (fun x : EucSpace 1 => ‖x‖ ^ 2) (curve s)‖ ≤
            ‖gradient (fun x : EucSpace 1 => ‖x‖ ^ 2) w‖ := by
  let phi : Ambient 1 → EucSpace 1 := fun x => x.2 • PiLp.single 2 (0 : Fin 1) 1
  have hphi : AnalyticAt ℝ phi 0 := analyticAt_snd.smul analyticAt_const
  have hphi0 : phi 0 = 0 := by simp [phi]
  have hnorm (x : Ambient 1) : ‖phi x‖ = |x.2| := by simp [phi, norm_smul]
  let level : Base 1 → ℝ := fun p => p 0
  have hlevel : AnalyticAt ℝ level 0 :=
    (ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0
  have horder : ExactOrderInLastVariable (fun x => ‖phi x‖ ^ 2 - level x.1) 2 := by
    have hslice : lastSlice (fun x => ‖phi x‖ ^ 2 - level x.1) = fun t : ℝ => t ^ 2 := by
      funext t
      simp [lastSlice, level, hnorm, sq_abs]
    rw [ExactOrderInLastVariable, hslice]
    have hord : analyticOrderAt (fun t : ℝ => t ^ 2) 0 = 2 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 2
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 2)).mp hord
  let gamma : ℝ → Base 1 := fun t _ => t
  have hgamma : AnalyticAt ℝ gamma 0 := AnalyticAt.pi (fun _ => analyticAt_id)
  have hK : ∀ᶠ x in nhds (0 : Ambient 1), phi x ∈ (univ : Set (EucSpace 1)) ↔
      ∀ i : Fin 0, (Fin.elim0 i : AnalyticSignRequirement).Holds
        ((Fin.elim0 i : Ambient 1 → ℝ) x) :=
    Eventually.of_forall (fun x => ⟨fun _ i => Fin.elim0 i, fun _ => mem_univ _⟩)
  have hprojected : ∀ delta : ℝ, 0 < delta → ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, |y| < delta ∧ ‖phi (gamma t, y)‖ ^ 2 = level (gamma t) ∧
        phi (gamma t, y) ∈ (univ : Set (EucSpace 1)) ∧
          ∀ w ∈ (univ : Set (EucSpace 1)), ‖w‖ ^ 2 = level (gamma t) →
            ‖gradient (fun x : EucSpace 1 => ‖x‖ ^ 2) (phi (gamma t, y))‖ ≤
              ‖gradient (fun x : EucSpace 1 => ‖x‖ ^ 2) w‖ := by
    intro delta hd
    have hsqrt : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), |Real.sqrt t| < delta := by
      have hc : ContinuousAt (fun t : ℝ => |Real.sqrt t|) 0 :=
        Real.continuous_sqrt.continuousAt.abs
      exact (hc.eventually (Iio_mem_nhds (by simpa using hd))).filter_mono nhdsWithin_le_nhds
    have hp : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    filter_upwards [hsqrt, hp] with t hs ht
    have henergy : ‖phi (gamma t, Real.sqrt t)‖ ^ 2 = t := by
      simpa only [hnorm, sq_abs] using Real.sq_sqrt ht.le
    refine ⟨Real.sqrt t, hs, henergy, mem_univ _, ?_⟩
    intro w hw hwE
    have heqnorm : ‖phi (gamma t, Real.sqrt t)‖ = ‖w‖ :=
      (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp (henergy.trans hwE.symm)
    rw [quadratic_energy_gradient, quadratic_energy_gradient]
    apply le_of_eq
    calc
      ‖(2 : ℝ) • phi (gamma t, Real.sqrt t)‖ =
          ‖(2 : ℝ)‖ * ‖phi (gamma t, Real.sqrt t)‖ := norm_smul _ _
      _ = ‖(2 : ℝ)‖ * ‖w‖ := by rw [heqnorm]
      _ = ‖(2 : ℝ) • w‖ := (norm_smul _ _).symm
  obtain ⟨q, curve, hq, hc, hc0, heq, hmin⟩ :=
    analytic_global_gradient_minimum_lifting (fun x : EucSpace 1 => ‖x‖ ^ 2) 0
      (quadratic_energy_analytic 0 (mem_univ _)) phi hphi hphi0 level hlevel horder
      gamma hgamma rfl univ (fun i : Fin 0 => Fin.elim0 i) (fun i => Fin.elim0 i)
      Fin.elim0 hK hprojected
  refine ⟨q, curve, hq, hc, hc0, heq, ?_⟩
  exact hmin.mono (fun s hs w hw => hs.2 w (mem_univ _) hw)

end Transformer.Normalization
