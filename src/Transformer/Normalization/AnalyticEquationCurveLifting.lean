/-
# Lifting analytic parameter curves through prepared zero sets

Real preparation and complete constrained polynomial branch lifting apply
to arbitrary analytic equations regular in the distinguished variable.
An analytic curve in the base parameters lifts after ramification while
retaining any finite family of accumulating analytic sign conditions.
-/

import Transformer.Normalization.ConstrainedRootLifting

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- An analytic parameter curve through the base origin lifts through an
analytic equation of finite distinguished-variable order, after positive
integral ramification. Every finite analytic equality or sign condition
holding along accumulating positive-side roots is retained. The lifted
equation holds on a full neighborhood. This proves the singular lifting
step in arbitrary parameter dimension for the curve-selection route in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. Selection of a base curve
in a projected set is a separate geometric step. -/
theorem analytic_equation_curve_lifting {n d l : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0) (horder : ExactOrderInLastVariable F d)
    (gamma : ℝ → Base n) (hgamma : AnalyticAt ℝ gamma 0) (hgamma0 : gamma 0 = 0)
    (C : Fin l → Ambient n → ℝ) (hC : ∀ i, AnalyticAt ℝ (C i) 0)
    (requirement : Fin l → AnalyticSignRequirement)
    (hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ F (gamma x.1, x.2) = 0 ∧
      ∀ i, (requirement i).Holds (C i (gamma x.1, x.2))) :
    ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), F (gamma (s ^ q), g s) = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∀ i,
        (requirement i).Holds (C i (gamma (s ^ q), g s)) := by
  obtain ⟨A, U, hprep⟩ := exists_isWeierstrassPreparation hF horder
  let a : Fin d → ℝ → ℝ := fun i t => A i (gamma t)
  have ha (i : Fin d) : AnalyticAt ℝ (a i) 0 :=
    (by simpa only [hgamma0] using hprep.1 i : AnalyticAt ℝ (A i) (gamma 0)).comp
      (f := gamma) (x := 0) hgamma
  have ha0 (i : Fin d) : a i 0 = 0 := by simpa only [a, hgamma0] using hprep.2.1 i
  let liftPath : ℝ × ℝ → Ambient n := fun x => (gamma x.1, x.2)
  have hlift : AnalyticAt ℝ liftPath 0 :=
    (hgamma.comp (f := fun x : ℝ × ℝ => x.1) (x := 0) analyticAt_fst).prod analyticAt_snd
  have hlift0 : liftPath 0 = 0 := by simp [liftPath, hgamma0, Prod.mk_zero_zero]
  have hliftT : Tendsto liftPath (nhds 0) (nhds 0) := by
    simpa only [hlift0] using hlift.continuousAt.tendsto
  let constraints : Fin l → ℝ × ℝ → ℝ := fun i x => C i (liftPath x)
  have hc (i : Fin l) : AnalyticAt ℝ (constraints i) 0 :=
    (by simpa only [hlift0] using hC i : AnalyticAt ℝ (C i) (liftPath 0)).comp
      (f := liftPath) (x := 0) hlift
  have hidentity : ∀ᶠ x in nhds (0 : ℝ × ℝ),
      F (gamma x.1, x.2) = U (liftPath x) * polynomialFamily d a x :=
    hliftT.eventually hprep.2.2.2.2
  have hunit : ∀ᶠ x in nhds (0 : ℝ × ℝ), U (liftPath x) ≠ 0 :=
    hliftT.eventually (hprep.2.2.1.continuousAt.eventually_ne hprep.2.2.2.1)
  have hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
      polynomialFamily d a (t, y) = 0 ∧ ∀ i,
        (requirement i).Holds (constraints i (t, y)) := by
    by_contra h
    have hno : ∀ᶠ t in nhds (0 : ℝ), 0 < t → ¬ ∃ y : ℝ,
        polynomialFamily d a (t, y) = 0 ∧ ∀ i,
          (requirement i).Holds (constraints i (t, y)) :=
      eventually_nhdsWithin_iff.mp (not_frequently.mp h)
    have hfst : Tendsto (fun x : ℝ × ℝ => x.1) (nhds 0) (nhds 0) :=
      continuous_fst.tendsto 0
    obtain ⟨x, hx⟩ := (hacc.and_eventually
      (hidentity.and (hunit.and (hfst.eventually hno)))).exists
    have hroot : polynomialFamily d a x = 0 :=
      (mul_eq_zero.mp (hx.2.1.symm.trans hx.1.2.1)).resolve_left hx.2.2.1
    exact hx.2.2.2 hx.1.1 ⟨x.2, hroot, hx.1.2.2⟩
  obtain ⟨q, g, hq, hg, hg0, hroot, hconditions⟩ :=
    analytic_root_with_sign_constraints a ha ha0 constraints hc requirement hroots
  refine ⟨q, g, hq, hg, hg0, ?_, hconditions⟩
  have hpath : Tendsto (fun s : ℝ => (s ^ q, g s)) (nhds 0) (nhds (0 : ℝ × ℝ)) := by
    have haPath : AnalyticAt ℝ (fun s : ℝ => (s ^ q, g s)) 0 :=
      (analyticAt_id.fun_pow q).prod hg
    simpa [hq.ne', hg0, Prod.mk_zero_zero] using haPath.continuousAt.tendsto
  filter_upwards [hroot, hpath.eventually hidentity] with s hs hi
  rw [hi, hs, mul_zero]

/-- The cusp equation `y³ = z₀²` over the analytic base curve `z₀(t)=t`
has positive roots accumulating at the origin via `(t,y)=(s³,s²)`. This
jointly satisfies preparation, base-curve, root-accumulation, and positive
sign hypotheses in a genuine singular example. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
    (∀ᶠ s in nhds (0 : ℝ), (g s) ^ 3 - (s ^ q) ^ 2 = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), 0 < g s := by
  let F : Ambient 1 → ℝ := fun x => x.2 ^ 3 - (x.1 0) ^ 2
  let L : Ambient 1 →L[ℝ] ℝ :=
    (ContinuousLinearMap.proj (0 : Fin 1)).comp (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
  have hF : AnalyticAt ℝ F 0 := (analyticAt_snd.fun_pow 3).fun_sub ((L.analyticAt 0).fun_pow 2)
  have horder : ExactOrderInLastVariable F 3 := by
    have hslice : lastSlice F = fun t : ℝ => t ^ 3 := by funext t; simp [lastSlice, F]
    rw [ExactOrderInLastVariable, hslice]
    have hord : analyticOrderAt (fun t : ℝ => t ^ 3) 0 = 3 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 3
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 3)).mp hord
  let gamma : ℝ → Base 1 := fun t _ => t
  have hgamma : AnalyticAt ℝ gamma 0 := AnalyticAt.pi (fun _ => analyticAt_id)
  have hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ F (gamma x.1, x.2) = 0 ∧
      ∀ i : Fin 1, AnalyticSignRequirement.positive.Holds x.2 := by
    have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    let path : ℝ → ℝ × ℝ := fun t => (t ^ 3, t ^ 2)
    have ht : Tendsto path (nhdsWithin 0 (Ioi 0)) (nhds (0 : ℝ × ℝ)) := by
      have hcPath : ContinuousAt path 0 := by fun_prop
      simpa [path, Prod.mk_zero_zero] using hcPath.tendsto.mono_left nhdsWithin_le_nhds
    have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply ht.frequently
    apply hpositive.frequently.mono
    intro t ht
    refine ⟨pow_pos ht 3, ?_, fun i => sq_pos_of_pos ht⟩
    change (t ^ 2) ^ 3 - (t ^ 3) ^ 2 = 0
    ring
  obtain ⟨q, g, hq, hg, hg0, hroot, hsign⟩ := analytic_equation_curve_lifting F hF horder
    gamma hgamma rfl (fun _ x => x.2) (fun _ => analyticAt_snd) (fun _ => .positive) hacc
  refine ⟨q, g, hq, hg, hg0, hroot, ?_⟩
  exact hsign.mono (fun t ht => ht 0)

end Transformer.Normalization
