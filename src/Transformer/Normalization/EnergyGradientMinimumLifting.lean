/-
# Finite-fiber gradient minima for every nonconstant analytic energy

Regular energy coordinates are constructed from analyticity. Including
the level as a base parameter then gives analytic gradient minima over
every supplied analytic parameter curve, with universal root comparison.
-/

import Transformer.Normalization.GradientFiberMinimumLifting

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- Every nonconstant analytic energy germ in positive dimension has
invertible coordinates in which gradient-norm minimization on the scalar
fibers can be lifted over any analytic parameter curve after ramification.
The energy is included among the parameters, and every nearby root on
that fiber is compared. Coordinates, finite directional order and the
minimizing branch are constructed from analyticity. This is the general
finite-fiber part of Appendix D.1, `lem: loj`, of arXiv:2510.22026v2;
selection and minimization across all remaining base coordinates remain
the higher-dimensional geometric step. -/
theorem exists_energy_gradient_minimum_lifting_coordinates {n : ℕ}
    (E : EucSpace (n + 1) → ℝ) (z : EucSpace (n + 1))
    (hE : AnalyticAt ℝ E z) (hnonconstant : ¬ ∀ᶠ y in nhds z, E y = E z) :
    ∃ L : Ambient n ≃L[ℝ] EucSpace (n + 1),
      ∀ gamma : ℝ → Base (n + 1), AnalyticAt ℝ gamma 0 → gamma 0 = 0 →
        (∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
          E (z + L ((fun i => gamma x.1 i.castSucc), x.2)) - E z = gamma x.1 (Fin.last n)) →
        ∃ (q : ℕ) (g : ℝ → ℝ) (r : ℝ), 0 < q ∧ 0 < r ∧
          AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
          (∀ᶠ s in nhds (0 : ℝ),
            E (z + L ((fun i => gamma (s ^ q) i.castSucc), g s)) - E z =
              gamma (s ^ q) (Fin.last n)) ∧
          ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), |g s| < r ∧
            ∀ y : ℝ, |y| < r →
              E (z + L ((fun i => gamma (s ^ q) i.castSucc), y)) - E z =
                gamma (s ^ q) (Fin.last n) →
              ‖gradient E (z + L ((fun i => gamma (s ^ q) i.castSucc), g s))‖ ≤
                ‖gradient E (z + L ((fun i => gamma (s ^ q) i.castSucc), y))‖ := by
  obtain ⟨L, d, _, _, horder⟩ := exists_regular_energy_coordinates E z hE hnonconstant
  let base : Base (n + 1) →L[ℝ] Base n :=
    ContinuousLinearMap.pi (fun i => ContinuousLinearMap.proj i.castSucc)
  let project : Ambient (n + 1) →L[ℝ] Ambient n :=
    (base.comp (ContinuousLinearMap.fst ℝ (Base (n + 1)) ℝ)).prod
      (ContinuousLinearMap.snd ℝ (Base (n + 1)) ℝ)
  let phi : Ambient (n + 1) → EucSpace (n + 1) := fun x => z + L (project x)
  have hphi : AnalyticAt ℝ phi 0 :=
    analyticAt_const.fun_add ((L.toContinuousLinearMap.analyticAt (project 0)).comp
      (f := project) (x := 0) (project.analyticAt 0))
  have hphi0 : phi 0 = z := by simp only [phi, map_zero, add_zero]
  let level : Base (n + 1) → ℝ := fun p => E z + p (Fin.last n)
  have hlevel : AnalyticAt ℝ level 0 :=
    analyticAt_const.fun_add
      ((ContinuousLinearMap.proj (Fin.last n) : Base (n + 1) →L[ℝ] ℝ).analyticAt 0)
  have hfamilyOrder : ExactOrderInLastVariable (fun x => E (phi x) - level x.1) d := by
    have hslice : lastSlice (fun x => E (phi x) - level x.1) =
        lastSlice (fun x : Ambient n => E (z + L x) - E z) := by
      funext t
      change E (z + L ((fun i : Fin n => (0 : Base (n + 1)) i.castSucc), t)) -
        (E z + (0 : Base (n + 1)) (Fin.last n)) = E (z + L (0, t)) - E z
      simp only [Pi.zero_apply, add_zero]
      rfl
    simpa only [ExactOrderInLastVariable, hslice] using horder
  refine ⟨L, ?_⟩
  intro gamma hgamma hgamma0 hacc
  have haccE : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
      E (phi (gamma x.1, x.2)) = level (gamma x.1) := by
    apply hacc.mono
    intro x hx
    refine ⟨hx.1, ?_⟩
    change E (z + L ((fun i => gamma x.1 i.castSucc), x.2)) =
      E z + gamma x.1 (Fin.last n)
    linarith [hx.2]
  obtain ⟨q, g, r, hq, hr, hg, hg0, heq, hmin⟩ :=
    analytic_gradient_fiber_minimum_lifting E z hE phi hphi hphi0
      level hlevel hfamilyOrder gamma hgamma hgamma0 haccE
  refine ⟨q, g, r, hq, hr, hg, hg0, ?_, ?_⟩
  · exact heq.mono (fun s hs => by
      change E (z + L ((fun i => gamma (s ^ q) i.castSucc), g s)) =
        E z + gamma (s ^ q) (Fin.last n) at hs
      linarith)
  · filter_upwards [hmin] with s hs
    refine ⟨hs.1, ?_⟩
    intro y hy hyE
    apply hs.2 y hy
    change E (z + L ((fun i => gamma (s ^ q) i.castSucc), y)) =
      E z + gamma (s ^ q) (Fin.last n)
    linarith

/-- For the degenerate energy `x₀²+x₁⁴`, the constructed coordinates
give a genuine moving level along their distinguished axis. Its roots
accumulate via the axis itself, and the selected branch compares the
gradient norm with every nearby root on that level. This exercises the
coordinate existence and all curve-lifting hypotheses jointly. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let E : EucSpace 2 → ℝ := fun x => (x 0) ^ 2 + (x 1) ^ 4
    ∃ (L : Ambient 1 ≃L[ℝ] EucSpace 2) (q : ℕ) (g : ℝ → ℝ) (r : ℝ),
      0 < q ∧ 0 < r ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
        (∀ᶠ s in nhds (0 : ℝ), E (L (0, g s)) = E (L (0, s ^ q))) ∧
        ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), |g s| < r ∧
          ∀ y : ℝ, |y| < r → E (L (0, y)) = E (L (0, s ^ q)) →
            ‖gradient E (L (0, g s))‖ ≤ ‖gradient E (L (0, y))‖ := by
  intro E
  have hE : AnalyticAt ℝ E 0 :=
    (((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2).fun_add
      (((EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 4)
  have hnonconstant : ¬ ∀ᶠ y in nhds (0 : EucSpace 2), E y = E 0 := by
    intro hzero
    let axis : ℝ → EucSpace 2 := fun t => t • PiLp.single 2 (0 : Fin 2) 1
    have ht : Tendsto axis (nhds 0) (nhds 0) := by
      have hc : ContinuousAt axis 0 := continuousAt_id.smul continuousAt_const
      simpa [axis] using hc.tendsto
    have htzero : ∀ᶠ t in nhds (0 : ℝ), t ^ 2 = 0 := by
      simpa [E, axis] using ht.eventually hzero
    obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp htzero
    have hz : (r / 2) ^ 2 = 0 := hball (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hr)] using half_lt_self hr)
    exact (sq_pos_of_pos (half_pos hr)).ne' hz
  obtain ⟨L, hlift⟩ := exists_energy_gradient_minimum_lifting_coordinates E 0 hE hnonconstant
  let f : ℝ → ℝ := fun t => E (L (0, t)) - E 0
  have haxis : AnalyticAt ℝ (fun t : ℝ => L (0, t)) 0 :=
    (by simpa only [Prod.mk_zero_zero, map_zero] using L.toContinuousLinearMap.analyticAt 0 :
      AnalyticAt ℝ L.toContinuousLinearMap ((0 : Base 1), (0 : ℝ))).comp
        (f := fun t : ℝ => ((0 : Base 1), t)) (x := 0) (analyticAt_const.prod analyticAt_id)
  have hf : AnalyticAt ℝ f 0 :=
    ((by simpa only [Prod.mk_zero_zero, map_zero] using hE :
      AnalyticAt ℝ E (L ((0 : Base 1), (0 : ℝ)))).comp
        (f := fun t : ℝ => L (0, t)) (x := 0) haxis).fun_sub analyticAt_const
  have hf0 : f 0 = 0 := by simp [f, Prod.mk_zero_zero]
  let gamma : ℝ → Base 2 := fun t i => if i = Fin.last 1 then f t else 0
  have hgamma : AnalyticAt ℝ gamma 0 := by
    apply AnalyticAt.pi
    intro i
    split_ifs
    · exact hf
    · exact analyticAt_const
  have hgamma0 : gamma 0 = 0 := by funext i; simp [gamma, hf0]
  have hbase (t : ℝ) : (fun i : Fin 1 => gamma t i.castSucc) = 0 := by
    funext i
    change (if i.castSucc = Fin.last 1 then f t else 0) = 0
    simp only [ite_eq_right_iff]
    intro hi
    exact (Fin.castSucc_ne_last i hi).elim
  have hlast (t : ℝ) : gamma t (Fin.last 1) = f t := by simp [gamma]
  have hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
      E (0 + L ((fun i => gamma x.1 i.castSucc), x.2)) - E 0 = gamma x.1 (Fin.last 1) := by
    have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    have ht : Tendsto (fun t : ℝ => (t, t)) (nhdsWithin 0 (Ioi 0)) (nhds (0 : ℝ × ℝ)) :=
      (tendsto_id.prodMk_nhds tendsto_id).mono_left nhdsWithin_le_nhds
    have hp : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply ht.frequently
    exact hp.frequently.mono (fun t ht =>
      ⟨ht, by simp only [hbase, zero_add, hlast]; rfl⟩)
  obtain ⟨q, g, r, hq, hr, hg, hg0, heq, hmin⟩ := hlift gamma hgamma hgamma0 hacc
  refine ⟨L, q, g, r, hq, hr, hg, hg0, ?_, ?_⟩
  · exact heq.mono (fun s hs => sub_left_inj.mp
      (show E (L (0, g s)) - E 0 = E (L (0, s ^ q)) - E 0 by
        simpa only [hbase, zero_add, hlast, f] using hs))
  · filter_upwards [hmin] with s hs
    refine ⟨hs.1, ?_⟩
    intro y hy hyE
    have hlevel : E (0 + L ((fun i => gamma (s ^ q) i.castSucc), y)) - E 0 =
        gamma (s ^ q) (Fin.last 1) := by
      simp only [hbase, zero_add, hlast, f, hyE]
    simpa only [hbase, zero_add] using hs.2 y hy hlevel

end Transformer.Normalization
