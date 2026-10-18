/-
# Local uniformization

Every real analytic function on `ℝⁿ` that does not vanish identically near `z` has
monomial charts at `z` (`hasMonomialCharts_of_analyticAt`). By induction on `n`: after a
translation and a linear change of coordinates the function has a finite order `d` in
the last variable. By strong induction on `d`, a function of `x'` times a function of
order `d` in `y` is, by Weierstrass preparation and a translation in `y`, a product
`m · V · P` with `P` a centered monic polynomial in `y`, which has monomial charts by
the product step. This is the local uniformization behind the gradient inequality of
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.Product
import Transformer.AnalyticPreparation.PreparationReconstruction
import Transformer.RealAnalyticGerms.RegularCoordinates

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base Ambient preparedPolynomial ExactOrderInLastVariable
  exists_isWeierstrassPreparation)

variable {n : ℕ}

theorem init_zero : (Fin.init (0 : Base (n + 1)) : Base n) = 0 := rfl

/-- **The step of the induction on the order.** A function of `x'` that does not vanish
identically times a function of order exactly `d` in the last variable has monomial
charts at `0`, given monomial charts in dimension `n` and for such products of lower
order. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_of_order_eq {d : ℕ}
    (hunif : ∀ (f : Base n → ℝ) (z : Base n), AnalyticAt ℝ f z → ¬ f =ᶠ[𝓝 z] 0 →
      HasMonomialCharts f z)
    (ih : ∀ d' < d, ∀ (m : Base n → ℝ) (g : Base (n + 1) → ℝ) (z : Base (n + 1)),
      AnalyticAt ℝ m (Fin.init z) → ¬ m =ᶠ[𝓝 (Fin.init z)] 0 → AnalyticAt ℝ g z →
      analyticOrderAt (fun y => g (Fin.snoc (Fin.init z) (z (Fin.last n) + y))) 0 ≤ d' →
      HasMonomialCharts (fun x => m (Fin.init x) * g x) z)
    {m : Base n → ℝ} {g : Base (n + 1) → ℝ} (hm : AnalyticAt ℝ m 0) (hm0 : ¬ m =ᶠ[𝓝 0] 0)
    (hg : AnalyticAt ℝ g 0) (hord : analyticOrderAt (fun y => g (Fin.snoc 0 y)) 0 = d) :
    HasMonomialCharts (fun x => m (Fin.init x) * g x) 0 := by
  -- Weierstrass preparation of `g` on `ℝⁿ × ℝ`
  have hf : AnalyticAt ℝ (fun x : Ambient n => g (Fin.snoc x.1 x.2)) 0 :=
    hg.comp_of_eq (AnalyticAt.snoc analyticAt_fst analyticAt_snd) snoc_zero_zero
  have hslice : AnalyticAt ℝ (fun y => g (Fin.snoc 0 y)) 0 :=
    hg.comp_of_eq (AnalyticAt.snoc analyticAt_const analyticAt_id) snoc_zero_zero
  have hexact : ExactOrderInLastVariable (fun x : Ambient n => g (Fin.snoc x.1 x.2)) d :=
    (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero hslice).1 hord
  obtain ⟨a, u, ha, ha0, hu, hu0, hprep⟩ := exists_isWeierstrassPreparation hf hexact
  obtain ⟨s, b, hs, hs0, hb, -, hbc, hsb⟩ := exists_centering a ha ha0
  -- the translation `y ↦ y + s x'`
  let T : Base (n + 1) → Base (n + 1) := fun x =>
    Fin.snoc (Fin.init x) (x (Fin.last n) + s (Fin.init x))
  let T' : Base (n + 1) → Base (n + 1) := fun x =>
    Fin.snoc (Fin.init x) (x (Fin.last n) - s (Fin.init x))
  have hT0 : T 0 = 0 := by
    simp only [T, init_zero, Pi.zero_apply, hs0, add_zero, snoc_zero_zero]
  have hTA : ∀ᶠ x in 𝓝 (0 : Base (n + 1)), AnalyticAt ℝ T x := by
    filter_upwards [(analyticAt_init (0 : Base (n + 1))).continuousAt.tendsto.eventually
      hs.eventually_analyticAt] with x hx
    exact AnalyticAt.snoc (analyticAt_init x)
      ((analyticAt_last x).fun_add (hx.comp (analyticAt_init x)))
  have hT'A : AnalyticAt ℝ T' (T 0) := by
    rw [hT0]
    exact AnalyticAt.snoc (analyticAt_init 0)
      ((analyticAt_last 0).fun_sub (hs.comp_of_eq (analyticAt_init 0) init_zero))
  have hT'T : T' (T 0) = 0 := by
    rw [hT0]
    simp only [T', init_zero, Pi.zero_apply, hs0, sub_zero, snoc_zero_zero]
  have hTT' : ∀ y, T (T' y) = y := fun y => by
    simp only [T, T', Fin.init_snoc, Fin.snoc_last, sub_add_cancel, Fin.snoc_init_self]
  -- the unit in the new coordinates
  let π : Base (n + 1) → Ambient n := fun x => (Fin.init x, x (Fin.last n) + s (Fin.init x))
  have hπA : AnalyticAt ℝ π 0 := (analyticAt_init 0).prod
    ((analyticAt_last 0).fun_add (hs.comp_of_eq (analyticAt_init 0) init_zero))
  have hπ0 : π 0 = 0 := by
    simp only [π, init_zero, Pi.zero_apply, hs0, add_zero]
    rfl
  have hV : AnalyticAt ℝ (fun x => u (π x)) 0 := hu.comp_of_eq hπA hπ0
  have hV0 : u (π 0) ≠ 0 := by rwa [hπ0]
  have hprod := hasMonomialCharts_product hunif
    (fun m V c x₀ hm hm0 hV hV0 hc hc0 hne =>
      hasMonomialCharts_of_centered ih hm hm0 hV hV0 hc hc0 hne) hm hm0 hV hV0 hb hbc
  have hcongr : (fun x => m (Fin.init x) * (u (π x) *
      preparedPolynomial d b (Fin.init x, x (Fin.last n)))) =ᶠ[𝓝 0]
      (fun x => m (Fin.init x) * g x) ∘ T := by
    have ht : Tendsto π (𝓝 0) (𝓝 0) := by
      have h := hπA.continuousAt.tendsto
      rwa [hπ0] at h
    filter_upwards [ht.eventually hprep] with x hx
    simp only [Function.comp_apply, T, Fin.init_snoc]
    simp only [π] at hx
    rw [hx, hsb]
  have h := HasMonomialCharts.comp (hprod.congr hcongr) hTA hT'A.continuousAt hT'T
    (Eventually.of_forall hTT')
  rwa [hT0] at h

/-- **Induction on the order.** A function of `x'` that does not vanish identically times
a function of order at most `d` in the last variable has monomial charts, given monomial
charts in dimension `n`. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_mul_of_order_le
    (hunif : ∀ (f : Base n → ℝ) (z : Base n), AnalyticAt ℝ f z → ¬ f =ᶠ[𝓝 z] 0 →
      HasMonomialCharts f z) (d : ℕ) :
    ∀ (m : Base n → ℝ) (g : Base (n + 1) → ℝ) (z : Base (n + 1)),
      AnalyticAt ℝ m (Fin.init z) → ¬ m =ᶠ[𝓝 (Fin.init z)] 0 → AnalyticAt ℝ g z →
      analyticOrderAt (fun y => g (Fin.snoc (Fin.init z) (z (Fin.last n) + y))) 0 ≤ d →
      HasMonomialCharts (fun x => m (Fin.init x) * g x) z := by
  induction d using Nat.strong_induction_on with
  | _ d ih =>
  intro m g z hm hm0 hg hord
  obtain ⟨k, hk⟩ := ENat.ne_top_iff_exists.1 (ne_top_of_le_ne_top (ENat.natCast_ne_top d) hord)
  have hkd : k ≤ d := by
    rw [← hk] at hord
    exact_mod_cast hord
  rcases hkd.lt_or_eq with hkd | hkd
  · exact ih k hkd m g z hm hm0 hg hk.symm.le
  refine hasMonomialCharts_of_add (hasMonomialCharts_of_order_eq hunif ih
    (m := fun x' => m (x' + Fin.init z)) (g := fun x => g (x + z))
    (hm.comp_of_eq (analyticAt_id.fun_add analyticAt_const) (zero_add _)) ?_
    (hg.comp_of_eq (analyticAt_id.fun_add analyticAt_const) (zero_add _)) ?_)
  · intro h
    apply hm0
    have ht : Tendsto (fun x' : Base n => x' - Fin.init z) (𝓝 (Fin.init z)) (𝓝 0) := by
      simpa using (continuous_sub_right (Fin.init z)).continuousAt.tendsto (x := Fin.init z)
    filter_upwards [ht.eventually h] with x' hx'
    simpa using hx'
  · have hfun : (fun y => g (Fin.snoc 0 y + z)) =
        fun y => g (Fin.snoc (Fin.init z) (z (Fin.last n) + y)) := by
      funext y
      conv_lhs => rw [← Fin.snoc_init_self z]
      rw [snoc_add_snoc, zero_add, add_comm y]
    rw [hfun, ← hk, hkd]

/-- **Local uniformization.** Every real analytic function on `ℝⁿ` that does not vanish
identically near `z` has monomial charts at `z`. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_of_analyticAt {f : Base n → ℝ} {z : Base n}
    (hf : AnalyticAt ℝ f z) (hf0 : ¬ f =ᶠ[𝓝 z] 0) : HasMonomialCharts f z := by
  induction n with
  | zero =>
    refine hasMonomialCharts_of_normalCrossingAt
      (normalCrossingAt_of_ne_zero hf fun hz => hf0 ?_)
    exact Eventually.of_forall fun x => by simp [Subsingleton.elim x z, hz]
  | succ n ih =>
    refine hasMonomialCharts_of_add ?_
    have hF : AnalyticAt ℝ (fun x => f (x + z)) 0 :=
      hf.comp_of_eq (analyticAt_id.fun_add analyticAt_const) (zero_add z)
    have hF0 : ¬ (fun x => f (x + z)) =ᶠ[𝓝 0] 0 := by
      intro h
      apply hf0
      have ht : Tendsto (fun x : Base (n + 1) => x - z) (𝓝 z) (𝓝 0) := by
        simpa using (continuous_sub_right z).continuousAt.tendsto (x := z)
      filter_upwards [ht.eventually h] with x hx
      simpa using hx
    obtain ⟨L, d, hL⟩ := RealAnalyticGerms.exists_regularizing_coordinateEquiv hF hF0
    have hG : AnalyticAt ℝ (fun x => f (L x + z)) 0 :=
      hF.comp_of_eq ((L : Base (n + 1) →L[ℝ] Base (n + 1)).analyticAt 0) (map_zero L)
    have hsymm : ∀ y : ℝ, (RealAnalyticGerms.wptAmbientEquiv n).symm ((0 : Base n), y) =
        (Fin.snoc 0 y : Base (n + 1)) := fun y => by
      funext i
      induction i using Fin.lastCases with
      | last => simp
      | cast i => simp
    have hslice : AnalyticAt ℝ (fun y => f (L (Fin.snoc 0 y) + z)) 0 :=
      hG.comp_of_eq (AnalyticAt.snoc analyticAt_const analyticAt_id) snoc_zero_zero
    have hord : analyticOrderAt (fun y => f (L (Fin.snoc 0 y) + z)) 0 = d := by
      have hls : AnalyticPreparation.lastSlice
          (fun x : Ambient n => f (L ((RealAnalyticGerms.wptAmbientEquiv n).symm x) + z)) =
          fun y => f (L (Fin.snoc 0 y) + z) := by
        funext y
        simp only [AnalyticPreparation.lastSlice, hsymm]
      refine (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero hslice).2 ?_
      simpa only [ExactOrderInLastVariable, hls] using hL
    have h := hasMonomialCharts_mul_of_order_le (fun f z hf hf0 => ih hf hf0) d (fun _ => 1)
      (fun x => f (L x + z)) 0 analyticAt_const
      (fun h => one_ne_zero (h.self_of_nhds : (1 : ℝ) = 0)) hG
      (by simp only [init_zero, Pi.zero_apply, zero_add, hord, le_refl])
    have h2 : HasMonomialCharts ((fun x => f (x + z)) ∘ L) 0 :=
      h.congr (Eventually.of_forall fun x => by simp only [Function.comp_apply, one_mul])
    have h3 := HasMonomialCharts.comp h2
      (Eventually.of_forall fun x => (L : Base (n + 1) →L[ℝ] Base (n + 1)).analyticAt x)
      L.symm.continuous.continuousAt (by simp) (Eventually.of_forall fun y => L.apply_symm_apply y)
    rwa [map_zero] at h3

end Transformer.Uniformization
