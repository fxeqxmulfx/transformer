/-
# Local uniformization: germs of analytic functions form a domain

If a product of two analytic functions vanishes near a point, one factor
vanishes near that point: by the identity theorem on a ball, the second factor
vanishes near any point of the ball where the first does not. Consequences:
products of nonzero germs are nonzero, and a common factor `x i - w i` cancels.
An analytic function vanishing on the hyperplane `x i = w i` near `w` is
`(x i - w i)` times an analytic function: after splitting off the coordinate
`i`, this is real Weierstrass division by the distinguished variable.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Landscape.Uniformization.NormalCrossing
import Transformer.Landscape.AnalyticPreparation.AnalyticDivision
import Mathlib.Analysis.Analytic.Uniqueness
import Mathlib.Topology.Algebra.Module.FiniteDimension

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base Ambient preparedPolynomial
  exists_analyticWeierstrassDivision)

variable {n : ℕ}

/-- If a product of analytic functions vanishes near `w`, one factor does.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eventuallyEq_zero_or_of_mul {f g : Base n → ℝ} {w : Base n} (hf : AnalyticAt ℝ f w)
    (hg : AnalyticAt ℝ g w) (hfg : (fun x => f x * g x) =ᶠ[𝓝 w] 0) :
    f =ᶠ[𝓝 w] 0 ∨ g =ᶠ[𝓝 w] 0 := by
  obtain ⟨r, hr, hfr⟩ := hf.exists_ball_analyticOnNhd
  obtain ⟨s, hs, hgs⟩ := hg.exists_ball_analyticOnNhd
  have hρ : 0 < min r s := lt_min hr hs
  have hfρ : AnalyticOnNhd ℝ f (Metric.ball w (min r s)) :=
    hfr.mono (Metric.ball_subset_ball (min_le_left _ _))
  have hgρ : AnalyticOnNhd ℝ g (Metric.ball w (min r s)) :=
    hgs.mono (Metric.ball_subset_ball (min_le_right _ _))
  have hconn : IsPreconnected (Metric.ball w (min r s)) := (convex_ball w _).isPreconnected
  have hprod : EqOn (fun x => f x * g x) 0 (Metric.ball w (min r s)) :=
    (hfρ.mul hgρ).eqOn_zero_of_preconnected_of_eventuallyEq_zero hconn
      (Metric.mem_ball_self hρ) hfg
  by_contra hcon
  rw [not_or] at hcon
  obtain ⟨x, hx, hfx⟩ : ∃ x ∈ Metric.ball w (min r s), f x ≠ 0 := by
    by_contra h
    push Not at h
    exact hcon.1 (eventually_of_mem (Metric.ball_mem_nhds w hρ) h)
  have hgx : g =ᶠ[𝓝 x] 0 := by
    filter_upwards [(hfρ x hx).continuousAt.eventually_ne hfx,
      Metric.isOpen_ball.mem_nhds hx] with y hy hyb
    have := hprod hyb
    simp only [Pi.zero_apply, mul_eq_zero] at this
    exact this.resolve_left hy
  exact hcon.2 (eventually_of_mem (Metric.ball_mem_nhds w hρ)
    (hgρ.eqOn_zero_of_preconnected_of_eventuallyEq_zero hconn hx hgx))

/-- A finite product of analytic germs that do not vanish identically does
not vanish identically. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem not_eventuallyEq_zero_prod {ι : Type*} (s : Finset ι) {F : ι → Base n → ℝ}
    {w : Base n} (hF : ∀ i ∈ s, AnalyticAt ℝ (F i) w) (hF0 : ∀ i ∈ s, ¬ F i =ᶠ[𝓝 w] 0) :
    ¬ (fun x => ∏ i ∈ s, F i x) =ᶠ[𝓝 w] 0 := by
  classical
  induction s using Finset.induction_on with
  | empty =>
    intro h
    simpa using h.self_of_nhds
  | insert a s ha ih =>
    intro h
    simp only [Finset.prod_insert ha] at h
    rcases eventuallyEq_zero_or_of_mul (hF a (Finset.mem_insert_self a s))
      (Finset.analyticAt_fun_prod _ fun i hi => hF i (Finset.mem_insert_of_mem hi)) h
      with h1 | h1
    · exact hF0 a (Finset.mem_insert_self a s) h1
    · exact ih (fun i hi => hF i (Finset.mem_insert_of_mem hi))
        (fun i hi => hF0 i (Finset.mem_insert_of_mem hi)) h1

/-- A coordinate function `x i - w i` does not vanish identically near `w`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem not_eventuallyEq_zero_coord (i : Fin n) (w : Base n) :
    ¬ (fun x : Base n => x i - w i) =ᶠ[𝓝 w] 0 := by
  refine NormalCrossingAt.not_eventuallyEq_zero ⟨Pi.single i 1, fun _ => 1,
    analyticAt_const, one_ne_zero, Eventually.of_forall fun x => ?_⟩
  rw [monomial_single, one_mul, Pi.sub_apply]

/-- A common factor `x i - w i` cancels between analytic functions.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eventuallyEq_of_coord_mul {A B : Base n → ℝ} {w : Base n} (i : Fin n)
    (hA : AnalyticAt ℝ A w) (hB : AnalyticAt ℝ B w)
    (h : ∀ᶠ x in 𝓝 w, (x i - w i) * A x = (x i - w i) * B x) : A =ᶠ[𝓝 w] B := by
  have hcoord : AnalyticAt ℝ (fun x : Base n => x i - w i) w :=
    ((ContinuousLinearMap.proj i : Base n →L[ℝ] ℝ).analyticAt w).sub analyticAt_const
  have h0 : (fun x => (x i - w i) * (A x - B x)) =ᶠ[𝓝 w] 0 := by
    filter_upwards [h] with x hx
    rw [Pi.zero_apply, mul_sub, hx, sub_self]
  rcases eventuallyEq_zero_or_of_mul hcoord (hA.fun_sub hB) h0 with h1 | h1
  · exact absurd h1 (not_eventuallyEq_zero_coord i w)
  · filter_upwards [h1] with x hx
    exact sub_eq_zero.1 hx

/-- The splitting `x ↦ (x without x i, x i)` of `ℝⁿ⁺¹`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def coordSplit {m : ℕ} (i : Fin (m + 1)) : Base (m + 1) ≃L[ℝ] Ambient m :=
  LinearEquiv.toContinuousLinearEquiv
    { toFun := fun x => (fun j => x (i.succAbove j), x i)
      invFun := fun y => Fin.insertNth i y.2 y.1
      left_inv := fun x => Fin.insertNth_self_removeNth i x
      right_inv := fun y => by
        ext <;> simp [Fin.insertNth_apply_succAbove, Fin.insertNth_apply_same]
      map_add' := fun x y => by ext <;> simp
      map_smul' := fun c x => by ext <;> simp }

theorem coordSplit_apply {m : ℕ} (i : Fin (m + 1)) (x : Base (m + 1)) :
    coordSplit i x = (fun j => x (i.succAbove j), x i) := rfl

theorem coordSplit_symm_apply {m : ℕ} (i : Fin (m + 1)) (y : Ambient m) :
    (coordSplit i).symm y = Fin.insertNth i y.2 y.1 := rfl

/-- An analytic function vanishing on the hyperplane `x i = w i` near `w`
is divisible by `x i - w i`. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem exists_eq_coord_mul {F : Base n → ℝ} {w : Base n} (i : Fin n)
    (hF : AnalyticAt ℝ F w) (h0 : ∀ᶠ x in 𝓝 w, F (Function.update x i (w i)) = 0) :
    ∃ G : Base n → ℝ, AnalyticAt ℝ G w ∧ ∀ᶠ x in 𝓝 w, F x = (x i - w i) * G x := by
  cases n with
  | zero => exact i.elim0
  | succ m =>
  let L := coordSplit i
  let h : Ambient m → ℝ := fun y => F (w + L.symm y)
  have haff : ∀ y, AnalyticAt ℝ (fun y : Ambient m => w + L.symm y) y := fun y =>
    analyticAt_const.add ((L.symm : Ambient m →L[ℝ] Base (m + 1)).analyticAt y)
  have hh : AnalyticAt ℝ h 0 := by
    have hF' : AnalyticAt ℝ F (w + L.symm 0) := by rwa [map_zero, add_zero]
    exact hF'.comp (f := fun y : Ambient m => w + L.symm y) (haff 0)
  obtain ⟨q, rem, hq, -, hdiv⟩ := exists_analyticWeierstrassDivision (d := 1) h hh
    (fun _ => 0) (fun _ => analyticAt_const) (fun _ => rfl)
  have hdiv' : ∀ᶠ y in 𝓝 (0 : Ambient m), h y = q y * y.2 + rem 0 y.1 := by
    filter_upwards [hdiv] with y hy
    simpa [preparedPolynomial] using hy
  -- the remainder vanishes, since `h` vanishes on the hyperplane `y.2 = 0`
  have hslice : Tendsto (fun y : Ambient m => (y.1, (0 : ℝ))) (𝓝 0) (𝓝 0) := by
    have : Continuous (fun y : Ambient m => (y.1, (0 : ℝ))) := by fun_prop
    exact this.tendsto' 0 0 rfl
  have hpt : Tendsto (fun y : Ambient m => w + L.symm y) (𝓝 0) (𝓝 w) := by
    have := ((haff 0).continuousAt).tendsto
    rwa [map_zero, add_zero] at this
  have hrem : ∀ᶠ y in 𝓝 (0 : Ambient m), rem 0 y.1 = 0 := by
    filter_upwards [hslice.eventually hdiv', hslice.eventually (hpt.eventually h0)]
      with y hy hy0
    have hupd : Function.update (w + L.symm (y.1, 0)) i (w i) = w + L.symm (y.1, 0) := by
      have hi : L.symm (y.1, 0) i = 0 := by
        simp only [L, coordSplit_symm_apply, Fin.insertNth_apply_same]
      rw [Function.update_eq_iff]
      simp [hi]
    simp only [h, hupd] at hy hy0
    linarith
  have hL : Tendsto (fun x : Base (m + 1) => L (x - w)) (𝓝 w) (𝓝 0) := by
    have : Continuous (fun x : Base (m + 1) => L (x - w)) := by fun_prop
    exact this.tendsto' w 0 (by simp)
  refine ⟨fun x => q (L (x - w)), ?_, ?_⟩
  · have hqL : AnalyticAt ℝ (fun v => q (L v)) 0 := by
      have hq' : AnalyticAt ℝ q (L 0) := by rwa [map_zero]
      exact hq'.comp (L.analyticAt 0)
    have := hqL.comp_sub w
    rwa [zero_add] at this
  · filter_upwards [hL.eventually hdiv', hL.eventually hrem] with x hx hx0
    have hx' : h (L (x - w)) = F x := by simp [h]
    rw [← hx', hx, hx0, add_zero, mul_comm]
    simp [L, coordSplit_apply]

end Transformer.Landscape.Uniformization
