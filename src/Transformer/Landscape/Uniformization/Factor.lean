/-
# Local uniformization: factors of functions with normal crossings

If `F * H` has normal crossings at `w` and `F`, `H` are analytic there, then
`F` has normal crossings at `w`. By induction on the degree of the monomial:
the product vanishes on a hyperplane `x i = w i` of the monomial, so one
factor does; it is divisible by `x i - w i`, which then cancels. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Landscape.Uniformization.Domain

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base)

variable {n : ℕ}

/-- A coordinate `x i - w i` has normal crossings at `w`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem normalCrossingAt_coord (i : Fin n) (w : Base n) :
    NormalCrossingAt (fun x => x i - w i) w :=
  ⟨Pi.single i 1, fun _ => 1, analyticAt_const, one_ne_zero,
    Eventually.of_forall fun x => by rw [monomial_single, one_mul, Pi.sub_apply]⟩

/-- A monomial whose exponent is positive at `i` splits off the coordinate
`i`. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem monomial_eq_coord_mul {C : Fin n → ℕ} {i : Fin n} (hi : C i ≠ 0) (y : Base n) :
    monomial C y = y i * monomial (C - Pi.single i 1) y := by
  have hC : C = (C - Pi.single i 1) + Pi.single i 1 := by
    funext j
    by_cases hj : j = i
    · subst hj
      simp only [Pi.add_apply, Pi.sub_apply, Pi.single_eq_same]
      omega
    · simp [hj]
  conv_lhs => rw [hC]
  rw [monomial_add_exp, monomial_single, mul_comm]

theorem sum_sub_single {C : Fin n → ℕ} {i : Fin n} (hi : C i ≠ 0) :
    ∑ j, (C - Pi.single i 1 : Fin n → ℕ) j + 1 = ∑ j, C j := by
  rw [← Finset.sum_erase_add _ _ (Finset.mem_univ i),
    ← Finset.sum_erase_add _ _ (Finset.mem_univ i), add_assoc]
  congr 1
  · refine Finset.sum_congr rfl fun j hj => ?_
    simp [Finset.ne_of_mem_erase hj]
  · simp only [Pi.sub_apply, Pi.single_eq_same]
    omega

theorem normalCrossingAt_of_mul_aux {w : Base n} (k : ℕ) :
    ∀ (F H u : Base n → ℝ) (C : Fin n → ℕ), ∑ j, C j = k → AnalyticAt ℝ F w →
      AnalyticAt ℝ H w → AnalyticAt ℝ u w → u w ≠ 0 →
      (∀ᶠ x in 𝓝 w, F x * H x = u x * monomial C (x - w)) → NormalCrossingAt F w := by
  induction k with
  | zero =>
    intro F H u C hC hF hH hu huw h
    have hC0 : C = 0 := funext fun j => Finset.sum_eq_zero_iff.1 hC j (Finset.mem_univ j)
    subst hC0
    have hw := h.self_of_nhds
    rw [monomial_zero_exp, mul_one] at hw
    exact normalCrossingAt_of_ne_zero hF (left_ne_zero_of_mul (hw ▸ huw))
  | succ k ih =>
    intro F H u C hC hF hH hu huw h
    obtain ⟨i, hi⟩ : ∃ i, C i ≠ 0 := by
      by_contra hne
      push Not at hne
      simp [hne] at hC
    have hk : ∑ j, (C - Pi.single i 1 : Fin n → ℕ) j = k := by
      have := sum_sub_single hi
      omega
    let ι : Base n → Base n := fun x => Function.update x i (w i)
    have hιw : ι w = w := Function.update_eq_self i w
    have hιA : AnalyticAt ℝ ι w := analyticAt_pi_iff.2 fun j => by
      by_cases hj : j = i
      · subst hj
        simp only [Function.update_self]
        exact analyticAt_const
      · simp only [Function.update_of_ne hj]
        exact (ContinuousLinearMap.proj j : Base n →L[ℝ] ℝ).analyticAt w
    have hιt : Tendsto ι (𝓝 w) (𝓝 w) := by
      have := hιA.continuousAt.tendsto
      rwa [hιw] at this
    have hF' : AnalyticAt ℝ F (ι w) := by rwa [hιw]
    have hH' : AnalyticAt ℝ H (ι w) := by rwa [hιw]
    have hprod : (fun x => F (ι x) * H (ι x)) =ᶠ[𝓝 w] 0 := by
      filter_upwards [hιt.eventually h] with x hx
      rw [hx, Pi.zero_apply]
      have : monomial C (ι x - w) = 0 := by
        by_contra hne
        exact (monomial_ne_zero_iff C _).1 hne i hi (by simp [ι])
      rw [this, mul_zero]
    have hsplit : ∀ x, u x * monomial C (x - w) =
        (x i - w i) * (u x * monomial (C - Pi.single i 1) (x - w)) := fun x => by
      rw [monomial_eq_coord_mul hi, Pi.sub_apply]
      ring
    have hrest : AnalyticAt ℝ (fun x => u x * monomial (C - Pi.single i 1) (x - w)) w :=
      hu.fun_mul (analyticAt_monomial_sub _ w w)
    rcases eventuallyEq_zero_or_of_mul (hF'.comp hιA) (hH'.comp hιA) hprod with h0 | h0
    · obtain ⟨G, hG, hFG⟩ := exists_eq_coord_mul i hF h0
      have hGH : (fun x => G x * H x) =ᶠ[𝓝 w]
          fun x => u x * monomial (C - Pi.single i 1) (x - w) := by
        refine eventuallyEq_of_coord_mul i (hG.fun_mul hH) hrest ?_
        filter_upwards [h, hFG] with x hx hFx
        rw [← hsplit, ← hx, hFx]
        ring
      exact ((normalCrossingAt_coord i w).mul (ih G H u _ hk hG hH hu huw hGH)).congr
        (hFG.mono fun x hx => hx.symm)
    · obtain ⟨G, hG, hHG⟩ := exists_eq_coord_mul i hH h0
      have hFG : (fun x => F x * G x) =ᶠ[𝓝 w]
          fun x => u x * monomial (C - Pi.single i 1) (x - w) := by
        refine eventuallyEq_of_coord_mul i (hF.fun_mul hG) hrest ?_
        filter_upwards [h, hHG] with x hx hHx
        rw [← hsplit, ← hx, hHx]
        ring
      exact ih F G u _ hk hF hG hu huw hFG

/-- A factor of a function with normal crossings has normal crossings.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem NormalCrossingAt.of_mul {F H : Base n → ℝ} {w : Base n} (hF : AnalyticAt ℝ F w)
    (hH : AnalyticAt ℝ H w) (h : NormalCrossingAt (fun x => F x * H x) w) :
    NormalCrossingAt F w := by
  obtain ⟨C, u, hu, huw, hFH⟩ := h
  exact normalCrossingAt_of_mul_aux _ F H u C rfl hF hH hu huw hFH

/-- Every factor of a finite product with normal crossings has normal
crossings. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem NormalCrossingAt.of_prod {ι : Type*} (s : Finset ι) {F : ι → Base n → ℝ}
    {w : Base n} (hF : ∀ i ∈ s, AnalyticAt ℝ (F i) w)
    (h : NormalCrossingAt (fun x => ∏ i ∈ s, F i x) w) {i : ι} (hi : i ∈ s) :
    NormalCrossingAt (F i) w := by
  classical
  have hs : (fun x => ∏ j ∈ s, F j x) = fun x => F i x * ∏ j ∈ s.erase i, F j x := by
    funext x
    rw [Finset.mul_prod_erase s (fun j => F j x) hi]
  rw [hs] at h
  exact h.of_mul (hF i hi)
    (Finset.analyticAt_fun_prod _ fun j hj => hF j (Finset.mem_of_mem_erase hj))

end Transformer.Landscape.Uniformization
