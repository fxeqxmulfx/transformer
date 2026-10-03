/-
# AMSGrad — genuine average-gap convergence for a fixed convex objective

arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5. The printed claim
`R(T)/T -> 0` is false for general changing online losses. For a fixed
objective and a feasible minimizer, every regret summand is nonnegative,
so the already proved upper bound supplies the missing lower half.
The original projected AMSGrad update is used without any new guard.
-/

import Transformer.AMSGrad.Section4_Corollary
import Transformer.AMSGrad.Section4_OfflineWitness

open scoped Topology
open Filter Finset

noncomputable section

namespace Transformer.AMSGrad

variable {d : ℕ}

/-- For a fixed objective and a feasible minimizer, the original AMSGrad
regret is nonnegative. This additional offline condition is precisely
what is missing from the printed general-online lower limit.
Source: arXiv:1904.03590v4, §2 and Corollary 4.5, offline specialization. -/
theorem offline_regret_nonneg {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (f : Vec d → ℝ) (hf : ∀ t, S.f t = f)
    (star : Vec d) (hmin : ∀ x ∈ F, f star ≤ f x) (T : ℕ) :
    0 ≤ S.regret amsgradRule star T := by
  apply Finset.sum_nonneg
  intro t ht
  rw [hf]
  exact sub_nonneg.mpr (hmin _ (x_mem hS amsgradRule t))

/-- Nonconstant costs satisfy the offline nonnegativity assumptions,
arXiv:1904.03590v4, §2 and Corollary 4.5, offline specialization. -/
example :
    let S := offlineLinearSetup (fun t => 1 / Real.sqrt t) (fun _ => 0) (1 / 4)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 1 ∧
      (∀ t, S.f t = fun x => 1 * x 0) ∧
      (∀ x ∈ Set.Icc (fun _ : Fin 1 => (-1 : ℝ)) (fun _ => (1 : ℝ)),
        (fun y : Vec 1 => 1 * y 0) (fun _ => -1) ≤ (fun y : Vec 1 => 1 * y 0) x) :=
  ⟨offlineLinearSetup_online _ _ _, fun _ => rfl, offlineLinearSetup_minimum.2⟩

/-- The average objective gap of original projected AMSGrad tends to
zero for the source's geometric momentum schedule and `alpha/sqrt(t)`
steps. This is an actual limit, rather than only an online limsup bound.
The added fixed-objective/minimizer conditions are explicit.
Source: arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5, offline specialization. -/
theorem offline_average_gap_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (f : Vec d → ℝ) (hf : ∀ t, S.f t = f)
    (star : Vec d) (hstar : star ∈ F) (hmin : ∀ x ∈ F, f star ≤ f x)
    {α β₁ lam : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    Tendsto (fun T : ℕ => S.regret amsgradRule star T / T) atTop (𝓝 0) := by
  apply tendsto_order.mpr
  constructor
  · intro a ha
    exact Eventually.of_forall fun T => lt_of_lt_of_le ha
      (div_nonneg (offline_regret_nonneg hS f hf star hmin T) (Nat.cast_nonneg T))
  · intro a ha
    have h := cor_lambda hS hα hαt hβ₁ hβ₁' hl hl' hb hβ₂ hβ₂' hγ
      (show 0 < a / 2 by positivity)
    filter_upwards [h] with T hT
    exact lt_of_le_of_lt (hT star hstar) (by linarith)

/-- All geometric-schedule convergence hypotheses hold on a nonconstant
objective with positive momentum and a genuine feasible minimizer.
Source: arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5. -/
example :
    let S := offlineLinearSetup (fun t => 1 / Real.sqrt t)
      (fun t => (1 / 4) * (1 / 2) ^ (t - 1)) (1 / 4)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 1 ∧
      (∀ t, S.f t = fun x => 1 * x 0) ∧
      (fun _ : Fin 1 => (-1 : ℝ)) ∈ Set.Icc (fun _ => -1) (fun _ => 1) ∧
      (∀ x ∈ Set.Icc (fun _ : Fin 1 => (-1 : ℝ)) (fun _ => (1 : ℝ)),
        (fun y : Vec 1 => 1 * y 0) (fun _ => -1) ≤ (fun y : Vec 1 => 1 * y 0) x) ∧
      (0 : ℝ) < 1 ∧ S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧
      (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
      (∀ t, S.β₁ t = (1 / 4) * (1 / 2) ^ (t - 1)) ∧
      0 < S.β₂ ∧ S.β₂ < 1 ∧ (1 / 4) / Real.sqrt S.β₂ < 1 := by
  refine ⟨offlineLinearSetup_online _ _ _, fun _ => rfl,
    offlineLinearSetup_minimum.1, offlineLinearSetup_minimum.2,
    by norm_num, rfl, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => rfl, ?_⟩
  have hs : Real.sqrt (4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num [offlineLinearSetup, hs]

/-- Actual average-gap convergence also holds for the source's inverse
momentum schedule. Source: arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5,
offline specialization with an explicitly fixed objective and minimizer. -/
theorem offline_average_gap_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (f : Vec d → ℝ) (hf : ∀ t, S.f t = f)
    (star : Vec d) (hstar : star ∈ F) (hmin : ∀ x ∈ F, f star ≤ f x)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    Tendsto (fun T : ℕ => S.regret amsgradRule star T / T) atTop (𝓝 0) := by
  apply tendsto_order.mpr
  constructor
  · intro a ha
    exact Eventually.of_forall fun T => lt_of_lt_of_le ha
      (div_nonneg (offline_regret_nonneg hS f hf star hmin T) (Nat.cast_nonneg T))
  · intro a ha
    have h := cor_inv hS hα hαt hβ₁ hβ₁' hb hβ₂ hβ₂' hγ (show 0 < a / 2 by positivity)
    filter_upwards [h] with T hT
    exact lt_of_le_of_lt (hT star hstar) (by linarith)

/-- Inverse-schedule hypotheses hold on the same nonconstant objective,
arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5, offline specialization. -/
example :
    let S := offlineLinearSetup (fun t => 1 / Real.sqrt t) (fun t => (1 / 4) / t) (1 / 4)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 1 ∧
      (∀ t, S.f t = fun x => 1 * x 0) ∧
      (fun _ : Fin 1 => (-1 : ℝ)) ∈ Set.Icc (fun _ => -1) (fun _ => 1) ∧
      (∀ x ∈ Set.Icc (fun _ : Fin 1 => (-1 : ℝ)) (fun _ => (1 : ℝ)),
        (fun y : Vec 1 => 1 * y 0) (fun _ => -1) ≤ (fun y : Vec 1 => 1 * y 0) x) ∧
      (0 : ℝ) < 1 ∧ S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧
      (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) < 1 ∧ (∀ t, S.β₁ t = (1 / 4) / t) ∧
      0 < S.β₂ ∧ S.β₂ < 1 ∧ (1 / 4) / Real.sqrt S.β₂ < 1 := by
  refine ⟨offlineLinearSetup_online _ _ _, fun _ => rfl,
    offlineLinearSetup_minimum.1, offlineLinearSetup_minimum.2,
    by norm_num, rfl, by norm_num, by norm_num, fun _ => rfl, ?_⟩
  have hs : Real.sqrt (4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num [offlineLinearSetup, hs]

end Transformer.AMSGrad
