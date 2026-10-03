/-
# AMSGrad — convergence of the original algorithm's averaged losses

arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5, specialized to a
fixed convex objective and feasible minimizer. Both source momentum
schedules give an actual minimum-loss limit for averaged parameters.
The strict gamma condition corrects the source's division by `1-gamma`.
-/

import Transformer.AMSGrad.Section4_OfflineAverage

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGrad

variable {d : ℕ}

/-- The original projected AMSGrad algorithm with geometric momentum and
`alpha/sqrt(t)` steps converges in the loss of its averaged parameters
to the fixed convex objective's feasible minimum. Source:
arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5, offline specialization
with a strictly subunit gamma and an explicit fixed-objective condition. -/
theorem offline_average_loss_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (f : Vec d → ℝ) (hf : ∀ t, S.f t = f)
    (star : Vec d) (hstar : star ∈ F) (hmin : ∀ x ∈ F, f star ≤ f x)
    {α β₁ lam : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    Tendsto (fun T : ℕ => f (offlineAverage S T)) atTop (𝓝 (f star)) :=
  offline_average_loss_of_regret_limit hS f hf star hmin
    (offline_average_gap_lambda hS f hf star hstar hmin hα hαt
      hβ₁ hβ₁' hl hl' hb hβ₂ hβ₂' hγ)

/-- The complete geometric-schedule theorem applies to a nonconstant
loss, nonzero momentum and a genuine feasible minimizer. Source:
arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5, offline specialization. -/
example :
    let S := offlineLinearSetup (fun t => 1 / Real.sqrt t)
      (fun t => (1 / 4) * (1 / 2) ^ (t - 1)) (1 / 4)
    Tendsto (fun T : ℕ => (fun x : Vec 1 => 1 * x 0) (offlineAverage S T))
      atTop (𝓝 ((fun x : Vec 1 => 1 * x 0) (fun _ => -1))) := by
  dsimp only
  apply offline_average_loss_lambda (offlineLinearSetup_online _ _ _)
    (fun x : Vec 1 => 1 * x 0) (fun _ => rfl) (fun _ => -1)
    offlineLinearSetup_minimum.1 offlineLinearSetup_minimum.2
    (α := 1) (β₁ := 1 / 4) (lam := 1 / 2) (by norm_num) rfl
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (fun _ => rfl)
    (by norm_num [offlineLinearSetup]) (by norm_num [offlineLinearSetup])
  have hs : Real.sqrt (4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num [offlineLinearSetup, hs]

/-- The original projected algorithm also attains the fixed convex
objective's feasible minimum in the loss of its averaged parameters with
inverse momentum. Source: arXiv:1904.03590v4, Theorem 4.1 and Corollary 4.5,
offline specialization with the same strict gamma correction. -/
theorem offline_average_loss_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (f : Vec d → ℝ) (hf : ∀ t, S.f t = f)
    (star : Vec d) (hstar : star ∈ F) (hmin : ∀ x ∈ F, f star ≤ f x)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    Tendsto (fun T : ℕ => f (offlineAverage S T)) atTop (𝓝 (f star)) :=
  offline_average_loss_of_regret_limit hS f hf star hmin
    (offline_average_gap_inv hS f hf star hstar hmin hα hαt hβ₁ hβ₁' hb hβ₂ hβ₂' hγ)

/-- The inverse-schedule theorem also applies to a nonconstant objective
with positive momentum. Source: arXiv:1904.03590v4, Theorem 4.1 and
Corollary 4.5, offline specialization. -/
example :
    let S := offlineLinearSetup (fun t => 1 / Real.sqrt t) (fun t => (1 / 4) / t) (1 / 4)
    Tendsto (fun T : ℕ => (fun x : Vec 1 => 1 * x 0) (offlineAverage S T))
      atTop (𝓝 ((fun x : Vec 1 => 1 * x 0) (fun _ => -1))) := by
  dsimp only
  apply offline_average_loss_inv (offlineLinearSetup_online _ _ _)
    (fun x : Vec 1 => 1 * x 0) (fun _ => rfl) (fun _ => -1)
    offlineLinearSetup_minimum.1 offlineLinearSetup_minimum.2
    (α := 1) (β₁ := 1 / 4) (by norm_num) rfl
    (by norm_num) (by norm_num) (fun _ => rfl)
    (by norm_num [offlineLinearSetup]) (by norm_num [offlineLinearSetup])
  have hs : Real.sqrt (4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num [offlineLinearSetup, hs]

end Transformer.AMSGrad
