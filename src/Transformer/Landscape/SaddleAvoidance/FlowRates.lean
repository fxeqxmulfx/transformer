module

public import Transformer.Landscape.SaddleAvoidance.DescentEstimates
public import Transformer.Landscape.SaddleAvoidance.FlowDecay

/-!
# Saddle avoidance — rates of convergence of descent curves and gradient flows

Let `x` descend `f` from time `0` on at the rates `c₁ > 0` and `c₂ ≥ 0`
(`IsDescentCurve`) and converge to `w`, where the Łojasiewicz inequality holds
with an exponent `a`. By the estimates of `Landscape.SaddleAvoidance.DescentEstimates`
and the decay lemmas of `Landscape.SaddleAvoidance.FlowDecay`, `f (x t) - f w` and
`‖x t - w‖` decay exponentially for `a ≤ 1/2`
(`IsDescentCurve.exists_le_mul_exp_of_isLojasiewiczAt`), and as
`(t + 1) ^ (-1/(2a - 1))` and `(t + 1) ^ (-(1 - a)/(2a - 1))` for
`1/2 < a < 1` (`IsDescentCurve.exists_le_mul_rpow_of_isLojasiewiczAt`). This
covers the flows `ẋ = -G(x) ∇f(x)` with `G` uniformly positive definite and
bounded (`isDescentCurve_of_metric`), and gradient flows
(`exists_le_mul_exp_gradientFlow_of_isLojasiewiczAt`,
`exists_le_mul_rpow_gradientFlow_of_isLojasiewiczAt`).

This is the rate part of `prop:rates` of arXiv:1602.04915 for flows, with
`t + 1` for `k`; the source states it for gradient descent
(`Landscape.SaddleAvoidance.Rates`). As there, the limit need not be a local minimizer,
`a ≤ 1/2` may be `0` or negative, and the values of `f` decay too.
-/

@[expose] public section

open Filter Metric Set
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {x : ℝ → E} {a c₁ c₂ : ℝ}

/-- **Exponential rate for descent curves** (`prop:rates` of arXiv:1602.04915
for flows, `a ≤ 1/2`). Let `x` descend `f` from time `0` on at the rates
`c₁ > 0` and `c₂ ≥ 0` and converge to `w`, where the Łojasiewicz inequality
holds with `a ≤ 1/2`. Then for some `λ > 0`, `f (x t) - f w` and `‖x t - w‖`
are at most `C e^{-λ t}`. -/
theorem IsDescentCurve.exists_le_mul_exp_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hx : IsDescentCurve f c₁ c₂ 0 x) (hc₁ : 0 < c₁) (hc₂ : 0 ≤ c₂) (ha : a ≤ 1 / 2) {w : E}
    (hw : IsLojasiewiczAt f w a) (hlim : Tendsto x atTop (𝓝 w)) :
    ∃ C lam : ℝ, 0 < lam ∧ ∀ t, 0 ≤ t →
      f (x t) - f w ≤ C * Real.exp (-(lam * t)) ∧ ‖x t - w‖ ≤ C * Real.exp (-(lam * t)) := by
  obtain ⟨hfw, hcase⟩ := hx.exists_rate_estimates hf hc₁ hc₂ (by norm_num : (1 / 2 : ℝ) < 1)
    (hw.mono ha) hlim
  obtain ⟨lam, hlam, C₁, K, hK, hC₁, hd⟩ : ∃ lam > 0, ∃ C₁ K : ℝ, 0 ≤ K ∧
      (∀ᶠ t in atTop, f (x t) - f w ≤ C₁ * Real.exp (-(lam * t))) ∧
      ∀ᶠ t in atTop, ‖x t - w‖ ≤ K * (f (x t) - f w) ^ (1 / 2 : ℝ) := by
    rcases hcase with hev | ⟨c, hc, K, hK, T, hT0, hest⟩
    · refine ⟨1, one_pos, 0, 0, le_rfl, hev.mono fun t ht => ?_, hev.mono fun t ht => ?_⟩
      · rw [ht, sub_self, zero_mul]
      · rw [ht, sub_self, norm_zero, zero_mul]
    refine ⟨c, hc, (f (x T) - f w) * Real.exp (c * T), K, hK, eventually_atTop.2 ⟨T, fun t ht =>
      le_mul_exp_of_hasDerivAt (r := fun t => f (x t) - f w)
        (fun s hs => (hx.hasDerivAt_comp (hT0.trans hs) (hf _)).sub_const (f w))
        (fun s hs => ?_) ht⟩, eventually_atTop.2 ⟨T, fun t ht => ?_⟩⟩
    · have h := (hest s hs).2.1
      rwa [show 2 * (1 / 2 : ℝ) = 1 by norm_num, Real.rpow_one] at h
    · have h := (hest t ht).2.2
      rwa [show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num] at h
  obtain ⟨hrb, hdb⟩ := hx.exists_bounds hf hc₁.le w
  obtain ⟨C, hC0, hC⟩ := exists_le_mul_and_le_mul_rpow_of_eventually
    (r := fun t => f (x t) - f w) (d := fun t => ‖x t - w‖)
    (g := fun t => Real.exp (-(lam * t))) (fun t ht => sub_nonneg.2 (hfw t ht))
    (fun _ _ => Real.exp_pos _)
    (fun _ _ _ _ hst => Real.exp_le_exp.2 (neg_le_neg (mul_le_mul_of_nonneg_left hst hlam.le)))
    hK (by norm_num) hrb hdb hC₁ hd
  refine ⟨C, lam / 2, half_pos hlam, fun t ht => ⟨?_, ?_⟩⟩
  · refine (hC t ht).1.trans (mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 ?_) hC0)
    linarith [mul_nonneg hlam.le ht]
  · have h := (hC t ht).2
    rwa [← Real.exp_mul, show -(lam * t) * (1 / 2) = -(lam / 2 * t) by ring] at h

/-- **Polynomial rate for descent curves** (`prop:rates` of arXiv:1602.04915
for flows, `1/2 < a < 1`). Let `x` descend `f` from time `0` on at the rates
`c₁ > 0` and `c₂ ≥ 0` and converge to `w`, where the Łojasiewicz inequality
holds with `1/2 < a < 1`. Then `f (x t) - f w ≤ C (t + 1) ^ (-1/(2a - 1))`
and `‖x t - w‖ ≤ C (t + 1) ^ (-(1 - a)/(2a - 1))`. -/
theorem IsDescentCurve.exists_le_mul_rpow_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hx : IsDescentCurve f c₁ c₂ 0 x) (hc₁ : 0 < c₁) (hc₂ : 0 ≤ c₂) (ha : 1 / 2 < a)
    (ha1 : a < 1) {w : E} (hw : IsLojasiewiczAt f w a) (hlim : Tendsto x atTop (𝓝 w)) :
    ∃ C : ℝ, ∀ t, 0 ≤ t → f (x t) - f w ≤ C * (t + 1) ^ (-(1 / (2 * a - 1))) ∧
      ‖x t - w‖ ≤ C * (t + 1) ^ (-((1 - a) / (2 * a - 1))) := by
  obtain ⟨hfw, hcase⟩ := hx.exists_rate_estimates hf hc₁ hc₂ ha1 hw hlim
  have hβ : 0 < 2 * a - 1 := by linarith
  obtain ⟨C₁, K, hK, hC₁, hd⟩ : ∃ C₁ K : ℝ, 0 ≤ K ∧
      (∀ᶠ t in atTop, f (x t) - f w ≤ C₁ * (t + 1) ^ (-(1 / (2 * a - 1)))) ∧
      ∀ᶠ t in atTop, ‖x t - w‖ ≤ K * (f (x t) - f w) ^ (1 - a) := by
    rcases hcase with hev | ⟨c, hc, K, hK, T, hT0, hest⟩
    · refine ⟨0, 0, le_rfl, hev.mono fun t ht => ?_, hev.mono fun t ht => ?_⟩
      · rw [ht, sub_self, zero_mul]
      · rw [ht, sub_self, norm_zero, zero_mul]
    refine ⟨((2 * a - 1) * c / 2) ^ (-(1 / (2 * a - 1))), K, hK,
      eventually_atTop.2 ⟨2 * T + 1, fun t ht => le_mul_rpow_of_hasDerivAt
        (r := fun t => f (x t) - f w) hβ hc hT0 (fun s hs => (hest s hs).1)
        (fun s hs => (hx.hasDerivAt_comp (hT0.trans hs) (hf _)).sub_const (f w))
        (fun s hs => ?_) ht⟩, eventually_atTop.2 ⟨T, fun t ht => (hest t ht).2.2⟩⟩
    rw [show 1 + (2 * a - 1) = 2 * a by ring]
    exact (hest s hs).2.1
  obtain ⟨hrb, hdb⟩ := hx.exists_bounds hf hc₁.le w
  obtain ⟨C, -, hC⟩ := exists_le_mul_and_le_mul_rpow_of_eventually
    (r := fun t => f (x t) - f w) (d := fun t => ‖x t - w‖)
    (g := fun t => (t + 1) ^ (-(1 / (2 * a - 1)))) (fun t ht => sub_nonneg.2 (hfw t ht))
    (fun t ht => Real.rpow_pos_of_pos (by linarith) _)
    (fun s hs t _ hst => Real.rpow_le_rpow_of_nonpos (by linarith [mem_Ici.1 hs]) (by linarith)
      (neg_nonpos.2 (one_div_pos.2 hβ).le)) hK (by linarith) hrb hdb hC₁ hd
  refine ⟨C, fun t ht => ⟨(hC t ht).1, ?_⟩⟩
  rw [show -((1 - a) / (2 * a - 1)) = -(1 / (2 * a - 1)) * (1 - a) by ring,
    Real.rpow_mul (by linarith)]
  exact (hC t ht).2
/-- **Exponential rate for gradient flows** (`prop:rates` of
arXiv:1602.04915 for the flow `ẋ = -∇f(x)`, `a ≤ 1/2`): a gradient flow
converging to `w`, where the Łojasiewicz inequality holds with `a ≤ 1/2`,
converges at the rate `C e^{-λ t}` for some `λ > 0`, in `f` and in norm. -/
theorem exists_le_mul_exp_gradientFlow_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t) (ha : a ≤ 1 / 2) {w : E}
    (hw : IsLojasiewiczAt f w a) (hlim : Tendsto x atTop (𝓝 w)) :
    ∃ C lam : ℝ, 0 < lam ∧ ∀ t, 0 ≤ t →
      f (x t) - f w ≤ C * Real.exp (-(lam * t)) ∧ ‖x t - w‖ ≤ C * Real.exp (-(lam * t)) :=
  (isDescentCurve_of_gradientFlow hx).exists_le_mul_exp_of_isLojasiewiczAt hf one_pos
    zero_le_one ha hw hlim

/-- **Polynomial rate for gradient flows** (`prop:rates` of arXiv:1602.04915
for the flow `ẋ = -∇f(x)`, `1/2 < a < 1`): a gradient flow converging to `w`,
where the Łojasiewicz inequality holds with `1/2 < a < 1`, converges at the
rate `C (t + 1) ^ (-1/(2a - 1))` in `f` and `C (t + 1) ^ (-(1 - a)/(2a - 1))`
in norm. -/
theorem exists_le_mul_rpow_gradientFlow_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t) (ha : 1 / 2 < a) (ha1 : a < 1)
    {w : E} (hw : IsLojasiewiczAt f w a) (hlim : Tendsto x atTop (𝓝 w)) :
    ∃ C : ℝ, ∀ t, 0 ≤ t → f (x t) - f w ≤ C * (t + 1) ^ (-(1 / (2 * a - 1))) ∧
      ‖x t - w‖ ≤ C * (t + 1) ^ (-((1 - a) / (2 * a - 1))) :=
  (isDescentCurve_of_gradientFlow hx).exists_le_mul_rpow_of_isLojasiewiczAt hf one_pos
    zero_le_one ha ha1 hw hlim

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line and its gradient flow `e^{-t}`, which descends `f` at the rates `1` and
`1` and converges to `0`: there the Łojasiewicz inequality holds with
`a = 1/2`, and so with `a = 3/4`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let x := fun t : ℝ => Real.exp (-t)
    Differentiable ℝ f ∧ (∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t) ∧
      IsDescentCurve f 1 1 0 x ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 / 2 : ℝ) ≤ 1 / 2 ∧
      (1 / 2 : ℝ) < 3 / 4 ∧ (3 / 4 : ℝ) < 1 ∧ IsLojasiewiczAt f 0 (1 / 2) ∧
      IsLojasiewiczAt f 0 (3 / 4) ∧ Tendsto x atTop (𝓝 0) := by
  intro f x
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t := fun t _ => by
    rw [hg]
    simpa [x] using (hasDerivAt_neg t).exp
  exact ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, hx,
    isDescentCurve_of_gradientFlow hx, one_pos, zero_le_one, le_rfl, by norm_num, by norm_num,
    isLojasiewiczAt_half_mul_norm_sq, isLojasiewiczAt_half_mul_norm_sq.mono (by norm_num),
    Real.tendsto_exp_neg_atTop_nhds_zero⟩

end Transformer.Landscape.SaddleAvoidance
