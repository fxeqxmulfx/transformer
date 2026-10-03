import Transformer.AMSGrad.Section4_Terms

/-
# AMSGrad — the third term of Lemma 3.1

§4 of arXiv:1904.03590v4, proof of Theorem 4.1: the bounds on the third term
of Lemma 3.1 for `β_{1,t} = β₁λ^{t-1}`, (beta_1lambda^{t-1}), and for
`β_{1,t} = β₁/t`, (frac{beta_1}{t}).

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- **(eqthird), `β_{1,t} = β₁λ^{t-1}`.**  Under AMSGrad with `α_t = α/√t`,
`β₁ < 1`, `0 ≤ λ < 1` and `0 ≤ β₂ ≤ 1`, for `x* ∈ F`,

  `Σᵢ Σ_{t=2}^T β_{1,t}√v̂_{t-1,i}/(2α_{t-1}(1-β₁)) (x_{t,i} - x*ᵢ)² ≤ dD²G/(2α(1-β₁)(1-λ)²)`.

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1, (beta_1lambda^{t-1}). -/
theorem eqthird_lambda_le {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α β₁ lam : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁' : β₁ < 1) (hl : 0 ≤ lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1)) (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (T : ℕ)
    {xstar : Vec d} (hx : xstar ∈ F) :
    ∑ i, ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule (t - 1) i) /
        (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x amsgradRule t i - xstar i) ^ 2 ≤
      d * D ^ 2 * G / (2 * α * (1 - β₁) * (1 - lam) ^ 2) := by
  have hB : 0 < 1 - β₁ := by linarith
  have hL : 0 < 1 - lam := by linarith
  rw [show S.β₁ 1 = β₁ by rw [hb]; simp]
  set B := 1 - β₁
  set L := 1 - lam
  calc _ ≤ ∑ _i : Fin d, D ^ 2 * G / (2 * α * B) * (1 / L ^ 2) := sum_le_sum fun i _ => ?_
    _ = _ := by
      rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
      simp only [div_eq_mul_inv, mul_inv]
      ring
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  refine le_trans (sum_le_sum fun t ht => ?_)
    ((mul_sum _ _ _).symm.le.trans (mul_le_mul_of_nonneg_left (sum_sqrt_mul_pow_le hl hl' T)
      (div_nonneg (mul_nonneg (sq_nonneg D) hG) (by positivity))))
  obtain ⟨s, rfl⟩ : ∃ s, t = s + 1 := ⟨t - 1, by have := (mem_Icc.1 ht).1; omega⟩
  have hs : 0 < Real.sqrt (s : ℝ) :=
    Real.sqrt_pos.2 (Nat.cast_pos.2 (by have := (mem_Icc.1 ht).1; omega))
  simp only [hb, hαt, Nat.add_sub_cancel]
  have he : (S.x amsgradRule (s + 1) i - xstar i) ^ 2 ≤ D ^ 2 := by
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS amsgradRule _) _ hx i) 2
  have h1 : β₁ * lam ^ s ≤ lam ^ s := mul_le_of_le_one_left (pow_nonneg hl s) hβ₁'.le
  have h2 := mul_le_mul (vt hS hβ₂ hβ₂' s i) he (sq_nonneg _) hG
  have e1 : β₁ * lam ^ s * Real.sqrt (S.vhat amsgradRule s i) / (2 * (α / Real.sqrt s) * B) *
      (S.x amsgradRule (s + 1) i - xstar i) ^ 2 =
      β₁ * lam ^ s * (Real.sqrt (S.vhat amsgradRule s i) *
        (S.x amsgradRule (s + 1) i - xstar i) ^ 2) * (Real.sqrt s / (2 * α * B)) := by
    field_simp
  have e2 : D ^ 2 * G / (2 * α * B) * (Real.sqrt s * lam ^ s) =
      lam ^ s * (G * D ^ 2) * (Real.sqrt s / (2 * α * B)) := by
    ring
  rw [e1, e2]
  exact mul_le_mul_of_nonneg_right (mul_le_mul h1 h2 (by positivity) (pow_nonneg hl s))
    (by positivity)

/-- **(eqthird), `β_{1,t} = β₁/t`.**  Under AMSGrad with `α_t = α/√t`,
`β₁ < 1` and `0 ≤ β₂ ≤ 1`, for `x* ∈ F`,

  `Σᵢ Σ_{t=2}^T β_{1,t}√v̂_{t-1,i}/(2α_{t-1}(1-β₁)) (x_{t,i} - x*ᵢ)² ≤ dD²G√T/(α(1-β₁))`.

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1, (frac{beta_1}{t}). -/
theorem eqthird_inv_le {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α β₁ : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁' : β₁ < 1)
    (hb : ∀ t, S.β₁ t = β₁ / t) (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (T : ℕ)
    {xstar : Vec d} (hx : xstar ∈ F) :
    ∑ i, ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule (t - 1) i) /
        (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x amsgradRule t i - xstar i) ^ 2 ≤
      d * D ^ 2 * G * Real.sqrt T / (α * (1 - β₁)) := by
  have hB : 0 < 1 - β₁ := by linarith
  rw [show S.β₁ 1 = β₁ by rw [hb]; simp]
  set B := 1 - β₁
  calc _ ≤ ∑ _i : Fin d, D ^ 2 * G / (2 * α * B) * (2 * Real.sqrt T) :=
        sum_le_sum fun i _ => ?_
    _ = _ := by
      rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
      simp only [div_eq_mul_inv, mul_inv]
      ring
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  have hK : 0 ≤ D ^ 2 * G / (2 * α * B) :=
    div_nonneg (mul_nonneg (sq_nonneg D) hG) (by positivity)
  have hsum : ∑ t ∈ Icc 2 T, 1 / Real.sqrt (t : ℝ) ≤ 2 * Real.sqrt T :=
    (sum_le_sum_of_subset_of_nonneg (Icc_subset_Icc_left (by norm_num))
      fun _ _ _ => by positivity).trans (sum_inv_sqrt_le T)
  refine le_trans (sum_le_sum fun t ht => ?_)
    ((mul_sum _ _ _).symm.le.trans (mul_le_mul_of_nonneg_left hsum hK))
  obtain ⟨s, rfl⟩ : ∃ s, t = s + 1 := ⟨t - 1, by have := (mem_Icc.1 ht).1; omega⟩
  have hs0 : (1 : ℝ) ≤ s := Nat.one_le_cast.2 (by have := (mem_Icc.1 ht).1; omega)
  have hs : 0 < Real.sqrt (s : ℝ) := Real.sqrt_pos.2 (by linarith)
  simp only [hb, hαt, Nat.add_sub_cancel]
  push_cast
  have he : (S.x amsgradRule (s + 1) i - xstar i) ^ 2 ≤ D ^ 2 := by
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS amsgradRule _) _ hx i) 2
  have h2 := mul_le_mul (vt hS hβ₂ hβ₂' s i) he (sq_nonneg _) hG
  have h3 : β₁ * (Real.sqrt (S.vhat amsgradRule s i) *
      (S.x amsgradRule (s + 1) i - xstar i) ^ 2) ≤ G * D ^ 2 := by
    nlinarith [mul_nonneg (Real.sqrt_nonneg (S.vhat amsgradRule s i))
      (sq_nonneg (S.x amsgradRule (s + 1) i - xstar i))]
  have hs1 : 0 < Real.sqrt ((s : ℝ) + 1) := Real.sqrt_pos.2 (by linarith)
  have h4 : Real.sqrt s * Real.sqrt ((s : ℝ) + 1) ≤ (s : ℝ) + 1 := by
    have := Real.sqrt_le_sqrt (by linarith : (s : ℝ) ≤ s + 1)
    nlinarith [Real.sq_sqrt (by linarith : (0 : ℝ) ≤ s + 1)]
  have h5 : Real.sqrt s / (((s : ℝ) + 1) * (2 * α * B)) ≤
      1 / (Real.sqrt ((s : ℝ) + 1) * (2 * α * B)) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_le_mul_of_nonneg_right h4 (by positivity : (0 : ℝ) ≤ 2 * α * B)]
  have e1 : β₁ / ((s : ℝ) + 1) * Real.sqrt (S.vhat amsgradRule s i) /
      (2 * (α / Real.sqrt s) * B) * (S.x amsgradRule (s + 1) i - xstar i) ^ 2 =
      β₁ * (Real.sqrt (S.vhat amsgradRule s i) * (S.x amsgradRule (s + 1) i - xstar i) ^ 2) *
        (Real.sqrt s / (((s : ℝ) + 1) * (2 * α * B))) := by
    field_simp
  have e2 : D ^ 2 * G / (2 * α * B) * (1 / Real.sqrt ((s : ℝ) + 1)) =
      G * D ^ 2 * (1 / (Real.sqrt ((s : ℝ) + 1) * (2 * α * B))) := by
    field_simp
  rw [e1, e2]
  exact mul_le_mul h3 h5 (by positivity) (by positivity)

/-- The hypotheses of `eqthird_lambda_le` and `eqthird_inv_le` are satisfiable:
the zero cost on `[-1, 1]`, `α = 1`, `β₁ = 0`, `λ = 1/2`, `β₂ = 1/2`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧
      (1 / 2 : ℝ) < 1 ∧ (∀ t, S.β₁ t = 0 * (1 / 2) ^ (t - 1)) ∧ (∀ t, S.β₁ t = 0 / t) ∧
      0 ≤ S.β₂ ∧ S.β₂ ≤ 1 ∧ (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, one_pos, by norm_num, by norm_num,
    fun _ => by simp [zeroSetup], fun _ => by simp [zeroSetup], by norm_num [zeroSetup],
    by norm_num [zeroSetup], ⟨fun _ => by norm_num, fun _ => by norm_num⟩⟩

end AMSGrad
end Transformer
