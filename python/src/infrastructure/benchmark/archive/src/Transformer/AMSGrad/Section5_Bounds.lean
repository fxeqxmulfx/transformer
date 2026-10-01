import Transformer.AMSGrad.Section5_AdamX
import Transformer.AMSGrad.Section4_Telescope
import Transformer.AMSGrad.Section4_Terms

/-
# AdamX — the terms of Lemma 3.1

§5 of arXiv:1904.03590v4, proof of Theorem 5.1: the facts about AdamX that
bound the first and third terms of Lemma 3.1.

* `vtnew_div`: `√v̂_t/(1-β_{1,t}) ≤ G/(1-β₁)`, from Lemma 5.2.  This is what
  lets the first term telescope to `dD²G√T/(2α(1-β₁))`.
* `adamX_mono`: `√(t v̂_t)/(1-β_{1,t})` does not decrease, from
  `v̂_t ≥ (1-β_{1,t})²/(1-β_{1,t-1})² v̂_{t-1}`: the `t₀` of `eqmain_le_of` is `0`.
* `eqthird_adamX_le`: the third term, from Lemma 5.3.

Source: arXiv:1904.03590v4, §5, proof of Theorem 5.1.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- Under AdamX with `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1` and `0 ≤ β₂ ≤ 1`,
`√v̂_t/(1-β_{1,t}) ≤ G/(1-β₁)` for `t ≥ 1`.
Source: arXiv:1904.03590v4, §5, proof of Theorem 5.1, from Lemma 5.2. -/
theorem vtnew_div {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (t : ℕ) (ht : 1 ≤ t) (i : Fin d) :
    Real.sqrt (S.vhat (adamXRule S.β₁) t i) / (1 - S.β₁ t) ≤ G / (1 - S.β₁ 1) := by
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  obtain ⟨s, hs, e⟩ := (vtnew (fun t ht => (lt_of_le_of_lt (hβ₁ t ht).2 hβ₁').ne) ht i).2
  have hs1 : 1 ≤ s := (mem_Icc.mp hs).1
  have hbt : 0 < 1 - S.β₁ t := by linarith [(hβ₁ t ht).2]
  have hbs : 0 < 1 - S.β₁ s := by linarith [(hβ₁ s hs1).2]
  have hv := v_le hS hβ₂ hβ₂' (adamXRule S.β₁) s i
  have hv0 : 0 ≤ S.v (adamXRule S.β₁) s i := hv.1
  have hsv : Real.sqrt (S.v (adamXRule S.β₁) s i) ≤ G := by
    rw [← Real.sqrt_sq hG]; exact Real.sqrt_le_sqrt hv.2
  rw [e, Real.sqrt_mul' _ hv0, Real.sqrt_sq (div_nonneg hbt.le hbs.le),
    show (1 - S.β₁ t) / (1 - S.β₁ s) * Real.sqrt (S.v (adamXRule S.β₁) s i) / (1 - S.β₁ t) =
      Real.sqrt (S.v (adamXRule S.β₁) s i) / (1 - S.β₁ s) by field_simp]
  exact div_le_div₀ hG hsv (by linarith) (by linarith [(hβ₁ s hs1).2])

/-- Under AdamX with `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1` and `0 ≤ β₂ ≤ 1`,
`√((t-1)v̂_{t-1})/(1-β_{1,t-1}) ≤ √(t v̂_t)/(1-β_{1,t})` for every `t ≥ 1`.
Source: arXiv:1904.03590v4, §5, proof of Theorem 5.1. -/
theorem adamX_mono {S : Setup d} (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1)
    (hβ₁' : S.β₁ 1 < 1) (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (t : ℕ) (ht : 0 < t) (i : Fin d) :
    Real.sqrt (((t - 1 : ℕ) : ℝ) * S.vhat (adamXRule S.β₁) (t - 1) i) / (1 - S.β₁ (t - 1)) ≤
      Real.sqrt ((t : ℝ) * S.vhat (adamXRule S.β₁) t i) / (1 - S.β₁ t) := by
  have hb : ∀ t, 1 ≤ t → 0 < 1 - S.β₁ t := fun t ht => by linarith [(hβ₁ t ht).2]
  have hvn : ∀ n, 0 ≤ S.vhat (adamXRule S.β₁) n i := fun n => by
    cases n with
    | zero => exact le_rfl
    | succ n =>
      exact (v_nonneg hβ₂ hβ₂' (adamXRule S.β₁) (n + 1) i).trans
        (le_adamXRule S.β₁ (n + 1) (S.vhat (adamXRule S.β₁) n) (S.v (adamXRule S.β₁) (n + 1)) i)
  obtain ⟨s, rfl⟩ : ∃ s, t = s + 1 := ⟨t - 1, by omega⟩
  rw [Nat.add_sub_cancel]
  rcases Nat.eq_zero_or_pos s with rfl | hs
  · simp only [Nat.cast_zero, zero_mul, Real.sqrt_zero, zero_div]
    exact div_nonneg (Real.sqrt_nonneg _) (hb _ le_rfl).le
  set q := (1 - S.β₁ (s + 1)) / (1 - S.β₁ s) with hq
  have hq0 : 0 ≤ q := div_nonneg (hb _ (by omega)).le (hb s hs).le
  have h1 : q ^ 2 * S.vhat (adamXRule S.β₁) s i ≤ S.vhat (adamXRule S.β₁) (s + 1) i := by
    rw [adamX_vhat_succ S hs i]; exact le_max_left _ _
  have e : Real.sqrt ((s : ℝ) * S.vhat (adamXRule S.β₁) s i) / (1 - S.β₁ s) =
      Real.sqrt ((s : ℝ) * (q ^ 2 * S.vhat (adamXRule S.β₁) s i)) / (1 - S.β₁ (s + 1)) := by
    have := hb (s + 1) (by omega)
    have := hb s hs
    rw [show (s : ℝ) * (q ^ 2 * S.vhat (adamXRule S.β₁) s i) =
        q ^ 2 * ((s : ℝ) * S.vhat (adamXRule S.β₁) s i) by ring,
      Real.sqrt_mul (sq_nonneg q), Real.sqrt_sq hq0, hq]
    field_simp
  rw [e]
  refine div_le_div_of_nonneg_right (Real.sqrt_le_sqrt (mul_le_mul (by push_cast; linarith) h1
    (mul_nonneg (sq_nonneg q) (hvn s)) (Nat.cast_nonneg _))) (hb _ (by omega)).le

/-- **The third term, AdamX.**  With `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`
and `0 ≤ β₂ ≤ 1`, for `x* ∈ F`,

  `Σᵢ Σ_{t=2}^T β_{1,t}√v̂_{t-1,i}/(2α_{t-1}(1-β₁)) (x_{t,i} - x*ᵢ)²`
  `    ≤ dD²G/(2α(1-β₁)²) Σ_{t=2}^T β_{1,t}√(t-1)`.

Source: arXiv:1904.03590v4, §5, proof of Theorem 5.1, from Lemma 5.3. -/
theorem eqthird_adamX_le {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (T : ℕ) {xstar : Vec d} (hx : xstar ∈ F) :
    ∑ i, ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt (S.vhat (adamXRule S.β₁) (t - 1) i) /
        (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x (adamXRule S.β₁) t i - xstar i) ^ 2 ≤
      d * D ^ 2 * G / (2 * α * (1 - S.β₁ 1) ^ 2) *
        ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1) := by
  have hB : 0 < 1 - S.β₁ 1 := by linarith
  calc _ ≤ ∑ _i : Fin d, D ^ 2 * G / (2 * α * (1 - S.β₁ 1) ^ 2) *
        ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1) := sum_le_sum fun i _ => ?_
    _ = _ := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  rw [mul_sum]
  refine sum_le_sum fun t ht => ?_
  obtain ⟨s, rfl⟩ : ∃ s, t = s + 1 := ⟨t - 1, by have := (mem_Icc.1 ht).1; omega⟩
  have hs : 0 < Real.sqrt (s : ℝ) :=
    Real.sqrt_pos.2 (Nat.cast_pos.2 (by have := (mem_Icc.1 ht).1; omega))
  have hβt := (hβ₁ (s + 1) (by omega)).1
  have he : (S.x (adamXRule S.β₁) (s + 1) i - xstar i) ^ 2 ≤ D ^ 2 := by
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS _ _) _ hx i) 2
  have h2 := mul_le_mul (vt2 hS hβ₁ hβ₁' hβ₂ hβ₂' s i) he (sq_nonneg _)
    (div_nonneg hG hB.le)
  simp only [hαt, Nat.add_sub_cancel]
  rw [show (((s + 1 : ℕ) : ℝ) - 1) = s by push_cast; ring]
  have e1 : S.β₁ (s + 1) * Real.sqrt (S.vhat (adamXRule S.β₁) s i) /
      (2 * (α / Real.sqrt s) * (1 - S.β₁ 1)) * (S.x (adamXRule S.β₁) (s + 1) i - xstar i) ^ 2 =
      S.β₁ (s + 1) * Real.sqrt s / (2 * α * (1 - S.β₁ 1)) *
        (Real.sqrt (S.vhat (adamXRule S.β₁) s i) *
          (S.x (adamXRule S.β₁) (s + 1) i - xstar i) ^ 2) := by
    field_simp
  have e2 : D ^ 2 * G / (2 * α * (1 - S.β₁ 1) ^ 2) * (S.β₁ (s + 1) * Real.sqrt s) =
      S.β₁ (s + 1) * Real.sqrt s / (2 * α * (1 - S.β₁ 1)) * (G / (1 - S.β₁ 1) * D ^ 2) := by
    field_simp
  rw [e1, e2]
  exact mul_le_mul_of_nonneg_left h2 (by positivity)

/-- The hypotheses of `vtnew_div`, `adamX_mono` and `eqthird_adamX_le` are
satisfiable: the zero cost on `[-1, 1]`, `α = 1`, `β_{1,t} = 0`, `β₂ = 1/2`,
`x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 ≤ S.β₂ ∧ S.β₂ ≤ 1 ∧ (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    ⟨fun _ => by norm_num, fun _ => by norm_num⟩⟩

end AMSGrad
end Transformer
