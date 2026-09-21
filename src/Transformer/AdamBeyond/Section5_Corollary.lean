import Transformer.AdamBeyond.Section5_Regret
import Transformer.AdamBeyond.Section4_Corollary
import Transformer.AMSGrad.Section4_Rate

/-
# Adam and beyond — §5: Corollary 2

Corollary 2 of arXiv:1904.09237 and the remark that `β_{1,t} = β₁/t` still
gives `O(√T)`, both from Theorem 5 (`adamNC_regret`, `Section5_Regret`).

**What the source says and what is carried here.**

* Corollary 2 is missing its left-hand side "`R_T ≤`", restored here.  Its
  second term is `β₁D²G/((1-β₁)²(1-λ)²)`; Theorem 5 gives it with the factor
  `d/α`, as for Corollary 1, restored here.  Its `ζ` is left implicit; the
  setting satisfies both conditions with `ζ = α` (`adamNC_inv_cond`), which is
  the `ζ` stated.  With `v_{T,i} = ‖g_{1:T,i}‖²/T` the first term of
  Theorem 5 is `D²/(2α(1-β₁)) Σᵢ ‖g_{1:T,i}‖₂`, as the source writes.

* The remark "one can use `β_{1t} = β₁/t` and still ensure a data-dependent
  regret of `O(√T)`" is stated as `R_T ≤ K√T` for every `T` and `x* ∈ F`.

Not transcribed: "it is easy to generalize this result for similar settings
of `β_{2t}`", which names no statement.

Source: arXiv:1904.09237, §5, Corollary 2 and the paragraph after it.
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- With `β_{2t} = 1 - 1/t`, `√T √v_{T,i} = ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.09237, §5, the sentence after Corollary 2. -/
theorem sqrt_mul_sqrt_vhat_inv {S : Setup d} (hβ₂ : S.β₂ = 0) (T : ℕ) (i : Fin d) :
    Real.sqrt T * Real.sqrt (S.vhat (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) T i) =
      S.gnorm (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) T i := by
  rw [adamNC_vhat_inv hβ₂, ← Real.sqrt_mul (Nat.cast_nonneg _), Setup.gnorm]
  rcases Nat.eq_zero_or_pos T with rfl | hT
  · simp
  · have : (T : ℝ) ≠ 0 := by positivity
    congr 1
    field_simp

/-- With `β_{2t} = 1 - 1/t`, `√v_{t,i} ≤ G∞`: `v_{t,i}` is the mean of
`g²_{j,i}`, `j ≤ t`.

Source: arXiv:1904.09237, §5, Corollary 2, from its proof by Theorem 5. -/
theorem sqrt_vhat_inv_le {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (hβ₂ : S.β₂ = 0) (t : ℕ) (i : Fin d) :
    Real.sqrt (S.vhat (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) t i) ≤ G := by
  rcases Nat.eq_zero_or_pos t with rfl | ht
  · rw [adamNC_vhat_inv hβ₂]
    simpa using (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  · have h := gnorm_le hS (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) t i
    rw [← sqrt_mul_sqrt_vhat_inv hβ₂, mul_comm G] at h
    exact le_of_mul_le_mul_left h (Real.sqrt_pos.2 (by exact_mod_cast ht))

/-- **Corollary 2**, with "`R_T ≤`" and the factor `d/α` of its second term
restored and `ζ = α`.  For AdamNC with `β_{2,t} = 1 - 1/t`, `α_t = α/√t` and
`β_{1,t} = β₁λ^{t-1}`, `0 ≤ β₁ < 1`, `0 ≤ λ < 1`, for every `x* ∈ F`,

  `R_T ≤ D²/(2α(1-β₁)) Σᵢ ‖g_{1:T,i}‖₂ + dβ₁D²G/(α(1-β₁)²(1-λ)²)`
  `     + 2α/(1-β₁)³ Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.09237, §5, Corollary 2. -/
theorem adamNC_regret_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (hβ₂0 : S.β₂ = 0) {α β₁ lam : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hl : 0 ≤ lam) (hl' : lam < 1) (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (T : ℕ) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) xstar T ≤
      D ^ 2 / (2 * α * (1 - β₁)) * ∑ i, S.gnorm (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) T i
      + d * β₁ * D ^ 2 * G / (α * (1 - β₁) ^ 2 * (1 - lam) ^ 2)
      + 2 * α / (1 - β₁) ^ 3 * ∑ i, S.gnorm (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) T i := by
  set R : Rule d := adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)
  have hS1 : S.β₁ 1 = β₁ := by rw [hb]; simp
  have hβt : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1 := fun t _ => by
    rw [hb, hS1]
    exact ⟨by positivity, mul_le_of_le_one_right hβ₁ (pow_le_one₀ hl hl'.le)⟩
  have h5 := adamNC_regret hS hα hαt hβt (hS1 ▸ hβ₁') T hα
    (fun t ht i => (adamNC_inv_cond hβ₂0 hα hαt i).1 t (mem_Icc.1 ht).1)
    (fun t ht i => (adamNC_inv_cond hβ₂0 hα hαt i).2 t (mem_Icc.1 ht).1) hxstar
  rw [hS1, sum_congr rfl fun i _ => sqrt_mul_sqrt_vhat_inv hβ₂0 T i] at h5
  have hdG : 0 ≤ (d : ℝ) * G := by
    rcases Nat.eq_zero_or_pos d with h | h
    · simp [h]
    · exact mul_nonneg (Nat.cast_nonneg _) ((abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem ⟨0, h⟩))
  have h2 : ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat R t i) / S.α t ≤
      d * G * β₁ / α * ∑ t ∈ Icc 1 T, lam ^ (t - 1) * Real.sqrt t := by
    rw [mul_sum]
    refine sum_le_sum fun t _ => ?_
    have hc : 0 ≤ β₁ * lam ^ (t - 1) * Real.sqrt t / α := by positivity
    calc ∑ i, S.β₁ t * Real.sqrt (S.vhat R t i) / S.α t
        = ∑ i, β₁ * lam ^ (t - 1) * Real.sqrt t / α * Real.sqrt (S.vhat R t i) := by
          refine sum_congr rfl fun i _ => ?_
          rw [hb, hαt]; simp only; rw [div_div_eq_mul_div]; ring
      _ ≤ ∑ _i : Fin d, β₁ * lam ^ (t - 1) * Real.sqrt t / α * G :=
          sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (sqrt_vhat_inv_le hS hβ₂0 t i) hc
      _ = _ := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have h2' := mul_le_mul_of_nonneg_left (sum_lambda_sqrt_le hl hl' T)
    (show 0 ≤ d * G * β₁ / α by positivity)
  have hk : 0 ≤ D ^ 2 / (1 - β₁) ^ 2 := by positivity
  have e : d * β₁ * D ^ 2 * G / (α * (1 - β₁) ^ 2 * (1 - lam) ^ 2) =
      D ^ 2 / (1 - β₁) ^ 2 * (d * G * β₁ / α * (1 / (1 - lam) ^ 2)) := by
    generalize 1 - β₁ = q; generalize 1 - lam = r; ring
  have := mul_le_mul_of_nonneg_left (h2.trans h2') hk
  rw [← e] at this
  linarith

/-- **`β_{1,t} = β₁/t` gives `O(√T)` regret for AdamNC.**  With
`β_{2,t} = 1 - 1/t`, `α_t = α/√t` and `β_{1,t} = β₁/t`, `0 ≤ β₁ < 1`, there is
`K` with `R_T ≤ K√T` for every `T` and every `x* ∈ F`.

Source: arXiv:1904.09237, §5, the paragraph after Corollary 2. -/
theorem adamNC_regret_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (hβ₂0 : S.β₂ = 0) {α β₁ : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hb : ∀ t, S.β₁ t = β₁ / t) :
    ∃ K : ℝ, ∀ T : ℕ, ∀ xstar ∈ F,
      S.regret (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) xstar T ≤ K * Real.sqrt T := by
  set R : Rule d := adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)
  have hS1 : S.β₁ 1 = β₁ := by rw [hb]; simp
  have hβt : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1 := fun t ht => by
    rw [hb, hS1]
    exact ⟨by positivity, div_le_self hβ₁ (by exact_mod_cast ht)⟩
  have hdG : 0 ≤ (d : ℝ) * G := by
    rcases Nat.eq_zero_or_pos d with h | h
    · simp [h]
    · exact mul_nonneg (Nat.cast_nonneg _) ((abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem ⟨0, h⟩))
  have hB : 0 < 1 - β₁ := by linarith
  refine ⟨D ^ 2 / (2 * α * (1 - β₁)) * (d * G) + D ^ 2 / (1 - β₁) ^ 2 * (d * G * β₁ / α * 2)
    + 2 * α / (1 - β₁) ^ 3 * (d * G), fun T xstar hx => ?_⟩
  have h5 := adamNC_regret hS hα hαt hβt (hS1 ▸ hβ₁') T hα
    (fun t ht i => (adamNC_inv_cond hβ₂0 hα hαt i).1 t (mem_Icc.1 ht).1)
    (fun t ht i => (adamNC_inv_cond hβ₂0 hα hαt i).2 t (mem_Icc.1 ht).1) hx
  rw [hS1, sum_congr rfl fun i _ => sqrt_mul_sqrt_vhat_inv hβ₂0 T i] at h5
  have h1 : ∑ i, S.gnorm R T i ≤ d * G * Real.sqrt T := by
    refine (sum_le_sum fun i _ => gnorm_le hS R T i).trans_eq ?_
    rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have h2 : ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat R t i) / S.α t ≤
      d * G * β₁ / α * (2 * Real.sqrt T) := by
    calc ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat R t i) / S.α t
        ≤ ∑ t ∈ Icc 1 T, d * G * β₁ / α * (1 / Real.sqrt t) := sum_le_sum fun t ht => by
          have ht0 : (0 : ℝ) < t := by exact_mod_cast (mem_Icc.1 ht).1
          obtain ⟨r, hr⟩ : ∃ r, r = Real.sqrt t := ⟨_, rfl⟩
          have hr0 : 0 < r := hr ▸ Real.sqrt_pos.2 ht0
          have et : (t : ℝ) = r * r := hr ▸ (Real.mul_self_sqrt ht0.le).symm
          have hc : 0 ≤ β₁ / α * (1 / r) := by positivity
          calc ∑ i, S.β₁ t * Real.sqrt (S.vhat R t i) / S.α t
              = ∑ i, β₁ / α * (1 / r) * Real.sqrt (S.vhat R t i) := by
                refine sum_congr rfl fun i _ => ?_
                rw [hb, hαt]; simp only; rw [← hr, et]; field_simp
            _ ≤ ∑ _i : Fin d, β₁ / α * (1 / r) * G :=
                sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (sqrt_vhat_inv_le hS hβ₂0 t i) hc
            _ = _ := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, ← hr]; ring
      _ = d * G * β₁ / α * ∑ t ∈ Icc 1 T, 1 / Real.sqrt t := by rw [mul_sum]
      _ ≤ _ := mul_le_mul_of_nonneg_left (sum_inv_sqrt_le T)
          (div_nonneg (mul_nonneg hdG hβ₁) hα.le)
  have k1 := mul_le_mul_of_nonneg_left h1 (show 0 ≤ D ^ 2 / (2 * α * (1 - β₁)) by positivity)
  have k2 := mul_le_mul_of_nonneg_left h2 (show 0 ≤ D ^ 2 / (1 - β₁) ^ 2 by positivity)
  have k3 := mul_le_mul_of_nonneg_left h1 (show 0 ≤ 2 * α / (1 - β₁) ^ 3 by positivity)
  have e : (D ^ 2 / (2 * α * (1 - β₁)) * (d * G) + D ^ 2 / (1 - β₁) ^ 2 * (d * G * β₁ / α * 2)
      + 2 * α / (1 - β₁) ^ 3 * (d * G)) * Real.sqrt T =
      D ^ 2 / (2 * α * (1 - β₁)) * (d * G * Real.sqrt T)
      + D ^ 2 / (1 - β₁) ^ 2 * (d * G * β₁ / α * (2 * Real.sqrt T))
      + 2 * α / (1 - β₁) ^ 3 * (d * G * Real.sqrt T) := by ring
  rw [e]
  linarith

/-- The hypotheses of Corollary 2 and of the `β₁/t` remark are satisfiable:
the zero cost on `[-1, 1]`, `β₂ = 0`, `α = 1`, `β₁ = 0`, `λ = 0`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ S.β₂ = 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (∀ t, S.β₁ t = 0 * (0 : ℝ) ^ (t - 1)) ∧ (∀ t, S.β₁ t = 0 / t) ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, rfl, one_pos, rfl, le_rfl, one_pos, fun _ => by simp [zeroSetup],
    fun _ => by simp [zeroSetup], fun _ => by norm_num, fun _ => by norm_num⟩

end AdamBeyond
end Transformer
