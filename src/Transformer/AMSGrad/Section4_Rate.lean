import Transformer.AMSGrad.Section4_Lemmas

/-
# AMSGrad — the rate behind Corollary 4.5

§4 of arXiv:1904.03590v4, Corollary 4.5: the bound of Theorem 4.1 grows like
`√(T ln T)`, so it is `o(T)`.  The two facts used: `‖g_{1:T,i}‖₂ ≤ G√T`, and
`(a + b√T + c√(ln T + 1)√T)/T ≤ (|a| + |b| + |c|) √((ln T + 1)/T) → 0`.

Source: arXiv:1904.03590v4, §4, Corollary 4.5.
-/

open Finset Filter Topology

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- `‖g_{1:T,i}‖₂ ≤ G∞√T`, for every rule.
arXiv:1904.03590v4, §4, Corollary 4.5, from `‖∇f_t(x_t)‖_∞ ≤ G∞`. -/
theorem gnorm_le {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (R : Rule d) (T : ℕ) (i : Fin d) : S.gnorm R T i ≤ G * Real.sqrt T := by
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  have h : ∑ t ∈ Icc 1 T, S.g R t i ^ 2 ≤ T * G ^ 2 := by
    refine (sum_le_card_nsmul _ _ (G ^ 2) fun t _ => ?_).trans (by simp)
    exact sq_le_sq' (abs_le.1 (hS.grad_le t _ (x_mem hS R t) i)).1
      (abs_le.1 (hS.grad_le t _ (x_mem hS R t) i)).2
  refine (Real.sqrt_le_sqrt h).trans (le_of_eq ?_)
  rw [Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq hG, mul_comm]

/-- For `T ≥ 1`, `(a + b√T + c√(ln T + 1)√T)/T ≤ (|a| + |b| + |c|) √((ln T + 1)/T)`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem rate_le (a b c : ℝ) {T : ℕ} (hT : 1 ≤ T) :
    (a + b * Real.sqrt T + c * Real.sqrt (Real.log T + 1) * Real.sqrt T) / T ≤
      (|a| + |b| + |c|) * Real.sqrt ((Real.log T + 1) / T) := by
  have hT1 : (1 : ℝ) ≤ T := by exact_mod_cast hT
  have hlog : 0 ≤ Real.log T := Real.log_nonneg hT1
  set s := Real.sqrt T with hs
  set L := Real.sqrt (Real.log T + 1)
  have hs1 : 1 ≤ s := by rw [hs, ← Real.sqrt_one]; exact Real.sqrt_le_sqrt hT1
  have hL1 : 1 ≤ L := by rw [← Real.sqrt_one]; exact Real.sqrt_le_sqrt (by linarith)
  have hTs : (T : ℝ) = s * s := (Real.mul_self_sqrt (Nat.cast_nonneg _)).symm
  rw [Real.sqrt_div (by linarith)]
  change _ ≤ _ * (L / s)
  have hs0 : 0 < s := by linarith
  have hk : L / s * T = L * s := by rw [hTs]; field_simp
  rw [div_le_iff₀ (by linarith), mul_assoc (|a| + |b| + |c|), hk]
  have ha : a ≤ |a| * L * s := (le_abs_self a).trans
    (le_mul_of_one_le_right (abs_nonneg a) (one_le_mul_of_one_le_of_one_le hL1 hs1) |>.trans
      (le_of_eq (mul_assoc _ _ _).symm))
  have hb : b * s ≤ |b| * L * s :=
    mul_le_mul_of_nonneg_right ((le_abs_self b).trans
      (le_mul_of_one_le_right (abs_nonneg b) hL1)) hs0.le
  have hc : c * L * s ≤ |c| * L * s :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (le_abs_self c) (by linarith))
      hs0.le
  nlinarith

/-- `√((ln T + 1)/T) → 0`.  arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem tendsto_rate :
    Tendsto (fun T : ℕ => Real.sqrt ((Real.log T + 1) / T)) atTop (𝓝 0) := by
  have hlog : Tendsto (fun x : ℝ => Real.log x / x) atTop (𝓝 0) := by
    simpa using Real.tendsto_pow_log_div_mul_add_atTop 1 0 1 one_ne_zero
  have h := (hlog.comp tendsto_natCast_atTop_atTop).add
    (tendsto_const_div_atTop_nhds_zero_nat (𝕜 := ℝ) 1)
  rw [add_zero] at h
  have h' := (Real.continuous_sqrt.tendsto 0).comp h
  rw [Real.sqrt_zero] at h'
  refine h'.congr fun T => ?_
  simp [Function.comp, add_div]

/-- The hypothesis of `rate_le` is satisfiable: `T = 1`. -/
example : 1 ≤ (1 : ℕ) := le_rfl

end AMSGrad
end Transformer
