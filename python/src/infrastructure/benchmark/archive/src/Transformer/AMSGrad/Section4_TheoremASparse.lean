import Transformer.AMSGrad.Section4_TheoremAGeneral

/-
# AMSGrad — Theorem A with a sparse first-moment schedule

Theorem A of arXiv:1904.03590v4, §1, assumes only
`0 ≤ β₁,t ≤ β₁ = β₁,1 < 1`. Its printed second constant is proved
for every such schedule by `theorem_A` in `Section4_TheoremAFinite`.
The earlier Abel-based bound `abel_general` uses twice that constant while
leaving slack in the first term. Here the slack pays for the
difference when, coordinatewise,

  `Σ_{t=1}^T β₁,t √v̂_{t,i}/α_t ≤ (1 + β₁) √v̂_{T,i}/α_T`.

This condition on the actual run permits schedules with upward jumps. It
records another case where the Abel proof works directly; the full Theorem A
now follows by a different argument.

Source: arXiv:1904.03590v4, §1, Theorem A, and §3, Lemma 3.1.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-- The scalar bound with Theorem A's printed constants follows from
`abel_general` when the weighted schedule mass is at most `(1 + β) a_T`.
This allows non-monotone schedules, unlike `abel_antitone`.

Source: arXiv:1904.03590v4, §3, the telescoping step in the attempted
proof of Theorem A. -/
theorem abel_printed_of_sparse
    (a b u : ℕ → ℝ) {β E : ℝ} (hβ : β < 1)
    (ha : ∀ t, 1 ≤ t → 0 ≤ a t) (hmono : ∀ t, 2 ≤ t → a (t - 1) ≤ a t)
    (hb : ∀ t, 1 ≤ t → 0 ≤ b t ∧ b t ≤ β)
    (hu : ∀ t, 1 ≤ t → 0 ≤ u t ∧ u t ≤ E)
    {T : ℕ} (hT : 1 ≤ T)
    (hmass : ∑ t ∈ Icc 1 T, b t * a t ≤ (1 + β) * a T) :
    ∑ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1))
        + ∑ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t ≤
      E * a T / (1 - β) + E / (2 * (1 - β)) * ∑ t ∈ Icc 1 T, b t * a t := by
  have h := abel_general a b u hβ ha hmono hb hu hT
  have hE : 0 ≤ E := (hu 1 le_rfl).1.trans (hu 1 le_rfl).2
  have hk : 0 < 1 - β := by linarith
  set Z := ∑ t ∈ Icc 1 T, b t * a t with hZ
  have hdiff : E / (2 * (1 - β)) * (Z - (1 + β) * a T) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos (by positivity) (sub_nonpos.mpr hmass)
  have heq : E * a T / 2 + E / (1 - β) * Z =
      E * a T / (1 - β) + E / (2 * (1 - β)) * Z +
        E / (2 * (1 - β)) * (Z - (1 + β) * a T) := by
    field_simp
    ring
  rw [heq] at h
  linarith

/-- The scalar hypotheses allow an upward jump: `b₁ = b₃ = 1/2`, `b₂ = 0`,
with positive weights and distances. -/
example := abel_printed_of_sparse (fun _ : ℕ => (1 : ℝ))
    (fun t : ℕ => if t = 1 ∨ t = 3 then (1 / 2 : ℝ) else 0)
    (fun _ : ℕ => (1 : ℝ)) (β := 1 / 2) (E := 1)
    (by norm_num)
    (by intro t ht; norm_num)
    (by intro t ht; norm_num)
    (by intro t ht; split_ifs <;> norm_num)
    (by intro t ht; norm_num)
    (T := 3) (by norm_num)
    (by norm_num [Finset.sum_Icc_succ_top])

variable {d : ℕ}

/-- **Theorem A's printed bound for sparse schedules.** Under every
hypothesis of Theorem A and the additional coordinatewise mass condition
`Σ β₁,t (√v̂_t/α_t) ≤ (1 + β₁) √v̂_T/α_T`, its exact printed regret bound
holds. The extra condition may hold for schedules that increase after a
decrease; it is not part of the source's Theorem A.

Source: arXiv:1904.03590v4, §1, Theorem A (special case); §3, Lemma 3.1. -/
theorem theorem_A_sparse {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hxstar : xstar ∈ F)
    (hmass : ∀ i : Fin d,
      ∑ t ∈ Finset.Icc 1 T, S.β₁ t *
        (Real.sqrt (S.vhat amsgradRule t i) / S.α t) ≤
      (1 + S.β₁ 1) * (Real.sqrt (S.vhat amsgradRule T i) / S.α T)) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (2 * (1 - S.β₁ 1)) *
          ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  have hK := fun i : Fin d => abel_printed_of_sparse
    (fun t => Real.sqrt (S.vhat amsgradRule t i) / S.α t) S.β₁
    (fun t => (S.x amsgradRule t i - xstar i) ^ 2) (β := S.β₁ 1) (E := D ^ 2) hβ₁'
    (fun t ht => div_nonneg (Real.sqrt_nonneg _) (hα' t ht).le)
    (fun t ht => sqrt_vhat_div_mono hαt hα ht i) hβ₁
    (fun t ht => ⟨sq_nonneg _, by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS _ t) _ hxstar i) 2⟩)
    hT (hmass i)
  have h := regret_le_of_abel hS hα hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ hT hxstar hK
  have hsqrtT : 0 < Real.sqrt (T : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast hT)
  have heqT (i : Fin d) :
      D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / (1 - S.β₁ 1) =
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * Real.sqrt (S.vhat amsgradRule T i) := by
    rw [hαt]
    field_simp
  have heqB (i : Fin d) :
      ∑ t ∈ Icc 1 T, S.β₁ t * (Real.sqrt (S.vhat amsgradRule t i) / S.α t) =
        ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    exact sum_congr rfl fun t _ => by ring
  calc
    S.regret amsgradRule xstar T ≤
        ∑ i, (D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / (1 - S.β₁ 1) +
          D ^ 2 / (2 * (1 - S.β₁ 1)) *
            ∑ t ∈ Icc 1 T, S.β₁ t * (Real.sqrt (S.vhat amsgradRule t i) / S.α t)) +
        α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := h
    _ = _ := by
      have hfirst :
          (∑ i, D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / (1 - S.β₁ 1)) =
            D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) *
              ∑ i, Real.sqrt (S.vhat amsgradRule T i) := by
        rw [mul_sum]
        exact sum_congr rfl fun i _ => heqT i
      have hsecond :
          (∑ i, D ^ 2 / (2 * (1 - S.β₁ 1)) *
            ∑ t ∈ Icc 1 T, S.β₁ t * (Real.sqrt (S.vhat amsgradRule t i) / S.α t)) =
            D ^ 2 / (2 * (1 - S.β₁ 1)) *
              ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
        rw [mul_sum]
        exact sum_congr rfl fun i _ => by rw [heqB i]
      rw [sum_add_distrib, hfirst, hsecond]

/-- The hypotheses of `theorem_A_sparse` are satisfiable in the nonzero-gradient
run of Example 3.2 at `T = 1`, with comparator `x* = -1`. -/
example := theorem_A_sparse (S := exaSetup) (F := Set.Icc (fun _ => -1) (fun _ => 1))
    (D := 2) (G := 1010) (α := 0.001) isOnlineConvex_exa (by norm_num) rfl
    (fun t _ => ⟨by show (0 : ℝ) ≤ 0.9 * 0.001 ^ (t - 1); positivity,
      by show (0.9 : ℝ) * 0.001 ^ (t - 1) ≤ 0.9 * 0.001 ^ (1 - 1)
         exact mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (by norm_num) (by norm_num)
           (Nat.zero_le _)) (by norm_num)⟩)
    (by show (0.9 : ℝ) * 0.001 ^ (1 - 1) < 1; norm_num)
    (by show (0 : ℝ) < 0.999; norm_num) (by show (0.999 : ℝ) < 1; norm_num)
    (by
      show (0.9 : ℝ) * 0.001 ^ (1 - 1) / Real.sqrt 0.999 < 1
      rw [div_lt_one (Real.sqrt_pos.2 (by norm_num))]
      have : (0.9 : ℝ) < Real.sqrt 0.999 := Real.lt_sqrt (by norm_num) |>.2 (by norm_num)
      norm_num at this ⊢; linarith)
    (le_refl 1) (xstar := fun _ => -1) ⟨fun _ => le_rfl, fun _ => by norm_num⟩
    (by
      intro i
      simp only [Icc_self, sum_singleton]
      have ha : 0 ≤ Real.sqrt (exaSetup.vhat amsgradRule 1 i) / exaSetup.α 1 := by
        apply div_nonneg (Real.sqrt_nonneg _)
        norm_num [exaSetup]
      have hb : exaSetup.β₁ 1 = 0.9 := by norm_num [exaSetup]
      rw [hb]
      nlinarith)

end AMSGrad
end Transformer
