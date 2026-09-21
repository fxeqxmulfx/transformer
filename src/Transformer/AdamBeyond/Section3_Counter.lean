import Transformer.AdamBeyond.Section3_Run

/-
# Adam and beyond — §3: Theorems 1 and 6, Adam does not converge

On the run of `Section3_Run`, the iterate is back at `x = 1` at the start of
every block of three steps, so each block costs at least `6s - 2s - 2s = 2s`
of regret against `x* = -1`, and `R_T ≥ (2s/3) T` for every `T`.

**What the source says and what is carried here.**

* "`R_T/T ↛ 0`" is stated in the stronger form it is proved in: a fixed
  `x* ∈ F` and a `c > 0` with `c T ≤ R_T` for every `T`.  The regret against the
  minimizer of `Σ f_t` over `F` is at least the regret against `x*`.

* The source concludes `R_T ≥ (2C - 4)T/3` under `C ≥ 2`, which is `0` at
  `C = 2`; §3 has `C > 2`.  Here `C = 3`.

* Theorem 6 asks for "all the conditions in [Kingma & Ba]"; of those that bear
  on this run, `β₁ = 0`, `0 < β₂ < 1`, `α_t = α/√t`, bounded gradients and a
  bounded feasible set are part of the statement.  Its proof displays
  `√((β₂C² + 1 - β₂)/2) = √(1 - β₂ + ε/C²)`, which is false as written; it holds
  with `+ ε` on the left.

* Adam's `Γ_t` is indefinite: `gamma_adam_neg`, `Γ₂ < 0` on the same run.

Source: arXiv:1904.09237, §3, Theorem 1 and the remark after it, eq:gamma-t;
Appendix, proofs of Theorems 1 and 6.
-/

namespace Transformer
namespace AdamBeyond

open AMSGrad

/-- The regret of one step against `x* = -1` is at least `-2s`. -/
theorem cex_term_ge {s ε : ℝ} (hs : 0 ≤ s) (α : ℕ → ℝ) (t : ℕ) :
    -(2 * s) ≤ (cexSetup s α).f t ((cexSetup s α).x (adamEpsRule ε) t) - (cexSetup s α).f t
      (fun _ => -1) := by
  have h := x_mem (isOnlineConvex_cex hs α) (adamEpsRule ε) t
  have h1 := h.1 0
  have h2 := h.2 0
  simp only [cexSetup, slope] at h1 h2 ⊢
  split_ifs <;> nlinarith

/-- `R_T ≥ (2s/3) T` against `x* = -1`, under the hypotheses of `cex_block`. -/
theorem cex_regret {s ε : ℝ} {α : ℕ → ℝ} (hs : 0 < s) (hε : 0 ≤ ε) (hε' : ε ≤ s ^ 2 / 100)
    (hα₀ : ∀ t, 0 ≤ α t) (hα₁ : ∀ k, α (3 * k + 1) * (30 / 29) ≤ 1)
    (hα : ∀ k, α (3 * k + 1) * (30 / 29) ≤ α (3 * k + 2) * (20 / 21) + α (3 * k + 3) * (100 / 101))
    (T : ℕ) : 2 * s / 3 * T ≤ (cexSetup s α).regret (adamEpsRule ε) (fun _ => -1) T := by
  set S := cexSetup s α
  set R : Rule 1 := adamEpsRule ε
  have succ : ∀ n, S.regret R (fun _ => -1) (n + 1) = S.regret R (fun _ => -1) n
      + (S.f (n + 1) (S.x R (n + 1)) - S.f (n + 1) (fun _ => -1)) := fun n => by
    simp only [Setup.regret]; rw [Finset.sum_Icc_succ_top (by omega)]
  have ge := fun n => cex_term_ge (ε := ε) hs.le α n
  have first : ∀ k, S.f (3 * k + 1) (S.x R (3 * k + 1)) - S.f (3 * k + 1) (fun _ => -1) = 6 * s :=
    fun k => by
      have hx : (S.x R (3 * k + 1)) 0 = 1 := (cex_block hs hε hε' hα₀ hα₁ hα k).1
      simp only [S, cexSetup, slope, show (3 * k + 1) % 3 = 1 by omega, ↓reduceIte] at hx ⊢
      rw [hx]; ring
  have blocks : ∀ k, 2 * s * k ≤ S.regret R (fun _ => -1) (3 * k) := by
    intro k
    induction k with
    | zero => simp [Setup.regret]
    | succ k ih =>
      rw [show 3 * (k + 1) = 3 * k + 1 + 1 + 1 by ring, succ, succ, succ, first]
      have := ge (3 * k + 1 + 1); have := ge (3 * k + 1 + 1 + 1)
      push_cast; linarith
  obtain ⟨k, r, hr, rfl⟩ : ∃ k r, r < 3 ∧ T = 3 * k + r := ⟨T / 3, T % 3, Nat.mod_lt _ (by norm_num),
    (Nat.div_add_mod T 3).symm⟩
  have hk := blocks k
  interval_cases r
  · simp only [add_zero]; push_cast; linarith
  · rw [succ, first]; push_cast; linarith
  · rw [succ, succ, first]; have := ge (3 * k + 1 + 1); push_cast; linarith

/-- `α_t = (1/2)/√t` satisfies the hypotheses of `cex_block`. -/
theorem half_div_sqrt_hyp :
    (∀ t, 0 ≤ (fun t : ℕ => 1 / 2 / Real.sqrt t) t) ∧
    (∀ k, (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 1) * (30 / 29) ≤ 1) ∧
    (∀ k, (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 1) * (30 / 29) ≤
      (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 2) * (20 / 21) +
        (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 3) * (100 / 101)) := by
  refine ⟨fun t => by positivity, fun k => ?_, fun k => ?_⟩
  · have h : 1 ≤ Real.sqrt ((3 * k + 1 : ℕ) : ℝ) := Real.one_le_sqrt.mpr (by push_cast; linarith)
    simp only
    rw [div_mul_eq_mul_div, div_le_one (by linarith)]
    nlinarith
  · simp only
    set a := Real.sqrt ((3 * k + 1 : ℕ) : ℝ)
    set b := Real.sqrt ((3 * k + 2 : ℕ) : ℝ)
    set c := Real.sqrt ((3 * k + 3 : ℕ) : ℝ)
    have ha : 0 < a := Real.sqrt_pos.mpr (by positivity)
    have hb : 0 < b := Real.sqrt_pos.mpr (by positivity)
    have hc : 0 < c := Real.sqrt_pos.mpr (by positivity)
    have hab : 7 / 10 * b ≤ a := Real.le_sqrt_of_sq_le (by
      rw [mul_pow, Real.sq_sqrt (by positivity)]; push_cast; linarith)
    have hac : 57 / 100 * c ≤ a := Real.le_sqrt_of_sq_le (by
      rw [mul_pow, Real.sq_sqrt (by positivity)]; push_cast; linarith)
    have e1 : 7 / 10 / a ≤ 1 / b := by rw [div_le_div_iff₀ ha hb]; linarith
    have e2 : 57 / 100 / a ≤ 1 / c := by rw [div_le_div_iff₀ ha hc]; linarith
    have u : 0 ≤ 1 / a := by positivity
    have q : ∀ x : ℝ, 1 / 2 / x = 1 / 2 * (1 / x) := fun x => by ring
    rw [q, q, q, div_eq_mul_one_div (7 / 10 : ℝ)] at *
    rw [div_eq_mul_one_div (57 / 100 : ℝ)] at e2
    linarith

/-- The hypotheses of `cex_block`, `cex_regret` and `block_arith` are satisfiable:
`s = 1`, `ε = 0`, `α_t = (1/2)/√t`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ^ 2 / 100 ∧
    (∀ t, 0 ≤ (fun t : ℕ => 1 / 2 / Real.sqrt t) t) ∧
    (∀ k, (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 1) * (30 / 29) ≤ 1) ∧
    (∀ k, (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 1) * (30 / 29) ≤
      (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 2) * (20 / 21) +
        (fun t : ℕ => 1 / 2 / Real.sqrt t) (3 * k + 3) * (100 / 101)) :=
  ⟨one_pos, le_rfl, by norm_num, half_div_sqrt_hyp⟩

/-- The hypothesis of `counter_example_epsilon` is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

/-- **Theorem 6.**  For every `ε > 0`, Adam with the update eq:mod-update has
non-zero average regret on an online convex problem: `c T ≤ R_T` for every `T`.
arXiv:1904.09237, §3 and Appendix, Theorem 6. -/
theorem counter_example_epsilon {ε : ℝ} (hε : 0 < ε) :
    ∃ (S : Setup 1) (F : Set (Vec 1)) (D G : ℝ), IsOnlineConvex S F D G ∧
      (∃ α > 0, S.α = fun t : ℕ => α / Real.sqrt t) ∧ (∀ t, S.β₁ t = 0) ∧
      0 < S.β₂ ∧ S.β₂ < 1 ∧
      ∃ xstar ∈ F, ∃ c > 0, ∀ T : ℕ, c * T ≤ S.regret (adamEpsRule ε) xstar T := by
  have hs : 0 < 10 * Real.sqrt ε := by positivity
  have hs2 : ε ≤ (10 * Real.sqrt ε) ^ 2 / 100 := by
    rw [mul_pow, Real.sq_sqrt hε.le]; linarith
  obtain ⟨h₀, h₁, h⟩ := half_div_sqrt_hyp
  exact ⟨_, _, _, _, isOnlineConvex_cex hs.le _, ⟨1 / 2, by norm_num, rfl⟩, fun _ => rfl,
    by norm_num [cexSetup], by norm_num [cexSetup], fun _ => -1, ⟨fun _ => le_rfl,
      fun _ => by norm_num⟩, _, by positivity, cex_regret hs hε.le hs2 h₀ h₁ h⟩

/-- **Theorem 1.**  Adam has non-zero average regret on an online convex
problem: `c T ≤ R_T` for every `T`.  arXiv:1904.09237, §3, Theorem 1. -/
theorem counter_example :
    ∃ (S : Setup 1) (F : Set (Vec 1)) (D G : ℝ), IsOnlineConvex S F D G ∧
      (∃ α > 0, S.α = fun t : ℕ => α / Real.sqrt t) ∧ (∀ t, S.β₁ t = 0) ∧
      0 < S.β₂ ∧ S.β₂ < 1 ∧
      ∃ xstar ∈ F, ∃ c > 0, ∀ T : ℕ, c * T ≤ S.regret adamRule xstar T := by
  obtain ⟨h₀, h₁, h⟩ := half_div_sqrt_hyp
  rw [← adamEpsRule_zero]
  exact ⟨_, _, _, _, isOnlineConvex_cex zero_le_one _, ⟨1 / 2, by norm_num, rfl⟩, fun _ => rfl,
    by norm_num [cexSetup], by norm_num [cexSetup], fun _ => -1, ⟨fun _ => le_rfl,
      fun _ => by norm_num⟩, _, by norm_num,
    cex_regret one_pos le_rfl (by norm_num) h₀ h₁ h⟩

/-- **The remark after Theorem 1.**  The same holds with a constant step size
`α_t = α`.  arXiv:1904.09237, §3, after Theorem 1. -/
theorem counter_example_const :
    ∃ (S : Setup 1) (F : Set (Vec 1)) (D G : ℝ), IsOnlineConvex S F D G ∧
      (∃ α > 0, S.α = fun _ => α) ∧ (∀ t, S.β₁ t = 0) ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧
      ∃ xstar ∈ F, ∃ c > 0, ∀ T : ℕ, c * T ≤ S.regret adamRule xstar T := by
  rw [← adamEpsRule_zero]
  exact ⟨_, _, _, _, isOnlineConvex_cex zero_le_one (fun _ => 1 / 2), ⟨1 / 2, by norm_num, rfl⟩,
    fun _ => rfl, by norm_num [cexSetup], by norm_num [cexSetup], fun _ => -1,
    ⟨fun _ => le_rfl, fun _ => by norm_num⟩, _, by norm_num,
    cex_regret one_pos le_rfl (by norm_num) (fun _ => by norm_num) (fun _ => by norm_num)
      (fun _ => by norm_num)⟩

/-- **`Γ_t` is indefinite for Adam.**  With `α_t = α/√t`, `β₁ = 0`, `0 < β₂ < 1`,
Adam's `Γ₂` is negative.  arXiv:1904.09237, §3, after eq:gamma-t. -/
theorem gamma_adam_neg :
    ∃ (S : Setup 1) (F : Set (Vec 1)) (D G : ℝ), IsOnlineConvex S F D G ∧
      (∃ α > 0, S.α = fun t : ℕ => α / Real.sqrt t) ∧ (∀ t, S.β₁ t = 0) ∧
      0 < S.β₂ ∧ S.β₂ < 1 ∧ Gamma S adamRule 2 0 < 0 := by
  refine ⟨cexSetup 1 fun t : ℕ => 1 / 2 / Real.sqrt t, _, _, _, isOnlineConvex_cex zero_le_one _,
    ⟨1 / 2, by norm_num, rfl⟩, fun _ => rfl, by norm_num [cexSetup], by norm_num [cexSetup], ?_⟩
  rw [← adamEpsRule_zero, gamma_succ rfl]
  set S := cexSetup 1 fun t : ℕ => 1 / 2 / Real.sqrt t
  have v₁ : S.vhat (adamEpsRule 0) 1 0 = 891 / 100 := by
    change (S.step (adamEpsRule 0) 1 (S.state _ 0)).v 0 + 0 = _
    rw [cex_v]; norm_num [slope, Setup.state]
  have v₂ : S.vhat (adamEpsRule 0) (1 + 1) 0 = 10791 / 10000 := by
    change (S.step (adamEpsRule 0) 2 (S.step (adamEpsRule 0) 1 (S.state _ 0))).v 0 + 0 = _
    rw [cex_v, cex_v]; norm_num [slope, Setup.state]
  rw [v₁, v₂, div_neg_iff]
  refine Or.inr ⟨?_, by norm_num⟩
  rw [sub_neg, ← Real.sqrt_mul (by norm_num), Nat.cast_one, Real.sqrt_one, mul_one]
  exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)

end AdamBeyond
end Transformer
