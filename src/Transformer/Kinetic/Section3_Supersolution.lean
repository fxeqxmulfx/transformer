module

public import Transformer.Kinetic.Section3_ChainSums
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# An explicit supersolution for the Glauber derivatives

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 1. A perturbation of
token `b` reaches token `a > b` through chains `b < j_r < … < j_1 < a`, each
link weighing `ω ≲ 1/(j+1)`. The source sums the chains and gets
`(a+1)⁻¹ e^{√(C t log((a+1)/(b+1)))}`. Here the sum is replaced by a profile
that dominates it, for each `γ ∈ (0, 1]`:

  `Ĝ^b_a(t) = γ (e^{αt/γ} - 1) ((a+1)/(b+1))^γ / (a+1)`   for `a > b`,

`Ĝ^b_b = 1`, and `Ĝ^b_a = 0` for `a < b`. It satisfies
`Ĝ^b_a' ≥ α/(a+1) Σ_{c<a} Ĝ^b_c` (`sum_profile_le`), because
`γ Σ_{b<c<a} (c+1)^{γ-1} ≤ a^γ - (b+1)^γ` (`sum_rpow_le`), and at time `t` it
is at most `αt e^{αt/γ} ((a+1)/(b+1))^γ / (a+1)` (`profile_le`). Choosing
`γ ~ √(αt / log((a+1)/(b+1)))` recovers the source's bound.
-/

@[expose] public section

open Real

namespace Transformer
namespace Kinetic

/-- **The profile** `Ĝ^b_a(t)`. -/
noncomputable def profile (α γ : ℝ) (b a : ℕ) (t : ℝ) : ℝ :=
  if a < b then 0 else if a = b then 1 else
    γ * (Real.exp (α / γ * t) - 1) * ratioPow γ a b / ((a : ℝ) + 1)

/-- Its time derivative. -/
noncomputable def profileDeriv (α γ : ℝ) (b a : ℕ) (t : ℝ) : ℝ :=
  if a < b then 0 else if a = b then 0 else
    α * Real.exp (α / γ * t) * ratioPow γ a b / ((a : ℝ) + 1)

theorem hasDerivAt_profile (α : ℝ) {γ : ℝ} (hγ : γ ≠ 0) (b a : ℕ) (t : ℝ) :
    HasDerivAt (profile α γ b a) (profileDeriv α γ b a t) t := by
  unfold profile profileDeriv
  split_ifs
  · exact hasDerivAt_const t 0
  · exact hasDerivAt_const t 1
  · have h1 : HasDerivAt (fun s => α / γ * s) (α / γ) t := by
      simpa using (hasDerivAt_id t).const_mul (α / γ)
    have h : HasDerivAt (fun s => γ * (Real.exp (α / γ * s) - 1) * ratioPow γ a b / ((a : ℝ) + 1))
        (γ * (Real.exp (α / γ * t) * (α / γ)) * ratioPow γ a b / ((a : ℝ) + 1)) t :=
      ((h1.exp.sub_const 1).const_mul γ).mul_const _ |>.div_const _
    convert h using 1
    field_simp

theorem profile_zero (α γ : ℝ) (b a : ℕ) : profile α γ b a 0 = if a = b then 1 else 0 := by
  unfold profile
  split_ifs <;> first | rfl | omega | simp

theorem profile_nonneg {α γ : ℝ} (hα : 0 ≤ α) (hγ : 0 < γ) (b a : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    0 ≤ profile α γ b a t := by
  unfold profile
  have : 0 ≤ Real.exp (α / γ * t) - 1 := by
    rw [sub_nonneg]
    exact Real.one_le_exp (by positivity)
  have := ratioPow_pos γ a b
  split_ifs <;> positivity

theorem profileDeriv_nonneg {α γ : ℝ} (hα : 0 ≤ α) (b a : ℕ) (t : ℝ) :
    0 ≤ profileDeriv α γ b a t := by
  unfold profileDeriv
  have := ratioPow_pos γ a b
  split_ifs <;> positivity

theorem profile_mono {α γ : ℝ} (hα : 0 ≤ α) (hγ : 0 < γ) (b a : ℕ) {s t : ℝ} (hst : s ≤ t) :
    profile α γ b a s ≤ profile α γ b a t := by
  unfold profile
  have : Real.exp (α / γ * s) ≤ Real.exp (α / γ * t) :=
    Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left hst (by positivity))
  have := ratioPow_pos γ a b
  split_ifs
  · rfl
  · rfl
  · gcongr

/-- **The supersolution inequality** `α/(a+1) Σ_{c<a} Ĝ^b_c ≤ Ĝ^b_a'`. -/
theorem sum_profile_le {α γ : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (b a : ℕ) {t : ℝ}
    (ht : 0 ≤ t) :
    α / ((a : ℝ) + 1) * ∑ c ∈ Finset.range a, profile α γ b c t ≤ profileDeriv α γ b a t := by
  have hE : 1 ≤ Real.exp (α / γ * t) := Real.one_le_exp (by positivity)
  rcases lt_trichotomy a b with hab | rfl | hab
  · have : ∀ c ∈ Finset.range a, profile α γ b c t = 0 := fun c hc => by
      rw [profile, ite_eq_left ((Finset.mem_range.1 hc).trans hab)]
    rw [Finset.sum_eq_zero this, mul_zero]
    exact profileDeriv_nonneg hα b a t
  · have : ∀ c ∈ Finset.range a, profile α γ a c t = 0 := fun c hc => by
      rw [profile, ite_eq_left (Finset.mem_range.1 hc)]
    rw [Finset.sum_eq_zero this, mul_zero]
    exact profileDeriv_nonneg hα a a t
  · have hterm : ∀ c ∈ Finset.range a, profile α γ b c t ≤ (if c = b then 1 else 0) +
        (if b < c then γ * (Real.exp (α / γ * t) - 1) * (ratioPow γ c b / ((c : ℝ) + 1))
          else 0) := by
      intro c _
      unfold profile
      by_cases h1 : c < b
      · rw [ite_eq_left h1, ite_eq_right (by omega), ite_eq_right (by omega)]
        simp
      · by_cases h2 : c = b
        · subst h2
          simp
        · rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h2, ite_eq_left (by omega),
            mul_div_assoc, zero_add]
    have hsum := Finset.sum_le_sum hterm
    rw [Finset.sum_add_distrib, Finset.sum_ite_eq' (Finset.range a) b,
      ite_eq_left (Finset.mem_range.2 hab), ← Finset.sum_filter,
      show (Finset.range a).filter (fun c => b < c) = Finset.Ico (b + 1) a by
        ext c; simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]; omega,
      ← Finset.mul_sum] at hsum
    have hq := sum_ratioPow_div_le hγ0 hγ1 hab
    have hq1 := one_le_ratioPow hγ0.le hab.le
    have hmul : γ * (Real.exp (α / γ * t) - 1) *
        ∑ c ∈ Finset.Ico (b + 1) a, ratioPow γ c b / ((c : ℝ) + 1) ≤
        (Real.exp (α / γ * t) - 1) * (ratioPow γ a b - 1) := by
      have := mul_le_mul_of_nonneg_left hq (by nlinarith : 0 ≤ γ * (Real.exp (α / γ * t) - 1))
      calc _ ≤ γ * (Real.exp (α / γ * t) - 1) * ((ratioPow γ a b - 1) / γ) := this
        _ = (Real.exp (α / γ * t) - 1) * (ratioPow γ a b - 1) := by field_simp
    have hfinal : ∑ c ∈ Finset.range a, profile α γ b c t ≤
        Real.exp (α / γ * t) * ratioPow γ a b := by
      nlinarith
    unfold profileDeriv
    rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    calc α / ((a : ℝ) + 1) * ∑ c ∈ Finset.range a, profile α γ b c t
        ≤ α / ((a : ℝ) + 1) * (Real.exp (α / γ * t) * ratioPow γ a b) :=
          mul_le_mul_of_nonneg_left hfinal (by positivity)
      _ = α * Real.exp (α / γ * t) * ratioPow γ a b / ((a : ℝ) + 1) := by ring

/-- `γ (e^{αt/γ} - 1) ≤ αt e^{αt/γ}`, so `Ĝ^b_a(t) ≤ αt e^{αt/γ} q_γ(a,b) / (a+1)`. -/
theorem profile_le (α : ℝ) {γ : ℝ} (hγ : 0 < γ) {b a : ℕ} (hba : b < a) (t : ℝ) :
    profile α γ b a t ≤ α * t * Real.exp (α / γ * t) * ratioPow γ a b / ((a : ℝ) + 1) := by
  unfold profile
  rw [ite_eq_right (by omega), ite_eq_right (by omega)]
  have hx := Real.add_one_le_exp (-(α / γ * t))
  have hE := Real.exp_pos (α / γ * t)
  have hkey : γ * (Real.exp (α / γ * t) - 1) ≤ α * t * Real.exp (α / γ * t) := by
    rw [Real.exp_neg] at hx
    have h1 : 1 - α / γ * t ≤ (Real.exp (α / γ * t))⁻¹ := by linarith
    have h2 : Real.exp (α / γ * t) - 1 ≤ α / γ * t * Real.exp (α / γ * t) := by
      have := mul_le_mul_of_nonneg_right h1 hE.le
      rw [inv_mul_cancel₀ hE.ne'] at this
      nlinarith
    calc γ * (Real.exp (α / γ * t) - 1) ≤ γ * (α / γ * t * Real.exp (α / γ * t)) :=
          mul_le_mul_of_nonneg_left h2 hγ.le
      _ = α * t * Real.exp (α / γ * t) := by field_simp
  have := ratioPow_pos γ a b
  gcongr

/-- The hypotheses of the profile bounds hold at `α = 1` and `γ = 1` (`γ = 1/2` for
`sum_profile_le`): `hasDerivAt_profile`, `profile_nonneg`, `profileDeriv_nonneg`,
`profile_mono`, `sum_profile_le`, `profile_le`. -/
example := And.intro (hasDerivAt_profile 1 (γ := 1) one_ne_zero 0 1 0) <|
  And.intro (profile_nonneg (α := 1) (γ := 1) zero_le_one one_pos 0 1 (t := 1) zero_le_one) <|
  And.intro (profileDeriv_nonneg (α := 1) (γ := 1) zero_le_one 0 1 1) <|
  And.intro (profile_mono (α := 1) (γ := 1) zero_le_one one_pos 0 1 (s := 0) (t := 1)
    zero_le_one) <|
  And.intro (sum_profile_le (α := 1) (γ := 1 / 2) zero_le_one (by norm_num) (by norm_num) 0 1
    (t := 1) zero_le_one)
  (profile_le 1 (γ := 1) one_pos (b := 0) (a := 1) Nat.one_pos 1)

end Kinetic
end Transformer
