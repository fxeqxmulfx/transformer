import Transformer.AMSGrad.Section1_TheoremA
import Transformer.AMSGrad.Section2_Prelim

/-
# AMSGrad — one step of the proof of Lemma 3.1

§3 of arXiv:1904.03590v4: the pieces of the proof of Lemma 3.1 that concern a
single step `t`: the projection bound, the inequality (tvheq) for
`⟨g_t, x_t - x*⟩`, and the Young bound on the `m_{t-1}` term.

**What the source says and what is carried here.**

* The source applies Lemma 2.7 (McMahan and Streeter) with `Q = √V̂_t`, which
  asks `Q` to be positive definite.  `v̂_{t,i}` may vanish; the projection
  bound is proved here for every weight `w ≥ 0` straight from the minimizing
  property of the projection, `proj_le`.

* Where `v̂_{t,i} = 0` the source's quotients `m_{t,i}/√v̂_{t,i}` are
  undefined; there `m_{t,i} = 0` (`m_eq_zero`), both sides of each identity
  vanish coordinatewise, and the division by zero of Lean is harmless.

Source: arXiv:1904.03590v4, §3, proof of Lemma 3.1.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- `v_t ≥ 0` for every rule, when `0 ≤ β₂ ≤ 1`. -/
theorem v_nonneg {S : Setup d} (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (R : Rule d) (n : ℕ) :
    0 ≤ (S.state R n).v := by
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
    intro i
    change 0 ≤ S.β₂ * (S.state R n).v i + (1 - S.β₂) * grad (S.f (n + 1)) (S.state R n).x i ^ 2
    exact add_nonneg (mul_nonneg hβ₂ (ih i)) (mul_nonneg (by linarith) (sq_nonneg _))

/-- For `0 < β₂ < 1`, `v_{t,i} = 0` forces `g_{k,i} = 0` for `k ≤ t`, hence
`m_{t,i} = 0`. -/
theorem m_eq_zero {S : Setup d} (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (R : Rule d) {n : ℕ}
    {i : Fin d} (h : S.v R n i = 0) : S.m R n i = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have hv : S.v R (n + 1) i = S.β₂ * S.v R n i + (1 - S.β₂) * S.g R (n + 1) i ^ 2 := rfl
    have h0 : 0 ≤ S.v R n i := v_nonneg hβ₂.le hβ₂'.le R n i
    have h1 : 0 ≤ (1 - S.β₂) * S.g R (n + 1) i ^ 2 := mul_nonneg (by linarith) (sq_nonneg _)
    have hn : S.v R n i = 0 := by nlinarith
    have hg : S.g R (n + 1) i = 0 := by
      have : (1 - S.β₂) * S.g R (n + 1) i ^ 2 = 0 := by nlinarith
      rcases mul_eq_zero.mp this with h | h
      · linarith
      · exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h
    change S.β₁ (n + 1) * S.m R n i + (1 - S.β₁ (n + 1)) * S.g R (n + 1) i = 0
    rw [ih hn, hg]; ring

/-- For a rule with `v_t ≤ v̂_t` and `0 < β₂ < 1`, `v̂_{t,i} = 0` forces
`m_{t,i} = 0`. -/
theorem m_eq_zero_of_vhat {S : Setup d} {R : Rule d} (hR : ∀ t a b, b ≤ R t a b)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) {n : ℕ} {i : Fin d}
    (h : Real.sqrt (S.vhat R n i) = 0) : S.m R n i = 0 := by
  cases n with
  | zero => rfl
  | succ n =>
    refine m_eq_zero hβ₂ hβ₂' R ?_
    have h0 : 0 ≤ S.v R (n + 1) i := v_nonneg hβ₂.le hβ₂'.le R (n + 1) i
    have h1 : S.v R (n + 1) i ≤ S.vhat R (n + 1) i := hR (n + 1) (S.vhat R n) (S.v R (n + 1)) i
    have h2 : S.vhat R (n + 1) i ≤ 0 := Real.sqrt_eq_zero'.mp h
    linarith

/-- The weighted projection does not increase the weighted distance to a point
of `F`: `Σᵢ wᵢ(Π(y)ᵢ - x*ᵢ)² ≤ Σᵢ wᵢ(yᵢ - x*ᵢ)²` for `w ≥ 0`, `x* ∈ F`, `F`
convex.  arXiv:1904.03590v4, §3, proof of Lemma 3.1, via Lemma 2.7. -/
theorem proj_le {F : Set (Vec d)} {proj : Vec d → Vec d → Vec d} (hP : IsWeightedProj F proj)
    (hF : Convex ℝ F) {w : Vec d} (hw : ∀ i, 0 ≤ w i) (y : Vec d) {z : Vec d} (hz : z ∈ F) :
    ∑ i, w i * (proj w y i - z i) ^ 2 ≤ ∑ i, w i * (y i - z i) ^ 2 := by
  set p := proj w y
  have hp := hP.mem w y hw
  set A := ∑ i, w i * (p i - y i) * (z i - p i)
  set B := ∑ i, w i * (z i - p i) ^ 2
  have hB : 0 ≤ B := sum_nonneg fun i _ => mul_nonneg (hw i) (sq_nonneg _)
  -- The variational inequality `A ≥ 0`, along the segment from `p` to `z`.
  have hs : ∀ s : ℝ, 0 < s → s ≤ 1 → 0 ≤ 2 * A + s * B := by
    intro s hs0 hs1
    have hmem : p + s • (z - p) ∈ F := by
      have := hF hz hp (a := s) (b := 1 - s) hs0.le (by linarith) (by ring)
      convert this using 1; ext i; simp; ring
    have h := hP.le w y hw _ hmem
    have e : ∑ i, w i * ((p + s • (z - p)) i - y i) ^ 2 =
        ∑ i, w i * (p i - y i) ^ 2 + s * (2 * A + s * B) := by
      simp only [A, B, mul_sum, ← sum_add_distrib]
      refine sum_congr rfl fun i _ => ?_
      simp; ring
    rw [e] at h
    nlinarith
  have hA : 0 ≤ A := by
    by_contra hA
    push Not at hA
    rcases hB.eq_or_lt with hB0 | hB0
    · have := hs 1 one_pos le_rfl; rw [← hB0] at this; linarith
    · have := hs (min 1 (-A / B)) (lt_min one_pos (div_pos (by linarith) hB0)) (min_le_left _ _)
      have hm : min 1 (-A / B) * B ≤ -A := by
        calc min 1 (-A / B) * B ≤ -A / B * B := by gcongr; exact min_le_right _ _
          _ = -A := div_mul_cancel₀ _ hB0.ne'
      linarith
  have e : ∑ i, w i * (y i - z i) ^ 2 =
      ∑ i, w i * (p i - z i) ^ 2 + ∑ i, w i * (y i - p i) ^ 2 + 2 * A := by
    simp only [A, mul_sum, ← sum_add_distrib]
    refine sum_congr rfl fun i _ => ?_
    ring
  rw [e]
  nlinarith [sum_nonneg fun i (_ : i ∈ univ) => mul_nonneg (hw i) (sq_nonneg (y i - p i))]

/-- `mz ≤ w z²/(2a) + a m²/(2w)` for `a > 0`, `w ≥ 0`, and `m = 0` where
`w = 0`: the inequality `ab ≤ a²/2 + b²/2` of the proof of Lemma 3.1.
arXiv:1904.03590v4, §3, proof of Lemma 3.1. -/
theorem young {a w m z : ℝ} (ha : 0 < a) (hw : 0 ≤ w) (hm : w = 0 → m = 0) :
    m * z ≤ w / (2 * a) * z ^ 2 + a * m ^ 2 / (2 * w) := by
  rcases hw.eq_or_lt with hw0 | hw0
  · subst hw0; rw [hm rfl]; simp
  · rw [div_mul_eq_mul_div, div_add_div _ _ (by positivity) (by positivity),
      le_div_iff₀ (by positivity)]
    nlinarith [sq_nonneg (w * z - a * m), mul_pos ha hw0]

/-- **(tvheq).**  For a rule with `v_t ≤ v̂_t`, `α_t > 0`, `0 ≤ β_{1,t} < 1` and
`0 < β₂ < 1`, for `t ≥ 1` and `x* ∈ F`,

  `Σᵢ g_{t,i}(x_{t,i} - x*ᵢ) ≤ Σᵢ [√v̂_{t,i}/(2α_t(1-β_{1,t})) ((x_{t,i} - x*ᵢ)² - (x_{t+1,i} - x*ᵢ)²)`
  `    + α_t/(2(1-β_{1,t})) m²_{t,i}/√v̂_{t,i} + β_{1,t}/(1-β_{1,t}) m_{t-1,i}(x*ᵢ - x_{t,i})]`.

Source: arXiv:1904.03590v4, §3, proof of Lemma 3.1, (tvheq). -/
theorem step_ineq {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {R : Rule d} (hR : ∀ t a b, b ≤ R t a b) {t : ℕ} (ht : 1 ≤ t) (hα : 0 < S.α t)
    (hb : 0 ≤ S.β₁ t) (hb' : S.β₁ t < 1) (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    {xstar : Vec d} (hx : xstar ∈ F) :
    ∑ i, S.g R t i * (S.x R t i - xstar i) ≤
      ∑ i, (Real.sqrt (S.vhat R t i) / (2 * S.α t * (1 - S.β₁ t)) *
          ((S.x R t i - xstar i) ^ 2 - (S.x R (t + 1) i - xstar i) ^ 2)
        + S.α t / (2 * (1 - S.β₁ t)) * S.m R t i ^ 2 / Real.sqrt (S.vhat R t i)
        + S.β₁ t / (1 - S.β₁ t) * S.m R (t - 1) i * (xstar i - S.x R t i)) := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  set a := S.α (n + 1)
  set b := S.β₁ (n + 1)
  set w : Vec d := vsqrt (S.vhat R (n + 1))
  set y : Vec d := S.x R (n + 1) - a • (S.m R (n + 1) / w)
  have hx1 : S.x R (n + 1 + 1) = S.proj w y := rfl
  have hp := proj_le hS.proj hS.convex (w := w) (fun i => Real.sqrt_nonneg _) y hx
  rw [← hx1] at hp
  have h1b : 0 < 1 - b := by linarith
  set P : Fin d → ℝ := fun i => S.g R (n + 1) i * (S.x R (n + 1) i - xstar i)
  set Q : Fin d → ℝ := fun i => w i / (2 * a * (1 - b)) *
      ((S.x R (n + 1) i - xstar i) ^ 2 - (S.x R (n + 1 + 1) i - xstar i) ^ 2)
    + a / (2 * (1 - b)) * S.m R (n + 1) i ^ 2 / w i
    + b / (1 - b) * S.m R n i * (xstar i - S.x R (n + 1) i)
  have key : ∀ i, 2 * a * (1 - b) * P i + w i * (y i - xstar i) ^ 2 =
      2 * a * (1 - b) * Q i + w i * (S.x R (n + 1 + 1) i - xstar i) ^ 2 := by
    intro i
    have hm : S.m R (n + 1) i = b * S.m R n i + (1 - b) * S.g R (n + 1) i := rfl
    have hy : y i = S.x R (n + 1) i - a * (S.m R (n + 1) i / w i) := rfl
    have e1 : (1 - b) * (b / (1 - b)) = b := mul_div_cancel₀ _ h1b.ne'
    simp only [P, Q]
    rw [hy]
    by_cases hw : w i = 0
    · have hm0 := m_eq_zero_of_vhat hR hβ₂ hβ₂' hw
      rw [hw, hm0]
      rw [hm0] at hm
      simp only [div_zero, mul_zero, zero_mul, add_zero, zero_div, zero_pow two_ne_zero]
      linear_combination (-2 * a * (S.x R (n + 1) i - xstar i)) * hm
        + (2 * a * (S.x R (n + 1) i - xstar i) * S.m R n i) * e1
    · rw [hm]
      field_simp
      ring
  have hsum := sum_congr rfl fun i (_ : i ∈ univ) => key i
  rw [sum_add_distrib, sum_add_distrib, ← mul_sum, ← mul_sum] at hsum
  show ∑ i, P i ≤ ∑ i, Q i
  refine le_of_mul_le_mul_left ?_ (by positivity : 0 < 2 * a * (1 - b))
  linarith

/-- The hypotheses of `v_nonneg`, `m_eq_zero`, `m_eq_zero_of_vhat`, `proj_le`
and `step_ineq` are satisfiable: the zero cost on `[-1, 1]` under the rule
`v̂_t = v_t`, `α_t = 1`, `β_{1,t} = 0`, `β₂ = 1/2`, `t = 1`, `x* = w = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧
      Convex ℝ (Set.Icc (fun _ => -1 : Vec 1) (fun _ => 1)) ∧ (∀ i, (0 : Vec 1) i ≤ 0) ∧
      (∀ t (a b : Vec 1), b ≤ (fun _ _ b => b : Rule 1) t a b) ∧ 1 ≤ 1 ∧ 0 < S.α 1 ∧
      0 ≤ S.β₁ 1 ∧ S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧
      S.v (fun _ _ b => b) 0 0 = 0 ∧ Real.sqrt (S.vhat (fun _ _ b => b) 0 0) = 0 ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, convex_Icc _ _, fun _ => le_rfl, fun _ _ _ => le_rfl, le_rfl,
    one_pos, le_rfl, by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    rfl, show Real.sqrt 0 = 0 from Real.sqrt_zero, fun _ => by norm_num, fun _ => by norm_num⟩

/-- The hypotheses of `young` are satisfiable: `a = w = 1`, `m = 0`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ ((1 : ℝ) = 0 → (0 : ℝ) = 0) :=
  ⟨one_pos, zero_le_one, fun _ => rfl⟩

end AMSGrad
end Transformer
