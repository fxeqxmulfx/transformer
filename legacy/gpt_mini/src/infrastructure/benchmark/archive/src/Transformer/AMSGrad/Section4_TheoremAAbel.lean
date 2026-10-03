import Transformer.AMSGrad.Section3_Issue

/-
# AMSGrad — Theorem A: Abel summation with a varying `β_{1,t}`

The scalar inequalities behind the repair of Theorem A of arXiv:1904.03590v4
(`Section4_TheoremAFinite`).  In Lemma 3.1 the first term carries the weights
`c_t = a_t/(2(1-β_{1,t}))`, `a_t = √v̂_{t,i}/α_t`, which are not monotone when
`β_{1,t}` varies, so `abel_le` of `Section3_Issue` does not apply; together with
the `β_{1,t}` term of Lemma 3.1, one coordinate at a time, with
`u_t = (x_{t,i} - x*ᵢ)² ∈ [0, E]`:

* `abel_general`, for every schedule `0 ≤ β_{1,t} ≤ β`:
  `≤ E a_T/2 + E/(1-β) Σ_{t=1}^T β_{1,t} a_t`;
* `abel_antitone`, for a non-increasing schedule, where `c_t - c_{t-1} ≤
  (a_t - a_{t-1})/(2(1-β))`:
  `≤ E a_T/(2(1-β)) + E/(2(1-β)) Σ_{t=2}^T β_{1,t} a_{t-1}`;
* `not_abel_printed`: with the constants the source prints, `E a_T/(1-β)` and
  `E/(2(1-β)) Σ_{t=1}^T β_{1,t} a_t`, the scalar inequality is false for a schedule
  that alternates, so `abel_general` cannot be brought down to them.

Source: not a statement of arXiv:1904.03590v4; the estimates its Theorem A
needs, where the proof of Reddi et al. quoted there fails.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-- Abel summation against a non-decreasing weight: for `a_t ≥ 0` non-decreasing and
`0 ≤ u_t ≤ E`, `Σ_{t=1}^T a_t (u_t - u_{t+1}) + a_T u_{T+1} ≤ E a_T`. -/
theorem sum_mul_sub_le (a u : ℕ → ℝ) {E : ℝ} (ha : ∀ t, 1 ≤ t → 0 ≤ a t)
    (hmono : ∀ t, 2 ≤ t → a (t - 1) ≤ a t) (hu : ∀ t, 1 ≤ t → 0 ≤ u t ∧ u t ≤ E)
    {T : ℕ} (hT : 1 ≤ T) :
    ∑ t ∈ Icc 1 T, a t * (u t - u (t + 1)) + a T * u (T + 1) ≤ E * a T := by
  induction T, hT using Nat.le_induction with
  | base =>
    simp only [Icc_self, sum_singleton]
    nlinarith [mul_le_mul_of_nonneg_left (hu 1 le_rfl).2 (ha 1 le_rfl)]
  | succ n hn ih =>
    rw [sum_Icc_succ_top (by omega)]
    have h1 := hmono (n + 1) (by omega)
    rw [Nat.add_sub_cancel] at h1
    have h2 := mul_le_mul_of_nonneg_left (hu (n + 1) (by omega)).2 (sub_nonneg.2 h1)
    nlinarith

/-- The hypotheses of `sum_mul_sub_le` are satisfiable: `a = 0`, `u = 0`. -/
example : (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t, 2 ≤ t → (fun _ : ℕ => (0 : ℝ)) (t - 1) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t ∧ (fun _ : ℕ => (0 : ℝ)) t ≤ 0) ∧ 1 ≤ 1 :=
  ⟨fun _ _ => le_rfl, fun _ _ => le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩, le_rfl⟩

/-- **Abel summation with a varying `β_t`, every schedule.**  For `a_t ≥ 0` non-decreasing,
`0 ≤ β_t ≤ β < 1` and `0 ≤ u_t ≤ E`,

  `Σ_{t=1}^T a_t/(2(1-β_t)) (u_t - u_{t+1}) + Σ_{t=2}^T β_t a_{t-1}/(2(1-β)) u_t`
  `  ≤ E a_T/2 + E/(1-β) Σ_{t=1}^T β_t a_t`. -/
theorem abel_general (a b u : ℕ → ℝ) {β E : ℝ} (hβ : β < 1)
    (ha : ∀ t, 1 ≤ t → 0 ≤ a t) (hmono : ∀ t, 2 ≤ t → a (t - 1) ≤ a t)
    (hb : ∀ t, 1 ≤ t → 0 ≤ b t ∧ b t ≤ β) (hu : ∀ t, 1 ≤ t → 0 ≤ u t ∧ u t ≤ E)
    {T : ℕ} (hT : 1 ≤ T) :
    ∑ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1))
        + ∑ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t ≤
      E * a T / 2 + E / (1 - β) * ∑ t ∈ Icc 1 T, b t * a t := by
  have hk : 0 < 1 - β := by linarith
  have h1 : ∀ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1)) ≤
      a t / 2 * (u t - u (t + 1)) + E / (2 * (1 - β)) * (b t * a t) := by
    intro t ht
    have ht1 := (mem_Icc.1 ht).1
    obtain ⟨hb0, hb1⟩ := hb t ht1
    obtain ⟨hu0, huE⟩ := hu t ht1
    have hu1 := (hu (t + 1) (by omega)).1
    have hat := ha t ht1
    have hbt : 0 < 1 - b t := by linarith
    have e : a t / (2 * (1 - b t)) = a t / 2 + a t * b t / (2 * (1 - b t)) := by
      field_simp; ring
    have hq0 : 0 ≤ a t * b t / (2 * (1 - b t)) := by positivity
    have hq1 : a t * b t / (2 * (1 - b t)) ≤ a t * b t / (2 * (1 - β)) :=
      div_le_div_of_nonneg_left (mul_nonneg hat hb0) (by positivity) (by linarith)
    have hq2 : a t * b t / (2 * (1 - b t)) * (u t - u (t + 1)) ≤
        E / (2 * (1 - β)) * (b t * a t) :=
      calc _ ≤ a t * b t / (2 * (1 - b t)) * E := mul_le_mul_of_nonneg_left (by linarith) hq0
        _ ≤ a t * b t / (2 * (1 - β)) * E := mul_le_mul_of_nonneg_right hq1 (hu0.trans huE)
        _ = E / (2 * (1 - β)) * (b t * a t) := by ring
    rw [e, add_mul]
    linarith
  have h2 : ∀ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t ≤
      E / (2 * (1 - β)) * (b t * a t) := by
    intro t ht
    have ht2 := (mem_Icc.1 ht).1
    obtain ⟨hb0, hb1⟩ := hb t (by omega)
    obtain ⟨hu0, huE⟩ := hu t (by omega)
    have hat := ha (t - 1) (by omega)
    calc b t * a (t - 1) / (2 * (1 - β)) * u t
        ≤ b t * a (t - 1) / (2 * (1 - β)) * E :=
          mul_le_mul_of_nonneg_left huE (by positivity)
      _ ≤ b t * a t / (2 * (1 - β)) * E :=
          mul_le_mul_of_nonneg_right (div_le_div_of_nonneg_right
            (mul_le_mul_of_nonneg_left (hmono t ht2) hb0) (by positivity)) (hu0.trans huE)
      _ = E / (2 * (1 - β)) * (b t * a t) := by ring
  have hs1 := sum_le_sum h1
  have hs2 := sum_le_sum h2
  rw [sum_add_distrib, ← mul_sum] at hs1
  rw [← mul_sum] at hs2
  have hs3 : ∑ t ∈ Icc 2 T, b t * a t ≤ ∑ t ∈ Icc 1 T, b t * a t :=
    sum_le_sum_of_subset_of_nonneg (Icc_subset_Icc_left one_le_two) fun t ht _ =>
      mul_nonneg (hb t (mem_Icc.1 ht).1).1 (ha t (mem_Icc.1 ht).1)
  have hA := sum_mul_sub_le a u ha hmono hu hT
  have e : ∑ t ∈ Icc 1 T, a t / 2 * (u t - u (t + 1)) =
      1 / 2 * ∑ t ∈ Icc 1 T, a t * (u t - u (t + 1)) := by
    rw [mul_sum]; exact sum_congr rfl fun t _ => by ring
  have hE : 0 ≤ E := (hu 1 le_rfl).1.trans (hu 1 le_rfl).2
  have hκ : E / (1 - β) = 2 * (E / (2 * (1 - β))) := by field_simp
  have hκ0 : 0 ≤ E / (2 * (1 - β)) := by positivity
  have hu1 := (hu (T + 1) (by omega)).1
  have hAT : 0 ≤ a T * u (T + 1) := mul_nonneg (ha T hT) hu1
  rw [hκ]
  rw [e] at hs1
  generalize E / (2 * (1 - β)) = κ at *
  nlinarith [mul_le_mul_of_nonneg_left hs3 hκ0]

/-- The hypotheses of `abel_general` are satisfiable: all sequences zero. -/
example : (0 : ℝ) < 1 ∧ (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t, 2 ≤ t → (fun _ : ℕ => (0 : ℝ)) (t - 1) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t ∧ (fun _ : ℕ => (0 : ℝ)) t ≤ 0) ∧
    (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t ∧ (fun _ : ℕ => (0 : ℝ)) t ≤ 0) ∧ 1 ≤ 1 :=
  ⟨one_pos, fun _ _ => le_rfl, fun _ _ => le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    fun _ _ => ⟨le_rfl, le_rfl⟩, le_rfl⟩

/-- **Abel summation with a non-increasing `β_t`.**  For `a_t ≥ 0` non-decreasing,
`0 ≤ β_t ≤ β < 1` non-increasing and `0 ≤ u_t ≤ E`,

  `Σ_{t=1}^T a_t/(2(1-β_t)) (u_t - u_{t+1}) + Σ_{t=2}^T β_t a_{t-1}/(2(1-β)) u_t`
  `  ≤ E/(2(1-β)) (a_T + Σ_{t=2}^T β_t a_{t-1})`. -/
theorem abel_antitone (a b u : ℕ → ℝ) {β E : ℝ} (hβ : β < 1)
    (ha : ∀ t, 1 ≤ t → 0 ≤ a t) (hmono : ∀ t, 2 ≤ t → a (t - 1) ≤ a t)
    (hb : ∀ t, 1 ≤ t → 0 ≤ b t ∧ b t ≤ β) (hanti : ∀ t, 2 ≤ t → b t ≤ b (t - 1))
    (hu : ∀ t, 1 ≤ t → 0 ≤ u t ∧ u t ≤ E) {T : ℕ} (hT : 1 ≤ T) :
    ∑ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1))
        + ∑ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t ≤
      E / (2 * (1 - β)) * (a T + ∑ t ∈ Icc 2 T, b t * a (t - 1)) := by
  have hk : 0 < 1 - β := by linarith
  have hE : 0 ≤ E := (hu 1 le_rfl).1.trans (hu 1 le_rfl).2
  obtain ⟨κ, hκ⟩ : ∃ κ, κ = E / (2 * (1 - β)) := ⟨_, rfl⟩
  rw [← hκ]
  have key : ∀ T, 1 ≤ T → ∑ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1))
        + ∑ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t
        + a T / (2 * (1 - b T)) * u (T + 1) ≤
      κ * (a T + ∑ t ∈ Icc 2 T, b t * a (t - 1)) := by
    intro T hT
    induction T, hT using Nat.le_induction with
    | base =>
      have h1 : 0 < 1 - b 1 := by linarith [(hb 1 le_rfl).2]
      have ha1 := ha 1 le_rfl
      have hc : a 1 / (2 * (1 - b 1)) ≤ a 1 / (2 * (1 - β)) :=
        div_le_div_of_nonneg_left ha1 (by positivity) (by linarith [(hb 1 le_rfl).2])
      have hc0 : 0 ≤ a 1 / (2 * (1 - b 1)) := by positivity
      have hq : a 1 / (2 * (1 - b 1)) * u 1 ≤ κ * a 1 :=
        calc a 1 / (2 * (1 - b 1)) * u 1 ≤ a 1 / (2 * (1 - b 1)) * E :=
              mul_le_mul_of_nonneg_left (hu 1 le_rfl).2 hc0
          _ ≤ a 1 / (2 * (1 - β)) * E := mul_le_mul_of_nonneg_right hc hE
          _ = κ * a 1 := by rw [hκ]; ring
      have e2 : Icc 2 1 = ∅ := by decide
      rw [e2, Icc_self, sum_empty, sum_singleton]
      nlinarith [hq]
    | succ n hn ih =>
      rw [sum_Icc_succ_top (by omega), sum_Icc_succ_top (by omega), sum_Icc_succ_top (by omega)]
      have hm := hmono (n + 1) (by omega)
      have han := hanti (n + 1) (by omega)
      simp only [Nat.add_sub_cancel] at hm han ⊢
      obtain ⟨hb0, hb1⟩ := hb (n + 1) (by omega)
      obtain ⟨hbn0, hbn1⟩ := hb n hn
      have han0 := ha n hn
      have hu' := hu (n + 1) (by omega)
      have hp : 0 < 1 - b (n + 1) := by linarith
      have hq : 0 < 1 - b n := by linarith
      have hw : a (n + 1) / (2 * (1 - b (n + 1))) - a n / (2 * (1 - b n))
            + b (n + 1) * a n / (2 * (1 - β)) ≤
          (a (n + 1) - a n + b (n + 1) * a n) / (2 * (1 - β)) := by
        have h1 : a n / (2 * (1 - b (n + 1))) ≤ a n / (2 * (1 - b n)) :=
          div_le_div_of_nonneg_left han0 (by positivity) (by linarith)
        have h2 : (a (n + 1) - a n) / (2 * (1 - b (n + 1))) ≤ (a (n + 1) - a n) / (2 * (1 - β)) :=
          div_le_div_of_nonneg_left (sub_nonneg.2 hm) (by positivity) (by linarith)
        have e : a (n + 1) / (2 * (1 - b (n + 1))) - a n / (2 * (1 - b (n + 1))) =
            (a (n + 1) - a n) / (2 * (1 - b (n + 1))) := by ring
        have e' : (a (n + 1) - a n + b (n + 1) * a n) / (2 * (1 - β)) =
            (a (n + 1) - a n) / (2 * (1 - β)) + b (n + 1) * a n / (2 * (1 - β)) := by ring
        linarith
      have hU : 0 ≤ (a (n + 1) - a n + b (n + 1) * a n) / (2 * (1 - β)) := by
        have := sub_nonneg.2 hm
        positivity
      have hwu : (a (n + 1) / (2 * (1 - b (n + 1))) - a n / (2 * (1 - b n))
            + b (n + 1) * a n / (2 * (1 - β))) * u (n + 1) ≤
          κ * (a (n + 1) - a n + b (n + 1) * a n) :=
        calc _ ≤ (a (n + 1) - a n + b (n + 1) * a n) / (2 * (1 - β)) * u (n + 1) :=
                mul_le_mul_of_nonneg_right hw hu'.1
          _ ≤ (a (n + 1) - a n + b (n + 1) * a n) / (2 * (1 - β)) * E :=
                mul_le_mul_of_nonneg_left hu'.2 hU
          _ = κ * (a (n + 1) - a n + b (n + 1) * a n) := by rw [hκ]; ring
      nlinarith [hwu, ih]
  have h := key T hT
  have hnn : 0 ≤ a T / (2 * (1 - b T)) * u (T + 1) :=
    mul_nonneg (div_nonneg (ha T hT) (by linarith [(hb T hT).2])) (hu (T + 1) (by omega)).1
  linarith

/-- The hypotheses of `abel_antitone` are satisfiable: all sequences zero. -/
example : (0 : ℝ) < 1 ∧ (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t, 2 ≤ t → (fun _ : ℕ => (0 : ℝ)) (t - 1) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t ∧ (fun _ : ℕ => (0 : ℝ)) t ≤ 0) ∧
    (∀ t, 2 ≤ t → (fun _ : ℕ => (0 : ℝ)) t ≤ (fun _ : ℕ => 0) (t - 1)) ∧
    (∀ t, 1 ≤ t → (0 : ℝ) ≤ (fun _ : ℕ => 0) t ∧ (fun _ : ℕ => (0 : ℝ)) t ≤ 0) ∧ 1 ≤ 1 :=
  ⟨one_pos, fun _ _ => le_rfl, fun _ _ => le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    fun _ _ => le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩, le_rfl⟩

/-- **The printed constants are out of reach of the scalar bound.**  For arbitrary
`a_t ≥ 0` non-decreasing, `0 ≤ β_t ≤ β < 1` and `0 ≤ u_t ≤ E`, the inequality of
`abel_general` with the constants the source prints for Theorem A,

  `Σ_{t=1}^T a_t/(2(1-β_t)) (u_t - u_{t+1}) + Σ_{t=2}^T β_t a_{t-1}/(2(1-β)) u_t`
  `  ≤ E a_T/(1-β) + E/(2(1-β)) Σ_{t=1}^T β_t a_t`,

is false: `a ≡ 1`, `β = 1/2`, `β_t = 1/2` for even `t` and `0` for odd `t`, `u_t = 1` for
even `t` and `0` for odd `t`, `E = 1`, `T = 10` give `5 ≤ 4.5`.  A proof of Theorem A with
the printed constant for a schedule that is not monotone must therefore use more than
`0 ≤ u_t ≤ E` and the monotonicity of `a_t`. -/
theorem not_abel_printed :
    ¬ ∀ (a b u : ℕ → ℝ) (β E : ℝ), β < 1 → (∀ t, 1 ≤ t → 0 ≤ a t) →
      (∀ t, 2 ≤ t → a (t - 1) ≤ a t) → (∀ t, 1 ≤ t → 0 ≤ b t ∧ b t ≤ β) →
      (∀ t, 1 ≤ t → 0 ≤ u t ∧ u t ≤ E) → ∀ T : ℕ, 1 ≤ T →
      ∑ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1))
          + ∑ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t ≤
        E * a T / (1 - β) + E / (2 * (1 - β)) * ∑ t ∈ Icc 1 T, b t * a t := by
  intro h
  have := h (fun _ => 1) (fun t => if t % 2 = 0 then 1 / 2 else 0)
    (fun t => if t % 2 = 0 then 1 else 0) (1 / 2) 1 (by norm_num) (fun _ _ => zero_le_one)
    (fun _ _ => le_rfl) (fun t _ => by split_ifs <;> norm_num)
    (fun t _ => by split_ifs <;> norm_num) 10 (by norm_num)
  norm_num [Finset.sum_Icc_succ_top] at this

end AMSGrad
end Transformer
