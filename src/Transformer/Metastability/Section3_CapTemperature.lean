module

public import Transformer.Causal.Interaction
public import Transformer.Metastability.CircleWitness
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# Small angular caps and the derivative of the interaction

ArXiv:2410.06833v1, §3.2, `eq: tau.small`, `claim: 1`, and the
intra-cluster and cross-cluster Hessian estimates preceding `eq: Ht.second.lb`.
The chart of a cap is an interval. The stated diameter condition bounds
`τ(1+β)` and makes the interaction derivative positive inside the cap.
Across distinct caps its possible negative part is exponentially small.
-/

@[expose] public section

open Set Real

namespace Transformer.Metastability

/-- The real chart of a circle cap contains the interval between any two
of its points. Source: arXiv:2410.06833v1, §3.2, `eq: cones_2`, in the
ordered-cluster argument of `claim: 1`. -/
theorem interval_mem_cos_cap (τ ω a b : ℝ) (ha : |a - ω| ≤ π) (hb : |b - ω| ≤ π)
    (hca : 1 - 2 * τ ≤ Real.cos (a - ω)) (hcb : 1 - 2 * τ ≤ Real.cos (b - ω)) :
    ∀ u ∈ Icc a b, |u - ω| ≤ π ∧ 1 - 2 * τ ≤ Real.cos (u - ω) := by
  intro u hu
  have hau := abs_le.mp ha
  have hbu := abs_le.mp hb
  refine ⟨abs_le.mpr ⟨by linarith [hu.1], by linarith [hu.2]⟩, ?_⟩
  by_cases huω : u ≤ ω
  · have hc := Real.cos_le_cos_of_nonneg_of_le_pi (sub_nonneg.mpr huω)
      (by linarith [hau.1] : ω - a ≤ π) (by linarith [hu.1] : ω - u ≤ ω - a)
    rw [show ω - a = -(a - ω) by ring, show ω - u = -(u - ω) by ring,
      Real.cos_neg, Real.cos_neg] at hc
    exact hca.trans hc
  · have hc := Real.cos_le_cos_of_nonneg_of_le_pi (by linarith : 0 ≤ u - ω)
      hbu.2 (by linarith [hu.2] : u - ω ≤ b - ω)
    exact hcb.trans hc

/-- The interval hypotheses hold at the center of a nonempty cap. -/
example : |(0 : ℝ) - 0| ≤ π ∧ |(0 : ℝ) - 0| ≤ π ∧
    1 - 2 * (1 / 100 : ℝ) ≤ Real.cos (0 - 0) := by
  norm_num
  exact Real.pi_pos.le

/-- **The source's small-cap assumption controls the temperature-width product.**
Testing it at `ω±2√τ`, which lie in the cap by the quadratic cosine bound,
gives `τ(1+β)≤1/512`. Source: arXiv:2410.06833v1, §3.2, `eq: tau.small`. -/
theorem small_cap_temperature_bound (β τ δ ω : ℝ) (hβ : 1 < β)
    (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hsmall : ∀ u v : ℝ, |u - ω| ≤ π → |v - ω| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω) → 1 - 2 * τ ≤ Real.cos (v - ω) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2))) :
    τ * (1 + β) ≤ 1 / 512 := by
  let a := 2 * Real.sqrt τ
  have ha0 : 0 ≤ a := by dsimp [a]; positivity
  have ha2 : a ^ 2 = 4 * τ := by dsimp [a]; nlinarith [Real.sq_sqrt hτ.le]
  have haπ : a ≤ π := by nlinarith [Real.pi_gt_three]
  have hac : 1 - 2 * τ ≤ Real.cos a := by
    nlinarith [Real.one_sub_sq_div_two_le_cos (x := a)]
  have hdiam := hsmall (ω + a) (ω - a)
    (by simpa only [add_sub_cancel_left, abs_of_nonneg ha0] using haπ)
    (by simpa only [sub_sub_cancel_left, abs_neg, abs_of_nonneg ha0] using haπ)
    (by simpa only [add_sub_cancel_left] using hac)
    (by simpa only [sub_sub_cancel_left, Real.cos_neg] using hac)
  have hdist : |ω + a - (ω - a)| = 2 * a := by
    rw [show ω + a - (ω - a) = 2 * a by ring, abs_of_nonneg (by positivity)]
  rw [hdist] at hdiam
  have hden : 0 < β + 1 / 2 := by linarith
  have hratio : 0 ≤ (1 - δ) / (β + 1 / 2) := div_nonneg (by linarith) hden.le
  have hsq := Real.sq_sqrt hratio
  have hbound : 1024 * τ ≤ (1 - δ) / (β + 1 / 2) := by
    nlinarith [Real.sqrt_nonneg ((1 - δ) / (β + 1 / 2))]
  have hprod := (le_div_iff₀ hden).mp hbound
  nlinarith [mul_nonneg hτ.le (by linarith : 0 ≤ β)]

/-- All temperature-width inputs, including the diameter hypothesis,
are realized by a cap of `circle_witness`. -/
example : (1 : ℝ) < 1000 ∧ (0 : ℝ) < 1 / 10 ^ 8 ∧ (1 / 10 ^ 8 : ℝ) < 1 / 16 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ u v : ℝ, |u - 0| ≤ π → |v - 0| ≤ π →
      1 - 2 * (1 / 10 ^ 8 : ℝ) ≤ Real.cos (u - 0) →
      1 - 2 * (1 / 10 ^ 8 : ℝ) ≤ Real.cos (v - 0) →
      |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) := by
  obtain ⟨-, -, -, hsmall, -⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, hsmall 0⟩

/-- **A within-cap interaction derivative is bounded below by `1/2`.**
The stronger diameter in `eq: tau.small` gives `(β+1/2)x²≤1/64`.
The quadratic cosine bound and `exp z≥1+z` then suffice.
Source: arXiv:2410.06833v1, §3.2, the estimate before `eq: Ht.second.lb`. -/
theorem interaction_derivative_ge_half (β x : ℝ) (hβ : 0 ≤ β)
    (hsmall : (β + 1 / 2) * x ^ 2 ≤ 1 / 64) :
    (1 / 2 : ℝ) ≤ Causal.g_pot β x := by
  have hcos := Real.one_sub_sq_div_two_le_cos (x := x)
  have hsin := Real.sin_sq_le_sq (x := x)
  have hfactor : 63 / 64 ≤ Real.cos x - β * Real.sin x ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hsin hβ]
  have hx2 : β * x ^ 2 ≤ 1 / 64 := by nlinarith [sq_nonneg x]
  have harg : -(1 / 128 : ℝ) ≤ β * (Real.cos x - 1) := by
    nlinarith [mul_le_mul_of_nonneg_left hcos hβ]
  have hexp : 127 / 128 ≤ Real.exp (β * (Real.cos x - 1)) := by
    linarith [Real.add_one_le_exp (β * (Real.cos x - 1))]
  have hmul := mul_le_mul hexp hfactor (by norm_num : (0 : ℝ) ≤ 63 / 64)
    (Real.exp_pos _).le
  unfold Causal.g_pot
  nlinarith

/-- The within-cap derivative hypotheses hold at `β=1`, `x=0`. -/
example : (0 : ℝ) ≤ 1 ∧ ((1 : ℝ) + 1 / 2) * 0 ^ 2 ≤ 1 / 64 := by norm_num

/-- **The negative part of a cross-cap derivative is exponentially small.**
Source: arXiv:2410.06833v1, §3.2, the cross-cap Hessian estimate preceding
`eq: Ht.first.lb`. -/
theorem interaction_derivative_cross_lower (β α x : ℝ) (hβ : 0 ≤ β)
    (hcos : Real.cos x ≤ α) :
    -((1 + β) * Real.exp (-((1 - α) * β))) ≤ Causal.g_pot β x := by
  have hfactor : -(1 + β) ≤ Real.cos x - β * Real.sin x ^ 2 := by
    nlinarith [Real.neg_one_le_cos x, mul_le_mul_of_nonneg_left (Real.sin_sq_le_one x) hβ]
  have hexp : Real.exp (β * (Real.cos x - 1)) ≤ Real.exp (-((1 - α) * β)) :=
    Real.exp_le_exp.mpr (by nlinarith)
  calc
    _ ≤ -(1 + β) * Real.exp (β * (Real.cos x - 1)) := by
      simpa only [neg_mul] using mul_le_mul_of_nonpos_left hexp (by linarith : -(1 + β) ≤ 0)
    _ ≤ _ := by
      simpa only [Causal.g_pot, mul_comm] using
        mul_le_mul_of_nonneg_right hfactor (Real.exp_pos _).le

/-- The cross-cap hypotheses hold for an antipodal difference at `β=1`. -/
example : (0 : ℝ) ≤ 1 ∧ Real.cos π ≤ -1 := by simp

/-- **The total possible cross-cap derivative is smaller than `1/4`.**
The positive separation parameter `γ` gives `2n² exp(-(1-α)β)<τ`.
Combined with the cap-width estimate above, this dominates all `n`
cross interactions. Source: arXiv:2410.06833v1, §3.2, `d: condition_ineq_2`
and the estimate preceding `eq: Ht.first.lb`. -/
theorem cross_interaction_count_lt (n : ℕ) (β τ α : ℝ) (hn : 0 < n)
    (hβ : 0 < β) (hτ : 0 < τ) (hwidth : τ * (1 + β) ≤ 1 / 512)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ)) :
    (n : ℝ) * (1 + β) * Real.exp (-((1 - α) * β)) < 1 / 4 := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hA : 0 < 2 * (n : ℝ) ^ 2 / τ := by positivity
  have hcancel : (β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ)) * β =
      Real.log (2 * (n : ℝ) ^ 2 / τ) := by
    rw [mul_right_comm, inv_mul_cancel₀ (ne_of_gt hβ), one_mul]
  have hlog : Real.log (2 * (n : ℝ) ^ 2 / τ) < (1 - α - 8 * τ) * β := by
    nlinarith [mul_pos hγ hβ]
  have hexp := Real.exp_lt_exp.mpr hlog
  rw [Real.exp_log hA] at hexp
  have hEA : Real.exp (-((1 - α) * β)) * (2 * (n : ℝ) ^ 2 / τ) < 1 := by
    apply (mul_lt_mul_of_pos_left hexp (Real.exp_pos _)).trans_le
    rw [← Real.exp_add]
    apply Real.exp_le_one_iff.mpr
    nlinarith [mul_pos hτ hβ]
  have hquot : (Real.exp (-((1 - α) * β)) * (2 * (n : ℝ) ^ 2 / τ)) * τ =
      2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) := by
    field_simp
  have hτbound := mul_lt_mul_of_pos_right hEA hτ
  rw [hquot, one_mul] at hτbound
  have hprod := mul_lt_mul_of_pos_right hτbound (by linarith : 0 < 1 + β)
  have hcount : (n : ℝ) * ((1 + β) * Real.exp (-((1 - α) * β))) ≤
      (n : ℝ) ^ 2 * ((1 + β) * Real.exp (-((1 - α) * β))) :=
    mul_le_mul_of_nonneg_right (by nlinarith : (n : ℝ) ≤ (n : ℝ) ^ 2) (by positivity)
  nlinarith

/-- All cross-count hypotheses hold for the separated three-token witness. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    0 < 3 ∧ (0 : ℝ) < 1000 ∧ 0 < τ ∧ τ * (1 + 1000) ≤ 1 / 512 ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) := by
  intro τ α
  obtain ⟨-, -, hγ, -⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, hγ⟩

end Transformer.Metastability
