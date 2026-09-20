/-
# The number of modes of a Gaussian KDE — the two intervals of the proof

§1.3 of arXiv:2412.09080v3, `sec: sketch`: the truncation `eq:T` of `ℝ` to an
interval `T` on which the Gaussian approximation is valid, the inner interval
`eq:T'`, and the two propositions the main theorem is assembled from.

**What the source says and what is carried here.**

* `ω` is "a fixed, slow growing function such that `1 ≪ ω(β) ≪ log log β`".
  `IsSlowGrowth` reads the left half as `ω → ∞` — the source's `≪` is a ratio
  going to zero, which for a *growing* `ω` is exactly that, and it is what the
  error terms `e^{-ω(β)/4}` are meant to kill.

* `prop:main-int` and `prop:main-tail` are the two propositions; both are
  unproved here.  `thm:main-result` "follows from them provided
  `1 ≪ ω(β) ≪ log log β`", and the arithmetic that makes the window the right
  one is proved: `e^{ω(β)/2}√β = o(√(β log β))` is `isLittleO_tail_sqrt` — the
  tail bound of `prop:main-tail` is negligible against the count in `T`
  precisely because `ω(β) ≪ log log β` — and `e^{-ω(β)/4} → 0` is
  `tendsto_exp_neg_omega`, which is what the other error terms need from
  `1 ≪ ω(β)`.

Source: arXiv:2412.09080v3, `sec: sketch`, `eq:T`, `eq:T'`, `prop:main-int`,
`prop:main-tail`.
-/

import Transformer.Modes.Section1_Main

open Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-! ### The slow-growing `ω` -/

/-- **`1 ≪ ω(β) ≪ log log β`**, the window the truncation `eq:T` is cut at.

Source: arXiv:2412.09080v3, `sec: sketch`. -/
structure IsSlowGrowth (ω : ℝ → ℝ) : Prop where
  /-- `1 ≪ ω(β)`, for a growing `ω`. -/
  lower : Tendsto ω atTop atTop
  /-- `ω(β) ≪ log log β`. -/
  upper : ω =o[atTop] fun β => Real.log (Real.log β)

/-- The window is not empty: `ω(β) = √(log log β)` sits in it. -/
example : IsSlowGrowth fun β => Real.sqrt (Real.log (Real.log β)) where
  lower := Real.tendsto_sqrt_atTop.comp (Real.tendsto_log_atTop.comp Real.tendsto_log_atTop)
  upper := by
    have hll : Tendsto (fun β : ℝ => Real.log (Real.log β)) atTop atTop :=
      Real.tendsto_log_atTop.comp Real.tendsto_log_atTop
    have h := (isLittleO_rpow_rpow_atTop (by norm_num : (1 : ℝ) / 2 < 1)).comp_tendsto hll
    simp only [Function.comp_def, Real.rpow_one] at h
    refine h.congr' ?_ (by rfl)
    filter_upwards with β
    rw [Real.sqrt_eq_rpow]

/-- **`e^{-ω(β)/4} → 0`.**  This is what the error terms of the Edgeworth
expansion need from the left half of the window.

Source: arXiv:2412.09080v3, `sec: sketch`, the closing paragraph. -/
theorem tendsto_exp_neg_omega {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {B : ℕ → ℝ}
    (hB : Tendsto B atTop atTop) :
    Tendsto (fun k => Real.exp (-(ω (B k)) / 4)) atTop (nhds 0) := by
  have h1 : Tendsto (fun k => ω (B k) / 4) atTop atTop := by
    have := (hω.lower.comp hB).atTop_div_const (by norm_num : (0 : ℝ) < 4)
    simpa [Function.comp_def] using this
  exact Real.tendsto_exp_atBot.comp (by simpa [neg_div] using h1)

/-- **`e^{ω(β)/2}√β = o(√(β log β))`.**  This is what the right half of the
window buys: the tail bound of `prop:main-tail` is negligible against the count
in `T` of `prop:main-int`, so that `thm:main-result` follows from the two.

Source: arXiv:2412.09080v3, `sec: sketch`, the closing paragraph. -/
theorem isLittleO_tail_sqrt {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {B : ℕ → ℝ}
    (hB : Tendsto B atTop atTop) :
    (fun k => Real.exp (ω (B k) / 2) * Real.sqrt (B k)) =o[atTop]
      fun k => Real.sqrt (B k * Real.log (B k)) := by
  have hlogB : Tendsto (fun k => Real.log (B k)) atTop atTop := Real.tendsto_log_atTop.comp hB
  have hbound : ∀ᶠ k in atTop, Real.exp (ω (B k) / 2) ≤ Real.log (B k) ^ ((1 : ℝ) / 4) := by
    have h1 : ∀ᶠ β in atTop, ‖ω β‖ ≤ (1 / 2) * ‖Real.log (Real.log β)‖ :=
      isLittleO_iff.mp hω.upper (by norm_num)
    filter_upwards [hB.eventually h1, hlogB.eventually_gt_atTop 1] with k hk hlog1
    have hpos : (0 : ℝ) < Real.log (B k) := by linarith
    have hll : 0 ≤ Real.log (Real.log (B k)) := Real.log_nonneg hlog1.le
    have hω' : ω (B k) ≤ (1 / 2) * Real.log (Real.log (B k)) :=
      (Real.le_norm_self _).trans (hk.trans_eq (by rw [Real.norm_of_nonneg hll]))
    calc Real.exp (ω (B k) / 2) ≤ Real.exp ((1 / 4) * Real.log (Real.log (B k))) :=
          Real.exp_le_exp.mpr (by linarith)
      _ = Real.log (B k) ^ ((1 : ℝ) / 4) := by rw [Real.rpow_def_of_pos hpos]; ring_nf
  have hO : (fun k => Real.exp (ω (B k) / 2) * Real.sqrt (B k)) =O[atTop]
      fun k => Real.log (B k) ^ ((1 : ℝ) / 4) * Real.sqrt (B k) := by
    refine IsBigO.of_bound 1 ?_
    filter_upwards [hbound, hlogB.eventually_ge_atTop 0] with k hk hlog0
    rw [one_mul, Real.norm_of_nonneg (by positivity),
      Real.norm_of_nonneg (mul_nonneg (Real.rpow_nonneg hlog0 _) (Real.sqrt_nonneg _))]
    exact mul_le_mul_of_nonneg_right hk (Real.sqrt_nonneg _)
  refine hO.trans_isLittleO ?_
  have h1 : (fun k => Real.log (B k) ^ ((1 : ℝ) / 4)) =o[atTop]
      fun k => Real.log (B k) ^ ((1 : ℝ) / 2) :=
    (isLittleO_rpow_rpow_atTop (by norm_num : (1 : ℝ) / 4 < 1 / 2)).comp_tendsto hlogB
  refine (h1.mul_isBigO (isBigO_refl (fun k => Real.sqrt (B k)) atTop)).congr' (by rfl) ?_
  filter_upwards [hB.eventually_ge_atTop 0, hlogB.eventually_ge_atTop 0] with k hBk hlogk
  rw [Real.sqrt_mul hBk, Real.sqrt_eq_rpow (Real.log (B k))]
  ring

/-! ### The two intervals -/

/-- **Equation (eq:T).**  `T = [-√(2 log n - log β - ω(β)), √(2 log n - log β - ω(β))]`,
the interval the Gaussian approximation is valid on.

Source: arXiv:2412.09080v3, `eq:T`. -/
def intervalT (n : ℕ) (β w : ℝ) : Set ℝ :=
  Set.Icc (-Real.sqrt (2 * Real.log n - Real.log β - w))
    (Real.sqrt (2 * Real.log n - Real.log β - w))

/-- **Equation (eq:T').**  `T' = [-√(2 log n - 3 log β), √(2 log n - 3 log β)]`
if `β ≤ n^{2/3}`, and `T' = ∅` otherwise.

Source: arXiv:2412.09080v3, `eq:T'`. -/
def intervalT' (n : ℕ) (β : ℝ) : Set ℝ :=
  if β ≤ (n : ℝ) ^ ((2 : ℝ) / 3) then
    Set.Icc (-Real.sqrt (2 * Real.log n - 3 * Real.log β))
      (Real.sqrt (2 * Real.log n - 3 * Real.log β))
  else ∅

/-! ### `prop:main-int` and `prop:main-tail` -/

/-- **Proposition (prop:main-int), point 1.**  In the regime `n^c ≲ β ≲ n^{2-c}`,
the expected number of modes of `P̂_n` in `T` is `Θ(√(β log β))`.

Not proved here.

Source: arXiv:2412.09080v3, `prop:main-int`. -/
theorem prop_main_int_T (c : ℝ) (N : ℕ → ℕ) (B : ℕ → ℝ) (hreg : IsRegime c N B)
    (ω : ℝ → ℝ) (hω : IsSlowGrowth ω) :
    (fun k => expectedModesReal (B k) (N k) (intervalT (N k) (B k) (ω (B k)))) =Θ[atTop]
      fun k => Real.sqrt (B k * Real.log (B k)) := by
  sorry

/-- **Proposition (prop:main-int), point 2.**  In the same regime, the expected
number of modes of `P̂_n` in `T'` is `O(√β)`.

Not proved here.

Source: arXiv:2412.09080v3, `prop:main-int`. -/
theorem prop_main_int_T' (c : ℝ) (N : ℕ → ℕ) (B : ℕ → ℝ) (hreg : IsRegime c N B) :
    (fun k => expectedModesReal (B k) (N k) (intervalT' (N k) (B k))) =O[atTop]
      fun k => Real.sqrt (B k) := by
  sorry

/-- **Proposition (prop:main-tail).**  In the same regime, the expected number
of modes of `P̂_n` outside `T` is `O(e^{ω(β)/2}√β)`.

Not proved here; `sec:tail` proves it by the scale-space argument.

Source: arXiv:2412.09080v3, `prop:main-tail`. -/
theorem prop_main_tail (c : ℝ) (N : ℕ → ℕ) (B : ℕ → ℝ) (hreg : IsRegime c N B)
    (ω : ℝ → ℝ) (hω : IsSlowGrowth ω) :
    (fun k => expectedModesReal (B k) (N k) (intervalT (N k) (B k) (ω (B k)))ᶜ) =O[atTop]
      fun k => Real.exp (ω (B k) / 2) * Real.sqrt (B k) := by
  sorry

/-- The hypotheses of the three propositions are satisfiable: the regime
`β = n` of `IsRegime`, and the window's witness `ω(β) = √(log log β)`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) := by
  refine ⟨⟨one_pos, fun k => by positivity, by push_cast; exact tendsto_natSucc_atTop,
    by push_cast; exact tendsto_natSucc_atTop, ?_, ?_⟩, ?_, ?_⟩
  · refine (isBigO_refl (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop).congr' ?_ (by rfl)
    filter_upwards with k
    rw [Real.rpow_one]
  · refine (isBigO_refl (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop).congr' (by rfl) ?_
    filter_upwards with k
    rw [show (2 : ℝ) - 1 = 1 by norm_num, Real.rpow_one]
  · exact Real.tendsto_sqrt_atTop.comp (Real.tendsto_log_atTop.comp Real.tendsto_log_atTop)
  · have hll : Tendsto (fun β : ℝ => Real.log (Real.log β)) atTop atTop :=
      Real.tendsto_log_atTop.comp Real.tendsto_log_atTop
    have h := (isLittleO_rpow_rpow_atTop (by norm_num : (1 : ℝ) / 2 < 1)).comp_tendsto hll
    simp only [Function.comp_def, Real.rpow_one] at h
    refine h.congr' ?_ (by rfl)
    filter_upwards with β
    rw [Real.sqrt_eq_rpow]

end Modes
end Transformer
