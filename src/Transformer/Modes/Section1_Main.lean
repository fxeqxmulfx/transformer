/-
# The number of modes of a Gaussian KDE — the main result

§1.1 of arXiv:2412.09080v3, `thm:main-result`: in the regime
`n^c ≲ β ≲ n^{2-c}`, the expected number of modes of `eq:gkde` over all of `ℝ`
is `Θ(√(β log β))`, and almost all of them sit in the belt
`t² ∈ [2 log n - 3 log β, 2 log n - log β]`.

**What the source says and what is carried here.**

* The regime is `IsRegime c N B`: sample sizes `N k`, bandwidth parameters
  `B k`, both running to infinity, with `n^c ≲ β ≲ n^{2-c}`.  "For arbitrarily
  small `c > 0`" is the quantifier the user of the theorem supplies: the
  theorems below hold for each `c > 0` at once.

* Point 1 is `mainResult_expectedModes`, point 2 is
  `mainResult_expectedModes_compl_belt`.

* "Almost all modes lie in **two intervals** of length `Θ(√(log β))`" is the
  prose reading of point 2; the shape and the length of the belt are
  `Section1_Belt.lean`.

Source: arXiv:2412.09080v3, `thm:main-result`, `rem:minimax`, the belt-width
remark of §1.1.
-/

import Transformer.Modes.Section1_Mammen

open Filter Asymptotics
open scoped Topology ENNReal

namespace Transformer
namespace Modes

/-! ### The regime `n^c ≲ β ≲ n^{2-c}` -/

/-- **The regime of `thm:main-result`.**  `n^c ≲ β ≲ n^{2-c}` for a fixed
`c > 0`, as `n, β → ∞`, read along a sequence of sample sizes `N k` and
bandwidth parameters `B k`.

Source: arXiv:2412.09080v3, `thm:main-result`. -/
structure IsRegime (c : ℝ) (N : ℕ → ℕ) (B : ℕ → ℝ) : Prop where
  /-- The exponent is positive: `c > 0`. -/
  c_pos : 0 < c
  /-- The bandwidth parameter is a positive real. -/
  B_pos : ∀ k, 0 < B k
  /-- The sample size runs to infinity. -/
  tendsto_N : Tendsto (fun k => (N k : ℝ)) atTop atTop
  /-- The bandwidth parameter runs to infinity. -/
  tendsto_B : Tendsto B atTop atTop
  /-- `n^c ≲ β`. -/
  lower : (fun k => (N k : ℝ) ^ c) =O[atTop] B
  /-- `β ≲ n^{2-c}`. -/
  upper : B =O[atTop] fun k => (N k : ℝ) ^ (2 - c)

/-- The regime is satisfiable: `β = n` with `c = 1`.

Source: arXiv:2412.09080v3, `thm:main-result`, the regime `n^c ≲ β ≲ n^{2-c}`. -/
theorem isRegime_succ : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) where
  c_pos := one_pos
  B_pos k := by positivity
  tendsto_N := by push_cast; exact tendsto_natSucc_atTop
  tendsto_B := by push_cast; exact tendsto_natSucc_atTop
  lower := by
    refine (isBigO_refl (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop).congr' ?_ (by rfl)
    filter_upwards with k
    rw [Real.rpow_one]
  upper := by
    refine (isBigO_refl (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop).congr' (by rfl) ?_
    filter_upwards with k
    rw [show (2 : ℝ) - 1 = 1 by norm_num, Real.rpow_one]

/-! ### Point 1: the number of modes over `ℝ` -/

/-- **Theorem (thm:main-result), point 1.**  In expectation over the sample,
the number of modes of `P̂_n` over `ℝ` is `Θ(√(β log β))`.

Not proved here; `prop:main-int` and `prop:main-tail` of §1.3 are the two
halves of the source's argument.

Source: arXiv:2412.09080v3, `thm:main-result`, point 1. -/
theorem mainResult_expectedModes (c : ℝ) (N : ℕ → ℕ) (B : ℕ → ℝ) (hreg : IsRegime c N B) :
    (fun k => expectedModesReal (B k) (N k) Set.univ) =Θ[atTop]
      fun k => Real.sqrt (B k * Real.log (B k)) := by
  sorry

/-! ### Point 2: the belt the modes live in -/

/-- The belt of `thm:main-result`: the `t` with
`t² ∈ [2 log n - 3 log β, 2 log n - log β]`.

Source: arXiv:2412.09080v3, `thm:main-result`, point 2. -/
def belt (n : ℕ) (β : ℝ) : Set ℝ :=
  {t : ℝ | t ^ 2 ∈ Set.Icc (2 * Real.log n - 3 * Real.log β) (2 * Real.log n - Real.log β)}

/-- **Theorem (thm:main-result), point 2.**  The expected number of modes
outside the belt is finite and `o(√(β log β))`: almost all modes lie in it.
Finiteness is stated, since `expectedModesReal` reads `∞` as `0` and an upper
bound on it alone would say nothing about an infinite expectation.

Not proved here.

Source: arXiv:2412.09080v3, `thm:main-result`, point 2. -/
theorem mainResult_expectedModes_compl_belt (c : ℝ) (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hreg : IsRegime c N B) :
    (∀ᶠ k in atTop, expectedModes (B k) (N k) (belt (N k) (B k))ᶜ ≠ ∞) ∧
    (fun k => expectedModesReal (B k) (N k) (belt (N k) (B k))ᶜ) =o[atTop]
      fun k => Real.sqrt (B k * Real.log (B k)) := by
  sorry

/-- The hypothesis of both points is satisfiable, by the witness of
`IsRegime`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) := isRegime_succ

end Modes
end Transformer
