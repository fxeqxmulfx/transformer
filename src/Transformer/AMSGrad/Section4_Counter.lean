import Transformer.AMSGrad.Section3_Example
import Mathlib.Data.Nat.Log

/-
# AMSGrad — the average regret need not tend to zero

§4 of arXiv:1904.03590v4, Corollary 4.5, asserts `lim_{T→∞} R(T)/T = 0` under
the assumptions of Theorem 4.1, with `R(T)` the regret against the minimizer
`x*` of `Σ_{t≤T} f_t` over `F`.  Theorem 4.1 bounds `R(T)` from above only,
and the online iterates may beat every fixed point by a linear margin, so the
limit can be negative.

**The counterexample.**  `d = 1`, `F = [-1, 1]`, `f_t(x) = s_t x` with
`s_t = +1` on `[2^k, 2^{k+1})` for even `k` and `-1` for odd `k`, AMSGrad with
`α_t = 1/√t`, `β_{1,t} = 0` (both settings of Theorem 4.1 with `β₁ = 0`),
`β₂ = 1/2`, `x₁ = 0`.  Each block is as long as the time elapsed, while the
iterate needs only `O(√t)` steps to cross `F`, so it spends almost all of each
block at the endpoint where `f_t = -1`: `Σ_{t≤T} f_t(x_t) ≈ -T`.  At the end of
a block `|Σ_{t≤T} s_t| ≈ T/3`, so `R(T) ≈ -T + T/3 = -2T/3` for every
`x* ∈ F`.  Numerically `R(T)/T = -0.664` at `T = 2²² - 1`.

`signSetup` satisfies every hypothesis of Corollary 4.5 in both settings,
`isOnlineConvex_sign` and `sign_hyp`; `not_cor_lower`, in
`Section4_CounterRegret`, proves `R(T) ≤ -T/2` for infinitely many `T` and
every `x* ∈ F`.

Source: arXiv:1904.03590v4, §4, Corollary 4.5.
-/

namespace Transformer
namespace AMSGrad

/-- `s_t = +1` on the blocks `[2^k, 2^{k+1})` with `k` even, `-1` with `k` odd. -/
noncomputable def blockSign (t : ℕ) : ℝ := if Nat.log 2 t % 2 = 0 then 1 else -1

/-- The run of the counterexample to Corollary 4.5. -/
noncomputable def signSetup : Setup 1 :=
  ⟨fun _ => boxProj (-1) 1, fun t x => blockSign t * x 0, fun t => 1 / Real.sqrt t,
    fun _ => 0, 1 / 2, fun _ => 0⟩

/-- The counterexample satisfies the standing assumptions, with `D = 2`, `G = 1`. -/
theorem isOnlineConvex_sign :
    IsOnlineConvex signSetup (Set.Icc (fun _ => -1) (fun _ => 1)) 2 1 where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [signSetup], fun _ => by norm_num [signSetup]⟩
  convexOn t := ⟨convex_univ, fun x _ y _ a b _ _ _ => le_of_eq (by
    simp only [signSetup, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring)⟩
  differentiable t y := ((hasFDerivAt_apply (𝕜 := ℝ) 0 y).const_mul _).differentiableAt
  diam x hx y hy i := by
    have := hx.1 i; have := hx.2 i; have := hy.1 i; have := hy.2 i
    rw [abs_le]; constructor <;> linarith
  grad_le t x _ i := by
    change |grad (fun x : Vec 1 => blockSign t * x 0) x i| ≤ 1
    rw [grad_linear]
    unfold blockSign; split_ifs <;> norm_num

/-- The counterexample satisfies the remaining hypotheses of Corollary 4.5, in
both settings: `α = 1`, `β₁ = 0` (with `λ = 1/2`), `β₂ = 1/2`. -/
theorem sign_hyp :
    signSetup.α = (fun t : ℕ => 1 / Real.sqrt t) ∧
      (∀ t, signSetup.β₁ t = 0 * (1 / 2 : ℝ) ^ (t - 1)) ∧ (∀ t, signSetup.β₁ t = 0 / t) ∧
      0 < signSetup.β₂ ∧ signSetup.β₂ < 1 ∧ 0 / Real.sqrt signSetup.β₂ < 1 :=
  ⟨rfl, fun _ => by simp [signSetup], fun _ => by simp [signSetup], by norm_num [signSetup],
    by norm_num [signSetup], by simp⟩

end AMSGrad
end Transformer
