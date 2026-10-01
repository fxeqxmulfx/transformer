import Transformer.AMSGrad.Section1_TheoremA
import Mathlib.Analysis.Calculus.FDeriv.Mul

/-
# AMSGrad — Example 3.2

§3 of arXiv:1904.03590v4, Example 3.2: Reddi et al.'s synthetic cost on
`F = [-1, 1]`, `f_t(x) = 1010x` if `t mod 101 = 1` and `-10x` otherwise, run
from `x₁ = 1` with `β₁ = 0.9`, `β_{1,t} = β₁λ^{t-1}`, `λ = 0.001`,
`β₂ = 0.999`, `α_t = 0.001/√t`.

**What the source says and what is carried here.**

* The source computes `x₂ ≈ 0.99684` and `x₃ ≈ 0.99706` to sixteen digits.
  Only the signs matter: `(x₁ - x*)² - (x₂ - x*)² > 0` and
  `(x₂ - x*)² - (x₃ - x*)² < 0` with `x* = -1`, `exa_sign_one` and
  `exa_sign_two`, proved from `x₂ < 1` and `x₂ < x₃`.

* The run satisfies the standing assumptions, `isOnlineConvex_exa`.

Source: arXiv:1904.03590v4, §3, Example 3.2.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-- The slope of the cost at time `t`: `1010` if `t mod 101 = 1`, else `-10`. -/
def exaCoef (t : ℕ) : ℝ := if t % 101 = 1 then 1010 else -10

/-- The run of Example 3.2.  arXiv:1904.03590v4, §3, Example 3.2. -/
noncomputable def exaSetup : Setup 1 :=
  ⟨fun _ => boxProj (-1) 1, fun t x => exaCoef t * x 0, fun t => 0.001 / Real.sqrt t,
    fun t => 0.9 * 0.001 ^ (t - 1), 0.999, fun _ => 1⟩

/-- The gradient of a linear cost on `ℝ`. -/
theorem grad_linear (c : ℝ) (y : Vec 1) : grad (fun x : Vec 1 => c * x 0) y = fun _ => c := by
  have h : HasFDerivAt (fun x : Vec 1 => c * x 0) (c • ContinuousLinearMap.proj (0 : Fin 1)) y :=
    (hasFDerivAt_apply (𝕜 := ℝ) 0 y).const_mul c
  funext i
  fin_cases i
  simp [grad, h.fderiv]

/-- Example 3.2 satisfies the standing assumptions, with `D = 2`, `G = 1010`. -/
theorem isOnlineConvex_exa :
    IsOnlineConvex exaSetup (Set.Icc (fun _ => -1) (fun _ => 1)) 2 1010 where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [exaSetup], fun _ => le_rfl⟩
  convexOn t := ⟨convex_univ, fun x _ y _ a b _ _ _ => le_of_eq (by
    simp only [exaSetup, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring)⟩
  differentiable t y := ((hasFDerivAt_apply (𝕜 := ℝ) 0 y).const_mul _).differentiableAt
  diam x hx y hy i := by
    have := hx.1 i; have := hx.2 i; have := hy.1 i; have := hy.2 i
    rw [abs_le]; constructor <;> linarith
  grad_le t x _ i := by
    change |grad (fun x : Vec 1 => exaCoef t * x 0) x i| ≤ 1010
    rw [grad_linear]
    unfold exaCoef; split_ifs <;> norm_num

/-! ### The first two steps -/

/-- `y₁ = x₁ - α₁m₁/√v̂₁`, the point projected to give `x₂`. -/
noncomputable def exaY₁ : ℝ := 1 - 0.001 * (101 / Real.sqrt 1020.1)

/-- The first step: `m₁ = 101`, `v̂₁ = 1020.1`, `x₂ = Π_F(y₁)`.
arXiv:1904.03590v4, §3, Example 3.2. -/
theorem exa_state_one :
    (exaSetup.state amsgradRule 1).m 0 = 101 ∧ (exaSetup.state amsgradRule 1).vhat 0 = 1020.1 ∧
      (exaSetup.state amsgradRule 1).x 0 = max (-1) (min 1 exaY₁) := by
  have hg : grad (exaSetup.f 1) (fun _ => 1) = fun _ => 1010 := by
    change grad (fun x : Vec 1 => exaCoef 1 * x 0) _ = _
    rw [grad_linear]; norm_num [exaCoef]
  simp only [Setup.state, Setup.step, exaSetup] at hg ⊢
  rw [hg]
  refine ⟨by norm_num, by simp [amsgradRule]; norm_num, ?_⟩
  simp [amsgradRule, boxProj, vsqrt, exaY₁]
  norm_num

/-- `-1 < y₁ < 1`, so `x₂ = y₁`. -/
theorem exaY₁_mem : -1 < exaY₁ ∧ exaY₁ < 1 := by
  have hs : 1 < Real.sqrt 1020.1 := by
    rw [Real.lt_sqrt zero_le_one]; norm_num
  have hpos : 0 < 101 / Real.sqrt 1020.1 := by positivity
  have hlt : 101 / Real.sqrt 1020.1 < 101 := div_lt_self (by norm_num) hs
  unfold exaY₁
  constructor <;> nlinarith

/-- `x₂ = y₁ ∈ (-1, 1)`. -/
theorem exa_x_two : exaSetup.x amsgradRule 2 0 = exaY₁ := by
  have h := exa_state_one.2.2
  rw [min_eq_right exaY₁_mem.2.le, max_eq_right exaY₁_mem.1.le] at h
  exact h

/-- The second step: `x₃ = Π_F(x₂ - α₂m₂/√v̂₂)` with `m₂ < 0` and `v̂₂ > 0`.
arXiv:1904.03590v4, §3, Example 3.2. -/
theorem exa_state_two : ∃ m v : ℝ, m < 0 ∧ 0 < v ∧
    exaSetup.x amsgradRule 3 0 =
      max (-1) (min 1 (exaY₁ - 0.001 / Real.sqrt 2 * (m / Real.sqrt v))) := by
  obtain ⟨hm, hv, -⟩ := exa_state_one
  have hx := exa_x_two
  have hg : ∀ y : Vec 1, grad (exaSetup.f 2) y = fun _ => -10 := fun y => by
    change grad (fun x : Vec 1 => exaCoef 2 * x 0) _ = _
    rw [grad_linear]; norm_num [exaCoef]
  simp only [Setup.x] at hx
  show ∃ m v : ℝ, m < 0 ∧ 0 < v ∧
    (exaSetup.step amsgradRule 2 (exaSetup.state amsgradRule 1)).x 0 = _
  simp only [Setup.step]
  rw [hg]
  refine ⟨(exaSetup.β₁ 2 • (exaSetup.state amsgradRule 1).m
      + (1 - exaSetup.β₁ 2) • (fun _ => -10 : Vec 1)) 0,
    amsgradRule 2 (exaSetup.state amsgradRule 1).vhat
      (exaSetup.β₂ • (exaSetup.state amsgradRule 1).v
        + (1 - exaSetup.β₂) • (fun _ => -10 : Vec 1) ^ 2) 0, ?_, ?_, ?_⟩
  · have hβ : exaSetup.β₁ 2 = 0.9 * 0.001 := by simp [exaSetup]
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, hm, hβ]; norm_num
  · simp only [amsgradRule, Pi.sup_apply, hv]
    exact lt_sup_of_lt_left (by norm_num)
  · have hx' : (exaSetup.state amsgradRule 1).x 0 = exaY₁ := hx
    have hα : exaSetup.α 2 = 0.001 / Real.sqrt 2 := by simp [exaSetup]
    have hp : exaSetup.proj = fun _ => boxProj (-1) 1 := rfl
    rw [hp]
    simp only [boxProj, Pi.sub_apply, Pi.smul_apply, Pi.div_apply, smul_eq_mul, vsqrt, hα, hx']

/-- `x₂ < x₃`: the second step moves right, since `m₂ < 0`. -/
theorem exa_x_two_lt : exaSetup.x amsgradRule 2 0 < exaSetup.x amsgradRule 3 0 := by
  obtain ⟨m, v, hm, hv, h3⟩ := exa_state_two
  rw [h3, exa_x_two]
  have : 0 < -(0.001 / Real.sqrt 2 * (m / Real.sqrt v)) := by
    rw [neg_pos]
    exact mul_neg_of_pos_of_neg (by positivity) (div_neg_of_neg_of_pos hm (by positivity))
  exact lt_max_of_lt_right (lt_min exaY₁_mem.2 (by linarith))

/-- **Example 3.2, first sign.**  `(x₁ - x*)² - (x₂ - x*)² > 0` with `x* = -1`.
Source: arXiv:1904.03590v4, §3, Example 3.2. -/
theorem exa_sign_one :
    0 < (exaSetup.x amsgradRule 1 0 + 1) ^ 2 - (exaSetup.x amsgradRule 2 0 + 1) ^ 2 := by
  have h1 : exaSetup.x amsgradRule 1 0 = 1 := rfl
  rw [h1, exa_x_two]
  nlinarith [exaY₁_mem.1, exaY₁_mem.2]

/-- **Example 3.2, second sign.**  `(x₂ - x*)² - (x₃ - x*)² < 0` with `x* = -1`.
Source: arXiv:1904.03590v4, §3, Example 3.2. -/
theorem exa_sign_two :
    (exaSetup.x amsgradRule 2 0 + 1) ^ 2 - (exaSetup.x amsgradRule 3 0 + 1) ^ 2 < 0 := by
  have h := exa_x_two_lt
  have h2 : -1 < exaSetup.x amsgradRule 2 0 := exa_x_two ▸ exaY₁_mem.1
  nlinarith

end AMSGrad
end Transformer
