import Transformer.AMSGrad.Section4_TheoremASparse

/-
# AMSGrad — Theorem A's constants for an upward-jumping schedule

Theorem A of arXiv:1904.03590v4, §1, allows every schedule bounded by its
initial value. The proof in `Section4_TheoremASparse` gives the printed regret
bound for an actual AMSGrad run at `T = 3` with
`β₁,₁ = β₁,₃ = 1/2` and `β₁,₂ = 0`. Thus non-increasing schedules are not the
only ones for which the printed constants can be proved. This is a special
case, not a proof of the source's claim for every bounded schedule.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-- A bounded first-moment schedule with an upward jump at time three.
Source: arXiv:1904.03590v4, §1, Theorem A (example of its hypotheses). -/
noncomputable def bumpBeta (t : ℕ) : ℝ := if t = 1 ∨ t = 3 then 1 / 2 else 0

/-- A nonzero-gradient example using `bumpBeta` and the costs of §3, Example 3.2.
Source: arXiv:1904.03590v4, §1, Theorem A; §3, Example 3.2. -/
noncomputable def bumpSetup : Setup 1 := { exaSetup with β₁ := bumpBeta }

/-- Changing only the first-moment schedule preserves the standing assumptions.
Source: arXiv:1904.03590v4, §1, Theorem A (hypotheses). -/
theorem isOnlineConvex_bump :
    IsOnlineConvex bumpSetup (Set.Icc (fun _ => -1) (fun _ => 1)) 2 1010 where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [bumpSetup, exaSetup], fun _ => by norm_num [bumpSetup, exaSetup]⟩
  convexOn t := isOnlineConvex_exa.convexOn t
  differentiable t := isOnlineConvex_exa.differentiable t
  diam := isOnlineConvex_exa.diam
  grad_le t x hx i := isOnlineConvex_exa.grad_le t x hx i

/-- **Theorem A at three steps for an upward-jumping schedule.** If
`β₁,t = 1/2` at `t = 1, 3` and is zero otherwise, the exact printed regret
bound holds at `T = 3` for every feasible comparator. The source's Theorem A
asserts this for arbitrary `T` and every bounded schedule; neither follows
from this special case.

Source: arXiv:1904.03590v4, §1, Theorem A. -/
theorem theorem_A_bump {d : ℕ} {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ : S.β₁ = bumpBeta)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar 3 ≤
      D ^ 2 * Real.sqrt 3 / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule 3 i)
      + D ^ 2 / (2 * (1 - S.β₁ 1)) *
          ∑ i, ∑ t ∈ Icc 1 3, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log 3) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule 3 i := by
  have hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1 := by
    intro t ht
    rw [hβ]
    have h1 : bumpBeta 1 = 1 / 2 := by norm_num [bumpBeta]
    rw [h1]
    unfold bumpBeta
    split_ifs <;> norm_num
  have hβ₁' : S.β₁ 1 < 1 := by rw [hβ]; norm_num [bumpBeta]
  apply theorem_A_sparse hS hα hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ (T := 3)
    (by norm_num) hxstar
  intro i
  let a : ℕ → ℝ := fun t => Real.sqrt (S.vhat amsgradRule t i) / S.α t
  have ha13 : a 1 ≤ a 3 :=
    (sqrt_vhat_div_mono hαt hα (t := 2) (by norm_num) i).trans
      (sqrt_vhat_div_mono hαt hα (t := 3) (by norm_num) i)
  have ha3 : 0 ≤ a 3 := by
    dsimp [a]
    apply div_nonneg (Real.sqrt_nonneg _)
    rw [hαt]
    positivity
  change ∑ t ∈ Icc 1 3, S.β₁ t * a t ≤ (1 + S.β₁ 1) * a 3
  rw [hβ]
  norm_num [bumpBeta, sum_Icc_succ_top] at ⊢
  linarith

/-- All hypotheses above are realized by a nonzero-gradient run with
`β₁,₂ = 0 < β₁,₃ = 1/2` and comparator `x* = -1`. -/
example := theorem_A_bump (S := bumpSetup) (F := Set.Icc (fun _ => -1) (fun _ => 1))
    (D := 2) (G := 1010) (α := 0.001) isOnlineConvex_bump (by norm_num) rfl
    rfl (by norm_num [bumpSetup, exaSetup]) (by norm_num [bumpSetup, exaSetup])
    (by
      change (1 / 2 : ℝ) / Real.sqrt 0.999 < 1
      rw [div_lt_one (Real.sqrt_pos.2 (by norm_num))]
      exact Real.lt_sqrt (by norm_num) |>.2 (by norm_num))
    (xstar := fun _ => -1) ⟨fun _ => le_rfl, fun _ => by norm_num⟩

/-- The example schedule really has an upward jump. -/
example : bumpSetup.β₁ 2 < bumpSetup.β₁ 3 := by
  norm_num [bumpSetup, bumpBeta]

/-- The example run has a nonzero gradient at the first step. -/
example : bumpSetup.g amsgradRule 1 0 = 1010 := by
  change grad (fun x : Vec 1 => exaCoef 1 * x 0) (bumpSetup.x amsgradRule 1) 0 = 1010
  rw [grad_linear]
  norm_num [exaCoef]

end AMSGrad
end Transformer
