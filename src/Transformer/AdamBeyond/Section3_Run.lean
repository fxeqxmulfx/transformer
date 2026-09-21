import Transformer.AdamBeyond.Section2_Adam
import Transformer.AMSGrad.Section3_Example

/-
# Adam and beyond — §3: the run behind Theorems 1 and 6

The one-dimensional problem of the proof of Theorem 1: `F = [-1, 1]`,
`f_t(x) = C x` for `t mod 3 = 1` and `-x` otherwise, Adam with `β₁ = 0`, started
at `x₁ = 1`.  Every third step the gradient `C` is scaled down by `√v_t ≈ C`,
while the two gradients `-1` in between are scaled by `√v_t ≈ 1`, so the
iterate returns to `x = 1`, the worst point of `F`, after every block of three
steps (`cex_block`).

**What the source says and what is carried here.**

* The update with `ε` of eq:mod-update is the rule `adamEpsRule ε`,
  `v̂_t = v_t + ε`.  The projection is then weighted by `√(v_t + ε)`, not by
  `√v_t`; in one dimension every weighted projection onto `[-1, 1]` is the
  clamp, so the run is the same.  `adamEpsRule 0` is Adam.

* The run is scaled: `f_t(x) = 3s x` or `-s x`, `C = 3`, `β₂ = 1/100`, with
  `0 ≤ ε ≤ s²/100`; `s = 1, ε = 0` is Theorem 1's setting and `s = 10√ε` is
  Theorem 6's rescaling.  The source's constants, `β₂ = 1/(1 + C²)` for
  Theorem 1 and `β₂ = 2/((1 + C²)C²)` for Theorem 6, rest on
  `1/√(3t+3) ≥ 1/√(2(3t+1))`, which fails at `t = 0`; the constants here leave
  room for every block, the first included.

* `cex_block` takes any step sizes `α_t ≥ 0` with
  `α_{3k+1}·30/29 ≤ 1` and `α_{3k+1}·30/29 ≤ α_{3k+2}·20/21 + α_{3k+3}·100/101`;
  both `α_t = α/√t` and the constant `α_t = α` of the remark after Theorem 1
  satisfy them for `α = 1/2`.

Source: arXiv:1904.09237, §3, eq:mod-update; Appendix, proofs of Theorem 1
and Theorem 6.
-/

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- Adam with `ε` in the denominator: `v̂_t = v_t + ε`.
arXiv:1904.09237, §3, eq:mod-update. -/
def adamEpsRule (ε : ℝ) : Rule d := fun _ _ b => b + fun _ => ε

/-- `ε = 0` is Adam. -/
theorem adamEpsRule_zero : adamEpsRule (d := d) 0 = adamRule := by
  funext t a b i
  simp [adamEpsRule, adamRule]

/-- The slope of `f_t`: `3s` if `t mod 3 = 1`, else `-s`. -/
noncomputable def slope (s : ℝ) (t : ℕ) : ℝ := if t % 3 = 1 then 3 * s else -s

/-- The problem of Theorems 1 and 6, with step sizes `α`.
arXiv:1904.09237, Appendix, proof of Theorem 1. -/
noncomputable def cexSetup (s : ℝ) (α : ℕ → ℝ) : Setup 1 :=
  ⟨fun _ => boxProj (-1) 1, fun t x => slope s t * x 0, α, fun _ => 0, 1 / 100, fun _ => 1⟩

/-- The problem is an online convex one, with `D = 2`, `G = 3s`. -/
theorem isOnlineConvex_cex {s : ℝ} (hs : 0 ≤ s) (α : ℕ → ℝ) :
    IsOnlineConvex (cexSetup s α) (Set.Icc (fun _ => -1) (fun _ => 1)) 2 (3 * s) where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [cexSetup], fun _ => le_rfl⟩
  convexOn t := ⟨convex_univ, fun x _ y _ a b _ _ _ => le_of_eq (by
    simp only [cexSetup, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring)⟩
  differentiable t y := ((hasFDerivAt_apply (𝕜 := ℝ) 0 y).const_mul _).differentiableAt
  diam x hx y hy i := by
    have := hx.1 i; have := hx.2 i; have := hy.1 i; have := hy.2 i
    rw [abs_le]; constructor <;> linarith
  grad_le t x _ i := by
    change |grad (fun x : Vec 1 => slope s t * x 0) x i| ≤ 3 * s
    rw [grad_linear]
    unfold slope; split_ifs
    · rw [abs_of_nonneg (by linarith)]
    · rw [abs_neg, abs_of_nonneg hs]; linarith

/-- One step of the run: `v`. -/
theorem cex_v (s ε : ℝ) (α : ℕ → ℝ) (t : ℕ) (st : State 1) :
    ((cexSetup s α).step (adamEpsRule ε) t st).v 0 = 1 / 100 * st.v 0 + 99 / 100 * slope s t ^ 2 := by
  simp [Setup.step, cexSetup, grad_linear]
  norm_num

/-- One step of the run: `x`. -/
theorem cex_x (s ε : ℝ) (α : ℕ → ℝ) (t : ℕ) (st : State 1) :
    ((cexSetup s α).step (adamEpsRule ε) t st).x 0 = max (-1) (min 1
      (st.x 0 - α t * (slope s t / Real.sqrt (((cexSetup s α).step (adamEpsRule ε) t st).v 0 + ε)))) := by
  simp [Setup.step, cexSetup, grad_linear, boxProj, adamEpsRule, vsqrt]

/-- One block of three steps, as arithmetic: from `x = 1` and `0 ≤ v ≤ 9s²`, back to
`x = 1` and `0 ≤ v ≤ 9s²`. -/
theorem block_arith {s ε v α₁ α₂ α₃ v₁ v₂ v₃ x₁ x₂ x₃ : ℝ} (hs : 0 < s) (hε : 0 ≤ ε)
    (hε' : ε ≤ s ^ 2 / 100) (hv : 0 ≤ v) (hv' : v ≤ 9 * s ^ 2) (h₁ : 0 ≤ α₁) (h₂ : 0 ≤ α₂)
    (h₃ : 0 ≤ α₃) (hα₁ : α₁ * (30 / 29) ≤ 1)
    (hα : α₁ * (30 / 29) ≤ α₂ * (20 / 21) + α₃ * (100 / 101))
    (ev₁ : v₁ = 1 / 100 * v + 99 / 100 * (3 * s) ^ 2)
    (ex₁ : x₁ = max (-1) (min 1 (1 - α₁ * (3 * s / Real.sqrt (v₁ + ε)))))
    (ev₂ : v₂ = 1 / 100 * v₁ + 99 / 100 * (-s) ^ 2)
    (ex₂ : x₂ = max (-1) (min 1 (x₁ - α₂ * (-s / Real.sqrt (v₂ + ε)))))
    (ev₃ : v₃ = 1 / 100 * v₂ + 99 / 100 * (-s) ^ 2)
    (ex₃ : x₃ = max (-1) (min 1 (x₂ - α₃ * (-s / Real.sqrt (v₃ + ε))))) :
    x₃ = 1 ∧ 0 ≤ v₃ ∧ v₃ ≤ 9 * s ^ 2 := by
  have hs2 : 0 < s ^ 2 := by positivity
  -- the first step, down by `a₁ ≤ α₁·30/29`
  have r₁ : 29 / 10 * s ≤ Real.sqrt (v₁ + ε) :=
    Real.le_sqrt_of_sq_le (by rw [ev₁]; nlinarith)
  have r₁0 : 0 < Real.sqrt (v₁ + ε) := by linarith
  have a₁le : 3 * s / Real.sqrt (v₁ + ε) ≤ 30 / 29 := by
    rw [div_le_iff₀ r₁0]; linarith
  have a₁0 : 0 ≤ 3 * s / Real.sqrt (v₁ + ε) := by positivity
  have hA₁ : α₁ * (3 * s / Real.sqrt (v₁ + ε)) ≤ α₁ * (30 / 29) :=
    mul_le_mul_of_nonneg_left a₁le h₁
  have hA₁0 : 0 ≤ α₁ * (3 * s / Real.sqrt (v₁ + ε)) := mul_nonneg h₁ a₁0
  have ex₁' : x₁ = 1 - α₁ * (3 * s / Real.sqrt (v₁ + ε)) := by
    rw [ex₁, min_eq_right (by linarith), max_eq_right (by linarith)]
  -- the second and third steps, up by `a₂ ≥ α₂·20/21` and `a₃ ≥ α₃·100/101`
  have r₂ : Real.sqrt (v₂ + ε) ≤ 21 / 20 * s := by
    rw [Real.sqrt_le_left (by positivity)]; rw [ev₂, ev₁]; nlinarith
  have r₂0 : 0 < Real.sqrt (v₂ + ε) := Real.sqrt_pos.mpr (by rw [ev₂, ev₁]; nlinarith)
  have a₂ : 20 / 21 ≤ s / Real.sqrt (v₂ + ε) := by
    rw [le_div_iff₀ r₂0]; linarith
  have r₃ : Real.sqrt (v₃ + ε) ≤ 101 / 100 * s := by
    rw [Real.sqrt_le_left (by positivity)]; rw [ev₃, ev₂, ev₁]; nlinarith
  have r₃0 : 0 < Real.sqrt (v₃ + ε) := Real.sqrt_pos.mpr (by rw [ev₃, ev₂, ev₁]; nlinarith)
  have a₃ : 100 / 101 ≤ s / Real.sqrt (v₃ + ε) := by
    rw [le_div_iff₀ r₃0]; linarith
  have hA₂ : α₂ * (20 / 21) ≤ α₂ * (s / Real.sqrt (v₂ + ε)) := mul_le_mul_of_nonneg_left a₂ h₂
  have hA₃ : α₃ * (100 / 101) ≤ α₃ * (s / Real.sqrt (v₃ + ε)) := mul_le_mul_of_nonneg_left a₃ h₃
  have hA₂0 : 0 ≤ α₂ * (20 / 21) := by positivity
  have hA₃0 : 0 ≤ α₃ * (100 / 101) := by positivity
  rw [neg_div] at ex₂ ex₃
  simp only [mul_neg, sub_neg_eq_add] at ex₂ ex₃
  have ex₂' : x₂ = min 1 (x₁ + α₂ * (s / Real.sqrt (v₂ + ε))) := by
    rw [ex₂, max_eq_right (le_min (by norm_num) (by linarith))]
  refine ⟨?_, ?_, ?_⟩
  · rw [ex₃, max_eq_right (le_min (by norm_num) (by
      rw [ex₂']; exact le_add_of_le_of_nonneg (le_min (by norm_num) (by linarith)) (by linarith))),
      min_eq_left]
    rw [ex₂']
    rcases le_total 1 (x₁ + α₂ * (s / Real.sqrt (v₂ + ε))) with h | h
    · rw [min_eq_left h]; linarith
    · rw [min_eq_right h]; linarith
  · rw [ev₃, ev₂, ev₁]; positivity
  · rw [ev₃, ev₂, ev₁]; nlinarith

/-- **The main claim of the proof of Theorem 1.**  After every block of three
steps the iterate is back at `x_{3k+1} = 1`, with `0 ≤ v_{3k} ≤ 9s²`.
arXiv:1904.09237, Appendix, proof of Theorem 1. -/
theorem cex_block {s ε : ℝ} {α : ℕ → ℝ} (hs : 0 < s) (hε : 0 ≤ ε) (hε' : ε ≤ s ^ 2 / 100)
    (hα₀ : ∀ t, 0 ≤ α t) (hα₁ : ∀ k, α (3 * k + 1) * (30 / 29) ≤ 1)
    (hα : ∀ k, α (3 * k + 1) * (30 / 29) ≤ α (3 * k + 2) * (20 / 21) + α (3 * k + 3) * (100 / 101))
    (k : ℕ) : ((cexSetup s α).state (adamEpsRule ε) (3 * k)).x 0 = 1 ∧
      0 ≤ ((cexSetup s α).state (adamEpsRule ε) (3 * k)).v 0 ∧
      ((cexSetup s α).state (adamEpsRule ε) (3 * k)).v 0 ≤ 9 * s ^ 2 := by
  induction k with
  | zero => simp [Setup.state, cexSetup]; positivity
  | succ k ih =>
    obtain ⟨hx, hv, hv'⟩ := ih
    have e3 : 3 * (k + 1) = 3 * k + 2 + 1 := by ring
    have s₁ : slope s (3 * k + 1) = 3 * s := by simp [slope, Nat.add_mod]
    have s₂ : slope s (3 * k + 2) = -s := by simp [slope, Nat.add_mod]
    have s₃ : slope s (3 * k + 2 + 1) = -s := by
      simp only [slope, show (3 * k + 2 + 1) % 3 = 0 by omega]; norm_num
    set S := cexSetup s α
    set R : Rule 1 := adamEpsRule ε
    have e₁ : S.state R (3 * k + 1) = S.step R (3 * k + 1) (S.state R (3 * k)) := rfl
    have e₂ : S.state R (3 * k + 2) = S.step R (3 * k + 2) (S.state R (3 * k + 1)) := rfl
    have e₃ : S.state R (3 * k + 2 + 1) = S.step R (3 * k + 2 + 1) (S.state R (3 * k + 2)) :=
      rfl
    rw [e3]
    refine block_arith (v₁ := (S.state R (3 * k + 1)).v 0) (x₁ := (S.state R (3 * k + 1)).x 0)
      (v₂ := (S.state R (3 * k + 2)).v 0) (x₂ := (S.state R (3 * k + 2)).x 0)
      hs hε hε' hv hv' (hα₀ _) (hα₀ _) (hα₀ _) (hα₁ k) (hα k) ?_ ?_ ?_ ?_ ?_ ?_
    · rw [e₁, cex_v, s₁]
    · rw [e₁, cex_x, s₁, hx]
    · rw [e₂, cex_v, s₂, e₁]
    · rw [e₂, cex_x, s₂, e₁]
    · rw [e₃, cex_v, s₃, e₂]
    · rw [e₃, cex_x, s₃, e₂, show 3 * k + 2 + 1 = 3 * k + 3 by ring]

end AdamBeyond
end Transformer
