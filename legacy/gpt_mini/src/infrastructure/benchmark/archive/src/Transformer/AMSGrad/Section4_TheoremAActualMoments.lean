import Transformer.AMSGrad.Section3_Example
import Transformer.AMSGrad.Section4_Lemmas

/-
# AMSGrad — an alternating actual run for the Theorem A proof audit

The schedule and linear costs below give an admissible one-dimensional AMSGrad
run. Its corrected second moment equals one after the first step, while its
momentum alternates in sign. The construction is used to test the scalar
estimate behind Theorem A, rather than the regret theorem itself.

Source: arXiv:1904.03590v4, §1, Algorithm 1 and Theorem A; §3, Lemma 3.1.
-/

open Finset
namespace Transformer.AMSGrad

/-- Linear cost coefficients of the concrete run used to audit the
Theorem A telescoping step. Source: arXiv:1904.03590v4, §1, Algorithm 1; §3. -/
noncomputable def flipCoef (t : ℕ) : ℝ :=
  if t = 1 then -4 / 3 else if t % 2 = 0 then 1 else -1 / 2

/-- The first-moment schedule equals `1/2` at the first and even steps and
zero at later odd steps. Source: arXiv:1904.03590v4, §1, Theorem A hypotheses. -/
noncomputable def flipBeta (t : ℕ) : ℝ :=
  if t = 1 ∨ t % 2 = 0 then 1 / 2 else 0

/-- The concrete AMSGrad setup on `[-1,1]`, with `α_t = 100/√t` and
`β₂ = 7/16`. Source: arXiv:1904.03590v4, §1, Algorithm 1; §3. -/
noncomputable def flipSetup : Setup 1 :=
  ⟨fun _ => boxProj (-1) 1, fun t x => flipCoef t * x 0,
    fun t => 100 / Real.sqrt t, flipBeta, 7 / 16, fun _ => -1⟩

/-- Its gradient is the cost coefficient. Source: arXiv:1904.03590v4, §1,
Algorithm 1. -/
private theorem flip_g (t : ℕ) : flipSetup.g amsgradRule t 0 = flipCoef t := by
  change grad (fun x : Vec 1 => flipCoef t * x 0) (flipSetup.x amsgradRule t) 0 = flipCoef t
  rw [grad_linear]

/-- The first second moment equals one. Source: arXiv:1904.03590v4, §1,
Algorithm 1. -/
private theorem flip_v_one : flipSetup.v amsgradRule 1 0 = 1 := by
  change (7 / 16 : ℝ) * 0 + (1 - 7 / 16) * (flipSetup.g amsgradRule 1 0) ^ 2 = 1
  rw [flip_g]
  norm_num [flipCoef]

/-- Later squared gradients are at most one, so every second moment is at
most one. Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_v_le_one (t : ℕ) : flipSetup.v amsgradRule t 0 ≤ 1 := by
  induction t with
  | zero => change (0 : ℝ) ≤ 1; norm_num
  | succ n ih =>
    by_cases h1 : n + 1 = 1
    · have : n = 0 := by omega
      subst n
      exact flip_v_one.le
    · have hg : flipCoef (n + 1) ^ 2 ≤ 1 := by
        simp only [flipCoef, show n + 1 ≠ 1 by omega, ite_false]
        split_ifs <;> norm_num
      change (7 / 16 : ℝ) * flipSetup.v amsgradRule n 0 +
        (1 - 7 / 16) * (flipSetup.g amsgradRule (n + 1) 0) ^ 2 ≤ 1
      rw [flip_g]
      nlinarith

/-- The corrected second moment stays exactly one from the first step.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
theorem flip_vhat_one {t : ℕ} (ht : 1 ≤ t) :
    flipSetup.vhat amsgradRule t 0 = 1 := by
  induction t, ht using Nat.le_induction with
  | base =>
    rw [vhat_succ]
    change max (0 : ℝ) (flipSetup.v amsgradRule 1 0) = 1
    rw [flip_v_one]
    norm_num
  | succ n hn ih =>
    rw [vhat_succ, ih]
    exact max_eq_left (flip_v_le_one (n + 1))

/-- First moment at step one. Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_m_one : flipSetup.m amsgradRule 1 0 = -2 / 3 := by
  change flipBeta 1 * 0 + (1 - flipBeta 1) * flipSetup.g amsgradRule 1 0 = -2 / 3
  rw [flip_g]
  norm_num [flipBeta, flipCoef]

/-- First moment at step two. Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_m_two : flipSetup.m amsgradRule 2 0 = 1 / 6 := by
  change flipBeta 2 * flipSetup.m amsgradRule 1 0 +
    (1 - flipBeta 2) * flipSetup.g amsgradRule 2 0 = 1 / 6
  rw [flip_g, flip_m_one]
  norm_num [flipBeta, flipCoef]

/-- Later odd steps reset the moment to `-1/2`.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_m_odd {t : ℕ} (ht : 3 ≤ t) (hodd : t % 2 = 1) :
    flipSetup.m amsgradRule t 0 = -1 / 2 := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  have hn0 : n ≠ 0 := by omega
  have hp : (n + 1) % 2 ≠ 0 := by omega
  have hb : flipBeta (n + 1) = 0 := by simp [flipBeta, hn0, hp]
  have hg : flipCoef (n + 1) = -1 / 2 := by simp [flipCoef, hn0, hp]
  change flipBeta (n + 1) * flipSetup.m amsgradRule n 0 +
    (1 - flipBeta (n + 1)) * flipSetup.g amsgradRule (n + 1) 0 = -1 / 2
  rw [flip_g, hb, hg]
  ring

/-- From step four, every even moment is `1/4`.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_m_even {t : ℕ} (ht : 4 ≤ t) (heven : t % 2 = 0) :
    flipSetup.m amsgradRule t 0 = 1 / 4 := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  have hn : 3 ≤ n := by omega
  have hp : n % 2 = 1 := by omega
  have hm := flip_m_odd hn hp
  have hb : flipBeta (n + 1) = 1 / 2 := by simp [flipBeta, heven]
  have hg : flipCoef (n + 1) = 1 := by simp [flipCoef, heven]; omega
  change flipBeta (n + 1) * flipSetup.m amsgradRule n 0 +
    (1 - flipBeta (n + 1)) * flipSetup.g amsgradRule (n + 1) 0 = 1 / 4
  rw [flip_g, hb, hg, hm]
  ring

/-- The moment has alternating sign and magnitude at least `1/6`.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
theorem flip_m_bounds {t : ℕ} (ht : 1 ≤ t) :
    (t % 2 = 0 → 1 / 6 ≤ flipSetup.m amsgradRule t 0) ∧
    (t % 2 = 1 → flipSetup.m amsgradRule t 0 ≤ -1 / 6) := by
  constructor
  · intro hp
    by_cases h2 : t = 2
    · subst t; rw [flip_m_two]
    · have ht4 : 4 ≤ t := by omega
      rw [flip_m_even ht4 hp]
      norm_num
  · intro hp
    by_cases h1 : t = 1
    · subst t; rw [flip_m_one]; norm_num
    · have ht3 : 3 ≤ t := by omega
      rw [flip_m_odd ht3 hp]
      norm_num

/-- The concrete run satisfies the standing convexity, diameter, and gradient
assumptions of Theorem A. Source: arXiv:1904.03590v4, §1, Theorem A. -/
theorem flip_isOnlineConvex :
    IsOnlineConvex flipSetup (Set.Icc (fun _ => -1) (fun _ => 1)) 2 (4 / 3) where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [flipSetup], fun _ => by norm_num [flipSetup]⟩
  convexOn t := ⟨convex_univ, fun x _ y _ a b _ _ _ => le_of_eq (by
    simp only [flipSetup, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring)⟩
  differentiable t y := ((hasFDerivAt_apply (𝕜 := ℝ) 0 y).const_mul _).differentiableAt
  diam x hx y hy i := by
    have := hx.1 i; have := hx.2 i; have := hy.1 i; have := hy.2 i
    rw [abs_le]; constructor <;> linarith
  grad_le t x _ i := by
    change |grad (fun x : Vec 1 => flipCoef t * x 0) x i| ≤ 4 / 3
    rw [grad_linear]
    unfold flipCoef
    split_ifs <;> norm_num

/-- The schedule starts at its maximum, remains in `[0,1/2]`, and satisfies
`β₁/√β₂ < 1`. Source: arXiv:1904.03590v4, §1, Theorem A. -/
theorem flip_params :
    flipSetup.β₁ 1 = (1 / 2 : ℝ) ∧
    (∀ t, 1 ≤ t → 0 ≤ flipSetup.β₁ t ∧ flipSetup.β₁ t ≤ flipSetup.β₁ 1) ∧
    flipSetup.β₁ 1 < 1 ∧ 0 < flipSetup.β₂ ∧ flipSetup.β₂ < 1 ∧
    flipSetup.β₁ 1 / Real.sqrt flipSetup.β₂ < 1 := by
  have hb1 : flipSetup.β₁ 1 = (1 / 2 : ℝ) := by norm_num [flipSetup, flipBeta]
  refine ⟨hb1, ?_, ?_, ?_, ?_, ?_⟩
  · intro t ht
    rw [hb1]
    change 0 ≤ flipBeta t ∧ flipBeta t ≤ 1 / 2
    unfold flipBeta
    split_ifs <;> norm_num
  · rw [hb1]; norm_num
  · norm_num [flipSetup]
  · norm_num [flipSetup]
  · rw [hb1]
    change (1 / 2 : ℝ) / Real.sqrt (7 / 16) < 1
    rw [div_lt_one (Real.sqrt_pos.2 (by norm_num))]
    exact Real.lt_sqrt (by norm_num) |>.2 (by norm_num)

/-- Every hypothesis of the formalized Theorem A is satisfied at `T = 16`
with `x* = -1`. The identity `β₁ = β₁,₁ = 1/2` is `flip_params.1`.
Source: arXiv:1904.03590v4, §1, Theorem A. -/
theorem flip_theorem_A_hypotheses :
    IsOnlineConvex flipSetup (Set.Icc (fun _ => -1) (fun _ => 1)) 2 (4 / 3) ∧
    (0 : ℝ) < 100 ∧
    flipSetup.α = (fun t : ℕ => 100 / Real.sqrt t) ∧
    (∀ t, 1 ≤ t → 0 ≤ flipSetup.β₁ t ∧ flipSetup.β₁ t ≤ flipSetup.β₁ 1) ∧
    flipSetup.β₁ 1 < 1 ∧ 0 < flipSetup.β₂ ∧ flipSetup.β₂ < 1 ∧
    flipSetup.β₁ 1 / Real.sqrt flipSetup.β₂ < 1 ∧
    1 ≤ (16 : ℕ) ∧
    (fun _ : Fin 1 => (-1 : ℝ)) ∈ Set.Icc (fun _ => -1) (fun _ => 1) := by
  obtain ⟨-, hb, hb1, hb2, hb2', hγ⟩ := flip_params
  exact ⟨flip_isOnlineConvex, by norm_num, rfl, hb, hb1, hb2, hb2', hγ,
    by norm_num, ⟨fun _ => le_rfl, fun _ => by norm_num⟩⟩

/-- The index hypotheses used for the moments above have concrete witnesses.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
example : (1 ≤ 1) ∧ (3 ≤ 3 ∧ 3 % 2 = 1) ∧ (4 ≤ 4 ∧ 4 % 2 = 0) := by norm_num

end Transformer.AMSGrad
