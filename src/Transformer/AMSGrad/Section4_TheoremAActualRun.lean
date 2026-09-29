import Transformer.AMSGrad.Section4_TheoremAActualMoments

/-
# AMSGrad — iterates of the alternating run

For the concrete setup from `Section4_TheoremAActualMoments`, every step up to
sixteen crosses the box and clamps at the opposite endpoint. Consequently
the squared distance to `-1` alternates between zero and four, and the
scalar weight is `√t/100`.

Source: arXiv:1904.03590v4, §1, Algorithm 1; §3, Lemma 3.1.
-/

open Finset
namespace Transformer.AMSGrad

/-- The concrete one-dimensional AMSGrad step with `v̂_t = 1`.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_step (n : ℕ) :
    flipSetup.x amsgradRule (n + 2) 0 =
      max (-1) (min 1 (flipSetup.x amsgradRule (n + 1) 0 -
        100 / Real.sqrt (n + 1 : ℝ) * flipSetup.m amsgradRule (n + 1) 0)) := by
  have hv := flip_vhat_one (t := n + 1) (by omega)
  show (flipSetup.step amsgradRule (n + 1) (flipSetup.state amsgradRule n)).x 0 = _
  simp only [Setup.step]
  rw [show flipSetup.proj = fun _ => boxProj (-1) 1 from rfl]
  simp only [boxProj]
  congr 3
  simp only [Pi.sub_apply, Pi.smul_apply, Pi.div_apply, smul_eq_mul, vsqrt]
  change (flipSetup.state amsgradRule n).x 0 - flipSetup.α (n + 1) *
    (flipSetup.m amsgradRule (n + 1) 0 /
      Real.sqrt (flipSetup.vhat amsgradRule (n + 1) 0)) =
    flipSetup.x amsgradRule (n + 1) 0 -
      100 / Real.sqrt (n + 1 : ℝ) * flipSetup.m amsgradRule (n + 1) 0
  rw [show flipSetup.x amsgradRule (n + 1) = (flipSetup.state amsgradRule n).x from rfl]
  rw [hv]
  norm_num [flipSetup]

/-- Up to step sixteen, the smallest possible moment still moves the
unprojected iterate by more than the box diameter.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_large_step {t : ℕ} (ht : 1 ≤ t) (hT : t ≤ 16) :
    2 ≤ 100 / Real.sqrt (t : ℝ) * (1 / 6 : ℝ) := by
  have ht0 : (0 : ℝ) ≤ t := by positivity
  have ht16 : (t : ℝ) ≤ 16 := by exact_mod_cast hT
  have hs0 : 0 < Real.sqrt (t : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast ht)
  have hs4 : Real.sqrt (t : ℝ) ≤ 4 := by
    nlinarith [Real.sq_sqrt ht0, Real.sqrt_nonneg (t : ℝ)]
  rw [show 100 / Real.sqrt (t : ℝ) * (1 / 6 : ℝ) =
    100 / (6 * Real.sqrt (t : ℝ)) by ring]
  rw [le_div_iff₀ (by positivity : 0 < 6 * Real.sqrt (t : ℝ))]
  nlinarith

/-- The first seventeen iterates alternate between the endpoints of the
feasible interval. Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
private theorem flip_x_pattern (n : ℕ) (hn : n ≤ 16) :
    flipSetup.x amsgradRule (n + 1) 0 = if n % 2 = 0 then -1 else 1 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have hn' : n ≤ 16 := by omega
    have hx := ih hn'
    have ht : n + 1 ≤ 16 := by omega
    have hq := flip_large_step (t := n + 1) (by omega) ht
    have hq' : 2 ≤ 100 / Real.sqrt (n + 1 : ℝ) * (1 / 6 : ℝ) := by
      simpa only [Nat.cast_add, Nat.cast_one] using hq
    have hscale : 0 ≤ 100 / Real.sqrt (n + 1 : ℝ) := by positivity
    rw [show n + 1 + 1 = n + 2 by omega, flip_step n, hx]
    by_cases hp : n % 2 = 0
    · have hp' : (n + 1) % 2 = 1 := by omega
      have hm := (flip_m_bounds (t := n + 1) (by omega)).2 hp'
      have hmove : 2 ≤ -(100 / Real.sqrt (n + 1 : ℝ) *
          flipSetup.m amsgradRule (n + 1) 0) := by
        have hprod := mul_le_mul_of_nonneg_left hm hscale
        calc
          2 ≤ 100 / Real.sqrt (n + 1 : ℝ) * (1 / 6 : ℝ) := hq'
          _ ≤ -(100 / Real.sqrt (n + 1 : ℝ) *
              flipSetup.m amsgradRule (n + 1) 0) := by nlinarith only [hprod]
      have hpne : (n + 1) % 2 ≠ 0 := by omega
      simp only [hp, hpne, ite_true, ite_false]
      rw [min_eq_left (by linarith), max_eq_right (by norm_num)]
    · have hp' : (n + 1) % 2 = 0 := by omega
      have hm := (flip_m_bounds (t := n + 1) (by omega)).1 hp'
      have hmove : 2 ≤ 100 / Real.sqrt (n + 1 : ℝ) *
          flipSetup.m amsgradRule (n + 1) 0 := by
        exact hq'.trans (mul_le_mul_of_nonneg_left hm hscale)
      simp only [hp, hp', ite_true, ite_false]
      rw [min_eq_right (by linarith), max_eq_left (by linarith)]

/-- The scalar weight of the run from time one onward.
Source: arXiv:1904.03590v4, §3, Lemma 3.1. -/
noncomputable def flipA (t : ℕ) : ℝ := Real.sqrt t / 100

/-- Squared distance of the alternating iterate from the comparator `-1`.
Source: arXiv:1904.03590v4, §3, Lemma 3.1. -/
noncomputable def flipU (t : ℕ) : ℝ := if t % 2 = 0 then 4 else 0

/-- `a_t = √v̂_t/α_t = √t/100` for the concrete run.
Source: arXiv:1904.03590v4, §3, Lemma 3.1. -/
theorem flip_a_eq {t : ℕ} (ht : 1 ≤ t) :
    Real.sqrt (flipSetup.vhat amsgradRule t 0) / flipSetup.α t = flipA t := by
  rw [flip_vhat_one ht]
  change Real.sqrt 1 / (100 / Real.sqrt (t : ℝ)) = Real.sqrt t / 100
  norm_num

/-- `u_t = (x_t+1)^2` alternates between zero and four through step seventeen.
Source: arXiv:1904.03590v4, §3, Lemma 3.1. -/
theorem flip_u_eq {t : ℕ} (ht : 1 ≤ t) (hT : t ≤ 17) :
    (flipSetup.x amsgradRule t 0 - (-1)) ^ 2 = flipU t := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  have hn : n ≤ 16 := by omega
  rw [flip_x_pattern n hn]
  by_cases hp : n % 2 = 0
  · have hodd : (n + 1) % 2 ≠ 0 := by omega
    simp [flipU, hp, hodd]
  · have heven : (n + 1) % 2 = 0 := by omega
    norm_num [flipU, hp, heven]

/-- The time bounds in the iterate and weight lemmas have witnesses.
Source: arXiv:1904.03590v4, §1, Algorithm 1. -/
example : (1 ≤ 1) ∧ (1 ≤ 16) ∧ (0 ≤ 16) ∧ (1 ≤ 17) := by norm_num

end Transformer.AMSGrad
