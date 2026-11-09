module

public import Transformer.Grokking.GradientEvidence.MainAdamMoments

/-!
# Finite-horizon motion for the main model's AdamW

Source: `review/flat_spectrum.md`, *Main-model AdamW motion*, AD3.
The radius is computed from the initial coordinate, schedule and moment
bound. No sign, independence or loss-decrease property of gradients is used.
The theorem permits positive learning rates; a long constant-rate interval
can still exhaust the motion budget. Exact-real and rounded steps are separate.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset

/-- AD3: a damped update with an explicit rounding term stays in this box. -/
theorem damped_step_box (p u η ρ R B e : ℝ) (hη : 0 ≤ η)
    (ha : η * ρ ≤ 1) (hp : |p| ≤ R) (hu : |u| ≤ B) :
    |(1 - η * ρ) * p - η * u + e| ≤ (1 - η * ρ) * R + η * B + |e| := by
  have h1 := abs_add_le ((1-η*ρ)*p) (-(η*u))
  have h2 := abs_add_le ((1-η*ρ)*p-η*u) e
  have h3 := mul_le_mul_of_nonneg_left hp (sub_nonneg.mpr ha)
  have h4 := mul_le_mul_of_nonneg_left hu hη
  simp only [abs_neg, abs_mul, abs_of_nonneg (sub_nonneg.mpr ha), abs_of_nonneg hη] at h1
  rw [← sub_eq_add_neg] at h1
  linarith

/-- AD3: motion is controlled by learning rate times drift and decay, plus
the explicit rounding term. The adaptive direction may have either sign. -/
theorem damped_step_motion (p u η ρ B e : ℝ) (hη : 0 ≤ η)
    (hρ : 0 ≤ ρ) (hu : |u| ≤ B) :
    |((1 - η * ρ) * p - η * u + e) - p| ≤ η * (ρ * |p| + B) + |e| := by
  have he : ((1-η*ρ)*p-η*u+e)-p = -(η*(ρ*p+u))+e := by ring
  rw [he]
  have h1 := abs_add_le (ρ*p) u
  have h2 := abs_add_le (-(η*(ρ*p+u))) e
  simp only [abs_neg, abs_mul, abs_of_nonneg hη, abs_of_nonneg hρ] at h1 h2
  have hs : |ρ*p+u| ≤ ρ*|p|+B := by linarith
  have h3 := mul_le_mul_of_nonneg_left hs hη
  linarith

/-- AD3: a nonzero-rate step can preserve a fixed box when decay balances
the bounded forcing. This does not assert preservation of a feature subspace. -/
theorem damped_step_preserves_box (p u η ρ R : ℝ) (hη : 0 ≤ η)
    (ha : η * ρ ≤ 1) (hp : |p| ≤ R) (hu : |u| ≤ ρ * R) :
    |(1 - η * ρ) * p - η * u| ≤ R := by
  have h := damped_step_box p u η ρ R (ρ*R) 0 hη ha hp hu
  simp only [add_zero, abs_zero] at h
  nlinarith

/-- AD3: compute a radius from the schedule and a direction bound. -/
def adamRadius (p0 ρ : ℝ) (η B : ℕ → ℝ) : ℕ → ℝ
  | 0 => |p0|
  | t+1 => (1-η t*ρ)*adamRadius p0 ρ η B t + η t*B t

/-- AD3: exact main-model parameter recurrence, with its actual moment history. -/
def mainAdamPath (p0 ρ eps : ℝ) (η g : ℕ → ℝ) : ℕ → ℝ
  | 0 => p0
  | t+1 => (1-η t*ρ)*mainAdamPath p0 ρ eps η g t - η t*mainDirection g eps t

/-- AD3: the path recurrence is the existing buffered-Adam coordinate update
with .9/.95 betas, not a surrogate gradient-descent step. -/
theorem mainAdamPath_source (p0 ρ eps : ℝ) (η g : ℕ → ℝ) (t : ℕ) :
    mainAdamPath p0 ρ eps η g (t+1) =
      bufferedAdamUpdate (9/10) (19/20) eps (η t) ρ (t+1)
        (mainAdamPath p0 ρ eps η g t)
        (mainMoments g (t+1)).1 (mainMoments g (t+1)).2 := by
  simp only [mainAdamPath, mainDirection, bufferedAdamUpdate]
  ring

/-- AD3: every parameter coordinate stays in a schedule-computable box for
all time. No condition on the gradient sequence is imposed. -/
theorem mainAdam_parameter_bound (p0 ρ eps : ℝ) (η g : ℕ → ℝ)
    (heps : 0 < eps) (hη : ∀ t, 0 ≤ η t) (hscale : ∀ t, η t * ρ ≤ 1)
    (start n : ℕ) :
    |mainAdamPath p0 ρ eps η g (start+n)| ≤
      adamRadius (mainAdamPath p0 ρ eps η g start) ρ
        (fun t => η (start+t)) (fun t => mainDirectionBound (start+t)) n := by
  induction n with
  | zero => exact le_rfl
  | succ t ih =>
    have h := damped_step_box (mainAdamPath p0 ρ eps η g (start+t))
      (mainDirection g eps (start+t)) (η (start+t)) ρ
      (adamRadius (mainAdamPath p0 ρ eps η g start) ρ
        (fun j => η (start+j)) (fun j => mainDirectionBound (start+j)) t)
      (mainDirectionBound (start+t)) 0 (hη _) (hscale _) ih (main_direction_bound g eps heps _)
    simpa only [Nat.add_succ, mainAdamPath, Nat.add_eq, adamRadius, add_zero, abs_zero] using h

/-- AD3: a finite learning-rate budget controls displacement over arbitrarily
many coupled training steps. It is a sufficient bound, not a convergence claim. -/
theorem mainAdam_motion_bound (p0 ρ eps : ℝ) (η g : ℕ → ℝ)
    (hρ : 0 ≤ ρ) (heps : 0 < eps) (hη : ∀ t, 0 ≤ η t)
    (hscale : ∀ t, η t * ρ ≤ 1) (start n : ℕ) :
    |mainAdamPath p0 ρ eps η g (start+n) - mainAdamPath p0 ρ eps η g start| ≤
      ∑ t ∈ range n, η (start+t) * (ρ *
        adamRadius (mainAdamPath p0 ρ eps η g start) ρ
          (fun j => η (start+j)) (fun j => mainDirectionBound (start+j)) t +
          mainDirectionBound (start+t)) := by
  induction n with
  | zero => simp only [Nat.add_zero, sub_self, abs_zero, range_zero, sum_empty, le_refl]
  | succ t ih =>
    have hp := mainAdam_parameter_bound p0 ρ eps η g heps hη hscale start t
    have hm := damped_step_motion (mainAdamPath p0 ρ eps η g (start+t))
      (mainDirection g eps (start+t)) (η (start+t)) ρ (mainDirectionBound (start+t)) 0
      (hη _) hρ (main_direction_bound g eps heps _)
    simp only [add_zero, abs_zero] at hm
    have hb := mul_le_mul_of_nonneg_left hp hρ
    have hb' := mul_le_mul_of_nonneg_left hb (hη (start+t))
    have ht := abs_sub_le (mainAdamPath p0 ρ eps η g (start+(t+1)))
      (mainAdamPath p0 ρ eps η g (start+t)) (mainAdamPath p0 ρ eps η g start)
    rw [sum_range_succ]
    simp only [mainAdamPath, Nat.add_eq] at ht ⊢
    linarith

example := damped_step_box 2 (-1) (1/100) (1/10) 3 2 (1/1000000)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := damped_step_motion 2 (-1) (1/100) (1/10) 2 (1/1000000)
  (by norm_num) (by norm_num) (by norm_num)
example := damped_step_preserves_box 2 (-1) (1/100) (1/10) 10
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example (g : ℕ → ℝ) (n : ℕ) := mainAdam_parameter_bound 2 (1/10) (1/100000000)
  (fun _ => 1/100) g (by norm_num) (by intro t; norm_num) (by intro t; norm_num) 4000 n
example (g : ℕ → ℝ) (n : ℕ) := mainAdam_motion_bound 2 (1/10) (1/100000000)
  (fun _ => 1/100) g (by norm_num) (by norm_num) (by intro t; norm_num)
  (by intro t; norm_num) 4000 n

end Transformer.Grokking.GradientEvidence
