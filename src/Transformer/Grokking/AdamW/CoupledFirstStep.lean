import Transformer.Grokking.AdamW.FirstStep
import Transformer.Grokking.Composition.Basic

/-!
# Native first update of an actual compositional CE score

Source comparison: Nanda et al., arXiv:2301.05217v1, appendix Further
speculations on grokking, Hypothesis: Phase Transitions are inherent to
composition. The source's discussion is speculative. Explicit deviation:
the verified two-component score `x*y` and unit-weight binary CE, not a
GPTMini circuit. Apply the native lab AdamW at 91bb895 and PyTorch 2.14.1's
first-step moment equations with zero initial buffers.

At a positive aligned component state, derive the actual adaptive update
and its exact growth condition. This condition depends on epsilon and
amplitude; the coupled L2 origin-curvature threshold is not its coefficient.
Exactly absent components stay absent. Growing positive components reduces
CE, but can be confidence growth with no new correct held-out decisions.
No accumulated moments, stochastic gradient noise or repeated-step rule
formation are supplied by this fresh-buffer specialization.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.Composition

/-- Native coordinate updates of the actually differentiated coupled CE.
Source: the arXiv:2301.05217v1 appendix-inspired scalar specialization and
PyTorch 2.14.1 AdamW first moment/bias-correction equations at 91bb895. -/
noncomputable def coupledFirstPoint (b1 b2 eps decay eta x y : ℝ) : ℝ × ℝ :=
  (firstUpdate b1 b2 eps decay eta x (coupledGradient x y).1,
   firstUpdate b1 b2 eps decay eta y (coupledGradient x y).2)

/-- Both coordinate gradients agree on the aligned state, by actual CE
differentiation. Source: arXiv:2301.05217v1 appendix composition diagnostic;
this is a scalar specialization rather than a chosen update direction. -/
theorem coupled_aligned_gradient (s : ℝ) :
    coupledGradient s s = (-s / (Real.exp (s ^ 2) + 1), -s / (Real.exp (s ^ 2) + 1)) := by
  rw [coupledGradient_eq]
  simp only [pow_two]

/-- Exact first adaptive normalization of the negative aligned gradient.
Source: native AdamW at 91bb895; epsilon stays in the denominator and
the gradient partners come from the actual bilinear CE. -/
theorem coupled_aligned_first_direction (b1 b2 eps s : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hs : 0 < s) :
    firstDirection b1 b2 eps (-s / (Real.exp (s ^ 2) + 1)) =
      -s / (s + eps * (Real.exp (s ^ 2) + 1)) := by
  have hd : 0 < Real.exp (s ^ 2) + 1 := by positivity
  have hg : -s / (Real.exp (s ^ 2) + 1) < 0 :=
    div_neg_of_neg_of_pos (by linarith) hd
  rw [firstDirection_eq b1 b2 eps _ h1 h2, abs_of_neg hg]
  have hdn : Real.exp (s ^ 2) + 1 ≠ 0 := by positivity
  have hn : s + eps * (Real.exp (s ^ 2) + 1) ≠ 0 := by positivity
  have hi : -(-s / (Real.exp (s ^ 2) + 1)) + eps ≠ 0 := by
    have hx : 0 < s / (Real.exp (s ^ 2) + 1) := div_pos hs hd
    linarith
  field_simp

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 : ℝ) := by norm_num

/-- The actual aligned first point, including decoupled decay. Source:
native AdamW at 91bb895 and the explicit arXiv:2301.05217v1-inspired
CE score; zero moment buffers are part of firstUpdate's algorithm. -/
theorem coupled_aligned_first_point (b1 b2 eps decay eta s : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hs : 0 < s) :
    coupledFirstPoint b1 b2 eps decay eta s s =
      (s + eta * s * (1 / (s + eps * (Real.exp (s ^ 2) + 1)) - decay),
       s + eta * s * (1 / (s + eps * (Real.exp (s ^ 2) + 1)) - decay)) := by
  unfold coupledFirstPoint
  rw [coupled_aligned_gradient]
  unfold firstUpdate
  rw [coupled_aligned_first_direction b1 b2 eps s h1 h2 he hs]
  have hn : s + eps * (Real.exp (s ^ 2) + 1) ≠ 0 := by positivity
  congr 1 <;> field_simp <;> ring

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 : ℝ) := by norm_num

/-- Exactly absent components remain absent under the actual fresh
update. Source: the arXiv:2301.05217v1 appendix's all-absent argument;
this fixed point does not model nonzero random initial components. -/
theorem coupled_first_origin_fixed (b1 b2 eps decay eta : ℝ) :
    coupledFirstPoint b1 b2 eps decay eta 0 0 = (0, 0) := by
  unfold coupledFirstPoint
  rw [coupled_origin_stationary]
  norm_num [firstUpdate, firstDirection]

/-- Exact positive-amplitude growth condition under the fresh native
update. Source comparison: arXiv:2301.05217v1 appendix composition/size
competition; decoupled AdamW is not the coupled L2 penalized gradient. -/
theorem coupled_first_growth_iff (b1 b2 eps decay eta s : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < eta) (hs : 0 < s) :
    s < (coupledFirstPoint b1 b2 eps decay eta s s).1 ↔
      decay * (s + eps * (Real.exp (s ^ 2) + 1)) < 1 := by
  rw [coupled_aligned_first_point b1 b2 eps decay eta s h1 h2 he hs]
  dsimp only
  have hd : 0 < s + eps * (Real.exp (s ^ 2) + 1) := by positivity
  have hm : 0 < eta * s := mul_pos heta hs
  constructor
  · intro h
    have hp : 0 < eta * s * (1 / (s + eps * (Real.exp (s ^ 2) + 1)) - decay) := by linarith
    have hf := (mul_pos_iff_of_pos_left hm).mp hp
    exact (lt_div_iff₀ hd).mp (by linarith : decay < 1 / (s + eps * (Real.exp (s ^ 2) + 1)))
  · intro h
    have hi := (lt_div_iff₀ hd).mpr h
    have hp := mul_pos hm (show 0 < 1 / (s + eps * (Real.exp (s ^ 2) + 1)) - decay by linarith)
    linarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧ 0 < (1 : ℝ) := by norm_num

/-- Positive aligned growth strictly lowers actual CE for any positive
rate satisfying the derived condition. Source: arXiv:2301.05217v1
appendix confidence/size discussion; no new decision is asserted here. -/
theorem coupled_first_growth_lowers_CE (b1 b2 eps decay eta s : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < eta) (hs : 0 < s)
    (hg : decay * (s + eps * (Real.exp (s ^ 2) + 1)) < 1) :
    coupledLoss (coupledFirstPoint b1 b2 eps decay eta s s).1
      (coupledFirstPoint b1 b2 eps decay eta s s).2 < coupledLoss s s := by
  have hi := (coupled_first_growth_iff b1 b2 eps decay eta s h1 h2 he heta hs).mpr hg
  rw [coupled_aligned_first_point b1 b2 eps decay eta s h1 h2 he hs] at hi ⊢
  dsimp only at hi ⊢
  rw [coupledLoss_eq_exp, coupledLoss_eq_exp]
  apply Real.log_lt_log (by positivity)
  have ht : -((s + eta * s * (1 / (s + eps * (Real.exp (s ^ 2) + 1)) - decay)) *
      (s + eta * s * (1 / (s + eps * (Real.exp (s ^ 2) + 1)) - decay))) < -(s * s) := by
    nlinarith
  have hx := Real.exp_lt_exp.mpr ht
  linarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧ 0 < (1 : ℝ) ∧
    (0 : ℝ) * (1 + 1 * (Real.exp (1 ^ 2) + 1)) < 1 := by norm_num

/-- The epsilon-dependent upper decay threshold prevents growth at
every positive aligned amplitude. Source: native AdamW at 91bb895,
explicit specialization of the appendix competition hypothesis; this
is neither a GPTMini threshold nor the coupled L2 constant one-half. -/
theorem coupled_first_no_growth_above_threshold (b1 b2 eps decay eta s : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < eta) (hs : 0 < s)
    (hd : 1 / (2 * eps) ≤ decay) :
    (coupledFirstPoint b1 b2 eps decay eta s s).1 ≤ s := by
  have hp : 0 < 2 * eps := by positivity
  have hh : 1 ≤ decay * (2 * eps) := (div_le_iff₀ hp).mp hd
  have hl : 0 < decay := by
    have hq : 0 < 1 / (2 * eps) := by positivity
    linarith
  have hx := Real.one_le_exp (sq_nonneg s)
  have hb : 2 * eps < s + eps * (Real.exp (s ^ 2) + 1) := by nlinarith
  have hc : 1 ≤ decay * (s + eps * (Real.exp (s ^ 2) + 1)) := by nlinarith
  have hn : ¬ s < (coupledFirstPoint b1 b2 eps decay eta s s).1 := by
    rw [coupled_first_growth_iff b1 b2 eps decay eta s h1 h2 he heta hs]
    linarith
  linarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧ 0 < (1 : ℝ) ∧
    1 / (2 * (1 : ℝ)) ≤ 1 := by norm_num

end Transformer.Grokking.AdamW
