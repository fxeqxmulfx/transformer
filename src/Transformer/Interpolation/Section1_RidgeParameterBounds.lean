/-
# Original parameter norms and balanced ReLU clock scaling

ArXiv:2411.04551v3, §1, `eq: vf`, and §5, the norm
estimate in the proof of `thm: targets.atoms`. The actual single-neuron
output and input operator norms are bounded by their original vectors;
the actual bias norm is the absolute scalar bias. Rank-one attention
has parameter norm at most its scalar clock on a unit axis.

Positive homogeneity of the actual affine ReLU permits a clock factor
κ to be divided equally between the output and the affine gate.
The resulting original full vector field is exactly κ times the old
field, while the sum of parameter norms grows only as sqrt κ.
This is a redistribution of actual original parameters, not a change
to the vector field or to the time assigned to a phase.
-/

import Transformer.Interpolation.Section1_PhaseSequenceControls
import Transformer.Interpolation.Section3_ProjectionControls

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual ridge output operator has norm bounded by its output
vector. Source auxiliary: arXiv:2411.04551v3, §1 and §5, control norms. -/
theorem norm_reluRidgeParams_W_le (i : Fin d) (p u : EucSpace d) (b : ℝ) :
    ‖(reluRidgeParams i p u b).W‖ ≤ ‖p‖ :=
  norm_projectionControl_W_le i p

/-- The actual ridge input operator has norm bounded by its gate
vector. Source auxiliary: arXiv:2411.04551v3, §1 and §5. -/
theorem norm_reluRidgeParams_U_le (i : Fin d) (p u : EucSpace d) (b : ℝ) :
    ‖(reluRidgeParams i p u b).U‖ ≤ ‖u‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg u)
  intro x
  change ‖(-inner ℝ u x) • (EuclideanSpace.single i (1 : ℝ))‖ ≤ ‖u‖ * ‖x‖
  have he : ‖(EuclideanSpace.single i (1 : ℝ) : EucSpace d)‖ = 1 := by simp
  rw [norm_smul, Real.norm_eq_abs, abs_neg, he, mul_one]
  exact abs_real_inner_le_norm u x

/-- The actual original one-neuron bias has precisely its scalar
absolute value as norm. Source auxiliary: arXiv:2411.04551v3, §1 and §5. -/
theorem norm_reluRidgeParams_b (i : Fin d) (p u : EucSpace d) (b : ℝ) :
    ‖(reluRidgeParams i p u b).b‖ = |b| := by
  change ‖b • (EuclideanSpace.single i (1 : ℝ))‖ = |b|
  have he : ‖(EuclideanSpace.single i (1 : ℝ) : EucSpace d)‖ = 1 := by simp
  rw [norm_smul, Real.norm_eq_abs, he, mul_one]

/-- The complete original ridge tuple has its actual sum-norm bound.
Source auxiliary: arXiv:2411.04551v3, §1 and §5, the parameter norm
used in `thm: targets.atoms`. -/
theorem norm_reluRidgeParams_sum_le (i : Fin d) (p u : EucSpace d) (b : ℝ) :
    ‖(reluRidgeParams i p u b).V‖ + ‖(reluRidgeParams i p u b).B‖ +
      ‖(reluRidgeParams i p u b).W‖ + ‖(reluRidgeParams i p u b).U‖ +
      ‖(reluRidgeParams i p u b).b‖ ≤ ‖p‖ + ‖u‖ + |b| := by
  have hV : (reluRidgeParams i p u b).V = 0 := rfl
  have hB : (reluRidgeParams i p u b).B = 0 := rfl
  simp only [hV, hB, norm_zero, zero_add]
  linarith [norm_reluRidgeParams_W_le i p u b, norm_reluRidgeParams_U_le i p u b,
    norm_reluRidgeParams_b i p u b]

/-- Share a nonnegative clock factor between the actual output and
affine gate using the positive homogeneity of ReLU. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, clock realization at controlled norms. -/
noncomputable def balancedRidgeParams (i : Fin d) (p u : EucSpace d) (b κ : ℝ) : Params d :=
  reluRidgeParams i (sqrt κ • p) (sqrt κ • u) (sqrt κ * b)

/-- Balanced original parameters have only square-root growth of
their complete sum norm. Source auxiliary: arXiv:2411.04551v3,
§1 and §5, the actual parameter cost of clock scaling. -/
theorem norm_balancedRidgeParams_sum_le (i : Fin d) (p u : EucSpace d) (b κ : ℝ) :
    ‖(balancedRidgeParams i p u b κ).V‖ + ‖(balancedRidgeParams i p u b κ).B‖ +
      ‖(balancedRidgeParams i p u b κ).W‖ + ‖(balancedRidgeParams i p u b κ).U‖ +
      ‖(balancedRidgeParams i p u b κ).b‖ ≤ sqrt κ * (‖p‖ + ‖u‖ + |b|) := by
  have hh := norm_reluRidgeParams_sum_le i (sqrt κ • p) (sqrt κ • u) (sqrt κ * b)
  simpa only [balancedRidgeParams, norm_smul, Real.norm_of_nonneg (sqrt_nonneg κ),
    abs_mul, abs_of_nonneg (sqrt_nonneg κ), mul_add] using hh

/-- A genuine unit output direction leaves only the fixed affine
gate size in the square-root clock cost. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, prescribed-time ridge phases. -/
theorem norm_balancedRidgeParams_unit_output_le (i : Fin d) (p : SSphere d)
    (u : EucSpace d) (b κ : ℝ) :
    ‖(balancedRidgeParams i p u b κ).V‖ + ‖(balancedRidgeParams i p u b κ).B‖ +
      ‖(balancedRidgeParams i p u b κ).W‖ + ‖(balancedRidgeParams i p u b κ).U‖ +
      ‖(balancedRidgeParams i p u b κ).b‖ ≤ sqrt κ * (1 + ‖u‖ + |b|) := by
  simpa only [mem_sphere_zero_iff_norm.mp p.2] using
    norm_balancedRidgeParams_sum_le i (p : EucSpace d) u b κ

/-- Balanced original parameters realize precisely the requested
clock factor in the original full vector field, for every original
probability and ambient query. Source auxiliary: arXiv:2411.04551v3,
§1, `eq: vf`, and §5, parameter time scaling. -/
theorem fullVF_balancedRidgeParams (i : Fin d) (p u : EucSpace d) (b κ : ℝ) (hκ : 0 ≤ κ)
    (ν : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (fun _ => balancedRidgeParams i p u b κ) ν t x =
      κ • fullVF d (fun _ => reluRidgeParams i p u b) ν t x := by
  let s := sqrt κ
  have hs : 0 ≤ s := sqrt_nonneg κ
  have hs2 : s * s = κ := by nlinarith [sq_sqrt hκ]
  have hgate : max (s * b - inner ℝ x (s • u)) 0 = s * max (b - inner ℝ x u) 0 := by
    rw [real_inner_smul_right]
    calc
      max (s * b - s * inner ℝ x u) 0 = max (s * (b - inner ℝ x u)) (s * 0) := by
        rw [mul_sub, mul_zero]
      _ = s * max (b - inner ℝ x u) 0 := (mul_max_of_nonneg _ _ hs).symm
  rw [balancedRidgeParams, fullVF_reluRidgeParams, fullVF_reluRidgeParams]
  change max (s * b - inner ℝ x (s • u)) 0 • proj d x (s • p) =
    κ • (max (b - inner ℝ x u) 0 • proj d x p)
  rw [hgate, projection_field_smul, smul_smul, smul_smul]
  congr 1
  calc
    s * max (b - inner ℝ x u) 0 * s = (s * s) * max (b - inner ℝ x u) 0 := by ring
    _ = κ * max (b - inner ℝ x u) 0 := by rw [hs2]

/-- The actual rank-one value operator has its quadratic axis-norm
bound. Source: arXiv:2411.04551v3, §3.2, attention phases, and §5. -/
theorem norm_meanAxisParams_V_le (w : EucSpace d) (κ : ℝ) :
    ‖(meanAxisParams w κ).V‖ ≤ |κ| * ‖w‖ ^ 2 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  intro x
  change ‖κ • (inner ℝ w x • w)‖ ≤ (|κ| * ‖w‖ ^ 2) * ‖x‖
  rw [norm_smul, Real.norm_eq_abs, norm_smul, Real.norm_eq_abs]
  calc
    |κ| * (|inner ℝ w x| * ‖w‖) ≤ |κ| * ((‖w‖ * ‖x‖) * ‖w‖) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right (abs_real_inner_le_norm w x)
        (norm_nonneg w)) (abs_nonneg κ)
    _ = (|κ| * ‖w‖ ^ 2) * ‖x‖ := by ring

/-- On an actual unit axis, the complete original attention tuple
cost is at most its absolute clock. Source auxiliary:
arXiv:2411.04551v3, §3.2 and §5, the bounded isolation phases. -/
theorem norm_meanAxisParams_sum_le (w : SSphere d) (κ : ℝ) :
    ‖(meanAxisParams (w : EucSpace d) κ).V‖ + ‖(meanAxisParams (w : EucSpace d) κ).B‖ +
      ‖(meanAxisParams (w : EucSpace d) κ).W‖ + ‖(meanAxisParams (w : EucSpace d) κ).U‖ +
      ‖(meanAxisParams (w : EucSpace d) κ).b‖ ≤ |κ| := by
  simpa only [meanAxisParams, norm_zero, add_zero, mem_sphere_zero_iff_norm.mp w.2,
    one_pow, mul_one] using norm_meanAxisParams_V_le (w : EucSpace d) κ

/-- Genuine nonzero output and gate vectors, a nonzero bias and a
strictly positive clock jointly realize the balancing hypotheses.
The coordinate output also supplies a genuine unit attention axis. -/
example : ∃ (p u : EucSpace 2) (b κ : ℝ), 0 ≤ κ ∧ 0 < κ ∧
    ‖p‖ = 1 ∧ ‖u‖ = 1 ∧ b ≠ 0 := by
  refine ⟨PiLp.single 2 (0 : Fin 2) 1, PiLp.single 2 (1 : Fin 2) 1, 1, 2,
    by norm_num, by norm_num, by simp, by simp, one_ne_zero⟩

end Transformer.Interpolation
