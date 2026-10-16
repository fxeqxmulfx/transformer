/-
# A bounded spatial extension of the original attention field

Auxiliary for arXiv:2411.04551v3, §1, the characteristics of `eq: vf`,
and §2, `prop: targets.atoms`. Retract queries onto the radius-two ball
and multiply the projected average by a Lipschitz cutoff which is one
on the unit ball and zero outside the radius-two ball. The resulting
field is bounded and globally Lipschitz, uniformly over all laws. Its
restriction to the original sphere is exactly the manuscript's field.
-/

import Transformer.Interpolation.Section2_AttentionRegularity
import Transformer.GlobalFlow.TimeDependent

open Real MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Ambient projection bound for the extension of arXiv:2411.04551v3,
§1, `eq: vf`; no unit-vector assumption is made here. -/
theorem norm_proj_le_on_ball_two (x u : EucSpace d) (hx : ‖x‖ ≤ 2) :
    ‖proj d x u‖ ≤ 5 * ‖u‖ := by
  calc
    _ ≤ ‖u‖ + ‖inner ℝ x u • x‖ := norm_sub_le _ _
    _ = ‖u‖ + |inner ℝ x u| * ‖x‖ := by rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖u‖ + (2 * ‖u‖) * 2 := by
      gcongr
      exact (abs_real_inner_le_norm x u).trans (mul_le_mul_of_nonneg_right hx (norm_nonneg u))
    _ = _ := by ring

/-- Ambient spatial projection estimate, auxiliary for the
characteristics of arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem norm_proj_sub_proj_le_on_ball_two (x y u v : EucSpace d)
    (hx : ‖x‖ ≤ 2) (hy : ‖y‖ ≤ 2) (hv : ‖v‖ ≤ 1) :
    ‖proj d x u - proj d y v‖ ≤ 5 * ‖u - v‖ + 4 * ‖x - y‖ := by
  have he : proj d x u - proj d y v = proj d x (u - v) -
      (inner ℝ x v • (x - y) + inner ℝ (x - y) v • y) := by
    simp only [proj, inner_sub_right, inner_sub_left, sub_smul]
    module
  have h1 : ‖inner ℝ x v • (x - y)‖ ≤ 2 * ‖x - y‖ := by
    rw [norm_smul, Real.norm_eq_abs]
    gcongr
    calc
      _ ≤ ‖x‖ * ‖v‖ := abs_real_inner_le_norm x v
      _ ≤ 2 * 1 := mul_le_mul hx hv (norm_nonneg _) (by norm_num)
      _ = 2 := by ring
  have h2 : ‖inner ℝ (x - y) v • y‖ ≤ 2 * ‖x - y‖ := by
    rw [norm_smul, Real.norm_eq_abs]
    calc
      _ ≤ (‖x - y‖ * 1) * 2 := by
        gcongr
        exact (abs_real_inner_le_norm (x - y) v).trans
          (mul_le_mul_of_nonneg_left hv (norm_nonneg _))
      _ = _ := by ring
  rw [he]
  calc
    _ ≤ ‖proj d x (u - v)‖ + ‖inner ℝ x v • (x - y) + inner ℝ (x - y) v • y‖ := norm_sub_le _ _
    _ ≤ 5 * ‖u - v‖ + (2 * ‖x - y‖ + 2 * ‖x - y‖) :=
      add_le_add (norm_proj_le_on_ball_two x _ hx) ((norm_add_le _ _).trans (add_le_add h1 h2))
    _ = _ := by ring

/-- Radial cutoff for the characteristic extension of arXiv:2411.04551v3,
§1: one on the unit ball and zero outside the radius-two ball. -/
noncomputable def attentionCutoff (x : EucSpace d) : ℝ := min (max (2 - ‖x‖) 0) 1

/-- Cutoff range, auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem attentionCutoff_nonneg (x : EucSpace d) : 0 ≤ attentionCutoff x :=
  le_min (le_max_right _ _) zero_le_one

/-- Cutoff range, auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem attentionCutoff_le_one (x : EucSpace d) : attentionCutoff x ≤ 1 := min_le_right _ _

/-- Exact cutoff on the original sphere, for arXiv:2411.04551v3, §1. -/
theorem attentionCutoff_eq_one (x : EucSpace d) (hx : ‖x‖ ≤ 1) : attentionCutoff x = 1 := by
  exact min_eq_right ((by linarith : (1 : ℝ) ≤ 2 - ‖x‖).trans (le_max_left _ _))

/-- Compact spatial cutoff, auxiliary for arXiv:2411.04551v3, §1. -/
theorem attentionCutoff_eq_zero (x : EucSpace d) (hx : 2 ≤ ‖x‖) : attentionCutoff x = 0 := by
  dsimp only [attentionCutoff]
  rw [max_eq_right (by linarith)]
  exact min_eq_left zero_le_one

/-- Cutoff regularity for arXiv:2411.04551v3, §1, characteristics. -/
theorem lipschitzWith_attentionCutoff : LipschitzWith 1 (attentionCutoff (d := d)) := by
  have h : LipschitzWith 1 (fun x : EucSpace d => 2 - ‖x‖) := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    rw [Real.dist_eq, dist_eq_norm]
    change |(2 - ‖x‖) - (2 - ‖y‖)| ≤ 1 * ‖x - y‖
    rw [show (2 - ‖x‖) - (2 - ‖y‖) = ‖y‖ - ‖x‖ by ring, abs_sub_comm, one_mul]
    exact abs_norm_sub_norm_le x y
  exact (h.max_const 0).min_const 1

/-- Bounded extension of the actual field in arXiv:2411.04551v3, §1,
`eq: vf`. Off the sphere it uses the stated retraction and cutoff. -/
noncomputable def attentionGlobalField (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x : EucSpace d) : EucSpace d :=
  attentionCutoff x • proj d (GlobalFlow.ballRetract 2 x)
    (attentionAverage B ν (GlobalFlow.ballRetract 2 x))

/-- The extension equals the original velocity at every unit query.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem attentionGlobalField_eq_fullVF (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : SSphere d) : attentionGlobalField B ν x =
      fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) ν t x := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [attentionGlobalField, GlobalFlow.ballRetract_of_norm_le (by norm_num : (0 : ℝ) < 2) (by linarith),
    attentionCutoff_eq_one _ hx.le, one_smul, fullVF_eq_proj_attentionAverage]

/-- Uniform bound for the characteristic extension of arXiv:2411.04551v3,
§1, independent of the actual probability law. -/
theorem norm_attentionGlobalField_le (B : ParamMatrix d) (ν : Perspective.ProbSphere d) (x : EucSpace d) :
    ‖attentionGlobalField B ν x‖ ≤ 5 := by
  rw [attentionGlobalField, norm_smul, Real.norm_of_nonneg (attentionCutoff_nonneg x)]
  have hp : ‖proj d (GlobalFlow.ballRetract 2 x)
      (attentionAverage B ν (GlobalFlow.ballRetract 2 x))‖ ≤ 5 * 1 :=
    (norm_proj_le_on_ball_two (GlobalFlow.ballRetract 2 x) _
    (GlobalFlow.norm_ballRetract_le (by norm_num : (0 : ℝ) < 2) x)).trans
      (mul_le_mul_of_nonneg_left (norm_attentionAverage_le_one B ν _) (by norm_num : (0 : ℝ) ≤ 5))
  calc
    _ ≤ 1 * (5 * 1) := mul_le_mul (attentionCutoff_le_one x) hp (norm_nonneg _) zero_le_one
    _ = 5 := by ring

/-- Global spatial regularity for the characteristic extension of
arXiv:2411.04551v3, §1, uniformly over all original probability laws. -/
theorem lipschitzWith_attentionGlobalField (B : ParamMatrix d) (ν : Perspective.ProbSphere d) :
    LipschitzWith ⟨2 * (5 * (2 * exp (2 * (‖B‖ * 2)) * ‖B‖) + 4) + 5, by positivity⟩
      (attentionGlobalField B ν) := by
  let R (x : EucSpace d) := GlobalFlow.ballRetract 2 x
  let G (x : EucSpace d) := proj d (R x) (attentionAverage B ν (R x))
  let L : ℝ := 2 * exp (2 * (‖B‖ * 2)) * ‖B‖
  have hR (x : EucSpace d) : ‖R x‖ ≤ 2 := GlobalFlow.norm_ballRetract_le (by norm_num) x
  have hG (x : EucSpace d) : ‖G x‖ ≤ 5 := by
    exact (norm_proj_le_on_ball_two _ _ (hR x)).trans
      ((mul_le_mul_of_nonneg_left (norm_attentionAverage_le_one B ν _) (by norm_num : (0 : ℝ) ≤ 5)).trans_eq (by ring))
  have hLip (x y : EucSpace d) : ‖G x - G y‖ ≤ 2 * (5 * L + 4) * ‖x - y‖ := by
    have hAv := norm_attentionAverage_sub_le B ν 2 (R x) (R y) (hR x) (hR y)
    have hRet := (GlobalFlow.lipschitzWith_ballRetract (by norm_num : (0 : ℝ) < 2)).dist_le_mul x y
    rw [dist_eq_norm, dist_eq_norm] at hRet
    change ‖R x - R y‖ ≤ 2 * ‖x - y‖ at hRet
    calc
      _ ≤ 5 * ‖attentionAverage B ν (R x) - attentionAverage B ν (R y)‖ + 4 * ‖R x - R y‖ :=
        norm_proj_sub_proj_le_on_ball_two _ _ _ _ (hR x) (hR y) (norm_attentionAverage_le_one B ν _)
      _ ≤ 5 * (L * (2 * ‖x - y‖)) + 4 * (2 * ‖x - y‖) := by
        gcongr
        exact hAv.trans (mul_le_mul_of_nonneg_left hRet (by positivity))
      _ = _ := by ring
  apply LipschitzWith.of_dist_le_mul
  intro x y
  rw [dist_eq_norm, dist_eq_norm]
  change ‖attentionCutoff x • G x - attentionCutoff y • G y‖ ≤
    (2 * (5 * L + 4) + 5) * ‖x - y‖
  have hsplit : attentionCutoff x • G x - attentionCutoff y • G y =
      attentionCutoff x • (G x - G y) + (attentionCutoff x - attentionCutoff y) • G y := by module
  have hcut := lipschitzWith_attentionCutoff.dist_le_mul x y
  rw [Real.dist_eq, dist_eq_norm] at hcut
  simp only [NNReal.coe_one, one_mul] at hcut
  rw [hsplit]
  calc
    _ ≤ ‖attentionCutoff x • (G x - G y)‖ + ‖(attentionCutoff x - attentionCutoff y) • G y‖ := norm_add_le _ _
    _ = attentionCutoff x * ‖G x - G y‖ + |attentionCutoff x - attentionCutoff y| * ‖G y‖ := by
      rw [norm_smul, norm_smul, Real.norm_of_nonneg (attentionCutoff_nonneg x), Real.norm_eq_abs]
    _ ≤ 1 * (2 * (5 * L + 4) * ‖x - y‖) + ‖x - y‖ * 5 :=
      add_le_add (mul_le_mul (attentionCutoff_le_one x) (hLip x y) (norm_nonneg _) zero_le_one)
        (mul_le_mul hcut (hG y) (norm_nonneg _) (norm_nonneg _))
    _ = _ := by ring

/-- Distinct unit queries, bounded vector data and a radius-two query
jointly witness every projection, cutoff and spherical hypothesis. -/
example : ∃ (x y u v z : EucSpace 2), ‖x‖ ≤ 2 ∧ ‖y‖ ≤ 2 ∧ ‖v‖ ≤ 1 ∧
    ‖x‖ ≤ 1 ∧ 2 ≤ ‖z‖ ∧ x ∈ Sphere 2 ∧ y ∈ Sphere 2 ∧ u ≠ 0 := by
  let p : SSphere 2 := basePoint 1
  have hp : ‖(p : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  refine ⟨p, - (p : EucSpace 2), p, p, (2 : ℝ) • (p : EucSpace 2),
    by linarith, ?_, hp.le, hp.le, ?_, p.2, ?_, ?_⟩
  · simpa only [norm_neg] using (by linarith : ‖(p : EucSpace 2)‖ ≤ 2)
  · rw [norm_smul, hp]; norm_num
  · exact mem_sphere_zero_iff_norm.mpr (by rw [norm_neg, hp])
  · intro h
    rw [h, norm_zero] at hp
    norm_num at hp

end Transformer.Interpolation
