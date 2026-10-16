/-
# Lipschitz regularity of the actual original attention field

ArXiv:2411.04551v3, §1, `eq: vf`, and §2, `prop: targets.atoms`.

The numerator and partition estimates control their quotient uniformly
on every bounded query set and for every probability law. The exact
quotient difference splits into a numerator difference and a partition
difference multiplied by the second normalized average. Its norm is at
most one, so both differences have the same bound.

On the unit sphere this yields a spatial Lipschitz constant depending
only on the attention matrix. Tangent projection adds at most two to
that constant. Thus the actual field along any evolving probability
has the uniform spatial regularity required by characteristic transport;
no differentiability or transport property of that law is assumed.
-/

import Transformer.Interpolation.Section2_AttentionWeights

open Real MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Multiplying the actual normalized average by its positive partition
recovers the original numerator. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem partition_smul_attentionAverage (B : ParamMatrix d) (ν : Perspective.ProbSphere d) (x : EucSpace d) :
    Perspective.partitionMu d 1 ν (B x) • attentionAverage B ν x =
      ∫ z : SSphere d, exp (inner ℝ (B x) (z : EucSpace d)) • (z : EucSpace d) ∂(ν : Measure (SSphere d)) := by
  rw [attentionAverage, smul_smul,
    mul_inv_cancel₀ (Perspective.partitionMu_pos d 1 ν (B x)).ne', one_smul]

/-- The original average is uniformly Lipschitz on bounded query sets,
independently of the probability. Auxiliary for arXiv:2411.04551v3,
§2, `prop: targets.atoms`, and §1, the characteristic well-posedness step. -/
theorem norm_attentionAverage_sub_le (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (R : ℝ) (x y : EucSpace d) (hx : ‖x‖ ≤ R) (hy : ‖y‖ ≤ R) :
    ‖attentionAverage B ν x - attentionAverage B ν y‖ ≤
      (2 * exp (2 * (‖B‖ * R)) * ‖B‖) * ‖x - y‖ := by
  let Z (v : EucSpace d) := Perspective.partitionMu d 1 ν (B v)
  let N (v : EucSpace d) := ∫ z : SSphere d,
    exp (inner ℝ (B v) (z : EucSpace d)) • (z : EucSpace d) ∂(ν : Measure (SSphere d))
  let C := exp (‖B‖ * R) * ‖B‖ * ‖x - y‖
  have hZ : 0 < Z x := Perspective.partitionMu_pos d 1 ν (B x)
  have hN : ‖N x - N y‖ ≤ C := norm_attention_numerator_sub_le B ν R x y hx hy
  have hdZ : |Z y - Z x| ≤ C := by
    rw [abs_sub_comm]
    exact abs_attention_partition_sub_le B ν R x y hx hy
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hinv : (Z x)⁻¹ ≤ exp (‖B‖ * R) := by
    have h := (inv_le_inv₀ hZ (exp_pos (-(‖B‖ * R)))).mpr
      (attention_partition_lower_on_ball B ν R x hx)
    simpa only [exp_neg, inv_inv] using h
  have he : attentionAverage B ν x - attentionAverage B ν y =
      (Z x)⁻¹ • ((N x - N y) + (Z y - Z x) • attentionAverage B ν y) := by
    have hc : (Z x)⁻¹ • ((Z x) • attentionAverage B ν y) = attentionAverage B ν y := by
      rw [smul_smul, inv_mul_cancel₀ hZ.ne', one_smul]
    calc
      _ = (Z x)⁻¹ • (N x - (Z x) • attentionAverage B ν y) := by
        rw [smul_sub, hc]
        rfl
      _ = _ := by
        congr 1
        rw [sub_smul, partition_smul_attentionAverage]
        abel
  have hb : ‖(N x - N y) + (Z y - Z x) • attentionAverage B ν y‖ ≤ 2 * C := by
    calc
      _ ≤ ‖N x - N y‖ + ‖(Z y - Z x) • attentionAverage B ν y‖ := norm_add_le _ _
      _ ≤ C + C := add_le_add hN (by
        rw [norm_smul, Real.norm_eq_abs]
        exact (mul_le_mul_of_nonneg_right hdZ (norm_nonneg _)).trans
          (by simpa only [mul_one] using mul_le_mul_of_nonneg_left (norm_attentionAverage_le_one B ν y) hC))
      _ = _ := by ring
  rw [he, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hZ)]
  calc
    _ ≤ exp (‖B‖ * R) * (2 * C) := mul_le_mul hinv hb (norm_nonneg _) (exp_pos _).le
    _ = _ := by dsimp [C]; rw [show 2 * (‖B‖ * R) = ‖B‖ * R + ‖B‖ * R by ring, exp_add]; ring

/-- The bounded-query estimate gives an actual Lipschitz constant.
Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem lipschitzOnWith_attentionAverage (B : ParamMatrix d) (ν : Perspective.ProbSphere d) (R : ℝ) :
    LipschitzOnWith ⟨2 * exp (2 * (‖B‖ * R)) * ‖B‖, by positivity⟩ (attentionAverage B ν)
      (Metric.closedBall 0 R) := by
  apply lipschitzOnWith_iff_norm_sub_le.mpr
  intro x hx y hy
  exact norm_attentionAverage_sub_le B ν R x y (mem_closedBall_zero_iff.mp hx) (mem_closedBall_zero_iff.mp hy)

/-- Sphere-restricted attention has the same uniform spatial bound.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem lipschitzWith_attentionAverage_sphere (B : ParamMatrix d) (ν : Perspective.ProbSphere d) :
    LipschitzWith ⟨2 * exp (2 * ‖B‖) * ‖B‖, by positivity⟩
      (fun x : SSphere d => attentionAverage B ν x) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  change dist (attentionAverage B ν x) (attentionAverage B ν y) ≤
    (2 * exp (2 * ‖B‖) * ‖B‖) * dist x y
  rw [dist_eq_norm, Subtype.dist_eq, dist_eq_norm]
  simpa only [mul_one] using
    norm_attentionAverage_sub_le B ν 1 x y (mem_sphere_zero_iff_norm.mp x.2).le
      (mem_sphere_zero_iff_norm.mp y.2).le

/-- The actual attention average is continuous at every ambient query.
Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_attentionAverage (B : ParamMatrix d) (ν : Perspective.ProbSphere d) :
    Continuous (attentionAverage B ν) := by
  apply continuous_iff_continuousAt.mpr
  intro x
  have hb : x ∈ Metric.ball 0 (‖x‖ + 1) := mem_ball_zero_iff.mpr (by linarith)
  exact ((lipschitzOnWith_attentionAverage B ν (‖x‖ + 1)).continuousOn x
    (mem_closedBall_zero_iff.mpr (by linarith))).continuousAt (Metric.closedBall_mem_nhds_of_mem hb)

/-- The original attention-only field is uniformly Lipschitz on the
sphere for every probability and time. Auxiliary for arXiv:2411.04551v3,
§1, the well-posedness of `eq: ips` and `eq: cauchy.pb`. -/
theorem lipschitzOnWith_fullVF_attention_sphere (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (t : ℝ) : LipschitzOnWith ⟨2 * exp (2 * ‖B‖) * ‖B‖ + 2, by positivity⟩
    (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) ν t) (Sphere d) := by
  apply lipschitzOnWith_iff_norm_sub_le.mpr
  intro x hx y hy
  have hnx : ‖x‖ = 1 := mem_sphere_zero_iff_norm.mp hx
  have hny : ‖y‖ = 1 := mem_sphere_zero_iff_norm.mp hy
  have hm : ‖attentionAverage B ν x - attentionAverage B ν y‖ ≤
      (2 * exp (2 * ‖B‖) * ‖B‖) * ‖x - y‖ := by
    simpa only [mul_one] using norm_attentionAverage_sub_le B ν 1 x y hnx.le hny.le
  rw [fullVF_eq_proj_attentionAverage, fullVF_eq_proj_attentionAverage]
  change ‖proj d x (attentionAverage B ν x) - proj d y (attentionAverage B ν y)‖ ≤
    (2 * exp (2 * ‖B‖) * ‖B‖ + 2) * ‖x - y‖
  exact (norm_proj_sub_proj_le hnx hny (norm_attentionAverage_le_one B ν y) le_rfl hm).trans_eq (by ring)

/-- The original attention-only field is continuous at every ambient
query. Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_fullVF_attention (B : ParamMatrix d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    Continuous (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) ν t) := by
  change Continuous (fun x : EucSpace d =>
    fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) ν t x)
  simp_rw [fullVF_eq_proj_attentionAverage, proj]
  exact (continuous_attentionAverage B ν).sub
    ((continuous_id.inner (𝕜 := ℝ) (continuous_attentionAverage B ν)).smul continuous_id)

/-- A nonzero matrix, an actual probability and distinct unit queries
jointly realize all bounded-query and sphere hypotheses. -/
example : ∃ (B : ParamMatrix 2) (ν : Perspective.ProbSphere 2) (x y : SSphere 2),
    ‖(x : EucSpace 2)‖ ≤ 1 ∧ ‖(y : EucSpace 2)‖ ≤ 1 ∧ x ≠ y ∧
    B (x : EucSpace 2) = x ∧ (ν : Measure (SSphere 2)) univ = 1 := by
  let p : SSphere 2 := basePoint 1
  refine ⟨ContinuousLinearMap.id ℝ (EucSpace 2), Perspective.diracProb 2 p,
    p, antipode 2 p, (mem_sphere_zero_iff_norm.mp p.2).le,
    (mem_sphere_zero_iff_norm.mp (antipode 2 p).2).le, (antipode_ne 2 p).symm,
    rfl, IsProbabilityMeasure.measure_univ⟩

end Transformer.Interpolation
