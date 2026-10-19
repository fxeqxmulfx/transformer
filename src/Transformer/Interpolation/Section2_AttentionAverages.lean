module

public import Transformer.Interpolation.Basic
public import Transformer.Perspective.PartitionGradient
public import Mathlib.MeasureTheory.Function.L2Space

/-!
# Uniform bounds for original measure-valued attention averages

ArXiv:2411.04551v3, §1, `eq: vf`, and §2, `prop: targets.atoms`.

For a fixed attention matrix, all spherical scores lie between its
negative and positive operator norm. The partition function has the
corresponding exponential bounds for every initial probability. Thus
every normalized weight lies between two strictly positive constants.

The actual attention average has norm at most one and inherits every
almost-everywhere lower bound on a linear height. Its projected original
particle velocity consequently has norm at most one. These are the
pointwise field estimates used for hemisphere preservation and support
contraction; no support invariance or clustering conclusion is assumed.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The exact softmax average in the source's original velocity.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
noncomputable def attentionAverage (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x : EucSpace d) : EucSpace d :=
  (Perspective.partitionMu d 1 ν (B x))⁻¹ •
    ∫ y : SSphere d, exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d)
      ∂(ν : Measure (SSphere d))

/-- Every spherical score is bounded by the matrix operator norm.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem abs_attention_score_le (B : ParamMatrix d) (x y : SSphere d) :
    |inner ℝ (B (x : EucSpace d)) (y : EucSpace d)| ≤ ‖B‖ := by
  calc
    _ ≤ ‖B (x : EucSpace d)‖ * ‖(y : EucSpace d)‖ := abs_real_inner_le_norm _ _
    _ = ‖B (x : EucSpace d)‖ := by rw [mem_sphere_zero_iff_norm.mp y.2, mul_one]
    _ ≤ ‖B‖ * ‖(x : EucSpace d)‖ := B.le_opNorm _
    _ = ‖B‖ := by rw [mem_sphere_zero_iff_norm.mp x.2, mul_one]

/-- The original probability partition has uniform exponential bounds.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem attention_partition_bounds (B : ParamMatrix d) (ν : Perspective.ProbSphere d) (x : SSphere d) :
    exp (-‖B‖) ≤ Perspective.partitionMu d 1 ν (B x) ∧
      Perspective.partitionMu d 1 ν (B x) ≤ exp ‖B‖ := by
  have hi : Integrable (fun y : SSphere d => exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)))
      (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_ambient d 1 ν (B x)
  have hb (y : SSphere d) := abs_le.mp (abs_attention_score_le B x y)
  constructor
  · have h := integral_mono (integrable_const (exp (-‖B‖))) hi
      (fun y => exp_le_exp.mpr (hb y).1)
    simpa only [Perspective.partitionMu, one_mul, integral_const, probReal_univ, one_smul] using h
  · have h := integral_mono hi (integrable_const (exp ‖B‖))
      (fun y => exp_le_exp.mpr (hb y).2)
    simpa only [Perspective.partitionMu, one_mul, integral_const, probReal_univ, one_smul] using h

/-- Original normalized weights are uniformly bounded and strictly
positive. Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem attention_normalized_weight_bounds (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x y : SSphere d) :
    exp (-(2 * ‖B‖)) ≤ exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)) /
      Perspective.partitionMu d 1 ν (B x) ∧
    exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)) / Perspective.partitionMu d 1 ν (B x) ≤
      exp (2 * ‖B‖) := by
  obtain ⟨hl, hu⟩ := attention_partition_bounds B ν x
  have hb := abs_le.mp (abs_attention_score_le B x y)
  have hZ := Perspective.partitionMu_pos d 1 ν (B x)
  constructor
  · calc
      _ = exp (-‖B‖) / exp ‖B‖ := by rw [← exp_sub]; congr 1; ring
      _ ≤ _ := div_le_div₀ (exp_pos _).le (exp_le_exp.mpr hb.1) hZ hu
  · calc
      _ ≤ exp ‖B‖ / exp (-‖B‖) := div_le_div₀ (exp_pos _).le (exp_le_exp.mpr hb.2) (exp_pos _) hl
      _ = _ := by rw [← exp_sub]; congr 1; ring

/-- A genuine probability attention average stays in the closed unit
ball. Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem norm_attentionAverage_le_one (B : ParamMatrix d) (ν : Perspective.ProbSphere d) (x : EucSpace d) :
    ‖attentionAverage B ν x‖ ≤ 1 := by
  have hZ := Perspective.partitionMu_pos d 1 ν (B x)
  have hnum : ‖∫ y : SSphere d, exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d)
      ∂(ν : Measure (SSphere d))‖ ≤ Perspective.partitionMu d 1 ν (B x) := by
    calc
      _ ≤ ∫ y : SSphere d, ‖exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d)‖
          ∂(ν : Measure (SSphere d)) := norm_integral_le_integral_norm _
      _ = _ := by
        unfold Perspective.partitionMu
        simp only [one_mul]
        apply integral_congr_ae
        exact Eventually.of_forall fun y => by
          change ‖exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d)‖ = _
          rw [norm_smul, mem_sphere_zero_iff_norm.mp y.2, mul_one,
            Real.norm_eq_abs, abs_of_pos (exp_pos _)]
  rw [attentionAverage, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hZ)]
  calc
    _ ≤ (Perspective.partitionMu d 1 ν (B x))⁻¹ * Perspective.partitionMu d 1 ν (B x) :=
      mul_le_mul_of_nonneg_left hnum (inv_pos.mpr hZ).le
    _ = 1 := inv_mul_cancel₀ hZ.ne'

/-- The attention-only original field is this average projected onto
the sphere. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem fullVF_eq_proj_attentionAverage (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) :
    fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) ν t x =
      proj d x (attentionAverage B ν x) := by
  simp only [fullVF, attentionAverage, Perspective.partitionMu, one_mul,
    ContinuousLinearMap.id_apply, zero_apply, add_zero]

/-- Every original attention-only particle has speed at most one.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem norm_fullVF_attention_le_one (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : SSphere d) :
    ‖fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) ν t x‖ ≤ 1 := by
  rw [fullVF_eq_proj_attentionAverage]
  exact (norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) _).trans (norm_attentionAverage_le_one B ν x)

/-- The actual weighted average inherits any almost-everywhere lower
height. Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem le_inner_attentionAverage_of_ae (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x w : EucSpace d) (r : ℝ)
    (hr : ∀ᵐ y : SSphere d ∂(ν : Measure (SSphere d)), r ≤ inner ℝ w (y : EucSpace d)) :
    r ≤ inner ℝ w (attentionAverage B ν x) := by
  let a (y : SSphere d) := exp (inner ℝ (B x) (y : EucSpace d))
  have hi : Integrable (fun y : SSphere d => a y • (y : EucSpace d)) (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_smul d 1 ν (B x)
  have ha : Integrable a (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_ambient d 1 ν (B x)
  have hs : Integrable (fun y : SSphere d => a y * inner ℝ w (y : EucSpace d))
      (ν : Measure (SSphere d)) := by
    convert hi.inner_const w using 1
    funext y
    rw [real_inner_smul_left, real_inner_comm]
  have hb : (∫ y : SSphere d, a y * r ∂(ν : Measure (SSphere d))) ≤
      ∫ y : SSphere d, a y * inner ℝ w (y : EucSpace d) ∂(ν : Measure (SSphere d)) :=
    integral_mono_ae (ha.mul_const r) hs (hr.mono fun y hy => mul_le_mul_of_nonneg_left hy (exp_pos _).le)
  have hv : inner ℝ w (∫ y : SSphere d, a y • (y : EucSpace d) ∂(ν : Measure (SSphere d))) =
      ∫ y : SSphere d, a y * inner ℝ w (y : EucSpace d) ∂(ν : Measure (SSphere d)) := by
    rw [← integral_inner hi]
    simp only [real_inner_smul_right]
  have hZ := Perspective.partitionMu_pos d 1 ν (B x)
  rw [attentionAverage, real_inner_smul_right, hv]
  have hden : ∫ y : SSphere d, a y ∂(ν : Measure (SSphere d)) = Perspective.partitionMu d 1 ν (B x) := by
    simp only [Perspective.partitionMu, one_mul, a]
  rw [integral_mul_const, hden] at hb
  calc
    r = (Perspective.partitionMu d 1 ν (B x))⁻¹ * (Perspective.partitionMu d 1 ν (B x) * r) := by
      rw [← mul_assoc, inv_mul_cancel₀ hZ.ne', one_mul]
    _ ≤ _ := mul_le_mul_of_nonneg_left hb (inv_pos.mpr hZ).le

/-- At a supporting hemisphere boundary, the actual attention field
points inward. Source: arXiv:2411.04551v3, §2, the inward-pointing
argument in the proof of `prop: targets.atoms`. -/
theorem inner_attention_height_nonneg (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x : SSphere d) (w : EucSpace d) (r t : ℝ) (hr : 0 ≤ r)
    (hx : inner ℝ w (x : EucSpace d) = r)
    (hν : ∀ᵐ y : SSphere d ∂(ν : Measure (SSphere d)), r ≤ inner ℝ w (y : EucSpace d)) :
    0 ≤ inner ℝ w (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) ν t x) := by
  have hheight := le_inner_attentionAverage_of_ae B ν x w r hν
  have hinner : inner ℝ (x : EucSpace d) (attentionAverage B ν x) ≤ 1 := by
    have hh := real_inner_le_norm (x : EucSpace d) (attentionAverage B ν x)
    rw [mem_sphere_zero_iff_norm.mp x.2, one_mul] at hh
    exact hh.trans (norm_attentionAverage_le_one B ν x)
  rw [fullVF_eq_proj_attentionAverage, proj, inner_sub_right, real_inner_smul_right, hx]
  nlinarith [mul_nonneg hr (sub_nonneg.mpr hinner)]

/-- An actual atomic probability lies at positive height one, with a
nonzero original attention matrix and a genuine spherical query. -/
example (p : SSphere 2) :
    (0 : ℝ) < 1 ∧
    (∀ᵐ y : SSphere 2 ∂(Perspective.diracProb 2 p : Measure (SSphere 2)),
      (1 : ℝ) ≤ inner ℝ (p : EucSpace 2) (y : EucSpace 2)) ∧
    ‖(p : EucSpace 2)‖ = 1 ∧ inner ℝ (p : EucSpace 2) (p : EucSpace 2) = 1 ∧
    (ContinuousLinearMap.id ℝ (EucSpace 2)) (p : EucSpace 2) = p := by
  have hself : inner ℝ (p : EucSpace 2) (p : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp p.2, mul_one]
  refine ⟨zero_lt_one, ?_, mem_sphere_zero_iff_norm.mp p.2, hself, rfl⟩
  simp [Perspective.diracProb]

end Transformer.Interpolation
