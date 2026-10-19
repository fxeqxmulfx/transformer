module

public import Transformer.Interpolation.Section2_AttentionAverages
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Spatial bounds for original attention weights and their integrals

ArXiv:2411.04551v3, §1, `eq: vf`, and §2, `prop: targets.atoms`.

On a bounded set of queries the original exponential weights have a
uniform spatial Lipschitz bound. The mean value theorem for the scalar
exponential and the matrix operator norm give that bound independently
of the probability law. Integrating the scalar and vector differences
preserves the bound because the law has mass one.

The partition function also has a uniform positive lower bound on this
set. Its normalized weights integrate to exactly one. These estimates
will control the quotient defining the actual attention average, including
for the time-dependent probability supplied by an original weak solution.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A unit key gives the original ambient score's operator-norm bound.
Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem abs_attention_score_ambient_le (B : ParamMatrix d) (x : EucSpace d) (z : SSphere d) :
    |inner ℝ (B x) (z : EucSpace d)| ≤ ‖B‖ * ‖x‖ := by
  calc
    _ ≤ ‖B x‖ * ‖(z : EucSpace d)‖ := abs_real_inner_le_norm _ _
    _ = ‖B x‖ := by rw [mem_sphere_zero_iff_norm.mp z.2, mul_one]
    _ ≤ _ := B.le_opNorm x

/-- The scalar exponential is Lipschitz below a fixed upper score.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem abs_exp_sub_le_on_Iic (M a b : ℝ) (ha : a ≤ M) (hb : b ≤ M) :
    |exp a - exp b| ≤ exp M * |a - b| := by
  have h := Convex.norm_image_sub_le_of_norm_deriv_le
    (f := exp) (s := Iic M) (C := exp M)
    (fun z _ => (hasDerivAt_exp z).differentiableAt)
    (fun z hz => by rw [Real.deriv_exp, Real.norm_eq_abs, abs_of_pos (exp_pos _)]; exact exp_le_exp.mpr hz)
    (convex_Iic M) hb ha
  simpa only [Real.norm_eq_abs] using h

/-- Original exponential weights are uniformly Lipschitz on bounded
query sets. Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem abs_attention_weight_sub_le (B : ParamMatrix d) (R : ℝ) (x y : EucSpace d)
    (hx : ‖x‖ ≤ R) (hy : ‖y‖ ≤ R) (z : SSphere d) :
    |exp (inner ℝ (B x) (z : EucSpace d)) - exp (inner ℝ (B y) (z : EucSpace d))| ≤
      exp (‖B‖ * R) * ‖B‖ * ‖x - y‖ := by
  have hup (v : EucSpace d) (hv : ‖v‖ ≤ R) : inner ℝ (B v) (z : EucSpace d) ≤ ‖B‖ * R :=
    (le_abs_self _).trans ((abs_attention_score_ambient_le B v z).trans
      (mul_le_mul_of_nonneg_left hv (norm_nonneg _)))
  have h := abs_exp_sub_le_on_Iic (‖B‖ * R) _ _ (hup x hx) (hup y hy)
  have hd : |inner ℝ (B x) (z : EucSpace d) - inner ℝ (B y) (z : EucSpace d)| ≤ ‖B‖ * ‖x - y‖ := by
    rw [← inner_sub_left, ← map_sub]
    exact abs_attention_score_ambient_le B (x - y) z
  exact h.trans (by simpa only [mul_assoc] using mul_le_mul_of_nonneg_left hd (exp_pos _).le)

/-- The actual partition stays uniformly positive on a bounded query
set. Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem attention_partition_lower_on_ball (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (R : ℝ) (x : EucSpace d) (hx : ‖x‖ ≤ R) :
    exp (-(‖B‖ * R)) ≤ Perspective.partitionMu d 1 ν (B x) := by
  have hi : Integrable (fun z : SSphere d => exp (inner ℝ (B x) (z : EucSpace d)))
      (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_ambient d 1 ν (B x)
  have h := integral_mono (integrable_const (exp (-(‖B‖ * R)))) hi (fun z => by
    apply exp_le_exp.mpr
    have hh := neg_le_of_abs_le ((abs_attention_score_ambient_le B x z).trans
      (mul_le_mul_of_nonneg_left hx (norm_nonneg _)))
    linarith)
  simpa only [Perspective.partitionMu, one_mul, integral_const, probReal_univ, one_smul] using h

/-- Integrating original weights retains their spatial Lipschitz bound.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem abs_attention_partition_sub_le (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (R : ℝ) (x y : EucSpace d) (hx : ‖x‖ ≤ R) (hy : ‖y‖ ≤ R) :
    |Perspective.partitionMu d 1 ν (B x) - Perspective.partitionMu d 1 ν (B y)| ≤
      exp (‖B‖ * R) * ‖B‖ * ‖x - y‖ := by
  have hi (v : EucSpace d) : Integrable (fun z : SSphere d => exp (inner ℝ (B v) (z : EucSpace d)))
      (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_ambient d 1 ν (B v)
  unfold Perspective.partitionMu
  simp only [one_mul]
  rw [← integral_sub (hi x) (hi y), ← Real.norm_eq_abs]
  simpa only [probReal_univ, mul_one] using norm_integral_le_of_norm_le_const
    (μ := (ν : Measure (SSphere d))) (Eventually.of_forall fun z => by
      change ‖exp (inner ℝ (B x) (z : EucSpace d)) - exp (inner ℝ (B y) (z : EucSpace d))‖ ≤ _
      rw [Real.norm_eq_abs]
      exact abs_attention_weight_sub_le B R x y hx hy z)

/-- The original vector numerator has the same spatial bound as the
weights. Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem norm_attention_numerator_sub_le (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (R : ℝ) (x y : EucSpace d) (hx : ‖x‖ ≤ R) (hy : ‖y‖ ≤ R) :
    ‖(∫ z : SSphere d, exp (inner ℝ (B x) (z : EucSpace d)) • (z : EucSpace d)
      ∂(ν : Measure (SSphere d))) -
      ∫ z : SSphere d, exp (inner ℝ (B y) (z : EucSpace d)) • (z : EucSpace d)
        ∂(ν : Measure (SSphere d))‖ ≤ exp (‖B‖ * R) * ‖B‖ * ‖x - y‖ := by
  have hi (v : EucSpace d) : Integrable
      (fun z : SSphere d => exp (inner ℝ (B v) (z : EucSpace d)) • (z : EucSpace d))
      (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_smul d 1 ν (B v)
  rw [← integral_sub (hi x) (hi y)]
  simpa only [probReal_univ, mul_one] using norm_integral_le_of_norm_le_const
    (μ := (ν : Measure (SSphere d))) (Eventually.of_forall fun z => by
      change ‖exp (inner ℝ (B x) (z : EucSpace d)) • (z : EucSpace d) -
        exp (inner ℝ (B y) (z : EucSpace d)) • (z : EucSpace d)‖ ≤ _
      rw [← sub_smul, norm_smul, mem_sphere_zero_iff_norm.mp z.2, mul_one, Real.norm_eq_abs]
      exact abs_attention_weight_sub_le B R x y hx hy z)

/-- Original normalized attention weights are integrable even for an
ambient query. Auxiliary for arXiv:2411.04551v3, §1, the average in
`eq: vf`, as used in §2, `prop: targets.atoms`. -/
theorem integrable_attention_normalized_weight (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x : EucSpace d) : Integrable
    (fun z : SSphere d => exp (inner ℝ (B x) (z : EucSpace d)) / Perspective.partitionMu d 1 ν (B x))
    (ν : Measure (SSphere d)) := by
  have hi : Integrable (fun z : SSphere d => exp (inner ℝ (B x) (z : EucSpace d)))
      (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_ambient d 1 ν (B x)
  exact hi.div_const _

/-- The genuine normalized attention weights integrate to one.
Source: arXiv:2411.04551v3, §1, the self-attention average in `eq: vf`. -/
theorem integral_attention_normalized_weight (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x : EucSpace d) :
    (∫ z : SSphere d, exp (inner ℝ (B x) (z : EucSpace d)) / Perspective.partitionMu d 1 ν (B x)
      ∂(ν : Measure (SSphere d))) = 1 := by
  simp_rw [div_eq_mul_inv]
  rw [integral_mul_const]
  have he : (∫ z : SSphere d, exp (inner ℝ (B x) (z : EucSpace d)) ∂(ν : Measure (SSphere d))) =
      Perspective.partitionMu d 1 ν (B x) := by simp only [Perspective.partitionMu, one_mul]
  rw [he, mul_inv_cancel₀ (Perspective.partitionMu_pos d 1 ν (B x)).ne']

/-- Distinct antipodal queries, a nonzero attention matrix, an actual
probability law and finite real score bounds realize all spatial premises. -/
example : ∃ (B : ParamMatrix 2) (ν : Perspective.ProbSphere 2) (x y : SSphere 2),
    ‖(x : EucSpace 2)‖ ≤ 1 ∧ ‖(y : EucSpace 2)‖ ≤ 1 ∧ x ≠ y ∧
    B (x : EucSpace 2) = x ∧ (ν : Measure (SSphere 2)) univ = 1 ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 := by
  let p : SSphere 2 := basePoint 1
  refine ⟨ContinuousLinearMap.id ℝ (EucSpace 2), Perspective.diracProb 2 p,
    p, antipode 2 p, (mem_sphere_zero_iff_norm.mp p.2).le,
    (mem_sphere_zero_iff_norm.mp (antipode 2 p).2).le, (antipode_ne 2 p).symm,
    rfl, IsProbabilityMeasure.measure_univ, zero_le_one, le_rfl⟩

end Transformer.Interpolation
