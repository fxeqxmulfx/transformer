/-
# Exact projection transfers between original unit atoms

ArXiv:2411.04551v3, §4, the point matching in
`prop: interpolation.neural.ode`, using the constant tangent field
of §3.1, `lem: first.quadrant`.
Two distinct non-antipodal unit points admit a single exact positive
projection-flow transfer. Its axis is the normalized chord and is
orthogonal to their midpoint. The clock is the logarithm of the ratio
(2+r)/(2-r), where r is their actual chordal distance.

The explicit hyperbolic formula gives denominator one at this clock
and numerator equal to the target. The actual original rank-one
neural controls realize the transfer, including all original weak
solutions from the input atom. This is one transfer; finite-data
localization and protection of other points are separate steps.
-/

import Transformer.Interpolation.Section3_ProjectionControls
import Transformer.Interpolation.Section1_CloseAtomicInputs
import Mathlib.Analysis.SpecialFunctions.Log.Basic

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Distinct non-antipodal original unit points have chord length
strictly between zero and two. Source auxiliary:
arXiv:2411.04551v3, §4, the unit point interpolation geometry. -/
theorem chord_norm_pos_lt_two (p q : SSphere d)
    (hpq : p ≠ q) (hanti : q ≠ antipode d p) :
    0 < ‖(q : EucSpace d) - (p : EucSpace d)‖ ∧
      ‖(q : EucSpace d) - (p : EucSpace d)‖ < 2 := by
  have hp : ‖(p : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hq : ‖(q : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp q.2
  have hne : (q : EucSpace d) - (p : EucSpace d) ≠ 0 := by
    intro he
    exact hpq (Subtype.ext (sub_eq_zero.mp he).symm)
  have hplus : (q : EucSpace d) + (p : EucSpace d) ≠ 0 := by
    intro he
    exact hanti (Subtype.ext (eq_neg_of_add_eq_zero_left he))
  have hpos := pow_pos (norm_pos_iff.mpr hplus) 2
  have hs := norm_sub_sq_real (q : EucSpace d) (p : EucSpace d)
  have ha := norm_add_sq_real (q : EucSpace d) (p : EucSpace d)
  rw [hp, hq] at hs ha
  refine ⟨norm_pos_iff.mpr hne, ?_⟩
  nlinarith

/-- The actual constant tangent projection transfers any distinct
non-antipodal original unit pair in positive finite time. Its axis is
perpendicular to the pair's midpoint. Source auxiliary:
arXiv:2411.04551v3, §4, the point matching step. -/
theorem exists_exact_projection_transfer (p q : SSphere d)
    (hpq : p ≠ q) (hanti : q ≠ antipode d p) :
    ∃ (a : SSphere d) (t : ℝ), 0 < t ∧ projectionDenom (a : EucSpace d) t p = 1 ∧
      sphereProjectionFlow a t p = q ∧
      inner ℝ (a : EucSpace d) ((p : EucSpace d) + (q : EucSpace d)) = 0 := by
  let r := ‖(q : EucSpace d) - (p : EucSpace d)‖
  obtain ⟨hr, hr2⟩ := chord_norm_pos_lt_two p q hpq hanti
  have hr0 : r ≠ 0 := ne_of_gt hr
  have hden : 0 < 2 - r := by linarith
  have hden0 : 2 - r ≠ 0 := ne_of_gt hden
  have hsum0 : 2 + r ≠ 0 := by positivity
  have hquad : 4 - r ^ 2 ≠ 0 := by nlinarith
  let a : SSphere d := ⟨r⁻¹ • ((q : EucSpace d) - (p : EucSpace d)), by
    rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, Real.norm_of_nonneg hr.le]
    exact inv_mul_cancel₀ hr0⟩
  have hp : ‖(p : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hq : ‖(q : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp q.2
  have hs : r ^ 2 = 2 - 2 * inner ℝ (q : EucSpace d) (p : EucSpace d) := by
    dsimp only [r]
    rw [norm_sub_sq_real, hp, hq]
    ring
  have hap : inner ℝ (a : EucSpace d) (p : EucSpace d) = -r / 2 := by
    change inner ℝ (r⁻¹ • ((q : EucSpace d) - (p : EucSpace d))) (p : EucSpace d) = _
    rw [real_inner_smul_left, inner_sub_left, real_inner_self_eq_norm_sq, hp, one_pow]
    have hh : inner ℝ (q : EucSpace d) (p : EucSpace d) - 1 = -(r ^ 2) / 2 := by linarith
    rw [hh]
    field_simp
  let t := log ((2 + r) / (2 - r))
  have hratio : 1 < (2 + r) / (2 - r) := by
    rw [lt_div_iff₀ hden]
    linarith
  have ht : 0 < t := log_pos hratio
  have hc : cosh t = (4 + r ^ 2) / (4 - r ^ 2) := by
    dsimp only [t]
    rw [cosh_log (zero_lt_one.trans hratio)]
    field_simp
    ring
  have hsh : sinh t = (4 * r) / (4 - r ^ 2) := by
    dsimp only [t]
    rw [sinh_log (zero_lt_one.trans hratio)]
    field_simp
    ring
  have hD : projectionDenom (a : EucSpace d) t p = 1 := by
    rw [projectionDenom, hap, hc, hsh]
    field_simp
    ring
  have hN : sinh t + (cosh t - 1) * inner ℝ (a : EucSpace d) (p : EucSpace d) = r := by
    rw [hap, hc, hsh]
    field_simp
    ring
  refine ⟨a, t, ht, hD, ?_, ?_⟩
  · apply Subtype.ext
    change projectionFlow (a : EucSpace d) t p = (q : EucSpace d)
    rw [projectionFlow, hD, inv_one, one_smul, projectionNumerator, hN]
    change (p : EucSpace d) + r • (r⁻¹ • ((q : EucSpace d) - (p : EucSpace d))) = _
    rw [smul_smul, mul_inv_cancel₀ hr0, one_smul]
    abel
  · change inner ℝ (r⁻¹ • ((q : EucSpace d) - (p : EucSpace d)))
      ((p : EucSpace d) + (q : EucSpace d)) = 0
    rw [real_inner_smul_left, inner_sub_left, inner_add_right, inner_add_right,
      real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq, hp, hq]
    rw [real_inner_comm (q : EucSpace d) (p : EucSpace d)]
    ring

/-- Concrete original rank-one neural controls realize the exact
transfer and determine every original weak endpoint. Source auxiliary:
arXiv:2411.04551v3, §4, the single-pair matching step. -/
theorem exists_exact_atomic_projection_control (i : Fin d) (p q : SSphere d)
    (hpq : p ≠ q) (hanti : q ≠ antipode d p) :
    ∃ (a : SSphere d) (t : ℝ), 0 < t ∧
      (∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = Perspective.diracProb d p ∧
        cauchyPB d (fun _ => projectionControl i (a : EucSpace d)) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = Perspective.diracProb d p →
        cauchyPB d (fun _ => projectionControl i (a : EucSpace d)) μ →
        μ t = Perspective.diracProb d q := by
  obtain ⟨a, t, ht, _, hmap, _⟩ := exists_exact_projection_transfer p q hpq hanti
  have hc := cauchyPB_projectionControl i a (Perspective.diracProb d p) 1
  refine ⟨a, t, ht, ⟨(fun s => projectionLaw a (Perspective.diracProb d p) (1 * s)), ?_⟩, ?_⟩
  · simpa only [one_smul] using hc
  · intro μ h0 hμ
    apply Subtype.ext
    have h := projectionControl_endpoint i a (Perspective.diracProb d p) 1 t ht μ h0
      (by simpa only [one_smul] using hμ)
    rw [one_mul, coe_projectionLaw] at h
    change (μ t : Measure (SSphere d)) = Measure.map (sphereProjectionFlow a t) (Measure.dirac p) at h
    rw [Measure.map_dirac, hmap] at h
    exact h

/-- Distinct orthogonal actual unit atoms realize all non-antipodal
transfer hypotheses, with an available original neuron coordinate. -/
example : ∃ p q : SSphere 3, p ≠ q ∧ q ≠ antipode 3 p := by
  let p := atomicCirclePoint 0
  let q := atomicCirclePoint (π / 2)
  have h0 : (0 : ℝ) < π / 2 := by positivity
  have hπ : π / 2 < π := by linarith [pi_pos]
  refine ⟨p, q, (atomicCirclePoint_ne_zero (π / 2) h0 hπ).symm, ?_⟩
  intro he
  have hc := congrArg (fun x : SSphere 3 => (x : EucSpace 3) (1 : Fin 3)) he
  norm_num [p, q, atomicCirclePoint, atomicCircleVector, antipode] at hc

end Transformer.Interpolation
