/-
# Contraction of actual two-neuron trajectories in the inner cap

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
and `eq: dissipation.sphere` for the original projected perceptron.

The explicit radial coefficient dominates the quadratic calibration
error. On the inner core this gives radial dissipation with rate `1/2`.
The inward boundary estimate keeps the actual flow in that core, so the
original particle equation yields exponential decay of its distance to
the prescribed target and convergence as time tends to infinity.

The field retains both true ReLU activations throughout this argument.
-/

import Transformer.Interpolation.Section3_CapError
import Transformer.Interpolation.Section3_CapBoundary

open Real Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The calibrated original field contracts toward the target in the core.
Auxiliary for arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere`
and the constant-control construction toward `lem: two.balls`. -/
theorem twoRidge_cap_core_contraction (z w ω x : SSphere d) (c h σ ℓ : ℝ)
    (hh : 0 ≤ h) (hσ : 0 ≤ σ) (hℓ : 0 ≤ ℓ)
    (hb : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)
    (hα : 0 ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hx : σ ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) - h)
    (hlarge : 2 * ‖(z : EucSpace d) -
        ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) • (w : EucSpace d)‖ + 1 ≤
        ℓ * σ * ((h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) :
    inner ℝ (twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c)
      (capTargetOutput z w ω c h ℓ) (-(w : EucSpace d)) (-h) x)
      ((x : EucSpace d) - (ω : EucSpace d)) ≤
        -((1 / 2 : ℝ) * ‖(x : EucSpace d) - (ω : EucSpace d)‖ ^ 2) := by
  let a := inner ℝ (ω : EucSpace d) (z : EucSpace d) - c
  let b := inner ℝ (ω : EucSpace d) (w : EucSpace d) - h
  let A := ‖(z : EucSpace d) - (a / b) • (w : EucSpace d)‖
  let β := inner ℝ (x : EucSpace d) (w : EucSpace d) - h
  let ζ := inner ℝ (x : EucSpace d) (ω : EucSpace d)
  let κ := (h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2
  let r := (inner ℝ (x : EucSpace d) (z : EucSpace d) - c) - (a / b) * β
  let D := inner ℝ (proj d x ω) (z : EucSpace d)
  let n := ‖(x : EucSpace d) - (ω : EucSpace d)‖
  have hβ : 0 ≤ β := hσ.trans hx
  have hκ : 0 ≤ κ := div_nonneg (sq_nonneg _) (by norm_num)
  have hk := one_add_inner_lower_of_common_cap ω w x h (by linarith) (by linarith)
  have hm := mul_le_mul hx hk hκ hβ
  have hs := mul_le_mul_of_nonneg_left hm (div_nonneg hℓ hb.le)
  have hmain : 2 * A + 1 ≤ (ℓ * β / b) * (1 + ζ) := by
    refine hlarge.trans ?_
    change ℓ * σ * κ / b ≤ ℓ * β / b * (1 + ζ)
    change ℓ / b * (σ * κ) ≤ ℓ / b * (β * (1 + ζ)) at hs
    convert hs using 1 <;> ring
  have he := cap_calibration_error_le z w ω x c h hb
  have hd := abs_inner_proj_target_le x ω z
  have herr := mul_le_mul he hd (abs_nonneg D) (mul_nonneg (norm_nonneg _) (norm_nonneg _))
  change |r| * |D| ≤ (A * n) * n at herr
  have hneg := neg_le_abs (r * D)
  rw [abs_mul] at hneg
  have hprod : -r * D ≤ A * n ^ 2 := by nlinarith
  have hmn := mul_le_mul_of_nonneg_right hmain (sq_nonneg n)
  rw [inner_twoRidge_cap_sub_target z w ω x c h ℓ hα hβ]
  change -(ℓ * β / b * (1 + ζ) / 2) * n ^ 2 - r * D ≤ -((1 / 2 : ℝ) * n ^ 2)
  nlinarith

/-- A particle started in the core stays there and its actual distance
to the target decays at rate `1/2`. Auxiliary for arXiv:2411.04551v3,
Appendix, the large-time concentration step in `lem: two.balls`. -/
theorem norm_twoRidge_cap_core_sub_le (z w ω x₀ : SSphere d) (c h m σ ℓ : ℝ)
    (hh : 0 ≤ h) (hm : 0 < m) (hσ : 0 < σ) (hℓ : 0 ≤ ℓ)
    (hb : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)
    (hσz : σ < inner ℝ (z : EucSpace d) (w : EucSpace d) - h)
    (hmargin : ∀ x : SSphere d, h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) →
      m ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hq : (h + σ) * ‖capTargetOutput z w ω c h ℓ‖ ≤
      inner ℝ (capTargetOutput z w ω c h ℓ) (w : EucSpace d))
    (hlarge : 2 * ‖(z : EucSpace d) -
        ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) • (w : EucSpace d)‖ + 1 ≤
        ℓ * σ * ((h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h))
    (h0 : σ ≤ inner ℝ (x₀ : EucSpace d) (w : EucSpace d) - h) (t : ℝ) (ht : 0 ≤ t) :
    ‖(twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) (capTargetOutput z w ω c h ℓ)
      (-(w : EucSpace d)) (-h) x₀ t : EucSpace d) - (ω : EucSpace d)‖ ≤
        ‖(x₀ : EucSpace d) - (ω : EucSpace d)‖ * exp (-((1 / 2 : ℝ) * t)) := by
  let q := capTargetOutput z w ω c h ℓ
  let X := twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀
  let v (s : ℝ) := twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) (X s)
  have hcontr (s : ℝ) (hs : 0 ≤ s) : inner ℝ (v s) ((X s : EucSpace d) - (ω : EucSpace d)) ≤
      -((1 / 2 : ℝ) * ‖(X s : EucSpace d) - (ω : EucSpace d)‖ ^ 2) := by
    have hcore := twoRidge_cap_strip_invariant z w x₀ q c h m σ σ hh hm hσz hmargin hq
      hσ le_rfl h0 s hs
    have ha := hmargin (X s) (by linarith)
    exact twoRidge_cap_core_contraction z w ω (X s) c h σ ℓ hh hσ.le hℓ hb
      (hm.le.trans ha) hcore hlarge
  have he := norm_sub_le_of_contraction d (fun s => (X s : EucSpace d)) v ω (1 / 2)
    (fun s _ => hasDerivAt_twoRidgeFlow _ _ _ _ _ _ _ s) hcontr t ht
  simpa only [X, twoRidgeFlow_zero] using he

/-- Every actual core trajectory converges to the prescribed target.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem tendsto_twoRidge_cap_core (z w ω x₀ : SSphere d) (c h m σ ℓ : ℝ)
    (hh : 0 ≤ h) (hm : 0 < m) (hσ : 0 < σ) (hℓ : 0 ≤ ℓ)
    (hb : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)
    (hσz : σ < inner ℝ (z : EucSpace d) (w : EucSpace d) - h)
    (hmargin : ∀ x : SSphere d, h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) →
      m ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hq : (h + σ) * ‖capTargetOutput z w ω c h ℓ‖ ≤
      inner ℝ (capTargetOutput z w ω c h ℓ) (w : EucSpace d))
    (hlarge : 2 * ‖(z : EucSpace d) -
        ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) • (w : EucSpace d)‖ + 1 ≤
        ℓ * σ * ((h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h))
    (h0 : σ ≤ inner ℝ (x₀ : EucSpace d) (w : EucSpace d) - h) :
    Tendsto (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) (capTargetOutput z w ω c h ℓ)
      (-(w : EucSpace d)) (-h) x₀) atTop (nhds ω) := by
  have hlin : Tendsto (fun t : ℝ => (1 / 2 : ℝ) * t) atTop atTop :=
    (tendsto_const_mul_atTop_of_pos (by norm_num : (0 : ℝ) < 1 / 2)).2 tendsto_id
  have he : Tendsto (fun t : ℝ => ‖(x₀ : EucSpace d) - (ω : EucSpace d)‖ * exp (-((1 / 2 : ℝ) * t)))
      atTop (nhds 0) := by
    simpa only [Function.comp_def, mul_zero] using
      (tendsto_exp_neg_atTop_nhds_zero.comp hlin).const_mul ‖(x₀ : EucSpace d) - (ω : EucSpace d)‖
  apply Metric.tendsto_nhds.mpr
  intro ε hε
  filter_upwards [he.eventually (gt_mem_nhds hε), eventually_ge_atTop (0 : ℝ)] with t ht ht0
  exact (norm_twoRidge_cap_core_sub_le z w ω x₀ c h m σ ℓ hh hm hσ hℓ hb hσz hmargin hq hlarge h0 t ht0).trans_lt ht

/-- A genuine nested cap and finite explicit radial coefficient jointly
satisfy all field, invariance, contraction and initial-core hypotheses. -/
example : ∃ z : SSphere 2,
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) ≤ 100 ∧
    0 < inner ℝ (z : EucSpace 2) (z : EucSpace 2) - 1 / 2 ∧
    (1 / 4 : ℝ) < inner ℝ (z : EucSpace 2) (z : EucSpace 2) - 1 / 2 ∧
    (∀ x : SSphere 2, (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (z : EucSpace 2) →
      (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (z : EucSpace 2) - 0) ∧
    (1 / 2 + 1 / 4 : ℝ) * ‖capTargetOutput z z z 0 (1 / 2) 100‖ ≤
      inner ℝ (capTargetOutput z z z 0 (1 / 2) 100) (z : EucSpace 2) ∧
    2 * ‖(z : EucSpace 2) - (2 : ℝ) • (z : EucSpace 2)‖ + 1 ≤
      (100 : ℝ) * (1 / 4) * ((1 / 2 + 1) ^ 2 / 2) / (1 / 2) ∧
    0 ≤ inner ℝ (z : EucSpace 2) (z : EucSpace 2) - 0 ∧
    (1 / 4 : ℝ) ≤ inner ℝ (z : EucSpace 2) (z : EucSpace 2) - 1 / 2 ∧ (0 : ℝ) ≤ 1 := by
  let z : SSphere 2 := basePoint 1
  have hn : ‖(z : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp z.2
  have hi : inner ℝ (z : EucSpace 2) (z : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, hn]; norm_num
  have he : (z : EucSpace 2) - (2 : ℝ) • (z : EucSpace 2) = -(z : EucSpace 2) := by module
  have hq : capTargetOutput z z z 0 (1 / 2) 100 = (198 : ℝ) • (z : EucSpace 2) := by
    simp only [capTargetOutput, hi]
    norm_num
    module
  refine ⟨z, by norm_num, by norm_num, by norm_num, by norm_num, ?_, ?_, ?_, ?_, ?_, ?_, ?_, zero_le_one⟩
  · rw [hi]; norm_num
  · rw [hi]; norm_num
  · intro x hx; simpa only [sub_zero] using hx
  · rw [hq, norm_smul, real_inner_smul_left, hn, hi]; norm_num
  · rw [he, norm_neg, hn]; norm_num
  · rw [hi]; norm_num
  · rw [hi]; norm_num

end Transformer.Interpolation
