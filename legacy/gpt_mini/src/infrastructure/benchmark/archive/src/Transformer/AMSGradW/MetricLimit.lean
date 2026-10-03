/-
# A finite limit for the actual AMSGradW metric

arXiv:1904.03590v4, Algorithm 1 and §6, AMSGradW extension.
All bounds follow from the actual run, objective and initial data.
-/

import Transformer.AMSGradW.Bounds

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGradW

variable {d : ℕ}

/-- An explicit finite radius from initial data; the denominator is
positive in the decay-dominated regime. Source: arXiv:1904.03590v4,
§4, AMSGradW convergence extension. -/
def radius (ε wd L : ℝ) (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) : ℝ :=
  ‖WithLp.ofLp initial‖ + ‖coordinateGradient f 0‖ / (wd * ε - L)

/-- The radius supplies every initialization bound, without an optimizer
trajectory assumption. Source: arXiv:1904.03590v4, §4, AMSGradW. -/
theorem radius_bounds (ε wd L : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hdom : L < wd * ε) :
    0 ≤ radius ε wd L f initial ∧ ‖WithLp.ofLp initial‖ ≤ radius ε wd L f initial ∧
      ‖coordinateGradient f 0‖ ≤ (wd * ε - L) * radius ε wd L f initial := by
  have hp : 0 < wd * ε - L := sub_pos.mpr hdom
  have hn := div_nonneg (norm_nonneg (coordinateGradient f 0)) hp.le
  refine ⟨add_nonneg (norm_nonneg _) hn, le_add_of_nonneg_right hn, ?_⟩
  have he : (wd * ε - L) * (‖coordinateGradient f 0‖ / (wd * ε - L)) =
      ‖coordinateGradient f 0‖ := by field_simp
  dsimp only [radius]
  nlinarith [mul_nonneg hp.le (norm_nonneg (WithLp.ofLp initial))]

/-- Strict domination has positive-decay witnesses,
arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : (1 : ℝ) < 2 * 1 := by norm_num

/-- Bounds for the actual first, second and running-maximum histories.
No bound on gradients or parameters along the run is assumed.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
theorem trainingRun_history_bound (η ε wd β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hgrad : LipschitzGradient f L)
    (hη : 0 ≤ η) (hε : 0 < ε) (hwd : 0 ≤ wd) (hL : 0 ≤ L)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1)
    (hstep : η * wd ≤ 1) (hdom : L < wd * ε) (t : ℕ) (i : Fin d) :
    let C := wd * ε * radius ε wd L f initial
    0 ≤ (trainingRun η ε wd β β₂ f initial t).second i ∧
      0 ≤ (trainingRun η ε wd β β₂ f initial t).maximum i ∧
      (trainingRun η ε wd β β₂ f initial t).second i ≤ C ^ 2 ∧
      (trainingRun η ε wd β β₂ f initial t).maximum i ≤ C ^ 2 := by
  let B := radius ε wd L f initial
  let C := wd * ε * B
  obtain ⟨hB, hi, hz⟩ := radius_bounds ε wd L f initial hdom
  have hg : ∀ k, ‖coordinateGradient f
      (WithLp.ofLp (trainingRun η ε wd β β₂ f initial k).position)‖ ≤ C := by
    intro k
    have hb := (trainingRun_bound η ε wd β β₂ L B f initial hgrad hη hε hwd hL
      hβ hβ' hstep hB hi hz k).1
    have hg := gradient_norm_bound f L hgrad
      (WithLp.ofLp (trainingRun η ε wd β β₂ f initial k).position)
    have hm := mul_le_mul_of_nonneg_left hb hL
    dsimp only [C, B] at *
    linarith
  induction t with
  | zero => exact ⟨le_rfl, le_rfl, sq_nonneg C, sq_nonneg C⟩
  | succ t ih =>
    let s := trainingRun η ε wd β β₂ f initial t
    let g := gradient f s.position i
    have hab : |g| ≤ C := (norm_le_pi_norm (coordinateGradient f
      (WithLp.ofLp s.position)) i).trans (hg t)
    have hsq : g ^ 2 ≤ C ^ 2 := by
      have hc := (norm_nonneg (coordinateGradient f (WithLp.ofLp s.position))).trans (hg t)
      nlinarith [sq_abs g, abs_nonneg g]
    have hv0 : 0 ≤ β₂ * s.second i + (1 - β₂) * g ^ 2 :=
      add_nonneg (mul_nonneg hβ₂ ih.1) (mul_nonneg (sub_nonneg.mpr hβ₂') (sq_nonneg g))
    have hv : β₂ * s.second i + (1 - β₂) * g ^ 2 ≤ C ^ 2 := by
      have h1 := mul_le_mul_of_nonneg_left ih.2.2.1 hβ₂
      have h2 := mul_le_mul_of_nonneg_left hsq (sub_nonneg.mpr hβ₂')
      dsimp only [C, B, s, g] at *
      nlinarith
    exact ⟨hv0, le_trans ih.2.1 (le_max_left _ _), hv, max_le ih.2.2.2 hv⟩

/-- All history-bound assumptions hold with positive decay and momentum,
arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
example : LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (1 / 4 : ℝ) * 2 ≤ 1 ∧ (1 : ℝ) < 2 * 1 := by
  exact ⟨energy_lipschitz, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- The actual, history-dependent denominator converges to a finite
positive vector; its convergence is a conclusion, not a hypothesis.
Source: arXiv:1904.03590v4, Algorithm 1 and §6, AMSGradW extension. -/
theorem trainingRun_metric_limit (η ε wd β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hgrad : LipschitzGradient f L)
    (hη : 0 ≤ η) (hε : 0 < ε) (hwd : 0 ≤ wd) (hL : 0 ≤ L)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1)
    (hstep : η * wd ≤ 1) (hdom : L < wd * ε) :
    ∃ D : Fin d → ℝ, (∀ i, ε ≤ D i) ∧
      Tendsto (fun t => AMSGrad.trainingDenominator ε
        (trainingRun η ε wd β β₂ f initial t)) atTop (𝓝 D) := by
  let a := fun t => AMSGrad.trainingDenominator ε (trainingRun η ε wd β β₂ f initial t)
  have hm : ∀ i, Monotone (fun t => a t i) := by
    intro i
    apply monotone_nat_of_le_succ
    intro t
    exact add_le_add le_rfl (Real.sqrt_le_sqrt (le_max_left _ _))
  have hb : ∀ i, BddAbove (Set.range (fun t => a t i)) := by
    intro i
    refine ⟨ε + |wd * ε * radius ε wd L f initial|, ?_⟩
    rintro y ⟨t, rfl⟩
    have h := (trainingRun_history_bound η ε wd β β₂ L f initial hgrad hη hε hwd hL
      hβ hβ' hβ₂ hβ₂' hstep hdom t i).2.2.2
    exact add_le_add le_rfl ((Real.sqrt_le_sqrt h).trans_eq (Real.sqrt_sq_eq_abs _))
  refine ⟨fun i => ⨆ t, a t i, fun i => ?_, tendsto_pi_nhds.mpr fun i =>
    tendsto_atTop_ciSup (hm i) (hb i)⟩
  exact (denominator_lower ε (trainingRun η ε wd β β₂ f initial 0) i).trans
    (le_ciSup (hb i) 0)

/-- Metric-limit hypotheses are inhabited, arXiv:1904.03590v4, §6, AMSGradW. -/
example : LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (1 / 4 : ℝ) * 2 ≤ 1 ∧ (1 : ℝ) < 2 * 1 := by
  exact ⟨energy_lipschitz, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

end Transformer.AMSGradW
