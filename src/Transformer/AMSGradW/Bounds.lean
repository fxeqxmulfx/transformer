/-
# Bounds derived from decay and the objective

AMSGradW extension of arXiv:1904.03590v4, Algorithm 1 and §4.
The sufficient regime `L < decay*epsilon` bounds actual weights and
moments without assuming anything about the optimizer trajectory.
-/

import Transformer.AMSGradW.Basic

noncomputable section

namespace Transformer.AMSGradW

variable {d : ℕ}

/-- Coordinate division by a positive denominator controls the maximum
norm. Source: arXiv:1904.03590v4, §6, AMSGradW extension. -/
theorem division_norm_bound (ε : ℝ) (m D : Fin d → ℝ)
    (hε : 0 < ε) (hD : ∀ i, ε ≤ D i) :
    ‖fun i => m i / D i‖ ≤ ‖m‖ / ε := by
  apply (pi_norm_le_iff_of_nonneg (div_nonneg (norm_nonneg _) hε.le)).mpr
  intro i
  rw [norm_div, show ‖D i‖ = D i by
    exact Real.norm_of_nonneg (hε.trans_le (hD i)).le]
  exact (div_le_div_of_nonneg_left (norm_nonneg _) hε (hD i)).trans
    (div_le_div_of_nonneg_right (norm_le_pi_norm m i) hε.le)

/-- Nonzero division-bound witness, arXiv:1904.03590v4, §6, AMSGradW. -/
example : (0 : ℝ) < 1 ∧ (∀ i : Fin 1, (1 : ℝ) ≤ (fun _ => 2) i) := by
  exact ⟨by norm_num, fun i => by norm_num⟩

/-- The objective controls its gradient at every point from its value
at zero. Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
theorem gradient_norm_bound (f : TrainingSpace d → ℝ) (L : ℝ)
    (hgrad : LipschitzGradient f L) (x : Fin d → ℝ) :
    ‖coordinateGradient f x‖ ≤ L * ‖x‖ + ‖coordinateGradient f 0‖ := by
  have h : ‖coordinateGradient f x - coordinateGradient f 0‖ ≤ L * ‖x‖ := by
    simpa only [sub_zero] using hgrad x 0
  exact (norm_le_norm_sub_add (coordinateGradient f x) (coordinateGradient f 0)).trans
    (add_le_add h le_rfl)

/-- Nonconstant objective witness, arXiv:1904.03590v4, §4, AMSGradW. -/
example : LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 :=
  energy_lipschitz

/-- Genuine EMA momentum is bounded by the old momentum and gradient.
Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW extension. -/
theorem momentum_norm_bound (η ε wd β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (hβ : 0 ≤ β) (hβ' : β ≤ 1) :
    ‖(trainingStep η ε wd β β₂ f s).momentum‖ ≤
      β * ‖s.momentum‖ + (1 - β) * ‖coordinateGradient f (WithLp.ofLp s.position)‖ := by
  have heq : (trainingStep η ε wd β β₂ f s).momentum =
      β • s.momentum + (1 - β) • coordinateGradient f (WithLp.ofLp s.position) := by
    ext i
    rfl
  rw [heq]
  simpa only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hβ,
    abs_of_nonneg (sub_nonneg.mpr hβ')] using norm_add_le
      (β • s.momentum) ((1 - β) • coordinateGradient f (WithLp.ofLp s.position))

/-- Positive momentum witness, arXiv:1904.03590v4, Algorithm 1, AMSGradW. -/
example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 := by norm_num

/-- The decay contraction and actual adaptive momentum control the
next parameter. Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW. -/
theorem position_norm_bound (η ε wd β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (hη : 0 ≤ η) (hε : 0 < ε) (hstep : η * wd ≤ 1) :
    ‖WithLp.ofLp (trainingStep η ε wd β β₂ f s).position‖ ≤
      (1 - η * wd) * ‖WithLp.ofLp s.position‖ +
        η / ε * ‖(trainingStep η ε wd β β₂ f s).momentum‖ := by
  let next := trainingStep η ε wd β β₂ f s
  let D := AMSGrad.trainingDenominator ε next
  have heq : WithLp.ofLp next.position = (1 - η * wd) • WithLp.ofLp s.position -
      η • (fun i => next.momentum i / D i) := by
    ext i
    simpa only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul, mul_div_assoc] using
      (trainingStep_coordinates η ε wd β β₂ f s i).2
  rw [heq]
  have h := norm_sub_le ((1 - η * wd) • WithLp.ofLp s.position)
    (η • (fun i => next.momentum i / D i))
  simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr hstep),
    abs_of_nonneg hη] at h
  have hdiv := division_norm_bound ε next.momentum D hε (denominator_lower ε next)
  have hm := mul_le_mul_of_nonneg_left hdiv hη
  have hid : η * (‖next.momentum‖ / ε) = η / ε * ‖next.momentum‖ := by ring
  rw [hid] at hm
  dsimp only [next] at *
  nlinarith

/-- Nonzero decay/step witness, arXiv:1904.03590v4, Algorithm 1, AMSGradW. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) < 1 ∧ (1 / 4 : ℝ) * 2 ≤ 1 := by norm_num

/-- Inductive actual-state bound from initial data and objective
Lipschitz continuity. No bounded-iterate premise occurs.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
theorem trainingRun_bound (η ε wd β β₂ L B : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hgrad : LipschitzGradient f L)
    (hη : 0 ≤ η) (hε : 0 < ε) (hwd : 0 ≤ wd) (hL : 0 ≤ L)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hstep : η * wd ≤ 1) (hB : 0 ≤ B)
    (hinit : ‖WithLp.ofLp initial‖ ≤ B)
    (hzero : ‖coordinateGradient f 0‖ ≤ (wd * ε - L) * B) (t : ℕ) :
    ‖WithLp.ofLp (trainingRun η ε wd β β₂ f initial t).position‖ ≤ B ∧
      ‖(trainingRun η ε wd β β₂ f initial t).momentum‖ ≤ wd * ε * B := by
  induction t with
  | zero => exact ⟨hinit, by simpa [trainingRun] using mul_nonneg (mul_nonneg hwd hε.le) hB⟩
  | succ t ih =>
    let s := trainingRun η ε wd β β₂ f initial t
    have hg := (gradient_norm_bound f L hgrad (WithLp.ofLp s.position)).trans
      (add_le_add (mul_le_mul_of_nonneg_left ih.1 hL) le_rfl)
    have hg' : ‖coordinateGradient f (WithLp.ofLp s.position)‖ ≤ wd * ε * B := by
      linarith
    have hm := momentum_norm_bound η ε wd β β₂ f s hβ hβ'
    have hm' : ‖(trainingStep η ε wd β β₂ f s).momentum‖ ≤ wd * ε * B := by
      have h1 := mul_le_mul_of_nonneg_left ih.2 hβ
      have h2 := mul_le_mul_of_nonneg_left hg' (sub_nonneg.mpr hβ')
      dsimp only [s] at *
      nlinarith
    refine ⟨?_, hm'⟩
    have hx := position_norm_bound η ε wd β β₂ f s hη hε hstep
    have h1 := mul_le_mul_of_nonneg_left ih.1 (sub_nonneg.mpr hstep)
    have h2 := mul_le_mul_of_nonneg_left hm' (div_nonneg hη hε.le)
    have hid : (1 - η * wd) * B + η / ε * (wd * ε * B) = B := by
      field_simp
      ring
    change ‖WithLp.ofLp (trainingStep η ε wd β β₂ f s).position‖ ≤ B
    linarith

/-- The complete bound assumptions hold with nonzero initial weights,
momentum and decay. Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧ (1 / 4 : ℝ) * 2 ≤ 1 ∧
    (0 : ℝ) ≤ 1 ∧ ‖(fun _ : Fin 1 => (1 : ℝ))‖ ≤ 1 ∧
    ‖coordinateGradient (Optimization.energy : TrainingSpace 1 → ℝ) 0‖ ≤ (2 * 1 - 1) * 1 := by
  refine ⟨energy_lipschitz, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, ?_, ?_⟩
  · exact (pi_norm_le_iff_of_nonneg (by norm_num)).mpr fun i => by norm_num
  · simp [coordinateGradient, Optimization.energy_gradient]

end Transformer.AMSGradW
