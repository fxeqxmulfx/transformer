module

public import Transformer.SequenceModel.Attention.Model

/-!
# Local regularity of the actual normalizers

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, R1.
Positive-epsilon RMSNorm is analytic everywhere. The clipped L2 normalizer
is analytic on both open sides of its clipping sphere, including zero on
the clipped side. The boundary is excluded explicitly, not assigned a
classical derivative by convention.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini Landscape.Neural
open scoped BigOperators

variable {P : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P]
  {d : ℕ} {v : P → EucSpace d} {x : P}

/-- R1: squared Euclidean norm is analytic in varying scalar coordinates. -/
theorem analyticAt_normSq (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x) :
    AnalyticAt ℝ (fun w => ‖v w‖ ^ 2) x := by
  simpa only [EuclideanSpace.norm_sq_eq, Real.norm_eq_abs, sq_abs] using
    (Finset.analyticAt_fun_sum Finset.univ (fun i _ => (hv i).fun_pow 2))

/-- R1: coordinatewise regularity gives continuous norm, also at zero. -/
theorem continuousAt_norm_of_analytic
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x) :
    ContinuousAt (fun w => ‖v w‖) x := by
  simpa only [Real.sqrt_sq (norm_nonneg _)] using
    (analyticAt_normSq hv).continuousAt.sqrt

/-- R1: the actual positive-epsilon RMSNorm, without dropping epsilon. -/
theorem analyticAt_rmsNormEps (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x) (i : Fin d) :
    AnalyticAt ℝ (fun w => rmsNormEps eps (v w) i) x := by
  have hp : 0 < ‖v x‖ ^ 2 + (d : ℝ) * eps :=
    add_pos_of_nonneg_of_pos (sq_nonneg _) (mul_pos (Nat.cast_pos.mpr hd) heps)
  have hs := analyticAt_sqrt_pos
    ((analyticAt_normSq hv).fun_add analyticAt_const) hp
  exact (analyticAt_const.fun_div hs (Real.sqrt_pos.mpr hp).ne').fun_mul (hv i)

/-- R1: clipped L2 normalization is analytic away from its clipping sphere.
The premise allows norm zero; it does not require unit-length values. -/
theorem analyticAt_normL2 {eps : ℝ} (heps : 0 < eps)
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x)
    (hboundary : ‖v x‖ ≠ eps) (i : Fin d) :
    AnalyticAt ℝ (fun w => normL2 eps (v w) i) x := by
  have hc := continuousAt_norm_of_analytic hv
  rcases lt_or_gt_of_ne hboundary with hlo | hhi
  · have he := hc.eventually_lt continuousAt_const hlo
    apply (show AnalyticAt ℝ (fun w => v w i / eps) x from (hv i).div_const).congr
    filter_upwards [he] with w hw
    simp only [normL2, PiLp.smul_apply, smul_eq_mul, max_eq_right hw.le]
    ring
  · have hp : 0 < ‖v x‖ := heps.trans hhi
    have hn : AnalyticAt ℝ (fun w => ‖v w‖) x := by
      simpa only [Real.sqrt_sq (norm_nonneg _)] using
        analyticAt_sqrt_pos (analyticAt_normSq hv) (sq_pos_of_pos hp)
    have he := continuousAt_const.eventually_lt hc hhi
    apply ((hv i).fun_div hn hp.ne').congr
    filter_upwards [he] with w hw
    simp only [normL2, PiLp.smul_apply, smul_eq_mul, max_eq_left hw.le]
    ring

/-- R1: arbitrary fixed rotary phases preserve local analyticity. -/
theorem analyticAt_rotateHead (angle : Fin (d / 2) → ℝ)
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x) (i : Fin d) :
    AnalyticAt ℝ (fun w => rotateHead d angle (v w) i) x := by
  unfold rotateHead
  rcases h : ropeSplit d i with (k | k) | j
  · simpa only [h, rotate, splitPairs, ↓reduceIte, Fin.isValue, one_ne_zero] using
      ((hv (ropeFst d k)).fun_mul analyticAt_const).fun_sub
        ((hv (ropeSnd d k)).fun_mul analyticAt_const)
  · simpa only [h, rotate, splitPairs, ↓reduceIte, Fin.isValue, one_ne_zero] using
      ((hv (ropeFst d k)).fun_mul analyticAt_const).fun_add
        ((hv (ropeSnd d k)).fun_mul analyticAt_const)
  · simpa only [WithLp.ofLp_toLp, h] using hv ((ropeSplit d).symm (.inr j))

example := analyticAt_normSq (v := fun t : ℝ => WithLp.toLp 2 ![t, t + 1])
  (x := 0) (by intro i; fin_cases i <;> dsimp <;> fun_prop)
example := continuousAt_norm_of_analytic
  (v := fun t : ℝ => WithLp.toLp 2 ![t, t]) (x := 0)
  (by intro i; fin_cases i <;> dsimp <;> fun_prop)
example := analyticAt_rmsNormEps (d := 2) (by decide) (eps := 1 / 8388608)
  (by norm_num) (v := fun t : ℝ => WithLp.toLp 2 ![t, t + 1]) (x := 0)
  (by intro i; fin_cases i <;> dsimp <;> fun_prop) 0
example := analyticAt_normL2 (eps := 1) (by norm_num)
  (v := fun t : ℝ => WithLp.toLp 2 ![t, t]) (x := 0)
  (by intro i; fin_cases i <;> dsimp <;> fun_prop) (by norm_num [EuclideanSpace.norm_eq, Fin.sum_univ_two]) 0
example := analyticAt_normL2 (eps := 1) (by norm_num)
  (v := fun t : ℝ => WithLp.toLp 2 ![t, 0]) (x := 2)
  (by intro i; fin_cases i <;> dsimp <;> fun_prop)
  (by norm_num [EuclideanSpace.norm_eq, Fin.sum_univ_two]) 0
example := analyticAt_rotateHead (fun _ : Fin 1 => (1 : ℝ) / 7)
  (v := fun t : ℝ => WithLp.toLp 2 ![t, t + 1]) (x := 0)
  (by intro i; fin_cases i <;> dsimp <;> fun_prop) 1

end Transformer.SequenceModel.Attention
