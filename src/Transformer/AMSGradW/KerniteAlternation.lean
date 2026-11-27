module

public import Transformer.AMSGradW.KerniteCommonGradient
public import Transformer.AMSGradW.ShiftedQuadratic
public import Transformer.OptimizerBenchmark.GuardFallback

/-! Two nonnegative minibatch losses with the same exact minimizer show
that a positive rate floor and decoupled decay need not give a common
clipped gradient, even when both batches are fitted by a single weight. -/

@[expose] public section

namespace Transformer.AMSGradW.Kernite

open Filter
open scoped Topology

noncomputable def alternatingWeight (t : ℕ) : ℝ := if t % 2 = 0 then 1 else 2

/-- Two alternating squared errors about the same parameter value one.
They are the losses of the scalar linear predictor on a unit feature
and a feature of size square root of two, with targets equal to the features. -/
noncomputable def alternatingQuadratic (t : ℕ) (w : Space (Fin 1)) : ℝ :=
  alternatingWeight t * shiftedEnergy w

noncomputable def alternatingQuadraticGradient (t : ℕ) (w : Fin 1 → ℝ) : Fin 1 → ℝ :=
  WithLp.ofLp (gradient (alternatingQuadratic t) (WithLp.toLp 2 w))

private theorem alternatingQuadraticGradient_apply (t : ℕ) (w : Fin 1 → ℝ) (i : Fin 1) :
    alternatingQuadraticGradient t w i = alternatingWeight t * (w i - 1) := by
  let x : Space (Fin 1) := WithLp.toLp 2 w
  let c : Space (Fin 1) := WithLp.toLp 2 (fun _ => 1)
  have hg : HasGradientAt (alternatingQuadratic t) (alternatingWeight t • (x - c)) x := by
    rw [hasGradientAt_iff_hasFDerivAt, map_smul]
    exact (shiftedEnergy_hasGradientAt x).hasFDerivAt.const_smul (alternatingWeight t)
  change gradient (alternatingQuadratic t) x i = alternatingWeight t * (w i - 1)
  rw [hg.gradient]
  rfl

private theorem continuous_clipped_scale (b : ℝ) :
    Continuous (fun w : Fin 1 → ℝ => clip (fun i => b * (w i - 1)) 0) := by
  have hi (i : Fin 1) : Continuous (fun w : Fin 1 → ℝ => w i) := continuous_apply i
  have h : Continuous (fun w : Fin 1 → ℝ => fun i => b * (w i - 1)) :=
    continuous_pi fun i => ((hi i).sub continuous_const).const_mul b
  have hD := (((PiLp.continuous_toLp 2 _).comp h).norm).max (continuous_const (y := (1 : ℝ)))
  exact ((continuous_apply 0).comp h).div hD (fun _ => (zero_lt_one.trans_le (le_max_right _ _)).ne')

/-- The two convex squared errors share an exact solution. Nevertheless,
for every initialization their alternating gradients preclude convergence
of the weights at a positive learning-rate floor. -/
theorem alternatingQuadratic_not_tendsto (rate : ℕ → ℝ) (initial : Fin 1 → ℝ)
    {η : ℝ} (hη : 0 < η) (hηlim : Tendsto rate atTop (𝓝 η)) :
    ¬ ∃ w, Tendsto (fun t => (run rate alternatingQuadraticGradient initial t).weights) atTop (𝓝 w) := by
  rintro ⟨w, hw⟩
  obtain ⟨V, _, hG⟩ := run_clippedGrad_tendsto_of_weights_tendsto rate alternatingQuadraticGradient initial w
    hη hηlim hw
  let D := Real.sqrt (V 0) + 1 / 100000000
  have hD : 0 < D := by dsimp only [D]; positivity
  have hagree (b : ℝ) (u : ℕ → ℕ) (hu : StrictMono u) (hb : ∀ t, alternatingWeight (u t) = b) :
      clip (fun i => b * (w i - 1)) 0 = -(1 / 10 : ℝ) * D * w 0 := by
    have hg := (tendsto_pi_nhds.mp hG 0).comp hu.tendsto_atTop
    apply tendsto_nhds_unique ((continuous_clipped_scale b).tendsto w |>.comp (hw.comp hu.tendsto_atTop))
    apply hg.congr
    intro t
    simp only [Function.comp_def]
    congr 1
    funext i
    rw [alternatingQuadraticGradient_apply, hb t]
  have hfirst := hagree 1 (fun n => 2 * n) (fun a b h => by change 2 * a < 2 * b; omega)
    (fun n => by simp [alternatingWeight])
  have hsecond := hagree 2 (fun n => 2 * n + 1) (fun a b h => by change 2 * a + 1 < 2 * b + 1; omega)
    (fun n => by simp [alternatingWeight])
  have hdecay := mul_neg_of_neg_of_pos (by norm_num : -(1 / 10 : ℝ) < 0) hD
  have hlo : 0 < w 0 := by
    by_contra! hz
    have hneg : clip (fun i => 1 * (w i - 1)) 0 < 0 :=
      div_neg_of_neg_of_pos (by change 1 * (w 0 - 1) < 0; linarith)
        (zero_lt_one.trans_le (le_max_right _ _))
    rw [hfirst] at hneg
    exact hneg.not_ge (mul_nonneg_of_nonpos_of_nonpos hdecay.le hz)
  have hhi : w 0 < 1 := by
    by_contra! hz
    have hnonneg : 0 ≤ clip (fun i => 1 * (w i - 1)) 0 :=
      div_nonneg (by change 0 ≤ 1 * (w 0 - 1); linarith) (zero_le_one.trans (le_max_right _ _))
    rw [hfirst] at hnonneg
    exact (mul_neg_of_neg_of_pos hdecay hlo).not_ge hnonneg
  have hnorm (g : Fin 1 → ℝ) : ‖(WithLp.toLp 2 g : Space (Fin 1))‖ = |g 0| := by
    rw [PiLp.norm_eq_of_L2, Fin.sum_univ_one, Real.sqrt_sq_eq_abs, Real.norm_eq_abs, abs_abs]
  let x := w 0 - 1
  have hx : x < 0 := sub_neg.mpr hhi
  have habs : |x| < 1 := abs_lt.mpr ⟨by dsimp [x]; linarith, by linarith⟩
  have heq : clip (fun i => 1 * (w i - 1)) 0 = x := by
    rw [clip, hnorm, one_mul, max_eq_right habs.le, div_one]
  have hlt : clip (fun i => 2 * (w i - 1)) 0 < x := by
    rw [clip, hnorm]
    change 2 * x / max |2 * x| 1 < x
    apply (div_lt_iff₀ (zero_lt_one.trans_le (le_max_right _ _))).2
    have hmax : max |2 * x| 1 < 2 := max_lt
      (by rw [abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]; linarith)
      (by norm_num)
    simpa only [mul_comm] using mul_lt_mul_of_neg_right hmax hx
  rw [← heq, hfirst, hsecond] at hlt
  exact (lt_irrefl _) hlt

/-- A shared interpolating weight does not guarantee a gradient limit
for the clipped, bias-corrected optimizer with positive decay. -/
theorem alternatingQuadratic_no_common_gradient (rate : ℕ → ℝ) (initial : Fin 1 → ℝ)
    {η : ℝ} (hrate : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (hη : 0 < η) (hηlim : Tendsto rate atTop (𝓝 η)) :
    ¬ ∃ G, Tendsto (fun t => clip (alternatingQuadraticGradient t
      (run rate alternatingQuadraticGradient initial t).weights)) atTop (𝓝 G) :=
  fun hG => alternatingQuadratic_not_tendsto rate initial hη hηlim
    ((run_weights_tendsto_iff_clippedGrad rate alternatingQuadraticGradient initial hrate hη hηlim).2 hG)

/-- Ordinary minibatch SGD converges geometrically on the same two losses.
Both SGD updates contract toward the shared interpolating solution. -/
theorem alternatingQuadratic_sgd_tendsto (η : ℝ) (initial : Space (Fin 1))
    (hη : 0 < η) (hη' : η < 1) :
    Tendsto (OptimizerBenchmark.batchSGD η alternatingQuadratic initial) atTop
      (𝓝 (WithLp.toLp 2 fun _ => 1)) := by
  let s := OptimizerBenchmark.batchSGD η alternatingQuadratic initial
  let q := max |1 - η| |1 - 2 * η|
  have hq0 : 0 ≤ q := (abs_nonneg _).trans (le_max_left _ _)
  have hq : q < 1 := max_lt (abs_lt.mpr ⟨by linarith, by linarith⟩)
    (abs_lt.mpr ⟨by linarith, by linarith⟩)
  have hstep (t) : s (t + 1) 0 - 1 = (1 - η * alternatingWeight t) * (s t 0 - 1) := by
    have hg : gradient (alternatingQuadratic t) (s t) 0 = alternatingWeight t * (s t 0 - 1) :=
      alternatingQuadraticGradient_apply t (WithLp.ofLp (s t)) 0
    change s t 0 - η * gradient (alternatingQuadratic t) (s t) 0 - 1 = _
    rw [hg]
    ring
  have hcontract (t) : |1 - η * alternatingWeight t| ≤ q := by
    unfold alternatingWeight
    split
    · simpa only [mul_one] using (le_max_left |1 - η| |1 - 2 * η|)
    · rw [mul_comm η]
      exact le_max_right _ _
  have herror (t) : |s t 0 - 1| ≤ q ^ t * |initial 0 - 1| := by
    induction t with
    | zero => simp [s, OptimizerBenchmark.batchSGD]
    | succ t ih =>
      rw [hstep, abs_mul, pow_succ']
      exact (mul_le_mul (hcontract t) ih (abs_nonneg _) hq0).trans_eq (mul_assoc _ _ _).symm
  have hzero : Tendsto (fun t => |s t 0 - 1|) atTop (𝓝 0) := by
    apply squeeze_zero (fun _ => abs_nonneg _) herror
    simpa only [zero_mul] using (tendsto_pow_atTop_nhds_zero_of_lt_one hq0 hq).mul_const |initial 0 - 1|
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  have heq (t) : ‖s t - WithLp.toLp 2 (fun _ => 1)‖ = |s t 0 - 1| := by
    rw [PiLp.norm_eq_of_L2, Fin.sum_univ_one, Real.sqrt_sq_eq_abs, Real.norm_eq_abs, abs_abs]
    rfl
  exact hzero.congr fun t => (heq t).symm

example (t : ℕ) : alternatingQuadratic t (WithLp.toLp 2 (fun _ => 1)) = 0 := by
  simp [alternatingQuadratic, shiftedEnergy, Optimization.energy]

example (initial : Fin 1 → ℝ) :=
  alternatingQuadratic_no_common_gradient (fun _ => 1 / 100000000) initial
    (fun _ => by norm_num) (by norm_num) tendsto_const_nhds

example (initial : Space (Fin 1)) :=
  alternatingQuadratic_sgd_tendsto (1 / 4) initial (by norm_num) (by norm_num)

end Transformer.AMSGradW.Kernite
