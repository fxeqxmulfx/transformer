module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SmoothLoss
public import Transformer.Grokking.FlatSpectrum.Training.SignUnique

/-!
# Logits that are quadratic in the product: the loss and its gradient

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE13, the bullet
"Logits that are not linear in the product", and item XE14.

A `1 × 1` product `x` is read through the logits `±(Re x)²` of two classes, which are
quadratic in the product and not linear. The cross-entropy of the class of `+` is

    L(X) = log (1 + e^{−2 (Re x)²})

(`sqLogitLoss`, `sqLogitLoss_eq_crossEntropy`). Its gradient is the real `1 × 1` matrix
`−4 r w(r)`, `r = Re x`, with the weight `w(r) = e^{−2r²} / (1 + e^{−2r²})` of the other class
(`sqLogitGrad`, `hasDerivAt_sqLogitLoss`). The loss is nonnegative, real-analytic in the
entries and has a continuous gradient, so small steps converge to a stationary point
(`exists_pos_tendsto_decayRun_sqLogit`, by `exists_pos_tendsto_decayRun_of_analytic`). It is
not convex: `L(0) = log 2` is its largest value and the gradient vanishes at `0`, so the
first-order inequality of WD1 fails at `X = 0` (`not_first_order_sqLogit`).

The two bounds used by `SquaredLogitRun`: `‖G X‖ ≤ 2 |Re x|` and `⟨G X, X⟩ ≥ −2 (Re x)²`
(`frobeniusSq_sqLogitGrad_le`, `neg_le_pairing_sqLogitGrad`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open Landscape.Neural (crossEntropy)

/-- The weight of the other class under the logits `±r²`: `e^{−2r²} / (1 + e^{−2r²})`. -/
noncomputable def sqWeight (r : ℝ) : ℝ := Real.exp (-(2 * r ^ 2)) / (1 + Real.exp (-(2 * r ^ 2)))

lemma sqWeight_nonneg (r : ℝ) : 0 ≤ sqWeight r := by
  unfold sqWeight
  positivity

lemma sqWeight_le_half (r : ℝ) : sqWeight r ≤ 1 / 2 := by
  have h1 : Real.exp (-(2 * r ^ 2)) ≤ 1 := Real.exp_le_one_iff.2 (by nlinarith [sq_nonneg r])
  have hpos := Real.exp_pos (-(2 * r ^ 2))
  rw [sqWeight, div_le_iff₀ (by positivity)]
  linarith

/-- The cross-entropy of the logits `±(Re x)²` of a `1 × 1` product, for the class of `+`. -/
noncomputable def sqLogitLoss (X : Matrix (Fin 1) (Fin 1) ℂ) : ℝ :=
  Real.log (1 + Real.exp (-(2 * (X 0 0).re ^ 2)))

/-- Its gradient: the real `1 × 1` matrix `−4 r w(r)` with `r = Re x`. -/
noncomputable def sqLogitGrad (X : Matrix (Fin 1) (Fin 1) ℂ) : Matrix (Fin 1) (Fin 1) ℂ :=
  (-(4 * (X 0 0).re * sqWeight (X 0 0).re)) • 1

/-- **The loss is the cross-entropy of two classes with the logits `±(Re x)²`.** -/
theorem sqLogitLoss_eq_crossEntropy (X : Matrix (Fin 1) (Fin 1) ℂ) :
    sqLogitLoss X = crossEntropy (fun c : Bool => (signCode c).re * (X 0 0).re ^ 2) true := by
  rw [crossEntropy_signCode, sqLogitLoss]
  simp [signCode]

lemma sqLogitLoss_nonneg (X : Matrix (Fin 1) (Fin 1) ℂ) : 0 ≤ sqLogitLoss X :=
  Real.log_nonneg (by linarith [Real.exp_pos (-(2 * (X 0 0).re ^ 2))])

lemma sqLogitLoss_zero : sqLogitLoss 0 = Real.log 2 := by
  norm_num [sqLogitLoss]

lemma sqLogitGrad_zero : sqLogitGrad 0 = 0 := by
  simp [sqLogitGrad]

/-- `⟨G X, H⟩ = −4 r w(r) Re h`. -/
lemma pairing_sqLogitGrad (X H : Matrix (Fin 1) (Fin 1) ℂ) :
    pairing (sqLogitGrad X) H = -(4 * (X 0 0).re * sqWeight (X 0 0).re) * (H 0 0).re := by
  rw [sqLogitGrad, pairing_smul_left]
  simp [pairing, Matrix.trace]

/-- **The field is the gradient of the loss along lines.** -/
theorem hasDerivAt_sqLogitLoss (X H : Matrix (Fin 1) (Fin 1) ℂ) :
    HasDerivAt (fun t : ℝ => sqLogitLoss (X + t • H)) (pairing (sqLogitGrad X) H) 0 := by
  have hre : ∀ t : ℝ, ((X + t • H) 0 0).re = (X 0 0).re + t * (H 0 0).re := fun t => by
    simp
  have hlin : HasDerivAt (fun t : ℝ => (X 0 0).re + t * (H 0 0).re) (H 0 0).re 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (H 0 0).re).const_add (X 0 0).re
  have hexp := ((((hlin.fun_pow 2).const_mul 2).fun_neg).exp).const_add 1
  have hlog := hexp.log (by positivity)
  simp only [sqLogitLoss, hre]
  convert hlog using 1
  rw [pairing_sqLogitGrad, sqWeight]
  simp only [zero_mul, add_zero]
  have hpos : 0 < 1 + Real.exp (-(2 * (X 0 0).re ^ 2)) := by positivity
  field_simp
  ring

lemma continuous_sqLogitGrad : Continuous sqLogitGrad := by
  have hre : Continuous fun X : Matrix (Fin 1) (Fin 1) ℂ => (X 0 0).re :=
    Complex.continuous_re.comp (continuous_id.matrix_elem 0 0)
  have hw : Continuous fun X : Matrix (Fin 1) (Fin 1) ℂ => sqWeight (X 0 0).re := by
    unfold sqWeight
    exact (Real.continuous_exp.comp ((hre.pow 2).const_mul 2).neg).div
      (continuous_const.add (Real.continuous_exp.comp ((hre.pow 2).const_mul 2).neg))
      fun X => by positivity
  exact (((continuous_const.mul hre).mul hw).neg).smul continuous_const

/-- The loss is real-analytic in the entries, in the form of WD8 and WD9. -/
theorem analyticAt_sqLogitLoss (y₀ : EuclideanSpace ℂ (Fin 1 × Fin 1)) :
    AnalyticAt ℝ (fun y : EuclideanSpace ℂ (Fin 1 × Fin 1) =>
      sqLogitLoss (of fun i j => y (i, j) : Matrix (Fin 1) (Fin 1) ℂ)) y₀ := by
  have hre : AnalyticAt ℝ (fun y : EuclideanSpace ℂ (Fin 1 × Fin 1) => (y (0, 0)).re) y₀ :=
    (Complex.reCLM.analyticAt _).comp
      (((EuclideanSpace.proj ((0 : Fin 1), (0 : Fin 1))).restrictScalars ℝ).analyticAt y₀)
  have h1 : AnalyticAt ℝ (fun y : EuclideanSpace ℂ (Fin 1 × Fin 1) =>
      1 + Real.exp (-(2 * (y (0, 0)).re ^ 2))) y₀ :=
    analyticAt_const.fun_add ((analyticAt_const.fun_mul (hre.fun_pow 2)).fun_neg).rexp'
  have hpos : (0 : ℝ) < 1 + Real.exp (-(2 * (y₀ (0, 0)).re ^ 2)) := by positivity
  exact AnalyticAt.comp (g := Real.log) (analyticAt_log hpos) h1

/-- **Small steps converge for the quadratic logits**, to a stationary point: the first
statement of WD9 for a loss that is not convex. -/
theorem exists_pos_tendsto_decayRun_sqLogit {k : Type*} [Fintype k] {lam : ℝ} (hlam : 0 < lam)
    (c : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ (U₀ V₀ : Matrix (Fin 1) k ℂ),
      decayedLoss sqLogitLoss lam U₀ V₀ ≤ c → ∃ (U V : Matrix (Fin 1) k ℂ),
        Tendsto (decayRun sqLogitGrad η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
          lift lam (sqLogitGrad (U * Vᴴ)) * stack U V = 0 :=
  exists_pos_tendsto_decayRun_of_analytic hasDerivAt_sqLogitLoss continuous_sqLogitGrad
    analyticAt_sqLogitLoss hlam sqLogitLoss_nonneg c

/-- A product with a nonzero real part has less loss than `0`. -/
lemma sqLogitLoss_lt_zero {X : Matrix (Fin 1) (Fin 1) ℂ} (h : (X 0 0).re ≠ 0) :
    sqLogitLoss X < sqLogitLoss 0 := by
  rw [sqLogitLoss_zero, sqLogitLoss]
  have h1 : Real.exp (-(2 * (X 0 0).re ^ 2)) < 1 :=
    Real.exp_lt_one_iff.2 (by nlinarith [sq_pos_of_ne_zero h])
  exact Real.log_lt_log (by positivity) (by linarith)

/-- **The loss is not convex**: the first-order inequality of WD1 fails at `X = 0`. -/
theorem not_first_order_sqLogit :
    ¬ ∀ X Y, sqLogitLoss X + pairing (sqLogitGrad X) (Y - X) ≤ sqLogitLoss Y := fun h => by
  have h1 := h 0 1
  rw [sqLogitGrad_zero, pairing_zero_left, add_zero] at h1
  exact absurd (sqLogitLoss_lt_zero (X := 1) (by simp)) (not_lt.2 h1)

/-- `‖G X‖² ≤ (2 Re x)²`. -/
lemma frobeniusSq_sqLogitGrad_le (X : Matrix (Fin 1) (Fin 1) ℂ) :
    frobeniusSq (sqLogitGrad X) ≤ (2 * (X 0 0).re) ^ 2 := by
  have h0 := sqWeight_nonneg (X 0 0).re
  have h1 := sqWeight_le_half (X 0 0).re
  have hw : sqWeight (X 0 0).re ^ 2 ≤ (1 / 2) ^ 2 := pow_le_pow_left₀ h0 h1 2
  rw [sqLogitGrad, frobeniusSq_smul]
  simp only [frobeniusSq, Finset.univ_unique, Fin.default_eq_zero, Finset.sum_singleton,
    Matrix.one_apply_eq, norm_one, one_pow, mul_one]
  nlinarith [sq_nonneg (X 0 0).re, mul_le_mul_of_nonneg_left hw (sq_nonneg (X 0 0).re)]

/-- `⟨G X, X⟩ ≥ −2 (Re x)²`. -/
lemma neg_le_pairing_sqLogitGrad (X : Matrix (Fin 1) (Fin 1) ℂ) :
    -(2 * (X 0 0).re ^ 2) ≤ pairing (sqLogitGrad X) X := by
  rw [pairing_sqLogitGrad]
  nlinarith [sqWeight_le_half (X 0 0).re, sqWeight_nonneg (X 0 0).re, sq_nonneg (X 0 0).re,
    mul_le_mul_of_nonneg_left (sqWeight_le_half (X 0 0).re) (sq_nonneg (X 0 0).re)]

/-! The witnesses: the loss at the product `1`, and small steps at the weight decay `1` below
the level `1`. -/

example : sqLogitLoss (1 : Matrix (Fin 1) (Fin 1) ℂ) < sqLogitLoss 0 :=
  sqLogitLoss_lt_zero (by simp)

example := exists_pos_tendsto_decayRun_sqLogit (k := Fin 2) one_pos 1

end Transformer.Grokking.FlatSpectrum.Training.Scope
