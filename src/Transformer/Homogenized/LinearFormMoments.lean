/-
# Homogenized Transformers — the second moment of a linear form

If the coordinates `X_p` are uncorrelated with common variance `σ²` and
independent of `Y`, and the coefficients `c_p(Y)` are bounded, then

  `E(Σ_p X_p c_p(Y))² = σ² E Σ_p c_p(Y)²`

(`integral_sum_mul_sq`): expanding the square, independence splits each term
into `E[c_p c_q] E[X_p X_q]`, and only the diagonal survives.  This is the
computation behind consequence (i) of assumption (G), made in
`Homogenized.GaussianMoments` for a Gaussian value matrix.

Source: arXiv:2604.01978v1, §2.3.3, item (i).
-/

import Mathlib.Probability.Independence.Integration

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

variable {Ω Z ι : Type*} [MeasurableSpace Ω] [MeasurableSpace Z] {P : Measure Ω}

/-- A product of two coordinates of a measurable vector is measurable. -/
theorem measurable_mul_apply {c : Z → ι → ℝ} (hc : Measurable c) (p q : ι) :
    Measurable fun z => c z p * c z q :=
  ((measurable_pi_apply p).comp hc).mul ((measurable_pi_apply q).comp hc)

/-- One term of the expanded square: a bounded coefficient times an integrable
product of coordinates. -/
theorem integrable_mul_mul_apply [IsFiniteMeasure P] {X : Ω → ι → ℝ} {Y : Ω → Z}
    (hY : Measurable Y) (hint : ∀ p q, Integrable (fun ω => X ω p * X ω q) P)
    {c : Z → ι → ℝ} (hc : Measurable c) {C : ℝ} (hC : ∀ ω p, |c (Y ω) p| ≤ C) (p q : ι) :
    Integrable (fun ω => (c (Y ω) p * c (Y ω) q) * (X ω p * X ω q)) P := by
  refine (hint p q).bdd_mul (c := C * C)
    ((measurable_mul_apply hc p q).comp hY).aestronglyMeasurable
    (Filter.Eventually.of_forall fun ω => ?_)
  rw [Real.norm_eq_abs, abs_mul]
  exact mul_le_mul (hC ω p) (hC ω q) (abs_nonneg _) ((abs_nonneg _).trans (hC ω p))

variable [Fintype ι]

/-- The square of a linear form, expanded. -/
theorem sum_mul_sq_eq (X c : ι → ℝ) :
    (∑ p, X p * c p) ^ 2 = ∑ p, ∑ q, (c p * c q) * (X p * X q) := by
  rw [sq, Finset.sum_mul_sum]
  exact Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => by ring

/-- The square of a linear form in square-integrable coordinates, with bounded
coefficients, is integrable. -/
theorem integrable_sum_mul_sq [IsFiniteMeasure P] {X : Ω → ι → ℝ} {Y : Ω → Z}
    (hY : Measurable Y) (hint : ∀ p q, Integrable (fun ω => X ω p * X ω q) P)
    {c : Z → ι → ℝ} (hc : Measurable c) {C : ℝ} (hC : ∀ ω p, |c (Y ω) p| ≤ C) :
    Integrable (fun ω => (∑ p, X ω p * c (Y ω) p) ^ 2) P := by
  simp_rw [sum_mul_sq_eq]
  exact integrable_finsetSum _ fun p _ => integrable_finsetSum _ fun q _ =>
    integrable_mul_mul_apply hY hint hc hC p q

/-- **A linear form in uncorrelated coordinates.**  If the coordinates `X_p`
are uncorrelated with variance `σ²` and independent of `Y`, and the
coefficients `c_p(Y)` are bounded, then

  `E(Σ_p X_p c_p(Y))² = σ² E Σ_p c_p(Y)²`.

Source: arXiv:2604.01978v1, §2.3.3, item (i) (the computation behind it). -/
theorem integral_sum_mul_sq [DecidableEq ι] [IsFiniteMeasure P] {X : Ω → ι → ℝ} {Y : Ω → Z}
    (hX : Measurable X) (hY : Measurable Y) (hXY : IndepFun X Y P) {σ2 : ℝ}
    (hmom : ∀ p q, ∫ ω, X ω p * X ω q ∂P = if p = q then σ2 else 0)
    (hint : ∀ p q, Integrable (fun ω => X ω p * X ω q) P)
    {c : Z → ι → ℝ} (hc : Measurable c) {C : ℝ} (hC : ∀ ω p, |c (Y ω) p| ≤ C) :
    ∫ ω, (∑ p, X ω p * c (Y ω) p) ^ 2 ∂P = σ2 * ∫ ω, ∑ p, c (Y ω) p ^ 2 ∂P := by
  have hterm := integrable_mul_mul_apply hY hint hc hC
  have hsplit : ∀ p q, ∫ ω, (c (Y ω) p * c (Y ω) q) * (X ω p * X ω q) ∂P =
      (∫ ω, c (Y ω) p * c (Y ω) q ∂P) * ∫ ω, X ω p * X ω q ∂P := fun p q => by
    have hXm : Measurable fun x : ι → ℝ => x p * x q :=
      (measurable_pi_apply p).mul (measurable_pi_apply q)
    exact (hXY.symm.comp (measurable_mul_apply hc p q) hXm).integral_mul_eq_mul_integral
      ((measurable_mul_apply hc p q).comp hY).aestronglyMeasurable
      (hXm.comp hX).aestronglyMeasurable
  have hsq : ∀ p, Integrable (fun ω => c (Y ω) p ^ 2) P := fun p => by
    refine Integrable.of_bound
      ((((measurable_pi_apply p).comp hc).comp hY).pow_const 2).aestronglyMeasurable
      (C * C) (Filter.Eventually.of_forall fun ω => ?_)
    rw [Real.norm_eq_abs, abs_pow, sq]
    exact mul_le_mul (hC ω p) (hC ω p) (abs_nonneg _) ((abs_nonneg _).trans (hC ω p))
  have hrow : ∀ p, ∫ ω, ∑ q, (c (Y ω) p * c (Y ω) q) * (X ω p * X ω q) ∂P =
      σ2 * ∫ ω, c (Y ω) p ^ 2 ∂P := fun p => by
    rw [integral_finsetSum _ fun q _ => hterm p q]
    simp only [hsplit, hmom, mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true, sq]
    ring
  simp_rw [sum_mul_sq_eq]
  rw [integral_finsetSum _ fun p _ => integrable_finsetSum _ fun q _ => hterm p q,
    integral_finsetSum _ fun p _ => hsq p, Finset.mul_sum]
  exact Finset.sum_congr rfl fun p _ => hrow p

/-- The hypotheses of `measurable_mul_apply`, `integrable_mul_mul_apply`,
`integrable_sum_mul_sq` and `integral_sum_mul_sq` are satisfiable: zero
coordinates, of variance `0`, and zero coefficients. -/
example :
    Measurable (fun _ : Unit => (0 : Unit → ℝ)) ∧ Measurable (fun _ : Unit => ()) ∧
      IndepFun (fun _ : Unit => (0 : Unit → ℝ)) (fun _ : Unit => ()) (Measure.dirac ()) ∧
      (∀ p q : Unit, ∫ _ω, (0 : ℝ) * 0 ∂(Measure.dirac ()) = if p = q then 0 else 0) ∧
      (∀ _p _q : Unit, Integrable (fun _ω : Unit => (0 : ℝ) * 0) (Measure.dirac ())) ∧
      Measurable (fun (_ : Unit) (_ : Unit) => (0 : ℝ)) ∧ ∀ _ω _p : Unit, |(0 : ℝ)| ≤ 0 :=
  ⟨measurable_const, measurable_const, indepFun_const_left _ _, fun _ _ => by simp,
    fun _ _ => integrable_const _, measurable_const, fun _ _ => by simp⟩

end Homogenized
end Transformer
