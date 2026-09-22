/-
# Homogenized Transformers — the second moments of linear forms

If the coordinates `X_p` are uncorrelated with common variance `σ²` and
independent of `Y`, and the coefficients `c_p(Y)`, `c'_p(Y)` are bounded, then

  `E[(Σ_p X_p c_p(Y)) (Σ_q X_q c'_q(Y))] = σ² E Σ_p c_p(Y) c'_p(Y)`

(`integral_sum_mul_mul_sum_mul`): expanding the product, independence splits
each term into `E[c_p c'_q] E[X_p X_q]`, and only the diagonal survives.  At
`c' = c` this is `E(Σ_p X_p c_p(Y))² = σ² E Σ_p c_p(Y)²`
(`integral_sum_mul_sq`).  These are the computations behind consequence (i) of
assumption (G) and behind `lem:lemma_app`, made in
`Homogenized.GaussianMoments` and `Homogenized.GaussianCrossMoments` for a
Gaussian value matrix.

Source: arXiv:2604.01978v1, §2.3.3, item (i), and the proof of `lem:lemma_app`.
-/

import Mathlib.Probability.Independence.Integration

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

variable {Ω Z ι : Type*} [MeasurableSpace Ω] [MeasurableSpace Z] {P : Measure Ω}

/-- A product of coordinates of two measurable vectors is measurable. -/
theorem measurable_mul_apply {c c' : Z → ι → ℝ} (hc : Measurable c) (hc' : Measurable c')
    (p q : ι) : Measurable fun z => c z p * c' z q :=
  ((measurable_pi_apply p).comp hc).mul ((measurable_pi_apply q).comp hc')

/-- One term of the expanded product: a bounded coefficient times an integrable
product of coordinates. -/
theorem integrable_mul_mul_apply [IsFiniteMeasure P] {X : Ω → ι → ℝ} {Y : Ω → Z}
    (hY : Measurable Y) (hint : ∀ p q, Integrable (fun ω => X ω p * X ω q) P)
    {c c' : Z → ι → ℝ} (hc : Measurable c) (hc' : Measurable c') {C C' : ℝ}
    (hC : ∀ ω p, |c (Y ω) p| ≤ C) (hC' : ∀ ω p, |c' (Y ω) p| ≤ C') (p q : ι) :
    Integrable (fun ω => (c (Y ω) p * c' (Y ω) q) * (X ω p * X ω q)) P := by
  refine (hint p q).bdd_mul (c := C * C')
    ((measurable_mul_apply hc hc' p q).comp hY).aestronglyMeasurable
    (Filter.Eventually.of_forall fun ω => ?_)
  rw [Real.norm_eq_abs, abs_mul]
  exact mul_le_mul (hC ω p) (hC' ω q) (abs_nonneg _) ((abs_nonneg _).trans (hC ω p))

variable [Fintype ι]

/-- The product of two linear forms in the same coordinates, expanded. -/
theorem sum_mul_mul_sum_mul_eq (X c c' : ι → ℝ) :
    (∑ p, X p * c p) * (∑ q, X q * c' q) = ∑ p, ∑ q, (c p * c' q) * (X p * X q) := by
  rw [Finset.sum_mul_sum]
  exact Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => by ring

/-- The product of two linear forms in square-integrable coordinates, with
bounded coefficients, is integrable. -/
theorem integrable_sum_mul_mul_sum_mul [IsFiniteMeasure P] {X : Ω → ι → ℝ} {Y : Ω → Z}
    (hY : Measurable Y) (hint : ∀ p q, Integrable (fun ω => X ω p * X ω q) P)
    {c c' : Z → ι → ℝ} (hc : Measurable c) (hc' : Measurable c') {C C' : ℝ}
    (hC : ∀ ω p, |c (Y ω) p| ≤ C) (hC' : ∀ ω p, |c' (Y ω) p| ≤ C') :
    Integrable (fun ω => (∑ p, X ω p * c (Y ω) p) * (∑ q, X ω q * c' (Y ω) q)) P := by
  simp_rw [sum_mul_mul_sum_mul_eq]
  exact integrable_finsetSum _ fun p _ => integrable_finsetSum _ fun q _ =>
    integrable_mul_mul_apply hY hint hc hc' hC hC' p q

/-- The square of a linear form in square-integrable coordinates, with bounded
coefficients, is integrable. -/
theorem integrable_sum_mul_sq [IsFiniteMeasure P] {X : Ω → ι → ℝ} {Y : Ω → Z}
    (hY : Measurable Y) (hint : ∀ p q, Integrable (fun ω => X ω p * X ω q) P)
    {c : Z → ι → ℝ} (hc : Measurable c) {C : ℝ} (hC : ∀ ω p, |c (Y ω) p| ≤ C) :
    Integrable (fun ω => (∑ p, X ω p * c (Y ω) p) ^ 2) P := by
  simp_rw [sq]
  exact integrable_sum_mul_mul_sum_mul hY hint hc hc hC hC

/-- **Two linear forms in uncorrelated coordinates.**  If the coordinates `X_p`
are uncorrelated with variance `σ²` and independent of `Y`, and the
coefficients `c_p(Y)`, `c'_p(Y)` are bounded, then

  `E[(Σ_p X_p c_p(Y)) (Σ_q X_q c'_q(Y))] = σ² E Σ_p c_p(Y) c'_p(Y)`.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app` (the computation behind
`E_V[V y y'ᵀ Vᵀ] = σ_V² ⟨y, y'⟩ I_d`). -/
theorem integral_sum_mul_mul_sum_mul [DecidableEq ι] [IsFiniteMeasure P] {X : Ω → ι → ℝ}
    {Y : Ω → Z} (hX : Measurable X) (hY : Measurable Y) (hXY : IndepFun X Y P) {σ2 : ℝ}
    (hmom : ∀ p q, ∫ ω, X ω p * X ω q ∂P = if p = q then σ2 else 0)
    (hint : ∀ p q, Integrable (fun ω => X ω p * X ω q) P)
    {c c' : Z → ι → ℝ} (hc : Measurable c) (hc' : Measurable c') {C C' : ℝ}
    (hC : ∀ ω p, |c (Y ω) p| ≤ C) (hC' : ∀ ω p, |c' (Y ω) p| ≤ C') :
    ∫ ω, (∑ p, X ω p * c (Y ω) p) * (∑ q, X ω q * c' (Y ω) q) ∂P =
      σ2 * ∫ ω, ∑ p, c (Y ω) p * c' (Y ω) p ∂P := by
  have hterm := integrable_mul_mul_apply hY hint hc hc' hC hC'
  have hsplit : ∀ p q, ∫ ω, (c (Y ω) p * c' (Y ω) q) * (X ω p * X ω q) ∂P =
      (∫ ω, c (Y ω) p * c' (Y ω) q ∂P) * ∫ ω, X ω p * X ω q ∂P := fun p q => by
    have hXm : Measurable fun x : ι → ℝ => x p * x q :=
      (measurable_pi_apply p).mul (measurable_pi_apply q)
    exact (hXY.symm.comp (measurable_mul_apply hc hc' p q) hXm).integral_mul_eq_mul_integral
      ((measurable_mul_apply hc hc' p q).comp hY).aestronglyMeasurable
      (hXm.comp hX).aestronglyMeasurable
  have hdiag : ∀ p, Integrable (fun ω => c (Y ω) p * c' (Y ω) p) P := fun p => by
    refine Integrable.of_bound ((measurable_mul_apply hc hc' p p).comp hY).aestronglyMeasurable
      (C * C') (Filter.Eventually.of_forall fun ω => ?_)
    rw [Real.norm_eq_abs, abs_mul]
    exact mul_le_mul (hC ω p) (hC' ω p) (abs_nonneg _) ((abs_nonneg _).trans (hC ω p))
  have hrow : ∀ p, ∫ ω, ∑ q, (c (Y ω) p * c' (Y ω) q) * (X ω p * X ω q) ∂P =
      σ2 * ∫ ω, c (Y ω) p * c' (Y ω) p ∂P := fun p => by
    rw [integral_finsetSum _ fun q _ => hterm p q]
    simp only [hsplit, hmom, mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    ring
  simp_rw [sum_mul_mul_sum_mul_eq]
  rw [integral_finsetSum _ fun p _ => integrable_finsetSum _ fun q _ => hterm p q,
    integral_finsetSum _ fun p _ => hdiag p, Finset.mul_sum]
  exact Finset.sum_congr rfl fun p _ => hrow p

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
  simp_rw [sq]
  exact integral_sum_mul_mul_sum_mul hX hY hXY hmom hint hc hc hC hC

/-- The hypotheses of the lemmas of this file are satisfiable: zero
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
