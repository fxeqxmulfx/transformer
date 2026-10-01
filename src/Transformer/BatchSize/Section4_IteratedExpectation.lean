/-
# Markov expectations equal the actual finite-step sampler

arXiv:2506.12543v1, Section 4.3, Theorem 1.
Splitting the independent Gaussian innovations proves the expectation
recursion; iteration of the Markov operator is not a replacement definition
of the optimizer law.
-/

import Transformer.BatchSize.Section4_ExpectationOperator

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Split the first independent Gaussian innovation from the remaining
steps in the actual optimizer expectation, Section 4.3, Theorem 1. -/
theorem discreteEndpoint_expectation_succ {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ)
    (φ : BoundedObservable (EucSpace d)) (n : ℕ) (x : EucSpace d) :
    (∫ z, φ.val (discreteEndpoint method η B f σ x (n + 1) z)
      ∂innovationLaw d (n + 1)) =
    ∫ z, (∫ w, φ.val (discreteEndpoint method η B f σ
      (stochasticStep method η B f σ x z) n w) ∂innovationLaw d n)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1) := by
  let ν := diagonalNoiseLaw 1 (fun _ : Fin d => 0) (fun _ => 1)
  let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => Fin d → ℝ) 0
  have hp : MeasurePreserving e (innovationLaw d (n + 1)) (ν.prod (innovationLaw d n)) := by
    simpa [e, ν, innovationLaw] using
      measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => ν) 0
  let G := fun p : (Fin d → ℝ) × (Fin n → Fin d → ℝ) =>
    φ.val (discreteEndpoint method η B f σ x (n + 1) (e.symm p))
  have hg : Measurable G :=
    φ.property.1.comp ((discreteEndpoint_measurable method η B f σ x hf hσ (n + 1)).comp
      e.symm.measurable)
  have hi : Integrable G (ν.prod (innovationLaw d n)) := by
    obtain ⟨C, hC⟩ := φ.property.2
    apply (integrable_const C).mono' hg.aestronglyMeasurable
    exact ae_of_all _ (fun p => by
      simpa [G, Real.norm_eq_abs] using hC
        (discreteEndpoint method η B f σ x (n + 1) (e.symm p)))
  have hr (p : (Fin d → ℝ) × (Fin n → Fin d → ℝ)) :
      G p = φ.val (discreteEndpoint method η B f σ
        (stochasticStep method η B f σ x p.1) n p.2) := by
    have he : e.symm p = Fin.cons p.1 p.2 := by
      ext j
      simp [e, MeasurableEquiv.piFinSuccAbove_symm_apply]
    dsimp [G]
    rw [he]
    simp [discreteEndpoint, List.ofFn_succ]
  calc
    (∫ z, φ.val (discreteEndpoint method η B f σ x (n + 1) z)
        ∂innovationLaw d (n + 1)) = ∫ z, G (e z) ∂innovationLaw d (n + 1) := by
      simp [G]
    _ = ∫ p, G p ∂ν.prod (innovationLaw d n) := hp.integral_comp e.measurableEmbedding G
    _ = ∫ z, (∫ w, G (z, w) ∂innovationLaw d n) ∂ν := integral_prod G hi
    _ = _ := by simp_rw [hr]; rfl

/-- Joint nonvacuity of the expectation recursion's regularity
hypotheses, Section 4.3. -/
example : ContDiff ℝ 1 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) :=
  ⟨contDiff_const, continuous_const⟩

/-- Iterating the actual Markov expectation operator computes the
expectation of the independent-noise fold sampler, Section 4.3, Theorem 1. -/
theorem iteratedExpectation_eq_sampler {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ)
    (φ : BoundedObservable (EucSpace d)) (n : ℕ) :
    ∀ x, (((discreteExpectationOperator method η B f σ hf hσ)^[n]) φ).val x =
      ∫ z, φ.val (discreteEndpoint method η B f σ x n z) ∂innovationLaw d n := by
  induction n with
  | zero => intro x; simp [discreteEndpoint]
  | succ n ih =>
    intro x
    rw [Function.iterate_succ_apply']
    change (∫ z, (((discreteExpectationOperator method η B f σ hf hσ)^[n]) φ).val
      (stochasticStep method η B f σ x z)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) = _
    simp_rw [ih]
    exact (discreteEndpoint_expectation_succ method η B f σ hf hσ φ n x).symm

/-- Joint nonvacuity of the iterated sampler's hypotheses, Section 4.3. -/
example : ContDiff ℝ 1 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) :=
  ⟨contDiff_const, continuous_const⟩

/-- The iterated expectation equals integration under the actual
optimizer law used in Theorem 1, Section 4.3. -/
theorem iteratedExpectation_eq_discreteLaw {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ)
    (φ : BoundedObservable (EucSpace d)) (n : ℕ) (x : EucSpace d) :
    (((discreteExpectationOperator method η B f σ hf hσ)^[n]) φ).val x =
      ∫ y, φ.val y ∂discreteLaw method η B f σ x n := by
  rw [discreteLaw, integral_map
    (discreteEndpoint_measurable method η B f σ x hf hσ n).aemeasurable
    φ.property.1.aestronglyMeasurable]
  exact iteratedExpectation_eq_sampler method η B f σ hf hσ φ n x

/-- Joint nonvacuity of the law comparison's hypotheses, Section 4.3. -/
example : ContDiff ℝ 1 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) :=
  ⟨contDiff_const, continuous_const⟩

end Transformer.BatchSize
