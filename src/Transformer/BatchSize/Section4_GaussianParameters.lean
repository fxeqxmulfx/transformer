/-
# Actual Gaussian states and generators with variable parameters

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The coordinate formulas establish joint continuity and measurability
before the frozen Gaussian identity is applied to random coefficients.
-/

import Transformer.BatchSize.Section4_FrozenGaussianGenerator

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Joint measurability of the actual affine Gaussian state,
Section 4.3 (2)--(3), before elapsed-time rescaling. -/
theorem gaussianAffineState_measurable_parameters {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (X : Ω → EucSpace d) (A z : Ω → Fin d → ℝ) (hX : Measurable X)
    (hA : ∀ k, Measurable (fun ω => A ω k)) (hz : ∀ k, Measurable (fun ω => z ω k)) :
    Measurable (fun ω => gaussianAffineState (X ω) (A ω) (z ω)) := by
  have hnoise : Measurable (fun ω => gaussianDiagonalMap (A ω) (z ω)) :=
    (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.measurable.comp
      (Measurable.of_eval fun k => (hA k).mul (hz k))
  exact hX.add hnoise

/-- Rescaling the actual affine innovation gives precisely the
frozen Gaussian transition state, Section 4.3 (2)--(3). -/
theorem gaussianAffineState_rescale {d : ℕ} (x b : EucSpace d) (A : Fin d → ℝ)
    (t : ℝ) (z : Fin d → ℝ) :
    gaussianAffineState (x + t • b) A (fun k => Real.sqrt t * z k) =
      gaussianVectorState x b A t z := by
  ext k
  simp only [gaussianAffineState, gaussianVectorState, PiLp.add_apply, PiLp.smul_apply,
    smul_eq_mul, gaussianDiagonalMap_apply]
  ring

/-- Joint continuity of the actual Gaussian state in its state,
drift, amplitudes, time and innovation, Section 4.3 (2)--(3). -/
theorem gaussianVectorState_continuous_parameters {Ω : Type*} [TopologicalSpace Ω] {d : ℕ}
    (X b : Ω → EucSpace d) (A z : Ω → Fin d → ℝ) (t : Ω → ℝ)
    (hX : Continuous X) (hb : Continuous b) (hA : ∀ k, Continuous (fun ω => A ω k))
    (hz : ∀ k, Continuous (fun ω => z ω k)) (ht : Continuous t) :
    Continuous (fun ω => gaussianVectorState (X ω) (b ω) (A ω) (t ω) (z ω)) := by
  have hnoise : Continuous (fun ω => gaussianDiagonalMap (A ω) (z ω)) :=
    (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp
      (continuous_pi fun k => (hA k).mul (hz k))
  exact (hX.add (ht.smul hb)).add ((Real.continuous_sqrt.comp ht).smul hnoise)

/-- Joint measurability of the actual Gaussian state with random
coefficients, Section 4.3 (2)--(3). -/
theorem gaussianVectorState_measurable_parameters {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (X b : Ω → EucSpace d) (A z : Ω → Fin d → ℝ) (t : Ω → ℝ)
    (hX : Measurable X) (hb : Measurable b) (hA : ∀ k, Measurable (fun ω => A ω k))
    (hz : ∀ k, Measurable (fun ω => z ω k)) (ht : Measurable t) :
    Measurable (fun ω => gaussianVectorState (X ω) (b ω) (A ω) (t ω) (z ω)) := by
  have hnoise : Measurable (fun ω => gaussianDiagonalMap (A ω) (z ω)) :=
    (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.measurable.comp
      (Measurable.of_eval fun k => (hA k).mul (hz k))
  exact (hX.add (ht.smul hb)).add ((Real.continuous_sqrt.measurable.comp ht).smul hnoise)

/-- Joint continuity of the actual frozen generator in its drift,
diagonal amplitudes and evaluation state, Section 4.3 (2)--(3). -/
theorem frozenGaussianGenerator_continuous_parameters {Ω : Type*} [TopologicalSpace Ω] {d : ℕ}
    (b : Ω → EucSpace d) (A : Ω → Fin d → ℝ) (Y : Ω → EucSpace d)
    (hb : Continuous b) (hA : ∀ k, Continuous (fun ω => A ω k)) (hY : Continuous Y)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Continuous (fun ω => frozenGaussianGenerator (b ω) (A ω) φ (Y ω)) := by
  exact ((hφ.1.continuous_fderiv (by norm_num) |>.comp hY).clm_apply hb).add
    ((continuous_finsetSum Finset.univ fun k _ => (hA k).pow 2 |>.mul
      ((ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => EucSpace d) ℝ
        (fun _ => EuclideanSpace.single k 1)).continuous.comp
          (hφ.1.continuous_iteratedFDeriv' (m := 2) |>.comp hY))).div_const 2)

/-- Joint measurability of the actual frozen generator with random
coefficients, Section 4.3 (2)--(3), for genuine bounded C2 tests. -/
theorem frozenGaussianGenerator_measurable_parameters {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (b : Ω → EucSpace d) (A : Ω → Fin d → ℝ) (Y : Ω → EucSpace d)
    (hb : Measurable b) (hA : ∀ k, Measurable (fun ω => A ω k)) (hY : Measurable Y)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Measurable (fun ω => frozenGaussianGenerator (b ω) (A ω) φ (Y ω)) := by
  have heval : Continuous (fun p : (EucSpace d →L[ℝ] ℝ) × EucSpace d => p.1 p.2) :=
    continuous_fst.clm_apply continuous_snd
  exact (heval.measurable.comp
    ((hφ.1.continuous_fderiv (by norm_num) |>.measurable.comp hY).prodMk hb)).add
    ((Finset.measurable_fun_sum Finset.univ fun k _ => (hA k).pow_const 2 |>.mul
      (((ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => EucSpace d) ℝ
        (fun _ => EuclideanSpace.single k 1)).continuous.comp
          (hφ.1.continuous_iteratedFDeriv' (m := 2))).measurable.comp hY)).div_const 2)


/-- Measurability of the actual Gaussian expectation with random
state, drift, amplitudes and time, Section 4.3 (2)--(3). -/
theorem gaussianVectorFlow_measurable_parameters {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (X b : Ω → EucSpace d) (A : Ω → Fin d → ℝ) (t : Ω → ℝ)
    (hX : Measurable X) (hb : Measurable b) (hA : ∀ k, Measurable (fun ω => A ω k))
    (ht : Measurable t) (φ : EucSpace d → ℝ) (hφ : Continuous φ) :
    Measurable (fun ω => gaussianVectorFlow φ (X ω) (b ω) (A ω) (t ω)) := by
  have hstate := gaussianVectorState_measurable_parameters
    (fun p : Ω × (Fin d → ℝ) => X p.1) (fun p => b p.1) (fun p => A p.1)
    Prod.snd (fun p => t p.1) (hX.comp measurable_fst) (hb.comp measurable_fst)
    (fun k => (hA k).comp measurable_fst)
    (fun k => (measurable_pi_apply k).comp measurable_snd) (ht.comp measurable_fst)
  exact (hφ.measurable.comp hstate).stronglyMeasurable.integral_prod_right'.measurable

/-- Measurability of the genuine frozen-generator expectation with
random parameters, Section 4.3 (2)--(3). -/
theorem gaussianGeneratorFlow_measurable_parameters {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (X b : Ω → EucSpace d) (A : Ω → Fin d → ℝ) (t : Ω → ℝ)
    (hX : Measurable X) (hb : Measurable b) (hA : ∀ k, Measurable (fun ω => A ω k))
    (ht : Measurable t) (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Measurable (fun ω => gaussianVectorFlow (frozenGaussianGenerator (b ω) (A ω) φ)
      (X ω) (b ω) (A ω) (t ω)) := by
  have hstate := gaussianVectorState_measurable_parameters
    (fun p : Ω × (Fin d → ℝ) => X p.1) (fun p => b p.1) (fun p => A p.1)
    Prod.snd (fun p => t p.1) (hX.comp measurable_fst) (hb.comp measurable_fst)
    (fun k => (hA k).comp measurable_fst)
    (fun k => (measurable_pi_apply k).comp measurable_snd) (ht.comp measurable_fst)
  have hg := frozenGaussianGenerator_measurable_parameters
    (fun p : Ω × (Fin d → ℝ) => b p.1) (fun p => A p.1) _
    (hb.comp measurable_fst) (fun k => (hA k).comp measurable_fst) hstate φ hφ
  exact hg.stronglyMeasurable.integral_prod_right'.measurable

/-- Joint nonvacuity of variable-state hypotheses, Section 4.3:
identity states with varying coordinate amplitudes and real time. -/
example : Continuous (id : EucSpace 1 → EucSpace 1) ∧
    Measurable (id : EucSpace 1 → EucSpace 1) ∧
    (∀ k : Fin 1, Continuous (fun x : EucSpace 1 => x k)) ∧
    (∀ k : Fin 1, Measurable (fun x : EucSpace 1 => x k)) :=
  ⟨continuous_id, measurable_id, fun k => PiLp.continuous_apply 2 (fun _ : Fin 1 => ℝ) k,
    fun k => (PiLp.continuous_apply 2 (fun _ : Fin 1 => ℝ) k).measurable⟩

/-- Nonvacuity of the C2 observable hypothesis for the variable
generator, Section 4.3: a normalized nonzero observable. -/
example : BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
