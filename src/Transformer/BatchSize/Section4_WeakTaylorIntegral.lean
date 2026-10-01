/-
# Averaging the Taylor remainder

arXiv:2506.12543v1, Section 3.2, equation (1), and Section 4.3, Theorem 1.
An L2 random update gives a weak Taylor bound controlled by its second moment.
-/

import Transformer.BatchSize.Section3_WeakTaylor

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Weak Taylor bound for a genuine square-integrable random update,
used to compare a stochastic optimizer with its mean drift in Section 4.3.
No diffusion existence or approximation theorem is assumed. -/
theorem weak_taylor_integral {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (V : Ω → EucSpace d)
    (hV : Measurable V) (hV2 : MemLp V 2 P) (φ : EucSpace d → ℝ) (M : ℝ)
    (hφ : ContDiff ℝ 2 φ) (hH : ∀ y, ‖iteratedFDeriv ℝ 2 φ y‖ ≤ M) (x : EucSpace d) :
    |(∫ ω, φ (x + V ω) ∂P) - φ x - fderiv ℝ φ x (∫ ω, V ω ∂P)| ≤
      M * (∫ ω, ‖V ω‖ ^ 2 ∂P) := by
  let R : Ω → ℝ := fun ω => φ (x + V ω) - φ x - fderiv ℝ φ x (V ω)
  have hpoint (ω : Ω) : |R ω| ≤ M * ‖V ω‖ ^ 2 :=
    uniform_first_order_taylor φ M hφ hH x (V ω)
  have hiV : Integrable V P := hV2.integrable (by norm_num)
  have hiNorm : Integrable (fun ω => ‖V ω‖ ^ 2) P := hV2.integrable_norm_pow (by norm_num)
  have hiBound := hiNorm.const_mul M
  have hxV : Measurable (fun ω => x + V ω) := by
    simpa [Pi.add_def] using (measurable_const (a := x)).add hV
  have hR : Measurable R :=
    (hφ.continuous.measurable.comp hxV).sub measurable_const |>.sub
      ((fderiv ℝ φ x).continuous.measurable.comp hV)
  have hiL := (fderiv ℝ φ x).integrable_comp hiV
  have hiR : Integrable R P := by
    apply hiBound.mono' hR.aestronglyMeasurable
    exact ae_of_all _ (fun ω => by simpa [Real.norm_eq_abs] using hpoint ω)
  have hiφ : Integrable (fun ω => φ (x + V ω)) P := by
    convert (hiR.add (integrable_const (φ x))).add hiL using 1
    funext ω
    dsimp [R]
    ring
  have hr : (∫ ω, R ω ∂P) = (∫ ω, φ (x + V ω) ∂P) - φ x -
      fderiv ℝ φ x (∫ ω, V ω ∂P) := by
    change (∫ ω, φ (x + V ω) - φ x - fderiv ℝ φ x (V ω) ∂P) = _
    rw [integral_sub (f := fun ω => φ (x + V ω) - φ x)
        (g := fun ω => fderiv ℝ φ x (V ω)) (hiφ.sub (integrable_const _)) hiL,
      integral_sub (f := fun ω => φ (x + V ω)) (g := fun _ => φ x)
        hiφ (integrable_const _), (fderiv ℝ φ x).integral_comp_comm hiV]
    simp
  have hb := norm_integral_le_of_norm_le (f := R) hiBound
    (ae_of_all P (fun ω => by simpa [Real.norm_eq_abs] using hpoint ω))
  rw [hr, integral_const_mul] at hb
  simpa only [Real.norm_eq_abs] using hb

/-- Nonvacuity of the weak Taylor assumptions, Section 4.3. A constant
unit update is measurable and L2, and a constant test has zero Hessian. -/
example : Measurable (fun _ : Unit => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ∧
    MemLp (fun _ : Unit => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 2 (Measure.dirac ()) ∧
    ContDiff ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    (∀ y : EucSpace 1, ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) y‖ ≤ 1) := by
  refine ⟨measurable_const, memLp_const _, contDiff_const, ?_⟩
  intro y
  rw [iteratedFDeriv_const_of_ne (by norm_num)]
  simp

end Transformer.BatchSize
