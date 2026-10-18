/-
# Phase model — gradient descent

Gradient descent on a phase energy `V` with integer rows `a_r`, with a step
`0 < η` and `η Σ_r ‖a_r‖² < 1`.  Every orbit comes arbitrarily close, up to
periods, to a point of the cube (`exists_frequently_periods`), and each such
limit point is a critical point of `V`
(`gradient_eq_zero_of_frequently_periods`).  As `V` is analytic, every orbit
converges, to a critical point (`exists_tendsto_gradient_phaseEnergy_eq_zero`).
An orbit with a limit point at a zero of `V` where no nonzero vector is
orthogonal to every row converges to a zero
(`exists_tendsto_phaseEnergy_eq_zero`).  For almost every start,
an orbit with a limit point where the Hessian is injective converges to a
local minimizer differing from it by a period
(`ae_exists_tendsto_isLocalMin_of_frequently_periods`); when the Hessian is
injective at every critical point, almost every orbit converges to a local
minimizer (`ae_exists_tendsto_isLocalMin_periods`).
-/

import Transformer.PhaseModel.Periods
import Transformer.SaddleAvoidance.Nondegenerate
import Transformer.SaddleAvoidance.Analytic

open Filter MeasureTheory Real
open scoped Topology NNReal

namespace Transformer.PhaseModel

open SaddleAvoidance (gradientStep)

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ} {R : Type*} [Fintype R] (a : R → Fin n → ℤ) {η : ℝ}

/-- **Every orbit has a limit point up to periods in the cube.** -/
theorem exists_frequently_periods (η : ℝ) (x : EucSpace n) :
    ∃ z ∈ cube n, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods n,
      ‖(gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x + v - z‖ < ε :=
  SaddleAvoidance.exists_frequently_norm_add_sub_lt (isCompact_cube n)
    fun _ => exists_add_mem_cube _

/-- **Every limit point up to periods is a critical point** of the phase
energy, for `0 < η` with `η Σ_r ‖a_r‖² < 1`. -/
theorem gradient_eq_zero_of_frequently_periods (hη : 0 < η)
    (hηL : η * ∑ r, ‖intVec (a r)‖ ^ 2 < 1) {x z : EucSpace n}
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods n,
      ‖(gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x + v - z‖ < ε) :
    gradient (phaseEnergy fun r => intVec (a r)) z = 0 :=
  SaddleAvoidance.gradient_eq_zero_of_frequently
    ((contDiff_phaseEnergy _ (k := 1)).differentiable one_ne_zero)
    (lipschitzWith_gradient_phaseEnergy _) hη (by push_cast; exact hηL) (isClosed_periods n)
    (fun _ hv ψ => phaseEnergy_add_of_mem_periods a hv ψ) hz

/-- **Gradient descent on a phase energy with integer rows converges from
every start, to a critical point**, for `0 < η` with `η Σ_r ‖a_r‖² < 1`: the
orbit has a limit point up to periods, and the energy is analytic, so it
satisfies the Łojasiewicz inequality there. -/
theorem exists_tendsto_gradient_phaseEnergy_eq_zero (hη : 0 < η)
    (hηL : η * ∑ r, ‖intVec (a r)‖ ^ 2 < 1) (x : EucSpace n) :
    ∃ w, Tendsto (fun k => (gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x) atTop
      (𝓝 w) ∧ gradient (phaseEnergy fun r => intVec (a r)) w = 0 := by
  obtain ⟨z, -, hz⟩ := exists_frequently_periods a η x
  obtain ⟨w, hw, hwx⟩ := SaddleAvoidance.exists_tendsto_of_analyticOnNhd
    (analyticOnNhd_phaseEnergy _) (lipschitzWith_gradient_phaseEnergy _) hη
    (by push_cast; exact hηL) (fun _ hv ψ => phaseEnergy_add_of_mem_periods a hv ψ) hz
  exact ⟨w, hwx, hw⟩

/-- **An orbit with a nondegenerate zero as a limit point converges to a
zero** of the phase energy.  Let `0 < η` with `η Σ_r ‖a_r‖² < 1`, let
`V(z) = 0`, and let no nonzero vector be orthogonal to every row.  If the
orbit comes arbitrarily close to `z` up to periods, it converges to a zero
`w` with `z - w ∈ 2πℤⁿ`. -/
theorem exists_tendsto_phaseEnergy_eq_zero (hη : 0 < η)
    (hηL : η * ∑ r, ‖intVec (a r)‖ ^ 2 < 1) {x z : EucSpace n}
    (hz0 : phaseEnergy (fun r => intVec (a r)) z = 0)
    (hspan : ∀ h, (∀ r, ⟪intVec (a r), h⟫ = 0) → h = 0)
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods n,
      ‖(gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x + v - z‖ < ε) :
    ∃ w, Tendsto (fun k => (gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x) atTop
      (𝓝 w) ∧ phaseEnergy (fun r => intVec (a r)) w = 0 ∧ z - w ∈ periods n := by
  have hfd := (contDiff_phaseEnergy (fun r => intVec (a r)) (k := 1)).differentiable one_ne_zero
  obtain ⟨w, hw⟩ := SaddleAvoidance.exists_tendsto_of_isLojasiewiczAt hfd
    (lipschitzWith_gradient_phaseEnergy _) hη (by push_cast; exact hηL)
    (fun _ hv ψ => phaseEnergy_add_of_mem_periods a hv ψ) (by norm_num) (by norm_num)
    (SaddleAvoidance.isLojasiewiczAt_of_injective hfd
      (hasFDerivAt_gradient_phaseEnergy _ z).differentiableAt.hasFDerivAt
      (gradient_eq_zero_of_phaseEnergy_eq_zero _ hz0)
      (injective_fderiv_gradient_of_phaseEnergy_eq_zero _ hz0 hspan)) hz
  have hzw := SaddleAvoidance.sub_mem_of_tendsto_of_frequently (isClosed_periods n) hw hz
  refine ⟨w, hw, ?_, hzw⟩
  rw [← phaseEnergy_add_of_mem_periods a hzw w, add_sub_cancel]
  exact hz0

/-- **Gradient descent on a phase energy converges to a local minimizer from
a nondegenerate limit point**, for almost every start.  Let `0 < η` with
`η Σ_r ‖a_r‖² < 1`.  For a start drawn from a law `ν` absolutely continuous
with respect to a Haar measure, almost surely: if `z` is a limit point of
the orbit up to periods and the Hessian at `z` is injective, then the orbit
converges to a local minimizer `w` with `z - w ∈ 2πℤⁿ`. -/
theorem ae_exists_tendsto_isLocalMin_of_frequently_periods (μ : Measure (EucSpace n))
    [μ.IsAddHaarMeasure] {ν : Measure (EucSpace n)} (hν : ν ≪ μ) (hη : 0 < η)
    (hηL : η * ∑ r, ‖intVec (a r)‖ ^ 2 < 1) :
    ∀ᵐ x ∂ν, ∀ z, Function.Injective
        (fderiv ℝ (gradient (phaseEnergy fun r => intVec (a r))) z) →
      (∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods n,
        ‖(gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x + v - z‖ < ε) →
      ∃ w, Tendsto (fun k => (gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x)
        atTop (𝓝 w) ∧ IsLocalMin (phaseEnergy fun r => intVec (a r)) w ∧ z - w ∈ periods n :=
  SaddleAvoidance.ae_exists_tendsto_isLocalMin_of_frequently μ hν (contDiff_phaseEnergy _)
    (lipschitzWith_gradient_phaseEnergy _) hη (by push_cast; exact hηL) (isClosed_periods n)
    fun _ hv ψ => phaseEnergy_add_of_mem_periods a hv ψ

/-- **Gradient descent on a phase energy whose critical points are all
nondegenerate converges to a local minimizer from almost every start.**  Let
`0 < η` with `η Σ_r ‖a_r‖² < 1`, and let the Hessian be injective at every
critical point.  For a start drawn from a law `ν` absolutely continuous with
respect to a Haar measure, the orbit almost surely converges to a local
minimizer. -/
theorem ae_exists_tendsto_isLocalMin_periods (μ : Measure (EucSpace n)) [μ.IsAddHaarMeasure]
    {ν : Measure (EucSpace n)} (hν : ν ≪ μ) (hη : 0 < η)
    (hηL : η * ∑ r, ‖intVec (a r)‖ ^ 2 < 1)
    (hcrit : ∀ z, gradient (phaseEnergy fun r => intVec (a r)) z = 0 →
      Function.Injective (fderiv ℝ (gradient (phaseEnergy fun r => intVec (a r))) z)) :
    ∀ᵐ x ∂ν, ∃ w, Tendsto (fun k => (gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x)
      atTop (𝓝 w) ∧ IsLocalMin (phaseEnergy fun r => intVec (a r)) w :=
  SaddleAvoidance.ae_exists_tendsto_isLocalMin_of_injective μ hν (contDiff_phaseEnergy _)
    (lipschitzWith_gradient_phaseEnergy _) hη (by push_cast; exact hηL) (isClosed_periods n)
    (fun _ hv ψ => phaseEnergy_add_of_mem_periods a hv ψ) (isCompact_cube n) exists_add_mem_cube
    hcrit

/-- The hypotheses of the theorems above hold for the single row `e_0` in
dimension one, `V(ψ) = 1 - cos ψ_0`, with Lebesgue measure and `η = 1/2`:
`‖e_0‖² = 1`, `0` is a zero where only `0` is orthogonal to the row, the
orbit of `0` stays at `0`, and at every critical point the cosine of the
relation is `±1`, a strict sign. -/
example : let a := fun (_ : Unit) (_ : Fin 1) => (1 : ℤ)
    let V := phaseEnergy fun r => intVec (a r)
    (volume : Measure (EucSpace 1)).IsAddHaarMeasure ∧
      (volume : Measure (EucSpace 1)) ≪ volume ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ∑ r, ‖intVec (a r)‖ ^ 2 < 1 ∧ V 0 = 0 ∧
      (∀ h, (∀ r, ⟪intVec (a r), h⟫ = 0) → h = 0) ∧
      (∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods 1,
        ‖(gradientStep V (1 / 2))^[k] 0 + v - 0‖ < ε) ∧
      ∀ z, gradient V z = 0 → Function.Injective (fderiv ℝ (gradient V) z) := by
  intro a V
  have hspan : ∀ h, (∀ r, ⟪intVec (a r), h⟫ = 0) → h = 0 := fun h hh => by
    have h0 := hh ()
    simp only [inner_intVec, a, Int.cast_one, one_mul, Fin.sum_univ_one] at h0
    refine PiLp.ext fun i => ?_
    rw [Subsingleton.elim i 0, h0]
    rfl
  have hV0 : V 0 = 0 := by simp [V, phaseEnergy]
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl, by norm_num, ?_, hV0, hspan,
    fun ε hε => ?_, fun z hz => ?_⟩
  · simp only [norm_intVec_sq, a, Int.cast_one, one_pow, Finset.sum_const, Finset.card_univ,
      Fintype.card_unit, Fintype.card_fin, one_smul]
    norm_num
  · have hg : gradient V 0 = 0 := gradient_eq_zero_of_phaseEnergy_eq_zero _ hV0
    have h0 : gradientStep V (1 / 2) 0 = 0 := by
      simp [SaddleAvoidance.gradientStep, hg]
    refine Frequently.of_forall fun k => ⟨0, fun _ => ⟨0, by simp⟩, ?_⟩
    rw [Function.iterate_fixed h0, add_zero, sub_zero, norm_zero]
    exact hε
  · have he : intVec (a ()) ≠ 0 := fun he => by
      have h1 := congrArg (fun v : EucSpace 1 => v.ofLp 0) he
      simp [intVec, a] at h1
    have hs : sin ⟪intVec (a ()), z⟫ = 0 := by
      rw [gradient_phaseEnergy] at hz
      simp only [Fintype.sum_unique] at hz
      exact (smul_eq_zero.1 hz).resolve_right he
    refine injective_fderiv_gradient_phaseEnergy _ (s := cos ⟪intVec (a ()), z⟫)
      (fun r => ?_) hspan
    obtain ⟨⟩ := r
    rcases sin_eq_zero_iff_cos_eq.1 hs with h | h <;> rw [h] <;> norm_num

/-- `exists_tendsto_gradient_phaseEnergy_eq_zero` applies to the single row
`e_0` in dimension one with `η = 1/2`, as `‖e_0‖² = 1`, from the start `0`. -/
example := exists_tendsto_gradient_phaseEnergy_eq_zero (fun (_ : Unit) (_ : Fin 1) => (1 : ℤ))
  (η := 1 / 2) (by norm_num) (by
    simp only [norm_intVec_sq, Int.cast_one, one_pow, Finset.sum_const, Finset.card_univ,
      Fintype.card_unit, Fintype.card_fin, one_smul]
    norm_num) 0

end Transformer.PhaseModel
