import Transformer.Modes.Section3_DensityObstruction
import Transformer.Modes.Section3_PsiBounds

/-
# The number of modes of a Gaussian KDE — counterexamples to `eq:error-higher`

In §3.2, `lem:error-higher` claims globally uniform weighted errors between
the standardized density and the Gaussian, or its first Edgeworth correction.
Both comparison functions are bounded. Hence any finite uniform error would
produce a bounded density, contrary to the Gaussian-tail mass obstruction
when `β > 2`, `0 < n < β+2`.

This refutes the pointwise estimates themselves, without assuming continuity
of the density. The source's displayed `s = 3` formula omits the Edgeworth
term; both that literal display and the corrected residual `g₃` are refuted.
The constants on the right may be arbitrary real numbers: in particular,
the proposed moment-dependent bounds cannot hold at even a single such `t`.

The asymptotic regime `n = β = k+1`, with `c = 1`, lies in this counterexample
range eventually. The negative uniform statements retaining the source's
rates and quantifiers are recorded in `Section3_ErrorHigher.lean`.

Source: arXiv:2412.09080v3, §2.2, the definition of `φ`; §3.1, `eq:psi`;
§3.2, `lem:error-higher`, `eq:error-higher`; §5.5, `lem: pt.bdd`.
-/

open Real MeasureTheory

namespace Transformer.Modes

/-- The Gaussian target in `eq:error-higher` is bounded on the entire plane.
Source: arXiv:2412.09080v3, §2.2, the definition of `φ`. -/
theorem abs_phi2_le (z : ℝ × ℝ) : |phi2 z| ≤ (2 * π)⁻¹ := by
  have hφ : 0 ≤ phi2 z := by unfold phi2; positivity
  rw [abs_of_nonneg hφ, phi2]
  calc (2 * π)⁻¹ * Real.exp (-(z.1 ^ 2 + z.2 ^ 2) / 2)
      ≤ (2 * π)⁻¹ * 1 := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        exact Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg z.1, sq_nonneg z.2])
    _ = _ := mul_one _

/-- The first Edgeworth correction is bounded, with the source's third
moment retained in the constant. Source: arXiv:2412.09080v3, §3.1, `eq:psi`. -/
theorem abs_psi_lawY_le {β : ℝ} (hβ : 0 < β) (t : ℝ) (z : ℝ × ℝ) :
    |psiOf (lawY β t) z| ≤ 876 * (2 * π)⁻¹ * etaMoment β t 3 := by
  have he : Real.exp (-(eucl z ^ 2) / 4) ≤ 1 :=
    Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg (eucl z)])
  calc |psiOf (lawY β t) z|
      ≤ 876 * (2 * π)⁻¹ * etaMoment β t 3 * Real.exp (-(eucl z ^ 2) / 4) :=
        abs_psi_lawY_le_gaussian hβ t z
    _ ≤ (876 * (2 * π)⁻¹ * etaMoment β t 3) * 1 :=
        mul_le_mul_of_nonneg_left he (by positivity [etaMoment_nonneg β t 3])
    _ = _ := mul_one _

/-- The correction bound's positivity assumption has a positive witness. -/
example : (0 : ℝ) < 1 := one_pos

/-- The complete first-order Edgeworth target is bounded for every sample
count. Source: arXiv:2412.09080v3, §3.2, the definition of `g₃`. -/
theorem abs_edgeworth_target_le (n : ℕ) {β : ℝ} (hβ : 0 < β) (t : ℝ) (z : ℝ × ℝ) :
    |phi2 z + (Real.sqrt n)⁻¹ * psiOf (lawY β t) z| ≤
      (2 * π)⁻¹ + (Real.sqrt n)⁻¹ * (876 * (2 * π)⁻¹ * etaMoment β t 3) := by
  calc |phi2 z + (Real.sqrt n)⁻¹ * psiOf (lawY β t) z|
      ≤ |phi2 z| + |(Real.sqrt n)⁻¹ * psiOf (lawY β t) z| := abs_add_le _ _
    _ = |phi2 z| + (Real.sqrt n)⁻¹ * |psiOf (lawY β t) z| := by
        rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ (Real.sqrt n)⁻¹)]
    _ ≤ (2 * π)⁻¹ + (Real.sqrt n)⁻¹ *
        (876 * (2 * π)⁻¹ * etaMoment β t 3) :=
        add_le_add (abs_phi2_le z)
          (mul_le_mul_of_nonneg_left (abs_psi_lawY_le hβ t z) (by positivity))

/-- A positive bandwidth satisfies the Edgeworth target's only hypothesis. -/
example : (0 : ℝ) < 1 := one_pos

/-- **Counterexample to `eq:error-higher`, `s = 2`.** No density can satisfy
any finite globally weighted error bound in this parameter range.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, `eq:error-higher`. -/
theorem not_weighted_gaussian_density_scaledSum {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ,
      IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ∧
      ∃ E : ℝ, ∀ z, (1 + eucl z ^ 2) * |q z - phi2 z| ≤ E := by
  apply not_weighted_density_approximation hn hβ hnβ t abs_phi2_le
  intro z
  linarith [sq_nonneg (eucl z)]

/-- The second-order counterexample holds inside the regime `β = n`. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 5 ∧ (5 : ℝ) < 5 + 2 := by norm_num

/-- **The literal `s = 3` display in `eq:error-higher` is also false.**
The source writes `q_t-φ` even for `s = 3`; increasing the weight cannot
remove the bounded-density obstruction.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, `eq:error-higher`. -/
theorem not_weighted_gaussian_cube_density_scaledSum {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ,
      IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ∧
      ∃ E : ℝ, ∀ z, (1 + eucl z ^ 3) * |q z - phi2 z| ≤ E := by
  apply not_weighted_density_approximation hn hβ hnβ t abs_phi2_le
  intro z
  have hr : 0 ≤ eucl z := Real.sqrt_nonneg _
  have hcube : 0 ≤ eucl z ^ 3 := pow_nonneg hr _
  linarith

/-- All numerical assumptions of the literal `s = 3` refutation hold. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 5 ∧ (5 : ℝ) < 5 + 2 := by norm_num

/-- **Correcting the missing Edgeworth term does not repair the claim.**
The residual `g₃ = q_t-φ-n^{-1/2}ψ` cannot have a finite uniform weighted
bound either, since the corrected comparison function is bounded.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, corrected `eq:error-higher`. -/
theorem not_weighted_edgeworth_density_scaledSum {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ,
      IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ∧
      ∃ E : ℝ, ∀ z, (1 + eucl z ^ 3) *
        |q z - phi2 z - (Real.sqrt n)⁻¹ * psiOf (lawY β t) z| ≤ E := by
  rintro ⟨q, hqP, E, hE⟩
  have hβ0 : 0 < β := by linarith
  apply not_weighted_density_approximation hn hβ hnβ t
    (abs_edgeworth_target_le n hβ0 t) (w := fun z => 1 + eucl z ^ 3)
  · intro z
    have hr : 0 ≤ eucl z := Real.sqrt_nonneg _
    have hcube : 0 ≤ eucl z ^ 3 := pow_nonneg hr _
    linarith
  · refine ⟨q, hqP, E, fun z => ?_⟩
    simpa only [sub_add_eq_sub_sub] using hE z

/-- The corrected residual's counterexample has simultaneous witnesses. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 5 ∧ (5 : ℝ) < 5 + 2 := by norm_num

/-- At the source's central observation point and a valid sample count,
even a density without continuity cannot obey the second-order estimate. -/
example : ¬ ∃ q : ℝ × ℝ → ℝ,
    IsDensityOf (Measure.pi fun _ : Fin 5 => lawY 5 0) (scaledSum 5) q ∧
    ∃ E : ℝ, ∀ z, (1 + eucl z ^ 2) * |q z - phi2 z| ≤ E :=
  not_weighted_gaussian_density_scaledSum (by norm_num) (by norm_num) (by norm_num) 0

/-- The source's literal cubic display already fails at the same central
observation point, with no continuity requirement. -/
example : ¬ ∃ q : ℝ × ℝ → ℝ,
    IsDensityOf (Measure.pi fun _ : Fin 5 => lawY 5 0) (scaledSum 5) q ∧
    ∃ E : ℝ, ∀ z, (1 + eucl z ^ 3) * |q z - phi2 z| ≤ E :=
  not_weighted_gaussian_cube_density_scaledSum
    (by norm_num) (by norm_num) (by norm_num) 0

/-- Including the first Edgeworth correction still gives an impossible
uniform bound at the center of the source's window. -/
example : ¬ ∃ q : ℝ × ℝ → ℝ,
    IsDensityOf (Measure.pi fun _ : Fin 5 => lawY 5 0) (scaledSum 5) q ∧
    ∃ E : ℝ, ∀ z, (1 + eucl z ^ 3) *
      |q z - phi2 z - (Real.sqrt 5)⁻¹ * psiOf (lawY 5 0) z| ≤ E :=
  not_weighted_edgeworth_density_scaledSum (by norm_num) (by norm_num) (by norm_num) 0

end Transformer.Modes
