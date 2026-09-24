/-
# The Gaussian-mixture tail event implies separated radial projections

The probabilistic event controlled by `GaussianMixtureTail` is connected here
to the geometric event of `prop: mixture.of.gaussians`. The key is the
quadratic cap estimate in `InitialCapGeometry`.

Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`.
-/

import Transformer.Metastability.GaussianMixtureTail
import Transformer.Metastability.InitialCapGeometry

open Real MeasureTheory

namespace Transformer
namespace Metastability

/-- If every nonzero sample lies below the Gaussian radius threshold around
some centre, then the radial projections are `(β, ε)`-separated.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem projectedSeparated_of_not_far_centers (d n r : ℕ) (β ε σ R : ℝ)
    (hσ : 0 < σ) (hr : 0 < r) (w : Idx r → SSphere d)
    (hcent : isCentered d n β ε r w)
    (hR : 8 * σ ^ 2 * R ≤ (r : ℝ) * ε)
    (X : Idx n → EucSpace d) (hX : ∀ i, X i ≠ 0)
    (hnotfar : ¬ ∃ i : Idx n, ∀ q : Idx r,
      R ≤ ‖X i - Real.sqrt (r : ℝ) • (w q : EucSpace d)‖ ^ 2 / (4 * σ ^ 2)) :
    X ∈ projectedSeparated d n β ε := by
  have hrℝ : (0 : ℝ) < r := by exact_mod_cast hr
  have ha : 0 < Real.sqrt (r : ℝ) := Real.sqrt_pos.2 hrℝ
  have hden : 0 < 4 * σ ^ 2 := by positivity
  apply projectedSeparated_of_sq_near_scaled_centers d n β ε r hr w hcent X hX
  intro i
  have hnotall : ¬ ∀ q : Idx r,
      R ≤ ‖X i - Real.sqrt (r : ℝ) • (w q : EucSpace d)‖ ^ 2 /
        (4 * σ ^ 2) := by
    intro hall
    exact hnotfar ⟨i, hall⟩
  push Not at hnotall
  obtain ⟨q, hq⟩ := hnotall
  refine ⟨q, ?_⟩
  have hdist : ‖X i - Real.sqrt (r : ℝ) • (w q : EucSpace d)‖ ^ 2 <
      (r : ℝ) * ε / 2 := by
    have h₁ := (div_lt_iff₀ hden).mp hq
    nlinarith [hR]
  have hvec : X i - Real.sqrt (r : ℝ) • (w q : EucSpace d) =
      Real.sqrt (r : ℝ) •
        ((Real.sqrt (r : ℝ))⁻¹ • X i - (w q : EucSpace d)) := by
    rw [smul_sub, smul_smul, mul_inv_cancel₀ ha.ne', one_smul]
  have hnorm : ‖X i - Real.sqrt (r : ℝ) • (w q : EucSpace d)‖ ^ 2 =
      (r : ℝ) * ‖(Real.sqrt (r : ℝ))⁻¹ • X i - (w q : EucSpace d)‖ ^ 2 := by
    rw [hvec, norm_smul, Real.norm_eq_abs, abs_of_pos ha, mul_pow,
      Real.sq_sqrt hrℝ.le]
  rw [hnorm] at hdist
  have hscaled' : (r : ℝ) *
      ‖(Real.sqrt (r : ℝ))⁻¹ • X i - (w q : EucSpace d)‖ ^ 2 <
        (r : ℝ) * (ε / 2) := by
    convert hdist using 1; ring
  have hscaled : ‖(Real.sqrt (r : ℝ))⁻¹ • X i - (w q : EucSpace d)‖ ^ 2 <
      ε / 2 := lt_of_mul_lt_mul_left hscaled' hrℝ.le
  linarith

/-- The corrected `n`-sample mixture law gives a quantitative lower bound
on separated radial projections whenever the squared-radius threshold fits
inside the cap. `mixture_separation` combines this with the numerical scale
bound to obtain the stated `2e^{-d}` estimate.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem mixtureLaw_projectedSeparated_markov (d n r : ℕ) (hd : 0 < d)
    (β ε σ R : ℝ) (hσ : 0 < σ) (hr : 0 < r)
    (w : Idx r → SSphere d) (hcent : isCentered d n β ε r w)
    (hR : 8 * σ ^ 2 * R ≤ (r : ℝ) * ε) :
    Real.exp R *
      (1 - (mixtureLaw d n r σ fun q => (w q : EucSpace d)).real
        (projectedSeparated d n β ε)) ≤
      (n : ℝ) * (2 : ℝ) ^ ((d : ℝ) / 2) := by
  let μ : Measure (Idx n → EucSpace d) :=
    mixtureLaw d n r σ fun q => (w q : EucSpace d)
  let B : Set (Idx n → EucSpace d) :=
    {X | ∃ i : Idx n, ∀ q : Idx r,
      R ≤ ‖X i - Real.sqrt (r : ℝ) • (w q : EucSpace d)‖ ^ 2 / (4 * σ ^ 2)}
  let Z : Set (Idx n → EucSpace d) := {X | ∃ i : Idx n, X i = 0}
  let _ : IsProbabilityMeasure μ :=
    mixtureLaw_probability d n r σ hσ hr (fun q => (w q : EucSpace d))
  have hB : MeasurableSet B := by
    dsimp [B]
    measurability
  have hZ : μ Z = 0 := by
    have hae := mixtureLaw_nonzero_ae d n r hd σ hσ hr
      (fun q => (w q : EucSpace d))
    have hz := (ae_iff).mp hae
    simpa only [Z, μ, not_forall, ne_eq, not_not] using hz
  have hsubset : Bᶜ \ Z ⊆ projectedSeparated d n β ε := by
    intro X hX
    have hnotfar : X ∉ B := hX.1
    have hnonzero : ∀ i, X i ≠ 0 := by
      intro i hi
      exact hX.2 ⟨i, hi⟩
    exact projectedSeparated_of_not_far_centers d n r β ε σ R hσ hr w hcent hR
      X hnonzero hnotfar
  have hbound : μ.real (Bᶜ \ Z) ≤ μ.real (projectedSeparated d n β ε) :=
    measureReal_mono hsubset (measure_lt_top μ _).ne
  have hdiff : μ.real (Bᶜ \ Z) = μ.real Bᶜ := by
    simp only [measureReal_def, measure_sdiff_null hZ]
  have hcompl : μ.real Bᶜ = 1 - μ.real B := by
    rw [measureReal_compl hB]
    simp [measureReal_def]
  have htail : Real.exp R * μ.real B ≤
      (n : ℝ) * (2 : ℝ) ^ ((d : ℝ) / 2) :=
    mixtureLaw_far_centers_tail d n r σ hσ hr
      (fun q => (w q : EucSpace d)) R
  change Real.exp R * (1 - μ.real (projectedSeparated d n β ε)) ≤ _
  have hbad : 1 - μ.real (projectedSeparated d n β ε) ≤ μ.real B := by
    rw [hdiff, hcompl] at hbound
    linarith
  exact (mul_le_mul_of_nonneg_left hbad (Real.exp_pos R).le).trans htail

/-- Unit scale, one centre and zero displacement satisfy the numeric
threshold in the deterministic bridge. -/
example : (0 : ℝ) < 1 ∧ (0 : ℕ) < 1 ∧
    8 * (1 : ℝ) ^ 2 * 0 ≤ ((1 : ℕ) : ℝ) * 1 := by norm_num

end Metastability
end Transformer
