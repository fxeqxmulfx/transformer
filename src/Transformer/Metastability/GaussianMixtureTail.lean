/-
# A radius tail bound for the corrected Gaussian mixture

A sample far from every component centre is, in particular, far from the
centre of the component that generated it. Averaging the one-component
exponential Markov bound therefore gives the same estimate for the mixture.

This is a partial concentration estimate for `prop: mixture.of.gaussians`;
it does not replace the sharper concentration bound needed for that result.

Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`.
-/

import Transformer.Metastability.GaussianMixture
import Transformer.Metastability.GaussianTail

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

/-- The measure of points whose squared distance from *every* mixture centre
exceeds `4σ² R` obeys the same exponential Markov bound as a single component.
The printed density's normalizer is corrected as in `GaussianMixture`.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem gaussianMixture_far_centers_tail (d r : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (hr : 0 < r) (w : Idx r → EucSpace d) (R : ℝ) :
    Real.exp R *
      (volume.withDensity fun x : EucSpace d =>
        ENNReal.ofReal (gaussianMixtureDensity d r σ w x)).real
        {x | ∀ i : Idx r,
          R ≤ ‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (4 * σ ^ 2)} ≤
      (2 : ℝ) ^ ((d : ℝ) / 2) := by
  let F (i : Idx r) : Measure (EucSpace d) := volume.withDensity fun x =>
    ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
      Real.exp (-‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (2 * σ ^ 2)))
  let E : Set (EucSpace d) :=
    {x | ∀ i : Idx r, R ≤ ‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (4 * σ ^ 2)}
  have htail (i : Idx r) : Real.exp R * (F i).real E ≤
      (2 : ℝ) ^ ((d : ℝ) / 2) := by
    let _ : IsProbabilityMeasure (F i) :=
      radial_gaussian_probability d σ hσ (Real.sqrt (r : ℝ) • w i)
    have hmono : (F i).real E ≤
        (F i).real {x | R ≤ ‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (4 * σ ^ 2)} :=
      measureReal_mono (by intro x hx; exact hx i) (measure_lt_top (F i) _).ne
    exact (mul_le_mul_of_nonneg_left hmono (Real.exp_pos R).le).trans
      (radial_gaussian_exp_norm_sq_tail d σ hσ (Real.sqrt (r : ℝ) • w i) R)
  have hsum : (∑ i : Idx r, F i).real E = ∑ i : Idx r, (F i).real E := by
    simp only [measureReal_def, Measure.finsetSum_apply]
    rw [ENNReal.toReal_sum]
    intro i _
    let _ : IsProbabilityMeasure (F i) :=
      radial_gaussian_probability d σ hσ (Real.sqrt (r : ℝ) • w i)
    exact (measure_lt_top (F i) E).ne
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  have hfactor : (ENNReal.ofReal ((r : ℝ)⁻¹)).toReal = (r : ℝ)⁻¹ :=
    ENNReal.toReal_ofReal (inv_pos.mpr hrpos).le
  have hmix : (volume.withDensity fun x : EucSpace d =>
      ENNReal.ofReal (gaussianMixtureDensity d r σ w x)) =
      ENNReal.ofReal ((r : ℝ)⁻¹) • ∑ i : Idx r, F i :=
    gaussianMixtureComponent_eq_average d r σ hσ hr w
  change Real.exp R *
    (volume.withDensity fun x : EucSpace d =>
      ENNReal.ofReal (gaussianMixtureDensity d r σ w x)).real E ≤ _
  rw [hmix, measureReal_ennreal_smul_apply, hfactor, hsum]
  calc
    Real.exp R * ((r : ℝ)⁻¹ * ∑ i : Idx r, (F i).real E) =
        (r : ℝ)⁻¹ * ∑ i : Idx r, Real.exp R * (F i).real E := by
      rw [← Finset.mul_sum]
      ring
    _ ≤ (r : ℝ)⁻¹ * ∑ _i : Idx r, (2 : ℝ) ^ ((d : ℝ) / 2) := by
      exact mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => htail i)
        (inv_pos.mpr hrpos).le
    _ = (2 : ℝ) ^ ((d : ℝ) / 2) := by
      simp [hr.ne']

/-- For `n` independent samples from the corrected mixture, the event that
at least one sample is far from every component centre has at most `n` times
the one-sample exponential Markov bound.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem mixtureLaw_far_centers_tail (d n r : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (hr : 0 < r) (w : Idx r → EucSpace d) (R : ℝ) :
    Real.exp R * (mixtureLaw d n r σ w).real
      {X | ∃ i : Idx n, ∀ q : Idx r,
        R ≤ ‖X i - Real.sqrt (r : ℝ) • w q‖ ^ 2 / (4 * σ ^ 2)} ≤
      (n : ℝ) * (2 : ℝ) ^ ((d : ℝ) / 2) := by
  let E : Set (EucSpace d) :=
    {x | ∀ q : Idx r, R ≤ ‖x - Real.sqrt (r : ℝ) • w q‖ ^ 2 / (4 * σ ^ 2)}
  let m : Measure (EucSpace d) := volume.withDensity fun x =>
    ENNReal.ofReal (gaussianMixtureDensity d r σ w x)
  have hE : MeasurableSet E := by
    dsimp [E]
    measurability
  have hmp (i : Idx n) : MeasurePreserving (Function.eval i) (mixtureLaw d n r σ w) m := by
    let _ : IsProbabilityMeasure m := gaussianMixtureComponent_probability d r σ hσ hr w
    simpa only [mixtureLaw, m] using (measurePreserving_eval (fun _ : Idx n => m) i)
  have hmarginal (i : Idx n) :
      (mixtureLaw d n r σ w).real {X | X i ∈ E} = m.real E := by
    simpa only [Set.preimage, Function.eval_apply] using
      (hmp i).measureReal_preimage hE.nullMeasurableSet
  have hset : {X : Idx n → EucSpace d | ∃ i : Idx n, X i ∈ E} =
      ⋃ i ∈ (Finset.univ : Finset (Idx n)), {X | X i ∈ E} := by
    ext X
    simp
  change Real.exp R * (mixtureLaw d n r σ w).real
    {X | ∃ i : Idx n, X i ∈ E} ≤ _
  have hbound := measureReal_biUnion_finset_le (μ := mixtureLaw d n r σ w)
    (Finset.univ : Finset (Idx n)) (fun i => {X | X i ∈ E})
  rw [← hset] at hbound
  calc
    Real.exp R * (mixtureLaw d n r σ w).real {X | ∃ i : Idx n, X i ∈ E} ≤
        Real.exp R * ∑ i : Idx n,
          (mixtureLaw d n r σ w).real {X | X i ∈ E} :=
      mul_le_mul_of_nonneg_left hbound (Real.exp_pos R).le
    _ = ∑ i : Idx n, Real.exp R * m.real E := by
      rw [Finset.mul_sum]
      simp_rw [hmarginal]
    _ ≤ ∑ _i : Idx n, (2 : ℝ) ^ ((d : ℝ) / 2) :=
      Finset.sum_le_sum fun _ _ => gaussianMixture_far_centers_tail d r σ hσ hr w R
    _ = (n : ℝ) * (2 : ℝ) ^ ((d : ℝ) / 2) := by simp

/-- The mixture-tail hypotheses are satisfiable with one component and unit
scale. -/
example : (0 : ℝ) < 1 ∧ (0 : ℕ) < 1 := ⟨one_pos, Nat.one_pos⟩

end Metastability
end Transformer
