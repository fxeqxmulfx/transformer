/-
# The normalized Gaussian mixture law

The manuscript's `eq: gaussian.mixture` prints a one-dimensional normalizer
for a density on `ℝ^d`. With the corrected `d`-dimensional normalizer, this
file proves that both the one-sample mixture and its `n`-fold product are
probability measures when `r ≥ 1` and `σ > 0`.

Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`.
-/

import Transformer.Metastability.GaussianIntegral
import Mathlib.MeasureTheory.Constructions.Pi

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Metastability

/-- **Equation (eq: gaussian.mixture), with corrected normalization.** Density
of the Gaussian mixture on `ℝ^d`:

  `f(x) = (1/(r (2π σ²)^(d/2))) Σ_{i=1}^r exp(-‖x - √r w_i‖² / (2 σ²))`.

The source prints the denominator `r √(2πσ²)`, which is the `d = 1`
normalizer; the displayed law in general dimension requires the power `d/2`.

Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
noncomputable def gaussianMixtureDensity
    (d r : ℕ) (σ : ℝ) (w : Idx r → EucSpace d) (x : EucSpace d) : ℝ :=
  (1 / ((r : ℝ) * (2 * Real.pi * σ^2) ^ ((d : ℝ) / 2))) *
    ∑ i : Idx r,
      Real.exp (-‖x - Real.sqrt (r : ℝ) • (w i)‖^2 / (2 * σ^2))

/-- The mixture density is an average of correctly normalized radial
Gaussians. Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem gaussianMixtureDensity_eq_sum_radial (d r : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (w : Idx r → EucSpace d) (x : EucSpace d) :
    gaussianMixtureDensity d r σ w x =
      (1 / (r : ℝ)) * ∑ i : Idx r,
        (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
          Real.exp (-‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (2 * σ ^ 2)) := by
  unfold gaussianMixtureDensity
  have hpow : (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) =
      ((2 * Real.pi * σ ^ 2) ^ ((d : ℝ) / 2))⁻¹ := by
    rw [show -(d : ℝ) / 2 = -((d : ℝ) / 2) by ring]
    rw [Real.rpow_neg (by positivity : 0 ≤ 2 * Real.pi * σ ^ 2)]
  rw [hpow]
  rw [← Finset.mul_sum]
  ring

/-- The corrected Gaussian mixture density is integrable.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem gaussianMixtureDensity_integrable (d r : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (w : Idx r → EucSpace d) : Integrable (gaussianMixtureDensity d r σ w) volume := by
  have heq : gaussianMixtureDensity d r σ w =
      (fun x : EucSpace d => (1 / (r : ℝ)) *
        ∑ i : Idx r,
          (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
            Real.exp (-‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (2 * σ ^ 2))) := by
    funext x
    exact gaussianMixtureDensity_eq_sum_radial d r σ hσ w x
  rw [heq]
  exact (integrable_finsetSum Finset.univ (fun i _ =>
    radial_gaussian_integrable d σ hσ (Real.sqrt (r : ℝ) • w i))).const_mul _

/-- The corrected mixture density integrates to one for `r ≥ 1` and `σ > 0`.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem gaussianMixtureDensity_integral (d r : ℕ) (σ : ℝ) (hσ : 0 < σ) (hr : 0 < r)
    (w : Idx r → EucSpace d) :
    ∫ x : EucSpace d, gaussianMixtureDensity d r σ w x = 1 := by
  have heq : (fun x : EucSpace d => gaussianMixtureDensity d r σ w x) =
      (fun x : EucSpace d => (1 / (r : ℝ)) *
        ∑ i : Idx r,
          (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
            Real.exp (-‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (2 * σ ^ 2))) := by
    funext x
    exact gaussianMixtureDensity_eq_sum_radial d r σ hσ w x
  rw [heq, integral_const_mul]
  rw [integral_finsetSum (s := Finset.univ) (fun i _ =>
    radial_gaussian_integrable d σ hσ (Real.sqrt (r : ℝ) • w i))]
  simp only [radial_gaussian_integral d σ hσ]
  simp [hr.ne']

/-- The corrected mixture density is nonnegative.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem gaussianMixtureDensity_nonneg (d r : ℕ) (σ : ℝ)
    (w : Idx r → EucSpace d) (x : EucSpace d) :
    0 ≤ gaussianMixtureDensity d r σ w x := by
  unfold gaussianMixtureDensity
  positivity

/-- One draw from the corrected mixture has a probability law.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem gaussianMixtureComponent_probability (d r : ℕ) (σ : ℝ)
    (hσ : 0 < σ) (hr : 0 < r) (w : Idx r → EucSpace d) :
    IsProbabilityMeasure (volume.withDensity fun x =>
      ENNReal.ofReal (gaussianMixtureDensity d r σ w x)) := by
  constructor
  rw [withDensity_apply _ MeasurableSet.univ]
  simp only [Measure.restrict_univ]
  rw [← ofReal_integral_eq_lintegral_ofReal
    (gaussianMixtureDensity_integrable d r σ hσ w)
    (Filter.Eventually.of_forall (gaussianMixtureDensity_nonneg d r σ w))]
  rw [gaussianMixtureDensity_integral d r σ hσ hr w]
  simp

/-- The corrected one-sample law is the uniform average of the radial
Gaussian component measures. This is the measure-level form of
`eq: gaussian.mixture` needed to reduce `prop: mixture.of.gaussians` to a
one-component concentration bound. The manuscript's one-dimensional
normalizer is corrected as in `gaussianMixtureDensity`.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem gaussianMixtureComponent_eq_average (d r : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (hr : 0 < r) (w : Idx r → EucSpace d) :
    volume.withDensity (fun x => ENNReal.ofReal (gaussianMixtureDensity d r σ w x)) =
      ENNReal.ofReal ((r : ℝ)⁻¹) •
        ∑ i : Idx r, volume.withDensity (fun x : EucSpace d =>
          ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
            Real.exp (-‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (2 * σ ^ 2)))) := by
  let f (i : Idx r) (x : EucSpace d) : ℝ :=
    (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
      Real.exp (-‖x - Real.sqrt (r : ℝ) • w i‖ ^ 2 / (2 * σ ^ 2))
  let F (i : Idx r) (x : EucSpace d) : ENNReal := ENNReal.ofReal (f i x)
  have hF (i : Idx r) : Measurable (F i) := by
    dsimp [F, f]
    fun_prop
  have hval (x : EucSpace d) :
      ENNReal.ofReal (gaussianMixtureDensity d r σ w x) =
        ENNReal.ofReal ((r : ℝ)⁻¹) * ∑ i : Idx r, F i x := by
    have hrℝ : (0 : ℝ) < (r : ℝ) := by exact_mod_cast hr
    rw [gaussianMixtureDensity_eq_sum_radial d r σ hσ w x]
    rw [show (1 / (r : ℝ)) = (r : ℝ)⁻¹ by ring]
    rw [ENNReal.ofReal_mul (inv_pos.mpr hrℝ).le]
    rw [ENNReal.ofReal_sum_of_nonneg (s := Finset.univ)
      (f := fun i : Idx r => f i x) (by intro i _; dsimp [f]; positivity)]
  ext E
  rw [withDensity_apply', Measure.smul_apply]
  simp only [Measure.finsetSum_apply]
  simp_rw [withDensity_apply']
  conv_lhs => arg 2; ext x; rw [hval x]
  rw [lintegral_const_mul' (ENNReal.ofReal ((r : ℝ)⁻¹))
    (fun x : EucSpace d => ∑ i : Idx r, F i x) (by simp)]
  rw [lintegral_finsetSum (s := Finset.univ) (fun i _ => hF i)]
  rfl

/-- The law of `n` i.i.d. draws from `eq: gaussian.mixture`. The corrected
density is a probability density for `r ≥ 1`, `σ > 0`.

Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
noncomputable def mixtureLaw (d n r : ℕ) (σ : ℝ) (w : Idx r → EucSpace d) :
    Measure (Idx n → EucSpace d) :=
  Measure.pi fun _ : Idx n =>
    volume.withDensity fun x => ENNReal.ofReal (gaussianMixtureDensity d r σ w x)

/-- The `n`-sample mixture law is a probability measure.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem mixtureLaw_probability (d n r : ℕ) (σ : ℝ) (hσ : 0 < σ) (hr : 0 < r)
    (w : Idx r → EucSpace d) : IsProbabilityMeasure (mixtureLaw d n r σ w) := by
  unfold mixtureLaw
  let _ : IsProbabilityMeasure (volume.withDensity fun x =>
      ENNReal.ofReal (gaussianMixtureDensity d r σ w x)) :=
    gaussianMixtureComponent_probability d r σ hσ hr w
  infer_instance

/-- Every sample from the corrected mixture is nonzero almost surely in
positive dimension. This removes the exceptional radial-projection case in
`prop: mixture.of.gaussians`; it uses absolute continuity of the component
density and the marginals of the product law.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem mixtureLaw_nonzero_ae (d n r : ℕ) (hd : 0 < d) (σ : ℝ) (hσ : 0 < σ)
    (hr : 0 < r) (w : Idx r → EucSpace d) :
    ∀ᵐ X ∂mixtureLaw d n r σ w, ∀ i : Idx n, X i ≠ 0 := by
  let m : Measure (EucSpace d) := volume.withDensity fun x =>
    ENNReal.ofReal (gaussianMixtureDensity d r σ w x)
  let _ : IsProbabilityMeasure m := gaussianMixtureComponent_probability d r σ hσ hr w
  have : Nontrivial (EucSpace d) :=
    nontrivial_of_ne (EuclideanSpace.single (⟨0, hd⟩ : Fin d) (1 : ℝ)) 0 (by simp)
  have hzero : m ({0} : Set (EucSpace d)) = 0 := by
    exact (withDensity_absolutelyContinuous volume _).null_mono (by simp)
  have hsingle : ∀ᵐ x ∂m, x ≠ 0 := by
    rw [ae_iff]
    simpa only [ne_eq, not_not, Set.ofPred_eq_eq_singleton] using hzero
  rw [ae_all_iff]
  intro i
  have hmp : MeasurePreserving (Function.eval i) (mixtureLaw d n r σ w) m := by
    simpa only [mixtureLaw, m] using (measurePreserving_eval (fun _ : Idx n => m) i)
  exact hmp.quasiMeasurePreserving.ae hsingle

/-- The normalization hypotheses are satisfiable: one component with unit
scale. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧ 0 < (1 : ℕ) :=
  ⟨by norm_num, one_pos, one_pos⟩

end Metastability
end Transformer
