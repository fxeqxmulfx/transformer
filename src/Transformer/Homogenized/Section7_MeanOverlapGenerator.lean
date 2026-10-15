/-
# The actual Gaussian generator of the original empirical mean

ArXiv:2604.01978v1, §7, `eq:m_decomp_clean` and
`eq:F_def_clean` in the proof of `thm:clustering_small_beta`.
The actual generator of the original mean observable is the original
normalized average of the actual pair-overlap generators.

All head expectations of the original mean sphere Hessian are proved
integrable from the genuine Gaussian projected moments. The generator
identity therefore follows from finite sums and actual Bochner
linearity, including the full Riemannian correction.

At the source's standard value variance `σV²=1/d`, the generator is
exactly `meanOverlapDrift`. At inverse temperature zero its logistic
mismatch is at most `2/d`, by the original configuration formulas.

The actual finite-horizon adapted continuous sphere martingale problem
also supplies the compensated mean test directly. No pathwise noise
estimate or drift decomposition is assumed as a separate input.
-/

import Transformer.Homogenized.Section7_MeanOverlapCalculus
import Transformer.Homogenized.Section7_ExponentialGenerator
import Transformer.Homogenized.Section7_SphereMartingale

open Real MeasureTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Integrability of the actual mean sphere Hessian follows from
integrability of all original pair Hessians. Source: arXiv:2604.01978v1,
§7, the averaging in `eq:m_decomp_clean`. -/
theorem integrable_sphHess_meanOverlap {d n : ℕ} (ρ : Measure (HeadParam d))
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d) (X : Idx n → EucSpace d)
    (hH : ∀ i j : Idx n, Integrable (fun θ => sphHess
      (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X (G θ X)) ρ) :
    Integrable (fun θ => sphHess (meanOverlap (d := d) (n := n)) X (G θ X)) ρ := by
  simp_rw [sphHess_meanOverlap]
  exact (integrable_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj => hH i j).const_mul _

/-- The original Gaussian mean sphere Hessian is genuinely integrable.
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`,
using the proved actual projected Gaussian moments. -/
theorem integrable_sphHess_meanOverlap_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    Integrable (fun θ => sphHess (meanOverlap (d := d) (n := n)) X (noiseField β 1 1 ρ θ X)) ρ :=
  integrable_sphHess_meanOverlap ρ (noiseField β 1 1 ρ) X
    (fun i j => integrable_sphHess_overlap_gaussian β hρ X hX i j)

/-- **The actual generator commutes with the original mean average.**
The only head-integrability premises concern the actual pair Hessians.
Source: arXiv:2604.01978v1, §7, `eq:F_def_clean`. -/
theorem sphGenerator_meanOverlap {d n : ℕ} (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d) (X : Idx n → EucSpace d)
    (hH : ∀ i j : Idx n, Integrable (fun θ => sphHess
      (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X (G θ X)) ρ) :
    sphGenerator ρ B G (meanOverlap (d := d) (n := n)) X =
      ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        sphGenerator ρ B G (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X := by
  have hInt : (∫ θ, ∑ i : Idx n, ∑ j : Idx n,
      sphHess (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X (G θ X) ∂ρ) =
      ∑ i : Idx n, ∑ j : Idx n, ∫ θ,
        sphHess (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X (G θ X) ∂ρ := by
    rw [integral_finsetSum _ (fun i hi => integrable_finsetSum _ fun j hj => hH i j)]
    apply Finset.sum_congr rfl
    intro i hi
    exact integral_finsetSum _ fun j hj => hH i j
  have hfd (i j : Idx n) := (hasFDerivAt_overlap i j X).fderiv
  simp only [sphGenerator, fderiv_meanOverlap, meanOverlapForm_apply, hfd]
  simp_rw [sphHess_meanOverlap]
  rw [integral_const_mul, hInt]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

/-- At the source's actual Gaussian normalization, the mean generator
is its genuine averaged overlap drift. Source: arXiv:2604.01978v1,
§7, `eq:m_decomp_clean` and `eq:F_def_clean`. -/
theorem sphGenerator_meanOverlap_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (meanOverlap (d := d) (n := n)) X =
      meanOverlapDrift β ρ X := by
  rw [sphGenerator_meanOverlap ρ (bField β ρ) (noiseField β 1 1 ρ) X
    (fun i j => integrable_sphHess_overlap_gaussian β hρ X hX i j)]
  simp only [ito_formula_overlap β σV σA ρ hρ hσ X hX, meanOverlapDrift]

/-- **The actual original zero-temperature generator has logistic mismatch `2/d`.**
Every covariance and integrability step is proved from the original Gaussian law.
Source: arXiv:2604.01978v1, §7, part (a) of `eq:drift_mismatch_target_clean`. -/
theorem abs_sphGenerator_meanOverlap_zero_sub_logistic_le {d n : ℕ} (hd : 0 < d)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    |sphGenerator ρ (bField 0 ρ) (noiseField 0 1 1 ρ) (meanOverlap (d := d) (n := n)) X -
      meanOverlap X * (1 - meanOverlap X)| ≤ 2 / (d : ℝ) := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  rw [sphGenerator_meanOverlap_gaussian 0 hρ hσ X hX]
  exact abs_meanOverlapDrift_zero_sub_logistic_le hd ρ X hX

/-- The actual finite-dimensional zero-temperature generator lies below
its logistic approximation. This keeps the sign of the proved correction.
Source: arXiv:2604.01978v1, §7, the exact signed correction in part (a)
of the proof of `eq:drift_mismatch_target_clean`. -/
theorem sphGenerator_meanOverlap_zero_le_logistic {d n : ℕ} (hd : 0 < d)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    sphGenerator ρ (bField 0 ρ) (noiseField 0 1 1 ρ) (meanOverlap (d := d) (n := n)) X ≤
      meanOverlap X * (1 - meanOverlap X) := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  rw [sphGenerator_meanOverlap_gaussian 0 hρ hσ X hX]
  exact meanOverlapDrift_zero_le_logistic hd ρ X hX

/-- The full original adapted continuous martingale problem supplies
the actual compensated mean observable. Source: arXiv:2604.01978v1,
§7, `eq:m_decomp_clean`, and Appendix, `def: ito.formula`. -/
theorem IsSphereMartingaleSolution.meanOverlap {d n : ℕ} {ρ : Measure (HeadParam d)}
    {B : (Idx n → EucSpace d) → Idx n → EucSpace d}
    {G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d}
    {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ m}
    {T : ℝ} {X : ℝ → Ω → (Idx n → EucSpace d)}
    (h : IsSphereMartingaleSolution ρ B G P ℱ T X) :
    IsMartingaleOn T ℱ P (fun t ω => meanOverlap (X t ω) - meanOverlap (X 0 ω) -
      ∫ s in (0 : ℝ)..t, sphGenerator ρ B G (meanOverlap (d := d) (n := n)) (X s ω)) :=
  h.2.2.2 _ contDiff_meanOverlap

/-- Positive original Gaussian value variance, the source's exact normalization
and actual orthogonal unit tokens jointly satisfy every Gaussian generator premise.
The original pair Hessians are also integrable for the general averaging theorem. -/
example : ∃ X : Idx 2 → EucSpace 3,
    (0 : ℕ) < 3 ∧ IsGaussianHeadLaw 3 (stdSigmaV 3) 1 (gaussHeadLaw 3 (stdSigmaV 3) 1) ∧
    (stdSigmaV 3 : ℝ) ^ 2 = 1 / (3 : ℝ) ∧ (∀ i, ‖X i‖ = 1) ∧
    (∀ i j : Idx 2, Integrable (fun θ => sphHess
      (fun Y : Idx 2 → EucSpace 3 => inner ℝ (Y i) (Y j)) X
        (noiseField 0 1 1 (gaussHeadLaw 3 (stdSigmaV 3) 1) θ X)) (gaussHeadLaw 3 (stdSigmaV 3) 1)) := by
  let X : Idx 2 → EucSpace 3 := ![EuclideanSpace.single (0 : Fin 3) 1, EuclideanSpace.single (1 : Fin 3) 1]
  have hX : ∀ i, ‖X i‖ = 1 := by intro i; fin_cases i <;> simp [X]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 3 (stdSigmaV 3) 1
  exact ⟨X, by decide, hρ, stdSigmaV_sq 3, hX,
    fun i j => integrable_sphHess_overlap_gaussian 0 hρ X hX i j⟩

/-- The mean martingale-test premise is realized by the actual full-test
solution with orthogonal unit tokens and zero original head fields. -/
example : ∃ ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit),
    IsSphereMartingaleSolution (Measure.dirac (0 : HeadParam 2))
      (bField 1 (Measure.dirac 0)) (noiseField 1 1 1 (Measure.dirac 0)) (Measure.dirac ()) ℱ 1
      (fun _ _ => (fun i : Idx 2 => EuclideanSpace.single i (1 : ℝ))) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  exact ⟨ℱ, isSphereMartingaleSolution_dirac_zero (Measure.dirac ()) ℱ 1 1 1 1 _ (fun i => by simp)⟩

end Transformer.Homogenized
