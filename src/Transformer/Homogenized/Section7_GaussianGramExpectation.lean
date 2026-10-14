/-
# The actual Gram observable expectation bound

ArXiv:2604.01978v1, §7, eq:gaussian_interp_0 through
eq:Lispchitz_bound_softmax. The original full covariance trace has a
proved pointwise quadratic envelope. Its integral is computed from
the actual Gaussian interpolation moments, with no independence
between random attention weights and squared features assumed.

The proved finite-index polynomial Gaussian interpolation theorem
then gives the original expectation difference with constant 20 and
the original dimension, variance, temperature and token-count powers.
The hypotheses specify the actual Gaussian laws, means, covariances,
unit original tokens and entrywise Gram difference. No Hessian bound
or expectation-difference estimate is assumed.
-/

import Transformer.Homogenized.Section7_GaussianBlendMoments
import Transformer.Homogenized.Section7_GaussianGramCovarianceTrace
import Transformer.Homogenized.Section7_GaussianIndexedInterpolation

open MeasureTheory ProbabilityTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Integrating the proved actual Hessian bound gives an explicit uniform
bound on the actual covariance-trace expectation. Source: arXiv:2604.01978v1,
§7, the moment step in eq:Lispchitz_bound_softmax. -/
theorem abs_integral_gaussianGramObservable_covariance_trace_le {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) (Z W : Ω → EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (hZ : HasGaussianLaw Z P) (hW : HasGaussianLaw W P) (hind : IndepFun Z W P)
    (hmZ : ∀ p, ∫ ω, Z ω p ∂P = 0) (hmW : ∀ p, ∫ ω, W ω p ∂P = 0)
    (σA : ℝ≥0) (hZZ : ∀ p, ∫ ω, Z ω p ^ 2 ∂P = (σA : ℝ) ^ 2)
    (hWW : ∀ p, ∫ ω, W ω p ^ 2 ∂P = (σA : ℝ) ^ 2)
    (hn : 0 < n) (β : ℝ) (X Y : Idx n → EucSpace d)
    (A : Matrix (Idx n) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1) (i j : Idx n)
    (ε : ℝ) (hR : ∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ ε)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    |(1 / 2 : ℝ) * ∫ ω, ∑ p : GaussianFeatureIdx d n, ∑ r : GaussianFeatureIdx d n,
      (gaussianFeatureCovariance σA X p r - gaussianFeatureCovariance σA Y p r) *
        iteratedFDeriv ℝ 2 (gaussianGramObservable β A i j)
          (Real.sqrt t • Z ω + Real.sqrt (1 - t) • W ω)
          ![EuclideanSpace.single p 1, EuclideanSpace.single r 1] ∂P| ≤
      20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ) * ε := by
  let U (ω : Ω) := Real.sqrt t • Z ω + Real.sqrt (1 - t) • W ω
  have hi (p : GaussianFeatureIdx d n) : Integrable (fun ω => U ω p ^ 2) P :=
    gaussian_interp_coordinate_sq_integrable P Z W hZ hW t p
  have hG : Integrable (fun ω => ∑ m : Idx n, ∑ q : Fin d, U ω (1, m, q) ^ 2) P :=
    integrable_finsetSum _ fun m hm => integrable_finsetSum _ fun q hq => hi (1, m, q)
  have hHi : Integrable (fun ω => ∑ q : Fin d, U ω (0, i, q) ^ 2) P :=
    integrable_finsetSum _ fun q hq => hi (0, i, q)
  have hHj : Integrable (fun ω => ∑ q : Fin d, U ω (0, j, q) ^ 2) P :=
    integrable_finsetSum _ fun q hq => hi (0, j, q)
  have hHij : Integrable (fun ω => (∑ q : Fin d, U ω (0, i, q) ^ 2) +
      ∑ q : Fin d, U ω (0, j, q) ^ 2) P := hHi.add hHj
  have hEnv : Integrable (fun ω => (σA : ℝ) ^ 2 * β ^ 2 * ε *
      (20 * (∑ m : Idx n, ∑ q : Fin d, U ω (1, m, q) ^ 2) +
        10 * ((∑ q : Fin d, U ω (0, i, q) ^ 2) + ∑ q : Fin d, U ω (0, j, q) ^ 2))) P := by
    exact ((hG.const_mul 20).add (hHij.const_mul 10)).const_mul _
  have hb := norm_integral_le_of_norm_le hEnv
    (ae_of_all P fun ω => by
      rw [Real.norm_eq_abs]
      exact abs_gaussianGramObservable_covariance_trace_le hn β σA X Y A hA i j (U ω) ε hR)
  rw [Real.norm_eq_abs, integral_const_mul,
    integral_add (hG.const_mul 20) (hHij.const_mul 10),
    integral_const_mul, integral_const_mul, integral_add hHi hHj] at hb
  have hgm := gaussian_interp_block_energy P Z W hZ hW hind hmZ hmW
    ((σA : ℝ) ^ 2) hZZ hWW t ht 1
  have him := gaussian_interp_feature_energy P Z W hZ hW hind hmZ hmW
    ((σA : ℝ) ^ 2) hZZ hWW t ht 0 i
  have hjm := gaussian_interp_feature_energy P Z W hZ hW hind hmZ hmW
    ((σA : ℝ) ^ 2) hZZ hWW t ht 0 j
  change (∫ ω, ∑ m : Idx n, ∑ q : Fin d, U ω (1, m, q) ^ 2 ∂P) = _ at hgm
  change (∫ ω, ∑ q : Fin d, U ω (0, i, q) ^ 2 ∂P) = _ at him
  change (∫ ω, ∑ q : Fin d, U ω (0, j, q) ^ 2 ∂P) = _ at hjm
  rw [hgm, him, hjm] at hb
  have hε : 0 ≤ ε := (abs_nonneg _).trans (hR i i)
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hN : 20 * (n : ℝ) + 20 ≤ 40 * (n : ℝ) := by linarith
  have hc : 0 ≤ (1 / 2 : ℝ) * ((σA : ℝ) ^ 2 * β ^ 2 * ε * (d : ℝ) * (σA : ℝ) ^ 2) := by positivity
  calc
    _ = (1 / 2 : ℝ) * |∫ ω, ∑ p : GaussianFeatureIdx d n, ∑ r : GaussianFeatureIdx d n,
        (gaussianFeatureCovariance σA X p r - gaussianFeatureCovariance σA Y p r) *
          iteratedFDeriv ℝ 2 (gaussianGramObservable β A i j) (U ω)
            ![EuclideanSpace.single p 1, EuclideanSpace.single r 1] ∂P| := by
      rw [abs_mul]
      norm_num [U]
    _ ≤ (1 / 2 : ℝ) * ((σA : ℝ) ^ 2 * β ^ 2 * ε *
        (20 * ((n : ℝ) * (d : ℝ) * (σA : ℝ) ^ 2) +
          10 * ((d : ℝ) * (σA : ℝ) ^ 2 + (d : ℝ) * (σA : ℝ) ^ 2))) :=
      mul_le_mul_of_nonneg_left hb (by norm_num)
    _ = ((1 / 2 : ℝ) * ((σA : ℝ) ^ 2 * β ^ 2 * ε * (d : ℝ) * (σA : ℝ) ^ 2)) *
        (20 * (n : ℝ) + 20) := by ring
    _ ≤ ((1 / 2 : ℝ) * ((σA : ℝ) ^ 2 * β ^ 2 * ε * (d : ℝ) * (σA : ℝ) ^ 2)) *
        (40 * (n : ℝ)) := mul_le_mul_of_nonneg_left hN hc
    _ = _ := by ring

/-- Original unit tokens give the common diagonal feature variance.
Source: arXiv:2604.01978v1, §7, the unit diagonal in eq:SIGMA_R. -/
theorem gaussianFeatureCovariance_self_of_unit {d n : ℕ} (σA : ℝ≥0) (X : Idx n → EucSpace d)
    (hX : ∀ a, ‖X a‖ = 1) (p : GaussianFeatureIdx d n) :
    gaussianFeatureCovariance σA X p p = (σA : ℝ) ^ 2 := by
  simp [gaussianFeatureCovariance, hX]

/-- The full normalized overlap observable has the source's Gram Lipschitz
scaling, with constant 20, for its actual Gaussian feature laws.
Source: arXiv:2604.01978v1, §7, eq:Lispchitz_bound_softmax. The random
softmax weights are normalized pointwise before expectation, correcting
the source's placement of random weights outside the expectation. -/
theorem abs_integral_gaussianGramObservable_sub_le {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) (Z W : Ω → EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (hZ : HasGaussianLaw Z P) (hW : HasGaussianLaw W P) (hind : IndepFun Z W P)
    (hmZ : ∀ p, ∫ ω, Z ω p ∂P = 0) (hmW : ∀ p, ∫ ω, W ω p ∂P = 0)
    (σA : ℝ≥0) (X Y : Idx n → EucSpace d) (hX : ∀ a, ‖X a‖ = 1) (hY : ∀ a, ‖Y a‖ = 1)
    (hZZ : ∀ p r, ∫ ω, Z ω p * Z ω r ∂P = gaussianFeatureCovariance σA X p r)
    (hWW : ∀ p r, ∫ ω, W ω p * W ω r ∂P = gaussianFeatureCovariance σA Y p r)
    (hn : 0 < n) (β : ℝ) (A : Matrix (Idx n) (Idx n) ℝ)
    (hA : ∀ k l, |A k l| ≤ 1) (i j : Idx n) (ε : ℝ)
    (hR : ∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ ε) :
    |(∫ ω, gaussianGramObservable β A i j (Z ω) ∂P) -
      ∫ ω, gaussianGramObservable β A i j (W ω) ∂P| ≤
        20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ) * ε := by
  obtain ⟨hφ, C, p, _, hb⟩ := gaussianGramObservable_polynomial_growth (d := d) hn β A i j
  apply gaussian_interp_indexed_expectation_difference_le P Z W hZ hW
    (gaussianFeatureCovariance σA X) (gaussianFeatureCovariance σA Y) hmZ hmW hZZ hWW hind
    (gaussianGramObservable β A i j) hφ C p hb
  intro t ht
  apply abs_integral_gaussianGramObservable_covariance_trace_le P Z W hZ hW hind hmZ hmW σA
    (fun p => by simpa only [pow_two] using (hZZ p p).trans (gaussianFeatureCovariance_self_of_unit σA X hX p))
    (fun p => by simpa only [pow_two] using (hWW p p).trans (gaussianFeatureCovariance_self_of_unit σA Y hY p))
    hn β X Y A hA i j ε hR t ht
/-- Different unit-token Gram matrices, bounded identity coefficients,
and independent singular zero Gaussian laws satisfy all hypotheses.
The preceding pointwise and moment modules also exhibit positive variances. -/
example : ∃ X Y : Idx 2 → EucSpace 2,
    (0 : ℕ) < 2 ∧ (∀ a, ‖X a‖ = 1) ∧ (∀ a, ‖Y a‖ = 1) ∧
    (∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ 1) ∧
    (∀ k l : Idx 2, |(1 : Matrix (Idx 2) (Idx 2) ℝ) k l| ≤ 1) ∧
    HasGaussianLaw (fun _ : Unit => (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2))) (Measure.dirac ()) ∧
    IndepFun (fun _ : Unit => (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)))
      (fun _ : Unit => (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2))) (Measure.dirac ()) ∧
    (∀ p : GaussianFeatureIdx 2 2, ∫ _ω : Unit, (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) p
      ∂Measure.dirac () = 0) ∧
    (∀ p : GaussianFeatureIdx 2 2, ∫ _ω : Unit, (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) p ^ 2
      ∂Measure.dirac () = (0 : ℝ≥0)) ∧
    (∀ p r : GaussianFeatureIdx 2 2, ∫ _ω : Unit,
      (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) p * (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) r
      ∂Measure.dirac () = gaussianFeatureCovariance 0 X p r) ∧
    (∀ p r : GaussianFeatureIdx 2 2, ∫ _ω : Unit,
      (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) p * (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) r
      ∂Measure.dirac () = gaussianFeatureCovariance 0 Y p r) ∧
    ((1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1) := by
  let X : Idx 2 → EucSpace 2 := fun a => EuclideanSpace.single a 1
  let Y : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Fin 2) 1
  refine ⟨X, Y, by norm_num, fun a => by simp [X], fun a => by simp [Y], ?_, ?_, ?_,
    indepFun_const_left _ _, fun p => by simp, fun p => by simp,
    fun p r => by simp [gaussianFeatureCovariance], fun p r => by simp [gaussianFeatureCovariance],
    by norm_num⟩
  · intro a b
    fin_cases a <;> fin_cases b <;> norm_num [X, Y, EuclideanSpace.inner_single_left, PiLp.single_apply]
  · intro k l
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  · refine ⟨measurable_const.aemeasurable, ?_⟩
    rw [Measure.map_const]
    simp
    infer_instance

end Transformer.Homogenized
