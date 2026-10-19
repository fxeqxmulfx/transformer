module

public import Transformer.Homogenized.Section7_GaussianGramHessian
public import Transformer.Homogenized.Section7_GaussianFeatureCovariance

/-!
# The original covariance-difference trace of the full Gram observable

ArXiv:2604.01978v1, §7, `eq:SIGMA_R`, `eq:gaussian_interp_0`
and `eq:steins_lemma`.
The covariance of the original Gaussian query/key features has already
been proved from their actual independent entry law. Subtracting its
two block Gram matrices gives the exact original covariance difference.

Matching the original block and feature-coordinate indicators reduces
its full coordinate trace to the actual block traces of the original
Gram observable, with exactly the original `σA²` factor.
The proved normalized pointwise block estimate then bounds the complete
original covariance trace by original feature squared norms.
Arbitrary original token configurations and singular covariances are
included. The hypotheses bound only the original Gram difference and
Gram coefficients, not any derivative or covariance trace.
The unit-token singular reference example has a nonzero actual trace.
-/

@[expose] public section

open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Matching original block and feature-coordinate indicators select
exactly the original trace entry. Source: arXiv:2604.01978v1,
§7, the tensor identity used in `eq:steins_lemma`. -/
theorem sum_two_coordinate_delta {ι κ : Type*} [Fintype ι] [Fintype κ]
    [DecidableEq ι] [DecidableEq κ] (F : ι → κ → ℝ) (i : ι) (j : κ) (C : ℝ) :
    (∑ a, ∑ b, (if i = a ∧ j = b then C else 0) * F a b) = C * F i j := by
  calc
    _ = ∑ b, (if i = i ∧ j = b then C else 0) * F i b :=
      Finset.sum_eq_single i (fun a ha hai => by simp [Ne.symm hai]) (by simp)
    _ = _ := by simp only [true_and, ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

/-- The original two-block tensor matrix contracts with an arbitrary
actual coordinate matrix exactly through its matching feature axes.
Source: arXiv:2604.01978v1, §7, `eq:SIGMA_R` and `eq:steins_lemma`. -/
theorem gaussianFeature_block_matrix_trace {d n : ℕ} (C : ℝ) (B : Idx n → Idx n → ℝ)
    (T : GaussianFeatureIdx d n → GaussianFeatureIdx d n → ℝ) :
    (∑ p : GaussianFeatureIdx d n, ∑ r : GaussianFeatureIdx d n,
      (if p.1 = r.1 ∧ p.2.2 = r.2.2 then C * B p.2.1 r.2.1 else 0) * T p r) =
      C * ∑ c : Fin 2, ∑ q : Fin d, ∑ a, ∑ b, B a b * T (c, a, q) (c, b, q) := by
  simp only [Fintype.sum_prod_type]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro c hc
  rw [Finset.sum_comm, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro q hq
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a ha
  rw [Finset.sum_comm]
  have h (b : Idx n) := sum_two_coordinate_delta
    (fun e r => T (c, a, q) (e, b, r)) c q (C * B a b)
  simp_rw [h]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun b hb => by ring

/-- The difference of the proved original feature covariance matrices
is the original variance times the original Gram difference in each block.
Source: arXiv:2604.01978v1, §7, `eq:gaussian_interp_0`. -/
theorem gaussianFeatureCovariance_sub_apply {d n : ℕ} (σA : ℝ≥0) (X Y : Idx n → EucSpace d)
    (p r : GaussianFeatureIdx d n) :
    (gaussianFeatureCovariance σA X p r - gaussianFeatureCovariance σA Y p r) =
      if p.1 = r.1 ∧ p.2.2 = r.2.2 then (σA : ℝ) ^ 2 *
        (inner (𝕜 := ℝ) (X p.2.1) (X r.2.1) - inner (𝕜 := ℝ) (Y p.2.1) (Y r.2.1)) else 0 := by
  unfold gaussianFeatureCovariance
  split_ifs <;> ring

/-- The original full covariance difference contracts exactly through
its original Gram difference and two matching feature-coordinate blocks.
Source: arXiv:2604.01978v1, §7, the reduction in `eq:steins_lemma`. -/
theorem gaussianFeatureCovariance_difference_trace {d n : ℕ} (σA : ℝ≥0) (X Y : Idx n → EucSpace d)
    (T : GaussianFeatureIdx d n → GaussianFeatureIdx d n → ℝ) :
    (∑ p : GaussianFeatureIdx d n, ∑ r : GaussianFeatureIdx d n,
      (gaussianFeatureCovariance σA X p r - gaussianFeatureCovariance σA Y p r) * T p r) =
      (σA : ℝ) ^ 2 * ∑ c : Fin 2, ∑ q : Fin d, ∑ a, ∑ b,
        (inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)) * T (c, a, q) (c, b, q) := by
  simp_rw [gaussianFeatureCovariance_sub_apply]
  exact gaussianFeature_block_matrix_trace ((σA : ℝ) ^ 2)
    (fun a b => inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)) T

/-- The actual full covariance trace of the original Gram observable
has the proved pointwise squared-feature bound, with the exact variance.
Source: arXiv:2604.01978v1, §7, the estimate before
`eq:Lispchitz_bound_softmax`, with normalization before expectation. -/
theorem abs_gaussianGramObservable_covariance_trace_le {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (σA : ℝ≥0) (X Y : Idx n → EucSpace d) (A : Matrix (Idx n) (Idx n) ℝ)
    (hA : ∀ k l, |A k l| ≤ 1) (i j : Idx n)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (ε : ℝ)
    (hR : ∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ ε) :
    |∑ p : GaussianFeatureIdx d n, ∑ r : GaussianFeatureIdx d n,
      (gaussianFeatureCovariance σA X p r - gaussianFeatureCovariance σA Y p r) *
        iteratedFDeriv ℝ 2 (gaussianGramObservable β A i j) z
          ![EuclideanSpace.single p 1, EuclideanSpace.single r 1]| ≤
      (σA : ℝ) ^ 2 * β ^ 2 * ε *
        (20 * (∑ m : Idx n, ∑ q : Fin d, z (1, m, q) ^ 2) +
          10 * ((∑ q : Fin d, z (0, i, q) ^ 2) + ∑ q : Fin d, z (0, j, q) ^ 2)) := by
  rw [gaussianFeatureCovariance_difference_trace, abs_mul, abs_of_nonneg (sq_nonneg (σA : ℝ))]
  have h := abs_gaussianGramObservable_block_trace_le hn β A _ hA i j z ε hR
  exact (mul_le_mul_of_nonneg_left h (sq_nonneg (σA : ℝ))).trans_eq (by ring)

/-- Original unit tokens, a singular constant-token reference, bounded
identity coefficients and nonzero biased features give the genuine full
covariance-difference trace `3/32`, with every bound premise satisfied. -/
example : ∃ (X Y : Idx 2 → EucSpace 2) (z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)),
    (0 : ℕ) < 2 ∧ (∀ a, ‖X a‖ = 1) ∧ (∀ a, ‖Y a‖ = 1) ∧
    (∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ 1) ∧
    (∀ k l : Idx 2, |(1 : Matrix (Idx 2) (Idx 2) ℝ) k l| ≤ 1) ∧
    (∑ p : GaussianFeatureIdx 2 2, ∑ r : GaussianFeatureIdx 2 2,
      (gaussianFeatureCovariance 1 X p r - gaussianFeatureCovariance 1 Y p r) *
        iteratedFDeriv ℝ 2 (gaussianGramObservable (d := 2) 1 (1 : Matrix (Idx 2) (Idx 2) ℝ) 0 0) z
          ![EuclideanSpace.single p 1, EuclideanSpace.single r 1]) = (3 / 32 : ℝ) := by
  let X : Idx 2 → EucSpace 2 := fun a => EuclideanSpace.single a 1
  let Y : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Fin 2) 1
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  have hs : gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] := by
    ext j
    fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]
  have hw0 : Perspective.softmaxWeight ![Real.log 3, 0] (0 : Idx 2) = (3 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  have hw1 : Perspective.softmaxWeight ![Real.log 3, 0] (1 : Idx 2) = (1 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  refine ⟨X, Y, z, by norm_num, fun a => by simp [X], fun a => by simp [Y], ?_, ?_, ?_⟩
  · intro a b
    fin_cases a <;> fin_cases b <;> norm_num [X, Y, EuclideanSpace.inner_single_left, PiLp.single_apply]
  · intro k l
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  · rw [gaussianFeatureCovariance_difference_trace]
    simp only [Fin.sum_univ_two]
    simp [X, Y, EuclideanSpace.inner_single_left]
    simp only [iteratedFDeriv_two_gaussianGramObservable (n := 2) (by norm_num), Fin.sum_univ_two,
      Matrix.one_apply]
    simp only [iteratedFDeriv_two_gaussianAttention_product (n := 2) (by norm_num)]
    simp only [iteratedFDeriv_two_gaussianAttention_query_axes (n := 2) (by norm_num),
      iteratedFDeriv_two_gaussianAttention_key_axes (n := 2) (by norm_num),
      fderiv_gaussianAttention_query_direction (n := 2) (by norm_num),
      fderiv_gaussianAttention_key_axis (n := 2) (by norm_num)]
    simp only [hs]
    simp only [iteratedFDeriv_two_softmaxWeight (m := 2) (by norm_num),
      fderiv_softmaxWeight_factor (m := 2) (by norm_num)]
    norm_num [z, PiLp.single_apply, hw0, hw1]

end Transformer.Homogenized
