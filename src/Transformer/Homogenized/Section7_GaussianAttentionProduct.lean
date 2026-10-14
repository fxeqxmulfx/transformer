/-
# Actual first derivatives and product Hessians of Gaussian attention

ArXiv:2604.01978v1, §7, the two attention factors differentiated
before `eq:estimate_bound_h` and `eq:estimate_bound_g`.
The original query and key axis derivatives keep their genuine token
support, their original feature coordinates and their actual softmax
score-coordinate derivatives.

The actual product Hessian retains both factor Hessians and both mixed
first derivatives. Its attention specialization applies to arbitrary
pairs of rows and columns, including equal rows and equal factors.
Those self-pairs occur in the original diagonal overlaps used in Gram
drift. The nonzero biased self-pair example witnesses the extra terms:
its actual key Hessian is `-9/128`, while the single weight's is `-3/32`.
-/

import Transformer.Homogenized.Section7_GaussianAttentionHessian

open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual product Hessian retains both factor Hessians and
both mixed first-derivative terms, including equal factors.
Source: arXiv:2604.01978v1, §7, the product differentiation
before `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem iteratedFDeriv_two_mul_apply {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (f g : E → ℝ) (hf : ContDiff ℝ 2 f) (hg : ContDiff ℝ 2 g) (x v w : E) :
    iteratedFDeriv ℝ 2 (fun y => f y * g y) x ![v, w] =
      g x * iteratedFDeriv ℝ 2 f x ![v, w] + f x * iteratedFDeriv ℝ 2 g x ![v, w] +
        fderiv ℝ f x v * fderiv ℝ g x w + fderiv ℝ f x w * fderiv ℝ g x v := by
  let h := fun y => f y * g y
  have hDf : ContDiff ℝ 1 (fderiv ℝ f) := hf.fderiv_right (by norm_num)
  have hDg : ContDiff ℝ 1 (fderiv ℝ g) := hg.fderiv_right (by norm_num)
  have hDh : ContDiff ℝ 1 (fderiv ℝ h) := (hf.mul hg).fderiv_right (by norm_num)
  have heval := (((hDh.differentiable (by norm_num)) x).hasFDerivAt.clm_apply
    (hasFDerivAt_const w x)).fderiv
  have haxis : fderiv ℝ (fun y => fderiv ℝ h y w) x v = fderiv ℝ (fderiv ℝ h) x v w := by
    rw [heval]
    simp only [ContinuousLinearMap.comp_zero, zero_add, ContinuousLinearMap.flip_apply]
  have hfun : (fun y => fderiv ℝ h y w) =
      fun y => g y * fderiv ℝ f y w + f y * fderiv ℝ g y w := by
    funext y
    rw [show h = fun y => f y * g y from rfl,
      ((hf.differentiable (by norm_num) y).hasFDerivAt.fun_mul
        (hg.differentiable (by norm_num) y).hasFDerivAt).fderiv]
    simp only [add_apply, smul_apply, smul_eq_mul]
    ring
  have hpf := (hDf.differentiable (by norm_num) x).hasFDerivAt.clm_apply (hasFDerivAt_const w x)
  have hpg := (hDg.differentiable (by norm_num) x).hasFDerivAt.clm_apply (hasFDerivAt_const w x)
  have hp := (((hg.differentiable (by norm_num) x).hasFDerivAt.fun_mul hpf).fun_add
    ((hf.differentiable (by norm_num) x).hasFDerivAt.fun_mul hpg)).fderiv
  rw [iteratedFDeriv_two_apply]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  rw [← haxis, hfun, hp, iteratedFDeriv_two_apply, iteratedFDeriv_two_apply]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, add_apply, smul_apply, smul_eq_mul,
    ContinuousLinearMap.comp_zero, zero_add, ContinuousLinearMap.flip_apply]
  ring

/-- The actual original attention derivative is the genuine original
softmax derivative acting on the actual original score derivative.
Source: arXiv:2604.01978v1, §7, the chain rule used in
`eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem fderiv_gaussianAttention_apply {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z v : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i k : Idx n) :
    fderiv ℝ (fun y => Perspective.softmaxWeight (gaussianFeatureScores β y i) k) z v =
      fderiv ℝ (fun u : Idx n → ℝ => Perspective.softmaxWeight u k) (gaussianFeatureScores β z i)
        (fderiv ℝ (fun y => gaussianFeatureScores β y i) z v) := by
  have h := fderiv_comp z ((contDiff_softmaxWeight hn k).differentiable (by simp) _)
    (hasFDerivAt_gaussianFeatureScores β z i).differentiableAt
  change fderiv ℝ ((fun u : Idx n → ℝ => Perspective.softmaxWeight u k) ∘
    (fun y => gaussianFeatureScores β y i)) z v = _
  rw [h]
  rfl

/-- The actual key-coordinate attention derivative retains its
original query coordinate and original score-coordinate derivative.
Source: arXiv:2604.01978v1, §7, the first-derivative terms in
`eq:estimate_bound_g`. -/
theorem fderiv_gaussianAttention_key_axis {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i k a : Idx n) (q : Fin d) :
    fderiv ℝ (fun y => Perspective.softmaxWeight (gaussianFeatureScores β y i) k) z
      (EuclideanSpace.single (1, a, q) 1) =
      β * z (0, i, q) * fderiv ℝ (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
        (gaussianFeatureScores β z i) (Pi.single a 1) := by
  rw [fderiv_gaussianAttention_apply hn, fderiv_gaussianFeatureScores_key_axis]
  have hs : (Pi.single a (β * z (0, i, q)) : Idx n → ℝ) =
      (β * z (0, i, q)) • Pi.single a 1 := by
    simpa only [smul_eq_mul, mul_one] using
      Pi.single_smul (f := fun _ : Idx n => ℝ) a (β * z (0, i, q)) (1 : ℝ)
  rw [hs, map_smul]
  rfl

/-- The actual query-coordinate attention derivative has its genuine
token support and is the original key-weighted score derivative sum.
Source: arXiv:2604.01978v1, §7, the first-derivative terms in
`eq:estimate_bound_h`. -/
theorem fderiv_gaussianAttention_query_axis {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i k a : Idx n) (q : Fin d) :
    fderiv ℝ (fun y => Perspective.softmaxWeight (gaussianFeatureScores β y i) k) z
      (EuclideanSpace.single (0, a, q) 1) =
      if a = i then β * ∑ l : Idx n, z (1, l, q) *
        fderiv ℝ (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
          (gaussianFeatureScores β z i) (Pi.single l 1) else 0 := by
  rw [fderiv_gaussianAttention_apply hn, fderiv_gaussianFeatureScores_query_axis]
  by_cases hai : a = i
  · simp only [hai, ite_true]
    have hv : (fun l : Idx n => β * z (1, l, q)) =
        ∑ l : Idx n, (β * z (1, l, q)) • Pi.single l 1 := by
      ext l
      simp [Pi.single_apply]
    rw [hv, map_sum]
    simp only [map_smul, smul_eq_mul, mul_assoc, Finset.mul_sum]
  · simp [hai]

/-- The original attention product has the actual four-term Hessian,
with every term derived from the original functions' proved smoothness.
Source: arXiv:2604.01978v1, §7, the two attention factors in its Hessian
calculation. This formula also includes the self-pairs needed for `s_{μ_X}(x_i)`. -/
theorem iteratedFDeriv_two_gaussianAttention_product {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z v w : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l : Idx n) :
    let f := fun y : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k
    let g := fun y : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
      Perspective.softmaxWeight (gaussianFeatureScores β y j) l
    iteratedFDeriv ℝ 2 (fun y => f y * g y) z ![v, w] =
      g z * iteratedFDeriv ℝ 2 f z ![v, w] + f z * iteratedFDeriv ℝ 2 g z ![v, w] +
        fderiv ℝ f z v * fderiv ℝ g z w + fderiv ℝ f z w * fderiv ℝ g z v := by
  have hc (a b : Idx n) : ContDiff ℝ 2 (fun y : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
      Perspective.softmaxWeight (gaussianFeatureScores β y a) b) :=
    ((hasTemperateGrowth_softmaxWeight hn b).comp
      (hasTemperateGrowth_gaussianFeatureScores β a)).1.of_le (by simp)
  exact iteratedFDeriv_two_mul_apply _ _ (hc i k) (hc j l) z v w

/-- Actual nonzero original features give `[log 3, 0]` scores,
satisfy every smoothness and positive-score-count hypothesis,
and give a nonzero actual self-pair key Hessian of `-9/128`. -/
example : ∃ z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2), (0 : ℕ) < 2 ∧
    ContDiff ℝ 2 (fun y : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) =>
      Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0) ∧
    iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0 *
        Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0) z
      ![EuclideanSpace.single (1, 0, 0) 1, EuclideanSpace.single (1, 0, 0) 1] = -(9 / 128 : ℝ) := by
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  have hs : gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] := by
    ext j
    fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]
  have hw : Perspective.softmaxWeight ![Real.log 3, 0] (0 : Idx 2) = (3 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  refine ⟨z, by norm_num,
    ((hasTemperateGrowth_softmaxWeight (by norm_num) (0 : Idx 2)).comp
      (hasTemperateGrowth_gaussianFeatureScores _ _)).1.of_le (by simp), ?_⟩
  rw [iteratedFDeriv_two_gaussianAttention_product (by norm_num)]
  dsimp only
  rw [iteratedFDeriv_two_gaussianAttention_key_axes (by norm_num), fderiv_gaussianAttention_key_axis (by norm_num),
    hs, iteratedFDeriv_two_softmaxWeight (by norm_num), fderiv_softmaxWeight_factor (by norm_num), hw]
  norm_num [z, PiLp.single_apply]

end Transformer.Homogenized

