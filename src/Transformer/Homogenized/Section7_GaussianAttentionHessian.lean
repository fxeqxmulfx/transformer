/-
# Actual query and key Hessians of the original Gaussian attention

ArXiv:2604.01978v1, §7, the Hessian computations before
`eq:estimate_bound_h` and `eq:estimate_bound_g`.
The genuine second derivative chain rule first retains both terms.
The proved vanishing of the actual bilinear score Hessian within
one feature block then removes the inner-Hessian term.

The query-coordinate Hessian retains its actual token support and
both original key coordinate vectors. The key-coordinate Hessian
retains the two original query coordinates. Both preserve the exact
square of the inverse temperature and the actual softmax Hessian.
No Hessian formula or same-block cancellation is assumed. The biased
original feature example has a nonzero actual key Hessian.
-/

import Transformer.Homogenized.Section7_GaussianScoreDerivatives
import Transformer.Homogenized.Section7_SoftmaxHessian

open scoped BigOperators

namespace Transformer.Homogenized

variable {E F G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [NormedAddCommGroup G] [NormedSpace ℝ G]

/-- The actual second derivative of a composition retains the inner
Hessian term as well as the outer Hessian acting on first derivatives.
Source: arXiv:2604.01978v1, §7, the chain rule used before
`eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem iteratedFDeriv_two_comp_apply (f : E → F) (g : F → G) (hf : ContDiff ℝ 2 f) (hg : ContDiff ℝ 2 g)
    (x v w : E) :
    iteratedFDeriv ℝ 2 (g ∘ f) x ![v, w] =
      iteratedFDeriv ℝ 2 g (f x) ![fderiv ℝ f x v, fderiv ℝ f x w] +
        fderiv ℝ g (f x) (iteratedFDeriv ℝ 2 f x ![v, w]) := by
  have hDf : ContDiff ℝ 1 (fderiv ℝ f) := hf.fderiv_right (by norm_num)
  have hDg : ContDiff ℝ 1 (fderiv ℝ g) := hg.fderiv_right (by norm_num)
  have hDgf : ContDiff ℝ 1 (fderiv ℝ (g ∘ f)) := (hg.comp hf).fderiv_right (by norm_num)
  have heval := (((hDgf.differentiable (by norm_num)) x).hasFDerivAt.clm_apply
    (hasFDerivAt_const w x)).fderiv
  have haxis : fderiv ℝ (fun y => fderiv ℝ (g ∘ f) y w) x v =
      fderiv ℝ (fderiv ℝ (g ∘ f)) x v w := by
    rw [heval]
    simp only [ContinuousLinearMap.comp_zero, zero_add, ContinuousLinearMap.flip_apply]
  have hfun : (fun y => fderiv ℝ (g ∘ f) y w) =
      fun y => fderiv ℝ g (f y) (fderiv ℝ f y w) := by
    funext y
    rw [fderiv_comp y (hg.differentiable (by norm_num) (f y)) (hf.differentiable (by norm_num) y)]
    rfl
  have hgrad := ((hDg.differentiable (by norm_num) (f x)).hasFDerivAt.comp x
    (hf.differentiable (by norm_num) x).hasFDerivAt)
  have hvel := (hDf.differentiable (by norm_num) x).hasFDerivAt.clm_apply (hasFDerivAt_const w x)
  have hp := (hgrad.clm_apply hvel).fderiv
  simp only [Function.comp_def] at hp
  rw [iteratedFDeriv_two_apply]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  rw [← haxis, hfun, hp, iteratedFDeriv_two_apply, iteratedFDeriv_two_apply]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, add_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.comp_zero, zero_add, ContinuousLinearMap.flip_apply]
  exact add_comm _ _

/-- Pulling the two genuine directional scalars from a bilinear
map produces their exact product. Source: arXiv:2604.01978v1,
§7, the two inverse-temperature factors in the Hessian formulas
before `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem multilinear_two_smul_apply (T : ContinuousMultilinearMap ℝ (fun _ : Fin 2 => E) F)
    (a b : ℝ) (v w : E) : T ![a • v, b • w] = (a * b) • T ![v, w] := by
  have h := T.map_smul_univ ![a, b] ![v, w]
  convert h using 1
  · congr 1
    funext k
    fin_cases k <;> rfl
  · simp [Fin.prod_univ_two]

/-- The actual attention Hessian within either original feature block
is the actual softmax Hessian acting on the actual first score derivatives.
The omitted inner Hessian term is proved zero, not assumed.
Source: arXiv:2604.01978v1, §7, the same-block Hessian computations. -/
theorem iteratedFDeriv_two_gaussianAttention_same_block {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i k a b : Idx n)
    (c : Fin 2) (q r : Fin d) :
    iteratedFDeriv ℝ 2
      (fun y => Perspective.softmaxWeight (gaussianFeatureScores β y i) k) z
      ![EuclideanSpace.single (c, a, q) 1, EuclideanSpace.single (c, b, r) 1] =
      iteratedFDeriv ℝ 2 (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
        (gaussianFeatureScores β z i)
        ![fderiv ℝ (fun y => gaussianFeatureScores β y i) z (EuclideanSpace.single (c, a, q) 1),
          fderiv ℝ (fun y => gaussianFeatureScores β y i) z (EuclideanSpace.single (c, b, r) 1)] := by
  have hf : ContDiff ℝ 2 (fun y : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
      gaussianFeatureScores β y i) := (hasTemperateGrowth_gaussianFeatureScores β i).1.of_le (by simp)
  have hg : ContDiff ℝ 2 (fun u : Idx n → ℝ => Perspective.softmaxWeight u k) :=
    (contDiff_softmaxWeight hn k).of_le (by simp)
  have h := iteratedFDeriv_two_comp_apply _ _ hf hg z (EuclideanSpace.single (c, a, q) 1)
    (EuclideanSpace.single (c, b, r) 1)
  simpa only [Function.comp_def, iteratedFDeriv_two_gaussianFeatureScores_same_block,
    map_zero, add_zero] using h

/-- The actual key-coordinate attention Hessian keeps its original
query coordinates and the original softmax score-coordinate Hessian.
Source: arXiv:2604.01978v1, §7, the key-coordinate computation
before `eq:estimate_bound_g`. -/
theorem iteratedFDeriv_two_gaussianAttention_key_axes {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i k a b : Idx n) (q r : Fin d) :
    iteratedFDeriv ℝ 2
      (fun y => Perspective.softmaxWeight (gaussianFeatureScores β y i) k) z
      ![EuclideanSpace.single (1, a, q) 1, EuclideanSpace.single (1, b, r) 1] =
      (β ^ 2 * z (0, i, q) * z (0, i, r)) *
        iteratedFDeriv ℝ 2 (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
          (gaussianFeatureScores β z i) ![Pi.single a 1, Pi.single b 1] := by
  rw [iteratedFDeriv_two_gaussianAttention_same_block hn, fderiv_gaussianFeatureScores_key_axis,
    fderiv_gaussianFeatureScores_key_axis]
  have hs (a : Idx n) (t : ℝ) : (Pi.single a t : Idx n → ℝ) = t • Pi.single a 1 := by
    simpa only [smul_eq_mul, mul_one] using
      Pi.single_smul (f := fun _ : Idx n => ℝ) a t (1 : ℝ)
  rw [hs a (β * z (0, i, q)), hs b (β * z (0, i, r))]
  rw [multilinear_two_smul_apply]
  simp only [smul_eq_mul]
  ring

/-- The actual query-coordinate attention Hessian is supported only
on its own query token and retains both original key coordinate vectors.
Source: arXiv:2604.01978v1, §7, the query-coordinate computation
before `eq:estimate_bound_h`. -/
theorem iteratedFDeriv_two_gaussianAttention_query_axes {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i k a b : Idx n) (q r : Fin d) :
    iteratedFDeriv ℝ 2
      (fun y => Perspective.softmaxWeight (gaussianFeatureScores β y i) k) z
      ![EuclideanSpace.single (0, a, q) 1, EuclideanSpace.single (0, b, r) 1] =
      if a = i ∧ b = i then β ^ 2 *
        iteratedFDeriv ℝ 2 (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
          (gaussianFeatureScores β z i) ![(fun l => z (1, l, q)), (fun l => z (1, l, r))] else 0 := by
  rw [iteratedFDeriv_two_gaussianAttention_same_block hn, fderiv_gaussianFeatureScores_query_axis,
    fderiv_gaussianFeatureScores_query_axis]
  by_cases hai : a = i
  · by_cases hbi : b = i
    · simp only [hai, hbi, ite_true, and_self]
      change (iteratedFDeriv ℝ 2 (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
        (gaussianFeatureScores β z i)) ![β • (fun l => z (1, l, q)), β • (fun l => z (1, l, r))] = _
      rw [multilinear_two_smul_apply]
      simp only [smul_eq_mul, pow_two]
    · simp only [hai, hbi, ite_true, ite_false, true_and]
      exact (iteratedFDeriv ℝ 2 (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
        (gaussianFeatureScores β z i)).map_coord_zero (1 : Fin 2) rfl
  · simp only [hai, ite_false, false_and]
    exact (iteratedFDeriv ℝ 2 (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
      (gaussianFeatureScores β z i)).map_coord_zero (0 : Fin 2) rfl

/-- Nonzero original query and key features give the biased score row
`[log 3, 0]`, witness all smoothness and positivity hypotheses,
and have a genuine key-coordinate Hessian equal to `-3/32`. -/
example : ∃ z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2), (0 : ℕ) < 2 ∧
    ContDiff ℝ 2 (fun y : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) =>
      gaussianFeatureScores 1 y (0 : Idx 2)) ∧
    ContDiff ℝ 2 (fun u : Idx 2 → ℝ => Perspective.softmaxWeight u 0) ∧
    iteratedFDeriv ℝ 2
      (fun y => Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0) z
      ![EuclideanSpace.single (1, 0, 0) 1, EuclideanSpace.single (1, 0, 0) 1] = -(3 / 32 : ℝ) := by
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  have hs : gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] := by
    ext j
    fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]
  have hw : Perspective.softmaxWeight ![Real.log 3, 0] (0 : Idx 2) = (3 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  refine ⟨z, by norm_num, (hasTemperateGrowth_gaussianFeatureScores _ _).1.of_le (by simp),
    (contDiff_softmaxWeight (by norm_num) _).of_le (by simp), ?_⟩
  rw [iteratedFDeriv_two_gaussianAttention_key_axes (by norm_num), hs, iteratedFDeriv_two_softmaxWeight (by norm_num), hw]
  norm_num [z, PiLp.single_apply]

end Transformer.Homogenized

