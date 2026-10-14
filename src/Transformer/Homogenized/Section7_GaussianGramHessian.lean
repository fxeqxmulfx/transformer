/-
# The actual full Gram observable Hessian and its original block trace

ArXiv:2604.01978v1, §7, the original overlap sum after
`eq:SIGMA_R` and its trace in `eq:steins_lemma`.
The original Gram observable is a finite weighted sum of actual
attention products. Its genuine Hessian is obtained using their proved
smoothness and the actual finite derivative sum rules.

Exchanging that output sum with the actual feature-coordinate trace
identifies the full observable trace with the proved original pair
traces. The normalized pointwise bound then retains only the original
feature squared norms, with query constant twenty and key constant ten.
The proof keeps actual derivatives and original indices throughout;
no Hessian expansion or trace bound is an input.
-/

import Transformer.Homogenized.Section7_GaussianAttentionTraceBounds

open scoped BigOperators

namespace Transformer.Homogenized

/-- The genuine full Gram observable Hessian is the finite Gram-weighted
sum of genuine attention-pair Hessians, by proved original smoothness.
Source: arXiv:2604.01978v1, §7, the full overlap sum after `eq:SIGMA_R`. -/
theorem iteratedFDeriv_two_gaussianGramObservable {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (A : Matrix (Idx n) (Idx n) ℝ) (i j : Idx n)
    (z v w : EuclideanSpace ℝ (GaussianFeatureIdx d n)) :
    iteratedFDeriv ℝ 2 (gaussianGramObservable β A i j) z ![v, w] =
      ∑ k, ∑ l, A k l * iteratedFDeriv ℝ 2 (fun y =>
        Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
          Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z ![v, w] := by
  let φ (k l : Idx n) := fun y : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
    Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
      Perspective.softmaxWeight (gaussianFeatureScores β y j) l
  have hc (a b : Idx n) : ContDiff ℝ 2 (fun y : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
      Perspective.softmaxWeight (gaussianFeatureScores β y a) b) :=
    ((hasTemperateGrowth_softmaxWeight hn b).comp
      (hasTemperateGrowth_gaussianFeatureScores β a)).1.of_le (by simp)
  have hφ (k l : Idx n) : ContDiff ℝ 2 (φ k l) := (hc i k).mul (hc j l)
  have hF (k l : Idx n) : ContDiff ℝ 2 (A k l • φ k l) := (hφ k l).const_smul (A k l)
  have hsum (k : Idx n) : ContDiff ℝ 2 (∑ l, A k l • φ k l) := by
    convert ContDiff.sum (s := Finset.univ) (fun l hl => hF k l) using 1
    funext y
    simp only [Finset.sum_apply]
  have hfun : gaussianGramObservable (d := d) β A i j = ∑ k, ∑ l, A k l • φ k l := by
    funext y
    simp only [gaussianGramObservable, Finset.sum_apply]
    apply Finset.sum_congr rfl
    intro k hk
    apply Finset.sum_congr rfl
    intro l hl
    change A k l * Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
      Perspective.softmaxWeight (gaussianFeatureScores β y j) l =
      A k l * (Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l)
    ring
  rw [hfun, iteratedFDeriv_sum_apply (fun k hk => (hsum k).contDiffAt), sum_apply]
  apply Finset.sum_congr rfl
  intro k hk
  rw [iteratedFDeriv_sum_apply (fun l hl => (hF k l).contDiffAt), sum_apply]
  apply Finset.sum_congr rfl
  intro l hl
  rw [iteratedFDeriv_const_smul_apply (hφ k l).contDiffAt]
  rfl

/-- The two original finite contractions can be exchanged exactly.
Source: arXiv:2604.01978v1, §7, exchanging the output Gram sum
with the actual feature-coordinate trace in `eq:steins_lemma`. -/
theorem sum_weighted_matrix_comm {ι : Type*} [Fintype ι] {n : ℕ}
    (A : Idx n → Idx n → ℝ) (B : ι → ℝ) (T : ι → Idx n → Idx n → ℝ) :
    (∑ x, B x * ∑ k, ∑ l, A k l * T x k l) =
      ∑ k, ∑ l, A k l * ∑ x, B x * T x k l := by
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k hk
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun l hl => Finset.sum_congr rfl fun x hx => by ring

/-- The actual full Gram observable block trace is exactly the
Gram-weighted sum of actual original attention-pair block traces.
Source: arXiv:2604.01978v1, §7, `eq:steins_lemma` for the overlap sum. -/
theorem gaussianGramObservable_block_trace {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (A B : Matrix (Idx n) (Idx n) ℝ) (i j : Idx n)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) :
    (∑ c : Fin 2, ∑ q : Fin d, ∑ a, ∑ b,
      B a b * iteratedFDeriv ℝ 2 (gaussianGramObservable β A i j) z
        ![EuclideanSpace.single (c, a, q) 1, EuclideanSpace.single (c, b, q) 1]) =
      ∑ k, ∑ l, A k l * (∑ c : Fin 2, ∑ q : Fin d, ∑ a, ∑ b,
        B a b * iteratedFDeriv ℝ 2 (fun y =>
          Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
            Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
          ![EuclideanSpace.single (c, a, q) 1, EuclideanSpace.single (c, b, q) 1]) := by
  simp_rw [iteratedFDeriv_two_gaussianGramObservable hn β A i j]
  have h := sum_weighted_matrix_comm
    (ι := Fin 2 × Fin d × Idx n × Idx n) A (fun p => B p.2.2.1 p.2.2.2)
    (fun p k l => iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (p.1, p.2.2.1, p.2.1) 1,
        EuclideanSpace.single (p.1, p.2.2.2, p.2.1) 1])
  simpa only [Fintype.sum_prod_type] using h

/-- The genuine full observable block trace is bounded pointwise by
the original query/key squared norms with constants twenty and ten.
Source: arXiv:2604.01978v1, §7, the bound used for
`eq:Lispchitz_bound_softmax`, with normalization before expectation. -/
theorem abs_gaussianGramObservable_block_trace_le {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (A B : Matrix (Idx n) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1)
    (i j : Idx n) (z : EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (ε : ℝ) (hB : ∀ a b, |B a b| ≤ ε) :
    |∑ c : Fin 2, ∑ q : Fin d, ∑ a, ∑ b,
      B a b * iteratedFDeriv ℝ 2 (gaussianGramObservable β A i j) z
        ![EuclideanSpace.single (c, a, q) 1, EuclideanSpace.single (c, b, q) 1]| ≤
      β ^ 2 * ε * (20 * (∑ m : Idx n, ∑ q : Fin d, z (1, m, q) ^ 2) +
        10 * ((∑ q : Fin d, z (0, i, q) ^ 2) + ∑ q : Fin d, z (0, j, q) ^ 2)) := by
  rw [gaussianGramObservable_block_trace hn]
  exact abs_sum_weighted_gaussianAttention_block_traces_le hn β z i j A B hA ε hB

/-- A nonzero biased original feature vector and bounded identity Gram
coefficients give the actual overlap `5/8` and a nonzero genuine
key-coordinate observable Hessian of `3/64`. -/
example : ∃ z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2), (0 : ℕ) < 2 ∧
    (∀ k l : Idx 2, |(1 : Matrix (Idx 2) (Idx 2) ℝ) k l| ≤ 1) ∧
    gaussianGramObservable 1 (1 : Matrix (Idx 2) (Idx 2) ℝ) 0 0 z = (5 / 8 : ℝ) ∧
    iteratedFDeriv ℝ 2 (gaussianGramObservable (d := 2) 1 (1 : Matrix (Idx 2) (Idx 2) ℝ) 0 0) z
      ![EuclideanSpace.single (1, 0, 0) 1, EuclideanSpace.single (1, 0, 0) 1] = (3 / 64 : ℝ) := by
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  have hs : gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] := by
    ext j
    fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]
  have hw0 : Perspective.softmaxWeight ![Real.log 3, 0] (0 : Idx 2) = (3 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  have hw1 : Perspective.softmaxWeight ![Real.log 3, 0] (1 : Idx 2) = (1 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  refine ⟨z, by norm_num, ?_, ?_, ?_⟩
  · intro k l
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  · norm_num [gaussianGramObservable, hs, Fin.sum_univ_two, Matrix.one_apply, hw0, hw1]
  · rw [iteratedFDeriv_two_gaussianGramObservable (by norm_num)]
    simp only [Fin.sum_univ_two, Matrix.one_apply,
      iteratedFDeriv_two_gaussianAttention_product_key_axes (n := 2) (by norm_num)]
    rw [hs]
    simp only [iteratedFDeriv_two_softmaxWeight (m := 2) (by norm_num),
      fderiv_softmaxWeight_factor (m := 2) (by norm_num)]
    norm_num [z, PiLp.single_apply, hw0, hw1]

end Transformer.Homogenized

