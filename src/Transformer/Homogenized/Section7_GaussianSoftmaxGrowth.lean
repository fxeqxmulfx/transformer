module

public import Transformer.Homogenized.SoftmaxSmooth
public import Transformer.Homogenized.Section7_GaussianHeadFeatures
public import Mathlib.Analysis.Distribution.TemperateGrowth

/-!
# Polynomial growth of the actual Gaussian softmax overlap observable

ArXiv:2604.01978v1, §7, the justification of Gaussian interpolation
before `eq:interpolation_interm`, using `claim:softmax_is_smooth`.
The original finite softmax coordinate derivatives have already been
proved bounded at every order. Expanding actual multilinear derivatives
on coordinate vectors now bounds their genuine operator norms.

The resulting softmax has polynomial growth at every derivative order.
The original Gaussian feature scores are actual bilinear polynomials.
Composing these functions, multiplying the two actual softmax weights,
and taking the finite Gram coefficient sum proves polynomial growth
for the actual original overlap observable. In particular, one common
polynomial controls its value and first two actual derivatives, exactly
as required by the proved Gaussian interpolation theorem.
-/

@[expose] public section
open scoped BigOperators
namespace Transformer.Homogenized
/-- A bound on actual coordinate values bounds the norm of a finite
multilinear form, with an explicit finite-dimensional factor.
Source: arXiv:2604.01978v1, §7, the operator-norm justification of
polynomial derivative growth before `eq:interpolation_interm`. -/
theorem norm_multilinear_pi_le_coordinate_bound {ι : Type*} [Fintype ι] [DecidableEq ι] {k : ℕ}
    (T : ContinuousMultilinearMap ℝ (fun _ : Fin k => ι → ℝ) ℝ)
    (C : ℝ) (hC : 0 ≤ C)
    (hb : ∀ r : Fin k → ι, ‖T (fun i => Pi.single (r i) (1 : ℝ))‖ ≤ C) :
    ‖T‖ ≤ (Fintype.card ι : ℝ) ^ k * C := by
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  have hv (i : Fin k) : (∑ q : ι, v i q • Pi.single q (1 : ℝ)) = v i := by
    ext q
    simp [Pi.single_apply]
  have he : T v = ∑ r : Fin k → ι,
      (∏ i : Fin k, v i (r i)) • T (fun i => Pi.single (r i) (1 : ℝ)) := by
    calc
      T v = T (fun i => ∑ q : ι, v i q • Pi.single q (1 : ℝ)) := by simp only [hv]
      _ = _ := by rw [T.map_sum]; simp only [T.map_smul_univ]
  rw [he]
  calc
    _ ≤ ∑ r : Fin k → ι,
        ‖(∏ i : Fin k, v i (r i)) • T (fun i => Pi.single (r i) (1 : ℝ))‖ := norm_sum_le _ _
    _ ≤ ∑ _r : Fin k → ι, (∏ i : Fin k, ‖v i‖) * C := by
      apply Finset.sum_le_sum
      intro r hr
      rw [norm_smul]
      have hp : ‖∏ i : Fin k, v i (r i)‖ ≤ ∏ i : Fin k, ‖v i‖ :=
        (Finset.norm_prod_le _ _).trans
          (Finset.prod_le_prod₀ (fun i hi => norm_nonneg _) (fun i hi => norm_le_pi_norm (v i) (r i)))
      exact mul_le_mul hp (hb r) (norm_nonneg _) (by positivity)
    _ = (Fintype.card ι : ℝ) ^ k * C * ∏ i : Fin k, ‖v i‖ := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin,
        nsmul_eq_mul, Nat.cast_pow]
      ring

/-- Every actual softmax derivative has a global operator-norm bound
at each fixed finite score dimension. Source: arXiv:2604.01978v1,
`claim:softmax_is_smooth`, used in §7 before `eq:interpolation_interm`.
The source's first-derivative sign is corrected in `SoftmaxDerivatives`. -/
theorem norm_iteratedFDeriv_softmaxWeight_le {m : ℕ} (hm : 0 < m) (k : ℕ) (j : Idx m) (u : Idx m → ℝ) :
    ‖iteratedFDeriv ℝ k (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u‖ ≤
      (m : ℝ) ^ k * monomialBound k 1 := by
  have hfun : (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) =
      softmaxMonomial (fun _ : Fin (0 + 1) => j) := by
    funext v
    simp [softmaxMonomial]
  have hc := monomialBound_nonneg k 1
  have hb (r : Fin k → Idx m) :
      ‖iteratedFDeriv ℝ k (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
        (fun i => Pi.single (r i) (1 : ℝ))‖ ≤ monomialBound k 1 := by
    rw [Real.norm_eq_abs, hfun]
    exact (abs_iteratedFDeriv_softmaxMonomial_le hm k 0 _ r u).trans
      (mul_le_of_le_one_right hc (softmaxWeight_le_one hm u j))
  simpa only [Fintype.card_fin] using norm_multilinear_pi_le_coordinate_bound _ (monomialBound k 1) hc hb

/-- The actual original softmax has polynomial growth at every
actual derivative order. Source: arXiv:2604.01978v1, §7, the regularity
justification before `eq:interpolation_interm`. -/
theorem hasTemperateGrowth_softmaxWeight {m : ℕ} (hm : 0 < m) (j : Idx m) :
    Function.HasTemperateGrowth (fun u : Idx m → ℝ => Perspective.softmaxWeight u j) := by
  refine ⟨contDiff_softmaxWeight hm j, fun k => ⟨0, (m : ℝ) ^ k * monomialBound k 1, ?_⟩⟩
  intro u
  simpa only [pow_zero, mul_one] using norm_iteratedFDeriv_softmaxWeight_le hm k j u

/-- The actual original Gaussian feature scores are polynomials,
so their actual derivatives have polynomial growth.
Source: arXiv:2604.01978v1, §7, the bilinear scores before `eq:SIGMA_R`. -/
theorem hasTemperateGrowth_gaussianFeatureScores {d n : ℕ} (β : ℝ) (i : Idx n) :
    Function.HasTemperateGrowth (fun z : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
      gaussianFeatureScores β z i) := by
  have hcoord (j : Idx n) : Function.HasTemperateGrowth
      (fun z : EuclideanSpace ℝ (GaussianFeatureIdx d n) => gaussianFeatureScores β z i j) := by
    exact (Function.HasTemperateGrowth.const β).fun_mul
      (Function.HasTemperateGrowth.sum fun q hq =>
        (EuclideanSpace.proj (𝕜 := ℝ) (ι := GaussianFeatureIdx d n) (0, i, q)).hasTemperateGrowth.fun_mul
          (EuclideanSpace.proj (𝕜 := ℝ) (ι := GaussianFeatureIdx d n) (1, j, q)).hasTemperateGrowth)
  have hsum := Function.HasTemperateGrowth.sum (s := Finset.univ) fun j hj =>
    (hcoord j).fun_smul (Function.HasTemperateGrowth.const (Pi.single j (1 : ℝ) : Idx n → ℝ))
  convert hsum using 1
  funext z
  ext j
  simp [Pi.single_apply]

/-- The original overlap observable with its fixed Gram coefficients,
as an actual function of the original Gaussian query/key features.
Source: arXiv:2604.01978v1, §7, the displayed formula for
`s_{μ_X}(x_i,x_j)` immediately after `eq:SIGMA_R`. -/
noncomputable def gaussianGramObservable {d n : ℕ} (β : ℝ) (A : Matrix (Idx n) (Idx n) ℝ)
    (i j : Idx n) (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) : ℝ :=
  ∑ k, ∑ l, A k l * Perspective.softmaxWeight (gaussianFeatureScores β z i) k *
    Perspective.softmaxWeight (gaussianFeatureScores β z j) l

/-- The original finite overlap sum has polynomial growth at every
actual derivative order. Source: arXiv:2604.01978v1, §7, the regularity
justification before `eq:interpolation_interm`, applied before expectation. -/
theorem hasTemperateGrowth_gaussianGramObservable {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (A : Matrix (Idx n) (Idx n) ℝ) (i j : Idx n) :
    Function.HasTemperateGrowth (gaussianGramObservable (d := d) β A i j) := by
  have hχ (a k : Idx n) : Function.HasTemperateGrowth
      (fun z : EuclideanSpace ℝ (GaussianFeatureIdx d n) =>
        Perspective.softmaxWeight (gaussianFeatureScores β z a) k) :=
    (hasTemperateGrowth_softmaxWeight hn k).comp (hasTemperateGrowth_gaussianFeatureScores (d := d) β a)
  exact Function.HasTemperateGrowth.sum fun k hk => Function.HasTemperateGrowth.sum fun l hl =>
    ((Function.HasTemperateGrowth.const (A k l)).fun_mul (hχ i k)).fun_mul (hχ j l)

/-- The actual original overlap observable satisfies the complete
regularity and polynomial-growth premises of proved Gaussian
interpolation. Source: arXiv:2604.01978v1, §7, the justification before
`eq:interpolation_interm`. No growth premise is assumed here. -/
theorem gaussianGramObservable_polynomial_growth {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (A : Matrix (Idx n) (Idx n) ℝ) (i j : Idx n) :
    ContDiff ℝ 2 (gaussianGramObservable (d := d) β A i j) ∧ ∃ (C : ℝ) (p : ℕ), 0 ≤ C ∧
      ∀ k ≤ 2, ∀ z, ‖iteratedFDeriv ℝ k (gaussianGramObservable (d := d) β A i j) z‖ ≤ C * (1 + ‖z‖) ^ p := by
  have h := hasTemperateGrowth_gaussianGramObservable (d := d) hn β A i j
  obtain ⟨p, C, hC, hb⟩ := h.norm_iteratedFDeriv_le_uniform 2
  exact ⟨h.1.of_le (by simp), C, p, hC, hb⟩

example : ∃ u : Idx 2 → ℝ, (0 : ℕ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    ∀ r : Fin 1 → Idx 2,
      ‖iteratedFDeriv ℝ 1 (fun v : Idx 2 → ℝ => Perspective.softmaxWeight v 0) u
        (fun i => Pi.single (r i) (1 : ℝ))‖ ≤ 1 := by
  refine ⟨0, by norm_num, zero_le_one, ?_⟩
  intro r
  rw [iteratedFDeriv_one_apply, fderiv_softmaxWeight (by norm_num)]
  have h : ∀ j : Idx 2, Perspective.softmaxWeight 0 j = (1 / 2 : ℝ) := by
    intro j
    norm_num [Perspective.softmaxWeight, Fin.sum_univ_two]
  simp only [h]
  split_ifs <;> norm_num

example : (0 : ℕ) < 2 ∧ Function.HasTemperateGrowth
    (fun z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) =>
      gaussianFeatureScores 1 z (0 : Idx 2)) ∧
    Function.HasTemperateGrowth
      (gaussianGramObservable (d := 2) 1 (1 : Matrix (Idx 2) (Idx 2) ℝ) 0 1) :=
  ⟨by norm_num, hasTemperateGrowth_gaussianFeatureScores _ _, hasTemperateGrowth_gaussianGramObservable (by norm_num) _ _ _ _⟩
end Transformer.Homogenized
