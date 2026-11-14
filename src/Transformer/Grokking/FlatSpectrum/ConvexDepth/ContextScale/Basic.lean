module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation.Batch

/-!
# Fixed confidence scales on actual context features

Source: `review/flat_spectrum.md`, CM43. A fixed input-dependent scale acts
before the complete complex head. A scale in [0,1] preserves the coordinate
bound; a positive scale preserves raw-class comparisons. These statements
do not preserve comparisons between sums of class probabilities.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {K I X C : Type*} [Fintype K]

/-- CM43 scales the fixed features, leaving every trained coefficient intact.
The scale is frozen from training data before optimizing the head. -/
noncomputable def features (scale : X → ℝ) (source : K → X → ℝ) : K → X → ℝ :=
  fun k x => scale x * source k x

/-- The actual finite dot product has the declared scaled logits, CM43. -/
theorem logits (scale : X → ℝ) (source : K → X → ℝ)
    (W : K → C → ℝ) (x : X) :
    headLogits (features scale source) W x = scale x • headLogits source W x := by
  funext c
  simp only [headLogits, features, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  ring

omit [Fintype K] in
/-- Shrinking fixed features preserves their existing coordinate bound,
CM43. No new normalization or coefficient budget is hidden in the scale. -/
theorem features_bound (scale : X → ℝ) (source : K → X → ℝ) (x : X)
    (hzero : 0 ≤ scale x) (hone : scale x ≤ 1) {M : ℝ} (hM : 0 ≤ M)
    (hsource : ∀ k, |source k x| ≤ M) (k : K) :
    |features scale source k x| ≤ M := by
  rw [features, abs_mul, abs_of_nonneg hzero]
  calc
    _ ≤ scale x * M := mul_le_mul_of_nonneg_left (hsource k) hzero
    _ ≤ 1 * M := mul_le_mul_of_nonneg_right hone hM
    _ = M := one_mul M

/-- The original absolute-sum coefficient budget bounds the scaled actual
head, CM43. The scale is a fixed feature, not an additional learned factor. -/
theorem logits_bound (scale : X → ℝ) (source : K → X → ℝ) (x : X)
    (hzero : 0 ≤ scale x) (hone : scale x ≤ 1) {M R : ℝ} (hM : 0 ≤ M)
    (hsource : ∀ k, |source k x| ≤ M) {W : K → C → ℝ}
    (hW : W ∈ headBudget R) (c : C) :
    |headLogits (features scale source) W x c| ≤ R * M := by
  apply headLogits_bound _ hW hM x
  exact features_bound scale source x hzero hone hM hsource

omit [Fintype K] in
/-- Flat complex source codes give the same bound after stochastic context
interpolation and fixed shrinking, CM43. The query matrix need not be unitary. -/
theorem complex_features_bound [Fintype I] [DecidableEq I]
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (w : X → I → ℝ) (scale : X → ℝ) (x : X)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1)
    (hzero : 0 ≤ scale x) (hone : scale x ≤ 1) (k : I × Bool) :
    |features scale (ContextInterpolation.features w (complexFeatures U)) k x| ≤ 1 := by
  apply features_bound scale _ x hzero hone (by norm_num)
  exact ContextInterpolation.complex_features_bound U hU w x hw hs

/-- A fixed positive scale preserves every strict raw-class comparison,
CM43. It does not assert preservation of pooled answer probabilities. -/
theorem raw_lt_iff (scale : X → ℝ) (source : K → X → ℝ)
    (W : K → C → ℝ) (x : X) (hscale : 0 < scale x) (c d : C) :
    headLogits (features scale source) W x c < headLogits (features scale source) W x d ↔
      headLogits source W x c < headLogits source W x d := by
  rw [logits]
  simp only [Pi.smul_apply, smul_eq_mul]
  constructor
  · intro h
    nlinarith
  · intro h
    exact mul_lt_mul_of_pos_left h hscale

/-- Raw ties and non-strict order are preserved as well, CM43. A deterministic
argmax with the same tie rule therefore retains its raw prediction. -/
theorem raw_le_iff (scale : X → ℝ) (source : K → X → ℝ)
    (W : K → C → ℝ) (x : X) (hscale : 0 < scale x) (c d : C) :
    headLogits (features scale source) W x c ≤ headLogits (features scale source) W x d ↔
      headLogits source W x c ≤ headLogits source W x d := by
  rw [logits]
  simp only [Pi.smul_apply, smul_eq_mul]
  constructor
  · intro h
    nlinarith
  · intro h
    exact mul_le_mul_of_nonneg_left h hscale.le

omit [Fintype K] in
/-- Ordinary CE is convex in a single scale when its complete logit vector
is fixed, CM43. This is separate from convexity in the head at a frozen scale. -/
theorem ce_scale_convex [Fintype C] (z : C → ℝ) (y : C) :
    ConvexOn ℝ Set.univ (fun t : ℝ => crossEntropy (t • z) y) := by
  refine ⟨convex_univ, ?_⟩
  intro t _ u _ a b ha hb hab
  have h := (ConvexRecall.convex_crossEntropy y).2
    (Set.mem_univ (t • z)) (Set.mem_univ (u • z)) ha hb hab
  simpa only [add_smul, smul_smul, smul_eq_mul] using h

omit [Fintype K] in
/-- Jensen compares shrinking to a mixture of the original CE and zero-logit
CE, CM43. It does not claim that shrinking decreases every query's CE. -/
theorem ce_scale_le_mix [Fintype C] (z : C → ℝ) (y : C) {t : ℝ}
    (hzero : 0 ≤ t) (hone : t ≤ 1) :
    crossEntropy (t • z) y ≤
      t * crossEntropy z y + (1 - t) * crossEntropy (0 : C → ℝ) y := by
  have h := (ConvexRecall.convex_crossEntropy y).2
    (Set.mem_univ z) (Set.mem_univ (0 : C → ℝ)) hzero (sub_nonneg.mpr hone)
    (by ring : t + (1 - t) = 1)
  simpa only [smul_zero, add_zero, smul_eq_mul] using h

omit [Fintype K] in
/-- The actual conjugate complex initializer yields a scaled raw-label vote,
CM43. This identity justifies the canonical training-only confidence diagnostic. -/
theorem initializer_logits [Fintype I] [DecidableEq I] [DecidableEq C]
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (w : X → I → ℝ) (scale : X → ℝ) (labels : I → C) (a : ℝ) (x : X) (c : C) :
    headLogits (features scale (ContextInterpolation.features w (complexFeatures U)))
      (complexCoefficients (complexInitializer U labels a)) x c =
      scale x * (a * ContextInterpolation.labelMass w labels x c) := by
  rw [logits]
  simp only [Pi.smul_apply, smul_eq_mul]
  rw [ContextInterpolation.initializer_logits U hU]

-- Nonzero complex features and a strictly positive shrink instantiate all
-- conditional bounds and comparisons. All four raw classes remain present.
example := features_bound (fun _ : Fin 4 => (1 / 2 : ℝ)) (complexFeatures fourier4)
  0 (by norm_num) (by norm_num) (by norm_num : (0 : ℝ) ≤ 1)
  (fun k => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k 0) (0, false)
example := logits_bound (fun _ : Fin 4 => (1 / 2 : ℝ)) (complexFeatures fourier4)
  0 (by norm_num) (by norm_num) (by norm_num : (0 : ℝ) ≤ 1)
  (fun k => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k 0)
  (W := fun _ (_ : Fin 4) => 1) (R := 8)
  (by intro c; norm_num [Fintype.card_prod]) 0
example := complex_features_bound fourier4 fourier4_unitary
  (ContextInterpolation.pointWeights (id : Fin 4 → Fin 4)) (fun _ => 1 / 2) 0
  (ContextInterpolation.pointWeights_nonneg id 0) (ContextInterpolation.pointWeights_sum id 0)
  (by norm_num) (by norm_num) (0, true)
example := raw_lt_iff (fun _ : Fin 4 => (1 / 2 : ℝ)) (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 id 4)) 0 (by norm_num) 0 1
example := raw_le_iff (fun _ : Fin 4 => (1 / 2 : ℝ)) (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 id 4)) 0 (by norm_num) 1 2
example := ce_scale_le_mix (fun c : Fin 4 => if c = 0 then (4 : ℝ) else 0) 0
  (t := 1 / 2) (by norm_num) (by norm_num)
example := initializer_logits fourier4 fourier4_unitary
  (ContextInterpolation.pointWeights (id : Fin 4 → Fin 4)) (fun _ => 1 / 2) id 4 0 0

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale
