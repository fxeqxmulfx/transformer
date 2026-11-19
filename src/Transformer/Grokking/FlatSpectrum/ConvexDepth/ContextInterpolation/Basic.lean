module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PartialFallback.Batch
public import Mathlib.Analysis.Convex.Jensen

/-!
# Fixed context interpolation of actual complex features

Source: `review/convex/moments.md`, CM42. A fixed stochastic row combines
features at source states, using the same complete real/imaginary coefficient
tensor. Flatness bounds those source features; the interpolated query matrix
need not be unitary. No statement infers an unseen raw label from flatness.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation

open scoped BigOperators

variable {K I X C : Type*} [Fintype K] [Fintype I]

/-- Fixed context weights act on source features, before the learned head,
CM42. The weights may depend on observed data but are frozen during CE training. -/
noncomputable def features (w : X → I → ℝ) (source : K → I → ℝ) : K → X → ℝ :=
  fun k x => ∑ i, w x i * source k i

/-- The actual finite head equals the weighted source logits, CM42.
There are no separately trained interpolation and decoder factors. -/
theorem logits (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) :
    headLogits (features w source) W x = ∑ i, w x i • headLogits source W i := by
  funext c
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, headLogits, features, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro k _
  ring

omit [Fintype K] in
/-- A stochastic interpolation row preserves a source coordinate bound,
CM42. It does not assert that the interpolated rows are orthogonal. -/
theorem features_bound (w : X → I → ℝ) (source : K → I → ℝ) (x : X)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1) {M : ℝ}
    (hsource : ∀ k i, |source k i| ≤ M) (k : K) :
    |features w source k x| ≤ M := by
  calc
    _ ≤ ∑ i, |w x i * source k i| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, w x i * |source k i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_of_nonneg (hw i)]
    _ ≤ ∑ i, w x i * M := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_left (hsource k i) (hw i)
    _ = M := by rw [← Finset.sum_mul, hs, one_mul]

/-- Every actual coefficient is still charged to the same budget, CM42.
The stochastic map adds no learned coordinates or uncharged output decoder. -/
theorem logits_bound (w : X → I → ℝ) (source : K → I → ℝ) (x : X)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1) {M R : ℝ}
    (hM : 0 ≤ M) (hsource : ∀ k i, |source k i| ≤ M)
    {W : K → C → ℝ} (hW : W ∈ headBudget R) (c : C) :
    |headLogits (features w source) W x c| ≤ R * M := by
  apply headLogits_bound _ hW hM x
  exact features_bound w source x hw hs hsource

omit [Fintype K] in
/-- Unitary complex source codes give the actual interpolated real and
imaginary coordinate bound, CM42, for arbitrary fixed stochastic rows. -/
theorem complex_features_bound [DecidableEq I] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (w : X → I → ℝ) (x : X)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1) (k : I × Bool) :
    |features w (complexFeatures U) k x| ≤ 1 := by
  apply features_bound w (complexFeatures U) x hw hs
  exact fun k i => ConvexDepth.complexFeatures_bound U hU k i

omit [Fintype K] in
/-- Dirac weights retain an original source state exactly, CM42. -/
noncomputable def pointWeights [DecidableEq I] (state : X → I) (x : X) (i : I) : ℝ :=
  if i = state x then 1 else 0

omit [Fintype K] [Fintype I] in
theorem pointWeights_nonneg [DecidableEq I] (state : X → I) (x : X) (i : I) :
    0 ≤ pointWeights state x i := by
  unfold pointWeights
  split_ifs <;> norm_num

omit [Fintype K] in
theorem pointWeights_sum [DecidableEq I] (state : X → I) (x : X) :
    ∑ i, pointWeights state x i = 1 := by
  simp [pointWeights]

omit [Fintype K] in
/-- An observed query whose row is Dirac has unchanged features for all
coefficient tensors, CM42. This is the premise used by the batch identities. -/
theorem features_at [DecidableEq I] (w : X → I → ℝ) (source : K → I → ℝ)
    (x : X) (i : I) (hi : ∀ j, w x j = if j = i then 1 else 0) (k : K) :
    features w source k x = source k i := by
  simp only [features, hi]
  simp

-- Nonzero actual complex features instantiate all conditional bounds.
example := features_bound (pointWeights (id : Fin 4 → Fin 4)) (complexFeatures fourier4)
  0 (pointWeights_nonneg id 0) (pointWeights_sum id 0)
  (fun k i => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k i) (0, false)
example := logits_bound (pointWeights (id : Fin 4 → Fin 4)) (complexFeatures fourier4)
  0 (pointWeights_nonneg id 0) (pointWeights_sum id 0) (by norm_num : (0 : ℝ) ≤ 1)
  (fun k i => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k i)
  (W := fun _ (_ : Bool) => 1) (R := 8)
  (by intro c; norm_num [Fintype.card_prod]) true
example := complex_features_bound fourier4 fourier4_unitary
  (pointWeights (id : Fin 4 → Fin 4)) 0 (pointWeights_nonneg id 0)
  (pointWeights_sum id 0) (0, true)
example := features_at (pointWeights (id : Fin 4 → Fin 4)) (complexFeatures fourier4)
  0 0 (fun _ => rfl) (0, false)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation
