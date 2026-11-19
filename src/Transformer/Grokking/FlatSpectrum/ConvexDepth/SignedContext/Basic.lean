module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation.Batch

/-!
# Bounded signed context features

Source: `review/convex/moments.md`, CM51. Fixed signed rows act on the actual
complex source features. Their absolute mass measures amplification. Dividing
by the larger of one and that mass retains the original feature/head bound.
The finite-product and convex ordinary-CE identities of CM42 allow signed rows;
its positive-weight Jensen and same-label transfer theorems do not.
Reference-score error and a separately assumed query reconstruction residual
give a conditional raw-margin guarantee. These target premises are not implied
by the observable row or by flatness.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext

open scoped BigOperators
open ContextInterpolation (features)

variable {K I X C : Type*} [Fintype I]

/-- Observable absolute mass of a fixed signed source row, CM51. -/
noncomputable def mass (w : I → ℝ) : ℝ := ∑ i, |w i|

/-- Input-only normalization prevents amplification of bounded coordinates. -/
noncomputable def normalizer (w : I → ℝ) : ℝ := max 1 (mass w)

/-- The normalized row is fixed during complete-head CE optimization. -/
noncomputable def normalized (w : I → ℝ) (i : I) : ℝ := w i / normalizer w

theorem mass_nonneg (w : I → ℝ) : 0 ≤ mass w :=
  Finset.sum_nonneg (fun _ _ => abs_nonneg _)

theorem normalizer_pos (w : I → ℝ) : 0 < normalizer w := by
  have h := le_max_left (1 : ℝ) (mass w)
  change 0 < max 1 (mass w)
  linarith

/-- Absolute mass after normalization, including the zero row, CM51. -/
theorem normalized_mass (w : I → ℝ) :
    mass (normalized w) = mass w / normalizer w := by
  unfold mass normalized
  simp_rw [abs_div, abs_of_pos (normalizer_pos w)]
  rw [Finset.sum_div]

theorem normalized_mass_le_one (w : I → ℝ) : mass (normalized w) ≤ 1 := by
  rw [normalized_mass, div_le_iff₀ (normalizer_pos w), one_mul]
  exact le_max_right 1 (mass w)

/-- Signed rows amplify a source coordinate by at most their absolute mass,
CM51. No positivity or unit-sum hypothesis is imposed on the row. -/
theorem features_bound (w : X → I → ℝ) (source : K → I → ℝ) (x : X)
    {M : ℝ} (hsource : ∀ k i, |source k i| ≤ M) (k : K) :
    |features w source k x| ≤ mass (w x) * M := by
  calc
    _ ≤ ∑ i, |w x i * source k i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |w x i| * M := by
      apply Finset.sum_le_sum
      intro i _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hsource k i) (abs_nonneg _)
    _ = mass (w x) * M := by rw [← Finset.sum_mul]; rfl

/-- A normalized signed map retains the original coordinate bound, CM51. -/
theorem normalized_features_bound (w : X → I → ℝ) (source : K → I → ℝ)
    (x : X) {M : ℝ} (hM : 0 ≤ M) (hsource : ∀ k i, |source k i| ≤ M) (k : K) :
    |features (fun x => normalized (w x)) source k x| ≤ M := by
  calc
    _ ≤ mass (normalized (w x)) * M := features_bound _ source x hsource k
    _ ≤ 1 * M := mul_le_mul_of_nonneg_right (normalized_mass_le_one (w x)) hM
    _ = M := one_mul M

/-- Flat unitary source codes give the coordinate bound one for signed rows,
CM51. The new query matrix is not asserted to be unitary. -/
theorem complex_features_bound [DecidableEq I] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (w : X → I → ℝ) (x : X) (k : I × Bool) :
    |features (fun x => normalized (w x)) (complexFeatures U) k x| ≤ 1 := by
  exact normalized_features_bound w _ x (by norm_num)
    (fun k i => ConvexDepth.complexFeatures_bound U hU k i) k

/-- Every real and imaginary head coefficient remains charged, CM51. -/
theorem logits_bound [Fintype K] (w : X → I → ℝ) (source : K → I → ℝ)
    (x : X) {M R : ℝ} (hM : 0 ≤ M) (hsource : ∀ k i, |source k i| ≤ M)
    {W : K → C → ℝ} (hW : W ∈ headBudget R) (c : C) :
    |headLogits (features (fun x => normalized (w x)) source) W x c| ≤ R * M := by
  exact headLogits_bound _ hW hM x (normalized_features_bound w source x hM hsource) c

/-- The normalized prediction is the signed combination of actual source
logits divided by its positive normalizer, CM51. This retains the full head. -/
theorem normalized_logits [Fintype K] (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (c : C) :
    headLogits (features (fun x => normalized (w x)) source) W x c =
      (∑ i, w x i * headLogits source W i c) / normalizer (w x) := by
  rw [ContextInterpolation.logits]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, normalized]
  simp_rw [div_mul_eq_mul_div]
  rw [Finset.sum_div]

/-- Original observed Dirac rows are unchanged by normalization, CM51. -/
theorem normalized_observed [DecidableEq I] (w : I → ℝ) (s : I)
    (hw : ∀ i, w i = if i = s then 1 else 0) (i : I) :
    normalized w i = if i = s then 1 else 0 := by
  have hm : mass w = 1 := by
    unfold mass
    calc
      _ = ∑ j : I, if j = s then (1 : ℝ) else 0 := by
        apply Finset.sum_congr rfl
        intro j _
        by_cases h : j = s <;> simp [hw, h]
      _ = 1 := by simp
  simp [normalized, normalizer, hm, hw]

/-- The source-score error is amplified by the absolute mass, CM51. -/
theorem combination_error (w : I → ℝ) (z reference : I → ℝ) {ε : ℝ}
    (hε : ∀ i, |z i - reference i| ≤ ε) :
    |(∑ i, w i * z i) - ∑ i, w i * reference i| ≤ mass w * ε := by
  rw [← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  calc
    _ ≤ ∑ i, |w i * (z i - reference i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |w i| * ε := by
      apply Finset.sum_le_sum
      intro i _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hε i) (abs_nonneg _)
    _ = mass w * ε := by rw [← Finset.sum_mul]; rfl

/-- The actual normalized head is close to a supplied query score when both
observed score error and reference reconstruction residual are bounded, CM51. -/
theorem query_error [Fintype K] (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (reference : I → C → ℝ) (query : C → ℝ)
    (x : X) (c : C) {ε δ : ℝ}
    (hε : ∀ i, |headLogits source W i c - reference i c| ≤ ε)
    (hδ : |(∑ i, w x i * reference i c) - query c| ≤ δ) :
    |headLogits (features (fun x => normalized (w x)) source) W x c -
        query c / normalizer (w x)| ≤
      (mass (w x) * ε + δ) / normalizer (w x) := by
  rw [normalized_logits, ← sub_div, abs_div, abs_of_pos (normalizer_pos (w x))]
  apply div_le_div_of_nonneg_right _ (le_of_lt (normalizer_pos (w x)))
  calc
    _ ≤ |(∑ i, w x i * headLogits source W i c) - ∑ i, w x i * reference i c| +
        |(∑ i, w x i * reference i c) - query c| := abs_sub_le _ _ _
    _ ≤ mass (w x) * ε + δ := add_le_add (combination_error (w x) _ _ hε) hδ

/-- A reference raw margin larger than twice the unnormalized error survives
in the actual complete head, CM51. The reference/residual premises are explicit. -/
theorem strict_margin [Fintype K] (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (reference : I → C → ℝ) (query : C → ℝ)
    (x : X) (y c : C) {ε δ : ℝ}
    (hε : ∀ i d, |headLogits source W i d - reference i d| ≤ ε)
    (hδ : ∀ d, |(∑ i, w x i * reference i d) - query d| ≤ δ)
    (hm : query c + 2 * (mass (w x) * ε + δ) < query y) :
    headLogits (features (fun x => normalized (w x)) source) W x c <
      headLogits (features (fun x => normalized (w x)) source) W x y := by
  have hc := (abs_le.mp (query_error w source W reference query x c (fun i => hε i c) (hδ c))).2
  have hy := (abs_le.mp (query_error w source W reference query x y (fun i => hε i y) (hδ y))).1
  have hm' := (div_lt_div_iff_of_pos_right (normalizer_pos (w x))).2 hm
  rw [add_div, mul_div_assoc] at hm'
  linarith

-- A genuinely signed row inhabits the conditional bounds with nonzero sources.
noncomputable def exampleRow : Fin 4 → ℝ := ![-1, 1, 1, 0]

example : mass exampleRow = 3 := by norm_num [mass, exampleRow, Fin.sum_univ_four]
example := features_bound (fun _ : Unit => exampleRow) (complexFeatures fourier4) ()
  (fun k i => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k i) (0, false)
example := normalized_features_bound (fun _ : Unit => exampleRow) (complexFeatures fourier4) ()
  (by norm_num : (0 : ℝ) ≤ 1)
  (fun k i => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k i) (0, true)
example := complex_features_bound fourier4 fourier4_unitary (fun _ : Unit => exampleRow) () (1, true)
example := logits_bound (fun _ : Unit => exampleRow) (complexFeatures fourier4) ()
  (by norm_num : (0 : ℝ) ≤ 1)
  (fun k i => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k i)
  (W := fun _ (_ : Bool) => 1) (R := 8)
  (by intro c; norm_num [Fintype.card_prod]) true
example := normalized_observed (ContextInterpolation.pointWeights (id : Fin 4 → Fin 4) 2) 2
  (fun _ => rfl) 1
example := combination_error exampleRow (fun i => (i.val : ℝ) + 1)
  (fun i => (i.val : ℝ)) (ε := 1) (by intro i; norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext
