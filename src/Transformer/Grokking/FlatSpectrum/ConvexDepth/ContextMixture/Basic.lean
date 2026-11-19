module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence.Batch

/-!
# Convex mixtures of fixed context feature maps

Source: `review/convex/moments.md`, CM45. The finite bank and complete complex
head are fixed while the mixture is learned. The source spectrum bounds the
bank; it does not choose a neighborhood or identify a missing raw target.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture

open scoped BigOperators

variable {J K X C I : Type*} [Fintype J] [Fintype K]

omit [Fintype K] in
/-- Ordinary nonnegative mixture coefficients with total mass one, CM45. -/
def simplex : Set (J → ℝ) := {a | (∀ j, 0 ≤ a j) ∧ ∑ j, a j = 1}

omit [Fintype K] in
/-- The complete mixture constraint is convex, CM45. The elementary argument
also appears in Mathlib's deprecated `convex_stdSimplex`; no deprecated API is used. -/
theorem simplex_convex : Convex ℝ (simplex : Set (J → ℝ)) := by
  intro a ha b hb t u ht hu htu
  refine ⟨fun j => add_nonneg (mul_nonneg ht (ha.1 j)) (mul_nonneg hu (hb.1 j)), ?_⟩
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_add_distrib,
    ← Finset.mul_sum, ha.2, hb.2, mul_one]
  exact htu

omit [Fintype K] in
/-- Mix actual fixed features before applying the complete source head,
CM45. Every bank element may use a different declared context rule. -/
noncomputable def features (bank : J → K → X → ℝ) (a : J → ℝ) : K → X → ℝ :=
  fun k x => ∑ j, a j * bank j k x

omit [Fintype K] in
/-- The feature mixture is linear in all mixture coefficients, CM45.
The bank remains fixed throughout this identity. -/
theorem features_linear (bank : J → K → X → ℝ) (a b : J → ℝ) (t u : ℝ) :
    features bank (t • a + u • b) = t • features bank a + u • features bank b := by
  funext k x
  simp only [features, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
    Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Expanding the actual complex-head products gives the mixture of the
bank's actual logits, CM45. The same complete tensor is used in every term. -/
theorem logits (bank : J → K → X → ℝ) (a : J → ℝ) (W : K → C → ℝ) (x : X) :
    headLogits (features bank a) W x = ∑ j, a j • headLogits (bank j) W x := by
  funext c
  simp only [headLogits, features, Finset.sum_apply, Pi.smul_apply, smul_eq_mul,
    Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- The full logit vector is linear in the mixture with the head fixed,
CM45. This is not a joint statement about changing the head and mixture. -/
theorem logits_linear (bank : J → K → X → ℝ) (a b : J → ℝ) (t u : ℝ)
    (W : K → C → ℝ) (x : X) :
    headLogits (features bank (t • a + u • b)) W x =
      t • headLogits (features bank a) W x + u • headLogits (features bank b) W x := by
  rw [logits, logits, logits]
  funext c
  simp only [Finset.sum_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
    Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j _
  ring

omit [Fintype K] in
/-- A simplex mixture retains the bank's coordinate bound, CM45. Context
rows may repeat; orthogonality of the input-query matrix is not assumed. -/
theorem features_bound (bank : J → K → X → ℝ) {a : J → ℝ}
    (ha : a ∈ simplex) (x : X) {M : ℝ}
    (hbank : ∀ j k, |bank j k x| ≤ M) (k : K) : |features bank a k x| ≤ M := by
  exact ContextInterpolation.features_bound (fun _ : Unit => a)
    (fun k j => bank j k x) () ha.1 ha.2 (fun k j => hbank j k) k

/-- Every original real and imaginary coefficient remains charged to the
same head budget, CM45. Mixture weights add no free output decoder. -/
theorem logits_bound (bank : J → K → X → ℝ) {a : J → ℝ} (ha : a ∈ simplex)
    (x : X) {M R : ℝ} (hM : 0 ≤ M) (hbank : ∀ j k, |bank j k x| ≤ M)
    {W : K → C → ℝ} (hW : W ∈ headBudget R) (c : C) :
    |headLogits (features bank a) W x c| ≤ R * M := by
  apply headLogits_bound _ hW hM x
  exact features_bound bank ha x hbank

omit [Fintype K] in
/-- Actual complex context banks with fixed confidence scales in [0,1]
retain their unitary-source bound after mixing, CM45. -/
theorem complex_features_bound [Fintype I] [DecidableEq I] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (w : J → X → I → ℝ)
    (scale : J → X → ℝ) {a : J → ℝ} (ha : a ∈ simplex) (x : X)
    (hw : ∀ j i, 0 ≤ w j x i) (hs : ∀ j, ∑ i, w j x i = 1)
    (hzero : ∀ j, 0 ≤ scale j x) (hone : ∀ j, scale j x ≤ 1) (k : I × Bool) :
    |features (fun j => ContextScale.features (scale j)
      (ContextInterpolation.features (w j) (complexFeatures U))) a k x| ≤ 1 := by
  apply features_bound _ ha x
  intro j k
  exact ContextScale.complex_features_bound U hU (w j) (scale j) x
    (hw j) (hs j) (hzero j) (hone j) k

omit [Fintype K] in
/-- If every bank element keeps an observed feature row, any mixture with
total mass one keeps it too, CM45. Nonnegativity is unnecessary for this identity. -/
theorem features_at (bank : J → K → X → ℝ) (source : K → X → ℝ) (a : J → ℝ)
    (hs : ∑ j, a j = 1) (x : X) (hbank : ∀ j k, bank j k x = source k x) (k : K) :
    features bank a k x = source k x := by
  change (∑ j, a j * bank j k x) = source k x
  simp_rw [hbank]
  rw [← Finset.sum_mul, hs, one_mul]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture
