module

public import Transformer.Grokking.Convex.ConvexDepth.Fallback.Batch

/-!
# Charge every coefficient created by shared parameters

Source: `review/convex/moments.md`, CM55. The fixed affine map expands shared
parameters into the actual raw-class coefficient tensor. A weighted absolute
budget accounts for every occurrence of a parameter, including a nonzero
fixed offset. Ordinary CE remains convex on this sufficient parameter domain.
Feature discovery, decoder discovery and unseen-label correctness are separate.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ParameterSharing

open scoped BigOperators

variable {P K C X S : Type*} [Fintype P] [Fintype K]

/-- A weighted absolute budget on compact parameters, CM55. The weights charge
their complete expanded coefficients, not merely the number of stored scalars. -/
def weightedDomain (w : P → ℝ) (R : ℝ) : Set (P → ℝ) :=
  {θ | ∑ p, w p * |θ p| ≤ R}

/-- Nonnegative expansion costs give a convex parameter domain, CM55. -/
theorem weightedDomain_convex (w : P → ℝ) (hw : ∀ p, 0 ≤ w p) (R : ℝ) :
    Convex ℝ (weightedDomain w R) := by
  intro θ hθ η hη a b ha hb hab
  change (∑ p, w p * |a * θ p + b * η p|) ≤ R
  calc
    _ ≤ ∑ p, w p * (a * |θ p| + b * |η p|) := by
      apply Finset.sum_le_sum
      intro p hp
      apply mul_le_mul_of_nonneg_left _ (hw p)
      simpa [abs_mul, abs_of_nonneg ha, abs_of_nonneg hb] using
        abs_add_le (a * θ p) (b * η p)
    _ = a * (∑ p, w p * |θ p|) + b * (∑ p, w p * |η p|) := by
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro p hp
      ring
    _ ≤ a * R + b * R := add_le_add
      (mul_le_mul_of_nonneg_left hθ ha) (mul_le_mul_of_nonneg_left hη hb)
    _ = R := by rw [← add_mul, hab, one_mul]

/-- Actual complete coefficient expansion without independent learned factors,
CM55. Each parameter may occur in any number of blocks and raw classes. -/
noncomputable def coefficientMap (A : P → K → C → ℝ) :
    (P → ℝ) →ₗ[ℝ] (K → C → ℝ) where
  toFun θ k c := ∑ p, A p k c * θ p
  map_add' θ η := by
    funext k c
    simp [mul_add, Finset.sum_add_distrib]
  map_smul' a θ := by
    funext k c
    change (∑ p, A p k c * (a * θ p)) = a * ∑ p, A p k c * θ p
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro p hp
    ring

/-- Fixed offsets and shared learned coordinates form one affine complete head,
CM55. The offset is charged in addition to all shared coefficients. -/
noncomputable def parameterMap (A : P → K → C → ℝ) (B : K → C → ℝ) :
    (P → ℝ) →ᵃ[ℝ] (K → C → ℝ) :=
  AffineMap.const ℝ (P → ℝ) B + (coefficientMap A).toAffineMap

/-- An explicit expansion-cost bound transfers a compact weighted budget to
the whole per-class coefficient budget, CM55. No sign restriction on the
parameters, fixed map or offset is needed. -/
theorem parameterMap_budget (A : P → K → C → ℝ) (B : K → C → ℝ)
    (w : P → ℝ) {R T : ℝ} (hA : ∀ p c, ∑ k, |A p k c| ≤ w p)
    (hB : B ∈ headBudget T) {θ : P → ℝ} (hθ : θ ∈ weightedDomain w R) :
    parameterMap A B θ ∈ headBudget (T + R) := by
  intro c
  change (∑ k, |B k c + ∑ p, A p k c * θ p|) ≤ T + R
  calc
    _ ≤ ∑ k, (|B k c| + ∑ p, |A p k c| * |θ p|) := by
      apply Finset.sum_le_sum
      intro k hk
      refine (abs_add_le _ _).trans (add_le_add (le_refl _) ?_)
      simpa only [abs_mul] using
        Finset.abs_sum_le_sum_abs (fun p => A p k c * θ p) Finset.univ
    _ = (∑ k, |B k c|) + ∑ p, (∑ k, |A p k c|) * |θ p| := by
      rw [Finset.sum_add_distrib, Finset.sum_comm]
      simp only [Finset.sum_mul]
    _ ≤ T + ∑ p, w p * |θ p| := by
      apply add_le_add (hB c)
      apply Finset.sum_le_sum
      intro p hp
      exact mul_le_mul_of_nonneg_right (hA p c) (abs_nonneg _)
    _ ≤ T + R := add_le_add (le_refl T) hθ

/-- The same feature bound controls actual shared-head logits at the fully
charged radius, CM55. Flat complex codes supply one such feature bound. -/
theorem logits_bound (A : P → K → C → ℝ) (B : K → C → ℝ) (w : P → ℝ)
    (φ : K → X → ℝ) {R T M : ℝ} (hA : ∀ p c, ∑ k, |A p k c| ≤ w p)
    (hB : B ∈ headBudget T) {θ : P → ℝ} (hθ : θ ∈ weightedDomain w R)
    (hM : 0 ≤ M) (x : X) (hφ : ∀ k, |φ k x| ≤ M) (c : C) :
    |headLogits φ (parameterMap A B θ) x c| ≤ (T + R) * M :=
  headLogits_bound φ (parameterMap_budget A B w hA hB hθ) hM x hφ c

/-- Ordinary full-vocabulary CE remains convex in the shared parameters on
their sufficient weighted domain, CM55. The feature and expansion maps are
fixed; neither their discovery nor unknown target correctness is inferred. -/
theorem loss_convex [Fintype C] [Fintype S]
    (A : P → K → C → ℝ) (B : K → C → ℝ) (w : P → ℝ) (hw : ∀ p, 0 ≤ w p)
    (φ : K → X → ℝ) (inputs : S → X) (labels : S → C) (R T : ℝ)
    (hA : ∀ p c, ∑ k, |A p k c| ≤ w p) (hB : B ∈ headBudget T) :
    ConvexOn ℝ (weightedDomain w R)
      (fun θ => headLoss φ inputs labels (parameterMap A B θ)) := by
  have h := (headLoss_convex φ inputs labels (T + R)).comp_affineMap (parameterMap A B)
  exact h.subset (fun θ hθ => parameterMap_budget A B w hA hB hθ)
    (weightedDomain_convex w hw R)

/-- Original repeated record IDs and the actual final-batch denominator are
retained under the same sharing map, CM55. -/
theorem batchLoss_convex [Fintype C]
    (A : P → K → C → ℝ) (B : K → C → ℝ) (w : P → ℝ) (hw : ∀ p, 0 ≤ w p)
    (φ : K → X → ℝ) (inputs : S → X) (labels : S → C) (batch : Finset S) (R T : ℝ)
    (hA : ∀ p c, ∑ k, |A p k c| ≤ w p) (hB : B ∈ headBudget T) :
    ConvexOn ℝ (weightedDomain w R)
      (fun θ => Fallback.batchLoss φ inputs labels batch (parameterMap A B θ)) := by
  have h := loss_convex A B w hw φ (fun s : ↥batch => inputs s.val)
    (fun s : ↥batch => labels s.val) R T hA hB
  have hc : 0 ≤ (batch.card : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  simpa [Fallback.batchLoss, smul_eq_mul] using ConvexOn.smul hc h

end Transformer.Grokking.Convex.ConvexDepth.ParameterSharing
