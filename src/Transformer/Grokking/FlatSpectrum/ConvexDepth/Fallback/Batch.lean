module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Domain
public import Mathlib.Analysis.Calculus.FDeriv.Const

/-!
# Original normalized CE batches under fixed support routing

Source: `review/flat_spectrum.md`, CM38. Batches contain original record IDs,
so distinct IDs retain repeated inputs and label multiplicity. On covered
training records the actual routed loss is the candidate's loss for every
coefficient tensor. Derivative equality follows from this function identity;
neither the parent nor the support predicate is differentiated.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback

variable {K L X C S : Type*} [Fintype K] [Fintype L] [Fintype C]

/-- Ordinary full-vocabulary CE averaged over the actual batch of original
record IDs, CM38. An empty batch has zero objective by the zero-inverse convention. -/
noncomputable def batchLoss (φ : K → X → ℝ) (inputs : S → X) (labels : S → C)
    (batch : Finset S) (W : K → C → ℝ) : ℝ :=
  (batch.card : ℝ)⁻¹ * headLoss φ (fun s : ↥batch => inputs s.val)
    (fun s : ↥batch => labels s.val) W

/-- The original batch normalization preserves convexity on the affine
parent-frozen domain, CM38, including a shorter final minibatch. -/
theorem batchLoss_convex (known : X → Bool) (candidate : K → X → ℝ) (previous : L → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (parent : L → C → ℝ) (R : ℝ) :
    ConvexOn ℝ (parentDomain parent R)
      (batchLoss (features known candidate previous) inputs labels batch) := by
  have h := loss_convex known candidate previous
    (fun s : ↥batch => inputs s.val) (fun s : ↥batch => labels s.val) parent R
  have hc : 0 ≤ (batch.card : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  refine ⟨parentDomain_convex parent R, ?_⟩
  intro W hW V hV a b ha hb hab
  have hm := mul_le_mul_of_nonneg_left (h.2 hW hV ha hb hab) hc
  change (batch.card : ℝ)⁻¹ * _ ≤ a * ((batch.card : ℝ)⁻¹ * _) +
    b * ((batch.card : ℝ)⁻¹ * _)
  calc
    _ ≤ (batch.card : ℝ)⁻¹ * (a * headLoss _ _ _ W + b * headLoss _ _ _ V) := hm
    _ = _ := by ring

/-- Every covered original record uses exactly the candidate's logits;
therefore its normalized ordinary CE is identical for all parameters, CM38.
This hypothesis concerns observed support, not unknown target correctness. -/
theorem batch_loss_eq (known : X → Bool) (candidate : K → X → ℝ) (previous : L → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, known (inputs s) = true)
    (W : K → C → ℝ) (parent : L → C → ℝ) :
    batchLoss (features known candidate previous) inputs labels batch (coefficients W parent) =
      batchLoss candidate inputs labels batch W := by
  unfold batchLoss headLoss
  congr 1
  apply Finset.sum_congr rfl
  intro s _
  rw [logits, ite_eq_left (covered s.val s.property)]

/-- Equality of the actual batch objectives transfers their Fréchet
derivatives in candidate coordinates, CM38. At differentiability points these
are the actual derivatives; elsewhere `fderiv` retains its library convention.
No derivative of the fixed route or parent is introduced. -/
theorem batch_fderiv_eq (known : X → Bool) (candidate : K → X → ℝ) (previous : L → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, known (inputs s) = true)
    (parent : L → C → ℝ) (W : K → C → ℝ) :
    fderiv ℝ (fun V => batchLoss (features known candidate previous) inputs labels batch
      (coefficients V parent)) W = fderiv ℝ (batchLoss candidate inputs labels batch) W := by
  have hfun : (fun V => batchLoss (features known candidate previous) inputs labels batch
      (coefficients V parent)) = batchLoss candidate inputs labels batch := by
    funext V
    exact batch_loss_eq known candidate previous inputs labels batch covered V parent
  rw [hfun]

/-- Actual derivative existence and the derivative itself transfer, CM38.
The ordinary candidate CE derivative can be reused without differentiating a
default or changing the batch denominator. -/
theorem batch_hasFDerivAt (known : X → Bool) (candidate : K → X → ℝ) (previous : L → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, known (inputs s) = true)
    (parent : L → C → ℝ) (W : K → C → ℝ) (D : (K → C → ℝ) →L[ℝ] ℝ)
    (hd : HasFDerivAt (batchLoss candidate inputs labels batch) D W) :
    HasFDerivAt (fun V => batchLoss (features known candidate previous) inputs labels batch
      (coefficients V parent)) D W := by
  have hfun : (fun V => batchLoss (features known candidate previous) inputs labels batch
      (coefficients V parent)) = batchLoss candidate inputs labels batch := by
    funext V
    exact batch_loss_eq known candidate previous inputs labels batch covered V parent
  rw [hfun]
  exact hd

private def knownExample (x : Fin 4) : Bool := decide (x ≠ 3)
private def inputExample (s : Fin 3) : Fin 4 := s.castSucc
private def labelExample (s : Fin 3) : Bool := decide (s = 1)
private def candidateFeaturesExample (k : Fin 2) (x : Fin 4) : ℝ :=
  (k.val + 1) * (x.val + 1)
private def parentFeaturesExample (k : Fin 2) (x : Fin 4) : ℝ :=
  (k.val + 1) * (x.val + 1) ^ 2

private theorem coveredExample (s : Fin 3) : knownExample (inputExample s) = true := by
  fin_cases s <;> decide

example := batchLoss_convex knownExample candidateFeaturesExample parentFeaturesExample
  inputExample labelExample (Finset.univ : Finset (Fin 3)) (fun _ _ => 1) 8

-- A full batch and the unequal two-record/one-record minibatches use their
-- original normalization. The coefficient values need not minimize CE.
example := batch_loss_eq knownExample candidateFeaturesExample parentFeaturesExample
  inputExample labelExample (Finset.univ : Finset (Fin 3)) (fun s _ => coveredExample s)
  (fun _ _ => 1) (fun _ _ => 2)

example := batch_loss_eq knownExample candidateFeaturesExample parentFeaturesExample
  inputExample labelExample ({0, 1} : Finset (Fin 3)) (fun s _ => coveredExample s)
  (fun _ _ => 1) (fun _ _ => 2)

example := batch_loss_eq knownExample candidateFeaturesExample parentFeaturesExample
  inputExample labelExample ({2} : Finset (Fin 3)) (fun s _ => coveredExample s)
  (fun _ _ => 1) (fun _ _ => 2)

example := batch_fderiv_eq knownExample candidateFeaturesExample parentFeaturesExample
  inputExample labelExample ({2} : Finset (Fin 3)) (fun s _ => coveredExample s)
  (fun _ _ => 2) (fun _ _ => 1)

-- A nonempty three-record batch with zero candidate features has constant
-- ordinary CE. This supplies a concrete derivative-existence premise.
example : HasFDerivAt (𝕜 := ℝ)
    (fun W : Fin 2 → Bool → ℝ => batchLoss
      (features knownExample (fun (_ : Fin 2) (_ : Fin 4) => 0) parentFeaturesExample)
      inputExample labelExample Finset.univ (coefficients W (fun _ _ => 2)))
    0 (fun _ _ => 1) := by
  apply batch_hasFDerivAt
    knownExample (fun (_ : Fin 2) (_ : Fin 4) => 0) parentFeaturesExample
    inputExample labelExample Finset.univ (fun s _ => coveredExample s)
  have hconst : batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
      inputExample labelExample Finset.univ =
      fun _ : Fin 2 → Bool → ℝ => batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
        inputExample labelExample Finset.univ 0 := by
    have hz (V : Fin 2 → Bool → ℝ) (x : Fin 4) :
        headLogits (fun (_ : Fin 2) (_ : Fin 4) => 0) V x = (fun _ : Bool => 0) := by
      funext c
      simp [headLogits]
    funext W
    simp only [batchLoss, headLoss, hz]
  rw [hconst]
  exact hasFDerivAt_const _ _

-- The zero-step forward state is the same candidate/parent pair; an empty
-- derivative batch has zero objective and supplies no learning claim.
example : batchLoss candidateFeaturesExample inputExample labelExample ∅ (fun _ _ => 1) = 0 := by
  simp [batchLoss, headLoss]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback
