module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PartialFallback.Basic

/-!
# Ordinary CE batches with a partial saved predecessor

Source: `review/flat_spectrum.md`, CM41. Original record IDs, repeated examples
and final minibatch denominators are retained. Availability is a fixed input
predicate. Supported training records use the candidate objective for every
coefficient tensor, so its actual derivatives transfer without new optimization.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.PartialFallback

variable {K L X C S : Type*} [Fintype K] [Fintype L] [Fintype C]

/-- Ordinary normalized raw-label CE is convex on the affine domain
freezing the saved parent, CM41. Both coefficient blocks retain their budget. -/
theorem batchLoss_convex (known available : X → Bool) (candidate : K → X → ℝ)
    (previous : L → {x // available x = true} → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (parent : L → C → ℝ) (R : ℝ) :
    ConvexOn ℝ (Fallback.parentDomain parent R)
      (Fallback.batchLoss (features known available candidate previous) inputs labels batch) := by
  exact Fallback.batchLoss_convex (useCandidate known available) candidate
    (zeroExtend available previous) inputs labels batch parent R

/-- Candidate-supported original records retain exactly their normalized
raw CE for all candidate coefficients, CM41. No parent value outside its domain
or correctness of a held-out answer is needed. -/
theorem batch_loss_eq (known available : X → Bool) (candidate : K → X → ℝ)
    (previous : L → {x // available x = true} → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, known (inputs s) = true)
    (W : K → C → ℝ) (parent : L → C → ℝ) :
    Fallback.batchLoss (features known available candidate previous) inputs labels batch
      (Fallback.coefficients W parent) = Fallback.batchLoss candidate inputs labels batch W := by
  apply Fallback.batch_loss_eq (useCandidate known available) candidate
    (zeroExtend available previous) inputs labels batch
  intro s hs
  simp [useCandidate, covered s hs]

/-- The complete Fréchet derivative in candidate coordinates is unchanged
on any supported original batch, CM41, including repeated records and short tails. -/
theorem batch_fderiv_eq (known available : X → Bool) (candidate : K → X → ℝ)
    (previous : L → {x // available x = true} → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, known (inputs s) = true)
    (parent : L → C → ℝ) (W : K → C → ℝ) :
    fderiv ℝ (fun V => Fallback.batchLoss (features known available candidate previous)
      inputs labels batch (Fallback.coefficients V parent)) W =
      fderiv ℝ (Fallback.batchLoss candidate inputs labels batch) W := by
  apply Fallback.batch_fderiv_eq (useCandidate known available) candidate
    (zeroExtend available previous) inputs labels batch
  intro s hs
  simp [useCandidate, covered s hs]

/-- Actual derivative existence transfers from the original candidate CE,
CM41. The fixed availability predicate is never differentiated. -/
theorem batch_hasFDerivAt (known available : X → Bool) (candidate : K → X → ℝ)
    (previous : L → {x // available x = true} → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, known (inputs s) = true)
    (parent : L → C → ℝ) (W : K → C → ℝ) (D : (K → C → ℝ) →L[ℝ] ℝ)
    (hd : HasFDerivAt (Fallback.batchLoss candidate inputs labels batch) D W) :
    HasFDerivAt (fun V => Fallback.batchLoss (features known available candidate previous)
      inputs labels batch (Fallback.coefficients V parent)) D W := by
  apply Fallback.batch_hasFDerivAt (useCandidate known available) candidate
    (zeroExtend available previous) inputs labels batch _ parent W D hd
  intro s hs
  simp [useCandidate, covered s hs]

private def knownExample (x : Fin 4) : Bool := decide (x.val < 2)
private def availableExample (x : Fin 4) : Bool := decide (x.val < 3)
private def inputExample (s : Fin 3) : Fin 4 := if s = 2 then 1 else 0
private def labelExample (s : Fin 3) : Bool := decide (s = 2)
private def phiExample (k : Fin 2) (x : Fin 4) : ℝ := k.val + x.val + 1
private def psiExample (k : Fin 2) (x : {x // availableExample x = true}) : ℝ :=
  2 * k.val + x.val.val + 1

private theorem coveredExample (s : Fin 3) : knownExample (inputExample s) = true := by
  fin_cases s <;> decide

example := batchLoss_convex knownExample availableExample phiExample psiExample
  inputExample labelExample (Finset.univ : Finset (Fin 3)) (fun _ _ => 2) 8

-- IDs 0 and 1 are distinct records of the same input and raw label. The full
-- batch and the two-record/one-record minibatches keep those original weights.
example := batch_loss_eq knownExample availableExample phiExample psiExample
  inputExample labelExample (Finset.univ : Finset (Fin 3))
  (fun s _ => coveredExample s) (fun _ _ => 1) (fun _ _ => 2)
example := batch_loss_eq knownExample availableExample phiExample psiExample
  inputExample labelExample ({0, 1} : Finset (Fin 3))
  (fun s _ => coveredExample s) (fun _ _ => 1) (fun _ _ => 2)
example := batch_loss_eq knownExample availableExample phiExample psiExample
  inputExample labelExample ({2} : Finset (Fin 3))
  (fun s _ => coveredExample s) (fun _ _ => 1) (fun _ _ => 2)
example := batch_fderiv_eq knownExample availableExample phiExample psiExample
  inputExample labelExample ({2} : Finset (Fin 3))
  (fun s _ => coveredExample s) (fun _ _ => 2) (fun _ _ => 1)

-- A nonempty raw two-class batch supplies an explicit derivative-existence
-- premise: zero candidate features give constant CE in its free coefficients.
example : HasFDerivAt (𝕜 := ℝ)
    (fun W : Fin 2 → Bool → ℝ => Fallback.batchLoss
      (features knownExample availableExample (fun (_ : Fin 2) (_ : Fin 4) => 0) psiExample)
      inputExample labelExample Finset.univ (Fallback.coefficients W (fun _ _ => 2)))
    0 (fun _ _ => 1) := by
  apply batch_hasFDerivAt knownExample availableExample
    (fun (_ : Fin 2) (_ : Fin 4) => 0) psiExample inputExample labelExample Finset.univ
    (fun s _ => coveredExample s)
  have hconst : Fallback.batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
      inputExample labelExample Finset.univ =
      fun _ : Fin 2 → Bool → ℝ => Fallback.batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
        inputExample labelExample Finset.univ 0 := by
    have hz (V : Fin 2 → Bool → ℝ) (x : Fin 4) :
        headLogits (fun (_ : Fin 2) (_ : Fin 4) => 0) V x = (fun _ : Bool => 0) := by
      funext c
      simp [headLogits]
    funext W
    simp only [Fallback.batchLoss, headLoss, hz]
  rw [hconst]
  exact hasFDerivAt_const _ _

end Transformer.Grokking.FlatSpectrum.ConvexDepth.PartialFallback
