module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Compact
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RefinedCE

/-!
# Full-kernel states in the bounded complex CE model

Source: `review/flat_spectrum.md`, CM34. Instantiate the actual block Fourier
head with the finite range of full query vectors. Both the latent defaults
and the observed-input refinement are charged to the budget. Ordinary CE
remains convex after freezing the code construction. Unseen decoded
correctness requires an accepted match and the explicit target assumptions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

open ConvexDepth Matrix

variable {R I J C K A D S : Type*} [CommRing R]
variable [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J]
variable [Fintype A] [DecidableEq A]
variable {M : Submodule R (Coordinates I J C → R)}

/-- Actual complex forward multiplication after any finite table-CE
schedule, using the full query code as its latent state, CM34. -/
noncomputable def vectorCELogits (b : Module.Basis K R M)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (representative : CodeStates b → A)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ) (x : I × J) : A → ℝ :=
  headLogits (complexFeatures (fiberFourier (refinedBlock observed (codeState b))))
    (complexCoefficients
      (refinedCEHead observed (codeState b) labels representative batches rates n))
    (refinedState observed (codeState b) x)

/-- The finite-range bound controls the actual real/imaginary coefficient
budget, CM34. It charges all observations and at most the active input product. -/
theorem vectorCE_budget (b : Module.Basis K R M)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (representative : CodeStates b → A)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ) :
    complexCoefficients
      (refinedCEHead observed (codeState b) labels representative batches rates n) ∈
      headBudget (64 * ((observed.card : ℝ) + Fintype.card I * Fintype.card J)) := by
  have h := refinedCE_budget observed (codeState b) labels representative batches rates n
  intro a
  exact (h a).trans (codeState_budget_bound b observed)

/-- The ordinary raw-label CE objective is convex on the feasible table
domain after freezing the full-kernel basis and its input map, CM34. This
does not claim joint convexity of code discovery and table optimization. -/
theorem vectorCE_loss_convex {T : Type*} [Fintype T] (b : Module.Basis K R M)
    (observed : Finset (I × J)) (representative : CodeStates b → A)
    (inputs : T → I × J) (labels : T → A) :
    ConvexOn ℝ (defaultTableDomain 32 (latentDefaultMargins representative))
      (fun V => headLoss (complexFeatures (fiberFourier (refinedBlock observed (codeState b))))
        (fun t => refinedState observed (codeState b) (inputs t)) labels
        (complexCoefficients
          (tableHead (fiberFourier (refinedBlock observed (codeState b))) V))) := by
  exact refinedBox_loss_convex observed (codeState b) representative inputs labels

/-- Full-batch or minibatch CE updates preserve every declared unseen
default exactly, CM34. Arbitrary defaults are not promoted to correct answers. -/
theorem vectorCE_unseen_logits (b : Module.Basis K R M)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (representative : CodeStates b → A)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ)
    {x : I × J} (hx : x ∉ observed) :
    vectorCELogits b observed labels representative batches rates n x =
      latentDefaultMargins representative (codeState b x) := by
  exact refinedCE_unseen_logits observed (codeState b) labels representative batches rates n hx

/-- Every observed raw class remains a strict winner after any finite
nonnegative-rate full/minibatch schedule with the complete code, CM34. -/
theorem vectorCE_seen_winner (b : Module.Basis K R M)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (representative : CodeStates b → A)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (hr : ∀ n, 0 ≤ rates n)
    (n : ℕ) {x : I × J} (hx : x ∈ observed) {a : A} (ha : a ≠ labels ⟨x, hx⟩) :
    vectorCELogits b observed labels representative batches rates n x a <
      vectorCELogits b observed labels representative batches rates n x (labels ⟨x, hx⟩) := by
  exact refinedCE_seen_winner observed (codeState b) labels representative batches rates hr n hx ha

/-- Ordinary full-vocabulary CE at each observed input stays below log2
through the same finite schedule, CM34. No held-out loss conclusion is inferred. -/
theorem vectorCE_seen_ce (b : Module.Basis K R M)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (representative : CodeStates b → A) (hcard : Fintype.card A ≤ 1024)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (hr : ∀ n, 0 ≤ rates n)
    (n : ℕ) {x : I × J} (hx : x ∈ observed) :
    Landscape.Neural.crossEntropy
      (vectorCELogits b observed labels representative batches rates n x) (labels ⟨x, hx⟩) <
        Real.log 2 := by
  exact refinedCE_seen_ce observed (codeState b) labels representative hcard batches rates hr n hx

/-- A checked vector match plus the unknown-code task assumptions makes
the actual complex head's pooled readout correct at an unobserved input,
CM34. No guessed answer or post-training evaluation enters the model. -/
theorem vectorCE_unseen_decoded [DecidableEq D]
    {cell : S → I × J} {recorded : S → C} {i0 : I} {c0 : C}
    (b : Module.Basis K R (observedKernel cell recorded i0 c0))
    (name : C → A) (e : A → R) (q : A → D)
    (hq : ∀ a a', e a = e a' → q a = q a')
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (recorded t))
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (bank : List C) (fallback : C)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ)
    (hcard : Fintype.card A ≤ 1024) (i : I) (j : J) (hx : (i, j) ∉ observed)
    {c : C} (hfound : chooseClass b (queryCode b i j) bank = some c)
    {d : D} (hd : d ≠ q (y i j)) :
    quotientScore q (vectorCELogits b observed labels
      (fun z => name (codeRepresentative b bank fallback z)) batches rates n (i, j)) d <
    quotientScore q (vectorCELogits b observed labels
      (fun z => name (codeRepresentative b bank fallback z)) batches rates n (i, j))
        (q (y i j)) := by
  rw [vectorCE_unseen_logits b observed labels _ batches rates n hx]
  have hr := representative_readout_sound b name e q hq r s y hreal hobs bank fallback i j hfound
  rw [← hr] at hd ⊢
  exact quotientScore_margin_strict q _ 24 (native_margin_exceeds_vocabulary hcard) hd

example := vectorCE_budget diagonalBasis ({(false, false)} : Finset (Bool × Bool))
  (fun _ => false) (codeRepresentative diagonalBasis [false, true] false)
  (fun _ => Finset.univ) (fun _ => 1) 7

example := vectorCE_loss_convex diagonalBasis ({(false, false)} : Finset (Bool × Bool))
  (codeRepresentative diagonalBasis [false, true] false) (fun b : Bool => (b, b)) id

example := vectorCE_unseen_logits diagonalBasis ({(false, false)} : Finset (Bool × Bool))
  (fun _ => false) (codeRepresentative diagonalBasis [false, true] false)
  (fun _ => Finset.univ) (fun _ => 1) 7 (x := (true, false)) (by simp)

example := vectorCE_seen_winner diagonalBasis ({(false, false)} : Finset (Bool × Bool))
  (fun _ => false) (codeRepresentative diagonalBasis [false, true] false)
  (fun _ => Finset.univ) (fun _ => 1) (by intro n; norm_num) 7
  (x := (false, false)) (by simp) (a := true) (by decide)

example := vectorCE_seen_ce diagonalBasis ({(false, false)} : Finset (Bool × Bool))
  (fun _ => false) (codeRepresentative diagonalBasis [false, true] false) (by decide)
  (fun _ => Finset.univ) (fun _ => 1) (by intro n; norm_num) 7
  (x := (false, false)) (by simp)

open DecodedConsequences (aliasPhases aliasTarget alias_observed)

/-- A concrete original observed subset for the positive unseen-head witness, CM34. -/
def aliasHeadObserved : Finset (Fin 3 × Fin 3) := {(0, 0), (1, 0)}

/-- The original raw labels at the two supplied observed inputs, CM34. -/
def aliasHeadLabels (x : ↥aliasHeadObserved) : ZMod 2 × Bool :=
  aliasTarget true x.val.1 x.val.2

example := vectorCE_unseen_decoded aliasBasis id Prod.fst Prod.fst (fun _ _ h => h)
  aliasPhases aliasPhases (aliasTarget true) (alias_observed.2.2.2 true)
  (alias_observed.2.2.1 true) aliasHeadObserved aliasHeadLabels [(0, false)] (1, true)
  (fun _ => Finset.univ) (fun _ => 1) 7 (by decide) 2 2
  (by decide) (c := (0, false)) (by rw [alias_match]; exact chooseClass_first _ _ [])
  (d := 1) (by decide)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

