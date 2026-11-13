module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.Limits
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.ComplexHead

/-!
# Derived readout grouping in the actual raw-vocabulary complex head

Source: `review/flat_spectrum.md`, CM35. Grouping changes the declared feature
construction, while observed-input refinement keeps the original raw labels
and full-vocabulary CE. A representative of each known answer supplies a
raw default. Accepted unseen queries have the correct pooled answer after
any finite CE schedule; their raw alias may still be wrong. The code and
target compatibility premises are not established by this head construction.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

open KernelCodes ConvexDepth

variable {R I J A D K S : Type*} [CommRing R] [Fintype R] [Fintype D]
variable {cell : S → I × J} {recorded : S → A} {i0 : I} {d0 : D}

/-- A declared raw representative preserves the identified original
answer, CM35. Choosing it does not identify which raw alias the target uses. -/
theorem grouped_representative_sound (q : A → D)
    (b : Module.Basis K R (observedKernel cell (q ∘ recorded) i0 d0))
    (lift : D → A) (hlift : ∀ d, q (lift d) = d)
    (e : A → R) (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R)
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = recorded t)
    (bank : List D) (fallback : D) (i : I) (j : J) {d : D}
    (hfound : chooseClass b (queryCode b i j) bank = some d) :
    q (lift (codeRepresentative b bank fallback (codeState b (i, j)))) = q (y i j) := by
  rw [representative_of_match b bank fallback i j hfound, hlift]
  exact (choose_readout_sound q b e hsurj hcompat hcard r s y hreal hobs
    i j bank hfound).symm

variable [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J]
variable [Fintype A] [DecidableEq A] [DecidableEq D]

/-- Actual real/imaginary Fourier coefficients preserve the correct
declared answer at an accepted unseen query, CM35. The bound and updates are
the existing raw-vocabulary head, with no grouped-label CE substitution. -/
theorem grouped_vectorCE_unseen_decoded (q : A → D)
    (b : Module.Basis K R (observedKernel cell (q ∘ recorded) i0 d0))
    (lift : D → A) (hlift : ∀ d, q (lift d) = d)
    (e : A → R) (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R)
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = recorded t)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (bank : List D) (fallback : D)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ)
    (hraw : Fintype.card A ≤ 1024) (i : I) (j : J) (hx : (i, j) ∉ observed)
    {d : D} (hfound : chooseClass b (queryCode b i j) bank = some d)
    {other : D} (hne : other ≠ q (y i j)) :
    quotientScore q (vectorCELogits b observed labels
      (fun z => lift (codeRepresentative b bank fallback z)) batches rates n (i, j)) other <
    quotientScore q (vectorCELogits b observed labels
      (fun z => lift (codeRepresentative b bank fallback z)) batches rates n (i, j))
        (q (y i j)) := by
  rw [vectorCE_unseen_logits b observed labels _ batches rates n hx]
  have hr := grouped_representative_sound q b lift hlift e hsurj hcompat hcard
    r s y hreal hobs bank fallback i j hfound
  rw [← hr] at hne ⊢
  exact quotientScore_margin_strict q _ 24 (native_margin_exceeds_vocabulary hraw) hne

/-- Exactly the three actual observed inputs in the strict witness, CM35. -/
def cornerObserved : Finset (Bool × Bool) :=
  {(false, false), (true, false), (false, true)}

/-- The raw observed labels, including two different aliases of answer one,
CM35. No unobserved target is an input to this head. -/
def cornerHeadLabels (x : ↥cornerObserved) : ZMod 2 × Bool :=
  if x.val.1 then (1, false) else if x.val.2 then (1, true) else (0, false)

/-- A deterministic raw representative of each declared answer, CM35. -/
def cornerLift (d : ZMod 2) : ZMod 2 × Bool := (d, false)

/-- The complex head sees exactly the three original raw observations,
including their distinct aliases, CM35. -/
theorem corner_head_binding : ∀ t, ∃ h : cornerCells t ∈ cornerObserved,
    cornerHeadLabels ⟨cornerCells t, h⟩ = cornerLabels t := by
  intro t
  fin_cases t <;> exact ⟨by decide, by decide⟩

/-- Alternating genuine minibatches of sizes one and two, CM35. -/
def cornerBatches (step : ℕ) : Finset ↥cornerObserved :=
  if step % 2 = 0 then {⟨(false, false), by decide⟩}
  else {⟨(true, false), by decide⟩, ⟨(false, true), by decide⟩}

/-- The very same complex head prefers the wrong unseen raw alias,
despite the correct pooled answer proved below, CM35. -/
theorem corner_complex_raw_miss :
    vectorCELogits cornerGroupedBasis cornerObserved cornerHeadLabels
      (fun z => cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0 z))
      (fun _ => Finset.univ) (fun _ => 1) 3 (true, true) (cornerTarget true true true) <
    vectorCELogits cornerGroupedBasis cornerObserved cornerHeadLabels
      (fun z => cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0 z))
      (fun _ => Finset.univ) (fun _ => 1) 3 (true, true) (0, false) := by
  rw [vectorCE_unseen_logits cornerGroupedBasis cornerObserved cornerHeadLabels _
    (fun _ => Finset.univ) (fun _ => 1) 3 (by decide : (true, true) ∉ cornerObserved)]
  have hp := representative_of_match cornerGroupedBasis [0, 1] 0 true true
    (by rw [corner_grouped_match]; exact chooseClass_first _ (0 : ZMod 2) _)
  norm_num [latentDefaultMargins, hp, cornerLift, cornerTarget]

example : ∃ h : cornerCells 1 ∈ cornerObserved,
    cornerHeadLabels ⟨cornerCells 1, h⟩ = cornerLabels 1 := corner_head_binding 1

example : Prod.fst (cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0
    (codeState cornerGroupedBasis (true, true)))) = (cornerTarget true true true).1 :=
  grouped_representative_sound Prod.fst cornerGroupedBasis cornerLift (fun _ => rfl)
    Prod.fst (fun d => ⟨(d, false), rfl⟩) (fun _ _ h => h) rfl cornerPhase cornerPhase
    (cornerTarget true) (corner_target_realized true) (corner_target_observed true)
    [0, 1] 0 true true (by rw [corner_grouped_match]; exact chooseClass_first _ _ _)

example := grouped_vectorCE_unseen_decoded Prod.fst cornerGroupedBasis cornerLift
  (fun _ => rfl) Prod.fst (fun d => ⟨(d, false), rfl⟩) (fun _ _ h => h) rfl
  cornerPhase cornerPhase (cornerTarget true) (corner_target_realized true)
  (corner_target_observed true) cornerObserved cornerHeadLabels [0, 1] 0
  (fun _ => Finset.univ) (fun _ => 1) 3 (by decide) true true (by decide)
  (d := 0) (by rw [corner_grouped_match]; exact chooseClass_first _ _ _)
  (other := 1) (by decide)

example := grouped_vectorCE_unseen_decoded Prod.fst cornerGroupedBasis cornerLift
  (fun _ => rfl) Prod.fst (fun d => ⟨(d, false), rfl⟩) (fun _ _ h => h) rfl
  cornerPhase cornerPhase (cornerTarget true) (corner_target_realized true)
  (corner_target_observed true) cornerObserved cornerHeadLabels [0, 1] 0
  cornerBatches (fun _ => 1) 4 (by decide) true true (by decide)
  (d := 0) (by rw [corner_grouped_match]; exact chooseClass_first _ _ _)
  (other := 1) (by decide)

example := grouped_vectorCE_unseen_decoded Prod.fst cornerGroupedBasis cornerLift
  (fun _ => rfl) Prod.fst (fun d => ⟨(d, false), rfl⟩) (fun _ _ h => h) rfl
  cornerPhase cornerPhase (cornerTarget true) (corner_target_realized true)
  (corner_target_observed true) cornerObserved cornerHeadLabels [0, 1] 0
  (fun _ => Finset.univ) (fun _ => 1) 0 (by decide) true true (by decide)
  (d := 0) (by rw [corner_grouped_match]; exact chooseClass_first _ _ _)
  (other := 1) (by decide)

example := vectorCE_budget cornerGroupedBasis cornerObserved cornerHeadLabels
  (fun z => cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0 z))
  (fun _ => Finset.univ) (fun _ => 1) 3

example := vectorCE_loss_convex cornerGroupedBasis cornerObserved
  (fun z => cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0 z))
  cornerCells cornerLabels

example := vectorCE_seen_ce cornerGroupedBasis cornerObserved cornerHeadLabels
  (fun z => cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0 z)) (by decide)
  (fun _ => Finset.univ) (fun _ => 1) (by intro n; norm_num) 3
  (x := (false, true)) (by decide)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
