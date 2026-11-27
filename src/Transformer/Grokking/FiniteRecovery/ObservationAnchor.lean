module

public import Transformer.Grokking.FiniteRecovery.ArgandCandidateSelection
public import Transformer.Grokking.FiniteRecovery.HiddenProbability

/-! An observation-only reference for quadratic decay on native Argand weights.
A candidate must have strict predictions, belong to the prescribed admissible
class and fit the observed graph. Enumerating rational coordinates is an
existence construction when an efficient class-specific solver is unavailable. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set
open scoped Classical

variable {A : Type*} [Fintype A] [Nonempty A] {H d L T : ℕ}

/-- Only predictions and the supplied graph are inspected. -/
def fitsFamily (r : ℕ) (family : Set (Word A (T+1) → A))
    (part : Finset (Word A (T+1) × A)) (w : Weights A H d L) : Prop :=
  fitsObserved r part w ∧ (Model.fromWeights w).predictedToken (T := T) r ∈ family

/-- A reference vector computed from the known class and the revealed
labels. No hidden label or representing weights are arguments. -/
noncomputable def observationAnchor (r : ℕ) (family : Set (Word A (T+1) → A))
    (part : Finset (Word A (T+1) × A)) : Weights A H d L :=
  let ok := fun n => fitsFamily r family part (rationalCandidate (0 : Weights A H d L) n)
  if h : ∃ n, ok n then rationalCandidate 0 (Nat.find h) else 0

/-- Every class-consistent reference has the correct hidden predictions
when the observation map is injective on the class. This applies to any
way of finding such weights, irrespective of computational cost. -/
theorem realizes_of_fitsFamily (r : ℕ) (family : Set (Word A (T+1) → A))
    (S : Finset (Word A (T+1))) (hid : InjOn (observe S) family)
    {f : Word A (T+1) → A} (hf : f ∈ family) {w : Weights A H d L}
    (hw : fitsFamily r family (observe S f) w) : (Model.fromWeights w).realizes r f := by
  let g := (Model.fromWeights w).predictedToken (T := T) r
  have he : g = f := hid hw.2 hf (observe_eq_iff_eqOn.mpr fun x hx =>
    hw.1.2 (x,f x) (mem_observe.mpr ⟨hx,rfl⟩))
  exact he ▸ hw.1.1

/-- Strict representability guarantees that enumeration reaches an open
set of fitting candidates. Identifiability then selects the whole target. -/
theorem observationAnchor_realizes (r : ℕ) (family : Set (Word A (T+1) → A))
    (S : Finset (Word A (T+1))) (hid : InjOn (observe S) family)
    {f : Word A (T+1) → A} (hf : f ∈ family)
    (hr : ∃ w : Weights A H d L, (Model.fromWeights w).realizes r f) :
    (Model.fromWeights (observationAnchor (H := H) (d := d) (L := L) r family (observe S f))).realizes r f := by
  obtain ⟨n,hn⟩ := (exists_rationalCandidate_iff (0 : Weights A H d L) r f).2 hr
  let w := rationalCandidate (0 : Weights A H d L) n
  have he : (Model.fromWeights w).predictedToken (T := T) r = f :=
    (Model.fromWeights w).predictedToken_eq_of_realizes r hn
  have hex : ∃ j, fitsFamily r family (observe S f) (rationalCandidate (0 : Weights A H d L) j) := by
    refine ⟨n,?_,?_⟩
    · change (Model.fromWeights w).realizes r ((Model.fromWeights w).predictedToken r) ∧ _
      rw [he]
      exact ⟨hn,fun _ hp => (mem_observe.mp hp).2⟩
    · simpa only [show rationalCandidate (0 : Weights A H d L) n = w from rfl, he] using hf
  apply realizes_of_fitsFamily r family S hid hf
  simpa only [observationAnchor, dite_eq_left hex] using Nat.find_spec hex

end Transformer.Grokking.FiniteRecovery.Argand
