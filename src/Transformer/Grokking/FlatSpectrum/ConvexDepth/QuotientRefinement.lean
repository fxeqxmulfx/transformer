module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientCollision

/-!
# Refine observed inputs without changing their learned quotient

Source: `review/flat_spectrum.md`, CM31. Observed input identities are extra
fixed categorical states. Their labels are readout assignments, never feature
inputs. Unobserved inputs retain the learned latent state and its declared
representative. The hypotheses certify compatibility of that representation;
this construction does not discover it or identify unseen raw alias labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

variable {X Z C D : Type*} [DecidableEq X]

/-- CM31's state extractor reads only observed input identities and a fixed
latent map. No observed or held-out label is an argument. -/
def refinedState (observed : Finset X) (latent : X → Z) (x : X) : ↥observed ⊕ Z :=
  if h : x ∈ observed then Sum.inl ⟨x, h⟩ else Sum.inr (latent x)

/-- Raw labels are assigned to observed states; a declared representative is
assigned to each latent state. This is the head's table, not its features. -/
def refinedLabels (observed : Finset X) (labels : ↥observed → C)
    (representative : Z → C) : ↥observed ⊕ Z → C := Sum.elim labels representative

/-- The explicit forward prediction before its complex-head realization,
CM31. There is no oracle label argument for unobserved inputs. -/
def refinedPrediction (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (x : X) : C :=
  refinedLabels observed labels representative (refinedState observed latent x)

/-- Dropping the observed-input identity recovers the old latent state,
so every old state-dependent logit map remains representable, CM31. -/
theorem refinedState_forget (observed : Finset X) (latent : X → Z) (x : X) :
    Sum.elim (fun s : ↥observed => latent s) id (refinedState observed latent x) =
      latent x := by
  by_cases hx : x ∈ observed <;> simp [refinedState, hx]

/-- Every old latent-state logit map has an explicit pullback to the
refined states. This is representability, not a claim about an unchanged
coefficient budget; the enlarged state count is charged below, CM31. -/
theorem refinedState_retains_logits (observed : Finset X) (latent : X → Z)
    (oldLogits : Z → C → ℝ) :
    ∃ expanded : (↥observed ⊕ Z) → C → ℝ, ∀ x,
      expanded (refinedState observed latent x) = oldLogits (latent x) := by
  refine ⟨fun s => oldLogits (Sum.elim (fun o : ↥observed => latent o) id s), ?_⟩
  intro x
  exact congrArg oldLogits (refinedState_forget observed latent x)

/-- Distinct observed inputs are now distinct features even if their latent
codes coincide. This removes the shared-feature obstruction of CM31. -/
theorem refinedState_separates (observed : Finset X) (latent : X → Z)
    {x y : X} (hx : x ∈ observed) (hy : y ∈ observed) (hne : x ≠ y) :
    refinedState observed latent x ≠ refinedState observed latent y := by
  simp only [refinedState, dite_eq_left hx, dite_eq_left hy, ne_eq, Sum.inl.injEq]
  intro h
  exact hne (congrArg Subtype.val h)

/-- Observed-input states and unobserved defaults occupy disjoint parts
of the fixed feature alphabet, even when their latent code agrees, CM31. -/
theorem refinedState_seen_ne_unseen (observed : Finset X) (latent : X → Z)
    {x y : X} (hx : x ∈ observed) (hy : y ∉ observed) :
    refinedState observed latent x ≠ refinedState observed latent y := by
  simp [refinedState, hx, hy]

/-- Every consistent recorded raw label is reproduced, including distinct
aliases with the same quotient code, CM31. -/
theorem refinedPrediction_seen (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) {x : X} (hx : x ∈ observed) :
    refinedPrediction observed latent labels representative x = labels ⟨x, hx⟩ := by
  simp [refinedPrediction, refinedState, refinedLabels, hx]

/-- Unobserved inputs keep the declared latent representative. Their raw
truth is not recovered by this equation, CM31. -/
theorem refinedPrediction_unseen (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) {x : X} (hx : x ∉ observed) :
    refinedPrediction observed latent labels representative x = representative (latent x) := by
  simp [refinedPrediction, refinedState, refinedLabels, hx]

/-- Unseen inputs in one latent fiber still share a raw prediction. The
refinement repairs observed collisions, not unseen alias identification. -/
theorem refinedPrediction_unseen_agree (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) {x y : X}
    (hx : x ∉ observed) (hy : y ∉ observed) (hxy : latent x = latent y) :
    refinedPrediction observed latent labels representative x =
      refinedPrediction observed latent labels representative y := by
  rw [refinedPrediction_unseen _ _ _ _ hx, refinedPrediction_unseen _ _ _ _ hy, hxy]

/-- If recorded raw labels and latent representatives have the prescribed
quotient, the refined predictions retain that quotient at every input. -/
theorem refinedPrediction_quotient (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (quotient : C → D) (code : Z → D)
    (hlabels : ∀ x : ↥observed, quotient (labels x) = code (latent x))
    (hrepresentative : ∀ z, quotient (representative z) = code z) (x : X) :
    quotient (refinedPrediction observed latent labels representative x) = code (latent x) := by
  by_cases hx : x ∈ observed
  · rw [refinedPrediction_seen _ _ _ _ hx]
    exact hlabels ⟨x, hx⟩
  · rw [refinedPrediction_unseen _ _ _ _ hx]
    exact hrepresentative (latent x)

/-- A separately identified quotient answer transfers to the refined head,
CM31. The identifying premise can come from CM30; it is not a CE consequence. -/
theorem refinedPrediction_decoded (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (quotient : C → D) (code : Z → D)
    (hlabels : ∀ x : ↥observed, quotient (labels x) = code (latent x))
    (hrepresentative : ∀ z, quotient (representative z) = code z)
    (target : X → C) (x : X) (hidentified : code (latent x) = quotient (target x)) :
    quotient (refinedPrediction observed latent labels representative x) = quotient (target x) := by
  exact (refinedPrediction_quotient observed latent labels representative quotient code
    hlabels hrepresentative x).trans hidentified

omit [DecidableEq X] in
/-- The extra states cost one per distinct observed input, CM31. This
explicit dimension enters the Fourier head budget and is not free capacity. -/
theorem refinedState_card [Fintype Z] (observed : Finset X) :
    Fintype.card (↥observed ⊕ Z) = observed.card + Fintype.card Z := by
  rw [Fintype.card_sum, Fintype.card_coe]

example := refinedState_forget ({false} : Finset Bool) (fun x => x) true

example := refinedState_retains_logits ({false} : Finset Bool) id
  (fun z c : Bool => if z = c then (1 : ℝ) else -1)

example := refinedState_separates (Finset.univ : Finset Bool) (fun _ => ())
  (x := false) (y := true) (by simp) (by simp) (by decide)

example := refinedState_seen_ne_unseen ({false} : Finset Bool) (fun _ => ())
  (x := false) (y := true) (by simp) (by simp)

example := refinedPrediction_seen ({false} : Finset Bool) (fun x => x)
  (fun x => x.val) id (x := false) (by simp)

example := refinedPrediction_unseen ({false} : Finset Bool) (fun x => x)
  (fun x => x.val) id (x := true) (by simp)

example := refinedPrediction_unseen_agree ({0} : Finset (Fin 3)) (fun _ => ())
  (fun _ => false) (fun _ => true) (x := 1) (y := 2) (by simp) (by simp) rfl

example := refinedPrediction_quotient ({false} : Finset Bool) (fun x => x)
  (fun x => x.val) id id id (fun _ => rfl) (fun _ => rfl) true

example := refinedPrediction_decoded ({false} : Finset Bool) (fun x => x)
  (fun x => x.val) id id id (fun _ => rfl) (fun _ => rfl) id true rfl

end Transformer.Grokking.FlatSpectrum.ConvexDepth
