module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientReadout

/-!
# Convex training constraints that retain the latent defaults

Source: `review/flat_spectrum.md`, CM31. Ordinary CE on observed inputs
does not constrain unobserved-state logits. Keep those declared logits by
affine equalities in the actual complex head coordinates. The domain stays
convex, contains the explicit initializer and preserves the decoded default
after every update that remains feasible. No optimizer convergence or
unproved feasibility of a native projected update is asserted here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

variable {X Z C D : Type*} [DecidableEq X] [Fintype Z] [Nonempty Z] [DecidableEq Z]
  [Fintype C] [DecidableEq C]

/-- A convex coefficient budget intersected with affine equations fixing
the latent-default logits. Observed-state logits remain trainable, CM31. -/
def refinedHeadDomain (observed : Finset X) (representative : Z → C) (a R : ℝ) :
    Set ((↥observed ⊕ Z) × Bool → C → ℝ) :=
  {W | W ∈ headBudget R ∧ ∀ z,
    headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z))) W (Sum.inr z) =
      fun c => if c = representative z then a else 0}

omit [Fintype C] [DecidableEq X] [DecidableEq Z] in
/-- Fixing the default logits introduces no product of learned parameters;
the complete constrained domain is jointly convex, CM31. -/
theorem refinedHeadDomain_convex (observed : Finset X) (representative : Z → C) (a R : ℝ) :
    Convex ℝ (refinedHeadDomain observed representative a R) := by
  intro W hW V hV α β hα hβ hab
  refine ⟨headBudget_convex R hW.1 hV.1 hα hβ hab, ?_⟩
  intro z
  rw [headLogits_linear, hW.2 z, hV.2 z]
  funext c
  change α * (if c = representative z then a else 0) +
    β * (if c = representative z then a else 0) = _
  rw [← add_mul, hab, one_mul]

/-- The explicit raw-interpolating complex initializer satisfies every
affine default constraint as well as the native budget, CM31. -/
theorem refinedHeadDomain_initial (observed : Finset X) (labels : ↥observed → C)
    (representative : Z → C) :
    complexCoefficients (refinedComplexHead observed labels representative 24) ∈
      refinedHeadDomain observed representative 24
        (64 * ((observed.card : ℝ) + (Fintype.card Z : ℝ))) := by
  refine ⟨refinedComplex_budget _ _ _, ?_⟩
  intro z
  exact complexInitializer_logits _ (categoricalFourierOn_unitary _) _ _ (Sum.inr z)

omit [Fintype C] [DecidableEq X] [DecidableEq Z] in
/-- A convex update preserves the default equations when both endpoints
are feasible. This does not assert feasibility of a gradient proposal. -/
theorem refinedHeadDomain_segment (observed : Finset X) (representative : Z → C) (a R : ℝ)
    {W V : (↥observed ⊕ Z) × Bool → C → ℝ}
    (hW : W ∈ refinedHeadDomain observed representative a R)
    (hV : V ∈ refinedHeadDomain observed representative a R)
    {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η ≤ 1) :
    (1 - η) • W + η • V ∈ refinedHeadDomain observed representative a R := by
  exact refinedHeadDomain_convex _ _ _ _ hW hV (sub_nonneg.mpr hη1) hη0 (by ring)

omit [DecidableEq Z] in
/-- Ordinary full-vocabulary CE remains convex on the constrained domain.
The input/state map and representatives are fixed before training, CM31. -/
theorem refinedHeadDomain_loss_convex {S : Type*} [Fintype S]
    (observed : Finset X) (latent : X → Z) (representative : Z → C) (a R : ℝ)
    (inputs : S → X) (labels : S → C) :
    ConvexOn ℝ (refinedHeadDomain observed representative a R)
      (headLoss (complexFeatures (categoricalFourierOn (↥observed ⊕ Z)))
        (fun s => refinedState observed latent (inputs s)) labels) := by
  refine ⟨refinedHeadDomain_convex _ _ _ _, ?_⟩
  intro W hW V hV α β hα hβ hab
  exact (headLoss_convex _ _ _ R).2 hW.1 hV.1 hα hβ hab

omit [Fintype C] [DecidableEq Z] in
/-- Every feasible updated head retains the actual default logits at an
unobserved input. This guarantee is stronger than one initializer snapshot. -/
theorem refinedHeadDomain_unseen_logits (observed : Finset X) (latent : X → Z)
    (representative : Z → C) {a R : ℝ} {W : (↥observed ⊕ Z) × Bool → C → ℝ}
    (hW : W ∈ refinedHeadDomain observed representative a R)
    {x : X} (hx : x ∉ observed) :
    headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z))) W
      (refinedState observed latent x) =
        fun c => if c = representative (latent x) then a else 0 := by
  simpa [refinedState, hx] using hW.2 (latent x)

omit [DecidableEq Z] in
/-- If the fixed representative is compatible with the learned quotient,
every feasible updated head keeps its pooled readout on unseen inputs. -/
theorem refinedHeadDomain_unseen_pooled [DecidableEq D]
    (observed : Finset X) (latent : X → Z) (representative : Z → C)
    (quotient : C → D) (code : Z → D)
    (hrepresentative : ∀ z, quotient (representative z) = code z)
    (hcard : Fintype.card C ≤ 1024) {R : ℝ} {W : (↥observed ⊕ Z) × Bool → C → ℝ}
    (hW : W ∈ refinedHeadDomain observed representative 24 R)
    {x : X} (hx : x ∉ observed) {d : D} (hd : d ≠ code (latent x)) :
    quotientScore quotient
      (headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z))) W
        (refinedState observed latent x)) d <
    quotientScore quotient
      (headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z))) W
        (refinedState observed latent x)) (code (latent x)) := by
  rw [refinedHeadDomain_unseen_logits _ _ _ hW hx]
  rw [← hrepresentative (latent x)] at hd ⊢
  exact quotientScore_margin_strict _ _ _ (native_margin_exceeds_vocabulary hcard) hd

omit [DecidableEq Z] in
/-- A separately identified decoded target remains correct for any feasible
iterate, without an assumption that the optimizer reaches a minimum, CM31. -/
theorem refinedHeadDomain_unseen_decoded [DecidableEq D]
    (observed : Finset X) (latent : X → Z) (representative : Z → C)
    (quotient : C → D) (code : Z → D)
    (hrepresentative : ∀ z, quotient (representative z) = code z)
    (hcard : Fintype.card C ≤ 1024) {R : ℝ} {W : (↥observed ⊕ Z) × Bool → C → ℝ}
    (hW : W ∈ refinedHeadDomain observed representative 24 R)
    (target : X → C) {x : X} (hx : x ∉ observed)
    (hidentified : code (latent x) = quotient (target x)) {d : D}
    (hd : d ≠ quotient (target x)) :
    quotientScore quotient
      (headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z))) W
        (refinedState observed latent x)) d <
    quotientScore quotient
      (headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z))) W
        (refinedState observed latent x)) (quotient (target x)) := by
  rw [← hidentified] at hd ⊢
  exact refinedHeadDomain_unseen_pooled _ _ _ _ _ hrepresentative hcard hW hx hd

example := refinedHeadDomain_initial ({false} : Finset Bool) (fun x => x.val) id

example := refinedHeadDomain_segment ({false} : Finset Bool) id 24 192
  (by have h := refinedHeadDomain_initial ({false} : Finset Bool) (fun x => x.val) id
      norm_num at h
      exact h)
  (by have h := refinedHeadDomain_initial ({false} : Finset Bool) (fun x => !x.val) id
      norm_num at h
      exact h)
  (η := 1 / 2) (by norm_num) (by norm_num)

example := refinedHeadDomain_unseen_logits ({false} : Finset Bool) id id
  (refinedHeadDomain_initial ({false} : Finset Bool) (fun x => x.val) id)
  (x := true) (by simp)

example := refinedHeadDomain_unseen_pooled ({false} : Finset Bool) id id id id
  (fun _ => rfl) (by decide)
  (refinedHeadDomain_initial ({false} : Finset Bool) (fun x => x.val) id)
  (x := true) (by simp) (d := false) (by decide)

example := refinedHeadDomain_unseen_decoded ({false} : Finset Bool) id id id id
  (fun _ => rfl) (by decide)
  (refinedHeadDomain_initial ({false} : Finset Bool) (fun x => x.val) id)
  id (x := true) (by simp) rfl (d := false) (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth
