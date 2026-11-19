module

public import Transformer.Grokking.Convex.ConvexChannel.SharedLearning.Basic

/-!
# Shared routing and output identification from answer CE

Source: `review/convex/moments.md`, CM73. Wrong slots must have observed
same-value/different-label collisions; the target slot must cover its value
alphabet. These predicates concern the actual observations, not the optimizer's
parameters or its unknown predictions. Under realizability they identify the
entire joint tensor from empirical CE optimality. Query sharing then transfers
the result to all aligned dictionaries, including unobserved combinations.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.SharedLearning

open scoped BigOperators

variable {Q K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]
  [Nonempty V] [DecidableEq K] [DecidableEq C] [Nontrivial C]

/-- Actual observations cover the selected slot's categorical values, CM73. -/
def Covers (data : S → K → V) (slot : K) : Prop :=
  ∀ v, ∃ s, data s slot = v

/-- Each other slot has an observed collision with different answers, CM73.
No success or target prediction is part of this finite-data predicate. -/
def SeparatesSlots (data : S → K → V) (label : S → C) (slot : K) : Prop :=
  ∀ k, k ≠ slot → ∃ s t, data s k = data t k ∧ label s ≠ label t

/-- All globally optimal shared tensors equal the unknown selected routing
and decoder when observations identify them, CM73. Every unknown joint entry
is determined through the declared sharing, not through a hard label anchor. -/
theorem eq_selected_of_loss_le (data : S → K → V) (label : S → C)
    (slot : K) (decode : V → C) (hlabel : ∀ s, label s = decode (data s slot))
    (hcover : Covers data slot) (hsep : SeparatesSlots data label slot)
    {A : Channel Unit K V C} (hA : Feasible A)
    (hmin : answerLoss (fun _ : S => ()) data label A ≤
      answerLoss (fun _ : S => ()) data label (selected slot decode)) :
    A = selected slot decode := by
  have hm := observed_mass_one data label slot decode hlabel hA hmin
  have hoff (k : K) (hk : k ≠ slot) (v : V) (c : C) : A () k v c = 0 := by
    obtain ⟨s, t, hv, hc⟩ := hsep k hk
    exact collision_zero hA (data s) (data t) (label s) (label t)
      (hm s) (hm t) k hv hc v c
  have hon (v : V) : A () slot v (decode v) = 1 := by
    obtain ⟨s, hs⟩ := hcover v
    have h := hm s
    change (∑ k, A () k (data s k) (label s)) = 1 at h
    rw [Finset.sum_eq_single slot (fun k _ hk => hoff k hk _ _) (by simp),
      hlabel s, hs] at h
    exact h
  have hwrong (v : V) (c : C) (hc : c ≠ decode v) : A () slot v c = 0 := by
    obtain ⟨s, hs⟩ := hcover v
    have hcl : c ≠ label s := by simpa [hlabel s, hs] using hc
    simpa only [hs] using atom_eq_zero hA () (data s) (label s) (hm s) slot c hcl
  funext q k v c
  cases q
  by_cases hk : k = slot
  · subst k
    by_cases hc : c = decode v
    · subst c
      simpa [selected, Forward.channelOf] using hon v
    · simpa [selected, Forward.channelOf, hc] using hwrong v c hc
  · simpa [selected, Forward.channelOf, hk] using hoff k hk v c

/-- An optimum of the actual shared answer loss predicts the selected rule
on every aligned dictionary, CM73. The query and full input combination may
both be absent from training. The alignment remains a stated input-only prior. -/
theorem trained_shared_logits (align : Q → K ≃ K) (query : S → Q)
    (data : S → K → V) (label : S → C) (slot : K) (decode : V → C)
    (hlabel : ∀ s, label s = decode (alignedData align (query s) (data s) slot))
    (hcover : Covers (fun s => alignedData align (query s) (data s)) slot)
    (hsep : SeparatesSlots (fun s => alignedData align (query s) (data s)) label slot)
    {A : Channel Unit K V C} (hA : Feasible A)
    (hmin : IsMinOn (fun B => answerLoss query data label (tiedChannel align B))
      {B | Feasible B} A) (q : Q) (D : K → V) (c : C) :
    logits (tiedChannel align A) q D c =
      if c = decode (alignedData align q D slot) then 1 else 0 := by
  have hl := hmin (feasible_selected slot decode)
  simp only [answerLoss_tiedChannel] at hl
  have he := eq_selected_of_loss_le _ label slot decode hlabel hcover hsep hA hl
  rw [he, logits_tiedChannel, logits_selected]

/-- Identification yields a strict correct raw class, CM73. It is distinct
from a convergence claim about any minibatch algorithm or floating-point loop. -/
theorem trained_shared_strict (align : Q → K ≃ K) (query : S → Q)
    (data : S → K → V) (label : S → C) (slot : K) (decode : V → C)
    (hlabel : ∀ s, label s = decode (alignedData align (query s) (data s) slot))
    (hcover : Covers (fun s => alignedData align (query s) (data s)) slot)
    (hsep : SeparatesSlots (fun s => alignedData align (query s) (data s)) label slot)
    {A : Channel Unit K V C} (hA : Feasible A)
    (hmin : IsMinOn (fun B => answerLoss query data label (tiedChannel align B))
      {B | Feasible B} A) (q : Q) (D : K → V) (c : C)
    (hc : c ≠ decode (alignedData align q D slot)) :
    logits (tiedChannel align A) q D c <
      logits (tiedChannel align A) q D (decode (alignedData align q D slot)) := by
  rw [trained_shared_logits align query data label slot decode hlabel hcover hsep hA hmin,
    trained_shared_logits align query data label slot decode hlabel hcover hsep hA hmin]
  simp [hc]

end Transformer.Grokking.Convex.ConvexChannel.SharedLearning
