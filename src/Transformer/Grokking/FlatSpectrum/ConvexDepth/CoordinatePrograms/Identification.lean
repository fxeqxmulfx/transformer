module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms.Head

/-!
# Observed feature-span identification and convex ordinary CE

Source: `review/flat_spectrum.md`, CM57. Observable linear-combination
certificates and a separate semantic prior identify actual query centers.
The complete affine head retains convex full/minibatch CE on the joint domain.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms

open scoped BigOperators

variable {P A C : Type*} [Fintype P]

/-- Observable feature-span coverage identifies the query center under an
unknown realizable coordinate decoder, CM57. Query labels never enter the
anchor constraints; realizability connects the derived coordinate to a target. -/
theorem query_identified [Fintype A] (θ : Fin 6 → ℝ) (rows : A → Fin 5 → ℝ) (values : A → ℝ)
    (β h : Fin 5 → ℝ) (v : ℝ) (weights : Option A → ℝ)
    (hcopy : θ 4 = θ 5) (hobs : ∀ a, center θ (rows a) = θ 5 * values a)
    (hβcopy : β 4 = 1) (hreal : ∀ a, values a = dot β (rows a))
    (hquery : v = dot β h)
    (hspan : ∀ j, h j = ∑ a, weights a * augmentedRows rows a j) :
    center θ h = θ 5 * v := by
  have hh : h = fun j => ∑ a, weights a * augmentedRows rows a j := funext hspan
  change dot (fun j => θ j.castSucc) h = θ 5 * v
  rw [hh, dot_combination]
  change (∑ a, weights a * center θ (augmentedRows rows a)) = θ 5 * v
  simp_rw [augmented_anchors θ rows values hcopy hobs]
  rw [hquery, hh, dot_combination, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a ha
  rw [augmented_realizable β rows values hβcopy hreal a]
  ring

/-- The actual local score gap follows from the identified center, CM57. This
algebraic lemma alone is not an unseen-label identification theorem. -/
theorem score_difference (e : C → ℝ) (θ : Fin 6 → ℝ) (h : Fin 5 → ℝ) (y c : C)
    (hc : center θ h = θ 5 * e y) :
    score e θ h y - score e θ h c = θ 5 * (e y - e c) ^ 2 := by
  unfold score
  rw [hc]
  ring

/-- Every positive component is identified from its observed span and prior;
the perspective budget annihilates all remaining components, CM57. -/
theorem all_centers_identified [Fintype A] (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (B : P → ℝ) {R δ : ℝ} {θ : P × Fin 6 → ℝ}
    (hθ : θ ∈ domain rows values B R δ) (h : P → Fin 5 → ℝ) (v : ℝ)
    (hcovered : ∀ p, 0 < θ (p, 5) → ∃ (β : Fin 5 → ℝ) (weights : Option A → ℝ),
      β 4 = 1 ∧ (∀ a, values p a = dot β (rows p a)) ∧ v = dot β (h p) ∧
      ∀ j, h p j = ∑ a, weights a * augmentedRows (rows p) a j) :
    ∀ p, center (fun j => θ (p, j)) (h p) = θ (p, 5) * v := by
  intro p
  by_cases hp : θ (p, 5) = 0
  · have hz := zero_mass (fun j => θ (p, j)) (B p) (hθ.2.2.2.1 p) hp
    simp [center, hz]
  · have hpos : 0 < θ (p, 5) := lt_of_le_of_ne (hθ.2.1 p) (Ne.symm hp)
    obtain ⟨β, weights, hβ, hr, hq, hs⟩ := hcovered p hpos
    exact query_identified _ (rows p) (values p) β (h p) v weights
      (hθ.2.2.2.2.1 p) (hθ.2.2.2.2.2 p) hβ hr hq hs

/-- Identified components share a strict all-raw-class winner when their total
mass is positive and the supplied coordinates are injective, CM57. -/
theorem mixture_strict (e : C → ℝ) (he : Function.Injective e)
    (θ : P × Fin 6 → ℝ) (h : P → Fin 5 → ℝ) (y c : C)
    (hg : 0 < ∑ p, θ (p, 5)) (hc : c ≠ y)
    (hid : ∀ p, center (fun j => θ (p, j)) (h p) = θ (p, 5) * e y) :
    (∑ p, score e (fun j => θ (p, j)) (h p) c) <
      ∑ p, score e (fun j => θ (p, j)) (h p) y := by
  have hgap : (∑ p, score e (fun j => θ (p, j)) (h p) y) -
      (∑ p, score e (fun j => θ (p, j)) (h p) c) =
      (∑ p, θ (p, 5)) * (e y - e c) ^ 2 := by
    rw [← Finset.sum_sub_distrib, Finset.sum_mul]
    exact Finset.sum_congr rfl (fun p _ => score_difference e _ (h p) y c (hid p))
  have hn : e y - e c ≠ 0 := sub_ne_zero.mpr (fun h => hc (he h).symm)
  have hp := mul_pos hg (sq_pos_of_ne_zero hn)
  rw [← hgap] at hp
  linarith

variable {X S : Type*} [DecidableEq P]

/-- Ordinary full-vocabulary CE is convex on the complete sufficient parameter
budget, with all masses and scaled decoder coefficients learned jointly, CM57. -/
theorem loss_convex [Fintype C] [Fintype S]
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (h : P → X → Fin 5 → ℝ)
    (inputs : S → X) (labels : S → C) (R : ℝ) :
    ConvexOn ℝ (ParameterSharing.weightedDomain (fun p : P × Fin 6 => costs p.2) R)
      (fun θ => headLoss (features h) inputs labels (head e θ)) :=
  ParameterSharing.loss_convex (expansion e) 0 (fun p : P × Fin 6 => costs p.2)
    (fun p => le_of_lt (costs_pos p.2)) (features h) inputs labels R 0
    (expansion_bound e he) (zero_mem_headBudget (le_refl (0 : ℝ)))

/-- Minibatch CE retains repeated original record IDs and divides by the
actual batch size, including a short tail, CM57. -/
theorem batchLoss_convex [Fintype C]
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (h : P → X → Fin 5 → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (R : ℝ) :
    ConvexOn ℝ (ParameterSharing.weightedDomain (fun p : P × Fin 6 => costs p.2) R)
      (fun θ => Fallback.batchLoss (features h) inputs labels batch (head e θ)) :=
  ParameterSharing.batchLoss_convex (expansion e) 0 (fun p : P × Fin 6 => costs p.2)
    (fun p => le_of_lt (costs_pos p.2)) (features h) inputs labels batch R 0
    (expansion_bound e he) (zero_mem_headBudget (le_refl (0 : ℝ)))

/-- Homogeneous observations, the copy prior and perspective restrictions
retain convex full CE on the actual joint domain, CM57. -/
theorem domain_loss_convex [Fintype C] [Fintype S]
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (h : P → X → Fin 5 → ℝ)
    (inputs : S → X) (labels : S → C) (rows : P → A → Fin 5 → ℝ)
    (values : P → A → ℝ) (B : P → ℝ) (R δ : ℝ) :
    ConvexOn ℝ (domain rows values B R δ)
      (fun θ => headLoss (features h) inputs labels (head e θ)) :=
  (loss_convex e he h inputs labels R).subset (fun _ hθ => hθ.1)
    (domain_convex rows values B R δ)

/-- The same joint domain retains convex ordinary minibatch CE, CM57. -/
theorem domain_batchLoss_convex [Fintype C]
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (h : P → X → Fin 5 → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ) (B : P → ℝ) (R δ : ℝ) :
    ConvexOn ℝ (domain rows values B R δ)
      (fun θ => Fallback.batchLoss (features h) inputs labels batch (head e θ)) :=
  (batchLoss_convex e he h inputs labels batch R).subset (fun _ hθ => hθ.1)
    (domain_convex rows values B R δ)

/-- A positive total mass gives a strict raw winner through the actual complex
product, when every active program has an observed span certificate and a
realizable unknown decoder. Inactive programs vanish by the proved budget
lemma; uncovered queries remain outside this conclusion, CM57. -/
theorem identified_prediction [Fintype A]
    (e : C → ℝ) (he : Function.Injective e) (h : P → X → Fin 5 → ℝ)
    (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ) (B : P → ℝ)
    {R δ : ℝ} (hδ : 0 < δ) {θ : P × Fin 6 → ℝ} (hθ : θ ∈ domain rows values B R δ)
    (x : X) (y c : C) (hc : c ≠ y)
    (hcovered : ∀ p, 0 < θ (p, 5) → ∃ (β : Fin 5 → ℝ) (weights : Option A → ℝ),
      β 4 = 1 ∧ (∀ a, values p a = dot β (rows p a)) ∧ e y = dot β (h p x) ∧
      ∀ j, h p x j = ∑ a, weights a * augmentedRows (rows p) a j) :
    headLogits (features h) (head e θ) x c < headLogits (features h) (head e θ) x y := by
  have hid := all_centers_identified rows values B hθ (fun p => h p x) (e y) hcovered
  rw [head_logits, head_logits]
  exact div_lt_div_of_pos_right
    (mixture_strict e he θ (fun p => h p x) y c (lt_of_lt_of_le hδ hθ.2.2.1) hc hid)
    (SignedContext.normalizer_pos _)

/-- Observed span certificates and semantic realizability derive the exact
gap of the actual complete complex head, including its common normalization.
Injectivity and positive mass are needed for strictness, not this identity,
CM57. -/
theorem identified_gap [Fintype A]
    (e : C → ℝ) (h : P → X → Fin 5 → ℝ)
    (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ) (B : P → ℝ)
    {R δ : ℝ} {θ : P × Fin 6 → ℝ} (hθ : θ ∈ domain rows values B R δ)
    (x : X) (y c : C)
    (hcovered : ∀ p, 0 < θ (p, 5) → ∃ (β : Fin 5 → ℝ) (weights : Option A → ℝ),
      β 4 = 1 ∧ (∀ a, values p a = dot β (rows p a)) ∧ e y = dot β (h p x) ∧
      ∀ j, h p x j = ∑ a, weights a * augmentedRows (rows p) a j) :
    headLogits (features h) (head e θ) x y - headLogits (features h) (head e θ) x c =
      ((∑ p, θ (p, 5)) * (e y - e c) ^ 2) / SignedContext.normalizer (bankRow h x) := by
  have hid := all_centers_identified rows values B hθ (fun p => h p x) (e y) hcovered
  rw [head_logits, head_logits, ← sub_div, ← Finset.sum_sub_distrib, Finset.sum_mul]
  congr 1
  exact Finset.sum_congr rfl (fun p _ => score_difference e _ (h p x) y c (hid p))

end Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms
