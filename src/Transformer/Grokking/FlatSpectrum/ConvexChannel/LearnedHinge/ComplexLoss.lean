module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Complex

/-!
# Convex ordinary CE for the actual shared complex ReLU computation

Source: `review/flat_spectrum.md`, CM75. Both real and imaginary activation
conditions are kept. Every object sequence uses the same learned complex
representations and readouts. Convexity applies on this declared domain,
not on unconstrained complex factor weights or freely changing masks.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The two actual components of an input read from a complex representation,
CM75. `true` selects real and `false` selects imaginary. -/
def components (input : X → E →ₗ[ℝ] ℂ) (x : X × Bool) : E →ₗ[ℝ] ℝ :=
  if x.2 then Complex.reLm.comp (input x.1) else Complex.imLm.comp (input x.1)

/-- All complex representation coordinates obey both activation conditions,
CM75. This is the actual intersection of input-only half-spaces. -/
def complexDomain (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) : Set (Parameters J C E) :=
  domain (components input) {x | x.1 ∈ region} mask

/-- The complete complex activation domain is convex, CM75. -/
theorem complexDomain_convex (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) : Convex ℝ (complexDomain input region mask : Set (Parameters J C E)) :=
  domain_convex _ _ _

variable [Fintype J]

/-- The actual complete complex logit retains the two component contributions
on the declared input cone, CM75. Neither contribution is dropped. -/
theorem complexLogits_on_cone (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {p : Parameters J C E}
    (hp : p ∈ complexDomain input region mask) {x : X} (hx : x ∈ region) :
    complexLogits p (input x) =
      gatedForward (fun j => mask j (x, true)) (components input (x, true)) p -
      gatedForward (fun j => mask j (x, false)) (components input (x, false)) p := by
  rw [complexLogits_eq]
  change forward p (components input (x, true)) - forward p (components input (x, false)) = _
  rw [forward_eq (components input) _ mask hp (x := (x, true)) hx,
    forward_eq (components input) _ mask hp (x := (x, false)) hx]

/-- Parameter combinations pass through the actual complex network at every
admitted input, CM75. Shared representation and readout are both recovered
anew from the combined point. -/
theorem complexLogits_combination (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {p q : Parameters J C E}
    (hp : p ∈ complexDomain input region mask) (hq : q ∈ complexDomain input region mask)
    {x : X} (hx : x ∈ region) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    complexLogits (a • p + b • q) (input x) =
      a • complexLogits p (input x) + b • complexLogits q (input x) := by
  have hm := complexDomain_convex input region mask hp hq ha hb hab
  rw [complexLogits_on_cone input region mask hm hx,
    complexLogits_on_cone input region mask hp hx,
    complexLogits_on_cone input region mask hq hx,
    gatedForward_combination, gatedForward_combination]
  funext c
  simp only [Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  ring

variable [Fintype C]

/-- Ordinary full-vocabulary CE through the actual complex computation,
CM75. The target is an output object, never an internal activation label. -/
noncomputable def complexLoss (input : E →ₗ[ℝ] ℂ) (target : C)
    (p : Parameters J C E) : ℝ := crossEntropy (complexLogits p input) target

/-- Every possible next-object CE is jointly convex on the actual complex
domain at every admitted input, CM75. -/
theorem complexLoss_convex (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {x : X} (hx : x ∈ region) (target : C) :
    ConvexOn ℝ (complexDomain input region mask : Set (Parameters J C E))
      (complexLoss (input x) target) := by
  refine ⟨complexDomain_convex input region mask, ?_⟩
  intro p hp q hq a b ha hb hab
  simp only [complexLoss, complexLogits_combination input region mask hp hq hx a b ha hb hab]
  exact (ConvexRecall.convex_crossEntropy target).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- The actual complex next-object distribution is strictly positive,
CM75. This is the same softmax used by ordinary CE. -/
theorem complexProbability_pos (p : Parameters J C E) (input : E →ₗ[ℝ] ℂ) (c : C) :
    0 < Geometry.probability (complexLogits p input) c := Geometry.probability_pos _ _

/-- The actual complex next-object distribution retains and normalizes
every original class, CM75. -/
theorem complexProbability_sum (p : Parameters J C E) (input : E →ₗ[ℝ] ℂ) (witness : C) :
    ∑ c, Geometry.probability (complexLogits p input) c = 1 := Geometry.sum_probability _ witness

/-- A complex minibatch retains the original record indices, CM75. -/
noncomputable def complexBatchLoss {N : ℕ} (input : X → E →ₗ[ℝ] ℂ)
    (data : Fin N → X) (target : Fin N → C) (p : Parameters J C E) : ℝ :=
  ∑ i, complexLoss (input (data i)) (target i) p

/-- All indexed complex minibatches, including short tails and repeated
records, have a convex ordinary CE objective on the same domain, CM75. -/
theorem complexBatchLoss_convex {N : ℕ} (input : X → E →ₗ[ℝ] ℂ)
    (region : Set X) (mask : J → X × Bool → Bool) (data : Fin N → X) (target : Fin N → C)
    (hdata : ∀ i, data i ∈ region) :
    ConvexOn ℝ (complexDomain input region mask : Set (Parameters J C E))
      (complexBatchLoss input data target) := by
  refine ⟨complexDomain_convex input region mask, ?_⟩
  intro p hp q hq a b ha hb hab
  simp only [complexBatchLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i hi
  exact (complexLoss_convex input region mask (hdata i) (target i)).2 hp hq ha hb hab

example := complexLogits_on_cone (J := Unit) (C := Bool)
  (fun _ : Bool => (LinearMap.id : ℂ →ₗ[ℝ] ℂ)) Set.univ (fun _ _ => true)
  (zero_mem _ _ _) (Set.mem_univ false)
example := complexLogits_combination (J := Unit) (C := Bool)
  (fun _ : Bool => (LinearMap.id : ℂ →ₗ[ℝ] ℂ)) Set.univ (fun _ _ => true)
  (zero_mem _ _ _) (zero_mem _ _ _) (Set.mem_univ false)
  (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)
example := complexLoss_convex (J := Unit) (C := Bool)
  (fun _ : Bool => (LinearMap.id : ℂ →ₗ[ℝ] ℂ)) Set.univ (fun _ _ => true)
  (Set.mem_univ false) true
example := complexBatchLoss_convex (J := Unit) (C := Bool)
  (fun _ : Bool => (LinearMap.id : ℂ →ₗ[ℝ] ℂ)) Set.univ (fun _ _ => true)
  (fun _ : Fin 3 => false) (fun _ => true) (fun _ => Set.mem_univ _)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge
