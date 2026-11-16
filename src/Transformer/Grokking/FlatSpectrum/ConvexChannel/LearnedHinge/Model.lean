module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Cone

/-!
# Ordinary next-object CE through the shared learned ReLU network

Source: `review/flat_spectrum.md`, CM75. Each signed neuron has a recovered
learned representation and readout amplitude. The class directions and sign
slots are fixed architecture; amplitudes and representation vectors train
together through their joint coordinates. Every sequence uses the same
parameter table. The target occurs only in ordinary full-vocabulary CE.

The convexity theorem retains the actual activation restrictions. It does
not assert convexity at inputs omitted from the declared region, nor that
finding or enumerating all useful activation patterns is inexpensive.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable [Fintype J] [Fintype C]

/-- The actual two learned layers, retaining every positive and negative
neuron and every output class, CM75. -/
noncomputable def forward (p : Parameters J C E) (input : E →ₗ[ℝ] ℝ) (c : C) : ℝ :=
  ∑ j, (neuron (p j c true) input - neuron (p j c false) input)

/-- Full-vocabulary probabilities from the actual computed logits, CM75. -/
noncomputable def probability (p : Parameters J C E) (input : E →ₗ[ℝ] ℝ) (c : C) : ℝ :=
  Geometry.probability (forward p input) c

/-- Ordinary next-object CE, with no hidden-path or gate target, CM75. -/
noncomputable def loss (input : E →ₗ[ℝ] ℝ) (target : C) (p : Parameters J C E) : ℝ :=
  crossEntropy (forward p input) target

/-- The learned network gives positive probability to every object, CM75. -/
theorem probability_pos (p : Parameters J C E) (input : E →ₗ[ℝ] ℝ) (c : C) :
    0 < probability p input c := Geometry.probability_pos _ _

/-- The same full next-object vocabulary is normalized at every point,
CM75; the witness object certifies only nonemptiness. -/
theorem probability_sum (p : Parameters J C E) (input : E →ₗ[ℝ] ℝ) (witness : C) :
    ∑ c, probability p input c = 1 := Geometry.sum_probability _ witness

/-- The linear expression derived from actual feasible ReLUs, CM75.
It is used only with the stated input-cone hypotheses. -/
noncomputable def gatedForward (mask : J → Bool) (input : E →ₗ[ℝ] ℝ)
    (p : Parameters J C E) (c : C) : ℝ :=
  ∑ j, (gated (mask j) (input (p j c true)) - gated (mask j) (input (p j c false)))

omit [Fintype C] in
/-- Actual inference equals its derived linear expression on the full
declared input region, CM75. The factors remain shared across those inputs. -/
theorem forward_eq (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool)
    {p : Parameters J C E} (hp : p ∈ domain input region mask) {x : X} (hx : x ∈ region) :
    forward p (input x) = gatedForward (fun j => mask j x) (input x) p := by
  funext c
  unfold forward gatedForward
  apply Finset.sum_congr rfl
  intro j hj
  rw [neuron_eq_gated input region mask hp j c true hx,
    neuron_eq_gated input region mask hp j c false hx]

omit [Fintype C] in
/-- The complete constrained logit expression respects joint parameter
combinations, CM75. No factor is fixed during the combination. -/
theorem gatedForward_combination (mask : J → Bool) (input : E →ₗ[ℝ] ℝ)
    (p q : Parameters J C E) (a b : ℝ) :
    gatedForward mask input (a • p + b • q) =
      a • gatedForward mask input p + b • gatedForward mask input q := by
  funext c
  simp only [gatedForward, Pi.add_apply, Pi.smul_apply, map_add, map_smul,
    smul_eq_mul, gated_combination]
  simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

/-- Ordinary answer CE is jointly convex in every representation/readout
coordinate on the actual domain, CM75. The input membership premise is
essential; balanced recovery alone does not give this result globally. -/
theorem loss_convex (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool)
    {x : X} (hx : x ∈ region) (target : C) :
    ConvexOn ℝ (domain input region mask : Set (Parameters J C E)) (loss (input x) target) := by
  refine ⟨domain_convex input region mask, ?_⟩
  intro p hp q hq a b ha hb hab
  have hmix := domain_convex input region mask hp hq ha hb hab
  simp only [loss, forward_eq input region mask hp hx, forward_eq input region mask hq hx,
    forward_eq input region mask hmix hx, gatedForward_combination]
  exact (ConvexRecall.convex_crossEntropy target).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- An indexed minibatch retains repeated records and any short tail,
CM75. Labels enter only the ordinary CE terms. -/
noncomputable def batchLoss {N : ℕ} (input : X → E →ₗ[ℝ] ℝ)
    (data : Fin N → X) (target : Fin N → C) (p : Parameters J C E) : ℝ :=
  ∑ i, loss (input (data i)) (target i) p

/-- Every actual indexed minibatch objective has the same joint convexity
on the common cone, CM75. This is not optimizer convergence or a rule for
changing masks between minibatches. -/
theorem batchLoss_convex {N : ℕ} (input : X → E →ₗ[ℝ] ℝ)
    (region : Set X) (mask : J → X → Bool) (data : Fin N → X) (target : Fin N → C)
    (hdata : ∀ i, data i ∈ region) :
    ConvexOn ℝ (domain input region mask : Set (Parameters J C E)) (batchLoss input data target) := by
  refine ⟨domain_convex input region mask, ?_⟩
  intro p hp q hq a b ha hb hab
  simp only [batchLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i hi
  exact (loss_convex input region mask (hdata i) (target i)).2 hp hq ha hb hab

example := forward_eq (fun _ : Bool => LinearMap.fst ℝ ℝ ℝ)
  Set.univ (fun _ : Unit => fun _ : Bool => true)
  (zero_mem (C := Bool) _ _ _) (Set.mem_univ false)
example := loss_convex (E := ℝ × ℝ) (J := Unit) (C := Bool)
  (fun _ : Bool => LinearMap.fst ℝ ℝ ℝ) Set.univ (fun _ _ => true) (Set.mem_univ false) true
example := batchLoss_convex (E := ℝ × ℝ) (J := Unit) (C := Bool)
  (fun _ : Bool => LinearMap.fst ℝ ℝ ℝ) Set.univ (fun _ _ => true)
  (fun _ : Fin 3 => false) (fun _ => true) (fun _ => Set.mem_univ _)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge
