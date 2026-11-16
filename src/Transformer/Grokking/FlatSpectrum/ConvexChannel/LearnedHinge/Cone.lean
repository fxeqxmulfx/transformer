module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Basic

/-!
# Shared learned ReLU coordinates on explicit input activation cones

Source: `review/flat_spectrum.md`, CM75. Sign constraints refer only to
inputs and architecture masks, never their next-object labels. The actual
learned ReLU is affine in a joint coordinate on the declared cone. A mask
is a constraint, not an assumed prediction on an unobserved prefix.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Orient a sign constraint without making a label assertion, CM75. -/
def orient (active : Bool) (t : ℝ) : ℝ := if active then t else -t

/-- The linear expression equal to a feasible actual ReLU, CM75. -/
def gated (active : Bool) (t : ℝ) : ℝ := if active then t else 0

/-- Both sign orientations respect all parameter combinations, CM75. -/
theorem orient_combination (active : Bool) (a b s t : ℝ) :
    orient active (a * s + b * t) = a * orient active s + b * orient active t := by
  cases active <;> simp [orient, add_comm]

/-- Both constrained linear branches respect parameter combinations, CM75. -/
theorem gated_combination (active : Bool) (a b s t : ℝ) :
    gated active (a * s + b * t) = a * gated active s + b * gated active t := by
  cases active <;> simp [gated]

/-- A sign constraint makes the actual ReLU equal its indicated branch,
CM75. At zero either mask is admitted and both expressions equal zero. -/
theorem hinge_eq_gated (active : Bool) (t : ℝ) (h : 0 ≤ orient active t) :
    hinge t = gated active t := by
  cases active with
  | false =>
    have ht : t ≤ 0 := by simpa [orient] using h
    exact max_eq_right ht
  | true =>
    have ht : 0 ≤ t := by simpa [orient] using h
    exact max_eq_left ht

/-- One complete joint vector for each slot, output class, and readout sign,
CM75. Slots share the same weights across all input sequences. -/
abbrev Parameters (J C E : Type*) := J → C → Bool → E

/-- The actual input-dependent activation restrictions, CM75. Convexity
holds only at inputs in `region`; taking all admitted prefixes
as that region is a stronger, potentially much larger constraint system. -/
def domain (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool) :
    Set (Parameters J C E) :=
  {p | ∀ j c s x, x ∈ region → 0 ≤ orient (mask j x) (input x (p j c s))}

/-- The zero initialization inhabits every input-only cone, CM75. -/
theorem zero_mem (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool) :
    (0 : Parameters J C E) ∈ domain input region mask := by
  intro j c s x hx
  simp [orient]

/-- All representation/readout joint coordinates have one convex domain,
CM75. No separate-layer convexity or independent input table is used. -/
theorem domain_convex (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool) :
    Convex ℝ (domain input region mask : Set (Parameters J C E)) := by
  intro p hp q hq a b ha hb hab j c s x hx
  change 0 ≤ orient (mask j x) (input x (a • p j c s + b • q j c s))
  rw [map_add, map_smul, map_smul, smul_eq_mul, smul_eq_mul, orient_combination]
  exact add_nonneg (mul_nonneg ha (hp j c s x hx)) (mul_nonneg hb (hq j c s x hx))

/-- Every feasible neuron computes its actual constrained linear branch,
CM75. The recovery identity itself remains valid outside this region. -/
theorem neuron_eq_gated (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool)
    {p : Parameters J C E} (hp : p ∈ domain input region mask)
    (j : J) (c : C) (s : Bool) {x : X} (hx : x ∈ region) :
    neuron (p j c s) (input x) = gated (mask j x) (input x (p j c s)) := by
  rw [neuron_eq]
  exact hinge_eq_gated _ _ (hp j c s x hx)

example := hinge_eq_gated true 3 (by norm_num [orient])
example := neuron_eq_gated (fun _ : Bool => LinearMap.fst ℝ ℝ ℝ)
  Set.univ (fun _ : Unit => fun _ : Bool => true)
  (zero_mem _ _ _) () false true (Set.mem_univ false)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge
