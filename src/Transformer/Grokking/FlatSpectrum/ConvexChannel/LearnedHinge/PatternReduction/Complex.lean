module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction.Basic

/-!
# Separate components while retaining the whole complex forward

Source: `review/convex/moments.md`, CM76. Raw coordinate reads act on real and
imaginary weights by the same real linear map. This property, not arbitrary
real linear complex input maps, permits independent pattern aggregation.
The final theorem retains both components of every actual complex product.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction

open scoped BigOperators

variable {D X J C : Type*}

/-- Read every real or imaginary weight coordinate, CM76. -/
def project (realPart : Bool) : (D → ℂ) →ₗ[ℝ] (D → ℝ) where
  toFun w d := if realPart then (w d).re else (w d).im
  map_add' w v := by ext d; cases realPart <;> simp
  map_smul' a w := by ext d; cases realPart <;> simp

/-- Recombine independently learned components into a complex vector, CM76. -/
def join (r i : D → ℝ) : D → ℂ := fun d => ⟨r d, i d⟩

/-- Recombination retains each complete component vector exactly, CM76. -/
theorem project_join (r i : D → ℝ) (k : Bool) :
    project k (join r i) = if k then r else i := by
  cases k <;> rfl

/-- The complex extension of an actual real coordinate read, CM76. -/
def complexInput (input : (D → ℝ) →ₗ[ℝ] ℝ) : (D → ℂ) →ₗ[ℝ] ℂ where
  toFun w := ⟨input (project true w), input (project false w)⟩
  map_add' w v := by apply Complex.ext <;> simp
  map_smul' a w := by apply Complex.ext <;> simp

variable [Fintype D]

/-- Both component reads use the corresponding learned coordinates, CM76. -/
theorem components_complexInput (input : X → (D → ℝ) →ₗ[ℝ] ℝ)
    (x : X) (k : Bool) (w : D → ℂ) :
    components (fun x => complexInput (input x)) (x, k) w = input x (project k w) := by
  cases k <;> rfl

/-- Retain every class, sign and slot in a component projection, CM76. -/
def projectParameters (k : Bool) (p : Parameters J C (D → ℂ)) : Parameters J C (D → ℝ) :=
  fun j c s => project k (p j c s)

/-- Feasibility of the complex model gives each complete component cone,
CM76. No imaginary or output coordinate is discarded. -/
theorem project_mem (input : X → (D → ℝ) →ₗ[ℝ] ℝ) (region : Set X)
    (mask : J → X × Bool → Bool) {p : Parameters J C (D → ℂ)}
    (hp : p ∈ complexDomain (fun x => complexInput (input x)) region mask) (k : Bool) :
    projectParameters k p ∈ domain input region (fun j x => mask j (x, k)) := by
  intro j c s x hx
  simpa only [components_complexInput, projectParameters] using hp j c s (x, k) hx

/-- The entire complex prediction factors into its two actual component
networks, CM76. This is equality in complex numbers, not only in real logits. -/
theorem complexForward_components [Fintype J] (p : Parameters J C (D → ℂ))
    (input : (D → ℝ) →ₗ[ℝ] ℝ) (c : C) :
    complexForward p (complexInput input) c = (1 + Complex.I) *
      ⟨forward (projectParameters true p) input c, forward (projectParameters false p) input c⟩ := by
  unfold complexForward
  simp only [complexNeuron_eq, ← mul_sub, ← Finset.mul_sum]
  congr 1
  apply Complex.ext
  · change Complex.reLm (∑ j, (splitHinge (complexInput input (p j c true)) -
      splitHinge (complexInput input (p j c false)))) = _
    rw [map_sum]
    change (∑ j, (splitHinge (complexInput input (p j c true)) -
      splitHinge (complexInput input (p j c false))).re) = _
    simp [splitHinge, complexInput, forward, neuron_eq, projectParameters]
  · change Complex.imLm (∑ j, (splitHinge (complexInput input (p j c true)) -
      splitHinge (complexInput input (p j c false)))) = _
    rw [map_sum]
    change (∑ j, (splitHinge (complexInput input (p j c true)) -
      splitHinge (complexInput input (p j c false))).im) = _
    simp [splitHinge, complexInput, forward, neuron_eq, projectParameters]

example := project_mem (D := Bool) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.proj true) Set.univ (fun _ _ => true) (zero_mem _ _ _) false

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction
