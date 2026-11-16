module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Neuron

/-!
# Additive learned directions inside the complete activation cones

Source: `review/flat_spectrum.md`, CM77. The actual domain is a cone, not
only a convex set. Feasible parameter vectors add, and actual inference adds
on every constrained input. This lets a singleton-prefix direction perturb
an arbitrary already fitted point without changing observed predictions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen

open PatternReduction

variable {E X J C D : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Both component sign orientations are additive, CM77. -/
theorem orient_add (active : Bool) (a b : ℝ) :
    orient active (a + b) = orient active a + orient active b := by
  cases active <;> simp [orient, add_comm]

/-- Adding two feasible learned points retains every sign constraint,
CM77. No parameter, class or sign is removed. -/
theorem domain_add (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool)
    {p q : Parameters J C E} (hp : p ∈ domain input region mask)
    (hq : q ∈ domain input region mask) : p + q ∈ domain input region mask := by
  intro j c s x hx
  change 0 ≤ orient (mask j x) (input x (p j c s + q j c s))
  rw [map_add, orient_add]
  exact add_nonneg (hp j c s x hx) (hq j c s x hx)

/-- The complete complex domain is closed under feasible addition, CM77. -/
theorem complexDomain_add (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → Pattern X) {p q : Parameters J C E}
    (hp : p ∈ complexDomain input region mask) (hq : q ∈ complexDomain input region mask) :
    p + q ∈ complexDomain input region mask := domain_add _ _ _ hp hq

/-- Actual recovered real-neuron inference adds on the full declared region,
CM77. The identity is not asserted after leaving the activation domain. -/
theorem forward_add [Fintype J] (input : X → E →ₗ[ℝ] ℝ) (region : Set X) (mask : J → X → Bool)
    {p q : Parameters J C E} (hp : p ∈ domain input region mask) (hq : q ∈ domain input region mask)
    {x : X} (hx : x ∈ region) : forward (p + q) (input x) = forward p (input x) + forward q (input x) := by
  rw [forward_eq input region mask (domain_add input region mask hp hq) hx,
    forward_eq input region mask hp hx, forward_eq input region mask hq hx]
  simpa using gatedForward_combination (fun j => mask j x) (input x) p q 1 1

/-- Component projection retains the addition of every whole vector, CM77. -/
theorem projectParameters_add (p q : Parameters J C (D → ℂ)) (k : Bool) :
    projectParameters k (p + q) = projectParameters k p + projectParameters k q := by
  funext j c s
  exact (project k).map_add _ _

/-- The actual full complex products add on the declared component cones,
CM77. Both real and imaginary outputs are preserved. -/
theorem complexForward_add [Fintype J] [Fintype D]
    (input : X → (D → ℝ) →ₗ[ℝ] ℝ) (region : Set X) (mask : J → Pattern X)
    {p q : Parameters J C (D → ℂ)}
    (hp : p ∈ complexDomain (fun x => complexInput (input x)) region mask)
    (hq : q ∈ complexDomain (fun x => complexInput (input x)) region mask)
    {x : X} (hx : x ∈ region) :
    complexForward (p + q) (complexInput (input x)) =
      complexForward p (complexInput (input x)) + complexForward q (complexInput (input x)) := by
  funext c
  rw [complexForward_components, projectParameters_add, projectParameters_add,
    forward_add input region _ (project_mem input region mask hp true)
      (project_mem input region mask hq true) hx,
    forward_add input region _ (project_mem input region mask hp false)
      (project_mem input region mask hq false) hx]
  change _ = complexForward p (complexInput (input x)) c + complexForward q (complexInput (input x)) c
  rw [complexForward_components, complexForward_components]
  apply Complex.ext <;> simp [Complex.mul_re, Complex.mul_im] <;> ring

example := domain_add (E := ℝ) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.id) Set.univ (fun _ _ => true) (zero_mem _ _ _) (zero_mem _ _ _)
example := complexDomain_add (E := ℂ) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.id) Set.univ (fun _ _ => true) (zero_mem _ _ _) (zero_mem _ _ _)
example := forward_add (E := ℝ) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.id) Set.univ (fun _ _ => true) (zero_mem _ _ _) (zero_mem _ _ _) (Set.mem_univ false)
example := complexForward_add (D := Bool) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.proj true) Set.univ (fun _ _ => true)
  (zero_mem _ _ _) (zero_mem _ _ _) (Set.mem_univ false)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen
