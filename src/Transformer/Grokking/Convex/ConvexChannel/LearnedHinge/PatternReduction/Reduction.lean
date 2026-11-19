module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction.Complex

/-!
# Exact aggregation into one common real and imaginary pattern family

Source: `review/convex/moments.md`, CM76. Aggregate each component by its
own sign pattern, then join the two complete vectors at each scalar pattern.
Every original hidden slot contributes. The resulting actual complex model
has one slot per scalar pattern, with both component constraints retained.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction

variable {D X J C : Type*} [Fintype D] [Fintype X] [DecidableEq X] [Fintype J]

/-- Apply one scalar sign pattern to each component of a complex neuron,
CM76. Zero components satisfy either selected sign. -/
def diagonal (m : ScalarPattern X) : Pattern X := fun x => m x.1

/-- Sum every real and imaginary vector into its own pattern group, CM76.
Joining the groups retains complex representations and full complex products. -/
noncomputable def reduce (mask : J → Pattern X) (p : Parameters J C (D → ℂ)) :
    Parameters (ScalarPattern X) C (D → ℂ) := fun m c s =>
  join (collapse (fun j x => mask j (x, true)) (projectParameters true p) m c s)
    (collapse (fun j x => mask j (x, false)) (projectParameters false p) m c s)

omit [DecidableEq X] in
/-- The reduced complex model contains exactly each aggregated component,
CM76. This does not replace a complex multiplication by a real projection. -/
theorem project_reduce (mask : J → Pattern X) (p : Parameters J C (D → ℂ)) (k : Bool) :
    projectParameters k (reduce mask p) =
      collapse (fun j x => mask j (x, k)) (projectParameters k p) := by
  funext m c s
  cases k <;> rfl

omit [DecidableEq X] in
/-- The single scalar-pattern family satisfies both complex activation
conditions at every declared input, CM76. -/
theorem reduce_mem (input : X → (D → ℝ) →ₗ[ℝ] ℝ) (region : Set X)
    (mask : J → Pattern X) {p : Parameters J C (D → ℂ)}
    (hp : p ∈ complexDomain (fun x => complexInput (input x)) region mask) :
    reduce mask p ∈ complexDomain (fun x => complexInput (input x)) region diagonal := by
  intro m c s x hx
  rcases x with ⟨x, k⟩
  have h := collapse_mem input region (fun j x => mask j (x, k))
    (project_mem input region mask hp k) m c s x hx
  rw [← project_reduce mask p k] at h
  simpa only [components_complexInput, projectParameters, diagonal] using h

/-- Reduction preserves the whole complex prediction at every constrained
input, CM76. Outside that input region no equality is inferred from this proof. -/
theorem reduce_forward (input : X → (D → ℝ) →ₗ[ℝ] ℝ) (region : Set X)
    (mask : J → Pattern X) {p : Parameters J C (D → ℂ)}
    (hp : p ∈ complexDomain (fun x => complexInput (input x)) region mask)
    {x : X} (hx : x ∈ region) :
    complexForward (reduce mask p) (complexInput (input x)) =
      complexForward p (complexInput (input x)) := by
  funext c
  rw [complexForward_components, complexForward_components,
    project_reduce, project_reduce,
    forward_collapse input region _ (project_mem input region mask hp true) hx,
    forward_collapse input region _ (project_mem input region mask hp false) hx]

/-- Ordinary CE remains jointly convex on the reduced actual complex domain,
CM76. The parameter reduction did not turn the domain into an unconstrained space. -/
theorem reduced_loss_convex [Fintype C] (input : X → (D → ℝ) →ₗ[ℝ] ℝ)
    (region : Set X) {x : X} (hx : x ∈ region) (target : C) :
    ConvexOn ℝ (complexDomain (fun x => complexInput (input x)) region diagonal :
      Set (Parameters (ScalarPattern X) C (D → ℂ)))
      (complexLoss (complexInput (input x)) target) :=
  complexLoss_convex _ region diagonal hx target

/-- Every indexed minibatch has exactly the same ordinary CE after reduction,
CM76. Original repeated observations and any short tail retain their indices. -/
theorem reduce_batchLoss [Fintype C] {N : ℕ} (input : X → (D → ℝ) →ₗ[ℝ] ℝ)
    (region : Set X) (mask : J → Pattern X) {p : Parameters J C (D → ℂ)}
    (hp : p ∈ complexDomain (fun x => complexInput (input x)) region mask)
    (data : Fin N → X) (target : Fin N → C) (hdata : ∀ i, data i ∈ region) :
    complexBatchLoss (fun x => complexInput (input x)) data target (reduce mask p) =
      complexBatchLoss (fun x => complexInput (input x)) data target p := by
  unfold complexBatchLoss
  apply Finset.sum_congr rfl
  intro i hi
  unfold complexLoss complexLogits
  rw [reduce_forward input region mask hp (hdata i)]

/-- The new full real-coordinate count retains every class, sign, complex
component, and weight coordinate, CM76; the old slot multiplicity disappears. -/
theorem reduced_real_coordinate_count (X C D : Type*)
    [Fintype X] [DecidableEq X] [Fintype C] [Fintype D] :
    Fintype.card (Bool × ScalarPattern X × C × Bool × D) =
      4 * Fintype.card C * Fintype.card D * 2 ^ Fintype.card X := by
  simp only [Fintype.card_prod, Fintype.card_bool, scalar_pattern_count]
  ring

example := reduce_mem (D := Bool) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.proj true) Set.univ (fun _ _ => true) (zero_mem _ _ _)
example := reduce_forward (D := Bool) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.proj true) Set.univ (fun _ _ => true) (zero_mem _ _ _) (Set.mem_univ false)
example := reduced_loss_convex (D := Bool) (X := Bool) (C := Bool)
  (fun _ => LinearMap.proj true) Set.univ (Set.mem_univ false) true
example := reduce_batchLoss (D := Bool) (X := Bool) (J := Unit) (C := Bool)
  (fun _ => LinearMap.proj true) Set.univ (fun _ _ => true) (zero_mem _ _ _)
  (fun _ : Fin 3 => false) (fun _ => true) (fun _ => Set.mem_univ _)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction
