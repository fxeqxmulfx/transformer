module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction.Reduction

/-!
# Embed the reduced family back into the complete original family

Source: `review/flat_spectrum.md`, CM76. A reduced point occupies diagonal
pattern slots; all other logical coordinates are exactly zero. This reverse
construction proves that aggregation does not enlarge the prediction family.
A witness old hidden slot is needed: an empty original model has only zero
predictions. The reverse identity holds at every linear input read.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction

open scoped BigOperators

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A complex activation pattern is precisely a pair of scalar patterns,
CM76. The first and second components are real and imaginary respectively. -/
def splitPatterns : Pattern X ≃ ScalarPattern X × ScalarPattern X where
  toFun m := (fun x => m (x, true), fun x => m (x, false))
  invFun ms x := if x.2 then ms.1 x.1 else ms.2 x.1
  left_inv m := by funext ⟨x, k⟩; cases k <;> rfl
  right_inv ms := rfl

/-- Equal component patterns reconstruct their common diagonal, CM76. -/
theorem diagonal_split (m : Pattern X) (hm : (splitPatterns m).2 = (splitPatterns m).1) :
    diagonal (splitPatterns m).1 = m := by
  funext ⟨x, k⟩
  cases k
  · exact (congrFun hm x).symm
  · rfl

/-- Restore the full Cartesian pattern index with exact zeros elsewhere,
CM76. This is an embedding into the old domain, not a sampled dictionary. -/
noncomputable def inflate (q : Parameters (ScalarPattern X) C E) : Parameters (Pattern X) C E := by
  classical
  exact fun m c s => if (splitPatterns m).2 = (splitPatterns m).1 then q (splitPatterns m).1 c s else 0

/-- The embedded point obeys every old complex sign constraint, CM76. -/
theorem inflate_mem (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    {q : Parameters (ScalarPattern X) C E} (hq : q ∈ complexDomain input region diagonal) :
    inflate q ∈ complexDomain input region (fun m x => m x) := by
  classical
  intro m c s x hx
  by_cases hm : (splitPatterns m).2 = (splitPatterns m).1
  · have h := hq (splitPatterns m).1 c s x hx
    change 0 ≤ orient (diagonal (splitPatterns m).1 x)
      (components input x (q (splitPatterns m).1 c s)) at h
    rw [diagonal_split m hm] at h
    simpa only [inflate, ite_eq_left hm] using h
  · simp [inflate, hm, orient]

variable [Fintype X] [DecidableEq X]

/-- Reintroducing the full pattern index leaves the actual entire complex
forward unchanged at every input read, CM76. -/
theorem inflate_forward (q : Parameters (ScalarPattern X) C E) (input : E →ₗ[ℝ] ℂ) :
    complexForward (inflate q) input = complexForward q input := by
  classical
  funext c
  unfold complexForward
  calc
    _ = ∑ ms : ScalarPattern X × ScalarPattern X,
        if ms.2 = ms.1 then complexNeuron (q ms.1 c true) input -
          complexNeuron (q ms.1 c false) input else 0 := by
      apply Fintype.sum_equiv splitPatterns
      intro m
      by_cases hm : (splitPatterns m).2 = (splitPatterns m).1
      · simp [inflate, hm]
      · simp [inflate, hm, complexNeuron_eq, splitHinge, hinge]
    _ = _ := by rw [Fintype.sum_prod_type]; simp

/-- Restore any nonempty original hidden-slot index, CM76. The distinguished
slot is architecture only and is independent of the input and target. -/
noncomputable def inflateSlots (j₀ : J) (q : Parameters (ScalarPattern X) C E) :
    Parameters (J × Pattern X) C E := by
  classical
  exact fun jm c s => if jm.1 = j₀ then inflate q jm.2 c s else 0

omit [Fintype X] [DecidableEq X] in
/-- The restored slot multiplicity remains in the complete old domain,
CM76. Empty hidden indices are excluded by the witness `j₀`. -/
theorem inflateSlots_mem (input : X → E →ₗ[ℝ] ℂ) (region : Set X) (j₀ : J)
    {q : Parameters (ScalarPattern X) C E} (hq : q ∈ complexDomain input region diagonal) :
    inflateSlots j₀ q ∈ complexDomain input region (fun jm x => jm.2 x) := by
  classical
  intro jm c s x hx
  by_cases hj : jm.1 = j₀
  · simpa only [inflateSlots, ite_eq_left hj] using inflate_mem input region hq jm.2 c s x hx
  · simp [inflateSlots, hj, orient]

/-- Restoring all old hidden slots preserves the entire complex function,
CM76, with every zero slot included in the formal sum. -/
theorem inflateSlots_forward [Fintype J] (j₀ : J)
    (q : Parameters (ScalarPattern X) C E) (input : E →ₗ[ℝ] ℂ) :
    complexForward (inflateSlots j₀ q) input = complexForward q input := by
  classical
  rw [← inflate_forward q input]
  funext c
  unfold complexForward
  rw [Fintype.sum_prod_type]
  have h (j : J) : (∑ m, (complexNeuron (inflateSlots j₀ q (j, m) c true) input -
      complexNeuron (inflateSlots j₀ q (j, m) c false) input)) =
      if j = j₀ then ∑ m, (complexNeuron (inflate q m c true) input -
        complexNeuron (inflate q m c false) input) else 0 := by
    by_cases hj : j = j₀
    · simp [inflateSlots, hj]
    · simp [inflateSlots, hj, complexNeuron_eq, splitHinge, hinge]
  simp_rw [h]
  simp

example := diagonal_split (diagonal (fun x : Bool => x)) rfl
example := inflate_mem (E := ℂ) (X := Bool) (C := Bool)
  (fun _ => LinearMap.id) Set.univ (zero_mem _ _ _)
example := inflateSlots_mem (E := ℂ) (X := Bool) (C := Bool)
  (fun _ => LinearMap.id) Set.univ () (zero_mem _ _ _)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction
