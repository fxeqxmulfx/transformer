module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction.Sequence

/-!
# Nonconstant complex reduction with distinct component activation patterns

Source: `review/convex/moments.md`, CM76. Both original neurons have active
real parts at the true input. Their imaginary parts have different patterns.
All negative coordinates are retained. The reduced full complex predictions
are (-7+7i, 4+10i), so the preferred output object depends on the input.
This is an exact representation witness, not a training experiment.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction.Witness

noncomputable section

/-- Each input object reads its learned coordinate. -/
def input (x : Bool) : (Bool → ℝ) →ₗ[ℝ] ℝ := LinearMap.proj x

/-- Two complex learned representation vectors with four distinct values. -/
def vectors (j d : Bool) : ℂ :=
  if j then (if d then ⟨2, 3⟩ else ⟨-1, -4⟩)
  else (if d then ⟨5, -6⟩ else ⟨-2, 7⟩)

/-- Two neurons for one output class; other classes and signs remain present. -/
def weights : Parameters Bool Bool (Bool → ℂ) :=
  fun j c s => if c && s then vectors j else 0

/-- The real and imaginary components do not share the same source masks. -/
def mask (j : Bool) (x : Bool × Bool) : Bool := if x.2 || j then x.1 else !x.1

/-- All source coordinates satisfy their declared signs, CM76. -/
theorem source_mem : weights ∈ complexDomain (fun x => complexInput (input x)) Set.univ mask := by
  intro j c s x hx
  rcases x with ⟨x, k⟩
  cases j <;> cases c <;> cases s <;> cases x <;> cases k <;>
    norm_num [components_complexInput, project, input, weights, vectors, mask, orient]

/-- The forward reduction supplies an inhabited nonzero reduced domain,
CM76, with all real and imaginary sign constraints checked. -/
theorem reduced_mem : reduce mask weights ∈
    complexDomain (fun x => complexInput (input x)) Set.univ diagonal :=
  reduce_mem input Set.univ mask source_mem

/-- Direct evaluation of the two complete original complex neuron products,
CM76. No imaginary term is dropped from this calculation. -/
theorem source_forward (x c : Bool) :
    complexForward weights (complexInput (input x)) c =
      if c then (if x then (⟨4, 10⟩ : ℂ) else ⟨-7, 7⟩) else 0 := by
  cases x <;> cases c <;>
    norm_num [complexForward, complexNeuron_eq, Fintype.sum_bool, weights, vectors,
      complexInput, project, input, splitHinge, hinge] <;>
    apply Complex.ext <;> norm_num

/-- Actual reduced complex inference retains the two different outputs,
CM76, including their imaginary components. -/
theorem reduced_forward (x c : Bool) :
    complexForward (reduce mask weights) (complexInput (input x)) c =
      if c then (if x then (⟨4, 10⟩ : ℂ) else ⟨-7, 7⟩) else 0 := by
  rw [reduce_forward input Set.univ mask source_mem (Set.mem_univ x)]
  exact source_forward x c

/-- The reduced network prefers distinct next objects at the two inputs,
CM76. Compression did not replace the model with a constant predictor. -/
theorem predictions_differ :
    complexLogits (reduce mask weights) (complexInput (input false)) true <
      complexLogits (reduce mask weights) (complexInput (input false)) false ∧
    complexLogits (reduce mask weights) (complexInput (input true)) false <
      complexLogits (reduce mask weights) (complexInput (input true)) true := by
  norm_num [complexLogits, reduced_forward]

example := reduce_forward input Set.univ mask source_mem (Set.mem_univ false)
example := inflateSlots_mem (fun x => complexInput (input x)) Set.univ () reduced_mem

end

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction.Witness
