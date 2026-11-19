import Transformer.Grokking.Convex.ConvexDepth.LatentPrograms

/-! Task334, before native fitting. CM53 covers a fixed count-state program
and a categorical decoder learned from observations; CM54 transfers accepted
logits through the actual growing output prefix. The native bank must retain
its realizability and coverage limits. Source flatness supplies bounds only. -/

open Transformer.Grokking.Convex.ConvexDepth
open Transformer.Landscape.Neural (crossEntropy)
namespace PrefixCountingAudit
open LatentPrograms

-- Two generic count programs. Neither consumes a raw output label.
def program (symbol : Bool) (history : List Bool) : Fin 4 :=
  ⟨history.count symbol % 4, Nat.mod_lt _ (by decide)⟩

-- Three raw classes, with a noninjective decoder learned from four rows.
def samples : List (List Bool × Fin 3) :=
  [([], 2), ([true], 0), ([true, true], 2), ([true, true, true], 0)]

example : retained program [false, true] samples = [true] := by decide

example : predict program [false, true] samples (List.replicate 5 true) = some 0 ∧
    List.replicate 5 true ∉ samples.map Prod.fst := by decide

-- Removing the fourth observation leaves a genuinely unknown count state.
example : predict program [false, true] (samples.take 3) (List.replicate 7 true) = none := by
  decide

def point : Parameters Bool (Fin 4) (Fin 3) := vertex program samples 1 true

example : point ∈ domain program [false, true] samples := by
  exact vertex_mem program [false, true] samples 1 true (by decide)

example : initializeTensor program [false, true] samples 1 = some point := by
  unfold initializeTensor
  have h : retained program [false, true] samples = [true] := by decide
  rw [h]
  rfl

-- The complete complex Fourier head, including imaginary coordinates,
-- gives the observed raw categorical label on a new covered count state.
example : headLogits (features fourier4 program) (head fourier4 24 point)
    (List.replicate 5 true) = fun c => if c = (0 : Fin 3) then 24 else 0 := by
  apply accepted_logits fourier4 fourier4_unitary program [false, true] samples 24
  · exact vertex_mem program [false, true] samples 1 true (by decide)
  · decide

example : head fourier4 24 point ∈ headBudget 192 := by
  have h := head_budget fourier4 fourier4_unitary program [false, true] samples 24
    (vertex_mem program [false, true] samples 1 true (by decide))
  norm_num at h ⊢
  exact h

example : ConvexOn ℝ (domain program [false, true] samples)
    (fun q => crossEntropy (headLogits (features fourier4 program)
      (head fourier4 24 q) (List.replicate 5 true)) (0 : Fin 3)) := by
  exact ce_convex fourier4 fourier4_unitary program [false, true] samples 24
    (List.replicate 5 true) 0

end PrefixCountingAudit

#print axioms Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.initialize_mem
#print axioms Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.initialize_exists
#print axioms Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.head_budget
#print axioms Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.ce_convex
#print axioms Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.accepted_logits
#print axioms Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.accepted_generation
#print axioms Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.initialized_generation
