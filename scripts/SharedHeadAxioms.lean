import Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext.Basic

open Transformer.Grokking.FlatSpectrum.ConvexDepth
open scoped BigOperators

#print axioms headBudget_convex
#print axioms headLoss_convex
#print axioms headLogits_bound
#print axioms ContextInterpolation.logits
#print axioms ContextInterpolation.batchLoss_convex
#print axioms SignedContext.normalized_mass_le_one
#print axioms SignedContext.normalized_features_bound
#print axioms SignedContext.logits_bound

namespace SharedHeadAudit

-- These fixed rows change observed features too; neither is a Dirac row.
noncomputable def row (x : Fin 2) : Fin 4 → ℝ :=
  ![1, 1, (x.val : ℝ), -(x.val : ℝ)]

def inputs (s : Fin 3) : Fin 2 := if s = 2 then 1 else 0
def labels (s : Fin 3) : Fin 4 := if s = 2 then 1 else 0

example (x : Fin 2) (k : Fin 4 × Bool) :
    |ContextInterpolation.features (fun x => SignedContext.normalized (row x))
      (complexFeatures fourier4) k x| ≤ 1 :=
  SignedContext.normalized_features_bound row (complexFeatures fourier4) x
    (by norm_num : (0 : ℝ) ≤ 1)
    (fun k i => complexFeatures_bound fourier4 fourier4_unitary k i) k

-- All coefficients learn against the actual new observed logits. Original
-- record IDs repeat input0, and the final minibatch has its real denominator1.
example := ContextInterpolation.batchLoss_convex
  (fun x => SignedContext.normalized (row x)) (complexFeatures fourier4)
  inputs labels (Finset.univ : Finset (Fin 3)) 65536

example := ContextInterpolation.batchLoss_convex
  (fun x => SignedContext.normalized (row x)) (complexFeatures fourier4)
  inputs labels ({2} : Finset (Fin 3)) 65536

end SharedHeadAudit
