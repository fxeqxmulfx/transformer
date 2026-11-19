module

public import Transformer.Grokking.Convex.ConvexDepth.GatedContext.Witness

/-!+# The local Fourier model satisfies the full bound and batch hypotheses

Source: `review/convex/moments.md`, CM46. These concrete instances use a
nonzero complete Fourier tensor, a nonzero radius, two groups and two context
maps. They instantiate every conditional generic result, including observed
rows, repeated-record batches and tail derivatives.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.GatedContext

open scoped BigOperators

example := weights_sum witnessGate witnessBase (1 / 2) ContextMixture.witnessMix 2
  witnessBase_mem.2 (witnessGate_mem 2).2 (fun g => (ContextMixture.witnessMix_mem g).2)

example := weights_mem witnessGate witnessBase_mem (by norm_num : (0 : ℝ) ≤ 1 / 2)
  (by norm_num : (1 / 2 : ℝ) ≤ 1) witnessLocal_mem 2 (witnessGate_mem 2)

example := weights_affine witnessGate witnessBase (1 / 2) ContextMixture.witnessMix
  (fun _ => ContextMixture.witnessMix false) 2 (1 / 4) (3 / 4) (by norm_num)

example := features_at witnessGate ContextMixture.witnessBank (complexFeatures fourier4)
  witnessBase (1 / 2) ContextMixture.witnessMix 0 witnessBase_mem.2
  (witnessGate_mem 0).2 (fun g => (ContextMixture.witnessMix_mem g).2)
  (by intro j k; exact ContextMixture.witness_rows 0 j k) (0, true)

example := features_bound witnessGate ContextMixture.witnessBank witnessBase_mem
  (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (1 / 2 : ℝ) ≤ 1)
  witnessLocal_mem 2 (witnessGate_mem 2)
  (fun j k => ContextMixture.witnessBank_bound j 2 k) (0, true)

example := complex_features_bound fourier4 fourier4_unitary witnessGate
  (fun j => ContextInterpolation.pointWeights (ContextMixture.witnessState j))
  (fun _ _ => 1 / 2) witnessBase_mem
  (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (1 / 2 : ℝ) ≤ 1)
  witnessLocal_mem 2 (witnessGate_mem 2)
  (fun j => ContextInterpolation.pointWeights_nonneg (ContextMixture.witnessState j) 2)
  (fun j => ContextInterpolation.pointWeights_sum (ContextMixture.witnessState j) 2)
  (fun _ => by norm_num) (fun _ => by norm_num) (0, true)

example := logits_affine witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
  ContextMixture.witnessMix (fun _ => ContextMixture.witnessMix false)
  ContextMixture.witnessHead 2 (1 / 4) (3 / 4) (by norm_num)

example := logits_bound witnessGate ContextMixture.witnessBank witnessBase_mem
  (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (1 / 2 : ℝ) ≤ 1)
  witnessLocal_mem 2 (witnessGate_mem 2) (by norm_num : (0 : ℝ) ≤ 1)
  (fun j k => ContextMixture.witnessBank_bound j 2 k) ContextMixture.witnessHead_mem 0

example := logits_change_bound witnessGate ContextMixture.witnessBank witnessBase
  (by norm_num : (0 : ℝ) ≤ 1 / 2) ContextMixture.witnessMix
  (fun _ => ContextMixture.witnessMix false) 3 (witnessGate_mem 3).1
  (by norm_num : (0 : ℝ) ≤ 1) (fun j k => ContextMixture.witnessBank_bound j 3 k)
  ContextMixture.witnessHead_mem 2

example := ce_quadratic witnessGate ContextMixture.witnessBank witnessBase
  (by norm_num : (0 : ℝ) ≤ 1 / 2) ContextMixture.witnessMix
  (fun _ => ContextMixture.witnessMix false) 3 2 (witnessGate_mem 3).1
  (by norm_num : (0 : ℝ) ≤ 1) (fun j k => ContextMixture.witnessBank_bound j 3 k)
  ContextMixture.witnessHead_mem

example := batch_loss_eq witnessGate ContextMixture.witnessBank (complexFeatures fourier4)
  witnessBase (1 / 2) ContextMixture.witnessMix witnessBase_mem.2
  (fun g => (ContextMixture.witnessMix_mem g).2) ContextInterpolation.batchInputs
  (fun _ => (0 : Fin 4)) (Finset.univ : Finset (Fin 3))
  (fun s _ => (witnessGate_mem (ContextInterpolation.batchInputs s)).2)
  (fun s _ => ContextMixture.witness_rows s)
  ContextMixture.witnessHead

example := batch_fderiv_eq witnessGate ContextMixture.witnessBank (complexFeatures fourier4)
  witnessBase (1 / 2) ContextMixture.witnessMix witnessBase_mem.2
  (fun g => (ContextMixture.witnessMix_mem g).2) ContextInterpolation.batchInputs
  (fun _ => (0 : Fin 4)) ({2} : Finset (Fin 3))
  (fun s _ => (witnessGate_mem (ContextInterpolation.batchInputs s)).2)
  (fun s _ => ContextMixture.witness_rows s)
  ContextMixture.witnessHead

end Transformer.Grokking.Convex.ConvexDepth.GatedContext
