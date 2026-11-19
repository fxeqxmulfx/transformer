module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard.Witness

/-!
# Inhabited constrained complex models

Source: `review/convex/moments.md`, CM47. Every conditional margin theorem is
instantiated on the nonzero Fourier source, with retention one half, radius
one half and all raw classes. The constrained tensor remains inside CM46's
original feature/head bounds and preserves repeated-record batch derivatives.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard

open scoped BigOperators

example := reference_mem (scores ContextMixture.witnessMix) ContextMixture.witnessLabels
  (by norm_num : (1 / 2 : ℝ) ≤ 1) (fun x c => (reference_first x).1 c)

example := reference_mem_domain GatedContext.witnessGate ContextMixture.witnessBank
  GatedContext.witnessBase (1 / 2) ContextMixture.witnessHead ContextMixture.witnessMix
  ContextMixture.witnessLabels (by norm_num : (1 / 2 : ℝ) ≤ 1)
  GatedContext.witnessLocal_mem (fun x c => (reference_first x).1 c)

example : softened ∈ domain GatedContext.witnessGate ContextMixture.witnessBank
    GatedContext.witnessBase (1 / 2) ContextMixture.witnessHead ContextMixture.witnessMix
    ContextMixture.witnessLabels (1 / 2) := by
  exact ⟨softened_rows, softened_mem⟩

example := ce_convex GatedContext.witnessGate ContextMixture.witnessBank
  GatedContext.witnessBase (1 / 2) ContextMixture.witnessHead ContextMixture.witnessMix
  ContextMixture.witnessLabels (1 / 2) 2 2

example := diagnostic_ce_convex GatedContext.witnessGate ContextMixture.witnessBank
  (fun j => ContextMixture.witnessBank (!j)) GatedContext.witnessBase (1 / 2)
  ContextMixture.witnessHead ContextMixture.witnessMix ContextMixture.witnessLabels (1 / 2) 2 2

example := weak_comparison softened_mem (by norm_num : (0 : ℝ) ≤ 1 / 2) 2 2
  (by norm_num [reference_scores, ContextMixture.witnessLabels])

example := strict_comparison softened_mem (by norm_num : (0 : ℝ) < 1 / 2) 3 0
  (by norm_num [reference_scores, ContextMixture.witnessLabels])

example := firstMax_preserved softened_mem (by norm_num : (0 : ℝ) < 1 / 2)
  3 (reference_first 3)

example := predictions_eq softened_mem (by norm_num : (0 : ℝ) < 1 / 2)
  ContextMixture.witnessLabels reference_first (by
    intro x
    fin_cases x <;> constructor <;> intro c <;> fin_cases c <;>
      norm_num [softened_scores, ContextMixture.witnessLabels])

example := batch_correct_eq softened_mem (by norm_num : (0 : ℝ) < 1 / 2)
  ContextMixture.witnessLabels reference_first (by
    intro x
    fin_cases x <;> constructor <;> intro c <;> fin_cases c <;>
      norm_num [softened_scores, ContextMixture.witnessLabels])
  ContextInterpolation.batchInputs (fun _ => (0 : Fin 4)) (Finset.univ : Finset (Fin 3))

example : (ContextMixture.witnessLabels 3 : Fin 4) = 2 := by
  apply firstMax_unique
  · exact firstMax_preserved softened_mem (by norm_num : (0 : ℝ) < 1 / 2)
      3 (reference_first 3)
  · constructor <;> intro c <;> fin_cases c <;> norm_num [softened_scores,
      ContextMixture.witnessLabels]

example := GatedContext.logits_bound GatedContext.witnessGate ContextMixture.witnessBank
  GatedContext.witnessBase_mem (by norm_num : (0 : ℝ) ≤ 1 / 2)
  (by norm_num : (1 / 2 : ℝ) ≤ 1) softened_rows 2 (GatedContext.witnessGate_mem 2)
  (by norm_num : (0 : ℝ) ≤ 1) (fun j k => ContextMixture.witnessBank_bound j 2 k)
  ContextMixture.witnessHead_mem 0

example := GatedContext.batch_fderiv_eq GatedContext.witnessGate ContextMixture.witnessBank
  (complexFeatures fourier4) GatedContext.witnessBase (1 / 2) softened
  GatedContext.witnessBase_mem.2 (fun g => (softened_rows g).2)
  ContextInterpolation.batchInputs (fun _ => (0 : Fin 4)) ({2} : Finset (Fin 3))
  (fun s _ => (GatedContext.witnessGate_mem (ContextInterpolation.batchInputs s)).2)
  (fun s _ => ContextMixture.witness_rows s) ContextMixture.witnessHead

end Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard
