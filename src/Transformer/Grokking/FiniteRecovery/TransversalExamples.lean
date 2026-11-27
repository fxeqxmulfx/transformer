module

public import Transformer.Grokking.FiniteRecovery.ArgandTransversal
public import Transformer.Grokking.FiniteRecovery.ArgandBinaryInitialization
public import Transformer.Grokking.FiniteRecovery.ArgandCapacityExact

/-! Specializations of the family-wide transversal theorem.
The modular-sum staircase gives the known exact nuclear sample counts.
The last-token and constant family has explicit native weight witnesses,
so the full search convergence theorem applies without capacity hypotheses. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set FlatSpectrum Matrix

/-- A nonempty class of unit-modulus rank-one square matrices has the
same transversal number at every size. One staircase hits the entire
class. One member already gives the connected-pattern lower bound. -/
theorem transversalNumber_unit_rankOne {s : ℕ} (targets : Set (Matrix (Fin (s+1)) (Fin (s+1)) ℂ))
    (hn : targets.Nonempty) (hr : ∀ K ∈ targets, ∃ a b, K = vecMulVec a b ∧ ∀ i j, ‖a i * b j‖ = 1) :
    transversalNumber (recoveryHypergraph targets univ nuclearNorm) = 2*s+1 := by
  classical
  have least := isLeast_nuclear_observations targets
  apply le_antisymm
  · apply least.2
    refine ⟨(staircase s).toFinset, ?_, ?_⟩
    · rw [← Set.ncard_eq_toFinset_card', ncard_staircase]
    · intro K hK
      obtain ⟨a,b,rfl,hunit⟩ := hr K hK
      simpa only [Set.coe_toFinset, NuclearRecovers, completions] using
        forall_isMinOn_vecMulVec_staircase (u := a) (v := b) hunit
  · obtain ⟨cells,hcard,hcells⟩ := least.1
    obtain ⟨K,hK⟩ := hn
    obtain ⟨a,b,rfl,hunit⟩ := hr K hK
    have h := two_mul_card_le_ncard_add_one_of_forall_isMinOn (u := a) (v := b) hunit (hcells _ hK)
    rw [Fintype.card_fin, Set.ncard_coe_finset, hcard] at h
    omega

/-- Modular sums at any even length instantiate the same hypergraph. -/
theorem transversalNumber_modularSum {p k s : ℕ} [NeZero p] (e : Fin (s+1) ≃ Word (ZMod p) k) :
    transversalNumber (recoveryHypergraph {targetKernel (enumeratedCut e) ZMod.stdAddChar (modularSum e)}
      univ nuclearNorm) = 2*s+1 := by
  apply transversalNumber_unit_rankOne _ (singleton_nonempty _)
  rintro _ ⟨rfl⟩
  exact ⟨_, _, targetKernel_modularSum e, fun _ _ => by simp⟩

example : transversalNumber (recoveryHypergraph
    {targetKernel (enumeratedCut (wordEnumeration 2 4)) ZMod.stdAddChar (modularSum (wordEnumeration 2 4))}
    univ nuclearNorm) = 31 := transversalNumber_modularSum (s := 15) _
example : transversalNumber (recoveryHypergraph
    {targetKernel (enumeratedCut (wordEnumeration 2 8)) ZMod.stdAddChar (modularSum (wordEnumeration 2 8))}
    univ nuclearNorm) = 511 := transversalNumber_modularSum (s := 255) _
example : transversalNumber (recoveryHypergraph
    {targetKernel (enumeratedCut (wordEnumeration 2 16)) ZMod.stdAddChar (modularSum (wordEnumeration 2 16))}
    univ nuclearNorm) = 131071 := transversalNumber_modularSum (s := 65535) _

namespace Argand

/-- Both constants and copying the final token are represented by full
native weights, for any depth and number of passes. -/
theorem lastTokenFamily_native_realizes (T layers repeats : ℕ) (f : Word Bool (T+1) → Bool)
    (hf : f ∈ lastTokenFamily T) : ∃ w : Weights Bool 1 1 layers, (Model.fromWeights w).realizes repeats f := by
  obtain ⟨mode,rfl⟩ := hf
  refine ⟨sgdInitialWeights true (binaryWitnessDraw 1 mode), ?_⟩
  intro x a ha
  let label := mode.getD (x (Fin.last T))
  have flip : ∀ a b : Bool, a ≠ b → a = !b := by decide
  have he := flip a label ha
  have hg := binaryWitness_gap (layers := layers) (r := 1) le_rfl mode repeats x (Fin.last T)
  rw [he]
  exact sub_pos.mp ((by norm_num : (0:ℝ) < 1/4).trans_le hg)

/-- Three different functions share a single observation mask and the
same initialized model size. The descent proposals use the native WSD
schedule. The search converges from every draw. -/
example (layers repeats n : ℕ) :=
  exists_minimum_transversal_argand (H := 1) (d := 1) (layers := layers) true repeats
    (1/10) (wsdRate 20000) (appendCut Bool (n+1) (n+1))
    (fun b : Bool => if b then (1:ℂ) else -1)
    (by intro a b h; cases a <;> cases b <;> norm_num at *)
    (lastTokenFamily ((n+1)+n)) (lastTokenFamily_native_realizes _ _ _)

end Argand
end Transformer.Grokking.FiniteRecovery
