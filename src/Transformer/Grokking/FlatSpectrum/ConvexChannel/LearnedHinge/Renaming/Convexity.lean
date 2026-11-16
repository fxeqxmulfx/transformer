module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Basic

/-!
# Convex ordinary answer CE for two shared learned complex passes

Source: `review/flat_spectrum.md`, CM78. The original input-only domain covers
all admitted raw prefixes, including each renamed prefix. The same joint
parameter point supplies both actual forwards. Indexed minibatch sums keep
all original records and short tails; no inner likelihood is substituted.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming

open scoped BigOperators

variable {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}

/-- Parameter combinations pass through both complete learned paths, CM78.
The activation domain is imposed on the entire bounded-prefix universe. -/
theorem pairedLogits_combination (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    {p q : Parameters J O (ObjectWeights O T)}
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hq : q ∈ complexDomain rawInput Set.univ mask)
    (tokens : Prefix O T) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    pairedLogits σ (a • p + b • q) tokens =
      a • pairedLogits σ p tokens + b • pairedLogits σ q tokens := by
  have hcomb (x : Prefix O T) : sequenceLogits (a • p + b • q) x =
      a • sequenceLogits p x + b • sequenceLogits q x :=
    complexLogits_combination (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
      (rawInput (O := O) (T := T)) Set.univ mask (p := p) (q := q) hp hq
      (Set.mem_univ x) a b ha hb hab
  funext c
  simp only [pairedLogits_average, hcomb, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- Ordinary next-object CE remains jointly convex in every learned
coordinate, CM78. No involutivity is needed for this geometry statement. -/
theorem pairedLoss_convex (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (tokens : Prefix O T) (target : O) :
    ConvexOn ℝ (complexDomain rawInput Set.univ mask : Set (Parameters J O (ObjectWeights O T)))
      (pairedLoss σ tokens target) := by
  refine ⟨complexDomain_convex (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
    (rawInput (O := O) (T := T)) Set.univ mask, ?_⟩
  intro p hp q hq a b ha hb hab
  simp only [pairedLoss, pairedLogits_combination σ mask hp hq tokens a b ha hb hab]
  exact (ConvexRecall.convex_crossEntropy target).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- The original indexed minibatch, without deduplication or tail removal, CM78. -/
noncomputable def pairedBatchLoss {N : ℕ} (σ : Equiv.Perm O)
    (data : Fin N → Prefix O T) (labels : Fin N → O) (p : Parameters J O (ObjectWeights O T)) : ℝ :=
  ∑ i, pairedLoss σ (data i) (labels i) p

/-- Every such minibatch has convex ordinary CE on the unchanged full
domain, CM78. This is not a convergence theorem for a particular optimizer. -/
theorem pairedBatchLoss_convex {N : ℕ} (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (data : Fin N → Prefix O T) (labels : Fin N → O) :
    ConvexOn ℝ (complexDomain rawInput Set.univ mask : Set (Parameters J O (ObjectWeights O T)))
      (pairedBatchLoss σ data labels) := by
  refine ⟨complexDomain_convex (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
    (rawInput (O := O) (T := T)) Set.univ mask, ?_⟩
  intro p hp q hq a b ha hb hab
  simp only [pairedBatchLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i hi
  exact (pairedLoss_convex σ mask (data i) (labels i)).2 hp hq ha hb hab

example := pairedLogits_combination (J := Unit) (T := 2) (Equiv.refl Bool)
  (fun _ _ => true) (p := 0) (q := 0) (zero_mem _ _ _) (zero_mem _ _ _)
  ⟨[true, false], by decide⟩ (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming
