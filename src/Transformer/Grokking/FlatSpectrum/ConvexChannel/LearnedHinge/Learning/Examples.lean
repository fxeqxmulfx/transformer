module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning.Witness
public import Mathlib.Logic.Equiv.Bool

/-!
# Inhabited full-domain learning premises

Source: `review/flat_spectrum.md`, CM82. Every conditional learning theorem
is exercised at the concrete nonzero complex point. Queries perturb a real
learned coordinate while retaining the imaginary computation. The three-record
batch repeats one record, and the one-record batch supplies a short-tail case.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning

open Witness

noncomputable section

local notation "P₀" => point (3 + 3 * Complex.I)
local notation "V₀" => point 1
local notation "M₀" => (fun (_ : Unit) (_ : Prefix Bool 1 × Bool) => true)

/-- The positive endpoint is feasible for every admitted prefix, CM82. -/
theorem witness_plus : P₀ + (1 : ℝ) • V₀ ∈ complexDomain rawInput Set.univ M₀ :=
  point_queries (by norm_num)

/-- The negative endpoint is feasible for every admitted prefix, CM82. -/
theorem witness_minus : P₀ + (-1 : ℝ) • V₀ ∈ complexDomain rawInput Set.univ M₀ :=
  point_queries (by norm_num)

example := logits_on_domain (E := ObjectWeights Bool 1) (X := Prefix Bool 1)
  (J := Unit) (C := Bool) (rawInput (O := Bool) (T := 1)) Set.univ M₀
  point_mem (Set.mem_univ (sample true))

example := logits_difference (E := ObjectWeights Bool 1) (X := Prefix Bool 1)
  (J := Unit) (C := Bool) (rawInput (O := Bool) (T := 1)) Set.univ M₀
  point_mem witness_plus (Set.mem_univ (sample true))

example := logits_shift (E := ObjectWeights Bool 1) (X := Prefix Bool 1)
  (J := Unit) (C := Bool) (rawInput (O := Bool) (T := 1)) Set.univ M₀
  (p := P₀) (v := V₀) 1 point_mem witness_plus (Set.mem_univ (sample true))

example := complexLoss_first_order (E := ObjectWeights Bool 1) (X := Prefix Bool 1)
  (J := Unit) (C := Bool) (rawInput (O := Bool) (T := 1)) Set.univ M₀
  point_mem witness_plus (Set.mem_univ (sample true)) true

example := query_interval
  (complexDomain_convex (E := ObjectWeights Bool 1) (X := Prefix Bool 1)
    (J := Unit) (C := Bool) (rawInput (O := Bool) (T := 1)) Set.univ M₀)
  (p := P₀) (v := V₀) (by norm_num : (0 : ℝ) < 1) point_mem witness_plus witness_minus
  (by norm_num : |(1 / 2 : ℝ)| ≤ 1)

example := complexLoss_hasDerivAt (E := ObjectWeights Bool 1) (X := Prefix Bool 1)
  (J := Unit) (C := Bool) (rawInput (O := Bool) (T := 1)) Set.univ M₀
  (p := P₀) (v := V₀) (by norm_num : (0 : ℝ) < 1)
  point_mem witness_plus witness_minus (Set.mem_univ (sample true)) true

example := sequenceLoss_first_order M₀ point_mem witness_plus (sample true) true
example := sequenceLoss_hasDerivAt M₀ (p := P₀) (v := V₀) (by norm_num : (0 : ℝ) < 1)
  point_mem witness_plus witness_minus (sample true) true

example := pairedLogits_difference Equiv.boolNot M₀ point_mem witness_plus (sample true)
example := pairedLogits_shift Equiv.boolNot M₀ (p := P₀) (v := V₀) 1
  point_mem witness_plus (sample true)
example := pairedLoss_first_order Equiv.boolNot M₀ point_mem witness_plus (sample true) true
example := pairedLoss_hasDerivAt Equiv.boolNot M₀ (p := P₀) (v := V₀)
  (by norm_num : (0 : ℝ) < 1) point_mem witness_plus witness_minus (sample true) true

example := meanBatchLoss_first_order M₀ records targets point_mem witness_plus
example := meanBatchLoss_hasDerivAt M₀ records targets (p := P₀) (v := V₀)
  (by norm_num : (0 : ℝ) < 1) point_mem witness_plus witness_minus

example := meanPairedBatchLoss_first_order (Equiv.refl Bool) M₀ records targets
  point_mem witness_plus
example := meanPairedBatchLoss_hasDerivAt (Equiv.refl Bool) M₀ records targets
  (p := P₀) (v := V₀) (by norm_num : (0 : ℝ) < 1) point_mem witness_plus witness_minus

example := meanBatchLoss_first_order M₀ (fun _ : Fin 1 => sample true) (fun _ => true)
  point_mem witness_plus
example := meanPairedBatchLoss_hasDerivAt (Equiv.refl Bool) M₀
  (fun _ : Fin 1 => sample true) (fun _ => true) (p := P₀) (v := V₀)
  (by norm_num : (0 : ℝ) < 1) point_mem witness_plus witness_minus

end

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning
