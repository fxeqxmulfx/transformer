module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction.Embedding

/-!
# Exact pattern reduction for every bounded raw object prefix

Source: `review/convex/moments.md`, CM76. Position/object reads satisfy the
component-separation hypothesis. Applying the reduction to the whole finite
prefix universe preserves the entire admitted sequence function, ordinary
CE and probabilities. The reverse embedding uses any nonempty old slot
index. The smaller representation is still exponential in prefix count.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction

open scoped BigOperators

/-- The real counterpart of the original raw position/object lookup, CM76.
The lookup selects learned values; it supplies no arithmetic features. -/
noncomputable def realInput {O : Type*} {T : ℕ} (tokens : Prefix O T) :
    (Coordinate O T → ℝ) →ₗ[ℝ] ℝ :=
  LinearMap.proj none + ∑ i : Fin T,
    match tokens.val[i.val]? with
    | none => 0
    | some o => LinearMap.proj (some (i, o))

/-- Every visible learned coordinate and the bias are read, CM76. -/
theorem realInput_apply {O : Type*} {T : ℕ} (tokens : Prefix O T) (w : Coordinate O T → ℝ) :
    realInput tokens w = w none + ∑ i : Fin T,
      match tokens.val[i.val]? with
      | none => 0
      | some o => w (some (i, o)) := by
  simp only [realInput, LinearMap.add_apply, LinearMap.proj_apply, LinearMap.sum_apply]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  cases tokens.val[i.val]? <;> rfl

/-- The original complete complex raw-input map is exactly the complex
extension needed by the reduction, CM76. This is not a changed encoder. -/
theorem rawInput_complexInput {O : Type*} {T : ℕ} (tokens : Prefix O T) :
    rawInput tokens = complexInput (realInput tokens) := by
  ext w
  apply Complex.ext
  · change (rawInput tokens w).re = realInput tokens (project true w)
    rw [rawInput_apply, realInput_apply]
    change Complex.reLm (w none + ∑ i : Fin T,
      match tokens.val[i.val]? with | none => 0 | some o => w (some (i, o))) = _
    rw [map_add, map_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    cases tokens.val[i.val]? <;> rfl
  · change (rawInput tokens w).im = realInput tokens (project false w)
    rw [rawInput_apply, realInput_apply]
    change Complex.imLm (w none + ∑ i : Fin T,
      match tokens.val[i.val]? with | none => 0 | some o => w (some (i, o))) = _
    rw [map_add, map_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    cases tokens.val[i.val]? <;> rfl

/-- One learned complex vector per scalar pattern, output class and sign,
CM76. All original position/object coordinates remain in each vector. -/
abbrev ReducedSequenceParameters (O : Type*) (T : ℕ) :=
  Parameters (ScalarPattern (Prefix O T)) O (ObjectWeights O T)

variable {O J : Type*} [Fintype O] [DecidableEq O] [Fintype J] {T : ℕ}

/-- The reduced ordinary next-object loss is jointly convex at every
admitted raw prefix, CM76, on the explicitly retained full-prefix domain. -/
theorem reducedSequence_loss_convex (tokens : Prefix O T) (target : O) :
    ConvexOn ℝ (complexDomain rawInput Set.univ diagonal : Set (ReducedSequenceParameters O T))
      (sequenceLoss tokens target) :=
  sequenceLoss_convex (O := O) (J := ScalarPattern (Prefix O T)) (T := T)
    Set.univ diagonal (Set.mem_univ tokens) target

/-- Every feasible complete old point has exactly the same whole complex
prediction at every bounded raw prefix after reduction, CM76. The map merges
redundant parameters and is not an inverse on individual factor values. -/
theorem sequence_reduction (p : CompleteParameters J O T)
    (hp : p ∈ complexDomain rawInput Set.univ (fun jm x => jm.2 x)) :
    ∃ q : ReducedSequenceParameters O T,
      q ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ tokens : Prefix O T, complexForward q (rawInput tokens) = complexForward p (rawInput tokens) := by
  let input : Prefix O T → (Coordinate O T → ℝ) →ₗ[ℝ] ℝ := realInput
  have he : (fun x => complexInput (input x)) = (rawInput (O := O) (T := T)) := by
    funext x
    exact (rawInput_complexInput x).symm
  have hp' : p ∈ complexDomain (fun x => complexInput (input x)) Set.univ (fun jm x => jm.2 x) := by
    rw [he]
    exact hp
  refine ⟨reduce (fun jm x => jm.2 x) p, ?_, ?_⟩
  · rw [← he]
    exact reduce_mem input Set.univ _ hp'
  · intro tokens
    rw [rawInput_complexInput]
    exact reduce_forward input Set.univ _ hp' (Set.mem_univ tokens)

/-- The reduced family still covers every original physical phased model
on every bounded raw prefix, CM76, including prefixes absent from training. -/
theorem reducedSequence_covers (w : J → ObjectWeights O T) (a : J → O → ℝ) :
    ∃ q : ReducedSequenceParameters O T,
      q ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ tokens : Prefix O T, complexForward q (rawInput tokens) = factoredForward w a (rawInput tokens) := by
  obtain ⟨p, hp, hpred⟩ := completeSequence_covers w a
  obtain ⟨q, hq, hsame⟩ := sequence_reduction p hp
  exact ⟨q, hq, fun tokens => (hsame tokens).trans (hpred tokens)⟩

/-- A reduced point is represented in the old complete family with any
nonempty hidden index, CM76. Thus the reduction does not relax its function class. -/
theorem sequence_reverse (j₀ : J) (q : ReducedSequenceParameters O T)
    (hq : q ∈ complexDomain rawInput Set.univ diagonal) :
    ∃ p : CompleteParameters J O T,
      p ∈ complexDomain rawInput Set.univ (fun jm x => jm.2 x) ∧
      ∀ tokens : Prefix O T, complexForward p (rawInput tokens) = complexForward q (rawInput tokens) := by
  let input : Prefix O T → ObjectWeights O T →ₗ[ℝ] ℂ := rawInput
  refine ⟨inflateSlots j₀ q, ?_, ?_⟩
  · exact inflateSlots_mem (E := ObjectWeights O T) (X := Prefix O T)
      (J := J) (C := O) input Set.univ j₀ (q := q) hq
  · intro tokens
    exact inflateSlots_forward j₀ q (input tokens)

/-- All real storage coordinates after exact reduction, CM76. The prefix
exponent remains; no efficient optimizer or stress-run admission is asserted. -/
theorem reducedSequence_real_coordinate_count (O : Type*) [Fintype O] [DecidableEq O] (T : ℕ) :
    Fintype.card (Bool × ScalarPattern (Prefix O T) × O × Bool × Coordinate O T) =
      4 * Fintype.card O * (1 + T * Fintype.card O) * 2 ^ Fintype.card (Prefix O T) := by
  rw [reduced_real_coordinate_count]
  simp only [Fintype.card_option, Fintype.card_prod, Fintype.card_fin]
  ring

example := sequence_reduction (J := Unit) (O := Bool) (T := 2) 0 (zero_mem _ _ _)
example := sequence_reverse (O := Bool) (T := 2) () 0 (zero_mem _ _ _)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction
