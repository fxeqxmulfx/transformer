module

public import Transformer.Grokking.Convex.ConvexDepth.UniformRows.Projection
public import Transformer.Grokking.Convex.ConvexDepth.ObservedRows.Witness

/-!
# Nonconstant uniform projection and the surviving nonuniform domain

Source: `review/convex/moments.md`, CM69. The many-to-one observed dictionary has
an unknown two-class row. Its mean is a feasible exact self-projection, while
the two old nonuniform completions still give opposite unseen complex logits.
No identification of the unseen answer follows from projection symmetry.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.UniformRows

open scoped BigOperators
open ObservedRows
open LatentPrograms (witnessProgram witnessSamples features head Parameters)

/-- The nonconstant CM65 witness with only its unknown rows balanced, CM69. -/
noncomputable def witnessBalanced : Reduced witnessLabels :=
  balanced witnessLabels (witnessPoint 4 false)

/-- The comparison point satisfies the original complete constraints, CM69. -/
theorem witnessBalanced_mem : witnessBalanced ∈ domain witnessLabels
    (LatentPrograms.retained witnessProgram [false, true] witnessSamples) 1 8 := by
  exact balanced_mem _ _ _ _ _ (witnessPoint_mem 4 false (by norm_num) (by norm_num))

/-- Its unknown rows are uniform; observed labels remain nonconstant, CM69. -/
theorem witnessBalanced_uniform : ClassUniform witnessLabels witnessBalanced := by
  intro j s c d
  rfl

/-- Both actual unknown raw-class entries are two, at total mass four, CM69. -/
theorem witnessBalanced_values :
    witnessBalanced.2 true ⟨3, by decide⟩ false = 2 ∧
      witnessBalanced.2 true ⟨3, by decide⟩ true = 2 := by
  norm_num [witnessBalanced, balanced, witnessPoint, restrict, BoundedLatent.witnessPoint,
    LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]

/-- The same complete feasible domain also contains a nonuniform unknown
row, CM69. Uniformity is not silently added to its constraints. -/
theorem witness_nonuniform_feasible : witnessPoint 4 false ∈ domain witnessLabels
    (LatentPrograms.retained witnessProgram [false, true] witnessSamples) 1 8 ∧
      ¬ ClassUniform witnessLabels (witnessPoint 4 false) := by
  refine ⟨witnessPoint_mem 4 false (by norm_num) (by norm_num), ?_⟩
  intro h
  have he := h true ⟨3, by decide⟩ false true
  rw [witness_free_row.1, witness_free_row.2.1] at he
  norm_num at he

/-- This nonuniform point has a strictly positive remainder, CM69. -/
theorem witness_positive_variance : 0 < variance witnessLabels (witnessPoint 4 false) := by
  apply lt_of_le_of_ne (variance_nonneg _ _)
  intro hz
  have hu := (variance_zero_iff witnessLabels (witnessPoint 4 false)).mp hz.symm
  exact witness_nonuniform_feasible.2 (fun j s c d => (hu j s c).trans (hu j s d).symm)

/-- The complete joint tensor corresponding to the balanced witness, CM69. -/
noncomputable abbrev witnessFull := expand witnessLabels witnessBalanced

/-- Every actual full coefficient of the nonconstant witness, CM69;
observed entries retain distinct labels and the unknown row retains both classes. -/
theorem witnessFull_entry (j : Bool) (s : Fin 4) (c : Bool) :
    witnessFull.2 j s c = if j then
      (if s = 3 then 2 else if c = decide (s = 2) then 4 else 0) else 0 := by
  change (expand witnessLabels witnessBalanced).2 j s c = _
  cases j
  · cases h : witnessLabels false s with
    | none =>
      rw [expand_unknown witnessLabels witnessBalanced false ⟨s, h⟩]
      norm_num [witnessBalanced, balanced, witnessPoint, restrict, BoundedLatent.witnessPoint,
        LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]
    | some y =>
      rw [expand_known witnessLabels witnessBalanced false s y h]
      norm_num [witnessBalanced, balanced, witnessPoint, restrict, BoundedLatent.witnessPoint,
        LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]
  · fin_cases s
    · change (expand witnessLabels witnessBalanced).2 true 0 c = if c = false then 4 else 0
      rw [expand_known witnessLabels witnessBalanced true 0 false (by decide) c]
      norm_num [witnessBalanced, balanced, witnessPoint, restrict, BoundedLatent.witnessPoint,
        LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]
    · change (expand witnessLabels witnessBalanced).2 true 1 c = if c = false then 4 else 0
      rw [expand_known witnessLabels witnessBalanced true 1 false (by decide) c]
      norm_num [witnessBalanced, balanced, witnessPoint, restrict, BoundedLatent.witnessPoint,
        LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]
    · change (expand witnessLabels witnessBalanced).2 true 2 c = if c = true then 4 else 0
      rw [expand_known witnessLabels witnessBalanced true 2 true (by decide) c]
      norm_num [witnessBalanced, balanced, witnessPoint, restrict, BoundedLatent.witnessPoint,
        LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]
    · change (expand witnessLabels witnessBalanced).2 true 3 c = 2
      rw [expand_unknown witnessLabels witnessBalanced true ⟨3, by decide⟩ c]
      norm_num [witnessBalanced, balanced, witnessPoint, restrict, BoundedLatent.witnessPoint,
        LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]

/-- The complete original domain, not only the reduced one, contains it, CM69. -/
theorem witnessFull_mem : witnessFull ∈
    BoundedLatent.domain witnessProgram [false, true] witnessSamples 1 8 := by
  exact expand_mem witnessProgram [false, true] witnessSamples witnessBalanced_mem

/-- Its zero self-distance is a global minimum among all full tensors,
CM69. This supplies an actual minimizer for the conditional full theorem. -/
theorem witnessFull_minimum (q : Parameters Bool (Fin 4) Bool) :
    fullSquaredDistance witnessFull witnessFull ≤ fullSquaredDistance q witnessFull := by
  have hn : 0 ≤ fullSquaredDistance q witnessFull :=
    add_nonneg (Finset.sum_nonneg fun j hj => sq_nonneg _)
      (Finset.sum_nonneg fun j hj => Finset.sum_nonneg fun s hs =>
        Finset.sum_nonneg fun c hc => sq_nonneg _)
  simpa [fullSquaredDistance] using hn

/-- The actual complete complex products tie at the unobserved state,
CM69; uniform projection has not supplied its missing raw label. -/
theorem witness_unseen_tie (c : Bool) :
    headLogits (features fourier4 witnessProgram) (head fourier4 1 witnessFull) (0, 3) c = 2 := by
  rw [LatentPrograms.logits fourier4 fourier4_unitary]
  simp_rw [witnessFull_entry]
  cases c <;> norm_num [Fintype.sum_bool, witnessProgram]

/-- A nonuniform feasible tensor is also an exact projection of itself,
CM69. Thus the uniform-proposal hypothesis cannot be removed. -/
theorem witness_nonuniform_self_projection :
    ∃ q ∈ BoundedLatent.domain witnessProgram [false, true] witnessSamples 1 8,
      ¬ ClassUniform witnessLabels (restrict witnessLabels q) ∧
        ∀ t, fullSquaredDistance q q ≤ fullSquaredDistance t q := by
  let q := expand witnessLabels (witnessPoint 4 false)
  refine ⟨q, expand_mem witnessProgram [false, true] witnessSamples witness_nonuniform_feasible.1, ?_, ?_⟩
  · change ¬ ClassUniform witnessLabels (restrict witnessLabels (expand witnessLabels (witnessPoint 4 false)))
    rw [restrict_expand]
    exact witness_nonuniform_feasible.2
  · intro t
    have hn : 0 ≤ fullSquaredDistance t q :=
      add_nonneg (Finset.sum_nonneg fun j hj => sq_nonneg _)
        (Finset.sum_nonneg fun j hj => Finset.sum_nonneg fun s hs =>
          Finset.sum_nonneg fun c hc => sq_nonneg _)
    simpa [fullSquaredDistance] using hn

/-- The balanced witness still uses nonzero imaginary head coordinates,
CM69. Its complete complex representation has not collapsed to a real toy. -/
theorem witness_imaginary_head : head fourier4 1 witnessFull (true, (1, true)) false = -1 := by
  change ((tableHead fourier4 (fun s c => 1 * witnessFull.2 true s c)) 1 false).im = -1
  simp_rw [witnessFull_entry]
  norm_num [tableHead, complexTable, Matrix.mul_apply, Matrix.conjTranspose_apply,
    Fin.sum_univ_four, fourier4]

example := card_pos (C := Bool)
example := sum_mean (C := Bool) 4
example := sum_centered (fun b : Bool => if b then 3 else 1) 4 (by norm_num [Fintype.sum_bool])
example := square_decomposition (fun b : Bool => if b then 3 else 1) 4 (-3)
  (by norm_num [Fintype.sum_bool])
example := square_remainder_nonneg (fun b : Bool => if b then 3 else 1) 4
example := square_remainder_zero_iff (fun _ : Bool => (2 : ℝ)) 4
example := uniform_row_closest (fun b : Bool => if b then 3 else 1) 4 (-3)
  (by norm_num [Fintype.sum_bool])
example := uniform_row_unique (fun _ : Bool => (1 : ℝ)) 2 3
  (by norm_num [Fintype.sum_bool]) (by norm_num [Fintype.sum_bool])
-- Zero mass uses the same actual cardinality and unique uniform row.
example := uniform_row_unique (fun _ : Bool => (0 : ℝ)) 0 (-3)
  (by norm_num [Fintype.sum_bool]) (by norm_num [Fintype.sum_bool])
example := balanced_mem witnessLabels _ 1 8 _ (witnessPoint_mem 4 false (by norm_num) (by norm_num))
example := variance_nonneg witnessLabels (witnessPoint 4 false)
example := variance_zero_iff witnessLabels witnessBalanced
example := distance_decomposition witnessLabels (witnessPoint 4 false) witnessBalanced
  witness_nonuniform_feasible.1.2.2.1 witnessBalanced_uniform
example := balanced_objective witnessLabels witnessBalanced witnessBalanced false witnessBalanced_uniform
example := weightedDistance_nonneg witnessLabels (witnessPoint 4 false) witnessBalanced
example := minimizer_uniform witnessLabels
  (LatentPrograms.retained witnessProgram [false, true] witnessSamples)
  1 8 witnessBalanced witnessBalanced witnessBalanced_mem witnessBalanced_uniform
  (by intro q hq; simpa [weightedDistance] using weightedDistance_nonneg witnessLabels q witnessBalanced)
example := full_minimizer_uniform witnessProgram [false, true] witnessSamples 1 8
  witnessFull witnessFull witnessFull_mem
  (by intro j s c d; simp only [witnessFull, expand_unknown]; exact witnessBalanced_uniform j s c d)
  (fun q _ => witnessFull_minimum q)
example := witnessFull_minimum (expand witnessLabels (witnessPoint 4 false))

-- The original nonuniform completions still have opposite strict complex margins.
example := ObservedRows.witness_unseen_opposite
example := complete_budget fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples witnessBalanced_mem
-- Actual indexed CE retains duplicated observations and a singleton tail.
example := batchLoss_eq fourier4 witnessProgram [false, true] witnessSamples witnessFull_mem
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val)) Finset.univ
example := batchLoss_eq fourier4 witnessProgram [false, true] witnessSamples witnessFull_mem
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val)) {3}

end Transformer.Grokking.Convex.ConvexDepth.UniformRows
