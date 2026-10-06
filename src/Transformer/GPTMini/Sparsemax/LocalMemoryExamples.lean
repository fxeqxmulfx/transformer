import Transformer.GPTMini.Sparsemax.LocalNearestMemory

/-!
# Changing genuine compact embeddings, tight sparse routes and output freedom

Derived witnesses for arXiv:1602.02068v2, Eq. (1), with common values at
`73f8a0b`. The two-slot example changes Q and K squared norms from one to
two and changes identity attention to self-weight three quarters with
off-diagonal weight one quarter in both directions. A four-slot example
has exactly three active routes in an interior row, so the bound is tight
and does not merely follow from a dictionary of at most three slots.

Despite these actual embedding and support changes, every fixed output
table and probability data encoder produce a noninjective forward map in
the attention parameters after the global value decoder. This concrete
counterexample records the remaining output-only nonidentifiability.
It does not rule out a separate convex criterion selecting parameters.
No Python training, particular task loss or text-generalization claim is made.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The zero compact parameters give genuine identity sparsemax attention at every size.
Source: the actual identity endpoint of the derived arXiv:1602.02068v2, Eq. (1) family. -/
theorem localMemoryAttention_zero (N : ℕ) :
    memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters N)) = 1 := by
  rw [localMemoryAttention_normalized 1 (3 / 4) _ (by norm_num)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))]
  change memoryGramScores (localMemoryCore (0 : Fin N → ℝ)) = 1
  rw [localMemoryCore_zero, memoryIdentityGram_scores]

/-- Actual changed two-slot attention has learned support in both directions.
Source: the explicit permutation mixture before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryExample_attention : memoryGramAttention (localMemoryGram localMemoryExampleParameters) =
    Matrix.of (fun i j => if i = j then 3 / 4 else 1 / 4) := by
  rw [localMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) localMemoryExampleParameters_mem]
  ext i j
  rw [localMemoryCore_scores_apply, Fintype.sum_option]
  fin_cases i <;> fin_cases j <;> norm_num [localMemoryExampleParameters, localMemoryWeights,
    Fin.sum_univ_one, localMemoryPermutation, Equiv.swap_apply_def]

/-- Both genuine learned embedding families change their squared norms.
Source: the compact query/key diagonal additions before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryExample_norms_change :
    localMemoryGram (0 : LocalMemoryParameters 1) (Sum.inl 0) (Sum.inl 0) ≠
      localMemoryGram localMemoryExampleParameters (Sum.inl 0) (Sum.inl 0) ∧
    localMemoryGram (0 : LocalMemoryParameters 1) (Sum.inr 0) (Sum.inr 0) ≠
      localMemoryGram localMemoryExampleParameters (Sum.inr 0) (Sum.inr 0) := by
  rw [localMemoryGram_diagonal, localMemoryGram_diagonal,
    localMemoryGram_diagonal, localMemoryGram_diagonal]
  norm_num [localMemoryExampleParameters]

/-- Actual sparse support changes in both directions while the whole parameter domain is convex.
Source: actual variational arXiv:1602.02068v2, Eq. (1), on the two explicit compact endpoints. -/
theorem localMemoryExample_support_changes :
    memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters 1)) 0 1 = 0 ∧
    memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters 1)) 1 0 = 0 ∧
    0 < memoryGramAttention (localMemoryGram localMemoryExampleParameters) 0 1 ∧
    0 < memoryGramAttention (localMemoryGram localMemoryExampleParameters) 1 0 := by
  rw [localMemoryAttention_zero, localMemoryExample_attention]
  norm_num

/-- Four slots learn positive weights on the first two path edges and independent norm additions.
Source: the tight three-route witness for arXiv:1602.02068v2, Eq. (1). -/
def localMemoryFourParameters : LocalMemoryParameters 3 :=
  ((fun e => if e = 2 then 0 else 1 / 10), (fun _ => 1))

/-- The nonzero four-slot parameters inhabit the strict-inverse compact domain.
Source: the explicit compact witness for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryFourParameters_mem :
    localMemoryFourParameters ∈ localMemoryParameterDomain 3 4 (3 / 4) := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro e
    change 0 ≤ if e = 2 then (0 : ℝ) else 1 / 10
    split_ifs <;> norm_num
  · norm_num [localMemoryFourParameters, Fin.sum_univ_three]
  · intro x
    norm_num [localMemoryFourParameters]

/-- A genuine actual interior attention row has three positive weights and a distant exact zero.
Source: the computed atom scalar products before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryFour_attention_row : memoryGramAttention (localMemoryGram localMemoryFourParameters) 1 =
    (fun j => if j = 0 then 1 / 10 else if j = 1 then 4 / 5 else if j = 2 then 1 / 10 else 0) := by
  rw [localMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) localMemoryFourParameters_mem]
  funext j
  rw [localMemoryCore_scores_apply, Fintype.sum_option]
  fin_cases j <;> norm_num [localMemoryFourParameters, localMemoryWeights, Fin.sum_univ_three,
    localMemoryPermutation, Equiv.swap_apply_def]

/-- The three-route bound is attained with four learned memory slots.
Source: actual variational arXiv:1602.02068v2, Eq. (1), on the nonidentity compact witness. -/
theorem localMemoryFour_support_card :
    (Finset.univ.filter (fun j => memoryGramAttention (localMemoryGram localMemoryFourParameters) 1 j ≠ 0)).card =
      3 := by
  classical
  have h : Finset.univ.filter (fun j =>
      memoryGramAttention (localMemoryGram localMemoryFourParameters) 1 j ≠ 0) =
      ({0, 1, 2} : Finset (Fin 4)) := by
    rw [localMemoryFour_attention_row]
    ext j
    fin_cases j <;> norm_num
  rw [h]
  norm_num

/-- A genuinely unseen query also attains three actual attention routes.
Source: nearest-data input to the compact arXiv:1602.02068v2, Eq. (1) memory.
The query 5/4 is distinct from all four registered real observations. -/
theorem localMemoryFour_unseen_support_card :
    (Finset.univ.filter (fun j => contextMemoryAttention (localMemoryGram localMemoryFourParameters)
      (nearestPrototypeCodes (fun x y : ℝ => |x - y|) (fun j : Fin 4 => (j.val : ℝ))
        (fun _ : Fin 1 => (5 / 4 : ℝ))) 0 j ≠ 0)).card = 3 := by
  classical
  rw [contextNearestAttention_row 4 (3 / 4) _ _ _ _
    (localMemoryGram_mem _ _ _ (by norm_num) localMemoryFourParameters_mem)]
  have hm := nearestPrototype_min (fun x y : ℝ => |x - y|)
    (fun j : Fin 4 => (j.val : ℝ)) (5 / 4) 1
  have hi : nearestPrototype (fun x y : ℝ => |x - y|)
      (fun j : Fin 4 => (j.val : ℝ)) (5 / 4) = 1 := by
    generalize nearestPrototype (fun x y : ℝ => |x - y|)
      (fun j : Fin 4 => (j.val : ℝ)) (5 / 4) = j at hm ⊢
    fin_cases j
    · norm_num at hm
    · rfl
    · norm_num at hm
    · norm_num at hm
  rw [hi]
  exact localMemoryFour_support_card

/-- For every fixed global output table, actual output fitting cannot identify compact attention.
Source: a concrete counterexample to output-only identifiability in the derived
arXiv:1602.02068v2, Eq. (1) chart. The encoder may be any probability table. -/
theorem localJointMemoryForward_not_injective {R D : ℕ}
    (M : Matrix (Fin R) (Fin 2) ℝ) (Z : Matrix (Fin 2) (Fin D) ℝ)
    (hM : M ∈ contextCodeDomain R 1) :
    ¬ Function.Injective (fun p : {p // p ∈ localMemoryParameterDomain 1 4 (3 / 4)} =>
      localJointMemoryForward M (p.val, Z)) := by
  intro hi
  let p : {p // p ∈ localMemoryParameterDomain 1 4 (3 / 4)} :=
    ⟨0, zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)⟩
  let q : {p // p ∈ localMemoryParameterDomain 1 4 (3 / 4)} :=
    ⟨localMemoryExampleParameters, localMemoryExampleParameters_mem⟩
  have he : p = q := hi (localJointMemoryForward_parameter_invariant 4 (3 / 4)
    M p.val q.val Z (by norm_num) hM p.property q.property)
  have hc := congrArg (fun x : {p // p ∈ localMemoryParameterDomain 1 4 (3 / 4)} => x.val.1 0) he
  norm_num [p, q, localMemoryExampleParameters] at hc

/-- Identity data codes and nonconstant target outputs inhabit the nonidentifiability premise. -/
example : ¬ Function.Injective
    (fun p : {p // p ∈ localMemoryParameterDomain 1 4 (3 / 4)} =>
      localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
        (p.val, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ)))) :=
  localJointMemoryForward_not_injective _ _ (oneHotContextCodes_mem _)

end Transformer.GPTMini.Sparsemax
