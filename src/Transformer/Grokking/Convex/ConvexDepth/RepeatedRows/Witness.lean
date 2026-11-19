module

public import Transformer.Grokking.Convex.ConvexDepth.RepeatedRows.Complex
public import Transformer.Grokking.Convex.ConvexDepth.UniformRows

/-!
# Complete nonconstant and bit-word witnesses

Source: `review/convex/moments.md`, CM70. Uniform rows need one stored value;
nonuniform rows retain all exceptions and opposite unseen answers. Actual
Fourier coefficients include a nonzero imaginary entry. The generic codec
also keeps the distinct words used for signed zero and the smallest subnormal.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.RepeatedRows

open ObservedRows
open LatentPrograms (witnessProgram witnessSamples head features)

/-- The actual balanced unknown row needs one value, CM70. -/
theorem witness_uniform_slots :
    slots ((pack witnessLabels (fun _ _ => 2) UniformRows.witnessBalanced).2
      true ⟨3, by decide⟩) = 1 := by
  have hr : UniformRows.witnessBalanced.2 true ⟨3, by decide⟩ = fun _ => (2 : ℝ) := by
    funext c
    cases c
    · exact UniformRows.witnessBalanced_values.1
    · exact UniformRows.witnessBalanced_values.2
  change slots (encode 2 (UniformRows.witnessBalanced.2 true ⟨3, by decide⟩)) = 1
  rw [hr]
  exact slots_constant 2

/-- The original nonuniform unknown row retains both values, CM70. -/
theorem witness_nonuniform_slots :
    slots ((pack witnessLabels (fun _ _ => 4) (witnessPoint 4 false)).2
      true ⟨3, by decide⟩) = 2 := by
  classical
  have hr : (witnessPoint 4 false).2 true ⟨3, by decide⟩ =
      fun c => if c then (0 : ℝ) else 4 := by
    funext c
    cases c
    · exact witness_free_row.1
    · exact witness_free_row.2.1
  change slots (encode 4 ((witnessPoint 4 false).2 true ⟨3, by decide⟩)) = 2
  rw [hr]
  norm_num [slots, encode, Fintype.univ_bool, Finset.filter_insert, Finset.filter_singleton]

/-- Both full decoder completions pass through both storage layers, CM70. -/
noncomputable def witnessStoredHead (choice : Bool) : Bool × (Fin 4 × Bool) → Bool → ℝ :=
  unpackHead (packHead (fun _ => 0) (head fourier4 1
    (expand witnessLabels (unpack witnessLabels
      (pack witnessLabels (fun _ _ => 0) (witnessPoint 4 choice))))))

/-- Identical observations still permit opposite actual unseen complex
answers after compression, CM70. The representation adds no unknown label. -/
theorem witness_opposite_answers :
    headLogits (features fourier4 witnessProgram) (witnessStoredHead false) (0, 3) false = 4 ∧
    headLogits (features fourier4 witnessProgram) (witnessStoredHead false) (0, 3) true = 0 ∧
    headLogits (features fourier4 witnessProgram) (witnessStoredHead true) (0, 3) false = 0 ∧
    headLogits (features fourier4 witnessProgram) (witnessStoredHead true) (0, 3) true = 4 := by
  simp only [witnessStoredHead, complete_head]
  exact witness_unseen_opposite

/-- Compression preserves a nonzero imaginary coefficient of the actual
balanced Fourier head, CM70. Both components remain in the logical tensor. -/
theorem witness_imaginary_coefficient :
    unpackHead (packHead (fun _ => 0) (head fourier4 1 (expand witnessLabels
      (unpack witnessLabels (pack witnessLabels (fun _ _ => 2) UniformRows.witnessBalanced)))))
      (true, (1, true)) false = -1 := by
  rw [complete_head]
  exact UniformRows.witness_imaginary_head

/-- Four nonconstant latent states with five raw classes, CM70. The last
state has an arbitrary uniform value and the other states have two raw labels. -/
noncomputable def witnessTable (s : Fin 4) (c : Fin 5) : ℝ :=
  if s = 3 then 4 / 5 else if c = (if s = 2 then 1 else 0) then 4 else 0

/-- Every real or imaginary Fourier row uses at most three values for five
logical raw classes, CM70; all complete coefficients are still reconstructible. -/
theorem witness_head_slots (f : Fin 4) (imaginary : Bool) :
    slots (encode (complexCoefficients (tableHead fourier4 witnessTable) (f, imaginary) 4)
      (complexCoefficients (tableHead fourier4 witnessTable) (f, imaginary))) ≤ 3 := by
  have hU : ∀ s : Fin 4, s ∉ (Finset.univ : Finset (Fin 4)) → fourier4 s f = 0 := by
    intro s hs
    exact False.elim (hs (Finset.mem_univ s))
  have hT : ∀ s : Fin 4, s ∈ (Finset.univ : Finset (Fin 4)) →
      ∀ c : Fin 5, c ∉ ({0, 1} : Finset (Fin 5)) → witnessTable s c = witnessTable s 4 := by
    intro s _ c hc
    have hc₀ : c ≠ 0 := by intro he; apply hc; simp [he]
    have hc₁ : c ≠ 1 := by intro he; apply hc; simp [he]
    fin_cases s <;> simp [witnessTable, hc₀, hc₁]
  have h := head_slots_bound fourier4 witnessTable f imaginary 4 Finset.univ {0, 1} hU hT
  norm_num at h ⊢
  exact h

/-- Distinct IEEE-754 words for positive zero, negative zero and the least
positive subnormal, CM70. This states bit preservation, not floating arithmetic. -/
def witnessWords (i : Fin 3) : UInt64 :=
  if i = 0 then 0 else if i = 1 then 9223372036854775808 else 1

/-- All three words survive exact encoding, including the sign of zero,
CM70; no numerical-equality test merges their representations. -/
theorem witness_word_roundtrip :
    (read (encode 0 witnessWords) 0).toNat = 0 ∧
    (read (encode 0 witnessWords) 1).toNat = 9223372036854775808 ∧
    (read (encode 0 witnessWords) 2).toNat = 1 := by
  rw [read_encode]
  norm_num [witnessWords, UInt64.toNat_ofNat]

/-- These are different stored words even when two denote numerical zero,
CM70. Explicit exceptions therefore remain available for both of them. -/
theorem witness_word_distinct : witnessWords 0 ≠ witnessWords 1 ∧
    witnessWords 0 ≠ witnessWords 2 := by decide

example := unpack_packHead (fun _ : Bool × (Fin 4 × Bool) => 0)
  (head fourier4 1 UniformRows.witnessFull)
example := complete_head fourier4 witnessLabels (fun _ _ => 0)
  (witnessPoint 4 false) (fun _ => 0)
example := complex_logits fourier4 witnessProgram witnessLabels (fun _ _ => 0)
  (witnessPoint 4 true) (fun _ => 0) (0, 3)
example := complex_budget fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples (fun _ _ => 0) (witnessPoint 4 false)
  (witnessPoint_mem 4 false (by norm_num) (by norm_num)) (fun _ => 0)
example := table_column_agreement fourier4 witnessTable 1 2 4 Finset.univ
  (by intro s hs; exact False.elim (hs (Finset.mem_univ s)))
  (by intro s _; fin_cases s <;> norm_num [witnessTable])
example := head_slots_bound fourier4 witnessTable 1 true 4 Finset.univ {0, 1}
  (by intro s hs; exact False.elim (hs (Finset.mem_univ s)))
  (by
    intro s _ c hc
    have hc₀ : c ≠ 0 := by intro he; apply hc; simp [he]
    have hc₁ : c ≠ 1 := by intro he; apply hc; simp [he]
    fin_cases s <;> simp [witnessTable, hc₀, hc₁])
example := witness_head_slots 1 true
example := uniform_unknown_head_slots (1 : Matrix (Fin 4) (Fin 4) ℂ)
  witnessLabels UniformRows.witnessBalanced true 3 false false {3} ∅
  (by
    intro s hs
    have hn : s ≠ 3 := by simpa using hs
    simp [hn])
  (by
    intro s hs y hy
    have he : s = 3 := by simpa using hs
    subst s
    have hn : witnessLabels true 3 = none := by decide
    rw [hn] at hy
    cases hy)
  (by simp) UniformRows.witnessBalanced_uniform
-- Zero mass needs no special representation or positivity assumption.
example := unpack_pack witnessLabels (fun _ _ => 0) (witnessPoint 0 true)
-- Distinct original IDs retain duplicated observations and a singleton tail.
example := complex_batch_ce fourier4 witnessProgram witnessLabels (fun _ _ => 0)
  (witnessPoint 4 false) (fun _ => 0)
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2))
  (fun i => decide (2 ≤ i.val)) Finset.univ
example := complex_batch_ce fourier4 witnessProgram witnessLabels (fun _ _ => 0)
  (witnessPoint 4 true) (fun _ => 0)
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2))
  (fun i => decide (2 ≤ i.val)) {3}

end Transformer.Grokking.Convex.ConvexDepth.RepeatedRows
