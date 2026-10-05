import Transformer.GPTMini.Sparsemax.ContextMemory
import Transformer.GPTMini.Sparsemax.Uniform

/-!
# Causal context codes from data, including repeated tokens

Derived encoder for the dictionary architecture before sparsemax
arXiv:1602.02068v2, Eq. (1). Project zero position scores onto the causal
simplex, then aggregate its probability mass by token identity. These
fixed data codes mix learned query embeddings; they are not attention
targets. Repeated tokens contribute once per visible occurrence.

Every context length and every token pattern is allowed. Changing future
tokens leaves the code unchanged, proved directly using the causal zeros.
The concrete encoder retains prefix frequencies and loses word order;
the general memory results allow other fixed causal probability codes.
Learned memory keys and values are shared across all these contexts.
Dictionary attention is distinct from attention to input occurrences.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Zero scores give exactly the uniform probability on every causal prefix.
Source: arXiv:1602.02068v2, §2.2, Proposition 1, threshold `-1/N`,
where N is the number of visible positions rather than the padded length. -/
theorem sparseWeights_zero_uniform {T : ℕ} (i : Fin T) : sparseWeights (fun _ => 0) i =
    fun j => if j ≤ i then 1 / ((visiblePositions i).card : ℝ) else 0 := by
  let n : ℝ := (visiblePositions i).card
  have hn : 0 < n := Nat.cast_pos.mpr (visiblePositions_card_pos i)
  have he : thresholdWeights (fun _ : Fin T => 0) i (-(1 / n)) =
      fun j => if j ≤ i then 1 / n else 0 := by
    funext j
    by_cases hj : j ≤ i
    · simp only [thresholdWeights, hj, ite_true, zero_sub, neg_neg,
        max_eq_left (le_of_lt (one_div_pos.mpr hn))]
    · simp only [thresholdWeights, hj, ite_false]
  have hm : (∑ j, thresholdWeights (fun _ : Fin T => 0) i (-(1 / n)) j) = 1 := by
    rw [he, ← Finset.sum_filter]
    change (∑ j ∈ visiblePositions i, (1 / n : ℝ)) = 1
    rw [Finset.sum_const, nsmul_eq_mul]
    change n * (1 / n) = 1
    field_simp [ne_of_gt hn]
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hm, he]

/-- Causal position probabilities pushed forward to fixed token identities.
Source: the new data encoder before Eq. (1), using its zero-score causal
projection. It does not use labels or a teacher attention route. -/
def contextPrefixCode {R T N : ℕ} (tokens : Fin R → Fin T → Fin (N + 1))
    (rows : Fin R → Fin T) : Matrix (Fin R) (Fin (N + 1)) ℝ :=
  Matrix.of (fun r k => ∑ j, if tokens r j = k then
    sparseWeights (fun _ => 0) (rows r) j else 0)

/-- All causal data codes are probability mixtures, with repetitions allowed.
Source: the derived token aggregation of the feasible sparsemax simplex. -/
theorem contextPrefixCode_mem {R T N : ℕ} (tokens : Fin R → Fin T → Fin (N + 1))
    (rows : Fin R → Fin T) : contextPrefixCode tokens rows ∈ contextCodeDomain R N := by
  intro r
  have hs := (sparseWeights_spec (fun _ : Fin T => 0) (rows r)).1
  refine ⟨?_, ?_, fun j hj => False.elim (hj (Set.mem_univ j))⟩
  · intro k
    change 0 ≤ ∑ j, if tokens r j = k then sparseWeights (fun _ => 0) (rows r) j else 0
    apply Finset.sum_nonneg
    intro j _
    split_ifs
    · exact hs.1 j
    · exact le_rfl
  · change (∑ k, ∑ j, if tokens r j = k then
      sparseWeights (fun _ => 0) (rows r) j else 0) = 1
    rw [Finset.sum_comm]
    simp only [Fintype.sum_ite_eq]
    exact hs.2.1

/-- Agreement on visible observations suffices for equal query codes.
Source: the derived causal data encoder; future occurrences carry zero mass. -/
theorem contextPrefixCode_eq_of_visible {R T N : ℕ}
    (tokens other : Fin R → Fin T → Fin (N + 1)) (rows : Fin R → Fin T)
    (ht : ∀ r j, j ≤ rows r → tokens r j = other r j) :
    contextPrefixCode tokens rows = contextPrefixCode other rows := by
  ext r k
  simp only [contextPrefixCode, Matrix.of_apply]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : j ≤ rows r
  · rw [ht r j hj]
  · have hz := (sparseWeights_spec (fun _ : Fin T => 0) (rows r)).1.2.2 j hj
    simp only [hz, ite_self]

/-- Different future tokens inhabit the causal-code equality premises. -/
example : contextPrefixCode (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0) =
    contextPrefixCode (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2))
      (fun _ => 0) := by
  apply contextPrefixCode_eq_of_visible
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- Zero scores give the exact three-position average used by the data witness.
Source: arXiv:1602.02068v2, Proposition 1, threshold `-1/3`. -/
theorem sparseWeights_three_equal : sparseWeights (fun _ : Fin 3 => 0) 2 =
    fun _ => (1 / 3 : ℝ) := by
  have he : thresholdWeights (fun _ : Fin 3 => 0) 2 (-(1 / 3)) = fun _ => (1 / 3 : ℝ) := by
    funext j
    have hj : j ≤ (2 : Fin 3) := Fin.le_last j
    norm_num [thresholdWeights, hj]
  have hm : (∑ j : Fin 3, thresholdWeights (fun _ : Fin 3 => 0) 2 (-(1 / 3)) j) = 1 := by
    rw [he]
    norm_num [Fin.sum_univ_three]
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hm, he]

/-- Three contexts with different repeated observations and a common vocabulary.
Source: the derived causal data-code witness. Rows see every position. -/
def memoryExampleTokens : Fin 3 → Fin 3 → Fin 2 :=
  fun r j => if r = 0 then 1 else if r = 1 then (if j = 0 then 0 else 1)
    else (if j = 2 then 1 else 0)

/-- Codes for three actual prefixes, all computed from data alone.
Source: the derived global-memory witness with repeated token observations. -/
def memoryExampleCodes : Matrix (Fin 3) (Fin 2) ℝ :=
  contextPrefixCode memoryExampleTokens (fun _ => 2)

/-- Every code in the repeated-token witness satisfies the probability constraints.
Source: the generic causal aggregation theorem, without hand-supplied routes. -/
theorem memoryExampleCodes_mem : memoryExampleCodes ∈ contextCodeDomain 3 1 :=
  contextPrefixCode_mem _ _

/-- Actual data codes are `(0,1)`, `(1/3,2/3)` and `(2/3,1/3)`.
Source: zero-score causal sparsemax followed by token-count aggregation. -/
theorem memoryExampleCodes_apply (r : Fin 3) (k : Fin 2) :
    memoryExampleCodes r k = if r = 0 then (if k = 0 then 0 else 1)
      else if r = 1 then (if k = 0 then 1 / 3 else 2 / 3)
      else (if k = 0 then 2 / 3 else 1 / 3) := by
  unfold memoryExampleCodes contextPrefixCode
  simp only [Matrix.of_apply, sparseWeights_three_equal]
  fin_cases r <;> fin_cases k <;> norm_num [Fin.sum_univ_three, memoryExampleTokens]

/-- Repetition changes the code: the two mixed prefixes are distinguishable.
Source: the concrete causal data encoder rather than an assumed target route. -/
theorem memoryExampleCodes_contexts_differ : memoryExampleCodes 1 ≠ memoryExampleCodes 2 := by
  intro h
  have hi := congrFun h 0
  norm_num [memoryExampleCodes_apply] at hi

/-- The inputs contain genuine repetitions in different contexts.
Source: the three concrete prefix streams for the common memory. -/
theorem memoryExampleTokens_repeated :
    memoryExampleTokens 1 1 = memoryExampleTokens 1 2 ∧
      memoryExampleTokens 2 0 = memoryExampleTokens 2 1 ∧
      memoryExampleTokens 1 0 ≠ memoryExampleTokens 2 2 := by
  norm_num [memoryExampleTokens]

/-- Tokens absent from the observed prefix have exactly zero code mass.
Source: the derived causal encoder; future copies do not change this statement. -/
theorem contextPrefixCode_zero_of_absent {R T N : ℕ}
    (tokens : Fin R → Fin T → Fin (N + 1)) (rows : Fin R → Fin T)
    (r : Fin R) (k : Fin (N + 1)) (hk : ∀ j, j ≤ rows r → tokens r j ≠ k) :
    contextPrefixCode tokens rows r k = 0 := by
  simp only [contextPrefixCode, Matrix.of_apply]
  apply Finset.sum_eq_zero
  intro j _
  by_cases hj : j ≤ rows r
  · simp only [hk j hj, ite_false]
  · have hz := (sparseWeights_spec (fun _ : Fin T => 0) (rows r)).1.2.2 j hj
    simp only [hz, ite_self]

/-- The repeated-token witness inhabits the absent-token premise. -/
example : contextPrefixCode memoryExampleTokens (fun _ => 2) 0 0 = 0 := by
  apply contextPrefixCode_zero_of_absent
  intro j _
  norm_num [memoryExampleTokens]

end Transformer.GPTMini.Sparsemax
