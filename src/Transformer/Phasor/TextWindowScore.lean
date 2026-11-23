module

public import Transformer.Phasor.ImagePatchScore

/-!
# Chronological next-character windows and their score

Source: DiffGPT `5a37c515`, `current_grokking/probes.py::text` and
`data.py::DenseTask.metrics`. Every position of a 128-character input
predicts the next character in its own chronological segment. The source
selects by the negative of the exhaustive mean token cross-entropy.
-/

@[expose] public section
namespace Transformer.Phasor.TextWindowScore

/-- `probes.text` samples starts strictly below the segment length minus
the context. Even the last target then precedes the first character of
the next segment. Indices here are absolute positions in the corpus. -/
theorem window_bounds (base size width start : ℕ) (pos : Fin width)
    (sampled : start < size - width) :
    base ≤ base + start + pos.val ∧
      base + start + pos.val < base + start + pos.val + 1 ∧
      base + start + pos.val + 1 < base + size := by
  have := pos.isLt
  omega

/-- `probes.text`. Changing the entire corpus outside the segment
preserves every input and next-character label in the sampled window. -/
theorem window_local {A : Type*} (left right : ℕ → A) (base size width start : ℕ)
    (sampled : start < size - width)
    (agree : ∀ i, base ≤ i → i < base + size → left i = right i) (pos : Fin width) :
    let i := base + start + pos.val
    left i = right i ∧ left (i+1) = right (i+1) := by
  obtain ⟨lo, next, hi⟩ := window_bounds base size width start pos sampled
  exact ⟨agree _ lo (by omega), agree _ (by omega) hi⟩

/-- `DenseTask.metrics` pools token losses before division. Combining
two evaluation batches weights their negative mean scores by token count,
including when the final batch is smaller. -/
theorem pooled_score (a b : ℝ) (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
    -((a + b) / (m + n : ℝ)) =
      ((m : ℝ) * (-(a / m)) + (n : ℝ) * (-(b / n))) / (m + n : ℝ) := by
  have posM : (0 : ℝ) < m := by exact_mod_cast hm
  have posN : (0 : ℝ) < n := by exact_mod_cast hn
  field_simp
  ring

/-- `protocol.py` gives Tiny Shakespeare a target of -1.63. Every
held-out row has 128 supervised tokens. The decision can equivalently
be made by comparing the total loss to the corresponding token budget. -/
theorem text_target (total : ℝ) (rows : ℕ) (positive : 0 < rows) :
    -(163/100 : ℝ) ≤ -(total / (rows * 128 : ℝ)) ↔
      total ≤ (163/100 : ℝ) * (rows * 128) := by
  have hp : (0 : ℝ) < rows := by exact_mod_cast positive
  rw [neg_le_neg_iff, div_le_iff₀ (by positivity)]

example : 17 ≤ 17+3+(127 : ℕ) ∧ 17+3+127 < 17+3+127+1 ∧
    17+3+127+1 < 17+132 := window_bounds 17 132 128 3 127 (by decide)

example (pos : Fin 128) :
    let f := fun i : ℕ => if i < 132 then i else 0
    let i := 3 + pos.val
    f i = i ∧ f (i+1) = i+1 := by
  apply window_local (fun i => if i < 132 then i else 0) id 0 132 128 3 (by decide)
  intro i _ hi
  simp [show i < 132 by omega]

example : -(((1 : ℝ)+5)/(1+3)) = (1*(-(1/1)) + 3*(-(5/3)))/(1+3) := by
  simpa using pooled_score (1 : ℝ) 5 1 3 (by decide) (by decide)

example (total : ℝ) : -(163/100 : ℝ) ≤ -(total/(1024*128)) ↔
    total ≤ 163/100 * (1024*128) := text_target total 1024 (by decide)

end Transformer.Phasor.TextWindowScore
