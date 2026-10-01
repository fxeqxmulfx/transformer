import Transformer.Memorization.Section2_Shannon

/-!
# Exact synthetic-dataset entropy

arXiv:2505.24832v3, Section 3.2: `N` sequences of `S` independent,
uniform vocabulary tokens contain `N S log₂ V` bits. A finite vector
of length `N*S` preserves duplicate samples and all coordinate positions.
-/

namespace Transformer.Memorization

open MeasureTheory ProbabilityTheory

/-- Section 3.2: the uniform distribution on the complete token vector. -/
noncomputable def uniformTokenData (N S V : ℕ) : Measure (Fin (N * S) → Fin V) :=
  uniformOn Set.univ

/-- Section 3.2: the exact Shannon entropy of the synthetic-data sampling
procedure. A nonempty vocabulary is required for a probability measure. -/
theorem uniform_dataset_entropy (N S V : ℕ) (hV : 0 < V) :
    entropy id (uniformTokenData N S V) / Real.log 2 =
      (N * S : ℕ) * (Real.log (V : ℝ) / Real.log 2) := by
  let : Nonempty (Fin V) := ⟨⟨0, hV⟩⟩
  have he := (isUniform_uniformOn (A := (Set.univ : Set (Fin (N * S) → Fin V)))).entropy_eq'
    Set.finite_univ measurable_id
  simpa [uniformTokenData, Set.ncard_univ, Fintype.card_fun, Real.log_pow, mul_div_assoc]
    using congrArg (fun r : ℝ => r / Real.log 2) he

/-- Section 3.2: the paper uses the nonempty vocabulary `V=2048`. -/
example : 0 < (2048 : ℕ) := by decide

/-- Section 3.2: with the paper's default `S=64`, `V=2048=2¹¹`,
each sequence has exactly 704 bits of Shannon information. -/
theorem paper_synthetic_dataset_bits (N : ℕ) :
    entropy id (uniformTokenData N 64 2048) / Real.log 2 = (704 * N : ℕ) := by
  rw [uniform_dataset_entropy N 64 2048 (by decide)]
  push_cast
  have hp : (2048 : ℝ) = 2 ^ 11 := by norm_num
  rw [hp, Real.log_pow]
  have hlog : Real.log 2 ≠ 0 := (Real.log_pos (by norm_num)).ne'
  field_simp
  ring

/-- Section 3.2: uniform independent bits have exactly `n` bits of entropy. -/
theorem uniform_bool_vector_entropy (n : ℕ) :
    entropy id (uniformOn (Set.univ : Set (Fin n → Bool))) / Real.log 2 = (n : ℝ) := by
  have he := (isUniform_uniformOn (A := (Set.univ : Set (Fin n → Bool)))).entropy_eq'
    Set.finite_univ measurable_id
  have hlog : Real.log 2 ≠ 0 := (Real.log_pos (by norm_num)).ne'
  simpa [Set.ncard_univ, Fintype.card_fun, Real.log_pow, hlog] using
    congrArg (fun r : ℝ => r / Real.log 2) he

end Transformer.Memorization
