module

public import Transformer.Phasor.SuccessorDisplacement

/-!
# Centered frequencies of the causal token prefix

Source: the prefix-frequency trial in `experiments/phasor_value_transport/README.md`.
The empirical frequencies before the preceding token give a read of the tied
embedding table. Centering at the current token makes a constant prefix inert.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ}

/-- Relative frequency of class `c` in the first `n` tokens, zero for an empty prefix. -/
noncomputable def prefixFrequency (xs : ℕ → Fin p) (n : ℕ) (c : Fin p) : ℝ :=
  (n : ℝ)⁻¹ * (((Finset.range n).filter fun j => xs j = c).card : ℝ)

/-- Reading tied codes with the histogram produces their arithmetic mean
over token positions, retaining the multiplicity of every repeated token. -/
theorem prefixFrequency_read (xs : ℕ → Fin p) (n : ℕ) (E : Fin p → EucSpace D) :
    ∑ c, prefixFrequency xs n c • E c = (n : ℝ)⁻¹ • ∑ j ∈ Finset.range n, E (xs j) := by
  simp only [prefixFrequency, mul_smul, ← Finset.smul_sum]
  congr 1
  simpa only [Finset.sum_const, Nat.cast_smul_eq_nsmul] using
    Finset.sum_fiberwise' (Finset.range n) xs E

/-- Center the prefix distribution at the current token `b`. An empty
prefix supplies zero coefficients for the whole vocabulary. -/
noncomputable def centeredFrequency (xs : ℕ → Fin p) (n : ℕ) (b c : Fin p) : ℝ :=
  if n = 0 then 0 else prefixFrequency xs n c - if c = b then 1 else 0

/-- The hidden update expressed directly through the original ordered prefix. -/
noncomputable def prefixMeanOffset (E : Fin p → EucSpace D) (xs : ℕ → Fin p)
    (n : ℕ) (b : Fin p) : EucSpace D :=
  (n : ℝ)⁻¹ • ∑ j ∈ Finset.range n, (E (xs j) - E b)

/-- The centered vocabulary read exactly implements the mean hidden
displacement, including all zero-length and single-class cases. -/
theorem centeredFrequency_read (xs : ℕ → Fin p) (n : ℕ) (b : Fin p) (E : Fin p → EucSpace D) :
    ∑ c, centeredFrequency xs n b c • E c = prefixMeanOffset E xs n b := by
  by_cases hn : n = 0
  · subst n
    simp [centeredFrequency, prefixMeanOffset]
  · have hreal : (n : ℝ) ≠ 0 := by exact_mod_cast hn
    simp only [centeredFrequency, ite_eq_right hn, sub_smul, Finset.sum_sub_distrib,
      ite_smul, one_smul, zero_smul, Fintype.sum_ite_eq', prefixFrequency_read,
      prefixMeanOffset, Finset.sum_const, Finset.card_range, smul_sub,
      ← Nat.cast_smul_eq_nsmul ℝ n, smul_smul, inv_mul_cancel₀ hreal, one_smul]

/-- Adding centered frequencies to any signed vocabulary weights adds the
mean displacement in hidden space and preserves the rest of the read. -/
theorem add_frequency_read (xs : ℕ → Fin p) (n : ℕ) (b : Fin p) (E : Fin p → EucSpace D)
    (q : Fin p → ℝ) (strength : ℝ) :
    ∑ c, (q c + strength * centeredFrequency xs n b c) • E c =
      (∑ c, q c • E c) + strength • prefixMeanOffset E xs n b := by
  simp only [add_smul, mul_smul, Finset.sum_add_distrib, ← Finset.smul_sum, centeredFrequency_read]

/-- The mean offset is zero on every prefix consisting of the current token. -/
theorem prefixMeanOffset_eq_zero (E : Fin p → EucSpace D) (xs : ℕ → Fin p) (n : ℕ)
    (b : Fin p) (h : ∀ j < n, xs j = b) : prefixMeanOffset E xs n b = 0 := by
  have hz : ∑ j ∈ Finset.range n, (E (xs j) - E b) = 0 :=
    Finset.sum_eq_zero fun j hj => by rw [h j (Finset.mem_range.mp hj), sub_self]
  simp [prefixMeanOffset, hz]

/-- Add the mean displacement before the successor residual RMS. The
prefix excludes the two positions already used by the pair-value read. -/
noncomputable def Model.frequencyLogit [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (strength frequency gain eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) : ℝ :=
  let a := xs (i-1)
  let b := xs i
  let r := successorAddress xs i
  let w := Perspective.softmaxWeight (M.logits eps scale xs L (i-1))
  let raw := transportedValue M.embedding w (a*b) + (T.withSuccessor xs i).prefixRead .xor (0,0) xs i
  ⟪M.embedding c, M.feature eps scale xs L i + gain • GPTMini.rmsNormEps eps
    ((raw + strength • (M.embedding r - M.embedding b)) + frequency • prefixMeanOffset M.embedding xs (i-1) b)⟫

/-- All class logits at the first two positions agree with the original
displacement architecture, for every width, gain and choice of weights. -/
theorem Model.frequencyLogit_eq_of_short [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (strength frequency gain eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (hi : i ≤ 1) (c : Fin p) :
    M.frequencyLogit T strength frequency gain eps scale xs L i c =
      M.displacedSuccessorLogit T strength gain eps scale xs L i c := by
  have hn : i-1 = 0 := by omega
  simp [frequencyLogit, displacedSuccessorLogit, prefixMeanOffset, hn]

example (E : Fin 5 → EucSpace 8) (n : ℕ) : prefixMeanOffset E (fun _ => 3) n 3 = 0 :=
  prefixMeanOffset_eq_zero E _ n 3 (fun _ _ => rfl)

example (M : Model 128 32 4 171 97) (T : PairTables 128 97) (xs : ℕ → Fin 97)
    (L : ℕ) (c : Fin 97) :
    M.frequencyLogit T 1 (1/2) (3/4) 1 1 xs L 1 c =
      M.displacedSuccessorLogit T 1 (3/4) 1 1 xs L 1 c :=
  M.frequencyLogit_eq_of_short T _ _ _ _ _ xs L 1 (by decide) c

end Transformer.Phasor
