import Transformer.GPTMini.Sparsemax.PrefixMemoryCodes

/-!
# Free learned query-key matching atoms on actual causal token occurrences

New candidate after arXiv:2211.11052v1, §3.1 and Appendix A.4. Unlike
the paper's replacement by free positional simplex rows, each atom below
retains shared learned Q/K/value embeddings and the genuine sparsemax
projection of arXiv:1602.02068v2, Eq. (1). Values are independent physical
parameters; there is no attention inverse or feature-response cancellation.

Token identities are observed inputs. Positions may be included in these
input identities; RoPE, normalization, FFN and a shared fixed-width layer
stack are not asserted. The later convex mixture ranges over every head
inside a numerical parameter box, rather than a selected interaction bank.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex Transformer.ConvexRecall
open scoped BigOperators

/-- Independent physical query, key and original value embeddings.
Source: the new causal matching variant of §3.1's attention-only model. -/
abbrev MatchingHead (V H D : ℕ) :=
  (Matrix (Fin V) (Fin H) ℝ × Matrix (Fin V) (Fin H) ℝ) × Matrix (Fin V) (Fin D) ℝ

/-- Actual dot products of learned content embeddings, shared across contexts.
Source: arXiv:2211.11052v1, `eq:attention_only`, before sparsemax Eq. (1). -/
def matchingHeadScores {V H D T : ℕ} (h : MatchingHead V H D)
    (tokens : Fin T → Fin V) (row j : Fin T) : ℝ :=
  ∑ d, h.1.1 (tokens row) d * h.1.2 (tokens j) d

/-- Genuine causal attention times the independent original common values.
Source: arXiv:1602.02068v2, Eq. (1), in the attention-only model of §3.1. -/
def matchingHeadOutput {V H D T : ℕ} (h : MatchingHead V H D)
    (tokens : Fin T → Fin V) (row : Fin T) (channel : Fin D) : ℝ :=
  ∑ j, sparseWeights (matchingHeadScores h tokens row) row j * h.2 (tokens j) channel

/-- Scalar observed outputs; samples may repeat a context with different rows or channels.
Source: the finite-sample atomic construction following Appendix A.4. -/
def matchingHeadSample {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (h : MatchingHead V H D) : Fin R → ℝ :=
  fun r => matchingHeadOutput h (tokens r) (rows r) (channels r)

/-- Numerical bounds on every free Q/K/value coordinate, without prescribed interactions or supports.
Source: a new bounded atomic model after §3.1; not the paper's four-matrix weight decay. -/
def matchingHeadBox (V H D : ℕ) (cap : ℝ) : Set (MatchingHead V H D) :=
  {h | (∀ v d, -cap ≤ h.1.1 v d ∧ h.1.1 v d ≤ cap) ∧
    (∀ v d, -cap ≤ h.1.2 v d ∧ h.1.2 v d ≤ cap) ∧
    (∀ v d, -cap ≤ h.2 v d ∧ h.2 v d ≤ cap)}

/-- Future occurrences receive zero genuine attention at every learned parameter assignment.
Source: the causal simplex in sparsemax Eq. (1). -/
theorem matchingHeadWeights_zero_above {V H D T : ℕ} (h : MatchingHead V H D)
    (tokens : Fin T → Fin V) (row j : Fin T) (hj : row < j) :
    sparseWeights (matchingHeadScores h tokens row) row j = 0 :=
  (sparseWeights_spec _ _).1.2.2 j (not_le.mpr hj)

/-- A genuinely nonzero head still excludes a changed future token. -/
example : sparseWeights (matchingHeadScores (((fun _ : Fin 2 => fun _ : Fin 1 => (1 : ℝ)),
    (fun _ => fun _ => 1)), (fun _ => fun _ : Fin 1 => 1)) id 0) 0 1 = 0 :=
  matchingHeadWeights_zero_above _ _ _ _ (by decide)

/-- Every actual head is causal, including its learned query, keys and values.
Source: sparsemax Eq. (1) and §2.2, with only visible-score equality required. -/
theorem matchingHeadOutput_causal {V H D T : ℕ} (h : MatchingHead V H D)
    (tokens other : Fin T → Fin V) (row : Fin T) (channel : Fin D)
    (ht : ∀ j, j ≤ row → tokens j = other j) :
    matchingHeadOutput h tokens row channel = matchingHeadOutput h other row channel := by
  obtain ⟨τ, hτ⟩ := sparseWeights_exists_threshold (matchingHeadScores h tokens row) row
  have he : thresholdWeights (matchingHeadScores h other row) row τ =
      thresholdWeights (matchingHeadScores h tokens row) row τ := by
    funext j
    by_cases hj : j ≤ row
    · simp only [thresholdWeights, hj, ite_true, matchingHeadScores, ht row le_rfl, ht j hj]
    · simp only [thresholdWeights, hj, ite_false]
  have hs := (sparseWeights_spec (matchingHeadScores h tokens row) row).1.2.1
  rw [hτ] at hs
  have ho := thresholdWeights_eq_sparseWeights (matchingHeadScores h other row) row τ (by rwa [he])
  have hw : sparseWeights (matchingHeadScores h tokens row) row =
      sparseWeights (matchingHeadScores h other row) row := by rw [hτ, ← he, ho]
  unfold matchingHeadOutput
  rw [hw]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : j ≤ row
  · rw [ht j hj]
  · rw [(sparseWeights_spec (matchingHeadScores h other row) row).1.2.2 j hj]
    simp only [zero_mul]

/-- Distinct hidden continuations inhabit all actual-head causality premises. -/
example (h : MatchingHead 2 1 1) : matchingHeadOutput h (fun _ : Fin 2 => 0) 0 0 =
    matchingHeadOutput h (fun j : Fin 2 => if j = 0 then 0 else 1) 0 0 := by
  apply matchingHeadOutput_causal
  intro j hj
  have he : j = 0 := by omega
  subst j
  rfl

/-- Output coefficients can be absorbed into physical values while preserving the learned Q/K.
Source: attention-only head summation after §3.1, without an inverse value decoder. -/
def matchingHeadScaleValues {V H D : ℕ} (a : ℝ) (h : MatchingHead V H D) : MatchingHead V H D :=
  (h.1, a • h.2)

/-- Scaling original values scales the actual attention output.
Source: `eq:attention_only`, with genuine sparsemax Eq. (1) attention retained. -/
theorem matchingHeadOutput_scaleValues {V H D T : ℕ} (a : ℝ) (h : MatchingHead V H D)
    (tokens : Fin T → Fin V) (row : Fin T) (channel : Fin D) :
    matchingHeadOutput (matchingHeadScaleValues a h) tokens row channel =
      a * matchingHeadOutput h tokens row channel := by
  unfold matchingHeadOutput matchingHeadScaleValues
  change (∑ j, sparseWeights (matchingHeadScores h tokens row) row j * (a * h.2 (tokens j) channel)) = _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- A free scalar matching witness; both Q and K change with t, while original values stay fixed.
Source: the new two-token specialization of sparsemax Eq. (1). -/
def matchingScalarHead (t : ℝ) : MatchingHead 2 1 1 :=
  ((Matrix.of (fun v _ => if v = 1 then t else 0), Matrix.of (fun v _ => if v = 1 then t else 0)),
    Matrix.of (fun v _ => if v = 1 then 1 else 0))

/-- Actual bounded scalar heads inhabit the numerical atom domain.
Source: the derived physical matching witness after sparsemax Eq. (1). -/
theorem matchingScalarHead_mem (t : ℝ) (ht : -1 ≤ t ∧ t ≤ 1) :
    matchingScalarHead t ∈ matchingHeadBox 2 1 1 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> intro v d <;> fin_cases v <;> norm_num [matchingScalarHead, ht.1, ht.2]

/-- Both a nonzero matching head and the uniform head are feasible. -/
example : matchingScalarHead 1 ∈ matchingHeadBox 2 1 1 1 ∧
    matchingScalarHead 0 ∈ matchingHeadBox 2 1 1 1 :=
  ⟨matchingScalarHead_mem _ (by norm_num), matchingScalarHead_mem _ (by norm_num)⟩

/-- Uniform genuine attention predicts one half from the two independent original values.
Source: sparsemax §2.2, threshold -1/2, in the physical matching witness. -/
theorem matchingScalarHead_zero_output : matchingHeadOutput (matchingScalarHead 0) id 1 0 = 1 / 2 := by
  have hs : matchingHeadScores (matchingScalarHead 0) id 1 = fun _ => (0 : ℝ) := by
    ext j
    norm_num [matchingHeadScores, matchingScalarHead]
  unfold matchingHeadOutput
  rw [hs, sparseWeights_two_equal]
  norm_num [matchingScalarHead, Fin.sum_univ_two]

/-- Learned matching changes actual predictions with no compensating change of values.
Source: sparsemax §2.2, exact unit score gap, in the same physical atom domain. -/
theorem matchingScalarHead_one_output : matchingHeadOutput (matchingScalarHead 1) id 1 0 = 1 := by
  have hw : sparseWeights (matchingHeadScores (matchingScalarHead 1) id 1) 1 = basis 1 := by
    apply sparseWeights_eq_basis_of_gap _ _ _ le_rfl
    intro j hj hn
    fin_cases j
    · norm_num [matchingHeadScores, matchingScalarHead]
    · simp at hn
  unfold matchingHeadOutput
  rw [hw]
  norm_num [basis, matchingScalarHead, Fin.sum_univ_two]

/-- Every genuine output lies between the numerical original-value bounds.
Source: sparsemax Eq. (1)'s simplex weights, before the new atomic mixture is formed. -/
theorem matchingHeadOutput_bounds {V H D T : ℕ} (cap : ℝ) (h : MatchingHead V H D)
    (hh : h ∈ matchingHeadBox V H D cap) (tokens : Fin T → Fin V)
    (row : Fin T) (channel : Fin D) :
    -cap ≤ matchingHeadOutput h tokens row channel ∧
      matchingHeadOutput h tokens row channel ≤ cap := by
  have ha := (sparseWeights_spec (matchingHeadScores h tokens row) row).1
  have hl : (∑ j, sparseWeights (matchingHeadScores h tokens row) row j * (-cap)) ≤
      matchingHeadOutput h tokens row channel := by
    apply Finset.sum_le_sum
    intro j hj
    exact mul_le_mul_of_nonneg_left (hh.2.2 (tokens j) channel).1 (ha.1 j)
  have hu : matchingHeadOutput h tokens row channel ≤
      ∑ j, sparseWeights (matchingHeadScores h tokens row) row j * cap := by
    apply Finset.sum_le_sum
    intro j hj
    exact mul_le_mul_of_nonneg_left (hh.2.2 (tokens j) channel).2 (ha.1 j)
  rw [← Finset.sum_mul, ha.2.1, one_mul] at hl hu
  exact ⟨hl, hu⟩

/-- Nonuniform learned matching with unit independent values inhabits the actual-output bound. -/
example : -1 ≤ matchingHeadOutput (matchingScalarHead 1) id 1 0 ∧
    matchingHeadOutput (matchingScalarHead 1) id 1 0 ≤ 1 :=
  matchingHeadOutput_bounds _ _ (matchingScalarHead_mem _ (by norm_num)) _ _ _

end Transformer.GPTMini.Sparsemax
