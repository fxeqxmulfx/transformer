import Transformer.GPTMini.Sparsemax.AtomicMatchingMixture

/-!
# The content-only matching encoder forgets key/value binding

New limitation of the genuine content-only atomic model following
arXiv:2211.11052v1, §3.1 and Appendix A.4. Permuting visible occurrences
while fixing the query position preserves every actual head output, for
arbitrary learned Q/K and original values. The proof uses the threshold
description of the original sparsemax projection, arXiv:1602.02068v2, §2.2.
Every finite mixture inherits the invariance.

Two concrete key/value tables differ only by exchanging their written
values, but their correct query answers differ. The old model has sharp
squared-error sum at least 1/2, independently of its learned parameters,
head count, optimization method or numerical coordinate bound.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

/-- A visible occurrence permutation fixing the query preserves the actual content-head output.
Source: a new invariance obstruction for §3.1, using sparsemax §2.2's threshold characterization. -/
theorem matchingHeadOutput_perm {V H D T : ℕ} (h : MatchingHead V H D)
    (tokens : Fin T → Fin V) (σ : Equiv.Perm (Fin T)) (row : Fin T) (channel : Fin D)
    (hr : σ row = row) (hv : ∀ j, j ≤ row ↔ σ j ≤ row) :
    matchingHeadOutput h (tokens ∘ σ) row channel = matchingHeadOutput h tokens row channel := by
  let s := matchingHeadScores h tokens row
  let t := matchingHeadScores h (tokens ∘ σ) row
  have hs (j : Fin T) : t j = s (σ j) := by
    simp only [t, s, matchingHeadScores, Function.comp_apply, hr]
  obtain ⟨τ, hτ⟩ := sparseWeights_exists_threshold s row
  have he : thresholdWeights t row τ = thresholdWeights s row τ ∘ σ := by
    funext j
    by_cases hj : j ≤ row
    · have hp := (hv j).mp hj
      simp only [thresholdWeights, Function.comp_apply, hj, hp, ite_true, hs]
    · have hp : ¬σ j ≤ row := fun hp => hj ((hv j).mpr hp)
      simp only [thresholdWeights, Function.comp_apply, hj, hp, ite_false]
  have hn : ∑ j, thresholdWeights t row τ j = 1 := by
    rw [he]
    change (∑ j, thresholdWeights s row τ (σ j)) = 1
    rw [Equiv.sum_comp, ← hτ]
    exact (sparseWeights_spec s row).1.2.1
  have ht := thresholdWeights_eq_sparseWeights t row τ hn
  have hw : sparseWeights t row = sparseWeights s row ∘ σ := by
    rw [← ht, he, ← hτ]
  change (∑ j, sparseWeights t row j * h.2 ((tokens ∘ σ) j) channel) =
    ∑ j, sparseWeights s row j * h.2 (tokens j) channel
  rw [hw]
  exact Equiv.sum_comp σ (fun j => sparseWeights s row j * h.2 (tokens j) channel)

/-- A real visible exchange satisfies both premises while keeping the final query fixed. -/
example (h : MatchingHead 3 1 1) : matchingHeadOutput h ((id : Fin 3 → Fin 3) ∘
    Equiv.swap 0 1) 2 0 = matchingHeadOutput h id 2 0 := by
  apply matchingHeadOutput_perm
  · norm_num [Equiv.swap_apply_def]
  · intro j
    have hj : j ≤ (2 : Fin 3) := by omega
    have hp : Equiv.swap 0 1 j ≤ (2 : Fin 3) := by omega
    exact ⟨fun _ => hp, fun _ => hj⟩

/-- Any stored or newly selected head retains the same content permutation invariance.
Source: genuine occurrence attention before Appendix A.4's finite mixture sum. -/
theorem matchingMixtureSample_perm {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (σ : Fin R → Equiv.Perm (Fin T)) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (μ : MatchingMixture V H D) (hr : ∀ r, σ r (rows r) = rows r)
    (hv : ∀ r j, j ≤ rows r ↔ σ r j ≤ rows r) :
    matchingMixtureSample (fun r => tokens r ∘ σ r) rows channels μ =
      matchingMixtureSample tokens rows channels μ := by
  ext r
  rw [matchingMixtureSample_apply, matchingMixtureSample_apply]
  apply Finset.sum_congr rfl
  intro h hh
  rw [matchingHeadOutput_perm h (tokens r) (σ r) (rows r) (channels r) (hr r) (hv r)]

/-- An arbitrary real mixture satisfies the nontrivial visible-swap premises. -/
example (μ : MatchingMixture 3 1 1) : matchingMixtureSample
    (fun _ : Fin 1 => (id : Fin 3 → Fin 3) ∘ Equiv.swap 0 1) (fun _ => 2) (fun _ => 0) μ =
      matchingMixtureSample (fun _ : Fin 1 => (id : Fin 3 → Fin 3)) (fun _ => 2) (fun _ => 0) μ := by
  apply matchingMixtureSample_perm (σ := fun _ => Equiv.swap 0 1)
  · intro r
    norm_num [Equiv.swap_apply_def]
  · intro r j
    have hj : j ≤ (2 : Fin 3) := by omega
    have hp : Equiv.swap 0 1 j ≤ (2 : Fin 3) := by omega
    exact ⟨fun _ => hp, fun _ => hj⟩

/-- BOS, two key/value writes, and key one's query; the two written values are exchanged.
Source: the new associative binding witness after §3.1, with no attention-route targets. -/
def pairedBindingTokens (r : Fin 2) : Fin 6 → Fin 5 :=
  ![0, 1, if r = 0 then 3 else 4, 2, if r = 0 then 4 else 3, 1]

/-- The concrete table exchange leaves the final query untouched.
Source: the physical key/value binding obstruction after §3.1. -/
def bindingSwap : Equiv.Perm (Fin 6) := Equiv.swap 2 4

/-- Both concrete inputs have precisely the same visible token multiset and query.
Source: the actual data witness for the content-only encoder restriction. -/
theorem pairedBinding_swap : pairedBindingTokens 1 = pairedBindingTokens 0 ∘ bindingSwap := by
  ext j
  fin_cases j <;> norm_num [pairedBindingTokens, bindingSwap, Equiv.swap_apply_def]

/-- Free original values and learned Q/K cannot repair the old head's binding collision.
Source: the new content-only obstruction, without zero-query or flat-support assumptions. -/
theorem matchingHead_binding_duplicate {H : ℕ} (h : MatchingHead 5 H 1) :
    matchingHeadOutput h (pairedBindingTokens 0) 5 0 =
      matchingHeadOutput h (pairedBindingTokens 1) 5 0 := by
  rw [pairedBinding_swap]
  symm
  apply matchingHeadOutput_perm
  · norm_num [bindingSwap, Equiv.swap_apply_def]
  · intro j
    have hj : j ≤ (5 : Fin 6) := by omega
    have hp : bindingSwap j ≤ (5 : Fin 6) := by omega
    exact ⟨fun _ => hp, fun _ => hj⟩

/-- More learned heads do not remove the content-only binding collision.
Source: the actual Appendix A.4 matching mixture, inheriting the head invariance. -/
theorem matchingMixture_binding_duplicate {H : ℕ} (μ : MatchingMixture 5 H 1) :
    matchingMixtureSample pairedBindingTokens (fun _ => 5) (fun _ => 0) μ 0 =
      matchingMixtureSample pairedBindingTokens (fun _ => 5) (fun _ => 0) μ 1 := by
  rw [matchingMixtureSample_apply, matchingMixtureSample_apply]
  apply Finset.sum_congr rfl
  intro h hh
  rw [matchingHead_binding_duplicate]

/-- The two answer labels, scalar codes for the two possible original value tokens.
Source: ordinary answers in the new binding witness, not target attention positions. -/
def pairedBindingTarget (r : Fin 2) : ℝ := if r = 0 then 0 else 1

/-- The old model's literal answer-only squared loss on the swapped tables.
Source: the new two-context criterion following Appendix A.4. -/
def contentBindingError {H : ℕ} (μ : MatchingMixture 5 H 1) : ℝ :=
  ∑ r, (matchingMixtureSample pairedBindingTokens (fun _ => 5) (fun _ => 0) μ r -
    pairedBindingTarget r) ^ 2

/-- Every content-only physical mixture has error at least one half, regardless of training.
Source: the sharp binding-encoder limitation of the new Appendix A.4 architecture. -/
theorem contentBindingError_lower {H : ℕ} (μ : MatchingMixture 5 H 1) :
    (1 / 2 : ℝ) ≤ contentBindingError μ := by
  unfold contentBindingError
  simp only [Fin.sum_univ_two, pairedBindingTarget, ite_true,
    show (1 : Fin 2) ≠ 0 from by decide, ite_false, sub_zero]
  rw [← matchingMixture_binding_duplicate]
  nlinarith [sq_nonneg (matchingMixtureSample pairedBindingTokens (fun _ => 5) (fun _ => 0) μ 0 - 1 / 2)]

/-- A constant original value table attains the lower bound on the old actual forward. -/
example : contentBindingError (Finsupp.single
    (((0 : Matrix (Fin 5) (Fin 1) ℝ), 0), (fun _ _ => (1 / 2 : ℝ))) 1) = 1 / 2 := by
  let h : MatchingHead 5 1 1 := ((0, 0), fun _ _ => (1 / 2 : ℝ))
  have ho (r : Fin 2) : matchingHeadOutput h (pairedBindingTokens r) 5 0 = 1 / 2 := by
    change (∑ j, sparseWeights (matchingHeadScores h (pairedBindingTokens r) 5) 5 j * (1 / 2)) = _
    rw [← Finset.sum_mul, (sparseWeights_spec _ _).1.2.1, one_mul]
  change contentBindingError (Finsupp.single h 1) = _
  unfold contentBindingError
  rw [matchingMixture_single_output]
  norm_num [matchingHeadSample, ho, pairedBindingTarget, Fin.sum_univ_two]

end Transformer.GPTMini.Sparsemax
