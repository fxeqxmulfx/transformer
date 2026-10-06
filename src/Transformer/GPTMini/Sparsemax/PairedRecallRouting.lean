import Transformer.GPTMini.Sparsemax.PairedRecallHead

/-!
# Exact causal recall of 256 keys with one bounded width-eight head

New capacity guarantee for the contextual atomic model following
arXiv:2211.11052v1, §3.1 and Appendix A.4. The compact integer geometry
produces the score gap of original sparsemax Eq. (1), arXiv:1602.02068v2,
§2.2. The head reads an independent original value at the occurrence
immediately after the queried key. A unique visible matching predecessor
is an explicit input premise, satisfied by easy MQAR's distinct writes
and nonrepeated queries. Rewrites requiring the latest value are not covered.

The proof applies to arbitrary original values and any finite set of such
inputs. No target attention routes are passed to a training procedure. This
is an expressivity witness inside the full freely selectable physical-head
family, not a guarantee that random numerical head search will find it.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex Transformer.ConvexRecall
open scoped BigOperators

/-- Actual causal sparsemax selects the value occurrence with the unique matching predecessor.
Source: the new compact recall witness, certified by sparsemax Eq. (1) and §2.2. -/
theorem pairedRecall_weights {V D T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (tokens : Fin T → Fin V) (row dest : Fin T) (i : Fin 256)
    (hq : roles (tokens row) = some i) (hd : dest ≤ row)
    (hs : roles (tokens (pairedPrevious dest)) = some i)
    (hu : ∀ j, j ≤ row → j ≠ dest → roles (tokens (pairedPrevious j)) ≠ some i) :
    sparseWeights (pairedHeadScores (H := 4) (pairedRecallHead roles values) tokens row) row =
      basis dest := by
  have hself := pairedRecall_scores_self roles values tokens row dest i hq hs
  apply sparseWeights_eq_basis_of_gap _ _ _ hd
  intro j hj hne
  rw [hself]
  cases hk : roles (tokens (pairedPrevious j)) with
  | none =>
    rw [pairedRecall_scores_nonkey roles values tokens row j i hq hk]
    norm_num
  | some k =>
    have hik : i ≠ k := by
      intro he
      subst k
      exact hu j hj hne hk
    have hg := pairedRecall_scores_other roles values tokens row j i k hq hk hik
    linarith

/-- Both exchanged tables satisfy every uniqueness and visibility premise of actual routing. -/
example (r : Fin 2) : sparseWeights (pairedHeadScores (H := 4)
    (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) (pairedBindingTokens r) 5) 5 =
      basis 2 := by
  apply pairedRecall_weights (i := 0)
  · fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens]
  · decide
  · fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]
  · intro j hj hn
    fin_cases r <;> fin_cases j <;>
      simp_all [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]

/-- The real head output is the original value following the queried key, with no inverse decoder.
Source: the genuine sparsemax/value product of the repaired §3.1 head. -/
theorem pairedRecall_read_values {V D T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (tokens : Fin T → Fin V) (row dest : Fin T)
    (channel : Fin D) (i : Fin 256) (hq : roles (tokens row) = some i) (hd : dest ≤ row)
    (hs : roles (tokens (pairedPrevious dest)) = some i)
    (hu : ∀ j, j ≤ row → j ≠ dest → roles (tokens (pairedPrevious j)) ≠ some i) :
    pairedHeadOutput (H := 4) (pairedRecallHead roles values) tokens row channel =
      values (tokens dest) channel := by
  rw [pairedHeadOutput_formula, pairedRecall_weights roles values tokens row dest i hq hd hs hu]
  simp [basis, pairedRecallHead]

/-- The two-table inputs satisfy all premises and read their different original values. -/
example (r : Fin 2) : pairedHeadOutput (H := 4)
    (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) (pairedBindingTokens r) 5 0 =
      pairedRecallBindingValues (pairedBindingTokens r 2) 0 := by
  apply pairedRecall_read_values (i := 0)
  · fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens]
  · decide
  · fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]
  · intro j hj hn
    fin_cases r <;> fin_cases j <;>
      simp_all [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]

/-- Any finite collection of uniquely bound answer observations is fit by one genuine compact head.
Source: the new capacity guarantee following Appendix A.4; value labels are ordinary answer data. -/
theorem pairedRecall_single_fit {V D R T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (tokens : Fin R → Fin T → Fin V)
    (rows dests : Fin R → Fin T) (channels : Fin R → Fin D) (keys : Fin R → Fin 256) (answers : Fin R → ℝ)
    (hq : ∀ r, roles (tokens r (rows r)) = some (keys r))
    (hd : ∀ r, dests r ≤ rows r) (hs : ∀ r, roles (tokens r (pairedPrevious (dests r))) = some (keys r))
    (hu : ∀ r j, j ≤ rows r → j ≠ dests r → roles (tokens r (pairedPrevious j)) ≠ some (keys r))
    (ha : ∀ r, values (tokens r (dests r)) (channels r) = answers r) :
    pairedMixtureSample (H := 4) tokens rows channels
      (Finsupp.single (pairedRecallHead roles values) 1) = answers := by
  rw [pairedMixture_single_output]
  ext r
  change pairedHeadOutput (H := 4) (pairedRecallHead roles values) (tokens r) (rows r) (channels r) = _
  rw [pairedRecall_read_values roles values (tokens r) (rows r) (dests r) (channels r)
    (keys r) (hq r) (hd r) (hs r) (hu r)]
  exact ha r

/-- The compact 256-key head fits the same binding witness rejected by every content-only mixture.
Source: the exact physical construction after §3.1, with original independent values. -/
theorem pairedRecall_binding_fit : pairedMixtureSample (H := 4) pairedBindingTokens
    (fun _ => 5) (fun _ => 0)
    (Finsupp.single (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) 1) =
      pairedBindingTarget := by
  apply pairedRecall_single_fit (dests := fun _ => 2) (keys := fun _ => 0)
  · intro r
    fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens]
  · intro r
    decide
  · intro r
    fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]
  · intro r j hj hn
    fin_cases r <;> fin_cases j <;>
      simp_all [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]
  · intro r
    fin_cases r <;> norm_num [pairedRecallBindingValues, pairedBindingTokens, pairedBindingTarget]

/-- The fully physical two-table example satisfies every finite answer-fitting premise. -/
example : pairedMixtureSample (H := 4) pairedBindingTokens (fun _ => 5) (fun _ => 0)
    (Finsupp.single (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) 1) =
      pairedBindingTarget := pairedRecall_binding_fit

/-- The compact fitting witness is a feasible state of the unchanged convex atomic domain.
Source: original value bounds and the numerical unit-mass architecture following Appendix A.4. -/
theorem pairedRecall_binding_state_mem : Finsupp.single
    (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) (1 : ℝ) ∈
      matchingMixtureDomain 5 8 1 4 := by
  apply matchingMixture_single_mem
  apply pairedRecallHead_mem
  intro v d
  fin_cases v <;> norm_num [pairedRecallBindingValues]

/-- The cap-four compact head eliminates the old encoder's sharp error one half as well.
Source: the actual width-eight recall construction after §3.1 and Appendix A.4. -/
theorem pairedRecall_binding_error_zero : pairedBindingError (H := 4)
    (Finsupp.single (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) 1) = 0 := by
  unfold pairedBindingError
  rw [pairedRecall_binding_fit]
  simp

/-- The concrete fitting physical head attains a global minimum of ordinary answer squared error.
Source: the compact recall capacity witness following Appendix A.4, not a route objective. -/
theorem pairedRecall_binding_minimum (μ : MatchingMixture 5 8 1) : pairedBindingError (H := 4)
    (Finsupp.single (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) 1) ≤
      pairedBindingError (H := 4) μ := by
  rw [pairedRecall_binding_error_zero]
  exact Finset.sum_nonneg (fun r hr => sq_nonneg _)

end Transformer.GPTMini.Sparsemax
