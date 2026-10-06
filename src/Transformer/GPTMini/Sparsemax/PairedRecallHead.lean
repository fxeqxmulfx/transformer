import Transformer.GPTMini.Sparsemax.PairedRecallGeometry

/-!
# A genuine width-eight recall head with bounded original values

New compact witness for the repaired architecture following arXiv:2211.11052v1,
§3.1. Four predecessor-role coordinates hold the 256 integer key codes; four
current-role coordinates are zero in this particular witness. Query/key
tables are ordinary physical parameters. The architecture does not constrain
them to these codes or to each other during learning.

An observed input role identifies key tokens. Other tokens get zero key code.
Independent original value tables remain arbitrary. Identical keys have score
thirty, distinct keys score at most twenty-nine, and nonkeys score zero. All
coordinates fit cap four. Later routing uses actual sparsemax Eq. (1) from
arXiv:1602.02068v2, not a supplied inference response or pair-value dictionary.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The physical token table of the compact witness, with a zero current-key role.
Source: the new signed-permutation embedding construction after §3.1. -/
def pairedRecallTable {V : ℕ} (roles : Fin V → Option (Fin 256)) : Matrix (Fin V) (Fin 8) ℝ :=
  fun v d => if hd : d.val < 4 then 0 else match roles v with
    | none => 0
    | some i => (pairedRecallCode i ⟨d.val - 4, by omega⟩ : ℝ)

/-- One genuine physical head, preserving every independently chosen original value coordinate.
Source: the repaired §3.1 head; equal Q/K here is a witness choice, not a training restriction. -/
def pairedRecallHead {V D : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) : MatchingHead V 8 D :=
  ((pairedRecallTable roles, pairedRecallTable roles), values)

/-- Bounded original values give a feasible compact recall head at the experiment's cap four.
Source: the numerical head box following Appendix A.4, using the explicit integer codes. -/
theorem pairedRecallHead_mem {V D : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (hv : ∀ v d, -4 ≤ values v d ∧ values v d ≤ 4) :
    pairedRecallHead roles values ∈ matchingHeadBox V 8 D 4 := by
  have hc (v : Fin V) (d : Fin 8) : -4 ≤ pairedRecallTable roles v d ∧ pairedRecallTable roles v d ≤ 4 := by
    unfold pairedRecallTable
    split_ifs
    · norm_num
    · cases roles v with
      | none => norm_num
      | some i =>
        dsimp only
        exact pairedRecallCode_real_bounds i ⟨d.val - 4, by omega⟩
  exact ⟨hc, hc, hv⟩

/-- Key roles for the real two-table binding witness; value/BOS tokens are nonkeys.
Source: the ordinary input format of the new associative recall example after §3.1. -/
def pairedRecallBindingRoles (v : Fin 5) : Option (Fin 256) :=
  if v = 1 then some 0 else if v = 2 then some 1 else none

/-- The independent scalar values of the two-table witness.
Source: the original value readout in the repaired binding example after §3.1. -/
def pairedRecallBindingValues (v : Fin 5) : Fin 1 → ℝ :=
  fun _ => if v = 4 then 1 else 0

/-- The example's actual original values satisfy every compact-head box premise. -/
example : pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues ∈ matchingHeadBox 5 8 1 4 := by
  apply pairedRecallHead_mem
  intro v d
  fin_cases v <;> norm_num [pairedRecallBindingValues]

/-- A query and a predecessor key use the actual inner product of their compact codes.
Source: genuine dot products of the repaired §3.1 head, before sparsemax Eq. (1). -/
theorem pairedRecall_scores_key {V D T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (tokens : Fin T → Fin V) (row j : Fin T)
    (i k : Fin 256) (hq : roles (tokens row) = some i)
    (hk : roles (tokens (pairedPrevious j)) = some k) :
    pairedHeadScores (H := 4) (pairedRecallHead roles values) tokens row j =
      ∑ d, (pairedRecallCode i d : ℝ) * (pairedRecallCode k d : ℝ) := by
  rw [pairedHeadScores_formula]
  norm_num [pairedRecallHead, pairedRecallTable, hq, hk, Fin.sum_univ_succ]

/-- The correct written value position satisfies both physical key-role premises. -/
example : pairedHeadScores (H := 4) (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues)
    (pairedBindingTokens 0) 5 2 = ∑ d, (pairedRecallCode 0 d : ℝ) * (pairedRecallCode 0 d : ℝ) := by
  apply pairedRecall_scores_key
  · norm_num [pairedRecallBindingRoles, pairedBindingTokens]
  · norm_num [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]

/-- A predecessor which is not a key has score zero against any recognized query.
Source: the compact physical token lookup, not a masked-out supplied attention target. -/
theorem pairedRecall_scores_nonkey {V D T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (tokens : Fin T → Fin V) (row j : Fin T)
    (i : Fin 256) (hq : roles (tokens row) = some i)
    (hk : roles (tokens (pairedPrevious j)) = none) :
    pairedHeadScores (H := 4) (pairedRecallHead roles values) tokens row j = 0 := by
  rw [pairedHeadScores_formula]
  norm_num [pairedRecallHead, pairedRecallTable, hq, hk, Fin.sum_univ_succ]

/-- The visible BOS position inhabits both query/nonkey premises of the actual forward. -/
example : pairedHeadScores (H := 4) (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues)
    (pairedBindingTokens 0) 5 0 = 0 := by
  apply pairedRecall_scores_nonkey (i := 0)
  · norm_num [pairedRecallBindingRoles, pairedBindingTokens]
  · norm_num [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]

/-- Any correctly matched predecessor has score thirty for this genuine physical witness.
Source: the equal-norm signed key codes in the new §3.1 construction. -/
theorem pairedRecall_scores_self {V D T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (tokens : Fin T → Fin V) (row j : Fin T)
    (i : Fin 256) (hq : roles (tokens row) = some i)
    (hk : roles (tokens (pairedPrevious j)) = some i) :
    pairedHeadScores (H := 4) (pairedRecallHead roles values) tokens row j = 30 := by
  rw [pairedRecall_scores_key roles values tokens row j i i hq hk, pairedRecallCode_real_self]

/-- Both swapped tables satisfy the matched-key premises, despite their different correct values. -/
example (r : Fin 2) : pairedHeadScores (H := 4)
    (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) (pairedBindingTokens r) 5 2 = 30 := by
  apply pairedRecall_scores_self (i := 0)
  · fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens]
  · fin_cases r <;> norm_num [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]

/-- Distinct predecessor keys have scores at least one below the correct match.
Source: the integer geometry certificate before genuine sparsemax Eq. (1). -/
theorem pairedRecall_scores_other {V D T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (tokens : Fin T → Fin V) (row j : Fin T)
    (i k : Fin 256) (hq : roles (tokens row) = some i)
    (hk : roles (tokens (pairedPrevious j)) = some k) (hne : i ≠ k) :
    pairedHeadScores (H := 4) (pairedRecallHead roles values) tokens row j ≤ 29 := by
  rw [pairedRecall_scores_key roles values tokens row j i k hq hk]
  exact pairedRecallCode_real_gap i k hne

/-- The other written value position satisfies every distinct-key score premise. -/
example : pairedHeadScores (H := 4) (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues)
    (pairedBindingTokens 0) 5 4 ≤ 29 := by
  apply pairedRecall_scores_other (i := 0) (k := 1)
  · norm_num [pairedRecallBindingRoles, pairedBindingTokens]
  · norm_num [pairedRecallBindingRoles, pairedBindingTokens, pairedPrevious]
  · decide

/-- Every output keeps the original-value cap, independent of contextual support changes.
Source: the numerical value box and genuine sparsemax Eq. (1) in the new §3.1 head. -/
theorem pairedRecallHead_outputBounds {V D T : ℕ} (roles : Fin V → Option (Fin 256))
    (values : Matrix (Fin V) (Fin D) ℝ) (hv : ∀ v d, -4 ≤ values v d ∧ values v d ≤ 4)
    (tokens : Fin T → Fin V) (row : Fin T) (channel : Fin D) :
    -4 ≤ pairedHeadOutput (H := 4) (pairedRecallHead roles values) tokens row channel ∧
      pairedHeadOutput (H := 4) (pairedRecallHead roles values) tokens row channel ≤ 4 :=
  pairedHeadOutput_bounds (H := 4) 4 _ (pairedRecallHead_mem roles values hv) _ _ _

/-- Both differently bound tables satisfy all original-value output-bound premises. -/
example (r : Fin 2) : -4 ≤ pairedHeadOutput (H := 4)
    (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues) (pairedBindingTokens r) 5 0 ∧
    pairedHeadOutput (H := 4) (pairedRecallHead pairedRecallBindingRoles pairedRecallBindingValues)
      (pairedBindingTokens r) 5 0 ≤ 4 := by
  apply pairedRecallHead_outputBounds
  intro v d
  fin_cases v <;> norm_num [pairedRecallBindingValues]

end Transformer.GPTMini.Sparsemax
