import Transformer.GPTMini.Convex.Structured.LearnedParityParameters

/-!
# Kernel-checked finite rational learned-row certificates

Source: exact completed checkpoint coordinates in LearnedParityData and
independent original parity vocabulary/channel labels. Every saved row's
actual target and width are explicit. The kernel checks all six runs and
all relevant rational rival comparisons with gap twenty; padding columns
are excluded. Lean's kernel reduces every rational comparison; no exp
estimate or assumed correct prediction is used. Real coordinate gaps
follow by ordered rational casts.

The initial and branch rows target zero. The nine transition targets
retain BOS, counted bit phases, SEP and original answer completion. The
fifteen emission targets are the original base-four channel digits of
EVEN=24, ODD=25 and EOS=17. Their relation to the real learned value
coordinate layout is independently checked below, before any inference
correctness transfer. The rational coordinate data came from completed
ordinary-AdamW runs, not finite sharp-row construction weights.
-/

namespace Transformer.GPTMini.Convex.Structured

/-- Actual row widths of the saved original initial/transition/emission/branch tables.
Source: six states, four output channels and two learned mixture branches; zero storage padding is not a competitor. -/
def learnedParityRowWidth (row : Fin 26) : ℕ :=
  if row.val < 10 then 6 else if row.val < 25 then 4 else 2

/-- Independently specified correct state/channel indices for each actual saved row.
Source: original parity bit/SEP/answer transitions and base-four digits of raw EVEN/ODD/EOS IDs, not learned argmaxes. -/
def learnedParityRowTarget : Fin 26 → Fin 6 :=
  ![0, 0, 0, 1, 1, 0, 2, 3, 4, 4, 0, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 0, 1, 0, 0, 0]

/-- Every independently specified saved target lies in its original actual row, including the two-channel branch.
Source: the complete explicit finite width/target tables, checked by the kernel. -/
theorem learnedParity_targets_valid : ∀ row, (learnedParityRowTarget row).val < learnedParityRowWidth row := by
  decide

set_option maxHeartbeats 1000000 in
/-- All six actual rational snapshots have gap at least twenty against every physical rival in every required learned row.
Source: literal exact gained dyadic logits, independently specified data targets and original row widths, checked by kernel computation. -/
theorem learnedParity_rows_gap_twenty : ∀ (run : Fin 6) (row : Fin 26) (rival : Fin 6),
    rival.val < learnedParityRowWidth row → rival ≠ learnedParityRowTarget row →
    learnedParityRows run row rival ≤ learnedParityRows run row (learnedParityRowTarget row) - 20 := by
  decide +kernel

example : (1 : Fin 6).val < learnedParityRowWidth 0 ∧ (1 : Fin 6) ≠ learnedParityRowTarget 0 := by decide

/-- Every saved actual rational row therefore meets the sufficient gap-eleven certificate, with no transcendental check.
Source: the kernel-derived stronger gap twenty and the exact numerical comparison eleven <= twenty. -/
theorem learnedParity_rows_gap_eleven (run : Fin 6) (row : Fin 26) (rival : Fin 6)
    (hwidth : rival.val < learnedParityRowWidth row) (hne : rival ≠ learnedParityRowTarget row) :
    learnedParityRows run row rival ≤ learnedParityRows run row (learnedParityRowTarget row) - 11 := by
  have hg := learnedParity_rows_gap_twenty run row rival hwidth hne
  linarith

example : (1 : Fin 6).val < learnedParityRowWidth 25 ∧ (1 : Fin 6) ≠ learnedParityRowTarget 25 := by decide

/-- The exact rational gaps are the same actual unrestricted real-coordinate inequalities used by true softmax confidence.
Source: ordered rational-to-real embedding and the independently kernel-checked saved row comparisons. -/
theorem learnedParity_rows_real_gap (run : Fin 6) (row : Fin 26) (rival : Fin 6)
    (hwidth : rival.val < learnedParityRowWidth row) (hne : rival ≠ learnedParityRowTarget row) :
    ((learnedParityRows run row rival : ℚ) : ℝ) ≤
      ((learnedParityRows run row (learnedParityRowTarget row) : ℚ) : ℝ) - 11 := by
  have hg := learnedParity_rows_gap_eleven run row rival hwidth hne
  exact_mod_cast hg

example : (1 : Fin 6).val < learnedParityRowWidth 9 ∧ (1 : Fin 6) ≠ learnedParityRowTarget 9 := by decide

/-- The initial row's physical target remains state zero independently of the checkpoint.
Source: raw parity starts at zero; this is a finite data-index identity, not a correctly inferred state premise. -/
theorem learnedParity_initial_target : learnedParityRowTarget 0 = 0 := by decide

/-- The actual saved learned mixing row's checked target is the state branch zero.
Source: original complete raw parity branch labels, independent of all learned branch values. -/
theorem learnedParity_branch_target : learnedParityRowTarget 25 = 0 := by decide

/-- Every needed value row has exactly the independent true parity label's actual output-channel code.
Source: original saved state/channel index order and real vocabularyCode/outputDigit of EVEN/ODD/EOS, exhaustively checked at all fifteen rows. -/
theorem learnedParity_emission_target (state : Fin 6) (h : Fin 5) (hs : 2 ≤ state.val ∧ state.val ≤ 4) :
    learnedParityRowTarget (learnedParityEmissionRow state h hs) =
      ⟨(outputDigit (vocabularyCode (by decide) (paritySharedLabel state)) h).val, by
        have hd := (outputDigit (vocabularyCode (by decide) (paritySharedLabel state)) h).isLt
        omega⟩ := by
  fin_cases state <;> norm_num at hs
  all_goals fin_cases h <;> apply Fin.ext <;>
    norm_num [learnedParityEmissionRow, learnedParityRowTarget, paritySharedLabel, vocabularyCode, outputDigit]

example : 2 ≤ (2 : Fin 6).val ∧ (2 : Fin 6).val ≤ 4 := by decide

/-- Each required saved value row has its original four physical channels, so storage padding never enters the confidence bound.
Source: actual saved state/channel indices ten through twenty-four and original learned emission width four. -/
theorem learnedParity_emission_width (state : Fin 6) (h : Fin 5) (hs : 2 ≤ state.val ∧ state.val ≤ 4) :
    learnedParityRowWidth (learnedParityEmissionRow state h hs) = 4 := by
  have hh := h.isLt
  have hlo : ¬ (learnedParityEmissionRow state h hs).val < 10 := by
    change ¬ 10 + 5 * (state.val - 2) + h.val < 10
    omega
  have hhi : (learnedParityEmissionRow state h hs).val < 25 := by
    change 10 + 5 * (state.val - 2) + h.val < 25
    omega
  unfold learnedParityRowWidth
  rw [ite_eq_right hlo, ite_eq_left hhi]

example : 2 ≤ (4 : Fin 6).val ∧ (4 : Fin 6).val ≤ 4 := by decide

/-- The stored transition target order agrees with all nine independently required genuine raw parity transitions.
Source: BOS, four raw bit cases, two SEP cases and both original answer/completion cases; no learned transition is replaced. -/
theorem learnedParity_transition_targets :
    learnedParityRowTarget 1 = paritySharedRule 1 0 ∧
    learnedParityRowTarget 2 = paritySharedRule 21 0 ∧
    learnedParityRowTarget 3 = paritySharedRule 21 1 ∧
    learnedParityRowTarget 4 = paritySharedRule 22 0 ∧
    learnedParityRowTarget 5 = paritySharedRule 22 1 ∧
    learnedParityRowTarget 6 = paritySharedRule 18 0 ∧
    learnedParityRowTarget 7 = paritySharedRule 18 1 ∧
    learnedParityRowTarget 8 = paritySharedRule 24 2 ∧
    learnedParityRowTarget 9 = paritySharedRule 25 3 := by
  decide

/-- Real saved initial logits meet the exact sufficient gap-eleven predicate at the actual full parameter family.
Source: genuine initial read recovery and kernel-checked original six-state rivals, with all other full fields arbitrary. -/
theorem learnedParity_initial_gap (run : Fin 6) (other : BindingParameters 68 19) :
    RowLogitGap (fun state => sharedInitialRead state (learnedParityParameters run other).1) 0 11 := by
  intro rival hr
  simp_rw [learnedParity_initial_read]
  have hw : rival.val < learnedParityRowWidth 0 := by
    change rival.val < 6
    exact rival.isLt
  have hg := learnedParity_rows_real_gap run 0 rival hw (by rw [learnedParity_initial_target]; exact hr)
  rw [learnedParity_initial_target] at hg
  exact hg

/-- Real saved learned branch logits meet the sufficient gap without an assumed correct task switch in inference.
Source: actual mixedHeadRead recovery, true original two-channel rivals and exact saved rational confidence conditions. -/
theorem learnedParity_branch_gap (run : Fin 6) (other : BindingParameters 68 19) :
    RowLogitGap (fun head => mixedHeadRead head (learnedParityParameters run other)) 0 11 := by
  intro rival hr
  have hw : rival.val < learnedParityRowWidth 25 := by
    change rival.val < 2
    exact rival.isLt
  have hn : (⟨rival.val, by have h := rival.isLt; omega⟩ : Fin 6) ≠ learnedParityRowTarget 25 := by
    rw [learnedParity_branch_target]
    intro he
    have hv := congrArg Fin.val he
    apply hr
    exact Fin.ext hv
  have hg := learnedParity_rows_real_gap run 25 ⟨rival.val, by have h := rival.isLt; omega⟩ hw hn
  rw [learnedParity_branch_target] at hg
  simp_rw [learnedParity_head_read]
  exact hg

end Transformer.GPTMini.Convex.Structured
