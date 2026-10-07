import Transformer.GPTMini.Semantics.RecallParserLatest
import Transformer.Basis.Encoding.Recall

/-!
# Complete original softmax GPTMini recall through List Int -> List Int

Source: MQAR.targets and Basis's eight/sixteen-write recipes at cbafbe9,
GPTMini.forward at f11b6e2, and the actual operator/margin proofs from
RecallEmbedding through RecallReadout. The given ordinary two-layer,
64-wide parameter family computes every validated raw recall answer
through its actual checked final-row greedy integer adapter.

The original raw parser derives BOS, all alphabet/adjacency conditions,
the finite selected last write and its chronology. No prepared state,
paired-key input, routing certificate, model score or correct logit is
an assumption of these final correctness results. All 256 keys/values,
earlier different-valued overwrites, later unrelated writes, arbitrary
validated fillers and earlier queries within context 64 are covered.

This is universal real-arithmetic capacity at shared epsilon in [0,1],
including both actual Basis recall grammars. The architecture remains
the small width-64/two-layer model even for the hard grammar; a lift to
the experiment's large width-128/six-layer model is not proved here.
The shared formal head/RMS epsilon is not Python's unequal defaults.
Neither AdamW finding these weights, floating-point correctness, full
depth correctness nor convexity of the trainable parameters follows.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- The actual integer callback appends the true raw answer from complete successful parsing of the same finite input.
Source: the genuine full-model callback and parser-derived layout/last-write conditions; no semantic model premise remains. -/
theorem recallModel_parsed_cons (P : ℕ) (rewrites : Bool) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    (head : Fin recallConfig.vocab_size) (tail : List (Fin recallConfig.vocab_size)) (hlen : tail.length + 1 ≤ 64)
    (answer : ℤ) (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    tokenFunction recallConfig (recallModelParams P eps) eps (decodeTokens (head :: tail)) =
      decodeTokens (head :: tail) ++ [answer] := by
  obtain ⟨selected, previous, key, value, hq, hp, hk, hv, hin, hvis, hl, ha⟩ :=
    recallParser_selected P rewrites head tail answer hanswer
  have h := recallModel_tokens P eps heps hclip head tail hlen ⟨0, by simp⟩ selected previous
    (recallParser_layout P rewrites head tail answer hanswer) key value hq hp hk hv hin hvis hl
  rw [ha] at h
  exact h

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by
  exact ⟨by norm_num, by norm_num, by decide, recallOverwriteTokens_answer⟩

/-- Every accepted original integer recall prefix yields its computed raw last value through the genuine model interface.
Source: full parser legality, exact checked finite encoding, actual context cap and the derived whole-model greedy answer. -/
theorem recallModel_parsed_tokens (P : ℕ) (rewrites : Bool) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    (tokens : Tokens) (hlen : tokens.length ≤ 64) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ tokens = some answer) :
    tokenFunction recallConfig (recallModelParams P eps) eps tokens = tokens ++ [answer] := by
  have hlegal : Encoding.Legal recallConfig.vocab_size tokens := Encoding.answer_legal _ _ _ hanswer
  obtain ⟨encoded, hencode⟩ := Encoding.encode_exists tokens hlegal
  have hd := decode_encode hencode
  have hc : encoded.length ≤ 64 := by rw [encode_length hencode]; exact hlen
  have hp : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens encoded) = some answer := by rw [hd]; exact hanswer
  cases encoded with
  | nil => simp only [decodeTokens, List.map_nil, recallAnswer] at hp; contradiction
  | cons head tail =>
      rw [← hd]
      exact recallModel_parsed_cons P rewrites eps heps hclip head tail hc answer hp

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ [1, 36, 292, 36, 293, 36] = some 293 := by
  exact ⟨by norm_num, by norm_num, by decide, by decide⟩

/-- The genuine original model agrees with the independent full raw recall function on every supervised prefix.
Source: validated integer recall grammar and the actual callback's derived prediction, including hard-mode last overwrites. -/
theorem recallModel_prefix_correct (P : ℕ) (rewrites : Bool) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    (tokens : Tokens) (hprefix : RecallPrefix ⟨256, P, rewrites⟩ 64 tokens) :
    tokenFunction recallConfig (recallModelParams P eps) eps tokens = recallFunction ⟨256, P, rewrites⟩ tokens := by
  obtain ⟨hlen, answer, hanswer⟩ := hprefix
  rw [(recallFunction_of_answer _ _ _ hanswer).1]
  exact recallModel_parsed_tokens P rewrites eps heps hclip tokens hlen answer hanswer

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    RecallPrefix ⟨256, 16, true⟩ 64 (rewritePrefix 292 300) := by
  exact ⟨by norm_num, by norm_num, by decide, 300, by decide⟩

/-- Given original compact parameters solve the complete raw Basis recall grammar in either mode, through List Int -> List Int.
Source: actual easy eight-write/hard sixteen-write recipes and full prefix correctness; no correct-logit or SolvesTask premise. -/
theorem recallModel_solves_recall (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) (mode : Mode) :
    SolvesTask (tokenFunction recallConfig (recallModelParams (if mode = .easy then 8 else 16) eps) eps) mode .recall := by
  intro tokens hprefix
  cases mode with
  | easy =>
      change tokenFunction recallConfig (recallModelParams 8 eps) eps tokens = recallFunction easyRecall tokens
      exact recallModel_prefix_correct 8 false eps heps hclip tokens hprefix
  | hard =>
      change tokenFunction recallConfig (recallModelParams 16 eps) eps tokens = recallFunction hardRecall tokens
      exact recallModel_prefix_correct 16 true eps heps hclip tokens hprefix

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- At the original RMS epsilon, the given compact model solves every validated recall prefix in both Basis recipes.
Source: RMS epsilon 1e-5 in the archived model, with the documented shared formal head epsilon and real arithmetic. -/
theorem recallModel_reference_rms_solves_recall (mode : Mode) :
    SolvesTask (tokenFunction recallConfig (recallModelParams (if mode = .easy then 8 else 16) (1 / 100000))
      (1 / 100000)) mode .recall :=
  recallModel_solves_recall _ (by norm_num) (by norm_num) mode

/-- Concrete compact parameter assignments simultaneously cover the actual eight-write and sixteen-write Basis grammars.
Source: both complete raw correctness results, with separate task recipes as in the benchmark. -/
theorem recallModel_reference_both_modes :
    SolvesTask (tokenFunction recallConfig (recallModelParams 8 (1 / 100000)) (1 / 100000)) .easy .recall ∧
      SolvesTask (tokenFunction recallConfig (recallModelParams 16 (1 / 100000)) (1 / 100000)) .hard .recall := by
  exact ⟨recallModel_reference_rms_solves_recall .easy, recallModel_reference_rms_solves_recall .hard⟩

/-- The actual easy eight-write control returns key 36's first bound value while retaining all seven later unrelated records.
Source: the full original raw bindingPrefix control, not the earlier two-record overwrite witness. -/
theorem recallModel_easy_control (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) :
    tokenFunction recallConfig (recallModelParams 8 eps) eps (bindingPrefix 292 293) = bindingPrefix 292 293 ++ [292] := by
  exact recallModel_parsed_tokens 8 false eps heps hclip _ (by decide) 292 (by decide)

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- The actual hard sixteen-write control returns the ninth write's changed value and retains all seven later unrelated rewrites.
Source: Basis hard MQAR's eight keys and eight changed-value overwrites, with the genuine full raw input. -/
theorem recallModel_hard_control (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) :
    tokenFunction recallConfig (recallModelParams 16 eps) eps (rewritePrefix 292 300) = rewritePrefix 292 300 ++ [300] := by
  exact recallModel_parsed_tokens 16 true eps heps hclip _ (by decide) 300 (by decide)

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- Swapping actual neighboring values changes the original model's answer, repairing the bag encoder's binding collision.
Source: the real eight-write easy controls with identical token bags and query key, but different raw adjacent bindings. -/
theorem recallModel_swapped_bindings (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) :
    tokenFunction recallConfig (recallModelParams 8 eps) eps (bindingPrefix 292 293) = bindingPrefix 292 293 ++ [292] ∧
      tokenFunction recallConfig (recallModelParams 8 eps) eps (bindingPrefix 293 292) = bindingPrefix 293 292 ++ [293] := by
  exact ⟨recallModel_easy_control eps heps hclip,
    recallModel_parsed_tokens 8 false eps heps hclip _ (by decide) 293 (by decide)⟩

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-!
The same fixed given parameters handle every accepted prefix in each
recipe. Neither the selected record nor the desired answer is passed to
the model: both occur only in the independent data-side parser proof.
Universal correctness includes input lengths shorter than the sampler's
minimum and all legal intermediate query positions. The ordinary public
adapter still preserves input and appends exactly one token on every
integer list, with its specified fallback outside the supervised grammar.
-/

end Transformer.GPTMini.Semantics
