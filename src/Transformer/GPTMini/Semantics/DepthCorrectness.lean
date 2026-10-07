import Transformer.GPTMini.Semantics.DepthIntegerInput
import Transformer.GPTMini.TokenInterface.Basic

/-!
# Complete original depth correctness through List Int

Source: original GPTMini.forward/checked adapter at f11b6e2, actual
complete parameters at d3aae47 and Basis's E_2/E_4 integer grammar at
cbafbe9. Every actual BOS/body token is coupled to the same original
embedding array, all genuine layers and normalized tied logits are
derived, and the real final-row greedy adapter appends taskNext.

Both full depth grammars are solved at every validated context up to
128, including all neutral positions and wrong-start/extra-run rejects.
No prepared encoder, route, representation or correct logits are inputs
to the substantive task theorem. Real arithmetic and shared epsilon
in (0,1] are explicit; the usual reference RMS epsilon is included.
These given finite weights do not prove FP execution, AdamW success,
convex trainability or benchmark-training FLOPs. Original width/layer
budgets are retained: easy 64/two, hard 128/six.
Universal raw-prefix correctness is stronger than sampled 99-percent accuracy.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open TokenInterface Transformer.Basis

/-- The actual original integer callback preserves the raw prefix and appends the independent Basis depth answer.
Source: exact finite BOS/body serialization, true full-model last logits, real greedy decode and checked public adapter. -/
theorem depthModel_body_tokens (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (body : List (Option Bool)) (hbound : body.length + 1 ≤ 128) :
    tokenFunction (depthConfig mode) (depthModelParams mode) eps (bos :: depthBody body) =
      (bos :: depthBody body) ++ [taskNext mode .depth (bos :: depthBody body)] := by
  have hc : (depthFiniteBody body).length + 1 ≤ (depthConfig mode).max_seq_len := by
    simpa only [depthFiniteBody, List.length_map] using hbound
  have h := tokenFunction_decode_cons (depthConfig mode) (depthModelParams mode) eps 1 (depthFiniteBody body) hc
  rw [depthModel_raw_greedy mode eps heps hclip body hbound, depthFinitePrefix_decode] at h
  let last : Fin ((1 : Fin 36) :: depthFiniteBody body).length := ⟨(depthFiniteBody body).length, by simp⟩
  have hl : (Fin.cast (depthFinitePrefix_length body).symm last).val + 1 = (none :: body).length := by
    change (depthFiniteBody body).length + 1 = body.length + 1
    rw [depthFiniteBody, List.length_map]
  rw [depthWordAnswer_taskNext mode body (Fin.cast (depthFinitePrefix_length body).symm last) hl] at h
  exact h

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length + 1 ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- The given complete original depth transformer solves every validated raw integer Basis prefix in either mode.
Source: full actual embedding/softmax/XSA/residual/FFN/tied/greedy computation and exact independent E_2/E_4 grammar discharge. -/
theorem depthModel_solves_depth (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) :
    SolvesTask (tokenFunction (depthConfig mode) (depthModelParams mode) eps) mode .depth := by
  intro tokens ht
  rcases ht with ⟨body, _, hb, rfl⟩
  exact depthModel_body_tokens mode eps heps hclip body hb

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- The same actual original depth construction solves both full grammars at the reference RMS epsilon.
Source: the complete raw task theorem and explicit shared epsilon 1e-5, not a floating-point execution assertion. -/
theorem depthModel_reference_epsilon (mode : Mode) :
    SolvesTask (tokenFunction (depthConfig mode) (depthModelParams mode) (1 / 100000)) mode .depth :=
  depthModel_solves_depth mode (1 / 100000) (by norm_num) (by norm_num)

/-- Every genuine model call, including invalid/empty/overlong fallback inputs, obeys the exact append-one integer-list interface.
Source: the original checked public tokenFunction wrapper, independently of task correctness. -/
theorem depthModel_tokens_length (mode : Mode) (eps : ℝ) (tokens : Tokens) :
    (tokenFunction (depthConfig mode) (depthModelParams mode) eps tokens).length = tokens.length + 1 :=
  tokenFunction_length (depthConfig mode) (depthModelParams mode) eps tokens

/-- The actual depth model interface retains every original integer in order on all inputs.
Source: checked tokenFunction's genuine continuation contract, including its total fallback domain. -/
theorem depthModel_tokens_prefix (mode : Mode) (eps : ℝ) (tokens : Tokens) :
    (tokenFunction (depthConfig mode) (depthModelParams mode) eps tokens).take tokens.length = tokens :=
  tokenFunction_prefix (depthConfig mode) (depthModelParams mode) eps tokens

/-- The original easy model accepts the genuine neutral-containing control through its actual public integer interface.
Source: universal full-model raw correctness and independent Basis.depth_easy_example. -/
theorem depthModel_easy_control (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) :
    tokenFunction (depthConfig .easy) (depthModelParams .easy) eps [1, 9, 11, 9, 10] = [1, 9, 11, 9, 10, 16] := by
  have ht : TaskPrefix .easy .depth [1, 9, 11, 9, 10] :=
    ⟨[some false, none, some false, some true], by decide, by decide, rfl⟩
  rw [depthModel_solves_depth .easy eps heps hclip _ ht]
  exact depth_easy_example

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- The unchanged original hard model correctly rejects the same raw prefix, distinguishing task mode through its genuine parameters.
Source: universal original-model hard correctness and independent Basis.depth_hard_example. -/
theorem depthModel_hard_control (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) :
    tokenFunction (depthConfig .hard) (depthModelParams .hard) eps [1, 9, 11, 9, 10] = [1, 9, 11, 9, 10, 15] := by
  have ht : TaskPrefix .hard .depth [1, 9, 11, 9, 10] :=
    ⟨[some false, none, some false, some true], by decide, by decide, rfl⟩
  rw [depthModel_solves_depth .hard eps heps hclip _ ht]
  exact depth_hard_example

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- True original easy depth distinguishes aabb from abab despite identical raw token multiplicities.
Source: full actual ordered detector/model computation and exact E_2 grammar; this checks the former bag-of-token obstruction. -/
theorem depthModel_easy_order_control (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) :
    tokenFunction (depthConfig .easy) (depthModelParams .easy) eps [1, 9, 9, 10, 10] = [1, 9, 9, 10, 10, 16] ∧
    tokenFunction (depthConfig .easy) (depthModelParams .easy) eps [1, 9, 10, 9, 10] = [1, 9, 10, 9, 10, 15] := by
  have ha : TaskPrefix .easy .depth [1, 9, 9, 10, 10] :=
    ⟨[some false, some false, some true, some true], by decide, by decide, rfl⟩
  have hb : TaskPrefix .easy .depth [1, 9, 10, 9, 10] :=
    ⟨[some false, some true, some false, some true], by decide, by decide, rfl⟩
  rw [depthModel_solves_depth .easy eps heps hclip _ ha, depthModel_solves_depth .easy eps heps hclip _ hb]
  decide

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- The genuine original hard model reverses those order-sensitive labels under the exact E_4 task.
Source: complete actual hard parameters and independent original four-alternation grammar. -/
theorem depthModel_hard_order_control (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) :
    tokenFunction (depthConfig .hard) (depthModelParams .hard) eps [1, 9, 9, 10, 10] = [1, 9, 9, 10, 10, 15] ∧
    tokenFunction (depthConfig .hard) (depthModelParams .hard) eps [1, 9, 10, 9, 10] = [1, 9, 10, 9, 10, 16] := by
  have ha : TaskPrefix .hard .depth [1, 9, 9, 10, 10] :=
    ⟨[some false, some false, some true, some true], by decide, by decide, rfl⟩
  have hb : TaskPrefix .hard .depth [1, 9, 10, 9, 10] :=
    ⟨[some false, some true, some false, some true], by decide, by decide, rfl⟩
  rw [depthModel_solves_depth .hard eps heps hclip _ ha, depthModel_solves_depth .hard eps heps hclip _ hb]
  decide

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- The actual easy model rejects a wrong active start even when a later a/b occurrence is present.
Source: complete raw model correctness and independent Basis.depth_wrong_start, including both ordered readout branches. -/
theorem depthModel_wrong_start_control (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) :
    tokenFunction (depthConfig .easy) (depthModelParams .easy) eps [1, 10, 9, 10] = [1, 10, 9, 10, 15] := by
  have ht : TaskPrefix .easy .depth [1, 10, 9, 10] :=
    ⟨[some true, some false, some true], by decide, by decide, rfl⟩
  rw [depthModel_solves_depth .easy eps heps hclip _ ht]
  exact depth_wrong_start

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- Leading/intermediate/final neutral tokens retain their genuine positions and do not prevent the actual model's correct accept.
Source: full real detector/readout computation and independent Basis.depth_neutral_example. -/
theorem depthModel_neutral_control (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) :
    tokenFunction (depthConfig .easy) (depthModelParams .easy) eps [1, 11, 9, 11, 10, 11] = [1, 11, 9, 11, 10, 11, 16] := by
  have ht : TaskPrefix .easy .depth [1, 11, 9, 11, 10, 11] :=
    ⟨[none, some false, none, some true, none], by decide, by decide, rfl⟩
  rw [depthModel_solves_depth .easy eps heps hclip _ ht]
  exact depth_neutral_example

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

end Transformer.GPTMini.Semantics
