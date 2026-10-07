import Transformer.GPTMini.Convex.Structured.SharedInterface
import Transformer.GPTMini.Convex.Structured.DepthReference
import Transformer.Basis.Encoding.Tasks

/-!
# Actual finite learned causal head solves both raw Basis depth modes

Source: Basis E2/E4 recipes at cbafbe9, their independent raw reference
semantics at 6c91a98 and actual shared finite stochastic weights at
d72d3b3. The same unrestricted 52-field/state-emission head class as
parity is used. Only data targets and given task/mode parameters differ,
as in the benchmark. Inference never calls a run counter or depth oracle.

One finite shared table handles every valid raw prefix through the true
normalized state/value mean and checked greedy integer-list callback.
This head-level correctness is not full prenorm/residual/tied stack
realization or AdamW/finite-precision convergence. Complete recall and
two-head mixing remain open. The globally convex training objective uses
explicit raw data-derived complete state/channel labels, not output-only CE.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- Data/witness transitions consume the actual finite depth token IDs and preserve their integer semantics.
Source: the independently proved raw six-state rule, used only for supervision and given-weight construction. -/
def depthSharedRule (token : Fin 36) : Fin 6 → Fin 6 := depthRawReference (token.val : ℤ)

/-- Checked finite/raw serialization preserves every real reference transition and its chronological endpoint.
Source: referenceRun_map and Basis.decodeTokens; neither order nor neutral positions are discarded by the raw scan. -/
theorem depthShared_run (tokens : List (Fin 36)) :
    referenceRun depthSharedRule 0 tokens = referenceRun depthRawReference 0 (decodeTokens tokens) := by
  unfold decodeTokens
  rw [referenceRun_map]
  rfl

/-- A single finite common vocabulary/head weight table for a depth task, independent of all its raw prefixes.
Source: actual shared finite transition/initial/emission assignment and gain log(100000); k selects value targets only. -/
def depthSharedParameters (k : ℕ) : SharedParameters 36 128 :=
  referenceSharedParameters depthSharedRule (fun state => vocabularyCode (by decide) (depthReferenceLabel k state)) 0 referenceGain

/-- The actual freely learned standalone head exposes the same complete integer-list continuation interface.
Source: checked 36-token/128-position learned state/value callback and real greedy decoder, without semantic inference code. -/
def depthSharedFunction (k : ℕ) : Tokens → Tokens :=
  sharedMarkovFunction (by decide) (by decide) (depthSharedParameters k)

/-- Actual finite learned rows and emissions predict the independent raw depth answer on every bounded valid prefix.
Source: full raw finite-state semantic proof, genuine joint confidence, all-vocabulary margins, exact input encoding and actual greedy readout. -/
theorem depthShared_next (k : ℕ) (hk : 0 < k) (hcap : k < 5) (tokens : Tokens) (hprefix : DepthPrefix 128 tokens) :
    sharedMarkovNext (by decide) (by decide) (depthSharedParameters k) tokens = depthNext k tokens := by
  obtain ⟨head, tail, he, hlen⟩ := Encoding.depth_inputs .easy tokens hprefix
  have h128 : (head :: tail).length ≤ 128 := by
    simpa only [List.length_cons] using hlen
  rw [sharedMarkovNext_of_encode (by decide) (by decide) (depthSharedParameters k) tokens head tail he hlen]
  unfold depthSharedParameters
  rw [referenceShared_best (by decide) (by decide) depthSharedRule (depthReferenceLabel k) 0 (head :: tail) h128,
    depthShared_run, decode_encode he]
  exact depthRawReference_next k 128 hk hcap tokens hprefix

example : (0 : ℕ) < 2 ∧ (2 : ℕ) < 5 ∧ DepthPrefix 128 [1, 9, 11, 10] :=
  ⟨by omega, by omega, [some false, none, some true], by decide, by decide, rfl⟩

/-- The genuine standalone learned state/value callback solves the entire raw depth grammar in either Basis mode.
Source: actual k=2/k=4 task recipes and independently derived head predictions, with no correct-logit or encoder premise. -/
theorem depthShared_solves (mode : Mode) :
    SolvesTask (depthSharedFunction (if mode = .easy then 2 else 4)) mode .depth := by
  apply (solvesTask_extend_iff _ mode .depth).mpr
  intro tokens hprefix
  apply depthShared_next _ _ _ tokens hprefix
  · cases mode <;> decide
  · cases mode <;> decide

/-- Real intermediate state/output-channel supervision is generated from the actual raw training word.
Source: independently verified physical depth transitions and chronological referencePath; labels never enter the actual forward interface. -/
def depthSharedTargets (k : ℕ) (tokens : List (Fin 36)) : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) tokens.length :=
  (0, (referencePath depthSharedRule 0 tokens,
    outputDigit (vocabularyCode (by decide) (depthReferenceLabel k (referenceRun depthSharedRule 0 tokens)))))

/-- The actual complete data-target likelihood is globally convex in every free shared transition, initial and output-value weight.
Source: sharedMarkovNLL_convex on the same unrestricted parameter domain used by the true inference callback; no encoder or values are frozen. -/
theorem depthShared_training_convex (k : ℕ) (tokens : List (Fin 36)) :
    ConvexOn ℝ Set.univ (markovNLL (sharedInitialRead (V := 36) (C := 128))
      sharedTransitionRead sharedEmissionRead tokens (depthSharedTargets k tokens)) :=
  sharedMarkovNLL_convex tokens (depthSharedTargets k tokens)

/-- Actual complete depth output targets are the independent checked raw Basis answer's true decoder digits.
Source: full raw finite-state semantic equivalence and exact finite/raw serialization, with only true raw prefix validity assumed. -/
theorem depthSharedTargets_answer (k : ℕ) (hk : 0 < k) (hcap : k < 5) (tokens : List (Fin 36))
    (hprefix : DepthPrefix 128 (decodeTokens tokens)) :
    (depthSharedTargets k tokens).2.2 = outputDigit
      ⟨(depthNext k (decodeTokens tokens)).toNat, by
        have h := depthRawReference_next k 128 hk hcap (decodeTokens tokens) hprefix
        omega⟩ := by
  have h := depthRawReference_next k 128 hk hcap (decodeTokens tokens) hprefix
  dsimp only [depthSharedTargets]
  apply congrArg outputDigit
  apply Fin.ext
  change (depthReferenceLabel k (referenceRun depthSharedRule 0 tokens)).val = (depthNext k (decodeTokens tokens)).toNat
  rw [depthShared_run]
  omega

example : (0 : ℕ) < 4 ∧ (4 : ℕ) < 5 ∧ DepthPrefix 128 (decodeTokens [1, 9, 10, 9, (10 : Fin 36)]) :=
  ⟨by omega, by omega, [some false, some true, some false, some true], by decide, by decide, rfl⟩

/-- The same actual learned k=2 callback gives opposite labels on equal-multiplicity differently ordered raw inputs.
Source: genuine whole-task correctness and independent Basis subsequence semantics, excluding the bag-of-tokens encoder collision. -/
theorem depthShared_order_control :
    depthSharedFunction 2 [1, 9, 9, 10, 10] = [1, 9, 9, 10, 10, 16] ∧
    depthSharedFunction 2 [1, 9, 10, 9, 10] = [1, 9, 10, 9, 10, 15] := by
  have hp : DepthPrefix 128 [1, 9, 9, 10, 10] :=
    ⟨[some false, some false, some true, some true], by decide, by decide, rfl⟩
  have hn : DepthPrefix 128 [1, 9, 10, 9, 10] :=
    ⟨[some false, some true, some false, some true], by decide, by decide, rfl⟩
  have hpa : depthNext 2 [1, 9, 9, 10, 10] = 16 := by decide
  have hna : depthNext 2 [1, 9, 10, 9, 10] = 15 := by decide
  constructor
  · have h := depthShared_next 2 (by omega) (by omega) _ hp
    rw [hpa] at h
    exact congrArg (fun token : ℤ => [1, 9, 9, 10, 10] ++ [token]) h
  · have h := depthShared_next 2 (by omega) (by omega) _ hn
    rw [hna] at h
    exact congrArg (fun token : ℤ => [1, 9, 10, 9, 10] ++ [token]) h

/-- The same raw word has its required distinct easy and hard labels under the corresponding finite shared task weights.
Source: both complete mode theorems and the independently computed E2/E4 difference, with no mode passed to inference. -/
theorem depthShared_mode_control :
    depthSharedFunction 2 [1, 9, 11, 9, 10] = [1, 9, 11, 9, 10, 16] ∧
    depthSharedFunction 4 [1, 9, 11, 9, 10] = [1, 9, 11, 9, 10, 15] := by
  have hp : DepthPrefix 128 [1, 9, 11, 9, 10] :=
    ⟨[some false, none, some false, some true], by decide, by decide, rfl⟩
  have hpa : depthNext 2 [1, 9, 11, 9, 10] = 16 := by decide
  have hna : depthNext 4 [1, 9, 11, 9, 10] = 15 := by decide
  constructor
  · have h := depthShared_next 2 (by omega) (by omega) _ hp
    rw [hpa] at h
    exact congrArg (fun token : ℤ => [1, 9, 11, 9, 10] ++ [token]) h
  · have h := depthShared_next 4 (by omega) (by omega) _ hp
    rw [hna] at h
    exact congrArg (fun token : ℤ => [1, 9, 11, 9, 10] ++ [token]) h

/-- Every actual raw learned-depth callback retains the required length contract on its entire integer-list domain.
Source: the genuine shared learned-head checked adapter, including invalid and overlong inputs. -/
theorem depthShared_length (k : ℕ) (tokens : Tokens) : (depthSharedFunction k tokens).length = tokens.length + 1 :=
  sharedMarkovFunction_length _ _ _ _

/-- All actual raw tokens remain in their original order through every learned-depth head callback.
Source: the same checked shared inference interface, independent of task validity and given-weight correctness. -/
theorem depthShared_prefix (k : ℕ) (tokens : Tokens) : (depthSharedFunction k tokens).take tokens.length = tokens :=
  sharedMarkovFunction_prefix _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
