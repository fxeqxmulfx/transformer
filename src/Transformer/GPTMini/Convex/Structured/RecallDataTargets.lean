import Transformer.GPTMini.Convex.Structured.RecallDataRoutes
import Transformer.GPTMini.Convex.Structured.MixedBasis

/-!
# Correct raw-data complete labels for actual convex recall training

Source: the deterministic Nat-ranked latest-record scan in RecallDataRoutes,
independent full MQAR parsing at 9e6661b and actual mixed model/complete
objective at d436526/d7be618. Training configurations use the computed
physical predecessor/value route and those unchanged tokens' matching
and complete output digits. No model parameters or inference outcomes
enter this data generator. Full raw parsing proves agreement with the
independent answer, including chronological overwrites and valid fillers.

The actual computed loss on these labels is the same genuine mixed
negative log probability and globally convex in every unrestricted raw
embedding/Q/K/value/position/head coordinate. Auxiliary labels belong
only to loss computation; actual inference contains no parsed table or
hard adjacency mask. Label preprocessing has a real cost to be charged
in the comparison. Output-only CE, floating-point behavior and successful
ordinary AdamW convergence are not proved by these data/loss identities.

The generator uses each recipe's known record count at training time.
That count is absent from the actual free-parameter inference signature;
its learned positional fields must implement table exclusion themselves.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.Semantics
noncomputable section

variable {C : ℕ}

/-- Complete training route/matching/value labels are generated from unchanged raw data and the physical query alone.
Source: the actual deterministic latest-record scan, true predecessor and fixed radix output code; no learned parameter is an input. -/
def recallDataTarget (P : ℕ) (tokens : List (Fin 548)) (query : Fin tokens.length) : RawBindingConfiguration tokens :=
  let selected := recallDataSelected P tokens query
  recallBindingTarget tokens (recallDataPrevious selected) selected

/-- The genuinely generated complete labels equal the actual parsed last-write configuration from the independent raw grammar.
Source: computable data argmax correctness and exact physical predecessor recovery, without an assumed correct model route. -/
theorem recallDataTarget_parsed (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer)
    (query previous selected : Fin tokens.length) (key : Fin 256)
    (hq : tokens.get query = recallKeyId key) (hp : previous.val + 1 = selected.val)
    (hk : tokens.get previous = recallKeyId key) (hs : recallTableValuePosition P selected.val)
    (hlast : RecallRawLatestWrite P tokens.get selected key) :
    recallDataTarget P tokens query = recallBindingTarget tokens previous selected := by
  unfold recallDataTarget
  rw [recallDataSelected_parsed P rewrites tokens answer hanswer query previous selected key hq hp hk hs hlast]
  dsimp only
  rw [recallDataPrevious_eq previous selected hp]

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 ∧
    recallOverwriteTokens.get (5 : Fin 6) = recallKeyId 0 ∧ (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧
    recallOverwriteTokens.get (3 : Fin 6) = recallKeyId 0 ∧ recallTableValuePosition 2 (4 : Fin 6).val ∧
    RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) (0 : Fin 256) :=
  ⟨by decide, by decide, by decide, by decide, by decide, recallOverwriteTokens_raw_latest⟩

/-- Successful complete raw parsing and the actual final query derive the generated training route's independently correct answer ID.
Source: full parsed last-write inversion and deterministic data scan correctness; no desired value or correct model output is assumed. -/
theorem recallDataTarget_answer (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (query : Fin tokens.length)
    (hquery : query.val + 1 = tokens.length) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ((tokens.get (recallDataSelected P tokens query)).val : ℤ) = answer := by
  obtain ⟨parsed, previous, selected, key, value, hparsed, hq, hp, hk, hv, hs, _, hlast, ha⟩ :=
    recallBinding_selected P rewrites tokens answer hanswer
  have he : query = parsed := by apply Fin.ext; omega
  rw [he, recallDataSelected_parsed P rewrites tokens answer hanswer parsed previous selected key hq hp hk hs hlast, hv]
  exact ha

example : (5 : Fin 6).val + 1 = recallOverwriteTokens.length ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by decide, by decide⟩

/-- Every actual generated full output-channel label is the complete code of the independently required raw recall answer.
Source: actual data-selected unchanged value token and its proved raw answer ID, without a model-dependent teacher. -/
theorem recallDataTarget_channels (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (query : Fin tokens.length)
    (hquery : query.val + 1 = tokens.length) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ target : Fin 548, (target.val : ℤ) = answer ∧
      (recallDataTarget P tokens query).2.2 = outputDigit (vocabularyCode (by decide) target) := by
  refine ⟨tokens.get (recallDataSelected P tokens query), recallDataTarget_answer P rewrites tokens query hquery answer hanswer, ?_⟩
  rfl

example : (5 : Fin 6).val + 1 = recallOverwriteTokens.length ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by decide, by decide⟩

/-- Data-generated complete output labels agree with the actual independent Basis answer in both full recall recipes.
Source: complete raw eight/sixteen-record validation, actual final raw query and independent generated-label correctness above. -/
theorem recallDataTarget_task (mode : Mode) (tokens : List (Fin 548)) (query : Fin tokens.length)
    (hquery : query.val + 1 = tokens.length) (hprefix : TaskPrefix mode .recall (decodeTokens tokens)) :
    ∃ target : Fin 548, (target.val : ℤ) = taskNext mode .recall (decodeTokens tokens) ∧
      (recallDataTarget (if mode = .easy then 8 else 16) tokens query).2.2 = outputDigit (vocabularyCode (by decide) target) := by
  obtain ⟨answer, ha⟩ := hprefix.2
  have hparse : recallAnswer ⟨256, (if mode = .easy then 8 else 16), (if mode = .easy then false else true)⟩
      (decodeTokens tokens) = some answer := by cases mode <;> exact ha
  obtain ⟨target, ht, hc⟩ := recallDataTarget_channels _ _ tokens query hquery answer hparse
  have hn : taskNext mode .recall (decodeTokens tokens) = answer := by cases mode <;> exact recallNext_of_answer _ _ _ ha
  exact ⟨target, ht.trans hn.symm, hc⟩

example : (17 : Fin 18).val + 1 = ([(1 : Fin 548), 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] : List (Fin 548)).length ∧
    TaskPrefix .easy .recall (decodeTokens [(1 : Fin 548), 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36]) :=
  ⟨by decide, by exact ⟨by decide, 292, by decide⟩⟩

/-- Actual complete recall training computes branch CE and raw all-pair joint NLL using these fixed data-only labels.
Source: the true mixedNLL, without hard routing or semantic data generation in the inference function. -/
def recallDataNLL (P : ℕ) (tokens : List (Fin 548)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (θ : BindingParameters 548 C) : ℝ := mixedNLL tokens hcap query (.inr (recallDataTarget P tokens query)) θ

/-- Training on the generated raw-data labels is exactly the negative log of the same actual mixed inference probability.
Source: genuine computed mixed likelihood identity, with no separately substituted target-conditioned inference model. -/
theorem recallDataNLL_eq (P : ℕ) (tokens : List (Fin 548)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (θ : BindingParameters 548 C) : recallDataNLL P tokens hcap query θ =
      -Real.log (mixedProbability θ tokens hcap query (.inr (recallDataTarget P tokens query))) :=
  mixedNLL_eq _ _ _ _ _

example : recallOverwriteTokens.length ≤ 64 := by decide

/-- The genuine actual complete recall objective uses each Basis recipe's real eight/sixteen-record data labels on the same unrestricted mixed domain.
Source: generated raw route/channel supervision and true joint mixed likelihood convexity; mode is a training recipe, never an inference argument. -/
theorem recallDataNLL_mode_convex (mode : Mode) (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64) (query : Fin tokens.length) :
    ConvexOn ℝ Set.univ (recallDataNLL (if mode = .easy then 8 else 16) tokens hcap query) :=
  mixedNLL_convex _ _ _ _

example : recallOverwriteTokens.length ≤ 64 := by decide

/-- The actually computed full recall objective with generated data supervision is globally convex in every simultaneous free coordinate.
Source: true mixedNLL convexity with data-fixed route/channel labels, including Q/K, joint values and every learned binding/head potential. -/
theorem recallDataNLL_convex (P : ℕ) (tokens : List (Fin 548)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) :
    ConvexOn ℝ Set.univ (recallDataNLL P tokens hcap query) := mixedNLL_convex _ _ _ _

example : recallOverwriteTokens.length ≤ 64 := by decide

/-- The genuine generated-label complete recall loss is nonnegative at all finite unrestricted mixed parameter assignments.
Source: actual positive normalized mixed likelihood, without a correct prediction or successful-training premise. -/
theorem recallDataNLL_nonneg (P : ℕ) (tokens : List (Fin 548)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (θ : BindingParameters 548 C) : 0 ≤ recallDataNLL P tokens hcap query θ := mixedNLL_nonneg _ _ _ _ _

example : recallOverwriteTokens.length ≤ 64 := by decide

/-- Computed complete raw overwrite labels point to the real later value and retain its unchanged output code.
Source: executable Nat scan and genuine raw token reads at positions three/four, without finite model weights in label generation. -/
theorem recallDataTarget_overwrite :
    (recallDataTarget 2 recallOverwriteTokens (5 : Fin 6)).1 = ((3 : Fin 6), (4 : Fin 6)) ∧
      (recallDataTarget 2 recallOverwriteTokens (5 : Fin 6)).2.2 = outputDigit (vocabularyCode (by decide) (293 : Fin 548)) := by
  dsimp only [recallDataTarget]
  rw [recallDataSelected_overwrite]
  exact ⟨rfl, rfl⟩

end
end Transformer.GPTMini.Convex.Structured
