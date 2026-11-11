module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Learning

/-!
# Direct construction of a global ordinary-CE optimizer

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF3-CF4.
The input examples are grouped by query. If each row has a consistent key,
the actual finite learner returns unit target logits on every observed
sample, attaining the global minimum over the entire convex channel domain.
This theorem requires no optimizer convergence premise. The explicit fallback
only makes the mathematical channel total when learning fails; the success
theorems guarantee it is unused, and `learn` itself reports failure as `none`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

open scoped BigOperators

variable {Q K V C S : Type*} [DecidableEq K] [DecidableEq V] [DecidableEq C]

/-- A selected key and decoder define a deterministic joint channel (CF3). -/
def channelOf (result : Q → K × (V → C)) : Channel Q K V C :=
  fun q k v c => if k = (result q).1 then (if c = (result q).2 v then 1 else 0) else 0

/-- Use the computed row result; a fallback is explicit for failed rows (CF3). -/
def learnedRow (keys : List K) (samples : List (Sample K V C)) (default : C)
    (fallback : K × (V → C)) : K × (V → C) :=
  (learn keys samples default).getD fallback

/-- The channel assembled from the independently learned query rows (CF3). -/
def learnedChannel (keys : List K) (rows : Q → List (Sample K V C)) (default : C)
    (fallback : K × (V → C)) : Channel Q K V C :=
  channelOf (fun q => learnedRow keys (rows q) default fallback)

omit [DecidableEq V] in
/-- Every returned deterministic channel satisfies all joint mass constraints (CF3). -/
theorem feasible_channelOf [Fintype K] [Fintype C] (result : Q → K × (V → C)) :
    Feasible (channelOf result) := by
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    simp only [channelOf]
    split_ifs <;> norm_num
  · intro q k v w
    by_cases h : k = (result q).1 <;> simp [channelOf, h]
  · intro q v
    simp [channelOf]

omit [DecidableEq V] in
/-- The actual forward reads the learned key and produces its decoded class (CF3). -/
theorem logits_channelOf [Fintype K] (result : Q → K × (V → C))
    (q : Q) (D : K → V) (c : C) :
    logits (channelOf result) q D c = if c = (result q).2 (D (result q).1) then 1 else 0 := by
  simp [logits, channelOf]

omit [DecidableEq K] in
/-- Realizable rows use a successful return and fit every supplied observation (CF3). -/
theorem learnedRow_fits (keys : List K) (samples : List (Sample K V C)) (default : C)
    (fallback : K × (V → C))
    (hreal : ∃ k ∈ keys, ∃ f : V → C, ∀ s ∈ samples, f (s.1 k) = s.2) :
    ∀ s ∈ samples, (learnedRow keys samples default fallback).2
      (s.1 (learnedRow keys samples default fallback).1) = s.2 := by
  obtain ⟨k, hk, f, hf⟩ := hreal
  obtain ⟨a, g, hg⟩ := learn_complete keys samples default k f hk hf
  simpa [learnedRow, hg] using (learn_sound keys samples default hg).2

/-- Every training sample gets the ideal unit target logits from the computed learner (CF3). -/
theorem learned_sample_logits [Fintype K]
    (keys : List K) (rows : Q → List (Sample K V C)) (default : C)
    (fallback : K × (V → C)) (query : S → Q) (data : S → K → V) (label : S → C)
    (hmem : ∀ s, (data s, label s) ∈ rows (query s))
    (hreal : ∀ q, ∃ k ∈ keys, ∃ f : V → C, ∀ s ∈ rows q, f (s.1 k) = s.2)
    (s : S) : logits (learnedChannel keys rows default fallback) (query s) (data s) =
      ConvexRecall.targetLogits (label s) := by
  have hf := learnedRow_fits keys (rows (query s)) default fallback (hreal (query s))
    (data s, label s) (hmem s)
  funext c
  simp only [learnedChannel, logits_channelOf, ConvexRecall.targetLogits]
  rw [hf]

/-- The finite forward learner attains the global ordinary answer-CE minimum
over all feasible joint channels, with no training convergence assumption (CF3). -/
theorem learned_isMinOn [Fintype K] [Fintype C] [Fintype S] [Nonempty V]
    (keys : List K) (rows : Q → List (Sample K V C)) (default : C)
    (fallback : K × (V → C)) (query : S → Q) (data : S → K → V) (label : S → C)
    (hmem : ∀ s, (data s, label s) ∈ rows (query s))
    (hreal : ∀ q, ∃ k ∈ keys, ∃ f : V → C, ∀ s ∈ rows q, f (s.1 k) = s.2) :
    Feasible (learnedChannel keys rows default fallback) ∧
      IsMinOn (answerLoss query data label) {W | Feasible W}
        (learnedChannel keys rows default fallback) := by
  refine ⟨feasible_channelOf _, ?_⟩
  intro W hW
  apply Finset.sum_le_sum
  intro s _
  rw [learned_sample_logits keys rows default fallback query data label hmem hreal]
  exact ConvexRecall.target_crossEntropy_le _ (logits_nonneg hW (query s) (data s))
    (logits_le_one hW (query s) (data s) _)

/-- Collision and coverage conditions recover the whole generated-recall
tensor, hence all test dictionaries, after the finite scan (CF4). -/
theorem learned_eq_generator (keys : List K) (rows : K → List (Sample K V C))
    (default : C) (fallback : K × (V → C)) (g : K → V → C)
    (hkeys : ∀ q, q ∈ keys)
    (hlabel : ∀ q s, s ∈ rows q → g q (s.1 q) = s.2)
    (hcover : ∀ q v, ∃ s ∈ rows q, s.1 q = v)
    (hconflict : ∀ q k, k ∈ keys → k ≠ q → ∃ s ∈ rows q, ∃ t ∈ rows q,
      s.1 k = t.1 k ∧ s.2 ≠ t.2) :
    learnedChannel keys rows default fallback = generator g := by
  have hr (q : K) := learn_eq_identified keys (rows q) default q (g q)
    (hkeys q) (hlabel q) (hcover q) (hconflict q)
  funext q k v c
  simp [learnedChannel, channelOf, learnedRow, hr q, generator]

example := learnedRow_fits [false] [(id, true)] false (true, id)
  ⟨false, by simp, Bool.not, by simp⟩

example := learned_sample_logits [false] (fun _ : Unit => [(id, true)]) false (true, id)
  (fun _ : Unit => ()) (fun _ => id) (fun _ => true) (by simp)
  (fun _ => ⟨false, by simp, Bool.not, by simp⟩) ()

example := learned_isMinOn [false] (fun _ : Unit => [(id, true)]) false (true, id)
  (fun _ : Unit => ()) (fun _ => id) (fun _ => true) (by simp)
  (fun _ => ⟨false, by simp, Bool.not, by simp⟩)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward
