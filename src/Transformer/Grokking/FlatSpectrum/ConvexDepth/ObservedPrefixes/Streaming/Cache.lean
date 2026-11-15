module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes.Streaming.Basic

/-!
# A bounded stream cache must retain cheaper equivalent queries

Source: `review/flat_spectrum.md`, CM62. Complete observed signatures preserve
the indexed count comparisons. A previously tested cheaper syntax also
preserves the combined size cap. A more expensive cached syntax does not
justify dropping a cheaper arrival. Evicting a cache entry permits repeated
work; it must not prevent that candidate from being checked again.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes.Streaming

variable {A : Type*}

/-- Full prefix signatures, with closure and observed-subset premises,
preserve the actual comparison fit for either streamed side (CM62). -/
theorem countFit_congr (pool observed : Finset (List A)) (labels : List A → Bool)
    (left altLeft right altRight extra : List A → Bool) (offset : ℤ)
    (hclosed : PrefixClosed pool) (hobs : observed ⊆ pool)
    (hl : ObservedEq pool left altLeft) (hr : ObservedEq pool right altRight) :
    CountFit observed labels left right extra offset ↔
      CountFit observed labels altLeft altRight extra offset := by
  have heq (w : List A) (hw : w ∈ observed) :
      decide ((prefixCount left w : ℤ) < prefixCount right w + prefixCount extra w + offset) =
      decide ((prefixCount altLeft w : ℤ) < prefixCount altRight w + prefixCount extra w + offset) := by
    rw [prefixCount_congr pool left altLeft hclosed hl (hobs hw),
      prefixCount_congr pool right altRight hclosed hr (hobs hw)]
  constructor
  · intro h w hw
    rw [← heq w hw]
    exact h w hw
  · intro h w hw
    rw [heq w hw]
    exact h w hw

/-- Equal full signatures and no larger measured child sizes preserve a
size-bounded observed fit (CM62). Native syntax sizes must be checked against
their ASTs; no Rust heap or global semantic-equivalence theorem is asserted. -/
theorem cheaper_pair_preserves_fit (pool observed : Finset (List A))
    (labels : List A → Bool) (left altLeft right altRight extra : List A → Bool)
    (offset : ℤ) (hclosed : PrefixClosed pool) (hobs : observed ⊆ pool)
    (hl : ObservedEq pool left altLeft) (hr : ObservedEq pool right altRight)
    (leftSize altLeftSize rightSize altRightSize overhead limit : ℕ)
    (hleft : altLeftSize ≤ leftSize) (hright : altRightSize ≤ rightSize)
    (hsize : leftSize + rightSize + overhead ≤ limit)
    (hfit : CountFit observed labels left right extra offset) :
    CountFit observed labels altLeft altRight extra offset ∧
      altLeftSize + altRightSize + overhead ≤ limit := by
  constructor
  · exact (countFit_congr pool observed labels left altLeft right altRight extra offset
      hclosed hobs hl hr).mp hfit
  · omega

/-- A previously tested no-larger equivalent pair justifies reusing its
refusal (CM62). Cache eviction is not such a certificate: absent entries must
be tested again, and a smaller arrival need not satisfy these premises. -/
theorem cached_refusal_preserved (pool observed : Finset (List A))
    (labels : List A → Bool) (left altLeft right altRight extra : List A → Bool)
    (offset : ℤ) (hclosed : PrefixClosed pool) (hobs : observed ⊆ pool)
    (hl : ObservedEq pool left altLeft) (hr : ObservedEq pool right altRight)
    (leftSize altLeftSize rightSize altRightSize overhead limit : ℕ)
    (hleft : altLeftSize ≤ leftSize) (hright : altRightSize ≤ rightSize)
    (hrefused : ¬ (CountFit observed labels altLeft altRight extra offset ∧
      altLeftSize + altRightSize + overhead ≤ limit)) :
    ¬ (CountFit observed labels left right extra offset ∧
      leftSize + rightSize + overhead ≤ limit) := by
  rintro ⟨hfit, hsize⟩
  exact hrefused (cheaper_pair_preserves_fit pool observed labels left altLeft right altRight
    extra offset hclosed hobs hl hr leftSize altLeftSize rightSize altRightSize overhead limit
    hleft hright hsize hfit)

/-- Two observed words have different target truth values (CM62's witness). -/
def toyLabels (w : List Bool) : Bool := decide (w = [true])

/-- A concrete fit uses both target truth values and actual inclusive counts
(CM62). It is the elapsed-count versus last-symbol-count comparison. -/
theorem toy_fit : CountFit toyPool toyLabels (fun _ => true) lastTrue (fun _ => false) 1 := by
  intro w hw
  have hpool : toyPool = {[true], [true, false]} := by decide
  rw [hpool] at hw
  have hcases : w = [true] ∨ w = [true, false] := by
    simpa only [Finset.mem_insert, Finset.mem_singleton] using hw
  rcases hcases with rfl | rfl <;> decide

/-- Keeping only the constant-false child misses the second observed label
even though another candidate fits both words (CM62). -/
theorem toy_false_refused :
    ¬ CountFit toyPool toyLabels (fun _ => false) lastTrue (fun _ => false) 1 := by
  intro h
  have hbad := h [true, false] (by decide)
  change true = false at hbad
  cases hbad

/-- Actual finite count programs witness why a capped retained bank cannot
certify absence of a fit in the complete bank (CM62). -/
theorem truncated_bank_misses_fit :
    (∃ b ∈ ({false, true} : Finset Bool),
      CountFit toyPool toyLabels (fun _ => b) lastTrue (fun _ => false) 1) ∧
    ¬ (∃ b ∈ ({false} : Finset Bool),
      CountFit toyPool toyLabels (fun _ => b) lastTrue (fun _ => false) 1) := by
  constructor
  · exact ⟨true, by decide, toy_fit⟩
  · rintro ⟨b, hb, hfit⟩
    have heq : b = false := Finset.mem_singleton.mp hb
    subst b
    exact toy_false_refused hfit

/-- Both observed target values occur in the streaming witness (CM62). -/
theorem toy_nonconstant_targets : toyLabels [true] = true ∧ toyLabels [true, false] = false := by
  decide

/-- A double-negated symbol program has the same observed signature as the
one-node symbol. Its native tree has three nodes rather than one (CM62). -/
theorem double_negated_observed : ObservedEq toyPool (fun w => !(!lastTrue w)) lastTrue := by
  intro w _
  exact Bool.not_not _

/-- For the witness's actual tree costs, a three-node constant-true anchor
and five-node count/offset wrapper fit under nine nodes only with the cheaper
symbol query. A larger cached equivalent cannot justify skipping it (CM62). -/
theorem size_cache_counterexample :
    ObservedEq toyPool (fun w => !(!lastTrue w)) lastTrue ∧
    CountFit toyPool toyLabels (fun _ => true) lastTrue (fun _ => false) 1 ∧
    ¬ (3 + 3 + 5 ≤ (9 : ℕ)) ∧ 3 + 1 + 5 ≤ (9 : ℕ) := by
  exact ⟨double_negated_observed, toy_fit, by decide, by decide⟩

example : CountFit toyPool toyLabels (fun _ => true) onlyFirst (fun _ => false) 1 :=
  (countFit_congr toyPool toyPool toyLabels (fun _ => true) (fun _ => true)
    lastTrue onlyFirst (fun _ => false) 1 (closure_closed _) (fun _ h => h)
    (fun _ _ => rfl) toy_observed).mp toy_fit

example : CountFit toyPool toyLabels (fun _ => true) lastTrue (fun _ => false) 1 ∧
    3 + 1 + 5 ≤ (11 : ℕ) := by
  have hfit := (countFit_congr toyPool toyPool toyLabels (fun _ => true) (fun _ => true)
    (fun w => !(!lastTrue w)) lastTrue (fun _ => false) 1 (closure_closed _)
    (fun _ h => h) (fun _ _ => rfl) double_negated_observed).mpr toy_fit
  exact cheaper_pair_preserves_fit toyPool toyPool toyLabels (fun _ => true) (fun _ => true)
    (fun w => !(!lastTrue w)) lastTrue (fun _ => false) 1 (closure_closed _) (fun _ h => h)
    (fun _ _ => rfl) double_negated_observed 3 3 3 1 5 11
    (by decide) (by decide) (by decide) hfit

example := countFit_left_bounds toyPool toyLabels (fun _ => true) lastTrue (fun _ => false) 1
example := countFit_right_bounds toyPool toyLabels (fun _ => true) lastTrue (fun _ => false) 1

example : ¬ (CountFit toyPool toyLabels (fun _ => false) lastTrue (fun _ => false) 1 ∧
    1 + 1 + 5 ≤ (16 : ℕ)) :=
  cached_refusal_preserved toyPool toyPool toyLabels (fun _ => false) (fun _ => false)
    lastTrue lastTrue (fun _ => false) 1 (closure_closed _) (fun _ h => h)
    (fun _ _ => rfl) (fun _ _ => rfl) 1 1 1 1 5 16 (by decide) (by decide)
    (fun h => toy_false_refused h.1)

example := stream_left_candidates ({()} : Finset Unit) (Finset.univ : Finset Bool)
  (fun b => {b}) toyPool toyLabels (fun b _ => b) (fun _ => lastTrue) (fun _ => false) 1

example := stream_right_candidates ({()} : Finset Unit) (Finset.univ : Finset Bool)
  (fun b => {b}) toyPool toyLabels (fun _ _ => true)
  (fun b => if b then lastTrue else fun _ => true) (fun _ => false) 1

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes.Streaming
