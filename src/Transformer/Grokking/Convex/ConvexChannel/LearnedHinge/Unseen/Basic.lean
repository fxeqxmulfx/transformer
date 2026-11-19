module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction
public import Mathlib.Data.List.OfFn

/-!
# A learned raw-prefix neuron can isolate one unobserved full prefix

Source: `review/convex/moments.md`, CM77. The representation is a permissible
learned bias and position/object weight vector. Its ReLU activates only on
one full-length object prefix, and is zero on every other admitted prefix,
including shorter prefixes. This is a parameter witness, not a task program
supplied to a learner or an assumption about answers.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen

open scoped BigOperators

variable {O : Type*} [DecidableEq O] {T : ℕ}

/-- Count actual object matches at every original position, CM77. -/
def matchCount (needle : Fin T → O) (tokens : Prefix O T) : ℕ :=
  (Finset.univ.filter (fun i : Fin T => tokens.val[i.val]? = some (needle i))).card

/-- Missing positions cannot count as matches, CM77. -/
theorem matchCount_le (needle : Fin T → O) (tokens : Prefix O T) : matchCount needle tokens ≤ T := by
  exact (Finset.card_filter_le _ _).trans_eq (by simp)

/-- All positions match exactly when the complete raw lists agree, CM77.
The input length cap is used only to rule out unrepresented extra positions. -/
theorem matchCount_eq_iff (needle : Fin T → O) (tokens : Prefix O T) :
    matchCount needle tokens = T ↔ tokens.val = List.ofFn needle := by
  have hcount : matchCount needle tokens = T ↔
      ∀ i : Fin T, tokens.val[i.val]? = some (needle i) := by
    simpa only [matchCount, Finset.card_univ, Fintype.card_fin, Finset.mem_univ, forall_const]
      using (Finset.card_filter_eq_iff (s := Finset.univ)
        (p := fun i : Fin T => tokens.val[i.val]? = some (needle i)))
  rw [hcount]
  constructor
  · intro h
    apply List.ext_getElem?
    intro n
    by_cases hn : n < T
    · simpa only [List.getElem?_ofFn, dite_eq_left hn] using h ⟨n, hn⟩
    · have hlen : tokens.val.length ≤ n := tokens.property.trans (by omega)
      rw [List.getElem?_eq_none hlen, List.getElem?_ofFn, dite_eq_right hn]
  · intro h i
    rw [h, List.getElem?_ofFn, dite_eq_left i.isLt]

/-- A different admitted prefix loses at least one whole matching position,
CM77. This provides the strict activation gap without an approximate threshold. -/
theorem matchCount_lt (needle : Fin T → O) (tokens : Prefix O T)
    (hne : tokens.val ≠ List.ofFn needle) : matchCount needle tokens < T := by
  have hle := matchCount_le needle tokens
  have hne' : matchCount needle tokens ≠ T := fun h => hne ((matchCount_eq_iff needle tokens).mp h)
  omega

/-- The real preactivation formed by learned raw object weights, CM77. -/
def score (needle : Fin T → O) (tokens : Prefix O T) : ℝ :=
  1 - (T : ℝ) + matchCount needle tokens

/-- The exact target prefix has unit preactivation, CM77. -/
theorem score_self (needle : Fin T → O) :
    score needle ⟨List.ofFn needle, by simp⟩ = 1 := by
  have h := (matchCount_eq_iff needle ⟨List.ofFn needle, by simp⟩).mpr rfl
  simp [score, h]

/-- Every other admitted prefix has nonpositive preactivation, CM77. -/
theorem score_nonpos (needle : Fin T → O) (tokens : Prefix O T)
    (hne : tokens.val ≠ List.ofFn needle) : score needle tokens ≤ 0 := by
  have h : matchCount needle tokens + 1 ≤ T := matchCount_lt needle tokens hne
  have hr : (matchCount needle tokens : ℝ) + 1 ≤ (T : ℝ) := by exact_mod_cast h
  unfold score
  linarith

/-- The actual rectified response is the singleton-prefix indicator, CM77.
The indicator is proved from learned linear weights in the following module. -/
theorem hinge_score (needle : Fin T → O) (tokens : Prefix O T) :
    hinge (score needle tokens) = if tokens.val = List.ofFn needle then 1 else 0 := by
  by_cases h : tokens.val = List.ofFn needle
  · have hc := (matchCount_eq_iff needle tokens).mpr h
    simp [score, hc, h, hinge]
  · rw [ite_eq_right h]
    exact max_eq_right (score_nonpos needle tokens h)

example := matchCount_lt (fun _ : Fin 2 => true) (⟨[false, true], by decide⟩ : Prefix Bool 2)
  (by decide)
example := score_nonpos (fun _ : Fin 2 => true) (⟨[true], by decide⟩ : Prefix Bool 2)
  (by decide)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen
