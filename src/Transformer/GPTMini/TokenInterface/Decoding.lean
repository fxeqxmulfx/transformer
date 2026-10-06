import Transformer.GPTMini.Properties.OutputSimplex
import Mathlib.Data.List.MinMax
import Mathlib.Tactic

/-!
# Deterministic decoding of actual GPTMini outputs

Source: GPTMini.forward in the archived gpt_mini.py at f11b6e2 and the
new List Int continuation adapter. Greedy decoding chooses the first
maximizer in ascending vocabulary order. The model's attention remains
the causal softmax/QKNorm/XSA head; this file concerns its final readout.

No correct-answer oracle enters the decoder. A strict logit certificate
is sufficient to verify a prediction. Establishing the required logit
property for given parameters remains a task-dependent verification step.
-/

namespace Transformer.GPTMini.TokenInterface

open scoped Classical
noncomputable section

/-- Greedy prediction with the smallest token ID winning ties.
Source: the new deterministic counterpart of torch.argmax over the vocabulary. -/
def bestToken {V : ℕ} (hV : 0 < V) (score : Fin V → ℝ) : Fin V :=
  ((List.finRange V).argmax score).getD ⟨0, hV⟩

/-- The fallback in bestToken is never used for a nonempty vocabulary.
Source: List.argmax on all finite vocabulary IDs, with the model's positivity invariant. -/
theorem bestToken_some {V : ℕ} (hV : 0 < V) (score : Fin V → ℝ) :
    (List.finRange V).argmax score = some (bestToken hV score) := by
  cases he : (List.finRange V).argmax score with
  | none =>
      have hn := List.argmax_eq_none.mp he
      have hz := List.mem_finRange (⟨0, hV⟩ : Fin V)
      rw [hn] at hz
      contradiction
  | some token => simp [bestToken, he]

example : 0 < 3 := by omega

/-- The decoder selects a global maximum, even when several logits are tied.
Source: the actual finite-vocabulary greedy decoding rule. -/
theorem bestToken_le {V : ℕ} (hV : 0 < V) (score : Fin V → ℝ) (v : Fin V) :
    score v ≤ score (bestToken hV score) :=
  List.le_of_mem_argmax (List.mem_finRange v) (bestToken_some hV score)

example : 0 < 26 := by omega

/-- Exact greedy correctness needs a maximum and the earliest ID among tied maxima.
Source: List.argmax's complete characterization on the ascending vocabulary enumeration. -/
theorem bestToken_eq_iff {V : ℕ} (hV : 0 < V) (score : Fin V → ℝ) (answer : Fin V) :
    bestToken hV score = answer ↔
      (∀ v, score v ≤ score answer) ∧
        ∀ v, score answer ≤ score v → answer.val ≤ v.val := by
  have he : bestToken hV score = answer ↔
      (List.finRange V).argmax score = some answer := by
    constructor
    · intro h
      rw [← h]
      exact bestToken_some hV score
    · intro h
      simp [bestToken, h]
  rw [he, List.argmax_eq_some_iff]
  simp only [List.mem_finRange, List.idxOf_finRange, true_and, forall_const]

example : 0 < 36 := by omega

/-- A strict margin over every other vocabulary token verifies the discrete answer.
Source: the new sufficient prediction certificate; it does not assume attention routes. -/
theorem bestToken_of_strict {V : ℕ} (hV : 0 < V) (score : Fin V → ℝ) (answer : Fin V)
    (hmargin : ∀ v, v ≠ answer → score v < score answer) : bestToken hV score = answer := by
  by_contra hne
  exact not_lt_of_ge (bestToken_le hV score answer) (hmargin _ hne)

example : 0 < 3 ∧ ∀ v : Fin 3, v ≠ 1 →
    (if v = 1 then (2 : ℝ) else 0) < (if (1 : Fin 3) = 1 then 2 else 0) := by
  refine ⟨by omega, ?_⟩
  intro v hv
  simp [hv]

/-- Order-equivalent scores give exactly the same answer, including the tie rule.
Source: the new decoder bridge, stronger than merely sharing some maximizer. -/
theorem bestToken_congr_order {V : ℕ} (hV : 0 < V) (f g : Fin V → ℝ)
    (horder : ∀ a b, f a ≤ f b ↔ g a ≤ g b) : bestToken hV f = bestToken hV g := by
  have hf := List.argmax_eq_some_iff.mp (bestToken_some hV f)
  have hg : (List.finRange V).argmax g = some (bestToken hV f) :=
    List.argmax_eq_some_iff.mpr ⟨hf.1,
      fun a ha => (horder a _).mp (hf.2.1 a ha),
      fun a ha hga => hf.2.2 a ha ((horder _ a).mpr hga)⟩
  simp only [bestToken, hg, Option.getD_some]

example : 0 < 3 ∧ ∀ a b : Fin 3,
    (a.val : ℝ) ≤ b.val ↔ 2 * (a.val : ℝ) + 1 ≤ 2 * (b.val : ℝ) + 1 := by
  refine ⟨by omega, ?_⟩
  intro a b
  constructor <;> intro h <;> linarith

/-- Equal logits select token zero, so uninformative readouts cannot pass by arbitrary ties.
Source: torch.argmax's first-index rule, proved from the ordered vocabulary enumeration. -/
theorem bestToken_const {V : ℕ} (hV : 0 < V) (c : ℝ) :
    bestToken hV (fun _ => c) = ⟨0, hV⟩ := by
  have he : (List.finRange V).argmax (fun _ => c) = some ⟨0, hV⟩ := by
    apply List.argmax_eq_some_iff.mpr
    refine ⟨List.mem_finRange _, fun _ _ => le_rfl, ?_⟩
    intro a ha hle
    simp only [List.idxOf_finRange]
    exact Nat.zero_le a.val
  simp [bestToken, he]

example : 0 < 26 := by omega

/-- A common logit shift never changes the discrete continuation.
Source: the actual greedy decoder's comparison rule, including tied outputs. -/
theorem bestToken_add_shift {V : ℕ} (hV : 0 < V) (score : Fin V → ℝ) (c : ℝ) :
    bestToken hV (fun v => score v + c) = bestToken hV score := by
  apply bestToken_congr_order
  intro a b
  exact add_le_add_iff_right c

example : 0 < 36 := by omega

/-- Output softmax preserves all weak logit comparisons in the actual forward pass.
Source: GPTMini.Model.softmaxOutput; its common denominator is strictly positive. -/
theorem softmaxOutput_le_iff (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (a b : Fin cfg.vocab_size) :
    softmaxOutput cfg params eps positions tokens i a ≤
        softmaxOutput cfg params eps positions tokens i b ↔
      forward cfg params eps positions tokens i a ≤ forward cfg params eps positions tokens i b := by
  unfold softmaxOutput
  rw [div_le_div_iff_of_pos_right
    (Properties.softmaxOutput_denom_pos cfg params eps positions tokens i cfg.vocab_pos),
    Real.exp_le_exp]

/-- Output softmax also preserves strict margins, without an infinite-temperature limit.
Source: the actual GPTMini final softmax and strict monotonicity of exp. -/
theorem softmaxOutput_lt_iff (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (a b : Fin cfg.vocab_size) :
    softmaxOutput cfg params eps positions tokens i a <
        softmaxOutput cfg params eps positions tokens i b ↔
      forward cfg params eps positions tokens i a < forward cfg params eps positions tokens i b := by
  unfold softmaxOutput
  rw [div_lt_div_iff_of_pos_right
    (Properties.softmaxOutput_denom_pos cfg params eps positions tokens i cfg.vocab_pos),
    Real.exp_lt_exp]

/-- Greedy logits and greedy output probabilities are identical, including ties.
Source: GPTMini's complete forward, not a surrogate head or a task oracle. -/
theorem bestToken_softmaxOutput (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) :
    bestToken cfg.vocab_pos (forward cfg params eps positions tokens i) =
      bestToken cfg.vocab_pos (softmaxOutput cfg params eps positions tokens i) := by
  apply bestToken_congr_order
  intro a b
  exact (softmaxOutput_le_iff cfg params eps positions tokens i a b).symm

/-- A finite positive logit margin is enough to make output-softmax decoding exact.
Source: the new certificate rule; exact decoding does not mean probability one. -/
theorem bestToken_softmax_of_strict {V : ℕ} (hV : 0 < V) (score : Fin V → ℝ)
    (answer : Fin V) (hmargin : ∀ v, v ≠ answer → score v < score answer) :
    bestToken hV (fun v => Real.exp (score v)) = answer := by
  apply bestToken_of_strict
  intro v hv
  exact Real.exp_lt_exp.mpr (hmargin v hv)

example : 0 < 3 ∧ ∀ v : Fin 3, v ≠ 2 →
    (if v = 2 then (1 : ℝ) else -1) < (if (2 : Fin 3) = 2 then 1 else -1) := by
  refine ⟨by omega, ?_⟩
  intro v hv
  simp [hv]

end
end Transformer.GPTMini.TokenInterface
