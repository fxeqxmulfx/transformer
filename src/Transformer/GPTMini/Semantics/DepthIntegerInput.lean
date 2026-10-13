import Transformer.GPTMini.Semantics.DepthModel

/-!
# Exact raw integer/finite depth input and original-model coupling

Source: Basis vocabulary/depthBody at cbafbe9 and the actual complete
original depth parameters at d3aae47. The finite body serializes only
the current raw letter, and its true BOS entry remains token one.
The neutral leading internal position has the same real embedding as
BOS, while the original integer IDs are kept distinct by serialization.

All original positions and lengths are retained. Finite list decoding
is exactly the validated integer prefix, and every actual embedding
lookup is coupled to the raw-word array. A harmless length transport
connects the same original forward call, without any operator change.
The independent last-position semantic target is proved to be precisely
Basis's integer taskNext in both E_2/E_4 modes. These are data/model
representation bridges; full checked adapter correctness follows.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- The actual finite depth body, with one ordinary vocabulary token per raw letter including neutrals.
Source: depthBody's A/B/neutral IDs and the same token-local depthLetterToken serializer. -/
def depthFiniteBody (body : List (Option Bool)) : List (Fin 36) := body.map depthLetterToken

/-- The internal leading neutral and actual BOS-prefixed finite list have identical original lengths.
Source: one-for-one raw serialization, without compression of neutral positions. -/
theorem depthFinitePrefix_length (body : List (Option Bool)) :
    (none :: body).length = ((1 : Fin 36) :: depthFiniteBody body).length := by
  simp only [depthFiniteBody, List.length_cons, List.length_map]

/-- Decoding the true finite BOS/body input gives exactly the original integer prefix.
Source: unchanged actual BOS/A/B/neutral IDs, not a modulo or semantic-label encoding. -/
theorem depthFinitePrefix_decode (body : List (Option Bool)) :
    decodeTokens ((1 : Fin 36) :: depthFiniteBody body) = bos :: depthBody body := by
  simp only [decodeTokens, depthFiniteBody, List.map_cons, List.map_map]
  change bos :: body.map _ = bos :: depthBody body
  congr 1
  unfold depthBody
  apply List.map_congr_left
  intro letter _
  cases letter with
  | none => rfl
  | some b => cases b <;> rfl

/-- Every true BOS/body embedding lookup equals the corresponding actual raw-word embedding at that same original position.
Source: genuine token-local serialization, exact finite indexing and the proved distinct-ID BOS/neutral embedding equality. -/
theorem depthFinitePrefix_embedding (mode : Mode) (gain : ℝ) (body : List (Option Bool))
    (i : Fin (none :: body).length) :
    depthRawEmbedding mode gain (((1 : Fin 36) :: depthFiniteBody body).get (Fin.cast (depthFinitePrefix_length body) i)) =
      depthWordInput mode gain (none :: body) i := by
  rcases i with ⟨n, hn⟩
  cases n with
  | zero =>
      change depthRawEmbedding mode gain (1 : Fin 36) = depthRawEmbedding mode gain (depthLetterToken none)
      have ht : depthLetterToken none = (11 : Fin 36) := by apply Fin.ext; rfl
      rw [ht]
      exact depthRawEmbedding_bos_neutral mode gain
  | succ n =>
      simp only [depthWordInput, depthFiniteBody, List.get_eq_getElem, Fin.cast,
        List.getElem_cons_succ, List.getElem_map]

/-- The original forward margin is unchanged by exact raw-word/finite-array length transport.
Source: the actual complete-model theorem on token-local embeddings; equality transport changes only dependent index types. -/
theorem depthModel_logits_cast (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hbound : word.length ≤ 128) {T : ℕ} (hlen : word.length = T)
    (positions : Fin T → ℝ) (tokens : Fin T → Fin 36)
    (hembed : ∀ i, depthRawEmbedding mode depthLabelGain (tokens (Fin.cast hlen i)) = depthWordInput mode depthLabelGain word i)
    (i : Fin T) (token : Fin 36) (hne : token ≠ depthWordAnswer mode word (Fin.cast hlen.symm i)) :
    forward (depthConfig mode) (depthModelParams mode) eps positions tokens i token <
      forward (depthConfig mode) (depthModelParams mode) eps positions tokens i
        (depthWordAnswer mode word (Fin.cast hlen.symm i)) := by
  cases hlen
  simp only [Fin.cast_refl, id_eq] at hembed hne ⊢
  exact depthModel_logits mode eps heps hclip word hbound positions tokens hembed i token hne

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 ∧
    (∀ i : Fin ([none, some false, some true] : List (Option Bool)).length,
      depthRawEmbedding .easy depthLabelGain
        (((1 : Fin 36) :: depthFiniteBody [some false, some true]).get (Fin.cast (depthFinitePrefix_length _) i)) =
          depthWordInput .easy depthLabelGain [none, some false, some true] i) ∧
    (0 : Fin 36) ≠ depthWordAnswer .easy [none, some false, some true] 2 := by
  refine ⟨by norm_num, by norm_num, by decide, depthFinitePrefix_embedding .easy depthLabelGain _, ?_⟩
  unfold depthWordAnswer
  split_ifs <;> decide

/-- The actual last-position semantic target equals the independent raw integer Basis depth answer in both modes.
Source: full-word occurrence/subsequence equivalence, retained leading neutral, exact E_2/E_4 patterns and taskNext's actual mode parameters. -/
theorem depthWordAnswer_taskNext (mode : Mode) (body : List (Option Bool)) (i : Fin (none :: body).length)
    (hlast : i.val + 1 = (none :: body).length) :
    ((depthWordAnswer mode (none :: body) i).val : ℤ) = taskNext mode .depth (bos :: depthBody body) := by
  rw [depthWordAnswer_last mode (none :: body) i hlast]
  cases mode with
  | easy =>
      change ((if (∃ j : Fin (none :: body).length, DepthOccurrence 1 true (none :: body) j) ∧
        ¬(∃ j : Fin (none :: body).length, DepthOccurrence 1 false (none :: body) j) then (16 : Fin 36) else 15).val : ℤ) =
          depthNext 2 (bos :: depthBody body)
      rw [depthNext_body]
      simp only [depthOccurrence_exists, depthEndPattern_easy.1, depthEndPattern_easy.2, List.reduceOption_cons_of_none]
      split_ifs <;> rfl
  | hard =>
      change ((if (∃ j : Fin (none :: body).length, DepthOccurrence 3 true (none :: body) j) ∧
        ¬(∃ j : Fin (none :: body).length, DepthOccurrence 3 false (none :: body) j) then (16 : Fin 36) else 15).val : ℤ) =
          depthNext 4 (bos :: depthBody body)
      rw [depthNext_body]
      simp only [depthOccurrence_exists, depthEndPattern_hard.1, depthEndPattern_hard.2, List.reduceOption_cons_of_none]
      split_ifs <;> rfl

example : (2 : Fin ([none, some false, some true] : List (Option Bool)).length).val + 1 =
    ([none, some false, some true] : List (Option Bool)).length := by decide

/-- The true model has strict raw BOS/body logits with every embedding requirement discharged by actual serialization.
Source: complete original forward, exact finite-length transport and derived token-local BOS/body embedding equality. -/
theorem depthModel_raw_logits (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (body : List (Option Bool)) (hbound : body.length + 1 ≤ 128)
    (i : Fin ((1 : Fin 36) :: depthFiniteBody body).length) (token : Fin 36)
    (hne : token ≠ depthWordAnswer mode (none :: body) (Fin.cast (depthFinitePrefix_length body).symm i)) :
    forward (depthConfig mode) (depthModelParams mode) eps (fun j => (j.val : ℝ))
      ((1 : Fin 36) :: depthFiniteBody body).get i token <
    forward (depthConfig mode) (depthModelParams mode) eps (fun j => (j.val : ℝ))
      ((1 : Fin 36) :: depthFiniteBody body).get i
        (depthWordAnswer mode (none :: body) (Fin.cast (depthFinitePrefix_length body).symm i)) := by
  exact depthModel_logits_cast mode eps heps hclip (none :: body) (by simpa only [List.length_cons] using hbound)
    (depthFinitePrefix_length body) (fun j => (j.val : ℝ)) ((1 : Fin 36) :: depthFiniteBody body).get
    (depthFinitePrefix_embedding mode depthLabelGain body) i token hne

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, some true] : List (Option Bool)).length + 1 ≤ 128 ∧
    (0 : Fin 36) ≠ depthWordAnswer .easy [none, some false, some true] 2 := by
  refine ⟨by norm_num, by norm_num, by decide, ?_⟩
  unfold depthWordAnswer
  split_ifs <;> decide

/-- Actual original final-row greedy logits on the finite BOS/body prefix return the independent semantic answer without an input-encoding premise.
Source: genuine lastLogits on the same array, raw all-vocabulary margins and the actual deterministic greedy decoder. -/
theorem depthModel_raw_greedy (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (body : List (Option Bool)) (hbound : body.length + 1 ≤ 128) :
    let last : Fin ((1 : Fin 36) :: depthFiniteBody body).length := ⟨(depthFiniteBody body).length, by simp⟩
    TokenInterface.bestToken (depthConfig mode).vocab_pos
      (TokenInterface.lastLogits (depthConfig mode) (depthModelParams mode) eps 1 (depthFiniteBody body)) =
        depthWordAnswer mode (none :: body) (Fin.cast (depthFinitePrefix_length body).symm last) := by
  apply TokenInterface.bestToken_of_strict
  exact depthModel_raw_logits mode eps heps hclip body hbound _

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length + 1 ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

end Transformer.GPTMini.Semantics
