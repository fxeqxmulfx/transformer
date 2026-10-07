import Transformer.GPTMini.Semantics.DepthFeatureBounds

/-!
# Actual token-local depth inputs, with every neutral position retained

Source: Basis.depthBody at cbafbe9, vocabulary A=9/B=10/neutral=11
and the genuine tied embedding table of DepthEmbedding. The internal
word adapter looks only at the current raw letter. None maps to the
neutral token; its actual embedding equals BOS's constant embedding,
so a leading none can later be coupled to the true BOS serialization.

All initial detector conditions are derived from the actual local
embedding: constant one, binary mutually exclusive raw A/B types,
fresh later channels, zero opposite self and norm between one/two.
No prefix pattern, desired label or whole-word oracle defines a row.

This supplies the base of the real hidden-state induction. Subsequent
blocks must derive alternating occurrences, and the final original
model/checked integer interface must still be coupled to raw Basis.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- The ordinary raw token ID of one mathematical letter, including neutral.
Source: depthBody's exact local vocabulary map; every position is retained. -/
def depthLetterToken : Option Bool → Fin 36
  | none => ⟨11, by decide⟩
  | some false => ⟨9, by decide⟩
  | some true => ⟨10, by decide⟩

/-- Both real head/FFN branches retain the same A=false, B=true convention.
Source: fixed depthTypeCoordinate's branch-zero/branch-one assignment. -/
def depthBranchLetter (b : Fin 2) : Bool := decide (b.val = 1)

/-- Switching the actual branch switches its raw letter.
Source: the two-column integer assignment, used in the opposite-ending recurrence. -/
theorem depthBranchLetter_opposite (b : Fin 2) :
    depthBranchLetter ⟨1 - b.val, by have hb := b.isLt; omega⟩ = !depthBranchLetter b := by
  fin_cases b <;> decide

/-- Every local letter token is in the actual original depth input alphabet.
Source: the checked raw vocabulary IDs, without a premise about ordered patterns. -/
theorem depthLetterToken_raw (letter : Option Bool) : DepthRawToken (depthLetterToken letter) := by
  cases letter with
  | none => exact Or.inr (Or.inr (Or.inr rfl))
  | some b =>
      cases b
      · exact Or.inr (Or.inl rfl)
      · exact Or.inr (Or.inr (Or.inl rfl))

/-- The real embedding at each original word position depends only on its own ordinary token.
Source: depthBody's local encoding followed by the unchanged tied embedding table. -/
noncomputable def depthWordInput (mode : Mode) (gain : ℝ) (word : List (Option Bool))
    (i : Fin word.length) : EucSpace (depthConfig mode).d_model :=
  depthRawEmbedding mode gain (depthLetterToken (word.get i))

/-- The genuine local input keeps the protected constant one at every position.
Source: the actual raw-token embedding theorem, instantiated by the local serialization. -/
theorem depthWordInput_constant (mode : Mode) (gain : ℝ) (word : List (Option Bool)) (i : Fin word.length) :
    depthWordInput mode gain word i (depthCoordinate mode 0) = 1 := by
  exact depthRawEmbedding_constant mode gain _ (depthLetterToken_raw (word.get i))

/-- The actual raw embedding type coordinate agrees exactly with the current original-position letter.
Source: the full real token-local table and two disjoint raw type axes; no occurrence predicate defines an embedding. -/
theorem depthWordInput_type (mode : Mode) (gain : ℝ) (word : List (Option Bool)) (i : Fin word.length) (b : Fin 2) :
    depthWordInput mode gain word i (depthTypeCoordinate mode b) =
      if word.get i = some (depthBranchLetter b) then 1 else 0 := by
  unfold depthWordInput depthTypeCoordinate
  cases hl : word.get i with
  | none => fin_cases b <;> norm_num [depthLetterToken, depthRawEmbedding,
      depthBranchLetter, depthAxis_coordinate]
  | some letter => cases letter <;> fin_cases b <;> norm_num [depthLetterToken,
      depthRawEmbedding, depthBranchLetter, depthAxis_coordinate]

/-- Each actual token-local raw type is zero or one, simultaneously in both branches.
Source: the evaluated real embedding coordinates and raw-letter equality. -/
theorem depthWordInput_type_binary (mode : Mode) (gain : ℝ) (word : List (Option Bool)) (i : Fin word.length) (b : Fin 2) :
    depthWordInput mode gain word i (depthTypeCoordinate mode b) = 0 ∨
      depthWordInput mode gain word i (depthTypeCoordinate mode b) = 1 := by
  rw [depthWordInput_type]
  split_ifs
  · exact Or.inr rfl
  · exact Or.inl rfl

/-- Every real signal/feature stage channel starts at exactly zero on all raw words.
Source: actual local embedding freshness and all stage axes being at least three. -/
theorem depthWordInput_fresh (mode : Mode) (gain : ℝ) (word : List (Option Bool)) (i : Fin word.length)
    (stage : Fin 3) (part : Fin 4) :
    depthWordInput mode gain word i (depthCoordinate mode (depthStageAxis stage part)) = 0 := by
  apply depthRawEmbedding_fresh mode gain _ (depthLetterToken_raw (word.get i))
  all_goals
    intro he
    have hv := congrArg (fun c : Fin 24 => c.val) he
    dsimp only [depthStageAxis] at hv
    omega

/-- The same raw embedding gives genuine input norm between one and two, independent of the tied label gain.
Source: actual raw vocabulary/embedding bounds; labels are not inserted into the input word. -/
theorem depthWordInput_norm (mode : Mode) (gain : ℝ) (word : List (Option Bool)) (i : Fin word.length) :
    1 ≤ ‖depthWordInput mode gain word i‖ ∧ ‖depthWordInput mode gain word i‖ ≤ 2 := by
  exact depthRawEmbedding_norm mode gain _ (depthLetterToken_raw (word.get i))

/-- A matching actual raw type excludes its opposite true local embedding feature at that same position.
Source: mutually exclusive current raw A/B letter IDs, deriving the first original XSA zero-self condition. -/
theorem depthWordInput_opposite (mode : Mode) (gain : ℝ) (word : List (Option Bool)) (i : Fin word.length) (b : Fin 2)
    (hk : depthWordInput mode gain word i (depthTypeCoordinate mode b) = 1) :
    depthWordInput mode gain word i
      (depthTypeCoordinate mode ⟨1 - b.val, by have hb := b.isLt; omega⟩) = 0 := by
  rw [depthWordInput_type] at hk
  have hmatch : word.get i = some (depthBranchLetter b) := by
    by_contra hne
    rw [ite_eq_right hne] at hk
    norm_num at hk
  rw [depthWordInput_type, hmatch, depthBranchLetter_opposite]
  cases depthBranchLetter b <;> norm_num

example : depthWordInput .easy 1 [some false] 0 (depthTypeCoordinate .easy 0) = 1 := by
  rw [depthWordInput_type]
  norm_num [depthBranchLetter]

/-- All incoming conditions for the first complete original detector are derived from the actual raw-word embedding array.
Source: real local coordinates/freshness, quantitative zero-or-unit source features and mutual raw-type exclusion.
No prepared hidden representation is a premise. -/
theorem depthWordInput_detectorInput (mode : Mode) (gain : ℝ) (word : List (Option Bool)) :
    DepthDetectorInput mode 0 (depthWordInput mode gain word) := by
  refine ⟨depthWordInput_constant mode gain word, ?_, ?_, ?_, ?_, ?_⟩
  · intro b j
    exact depthWordInput_type_binary mode gain word j b
  · intro b j
    exact depthWordInput_fresh mode gain word j 0 ⟨b.val, by have hb := b.isLt; omega⟩
  · intro b j
    exact depthWordInput_fresh mode gain word j 0 ⟨2 + b.val, by have hb := b.isLt; omega⟩
  · intro b j
    rw [depthSourceCoordinate_initial]
    rcases depthWordInput_type_binary mode gain word j ⟨1 - b.val, by have hb := b.isLt; omega⟩ with hz | hone
    · exact Or.inl hz
    · apply Or.inr
      rw [hone]
      norm_num [depthScaleLower]
  · intro b j hk
    rw [depthSourceCoordinate_initial]
    exact depthWordInput_opposite mode gain word j b hk

/-- The actual BOS embedding equals the neutral embedding used by a leading none in the internal word adapter.
Source: both original token-local entries carry only the same protected constant; their raw integer IDs remain distinct. -/
theorem depthRawEmbedding_bos_neutral (mode : Mode) (gain : ℝ) :
    depthRawEmbedding mode gain (1 : Fin 36) = depthRawEmbedding mode gain (11 : Fin 36) := by
  unfold depthRawEmbedding
  norm_num

end Transformer.GPTMini.Semantics
