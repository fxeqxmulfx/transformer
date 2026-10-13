import Transformer.GPTMini.Semantics.DepthWordInput

/-!
# Quantitative original depth coordinates and independent occurrences

Source: the fixed original residual layout at f11b6e2 and the neutral-
preserving DepthOccurrence recurrence for Basis E_2/E_4 at cbafbe9.
Level zero is the actual raw A/B type. Levels one/two/three are the
genuine FFN targets after the first/second/third detector blocks.
Every actual head reads the previous opposite-ending level.

The representation predicate constrains a supplied real state: a true
independent ordered occurrence has amplitude in [r^2,128]; its absence
has coordinate zero. It does not define a model state from the word or
assume correct logits. The initial actual token-local embedding derives
this predicate at level zero for every raw word. The forthcoming actual
block induction must derive its later instances, including all bounds.

Within this explicit representation, threshold/positivity detect the
independent occurrence, and a matching current letter excludes opposite
self. These statements keep original positions and neutral letters.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- One actual residual coordinate for each of four alternating-occurrence levels.
Source: raw axes one/two followed by genuine stage flag axes five/six, nine/ten and thirteen/fourteen. -/
def depthLevelCoordinate (mode : Mode) (level : Fin 4) (b : Fin 2) : Fin (depthConfig mode).d_model :=
  depthCoordinate mode ⟨1 + 4 * level.val + b.val, by have hl := level.isLt; have hb := b.isLt; omega⟩

/-- The initial level is precisely the true raw embedding type coordinate.
Source: the same original integer-axis layout, without a separate semantic encoder. -/
theorem depthLevelCoordinate_initial (mode : Mode) (b : Fin 2) :
    depthLevelCoordinate mode 0 b = depthTypeCoordinate mode b := by
  rfl

/-- Each next occurrence level is exactly this actual stage's FFN output column.
Source: the original four-channel stage layout and fixed genuine detector target. -/
theorem depthLevelCoordinate_successor (mode : Mode) (stage : Fin 3) (b : Fin 2) :
    depthLevelCoordinate mode ⟨stage.val + 1, by have hs := stage.isLt; omega⟩ b =
      depthFeatureCoordinate mode stage b := by
  apply congrArg (depthCoordinate mode)
  apply Fin.ext
  change 1 + 4 * (stage.val + 1) + b.val = 3 + 4 * stage.val + (2 + b.val)
  omega

/-- Every genuine detector head reads its current opposite-ending level coordinate.
Source: fixed fused-QKV source rows, including the first raw-type read and both later FFN reads. -/
theorem depthSourceCoordinate_level (mode : Mode) (stage : Fin 3) (b : Fin 2) :
    depthSourceCoordinate mode stage b =
      depthLevelCoordinate mode ⟨stage.val, by have hs := stage.isLt; omega⟩
        ⟨1 - b.val, by have hb := b.isLt; omega⟩ := by
  apply congrArg (depthCoordinate mode)
  apply Fin.ext
  change 2 + 4 * stage.val - b.val = 1 + 4 * stage.val + (1 - b.val)
  have hb := b.isLt
  omega

/-- Quantitative representation of independent ordered occurrences by specified actual real coordinates.
Source: DepthOccurrence and the proved genuine zero-or-[r^2,128] detector transition.
This is only a predicate of its supplied state; actual states remain computed by ordinary blockForward. -/
def DepthLevelRepresentation (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) : Prop :=
  ∀ b i, (DepthOccurrence level.val (depthBranchLetter b) word i →
      depthScaleLower ^ 2 ≤ x i (depthLevelCoordinate mode level b) ∧ x i (depthLevelCoordinate mode level b) ≤ 128) ∧
    (¬DepthOccurrence level.val (depthBranchLetter b) word i → x i (depthLevelCoordinate mode level b) = 0)

/-- Every actual raw-word embedding represents the independent one-letter base recurrence quantitatively.
Source: true token-local type coordinates, DepthOccurrence's zero-level criterion and the fixed r≤1.
No prepared semantic representation is a hypothesis. -/
theorem depthLevelRepresentation_input (mode : Mode) (gain : ℝ) (word : List (Option Bool)) :
    DepthLevelRepresentation mode 0 word (depthWordInput mode gain word) := by
  intro b i
  constructor
  · intro hp
    have hm := (depthOccurrence_zero (depthBranchLetter b) word i).mp hp
    rw [depthLevelCoordinate_initial, depthWordInput_type, ite_eq_left hm]
    norm_num [depthScaleLower]
  · intro hp
    have hm : word.get i ≠ some (depthBranchLetter b) :=
      fun hm => hp ((depthOccurrence_zero (depthBranchLetter b) word i).mpr hm)
    rw [depthLevelCoordinate_initial, depthWordInput_type, ite_eq_right hm]

/-- Independent occurrence representation gives the exact real zero-or-positive gap and cap at every position.
Source: the two explicit numerical representation clauses; neither Boolean values nor unit amplitudes are assumed. -/
theorem depthLevelRepresentation_bounds (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) (h : DepthLevelRepresentation mode level word x)
    (b : Fin 2) (i : Fin word.length) :
    (x i (depthLevelCoordinate mode level b) = 0 ∨ depthScaleLower ^ 2 ≤ x i (depthLevelCoordinate mode level b)) ∧
      x i (depthLevelCoordinate mode level b) ≤ 128 := by
  by_cases hp : DepthOccurrence level.val (depthBranchLetter b) word i
  · exact ⟨Or.inr ((h b i).1 hp).1, ((h b i).1 hp).2⟩
  · rw [(h b i).2 hp]
    exact ⟨Or.inl rfl, by norm_num⟩

example : DepthLevelRepresentation .easy 0 [some false, none, some true]
    (depthWordInput .easy 1 [some false, none, some true]) := depthLevelRepresentation_input .easy 1 _

/-- A quantitative real feature reaches the shared floor exactly when its independent ordered occurrence is true.
Source: the proved positive r^2 floor and exact zero under absence in the supplied representation. -/
theorem depthLevelRepresentation_threshold_iff (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) (h : DepthLevelRepresentation mode level word x)
    (b : Fin 2) (i : Fin word.length) :
    depthScaleLower ^ 2 ≤ x i (depthLevelCoordinate mode level b) ↔ DepthOccurrence level.val (depthBranchLetter b) word i := by
  constructor
  · intro hf
    by_contra hp
    rw [(h b i).2 hp] at hf
    have hr := sq_pos_of_pos depthScaleLower_bounds.1
    linarith
  · intro hp
    exact ((h b i).1 hp).1

example : DepthLevelRepresentation .hard 0 [some false, none, some true]
    (depthWordInput .hard 1 [some false, none, some true]) := depthLevelRepresentation_input .hard 1 _

/-- Strict positivity of the actual real feature detects the same independent ordered occurrence.
Source: numerical positive floor and exact absence, independent of the actual positive RMS amplitude. -/
theorem depthLevelRepresentation_positive_iff (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) (h : DepthLevelRepresentation mode level word x)
    (b : Fin 2) (i : Fin word.length) :
    0 < x i (depthLevelCoordinate mode level b) ↔ DepthOccurrence level.val (depthBranchLetter b) word i := by
  have hr := sq_pos_of_pos depthScaleLower_bounds.1
  constructor
  · intro hf
    by_contra hp
    rw [(h b i).2 hp] at hf
    exact False.elim ((lt_irrefl 0) hf)
  · intro hp
    have hf := ((h b i).1 hp).1
    linarith

example : DepthLevelRepresentation .easy 0 [some false, none, some true]
    (depthWordInput .easy 1 [some false, none, some true]) := depthLevelRepresentation_input .easy 1 _

/-- A genuine current letter forces the represented opposite-ending self-coordinate to zero at every level.
Source: independent ordered occurrence's raw-type exclusion and the true quantitative absence clause. -/
theorem depthLevelRepresentation_opposite_self (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) (h : DepthLevelRepresentation mode level word x)
    (b : Fin 2) (i : Fin word.length) (hcurrent : word.get i = some (depthBranchLetter b)) :
    x i (depthLevelCoordinate mode level ⟨1 - b.val, by have hb := b.isLt; omega⟩) = 0 := by
  apply (h _ i).2
  rw [depthBranchLetter_opposite]
  exact depthOccurrence_opposite level.val (depthBranchLetter b) word i hcurrent

example : DepthLevelRepresentation .easy 0 [some false, none, some true]
    (depthWordInput .easy 1 [some false, none, some true]) ∧
    ([some false, none, some true] : List (Option Bool)).get (0 : Fin 3) = some (depthBranchLetter 0) := by
  exact ⟨depthLevelRepresentation_input .easy 1 _, by decide⟩

end Transformer.GPTMini.Semantics
