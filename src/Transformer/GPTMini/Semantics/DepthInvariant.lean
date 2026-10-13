import Transformer.GPTMini.Semantics.DepthLevels

/-!
# Explicit quantified invariant for actual depth residual arrays

Source: fixed original depth block layout at f11b6e2 and the independent
neutral-preserving Basis recurrence at cbafbe9. The invariant records
protected raw constant/types, quantitative representation of one current
occurrence level, fresh future-stage channels and actual norm growth.
It constrains a supplied real array; it never computes a hidden state
from an oracle or a desired label. Actual states use blockForward.

The real token-local raw embedding derives the whole invariant at level
zero for every word. Each applicable invariant instance implies all
incoming conditions required by the genuine next detector, including
its matching-type zero opposite self. Within this explicit predicate,
norms are at most 866, within the shared real RMS applicability domain.

The actual block-preservation and complete hidden-state induction are
subsequent obligations; later invariant instances are not assumed as
inputs to the final model correctness theorem.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- Local quantitative conditions on a supplied real residual array at one of four possible levels.
Source: actual raw-type protection, DepthLevelRepresentation, future-channel freshness and M+288 block growth.
This is a predicate of its state argument; the ordinary model state remains independently computed. -/
def DepthStateInvariant (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) : Prop :=
  (∀ i, x i (depthCoordinate mode 0) = 1) ∧
  (∀ b i, x i (depthTypeCoordinate mode b) = if word.get i = some (depthBranchLetter b) then 1 else 0) ∧
  DepthLevelRepresentation mode level word x ∧
  (∀ stage : Fin 3, level.val ≤ stage.val → ∀ part : Fin 4, ∀ i,
    x i (depthCoordinate mode (depthStageAxis stage part)) = 0) ∧
  (∀ i, ‖x i‖ ≤ 2 + 288 * (level.val : ℝ))

/-- The actual token-local raw-word input satisfies the full numerical/data invariant at level zero.
Source: real embedding coordinates, initially empty stage channels, independent zero-level recurrence and input norm two.
No correct-output or prepared representation hypothesis is needed. -/
theorem depthStateInvariant_input (mode : Mode) (gain : ℝ) (word : List (Option Bool)) :
    DepthStateInvariant mode 0 word (depthWordInput mode gain word) := by
  refine ⟨depthWordInput_constant mode gain word, ?_, depthLevelRepresentation_input mode gain word, ?_, ?_⟩
  · intro b i
    exact depthWordInput_type mode gain word i b
  · exact fun stage _ part i => depthWordInput_fresh mode gain word i stage part
  · intro i
    simpa using (depthWordInput_norm mode gain word i).2

/-- The same preserved real raw type is one exactly at the current corresponding raw letter.
Source: the invariant's real type read and the ordinary raw-letter equality, used by the actual ordered transition. -/
theorem depthStateInvariant_type_iff (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) (h : DepthStateInvariant mode level word x)
    (b : Fin 2) (i : Fin word.length) :
    x i (depthTypeCoordinate mode b) = 1 ↔ word.get i = some (depthBranchLetter b) := by
  rw [h.2.1 b i]
  by_cases hm : word.get i = some (depthBranchLetter b)
  · rw [ite_eq_left hm]
    exact ⟨fun _ => hm, fun _ => rfl⟩
  · rw [ite_eq_right hm]
    constructor
    · intro hz
      norm_num at hz
    · intro hz
      exact False.elim (hm hz)

example : DepthStateInvariant .easy 0 [some false, none, some true]
    (depthWordInput .easy 1 [some false, none, some true]) := depthStateInvariant_input .easy 1 _

/-- Every applicable quantified invariant derives all incoming conditions of the real next detector.
Source: actual coordinate coupling to the previous opposite level, genuine future freshness and independent opposite-current exclusion.
No FFN gap, normalized feature or zero-self condition is separately assumed. -/
theorem depthStateInvariant_detectorInput (mode : Mode) (stage : Fin 3) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨stage.val, by have hs := stage.isLt; omega⟩ word x) :
    DepthDetectorInput mode stage x := by
  refine ⟨h.1, ?_, ?_, ?_, ?_, ?_⟩
  · intro b i
    rw [h.2.1 b i]
    split_ifs
    · exact Or.inr rfl
    · exact Or.inl rfl
  · intro b i
    exact h.2.2.2.1 stage (by change stage.val ≤ stage.val; omega) ⟨b.val, by have hb := b.isLt; omega⟩ i
  · intro b i
    exact h.2.2.2.1 stage (by change stage.val ≤ stage.val; omega) ⟨2 + b.val, by have hb := b.isLt; omega⟩ i
  · intro b i
    rw [depthSourceCoordinate_level]
    exact (depthLevelRepresentation_bounds mode _ word x h.2.2.1 _ i).1
  · intro b i hk
    have hm := (depthStateInvariant_type_iff mode _ word x h b i).mp hk
    rw [depthSourceCoordinate_level]
    exact depthLevelRepresentation_opposite_self mode _ word x h.2.2.1 b i hm

example : DepthStateInvariant .hard 0 [some false, none, some true]
    (depthWordInput .hard 1 [some false, none, some true]) := depthStateInvariant_input .hard 1 _

/-- The real source of each actual head retains the same quantitative zero-or-positive gap and cap.
Source: genuine fused source/previous-level coupling and the supplied independent occurrence representation. -/
theorem depthStateInvariant_source_bounds (mode : Mode) (stage : Fin 3) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨stage.val, by have hs := stage.isLt; omega⟩ word x) (b : Fin 2) (i : Fin word.length) :
    (x i (depthSourceCoordinate mode stage b) = 0 ∨ depthScaleLower ^ 2 ≤ x i (depthSourceCoordinate mode stage b)) ∧
      x i (depthSourceCoordinate mode stage b) ≤ 128 := by
  rw [depthSourceCoordinate_level]
  exact depthLevelRepresentation_bounds mode _ word x h.2.2.1 _ i

example : DepthStateInvariant .easy 0 [some false, none, some true]
    (depthWordInput .easy 1 [some false, none, some true]) := depthStateInvariant_input .easy 1 _

/-- Each genuine source feature reaches its threshold exactly at an independent opposite-ending occurrence.
Source: actual fused source row, quantitative level representation and the same raw branch/letter switch. -/
theorem depthStateInvariant_source_iff (mode : Mode) (stage : Fin 3) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model)
    (h : DepthStateInvariant mode ⟨stage.val, by have hs := stage.isLt; omega⟩ word x) (b : Fin 2) (i : Fin word.length) :
    depthScaleLower ^ 2 ≤ x i (depthSourceCoordinate mode stage b) ↔
      DepthOccurrence stage.val (!depthBranchLetter b) word i := by
  rw [depthSourceCoordinate_level, depthLevelRepresentation_threshold_iff mode _ word x h.2.2.1,
    depthBranchLetter_opposite]

example : DepthStateInvariant .hard 0 [some false, none, some true]
    (depthWordInput .hard 1 [some false, none, some true]) := depthStateInvariant_input .hard 1 _

/-- The explicit growth clause bounds every supplied invariant array by 866.
Source: the numerical bound 2+288*level and level at most three; actual block induction must derive that clause. -/
theorem depthStateInvariant_norm (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) (h : DepthStateInvariant mode level word x)
    (i : Fin word.length) : ‖x i‖ ≤ 866 := by
  have hn : level.val ≤ 3 := by have hl := level.isLt; omega
  have hr : (level.val : ℝ) ≤ 3 := by exact_mod_cast hn
  have hb := h.2.2.2.2 i
  linarith

example : DepthStateInvariant .easy 0 [some false, none, some true]
    (depthWordInput .easy 1 [some false, none, some true]) := depthStateInvariant_input .easy 1 _

/-- Every true invariant array lies inside the quantitative domain for the actual next RMS/flag bounds.
Source: its derived norm cap 866, below the required incoming bound 4064. -/
theorem depthStateInvariant_domain (mode : Mode) (level : Fin 4) (word : List (Option Bool))
    (x : Fin word.length → EucSpace (depthConfig mode).d_model) (h : DepthStateInvariant mode level word x) :
    ∀ i, ‖x i‖ ≤ 4064 := by
  intro i
  exact (depthStateInvariant_norm mode level word x h i).trans (by norm_num)

example : DepthStateInvariant .hard 0 [some false, none, some true]
    (depthWordInput .hard 1 [some false, none, some true]) := depthStateInvariant_input .hard 1 _

end Transformer.GPTMini.Semantics
