import Transformer.GPTMini.Sparsemax.AtomicMatchingHead
import Mathlib.Topology.Maps.Proper.Basic

/-!
# Continuity of genuine matching across sparsemax support changes

New attainment prerequisite for the physical atomic architecture following
arXiv:2211.11052v1, Appendix A.4. Sparsemax is the original variational
projection of arXiv:1602.02068v2, Eq. (1), on the repository's causal
simplex. Its support may change: there is no gap or fixed-support premise.

The closed graph follows from actual global inference optimality, and
compactness of the simplex turns this into continuity. Independent shared
Q/K/value tables then give continuous genuine observed head responses.
Continuity does not assert convexity of these raw physical parameters.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The actual sparsemax inference graph is closed, including support boundaries.
Source: sparsemax Eq. (1), with a causal rather than full simplex. -/
theorem sparseWeights_graph_closed {T : ℕ} (row : Fin T) :
    IsClosed (Function.graph (fun scores : Fin T → ℝ =>
      (⟨sparseWeights scores row, (sparseWeights_spec scores row).1⟩ :
        simplexOn {j : Fin T | j ≤ row}))) := by
  let S := simplexOn {j : Fin T | j ≤ row}
  have he : Function.graph (fun scores : Fin T → ℝ =>
      (⟨sparseWeights scores row, (sparseWeights_spec scores row).1⟩ : S)) =
      ⋂ b : S, {p : (Fin T → ℝ) × S |
        routingObjective (fun j => -p.1 j / 2) p.2.val ≤
          routingObjective (fun j => -p.1 j / 2) b.val} := by
    ext p
    simp only [Function.graph, Set.mem_ofPred_eq, Set.mem_iInter]
    constructor
    · intro hp b
      have hev : sparseWeights p.1 row = p.2.val := congrArg Subtype.val hp
      rw [← hev]
      exact (sparseWeights_spec p.1 row).2 b.val b.property
    · intro hp
      apply Subtype.ext
      exact routing_minimizers_eq {j : Fin T | j ≤ row} (fun j => -p.1 j / 2)
        (sparseWeights p.1 row) p.2.val (sparseWeights_spec p.1 row).1 p.2.property
        (sparseWeights_spec p.1 row).2 (fun b hb => hp ⟨b, hb⟩)
  rw [he]
  apply isClosed_iInter
  intro b
  have hv : Continuous (fun p : (Fin T → ℝ) × S => p.2.val) :=
    continuous_subtype_val.comp continuous_snd
  have hl : Continuous (fun p : (Fin T → ℝ) × S =>
      routingObjective (fun j => -p.1 j / 2) p.2.val) := by
    unfold routingObjective
    apply Continuous.add
    · apply Continuous.div_const
      apply continuous_finsetSum
      intro j hj
      exact ((continuous_apply j).comp hv).pow 2
    · apply continuous_finsetSum
      intro j hj
      exact ((continuous_apply j).comp hv).mul
        (((continuous_apply j).comp continuous_fst).neg.div_const 2)
  have hr : Continuous (fun p : (Fin T → ℝ) × S =>
      routingObjective (fun j => -p.1 j / 2) b.val) := by
    unfold routingObjective
    fun_prop
  exact isClosed_le hl hr

/-- Original variational causal sparsemax is continuous at every score assignment.
Source: sparsemax Eq. (1); compact feasible weights and the closed optimality graph. -/
theorem sparseWeights_continuous {T : ℕ} (row : Fin T) :
    Continuous (fun scores : Fin T → ℝ => sparseWeights scores row) := by
  let S := simplexOn {j : Fin T | j ≤ row}
  let : CompactSpace S := isCompact_iff_compactSpace.mp (simplexOn_compact _)
  have hc := continuous_of_isClosed_graph (sparseWeights_graph_closed row)
  exact continuous_subtype_val.comp hc

/-- Every Q/K score coordinate is continuous in the two independent learned embeddings.
Source: attention-only dot products in arXiv:2211.11052v1, §3.1. -/
theorem matchingHeadScores_continuous {V H D T : ℕ}
    (tokens : Fin T → Fin V) (row j : Fin T) :
    Continuous (fun h : MatchingHead V H D => matchingHeadScores h tokens row j) := by
  unfold matchingHeadScores
  fun_prop

/-- Physical output continuity includes changes of the true sparsemax attention support.
Source: sparsemax Eq. (1) times independent original common values, following Appendix A.4. -/
theorem matchingHeadOutput_continuous {V H D T : ℕ}
    (tokens : Fin T → Fin V) (row : Fin T) (channel : Fin D) :
    Continuous (fun h : MatchingHead V H D => matchingHeadOutput h tokens row channel) := by
  have hs : Continuous (fun h : MatchingHead V H D => matchingHeadScores h tokens row) := by
    apply continuous_pi
    intro j
    exact matchingHeadScores_continuous tokens row j
  have hw := (sparseWeights_continuous row).comp hs
  unfold matchingHeadOutput
  apply continuous_finsetSum
  intro j hj
  exact ((continuous_apply j).comp hw).mul (by fun_prop)

/-- Every finite actual-head observation table is continuous in all physical learned coordinates.
Source: the genuine Q/K/value atomic family following Appendix A.4. -/
theorem matchingHeadSample_continuous {V H D R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    Continuous (matchingHeadSample (H := H) tokens rows channels) := by
  apply continuous_pi
  intro r
  exact matchingHeadOutput_continuous (tokens r) (rows r) (channels r)

/-- All actual attention coordinates vary continuously without prescribing their support.
Source: sparsemax Eq. (1) applied to the learned content score rows of §3.1. -/
theorem matchingHeadWeights_continuous {V H D T : ℕ}
    (tokens : Fin T → Fin V) (row : Fin T) :
    Continuous (fun h : MatchingHead V H D => sparseWeights (matchingHeadScores h tokens row) row) := by
  apply (sparseWeights_continuous row).comp
  apply continuous_pi
  intro j
  exact matchingHeadScores_continuous tokens row j

/-- The entire causal block response, with every output channel, is continuous in one shared head.
Source: the genuine attention/value product before the Appendix A.4 mixture. -/
theorem matchingHeadBlock_continuous {V H D T : ℕ} (tokens : Fin T → Fin V) :
    Continuous (fun h : MatchingHead V H D => fun row : Fin T => fun channel : Fin D =>
      matchingHeadOutput h tokens row channel) := by
  apply continuous_pi
  intro row
  apply continuous_pi
  intro channel
  exact matchingHeadOutput_continuous tokens row channel

/-- The free numerical Q/K/value parameter box is compact, with no sign assumption on its cap.
A negative cap can make it empty. Source: the bounded physical family following Appendix A.4. -/
theorem matchingHeadBox_compact (V H D : ℕ) (cap : ℝ) :
    IsCompact (matchingHeadBox V H D cap) := by
  have he : matchingHeadBox V H D cap =
      (((Set.Icc (-cap) cap).matrix : Set (Matrix (Fin V) (Fin H) ℝ)) ×ˢ
        (Set.Icc (-cap) cap).matrix) ×ˢ
        ((Set.Icc (-cap) cap).matrix : Set (Matrix (Fin V) (Fin D) ℝ)) := by
    ext h
    change ((∀ v d, -cap ≤ h.1.1 v d ∧ h.1.1 v d ≤ cap) ∧
      (∀ v d, -cap ≤ h.1.2 v d ∧ h.1.2 v d ≤ cap) ∧
      (∀ v d, -cap ≤ h.2 v d ∧ h.2 v d ≤ cap)) ↔ _
    constructor
    · rintro ⟨hq, hk, hv⟩
      exact ⟨⟨hq, hk⟩, hv⟩
    · rintro ⟨⟨hq, hk⟩, hv⟩
      exact ⟨hq, hk, hv⟩
  rw [he]
  have hc : IsCompact (Set.Icc (-cap) cap) := isCompact_Icc
  exact (hc.matrix.prod hc.matrix).prod hc.matrix

/-- Zero original tables supply a genuine feasible head whenever the numerical cap is nonnegative.
Source: the complete physical parameter family following Appendix A.4. -/
theorem matchingHeadBox_zero_mem (V H D : ℕ) (cap : ℝ) (hc : 0 ≤ cap) :
    (0 : MatchingHead V H D) ∈ matchingHeadBox V H D cap := by
  refine ⟨?_, ?_, ?_⟩ <;> intro v d <;> change -cap ≤ 0 ∧ 0 ≤ cap
  all_goals exact ⟨by linarith, hc⟩

/-- A positive-dimensional free matching domain inhabits the nonnegative-cap premise. -/
example : (0 : MatchingHead 2 1 1) ∈ matchingHeadBox 2 1 1 1 :=
  matchingHeadBox_zero_mem _ _ _ _ (by norm_num)

end Transformer.GPTMini.Sparsemax
