import Transformer.GPTMini.Sparsemax.AnchoredScores
import Transformer.GPTMini.Sparsemax.ValueSpan
import Mathlib.LinearAlgebra.Basis.Basic
import Mathlib.LinearAlgebra.StdBasis

/-!
# An enforceable active-value span

Derived modification of the linear value sum in `Attention.forward`
at commit `73f8a0b`, using arXiv:1602.02068v2, §2.2 and §2.5.
For a `d`-dimensional output, prepend `d + 1` anchor values: a common
trainable base and that base plus each fixed basis vector multiplied by
a positive learned exponential scale. Ordinary values remain arbitrary.

The bounded prefix scores keep all anchors active. Their differences
span the output for every finite parameter assignment, so the earlier
conditional span premise now follows from the architecture. This includes
arbitrary changes of the base, value scales, score parameters and ordinary
values. No optimizer-specific preservation assumption is needed.

This is a new anchored row architecture, not a theorem that the existing
query/key transformer already enforces the restriction. The result concerns
the ordinary task derivative at the frozen value readout; a zero output
derivative and global convergence remain outside its conclusion.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A trainable translated, positively scaled coordinate frame followed
by arbitrary ordinary values. Derived architecture for the value sum
at `73f8a0b`, motivated by arXiv:1602.02068v2, §2.5. -/
def anchoredValues {d N : ℕ} (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinary : Fin N → E) : Fin (d + 1 + N) → E :=
  Fin.addCases (Fin.cases base (fun k => base + Real.exp (scales k) • frame k)) ordinary

/-- Every basis direction is an anchor difference with nonzero scale.
Source context: the derived active-value restriction for §2.5 of
arXiv:1602.02068v2, with the value readout from `73f8a0b`. -/
theorem anchoredValues_difference {d N : ℕ} (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinary : Fin N → E) (k : Fin d) :
    anchoredValues frame base scales ordinary (Fin.castAdd N k.succ) -
      anchoredValues frame base scales ordinary (Fin.castAdd N 0) =
      Real.exp (scales k) • frame k := by
  simp [anchoredValues]

/-- The construction automatically satisfies the complete active-value
span, at every finite parameter assignment and every query after its
anchors. Source: derived from §2.2 and §2.5 of arXiv:1602.02068v2,
with the actual value aggregation at `73f8a0b`; the span is proved here,
rather than included as a hypothesis about learned values. -/
theorem anchoredValues_active_span {d N : ℕ} (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinaryValues : Fin N → E)
    (cap : ℝ) (parameters : Fin (d + 1) → ℝ) (ordinaryScores : Fin N → ℝ)
    (i : Fin (d + 1 + N)) (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i) :
    Submodule.span ℝ (activeValueDifferences (anchoredScores cap parameters ordinaryScores) i
      (anchoredValues frame base scales ordinaryValues)) = ⊤ := by
  let span := Submodule.span ℝ
    (activeValueDifferences (anchoredScores cap parameters ordinaryScores) i
      (anchoredValues frame base scales ordinaryValues))
  have hmem (k : Fin d) : frame k ∈ span := by
    have hp (a : Fin (d + 1)) :=
      anchoredScores_anchor_positive cap parameters ordinaryScores i hc hcap hvisible a
    have hd : Real.exp (scales k) • frame k ∈ span := by
      apply Submodule.subset_span
      refine ⟨Fin.castAdd N k.succ, Fin.castAdd N 0, hp k.succ, hp 0, ?_⟩
      exact (anchoredValues_difference frame base scales ordinaryValues k).symm
    have hs := span.smul_mem (Real.exp (scales k))⁻¹ hd
    simpa only [smul_smul, inv_mul_cancel₀ (ne_of_gt (Real.exp_pos _)), one_smul] using hs
  have hle : Submodule.span ℝ (Set.range frame) ≤ span := by
    apply Submodule.span_le.mpr
    rintro v ⟨k, rfl⟩
    exact hmem k
  rw [frame.span_eq] at hle
  exact le_antisymm le_top hle

/-- All span hypotheses are inhabited with arbitrary ordinary values
and a visible zero weight. Source context: the derived prefix construction
for arXiv:1602.02068v2, §2.2 and §2.5. -/
example : Submodule.span ℝ
    (activeValueDifferences
      (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2
      (anchoredValues (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0) (fun _ : Fin 1 => 7))) = ⊤ := by
  apply anchoredValues_active_span _ _ _ _ _ _ _ 2 (by norm_num) (by norm_num)
  exact fun a => Fin.le_last (Fin.castAdd 1 a)

/-- Some anchor difference distinguishes every nonzero output derivative.
Source: the derived frame restriction for §2.5 of arXiv:1602.02068v2,
composed with the frozen value sum at `73f8a0b`. The separating pair
uses two anchors, so ordinary value or score directions are unnecessary. -/
theorem anchoredValues_separates_gradient {d N : ℕ} (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinary : Fin N → E)
    (gradient : E →L[ℝ] ℝ) (hg : gradient ≠ 0) :
    ∃ k : Fin d, gradient
      (anchoredValues frame base scales ordinary (Fin.castAdd N k.succ) -
        anchoredValues frame base scales ordinary (Fin.castAdd N 0)) ≠ 0 := by
  classical
  by_contra hn
  have hkill (k : Fin d) : gradient (frame k) = 0 := by
    have hd : gradient (Real.exp (scales k) • frame k) = 0 := by
      by_contra he
      exact hn ⟨k, by simpa only [anchoredValues_difference] using he⟩
    simp only [map_smul, smul_eq_mul] at hd
    exact (mul_eq_zero.mp hd).resolve_left (ne_of_gt (Real.exp_pos _))
  apply hg
  apply ContinuousLinearMap.ext
  intro v
  change gradient v = (0 : ℝ)
  rw [← frame.sum_repr v]
  simp only [map_sum, map_smul, hkill, smul_zero, Finset.sum_const_zero]

/-- A nonzero ordinary output coordinate inhabits the separation premise.
Source context: arXiv:1602.02068v2, §2.5, the derived anchor frame. -/
example : ∃ k : Fin 1, (ContinuousLinearMap.proj 0 : (Fin 1 → ℝ) →L[ℝ] ℝ)
    (anchoredValues (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0) (fun _ : Fin 1 => 7)
      (Fin.castAdd 1 k.succ) -
    anchoredValues (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0) (fun _ : Fin 1 => 7)
      (Fin.castAdd 1 0)) ≠ 0 := by
  apply anchoredValues_separates_gradient (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0)
    (fun _ : Fin 1 => 7) (ContinuousLinearMap.proj 0)
  intro h
  have he := congrArg (fun f : (Fin 1 → ℝ) →L[ℝ] ℝ => f (fun _ => 1)) h
  norm_num at he

/-- Nonzero ordinary output derivatives cannot disappear in the score
path of this architecture, for arbitrary finite trainable assignments.
Source: the derived enforcement of §2.5's active-value condition for
arXiv:1602.02068v2, with `Attention.forward`'s readout at `73f8a0b`.
The span condition is a conclusion of the construction, not a premise. -/
theorem anchoredTaskLoss_no_zero_score_derivative {d N : ℕ}
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinaryScores : Fin N → ℝ) (i : Fin (d + 1 + N))
    (loss : E → ℝ) (gradient : E →L[ℝ] ℝ) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i) (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (anchoredScores cap parameters ordinaryScores) i))) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun z => loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights z i))) 0 (anchoredScores cap parameters ordinaryScores) :=
  taskLoss_no_zero_score_derivative_of_active_value_span _ i _ loss gradient
    (anchoredValues_active_span frame base scales ordinaryValues cap parameters ordinaryScores i
      hc hcap hvisible) hg hl

/-- The complete task-loss premises hold for an ordinary coordinate
readout with two anchors and one inactive ordinary position. Source context:
arXiv:1602.02068v2, §2.5, the derived enforced-span architecture. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun z : Fin 3 → ℝ => (ContinuousLinearMap.proj 0 : (Fin 1 → ℝ) →L[ℝ] ℝ)
      (frozenValueReadout
        (anchoredValues (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
        (sparseWeights z 2))) 0
      (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) := by
  apply anchoredTaskLoss_no_zero_score_derivative (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0)
    (fun _ : Fin 1 => 7) (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2
    (fun x : Fin 1 → ℝ => x 0) (ContinuousLinearMap.proj 0) (by norm_num) (by norm_num)
    (fun a => Fin.le_last (Fin.castAdd 1 a))
  · intro h
    have he := congrArg (fun f : (Fin 1 → ℝ) →L[ℝ] ℝ => f (fun _ => 1)) h
    norm_num at he
  · exact (ContinuousLinearMap.proj 0 : (Fin 1 → ℝ) →L[ℝ] ℝ).hasFDerivAt

end Transformer.GPTMini.Sparsemax
