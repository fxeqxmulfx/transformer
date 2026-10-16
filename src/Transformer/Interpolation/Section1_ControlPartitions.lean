/-
# Counting original control pieces under joining and time changes

ArXiv:2411.04551v3, §1, `eq: vf`, and Appendix,
the switch bound and time rescaling in `lem: tubular.mass.movement`.

Appending a constant phase appends one terminal point to the actual
finite grid of switching times. Its weak monotonicity is retained, and
the joined original controls are constant on every required interval.
Equal grid endpoints are allowed, as in the source's bound on the number
of pieces rather than a demand for exactly that many nonempty phases.

Joining the original neural matrices componentwise gives exactly the
joined original parameter curve, with the second phase's elapsed clock.
A positive change of time divides the switching grid by the time factor
and preserves its number of pieces, including for the actual output
rescaling of both original matrices `V,W`.
-/

import Transformer.Interpolation.Section1_WeakConcatenation
import Transformer.Interpolation.Section1_TimeRescaling
import Transformer.Interpolation.Section4_NeuralSpeed
import Mathlib.Order.Fin.Basic

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Appending an endpoint after the previous last time preserves the
switching grid's monotonicity. Auxiliary for arXiv:2411.04551v3,
Appendix, the switch bound in `lem: tubular.mass.movement`. -/
theorem monotone_fin_snoc_real {K : ℕ} (t : Fin (K + 1) → ℝ) (T : ℝ)
    (ht : Monotone t) (hT : t (Fin.last K) ≤ T) : Monotone (Fin.snoc t T) := by
  apply Fin.monotone_iff_le_succ.mpr
  refine Fin.lastCases ?_ (fun i => ?_)
  · simpa only [Fin.snoc_castSucc, Fin.succ_last, Fin.snoc_last] using hT
  · rw [← Fin.castSucc_succ, Fin.snoc_castSucc, Fin.snoc_castSucc]
    exact ht (show i.castSucc ≤ i.succ by change i.val ≤ i.val + 1; omega)

/-- One appended actual constant phase adds at most one control piece.
Source: arXiv:2411.04551v3, Appendix, the switch accounting in
`lem: tubular.mass.movement`. -/
theorem piecewiseConstant_concatenate_constant (θ : TimeParams d) (θ₂ : Params d)
    (S T : ℝ) (K : ℕ) (hST : S ≤ T) (hθ : PiecewiseConstant d θ S K) :
    PiecewiseConstant d (concatenate S θ (fun _ => θ₂)) T (K + 1) := by
  obtain ⟨t, hmono, hzero, hlast, hconst⟩ := hθ
  have hbound (i : Fin (K + 1)) : t i ≤ S := by
    calc t i ≤ t (Fin.last K) := hmono (Fin.le_last i)
         _ = S := hlast
  refine ⟨Fin.snoc t T, monotone_fin_snoc_real t T hmono (by rw [hlast]; exact hST), ?_,
    by simp only [Fin.snoc_last], ?_⟩
  · have hz : (0 : Fin (K + 2)) = (0 : Fin (K + 1)).castSucc := rfl
    rw [hz, Fin.snoc_castSucc, hzero]
  · refine Fin.lastCases ?_ (fun i => ?_)
    · intro s s' hs hs'
      rw [Fin.snoc_castSucc, Fin.succ_last, Fin.snoc_last, hlast] at hs hs'
      simp only [concatenate, ite_eq_right (not_lt_of_ge hs.1), ite_eq_right (not_lt_of_ge hs'.1)]
    · intro s s' hs hs'
      rw [← Fin.castSucc_succ, Fin.snoc_castSucc, Fin.snoc_castSucc] at hs hs'
      have hsS : s < S := hs.2.trans_le (hbound i.succ)
      have hsS' : s' < S := hs'.2.trans_le (hbound i.succ)
      simp only [concatenate, ite_eq_left hsS, ite_eq_left hsS']
      exact hconst i s s' hs hs'

/-- One original constant control uses one piece on a nonnegative horizon.
Auxiliary for arXiv:2411.04551v3, Appendix, the constant phases of
`lem: tubular.mass.movement`. -/
theorem piecewiseConstant_constant (θ : Params d) (T : ℝ) (hT : 0 ≤ T) :
    PiecewiseConstant d (fun _ => θ) T 1 := by
  refine ⟨![0, T], Fin.monotone_iff_le_succ.mpr ?_, by simp, by simp,
    fun _ _ _ _ _ => rfl⟩
  intro i
  fin_cases i
  exact hT

/-- The componentwise neural joining is exactly the original parameter
joining, with both attention matrices zero. Source: arXiv:2411.04551v3,
§1, `eq: vf`, and Appendix, `eq: neural.pde.sphere`. -/
theorem neuralParams_concatenate (S : ℝ) (W₁ U₁ W₂ U₂ : ℝ → ParamMatrix d)
    (b₁ b₂ : ℝ → EucSpace d) :
    neuralParams d (concatenate S W₁ W₂) (concatenate S U₁ U₂) (concatenate S b₁ b₂) =
      concatenate S (neuralParams d W₁ U₁ b₁) (neuralParams d W₂ U₂ b₂) := by
  funext t
  by_cases ht : t < S <;> simp only [neuralParams, concatenate, ht, ite_true, ite_false]

/-- Appending original constant neural matrices retains precisely the
source's piece bound. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`. -/
theorem piecewiseConstant_neural_concatenate_constant
    (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) (W₂ U₂ : ParamMatrix d) (b₂ : EucSpace d)
    (S T : ℝ) (K : ℕ) (hST : S ≤ T) (hθ : PiecewiseConstant d (neuralParams d W U b) S K) :
    PiecewiseConstant d (neuralParams d (concatenate S W (fun _ => W₂))
      (concatenate S U (fun _ => U₂)) (concatenate S b (fun _ => b₂))) T (K + 1) := by
  rw [neuralParams_concatenate]
  exact piecewiseConstant_concatenate_constant (neuralParams d W U b)
    { V := 0, B := 0, W := W₂, U := U₂, b := b₂ } S T K hST hθ

/-- A fixed map of original parameter values and a positive clock factor
preserve the number of pieces on the rescaled horizon. Auxiliary for
arXiv:2411.04551v3, Appendix, time rescaling in `lem: two.balls`. -/
theorem piecewiseConstant_comp_pos (θ : TimeParams d) (F : Params d → Params d)
    (T : ℝ) (K : ℕ) (κ : ℝ) (hκ : 0 < κ) (hθ : PiecewiseConstant d θ T K) :
    PiecewiseConstant d (fun s => F (θ (κ * s))) (T / κ) K := by
  obtain ⟨t, hmono, hzero, hlast, hconst⟩ := hθ
  refine ⟨fun i => t i / κ,
    fun a b hab => (div_le_div_iff_of_pos_right hκ).mpr (hmono hab), ?_, ?_, ?_⟩
  · change t 0 / κ = 0
    rw [hzero, zero_div]
  · change t (Fin.last K) / κ = T / κ
    rw [hlast]
  · intro k s s' hs hs'
    have hIn (r : ℝ) (hr : r ∈ Ico (t k.castSucc / κ) (t k.succ / κ)) :
        κ * r ∈ Ico (t k.castSucc) (t k.succ) := by
      constructor
      · have hh := (div_le_iff₀ hκ).mp hr.1
        simpa only [mul_comm] using hh
      · have hh := (lt_div_iff₀ hκ).mp hr.2
        simpa only [mul_comm] using hh
    exact congrArg F (hconst k (κ * s) (κ * s') (hIn s hs) (hIn s' hs'))

/-- The exact original time rescaling preserves the number of control
pieces. Source: arXiv:2411.04551v3, §1, `eq: vf`, and Appendix,
the prescribed-time construction in `lem: two.balls`. -/
theorem piecewiseConstant_rescaleTimeParams (θ : TimeParams d) (T : ℝ) (K : ℕ)
    (κ : ℝ) (hκ : 0 < κ) (hθ : PiecewiseConstant d θ T K) :
    PiecewiseConstant d (rescaleTimeParams κ θ) (T / κ) K :=
  piecewiseConstant_comp_pos θ (fun a =>
    { V := κ • a.V, B := a.B, W := κ • a.W, U := a.U, b := a.b }) T K κ hκ hθ

/-- Positive actual horizons, a nonempty two-point grid, a positive clock
factor and original constant control curves jointly satisfy every grid,
joining and time-rescaling hypothesis. -/
example (θ : Params 2) :
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧ (0 : ℝ) < 2 ∧
    Monotone (![0, 1] : Fin 2 → ℝ) ∧ (![0, 1] : Fin 2 → ℝ) (Fin.last 1) ≤ 2 ∧
    PiecewiseConstant 2 (fun _ => θ) 1 1 ∧
    PiecewiseConstant 2 (neuralParams 2 (fun _ => θ.W) (fun _ => θ.U) (fun _ => θ.b)) 1 1 := by
  refine ⟨zero_le_one, by norm_num, by norm_num, ?_, by norm_num,
    piecewiseConstant_constant θ 1 zero_le_one, ?_⟩
  · apply Fin.monotone_iff_le_succ.mpr
    intro i
    fin_cases i
    norm_num
  · exact piecewiseConstant_constant
      { V := 0, B := 0, W := θ.W, U := θ.U, b := θ.b } 1 zero_le_one

end Transformer.Interpolation
