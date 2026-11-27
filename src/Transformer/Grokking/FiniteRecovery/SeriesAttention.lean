module

public import Transformer.Grokking.FiniteRecovery.Argand
public import Transformer.Grokking.FiniteRecovery.SeriesCodec
public import Transformer.RoPE.ReRoPE2D.Defs

/-! Axial LeRoPE for a joint context of aligned series. The row coordinate
is supplied by the problem, and the time coordinate controls causality.
This uses PhasorV2's learned frequency grid and the axial band assignment
of Transformer.RoPE.ReRoPE2D. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped ComplexConjugate

variable {R A : Type*} {T H d layers : ℕ}

/-- Each complex band turns with the assigned coordinate of the position. -/
noncomputable def axialTurn (θ : Fin d → ℝ) (axis : Fin d → Fin 2)
    (position : Fin 2 → ℝ) (q : Fin d → ℂ) : Fin d → ℂ :=
  fun m => turn θ (position (axis m)) q m

/-- The unscaled complex query-key product uses only the relative
position, with the row and time offsets on their respective bands. -/
theorem axialTurn_relative (θ : Fin d → ℝ) (axis : Fin d → Fin 2)
    (x y : Fin 2 → ℝ) (q k : Fin d → ℂ) :
    (∑ m, conj (axialTurn θ axis x q m) * axialTurn θ axis y k m) =
      ∑ m, conj (q m) * axialTurn θ axis (y - x) k m := by
  refine Finset.sum_congr rfl fun m _ => ?_
  simp only [axialTurn, turn, Pi.sub_apply]
  have h : Complex.exp (((y (axis m) - x (axis m)) * θ m : ℝ) * Complex.I) =
      conj (Complex.exp ((x (axis m) * θ m : ℝ) * Complex.I)) *
        Complex.exp ((y (axis m) * θ m : ℝ) * Complex.I) := by
    rw [← Complex.exp_conj, map_mul, Complex.conj_ofReal, Complex.conj_I, ← Complex.exp_add]
    congr 1
    push_cast
    ring
  rw [map_mul, h]
  ring

/-- Grid keys are visible only at known sites at or before the query's time.
Row coordinates can describe meaningful sensor positions or feature order. -/
noncomputable def gridGeometry (row : R → ℝ) (axis : Fin d → Fin 2) (alpha : Fin d → ℝ)
    (known : R × Fin T → Bool) : AttentionGeometry (R × Fin T) d where
  rotate p := axialTurn (leropeFrequency 10000 alpha) axis ![row p.1, (p.2.val : ℝ)]
  visible i j := known j && decide (j.2 ≤ i.2)
  time p := p.2.val

/-- The same mask excludes future and unavailable keys from both softmaxes. -/
theorem maskedSoftmax_grid_zero [Fintype R] (row : R → ℝ) (axis : Fin d → Fin 2)
    (alpha : Fin d → ℝ) (known : R × Fin T → Bool) (logits : R × Fin T → ℝ)
    (query key : R × Fin T) (h : known key = false ∨ query.2 < key.2) :
    maskedSoftmax ((gridGeometry row axis alpha known).visible query) logits key = 0 := by
  apply maskedSoftmax_eq_zero
  rcases h with h | h
  · simp [gridGeometry, h]
  · simp [gridGeometry, not_le.mpr h]

/-- The complete skip/write stack with axial attention, using the tied
readout at a selected grid position. -/
noncomputable def Model.gridLogits [Fintype R] (model : Model A H d layers) (repeats : ℕ)
    (row : R → ℝ) (axis : Fin d → Fin 2) (known : R × Fin T → Bool)
    (tokens : R × Fin T → A) (query : R × Fin T) : A → ℝ :=
  model.logitsWith repeats (gridGeometry row axis model.alpha known) tokens query

example (logits : Fin 2 × Fin 8 → ℝ) :
    maskedSoftmax ((gridGeometry (fun r : Fin 2 => r.val)
      (Transformer.RoPE.ReRoPE2D.glmAxis (p := 16)) 0 (fun _ => true)).visible (0, 2))
      logits (1, 3) = 0 :=
  maskedSoftmax_grid_zero _ _ _ _ _ _ _ (Or.inr (by decide))

noncomputable example (model : Model (Fin 256) 4 16 2) (grid : Fin 3 → FirstDifferenceGrid)
    (past : Fin 3 → Fin 9 → ℝ) : Fin 256 → ℝ :=
  model.gridLogits 2 (fun r : Fin 3 => r.val) Transformer.RoPE.ReRoPE2D.glmAxis
    (fun _ => true) (FirstDifferenceGrid.jointContext (vocab := 256) (by decide) grid past) (0, 7)

end Transformer.Grokking.FiniteRecovery.Argand
