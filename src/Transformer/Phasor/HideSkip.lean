module

public import Transformer.Phasor.PhaseHiding

/-!
# A skip map and phases of main that hide equal streams

T3d of `review/residual_connections.md`, *What the extra streams can hold*. The skip logits
`hideLogits` are `ln 22` between stream `s` and its mirror `3 − s` and `ln 3` elsewhere. Their
exponentials have every row and column sum `31`, so the twenty-cycle Sinkhorn map is
`hideSkip = (19 Π + 3 J)/31` (`finiteSinkhorn_hideLogits`), with `Π` the reversal of the streams and
`J` all ones: positive, symmetric and doubly stochastic. The phase parameters `√15, −√15, −√3`
(`hidePhase`) give, through main's `φ_s = arctan b_s`, the angles `0, θ, −θ, −π/3` with
`cos θ = 1/4` (`hideAngle`, `cos_hideAngle`, `sin_hideAngle`). With `c` and `s` the cosines and
sines of the angles:

* `sum_phaseCos_hide`: `∑_s C_s = cᵀ P c + sᵀ P s = 267/248 − 267/248 = 0`, so the skip hides four
  equal streams at every even width (`sum_phaseMix_hide`);
* `phaseCos_hide_zero`, `phaseSin_hide_zero`: `a_0 = 1/2 + i 11√3/31`, so stream `0` keeps them
  (`phaseMix_hide_zero_ne`).

`HideRouter` puts them in a router. Exact real arithmetic.
-/

@[expose] public section

open Finset

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual

/-- The matrix `19 Π + 3 J`: `22` between stream `s` and its mirror `3 − s`, `3` elsewhere. -/
noncomputable def hideCounts : Matrix (Fin 4) (Fin 4) ℝ :=
  !![3, 3, 3, 22; 3, 3, 22, 3; 3, 22, 3, 3; 22, 3, 3, 3]

/-- The skip logits `ln 22` between a stream and its mirror, `ln 3` elsewhere. -/
noncomputable def hideLogits : Matrix (Fin 4) (Fin 4) ℝ :=
  Matrix.of fun s t => Real.log (hideCounts s t)

/-- Their twenty-cycle map `(19 Π + 3 J)/31`. -/
noncomputable def hideSkip : Matrix (Fin 4) (Fin 4) ℝ :=
  Matrix.of fun s t => hideCounts s t / 31

/-- The entries of `19 Π + 3 J` are positive. -/
theorem hideCounts_pos (s t : Fin 4) : 0 < hideCounts s t := by
  fin_cases s <;> fin_cases t <;> norm_num [hideCounts]

/-- `19 Π + 3 J` is symmetric. -/
theorem hideCounts_symm (s t : Fin 4) : hideCounts s t = hideCounts t s := by
  fin_cases s <;> fin_cases t <;> rfl

/-- Its rows sum to `31`. -/
theorem sum_hideCounts_row (s : Fin 4) : ∑ t, hideCounts s t = 31 := by
  fin_cases s <;> norm_num [hideCounts, Fin.sum_univ_four]

/-- And so do its columns. -/
theorem sum_hideCounts_col (t : Fin 4) : ∑ s, hideCounts s t = 31 := by
  rw [← sum_hideCounts_row t]
  exact Finset.sum_congr rfl fun s _ => hideCounts_symm s t

/-- `hideSkip` is symmetric. -/
theorem hideSkip_symm (s t : Fin 4) : hideSkip s t = hideSkip t s := by
  simp only [hideSkip, Matrix.of_apply, hideCounts_symm s t]

/-- **The twenty-cycle map of `hideLogits` is `(19 Π + 3 J)/31`**: the logits are balanced. -/
theorem finiteSinkhorn_hideLogits : finiteSinkhorn hideLogits 20 = hideSkip := by
  have he : ∀ s t, Real.exp (hideLogits s t) = hideCounts s t := fun s t =>
    Real.exp_log (hideCounts_pos s t)
  rw [finiteSinkhorn_balanced (c := 31) (by norm_num) (fun s => by simp only [he,
    sum_hideCounts_row]) (fun t => by simp only [he, sum_hideCounts_col])]
  ext s t
  simp only [Matrix.of_apply, he, hideSkip]

/-- For a symmetric skip the imaginary parts sum to zero at any angles. -/
example (φ : Fin 4 → ℝ) := sum_phaseSin_eq_zero hideSkip_symm φ

/-- The phase parameters `√15, −√15, −√3` of streams `1, 2, 3`. -/
noncomputable def hidePhase : Fin 3 → ℝ := ![Real.sqrt 15, -Real.sqrt 15, -Real.sqrt 3]

/-- The angles `0, θ, −θ, −π/3` of the four streams, `θ = arctan √15`. -/
noncomputable def hideAngle : Fin 4 → ℝ :=
  ![0, Real.arctan (Real.sqrt 15), Real.arctan (-Real.sqrt 15), Real.arctan (-Real.sqrt 3)]

/-- `cos (arctan √a) = 1/b` when `1 + a = b²`. -/
theorem cos_arctan_sqrt {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (h : 1 + a = b ^ 2) :
    Real.cos (Real.arctan (Real.sqrt a)) = 1 / b := by
  rw [Real.cos_arctan, Real.sq_sqrt ha, h, Real.sqrt_sq hb]

/-- `sin (arctan √a) = √a/b` when `1 + a = b²`. -/
theorem sin_arctan_sqrt {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (h : 1 + a = b ^ 2) :
    Real.sin (Real.arctan (Real.sqrt a)) = Real.sqrt a / b := by
  rw [Real.sin_arctan, Real.sq_sqrt ha, h, Real.sqrt_sq hb]

/-- `θ = arctan √15` has `cos θ = 1/4`. -/
example : Real.cos (Real.arctan (Real.sqrt 15)) = 1 / 4 :=
  cos_arctan_sqrt (by norm_num) (by norm_num) (by norm_num)

/-- And `sin θ = √15/4`. -/
example : Real.sin (Real.arctan (Real.sqrt 15)) = Real.sqrt 15 / 4 :=
  sin_arctan_sqrt (by norm_num) (by norm_num) (by norm_num)

/-- The cosines `1, 1/4, 1/4, 1/2` of the angles. -/
theorem cos_hideAngle (s : Fin 4) : Real.cos (hideAngle s) = ![1, 1 / 4, 1 / 4, 1 / 2] s := by
  have h15 := cos_arctan_sqrt (a := 15) (b := 4) (by norm_num) (by norm_num) (by norm_num)
  have h3 := cos_arctan_sqrt (a := 3) (b := 2) (by norm_num) (by norm_num) (by norm_num)
  fin_cases s <;> simp only [hideAngle, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Fin.isValue,
    Matrix.cons_val, Real.cos_zero, Real.arctan_neg, Real.cos_neg, h15, h3]

/-- The sines `0, √15/4, −√15/4, −√3/2` of the angles. -/
theorem sin_hideAngle (s : Fin 4) : Real.sin (hideAngle s) =
    ![0, Real.sqrt 15 / 4, -(Real.sqrt 15 / 4), -(Real.sqrt 3 / 2)] s := by
  have h15 := sin_arctan_sqrt (a := 15) (b := 4) (by norm_num) (by norm_num) (by norm_num)
  have h3 := sin_arctan_sqrt (a := 3) (b := 2) (by norm_num) (by norm_num) (by norm_num)
  fin_cases s <;> simp only [hideAngle, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Fin.isValue,
    Matrix.cons_val, Real.sin_zero, Real.arctan_neg, Real.sin_neg, h15, h3]

/-- **The real parts sum to zero**: `cᵀ P c + sᵀ P s = 0`. -/
theorem sum_phaseCos_hide : ∑ s, phaseCos hideSkip hideAngle s = 0 := by
  have hu := Real.sq_sqrt (show (0 : ℝ) ≤ 15 by norm_num)
  have hw := Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num)
  simp only [phaseCos, Real.cos_sub, cos_hideAngle, sin_hideAngle, hideSkip, hideCounts,
    Matrix.of_apply, Fin.sum_univ_four, Matrix.cons_val]
  linear_combination (3 / 124) * hw - (19 / 248) * hu

/-- The real part of `a_0` is `1/2`. -/
theorem phaseCos_hide_zero : phaseCos hideSkip hideAngle 0 = 1 / 2 := by
  simp only [phaseCos, Real.cos_sub, cos_hideAngle, sin_hideAngle, hideSkip, hideCounts,
    Matrix.of_apply, Fin.sum_univ_four, Matrix.cons_val]
  ring

/-- The imaginary part of `a_0` is `11√3/31`. -/
theorem phaseSin_hide_zero : phaseSin hideSkip hideAngle 0 = 11 * Real.sqrt 3 / 31 := by
  simp only [phaseSin, Real.sin_sub, cos_hideAngle, sin_hideAngle, hideSkip, hideCounts,
    Matrix.of_apply, Fin.sum_univ_four, Matrix.cons_val]
  ring

variable {D : ℕ}

/-- **The skip hides four equal streams**: after it they sum to zero, at every even width. -/
theorem sum_phaseMix_hide (hD : Even D) (h : EucSpace D) :
    ∑ s, phaseMix (headPhaseAction D) hideSkip hideAngle (fun _ => h) s = 0 :=
  sum_phaseMix_const_eq_zero hD hideSkip_symm sum_phaseCos_hide h

/-- At main's width. -/
example (h : EucSpace 128) := sum_phaseMix_hide ⟨64, rfl⟩ h

/-- **Stream `0` keeps them**: `a_0 h ≠ 0` for `h ≠ 0`, at every even width. -/
theorem phaseMix_hide_zero_ne (hD : Even D) {h : EucSpace D} (hh : h ≠ 0) :
    phaseMix (headPhaseAction D) hideSkip hideAngle (fun _ => h) 0 ≠ 0 :=
  phaseMix_const_ne_zero hD (by rw [phaseCos_hide_zero]; norm_num) hh

/-- The unit vector `e₀` at main's width is a nonzero stream. -/
theorem single_zero_ne_zero : (EuclideanSpace.single 0 1 : EucSpace 128) ≠ 0 := fun h0 => by
  simpa using congrArg (fun v : EucSpace 128 => v 0) h0

/-- At main's width, on `e₀`, through the general statement. -/
example : phaseMix (headPhaseAction 128) hideSkip hideAngle
    (fun _ => EuclideanSpace.single 0 1) 0 ≠ 0 :=
  phaseMix_const_ne_zero ⟨64, rfl⟩ (by rw [phaseCos_hide_zero]; norm_num) single_zero_ne_zero

/-- And through this one. -/
example : phaseMix (headPhaseAction 128) hideSkip hideAngle
    (fun _ => EuclideanSpace.single 0 1) 0 ≠ 0 :=
  phaseMix_hide_zero_ne ⟨64, rfl⟩ single_zero_ne_zero

end Transformer.Phasor
